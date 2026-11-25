"""硬件解码 surface 池与送显队列。

修复点（与题目一一对应）：

1. surface 按引用计数持有 —— 解码端与显示端各自 ``acquire`` / ``release``，
   引用计数归零之前任何人都不得覆写或回收 surface。
2. 池扩容统一登记 —— 池耗尽而新建的纹理必须经过 ``pool_register`` 进入登记表，
   杜绝“新建纹理游离在池外、回收时遗漏”的泄漏。
3. 按 PTS 重排输出队列 —— B 帧可能先于参考帧完成解码，输出队列按
   ``(PTS, 解码序号)`` 稳定排序，保证送显顺序与时间戳顺序一致。
4. 色彩参数以码流为准 —— 仅当码流显式标记 ``unspecified`` 时才回退到容器
   元数据，再不行才使用默认值，修复按容器元数据设色导致的色偏。
5. seek 时作废在途解码 —— 池维护 generation 纪元号，seek 递增纪元，旧纪元
   提交的解码结果即使晚到也一律丢弃，不得覆盖新位置首帧。
6. 池回收延迟到显示完成 —— present 时显示端持有引用，fence 通知显示完成后
   才释放最后一个引用并把 surface 放回空闲链表。

模块只依赖标准库，单线程确定性模型，fence 由 ``delayed_recycle`` 模拟触发。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional

__all__ = [
    "surface_ref",
    "pool_register",
    "pts_reorder",
    "color_params",
    "seek_invalidate",
    "delayed_recycle",
    "determinism",
    "validation",
    "audit",
    "fallback",
    "decode_queue",
    "metrics",
    "present",
    "surface_alloc",
    "flush",
    "Surface",
    "SurfaceState",
    "SurfacePool",
    "Frame",
    "ReorderBuffer",
    "ColorParams",
    "PresentToken",
    "DecodeQueue",
]

# 色彩字段“码流未指定”的全部合法取值（大小写不敏感比较）。
_UNSPECIFIED = {None, "", "unspecified", "unknown", "reserved"}

_DEFAULT_COLOR = {
    "primaries": "bt709",
    "transfer": "bt709",
    "matrix": "bt709",
    "range": "limited",
    "chroma_location": "left",
}
_COLOR_FIELDS = tuple(_DEFAULT_COLOR.keys())


class SurfaceState(Enum):
    """surface 生命周期状态。"""

    FREE = "free"            # 在空闲链表中，可被分配
    DECODING = "decoding"    # 已交给硬件解码器，写入中
    READY = "ready"          # 解码完成、在 PTS 重排队列中等待送显
    DISPLAYING = "displaying"  # 已送显，显示端仍在读，禁止覆写


@dataclass
class Surface:
    """一块可被解码端写入、显示端读取的纹理 surface。"""

    id: Any
    texture: Any = None
    state: SurfaceState = SurfaceState.FREE
    refcount: int = 0
    generation: int = 0
    pts: Optional[int] = None
    seq: Optional[int] = None
    is_fallback: bool = False
    owners: set = field(default_factory=set)

    def acquire(self, owner: str) -> None:
        self.refcount += 1
        self.owners.add(owner)

    def release(self, owner: str) -> None:
        if self.refcount <= 0:
            raise ValueError(f"surface {self.id!r} 引用计数已为零，不能再次释放")
        if owner not in self.owners:
            raise ValueError(f"owner {owner!r} 未持有 surface {self.id!r}")
        self.refcount -= 1
        self.owners.discard(owner)


@dataclass
class Frame:
    """一次解码完成的输出帧，挂在 PTS 重排队列里。"""

    surface: Surface
    pts: int
    seq: int
    generation: int
    color: "ColorParams"

    def sort_key(self):
        # PTS 相同则按解码序号、再按 surface id 兜底，保证全序且确定。
        return (self.pts, self.seq, _id_key(self.surface.id))


@dataclass
class ColorParams:
    """合并后的色彩参数，sources 记录每个字段最终取自哪一层。"""

    primaries: str
    transfer: str
    matrix: str
    range: str
    chroma_location: str
    sources: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in _COLOR_FIELDS}


@dataclass
class PresentToken:
    """一次送显的 fence 凭证；显示完成后置 signaled，方可触发延迟回收。"""

    surface: Surface
    generation: int
    signaled: bool = False

    def complete(self) -> None:
        self.signaled = True


class DecodeQueue:
    """解码提交队列，每个工作项打上提交时的 generation 纪元号。"""

    def __init__(self) -> None:
        self._pending: list = []
        self._seq = 0

    def submit(self, work: Any, generation: int) -> int:
        self._seq += 1
        self._pending.append({"work": work, "generation": generation, "seq": self._seq})
        return self._seq

    def pop(self) -> Optional[dict]:
        if not self._pending:
            return None
        return self._pending.pop(0)

    def clear(self) -> int:
        n = len(self._pending)
        self._pending.clear()
        return n

    def __len__(self) -> int:
        return len(self._pending)


class ReorderBuffer:
    """PTS 重排缓冲：乱序进入、按 (PTS, seq) 有序流出。"""

    def __init__(self, capacity: int = 32) -> None:
        if capacity <= 0:
            raise ValueError("重排缓冲容量必须为正整数")
        self.capacity = capacity
        self._frames: list[Frame] = []

    def push(self, frame: Frame) -> None:
        if len(self._frames) >= self.capacity:
            raise OverflowError("PTS 重排缓冲已满，请先 pop/drain")
        self._frames.append(frame)

    def pop(self) -> Optional[Frame]:
        if not self._frames:
            return None
        self._frames.sort(key=Frame.sort_key)
        return self._frames.pop(0)

    def drain(self) -> list[Frame]:
        ordered = sorted(self._frames, key=Frame.sort_key)
        self._frames.clear()
        return ordered

    def clear(self) -> int:
        n = len(self._frames)
        self._frames.clear()
        return n

    def __len__(self) -> int:
        return len(self._frames)


def _id_key(value: Any):
    """可比较的 id 键：不同类型也不要在排序时抛 TypeError。"""
    return (type(value).__name__, str(value))


class SurfacePool:
    """surface 登记表 + 空闲链表 + 纪元/重排/送显一体化状态机。"""

    FALLBACK_ID = "__fallback_scratch__"

    def __init__(self, capacity: int = 8) -> None:
        if capacity <= 0:
            raise ValueError("surface 池容量必须为正整数")
        self.capacity = capacity
        self.registry: dict[Any, Surface] = {}
        self.free_list: list[Any] = []
        self.generation = 0
        self.reorder = ReorderBuffer()
        self.queue = DecodeQueue()
        self.fences: list[PresentToken] = []
        self._auto_id = 0
        self._scratch: Optional[Surface] = None
        self.metrics = {
            "allocated": 0,
            "registered": 0,
            "expansion_count": 0,
            "fallback_hits": 0,
            "acquires": 0,
            "releases": 0,
            "recycled": 0,
            "presented": 0,
            "stale_dropped": 0,
            "reordered": 0,
            "seeks": 0,
            "flushes": 0,
        }

    # ------------------------------------------------------------------ #
    # 修复 2：池扩容统一登记
    # ------------------------------------------------------------------ #
    def next_surface_id(self) -> int:
        # 跳过外部显式登记占用的 id，避免自动分配与显式 id 撞号。
        self._auto_id += 1
        while self._auto_id in self.registry:
            self._auto_id += 1
        return self._auto_id

    def register(self, surface: Surface) -> Surface:
        """把纹理登记进池；扩容路径与外部新建路径都必须走这里。"""
        if not isinstance(surface, Surface):
            raise TypeError("register 只接受 Surface 实例")
        existing = self.registry.get(surface.id)
        if existing is not None:
            if existing is surface:
                return surface  # 幂等：重复登记同一对象直接忽略
            raise ValueError(f"surface id {surface.id!r} 已被另一块纹理占用")
        self.registry[surface.id] = surface
        surface.generation = self.generation
        if surface.state is SurfaceState.FREE and surface.id not in self.free_list:
            self.free_list.append(surface.id)
        self.metrics["registered"] += 1
        return surface

    def _new_surface(self, sid: Any = None) -> Surface:
        surface = Surface(id=sid if sid is not None else self.next_surface_id())
        # 统一登记，保证扩容出来的纹理同样可被回收。
        self.register(surface)
        return surface

    # ------------------------------------------------------------------ #
    # 修复 1：引用计数持有的分配路径
    # ------------------------------------------------------------------ #
    def alloc(self) -> Surface:
        """取一块空闲 surface；池耗尽时扩容并统一登记，顶到上限走兜底。"""
        surface = None
        while self.free_list:
            sid = self.free_list[0]
            candidate = self.registry[sid]
            if candidate.state is SurfaceState.FREE and candidate.refcount == 0:
                self.free_list.pop(0)  # 只有真正取走时才出链表
                surface = candidate
                break
            # 脏项（理论上不应存在）：移出链表并继续找，不静默吞掉
            self.free_list.pop(0)
        if surface is None:
            if len(self.registry) < self.capacity:
                surface = self._new_surface()
                self.free_list.remove(surface.id)  # 新建后立即投入使用
                self.metrics["expansion_count"] += 1
            else:
                return self._fallback("pool-exhausted")

        surface.state = SurfaceState.DECODING
        surface.generation = self.generation
        surface.pts = None
        surface.seq = None
        surface.acquire("decoder")  # 解码端持有，显示端读完前不得覆写
        self.metrics["allocated"] += 1
        self.metrics["acquires"] += 1
        return surface

    def acquire(self, surface: Surface, owner: str) -> int:
        self._require_registered(surface)
        # FREE surface 一旦被持有，必须离开空闲链表并转入在用状态，
        # 否则会被分配路径二次分配，形成“显示端还在读、解码端已覆写”。
        if surface.state is SurfaceState.FREE and not surface.is_fallback:
            surface.state = SurfaceState.READY
            if surface.id in self.free_list:
                self.free_list.remove(surface.id)
        surface.acquire(owner)
        self.metrics["acquires"] += 1
        return surface.refcount

    def release(self, surface: Surface, owner: str) -> int:
        self._require_registered(surface)
        surface.release(owner)
        self.metrics["releases"] += 1
        if surface.refcount == 0:
            self._recycle(surface)
        return surface.refcount

    def _recycle(self, surface: Surface) -> None:
        surface.state = SurfaceState.FREE
        surface.pts = None
        surface.seq = None
        surface.owners.clear()
        if not surface.is_fallback and surface.id not in self.free_list:
            self.free_list.append(surface.id)
        self.metrics["recycled"] += 1

    def _require_registered(self, surface: Surface) -> None:
        if self.registry.get(surface.id) is not surface:
            raise ValueError(f"surface {surface.id!r} 未登记在本池中")

    # ------------------------------------------------------------------ #
    # 修复 5：seek 纪元 + 在途解码作废
    # ------------------------------------------------------------------ #
    def submit_decode(self, work: Any = None) -> int:
        return self.queue.submit(work, self.generation)

    def decode_complete(
        self,
        surface: Surface,
        pts: int,
        seq: int,
        bitstream_color: Optional[dict] = None,
        container_color: Optional[dict] = None,
    ) -> Optional[Frame]:
        """解码完成回调：旧纪元结果丢弃，新纪元结果进 PTS 重排队列。"""
        self._require_registered(surface)
        if surface.state is not SurfaceState.DECODING:
            raise ValueError(f"surface {surface.id!r} 状态 {surface.state}，不能完成解码")
        if not isinstance(pts, (int, float)) or pts < 0:
            raise ValueError("PTS 必须是非负数值")
        if not isinstance(seq, int) or seq < 0:
            raise ValueError("解码序号 seq 必须是非负整数")

        if surface.generation != self.generation:
            # seek 之后晚到的旧帧：直接作废，绝不能覆盖新位置首帧。
            self.release(surface, "decoder")
            self.metrics["stale_dropped"] += 1
            return None

        color = resolve_color(bitstream_color, container_color)
        surface.pts = int(pts)
        surface.seq = seq
        surface.state = SurfaceState.READY
        # 解码端引用转交给输出队列持有，计数不变（仍是“有人在用”）。
        frame = Frame(surface=surface, pts=surface.pts, seq=seq,
                      generation=self.generation, color=color)
        self.reorder.push(frame)
        self.metrics["reordered"] += 1
        return frame

    def pop_next(self) -> Optional[Frame]:
        return self.reorder.pop()

    # ------------------------------------------------------------------ #
    # 修复 6：回收延迟到显示完成
    # ------------------------------------------------------------------ #
    def present(self, frame_or_surface) -> PresentToken:
        surface = frame_or_surface.surface if isinstance(frame_or_surface, Frame) else frame_or_surface
        self._require_registered(surface)
        if surface.state not in (SurfaceState.READY, SurfaceState.DECODING):
            raise ValueError(f"surface {surface.id!r} 状态 {surface.state}，不能送显")
        surface.state = SurfaceState.DISPLAYING
        surface.acquire("display")  # 显示端读期间持引用
        self.metrics["acquires"] += 1
        self.release(surface, "decoder")  # 输出队列移交完毕，释放解码端引用
        token = PresentToken(surface=surface, generation=self.generation)
        self.fences.append(token)
        self.metrics["presented"] += 1
        return token

    def recycle_completed(self) -> list[Any]:
        """处理已 signal 的 present fence：显示完成才释放最后引用并回收。"""
        recycled_ids: list[Any] = []
        remaining: list[PresentToken] = []
        for token in self.fences:
            if not token.signaled:
                remaining.append(token)
                continue
            if token.surface.refcount > 0 and "display" in token.surface.owners:
                self.release(token.surface, "display")
                recycled_ids.append(token.surface.id)
        self.fences = remaining
        return recycled_ids

    def signal_all_fences(self) -> None:
        """测试/模拟辅助：把所有在途 fence 标记为显示完成。"""
        for token in self.fences:
            token.complete()

    # ------------------------------------------------------------------ #
    # seek / flush
    # ------------------------------------------------------------------ #
    def seek(self) -> int:
        self.generation += 1
        self.metrics["seeks"] += 1
        self.queue.clear()
        # 重排队列里尚未送显的旧帧立即回收；DECODING 的在途帧等完成回调时丢弃。
        for frame in self.reorder.drain():
            if frame.surface.state is SurfaceState.READY:
                self.release(frame.surface, "decoder")
        return self.generation

    def flush(self) -> int:
        self.generation += 1
        self.metrics["flushes"] += 1
        self.queue.clear()
        for frame in self.reorder.drain():
            if frame.surface.state is SurfaceState.READY:
                self.release(frame.surface, "decoder")
        # 连显示中的也强制收回（stop/切换流场景），fence 全部作废。
        for token in self.fences:
            s = token.surface
            if s.state is SurfaceState.DISPLAYING:
                s.owners.discard("display")
                s.refcount = max(0, s.refcount - 1)
                self.metrics["releases"] += 1
                if s.refcount == 0:
                    self._recycle(s)
        self.fences.clear()
        return self.generation

    # ------------------------------------------------------------------ #
    # 兜底纹理与审计
    # ------------------------------------------------------------------ #
    def _fallback(self, reason: str) -> Surface:
        self.metrics["fallback_hits"] += 1
        if self._scratch is None:
            self._scratch = Surface(id=self.FALLBACK_ID, is_fallback=True)
            self.register(self._scratch)  # 兜底纹理同样登记，不游离在池外
        scratch = self._scratch
        scratch.state = SurfaceState.DECODING
        scratch.generation = self.generation
        if scratch.refcount == 0:
            scratch.acquire("decoder")
            self.metrics["acquires"] += 1
        return scratch

    def audit(self) -> dict:
        issues: list[str] = []
        registered = set(self.registry)
        free_set = set(self.free_list)
        if len(free_set) != len(self.free_list):
            issues.append("空闲链表存在重复 id")
        if not free_set <= registered:
            issues.append("空闲链表含未登记 surface")
        for sid in self.free_list:
            free_surface = self.registry[sid]
            if free_surface.state is not SurfaceState.FREE:
                issues.append(f"surface {sid!r} 在空闲链表但状态为 {free_surface.state.value}")
            elif free_surface.refcount != 0 and not free_surface.is_fallback:
                issues.append(f"surface {sid!r} 在空闲链表但引用计数={free_surface.refcount}")
        for sid, surface in self.registry.items():
            if surface.refcount < 0:
                issues.append(f"surface {sid!r} 引用计数为负")
            if surface.state is SurfaceState.FREE:
                if surface.refcount != 0 and not surface.is_fallback:
                    issues.append(f"surface {sid!r} 标记 FREE 但引用计数={surface.refcount}")
            elif surface.refcount == 0:
                issues.append(f"surface {sid!r} 处于 {surface.state.value} 但无人持引用")
            if surface.state is SurfaceState.FREE and not surface.is_fallback and sid not in self.free_list:
                issues.append(f"surface {sid!r} 已 FREE 却不在空闲链表")
        # 只读遍历重排队列，审计本身不得产生副作用
        for frame in self.reorder._frames:
            if frame.surface.id not in registered:
                issues.append(f"重排队列引用了未登记 surface {frame.surface.id!r}")
        for token in self.fences:
            if token.surface.id not in registered:
                issues.append(f"present fence 引用了未登记 surface {token.surface.id!r}")
        return {
            "ok": not issues,
            "issues": issues,
            "generation": self.generation,
            "summary": {
                "registered": len(self.registry),
                "free": len(self.free_list),
                "in_reorder": len(self.reorder),
                "in_flight_decode": len(self.queue),
                "displaying": len(self.fences),
            },
            "metrics": dict(self.metrics),
        }


# ---------------------------------------------------------------------- #
# 色彩参数合并：修复 4 —— 码流优先，容器仅兜底
# ---------------------------------------------------------------------- #
def _lookup(source, key: str):
    if source is None:
        return None
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _is_unspecified(value) -> bool:
    """判断某层是否显式给出了该字段（None/空串/unspecified/unknown 均视为缺省）。"""
    if value is None:
        return True
    return str(value).strip().lower() in _UNSPECIFIED


def resolve_color(bitstream: Optional[dict] = None,
                  container: Optional[dict] = None) -> ColorParams:
    values: dict[str, str] = {}
    sources: dict[str, str] = {}
    for key in _COLOR_FIELDS:
        bs_value = _lookup(bitstream, key)
        if not _is_unspecified(bs_value):
            values[key] = bs_value
            sources[key] = "bitstream"
            continue
        ct_value = _lookup(container, key)
        if not _is_unspecified(ct_value):
            values[key] = ct_value
            sources[key] = "container"
            continue
        values[key] = _DEFAULT_COLOR[key]
        sources[key] = "default"
    return ColorParams(sources=sources, **values)


# ---------------------------------------------------------------------- #
# 模块级默认池与入参归一化（同时兼容契约测试的裸 int 入参）
# ---------------------------------------------------------------------- #
_DEFAULT_POOL: Optional[SurfacePool] = None


def _default() -> SurfacePool:
    global _DEFAULT_POOL
    if _DEFAULT_POOL is None:
        _DEFAULT_POOL = SurfacePool()
    return _DEFAULT_POOL


def _coerce_pool(pool: Any) -> SurfacePool:
    return pool if isinstance(pool, SurfacePool) else _default()


def _coerce_surface(target: Any, pool: Any = None) -> tuple[Surface, SurfacePool]:
    p = _coerce_pool(pool)
    if isinstance(target, Surface):
        return target, p
    if target is None:
        return p.alloc(), p
    sid = int(target)
    surface = p.registry.get(sid)
    if surface is None:
        surface = p._new_surface(sid)
    return surface, p


# ---------------------------------------------------------------------- #
# 15 个对外入口
# ---------------------------------------------------------------------- #
def surface_ref(target, action: str = "acquire", owner: str = "decoder"):
    """修复 1：按引用计数持有 surface。

    action 取 ``acquire`` / ``release`` / ``stats``，返回当前引用计数（stats
    返回包含计数、状态与持有者的字典）。
    """
    surface, pool = _coerce_surface(target)
    if action == "acquire":
        return pool.acquire(surface, owner)
    if action == "release":
        return pool.release(surface, owner)
    if action == "stats":
        return {
            "id": surface.id,
            "state": surface.state.value,
            "refcount": surface.refcount,
            "owners": sorted(surface.owners),
        }
    raise ValueError(f"未知 action {action!r}，应为 acquire/release/stats")


def pool_register(pool=None, surface=None):
    """修复 2：池扩容/外部新建纹理统一登记入口，幂等。

    支持 ``pool_register(surface)``（登记到默认池）、
    ``pool_register(pool, surface)``、``pool_register(surface_id)``
    三种形态；已登记的 id 直接返回既有 surface。
    """
    if isinstance(pool, Surface):
        surface, pool = pool, _default()
    else:
        pool = _coerce_pool(pool)
    if surface is None:
        surface = Surface(id=pool.next_surface_id())
    elif isinstance(surface, int):
        sid = surface
        if sid in pool.registry:
            return pool.registry[sid]
        surface = Surface(id=sid)
    return pool.register(surface)


def pts_reorder(frames, *, flush_remaining: bool = False):
    """修复 3：按 PTS 重排。

    - 传入帧列表（dict / Frame / Surface 均可）→ 返回按 ``(PTS, seq)``
      排好序的新列表，输入顺序不影响结果；
    - 传入正整数 → 构造一个该容量的 :class:`ReorderBuffer`。
    """
    if isinstance(frames, int):
        return ReorderBuffer(capacity=frames)
    if isinstance(frames, ReorderBuffer):
        return frames.drain() if flush_remaining else frames

    def _pts(item):
        if isinstance(item, Frame):
            return item.pts, item.seq, _id_key(item.surface.id)
        if isinstance(item, Surface):
            return item.pts, item.seq if item.seq is not None else 0, _id_key(item.id)
        pts = item.get("pts")
        if not isinstance(pts, (int, float)) or pts < 0:
            raise ValueError("重排帧必须带非负数值 pts")
        seq = item.get("seq", 0)
        sid = item.get("surface_id", item.get("id", 0))
        return pts, seq, _id_key(sid)

    return sorted(frames, key=_pts)


def color_params(bitstream=None, container=None):
    """修复 4：色彩参数以码流实际值为准，容器元数据仅在码流未指定时兜底。"""
    if bitstream is None and container is None:
        bitstream = {}
    return resolve_color(bitstream, container)


def seek_invalidate(pool=None):
    """修复 5：seek 递增纪元，作废在途解码与待显帧，返回新纪元号。"""
    return _coerce_pool(pool).seek()


def delayed_recycle(pool=None):
    """修复 6：fence 通知显示完成后才回收 surface，返回本次回收的 id 列表。"""
    p = _coerce_pool(pool)
    p.signal_all_fences()  # 模拟显示 fence 全部就绪
    return p.recycle_completed()


def determinism(frames):
    """确定性保证：同一组帧无论到达顺序如何，重排结果完全一致。

    传帧列表返回稳定排序结果；传正整数返回排序策略对象（含排序键说明）。
    """
    if isinstance(frames, int):
        return {"key": ("pts", "seq", "surface_id"), "stable": True, "capacity": frames}
    first = pts_reorder(frames)
    second = pts_reorder(list(reversed(list(frames))))
    if first != second:
        raise RuntimeError("重排结果不确定：打乱输入后输出不一致")
    return first


def validation(target):
    """入参/状态校验，非法即抛 ValueError/TypeError，合法返回 True。"""
    if isinstance(target, Surface):
        if target.refcount < 0:
            raise ValueError(f"surface {target.id!r} 引用计数为负")
        if target.state is SurfaceState.FREE and target.refcount != 0 and not target.is_fallback:
            raise ValueError(f"surface {target.id!r} FREE 但引用计数非零")
        if target.state is not SurfaceState.FREE and target.refcount == 0:
            raise ValueError(f"surface {target.id!r} 非 FREE 却无持有者")
        return True
    if isinstance(target, Frame):
        if not isinstance(target.pts, (int, float)) or target.pts < 0:
            raise ValueError("帧 PTS 非法")
        return validation(target.surface)
    pool = _coerce_pool(target)
    report = pool.audit()
    if not report["ok"]:
        raise ValueError("池状态校验失败: " + "; ".join(report["issues"]))
    return True


def audit(pool=None):
    """审计池：登记表/空闲链表/引用计数/fence 一致性，返回结构化报告。"""
    return _coerce_pool(pool).audit()


def fallback(pool=None, reason: str = "pool-exhausted"):
    """池顶到容量上限且无空闲 surface 时的安全兜底，返回共享 scratch surface。"""
    return _coerce_pool(pool)._fallback(reason)


def decode_queue(pool=None):
    """返回池的解码提交队列（工作项带 generation 纪元标记）。"""
    return _coerce_pool(pool).queue


def metrics(pool=None):
    """返回池的全部运行计数快照。"""
    return dict(_coerce_pool(pool).metrics)


def present(target, pool=None):
    """送显：显示端加引用、解码端释放引用，返回 fence 凭证。"""
    if isinstance(target, Frame):
        return target.surface and _coerce_pool(pool).present(target)
    surface, p = _coerce_surface(target, pool)
    if surface.state is SurfaceState.FREE:
        surface.state = SurfaceState.DECODING
        surface.generation = p.generation
        surface.acquire("decoder")
        p.metrics["acquires"] += 1
    return p.present(surface)


def surface_alloc(pool=None):
    """从池分配 surface；耗尽则扩容并统一登记，扩容失败走兜底。"""
    return _coerce_pool(pool).alloc()


def flush(pool=None):
    """全量冲掉解码/重排/显示管线并递增纪元，返回新纪元号。"""
    return _coerce_pool(pool).flush()
