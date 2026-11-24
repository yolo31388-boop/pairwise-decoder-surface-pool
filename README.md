# pairwise-decoder-surface-pool

## 领域

游戏、图形与多媒体

## 主题

解码 surface 池与送显顺序

## 说明

引用计数持有、PTS 重排与 seek 作废在途解码。

初始状态下 `media/decoder_surface_pool.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
