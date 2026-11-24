import pytest
import media.decoder_surface_pool as mod

CASES = [
    ("surface_ref", (1,)),
    ("pool_register", (1,)),
    ("pts_reorder", (1,)),
    ("color_params", (1,)),
    ("seek_invalidate", (1,)),
    ("delayed_recycle", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("decode_queue", (1,)),
    ("metrics", (1,)),
    ("present", (1,)),
    ("surface_alloc", (1,)),
    ("flush", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
