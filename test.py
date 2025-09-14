import cupy as cp
from common.backend import is_cupy
print("backend:", "cupy" if is_cupy else "numpy")
a = cp.random.random((1<<20,))
cp.cuda.Device().synchronize()
print("cupy ok; mem used:", cp.get_default_memory_pool().used_bytes()//(1024*1024), "MiB")
