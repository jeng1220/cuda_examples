import argparse
import ctypes
import pathlib
from enum import IntEnum

import cupyx as cpx
import numpy as np

class HostMemoryType(IntEnum):
    LOCK_FREE = 0
    PINNED = 1
    MANAGED = 2

def matul_multigpu(a, b, c, m, n, k):
    a_ptr = a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    b_ptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    c_ptr = c.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    libname = pathlib.Path().absolute() / "libsgemm_multigpu.so"
    c_lib = ctypes.CDLL(libname)
    c_lib.sgemm_multi_gpu(a_ptr, b_ptr, c_ptr, m, n, k, int(HostMemoryType.PINNED))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=32768, help="matrix size")
    parser.add_argument('--verify', action="store_true", help='cmpare to cublas and numpy')
    parser.add_argument('--loop', type=int, default=10, help="run matmul multiple times")
    args = parser.parse_args()

    sz = args.size
    m = sz
    n = sz
    k = sz

    if args.verify:
        np.random.seed(seed=0)
        tmp = np.random.randint(low=0, high=2, size=(m, k)).astype(np.float32)
        a = cpx.empty_like_pinned(tmp)
        a[:] = tmp
        tmp = np.random.randint(low=0, high=2, size=(k, n)).astype(np.float32)
        b = cpx.empty_like_pinned(tmp)
        b[:] = tmp
        tmp = np.random.randint(low=0, high=2, size=(m, n)).astype(np.float32)
        c = cpx.empty_like_pinned(tmp)
        c[:] = tmp
    else:
        a = cpx.empty_pinned((m, k), dtype=np.float32)
        b = cpx.empty_pinned((k, n), dtype=np.float32)
        c = cpx.empty_pinned((m, n), dtype=np.float32)

    for _ in range(args.loop):
        matul_multigpu(a, b, c, m, n, k)

    if args.verify:
        ref = np.matmul(a, b)
        np.testing.assert_allclose(c, ref, rtol=1e-3, atol=1e-3)
        print("PASSED")
