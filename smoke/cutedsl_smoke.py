# gb300-gemm-anatomy Phase 0 CuTe DSL smoke test.
#
# JIT-compiles and executes a real, tiny, non-GEMM elementwise-add kernel with
# the pinned CuTe DSL (nvidia-cutlass-dsl 4.6.1), validates every output
# element numerically, and reports the device architecture. Importing this
# module has no side effects; all GPU work happens in main().
#
# Adapted from NVIDIA CUTLASS v4.6.1 at commit
# e05f953a5b3d38adc240df2ff928e0421c2abba3 (github.com/NVIDIA/cutlass):
#   - the naive elementwise-add kernel of the CuTe DSL quick start
#     (docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/quick_start.html)
#   - the raw-pointer host pattern (make_ptr / cute.make_tensor) of
#     examples/python/CuTeDSL/dsl_tutorials/call_bypass_dlpack.py
# Those examples are published under the BSD-3-Clause license; notice below.
#
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys

import numpy as np

import cuda.bindings.driver as cuda_driver

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr

# Tiny deterministic problem: 64 x 32 float32, one exact-add pass.
M = 64
N = 32
THREADS_PER_BLOCK = 256


@cute.kernel
def add_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    idx = bidx * bdim + tidx
    _, n = gA.shape
    mi = idx // n
    ni = idx % n
    gC[mi, ni] = gA[mi, ni] + gB[mi, ni]


@cute.jit
def add_launcher(
    a_ptr: cute.Pointer,
    b_ptr: cute.Pointer,
    c_ptr: cute.Pointer,
    m: cutlass.Constexpr,
    n: cutlass.Constexpr,
    threads_per_block: cutlass.Constexpr,
):
    # Row-major (m, n) layout over raw global-memory pointers.
    layout = cute.make_ordered_layout((m, n), order=(1, 0))
    mA = cute.make_tensor(a_ptr, layout=layout)
    mB = cute.make_tensor(b_ptr, layout=layout)
    mC = cute.make_tensor(c_ptr, layout=layout)
    # m*n is a multiple of threads_per_block, so every thread is in range.
    add_kernel(mA, mB, mC).launch(
        grid=[(m * n) // threads_per_block, 1, 1],
        block=[threads_per_block, 1, 1],
    )


def error_name(err) -> str:
    """Best-effort symbolic name for a CUresult (e.g. CUDA_ERROR_...)."""
    name_err, name = cuda_driver.cuGetErrorName(err)
    if name_err != cuda_driver.CUresult.CUDA_SUCCESS:
        return f"CUresult({int(err)})"
    return name.decode() if isinstance(name, bytes) else str(name)


def check_cuda(result, what):
    """Unpack a cuda.bindings (err, value...) tuple, raising on any error."""
    err, rest = result[0], result[1:]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {error_name(err)}")
    if len(rest) == 0:
        return None
    if len(rest) == 1:
        return rest[0]
    return rest


def cleanup(device, buffers) -> bool:
    """Free device buffers and release the primary context, checking every
    return code. Prints each cleanup failure and returns False if any cleanup
    call failed. Never raises, so an in-flight primary exception is not
    replaced or hidden."""
    ok = True
    for label, buf in buffers:
        err = cuda_driver.cuMemFree(buf)[0]
        if err != cuda_driver.CUresult.CUDA_SUCCESS:
            print(f"CUTEDSL_SMOKE cleanup_error=cuMemFree({label}):{error_name(err)}")
            ok = False
    err = cuda_driver.cuDevicePrimaryCtxRelease(device)[0]
    if err != cuda_driver.CUresult.CUDA_SUCCESS:
        print(f"CUTEDSL_SMOKE cleanup_error=cuDevicePrimaryCtxRelease:{error_name(err)}")
        ok = False
    return ok


def main() -> int:
    assert (M * N) % THREADS_PER_BLOCK == 0

    check_cuda(cuda_driver.cuInit(0), "cuInit")
    device_count = check_cuda(cuda_driver.cuDeviceGetCount(), "cuDeviceGetCount")
    print(f"CUTEDSL_SMOKE device_count={device_count}")
    if device_count != 1:
        print("CUTEDSL_SMOKE error=UnexpectedDeviceCount expected exactly 1 visible device")
        print("CUTEDSL_SMOKE_RESULT=FAIL")
        return 1

    device = check_cuda(cuda_driver.cuDeviceGet(0), "cuDeviceGet")
    name = check_cuda(cuda_driver.cuDeviceGetName(128, device), "cuDeviceGetName")
    name = name.decode(errors="replace").rstrip("\x00").strip()
    cc_major = check_cuda(
        cuda_driver.cuDeviceGetAttribute(
            cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            device,
        ),
        "cuDeviceGetAttribute(cc major)",
    )
    cc_minor = check_cuda(
        cuda_driver.cuDeviceGetAttribute(
            cuda_driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            device,
        ),
        "cuDeviceGetAttribute(cc minor)",
    )
    print(f"CUTEDSL_SMOKE name={name}")
    print(f"CUTEDSL_SMOKE compute_capability={cc_major}.{cc_minor}")

    context = check_cuda(cuda_driver.cuDevicePrimaryCtxRetain(device), "cuDevicePrimaryCtxRetain")
    check_cuda(cuda_driver.cuCtxSetCurrent(context), "cuCtxSetCurrent")

    # Buffers are recorded as they are successfully allocated so that cleanup
    # frees exactly what exists, on both the success and the error path.
    buffers = []
    status = 1
    primary_exc = None
    try:
        # Deterministic inputs whose float32 sum is exact: a=i, b=2i, c=3i.
        a_host = np.arange(M * N, dtype=np.float32).reshape(M, N)
        b_host = 2.0 * a_host
        c_host = np.full((M, N), -1.0, dtype=np.float32)
        nbytes = a_host.nbytes

        d_a = check_cuda(cuda_driver.cuMemAlloc(nbytes), "cuMemAlloc(a)")
        buffers.append(("a", d_a))
        d_b = check_cuda(cuda_driver.cuMemAlloc(nbytes), "cuMemAlloc(b)")
        buffers.append(("b", d_b))
        d_c = check_cuda(cuda_driver.cuMemAlloc(nbytes), "cuMemAlloc(c)")
        buffers.append(("c", d_c))
        check_cuda(cuda_driver.cuMemcpyHtoD(d_a, a_host, nbytes), "cuMemcpyHtoD(a)")
        check_cuda(cuda_driver.cuMemcpyHtoD(d_b, b_host, nbytes), "cuMemcpyHtoD(b)")
        check_cuda(cuda_driver.cuMemcpyHtoD(d_c, c_host, nbytes), "cuMemcpyHtoD(c)")

        a_ptr = make_ptr(cutlass.Float32, int(d_a), cute.AddressSpace.gmem, assumed_align=16)
        b_ptr = make_ptr(cutlass.Float32, int(d_b), cute.AddressSpace.gmem, assumed_align=16)
        c_ptr = make_ptr(cutlass.Float32, int(d_c), cute.AddressSpace.gmem, assumed_align=16)

        print("CUTEDSL_SMOKE jit=compiling")
        add_launcher(a_ptr, b_ptr, c_ptr, M, N, THREADS_PER_BLOCK)
        check_cuda(cuda_driver.cuCtxSynchronize(), "cuCtxSynchronize")
        print("CUTEDSL_SMOKE jit=executed")

        check_cuda(cuda_driver.cuMemcpyDtoH(c_host, d_c, nbytes), "cuMemcpyDtoH(c)")

        expected = a_host + b_host
        mismatches = int(np.count_nonzero(c_host != expected))
        print(f"CUTEDSL_SMOKE elements={M * N} mismatches={mismatches}")
        if mismatches != 0:
            print("CUTEDSL_SMOKE correctness=MISMATCH")
            status = 1
        else:
            print("CUTEDSL_SMOKE correctness=OK")
            status = 0
    except Exception as exc:
        primary_exc = exc

    cleanup_ok = cleanup(device, buffers)

    if primary_exc is not None:
        # Re-raised only after cleanup ran; cleanup failures were printed
        # above and do not replace the primary error.
        raise primary_exc
    if not cleanup_ok:
        print("CUTEDSL_SMOKE cleanup=FAILED")
        status = 1
    print(f"CUTEDSL_SMOKE_RESULT={'PASS' if status == 0 else 'FAIL'}")
    return status


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # ensure the error name reaches the preflight log
        print(f"CUTEDSL_SMOKE exception={exc}")
        print("CUTEDSL_SMOKE_RESULT=FAIL")
        sys.exit(1)
