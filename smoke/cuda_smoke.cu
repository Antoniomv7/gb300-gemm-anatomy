// gb300-gemm-anatomy Phase 0 CUDA smoke test.
//
// Trivial deterministic single-GPU vector addition on tiny buffers with full
// API/launch error checking, explicit synchronization, and complete cleanup.
// Emits concise parseable output (device count, compute capability, numeric
// correctness) for scripts/preflight.sh. No benchmarking of any kind.
//
// Exit codes: 0 = pass, 1 = any failure. On CUDA errors the cudaGetErrorName
// string (e.g. cudaErrorUnsupportedPtxVersion) is printed so the preflight
// can distinguish driver incompatibility (BLOCKED_DRIVER) from real failures.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA_SMOKE error=%s detail=\"%s\" at %s:%d\n", \
                         cudaGetErrorName(err_), cudaGetErrorString(err_),   \
                         __FILE__, __LINE__);                                \
            std::printf("CUDA_SMOKE_RESULT=FAIL\n");                         \
            return 1;                                                        \
        }                                                                    \
    } while (0)

namespace {

constexpr int kN = 1024;          // tiny working set: 3 * 4 KiB
constexpr int kThreads = 256;
constexpr int kBlocks = kN / kThreads;

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

}  // namespace

int main() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::printf("CUDA_SMOKE device_count=%d\n", device_count);
    if (device_count != 1) {
        std::fprintf(stderr,
                     "CUDA_SMOKE error=UnexpectedDeviceCount detail=\"expected exactly 1 visible device\"\n");
        std::printf("CUDA_SMOKE_RESULT=FAIL\n");
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("CUDA_SMOKE name=%s\n", prop.name);
    std::printf("CUDA_SMOKE compute_capability=%d.%d\n", prop.major, prop.minor);

    float* h_a = static_cast<float*>(std::malloc(kN * sizeof(float)));
    float* h_b = static_cast<float*>(std::malloc(kN * sizeof(float)));
    float* h_c = static_cast<float*>(std::malloc(kN * sizeof(float)));
    if (h_a == nullptr || h_b == nullptr || h_c == nullptr) {
        std::fprintf(stderr, "CUDA_SMOKE error=HostAllocFailed\n");
        std::printf("CUDA_SMOKE_RESULT=FAIL\n");
        return 1;
    }
    // Deterministic inputs whose float sum is exact: a[i]=i, b[i]=2i, c[i]=3i.
    for (int i = 0; i < kN; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
        h_c[i] = -1.0f;
    }

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, kN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, kN * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_a, h_a, kN * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, kN * sizeof(float), cudaMemcpyHostToDevice));

    vector_add<<<kBlocks, kThreads>>>(d_a, d_b, d_c, kN);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, kN * sizeof(float), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < kN; ++i) {
        if (h_c[i] != static_cast<float>(3 * i)) {
            ++mismatches;
        }
    }
    std::printf("CUDA_SMOKE elements=%d mismatches=%d\n", kN, mismatches);

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);

    if (mismatches != 0) {
        std::printf("CUDA_SMOKE correctness=MISMATCH\n");
        std::printf("CUDA_SMOKE_RESULT=FAIL\n");
        return 1;
    }
    std::printf("CUDA_SMOKE correctness=OK\n");
    std::printf("CUDA_SMOKE_RESULT=PASS\n");
    return 0;
}
