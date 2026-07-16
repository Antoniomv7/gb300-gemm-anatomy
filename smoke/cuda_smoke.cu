// gb300-gemm-anatomy Phase 0 CUDA smoke test.
//
// Trivial deterministic single-GPU vector addition on tiny buffers with full
// API/launch error checking, explicit synchronization, and complete cleanup.
// Emits concise parseable output (device count, compute capability, numeric
// correctness) for scripts/preflight.sh. No benchmarking of any kind.
//
// Cleanup is RAII-based: host buffers are std::vector and device buffers are
// freed by DeviceBuffer destructors, so every successfully allocated resource
// is released on every error path. Cleanup never masks the original CUDA
// error: the failing call's error is printed first and destructors ignore
// their own cudaFree status.
//
// Exit codes: 0 = pass, 1 = any failure. On CUDA errors the cudaGetErrorName
// string (e.g. cudaErrorUnsupportedPtxVersion) is printed so the preflight
// can distinguish driver incompatibility (BLOCKED_DRIVER) from real failures.

#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err_ = (call);                                           \
        if (err_ != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA_SMOKE error=%s detail=\"%s\" at %s:%d\n", \
                         cudaGetErrorName(err_), cudaGetErrorString(err_),   \
                         __FILE__, __LINE__);                                \
            return 1;                                                        \
        }                                                                    \
    } while (0)

namespace {

constexpr int kN = 1024;          // tiny working set: 3 * 4 KiB
constexpr int kThreads = 256;
constexpr int kBlocks = kN / kThreads;

// Owns one device allocation; frees it on scope exit. The cudaFree status is
// deliberately ignored so cleanup cannot overwrite the primary error already
// reported by CUDA_CHECK.
class DeviceBuffer {
 public:
    DeviceBuffer() = default;
    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            (void)cudaFree(ptr_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    cudaError_t allocate(size_t bytes) { return cudaMalloc(&ptr_, bytes); }
    float* get() const { return ptr_; }

 private:
    float* ptr_ = nullptr;
};

__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// All checks live here so that CUDA_CHECK's early returns unwind through the
// RAII owners declared below. main() only reports the final result.
int run_smoke() {
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::printf("CUDA_SMOKE device_count=%d\n", device_count);
    if (device_count != 1) {
        std::fprintf(stderr,
                     "CUDA_SMOKE error=UnexpectedDeviceCount detail=\"expected exactly 1 visible device\"\n");
        return 1;
    }

    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("CUDA_SMOKE name=%s\n", prop.name);
    std::printf("CUDA_SMOKE compute_capability=%d.%d\n", prop.major, prop.minor);

    // Deterministic inputs whose float sum is exact: a[i]=i, b[i]=2i, c[i]=3i.
    std::vector<float> h_a(kN), h_b(kN), h_c(kN, -1.0f);
    for (int i = 0; i < kN; ++i) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    constexpr size_t kBytes = kN * sizeof(float);
    DeviceBuffer d_a, d_b, d_c;
    CUDA_CHECK(d_a.allocate(kBytes));
    CUDA_CHECK(d_b.allocate(kBytes));
    CUDA_CHECK(d_c.allocate(kBytes));
    CUDA_CHECK(cudaMemcpy(d_a.get(), h_a.data(), kBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b.get(), h_b.data(), kBytes, cudaMemcpyHostToDevice));

    vector_add<<<kBlocks, kThreads>>>(d_a.get(), d_b.get(), d_c.get(), kN);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c.data(), d_c.get(), kBytes, cudaMemcpyDeviceToHost));

    int mismatches = 0;
    for (int i = 0; i < kN; ++i) {
        if (h_c[i] != static_cast<float>(3 * i)) {
            ++mismatches;
        }
    }
    std::printf("CUDA_SMOKE elements=%d mismatches=%d\n", kN, mismatches);
    if (mismatches != 0) {
        std::printf("CUDA_SMOKE correctness=MISMATCH\n");
        return 1;
    }
    std::printf("CUDA_SMOKE correctness=OK\n");
    return 0;
}

}  // namespace

int main() {
    int rc = run_smoke();
    std::printf("CUDA_SMOKE_RESULT=%s\n", rc == 0 ? "PASS" : "FAIL");
    return rc;
}
