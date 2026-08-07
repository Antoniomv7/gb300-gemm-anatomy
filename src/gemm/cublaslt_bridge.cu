// P3.3 - minimal C-ABI bridge to cuBLASLt for one frozen BF16 GEMM.
//
// This file owns no GEMM kernel. It creates a cuBLASLt handle, the operation
// descriptor, four matrix layouts, a preference object, queries the vendor
// heuristic, validates the selected algorithm, allocates exactly the workspace
// that algorithm requires, exports the selected algorithm's metadata, and
// launches `cublasLtMatmul`. Nothing else. No NVIDIA source is copied, forked,
// patched, or vendored: only the public cuBLASLt API declared in the pinned
// CUDA 13.1 headers is called.
//
// The whole GEMM configuration is frozen here as compile-time constants, not
// as parameters: `p33_plan_create` accepts only device pointers and a stream.
// The shape, leading dimensions, transpose modes, memory orders, data types,
// compute and scale types, pointer mode, epilogue, alpha, beta, workspace
// limit, and requested heuristic count therefore cannot be changed by the
// caller. Every one of them is also reported back through `p33_plan_info_t`,
// so the Python wrapper can assert that this translation unit and the wrapper
// agree, instead of restating the contract twice and hoping.
//
// Contract with the caller:
//
//   * Every exported function is `extern "C"` and returns `int` (0 on success,
//     non-zero on failure) or a plain pointer/size. No C++ exception may cross
//     the boundary: every entry point has a catch-all handler.
//   * Nothing is ever written to stdout or stderr. A failure records a message
//     retrievable with `p33_last_error()`.
//   * The bridge never chooses a fallback. If the frozen configuration has no
//     supported algorithm, it fails and says so.
//
// Algorithm policy (fixed, non-autotuned, never benchmarked):
//
//   * workspace limit exactly P33_WORKSPACE_LIMIT_BYTES;
//   * exactly P33_HEURISTIC_REQUESTED heuristic results requested;
//   * CUBLASLT_SEARCH_BEST_FIT;
//   * the first returned entry whose state is CUBLAS_STATUS_SUCCESS is taken;
//   * that algorithm is re-validated with cublasLtMatmulAlgoCheck();
//   * it is rejected if it needs more workspace than the fixed limit;
//   * exactly the required workspace is allocated (a null pointer when the
//     requirement is zero);
//   * only that one algorithm is ever executed.
//
// No candidate is timed, compared, or ranked here: this bridge measures
// nothing. The Python wrapper owns every timer.

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <new>

// --- Frozen configuration ----------------------------------------------------
//
// (M,N,K,L) = (4096,4096,4096,1), C = A x B^T, BF16 x BF16 -> FP32 with FP32
// accumulation. A is M x K row-major (lda = K), B is N x K row-major
// (ldb = K), C and D are M x N row-major (ldc = ldd = N). These match the
// physical layouts the pinned CuTe DSL tensor factory produces for
// a_major=k, b_major=k, c_major=n, so no data is transposed, copied, or
// reinterpreted anywhere in this unit.

#define P33_ABI_VERSION 1

static const int64_t P33_M = 4096;
static const int64_t P33_N = 4096;
static const int64_t P33_K = 4096;
static const int64_t P33_BATCH_COUNT = 1;

static const int64_t P33_LDA = P33_K;
static const int64_t P33_LDB = P33_K;
static const int64_t P33_LDC = P33_N;
static const int64_t P33_LDD = P33_N;

static const cublasOperation_t P33_TRANSA = CUBLAS_OP_N;
static const cublasOperation_t P33_TRANSB = CUBLAS_OP_T;

static const cublasLtOrder_t P33_ORDER = CUBLASLT_ORDER_ROW;

static const cudaDataType_t P33_AB_TYPE = CUDA_R_16BF;
static const cudaDataType_t P33_CD_TYPE = CUDA_R_32F;
static const cublasComputeType_t P33_COMPUTE_TYPE = CUBLAS_COMPUTE_32F;
static const cudaDataType_t P33_SCALE_TYPE = CUDA_R_32F;

static const cublasLtPointerMode_t P33_POINTER_MODE = CUBLASLT_POINTER_MODE_HOST;
static const cublasLtEpilogue_t P33_EPILOGUE = CUBLASLT_EPILOGUE_DEFAULT;

static const float P33_ALPHA = 1.0f;
static const float P33_BETA = 0.0f;

// Exactly 64 MiB. Never negotiated, never retried with another value.
static const uint64_t P33_WORKSPACE_LIMIT_BYTES = 67108864ULL;

// Exactly 32 heuristic results are requested; the first supported one wins.
static const int P33_HEURISTIC_REQUESTED = 32;

static const cublasLtMatmulSearch_t P33_SEARCH_MODE = CUBLASLT_SEARCH_BEST_FIT;

// The largest pointer alignment this bridge will ever claim. cuBLASLt's own
// default for the minimum-alignment preferences is 256 bytes; claiming more
// than a pointer actually satisfies would let the heuristic return an
// algorithm the data cannot legally feed.
static const uint32_t P33_MAX_ALIGNMENT_BYTES = 256u;

// --- Error reporting ---------------------------------------------------------

static thread_local char g_last_error[1024] = {0};

static void p33_clear_error(void) { g_last_error[0] = '\0'; }

static void p33_set_error(const char* message) {
    if (message == nullptr) {
        g_last_error[0] = '\0';
        return;
    }
    std::snprintf(g_last_error, sizeof(g_last_error), "%s", message);
}

static void p33_set_error_status(const char* what, cublasStatus_t status) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s failed with cublasStatus_t=%d",
                  what, static_cast<int>(status));
}

static void p33_set_error_cuda(const char* what, cudaError_t status) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s failed with cudaError_t=%d (%s)",
                  what, static_cast<int>(status), cudaGetErrorString(status));
}

// --- Exported metadata -------------------------------------------------------
//
// Every member is 8 bytes wide so the struct has one unambiguous layout with
// no padding, which keeps the ctypes mirror on the Python side exact. The
// wrapper additionally checks sizeof() against p33_plan_info_size().

extern "C" {

typedef struct P33PlanInfo {
    int64_t abi_version;
    int64_t cublaslt_version;

    int64_t m;
    int64_t n;
    int64_t k;
    int64_t batch_count;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t ldd;

    int64_t transa;
    int64_t transb;
    int64_t order_a;
    int64_t order_b;
    int64_t order_c;
    int64_t order_d;
    int64_t type_a;
    int64_t type_b;
    int64_t type_c;
    int64_t type_d;
    int64_t compute_type;
    int64_t scale_type;
    int64_t pointer_mode;
    int64_t epilogue;

    int64_t search_mode;
    int64_t workspace_limit_bytes;
    int64_t workspace_bytes;
    int64_t workspace_is_null;

    int64_t heuristic_requested;
    int64_t heuristic_returned;
    int64_t heuristic_index;

    int64_t alignment_a_bytes;
    int64_t alignment_b_bytes;
    int64_t alignment_c_bytes;
    int64_t alignment_d_bytes;

    int64_t algo_id;
    int64_t tile_id;
    int64_t stages_id;
    int64_t split_k;
    int64_t reduction_scheme;
    int64_t cta_swizzling;
    int64_t custom_option;
    int64_t inner_shape_id;
    int64_t cluster_shape_id;

    double waves_count;
    double alpha;
    double beta;
} P33PlanInfo;

}  // extern "C"

// --- Plan --------------------------------------------------------------------

struct P33Plan {
    cublasLtHandle_t handle;
    cublasLtMatmulDesc_t operation;
    cublasLtMatrixLayout_t layout_a;
    cublasLtMatrixLayout_t layout_b;
    cublasLtMatrixLayout_t layout_c;
    cublasLtMatrixLayout_t layout_d;
    cublasLtMatmulAlgo_t algo;

    const void* a;
    const void* b;
    const void* c;
    void* d;

    void* workspace;
    size_t workspace_bytes;

    cudaStream_t stream;
};

static void p33_plan_release(P33Plan* plan) {
    if (plan == nullptr) {
        return;
    }
    if (plan->workspace != nullptr) {
        cudaFree(plan->workspace);
        plan->workspace = nullptr;
    }
    if (plan->layout_d != nullptr) {
        cublasLtMatrixLayoutDestroy(plan->layout_d);
    }
    if (plan->layout_c != nullptr) {
        cublasLtMatrixLayoutDestroy(plan->layout_c);
    }
    if (plan->layout_b != nullptr) {
        cublasLtMatrixLayoutDestroy(plan->layout_b);
    }
    if (plan->layout_a != nullptr) {
        cublasLtMatrixLayoutDestroy(plan->layout_a);
    }
    if (plan->operation != nullptr) {
        cublasLtMatmulDescDestroy(plan->operation);
    }
    if (plan->handle != nullptr) {
        cublasLtDestroy(plan->handle);
    }
    delete plan;
}

// The largest power of two, capped at P33_MAX_ALIGNMENT_BYTES, that divides
// the given address. Never overstates: the returned value is a true divisor of
// the pointer, so the minimum-alignment preference derived from it can only
// exclude algorithms the buffer genuinely cannot satisfy.
static uint32_t p33_pointer_alignment(const void* pointer) {
    const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
    if (address == 0) {
        return 0u;
    }
    uint32_t alignment = 1u;
    while (alignment < P33_MAX_ALIGNMENT_BYTES &&
           (address % (static_cast<uintptr_t>(alignment) * 2u)) == 0u) {
        alignment *= 2u;
    }
    return alignment;
}

static int p33_create_layout(cublasLtMatrixLayout_t* layout,
                             cudaDataType_t type,
                             uint64_t rows,
                             uint64_t cols,
                             int64_t ld,
                             const char* name) {
    cublasStatus_t status = cublasLtMatrixLayoutCreate(layout, type, rows, cols, ld);
    if (status != CUBLAS_STATUS_SUCCESS) {
        char what[128];
        std::snprintf(what, sizeof(what), "cublasLtMatrixLayoutCreate(%s)", name);
        p33_set_error_status(what, status);
        return 1;
    }

    const int32_t order = static_cast<int32_t>(P33_ORDER);
    status = cublasLtMatrixLayoutSetAttribute(*layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                                              sizeof(order));
    if (status != CUBLAS_STATUS_SUCCESS) {
        char what[128];
        std::snprintf(what, sizeof(what), "cublasLtMatrixLayoutSetAttribute(ORDER, %s)", name);
        p33_set_error_status(what, status);
        return 1;
    }

    const int32_t batch_count = static_cast<int32_t>(P33_BATCH_COUNT);
    status = cublasLtMatrixLayoutSetAttribute(*layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                                              &batch_count, sizeof(batch_count));
    if (status != CUBLAS_STATUS_SUCCESS) {
        char what[128];
        std::snprintf(what, sizeof(what), "cublasLtMatrixLayoutSetAttribute(BATCH_COUNT, %s)",
                      name);
        p33_set_error_status(what, status);
        return 1;
    }
    return 0;
}

// Reads one algorithm configuration attribute into an int64_t, failing closed
// if the attribute is unavailable or the library writes an unexpected width.
template <typename T>
static int p33_read_algo_config(const cublasLtMatmulAlgo_t* algo,
                                cublasLtMatmulAlgoConfigAttributes_t attribute,
                                const char* name,
                                int64_t* out) {
    T value = T();
    size_t written = 0;
    const cublasStatus_t status =
        cublasLtMatmulAlgoConfigGetAttribute(algo, attribute, &value, sizeof(value), &written);
    if (status != CUBLAS_STATUS_SUCCESS) {
        char what[160];
        std::snprintf(what, sizeof(what), "cublasLtMatmulAlgoConfigGetAttribute(%s)", name);
        p33_set_error_status(what, status);
        return 1;
    }
    if (written != sizeof(value)) {
        std::snprintf(g_last_error, sizeof(g_last_error),
                      "cublasLtMatmulAlgoConfigGetAttribute(%s) wrote %zu bytes, expected %zu",
                      name, written, sizeof(value));
        return 1;
    }
    *out = static_cast<int64_t>(value);
    return 0;
}

extern "C" {

int p33_bridge_abi_version(void) { return P33_ABI_VERSION; }

size_t p33_plan_info_size(void) { return sizeof(P33PlanInfo); }

const char* p33_last_error(void) { return g_last_error; }

size_t p33_cublaslt_version(void) {
    // The runtime library's own version, not a repository constant and not a
    // build-time macro: this is what actually executed.
    return cublasLtGetVersion();
}

// Creates the whole cuBLASLt plan for the frozen configuration and reports the
// selected algorithm's metadata. This is the only function the wrapper times
// as `setup_time_ms`; it performs no matmul.
int p33_plan_create(const void* a,
                    const void* b,
                    const void* c,
                    void* d,
                    void* stream,
                    P33Plan** out_plan,
                    P33PlanInfo* out_info) {
    p33_clear_error();

    if (out_plan == nullptr || out_info == nullptr) {
        p33_set_error("p33_plan_create: out_plan and out_info must not be null");
        return 1;
    }
    *out_plan = nullptr;
    std::memset(out_info, 0, sizeof(*out_info));

    if (a == nullptr || b == nullptr || c == nullptr || d == nullptr) {
        p33_set_error("p33_plan_create: every operand pointer must be a valid device pointer");
        return 1;
    }

    P33Plan* plan = nullptr;
    try {
        plan = new (std::nothrow) P33Plan();
        if (plan == nullptr) {
            p33_set_error("p33_plan_create: out of host memory");
            return 1;
        }
        std::memset(plan, 0, sizeof(*plan));
        plan->a = a;
        plan->b = b;
        plan->c = c;
        plan->d = d;
        plan->stream = static_cast<cudaStream_t>(stream);

        cublasStatus_t status = cublasLtCreate(&plan->handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            p33_set_error_status("cublasLtCreate", status);
            p33_plan_release(plan);
            return 1;
        }

        status = cublasLtMatmulDescCreate(&plan->operation, P33_COMPUTE_TYPE, P33_SCALE_TYPE);
        if (status != CUBLAS_STATUS_SUCCESS) {
            p33_set_error_status("cublasLtMatmulDescCreate", status);
            p33_plan_release(plan);
            return 1;
        }

        const int32_t transa = static_cast<int32_t>(P33_TRANSA);
        status = cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSA,
                                                &transa, sizeof(transa));
        if (status != CUBLAS_STATUS_SUCCESS) {
            p33_set_error_status("cublasLtMatmulDescSetAttribute(TRANSA)", status);
            p33_plan_release(plan);
            return 1;
        }

        const int32_t transb = static_cast<int32_t>(P33_TRANSB);
        status = cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_TRANSB,
                                                &transb, sizeof(transb));
        if (status != CUBLAS_STATUS_SUCCESS) {
            p33_set_error_status("cublasLtMatmulDescSetAttribute(TRANSB)", status);
            p33_plan_release(plan);
            return 1;
        }

        const int32_t pointer_mode = static_cast<int32_t>(P33_POINTER_MODE);
        status = cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                                &pointer_mode, sizeof(pointer_mode));
        if (status != CUBLAS_STATUS_SUCCESS) {
            p33_set_error_status("cublasLtMatmulDescSetAttribute(POINTER_MODE)", status);
            p33_plan_release(plan);
            return 1;
        }

        // The epilogue attribute is documented as uint32_t, unlike the int32_t
        // TRANSA/TRANSB/POINTER_MODE attributes above; the widths are written
        // out separately here rather than assumed to be uniform.
        const uint32_t epilogue = static_cast<uint32_t>(P33_EPILOGUE);
        status = cublasLtMatmulDescSetAttribute(plan->operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                                &epilogue, sizeof(epilogue));
        if (status != CUBLAS_STATUS_SUCCESS) {
            p33_set_error_status("cublasLtMatmulDescSetAttribute(EPILOGUE)", status);
            p33_plan_release(plan);
            return 1;
        }

        if (p33_create_layout(&plan->layout_a, P33_AB_TYPE, static_cast<uint64_t>(P33_M),
                              static_cast<uint64_t>(P33_K), P33_LDA, "A") != 0 ||
            p33_create_layout(&plan->layout_b, P33_AB_TYPE, static_cast<uint64_t>(P33_N),
                              static_cast<uint64_t>(P33_K), P33_LDB, "B") != 0 ||
            p33_create_layout(&plan->layout_c, P33_CD_TYPE, static_cast<uint64_t>(P33_M),
                              static_cast<uint64_t>(P33_N), P33_LDC, "C") != 0 ||
            p33_create_layout(&plan->layout_d, P33_CD_TYPE, static_cast<uint64_t>(P33_M),
                              static_cast<uint64_t>(P33_N), P33_LDD, "D") != 0) {
            p33_plan_release(plan);
            return 1;
        }

        const uint32_t alignment_a = p33_pointer_alignment(a);
        const uint32_t alignment_b = p33_pointer_alignment(b);
        const uint32_t alignment_c = p33_pointer_alignment(c);
        const uint32_t alignment_d = p33_pointer_alignment(d);

        cublasLtMatmulPreference_t preference = nullptr;
        status = cublasLtMatmulPreferenceCreate(&preference);
        if (status != CUBLAS_STATUS_SUCCESS) {
            p33_set_error_status("cublasLtMatmulPreferenceCreate", status);
            p33_plan_release(plan);
            return 1;
        }

        struct PreferenceSetting {
            cublasLtMatmulPreferenceAttributes_t attribute;
            const void* value;
            size_t size;
            const char* name;
        };

        const uint32_t search_mode = static_cast<uint32_t>(P33_SEARCH_MODE);
        const uint64_t workspace_limit = P33_WORKSPACE_LIMIT_BYTES;
        const PreferenceSetting settings[] = {
            {CUBLASLT_MATMUL_PREF_SEARCH_MODE, &search_mode, sizeof(search_mode), "SEARCH_MODE"},
            {CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_limit, sizeof(workspace_limit),
             "MAX_WORKSPACE_BYTES"},
            {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &alignment_a, sizeof(alignment_a),
             "MIN_ALIGNMENT_A_BYTES"},
            {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &alignment_b, sizeof(alignment_b),
             "MIN_ALIGNMENT_B_BYTES"},
            {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &alignment_c, sizeof(alignment_c),
             "MIN_ALIGNMENT_C_BYTES"},
            {CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &alignment_d, sizeof(alignment_d),
             "MIN_ALIGNMENT_D_BYTES"},
        };

        for (const PreferenceSetting& setting : settings) {
            status = cublasLtMatmulPreferenceSetAttribute(preference, setting.attribute,
                                                          setting.value, setting.size);
            if (status != CUBLAS_STATUS_SUCCESS) {
                char what[160];
                std::snprintf(what, sizeof(what), "cublasLtMatmulPreferenceSetAttribute(%s)",
                              setting.name);
                p33_set_error_status(what, status);
                cublasLtMatmulPreferenceDestroy(preference);
                p33_plan_release(plan);
                return 1;
            }
        }

        cublasLtMatmulHeuristicResult_t results[P33_HEURISTIC_REQUESTED];
        std::memset(results, 0, sizeof(results));
        int returned = 0;
        status = cublasLtMatmulAlgoGetHeuristic(plan->handle, plan->operation, plan->layout_a,
                                                plan->layout_b, plan->layout_c, plan->layout_d,
                                                preference, P33_HEURISTIC_REQUESTED, results,
                                                &returned);
        cublasLtMatmulPreferenceDestroy(preference);
        if (status != CUBLAS_STATUS_SUCCESS) {
            p33_set_error_status("cublasLtMatmulAlgoGetHeuristic", status);
            p33_plan_release(plan);
            return 1;
        }
        if (returned <= 0 || returned > P33_HEURISTIC_REQUESTED) {
            std::snprintf(g_last_error, sizeof(g_last_error),
                          "cublasLtMatmulAlgoGetHeuristic returned %d results for the frozen "
                          "P3.3 configuration; no algorithm is available and P3.3 never retries "
                          "with another layout, type, compute mode, workspace limit, or API",
                          returned);
            p33_plan_release(plan);
            return 1;
        }

        // The first supported entry wins. No candidate is executed, timed, or
        // compared here: this is a vendor-heuristic selection, not a search.
        int selected = -1;
        for (int index = 0; index < returned; ++index) {
            if (results[index].state == CUBLAS_STATUS_SUCCESS) {
                selected = index;
                break;
            }
        }
        if (selected < 0) {
            std::snprintf(g_last_error, sizeof(g_last_error),
                          "none of the %d heuristic results reported CUBLAS_STATUS_SUCCESS for "
                          "the frozen P3.3 configuration",
                          returned);
            p33_plan_release(plan);
            return 1;
        }

        if (static_cast<uint64_t>(results[selected].workspaceSize) > P33_WORKSPACE_LIMIT_BYTES) {
            std::snprintf(g_last_error, sizeof(g_last_error),
                          "heuristic result %d needs %zu workspace bytes, above the fixed limit "
                          "of %llu",
                          selected, results[selected].workspaceSize,
                          static_cast<unsigned long long>(P33_WORKSPACE_LIMIT_BYTES));
            p33_plan_release(plan);
            return 1;
        }

        plan->algo = results[selected].algo;

        // Re-validate exactly the algorithm that will run, against exactly the
        // descriptors it will run with.
        cublasLtMatmulHeuristicResult_t checked;
        std::memset(&checked, 0, sizeof(checked));
        status = cublasLtMatmulAlgoCheck(plan->handle, plan->operation, plan->layout_a,
                                         plan->layout_b, plan->layout_c, plan->layout_d,
                                         &plan->algo, &checked);
        if (status != CUBLAS_STATUS_SUCCESS) {
            p33_set_error_status("cublasLtMatmulAlgoCheck", status);
            p33_plan_release(plan);
            return 1;
        }
        if (checked.state != CUBLAS_STATUS_SUCCESS) {
            std::snprintf(g_last_error, sizeof(g_last_error),
                          "cublasLtMatmulAlgoCheck rejected heuristic result %d with "
                          "cublasStatus_t=%d",
                          selected, static_cast<int>(checked.state));
            p33_plan_release(plan);
            return 1;
        }
        if (static_cast<uint64_t>(checked.workspaceSize) > P33_WORKSPACE_LIMIT_BYTES) {
            std::snprintf(g_last_error, sizeof(g_last_error),
                          "the validated algorithm needs %zu workspace bytes, above the fixed "
                          "limit of %llu",
                          checked.workspaceSize,
                          static_cast<unsigned long long>(P33_WORKSPACE_LIMIT_BYTES));
            p33_plan_release(plan);
            return 1;
        }

        // Exactly what the validated algorithm requires, and a null pointer
        // when it requires nothing at all.
        plan->workspace_bytes = checked.workspaceSize;
        if (plan->workspace_bytes > 0) {
            const cudaError_t allocated = cudaMalloc(&plan->workspace, plan->workspace_bytes);
            if (allocated != cudaSuccess) {
                plan->workspace = nullptr;
                p33_set_error_cuda("cudaMalloc(cuBLASLt workspace)", allocated);
                p33_plan_release(plan);
                return 1;
            }
        } else {
            plan->workspace = nullptr;
        }

        int64_t algo_id = 0;
        int64_t tile_id = 0;
        int64_t stages_id = 0;
        int64_t split_k = 0;
        int64_t reduction_scheme = 0;
        int64_t cta_swizzling = 0;
        int64_t custom_option = 0;
        int64_t inner_shape_id = 0;
        int64_t cluster_shape_id = 0;

        if (p33_read_algo_config<int32_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_ID, "ID", &algo_id) != 0 ||
            p33_read_algo_config<uint32_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_TILE_ID, "TILE_ID",
                                           &tile_id) != 0 ||
            p33_read_algo_config<uint32_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
                                           "STAGES_ID", &stages_id) != 0 ||
            p33_read_algo_config<int32_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
                                          "SPLITK_NUM", &split_k) != 0 ||
            p33_read_algo_config<uint32_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
                                           "REDUCTION_SCHEME", &reduction_scheme) != 0 ||
            p33_read_algo_config<uint32_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
                                           "CTA_SWIZZLING", &cta_swizzling) != 0 ||
            p33_read_algo_config<uint32_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
                                           "CUSTOM_OPTION", &custom_option) != 0 ||
            p33_read_algo_config<uint16_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
                                           "INNER_SHAPE_ID", &inner_shape_id) != 0 ||
            p33_read_algo_config<uint16_t>(&plan->algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
                                           "CLUSTER_SHAPE_ID", &cluster_shape_id) != 0) {
            p33_plan_release(plan);
            return 1;
        }

        out_info->abi_version = P33_ABI_VERSION;
        out_info->cublaslt_version = static_cast<int64_t>(cublasLtGetVersion());

        out_info->m = P33_M;
        out_info->n = P33_N;
        out_info->k = P33_K;
        out_info->batch_count = P33_BATCH_COUNT;
        out_info->lda = P33_LDA;
        out_info->ldb = P33_LDB;
        out_info->ldc = P33_LDC;
        out_info->ldd = P33_LDD;

        out_info->transa = static_cast<int64_t>(P33_TRANSA);
        out_info->transb = static_cast<int64_t>(P33_TRANSB);
        out_info->order_a = static_cast<int64_t>(P33_ORDER);
        out_info->order_b = static_cast<int64_t>(P33_ORDER);
        out_info->order_c = static_cast<int64_t>(P33_ORDER);
        out_info->order_d = static_cast<int64_t>(P33_ORDER);
        out_info->type_a = static_cast<int64_t>(P33_AB_TYPE);
        out_info->type_b = static_cast<int64_t>(P33_AB_TYPE);
        out_info->type_c = static_cast<int64_t>(P33_CD_TYPE);
        out_info->type_d = static_cast<int64_t>(P33_CD_TYPE);
        out_info->compute_type = static_cast<int64_t>(P33_COMPUTE_TYPE);
        out_info->scale_type = static_cast<int64_t>(P33_SCALE_TYPE);
        out_info->pointer_mode = static_cast<int64_t>(P33_POINTER_MODE);
        out_info->epilogue = static_cast<int64_t>(P33_EPILOGUE);

        out_info->search_mode = static_cast<int64_t>(P33_SEARCH_MODE);
        out_info->workspace_limit_bytes = static_cast<int64_t>(P33_WORKSPACE_LIMIT_BYTES);
        out_info->workspace_bytes = static_cast<int64_t>(plan->workspace_bytes);
        out_info->workspace_is_null = (plan->workspace == nullptr) ? 1 : 0;

        out_info->heuristic_requested = P33_HEURISTIC_REQUESTED;
        out_info->heuristic_returned = returned;
        out_info->heuristic_index = selected;

        out_info->alignment_a_bytes = static_cast<int64_t>(alignment_a);
        out_info->alignment_b_bytes = static_cast<int64_t>(alignment_b);
        out_info->alignment_c_bytes = static_cast<int64_t>(alignment_c);
        out_info->alignment_d_bytes = static_cast<int64_t>(alignment_d);

        out_info->algo_id = algo_id;
        out_info->tile_id = tile_id;
        out_info->stages_id = stages_id;
        out_info->split_k = split_k;
        out_info->reduction_scheme = reduction_scheme;
        out_info->cta_swizzling = cta_swizzling;
        out_info->custom_option = custom_option;
        out_info->inner_shape_id = inner_shape_id;
        out_info->cluster_shape_id = cluster_shape_id;

        out_info->waves_count = static_cast<double>(checked.wavesCount);
        out_info->alpha = static_cast<double>(P33_ALPHA);
        out_info->beta = static_cast<double>(P33_BETA);

        *out_plan = plan;
        return 0;
    } catch (...) {
        p33_set_error("p33_plan_create: an unexpected C++ exception was caught at the C boundary");
        p33_plan_release(plan);
        return 1;
    }
}

// Launches exactly one cublasLtMatmul with the selected algorithm on the
// plan's stream. This is the measured path; there is no other one, and no
// fallback to cublasGemmEx, cublasGemmStridedBatchedEx, an ordinary cuBLAS
// GEMM, or a framework operator exists anywhere in this translation unit.
int p33_plan_execute(P33Plan* plan) {
    p33_clear_error();
    if (plan == nullptr) {
        p33_set_error("p33_plan_execute: plan must not be null");
        return 1;
    }
    try {
        const float alpha = P33_ALPHA;
        const float beta = P33_BETA;
        const cublasStatus_t status = cublasLtMatmul(plan->handle,
                                                     plan->operation,
                                                     &alpha,
                                                     plan->a,
                                                     plan->layout_a,
                                                     plan->b,
                                                     plan->layout_b,
                                                     &beta,
                                                     plan->c,
                                                     plan->layout_c,
                                                     plan->d,
                                                     plan->layout_d,
                                                     &plan->algo,
                                                     plan->workspace,
                                                     plan->workspace_bytes,
                                                     plan->stream);
        if (status != CUBLAS_STATUS_SUCCESS) {
            p33_set_error_status("cublasLtMatmul", status);
            return 1;
        }
        return 0;
    } catch (...) {
        p33_set_error("p33_plan_execute: an unexpected C++ exception was caught at the C boundary");
        return 1;
    }
}

// Synchronizes the plan's stream. Kept here so the wrapper's host-clock
// timings bound completed GPU work rather than an asynchronous launch.
int p33_stream_synchronize(P33Plan* plan) {
    p33_clear_error();
    if (plan == nullptr) {
        p33_set_error("p33_stream_synchronize: plan must not be null");
        return 1;
    }
    try {
        const cudaError_t status = cudaStreamSynchronize(plan->stream);
        if (status != cudaSuccess) {
            p33_set_error_cuda("cudaStreamSynchronize", status);
            return 1;
        }
        return 0;
    } catch (...) {
        p33_set_error("p33_stream_synchronize: an unexpected C++ exception was caught");
        return 1;
    }
}

int p33_plan_destroy(P33Plan* plan) {
    p33_clear_error();
    try {
        p33_plan_release(plan);
        return 0;
    } catch (...) {
        p33_set_error("p33_plan_destroy: an unexpected C++ exception was caught at the C boundary");
        return 1;
    }
}

}  // extern "C"
