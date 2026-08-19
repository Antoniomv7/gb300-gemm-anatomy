#!/usr/bin/env python3
"""GPU-free contract checker for the P3.5 five-shape comparison wrapper.

This checker is deliberately independent of ``src/gemm/gemm_comparison.py``: it
carries its own copy of the five-shape table, its own copy of the
four-candidate table, its own copy of the frozen 100-field ``p35.v1`` CSV
schema, its own copy of every frozen categorical value, and its own
implementation of every comparison formula. It never imports a P3.5 constant as
its expected truth. A drift in either the wrapper or the checker therefore shows
up as a disagreement rather than as two copies of the same mistake.

It uses only the Python standard library and never initializes CUDA. The only
subprocesses it starts are Python interpreters running the wrapper behind an
import guard that makes any attempt to import PyTorch, CuTe DSL, the CUDA
bindings, ctypes, or either upstream example a hard failure - which is how
"``--help`` and ``--self-test`` are GPU-free" is proved rather than assumed.

What it validates:

* the five-shape table and its order, and that no other shape can be injected;
* the four-candidate table and its order, and that no fifth candidate exists;
* the exact ``p35.v1`` schema and field order;
* every frozen categorical value, and every frozen cuBLASLt policy value;
* method-specific applicable / ``not_applicable`` fields;
* the comparison formulas, re-derived independently from the serialized rows;
* the ranking rule and its frozen-order tie break;
* the best-CuTe-DSL selection and the single ``is_best_cutedsl`` row per shape;
* exact twenty-row completeness in shape-major order;
* identical provenance across the whole run;
* correctness-before-timing, and that a row can exist only for a passed check;
* ``publishable=false`` everywhere;
* all-or-nothing stdout: a synthetic failure at any candidate position of an
  early, a middle, and the final shape produces no CSV at all; a cleanup failure
  after all twenty rows are prepared also produces no CSV; and a native write
  to descriptor 1 during the measurement cannot contaminate it;
* the absence of forbidden CLI controls;
* the absence of alternative algorithms and of any fallback GEMM API in the
  P3.5 bridge, plus its single ``cublasLtMatmul`` call site and checked release
  status for every native resource;
* the unchanged P3.1-P3.4 contracts, files, targets, schemas, and status.

Usage:
  check_gemm_comparison_p35.py [repository-root]
  check_gemm_comparison_p35.py --self-test

Exit code: 0 only when the selected validation passes, 1 on a contract or
synthetic-test failure, and 2 on a usage error.
"""

import ast
import csv
import io
import math
import re
import subprocess
import sys
from pathlib import Path

# --- Independent frozen expectations ----------------------------------------

WRAPPER_RELATIVE_PATH = "src/gemm/gemm_comparison.py"
BRIDGE_RELATIVE_PATH = "src/gemm/cublaslt_bridge_p35.cu"
CHECKER_RELATIVE_PATH = "scripts/check_gemm_comparison_p35.py"
PROTOCOL_RELATIVE_PATH = "src/gemm/P3_5_PROTOCOL.md"
GLOBAL_CONTRACT_FILE = "VERSIONS.env"
PHASE3_CONTRACT_FILE = "PHASE3_VERSIONS.env"

# The closed units this one must not weaken.
P32_WRAPPER_RELATIVE_PATH = "src/gemm/cutedsl_gemm.py"
P33_WRAPPER_RELATIVE_PATH = "src/gemm/cublaslt_gemm.py"
P33_BRIDGE_RELATIVE_PATH = "src/gemm/cublaslt_bridge.cu"
P34_WRAPPER_RELATIVE_PATH = "src/gemm/cutedsl_variants.py"
P1_AGGREGATOR_RELATIVE_PATH = "scripts/aggregate_exp01_memory_paths.py"
P2_AGGREGATOR_RELATIVE_PATH = "scripts/aggregate_exp02_umma_throughput.py"

UPSTREAM_CHECKOUT_DIR = Path("/opt/cutlass")

# The five and only five (M,N,K,L) shapes, restated independently and in the
# order P3.5 must execute them.
EXPECTED_SHAPES = (
    (4096, 4096, 4096, 1),
    (8192, 8192, 8192, 1),
    (16384, 512, 4096, 1),
    (32768, 512, 4096, 1),
    (512, 16384, 4096, 1),
)
EXPECTED_SHAPE_COUNT = 5
EXPECTED_SHAPE_IDS = tuple(f"{m}x{n}x{k}x{l}" for (m, n, k, l) in EXPECTED_SHAPES)

# The four and only four candidates, in the frozen order.
# variant -> (method, scheduler, tiler, cluster, use_2cta, source, persistent)
EXPECTED_CANDIDATE_TABLE = {
    "nonpersistent_1cta": (
        "cutedsl", "nonpersistent", (128, 128), (1, 1), False, "nonpersistent", False,
    ),
    "persistent_1cta": (
        "cutedsl", "static_persistent", (128, 128), (1, 1), False, "persistent", True,
    ),
    "persistent_2cta": (
        "cutedsl", "static_persistent", (256, 128), (2, 1), True, "persistent", True,
    ),
    "heuristic_first_supported": (
        "cublaslt", None, None, None, None, None, False,
    ),
}
EXPECTED_CANDIDATE_ORDER = (
    "nonpersistent_1cta",
    "persistent_1cta",
    "persistent_2cta",
    "heuristic_first_supported",
)
EXPECTED_CANDIDATE_COUNT = 4
EXPECTED_CUBLASLT_INDEX = 3
EXPECTED_CUTEDSL_VARIANTS = ("nonpersistent_1cta", "persistent_1cta", "persistent_2cta")
EXPECTED_ROW_COUNT = EXPECTED_SHAPE_COUNT * EXPECTED_CANDIDATE_COUNT  # 20
EXPECTED_LINE_COUNT = EXPECTED_ROW_COUNT + 1  # 21

EXPECTED_NOT_APPLICABLE = "not_applicable"

# The frozen, ordered 100-field p35.v1 schema, restated independently.
EXPECTED_CSV_FIELDS = (
    "schema_version",
    "experiment",
    "unit",
    "run_kind",
    "shape_index",
    "shape_id",
    "candidate_index",
    "method",
    "variant",
    "m",
    "n",
    "k",
    "l",
    "ab_dtype",
    "acc_dtype",
    "c_dtype",
    "a_major",
    "b_major",
    "c_major",
    "scheduler",
    "mma_tiler_m",
    "mma_tiler_n",
    "cluster_m",
    "cluster_n",
    "use_2cta_instrs",
    "use_tma_store",
    "max_active_clusters",
    "order_a",
    "order_b",
    "order_c",
    "order_d",
    "transa",
    "transb",
    "lda",
    "ldb",
    "ldc",
    "ldd",
    "compute_type",
    "scale_type",
    "pointer_mode",
    "epilogue",
    "alpha",
    "beta",
    "search_mode",
    "workspace_limit_bytes",
    "workspace_bytes",
    "alignment_a_bytes",
    "alignment_b_bytes",
    "alignment_c_bytes",
    "alignment_d_bytes",
    "heuristic_requested",
    "heuristic_returned",
    "heuristic_index",
    "algo_id",
    "tile_id",
    "stages_id",
    "split_k",
    "reduction_scheme",
    "cta_swizzling",
    "custom_option",
    "inner_shape_id",
    "cluster_shape_id",
    "waves_count",
    "cublaslt_version",
    "seed",
    "reference",
    "atol",
    "rtol",
    "correctness",
    "max_abs_error",
    "max_rel_error",
    "compile_time_ms",
    "setup_time_ms",
    "first_launch_ms",
    "kernel_time_ms",
    "warmup_iterations",
    "iterations",
    "cache_mode",
    "flop_count",
    "tflops",
    "throughput_ratio_vs_cublaslt",
    "gap_to_cublaslt_pct",
    "rank_within_shape",
    "best_cutedsl_variant",
    "is_best_cutedsl",
    "gpu_name",
    "gpu_uuid",
    "compute_capability",
    "driver_version",
    "cuda_toolkit_version",
    "torch_cuda_version",
    "cutedsl_version",
    "cutlass_commit",
    "operand_factory_sha256",
    "upstream_kernel_file",
    "upstream_kernel_git_blob",
    "upstream_kernel_sha256",
    "git_commit",
    "git_dirty",
    "publishable",
)
EXPECTED_FIELD_COUNT = 100

# The schemas P3.5 must neither reuse nor reinterpret.
CLOSED_SCHEMA_VERSIONS = ("p32.v1", "p33.v1", "p34.v1")

# Fields that carry a real value only on a CuTe DSL row.
EXPECTED_CUTEDSL_ONLY_FIELDS = (
    "scheduler", "mma_tiler_m", "mma_tiler_n", "cluster_m", "cluster_n",
    "use_2cta_instrs", "use_tma_store", "max_active_clusters", "compile_time_ms",
    "upstream_kernel_file", "upstream_kernel_git_blob", "upstream_kernel_sha256",
)
# Fields that carry a real value only on the cuBLASLt row.
EXPECTED_CUBLASLT_ONLY_FIELDS = (
    "order_a", "order_b", "order_c", "order_d", "transa", "transb",
    "lda", "ldb", "ldc", "ldd", "compute_type", "scale_type", "pointer_mode",
    "epilogue", "alpha", "beta", "search_mode", "workspace_limit_bytes",
    "workspace_bytes", "alignment_a_bytes", "alignment_b_bytes", "alignment_c_bytes",
    "alignment_d_bytes", "heuristic_requested", "heuristic_returned", "heuristic_index",
    "algo_id", "tile_id", "stages_id", "split_k", "reduction_scheme", "cta_swizzling",
    "custom_option", "inner_shape_id", "cluster_shape_id", "waves_count",
    "cublaslt_version", "setup_time_ms",
)

# Values that must be identical in all twenty rows.
EXPECTED_FIXED_ROW_VALUES = {
    "schema_version": "p35.v1",
    "experiment": "exp03_cutedsl_vs_cublaslt",
    "unit": "P3.5",
    "run_kind": "smoke",
    "l": "1",
    "ab_dtype": "BFloat16",
    "acc_dtype": "Float32",
    "c_dtype": "Float32",
    "a_major": "k",
    "b_major": "k",
    "c_major": "n",
    "seed": "1111",
    "reference": "torch_cuda_fp32_ieee",
    "correctness": "PASS",
    "atol": "0.100000000",
    "rtol": "0.000010000",
    "cache_mode": "hot",
    "publishable": "false",
}

# The frozen cuBLASLt descriptor and algorithm policy, restated independently.
EXPECTED_CUBLASLT_POLICY = {
    "order_a": "CUBLASLT_ORDER_ROW",
    "order_b": "CUBLASLT_ORDER_ROW",
    "order_c": "CUBLASLT_ORDER_ROW",
    "order_d": "CUBLASLT_ORDER_ROW",
    "transa": "CUBLAS_OP_N",
    "transb": "CUBLAS_OP_T",
    "compute_type": "CUBLAS_COMPUTE_32F",
    "scale_type": "CUDA_R_32F",
    "pointer_mode": "CUBLASLT_POINTER_MODE_HOST",
    "epilogue": "CUBLASLT_EPILOGUE_DEFAULT",
    "search_mode": "CUBLASLT_SEARCH_BEST_FIT",
    "workspace_limit_bytes": "67108864",
    "heuristic_requested": "32",
    "alpha": "1.000000000",
    "beta": "0.000000000",
}
EXPECTED_WORKSPACE_LIMIT_BYTES = 67108864
EXPECTED_HEURISTIC_REQUESTED = 32

EXPECTED_CUDA_ARCH = "sm_103a"
EXPECTED_COMPUTE_CAPABILITY = "10.3"

# The FLOP factor: one multiplication plus one addition per multiply-accumulate.
EXPECTED_FLOPS_PER_MAC = 2

ERROR_FIELDS = ("max_abs_error", "max_rel_error")
TOLERANCE_FIELDS = ("atol", "rtol")
COUNT_FIELDS = ("warmup_iterations", "iterations")
TIMING_DECIMALS = 6
ERROR_DECIMALS = 9
TOLERANCE_DECIMALS = 9
SCALAR_DECIMALS = 9
WAVES_DECIMALS = 6
TFLOPS_DECIMALS = 6
RATIO_DECIMALS = 9
GAP_DECIMALS = 6

# Tolerances used only when a serialized decimal is re-derived from other
# serialized decimals; they absorb the deterministic fixed-point rounding and
# are far tighter than any real formula error could be.
CHECK_RTOL = 1e-4
CHECK_ATOL = 1e-6

MIN_ITERATIONS = 1
MAX_WARMUP_ITERATIONS = 100
MAX_ITERATIONS = 100
SMOKE_WARMUP_ITERATIONS = 2
SMOKE_ITERATIONS = 10

ALLOWED_CLI_OPTIONS = frozenset(
    {"--help", "--self-test", "--warmup-iterations", "--iterations"}
)

# Option spellings that would reopen a frozen scientific parameter, a frozen
# shape, a frozen candidate, the algorithm policy, or the output contract.
FORBIDDEN_CLI_FRAGMENTS = (
    "mnkl", "shape", "dtype", "major", "tiler", "cluster", "tma", "persistent",
    "2cta", "cta-group", "cta_group", "scheduler", "variant", "method", "candidate",
    "seed", "tolerance", "atol", "rtol", "skip-ref", "skip_ref", "cold-l2", "cold_l2",
    "ref-check", "ref_check", "source", "path", "example", "publish", "cache",
    "gpu", "device", "autotune", "algo", "workspace", "heuristic", "order", "trans",
    "lda", "ldb", "ldc", "ldd", "leading", "alpha", "beta", "epilogue", "search",
    "output", "out-file", "out_file", "csv", "partial", "only", "filter", "select",
    "input", "config",
)

# Identifier fragments that must not appear as code in the wrapper. Prose in
# docstrings and comments is exempt: the scan runs over Python NAME tokens
# only, so a sentence explaining that P3.5 computes no confidence interval is
# fine, while a confidence_interval variable is not. P3.5 legitimately owns
# tflops, a ratio, a gap, and a rank, so those are NOT banned here - but every
# Phase 4 statistic and every roofline/attribution quantity is.
FORBIDDEN_SOURCE_IDENTIFIERS = (
    "confidence", "pvalue", "p_value", "bootstrap", "outlier", "roofline",
    "arithmetic_intensity", "bandwidth", "utilization", "occupancy_pct",
    "empirical_ceiling", "nsight", "ncu", "sass", "autotune", "campaign",
    "skip_ref_check", "use_cold_l2", "matplotlib", "pyplot", "dashboard",
)

# Torch/framework operators that must never replace a measured candidate. The
# untimed FP32 oracle deliberately uses torch.einsum.
FORBIDDEN_TORCH_MATMUL_ATTRS = frozenset(
    {"matmul", "mm", "bmm", "addmm", "baddbmm", "addbmm", "linear"}
)

# The upstream helpers P3.5 must never use, because they fuse compilation, the
# first launch, correctness, and benchmarking into one number.
FORBIDDEN_UPSTREAM_HELPERS = ("run", "benchmark", "compile_bmm", "prepare_tensors")

GPU_STACK_MODULES = ("torch", "cutlass", "cuda", "numpy", "pynvml", "ctypes")

# The pinned keys P3.5 reuses. P3.5 introduces none of its own.
REQUIRED_P31_CONTRACT_KEYS = (
    "CUTEDSL_P31_EXAMPLE_PATH",
    "CUTEDSL_P31_EXAMPLE_GIT_BLOB",
    "CUTEDSL_P31_EXAMPLE_SHA256",
)
REQUIRED_P34_CONTRACT_KEYS = (
    "CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH",
    "CUTEDSL_P34_PERSISTENT_EXAMPLE_GIT_BLOB",
    "CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256",
)

# --- The P3.5 bridge ---------------------------------------------------------

REQUIRED_BRIDGE_CALL = "cublasLtMatmul"
REQUIRED_BRIDGE_CALLS = (
    "cublasLtCreate",
    "cublasLtMatmulDescCreate",
    "cublasLtMatrixLayoutCreate",
    "cublasLtMatmulPreferenceCreate",
    "cublasLtMatmulAlgoGetHeuristic",
    "cublasLtMatmulAlgoCheck",
    "cublasLtMatmulAlgoConfigGetAttribute",
    "cublasLtGetVersion",
    "cublasLtMatmul",
    "cublasLtDestroy",
)
REQUIRED_BRIDGE_RELEASE_CALLS = (
    "cudaFree",
    "cublasLtMatrixLayoutDestroy",
    "cublasLtMatmulDescDestroy",
    "cublasLtMatmulPreferenceDestroy",
    "cublasLtDestroy",
)
# Any of these in the measured translation unit means a fallback GEMM API or an
# alternative algorithm-enumeration path exists.
FORBIDDEN_GEMM_ENTRY_POINTS = (
    "cublasGemmEx",
    "cublasGemmStridedBatchedEx",
    "cublasGemmBatchedEx",
    "cublasSgemm",
    "cublasHgemm",
    "cublasDgemm",
    "cublasLtMatmulAlgoGetIds",
    "cublasLtMatmulAlgoInit",
    "cublasLtMatmulAlgoCapGetAttribute",
)
FORBIDDEN_BRIDGE_TIMING = (
    "cudaEventRecord",
    "cudaEventElapsedTime",
    "std::chrono",
    "clock_gettime",
    "gettimeofday",
    "nvtxRange",
)
FORBIDDEN_BRIDGE_OUTPUT = (
    r"(?<![a-zA-Z0-9_])printf\s*\(",
    r"(?<![a-zA-Z0-9_])puts\s*\(",
    r"(?<![a-zA-Z0-9_])fputs\s*\(",
    r"std::cout",
    r"std::cerr",
    r"std::clog",
    r"fprintf\s*\(\s*std(out|err)",
)
REQUIRED_BRIDGE_EXPORTS = (
    "p35_bridge_abi_version",
    "p35_plan_info_size",
    "p35_last_error",
    "p35_cublaslt_version",
    "p35_shape_count",
    "p35_shape_at",
    "p35_plan_create",
    "p35_plan_execute",
    "p35_stream_synchronize",
    "p35_plan_destroy",
)
REQUIRED_ALGO_CONFIG_ATTRIBUTES = (
    "CUBLASLT_ALGO_CONFIG_ID",
    "CUBLASLT_ALGO_CONFIG_TILE_ID",
    "CUBLASLT_ALGO_CONFIG_STAGES_ID",
    "CUBLASLT_ALGO_CONFIG_SPLITK_NUM",
    "CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME",
    "CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING",
    "CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION",
    "CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID",
    "CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID",
)
REQUIRED_ALIGNMENT_PREFERENCES = (
    "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES",
    "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES",
    "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES",
    "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES",
)
# The frozen `static const` declarations the bridge must carry, independently
# of whatever the Python wrapper says.
EXPECTED_BRIDGE_CONSTANTS = {
    "P35_BATCH_COUNT": "1",
    "P35_TRANSA": "CUBLAS_OP_N",
    "P35_TRANSB": "CUBLAS_OP_T",
    "P35_ORDER": "CUBLASLT_ORDER_ROW",
    "P35_AB_TYPE": "CUDA_R_16BF",
    "P35_CD_TYPE": "CUDA_R_32F",
    "P35_COMPUTE_TYPE": "CUBLAS_COMPUTE_32F",
    "P35_SCALE_TYPE": "CUDA_R_32F",
    "P35_POINTER_MODE": "CUBLASLT_POINTER_MODE_HOST",
    "P35_EPILOGUE": "CUBLASLT_EPILOGUE_DEFAULT",
    "P35_ALPHA": "1.0f",
    "P35_BETA": "0.0f",
    "P35_WORKSPACE_LIMIT_BYTES": "67108864ULL",
    "P35_HEURISTIC_REQUESTED": "32",
    "P35_SEARCH_MODE": "CUBLASLT_SEARCH_BEST_FIT",
    "P35_MAX_ALIGNMENT_BYTES": "256u",
}
# Overflow-safety helpers the bridge must genuinely carry.
REQUIRED_BRIDGE_OVERFLOW_MARKERS = ("INT64_MAX", "SIZE_MAX")

_RE_HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")
_RE_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_RE_GPU_UUID = re.compile(r"\AGPU-[0-9a-fA-F][0-9a-fA-F-]+\Z")
_RE_DOTTED_VERSION = re.compile(r"\A[0-9]+(\.[0-9]+)*\Z")
_RE_COMPUTE_CAPABILITY = re.compile(r"\A[0-9]+\.[0-9]+\Z")
_RE_POSITIVE_INT = re.compile(r"\A[1-9][0-9]*\Z")
_RE_NONNEGATIVE_INT = re.compile(r"\A(0|[1-9][0-9]*)\Z")
_RE_ENV_LINE = re.compile(r"\A([A-Z][A-Z0-9_]*)=(\S*)\Z")
_RE_SAFE_TEXT = re.compile(r"\A[^\x00-\x1f\x7f]+\Z")
_RE_UPSTREAM_REL_PATH = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._/-]*\.py\Z")

# P3.5 retains the closed P3.2/P3.3/P3.4 FP32 oracle policy.
FORBIDDEN_FP32_API_SPELLINGS = (
    "allow_tf32", "set_float32_matmul_precision", "get_float32_matmul_precision",
)
REQUIRED_FP32_API_SPELLINGS = ("fp32_precision",)
FP32_PRECISION_IEEE = "ieee"

FORBIDDEN_SMOKE_FILTERS = ("grep -v", "| sed", "| tail", "| head", "| awk", "| grep")

SMOKE_TARGET = "gemm-comparison-p35-smoke"
CHECK_TARGET = "gemm-comparison-p35-check"
CHECK_PREREQUISITE = "gemm-cutedsl-p34-check"
LAUNCHER_RELATIVE_PATH = "scripts/run_container.sh"
LAUNCHER_DATA_MODE_VARIABLE = "RUN_CONTAINER_STDOUT_IS_DATA"
GPU_INDEX_VARIABLE = "BLACKWELL_GPU_INDEX"
SMOKE_SUCCESS_SENTENCE = (
    "P3.5 smoke completed: every candidate of every shape passed correctness before its "
    "warm-up and steady-state timing."
)

# Status assertions that must remain true for the closed units.
CLOSED_STATUS_LINES = (
    "| P3.1 | Pinned official CuTe DSL example | YES | YES | YES |",
    "| P3.2 | One-shape wrapper | YES | YES | YES |",
    "| P3.3 | cuBLASLt baseline | YES | YES | YES |",
    "| P3.4 | Three execution variants | YES | YES | YES |",
)
# P3.5 itself is implemented, independently audited, and GB300-verified.
EXPECTED_P35_STATUS_LINE = "| P3.5 | Five shapes and comparison | YES | YES | YES |"
FORBIDDEN_P35_STATUS_LINES = (
    "| P3.5 | Five shapes and comparison | NO | NO | NO |",
    "| P3.5 | Five shapes and comparison | YES | YES | NO |",
    "| P3.5 | Five shapes and comparison | YES | NO | YES |",
    "| P3.5 | Five shapes and comparison | YES | NO | NO |",
)
# The Phase 4 frontier. P4.1 (the campaign orchestrator) is closed after
# independent audit and GB300 verification. P4.2 (the frozen protocol for one
# accepted pilot plus three final campaigns, and its cross-campaign checker) is
# closed after independent audit and GB300 verification. P4.3 (the offline
# integrated analysis) is implemented, and is neither independently audited nor
# run against the real evidence. This guard still rejects every stale or
# impossible P4.1, P4.2, and P4.3 state -- including any claim that P4.3 has
# been audited or verified. (Before P4.1 landed, this tuple demanded the literal
# "| P4.1 | Orchestrator | NO | NO | NO |", which structurally forbade P4.1
# from ever being implemented -- exactly the stale-frontier situation P3.5
# itself had to correct for P3.4. See src/phase4/P4_1_PROTOCOL.md section 9.1.
# The P4.2 row was advanced the same way, and only that row: see
# src/phase4/P4_2_PROTOCOL.md section 9.1. The P4.3 row was advanced by exactly
# one step for the same reason -- see src/phase4/P4_3_PROTOCOL.md section 6.1 --
# and, on P4.3's acceptance, by exactly one further step to the closed
# "YES | YES | YES": see src/phase4/P4_3_PROTOCOL.md section 17.)
PHASE4_STATUS_LINES = (
    "| P4.1 | Orchestrator | YES | YES | YES |",
    "| P4.2 | Pilot plus three final campaigns | YES | YES | YES |",
    "| P4.3 | Integrated analysis, documentation, audit | YES | YES | YES |",
)
FORBIDDEN_PHASE4_STATUS_LINES = (
    "| P4.1 | Orchestrator | NO | NO | NO |",
    "| P4.1 | Orchestrator | YES | YES | NO |",
    "| P4.1 | Orchestrator | YES | NO | YES |",
    "| P4.1 | Orchestrator | YES | NO | NO |",
    "| P4.2 | Pilot plus three final campaigns | NO | NO | NO |",
    "| P4.2 | Pilot plus three final campaigns | NO | YES | NO |",
    "| P4.2 | Pilot plus three final campaigns | NO | NO | YES |",
    "| P4.2 | Pilot plus three final campaigns | NO | YES | YES |",
    "| P4.2 | Pilot plus three final campaigns | YES | NO | NO |",
    "| P4.2 | Pilot plus three final campaigns | YES | YES | NO |",
    "| P4.2 | Pilot plus three final campaigns | YES | NO | YES |",
    "| P4.3 | Integrated analysis, documentation, audit | NO | NO | NO |",
    "| P4.3 | Integrated analysis, documentation, audit | NO | YES | NO |",
    "| P4.3 | Integrated analysis, documentation, audit | NO | NO | YES |",
    "| P4.3 | Integrated analysis, documentation, audit | NO | YES | YES |",
    "| P4.3 | Integrated analysis, documentation, audit | YES | NO | NO |",
    "| P4.3 | Integrated analysis, documentation, audit | YES | YES | NO |",
    "| P4.3 | Integrated analysis, documentation, audit | YES | NO | YES |",
)

_GUARD_PRELUDE = """
import sys

_BLOCKED = {blocked!r}


class _ImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _BLOCKED:
            raise AssertionError("GPU-free contract violated: import of " + fullname)
        return None


sys.meta_path.insert(0, _ImportGuard())
"""

GPU_FREE_GUARD = (
    _GUARD_PRELUDE
    + """
sys.argv = [{argv0!r}] + {argv!r}
import runpy

runpy.run_path({wrapper!r}, run_name="__main__")
"""
)

# Drives the real main() with execute_measurement replaced by a function that
# fails at exactly one (shape, candidate) position, proving the all-or-nothing
# stdout contract through the real descriptor plumbing.
_PROBE_PRELUDE = """
import importlib.util

spec = importlib.util.spec_from_file_location("p35_probe", {wrapper!r})
module = importlib.util.module_from_spec(spec)
sys.modules["p35_probe"] = module
spec.loader.exec_module(module)
"""

FAILING_POSITION_PROBE = (
    _GUARD_PRELUDE
    + _PROBE_PRELUDE
    + """
_FAIL_SHAPE = {fail_shape!r}
_FAIL_CANDIDATE = {fail_candidate!r}


def _fake_execute(warmup_iterations, iterations):
    rows = []
    for shape_index in range(module.FROZEN_SHAPE_COUNT):
        shape_rows = module._synthetic_shape_rows(shape_index)
        for candidate_index in range(module.FROZEN_CANDIDATE_COUNT):
            if (shape_index, candidate_index) == (_FAIL_SHAPE, _FAIL_CANDIDATE):
                raise module.CorrectnessError(
                    "synthetic failure at shape %d candidate %d"
                    % (shape_index, candidate_index)
                )
            rows.append(shape_rows[candidate_index])
    return module.serialize_rows(rows)


module.execute_measurement = _fake_execute
sys.exit(module.main([]))
"""
)

# The same probe with no injected failure, proving the success path really does
# write exactly twenty-one lines to stdout.
SUCCESS_PATH_PROBE = (
    _GUARD_PRELUDE
    + _PROBE_PRELUDE
    + """
def _fake_execute(warmup_iterations, iterations):
    return module.serialize_rows(module._synthetic_rows())


module.execute_measurement = _fake_execute
sys.exit(module.main([]))
"""
)

# A native-style write straight to descriptor 1 during the measurement must be
# swallowed by the wrapper's descriptor redirection, not appear on stdout.
STDOUT_CONTAMINATION_PROBE = (
    _GUARD_PRELUDE
    + _PROBE_PRELUDE
    + """
import os


def _fake_execute(warmup_iterations, iterations):
    os.write(1, b"CONTAMINATION-FROM-A-NATIVE-LIBRARY\\n")
    print("CONTAMINATION-FROM-PYTHON")
    sys.stdout.flush()
    return module.serialize_rows(module._synthetic_rows())


module.execute_measurement = _fake_execute
sys.exit(module.main([]))
    """
)

# A cleanup error after all twenty synthetic rows have been prepared is still a
# failed run: the real main() must emit neither the buffered header nor any row.
CLEANUP_FAILURE_PROBE = (
    _GUARD_PRELUDE
    + _PROBE_PRELUDE
    + """
def _fake_execute(warmup_iterations, iterations):
    rows = module._synthetic_rows()

    def _fail_cleanup():
        raise module.BridgeError("synthetic cleanup failure after twenty prepared rows")

    module._cleanup_preserving_primary(_fail_cleanup, "synthetic final cleanup")
    return module.serialize_rows(rows)


module.execute_measurement = _fake_execute
sys.exit(module.main([]))
"""
)


# --- Pure validators ---------------------------------------------------------


def validate_csv_schema(fields) -> list:
    """Check the frozen field list, its order, and its absence of Phase 4 metrics."""
    errors = []
    if tuple(fields) != EXPECTED_CSV_FIELDS:
        actual = tuple(fields)
        if set(actual) != set(EXPECTED_CSV_FIELDS):
            missing = sorted(set(EXPECTED_CSV_FIELDS) - set(actual))
            unknown = sorted(set(actual) - set(EXPECTED_CSV_FIELDS))
            if missing:
                errors.append(f"the CSV schema is missing field(s): {', '.join(missing)}")
            if unknown:
                errors.append(f"the CSV schema has unknown field(s): {', '.join(unknown)}")
        else:
            errors.append("the CSV schema has the right fields in the wrong order")
    if len(set(fields)) != len(tuple(fields)):
        errors.append("the CSV schema contains a duplicate field name")
    if len(tuple(fields)) != EXPECTED_FIELD_COUNT and set(fields) == set(EXPECTED_CSV_FIELDS):
        errors.append(f"the CSV schema has {len(tuple(fields))} fields, expected 100")
    for field in fields:
        if re.search(
            r"confidence|p_value|pvalue|bootstrap|outlier|roofline|bandwidth|"
            r"utilization|intensity|significan|ceiling",
            field,
        ):
            errors.append(f"the CSV schema exposes a Phase 4 quantity: {field}")
    return errors


def validate_shape_table(shapes) -> list:
    """Check the five-shape table exactly, including its order."""
    errors = []
    if not isinstance(shapes, (list, tuple)):
        return ["the frozen shape table is not a sequence"]
    if len(shapes) != EXPECTED_SHAPE_COUNT:
        errors.append(f"P3.5 has exactly 5 shapes, the table has {len(shapes)}")
    normalized = tuple(tuple(shape) for shape in shapes)
    if len(set(normalized)) != len(normalized):
        errors.append(f"the shape table contains a duplicate shape: {normalized}")
    if normalized != EXPECTED_SHAPES:
        errors.append(f"the shape table {normalized} is not the frozen {EXPECTED_SHAPES}")
    for shape in normalized:
        if len(shape) != 4:
            errors.append(f"shape {shape} is not an (M,N,K,L) quadruple")
            continue
        if shape[3] != 1:
            errors.append(f"shape {shape} does not have L=1")
        for value in shape:
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                errors.append(f"shape {shape} has a non-positive-integer extent")
                break
    return errors


def validate_candidate_table(candidates) -> list:
    """Check the four-candidate table exactly, including its order."""
    errors = []
    if not isinstance(candidates, (list, tuple)):
        return ["the frozen candidate table is not a sequence"]
    if len(candidates) != EXPECTED_CANDIDATE_COUNT:
        errors.append(f"P3.5 has exactly 4 candidates, the table has {len(candidates)}")
    names = tuple(spec.get("variant") for spec in candidates)
    if len(set(names)) != len(names):
        errors.append(f"the candidate table contains a duplicate candidate: {names}")
    if names != EXPECTED_CANDIDATE_ORDER:
        errors.append(
            f"the candidate order {names} is not the frozen {EXPECTED_CANDIDATE_ORDER}"
        )
    for index, spec in enumerate(candidates):
        name = spec.get("variant")
        if name not in EXPECTED_CANDIDATE_TABLE:
            errors.append(f"{name!r} is not one of the four frozen candidates")
            continue
        method, scheduler, tiler, cluster, use_2cta, source, persistent = (
            EXPECTED_CANDIDATE_TABLE[name]
        )
        if spec.get("method") != method:
            errors.append(f"candidate {name}: method {spec.get('method')!r} != {method!r}")
        if spec.get("persistent") is not persistent:
            errors.append(
                f"candidate {name}: persistent {spec.get('persistent')!r} != {persistent!r}"
            )
        if method == "cublaslt":
            if index != EXPECTED_CUBLASLT_INDEX:
                errors.append(
                    f"the cuBLASLt baseline must be candidate {EXPECTED_CUBLASLT_INDEX + 1}, "
                    f"found at {index + 1}"
                )
            continue
        actual = (
            spec.get("scheduler"),
            tuple(spec.get("mma_tiler_mn") or ()),
            tuple(spec.get("cluster_shape_mn") or ()),
            spec.get("use_2cta_instrs"),
            spec.get("source"),
        )
        expected = (scheduler, tiler, cluster, use_2cta, source)
        if actual != expected:
            errors.append(f"candidate {name}: {actual} != frozen {expected}")
        # The 2-CTA geometry must keep a per-CTA M extent of 128, exactly as
        # the closed P3.4 contract requires.
        actual_tiler = tuple(spec.get("mma_tiler_mn") or ())
        actual_cluster = tuple(spec.get("cluster_shape_mn") or ())
        if spec.get("use_2cta_instrs"):
            if len(actual_tiler) != 2 or len(actual_cluster) != 2 or actual_cluster[0] != 2:
                errors.append(f"candidate {name}: a 2-CTA candidate needs a cluster M of 2")
            elif actual_tiler[0] // actual_cluster[0] != 128:
                errors.append(
                    f"candidate {name}: tiler M {actual_tiler[0]} over cluster M "
                    f"{actual_cluster[0]} gives a per-CTA M extent of "
                    f"{actual_tiler[0] // actual_cluster[0]}, not 128"
                )
        if spec.get("persistent") and spec.get("upstream_class") != "PersistentDenseGemmKernel":
            errors.append(
                f"candidate {name}: a persistent candidate is routed to "
                f"{spec.get('upstream_class')!r}"
            )
        if not spec.get("persistent") and spec.get("upstream_class") != "DenseGemmKernel":
            errors.append(
                f"candidate {name}: a non-persistent candidate is routed to "
                f"{spec.get('upstream_class')!r}"
            )
    cublaslt_rows = [spec for spec in candidates if spec.get("method") == "cublaslt"]
    if len(cublaslt_rows) != 1:
        errors.append(
            f"exactly one candidate must be the cuBLASLt baseline, found {len(cublaslt_rows)}"
        )
    return errors


def validate_frozen_config(config) -> list:
    """Check the wrapper's frozen configuration mapping against this checker's."""
    errors = []
    if not isinstance(config, dict):
        return ["the frozen configuration is not a mapping"]
    expected = {
        "schema_version": "p35.v1",
        "experiment": "exp03_cutedsl_vs_cublaslt",
        "unit": "P3.5",
        "run_kind": "smoke",
        "shapes": EXPECTED_SHAPES,
        "ab_dtype": "BFloat16",
        "acc_dtype": "Float32",
        "c_dtype": "Float32",
        "a_major": "k",
        "b_major": "k",
        "c_major": "n",
        "use_tma_store": True,
        "seed": 1111,
        "reference": "torch_cuda_fp32_ieee",
        "atol": 1e-1,
        "rtol": 1e-5,
        "cache_mode": "hot",
        "workspace_limit_bytes": EXPECTED_WORKSPACE_LIMIT_BYTES,
        "heuristic_requested": EXPECTED_HEURISTIC_REQUESTED,
        "search_mode": "CUBLASLT_SEARCH_BEST_FIT",
        "flops_per_mac": EXPECTED_FLOPS_PER_MAC,
        "publishable": False,
    }
    for key, want in sorted(expected.items()):
        if key not in config:
            errors.append(f"the frozen configuration is missing {key}")
            continue
        actual = config[key]
        if isinstance(want, bool):
            if actual is not want:
                errors.append(f"frozen {key} is {actual!r}, expected {want!r}")
        elif isinstance(want, float):
            if not isinstance(actual, float) or actual != want:
                errors.append(f"frozen {key} is {actual!r}, expected {want!r}")
        elif isinstance(want, tuple):
            if tuple(tuple(entry) for entry in (actual or ())) != want:
                errors.append(f"frozen {key} is {actual!r}, expected {want!r}")
        elif actual != want:
            errors.append(f"frozen {key} is {actual!r}, expected {want!r}")
    unknown = sorted(set(config) - set(expected))
    if unknown:
        errors.append(f"the frozen configuration has unexpected key(s): {', '.join(unknown)}")
    if config.get("schema_version") in CLOSED_SCHEMA_VERSIONS:
        errors.append(f"P3.5 reuses a closed schema version: {config.get('schema_version')}")
    return errors


# --- This checker's independent comparison arithmetic ------------------------


def expected_flop_count(mnkl) -> int:
    """2 x M x N x K, as an exact integer, restated independently."""
    m, n, k, _l = mnkl
    return EXPECTED_FLOPS_PER_MAC * m * n * k


def expected_tflops(flop_count: int, kernel_time_ms: float) -> float:
    """flop_count / (kernel_time_ms * 1e9), restated independently."""
    return flop_count / (kernel_time_ms * 1e9)


def expected_ranking(kernel_times):
    """Ascending kernel time, exact ties broken by the frozen candidate order."""
    order = sorted(range(len(kernel_times)), key=lambda index: (kernel_times[index], index))
    ranks = [0] * len(kernel_times)
    for position, index in enumerate(order):
        ranks[index] = position + 1
    return ranks


def expected_best_cutedsl_index(kernel_times) -> int:
    """The fastest of the three CuTe DSL candidates, frozen order on a tie."""
    cutedsl_indices = tuple(
        index for index, name in enumerate(EXPECTED_CANDIDATE_ORDER)
        if EXPECTED_CANDIDATE_TABLE[name][0] == "cutedsl"
    )
    return min(cutedsl_indices, key=lambda index: (kernel_times[index], index))


def _close(actual: float, expected: float) -> bool:
    if not math.isfinite(actual) or not math.isfinite(expected):
        return False
    return abs(actual - expected) <= CHECK_ATOL + CHECK_RTOL * abs(expected)


def _validate_decimal_field(field, text, decimals, strictly_positive, errors) -> None:
    if not re.fullmatch(rf"(0|[1-9][0-9]*)\.[0-9]{{{decimals}}}", text):
        errors.append(
            f"{field}: {text!r} is not a fixed-point decimal with {decimals} fractional digits"
        )
        return
    value = float(text)
    if not math.isfinite(value):
        errors.append(f"{field}: {text!r} is not finite")
    elif strictly_positive and value <= 0.0:
        errors.append(f"{field}: {text!r} must be strictly positive")


def _validate_signed_decimal_field(field, text, decimals, errors) -> None:
    if not re.fullmatch(rf"-?(0|[1-9][0-9]*)\.[0-9]{{{decimals}}}", text):
        errors.append(
            f"{field}: {text!r} is not a signed fixed-point decimal with {decimals} "
            "fractional digits"
        )
        return
    value = float(text)
    if not math.isfinite(value):
        errors.append(f"{field}: {text!r} is not finite")
    elif text.startswith("-") and value == 0.0:
        errors.append(f"{field}: {text!r} is a negative zero")


def validate_row_mapping(row) -> list:
    """This checker's own, independent validator for one serialized row."""
    errors = []
    if not isinstance(row, dict):
        return ["a CSV row must be a mapping"]

    missing = sorted(set(EXPECTED_CSV_FIELDS) - set(row))
    unknown = sorted(set(row) - set(EXPECTED_CSV_FIELDS))
    if missing:
        errors.append(f"missing field(s): {', '.join(missing)}")
    if unknown:
        errors.append(f"unknown field(s): {', '.join(unknown)}")
    if missing or unknown:
        return errors

    for field in EXPECTED_CSV_FIELDS:
        value = row[field]
        if not isinstance(value, str):
            errors.append(f"{field}: {value!r} is not a string")
            return errors
        if value == "" or not _RE_SAFE_TEXT.match(value):
            errors.append(f"{field}: {value!r} is empty or contains control characters")

    for field, want in sorted(EXPECTED_FIXED_ROW_VALUES.items()):
        if row[field] != want:
            errors.append(f"{field}: {row[field]!r} != frozen {want!r}")

    # --- the shape must be one of the five, and self-consistent --------------
    if not _RE_POSITIVE_INT.match(row["shape_index"]):
        errors.append(f"shape_index: {row['shape_index']!r} is not a positive integer")
        return errors
    shape_index = int(row["shape_index"]) - 1
    if not 0 <= shape_index < EXPECTED_SHAPE_COUNT:
        errors.append(
            f"shape_index: {row['shape_index']} is outside 1..{EXPECTED_SHAPE_COUNT}; P3.5 "
            "runs exactly the five frozen shapes"
        )
        return errors
    mnkl = EXPECTED_SHAPES[shape_index]
    m, n, k, l = mnkl
    for field, want in (
        ("m", str(m)), ("n", str(n)), ("k", str(k)), ("l", str(l)),
        ("shape_id", EXPECTED_SHAPE_IDS[shape_index]),
    ):
        if row[field] != want:
            errors.append(
                f"shape {row['shape_index']}: {field}={row[field]!r} != frozen {want!r}; an "
                "arbitrary shape can never be emitted"
            )

    # --- the candidate must be one of the four, and self-consistent ----------
    if not _RE_POSITIVE_INT.match(row["candidate_index"]):
        errors.append(f"candidate_index: {row['candidate_index']!r} is not a positive integer")
        return errors
    candidate_index = int(row["candidate_index"]) - 1
    if not 0 <= candidate_index < EXPECTED_CANDIDATE_COUNT:
        errors.append(
            f"candidate_index: {row['candidate_index']} is outside "
            f"1..{EXPECTED_CANDIDATE_COUNT}"
        )
        return errors
    variant = EXPECTED_CANDIDATE_ORDER[candidate_index]
    if row["variant"] != variant:
        errors.append(
            f"candidate {row['candidate_index']}: variant={row['variant']!r} != frozen "
            f"{variant!r}"
        )
        return errors
    method, scheduler, tiler, cluster, use_2cta, _source, persistent = (
        EXPECTED_CANDIDATE_TABLE[variant]
    )
    if row["method"] != method:
        errors.append(f"{variant}: method={row['method']!r} != frozen {method!r}")
        return errors

    # --- method-specific applicability ---------------------------------------
    inapplicable = (
        EXPECTED_CUBLASLT_ONLY_FIELDS if method == "cutedsl" else EXPECTED_CUTEDSL_ONLY_FIELDS
    )
    applicable = (
        EXPECTED_CUTEDSL_ONLY_FIELDS if method == "cutedsl" else EXPECTED_CUBLASLT_ONLY_FIELDS
    )
    for field in inapplicable:
        if row[field] != EXPECTED_NOT_APPLICABLE:
            errors.append(
                f"{variant}: {field}={row[field]!r} must be {EXPECTED_NOT_APPLICABLE!r} for "
                f"method={method}"
            )
    for field in applicable:
        if field == "max_active_clusters":
            continue
        if row[field] == EXPECTED_NOT_APPLICABLE:
            errors.append(
                f"{variant}: {field} must carry a real value for method={method}"
            )

    if method == "cutedsl":
        errors.extend(_validate_cutedsl_row(row, variant, scheduler, tiler, cluster,
                                            use_2cta, persistent))
    else:
        errors.extend(_validate_cublaslt_row(row, mnkl))

    # --- shared numeric discipline -------------------------------------------
    for field in ("git_dirty", "publishable", "is_best_cutedsl"):
        if row[field] not in ("true", "false"):
            errors.append(f"{field}: {row[field]!r} is not a canonical lowercase boolean")
    if method == "cublaslt" and row["is_best_cutedsl"] != "false":
        errors.append("the cuBLASLt baseline row can never be the best CuTe DSL variant")

    for field in COUNT_FIELDS:
        if not _RE_POSITIVE_INT.match(row[field]):
            errors.append(f"{field}: {row[field]!r} is not a positive integer")
        else:
            maximum = MAX_WARMUP_ITERATIONS if field == "warmup_iterations" else MAX_ITERATIONS
            if not MIN_ITERATIONS <= int(row[field]) <= maximum:
                errors.append(f"{field}: {row[field]} is outside [{MIN_ITERATIONS}, {maximum}]")

    for field in ("first_launch_ms", "kernel_time_ms"):
        _validate_decimal_field(field, row[field], TIMING_DECIMALS, True, errors)
    for field in ERROR_FIELDS:
        _validate_decimal_field(field, row[field], ERROR_DECIMALS, False, errors)
    for field in TOLERANCE_FIELDS:
        _validate_decimal_field(field, row[field], TOLERANCE_DECIMALS, True, errors)

    # --- the comparison fields, re-derived from this row alone ---------------
    if not _RE_POSITIVE_INT.match(row["flop_count"]):
        errors.append(f"flop_count: {row['flop_count']!r} is not a positive integer")
    elif int(row["flop_count"]) != expected_flop_count(mnkl):
        errors.append(
            f"flop_count: {row['flop_count']} != the exact 2*M*N*K value "
            f"{expected_flop_count(mnkl)} for {EXPECTED_SHAPE_IDS[shape_index]}"
        )
    _validate_decimal_field("tflops", row["tflops"], TFLOPS_DECIMALS, True, errors)
    _validate_decimal_field(
        "throughput_ratio_vs_cublaslt", row["throughput_ratio_vs_cublaslt"],
        RATIO_DECIMALS, True, errors,
    )
    _validate_signed_decimal_field(
        "gap_to_cublaslt_pct", row["gap_to_cublaslt_pct"], GAP_DECIMALS, errors
    )
    if not _RE_POSITIVE_INT.match(row["rank_within_shape"]):
        errors.append(f"rank_within_shape: {row['rank_within_shape']!r} is not a positive int")
    elif not 1 <= int(row["rank_within_shape"]) <= EXPECTED_CANDIDATE_COUNT:
        errors.append(
            f"rank_within_shape: {row['rank_within_shape']} is outside "
            f"[1, {EXPECTED_CANDIDATE_COUNT}]"
        )
    if row["best_cutedsl_variant"] not in EXPECTED_CUTEDSL_VARIANTS:
        errors.append(
            f"best_cutedsl_variant: {row['best_cutedsl_variant']!r} is not one of the three "
            "CuTe DSL variants"
        )
    if method == "cublaslt":
        if row["throughput_ratio_vs_cublaslt"] != "1.000000000":
            errors.append(
                "the cuBLASLt baseline row must carry a throughput ratio of exactly 1, got "
                f"{row['throughput_ratio_vs_cublaslt']!r}"
            )
        if row["gap_to_cublaslt_pct"] != "0.000000":
            errors.append(
                "the cuBLASLt baseline row must carry a gap of exactly 0, got "
                f"{row['gap_to_cublaslt_pct']!r}"
            )
    # The per-row tflops must equal flop_count/(kernel_time_ms*1e9).
    if _RE_POSITIVE_INT.match(row["flop_count"]) and re.fullmatch(
        r"(0|[1-9][0-9]*)\.[0-9]{6}", row["kernel_time_ms"]
    ):
        time_ms = float(row["kernel_time_ms"])
        if time_ms > 0.0:
            want = expected_tflops(int(row["flop_count"]), time_ms)
            try:
                got = float(row["tflops"])
            except ValueError:
                got = float("nan")
            if not _close(got, want):
                errors.append(
                    f"{variant}: tflops={row['tflops']} does not equal "
                    f"flop_count/(kernel_time_ms*1e9) = {want!r}"
                )

    # --- provenance -----------------------------------------------------------
    if not _RE_HEX40.match(row["cutlass_commit"]):
        errors.append(f"cutlass_commit: {row['cutlass_commit']!r} is not a 40-hex commit")
    if not _RE_HEX40.match(row["git_commit"]):
        errors.append(f"git_commit: {row['git_commit']!r} is not a 40-hex commit")
    if not _RE_HEX64.match(row["operand_factory_sha256"]):
        errors.append("operand_factory_sha256 is not a 64-hex digest")
    if not _RE_GPU_UUID.match(row["gpu_uuid"]):
        errors.append(f"gpu_uuid: {row['gpu_uuid']!r} is malformed")
    if not _RE_COMPUTE_CAPABILITY.match(row["compute_capability"]):
        errors.append(f"compute_capability: {row['compute_capability']!r} is malformed")
    for field in ("driver_version", "cuda_toolkit_version", "torch_cuda_version",
                  "cutedsl_version"):
        if not _RE_DOTTED_VERSION.match(row[field]):
            errors.append(f"{field}: {row[field]!r} is not a dotted version")
    return errors


def _validate_cutedsl_row(row, variant, scheduler, tiler, cluster, use_2cta, persistent) -> list:
    errors = []
    for field, want in (
        ("scheduler", scheduler),
        ("mma_tiler_m", str(tiler[0])),
        ("mma_tiler_n", str(tiler[1])),
        ("cluster_m", str(cluster[0])),
        ("cluster_n", str(cluster[1])),
        ("use_2cta_instrs", "true" if use_2cta else "false"),
        ("use_tma_store", "true"),
    ):
        if row[field] != want:
            errors.append(f"{variant}: {field}={row[field]!r} != frozen {want!r}")
    if persistent:
        if not _RE_POSITIVE_INT.match(row["max_active_clusters"]):
            errors.append(
                f"{variant}: max_active_clusters={row['max_active_clusters']!r} must be a "
                "positive decimal integer for a persistent variant"
            )
    elif row["max_active_clusters"] != EXPECTED_NOT_APPLICABLE:
        errors.append(
            f"{variant}: max_active_clusters={row['max_active_clusters']!r} must be "
            f"{EXPECTED_NOT_APPLICABLE!r} for the non-persistent variant"
        )
    _validate_decimal_field(
        "compile_time_ms", row["compile_time_ms"], TIMING_DECIMALS, True, errors
    )
    if not _RE_HEX40.match(row["upstream_kernel_git_blob"]):
        errors.append("upstream_kernel_git_blob is not a 40-hex blob")
    if not _RE_HEX64.match(row["upstream_kernel_sha256"]):
        errors.append("upstream_kernel_sha256 is not a 64-hex digest")
    if not _RE_UPSTREAM_REL_PATH.match(row["upstream_kernel_file"]) or ".." in Path(
        row["upstream_kernel_file"]
    ).parts:
        errors.append(
            f"upstream_kernel_file: {row['upstream_kernel_file']!r} is not a relative "
            "upstream .py path"
        )
    return errors


def _validate_cublaslt_row(row, mnkl) -> list:
    """The frozen descriptor contract and algorithm policy of one cuBLASLt row."""
    errors = []
    _m, n, k, _l = mnkl
    for field, want in sorted(EXPECTED_CUBLASLT_POLICY.items()):
        if row[field] != want:
            errors.append(
                f"cuBLASLt row: {field}={row[field]!r} != the frozen P3.3 policy {want!r}"
            )
    # The leading dimensions are derived from the shape, never supplied.
    for field, want in (("lda", str(k)), ("ldb", str(k)), ("ldc", str(n)), ("ldd", str(n))):
        if row[field] != want:
            errors.append(
                f"cuBLASLt row: {field}={row[field]!r} != the shape-derived {want!r}"
            )
    _validate_decimal_field("waves_count", row["waves_count"], WAVES_DECIMALS, False, errors)
    _validate_decimal_field("setup_time_ms", row["setup_time_ms"], TIMING_DECIMALS, True, errors)

    for field in ("workspace_bytes", "heuristic_index", "algo_id", "tile_id", "stages_id",
                  "split_k", "reduction_scheme", "cta_swizzling", "custom_option",
                  "inner_shape_id", "cluster_shape_id", "cublaslt_version"):
        if not _RE_NONNEGATIVE_INT.match(row[field]):
            errors.append(f"{field}: {row[field]!r} is not a non-negative decimal integer")
    for field in ("alignment_a_bytes", "alignment_b_bytes", "alignment_c_bytes",
                  "alignment_d_bytes", "heuristic_returned"):
        if not _RE_POSITIVE_INT.match(row[field]):
            errors.append(f"{field}: {row[field]!r} is not a positive decimal integer")
    if _RE_NONNEGATIVE_INT.match(row["cublaslt_version"]) and int(row["cublaslt_version"]) <= 0:
        errors.append("cublaslt_version must be positive")
    if _RE_NONNEGATIVE_INT.match(row["workspace_bytes"]):
        if int(row["workspace_bytes"]) > EXPECTED_WORKSPACE_LIMIT_BYTES:
            errors.append(
                f"workspace_bytes: {row['workspace_bytes']} exceeds the frozen limit "
                f"{EXPECTED_WORKSPACE_LIMIT_BYTES}"
            )
    if _RE_POSITIVE_INT.match(row["heuristic_returned"]):
        returned = int(row["heuristic_returned"])
        if returned > EXPECTED_HEURISTIC_REQUESTED:
            errors.append(
                f"heuristic_returned: {returned} exceeds the frozen request "
                f"{EXPECTED_HEURISTIC_REQUESTED}"
            )
        if _RE_NONNEGATIVE_INT.match(row["heuristic_index"]):
            if int(row["heuristic_index"]) >= returned:
                errors.append(
                    f"heuristic_index: {row['heuristic_index']} is not below "
                    f"heuristic_returned {returned}"
                )
    for field in ("alignment_a_bytes", "alignment_b_bytes", "alignment_c_bytes",
                  "alignment_d_bytes"):
        if _RE_POSITIVE_INT.match(row[field]):
            value = int(row[field])
            if value & (value - 1):
                errors.append(f"{field}: {value} is not a power of two")
    return errors


def validate_serialized_output(text) -> list:
    """Require exactly one header and exactly twenty rows in shape-major order."""
    errors = []
    if not isinstance(text, str):
        return ["the serialized output is not a string"]
    lines = text.splitlines()
    if len(lines) != EXPECTED_LINE_COUNT:
        errors.append(
            f"the serialized output has {len(lines)} line(s), expected exactly "
            f"{EXPECTED_LINE_COUNT}"
        )
        return errors
    if lines[0] != ",".join(EXPECTED_CSV_FIELDS):
        errors.append("the CSV header does not match the frozen field order")
    parsed = [dict(row) for row in csv.DictReader(io.StringIO(text))]
    if len(parsed) != EXPECTED_ROW_COUNT:
        errors.append(
            f"the serialized output parses to {len(parsed)} row(s), expected "
            f"{EXPECTED_ROW_COUNT}"
        )
        return errors

    observed = [(row.get("shape_index"), row.get("candidate_index")) for row in parsed]
    expected = [
        (str(shape + 1), str(candidate + 1))
        for shape in range(EXPECTED_SHAPE_COUNT)
        for candidate in range(EXPECTED_CANDIDATE_COUNT)
    ]
    if observed != expected:
        errors.append(
            f"the rows are not in shape-major frozen order: got {observed}, expected {expected}"
        )
    observed_variants = [row.get("variant") for row in parsed]
    expected_variants = list(EXPECTED_CANDIDATE_ORDER) * EXPECTED_SHAPE_COUNT
    if observed_variants != expected_variants:
        errors.append(
            f"the candidate order inside the shapes is {observed_variants}, expected "
            f"{expected_variants}"
        )
    observed_shape_ids = [row.get("shape_id") for row in parsed]
    if sorted(set(observed_shape_ids)) != sorted(EXPECTED_SHAPE_IDS):
        errors.append(
            f"the twenty rows describe the shapes {sorted(set(observed_shape_ids))}, expected "
            f"{sorted(EXPECTED_SHAPE_IDS)}"
        )

    # One run: identical provenance and identical run-level settings.
    for field in (
        "gpu_name", "gpu_uuid", "compute_capability", "driver_version",
        "cuda_toolkit_version", "torch_cuda_version", "cutedsl_version", "cutlass_commit",
        "operand_factory_sha256", "git_commit", "git_dirty", "seed", "atol", "rtol",
        "warmup_iterations", "iterations", "schema_version", "experiment", "unit",
        "run_kind", "cache_mode", "publishable",
    ):
        values = {row.get(field) for row in parsed}
        if len(values) != 1:
            errors.append(f"{field} differs between rows of one run: {sorted(values)}")
    # cublaslt_version carries a real value only on the five cuBLASLt rows, so
    # it is compared across exactly those.
    cublaslt_versions = {
        row.get("cublaslt_version") for row in parsed if row.get("method") == "cublaslt"
    }
    if len(cublaslt_versions) != 1:
        errors.append(
            f"cublaslt_version differs between rows of one run: {sorted(cublaslt_versions)}"
        )

    for row in parsed:
        errors.extend(validate_row_mapping(row))
    if errors:
        return errors

    for shape_index in range(EXPECTED_SHAPE_COUNT):
        block = parsed[
            shape_index * EXPECTED_CANDIDATE_COUNT:(shape_index + 1) * EXPECTED_CANDIDATE_COUNT
        ]
        errors.extend(validate_shape_block(shape_index, block))
    return errors


def validate_shape_block(shape_index, block) -> list:
    """Re-derive one shape's comparison fields from its four serialized rows."""
    errors = []
    mnkl = EXPECTED_SHAPES[shape_index]
    label = EXPECTED_SHAPE_IDS[shape_index]
    flop_count = expected_flop_count(mnkl)

    try:
        times = [float(row["kernel_time_ms"]) for row in block]
        tflops = [float(row["tflops"]) for row in block]
        ratios = [float(row["throughput_ratio_vs_cublaslt"]) for row in block]
        gaps = [float(row["gap_to_cublaslt_pct"]) for row in block]
        ranks = [int(row["rank_within_shape"]) for row in block]
    except (ValueError, KeyError) as exc:
        return [f"{label}: a comparison field could not be read: {exc}"]
    if any(not math.isfinite(value) or value <= 0.0 for value in times):
        return [f"{label}: a kernel_time_ms is not finite and strictly positive"]

    for index, row in enumerate(block):
        if int(row["flop_count"]) != flop_count:
            errors.append(
                f"{label}/{row['variant']}: flop_count={row['flop_count']} != {flop_count}"
            )
        want = expected_tflops(flop_count, times[index])
        if not _close(tflops[index], want):
            errors.append(
                f"{label}/{row['variant']}: tflops={tflops[index]!r} does not equal "
                f"flop_count/(kernel_time_ms*1e9) = {want!r}"
            )

    baseline_tflops = tflops[EXPECTED_CUBLASLT_INDEX]
    baseline_time = times[EXPECTED_CUBLASLT_INDEX]
    for index, row in enumerate(block):
        if index == EXPECTED_CUBLASLT_INDEX:
            if ratios[index] != 1.0 or gaps[index] != 0.0:
                errors.append(
                    f"{label}: the baseline row must carry ratio 1 and gap 0, got "
                    f"{ratios[index]!r} / {gaps[index]!r}"
                )
            continue
        want_ratio = tflops[index] / baseline_tflops
        if not _close(ratios[index], want_ratio):
            errors.append(
                f"{label}/{row['variant']}: throughput_ratio_vs_cublaslt={ratios[index]!r} "
                f"does not equal candidate_tflops/cublaslt_tflops = {want_ratio!r}"
            )
        want_ratio_time = baseline_time / times[index]
        if not _close(ratios[index], want_ratio_time):
            errors.append(
                f"{label}/{row['variant']}: throughput_ratio_vs_cublaslt={ratios[index]!r} "
                f"does not equal cublaslt_kernel_time_ms/candidate_kernel_time_ms = "
                f"{want_ratio_time!r}"
            )
        want_gap = 100.0 * (1.0 - ratios[index])
        if not _close(gaps[index], want_gap):
            errors.append(
                f"{label}/{row['variant']}: gap_to_cublaslt_pct={gaps[index]!r} does not "
                f"equal 100*(1 - throughput_ratio_vs_cublaslt) = {want_gap!r}"
            )
        # The documented sign convention must hold exactly.
        if ratios[index] > 1.0 and gaps[index] >= 0.0:
            errors.append(
                f"{label}/{row['variant']}: a candidate faster than cuBLASLt must carry a "
                f"NEGATIVE gap, got {gaps[index]!r} (a clamped gap is forbidden)"
            )
        if ratios[index] < 1.0 and gaps[index] <= 0.0:
            errors.append(
                f"{label}/{row['variant']}: a candidate slower than cuBLASLt must carry a "
                f"positive gap, got {gaps[index]!r}"
            )

    want_ranks = expected_ranking(times)
    if ranks != want_ranks:
        errors.append(
            f"{label}: rank_within_shape {ranks} does not match the ascending "
            f"kernel_time_ms ranking {want_ranks} (ties broken by the frozen candidate order)"
        )
    if sorted(ranks) != list(range(1, EXPECTED_CANDIDATE_COUNT + 1)):
        errors.append(f"{label}: the ranks {ranks} are not a permutation of 1..4")

    best_index = expected_best_cutedsl_index(times)
    want_best = EXPECTED_CANDIDATE_ORDER[best_index]
    declared = {row["best_cutedsl_variant"] for row in block}
    if declared != {want_best}:
        errors.append(
            f"{label}: best_cutedsl_variant is {sorted(declared)}, expected exactly "
            f"{want_best!r} on all four rows"
        )
    flags = [row["is_best_cutedsl"] == "true" for row in block]
    if sum(flags) != 1:
        errors.append(
            f"{label}: exactly one row must carry is_best_cutedsl=true, got {sum(flags)}"
        )
    elif not flags[best_index]:
        errors.append(
            f"{label}: is_best_cutedsl is set on "
            f"{EXPECTED_CANDIDATE_ORDER[flags.index(True)]!r}, but the fastest CuTe DSL "
            f"variant is {want_best!r}"
        )
    return errors


def validate_cli_options(options) -> list:
    """Require exactly the four permitted controls and nothing reopened."""
    errors = []
    actual = set(options)
    unknown = sorted(actual - ALLOWED_CLI_OPTIONS)
    missing = sorted(ALLOWED_CLI_OPTIONS - actual)
    if unknown:
        errors.append(f"the command line exposes forbidden option(s): {', '.join(unknown)}")
    if missing:
        errors.append(f"the command line is missing option(s): {', '.join(missing)}")
    for option in actual:
        normalized = option.lstrip("-").replace("_", "-").lower()
        for fragment in FORBIDDEN_CLI_FRAGMENTS:
            if fragment.replace("_", "-") in normalized:
                errors.append(
                    f"option {option} reopens the frozen contract (matches {fragment!r})"
                )
    return errors


# --- Source structure --------------------------------------------------------


def python_name_tokens(source):
    """Yield every Python NAME token, so prose in comments/strings is exempt."""
    import tokenize

    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NAME:
            yield token.string


def validate_source(source) -> list:
    """Structural checks over the wrapper's own code (not its prose)."""
    errors = []
    try:
        names = list(python_name_tokens(source))
    except Exception as exc:  # noqa: BLE001 - a token error is a hard failure
        return [f"the wrapper could not be tokenized: {exc}"]

    lowered = [name.lower() for name in names]
    for fragment in FORBIDDEN_SOURCE_IDENTIFIERS:
        hits = sorted({name for name in lowered if fragment in name})
        if hits:
            errors.append(
                f"the wrapper defines or uses forbidden identifier(s) containing "
                f"{fragment!r}: {', '.join(hits)}"
            )
    errors.extend(validate_no_framework_matmul(source))
    errors.extend(validate_no_upstream_helpers(source))
    errors.extend(validate_no_result_files(source))
    return errors


def validate_cleanup_source(source) -> list:
    """Require both real cleanup paths to use the fail-closed exception policy."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"the wrapper is not syntactically valid: {exc}"]

    errors = []
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    helper = functions.get("_cleanup_preserving_primary")
    if helper is None:
        errors.append(
            "the wrapper has no _cleanup_preserving_primary helper; cleanup cannot be "
            "fail-closed without masking an active primary error"
        )

    bridge_destroy = None
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "CublasLtBridge":
            continue
        bridge_destroy = next(
            (
                child
                for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name == "destroy"
            ),
            None,
        )
        break
    if bridge_destroy is None:
        errors.append("CublasLtBridge.destroy() is missing")
    elif not any(
        isinstance(node, ast.Raise)
        and isinstance(node.exc, ast.Call)
        and isinstance(node.exc.func, ast.Name)
        and node.exc.func.id == "BridgeError"
        for node in ast.walk(bridge_destroy)
    ):
        errors.append(
            "CublasLtBridge.destroy() does not raise BridgeError when native plan release fails"
        )

    for function_name, cleanup_argument in (
        ("_measure_cublaslt_candidate", "destroy"),
        ("_measure_shape", "release_shape_memory"),
    ):
        function = functions.get(function_name)
        if function is None:
            errors.append(f"the wrapper is missing {function_name}()")
            continue
        protected = False
        for node in ast.walk(function):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_cleanup_preserving_primary"
                and node.args
            ):
                continue
            first = node.args[0]
            if cleanup_argument == "destroy":
                protected = (
                    isinstance(first, ast.Attribute)
                    and isinstance(first.value, ast.Name)
                    and first.value.id == "bridge"
                    and first.attr == "destroy"
                )
            else:
                protected = isinstance(first, ast.Name) and first.id == cleanup_argument
            if protected:
                break
        if not protected:
            errors.append(
                f"{function_name}() does not route {cleanup_argument} through "
                "_cleanup_preserving_primary()"
            )

    shape_function = functions.get("_measure_shape")
    if shape_function is not None:
        called_attributes = {
            node.func.attr
            for node in ast.walk(shape_function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        for required in ("synchronize", "empty_cache"):
            if required not in called_attributes:
                errors.append(
                    f"_measure_shape() no longer performs the required {required} cleanup"
                )
    return errors


def validate_no_framework_matmul(source) -> list:
    """Reject a framework matmul as a measured path, structurally."""
    errors = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"the wrapper is not syntactically valid: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            errors.append(
                f"line {node.lineno}: the wrapper uses the @ matmul operator; every measured "
                "path must be a pinned CuTe DSL kernel or the cuBLASLt bridge"
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_TORCH_MATMUL_ATTRS:
                errors.append(
                    f"line {node.lineno}: the wrapper calls .{node.func.attr}(); a framework "
                    "operation must never be a measured candidate"
                )
    return errors


def validate_no_upstream_helpers(source) -> list:
    """Reject any call into an upstream driver or benchmarking helper."""
    errors = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"the wrapper is not syntactically valid: {exc}"]
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in FORBIDDEN_UPSTREAM_HELPERS:
            continue
        target = node.func.value
        target_name = target.id if isinstance(target, ast.Name) else (
            target.attr if isinstance(target, ast.Attribute) else None
        )
        if target_name in ("module", "modules", "factory_module", "upstream", "mod"):
            errors.append(
                f"line {node.lineno}: the wrapper calls the upstream helper "
                f"{target_name}.{node.func.attr}(); P3.5 owns every timer and never uses "
                "an upstream driver"
            )
    return errors


def validate_no_result_files(source) -> list:
    """Reject result-file or campaign-directory creation, structurally."""
    errors = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"the wrapper is not syntactically valid: {exc}"]
    scratch = _scratch_only_lines(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or node.lineno in scratch:
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else None
        )
        if name in ("mkdir", "makedirs"):
            errors.append(f"line {node.lineno}: the wrapper creates a directory")
        if name == "write_text":
            errors.append(f"line {node.lineno}: the wrapper writes a file")
        if name == "open":
            for argument in list(node.args[1:2]) + [
                keyword.value for keyword in node.keywords if keyword.arg == "mode"
            ]:
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if any(flag in argument.value for flag in ("w", "a", "x", "+")):
                        errors.append(f"line {node.lineno}: the wrapper opens a file for writing")
    for marker in ("results/raw", "results/preflight"):
        if marker in source:
            errors.append(f"the wrapper references the result tree {marker!r}")
    return errors


def _scratch_only_lines(tree) -> set:
    """Line numbers inside functions that write only into a TemporaryDirectory."""
    scratch = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "TemporaryDirectory"
            for inner in ast.walk(node)
        ):
            for inner in ast.walk(node):
                if hasattr(inner, "lineno"):
                    scratch.add(inner.lineno)
    return scratch


def validate_fp32_precision_policy(source) -> list:
    """Require the PyTorch 2.10 fp32_precision API and nothing overlapping."""
    errors = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"the wrapper is not syntactically valid: {exc}"]

    forbidden = set(FORBIDDEN_FP32_API_SPELLINGS)
    seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in forbidden:
                errors.append(
                    f"line {node.lineno}: the wrapper accesses the forbidden FP32 API "
                    f".{node.attr}"
                )
            if node.attr in REQUIRED_FP32_API_SPELLINGS:
                seen.add(node.attr)
        elif isinstance(node, ast.Name) and node.id in forbidden:
            errors.append(
                f"line {node.lineno}: the wrapper references the forbidden FP32 API {node.id}"
            )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value in forbidden:
                errors.append(
                    f"line {node.lineno}: the wrapper names the forbidden FP32 API "
                    f"{node.value!r} as a string, which getattr/setattr would honour"
                )
            if node.value in REQUIRED_FP32_API_SPELLINGS:
                seen.add(node.value)

    for spelling in REQUIRED_FP32_API_SPELLINGS:
        if spelling not in seen:
            errors.append(f"the wrapper never uses the required FP32 API {spelling!r}")
    if f'"{FP32_PRECISION_IEEE}"' not in source and f"'{FP32_PRECISION_IEEE}'" not in source:
        errors.append(f"the wrapper never requires fp32_precision == {FP32_PRECISION_IEEE!r}")
    return errors


def validate_pin_linkage(source, contract) -> list:
    """Prove the wrapper reads the pins instead of duplicating them."""
    errors = []
    for name in ('"VERSIONS.env"', '"PHASE3_VERSIONS.env"'):
        if name not in source:
            errors.append(f"the wrapper does not read {name}")
    for key, value in sorted(contract.items()):
        if len(value) < 3:
            continue
        if value in source:
            errors.append(
                f"the wrapper duplicates the pinned {key} value as a literal instead of "
                "reading it from the version contract"
            )
        if key == "CUTLASS_VERSION" and value.lstrip("v") in source:
            errors.append("the wrapper duplicates the pinned CuTe DSL version as a literal")
    return errors


def validate_no_new_pins(global_values, phase3_values) -> list:
    """P3.5 reuses the existing pins and introduces none of its own."""
    errors = []
    for key in REQUIRED_P31_CONTRACT_KEYS + REQUIRED_P34_CONTRACT_KEYS:
        if key not in phase3_values:
            errors.append(f"{PHASE3_CONTRACT_FILE} is missing the reused key {key}")
    for key in sorted(phase3_values):
        if key.startswith("CUTEDSL_P35") or key.startswith("P35_"):
            errors.append(
                f"{PHASE3_CONTRACT_FILE} gained the P3.5 key {key}; P3.5 needs no new pin"
            )
    for key in sorted(global_values):
        if key.startswith("CUTEDSL_P3") or key.startswith("P35_"):
            errors.append(f"{GLOBAL_CONTRACT_FILE} gained a Phase 3 key ({key})")
    return errors


# --- The P3.5 cuBLASLt bridge ------------------------------------------------


def strip_c_comments(source) -> str:
    """Remove // and /* */ comments so prose cannot satisfy a code check."""
    without_block = re.sub(r"/\*.*?\*/", " ", source, flags=re.S)
    return re.sub(r"//[^\n]*", " ", without_block)


def extract_bridge_constants(source) -> dict:
    """Return the bridge's ``static const ... NAME = VALUE;`` declarations."""
    constants = {}
    for match in re.finditer(
        r"static\s+const\s+[A-Za-z_][A-Za-z0-9_:\s\*]*?\s(P35_[A-Z0-9_]+)\s*=\s*([^;]+);",
        source,
    ):
        constants[match.group(1)] = match.group(2).strip()
    for match in re.finditer(r"#define\s+(P35_[A-Z0-9_]+)\s+(\S+)", source):
        constants.setdefault(match.group(1), match.group(2).strip())
    return constants


def extract_bridge_shapes(source) -> tuple:
    """Return the (M,N,K) allowlist declared in the bridge's own C array."""
    match = re.search(r"P35_SHAPES\s*\[\s*\]\s*\[\s*3\s*\]\s*=\s*\{(.*?)\}\s*;",
                      strip_c_comments(source), flags=re.S)
    if match is None:
        return ()
    shapes = []
    for entry in re.finditer(r"\{\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}", match.group(1)):
        shapes.append(tuple(int(value) for value in entry.groups()))
    return tuple(shapes)


def validate_bridge_source(source) -> list:
    """Structural checks over the P3.5 C-ABI bridge."""
    errors = []
    code = strip_c_comments(source)

    shapes = extract_bridge_shapes(source)
    expected = tuple((m, n, k) for (m, n, k, _l) in EXPECTED_SHAPES)
    if shapes != expected:
        errors.append(
            f"the bridge's own shape allowlist is {shapes}, expected {expected}; the C side "
            "must independently freeze exactly the five P3.5 shapes in the frozen order"
        )

    constants = extract_bridge_constants(source)
    for name, want in sorted(EXPECTED_BRIDGE_CONSTANTS.items()):
        if name not in constants:
            errors.append(f"the bridge does not declare the frozen constant {name}")
        elif constants[name] != want:
            errors.append(
                f"the bridge declares {name} = {constants[name]!r}, expected {want!r}"
            )

    for call in REQUIRED_BRIDGE_CALLS:
        if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(call)}\s*\(", code):
            errors.append(f"the bridge never calls {call}()")

    for call in REQUIRED_BRIDGE_RELEASE_CALLS:
        if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(call)}\s*\(", code):
            errors.append(f"the bridge never releases resources with {call}()")
        if re.search(rf"(?m)^[ \t]*{re.escape(call)}\s*\(", code):
            errors.append(
                f"the bridge discards the return status of {call}(); every release failure "
                "must invalidate an otherwise successful run"
            )

    if not re.search(
        r"static\s+int\s+p35_plan_release\s*\([^)]*bool\s+preserve_existing_error",
        code,
    ):
        errors.append(
            "p35_plan_release() is not a status-returning cleanup that can preserve a primary "
            "diagnostic"
        )
    if "p35_preference_release" not in code:
        errors.append(
            "the bridge has no checked release path for its temporary cuBLASLt preference"
        )
    elif not re.search(r"static\s+int\s+p35_preference_release\s*\(", code):
        errors.append("p35_preference_release() does not return a checked status")
    for helper in ("p35_plan_release", "p35_preference_release"):
        if re.search(rf"(?m)^[ \t]*{helper}\s*\(", code):
            errors.append(
                f"the bridge discards the aggregate status of {helper}()"
            )
    if not re.search(r"return\s+p35_plan_release\s*\(\s*plan\s*,\s*false\s*\)\s*;", code):
        errors.append(
            "p35_plan_destroy() does not propagate the native plan-release status"
        )

    matmul_calls = re.findall(
        rf"(?<![A-Za-z0-9_]){re.escape(REQUIRED_BRIDGE_CALL)}\s*\(", code
    )
    if len(matmul_calls) != 1:
        errors.append(
            f"the bridge calls {REQUIRED_BRIDGE_CALL}() {len(matmul_calls)} time(s); exactly "
            "one measured call site must exist, and no candidate may be executed for "
            "benchmarking"
        )

    for entry_point in FORBIDDEN_GEMM_ENTRY_POINTS:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(entry_point)}(?![A-Za-z0-9_])", code):
            errors.append(
                f"the bridge references {entry_point}; a measured fallback to another GEMM "
                "API or an alternative algorithm-enumeration path is forbidden"
            )

    for export in REQUIRED_BRIDGE_EXPORTS:
        if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(export)}\s*\(", code):
            errors.append(f"the bridge does not define the versioned entry point {export}()")

    for attribute in REQUIRED_ALGO_CONFIG_ATTRIBUTES:
        if attribute not in code:
            errors.append(f"the bridge never records the algorithm attribute {attribute}")
    for preference in REQUIRED_ALIGNMENT_PREFERENCES:
        if preference not in code:
            errors.append(f"the bridge never sets {preference}")

    for facility in FORBIDDEN_BRIDGE_TIMING:
        if facility in code:
            errors.append(
                f"the bridge uses the timing facility {facility!r}; the bridge measures "
                "nothing and never benchmarks a candidate"
            )
    for pattern in FORBIDDEN_BRIDGE_OUTPUT:
        if re.search(pattern, code):
            errors.append(
                f"the bridge writes to a standard stream (matched {pattern!r}); the bridge "
                "must produce no output"
            )

    if 'extern "C"' not in code:
        errors.append("the bridge does not expose a C-compatible ABI")
    if "catch (...)" not in code and "catch(...)" not in code:
        errors.append("the bridge has no catch-all handler; a C++ exception could cross the ABI")
    if "__global__" in code or "__device__" in code:
        errors.append("the bridge defines a custom CUDA kernel; it must own no GEMM kernel")
    if "CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES" not in code:
        errors.append("the bridge never sets the workspace limit preference")
    for marker in REQUIRED_BRIDGE_OVERFLOW_MARKERS:
        if marker not in code:
            errors.append(
                f"the bridge never validates a derived size against {marker}; every dimension "
                "and size calculation must be checked for overflow"
            )
    if "P35_ABI_VERSION" not in code:
        errors.append("the bridge exposes no versioned P3.5 ABI constant")
    # The geometry must be gated through the allowlist before any descriptor.
    if not re.search(r"p35_shape_index_of\s*\(", code):
        errors.append(
            "the bridge has no shape-allowlist gate; an arbitrary geometry could reach a "
            "descriptor"
        )
    return errors


def validate_shared_object(library_path, run) -> list:
    """Inspect the compiled bridge's dynamic symbols, when it is present.

    ``run`` is injected so the self-test can drive this validator with
    synthetic ``nm``/``readelf`` output instead of a real shared object.
    """
    errors = []
    defined = run(["nm", "-D", "--defined-only", str(library_path)])
    undefined = run(["nm", "-D", "-u", str(library_path)])
    dynamic = run(["readelf", "-d", str(library_path)])
    if defined is None or undefined is None or dynamic is None:
        return [f"the compiled bridge {library_path} could not be inspected"]

    if not re.search(r"(?<![A-Za-z0-9_])cublasLtMatmul(?![A-Za-z0-9_])", undefined):
        errors.append("the measured path does not reference cublasLtMatmul")
    for required in ("cublasLtMatmulAlgoCheck", "cublasLtMatmulAlgoGetHeuristic"):
        if not re.search(rf"(?<![A-Za-z0-9_]){required}(?![A-Za-z0-9_])", undefined):
            errors.append(f"the compiled bridge does not reference {required}")
    for export in REQUIRED_BRIDGE_EXPORTS:
        if not re.search(rf"(?<![A-Za-z0-9_]){export}(?![A-Za-z0-9_])", defined):
            errors.append(f"the compiled bridge does not export {export}")
    for forbidden in FORBIDDEN_GEMM_ENTRY_POINTS:
        if re.search(rf"(?<![A-Za-z0-9_]){forbidden}(?![A-Za-z0-9_])", defined + undefined):
            errors.append(
                f"the compiled bridge references the forbidden fallback API {forbidden}"
            )
    if "libcublasLt.so" not in dynamic:
        errors.append("the compiled bridge is not linked against libcublasLt")
    if "libcudart.so" not in dynamic:
        errors.append("the compiled bridge is not linked against libcudart")
    return errors


# --- Make integration --------------------------------------------------------


def parse_make_variables(makefile_text) -> dict:
    """Return the Makefile's simple ``NAME := value`` assignments."""
    variables = {}
    for match in re.finditer(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:=\s*(.*)$", makefile_text,
                             flags=re.M):
        variables[match.group(1)] = match.group(2).strip()
    for _ in range(4):
        for name, value in list(variables.items()):
            variables[name] = expand_make_variables(value, variables, expand_nested=False)
    return variables


def expand_make_variables(text, variables, expand_nested=True) -> str:
    """Substitute ``$(NAME)`` references for which a definition is known."""
    rounds = 4 if expand_nested else 1
    for _ in range(rounds):
        replaced = re.sub(
            r"\$\(([A-Za-z_][A-Za-z0-9_]*)\)",
            lambda match: variables.get(match.group(1), match.group(0)),
            text,
        )
        if replaced == text:
            break
        text = replaced
    return text


def extract_make_recipe(makefile_text, target) -> list:
    """Return the recipe lines of one Make target, without the target line."""
    lines = makefile_text.splitlines()
    recipe = []
    collecting = False
    for line in lines:
        if line.startswith(f"{target}:"):
            collecting = True
            continue
        if collecting:
            if line.startswith("\t"):
                recipe.append(line)
                continue
            if line.strip() == "":
                continue
            break
    return recipe


def logical_recipe_lines(recipe) -> list:
    """Group physical recipe lines into logical (backslash-joined) ones."""
    logical = []
    current = []
    for line in recipe:
        current.append(line)
        if not line.rstrip("\n").endswith("\\"):
            logical.append("\n".join(current))
            current = []
    if current:
        logical.append("\n".join(current))
    return logical


def validate_smoke_recipe(recipe, variables=None) -> list:
    """Check the smoke target's stdout, ordering, and exit-status contract."""
    errors = []
    if not recipe:
        return [f"the {SMOKE_TARGET} recipe is empty or was not found"]

    logical = logical_recipe_lines(recipe)
    joined = "\n".join(recipe)
    if variables:
        joined = expand_make_variables(joined, variables)
        logical = [expand_make_variables(block, variables) for block in logical]

    for block in logical:
        body = block.splitlines()[0].lstrip("\t")
        if not body.startswith("@"):
            errors.append(f"recipe line is not quiet (missing @): {body[:70]!r}")

    for line in recipe:
        for match in re.finditer(r"\becho\b", line):
            if ">&2" not in line[match.end():]:
                errors.append(f"an echo is not redirected to stderr: {line.strip()[:70]!r}")
                break

    first = logical[0] if logical else ""
    if GPU_INDEX_VARIABLE not in first:
        errors.append(
            f"the first {SMOKE_TARGET} recipe step does not validate {GPU_INDEX_VARIABLE}"
        )
    for forbidden in ("docker", "nvcc", LAUNCHER_RELATIVE_PATH):
        if forbidden in first:
            errors.append(
                f"the first recipe step already runs {forbidden!r} before "
                f"{GPU_INDEX_VARIABLE} is validated"
            )

    if f"{LAUNCHER_DATA_MODE_VARIABLE}=1" not in joined:
        errors.append(
            f"the smoke target does not set {LAUNCHER_DATA_MODE_VARIABLE}=1, so launcher "
            "and entrypoint text would contaminate stdout"
        )
    if LAUNCHER_RELATIVE_PATH not in joined:
        errors.append(f"the smoke target does not go through {LAUNCHER_RELATIVE_PATH}")
    if re.search(r"(?<![\w/-])docker\s+run\b", joined):
        errors.append("the smoke target invokes Docker directly")
    if "--gpus all" in joined:
        errors.append("the smoke target exposes every GPU")

    if "rev-parse HEAD" not in joined:
        errors.append("the smoke target does not revalidate the pinned CUTLASS commit")
    if "sha256sum" not in joined:
        errors.append("the smoke target never computes a SHA-256")
    digests = set(re.findall(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", joined))
    if len(digests) < 2:
        errors.append(
            f"the smoke target references {len(digests)} pinned SHA-256 digest(s); BOTH "
            "official sources must be revalidated inside the GPU container"
        )
    if "nvcc" not in joined:
        errors.append("the smoke target never compiles the P3.5 cuBLASLt bridge")
    if "/tmp/" not in joined:
        errors.append("the smoke target does not build the bridge into container-private /tmp")

    if not re.search(rf"--warmup-iterations {SMOKE_WARMUP_ITERATIONS}(?![0-9])", joined):
        errors.append(f"the smoke target does not use exactly {SMOKE_WARMUP_ITERATIONS} warm-ups")
    if not re.search(rf"(?<!-)--iterations {SMOKE_ITERATIONS}(?![0-9])", joined):
        errors.append(
            f"the smoke target does not use exactly {SMOKE_ITERATIONS} measured launches"
        )

    if "|| status=$$?" not in joined:
        errors.append("the smoke target does not capture the launcher exit status")
    if not joined.rstrip().endswith("exit $$status"):
        errors.append("the smoke target does not end by preserving the exit status")

    if SMOKE_SUCCESS_SENTENCE not in joined:
        errors.append("the smoke target never reports success accurately")
    else:
        guard = re.search(r'if \[ "\$\$status" -eq 0 \]; then', joined)
        if guard is None:
            errors.append("the success statement is not guarded by a zero exit status")
        elif joined.index(SMOKE_SUCCESS_SENTENCE) < guard.end():
            errors.append("the success statement is printed before the exit-status guard")

    upper = joined.upper()
    for required, description in (
        ("FUNCTIONAL COMPARISON EVIDENCE", "that this is functional comparison evidence"),
        ("FIVE SHAPES", "that all five shapes were required"),
        ("FOUR CANDIDATES", "that all four candidates were required"),
        ("NON-PUBLISHABLE", "that every row is non-publishable"),
        ("NO FINAL CAMPAIGN", "that no final campaign was performed"),
        ("STATISTICAL", "that no statistical conclusion was drawn"),
        ("NSIGHT", "that no Nsight analysis was performed"),
        ("PHASE 4", "that no Phase 4 interpretation was performed"),
    ):
        if required not in upper:
            errors.append(f"the smoke target does not state {description}")

    for construct in FORBIDDEN_SMOKE_FILTERS:
        if construct in joined:
            errors.append(
                f"the smoke target filters its output with {construct!r}; unexpected stdout "
                "must surface, not be discarded"
            )
    return errors


def validate_check_recipe(makefile_text, recipe) -> list:
    """Check that the GPU-free gate is really GPU-free and complete."""
    errors = []
    if not recipe:
        return [f"the {CHECK_TARGET} recipe is empty or was not found"]
    joined = "\n".join(recipe)

    if not re.search(rf"^{re.escape(CHECK_TARGET)}: {re.escape(CHECK_PREREQUISITE)}$",
                     makefile_text, flags=re.M):
        errors.append(f"{CHECK_TARGET} does not depend on {CHECK_PREREQUISITE}")

    for required, description in (
        ("--network none", "no network"),
        ("--cap-drop ALL", "all capabilities dropped"),
        ("no-new-privileges", "no privilege escalation"),
        ("--user", "the invoking UID/GID"),
        (":ro", "a read-only repository mount"),
        ("PYTHONPYCACHEPREFIX", "bytecode redirected out of the read-only mount"),
        ("pip check", "the dependency-graph gate"),
        ("py_compile", "the Python syntax check"),
        ("--self-test", "the GPU-free self-tests"),
        ("rev-parse HEAD", "the CUTLASS checkout revalidation"),
        ("nvcc", "the P3.5 bridge compilation"),
        ("nm -D", "the ELF symbol inspection"),
        ("readelf -d", "the dynamic dependency inspection"),
        ("cublasLtMatmul", "the proof that the bridge references cublasLtMatmul"),
    ):
        if required not in joined:
            errors.append(f"{CHECK_TARGET} does not use {required!r} ({description})")
    if "sha256sum" not in joined:
        errors.append(f"{CHECK_TARGET} never computes a SHA-256")
    if "hash-object" not in joined:
        errors.append(f"{CHECK_TARGET} never computes a Git blob SHA")
    for expectation in ("P31_EXAMPLE_GIT_BLOB", "P31_EXAMPLE_SHA256",
                        "P34_EXAMPLE_GIT_BLOB", "P34_EXAMPLE_SHA256"):
        if expectation not in joined:
            errors.append(
                f"{CHECK_TARGET} never checks {expectation}; BOTH official sources must be "
                "validated by Git blob and SHA-256"
            )
    for forbidden in FORBIDDEN_GEMM_ENTRY_POINTS[:5]:
        if forbidden not in joined:
            errors.append(
                f"{CHECK_TARGET} does not prove the compiled bridge is free of {forbidden}"
            )
    if "/tmp/" not in joined:
        errors.append(f"{CHECK_TARGET} does not build into container-private /tmp")
    if "--gpus" in joined:
        errors.append(f"{CHECK_TARGET} exposes a GPU; the gate must be GPU-free")
    return errors


def parse_env_file(path) -> dict:
    """Independently parse a KEY=VALUE version contract."""
    values = {}
    text = Path(path).read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _RE_ENV_LINE.match(line)
        if match is None:
            raise ValueError(f"{path}:{lineno}: malformed contract line {raw!r}")
        key, value = match.group(1), match.group(2)
        if key in values:
            raise ValueError(f"{path}:{lineno}: duplicate contract key {key}")
        values[key] = value
    return values


def validate_status_documents(plan_text, protocol_text, readme_text) -> list:
    """Require the closed P3.5/Phase 3 status and untouched closed units."""
    errors = []
    if EXPECTED_P35_STATUS_LINE not in plan_text:
        errors.append(
            f"PLAN.md does not record P3.5 as implemented, audited, and verified "
            f"({EXPECTED_P35_STATUS_LINE!r})"
        )
    for wrong in FORBIDDEN_P35_STATUS_LINES:
        if wrong in plan_text:
            errors.append(f"PLAN.md records a stale or invalid P3.5 status: {wrong!r}")
    for closed in CLOSED_STATUS_LINES:
        if closed not in plan_text:
            errors.append(f"PLAN.md no longer records {closed!r}")
    for phase4 in PHASE4_STATUS_LINES:
        if phase4 not in plan_text:
            errors.append(f"PLAN.md no longer records the Phase 4 frontier row {phase4!r}")
    for wrong in FORBIDDEN_PHASE4_STATUS_LINES:
        if wrong in plan_text:
            errors.append(f"PLAN.md records a stale or premature Phase 4 status: {wrong!r}")

    if "P3.5 = YES / YES / YES" not in protocol_text:
        errors.append(f"{PROTOCOL_RELATIVE_PATH} does not state P3.5 = YES / YES / YES")
    if "P3.5 creates no publishable performance result" not in protocol_text:
        errors.append(
            f"{PROTOCOL_RELATIVE_PATH} does not state that P3.5 creates no publishable result"
        )
    for required in (
        "P3.5 was independently audited",
        "P3.5 was verified on GB300",
        "P3.5 is closed",
        "Phase 3 is closed",
    ):
        if required not in protocol_text:
            errors.append(f"{PROTOCOL_RELATIVE_PATH} omits the closure fact {required!r}")
    for stale in (
        "P3.5 = YES / NO / NO",
        "No independent audit of P3.5 has been performed",
        "no GB300 run of P3.5 exists",
        "Phase 3 remains open",
        "Phase 3: OPEN",
    ):
        if stale in protocol_text:
            errors.append(f"{PROTOCOL_RELATIVE_PATH} retains stale status: {stale!r}")
    if "P3.5 (five shapes and comparison)" not in readme_text:
        errors.append("README.md does not describe P3.5")
    for required in ("P3.5: CLOSED", "Phase 3: CLOSED"):
        if required not in readme_text:
            errors.append(f"README.md omits the closure status: {required!r}")
    return errors


# --- Checks against the real repository --------------------------------------


def _load_module(path, name):
    """Import a Python file as a library under a private name."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run_guarded(wrapper_path, argv):
    """Run the wrapper behind the GPU-free import guard."""
    script = GPU_FREE_GUARD.format(
        blocked=list(GPU_STACK_MODULES),
        argv0=str(wrapper_path),
        argv=list(argv),
        wrapper=str(wrapper_path),
    )
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600, check=False
    )


def _run_probe(script):
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600, check=False
    )


def _tool_runner(command):
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=300, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def validate_cleanup_runtime(module) -> list:
    """Adversarially exercise cleanup success/failure semantics without a GPU."""
    errors = []

    class FailingDestroyLibrary:
        @staticmethod
        def p35_plan_destroy(_plan):
            return 1

        @staticmethod
        def p35_last_error():
            return b"synthetic native plan-release failure"

    bridge = object.__new__(module.CublasLtBridge)
    bridge._plan = object()
    bridge._lib = FailingDestroyLibrary()
    try:
        bridge.destroy()
    except module.BridgeError as exc:
        if "synthetic native plan-release failure" not in str(exc):
            errors.append(
                f"CublasLtBridge.destroy() raised the wrong diagnostic: {str(exc)!r}"
            )
    except BaseException as exc:  # noqa: BLE001
        errors.append(f"CublasLtBridge.destroy() raised the wrong exception type: {exc!r}")
    else:
        errors.append("CublasLtBridge.destroy() accepted a synthetic native release failure")
    if bridge._plan is not None:
        errors.append("CublasLtBridge.destroy() retained a failed native plan handle")

    cleanup_error = module.BridgeError("synthetic cleanup failure after success")

    def fail_cleanup():
        raise cleanup_error

    try:
        module._cleanup_preserving_primary(fail_cleanup, "synthetic successful operation")
    except BaseException as exc:  # noqa: BLE001
        if exc is not cleanup_error:
            errors.append(f"cleanup after success raised the wrong exception: {exc!r}")
    else:
        errors.append("cleanup after success did not fail closed")

    primary_error = module.CorrectnessError("synthetic primary correctness failure")
    original_log = module.log
    module.log = lambda _message: None
    try:
        try:
            raise primary_error
        except module.CorrectnessError:
            module._cleanup_preserving_primary(fail_cleanup, "synthetic failed operation")
            if sys.exc_info()[1] is not primary_error:
                errors.append("cleanup replaced the active primary exception state")
    except BaseException as exc:  # noqa: BLE001
        errors.append(f"cleanup masked the active primary exception with {exc!r}")
    finally:
        module.log = original_log
    return errors


def check_wrapper(repo_root) -> list:
    """Run the whole contract check against the real repository."""
    errors = []
    root = Path(repo_root).resolve()

    wrapper_path = root / WRAPPER_RELATIVE_PATH
    bridge_path = root / BRIDGE_RELATIVE_PATH
    checker_path = root / CHECKER_RELATIVE_PATH
    protocol_path = root / PROTOCOL_RELATIVE_PATH
    makefile_path = root / "Makefile"
    plan_path = root / "PLAN.md"
    readme_path = root / "README.md"

    for path in (wrapper_path, bridge_path, checker_path, protocol_path, makefile_path,
                 plan_path, readme_path):
        if not path.is_file():
            errors.append(f"required file is missing: {path.relative_to(root)}")
    if errors:
        return errors

    wrapper_source = wrapper_path.read_text(encoding="utf-8")
    bridge_source = bridge_path.read_text(encoding="utf-8")
    makefile_text = makefile_path.read_text(encoding="utf-8")

    module = _load_module(wrapper_path, "p35_wrapper_under_test")

    # 1. Schema, shape table, candidate table, and frozen configuration.
    errors.extend(validate_csv_schema(module.CSV_FIELDS))
    errors.extend(validate_shape_table(module.FROZEN_SHAPES))
    errors.extend(validate_candidate_table(module.FROZEN_CANDIDATES))
    errors.extend(validate_frozen_config(module.FROZEN_CONFIG))
    if tuple(module.FROZEN_CANDIDATE_ORDER) != EXPECTED_CANDIDATE_ORDER:
        errors.append(
            f"the wrapper's candidate order {tuple(module.FROZEN_CANDIDATE_ORDER)} is not "
            f"{EXPECTED_CANDIDATE_ORDER}"
        )
    if tuple(module.FROZEN_SHAPE_IDS) != EXPECTED_SHAPE_IDS:
        errors.append(
            f"the wrapper's shape identifiers {tuple(module.FROZEN_SHAPE_IDS)} are not "
            f"{EXPECTED_SHAPE_IDS}"
        )
    if module.NOT_APPLICABLE != EXPECTED_NOT_APPLICABLE:
        errors.append(f"the wrapper's not-applicable marker is {module.NOT_APPLICABLE!r}")
    if tuple(sorted(module.CUTEDSL_ONLY_FIELDS)) != tuple(sorted(EXPECTED_CUTEDSL_ONLY_FIELDS)):
        errors.append("the wrapper's CuTe-DSL-only field set differs from the frozen one")
    if tuple(sorted(module.CUBLASLT_ONLY_FIELDS)) != tuple(
        sorted(EXPECTED_CUBLASLT_ONLY_FIELDS)
    ):
        errors.append("the wrapper's cuBLASLt-only field set differs from the frozen one")
    for name, want in (
        ("EXPECTED_ROW_COUNT", EXPECTED_ROW_COUNT),
        ("FROZEN_SHAPE_COUNT", EXPECTED_SHAPE_COUNT),
        ("FROZEN_CANDIDATE_COUNT", EXPECTED_CANDIDATE_COUNT),
        ("CUBLASLT_CANDIDATE_INDEX", EXPECTED_CUBLASLT_INDEX),
        ("MIN_ITERATIONS", MIN_ITERATIONS),
        ("MAX_WARMUP_ITERATIONS", MAX_WARMUP_ITERATIONS),
        ("MAX_ITERATIONS", MAX_ITERATIONS),
        ("FLOPS_PER_MAC", EXPECTED_FLOPS_PER_MAC),
        ("FROZEN_WORKSPACE_LIMIT_BYTES", EXPECTED_WORKSPACE_LIMIT_BYTES),
        ("FROZEN_HEURISTIC_REQUESTED", EXPECTED_HEURISTIC_REQUESTED),
    ):
        if getattr(module, name, None) != want:
            errors.append(
                f"the wrapper declares {name}={getattr(module, name, None)!r}, expected {want!r}"
            )

    # 2. The wrapper's comparison arithmetic must agree with this checker's own.
    errors.extend(_check_comparison_agreement(module))

    # 3. Serialization of twenty synthetic rows, through the wrapper's own code.
    try:
        rows = module._synthetic_rows()
        errors.extend(validate_serialized_output(module.serialize_rows(rows)))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"the wrapper cannot serialize twenty synthetic valid rows: {exc}")
        rows = None

    # 4. Adversarial rows must be rejected by the wrapper's own validator and
    #    by this checker's independent one.
    if rows is not None:
        errors.extend(_check_adversarial_rows(module, rows))

    # 5. A failed or skipped correctness check can never build a row.
    for bad in ("FAIL", "SKIPPED", ""):
        try:
            module.build_row(
                shape_index=0, candidate_index=0, correctness=bad,
                max_abs_error=0.0, max_rel_error=0.0, compile_time_ms=1.0,
                setup_time_ms=None, first_launch_ms=1.0, kernel_time_ms=1.0,
                warmup_iterations=2, iterations=10, max_active_clusters=None,
                comparison=module.compute_shape_comparison(
                    module.FROZEN_SHAPES[0], (1.0, 2.0, 3.0, 4.0)
                )[0],
                provenance=module._synthetic_provenance(),
                operand_factory_sha256="6" * 64,
                upstream=module._synthetic_upstream("nonpersistent"), plan=None,
            )
        except Exception:  # noqa: BLE001
            pass
        else:
            errors.append(f"the wrapper built a row with correctness={bad!r}")

    # 6. Source-level structure and the FP32 oracle policy.
    errors.extend(validate_source(wrapper_source))
    errors.extend(validate_cleanup_source(wrapper_source))
    errors.extend(validate_cleanup_runtime(module))
    errors.extend(validate_fp32_precision_policy(wrapper_source))
    errors.extend(validate_bridge_source(bridge_source))

    # 7. The command line.
    errors.extend(validate_cli_options(
        set(re.findall(r"--[a-z0-9][a-z0-9-]*", module.build_arg_parser().format_help()))
    ))

    # 8. Version contracts and pin linkage.
    try:
        global_values = parse_env_file(root / GLOBAL_CONTRACT_FILE)
        phase3_values = parse_env_file(root / PHASE3_CONTRACT_FILE)
        errors.extend(validate_no_new_pins(global_values, phase3_values))
        combined = dict(global_values)
        combined.update(phase3_values)
        errors.extend(validate_pin_linkage(wrapper_source, combined))
        wrapper_contract = module.load_pinned_contract(root)
        for key, value in sorted(combined.items()):
            if key in wrapper_contract and wrapper_contract[key] != value:
                errors.append(
                    f"the wrapper resolved {key}={wrapper_contract[key]!r}, the contract "
                    f"file says {value!r}"
                )
        for key in REQUIRED_P31_CONTRACT_KEYS + REQUIRED_P34_CONTRACT_KEYS:
            if key not in wrapper_contract:
                errors.append(f"the wrapper never resolves the pinned key {key}")
        if wrapper_contract.get("CUDA_ARCH") != EXPECTED_CUDA_ARCH:
            errors.append(
                f"the pinned architecture is {wrapper_contract.get('CUDA_ARCH')!r}, "
                f"P3.5 targets {EXPECTED_CUDA_ARCH!r}"
            )
        if wrapper_contract.get("EXPECTED_COMPUTE_CAPABILITY") != EXPECTED_COMPUTE_CAPABILITY:
            errors.append(
                f"{EXPECTED_CUDA_ARCH} must derive compute capability "
                f"{EXPECTED_COMPUTE_CAPABILITY}"
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"the pinned contracts could not be resolved: {exc}")

    # 9. VERSIONS.env must still satisfy the closed P1/P2 aggregators.
    for relative in (P1_AGGREGATOR_RELATIVE_PATH, P2_AGGREGATOR_RELATIVE_PATH):
        aggregator_path = root / relative
        if not aggregator_path.is_file():
            errors.append(f"closed aggregator {relative} is missing")
            continue
        try:
            aggregator = _load_module(aggregator_path, f"p35_closed_{Path(relative).stem}")
            aggregator.parse_versions_env(root / GLOBAL_CONTRACT_FILE)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"{GLOBAL_CONTRACT_FILE} no longer parses under the closed {relative} "
                f"allowlist: {exc}"
            )

    # 10. --help and --self-test are GPU-free.
    for argv in (["--help"], ["--self-test"]):
        completed = _run_guarded(wrapper_path, argv)
        if completed.returncode != 0:
            errors.append(
                f"the wrapper failed the GPU-free guard with {argv}: exit "
                f"{completed.returncode}; {completed.stderr.strip()[-400:]}"
            )
        if argv == ["--self-test"] and completed.stdout:
            errors.append("--self-test wrote to stdout; every diagnostic belongs on stderr")

    # 11. stdout is all-or-nothing, at every candidate position of an early, a
    #     middle, and the final shape; and a native write to descriptor 1
    #     during the measurement cannot contaminate the CSV.
    for fail_shape in (0, 2, EXPECTED_SHAPE_COUNT - 1):
        for fail_candidate in range(EXPECTED_CANDIDATE_COUNT):
            probe = FAILING_POSITION_PROBE.format(
                blocked=list(GPU_STACK_MODULES), wrapper=str(wrapper_path),
                fail_shape=fail_shape, fail_candidate=fail_candidate,
            )
            completed = _run_probe(probe)
            if completed.returncode == 0:
                errors.append(
                    f"a synthetic failure at shape {fail_shape + 1} candidate "
                    f"{fail_candidate + 1} did not produce a non-zero exit status"
                )
            if completed.stdout:
                errors.append(
                    f"a synthetic failure at shape {fail_shape + 1} candidate "
                    f"{fail_candidate + 1} still wrote {len(completed.stdout)} byte(s) to stdout"
                )
    completed = _run_probe(
        SUCCESS_PATH_PROBE.format(blocked=list(GPU_STACK_MODULES), wrapper=str(wrapper_path))
    )
    if completed.returncode != 0:
        errors.append(f"the synthetic success path failed: {completed.stderr.strip()[-400:]}")
    else:
        errors.extend(validate_serialized_output(completed.stdout))
    completed = _run_probe(
        STDOUT_CONTAMINATION_PROBE.format(
            blocked=list(GPU_STACK_MODULES), wrapper=str(wrapper_path)
        )
    )
    if completed.returncode != 0:
        errors.append(
            f"the stdout-contamination probe failed: {completed.stderr.strip()[-400:]}"
        )
    else:
        if "CONTAMINATION" in completed.stdout:
            errors.append(
                "a write to descriptor 1 during the measurement reached stdout; the CSV "
                "contract is not protected against native output"
            )
        errors.extend(validate_serialized_output(completed.stdout))
    completed = _run_probe(
        CLEANUP_FAILURE_PROBE.format(
            blocked=list(GPU_STACK_MODULES), wrapper=str(wrapper_path)
        )
    )
    if completed.returncode == 0:
        errors.append("a synthetic cleanup failure produced a successful exit status")
    if completed.stdout:
        errors.append(
            "a synthetic cleanup failure after twenty prepared rows still emitted "
            f"{len(completed.stdout)} byte(s) to stdout"
        )

    # 12. The compiled bridge, when the GPU-free gate has already built it.
    library_path = Path(getattr(module, "BRIDGE_LIBRARY_PATH", "/nonexistent"))
    if library_path.is_file():
        errors.extend(validate_shared_object(library_path, _tool_runner))
    else:
        print(
            f"check_gemm_comparison_p35: note: {library_path} is not built; the ELF checks "
            f"are skipped here and are executed by make {CHECK_TARGET}",
            file=sys.stderr,
        )

    # 13. Make integration.
    make_variables = {}
    for contract_file in (GLOBAL_CONTRACT_FILE, PHASE3_CONTRACT_FILE):
        try:
            make_variables.update(parse_env_file(root / contract_file))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{contract_file} could not be parsed as Make variables: {exc}")
    make_variables.update(parse_make_variables(makefile_text))
    for name, value in list(make_variables.items()):
        make_variables[name] = expand_make_variables(value, make_variables)
    errors.extend(validate_smoke_recipe(
        extract_make_recipe(makefile_text, SMOKE_TARGET), make_variables
    ))
    errors.extend(validate_check_recipe(
        makefile_text, extract_make_recipe(makefile_text, CHECK_TARGET)
    ))
    if not re.search(rf"^{re.escape(SMOKE_TARGET)}:$", makefile_text, flags=re.M):
        errors.append(f"{SMOKE_TARGET} must have no Make prerequisite")

    # 14. Truthful status documents, and the closed units untouched.
    errors.extend(validate_status_documents(
        plan_path.read_text(encoding="utf-8"),
        protocol_path.read_text(encoding="utf-8"),
        readme_path.read_text(encoding="utf-8"),
    ))
    for relative, marker in (
        (P32_WRAPPER_RELATIVE_PATH, 'SCHEMA_VERSION = "p32.v1"'),
        (P33_WRAPPER_RELATIVE_PATH, 'SCHEMA_VERSION = "p33.v1"'),
        (P34_WRAPPER_RELATIVE_PATH, 'SCHEMA_VERSION = "p34.v1"'),
    ):
        path = root / relative
        if not path.is_file():
            errors.append(f"closed unit file {relative} is missing")
        elif marker not in path.read_text(encoding="utf-8"):
            errors.append(f"P3.5 changed the closed {relative} schema version")
    if not (root / P33_BRIDGE_RELATIVE_PATH).is_file():
        errors.append(f"closed unit file {P33_BRIDGE_RELATIVE_PATH} is missing")
    else:
        closed_bridge = (root / P33_BRIDGE_RELATIVE_PATH).read_text(encoding="utf-8")
        if "p33_plan_create" not in closed_bridge:
            errors.append("P3.5 changed the closed P3.3 bridge ABI")
        if "p35_" in closed_bridge:
            errors.append("P3.5 leaked its own ABI into the closed P3.3 bridge")
    # Every closed one-shape wrapper must still freeze exactly one shape.
    for relative in (P32_WRAPPER_RELATIVE_PATH, P33_WRAPPER_RELATIVE_PATH,
                     P34_WRAPPER_RELATIVE_PATH):
        path = root / relative
        if path.is_file():
            text = path.read_text(encoding="utf-8")
            for marker in ("FROZEN_M = 4096", "FROZEN_N = 4096", "FROZEN_K = 4096",
                           "FROZEN_L = 1"):
                if marker not in text:
                    errors.append(
                        f"P3.5 weakened the closed one-shape restriction of {relative} "
                        f"(missing {marker!r})"
                    )
    for closed_target in ("gemm-cutedsl-p31-check", "gemm-cutedsl-p31-smoke",
                          "gemm-cutedsl-p32-check", "gemm-cutedsl-p32-smoke",
                          "gemm-cublaslt-p33-check", "gemm-cublaslt-p33-smoke",
                          "gemm-cutedsl-p34-check", "gemm-cutedsl-p34-smoke"):
        if not re.search(rf"^{re.escape(closed_target)}:", makefile_text, flags=re.M):
            errors.append(f"P3.5 removed the closed Make target {closed_target}")
    return errors


def _check_comparison_agreement(module) -> list:
    """The wrapper's comparison must agree with this checker's own arithmetic."""
    errors = []
    probes = (
        (7.5, 6.5, 6.0, 5.0),
        (12.0, 11.0, 13.0, 10.0),
        (3.0, 4.0, 5.0, 6.0),
        (2.5, 2.5, 4.0, 3.0),
        (9.0, 8.0, 8.0, 8.5),
        (1.0, 1.0, 1.0, 1.0),
    )
    for shape_index, mnkl in enumerate(EXPECTED_SHAPES):
        want_flop = expected_flop_count(mnkl)
        try:
            got_flop = module.compute_flop_count(module.FROZEN_SHAPES[shape_index])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"the wrapper cannot compute flop_count for {mnkl}: {exc}")
            continue
        if got_flop != want_flop:
            errors.append(
                f"{EXPECTED_SHAPE_IDS[shape_index]}: the wrapper computes flop_count="
                f"{got_flop}, the exact 2*M*N*K value is {want_flop}"
            )
        if not isinstance(got_flop, int):
            errors.append("the wrapper's flop_count is not an exact integer")
    for times in probes:
        mnkl = EXPECTED_SHAPES[0]
        try:
            got = module.compute_shape_comparison(module.FROZEN_SHAPES[0], times)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"the wrapper cannot compare {times}: {exc}")
            continue
        flop = expected_flop_count(mnkl)
        want_ranks = expected_ranking(list(times))
        want_best = EXPECTED_CANDIDATE_ORDER[expected_best_cutedsl_index(list(times))]
        baseline = expected_tflops(flop, times[EXPECTED_CUBLASLT_INDEX])
        for index, entry in enumerate(got):
            want_tflops = expected_tflops(flop, times[index])
            if not _close(entry["tflops"], want_tflops):
                errors.append(f"{times}: candidate {index} tflops disagreement")
            want_ratio = 1.0 if index == EXPECTED_CUBLASLT_INDEX else (
                want_tflops / baseline
            )
            if not _close(entry["throughput_ratio_vs_cublaslt"], want_ratio):
                errors.append(f"{times}: candidate {index} ratio disagreement")
            want_gap = 0.0 if index == EXPECTED_CUBLASLT_INDEX else 100.0 * (1.0 - want_ratio)
            if not _close(entry["gap_to_cublaslt_pct"], want_gap):
                errors.append(f"{times}: candidate {index} gap disagreement")
            if entry["rank_within_shape"] != want_ranks[index]:
                errors.append(
                    f"{times}: candidate {index} rank {entry['rank_within_shape']} != "
                    f"{want_ranks[index]}"
                )
            if entry["best_cutedsl_variant"] != want_best:
                errors.append(f"{times}: best_cutedsl_variant disagreement")
    # A faster-than-cuBLASLt candidate must produce an unclamped negative gap.
    faster = module.compute_shape_comparison(module.FROZEN_SHAPES[0], (3.0, 4.0, 5.0, 6.0))
    if faster[0]["gap_to_cublaslt_pct"] >= 0.0:
        errors.append(
            "the wrapper clamps or mis-signs the gap of a candidate faster than cuBLASLt"
        )
    # Non-finite and non-positive times must be refused.
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        try:
            module.compute_shape_comparison(module.FROZEN_SHAPES[0], (bad, 1.0, 1.0, 1.0))
        except Exception:  # noqa: BLE001
            pass
        else:
            errors.append(f"the wrapper compared a kernel_time_ms of {bad!r}")
    # Neither three nor five candidates may be compared.
    for bad in ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0, 4.0, 5.0)):
        try:
            module.compute_shape_comparison(module.FROZEN_SHAPES[0], bad)
        except Exception:  # noqa: BLE001
            pass
        else:
            errors.append(f"the wrapper compared {len(bad)} candidates")
    return errors


def _check_adversarial_rows(module, rows) -> list:
    """Both validators must reject every adversarial row and row set."""
    errors = []
    adversarial = {
        "publishable=true": (0, {"publishable": "true"}),
        "correctness=FAIL": (0, {"correctness": "FAIL"}),
        "an injected arbitrary shape": (0, {"m": "1024", "shape_id": "1024x4096x4096x1"}),
        "an out-of-range shape index": (0, {"shape_index": "6"}),
        "an out-of-range candidate index": (0, {"candidate_index": "5"}),
        "a changed dtype": (0, {"ab_dtype": "Float16"}),
        "a changed major": (0, {"c_major": "m"}),
        "a changed seed": (0, {"seed": "2222"}),
        "a wrong flop count": (0, {"flop_count": "12345"}),
        "a zero timing": (0, {"kernel_time_ms": "0.000000"}),
        "a NaN timing": (0, {"kernel_time_ms": "nan"}),
        "an infinite timing": (0, {"first_launch_ms": "inf"}),
        "a negative timing": (0, {"kernel_time_ms": "-1.000000"}),
        "a rank outside 1..4": (0, {"rank_within_shape": "5"}),
        "a NaN gap": (0, {"gap_to_cublaslt_pct": "nan"}),
        "a negative-zero gap": (0, {"gap_to_cublaslt_pct": "-0.000000"}),
        "the baseline named as the best CuTe DSL variant": (
            0, {"best_cutedsl_variant": "heuristic_first_supported"}),
        "the baseline marked as the best CuTe DSL row": (3, {"is_best_cutedsl": "true"}),
        "a baseline ratio other than 1": (3, {"throughput_ratio_vs_cublaslt": "0.900000000"}),
        "a baseline gap other than 0": (3, {"gap_to_cublaslt_pct": "1.000000"}),
        "a CuTe DSL row carrying cuBLASLt algorithm metadata": (0, {"algo_id": "21"}),
        "a CuTe DSL row carrying a cuBLASLt setup time": (0, {"setup_time_ms": "1.000000"}),
        "a cuBLASLt row carrying a compile time": (3, {"compile_time_ms": "1.000000"}),
        "a cuBLASLt row carrying a CuTe scheduler": (3, {"scheduler": "nonpersistent"}),
        "a cuBLASLt row with malformed metadata": (3, {"algo_id": "not_applicable"}),
        "a wrong cuBLASLt search mode": (3, {"search_mode": "CUBLASLT_SEARCH_RESERVED_02"}),
        "a changed cuBLASLt workspace limit": (3, {"workspace_limit_bytes": "134217728"}),
        "a changed heuristic request count": (3, {"heuristic_requested": "64"}),
        "a changed transpose policy": (3, {"transb": "CUBLAS_OP_N"}),
        "a column-major cuBLASLt layout": (3, {"order_a": "CUBLASLT_ORDER_COL"}),
        "a wrong leading dimension": (3, {"lda": "1"}),
        "an over-limit workspace": (3, {"workspace_bytes": "67108865"}),
        "a non-power-of-two alignment": (3, {"alignment_a_bytes": "100"}),
        "a non-persistent row with a cluster count": (0, {"max_active_clusters": "148"}),
        "a persistent row without a cluster count": (1, {"max_active_clusters": "not_applicable"}),
        "persistent_2cta with use_2cta_instrs=false": (2, {"use_2cta_instrs": "false"}),
        "a disabled TMA store": (0, {"use_tma_store": "false"}),
        "a non-canonical boolean": (0, {"git_dirty": "True"}),
        "a performance-derived extra field": (0, {"speedup": "1.0"}),
    }
    for description, (index, override) in sorted(adversarial.items()):
        candidate = {**rows[index], **override}
        try:
            module.validate_row(candidate)
        except Exception:  # noqa: BLE001 - any rejection is expected
            pass
        else:
            errors.append(f"the wrapper accepted a row with {description}")
        if not validate_row_mapping(candidate):
            errors.append(f"this checker accepted a row with {description}")

    row_sets = {
        "a missing row": rows[:-1],
        "an extra row": rows + [rows[0]],
        "a duplicated shape block": rows[:4] + rows[:4] + rows[8:],
        "reordered candidates inside a shape": [rows[1], rows[0]] + rows[2:],
        "reordered shape blocks": rows[4:8] + rows[0:4] + rows[8:],
        "a wrong TFLOP/s": [{**rows[0], "tflops": "1.000000"}] + rows[1:],
        "a wrong ratio": [{**rows[0], "throughput_ratio_vs_cublaslt": "0.500000000"}] + rows[1:],
        "a wrong gap": [{**rows[0], "gap_to_cublaslt_pct": "1.000000"}] + rows[1:],
        "a clamped negative gap": (
            rows[:8] + [{**rows[8], "gap_to_cublaslt_pct": "0.000000"}] + rows[9:]),
        "a wrong rank": [{**rows[0], "rank_within_shape": "1"}] + rows[1:],
        "an inconsistent best variant": (
            [{**rows[0], "best_cutedsl_variant": "persistent_1cta"}] + rows[1:]),
        "zero is_best_cutedsl rows in a shape": (
            rows[:2] + [{**rows[2], "is_best_cutedsl": "false"}] + rows[3:]),
        "two is_best_cutedsl rows in a shape": (
            [{**rows[0], "is_best_cutedsl": "true"}] + rows[1:]),
        "mixed GPU provenance": (
            [{**rows[0], "gpu_uuid": "GPU-11111111-1111-1111-1111-111111111111"}] + rows[1:]),
        "mixed git provenance": [{**rows[0], "git_commit": "9" * 40}] + rows[1:],
        "mixed iteration counts": [{**rows[0], "iterations": "11"}] + rows[1:],
        "publishable=true somewhere": [{**rows[0], "publishable": "true"}] + rows[1:],
    }
    for description, candidate_rows in sorted(row_sets.items()):
        try:
            module.validate_rows(candidate_rows)
        except Exception:  # noqa: BLE001
            pass
        else:
            errors.append(f"the wrapper accepted {description}")
        if not validate_serialized_output(_serialize(candidate_rows)):
            errors.append(f"this checker accepted {description}")
    return errors


# --- Self-test ---------------------------------------------------------------


_GOOD_KERNEL_TIMES = (
    (7.5, 6.5, 6.0, 5.0),
    (12.0, 11.0, 13.0, 10.0),
    (3.0, 4.0, 5.0, 6.0),
    (2.5, 2.5, 4.0, 3.0),
    (9.0, 8.0, 8.0, 8.5),
)


def _good_row(shape_index, candidate_index) -> dict:
    """A well-formed row built only from this checker's own expectations."""
    mnkl = EXPECTED_SHAPES[shape_index]
    m, n, k, l = mnkl
    variant = EXPECTED_CANDIDATE_ORDER[candidate_index]
    method, scheduler, tiler, cluster, use_2cta, source, persistent = (
        EXPECTED_CANDIDATE_TABLE[variant]
    )
    times = list(_GOOD_KERNEL_TIMES[shape_index])
    flop = expected_flop_count(mnkl)
    tflops = [expected_tflops(flop, value) for value in times]
    ranks = expected_ranking(times)
    best_index = expected_best_cutedsl_index(times)
    if candidate_index == EXPECTED_CUBLASLT_INDEX:
        ratio, gap = 1.0, 0.0
    else:
        ratio = tflops[candidate_index] / tflops[EXPECTED_CUBLASLT_INDEX]
        gap = 100.0 * (1.0 - ratio)
    gap_text = f"{gap:.{GAP_DECIMALS}f}"
    if float(gap_text) == 0.0:
        gap_text = f"{0.0:.{GAP_DECIMALS}f}"

    row = dict(EXPECTED_FIXED_ROW_VALUES)
    row.update(
        {
            "shape_index": str(shape_index + 1),
            "shape_id": EXPECTED_SHAPE_IDS[shape_index],
            "candidate_index": str(candidate_index + 1),
            "method": method,
            "variant": variant,
            "m": str(m), "n": str(n), "k": str(k), "l": str(l),
            "max_abs_error": "0.000000000",
            "max_rel_error": "0.000000000",
            "first_launch_ms": "12.250000",
            "kernel_time_ms": f"{times[candidate_index]:.6f}",
            "warmup_iterations": "2",
            "iterations": "10",
            "flop_count": str(flop),
            "tflops": f"{tflops[candidate_index]:.{TFLOPS_DECIMALS}f}",
            "throughput_ratio_vs_cublaslt": f"{ratio:.{RATIO_DECIMALS}f}",
            "gap_to_cublaslt_pct": gap_text,
            "rank_within_shape": str(ranks[candidate_index]),
            "best_cutedsl_variant": EXPECTED_CANDIDATE_ORDER[best_index],
            "is_best_cutedsl": "true" if candidate_index == best_index else "false",
            "gpu_name": "SYNTHETIC TEST DEVICE",
            "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "compute_capability": "9.9",
            "driver_version": "999.99.99",
            "cuda_toolkit_version": "99.9.9",
            "torch_cuda_version": "98.7",
            "cutedsl_version": "97.6.5",
            "cutlass_commit": "1" * 40,
            "operand_factory_sha256": "6" * 64,
            "git_commit": "0" * 40,
            "git_dirty": "false",
        }
    )
    if method == "cutedsl":
        row.update({field: EXPECTED_NOT_APPLICABLE for field in EXPECTED_CUBLASLT_ONLY_FIELDS})
        row.update(
            {
                "scheduler": scheduler,
                "mma_tiler_m": str(tiler[0]),
                "mma_tiler_n": str(tiler[1]),
                "cluster_m": str(cluster[0]),
                "cluster_n": str(cluster[1]),
                "use_2cta_instrs": "true" if use_2cta else "false",
                "use_tma_store": "true",
                "max_active_clusters": "148" if persistent else EXPECTED_NOT_APPLICABLE,
                "compile_time_ms": "1234.500000",
                "upstream_kernel_file": (
                    "examples/synthetic/dense_gemm_persistent.py" if source == "persistent"
                    else "examples/synthetic/dense_gemm.py"
                ),
                "upstream_kernel_git_blob": ("3" * 40) if source == "persistent" else ("2" * 40),
                "upstream_kernel_sha256": ("4" * 64) if source == "persistent" else ("5" * 64),
            }
        )
    else:
        row.update({field: EXPECTED_NOT_APPLICABLE for field in EXPECTED_CUTEDSL_ONLY_FIELDS})
        row.update(dict(EXPECTED_CUBLASLT_POLICY))
        row.update(
            {
                "lda": str(k), "ldb": str(k), "ldc": str(n), "ldd": str(n),
                "workspace_bytes": "4194304",
                "alignment_a_bytes": "256", "alignment_b_bytes": "256",
                "alignment_c_bytes": "256", "alignment_d_bytes": "256",
                "heuristic_returned": "8", "heuristic_index": "0",
                "algo_id": "21", "tile_id": "27", "stages_id": "15", "split_k": "0",
                "reduction_scheme": "0", "cta_swizzling": "0", "custom_option": "0",
                "inner_shape_id": "0", "cluster_shape_id": "0",
                "waves_count": "1.500000", "cublaslt_version": "999999",
                "setup_time_ms": "12.500000",
            }
        )
    return row


def _good_rows() -> list:
    return [
        _good_row(shape_index, candidate_index)
        for shape_index in range(EXPECTED_SHAPE_COUNT)
        for candidate_index in range(EXPECTED_CANDIDATE_COUNT)
    ]


def _serialize(rows) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(EXPECTED_CSV_FIELDS), extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _good_shape_table() -> list:
    return [tuple(shape) for shape in EXPECTED_SHAPES]


def _good_candidate_table() -> list:
    table = []
    for variant in EXPECTED_CANDIDATE_ORDER:
        method, scheduler, tiler, cluster, use_2cta, source, persistent = (
            EXPECTED_CANDIDATE_TABLE[variant]
        )
        table.append({
            "variant": variant,
            "method": method,
            "scheduler": scheduler,
            "upstream_class": (
                None if method == "cublaslt"
                else ("PersistentDenseGemmKernel" if persistent else "DenseGemmKernel")
            ),
            "mma_tiler_mn": tiler,
            "cluster_shape_mn": cluster,
            "use_2cta_instrs": use_2cta,
            "source": source,
            "persistent": persistent,
        })
    return table


_GOOD_SMOKE_RECIPE = [
    "\t@if [ -z \"$${BLACKWELL_GPU_INDEX:-}\" ]; then \\",
    "\t\techo \"ERROR: BLACKWELL_GPU_INDEX must be set explicitly.\" >&2; \\",
    "\t\texit 2; \\",
    "\tfi",
    "\t@status=0; \\",
    "\tRUN_CONTAINER_STDOUT_IS_DATA=1 scripts/run_container.sh bash -c 'set -euo pipefail; \\",
    "\t\thead_commit=\"$$(git rev-parse HEAD)\"; \\",
    "\t\t[ \"$$(sha256sum fileA | cut -d\" \" -f1)\" = \"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\" ]; \\",
    "\t\t[ \"$$(sha256sum fileB | cut -d\" \" -f1)\" = \"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\" ]; \\",
    "\t\tmkdir -p /tmp/p35-bridge; \\",
    "\t\tnvcc -shared -o /tmp/p35-bridge/lib.so src/gemm/cublaslt_bridge_p35.cu >&2; \\",
    "\t\texec python3 src/gemm/gemm_comparison.py \\",
    "\t\t\t--warmup-iterations 2 \\",
    "\t\t\t--iterations 10' || status=$$?; \\",
    "\techo \"P3.5 FUNCTIONAL COMPARISON EVIDENCE ONLY: FIVE SHAPES and FOUR CANDIDATES\" >&2; \\",
    "\techo \"were required. ALL ROWS ARE NON-PUBLISHABLE. NO FINAL CAMPAIGN, NO\" >&2; \\",
    "\techo \"STATISTICAL conclusion, NO NSIGHT analysis, and NO PHASE 4 interpretation.\" >&2; \\",
    "\tif [ \"$$status\" -eq 0 ]; then \\",
    "\t\techo \"" + SMOKE_SUCCESS_SENTENCE + "\" >&2; \\",
    "\telse \\",
    "\t\techo \"P3.5 smoke FAILED\" >&2; \\",
    "\tfi; \\",
    "\texit $$status",
]

_GOOD_BRIDGE_SOURCE = """
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <cstdint>
#define P35_ABI_VERSION 1
static const int64_t P35_SHAPES[][3] = {
    {4096, 4096, 4096},
    {8192, 8192, 8192},
    {16384, 512, 4096},
    {32768, 512, 4096},
    {512, 16384, 4096},
};
static const int64_t P35_BATCH_COUNT = 1;
static const cublasOperation_t P35_TRANSA = CUBLAS_OP_N;
static const cublasOperation_t P35_TRANSB = CUBLAS_OP_T;
static const cublasLtOrder_t P35_ORDER = CUBLASLT_ORDER_ROW;
static const cudaDataType_t P35_AB_TYPE = CUDA_R_16BF;
static const cudaDataType_t P35_CD_TYPE = CUDA_R_32F;
static const cublasComputeType_t P35_COMPUTE_TYPE = CUBLAS_COMPUTE_32F;
static const cudaDataType_t P35_SCALE_TYPE = CUDA_R_32F;
static const cublasLtPointerMode_t P35_POINTER_MODE = CUBLASLT_POINTER_MODE_HOST;
static const cublasLtEpilogue_t P35_EPILOGUE = CUBLASLT_EPILOGUE_DEFAULT;
static const float P35_ALPHA = 1.0f;
static const float P35_BETA = 0.0f;
static const uint64_t P35_WORKSPACE_LIMIT_BYTES = 67108864ULL;
static const int P35_HEURISTIC_REQUESTED = 32;
static const cublasLtMatmulSearch_t P35_SEARCH_MODE = CUBLASLT_SEARCH_BEST_FIT;
static const uint32_t P35_MAX_ALIGNMENT_BYTES = 256u;
static int p35_shape_index_of(int64_t m, int64_t n, int64_t k) { return 0; }
static int guard(int64_t a, int64_t b) { return a > INT64_MAX / b || (size_t)a > SIZE_MAX; }
static int p35_plan_release(void* plan, bool preserve_existing_error) {
    int failed = 0;
    const cudaError_t cuda_status = cudaFree(0);
    if (cuda_status != cudaSuccess) { failed = 1; }
    cublasStatus_t status = cublasLtMatrixLayoutDestroy(0);
    if (status != CUBLAS_STATUS_SUCCESS) { failed = 1; }
    status = cublasLtMatmulDescDestroy(0);
    if (status != CUBLAS_STATUS_SUCCESS) { failed = 1; }
    status = cublasLtDestroy(0);
    if (status != CUBLAS_STATUS_SUCCESS) { failed = 1; }
    return failed;
}
static int p35_preference_release(void) {
    const cublasStatus_t status = cublasLtMatmulPreferenceDestroy(0);
    return status == CUBLAS_STATUS_SUCCESS ? 0 : 1;
}
extern "C" {
int p35_bridge_abi_version(void) { return P35_ABI_VERSION; }
size_t p35_plan_info_size(void) { return 8; }
const char* p35_last_error(void) { return ""; }
size_t p35_cublaslt_version(void) { return cublasLtGetVersion(); }
size_t p35_shape_count(void) { return 5; }
int p35_shape_at(size_t i, int64_t* m, int64_t* n, int64_t* k) { return 0; }
int p35_plan_create(void) {
    try {
        cublasLtCreate(0);
        cublasLtMatmulDescCreate(0, P35_COMPUTE_TYPE, P35_SCALE_TYPE);
        cublasLtMatrixLayoutCreate(0, P35_AB_TYPE, 0, 0, 0);
        cublasLtMatmulPreferenceCreate(0);
        cublasLtMatmulPreferenceSetAttribute(0, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 0, 0);
        cublasLtMatmulPreferenceSetAttribute(0, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, 0, 0);
        cublasLtMatmulPreferenceSetAttribute(0, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, 0, 0);
        cublasLtMatmulPreferenceSetAttribute(0, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, 0, 0);
        cublasLtMatmulPreferenceSetAttribute(0, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, 0, 0);
        cublasLtMatmulAlgoGetHeuristic(0);
        cublasLtMatmulAlgoCheck(0);
        cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_ID, 0, 0, 0);
        cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_TILE_ID, 0, 0, 0);
        cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_STAGES_ID, 0, 0, 0);
        cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 0, 0, 0);
        cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, 0, 0, 0);
        cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 0, 0, 0);
        cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, 0, 0, 0);
        cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, 0, 0, 0);
        cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, 0, 0, 0);
        if (p35_preference_release() != 0) { return 1; }
        return 0;
    } catch (...) { return 1; }
}
int p35_plan_execute(void) { cublasLtMatmul(0); return 0; }
int p35_stream_synchronize(void) { return 0; }
int p35_plan_destroy(void* plan) { return p35_plan_release(plan, false); }
}
"""

_GOOD_CLEANUP_SOURCE = """
import sys

class BridgeError(Exception):
    pass

def _cleanup_preserving_primary(cleanup, description):
    primary_error_active = sys.exc_info()[0] is not None
    try:
        cleanup()
    except BaseException:
        if not primary_error_active:
            raise

class CublasLtBridge:
    def destroy(self):
        if self.failed:
            raise BridgeError("release failed")

def _measure_cublaslt_candidate(bridge):
    try:
        return 1
    finally:
        _cleanup_preserving_primary(bridge.destroy, "plan release")

def _measure_shape(torch):
    try:
        return 1
    finally:
        def release_shape_memory():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        _cleanup_preserving_primary(release_shape_memory, "shape release")
"""


def run_self_test() -> int:
    """Prove this checker rejects what it claims to reject, GPU-free."""
    failures = []

    def check(name, condition, detail=""):
        if condition:
            print(f"  ok   {name}", file=sys.stderr)
        else:
            failures.append(f"{name}{': ' + detail if detail else ''}")
            print(f"  FAIL {name} {detail}", file=sys.stderr)

    def rejects(name, errors, fragment=""):
        if not errors:
            check(name, False, "no error was reported")
        elif fragment and not any(fragment in error for error in errors):
            check(name, False, f"errors were {errors!r}")
        else:
            check(name, True)

    print("check_gemm_comparison_p35 --self-test (GPU-free)", file=sys.stderr)

    # --- Schema --------------------------------------------------------------
    check("the frozen schema has 100 fields", len(EXPECTED_CSV_FIELDS) == 100,
          str(len(EXPECTED_CSV_FIELDS)))
    check("the frozen schema has no duplicate", len(set(EXPECTED_CSV_FIELDS)) == 100)
    check("the correct schema is accepted", validate_csv_schema(EXPECTED_CSV_FIELDS) == [])
    rejects("a reordered schema is rejected",
            validate_csv_schema(tuple(reversed(EXPECTED_CSV_FIELDS))), "wrong order")
    rejects("a schema missing a field is rejected",
            validate_csv_schema(EXPECTED_CSV_FIELDS[:-1]), "missing field")
    rejects("a schema with an extra field is rejected",
            validate_csv_schema(EXPECTED_CSV_FIELDS + ("extra",)), "unknown field")
    for phase4 in ("confidence_interval", "roofline_efficiency", "hbm_bandwidth",
                   "sm_utilization", "arithmetic_intensity", "empirical_ceiling_pct"):
        rejects(f"a schema exposing {phase4} is rejected",
                validate_csv_schema(EXPECTED_CSV_FIELDS + (phase4,)))
    check("the method-specific field sets are disjoint",
          not (set(EXPECTED_CUTEDSL_ONLY_FIELDS) & set(EXPECTED_CUBLASLT_ONLY_FIELDS)))

    # --- Shape table ---------------------------------------------------------
    good_shapes = _good_shape_table()
    check("the five frozen shapes are accepted", validate_shape_table(good_shapes) == [],
          str(validate_shape_table(good_shapes)))
    rejects("a missing shape is rejected", validate_shape_table(good_shapes[:4]), "exactly 5")
    rejects("an extra shape is rejected",
            validate_shape_table(good_shapes + [(1024, 1024, 1024, 1)]), "exactly 5")
    rejects("a duplicated shape is rejected",
            validate_shape_table([good_shapes[0]] + good_shapes[1:4] + [good_shapes[0]]),
            "duplicate")
    rejects("a reordered shape table is rejected",
            validate_shape_table([good_shapes[1], good_shapes[0]] + good_shapes[2:]),
            "not the frozen")
    rejects("an arbitrary injected shape is rejected",
            validate_shape_table(good_shapes[:4] + [(1024, 1024, 1024, 1)]), "not the frozen")
    rejects("a batched shape is rejected",
            validate_shape_table(good_shapes[:4] + [(512, 16384, 4096, 2)]), "L=1")
    rejects("a zero extent is rejected",
            validate_shape_table(good_shapes[:4] + [(0, 16384, 4096, 1)]))

    # --- Candidate table -----------------------------------------------------
    good_candidates = _good_candidate_table()
    check("the four frozen candidates are accepted",
          validate_candidate_table(good_candidates) == [],
          str(validate_candidate_table(good_candidates)))
    rejects("a missing candidate is rejected",
            validate_candidate_table(good_candidates[:3]), "exactly 4")
    rejects("a fifth candidate is rejected",
            validate_candidate_table(good_candidates + [good_candidates[0]]), "exactly 4")
    rejects("a duplicated candidate is rejected",
            validate_candidate_table(
                [good_candidates[0], good_candidates[0]] + good_candidates[2:]),
            "duplicate")
    rejects("a reordered candidate table is rejected",
            validate_candidate_table(
                [good_candidates[1], good_candidates[0]] + good_candidates[2:]), "order")
    rejects("the baseline moved out of the last position is rejected",
            validate_candidate_table(
                [good_candidates[3]] + good_candidates[:3]), "order")
    rejects("two cuBLASLt baselines are rejected",
            validate_candidate_table(
                good_candidates[:3] + [{**good_candidates[3], "variant": "persistent_2cta"}]))
    rejects("persistent_2cta with use_2cta_instrs=false is rejected",
            validate_candidate_table(
                good_candidates[:2] + [{**good_candidates[2], "use_2cta_instrs": False}]
                + good_candidates[3:]))
    rejects("persistent_2cta with a (1,1) cluster is rejected",
            validate_candidate_table(
                good_candidates[:2] + [{**good_candidates[2], "cluster_shape_mn": (1, 1)}]
                + good_candidates[3:]),
            "cluster M of 2")
    rejects("the wrong 2-CTA tiler is rejected",
            validate_candidate_table(
                good_candidates[:2] + [{**good_candidates[2], "mma_tiler_mn": (128, 128)}]
                + good_candidates[3:]),
            "per-CTA M extent")
    rejects("a persistent candidate routed to DenseGemmKernel is rejected",
            validate_candidate_table(
                [good_candidates[0], {**good_candidates[1], "upstream_class": "DenseGemmKernel"}]
                + good_candidates[2:]),
            "routed to")

    # --- Rows and output -----------------------------------------------------
    rows = _good_rows()
    check("twenty well-formed rows are accepted",
          validate_serialized_output(_serialize(rows)) == [],
          str(validate_serialized_output(_serialize(rows))[:3]))
    check("a well-formed row is accepted", validate_row_mapping(rows[0]) == [],
          str(validate_row_mapping(rows[0])))
    check("a well-formed cuBLASLt row is accepted", validate_row_mapping(rows[3]) == [],
          str(validate_row_mapping(rows[3])))
    rejects("a nineteen-row output is rejected",
            validate_serialized_output(_serialize(rows[:-1])), "expected exactly 21")
    rejects("a twenty-one-row output is rejected",
            validate_serialized_output(_serialize(rows + [rows[0]])), "expected exactly 21")
    rejects("a header-only output is rejected",
            validate_serialized_output(_serialize([])), "expected exactly 21")
    rejects("an empty output is rejected", validate_serialized_output(""))
    rejects("reordered candidates inside a shape are rejected",
            validate_serialized_output(_serialize([rows[1], rows[0]] + rows[2:])),
            "shape-major")
    rejects("reordered shape blocks are rejected",
            validate_serialized_output(_serialize(rows[4:8] + rows[0:4] + rows[8:])),
            "shape-major")
    rejects("a duplicated shape block is rejected",
            validate_serialized_output(_serialize(rows[:4] + rows[:4] + rows[8:])),
            "shape-major")
    rejects("a missing candidate is rejected",
            validate_serialized_output(_serialize(rows[:3] + rows[4:])), "expected exactly 21")

    for description, (index, override) in sorted({
        "publishable=true": (0, {"publishable": "true"}),
        "correctness=FAIL": (0, {"correctness": "FAIL"}),
        "correctness=SKIPPED": (0, {"correctness": "SKIPPED"}),
        "an arbitrary injected shape": (0, {"m": "1024"}),
        "an out-of-range shape index": (0, {"shape_index": "6"}),
        "an out-of-range candidate index": (0, {"candidate_index": "5"}),
        "a wrong flop count": (0, {"flop_count": "12345"}),
        "a zero flop count": (0, {"flop_count": "0"}),
        "a changed dtype": (0, {"ab_dtype": "Float16"}),
        "a changed major": (0, {"a_major": "m"}),
        "a changed seed": (0, {"seed": "2222"}),
        "a disabled TMA store": (0, {"use_tma_store": "false"}),
        "a NaN timing": (0, {"kernel_time_ms": "nan"}),
        "an infinite timing": (0, {"first_launch_ms": "inf"}),
        "a zero timing": (0, {"kernel_time_ms": "0.000000"}),
        "a negative timing": (0, {"kernel_time_ms": "-1.000000"}),
        "an exponent-notation timing": (0, {"kernel_time_ms": "1e3"}),
        "a wrong-precision timing": (0, {"kernel_time_ms": "7.5"}),
        "a NaN TFLOP/s": (0, {"tflops": "nan"}),
        "a zero TFLOP/s": (0, {"tflops": "0.000000"}),
        "an incorrect TFLOP/s": (0, {"tflops": "1.000000"}),
        "a rank of zero": (0, {"rank_within_shape": "0"}),
        "a rank above four": (0, {"rank_within_shape": "5"}),
        "a NaN gap": (0, {"gap_to_cublaslt_pct": "nan"}),
        "a negative-zero gap": (0, {"gap_to_cublaslt_pct": "-0.000000"}),
        "the baseline named as best_cutedsl_variant": (
            0, {"best_cutedsl_variant": "heuristic_first_supported"}),
        "an unknown best_cutedsl_variant": (0, {"best_cutedsl_variant": "persistent_4cta"}),
        "the baseline marked as the best CuTe DSL row": (3, {"is_best_cutedsl": "true"}),
        "a baseline ratio other than 1": (3, {"throughput_ratio_vs_cublaslt": "0.900000000"}),
        "a baseline gap other than 0": (3, {"gap_to_cublaslt_pct": "1.000000"}),
        "a CuTe DSL row carrying cuBLASLt metadata": (0, {"algo_id": "21"}),
        "a CuTe DSL row carrying a setup time": (0, {"setup_time_ms": "1.000000"}),
        "a cuBLASLt row carrying a compile time": (3, {"compile_time_ms": "1.000000"}),
        "a cuBLASLt row carrying a scheduler": (3, {"scheduler": "nonpersistent"}),
        "malformed cuBLASLt metadata": (3, {"algo_id": "not_applicable"}),
        "a malformed cuBLASLt waves count": (3, {"waves_count": "nan"}),
        "a wrong cuBLASLt search mode": (3, {"search_mode": "CUBLASLT_SEARCH_RESERVED_02"}),
        "a changed cuBLASLt workspace limit": (3, {"workspace_limit_bytes": "134217728"}),
        "a changed heuristic request count": (3, {"heuristic_requested": "64"}),
        "a changed transpose policy": (3, {"transb": "CUBLAS_OP_N"}),
        "a column-major cuBLASLt layout": (3, {"order_d": "CUBLASLT_ORDER_COL"}),
        "a changed cuBLASLt compute type": (3, {"compute_type": "CUBLAS_COMPUTE_32F_FAST_TF32"}),
        "a changed alpha": (3, {"alpha": "2.000000000"}),
        "a changed beta": (3, {"beta": "1.000000000"}),
        "a wrong leading dimension": (3, {"lda": "1"}),
        "an over-limit workspace": (3, {"workspace_bytes": "67108865"}),
        "more returned heuristics than requested": (3, {"heuristic_returned": "33"}),
        "a heuristic index past the returned count": (3, {"heuristic_index": "8"}),
        "a non-power-of-two alignment": (3, {"alignment_a_bytes": "100"}),
        "a non-persistent row with a cluster count": (0, {"max_active_clusters": "148"}),
        "a persistent row without a cluster count": (
            1, {"max_active_clusters": "not_applicable"}),
        "persistent_2cta with use_2cta_instrs=false": (2, {"use_2cta_instrs": "false"}),
        "persistent_2cta with a (1,1) cluster": (2, {"cluster_m": "1", "cluster_n": "1"}),
        "a non-canonical boolean": (0, {"git_dirty": "TRUE"}),
        "a malformed GPU UUID": (0, {"gpu_uuid": "0000"}),
        "a malformed operand-factory digest": (0, {"operand_factory_sha256": "abc"}),
        "an absolute upstream path": (0, {"upstream_kernel_file": "/opt/cutlass/x.py"}),
        "an out-of-range iteration count": (0, {"iterations": "101"}),
    }.items()):
        rejects(f"a row with {description} is rejected",
                validate_row_mapping({**rows[index], **override}))
    rejects("a row missing a field is rejected",
            validate_row_mapping({k: v for k, v in rows[0].items() if k != "tflops"}),
            "missing field")
    rejects("a row with an extra field is rejected",
            validate_row_mapping({**rows[0], "speedup": "1.0"}), "unknown field")

    # Comparison invariants that only show up across a whole shape block.
    rejects("a wrong ratio is rejected",
            validate_serialized_output(_serialize(
                [{**rows[0], "throughput_ratio_vs_cublaslt": "0.500000000"}] + rows[1:])),
            "candidate_tflops/cublaslt_tflops")
    rejects("a wrong gap is rejected",
            validate_serialized_output(_serialize(
                [{**rows[0], "gap_to_cublaslt_pct": "1.000000"}] + rows[1:])),
            "100*(1 - throughput_ratio")
    rejects("a clamped negative gap is rejected",
            validate_serialized_output(_serialize(
                rows[:8] + [{**rows[8], "gap_to_cublaslt_pct": "0.000000"}] + rows[9:])),
            "gap")
    rejects("a positive gap on a faster candidate is rejected",
            validate_serialized_output(_serialize(
                rows[:8] + [{**rows[8], "gap_to_cublaslt_pct": "100.000000"}] + rows[9:])),
            "gap")
    rejects("a wrong rank is rejected",
            validate_serialized_output(_serialize(
                [{**rows[0], "rank_within_shape": "1"}] + rows[1:])), "ranking")
    rejects("a wrong tie break is rejected",
            validate_serialized_output(_serialize(
                rows[:12] + [{**rows[12], "rank_within_shape": "2"},
                             {**rows[13], "rank_within_shape": "1"}] + rows[14:])),
            "ranking")
    rejects("an inconsistent best_cutedsl_variant is rejected",
            validate_serialized_output(_serialize(
                [{**rows[0], "best_cutedsl_variant": "persistent_1cta"}] + rows[1:])),
            "best_cutedsl_variant is")
    rejects("zero is_best_cutedsl rows in a shape is rejected",
            validate_serialized_output(_serialize(
                rows[:2] + [{**rows[2], "is_best_cutedsl": "false"}] + rows[3:])),
            "exactly one row")
    rejects("two is_best_cutedsl rows in a shape is rejected",
            validate_serialized_output(_serialize(
                [{**rows[0], "is_best_cutedsl": "true"}] + rows[1:])), "exactly one row")
    rejects("mixed GPU provenance is rejected",
            validate_serialized_output(_serialize(
                [{**rows[0], "gpu_uuid": "GPU-11111111-1111-1111-1111-111111111111"}]
                + rows[1:])),
            "differs between rows")
    rejects("mixed git provenance is rejected",
            validate_serialized_output(_serialize(
                [{**rows[0], "git_commit": "9" * 40}] + rows[1:])), "differs between rows")
    rejects("mixed iteration counts are rejected",
            validate_serialized_output(_serialize(
                [{**rows[0], "iterations": "11"}] + rows[1:])), "differs between rows")
    rejects("mixed cuBLASLt runtime versions are rejected",
            validate_serialized_output(_serialize(
                rows[:3] + [{**rows[3], "cublaslt_version": "1"}] + rows[4:])),
            "differs between rows")

    # --- CLI -----------------------------------------------------------------
    check("the permitted option set is accepted", validate_cli_options(ALLOWED_CLI_OPTIONS) == [])
    for bad in ("--shape", "--mnkl", "--method", "--variant", "--candidate", "--seed",
                "--atol", "--skip-ref-check", "--workspace", "--algo-id", "--heuristic",
                "--output", "--csv-file", "--only-shape", "--publish", "--partial",
                "--input", "--config", "--gpu-index"):
        rejects(f"the forbidden option {bad} is rejected",
                validate_cli_options(set(ALLOWED_CLI_OPTIONS) | {bad}))
    rejects("a missing permitted option is rejected",
            validate_cli_options(ALLOWED_CLI_OPTIONS - {"--iterations"}), "missing option")

    # --- Wrapper source structure --------------------------------------------
    check("a clean wrapper body is accepted",
          validate_source("import torch\n\n\ndef f(x):\n    return x\n") == [])
    for description, source in (
        ("a confidence interval", "confidence_interval = (0.0, 1.0)\n"),
        ("a bootstrap", "bootstrap_median = 1.0\n"),
        ("an outlier filter", "outlier_mask = []\n"),
        ("a roofline", "roofline_limit = 1.0\n"),
        ("a bandwidth figure", "hbm_bandwidth = 1.0\n"),
        ("an Nsight call", "ncu_report = 1\n"),
        ("a campaign directory", "campaign_dir = 'x'\n"),
        ("a torch.matmul path", "c = torch.matmul(a, b)\n"),
        ("a torch.bmm path", "c = torch.bmm(a, b)\n"),
        ("an @ operator path", "c = a @ b\n"),
        ("an upstream run() call", "module.run(1)\n"),
        ("an upstream benchmark call", "module.benchmark(1)\n"),
        ("a result-file write", "open('results/raw/x.csv', 'w')\n"),
        ("a plot", "import matplotlib\n"),
    ):
        rejects(f"{description} is rejected", validate_source(source))
    check("torch.einsum as the untimed oracle is allowed",
          validate_source("r = torch.einsum('mkl,nkl->mnl', a, b)\n") == [])
    check("the comparison quantities P3.5 owns are allowed",
          validate_source("tflops = 1.0\nratio = 1.0\ngap = 0.0\nrank = 1\n") == [])
    check("subprocess.run is not mistaken for an upstream helper",
          validate_no_upstream_helpers("import subprocess\nsubprocess.run(['git'])\n") == [])

    # --- Fail-closed cleanup --------------------------------------------------
    check("the complete cleanup policy is accepted",
          validate_cleanup_source(_GOOD_CLEANUP_SOURCE) == [],
          str(validate_cleanup_source(_GOOD_CLEANUP_SOURCE)))
    rejects("a bridge destroy method that suppresses release failure is rejected",
            validate_cleanup_source(
                _GOOD_CLEANUP_SOURCE.replace(
                    'raise BridgeError("release failed")', "return")),
            "does not raise BridgeError")
    rejects("an unprotected cuBLASLt plan cleanup is rejected",
            validate_cleanup_source(
                _GOOD_CLEANUP_SOURCE.replace(
                    '_cleanup_preserving_primary(bridge.destroy, "plan release")',
                    "bridge.destroy()")),
            "_measure_cublaslt_candidate")
    rejects("an unprotected shape-memory cleanup is rejected",
            validate_cleanup_source(
                _GOOD_CLEANUP_SOURCE.replace(
                    '_cleanup_preserving_primary(release_shape_memory, "shape release")',
                    "release_shape_memory()")),
            "_measure_shape")
    rejects("shape cleanup without empty_cache is rejected",
            validate_cleanup_source(
                _GOOD_CLEANUP_SOURCE.replace("torch.cuda.empty_cache()", "pass")),
            "empty_cache")

    # --- FP32 policy ----------------------------------------------------------
    check("the required FP32 policy is accepted",
          validate_fp32_precision_policy(
              'm.fp32_precision = "ieee"\n# the legacy allow_tf32 alias is never used\n') == [])
    rejects("a legacy allow_tf32 write is rejected",
            validate_fp32_precision_policy(
                'm.fp32_precision = "ieee"\nm.allow_tf32 = False\n'), "allow_tf32")
    rejects("set_float32_matmul_precision is rejected",
            validate_fp32_precision_policy(
                'torch.set_float32_matmul_precision("highest")\nm.fp32_precision = "ieee"\n'),
            "set_float32_matmul_precision")
    rejects("a wrapper that never uses fp32_precision is rejected",
            validate_fp32_precision_policy('x = "ieee"\n'), "never uses the required")

    # --- Bridge source --------------------------------------------------------
    check("a well-formed P3.5 bridge is accepted",
          validate_bridge_source(_GOOD_BRIDGE_SOURCE) == [],
          str(validate_bridge_source(_GOOD_BRIDGE_SOURCE)))
    rejects("a bridge with a fourth shape only is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE.replace("    {512, 16384, 4096},\n", "")),
            "shape allowlist")
    rejects("a bridge with an injected shape is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE.replace("{512, 16384, 4096}", "{1024, 1024, 1024}")),
            "shape allowlist")
    rejects("a bridge with reordered shapes is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE.replace(
                    "    {4096, 4096, 4096},\n    {8192, 8192, 8192},\n",
                    "    {8192, 8192, 8192},\n    {4096, 4096, 4096},\n")),
            "shape allowlist")
    rejects("a bridge with no shape gate is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE.replace("p35_shape_index_of", "unused_helper")),
            "shape-allowlist gate")
    for forbidden in ("cublasGemmEx", "cublasGemmStridedBatchedEx", "cublasSgemm",
                      "cublasHgemm", "cublasLtMatmulAlgoGetIds", "cublasLtMatmulAlgoInit"):
        rejects(f"a bridge referencing {forbidden} is rejected",
                validate_bridge_source(
                    _GOOD_BRIDGE_SOURCE + f"\nint fallback(void) {{ return {forbidden}(0); }}\n"),
                forbidden)
    rejects("a bridge with two cublasLtMatmul call sites is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE + "\nint again(void) { cublasLtMatmul(0); return 0; }\n"),
            "exactly")
    rejects("a bridge that never calls cublasLtMatmul is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE.replace("cublasLtMatmul(0)", "nothing(0)")),
            "cublasLtMatmul")
    rejects("a bridge that never validates the algorithm is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE.replace("cublasLtMatmulAlgoCheck(0)", "nothing(0)")),
            "cublasLtMatmulAlgoCheck")
    rejects("a bridge with a CUDA kernel is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE + "\n__global__ void k(void) {}\n"), "CUDA kernel")
    rejects("a bridge that prints is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE + '\nvoid t(void) { printf("x"); }\n'), "standard stream")
    rejects("a bridge that times is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE + "\nvoid t(void) { cudaEventRecord(0); }\n"),
            "timing facility")
    rejects("a bridge with no catch-all handler is rejected",
            validate_bridge_source(_GOOD_BRIDGE_SOURCE.replace("catch (...)", "catch (int)")),
            "catch-all")
    rejects("a bridge with no overflow validation is rejected",
            validate_bridge_source(_GOOD_BRIDGE_SOURCE.replace("INT64_MAX", "0")), "overflow")
    rejects("a bridge changing the workspace limit is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE.replace("67108864ULL", "134217728ULL")),
            "P35_WORKSPACE_LIMIT_BYTES")
    rejects("a bridge changing the heuristic count is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE.replace("P35_HEURISTIC_REQUESTED = 32",
                                            "P35_HEURISTIC_REQUESTED = 64")),
            "P35_HEURISTIC_REQUESTED")
    rejects("a bridge changing the search mode is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE.replace("CUBLASLT_SEARCH_BEST_FIT",
                                            "CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID")),
            "P35_SEARCH_MODE")
    rejects("a bridge changing the transpose policy is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE.replace("P35_TRANSB = CUBLAS_OP_T",
                                            "P35_TRANSB = CUBLAS_OP_N")),
            "P35_TRANSB")
    rejects("a bridge missing an export is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE.replace("int p35_shape_at(", "int renamed_shape_at(")),
            "p35_shape_at")
    for release_call, checked_line, unchecked_line in (
        ("cudaFree", "const cudaError_t cuda_status = cudaFree(0);", "cudaFree(0);"),
        (
            "cublasLtMatrixLayoutDestroy",
            "cublasStatus_t status = cublasLtMatrixLayoutDestroy(0);",
            "cublasLtMatrixLayoutDestroy(0); cublasStatus_t status = CUBLAS_STATUS_SUCCESS;",
        ),
        (
            "cublasLtMatmulDescDestroy",
            "status = cublasLtMatmulDescDestroy(0);",
            "cublasLtMatmulDescDestroy(0);",
        ),
        (
            "cublasLtMatmulPreferenceDestroy",
            "const cublasStatus_t status = cublasLtMatmulPreferenceDestroy(0);",
            "cublasLtMatmulPreferenceDestroy(0); const cublasStatus_t status = "
            "CUBLAS_STATUS_SUCCESS;",
        ),
        (
            "cublasLtDestroy",
            "status = cublasLtDestroy(0);",
            "cublasLtDestroy(0);",
        ),
    ):
        rejects(
            f"a bridge discarding {release_call} status is rejected",
            validate_bridge_source(_GOOD_BRIDGE_SOURCE.replace(checked_line, unchecked_line)),
            "discards the return status",
        )
    rejects("p35_plan_destroy dropping the aggregate cleanup status is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE.replace(
                    "return p35_plan_release(plan, false);",
                    "p35_plan_release(plan, false); return 0;")),
            "does not propagate")
    rejects("plan creation dropping the aggregate preference-release status is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE_SOURCE.replace(
                    "if (p35_preference_release() != 0) { return 1; }",
                    "p35_preference_release();")),
            "discards the aggregate status")

    # --- Compiled shared object ----------------------------------------------
    def _fake_tools(defined, undefined, dynamic):
        def run(command):
            if command[0] == "readelf":
                return dynamic
            if "--defined-only" in command:
                return defined
            return undefined
        return run

    good_defined = "\n".join(f"0000 T {name}" for name in REQUIRED_BRIDGE_EXPORTS)
    good_undefined = "U cublasLtMatmul\nU cublasLtMatmulAlgoCheck\nU cublasLtMatmulAlgoGetHeuristic"
    good_dynamic = "NEEDED libcublasLt.so.13\nNEEDED libcudart.so.13"
    check("a well-formed shared object is accepted",
          validate_shared_object(
              Path("x"), _fake_tools(good_defined, good_undefined, good_dynamic)) == [],
          str(validate_shared_object(
              Path("x"), _fake_tools(good_defined, good_undefined, good_dynamic))))
    rejects("a shared object without cublasLtMatmul is rejected",
            validate_shared_object(
                Path("x"), _fake_tools(good_defined, "U something_else", good_dynamic)),
            "cublasLtMatmul")
    rejects("a shared object referencing a fallback GEMM API is rejected",
            validate_shared_object(
                Path("x"),
                _fake_tools(good_defined, good_undefined + "\nU cublasGemmEx", good_dynamic)),
            "cublasGemmEx")
    rejects("a shared object missing an export is rejected",
            validate_shared_object(
                Path("x"),
                _fake_tools(good_defined.replace("p35_shape_at", "gone"), good_undefined,
                            good_dynamic)),
            "p35_shape_at")
    rejects("a shared object not linked against libcublasLt is rejected",
            validate_shared_object(
                Path("x"), _fake_tools(good_defined, good_undefined, "NEEDED libcudart.so.13")),
            "libcublasLt")
    rejects("an uninspectable shared object is rejected",
            validate_shared_object(Path("x"), lambda command: None), "could not be inspected")

    # --- Make recipes ---------------------------------------------------------
    check("a well-formed smoke recipe is accepted",
          validate_smoke_recipe(_GOOD_SMOKE_RECIPE) == [],
          str(validate_smoke_recipe(_GOOD_SMOKE_RECIPE)))
    rejects("a smoke recipe that echoes the recipe is rejected",
            validate_smoke_recipe([l.replace("\t@", "\t", 1) for l in _GOOD_SMOKE_RECIPE]),
            "not quiet")
    rejects("a smoke recipe echoing to stdout is rejected",
            validate_smoke_recipe([l.replace(" >&2", "") for l in _GOOD_SMOKE_RECIPE]),
            "not redirected to stderr")
    rejects("a smoke recipe not validating the GPU index first is rejected",
            validate_smoke_recipe(_GOOD_SMOKE_RECIPE[4:]), GPU_INDEX_VARIABLE)
    rejects("a smoke recipe calling Docker directly is rejected",
            validate_smoke_recipe(
                [l.replace("scripts/run_container.sh", "docker run --rm")
                 for l in _GOOD_SMOKE_RECIPE]), "Docker")
    rejects("a smoke recipe exposing every GPU is rejected",
            validate_smoke_recipe(
                [l.replace("bash -c", "--gpus all bash -c") for l in _GOOD_SMOKE_RECIPE]),
            "every GPU")
    rejects("a smoke recipe validating only one source is rejected",
            validate_smoke_recipe([l for l in _GOOD_SMOKE_RECIPE if "b" * 64 not in l]),
            "BOTH official sources")
    rejects("a smoke recipe that never compiles the bridge is rejected",
            validate_smoke_recipe([l for l in _GOOD_SMOKE_RECIPE if "nvcc" not in l]),
            "compiles the P3.5 cuBLASLt bridge")
    rejects("a smoke recipe with the wrong iteration counts is rejected",
            validate_smoke_recipe(
                [l.replace("--iterations 10", "--iterations 1000") for l in _GOOD_SMOKE_RECIPE]),
            "measured launches")
    rejects("a smoke recipe with an unguarded success message is rejected",
            validate_smoke_recipe(
                [l for l in _GOOD_SMOKE_RECIPE if 'if [ "$$status" -eq 0 ]' not in l]),
            "guarded")
    rejects("a smoke recipe that drops the exit status is rejected",
            validate_smoke_recipe(
                [l for l in _GOOD_SMOKE_RECIPE if not l.strip().startswith("exit $$status")]),
            "exit status")
    rejects("a smoke recipe filtering stdout is rejected",
            validate_smoke_recipe(_GOOD_SMOKE_RECIPE[:-1]
                                  + ["\t\t| grep -v x; \\", "\texit $$status"]),
            "filters its output")
    for missing, fragment in (
        ("FIVE SHAPES", "five shapes"),
        ("FOUR CANDIDATES", "four candidates"),
        ("NON-PUBLISHABLE", "non-publishable"),
        ("NO FINAL CAMPAIGN", "final campaign"),
        ("STATISTICAL", "statistical"),
        ("NSIGHT", "Nsight"),
        ("PHASE 4", "Phase 4"),
    ):
        rejects(f"a smoke notice missing {missing!r} is rejected",
                validate_smoke_recipe(
                    [l.replace(missing, "REMOVED") for l in _GOOD_SMOKE_RECIPE]), fragment)

    # --- Status documents -----------------------------------------------------
    good_plan = (
        "| P3.1 | Pinned official CuTe DSL example | YES | YES | YES |\n"
        "| P3.2 | One-shape wrapper | YES | YES | YES |\n"
        "| P3.3 | cuBLASLt baseline | YES | YES | YES |\n"
        "| P3.4 | Three execution variants | YES | YES | YES |\n"
        "| P3.5 | Five shapes and comparison | YES | YES | YES |\n"
        "| P4.1 | Orchestrator | YES | YES | YES |\n"
        "| P4.2 | Pilot plus three final campaigns | YES | YES | YES |\n"
        "| P4.3 | Integrated analysis, documentation, audit | YES | YES | YES |\n"
    )
    good_protocol = (
        "Status: `P3.5 = YES / YES / YES`.\n"
        "P3.5 creates no publishable performance result.\n"
        "P3.5 was independently audited after remediation.\n"
        "P3.5 was verified on GB300.\n"
        "P3.5 is closed. Phase 3 is closed.\n"
    )
    good_readme = (
        "P3.5 (five shapes and comparison) is closed. "
        "P3.5: CLOSED; Phase 3: CLOSED.\n"
    )
    check("closed status documents are accepted",
          validate_status_documents(good_plan, good_protocol, good_readme) == [],
          str(validate_status_documents(good_plan, good_protocol, good_readme)))
    for wrong in FORBIDDEN_P35_STATUS_LINES:
        rejects(f"a PLAN.md recording {wrong!r} is rejected",
                validate_status_documents(
                    good_plan.replace(EXPECTED_P35_STATUS_LINE, wrong),
                    good_protocol, good_readme))
    rejects("a PLAN.md that weakens a closed unit is rejected",
            validate_status_documents(
                good_plan.replace("| P3.2 | One-shape wrapper | YES | YES | YES |",
                                  "| P3.2 | One-shape wrapper | YES | NO | NO |"),
                good_protocol, good_readme), "no longer records")
    rejects("a PLAN.md that regresses P4.2 to unimplemented is rejected",
            validate_status_documents(
                good_plan.replace("| P4.2 | Pilot plus three final campaigns | YES | YES | YES |",
                                  "| P4.2 | Pilot plus three final campaigns | NO | NO | NO |"),
                good_protocol, good_readme), "Phase 4 frontier row")
    rejects("a PLAN.md that regresses closed P4.2 to implemented-only is rejected",
            validate_status_documents(
                good_plan.replace("| P4.2 | Pilot plus three final campaigns | YES | YES | YES |",
                                  "| P4.2 | Pilot plus three final campaigns | YES | NO | NO |"),
                good_protocol, good_readme), "Phase 4 frontier row")
    rejects("a PLAN.md that reopens accepted P4.3 to implemented-only is rejected",
            validate_status_documents(
                good_plan.replace(
                    "| P4.3 | Integrated analysis, documentation, audit | YES | YES | YES |",
                    "| P4.3 | Integrated analysis, documentation, audit | YES | NO | NO |"),
                good_protocol, good_readme), "Phase 4 frontier row")
    rejects("a PLAN.md that still calls P4.3 unimplemented is rejected",
            validate_status_documents(
                good_plan.replace(
                    "| P4.3 | Integrated analysis, documentation, audit | YES | YES | YES |",
                    "| P4.3 | Integrated analysis, documentation, audit | NO | NO | NO |"),
                good_protocol, good_readme), "Phase 4 frontier row")
    # Each stale or impossible Phase 4 row is substituted for the row it owns,
    # so the rejection is caused by that row and not by an unrelated absence.
    for premature in FORBIDDEN_PHASE4_STATUS_LINES:
        if "| P4.1 |" in premature:
            owned = "| P4.1 | Orchestrator | YES | YES | YES |"
        elif "| P4.2 |" in premature:
            owned = "| P4.2 | Pilot plus three final campaigns | YES | YES | YES |"
        else:
            owned = "| P4.3 | Integrated analysis, documentation, audit | YES | YES | YES |"
        rejects(f"a PLAN.md recording the stale or invalid Phase 4 status {premature!r} "
                f"is rejected",
                validate_status_documents(
                    good_plan.replace(owned, premature), good_protocol, good_readme))
    rejects("a stale pre-closure protocol status is rejected",
            validate_status_documents(
                good_plan,
                good_protocol.replace("P3.5 = YES / YES / YES", "P3.5 = YES / NO / NO"),
                good_readme), "stale status")
    for claim, fragment in (
        ("P3.5 was independently audited after remediation.\n", "independently audited"),
        ("P3.5 was verified on GB300.\n", "verified on GB300"),
        ("P3.5 is closed. ", "P3.5 is closed"),
        ("Phase 3 is closed.\n", "Phase 3 is closed"),
    ):
        rejects(f"a protocol omitting {fragment!r} is rejected",
                validate_status_documents(
                    good_plan, good_protocol.replace(claim, ""), good_readme),
                "closure fact")
    rejects("a protocol retaining the pre-audit claim is rejected",
            validate_status_documents(
                good_plan,
                good_protocol + "No independent audit of P3.5 has been performed.\n",
                good_readme),
            "stale status")
    rejects("a protocol without the non-publishable statement is rejected",
            validate_status_documents(
                good_plan,
                good_protocol.replace(
                    "P3.5 creates no publishable performance result.", ""),
                good_readme), "no publishable result")
    rejects("a README that omits the Phase 3 closure is rejected",
            validate_status_documents(
                good_plan, good_protocol,
                good_readme.replace("Phase 3: CLOSED", "Phase 3: OPEN")),
            "closure status")
    rejects("a README that omits the P3.5 closure is rejected",
            validate_status_documents(
                good_plan, good_protocol,
                good_readme.replace("P3.5: CLOSED", "P3.5: OPEN")),
            "closure status")
    rejects("a README that never describes P3.5 is rejected",
            validate_status_documents(good_plan, good_protocol, ""), "does not describe P3.5")

    # --- New pins -------------------------------------------------------------
    check("reusing the existing pins is accepted",
          validate_no_new_pins(
              {"CUDA_VERSION": "13.1.0"},
              {key: "x" for key in REQUIRED_P31_CONTRACT_KEYS + REQUIRED_P34_CONTRACT_KEYS}) == [])
    rejects("a new P3.5 pin is rejected",
            validate_no_new_pins(
                {},
                {**{key: "x" for key in
                    REQUIRED_P31_CONTRACT_KEYS + REQUIRED_P34_CONTRACT_KEYS},
                 "CUTEDSL_P35_EXAMPLE_PATH": "x"}),
            "needs no new pin")
    rejects("a Phase 3 key added to the global contract is rejected",
            validate_no_new_pins(
                {"CUTEDSL_P31_EXAMPLE_PATH": "x"},
                {key: "x" for key in REQUIRED_P31_CONTRACT_KEYS + REQUIRED_P34_CONTRACT_KEYS}),
            "gained a Phase 3 key")

    # --- Make helpers ---------------------------------------------------------
    variables = parse_make_variables("A := src/gemm\nB := $(A)/gemm_comparison.py\n")
    check("simply expanded Make variables are parsed",
          variables.get("B") == "src/gemm/gemm_comparison.py", str(variables))
    makefile = "a-target: dep\n\t@echo one\n\t@echo two\n\nother:\n\t@echo three\n"
    check("recipe extraction stops at the next target",
          extract_make_recipe(makefile, "a-target") == ["\t@echo one", "\t@echo two"])
    check("a missing target yields no recipe", extract_make_recipe(makefile, "absent") == [])

    if failures:
        print(f"SELF-TEST: FAIL ({len(failures)} case(s))", file=sys.stderr)
        return 1
    print("SELF-TEST: PASS", file=sys.stderr)
    return 0


# --- Entry point -------------------------------------------------------------


def main(argv=None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)

    if arguments == ["--self-test"]:
        return run_self_test()
    if (arguments and arguments[0].startswith("-")) or len(arguments) > 1:
        print("usage: check_gemm_comparison_p35.py [repository-root] | --self-test",
              file=sys.stderr)
        return 2

    root = Path(arguments[0]) if arguments else Path(__file__).resolve().parents[1]
    if not root.is_dir():
        print(f"check_gemm_comparison_p35: {root} is not a directory", file=sys.stderr)
        return 2

    print(f"check_gemm_comparison_p35: checking {root}", file=sys.stderr)
    try:
        errors = check_wrapper(root)
    except Exception as exc:  # noqa: BLE001 - a checker crash is a failed check
        print(f"check_gemm_comparison_p35: the check itself failed: {exc}", file=sys.stderr)
        return 1

    if errors:
        print(f"check_gemm_comparison_p35: FAIL ({len(errors)} finding(s))", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("check_gemm_comparison_p35: OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
