#!/usr/bin/env python3
"""GPU-free contract checker for the P3.3 cuBLASLt GEMM baseline.

This checker is deliberately independent of ``src/gemm/cublaslt_gemm.py`` and
of ``src/gemm/cublaslt_bridge.cu``: it carries its own copy of the frozen P3.3
configuration, its own copy of the frozen 77-field CSV schema, and its own row
validator. A drift in the wrapper, in the bridge, or in this checker therefore
shows up as a disagreement between three independent statements of the same
contract, rather than as three copies of one mistake.

It uses only the Python standard library and never initializes CUDA. The only
subprocesses it starts are Python interpreters running the wrapper's ``--help``
and ``--self-test`` behind an import guard that makes any attempt to import
PyTorch, CuTe DSL, the CUDA bindings, or even ``ctypes`` a hard failure - which
is how "``--help`` and ``--self-test`` are GPU-free and never load the shared
bridge" is proved rather than assumed - plus, when the compiled bridge is
present, the ELF/symbol tools that inspect it.

What it validates:

* the wrapper's and the bridge's frozen configurations match the frozen P3.3
  descriptor contract exactly, and match each other;
* the CSV field names and their order match the frozen schema exactly, and the
  P3.2 ``p32.v1`` schema is neither reused nor reinterpreted;
* one synthetic valid row serializes to exactly one header and one data row;
* missing, duplicate, unknown, non-finite, non-canonical, and wrongly typed
  fields are all rejected, by the wrapper's validator and by this checker's own;
* ``publishable`` is fixed to ``false`` and a successful row is always
  ``correctness=PASS``; a failed or skipped check cannot build a row at all;
* the command line exposes no shape, type, layout, transpose, leading
  dimension, alpha, beta, epilogue, workspace, heuristic, algorithm,
  cache-mode, or publication control, and no way to skip the reference check;
* the measured path is a direct ``cublasLtMatmul`` call, and no fallback to
  ``cublasGemmEx``, an ordinary cuBLAS GEMM, or a Torch matmul exists;
* the bridge benchmarks no candidate, exposes no timing API, and writes nothing
  to stdout;
* the Make targets keep stdout a pure data stream, validate the GPU index
  before any work, and never claim success after a failed inner command;
* the wrapper writes no result file and creates no campaign directory, and
  implements no P3.4/P3.5 functionality.

Usage:
  check_cublaslt_gemm_p33.py [repository-root]
  check_cublaslt_gemm_p33.py --self-test

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
import tempfile
from pathlib import Path

# --- Independent frozen expectations ----------------------------------------

WRAPPER_RELATIVE_PATH = "src/gemm/cublaslt_gemm.py"
BRIDGE_RELATIVE_PATH = "src/gemm/cublaslt_bridge.cu"
CHECKER_RELATIVE_PATH = "scripts/check_cublaslt_gemm_p33.py"
PROTOCOL_RELATIVE_PATH = "src/gemm/P3_3_PROTOCOL.md"
P32_WRAPPER_RELATIVE_PATH = "src/gemm/cutedsl_gemm.py"
GLOBAL_CONTRACT_FILE = "VERSIONS.env"
PHASE3_CONTRACT_FILE = "PHASE3_VERSIONS.env"

# The compiled bridge, at the fixed container-private location the Make targets
# build it into. Absent on a bare host checkout; present inside the gate.
BRIDGE_LIBRARY_PATH = Path("/tmp/p33-bridge/libp33_cublaslt_bridge.so")

EXPECTED_CSV_FIELDS = (
    "schema_version",
    "experiment",
    "unit",
    "run_kind",
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
    "seed",
    "reference",
    "atol",
    "rtol",
    "correctness",
    "max_abs_error",
    "max_rel_error",
    "setup_time_ms",
    "first_launch_ms",
    "kernel_time_ms",
    "warmup_iterations",
    "iterations",
    "cache_mode",
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
    "gpu_name",
    "gpu_uuid",
    "compute_capability",
    "driver_version",
    "cuda_toolkit_version",
    "torch_cuda_version",
    "cutedsl_version",
    "cublaslt_version",
    "cutlass_commit",
    "upstream_example_sha256",
    "git_commit",
    "git_dirty",
    "publishable",
)

EXPECTED_FROZEN_CONFIG = {
    "schema_version": "p33.v1",
    "experiment": "exp03_cutedsl_vs_cublaslt",
    "unit": "P3.3",
    "run_kind": "smoke",
    "method": "cublaslt",
    "variant": "heuristic_first_supported",
    "m": 4096,
    "n": 4096,
    "k": 4096,
    "l": 1,
    "ab_dtype": "BFloat16",
    "acc_dtype": "Float32",
    "c_dtype": "Float32",
    "a_major": "k",
    "b_major": "k",
    "c_major": "n",
    "order_a": "CUBLASLT_ORDER_ROW",
    "order_b": "CUBLASLT_ORDER_ROW",
    "order_c": "CUBLASLT_ORDER_ROW",
    "order_d": "CUBLASLT_ORDER_ROW",
    "transa": "CUBLAS_OP_N",
    "transb": "CUBLAS_OP_T",
    "lda": 4096,
    "ldb": 4096,
    "ldc": 4096,
    "ldd": 4096,
    "compute_type": "CUBLAS_COMPUTE_32F",
    "scale_type": "CUDA_R_32F",
    "pointer_mode": "CUBLASLT_POINTER_MODE_HOST",
    "epilogue": "CUBLASLT_EPILOGUE_DEFAULT",
    "alpha": 1.0,
    "beta": 0.0,
    "seed": 1111,
    "reference": "torch_cuda_fp32_ieee",
    "atol": 1e-1,
    "rtol": 1e-5,
    "cache_mode": "hot",
    "workspace_limit_bytes": 67108864,
    "heuristic_requested": 32,
    "publishable": False,
}

# The P3.2 schema this unit must never reuse or reinterpret.
P32_SCHEMA_VERSION = "p32.v1"
P32_TIMING_FIELD = "compile_time_ms"

EXPECTED_CUDA_ARCH = "sm_103a"
EXPECTED_COMPUTE_CAPABILITY = "10.3"

EXPECTED_WORKSPACE_LIMIT_BYTES = 67108864
EXPECTED_HEURISTIC_REQUESTED = 32
EXPECTED_SEARCH_MODE = "CUBLASLT_SEARCH_BEST_FIT"

EXPECTED_FIXED_ROW_VALUES = {
    "schema_version": "p33.v1",
    "experiment": "exp03_cutedsl_vs_cublaslt",
    "unit": "P3.3",
    "run_kind": "smoke",
    "method": "cublaslt",
    "variant": "heuristic_first_supported",
    "m": "4096",
    "n": "4096",
    "k": "4096",
    "l": "1",
    "ab_dtype": "BFloat16",
    "acc_dtype": "Float32",
    "c_dtype": "Float32",
    "a_major": "k",
    "b_major": "k",
    "c_major": "n",
    "order_a": "CUBLASLT_ORDER_ROW",
    "order_b": "CUBLASLT_ORDER_ROW",
    "order_c": "CUBLASLT_ORDER_ROW",
    "order_d": "CUBLASLT_ORDER_ROW",
    "transa": "CUBLAS_OP_N",
    "transb": "CUBLAS_OP_T",
    "lda": "4096",
    "ldb": "4096",
    "ldc": "4096",
    "ldd": "4096",
    "compute_type": "CUBLAS_COMPUTE_32F",
    "scale_type": "CUDA_R_32F",
    "pointer_mode": "CUBLASLT_POINTER_MODE_HOST",
    "epilogue": "CUBLASLT_EPILOGUE_DEFAULT",
    "alpha": "1.000000000",
    "beta": "0.000000000",
    "seed": "1111",
    "reference": "torch_cuda_fp32_ieee",
    "atol": "0.100000000",
    "rtol": "0.000010000",
    "correctness": "PASS",
    "cache_mode": "hot",
    "workspace_limit_bytes": "67108864",
    "heuristic_requested": "32",
    "publishable": "false",
}

TIMING_FIELDS = ("setup_time_ms", "first_launch_ms", "kernel_time_ms")
ERROR_FIELDS = ("max_abs_error", "max_rel_error")
TOLERANCE_FIELDS = ("atol", "rtol")
SCALAR_FIELDS = ("alpha", "beta")
COUNT_FIELDS = ("warmup_iterations", "iterations")
BOOL_FIELDS = ("git_dirty", "publishable")
NONNEGATIVE_INT_FIELDS = (
    "workspace_bytes",
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
    "cublaslt_version",
)
POSITIVE_INT_FIELDS = (
    "alignment_a_bytes",
    "alignment_b_bytes",
    "alignment_c_bytes",
    "alignment_d_bytes",
    "heuristic_returned",
)
ALIGNMENT_FIELDS = (
    "alignment_a_bytes",
    "alignment_b_bytes",
    "alignment_c_bytes",
    "alignment_d_bytes",
)
TIMING_DECIMALS = 6
ERROR_DECIMALS = 9
TOLERANCE_DECIMALS = 9
SCALAR_DECIMALS = 9
WAVES_DECIMALS = 6

MIN_ITERATIONS = 1
MAX_WARMUP_ITERATIONS = 100
MAX_ITERATIONS = 100
SMOKE_WARMUP_ITERATIONS = 2
SMOKE_ITERATIONS = 10

ALLOWED_CLI_OPTIONS = frozenset(
    {"--help", "--self-test", "--warmup-iterations", "--iterations"}
)

# Option spellings that would reopen a frozen property of the descriptor
# contract or of the algorithm policy. Checked in both the dashed and the
# underscored spelling.
FORBIDDEN_CLI_FRAGMENTS = (
    "mnkl",
    "shape",
    "dtype",
    "major",
    "order",
    "trans",
    "lda",
    "ldb",
    "ldc",
    "ldd",
    "leading",
    "alpha",
    "beta",
    "epilogue",
    "workspace",
    "heuristic",
    "algo",
    "tile",
    "stages",
    "split",
    "swizzl",
    "cluster",
    "autotune",
    "search",
    "skip-ref",
    "skip_ref",
    "cold-l2",
    "cold_l2",
    "cache",
    "publish",
    "tolerance",
    "atol",
    "rtol",
    "seed",
    "variant",
    "method",
    "gpu",
    "device",
    "persistent",
    "2cta",
)

# Identifier fragments that must not appear as code in the wrapper. Prose in
# docstrings and comments is exempt: the scan runs over Python NAME tokens
# only, so a sentence explaining that P3.3 computes no TFLOP/s and makes no
# comparison is fine, while a tflops variable or a campaign_dir variable is
# not. "cublas" is deliberately NOT banned here - this unit is the cuBLASLt
# baseline - and the forbidden vendor entry points are handled structurally
# below instead.
FORBIDDEN_SOURCE_IDENTIFIERS = (
    "tflop",
    "speedup",
    "efficiency",
    "bandwidth",
    "utilization",
    "winner",
    "nsight",
    "autotune",
    "campaign",
    "skip_ref_check",
    "use_cold_l2",
    "dense_gemm_persistent",
)

# Vendor and framework entry points that would mean the measured path is not a
# direct cublasLtMatmul call. Checked as whole identifiers in both the wrapper
# and the bridge.
FORBIDDEN_GEMM_ENTRY_POINTS = (
    "cublasGemmEx",
    "cublasGemmBatchedEx",
    "cublasGemmStridedBatchedEx",
    "cublasSgemm",
    "cublasHgemm",
    "cublasSgemmEx",
    "cublasGemmGroupedBatchedEx",
    "cublasLtMatmulAlgoGetIds",
    "cublasLtMatmulAlgoInit",
)

# Torch/framework operators that must never be the measured baseline. The
# untimed FP32 oracle deliberately uses torch.einsum, which is not in this set.
FORBIDDEN_TORCH_MATMUL_ATTRS = frozenset(
    {"matmul", "mm", "bmm", "addmm", "baddbmm", "addbmm", "linear", "dot"}
)

# The bridge must call this exactly once and must contain no timing facility.
REQUIRED_BRIDGE_CALL = "cublasLtMatmul"
FORBIDDEN_BRIDGE_TIMING = (
    "cudaEventRecord",
    "cudaEventElapsedTime",
    "std::chrono",
    "clock_gettime",
    "gettimeofday",
    "nvtxRange",
)
# The bridge must never write to a standard stream.
FORBIDDEN_BRIDGE_OUTPUT = (
    r"(?<![a-zA-Z0-9_])printf\s*\(",
    r"(?<![a-zA-Z0-9_])puts\s*\(",
    r"(?<![a-zA-Z0-9_])fputs\s*\(",
    r"std::cout",
    r"std::cerr",
    r"std::clog",
    r"fprintf\s*\(\s*std(out|err)",
)

# The frozen `static const` declarations the bridge must carry, independently
# of whatever the Python wrapper says.
EXPECTED_BRIDGE_CONSTANTS = {
    "P33_M": "4096",
    "P33_N": "4096",
    "P33_K": "4096",
    "P33_BATCH_COUNT": "1",
    "P33_LDA": "P33_K",
    "P33_LDB": "P33_K",
    "P33_LDC": "P33_N",
    "P33_LDD": "P33_N",
    "P33_TRANSA": "CUBLAS_OP_N",
    "P33_TRANSB": "CUBLAS_OP_T",
    "P33_ORDER": "CUBLASLT_ORDER_ROW",
    "P33_AB_TYPE": "CUDA_R_16BF",
    "P33_CD_TYPE": "CUDA_R_32F",
    "P33_COMPUTE_TYPE": "CUBLAS_COMPUTE_32F",
    "P33_SCALE_TYPE": "CUDA_R_32F",
    "P33_POINTER_MODE": "CUBLASLT_POINTER_MODE_HOST",
    "P33_EPILOGUE": "CUBLASLT_EPILOGUE_DEFAULT",
    "P33_ALPHA": "1.0f",
    "P33_BETA": "0.0f",
    "P33_WORKSPACE_LIMIT_BYTES": "67108864ULL",
    "P33_HEURISTIC_REQUESTED": "32",
    "P33_SEARCH_MODE": "CUBLASLT_SEARCH_BEST_FIT",
}

# cuBLASLt calls the bridge must make, so that the auditable policy is present
# rather than merely documented.
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

# The nine selected-algorithm configuration attributes that must be recorded.
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
REQUIRED_ALGO_CONFIG_WIDTHS = {
    "CUBLASLT_ALGO_CONFIG_ID": "int32_t",
    "CUBLASLT_ALGO_CONFIG_TILE_ID": "uint32_t",
    "CUBLASLT_ALGO_CONFIG_STAGES_ID": "uint32_t",
    "CUBLASLT_ALGO_CONFIG_SPLITK_NUM": "uint32_t",
    "CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME": "uint32_t",
    "CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING": "uint32_t",
    "CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION": "uint32_t",
    "CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID": "uint16_t",
    "CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID": "uint16_t",
}

# The four minimum-alignment preferences that must be derived from the real
# device pointers.
REQUIRED_ALIGNMENT_PREFERENCES = (
    "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES",
    "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES",
    "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES",
    "CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES",
)

# Modules whose import would mean the "GPU-free" claim is false. ctypes is
# included because loading the compiled bridge is exactly what --help and
# --self-test must never do.
GPU_STACK_MODULES = ("torch", "cutlass", "cuda", "numpy", "pynvml", "ctypes")

# Functionality that belongs to P3.4, P3.5, or Phase 4 and must not appear.
FORBIDDEN_LATER_UNIT_IDENTIFIERS = (
    "persistent",
    "use_2cta_instrs",
    "sweep",
    "comparison",
    "aggregate",
    "manifest",
)

_RE_HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")
_RE_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_RE_GPU_UUID = re.compile(r"\AGPU-[0-9a-fA-F][0-9a-fA-F-]+\Z")
_RE_DOTTED_VERSION = re.compile(r"\A[0-9]+(\.[0-9]+)*\Z")
_RE_COMPUTE_CAPABILITY = re.compile(r"\A[0-9]+\.[0-9]+\Z")
_RE_POSITIVE_INT = re.compile(r"\A[1-9][0-9]*\Z")
_RE_NONNEGATIVE_INT = re.compile(r"\A(0|[1-9][0-9]*)\Z")
_RE_ENV_LINE = re.compile(r"\A([A-Z][A-Z0-9_]*)=(\S*)\Z")
_RE_SAFE_TEXT = re.compile(r"\A[^\x00-\x1f\x7f]+\Z")

# P3.3 retains the P3.2 FP32 oracle policy: exclusively the PyTorch 2.10
# fp32_precision API. In 2.10 the legacy allow_tf32 property is an alias of the
# same setting, mixing the two is unsupported, and the last write silently
# wins, so any appearance of the legacy spellings - or of an overlapping global
# precision API - is a contract failure.
FORBIDDEN_FP32_API_SPELLINGS = (
    "allow_tf32",
    "set_float32_matmul_precision",
    "get_float32_matmul_precision",
    "cudnn",
)
REQUIRED_FP32_API_SPELLINGS = ("fp32_precision",)
FP32_PRECISION_IEEE = "ieee"
FP32_PRECISION_UNSET = "none"

# Constructs that would let a Make target silently drop unexpected stdout
# instead of letting it surface as a contract violation.
FORBIDDEN_SMOKE_FILTERS = ("grep -v", "| sed", "| tail", "| head", "| awk", "| grep")

SMOKE_TARGET = "gemm-cublaslt-p33-smoke"
CHECK_TARGET = "gemm-cublaslt-p33-check"
CHECK_PREREQUISITE = "gemm-cutedsl-p32-check"
LAUNCHER_RELATIVE_PATH = "scripts/run_container.sh"
LAUNCHER_DATA_MODE_VARIABLE = "RUN_CONTAINER_STDOUT_IS_DATA"
GPU_INDEX_VARIABLE = "BLACKWELL_GPU_INDEX"
SMOKE_SUCCESS_SENTENCE = (
    "P3.3 smoke completed: correctness passed before warm-up and steady-state timing."
)

# The subprocess guard: any import of the GPU stack (or of ctypes, which is how
# the bridge would be loaded) aborts the child.
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

# Drives the real main() with execute_measurement replaced by a correctness
# failure, proving the no-CSV contract through the real descriptor plumbing.
CORRECTNESS_FAILURE_PROBE = (
    _GUARD_PRELUDE
    + """
import importlib.util

spec = importlib.util.spec_from_file_location("p33_probe", {wrapper!r})
module = importlib.util.module_from_spec(spec)
sys.modules["p33_probe"] = module
spec.loader.exec_module(module)


def _raise_correctness_failure(*args, **kwargs):
    raise module.CorrectnessError("synthetic mismatch: 1 element exceeds atol/rtol")


module.execute_measurement = _raise_correctness_failure
sys.exit(module.main([]))
"""
)


# --- Pure validators ---------------------------------------------------------


def validate_csv_schema(fields) -> list:
    """Check the frozen field list, its order, and its absence of metrics."""
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
    for field in fields:
        if re.search(r"tflop|speedup|efficien|bandwidth|utilization|throughput|winner", field):
            errors.append(f"the CSV schema exposes a performance metric: {field}")
    if P32_TIMING_FIELD in tuple(fields):
        errors.append(
            f"the P3.3 schema reuses the P3.2 field name {P32_TIMING_FIELD}; cuBLASLt setup "
            "is not compilation"
        )
    if "setup_time_ms" not in tuple(fields):
        errors.append("the P3.3 schema has no setup_time_ms field")
    return errors


def validate_frozen_config(config) -> list:
    """Check the frozen configuration mapping against this checker's copy."""
    errors = []
    if not isinstance(config, dict):
        return ["the frozen configuration is not a mapping"]
    for key, expected in sorted(EXPECTED_FROZEN_CONFIG.items()):
        if key not in config:
            errors.append(f"the frozen configuration is missing {key}")
            continue
        actual = config[key]
        if isinstance(expected, float):
            if not isinstance(actual, float) or not math.isclose(
                actual, expected, rel_tol=0.0, abs_tol=0.0
            ):
                errors.append(f"frozen {key} is {actual!r}, expected {expected!r}")
        elif isinstance(expected, bool):
            if actual is not expected:
                errors.append(f"frozen {key} is {actual!r}, expected {expected!r}")
        elif actual != expected:
            errors.append(f"frozen {key} is {actual!r}, expected {expected!r}")
    unknown = sorted(set(config) - set(EXPECTED_FROZEN_CONFIG))
    if unknown:
        errors.append(f"the frozen configuration has unexpected key(s): {', '.join(unknown)}")
    if config.get("schema_version") == P32_SCHEMA_VERSION:
        errors.append("the P3.3 unit reuses the P3.2 schema version")
    return errors


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

    for field, expected in sorted(EXPECTED_FIXED_ROW_VALUES.items()):
        if row[field] != expected:
            errors.append(f"{field}: {row[field]!r} != frozen {expected!r}")

    for field in BOOL_FIELDS:
        if row[field] not in ("true", "false"):
            errors.append(f"{field}: {row[field]!r} is not a canonical lowercase boolean")

    for field in COUNT_FIELDS:
        if not _RE_POSITIVE_INT.match(row[field]):
            errors.append(f"{field}: {row[field]!r} is not a positive integer")
        else:
            maximum = MAX_WARMUP_ITERATIONS if field == "warmup_iterations" else MAX_ITERATIONS
            if not MIN_ITERATIONS <= int(row[field]) <= maximum:
                errors.append(f"{field}: {row[field]} is outside [{MIN_ITERATIONS}, {maximum}]")

    for field in TIMING_FIELDS:
        _validate_decimal_field(field, row[field], TIMING_DECIMALS, True, errors)
    for field in ERROR_FIELDS:
        _validate_decimal_field(field, row[field], ERROR_DECIMALS, False, errors)
    for field in TOLERANCE_FIELDS:
        _validate_decimal_field(field, row[field], TOLERANCE_DECIMALS, True, errors)
    _validate_decimal_field("alpha", row["alpha"], SCALAR_DECIMALS, True, errors)
    _validate_decimal_field("beta", row["beta"], SCALAR_DECIMALS, False, errors)
    _validate_decimal_field("waves_count", row["waves_count"], WAVES_DECIMALS, False, errors)

    for field in NONNEGATIVE_INT_FIELDS:
        if not _RE_NONNEGATIVE_INT.match(row[field]):
            errors.append(f"{field}: {row[field]!r} is not a canonical non-negative integer")
    for field in POSITIVE_INT_FIELDS:
        if not _RE_POSITIVE_INT.match(row[field]):
            errors.append(f"{field}: {row[field]!r} is not a canonical positive integer")

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
    for field in ALIGNMENT_FIELDS:
        if _RE_POSITIVE_INT.match(row[field]):
            value = int(row[field])
            if value & (value - 1):
                errors.append(f"{field}: {value} is not a power of two")

    if not _RE_HEX40.match(row["cutlass_commit"]):
        errors.append(f"cutlass_commit: {row['cutlass_commit']!r} is not a 40-hex commit")
    if not _RE_HEX40.match(row["git_commit"]):
        errors.append(f"git_commit: {row['git_commit']!r} is not a 40-hex commit")
    if not _RE_HEX64.match(row["upstream_example_sha256"]):
        errors.append("upstream_example_sha256 is not a 64-hex digest")
    if not _RE_GPU_UUID.match(row["gpu_uuid"]):
        errors.append(f"gpu_uuid: {row['gpu_uuid']!r} is malformed")
    if not _RE_COMPUTE_CAPABILITY.match(row["compute_capability"]):
        errors.append(f"compute_capability: {row['compute_capability']!r} is malformed")
    for field in ("driver_version", "cuda_toolkit_version", "torch_cuda_version",
                  "cutedsl_version"):
        if not _RE_DOTTED_VERSION.match(row[field]):
            errors.append(f"{field}: {row[field]!r} is not a dotted version")
    return errors


def validate_serialized_output(text) -> list:
    """Require exactly one header line and exactly one data row."""
    errors = []
    if not isinstance(text, str):
        return ["the serialized output is not a string"]
    lines = text.splitlines()
    if len(lines) != 2:
        errors.append(f"the serialized output has {len(lines)} line(s), expected exactly 2")
        return errors
    if lines[0] != ",".join(EXPECTED_CSV_FIELDS):
        errors.append("the CSV header does not match the frozen field order")
    parsed = list(csv.DictReader(io.StringIO(text)))
    if len(parsed) != 1:
        errors.append(f"the serialized output parses to {len(parsed)} row(s), expected 1")
        return errors
    errors.extend(validate_row_mapping(dict(parsed[0])))
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
    for entry_point in FORBIDDEN_GEMM_ENTRY_POINTS:
        if entry_point in names:
            errors.append(f"the wrapper references the forbidden entry point {entry_point}")
    errors.extend(validate_no_framework_matmul(source))
    errors.extend(validate_no_result_files(source))
    return errors


def validate_no_framework_matmul(source) -> list:
    """Reject a framework matmul as the measured path, structurally.

    The untimed FP32 oracle uses ``torch.einsum`` on purpose; that is a
    correctness reference, not a measured method, and it is not matched here.
    Everything that would make a Torch operator the baseline is.
    """
    errors = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"the wrapper is not syntactically valid: {exc}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            errors.append(
                f"line {node.lineno}: the wrapper uses the @ matmul operator; the measured "
                "path must be a direct cublasLtMatmul call"
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_TORCH_MATMUL_ATTRS:
                errors.append(
                    f"line {node.lineno}: the wrapper calls .{node.func.attr}(); a framework "
                    "operation must never be the measured baseline"
                )
    return errors


def _scratch_only_functions(tree) -> set:
    """Return the line ranges of functions that write only into a temp dir.

    A GPU-free self-test helper that writes a synthetic fixture into a
    ``tempfile.TemporaryDirectory()`` and lets it be deleted again creates no
    result file and no campaign directory. Everything else that writes is a
    contract violation, so the exemption is scoped to exactly those functions
    rather than to a filename pattern.
    """
    scratch = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        uses_tempdir = any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Attribute)
            and inner.func.attr == "TemporaryDirectory"
            for inner in ast.walk(node)
        )
        if uses_tempdir:
            for inner in ast.walk(node):
                if hasattr(inner, "lineno"):
                    scratch.add(inner.lineno)
    return scratch


def validate_no_result_files(source) -> list:
    """Reject result-file or campaign-directory creation, structurally."""
    errors = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"the wrapper is not syntactically valid: {exc}"]
    scratch = _scratch_only_functions(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if node.lineno in scratch:
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
                            errors.append(
                                f"line {node.lineno}: the wrapper opens a file for writing"
                            )
    for marker in ("results/raw", "results/preflight"):
        if marker in source:
            errors.append(f"the wrapper references the result tree {marker!r}")
    return errors


def validate_fp32_precision_policy(source) -> list:
    """Require the PyTorch 2.10 fp32_precision API and nothing overlapping.

    The scan is structural: only real attribute accesses, call targets, and
    bare names count. Prose that *explains* why the legacy ``allow_tf32``
    alias and ``set_float32_matmul_precision()`` are never used is therefore
    legal, while a single actual read or write of either is a hard failure -
    which is the distinction that matters, because in PyTorch 2.10 the two are
    aliases of one setting and the last write silently wins.
    """
    errors = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"the wrapper is not syntactically valid: {exc}"]

    forbidden = set(FORBIDDEN_FP32_API_SPELLINGS)
    required_seen = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in forbidden:
                errors.append(
                    f"line {node.lineno}: the wrapper accesses the forbidden FP32 API "
                    f".{node.attr}; only the fp32_precision API may be used"
                )
            if node.attr in REQUIRED_FP32_API_SPELLINGS:
                required_seen.add(node.attr)
        elif isinstance(node, ast.Name) and node.id in forbidden:
            errors.append(
                f"line {node.lineno}: the wrapper references the forbidden FP32 API "
                f"{node.id}"
            )
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            # A getattr()/setattr() string would bypass the attribute scan.
            if node.value in forbidden:
                errors.append(
                    f"line {node.lineno}: the wrapper names the forbidden FP32 API "
                    f"{node.value!r} as a string, which getattr/setattr would honour"
                )
            if node.value in REQUIRED_FP32_API_SPELLINGS:
                required_seen.add(node.value)

    for spelling in REQUIRED_FP32_API_SPELLINGS:
        if spelling not in required_seen:
            errors.append(f"the wrapper never uses the required FP32 API {spelling!r}")
    if f'"{FP32_PRECISION_IEEE}"' not in source and f"'{FP32_PRECISION_IEEE}'" not in source:
        errors.append(f"the wrapper never requires fp32_precision == {FP32_PRECISION_IEEE!r}")
    if FP32_PRECISION_UNSET in source and "reject" not in source.lower():
        errors.append("the wrapper mentions the unset 'none' policy without rejecting it")
    return errors


# --- Bridge validators -------------------------------------------------------


def strip_c_comments(source) -> str:
    """Remove // and /* */ comments so prose cannot satisfy a code check."""
    without_block = re.sub(r"/\*.*?\*/", " ", source, flags=re.S)
    return re.sub(r"//[^\n]*", " ", without_block)


def extract_bridge_constants(source) -> dict:
    """Return the bridge's ``static const`` declarations as name -> literal."""
    code = strip_c_comments(source)
    constants = {}
    pattern = re.compile(
        r"static\s+const\s+[A-Za-z_][A-Za-z0-9_:<>\s\*]*?\s(P33_[A-Z0-9_]+)\s*=\s*([^;]+);"
    )
    for match in pattern.finditer(code):
        constants[match.group(1)] = match.group(2).strip()
    # #define-style constants are recorded too, so a value moved between the
    # two forms is still seen.
    for match in re.finditer(r"#define\s+(P33_[A-Z0-9_]+)\s+([^\n]+)", code):
        constants.setdefault(match.group(1), match.group(2).strip())
    return constants


def validate_bridge_source(source) -> list:
    """Structural checks over the C-ABI bridge."""
    errors = []
    code = strip_c_comments(source)

    constants = extract_bridge_constants(source)
    for name, expected in sorted(EXPECTED_BRIDGE_CONSTANTS.items()):
        if name not in constants:
            errors.append(f"the bridge does not declare the frozen constant {name}")
        elif constants[name] != expected:
            errors.append(
                f"the bridge declares {name} = {constants[name]!r}, expected {expected!r}"
            )

    for call in REQUIRED_BRIDGE_CALLS:
        if not re.search(rf"(?<![A-Za-z0-9_]){re.escape(call)}\s*\(", code):
            errors.append(f"the bridge never calls {call}()")

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
                "API is forbidden"
            )

    for attribute in REQUIRED_ALGO_CONFIG_ATTRIBUTES:
        if attribute not in code:
            errors.append(f"the bridge never records the algorithm attribute {attribute}")
    for attribute, width in REQUIRED_ALGO_CONFIG_WIDTHS.items():
        pattern = (
            rf"p33_read_algo_config\s*<\s*{re.escape(width)}\s*>\s*\("
            rf"\s*[^,\n]+\s*,\s*{re.escape(attribute)}\b"
        )
        if not re.search(pattern, code):
            errors.append(
                f"the bridge does not read {attribute} at its documented {width} width"
            )
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
    if "CUBLASLT_SEARCH_BEST_FIT" not in code:
        errors.append(f"the bridge does not use {EXPECTED_SEARCH_MODE}")
    if "CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES" not in code:
        errors.append("the bridge never sets the workspace limit preference")
    return errors


def validate_shared_object(library_path, run) -> list:
    """Inspect the compiled bridge's dynamic symbols, when it is present.

    ``run`` is injected so the self-test can drive this validator with
    synthetic, adversarial tool output instead of a real shared object.
    """
    errors = []
    defined = run(["nm", "-D", "--defined-only", str(library_path)])
    undefined = run(["nm", "-D", "-u", str(library_path)])
    needed = run(["readelf", "-d", str(library_path)])

    if defined is None or undefined is None or needed is None:
        return [f"cannot inspect the shared object {library_path}"]

    symbols = set(re.findall(r"\b(p33_[a-z0-9_]+)\b", defined))
    for required in ("p33_plan_create", "p33_plan_execute", "p33_plan_destroy",
                     "p33_cublaslt_version", "p33_last_error", "p33_plan_info_size",
                     "p33_bridge_abi_version"):
        if required not in symbols:
            errors.append(f"the shared object does not export {required}")

    if not re.search(rf"(?<![A-Za-z0-9_]){REQUIRED_BRIDGE_CALL}(?![A-Za-z0-9_])", undefined):
        errors.append(
            f"the shared object does not reference {REQUIRED_BRIDGE_CALL}; the measured path "
            "is not a direct cuBLASLt matmul"
        )
    for entry_point in FORBIDDEN_GEMM_ENTRY_POINTS:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(entry_point)}(?![A-Za-z0-9_])",
                     defined + "\n" + undefined):
            errors.append(
                f"the shared object references {entry_point}; a measured fallback is forbidden"
            )
    for library in ("libcublasLt.so", "libcudart.so"):
        if library not in needed:
            errors.append(f"the shared object is not linked against {library}")
    return errors


# --- Make integration --------------------------------------------------------


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


def parse_make_variables(makefile_text) -> dict:
    """Return the Makefile's simple ``NAME := value`` assignments.

    Recipes are read as raw text, so a step written as ``$(GEMM_P33_BRIDGE_DIR)``
    has to be expanded before it can be checked against the path it must use.
    Only ``:=`` (simply expanded) assignments are collected, which is what this
    Makefile uses, and expansion is applied repeatedly so nested references
    resolve.
    """
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
    """Check the smoke target's stdout, ordering, and exit-status contract.

    ``variables`` carries the Makefile's ``:=`` assignments so that a recipe
    step written as ``$(GEMM_P33_BRIDGE_DIR)`` is checked against the path it
    actually expands to, rather than against its unexpanded spelling.
    """
    errors = []
    if not recipe:
        return [f"the {SMOKE_TARGET} recipe is empty or was not found"]

    logical = logical_recipe_lines(recipe)
    joined = "\n".join(recipe)
    if variables:
        joined = expand_make_variables(joined, variables)
        logical = [expand_make_variables(block, variables) for block in logical]

    # 1. Make must not echo the recipe onto stdout.
    for block in logical:
        body = block.splitlines()[0].lstrip("\t")
        if not body.startswith("@"):
            errors.append(f"recipe line is not quiet (missing @): {body[:70]!r}")

    # 2. Every human-readable echo goes to stderr.
    for line in recipe:
        for match in re.finditer(r"\becho\b", line):
            tail = line[match.end():]
            if ">&2" not in tail:
                errors.append(f"an echo is not redirected to stderr: {line.strip()[:70]!r}")
                break

    # 3. The very first recipe step validates the GPU index, before Docker, any
    #    compilation, or any other work can begin.
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

    # 4. The launcher runs in its opt-in data-stream mode, through the audited
    #    launcher, and never through Docker directly.
    if f"{LAUNCHER_DATA_MODE_VARIABLE}=1" not in joined:
        errors.append(
            f"the smoke target does not set {LAUNCHER_DATA_MODE_VARIABLE}=1, so launcher "
            "and entrypoint text would contaminate stdout"
        )
    if LAUNCHER_RELATIVE_PATH not in joined:
        errors.append(f"the smoke target does not go through {LAUNCHER_RELATIVE_PATH}")
    if re.search(r"(?<![\w/-])docker\s+run\b", joined):
        errors.append("the smoke target invokes Docker directly")

    # 5. The bridge is compiled inside the already-selected GPU container, into
    #    container-private /tmp, and the upstream identity is re-checked there.
    if "nvcc" not in joined:
        errors.append("the smoke target never compiles the cuBLASLt bridge")
    if "/tmp/" not in joined:
        errors.append("the smoke target does not build into container-private /tmp")
    if "rev-parse HEAD" not in joined:
        errors.append("the smoke target does not revalidate the pinned CUTLASS commit")
    if "sha256sum" not in joined:
        errors.append("the smoke target does not revalidate the upstream SHA-256")

    # 6. Exactly the frozen non-publishable iteration counts. The trailing
    #    boundary matters: without it, "--iterations 1000" would satisfy a
    #    substring test for "--iterations 10".
    if not re.search(rf"--warmup-iterations {SMOKE_WARMUP_ITERATIONS}(?![0-9])", joined):
        errors.append(f"the smoke target does not use exactly {SMOKE_WARMUP_ITERATIONS} warm-ups")
    if not re.search(rf"(?<!-)--iterations {SMOKE_ITERATIONS}(?![0-9])", joined):
        errors.append(
            f"the smoke target does not use exactly {SMOKE_ITERATIONS} measured launches"
        )

    # 7. The exit status is captured and preserved.
    if "|| status=$$?" not in joined:
        errors.append("the smoke target does not capture the launcher exit status")
    if not joined.rstrip().endswith("exit $$status"):
        errors.append("the smoke target does not end by preserving the exit status")

    # 8. No unconditional success claim.
    if SMOKE_SUCCESS_SENTENCE not in joined:
        errors.append("the smoke target never reports success accurately")
    else:
        guard = re.search(r'if \[ "\$\$status" -eq 0 \]; then', joined)
        if guard is None:
            errors.append("the success statement is not guarded by a zero exit status")
        elif joined.index(SMOKE_SUCCESS_SENTENCE) < guard.end():
            errors.append("the success statement is printed before the exit-status guard")
    for claim in ("Correctness passed before any timing ran", "compilation succeeded"):
        if claim in joined:
            errors.append(f"the smoke target makes the unconditional claim {claim!r}")

    # 9. The non-publishable notice must be present.
    if "NOT A PERFORMANCE COMPARISON" not in joined.upper():
        errors.append(
            "the smoke target does not state that this is not a performance comparison"
        )

    # 10. No filtering that could silently swallow unexpected stdout.
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
        ("nvcc", "the bridge compilation"),
        ("-lcublasLt", "a direct link against cuBLASLt"),
        ("-lcudart", "a direct link against the CUDA runtime"),
        ("-std=c++17", "C++17"),
        ("-O3", "-O3"),
        ("-lineinfo", "-lineinfo"),
        ("--self-test", "the GPU-free self-tests"),
        ("py_compile", "the Python syntax check"),
        ("nm -D", "an ELF symbol inspection"),
    ):
        if required not in joined:
            errors.append(f"{CHECK_TARGET} does not use {required!r} ({description})")

    if "--gpus" in joined:
        errors.append(f"{CHECK_TARGET} exposes a GPU; the gate must be GPU-free")
    if "PYTHONPYCACHEPREFIX" not in joined:
        errors.append(f"{CHECK_TARGET} does not redirect bytecode out of the read-only mount")
    if REQUIRED_BRIDGE_CALL not in joined:
        errors.append(
            f"{CHECK_TARGET} never proves that the measured path references "
            f"{REQUIRED_BRIDGE_CALL}"
        )
    for entry_point in ("cublasGemmEx",):
        if entry_point not in joined:
            errors.append(f"{CHECK_TARGET} never rejects a measured fallback to {entry_point}")
    return errors


def validate_launcher_untouched(source) -> list:
    """The audited launcher must already own the data-stream mode, unchanged."""
    errors = []
    if LAUNCHER_DATA_MODE_VARIABLE not in source:
        errors.append(f"{LAUNCHER_RELATIVE_PATH} has no {LAUNCHER_DATA_MODE_VARIABLE} mode")
    if "--entrypoint" not in source:
        errors.append(
            "the launcher never bypasses the image entrypoint, whose banner would "
            "contaminate a data stream on stdout"
        )
    if "--gpus all" in source or "NVIDIA_VISIBLE_DEVICES=all" in source:
        errors.append("the launcher exposes every GPU")
    return errors


def validate_provenance_linkage(source, wrapper_contract, parsed_contract) -> list:
    """Prove provenance is read from the pinned contracts, not redefined."""
    errors = []
    for name in ('"VERSIONS.env"', '"PHASE3_VERSIONS.env"'):
        if name not in source:
            errors.append(f"the wrapper does not read {name}")

    for key, value in sorted(parsed_contract.items()):
        if len(value) < 3:
            continue
        if value in source:
            errors.append(
                f"the wrapper redefines the pinned {key} value as a literal instead of "
                "reading it from the version contract"
            )
        if key == "CUTLASS_VERSION" and value.lstrip("v") in source:
            errors.append("the wrapper redefines the pinned CuTe DSL version as a literal")

    if not isinstance(wrapper_contract, dict):
        return errors + ["the wrapper did not return a contract mapping"]

    # The wrapper deliberately reads only the subset of pinned keys P3.3 needs;
    # reading fewer keys is not a violation, resolving one of them differently
    # from the contract file is. Every key it *does* claim must therefore match.
    for key, value in sorted(parsed_contract.items()):
        if key in wrapper_contract and wrapper_contract[key] != value:
            errors.append(
                f"the wrapper resolved {key}={wrapper_contract[key]!r}, "
                f"the contract file says {value!r}"
            )
    for key in ("CUDA_VERSION", "CUTLASS_VERSION", "CUTLASS_COMMIT", "CUDA_ARCH",
                "PYTORCH_VERSION", "PYTORCH_CUDA_VERSION", "CUTEDSL_P31_EXAMPLE_PATH",
                "CUTEDSL_P31_EXAMPLE_GIT_BLOB", "CUTEDSL_P31_EXAMPLE_SHA256"):
        if key not in wrapper_contract:
            errors.append(f"the wrapper never resolves the pinned key {key}")
    if wrapper_contract.get("CUDA_ARCH") != EXPECTED_CUDA_ARCH:
        errors.append(
            f"the pinned architecture is {wrapper_contract.get('CUDA_ARCH')!r}, "
            f"P3.3 targets {EXPECTED_CUDA_ARCH!r}"
        )
    if wrapper_contract.get("EXPECTED_COMPUTE_CAPABILITY") != EXPECTED_COMPUTE_CAPABILITY:
        errors.append(
            f"{EXPECTED_CUDA_ARCH} must derive compute capability "
            f"{EXPECTED_COMPUTE_CAPABILITY}, the wrapper derived "
            f"{wrapper_contract.get('EXPECTED_COMPUTE_CAPABILITY')!r}"
        )
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
    """Require truthful, non-overstated P3.3 status claims."""
    errors = []
    if "P3.3 | cuBLASLt baseline | YES | YES | YES |" not in plan_text:
        errors.append("PLAN.md does not record P3.3 as YES / YES / YES")
    for stale in (
        "P3.3 | cuBLASLt baseline | YES | YES | NO |",
        "P3.3 | cuBLASLt baseline | YES | NO | YES |",
        "P3.3 | cuBLASLt baseline | YES | NO | NO |",
        "P3.3 | cuBLASLt baseline | NO | NO | NO |",
    ):
        if stale in plan_text:
            errors.append(f"PLAN.md records a stale P3.3 status: {stale!r}")
    for later in (
        "P3.4 | Three execution variants | NO | NO | NO |",
        "P3.5 | Five shapes and comparison | NO | NO | NO |",
    ):
        if later not in plan_text:
            errors.append(f"PLAN.md no longer records {later!r}")

    if "P3.3 = YES / YES / YES" not in protocol_text:
        errors.append(f"{PROTOCOL_RELATIVE_PATH} does not state P3.3 = YES / YES / YES")
    if "P3.3 creates no publishable performance result" not in protocol_text:
        errors.append(
            f"{PROTOCOL_RELATIVE_PATH} does not state that P3.3 creates no publishable result"
        )
    for required in ("independently audited", "verified on GB300"):
        if required.lower() not in protocol_text.lower():
            errors.append(
                f"{PROTOCOL_RELATIVE_PATH} does not record that P3.3 is {required}"
            )
    if "P3.3: CLOSED" not in readme_text:
        errors.append("README.md does not record P3.3 as CLOSED")
    return errors


# --- Checks against the real wrapper -----------------------------------------


def _load_wrapper_module(wrapper_path):
    """Import the wrapper as a library (its heavy imports stay deferred)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("p33_wrapper_under_test", str(wrapper_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["p33_wrapper_under_test"] = module
    spec.loader.exec_module(module)
    return module


def _run_guarded(wrapper_path, argv):
    """Run the wrapper behind the GPU-free import guard, returning the result."""
    script = GPU_FREE_GUARD.format(
        blocked=list(GPU_STACK_MODULES),
        argv0=str(wrapper_path),
        argv=list(argv),
        wrapper=str(wrapper_path),
    )
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False
    )


def _nm_readelf(command):
    """Run one ELF inspection tool, returning None when it is unavailable."""
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout


def check_wrapper(repo_root) -> list:
    """Run the whole contract check against the real repository."""
    errors = []
    root = Path(repo_root).resolve()

    wrapper_path = root / WRAPPER_RELATIVE_PATH
    bridge_path = root / BRIDGE_RELATIVE_PATH
    protocol_path = root / PROTOCOL_RELATIVE_PATH
    checker_path = root / CHECKER_RELATIVE_PATH
    makefile_path = root / "Makefile"
    plan_path = root / "PLAN.md"
    readme_path = root / "README.md"
    launcher_path = root / LAUNCHER_RELATIVE_PATH

    for path in (wrapper_path, bridge_path, protocol_path, checker_path, makefile_path,
                 plan_path, readme_path, launcher_path):
        if not path.is_file():
            errors.append(f"required file is missing: {path.relative_to(root)}")
    if errors:
        return errors

    wrapper_source = wrapper_path.read_text(encoding="utf-8")
    bridge_source = bridge_path.read_text(encoding="utf-8")
    makefile_text = makefile_path.read_text(encoding="utf-8")

    # 1. The wrapper's own declarations.
    module = _load_wrapper_module(wrapper_path)
    errors.extend(validate_csv_schema(module.CSV_FIELDS))
    errors.extend(validate_frozen_config(module.FROZEN_CONFIG))

    for name, expected in (
        ("FROZEN_WORKSPACE_LIMIT_BYTES", EXPECTED_WORKSPACE_LIMIT_BYTES),
        ("FROZEN_HEURISTIC_REQUESTED", EXPECTED_HEURISTIC_REQUESTED),
        ("FROZEN_SEARCH_MODE", EXPECTED_SEARCH_MODE),
        ("MIN_ITERATIONS", MIN_ITERATIONS),
        ("MAX_WARMUP_ITERATIONS", MAX_WARMUP_ITERATIONS),
        ("MAX_ITERATIONS", MAX_ITERATIONS),
    ):
        actual = getattr(module, name, None)
        if actual != expected:
            errors.append(f"the wrapper declares {name}={actual!r}, expected {expected!r}")

    # 2. Serialization of a synthetic row, through the wrapper's own code.
    try:
        row = module._synthetic_row()
        errors.extend(validate_row_mapping(row))
        errors.extend(validate_serialized_output(module.serialize_row(row)))
    except Exception as exc:  # noqa: BLE001 - a failure here is a contract failure
        errors.append(f"the wrapper cannot serialize a synthetic valid row: {exc}")
        row = None

    # 3. Adversarial rows must be rejected by the wrapper's own validator.
    if row is not None:
        adversarial = {
            "publishable=true": {"publishable": "true"},
            "correctness=FAIL": {"correctness": "FAIL"},
            "a changed shape": {"m": "8192"},
            "a changed leading dimension": {"lda": "2048"},
            "a changed transpose": {"transb": "CUBLAS_OP_N"},
            "a changed order": {"order_d": "CUBLASLT_ORDER_COL"},
            "a changed epilogue": {"epilogue": "CUBLASLT_EPILOGUE_RELU"},
            "a changed alpha": {"alpha": "2.000000000"},
            "a changed beta": {"beta": "1.000000000"},
            "a changed workspace limit": {"workspace_limit_bytes": "134217728"},
            "a changed heuristic request": {"heuristic_requested": "64"},
            "a workspace above the limit": {"workspace_bytes": "134217729"},
            "a non-finite timing": {"kernel_time_ms": "nan"},
            "a zero timing": {"kernel_time_ms": "0.000000"},
            "an exponent-notation timing": {"kernel_time_ms": "1e3"},
            "a non-canonical boolean": {"git_dirty": "True"},
            "a non-canonical integer": {"algo_id": "007"},
            "an unknown field": {"tflops": "1.0"},
            "a P3.2 field name": {"compile_time_ms": "1.000000"},
        }
        for description, override in sorted(adversarial.items()):
            candidate = {**row, **override}
            try:
                module.validate_row(candidate)
            except Exception:  # noqa: BLE001 - any rejection is the expected outcome
                pass
            else:
                errors.append(f"the wrapper accepted a row with {description}")
            if not validate_row_mapping(candidate):
                errors.append(f"this checker accepted a row with {description}")

        missing_field = {k: v for k, v in row.items() if k != "algo_id"}
        try:
            module.validate_row(missing_field)
        except Exception:  # noqa: BLE001
            pass
        else:
            errors.append("the wrapper accepted a row missing algo_id")

    # 4. A failed or skipped correctness check can never build a row.
    for bad in ("FAIL", "SKIPPED", ""):
        try:
            module.build_row(
                correctness=bad,
                max_abs_error=0.0,
                max_rel_error=0.0,
                setup_time_ms=1.0,
                first_launch_ms=1.0,
                kernel_time_ms=1.0,
                warmup_iterations=2,
                iterations=10,
                provenance=module._synthetic_provenance(),
                upstream=module._synthetic_upstream(),
                plan=module._synthetic_plan(),
            )
        except Exception:  # noqa: BLE001
            pass
        else:
            errors.append(f"the wrapper built a row with correctness={bad!r}")

    # 5. Source-level structure.
    errors.extend(validate_source(wrapper_source))
    errors.extend(validate_fp32_precision_policy(wrapper_source))
    errors.extend(validate_bridge_source(bridge_source))

    for identifier in FORBIDDEN_LATER_UNIT_IDENTIFIERS:
        try:
            names = [name.lower() for name in python_name_tokens(wrapper_source)]
        except Exception:  # noqa: BLE001
            names = []
        hits = sorted({name for name in names if identifier in name})
        if hits:
            errors.append(
                f"the wrapper implements P3.4/P3.5 functionality (identifier(s) "
                f"{', '.join(hits)})"
            )

    # 6. The command line.
    errors.extend(validate_cli_options(
        set(re.findall(r"--[a-z0-9][a-z0-9-]*", module.build_arg_parser().format_help()))
    ))

    # 7. Provenance linkage.
    try:
        parsed = {}
        parsed.update(parse_env_file(root / GLOBAL_CONTRACT_FILE))
        parsed.update(parse_env_file(root / PHASE3_CONTRACT_FILE))
        wrapper_contract = module.load_pinned_contract(root)
        errors.extend(validate_provenance_linkage(wrapper_source, wrapper_contract, parsed))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"the pinned contracts could not be resolved: {exc}")

    # 8. P3.3 must add no key to either version contract.
    for contract_file in (GLOBAL_CONTRACT_FILE, PHASE3_CONTRACT_FILE):
        text = (root / contract_file).read_text(encoding="utf-8")
        for forbidden in ("CUBLAS", "CUBLASLT_VERSION", "P33_"):
            if re.search(rf"^{forbidden}", text, flags=re.M):
                errors.append(f"{contract_file} gained a P3.3 key starting with {forbidden!r}")

    # 9. --help and --self-test are GPU-free and never load the bridge.
    for argv in (["--help"], ["--self-test"]):
        completed = _run_guarded(wrapper_path, argv)
        if completed.returncode != 0:
            errors.append(
                f"the wrapper failed the GPU-free guard with {argv}: exit "
                f"{completed.returncode}; {completed.stderr.strip()[-300:]}"
            )
        if argv == ["--self-test"] and completed.stdout:
            errors.append("--self-test wrote to stdout; every diagnostic belongs on stderr")

    # 10. A correctness failure emits no CSV at all.
    probe = CORRECTNESS_FAILURE_PROBE.format(
        blocked=list(GPU_STACK_MODULES), wrapper=str(wrapper_path)
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=300, check=False
    )
    if completed.returncode == 0:
        errors.append("a correctness failure did not produce a non-zero exit status")
    if completed.stdout:
        errors.append(
            f"a correctness failure still wrote {len(completed.stdout)} byte(s) to stdout"
        )

    # 11. Make integration.
    make_variables = parse_make_variables(makefile_text)
    errors.extend(validate_smoke_recipe(
        extract_make_recipe(makefile_text, SMOKE_TARGET), make_variables
    ))
    errors.extend(validate_check_recipe(
        makefile_text, extract_make_recipe(makefile_text, CHECK_TARGET)
    ))
    for target in (SMOKE_TARGET, CHECK_TARGET):
        if not re.search(rf"^\t{re.escape(target)} ", makefile_text, flags=re.M) and \
                target not in makefile_text:
            errors.append(f"{target} is not defined in the Makefile")
    if not re.search(rf"^{re.escape(SMOKE_TARGET)}:$", makefile_text, flags=re.M):
        errors.append(f"{SMOKE_TARGET} must have no Make prerequisite")
    errors.extend(validate_launcher_untouched(launcher_path.read_text(encoding="utf-8")))

    # 12. Truthful status documents.
    errors.extend(validate_status_documents(
        plan_path.read_text(encoding="utf-8"),
        protocol_path.read_text(encoding="utf-8"),
        readme_path.read_text(encoding="utf-8"),
    ))

    # 13. The P3.2 unit must be untouched by P3.3.
    p32_wrapper = root / P32_WRAPPER_RELATIVE_PATH
    if p32_wrapper.is_file():
        p32_source = p32_wrapper.read_text(encoding="utf-8")
        if 'SCHEMA_VERSION = "p32.v1"' not in p32_source:
            errors.append("P3.3 changed the frozen P3.2 schema version")
        if "cublaslt" in p32_source.lower().replace("cublaslt baseline", ""):
            pass  # prose mentioning the baseline is expected; code is banned by P3.2's checker

    # 14. The compiled bridge, when the gate has already built it.
    if BRIDGE_LIBRARY_PATH.is_file():
        errors.extend(validate_shared_object(BRIDGE_LIBRARY_PATH, _nm_readelf))
    else:
        print(
            f"check_cublaslt_gemm_p33: note: {BRIDGE_LIBRARY_PATH} is not built; the ELF "
            "checks are skipped here and are executed by make gemm-cublaslt-p33-check",
            file=sys.stderr,
        )
    return errors


# --- Self-test ---------------------------------------------------------------


def _good_row() -> dict:
    row = dict(EXPECTED_FIXED_ROW_VALUES)
    row.update(
        {
            "max_abs_error": "0.000000000",
            "max_rel_error": "0.000000000",
            "setup_time_ms": "12.500000",
            "first_launch_ms": "3.250000",
            "kernel_time_ms": "7.500000",
            "warmup_iterations": "2",
            "iterations": "10",
            "workspace_bytes": "4194304",
            "alignment_a_bytes": "256",
            "alignment_b_bytes": "256",
            "alignment_c_bytes": "256",
            "alignment_d_bytes": "256",
            "heuristic_returned": "8",
            "heuristic_index": "0",
            "algo_id": "21",
            "tile_id": "27",
            "stages_id": "15",
            "split_k": "0",
            "reduction_scheme": "0",
            "cta_swizzling": "0",
            "custom_option": "0",
            "inner_shape_id": "0",
            "cluster_shape_id": "0",
            "waves_count": "1.500000",
            "gpu_name": "SYNTHETIC TEST DEVICE",
            "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "compute_capability": "9.9",
            "driver_version": "999.99.99",
            "cuda_toolkit_version": "99.9.9",
            "torch_cuda_version": "98.7",
            "cutedsl_version": "97.6.5",
            "cublaslt_version": "999999",
            "cutlass_commit": "1" * 40,
            "upstream_example_sha256": "2" * 64,
            "git_commit": "0" * 40,
            "git_dirty": "false",
        }
    )
    return row


def _serialize(row) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(EXPECTED_CSV_FIELDS),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerow(row)
    return buffer.getvalue()


_GOOD_BRIDGE = '''
#include <cublasLt.h>
#include <cuda_runtime.h>
static const long P33_M = 4096;
static const long P33_N = 4096;
static const long P33_K = 4096;
static const long P33_BATCH_COUNT = 1;
static const long P33_LDA = P33_K;
static const long P33_LDB = P33_K;
static const long P33_LDC = P33_N;
static const long P33_LDD = P33_N;
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
static const unsigned long P33_WORKSPACE_LIMIT_BYTES = 67108864ULL;
static const int P33_HEURISTIC_REQUESTED = 32;
static const cublasLtMatmulSearch_t P33_SEARCH_MODE = CUBLASLT_SEARCH_BEST_FIT;
extern "C" {
int setup(void) {
  try {
    cublasLtCreate(0);
    cublasLtMatmulDescCreate(0, P33_COMPUTE_TYPE, P33_SCALE_TYPE);
    cublasLtMatrixLayoutCreate(0, P33_AB_TYPE, 0, 0, 0);
    cublasLtMatmulPreferenceCreate(0);
    cublasLtMatmulPreferenceSetAttribute(0, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, 0, 0);
    cublasLtMatmulPreferenceSetAttribute(0, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, 0, 0);
    cublasLtMatmulPreferenceSetAttribute(0, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, 0, 0);
    cublasLtMatmulPreferenceSetAttribute(0, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, 0, 0);
    cublasLtMatmulPreferenceSetAttribute(0, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, 0, 0);
    cublasLtMatmulAlgoGetHeuristic(0, 0, 0, 0, 0, 0, 0, P33_HEURISTIC_REQUESTED, 0, 0);
    cublasLtMatmulAlgoCheck(0, 0, 0, 0, 0, 0, 0, 0);
    cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_ID, 0, 0, 0);
    cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_TILE_ID, 0, 0, 0);
    cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_STAGES_ID, 0, 0, 0);
    cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, 0, 0, 0);
    cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, 0, 0, 0);
    cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, 0, 0, 0);
    cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, 0, 0, 0);
    cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, 0, 0, 0);
    cublasLtMatmulAlgoConfigGetAttribute(0, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, 0, 0, 0);
    p33_read_algo_config<int32_t>(0, CUBLASLT_ALGO_CONFIG_ID, "ID", 0);
    p33_read_algo_config<uint32_t>(0, CUBLASLT_ALGO_CONFIG_TILE_ID, "TILE_ID", 0);
    p33_read_algo_config<uint32_t>(0, CUBLASLT_ALGO_CONFIG_STAGES_ID, "STAGES_ID", 0);
    p33_read_algo_config<uint32_t>(0, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, "SPLITK_NUM", 0);
    p33_read_algo_config<uint32_t>(0, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, "REDUCTION_SCHEME", 0);
    p33_read_algo_config<uint32_t>(0, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, "CTA_SWIZZLING", 0);
    p33_read_algo_config<uint32_t>(0, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION, "CUSTOM_OPTION", 0);
    p33_read_algo_config<uint16_t>(0, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, "INNER_SHAPE_ID", 0);
    p33_read_algo_config<uint16_t>(0, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, "CLUSTER_SHAPE_ID", 0);
    cublasLtGetVersion();
    cublasLtDestroy(0);
  } catch (...) {
    return 1;
  }
  return 0;
}
int run(void) { return cublasLtMatmul(0, 0, &P33_ALPHA, 0, 0, 0, 0, &P33_BETA, 0, 0, 0, 0, 0, 0, 0, 0); }
}
'''

_GOOD_NM_DEFINED = """
0000000000002c10 T p33_bridge_abi_version
0000000000002c50 T p33_cublaslt_version
0000000000002c30 T p33_last_error
0000000000002c60 T p33_plan_create
0000000000003ef0 T p33_plan_destroy
0000000000003cb0 T p33_plan_execute
0000000000002c20 T p33_plan_info_size
0000000000003e10 T p33_stream_synchronize
"""
_GOOD_NM_UNDEFINED = """
                 U cublasLtCreate@libcublasLt.so.13
                 U cublasLtMatmul@libcublasLt.so.13
                 U cublasLtMatmulAlgoCheck@libcublasLt.so.13
"""
_GOOD_READELF = """
 0x0000000000000001 (NEEDED)             Shared library: [libcublasLt.so.13]
 0x0000000000000001 (NEEDED)             Shared library: [libcudart.so.13]
"""

_GOOD_SMOKE_RECIPE = [
    "\t@if [ -z \"$${BLACKWELL_GPU_INDEX:-}\" ]; then \\",
    "\t\techo \"ERROR: BLACKWELL_GPU_INDEX must be set explicitly.\" >&2; \\",
    "\t\texit 2; \\",
    "\tfi",
    "\t@status=0; \\",
    "\tRUN_CONTAINER_STDOUT_IS_DATA=1 scripts/run_container.sh bash -c 'set -euo pipefail; \\",
    "\t\tmkdir -p /tmp/p33-bridge; \\",
    "\t\tnvcc -std=c++17 -O3 -lineinfo -o /tmp/p33-bridge/lib.so src/gemm/cublaslt_bridge.cu; \\",
    "\t\thead_commit=\"$$(git rev-parse HEAD)\"; \\",
    "\t\tsha=\"$$(sha256sum file | cut -d\" \" -f1)\"; \\",
    "\t\texec python3 src/gemm/cublaslt_gemm.py \\",
    "\t\t\t--warmup-iterations 2 \\",
    "\t\t\t--iterations 10' || status=$$?; \\",
    "\techo \"P3.3 FUNCTIONAL VERIFICATION ONLY -- NOT A PERFORMANCE COMPARISON.\" >&2; \\",
    "\tif [ \"$$status\" -eq 0 ]; then \\",
    "\t\techo \"" + SMOKE_SUCCESS_SENTENCE + "\" >&2; \\",
    "\telse \\",
    "\t\techo \"P3.3 smoke FAILED\" >&2; \\",
    "\tfi; \\",
    "\texit $$status",
]


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

    print("check_cublaslt_gemm_p33 --self-test (GPU-free)", file=sys.stderr)

    # Schema.
    check("the frozen schema has 77 fields", len(EXPECTED_CSV_FIELDS) == 77,
          str(len(EXPECTED_CSV_FIELDS)))
    check("the frozen schema has no duplicate", len(set(EXPECTED_CSV_FIELDS)) == 77)
    check("the correct schema is accepted", validate_csv_schema(EXPECTED_CSV_FIELDS) == [])
    rejects("a reordered schema is rejected",
            validate_csv_schema(tuple(reversed(EXPECTED_CSV_FIELDS))), "wrong order")
    rejects("a schema with an extra field is rejected",
            validate_csv_schema(EXPECTED_CSV_FIELDS + ("tflops",)), "unknown field")
    rejects("a schema missing a field is rejected",
            validate_csv_schema(EXPECTED_CSV_FIELDS[:-1]), "missing field")
    rejects("reusing the P3.2 timing field is rejected",
            validate_csv_schema(EXPECTED_CSV_FIELDS + ("compile_time_ms",)), "P3.2 field")

    # Frozen configuration.
    check("the correct frozen config is accepted",
          validate_frozen_config(dict(EXPECTED_FROZEN_CONFIG)) == [])
    for field, bad in (
        ("m", 8192), ("k", 2048), ("lda", 2048), ("ldd", 8192),
        ("transa", "CUBLAS_OP_T"), ("transb", "CUBLAS_OP_N"),
        ("order_c", "CUBLASLT_ORDER_COL"),
        ("compute_type", "CUBLAS_COMPUTE_32F_FAST_TF32"),
        ("scale_type", "CUDA_R_16BF"),
        ("epilogue", "CUBLASLT_EPILOGUE_RELU"),
        ("pointer_mode", "CUBLASLT_POINTER_MODE_DEVICE"),
        ("alpha", 2.0), ("beta", 1.0), ("seed", 2222),
        ("workspace_limit_bytes", 134217728), ("heuristic_requested", 64),
        ("cache_mode", "cold"), ("publishable", True),
        ("schema_version", "p32.v1"), ("method", "cutedsl"),
    ):
        rejects(f"a frozen config with {field}={bad!r} is rejected",
                validate_frozen_config({**EXPECTED_FROZEN_CONFIG, field: bad}))

    # Rows.
    row = _good_row()
    check("a well-formed row is accepted", validate_row_mapping(row) == [],
          str(validate_row_mapping(row)))
    check("the documented non-split split_k=0 is accepted", row["split_k"] == "0")
    check("a well-formed row serializes to two lines",
          validate_serialized_output(_serialize(row)) == [])
    for description, override in sorted({
        "publishable=true": {"publishable": "true"},
        "correctness=FAIL": {"correctness": "FAIL"},
        "correctness=SKIPPED": {"correctness": "SKIPPED"},
        "a changed shape": {"n": "8192"},
        "a changed leading dimension": {"ldc": "2048"},
        "a changed transpose": {"transa": "CUBLAS_OP_T"},
        "a changed order": {"order_b": "CUBLASLT_ORDER_COL"},
        "a changed dtype": {"ab_dtype": "Float16"},
        "a changed epilogue": {"epilogue": "CUBLASLT_EPILOGUE_BIAS"},
        "a changed alpha": {"alpha": "2.000000000"},
        "a changed beta": {"beta": "1.000000000"},
        "a changed workspace limit": {"workspace_limit_bytes": "134217728"},
        "a changed heuristic request": {"heuristic_requested": "64"},
        "a workspace above the limit": {"workspace_bytes": "134217729"},
        "more results than requested": {"heuristic_returned": "33"},
        "an index past the returned count": {"heuristic_index": "8"},
        "a non-power-of-two alignment": {"alignment_c_bytes": "96"},
        "a NaN timing": {"kernel_time_ms": "nan"},
        "an infinite timing": {"setup_time_ms": "inf"},
        "a zero timing": {"first_launch_ms": "0.000000"},
        "an exponent-notation timing": {"kernel_time_ms": "1e3"},
        "a wrong-precision timing": {"kernel_time_ms": "7.5"},
        "a non-canonical boolean": {"git_dirty": "TRUE"},
        "a non-canonical integer": {"tile_id": "027"},
        "a malformed waves count": {"waves_count": "nan"},
        "a malformed cuBLASLt version": {"cublaslt_version": "13.2.0"},
        "a malformed GPU UUID": {"gpu_uuid": "0000"},
        "a malformed digest": {"upstream_example_sha256": "abc"},
        "an out-of-range iteration count": {"iterations": "101"},
        "a negative split_k": {"split_k": "-1"},
    }.items()):
        rejects(f"a row with {description} is rejected",
                validate_row_mapping({**row, **override}))
    rejects("a row missing a field is rejected",
            validate_row_mapping({k: v for k, v in row.items() if k != "waves_count"}),
            "missing field")
    rejects("a row with an unknown field is rejected",
            validate_row_mapping({**row, "tflops": "1.0"}), "unknown field")
    rejects("a two-row output is rejected",
            validate_serialized_output(_serialize(row) + _serialize(row).splitlines()[1] + "\n"))
    rejects("an empty output is rejected", validate_serialized_output(""))

    # CLI.
    check("the permitted option set is accepted",
          validate_cli_options(ALLOWED_CLI_OPTIONS) == [])
    for bad in ("--workspace-bytes", "--heuristic-count", "--algo-id", "--alpha", "--epilogue",
                "--lda", "--transa", "--shape", "--publish", "--skip-ref-check", "--cache-mode"):
        rejects(f"the forbidden option {bad} is rejected",
                validate_cli_options(set(ALLOWED_CLI_OPTIONS) | {bad}))
    rejects("a missing permitted option is rejected",
            validate_cli_options(ALLOWED_CLI_OPTIONS - {"--iterations"}), "missing option")

    # Wrapper source structure.
    check("a clean wrapper body is accepted",
          validate_source("import torch\n\n\ndef f(x):\n    return x\n") == [])
    for description, source in (
        ("a TFLOP/s variable", "tflops = 1.0\n"),
        ("a speedup variable", "speedup_vs_cutedsl = 2.0\n"),
        ("a campaign directory", "campaign_dir = 'x'\n"),
        ("a cublasGemmEx fallback", "cublasGemmEx(1)\n"),
        ("a torch.matmul baseline", "d = torch.matmul(a, b)\n"),
        ("a torch.mm baseline", "d = torch.mm(a, b)\n"),
        ("an @ operator baseline", "d = a @ b\n"),
        ("a result-file write", "open('results/raw/x.csv', 'w')\n"),
        ("a campaign mkdir", "import os\nos.makedirs('x')\n"),
    ):
        rejects(f"{description} is rejected", validate_source(source))
    check("torch.einsum as the untimed oracle is allowed",
          validate_source("r = torch.einsum('mkl,nkl->mnl', a, b)\n") == [])
    check(
        "a self-test helper writing into a TemporaryDirectory is allowed",
        validate_no_result_files(
            "import tempfile\n"
            "def _fixture(text):\n"
            "    with tempfile.TemporaryDirectory() as d:\n"
            "        p = Path(d) / 'x.env'\n"
            "        p.write_text(text)\n"
            "        return p\n"
        ) == [],
    )
    rejects(
        "a write outside a TemporaryDirectory is rejected",
        validate_no_result_files("def save(t):\n    Path('/x/row.csv').write_text(t)\n"),
        "writes a file",
    )

    # FP32 policy. The scan is structural, so prose explaining why the legacy
    # aliases are never used stays legal while any real access fails.
    check("the required FP32 policy is accepted",
          validate_fp32_precision_policy(
              'matmul.fp32_precision = "ieee"\n# the legacy allow_tf32 flag is never used\n'
          ) == [])
    check(
        "a docstring explaining allow_tf32 and set_float32_matmul_precision is allowed",
        validate_fp32_precision_policy(
            'def guard(m):\n'
            '    """Sets fp32_precision; the allow_tf32 alias and\n'
            '    torch.set_float32_matmul_precision() are aliases of one setting."""\n'
            '    m.fp32_precision = "ieee"\n'
        ) == [],
    )
    rejects("a legacy allow_tf32 write is rejected",
            validate_fp32_precision_policy(
                'matmul.fp32_precision = "ieee"\nmatmul.allow_tf32 = False\n'
            ), "allow_tf32")
    rejects("a legacy allow_tf32 read is rejected",
            validate_fp32_precision_policy(
                'matmul.fp32_precision = "ieee"\nx = matmul.allow_tf32\n'
            ), "allow_tf32")
    rejects("set_float32_matmul_precision is rejected",
            validate_fp32_precision_policy(
                'torch.set_float32_matmul_precision("highest")\nm.fp32_precision = "ieee"\n'
            ), "set_float32_matmul_precision")
    rejects("a getattr-style string bypass is rejected",
            validate_fp32_precision_policy(
                'setattr(m, "allow_tf32", True)\nm.fp32_precision = "ieee"\n'
            ), "allow_tf32")
    rejects("a wrapper that never uses fp32_precision is rejected",
            validate_fp32_precision_policy('x = "ieee"\n'), "never uses the required")

    # Makefile variable expansion.
    variables = parse_make_variables(
        "A := /tmp/p33-bridge\nB := $(A)/lib.so\nC = ignored\n"
    )
    check("simply expanded Make variables are parsed",
          variables.get("A") == "/tmp/p33-bridge" and variables.get("B")
          == "/tmp/p33-bridge/lib.so", str(variables))
    check("recursive Make assignments are ignored", "C" not in variables)
    check("an unexpanded recipe reference is resolved",
          expand_make_variables("mkdir -p $(B)", variables) == "mkdir -p /tmp/p33-bridge/lib.so")
    check("an unknown reference is left intact",
          expand_make_variables("$(UNKNOWN)", variables) == "$(UNKNOWN)")
    check(
        "a smoke recipe that builds into /tmp only via a Make variable is accepted",
        validate_smoke_recipe(
            [line.replace("/tmp/p33-bridge", "$(P33_DIR)") for line in _GOOD_SMOKE_RECIPE],
            {"P33_DIR": "/tmp/p33-bridge"},
        ) == [],
    )
    rejects(
        "a smoke recipe whose Make variable is not under /tmp is rejected",
        validate_smoke_recipe(
            [line.replace("/tmp/p33-bridge", "$(P33_DIR)") for line in _GOOD_SMOKE_RECIPE],
            {"P33_DIR": "build/p33-bridge"},
        ),
        "container-private /tmp",
    )

    # Bridge source structure.
    check("a well-formed bridge is accepted", validate_bridge_source(_GOOD_BRIDGE) == [],
          str(validate_bridge_source(_GOOD_BRIDGE)))
    for description, mutation, fragment in (
        ("a changed M", ("static const long P33_M = 4096;", "static const long P33_M = 8192;"),
         "P33_M"),
        ("a changed lda", ("static const long P33_LDA = P33_K;",
                           "static const long P33_LDA = 2048;"), "P33_LDA"),
        ("a changed transb", ("P33_TRANSB = CUBLAS_OP_T", "P33_TRANSB = CUBLAS_OP_N"),
         "P33_TRANSB"),
        ("a changed order", ("P33_ORDER = CUBLASLT_ORDER_ROW", "P33_ORDER = CUBLASLT_ORDER_COL"),
         "P33_ORDER"),
        ("a changed dtype", ("P33_AB_TYPE = CUDA_R_16BF", "P33_AB_TYPE = CUDA_R_16F"),
         "P33_AB_TYPE"),
        ("a changed compute type", ("P33_COMPUTE_TYPE = CUBLAS_COMPUTE_32F",
                                    "P33_COMPUTE_TYPE = CUBLAS_COMPUTE_32F_FAST_TF32"),
         "P33_COMPUTE_TYPE"),
        ("a changed epilogue", ("P33_EPILOGUE = CUBLASLT_EPILOGUE_DEFAULT",
                                "P33_EPILOGUE = CUBLASLT_EPILOGUE_RELU"), "P33_EPILOGUE"),
        ("a changed alpha", ("P33_ALPHA = 1.0f", "P33_ALPHA = 2.0f"), "P33_ALPHA"),
        ("a changed beta", ("P33_BETA = 0.0f", "P33_BETA = 1.0f"), "P33_BETA"),
        ("a changed workspace limit", ("P33_WORKSPACE_LIMIT_BYTES = 67108864ULL",
                                       "P33_WORKSPACE_LIMIT_BYTES = 134217728ULL"),
         "P33_WORKSPACE_LIMIT_BYTES"),
        ("a changed heuristic count", ("P33_HEURISTIC_REQUESTED = 32",
                                       "P33_HEURISTIC_REQUESTED = 64"),
         "P33_HEURISTIC_REQUESTED"),
        ("a changed search mode", ("P33_SEARCH_MODE = CUBLASLT_SEARCH_BEST_FIT",
                                   "P33_SEARCH_MODE = CUBLASLT_SEARCH_LIMITED_BY_ALGO_ID"),
         "P33_SEARCH_MODE"),
    ):
        rejects(f"a bridge with {description} is rejected",
                validate_bridge_source(_GOOD_BRIDGE.replace(*mutation)), fragment)
    rejects("a bridge without cublasLtMatmul is rejected",
            validate_bridge_source(_GOOD_BRIDGE.replace("cublasLtMatmul(0, 0, &P33_ALPHA",
                                                        "someOtherGemm(0, 0, &P33_ALPHA")),
            "cublasLtMatmul")
    rejects("a bridge with a cublasGemmEx fallback is rejected",
            validate_bridge_source(_GOOD_BRIDGE + "\nint fb(void){return cublasGemmEx(0);}\n"),
            "cublasGemmEx")
    rejects("a bridge that benchmarks candidates is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE.replace("int run(void)",
                                     "int bench(void){cudaEventRecord(0,0);return 0;}\nint run(void)")
            ), "timing facility")
    rejects("a bridge with two matmul call sites is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE.replace("int run(void)",
                                     "int run2(void){return cublasLtMatmul(0);}\nint run(void)")
            ), "call site")
    rejects("a bridge writing to stdout is rejected",
            validate_bridge_source(_GOOD_BRIDGE.replace("return 0;\n}", 'printf("x");return 0;\n}')),
            "standard stream")
    rejects("a bridge defining its own kernel is rejected",
            validate_bridge_source(_GOOD_BRIDGE + "\n__global__ void my_gemm(){}\n"),
            "custom CUDA kernel")
    rejects("a bridge without a catch-all is rejected",
            validate_bridge_source(_GOOD_BRIDGE.replace("catch (...)", "catch (int)")),
            "catch-all")
    rejects("a bridge missing an algorithm attribute is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE.replace(
                    "CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID",
                    "CUBLASLT_ALGO_CONFIG_UNKNOWN",
                )
            ), "CLUSTER_SHAPE_ID")
    rejects(
        "a bridge reading split_k with the wrong signed width is rejected",
        validate_bridge_source(
            _GOOD_BRIDGE.replace(
                "p33_read_algo_config<uint32_t>(0, CUBLASLT_ALGO_CONFIG_SPLITK_NUM",
                "p33_read_algo_config<int32_t>(0, CUBLASLT_ALGO_CONFIG_SPLITK_NUM",
            )
        ),
        "SPLITK_NUM",
    )
    rejects("a bridge missing an alignment preference is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE.replace("CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES", "0")
            ), "MIN_ALIGNMENT_D_BYTES")
    rejects("a bridge whose constant is only in a comment is rejected",
            validate_bridge_source(
                _GOOD_BRIDGE.replace("static const long P33_M = 4096;",
                                     "// static const long P33_M = 4096;")
            ), "P33_M")

    # Shared object inspection, driven with synthetic tool output.
    def fake_tools(defined, undefined, needed):
        def run(command):
            if "--defined-only" in command:
                return defined
            if "-u" in command:
                return undefined
            return needed
        return run

    check("a well-formed shared object is accepted",
          validate_shared_object(Path("/synthetic.so"),
                                 fake_tools(_GOOD_NM_DEFINED, _GOOD_NM_UNDEFINED,
                                            _GOOD_READELF)) == [])
    rejects("a shared object without cublasLtMatmul is rejected",
            validate_shared_object(Path("/synthetic.so"),
                                   fake_tools(_GOOD_NM_DEFINED,
                                              _GOOD_NM_UNDEFINED.replace("cublasLtMatmul@",
                                                                         "cublasSomething@"),
                                              _GOOD_READELF)),
            "cublasLtMatmul")
    rejects("a shared object with a cublasGemmEx fallback is rejected",
            validate_shared_object(Path("/synthetic.so"),
                                   fake_tools(_GOOD_NM_DEFINED,
                                              _GOOD_NM_UNDEFINED + "   U cublasGemmEx\n",
                                              _GOOD_READELF)),
            "cublasGemmEx")
    rejects("a shared object missing an export is rejected",
            validate_shared_object(Path("/synthetic.so"),
                                   fake_tools(
                                       _GOOD_NM_DEFINED.replace("p33_plan_execute", "p33_other"),
                                       _GOOD_NM_UNDEFINED, _GOOD_READELF)),
            "p33_plan_execute")
    rejects("a shared object not linked to cuBLASLt is rejected",
            validate_shared_object(Path("/synthetic.so"),
                                   fake_tools(_GOOD_NM_DEFINED, _GOOD_NM_UNDEFINED,
                                              _GOOD_READELF.replace("libcublasLt.so", "libx.so"))),
            "libcublasLt.so")
    rejects("an uninspectable shared object is rejected",
            validate_shared_object(Path("/synthetic.so"), lambda command: None))

    # Make recipes.
    check("a well-formed smoke recipe is accepted",
          validate_smoke_recipe(_GOOD_SMOKE_RECIPE) == [],
          str(validate_smoke_recipe(_GOOD_SMOKE_RECIPE)))
    rejects("a smoke recipe that echoes the recipe is rejected",
            validate_smoke_recipe([line.replace("\t@", "\t", 1) for line in _GOOD_SMOKE_RECIPE]),
            "not quiet")
    rejects("a smoke recipe echoing to stdout is rejected",
            validate_smoke_recipe([line.replace(" >&2", "") for line in _GOOD_SMOKE_RECIPE]),
            "not redirected to stderr")
    rejects("a smoke recipe without the data-stream mode is rejected",
            validate_smoke_recipe([line.replace("RUN_CONTAINER_STDOUT_IS_DATA=1 ", "")
                                   for line in _GOOD_SMOKE_RECIPE]),
            LAUNCHER_DATA_MODE_VARIABLE)
    rejects("a smoke recipe calling Docker directly is rejected",
            validate_smoke_recipe([line.replace("scripts/run_container.sh", "docker run --rm")
                                   for line in _GOOD_SMOKE_RECIPE]),
            "Docker")
    rejects("a smoke recipe not validating the GPU index first is rejected",
            validate_smoke_recipe(_GOOD_SMOKE_RECIPE[4:]), GPU_INDEX_VARIABLE)
    rejects("a smoke recipe filtering stdout is rejected",
            validate_smoke_recipe(_GOOD_SMOKE_RECIPE[:-1]
                                  + ["\t\t| grep -v x; \\", "\texit $$status"]),
            "filters its output")
    rejects("a smoke recipe with an unconditional success message is rejected",
            validate_smoke_recipe(
                [line for line in _GOOD_SMOKE_RECIPE
                 if 'if [ "$$status" -eq 0 ]' not in line]
            ), "guarded")
    rejects("a smoke recipe that drops the exit status is rejected",
            validate_smoke_recipe([line for line in _GOOD_SMOKE_RECIPE
                                   if not line.strip().startswith("exit $$status")]),
            "exit status")
    rejects("a smoke recipe with the wrong iteration counts is rejected",
            validate_smoke_recipe([line.replace("--iterations 10", "--iterations 1000")
                                   for line in _GOOD_SMOKE_RECIPE]),
            "measured launches")
    rejects("a smoke recipe that never recompiles the bridge is rejected",
            validate_smoke_recipe([line for line in _GOOD_SMOKE_RECIPE if "nvcc" not in line]),
            "compiles the cuBLASLt bridge")

    # Status documents.
    good_plan = (
        "| P3.3 | cuBLASLt baseline | YES | YES | YES |\n"
        "| P3.4 | Three execution variants | NO | NO | NO |\n"
        "| P3.5 | Five shapes and comparison | NO | NO | NO |\n"
    )
    good_protocol = (
        "Status: `P3.3 = YES / YES / YES`.\n"
        "P3.3 creates no publishable performance result.\n"
        "P3.3 is independently audited and verified on GB300.\n"
    )
    good_readme = "P3.3 (cuBLASLt baseline); P3.3: CLOSED.\n"
    check("truthful status documents are accepted",
          validate_status_documents(good_plan, good_protocol, good_readme) == [],
          str(validate_status_documents(good_plan, good_protocol, good_readme)))
    rejects("a stale PLAN.md status is rejected",
            validate_status_documents(
                good_plan.replace("YES | YES | YES", "YES | NO | NO"),
                good_protocol, good_readme), "stale")
    rejects("a PLAN.md that never records P3.3 is rejected",
            validate_status_documents("", good_protocol, good_readme), "YES / YES / YES")
    rejects("a protocol without the non-publishable statement is rejected",
            validate_status_documents(
                good_plan,
                good_protocol.replace("P3.3 creates no publishable performance result.", ""),
                good_readme), "no publishable result")

    # Launcher.
    check("the audited launcher shape is accepted",
          validate_launcher_untouched(
              'RUN_CONTAINER_STDOUT_IS_DATA\n--entrypoint /bin/bash\n') == [])
    rejects("a launcher exposing every GPU is rejected",
            validate_launcher_untouched(
                'RUN_CONTAINER_STDOUT_IS_DATA\n--entrypoint x\n--gpus all\n'), "every GPU")

    # Recipe extraction.
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
    if arguments and arguments[0].startswith("-"):
        print(__doc__.strip().splitlines()[-4], file=sys.stderr)
        print("usage: check_cublaslt_gemm_p33.py [repository-root] | --self-test",
              file=sys.stderr)
        return 2
    if len(arguments) > 1:
        print("usage: check_cublaslt_gemm_p33.py [repository-root] | --self-test",
              file=sys.stderr)
        return 2

    root = Path(arguments[0]) if arguments else Path(__file__).resolve().parents[1]
    if not root.is_dir():
        print(f"check_cublaslt_gemm_p33: {root} is not a directory", file=sys.stderr)
        return 2

    print(f"check_cublaslt_gemm_p33: checking {root}", file=sys.stderr)
    try:
        errors = check_wrapper(root)
    except Exception as exc:  # noqa: BLE001 - a checker crash is a failed check
        print(f"check_cublaslt_gemm_p33: the check itself failed: {exc}", file=sys.stderr)
        return 1

    if errors:
        print(f"check_cublaslt_gemm_p33: FAIL ({len(errors)} finding(s))", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("check_cublaslt_gemm_p33: OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
