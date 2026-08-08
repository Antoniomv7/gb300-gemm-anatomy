#!/usr/bin/env python3
"""GPU-free contract checker for the P3.4 three-variant CuTe DSL wrapper.

This checker is deliberately independent of ``src/gemm/cutedsl_variants.py``:
it carries its own copy of the frozen P3.4 variant table, its own copy of the
frozen 51-field CSV schema, and its own row validator. A drift in either the
wrapper or the checker therefore shows up as a disagreement rather than as two
copies of the same mistake.

It uses only the Python standard library and never initializes CUDA. The only
subprocesses it starts are Python interpreters running the wrapper behind an
import guard that makes any attempt to import PyTorch, CuTe DSL, the CUDA
bindings, or either upstream example a hard failure - which is how "``--help``
and ``--self-test`` are GPU-free" is proved rather than assumed.

What it validates:

* the exact P3.4 schema and field order, and that no P3.2/P3.3 schema is
  reused or reinterpreted;
* the exact three-variant set, their fixed order, and their exact tiler,
  cluster, scheduler, 2-CTA, and upstream-source mapping;
* the fixed (4096,4096,4096,1) shape and the fixed BF16/FP32/layout/TMA/seed/
  reference contract;
* that the command line exposes no frozen scientific parameter and no way to
  skip the correctness check or to run fewer than three variants;
* that every row is ``publishable=false`` and no comparison or
  performance-derived metric exists anywhere;
* both upstream pin sets, including the three new ``CUTEDSL_P34_*`` keys, and
  that the wrapper duplicates none of those pinned values as literals;
* that the persistent source backs both persistent variants and the
  non-persistent source backs the non-persistent one;
* that the pinned persistent example really contains the official static
  persistent scheduler path and its ``CtaGroup.TWO`` selection;
* that stdout is all-or-nothing: a synthetic failure in any of the three
  variant positions produces no CSV at all;
* that the closed P3.1-P3.3 files, targets, schemas, and recorded status are
  not weakened, and that ``VERSIONS.env`` still parses under the closed P1/P2
  aggregators' own allowlists.

Usage:
  check_cutedsl_variants_p34.py [repository-root]
  check_cutedsl_variants_p34.py --self-test

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

WRAPPER_RELATIVE_PATH = "src/gemm/cutedsl_variants.py"
CHECKER_RELATIVE_PATH = "scripts/check_cutedsl_variants_p34.py"
PROTOCOL_RELATIVE_PATH = "src/gemm/P3_4_PROTOCOL.md"
GLOBAL_CONTRACT_FILE = "VERSIONS.env"
PHASE3_CONTRACT_FILE = "PHASE3_VERSIONS.env"

# The closed units this one must not weaken.
P32_WRAPPER_RELATIVE_PATH = "src/gemm/cutedsl_gemm.py"
P33_WRAPPER_RELATIVE_PATH = "src/gemm/cublaslt_gemm.py"
P33_BRIDGE_RELATIVE_PATH = "src/gemm/cublaslt_bridge.cu"
P1_AGGREGATOR_RELATIVE_PATH = "scripts/aggregate_exp01_memory_paths.py"
P2_AGGREGATOR_RELATIVE_PATH = "scripts/aggregate_exp02_umma_throughput.py"

UPSTREAM_CHECKOUT_DIR = Path("/opt/cutlass")

EXPECTED_CSV_FIELDS = (
    "schema_version",
    "experiment",
    "unit",
    "run_kind",
    "method",
    "variant",
    "scheduler",
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
    "mma_tiler_m",
    "mma_tiler_n",
    "cluster_m",
    "cluster_n",
    "use_2cta_instrs",
    "use_tma_store",
    "max_active_clusters",
    "seed",
    "reference",
    "atol",
    "rtol",
    "correctness",
    "max_abs_error",
    "max_rel_error",
    "compile_time_ms",
    "first_launch_ms",
    "kernel_time_ms",
    "warmup_iterations",
    "iterations",
    "cache_mode",
    "gpu_name",
    "gpu_uuid",
    "compute_capability",
    "driver_version",
    "cuda_toolkit_version",
    "torch_cuda_version",
    "cutedsl_version",
    "cutlass_commit",
    "upstream_kernel_file",
    "upstream_kernel_git_blob",
    "upstream_kernel_sha256",
    "git_commit",
    "git_dirty",
    "publishable",
)

# The schemas P3.4 must neither reuse nor reinterpret.
CLOSED_SCHEMA_VERSIONS = ("p32.v1", "p33.v1")

# The frozen three-candidate table, restated independently of the wrapper.
# (upstream class, scheduler, tiler, cluster, use_2cta_instrs, source, persistent)
EXPECTED_VARIANT_TABLE = {
    "nonpersistent_1cta": (
        "DenseGemmKernel", "nonpersistent", (128, 128), (1, 1), False,
        "nonpersistent", False,
    ),
    "persistent_1cta": (
        "PersistentDenseGemmKernel", "static_persistent", (128, 128), (1, 1), False,
        "persistent", True,
    ),
    "persistent_2cta": (
        "PersistentDenseGemmKernel", "static_persistent", (256, 128), (2, 1), True,
        "persistent", True,
    ),
}
EXPECTED_VARIANT_ORDER = ("nonpersistent_1cta", "persistent_1cta", "persistent_2cta")

EXPECTED_MAX_ACTIVE_CLUSTERS_NOT_APPLICABLE = "not_applicable"

EXPECTED_FROZEN_CONFIG = {
    "schema_version": "p34.v1",
    "experiment": "exp03_cutedsl_vs_cublaslt",
    "unit": "P3.4",
    "run_kind": "smoke",
    "method": "cutedsl",
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
    "use_tma_store": True,
    "seed": 1111,
    "reference": "torch_cuda_fp32_ieee",
    "atol": 1e-1,
    "rtol": 1e-5,
    "cache_mode": "hot",
    "publishable": False,
}

EXPECTED_CUDA_ARCH = "sm_103a"
EXPECTED_COMPUTE_CAPABILITY = "10.3"

EXPECTED_FIXED_ROW_VALUES = {
    "schema_version": "p34.v1",
    "experiment": "exp03_cutedsl_vs_cublaslt",
    "unit": "P3.4",
    "run_kind": "smoke",
    "method": "cutedsl",
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
    "use_tma_store": "true",
    "seed": "1111",
    "reference": "torch_cuda_fp32_ieee",
    "correctness": "PASS",
    "atol": "0.100000000",
    "rtol": "0.000010000",
    "cache_mode": "hot",
    "publishable": "false",
}

TIMING_FIELDS = ("compile_time_ms", "first_launch_ms", "kernel_time_ms")
ERROR_FIELDS = ("max_abs_error", "max_rel_error")
TOLERANCE_FIELDS = ("atol", "rtol")
COUNT_FIELDS = ("warmup_iterations", "iterations")
BOOL_FIELDS = ("use_2cta_instrs", "use_tma_store", "git_dirty", "publishable")
GEOMETRY_INT_FIELDS = ("mma_tiler_m", "mma_tiler_n", "cluster_m", "cluster_n")
TIMING_DECIMALS = 6
ERROR_DECIMALS = 9
TOLERANCE_DECIMALS = 9

MIN_ITERATIONS = 1
MAX_WARMUP_ITERATIONS = 100
MAX_ITERATIONS = 100
SMOKE_WARMUP_ITERATIONS = 2
SMOKE_ITERATIONS = 10

ALLOWED_CLI_OPTIONS = frozenset(
    {"--help", "--self-test", "--warmup-iterations", "--iterations"}
)

# Option spellings that would reopen a frozen scientific parameter.
FORBIDDEN_CLI_FRAGMENTS = (
    "mnkl", "shape", "dtype", "major", "tiler", "cluster", "tma", "persistent",
    "2cta", "cta-group", "cta_group", "scheduler", "variant", "method", "seed",
    "tolerance", "atol", "rtol", "skip-ref", "skip_ref", "cold-l2", "cold_l2",
    "ref-check", "ref_check", "source", "path", "example", "publish", "cache",
    "gpu", "device", "autotune",
)

# Identifier fragments that must not appear as code in the wrapper. Prose in
# docstrings and comments is exempt: the scan runs over Python NAME tokens
# only, so a sentence explaining that P3.4 computes no TFLOP/s and makes no
# comparison is fine, while a tflops variable is not.
FORBIDDEN_SOURCE_IDENTIFIERS = (
    "tflop", "flops", "speedup", "efficiency", "bandwidth", "utilization",
    "winner", "ranking", "nsight", "autotune", "campaign", "cublas",
    "skip_ref_check", "use_cold_l2",
)

# Torch/framework operators that must never replace a CuTe DSL kernel as the
# measured path. The untimed FP32 oracle deliberately uses torch.einsum.
FORBIDDEN_TORCH_MATMUL_ATTRS = frozenset(
    {"matmul", "mm", "bmm", "addmm", "baddbmm", "addbmm", "linear"}
)

# The upstream helpers P3.4 must never use, because they fuse compilation, the
# first launch, correctness, and benchmarking into one number.
FORBIDDEN_UPSTREAM_HELPERS = ("run", "benchmark", "compile_bmm", "prepare_tensors")

GPU_STACK_MODULES = ("torch", "cutlass", "cuda", "numpy", "pynvml")

# The three new Phase 3 keys P3.4 introduces, plus the P3.1 keys it reuses.
REQUIRED_P34_CONTRACT_KEYS = (
    "CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH",
    "CUTEDSL_P34_PERSISTENT_EXAMPLE_GIT_BLOB",
    "CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256",
)
REQUIRED_P31_CONTRACT_KEYS = (
    "CUTEDSL_P31_EXAMPLE_PATH",
    "CUTEDSL_P31_EXAMPLE_GIT_BLOB",
    "CUTEDSL_P31_EXAMPLE_SHA256",
)

# Markers that must exist inside the pinned persistent example, proving the
# official static persistent scheduler and its 2-CTA selection are really the
# code path the two persistent variants exercise.
PERSISTENT_SOURCE_MARKERS = (
    "StaticPersistentTileScheduler",
    "CtaGroup.TWO",
    "PersistentDenseGemmKernel",
)
NON_PERSISTENT_SOURCE_MARKERS = ("DenseGemmKernel", "create_tensors")
# The non-persistent example must NOT contain the persistent scheduler, or the
# two sources would not be distinguishable.
NON_PERSISTENT_FORBIDDEN_MARKERS = ("StaticPersistentTileScheduler",)

_RE_HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")
_RE_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_RE_GPU_UUID = re.compile(r"\AGPU-[0-9a-fA-F][0-9a-fA-F-]+\Z")
_RE_DOTTED_VERSION = re.compile(r"\A[0-9]+(\.[0-9]+)*\Z")
_RE_COMPUTE_CAPABILITY = re.compile(r"\A[0-9]+\.[0-9]+\Z")
_RE_POSITIVE_INT = re.compile(r"\A[1-9][0-9]*\Z")
_RE_ENV_LINE = re.compile(r"\A([A-Z][A-Z0-9_]*)=(\S*)\Z")
_RE_SAFE_TEXT = re.compile(r"\A[^\x00-\x1f\x7f]+\Z")
_RE_UPSTREAM_REL_PATH = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._/-]*\.py\Z")

# P3.4 retains the closed P3.2/P3.3 FP32 oracle policy.
FORBIDDEN_FP32_API_SPELLINGS = (
    "allow_tf32", "set_float32_matmul_precision", "get_float32_matmul_precision",
)
REQUIRED_FP32_API_SPELLINGS = ("fp32_precision",)
FP32_PRECISION_IEEE = "ieee"

FORBIDDEN_SMOKE_FILTERS = ("grep -v", "| sed", "| tail", "| head", "| awk", "| grep")

SMOKE_TARGET = "gemm-cutedsl-p34-smoke"
CHECK_TARGET = "gemm-cutedsl-p34-check"
CHECK_PREREQUISITE = "gemm-cublaslt-p33-check"
LAUNCHER_RELATIVE_PATH = "scripts/run_container.sh"
LAUNCHER_DATA_MODE_VARIABLE = "RUN_CONTAINER_STDOUT_IS_DATA"
GPU_INDEX_VARIABLE = "BLACKWELL_GPU_INDEX"
SMOKE_SUCCESS_SENTENCE = (
    "P3.4 smoke completed: every variant passed correctness before its warm-up and "
    "steady-state timing."
)

# Status assertions that must remain true for the closed units.
CLOSED_STATUS_LINES = (
    "| P3.1 | Pinned official CuTe DSL example | YES | YES | YES |",
    "| P3.2 | One-shape wrapper | YES | YES | YES |",
    "| P3.3 | cuBLASLt baseline | YES | YES | YES |",
)
# P3.5 is a later unit that progresses on its own, so it is checked for having a
# status row and for not being recorded as audited or GB300-verified without
# being implemented - never pinned to one value, which would structurally forbid
# P3.5 from ever being implemented.
LATER_STATUS_PREFIXES = ("| P3.5 | Five shapes and comparison |",)
IMPOSSIBLE_LATER_STATUS_LINES = (
    "| P3.5 | Five shapes and comparison | NO | YES |",
    "| P3.5 | Five shapes and comparison | NO | NO | YES |",
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

# Drives the real main() with _measure_variant replaced by a function that
# fails at exactly one of the three positions, proving the all-or-nothing
# stdout contract through the real descriptor plumbing.
FAILING_POSITION_PROBE = (
    _GUARD_PRELUDE
    + """
import importlib.util

spec = importlib.util.spec_from_file_location("p34_probe", {wrapper!r})
module = importlib.util.module_from_spec(spec)
sys.modules["p34_probe"] = module
spec.loader.exec_module(module)

_FAIL_AT = {fail_at!r}
_state = {{"calls": 0}}
_ORDER = module.FROZEN_VARIANT_ORDER


def _fake_measure_variant(spec, **kwargs):
    index = _state["calls"]
    _state["calls"] += 1
    if index == _FAIL_AT:
        raise module.CorrectnessError(
            "synthetic failure at variant position " + str(index)
        )
    return module._synthetic_row(_ORDER[index])


def _fake_execute(warmup_iterations, iterations):
    rows = []
    for spec in module.FROZEN_VARIANTS:
        rows.append(_fake_measure_variant(spec))
    return module.serialize_rows(rows)


module.execute_measurement = _fake_execute
sys.exit(module.main([]))
"""
)

# The same probe with no injected failure, proving the success path really does
# write exactly four lines to stdout.
SUCCESS_PATH_PROBE = (
    _GUARD_PRELUDE
    + """
import importlib.util

spec = importlib.util.spec_from_file_location("p34_probe", {wrapper!r})
module = importlib.util.module_from_spec(spec)
sys.modules["p34_probe"] = module
spec.loader.exec_module(module)


def _fake_execute(warmup_iterations, iterations):
    return module.serialize_rows(module._synthetic_rows())


module.execute_measurement = _fake_execute
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
        if re.search(
            r"tflop|flops|speedup|efficien|bandwidth|utilization|throughput|winner|rank",
            field,
        ):
            errors.append(f"the CSV schema exposes a performance metric: {field}")
    return errors


def validate_variant_table(table) -> list:
    """Check the frozen three-candidate table exactly, including its order."""
    errors = []
    if not isinstance(table, (list, tuple)):
        return ["the frozen variant table is not a sequence"]
    if len(table) != 3:
        errors.append(f"P3.4 has exactly 3 variants, the table has {len(table)}")
    names = tuple(spec.get("variant") for spec in table)
    if len(set(names)) != len(names):
        errors.append(f"the variant table contains a duplicate variant: {names}")
    if names != EXPECTED_VARIANT_ORDER:
        errors.append(f"the variant order {names} is not the frozen {EXPECTED_VARIANT_ORDER}")
    for spec in table:
        name = spec.get("variant")
        if name not in EXPECTED_VARIANT_TABLE:
            errors.append(f"{name!r} is not one of the three frozen variants")
            continue
        expected = EXPECTED_VARIANT_TABLE[name]
        actual = (
            spec.get("upstream_class"),
            spec.get("scheduler"),
            tuple(spec.get("mma_tiler_mn", ())),
            tuple(spec.get("cluster_shape_mn", ())),
            spec.get("use_2cta_instrs"),
            spec.get("source"),
            spec.get("persistent"),
        )
        if actual != expected:
            errors.append(f"variant {name}: {actual} != frozen {expected}")
        # The 2-CTA geometry must keep a per-CTA M extent of 128.
        tiler = tuple(spec.get("mma_tiler_mn", ()))
        cluster = tuple(spec.get("cluster_shape_mn", ()))
        if spec.get("use_2cta_instrs"):
            if len(tiler) != 2 or len(cluster) != 2 or cluster[0] != 2:
                errors.append(f"variant {name}: a 2-CTA variant needs a cluster M of 2")
            elif tiler[0] // cluster[0] != 128:
                errors.append(
                    f"variant {name}: tiler M {tiler[0]} over cluster M {cluster[0]} gives a "
                    f"per-CTA M extent of {tiler[0] // cluster[0]}, not 128"
                )
        # A persistent variant must use the persistent class and source, and a
        # non-persistent variant must not.
        if spec.get("persistent") and spec.get("upstream_class") != "PersistentDenseGemmKernel":
            errors.append(
                f"variant {name}: a persistent variant is routed to "
                f"{spec.get('upstream_class')!r}"
            )
        if not spec.get("persistent") and spec.get("upstream_class") != "DenseGemmKernel":
            errors.append(
                f"variant {name}: a non-persistent variant is routed to "
                f"{spec.get('upstream_class')!r}"
            )
        if spec.get("persistent") and spec.get("source") != "persistent":
            errors.append(f"variant {name}: a persistent variant reads the wrong source")
        if not spec.get("persistent") and spec.get("source") != "nonpersistent":
            errors.append(f"variant {name}: a non-persistent variant reads the wrong source")
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
        if isinstance(expected, bool):
            if actual is not expected:
                errors.append(f"frozen {key} is {actual!r}, expected {expected!r}")
        elif isinstance(expected, float):
            if not isinstance(actual, float) or actual != expected:
                errors.append(f"frozen {key} is {actual!r}, expected {expected!r}")
        elif actual != expected:
            errors.append(f"frozen {key} is {actual!r}, expected {expected!r}")
    unknown = sorted(set(config) - set(EXPECTED_FROZEN_CONFIG))
    if unknown:
        errors.append(f"the frozen configuration has unexpected key(s): {', '.join(unknown)}")
    if config.get("schema_version") in CLOSED_SCHEMA_VERSIONS:
        errors.append(f"P3.4 reuses a closed schema version: {config.get('schema_version')}")
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

    variant = row["variant"]
    if variant not in EXPECTED_VARIANT_TABLE:
        errors.append(f"variant: {variant!r} is not one of the three frozen variants")
        return errors
    (_class, scheduler, tiler, cluster, use_2cta, _source, persistent) = (
        EXPECTED_VARIANT_TABLE[variant]
    )
    for field, expected in (
        ("scheduler", scheduler),
        ("mma_tiler_m", str(tiler[0])),
        ("mma_tiler_n", str(tiler[1])),
        ("cluster_m", str(cluster[0])),
        ("cluster_n", str(cluster[1])),
        ("use_2cta_instrs", "true" if use_2cta else "false"),
    ):
        if row[field] != expected:
            errors.append(f"{variant}: {field}={row[field]!r} != frozen {expected!r}")

    if persistent:
        if not _RE_POSITIVE_INT.match(row["max_active_clusters"]):
            errors.append(
                f"{variant}: max_active_clusters={row['max_active_clusters']!r} must be a "
                "positive decimal integer for a persistent variant"
            )
    elif row["max_active_clusters"] != EXPECTED_MAX_ACTIVE_CLUSTERS_NOT_APPLICABLE:
        errors.append(
            f"{variant}: max_active_clusters={row['max_active_clusters']!r} must be "
            f"{EXPECTED_MAX_ACTIVE_CLUSTERS_NOT_APPLICABLE!r} for the non-persistent variant"
        )

    for field in BOOL_FIELDS:
        if row[field] not in ("true", "false"):
            errors.append(f"{field}: {row[field]!r} is not a canonical lowercase boolean")
    for field in GEOMETRY_INT_FIELDS + COUNT_FIELDS:
        if not _RE_POSITIVE_INT.match(row[field]):
            errors.append(f"{field}: {row[field]!r} is not a positive integer")
    for field in COUNT_FIELDS:
        if _RE_POSITIVE_INT.match(row[field]):
            maximum = MAX_WARMUP_ITERATIONS if field == "warmup_iterations" else MAX_ITERATIONS
            if not MIN_ITERATIONS <= int(row[field]) <= maximum:
                errors.append(f"{field}: {row[field]} is outside [{MIN_ITERATIONS}, {maximum}]")

    for field in TIMING_FIELDS:
        _validate_decimal_field(field, row[field], TIMING_DECIMALS, True, errors)
    for field in ERROR_FIELDS:
        _validate_decimal_field(field, row[field], ERROR_DECIMALS, False, errors)
    for field in TOLERANCE_FIELDS:
        _validate_decimal_field(field, row[field], TOLERANCE_DECIMALS, True, errors)

    if not _RE_HEX40.match(row["cutlass_commit"]):
        errors.append(f"cutlass_commit: {row['cutlass_commit']!r} is not a 40-hex commit")
    if not _RE_HEX40.match(row["git_commit"]):
        errors.append(f"git_commit: {row['git_commit']!r} is not a 40-hex commit")
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
    """Require exactly one header and exactly three rows in the frozen order."""
    errors = []
    if not isinstance(text, str):
        return ["the serialized output is not a string"]
    lines = text.splitlines()
    if len(lines) != 4:
        errors.append(f"the serialized output has {len(lines)} line(s), expected exactly 4")
        return errors
    if lines[0] != ",".join(EXPECTED_CSV_FIELDS):
        errors.append("the CSV header does not match the frozen field order")
    parsed = [dict(row) for row in csv.DictReader(io.StringIO(text))]
    if len(parsed) != 3:
        errors.append(f"the serialized output parses to {len(parsed)} row(s), expected 3")
        return errors
    order = tuple(row["variant"] for row in parsed)
    if order != EXPECTED_VARIANT_ORDER:
        errors.append(f"the emitted variant order {order} is not {EXPECTED_VARIANT_ORDER}")
    # Every row must describe the same shape, seed, and provenance run.
    for field in ("m", "n", "k", "l", "seed", "git_commit", "gpu_uuid"):
        values = {row[field] for row in parsed}
        if len(values) != 1:
            errors.append(f"{field} differs between rows: {sorted(values)}")
    # The two persistent rows must name the same upstream source, and the
    # non-persistent row must name a different one.
    files = [row["upstream_kernel_file"] for row in parsed]
    if files[1] != files[2]:
        errors.append("the two persistent rows name different upstream sources")
    if files[0] == files[1]:
        errors.append(
            "the non-persistent row names the same upstream source as the persistent rows"
        )
    for row in parsed:
        errors.extend(validate_row_mapping(row))
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
    errors.extend(validate_no_framework_matmul(source))
    errors.extend(validate_no_upstream_helpers(source))
    errors.extend(validate_no_result_files(source))
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
                "path must be a pinned CuTe DSL kernel"
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in FORBIDDEN_TORCH_MATMUL_ATTRS:
                errors.append(
                    f"line {node.lineno}: the wrapper calls .{node.func.attr}(); a framework "
                    "operation must never be a measured variant"
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
        # Only a call on an upstream module object is a violation; the wrapper's
        # own subprocess.run() and similar are not.
        target = node.func.value
        target_name = target.id if isinstance(target, ast.Name) else (
            target.attr if isinstance(target, ast.Attribute) else None
        )
        if target_name in ("module", "modules", "factory_module", "upstream", "mod"):
            errors.append(
                f"line {node.lineno}: the wrapper calls the upstream helper "
                f"{target_name}.{node.func.attr}(); P3.4 owns every timer and never uses "
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
    """Require the PyTorch 2.10 fp32_precision API and nothing overlapping.

    The scan is structural: only real attribute accesses, bare names, and
    getattr/setattr strings count, so prose explaining why the legacy aliases
    are never used stays legal while any actual access fails.
    """
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


def validate_phase3_contract(values) -> list:
    """Require both upstream pin sets, well-formed and distinct."""
    errors = []
    for key in REQUIRED_P31_CONTRACT_KEYS + REQUIRED_P34_CONTRACT_KEYS:
        if key not in values:
            errors.append(f"{PHASE3_CONTRACT_FILE} is missing required key {key}")
    if errors:
        return errors

    persistent_path = values["CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH"]
    non_persistent_path = values["CUTEDSL_P31_EXAMPLE_PATH"]
    if persistent_path == non_persistent_path:
        errors.append("the persistent and non-persistent example pins name the same file")
    if not persistent_path.endswith("dense_gemm_persistent.py"):
        errors.append(
            f"CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH={persistent_path!r} does not name the "
            "official persistent example"
        )
    if not non_persistent_path.endswith("dense_gemm.py"):
        errors.append(
            f"CUTEDSL_P31_EXAMPLE_PATH={non_persistent_path!r} does not name the official "
            "non-persistent example"
        )
    for path_key in ("CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH", "CUTEDSL_P31_EXAMPLE_PATH"):
        path = values[path_key]
        if not _RE_UPSTREAM_REL_PATH.match(path) or ".." in Path(path).parts:
            errors.append(f"{path_key}={path!r} is not a safe relative upstream path")
    for blob_key in ("CUTEDSL_P34_PERSISTENT_EXAMPLE_GIT_BLOB", "CUTEDSL_P31_EXAMPLE_GIT_BLOB"):
        if not _RE_HEX40.match(values[blob_key]):
            errors.append(f"{blob_key} is not a 40-hex Git blob")
    for sha_key in ("CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256", "CUTEDSL_P31_EXAMPLE_SHA256"):
        if not _RE_HEX64.match(values[sha_key]):
            errors.append(f"{sha_key} is not a 64-hex SHA-256")
    if (values["CUTEDSL_P34_PERSISTENT_EXAMPLE_GIT_BLOB"]
            == values["CUTEDSL_P31_EXAMPLE_GIT_BLOB"]):
        errors.append("the two example pins share a Git blob; they cannot be different files")
    if (values["CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256"]
            == values["CUTEDSL_P31_EXAMPLE_SHA256"]):
        errors.append("the two example pins share a SHA-256; they cannot be different files")
    return errors


def validate_upstream_source_content(name, text) -> list:
    """Prove a pinned example really contains the code path P3.4 relies on."""
    errors = []
    if name == "persistent":
        for marker in PERSISTENT_SOURCE_MARKERS:
            if marker not in text:
                errors.append(
                    f"the pinned persistent example does not contain {marker!r}; the official "
                    "static persistent scheduler / 2-CTA path cannot be the measured one"
                )
    else:
        for marker in NON_PERSISTENT_SOURCE_MARKERS:
            if marker not in text:
                errors.append(
                    f"the pinned non-persistent example does not contain {marker!r}"
                )
        for marker in NON_PERSISTENT_FORBIDDEN_MARKERS:
            if marker in text:
                errors.append(
                    f"the pinned non-persistent example contains {marker!r}; the two sources "
                    "would not be distinguishable"
                )
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
    # Both official sources must be revalidated. After Make-variable expansion
    # the two pinned digests are literal 64-hex strings, so requiring two
    # DISTINCT ones is a stronger signal than counting sha256sum call sites -
    # and it does not care whether the recipe uses two explicit blocks or a
    # loop over both files.
    digests = set(re.findall(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])", joined))
    if len(digests) < 2:
        errors.append(
            f"the smoke target references {len(digests)} pinned SHA-256 digest(s); BOTH "
            "official sources must be revalidated inside the GPU container"
        )

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
    if "FUNCTIONAL VERIFICATION" not in upper:
        errors.append("the smoke target does not state that this is functional verification")
    if "NON-PUBLISHABLE" not in upper:
        errors.append("the smoke target does not state that the timings are non-publishable")
    if "COMPARISON" not in upper:
        errors.append(
            "the smoke target does not state that no variant or cuBLASLt comparison was made"
        )
    # P3.4 must not drive the P3.3 executable.
    if P33_WRAPPER_RELATIVE_PATH in joined:
        errors.append("the smoke target invokes the P3.3 cuBLASLt wrapper")

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
    ):
        if required not in joined:
            errors.append(f"{CHECK_TARGET} does not use {required!r} ({description})")
    if "sha256sum" not in joined:
        errors.append(f"{CHECK_TARGET} never computes a SHA-256")
    if "hash-object" not in joined:
        errors.append(f"{CHECK_TARGET} never computes a Git blob SHA")
    # Both sources must be validated by blob AND digest. The recipe passes the
    # pinned expectations in as environment variables, so requiring all four
    # names works whether the recipe uses two explicit blocks or a loop.
    for expectation in ("P31_EXAMPLE_GIT_BLOB", "P31_EXAMPLE_SHA256",
                        "P34_EXAMPLE_GIT_BLOB", "P34_EXAMPLE_SHA256"):
        if expectation not in joined:
            errors.append(
                f"{CHECK_TARGET} never checks {expectation}; BOTH official sources must be "
                "validated by Git blob and SHA-256"
            )
    for marker in ("StaticPersistentTileScheduler", "CtaGroup.TWO"):
        if marker not in joined:
            errors.append(
                f"{CHECK_TARGET} does not assert the pinned persistent source contains "
                f"{marker!r}"
            )
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
    """Require the audited/verified P3.4 closure and untouched closed units."""
    errors = []
    if "| P3.4 | Three execution variants | YES | YES | YES |" not in plan_text:
        errors.append("PLAN.md does not record P3.4 as YES / YES / YES")
    for stale_or_invalid in (
        "| P3.4 | Three execution variants | YES | YES | NO |",
        "| P3.4 | Three execution variants | YES | NO | YES |",
        "| P3.4 | Three execution variants | YES | NO | NO |",
        "| P3.4 | Three execution variants | NO | NO | NO |",
    ):
        if stale_or_invalid in plan_text:
            errors.append(f"PLAN.md records a stale or invalid P3.4 status: {stale_or_invalid!r}")
    for closed in CLOSED_STATUS_LINES:
        if closed not in plan_text:
            errors.append(f"PLAN.md no longer records {closed!r}")
    for later in LATER_STATUS_PREFIXES:
        if later not in plan_text:
            errors.append(f"PLAN.md no longer records a status row for {later!r}")
    for impossible in IMPOSSIBLE_LATER_STATUS_LINES:
        if impossible in plan_text:
            errors.append(
                f"PLAN.md records a later unit as audited or verified without being "
                f"implemented: {impossible!r}"
            )

    if "P3.4 = YES / YES / YES" not in protocol_text:
        errors.append(f"{PROTOCOL_RELATIVE_PATH} does not state P3.4 = YES / YES / YES")
    if "P3.4 creates no publishable performance result" not in protocol_text:
        errors.append(
            f"{PROTOCOL_RELATIVE_PATH} does not state that P3.4 creates no publishable result"
        )
    if "P3.4 (three execution variants)" not in readme_text:
        errors.append("README.md does not describe P3.4")
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
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False
    )


def _run_probe(script):
    return subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=300, check=False
    )


def check_wrapper(repo_root) -> list:
    """Run the whole contract check against the real repository."""
    errors = []
    root = Path(repo_root).resolve()

    wrapper_path = root / WRAPPER_RELATIVE_PATH
    checker_path = root / CHECKER_RELATIVE_PATH
    protocol_path = root / PROTOCOL_RELATIVE_PATH
    makefile_path = root / "Makefile"
    plan_path = root / "PLAN.md"
    readme_path = root / "README.md"

    for path in (wrapper_path, checker_path, protocol_path, makefile_path, plan_path,
                 readme_path):
        if not path.is_file():
            errors.append(f"required file is missing: {path.relative_to(root)}")
    if errors:
        return errors

    wrapper_source = wrapper_path.read_text(encoding="utf-8")
    makefile_text = makefile_path.read_text(encoding="utf-8")

    module = _load_module(wrapper_path, "p34_wrapper_under_test")

    # 1. Schema, variant table, and frozen configuration.
    errors.extend(validate_csv_schema(module.CSV_FIELDS))
    errors.extend(validate_variant_table(module.FROZEN_VARIANTS))
    errors.extend(validate_frozen_config(module.FROZEN_CONFIG))
    if tuple(module.FROZEN_VARIANT_ORDER) != EXPECTED_VARIANT_ORDER:
        errors.append(
            f"the wrapper's variant order {tuple(module.FROZEN_VARIANT_ORDER)} is not "
            f"{EXPECTED_VARIANT_ORDER}"
        )
    if module.MAX_ACTIVE_CLUSTERS_NOT_APPLICABLE != EXPECTED_MAX_ACTIVE_CLUSTERS_NOT_APPLICABLE:
        errors.append(
            "the wrapper's not-applicable cluster marker is "
            f"{module.MAX_ACTIVE_CLUSTERS_NOT_APPLICABLE!r}"
        )
    for name, expected in (
        ("MIN_ITERATIONS", MIN_ITERATIONS),
        ("MAX_WARMUP_ITERATIONS", MAX_WARMUP_ITERATIONS),
        ("MAX_ITERATIONS", MAX_ITERATIONS),
    ):
        if getattr(module, name, None) != expected:
            errors.append(f"the wrapper declares {name}={getattr(module, name, None)!r}")

    # 2. Serialization of three synthetic rows, through the wrapper's own code.
    try:
        rows = module._synthetic_rows()
        errors.extend(validate_serialized_output(module.serialize_rows(rows)))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"the wrapper cannot serialize three synthetic valid rows: {exc}")
        rows = None

    # 3. Adversarial rows must be rejected by the wrapper's own validator.
    if rows is not None:
        adversarial = {
            "publishable=true": (0, {"publishable": "true"}),
            "correctness=FAIL": (0, {"correctness": "FAIL"}),
            "another shape": (0, {"m": "8192"}),
            "a changed dtype": (0, {"ab_dtype": "Float16"}),
            "a changed major": (0, {"c_major": "m"}),
            "a disabled TMA store": (0, {"use_tma_store": "false"}),
            "a changed seed": (0, {"seed": "2222"}),
            "a non-persistent row with a cluster count": (0, {"max_active_clusters": "148"}),
            "a non-persistent row claiming the persistent scheduler": (
                0, {"scheduler": "static_persistent"}),
            "a persistent row claiming the non-persistent scheduler": (
                1, {"scheduler": "nonpersistent"}),
            "a persistent row without a cluster count": (
                1, {"max_active_clusters": EXPECTED_MAX_ACTIVE_CLUSTERS_NOT_APPLICABLE}),
            "persistent_2cta with use_2cta_instrs=false": (2, {"use_2cta_instrs": "false"}),
            "persistent_2cta with a (1,1) cluster": (2, {"cluster_m": "1", "cluster_n": "1"}),
            "the wrong 2-CTA tiler": (2, {"mma_tiler_m": "128"}),
            "an unknown variant": (0, {"variant": "persistent_4cta"}),
            "a non-finite timing": (0, {"kernel_time_ms": "nan"}),
            "a zero timing": (0, {"compile_time_ms": "0.000000"}),
            "a performance-derived field": (0, {"tflops": "1.0"}),
            "a non-canonical boolean": (0, {"git_dirty": "True"}),
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

        # Variant-set adversarials go through validate_rows.
        for description, candidate_rows in (
            ("a missing variant", rows[:2]),
            ("an extra variant", rows + [rows[0]]),
            ("a duplicated variant", [rows[0], rows[0], rows[2]]),
            ("a reordered variant set", [rows[1], rows[0], rows[2]]),
        ):
            try:
                module.validate_rows(candidate_rows)
            except Exception:  # noqa: BLE001
                pass
            else:
                errors.append(f"the wrapper accepted {description}")

    # 4. A failed or skipped correctness check can never build a row.
    for bad in ("FAIL", "SKIPPED", ""):
        try:
            module.build_row(
                variant=EXPECTED_VARIANT_ORDER[0],
                correctness=bad,
                max_abs_error=0.0,
                max_rel_error=0.0,
                compile_time_ms=1.0,
                first_launch_ms=1.0,
                kernel_time_ms=1.0,
                warmup_iterations=2,
                iterations=10,
                max_active_clusters=None,
                provenance=module._synthetic_provenance(),
                upstream=module._synthetic_upstream("nonpersistent"),
            )
        except Exception:  # noqa: BLE001
            pass
        else:
            errors.append(f"the wrapper built a row with correctness={bad!r}")

    # 5. Source-level structure.
    errors.extend(validate_source(wrapper_source))
    errors.extend(validate_fp32_precision_policy(wrapper_source))

    # 6. The command line.
    errors.extend(validate_cli_options(
        set(re.findall(r"--[a-z0-9][a-z0-9-]*", module.build_arg_parser().format_help()))
    ))

    # 7. Version contracts and pin linkage.
    try:
        global_values = parse_env_file(root / GLOBAL_CONTRACT_FILE)
        phase3_values = parse_env_file(root / PHASE3_CONTRACT_FILE)
        errors.extend(validate_phase3_contract(phase3_values))
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
                f"P3.4 targets {EXPECTED_CUDA_ARCH!r}"
            )
        if wrapper_contract.get("EXPECTED_COMPUTE_CAPABILITY") != EXPECTED_COMPUTE_CAPABILITY:
            errors.append(
                f"{EXPECTED_CUDA_ARCH} must derive compute capability "
                f"{EXPECTED_COMPUTE_CAPABILITY}"
            )
        # P3.4 adds no key to the global contract.
        for forbidden in ("CUTEDSL_P34", "CUTEDSL_P31"):
            if any(key.startswith(forbidden) for key in global_values):
                errors.append(f"{GLOBAL_CONTRACT_FILE} gained a Phase 3 key ({forbidden}*)")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"the pinned contracts could not be resolved: {exc}")

    # 8. VERSIONS.env must still satisfy the closed P1/P2 aggregators.
    for relative in (P1_AGGREGATOR_RELATIVE_PATH, P2_AGGREGATOR_RELATIVE_PATH):
        aggregator_path = root / relative
        if not aggregator_path.is_file():
            errors.append(f"closed aggregator {relative} is missing")
            continue
        try:
            aggregator = _load_module(aggregator_path, f"p34_closed_{Path(relative).stem}")
            aggregator.parse_versions_env(root / GLOBAL_CONTRACT_FILE)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                f"{GLOBAL_CONTRACT_FILE} no longer parses under the closed {relative} "
                f"allowlist: {exc}"
            )

    # 9. The pinned upstream sources, when the checkout is present.
    if UPSTREAM_CHECKOUT_DIR.is_dir():
        try:
            phase3_values = parse_env_file(root / PHASE3_CONTRACT_FILE)
            for name, path_key in (
                ("persistent", "CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH"),
                ("nonpersistent", "CUTEDSL_P31_EXAMPLE_PATH"),
            ):
                example = UPSTREAM_CHECKOUT_DIR / phase3_values[path_key]
                if not example.is_file():
                    errors.append(f"the pinned {name} example {example} is missing")
                    continue
                errors.extend(validate_upstream_source_content(
                    name, example.read_text(encoding="utf-8")
                ))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"the pinned upstream sources could not be inspected: {exc}")
    else:
        print(
            f"check_cutedsl_variants_p34: note: {UPSTREAM_CHECKOUT_DIR} is absent; the "
            "upstream source-content checks are skipped here and run inside "
            f"make {CHECK_TARGET}",
            file=sys.stderr,
        )

    # 10. --help and --self-test are GPU-free.
    for argv in (["--help"], ["--self-test"]):
        completed = _run_guarded(wrapper_path, argv)
        if completed.returncode != 0:
            errors.append(
                f"the wrapper failed the GPU-free guard with {argv}: exit "
                f"{completed.returncode}; {completed.stderr.strip()[-300:]}"
            )
        if argv == ["--self-test"] and completed.stdout:
            errors.append("--self-test wrote to stdout; every diagnostic belongs on stderr")

    # 11. stdout is all-or-nothing: a failure in ANY of the three positions
    #     must produce no CSV, and the clean path must produce exactly four lines.
    for position in range(3):
        probe = FAILING_POSITION_PROBE.format(
            blocked=list(GPU_STACK_MODULES), wrapper=str(wrapper_path), fail_at=position
        )
        completed = _run_probe(probe)
        if completed.returncode == 0:
            errors.append(
                f"a synthetic failure at variant position {position} did not produce a "
                "non-zero exit status"
            )
        if completed.stdout:
            errors.append(
                f"a synthetic failure at variant position {position} still wrote "
                f"{len(completed.stdout)} byte(s) to stdout"
            )
    completed = _run_probe(
        SUCCESS_PATH_PROBE.format(blocked=list(GPU_STACK_MODULES), wrapper=str(wrapper_path))
    )
    if completed.returncode != 0:
        errors.append(f"the synthetic success path failed: {completed.stderr.strip()[-300:]}")
    else:
        errors.extend(validate_serialized_output(completed.stdout))

    # 12. Make integration. The Makefile `include`s both version contracts, so
    #     their KEY=VALUE pairs are Make variables too; merging them here is
    #     what lets a recipe written as $(CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256)
    #     be checked against the digest it actually expands to.
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

    # 13. Truthful status documents, and the closed units untouched.
    errors.extend(validate_status_documents(
        plan_path.read_text(encoding="utf-8"),
        protocol_path.read_text(encoding="utf-8"),
        readme_path.read_text(encoding="utf-8"),
    ))
    for relative, marker in (
        (P32_WRAPPER_RELATIVE_PATH, 'SCHEMA_VERSION = "p32.v1"'),
        (P33_WRAPPER_RELATIVE_PATH, 'SCHEMA_VERSION = "p33.v1"'),
    ):
        path = root / relative
        if not path.is_file():
            errors.append(f"closed unit file {relative} is missing")
        elif marker not in path.read_text(encoding="utf-8"):
            errors.append(f"P3.4 changed the closed {relative} schema version")
    if not (root / P33_BRIDGE_RELATIVE_PATH).is_file():
        errors.append(f"closed unit file {P33_BRIDGE_RELATIVE_PATH} is missing")
    for closed_target in ("gemm-cutedsl-p32-check", "gemm-cutedsl-p32-smoke",
                          "gemm-cublaslt-p33-check", "gemm-cublaslt-p33-smoke"):
        if not re.search(rf"^{re.escape(closed_target)}:", makefile_text, flags=re.M):
            errors.append(f"P3.4 removed the closed Make target {closed_target}")
    return errors


# --- Self-test ---------------------------------------------------------------


def _good_row(variant) -> dict:
    (_class, scheduler, tiler, cluster, use_2cta, source, persistent) = (
        EXPECTED_VARIANT_TABLE[variant]
    )
    row = dict(EXPECTED_FIXED_ROW_VALUES)
    row.update(
        {
            "variant": variant,
            "scheduler": scheduler,
            "mma_tiler_m": str(tiler[0]),
            "mma_tiler_n": str(tiler[1]),
            "cluster_m": str(cluster[0]),
            "cluster_n": str(cluster[1]),
            "use_2cta_instrs": "true" if use_2cta else "false",
            "max_active_clusters": (
                "148" if persistent else EXPECTED_MAX_ACTIVE_CLUSTERS_NOT_APPLICABLE
            ),
            "max_abs_error": "0.000000000",
            "max_rel_error": "0.000000000",
            "compile_time_ms": "1234.500000",
            "first_launch_ms": "12.250000",
            "kernel_time_ms": "7.500000",
            "warmup_iterations": "2",
            "iterations": "10",
            "gpu_name": "SYNTHETIC TEST DEVICE",
            "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
            "compute_capability": "9.9",
            "driver_version": "999.99.99",
            "cuda_toolkit_version": "99.9.9",
            "torch_cuda_version": "98.7",
            "cutedsl_version": "97.6.5",
            "cutlass_commit": "1" * 40,
            "upstream_kernel_file": (
                "examples/synthetic/dense_gemm_persistent.py" if source == "persistent"
                else "examples/synthetic/dense_gemm.py"
            ),
            "upstream_kernel_git_blob": ("3" * 40) if source == "persistent" else ("2" * 40),
            "upstream_kernel_sha256": ("4" * 64) if source == "persistent" else ("5" * 64),
            "git_commit": "0" * 40,
            "git_dirty": "false",
        }
    )
    return row


def _good_rows() -> list:
    return [_good_row(variant) for variant in EXPECTED_VARIANT_ORDER]


def _serialize(rows) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(EXPECTED_CSV_FIELDS), extrasaction="raise", lineterminator="\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


def _good_variant_table() -> list:
    table = []
    for variant in EXPECTED_VARIANT_ORDER:
        (cls, scheduler, tiler, cluster, use_2cta, source, persistent) = (
            EXPECTED_VARIANT_TABLE[variant]
        )
        table.append({
            "variant": variant, "upstream_class": cls, "scheduler": scheduler,
            "mma_tiler_mn": tiler, "cluster_shape_mn": cluster,
            "use_2cta_instrs": use_2cta, "source": source, "persistent": persistent,
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
    "\t\texec python3 src/gemm/cutedsl_variants.py \\",
    "\t\t\t--warmup-iterations 2 \\",
    "\t\t\t--iterations 10' || status=$$?; \\",
    "\techo \"P3.4 FUNCTIONAL VERIFICATION ONLY -- ALL TIMINGS ARE NON-PUBLISHABLE\" >&2; \\",
    "\techo \"and NO variant or cuBLASLt COMPARISON has been performed.\" >&2; \\",
    "\tif [ \"$$status\" -eq 0 ]; then \\",
    "\t\techo \"" + SMOKE_SUCCESS_SENTENCE + "\" >&2; \\",
    "\telse \\",
    "\t\techo \"P3.4 smoke FAILED\" >&2; \\",
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

    print("check_cutedsl_variants_p34 --self-test (GPU-free)", file=sys.stderr)

    # Schema.
    check("the frozen schema has 51 fields", len(EXPECTED_CSV_FIELDS) == 51,
          str(len(EXPECTED_CSV_FIELDS)))
    check("the frozen schema has no duplicate", len(set(EXPECTED_CSV_FIELDS)) == 51)
    check("the correct schema is accepted", validate_csv_schema(EXPECTED_CSV_FIELDS) == [])
    rejects("a reordered schema is rejected",
            validate_csv_schema(tuple(reversed(EXPECTED_CSV_FIELDS))), "wrong order")
    rejects("a schema with a performance field is rejected",
            validate_csv_schema(EXPECTED_CSV_FIELDS + ("tflops",)))
    rejects("a schema with a speedup field is rejected",
            validate_csv_schema(EXPECTED_CSV_FIELDS + ("speedup",)))
    rejects("a schema missing a field is rejected",
            validate_csv_schema(EXPECTED_CSV_FIELDS[:-1]), "missing field")

    # Variant table.
    good_table = _good_variant_table()
    check("the frozen variant table is accepted", validate_variant_table(good_table) == [],
          str(validate_variant_table(good_table)))
    rejects("a missing variant is rejected", validate_variant_table(good_table[:2]),
            "exactly 3")
    rejects("an extra variant is rejected",
            validate_variant_table(good_table + [good_table[0]]), "exactly 3")
    rejects("a duplicated variant is rejected",
            validate_variant_table([good_table[0], good_table[0], good_table[2]]), "duplicate")
    rejects("a reordered variant table is rejected",
            validate_variant_table([good_table[1], good_table[0], good_table[2]]), "order")
    rejects("persistent_2cta with use_2cta_instrs=false is rejected",
            validate_variant_table(
                good_table[:2] + [{**good_table[2], "use_2cta_instrs": False}]))
    rejects("persistent_2cta with a (1,1) cluster is rejected",
            validate_variant_table(
                good_table[:2] + [{**good_table[2], "cluster_shape_mn": (1, 1)}]),
            "cluster M of 2")
    rejects("the wrong 2-CTA tiler is rejected",
            validate_variant_table(
                good_table[:2] + [{**good_table[2], "mma_tiler_mn": (128, 128)}]),
            "per-CTA M extent")
    rejects("a persistent variant routed to DenseGemmKernel is rejected",
            validate_variant_table(
                [good_table[0], {**good_table[1], "upstream_class": "DenseGemmKernel"},
                 good_table[2]]),
            "routed to")
    rejects("a non-persistent variant routed to PersistentDenseGemmKernel is rejected",
            validate_variant_table(
                [{**good_table[0], "upstream_class": "PersistentDenseGemmKernel"}]
                + good_table[1:]),
            "routed to")
    rejects("a persistent variant reading the non-persistent source is rejected",
            validate_variant_table(
                [good_table[0], {**good_table[1], "source": "nonpersistent"}, good_table[2]]),
            "wrong source")
    rejects("a non-persistent variant reading the persistent source is rejected",
            validate_variant_table(
                [{**good_table[0], "source": "persistent"}] + good_table[1:]),
            "wrong source")

    # Frozen configuration.
    check("the correct frozen config is accepted",
          validate_frozen_config(dict(EXPECTED_FROZEN_CONFIG)) == [])
    for field, bad in (
        ("m", 8192), ("n", 2048), ("k", 8192), ("l", 2),
        ("ab_dtype", "Float16"), ("acc_dtype", "Float16"), ("c_dtype", "BFloat16"),
        ("a_major", "m"), ("b_major", "n"), ("c_major", "m"),
        ("use_tma_store", False), ("seed", 2222), ("atol", 1e-2), ("rtol", 1e-3),
        ("cache_mode", "cold"), ("publishable", True), ("method", "cublaslt"),
        ("schema_version", "p32.v1"), ("schema_version", "p33.v1"),
    ):
        rejects(f"a frozen config with {field}={bad!r} is rejected",
                validate_frozen_config({**EXPECTED_FROZEN_CONFIG, field: bad}))

    # Rows and output.
    rows = _good_rows()
    check("three well-formed rows are accepted",
          validate_serialized_output(_serialize(rows)) == [],
          str(validate_serialized_output(_serialize(rows))))
    check("a well-formed row is accepted", validate_row_mapping(rows[0]) == [])
    rejects("a two-row output is rejected", validate_serialized_output(_serialize(rows[:2])),
            "expected exactly 4")
    rejects("a four-row output is rejected",
            validate_serialized_output(_serialize(rows + [rows[0]])), "expected exactly 4")
    rejects("a header-only output is rejected",
            validate_serialized_output(_serialize([])), "expected exactly 4")
    rejects("an empty output is rejected", validate_serialized_output(""))
    rejects("a reordered output is rejected",
            validate_serialized_output(_serialize([rows[1], rows[0], rows[2]])), "order")
    rejects("rows describing different shapes are rejected",
            validate_serialized_output(_serialize([{**rows[0], "m": "8192"}] + rows[1:])))
    rejects("the two persistent rows naming different sources is rejected",
            validate_serialized_output(_serialize(
                rows[:2] + [{**rows[2], "upstream_kernel_file": "examples/other.py"}])),
            "different upstream sources")
    rejects("the non-persistent row sharing the persistent source is rejected",
            validate_serialized_output(_serialize(
                [{**rows[0],
                  "upstream_kernel_file": rows[1]["upstream_kernel_file"]}] + rows[1:])),
            "same upstream source")

    for description, (index, override) in sorted({
        "publishable=true": (0, {"publishable": "true"}),
        "correctness=FAIL": (0, {"correctness": "FAIL"}),
        "correctness=SKIPPED": (0, {"correctness": "SKIPPED"}),
        "another shape": (0, {"k": "8192"}),
        "a changed dtype": (0, {"ab_dtype": "Float16"}),
        "a changed major": (0, {"a_major": "m"}),
        "a disabled TMA store": (0, {"use_tma_store": "false"}),
        "a changed seed": (0, {"seed": "2222"}),
        "an unknown variant": (0, {"variant": "persistent_4cta"}),
        "a non-persistent row with a cluster count": (0, {"max_active_clusters": "148"}),
        "a non-persistent row claiming the persistent scheduler": (
            0, {"scheduler": "static_persistent"}),
        "a persistent row claiming the non-persistent scheduler": (
            1, {"scheduler": "nonpersistent"}),
        "a persistent row without a cluster count": (
            1, {"max_active_clusters": EXPECTED_MAX_ACTIVE_CLUSTERS_NOT_APPLICABLE}),
        "a persistent row with a zero cluster count": (1, {"max_active_clusters": "0"}),
        "persistent_2cta with use_2cta_instrs=false": (2, {"use_2cta_instrs": "false"}),
        "persistent_2cta with a (1,1) cluster": (2, {"cluster_m": "1", "cluster_n": "1"}),
        "the wrong 2-CTA tiler": (2, {"mma_tiler_m": "128"}),
        "a NaN timing": (0, {"kernel_time_ms": "nan"}),
        "an infinite timing": (0, {"compile_time_ms": "inf"}),
        "a zero timing": (0, {"first_launch_ms": "0.000000"}),
        "an exponent-notation timing": (0, {"kernel_time_ms": "1e3"}),
        "a wrong-precision timing": (0, {"kernel_time_ms": "7.5"}),
        "a negative error": (0, {"max_abs_error": "-0.000000001"}),
        "a non-canonical boolean": (0, {"git_dirty": "TRUE"}),
        "a malformed GPU UUID": (0, {"gpu_uuid": "0000"}),
        "a malformed upstream blob": (0, {"upstream_kernel_git_blob": "abc"}),
        "a malformed upstream digest": (0, {"upstream_kernel_sha256": "abc"}),
        "an absolute upstream path": (0, {"upstream_kernel_file": "/opt/cutlass/x.py"}),
        "an out-of-range iteration count": (0, {"iterations": "101"}),
    }.items()):
        rejects(f"a row with {description} is rejected",
                validate_row_mapping({**rows[index], **override}))
    rejects("a row missing a field is rejected",
            validate_row_mapping({k: v for k, v in rows[0].items() if k != "scheduler"}),
            "missing field")
    rejects("a row with a performance-derived field is rejected",
            validate_row_mapping({**rows[0], "tflops": "1.0"}), "unknown field")

    # CLI.
    check("the permitted option set is accepted",
          validate_cli_options(ALLOWED_CLI_OPTIONS) == [])
    for bad in ("--variant", "--scheduler", "--mma-tiler", "--cluster-shape", "--persistent",
                "--use-2cta-instrs", "--seed", "--atol", "--skip-ref-check", "--shape",
                "--ab-dtype", "--c-major", "--example-path", "--use-tma-store"):
        rejects(f"the forbidden option {bad} is rejected",
                validate_cli_options(set(ALLOWED_CLI_OPTIONS) | {bad}))
    rejects("a missing permitted option is rejected",
            validate_cli_options(ALLOWED_CLI_OPTIONS - {"--iterations"}), "missing option")

    # Wrapper source structure.
    check("a clean wrapper body is accepted",
          validate_source("import torch\n\n\ndef f(x):\n    return x\n") == [])
    for description, source in (
        ("a TFLOP/s variable", "tflops = 1.0\n"),
        ("a speedup variable", "speedup_vs_persistent = 2.0\n"),
        ("a ranking variable", "ranking = []\n"),
        ("a cuBLASLt reference", "cublaslt_row = 1\n"),
        ("a torch.matmul path", "c = torch.matmul(a, b)\n"),
        ("a torch.bmm path", "c = torch.bmm(a, b)\n"),
        ("an @ operator path", "c = a @ b\n"),
        ("an upstream run() call", "module.run(1)\n"),
        ("an upstream benchmark call", "module.benchmark(1)\n"),
        ("a result-file write", "open('results/raw/x.csv', 'w')\n"),
    ):
        rejects(f"{description} is rejected", validate_source(source))
    check("torch.einsum as the untimed oracle is allowed",
          validate_source("r = torch.einsum('mkl,nkl->mnl', a, b)\n") == [])
    check("subprocess.run is not mistaken for an upstream helper",
          validate_no_upstream_helpers("import subprocess\nsubprocess.run(['git'])\n") == [])

    # FP32 policy.
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

    # Phase 3 contract.
    good_contract = {
        "CUTEDSL_P31_EXAMPLE_PATH": "examples/a/dense_gemm.py",
        "CUTEDSL_P31_EXAMPLE_GIT_BLOB": "a" * 40,
        "CUTEDSL_P31_EXAMPLE_SHA256": "b" * 64,
        "CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH": "examples/a/dense_gemm_persistent.py",
        "CUTEDSL_P34_PERSISTENT_EXAMPLE_GIT_BLOB": "c" * 40,
        "CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256": "d" * 64,
    }
    check("a well-formed Phase 3 contract is accepted",
          validate_phase3_contract(good_contract) == [],
          str(validate_phase3_contract(good_contract)))
    for key in REQUIRED_P34_CONTRACT_KEYS:
        rejects(f"a contract missing {key} is rejected",
                validate_phase3_contract({k: v for k, v in good_contract.items() if k != key}),
                "missing required key")
    rejects("a mismatched persistent path is rejected",
            validate_phase3_contract(
                {**good_contract,
                 "CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH": "examples/a/dense_gemm.py"}))
    rejects("a shared persistent blob is rejected",
            validate_phase3_contract(
                {**good_contract, "CUTEDSL_P34_PERSISTENT_EXAMPLE_GIT_BLOB": "a" * 40}),
            "share a Git blob")
    rejects("a shared persistent SHA-256 is rejected",
            validate_phase3_contract(
                {**good_contract, "CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256": "b" * 64}),
            "share a SHA-256")
    rejects("a malformed persistent blob is rejected",
            validate_phase3_contract(
                {**good_contract, "CUTEDSL_P34_PERSISTENT_EXAMPLE_GIT_BLOB": "xyz"}),
            "40-hex")
    rejects("a malformed persistent SHA-256 is rejected",
            validate_phase3_contract(
                {**good_contract, "CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256": "xyz"}),
            "64-hex")

    # Pin linkage.
    check("a wrapper that reads both contracts is accepted",
          validate_pin_linkage('"VERSIONS.env"\n"PHASE3_VERSIONS.env"\n',
                               {"CUTLASS_COMMIT": "e" * 40}) == [])
    rejects("a wrapper duplicating a pinned hash is rejected",
            validate_pin_linkage(
                '"VERSIONS.env"\n"PHASE3_VERSIONS.env"\nCOMMIT = "' + "e" * 40 + '"\n',
                {"CUTLASS_COMMIT": "e" * 40}),
            "duplicates the pinned")

    # Upstream source content.
    check("a persistent source with the official markers is accepted",
          validate_upstream_source_content(
              "persistent",
              "class PersistentDenseGemmKernel:\n"
              "utils.StaticPersistentTileScheduler.create()\n"
              "tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE\n") == [])
    rejects("a persistent source without the static scheduler is rejected",
            validate_upstream_source_content(
                "persistent", "class PersistentDenseGemmKernel:\nCtaGroup.TWO\n"),
            "StaticPersistentTileScheduler")
    rejects("a persistent source without the CtaGroup.TWO path is rejected",
            validate_upstream_source_content(
                "persistent",
                "class PersistentDenseGemmKernel:\nStaticPersistentTileScheduler\n"),
            "CtaGroup.TWO")
    check("a non-persistent source with its markers is accepted",
          validate_upstream_source_content(
              "nonpersistent", "class DenseGemmKernel:\ndef create_tensors():\n") == [])
    rejects("a non-persistent source containing the persistent scheduler is rejected",
            validate_upstream_source_content(
                "nonpersistent",
                "class DenseGemmKernel:\ndef create_tensors():\n"
                "StaticPersistentTileScheduler\n"),
            "not be distinguishable")

    # Make recipes.
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
            validate_smoke_recipe(
                [l for l in _GOOD_SMOKE_RECIPE if "b" * 64 not in l]),
            "BOTH official sources")
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
    rejects("a smoke recipe driving the P3.3 wrapper is rejected",
            validate_smoke_recipe(
                [l.replace("cutedsl_variants.py", "cublaslt_gemm.py")
                 for l in _GOOD_SMOKE_RECIPE]),
            "P3.3")

    # Status documents.
    good_plan = (
        "| P3.1 | Pinned official CuTe DSL example | YES | YES | YES |\n"
        "| P3.2 | One-shape wrapper | YES | YES | YES |\n"
        "| P3.3 | cuBLASLt baseline | YES | YES | YES |\n"
        "| P3.4 | Three execution variants | YES | YES | YES |\n"
        "| P3.5 | Five shapes and comparison | YES | NO | NO |\n"
    )
    good_protocol = (
        "Status: `P3.4 = YES / YES / YES`.\n"
        "P3.4 creates no publishable performance result.\n"
    )
    good_readme = "P3.4 (three execution variants) is closed as YES / YES / YES.\n"
    check("closed status documents are accepted",
          validate_status_documents(good_plan, good_protocol, good_readme) == [],
          str(validate_status_documents(good_plan, good_protocol, good_readme)))
    rejects("a stale pre-audit PLAN.md status is rejected",
            validate_status_documents(
                good_plan.replace(
                    "| P3.4 | Three execution variants | YES | YES | YES |",
                    "| P3.4 | Three execution variants | YES | NO | NO |"),
                good_protocol, good_readme), "stale or invalid")
    rejects("a PLAN.md that never records P3.4 is rejected",
            validate_status_documents("", good_protocol, good_readme), "YES / YES / YES")
    rejects("a PLAN.md that weakens a closed unit is rejected",
            validate_status_documents(
                good_plan.replace("| P3.2 | One-shape wrapper | YES | YES | YES |",
                                  "| P3.2 | One-shape wrapper | YES | NO | NO |"),
                good_protocol, good_readme), "no longer records")
    # P3.5 may progress on its own: unimplemented, implemented-but-unclosed, and
    # closed are all truthful states this closed P3.4 gate must accept.
    for later_status in (
        "| P3.5 | Five shapes and comparison | NO | NO | NO |",
        "| P3.5 | Five shapes and comparison | YES | NO | NO |",
        "| P3.5 | Five shapes and comparison | YES | YES | YES |",
    ):
        check("a later P3.5 status is accepted",
              validate_status_documents(
                  good_plan.replace("| P3.5 | Five shapes and comparison | YES | NO | NO |",
                                    later_status),
                  good_protocol, good_readme) == [],
              later_status)
    rejects("a PLAN.md that drops the P3.5 row is rejected",
            validate_status_documents(
                good_plan.replace(
                    "| P3.5 | Five shapes and comparison | YES | NO | NO |\n", ""),
                good_protocol, good_readme), "status row")
    rejects("a P3.5 audited without being implemented is rejected",
            validate_status_documents(
                good_plan.replace("| P3.5 | Five shapes and comparison | YES | NO | NO |",
                                  "| P3.5 | Five shapes and comparison | NO | YES | NO |"),
                good_protocol, good_readme), "without being")

    # Make variable expansion and recipe extraction.
    variables = parse_make_variables("A := src/gemm\nB := $(A)/cutedsl_variants.py\n")
    check("simply expanded Make variables are parsed",
          variables.get("B") == "src/gemm/cutedsl_variants.py", str(variables))
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
        print("usage: check_cutedsl_variants_p34.py [repository-root] | --self-test",
              file=sys.stderr)
        return 2

    root = Path(arguments[0]) if arguments else Path(__file__).resolve().parents[1]
    if not root.is_dir():
        print(f"check_cutedsl_variants_p34: {root} is not a directory", file=sys.stderr)
        return 2

    print(f"check_cutedsl_variants_p34: checking {root}", file=sys.stderr)
    try:
        errors = check_wrapper(root)
    except Exception as exc:  # noqa: BLE001 - a checker crash is a failed check
        print(f"check_cutedsl_variants_p34: the check itself failed: {exc}", file=sys.stderr)
        return 1

    if errors:
        print(f"check_cutedsl_variants_p34: FAIL ({len(errors)} finding(s))", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("check_cutedsl_variants_p34: OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
