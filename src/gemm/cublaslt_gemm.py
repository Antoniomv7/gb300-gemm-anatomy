#!/usr/bin/env python3
"""P3.3 - equivalent cuBLASLt BF16 GEMM baseline (frozen; no performance claim).

This is the vendor-library counterpart of the P3.2 one-shape CuTe DSL wrapper.
It executes exactly the same GEMM geometry, on exactly the same operands, with
a direct, explicit ``cublasLtMatmul`` call issued through a small
repository-owned C-ABI bridge (``src/gemm/cublaslt_bridge.cu``). No NVIDIA GEMM
implementation is copied, forked, patched, or vendored: only the public
cuBLASLt API of the pinned CUDA 13.1 toolkit is called.

Operand equivalence with P3.2 is the point of this unit, so it is established
structurally rather than by assertion. The pinned upstream example's own
``create_tensors`` is parsed from the verified ``/opt/cutlass`` checkout and its
tensor-construction sequence is compared against the sequence executed here:
the same ``cutlass.torch.matrix`` factory, the same seed, the same call order
(A, then B, then C), and the same dtypes and physical layouts. A different C++
RNG seeded with the same number would not be equivalent and is never used.

What this program measures, in this exact order:

1. environment and provenance (exactly one CUDA GPU, as logical device 0);
2. repository and pinned upstream provenance;
3. deterministic tensor allocation, entirely outside every timer;
4. cuBLASLt plan creation: descriptors, layouts, preference, vendor heuristic,
   algorithm validation, and workspace allocation;
5. ``setup_time_ms``     - a monotonic host clock around step 4 only;
6. ``first_launch_ms``   - the same host clock around one ``cublasLtMatmul``;
7. complete FP32 correctness validation against an untimed PyTorch CUDA oracle
   with TF32 (and every other reduced-precision FP32 matmul mode) disabled;
8. only if correctness passes: warm-up launches, then ``kernel_time_ms`` from
   CUDA events on the same stream, divided by the measured iteration count.

``setup_time_ms`` is deliberately **not** called ``compile_time_ms``: nothing is
compiled at run time here. cuBLASLt setup and CuTe DSL JIT compilation are
different costs and P3.3 never pretends otherwise, which is also why the P3.2
field name is not reused.

What this program is not: it is not an experimental campaign, not a
CuTe-versus-cuBLASLt comparison, and not a performance result. Every emitted row
carries ``publishable=false``. No TFLOP/s, speedup, efficiency, utilization,
bandwidth, or winner label is computed anywhere, and the three timings are P3.3
functional-verification evidence only. The comparison itself belongs to P3.5,
which does not exist.

Output contract:

* stdout receives exactly one CSV header line and exactly one CSV data row,
  and nothing else. To make that true even when a native library writes to
  file descriptor 1, descriptor 1 is redirected to descriptor 2 for the whole
  measurement and the real stdout is restored only to emit the two CSV lines,
  after correctness has already passed.
* stderr receives every human-readable message: progress, warnings, library
  output, and diagnostics.
* A correctness failure exits non-zero, prints a diagnostic to stderr, and
  emits no CSV header and no CSV data row; no warm-up and no steady-state
  timing runs in that case.

Usage:
  cublaslt_gemm.py [--warmup-iterations N] [--iterations N]
  cublaslt_gemm.py --self-test
  cublaslt_gemm.py --help

``--help`` and ``--self-test`` are GPU-free: they import neither PyTorch, nor
CuTe DSL, nor the CUDA bindings, and they never load the shared bridge. Every
heavy import and the library load are deferred into the measurement path.

Exit code: 0 only when the whole sequence succeeded and one valid row was
emitted, 1 on any contract, provenance, correctness, or execution failure, and
2 on a usage error.
"""

import argparse
import ast
import contextlib
import csv
import hashlib
import io
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# --- Frozen identity of this unit -------------------------------------------

SCHEMA_VERSION = "p33.v1"
EXPERIMENT = "exp03_cutedsl_vs_cublaslt"
UNIT = "P3.3"
RUN_KIND = "smoke"
METHOD = "cublaslt"
VARIANT = "heuristic_first_supported"
REFERENCE = "torch_cuda_fp32_ieee"
CACHE_MODE = "hot"
CORRECTNESS_PASS = "PASS"
PUBLISHABLE = "false"

# --- Frozen GEMM configuration ----------------------------------------------
#
# P3.3 executes exactly one configuration, identical in geometry, dtypes, and
# physical layout to P3.2. None of these values is reachable from the command
# line, from an environment variable, or from a configuration file: they are
# immutable constants of this unit, and the bridge freezes the same values
# independently in C so that the two must agree.

FROZEN_M = 4096
FROZEN_N = 4096
FROZEN_K = 4096
FROZEN_L = 1
FROZEN_MNKL = (FROZEN_M, FROZEN_N, FROZEN_K, FROZEN_L)

FROZEN_AB_DTYPE = "BFloat16"
FROZEN_ACC_DTYPE = "Float32"
FROZEN_C_DTYPE = "Float32"

# The same major-mode vocabulary P3.2 uses, because these select exactly the
# same physical layouts from the same upstream tensor factory.
FROZEN_A_MAJOR = "k"
FROZEN_B_MAJOR = "k"
FROZEN_C_MAJOR = "n"

# The explicit cuBLASLt descriptor contract. A is M x K row-major, B is N x K
# row-major, C and D are M x N row-major, and the operation is C = A x B^T.
# These leading dimensions are exactly the strides the frozen physical layouts
# already have, so no data is transposed, copied, or reinterpreted.
FROZEN_LDA = FROZEN_K
FROZEN_LDB = FROZEN_K
FROZEN_LDC = FROZEN_N
FROZEN_LDD = FROZEN_N

FROZEN_TRANSA = "CUBLAS_OP_N"
FROZEN_TRANSB = "CUBLAS_OP_T"
FROZEN_ORDER = "CUBLASLT_ORDER_ROW"
FROZEN_AB_CUDA_TYPE = "CUDA_R_16BF"
FROZEN_CD_CUDA_TYPE = "CUDA_R_32F"
FROZEN_COMPUTE_TYPE = "CUBLAS_COMPUTE_32F"
FROZEN_SCALE_TYPE = "CUDA_R_32F"
FROZEN_POINTER_MODE = "CUBLASLT_POINTER_MODE_HOST"
FROZEN_EPILOGUE = "CUBLASLT_EPILOGUE_DEFAULT"
FROZEN_SEARCH_MODE = "CUBLASLT_SEARCH_BEST_FIT"

FROZEN_ALPHA = 1.0
FROZEN_BETA = 0.0

# The fixed, non-autotuned algorithm policy. Exactly 64 MiB of workspace is
# offered, exactly 32 heuristic results are requested, and the first supported
# one is executed. No candidate is ever benchmarked.
FROZEN_WORKSPACE_LIMIT_BYTES = 67108864
FROZEN_HEURISTIC_REQUESTED = 32

FROZEN_SEED = 1111
FROZEN_ATOL = 1e-1
FROZEN_RTOL = 1e-5

# The only CUDA matmul FP32 policy P3.3 accepts for its correctness oracle,
# via the PyTorch 2.10 fp32_precision API and nothing else. The unset default
# is "none", which proves nothing and is rejected. This is the P3.2 policy,
# retained unchanged so both units are validated identically.
FP32_PRECISION_IEEE = "ieee"

# Safe denominator for the reported relative error, identical to P3.2. The
# reported max_rel_error is max(|d - ref| / max(|ref|, floor)), which stays
# finite and well defined where the reference is exactly zero. It is a
# diagnostic only: the pass/fail decision uses the elementwise criterion
# |d - ref| <= atol + rtol * |ref| at full precision, never this scalar.
REL_ERROR_DENOMINATOR_FLOOR = 1.0

# The numeric values of every frozen cuBLASLt enum, as declared in the pinned
# CUDA 13.1 headers. The bridge reports the values it actually compiled
# against; these are the wrapper's independent expectation, and a disagreement
# fails closed instead of being silently serialized.
CUBLAS_ENUM_VALUES = {
    "CUBLAS_OP_N": 0,
    "CUBLAS_OP_T": 1,
    "CUBLASLT_ORDER_ROW": 1,
    "CUDA_R_16BF": 14,
    "CUDA_R_32F": 0,
    "CUBLAS_COMPUTE_32F": 68,
    "CUBLASLT_POINTER_MODE_HOST": 0,
    "CUBLASLT_EPILOGUE_DEFAULT": 1,
    "CUBLASLT_SEARCH_BEST_FIT": 0,
}

FROZEN_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "experiment": EXPERIMENT,
    "unit": UNIT,
    "run_kind": RUN_KIND,
    "method": METHOD,
    "variant": VARIANT,
    "m": FROZEN_M,
    "n": FROZEN_N,
    "k": FROZEN_K,
    "l": FROZEN_L,
    "ab_dtype": FROZEN_AB_DTYPE,
    "acc_dtype": FROZEN_ACC_DTYPE,
    "c_dtype": FROZEN_C_DTYPE,
    "a_major": FROZEN_A_MAJOR,
    "b_major": FROZEN_B_MAJOR,
    "c_major": FROZEN_C_MAJOR,
    "order_a": FROZEN_ORDER,
    "order_b": FROZEN_ORDER,
    "order_c": FROZEN_ORDER,
    "order_d": FROZEN_ORDER,
    "transa": FROZEN_TRANSA,
    "transb": FROZEN_TRANSB,
    "lda": FROZEN_LDA,
    "ldb": FROZEN_LDB,
    "ldc": FROZEN_LDC,
    "ldd": FROZEN_LDD,
    "compute_type": FROZEN_COMPUTE_TYPE,
    "scale_type": FROZEN_SCALE_TYPE,
    "pointer_mode": FROZEN_POINTER_MODE,
    "epilogue": FROZEN_EPILOGUE,
    "alpha": FROZEN_ALPHA,
    "beta": FROZEN_BETA,
    "seed": FROZEN_SEED,
    "reference": REFERENCE,
    "atol": FROZEN_ATOL,
    "rtol": FROZEN_RTOL,
    "cache_mode": CACHE_MODE,
    "workspace_limit_bytes": FROZEN_WORKSPACE_LIMIT_BYTES,
    "heuristic_requested": FROZEN_HEURISTIC_REQUESTED,
    "publishable": False,
}

# --- Runtime controls (the only ones that exist) -----------------------------

DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_ITERATIONS = 20
MIN_ITERATIONS = 1
MAX_WARMUP_ITERATIONS = 100
MAX_ITERATIONS = 100

# --- Frozen CSV schema -------------------------------------------------------

CSV_FIELDS = (
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

# Deterministic decimal formats. Every real-valued field is serialized as a
# plain fixed-point decimal with exactly this many fractional digits: no
# exponent, no locale dependence, no shortest-round-trip ambiguity. Values
# below half of the last retained digit therefore serialize as zero, which is
# intentional and harmless because these fields are diagnostics and every
# decision (correctness, positivity, finiteness) is taken on the full-precision
# value before serialization.
DECIMALS_TIMING = 6  # milliseconds, i.e. nanosecond resolution
DECIMALS_ERROR = 9
DECIMALS_TOLERANCE = 9
DECIMALS_SCALAR = 9  # alpha and beta
DECIMALS_WAVES = 6

CSV_FIXED_VALUES = {
    "schema_version": SCHEMA_VERSION,
    "experiment": EXPERIMENT,
    "unit": UNIT,
    "run_kind": RUN_KIND,
    "method": METHOD,
    "variant": VARIANT,
    "m": str(FROZEN_M),
    "n": str(FROZEN_N),
    "k": str(FROZEN_K),
    "l": str(FROZEN_L),
    "ab_dtype": FROZEN_AB_DTYPE,
    "acc_dtype": FROZEN_ACC_DTYPE,
    "c_dtype": FROZEN_C_DTYPE,
    "a_major": FROZEN_A_MAJOR,
    "b_major": FROZEN_B_MAJOR,
    "c_major": FROZEN_C_MAJOR,
    "order_a": FROZEN_ORDER,
    "order_b": FROZEN_ORDER,
    "order_c": FROZEN_ORDER,
    "order_d": FROZEN_ORDER,
    "transa": FROZEN_TRANSA,
    "transb": FROZEN_TRANSB,
    "lda": str(FROZEN_LDA),
    "ldb": str(FROZEN_LDB),
    "ldc": str(FROZEN_LDC),
    "ldd": str(FROZEN_LDD),
    "compute_type": FROZEN_COMPUTE_TYPE,
    "scale_type": FROZEN_SCALE_TYPE,
    "pointer_mode": FROZEN_POINTER_MODE,
    "epilogue": FROZEN_EPILOGUE,
    "seed": str(FROZEN_SEED),
    "reference": REFERENCE,
    "correctness": CORRECTNESS_PASS,
    "cache_mode": CACHE_MODE,
    "workspace_limit_bytes": str(FROZEN_WORKSPACE_LIMIT_BYTES),
    "heuristic_requested": str(FROZEN_HEURISTIC_REQUESTED),
    "publishable": PUBLISHABLE,
}

CSV_TIMING_FIELDS = ("setup_time_ms", "first_launch_ms", "kernel_time_ms")
CSV_ERROR_FIELDS = ("max_abs_error", "max_rel_error")
CSV_TOLERANCE_FIELDS = ("atol", "rtol")
CSV_COUNT_FIELDS = ("warmup_iterations", "iterations")
CSV_BOOL_FIELDS = ("git_dirty", "publishable")

# Non-negative integer metadata serialized as canonical decimal strings.
CSV_NONNEGATIVE_INT_FIELDS = (
    "workspace_bytes",
    "heuristic_index",
    "algo_id",
    "tile_id",
    "stages_id",
    "reduction_scheme",
    "cta_swizzling",
    "custom_option",
    "inner_shape_id",
    "cluster_shape_id",
    "cublaslt_version",
)
# Strictly positive integer metadata.
CSV_POSITIVE_INT_FIELDS = (
    "alignment_a_bytes",
    "alignment_b_bytes",
    "alignment_c_bytes",
    "alignment_d_bytes",
    "heuristic_returned",
    "split_k",
)

BOOL_TRUE = "true"
BOOL_FALSE = "false"

_RE_HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")
_RE_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_RE_GPU_UUID = re.compile(r"\AGPU-[0-9a-fA-F][0-9a-fA-F-]+\Z")
_RE_DOTTED_VERSION = re.compile(r"\A[0-9]+(\.[0-9]+)*\Z")
_RE_COMPUTE_CAPABILITY = re.compile(r"\A[0-9]+\.[0-9]+\Z")
_RE_POSITIVE_INT = re.compile(r"\A[1-9][0-9]*\Z")
_RE_NONNEGATIVE_INT = re.compile(r"\A(0|[1-9][0-9]*)\Z")
_RE_ENV_LINE = re.compile(r"\A([A-Z][A-Z0-9_]*)=(\S*)\Z")
_RE_CUDA_ARCH = re.compile(r"\Asm_([0-9]+)([a-z]?)\Z")
_RE_SAFE_TEXT = re.compile(r"\A[^\x00-\x1f\x7f]+\Z")

# The pinned CUTLASS checkout inside the pinned image. P3.3 never executes the
# upstream kernel, but it reads that checkout to prove its own operand
# construction still matches the upstream factory, and it reports the same
# provenance fields P3.2 reports. Nothing is ever written there.
UPSTREAM_CHECKOUT_DIR = Path("/opt/cutlass")

# The compiled bridge, built into the container's own private /tmp by the Make
# targets immediately before the wrapper runs. This location is a constant, not
# a runtime control: there is no CLI option and no environment variable for it.
BRIDGE_LIBRARY_PATH = Path("/tmp/p33-bridge/libp33_cublaslt_bridge.so")
BRIDGE_SOURCE_RELATIVE = "src/gemm/cublaslt_bridge.cu"
BRIDGE_ABI_VERSION = 1

GLOBAL_CONTRACT_FILE = "VERSIONS.env"
PHASE3_CONTRACT_FILE = "PHASE3_VERSIONS.env"

# Keys read from the two version contracts. Nothing below is duplicated as a
# literal anywhere in this file: the pinned commit, blob, SHA-256, versions,
# and architecture exist here only as key names. P3.3 adds no key to either
# contract: cuBLASLt ships inside the already pinned CUDA 13.1 image, so there
# is no new package and no new version to pin, and the library's own runtime
# version is read from the library itself.
GLOBAL_CONTRACT_KEYS = ("CUDA_VERSION", "CUTLASS_VERSION", "CUTLASS_COMMIT", "CUDA_ARCH")
PHASE3_CONTRACT_KEYS = (
    "PYTORCH_VERSION",
    "PYTORCH_CUDA_VERSION",
    "CUTEDSL_P31_EXAMPLE_PATH",
    "CUTEDSL_P31_EXAMPLE_GIT_BLOB",
    "CUTEDSL_P31_EXAMPLE_SHA256",
)

# The upstream tensor factory P3.3 replicates, and the exact call sequence that
# replication must match. Each entry is (positional argument names, dtype
# argument name) for one cutlass.torch.matrix call, in upstream call order.
UPSTREAM_TENSOR_FACTORY = "create_tensors"
UPSTREAM_MATRIX_FACTORY = "matrix"
UPSTREAM_MATRIX_CALLS = (
    ("l", "m", "k", "ab_dtype"),
    ("l", "n", "k", "ab_dtype"),
    ("l", "m", "n", "c_dtype"),
)


class P33Error(Exception):
    """Any fail-closed P3.3 contract, provenance, or execution failure."""


class RowContractError(P33Error):
    """A CSV row violated the frozen P3.3 schema."""


class CorrectnessError(P33Error):
    """The complete result did not match the untimed FP32 reference."""


class BridgeError(P33Error):
    """The cuBLASLt bridge refused, failed, or disagreed with this wrapper."""


def log(message: str) -> None:
    """Write one human-readable progress/diagnostic line to stderr."""
    print(f"cublaslt_gemm: {message}", file=sys.stderr, flush=True)


# --- Version contracts -------------------------------------------------------


def repository_root() -> Path:
    """Locate the repository root that owns this file.

    ``src/gemm/cublaslt_gemm.py`` is two directories below the root both on the
    host and inside the container, where the repository is mounted at
    ``/workspace``.
    """
    root = Path(__file__).resolve().parents[2]
    for name in (GLOBAL_CONTRACT_FILE, PHASE3_CONTRACT_FILE):
        if not (root / name).is_file():
            raise P33Error(f"repository root {root} does not contain {name}")
    return root


def parse_env_file(path: Path) -> dict:
    """Parse a ``KEY=VALUE`` version contract strictly and fail closed.

    Blank lines and ``#`` comments are ignored. Anything else must match
    ``KEY=VALUE`` with an uppercase key and a whitespace-free value, and no key
    may appear twice.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise P33Error(f"cannot read version contract {path}: {exc}") from exc

    values: dict = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _RE_ENV_LINE.match(line)
        if match is None:
            raise P33Error(f"{path}:{lineno}: malformed contract line {raw!r}")
        key, value = match.group(1), match.group(2)
        if key in values:
            raise P33Error(f"{path}:{lineno}: duplicate contract key {key}")
        values[key] = value
    return values


def load_pinned_contract(repo_root=None) -> dict:
    """Read every pinned value P3.3 needs from the two version contracts.

    ``VERSIONS.env`` is the closed global contract that the audited P1/P2
    aggregators parse against their own closed key allowlist; P3.3 only reads
    it. ``PHASE3_VERSIONS.env`` is the Phase 3-only extension that P3.1
    created. P3.3 adds no key to either file: it introduces no package, and the
    cuBLASLt runtime version it reports comes from ``cublasLtGetVersion()``
    rather than from any pinned constant.
    """
    root = Path(repo_root) if repo_root is not None else repository_root()
    global_values = parse_env_file(root / GLOBAL_CONTRACT_FILE)
    phase3_values = parse_env_file(root / PHASE3_CONTRACT_FILE)

    contract = {}
    for key in GLOBAL_CONTRACT_KEYS:
        if key not in global_values:
            raise P33Error(f"{GLOBAL_CONTRACT_FILE} is missing required key {key}")
        contract[key] = global_values[key]
    for key in PHASE3_CONTRACT_KEYS:
        if key not in phase3_values:
            raise P33Error(f"{PHASE3_CONTRACT_FILE} is missing required key {key}")
        contract[key] = phase3_values[key]

    if not _RE_HEX40.match(contract["CUTLASS_COMMIT"]):
        raise P33Error(f"pinned CUTLASS_COMMIT is malformed: {contract['CUTLASS_COMMIT']!r}")
    if not _RE_HEX40.match(contract["CUTEDSL_P31_EXAMPLE_GIT_BLOB"]):
        raise P33Error("pinned CUTEDSL_P31_EXAMPLE_GIT_BLOB is malformed")
    if not _RE_HEX64.match(contract["CUTEDSL_P31_EXAMPLE_SHA256"]):
        raise P33Error("pinned CUTEDSL_P31_EXAMPLE_SHA256 is malformed")
    for key in ("CUDA_VERSION", "PYTORCH_CUDA_VERSION"):
        if not _RE_DOTTED_VERSION.match(contract[key]):
            raise P33Error(f"pinned {key} is malformed: {contract[key]!r}")
    if not contract["CUTLASS_VERSION"].startswith("v"):
        raise P33Error(f"pinned CUTLASS_VERSION is malformed: {contract['CUTLASS_VERSION']!r}")

    # Derived, never separately pinned.
    contract["CUTEDSL_VERSION"] = contract["CUTLASS_VERSION"][1:]
    if not _RE_DOTTED_VERSION.match(contract["CUTEDSL_VERSION"]):
        raise P33Error("pinned CuTe DSL version is malformed")

    example_path = contract["CUTEDSL_P31_EXAMPLE_PATH"]
    pure = Path(example_path)
    if pure.is_absolute() or ".." in pure.parts or not example_path.endswith(".py"):
        raise P33Error(f"pinned upstream example path is unsafe: {example_path!r}")

    contract["CUDA_MAJOR_MINOR"] = ".".join(contract["CUDA_VERSION"].split(".")[:2])
    contract["EXPECTED_COMPUTE_CAPABILITY"] = compute_capability_for_arch(contract["CUDA_ARCH"])
    return contract


def compute_capability_for_arch(cuda_arch: str) -> str:
    """Map a pinned ``sm_<digits>[a]`` target to its ``major.minor`` capability.

    NVIDIA's convention is that the final digit is the minor version and every
    preceding digit is the major one: ``sm_75`` is 7.5, ``sm_90`` is 9.0, and
    ``sm_100`` is 10.0. A trailing letter marks the architecture-specific form
    of the same capability. Deriving the capability from whatever the contract
    pins keeps the architecture pin in ``VERSIONS.env`` - it is deliberately
    not restated here - and still lets the wrapper reject a device that is not
    the pinned target.
    """
    match = _RE_CUDA_ARCH.match(cuda_arch)
    if match is None:
        raise P33Error(f"pinned CUDA_ARCH is malformed: {cuda_arch!r}")
    digits = match.group(1)
    if len(digits) < 2:
        raise P33Error(f"pinned CUDA_ARCH is malformed: {cuda_arch!r}")
    return f"{int(digits[:-1])}.{int(digits[-1])}"


# --- Upstream source identity ------------------------------------------------


def _git(args, cwd=None, safe_directory=None) -> str:
    """Run one read-only Git query and return its stripped stdout."""
    command = ["git"]
    if safe_directory is not None:
        # /opt/cutlass is a root-owned checkout inside the image while the
        # container runs as the invoking user, so each query carries its own
        # per-invocation safe.directory. Nothing is ever written there.
        command += ["-c", f"safe.directory={safe_directory}"]
    command += list(args)
    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise P33Error(f"git {' '.join(args)} could not be executed: {exc}") from exc
    if completed.returncode != 0:
        raise P33Error(
            f"git {' '.join(args)} failed with exit code {completed.returncode}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout.strip()


def sha256_of_file(path: Path) -> str:
    """Return the lowercase hexadecimal SHA-256 of a file, read in chunks."""
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise P33Error(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def verify_upstream_source(contract: dict) -> dict:
    """Prove the pinned upstream checkout and example file are byte-identical.

    Fails closed on a missing checkout, a wrong HEAD, any tracked or untracked
    modification, a symlinked or non-regular example file, a wrong Git blob
    SHA, or a wrong SHA-256. P3.3 does not execute that file, but it does
    replicate its tensor-construction sequence and report its identity, so the
    identity has to be proved. The checkout is only ever queried, never
    written.
    """
    checkout = UPSTREAM_CHECKOUT_DIR
    if not checkout.is_dir():
        raise P33Error(f"pinned CUTLASS checkout {checkout} is missing")

    head = _git(["-C", str(checkout), "rev-parse", "HEAD"], safe_directory=str(checkout))
    if head != contract["CUTLASS_COMMIT"]:
        raise P33Error(
            f"{checkout} HEAD {head} != pinned CUTLASS_COMMIT {contract['CUTLASS_COMMIT']}"
        )

    dirty = _git(
        ["-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        safe_directory=str(checkout),
    )
    if dirty:
        raise P33Error(f"{checkout} has tracked or untracked modifications")

    example = checkout / contract["CUTEDSL_P31_EXAMPLE_PATH"]
    if example.is_symlink():
        raise P33Error(f"{example} is a symlink")
    if not example.is_file():
        raise P33Error(f"{example} is not a regular file")

    blob = _git(
        ["-C", str(checkout), "hash-object", "--", str(example)],
        safe_directory=str(checkout),
    )
    if blob != contract["CUTEDSL_P31_EXAMPLE_GIT_BLOB"]:
        raise P33Error(
            f"{example} Git blob {blob} != pinned {contract['CUTEDSL_P31_EXAMPLE_GIT_BLOB']}"
        )

    sha256 = sha256_of_file(example)
    if sha256 != contract["CUTEDSL_P31_EXAMPLE_SHA256"]:
        raise P33Error(
            f"{example} SHA-256 {sha256} != pinned {contract['CUTEDSL_P31_EXAMPLE_SHA256']}"
        )

    return {"commit": head, "blob": blob, "sha256": sha256, "path": example}


def extract_upstream_matrix_calls(source: str) -> tuple:
    """Return the upstream tensor factory's ``matrix`` calls, in call order.

    Returns ``(calls, seeds)``, where ``calls`` is the ordered list of
    ``{"args": ..., "lineno": ...}`` records and ``seeds`` lists every literal
    passed to a ``manual_seed`` call inside the factory.

    The upstream file is parsed, never imported: P3.3 needs no upstream object,
    and parsing avoids executing roughly 1,800 lines of unrelated module-level
    code just to inspect one function. Each returned entry is the tuple of
    positional argument names of one ``cutlass.torch.matrix`` call, with any
    non-name argument rendered as its literal.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise P33Error(f"cannot parse the pinned upstream example: {exc}") from exc

    factory = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == UPSTREAM_TENSOR_FACTORY:
            factory = node
            break
    if factory is None:
        raise P33Error(
            f"the pinned upstream example no longer defines {UPSTREAM_TENSOR_FACTORY}()"
        )

    seeds = []
    calls = []
    for node in ast.walk(factory):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.attr if isinstance(node.func, ast.Attribute) else (
            node.func.id if isinstance(node.func, ast.Name) else None
        )
        if name == "manual_seed" and len(node.args) == 1:
            if isinstance(node.args[0], ast.Constant):
                seeds.append(node.args[0].value)
        elif name == UPSTREAM_MATRIX_FACTORY:
            rendered = []
            for argument in node.args:
                if isinstance(argument, ast.Name):
                    rendered.append(argument.id)
                elif isinstance(argument, ast.Constant):
                    rendered.append(repr(argument.value))
                elif isinstance(argument, ast.Compare):
                    # e.g. `a_major == "m"`; recorded by its left operand name
                    # so the checker can see which major mode drove the call.
                    left = argument.left
                    rendered.append(left.id if isinstance(left, ast.Name) else "<expr>")
                else:
                    rendered.append("<expr>")
            calls.append((tuple(rendered), node.lineno))

    calls.sort(key=lambda item: item[1])
    return [{"args": args, "lineno": lineno} for args, lineno in calls], seeds


def verify_upstream_tensor_factory(example: Path) -> dict:
    """Prove P3.3's operand construction still matches the upstream factory.

    This is what makes "exact P3.2 operand equivalence" checkable rather than
    merely asserted: P3.2 calls the upstream factory, P3.3 replicates it (it
    has to, because the upstream factory discards the device tensors for A and
    B, and P3.3 must retain the allocations to hand their device pointers to
    cuBLASLt), and this function fails closed if the two ever diverge.
    """
    try:
        source = example.read_text(encoding="utf-8")
    except OSError as exc:
        raise P33Error(f"cannot read the pinned upstream example: {exc}") from exc

    calls, seeds = extract_upstream_matrix_calls(source)

    if FROZEN_SEED not in seeds:
        raise P33Error(
            f"the pinned upstream tensor factory does not seed with {FROZEN_SEED}; "
            "P3.3 cannot claim operand equivalence with P3.2"
        )
    if len(calls) != len(UPSTREAM_MATRIX_CALLS):
        raise P33Error(
            f"the pinned upstream tensor factory makes {len(calls)} "
            f"{UPSTREAM_MATRIX_FACTORY}() call(s), P3.3 replicates "
            f"{len(UPSTREAM_MATRIX_CALLS)}"
        )
    for index, (call, expected) in enumerate(zip(calls, UPSTREAM_MATRIX_CALLS)):
        args = call["args"]
        if len(args) < 5:
            raise P33Error(
                f"upstream {UPSTREAM_MATRIX_FACTORY}() call {index} has too few arguments: {args}"
            )
        # (l, mode0, mode1, is_mode0_major, cutlass_dtype)
        actual = (args[0], args[1], args[2], args[4])
        if actual != expected:
            raise P33Error(
                f"upstream {UPSTREAM_MATRIX_FACTORY}() call {index} is {actual}, P3.3 "
                f"replicates {expected}; the operand construction has diverged"
            )
    return {"matrix_calls": [call["args"] for call in calls], "seeds": seeds}


# --- Environment and provenance ---------------------------------------------


def _query_nvidia_smi() -> dict:
    """Collect the allowlisted device fields for exactly one visible GPU."""
    command = [
        "nvidia-smi",
        "--query-gpu=uuid,name,driver_version",
        "--format=csv,noheader",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise P33Error(f"nvidia-smi could not be executed: {exc}") from exc
    if completed.returncode != 0:
        raise P33Error(
            f"nvidia-smi failed with exit code {completed.returncode}; "
            "device provenance is ambiguous"
        )

    rows = [row for row in csv.reader(io.StringIO(completed.stdout)) if row]
    if len(rows) != 1:
        raise P33Error(f"nvidia-smi reported {len(rows)} GPUs; exactly 1 must be visible")
    fields = [value.strip() for value in rows[0]]
    if len(fields) != 3:
        raise P33Error("nvidia-smi returned a malformed device row")

    uuid, name, driver_version = fields
    if not _RE_GPU_UUID.match(uuid):
        raise P33Error(f"nvidia-smi returned a malformed GPU UUID: {uuid!r}")
    if not name or not _RE_SAFE_TEXT.match(name):
        raise P33Error("nvidia-smi returned a malformed GPU name")
    if not _RE_DOTTED_VERSION.match(driver_version):
        raise P33Error(f"nvidia-smi returned a malformed driver version: {driver_version!r}")
    return {"gpu_uuid": uuid, "gpu_name": name, "driver_version": driver_version}


def _query_nvcc_major_minor() -> str:
    """Read the installed CUDA toolkit's ``release X.Y`` from nvcc."""
    try:
        completed = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise P33Error(f"nvcc could not be executed: {exc}") from exc
    if completed.returncode != 0:
        raise P33Error("nvcc --version failed; the CUDA toolkit version is ambiguous")
    match = re.search(r"release ([0-9]+)\.([0-9]+)", completed.stdout)
    if match is None:
        raise P33Error("nvcc --version did not report a release version")
    return f"{match.group(1)}.{match.group(2)}"


def _repository_git_state(root: Path) -> dict:
    """Record this repository's commit and dirty state."""
    commit = _git(["rev-parse", "HEAD"], cwd=root)
    if not _RE_HEX40.match(commit):
        raise P33Error(f"repository HEAD is malformed: {commit!r}")
    status = _git(["status", "--porcelain", "--untracked-files=all"], cwd=root)
    return {"git_commit": commit, "git_dirty": BOOL_TRUE if status else BOOL_FALSE}


def require_single_cuda_device(torch) -> None:
    """Require exactly one CUDA-visible GPU, used as logical device 0."""
    if not torch.cuda.is_available():
        raise P33Error("no CUDA device is available; P3.3 requires exactly one GPU")
    count = torch.cuda.device_count()
    if count != 1:
        raise P33Error(f"expected exactly 1 CUDA-visible GPU, saw {count}")
    torch.cuda.set_device(0)
    current = torch.cuda.current_device()
    if current != 0:
        raise P33Error(f"the selected CUDA device must be logical device 0, got {current}")


def require_ieee_fp32_matmul_api(torch):
    """Return the CUDA matmul backend, failing closed without the 2.10 API.

    P3.3 retains the P3.2 policy unchanged and uses **exclusively** the PyTorch
    2.10 ``fp32_precision`` API for CUDA matrix multiplication. The legacy
    ``allow_tf32`` property is never read and never written: in 2.10 the two are
    aliases of one setting, mixing them is unsupported, and the last write
    silently wins - setting ``allow_tf32`` after ``fp32_precision`` rewrites the
    policy to ``tf32`` without any error. ``torch.set_float32_matmul_precision()``
    is likewise never combined with it.
    """
    backends = getattr(torch, "backends", None)
    cuda_backend = getattr(backends, "cuda", None) if backends is not None else None
    matmul = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
    if matmul is None:
        raise P33Error(
            "this PyTorch does not expose torch.backends.cuda.matmul; the IEEE FP32 "
            "reference cannot be guaranteed"
        )
    if not hasattr(matmul, "fp32_precision"):
        raise P33Error(
            "this PyTorch does not support torch.backends.cuda.matmul.fp32_precision; "
            "P3.3 requires that API and never falls back to the legacy TF32 flag"
        )
    return matmul


@contextlib.contextmanager
def ieee_fp32_matmul(torch):
    """Guarantee IEEE FP32 CUDA matmul for the untimed correctness oracle.

    Sets ``torch.backends.cuda.matmul.fp32_precision`` to ``ieee``, reads the
    property back, and requires it to be exactly that string. ``none`` is the
    unset default and proves nothing, so it is rejected like any other value;
    an unavailable, malformed, or rejected setting fails closed *before* the
    reference is computed. The previous value of the same new API is restored
    on the way out; no legacy setting is ever read or restored.
    """
    matmul = require_ieee_fp32_matmul_api(torch)
    previous = matmul.fp32_precision
    try:
        matmul.fp32_precision = FP32_PRECISION_IEEE
    except Exception as exc:  # noqa: BLE001 - any rejection is fail-closed
        raise P33Error(
            f"torch.backends.cuda.matmul.fp32_precision={FP32_PRECISION_IEEE!r} was "
            f"rejected: {exc}"
        ) from exc

    effective = matmul.fp32_precision
    if effective != FP32_PRECISION_IEEE:
        _restore_fp32_precision(matmul, previous)
        raise P33Error(
            f"torch.backends.cuda.matmul.fp32_precision read back as {effective!r}, "
            f"not {FP32_PRECISION_IEEE!r}; the FP32 reference cannot be trusted"
        )
    try:
        yield
    finally:
        _restore_fp32_precision(matmul, previous)


def _restore_fp32_precision(matmul, previous) -> None:
    """Restore the previous new-API setting, reporting a failure to stderr."""
    try:
        matmul.fp32_precision = previous
    except Exception as exc:  # noqa: BLE001 - never mask the original failure
        log(f"WARNING: could not restore fp32_precision to {previous!r}: {exc}")


def collect_provenance(contract: dict, torch, cutlass) -> dict:
    """Collect only the allowlisted provenance fields, failing closed.

    Nothing outside the allowlist is read or recorded: no host name, no user,
    no path, no environment dump.
    """
    require_single_cuda_device(torch)

    device = _query_nvidia_smi()

    major, minor = torch.cuda.get_device_capability(0)
    compute_capability = f"{major}.{minor}"
    if not _RE_COMPUTE_CAPABILITY.match(compute_capability):
        raise P33Error(f"malformed compute capability {compute_capability!r}")
    if compute_capability != contract["EXPECTED_COMPUTE_CAPABILITY"]:
        raise P33Error(
            f"device compute capability {compute_capability} does not match the pinned "
            f"{contract['CUDA_ARCH']} target ({contract['EXPECTED_COMPUTE_CAPABILITY']})"
        )

    nvcc_major_minor = _query_nvcc_major_minor()
    if nvcc_major_minor != contract["CUDA_MAJOR_MINOR"]:
        raise P33Error(
            f"installed CUDA toolkit {nvcc_major_minor} does not match the pinned "
            f"{contract['CUDA_VERSION']}"
        )

    torch_version = str(torch.__version__)
    if torch_version != contract["PYTORCH_VERSION"]:
        raise P33Error(f"torch {torch_version} != pinned {contract['PYTORCH_VERSION']}")
    torch_cuda_version = torch.version.cuda
    if torch_cuda_version != contract["PYTORCH_CUDA_VERSION"]:
        raise P33Error(
            f"torch CUDA {torch_cuda_version} != pinned {contract['PYTORCH_CUDA_VERSION']}"
        )

    cutedsl_version = str(cutlass.__version__)
    if cutedsl_version != contract["CUTEDSL_VERSION"]:
        raise P33Error(f"CuTe DSL {cutedsl_version} != pinned {contract['CUTEDSL_VERSION']}")

    git_state = _repository_git_state(repository_root())

    provenance = {
        "gpu_name": device["gpu_name"],
        "gpu_uuid": device["gpu_uuid"],
        "compute_capability": compute_capability,
        "driver_version": device["driver_version"],
        "cuda_toolkit_version": contract["CUDA_VERSION"],
        "torch_cuda_version": torch_cuda_version,
        "cutedsl_version": cutedsl_version,
        "git_commit": git_state["git_commit"],
        "git_dirty": git_state["git_dirty"],
    }
    return provenance


# --- cuBLASLt bridge ---------------------------------------------------------


def _plan_info_type(ctypes):
    """Build the ctypes mirror of the bridge's ``P33PlanInfo`` struct.

    Every member is 8 bytes wide on both sides, so the layout is unambiguous;
    the loader additionally compares ``ctypes.sizeof`` against the size the
    compiled bridge reports and refuses to continue if they differ.
    """

    class P33PlanInfo(ctypes.Structure):
        _fields_ = [
            ("abi_version", ctypes.c_int64),
            ("cublaslt_version", ctypes.c_int64),
            ("m", ctypes.c_int64),
            ("n", ctypes.c_int64),
            ("k", ctypes.c_int64),
            ("batch_count", ctypes.c_int64),
            ("lda", ctypes.c_int64),
            ("ldb", ctypes.c_int64),
            ("ldc", ctypes.c_int64),
            ("ldd", ctypes.c_int64),
            ("transa", ctypes.c_int64),
            ("transb", ctypes.c_int64),
            ("order_a", ctypes.c_int64),
            ("order_b", ctypes.c_int64),
            ("order_c", ctypes.c_int64),
            ("order_d", ctypes.c_int64),
            ("type_a", ctypes.c_int64),
            ("type_b", ctypes.c_int64),
            ("type_c", ctypes.c_int64),
            ("type_d", ctypes.c_int64),
            ("compute_type", ctypes.c_int64),
            ("scale_type", ctypes.c_int64),
            ("pointer_mode", ctypes.c_int64),
            ("epilogue", ctypes.c_int64),
            ("search_mode", ctypes.c_int64),
            ("workspace_limit_bytes", ctypes.c_int64),
            ("workspace_bytes", ctypes.c_int64),
            ("workspace_is_null", ctypes.c_int64),
            ("heuristic_requested", ctypes.c_int64),
            ("heuristic_returned", ctypes.c_int64),
            ("heuristic_index", ctypes.c_int64),
            ("alignment_a_bytes", ctypes.c_int64),
            ("alignment_b_bytes", ctypes.c_int64),
            ("alignment_c_bytes", ctypes.c_int64),
            ("alignment_d_bytes", ctypes.c_int64),
            ("algo_id", ctypes.c_int64),
            ("tile_id", ctypes.c_int64),
            ("stages_id", ctypes.c_int64),
            ("split_k", ctypes.c_int64),
            ("reduction_scheme", ctypes.c_int64),
            ("cta_swizzling", ctypes.c_int64),
            ("custom_option", ctypes.c_int64),
            ("inner_shape_id", ctypes.c_int64),
            ("cluster_shape_id", ctypes.c_int64),
            ("waves_count", ctypes.c_double),
            ("alpha", ctypes.c_double),
            ("beta", ctypes.c_double),
        ]

    return P33PlanInfo


class CublasLtBridge:
    """Thin, fail-closed ctypes front end for ``cublaslt_bridge.cu``.

    The bridge owns every cuBLASLt object. This class owns nothing but the
    handle to it, translates a non-zero return code into a ``BridgeError``
    carrying the bridge's own diagnostic, and never silently retries.
    """

    def __init__(self, library_path: Path):
        import ctypes

        self._ctypes = ctypes
        if not library_path.is_file():
            raise BridgeError(
                f"the compiled cuBLASLt bridge {library_path} is missing; the Make targets "
                f"build it from {BRIDGE_SOURCE_RELATIVE} immediately before this wrapper runs"
            )
        try:
            self._lib = ctypes.CDLL(str(library_path))
        except OSError as exc:
            raise BridgeError(f"cannot load the cuBLASLt bridge {library_path}: {exc}") from exc

        self.info_type = _plan_info_type(ctypes)

        self._lib.p33_bridge_abi_version.restype = ctypes.c_int
        self._lib.p33_bridge_abi_version.argtypes = []
        self._lib.p33_plan_info_size.restype = ctypes.c_size_t
        self._lib.p33_plan_info_size.argtypes = []
        self._lib.p33_last_error.restype = ctypes.c_char_p
        self._lib.p33_last_error.argtypes = []
        self._lib.p33_cublaslt_version.restype = ctypes.c_size_t
        self._lib.p33_cublaslt_version.argtypes = []
        self._lib.p33_plan_create.restype = ctypes.c_int
        self._lib.p33_plan_create.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(self.info_type),
        ]
        self._lib.p33_plan_execute.restype = ctypes.c_int
        self._lib.p33_plan_execute.argtypes = [ctypes.c_void_p]
        self._lib.p33_stream_synchronize.restype = ctypes.c_int
        self._lib.p33_stream_synchronize.argtypes = [ctypes.c_void_p]
        self._lib.p33_plan_destroy.restype = ctypes.c_int
        self._lib.p33_plan_destroy.argtypes = [ctypes.c_void_p]

        abi = int(self._lib.p33_bridge_abi_version())
        if abi != BRIDGE_ABI_VERSION:
            raise BridgeError(
                f"the compiled bridge reports ABI version {abi}, this wrapper expects "
                f"{BRIDGE_ABI_VERSION}"
            )
        reported_size = int(self._lib.p33_plan_info_size())
        mirror_size = ctypes.sizeof(self.info_type)
        if reported_size != mirror_size:
            raise BridgeError(
                f"the compiled bridge reports a {reported_size}-byte metadata struct, this "
                f"wrapper mirrors {mirror_size} bytes; the ABI does not match"
            )
        self._plan = None

    def _error(self) -> str:
        raw = self._lib.p33_last_error()
        if not raw:
            return "the bridge reported no diagnostic"
        return raw.decode("utf-8", errors="replace")

    def cublaslt_version(self) -> int:
        """The runtime cuBLASLt version, read from the library itself."""
        return int(self._lib.p33_cublaslt_version())

    def create_plan(self, a_ptr, b_ptr, c_ptr, d_ptr, stream):
        ctypes = self._ctypes
        plan = ctypes.c_void_p(None)
        info = self.info_type()
        status = self._lib.p33_plan_create(
            ctypes.c_void_p(a_ptr),
            ctypes.c_void_p(b_ptr),
            ctypes.c_void_p(c_ptr),
            ctypes.c_void_p(d_ptr),
            ctypes.c_void_p(stream),
            ctypes.byref(plan),
            ctypes.byref(info),
        )
        if status != 0:
            raise BridgeError(f"cuBLASLt plan creation failed: {self._error()}")
        if not plan:
            raise BridgeError("cuBLASLt plan creation reported success but returned no plan")
        self._plan = plan
        return info

    def execute(self) -> None:
        if self._plan is None:
            raise BridgeError("no cuBLASLt plan exists; nothing can be executed")
        if self._lib.p33_plan_execute(self._plan) != 0:
            raise BridgeError(f"cublasLtMatmul failed: {self._error()}")

    def synchronize(self) -> None:
        if self._plan is None:
            raise BridgeError("no cuBLASLt plan exists; nothing can be synchronized")
        if self._lib.p33_stream_synchronize(self._plan) != 0:
            raise BridgeError(f"stream synchronization failed: {self._error()}")

    def destroy(self) -> None:
        if self._plan is None:
            return
        plan, self._plan = self._plan, None
        if self._lib.p33_plan_destroy(plan) != 0:
            log(f"WARNING: releasing the cuBLASLt plan failed: {self._error()}")


def validate_plan_info(info) -> dict:
    """Require the bridge's frozen contract to equal this wrapper's, exactly.

    The bridge freezes the shape, layout, transpose modes, types, compute and
    scale types, pointer mode, epilogue, scalars, workspace limit, and heuristic
    request count in C; this wrapper freezes them in Python. Neither is derived
    from the other, so an accidental or deliberate change on one side is a
    disagreement here rather than a silently different measurement.
    """
    expectations = (
        ("abi_version", BRIDGE_ABI_VERSION),
        ("m", FROZEN_M),
        ("n", FROZEN_N),
        ("k", FROZEN_K),
        ("batch_count", FROZEN_L),
        ("lda", FROZEN_LDA),
        ("ldb", FROZEN_LDB),
        ("ldc", FROZEN_LDC),
        ("ldd", FROZEN_LDD),
        ("transa", CUBLAS_ENUM_VALUES[FROZEN_TRANSA]),
        ("transb", CUBLAS_ENUM_VALUES[FROZEN_TRANSB]),
        ("order_a", CUBLAS_ENUM_VALUES[FROZEN_ORDER]),
        ("order_b", CUBLAS_ENUM_VALUES[FROZEN_ORDER]),
        ("order_c", CUBLAS_ENUM_VALUES[FROZEN_ORDER]),
        ("order_d", CUBLAS_ENUM_VALUES[FROZEN_ORDER]),
        ("type_a", CUBLAS_ENUM_VALUES[FROZEN_AB_CUDA_TYPE]),
        ("type_b", CUBLAS_ENUM_VALUES[FROZEN_AB_CUDA_TYPE]),
        ("type_c", CUBLAS_ENUM_VALUES[FROZEN_CD_CUDA_TYPE]),
        ("type_d", CUBLAS_ENUM_VALUES[FROZEN_CD_CUDA_TYPE]),
        ("compute_type", CUBLAS_ENUM_VALUES[FROZEN_COMPUTE_TYPE]),
        ("scale_type", CUBLAS_ENUM_VALUES[FROZEN_SCALE_TYPE]),
        ("pointer_mode", CUBLAS_ENUM_VALUES[FROZEN_POINTER_MODE]),
        ("epilogue", CUBLAS_ENUM_VALUES[FROZEN_EPILOGUE]),
        ("search_mode", CUBLAS_ENUM_VALUES[FROZEN_SEARCH_MODE]),
        ("workspace_limit_bytes", FROZEN_WORKSPACE_LIMIT_BYTES),
        ("heuristic_requested", FROZEN_HEURISTIC_REQUESTED),
    )
    for field, expected in expectations:
        actual = getattr(info, field)
        if int(actual) != int(expected):
            raise BridgeError(
                f"the compiled bridge reports {field}={actual}, this wrapper freezes "
                f"{expected}; P3.3 refuses to measure a configuration it did not specify"
            )

    if float(info.alpha) != FROZEN_ALPHA:
        raise BridgeError(f"the bridge reports alpha={info.alpha}, P3.3 freezes {FROZEN_ALPHA}")
    if float(info.beta) != FROZEN_BETA:
        raise BridgeError(f"the bridge reports beta={info.beta}, P3.3 freezes {FROZEN_BETA}")

    returned = int(info.heuristic_returned)
    index = int(info.heuristic_index)
    if not 1 <= returned <= FROZEN_HEURISTIC_REQUESTED:
        raise BridgeError(
            f"the heuristic returned {returned} result(s), which is outside "
            f"[1, {FROZEN_HEURISTIC_REQUESTED}]"
        )
    if not 0 <= index < returned:
        raise BridgeError(
            f"the selected heuristic index {index} is outside [0, {returned - 1}]"
        )

    workspace_bytes = int(info.workspace_bytes)
    if not 0 <= workspace_bytes <= FROZEN_WORKSPACE_LIMIT_BYTES:
        raise BridgeError(
            f"the selected algorithm needs {workspace_bytes} workspace bytes, outside "
            f"[0, {FROZEN_WORKSPACE_LIMIT_BYTES}]"
        )
    workspace_is_null = int(info.workspace_is_null)
    if workspace_is_null not in (0, 1):
        raise BridgeError(f"malformed workspace_is_null={workspace_is_null}")
    if (workspace_bytes == 0) != (workspace_is_null == 1):
        raise BridgeError(
            "the bridge must use a null workspace pointer exactly when the required "
            f"workspace is zero (bytes={workspace_bytes}, is_null={workspace_is_null})"
        )

    for field in ("alignment_a_bytes", "alignment_b_bytes", "alignment_c_bytes",
                  "alignment_d_bytes"):
        value = int(getattr(info, field))
        if value <= 0 or (value & (value - 1)) != 0:
            raise BridgeError(f"{field}={value} is not a positive power of two")

    waves = float(info.waves_count)
    if not math.isfinite(waves) or waves < 0.0:
        raise BridgeError(f"waves_count={waves!r} must be finite and non-negative")

    split_k = int(info.split_k)
    if split_k < 1:
        raise BridgeError(f"split_k={split_k} must be at least 1")

    for field in ("algo_id", "tile_id", "stages_id", "reduction_scheme", "cta_swizzling",
                  "custom_option", "inner_shape_id", "cluster_shape_id"):
        value = int(getattr(info, field))
        if value < 0:
            raise BridgeError(f"{field}={value} is negative; algorithm metadata is invalid")

    cublaslt_version = int(info.cublaslt_version)
    if cublaslt_version <= 0:
        raise BridgeError(f"cublasLtGetVersion() reported {cublaslt_version}")

    return {
        "workspace_bytes": str(workspace_bytes),
        "alignment_a_bytes": str(int(info.alignment_a_bytes)),
        "alignment_b_bytes": str(int(info.alignment_b_bytes)),
        "alignment_c_bytes": str(int(info.alignment_c_bytes)),
        "alignment_d_bytes": str(int(info.alignment_d_bytes)),
        "heuristic_returned": str(returned),
        "heuristic_index": str(index),
        "algo_id": str(int(info.algo_id)),
        "tile_id": str(int(info.tile_id)),
        "stages_id": str(int(info.stages_id)),
        "split_k": str(split_k),
        "reduction_scheme": str(int(info.reduction_scheme)),
        "cta_swizzling": str(int(info.cta_swizzling)),
        "custom_option": str(int(info.custom_option)),
        "inner_shape_id": str(int(info.inner_shape_id)),
        "cluster_shape_id": str(int(info.cluster_shape_id)),
        "waves_count": format_fixed(waves, DECIMALS_WAVES),
        "cublaslt_version": str(cublaslt_version),
    }


# --- CSV row -----------------------------------------------------------------


def format_fixed(value, decimals: int) -> str:
    """Serialize a finite, non-negative real as a plain fixed-point decimal."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RowContractError(f"{value!r} is not a real number")
    number = float(value)
    if not math.isfinite(number):
        raise RowContractError(f"{value!r} is not finite; NaN and infinity are forbidden")
    if number < 0.0:
        raise RowContractError(f"{value!r} is negative")
    text = f"{number:.{decimals}f}"
    if text.startswith("-"):  # guards against a negative zero surviving rounding
        raise RowContractError(f"{value!r} serialized to a negative decimal")
    return text


def build_row(
    correctness: str,
    max_abs_error,
    max_rel_error,
    setup_time_ms,
    first_launch_ms,
    kernel_time_ms,
    warmup_iterations: int,
    iterations: int,
    provenance: dict,
    upstream: dict,
    plan: dict,
) -> dict:
    """Build the single frozen CSV row, refusing anything but a passed check.

    This is the only way a row is constructed, so a failed or skipped
    correctness check cannot produce an emittable row.
    """
    if correctness != CORRECTNESS_PASS:
        raise RowContractError(
            f"refusing to build a row with correctness={correctness!r}; "
            f"only {CORRECTNESS_PASS} may be emitted"
        )

    for name, value in (
        ("setup_time_ms", setup_time_ms),
        ("first_launch_ms", first_launch_ms),
        ("kernel_time_ms", kernel_time_ms),
    ):
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise RowContractError(f"{name}={value!r} must be finite and strictly positive")

    row = dict(CSV_FIXED_VALUES)
    row.update(
        {
            "alpha": format_fixed(FROZEN_ALPHA, DECIMALS_SCALAR),
            "beta": format_fixed(FROZEN_BETA, DECIMALS_SCALAR),
            "atol": format_fixed(FROZEN_ATOL, DECIMALS_TOLERANCE),
            "rtol": format_fixed(FROZEN_RTOL, DECIMALS_TOLERANCE),
            "max_abs_error": format_fixed(max_abs_error, DECIMALS_ERROR),
            "max_rel_error": format_fixed(max_rel_error, DECIMALS_ERROR),
            "setup_time_ms": format_fixed(setup_time_ms, DECIMALS_TIMING),
            "first_launch_ms": format_fixed(first_launch_ms, DECIMALS_TIMING),
            "kernel_time_ms": format_fixed(kernel_time_ms, DECIMALS_TIMING),
            "warmup_iterations": str(int(warmup_iterations)),
            "iterations": str(int(iterations)),
            "cutlass_commit": upstream["commit"],
            "upstream_example_sha256": upstream["sha256"],
        }
    )
    row.update(plan)
    row.update(provenance)
    validate_row(row)
    return row


def validate_row(row) -> None:
    """Fail closed on any row that violates the frozen schema."""
    if not isinstance(row, dict):
        raise RowContractError("a CSV row must be a mapping")

    keys = set(row)
    expected = set(CSV_FIELDS)
    missing = sorted(expected - keys)
    if missing:
        raise RowContractError(f"missing field(s): {', '.join(missing)}")
    unknown = sorted(keys - expected)
    if unknown:
        raise RowContractError(f"unknown field(s): {', '.join(unknown)}")
    if len(CSV_FIELDS) != len(expected):
        raise RowContractError("the frozen schema contains a duplicate field name")

    for field in CSV_FIELDS:
        value = row[field]
        if not isinstance(value, str):
            raise RowContractError(f"{field}: value {value!r} is not a string")
        if value == "" or not _RE_SAFE_TEXT.match(value):
            raise RowContractError(
                f"{field}: value {value!r} is empty or contains control characters"
            )

    for field, fixed in CSV_FIXED_VALUES.items():
        if row[field] != fixed:
            raise RowContractError(f"{field}: {row[field]!r} != frozen {fixed!r}")

    for field in CSV_BOOL_FIELDS:
        if row[field] not in (BOOL_TRUE, BOOL_FALSE):
            raise RowContractError(
                f"{field}: {row[field]!r} is not a canonical lowercase boolean"
            )

    for field in CSV_COUNT_FIELDS:
        if not _RE_POSITIVE_INT.match(row[field]):
            raise RowContractError(f"{field}: {row[field]!r} is not a positive integer")
    _validate_bounded_count("warmup_iterations", row["warmup_iterations"], MAX_WARMUP_ITERATIONS)
    _validate_bounded_count("iterations", row["iterations"], MAX_ITERATIONS)

    for field in CSV_TIMING_FIELDS:
        _validate_decimal(field, row[field], DECIMALS_TIMING, strictly_positive=True)
    for field in CSV_ERROR_FIELDS:
        _validate_decimal(field, row[field], DECIMALS_ERROR, strictly_positive=False)
    for field in CSV_TOLERANCE_FIELDS:
        _validate_decimal(field, row[field], DECIMALS_TOLERANCE, strictly_positive=True)
    _validate_decimal("alpha", row["alpha"], DECIMALS_SCALAR, strictly_positive=True)
    _validate_decimal("beta", row["beta"], DECIMALS_SCALAR, strictly_positive=False)
    _validate_decimal("waves_count", row["waves_count"], DECIMALS_WAVES, strictly_positive=False)

    if float(row["alpha"]) != FROZEN_ALPHA:
        raise RowContractError(f"alpha: {row['alpha']!r} != frozen {FROZEN_ALPHA}")
    if float(row["beta"]) != FROZEN_BETA:
        raise RowContractError(f"beta: {row['beta']!r} != frozen {FROZEN_BETA}")

    for field in CSV_NONNEGATIVE_INT_FIELDS:
        if not _RE_NONNEGATIVE_INT.match(row[field]):
            raise RowContractError(
                f"{field}: {row[field]!r} is not a canonical non-negative decimal integer"
            )
    for field in CSV_POSITIVE_INT_FIELDS:
        if not _RE_POSITIVE_INT.match(row[field]):
            raise RowContractError(
                f"{field}: {row[field]!r} is not a canonical positive decimal integer"
            )

    workspace_bytes = int(row["workspace_bytes"])
    if workspace_bytes > FROZEN_WORKSPACE_LIMIT_BYTES:
        raise RowContractError(
            f"workspace_bytes: {workspace_bytes} exceeds the frozen limit "
            f"{FROZEN_WORKSPACE_LIMIT_BYTES}"
        )
    returned = int(row["heuristic_returned"])
    if returned > FROZEN_HEURISTIC_REQUESTED:
        raise RowContractError(
            f"heuristic_returned: {returned} exceeds the frozen request "
            f"{FROZEN_HEURISTIC_REQUESTED}"
        )
    if int(row["heuristic_index"]) >= returned:
        raise RowContractError(
            f"heuristic_index: {row['heuristic_index']} is not below heuristic_returned "
            f"{returned}"
        )
    for field in ("alignment_a_bytes", "alignment_b_bytes", "alignment_c_bytes",
                  "alignment_d_bytes"):
        value = int(row[field])
        if value & (value - 1):
            raise RowContractError(f"{field}: {value} is not a power of two")

    if not _RE_HEX40.match(row["cutlass_commit"]):
        raise RowContractError(
            f"cutlass_commit: {row['cutlass_commit']!r} is not a 40-hex commit"
        )
    if not _RE_HEX40.match(row["git_commit"]):
        raise RowContractError(f"git_commit: {row['git_commit']!r} is not a 40-hex commit")
    if not _RE_HEX64.match(row["upstream_example_sha256"]):
        raise RowContractError("upstream_example_sha256 is not a 64-hex digest")
    if not _RE_GPU_UUID.match(row["gpu_uuid"]):
        raise RowContractError(f"gpu_uuid: {row['gpu_uuid']!r} is malformed")
    if not _RE_COMPUTE_CAPABILITY.match(row["compute_capability"]):
        raise RowContractError(f"compute_capability: {row['compute_capability']!r} is malformed")
    for field in ("driver_version", "cuda_toolkit_version", "torch_cuda_version",
                  "cutedsl_version"):
        if not _RE_DOTTED_VERSION.match(row[field]):
            raise RowContractError(f"{field}: {row[field]!r} is not a dotted version")


def _validate_bounded_count(field: str, text: str, maximum: int) -> None:
    value = int(text)
    if not MIN_ITERATIONS <= value <= maximum:
        raise RowContractError(f"{field}: {value} is outside [{MIN_ITERATIONS}, {maximum}]")


def _validate_decimal(field: str, text: str, decimals: int, strictly_positive: bool) -> None:
    if not re.fullmatch(rf"(0|[1-9][0-9]*)\.[0-9]{{{decimals}}}", text):
        raise RowContractError(
            f"{field}: {text!r} is not a fixed-point decimal with {decimals} fractional digits"
        )
    value = float(text)
    if not math.isfinite(value):
        raise RowContractError(f"{field}: {text!r} is not finite")
    if strictly_positive and value <= 0.0:
        raise RowContractError(f"{field}: {text!r} must be strictly positive")


def serialize_row(row: dict) -> str:
    """Serialize the validated row with the csv module (never by concatenation)."""
    validate_row(row)
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(CSV_FIELDS),
        extrasaction="raise",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    writer.writerow(row)
    return buffer.getvalue()


# --- stdout discipline -------------------------------------------------------


def _redirect_stdout_to_stderr() -> int:
    """Send everything written to descriptor 1 to stderr; return the real one.

    A native library can write to descriptor 1, which would corrupt the
    two-line CSV contract. Redirecting at the descriptor level - rather than
    only rebinding ``sys.stdout`` - covers native writes too.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    os.dup2(2, 1)
    return saved


def _emit_on_saved_stdout(saved_fd: int, text: str) -> None:
    """Write the CSV to the real stdout and close the saved descriptor."""
    with os.fdopen(saved_fd, "wb", closefd=True) as handle:
        handle.write(text.encode("utf-8"))
        handle.flush()


# --- Operands ----------------------------------------------------------------


def _describe(tensor) -> str:
    return (
        f"shape={tuple(tensor.shape)} strides={tuple(tensor.stride())} "
        f"dtype={tensor.dtype} device={tensor.device}"
    )


def require_operand_layout(torch, tensor, name, shape, strides, dtype) -> None:
    """Fail closed unless a device operand has exactly the frozen layout.

    cuBLASLt is told the leading dimensions as constants; if the tensor that
    actually backs a pointer had a different shape, stride, dtype, or device,
    the library would read the wrong memory and the measurement would be
    meaningless. Nothing here is corrected, transposed, or made contiguous: a
    mismatch is a hard failure.
    """
    if tensor.dtype is not dtype:
        raise P33Error(f"operand {name} has dtype {tensor.dtype}, expected {dtype}")
    if tensor.device.type != "cuda":
        raise P33Error(f"operand {name} is on {tensor.device}, expected a CUDA device")
    if tuple(tensor.shape) != tuple(shape):
        raise P33Error(f"operand {name} has {_describe(tensor)}, expected shape {tuple(shape)}")
    if tuple(tensor.stride()) != tuple(strides):
        raise P33Error(
            f"operand {name} has {_describe(tensor)}, expected strides {tuple(strides)}; "
            "P3.3 never silently transposes or re-lays out data"
        )
    if tensor.data_ptr() == 0:
        raise P33Error(f"operand {name} has a null device pointer")


def create_operands(torch, cutlass_torch, cutlass):
    """Build the frozen operands exactly as the pinned upstream factory does.

    The upstream factory seeds once and then calls ``cutlass.torch.matrix``
    three times, for A, then B, then C. That order is what fixes the RNG
    stream, so it is reproduced here call for call, with the same seed, the
    same factory, the same dtypes, and the same major flags. The device copies
    are made in the same order with the same ``empty_like``/``copy_`` pair the
    upstream helper performs, and - unlike the upstream helper, which discards
    the device tensors for A and B - every device tensor is retained here,
    because those allocations are what the cuBLASLt bridge receives pointers
    into.
    """
    ab_dtype = getattr(cutlass, FROZEN_AB_DTYPE)
    c_dtype = getattr(cutlass, FROZEN_C_DTYPE)

    torch.manual_seed(FROZEN_SEED)
    a_cpu = cutlass_torch.matrix(FROZEN_L, FROZEN_M, FROZEN_K, FROZEN_A_MAJOR == "m", ab_dtype)
    b_cpu = cutlass_torch.matrix(FROZEN_L, FROZEN_N, FROZEN_K, FROZEN_B_MAJOR == "m", ab_dtype)
    c_cpu = cutlass_torch.matrix(FROZEN_L, FROZEN_M, FROZEN_N, FROZEN_C_MAJOR == "m", c_dtype)

    device_tensors = []
    for host in (a_cpu, b_cpu, c_cpu):
        device = torch.empty_like(host, device="cuda")
        device.copy_(host)
        device_tensors.append(device)
    a_gpu, b_gpu, c_gpu = device_tensors

    # The frozen physical layouts, restated as the exact shapes and strides the
    # descriptor contract assumes: A is M x K with stride (K, 1), B is N x K
    # with stride (K, 1), and C is M x N with stride (N, 1). The trailing L
    # mode is 1 and carries the whole-matrix stride.
    require_operand_layout(
        torch, a_gpu, "A", (FROZEN_M, FROZEN_K, FROZEN_L),
        (FROZEN_LDA, 1, FROZEN_M * FROZEN_K), torch.bfloat16,
    )
    require_operand_layout(
        torch, b_gpu, "B", (FROZEN_N, FROZEN_K, FROZEN_L),
        (FROZEN_LDB, 1, FROZEN_N * FROZEN_K), torch.bfloat16,
    )
    require_operand_layout(
        torch, c_gpu, "C/D", (FROZEN_M, FROZEN_N, FROZEN_L),
        (FROZEN_LDC, 1, FROZEN_M * FROZEN_N), torch.float32,
    )
    return a_cpu, b_cpu, a_gpu, b_gpu, c_gpu


# --- Measurement -------------------------------------------------------------


def execute_measurement(warmup_iterations: int, iterations: int) -> str:
    """Run the frozen sequence once and return the CSV text on success.

    Returning the text rather than writing it keeps the single emission point
    in ``main`` and makes it structurally impossible to emit a row from any
    path that did not complete this whole sequence.
    """
    contract = load_pinned_contract()

    # (1) The pinned upstream identity is proved before anything heavy is
    # imported, and the tensor-construction sequence P3.3 replicates is proved
    # to still match the upstream factory P3.2 calls.
    upstream = verify_upstream_source(contract)
    log(f"upstream verified: commit {upstream['commit']} sha256 {upstream['sha256']}")
    factory = verify_upstream_tensor_factory(upstream["path"])
    log(
        "operand equivalence with P3.2: the upstream factory still seeds "
        f"{FROZEN_SEED} and builds {len(factory['matrix_calls'])} matrices in the "
        "order P3.3 replicates"
    )

    import cutlass
    import cutlass.torch as cutlass_torch
    import torch

    # (2) Environment and provenance, before any tensor exists. The IEEE FP32
    # API is required up front so a PyTorch that cannot guarantee a trustworthy
    # correctness verdict fails closed before any GPU work is spent on it; the
    # setting itself is established and verified around the reference.
    log("collecting environment and provenance")
    require_ieee_fp32_matmul_api(torch)
    provenance = collect_provenance(contract, torch, cutlass)

    revalidated = verify_upstream_source(contract)
    if revalidated != upstream:
        raise P33Error("the pinned upstream source changed during provenance collection")
    log(
        f"device: {provenance['gpu_name']} uuid={provenance['gpu_uuid']} "
        f"cc={provenance['compute_capability']} driver={provenance['driver_version']}"
    )

    # (3) Tensor preparation, entirely outside every timer. Allocation, the
    # host-to-device conversion, and the reference operands are never timed.
    log("allocating tensors (outside every timer)")
    a_cpu, b_cpu, a_gpu, b_gpu, c_gpu = create_operands(torch, cutlass_torch, cutlass)

    torch_stream = torch.cuda.current_stream()
    bridge = CublasLtBridge(BRIDGE_LIBRARY_PATH)
    log(f"cuBLASLt runtime version: {bridge.cublaslt_version()}")

    # C and D are the same FP32 M x N buffer: beta is exactly 0, so C is never
    # read, and the descriptor contract gives C and D identical layouts.
    c_ptr = c_gpu.data_ptr()

    try:
        # (4)+(5) Plan creation only: descriptors, layouts, preference, the
        # vendor heuristic, algorithm validation, and the workspace. No matmul
        # runs inside this timer and nothing is compiled - which is exactly why
        # this field is setup_time_ms and not compile_time_ms.
        log("creating the cuBLASLt plan (descriptors, heuristic, workspace)")
        torch.cuda.synchronize()
        setup_start = time.perf_counter_ns()
        info = bridge.create_plan(
            a_gpu.data_ptr(), b_gpu.data_ptr(), c_ptr, c_ptr, torch_stream.cuda_stream
        )
        setup_time_ms = (time.perf_counter_ns() - setup_start) / 1e6

        plan_fields = validate_plan_info(info)
        log(
            f"heuristic: requested {FROZEN_HEURISTIC_REQUESTED}, returned "
            f"{plan_fields['heuristic_returned']}, selected index "
            f"{plan_fields['heuristic_index']} (algo_id {plan_fields['algo_id']}, "
            f"tile {plan_fields['tile_id']}, stages {plan_fields['stages_id']}, "
            f"split_k {plan_fields['split_k']}, workspace "
            f"{plan_fields['workspace_bytes']} B)"
        )

        # (6) Synchronize, then (7) exactly one measured first launch.
        torch.cuda.synchronize()
        log("first cublasLtMatmul launch (also the correctness-validated launch)")
        first_launch_start = time.perf_counter_ns()
        bridge.execute()
        torch.cuda.synchronize()
        first_launch_ms = (time.perf_counter_ns() - first_launch_start) / 1e6

        # (8)+(9) Correctness on the complete output of that first launch.
        max_abs_error, max_rel_error = validate_result(torch, a_cpu, b_cpu, c_gpu)
        log(
            f"correctness: {CORRECTNESS_PASS} "
            f"(max_abs_error={max_abs_error!r} max_rel_error={max_rel_error!r})"
        )

        # (10) Warm-up and (11) steady state, only after correctness passed.
        log(f"warm-up: {warmup_iterations} launch(es)")
        for _ in range(warmup_iterations):
            bridge.execute()
        torch.cuda.synchronize()

        log(f"steady state: {iterations} measured launch(es)")
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(torch_stream)
        for _ in range(iterations):
            bridge.execute()
        end_event.record(torch_stream)
        torch.cuda.synchronize()
        total_ms = start_event.elapsed_time(end_event)
    finally:
        bridge.destroy()

    if not math.isfinite(total_ms) or total_ms <= 0.0:
        raise P33Error(f"CUDA-event elapsed time {total_ms!r} is not finite and positive")
    # (12)
    kernel_time_ms = total_ms / iterations

    for name, value in (
        ("setup_time_ms", setup_time_ms),
        ("first_launch_ms", first_launch_ms),
        ("kernel_time_ms", kernel_time_ms),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise P33Error(f"{name}={value!r} is not finite and strictly positive")

    # (13)
    row = build_row(
        correctness=CORRECTNESS_PASS,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        setup_time_ms=setup_time_ms,
        first_launch_ms=first_launch_ms,
        kernel_time_ms=kernel_time_ms,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        provenance=provenance,
        upstream=upstream,
        plan=plan_fields,
    )
    return serialize_row(row)


def validate_result(torch, a_torch_cpu, b_torch_cpu, d_torch_gpu):
    """Validate the complete result against an untimed IEEE-FP32 CUDA oracle.

    This is the same oracle P3.2 uses, unchanged: the pinned PyTorch
    installation used purely as a correctness reference, never timed and never
    reported as a competing method. P3.3 is the cuBLASLt baseline; the oracle
    is not, and neither is compared against the other here - that comparison
    belongs to P3.5.

    The reference is computed under the IEEE FP32 guard, which fails closed
    unless ``fp32_precision`` reads back as exactly ``ieee``. There is no
    fallback reference, no CPU reference, and no reduced-precision path.
    """
    with ieee_fp32_matmul(torch):
        reference = torch.einsum(
            "mkl,nkl->mnl",
            a_torch_cpu.to(device="cuda", dtype=torch.float32),
            b_torch_cpu.to(device="cuda", dtype=torch.float32),
        )
    result = d_torch_gpu.to(dtype=torch.float32)

    if tuple(result.shape) != tuple(reference.shape):
        raise CorrectnessError(
            f"result shape {tuple(result.shape)} != reference shape {tuple(reference.shape)}"
        )
    if not bool(torch.isfinite(result).all()):
        raise CorrectnessError("the cuBLASLt result contains non-finite values")
    if not bool(torch.isfinite(reference).all()):
        raise CorrectnessError("the FP32 reference contains non-finite values")

    difference = (result - reference).abs()
    tolerated = FROZEN_ATOL + FROZEN_RTOL * reference.abs()
    mismatches = int((difference > tolerated).sum().item())

    denominator = reference.abs().clamp_min(REL_ERROR_DENOMINATOR_FLOOR)
    max_abs_error = float(difference.max().item())
    max_rel_error = float((difference / denominator).max().item())

    if not math.isfinite(max_abs_error) or not math.isfinite(max_rel_error):
        raise CorrectnessError("the measured error is not finite")

    if mismatches:
        raise CorrectnessError(
            f"{mismatches} element(s) exceed atol={FROZEN_ATOL} rtol={FROZEN_RTOL}; "
            f"max_abs_error={max_abs_error} max_rel_error={max_rel_error}"
        )
    return max_abs_error, max_rel_error


# --- Command line ------------------------------------------------------------


def bounded_int(minimum: int, maximum: int):
    """argparse type for a positive, explicitly bounded iteration count."""

    def parse(text: str) -> int:
        stripped = text.strip()
        if not re.fullmatch(r"[0-9]+", stripped):
            raise argparse.ArgumentTypeError(f"{text!r} is not a non-negative integer")
        value = int(stripped)
        if not minimum <= value <= maximum:
            raise argparse.ArgumentTypeError(
                f"{value} is outside the permitted range [{minimum}, {maximum}]"
            )
        return value

    return parse


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the whole P3.3 command line.

    The frozen GEMM configuration and the frozen algorithm policy are
    deliberately unreachable from here: there is no shape, dtype, layout,
    transpose, leading-dimension, alpha, beta, epilogue, workspace, heuristic,
    algorithm, cache-mode, or publication option, and no way to skip the
    reference check.
    """
    parser = argparse.ArgumentParser(
        prog="cublaslt_gemm.py",
        description=(
            "P3.3 cuBLASLt BF16 GEMM baseline. Executes exactly one frozen configuration "
            "through a direct cublasLtMatmul call on the same operands P3.2 uses, and emits "
            "one non-publishable CSV row of functional evidence. Not an experimental "
            "campaign, not a performance result, and not a comparison against CuTe DSL."
        ),
        epilog=(
            "Correctness is mandatory and always runs before any warm-up or steady-state "
            "timing. The emitted timings are P3.3 infrastructure evidence only."
        ),
    )
    parser.add_argument(
        "--warmup-iterations",
        type=bounded_int(MIN_ITERATIONS, MAX_WARMUP_ITERATIONS),
        default=DEFAULT_WARMUP_ITERATIONS,
        metavar="N",
        help=(
            f"untimed launches before the measured ones "
            f"[{MIN_ITERATIONS}..{MAX_WARMUP_ITERATIONS}], default {DEFAULT_WARMUP_ITERATIONS}"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=bounded_int(MIN_ITERATIONS, MAX_ITERATIONS),
        default=DEFAULT_ITERATIONS,
        metavar="N",
        help=(
            f"measured launches for kernel_time_ms "
            f"[{MIN_ITERATIONS}..{MAX_ITERATIONS}], default {DEFAULT_ITERATIONS}"
        ),
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the GPU-free contract self-test and exit (no CUDA, no output on stdout)",
    )
    return parser


# --- GPU-free self-test ------------------------------------------------------


def _synthetic_provenance() -> dict:
    """Obviously synthetic values: no pinned contract value is duplicated here."""
    return {
        "gpu_name": "SYNTHETIC TEST DEVICE",
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "compute_capability": "9.9",
        "driver_version": "999.99.99",
        "cuda_toolkit_version": "99.9.9",
        "torch_cuda_version": "98.7",
        "cutedsl_version": "97.6.5",
        "git_commit": "0" * 40,
        "git_dirty": BOOL_FALSE,
    }


def _synthetic_upstream() -> dict:
    return {"commit": "1" * 40, "sha256": "2" * 64}


def _synthetic_plan() -> dict:
    """Obviously synthetic cuBLASLt metadata within every frozen bound."""
    return {
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
        "split_k": "1",
        "reduction_scheme": "0",
        "cta_swizzling": "0",
        "custom_option": "0",
        "inner_shape_id": "0",
        "cluster_shape_id": "0",
        "waves_count": "1.500000",
        "cublaslt_version": "999999",
    }


def _synthetic_row() -> dict:
    return build_row(
        correctness=CORRECTNESS_PASS,
        max_abs_error=0.0,
        max_rel_error=0.0,
        setup_time_ms=12.5,
        first_launch_ms=3.25,
        kernel_time_ms=7.5,
        warmup_iterations=2,
        iterations=10,
        provenance=_synthetic_provenance(),
        upstream=_synthetic_upstream(),
        plan=_synthetic_plan(),
    )


class _SyntheticPlanInfo:
    """A mutable stand-in for the bridge's metadata struct, for the self-test."""

    def __init__(self, **overrides):
        values = {
            "abi_version": BRIDGE_ABI_VERSION,
            "cublaslt_version": 999999,
            "m": FROZEN_M,
            "n": FROZEN_N,
            "k": FROZEN_K,
            "batch_count": FROZEN_L,
            "lda": FROZEN_LDA,
            "ldb": FROZEN_LDB,
            "ldc": FROZEN_LDC,
            "ldd": FROZEN_LDD,
            "transa": CUBLAS_ENUM_VALUES[FROZEN_TRANSA],
            "transb": CUBLAS_ENUM_VALUES[FROZEN_TRANSB],
            "order_a": CUBLAS_ENUM_VALUES[FROZEN_ORDER],
            "order_b": CUBLAS_ENUM_VALUES[FROZEN_ORDER],
            "order_c": CUBLAS_ENUM_VALUES[FROZEN_ORDER],
            "order_d": CUBLAS_ENUM_VALUES[FROZEN_ORDER],
            "type_a": CUBLAS_ENUM_VALUES[FROZEN_AB_CUDA_TYPE],
            "type_b": CUBLAS_ENUM_VALUES[FROZEN_AB_CUDA_TYPE],
            "type_c": CUBLAS_ENUM_VALUES[FROZEN_CD_CUDA_TYPE],
            "type_d": CUBLAS_ENUM_VALUES[FROZEN_CD_CUDA_TYPE],
            "compute_type": CUBLAS_ENUM_VALUES[FROZEN_COMPUTE_TYPE],
            "scale_type": CUBLAS_ENUM_VALUES[FROZEN_SCALE_TYPE],
            "pointer_mode": CUBLAS_ENUM_VALUES[FROZEN_POINTER_MODE],
            "epilogue": CUBLAS_ENUM_VALUES[FROZEN_EPILOGUE],
            "search_mode": CUBLAS_ENUM_VALUES[FROZEN_SEARCH_MODE],
            "workspace_limit_bytes": FROZEN_WORKSPACE_LIMIT_BYTES,
            "workspace_bytes": 4194304,
            "workspace_is_null": 0,
            "heuristic_requested": FROZEN_HEURISTIC_REQUESTED,
            "heuristic_returned": 8,
            "heuristic_index": 0,
            "alignment_a_bytes": 256,
            "alignment_b_bytes": 256,
            "alignment_c_bytes": 256,
            "alignment_d_bytes": 256,
            "algo_id": 21,
            "tile_id": 27,
            "stages_id": 15,
            "split_k": 1,
            "reduction_scheme": 0,
            "cta_swizzling": 0,
            "custom_option": 0,
            "inner_shape_id": 0,
            "cluster_shape_id": 0,
            "waves_count": 1.5,
            "alpha": FROZEN_ALPHA,
            "beta": FROZEN_BETA,
        }
        values.update(overrides)
        for key, value in values.items():
            setattr(self, key, value)


_SYNTHETIC_UPSTREAM_FACTORY = '''
import torch
import cutlass.torch as cutlass_torch


def create_tensors(l, m, n, k, a_major, b_major, c_major, ab_dtype, c_dtype):
    torch.manual_seed(1111)
    a_torch_cpu = cutlass_torch.matrix(l, m, k, a_major == "m", ab_dtype)
    b_torch_cpu = cutlass_torch.matrix(l, n, k, b_major == "n", ab_dtype)
    c_torch_cpu = cutlass_torch.matrix(l, m, n, c_major == "m", c_dtype)
    return a_torch_cpu, b_torch_cpu, c_torch_cpu
'''


def run_self_test() -> int:
    """Prove the GPU-free half of the P3.3 contract, printing only to stderr."""
    failures = []

    def check(name, condition, detail=""):
        if condition:
            print(f"  ok   {name}", file=sys.stderr)
        else:
            failures.append(f"{name}{': ' + detail if detail else ''}")
            print(f"  FAIL {name} {detail}", file=sys.stderr)

    def rejects(name, callable_, expected_fragment):
        try:
            callable_()
        except P33Error as exc:
            check(name, expected_fragment in str(exc), f"message was {str(exc)!r}")
        else:
            check(name, False, "no error was raised")

    print("cublaslt_gemm --self-test (GPU-free)", file=sys.stderr)

    # Frozen schema.
    check("schema has 77 fields", len(CSV_FIELDS) == 77, str(len(CSV_FIELDS)))
    check("schema has no duplicate field", len(set(CSV_FIELDS)) == len(CSV_FIELDS))
    check("schema starts with schema_version", CSV_FIELDS[0] == "schema_version")
    check("schema ends with publishable", CSV_FIELDS[-1] == "publishable")
    check("schema version is p33.v1", SCHEMA_VERSION == "p33.v1")
    check(
        "the P3.2 schema is neither reused nor reinterpreted",
        SCHEMA_VERSION != "p32.v1" and "compile_time_ms" not in CSV_FIELDS,
    )
    check(
        "setup time is not misnamed as compilation",
        "setup_time_ms" in CSV_FIELDS and "compile_time_ms" not in CSV_FIELDS,
    )
    check(
        "no performance metric is in the schema",
        not any(
            re.search(r"tflop|speedup|efficien|bandwidth|utilization|throughput|winner", field)
            for field in CSV_FIELDS
        ),
    )

    # Frozen configuration.
    check("problem is (4096, 4096, 4096, 1)", FROZEN_MNKL == (4096, 4096, 4096, 1))
    check("leading dimensions are (K, K, N, N)",
          (FROZEN_LDA, FROZEN_LDB, FROZEN_LDC, FROZEN_LDD)
          == (FROZEN_K, FROZEN_K, FROZEN_N, FROZEN_N))
    check("transa is CUBLAS_OP_N", FROZEN_TRANSA == "CUBLAS_OP_N")
    check("transb is CUBLAS_OP_T", FROZEN_TRANSB == "CUBLAS_OP_T")
    check("every layout is row-major", FROZEN_ORDER == "CUBLASLT_ORDER_ROW")
    check("A/B type is BF16", FROZEN_AB_CUDA_TYPE == "CUDA_R_16BF")
    check("C/D type is FP32", FROZEN_CD_CUDA_TYPE == "CUDA_R_32F")
    check("compute type is FP32", FROZEN_COMPUTE_TYPE == "CUBLAS_COMPUTE_32F")
    check("scale type is FP32", FROZEN_SCALE_TYPE == "CUDA_R_32F")
    check("pointer mode is host", FROZEN_POINTER_MODE == "CUBLASLT_POINTER_MODE_HOST")
    check("epilogue is the identity default", FROZEN_EPILOGUE == "CUBLASLT_EPILOGUE_DEFAULT")
    check("alpha is 1 and beta is 0", (FROZEN_ALPHA, FROZEN_BETA) == (1.0, 0.0))
    check("workspace limit is exactly 64 MiB",
          FROZEN_WORKSPACE_LIMIT_BYTES == 67108864, str(FROZEN_WORKSPACE_LIMIT_BYTES))
    check("exactly 32 heuristic results are requested", FROZEN_HEURISTIC_REQUESTED == 32)
    check("the search mode is best fit", FROZEN_SEARCH_MODE == "CUBLASLT_SEARCH_BEST_FIT")
    check("variant is the first supported heuristic result",
          VARIANT == "heuristic_first_supported")
    check("method is cublaslt", METHOD == "cublaslt")
    check("seed is 1111", FROZEN_SEED == 1111)
    check("tolerances are 1e-1 / 1e-5", (FROZEN_ATOL, FROZEN_RTOL) == (1e-1, 1e-5))
    check("cache model is hot", CACHE_MODE == "hot")
    check("publishable is fixed to false", PUBLISHABLE == BOOL_FALSE)
    check("run_kind is smoke", RUN_KIND == "smoke")
    check("reference is the untimed torch CUDA FP32 oracle",
          REFERENCE == "torch_cuda_fp32_ieee")
    check("the operand majors match P3.2",
          (FROZEN_A_MAJOR, FROZEN_B_MAJOR, FROZEN_C_MAJOR) == ("k", "k", "n"))

    # Serialization.
    row = _synthetic_row()
    text = serialize_row(row)
    lines = text.splitlines()
    check("serialization emits exactly two lines", len(lines) == 2, str(len(lines)))
    check("header matches the frozen order", lines[0] == ",".join(CSV_FIELDS))
    parsed = list(csv.DictReader(io.StringIO(text)))
    check("exactly one data row is parsed back", len(parsed) == 1)
    check("the round trip is lossless", parsed and dict(parsed[0]) == row)
    check("timings use six fractional digits", row["kernel_time_ms"] == "7.500000")
    check("errors use nine fractional digits", row["max_abs_error"] == "0.000000000")
    check("atol serializes deterministically", row["atol"] == "0.100000000")
    check("rtol serializes deterministically", row["rtol"] == "0.000010000")
    check("alpha serializes deterministically", row["alpha"] == "1.000000000")
    check("beta serializes deterministically", row["beta"] == "0.000000000")
    check("the workspace limit is serialized exactly",
          row["workspace_limit_bytes"] == "67108864")
    check("the heuristic request is serialized exactly", row["heuristic_requested"] == "32")
    check("the row is not publishable", row["publishable"] == BOOL_FALSE)
    check("no field uses exponent notation",
          not any("e" in value.lower() and field not in ("gpu_name",)
                  for field, value in row.items()
                  if re.fullmatch(r"[0-9.eE+-]+", value)))

    # Rejections: correctness gating.
    for bad in ("FAIL", "SKIPPED", "pass", ""):
        rejects(
            f"correctness={bad!r} cannot build a row",
            lambda bad=bad: build_row(
                correctness=bad,
                max_abs_error=0.0,
                max_rel_error=0.0,
                setup_time_ms=1.0,
                first_launch_ms=1.0,
                kernel_time_ms=1.0,
                warmup_iterations=2,
                iterations=10,
                provenance=_synthetic_provenance(),
                upstream=_synthetic_upstream(),
                plan=_synthetic_plan(),
            ),
            "refusing to build a row",
        )

    # Rejections: schema shape.
    rejects(
        "a missing field is rejected",
        lambda: validate_row({k: v for k, v in row.items() if k != "kernel_time_ms"}),
        "missing field",
    )
    rejects(
        "an unknown field is rejected",
        lambda: validate_row({**row, "tflops": "1.0"}),
        "unknown field",
    )
    rejects(
        "a P3.2 field name is rejected",
        lambda: validate_row({**row, "compile_time_ms": "1.000000"}),
        "unknown field",
    )
    rejects(
        "a non-string value is rejected",
        lambda: validate_row({**row, "iterations": 10}),
        "not a string",
    )

    # Rejections: numbers.
    for bad in ("nan", "inf", "-inf", "0.000000", "-1.000000", "1e3", "1.5"):
        rejects(
            f"kernel_time_ms={bad!r} is rejected",
            lambda bad=bad: validate_row({**row, "kernel_time_ms": bad}),
            "kernel_time_ms",
        )
    rejects(
        "a non-finite float cannot be serialized",
        lambda: format_fixed(float("nan"), DECIMALS_TIMING),
        "not finite",
    )
    rejects(
        "a negative setup time is rejected",
        lambda: validate_row({**row, "setup_time_ms": "-1.000000"}),
        "setup_time_ms",
    )
    rejects(
        "a non-finite waves count is rejected",
        lambda: validate_row({**row, "waves_count": "nan"}),
        "waves_count",
    )

    # Rejections: frozen configuration.
    for field, bad in (
        ("m", "8192"), ("n", "8192"), ("k", "8192"), ("l", "2"),
        ("lda", "8192"), ("ldb", "8192"), ("ldc", "8192"), ("ldd", "8192"),
        ("transa", "CUBLAS_OP_T"), ("transb", "CUBLAS_OP_N"),
        ("order_a", "CUBLASLT_ORDER_COL"), ("order_d", "CUBLASLT_ORDER_COL"),
        ("ab_dtype", "Float16"), ("c_dtype", "BFloat16"),
        ("compute_type", "CUBLAS_COMPUTE_32F_FAST_TF32"),
        ("scale_type", "CUDA_R_16BF"),
        ("pointer_mode", "CUBLASLT_POINTER_MODE_DEVICE"),
        ("epilogue", "CUBLASLT_EPILOGUE_RELU"),
        ("workspace_limit_bytes", "134217728"),
        ("heuristic_requested", "64"),
        ("cache_mode", "cold"),
        ("method", "cutedsl"),
        ("variant", "nonpersistent_1cta"),
        ("seed", "2222"),
    ):
        rejects(
            f"a changed {field} is rejected",
            lambda field=field, bad=bad: validate_row({**row, field: bad}),
            "frozen",
        )
    rejects(
        "a changed alpha is rejected",
        lambda: validate_row({**row, "alpha": "2.000000000"}),
        "alpha",
    )
    rejects(
        "a changed beta is rejected",
        lambda: validate_row({**row, "beta": "1.000000000"}),
        "beta",
    )
    rejects(
        "publishable=true is rejected",
        lambda: validate_row({**row, "publishable": BOOL_TRUE}),
        "publishable",
    )
    rejects(
        "correctness=FAIL is rejected in a row",
        lambda: validate_row({**row, "correctness": "FAIL"}),
        "correctness",
    )
    rejects(
        "a non-canonical boolean is rejected",
        lambda: validate_row({**row, "git_dirty": "TRUE"}),
        "git_dirty",
    )

    # Rejections: cuBLASLt metadata bounds.
    rejects(
        "a workspace above the frozen limit is rejected",
        lambda: validate_row({**row, "workspace_bytes": str(FROZEN_WORKSPACE_LIMIT_BYTES + 1)}),
        "exceeds the frozen limit",
    )
    rejects(
        "more returned results than requested is rejected",
        lambda: validate_row({**row, "heuristic_returned": "33"}),
        "exceeds the frozen request",
    )
    rejects(
        "a heuristic index outside the returned range is rejected",
        lambda: validate_row({**row, "heuristic_index": "8"}),
        "heuristic_index",
    )
    rejects(
        "a non-power-of-two alignment is rejected",
        lambda: validate_row({**row, "alignment_a_bytes": "100"}),
        "power of two",
    )
    rejects(
        "a zero heuristic count is rejected",
        lambda: validate_row({**row, "heuristic_returned": "0"}),
        "heuristic_returned",
    )
    rejects(
        "a negative split_k is rejected",
        lambda: validate_row({**row, "split_k": "0"}),
        "split_k",
    )
    rejects(
        "a malformed cuBLASLt version is rejected",
        lambda: validate_row({**row, "cublaslt_version": "13.2"}),
        "cublaslt_version",
    )
    rejects(
        "a malformed GPU UUID is rejected",
        lambda: validate_row({**row, "gpu_uuid": "0000"}),
        "gpu_uuid",
    )
    rejects(
        "a malformed upstream digest is rejected",
        lambda: validate_row({**row, "upstream_example_sha256": "abc"}),
        "upstream_example_sha256",
    )
    rejects(
        "an out-of-range iteration count is rejected",
        lambda: validate_row({**row, "iterations": str(MAX_ITERATIONS + 1)}),
        "outside",
    )

    # Bridge metadata agreement.
    good_info = _SyntheticPlanInfo()
    plan_fields = validate_plan_info(good_info)
    check("a well-formed bridge report is accepted",
          plan_fields["algo_id"] == "21" and plan_fields["waves_count"] == "1.500000")
    for field, bad in (
        ("m", 8192), ("k", 2048), ("lda", 2048), ("ldd", 8192),
        ("transa", 1), ("transb", 0), ("order_a", 0), ("order_d", 0),
        ("type_a", 0), ("type_c", 14),
        ("compute_type", 77), ("scale_type", 14),
        ("pointer_mode", 1), ("epilogue", 2), ("search_mode", 1),
        ("workspace_limit_bytes", 134217728), ("heuristic_requested", 64),
        ("batch_count", 2), ("abi_version", 99),
    ):
        rejects(
            f"a bridge reporting {field}={bad} is rejected",
            lambda field=field, bad=bad: validate_plan_info(_SyntheticPlanInfo(**{field: bad})),
            field,
        )
    rejects(
        "a bridge reporting alpha != 1 is rejected",
        lambda: validate_plan_info(_SyntheticPlanInfo(alpha=2.0)),
        "alpha",
    )
    rejects(
        "a bridge reporting beta != 0 is rejected",
        lambda: validate_plan_info(_SyntheticPlanInfo(beta=1.0)),
        "beta",
    )
    rejects(
        "a bridge exceeding the workspace limit is rejected",
        lambda: validate_plan_info(
            _SyntheticPlanInfo(workspace_bytes=FROZEN_WORKSPACE_LIMIT_BYTES + 1)
        ),
        "workspace",
    )
    rejects(
        "a non-null pointer for a zero workspace is rejected",
        lambda: validate_plan_info(_SyntheticPlanInfo(workspace_bytes=0, workspace_is_null=0)),
        "null workspace pointer",
    )
    rejects(
        "a null pointer for a non-zero workspace is rejected",
        lambda: validate_plan_info(_SyntheticPlanInfo(workspace_bytes=4096, workspace_is_null=1)),
        "null workspace pointer",
    )
    rejects(
        "a heuristic returning nothing is rejected",
        lambda: validate_plan_info(_SyntheticPlanInfo(heuristic_returned=0)),
        "outside",
    )
    rejects(
        "a heuristic index past the returned count is rejected",
        lambda: validate_plan_info(_SyntheticPlanInfo(heuristic_returned=4, heuristic_index=4)),
        "outside",
    )
    rejects(
        "a non-finite waves count from the bridge is rejected",
        lambda: validate_plan_info(_SyntheticPlanInfo(waves_count=float("inf"))),
        "waves_count",
    )
    rejects(
        "a negative waves count from the bridge is rejected",
        lambda: validate_plan_info(_SyntheticPlanInfo(waves_count=-1.0)),
        "waves_count",
    )
    rejects(
        "an overstated non-power-of-two alignment is rejected",
        lambda: validate_plan_info(_SyntheticPlanInfo(alignment_b_bytes=96)),
        "power of two",
    )
    rejects(
        "invalid algorithm metadata is rejected",
        lambda: validate_plan_info(_SyntheticPlanInfo(algo_id=-1)),
        "algo_id",
    )

    # Upstream operand-equivalence proof, exercised on synthetic sources.
    calls, seeds = extract_upstream_matrix_calls(_SYNTHETIC_UPSTREAM_FACTORY)
    check("the upstream factory parser finds three matrix calls", len(calls) == 3,
          str(len(calls)))
    check("the upstream factory parser finds the frozen seed", seeds == [FROZEN_SEED],
          str(seeds))
    check(
        "the parsed call order is A, B, C",
        [call["args"][1] for call in calls] == ["m", "n", "m"]
        and [call["args"][2] for call in calls] == ["k", "k", "n"],
    )
    rejects(
        "a factory with a different seed is rejected",
        lambda: _verify_factory_text(_SYNTHETIC_UPSTREAM_FACTORY.replace("1111", "2222")),
        "does not seed",
    )
    rejects(
        "a factory with a reordered call is rejected",
        lambda: _verify_factory_text(
            _SYNTHETIC_UPSTREAM_FACTORY.replace(
                "cutlass_torch.matrix(l, n, k, b_major", "cutlass_torch.matrix(l, k, n, b_major"
            )
        ),
        "diverged",
    )
    rejects(
        "a factory with a dropped call is rejected",
        lambda: _verify_factory_text(
            "\n".join(
                line for line in _SYNTHETIC_UPSTREAM_FACTORY.splitlines()
                if "c_torch_cpu = " not in line
            )
        ),
        "call(s)",
    )
    rejects(
        "a missing factory is rejected",
        lambda: _verify_factory_text("def something_else():\n    pass\n"),
        "no longer defines",
    )

    # Command line.
    parser = build_arg_parser()
    options = set(re.findall(r"--[a-z0-9][a-z0-9-]*", parser.format_help()))
    check(
        "only the four permitted controls exist",
        options == {"--warmup-iterations", "--iterations", "--self-test", "--help"},
        str(sorted(options)),
    )
    parsed_args = parser.parse_args([])
    check(
        "defaults are the documented non-publishable ones",
        (parsed_args.warmup_iterations, parsed_args.iterations)
        == (DEFAULT_WARMUP_ITERATIONS, DEFAULT_ITERATIONS),
    )
    for bad in ("0", "-1", str(MAX_ITERATIONS + 1), "abc", ""):
        try:
            bounded_int(MIN_ITERATIONS, MAX_ITERATIONS)(bad)
        except argparse.ArgumentTypeError:
            check(f"iteration argument {bad!r} is rejected", True)
        else:
            check(f"iteration argument {bad!r} is rejected", False)

    # Version-contract parsing.
    rejects(
        "a malformed contract line is rejected",
        lambda: _parse_env_text("NOT A CONTRACT LINE\n"),
        "malformed contract line",
    )
    rejects(
        "a duplicate contract key is rejected",
        lambda: _parse_env_text("A_KEY=1\nA_KEY=2\n"),
        "duplicate contract key",
    )
    check(
        "comments and blank lines are ignored",
        _parse_env_text("# comment\n\nA_KEY=1\n") == {"A_KEY": "1"},
    )
    # The architecture itself is never re-declared here: only the derivation is
    # exercised, on targets that are deliberately not the pinned one, plus the
    # real pinned value read from the global contract.
    check("sm_75 maps to compute capability 7.5", compute_capability_for_arch("sm_75") == "7.5")
    check("sm_90 maps to compute capability 9.0", compute_capability_for_arch("sm_90") == "9.0")
    check("sm_100 maps to compute capability 10.0",
          compute_capability_for_arch("sm_100") == "10.0")
    real_contract = load_pinned_contract()
    check(
        "the pinned architecture derives a well-formed compute capability",
        bool(_RE_COMPUTE_CAPABILITY.match(real_contract["EXPECTED_COMPUTE_CAPABILITY"])),
        f"{real_contract['CUDA_ARCH']} -> {real_contract['EXPECTED_COMPUTE_CAPABILITY']}",
    )
    rejects(
        "a malformed architecture pin is rejected",
        lambda: compute_capability_for_arch("blackwell"),
        "malformed",
    )

    # No heavy import and no shared library load happened.
    for module_name in ("torch", "cutlass", "cuda", "ctypes"):
        check(f"{module_name} was not imported", module_name not in sys.modules)

    if failures:
        print(f"SELF-TEST: FAIL ({len(failures)} case(s))", file=sys.stderr)
        return 1
    print("SELF-TEST: PASS", file=sys.stderr)
    return 0


def _verify_factory_text(text: str) -> dict:
    """Run the upstream-factory proof over in-memory text, for the self-test."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic_upstream.py"
        path.write_text(text, encoding="utf-8")
        return verify_upstream_tensor_factory(path)


def _parse_env_text(text: str) -> dict:
    """Parse contract text from memory, for the GPU-free self-test only."""
    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.env"
        path.write_text(text, encoding="utf-8")
        return parse_env_file(path)


# --- Entry point -------------------------------------------------------------


def main(argv=None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.self_test:
        return run_self_test()

    # Descriptor 1 becomes stderr for the whole measurement; the real stdout is
    # restored only to emit the two CSV lines, and only on full success.
    saved_stdout_fd = _redirect_stdout_to_stderr()
    try:
        csv_text = execute_measurement(args.warmup_iterations, args.iterations)
    except CorrectnessError as exc:
        os.close(saved_stdout_fd)
        log(f"CORRECTNESS FAILED: {exc}")
        log("no CSV header and no CSV row are emitted; no warm-up or steady-state timing ran")
        return 1
    except P33Error as exc:
        os.close(saved_stdout_fd)
        log(f"FAIL: {exc}")
        return 1
    except BaseException:
        os.close(saved_stdout_fd)
        raise

    # Only a fully completed sequence, with correctness already passed, reaches
    # this line, and this is the only place a CSV row is ever written.
    _emit_on_saved_stdout(saved_stdout_fd, csv_text)
    log("emitted one non-publishable P3.3 row (functional evidence, not a result)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
