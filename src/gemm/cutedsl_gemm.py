#!/usr/bin/env python3
"""P3.2 - one-shape CuTe DSL BF16 GEMM wrapper (frozen; no performance claim).

This is a thin orchestration wrapper around the pinned, unmodified official
NVIDIA CuTe DSL dense GEMM example that P3.1 froze. It owns no GEMM kernel of
its own: the kernel class, the tensor factory, and the JIT entry point all come
from the upstream file, loaded read-only and in place from the pinned
``/opt/cutlass`` checkout after its commit, Git blob SHA, and SHA-256 have been
verified against the repository's two version contracts.

Why a wrapper exists at all: the upstream ``run()`` function combines JIT
compilation, the first launch, reference validation, and its own benchmarking
helper into a single call that returns one number. P3.2 needs the three costs
separated, so it drives the same upstream objects directly and never calls
``run()``.

What this program measures, in this exact order:

1. environment and provenance (exactly one CUDA GPU, as logical device 0);
2. deterministic tensor allocation, entirely outside every timer;
3. ``compile_time_ms``  - a monotonic host clock around ``cute.compile`` only;
4. ``first_launch_ms``  - a monotonic host clock around the first launch of the
   compiled kernel, whose output is also the tensor validated for correctness;
5. complete FP32 correctness validation against an untimed PyTorch CUDA oracle
   with TF32 (and every other reduced-precision FP32 matmul mode) disabled;
6. only if correctness passes: warm-up launches, then ``kernel_time_ms`` from
   CUDA events on the same stream, divided by the measured iteration count.

What this program is not: it is not an experimental campaign, not a cuBLASLt
comparison, and not a performance result. Every emitted row carries
``publishable=false``. No TFLOP/s, speedup, efficiency, utilization, or
bandwidth number is computed anywhere, the three timings are P3.2
functional-verification evidence only, and the untimed PyTorch oracle is a
correctness reference - never a competing method and never the P3.3 baseline.

Output contract:

* stdout receives exactly one CSV header line and exactly one CSV data row,
  and nothing else. To make that true even when the JIT toolchain writes to
  file descriptor 1 from native code, descriptor 1 is redirected to descriptor
  2 for the whole measurement and the real stdout is restored only to emit the
  two CSV lines, after correctness has already passed.
* stderr receives every human-readable message: progress, warnings, compiler
  output, and diagnostics.
* A correctness failure exits non-zero, prints a diagnostic to stderr, and
  emits no CSV header and no CSV data row; no warm-up and no steady-state
  timing runs in that case.

Usage:
  cutedsl_gemm.py [--warmup-iterations N] [--iterations N]
  cutedsl_gemm.py --self-test
  cutedsl_gemm.py --help

``--help`` and ``--self-test`` are GPU-free and import neither PyTorch, nor
CuTe DSL, nor the CUDA bindings: every heavy import is deferred into the
measurement path.

Exit code: 0 only when the whole sequence succeeded and one valid row was
emitted, 1 on any contract, provenance, correctness, or execution failure, and
2 on a usage error.
"""

import argparse
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

SCHEMA_VERSION = "p32.v1"
EXPERIMENT = "exp03_cutedsl_vs_cublaslt"
UNIT = "P3.2"
RUN_KIND = "smoke"
METHOD = "cutedsl"
VARIANT = "nonpersistent_1cta"
REFERENCE = "torch_cuda_fp32_ieee"
CACHE_MODE = "hot"
CORRECTNESS_PASS = "PASS"
PUBLISHABLE = "false"

# --- Frozen GEMM configuration ----------------------------------------------
#
# P3.2 executes exactly one configuration. None of these values is reachable
# from the command line, from an environment variable, or from a configuration
# file: they are immutable constants of this unit. The other four final shapes,
# the persistent scheduler, and the 2-CTA MMA group belong to P3.3-P3.5.

FROZEN_M = 4096
FROZEN_N = 4096
FROZEN_K = 4096
FROZEN_L = 1
FROZEN_MNKL = (FROZEN_M, FROZEN_N, FROZEN_K, FROZEN_L)

FROZEN_AB_DTYPE = "BFloat16"
FROZEN_ACC_DTYPE = "Float32"
FROZEN_C_DTYPE = "Float32"

FROZEN_A_MAJOR = "k"
FROZEN_B_MAJOR = "k"
FROZEN_C_MAJOR = "n"

FROZEN_MMA_TILER_MN = (128, 128)
FROZEN_CLUSTER_SHAPE_MN = (1, 1)
FROZEN_USE_2CTA_INSTRS = False
FROZEN_USE_TMA_STORE = True

FROZEN_SEED = 1111
FROZEN_ATOL = 1e-1
FROZEN_RTOL = 1e-5

# The only CUDA matmul FP32 policy P3.2 accepts for its correctness oracle,
# via the PyTorch 2.10 fp32_precision API and nothing else. The unset default
# is "none", which proves nothing and is rejected.
FP32_PRECISION_IEEE = "ieee"

# Safe denominator for the reported relative error. The reported
# ``max_rel_error`` is max(|c - ref| / max(|ref|, REL_ERROR_DENOMINATOR_FLOOR)),
# which stays finite and well defined where the reference is exactly zero.
# It is a diagnostic only: the pass/fail decision uses the elementwise
# criterion |c - ref| <= atol + rtol * |ref| at full precision, never this
# reported scalar.
REL_ERROR_DENOMINATOR_FLOOR = 1.0

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
    "mma_tiler_m": FROZEN_MMA_TILER_MN[0],
    "mma_tiler_n": FROZEN_MMA_TILER_MN[1],
    "cluster_m": FROZEN_CLUSTER_SHAPE_MN[0],
    "cluster_n": FROZEN_CLUSTER_SHAPE_MN[1],
    "use_2cta_instrs": FROZEN_USE_2CTA_INSTRS,
    "use_tma_store": FROZEN_USE_TMA_STORE,
    "seed": FROZEN_SEED,
    "reference": REFERENCE,
    "atol": FROZEN_ATOL,
    "rtol": FROZEN_RTOL,
    "cache_mode": CACHE_MODE,
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
    "mma_tiler_m",
    "mma_tiler_n",
    "cluster_m",
    "cluster_n",
    "use_2cta_instrs",
    "use_tma_store",
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
    "mma_tiler_m": str(FROZEN_MMA_TILER_MN[0]),
    "mma_tiler_n": str(FROZEN_MMA_TILER_MN[1]),
    "cluster_m": str(FROZEN_CLUSTER_SHAPE_MN[0]),
    "cluster_n": str(FROZEN_CLUSTER_SHAPE_MN[1]),
    "use_2cta_instrs": "false",
    "use_tma_store": "true",
    "seed": str(FROZEN_SEED),
    "reference": REFERENCE,
    "correctness": CORRECTNESS_PASS,
    "cache_mode": CACHE_MODE,
    "publishable": PUBLISHABLE,
}

CSV_TIMING_FIELDS = ("compile_time_ms", "first_launch_ms", "kernel_time_ms")
CSV_ERROR_FIELDS = ("max_abs_error", "max_rel_error")
CSV_TOLERANCE_FIELDS = ("atol", "rtol")
CSV_COUNT_FIELDS = ("warmup_iterations", "iterations")
CSV_BOOL_FIELDS = ("use_2cta_instrs", "use_tma_store", "git_dirty", "publishable")

BOOL_TRUE = "true"
BOOL_FALSE = "false"

_RE_HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")
_RE_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_RE_GPU_UUID = re.compile(r"\AGPU-[0-9a-fA-F][0-9a-fA-F-]+\Z")
_RE_DOTTED_VERSION = re.compile(r"\A[0-9]+(\.[0-9]+)*\Z")
_RE_COMPUTE_CAPABILITY = re.compile(r"\A[0-9]+\.[0-9]+\Z")
_RE_POSITIVE_INT = re.compile(r"\A[1-9][0-9]*\Z")
_RE_ENV_LINE = re.compile(r"\A([A-Z][A-Z0-9_]*)=(\S*)\Z")
_RE_CUDA_ARCH = re.compile(r"\Asm_([0-9]+)([a-z]?)\Z")
_RE_SAFE_TEXT = re.compile(r"\A[^\x00-\x1f\x7f]+\Z")

# The pinned CUTLASS checkout inside the pinned image. The image builds it at
# exactly CUTLASS_COMMIT (see Dockerfile) and P3.1 already executes the example
# in place from here; nothing is ever written to it, and this location is not
# configurable at runtime.
UPSTREAM_CHECKOUT_DIR = Path("/opt/cutlass")

GLOBAL_CONTRACT_FILE = "VERSIONS.env"
PHASE3_CONTRACT_FILE = "PHASE3_VERSIONS.env"

# Keys read from the two version contracts. Nothing below is duplicated as a
# literal anywhere in this file: the pinned commit, blob, SHA-256, versions,
# and architecture exist here only as key names.
GLOBAL_CONTRACT_KEYS = ("CUDA_VERSION", "CUTLASS_VERSION", "CUTLASS_COMMIT", "CUDA_ARCH")
PHASE3_CONTRACT_KEYS = (
    "PYTORCH_VERSION",
    "PYTORCH_CUDA_VERSION",
    "CUTEDSL_P31_EXAMPLE_PATH",
    "CUTEDSL_P31_EXAMPLE_GIT_BLOB",
    "CUTEDSL_P31_EXAMPLE_SHA256",
)


class P32Error(Exception):
    """Any fail-closed P3.2 contract, provenance, or execution failure."""


class RowContractError(P32Error):
    """A CSV row violated the frozen P3.2 schema."""


class CorrectnessError(P32Error):
    """The complete result did not match the untimed FP32 reference."""


def log(message: str) -> None:
    """Write one human-readable progress/diagnostic line to stderr."""
    print(f"cutedsl_gemm: {message}", file=sys.stderr, flush=True)


# --- Version contracts -------------------------------------------------------


def repository_root() -> Path:
    """Locate the repository root that owns this file.

    ``src/gemm/cutedsl_gemm.py`` is two directories below the root both on the
    host and inside the container, where the repository is mounted at
    ``/workspace``.
    """
    root = Path(__file__).resolve().parents[2]
    for name in (GLOBAL_CONTRACT_FILE, PHASE3_CONTRACT_FILE):
        if not (root / name).is_file():
            raise P32Error(f"repository root {root} does not contain {name}")
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
        raise P32Error(f"cannot read version contract {path}: {exc}") from exc

    values: dict = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _RE_ENV_LINE.match(line)
        if match is None:
            raise P32Error(f"{path}:{lineno}: malformed contract line {raw!r}")
        key, value = match.group(1), match.group(2)
        if key in values:
            raise P32Error(f"{path}:{lineno}: duplicate contract key {key}")
        values[key] = value
    return values


def load_pinned_contract(repo_root=None) -> dict:
    """Read every pinned value P3.2 needs from the two version contracts.

    ``VERSIONS.env`` is the closed global contract that the audited P1/P2
    aggregators parse against their own closed key allowlist; P3.2 only reads
    it. ``PHASE3_VERSIONS.env`` is the Phase 3-only extension that P3.1
    created. P3.2 adds no key to either file and reuses P3.1's already pinned
    upstream path, blob, and SHA-256 because it executes the same file.
    """
    root = Path(repo_root) if repo_root is not None else repository_root()
    global_values = parse_env_file(root / GLOBAL_CONTRACT_FILE)
    phase3_values = parse_env_file(root / PHASE3_CONTRACT_FILE)

    contract = {}
    for key in GLOBAL_CONTRACT_KEYS:
        if key not in global_values:
            raise P32Error(f"{GLOBAL_CONTRACT_FILE} is missing required key {key}")
        contract[key] = global_values[key]
    for key in PHASE3_CONTRACT_KEYS:
        if key not in phase3_values:
            raise P32Error(f"{PHASE3_CONTRACT_FILE} is missing required key {key}")
        contract[key] = phase3_values[key]

    if not _RE_HEX40.match(contract["CUTLASS_COMMIT"]):
        raise P32Error(f"pinned CUTLASS_COMMIT is malformed: {contract['CUTLASS_COMMIT']!r}")
    if not _RE_HEX40.match(contract["CUTEDSL_P31_EXAMPLE_GIT_BLOB"]):
        raise P32Error("pinned CUTEDSL_P31_EXAMPLE_GIT_BLOB is malformed")
    if not _RE_HEX64.match(contract["CUTEDSL_P31_EXAMPLE_SHA256"]):
        raise P32Error("pinned CUTEDSL_P31_EXAMPLE_SHA256 is malformed")
    for key in ("CUDA_VERSION", "PYTORCH_CUDA_VERSION"):
        if not _RE_DOTTED_VERSION.match(contract[key]):
            raise P32Error(f"pinned {key} is malformed: {contract[key]!r}")
    if not contract["CUTLASS_VERSION"].startswith("v"):
        raise P32Error(f"pinned CUTLASS_VERSION is malformed: {contract['CUTLASS_VERSION']!r}")

    # Derived, never separately pinned.
    contract["CUTEDSL_VERSION"] = contract["CUTLASS_VERSION"][1:]
    if not _RE_DOTTED_VERSION.match(contract["CUTEDSL_VERSION"]):
        raise P32Error("pinned CuTe DSL version is malformed")

    example_path = contract["CUTEDSL_P31_EXAMPLE_PATH"]
    pure = Path(example_path)
    if pure.is_absolute() or ".." in pure.parts or not example_path.endswith(".py"):
        raise P32Error(f"pinned upstream example path is unsafe: {example_path!r}")

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
        raise P32Error(f"pinned CUDA_ARCH is malformed: {cuda_arch!r}")
    digits = match.group(1)
    if len(digits) < 2:
        raise P32Error(f"pinned CUDA_ARCH is malformed: {cuda_arch!r}")
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
        raise P32Error(f"git {' '.join(args)} could not be executed: {exc}") from exc
    if completed.returncode != 0:
        raise P32Error(
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
        raise P32Error(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def verify_upstream_source(contract: dict) -> dict:
    """Prove the pinned upstream checkout and example file are byte-identical.

    Fails closed on a missing checkout, a wrong HEAD, any tracked or untracked
    modification, a symlinked or non-regular example file, a wrong Git blob
    SHA, or a wrong SHA-256. This runs before the module is imported and again
    before the run is trusted; the checkout is only ever queried, never
    written.
    """
    checkout = UPSTREAM_CHECKOUT_DIR
    if not checkout.is_dir():
        raise P32Error(f"pinned CUTLASS checkout {checkout} is missing")

    head = _git(["-C", str(checkout), "rev-parse", "HEAD"], safe_directory=str(checkout))
    if head != contract["CUTLASS_COMMIT"]:
        raise P32Error(
            f"{checkout} HEAD {head} != pinned CUTLASS_COMMIT {contract['CUTLASS_COMMIT']}"
        )

    dirty = _git(
        ["-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        safe_directory=str(checkout),
    )
    if dirty:
        raise P32Error(f"{checkout} has tracked or untracked modifications")

    example = checkout / contract["CUTEDSL_P31_EXAMPLE_PATH"]
    if example.is_symlink():
        raise P32Error(f"{example} is a symlink")
    if not example.is_file():
        raise P32Error(f"{example} is not a regular file")

    blob = _git(
        ["-C", str(checkout), "hash-object", "--", str(example)],
        safe_directory=str(checkout),
    )
    if blob != contract["CUTEDSL_P31_EXAMPLE_GIT_BLOB"]:
        raise P32Error(
            f"{example} Git blob {blob} != pinned {contract['CUTEDSL_P31_EXAMPLE_GIT_BLOB']}"
        )

    sha256 = sha256_of_file(example)
    if sha256 != contract["CUTEDSL_P31_EXAMPLE_SHA256"]:
        raise P32Error(
            f"{example} SHA-256 {sha256} != pinned {contract['CUTEDSL_P31_EXAMPLE_SHA256']}"
        )

    return {"commit": head, "blob": blob, "sha256": sha256, "path": example}


def load_upstream_module(example: Path):
    """Import the verified upstream example as a library, never as a script.

    The module is loaded under its own private name so the upstream
    ``if __name__ == "__main__"`` block - which parses arguments and calls
    ``run()`` - never executes. The file is read from ``/opt/cutlass`` and is
    neither copied, vendored, reformatted, nor patched.
    """
    import importlib.util

    module_name = "p32_pinned_upstream_dense_gemm"
    spec = importlib.util.spec_from_file_location(module_name, str(example))
    if spec is None or spec.loader is None:
        raise P32Error(f"cannot build an import spec for {example}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - fail closed with the real cause
        sys.modules.pop(module_name, None)
        raise P32Error(f"cannot import the pinned upstream example: {exc}") from exc

    for attribute in ("DenseGemmKernel", "create_tensors"):
        if not hasattr(module, attribute):
            raise P32Error(f"the pinned upstream example does not provide {attribute}")
    return module


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
        raise P32Error(f"nvidia-smi could not be executed: {exc}") from exc
    if completed.returncode != 0:
        raise P32Error(
            f"nvidia-smi failed with exit code {completed.returncode}; "
            "device provenance is ambiguous"
        )

    rows = [row for row in csv.reader(io.StringIO(completed.stdout)) if row]
    if len(rows) != 1:
        raise P32Error(f"nvidia-smi reported {len(rows)} GPUs; exactly 1 must be visible")
    fields = [value.strip() for value in rows[0]]
    if len(fields) != 3:
        raise P32Error("nvidia-smi returned a malformed device row")

    uuid, name, driver_version = fields
    if not _RE_GPU_UUID.match(uuid):
        raise P32Error(f"nvidia-smi returned a malformed GPU UUID: {uuid!r}")
    if not name or not _RE_SAFE_TEXT.match(name):
        raise P32Error("nvidia-smi returned a malformed GPU name")
    if not _RE_DOTTED_VERSION.match(driver_version):
        raise P32Error(f"nvidia-smi returned a malformed driver version: {driver_version!r}")
    return {"gpu_uuid": uuid, "gpu_name": name, "driver_version": driver_version}


def _query_nvcc_major_minor() -> str:
    """Read the installed CUDA toolkit's ``release X.Y`` from nvcc."""
    try:
        completed = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise P32Error(f"nvcc could not be executed: {exc}") from exc
    if completed.returncode != 0:
        raise P32Error("nvcc --version failed; the CUDA toolkit version is ambiguous")
    match = re.search(r"release ([0-9]+)\.([0-9]+)", completed.stdout)
    if match is None:
        raise P32Error("nvcc --version did not report a release version")
    return f"{match.group(1)}.{match.group(2)}"


def _repository_git_state(root: Path) -> dict:
    """Record this repository's commit and dirty state."""
    commit = _git(["rev-parse", "HEAD"], cwd=root)
    if not _RE_HEX40.match(commit):
        raise P32Error(f"repository HEAD is malformed: {commit!r}")
    status = _git(["status", "--porcelain", "--untracked-files=all"], cwd=root)
    return {"git_commit": commit, "git_dirty": BOOL_TRUE if status else BOOL_FALSE}


def require_single_cuda_device(torch) -> None:
    """Require exactly one CUDA-visible GPU, used as logical device 0."""
    if not torch.cuda.is_available():
        raise P32Error("no CUDA device is available; P3.2 requires exactly one GPU")
    count = torch.cuda.device_count()
    if count != 1:
        raise P32Error(f"expected exactly 1 CUDA-visible GPU, saw {count}")
    torch.cuda.set_device(0)
    current = torch.cuda.current_device()
    if current != 0:
        raise P32Error(f"the selected CUDA device must be logical device 0, got {current}")


def require_ieee_fp32_matmul_api(torch):
    """Return the CUDA matmul backend, failing closed without the 2.10 API.

    P3.2 uses **exclusively** the PyTorch 2.10 ``fp32_precision`` API for CUDA
    matrix multiplication. The legacy ``allow_tf32`` property is never read and
    never written: in 2.10 the two are aliases of one setting, mixing them is
    unsupported, and the last write silently wins - setting ``allow_tf32``
    after ``fp32_precision`` rewrites the policy to ``tf32`` without any error.
    ``torch.set_float32_matmul_precision()`` is likewise never combined with it.
    """
    backends = getattr(torch, "backends", None)
    cuda_backend = getattr(backends, "cuda", None) if backends is not None else None
    matmul = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
    if matmul is None:
        raise P32Error(
            "this PyTorch does not expose torch.backends.cuda.matmul; the IEEE FP32 "
            "reference cannot be guaranteed"
        )
    if not hasattr(matmul, "fp32_precision"):
        raise P32Error(
            "this PyTorch does not support torch.backends.cuda.matmul.fp32_precision; "
            "P3.2 requires that API and never falls back to the legacy TF32 flag"
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
        raise P32Error(
            f"torch.backends.cuda.matmul.fp32_precision={FP32_PRECISION_IEEE!r} was "
            f"rejected: {exc}"
        ) from exc

    effective = matmul.fp32_precision
    if effective != FP32_PRECISION_IEEE:
        _restore_fp32_precision(matmul, previous)
        raise P32Error(
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
        raise P32Error(f"malformed compute capability {compute_capability!r}")
    if compute_capability != contract["EXPECTED_COMPUTE_CAPABILITY"]:
        raise P32Error(
            f"device compute capability {compute_capability} does not match the pinned "
            f"{contract['CUDA_ARCH']} target ({contract['EXPECTED_COMPUTE_CAPABILITY']})"
        )

    nvcc_major_minor = _query_nvcc_major_minor()
    if nvcc_major_minor != contract["CUDA_MAJOR_MINOR"]:
        raise P32Error(
            f"installed CUDA toolkit {nvcc_major_minor} does not match the pinned "
            f"{contract['CUDA_VERSION']}"
        )

    torch_version = str(torch.__version__)
    if torch_version != contract["PYTORCH_VERSION"]:
        raise P32Error(f"torch {torch_version} != pinned {contract['PYTORCH_VERSION']}")
    torch_cuda_version = torch.version.cuda
    if torch_cuda_version != contract["PYTORCH_CUDA_VERSION"]:
        raise P32Error(
            f"torch CUDA {torch_cuda_version} != pinned {contract['PYTORCH_CUDA_VERSION']}"
        )

    cutedsl_version = str(cutlass.__version__)
    if cutedsl_version != contract["CUTEDSL_VERSION"]:
        raise P32Error(f"CuTe DSL {cutedsl_version} != pinned {contract['CUTEDSL_VERSION']}")

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
    compile_time_ms,
    first_launch_ms,
    kernel_time_ms,
    warmup_iterations: int,
    iterations: int,
    provenance: dict,
    upstream: dict,
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
        ("compile_time_ms", compile_time_ms),
        ("first_launch_ms", first_launch_ms),
        ("kernel_time_ms", kernel_time_ms),
    ):
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise RowContractError(f"{name}={value!r} must be finite and strictly positive")

    row = dict(CSV_FIXED_VALUES)
    row.update(
        {
            "atol": format_fixed(FROZEN_ATOL, DECIMALS_TOLERANCE),
            "rtol": format_fixed(FROZEN_RTOL, DECIMALS_TOLERANCE),
            "max_abs_error": format_fixed(max_abs_error, DECIMALS_ERROR),
            "max_rel_error": format_fixed(max_rel_error, DECIMALS_ERROR),
            "compile_time_ms": format_fixed(compile_time_ms, DECIMALS_TIMING),
            "first_launch_ms": format_fixed(first_launch_ms, DECIMALS_TIMING),
            "kernel_time_ms": format_fixed(kernel_time_ms, DECIMALS_TIMING),
            "warmup_iterations": str(int(warmup_iterations)),
            "iterations": str(int(iterations)),
            "cutlass_commit": upstream["commit"],
            "upstream_example_sha256": upstream["sha256"],
        }
    )
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
            raise RowContractError(f"{field}: value {value!r} is empty or contains control characters")

    for field, fixed in CSV_FIXED_VALUES.items():
        if row[field] != fixed:
            raise RowContractError(f"{field}: {row[field]!r} != frozen {fixed!r}")

    for field in CSV_BOOL_FIELDS:
        if row[field] not in (BOOL_TRUE, BOOL_FALSE):
            raise RowContractError(f"{field}: {row[field]!r} is not a canonical lowercase boolean")

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

    if not _RE_HEX40.match(row["cutlass_commit"]):
        raise RowContractError(f"cutlass_commit: {row['cutlass_commit']!r} is not a 40-hex commit")
    if not _RE_HEX40.match(row["git_commit"]):
        raise RowContractError(f"git_commit: {row['git_commit']!r} is not a 40-hex commit")
    if not _RE_HEX64.match(row["upstream_example_sha256"]):
        raise RowContractError("upstream_example_sha256 is not a 64-hex digest")
    if not _RE_GPU_UUID.match(row["gpu_uuid"]):
        raise RowContractError(f"gpu_uuid: {row['gpu_uuid']!r} is malformed")
    if not _RE_COMPUTE_CAPABILITY.match(row["compute_capability"]):
        raise RowContractError(f"compute_capability: {row['compute_capability']!r} is malformed")
    for field in ("driver_version", "cuda_toolkit_version", "torch_cuda_version", "cutedsl_version"):
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

    The JIT toolchain can write to descriptor 1 from native code, which would
    corrupt the two-line CSV contract. Redirecting at the descriptor level -
    rather than only rebinding ``sys.stdout`` - covers native writes too.
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


# --- Measurement -------------------------------------------------------------


def execute_measurement(warmup_iterations: int, iterations: int) -> str:
    """Run the frozen sequence once and return the CSV text on success.

    Returning the text rather than writing it keeps the single emission point
    in ``main`` and makes it structurally impossible to emit a row from any
    path that did not complete this whole sequence.
    """
    contract = load_pinned_contract()

    # The pinned upstream identity is proved before anything heavy is imported
    # and certainly before the upstream module itself is loaded.
    upstream = verify_upstream_source(contract)
    log(f"upstream verified: commit {upstream['commit']} sha256 {upstream['sha256']}")

    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    import torch

    # (6.1) Environment and provenance, before any tensor exists. The IEEE FP32
    # API is required up front so a PyTorch that cannot guarantee a trustworthy
    # correctness verdict fails closed before a JIT compilation is spent on it;
    # the setting itself is established and verified around the reference.
    log("collecting environment and provenance")
    require_ieee_fp32_matmul_api(torch)
    provenance = collect_provenance(contract, torch, cutlass)

    # (6.1.4) Revalidate the checkout and file identity, and require the two
    # independent observations to agree.
    revalidated = verify_upstream_source(contract)
    if revalidated != upstream:
        raise P32Error("the pinned upstream source changed during provenance collection")
    log(
        f"device: {provenance['gpu_name']} uuid={provenance['gpu_uuid']} "
        f"cc={provenance['compute_capability']} driver={provenance['driver_version']}"
    )

    module = load_upstream_module(upstream["path"])

    # The frozen seed is applied here and re-applied by the pinned upstream
    # helper; this guard fails closed if the pinned helper's own seed ever
    # stopped matching the value this wrapper reports.
    _assert_upstream_seed(module)

    # (6.2) Tensor preparation, entirely outside every timer.
    log("allocating tensors (outside every timer)")
    torch.manual_seed(FROZEN_SEED)
    ab_dtype = getattr(cutlass, FROZEN_AB_DTYPE)
    acc_dtype = getattr(cutlass, FROZEN_ACC_DTYPE)
    c_dtype = getattr(cutlass, FROZEN_C_DTYPE)

    (
        a_tensor,
        b_tensor,
        c_tensor,
        a_torch_cpu,
        b_torch_cpu,
        _c_torch_cpu,
        c_torch_gpu,
    ) = module.create_tensors(
        FROZEN_L,
        FROZEN_M,
        FROZEN_N,
        FROZEN_K,
        FROZEN_A_MAJOR,
        FROZEN_B_MAJOR,
        FROZEN_C_MAJOR,
        ab_dtype,
        c_dtype,
    )

    torch_stream = torch.cuda.current_stream()
    cute_stream = cuda_driver.CUstream(torch_stream.cuda_stream)

    gemm = module.DenseGemmKernel(
        acc_dtype,
        FROZEN_USE_2CTA_INSTRS,
        FROZEN_MMA_TILER_MN,
        FROZEN_CLUSTER_SHAPE_MN,
        FROZEN_USE_TMA_STORE,
    )
    if not gemm.can_implement(a_tensor, b_tensor, c_tensor):
        raise P32Error(
            "the pinned kernel cannot implement the frozen P3.2 configuration; "
            "P3.2 never falls back to another configuration"
        )
    log("can_implement: OK for the frozen configuration")

    # (6.3) Compilation only.
    log("compiling (JIT)")
    torch.cuda.synchronize()
    compile_start = time.perf_counter_ns()
    compiled_gemm = cute.compile(gemm, a_tensor, b_tensor, c_tensor, cute_stream)
    torch.cuda.synchronize()
    compile_time_ms = (time.perf_counter_ns() - compile_start) / 1e6

    # (6.4) First launch; its output is the tensor that gets validated.
    log("first launch (also the correctness-validated launch)")
    torch.cuda.synchronize()
    first_launch_start = time.perf_counter_ns()
    compiled_gemm(a_tensor, b_tensor, c_tensor, cute_stream)
    torch.cuda.synchronize()
    first_launch_ms = (time.perf_counter_ns() - first_launch_start) / 1e6

    max_abs_error, max_rel_error = validate_result(
        torch, a_torch_cpu, b_torch_cpu, c_torch_gpu
    )
    log(
        f"correctness: {CORRECTNESS_PASS} "
        f"(max_abs_error={max_abs_error!r} max_rel_error={max_rel_error!r})"
    )

    # (6.5) Warm-up and steady state, only after correctness passed.
    log(f"warm-up: {warmup_iterations} launch(es)")
    for _ in range(warmup_iterations):
        compiled_gemm(a_tensor, b_tensor, c_tensor, cute_stream)
    torch.cuda.synchronize()

    log(f"steady state: {iterations} measured launch(es)")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(torch_stream)
    for _ in range(iterations):
        compiled_gemm(a_tensor, b_tensor, c_tensor, cute_stream)
    end_event.record(torch_stream)
    torch.cuda.synchronize()
    total_ms = start_event.elapsed_time(end_event)
    if not math.isfinite(total_ms) or total_ms <= 0.0:
        raise P32Error(f"CUDA-event elapsed time {total_ms!r} is not finite and positive")
    kernel_time_ms = total_ms / iterations

    for name, value in (
        ("compile_time_ms", compile_time_ms),
        ("first_launch_ms", first_launch_ms),
        ("kernel_time_ms", kernel_time_ms),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise P32Error(f"{name}={value!r} is not finite and strictly positive")

    row = build_row(
        correctness=CORRECTNESS_PASS,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        compile_time_ms=compile_time_ms,
        first_launch_ms=first_launch_ms,
        kernel_time_ms=kernel_time_ms,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        provenance=provenance,
        upstream=upstream,
    )
    return serialize_row(row)


def _assert_upstream_seed(module) -> None:
    """Confirm the pinned upstream tensor factory still uses the frozen seed."""
    import inspect

    try:
        source = inspect.getsource(module.create_tensors)
    except (OSError, TypeError) as exc:
        raise P32Error(f"cannot read the pinned upstream tensor factory: {exc}") from exc
    if f"manual_seed({FROZEN_SEED})" not in source:
        raise P32Error(
            f"the pinned upstream tensor factory does not seed with {FROZEN_SEED}; "
            "the frozen seed cannot be reported"
        )


def validate_result(torch, a_torch_cpu, b_torch_cpu, c_torch_gpu):
    """Validate the complete result against an untimed IEEE-FP32 CUDA oracle.

    The oracle is the pinned PyTorch installation used purely as a correctness
    reference. It is never timed, never compared against, and is not a cuBLASLt
    baseline: P3.3 owns that and does not exist yet.

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
    result = c_torch_gpu.to(dtype=torch.float32)

    if tuple(result.shape) != tuple(reference.shape):
        raise CorrectnessError(
            f"result shape {tuple(result.shape)} != reference shape {tuple(reference.shape)}"
        )
    if not bool(torch.isfinite(result).all()):
        raise CorrectnessError("the kernel result contains non-finite values")
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
    """Build the whole P3.2 command line.

    The frozen GEMM configuration is deliberately unreachable from here: there
    is no shape, dtype, layout, tiler, cluster, TMA, scheduling, or MMA-group
    option, and no way to skip the reference check.
    """
    parser = argparse.ArgumentParser(
        prog="cutedsl_gemm.py",
        description=(
            "P3.2 one-shape CuTe DSL BF16 GEMM wrapper. Executes exactly one frozen "
            "configuration around the pinned, unmodified official NVIDIA example and "
            "emits one non-publishable CSV row of functional evidence. Not an "
            "experimental campaign and not a performance result."
        ),
        epilog=(
            "Correctness is mandatory and always runs before any warm-up or steady-state "
            "timing. The emitted timings are P3.2 infrastructure evidence only."
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


def _synthetic_row() -> dict:
    return build_row(
        correctness=CORRECTNESS_PASS,
        max_abs_error=0.0,
        max_rel_error=0.0,
        compile_time_ms=1234.5,
        first_launch_ms=12.25,
        kernel_time_ms=7.5,
        warmup_iterations=2,
        iterations=10,
        provenance=_synthetic_provenance(),
        upstream=_synthetic_upstream(),
    )


def run_self_test() -> int:
    """Prove the GPU-free half of the P3.2 contract, printing only to stderr."""
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
        except RowContractError as exc:
            check(name, expected_fragment in str(exc), f"message was {str(exc)!r}")
        except P32Error as exc:
            check(name, expected_fragment in str(exc), f"message was {str(exc)!r}")
        else:
            check(name, False, "no error was raised")

    print("cutedsl_gemm --self-test (GPU-free)", file=sys.stderr)

    # Frozen schema.
    check("schema has 47 fields", len(CSV_FIELDS) == 47, str(len(CSV_FIELDS)))
    check("schema has no duplicate field", len(set(CSV_FIELDS)) == len(CSV_FIELDS))
    check("schema starts with schema_version", CSV_FIELDS[0] == "schema_version")
    check("schema ends with publishable", CSV_FIELDS[-1] == "publishable")
    check(
        "no performance metric is in the schema",
        not any(
            re.search(r"tflop|speedup|efficien|bandwidth|utilization|throughput", field)
            for field in CSV_FIELDS
        ),
    )

    # Frozen configuration.
    check("problem is (4096, 4096, 4096, 1)", FROZEN_MNKL == (4096, 4096, 4096, 1))
    check("MMA tiler is (128, 128)", FROZEN_MMA_TILER_MN == (128, 128))
    check("cluster is (1, 1)", FROZEN_CLUSTER_SHAPE_MN == (1, 1))
    check("MMA group is one CTA", FROZEN_USE_2CTA_INSTRS is False)
    check("TMA store is used", FROZEN_USE_TMA_STORE is True)
    check("variant is non-persistent, 1 CTA", VARIANT == "nonpersistent_1cta")
    check("seed is 1111", FROZEN_SEED == 1111)
    check("tolerances are 1e-1 / 1e-5", (FROZEN_ATOL, FROZEN_RTOL) == (1e-1, 1e-5))
    check("cache model is hot", CACHE_MODE == "hot")
    check("publishable is fixed to false", PUBLISHABLE == BOOL_FALSE)
    check("run_kind is smoke", RUN_KIND == "smoke")
    check("reference is the untimed torch CUDA FP32 oracle", REFERENCE == "torch_cuda_fp32_ieee")

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
    check("2-CTA is recorded as false", row["use_2cta_instrs"] == BOOL_FALSE)
    check("the row is not publishable", row["publishable"] == BOOL_FALSE)

    # Rejections.
    rejects(
        "a failed correctness check cannot build a row",
        lambda: build_row(
            correctness="FAIL",
            max_abs_error=0.0,
            max_rel_error=0.0,
            compile_time_ms=1.0,
            first_launch_ms=1.0,
            kernel_time_ms=1.0,
            warmup_iterations=2,
            iterations=10,
            provenance=_synthetic_provenance(),
            upstream=_synthetic_upstream(),
        ),
        "refusing to build a row",
    )
    rejects(
        "a skipped correctness check cannot build a row",
        lambda: build_row(
            correctness="SKIPPED",
            max_abs_error=0.0,
            max_rel_error=0.0,
            compile_time_ms=1.0,
            first_launch_ms=1.0,
            kernel_time_ms=1.0,
            warmup_iterations=2,
            iterations=10,
            provenance=_synthetic_provenance(),
            upstream=_synthetic_upstream(),
        ),
        "refusing to build a row",
    )
    rejects(
        "a missing field is rejected",
        lambda: validate_row({key: value for key, value in row.items() if key != "kernel_time_ms"}),
        "missing field",
    )
    rejects(
        "an unknown field is rejected",
        lambda: validate_row({**row, "tflops": "1.0"}),
        "unknown field",
    )
    rejects(
        "a NaN timing is rejected",
        lambda: validate_row({**row, "kernel_time_ms": "nan"}),
        "kernel_time_ms",
    )
    rejects(
        "an infinite timing is rejected",
        lambda: validate_row({**row, "kernel_time_ms": "inf"}),
        "kernel_time_ms",
    )
    rejects(
        "a non-finite float cannot be serialized",
        lambda: format_fixed(float("nan"), DECIMALS_TIMING),
        "not finite",
    )
    rejects(
        "a zero kernel time is rejected",
        lambda: validate_row({**row, "kernel_time_ms": "0.000000"}),
        "strictly positive",
    )
    rejects(
        "a negative timing is rejected",
        lambda: validate_row({**row, "compile_time_ms": "-1.000000"}),
        "compile_time_ms",
    )
    rejects(
        "a wrongly typed count is rejected",
        lambda: validate_row({**row, "iterations": "ten"}),
        "iterations",
    )
    rejects(
        "a non-string value is rejected",
        lambda: validate_row({**row, "iterations": 10}),
        "not a string",
    )
    rejects(
        "publishable=true is rejected",
        lambda: validate_row({**row, "publishable": BOOL_TRUE}),
        "publishable",
    )
    rejects(
        "correctness=FAIL is rejected",
        lambda: validate_row({**row, "correctness": "FAIL"}),
        "correctness",
    )
    rejects(
        "a changed problem size is rejected",
        lambda: validate_row({**row, "m": "8192"}),
        "frozen",
    )
    rejects(
        "a 2-CTA row is rejected",
        lambda: validate_row({**row, "use_2cta_instrs": BOOL_TRUE}),
        "use_2cta_instrs",
    )
    rejects(
        "a non-canonical boolean is rejected",
        lambda: validate_row({**row, "git_dirty": "TRUE"}),
        "git_dirty",
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
    # real pinned value read from the global contract. The checker owns the
    # independent expectation about which target must be pinned.
    check("sm_75 maps to compute capability 7.5", compute_capability_for_arch("sm_75") == "7.5")
    check("sm_90 maps to compute capability 9.0", compute_capability_for_arch("sm_90") == "9.0")
    check("sm_100 maps to compute capability 10.0", compute_capability_for_arch("sm_100") == "10.0")
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

    # No heavy import happened.
    for module_name in ("torch", "cutlass", "cuda"):
        check(f"{module_name} was not imported", module_name not in sys.modules)

    if failures:
        print(f"SELF-TEST: FAIL ({len(failures)} case(s))", file=sys.stderr)
        return 1
    print("SELF-TEST: PASS", file=sys.stderr)
    return 0


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
    except P32Error as exc:
        os.close(saved_stdout_fd)
        log(f"FAIL: {exc}")
        return 1
    except BaseException:
        os.close(saved_stdout_fd)
        raise

    # Only a fully completed sequence, with correctness already passed, reaches
    # this line, and this is the only place a CSV row is ever written.
    _emit_on_saved_stdout(saved_stdout_fd, csv_text)
    log("emitted one non-publishable P3.2 row (functional evidence, not a result)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
