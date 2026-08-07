#!/usr/bin/env python3
"""P3.4 - three CuTe DSL execution variants at one frozen shape (no claim).

P3.2 established one CuTe DSL execution variant at the first final shape and
P3.3 established the cuBLASLt baseline for exactly the same geometry. P3.4 adds
the two remaining execution variants the project plan froze, so that all three
exist under one identical operand set, one identical correctness oracle, and
one identical timing discipline:

    nonpersistent_1cta   DenseGemmKernel,           (128,128) tiler, (1,1) cluster
    persistent_1cta      PersistentDenseGemmKernel, (128,128) tiler, (1,1) cluster
    persistent_2cta      PersistentDenseGemmKernel, (256,128) tiler, (2,1) cluster

The 2-CTA row deliberately uses an M tile of 256 so that each of the two
participating CTAs keeps a local M extent of 128 - the same two-SM geometry
P2.2 measured, and the shape NVIDIA's own persistent example documents for
``use_2cta_instrs=True``. No other tiler or cluster is ever substituted.

This repository owns no GEMM kernel. Both kernels come from the two pinned,
unmodified official NVIDIA examples in the pinned ``/opt/cutlass`` checkout,
loaded read-only and in place after their commit, Git blob SHA, and SHA-256
have been verified against the repository's two version contracts. Neither
upstream ``run()`` is ever called, and neither upstream benchmarking helper is
ever used: P3.4 owns every timer.

What this program measures, per variant, in this exact order:

1. environment, repository, and both upstream source identities;
2. one shared operand set, allocated once, entirely outside every timer;
3. the frozen variant object and its own official ``can_implement()`` check;
4. for a persistent variant, ``max_active_clusters`` from the official pinned
   hardware helper - never guessed, hard-coded, or overridable;
5. C reset to a sentinel and synchronized, outside every timer;
6. ``compile_time_ms``  - a monotonic host clock around ``cute.compile`` only;
7. ``first_launch_ms``  - a monotonic host clock around the first launch of the
   compiled kernel, whose output is also the tensor validated for correctness;
8. complete FP32 correctness validation against an untimed PyTorch CUDA oracle
   with TF32 (and every other reduced-precision FP32 matmul mode) disabled;
9. only if that variant's correctness passes: warm-up launches, then
   ``kernel_time_ms`` from CUDA events on the same stream, divided by the
   measured iteration count.

All three variants consume byte-identical A and B: the operands are built once
by the pinned non-persistent example's own ``create_tensors`` - the same
factory, seed, and call order P3.2 and P3.3 use - and are never mutated. Only C
is reset between variants, and it is reset to NaN so that any element a kernel
fails to write is a non-finite value that the complete-result check rejects
rather than a stale value that silently passes.

What this program is not: it is not an experimental campaign, not a comparison,
and not a performance result. Every emitted row carries ``publishable=false``.
No TFLOP/s, speedup, efficiency, utilization, bandwidth, ranking, or winner is
computed anywhere; the three timings are P3.4 functional-verification evidence
only. Comparing these variants against each other or against the P3.3 cuBLASLt
baseline is P3.5's job, and P3.5 does not exist.

Output contract:

* stdout receives exactly one CSV header line and exactly three CSV data rows,
  in the frozen variant order, and nothing else. The whole output is buffered
  and emitted only after all three variants have passed, so a failure in any
  position - including the third - produces no CSV at all rather than a
  truncated table. To make that true even when the JIT toolchain writes to file
  descriptor 1 from native code, descriptor 1 is redirected to descriptor 2 for
  the whole measurement and the real stdout is restored only to emit the four
  lines.
* stderr receives every human-readable message: progress, warnings, compiler
  output, and diagnostics.
* Any failure exits non-zero, prints a diagnostic to stderr, and emits no CSV
  header and no CSV row; the failing variant runs no warm-up and no
  steady-state timing.

Usage:
  cutedsl_variants.py [--warmup-iterations N] [--iterations N]
  cutedsl_variants.py --self-test
  cutedsl_variants.py --help

``--help`` and ``--self-test`` are GPU-free and import neither PyTorch, nor
CuTe DSL, nor the CUDA bindings, nor either upstream example: every heavy
import is deferred into the measurement path.

Exit code: 0 only when all three variants succeeded and four valid lines were
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

SCHEMA_VERSION = "p34.v1"
EXPERIMENT = "exp03_cutedsl_vs_cublaslt"
UNIT = "P3.4"
RUN_KIND = "smoke"
METHOD = "cutedsl"
REFERENCE = "torch_cuda_fp32_ieee"
CACHE_MODE = "hot"
CORRECTNESS_PASS = "PASS"
PUBLISHABLE = "false"

# --- Frozen GEMM configuration ----------------------------------------------
#
# P3.4 executes exactly one shape - the first of the five final shapes, the same
# one P3.2 and P3.3 use. None of these values is reachable from the command
# line, from an environment variable, or from a configuration file. The
# remaining four shapes and every comparison belong to P3.5.

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

FROZEN_USE_TMA_STORE = True

FROZEN_SEED = 1111
FROZEN_ATOL = 1e-1
FROZEN_RTOL = 1e-5

# --- Frozen variant table ----------------------------------------------------
#
# Exactly three candidates, one per execution variant, always executed in this
# order. This is not a search space: there is no autotuning, no ranking, and no
# fourth candidate.

VARIANT_NONPERSISTENT_1CTA = "nonpersistent_1cta"
VARIANT_PERSISTENT_1CTA = "persistent_1cta"
VARIANT_PERSISTENT_2CTA = "persistent_2cta"

SCHEDULER_NONPERSISTENT = "nonpersistent"
SCHEDULER_STATIC_PERSISTENT = "static_persistent"

# Which pinned upstream source owns each variant's kernel.
SOURCE_NONPERSISTENT = "nonpersistent"
SOURCE_PERSISTENT = "persistent"

UPSTREAM_CLASS_NONPERSISTENT = "DenseGemmKernel"
UPSTREAM_CLASS_PERSISTENT = "PersistentDenseGemmKernel"

# The canonical string recorded for a variant that has no cluster scheduler and
# therefore no max_active_clusters value at all. It is never a number, never
# zero, and never an empty field.
MAX_ACTIVE_CLUSTERS_NOT_APPLICABLE = "not_applicable"

FROZEN_VARIANTS = (
    {
        "variant": VARIANT_NONPERSISTENT_1CTA,
        "scheduler": SCHEDULER_NONPERSISTENT,
        "source": SOURCE_NONPERSISTENT,
        "upstream_class": UPSTREAM_CLASS_NONPERSISTENT,
        "mma_tiler_mn": (128, 128),
        "cluster_shape_mn": (1, 1),
        "use_2cta_instrs": False,
        "persistent": False,
    },
    {
        "variant": VARIANT_PERSISTENT_1CTA,
        "scheduler": SCHEDULER_STATIC_PERSISTENT,
        "source": SOURCE_PERSISTENT,
        "upstream_class": UPSTREAM_CLASS_PERSISTENT,
        "mma_tiler_mn": (128, 128),
        "cluster_shape_mn": (1, 1),
        "use_2cta_instrs": False,
        "persistent": True,
    },
    {
        # M tile 256 with a 2-CTA cluster keeps the per-CTA M extent at 128,
        # matching P2.2's two-SM geometry and NVIDIA's own documented 2-CTA
        # constraint (tiler M must be 128 or 256 when use_2cta_instrs=True, and
        # cluster M must be a multiple of 2).
        "variant": VARIANT_PERSISTENT_2CTA,
        "scheduler": SCHEDULER_STATIC_PERSISTENT,
        "source": SOURCE_PERSISTENT,
        "upstream_class": UPSTREAM_CLASS_PERSISTENT,
        "mma_tiler_mn": (256, 128),
        "cluster_shape_mn": (2, 1),
        "use_2cta_instrs": True,
        "persistent": True,
    },
)

FROZEN_VARIANT_ORDER = tuple(spec["variant"] for spec in FROZEN_VARIANTS)

# The only CUDA matmul FP32 policy P3.4 accepts for its correctness oracle,
# via the PyTorch 2.10 fp32_precision API and nothing else. The unset default
# is "none", which proves nothing and is rejected.
FP32_PRECISION_IEEE = "ieee"

# Safe denominator for the reported relative error, identical to P3.2/P3.3. The
# reported max_rel_error is max(|c - ref| / max(|ref|, floor)), which stays
# finite where the reference is exactly zero. It is a diagnostic only: the
# pass/fail decision uses the elementwise criterion |c - ref| <= atol +
# rtol * |ref| at full precision, never this scalar.
REL_ERROR_DENOMINATOR_FLOOR = 1.0

FROZEN_CONFIG = {
    "schema_version": SCHEMA_VERSION,
    "experiment": EXPERIMENT,
    "unit": UNIT,
    "run_kind": RUN_KIND,
    "method": METHOD,
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

# Deterministic decimal formats. Every real-valued field is serialized as a
# plain fixed-point decimal with exactly this many fractional digits: no
# exponent, no locale dependence, no shortest-round-trip ambiguity. Every
# decision (correctness, positivity, finiteness) is taken on the full-precision
# value before serialization.
DECIMALS_TIMING = 6  # milliseconds, i.e. nanosecond resolution
DECIMALS_ERROR = 9
DECIMALS_TOLERANCE = 9

# Values identical in all three rows.
CSV_FIXED_VALUES = {
    "schema_version": SCHEMA_VERSION,
    "experiment": EXPERIMENT,
    "unit": UNIT,
    "run_kind": RUN_KIND,
    "method": METHOD,
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
CSV_POSITIVE_INT_FIELDS = ("mma_tiler_m", "mma_tiler_n", "cluster_m", "cluster_n")

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
# A repository-relative upstream path: must start with an alphanumeric (so an
# absolute path is rejected) and must contain no ".." segment.
_RE_UPSTREAM_REL_PATH = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._/-]*\.py\Z")


def is_relative_upstream_path(path: str) -> bool:
    """True only for a safe, repository-relative upstream ``.py`` path."""
    if not isinstance(path, str) or not _RE_UPSTREAM_REL_PATH.match(path):
        return False
    return ".." not in Path(path).parts

# The pinned CUTLASS checkout inside the pinned image. The image builds it at
# exactly CUTLASS_COMMIT (see Dockerfile); nothing is ever written to it, and
# this location is not configurable at runtime.
UPSTREAM_CHECKOUT_DIR = Path("/opt/cutlass")

GLOBAL_CONTRACT_FILE = "VERSIONS.env"
PHASE3_CONTRACT_FILE = "PHASE3_VERSIONS.env"

# Keys read from the two version contracts. Nothing below is duplicated as a
# literal anywhere in this file: the pinned commit, blobs, SHA-256 digests,
# versions, and architecture exist here only as key names. P3.4 adds no key to
# VERSIONS.env; the three CUTEDSL_P34_* keys live in the Phase 3 contract.
GLOBAL_CONTRACT_KEYS = ("CUDA_VERSION", "CUTLASS_VERSION", "CUTLASS_COMMIT", "CUDA_ARCH")
PHASE3_CONTRACT_KEYS = (
    "PYTORCH_VERSION",
    "PYTORCH_CUDA_VERSION",
    "CUTEDSL_P31_EXAMPLE_PATH",
    "CUTEDSL_P31_EXAMPLE_GIT_BLOB",
    "CUTEDSL_P31_EXAMPLE_SHA256",
    "CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH",
    "CUTEDSL_P34_PERSISTENT_EXAMPLE_GIT_BLOB",
    "CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256",
)

# The two pinned upstream sources, keyed by the source name each variant names.
UPSTREAM_SOURCES = {
    SOURCE_NONPERSISTENT: {
        "path_key": "CUTEDSL_P31_EXAMPLE_PATH",
        "blob_key": "CUTEDSL_P31_EXAMPLE_GIT_BLOB",
        "sha256_key": "CUTEDSL_P31_EXAMPLE_SHA256",
        "kernel_class": UPSTREAM_CLASS_NONPERSISTENT,
        "module_name": "p34_pinned_upstream_dense_gemm",
    },
    SOURCE_PERSISTENT: {
        "path_key": "CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH",
        "blob_key": "CUTEDSL_P34_PERSISTENT_EXAMPLE_GIT_BLOB",
        "sha256_key": "CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256",
        "kernel_class": UPSTREAM_CLASS_PERSISTENT,
        "module_name": "p34_pinned_upstream_dense_gemm_persistent",
    },
}

# The operand factory lives in the non-persistent example only. P3.4
# deliberately never uses the persistent example's own tensor-generation path,
# because that would break byte-for-byte operand equivalence with P3.2 and P3.3.
OPERAND_FACTORY_SOURCE = SOURCE_NONPERSISTENT
OPERAND_FACTORY_NAME = "create_tensors"


class P34Error(Exception):
    """Any fail-closed P3.4 contract, provenance, or execution failure."""


class RowContractError(P34Error):
    """A CSV row violated the frozen P3.4 schema."""


class CorrectnessError(P34Error):
    """A variant's complete result did not match the untimed FP32 reference."""


def log(message: str) -> None:
    """Write one human-readable progress/diagnostic line to stderr."""
    print(f"cutedsl_variants: {message}", file=sys.stderr, flush=True)


# --- Version contracts -------------------------------------------------------


def repository_root() -> Path:
    """Locate the repository root that owns this file.

    ``src/gemm/cutedsl_variants.py`` is two directories below the root both on
    the host and inside the container, where the repository is mounted at
    ``/workspace``.
    """
    root = Path(__file__).resolve().parents[2]
    for name in (GLOBAL_CONTRACT_FILE, PHASE3_CONTRACT_FILE):
        if not (root / name).is_file():
            raise P34Error(f"repository root {root} does not contain {name}")
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
        raise P34Error(f"cannot read version contract {path}: {exc}") from exc

    values: dict = {}
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _RE_ENV_LINE.match(line)
        if match is None:
            raise P34Error(f"{path}:{lineno}: malformed contract line {raw!r}")
        key, value = match.group(1), match.group(2)
        if key in values:
            raise P34Error(f"{path}:{lineno}: duplicate contract key {key}")
        values[key] = value
    return values


def load_pinned_contract(repo_root=None) -> dict:
    """Read every pinned value P3.4 needs from the two version contracts.

    ``VERSIONS.env`` is the closed global contract that the audited P1/P2
    aggregators parse against their own closed key allowlist; P3.4 only reads
    it and adds nothing to it. ``PHASE3_VERSIONS.env`` is the Phase 3-only
    extension, which P3.4 grows by exactly the three ``CUTEDSL_P34_*`` keys
    identifying the second official example.
    """
    root = Path(repo_root) if repo_root is not None else repository_root()
    global_values = parse_env_file(root / GLOBAL_CONTRACT_FILE)
    phase3_values = parse_env_file(root / PHASE3_CONTRACT_FILE)

    contract = {}
    for key in GLOBAL_CONTRACT_KEYS:
        if key not in global_values:
            raise P34Error(f"{GLOBAL_CONTRACT_FILE} is missing required key {key}")
        contract[key] = global_values[key]
    for key in PHASE3_CONTRACT_KEYS:
        if key not in phase3_values:
            raise P34Error(f"{PHASE3_CONTRACT_FILE} is missing required key {key}")
        contract[key] = phase3_values[key]

    if not _RE_HEX40.match(contract["CUTLASS_COMMIT"]):
        raise P34Error(f"pinned CUTLASS_COMMIT is malformed: {contract['CUTLASS_COMMIT']!r}")
    for source in UPSTREAM_SOURCES.values():
        blob = contract[source["blob_key"]]
        sha256 = contract[source["sha256_key"]]
        path = contract[source["path_key"]]
        if not _RE_HEX40.match(blob):
            raise P34Error(f"pinned {source['blob_key']} is malformed")
        if not _RE_HEX64.match(sha256):
            raise P34Error(f"pinned {source['sha256_key']} is malformed")
        if not is_relative_upstream_path(path):
            raise P34Error(f"pinned {source['path_key']} is unsafe: {path!r}")

    # The two sources must be genuinely different files; a contract that
    # accidentally pointed both variants at one example would silently destroy
    # the whole point of this unit.
    non_persistent_path = contract[UPSTREAM_SOURCES[SOURCE_NONPERSISTENT]["path_key"]]
    persistent_path = contract[UPSTREAM_SOURCES[SOURCE_PERSISTENT]["path_key"]]
    if non_persistent_path == persistent_path:
        raise P34Error(
            "the pinned non-persistent and persistent examples are the same file; "
            "P3.4 requires two distinct official sources"
        )

    for key in ("CUDA_VERSION", "PYTORCH_CUDA_VERSION"):
        if not _RE_DOTTED_VERSION.match(contract[key]):
            raise P34Error(f"pinned {key} is malformed: {contract[key]!r}")
    if not contract["CUTLASS_VERSION"].startswith("v"):
        raise P34Error(f"pinned CUTLASS_VERSION is malformed: {contract['CUTLASS_VERSION']!r}")

    # Derived, never separately pinned.
    contract["CUTEDSL_VERSION"] = contract["CUTLASS_VERSION"][1:]
    if not _RE_DOTTED_VERSION.match(contract["CUTEDSL_VERSION"]):
        raise P34Error("pinned CuTe DSL version is malformed")

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
        raise P34Error(f"pinned CUDA_ARCH is malformed: {cuda_arch!r}")
    digits = match.group(1)
    if len(digits) < 2:
        raise P34Error(f"pinned CUDA_ARCH is malformed: {cuda_arch!r}")
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
        raise P34Error(f"git {' '.join(args)} could not be executed: {exc}") from exc
    if completed.returncode != 0:
        raise P34Error(
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
        raise P34Error(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def verify_upstream_sources(contract: dict) -> dict:
    """Prove both pinned upstream examples are byte-identical to their pins.

    Fails closed on a missing checkout, a wrong HEAD, any tracked or untracked
    modification, a symlinked or non-regular example file, a wrong Git blob
    SHA, or a wrong SHA-256 - for either file. The checkout is only ever
    queried, never written.
    """
    checkout = UPSTREAM_CHECKOUT_DIR
    if not checkout.is_dir():
        raise P34Error(f"pinned CUTLASS checkout {checkout} is missing")

    head = _git(["-C", str(checkout), "rev-parse", "HEAD"], safe_directory=str(checkout))
    if head != contract["CUTLASS_COMMIT"]:
        raise P34Error(
            f"{checkout} HEAD {head} != pinned CUTLASS_COMMIT {contract['CUTLASS_COMMIT']}"
        )

    dirty = _git(
        ["-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        safe_directory=str(checkout),
    )
    if dirty:
        raise P34Error(f"{checkout} has tracked or untracked modifications")

    sources = {}
    for name, source in sorted(UPSTREAM_SOURCES.items()):
        relative = contract[source["path_key"]]
        example = checkout / relative
        if example.is_symlink():
            raise P34Error(f"{example} is a symlink")
        if not example.is_file():
            raise P34Error(f"{example} is not a regular file")

        blob = _git(
            ["-C", str(checkout), "hash-object", "--", str(example)],
            safe_directory=str(checkout),
        )
        if blob != contract[source["blob_key"]]:
            raise P34Error(
                f"{relative} Git blob {blob} != pinned {contract[source['blob_key']]}"
            )

        sha256 = sha256_of_file(example)
        if sha256 != contract[source["sha256_key"]]:
            raise P34Error(
                f"{relative} SHA-256 {sha256} != pinned {contract[source['sha256_key']]}"
            )

        sources[name] = {
            "commit": head,
            "relative_path": relative,
            "path": example,
            "blob": blob,
            "sha256": sha256,
        }
    return sources


def load_upstream_module(example: Path, module_name: str):
    """Import a verified upstream example as a library, never as a script.

    The module is loaded under its own private name so the upstream
    ``if __name__ == "__main__"`` block - which parses arguments and calls
    ``run()`` - never executes. The file is read from ``/opt/cutlass`` and is
    neither copied, vendored, reformatted, nor patched.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, str(example))
    if spec is None or spec.loader is None:
        raise P34Error(f"cannot build an import spec for {example}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - fail closed with the real cause
        sys.modules.pop(module_name, None)
        raise P34Error(f"cannot import the pinned upstream example {example}: {exc}") from exc
    return module


def load_upstream_modules(sources: dict) -> dict:
    """Import both verified examples and prove each provides what P3.4 needs."""
    modules = {}
    for name, source in sorted(UPSTREAM_SOURCES.items()):
        module = load_upstream_module(sources[name]["path"], source["module_name"])
        kernel_class = source["kernel_class"]
        if not hasattr(module, kernel_class):
            raise P34Error(
                f"the pinned {name} example does not provide {kernel_class}; P3.4 never "
                "substitutes another kernel class"
            )
        modules[name] = module

    factory_module = modules[OPERAND_FACTORY_SOURCE]
    if not hasattr(factory_module, OPERAND_FACTORY_NAME):
        raise P34Error(
            f"the pinned {OPERAND_FACTORY_SOURCE} example does not provide "
            f"{OPERAND_FACTORY_NAME}"
        )

    # The two classes must be genuinely distinct objects; if the persistent
    # example ever re-exported the non-persistent class, the two persistent rows
    # would silently measure the non-persistent scheduler.
    non_persistent_class = getattr(modules[SOURCE_NONPERSISTENT], UPSTREAM_CLASS_NONPERSISTENT)
    persistent_class = getattr(modules[SOURCE_PERSISTENT], UPSTREAM_CLASS_PERSISTENT)
    if non_persistent_class is persistent_class:
        raise P34Error(
            "the persistent and non-persistent kernel classes are the same object; "
            "the two schedulers cannot be distinguished"
        )
    return modules


def _assert_upstream_seed(module) -> None:
    """Confirm the pinned upstream tensor factory still uses the frozen seed."""
    import inspect

    try:
        source = inspect.getsource(getattr(module, OPERAND_FACTORY_NAME))
    except (OSError, TypeError) as exc:
        raise P34Error(f"cannot read the pinned upstream tensor factory: {exc}") from exc
    if f"manual_seed({FROZEN_SEED})" not in source:
        raise P34Error(
            f"the pinned upstream tensor factory does not seed with {FROZEN_SEED}; "
            "the frozen seed cannot be reported"
        )


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
        raise P34Error(f"nvidia-smi could not be executed: {exc}") from exc
    if completed.returncode != 0:
        raise P34Error(
            f"nvidia-smi failed with exit code {completed.returncode}; "
            "device provenance is ambiguous"
        )

    rows = [row for row in csv.reader(io.StringIO(completed.stdout)) if row]
    if len(rows) != 1:
        raise P34Error(f"nvidia-smi reported {len(rows)} GPUs; exactly 1 must be visible")
    fields = [value.strip() for value in rows[0]]
    if len(fields) != 3:
        raise P34Error("nvidia-smi returned a malformed device row")

    uuid, name, driver_version = fields
    if not _RE_GPU_UUID.match(uuid):
        raise P34Error(f"nvidia-smi returned a malformed GPU UUID: {uuid!r}")
    if not name or not _RE_SAFE_TEXT.match(name):
        raise P34Error("nvidia-smi returned a malformed GPU name")
    if not _RE_DOTTED_VERSION.match(driver_version):
        raise P34Error(f"nvidia-smi returned a malformed driver version: {driver_version!r}")
    return {"gpu_uuid": uuid, "gpu_name": name, "driver_version": driver_version}


def _query_nvcc_major_minor() -> str:
    """Read the installed CUDA toolkit's ``release X.Y`` from nvcc."""
    try:
        completed = subprocess.run(
            ["nvcc", "--version"], capture_output=True, text=True, timeout=120, check=False
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise P34Error(f"nvcc could not be executed: {exc}") from exc
    if completed.returncode != 0:
        raise P34Error("nvcc --version failed; the CUDA toolkit version is ambiguous")
    match = re.search(r"release ([0-9]+)\.([0-9]+)", completed.stdout)
    if match is None:
        raise P34Error("nvcc --version did not report a release version")
    return f"{match.group(1)}.{match.group(2)}"


def _repository_git_state(root: Path) -> dict:
    """Record this repository's commit and dirty state."""
    commit = _git(["rev-parse", "HEAD"], cwd=root)
    if not _RE_HEX40.match(commit):
        raise P34Error(f"repository HEAD is malformed: {commit!r}")
    status = _git(["status", "--porcelain", "--untracked-files=all"], cwd=root)
    return {"git_commit": commit, "git_dirty": BOOL_TRUE if status else BOOL_FALSE}


def require_single_cuda_device(torch) -> None:
    """Require exactly one CUDA-visible GPU, used as logical device 0."""
    if not torch.cuda.is_available():
        raise P34Error("no CUDA device is available; P3.4 requires exactly one GPU")
    count = torch.cuda.device_count()
    if count != 1:
        raise P34Error(f"expected exactly 1 CUDA-visible GPU, saw {count}")
    torch.cuda.set_device(0)
    current = torch.cuda.current_device()
    if current != 0:
        raise P34Error(f"the selected CUDA device must be logical device 0, got {current}")


def require_ieee_fp32_matmul_api(torch):
    """Return the CUDA matmul backend, failing closed without the 2.10 API.

    P3.4 retains the closed P3.2/P3.3 policy unchanged and uses **exclusively**
    the PyTorch 2.10 ``fp32_precision`` API for CUDA matrix multiplication. The
    legacy ``allow_tf32`` property is never read and never written: in 2.10 the
    two are aliases of one setting, mixing them is unsupported, and the last
    write silently wins - setting ``allow_tf32`` after ``fp32_precision``
    rewrites the policy to ``tf32`` without any error.
    ``torch.set_float32_matmul_precision()`` is likewise never combined with it.
    """
    backends = getattr(torch, "backends", None)
    cuda_backend = getattr(backends, "cuda", None) if backends is not None else None
    matmul = getattr(cuda_backend, "matmul", None) if cuda_backend is not None else None
    if matmul is None:
        raise P34Error(
            "this PyTorch does not expose torch.backends.cuda.matmul; the IEEE FP32 "
            "reference cannot be guaranteed"
        )
    if not hasattr(matmul, "fp32_precision"):
        raise P34Error(
            "this PyTorch does not support torch.backends.cuda.matmul.fp32_precision; "
            "P3.4 requires that API and never falls back to the legacy TF32 flag"
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
        raise P34Error(
            f"torch.backends.cuda.matmul.fp32_precision={FP32_PRECISION_IEEE!r} was "
            f"rejected: {exc}"
        ) from exc

    effective = matmul.fp32_precision
    if effective != FP32_PRECISION_IEEE:
        _restore_fp32_precision(matmul, previous)
        raise P34Error(
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
        raise P34Error(f"malformed compute capability {compute_capability!r}")
    if compute_capability != contract["EXPECTED_COMPUTE_CAPABILITY"]:
        raise P34Error(
            f"device compute capability {compute_capability} does not match the pinned "
            f"{contract['CUDA_ARCH']} target ({contract['EXPECTED_COMPUTE_CAPABILITY']})"
        )

    nvcc_major_minor = _query_nvcc_major_minor()
    if nvcc_major_minor != contract["CUDA_MAJOR_MINOR"]:
        raise P34Error(
            f"installed CUDA toolkit {nvcc_major_minor} does not match the pinned "
            f"{contract['CUDA_VERSION']}"
        )

    torch_version = str(torch.__version__)
    if torch_version != contract["PYTORCH_VERSION"]:
        raise P34Error(f"torch {torch_version} != pinned {contract['PYTORCH_VERSION']}")
    torch_cuda_version = torch.version.cuda
    if torch_cuda_version != contract["PYTORCH_CUDA_VERSION"]:
        raise P34Error(
            f"torch CUDA {torch_cuda_version} != pinned {contract['PYTORCH_CUDA_VERSION']}"
        )

    cutedsl_version = str(cutlass.__version__)
    if cutedsl_version != contract["CUTEDSL_VERSION"]:
        raise P34Error(f"CuTe DSL {cutedsl_version} != pinned {contract['CUTEDSL_VERSION']}")

    git_state = _repository_git_state(repository_root())

    return {
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


# --- CSV rows ----------------------------------------------------------------


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


def frozen_variant_spec(variant: str) -> dict:
    """Return the one frozen specification for a variant name, or fail."""
    for spec in FROZEN_VARIANTS:
        if spec["variant"] == variant:
            return spec
    raise RowContractError(
        f"{variant!r} is not one of the three frozen P3.4 variants "
        f"{FROZEN_VARIANT_ORDER}"
    )


def build_row(
    variant: str,
    correctness: str,
    max_abs_error,
    max_rel_error,
    compile_time_ms,
    first_launch_ms,
    kernel_time_ms,
    warmup_iterations: int,
    iterations: int,
    max_active_clusters,
    provenance: dict,
    upstream: dict,
) -> dict:
    """Build one frozen CSV row, refusing anything but a passed check.

    This is the only way a row is constructed, so a failed or skipped
    correctness check cannot produce an emittable row. The variant's tiler,
    cluster, scheduler, 2-CTA flag, and source are taken from the frozen table
    rather than from the caller, so a row can never describe a configuration
    that was not the frozen one.
    """
    if correctness != CORRECTNESS_PASS:
        raise RowContractError(
            f"refusing to build a row with correctness={correctness!r}; "
            f"only {CORRECTNESS_PASS} may be emitted"
        )

    spec = frozen_variant_spec(variant)

    for name, value in (
        ("compile_time_ms", compile_time_ms),
        ("first_launch_ms", first_launch_ms),
        ("kernel_time_ms", kernel_time_ms),
    ):
        number = float(value)
        if not math.isfinite(number) or number <= 0.0:
            raise RowContractError(f"{name}={value!r} must be finite and strictly positive")

    # max_active_clusters is a positive decimal integer for a persistent
    # variant and the canonical not_applicable string for the non-persistent
    # one. Neither form is ever produced for the other kind of variant.
    if spec["persistent"]:
        if isinstance(max_active_clusters, bool) or not isinstance(max_active_clusters, int):
            raise RowContractError(
                f"{variant}: max_active_clusters must be an integer, got "
                f"{max_active_clusters!r}"
            )
        if max_active_clusters <= 0:
            raise RowContractError(
                f"{variant}: max_active_clusters={max_active_clusters} must be positive"
            )
        max_active_clusters_text = str(max_active_clusters)
    else:
        if max_active_clusters is not None:
            raise RowContractError(
                f"{variant}: a non-persistent variant has no max_active_clusters, got "
                f"{max_active_clusters!r}"
            )
        max_active_clusters_text = MAX_ACTIVE_CLUSTERS_NOT_APPLICABLE

    row = dict(CSV_FIXED_VALUES)
    row.update(
        {
            "variant": spec["variant"],
            "scheduler": spec["scheduler"],
            "mma_tiler_m": str(spec["mma_tiler_mn"][0]),
            "mma_tiler_n": str(spec["mma_tiler_mn"][1]),
            "cluster_m": str(spec["cluster_shape_mn"][0]),
            "cluster_n": str(spec["cluster_shape_mn"][1]),
            "use_2cta_instrs": BOOL_TRUE if spec["use_2cta_instrs"] else BOOL_FALSE,
            "max_active_clusters": max_active_clusters_text,
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
            "upstream_kernel_file": upstream["relative_path"],
            "upstream_kernel_git_blob": upstream["blob"],
            "upstream_kernel_sha256": upstream["sha256"],
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
            raise RowContractError(
                f"{field}: value {value!r} is empty or contains control characters"
            )

    for field, fixed in CSV_FIXED_VALUES.items():
        if row[field] != fixed:
            raise RowContractError(f"{field}: {row[field]!r} != frozen {fixed!r}")

    # The variant row must match the frozen table exactly: scheduler, tiler,
    # cluster, and 2-CTA flag are all decided by the variant name.
    spec = frozen_variant_spec(row["variant"])
    for field, expected_value in (
        ("scheduler", spec["scheduler"]),
        ("mma_tiler_m", str(spec["mma_tiler_mn"][0])),
        ("mma_tiler_n", str(spec["mma_tiler_mn"][1])),
        ("cluster_m", str(spec["cluster_shape_mn"][0])),
        ("cluster_n", str(spec["cluster_shape_mn"][1])),
        ("use_2cta_instrs", BOOL_TRUE if spec["use_2cta_instrs"] else BOOL_FALSE),
    ):
        if row[field] != expected_value:
            raise RowContractError(
                f"{row['variant']}: {field}={row[field]!r} != frozen {expected_value!r}"
            )

    if spec["persistent"]:
        if not _RE_POSITIVE_INT.match(row["max_active_clusters"]):
            raise RowContractError(
                f"{row['variant']}: max_active_clusters={row['max_active_clusters']!r} must "
                "be a positive decimal integer for a persistent variant"
            )
    elif row["max_active_clusters"] != MAX_ACTIVE_CLUSTERS_NOT_APPLICABLE:
        raise RowContractError(
            f"{row['variant']}: max_active_clusters={row['max_active_clusters']!r} must be "
            f"{MAX_ACTIVE_CLUSTERS_NOT_APPLICABLE!r} for the non-persistent variant"
        )

    for field in CSV_BOOL_FIELDS:
        if row[field] not in (BOOL_TRUE, BOOL_FALSE):
            raise RowContractError(
                f"{field}: {row[field]!r} is not a canonical lowercase boolean"
            )

    for field in CSV_POSITIVE_INT_FIELDS:
        if not _RE_POSITIVE_INT.match(row[field]):
            raise RowContractError(f"{field}: {row[field]!r} is not a positive integer")

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
        raise RowContractError(
            f"cutlass_commit: {row['cutlass_commit']!r} is not a 40-hex commit"
        )
    if not _RE_HEX40.match(row["git_commit"]):
        raise RowContractError(f"git_commit: {row['git_commit']!r} is not a 40-hex commit")
    if not _RE_HEX40.match(row["upstream_kernel_git_blob"]):
        raise RowContractError("upstream_kernel_git_blob is not a 40-hex blob")
    if not _RE_HEX64.match(row["upstream_kernel_sha256"]):
        raise RowContractError("upstream_kernel_sha256 is not a 64-hex digest")
    if not is_relative_upstream_path(row["upstream_kernel_file"]):
        raise RowContractError(
            f"upstream_kernel_file: {row['upstream_kernel_file']!r} is not a relative "
            "upstream .py path"
        )
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


def validate_rows(rows) -> None:
    """Require exactly the three frozen variants, once each, in fixed order."""
    if not isinstance(rows, (list, tuple)):
        raise RowContractError("the P3.4 result must be a sequence of rows")
    if len(rows) != len(FROZEN_VARIANTS):
        raise RowContractError(
            f"P3.4 emits exactly {len(FROZEN_VARIANTS)} rows, got {len(rows)}"
        )
    observed = tuple(row.get("variant") if isinstance(row, dict) else None for row in rows)
    if observed != FROZEN_VARIANT_ORDER:
        raise RowContractError(
            f"the variant order {observed} is not the frozen order {FROZEN_VARIANT_ORDER}"
        )
    for row in rows:
        validate_row(row)


def serialize_rows(rows) -> str:
    """Serialize all three validated rows with the csv module."""
    validate_rows(rows)
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(CSV_FIELDS),
        extrasaction="raise",
        lineterminator="\n",
        quoting=csv.QUOTE_MINIMAL,
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buffer.getvalue()


# --- stdout discipline -------------------------------------------------------


def _redirect_stdout_to_stderr() -> int:
    """Send everything written to descriptor 1 to stderr; return the real one.

    The JIT toolchain can write to descriptor 1 from native code, which would
    corrupt the four-line CSV contract. Redirecting at the descriptor level -
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


def _max_active_clusters(cutlass, spec: dict) -> int:
    """Query the official pinned hardware helper for a persistent variant.

    The value is never guessed, hard-coded, exposed as a CLI option, or read
    from an environment override: it comes from
    ``cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_size)``,
    exactly the helper the pinned persistent example itself uses, for this
    variant's own cluster size.
    """
    import cutlass.utils as utils

    cluster_m, cluster_n = spec["cluster_shape_mn"]
    cluster_size = cluster_m * cluster_n
    try:
        value = utils.HardwareInfo().get_max_active_clusters(cluster_size)
    except Exception as exc:  # noqa: BLE001 - fail closed with the real cause
        raise P34Error(
            f"{spec['variant']}: the official hardware helper could not report "
            f"max_active_clusters for cluster size {cluster_size}: {exc}"
        ) from exc

    if isinstance(value, bool) or not isinstance(value, int):
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise P34Error(
                f"{spec['variant']}: max_active_clusters={value!r} is not a number"
            ) from exc
        if not math.isfinite(numeric) or numeric != int(numeric):
            raise P34Error(
                f"{spec['variant']}: max_active_clusters={value!r} is not a finite integer"
            )
        value = int(numeric)
    if value <= 0:
        raise P34Error(
            f"{spec['variant']}: max_active_clusters={value} must be a positive integer"
        )
    return value


def _build_kernel(module, spec: dict, cutlass):
    """Instantiate the frozen kernel object for one variant.

    Both official classes take the same five constructor arguments; which class
    is used is fixed by the frozen table, never inferred and never substituted.
    """
    kernel_class = getattr(module, spec["upstream_class"])
    acc_dtype = getattr(cutlass, FROZEN_ACC_DTYPE)
    return kernel_class(
        acc_dtype,
        spec["use_2cta_instrs"],
        spec["mma_tiler_mn"],
        spec["cluster_shape_mn"],
        FROZEN_USE_TMA_STORE,
    )


def _can_implement(gemm, spec: dict, cutlass, a_tensor, b_tensor, c_tensor) -> None:
    """Run the official ``can_implement()`` check for this variant's class.

    The two upstream classes deliberately expose different signatures:
    ``DenseGemmKernel.can_implement(a, b, c)`` takes the tensors, while
    ``PersistentDenseGemmKernel.can_implement(mnkl, a_dtype, b_dtype, c_dtype,
    a_major, b_major, c_major)`` takes the problem description. Each is called
    in its own official form; P3.4 never falls back to another configuration
    when a check fails.
    """
    if spec["persistent"]:
        supported = gemm.can_implement(
            FROZEN_MNKL,
            a_tensor.element_type,
            b_tensor.element_type,
            c_tensor.element_type,
            FROZEN_A_MAJOR,
            FROZEN_B_MAJOR,
            FROZEN_C_MAJOR,
        )
    else:
        supported = gemm.can_implement(a_tensor, b_tensor, c_tensor)
    if not supported:
        raise P34Error(
            f"{spec['variant']}: the pinned {spec['upstream_class']} cannot implement the "
            f"frozen configuration (tiler {spec['mma_tiler_mn']}, cluster "
            f"{spec['cluster_shape_mn']}, use_2cta_instrs={spec['use_2cta_instrs']}); "
            "P3.4 never falls back to another configuration"
        )


def _launch(compiled_gemm, spec: dict, a_tensor, b_tensor, c_tensor, stream) -> None:
    """Launch a compiled CuTe kernel with its dynamic-only runtime signature.

    ``cute.compile`` bakes every ``cutlass.Constexpr`` parameter in at compile
    time and drops it from the compiled callable, which therefore takes only
    the dynamic arguments. Both pinned examples demonstrate exactly this: the
    non-persistent one compiles ``(gemm, a, b, c, stream)`` and calls
    ``(a, b, c, stream)``, and the persistent one compiles
    ``(bmm, gemm, a, b, c, max_active_clusters, stream, epilogue_op)`` and also
    calls ``(a, b, c, stream)``. A TypeError here means that contract changed,
    which is a hard failure rather than something to work around.
    """
    try:
        compiled_gemm(a_tensor, b_tensor, c_tensor, stream)
    except TypeError as exc:
        raise P34Error(
            f"{spec['variant']}: the compiled kernel rejected the dynamic-only launch "
            f"signature (a, b, c, stream): {exc}. The pinned CuTe DSL is expected to bake "
            "every cutlass.Constexpr argument in at compile time; P3.4 does not guess "
            "another signature"
        ) from exc


def _reset_output(torch, c_torch_gpu) -> None:
    """Reset the shared output buffer to a sentinel, outside every timer.

    NaN is used deliberately: any element a kernel fails to write stays
    non-finite and is rejected by the complete-result check, instead of
    surviving as a stale value from the previous variant that would silently
    pass. This runs outside every timer and is followed by a synchronize.
    """
    c_torch_gpu.fill_(float("nan"))
    torch.cuda.synchronize()


def execute_measurement(warmup_iterations: int, iterations: int) -> str:
    """Run all three variants once and return the CSV text on success.

    Returning the text rather than writing it keeps the single emission point
    in ``main`` and makes it structurally impossible to emit a partial table:
    a failure in any of the three positions propagates before anything reaches
    stdout.
    """
    contract = load_pinned_contract()

    # (1) Both pinned upstream identities are proved before anything heavy is
    # imported and certainly before either module is loaded.
    sources = verify_upstream_sources(contract)
    for name in sorted(sources):
        log(
            f"upstream verified ({name}): {sources[name]['relative_path']} "
            f"blob {sources[name]['blob']} sha256 {sources[name]['sha256']}"
        )

    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    import torch

    # (1b) Environment and provenance, before any tensor exists. The IEEE FP32
    # API is required up front so a PyTorch that cannot guarantee a trustworthy
    # correctness verdict fails closed before any JIT compilation is spent.
    log("collecting environment and provenance")
    require_ieee_fp32_matmul_api(torch)
    provenance = collect_provenance(contract, torch, cutlass)

    revalidated = verify_upstream_sources(contract)
    if revalidated != sources:
        raise P34Error("a pinned upstream source changed during provenance collection")
    log(
        f"device: {provenance['gpu_name']} uuid={provenance['gpu_uuid']} "
        f"cc={provenance['compute_capability']} driver={provenance['driver_version']}"
    )

    modules = load_upstream_modules(sources)
    factory_module = modules[OPERAND_FACTORY_SOURCE]
    _assert_upstream_seed(factory_module)

    # (2) One shared operand set, built exactly as P3.2 and P3.3 build theirs,
    # entirely outside every timer. A and B are created once and never mutated,
    # so all three variants consume byte-identical inputs.
    log("allocating the shared operands once (outside every timer)")
    torch.manual_seed(FROZEN_SEED)
    ab_dtype = getattr(cutlass, FROZEN_AB_DTYPE)
    c_dtype = getattr(cutlass, FROZEN_C_DTYPE)

    (
        a_tensor,
        b_tensor,
        c_tensor,
        a_torch_cpu,
        b_torch_cpu,
        _c_torch_cpu,
        c_torch_gpu,
    ) = factory_module.create_tensors(
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

    # The reference is computed once, outside every timer, and reused: A and B
    # are identical and immutable for all three variants.
    log("computing the untimed IEEE-FP32 reference once (outside every timer)")
    reference = compute_reference(torch, a_torch_cpu, b_torch_cpu)

    rows = []
    for spec in FROZEN_VARIANTS:
        rows.append(
            _measure_variant(
                spec=spec,
                modules=modules,
                sources=sources,
                cutlass=cutlass,
                cute=cute,
                torch=torch,
                a_tensor=a_tensor,
                b_tensor=b_tensor,
                c_tensor=c_tensor,
                c_torch_gpu=c_torch_gpu,
                reference=reference,
                torch_stream=torch_stream,
                cute_stream=cute_stream,
                warmup_iterations=warmup_iterations,
                iterations=iterations,
                provenance=provenance,
            )
        )

    # Only a fully completed sweep of all three variants reaches this line.
    return serialize_rows(rows)


def _measure_variant(
    spec,
    modules,
    sources,
    cutlass,
    cute,
    torch,
    a_tensor,
    b_tensor,
    c_tensor,
    c_torch_gpu,
    reference,
    torch_stream,
    cute_stream,
    warmup_iterations,
    iterations,
    provenance,
) -> dict:
    """Run the frozen sequence for exactly one variant and return its row."""
    variant = spec["variant"]
    log(
        f"--- {variant}: {spec['upstream_class']} tiler={spec['mma_tiler_mn']} "
        f"cluster={spec['cluster_shape_mn']} use_2cta_instrs={spec['use_2cta_instrs']} ---"
    )

    # (3) The frozen variant object, from the source the frozen table names.
    module = modules[spec["source"]]
    gemm = _build_kernel(module, spec, cutlass)

    # (4) The official can_implement() check for this class.
    _can_implement(gemm, spec, cutlass, a_tensor, b_tensor, c_tensor)
    log(f"{variant}: can_implement OK")

    # (5) max_active_clusters, for a persistent variant only.
    if spec["persistent"]:
        max_active_clusters = _max_active_clusters(cutlass, spec)
        log(f"{variant}: max_active_clusters={max_active_clusters} (official helper)")
        compile_args = (a_tensor, b_tensor, c_tensor, max_active_clusters, cute_stream)
    else:
        max_active_clusters = None
        compile_args = (a_tensor, b_tensor, c_tensor, cute_stream)

    # (6) Reset C and synchronize, outside every timer.
    _reset_output(torch, c_torch_gpu)

    # (7) Compilation only.
    log(f"{variant}: compiling (JIT)")
    torch.cuda.synchronize()
    compile_start = time.perf_counter_ns()
    compiled_gemm = cute.compile(gemm, *compile_args)
    torch.cuda.synchronize()
    compile_time_ms = (time.perf_counter_ns() - compile_start) / 1e6

    # (8) First launch; its output is the tensor that gets validated.
    log(f"{variant}: first launch (also the correctness-validated launch)")
    torch.cuda.synchronize()
    first_launch_start = time.perf_counter_ns()
    _launch(compiled_gemm, spec, a_tensor, b_tensor, c_tensor, cute_stream)
    torch.cuda.synchronize()
    first_launch_ms = (time.perf_counter_ns() - first_launch_start) / 1e6

    # (9) Complete-result correctness, before any warm-up or steady state.
    max_abs_error, max_rel_error = validate_result(torch, variant, reference, c_torch_gpu)
    log(
        f"{variant}: correctness {CORRECTNESS_PASS} "
        f"(max_abs_error={max_abs_error!r} max_rel_error={max_rel_error!r})"
    )

    # (10) Warm-up, only after this variant passed correctness.
    log(f"{variant}: warm-up {warmup_iterations} launch(es)")
    for _ in range(warmup_iterations):
        _launch(compiled_gemm, spec, a_tensor, b_tensor, c_tensor, cute_stream)
    torch.cuda.synchronize()

    # (11) Steady state on the kernel's own stream.
    log(f"{variant}: steady state {iterations} measured launch(es)")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record(torch_stream)
    for _ in range(iterations):
        _launch(compiled_gemm, spec, a_tensor, b_tensor, c_tensor, cute_stream)
    end_event.record(torch_stream)
    torch.cuda.synchronize()
    total_ms = start_event.elapsed_time(end_event)
    if not math.isfinite(total_ms) or total_ms <= 0.0:
        raise P34Error(
            f"{variant}: CUDA-event elapsed time {total_ms!r} is not finite and positive"
        )
    kernel_time_ms = total_ms / iterations

    for name, value in (
        ("compile_time_ms", compile_time_ms),
        ("first_launch_ms", first_launch_ms),
        ("kernel_time_ms", kernel_time_ms),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise P34Error(f"{variant}: {name}={value!r} is not finite and strictly positive")

    # (12) Build the row, but emit nothing yet.
    return build_row(
        variant=variant,
        correctness=CORRECTNESS_PASS,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        compile_time_ms=compile_time_ms,
        first_launch_ms=first_launch_ms,
        kernel_time_ms=kernel_time_ms,
        warmup_iterations=warmup_iterations,
        iterations=iterations,
        max_active_clusters=max_active_clusters,
        provenance=provenance,
        upstream=sources[spec["source"]],
    )


def compute_reference(torch, a_torch_cpu, b_torch_cpu):
    """Compute the untimed IEEE-FP32 CUDA reference once, outside every timer.

    This is the same oracle P3.2 and P3.3 use, unchanged: the pinned PyTorch
    installation used purely as a correctness reference. It is never timed,
    never reported as a competing method, and never compared against any
    variant's timing. Because A and B are identical and immutable across all
    three variants, one reference is correct for all of them.
    """
    with ieee_fp32_matmul(torch):
        reference = torch.einsum(
            "mkl,nkl->mnl",
            a_torch_cpu.to(device="cuda", dtype=torch.float32),
            b_torch_cpu.to(device="cuda", dtype=torch.float32),
        )
    if not bool(torch.isfinite(reference).all()):
        raise CorrectnessError("the FP32 reference contains non-finite values")
    return reference


def validate_result(torch, variant, reference, c_torch_gpu):
    """Validate one variant's complete result against the untimed reference.

    The output buffer was reset to NaN before this variant ran, so any element
    the kernel failed to write is still non-finite here and is rejected. There
    is no fallback reference, no CPU reference, and no reduced-precision path.
    """
    result = c_torch_gpu.to(dtype=torch.float32)

    if tuple(result.shape) != tuple(reference.shape):
        raise CorrectnessError(
            f"{variant}: result shape {tuple(result.shape)} != reference shape "
            f"{tuple(reference.shape)}"
        )
    if not bool(torch.isfinite(result).all()):
        raise CorrectnessError(
            f"{variant}: the result contains non-finite values; the output buffer was reset "
            "to NaN before this variant, so an element the kernel did not write stays NaN"
        )

    difference = (result - reference).abs()
    tolerated = FROZEN_ATOL + FROZEN_RTOL * reference.abs()
    mismatches = int((difference > tolerated).sum().item())

    denominator = reference.abs().clamp_min(REL_ERROR_DENOMINATOR_FLOOR)
    max_abs_error = float(difference.max().item())
    max_rel_error = float((difference / denominator).max().item())

    if not math.isfinite(max_abs_error) or not math.isfinite(max_rel_error):
        raise CorrectnessError(f"{variant}: the measured error is not finite")

    if mismatches:
        raise CorrectnessError(
            f"{variant}: {mismatches} element(s) exceed atol={FROZEN_ATOL} "
            f"rtol={FROZEN_RTOL}; max_abs_error={max_abs_error} max_rel_error={max_rel_error}"
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
    """Build the whole P3.4 command line.

    The frozen scientific contract is deliberately unreachable from here: there
    is no shape, dtype, layout, variant, scheduler, tiler, cluster,
    persistence, 2-CTA, seed, tolerance, source-path, or correctness option,
    and no way to run fewer than all three variants.
    """
    parser = argparse.ArgumentParser(
        prog="cutedsl_variants.py",
        description=(
            "P3.4 three CuTe DSL execution variants. Executes exactly three frozen "
            "candidates - nonpersistent_1cta, persistent_1cta, persistent_2cta - at one "
            "frozen shape on identical operands, around the two pinned unmodified official "
            "NVIDIA examples, and emits one non-publishable CSV row per variant. Not an "
            "experimental campaign, not a comparison, and not a performance result."
        ),
        epilog=(
            "Correctness is mandatory and always runs before any warm-up or steady-state "
            "timing, per variant. All four output lines are emitted only after all three "
            "variants pass. The emitted timings are P3.4 infrastructure evidence only."
        ),
    )
    parser.add_argument(
        "--warmup-iterations",
        type=bounded_int(MIN_ITERATIONS, MAX_WARMUP_ITERATIONS),
        default=DEFAULT_WARMUP_ITERATIONS,
        metavar="N",
        help=(
            f"untimed launches before the measured ones, per variant "
            f"[{MIN_ITERATIONS}..{MAX_WARMUP_ITERATIONS}], default {DEFAULT_WARMUP_ITERATIONS}"
        ),
    )
    parser.add_argument(
        "--iterations",
        type=bounded_int(MIN_ITERATIONS, MAX_ITERATIONS),
        default=DEFAULT_ITERATIONS,
        metavar="N",
        help=(
            f"measured launches for kernel_time_ms, per variant "
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


def _synthetic_upstream(source: str) -> dict:
    """Obviously synthetic, but distinct per source, as the real rows are."""
    if source == SOURCE_PERSISTENT:
        return {
            "commit": "1" * 40,
            "relative_path": "examples/synthetic/dense_gemm_persistent.py",
            "blob": "3" * 40,
            "sha256": "4" * 64,
        }
    return {
        "commit": "1" * 40,
        "relative_path": "examples/synthetic/dense_gemm.py",
        "blob": "2" * 40,
        "sha256": "5" * 64,
    }


def _synthetic_row(variant: str) -> dict:
    spec = frozen_variant_spec(variant)
    return build_row(
        variant=variant,
        correctness=CORRECTNESS_PASS,
        max_abs_error=0.0,
        max_rel_error=0.0,
        compile_time_ms=1234.5,
        first_launch_ms=12.25,
        kernel_time_ms=7.5,
        warmup_iterations=2,
        iterations=10,
        max_active_clusters=148 if spec["persistent"] else None,
        provenance=_synthetic_provenance(),
        upstream=_synthetic_upstream(spec["source"]),
    )


def _synthetic_rows() -> list:
    return [_synthetic_row(variant) for variant in FROZEN_VARIANT_ORDER]


def run_self_test() -> int:
    """Prove the GPU-free half of the P3.4 contract, printing only to stderr."""
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
        except P34Error as exc:
            check(name, expected_fragment in str(exc), f"message was {str(exc)!r}")
        else:
            check(name, False, "no error was raised")

    print("cutedsl_variants --self-test (GPU-free)", file=sys.stderr)

    # Frozen schema.
    check("schema has 51 fields", len(CSV_FIELDS) == 51, str(len(CSV_FIELDS)))
    check("schema has no duplicate field", len(set(CSV_FIELDS)) == len(CSV_FIELDS))
    check("schema starts with schema_version", CSV_FIELDS[0] == "schema_version")
    check("schema ends with publishable", CSV_FIELDS[-1] == "publishable")
    check("schema version is p34.v1", SCHEMA_VERSION == "p34.v1")
    check(
        "no performance metric is in the schema",
        not any(
            re.search(
                r"tflop|flops|speedup|efficien|bandwidth|utilization|throughput|winner|rank",
                field,
            )
            for field in CSV_FIELDS
        ),
    )
    for required in ("variant", "scheduler", "max_active_clusters", "upstream_kernel_file",
                     "upstream_kernel_git_blob", "upstream_kernel_sha256"):
        check(f"schema carries {required}", required in CSV_FIELDS)

    # Frozen variant table.
    check("exactly three variants exist", len(FROZEN_VARIANTS) == 3, str(len(FROZEN_VARIANTS)))
    check(
        "the frozen order is nonpersistent, persistent 1-CTA, persistent 2-CTA",
        FROZEN_VARIANT_ORDER
        == (VARIANT_NONPERSISTENT_1CTA, VARIANT_PERSISTENT_1CTA, VARIANT_PERSISTENT_2CTA),
        str(FROZEN_VARIANT_ORDER),
    )
    check("variant names are unique", len(set(FROZEN_VARIANT_ORDER)) == 3)
    expected_table = {
        VARIANT_NONPERSISTENT_1CTA: (
            UPSTREAM_CLASS_NONPERSISTENT, SCHEDULER_NONPERSISTENT,
            (128, 128), (1, 1), False, SOURCE_NONPERSISTENT, False,
        ),
        VARIANT_PERSISTENT_1CTA: (
            UPSTREAM_CLASS_PERSISTENT, SCHEDULER_STATIC_PERSISTENT,
            (128, 128), (1, 1), False, SOURCE_PERSISTENT, True,
        ),
        VARIANT_PERSISTENT_2CTA: (
            UPSTREAM_CLASS_PERSISTENT, SCHEDULER_STATIC_PERSISTENT,
            (256, 128), (2, 1), True, SOURCE_PERSISTENT, True,
        ),
    }
    for spec in FROZEN_VARIANTS:
        actual = (
            spec["upstream_class"], spec["scheduler"], spec["mma_tiler_mn"],
            spec["cluster_shape_mn"], spec["use_2cta_instrs"], spec["source"],
            spec["persistent"],
        )
        check(
            f"{spec['variant']} matches the frozen table",
            actual == expected_table[spec["variant"]],
            f"{actual} != {expected_table[spec['variant']]}",
        )
    check(
        "the 2-CTA variant keeps a per-CTA M extent of 128",
        frozen_variant_spec(VARIANT_PERSISTENT_2CTA)["mma_tiler_mn"][0]
        // frozen_variant_spec(VARIANT_PERSISTENT_2CTA)["cluster_shape_mn"][0] == 128,
    )
    check(
        "only the 2-CTA variant sets use_2cta_instrs",
        [spec["use_2cta_instrs"] for spec in FROZEN_VARIANTS] == [False, False, True],
    )
    check(
        "the non-persistent variant is the only one using the P3.1 source",
        [spec["source"] for spec in FROZEN_VARIANTS]
        == [SOURCE_NONPERSISTENT, SOURCE_PERSISTENT, SOURCE_PERSISTENT],
    )

    # Frozen configuration.
    check("problem is (4096, 4096, 4096, 1)", FROZEN_MNKL == (4096, 4096, 4096, 1))
    check("dtypes are BF16 in, FP32 accumulate, FP32 out",
          (FROZEN_AB_DTYPE, FROZEN_ACC_DTYPE, FROZEN_C_DTYPE)
          == ("BFloat16", "Float32", "Float32"))
    check("majors are k, k, n",
          (FROZEN_A_MAJOR, FROZEN_B_MAJOR, FROZEN_C_MAJOR) == ("k", "k", "n"))
    check("TMA store is used", FROZEN_USE_TMA_STORE is True)
    check("seed is 1111", FROZEN_SEED == 1111)
    check("tolerances are 1e-1 / 1e-5", (FROZEN_ATOL, FROZEN_RTOL) == (1e-1, 1e-5))
    check("cache model is hot", CACHE_MODE == "hot")
    check("publishable is fixed to false", PUBLISHABLE == BOOL_FALSE)
    check("run_kind is smoke", RUN_KIND == "smoke")
    check("method is cutedsl", METHOD == "cutedsl")
    check("reference is the untimed torch CUDA FP32 oracle",
          REFERENCE == "torch_cuda_fp32_ieee")
    check("the operand factory is the non-persistent example",
          OPERAND_FACTORY_SOURCE == SOURCE_NONPERSISTENT)

    # Serialization: exactly four lines, in the frozen order.
    rows = _synthetic_rows()
    text = serialize_rows(rows)
    lines = text.splitlines()
    check("serialization emits exactly four lines", len(lines) == 4, str(len(lines)))
    check("header matches the frozen order", lines[0] == ",".join(CSV_FIELDS))
    parsed = list(csv.DictReader(io.StringIO(text)))
    check("exactly three data rows are parsed back", len(parsed) == 3, str(len(parsed)))
    check("the round trip is lossless", [dict(row) for row in parsed] == rows)
    check("rows appear in the frozen variant order",
          tuple(row["variant"] for row in parsed) == FROZEN_VARIANT_ORDER)
    check("timings use six fractional digits", rows[0]["kernel_time_ms"] == "7.500000")
    check("errors use nine fractional digits", rows[0]["max_abs_error"] == "0.000000000")
    check("atol serializes deterministically", rows[0]["atol"] == "0.100000000")
    check("rtol serializes deterministically", rows[0]["rtol"] == "0.000010000")
    check("the non-persistent row has no cluster count",
          rows[0]["max_active_clusters"] == MAX_ACTIVE_CLUSTERS_NOT_APPLICABLE)
    check("both persistent rows carry a positive cluster count",
          rows[1]["max_active_clusters"] == "148" and rows[2]["max_active_clusters"] == "148")
    check("the 2-CTA row records the 256x128 tiler and 2x1 cluster",
          (rows[2]["mma_tiler_m"], rows[2]["mma_tiler_n"], rows[2]["cluster_m"],
           rows[2]["cluster_n"]) == ("256", "128", "2", "1"))
    check("only the 2-CTA row sets use_2cta_instrs",
          [row["use_2cta_instrs"] for row in rows] == [BOOL_FALSE, BOOL_FALSE, BOOL_TRUE])
    check("the two schedulers are recorded distinctly",
          [row["scheduler"] for row in rows]
          == [SCHEDULER_NONPERSISTENT, SCHEDULER_STATIC_PERSISTENT,
              SCHEDULER_STATIC_PERSISTENT])
    check("each row names the upstream source it used",
          rows[0]["upstream_kernel_file"] != rows[1]["upstream_kernel_file"]
          and rows[1]["upstream_kernel_file"] == rows[2]["upstream_kernel_file"])
    check("every row is non-publishable",
          all(row["publishable"] == BOOL_FALSE for row in rows))

    # Rejections: correctness gating.
    for bad in ("FAIL", "SKIPPED", "pass", ""):
        rejects(
            f"correctness={bad!r} cannot build a row",
            lambda bad=bad: build_row(
                variant=VARIANT_NONPERSISTENT_1CTA,
                correctness=bad,
                max_abs_error=0.0,
                max_rel_error=0.0,
                compile_time_ms=1.0,
                first_launch_ms=1.0,
                kernel_time_ms=1.0,
                warmup_iterations=2,
                iterations=10,
                max_active_clusters=None,
                provenance=_synthetic_provenance(),
                upstream=_synthetic_upstream(SOURCE_NONPERSISTENT),
            ),
            "refusing to build a row",
        )

    # Rejections: the variant set and order.
    rejects("an unknown variant cannot build a row",
            lambda: _synthetic_row("persistent_4cta"), "not one of the three frozen")
    rejects("a missing variant is rejected",
            lambda: validate_rows(rows[:2]), "exactly 3 rows")
    rejects("an extra variant is rejected",
            lambda: validate_rows(rows + [rows[0]]), "exactly 3 rows")
    rejects("a duplicated variant is rejected",
            lambda: validate_rows([rows[0], rows[0], rows[2]]), "frozen order")
    rejects("a reordered variant set is rejected",
            lambda: validate_rows([rows[1], rows[0], rows[2]]), "frozen order")

    # Rejections: variant/configuration mismatches.
    rejects(
        "persistent_2cta with use_2cta_instrs=false is rejected",
        lambda: validate_row({**rows[2], "use_2cta_instrs": BOOL_FALSE}),
        "use_2cta_instrs",
    )
    rejects(
        "persistent_2cta with a (1,1) cluster is rejected",
        lambda: validate_row({**rows[2], "cluster_m": "1", "cluster_n": "1"}),
        "cluster_m",
    )
    rejects(
        "persistent_2cta with the wrong tiler is rejected",
        lambda: validate_row({**rows[2], "mma_tiler_m": "128"}),
        "mma_tiler_m",
    )
    rejects(
        "a persistent row claiming the non-persistent scheduler is rejected",
        lambda: validate_row({**rows[1], "scheduler": SCHEDULER_NONPERSISTENT}),
        "scheduler",
    )
    rejects(
        "a non-persistent row claiming the persistent scheduler is rejected",
        lambda: validate_row({**rows[0], "scheduler": SCHEDULER_STATIC_PERSISTENT}),
        "scheduler",
    )
    rejects(
        "a non-persistent row with a cluster count is rejected",
        lambda: validate_row({**rows[0], "max_active_clusters": "148"}),
        "not_applicable",
    )
    rejects(
        "a persistent row without a cluster count is rejected",
        lambda: validate_row(
            {**rows[1], "max_active_clusters": MAX_ACTIVE_CLUSTERS_NOT_APPLICABLE}
        ),
        "positive decimal integer",
    )
    rejects(
        "a persistent row with a zero cluster count is rejected",
        lambda: validate_row({**rows[1], "max_active_clusters": "0"}),
        "positive decimal integer",
    )
    rejects(
        "a non-integer cluster count cannot build a row",
        lambda: build_row(
            variant=VARIANT_PERSISTENT_1CTA, correctness=CORRECTNESS_PASS,
            max_abs_error=0.0, max_rel_error=0.0, compile_time_ms=1.0,
            first_launch_ms=1.0, kernel_time_ms=1.0, warmup_iterations=2, iterations=10,
            max_active_clusters=1.5, provenance=_synthetic_provenance(),
            upstream=_synthetic_upstream(SOURCE_PERSISTENT),
        ),
        "must be an integer",
    )
    rejects(
        "a non-persistent variant given a cluster count cannot build a row",
        lambda: build_row(
            variant=VARIANT_NONPERSISTENT_1CTA, correctness=CORRECTNESS_PASS,
            max_abs_error=0.0, max_rel_error=0.0, compile_time_ms=1.0,
            first_launch_ms=1.0, kernel_time_ms=1.0, warmup_iterations=2, iterations=10,
            max_active_clusters=148, provenance=_synthetic_provenance(),
            upstream=_synthetic_upstream(SOURCE_NONPERSISTENT),
        ),
        "no max_active_clusters",
    )

    # Rejections: schema and numbers.
    rejects("a missing field is rejected",
            lambda: validate_row({k: v for k, v in rows[0].items() if k != "kernel_time_ms"}),
            "missing field")
    rejects("an unknown field is rejected",
            lambda: validate_row({**rows[0], "tflops": "1.0"}), "unknown field")
    rejects("a non-string value is rejected",
            lambda: validate_row({**rows[0], "iterations": 10}), "not a string")
    for bad in ("nan", "inf", "-inf", "0.000000", "-1.000000", "1e3", "7.5"):
        rejects(f"kernel_time_ms={bad!r} is rejected",
                lambda bad=bad: validate_row({**rows[0], "kernel_time_ms": bad}),
                "kernel_time_ms")
    rejects("a non-finite float cannot be serialized",
            lambda: format_fixed(float("nan"), DECIMALS_TIMING), "not finite")
    rejects("a negative error cannot be serialized",
            lambda: format_fixed(-1.0, DECIMALS_ERROR), "negative")
    rejects("a changed shape is rejected",
            lambda: validate_row({**rows[0], "m": "8192"}), "frozen")
    rejects("a changed dtype is rejected",
            lambda: validate_row({**rows[0], "ab_dtype": "Float16"}), "frozen")
    rejects("a changed major is rejected",
            lambda: validate_row({**rows[0], "c_major": "m"}), "frozen")
    rejects("a disabled TMA store is rejected",
            lambda: validate_row({**rows[0], "use_tma_store": BOOL_FALSE}), "frozen")
    rejects("a changed seed is rejected",
            lambda: validate_row({**rows[0], "seed": "2222"}), "frozen")
    rejects("publishable=true is rejected",
            lambda: validate_row({**rows[0], "publishable": BOOL_TRUE}), "publishable")
    rejects("correctness=FAIL in a row is rejected",
            lambda: validate_row({**rows[0], "correctness": "FAIL"}), "correctness")
    rejects("a non-canonical boolean is rejected",
            lambda: validate_row({**rows[0], "git_dirty": "TRUE"}), "git_dirty")
    rejects("a malformed GPU UUID is rejected",
            lambda: validate_row({**rows[0], "gpu_uuid": "0000"}), "gpu_uuid")
    rejects("a malformed upstream blob is rejected",
            lambda: validate_row({**rows[0], "upstream_kernel_git_blob": "abc"}),
            "upstream_kernel_git_blob")
    rejects("a malformed upstream digest is rejected",
            lambda: validate_row({**rows[0], "upstream_kernel_sha256": "abc"}),
            "upstream_kernel_sha256")
    rejects("an absolute upstream path is rejected",
            lambda: validate_row({**rows[0], "upstream_kernel_file": "/opt/cutlass/x.py"}),
            "upstream_kernel_file")
    rejects("an out-of-range iteration count is rejected",
            lambda: validate_row({**rows[0], "iterations": str(MAX_ITERATIONS + 1)}),
            "outside")

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
    rejects("a malformed contract line is rejected",
            lambda: _parse_env_text("NOT A CONTRACT LINE\n"), "malformed contract line")
    rejects("a duplicate contract key is rejected",
            lambda: _parse_env_text("A_KEY=1\nA_KEY=2\n"), "duplicate contract key")
    check("comments and blank lines are ignored",
          _parse_env_text("# comment\n\nA_KEY=1\n") == {"A_KEY": "1"})
    check("sm_75 maps to compute capability 7.5", compute_capability_for_arch("sm_75") == "7.5")
    check("sm_90 maps to compute capability 9.0", compute_capability_for_arch("sm_90") == "9.0")
    check("sm_100 maps to compute capability 10.0",
          compute_capability_for_arch("sm_100") == "10.0")
    rejects("a malformed architecture pin is rejected",
            lambda: compute_capability_for_arch("blackwell"), "malformed")

    real_contract = load_pinned_contract()
    check(
        "the pinned architecture derives a well-formed compute capability",
        bool(_RE_COMPUTE_CAPABILITY.match(real_contract["EXPECTED_COMPUTE_CAPABILITY"])),
        f"{real_contract['CUDA_ARCH']} -> {real_contract['EXPECTED_COMPUTE_CAPABILITY']}",
    )
    check(
        "both upstream sources are pinned and distinct",
        real_contract["CUTEDSL_P31_EXAMPLE_PATH"]
        != real_contract["CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH"],
    )
    check(
        "the persistent pin names the persistent example",
        real_contract["CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH"].endswith(
            "dense_gemm_persistent.py"
        ),
    )

    # No heavy import happened.
    for module_name in ("torch", "cutlass", "cuda"):
        check(f"{module_name} was not imported", module_name not in sys.modules)
    for module_name in tuple(source["module_name"] for source in UPSTREAM_SOURCES.values()):
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
    # restored only to emit the four CSV lines, and only on full success of all
    # three variants.
    saved_stdout_fd = _redirect_stdout_to_stderr()
    try:
        csv_text = execute_measurement(args.warmup_iterations, args.iterations)
    except CorrectnessError as exc:
        os.close(saved_stdout_fd)
        log(f"CORRECTNESS FAILED: {exc}")
        log(
            "no CSV header and no CSV row are emitted, including for variants that already "
            "passed; that variant ran no warm-up and no steady-state timing"
        )
        return 1
    except P34Error as exc:
        os.close(saved_stdout_fd)
        log(f"FAIL: {exc}")
        log("no CSV header and no CSV row are emitted")
        return 1
    except BaseException:
        os.close(saved_stdout_fd)
        raise

    # Only a fully completed three-variant sweep, with every correctness check
    # already passed, reaches this line, and this is the only place a CSV is
    # ever written.
    _emit_on_saved_stdout(saved_stdout_fd, csv_text)
    log(
        f"emitted {len(FROZEN_VARIANTS)} non-publishable P3.4 rows (functional evidence, "
        "not a result and not a comparison)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
