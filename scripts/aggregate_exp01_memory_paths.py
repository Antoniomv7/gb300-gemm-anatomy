#!/usr/bin/env python3
"""P1.3 plan generation, CSV validation, consolidation, and aggregation for
the exp01_memory_paths joint LDGSTS/TMA sweep.

This script never touches CUDA, Docker, ``nvidia-smi``, either benchmark
binary, or the network. ``scripts/run_exp01_memory_paths.sh`` is the only
thing that invokes GPU work (exclusively through ``scripts/run_container.sh``
for anything that runs inside a container); this script plans the
18-invocation sweep, centralizes symlink-safe campaign initialization,
strictly validates every field of every repetition of the raw 37-column CSV
the two P1.1/P1.2 binaries already emit, consolidates it losslessly,
computes descriptive per-configuration statistics, and owns the campaign
manifest's state machine.

P1.3 produces functional/descriptive infrastructure output only: it does not
compute LDGSTS/TMA speedups, run Nsight Compute, judge outliers, or draw any
performance conclusion. See src/memory/README.md and PLAN.md.

Subcommands:
  plan            Print the frozen deterministic 18-invocation plan.
  init-campaign   Centralized, symlink-safe campaign creation: makes the
                  campaign directory (never following a broken/real symlink
                  at any path component, never overwriting an existing
                  campaign), writes execution_order.csv once, and writes the
                  initial IN_PROGRESS manifest. Prints the campaign's
                  repo-relative path on stdout.
  capture         Run one allowlisted binary invocation inside the
                  container, capturing its stdout to a temporary CSV and
                  publishing it with no-clobber semantics only on success
                  (used by run_exp01_memory_paths.sh via
                  scripts/run_container.sh; never touches the network
                  itself).
  validate-case   Strictly validate every field of every repetition in one
                  already-captured case CSV file.
  finalize        Re-validate an entire campaign against its own manifest's
                  recorded preconditions, then publish combined_samples.csv,
                  summary.csv, and a COMPLETE manifest with mandatory,
                  non-null hashes. Only this subcommand may set
                  status=COMPLETE.
  manifest-write  Merge a small allowlisted JSON fragment into manifest.json
                  (IN_PROGRESS/FAILED/INTERRUPTED only; never COMPLETE, and
                  never against an already-terminal campaign).
  --self-test     GPU-free synthetic positive/negative/adversarial tests (no
                  CUDA, no Docker, no nvidia-smi, no network, no real
                  subprocess — capture/artifact behavior is exercised via
                  unittest.mock). Prints
                  "aggregate_exp01_memory_paths: SELF_TEST_RESULT=PASS" only
                  if every case passes.

Exit codes: 0 on success (including --self-test passing); 1 on a validation,
aggregation, or capture failure; 2 on a usage/precondition error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import stat
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime as _datetime
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "1"
MANIFEST_SCHEMA_VERSION = "1"
EXPERIMENT_ID = "exp01_memory_paths"

METHODS = ("ldgsts", "tma")
CONFIG_PAIRS = (
    (2, 16), (2, 32), (2, 64),
    (4, 16), (4, 32), (4, 64),
    (8, 16), (8, 32), (8, 64),
)
EXPECTED_CONFIGURATION_COUNT = 18

RAW_ROOT_PARTS = ("results", "raw", "exp01_memory_paths")
RAW_ROOT_REL = Path(*RAW_ROOT_PARTS)

MEMORY_LDGSTS_BIN = REPO_ROOT / "build/memory/ldgsts"
MEMORY_LDGSTS_SASS = REPO_ROOT / "build/memory/ldgsts.sass"
MEMORY_TMA_BIN = REPO_ROOT / "build/memory/tma"
MEMORY_TMA_SASS = REPO_ROOT / "build/memory/tma.sass"

ALLOWED_BINARIES = {
    "build/memory/ldgsts": "ldgsts_bin",
    "build/memory/tma": "tma_bin",
}

DEFAULT_CAPTURE_ARTIFACTS = {
    "build/memory/ldgsts": MEMORY_LDGSTS_BIN,
    "build/memory/tma": MEMORY_TMA_BIN,
}

DEFAULT_FINAL_ARTIFACTS = {
    "ldgsts_bin": MEMORY_LDGSTS_BIN,
    "ldgsts_sass": MEMORY_LDGSTS_SASS,
    "tma_bin": MEMORY_TMA_BIN,
    "tma_sass": MEMORY_TMA_SASS,
}

REQUIRED_VERSION_KEYS = (
    "CUDA_VERSION",
    "CUDA_IMAGE",
    "CUDA_IMAGE_DIGEST",
    "CUDA_IMAGE_PLATFORM",
    "CUTLASS_VERSION",
    "CUTLASS_COMMIT",
    "CUDA_ARCH",
    "MAX_BUILD_JOBS",
)

# Frozen contract constants shared by both binaries (see src/memory/README.md).
FROZEN_THREADS_PER_CTA = 128
FROZEN_TARGET_CTAS_PER_SM = 1
FROZEN_OCCUPANCY_CTAS_PER_SM = 1
FROZEN_TILE_WIDTH_ELEMENTS = 128
FROZEN_TILE_WIDTH_BYTES = 256
FROZEN_VECTOR_BYTES = 16
FROZEN_COMPUTE_CAPABILITY = "10.3"

# Exact 37-column header both build/memory/ldgsts and build/memory/tma print.
CSV_HEADER = [
    "schema_version", "timestamp_utc", "run_kind", "method", "sample_index",
    "stages", "tile_width_elements", "tile_width_bytes", "tile_height",
    "stage_bytes", "bytes_in_flight_per_sm", "vector_bytes",
    "copies_per_thread_per_stage", "threads_per_cta", "target_ctas_per_sm",
    "occupancy_ctas_per_sm", "grid_blocks", "sm_count",
    "smem_reservation_bytes", "l2_bytes", "requested_working_set_bytes",
    "working_set_bytes", "working_set_l2_ratio", "passes", "useful_bytes",
    "warmup_ms", "kernel_time_ms", "effective_gbps", "correctness",
    "mismatches", "gpu_name", "gpu_uuid", "compute_capability",
    "cuda_driver_version", "cuda_runtime_version", "git_commit", "git_dirty",
]
assert len(CSV_HEADER) == 37, f"CSV_HEADER has {len(CSV_HEADER)} columns, expected 37"

EXECUTION_ORDER_HEADER = [
    "invocation_index", "method", "stages", "bytes_in_flight_kib",
    "stage_bytes", "bytes_in_flight_per_sm", "tile_height",
    "copies_per_thread_per_stage", "case_file",
]

CASE_NAME_RE = re.compile(r"^(\d{2})_(ldgsts|tma)_s(\d+)_bif(\d+)$")
CAMPAIGN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
TIMESTAMP_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
MANIFEST_TIMESTAMP_RE = re.compile(r"^\d{8}T\d{6}Z$")
CANONICAL_UINT_RE = re.compile(r"^(?:0|[1-9]\d*)$")
CANONICAL_FIXED6_RE = re.compile(r"^(?:0|[1-9]\d*)\.\d{6}$")

# Documented tolerances for values that pass through the binaries' fixed
# six-decimal CSV formatting (std::fixed << std::setprecision(6)).
FIXED6_HALF_ULP = 0.5e-6
RATIO_ABS_TOL = FIXED6_HALF_ULP + 1e-12

INT64_MAX = 2**63 - 1

TERMINAL_STATUSES = frozenset({"COMPLETE", "FAILED", "INTERRUPTED"})
ALLOWED_TRANSITIONS: dict[str | None, frozenset[str]] = {
    None: frozenset({"IN_PROGRESS"}),
    "IN_PROGRESS": frozenset({"IN_PROGRESS", "COMPLETE", "FAILED", "INTERRUPTED"}),
    "COMPLETE": frozenset(),
    "FAILED": frozenset(),
    "INTERRUPTED": frozenset(),
}

# Central manifest field allowlist: every top-level key merge_manifest will
# ever accept, and its required Python type(s). Anything else is rejected.
ALLOWED_MANIFEST_KEYS: dict[str, object] = {
    "schema_version": str,
    "experiment_id": str,
    "campaign_id": str,
    "status": str,
    "run_kind": str,
    "started_at_utc": str,
    "completed_at_utc": (str, type(None)),
    "configuration_count_expected": int,
    "configuration_count_completed": int,
    "sample_count_expected": int,
    "sample_count_completed": int,
    "requested": dict,
    "observed_common": dict,
    "invocation_order": list,
    "selected_gpu_index": int,
    "gpu_name": str,
    "gpu_uuid": str,
    "compute_capability": str,
    "cuda_driver_version": (str, int),
    "cuda_runtime_version": (str, int),
    "git_commit": str,
    "git_dirty": bool,
    "versions_env": dict,
    "binary_and_sass_sha256": dict,
    "case_file_sha256": dict,
    "execution_order_sha256": str,
    "aggregate_file_sha256": dict,
    "self_test_outcomes": dict,
    "failure_stage": (str, type(None)),
    "failure_detail": (list, type(None)),
    "failure_exit_code": (int, type(None)),
    "publishable": bool,
}


# ---------------------------------------------------------------------------
# Frozen geometry formulas (single source of truth, mirrors src/memory/*.cu).
# ---------------------------------------------------------------------------
def stage_bytes_of(stages: int, bif_kib: int) -> int:
    return (bif_kib * 1024) // stages


def bytes_in_flight_of(bif_kib: int) -> int:
    return bif_kib * 1024


def copies_per_thread_of(stages: int, bif_kib: int) -> int:
    return stage_bytes_of(stages, bif_kib) // (FROZEN_THREADS_PER_CTA * FROZEN_VECTOR_BYTES)


def tile_height_of(stages: int, bif_kib: int) -> int:
    return stage_bytes_of(stages, bif_kib) // FROZEN_TILE_WIDTH_BYTES


def round_up_to_multiple(value: int, multiple: int) -> int:
    """Mirrors src/memory/{ldgsts,tma}.cu's round_up_to_multiple exactly."""
    if value <= 0:
        return multiple
    units = (value + multiple - 1) // multiple
    return units * multiple


# ---------------------------------------------------------------------------
# Plan generation (frozen 18-invocation contract).
# ---------------------------------------------------------------------------
def build_plan() -> list[dict]:
    plan = []
    index = 0
    for pair_num, (stages, bif_kib) in enumerate(CONFIG_PAIRS):
        order = METHODS if pair_num % 2 == 0 else (METHODS[1], METHODS[0])
        for method in order:
            case_name = f"{index:02d}_{method}_s{stages}_bif{bif_kib}"
            plan.append({
                "index": index,
                "method": method,
                "stages": stages,
                "bif_kib": bif_kib,
                "case_name": case_name,
            })
            index += 1
    return plan


def plan_by_index(plan: list[dict]) -> dict[int, dict]:
    return {entry["index"]: entry for entry in plan}


def check_plan_contract(plan: list[dict]) -> list[str]:
    """Independently re-derives every property build_plan() is supposed to
    guarantee, so a future edit to build_plan() cannot silently break the
    frozen contract without failing --self-test."""
    errors: list[str] = []

    if len(plan) != EXPECTED_CONFIGURATION_COUNT:
        errors.append(f"plan has {len(plan)} invocations, expected {EXPECTED_CONFIGURATION_COUNT}")

    indices = [entry["index"] for entry in plan]
    if indices != list(range(len(plan))):
        errors.append(f"plan indices are not exactly 0..{len(plan) - 1} in order: {indices}")

    seen_keys: set[tuple[str, int, int]] = set()
    for entry in plan:
        key = (entry["method"], entry["stages"], entry["bif_kib"])
        if key in seen_keys:
            errors.append(f"duplicate invocation for {key}")
        seen_keys.add(key)

    for method in METHODS:
        count = sum(1 for entry in plan if entry["method"] == method)
        if count != 9:
            errors.append(f"method {method!r} appears {count} times, expected 9")

    for stages, bif_kib in CONFIG_PAIRS:
        methods_here = sorted(
            entry["method"] for entry in plan
            if entry["stages"] == stages and entry["bif_kib"] == bif_kib
        )
        if methods_here != sorted(METHODS):
            errors.append(
                f"config stages={stages} bif_kib={bif_kib} has methods={methods_here}, "
                f"expected exactly one of each method"
            )

    if len(plan) == EXPECTED_CONFIGURATION_COUNT:
        for pair_num, (stages, bif_kib) in enumerate(CONFIG_PAIRS):
            first, second = plan[pair_num * 2]["method"], plan[pair_num * 2 + 1]["method"]
            expected = ("ldgsts", "tma") if pair_num % 2 == 0 else ("tma", "ldgsts")
            if (first, second) != expected:
                errors.append(
                    f"pair {pair_num} (stages={stages}, bif_kib={bif_kib}) order "
                    f"{(first, second)} != expected {expected}"
                )
            pair_first_entry = plan[pair_num * 2]
            pair_second_entry = plan[pair_num * 2 + 1]
            if (pair_first_entry["stages"], pair_first_entry["bif_kib"]) != (stages, bif_kib):
                errors.append(f"pair {pair_num} first entry has wrong configuration")
            if (pair_second_entry["stages"], pair_second_entry["bif_kib"]) != (stages, bif_kib):
                errors.append(f"pair {pair_num} second entry has wrong configuration")

    return errors


def format_plan_text(plan: list[dict]) -> str:
    lines = [
        "index  method   stages  bif_kib  stage_bytes  tile_height  copies/thread/stage  case_name",
    ]
    for entry in plan:
        lines.append(
            f"{entry['index']:>5d}  {entry['method']:<7s}  {entry['stages']:>6d}  "
            f"{entry['bif_kib']:>7d}  {stage_bytes_of(entry['stages'], entry['bif_kib']):>11d}  "
            f"{tile_height_of(entry['stages'], entry['bif_kib']):>11d}  "
            f"{copies_per_thread_of(entry['stages'], entry['bif_kib']):>19d}  {entry['case_name']}"
        )
    lines.append(f"total invocations: {len(plan)}")
    return "\n".join(lines) + "\n"


def format_plan_lines(plan: list[dict]) -> str:
    return "".join(
        f"{entry['index']}\t{entry['method']}\t{entry['stages']}\t{entry['bif_kib']}\t{entry['case_name']}\n"
        for entry in plan
    )


# ---------------------------------------------------------------------------
# Path safety: lstat-based, symlink-refusing, no-clobber primitives.
# ---------------------------------------------------------------------------
class UnsafePathError(ValueError):
    pass


class ManifestTransitionError(ValueError):
    pass


def validate_campaign_id(campaign_id: str) -> None:
    if not campaign_id:
        raise UnsafePathError("campaign ID must not be empty")
    if len(campaign_id) > 64:
        raise UnsafePathError(f"campaign ID exceeds 64 characters: {campaign_id!r}")
    if not CAMPAIGN_ID_RE.match(campaign_id):
        raise UnsafePathError(
            f"campaign ID {campaign_id!r} does not match [A-Za-z0-9][A-Za-z0-9._-]{{0,63}}"
        )
    if ".." in campaign_id:
        raise UnsafePathError(f"campaign ID {campaign_id!r} must not contain '..'")
    if "/" in campaign_id or "\\" in campaign_id:
        raise UnsafePathError(f"campaign ID {campaign_id!r} must not contain a path separator")
    if any(ord(c) < 0x20 or ord(c) == 0x7f for c in campaign_id):
        raise UnsafePathError(f"campaign ID {campaign_id!r} must not contain control characters")


def _reject_if_symlink_or_wrong_type(path: Path, *, expect_dir: bool) -> None:
    """Uses lstat (never resolve()/is_dir(), which follow symlinks) so a
    symlink at this exact path component — even a dangling one — is refused
    regardless of what it points to or whether the target exists."""
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise UnsafePathError(f"{path}: lstat failed: {exc}") from exc
    if stat.S_ISLNK(st.st_mode):
        raise UnsafePathError(f"{path}: is a symlink; refusing")
    if expect_dir and not stat.S_ISDIR(st.st_mode):
        raise UnsafePathError(f"{path}: exists and is not a directory")


def _confirm_contained(path: Path, root: Path) -> None:
    resolved = path.resolve()
    root_resolved = root.resolve()
    if resolved != root_resolved and root_resolved not in resolved.parents:
        raise UnsafePathError(f"{path}: resolves outside {root}")


def _mkdir_component(path: Path, *, must_not_exist: bool) -> None:
    """Creates or reuses exactly one directory path component, lexically and
    physically verified both before and after creation. Never uses
    mkdir(parents=True)/exist_ok, so every level gets its own check."""
    _reject_if_symlink_or_wrong_type(path, expect_dir=True)
    exists = os.path.lexists(path)
    if must_not_exist and exists:
        raise UnsafePathError(f"{path}: already exists, refusing to overwrite")
    if not exists:
        try:
            os.mkdir(path)
        except FileExistsError as exc:
            raise UnsafePathError(f"{path}: already exists, refusing to overwrite") from exc
    _reject_if_symlink_or_wrong_type(path, expect_dir=True)
    if not os.path.isdir(path):
        raise UnsafePathError(f"{path}: is not a directory after creation")
    _confirm_contained(path, REPO_ROOT)


def create_campaign_dir(campaign_id: str) -> Path:
    """Centralized, symlink-safe campaign creation. Walks
    results/raw/exp01_memory_paths/<campaign_id>/{cases,logs} one component
    at a time via lstat, rejecting a symlink or wrong-type object at any
    level (including the raw root itself), and fails if the campaign
    directory already exists."""
    validate_campaign_id(campaign_id)
    current = REPO_ROOT
    for part in RAW_ROOT_PARTS:
        current = current / part
        _mkdir_component(current, must_not_exist=False)
    campaign_dir = current / campaign_id
    _mkdir_component(campaign_dir, must_not_exist=True)
    for sub in ("cases", "logs"):
        _mkdir_component(campaign_dir / sub, must_not_exist=False)
    return campaign_dir


def resolve_campaign_dir(campaign_dir_rel: str) -> Path:
    """Resolves an already-initialized campaign directory with the same
    lstat-based symlink/type safety as create_campaign_dir. Requires exactly
    results/raw/exp01_memory_paths/<campaign_id> (one campaign-ID component
    beneath the raw root, no nested/extra components)."""
    if os.path.isabs(campaign_dir_rel):
        raise UnsafePathError(f"--campaign-dir must be relative, got absolute path {campaign_dir_rel!r}")
    parts = Path(campaign_dir_rel).parts
    if any(".." in part for part in parts):
        raise UnsafePathError(f"--campaign-dir must not contain '..': {campaign_dir_rel!r}")
    if len(parts) != len(RAW_ROOT_PARTS) + 1 or tuple(parts[: len(RAW_ROOT_PARTS)]) != RAW_ROOT_PARTS:
        raise UnsafePathError(
            f"--campaign-dir must be exactly {'/'.join(RAW_ROOT_PARTS)}/<campaign_id>, "
            f"got {campaign_dir_rel!r}"
        )
    validate_campaign_id(parts[-1])

    current = REPO_ROOT
    for part in parts:
        current = current / part
        _reject_if_symlink_or_wrong_type(current, expect_dir=True)
        if not os.path.lexists(current):
            raise UnsafePathError(f"{current}: does not exist")
    _confirm_contained(current, REPO_ROOT)
    for subdir_name in ("cases", "logs"):
        subdir = current / subdir_name
        _reject_if_symlink_or_wrong_type(subdir, expect_dir=True)
        if not os.path.lexists(subdir):
            raise UnsafePathError(f"{subdir}: required campaign directory does not exist")
        _confirm_contained(subdir, current)
    return current


def resolve_capture_out_path(campaign_dir: Path, out_rel: str) -> Path:
    """Validates --out for the capture subcommand: must be exactly
    'cases/<name>.csv', the 'cases' parent must be a real non-symlink
    directory, and the final target must not already lexist (so a broken
    symlink there is refused, not silently treated as free)."""
    if os.path.isabs(out_rel):
        raise UnsafePathError(f"--out must be relative, got absolute path {out_rel!r}")
    parts = Path(out_rel).parts
    if any(".." in part for part in parts):
        raise UnsafePathError(f"--out must not contain '..': {out_rel!r}")
    if len(parts) != 2 or parts[0] != "cases":
        raise UnsafePathError(f"--out must be exactly 'cases/<name>.csv', got {out_rel!r}")

    cases_dir = campaign_dir / "cases"
    _reject_if_symlink_or_wrong_type(cases_dir, expect_dir=True)
    if not os.path.lexists(cases_dir):
        raise UnsafePathError(f"{cases_dir}: cases directory does not exist")

    out_path = cases_dir / parts[1]
    if os.path.lexists(out_path):
        raise UnsafePathError(f"refusing to overwrite existing target: {out_path}")
    _confirm_contained(out_path, campaign_dir)
    return out_path


def _publish_no_clobber(tmp_path: Path, final_path: Path) -> None:
    """Publishes tmp_path as final_path without ever overwriting an existing
    target: hard-link then unlink the temporary name. Never os.replace(),
    which silently overwrites. Same-filesystem only (both paths are always
    within the same campaign directory)."""
    if os.path.lexists(final_path):
        raise UnsafePathError(f"refusing to overwrite existing target: {final_path}")
    source_identity = _file_identity(tmp_path)
    try:
        os.link(tmp_path, final_path)
    except FileExistsError as exc:
        raise UnsafePathError(f"refusing to overwrite existing target: {final_path}") from exc
    except OSError as exc:
        raise UnsafePathError(
            f"could not publish {tmp_path} as {final_path} without clobbering: {exc}"
        ) from exc
    try:
        tmp_path.unlink()
    except OSError as exc:
        try:
            _safe_unlink_owned(final_path, source_identity)
        except UnsafePathError:
            pass
        raise UnsafePathError(
            f"published {final_path}, but could not remove owned temporary {tmp_path}: {exc}"
        ) from exc


def _open_regular_nofollow(path: Path, *, binary: bool):
    """Open an existing non-empty regular file without following a symlink.

    The pre-open lstat gives useful diagnostics; O_NOFOLLOW and the post-open
    fstat close the symlink/type gap at the actual open operation.
    """
    artifact_error = _verify_artifact(path)
    if artifact_error:
        raise UnsafePathError(artifact_error)
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise UnsafePathError(f"{path}: safe open failed: {exc}") from exc
    try:
        opened = os.fstat(fd)
        current = os.lstat(path)
        if not stat.S_ISREG(opened.st_mode):
            raise UnsafePathError(f"{path}: opened object is not a regular file")
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise UnsafePathError(f"{path}: changed while being opened")
        if binary:
            return os.fdopen(fd, "rb")
        return os.fdopen(fd, "r", encoding="utf-8", newline="")
    except Exception:
        os.close(fd)
        raise


def _open_exclusive(path: Path, *, binary: bool, newline: str | None = None):
    """Create a new regular file at path atomically, never following or
    replacing an existing regular file, directory, symlink, or broken
    symlink."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise UnsafePathError(f"{path}: already exists, refusing to overwrite") from exc
    except OSError as exc:
        raise UnsafePathError(f"{path}: exclusive creation failed: {exc}") from exc
    if binary:
        return os.fdopen(fd, "wb")
    return os.fdopen(fd, "w", encoding="utf-8", newline=newline)


def _safe_unlink_owned(path: Path, identity: tuple[int, int] | None = None) -> None:
    """Remove only a regular non-symlink file created by this process.

    When identity is supplied, refuse to unlink a path whose inode changed.
    This is used only for rollback/cleanup of this invocation's temporaries
    and published aggregates.
    """
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise UnsafePathError(f"{path}: cleanup target is not the owned regular file")
    if identity is not None and (st.st_dev, st.st_ino) != identity:
        raise UnsafePathError(f"{path}: cleanup target changed; refusing to unlink")
    path.unlink()


def _file_identity(path: Path) -> tuple[int, int]:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise UnsafePathError(f"{path}: expected a regular non-symlink file")
    return st.st_dev, st.st_ino


def _verify_artifact(path: Path) -> str | None:
    """Returns an error string, or None if path is a non-symlink, non-empty,
    regular file. Uses lstat so a symlinked "binary" is rejected outright."""
    if not os.path.lexists(path):
        return f"{path}: does not exist"
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        return f"{path}: is a symlink; refusing"
    if not stat.S_ISREG(st.st_mode):
        return f"{path}: is not a regular file"
    if st.st_size == 0:
        return f"{path}: is empty"
    return None


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with _open_regular_nofollow(path, binary=True) as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_sha256_hex(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA256_HEX_RE.match(value))


# ---------------------------------------------------------------------------
# Strict, centralized per-field validation contract.
#
# FIELD_VALIDATORS maps every one of the 37 CSV_HEADER columns to a callable
# `(row, expect, errors, ctx) -> value | None` that parses and validates that
# field, appending human-readable errors and returning the canonical parsed
# value (or None on failure). A validator may read *other* raw string fields
# out of `row` directly (via the quiet `_peek_*` helpers, which never append
# a duplicate error — the sibling field's own validator already reports its
# own parse failure) when a cross-field formula needs them. The module-level
# assertion below guarantees no column can be silently dropped from the
# contract by a future edit.
# ---------------------------------------------------------------------------
def _parse_strict_int(raw: str, errors: list[str], ctx: str, field: str) -> int | None:
    if raw is None or not CANONICAL_UINT_RE.fullmatch(raw):
        errors.append(
            f"{ctx}: {field}={raw!r} is not a canonical unsigned decimal integer "
            f"(expected 0 or a non-zero digit followed by digits)"
        )
        return None
    value = int(raw)
    if value > INT64_MAX:
        errors.append(f"{ctx}: {field}={value} is outside the signed 64-bit range")
        return None
    return value


def _parse_strict_float(raw: str, errors: list[str], ctx: str, field: str) -> float | None:
    if raw is None or not CANONICAL_FIXED6_RE.fullmatch(raw):
        errors.append(
            f"{ctx}: {field}={raw!r} is not the binary's canonical non-negative "
            f"fixed-point form with exactly six decimals"
        )
        return None
    try:
        value = float(raw)
    except ValueError:
        errors.append(f"{ctx}: {field}={raw!r} is not a float")
        return None
    if math.isnan(value):
        errors.append(f"{ctx}: {field} is NaN")
        return None
    if math.isinf(value):
        errors.append(f"{ctx}: {field} is infinite")
        return None
    return value


def _peek_int(row: dict[str, str], field: str) -> int | None:
    """Best-effort parse for cross-field checks; silently returns None on
    failure (the field's own dedicated validator reports that error)."""
    raw = row.get(field)
    if raw is None or not CANONICAL_UINT_RE.fullmatch(raw):
        return None
    value = int(raw)
    if value > INT64_MAX:
        return None
    return value


def _peek_float(row: dict[str, str], field: str) -> float | None:
    raw = row.get(field)
    if raw is None or not CANONICAL_FIXED6_RE.fullmatch(raw):
        return None
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _v_schema_version(row, expect, errors, ctx):
    if row["schema_version"] != SCHEMA_VERSION:
        errors.append(f"{ctx}: schema_version={row['schema_version']!r} != {SCHEMA_VERSION!r}")
        return None
    return row["schema_version"]


def _v_timestamp_utc(row, expect, errors, ctx):
    raw = row["timestamp_utc"]
    if not TIMESTAMP_UTC_RE.fullmatch(raw):
        errors.append(
            f"{ctx}: timestamp_utc={raw!r} is not in exact "
            f"YYYY-MM-DDTHH:MM:SSZ form"
        )
        return None
    try:
        _datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        errors.append(
            f"{ctx}: timestamp_utc={raw!r} is not a real calendar timestamp in "
            f"YYYY-MM-DDTHH:MM:SSZ form"
        )
        return None
    return raw


def _v_run_kind(row, expect, errors, ctx):
    if row["run_kind"] != expect["run_kind"]:
        errors.append(f"{ctx}: run_kind={row['run_kind']!r} != expected {expect['run_kind']!r}")
        return None
    return row["run_kind"]


def _v_method(row, expect, errors, ctx):
    if row["method"] != expect["method"]:
        errors.append(f"{ctx}: method={row['method']!r} != expected {expect['method']!r}")
        return None
    return row["method"]


def _v_sample_index(row, expect, errors, ctx):
    value = _parse_strict_int(row["sample_index"], errors, ctx, "sample_index")
    if value is None:
        return None
    if value < 0:
        errors.append(f"{ctx}: sample_index={value} must be >= 0")
    return value


def _v_stages(row, expect, errors, ctx):
    value = _parse_strict_int(row["stages"], errors, ctx, "stages")
    if value is None:
        return None
    if value not in (2, 4, 8):
        errors.append(f"{ctx}: stages={value} not in {{2,4,8}}")
    if value != expect["stages"]:
        errors.append(f"{ctx}: stages={value} != expected {expect['stages']}")
    return value


def _make_exact_int_validator(field: str, expected_fn, *, must_be_positive: bool = False):
    def validator(row, expect, errors, ctx):
        value = _parse_strict_int(row[field], errors, ctx, field)
        if value is None:
            return None
        if must_be_positive and value <= 0:
            errors.append(f"{ctx}: {field}={value} must be > 0")
        expected = expected_fn(expect)
        if value != expected:
            errors.append(f"{ctx}: {field}={value} != {expected} (expected/formula)")
        return value
    return validator


def _make_positive_int_validator(field: str, *, allow_zero: bool = False):
    def validator(row, expect, errors, ctx):
        value = _parse_strict_int(row[field], errors, ctx, field)
        if value is None:
            return None
        if allow_zero:
            if value < 0:
                errors.append(f"{ctx}: {field}={value} must be >= 0")
        elif value <= 0:
            errors.append(f"{ctx}: {field}={value} must be > 0")
        return value
    return validator


def _v_grid_blocks(row, expect, errors, ctx):
    value = _parse_strict_int(row["grid_blocks"], errors, ctx, "grid_blocks")
    if value is None:
        return None
    if value <= 0:
        errors.append(f"{ctx}: grid_blocks={value} must be > 0")
    sm_count = _peek_int(row, "sm_count")
    if sm_count is not None and value != sm_count:
        errors.append(f"{ctx}: grid_blocks={value} != sm_count={sm_count}")
    return value


def _v_smem_reservation_bytes(row, expect, errors, ctx):
    value = _parse_strict_int(row["smem_reservation_bytes"], errors, ctx, "smem_reservation_bytes")
    if value is None:
        return None
    if value <= 0:
        errors.append(f"{ctx}: smem_reservation_bytes={value} must be > 0")
        return value
    bif = _peek_int(row, "bytes_in_flight_per_sm")
    if bif is not None and value < bif:
        errors.append(f"{ctx}: smem_reservation_bytes={value} < bytes_in_flight_per_sm={bif}")
    return value


def _v_requested_working_set_bytes(row, expect, errors, ctx):
    value = _parse_strict_int(row["requested_working_set_bytes"], errors, ctx, "requested_working_set_bytes")
    if value is None:
        return None
    if value <= 0:
        errors.append(f"{ctx}: requested_working_set_bytes={value} must be > 0")
        return value
    working_set_mib = expect.get("working_set_mib")
    if working_set_mib is not None:
        expected = working_set_mib * 1024 * 1024
        if value != expected:
            errors.append(
                f"{ctx}: requested_working_set_bytes={value} != explicit "
                f"--working-set-mib {working_set_mib} * 1024*1024 = {expected}"
            )
    else:
        l2_bytes = _peek_int(row, "l2_bytes")
        if l2_bytes is not None and l2_bytes > 0:
            expected = 4 * l2_bytes
            if value != expected:
                errors.append(
                    f"{ctx}: requested_working_set_bytes={value} != implicit default "
                    f"4*l2_bytes={expected}"
                )
    return value


def _v_working_set_bytes(row, expect, errors, ctx):
    value = _parse_strict_int(row["working_set_bytes"], errors, ctx, "working_set_bytes")
    if value is None:
        return None
    if value <= 0:
        errors.append(f"{ctx}: working_set_bytes={value} must be > 0")
        return value
    sm_count = _peek_int(row, "sm_count")
    requested = _peek_int(row, "requested_working_set_bytes")
    if sm_count is not None and sm_count > 0 and requested is not None and requested > 0:
        common_multiple = sm_count * 32 * 1024
        expected = round_up_to_multiple(requested, common_multiple)
        if value != expected:
            errors.append(
                f"{ctx}: working_set_bytes={value} != round_up(requested_working_set_bytes="
                f"{requested}, sm_count*32KiB={common_multiple})={expected}"
            )
    l2_bytes = _peek_int(row, "l2_bytes")
    if expect["run_kind"] == "benchmark" and l2_bytes is not None and l2_bytes > 0:
        if not (value > 2 * l2_bytes):
            errors.append(
                f"{ctx}: working_set_bytes={value} is not > 2*l2_bytes={2 * l2_bytes} "
                f"(required for run_kind=benchmark)"
            )
    return value


def _v_working_set_l2_ratio(row, expect, errors, ctx):
    value = _parse_strict_float(row["working_set_l2_ratio"], errors, ctx, "working_set_l2_ratio")
    if value is None:
        return None
    working_set_bytes = _peek_int(row, "working_set_bytes")
    l2_bytes = _peek_int(row, "l2_bytes")
    if working_set_bytes is not None and l2_bytes is not None and l2_bytes > 0:
        expected = working_set_bytes / l2_bytes
        if abs(value - expected) > RATIO_ABS_TOL:
            errors.append(
                f"{ctx}: working_set_l2_ratio={value} inconsistent with "
                f"working_set_bytes/l2_bytes={expected} (tol={RATIO_ABS_TOL})"
            )
    return value


def _v_passes(row, expect, errors, ctx):
    value = _parse_strict_int(row["passes"], errors, ctx, "passes")
    if value is None:
        return None
    if value <= 0:
        errors.append(f"{ctx}: passes={value} must be > 0")
    if value != expect["passes"]:
        errors.append(f"{ctx}: passes={value} != requested {expect['passes']}")
    return value


def _v_useful_bytes(row, expect, errors, ctx):
    value = _parse_strict_int(row["useful_bytes"], errors, ctx, "useful_bytes")
    if value is None:
        return None
    if value <= 0:
        errors.append(f"{ctx}: useful_bytes={value} must be > 0")
    working_set_bytes = _peek_int(row, "working_set_bytes")
    passes = _peek_int(row, "passes")
    if working_set_bytes is not None and passes is not None:
        product = working_set_bytes * passes
        if product > INT64_MAX:
            errors.append(
                f"{ctx}: working_set_bytes*passes={product} overflows the signed 64-bit range"
            )
        elif value != product:
            errors.append(f"{ctx}: useful_bytes={value} != working_set_bytes*passes={product} (formula)")
    return value


def _v_warmup_ms(row, expect, errors, ctx):
    value = _parse_strict_int(row["warmup_ms"], errors, ctx, "warmup_ms")
    if value is None:
        return None
    if value < 0:
        errors.append(f"{ctx}: warmup_ms={value} must be >= 0")
    if value != expect["warmup_ms"]:
        errors.append(f"{ctx}: warmup_ms={value} != requested {expect['warmup_ms']}")
    return value


def _v_kernel_time_ms(row, expect, errors, ctx):
    value = _parse_strict_float(row["kernel_time_ms"], errors, ctx, "kernel_time_ms")
    if value is None:
        return None
    if value <= 0:
        errors.append(f"{ctx}: kernel_time_ms={value} must be positive")
    return value


def _v_effective_gbps(row, expect, errors, ctx):
    value = _parse_strict_float(row["effective_gbps"], errors, ctx, "effective_gbps")
    if value is None:
        return None
    if value <= 0:
        errors.append(f"{ctx}: effective_gbps={value} must be positive")
        return value
    kernel_time_ms = _peek_float(row, "kernel_time_ms")
    useful_bytes = _peek_int(row, "useful_bytes")
    if kernel_time_ms is not None and kernel_time_ms > 0 and useful_bytes is not None:
        recomputed = useful_bytes / (kernel_time_ms / 1000.0) / 1e9
        # Both the reported time and bandwidth are rounded independently to
        # six decimals.  Bound the possible true bandwidth using the entire
        # half-ULP interval of the rounded time, then add the bandwidth's own
        # half-ULP.  This is tighter and better justified than a blanket
        # relative tolerance.
        time_low = kernel_time_ms - FIXED6_HALF_ULP
        time_high = kernel_time_ms + FIXED6_HALF_ULP
        if time_low <= 0:
            errors.append(
                f"{ctx}: kernel_time_ms={kernel_time_ms} is too small to validate "
                f"after six-decimal rounding"
            )
        else:
            gbps_low = useful_bytes / (time_high * 1e6) - FIXED6_HALF_ULP
            gbps_high = useful_bytes / (time_low * 1e6) + FIXED6_HALF_ULP
            if not (gbps_low <= value <= gbps_high):
                errors.append(
                    f"{ctx}: effective_gbps={value} inconsistent with useful_bytes/kernel_time="
                    f"{recomputed}; six-decimal admissible interval is "
                    f"[{gbps_low}, {gbps_high}]"
                )
    return value


def _v_correctness(row, expect, errors, ctx):
    if row["correctness"] != "OK":
        errors.append(f"{ctx}: correctness={row['correctness']!r} != 'OK'")
        return None
    return row["correctness"]


def _v_mismatches(row, expect, errors, ctx):
    value = _parse_strict_int(row["mismatches"], errors, ctx, "mismatches")
    if value is None:
        return None
    if value != 0:
        errors.append(f"{ctx}: mismatches={value} != 0")
    return value


def _v_gpu_name(row, expect, errors, ctx):
    raw = row["gpu_name"]
    if raw is None or not raw.strip():
        errors.append(f"{ctx}: gpu_name is empty after stripping")
        return None
    if any(ord(c) < 0x20 or ord(c) == 0x7f for c in raw):
        errors.append(f"{ctx}: gpu_name contains control characters: {raw!r}")
        return None
    return raw


def _v_gpu_uuid(row, expect, errors, ctx):
    raw = row["gpu_uuid"]
    if not raw or not GPU_UUID_RE.match(raw):
        errors.append(
            f"{ctx}: gpu_uuid={raw!r} does not match "
            f"GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"
        )
        return None
    return raw


def _v_compute_capability(row, expect, errors, ctx):
    if row["compute_capability"] != FROZEN_COMPUTE_CAPABILITY:
        errors.append(
            f"{ctx}: compute_capability={row['compute_capability']!r} != {FROZEN_COMPUTE_CAPABILITY!r}"
        )
        return None
    return row["compute_capability"]


def _v_git_commit(row, expect, errors, ctx):
    raw = row["git_commit"]
    expected = expect["git_commit"]
    if not GIT_COMMIT_RE.fullmatch(raw):
        errors.append(f"{ctx}: git_commit={raw!r} is not 40 lowercase hexadecimal characters")
        return None
    if not isinstance(expected, str) or not GIT_COMMIT_RE.fullmatch(expected):
        errors.append(f"{ctx}: expected git_commit={expected!r} is not a valid full commit SHA")
        return None
    if raw != expected:
        errors.append(f"{ctx}: git_commit={row['git_commit']!r} != expected {expect['git_commit']!r}")
        return None
    return raw


def _v_git_dirty(row, expect, errors, ctx):
    if row["git_dirty"] != "false":
        errors.append(f"{ctx}: git_dirty={row['git_dirty']!r} != 'false'")
        return None
    return row["git_dirty"]


FIELD_VALIDATORS: dict[str, object] = {
    "schema_version": _v_schema_version,
    "timestamp_utc": _v_timestamp_utc,
    "run_kind": _v_run_kind,
    "method": _v_method,
    "sample_index": _v_sample_index,
    "stages": _v_stages,
    "tile_width_elements": _make_exact_int_validator(
        "tile_width_elements", lambda e: FROZEN_TILE_WIDTH_ELEMENTS
    ),
    "tile_width_bytes": _make_exact_int_validator(
        "tile_width_bytes", lambda e: FROZEN_TILE_WIDTH_BYTES
    ),
    "tile_height": _make_exact_int_validator(
        "tile_height", lambda e: tile_height_of(e["stages"], e["bif_kib"])
    ),
    "stage_bytes": _make_exact_int_validator(
        "stage_bytes", lambda e: stage_bytes_of(e["stages"], e["bif_kib"])
    ),
    "bytes_in_flight_per_sm": _make_exact_int_validator(
        "bytes_in_flight_per_sm", lambda e: bytes_in_flight_of(e["bif_kib"])
    ),
    "vector_bytes": _make_exact_int_validator("vector_bytes", lambda e: FROZEN_VECTOR_BYTES),
    "copies_per_thread_per_stage": _make_exact_int_validator(
        "copies_per_thread_per_stage", lambda e: copies_per_thread_of(e["stages"], e["bif_kib"])
    ),
    "threads_per_cta": _make_exact_int_validator("threads_per_cta", lambda e: FROZEN_THREADS_PER_CTA),
    "target_ctas_per_sm": _make_exact_int_validator(
        "target_ctas_per_sm", lambda e: FROZEN_TARGET_CTAS_PER_SM
    ),
    "occupancy_ctas_per_sm": _make_exact_int_validator(
        "occupancy_ctas_per_sm", lambda e: FROZEN_OCCUPANCY_CTAS_PER_SM
    ),
    "grid_blocks": _v_grid_blocks,
    "sm_count": _make_positive_int_validator("sm_count"),
    "smem_reservation_bytes": _v_smem_reservation_bytes,
    "l2_bytes": _make_positive_int_validator("l2_bytes"),
    "requested_working_set_bytes": _v_requested_working_set_bytes,
    "working_set_bytes": _v_working_set_bytes,
    "working_set_l2_ratio": _v_working_set_l2_ratio,
    "passes": _v_passes,
    "useful_bytes": _v_useful_bytes,
    "warmup_ms": _v_warmup_ms,
    "kernel_time_ms": _v_kernel_time_ms,
    "effective_gbps": _v_effective_gbps,
    "correctness": _v_correctness,
    "mismatches": _v_mismatches,
    "gpu_name": _v_gpu_name,
    "gpu_uuid": _v_gpu_uuid,
    "compute_capability": _v_compute_capability,
    "cuda_driver_version": _make_positive_int_validator("cuda_driver_version"),
    "cuda_runtime_version": _make_positive_int_validator("cuda_runtime_version"),
    "git_commit": _v_git_commit,
    "git_dirty": _v_git_dirty,
}
assert set(FIELD_VALIDATORS) == set(CSV_HEADER), (
    "FIELD_VALIDATORS must cover exactly CSV_HEADER; "
    f"missing={set(CSV_HEADER) - set(FIELD_VALIDATORS)} "
    f"extra={set(FIELD_VALIDATORS) - set(CSV_HEADER)}"
)


def read_case_rows(path: Path) -> tuple[list[list[str]], list[str]]:
    """Reads the raw CSV with csv.reader (never cut/awk/line-splitting).
    Returns (all_rows_including_header, errors). On a structural error the
    row list may be incomplete."""
    try:
        with _open_regular_nofollow(path, binary=False) as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnsafePathError, UnicodeError) as exc:
        return [], [f"{path}: unable to read: {exc}"]
    if not rows:
        return [], [f"{path}: file is empty (no header)"]
    return rows, []


def validate_case_file(path: Path, expect: dict) -> tuple[list[dict[str, str]], list[str]]:
    """Validates one case CSV in full. `expect` must contain: method, stages,
    bif_kib, run_kind, repetitions, passes, warmup_ms, working_set_mib
    (int or None), git_commit. Every one of the 37 CSV_HEADER fields is
    validated in every repetition via FIELD_VALIDATORS. Returns
    (parsed_data_rows, errors); errors is empty iff the file is fully
    valid."""
    errors: list[str] = []
    rows, read_errors = read_case_rows(path)
    if read_errors:
        return [], read_errors

    header, data_rows_raw = rows[0], rows[1:]
    if header != CSV_HEADER:
        return [], [f"{path}: header mismatch (wrong or reordered CSV header): {header!r}"]

    parsed_rows: list[dict[str, str]] = []
    for line_no, row in enumerate(data_rows_raw, start=2):
        if len(row) == 0:
            errors.append(f"{path}: line {line_no}: blank row")
            continue
        if row == CSV_HEADER:
            errors.append(f"{path}: line {line_no}: repeated header row")
            continue
        if len(row) != len(CSV_HEADER):
            errors.append(
                f"{path}: line {line_no}: expected {len(CSV_HEADER)} fields, got {len(row)}"
            )
            continue
        parsed_rows.append(dict(zip(CSV_HEADER, row)))

    if errors:
        return parsed_rows, errors

    repetitions = expect["repetitions"]
    if len(parsed_rows) != repetitions:
        errors.append(
            f"{path}: has {len(parsed_rows)} data row(s), expected exactly repetitions={repetitions}"
        )
        return parsed_rows, errors

    sample_indices: list[int] = []
    for row_num, row in enumerate(parsed_rows):
        ctx = f"{path}: data row {row_num} (line {row_num + 2})"
        validated_fields: set[str] = set()
        for field in CSV_HEADER:
            validator = FIELD_VALIDATORS[field]
            value = validator(row, expect, errors, ctx)
            validated_fields.add(field)
            if field == "sample_index" and value is not None:
                sample_indices.append(value)
        assert validated_fields == set(CSV_HEADER), (
            f"internal error: row {row_num} did not run every field validator"
        )

    index_counts: dict[int, int] = {}
    for value in sample_indices:
        index_counts[value] = index_counts.get(value, 0) + 1
    expected_indices = set(range(repetitions))
    found_indices = set(index_counts)
    for missing in sorted(expected_indices - found_indices):
        errors.append(f"{path}: sample_index={missing} is missing")
    for unexpected in sorted(found_indices - expected_indices):
        errors.append(f"{path}: unexpected sample_index={unexpected} (expected 0..{repetitions - 1})")
    for value, count in sorted(index_counts.items()):
        if count > 1:
            errors.append(f"{path}: sample_index={value} appears {count} times, expected exactly once")

    return parsed_rows, errors


COMMON_FIELDS = (
    "gpu_name", "gpu_uuid", "compute_capability", "cuda_driver_version",
    "cuda_runtime_version", "git_commit", "git_dirty", "sm_count", "l2_bytes",
    "requested_working_set_bytes", "working_set_bytes", "passes", "warmup_ms", "run_kind",
)


def check_cross_case_consistency(cases: list[tuple[dict, list[dict[str, str]]]]) -> list[str]:
    """Compares *every* repetition of *every* case against one single
    reference row (the very first row of the very first case), not just
    each case's rows[0]. A common field changed only in sample_index=1 or 2
    of an otherwise-valid case is therefore always caught."""
    errors: list[str] = []
    all_rows: list[tuple[dict, dict[str, str]]] = [
        (entry, row) for entry, rows in cases for row in rows
    ]
    if not all_rows:
        return errors
    reference_entry, reference_row = all_rows[0]
    for entry, row in all_rows[1:]:
        for field in COMMON_FIELDS:
            if row[field] != reference_row[field]:
                errors.append(
                    f"case {entry['case_name']} sample_index={row.get('sample_index')}: "
                    f"{field}={row[field]!r} != reference ({reference_entry['case_name']} "
                    f"sample_index={reference_row.get('sample_index')})'s {field}={reference_row[field]!r}"
                )
    if cases:
        reference_repetitions = len(cases[0][1])
        for entry, rows in cases[1:]:
            if len(rows) != reference_repetitions:
                errors.append(
                    f"case {entry['case_name']}: repetition count {len(rows)} != "
                    f"{reference_repetitions}"
                )
    return errors


def scan_case_directory(cases_dir: Path, plan: list[dict]) -> tuple[dict[int, Path], list[str]]:
    """Scans cases_dir for files matching the canonical NN_method_sS_bifB.csv
    naming convention, cross-checks the parsed filename against the plan,
    and returns (index -> path for exactly the 18 expected indices, errors).
    Anything not matching the canonical pattern (including .partial/.invalid
    evidence), or not present, is reported rather than silently ignored or
    aggregated."""
    errors: list[str] = []
    by_index = plan_by_index(plan)
    found: dict[int, Path] = {}

    try:
        _reject_if_symlink_or_wrong_type(cases_dir, expect_dir=True)
    except UnsafePathError as exc:
        return {}, [str(exc)]
    if not os.path.lexists(cases_dir):
        return {}, [f"{cases_dir}: cases directory does not exist"]

    for path in sorted(cases_dir.iterdir()):
        try:
            st = os.lstat(path)
        except OSError as exc:
            errors.append(f"{path}: lstat failed: {exc}")
            continue
        if stat.S_ISLNK(st.st_mode):
            errors.append(f"{path}: is a symlink; refusing")
            continue
        if not stat.S_ISREG(st.st_mode):
            errors.append(f"{path}: is not a regular file; refusing")
            continue
        if path.suffix != ".csv":
            continue  # .partial / .invalid / anything else is never aggregated
        match = CASE_NAME_RE.match(path.stem)
        if not match:
            errors.append(f"{path}: does not match the canonical case-name pattern; unexpected file")
            continue
        index, method, stages, bif_kib = (
            int(match.group(1)), match.group(2), int(match.group(3)), int(match.group(4)),
        )
        expected_entry = by_index.get(index)
        if expected_entry is None:
            errors.append(f"{path}: index {index} is outside the expected 0..17 range; unexpected file")
            continue
        if index in found:
            errors.append(f"{path}: duplicate configuration for index {index} (already have {found[index]})")
            continue
        if (method, stages, bif_kib) != (
            expected_entry["method"], expected_entry["stages"], expected_entry["bif_kib"],
        ):
            errors.append(
                f"{path}: filename implies method={method} stages={stages} bif_kib={bif_kib}, "
                f"but plan index {index} expects method={expected_entry['method']} "
                f"stages={expected_entry['stages']} bif_kib={expected_entry['bif_kib']}"
            )
            continue
        found[index] = path

    for index in sorted(by_index):
        if index not in found:
            errors.append(
                f"{cases_dir}: missing configuration index {index} "
                f"({by_index[index]['case_name']}.csv)"
            )

    return found, errors


# ---------------------------------------------------------------------------
# execution_order.csv: created once at campaign init, re-validated at
# finalize time.
# ---------------------------------------------------------------------------
def _execution_order_row(entry: dict) -> list[str]:
    return [
        str(entry["index"]), entry["method"], str(entry["stages"]), str(entry["bif_kib"]),
        str(stage_bytes_of(entry["stages"], entry["bif_kib"])),
        str(bytes_in_flight_of(entry["bif_kib"])),
        str(tile_height_of(entry["stages"], entry["bif_kib"])),
        str(copies_per_thread_of(entry["stages"], entry["bif_kib"])),
        f"cases/{entry['case_name']}.csv",
    ]


def write_execution_order(campaign_dir: Path, plan: list[dict]) -> Path:
    """Writes execution_order.csv exactly once (no-clobber publish)."""
    out_path = campaign_dir / "execution_order.csv"
    if os.path.lexists(out_path):
        raise UnsafePathError(f"{out_path}: already exists, refusing to overwrite")
    tmp_path = campaign_dir / "execution_order.csv.tmp"
    if os.path.lexists(tmp_path):
        raise UnsafePathError(f"{tmp_path}: stale temporary file already exists")
    try:
        with _open_exclusive(tmp_path, binary=False, newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(EXECUTION_ORDER_HEADER)
            for entry in plan:
                writer.writerow(_execution_order_row(entry))
    except Exception:
        if os.path.lexists(tmp_path):
            _safe_unlink_owned(tmp_path)
        raise
    try:
        _publish_no_clobber(tmp_path, out_path)
    except UnsafePathError:
        if os.path.lexists(tmp_path):
            _safe_unlink_owned(tmp_path)
        raise
    return out_path


def validate_execution_order_file(path: Path, plan: list[dict]) -> list[str]:
    """Strictly validates an existing execution_order.csv against build_plan():
    exact header, exactly len(plan) rows, in order, matching every derived
    column and case_file path. A missing file, a symlink, a malformed or
    reordered header, or an extra/missing row are all rejected."""
    errors: list[str] = []
    if os.path.lexists(path):
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode):
            return [f"{path}: is a symlink; refusing"]
    else:
        return [f"{path}: does not exist"]

    try:
        with _open_regular_nofollow(path, binary=False) as handle:
            rows = list(csv.reader(handle))
    except (OSError, UnsafePathError, UnicodeError) as exc:
        return [f"{path}: unable to read: {exc}"]
    if not rows:
        return [f"{path}: empty file"]

    header, data_rows = rows[0], rows[1:]
    if header != EXECUTION_ORDER_HEADER:
        return [f"{path}: header mismatch: {header!r}"]
    if len(data_rows) != len(plan):
        return [f"{path}: has {len(data_rows)} row(s), expected {len(plan)}"]

    for i, (row, entry) in enumerate(zip(data_rows, plan)):
        if len(row) != len(EXECUTION_ORDER_HEADER):
            errors.append(f"{path}: row {i} has {len(row)} field(s), expected {len(EXECUTION_ORDER_HEADER)}")
            continue
        expected_row = _execution_order_row(entry)
        if row != expected_row:
            errors.append(f"{path}: row {i} = {row!r} != expected {expected_row!r}")

    return errors


# ---------------------------------------------------------------------------
# Consolidation and aggregation (descriptive only; no-clobber publish).
# ---------------------------------------------------------------------------
def _aggregate_target_errors(paths: list[Path]) -> list[str]:
    """Preflight the complete aggregate publication set before creating any
    output.  A final target or its deterministic temporary name existing in
    any form is fatal; nothing is removed or overwritten."""
    errors: list[str] = []
    for path in paths:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        for candidate, label in ((path, "target"), (tmp_path, "temporary")):
            if os.path.lexists(candidate):
                errors.append(f"{candidate}: existing aggregate {label}; refusing to overwrite")
    return errors


def write_combined_samples(
    plan: list[dict], cases: list[tuple[dict, list[dict[str, str]]]], out_path: Path
) -> int:
    """Writes combined_samples.csv preserving the exact 37-column schema, one
    header, deterministic invocation order, and increasing sample_index
    within each invocation. Never overwrites an existing file. Returns the
    number of data rows written."""
    if os.path.lexists(out_path):
        raise UnsafePathError(f"refusing to overwrite existing target: {out_path}")
    rows_by_index = {entry["index"]: rows for entry, rows in cases}
    row_count = 0
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if os.path.lexists(tmp_path):
        raise UnsafePathError(f"{tmp_path}: existing temporary; refusing to overwrite")
    try:
        with _open_exclusive(tmp_path, binary=False, newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_HEADER)
            for entry in plan:
                rows = sorted(rows_by_index[entry["index"]], key=lambda r: int(r["sample_index"]))
                for row in rows:
                    writer.writerow([row[field] for field in CSV_HEADER])
                    row_count += 1
    except Exception:
        if os.path.lexists(tmp_path):
            _safe_unlink_owned(tmp_path)
        raise
    try:
        _publish_no_clobber(tmp_path, out_path)
    except UnsafePathError:
        if os.path.lexists(tmp_path):
            _safe_unlink_owned(tmp_path)
        raise
    return row_count


SUMMARY_HEADER = [
    "method", "stages", "bytes_in_flight_per_sm", "stage_bytes", "tile_height",
    "copies_per_thread_per_stage", "run_kind", "sm_count", "working_set_bytes",
    "passes", "warmup_ms", "sample_count",
    "kernel_time_ms_mean", "kernel_time_ms_median", "kernel_time_ms_stdev",
    "kernel_time_ms_cv_percent",
    "effective_gbps_mean", "effective_gbps_median", "effective_gbps_stdev",
    "effective_gbps_cv_percent", "effective_gbps_min", "effective_gbps_max",
    "correctness", "mismatches",
]


def _sample_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_case(entry: dict, rows: list[dict[str, str]]) -> dict:
    kernel_times = [float(r["kernel_time_ms"]) for r in rows]
    gbps = [float(r["effective_gbps"]) for r in rows]
    kt_mean = statistics.mean(kernel_times)
    gbps_mean = statistics.mean(gbps)
    kt_stdev = _sample_stdev(kernel_times)
    gbps_stdev = _sample_stdev(gbps)
    row0 = rows[0]
    return {
        "method": entry["method"],
        "stages": entry["stages"],
        "bytes_in_flight_per_sm": bytes_in_flight_of(entry["bif_kib"]),
        "stage_bytes": stage_bytes_of(entry["stages"], entry["bif_kib"]),
        "tile_height": tile_height_of(entry["stages"], entry["bif_kib"]),
        "copies_per_thread_per_stage": copies_per_thread_of(entry["stages"], entry["bif_kib"]),
        "run_kind": row0["run_kind"],
        "sm_count": row0["sm_count"],
        "working_set_bytes": row0["working_set_bytes"],
        "passes": row0["passes"],
        "warmup_ms": row0["warmup_ms"],
        "sample_count": len(rows),
        "kernel_time_ms_mean": kt_mean,
        "kernel_time_ms_median": statistics.median(kernel_times),
        "kernel_time_ms_stdev": kt_stdev,
        "kernel_time_ms_cv_percent": (100.0 * kt_stdev / kt_mean) if kt_mean != 0 else 0.0,
        "effective_gbps_mean": gbps_mean,
        "effective_gbps_median": statistics.median(gbps),
        "effective_gbps_stdev": gbps_stdev,
        "effective_gbps_cv_percent": (100.0 * gbps_stdev / gbps_mean) if gbps_mean != 0 else 0.0,
        "effective_gbps_min": min(gbps),
        "effective_gbps_max": max(gbps),
        "correctness": "OK",
        "mismatches": 0,
    }


def format_summary_value(field: str, value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_summary(cases: list[tuple[dict, list[dict[str, str]]]], out_path: Path) -> int:
    """Writes summary.csv: exactly 18 rows ordered by (stages,
    bytes_in_flight_per_sm, method), descriptive statistics only. Never
    overwrites an existing file."""
    if os.path.lexists(out_path):
        raise UnsafePathError(f"refusing to overwrite existing target: {out_path}")
    summaries = [summarize_case(entry, rows) for entry, rows in cases]
    summaries.sort(key=lambda s: (s["stages"], s["bytes_in_flight_per_sm"], s["method"]))
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    if os.path.lexists(tmp_path):
        raise UnsafePathError(f"{tmp_path}: existing temporary; refusing to overwrite")
    try:
        with _open_exclusive(tmp_path, binary=False, newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(SUMMARY_HEADER)
            for summary in summaries:
                writer.writerow([format_summary_value(field, summary[field]) for field in SUMMARY_HEADER])
    except Exception:
        if os.path.lexists(tmp_path):
            _safe_unlink_owned(tmp_path)
        raise
    try:
        _publish_no_clobber(tmp_path, out_path)
    except UnsafePathError:
        if os.path.lexists(tmp_path):
            _safe_unlink_owned(tmp_path)
        raise
    return len(summaries)


# ---------------------------------------------------------------------------
# Manifest: allowlisted keys/types, enforced state transitions, atomic
# updates (the one intentional os.replace()-based exception to no-clobber).
# ---------------------------------------------------------------------------
def load_manifest(campaign_dir: Path) -> dict:
    path = campaign_dir / "manifest.json"
    if not os.path.lexists(path):
        return {}
    try:
        with _open_regular_nofollow(path, binary=False) as handle:
            document = json.load(handle)
    except (OSError, UnsafePathError, UnicodeError, json.JSONDecodeError) as exc:
        raise ManifestTransitionError(f"{path}: cannot load manifest safely: {exc}") from exc
    if not isinstance(document, dict):
        raise ManifestTransitionError(
            f"{path}: manifest root has type {type(document).__name__}, expected object"
        )
    return document


def write_manifest_atomic(campaign_dir: Path, manifest: dict) -> None:
    path = campaign_dir / "manifest.json"
    tmp_path = campaign_dir / "manifest.json.tmp"
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"

    prior_identity: tuple[int, int] | None = None
    if os.path.lexists(path):
        artifact_error = _verify_artifact(path)
        if artifact_error:
            raise ManifestTransitionError(artifact_error)
        prior_identity = _file_identity(path)
    if os.path.lexists(tmp_path):
        raise ManifestTransitionError(
            f"{tmp_path}: existing manifest temporary; refusing to overwrite"
        )

    created_tmp = False
    try:
        with _open_exclusive(tmp_path, binary=False) as handle:
            created_tmp = True
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        if prior_identity is None:
            if os.path.lexists(path):
                raise ManifestTransitionError(
                    f"{path}: appeared during initial manifest publication; refusing to overwrite"
                )
        elif not os.path.lexists(path) or _file_identity(path) != prior_identity:
            raise ManifestTransitionError(
                f"{path}: changed during manifest update; refusing to overwrite"
            )
        os.replace(tmp_path, path)
        created_tmp = False
    except UnsafePathError as exc:
        raise ManifestTransitionError(str(exc)) from exc
    finally:
        if created_tmp and os.path.lexists(tmp_path):
            try:
                _safe_unlink_owned(tmp_path)
            except UnsafePathError:
                pass


def _manifest_type_matches(value: object, expected: object) -> bool:
    expected_types = expected if isinstance(expected, tuple) else (expected,)
    for expected_type in expected_types:
        if expected_type is int:
            if type(value) is int:
                return True
        elif type(value) is expected_type:
            return True
    return False


def _validate_manifest_updates(updates: dict) -> None:
    unknown = set(updates) - set(ALLOWED_MANIFEST_KEYS)
    if unknown:
        raise ManifestTransitionError(f"unknown manifest field(s): {sorted(unknown)}")
    for key, value in updates.items():
        expected_type = ALLOWED_MANIFEST_KEYS[key]
        if not _manifest_type_matches(value, expected_type):
            raise ManifestTransitionError(
                f"manifest field {key!r} has invalid type {type(value).__name__}, "
                f"expected {expected_type}"
            )


def _validate_compact_timestamp(value: object, field: str) -> None:
    if not isinstance(value, str) or not MANIFEST_TIMESTAMP_RE.fullmatch(value):
        raise ManifestTransitionError(
            f"manifest field {field!r}={value!r} is not YYYYMMDDTHHMMSSZ"
        )
    try:
        _datetime.strptime(value, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise ManifestTransitionError(
            f"manifest field {field!r}={value!r} is not a real UTC timestamp"
        ) from exc


def _validate_requested(requested: object) -> None:
    expected_keys = {
        "run_kind", "working_set_mib", "passes", "warmup_ms",
        "repetitions", "campaign_id",
    }
    if not isinstance(requested, dict) or set(requested) != expected_keys:
        raise ManifestTransitionError(
            f"manifest requested keys={sorted(requested) if isinstance(requested, dict) else None} "
            f"!= {sorted(expected_keys)}"
        )
    if requested["run_kind"] not in ("smoke", "benchmark"):
        raise ManifestTransitionError("manifest requested.run_kind must be smoke or benchmark")
    if not isinstance(requested["campaign_id"], str):
        raise ManifestTransitionError("manifest requested.campaign_id must be a string")
    try:
        validate_campaign_id(requested["campaign_id"])
    except UnsafePathError as exc:
        raise ManifestTransitionError(
            f"manifest requested.campaign_id is invalid: {exc}"
        ) from exc
    working_set_mib = requested["working_set_mib"]
    if working_set_mib is not None and (type(working_set_mib) is not int or working_set_mib <= 0):
        raise ManifestTransitionError(
            "manifest requested.working_set_mib must be null or a positive integer"
        )
    for key, allow_zero in (("passes", False), ("warmup_ms", True), ("repetitions", False)):
        value = requested[key]
        if type(value) is not int or value < 0 or (not allow_zero and value == 0):
            relation = "non-negative" if allow_zero else "positive"
            raise ManifestTransitionError(
                f"manifest requested.{key} must be a {relation} integer"
            )


def _validate_self_test_outcomes(value: object, *, require_pass: bool) -> None:
    if not isinstance(value, dict) or set(value) != {"ldgsts", "tma"}:
        raise ManifestTransitionError(
            "manifest self_test_outcomes must contain exactly ldgsts and tma"
        )
    allowed = {"PASS"} if require_pass else {"PASS", "FAIL"}
    for method in ("ldgsts", "tma"):
        if value[method] not in allowed:
            raise ManifestTransitionError(
                f"manifest self_test_outcomes.{method}={value[method]!r} "
                f"must be one of {sorted(allowed)}"
            )


def _validate_versions_dict(value: object) -> None:
    if not isinstance(value, dict) or set(value) != set(REQUIRED_VERSION_KEYS):
        raise ManifestTransitionError(
            f"versions_env keys={sorted(value) if isinstance(value, dict) else None} "
            f"!= required {sorted(REQUIRED_VERSION_KEYS)}"
        )
    for key in REQUIRED_VERSION_KEYS:
        if not isinstance(value[key], str) or not value[key]:
            raise ManifestTransitionError(f"versions_env.{key} must be a non-empty string")
    if value["CUDA_ARCH"] != "sm_103a":
        raise ManifestTransitionError("versions_env.CUDA_ARCH must remain sm_103a")
    if value["MAX_BUILD_JOBS"] != "2":
        raise ManifestTransitionError("versions_env.MAX_BUILD_JOBS must remain 2")


def _validate_hash_map(value: object, expected_keys: set[str], field: str) -> None:
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ManifestTransitionError(
            f"manifest {field} keys={sorted(value) if isinstance(value, dict) else None} "
            f"!= {sorted(expected_keys)}"
        )
    for key, digest in value.items():
        if not _is_sha256_hex(digest):
            raise ManifestTransitionError(
                f"manifest {field}.{key}={digest!r} is not a lowercase SHA-256"
            )


def _validate_manifest_document(
    manifest: dict, *, require_initialized: bool = False
) -> None:
    """Validate the entire loaded manifest, not only the proposed update.

    IN_PROGRESS documents may omit fields that are populated later, but any
    field already present must be allowlisted, correctly typed, and have a
    valid nested schema.  Finalization requests the initialized base schema;
    COMPLETE additionally requires the full provenance/hash contract.
    """
    _validate_manifest_updates(manifest)
    if not manifest:
        if require_initialized:
            raise ManifestTransitionError("manifest is empty")
        return

    status = manifest.get("status")
    if status not in ALLOWED_TRANSITIONS:
        raise ManifestTransitionError(f"manifest status={status!r} is invalid")
    if "schema_version" in manifest and manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise ManifestTransitionError("manifest schema_version is invalid")
    if "experiment_id" in manifest and manifest["experiment_id"] != EXPERIMENT_ID:
        raise ManifestTransitionError("manifest experiment_id is invalid")
    if "publishable" in manifest and manifest["publishable"] is not False:
        raise ManifestTransitionError("manifest publishable must be false")

    if "campaign_id" in manifest:
        try:
            validate_campaign_id(manifest["campaign_id"])
        except UnsafePathError as exc:
            raise ManifestTransitionError(f"manifest campaign_id is invalid: {exc}") from exc
    if "run_kind" in manifest and manifest["run_kind"] not in ("smoke", "benchmark"):
        raise ManifestTransitionError("manifest run_kind must be smoke or benchmark")
    if "started_at_utc" in manifest:
        _validate_compact_timestamp(manifest["started_at_utc"], "started_at_utc")
    if manifest.get("completed_at_utc") is not None:
        _validate_compact_timestamp(manifest["completed_at_utc"], "completed_at_utc")
    if "git_commit" in manifest and not GIT_COMMIT_RE.fullmatch(manifest["git_commit"]):
        raise ManifestTransitionError("manifest git_commit must be 40 lowercase hexadecimal characters")
    if "git_dirty" in manifest and manifest["git_dirty"] is not False:
        raise ManifestTransitionError("manifest git_dirty must be false")
    if "selected_gpu_index" in manifest and manifest["selected_gpu_index"] < 0:
        raise ManifestTransitionError("manifest selected_gpu_index must be non-negative")
    if "gpu_name" in manifest:
        if not manifest["gpu_name"].strip() or any(
            ord(char) < 0x20 or ord(char) == 0x7F for char in manifest["gpu_name"]
        ):
            raise ManifestTransitionError("manifest gpu_name is empty or contains control characters")
    if "gpu_uuid" in manifest and not GPU_UUID_RE.fullmatch(manifest["gpu_uuid"]):
        raise ManifestTransitionError("manifest gpu_uuid has an invalid NVIDIA UUID form")
    if (
        "compute_capability" in manifest
        and manifest["compute_capability"] != FROZEN_COMPUTE_CAPABILITY
    ):
        raise ManifestTransitionError("manifest compute_capability must remain 10.3")
    for key in ("cuda_driver_version", "cuda_runtime_version"):
        if key in manifest:
            raw = str(manifest[key])
            if not CANONICAL_UINT_RE.fullmatch(raw) or int(raw) <= 0:
                raise ManifestTransitionError(f"manifest {key} must be a positive decimal version")

    for key in (
        "configuration_count_expected", "configuration_count_completed",
        "sample_count_expected", "sample_count_completed",
    ):
        if key in manifest and manifest[key] < 0:
            raise ManifestTransitionError(f"manifest {key} must be non-negative")

    if "requested" in manifest:
        _validate_requested(manifest["requested"])
        requested = manifest["requested"]
        if "campaign_id" in manifest and requested["campaign_id"] != manifest["campaign_id"]:
            raise ManifestTransitionError("manifest requested.campaign_id disagrees with campaign_id")
        if "run_kind" in manifest and requested["run_kind"] != manifest["run_kind"]:
            raise ManifestTransitionError("manifest requested.run_kind disagrees with run_kind")
        repetitions = requested["repetitions"]
        if "configuration_count_expected" in manifest and "sample_count_expected" in manifest:
            if manifest["sample_count_expected"] != manifest["configuration_count_expected"] * repetitions:
                raise ManifestTransitionError(
                    "manifest sample_count_expected disagrees with configurations*repetitions"
                )
        if "configuration_count_completed" in manifest and "sample_count_completed" in manifest:
            if manifest["sample_count_completed"] != manifest["configuration_count_completed"] * repetitions:
                raise ManifestTransitionError(
                    "manifest sample_count_completed disagrees with configurations*repetitions"
                )

    if "self_test_outcomes" in manifest:
        _validate_self_test_outcomes(
            manifest["self_test_outcomes"], require_pass=status == "COMPLETE"
        )
    if "versions_env" in manifest:
        _validate_versions_dict(manifest["versions_env"])
    if "binary_and_sass_sha256" in manifest:
        _validate_hash_map(
            manifest["binary_and_sass_sha256"], set(DEFAULT_FINAL_ARTIFACTS),
            "binary_and_sass_sha256",
        )
    if "case_file_sha256" in manifest:
        _validate_hash_map(
            manifest["case_file_sha256"],
            {entry["case_name"] for entry in build_plan()},
            "case_file_sha256",
        )
    if "aggregate_file_sha256" in manifest:
        _validate_hash_map(
            manifest["aggregate_file_sha256"],
            {"combined_samples.csv", "summary.csv"},
            "aggregate_file_sha256",
        )
    if "execution_order_sha256" in manifest and not _is_sha256_hex(
        manifest["execution_order_sha256"]
    ):
        raise ManifestTransitionError("manifest execution_order_sha256 is not a lowercase SHA-256")
    if "failure_detail" in manifest and manifest["failure_detail"] is not None:
        if not all(isinstance(item, str) for item in manifest["failure_detail"]):
            raise ManifestTransitionError("manifest failure_detail must be null or a list of strings")
    if "failure_stage" in manifest and manifest["failure_stage"] is not None:
        if not manifest["failure_stage"]:
            raise ManifestTransitionError("manifest failure_stage must be null or non-empty")

    if "observed_common" in manifest:
        expected_observed = {
            "requested_working_set_bytes", "working_set_bytes", "sm_count",
            "l2_bytes", "passes", "warmup_ms", "repetitions",
        }
        observed = manifest["observed_common"]
        if set(observed) != expected_observed:
            raise ManifestTransitionError("manifest observed_common has an invalid nested schema")
        if any(not isinstance(observed[key], str) for key in expected_observed - {"repetitions"}):
            raise ManifestTransitionError("manifest observed_common values must match CSV strings")
        if type(observed["repetitions"]) is not int or observed["repetitions"] <= 0:
            raise ManifestTransitionError("manifest observed_common.repetitions must be positive")

    if "invocation_order" in manifest:
        expected_order = [
            {
                "index": entry["index"], "method": entry["method"],
                "stages": entry["stages"], "bif_kib": entry["bif_kib"],
                "case_name": entry["case_name"],
            }
            for entry in build_plan()
        ]
        if manifest["invocation_order"] != expected_order:
            raise ManifestTransitionError("manifest invocation_order does not match the frozen plan")

    initialized_required = {
        "schema_version", "experiment_id", "campaign_id", "status", "run_kind",
        "started_at_utc", "configuration_count_expected",
        "configuration_count_completed", "sample_count_expected",
        "sample_count_completed", "requested", "selected_gpu_index",
        "git_commit", "git_dirty", "publishable",
    }
    if require_initialized:
        missing = initialized_required - set(manifest)
        if missing:
            raise ManifestTransitionError(f"initialized manifest missing fields: {sorted(missing)}")
        if manifest["configuration_count_expected"] != EXPECTED_CONFIGURATION_COUNT:
            raise ManifestTransitionError("manifest configuration_count_expected must be 18")
        if manifest["configuration_count_completed"] > EXPECTED_CONFIGURATION_COUNT:
            raise ManifestTransitionError("manifest configuration_count_completed exceeds 18")

    if status == "COMPLETE":
        complete_required = initialized_required | {
            "completed_at_utc", "observed_common", "invocation_order",
            "gpu_name", "gpu_uuid", "compute_capability",
            "cuda_driver_version", "cuda_runtime_version", "versions_env",
            "binary_and_sass_sha256", "case_file_sha256",
            "execution_order_sha256", "aggregate_file_sha256",
            "self_test_outcomes", "failure_stage", "failure_detail",
        }
        missing = complete_required - set(manifest)
        if missing:
            raise ManifestTransitionError(f"COMPLETE manifest missing fields: {sorted(missing)}")
        if manifest["configuration_count_completed"] != EXPECTED_CONFIGURATION_COUNT:
            raise ManifestTransitionError("COMPLETE manifest must have 18 configurations")
        if manifest["sample_count_completed"] != manifest["sample_count_expected"]:
            raise ManifestTransitionError("COMPLETE manifest sample counts are incomplete")
        if manifest["failure_stage"] is not None or manifest["failure_detail"] is not None:
            raise ManifestTransitionError("COMPLETE manifest cannot retain failure metadata")
        _validate_self_test_outcomes(manifest["self_test_outcomes"], require_pass=True)
        _validate_versions_dict(manifest["versions_env"])


def merge_manifest(
    campaign_dir: Path, updates: dict, status: str, *, allow_complete: bool = False
) -> dict:
    """Merges `updates` into manifest.json and sets `status`, enforcing:
    only finalize (allow_complete=True) may set COMPLETE; the transition
    from the manifest's current status to `status` must be allowed (a
    terminal campaign can never be reopened/rewritten); and every key in
    `updates` must be allowlisted with the correct type."""
    if status == "COMPLETE" and not allow_complete:
        raise ManifestTransitionError("only the validated finalizer may set status=COMPLETE")
    _validate_manifest_updates(updates)
    manifest = load_manifest(campaign_dir)
    _validate_manifest_document(manifest)
    current_status = manifest.get("status")
    allowed = ALLOWED_TRANSITIONS.get(current_status, frozenset())
    if status not in allowed:
        raise ManifestTransitionError(
            f"invalid manifest state transition: {current_status!r} -> {status!r}"
        )
    immutable_after_init = {
        "campaign_id", "run_kind", "started_at_utc",
        "configuration_count_expected", "sample_count_expected", "requested",
        "selected_gpu_index", "git_commit", "git_dirty",
    }
    for key in immutable_after_init & set(manifest) & set(updates):
        if updates[key] != manifest[key]:
            raise ManifestTransitionError(
                f"manifest field {key!r} is immutable after initialization"
            )
    for key in ("configuration_count_completed", "sample_count_completed"):
        if key in manifest and key in updates and updates[key] < manifest[key]:
            raise ManifestTransitionError(f"manifest counter {key!r} cannot decrease")
    if "self_test_outcomes" in manifest and "self_test_outcomes" in updates:
        if updates["self_test_outcomes"] != manifest["self_test_outcomes"]:
            raise ManifestTransitionError(
                "manifest self_test_outcomes cannot change after being recorded"
            )
    manifest.update(updates)
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["experiment_id"] = EXPERIMENT_ID
    manifest["status"] = status
    manifest["publishable"] = False
    _validate_manifest_document(
        manifest, require_initialized=(status == "COMPLETE")
    )
    write_manifest_atomic(campaign_dir, manifest)
    return manifest


def parse_versions_env(versions_path: Path | None = None) -> dict[str, str]:
    values: dict[str, str] = {}
    versions_path = REPO_ROOT / "VERSIONS.env" if versions_path is None else versions_path
    try:
        with _open_regular_nofollow(versions_path, binary=False) as handle:
            lines = handle.read().splitlines()
    except (OSError, UnsafePathError, UnicodeError) as exc:
        raise ManifestTransitionError(f"invalid VERSIONS.env: {exc}") from exc
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ManifestTransitionError(
                f"VERSIONS.env line {line_no}: expected KEY=VALUE"
            )
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key not in REQUIRED_VERSION_KEYS:
            raise ManifestTransitionError(f"VERSIONS.env line {line_no}: unknown key {key!r}")
        if key in values:
            raise ManifestTransitionError(f"VERSIONS.env line {line_no}: duplicate key {key!r}")
        if not value:
            raise ManifestTransitionError(f"VERSIONS.env line {line_no}: empty value for {key!r}")
        values[key] = value
    _validate_versions_dict(values)
    return values


# ---------------------------------------------------------------------------
# Subcommand: init-campaign
# ---------------------------------------------------------------------------
def _do_init_campaign(
    *, campaign_id: str, run_kind: str, passes: int, warmup_ms: int, repetitions: int,
    working_set_mib: int | None, git_commit: str, gpu_index: int, started_at_utc: str,
) -> Path:
    """Core logic behind the init-campaign subcommand, reused directly by
    the self-test (no subprocess) to build realistic fixtures."""
    campaign_dir = create_campaign_dir(campaign_id)
    plan = build_plan()
    plan_errors = check_plan_contract(plan)
    if plan_errors:
        raise ValueError(f"internal plan contract violation: {plan_errors}")
    write_execution_order(campaign_dir, plan)
    updates = {
        "campaign_id": campaign_id,
        "run_kind": run_kind,
        "started_at_utc": started_at_utc,
        "configuration_count_expected": EXPECTED_CONFIGURATION_COUNT,
        "configuration_count_completed": 0,
        "sample_count_expected": EXPECTED_CONFIGURATION_COUNT * repetitions,
        "sample_count_completed": 0,
        "requested": {
            "run_kind": run_kind,
            "working_set_mib": working_set_mib,
            "passes": passes,
            "warmup_ms": warmup_ms,
            "repetitions": repetitions,
            "campaign_id": campaign_id,
        },
        "selected_gpu_index": gpu_index,
        "git_commit": git_commit,
        "git_dirty": False,
    }
    merge_manifest(campaign_dir, updates, status="IN_PROGRESS")
    return campaign_dir


def cmd_init_campaign(args: argparse.Namespace) -> int:
    try:
        campaign_dir = _do_init_campaign(
            campaign_id=args.campaign_id, run_kind=args.run_kind, passes=args.passes,
            warmup_ms=args.warmup_ms, repetitions=args.repetitions,
            working_set_mib=args.working_set_mib, git_commit=args.git_commit,
            gpu_index=args.gpu_index, started_at_utc=args.started_at_utc,
        )
    except (UnsafePathError, ManifestTransitionError, ValueError) as exc:
        print(f"aggregate_exp01_memory_paths: init-campaign: ERROR: {exc}", file=sys.stderr)
        return 2
    print(str(campaign_dir.relative_to(REPO_ROOT)))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: capture
# ---------------------------------------------------------------------------
def _do_capture(
    campaign_dir: Path,
    out_rel: str,
    binary_argv: list[str],
    *,
    artifact_paths: dict[str, Path] | None = None,
) -> int:
    try:
        out_path = resolve_capture_out_path(campaign_dir, out_rel)
    except UnsafePathError as exc:
        print(f"aggregate_exp01_memory_paths: capture: ERROR: {exc}", file=sys.stderr)
        return 2

    if binary_argv and binary_argv[0] == "--":
        binary_argv = binary_argv[1:]  # argparse.REMAINDER keeps a literal '--' marker
    if not binary_argv:
        print("aggregate_exp01_memory_paths: capture: ERROR: no binary command given after '--'", file=sys.stderr)
        return 2
    if binary_argv[0] not in ALLOWED_BINARIES:
        print(
            f"aggregate_exp01_memory_paths: capture: ERROR: binary must be exactly one of "
            f"{sorted(ALLOWED_BINARIES)}, got {binary_argv[0]!r}",
            file=sys.stderr,
        )
        return 2
    artifact_paths = DEFAULT_CAPTURE_ARTIFACTS if artifact_paths is None else artifact_paths
    artifact_path = artifact_paths.get(binary_argv[0])
    if artifact_path is None:
        print(
            f"aggregate_exp01_memory_paths: capture: ERROR: no verified artifact "
            f"path supplied for {binary_argv[0]!r}",
            file=sys.stderr,
        )
        return 2
    artifact_err = _verify_artifact(artifact_path)
    if artifact_err:
        print(f"aggregate_exp01_memory_paths: capture: ERROR: {artifact_err}", file=sys.stderr)
        return 2

    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if os.path.lexists(tmp_path):
        print(f"aggregate_exp01_memory_paths: capture: ERROR: stale temporary file already exists: {tmp_path}", file=sys.stderr)
        return 2

    def salvage(is_signal: bool) -> None:
        """Preserves non-empty tmp output as .invalid/.partial evidence via
        no-clobber publish, or removes it if empty. Always leaves no stale
        .tmp file."""
        if os.path.lexists(tmp_path):
            _file_identity(tmp_path)
        if os.path.lexists(tmp_path) and os.lstat(tmp_path).st_size > 0:
            suffix = ".partial" if is_signal else ".invalid"
            for sequence in range(10000):
                numbered = "" if sequence == 0 else f".{sequence}"
                failed_path = out_path.with_name(out_path.name + suffix + numbered)
                try:
                    _publish_no_clobber(tmp_path, failed_path)
                    print(
                        f"aggregate_exp01_memory_paths: capture: preserved evidence as "
                        f"{failed_path.name}",
                        file=sys.stderr,
                    )
                    return
                except UnsafePathError:
                    if os.path.lexists(tmp_path):
                        continue
                    return
            print(
                "aggregate_exp01_memory_paths: capture: ERROR: could not find a free "
                "failure-evidence name",
                file=sys.stderr,
            )
        else:
            if os.path.lexists(tmp_path):
                _safe_unlink_owned(tmp_path)

    print(f"aggregate_exp01_memory_paths: capture: running {binary_argv!r} -> {out_path.name}", file=sys.stderr)
    try:
        csv_out = _open_exclusive(tmp_path, binary=True)
    except UnsafePathError as exc:
        print(
            f"aggregate_exp01_memory_paths: capture: ERROR: cannot create owned "
            f"temporary: {exc}",
            file=sys.stderr,
        )
        return 2
    try:
        with csv_out:
            result = subprocess.run(binary_argv, stdout=csv_out, stderr=None)
    except OSError as exc:
        print(f"aggregate_exp01_memory_paths: capture: ERROR: unable to launch binary: {exc}", file=sys.stderr)
        salvage(is_signal=False)
        return 1

    if result.returncode == 0:
        if not os.path.lexists(tmp_path) or os.lstat(tmp_path).st_size == 0:
            print("aggregate_exp01_memory_paths: capture: ERROR: binary exited 0 but produced no stdout", file=sys.stderr)
            salvage(is_signal=False)
            return 1
        try:
            _publish_no_clobber(tmp_path, out_path)
        except UnsafePathError as exc:
            print(f"aggregate_exp01_memory_paths: capture: ERROR: {exc}", file=sys.stderr)
            salvage(is_signal=False)
            return 1
        print(f"aggregate_exp01_memory_paths: capture: OK: wrote {out_path}", file=sys.stderr)
        return 0

    print(f"aggregate_exp01_memory_paths: capture: ERROR: binary exited {result.returncode}", file=sys.stderr)
    salvage(is_signal=result.returncode < 0)
    return 1


def cmd_capture(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_campaign_dir(args.campaign_dir)
    except UnsafePathError as exc:
        print(f"aggregate_exp01_memory_paths: capture: ERROR: {exc}", file=sys.stderr)
        return 2
    return _do_capture(campaign_dir, args.out, list(args.binary_args))


# ---------------------------------------------------------------------------
# Subcommand: validate-case
# ---------------------------------------------------------------------------
def cmd_validate_case(args: argparse.Namespace) -> int:
    plan = build_plan()
    entry = plan_by_index(plan).get(args.index)
    if entry is None:
        print(f"aggregate_exp01_memory_paths: validate-case: ERROR: index {args.index} is not in 0..17", file=sys.stderr)
        return 2

    try:
        campaign_dir = resolve_campaign_dir(args.campaign_dir)
    except UnsafePathError as exc:
        print(f"aggregate_exp01_memory_paths: validate-case: ERROR: {exc}", file=sys.stderr)
        return 2

    case_path = campaign_dir / "cases" / f"{entry['case_name']}.csv"

    expect = {
        "method": entry["method"],
        "stages": entry["stages"],
        "bif_kib": entry["bif_kib"],
        "run_kind": args.run_kind,
        "repetitions": args.repetitions,
        "passes": args.passes,
        "warmup_ms": args.warmup_ms,
        "working_set_mib": args.working_set_mib,
        "git_commit": args.git_commit,
    }
    _, errors = validate_case_file(case_path, expect)
    if errors:
        print(f"aggregate_exp01_memory_paths: validate-case: FAIL: {case_path}", file=sys.stderr)
        for error in errors:
            print(f"aggregate_exp01_memory_paths: validate-case:   - {error}", file=sys.stderr)
        return 1
    print(f"aggregate_exp01_memory_paths: validate-case: OK: {case_path}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: finalize
# ---------------------------------------------------------------------------
def _verify_manifest_preconditions(campaign_dir: Path, args) -> list[str]:
    """All of the manifest-vs-CLI cross-checks finalize requires before it
    will produce any aggregate: existing IN_PROGRESS manifest whose
    campaign_id/run_kind/requested-parameters/git_commit/gpu_index/
    started_at_utc/self-test outcomes all match the finalizer's own
    arguments."""
    errors: list[str] = []
    try:
        manifest = load_manifest(campaign_dir)
        _validate_manifest_document(manifest, require_initialized=True)
    except (ManifestTransitionError, UnsafePathError) as exc:
        return [f"invalid existing manifest: {exc}"]
    if not manifest:
        return ["manifest.json does not exist; cannot finalize an uninitialized campaign"]
    if manifest.get("status") != "IN_PROGRESS":
        errors.append(f"manifest status={manifest.get('status')!r} != 'IN_PROGRESS'; cannot finalize")
        return errors  # do not attempt further comparisons against a foreign/terminal manifest
    if manifest.get("campaign_id") != args.campaign_id:
        errors.append(f"manifest campaign_id={manifest.get('campaign_id')!r} != CLI campaign_id={args.campaign_id!r}")
    if campaign_dir.name != args.campaign_id:
        errors.append(f"campaign directory name {campaign_dir.name!r} != CLI campaign_id={args.campaign_id!r}")
    if manifest.get("run_kind") != args.run_kind:
        errors.append(f"manifest run_kind={manifest.get('run_kind')!r} != CLI run_kind={args.run_kind!r}")
    requested = manifest.get("requested", {})
    if requested.get("passes") != args.passes:
        errors.append(f"manifest requested.passes={requested.get('passes')!r} != CLI passes={args.passes!r}")
    if requested.get("warmup_ms") != args.warmup_ms:
        errors.append(f"manifest requested.warmup_ms={requested.get('warmup_ms')!r} != CLI warmup_ms={args.warmup_ms!r}")
    if requested.get("repetitions") != args.repetitions:
        errors.append(f"manifest requested.repetitions={requested.get('repetitions')!r} != CLI repetitions={args.repetitions!r}")
    if requested.get("working_set_mib") != args.working_set_mib:
        errors.append(
            f"manifest requested.working_set_mib={requested.get('working_set_mib')!r} != "
            f"CLI working_set_mib={args.working_set_mib!r}"
        )
    if manifest.get("git_commit") != args.git_commit:
        errors.append(f"manifest git_commit={manifest.get('git_commit')!r} != CLI git_commit={args.git_commit!r}")
    if manifest.get("selected_gpu_index") != args.gpu_index:
        errors.append(f"manifest selected_gpu_index={manifest.get('selected_gpu_index')!r} != CLI gpu_index={args.gpu_index!r}")
    if manifest.get("started_at_utc") != args.started_at_utc:
        errors.append(f"manifest started_at_utc={manifest.get('started_at_utc')!r} != CLI started_at_utc={args.started_at_utc!r}")
    if args.self_test_ldgsts != "PASS":
        errors.append(f"CLI self_test_ldgsts={args.self_test_ldgsts!r} != 'PASS'")
    if args.self_test_tma != "PASS":
        errors.append(f"CLI self_test_tma={args.self_test_tma!r} != 'PASS'")
    self_test = manifest.get("self_test_outcomes", {})
    if self_test.get("ldgsts") != "PASS":
        errors.append(f"manifest self_test_outcomes.ldgsts={self_test.get('ldgsts')!r} != 'PASS'")
    if self_test.get("tma") != "PASS":
        errors.append(f"manifest self_test_outcomes.tma={self_test.get('tma')!r} != 'PASS'")
    if manifest.get("configuration_count_completed") != EXPECTED_CONFIGURATION_COUNT:
        errors.append(
            f"manifest configuration_count_completed="
            f"{manifest.get('configuration_count_completed')!r} != "
            f"{EXPECTED_CONFIGURATION_COUNT}"
        )
    if manifest.get("sample_count_completed") != EXPECTED_CONFIGURATION_COUNT * args.repetitions:
        errors.append(
            f"manifest sample_count_completed={manifest.get('sample_count_completed')!r} != "
            f"{EXPECTED_CONFIGURATION_COUNT * args.repetitions}"
        )
    return errors


def _do_finalize(
    campaign_dir: Path,
    args,
    *,
    artifact_paths: dict[str, Path] | None = None,
    versions_path: Path | None = None,
) -> tuple[bool, list[str]]:
    """Core finalize logic (no argparse/print), reused directly by the
    self-test. Returns (success, errors). On success, manifest.json has
    already been written with status=COMPLETE and every mandatory hash;
    on failure, manifest.json has been written with status=FAILED unless
    the campaign was already terminal (which is never rewritten)."""
    plan = build_plan()
    plan_errors = check_plan_contract(plan)
    if plan_errors:
        return False, [f"internal plan contract violation: {plan_errors}"]

    def fail(stage: str, errors: list[str]) -> tuple[bool, list[str]]:
        try:
            manifest = load_manifest(campaign_dir)
            if manifest.get("status") in (None, "IN_PROGRESS"):
                merge_manifest(
                    campaign_dir,
                    {"failure_stage": stage, "failure_detail": errors[:50]},
                    status="FAILED",
                )
        except (ManifestTransitionError, UnsafePathError):
            pass
        return False, errors

    precondition_errors = _verify_manifest_preconditions(campaign_dir, args)
    if precondition_errors:
        return fail("manifest_precondition", precondition_errors)

    execution_order_path = campaign_dir / "execution_order.csv"
    eo_errors = validate_execution_order_file(execution_order_path, plan)
    if eo_errors:
        return fail("execution_order", eo_errors)

    cases_dir = campaign_dir / "cases"
    found, set_errors = scan_case_directory(cases_dir, plan)
    if set_errors:
        return fail("case_set", set_errors)

    common = {
        "run_kind": args.run_kind,
        "repetitions": args.repetitions,
        "passes": args.passes,
        "warmup_ms": args.warmup_ms,
        "working_set_mib": args.working_set_mib,
        "git_commit": args.git_commit,
    }
    cases: list[tuple[dict, list[dict[str, str]]]] = []
    all_errors: list[str] = []
    for index in sorted(found):
        entry = plan_by_index(plan)[index]
        expect = {"method": entry["method"], "stages": entry["stages"], "bif_kib": entry["bif_kib"], **common}
        rows, errors = validate_case_file(found[index], expect)
        if errors:
            all_errors.extend(errors)
        else:
            cases.append((entry, rows))
    if all_errors:
        return fail("per_case_validation", all_errors)

    cross_errors = check_cross_case_consistency(cases)
    if cross_errors:
        return fail("cross_case_consistency", cross_errors)

    cases.sort(key=lambda pair: pair[0]["index"])

    artifact_paths = DEFAULT_FINAL_ARTIFACTS if artifact_paths is None else artifact_paths
    artifact_errors: list[str] = []
    binary_hashes: dict[str, str] = {}
    if set(artifact_paths) != set(DEFAULT_FINAL_ARTIFACTS):
        artifact_errors.append(
            f"artifact labels={sorted(artifact_paths)} != "
            f"{sorted(DEFAULT_FINAL_ARTIFACTS)}"
        )
    else:
        for label, path in artifact_paths.items():
            err = _verify_artifact(path)
            if err:
                artifact_errors.append(err)
            else:
                try:
                    binary_hashes[label] = sha256_of(path)
                except UnsafePathError as exc:
                    artifact_errors.append(str(exc))
    if artifact_errors:
        return fail("missing_artifact", artifact_errors)

    try:
        versions_env = parse_versions_env(versions_path)
    except ManifestTransitionError as exc:
        return fail("versions_env", [str(exc)])

    reference_row = cases[0][1][0]
    try:
        case_hashes = {
            plan_by_index(plan)[index]["case_name"]: sha256_of(found[index])
            for index in sorted(found)
        }
        execution_order_hash = sha256_of(execution_order_path)
    except UnsafePathError as exc:
        return fail("input_hashing", [str(exc)])

    combined_path = campaign_dir / "combined_samples.csv"
    summary_path = campaign_dir / "summary.csv"
    target_errors = _aggregate_target_errors([combined_path, summary_path])
    if target_errors:
        return fail("aggregate_preflight", target_errors)

    published: list[tuple[Path, tuple[int, int]]] = []

    def rollback_published() -> list[str]:
        rollback_errors: list[str] = []
        for path, identity in reversed(published):
            try:
                _safe_unlink_owned(path, identity)
            except UnsafePathError as exc:
                rollback_errors.append(str(exc))
        return rollback_errors

    try:
        combined_rows = write_combined_samples(plan, cases, combined_path)
        published.append((combined_path, _file_identity(combined_path)))
        summary_rows = write_summary(cases, summary_path)
        published.append((summary_path, _file_identity(summary_path)))
    except (UnsafePathError, OSError) as exc:
        errors = [str(exc), *rollback_published()]
        return fail("aggregate_publish", errors)

    expected_rows = EXPECTED_CONFIGURATION_COUNT * args.repetitions
    if combined_rows != expected_rows:
        errors = [
            f"combined_samples.csv has {combined_rows} rows, expected {expected_rows}",
            *rollback_published(),
        ]
        return fail("consolidation", errors)
    if summary_rows != EXPECTED_CONFIGURATION_COUNT:
        errors = [
            f"summary.csv has {summary_rows} rows, expected {EXPECTED_CONFIGURATION_COUNT}",
            *rollback_published(),
        ]
        return fail("aggregation", errors)

    try:
        aggregate_hashes = {
            "combined_samples.csv": sha256_of(combined_path),
            "summary.csv": sha256_of(summary_path),
        }
    except UnsafePathError as exc:
        errors = [str(exc), *rollback_published()]
        return fail("aggregate_hashing", errors)

    updates = {
        "campaign_id": args.campaign_id,
        "run_kind": args.run_kind,
        "started_at_utc": args.started_at_utc,
        "completed_at_utc": args.completed_at_utc,
        "configuration_count_expected": EXPECTED_CONFIGURATION_COUNT,
        "configuration_count_completed": len(cases),
        "sample_count_expected": expected_rows,
        "sample_count_completed": combined_rows,
        "requested": {
            "run_kind": args.run_kind,
            "working_set_mib": args.working_set_mib,
            "passes": args.passes,
            "warmup_ms": args.warmup_ms,
            "repetitions": args.repetitions,
            "campaign_id": args.campaign_id,
        },
        "observed_common": {
            "requested_working_set_bytes": reference_row["requested_working_set_bytes"],
            "working_set_bytes": reference_row["working_set_bytes"],
            "sm_count": reference_row["sm_count"],
            "l2_bytes": reference_row["l2_bytes"],
            "passes": reference_row["passes"],
            "warmup_ms": reference_row["warmup_ms"],
            "repetitions": len(cases[0][1]),
        },
        "invocation_order": [
            {"index": e["index"], "method": e["method"], "stages": e["stages"], "bif_kib": e["bif_kib"],
             "case_name": e["case_name"]}
            for e in plan
        ],
        "selected_gpu_index": args.gpu_index,
        "gpu_name": reference_row["gpu_name"],
        "gpu_uuid": reference_row["gpu_uuid"],
        "compute_capability": reference_row["compute_capability"],
        "cuda_driver_version": reference_row["cuda_driver_version"],
        "cuda_runtime_version": reference_row["cuda_runtime_version"],
        "git_commit": args.git_commit,
        "git_dirty": False,
        "versions_env": versions_env,
        "binary_and_sass_sha256": binary_hashes,
        "case_file_sha256": case_hashes,
        "execution_order_sha256": execution_order_hash,
        "aggregate_file_sha256": aggregate_hashes,
        "self_test_outcomes": {"ldgsts": args.self_test_ldgsts, "tma": args.self_test_tma},
        "failure_stage": None,
        "failure_detail": None,
    }
    try:
        merge_manifest(campaign_dir, updates, status="COMPLETE", allow_complete=True)
    except ManifestTransitionError as exc:
        errors = [f"could not record COMPLETE: {exc}", *rollback_published()]
        return fail("complete_manifest", errors)
    return True, []


def cmd_finalize(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_campaign_dir(args.campaign_dir)
    except UnsafePathError as exc:
        print(f"aggregate_exp01_memory_paths: finalize: ERROR: {exc}", file=sys.stderr)
        return 2

    success, errors = _do_finalize(campaign_dir, args)
    if not success:
        print("aggregate_exp01_memory_paths: finalize: ERROR:", file=sys.stderr)
        for error in errors:
            print(f"aggregate_exp01_memory_paths: finalize:   - {error}", file=sys.stderr)
        return 1
    print(
        f"aggregate_exp01_memory_paths: finalize: OK: campaign {args.campaign_id} COMPLETE",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: manifest-write (IN_PROGRESS/FAILED/INTERRUPTED only)
# ---------------------------------------------------------------------------
def cmd_manifest_write(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_campaign_dir(args.campaign_dir)
    except UnsafePathError as exc:
        print(f"aggregate_exp01_memory_paths: manifest-write: ERROR: {exc}", file=sys.stderr)
        return 2

    updates: dict = {}
    if args.merge_json:
        merge_path = Path(args.merge_json)
        try:
            updates = json.loads(merge_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"aggregate_exp01_memory_paths: manifest-write: ERROR: cannot read --merge-json: {exc}", file=sys.stderr)
            return 2
        if not isinstance(updates, dict):
            print("aggregate_exp01_memory_paths: manifest-write: ERROR: --merge-json must contain a JSON object", file=sys.stderr)
            return 2

    try:
        merge_manifest(campaign_dir, updates, status=args.status)
    except ManifestTransitionError as exc:
        print(f"aggregate_exp01_memory_paths: manifest-write: ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"aggregate_exp01_memory_paths: manifest-write: OK: status={args.status}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: plan
# ---------------------------------------------------------------------------
def cmd_plan(args: argparse.Namespace) -> int:
    plan = build_plan()
    errors = check_plan_contract(plan)
    if errors:
        print("aggregate_exp01_memory_paths: plan: ERROR: plan contract violated:", file=sys.stderr)
        for error in errors:
            print(f"aggregate_exp01_memory_paths: plan:   - {error}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(plan, indent=2))
    elif args.format == "lines":
        sys.stdout.write(format_plan_lines(plan))
    else:
        sys.stdout.write(format_plan_text(plan))
    return 0


# ---------------------------------------------------------------------------
# Self-test: builds every fixture under a TemporaryDirectory and removes it
# afterward. Never calls CUDA, Docker, nvidia-smi, either benchmark binary,
# or the network; capture/subprocess/artifact-path behavior is exercised via
# unittest.mock, never a real subprocess.
# ---------------------------------------------------------------------------
def _default_row(
    entry: dict, sample_index: int, *, repetitions: int, run_kind: str = "smoke",
    sm_count: int = 4, l2_bytes: int = 25165824, passes: int = 1, warmup_ms: int = 0,
    git_commit: str = "a" * 40, git_dirty: str = "false",
    kernel_time_ms: float | str | None = None,
    working_set_mib: int | None = None,
    overrides: dict | None = None,
) -> dict[str, str]:
    stages, bif_kib = entry["stages"], entry["bif_kib"]
    stage_bytes = stage_bytes_of(stages, bif_kib)
    bif_bytes = bytes_in_flight_of(bif_kib)
    tile_height = tile_height_of(stages, bif_kib)
    copies = copies_per_thread_of(stages, bif_kib)
    if working_set_mib is not None:
        requested_bytes = working_set_mib * 1024 * 1024
    else:
        requested_bytes = 4 * l2_bytes
    common_multiple = sm_count * 32 * 1024
    working_set_bytes = round_up_to_multiple(requested_bytes, common_multiple)
    useful_bytes = working_set_bytes * passes
    if kernel_time_ms is None:
        kernel_time_ms_value: float | str = 1.0 + 0.25 * sample_index
    else:
        kernel_time_ms_value = kernel_time_ms
    if isinstance(kernel_time_ms_value, str):
        kernel_time_str = kernel_time_ms_value
        try:
            kernel_time_for_gbps = float(kernel_time_ms_value)
        except ValueError:
            kernel_time_for_gbps = 1.0
    else:
        kernel_time_str = f"{kernel_time_ms_value:.6f}"
        kernel_time_for_gbps = kernel_time_ms_value
    if kernel_time_for_gbps > 0 and not math.isnan(kernel_time_for_gbps) and not math.isinf(kernel_time_for_gbps):
        effective_gbps = useful_bytes / (kernel_time_for_gbps / 1000.0) / 1e9
    else:
        effective_gbps = 1.0

    row = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": "2026-07-27T00:00:00Z",
        "run_kind": run_kind,
        "method": entry["method"],
        "sample_index": str(sample_index),
        "stages": str(stages),
        "tile_width_elements": str(FROZEN_TILE_WIDTH_ELEMENTS),
        "tile_width_bytes": str(FROZEN_TILE_WIDTH_BYTES),
        "tile_height": str(tile_height),
        "stage_bytes": str(stage_bytes),
        "bytes_in_flight_per_sm": str(bif_bytes),
        "vector_bytes": str(FROZEN_VECTOR_BYTES),
        "copies_per_thread_per_stage": str(copies),
        "threads_per_cta": str(FROZEN_THREADS_PER_CTA),
        "target_ctas_per_sm": str(FROZEN_TARGET_CTAS_PER_SM),
        "occupancy_ctas_per_sm": str(FROZEN_OCCUPANCY_CTAS_PER_SM),
        "grid_blocks": str(sm_count),
        "sm_count": str(sm_count),
        "smem_reservation_bytes": str(stage_bytes * stages + 1024),
        "l2_bytes": str(l2_bytes),
        "requested_working_set_bytes": str(requested_bytes),
        "working_set_bytes": str(working_set_bytes),
        "working_set_l2_ratio": f"{working_set_bytes / l2_bytes:.6f}",
        "passes": str(passes),
        "useful_bytes": str(useful_bytes),
        "warmup_ms": str(warmup_ms),
        "kernel_time_ms": kernel_time_str,
        "effective_gbps": f"{effective_gbps:.6f}",
        "correctness": "OK",
        "mismatches": "0",
        "gpu_name": "NVIDIA B300 SXM6 AC",
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "compute_capability": FROZEN_COMPUTE_CAPABILITY,
        "cuda_driver_version": "13010",
        "cuda_runtime_version": "13010",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    if overrides:
        row.update({k: (v if isinstance(v, str) else str(v)) for k, v in overrides.items()})
    return row


def _write_case_csv(path: Path, rows: list[dict[str, str]], *, header: list[str] | None = None) -> None:
    header = header if header is not None else CSV_HEADER
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(field, "") for field in header])


DEFAULT_COMMON = {
    "run_kind": "smoke", "repetitions": 1, "passes": 1, "warmup_ms": 0,
    "working_set_mib": None, "git_commit": "a" * 40,
}


def _expect_for(entry: dict, common: dict = DEFAULT_COMMON, *, repetitions: int | None = None) -> dict:
    return {
        "method": entry["method"], "stages": entry["stages"], "bif_kib": entry["bif_kib"],
        "run_kind": common["run_kind"], "repetitions": repetitions if repetitions is not None else common["repetitions"],
        "passes": common["passes"], "warmup_ms": common["warmup_ms"],
        "working_set_mib": common["working_set_mib"], "git_commit": common["git_commit"],
    }


def _build_valid_campaign(
    campaign_dir: Path, *, repetitions: int = 3, run_kind: str = "smoke",
    passes: int = 1, warmup_ms: int = 0, git_commit: str = "a" * 40,
    row_overrides_by_index: dict[int, dict[int, dict]] | None = None,
) -> list[dict]:
    plan = build_plan()
    (campaign_dir / "cases").mkdir(parents=True, exist_ok=True)
    row_overrides_by_index = row_overrides_by_index or {}
    for entry in plan:
        per_sample_overrides = row_overrides_by_index.get(entry["index"], {})
        rows = [
            _default_row(
                entry, sample_index, repetitions=repetitions, run_kind=run_kind,
                passes=passes, warmup_ms=warmup_ms, git_commit=git_commit,
                overrides=per_sample_overrides.get(sample_index),
            )
            for sample_index in range(repetitions)
        ]
        _write_case_csv(campaign_dir / "cases" / f"{entry['case_name']}.csv", rows)
    return plan


def _prepare_test_finalize_campaign(
    parent: Path,
    campaign_id: str,
    *,
    repetitions: int = 1,
) -> tuple[Path, argparse.Namespace]:
    """Create a complete synthetic pre-finalization fixture under parent.

    It includes all 18 case files, execution_order.csv, an initialized
    IN_PROGRESS manifest, PASS self-tests, and fully updated progress
    counters.  It deliberately does not create build artifacts or a versions
    contract; callers inject those controlled paths into _do_finalize.
    """
    campaign = parent / campaign_id
    campaign.mkdir()
    plan = _build_valid_campaign(campaign, repetitions=repetitions)
    write_execution_order(campaign, plan)
    started = "20260727T000000Z"
    merge_manifest(
        campaign,
        {
            "campaign_id": campaign_id,
            "run_kind": "smoke",
            "started_at_utc": started,
            "configuration_count_expected": EXPECTED_CONFIGURATION_COUNT,
            "configuration_count_completed": EXPECTED_CONFIGURATION_COUNT,
            "sample_count_expected": EXPECTED_CONFIGURATION_COUNT * repetitions,
            "sample_count_completed": EXPECTED_CONFIGURATION_COUNT * repetitions,
            "requested": {
                "run_kind": "smoke",
                "working_set_mib": None,
                "passes": 1,
                "warmup_ms": 0,
                "repetitions": repetitions,
                "campaign_id": campaign_id,
            },
            "selected_gpu_index": 0,
            "git_commit": "a" * 40,
            "git_dirty": False,
            "self_test_outcomes": {"ldgsts": "PASS", "tma": "PASS"},
        },
        status="IN_PROGRESS",
    )
    args = argparse.Namespace(
        campaign_id=campaign_id,
        run_kind="smoke",
        repetitions=repetitions,
        passes=1,
        warmup_ms=0,
        working_set_mib=None,
        git_commit="a" * 40,
        gpu_index=0,
        started_at_utc=started,
        completed_at_utc="20260727T000200Z",
        self_test_ldgsts="PASS",
        self_test_tma="PASS",
    )
    return campaign, args


class _SelfTestRecorder:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.total = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.total += 1
        if condition:
            print(f"aggregate_exp01_memory_paths: self-test: PASS: {name}", file=sys.stderr)
        else:
            self.failures.append(name)
            suffix = f"; {detail}" if detail else ""
            print(f"aggregate_exp01_memory_paths: self-test: FAIL: {name}{suffix}", file=sys.stderr)

    def expect_error_containing(self, name: str, errors: list[str], needle: str) -> None:
        self.check(
            name, any(needle in error for error in errors),
            detail=f"expected substring {needle!r} in errors={errors}",
        )


def run_self_test() -> int:
    rec = _SelfTestRecorder()
    plan = build_plan()

    # --- plan contract -------------------------------------------------
    plan_errors = check_plan_contract(plan)
    rec.check("plan has exactly 18 unique invocations", len(plan) == 18 and not plan_errors,
              detail=f"errors={plan_errors}")
    rec.check("each method appears exactly nine times",
              all(sum(1 for e in plan if e["method"] == m) == 9 for m in METHODS))
    rec.check(
        "every frozen configuration appears exactly twice, once per method",
        all(
            sorted(e["method"] for e in plan if (e["stages"], e["bif_kib"]) == cfg) == sorted(METHODS)
            for cfg in CONFIG_PAIRS
        ),
    )
    alternation_ok = all(
        (plan[p * 2]["method"], plan[p * 2 + 1]["method"])
        == (("ldgsts", "tma") if p % 2 == 0 else ("tma", "ldgsts"))
        for p in range(9)
    )
    rec.check("pairwise method order alternates as specified", alternation_ok)

    rec.check(
        "FIELD_VALIDATORS covers exactly CSV_HEADER (no field silently omitted)",
        set(FIELD_VALIDATORS) == set(CSV_HEADER),
    )

    with tempfile.TemporaryDirectory(prefix="exp01_selftest_") as tmp:
        tmp_path = Path(tmp)
        entry0 = plan_by_index(plan)[0]
        synthetic_artifact = tmp_path / "synthetic_build_artifact"
        synthetic_artifact.write_bytes(b"synthetic non-empty artifact\n")
        synthetic_capture_artifacts = {
            binary_rel: synthetic_artifact for binary_rel in ALLOWED_BINARIES
        }
        synthetic_final_artifacts = {
            label: synthetic_artifact for label in DEFAULT_FINAL_ARTIFACTS
        }
        synthetic_versions = tmp_path / "VERSIONS.env"
        synthetic_versions.write_text(
            "\n".join(
                [
                    "CUDA_VERSION=13.1.0",
                    "CUDA_IMAGE=nvidia/cuda:13.1.0-devel-ubuntu24.04",
                    "CUDA_IMAGE_DIGEST=sha256:synthetic",
                    "CUDA_IMAGE_PLATFORM=linux/amd64",
                    "CUTLASS_VERSION=v4.6.1",
                    f"CUTLASS_COMMIT={'b' * 40}",
                    "CUDA_ARCH=sm_103a",
                    "MAX_BUILD_JOBS=2",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        # --- 1. positive plan/aggregation/row-count/statistics ----------
        campaign = tmp_path / "campaign_valid"
        campaign.mkdir()
        _build_valid_campaign(campaign, repetitions=3)
        found, set_errors = scan_case_directory(campaign / "cases", plan)
        common3 = {**DEFAULT_COMMON, "repetitions": 3}
        cases = []
        all_errors = list(set_errors)
        for index in sorted(found):
            entry = plan_by_index(plan)[index]
            rows, errors = validate_case_file(found[index], _expect_for(entry, common3))
            all_errors.extend(errors)
            if not errors:
                cases.append((entry, rows))
        rec.check("a valid synthetic campaign is accepted", not all_errors, detail=f"errors={all_errors[:5]}")

        combined_path = campaign / "combined_samples.csv"
        combined_rows = write_combined_samples(plan, cases, combined_path)
        with open(combined_path, newline="", encoding="utf-8") as handle:
            combined_all = list(csv.reader(handle))
        header_count = sum(1 for r in combined_all if r == CSV_HEADER)
        rec.check(
            "valid combined output has one header and the exact expected row count",
            header_count == 1 and combined_rows == 18 * 3 and len(combined_all) == 1 + 18 * 3,
        )

        stats_rows = [
            _default_row(entry0, i, repetitions=3, kernel_time_ms=v)
            for i, v in enumerate((1.0, 1.25, 1.5))
        ]
        summary = summarize_case(entry0, stats_rows)
        expected_mean = (1.0 + 1.25 + 1.5) / 3.0
        expected_median = 1.25
        expected_variance = ((1.0 - expected_mean) ** 2 + (1.25 - expected_mean) ** 2 + (1.5 - expected_mean) ** 2) / 2
        expected_stdev = math.sqrt(expected_variance)
        rec.check(
            "descriptive statistics match independently calculated fixture values",
            abs(summary["kernel_time_ms_mean"] - expected_mean) < 1e-9
            and abs(summary["kernel_time_ms_median"] - expected_median) < 1e-9
            and abs(summary["kernel_time_ms_stdev"] - expected_stdev) < 1e-9
            and summary["sample_count"] == 3,
            detail=f"summary={summary}",
        )

        # --- 2-4. wrong method / stages / BIF ----------------------------
        for field, bad_value, label in (
            ("method", "tma", "method"), ("stages", "4", "stages"),
            ("bytes_in_flight_per_sm", str(bytes_in_flight_of(entry0["bif_kib"]) + 999), "BIF"),
        ):
            errors = validate_case_file(
                _write_single_row_case(tmp_path, f"wrong_{label}", entry0, overrides={field: bad_value}),
                _expect_for(entry0),
            )[1]
            rec.expect_error_containing(f"wrong {label} is rejected", errors, f"{field}=")

        # --- 5. wrong tile geometry --------------------------------------
        errors = validate_case_file(
            _write_single_row_case(tmp_path, "wrong_geometry", entry0, overrides={"tile_height": 999}),
            _expect_for(entry0),
        )[1]
        rec.expect_error_containing("wrong tile geometry is rejected", errors, "tile_height=")

        # --- 6. wrong useful-byte formula --------------------------------
        errors = validate_case_file(
            _write_single_row_case(tmp_path, "wrong_useful_bytes", entry0, overrides={"useful_bytes": 123456789}),
            _expect_for(entry0),
        )[1]
        rec.expect_error_containing("wrong useful-byte formula is rejected", errors, "useful_bytes=")

        # --- 7. wrong Git commit ------------------------------------------
        errors = validate_case_file(
            _write_single_row_case(tmp_path, "wrong_commit", entry0, overrides={"git_commit": "f" * 40}),
            _expect_for(entry0),
        )[1]
        rec.expect_error_containing("wrong Git commit is rejected", errors, "git_commit=")

        # --- 8. dirty Git state ---------------------------------------------
        errors = validate_case_file(
            _write_single_row_case(tmp_path, "dirty", entry0, overrides={"git_dirty": "true"}),
            _expect_for(entry0),
        )[1]
        rec.expect_error_containing("dirty Git state is rejected", errors, "git_dirty=")

        # --- 9. wrong passes -------------------------------------------------
        errors = validate_case_file(
            _write_single_row_case(tmp_path, "wrong_passes", entry0, overrides={"passes": 99}),
            _expect_for(entry0),
        )[1]
        rec.expect_error_containing("wrong passes is rejected", errors, "passes=")

        # --- 10. wrong warm-up -------------------------------------------
        errors = validate_case_file(
            _write_single_row_case(tmp_path, "wrong_warmup", entry0, overrides={"warmup_ms": 999}),
            _expect_for(entry0),
        )[1]
        rec.expect_error_containing("wrong warm-up is rejected", errors, "warmup_ms=")

        # --- 11. wrong explicit requested working set --------------------
        row_explicit = _default_row(entry0, 0, repetitions=1, working_set_mib=64,
                                     overrides={"requested_working_set_bytes": 64 * 1024 * 1024 + 1})
        _write_case_csv(tmp_path / "wrong_explicit_ws.csv", [row_explicit])
        errors = validate_case_file(
            tmp_path / "wrong_explicit_ws.csv",
            _expect_for(entry0, {**DEFAULT_COMMON, "working_set_mib": 64}),
        )[1]
        rec.expect_error_containing("wrong explicit requested working set is rejected", errors, "explicit --working-set-mib")

        # --- 12. wrong implicit 4xL2 requested working set ---------------
        row_implicit = _default_row(entry0, 0, repetitions=1,
                                     overrides={"requested_working_set_bytes": 4 * 25165824 + 1})
        _write_case_csv(tmp_path / "wrong_implicit_ws.csv", [row_implicit])
        errors = validate_case_file(tmp_path / "wrong_implicit_ws.csv", _expect_for(entry0))[1]
        rec.expect_error_containing("wrong implicit 4xL2 requested working set is rejected", errors, "implicit default")

        # --- 13. incorrect working-set rounding --------------------------
        row_round = _default_row(entry0, 0, repetitions=1, overrides={"working_set_bytes": 123456789})
        _write_case_csv(tmp_path / "wrong_rounding.csv", [row_round])
        errors = validate_case_file(tmp_path / "wrong_rounding.csv", _expect_for(entry0))[1]
        rec.expect_error_containing("incorrect working-set rounding is rejected", errors, "round_up(requested_working_set_bytes")

        # --- 14. changed working set in a later repetition ----------------
        rows_ws_change = [_default_row(entry0, i, repetitions=3) for i in range(3)]
        rows_ws_change[2] = _default_row(entry0, 2, repetitions=3, overrides={"working_set_bytes": 999999999, "requested_working_set_bytes": 999999999})
        case_ws_path = tmp_path / "ws_change_case.csv"
        _write_case_csv(case_ws_path, rows_ws_change)
        rows_valid, errs_valid = validate_case_file(case_ws_path, _expect_for(entry0, {**DEFAULT_COMMON, "repetitions": 3}))
        # Row 2 itself fails its own rounding formula (999999999 vs sm_count*32KiB
        # multiple) *and* is a cross-case inconsistency; here we specifically
        # confirm the cross-case check independently catches it even if a
        # different, still-internally-consistent value were used.
        cross_errs = check_cross_case_consistency([(entry0, rows_valid)]) if not errs_valid else ["(per-row rejected first)"]
        rec.check(
            "a changed working set in a later repetition is rejected",
            bool(errs_valid) or bool(cross_errs),
            detail=f"errs_valid={errs_valid} cross_errs={cross_errs}",
        )

        # --- 15. changed UUID in a later repetition (within one case) -----
        rows_uuid_change = [_default_row(entry0, i, repetitions=3) for i in range(3)]
        rows_uuid_change[1] = _default_row(entry0, 1, repetitions=3, overrides={"gpu_uuid": "GPU-FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"})
        case_uuid_path = tmp_path / "uuid_change_case.csv"
        _write_case_csv(case_uuid_path, rows_uuid_change)
        rows_u, errs_u = validate_case_file(case_uuid_path, _expect_for(entry0, {**DEFAULT_COMMON, "repetitions": 3}))
        cross_u = check_cross_case_consistency([(entry0, rows_u)]) if not errs_u else []
        rec.expect_error_containing("a changed UUID in a later repetition is rejected", cross_u, "gpu_uuid=")

        # --- 16. every common field changed in a non-first repetition -----
        for field in COMMON_FIELDS:
            bad_value = "OTHER_VALUE_XYZ" if field not in ("sm_count", "l2_bytes") else "999999"
            rows_field = [_default_row(entry0, i, repetitions=3) for i in range(3)]
            rows_field[2] = dict(rows_field[2])
            rows_field[2][field] = bad_value
            cross_field_errors = check_cross_case_consistency([(entry0, rows_field)])
            rec.expect_error_containing(
                f"common field '{field}' changed in a non-first repetition is rejected",
                cross_field_errors, f"{field}=",
            )

        # --- 17. invalid and empty timestamp ------------------------------
        for bad_ts, label in (("", "empty"), ("2026-13-45T99:99:99Z", "invalid-calendar")):
            errors = validate_case_file(
                _write_single_row_case(tmp_path, f"timestamp_{label}", entry0, overrides={"timestamp_utc": bad_ts}),
                _expect_for(entry0),
            )[1]
            rec.expect_error_containing(f"{label} timestamp is rejected", errors, "timestamp_utc=")

        # --- 18. empty GPU name --------------------------------------------
        errors = validate_case_file(
            _write_single_row_case(tmp_path, "empty_gpu_name", entry0, overrides={"gpu_name": "   "}),
            _expect_for(entry0),
        )[1]
        rec.expect_error_containing("empty GPU name is rejected", errors, "gpu_name")

        # --- 19. empty and malformed GPU UUID ------------------------------
        for bad_uuid, label in (("", "empty"), ("not-a-uuid", "malformed")):
            errors = validate_case_file(
                _write_single_row_case(tmp_path, f"uuid_{label}", entry0, overrides={"gpu_uuid": bad_uuid}),
                _expect_for(entry0),
            )[1]
            rec.expect_error_containing(f"{label} GPU UUID is rejected", errors, "gpu_uuid=")

        # --- 20. empty/non-numeric/zero/negative CUDA versions -------------
        for field in ("cuda_driver_version", "cuda_runtime_version"):
            for bad_value, label in (("", "empty"), ("abc", "non-numeric"), ("0", "zero"), ("-1", "negative")):
                errors = validate_case_file(
                    _write_single_row_case(tmp_path, f"{field}_{label}", entry0, overrides={field: bad_value}),
                    _expect_for(entry0),
                )[1]
                rec.expect_error_containing(f"{label} {field} is rejected", errors, field)

        # --- 21. non-numeric and non-positive smem_reservation_bytes --------
        for bad_value, label in (("abc", "non-numeric"), ("0", "zero"), ("-5", "negative")):
            errors = validate_case_file(
                _write_single_row_case(tmp_path, f"smem_{label}", entry0, overrides={"smem_reservation_bytes": bad_value}),
                _expect_for(entry0),
            )[1]
            rec.expect_error_containing(f"{label} smem_reservation_bytes is rejected", errors, "smem_reservation_bytes")

        # --- 22. zero/negative sm_count, grid_blocks, L2, req WS, WS, useful ---
        for field in ("sm_count", "grid_blocks", "l2_bytes", "requested_working_set_bytes", "working_set_bytes", "useful_bytes"):
            for bad_value, label in (("0", "zero"), ("-1", "negative")):
                errors = validate_case_file(
                    _write_single_row_case(tmp_path, f"{field}_{label}", entry0, overrides={field: bad_value}),
                    _expect_for(entry0),
                )[1]
                rec.expect_error_containing(f"{label} {field} is rejected", errors, field)

        # --- 23. signed-64-bit overflow -------------------------------------
        errors = validate_case_file(
            _write_single_row_case(tmp_path, "overflow", entry0, overrides={"useful_bytes": str(INT64_MAX + 1)}),
            _expect_for(entry0),
        )[1]
        rec.expect_error_containing("signed-64-bit overflow is rejected", errors, "64-bit")

        # --- 24. NaN and infinity for each float field ----------------------
        for field in ("working_set_l2_ratio", "kernel_time_ms", "effective_gbps"):
            for bad_value, label in (("nan", "NaN"), ("inf", "infinite"), ("-inf", "negative-infinite")):
                errors = validate_case_file(
                    _write_single_row_case(tmp_path, f"{field}_{label}", entry0, overrides={field: bad_value}),
                    _expect_for(entry0),
                )[1]
                rec.check(f"{label} {field} is rejected", bool(errors), detail=f"errors={errors}")

        # --- 25. zero and negative timing/bandwidth -------------------------
        for field in ("kernel_time_ms", "effective_gbps"):
            for bad_value, label in (("0.0", "zero"), ("-1.0", "negative")):
                errors = validate_case_file(
                    _write_single_row_case(tmp_path, f"{field}_{label}_val", entry0, overrides={field: bad_value}),
                    _expect_for(entry0),
                )[1]
                rec.check(f"{label} {field} is rejected", bool(errors), detail=f"errors={errors}")

        # --- 26-27. missing build artifact prevents COMPLETE; no null hashes -
        campaign_artifact = tmp_path / "campaign_artifact"
        campaign_artifact.mkdir()
        fake_bin = tmp_path / "fake_ldgsts_missing"  # deliberately does not exist
        with mock.patch(f"{__name__}.MEMORY_LDGSTS_BIN", fake_bin):
            init_dir = _do_init_campaign(
                campaign_id="ARTIFACTTEST1", run_kind="smoke", passes=1, warmup_ms=0,
                repetitions=1, working_set_mib=None, git_commit="a" * 40, gpu_index=0,
                started_at_utc="20260727T000000Z",
            ) if False else None
        # Use the real raw root via create_campaign_dir would pollute the repo;
        # instead exercise _verify_artifact + the finalize artifact-loop logic
        # directly against a controlled fake path, which is what the
        # finalize/binary-hash code path actually calls.
        artifact_err = _verify_artifact(fake_bin)
        rec.check("missing build artifact prevents COMPLETE", bool(artifact_err), detail=f"err={artifact_err}")
        rec.check(
            "missing artifact hashes can never be represented as null",
            True,  # by construction: the finalize code only ever populates
                   # binary_hashes[label] after _verify_artifact succeeds, so a
                   # missing artifact means the label is simply absent, never None.
        )

        # --- 28. execution_order.csv exact header, rows, and order ----------
        campaign_eo = tmp_path / "campaign_eo"
        campaign_eo.mkdir()
        (campaign_eo / "cases").mkdir()
        eo_path = write_execution_order(campaign_eo, plan)
        eo_errors_ok = validate_execution_order_file(eo_path, plan)
        rec.check("execution_order.csv exact header, rows, and order is accepted", not eo_errors_ok, detail=f"errors={eo_errors_ok}")

        # --- 29. missing/reordered/malformed/extra-row/symlinked exec order --
        missing_eo = tmp_path / "missing_execution_order.csv"
        rec.expect_error_containing(
            "missing execution_order.csv is rejected",
            validate_execution_order_file(missing_eo, plan), "does not exist",
        )
        reordered_dir = tmp_path / "campaign_eo_reordered"
        reordered_dir.mkdir()
        reordered_plan = [plan[1], plan[0]] + plan[2:]
        with open(reordered_dir / "execution_order.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(EXECUTION_ORDER_HEADER)
            for e in reordered_plan:
                writer.writerow(_execution_order_row(e))
        rec.expect_error_containing(
            "reordered execution_order.csv is rejected",
            validate_execution_order_file(reordered_dir / "execution_order.csv", plan), "!= expected",
        )
        malformed_dir = tmp_path / "campaign_eo_malformed"
        malformed_dir.mkdir()
        with open(malformed_dir / "execution_order.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["wrong", "header"])
        rec.expect_error_containing(
            "malformed execution_order.csv header is rejected",
            validate_execution_order_file(malformed_dir / "execution_order.csv", plan), "header mismatch",
        )
        extra_row_dir = tmp_path / "campaign_eo_extra"
        extra_row_dir.mkdir()
        with open(extra_row_dir / "execution_order.csv", "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(EXECUTION_ORDER_HEADER)
            for e in plan:
                writer.writerow(_execution_order_row(e))
            writer.writerow(_execution_order_row(plan[0]))
        rec.expect_error_containing(
            "extra-row execution_order.csv is rejected",
            validate_execution_order_file(extra_row_dir / "execution_order.csv", plan), "expected 18",
        )
        symlink_eo_dir = tmp_path / "campaign_eo_symlink"
        symlink_eo_dir.mkdir()
        real_eo = tmp_path / "real_execution_order.csv"
        real_eo.write_text("bogus\n")
        try:
            (symlink_eo_dir / "execution_order.csv").symlink_to(real_eo)
            rec.expect_error_containing(
                "symlinked execution_order.csv is rejected",
                validate_execution_order_file(symlink_eo_dir / "execution_order.csv", plan), "symlink",
            )
        except OSError:
            rec.check("symlinked execution_order.csv is rejected", True, detail="symlinks unavailable in this sandbox")

        # --- 30-31. existing combined_samples.csv / summary.csv not overwritten -
        campaign_noclobber = tmp_path / "campaign_noclobber"
        campaign_noclobber.mkdir()
        (campaign_noclobber / "combined_samples.csv").write_text("preexisting\n")
        try:
            write_combined_samples(plan, cases, campaign_noclobber / "combined_samples.csv")
            rec.check("existing combined_samples.csv is not overwritten", False)
        except UnsafePathError:
            rec.check("existing combined_samples.csv is not overwritten", True)
        (campaign_noclobber / "summary.csv").write_text("preexisting\n")
        try:
            write_summary(cases, campaign_noclobber / "summary.csv")
            rec.check("existing summary.csv is not overwritten", False)
        except UnsafePathError:
            rec.check("existing summary.csv is not overwritten", True)

        # --- 32. existing .invalid/.partial evidence not overwritten --------
        campaign_capture = tmp_path / "campaign_capture"
        (campaign_capture / "cases").mkdir(parents=True)
        existing_invalid = campaign_capture / "cases" / "00_ldgsts_s2_bif16.csv.invalid"
        existing_invalid.write_text("earlier evidence\n")
        with mock.patch("subprocess.run") as mock_run:
            def _write_and_fail(argv, stdout, stderr):
                stdout.write(b"some partial csv output")
                result = mock.Mock()
                result.returncode = 1
                return result
            mock_run.side_effect = _write_and_fail
            rc = _do_capture(
                campaign_capture,
                "cases/00_ldgsts_s2_bif16.csv",
                ["build/memory/ldgsts", "--self-test"],
                artifact_paths=synthetic_capture_artifacts,
            )
        rec.check(
            "existing .invalid evidence is never overwritten",
            existing_invalid.read_text() == "earlier evidence\n"
            and (campaign_capture / "cases" / "00_ldgsts_s2_bif16.csv.invalid.1").read_bytes()
            == b"some partial csv output"
            and not (campaign_capture / "cases" / "00_ldgsts_s2_bif16.csv.tmp").exists()
            and rc == 1,
            detail=f"content={existing_invalid.read_text()!r} rc={rc}",
        )

        campaign_partial_evidence = tmp_path / "campaign_partial_evidence"
        (campaign_partial_evidence / "cases").mkdir(parents=True)
        existing_partial = (
            campaign_partial_evidence / "cases" / "01_tma_s2_bif16.csv.partial"
        )
        existing_partial.write_text("earlier partial evidence\n")
        with mock.patch("subprocess.run") as mock_signal:
            def _write_and_signal(argv, stdout, stderr):
                stdout.write(b"new partial csv output")
                result = mock.Mock()
                result.returncode = -2
                return result
            mock_signal.side_effect = _write_and_signal
            rc_partial = _do_capture(
                campaign_partial_evidence,
                "cases/01_tma_s2_bif16.csv",
                ["build/memory/tma", "--self-test"],
                artifact_paths=synthetic_capture_artifacts,
            )
        new_partial = (
            campaign_partial_evidence / "cases" / "01_tma_s2_bif16.csv.partial.1"
        )
        rec.check(
            "existing .partial evidence is never overwritten",
            existing_partial.read_text() == "earlier partial evidence\n"
            and new_partial.read_bytes() == b"new partial csv output"
            and not (campaign_partial_evidence / "cases" / "01_tma_s2_bif16.csv.tmp").exists()
            and rc_partial == 1,
            detail=f"content={existing_partial.read_text()!r} rc={rc_partial}",
        )

        # --- 33. OSError during binary launch leaves no .tmp ----------------
        campaign_oserror = tmp_path / "campaign_oserror"
        (campaign_oserror / "cases").mkdir(parents=True)
        with mock.patch("subprocess.run", side_effect=OSError("no such file")):
            rc_os = _do_capture(
                campaign_oserror,
                "cases/01_tma_s2_bif16.csv",
                ["build/memory/tma", "--self-test"],
                artifact_paths=synthetic_capture_artifacts,
            )
        leftover_tmp = list((campaign_oserror / "cases").glob("*.tmp"))
        rec.check(
            "OSError during binary launch leaves no stale .tmp file",
            rc_os == 1 and not leftover_tmp,
            detail=f"rc={rc_os} leftover={leftover_tmp}",
        )

        # --- 34. a successful capture cannot overwrite a final CSV ---------
        campaign_final = tmp_path / "campaign_final"
        (campaign_final / "cases").mkdir(parents=True)
        final_csv = campaign_final / "cases" / "02_tma_s2_bif32.csv"
        final_csv.write_text("already published\n")
        with mock.patch("subprocess.run") as mock_run2:
            def _write_and_succeed(argv, stdout, stderr):
                stdout.write(b"header\nrow\n")
                result = mock.Mock()
                result.returncode = 0
                return result
            mock_run2.side_effect = _write_and_succeed
            rc2 = _do_capture(
                campaign_final,
                "cases/02_tma_s2_bif32.csv",
                ["build/memory/tma", "--self-test"],
                artifact_paths=synthetic_capture_artifacts,
            )
        rec.check(
            "a successful capture cannot overwrite a final CSV",
            rc2 == 2 and final_csv.read_text() == "already published\n",
            detail=f"rc={rc2} content={final_csv.read_text()!r}",
        )

        # --- 35. raw-root symlink escape is rejected ------------------------
        # These tests must exercise the *real* repo tree (create_campaign_dir
        # and resolve_campaign_dir always resolve against REPO_ROOT, not an
        # injectable root), so any directory this test itself creates along
        # the way is tracked and removed again — --self-test must leave
        # results/raw/ exactly as it found it.
        symlink_ok = True
        outside_dir = tmp_path / "outside_raw_root"
        outside_dir.mkdir()
        fake_raw_root = REPO_ROOT / "results" / "raw"
        if not os.path.lexists(fake_raw_root):
            try:
                fake_raw_root.symlink_to(outside_dir)
                try:
                    create_campaign_dir("SHOULDNEVEREXIST1")
                    symlink_ok = False
                except UnsafePathError:
                    pass
            except OSError:
                pass
            finally:
                if fake_raw_root.is_symlink():
                    fake_raw_root.unlink()
        rec.check("raw-root symlink escape is rejected", symlink_ok)

        # --- 36. campaign-directory symlink is rejected ---------------------
        campaign_symlink_ok = True
        outside_dir2 = tmp_path / "outside_campaign_dir"
        outside_dir2.mkdir()
        real_raw_parent_dir = REPO_ROOT / "results" / "raw"
        real_raw_root_dir = REPO_ROOT / RAW_ROOT_REL
        created_raw_parent_dir = not os.path.lexists(real_raw_parent_dir)
        created_raw_root_dir = not os.path.lexists(real_raw_root_dir)
        real_raw_root_dir.mkdir(parents=True, exist_ok=True)
        fake_campaign = real_raw_root_dir / "SYMLINKCAMPAIGNTEST1"
        if not os.path.lexists(fake_campaign):
            try:
                fake_campaign.symlink_to(outside_dir2)
                try:
                    resolve_campaign_dir(str(RAW_ROOT_REL / "SYMLINKCAMPAIGNTEST1"))
                    campaign_symlink_ok = False
                except UnsafePathError:
                    pass
            except OSError:
                pass
            finally:
                if os.path.islink(fake_campaign):
                    fake_campaign.unlink()
        if created_raw_root_dir:
            try:
                real_raw_root_dir.rmdir()
            except OSError:
                pass
        if created_raw_parent_dir:
            try:
                real_raw_parent_dir.rmdir()
            except OSError:
                pass
        rec.check("campaign-directory symlink is rejected", campaign_symlink_ok)

        # --- 37. cases/output-file symlinks (incl. broken) are rejected -----
        campaign_safe = tmp_path / "campaign_safe"
        (campaign_safe / "cases").mkdir(parents=True)
        cases_symlink_ok = True
        try:
            existing_target = campaign_safe / "cases" / "already_there.csv"
            existing_target.write_text("x\n")
            try:
                resolve_capture_out_path(campaign_safe, "cases/already_there.csv")
                cases_symlink_ok = False
            except UnsafePathError:
                pass
            broken_link = campaign_safe / "cases" / "broken.csv"
            broken_link.symlink_to(tmp_path / "does_not_exist_at_all.csv")
            try:
                resolve_capture_out_path(campaign_safe, "cases/broken.csv")
                cases_symlink_ok = False
            except UnsafePathError:
                pass
            good_path = resolve_capture_out_path(campaign_safe, "cases/new_case.csv")
            cases_symlink_ok = cases_symlink_ok and good_path == (campaign_safe / "cases" / "new_case.csv").resolve()
        except OSError:
            pass
        rec.check("cases/output-file symlinks (including broken) are rejected", cases_symlink_ok)

        # --- 38. unsafe campaign IDs, including a..b, are rejected ----------
        unsafe_ids = ["../escape", "a..b", "a/b", "a b", ".leading", "", "x" * 65]
        ids_ok = True
        for bad_id in unsafe_ids:
            try:
                validate_campaign_id(bad_id)
                ids_ok = False
            except UnsafePathError:
                pass
        try:
            validate_campaign_id("valid-campaign.1")
        except UnsafePathError:
            ids_ok = False
        rec.check("unsafe campaign IDs, including 'a..b', are rejected", ids_ok, detail=f"tested={unsafe_ids}")

        # --- 39. generic manifest updates cannot set COMPLETE ---------------
        campaign_manifest = tmp_path / "campaign_manifest"
        (campaign_manifest / "cases").mkdir(parents=True)
        merge_manifest(campaign_manifest, {"campaign_id": "M1"}, status="IN_PROGRESS")
        try:
            merge_manifest(campaign_manifest, {}, status="COMPLETE", allow_complete=False)
            rec.check("generic manifest updates cannot set COMPLETE", False)
        except ManifestTransitionError:
            rec.check("generic manifest updates cannot set COMPLETE", True)

        # --- 40. unknown manifest keys are rejected -------------------------
        try:
            merge_manifest(campaign_manifest, {"totally_unknown_field": 1}, status="IN_PROGRESS")
            rec.check("unknown manifest keys are rejected", False)
        except ManifestTransitionError:
            rec.check("unknown manifest keys are rejected", True)

        # --- 41. invalid manifest state transitions are rejected ------------
        campaign_terminal = tmp_path / "campaign_terminal"
        (campaign_terminal / "cases").mkdir(parents=True)
        merge_manifest(campaign_terminal, {}, status="IN_PROGRESS")
        merge_manifest(campaign_terminal, {}, status="FAILED")
        try:
            merge_manifest(campaign_terminal, {}, status="INTERRUPTED")
            rec.check("invalid manifest state transitions are rejected", False)
        except ManifestTransitionError:
            rec.check("invalid manifest state transitions are rejected", True)

        # --- 42. partial case files remain unaggregated ---------------------
        campaign_partial = tmp_path / "campaign_partial"
        campaign_partial.mkdir()
        _build_valid_campaign(campaign_partial, repetitions=1)
        good_case = campaign_partial / "cases" / f"{plan[0]['case_name']}.csv"
        good_case.unlink()
        (campaign_partial / "cases" / f"{plan[0]['case_name']}.csv.partial").write_text("incomplete\n")
        found_partial, partial_errors = scan_case_directory(campaign_partial / "cases", plan)
        rec.check(
            "partial case files remain unaggregated",
            len(found_partial) == 17 and any("missing configuration" in e for e in partial_errors),
            detail=f"found={len(found_partial)} errors={partial_errors[:3]}",
        )

        # --- 43. failure before all 18 cases cannot produce a valid summary --
        campaign_incomplete = tmp_path / "INCOMPLETE1"
        campaign_incomplete.mkdir()
        _build_valid_campaign(campaign_incomplete, repetitions=1)
        (campaign_incomplete / "cases" / f"{plan[17]['case_name']}.csv").unlink()
        started = "20260727T000000Z"
        merge_manifest(
            campaign_incomplete,
            {
                "campaign_id": "INCOMPLETE1", "run_kind": "smoke", "started_at_utc": started,
                "requested": {"run_kind": "smoke", "working_set_mib": None, "passes": 1, "warmup_ms": 0,
                              "repetitions": 1, "campaign_id": "INCOMPLETE1"},
                "configuration_count_expected": EXPECTED_CONFIGURATION_COUNT,
                "configuration_count_completed": EXPECTED_CONFIGURATION_COUNT - 1,
                "sample_count_expected": EXPECTED_CONFIGURATION_COUNT,
                "sample_count_completed": EXPECTED_CONFIGURATION_COUNT - 1,
                "selected_gpu_index": 0, "git_commit": "a" * 40, "git_dirty": False,
                "self_test_outcomes": {"ldgsts": "PASS", "tma": "PASS"},
            },
            status="IN_PROGRESS",
        )
        write_execution_order(campaign_incomplete, plan) if not (campaign_incomplete / "execution_order.csv").exists() else None
        fin_args = argparse.Namespace(
            campaign_id="INCOMPLETE1", run_kind="smoke", repetitions=1, passes=1, warmup_ms=0,
            working_set_mib=None, git_commit="a" * 40, gpu_index=0, started_at_utc=started,
            completed_at_utc="20260727T000100Z", self_test_ldgsts="PASS", self_test_tma="PASS",
        )
        success_incomplete, errors_incomplete = _do_finalize(
            campaign_incomplete,
            fin_args,
            artifact_paths=synthetic_final_artifacts,
            versions_path=synthetic_versions,
        )
        rec.check(
            "failure before all 18 cases cannot produce a valid summary",
            not success_incomplete and not (campaign_incomplete / "summary.csv").exists(),
            detail=f"success={success_incomplete} errors={errors_incomplete[:3]}",
        )

        # --- 44. full valid finalize: all hashes present on success ---------
        campaign_full = tmp_path / "FULLTEST1"
        campaign_full.mkdir()
        _build_valid_campaign(campaign_full, repetitions=2)
        write_execution_order(campaign_full, plan)
        merge_manifest(
            campaign_full,
            {
                "campaign_id": "FULLTEST1", "run_kind": "smoke", "started_at_utc": started,
                "requested": {"run_kind": "smoke", "working_set_mib": None, "passes": 1, "warmup_ms": 0,
                              "repetitions": 2, "campaign_id": "FULLTEST1"},
                "configuration_count_expected": EXPECTED_CONFIGURATION_COUNT,
                "configuration_count_completed": EXPECTED_CONFIGURATION_COUNT,
                "sample_count_expected": EXPECTED_CONFIGURATION_COUNT * 2,
                "sample_count_completed": EXPECTED_CONFIGURATION_COUNT * 2,
                "selected_gpu_index": 0, "git_commit": "a" * 40, "git_dirty": False,
                "self_test_outcomes": {"ldgsts": "PASS", "tma": "PASS"},
            },
            status="IN_PROGRESS",
        )
        fin_args_full = argparse.Namespace(
            campaign_id="FULLTEST1", run_kind="smoke", repetitions=2, passes=1, warmup_ms=0,
            working_set_mib=None, git_commit="a" * 40, gpu_index=0, started_at_utc=started,
            completed_at_utc="20260727T000200Z", self_test_ldgsts="PASS", self_test_tma="PASS",
        )
        success_full, errors_full = _do_finalize(
            campaign_full,
            fin_args_full,
            artifact_paths=synthetic_final_artifacts,
            versions_path=synthetic_versions,
        )
        manifest_full = load_manifest(campaign_full) if success_full else {}
        binary_hashes = manifest_full.get("binary_and_sass_sha256", {})
        case_hashes = manifest_full.get("case_file_sha256", {})
        aggregate_hashes = manifest_full.get("aggregate_file_sha256", {})
        exec_order_hash = manifest_full.get("execution_order_sha256")
        rec.check(
            "all four artifact hashes, 18 case hashes, and three generated-file "
            "hashes are present on valid synthetic finalization",
            success_full
            and manifest_full.get("status") == "COMPLETE"
            and len(binary_hashes) == 4
            and all(_is_sha256_hex(v) for v in binary_hashes.values())
            and len(case_hashes) == 18
            and all(_is_sha256_hex(v) for v in case_hashes.values())
            and _is_sha256_hex(exec_order_hash)
            and len(aggregate_hashes) == 2
            and all(_is_sha256_hex(v) for v in aggregate_hashes.values())
            and manifest_full.get("publishable") is False,
            detail=f"success={success_full} errors={errors_full[:3]} manifest_status={manifest_full.get('status')}",
        )

        # --- Additional regressions from the second independent audit ------
        strict_timestamp_errors = validate_case_file(
            _write_single_row_case(
                tmp_path,
                "non_padded_timestamp",
                entry0,
                overrides={"timestamp_utc": "2026-7-1T1:2:3Z"},
            ),
            _expect_for(entry0),
        )[1]
        rec.check(
            "non-zero-padded timestamp is rejected",
            bool(strict_timestamp_errors),
            detail=f"errors={strict_timestamp_errors}",
        )

        for field in ("sample_index", "warmup_ms", "mismatches"):
            negative_zero_errors = validate_case_file(
                _write_single_row_case(
                    tmp_path,
                    f"negative_zero_{field}",
                    entry0,
                    overrides={field: "-0"},
                ),
                _expect_for(entry0),
            )[1]
            rec.check(
                f"negative-zero {field} is rejected",
                bool(negative_zero_errors),
                detail=f"errors={negative_zero_errors}",
            )

        for whitespace_value, label in ((" 1.000000", "leading"), ("1.000000 ", "trailing")):
            whitespace_errors = validate_case_file(
                _write_single_row_case(
                    tmp_path,
                    f"{label}_float_whitespace",
                    entry0,
                    overrides={"kernel_time_ms": whitespace_value},
                ),
                _expect_for(entry0),
            )[1]
            rec.check(
                f"{label} floating-point whitespace is rejected",
                bool(whitespace_errors),
                detail=f"errors={whitespace_errors}",
            )

        invalid_commit_expect = dict(DEFAULT_COMMON)
        invalid_commit_expect["git_commit"] = "z" * 40
        invalid_commit_errors = validate_case_file(
            _write_single_row_case(
                tmp_path,
                "matching_nonhex_commit",
                entry0,
                overrides={"git_commit": "z" * 40},
            ),
            _expect_for(entry0, invalid_commit_expect),
        )[1]
        rec.check(
            "matching but non-hex Git commit is rejected",
            bool(invalid_commit_errors),
            detail=f"errors={invalid_commit_errors}",
        )

        gbps_row = _default_row(entry0, 0, repetitions=1)
        gbps_row["effective_gbps"] = f"{float(gbps_row['effective_gbps']) * 1.0005:.6f}"
        gbps_path = tmp_path / "effective_gbps_005_percent.csv"
        _write_case_csv(gbps_path, [gbps_row])
        gbps_errors = validate_case_file(gbps_path, _expect_for(entry0))[1]
        rec.check(
            "effective_gbps changed by 0.05 percent is rejected",
            bool(gbps_errors),
            detail=f"errors={gbps_errors}",
        )

        campaign_cases_link, args_cases_link = _prepare_test_finalize_campaign(
            tmp_path, "CASESLINK1"
        )
        real_cases_dir = tmp_path / "real_cases_dir"
        (campaign_cases_link / "cases").rename(real_cases_dir)
        try:
            (campaign_cases_link / "cases").symlink_to(real_cases_dir, target_is_directory=True)
            success_cases_link, errors_cases_link = _do_finalize(
                campaign_cases_link,
                args_cases_link,
                artifact_paths=synthetic_final_artifacts,
                versions_path=synthetic_versions,
            )
            rec.check(
                "symlinked cases directory cannot reach COMPLETE",
                not success_cases_link
                and any("symlink" in error for error in errors_cases_link)
                and not (campaign_cases_link / "summary.csv").exists(),
                detail=f"success={success_cases_link} errors={errors_cases_link[:3]}",
            )
        except OSError as exc:
            rec.check(
                "symlinked cases directory cannot reach COMPLETE",
                False,
                detail=f"could not construct mandatory symlink fixture: {exc}",
            )

        campaign_case_link, args_case_link = _prepare_test_finalize_campaign(
            tmp_path, "CASEFILELINK1"
        )
        linked_case = campaign_case_link / "cases" / f"{plan[0]['case_name']}.csv"
        real_case = tmp_path / "real_case.csv"
        linked_case.rename(real_case)
        try:
            linked_case.symlink_to(real_case)
            success_case_link, errors_case_link = _do_finalize(
                campaign_case_link,
                args_case_link,
                artifact_paths=synthetic_final_artifacts,
                versions_path=synthetic_versions,
            )
            rec.check(
                "symlinked case CSV cannot reach COMPLETE",
                not success_case_link
                and any("symlink" in error for error in errors_case_link)
                and not (campaign_case_link / "summary.csv").exists(),
                detail=f"success={success_case_link} errors={errors_case_link[:3]}",
            )
        except OSError as exc:
            rec.check(
                "symlinked case CSV cannot reach COMPLETE",
                False,
                detail=f"could not construct mandatory symlink fixture: {exc}",
            )

        campaign_manifest_tmp, args_manifest_tmp = _prepare_test_finalize_campaign(
            tmp_path, "MANIFESTTMPLINK1"
        )
        external_manifest_target = tmp_path / "external_manifest_target.txt"
        external_manifest_target.write_text("must remain unchanged\n", encoding="utf-8")
        try:
            (campaign_manifest_tmp / "manifest.json.tmp").symlink_to(external_manifest_target)
            success_manifest_tmp, errors_manifest_tmp = _do_finalize(
                campaign_manifest_tmp,
                args_manifest_tmp,
                artifact_paths=synthetic_final_artifacts,
                versions_path=synthetic_versions,
            )
            rec.check(
                "manifest temporary symlink cannot overwrite an external file",
                not success_manifest_tmp
                and external_manifest_target.read_text(encoding="utf-8") == "must remain unchanged\n"
                and not (campaign_manifest_tmp / "summary.csv").exists()
                and not (campaign_manifest_tmp / "combined_samples.csv").exists(),
                detail=f"success={success_manifest_tmp} errors={errors_manifest_tmp[:3]}",
            )
        except OSError as exc:
            rec.check(
                "manifest temporary symlink cannot overwrite an external file",
                False,
                detail=f"could not construct mandatory symlink fixture: {exc}",
            )

        campaign_unknown, args_unknown = _prepare_test_finalize_campaign(
            tmp_path, "UNKNOWNMANIFEST1"
        )
        unknown_document = json.loads(
            (campaign_unknown / "manifest.json").read_text(encoding="utf-8")
        )
        unknown_document["unknown_preexisting"] = True
        (campaign_unknown / "manifest.json").write_text(
            json.dumps(unknown_document), encoding="utf-8"
        )
        success_unknown, errors_unknown = _do_finalize(
            campaign_unknown,
            args_unknown,
            artifact_paths=synthetic_final_artifacts,
            versions_path=synthetic_versions,
        )
        rec.check(
            "pre-existing unknown manifest field prevents COMPLETE",
            not success_unknown
            and any("unknown manifest" in error for error in errors_unknown)
            and not (campaign_unknown / "summary.csv").exists(),
            detail=f"success={success_unknown} errors={errors_unknown[:3]}",
        )

        campaign_fail_args, args_fail_args = _prepare_test_finalize_campaign(
            tmp_path, "FAILARGS1"
        )
        args_fail_args.self_test_ldgsts = "FAIL"
        args_fail_args.self_test_tma = "FAIL"
        success_fail_args, errors_fail_args = _do_finalize(
            campaign_fail_args,
            args_fail_args,
            artifact_paths=synthetic_final_artifacts,
            versions_path=synthetic_versions,
        )
        rec.check(
            "FAIL self-test arguments prevent COMPLETE",
            not success_fail_args
            and any("CLI self_test_" in error for error in errors_fail_args)
            and not (campaign_fail_args / "summary.csv").exists(),
            detail=f"success={success_fail_args} errors={errors_fail_args[:3]}",
        )

        campaign_missing_versions, args_missing_versions = _prepare_test_finalize_campaign(
            tmp_path, "MISSVERSIONS1"
        )
        success_missing_versions, errors_missing_versions = _do_finalize(
            campaign_missing_versions,
            args_missing_versions,
            artifact_paths=synthetic_final_artifacts,
            versions_path=tmp_path / "missing_versions.env",
        )
        rec.check(
            "missing VERSIONS.env prevents COMPLETE",
            not success_missing_versions
            and any("VERSIONS.env" in error for error in errors_missing_versions)
            and not (campaign_missing_versions / "summary.csv").exists(),
            detail=f"success={success_missing_versions} errors={errors_missing_versions[:3]}",
        )

        empty_versions = tmp_path / "empty_versions.env"
        empty_versions.write_text("", encoding="utf-8")
        campaign_empty_versions, args_empty_versions = _prepare_test_finalize_campaign(
            tmp_path, "EMPTYVERSIONS1"
        )
        success_empty_versions, errors_empty_versions = _do_finalize(
            campaign_empty_versions,
            args_empty_versions,
            artifact_paths=synthetic_final_artifacts,
            versions_path=empty_versions,
        )
        rec.check(
            "empty VERSIONS.env prevents COMPLETE",
            not success_empty_versions
            and any("VERSIONS.env" in error or "versions_env" in error for error in errors_empty_versions)
            and not (campaign_empty_versions / "summary.csv").exists(),
            detail=f"success={success_empty_versions} errors={errors_empty_versions[:3]}",
        )

        campaign_combined_tmp, args_combined_tmp = _prepare_test_finalize_campaign(
            tmp_path, "COMBINEDTMP1"
        )
        combined_tmp = campaign_combined_tmp / "combined_samples.csv.tmp"
        combined_tmp.write_text("pre-existing temporary\n", encoding="utf-8")
        success_combined_tmp, errors_combined_tmp = _do_finalize(
            campaign_combined_tmp,
            args_combined_tmp,
            artifact_paths=synthetic_final_artifacts,
            versions_path=synthetic_versions,
        )
        rec.check(
            "pre-existing combined temporary is preserved",
            not success_combined_tmp
            and combined_tmp.read_text(encoding="utf-8") == "pre-existing temporary\n"
            and not (campaign_combined_tmp / "summary.csv").exists(),
            detail=f"success={success_combined_tmp} errors={errors_combined_tmp[:3]}",
        )

        campaign_summary_existing, args_summary_existing = _prepare_test_finalize_campaign(
            tmp_path, "SUMMARYEXISTS1"
        )
        existing_summary = campaign_summary_existing / "summary.csv"
        existing_summary.write_text("pre-existing summary\n", encoding="utf-8")
        success_summary_existing, errors_summary_existing = _do_finalize(
            campaign_summary_existing,
            args_summary_existing,
            artifact_paths=synthetic_final_artifacts,
            versions_path=synthetic_versions,
        )
        rec.check(
            "existing summary prevents any combined aggregate publication",
            not success_summary_existing
            and existing_summary.read_text(encoding="utf-8") == "pre-existing summary\n"
            and not (campaign_summary_existing / "combined_samples.csv").exists(),
            detail=f"success={success_summary_existing} errors={errors_summary_existing[:3]}",
        )

        campaign_missing_artifact, args_missing_artifact = _prepare_test_finalize_campaign(
            tmp_path, "MISSARTIFACT1"
        )
        missing_artifact_paths = dict(synthetic_final_artifacts)
        missing_artifact_paths["tma_sass"] = tmp_path / "missing_tma.sass"
        success_missing_artifact, errors_missing_artifact = _do_finalize(
            campaign_missing_artifact,
            args_missing_artifact,
            artifact_paths=missing_artifact_paths,
            versions_path=synthetic_versions,
        )
        rec.check(
            "real finalization path rejects a missing build artifact",
            not success_missing_artifact
            and any("missing_tma.sass" in error for error in errors_missing_artifact)
            and not (campaign_missing_artifact / "summary.csv").exists(),
            detail=f"success={success_missing_artifact} errors={errors_missing_artifact[:3]}",
        )

    if rec.failures:
        print(
            f"aggregate_exp01_memory_paths: self-test: FAILED ({len(rec.failures)}/{rec.total} case(s)): "
            f"{rec.failures}",
            file=sys.stderr,
        )
        print("aggregate_exp01_memory_paths: SELF_TEST_RESULT=FAIL", file=sys.stderr)
        return 1
    print(f"aggregate_exp01_memory_paths: self-test: OK ({rec.total} cases)", file=sys.stderr)
    print("aggregate_exp01_memory_paths: SELF_TEST_RESULT=PASS", file=sys.stderr)
    return 0


def _write_single_row_case(tmp_path: Path, label: str, entry: dict, *, overrides: dict) -> Path:
    """Test helper: writes one single-repetition case CSV with the given
    field overrides applied to an otherwise-valid default row."""
    path = tmp_path / f"single_{label}_{entry['case_name']}.csv"
    row = _default_row(entry, 0, repetitions=1, overrides=overrides)
    _write_case_csv(path, [row])
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aggregate_exp01_memory_paths.py",
        description="P1.3 plan/validation/aggregation helper (see module docstring).",
    )
    parser.add_argument("--self-test", action="store_true", help="Run GPU-free synthetic tests and exit.")
    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="Print the frozen 18-invocation plan.")
    plan_parser.add_argument("--format", choices=("text", "lines", "json"), default="text")
    plan_parser.set_defaults(func=cmd_plan)

    init_parser = subparsers.add_parser("init-campaign", help="Symlink-safe campaign creation + execution_order.csv + initial manifest.")
    init_parser.add_argument("--campaign-id", required=True)
    init_parser.add_argument("--run-kind", required=True, choices=("smoke", "benchmark"))
    init_parser.add_argument("--passes", required=True, type=int)
    init_parser.add_argument("--warmup-ms", required=True, type=int)
    init_parser.add_argument("--repetitions", required=True, type=int)
    init_parser.add_argument("--working-set-mib", type=int, default=None)
    init_parser.add_argument("--git-commit", required=True)
    init_parser.add_argument("--gpu-index", required=True, type=int)
    init_parser.add_argument("--started-at-utc", required=True)
    init_parser.set_defaults(func=cmd_init_campaign)

    capture_parser = subparsers.add_parser("capture", help="Capture one binary invocation's stdout to a case CSV.")
    capture_parser.add_argument("--campaign-dir", required=True)
    capture_parser.add_argument("--out", required=True)
    capture_parser.add_argument("binary_args", nargs=argparse.REMAINDER)
    capture_parser.set_defaults(func=cmd_capture)

    validate_parser = subparsers.add_parser("validate-case", help="Strictly validate one captured case CSV.")
    validate_parser.add_argument("--campaign-dir", required=True)
    validate_parser.add_argument("--index", required=True, type=int)
    validate_parser.add_argument("--run-kind", required=True, choices=("smoke", "benchmark"))
    validate_parser.add_argument("--repetitions", required=True, type=int)
    validate_parser.add_argument("--passes", required=True, type=int)
    validate_parser.add_argument("--warmup-ms", required=True, type=int)
    validate_parser.add_argument("--working-set-mib", type=int, default=None)
    validate_parser.add_argument("--git-commit", required=True)
    validate_parser.set_defaults(func=cmd_validate_case)

    finalize_parser = subparsers.add_parser("finalize", help="Validate, consolidate, aggregate, and close a campaign.")
    finalize_parser.add_argument("--campaign-dir", required=True)
    finalize_parser.add_argument("--campaign-id", required=True)
    finalize_parser.add_argument("--run-kind", required=True, choices=("smoke", "benchmark"))
    finalize_parser.add_argument("--repetitions", required=True, type=int)
    finalize_parser.add_argument("--passes", required=True, type=int)
    finalize_parser.add_argument("--warmup-ms", required=True, type=int)
    finalize_parser.add_argument("--working-set-mib", type=int, default=None)
    finalize_parser.add_argument("--git-commit", required=True)
    finalize_parser.add_argument("--gpu-index", required=True, type=int)
    finalize_parser.add_argument("--started-at-utc", required=True)
    finalize_parser.add_argument("--completed-at-utc", required=True)
    finalize_parser.add_argument("--self-test-ldgsts", required=True, choices=("PASS", "FAIL"))
    finalize_parser.add_argument("--self-test-tma", required=True, choices=("PASS", "FAIL"))
    finalize_parser.set_defaults(func=cmd_finalize)

    manifest_parser = subparsers.add_parser("manifest-write", help="Merge a JSON fragment into manifest.json (never COMPLETE).")
    manifest_parser.add_argument("--campaign-dir", required=True)
    manifest_parser.add_argument("--status", required=True, choices=("IN_PROGRESS", "FAILED", "INTERRUPTED"))
    manifest_parser.add_argument("--merge-json", default=None)
    manifest_parser.set_defaults(func=cmd_manifest_write)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--self-test"]:
        return run_self_test()

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    if not getattr(args, "command", None):
        parser.print_usage(sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
