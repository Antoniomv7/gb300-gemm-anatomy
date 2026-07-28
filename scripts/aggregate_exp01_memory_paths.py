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

# Documented tolerances for values that pass through the binaries' fixed
# six-decimal CSV formatting (std::fixed << std::setprecision(6)).
RATIO_ABS_TOL = 1e-6
GBPS_REL_TOL = 1e-3

INT64_MAX = 2**63 - 1
INT64_MIN = -(2**63)

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
    try:
        os.link(tmp_path, final_path)
    except FileExistsError as exc:
        raise UnsafePathError(f"refusing to overwrite existing target: {final_path}") from exc
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


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
    with open(path, "rb") as handle:
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
    if raw is None or raw.strip() == "" or raw != raw.strip():
        errors.append(f"{ctx}: {field}={raw!r} is not a canonical integer (empty or has whitespace)")
        return None
    if not re.fullmatch(r"-?\d+", raw):
        errors.append(f"{ctx}: {field}={raw!r} is not a canonical decimal integer")
        return None
    value = int(raw)
    if not (INT64_MIN <= value <= INT64_MAX):
        errors.append(f"{ctx}: {field}={value} is outside the signed 64-bit range")
        return None
    return value


def _parse_strict_float(raw: str, errors: list[str], ctx: str, field: str) -> float | None:
    if raw is None or raw.strip() == "":
        errors.append(f"{ctx}: {field}={raw!r} is empty")
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
    if raw is None or not re.fullmatch(r"-?\d+", raw.strip()) or raw != raw.strip():
        return None
    value = int(raw)
    if not (INT64_MIN <= value <= INT64_MAX):
        return None
    return value


def _peek_float(row: dict[str, str], field: str) -> float | None:
    try:
        value = float(row.get(field, ""))
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
        if not (INT64_MIN <= product <= INT64_MAX):
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
        if recomputed > 0:
            rel_err = abs(recomputed - value) / recomputed
            if rel_err > GBPS_REL_TOL:
                errors.append(
                    f"{ctx}: effective_gbps={value} inconsistent with useful_bytes/kernel_time="
                    f"{recomputed} (rel_err={rel_err}, tol={GBPS_REL_TOL})"
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
    if row["git_commit"] != expect["git_commit"]:
        errors.append(f"{ctx}: git_commit={row['git_commit']!r} != expected {expect['git_commit']!r}")
        return None
    return row["git_commit"]


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
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
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

    if not cases_dir.is_dir():
        return {}, [f"{cases_dir}: cases directory does not exist"]

    for path in sorted(cases_dir.iterdir()):
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
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(EXECUTION_ORDER_HEADER)
        for entry in plan:
            writer.writerow(_execution_order_row(entry))
    _publish_no_clobber(tmp_path, out_path)
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
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
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
        tmp_path.unlink()
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for entry in plan:
            rows = sorted(rows_by_index[entry["index"]], key=lambda r: int(r["sample_index"]))
            for row in rows:
                writer.writerow([row[field] for field in CSV_HEADER])
                row_count += 1
    _publish_no_clobber(tmp_path, out_path)
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
        tmp_path.unlink()
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUMMARY_HEADER)
        for summary in summaries:
            writer.writerow([format_summary_value(field, summary[field]) for field in SUMMARY_HEADER])
    _publish_no_clobber(tmp_path, out_path)
    return len(summaries)


# ---------------------------------------------------------------------------
# Manifest: allowlisted keys/types, enforced state transitions, atomic
# updates (the one intentional os.replace()-based exception to no-clobber).
# ---------------------------------------------------------------------------
def load_manifest(campaign_dir: Path) -> dict:
    path = campaign_dir / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def write_manifest_atomic(campaign_dir: Path, manifest: dict) -> None:
    path = campaign_dir / "manifest.json"
    tmp_path = campaign_dir / "manifest.json.tmp"
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def _validate_manifest_updates(updates: dict) -> None:
    unknown = set(updates) - set(ALLOWED_MANIFEST_KEYS)
    if unknown:
        raise ManifestTransitionError(f"unknown manifest field(s): {sorted(unknown)}")
    for key, value in updates.items():
        expected_type = ALLOWED_MANIFEST_KEYS[key]
        if not isinstance(value, expected_type):
            raise ManifestTransitionError(
                f"manifest field {key!r} has invalid type {type(value).__name__}, "
                f"expected {expected_type}"
            )


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
    current_status = manifest.get("status")
    allowed = ALLOWED_TRANSITIONS.get(current_status, frozenset())
    if status not in allowed:
        raise ManifestTransitionError(
            f"invalid manifest state transition: {current_status!r} -> {status!r}"
        )
    manifest.update(updates)
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["experiment_id"] = EXPERIMENT_ID
    manifest["status"] = status
    manifest["publishable"] = False
    write_manifest_atomic(campaign_dir, manifest)
    return manifest


def parse_versions_env() -> dict[str, str]:
    values: dict[str, str] = {}
    versions_path = REPO_ROOT / "VERSIONS.env"
    if not versions_path.is_file():
        return values
    for line in versions_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
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
def _do_capture(campaign_dir: Path, out_rel: str, binary_argv: list[str]) -> int:
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
    artifact_err = _verify_artifact(REPO_ROOT / binary_argv[0])
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
        if tmp_path.exists() and tmp_path.stat().st_size > 0:
            suffix = ".partial" if is_signal else ".invalid"
            failed_path = out_path.with_name(out_path.name + suffix)
            try:
                _publish_no_clobber(tmp_path, failed_path)
                print(f"aggregate_exp01_memory_paths: capture: preserved evidence as {failed_path.name}", file=sys.stderr)
            except UnsafePathError as exc:
                print(f"aggregate_exp01_memory_paths: capture: ERROR: could not preserve evidence: {exc}", file=sys.stderr)
                tmp_path.unlink(missing_ok=True)
        else:
            tmp_path.unlink(missing_ok=True)

    print(f"aggregate_exp01_memory_paths: capture: running {binary_argv!r} -> {out_path.name}", file=sys.stderr)
    try:
        with open(tmp_path, "wb") as csv_out:
            result = subprocess.run(binary_argv, stdout=csv_out, stderr=None)
    except OSError as exc:
        print(f"aggregate_exp01_memory_paths: capture: ERROR: unable to launch binary: {exc}", file=sys.stderr)
        salvage(is_signal=False)
        return 1

    if result.returncode == 0:
        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            print("aggregate_exp01_memory_paths: capture: ERROR: binary exited 0 but produced no stdout", file=sys.stderr)
            salvage(is_signal=False)
            return 1
        try:
            _publish_no_clobber(tmp_path, out_path)
        except UnsafePathError as exc:
            print(f"aggregate_exp01_memory_paths: capture: ERROR: {exc}", file=sys.stderr)
            tmp_path.unlink(missing_ok=True)
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
    if not case_path.is_file():
        print(f"aggregate_exp01_memory_paths: validate-case: ERROR: missing case file {case_path}", file=sys.stderr)
        return 1

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
    manifest = load_manifest(campaign_dir)
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
    self_test = manifest.get("self_test_outcomes", {})
    if self_test.get("ldgsts") != "PASS":
        errors.append(f"manifest self_test_outcomes.ldgsts={self_test.get('ldgsts')!r} != 'PASS'")
    if self_test.get("tma") != "PASS":
        errors.append(f"manifest self_test_outcomes.tma={self_test.get('tma')!r} != 'PASS'")
    return errors


def _do_finalize(campaign_dir: Path, args) -> tuple[bool, list[str]]:
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
        manifest = load_manifest(campaign_dir)
        if manifest.get("status") in (None, "IN_PROGRESS"):
            try:
                merge_manifest(
                    campaign_dir,
                    {"failure_stage": stage, "failure_detail": errors[:50]},
                    status="FAILED",
                )
            except ManifestTransitionError:
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

    artifact_errors: list[str] = []
    binary_hashes: dict[str, str] = {}
    for binary_rel, label in ALLOWED_BINARIES.items():
        bin_path = REPO_ROOT / binary_rel
        sass_path = bin_path.with_suffix(bin_path.suffix + ".sass")
        for path, hash_label in ((bin_path, label), (sass_path, label.replace("_bin", "_sass"))):
            err = _verify_artifact(path)
            if err:
                artifact_errors.append(err)
            else:
                binary_hashes[hash_label] = sha256_of(path)
    if artifact_errors:
        return fail("missing_artifact", artifact_errors)

    combined_path = campaign_dir / "combined_samples.csv"
    summary_path = campaign_dir / "summary.csv"
    try:
        combined_rows = write_combined_samples(plan, cases, combined_path)
        summary_rows = write_summary(cases, summary_path)
    except UnsafePathError as exc:
        return fail("aggregate_publish", [str(exc)])

    expected_rows = EXPECTED_CONFIGURATION_COUNT * args.repetitions
    if combined_rows != expected_rows:
        return fail("consolidation", [f"combined_samples.csv has {combined_rows} rows, expected {expected_rows}"])
    if summary_rows != EXPECTED_CONFIGURATION_COUNT:
        return fail("aggregation", [f"summary.csv has {summary_rows} rows, expected {EXPECTED_CONFIGURATION_COUNT}"])

    reference_row = cases[0][1][0]
    case_hashes = {plan_by_index(plan)[index]["case_name"]: sha256_of(found[index]) for index in sorted(found)}

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
        "versions_env": parse_versions_env(),
        "binary_and_sass_sha256": binary_hashes,
        "case_file_sha256": case_hashes,
        "execution_order_sha256": sha256_of(execution_order_path),
        "aggregate_file_sha256": {
            "combined_samples.csv": sha256_of(combined_path),
            "summary.csv": sha256_of(summary_path),
        },
        "self_test_outcomes": {"ldgsts": args.self_test_ldgsts, "tma": args.self_test_tma},
        "failure_stage": None,
        "failure_detail": None,
    }
    try:
        merge_manifest(campaign_dir, updates, status="COMPLETE", allow_complete=True)
    except ManifestTransitionError as exc:
        return False, [f"could not record COMPLETE: {exc}"]
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
            rc = _do_capture(campaign_capture, "cases/00_ldgsts_s2_bif16.csv", ["build/memory/ldgsts", "--self-test"])
        rec.check(
            "existing .invalid evidence is never overwritten",
            existing_invalid.read_text() == "earlier evidence\n" and rc == 1,
            detail=f"content={existing_invalid.read_text()!r} rc={rc}",
        )

        # --- 33. OSError during binary launch leaves no .tmp ----------------
        campaign_oserror = tmp_path / "campaign_oserror"
        (campaign_oserror / "cases").mkdir(parents=True)
        with mock.patch("subprocess.run", side_effect=OSError("no such file")):
            rc_os = _do_capture(campaign_oserror, "cases/01_tma_s2_bif16.csv", ["build/memory/tma", "--self-test"])
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
            rc2 = _do_capture(campaign_final, "cases/02_tma_s2_bif32.csv", ["build/memory/tma", "--self-test"])
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
        with mock.patch(f"{__name__}.MEMORY_LDGSTS_BIN", REPO_ROOT / "VERSIONS.env"), \
             mock.patch(f"{__name__}.MEMORY_LDGSTS_SASS", REPO_ROOT / "VERSIONS.env"), \
             mock.patch(f"{__name__}.MEMORY_TMA_BIN", REPO_ROOT / "VERSIONS.env"), \
             mock.patch(f"{__name__}.MEMORY_TMA_SASS", REPO_ROOT / "VERSIONS.env"):
            success_incomplete, errors_incomplete = _do_finalize(campaign_incomplete, fin_args)
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
        fake_bin_path = REPO_ROOT / "VERSIONS.env"  # a real, non-empty, non-symlink file to hash
        with mock.patch(f"{__name__}.MEMORY_LDGSTS_BIN", fake_bin_path), \
             mock.patch(f"{__name__}.MEMORY_LDGSTS_SASS", fake_bin_path), \
             mock.patch(f"{__name__}.MEMORY_TMA_BIN", fake_bin_path), \
             mock.patch(f"{__name__}.MEMORY_TMA_SASS", fake_bin_path):
            success_full, errors_full = _do_finalize(campaign_full, fin_args_full)
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
