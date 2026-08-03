#!/usr/bin/env python3
"""P2.3 plan generation, CSV validation, consolidation, and aggregation for
the exp02_umma_throughput joint 1-SM/2-SM BF16 UMMA sweep.

This script never touches CUDA, Docker, ``nvidia-smi``, either UMMA binary,
or the network. ``scripts/run_exp02_umma_throughput.sh`` is the only thing
that invokes GPU work (exclusively through ``scripts/run_container.sh``);
this script plans the frozen 24-invocation sweep, centralizes symlink-safe
campaign initialization, strictly validates every field of every repetition
of the raw 37-column CSV that the already-audited ``build/compute/umma_1sm``
(P2.1) and ``build/compute/umma_2sm`` (P2.2) binaries already emit unmodified,
consolidates it losslessly, computes descriptive per-configuration
statistics, and owns the campaign manifest's state machine.

P2.3 reuses the audited P2.1/P2.2 binaries and their existing command-line
interfaces unmodified; it never introduces another CUDA kernel. P2.3
produces functional/descriptive infrastructure output only: it does not
compute TFLOP/s, an empirical Tensor Core ceiling, 1-SM versus 2-SM speedup,
scaling efficiency, saturation, a winning configuration, or any Nsight
Compute or publishable performance conclusion. See src/compute/P2_3_PROTOCOL.md
(the scientific boundary with P2.4) and PLAN.md.

Subcommands:
  plan            Print the frozen deterministic 24-invocation plan.
  init-campaign   Centralized, symlink-safe campaign creation: makes the
                  campaign directory (never following a broken/real symlink
                  at any path component, never overwriting an existing
                  campaign), writes execution_order.csv once, and writes the
                  initial IN_PROGRESS manifest. Prints the campaign's
                  repo-relative path on stdout.
  capture         Run one allowlisted binary invocation inside the
                  container, capturing its stdout to a temporary CSV and
                  publishing it with no-clobber semantics only on success
                  (used by run_exp02_umma_throughput.sh via
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
                  subprocess -- capture/artifact behavior is exercised via
                  unittest.mock). Runs entirely under a TemporaryDirectory;
                  never touches results/raw/. Prints
                  "aggregate_exp02_umma_throughput: SELF_TEST_RESULT=PASS"
                  only if every case passes.

Exit codes: 0 on success (including --self-test passing); 1 on a validation,
aggregation, or capture failure; 2 on a usage/precondition error.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import csv
import hashlib
import io
import itertools
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
EXPERIMENT_ID = "exp02_umma_throughput"

METHODS = ("umma_1sm", "umma_2sm")
METHOD_INFO = {
    "umma_1sm": {"cta_group": 1, "m": 128, "grid_blocks": 1, "binary": "build/compute/umma_1sm"},
    "umma_2sm": {"cta_group": 2, "m": 256, "grid_blocks": 2, "binary": "build/compute/umma_2sm"},
}
FROZEN_K = 16
FROZEN_THREADS_PER_CTA = 128
FROZEN_COMPUTE_CAPABILITY = "10.3"
FROZEN_OPERAND_PATH = "smem_smem"
FROZEN_INPUT_TYPE = "bf16"
FROZEN_ACCUMULATOR_TYPE = "fp32"
FROZEN_N_VALUES = (64, 128, 256)
FROZEN_DEPTH_VALUES = (4, 16, 64, 256)

# Frozen 12 logical (N, depth) pairs, in the exact required order (task
# section 3): N ascending, and within each N, depth ascending.
PAIR_ORDER = (
    (64, 4), (64, 16), (64, 64), (64, 256),
    (128, 4), (128, 16), (128, 64), (128, 256),
    (256, 4), (256, 16), (256, 64), (256, 256),
)
EXPECTED_PAIR_COUNT = 12
EXPECTED_CONFIGURATION_COUNT = 24

RAW_ROOT_PARTS = ("results", "raw", "exp02_umma_throughput")
RAW_ROOT_REL = Path(*RAW_ROOT_PARTS)

UMMA_1SM_BIN = REPO_ROOT / "build/compute/umma_1sm"
UMMA_1SM_SASS = REPO_ROOT / "build/compute/umma_1sm.sass"
UMMA_2SM_BIN = REPO_ROOT / "build/compute/umma_2sm"
UMMA_2SM_SASS = REPO_ROOT / "build/compute/umma_2sm.sass"

ALLOWED_BINARIES = {
    "build/compute/umma_1sm": "umma_1sm_bin",
    "build/compute/umma_2sm": "umma_2sm_bin",
}

DEFAULT_CAPTURE_ARTIFACTS = {
    "build/compute/umma_1sm": UMMA_1SM_BIN,
    "build/compute/umma_2sm": UMMA_2SM_BIN,
}

DEFAULT_FINAL_ARTIFACTS = {
    "umma_1sm_bin": UMMA_1SM_BIN,
    "umma_1sm_sass": UMMA_1SM_SASS,
    "umma_2sm_bin": UMMA_2SM_BIN,
    "umma_2sm_sass": UMMA_2SM_SASS,
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

# Exact 37-column header both build/compute/umma_1sm and build/compute/umma_2sm
# print (see src/compute/P2_PROTOCOL.md section 14 / P2_2_PROTOCOL.md section 12).
CSV_HEADER = [
    "schema_version", "timestamp_utc", "run_kind", "publishable", "method",
    "sample_index", "cta_group", "m", "n", "k", "depth", "iterations",
    "warmup_iterations", "repetitions", "umma_per_iteration", "total_umma",
    "flops_per_umma", "total_flops", "elapsed_cycles", "cycles_per_umma",
    "flops_per_cycle", "threads_per_cta", "grid_blocks", "tmem_columns",
    "operand_path", "input_type", "accumulator_type", "correctness",
    "mismatches", "max_abs_error", "gpu_name", "gpu_uuid",
    "compute_capability", "cuda_driver_version", "cuda_runtime_version",
    "git_commit", "git_dirty",
]
assert len(CSV_HEADER) == 37, f"CSV_HEADER has {len(CSV_HEADER)} columns, expected 37"

EXECUTION_ORDER_HEADER = [
    "invocation_index", "pair_index", "method", "cta_group", "m", "n", "k",
    "depth", "binary", "case_file",
]

COMBINED_HEADER = ["invocation_index", "pair_index", "case_name"] + CSV_HEADER

CAMPAIGN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
GPU_UUID_RE = re.compile(
    r"^GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
TIMESTAMP_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
MANIFEST_TIMESTAMP_RE = re.compile(r"^\d{8}T\d{6}Z$")
CANONICAL_UINT_RE = re.compile(r"^(?:0|[1-9]\d*)$")
CANONICAL_FIXED6_RE = re.compile(r"^(?:0|[1-9]\d*)\.\d{6}$")

# Documented tolerance for values that pass through the binaries' fixed
# six-decimal CSV formatting (std::fixed << std::setprecision(6)).
FIXED6_HALF_ULP = 0.5e-6
RATIO_ABS_TOL = FIXED6_HALF_ULP + 1e-12

INT64_MAX = 2**63 - 1

# CLI bounds enforced by scripts/run_exp02_umma_throughput.sh before any
# Docker/GPU/raw-results access, and re-validated in the manifest contract
# below. Chosen so every existing signed-64-bit FLOP/UMMA formula (see
# check_int64_safety()) stays far inside the int64 range for every one of the
# 24 frozen configurations. REPETITIONS_MIN=2 (audit repair): summary.csv's
# sample standard deviation and coefficient of variation are only
# statistically meaningful with at least two observations; a real P2.3
# campaign must never be able to request repetitions=1.
ITERATIONS_MIN, ITERATIONS_MAX = 1, 1_000_000
WARMUP_ITERATIONS_MIN, WARMUP_ITERATIONS_MAX = 0, 1_000_000
REPETITIONS_MIN, REPETITIONS_MAX = 2, 1_000_000

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
# Frozen plan generation (24-invocation contract, task sections 3 and 7).
# ---------------------------------------------------------------------------
def build_plan() -> list[dict]:
    plan = []
    index = 0
    for pair_index, (n, depth) in enumerate(PAIR_ORDER):
        order = METHODS if pair_index % 2 == 0 else (METHODS[1], METHODS[0])
        for method in order:
            info = METHOD_INFO[method]
            case_name = f"{index:02d}_{method}_n{n}_d{depth}"
            plan.append({
                "index": index,
                "pair_index": pair_index,
                "method": method,
                "cta_group": info["cta_group"],
                "m": info["m"],
                "n": n,
                "k": FROZEN_K,
                "depth": depth,
                "grid_blocks": info["grid_blocks"],
                "binary": info["binary"],
                "case_name": case_name,
            })
            index += 1
    return plan


def plan_by_index(plan: list[dict]) -> dict[int, dict]:
    return {entry["index"]: entry for entry in plan}


def check_int64_safety() -> list[str]:
    """Proves the frozen FLOP/UMMA formulas cannot overflow signed 64-bit
    for any frozen configuration at the runner's own maximum CLI bounds."""
    errors: list[str] = []
    max_m = max(info["m"] for info in METHOD_INFO.values())
    flops_per_umma_max = 2 * max_m * max(FROZEN_N_VALUES) * FROZEN_K
    total_umma_max = max(FROZEN_DEPTH_VALUES) * ITERATIONS_MAX
    total_flops_max = flops_per_umma_max * total_umma_max
    if total_flops_max > INT64_MAX:
        errors.append(
            f"worst-case total_flops={total_flops_max} would exceed "
            f"INT64_MAX={INT64_MAX} at ITERATIONS_MAX={ITERATIONS_MAX}"
        )
    return errors


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
        key = (entry["method"], entry["n"], entry["depth"])
        if key in seen_keys:
            errors.append(f"duplicate invocation for {key}")
        seen_keys.add(key)

    for method in METHODS:
        count = sum(1 for entry in plan if entry["method"] == method)
        if count != EXPECTED_PAIR_COUNT:
            errors.append(f"method {method!r} appears {count} times, expected {EXPECTED_PAIR_COUNT}")

    case_names = [entry["case_name"] for entry in plan]
    if len(set(case_names)) != len(case_names):
        errors.append("duplicate case_name values in plan")

    if len(plan) == EXPECTED_CONFIGURATION_COUNT:
        pair_counts = collections.Counter(entry["pair_index"] for entry in plan)
        if sorted(pair_counts) != list(range(EXPECTED_PAIR_COUNT)) or any(
            count != 2 for count in pair_counts.values()
        ):
            errors.append(f"pair_index values are not exactly 0..{EXPECTED_PAIR_COUNT - 1}, each twice: {dict(pair_counts)}")

        for pair_index, (n, depth) in enumerate(PAIR_ORDER):
            first, second = plan[pair_index * 2], plan[pair_index * 2 + 1]
            expected_order = METHODS if pair_index % 2 == 0 else (METHODS[1], METHODS[0])
            if (first["method"], second["method"]) != expected_order:
                errors.append(
                    f"pair {pair_index} (n={n}, depth={depth}) method order "
                    f"{(first['method'], second['method'])} != expected {expected_order}"
                )
            for entry in (first, second):
                if (entry["n"], entry["depth"]) != (n, depth):
                    errors.append(
                        f"pair {pair_index} entry index {entry['index']} has "
                        f"(n,depth)=({entry['n']},{entry['depth']}) != expected ({n},{depth})"
                    )
                if entry["pair_index"] != pair_index:
                    errors.append(
                        f"pair {pair_index} entry index {entry['index']} has "
                        f"pair_index={entry['pair_index']} != {pair_index}"
                    )
                info = METHOD_INFO[entry["method"]]
                for field in ("cta_group", "m", "grid_blocks", "binary"):
                    if entry[field] != info[field]:
                        errors.append(
                            f"entry index {entry['index']} method={entry['method']} "
                            f"{field}={entry[field]!r} != expected {info[field]!r}"
                        )
                if entry["k"] != FROZEN_K:
                    errors.append(f"entry index {entry['index']} k={entry['k']} != {FROZEN_K}")

    errors.extend(check_int64_safety())
    return errors


def format_plan_text(plan: list[dict]) -> str:
    lines = [
        "index  pair  method     cta_group  m    n    k   depth  binary                     case_name",
    ]
    for entry in plan:
        lines.append(
            f"{entry['index']:>5d}  {entry['pair_index']:>4d}  {entry['method']:<9s}  "
            f"{entry['cta_group']:>9d}  {entry['m']:>3d}  {entry['n']:>3d}  {entry['k']:>3d}  "
            f"{entry['depth']:>5d}  {entry['binary']:<25s}  {entry['case_name']}"
        )
    lines.append(f"total invocations: {len(plan)}")
    return "\n".join(lines) + "\n"


def format_plan_lines(plan: list[dict]) -> str:
    return "".join(
        f"{entry['index']}\t{entry['pair_index']}\t{entry['method']}\t{entry['cta_group']}\t"
        f"{entry['m']}\t{entry['n']}\t{entry['k']}\t{entry['depth']}\t{entry['binary']}\t{entry['case_name']}\n"
        for entry in plan
    )


# ---------------------------------------------------------------------------
# Path safety: lstat-based, symlink-refusing, no-clobber primitives.
#
# create_campaign_dir/resolve_campaign_dir accept an optional raw_root,
# defaulting to the real REPO_ROOT for every production call site (the CLI
# subcommands below never pass it). --self-test injects a TemporaryDirectory
# as raw_root so every symlink-escape adversarial case can be exercised
# without ever creating, following, or touching anything under the real
# results/raw/ tree.
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
    symlink at this exact path component -- even a dangling one -- is refused
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


def _mkdir_component(path: Path, *, must_not_exist: bool, root: Path) -> None:
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
    _confirm_contained(path, root)


def create_campaign_dir(campaign_id: str, *, raw_root: Path | None = None) -> Path:
    """Centralized, symlink-safe campaign creation. Walks
    results/raw/exp02_umma_throughput/<campaign_id>/{cases,logs} one
    component at a time via lstat, rejecting a symlink or wrong-type object
    at any level (including the raw root itself), and fails if the campaign
    directory already exists. raw_root defaults to the real repository root;
    only --self-test overrides it, always with a TemporaryDirectory."""
    validate_campaign_id(campaign_id)
    root = REPO_ROOT if raw_root is None else raw_root
    current = root
    for part in RAW_ROOT_PARTS:
        current = current / part
        _mkdir_component(current, must_not_exist=False, root=root)
    campaign_dir = current / campaign_id
    _mkdir_component(campaign_dir, must_not_exist=True, root=root)
    for sub in ("cases", "logs"):
        _mkdir_component(campaign_dir / sub, must_not_exist=False, root=root)
    return campaign_dir


def resolve_campaign_dir(campaign_dir_rel: str, *, raw_root: Path | None = None) -> Path:
    """Resolves an already-initialized campaign directory with the same
    lstat-based symlink/type safety as create_campaign_dir. Requires exactly
    results/raw/exp02_umma_throughput/<campaign_id> (one campaign-ID
    component beneath the raw root, no nested/extra components)."""
    root = REPO_ROOT if raw_root is None else raw_root
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

    current = root
    for part in parts:
        current = current / part
        _reject_if_symlink_or_wrong_type(current, expect_dir=True)
        if not os.path.lexists(current):
            raise UnsafePathError(f"{current}: does not exist")
    _confirm_contained(current, root)
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
# Private-path redaction (audit repair): generated manifests, failure
# details, and P2.3-generated logs must never store this host's absolute
# repository path or user-specific prefix (AGENTS.md: never store home paths
# or unrelated host metadata in logs, results, or committed files). Only the
# exact literal REPO_ROOT prefix is ever replaced -- this is applied solely
# to diagnostic/error text, never to CSV or manifest scientific data.
# ---------------------------------------------------------------------------
_REPO_ROOT_STR = str(REPO_ROOT)


def _redact_repo_root(text: str) -> str:
    """Replaces a literal occurrence of the canonical absolute repository
    root with a stable repository-relative representation. A no-op unless
    the exact REPO_ROOT string is present."""
    if not text or _REPO_ROOT_STR not in text:
        return text
    return text.replace(_REPO_ROOT_STR + os.sep, "").replace(_REPO_ROOT_STR, ".")


def _redact_failure_detail(errors: list[str]) -> list[str]:
    return [_redact_repo_root(error) for error in errors]


def _eprint(message: str) -> None:
    """print(..., file=sys.stderr), with the absolute repository root
    redacted first. Used by every subcommand's diagnostic output, since the
    runner redirects several subcommands' stderr directly into
    results/raw/.../logs/*.log."""
    print(_redact_repo_root(message), file=sys.stderr)


# ---------------------------------------------------------------------------
# Strict, centralized per-field validation contract.
#
# FIELD_VALIDATORS maps every one of the 37 CSV_HEADER columns to a callable
# `(row, expect, errors, ctx) -> value | None` that parses and validates that
# field, appending human-readable errors and returning the canonical parsed
# value (or None on failure). A validator may read *other* raw string fields
# out of `row` directly (via the quiet `_peek_*` helpers, which never append
# a duplicate error -- the sibling field's own validator already reports its
# own parse failure) when a cross-field formula needs them. `expect` carries
# the a-priori-known values for one case: method, cta_group, m, grid_blocks,
# n, depth, run_kind, iterations, warmup_iterations, repetitions, git_commit.
# The module-level assertion below guarantees no column can be silently
# dropped from the contract by a future edit.
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


def _v_schema_version(row, expect, errors, ctx):
    if row["schema_version"] != SCHEMA_VERSION:
        errors.append(f"{ctx}: schema_version={row['schema_version']!r} != {SCHEMA_VERSION!r}")
        return None
    return row["schema_version"]


def _v_timestamp_utc(row, expect, errors, ctx):
    raw = row["timestamp_utc"]
    if not TIMESTAMP_UTC_RE.fullmatch(raw):
        errors.append(f"{ctx}: timestamp_utc={raw!r} is not in exact YYYY-MM-DDTHH:MM:SSZ form")
        return None
    try:
        _datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ")
    except (ValueError, TypeError):
        errors.append(
            f"{ctx}: timestamp_utc={raw!r} is not a real calendar timestamp in YYYY-MM-DDTHH:MM:SSZ form"
        )
        return None
    return raw


def _v_run_kind(row, expect, errors, ctx):
    if row["run_kind"] != expect["run_kind"]:
        errors.append(f"{ctx}: run_kind={row['run_kind']!r} != expected {expect['run_kind']!r}")
        return None
    return row["run_kind"]


def _v_publishable(row, expect, errors, ctx):
    if row["publishable"] != "false":
        errors.append(f"{ctx}: publishable={row['publishable']!r} != literal 'false'")
        return None
    return row["publishable"]


def _v_method(row, expect, errors, ctx):
    if row["method"] not in METHODS:
        errors.append(f"{ctx}: method={row['method']!r} not in {METHODS}")
        return None
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


def _make_exact_int_validator(field: str, expected_fn, *, allowed_set: set[int] | None = None):
    def validator(row, expect, errors, ctx):
        value = _parse_strict_int(row[field], errors, ctx, field)
        if value is None:
            return None
        if allowed_set is not None and value not in allowed_set:
            errors.append(f"{ctx}: {field}={value} not in {sorted(allowed_set)}")
        expected = expected_fn(expect)
        if value != expected:
            errors.append(f"{ctx}: {field}={value} != {expected} (expected)")
        return value
    return validator


def _make_zero_int_validator(field: str):
    def validator(row, expect, errors, ctx):
        value = _parse_strict_int(row[field], errors, ctx, field)
        if value is None:
            return None
        if value != 0:
            errors.append(f"{ctx}: {field}={value} != 0")
        return value
    return validator


def _make_positive_int_validator(field: str):
    def validator(row, expect, errors, ctx):
        value = _parse_strict_int(row[field], errors, ctx, field)
        if value is None:
            return None
        if value <= 0:
            errors.append(f"{ctx}: {field}={value} must be > 0")
        return value
    return validator


def _make_exact_str_validator(field: str, expected_fn):
    def validator(row, expect, errors, ctx):
        value = row[field]
        expected = expected_fn(expect)
        if value != expected:
            errors.append(f"{ctx}: {field}={value!r} != {expected!r} (expected)")
            return None
        return value
    return validator


def _v_iterations(row, expect, errors, ctx):
    value = _parse_strict_int(row["iterations"], errors, ctx, "iterations")
    if value is None:
        return None
    if value < 1:
        errors.append(f"{ctx}: iterations={value} must be >= 1")
    if value != expect["iterations"]:
        errors.append(f"{ctx}: iterations={value} != requested {expect['iterations']}")
    return value


def _v_warmup_iterations(row, expect, errors, ctx):
    value = _parse_strict_int(row["warmup_iterations"], errors, ctx, "warmup_iterations")
    if value is None:
        return None
    if value < 0:
        errors.append(f"{ctx}: warmup_iterations={value} must be >= 0")
    if value != expect["warmup_iterations"]:
        errors.append(f"{ctx}: warmup_iterations={value} != requested {expect['warmup_iterations']}")
    return value


def _v_repetitions(row, expect, errors, ctx):
    value = _parse_strict_int(row["repetitions"], errors, ctx, "repetitions")
    if value is None:
        return None
    if value < 1:
        errors.append(f"{ctx}: repetitions={value} must be >= 1")
    if value != expect["repetitions"]:
        errors.append(f"{ctx}: repetitions={value} != requested {expect['repetitions']}")
    return value


def _v_umma_per_iteration(row, expect, errors, ctx):
    value = _parse_strict_int(row["umma_per_iteration"], errors, ctx, "umma_per_iteration")
    if value is None:
        return None
    depth = _peek_int(row, "depth")
    if depth is not None and value != depth:
        errors.append(f"{ctx}: umma_per_iteration={value} != depth={depth} (formula: umma_per_iteration=depth)")
    return value


def _v_total_umma(row, expect, errors, ctx):
    value = _parse_strict_int(row["total_umma"], errors, ctx, "total_umma")
    if value is None:
        return None
    depth = _peek_int(row, "depth")
    iterations = _peek_int(row, "iterations")
    if depth is not None and iterations is not None:
        expected = depth * iterations
        if expected > INT64_MAX:
            errors.append(f"{ctx}: depth*iterations={expected} overflows the signed 64-bit range")
        elif value != expected:
            errors.append(f"{ctx}: total_umma={value} != depth*iterations={expected} (formula)")
    return value


def _v_flops_per_umma(row, expect, errors, ctx):
    value = _parse_strict_int(row["flops_per_umma"], errors, ctx, "flops_per_umma")
    if value is None:
        return None
    m = _peek_int(row, "m")
    n = _peek_int(row, "n")
    if m is not None and n is not None:
        expected = 2 * m * n * FROZEN_K
        if expected > INT64_MAX:
            errors.append(f"{ctx}: 2*m*n*k={expected} overflows the signed 64-bit range")
        elif value != expected:
            errors.append(f"{ctx}: flops_per_umma={value} != 2*m*n*k={expected} (formula)")
    return value


def _v_total_flops(row, expect, errors, ctx):
    value = _parse_strict_int(row["total_flops"], errors, ctx, "total_flops")
    if value is None:
        return None
    flops_per_umma = _peek_int(row, "flops_per_umma")
    total_umma = _peek_int(row, "total_umma")
    if flops_per_umma is not None and total_umma is not None:
        expected = flops_per_umma * total_umma
        if expected > INT64_MAX:
            errors.append(f"{ctx}: flops_per_umma*total_umma={expected} overflows the signed 64-bit range")
        elif value != expected:
            errors.append(f"{ctx}: total_flops={value} != flops_per_umma*total_umma={expected} (formula)")
    return value


def _v_elapsed_cycles(row, expect, errors, ctx):
    value = _parse_strict_int(row["elapsed_cycles"], errors, ctx, "elapsed_cycles")
    if value is None:
        return None
    if value <= 0:
        errors.append(f"{ctx}: elapsed_cycles={value} must be positive")
    return value


def _v_cycles_per_umma(row, expect, errors, ctx):
    value = _parse_strict_float(row["cycles_per_umma"], errors, ctx, "cycles_per_umma")
    if value is None:
        return None
    if value <= 0:
        errors.append(f"{ctx}: cycles_per_umma={value} must be positive")
        return value
    elapsed_cycles = _peek_int(row, "elapsed_cycles")
    total_umma = _peek_int(row, "total_umma")
    if elapsed_cycles is not None and total_umma is not None and total_umma > 0:
        expected = elapsed_cycles / total_umma
        if abs(value - expected) > RATIO_ABS_TOL:
            errors.append(
                f"{ctx}: cycles_per_umma={value} inconsistent with "
                f"elapsed_cycles/total_umma={expected} (tol={RATIO_ABS_TOL})"
            )
    return value


def _v_flops_per_cycle(row, expect, errors, ctx):
    value = _parse_strict_float(row["flops_per_cycle"], errors, ctx, "flops_per_cycle")
    if value is None:
        return None
    if value <= 0:
        errors.append(f"{ctx}: flops_per_cycle={value} must be positive")
        return value
    total_flops = _peek_int(row, "total_flops")
    elapsed_cycles = _peek_int(row, "elapsed_cycles")
    if total_flops is not None and elapsed_cycles is not None and elapsed_cycles > 0:
        expected = total_flops / elapsed_cycles
        if abs(value - expected) > RATIO_ABS_TOL:
            errors.append(
                f"{ctx}: flops_per_cycle={value} inconsistent with "
                f"total_flops/elapsed_cycles={expected} (tol={RATIO_ABS_TOL})"
            )
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
        errors.append(f"{ctx}: gpu_uuid={raw!r} does not match GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx")
        return None
    return raw


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


FIELD_VALIDATORS: dict[str, object] = {
    "schema_version": _v_schema_version,
    "timestamp_utc": _v_timestamp_utc,
    "run_kind": _v_run_kind,
    "publishable": _v_publishable,
    "method": _v_method,
    "sample_index": _v_sample_index,
    "cta_group": _make_exact_int_validator("cta_group", lambda e: e["cta_group"]),
    "m": _make_exact_int_validator("m", lambda e: e["m"]),
    "n": _make_exact_int_validator("n", lambda e: e["n"], allowed_set=set(FROZEN_N_VALUES)),
    "k": _make_exact_int_validator("k", lambda e: FROZEN_K),
    "depth": _make_exact_int_validator("depth", lambda e: e["depth"], allowed_set=set(FROZEN_DEPTH_VALUES)),
    "iterations": _v_iterations,
    "warmup_iterations": _v_warmup_iterations,
    "repetitions": _v_repetitions,
    "umma_per_iteration": _v_umma_per_iteration,
    "total_umma": _v_total_umma,
    "flops_per_umma": _v_flops_per_umma,
    "total_flops": _v_total_flops,
    "elapsed_cycles": _v_elapsed_cycles,
    "cycles_per_umma": _v_cycles_per_umma,
    "flops_per_cycle": _v_flops_per_cycle,
    "threads_per_cta": _make_exact_int_validator("threads_per_cta", lambda e: FROZEN_THREADS_PER_CTA),
    "grid_blocks": _make_exact_int_validator("grid_blocks", lambda e: e["grid_blocks"]),
    "tmem_columns": _make_exact_int_validator("tmem_columns", lambda e: e["n"]),
    "operand_path": _make_exact_str_validator("operand_path", lambda e: FROZEN_OPERAND_PATH),
    "input_type": _make_exact_str_validator("input_type", lambda e: FROZEN_INPUT_TYPE),
    "accumulator_type": _make_exact_str_validator("accumulator_type", lambda e: FROZEN_ACCUMULATOR_TYPE),
    "correctness": _make_exact_str_validator("correctness", lambda e: "OK"),
    "mismatches": _make_zero_int_validator("mismatches"),
    "max_abs_error": _make_zero_int_validator("max_abs_error"),
    "gpu_name": _v_gpu_name,
    "gpu_uuid": _v_gpu_uuid,
    "compute_capability": _make_exact_str_validator("compute_capability", lambda e: FROZEN_COMPUTE_CAPABILITY),
    "cuda_driver_version": _make_positive_int_validator("cuda_driver_version"),
    "cuda_runtime_version": _make_positive_int_validator("cuda_runtime_version"),
    "git_commit": _v_git_commit,
    "git_dirty": _make_exact_str_validator("git_dirty", lambda e: "false"),
}
assert set(FIELD_VALIDATORS) == set(CSV_HEADER), (
    "FIELD_VALIDATORS must cover exactly CSV_HEADER; "
    f"missing={set(CSV_HEADER) - set(FIELD_VALIDATORS)} "
    f"extra={set(FIELD_VALIDATORS) - set(CSV_HEADER)}"
)


def read_case_rows(path: Path) -> tuple[list[list[str]], list[str]]:
    """Reads the raw CSV with csv.reader in strict mode (never cut/awk/
    line-splitting). Strict mode (audit repair) makes broken or unterminated
    quoting a hard csv.Error instead of a silently lenient reinterpretation,
    so malformed CSV syntax can never be misparsed into something that
    happens to still pass the exact-header/column-count/semantic checks
    below. _open_regular_nofollow already opens with newline="" (required
    for correct csv module behavior). Returns (all_rows_including_header,
    errors). On a structural error the row list may be incomplete."""
    try:
        with _open_regular_nofollow(path, binary=False) as handle:
            rows = list(csv.reader(handle, strict=True))
    except (OSError, UnsafePathError, UnicodeError, csv.Error) as exc:
        return [], [f"{path}: unable to read: {exc}"]
    if not rows:
        return [], [f"{path}: file is empty (no header)"]
    return rows, []


def validate_case_file(path: Path, expect: dict) -> tuple[list[dict[str, str]], list[str]]:
    """Validates one case CSV in full. `expect` must contain: method,
    cta_group, m, grid_blocks, n, depth, run_kind, iterations,
    warmup_iterations, repetitions, git_commit. Every one of the 37
    CSV_HEADER fields is validated in every repetition via FIELD_VALIDATORS.
    Returns (parsed_data_rows, errors); errors is empty iff the file is fully
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
            errors.append(f"{path}: line {line_no}: expected {len(CSV_HEADER)} fields, got {len(row)}")
            continue
        parsed_rows.append(dict(zip(CSV_HEADER, row)))

    if errors:
        return parsed_rows, errors

    repetitions = expect["repetitions"]
    if len(parsed_rows) != repetitions:
        errors.append(f"{path}: has {len(parsed_rows)} data row(s), expected exactly repetitions={repetitions}")
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
        assert validated_fields == set(CSV_HEADER), f"internal error: row {row_num} did not run every field validator"

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
    "cuda_runtime_version", "git_commit", "git_dirty", "run_kind",
    "iterations", "warmup_iterations", "repetitions",
)


def check_cross_case_consistency(cases: list[tuple[dict, list[dict[str, str]]]]) -> list[str]:
    """Compares *every* repetition of *every* case against one single
    reference row (the very first row of the very first case), not just
    each case's rows[0]. A common field changed only in a later sample of an
    otherwise-valid case is therefore always caught."""
    errors: list[str] = []
    all_rows: list[tuple[dict, dict[str, str]]] = [(entry, row) for entry, rows in cases for row in rows]
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
                errors.append(f"case {entry['case_name']}: repetition count {len(rows)} != {reference_repetitions}")
    return errors


def scan_case_directory(cases_dir: Path, plan: list[dict]) -> tuple[dict[int, Path], list[str]]:
    """Require the exact 24 case filenames generated by the frozen plan.

    Filenames are compared literally rather than parsed back into integers:
    semantically equivalent spellings such as ``n064`` are not canonical and
    must not replace the ``n64`` path referenced by execution_order.csv.
    Returns (index -> path for exactly the 24 expected indices, errors).
    Every directory entry is inspected via lstat (never following a
    symlink); the cases/ directory must contain exactly the 24 expected
    filenames, each a regular non-symlink file. Anything else -- a
    non-canonical .csv name, a non-.csv regular file (.partial/.invalid
    salvage evidence, notes.txt, a hidden file, ...), a subdirectory, a
    FIFO, or a symlink (dangling or not) at any expected or unexpected
    filename -- is reported as an error, never silently ignored or
    aggregated (audit repair: a non-.csv entry was previously skipped with a
    bare `continue`, so a stray file such as cases/notes.txt could sit next
    to a complete 24-case set without ever being flagged)."""
    errors: list[str] = []
    by_index = plan_by_index(plan)
    expected_by_filename = {f"{entry['case_name']}.csv": entry for entry in plan}
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
        expected_entry = expected_by_filename.get(path.name)
        if expected_entry is None:
            errors.append(f"{path}: unexpected case filename; expected an exact filename from the frozen plan")
            continue
        index = expected_entry["index"]
        if index in found:
            errors.append(f"{path}: duplicate configuration for index {index} (already have {found[index]})")
            continue
        found[index] = path

    for index in sorted(by_index):
        if index not in found:
            errors.append(f"{cases_dir}: missing configuration index {index} ({by_index[index]['case_name']}.csv)")

    return found, errors


# ---------------------------------------------------------------------------
# execution_order.csv: created once at campaign init, re-validated at
# finalize time.
# ---------------------------------------------------------------------------
def _execution_order_row(entry: dict) -> list[str]:
    return [
        str(entry["index"]), str(entry["pair_index"]), entry["method"], str(entry["cta_group"]),
        str(entry["m"]), str(entry["n"]), str(entry["k"]), str(entry["depth"]), entry["binary"],
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
    """Strictly validates an existing execution_order.csv against
    build_plan(): exact header, exactly len(plan) rows, in order, matching
    every derived column and case_file path. A missing file, a symlink, a
    malformed or reordered header, or an extra/missing row are all
    rejected."""
    if os.path.lexists(path):
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode):
            return [f"{path}: is a symlink; refusing"]
    else:
        return [f"{path}: does not exist"]

    try:
        with _open_regular_nofollow(path, binary=False) as handle:
            rows = list(csv.reader(handle, strict=True))
    except (OSError, UnsafePathError, UnicodeError, csv.Error) as exc:
        return [f"{path}: unable to read: {exc}"]
    if not rows:
        return [f"{path}: empty file"]

    header, data_rows = rows[0], rows[1:]
    if header != EXECUTION_ORDER_HEADER:
        return [f"{path}: header mismatch: {header!r}"]
    if len(data_rows) != len(plan):
        return [f"{path}: has {len(data_rows)} row(s), expected {len(plan)}"]

    errors: list[str] = []
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
    output. A final target or its deterministic temporary name existing in
    any form is fatal; nothing is removed or overwritten."""
    errors: list[str] = []
    for path in paths:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        for candidate, label in ((path, "target"), (tmp_path, "temporary")):
            if os.path.lexists(candidate):
                errors.append(f"{candidate}: existing aggregate {label}; refusing to overwrite")
    return errors


def write_combined_samples(plan: list[dict], cases: list[tuple[dict, list[dict[str, str]]]], out_path: Path) -> int:
    """Writes combined_samples.csv: every validated case-CSV field preserved
    unchanged, prefixed with stable traceability fields
    (invocation_index, pair_index, case_name), in deterministic execution-plan
    order with increasing sample_index within each invocation. Never
    overwrites an existing file. Returns the number of data rows written."""
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
            writer.writerow(COMBINED_HEADER)
            for entry in plan:
                rows = sorted(rows_by_index[entry["index"]], key=lambda r: int(r["sample_index"]))
                for row in rows:
                    writer.writerow(
                        [entry["index"], entry["pair_index"], entry["case_name"]]
                        + [row[field] for field in CSV_HEADER]
                    )
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
    "invocation_index", "pair_index", "case_name", "method", "cta_group", "m", "n", "k", "depth",
    "grid_blocks", "run_kind", "iterations", "warmup_iterations", "repetitions", "sample_count",
    "elapsed_cycles_mean", "elapsed_cycles_median", "elapsed_cycles_stdev", "elapsed_cycles_cv_percent",
    "elapsed_cycles_min", "elapsed_cycles_max",
    "cycles_per_umma_mean", "cycles_per_umma_median", "cycles_per_umma_stdev", "cycles_per_umma_cv_percent",
    "cycles_per_umma_min", "cycles_per_umma_max",
    "flops_per_cycle_mean", "flops_per_cycle_median", "flops_per_cycle_stdev", "flops_per_cycle_cv_percent",
    "flops_per_cycle_min", "flops_per_cycle_max",
    "correctness", "mismatches",
]


def _sample_stdev(values: list[float]) -> float:
    """Sample standard deviation. Raises rather than silently reporting 0.0
    for a single observation (audit repair): with REPETITIONS_MIN=2 enforced
    upstream (CLI, requested-parameter, and manifest validation), a
    real, validated case's rows must never number fewer than two, so
    reaching this branch indicates an upstream validation gap, not a
    legitimate single-sample statistic."""
    if len(values) < 2:
        raise ValueError(f"sample standard deviation requires at least 2 observations, got {len(values)}")
    return statistics.stdev(values)


def _numeric_stats(values: list[float]) -> dict[str, float]:
    mean = float(statistics.mean(values))
    stdev = _sample_stdev(values)
    return {
        "mean": mean,
        "median": float(statistics.median(values)),
        "stdev": stdev,
        "cv_percent": (100.0 * stdev / mean) if mean != 0 else 0.0,
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_case(entry: dict, rows: list[dict[str, str]]) -> dict:
    elapsed = [float(r["elapsed_cycles"]) for r in rows]
    cycles_per_umma = [float(r["cycles_per_umma"]) for r in rows]
    flops_per_cycle = [float(r["flops_per_cycle"]) for r in rows]
    row0 = rows[0]
    ec, cp, fp = _numeric_stats(elapsed), _numeric_stats(cycles_per_umma), _numeric_stats(flops_per_cycle)
    summary = {
        "invocation_index": entry["index"],
        "pair_index": entry["pair_index"],
        "case_name": entry["case_name"],
        "method": entry["method"],
        "cta_group": entry["cta_group"],
        "m": entry["m"],
        "n": entry["n"],
        "k": entry["k"],
        "depth": entry["depth"],
        "grid_blocks": entry["grid_blocks"],
        "run_kind": row0["run_kind"],
        "iterations": row0["iterations"],
        "warmup_iterations": row0["warmup_iterations"],
        "repetitions": row0["repetitions"],
        "sample_count": len(rows),
        "correctness": "OK",
        "mismatches": 0,
    }
    for prefix, stats in (("elapsed_cycles", ec), ("cycles_per_umma", cp), ("flops_per_cycle", fp)):
        for stat_name, value in stats.items():
            summary[f"{prefix}_{stat_name}"] = value
    return summary


def format_summary_value(field: str, value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_summary(cases: list[tuple[dict, list[dict[str, str]]]], out_path: Path) -> int:
    """Writes summary.csv: exactly 24 rows sorted by (n, depth, method) --
    i.e. by logical configuration, independent of the alternating execution
    order -- with descriptive statistics only (count via sample_count, mean,
    median, sample standard deviation, coefficient of variation, minimum,
    maximum). No speedups, no rankings, no saturation classification. Never
    overwrites an existing file."""
    if os.path.lexists(out_path):
        raise UnsafePathError(f"refusing to overwrite existing target: {out_path}")
    summaries = [summarize_case(entry, rows) for entry, rows in cases]
    summaries.sort(key=lambda s: (s["n"], s["depth"], s["method"]))
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
        raise ManifestTransitionError(f"{path}: manifest root has type {type(document).__name__}, expected object")
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
        raise ManifestTransitionError(f"{tmp_path}: existing manifest temporary; refusing to overwrite")

    created_tmp = False
    try:
        with _open_exclusive(tmp_path, binary=False) as handle:
            created_tmp = True
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

        if prior_identity is None:
            if os.path.lexists(path):
                raise ManifestTransitionError(f"{path}: appeared during initial manifest publication; refusing to overwrite")
        elif not os.path.lexists(path) or _file_identity(path) != prior_identity:
            raise ManifestTransitionError(f"{path}: changed during manifest update; refusing to overwrite")
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
                f"manifest field {key!r} has invalid type {type(value).__name__}, expected {expected_type}"
            )


def _validate_compact_timestamp(value: object, field: str) -> None:
    if not isinstance(value, str) or not MANIFEST_TIMESTAMP_RE.fullmatch(value):
        raise ManifestTransitionError(f"manifest field {field!r}={value!r} is not YYYYMMDDTHHMMSSZ")
    try:
        _datetime.strptime(value, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise ManifestTransitionError(f"manifest field {field!r}={value!r} is not a real UTC timestamp") from exc


def _validate_requested(requested: object) -> None:
    expected_keys = {"run_kind", "iterations", "warmup_iterations", "repetitions", "campaign_id"}
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
        raise ManifestTransitionError(f"manifest requested.campaign_id is invalid: {exc}") from exc
    bounds = {
        "iterations": (ITERATIONS_MIN, ITERATIONS_MAX),
        "warmup_iterations": (WARMUP_ITERATIONS_MIN, WARMUP_ITERATIONS_MAX),
        "repetitions": (REPETITIONS_MIN, REPETITIONS_MAX),
    }
    for key, (lo, hi) in bounds.items():
        value = requested[key]
        if type(value) is not int or value < lo or value > hi:
            raise ManifestTransitionError(f"manifest requested.{key} must be an integer in [{lo},{hi}]")


SELF_TEST_OUTCOME_STATES = frozenset({"PENDING", "STARTED", "PASS", "FAIL", "NOT_RUN"})

# Per-method self-test outcome lifecycle (audit repair). PENDING is the
# initial value for a method that has not started yet (recorded together
# with its sibling's STARTED, so both keys always exist -- the manifest
# schema requires exactly {"umma_1sm", "umma_2sm"} at all times). STARTED is
# recorded immediately before that method's device self-test runs, so an
# interrupted running self-test is never later reported as PASS. NOT_RUN is
# reserved for a method whose sibling failed first, so it never runs at all
# -- an invented PASS/FAIL is never recorded for it. PASS/FAIL/NOT_RUN are
# terminal: once recorded for a method, that method's outcome can never
# change again (enforced in merge_manifest via _check_self_test_transition).
SELF_TEST_TRANSITIONS: dict[str | None, frozenset[str]] = {
    # None (no outcome recorded yet) may go straight to any state, including
    # a terminal one -- this lets a manifest be initialized directly with a
    # known-final outcome (e.g. a synthetic test fixture, or -- in the real
    # runner -- a value that was already PASS by the time this exact key was
    # first written). What matters is what happens *after* a value is
    # recorded, not what the very first recorded value is allowed to be.
    None: frozenset({"PENDING", "STARTED", "PASS", "FAIL", "NOT_RUN"}),
    "PENDING": frozenset({"STARTED", "NOT_RUN"}),
    "STARTED": frozenset({"PASS", "FAIL"}),
    "PASS": frozenset(),
    "FAIL": frozenset(),
    "NOT_RUN": frozenset(),
}


def _validate_self_test_outcomes(value: object, *, require_pass: bool) -> None:
    if not isinstance(value, dict) or set(value) != {"umma_1sm", "umma_2sm"}:
        raise ManifestTransitionError("manifest self_test_outcomes must contain exactly umma_1sm and umma_2sm")
    allowed = {"PASS"} if require_pass else SELF_TEST_OUTCOME_STATES
    for method in METHODS:
        if value[method] not in allowed:
            raise ManifestTransitionError(f"manifest self_test_outcomes.{method}={value[method]!r} must be one of {sorted(allowed)}")


def _check_self_test_transition(old_outcomes: object, new_outcomes: dict) -> None:
    """Enforces the per-method SELF_TEST_TRANSITIONS lifecycle above instead
    of the blanket "the whole dict can never change" rule this replaces:
    PENDING/STARTED may still advance, but PASS/FAIL/NOT_RUN are terminal
    per method. A method re-recorded with its already-current value (e.g. a
    retried manifest-write) is always accepted."""
    old = old_outcomes if isinstance(old_outcomes, dict) else {}
    for method in METHODS:
        old_value = old.get(method)
        new_value = new_outcomes.get(method)
        if old_value == new_value:
            continue
        if new_value not in SELF_TEST_TRANSITIONS.get(old_value, frozenset()):
            raise ManifestTransitionError(
                f"manifest self_test_outcomes.{method}: illegal transition {old_value!r} -> {new_value!r}"
            )


def _validate_versions_dict(value: object) -> None:
    if not isinstance(value, dict) or set(value) != set(REQUIRED_VERSION_KEYS):
        raise ManifestTransitionError(
            f"versions_env keys={sorted(value) if isinstance(value, dict) else None} != required {sorted(REQUIRED_VERSION_KEYS)}"
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
            f"manifest {field} keys={sorted(value) if isinstance(value, dict) else None} != {sorted(expected_keys)}"
        )
    for key, digest in value.items():
        if not _is_sha256_hex(digest):
            raise ManifestTransitionError(f"manifest {field}.{key}={digest!r} is not a lowercase SHA-256")


def _expected_invocation_order() -> list[dict]:
    return [
        {
            "index": e["index"], "pair_index": e["pair_index"], "method": e["method"],
            "cta_group": e["cta_group"], "m": e["m"], "n": e["n"], "k": e["k"],
            "depth": e["depth"], "grid_blocks": e["grid_blocks"], "binary": e["binary"],
            "case_name": e["case_name"],
        }
        for e in build_plan()
    ]


def _validate_manifest_document(manifest: dict, *, require_initialized: bool = False) -> None:
    """Validate the entire loaded manifest, not only the proposed update.

    IN_PROGRESS documents may omit fields that are populated later, but any
    field already present must be allowlisted, correctly typed, and have a
    valid nested schema. Finalization requests the initialized base schema;
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
        if not manifest["gpu_name"].strip() or any(ord(char) < 0x20 or ord(char) == 0x7F for char in manifest["gpu_name"]):
            raise ManifestTransitionError("manifest gpu_name is empty or contains control characters")
    if "gpu_uuid" in manifest and not GPU_UUID_RE.fullmatch(manifest["gpu_uuid"]):
        raise ManifestTransitionError("manifest gpu_uuid has an invalid NVIDIA UUID form")
    if "compute_capability" in manifest and manifest["compute_capability"] != FROZEN_COMPUTE_CAPABILITY:
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
                raise ManifestTransitionError("manifest sample_count_expected disagrees with configurations*repetitions")
        if "configuration_count_completed" in manifest and "sample_count_completed" in manifest:
            if manifest["sample_count_completed"] != manifest["configuration_count_completed"] * repetitions:
                raise ManifestTransitionError("manifest sample_count_completed disagrees with configurations*repetitions")

    if "self_test_outcomes" in manifest:
        _validate_self_test_outcomes(manifest["self_test_outcomes"], require_pass=status == "COMPLETE")
    if "versions_env" in manifest:
        _validate_versions_dict(manifest["versions_env"])
    if "binary_and_sass_sha256" in manifest:
        _validate_hash_map(manifest["binary_and_sass_sha256"], set(DEFAULT_FINAL_ARTIFACTS), "binary_and_sass_sha256")
    if "case_file_sha256" in manifest:
        _validate_hash_map(
            manifest["case_file_sha256"], {entry["case_name"] for entry in build_plan()}, "case_file_sha256",
        )
    if "aggregate_file_sha256" in manifest:
        _validate_hash_map(manifest["aggregate_file_sha256"], {"combined_samples.csv", "summary.csv"}, "aggregate_file_sha256")
    if "execution_order_sha256" in manifest and not _is_sha256_hex(manifest["execution_order_sha256"]):
        raise ManifestTransitionError("manifest execution_order_sha256 is not a lowercase SHA-256")
    if "failure_detail" in manifest and manifest["failure_detail"] is not None:
        if not all(isinstance(item, str) for item in manifest["failure_detail"]):
            raise ManifestTransitionError("manifest failure_detail must be null or a list of strings")
    if "failure_stage" in manifest and manifest["failure_stage"] is not None:
        if not manifest["failure_stage"]:
            raise ManifestTransitionError("manifest failure_stage must be null or non-empty")

    if "invocation_order" in manifest and manifest["invocation_order"] != _expected_invocation_order():
        raise ManifestTransitionError("manifest invocation_order does not match the frozen plan")

    initialized_required = {
        "schema_version", "experiment_id", "campaign_id", "status", "run_kind",
        "started_at_utc", "configuration_count_expected",
        "configuration_count_completed", "sample_count_expected",
        "sample_count_completed", "requested", "selected_gpu_index",
        "git_commit", "git_dirty", "publishable", "self_test_outcomes",
    }
    if require_initialized:
        missing = initialized_required - set(manifest)
        if missing:
            raise ManifestTransitionError(f"initialized manifest missing fields: {sorted(missing)}")
        if manifest["configuration_count_expected"] != EXPECTED_CONFIGURATION_COUNT:
            raise ManifestTransitionError(f"manifest configuration_count_expected must be {EXPECTED_CONFIGURATION_COUNT}")
        if manifest["configuration_count_completed"] > EXPECTED_CONFIGURATION_COUNT:
            raise ManifestTransitionError(f"manifest configuration_count_completed exceeds {EXPECTED_CONFIGURATION_COUNT}")

    if status == "COMPLETE":
        complete_required = initialized_required | {
            "completed_at_utc", "invocation_order", "gpu_name", "gpu_uuid",
            "compute_capability", "cuda_driver_version", "cuda_runtime_version",
            "versions_env", "binary_and_sass_sha256", "case_file_sha256",
            "execution_order_sha256", "aggregate_file_sha256",
            "self_test_outcomes", "failure_stage", "failure_detail",
        }
        missing = complete_required - set(manifest)
        if missing:
            raise ManifestTransitionError(f"COMPLETE manifest missing fields: {sorted(missing)}")
        if manifest["configuration_count_completed"] != EXPECTED_CONFIGURATION_COUNT:
            raise ManifestTransitionError(f"COMPLETE manifest must have {EXPECTED_CONFIGURATION_COUNT} configurations")
        if manifest["sample_count_completed"] != manifest["sample_count_expected"]:
            raise ManifestTransitionError("COMPLETE manifest sample counts are incomplete")
        if manifest["failure_stage"] is not None or manifest["failure_detail"] is not None:
            raise ManifestTransitionError("COMPLETE manifest cannot retain failure metadata")
        _validate_self_test_outcomes(manifest["self_test_outcomes"], require_pass=True)
        _validate_versions_dict(manifest["versions_env"])


def merge_manifest(campaign_dir: Path, updates: dict, status: str, *, allow_complete: bool = False) -> dict:
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
        raise ManifestTransitionError(f"invalid manifest state transition: {current_status!r} -> {status!r}")
    immutable_after_init = {
        "campaign_id", "run_kind", "started_at_utc",
        "configuration_count_expected", "sample_count_expected", "requested",
        "selected_gpu_index", "git_commit", "git_dirty",
    }
    for key in immutable_after_init & set(manifest) & set(updates):
        if updates[key] != manifest[key]:
            raise ManifestTransitionError(f"manifest field {key!r} is immutable after initialization")
    for key in ("configuration_count_completed", "sample_count_completed"):
        if key in manifest and key in updates and updates[key] < manifest[key]:
            raise ManifestTransitionError(f"manifest counter {key!r} cannot decrease")
    if "self_test_outcomes" in updates:
        _check_self_test_transition(manifest.get("self_test_outcomes"), updates["self_test_outcomes"])
    manifest.update(updates)
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["experiment_id"] = EXPERIMENT_ID
    manifest["status"] = status
    manifest["publishable"] = False
    _validate_manifest_document(manifest, require_initialized=(status == "COMPLETE"))
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
            raise ManifestTransitionError(f"VERSIONS.env line {line_no}: expected KEY=VALUE")
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


def _expect_for_entry(entry: dict, *, run_kind: str, iterations: int, warmup_iterations: int, repetitions: int, git_commit: str) -> dict:
    info = METHOD_INFO[entry["method"]]
    return {
        "method": entry["method"],
        "cta_group": info["cta_group"],
        "m": info["m"],
        "grid_blocks": info["grid_blocks"],
        "n": entry["n"],
        "depth": entry["depth"],
        "run_kind": run_kind,
        "iterations": iterations,
        "warmup_iterations": warmup_iterations,
        "repetitions": repetitions,
        "git_commit": git_commit,
    }


# ---------------------------------------------------------------------------
# Subcommand: init-campaign
# ---------------------------------------------------------------------------
def _do_init_campaign(
    *, campaign_id: str, run_kind: str, iterations: int, warmup_iterations: int, repetitions: int,
    git_commit: str, gpu_index: int, started_at_utc: str, raw_root: Path | None = None,
) -> Path:
    """Core logic behind the init-campaign subcommand, reused directly by
    the self-test (no subprocess) to build realistic fixtures."""
    campaign_dir = create_campaign_dir(campaign_id, raw_root=raw_root)
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
            "run_kind": run_kind, "iterations": iterations, "warmup_iterations": warmup_iterations,
            "repetitions": repetitions, "campaign_id": campaign_id,
        },
        "selected_gpu_index": gpu_index,
        "git_commit": git_commit,
        "git_dirty": False,
        "self_test_outcomes": {"umma_1sm": "PENDING", "umma_2sm": "PENDING"},
    }
    merge_manifest(campaign_dir, updates, status="IN_PROGRESS")
    return campaign_dir


def cmd_init_campaign(args: argparse.Namespace) -> int:
    try:
        campaign_dir = _do_init_campaign(
            campaign_id=args.campaign_id, run_kind=args.run_kind, iterations=args.iterations,
            warmup_iterations=args.warmup_iterations, repetitions=args.repetitions,
            git_commit=args.git_commit, gpu_index=args.gpu_index, started_at_utc=args.started_at_utc,
        )
    except (UnsafePathError, ManifestTransitionError, ValueError) as exc:
        _eprint(f"aggregate_exp02_umma_throughput: init-campaign: ERROR: {exc}")
        return 2
    print(str(campaign_dir.relative_to(REPO_ROOT)))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: capture
# ---------------------------------------------------------------------------
def _do_capture(
    campaign_dir: Path, out_rel: str, binary_argv: list[str], *, artifact_paths: dict[str, Path] | None = None,
) -> int:
    try:
        out_path = resolve_capture_out_path(campaign_dir, out_rel)
    except UnsafePathError as exc:
        _eprint(f"aggregate_exp02_umma_throughput: capture: ERROR: {exc}")
        return 2

    if binary_argv and binary_argv[0] == "--":
        binary_argv = binary_argv[1:]  # argparse.REMAINDER keeps a literal '--' marker
    if not binary_argv:
        _eprint("aggregate_exp02_umma_throughput: capture: ERROR: no binary command given after '--'")
        return 2
    if binary_argv[0] not in ALLOWED_BINARIES:
        _eprint(
            f"aggregate_exp02_umma_throughput: capture: ERROR: binary must be exactly one of "
            f"{sorted(ALLOWED_BINARIES)}, got {binary_argv[0]!r}"
        )
        return 2
    artifact_paths = DEFAULT_CAPTURE_ARTIFACTS if artifact_paths is None else artifact_paths
    artifact_path = artifact_paths.get(binary_argv[0])
    if artifact_path is None:
        _eprint(
            f"aggregate_exp02_umma_throughput: capture: ERROR: no verified artifact path supplied for {binary_argv[0]!r}"
        )
        return 2
    artifact_err = _verify_artifact(artifact_path)
    if artifact_err:
        _eprint(f"aggregate_exp02_umma_throughput: capture: ERROR: {artifact_err}")
        return 2

    tmp_path = out_path.with_name(out_path.name + ".tmp")
    if os.path.lexists(tmp_path):
        _eprint(f"aggregate_exp02_umma_throughput: capture: ERROR: stale temporary file already exists: {tmp_path}")
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
                    _eprint(f"aggregate_exp02_umma_throughput: capture: preserved evidence as {failed_path.name}")
                    return
                except UnsafePathError:
                    if os.path.lexists(tmp_path):
                        continue
                    return
            _eprint("aggregate_exp02_umma_throughput: capture: ERROR: could not find a free failure-evidence name")
        else:
            if os.path.lexists(tmp_path):
                _safe_unlink_owned(tmp_path)

    _eprint(f"aggregate_exp02_umma_throughput: capture: running {binary_argv!r} -> {out_path.name}")
    try:
        csv_out = _open_exclusive(tmp_path, binary=True)
    except UnsafePathError as exc:
        _eprint(f"aggregate_exp02_umma_throughput: capture: ERROR: cannot create owned temporary: {exc}")
        return 2
    try:
        with csv_out:
            result = subprocess.run(binary_argv, stdout=csv_out, stderr=None)
    except OSError as exc:
        _eprint(f"aggregate_exp02_umma_throughput: capture: ERROR: unable to launch binary: {exc}")
        salvage(is_signal=False)
        return 1

    if result.returncode == 0:
        if not os.path.lexists(tmp_path) or os.lstat(tmp_path).st_size == 0:
            _eprint("aggregate_exp02_umma_throughput: capture: ERROR: binary exited 0 but produced no stdout")
            salvage(is_signal=False)
            return 1
        try:
            _publish_no_clobber(tmp_path, out_path)
        except UnsafePathError as exc:
            _eprint(f"aggregate_exp02_umma_throughput: capture: ERROR: {exc}")
            salvage(is_signal=False)
            return 1
        _eprint(f"aggregate_exp02_umma_throughput: capture: OK: wrote {out_path}")
        return 0

    _eprint(f"aggregate_exp02_umma_throughput: capture: ERROR: binary exited {result.returncode}")
    salvage(is_signal=result.returncode < 0)
    return 1


def cmd_capture(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_campaign_dir(args.campaign_dir)
    except UnsafePathError as exc:
        _eprint(f"aggregate_exp02_umma_throughput: capture: ERROR: {exc}")
        return 2
    return _do_capture(campaign_dir, args.out, list(args.binary_args))


# ---------------------------------------------------------------------------
# Subcommand: validate-case
# ---------------------------------------------------------------------------
def cmd_validate_case(args: argparse.Namespace) -> int:
    plan = build_plan()
    entry = plan_by_index(plan).get(args.index)
    if entry is None:
        _eprint(
            f"aggregate_exp02_umma_throughput: validate-case: ERROR: index {args.index} "
            f"is not in 0..{EXPECTED_CONFIGURATION_COUNT - 1}"
        )
        return 2

    try:
        campaign_dir = resolve_campaign_dir(args.campaign_dir)
    except UnsafePathError as exc:
        _eprint(f"aggregate_exp02_umma_throughput: validate-case: ERROR: {exc}")
        return 2

    case_path = campaign_dir / "cases" / f"{entry['case_name']}.csv"
    expect = _expect_for_entry(
        entry, run_kind=args.run_kind, iterations=args.iterations,
        warmup_iterations=args.warmup_iterations, repetitions=args.repetitions, git_commit=args.git_commit,
    )
    _, errors = validate_case_file(case_path, expect)
    if errors:
        _eprint(f"aggregate_exp02_umma_throughput: validate-case: FAIL: {case_path}")
        for error in errors:
            _eprint(f"aggregate_exp02_umma_throughput: validate-case:   - {error}")
        return 1
    _eprint(f"aggregate_exp02_umma_throughput: validate-case: OK: {case_path}")
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
    if requested.get("iterations") != args.iterations:
        errors.append(f"manifest requested.iterations={requested.get('iterations')!r} != CLI iterations={args.iterations!r}")
    if requested.get("warmup_iterations") != args.warmup_iterations:
        errors.append(
            f"manifest requested.warmup_iterations={requested.get('warmup_iterations')!r} != "
            f"CLI warmup_iterations={args.warmup_iterations!r}"
        )
    if requested.get("repetitions") != args.repetitions:
        errors.append(f"manifest requested.repetitions={requested.get('repetitions')!r} != CLI repetitions={args.repetitions!r}")
    if manifest.get("git_commit") != args.git_commit:
        errors.append(f"manifest git_commit={manifest.get('git_commit')!r} != CLI git_commit={args.git_commit!r}")
    if manifest.get("selected_gpu_index") != args.gpu_index:
        errors.append(f"manifest selected_gpu_index={manifest.get('selected_gpu_index')!r} != CLI gpu_index={args.gpu_index!r}")
    if manifest.get("started_at_utc") != args.started_at_utc:
        errors.append(f"manifest started_at_utc={manifest.get('started_at_utc')!r} != CLI started_at_utc={args.started_at_utc!r}")
    if args.self_test_umma_1sm != "PASS":
        errors.append(f"CLI self_test_umma_1sm={args.self_test_umma_1sm!r} != 'PASS'")
    if args.self_test_umma_2sm != "PASS":
        errors.append(f"CLI self_test_umma_2sm={args.self_test_umma_2sm!r} != 'PASS'")
    self_test = manifest.get("self_test_outcomes", {})
    if self_test.get("umma_1sm") != "PASS":
        errors.append(f"manifest self_test_outcomes.umma_1sm={self_test.get('umma_1sm')!r} != 'PASS'")
    if self_test.get("umma_2sm") != "PASS":
        errors.append(f"manifest self_test_outcomes.umma_2sm={self_test.get('umma_2sm')!r} != 'PASS'")
    if manifest.get("configuration_count_completed") != EXPECTED_CONFIGURATION_COUNT:
        errors.append(
            f"manifest configuration_count_completed={manifest.get('configuration_count_completed')!r} != "
            f"{EXPECTED_CONFIGURATION_COUNT}"
        )
    if manifest.get("sample_count_completed") != EXPECTED_CONFIGURATION_COUNT * args.repetitions:
        errors.append(
            f"manifest sample_count_completed={manifest.get('sample_count_completed')!r} != "
            f"{EXPECTED_CONFIGURATION_COUNT * args.repetitions}"
        )
    return errors


def _do_finalize(
    campaign_dir: Path, args, *, artifact_paths: dict[str, Path] | None = None, versions_path: Path | None = None,
) -> tuple[bool, list[str]]:
    """Core finalize logic (no argparse/print), reused directly by the
    self-test. Returns (success, errors). On success, manifest.json has
    already been written with status=COMPLETE and every mandatory hash; on
    failure, manifest.json has been written with status=FAILED unless the
    campaign was already terminal (which is never rewritten)."""
    plan = build_plan()
    plan_errors = check_plan_contract(plan)
    if plan_errors:
        return False, [f"internal plan contract violation: {plan_errors}"]

    def fail(stage: str, errors: list[str]) -> tuple[bool, list[str]]:
        try:
            manifest = load_manifest(campaign_dir)
            if manifest.get("status") in (None, "IN_PROGRESS"):
                # Audit repair: redact the absolute repository root from
                # every error string before it is persisted, so a
                # validation failure (which routinely embeds absolute
                # campaign/case paths built from REPO_ROOT) never stores
                # this host's absolute path in the manifest.
                merge_manifest(
                    campaign_dir,
                    {"failure_stage": stage, "failure_detail": _redact_failure_detail(errors[:50])},
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

    cases: list[tuple[dict, list[dict[str, str]]]] = []
    all_errors: list[str] = []
    for index in sorted(found):
        entry = plan_by_index(plan)[index]
        expect = _expect_for_entry(
            entry, run_kind=args.run_kind, iterations=args.iterations,
            warmup_iterations=args.warmup_iterations, repetitions=args.repetitions, git_commit=args.git_commit,
        )
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
        artifact_errors.append(f"artifact labels={sorted(artifact_paths)} != {sorted(DEFAULT_FINAL_ARTIFACTS)}")
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
        case_hashes = {plan_by_index(plan)[index]["case_name"]: sha256_of(found[index]) for index in sorted(found)}
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
        errors = [f"combined_samples.csv has {combined_rows} rows, expected {expected_rows}", *rollback_published()]
        return fail("consolidation", errors)
    if summary_rows != EXPECTED_CONFIGURATION_COUNT:
        errors = [f"summary.csv has {summary_rows} rows, expected {EXPECTED_CONFIGURATION_COUNT}", *rollback_published()]
        return fail("aggregation", errors)

    try:
        aggregate_hashes = {"combined_samples.csv": sha256_of(combined_path), "summary.csv": sha256_of(summary_path)}
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
            "run_kind": args.run_kind, "iterations": args.iterations,
            "warmup_iterations": args.warmup_iterations, "repetitions": args.repetitions,
            "campaign_id": args.campaign_id,
        },
        "invocation_order": _expected_invocation_order(),
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
        "self_test_outcomes": {"umma_1sm": args.self_test_umma_1sm, "umma_2sm": args.self_test_umma_2sm},
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
        _eprint(f"aggregate_exp02_umma_throughput: finalize: ERROR: {exc}")
        return 2

    success, errors = _do_finalize(campaign_dir, args)
    if not success:
        _eprint("aggregate_exp02_umma_throughput: finalize: ERROR:")
        for error in errors:
            _eprint(f"aggregate_exp02_umma_throughput: finalize:   - {error}")
        return 1
    _eprint(f"aggregate_exp02_umma_throughput: finalize: OK: campaign {args.campaign_id} COMPLETE")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: manifest-write (IN_PROGRESS/FAILED/INTERRUPTED only)
# ---------------------------------------------------------------------------
def cmd_manifest_write(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_campaign_dir(args.campaign_dir)
    except UnsafePathError as exc:
        _eprint(f"aggregate_exp02_umma_throughput: manifest-write: ERROR: {exc}")
        return 2

    updates: dict = {}
    if args.merge_json:
        merge_path = Path(args.merge_json)
        try:
            updates = json.loads(merge_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _eprint(f"aggregate_exp02_umma_throughput: manifest-write: ERROR: cannot read --merge-json: {exc}")
            return 2
        if not isinstance(updates, dict):
            _eprint("aggregate_exp02_umma_throughput: manifest-write: ERROR: --merge-json must contain a JSON object")
            return 2

    try:
        merge_manifest(campaign_dir, updates, status=args.status)
    except ManifestTransitionError as exc:
        _eprint(f"aggregate_exp02_umma_throughput: manifest-write: ERROR: {exc}")
        return 1
    _eprint(f"aggregate_exp02_umma_throughput: manifest-write: OK: status={args.status}")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: plan
# ---------------------------------------------------------------------------
def cmd_plan(args: argparse.Namespace) -> int:
    plan = build_plan()
    errors = check_plan_contract(plan)
    if errors:
        _eprint("aggregate_exp02_umma_throughput: plan: ERROR: plan contract violated:")
        for error in errors:
            _eprint(f"aggregate_exp02_umma_throughput: plan:   - {error}")
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
# afterward. Never calls CUDA, Docker, nvidia-smi, either UMMA binary, or the
# network; capture/subprocess/artifact-path behavior is exercised via
# unittest.mock, never a real subprocess. Symlink-escape checks against
# create_campaign_dir/resolve_campaign_dir use an injected raw_root so they
# never touch the real repository's results/raw/ tree.
# ---------------------------------------------------------------------------
DEFAULT_COMMON = {"run_kind": "smoke", "iterations": 20, "warmup_iterations": 5, "git_commit": "a" * 40}


def _default_row(
    entry: dict, sample_index: int, *, repetitions: int, run_kind: str = "smoke",
    iterations: int = 20, warmup_iterations: int = 5, git_commit: str = "a" * 40, git_dirty: str = "false",
    elapsed_cycles: int | str | None = None, gpu_name: str = "NVIDIA B300 SXM6 AC",
    gpu_uuid: str = "GPU-00000000-0000-0000-0000-000000000000",
    cuda_driver_version: str = "13010", cuda_runtime_version: str = "13010",
    overrides: dict | None = None,
) -> dict[str, str]:
    info = METHOD_INFO[entry["method"]]
    n, depth = entry["n"], entry["depth"]
    total_umma = depth * iterations
    flops_per_umma = 2 * info["m"] * n * FROZEN_K
    total_flops = flops_per_umma * total_umma
    if elapsed_cycles is None:
        elapsed_cycles_value: int | str = 1000 + 10 * sample_index + depth
    else:
        elapsed_cycles_value = elapsed_cycles
    if isinstance(elapsed_cycles_value, str):
        elapsed_cycles_str = elapsed_cycles_value
        try:
            elapsed_cycles_for_formula = int(elapsed_cycles_value)
        except ValueError:
            elapsed_cycles_for_formula = 1
    else:
        elapsed_cycles_str = str(elapsed_cycles_value)
        elapsed_cycles_for_formula = elapsed_cycles_value
    cycles_per_umma = elapsed_cycles_for_formula / total_umma if total_umma else 0.0
    flops_per_cycle = total_flops / elapsed_cycles_for_formula if elapsed_cycles_for_formula else 0.0

    row = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": "2026-08-03T00:00:00Z",
        "run_kind": run_kind,
        "publishable": "false",
        "method": entry["method"],
        "sample_index": str(sample_index),
        "cta_group": str(info["cta_group"]),
        "m": str(info["m"]),
        "n": str(n),
        "k": str(FROZEN_K),
        "depth": str(depth),
        "iterations": str(iterations),
        "warmup_iterations": str(warmup_iterations),
        "repetitions": str(repetitions),
        "umma_per_iteration": str(depth),
        "total_umma": str(total_umma),
        "flops_per_umma": str(flops_per_umma),
        "total_flops": str(total_flops),
        "elapsed_cycles": elapsed_cycles_str,
        "cycles_per_umma": f"{cycles_per_umma:.6f}",
        "flops_per_cycle": f"{flops_per_cycle:.6f}",
        "threads_per_cta": str(FROZEN_THREADS_PER_CTA),
        "grid_blocks": str(info["grid_blocks"]),
        "tmem_columns": str(n),
        "operand_path": FROZEN_OPERAND_PATH,
        "input_type": FROZEN_INPUT_TYPE,
        "accumulator_type": FROZEN_ACCUMULATOR_TYPE,
        "correctness": "OK",
        "mismatches": "0",
        "max_abs_error": "0",
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "compute_capability": FROZEN_COMPUTE_CAPABILITY,
        "cuda_driver_version": cuda_driver_version,
        "cuda_runtime_version": cuda_runtime_version,
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


def _expect_for(entry: dict, common: dict = DEFAULT_COMMON, *, repetitions: int) -> dict:
    return _expect_for_entry(
        entry, run_kind=common["run_kind"], iterations=common["iterations"],
        warmup_iterations=common["warmup_iterations"], repetitions=repetitions, git_commit=common["git_commit"],
    )


def _build_valid_campaign(
    campaign_dir: Path, *, repetitions: int = 3, run_kind: str = "smoke", iterations: int = 20,
    warmup_iterations: int = 5, git_commit: str = "a" * 40, row_overrides_by_index: dict[int, dict[int, dict]] | None = None,
) -> list[dict]:
    plan = build_plan()
    (campaign_dir / "cases").mkdir(parents=True, exist_ok=True)
    row_overrides_by_index = row_overrides_by_index or {}
    for entry in plan:
        per_sample_overrides = row_overrides_by_index.get(entry["index"], {})
        rows = [
            _default_row(
                entry, sample_index, repetitions=repetitions, run_kind=run_kind, iterations=iterations,
                warmup_iterations=warmup_iterations, git_commit=git_commit,
                overrides=per_sample_overrides.get(sample_index),
            )
            for sample_index in range(repetitions)
        ]
        _write_case_csv(campaign_dir / "cases" / f"{entry['case_name']}.csv", rows)
    return plan


def _prepare_test_finalize_campaign(parent: Path, campaign_id: str, *, repetitions: int = 2) -> tuple[Path, argparse.Namespace]:
    """Create a complete synthetic pre-finalization fixture under parent.

    It includes all 24 case files, execution_order.csv, an initialized
    IN_PROGRESS manifest, PASS self-tests, and fully updated progress
    counters. It deliberately does not create build artifacts or a versions
    contract; callers inject those controlled paths into _do_finalize.
    """
    campaign = parent / campaign_id
    campaign.mkdir()
    plan = _build_valid_campaign(campaign, repetitions=repetitions)
    write_execution_order(campaign, plan)
    started = "20260803T000000Z"
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
                "run_kind": "smoke", "iterations": 20, "warmup_iterations": 5,
                "repetitions": repetitions, "campaign_id": campaign_id,
            },
            "selected_gpu_index": 0,
            "git_commit": "a" * 40,
            "git_dirty": False,
            "self_test_outcomes": {"umma_1sm": "PASS", "umma_2sm": "PASS"},
        },
        status="IN_PROGRESS",
    )
    args = argparse.Namespace(
        campaign_id=campaign_id, run_kind="smoke", repetitions=repetitions, iterations=20,
        warmup_iterations=5, git_commit="a" * 40, gpu_index=0, started_at_utc=started,
        completed_at_utc="20260803T000200Z", self_test_umma_1sm="PASS", self_test_umma_2sm="PASS",
    )
    return campaign, args


class _SelfTestRecorder:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.total = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.total += 1
        if condition:
            print(f"aggregate_exp02_umma_throughput: self-test: PASS: {name}", file=sys.stderr)
        else:
            self.failures.append(name)
            suffix = f"; {detail}" if detail else ""
            print(f"aggregate_exp02_umma_throughput: self-test: FAIL: {name}{suffix}", file=sys.stderr)


def _write_single_row_case(tmp_path: Path, label: str, entry: dict, *, overrides: dict) -> Path:
    """Test helper: writes one single-repetition case CSV with the given
    field overrides applied to an otherwise-valid default row."""
    path = tmp_path / f"single_{label}_{entry['case_name']}.csv"
    row = _default_row(entry, 0, repetitions=1, overrides=overrides)
    _write_case_csv(path, [row])
    return path


def run_self_test() -> int:
    rec = _SelfTestRecorder()
    plan = build_plan()
    seq = itertools.count()

    with tempfile.TemporaryDirectory(prefix="exp02_selftest_") as tmp:
        tmp_path = Path(tmp)

        def check_field_rejected(name: str, entry: dict, overrides: dict, needle: str, *, repetitions: int = 1) -> None:
            path = _write_single_row_case(tmp_path, f"f{next(seq)}", entry, overrides=overrides)
            _, errors = validate_case_file(path, _expect_for(entry, repetitions=repetitions))
            rec.check(name, any(needle in e for e in errors), detail=str(errors))

        # --- 1-3: plan contract (count, uniqueness/ordering, alternation) --
        plan_errors = check_plan_contract(plan)
        rec.check("plan has exactly 24 configurations, unique, ordered, alternating", not plan_errors, detail=str(plan_errors))
        rec.check("plan indices are exactly 0..23 in order", [e["index"] for e in plan] == list(range(24)))
        rec.check(
            "plan pair_index is 0..11, each exactly twice",
            sorted(collections.Counter(e["pair_index"] for e in plan).values()) == [2] * 12,
        )
        alternation_ok = True
        for pair_index, (n, depth) in enumerate(PAIR_ORDER):
            first, second = plan[pair_index * 2], plan[pair_index * 2 + 1]
            expected_order = METHODS if pair_index % 2 == 0 else (METHODS[1], METHODS[0])
            if (
                (first["method"], second["method"]) != expected_order
                or (first["n"], first["depth"]) != (n, depth)
                or (second["n"], second["depth"]) != (n, depth)
            ):
                alternation_ok = False
        rec.check("every pair alternates 1-SM/2-SM order per the frozen (N,depth) sequence", alternation_ok)
        rec.check("int64 safety holds for the frozen matrix at the runner's max CLI bounds", not check_int64_safety())

        # --- 4-5: valid synthetic CSVs for both methods ---------------------
        entry_1sm = next(e for e in plan if e["method"] == "umma_1sm" and e["n"] == 128 and e["depth"] == 16)
        entry_2sm = next(e for e in plan if e["method"] == "umma_2sm" and e["n"] == 128 and e["depth"] == 16)
        for label, entry in (("umma_1sm", entry_1sm), ("umma_2sm", entry_2sm)):
            rows = [_default_row(entry, s, repetitions=3) for s in range(3)]
            path = tmp_path / f"valid_{label}.csv"
            _write_case_csv(path, rows)
            _, errors = validate_case_file(path, _expect_for(entry, repetitions=3))
            rec.check(f"valid synthetic {label} CSV passes validation", not errors, detail=str(errors))

        # --- malformed/unterminated CSV quoting (audit repair: strict mode) ------
        # A quoted field immediately followed by unescaped trailing text before
        # the next delimiter (e.g. `"a" b,c`) is the canonical strict-mode
        # csv.Error trigger: Python's default (non-strict) csv.reader silently
        # reinterprets it instead of rejecting it, which is exactly the audit
        # finding this regression reproduces and closes.
        malformed_path = tmp_path / "malformed_quoting.csv"
        malformed_path.write_text(
            ",".join(CSV_HEADER) + "\n" + '"' + entry_1sm["case_name"] + '" trailing,x,x\n', encoding="utf-8",
        )
        rows_bad, read_errors_bad = read_case_rows(malformed_path)
        rec.check(
            "a case CSV with malformed/unterminated quoting is rejected by strict csv.reader, never silently misparsed",
            bool(read_errors_bad) and not rows_bad,
            detail=str(read_errors_bad),
        )
        _, ve_errors_bad = validate_case_file(malformed_path, _expect_for(entry_1sm, repetitions=1))
        rec.check(
            "validate_case_file rejects the same malformed-quoting case CSV",
            bool(ve_errors_bad),
            detail=str(ve_errors_bad),
        )

        malformed_eo_path = tmp_path / "malformed_eo_execution_order.csv"
        malformed_eo_path.write_text(
            ",".join(EXECUTION_ORDER_HEADER) + "\n"
            + '"0" bogus,1,umma_1sm,1,128,64,16,4,build/compute/umma_1sm,cases/x.csv\n',
            encoding="utf-8",
        )
        eo_malformed_errors = validate_execution_order_file(malformed_eo_path, plan)
        rec.check(
            "execution_order.csv with malformed/unterminated quoting is rejected by strict csv.reader",
            bool(eo_malformed_errors),
            detail=str(eo_malformed_errors),
        )

        # --- structural failures: header/order/row-count/sample-index ------
        bad_header_path = tmp_path / "bad_header.csv"
        _write_case_csv(bad_header_path, [_default_row(entry_1sm, 0, repetitions=1)], header=list(reversed(CSV_HEADER)))
        _, errors = validate_case_file(bad_header_path, _expect_for(entry_1sm, repetitions=1))
        rec.check("a reordered header is rejected", any("header mismatch" in e for e in errors), detail=str(errors))

        wrong_count_path = tmp_path / "wrong_count.csv"
        _write_case_csv(wrong_count_path, [_default_row(entry_1sm, s, repetitions=3) for s in range(2)])
        _, errors = validate_case_file(wrong_count_path, _expect_for(entry_1sm, repetitions=3))
        rec.check("a wrong row count is rejected", any("data row(s), expected exactly repetitions" in e for e in errors), detail=str(errors))

        dup_index_path = tmp_path / "dup_index.csv"
        _write_case_csv(dup_index_path, [_default_row(entry_1sm, 0, repetitions=3), _default_row(entry_1sm, 0, repetitions=3), _default_row(entry_1sm, 2, repetitions=3)])
        _, errors = validate_case_file(dup_index_path, _expect_for(entry_1sm, repetitions=3))
        rec.check(
            "a duplicate sample_index is rejected",
            any("appears 2 times" in e for e in errors) and any("sample_index=1 is missing" in e for e in errors),
            detail=str(errors),
        )

        # --- method-specific field mismatches (both directions) ------------
        check_field_rejected("cta_group wrong for umma_1sm (claims 2)", entry_1sm, {"cta_group": "2"}, "cta_group=2")
        check_field_rejected("cta_group wrong for umma_2sm (claims 1)", entry_2sm, {"cta_group": "1"}, "cta_group=1")
        check_field_rejected("m wrong for umma_1sm (claims 256)", entry_1sm, {"m": "256"}, "m=256")
        check_field_rejected("m wrong for umma_2sm (claims 128)", entry_2sm, {"m": "128"}, "m=128")
        check_field_rejected("grid_blocks wrong for umma_1sm (claims 2)", entry_1sm, {"grid_blocks": "2"}, "grid_blocks=2")
        check_field_rejected("grid_blocks wrong for umma_2sm (claims 1)", entry_2sm, {"grid_blocks": "1"}, "grid_blocks=1")
        check_field_rejected("method field disagrees with the case's own method", entry_1sm, {"method": "umma_2sm"}, "method=")

        # --- n/depth/k domain and exact-match failures ----------------------
        check_field_rejected("n mismatch vs the expected configuration", entry_1sm, {"n": "64", "tmem_columns": "64"}, "n=64")
        check_field_rejected("n outside {64,128,256}", entry_1sm, {"n": "100", "tmem_columns": "100"}, "not in")
        check_field_rejected("depth mismatch vs the expected configuration", entry_1sm, {"depth": "4", "umma_per_iteration": "4"}, "depth=4")
        check_field_rejected("depth outside {4,16,64,256}", entry_1sm, {"depth": "5", "umma_per_iteration": "5"}, "not in")
        check_field_rejected("k != 16", entry_1sm, {"k": "8"}, "k=8")

        # --- requested-parameter mismatches ----------------------------------
        check_field_rejected("iterations mismatch vs requested", entry_1sm, {"iterations": "999"}, "iterations=999")
        check_field_rejected("warmup_iterations mismatch vs requested", entry_1sm, {"warmup_iterations": "999"}, "warmup_iterations=999")
        check_field_rejected("repetitions mismatch vs requested", entry_1sm, {"repetitions": "999"}, "repetitions=999")

        # --- UMMA/FLOP formula failures --------------------------------------
        check_field_rejected("umma_per_iteration formula wrong", entry_1sm, {"umma_per_iteration": "999"}, "umma_per_iteration=999")
        check_field_rejected("total_umma formula wrong", entry_1sm, {"total_umma": "1"}, "total_umma=1")
        check_field_rejected("flops_per_umma formula wrong", entry_1sm, {"flops_per_umma": "1"}, "flops_per_umma=1")
        check_field_rejected("total_flops formula wrong", entry_1sm, {"total_flops": "1"}, "total_flops=1")

        # --- elapsed_cycles / six-decimal consistency failures ---------------
        check_field_rejected("elapsed_cycles zero is rejected", entry_1sm, {"elapsed_cycles": "0"}, "elapsed_cycles=0")
        check_field_rejected("elapsed_cycles non-integer is rejected", entry_1sm, {"elapsed_cycles": "12.5"}, "elapsed_cycles")
        check_field_rejected("cycles_per_umma malformed (not six-decimal) is rejected", entry_1sm, {"cycles_per_umma": "1.5"}, "cycles_per_umma")
        check_field_rejected("cycles_per_umma NaN is rejected", entry_1sm, {"cycles_per_umma": "nan"}, "cycles_per_umma")
        check_field_rejected("cycles_per_umma Inf is rejected", entry_1sm, {"cycles_per_umma": "inf"}, "cycles_per_umma")
        check_field_rejected("cycles_per_umma far outside tolerance is rejected", entry_1sm, {"cycles_per_umma": "999999.000000"}, "cycles_per_umma")
        check_field_rejected("flops_per_cycle negative is rejected", entry_1sm, {"flops_per_cycle": "-1.000000"}, "flops_per_cycle")
        check_field_rejected("flops_per_cycle far outside tolerance is rejected", entry_1sm, {"flops_per_cycle": "0.000001"}, "flops_per_cycle")

        # --- frozen constant-field failures -----------------------------------
        check_field_rejected("threads_per_cta != 128 is rejected", entry_1sm, {"threads_per_cta": "256"}, "threads_per_cta=256")
        check_field_rejected("tmem_columns != n is rejected", entry_1sm, {"tmem_columns": "999"}, "tmem_columns=999")
        check_field_rejected("operand_path wrong is rejected", entry_1sm, {"operand_path": "gmem_smem"}, "operand_path=")
        check_field_rejected("input_type wrong is rejected", entry_1sm, {"input_type": "fp16"}, "input_type=")
        check_field_rejected("accumulator_type wrong is rejected", entry_1sm, {"accumulator_type": "fp16"}, "accumulator_type=")
        check_field_rejected("correctness != OK is rejected", entry_1sm, {"correctness": "FAIL"}, "correctness=")
        check_field_rejected("mismatches != 0 is rejected", entry_1sm, {"mismatches": "1"}, "mismatches=1")
        check_field_rejected("max_abs_error != 0 is rejected", entry_1sm, {"max_abs_error": "1"}, "max_abs_error=1")
        check_field_rejected("compute_capability != 10.3 is rejected", entry_1sm, {"compute_capability": "9.0"}, "compute_capability=")

        # --- correctness/provenance failures -----------------------------------
        check_field_rejected("gpu_name empty is rejected", entry_1sm, {"gpu_name": ""}, "gpu_name")
        check_field_rejected("gpu_name with control characters is rejected", entry_1sm, {"gpu_name": "bad\x01name"}, "gpu_name")
        check_field_rejected("gpu_uuid malformed is rejected", entry_1sm, {"gpu_uuid": "not-a-uuid"}, "gpu_uuid")
        check_field_rejected("git_commit malformed (not 40 hex) is rejected", entry_1sm, {"git_commit": "xyz"}, "git_commit")
        check_field_rejected("git_commit valid-hex-but-wrong is rejected", entry_1sm, {"git_commit": "b" * 40}, "git_commit")
        check_field_rejected("git_dirty != false is rejected", entry_1sm, {"git_dirty": "true"}, "git_dirty=")
        check_field_rejected("publishable != literal 'false' ('True') is rejected", entry_1sm, {"publishable": "True"}, "publishable=")
        check_field_rejected("publishable != literal 'false' ('0') is rejected", entry_1sm, {"publishable": "0"}, "publishable=")
        check_field_rejected("schema_version wrong is rejected", entry_1sm, {"schema_version": "2"}, "schema_version=")
        check_field_rejected("timestamp_utc malformed is rejected", entry_1sm, {"timestamp_utc": "not-a-time"}, "timestamp_utc=")
        check_field_rejected("run_kind mismatch vs requested is rejected", entry_1sm, {"run_kind": "benchmark"}, "run_kind=")

        # --- cross-case GPU/runtime/provenance drift ----------------------------
        good_cases = [(entry_1sm, [_default_row(entry_1sm, 0, repetitions=1)]), (entry_2sm, [_default_row(entry_2sm, 0, repetitions=1)])]
        rec.check("matching cross-case common fields pass", not check_cross_case_consistency(good_cases))

        drifted_uuid = _default_row(entry_2sm, 0, repetitions=1, overrides={"gpu_uuid": "GPU-11111111-1111-1111-1111-111111111111"})
        cc_errors = check_cross_case_consistency([(entry_1sm, [_default_row(entry_1sm, 0, repetitions=1)]), (entry_2sm, [drifted_uuid])])
        rec.check("cross-case gpu_uuid drift is detected", any("gpu_uuid" in e for e in cc_errors), detail=str(cc_errors))

        drifted_driver = _default_row(entry_2sm, 0, repetitions=1, overrides={"cuda_driver_version": "99999"})
        cc_errors2 = check_cross_case_consistency([(entry_1sm, [_default_row(entry_1sm, 0, repetitions=1)]), (entry_2sm, [drifted_driver])])
        rec.check("cross-case cuda_driver_version drift is detected", any("cuda_driver_version" in e for e in cc_errors2), detail=str(cc_errors2))

        drifted_iterations = _default_row(entry_2sm, 0, repetitions=1, iterations=999)
        cc_errors3 = check_cross_case_consistency([(entry_1sm, [_default_row(entry_1sm, 0, repetitions=1)]), (entry_2sm, [drifted_iterations])])
        rec.check("cross-case iterations drift is detected", any("iterations" in e for e in cc_errors3), detail=str(cc_errors3))

        # --- scan_case_directory: missing/duplicate/unexpected/symlink ----------
        plan_full = build_plan()

        complete_dir = tmp_path / "scan_complete"
        complete_dir.mkdir()
        for e in plan_full:
            _write_case_csv(complete_dir / f"{e['case_name']}.csv", [_default_row(e, 0, repetitions=1)])
        found_c, errs_c = scan_case_directory(complete_dir, plan_full)
        rec.check("scan_case_directory accepts a complete, correct 24-case set", not errs_c and len(found_c) == 24, detail=str(errs_c))

        missing_dir = tmp_path / "scan_missing"
        missing_dir.mkdir()
        for e in plan_full[1:]:
            _write_case_csv(missing_dir / f"{e['case_name']}.csv", [_default_row(e, 0, repetitions=1)])
        _, errs_m = scan_case_directory(missing_dir, plan_full)
        rec.check("scan_case_directory rejects a missing case", any("missing configuration index 0" in e for e in errs_m), detail=str(errs_m))

        noncanonical_dir = tmp_path / "scan_noncanonical"
        noncanonical_dir.mkdir()
        for e in plan_full:
            _write_case_csv(noncanonical_dir / f"{e['case_name']}.csv", [_default_row(e, 0, repetitions=1)])
        canonical_case = noncanonical_dir / f"{plan_full[0]['case_name']}.csv"
        noncanonical_case = noncanonical_dir / "00_umma_1sm_n064_d4.csv"
        canonical_case.rename(noncanonical_case)
        _, errs_nc = scan_case_directory(noncanonical_dir, plan_full)
        rec.check(
            "scan_case_directory rejects a semantically equivalent but non-canonical n064 filename",
            any("00_umma_1sm_n064_d4.csv" in e and "unexpected case filename" in e for e in errs_nc)
            and any("missing configuration index 0" in e for e in errs_nc),
            detail=str(errs_nc),
        )

        # audit repair: an unexpected entry in cases/ must be rejected, never
        # silently ignored, regardless of its extension or type -- a stray
        # notes.txt, a .partial salvage-evidence file, or a subdirectory must
        # each independently prevent scan_case_directory from returning a
        # clean 24-case set (and therefore prevent finalize from reaching
        # COMPLETE).
        unexpected_dir = tmp_path / "scan_unexpected"
        unexpected_dir.mkdir()
        for e in plan_full:
            _write_case_csv(unexpected_dir / f"{e['case_name']}.csv", [_default_row(e, 0, repetitions=1)])
        (unexpected_dir / "garbage.csv").write_text("not a real case\n")
        (unexpected_dir / "notes.txt").write_text("ignored\n")
        _, errs_u = scan_case_directory(unexpected_dir, plan_full)
        rec.check(
            "scan_case_directory rejects an unexpected .csv file with a non-canonical name",
            any("garbage.csv" in e and "unexpected case filename" in e for e in errs_u),
            detail=str(errs_u),
        )
        rec.check(
            "scan_case_directory rejects an unexpected non-.csv file (cases/notes.txt), never silently ignoring it",
            any("notes.txt" in e for e in errs_u),
            detail=str(errs_u),
        )

        partial_dir = tmp_path / "scan_partial"
        partial_dir.mkdir()
        for e in plan_full:
            _write_case_csv(partial_dir / f"{e['case_name']}.csv", [_default_row(e, 0, repetitions=1)])
        (partial_dir / f"{plan_full[0]['case_name']}.csv.partial").write_text("salvaged garbage\n")
        _, errs_p = scan_case_directory(partial_dir, plan_full)
        rec.check(
            "scan_case_directory rejects an unexpected .partial salvage-evidence file, never silently ignoring it",
            any(".partial" in e for e in errs_p),
            detail=str(errs_p),
        )

        subdir_dir = tmp_path / "scan_subdir"
        subdir_dir.mkdir()
        for e in plan_full:
            _write_case_csv(subdir_dir / f"{e['case_name']}.csv", [_default_row(e, 0, repetitions=1)])
        (subdir_dir / "unexpected_subdir").mkdir()
        _, errs_sd = scan_case_directory(subdir_dir, plan_full)
        rec.check(
            "scan_case_directory rejects an unexpected subdirectory in cases/, never silently ignoring it",
            any("unexpected_subdir" in e for e in errs_sd),
            detail=str(errs_sd),
        )

        symlink_dir = tmp_path / "scan_symlink"
        symlink_dir.mkdir()
        for e in plan_full:
            _write_case_csv(symlink_dir / f"{e['case_name']}.csv", [_default_row(e, 0, repetitions=1)])
        real_target = tmp_path / "outside_case.csv"
        _write_case_csv(real_target, [_default_row(plan_full[0], 0, repetitions=1)])
        symlinked_case_path = symlink_dir / f"{plan_full[0]['case_name']}.csv"
        symlinked_case_path.unlink()
        symlinked_case_path.symlink_to(real_target)
        _, errs_sym = scan_case_directory(symlink_dir, plan_full)
        rec.check("scan_case_directory rejects a symlinked case file", any("is a symlink" in e for e in errs_sym), detail=str(errs_sym))

        # --- execution_order.csv round-trip and tamper detection ---------------
        eo_dir = tmp_path / "eo_test"
        eo_dir.mkdir()
        write_execution_order(eo_dir, plan_full)
        eo_errors = validate_execution_order_file(eo_dir / "execution_order.csv", plan_full)
        rec.check("execution_order.csv round-trips with zero errors", not eo_errors, detail=str(eo_errors))
        with open(eo_dir / "execution_order.csv", newline="") as f:
            eo_row_count = sum(1 for _ in csv.reader(f)) - 1
        rec.check("execution_order.csv has exactly 24 data rows", eo_row_count == 24)

        try:
            write_execution_order(eo_dir, plan_full)
            eo_noclobber_ok = False
        except UnsafePathError:
            eo_noclobber_ok = True
        rec.check("execution_order.csv cannot be written twice (no-clobber)", eo_noclobber_ok)

        tampered_dir = tmp_path / "eo_tampered"
        tampered_dir.mkdir()
        write_execution_order(tampered_dir, plan_full)
        with open(tampered_dir / "execution_order.csv", newline="") as f:
            tampered_rows = list(csv.reader(f))
        tampered_rows[1][2] = "umma_2sm"  # corrupt row 0's method column
        with open(tampered_dir / "execution_order.csv", "w", newline="") as f:
            csv.writer(f).writerows(tampered_rows)
        tampered_errors = validate_execution_order_file(tampered_dir / "execution_order.csv", plan_full)
        rec.check("a tampered execution_order.csv row is rejected", bool(tampered_errors), detail=str(tampered_errors))

        # --- combined_samples.csv / summary.csv determinism and no-clobber ------
        agg_campaign = tmp_path / "agg_campaign"
        agg_campaign.mkdir()
        agg_plan = _build_valid_campaign(agg_campaign, repetitions=2)
        cases_for_agg = []
        agg_fixture_ok = True
        for e in agg_plan:
            rows, errs = validate_case_file(agg_campaign / "cases" / f"{e['case_name']}.csv", _expect_for(e, repetitions=2))
            if errs:
                agg_fixture_ok = False
            cases_for_agg.append((e, rows))
        rec.check("the 24-case aggregation fixture is independently valid", agg_fixture_ok)

        combined_path = agg_campaign / "combined_samples.csv"
        summary_path = agg_campaign / "summary.csv"
        combined_rows = write_combined_samples(agg_plan, cases_for_agg, combined_path)
        rec.check("combined_samples.csv has 24*repetitions rows", combined_rows == 48)
        with open(combined_path, newline="") as f:
            combined_header = next(csv.reader(f))
        rec.check("combined_samples.csv is prefixed with traceability fields", combined_header[:3] == ["invocation_index", "pair_index", "case_name"])
        rec.check("combined_samples.csv preserves the original 37-column schema unchanged", combined_header[3:] == CSV_HEADER)

        summary_rows = write_summary(cases_for_agg, summary_path)
        rec.check("summary.csv has exactly 24 rows", summary_rows == 24)

        try:
            write_combined_samples(agg_plan, cases_for_agg, combined_path)
            combined_noclobber_ok = False
        except UnsafePathError:
            combined_noclobber_ok = True
        rec.check("combined_samples.csv cannot be written twice (no-clobber)", combined_noclobber_ok)

        try:
            write_summary(cases_for_agg, summary_path)
            summary_noclobber_ok = False
        except UnsafePathError:
            summary_noclobber_ok = True
        rec.check("summary.csv cannot be written twice (no-clobber)", summary_noclobber_ok)

        combined_path_2 = agg_campaign / "combined_samples_2.csv"
        write_combined_samples(agg_plan, cases_for_agg, combined_path_2)
        rec.check("combined_samples.csv generation is deterministic", combined_path.read_bytes() == combined_path_2.read_bytes())

        summary_path_2 = agg_campaign / "summary_2.csv"
        write_summary(cases_for_agg, summary_path_2)
        rec.check("summary.csv generation is deterministic", summary_path.read_bytes() == summary_path_2.read_bytes())

        # --- summary statistics must be mathematically valid (audit repair) ------
        try:
            _numeric_stats([1000.0])
            single_sample_stats_rejected = False
        except ValueError:
            single_sample_stats_rejected = True
        rec.check(
            "_numeric_stats/_sample_stdev refuse to silently report zero stdev/CV for a single observation",
            single_sample_stats_rejected,
        )
        two_sample_stats = _numeric_stats([1000.0, 1002.0])
        rec.check(
            "_numeric_stats computes a real, non-zero sample stdev for two or more observations",
            two_sample_stats["stdev"] > 0.0,
            detail=str(two_sample_stats),
        )

        repetitions_floor_campaign = tmp_path / "repetitions_floor_campaign"
        repetitions_floor_campaign.mkdir()
        try:
            merge_manifest(
                repetitions_floor_campaign,
                {
                    "campaign_id": "repetitions_floor_campaign", "run_kind": "smoke",
                    "started_at_utc": "20260803T000000Z",
                    "configuration_count_expected": EXPECTED_CONFIGURATION_COUNT, "configuration_count_completed": 0,
                    "sample_count_expected": EXPECTED_CONFIGURATION_COUNT, "sample_count_completed": 0,
                    "requested": {
                        "run_kind": "smoke", "iterations": 20, "warmup_iterations": 5,
                        "repetitions": 1, "campaign_id": "repetitions_floor_campaign",
                    },
                    "selected_gpu_index": 0, "git_commit": "a" * 40, "git_dirty": False,
                },
                status="IN_PROGRESS",
            )
            repetitions_floor_rejected = False
        except ManifestTransitionError:
            repetitions_floor_rejected = True
        rec.check(
            "manifest requested.repetitions=1 is rejected before a campaign can be initialized (REPETITIONS_MIN=2)",
            repetitions_floor_rejected,
        )

        # --- private-path redaction (audit repair): a synthetic absolute
        # repository path must never survive into the manifest, failure
        # detail, or any P2.3-generated diagnostic/log text. ----------------
        synthetic_abs_case_path = (
            f"{REPO_ROOT}/results/raw/exp02_umma_throughput/FAKE_CAMPAIGN/cases/00_umma_1sm_n64_d4.csv"
        )
        synthetic_relative_suffix = "results/raw/exp02_umma_throughput/FAKE_CAMPAIGN/cases/00_umma_1sm_n64_d4.csv"

        redacted_single = _redact_repo_root(f"{synthetic_abs_case_path}: some validation error")
        rec.check(
            "_redact_repo_root removes the absolute repository root from a synthetic absolute repository path",
            str(REPO_ROOT) not in redacted_single and synthetic_relative_suffix in redacted_single,
            detail=redacted_single,
        )

        redacted_list = _redact_failure_detail(
            [f"{synthetic_abs_case_path}: another error", "an already-relative message"]
        )
        rec.check(
            "_redact_failure_detail (used by _do_finalize before persisting failure_detail) redacts every list entry",
            all(str(REPO_ROOT) not in e for e in redacted_list) and redacted_list[1] == "an already-relative message",
            detail=str(redacted_list),
        )

        eprint_buf = io.StringIO()
        with contextlib.redirect_stderr(eprint_buf):
            _eprint(f"aggregate_exp02_umma_throughput: test: synthetic {synthetic_abs_case_path}")
        eprint_logged = eprint_buf.getvalue()
        rec.check(
            "_eprint (used by every P2.3 subcommand whose stderr the runner redirects into logs/*.log) "
            "never leaks the absolute repository root",
            str(REPO_ROOT) not in eprint_logged and synthetic_relative_suffix in eprint_logged,
            detail=eprint_logged,
        )

        # End-to-end proof through the real _do_finalize code path (never
        # writing under the real results/raw/ tree -- only a read-only
        # existence probe of one deliberately nonexistent path beneath it):
        redaction_probe_root = tmp_path / "redaction_probe_root"
        redaction_probe_root.mkdir()
        redaction_probe_campaign, redaction_probe_args = _prepare_test_finalize_campaign(
            redaction_probe_root, "redaction_probe", repetitions=2,
        )
        redaction_probe_sass1 = tmp_path / "redaction_probe.sass"
        redaction_probe_sass1.write_bytes(b"x")
        redaction_probe_bin2 = tmp_path / "redaction_probe_bin2"
        redaction_probe_bin2.write_bytes(b"x")
        redaction_probe_sass2 = tmp_path / "redaction_probe2.sass"
        redaction_probe_sass2.write_bytes(b"x")
        redaction_probe_artifacts = {
            "umma_1sm_bin": REPO_ROOT / "results" / "raw" / "exp02_umma_throughput" / "DOES_NOT_EXIST_PROBE_FOR_SELFTEST",
            "umma_1sm_sass": redaction_probe_sass1,
            "umma_2sm_bin": redaction_probe_bin2,
            "umma_2sm_sass": redaction_probe_sass2,
        }
        success_redact, errors_redact = _do_finalize(
            redaction_probe_campaign, redaction_probe_args, artifact_paths=redaction_probe_artifacts,
        )
        rec.check(
            "finalize fails closed against a missing artifact referenced by a real absolute path under REPO_ROOT",
            not success_redact and any(str(REPO_ROOT) in e for e in errors_redact),
            detail=str(errors_redact),
        )
        redaction_probe_manifest = load_manifest(redaction_probe_campaign)
        persisted_detail = redaction_probe_manifest.get("failure_detail") or []
        rec.check(
            "that real finalize failure never persists the absolute repository root into the manifest's failure_detail",
            redaction_probe_manifest.get("status") == "FAILED" and all(str(REPO_ROOT) not in d for d in persisted_detail),
            detail=str(persisted_detail),
        )

        # --- manifest state machine ---------------------------------------------
        manifest_campaign = tmp_path / "manifest_campaign"
        manifest_campaign.mkdir()
        m1 = merge_manifest(
            manifest_campaign,
            {
                "campaign_id": "manifest_campaign", "run_kind": "smoke", "started_at_utc": "20260803T000000Z",
                "configuration_count_expected": EXPECTED_CONFIGURATION_COUNT, "configuration_count_completed": 0,
                "sample_count_expected": EXPECTED_CONFIGURATION_COUNT * 3, "sample_count_completed": 0,
                "requested": {"run_kind": "smoke", "iterations": 20, "warmup_iterations": 5, "repetitions": 3, "campaign_id": "manifest_campaign"},
                "selected_gpu_index": 0, "git_commit": "a" * 40, "git_dirty": False,
            },
            status="IN_PROGRESS",
        )
        rec.check("init manifest is IN_PROGRESS with publishable=false", m1["status"] == "IN_PROGRESS" and m1["publishable"] is False)

        try:
            merge_manifest(manifest_campaign, {}, status="COMPLETE")
            complete_without_flag_rejected = False
        except ManifestTransitionError:
            complete_without_flag_rejected = True
        rec.check("only the finalizer (allow_complete=True) may set COMPLETE", complete_without_flag_rejected)

        try:
            merge_manifest(manifest_campaign, {"campaign_id": "different"}, status="IN_PROGRESS")
            immutable_rejected = False
        except ManifestTransitionError:
            immutable_rejected = True
        rec.check("campaign_id is immutable after initialization", immutable_rejected)

        merge_manifest(manifest_campaign, {"configuration_count_completed": 5, "sample_count_completed": 15}, status="IN_PROGRESS")
        try:
            merge_manifest(manifest_campaign, {"configuration_count_completed": 2, "sample_count_completed": 6}, status="IN_PROGRESS")
            decrease_rejected = False
        except ManifestTransitionError:
            decrease_rejected = True
        rec.check("configuration_count_completed cannot decrease from a prior recorded value", decrease_rejected)

        try:
            merge_manifest(manifest_campaign, {"unknown_field": 1}, status="IN_PROGRESS")
            unknown_key_rejected = False
        except ManifestTransitionError:
            unknown_key_rejected = True
        rec.check("an unallowlisted manifest key is rejected", unknown_key_rejected)

        try:
            merge_manifest(manifest_campaign, {"git_dirty": "false"}, status="IN_PROGRESS")
            wrong_type_rejected = False
        except ManifestTransitionError:
            wrong_type_rejected = True
        rec.check("a manifest field with the wrong Python type is rejected", wrong_type_rejected)

        merge_manifest(manifest_campaign, {"self_test_outcomes": {"umma_1sm": "PASS", "umma_2sm": "PASS"}}, status="IN_PROGRESS")
        try:
            merge_manifest(manifest_campaign, {"self_test_outcomes": {"umma_1sm": "FAIL", "umma_2sm": "PASS"}}, status="IN_PROGRESS")
            self_test_immutable_ok = False
        except ManifestTransitionError:
            self_test_immutable_ok = True
        rec.check("terminal self_test_outcomes cannot change once recorded", self_test_immutable_ok)

        # --- self-test outcome truthfulness (audit repair): first-method
        # failure, second-method failure, both-pass, an interrupted running
        # self-test, and illegal per-method transitions. Mirrors exactly the
        # write sequence scripts/run_exp02_umma_throughput.sh now performs.
        def _init_self_test_campaign(campaign_id: str) -> Path:
            campaign_dir = tmp_path / campaign_id
            campaign_dir.mkdir()
            merge_manifest(
                campaign_dir,
                {
                    "campaign_id": campaign_id, "run_kind": "smoke", "started_at_utc": "20260803T000000Z",
                    "configuration_count_expected": EXPECTED_CONFIGURATION_COUNT, "configuration_count_completed": 0,
                    "sample_count_expected": EXPECTED_CONFIGURATION_COUNT * 2, "sample_count_completed": 0,
                    "requested": {
                        "run_kind": "smoke", "iterations": 20, "warmup_iterations": 5,
                        "repetitions": 2, "campaign_id": campaign_id,
                    },
                    "selected_gpu_index": 0, "git_commit": "a" * 40, "git_dirty": False,
                },
                status="IN_PROGRESS",
            )
            return campaign_dir

        first_fail_campaign = _init_self_test_campaign("self_test_first_fail")
        merge_manifest(first_fail_campaign, {"self_test_outcomes": {"umma_1sm": "STARTED", "umma_2sm": "PENDING"}}, status="IN_PROGRESS")
        merge_manifest(first_fail_campaign, {"self_test_outcomes": {"umma_1sm": "FAIL", "umma_2sm": "NOT_RUN"}}, status="FAILED")
        first_fail_manifest = load_manifest(first_fail_campaign)
        rec.check(
            "first-method self-test failure is recorded truthfully (FAIL), and the never-run sibling is NOT_RUN, never invented",
            first_fail_manifest["self_test_outcomes"] == {"umma_1sm": "FAIL", "umma_2sm": "NOT_RUN"}
            and first_fail_manifest["status"] == "FAILED",
            detail=str(first_fail_manifest.get("self_test_outcomes")),
        )
        try:
            _validate_self_test_outcomes(first_fail_manifest["self_test_outcomes"], require_pass=True)
            first_fail_complete_rejected = False
        except ManifestTransitionError:
            first_fail_complete_rejected = True
        rec.check("a first-method self-test failure can never satisfy the COMPLETE self-test contract", first_fail_complete_rejected)

        second_fail_campaign = _init_self_test_campaign("self_test_second_fail")
        merge_manifest(second_fail_campaign, {"self_test_outcomes": {"umma_1sm": "STARTED", "umma_2sm": "PENDING"}}, status="IN_PROGRESS")
        merge_manifest(second_fail_campaign, {"self_test_outcomes": {"umma_1sm": "PASS", "umma_2sm": "PENDING"}}, status="IN_PROGRESS")
        merge_manifest(second_fail_campaign, {"self_test_outcomes": {"umma_1sm": "PASS", "umma_2sm": "STARTED"}}, status="IN_PROGRESS")
        merge_manifest(second_fail_campaign, {"self_test_outcomes": {"umma_1sm": "PASS", "umma_2sm": "FAIL"}}, status="FAILED")
        second_fail_manifest = load_manifest(second_fail_campaign)
        rec.check(
            "second-method self-test failure is recorded truthfully after the first genuinely passed",
            second_fail_manifest["self_test_outcomes"] == {"umma_1sm": "PASS", "umma_2sm": "FAIL"}
            and second_fail_manifest["status"] == "FAILED",
            detail=str(second_fail_manifest.get("self_test_outcomes")),
        )

        both_pass_campaign = _init_self_test_campaign("self_test_both_pass")
        merge_manifest(both_pass_campaign, {"self_test_outcomes": {"umma_1sm": "STARTED", "umma_2sm": "PENDING"}}, status="IN_PROGRESS")
        merge_manifest(both_pass_campaign, {"self_test_outcomes": {"umma_1sm": "PASS", "umma_2sm": "PENDING"}}, status="IN_PROGRESS")
        merge_manifest(both_pass_campaign, {"self_test_outcomes": {"umma_1sm": "PASS", "umma_2sm": "STARTED"}}, status="IN_PROGRESS")
        merge_manifest(both_pass_campaign, {"self_test_outcomes": {"umma_1sm": "PASS", "umma_2sm": "PASS"}}, status="IN_PROGRESS")
        both_pass_manifest = load_manifest(both_pass_campaign)
        rec.check(
            "both self-tests genuinely passing is recorded truthfully",
            both_pass_manifest["self_test_outcomes"] == {"umma_1sm": "PASS", "umma_2sm": "PASS"},
            detail=str(both_pass_manifest.get("self_test_outcomes")),
        )
        try:
            _validate_self_test_outcomes(both_pass_manifest["self_test_outcomes"], require_pass=True)
            both_pass_complete_ok = True
        except ManifestTransitionError:
            both_pass_complete_ok = False
        rec.check("a campaign with both self-tests genuinely PASS satisfies the COMPLETE self-test contract", both_pass_complete_ok)

        interrupted_campaign = _init_self_test_campaign("self_test_interrupted")
        merge_manifest(interrupted_campaign, {"self_test_outcomes": {"umma_1sm": "STARTED", "umma_2sm": "PENDING"}}, status="IN_PROGRESS")
        merge_manifest(interrupted_campaign, {}, status="INTERRUPTED")
        interrupted_manifest = load_manifest(interrupted_campaign)
        rec.check(
            "an interrupted running self-test remains STARTED, never silently reported as PASS",
            interrupted_manifest["self_test_outcomes"]["umma_1sm"] == "STARTED"
            and interrupted_manifest["status"] == "INTERRUPTED",
            detail=str(interrupted_manifest.get("self_test_outcomes")),
        )

        illegal_transition_campaign = _init_self_test_campaign("self_test_illegal_transition")
        merge_manifest(illegal_transition_campaign, {"self_test_outcomes": {"umma_1sm": "FAIL", "umma_2sm": "NOT_RUN"}}, status="IN_PROGRESS")
        try:
            merge_manifest(illegal_transition_campaign, {"self_test_outcomes": {"umma_1sm": "FAIL", "umma_2sm": "STARTED"}}, status="IN_PROGRESS")
            not_run_resurrection_rejected = False
        except ManifestTransitionError:
            not_run_resurrection_rejected = True
        rec.check("a method recorded NOT_RUN can never later be reported as STARTED/PASS/FAIL", not_run_resurrection_rejected)

        terminal_campaign = tmp_path / "terminal_campaign"
        terminal_campaign.mkdir()
        merge_manifest(
            terminal_campaign,
            {
                "campaign_id": "terminal_campaign", "run_kind": "smoke", "started_at_utc": "20260803T000000Z",
                "configuration_count_expected": EXPECTED_CONFIGURATION_COUNT, "configuration_count_completed": 0,
                "sample_count_expected": EXPECTED_CONFIGURATION_COUNT * 2, "sample_count_completed": 0,
                "requested": {"run_kind": "smoke", "iterations": 20, "warmup_iterations": 5, "repetitions": 2, "campaign_id": "terminal_campaign"},
                "selected_gpu_index": 0, "git_commit": "a" * 40, "git_dirty": False,
            },
            status="IN_PROGRESS",
        )
        merge_manifest(terminal_campaign, {"failure_stage": "test", "failure_detail": ["x"]}, status="FAILED")
        try:
            merge_manifest(terminal_campaign, {}, status="IN_PROGRESS")
            terminal_reopen_rejected = False
        except ManifestTransitionError:
            terminal_reopen_rejected = True
        rec.check("a terminal (FAILED) campaign can never be reopened", terminal_reopen_rejected)

        # --- capture (mocked subprocess; never a real binary) -------------------
        capture_campaign = tmp_path / "capture_campaign"
        (capture_campaign / "cases").mkdir(parents=True)
        (capture_campaign / "logs").mkdir(parents=True)
        fake_umma_1sm_bin = tmp_path / "fake_umma_1sm"
        fake_umma_1sm_bin.write_bytes(b"#!/bin/sh\n")
        fake_umma_1sm_bin.chmod(0o755)

        def _fake_run_success(argv, stdout=None, stderr=None):
            stdout.write(b"schema_version,x\n1,y\n")
            return subprocess.CompletedProcess(argv, 0)

        with mock.patch("subprocess.run", side_effect=_fake_run_success):
            rc_ok = _do_capture(
                capture_campaign, "cases/capture_test.csv", ["build/compute/umma_1sm"],
                artifact_paths={"build/compute/umma_1sm": fake_umma_1sm_bin},
            )
        rec.check(
            "a successful mocked capture publishes the case CSV",
            rc_ok == 0 and (capture_campaign / "cases" / "capture_test.csv").exists(),
        )

        def _fake_run_fail(argv, stdout=None, stderr=None):
            if stdout is not None:
                stdout.write(b"partial garbage\n")
            return subprocess.CompletedProcess(argv, 1)

        with mock.patch("subprocess.run", side_effect=_fake_run_fail):
            rc_fail = _do_capture(
                capture_campaign, "cases/capture_fail.csv", ["build/compute/umma_1sm"],
                artifact_paths={"build/compute/umma_1sm": fake_umma_1sm_bin},
            )
        rec.check(
            "a failed mocked capture is salvaged as .invalid evidence, never published",
            rc_fail == 1
            and not (capture_campaign / "cases" / "capture_fail.csv").exists()
            and (capture_campaign / "cases" / "capture_fail.csv.invalid").exists(),
        )

        rc_disallowed = _do_capture(capture_campaign, "cases/disallowed.csv", ["build/compute/not_a_real_binary"])
        rec.check("capture rejects a non-allowlisted binary", rc_disallowed == 2)

        with mock.patch("subprocess.run", side_effect=_fake_run_success):
            rc_second = _do_capture(
                capture_campaign, "cases/capture_test.csv", ["build/compute/umma_1sm"],
                artifact_paths={"build/compute/umma_1sm": fake_umma_1sm_bin},
            )
        rec.check("capture refuses to overwrite an already-published case CSV", rc_second == 2)

        existing_target = capture_campaign / "cases" / "already_there.csv"
        existing_target.write_text("x\n")
        try:
            resolve_capture_out_path(capture_campaign, "cases/already_there.csv")
            capture_existing_ok = False
        except UnsafePathError:
            capture_existing_ok = True
        rec.check("resolve_capture_out_path refuses an already-existing target (no-clobber)", capture_existing_ok)

        broken_link = capture_campaign / "cases" / "broken.csv"
        broken_link.symlink_to(capture_campaign / "cases" / "does_not_exist_target.csv")
        try:
            resolve_capture_out_path(capture_campaign, "cases/broken.csv")
            capture_brokenlink_ok = False
        except UnsafePathError:
            capture_brokenlink_ok = True
        rec.check("resolve_capture_out_path refuses a broken symlink target", capture_brokenlink_ok)

        # --- finalize: full success and controlled failure paths ---------------
        fake_1sm_bin = tmp_path / "fake_bin_1sm"; fake_1sm_bin.write_bytes(b"x")
        fake_1sm_sass = tmp_path / "fake_sass_1sm.sass"; fake_1sm_sass.write_bytes(b"x")
        fake_2sm_bin = tmp_path / "fake_bin_2sm"; fake_2sm_bin.write_bytes(b"x")
        fake_2sm_sass = tmp_path / "fake_sass_2sm.sass"; fake_2sm_sass.write_bytes(b"x")
        fake_versions = tmp_path / "fake_VERSIONS.env"
        fake_versions.write_text(
            "CUDA_VERSION=13.1.0\nCUDA_IMAGE=nvidia/cuda:13.1.0-devel-ubuntu24.04\n"
            "CUDA_IMAGE_DIGEST=sha256:" + "0" * 64 + "\nCUDA_IMAGE_PLATFORM=linux/amd64\n"
            "CUTLASS_VERSION=v4.6.1\nCUTLASS_COMMIT=" + "0" * 40 + "\nCUDA_ARCH=sm_103a\nMAX_BUILD_JOBS=2\n"
        )
        real_artifacts = {
            "umma_1sm_bin": fake_1sm_bin, "umma_1sm_sass": fake_1sm_sass,
            "umma_2sm_bin": fake_2sm_bin, "umma_2sm_sass": fake_2sm_sass,
        }

        fin_root = tmp_path / "finalize_root"
        fin_root.mkdir()
        fin_campaign, fin_args = _prepare_test_finalize_campaign(fin_root, "finalize_ok", repetitions=2)
        success, fin_errors = _do_finalize(fin_campaign, fin_args, artifact_paths=real_artifacts, versions_path=fake_versions)
        rec.check("finalize succeeds against a complete, valid campaign", success, detail=str(fin_errors))
        final_manifest = load_manifest(fin_campaign)
        rec.check(
            "finalize publishes a COMPLETE manifest with all mandatory hashes",
            final_manifest.get("status") == "COMPLETE"
            and len(final_manifest.get("case_file_sha256", {})) == 24
            and len(final_manifest.get("binary_and_sass_sha256", {})) == 4
            and set(final_manifest.get("aggregate_file_sha256", {})) == {"combined_samples.csv", "summary.csv"},
        )
        rec.check(
            "finalize produces combined_samples.csv and summary.csv",
            (fin_campaign / "combined_samples.csv").exists() and (fin_campaign / "summary.csv").exists(),
        )

        success2, errors2 = _do_finalize(fin_campaign, fin_args, artifact_paths=real_artifacts, versions_path=fake_versions)
        rec.check("finalize refuses to re-finalize an already-terminal campaign", not success2)

        fin_bad_root = tmp_path / "finalize_bad_root"
        fin_bad_root.mkdir()
        fin_bad_campaign, fin_bad_args = _prepare_test_finalize_campaign(fin_bad_root, "finalize_bad", repetitions=2)
        success_bad, errors_bad = _do_finalize(fin_bad_campaign, fin_bad_args, artifact_paths={}, versions_path=fake_versions)
        rec.check("finalize fails closed when required artifacts are misconfigured/missing", not success_bad and bool(errors_bad))
        bad_manifest = load_manifest(fin_bad_campaign)
        rec.check("a failed finalize records status=FAILED, never COMPLETE", bad_manifest.get("status") == "FAILED")

        noncanonical_root = tmp_path / "finalize_noncanonical_root"
        noncanonical_root.mkdir()
        noncanonical_campaign, noncanonical_args = _prepare_test_finalize_campaign(
            noncanonical_root, "finalize_noncanonical", repetitions=2,
        )
        canonical_case = noncanonical_campaign / "cases" / "00_umma_1sm_n64_d4.csv"
        noncanonical_case = noncanonical_campaign / "cases" / "00_umma_1sm_n064_d4.csv"
        canonical_case.rename(noncanonical_case)
        success_nc, errors_nc = _do_finalize(
            noncanonical_campaign, noncanonical_args,
            artifact_paths=real_artifacts, versions_path=fake_versions,
        )
        noncanonical_manifest = load_manifest(noncanonical_campaign)
        rec.check(
            "finalize rejects n064 in place of the exact execution_order.csv filename and never reaches COMPLETE",
            not success_nc
            and any("00_umma_1sm_n064_d4.csv" in error for error in errors_nc)
            and noncanonical_manifest.get("status") == "FAILED",
            detail=f"errors={errors_nc}; status={noncanonical_manifest.get('status')}",
        )

        mismatch_root = tmp_path / "finalize_mismatch_root"
        mismatch_root.mkdir()
        mismatch_campaign, mismatch_args = _prepare_test_finalize_campaign(mismatch_root, "finalize_mismatch", repetitions=2)
        mismatch_args.repetitions = 999
        success_mm, errors_mm = _do_finalize(mismatch_campaign, mismatch_args, artifact_paths=real_artifacts, versions_path=fake_versions)
        rec.check(
            "finalize rejects CLI args that disagree with the recorded manifest",
            not success_mm and any("repetitions" in e for e in errors_mm),
        )

        # --- raw_root-injected symlink-escape tests (never touch the real repo) -
        fake_repo_root = tmp_path / "fake_repo"
        fake_repo_root.mkdir()
        outside_target = tmp_path / "outside_raw"
        outside_target.mkdir()
        (fake_repo_root / "results").mkdir()
        (fake_repo_root / "results" / "raw").symlink_to(outside_target)
        try:
            create_campaign_dir("SHOULDNEVEREXIST", raw_root=fake_repo_root)
            raw_root_symlink_ok = False
        except UnsafePathError:
            raw_root_symlink_ok = True
        rec.check("a symlinked raw root is rejected (via an injected root, never the real repo)", raw_root_symlink_ok)

        fake_repo_root2 = tmp_path / "fake_repo2"
        (fake_repo_root2 / "results" / "raw" / "exp02_umma_throughput").mkdir(parents=True)
        outside_target2 = tmp_path / "outside_campaign"
        outside_target2.mkdir()
        (fake_repo_root2 / "results" / "raw" / "exp02_umma_throughput" / "EVIL").symlink_to(outside_target2)
        try:
            resolve_campaign_dir("results/raw/exp02_umma_throughput/EVIL", raw_root=fake_repo_root2)
            campaign_symlink_ok = False
        except UnsafePathError:
            campaign_symlink_ok = True
        rec.check("a symlinked campaign directory is rejected (via an injected root)", campaign_symlink_ok)

        fake_repo_root3 = tmp_path / "fake_repo3"
        try:
            create_campaign_dir("../escape", raw_root=fake_repo_root3)
            dotdot_ok = False
        except UnsafePathError:
            dotdot_ok = True
        rec.check("a campaign ID containing '..' is rejected", dotdot_ok)

        fake_repo_root4 = tmp_path / "fake_repo4"
        fake_repo_root4.mkdir()
        create_campaign_dir("dup_campaign", raw_root=fake_repo_root4)
        try:
            create_campaign_dir("dup_campaign", raw_root=fake_repo_root4)
            campaign_noclobber_ok = False
        except UnsafePathError:
            campaign_noclobber_ok = True
        rec.check("an existing campaign directory cannot be silently reused (no-clobber)", campaign_noclobber_ok)

        fake_repo_root5 = tmp_path / "fake_repo5"
        fake_repo_root5.mkdir()
        init_dir = _do_init_campaign(
            campaign_id="init_ok", run_kind="smoke", iterations=20, warmup_iterations=5, repetitions=3,
            git_commit="a" * 40, gpu_index=0, started_at_utc="20260803T000000Z", raw_root=fake_repo_root5,
        )
        init_manifest = load_manifest(init_dir)
        rec.check(
            "_do_init_campaign (fully isolated via raw_root) writes a valid IN_PROGRESS manifest and execution_order.csv",
            init_manifest.get("status") == "IN_PROGRESS"
            and init_manifest.get("configuration_count_expected") == EXPECTED_CONFIGURATION_COUNT
            and init_manifest.get("self_test_outcomes") == {"umma_1sm": "PENDING", "umma_2sm": "PENDING"}
            and (init_dir / "execution_order.csv").exists(),
        )

    if rec.failures:
        print(
            f"aggregate_exp02_umma_throughput: self-test: FAILED ({len(rec.failures)}/{rec.total} case(s)): {rec.failures}",
            file=sys.stderr,
        )
        print("aggregate_exp02_umma_throughput: SELF_TEST_RESULT=FAIL", file=sys.stderr)
        return 1
    print(f"aggregate_exp02_umma_throughput: self-test: OK ({rec.total} cases)", file=sys.stderr)
    print("aggregate_exp02_umma_throughput: SELF_TEST_RESULT=PASS", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aggregate_exp02_umma_throughput.py",
        description="P2.3 plan/validation/aggregation helper (see module docstring).",
    )
    parser.add_argument("--self-test", action="store_true", help="Run GPU-free synthetic tests and exit.")
    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="Print the frozen 24-invocation plan.")
    plan_parser.add_argument("--format", choices=("text", "lines", "json"), default="text")
    plan_parser.set_defaults(func=cmd_plan)

    init_parser = subparsers.add_parser("init-campaign", help="Symlink-safe campaign creation + execution_order.csv + initial manifest.")
    init_parser.add_argument("--campaign-id", required=True)
    init_parser.add_argument("--run-kind", required=True, choices=("smoke", "benchmark"))
    init_parser.add_argument("--iterations", required=True, type=int)
    init_parser.add_argument("--warmup-iterations", required=True, type=int)
    init_parser.add_argument("--repetitions", required=True, type=int)
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
    validate_parser.add_argument("--iterations", required=True, type=int)
    validate_parser.add_argument("--warmup-iterations", required=True, type=int)
    validate_parser.add_argument("--repetitions", required=True, type=int)
    validate_parser.add_argument("--git-commit", required=True)
    validate_parser.set_defaults(func=cmd_validate_case)

    finalize_parser = subparsers.add_parser("finalize", help="Validate, consolidate, aggregate, and close a campaign.")
    finalize_parser.add_argument("--campaign-dir", required=True)
    finalize_parser.add_argument("--campaign-id", required=True)
    finalize_parser.add_argument("--run-kind", required=True, choices=("smoke", "benchmark"))
    finalize_parser.add_argument("--iterations", required=True, type=int)
    finalize_parser.add_argument("--warmup-iterations", required=True, type=int)
    finalize_parser.add_argument("--repetitions", required=True, type=int)
    finalize_parser.add_argument("--git-commit", required=True)
    finalize_parser.add_argument("--gpu-index", required=True, type=int)
    finalize_parser.add_argument("--started-at-utc", required=True)
    finalize_parser.add_argument("--completed-at-utc", required=True)
    finalize_parser.add_argument("--self-test-umma-1sm", dest="self_test_umma_1sm", required=True, choices=("PASS", "FAIL"))
    finalize_parser.add_argument("--self-test-umma-2sm", dest="self_test_umma_2sm", required=True, choices=("PASS", "FAIL"))
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
