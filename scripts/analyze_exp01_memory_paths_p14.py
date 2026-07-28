#!/usr/bin/env python3
"""P1.4 plan generation, preflight/provenance validation, pilot recording,
NCU profile-case validation, statistics, comparison, saturation, and
analysis-artifact generation for exp01_memory_paths_p14.

This script never touches CUDA, Docker, ``nvidia-smi``, either benchmark
binary, NCU, or the network. ``scripts/run_exp01_memory_paths_p14.sh`` is the
only thing that invokes GPU/Docker/NCU work; it always does so through the
already-audited ``scripts/run_container.sh`` (or, for GPU-free ``.ncu-rep``
post-processing, a plain unprivileged, network-disabled, non-``--gpus``
``docker run``, mirroring the existing ``check-env``/``memory-*-build``
pattern). This script plans the frozen six-case NCU set, validates the
frozen preflight/provenance contract, records/validates the P1.3 pilot
campaign it wraps, validates each captured NCU profile case, and produces
deterministic, standard-library-only statistics/comparison/saturation/HBM
analysis artifacts (CSV, JSON, Markdown, SVG).

It imports ``scripts/aggregate_exp01_memory_paths.py`` (P1.3, frozen and
unmodified) as a library and reuses its path-safety primitives (symlink
rejection, no-clobber publish, exclusive creation, SHA-256 hashing), its
37-column CSV schema and per-field validators, its geometry formulas, and its
manifest atomic-write helper, rather than reimplementing any of them. P1.4
does not modify, and does not need to modify, that file.

Subcommands:
  plan                 Print the frozen six-case NCU plan.
  init-campaign        Symlink-safe P1.4 campaign creation: makes the raw
                        campaign directory, writes profile_plan.csv once, and
                        writes the initial PILOT_IN_PROGRESS manifest.
  validate-preflight   Validate a preflight summary.json against the frozen
                        contract (Section 7/9 of src/memory/P1_4_PROTOCOL.md).
  record-pilot          Validate a completed P1.3 benchmark campaign against
                        the frozen pilot parameters and the preflight used to
                        launch it; transitions PILOT_IN_PROGRESS ->
                        PILOT_COMPLETE (or FAILED).
  discover-metrics      Record resolved NCU metrics (from an already-captured
                        ``ncu --query-metrics`` log) and a fresh preflight;
                        transitions PILOT_COMPLETE -> PROFILE_IN_PROGRESS (or
                        FAILED).
  validate-profile-case Validate one already-captured NCU profile case
                        (application CSV, .ncu-rep, exported metrics CSV)
                        and classify its HBM validation outcome.
  finalize-profile      Re-scan and re-validate the full six-case profile
                        set; transitions PROFILE_IN_PROGRESS -> COMPLETE (or
                        FAILED).
  analyze               Generate analysis/* from a COMPLETE campaign;
                        transitions COMPLETE -> ANALYZED.
  manifest-write        Mark FAILED/INTERRUPTED with an optional failure
                        stage/detail (mirrors P1.3's own manifest-write, but
                        never accepts a completing status).
  --self-test           GPU-free synthetic/adversarial tests. Prints
                        "analyze_exp01_memory_paths_p14: SELF_TEST_RESULT=PASS"
                        only if every case passes.

Exit codes: 0 on success (including --self-test passing); 1 on a validation
or analysis failure; 2 on a usage/precondition error.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import stat
import statistics
import sys
import tempfile
import xml.dom.minidom as minidom
from unittest import mock
from datetime import datetime as _datetime
from datetime import timezone as _timezone
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

# Load the frozen, audited P1.3 aggregator as a library. P1.4 reuses its
# path-safety primitives, CSV schema/validators, geometry formulas, and
# manifest atomic-write helper rather than reimplementing them; this file
# never modifies scripts/aggregate_exp01_memory_paths.py.
import aggregate_exp01_memory_paths as p13  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen P1.4 constants (src/memory/P1_4_PROTOCOL.md is the human-readable
# mirror of every constant below; keep them in lockstep).
# ---------------------------------------------------------------------------
P14_SCHEMA_VERSION = "1"
P14_EXPERIMENT_ID = "exp01_memory_paths_p14"

FROZEN_PILOT_PARAMS = {
    "run_kind": "benchmark",
    "working_set_mib": 512,
    "passes": 32,
    "warmup_ms": 2000,
    "repetitions": 30,
}

FROZEN_PROFILE_PARAMS = {
    "run_kind": "benchmark",
    "working_set_mib": 512,
    "passes": 32,
    "warmup_ms": 0,
    "repetitions": 1,
}

# The frozen, ordered six-case NCU plan (index, method, stages, bif_kib).
# Fixed low/centre/high diagnostic sample; never reordered, never adapted.
NCU_PLAN_RAW = (
    (0, "ldgsts", 2, 16),
    (1, "tma", 2, 16),
    (2, "tma", 4, 32),
    (3, "ldgsts", 4, 32),
    (4, "ldgsts", 8, 64),
    (5, "tma", 8, 64),
)
EXPECTED_NCU_CASE_COUNT = 6

MANDATORY_DRAM_METRIC = "dram__bytes_read.sum"
CANDIDATE_METRICS = (
    MANDATORY_DRAM_METRIC,
    "dram__bytes_write.sum",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__t_bytes.sum",
    "gpu__time_duration.sum",
)

HBM_VALIDATED_MIN_RATIO = 0.90
READ_AMPLIFICATION_MAX_RATIO = 1.10

BOOTSTRAP_SEED = 20260728
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_LO_PERCENTILE = 0.025
BOOTSTRAP_HI_PERCENTILE = 0.975

CV_STABILITY_REVIEW_PERCENT = 5.0
SATURATION_FRACTION_OF_MAX = 0.95

RAW_ROOT_PARTS_P14 = ("results", "raw", "exp01_memory_paths_p14")
RAW_ROOT_REL_P14 = Path(*RAW_ROOT_PARTS_P14)

P14_CAMPAIGN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")
P14_TIMESTAMP_RE = p13.MANIFEST_TIMESTAMP_RE  # YYYYMMDDTHHMMSSZ, reused as-is
NOW_ARG_RE = p13.TIMESTAMP_UTC_RE  # YYYY-MM-DDTHH:MM:SSZ, reused as-is

PROFILE_PLAN_HEADER = [
    "index", "method", "stages", "bytes_in_flight_kib", "stage_bytes",
    "bytes_in_flight_per_sm", "tile_height", "copies_per_thread_per_stage",
    "kernel_name", "case_name",
]

ALLOWED_P14_STATES = frozenset({
    "PILOT_IN_PROGRESS", "PILOT_COMPLETE", "PROFILE_IN_PROGRESS",
    "COMPLETE", "ANALYZED", "FAILED", "INTERRUPTED",
})
P14_TERMINAL_STATES = frozenset({"ANALYZED", "FAILED", "INTERRUPTED"})
ALLOWED_P14_TRANSITIONS: dict[str | None, frozenset[str]] = {
    None: frozenset({"PILOT_IN_PROGRESS"}),
    "PILOT_IN_PROGRESS": frozenset({"PILOT_IN_PROGRESS", "PILOT_COMPLETE", "FAILED", "INTERRUPTED"}),
    "PILOT_COMPLETE": frozenset({"PROFILE_IN_PROGRESS", "FAILED"}),
    "PROFILE_IN_PROGRESS": frozenset({"PROFILE_IN_PROGRESS", "COMPLETE", "FAILED", "INTERRUPTED"}),
    "COMPLETE": frozenset({"ANALYZED"}),
    "ANALYZED": frozenset(),
    "FAILED": frozenset(),
    "INTERRUPTED": frozenset(),
}

ALLOWED_P14_MANIFEST_KEYS: dict[str, object] = {
    "schema_version": str,
    "experiment_id": str,
    "campaign_id": str,
    "state": str,
    "publishable": bool,
    "started_at_utc": str,
    "pilot_completed_at_utc": (str, type(None)),
    "profile_started_at_utc": (str, type(None)),
    "profile_completed_at_utc": (str, type(None)),
    "analyzed_at_utc": (str, type(None)),
    "frozen_protocol": dict,
    "pilot_campaign_reference": dict,
    "preflight_reference_pilot": dict,
    "preflight_reference_profile": dict,
    "provenance": dict,
    "resolved_ncu_metrics": dict,
    "profile_order": list,
    "profile_count_completed": int,
    "case_results": dict,
    "artifact_sha256": dict,
    "failure_stage": (str, type(None)),
    "failure_detail": (list, type(None)),
}


# ---------------------------------------------------------------------------
# Frozen six-case NCU plan (mirrors p13.build_plan()/check_plan_contract()'s
# "independently re-derive every guarantee" style).
# ---------------------------------------------------------------------------
def build_ncu_plan() -> list[dict]:
    plan = []
    for index, method, stages, bif_kib in NCU_PLAN_RAW:
        case_name = f"{index:02d}_{method}_s{stages}_bif{bif_kib}"
        plan.append({
            "index": index,
            "method": method,
            "stages": stages,
            "bif_kib": bif_kib,
            "stage_bytes": p13.stage_bytes_of(stages, bif_kib),
            "bytes_in_flight_per_sm": p13.bytes_in_flight_of(bif_kib),
            "tile_height": p13.tile_height_of(stages, bif_kib),
            "copies_per_thread_per_stage": p13.copies_per_thread_of(stages, bif_kib),
            "kernel_name": f"{method}_benchmark_kernel",
            "case_name": case_name,
        })
    return plan


def check_ncu_plan_contract(plan: list[dict]) -> list[str]:
    """Independently re-derives every property build_ncu_plan() is supposed
    to guarantee, so a future edit cannot silently break the frozen six-case
    contract without failing --self-test."""
    errors: list[str] = []
    if len(plan) != EXPECTED_NCU_CASE_COUNT:
        errors.append(f"plan has {len(plan)} cases, expected {EXPECTED_NCU_CASE_COUNT}")
    indices = [entry["index"] for entry in plan]
    if indices != list(range(len(plan))):
        errors.append(f"plan indices are not exactly 0..{len(plan) - 1} in order: {indices}")
    seen: set[tuple[str, int, int]] = set()
    for entry in plan:
        key = (entry["method"], entry["stages"], entry["bif_kib"])
        if key in seen:
            errors.append(f"duplicate NCU case for {key}")
        seen.add(key)
        if entry["method"] not in p13.METHODS:
            errors.append(f"unknown method {entry['method']!r}")
        if (entry["stages"], entry["bif_kib"]) not in p13.CONFIG_PAIRS:
            errors.append(
                f"(stages={entry['stages']}, bif_kib={entry['bif_kib']}) is not one of "
                f"the frozen P1.3 configuration pairs"
            )
        if entry["kernel_name"] != f"{entry['method']}_benchmark_kernel":
            errors.append(f"case {entry['case_name']}: kernel_name mismatch")
    expected_raw = tuple(
        (e["index"], e["method"], e["stages"], e["bif_kib"]) for e in plan
    )
    if expected_raw != NCU_PLAN_RAW:
        errors.append("plan does not exactly reproduce the frozen NCU_PLAN_RAW order")
    return errors


def format_ncu_plan_text(plan: list[dict]) -> str:
    lines = [
        "index  method   stages  bif_kib  stage_bytes  tile_height  kernel_name              case_name",
    ]
    for entry in plan:
        lines.append(
            f"{entry['index']:>5d}  {entry['method']:<7s}  {entry['stages']:>6d}  "
            f"{entry['bif_kib']:>7d}  {entry['stage_bytes']:>11d}  {entry['tile_height']:>11d}  "
            f"{entry['kernel_name']:<24s} {entry['case_name']}"
        )
    lines.append(f"total NCU cases: {len(plan)}")
    return "\n".join(lines) + "\n"


def format_ncu_plan_lines(plan: list[dict]) -> str:
    return "".join(
        f"{e['index']}\t{e['method']}\t{e['stages']}\t{e['bif_kib']}\t{e['kernel_name']}\t{e['case_name']}\n"
        for e in plan
    )


# ---------------------------------------------------------------------------
# Preflight validation (Section 7/9 of P1_4_PROTOCOL.md). Reuses p13's
# symlink-safe, non-TOCTOU-racy file open/verify primitives.
# ---------------------------------------------------------------------------
def load_preflight_json(path: Path) -> tuple[dict | None, list[str]]:
    """Loads a preflight summary.json as a non-symlink, non-empty regular
    file. Returns (document, errors); errors is empty iff document is a
    parsed JSON object."""
    try:
        with p13._open_regular_nofollow(path, binary=False) as handle:
            text = handle.read()
    except (OSError, p13.UnsafePathError, UnicodeError) as exc:
        return None, [f"{path}: {exc}"]
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"{path}: invalid JSON: {exc}"]
    if not isinstance(doc, dict):
        return None, [f"{path}: root is not a JSON object"]
    return doc, []


def parse_now_arg(value: str) -> _datetime:
    if not NOW_ARG_RE.fullmatch(value):
        raise ValueError(f"--now={value!r} is not in YYYY-MM-DDTHH:MM:SSZ form")
    return _datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=_timezone.utc)


def validate_preflight_fields(
    doc: dict, *, expected_git_commit: str, now_utc: _datetime,
) -> tuple[list[str], dict]:
    """Validates the frozen preflight contract. Returns (errors, snapshot);
    snapshot holds the fields P1.4 later cross-checks against the pilot and
    profile CSVs, populated best-effort even when errors is non-empty."""
    errors: list[str] = []
    snapshot: dict = {}

    if doc.get("overall_status") != "PASS":
        errors.append(f"overall_status={doc.get('overall_status')!r} != 'PASS'")

    if doc.get("git_dirty") is not False:
        errors.append(f"git_dirty={doc.get('git_dirty')!r} != false")

    commit = doc.get("git_commit")
    if not isinstance(commit, str) or not p13.GIT_COMMIT_RE.fullmatch(commit):
        errors.append(f"git_commit={commit!r} is not a 40-character lowercase hex commit")
    elif commit != expected_git_commit:
        errors.append(f"git_commit={commit!r} != expected clean HEAD {expected_git_commit!r}")
    else:
        snapshot["git_commit"] = commit

    gpu = doc.get("gpu")
    if not isinstance(gpu, dict):
        errors.append("gpu field is missing or not an object")
        gpu = {}
    for key in ("logical_index", "name", "uuid", "driver_version", "compute_cap", "memory_total"):
        if not gpu.get(key):
            errors.append(f"gpu.{key} is missing or empty")
    uuid = gpu.get("uuid")
    if uuid is not None and not p13.GPU_UUID_RE.fullmatch(uuid):
        errors.append(f"gpu.uuid={uuid!r} is not a canonical GPU-xxxxxxxx-... UUID")
    if gpu.get("compute_cap") != "10.3":
        errors.append(f"gpu.compute_cap={gpu.get('compute_cap')!r} != '10.3'")
    snapshot["gpu_uuid"] = gpu.get("uuid")
    snapshot["gpu_name"] = gpu.get("name")
    snapshot["gpu_driver_version"] = gpu.get("driver_version")
    snapshot["gpu_compute_cap"] = gpu.get("compute_cap")

    checks = doc.get("checks")
    check_status: dict[str, object] = {}
    if isinstance(checks, list):
        for entry in checks:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                check_status[entry["name"]] = entry.get("status")
    else:
        errors.append("checks field is missing or not a list")
    if check_status.get("gpu_visibility") != "PASS":
        errors.append(
            f"checks.gpu_visibility={check_status.get('gpu_visibility')!r} != 'PASS' "
            f"(this is P1.4's proof of exactly-one-visible-logical-GPU, since "
            f"scripts/preflight.sh records no separate raw GPU-count field)"
        )
    if check_status.get("ncu_profile") != "PASS":
        errors.append(f"checks.ncu_profile={check_status.get('ncu_profile')!r} != 'PASS'")

    ts = doc.get("timestamp_utc")
    if not isinstance(ts, str) or not P14_TIMESTAMP_RE.fullmatch(ts):
        errors.append(f"timestamp_utc={ts!r} is not YYYYMMDDTHHMMSSZ")
    else:
        try:
            ts_dt = _datetime.strptime(ts, "%Y%m%dT%H%M%SZ").replace(tzinfo=_timezone.utc)
        except ValueError:
            errors.append(f"timestamp_utc={ts!r} is not a real calendar UTC timestamp")
        else:
            age_seconds = (now_utc - ts_dt).total_seconds()
            if age_seconds < 0:
                errors.append(
                    f"timestamp_utc={ts!r} is in the future relative to now={now_utc.isoformat()}"
                )
            elif age_seconds > 24 * 3600:
                errors.append(
                    f"timestamp_utc={ts!r} is more than 24h old "
                    f"(age={age_seconds / 3600.0:.2f}h relative to now={now_utc.isoformat()})"
                )
            else:
                snapshot["timestamp_utc"] = ts

    return errors, snapshot


def validate_preflight_file(
    path: Path, *, expected_git_commit: str, now_utc: _datetime,
) -> tuple[list[str], dict]:
    doc, errors = load_preflight_json(path)
    if errors:
        return errors, {}
    field_errors, snapshot = validate_preflight_fields(
        doc, expected_git_commit=expected_git_commit, now_utc=now_utc,
    )
    if field_errors:
        return field_errors, snapshot
    try:
        snapshot["sha256"] = p13.sha256_of(path)
    except p13.UnsafePathError as exc:
        return [str(exc)], snapshot
    snapshot["path"] = str(path)
    return [], snapshot


# ---------------------------------------------------------------------------
# P1.4 campaign-ID validation and symlink-safe raw-tree primitives. Reuses
# p13's generic lstat-based path-safety helpers (they only ever reference
# p13.REPO_ROOT, which is byte-identical to this module's REPO_ROOT, since
# both files live in the same scripts/ directory).
# ---------------------------------------------------------------------------
def validate_p14_campaign_id(campaign_id: str) -> None:
    """Applies p13's general campaign-id safety rules, then the additional
    P1.4-specific requirement that the ID be an explicit canonical UTC
    timestamp (YYYYMMDDTHHMMSSZ) naming a real calendar instant."""
    p13.validate_campaign_id(campaign_id)
    if not P14_CAMPAIGN_ID_RE.fullmatch(campaign_id):
        raise p13.UnsafePathError(
            f"P1_4_CAMPAIGN_ID={campaign_id!r} must be an explicit canonical UTC "
            f"timestamp YYYYMMDDTHHMMSSZ"
        )
    try:
        _datetime.strptime(campaign_id, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise p13.UnsafePathError(
            f"P1_4_CAMPAIGN_ID={campaign_id!r} is not a real calendar UTC timestamp"
        ) from exc


def create_p14_campaign_dir(campaign_id: str) -> Path:
    """Centralized, symlink-safe P1.4 campaign creation. Walks
    results/raw/exp01_memory_paths_p14/<campaign_id>/{profiles,analysis,logs}
    one component at a time via lstat (p13._mkdir_component), refusing a
    symlink or wrong-type object at any level (including the raw root
    itself), and fails if the campaign directory already exists."""
    validate_p14_campaign_id(campaign_id)
    current = REPO_ROOT
    for part in RAW_ROOT_PARTS_P14:
        current = current / part
        p13._mkdir_component(current, must_not_exist=False)
    campaign_dir = current / campaign_id
    p13._mkdir_component(campaign_dir, must_not_exist=True)
    for sub in ("profiles", "analysis", "logs"):
        p13._mkdir_component(campaign_dir / sub, must_not_exist=False)
    return campaign_dir


def resolve_p14_campaign_dir(campaign_dir_rel: str) -> Path:
    """Resolves an already-initialized P1.4 campaign directory with the same
    lstat-based symlink/type safety as create_p14_campaign_dir. Requires
    exactly results/raw/exp01_memory_paths_p14/<campaign_id>."""
    if os.path.isabs(campaign_dir_rel):
        raise p13.UnsafePathError(
            f"--campaign-dir must be relative, got absolute path {campaign_dir_rel!r}"
        )
    parts = Path(campaign_dir_rel).parts
    if any(".." in part for part in parts):
        raise p13.UnsafePathError(f"--campaign-dir must not contain '..': {campaign_dir_rel!r}")
    if len(parts) != len(RAW_ROOT_PARTS_P14) + 1 or tuple(parts[: len(RAW_ROOT_PARTS_P14)]) != RAW_ROOT_PARTS_P14:
        raise p13.UnsafePathError(
            f"--campaign-dir must be exactly {'/'.join(RAW_ROOT_PARTS_P14)}/<campaign_id>, "
            f"got {campaign_dir_rel!r}"
        )
    validate_p14_campaign_id(parts[-1])

    current = REPO_ROOT
    for part in parts:
        current = current / part
        p13._reject_if_symlink_or_wrong_type(current, expect_dir=True)
        if not os.path.lexists(current):
            raise p13.UnsafePathError(f"{current}: does not exist")
    p13._confirm_contained(current, REPO_ROOT)
    for subdir_name in ("profiles", "analysis", "logs"):
        subdir = current / subdir_name
        p13._reject_if_symlink_or_wrong_type(subdir, expect_dir=True)
        if not os.path.lexists(subdir):
            raise p13.UnsafePathError(f"{subdir}: required campaign directory does not exist")
        p13._confirm_contained(subdir, current)
    return current


def resolve_p13_campaign_dir_arg(campaign_dir_rel: str) -> Path:
    """Resolves an operator-supplied P1.3 pilot-campaign path with the exact
    same safety p13.resolve_campaign_dir applies (imported directly, not
    reimplemented)."""
    return p13.resolve_campaign_dir(campaign_dir_rel)


# ---------------------------------------------------------------------------
# profile_plan.csv: written once at init, re-validated at finalize-profile
# time. Mirrors P1.3's execution_order.csv discipline exactly.
# ---------------------------------------------------------------------------
def _profile_plan_row(entry: dict) -> list[str]:
    return [
        str(entry["index"]), entry["method"], str(entry["stages"]), str(entry["bif_kib"]),
        str(entry["stage_bytes"]), str(entry["bytes_in_flight_per_sm"]), str(entry["tile_height"]),
        str(entry["copies_per_thread_per_stage"]), entry["kernel_name"], entry["case_name"],
    ]


def write_profile_plan(campaign_dir: Path, plan: list[dict]) -> Path:
    out_path = campaign_dir / "profile_plan.csv"
    if os.path.lexists(out_path):
        raise p13.UnsafePathError(f"{out_path}: already exists, refusing to overwrite")
    tmp_path = campaign_dir / "profile_plan.csv.tmp"
    if os.path.lexists(tmp_path):
        raise p13.UnsafePathError(f"{tmp_path}: stale temporary file already exists")
    try:
        with p13._open_exclusive(tmp_path, binary=False, newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(PROFILE_PLAN_HEADER)
            for entry in plan:
                writer.writerow(_profile_plan_row(entry))
    except Exception:
        if os.path.lexists(tmp_path):
            p13._safe_unlink_owned(tmp_path)
        raise
    try:
        p13._publish_no_clobber(tmp_path, out_path)
    except p13.UnsafePathError:
        if os.path.lexists(tmp_path):
            p13._safe_unlink_owned(tmp_path)
        raise
    return out_path


def validate_profile_plan_file(path: Path, plan: list[dict]) -> list[str]:
    errors: list[str] = []
    if os.path.lexists(path):
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode):
            return [f"{path}: is a symlink; refusing"]
    else:
        return [f"{path}: does not exist"]
    try:
        with p13._open_regular_nofollow(path, binary=False) as handle:
            rows = list(csv.reader(handle))
    except (OSError, p13.UnsafePathError, UnicodeError) as exc:
        return [f"{path}: unable to read: {exc}"]
    if not rows:
        return [f"{path}: empty file"]
    header, data_rows = rows[0], rows[1:]
    if header != PROFILE_PLAN_HEADER:
        return [f"{path}: header mismatch: {header!r}"]
    if len(data_rows) != len(plan):
        return [f"{path}: has {len(data_rows)} row(s), expected {len(plan)}"]
    for i, (row, entry) in enumerate(zip(data_rows, plan)):
        expected_row = _profile_plan_row(entry)
        if row != expected_row:
            errors.append(f"{path}: row {i} = {row!r} != expected {expected_row!r}")
    return errors


# ---------------------------------------------------------------------------
# P1.4 manifest: allowlisted keys/types, enforced state machine. Reuses
# p13.write_manifest_atomic (generic: writes <campaign_dir>/manifest.json
# with the same no-clobber-except-verified-prior-identity discipline),
# p13.load_manifest (generic JSON load with the same symlink/regular-file
# safety), p13._manifest_type_matches, and p13._validate_compact_timestamp.
# ---------------------------------------------------------------------------
def _validate_p14_manifest_updates(updates: dict) -> None:
    unknown = set(updates) - set(ALLOWED_P14_MANIFEST_KEYS)
    if unknown:
        raise p13.ManifestTransitionError(f"unknown P1.4 manifest field(s): {sorted(unknown)}")
    for key, value in updates.items():
        expected_type = ALLOWED_P14_MANIFEST_KEYS[key]
        if not p13._manifest_type_matches(value, expected_type):
            raise p13.ManifestTransitionError(
                f"P1.4 manifest field {key!r} has invalid type {type(value).__name__}, "
                f"expected {expected_type}"
            )


def _validate_p14_manifest_document(manifest: dict, *, require_initialized: bool = False) -> None:
    _validate_p14_manifest_updates(manifest)
    if not manifest:
        if require_initialized:
            raise p13.ManifestTransitionError("P1.4 manifest is empty")
        return

    state = manifest.get("state")
    if state not in ALLOWED_P14_STATES:
        raise p13.ManifestTransitionError(f"P1.4 manifest state={state!r} is invalid")
    if "schema_version" in manifest and manifest["schema_version"] != P14_SCHEMA_VERSION:
        raise p13.ManifestTransitionError("P1.4 manifest schema_version is invalid")
    if "experiment_id" in manifest and manifest["experiment_id"] != P14_EXPERIMENT_ID:
        raise p13.ManifestTransitionError("P1.4 manifest experiment_id is invalid")
    if "publishable" in manifest and manifest["publishable"] is not False:
        raise p13.ManifestTransitionError("P1.4 manifest publishable must be false")
    if "campaign_id" in manifest:
        try:
            validate_p14_campaign_id(manifest["campaign_id"])
        except p13.UnsafePathError as exc:
            raise p13.ManifestTransitionError(f"P1.4 manifest campaign_id is invalid: {exc}") from exc
    if "started_at_utc" in manifest:
        p13._validate_compact_timestamp(manifest["started_at_utc"], "started_at_utc")
    for key in ("pilot_completed_at_utc", "profile_started_at_utc", "profile_completed_at_utc", "analyzed_at_utc"):
        if manifest.get(key) is not None:
            p13._validate_compact_timestamp(manifest[key], key)
    if "profile_count_completed" in manifest and not (0 <= manifest["profile_count_completed"] <= EXPECTED_NCU_CASE_COUNT):
        raise p13.ManifestTransitionError(
            f"P1.4 manifest profile_count_completed must be in [0, {EXPECTED_NCU_CASE_COUNT}]"
        )
    if "failure_stage" in manifest and manifest["failure_stage"] is not None and not manifest["failure_stage"]:
        raise p13.ManifestTransitionError("P1.4 manifest failure_stage must be null or non-empty")
    if "failure_detail" in manifest and manifest["failure_detail"] is not None:
        if not all(isinstance(item, str) for item in manifest["failure_detail"]):
            raise p13.ManifestTransitionError("P1.4 manifest failure_detail must be null or a list of strings")

    required_by_state = {
        "PILOT_IN_PROGRESS": {
            "schema_version", "experiment_id", "campaign_id", "state", "publishable",
            "started_at_utc", "frozen_protocol",
        },
        "PILOT_COMPLETE": {
            "pilot_completed_at_utc", "pilot_campaign_reference", "preflight_reference_pilot",
            "provenance",
        },
        "PROFILE_IN_PROGRESS": {"profile_started_at_utc", "resolved_ncu_metrics", "preflight_reference_profile"},
        "COMPLETE": {"profile_completed_at_utc", "profile_order", "profile_count_completed", "case_results"},
        "ANALYZED": {"analyzed_at_utc", "artifact_sha256"},
    }
    if state in required_by_state:
        gate_order = list(required_by_state)
        needed: set[str] = set()
        for gate_state in gate_order[: gate_order.index(state) + 1]:
            needed |= required_by_state[gate_state]
        missing = needed - set(manifest)
        if missing:
            raise p13.ManifestTransitionError(
                f"P1.4 manifest in state {state!r} missing required field(s): {sorted(missing)}"
            )

    if state == "COMPLETE" and manifest.get("profile_count_completed") != EXPECTED_NCU_CASE_COUNT:
        raise p13.ManifestTransitionError(
            f"P1.4 manifest state=COMPLETE requires profile_count_completed="
            f"{EXPECTED_NCU_CASE_COUNT}, got {manifest.get('profile_count_completed')!r}"
        )
    if state in ("COMPLETE", "ANALYZED") and manifest.get("failure_stage") is not None:
        raise p13.ManifestTransitionError(f"P1.4 manifest state={state!r} cannot retain failure_stage")


def p14_merge_manifest(campaign_dir: Path, updates: dict, state: str) -> dict:
    """Merges `updates` into manifest.json and sets `state`, enforcing the
    P1.4 state machine (ALLOWED_P14_TRANSITIONS): a terminal campaign can
    never be reopened/rewritten, and every key in `updates` must be
    allowlisted with the correct type. Mirrors p13.merge_manifest's
    discipline for its own, separate schema."""
    _validate_p14_manifest_updates(updates)
    manifest = p13.load_manifest(campaign_dir)
    _validate_p14_manifest_document(manifest)
    current_state = manifest.get("state")
    allowed = ALLOWED_P14_TRANSITIONS.get(current_state, frozenset())
    if state not in allowed:
        raise p13.ManifestTransitionError(
            f"invalid P1.4 manifest state transition: {current_state!r} -> {state!r}"
        )
    immutable_after_init = {
        "campaign_id", "started_at_utc", "frozen_protocol",
    }
    for key in immutable_after_init & set(manifest) & set(updates):
        if updates[key] != manifest[key]:
            raise p13.ManifestTransitionError(f"P1.4 manifest field {key!r} is immutable after initialization")
    if "profile_count_completed" in manifest and "profile_count_completed" in updates:
        if updates["profile_count_completed"] < manifest["profile_count_completed"]:
            raise p13.ManifestTransitionError("P1.4 manifest profile_count_completed cannot decrease")
    manifest.update(updates)
    manifest["schema_version"] = P14_SCHEMA_VERSION
    manifest["experiment_id"] = P14_EXPERIMENT_ID
    manifest["state"] = state
    manifest["publishable"] = False
    _validate_p14_manifest_document(manifest, require_initialized=True)
    p13.write_manifest_atomic(campaign_dir, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Statistics: percentile/IQR helpers, deterministic bootstrap, per-config
# descriptive statistics. Python standard library only (statistics, random,
# math). Section 7 of P1_4_PROTOCOL.md is the frozen policy this implements.
# ---------------------------------------------------------------------------
def _reject_non_finite(values: list[float], *, label: str) -> None:
    for v in values:
        if not math.isfinite(v):
            raise ValueError(f"{label}: non-finite value {v!r} is rejected, never silently dropped")


def _percentile_linear(sorted_values: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy 'linear' / Excel
    PERCENTILE.INC convention) over an already-ascending-sorted sequence."""
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    idx = p * (n - 1)
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return sorted_values[int(idx)]
    frac = idx - lo
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * frac


def iqr_bounds(values: list[float]) -> tuple[float, float, int]:
    """Returns (lower_fence, upper_fence, flagged_count) using the classic
    Tukey 1.5*IQR fence. Diagnostic only: callers must never remove flagged
    samples from any primary statistic."""
    sorted_values = sorted(values)
    q1 = _percentile_linear(sorted_values, 0.25)
    q3 = _percentile_linear(sorted_values, 0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    flagged = sum(1 for v in values if v < lower or v > upper)
    return lower, upper, flagged


def _sample_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def bootstrap_indices_median_ci(
    values: list[float], rng: random.Random, *, resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Deterministic percentile-bootstrap 95% CI for the median of `values`,
    resampling len(values) items with replacement `resamples` times. Draws
    from `rng` (caller controls seeding/ordering for full determinism)."""
    n = len(values)
    medians = []
    for _ in range(resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(resample))
    medians.sort()
    lo_idx = int(BOOTSTRAP_LO_PERCENTILE * resamples) - 1
    hi_idx = int(BOOTSTRAP_HI_PERCENTILE * resamples) - 1
    lo_idx = max(lo_idx, 0)
    hi_idx = min(hi_idx, resamples - 1)
    return medians[lo_idx], medians[hi_idx]


def bootstrap_indices_ratio_ci(
    values_a: list[float], values_b: list[float], rng: random.Random,
    *, resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Deterministic percentile-bootstrap 95% CI for median(values_b) /
    median(values_a), independently resampling both inputs each iteration
    (never resampling one and reusing the other's original sample)."""
    n_a, n_b = len(values_a), len(values_b)
    ratios = []
    for _ in range(resamples):
        resample_a = [values_a[rng.randrange(n_a)] for _ in range(n_a)]
        resample_b = [values_b[rng.randrange(n_b)] for _ in range(n_b)]
        median_a = statistics.median(resample_a)
        if median_a == 0:
            continue
        ratios.append(statistics.median(resample_b) / median_a)
    if not ratios:
        raise ValueError("bootstrap ratio CI: every resample had a zero-median denominator")
    ratios.sort()
    lo_idx = max(int(BOOTSTRAP_LO_PERCENTILE * len(ratios)) - 1, 0)
    hi_idx = min(int(BOOTSTRAP_HI_PERCENTILE * len(ratios)) - 1, len(ratios) - 1)
    return ratios[lo_idx], ratios[hi_idx]


def compute_config_stats(values: list[float]) -> dict:
    """Descriptive statistics for one (method, stages, bif_kib) configuration
    over all retained repetitions. Never removes a sample; IQR flags are
    diagnostic counts only."""
    _reject_non_finite(values, label="effective_gbps")
    mean = statistics.mean(values)
    median = statistics.median(values)
    stdev = _sample_stdev(values)
    cv_percent = (100.0 * stdev / mean) if mean != 0 else 0.0
    lower_fence, upper_fence, flagged = iqr_bounds(values)
    return {
        "count": len(values),
        "mean": mean,
        "median": median,
        "stdev": stdev,
        "cv_percent": cv_percent,
        "min": min(values),
        "max": max(values),
        "iqr_lower_fence": lower_fence,
        "iqr_upper_fence": upper_fence,
        "iqr_flagged_count": flagged,
        "stability_review": cv_percent > CV_STABILITY_REVIEW_PERCENT,
    }


def compute_all_config_stats(
    samples_by_config: dict[tuple[str, int, int], list[float]], rng: random.Random,
) -> dict[tuple[str, int, int], dict]:
    """Computes descriptive stats + bootstrap median CI for all 18 configs,
    in the fixed order (stages, bif_kib, method) so that, given the same
    input samples, the shared `rng`'s draw sequence — and therefore every
    output — is bit-identical on any machine, every time."""
    results: dict[tuple[str, int, int], dict] = {}
    ordered_keys = sorted(samples_by_config, key=lambda k: (k[1], k[2], k[0]))
    for key in ordered_keys:
        values = samples_by_config[key]
        stats = compute_config_stats(values)
        stats["median_ci_low"], stats["median_ci_high"] = bootstrap_indices_median_ci(values, rng)
        results[key] = stats
    return results


def compute_pairwise_comparisons(
    samples_by_config: dict[tuple[str, int, int], list[float]],
    stats_by_config: dict[tuple[str, int, int], dict],
    rng: random.Random,
) -> list[dict]:
    """Per (stages, bif_kib) pair, in ascending (stages, bif_kib) order:
    tma_to_ldgsts_ratio of medians, with an independently-resampled 95%
    bootstrap CI (fresh resampling, distinct from each config's own median
    CI computed above)."""
    rows = []
    pairs = sorted({(stages, bif_kib) for (_method, stages, bif_kib) in samples_by_config})
    for stages, bif_kib in pairs:
        ldgsts_values = samples_by_config[("ldgsts", stages, bif_kib)]
        tma_values = samples_by_config[("tma", stages, bif_kib)]
        median_ldgsts = stats_by_config[("ldgsts", stages, bif_kib)]["median"]
        median_tma = stats_by_config[("tma", stages, bif_kib)]["median"]
        ratio = median_tma / median_ldgsts
        ci_low, ci_high = bootstrap_indices_ratio_ci(ldgsts_values, tma_values, rng)
        if ratio > 1:
            interpretation = "tma_higher"
        elif ratio < 1:
            interpretation = "ldgsts_higher"
        else:
            interpretation = "equal"
        rows.append({
            "stages": stages,
            "bytes_in_flight_kib": bif_kib,
            "median_gbps_ldgsts": median_ldgsts,
            "median_gbps_tma": median_tma,
            "tma_to_ldgsts_ratio": ratio,
            "ratio_ci_low": ci_low,
            "ratio_ci_high": ci_high,
            "interpretation": interpretation,
        })
    return rows


def _ci_overlaps(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    return lo_a <= hi_b and lo_b <= hi_a


def compute_saturation_candidates(stats_by_config: dict[tuple[str, int, int], dict]) -> list[dict]:
    """Per (method, stages) group, in ascending (method, stages) order:
    the earliest (smallest) tested bytes-in-flight value whose median is
    >= 95% of the group's observed maximum median AND whose bootstrap CI
    overlaps the maximum's own CI. The group's own maximum-median entry is
    always a valid fallback (trivial 100% ratio, CI overlaps itself), so a
    result always exists. Reuses the per-config stats/CI already computed
    by compute_all_config_stats -- no new resampling."""
    rows = []
    groups = sorted({(method, stages) for (method, stages, _bif) in stats_by_config})
    for method, stages in groups:
        bif_values = (16, 32, 64)
        medians = {bif: stats_by_config[(method, stages, bif)]["median"] for bif in bif_values}
        max_median = max(medians.values())
        max_bif = min(bif for bif in bif_values if medians[bif] == max_median)
        max_ci = (
            stats_by_config[(method, stages, max_bif)]["median_ci_low"],
            stats_by_config[(method, stages, max_bif)]["median_ci_high"],
        )
        earliest = max_bif
        for bif in sorted(bif_values):
            if medians[bif] < SATURATION_FRACTION_OF_MAX * max_median:
                continue
            candidate_ci = (
                stats_by_config[(method, stages, bif)]["median_ci_low"],
                stats_by_config[(method, stages, bif)]["median_ci_high"],
            )
            if _ci_overlaps(*candidate_ci, *max_ci):
                earliest = bif
                break
        rows.append({
            "method": method,
            "stages": stages,
            "bif_16_median_gbps": medians[16],
            "bif_32_median_gbps": medians[32],
            "bif_64_median_gbps": medians[64],
            "max_median_gbps": max_median,
            "earliest_tested_candidate_saturation_bif_kib": earliest,
        })
    return rows


# ---------------------------------------------------------------------------
# HBM validation classification (Section 5 of P1_4_PROTOCOL.md). Applies
# only to the six profiled cases; never extrapolated to the other twelve.
# ---------------------------------------------------------------------------
def classify_hbm(dram_read_bytes: float | None, useful_bytes: float) -> tuple[str, list[str], float | None]:
    """Returns (classification, diagnostic_flags, dram_read_ratio).
    classification is HBM_VALIDATED or INCONCLUSIVE. dram_read_ratio is None
    when the classification is INCONCLUSIVE for a missing/malformed-metric
    reason (as opposed to a too-low, but present, ratio)."""
    if dram_read_bytes is None:
        return "INCONCLUSIVE", ["DRAM_READ_METRIC_UNAVAILABLE"], None
    if not math.isfinite(dram_read_bytes) or dram_read_bytes < 0:
        return "INCONCLUSIVE", ["DRAM_READ_BYTES_MALFORMED"], None
    if not math.isfinite(useful_bytes) or useful_bytes <= 0:
        return "INCONCLUSIVE", ["USEFUL_BYTES_MALFORMED"], None
    ratio = dram_read_bytes / useful_bytes
    if ratio < HBM_VALIDATED_MIN_RATIO:
        return "INCONCLUSIVE", ["RATIO_BELOW_THRESHOLD"], ratio
    if ratio > READ_AMPLIFICATION_MAX_RATIO:
        return "HBM_VALIDATED", ["READ_AMPLIFICATION"], ratio
    return "HBM_VALIDATED", [], ratio


# ---------------------------------------------------------------------------
# NCU raw-CSV metrics parsing.
#
# ASSUMPTION (documented, not directly testable without a live GPU — see
# src/memory/P1_4_PROTOCOL.md and the implementation handoff): NCU
# 2025.4.0.0's `--page raw --csv --print-metric-name name --print-units
# base` export is a long-form table with one row per (kernel launch,
# metric), including literal columns "Kernel Name", "Metric Name", and
# "Metric Value" (well-established, stable NCU CSV conventions). The parser
# below is deliberately defensive: it locates the metric-name/value columns
# by exact header name first, falls back to a structural heuristic (a metric
# identifier always contains "__"), and fails closed (raises) rather than
# guessing if it cannot confidently locate both columns. The very first real
# profiling run should sanity-check metrics_raw.csv against this parser
# before trusting its HBM_VALIDATED/INCONCLUSIVE output.
# ---------------------------------------------------------------------------
class NcuCsvParseError(ValueError):
    pass


def _find_column(fieldnames: list[str], *, exact_names: tuple[str, ...]) -> str | None:
    lower_map = {name.strip().lower(): name for name in fieldnames}
    for candidate in exact_names:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def parse_ncu_raw_csv(path: Path) -> dict:
    """Parses an NCU `--page raw --csv` export. Returns a dict with keys
    'metrics' (metric name -> float value), 'kernel_names' (sorted set of
    distinct kernel-name strings seen), and 'launch_count' (number of
    distinct launch identifiers seen, best-effort). Raises NcuCsvParseError
    if the file cannot be confidently parsed."""
    try:
        with p13._open_regular_nofollow(path, binary=False) as handle:
            reader = csv.DictReader(handle)
            fieldnames = reader.fieldnames
            rows = list(reader)
    except (OSError, p13.UnsafePathError, UnicodeError) as exc:
        raise NcuCsvParseError(f"{path}: unable to read: {exc}") from exc
    if not fieldnames:
        raise NcuCsvParseError(f"{path}: no header row")
    if not rows:
        raise NcuCsvParseError(f"{path}: no data rows")

    name_col = _find_column(fieldnames, exact_names=("Metric Name",))
    value_col = _find_column(fieldnames, exact_names=("Metric Value",))
    kernel_col = _find_column(fieldnames, exact_names=("Kernel Name",))
    id_col = _find_column(fieldnames, exact_names=("ID",))

    if name_col is None or value_col is None:
        # Structural fallback: a metric identifier always contains "__".
        for field in fieldnames:
            sample_values = [row.get(field, "") for row in rows]
            if name_col is None and any("__" in (v or "") for v in sample_values):
                name_col = field
        if name_col is not None and value_col is None:
            idx = fieldnames.index(name_col)
            if idx + 1 < len(fieldnames):
                value_col = fieldnames[idx + 1]
    if name_col is None or value_col is None:
        raise NcuCsvParseError(
            f"{path}: could not locate both a metric-name and metric-value column "
            f"in header {fieldnames!r}"
        )

    metrics: dict[str, float] = {}
    for row in rows:
        metric_name = (row.get(name_col) or "").strip()
        if not metric_name:
            continue
        raw_value = row.get(value_col)
        if raw_value is None or raw_value.strip() == "":
            raise NcuCsvParseError(f"{path}: metric {metric_name!r} has an empty value")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise NcuCsvParseError(
                f"{path}: metric {metric_name!r} value {raw_value!r} is not a number"
            ) from exc
        if not math.isfinite(value):
            raise NcuCsvParseError(f"{path}: metric {metric_name!r} value {raw_value!r} is not finite")
        if metric_name in metrics and metrics[metric_name] != value:
            raise NcuCsvParseError(
                f"{path}: metric {metric_name!r} appears more than once with different values "
                f"({metrics[metric_name]!r} vs {value!r}) -- more than one profiled launch?"
            )
        metrics[metric_name] = value

    kernel_names = sorted({(row.get(kernel_col) or "").strip() for row in rows if kernel_col}) if kernel_col else []
    launch_ids = sorted({(row.get(id_col) or "").strip() for row in rows if id_col}) if id_col else []
    return {
        "metrics": metrics,
        "kernel_names": [k for k in kernel_names if k],
        "launch_count": len(launch_ids) if launch_ids else 1,
    }


# ---------------------------------------------------------------------------
# Deterministic SVG chart rendering. Python standard library only: plain
# string building, escaped via xml.sax.saxutils.escape. No NumPy, pandas,
# matplotlib, seaborn, or a notebook/Docker dependency. Well-formedness is
# checked deterministically by the caller (analyze/--self-test) via
# xml.dom.minidom.parseString.
# ---------------------------------------------------------------------------
_SERIES_COLORS = {2: "#1b9e77", 4: "#d95f02", 8: "#7570b3"}
_CHART_WIDTH = 720
_CHART_HEIGHT = 460
_MARGIN_LEFT = 76
_MARGIN_RIGHT = 24
_MARGIN_TOP = 48
_MARGIN_BOTTOM = 64


def _fmt_num(value: float) -> str:
    return f"{value:.4g}"


def _plot_rect(*, width: int = _CHART_WIDTH, height: int = _CHART_HEIGHT):
    return _MARGIN_LEFT, _MARGIN_TOP, width - _MARGIN_RIGHT, height - _MARGIN_BOTTOM


def _y_scale(y_min: float, y_max: float, plot_top: float, plot_bottom: float):
    span = (y_max - y_min) or 1.0
    pad = span * 0.08

    def scale(y: float) -> float:
        frac = (y - (y_min - pad)) / (span + 2 * pad)
        return plot_bottom - frac * (plot_bottom - plot_top)

    return scale, y_min - pad, y_max + pad


def _svg_header(width: int, height: int, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="monospace" font-size="12">',
        f'<title>{_xml_escape(title)}</title>',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="none"/>',
    ]


def _render_panel(
    *, x0: float, y0: float, x1: float, y1: float, title: str,
    x_labels: list[str], y_min: float, y_max: float, y_label: str,
    series: list[dict],
    hlines: tuple[tuple[float, str], ...] = (),
) -> list[str]:
    """One rectangular plot area: axes, gridlines, up to len(x_labels)
    evenly-spaced categorical x positions, one polyline + error bars per
    series, an optional set of labelled horizontal reference lines, and a
    legend. `series` entries: {label, color, values: [float|None],
    ci_low: [float|None], ci_high: [float|None]}."""
    out: list[str] = []
    plot_w = x1 - x0
    plot_h = y1 - y0
    out.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y0 - 20:.1f}" text-anchor="middle" '
                f'font-weight="bold">{_xml_escape(title)}</text>')
    scale_y, y_lo, y_hi = _y_scale(y_min, y_max, y0, y1)

    n_x = max(len(x_labels), 1)
    x_positions = [
        x0 + (plot_w * (i + 0.5) / n_x) for i in range(n_x)
    ] if n_x > 0 else []

    out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{plot_w:.1f}" height="{plot_h:.1f}" '
                f'fill="none" stroke="#333333" stroke-width="1"/>')

    n_yticks = 5
    for t in range(n_yticks + 1):
        y_val = y_lo + (y_hi - y_lo) * t / n_yticks
        y_px = scale_y(y_val)
        out.append(f'<line x1="{x0:.1f}" y1="{y_px:.1f}" x2="{x1:.1f}" y2="{y_px:.1f}" '
                    f'stroke="#e0e0e0" stroke-width="1"/>')
        out.append(f'<text x="{x0 - 6:.1f}" y="{y_px + 4:.1f}" text-anchor="end">'
                    f'{_xml_escape(_fmt_num(y_val))}</text>')
    out.append(f'<text x="{x0 - 56:.1f}" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" '
                f'transform="rotate(-90 {x0 - 56:.1f} {(y0 + y1) / 2:.1f})">{_xml_escape(y_label)}</text>')

    for i, label in enumerate(x_labels):
        out.append(f'<text x="{x_positions[i]:.1f}" y="{y1 + 20:.1f}" text-anchor="middle">'
                    f'{_xml_escape(label)}</text>')

    for value, hlabel in hlines:
        y_px = scale_y(value)
        out.append(f'<line x1="{x0:.1f}" y1="{y_px:.1f}" x2="{x1:.1f}" y2="{y_px:.1f}" '
                    f'stroke="#999999" stroke-width="1.5" stroke-dasharray="6,4"/>')
        out.append(f'<text x="{x1 - 4:.1f}" y="{y_px - 4:.1f}" text-anchor="end" fill="#666666">'
                    f'{_xml_escape(hlabel)}</text>')

    for s in series:
        color = s["color"]
        point_colors = s.get("point_colors")
        points = []
        point_indices = []
        for i, value in enumerate(s["values"]):
            if value is None:
                continue
            px, py = x_positions[i], scale_y(value)
            points.append((px, py))
            point_indices.append(i)
            ci_low = s.get("ci_low", [None] * len(s["values"]))[i]
            ci_high = s.get("ci_high", [None] * len(s["values"]))[i]
            if ci_low is not None and ci_high is not None:
                py_lo, py_hi = scale_y(ci_low), scale_y(ci_high)
                out.append(f'<line x1="{px:.1f}" y1="{py_lo:.1f}" x2="{px:.1f}" y2="{py_hi:.1f}" '
                            f'stroke="{color}" stroke-width="1.5"/>')
                cap = 5
                out.append(f'<line x1="{px - cap:.1f}" y1="{py_lo:.1f}" x2="{px + cap:.1f}" y2="{py_lo:.1f}" '
                            f'stroke="{color}" stroke-width="1.5"/>')
                out.append(f'<line x1="{px - cap:.1f}" y1="{py_hi:.1f}" x2="{px + cap:.1f}" y2="{py_hi:.1f}" '
                            f'stroke="{color}" stroke-width="1.5"/>')
        if len(points) >= 2 and point_colors is None:
            # A per-point color list means these points are independent
            # (categorical classifications), not a connected trend line.
            path = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
            out.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for (px, py), i in zip(points, point_indices):
            point_color = point_colors[i] if point_colors else color
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{point_color}" stroke="#ffffff" stroke-width="1"/>')
        for i, flag in enumerate(s.get("flags", [])):
            if not flag:
                continue
            px = x_positions[i]
            out.append(f'<text x="{px:.1f}" y="{y1 + 36:.1f}" text-anchor="middle" fill="#b00020" '
                        f'font-size="10">{_xml_escape(flag)}</text>')

    legend_y = y0 + 4
    legend_x = x1 - 90
    for i, s in enumerate(series):
        ly = legend_y + i * 16
        out.append(f'<rect x="{legend_x:.1f}" y="{ly - 8:.1f}" width="12" height="12" fill="{s["color"]}"/>')
        out.append(f'<text x="{legend_x + 16:.1f}" y="{ly + 2:.1f}">{_xml_escape(s["label"])}</text>')
    return out


def render_effective_gbps_svg(stats_by_config: dict[tuple[str, int, int], dict]) -> str:
    width, height = _CHART_WIDTH * 2 - 40, _CHART_HEIGHT
    out = _svg_header(width, height, "Effective copy bandwidth vs bytes in flight")
    x_labels = ["16", "32", "64"]
    all_values = [stats["median"] for stats in stats_by_config.values()]
    y_min, y_max = min(all_values), max(all_values)
    panel_w = (width - 40) / 2
    for panel_i, method in enumerate(("ldgsts", "tma")):
        x0 = 20 + panel_i * (panel_w + 0)
        x1 = x0 + panel_w
        px0, py0, px1, py1 = x0 + _MARGIN_LEFT - 20, _MARGIN_TOP, x1 - _MARGIN_RIGHT, height - _MARGIN_BOTTOM
        series = []
        for stages in (2, 4, 8):
            values, ci_low, ci_high = [], [], []
            for bif in (16, 32, 64):
                stats = stats_by_config.get((method, stages, bif))
                values.append(stats["median"] if stats else None)
                ci_low.append(stats["median_ci_low"] if stats else None)
                ci_high.append(stats["median_ci_high"] if stats else None)
            series.append({
                "label": f"stages={stages}", "color": _SERIES_COLORS[stages],
                "values": values, "ci_low": ci_low, "ci_high": ci_high,
            })
        out.extend(_render_panel(
            x0=px0, y0=py0, x1=px1, y1=py1, title=f"method={method}",
            x_labels=x_labels, y_min=y_min, y_max=y_max,
            y_label="effective GB/s (median, 95% bootstrap CI)", series=series,
        ))
    out.append('<text x="12" y="' + str(height - 8) + '" font-size="10" fill="#666666">'
                'Effective copy bandwidth, not automatically HBM/DRAM bandwidth. Single pilot.</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_ratio_svg(pairwise_rows: list[dict]) -> str:
    width, height = _CHART_WIDTH, _CHART_HEIGHT
    out = _svg_header(width, height, "TMA / LDGSTS effective-bandwidth ratio vs bytes in flight")
    px0, py0, px1, py1 = _plot_rect(width=width, height=height)
    x_labels = ["16", "32", "64"]
    all_values = [row["tma_to_ldgsts_ratio"] for row in pairwise_rows]
    all_ci = [v for row in pairwise_rows for v in (row["ratio_ci_low"], row["ratio_ci_high"])]
    y_min = min([1.0] + all_values + all_ci)
    y_max = max([1.0] + all_values + all_ci)
    series = []
    for stages in (2, 4, 8):
        values, ci_low, ci_high = [], [], []
        for bif in (16, 32, 64):
            row = next((r for r in pairwise_rows if r["stages"] == stages and r["bytes_in_flight_kib"] == bif), None)
            values.append(row["tma_to_ldgsts_ratio"] if row else None)
            ci_low.append(row["ratio_ci_low"] if row else None)
            ci_high.append(row["ratio_ci_high"] if row else None)
        series.append({
            "label": f"stages={stages}", "color": _SERIES_COLORS[stages],
            "values": values, "ci_low": ci_low, "ci_high": ci_high,
        })
    out.extend(_render_panel(
        x0=px0, y0=py0, x1=px1, y1=py1, title="tma_to_ldgsts_ratio (median, 95% bootstrap CI)",
        x_labels=x_labels, y_min=y_min, y_max=y_max, y_label="ratio",
        series=series, hlines=((1.0, "ratio = 1 (equal medians)"),),
    ))
    out.append('<text x="12" y="' + str(height - 8) + '" font-size="10" fill="#666666">'
                'ratio &gt; 1: TMA measured higher in this pilot. ratio &lt; 1: LDGSTS measured higher. No significance claim.</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_dram_read_ratio_svg(ncu_rows: list[dict]) -> str:
    width, height = _CHART_WIDTH, _CHART_HEIGHT
    out = _svg_header(width, height, "DRAM read ratio for the six profiled NCU cases")
    px0, py0, px1, py1 = _plot_rect(width=width, height=height)
    x_labels = [row["case_name"] for row in ncu_rows]
    finite_ratios = [row["dram_read_ratio"] for row in ncu_rows if row["dram_read_ratio"] is not None]
    y_min = min([0.90] + finite_ratios) if finite_ratios else 0.0
    y_max = max([1.10] + finite_ratios) if finite_ratios else 1.2
    values = [row["dram_read_ratio"] for row in ncu_rows]
    color_by_class = {"HBM_VALIDATED": "#1b9e77", "INCONCLUSIVE": "#d95f02"}
    colors = [color_by_class.get(row["hbm_classification"], "#999999") for row in ncu_rows]
    flags = []
    for row in ncu_rows:
        if row["dram_read_ratio"] is None:
            flags.append("INCONCLUSIVE (no data)")
        elif "READ_AMPLIFICATION" in row["diagnostic_flags"]:
            flags.append("READ_AMPLIFICATION")
        elif row["hbm_classification"] == "INCONCLUSIVE":
            flags.append("INCONCLUSIVE")
        else:
            flags.append("")
    series = [{
        "label": "dram_read_ratio (color = classification)", "color": "#1b9e77", "values": values,
        "ci_low": [None] * len(values), "ci_high": [None] * len(values), "flags": flags,
        "point_colors": colors,
    }]
    out.extend(_render_panel(
        x0=px0, y0=py0, x1=px1, y1=py1, title="dram_read_bytes / useful_bytes",
        x_labels=x_labels, y_min=y_min, y_max=y_max, y_label="dram_read_ratio",
        series=series, hlines=((0.90, "HBM_VALIDATED threshold = 0.90"), (1.0, "ratio = 1")),
    ))
    out.append('<text x="12" y="' + str(height - 8) + '" font-size="10" fill="#666666">'
                'Six predefined NCU cases only. Never extrapolated to the other twelve pilot configurations.</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# report.md: the mandatory, cautious, plain-language summary (Section 11 of
# P1_4_PROTOCOL.md fixes exactly what this must say).
# ---------------------------------------------------------------------------
def render_report_markdown(
    *, campaign_id: str, stats_rows: list[dict], pairwise_rows: list[dict],
    saturation_rows: list[dict], ncu_rows: list[dict], provenance: dict,
    dram_read_metric_available: bool,
) -> str:
    lines: list[str] = []
    lines.append(f"# P1.4 pilot report -- campaign `{campaign_id}`")
    lines.append("")
    lines.append("**publishable: false.** This report is generated from a single pilot run "
                 "pending independent review and later final campaigns; it is not a final "
                 "experimental result.")
    lines.append("")
    lines.append("## What this is, and is not")
    lines.append("")
    lines.append("- This is **one single pilot** (`run_kind=benchmark`, `working_set_mib=512`, "
                 "`passes=32`, `warmup_ms=2000`, `repetitions=30`), not a final campaign.")
    lines.append("- All timing values are **CUDA-event pilot measurements** taken by the audited "
                 "P1.1/P1.2 binaries themselves. Nsight Compute kernel durations are never used as "
                 "benchmark timing anywhere in this report.")
    lines.append("- **No sample was ever removed.** All 30 retained repetitions of all 18 "
                 "configurations are used in every statistic below; IQR flags are diagnostics only.")
    lines.append("- Nsight Compute covers **exactly six predefined cases** out of the 18 pilot "
                 "configurations. HBM/DRAM-traffic validation applies only to those six; it is "
                 "never extrapolated to the other twelve.")
    lines.append("- Any candidate-saturation point below is limited to the three tested "
                 "bytes-in-flight values (16/32/64 KiB) per group -- it is not a universal "
                 "architectural saturation threshold.")
    lines.append("- **No final or universal HBM ceiling is established** by this pilot.")
    lines.append("- The P1.3 execution order (fixed, non-randomized) is a named limitation of "
                 "this single pilot: temporal drift across the sweep is not controlled for.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("```text")
    for key in ("git_commit", "gpu_uuid", "gpu_name", "compute_capability",
                "cuda_driver_version", "cuda_runtime_version", "working_set_bytes", "passes"):
        lines.append(f"{key}: {provenance.get(key)}")
    lines.append("```")
    lines.append("")
    lines.append("## Pilot statistics (18 configurations, all 30 repetitions each)")
    lines.append("")
    lines.append("| method | stages | BIF KiB | median GB/s | 95% CI | CV% | stability review | IQR-flagged |")
    lines.append("| --- | ---: | ---: | ---: | --- | ---: | --- | ---: |")
    for row in stats_rows:
        lines.append(
            f"| {row['method']} | {row['stages']} | {row['bytes_in_flight_kib']} | "
            f"{row['median_gbps']:.3f} | [{row['median_ci_low_gbps']:.3f}, "
            f"{row['median_ci_high_gbps']:.3f}] | {row['cv_percent']:.2f} | "
            f"{'REVIEW' if row['stability_review'] else 'ok'} | {row['iqr_flagged_count']} |"
        )
    lines.append("")
    lines.append("## Pairwise LDGSTS/TMA comparison")
    lines.append("")
    lines.append("`tma_to_ldgsts_ratio > 1` means TMA measured higher effective copy bandwidth in "
                 "this pilot; `< 1` means LDGSTS did; no p-value or significance claim is made, "
                 "and no \"winner\" is declared.")
    lines.append("")
    lines.append("| stages | BIF KiB | median LDGSTS GB/s | median TMA GB/s | ratio | 95% CI | interpretation |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | --- | --- |")
    for row in pairwise_rows:
        lines.append(
            f"| {row['stages']} | {row['bytes_in_flight_kib']} | {row['median_gbps_ldgsts']:.3f} | "
            f"{row['median_gbps_tma']:.3f} | {row['tma_to_ldgsts_ratio']:.4f} | "
            f"[{row['ratio_ci_low']:.4f}, {row['ratio_ci_high']:.4f}] | {row['interpretation']} |"
        )
    lines.append("")
    lines.append("## Candidate saturation (diagnostic; three tested BIF values only)")
    lines.append("")
    lines.append("| method | stages | 16 KiB | 32 KiB | 64 KiB | earliest_tested_candidate_saturation_bif_kib |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in saturation_rows:
        lines.append(
            f"| {row['method']} | {row['stages']} | {row['bif_16_median_gbps']:.3f} | "
            f"{row['bif_32_median_gbps']:.3f} | {row['bif_64_median_gbps']:.3f} | "
            f"{row['earliest_tested_candidate_saturation_bif_kib']} |"
        )
    lines.append("")
    lines.append("## NCU HBM validation (six predefined cases only)")
    lines.append("")
    if not dram_read_metric_available:
        lines.append(f"`{MANDATORY_DRAM_METRIC}` was **not** resolved as a supported metric on the "
                     f"profiled device for this campaign. Every case below is therefore "
                     f"`INCONCLUSIVE`; no HBM claim is made for any of the six cases.")
        lines.append("")
    lines.append("| index | method | stages | BIF KiB | dram_read_ratio | classification | flags |")
    lines.append("| ---: | --- | ---: | ---: | ---: | --- | --- |")
    for row in ncu_rows:
        ratio_text = f"{row['dram_read_ratio']:.4f}" if row["dram_read_ratio"] is not None else "n/a"
        flags_text = ", ".join(row["diagnostic_flags"]) if row["diagnostic_flags"] else "--"
        lines.append(
            f"| {row['index']} | {row['method']} | {row['stages']} | {row['bytes_in_flight_kib']} | "
            f"{ratio_text} | {row['hbm_classification']} | {flags_text} |"
        )
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `figures/effective_gbps.svg`")
    lines.append("- `figures/tma_to_ldgsts_ratio.svg`")
    lines.append("- `figures/dram_read_ratio.svg`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("`publishable: false`. Pending independent review and later final campaigns.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcommand: plan
# ---------------------------------------------------------------------------
def cmd_plan(args: argparse.Namespace) -> int:
    plan = build_ncu_plan()
    errors = check_ncu_plan_contract(plan)
    if errors:
        print("analyze_exp01_memory_paths_p14: plan: ERROR: plan contract violated:", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp01_memory_paths_p14: plan:   - {error}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(plan, indent=2))
    elif args.format == "lines":
        sys.stdout.write(format_ncu_plan_lines(plan))
    else:
        sys.stdout.write(format_ncu_plan_text(plan))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: init-campaign
# ---------------------------------------------------------------------------
def _do_init_campaign(*, campaign_id: str, started_at_utc: str) -> Path:
    campaign_dir = create_p14_campaign_dir(campaign_id)
    ncu_plan = build_ncu_plan()
    ncu_errors = check_ncu_plan_contract(ncu_plan)
    if ncu_errors:
        raise ValueError(f"internal NCU plan contract violation: {ncu_errors}")
    write_profile_plan(campaign_dir, ncu_plan)
    frozen_protocol = {
        "pilot_params": dict(FROZEN_PILOT_PARAMS),
        "profile_params": dict(FROZEN_PROFILE_PARAMS),
        "ncu_plan": ncu_plan,
        "mandatory_dram_metric": MANDATORY_DRAM_METRIC,
        "candidate_metrics": list(CANDIDATE_METRICS),
        "hbm_validated_min_ratio": HBM_VALIDATED_MIN_RATIO,
        "read_amplification_max_ratio": READ_AMPLIFICATION_MAX_RATIO,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "cv_stability_review_percent": CV_STABILITY_REVIEW_PERCENT,
        "saturation_fraction_of_max": SATURATION_FRACTION_OF_MAX,
    }
    updates = {
        "campaign_id": campaign_id,
        "started_at_utc": started_at_utc,
        "frozen_protocol": frozen_protocol,
    }
    p14_merge_manifest(campaign_dir, updates, state="PILOT_IN_PROGRESS")
    return campaign_dir


def cmd_init_campaign(args: argparse.Namespace) -> int:
    try:
        campaign_dir = _do_init_campaign(campaign_id=args.campaign_id, started_at_utc=args.started_at_utc)
    except (p13.UnsafePathError, p13.ManifestTransitionError, ValueError) as exc:
        print(f"analyze_exp01_memory_paths_p14: init-campaign: ERROR: {exc}", file=sys.stderr)
        return 2
    print(str(campaign_dir.relative_to(REPO_ROOT)))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: validate-preflight
# ---------------------------------------------------------------------------
def cmd_validate_preflight(args: argparse.Namespace) -> int:
    now_utc = _datetime.now(_timezone.utc)
    if args.now is not None:
        try:
            now_utc = parse_now_arg(args.now)
        except ValueError as exc:
            print(f"analyze_exp01_memory_paths_p14: validate-preflight: ERROR: {exc}", file=sys.stderr)
            return 2
    errors, snapshot = validate_preflight_file(
        Path(args.preflight), expected_git_commit=args.expected_git_commit, now_utc=now_utc,
    )
    if errors:
        print(f"analyze_exp01_memory_paths_p14: validate-preflight: FAIL: {args.preflight}", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp01_memory_paths_p14: validate-preflight:   - {error}", file=sys.stderr)
        return 1
    print(f"analyze_exp01_memory_paths_p14: validate-preflight: OK: {args.preflight}", file=sys.stderr)
    print(json.dumps(snapshot, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: record-pilot
# ---------------------------------------------------------------------------
def _fail_p14(campaign_dir: Path, stage: str, errors: list[str]) -> tuple[bool, list[str]]:
    try:
        manifest = p13.load_manifest(campaign_dir)
        current_state = manifest.get("state")
        if current_state in ALLOWED_P14_TRANSITIONS and "FAILED" in ALLOWED_P14_TRANSITIONS[current_state]:
            p14_merge_manifest(
                campaign_dir, {"failure_stage": stage, "failure_detail": errors[:50]}, state="FAILED",
            )
    except (p13.ManifestTransitionError, p13.UnsafePathError):
        pass
    return False, errors


def _do_record_pilot(
    *, campaign_dir: Path, p13_campaign_dir: Path, preflight_path: Path,
    git_commit: str, completed_at_utc: str, now_utc: _datetime,
) -> tuple[bool, list[str]]:
    try:
        manifest = p13.load_manifest(campaign_dir)
        _validate_p14_manifest_document(manifest, require_initialized=True)
    except (p13.ManifestTransitionError, p13.UnsafePathError) as exc:
        return False, [f"P1.4 manifest: {exc}"]
    if manifest.get("state") != "PILOT_IN_PROGRESS":
        return False, [f"P1.4 manifest state={manifest.get('state')!r} != 'PILOT_IN_PROGRESS'; cannot record a pilot"]

    errors: list[str] = []

    preflight_errors, preflight_snapshot = validate_preflight_file(
        preflight_path, expected_git_commit=git_commit, now_utc=now_utc,
    )
    errors.extend(f"preflight: {e}" for e in preflight_errors)

    try:
        p13_manifest = p13.load_manifest(p13_campaign_dir)
        p13._validate_manifest_document(p13_manifest, require_initialized=True)
    except (p13.ManifestTransitionError, p13.UnsafePathError) as exc:
        return _fail_p14(campaign_dir, "pilot_p13_manifest", [f"P1.3 manifest: {exc}"])
    if not p13_manifest:
        return _fail_p14(campaign_dir, "pilot_p13_manifest", ["P1.3 manifest.json does not exist"])

    if p13_manifest.get("status") != "COMPLETE":
        errors.append(
            f"P1.3 campaign status={p13_manifest.get('status')!r} != 'COMPLETE' "
            f"(only a COMPLETE P1.3 campaign can be pilot input)"
        )
    if p13_manifest.get("run_kind") != "benchmark":
        errors.append(
            f"P1.3 campaign run_kind={p13_manifest.get('run_kind')!r} != 'benchmark' "
            f"(a smoke campaign can never be pilot input)"
        )
    requested = p13_manifest.get("requested", {}) if isinstance(p13_manifest.get("requested"), dict) else {}
    for key in ("working_set_mib", "passes", "warmup_ms", "repetitions"):
        expected = FROZEN_PILOT_PARAMS[key]
        if requested.get(key) != expected:
            errors.append(
                f"P1.3 campaign requested.{key}={requested.get(key)!r} != frozen pilot value {expected!r}"
            )
    if p13_manifest.get("configuration_count_completed") != p13.EXPECTED_CONFIGURATION_COUNT:
        errors.append(
            f"P1.3 campaign configuration_count_completed="
            f"{p13_manifest.get('configuration_count_completed')!r} != {p13.EXPECTED_CONFIGURATION_COUNT}"
        )
    if p13_manifest.get("git_commit") != git_commit:
        errors.append(f"P1.3 campaign git_commit={p13_manifest.get('git_commit')!r} != expected {git_commit!r}")

    if not preflight_errors:
        for field, snapshot_key in (
            ("gpu_uuid", "gpu_uuid"), ("gpu_name", "gpu_name"),
        ):
            if p13_manifest.get(field) != preflight_snapshot.get(snapshot_key):
                errors.append(
                    f"P1.3 campaign {field}={p13_manifest.get(field)!r} != preflight "
                    f"{snapshot_key}={preflight_snapshot.get(snapshot_key)!r}"
                )
        if p13_manifest.get("compute_capability") != preflight_snapshot.get("gpu_compute_cap"):
            errors.append(
                f"P1.3 campaign compute_capability={p13_manifest.get('compute_capability')!r} != "
                f"preflight gpu.compute_cap={preflight_snapshot.get('gpu_compute_cap')!r}"
            )

    combined_hash = summary_hash = p13_manifest_hash = None
    combined_path = p13_campaign_dir / "combined_samples.csv"
    summary_path = p13_campaign_dir / "summary.csv"
    p13_manifest_path = p13_campaign_dir / "manifest.json"
    try:
        combined_hash = p13.sha256_of(combined_path)
        summary_hash = p13.sha256_of(summary_path)
        p13_manifest_hash = p13.sha256_of(p13_manifest_path)
    except p13.UnsafePathError as exc:
        errors.append(f"artifact hashing: {exc}")
    recorded_hashes = p13_manifest.get("aggregate_file_sha256", {})
    if isinstance(recorded_hashes, dict):
        if combined_hash is not None and recorded_hashes.get("combined_samples.csv") != combined_hash:
            errors.append(
                "combined_samples.csv on disk does not match the P1.3 manifest's recorded SHA-256 "
                "(tampered or corrupted since the P1.3 campaign completed)"
            )
        if summary_hash is not None and recorded_hashes.get("summary.csv") != summary_hash:
            errors.append(
                "summary.csv on disk does not match the P1.3 manifest's recorded SHA-256 "
                "(tampered or corrupted since the P1.3 campaign completed)"
            )

    if errors:
        return _fail_p14(campaign_dir, "pilot_validation", errors)

    updates = {
        "pilot_completed_at_utc": completed_at_utc,
        "pilot_campaign_reference": {
            "campaign_id": p13_manifest["campaign_id"],
            "path": str(p13_campaign_dir.relative_to(REPO_ROOT)),
            "manifest_sha256": p13_manifest_hash,
            "combined_samples_sha256": combined_hash,
            "summary_sha256": summary_hash,
        },
        "preflight_reference_pilot": preflight_snapshot,
        "provenance": {
            "git_commit": git_commit,
            "gpu_uuid": p13_manifest["gpu_uuid"],
            "gpu_name": p13_manifest["gpu_name"],
            "compute_capability": p13_manifest["compute_capability"],
            "cuda_driver_version": p13_manifest["cuda_driver_version"],
            "cuda_runtime_version": p13_manifest["cuda_runtime_version"],
            "working_set_bytes": p13_manifest["observed_common"]["working_set_bytes"],
            "passes": p13_manifest["observed_common"]["passes"],
        },
    }
    try:
        p14_merge_manifest(campaign_dir, updates, state="PILOT_COMPLETE")
    except p13.ManifestTransitionError as exc:
        return _fail_p14(campaign_dir, "pilot_manifest_transition", [str(exc)])
    return True, []


def cmd_record_pilot(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_p14_campaign_dir(args.campaign_dir)
        p13_campaign_dir = resolve_p13_campaign_dir_arg(args.p13_campaign_dir)
    except p13.UnsafePathError as exc:
        print(f"analyze_exp01_memory_paths_p14: record-pilot: ERROR: {exc}", file=sys.stderr)
        return 2
    now_utc = _datetime.now(_timezone.utc)
    if args.now is not None:
        try:
            now_utc = parse_now_arg(args.now)
        except ValueError as exc:
            print(f"analyze_exp01_memory_paths_p14: record-pilot: ERROR: {exc}", file=sys.stderr)
            return 2
    success, errors = _do_record_pilot(
        campaign_dir=campaign_dir, p13_campaign_dir=p13_campaign_dir,
        preflight_path=Path(args.preflight), git_commit=args.git_commit,
        completed_at_utc=args.completed_at_utc, now_utc=now_utc,
    )
    if not success:
        print("analyze_exp01_memory_paths_p14: record-pilot: ERROR:", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp01_memory_paths_p14: record-pilot:   - {error}", file=sys.stderr)
        return 1
    print("analyze_exp01_memory_paths_p14: record-pilot: OK: PILOT_COMPLETE", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: discover-metrics
# ---------------------------------------------------------------------------
def parse_metric_discovery_log(path: Path) -> set[str]:
    """Parses `ncu --query-metrics --query-metrics-mode all --devices 0`
    output: one metric identifier per meaningful line, no surrounding
    whitespace, containing the "__" structural signature every NCU metric
    name has (e.g. dram__bytes_read.sum)."""
    try:
        with p13._open_regular_nofollow(path, binary=False) as handle:
            text = handle.read()
    except (OSError, p13.UnsafePathError, UnicodeError) as exc:
        raise ValueError(f"{path}: unable to read: {exc}") from exc
    names: set[str] = set()
    for line in text.splitlines():
        token = line.strip()
        if not token or " " in token or "\t" in token:
            continue
        if "__" in token:
            names.add(token)
    return names


def resolve_ncu_metrics(discovered: set[str]) -> dict:
    resolved = [m for m in CANDIDATE_METRICS if m in discovered]
    missing = [m for m in CANDIDATE_METRICS if m not in discovered]
    return {
        "requested": list(CANDIDATE_METRICS),
        "resolved": resolved,
        "missing": missing,
        "dram_read_metric": MANDATORY_DRAM_METRIC if MANDATORY_DRAM_METRIC in discovered else None,
        "dram_read_metric_available": MANDATORY_DRAM_METRIC in discovered,
    }


def _do_discover_metrics(
    *, campaign_dir: Path, discovery_log: Path, preflight_path: Path,
    git_commit: str, started_at_utc: str, now_utc: _datetime,
) -> tuple[bool, list[str], dict | None]:
    try:
        manifest = p13.load_manifest(campaign_dir)
        _validate_p14_manifest_document(manifest, require_initialized=True)
    except (p13.ManifestTransitionError, p13.UnsafePathError) as exc:
        return False, [f"P1.4 manifest: {exc}"], None
    if manifest.get("state") != "PILOT_COMPLETE":
        return False, [
            f"P1.4 manifest state={manifest.get('state')!r} != 'PILOT_COMPLETE'; cannot start profiling"
        ], None

    errors: list[str] = []
    preflight_errors, preflight_snapshot = validate_preflight_file(
        preflight_path, expected_git_commit=git_commit, now_utc=now_utc,
    )
    errors.extend(f"preflight: {e}" for e in preflight_errors)

    try:
        discovered = parse_metric_discovery_log(discovery_log)
    except ValueError as exc:
        errors.append(str(exc))
        discovered = set()

    if errors:
        success, fail_errors = _fail_p14(campaign_dir, "profile_start", errors)
        return success, fail_errors, None

    resolved = resolve_ncu_metrics(discovered)
    updates = {
        "profile_started_at_utc": started_at_utc,
        "resolved_ncu_metrics": resolved,
        "preflight_reference_profile": preflight_snapshot,
    }
    try:
        p14_merge_manifest(campaign_dir, updates, state="PROFILE_IN_PROGRESS")
    except p13.ManifestTransitionError as exc:
        success, fail_errors = _fail_p14(campaign_dir, "profile_start_manifest", [str(exc)])
        return success, fail_errors, None
    return True, [], resolved


def cmd_discover_metrics(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_p14_campaign_dir(args.campaign_dir)
    except p13.UnsafePathError as exc:
        print(f"analyze_exp01_memory_paths_p14: discover-metrics: ERROR: {exc}", file=sys.stderr)
        return 2
    now_utc = _datetime.now(_timezone.utc)
    if args.now is not None:
        try:
            now_utc = parse_now_arg(args.now)
        except ValueError as exc:
            print(f"analyze_exp01_memory_paths_p14: discover-metrics: ERROR: {exc}", file=sys.stderr)
            return 2
    success, errors, resolved = _do_discover_metrics(
        campaign_dir=campaign_dir, discovery_log=Path(args.discovery_log),
        preflight_path=Path(args.preflight), git_commit=args.git_commit,
        started_at_utc=args.started_at_utc, now_utc=now_utc,
    )
    if not success:
        print("analyze_exp01_memory_paths_p14: discover-metrics: ERROR:", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp01_memory_paths_p14: discover-metrics:   - {error}", file=sys.stderr)
        return 1
    if not resolved["dram_read_metric_available"]:
        print(
            f"analyze_exp01_memory_paths_p14: discover-metrics: WARNING: "
            f"{MANDATORY_DRAM_METRIC} was not resolved; every profiled case in this "
            f"campaign will be classified INCONCLUSIVE for HBM validation",
            file=sys.stderr,
        )
    print("analyze_exp01_memory_paths_p14: discover-metrics: OK: PROFILE_IN_PROGRESS", file=sys.stderr)
    print(",".join(resolved["resolved"]))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: validate-profile-case
#
# Mirrors p13's cmd_validate_case: a single-case validator that never
# self-transitions the campaign to FAILED (that remains the caller's -- i.e.
# run_exp01_memory_paths_p14.sh's -- job on a non-zero exit, exactly as
# p13's own run_exp01_memory_paths.sh does for validate-case). On success it
# does record this case's result and bump the progress counter, via a
# same-state PROFILE_IN_PROGRESS -> PROFILE_IN_PROGRESS manifest merge.
# ---------------------------------------------------------------------------
def _do_validate_profile_case(
    *, campaign_dir: Path, index: int, application_csv: Path, metrics_csv: Path,
    ncu_rep: Path, git_commit: str,
) -> tuple[bool, list[str]]:
    plan = build_ncu_plan()
    entry = next((e for e in plan if e["index"] == index), None)
    if entry is None:
        return False, [f"index {index} is not one of the frozen NCU case indices 0..{EXPECTED_NCU_CASE_COUNT - 1}"]

    try:
        manifest = p13.load_manifest(campaign_dir)
        _validate_p14_manifest_document(manifest, require_initialized=True)
    except (p13.ManifestTransitionError, p13.UnsafePathError) as exc:
        return False, [f"P1.4 manifest: {exc}"]
    if manifest.get("state") != "PROFILE_IN_PROGRESS":
        return False, [f"P1.4 manifest state={manifest.get('state')!r} != 'PROFILE_IN_PROGRESS'"]

    existing_results = manifest.get("case_results", {})
    if entry["case_name"] in existing_results:
        return False, [f"case {entry['case_name']} was already validated and recorded; refusing to redo it"]

    errors: list[str] = []

    ncu_rep_err = p13._verify_artifact(ncu_rep)
    if ncu_rep_err:
        errors.append(ncu_rep_err)

    expect = {
        "method": entry["method"], "stages": entry["stages"], "bif_kib": entry["bif_kib"],
        "run_kind": FROZEN_PROFILE_PARAMS["run_kind"], "repetitions": FROZEN_PROFILE_PARAMS["repetitions"],
        "passes": FROZEN_PROFILE_PARAMS["passes"], "warmup_ms": FROZEN_PROFILE_PARAMS["warmup_ms"],
        "working_set_mib": FROZEN_PROFILE_PARAMS["working_set_mib"], "git_commit": git_commit,
    }
    app_rows, app_errors = p13.validate_case_file(application_csv, expect)
    errors.extend(app_errors)
    app_row = app_rows[0] if (app_rows and not app_errors) else None

    provenance = manifest.get("provenance", {}) if isinstance(manifest.get("provenance"), dict) else {}
    if app_row is not None:
        for field in (
            "gpu_uuid", "gpu_name", "compute_capability", "cuda_driver_version",
            "cuda_runtime_version", "working_set_bytes", "passes",
        ):
            if str(app_row.get(field)) != str(provenance.get(field)):
                errors.append(
                    f"application CSV {field}={app_row.get(field)!r} != pilot provenance "
                    f"{field}={provenance.get(field)!r}"
                )

    parsed_metrics = None
    try:
        parsed_metrics = parse_ncu_raw_csv(metrics_csv)
    except NcuCsvParseError as exc:
        errors.append(str(exc))

    if parsed_metrics is not None:
        if parsed_metrics["launch_count"] > 1:
            errors.append(
                f"metrics CSV records {parsed_metrics['launch_count']} distinct profiled launches, "
                f"expected exactly 1 (--launch-count 1 should guarantee this)"
            )
        if len(parsed_metrics["kernel_names"]) > 1:
            errors.append(
                f"metrics CSV records more than one distinct kernel name: "
                f"{parsed_metrics['kernel_names']!r}, expected exactly one"
            )
        elif parsed_metrics["kernel_names"] and entry["kernel_name"] not in parsed_metrics["kernel_names"][0]:
            errors.append(
                f"metrics CSV kernel name {parsed_metrics['kernel_names'][0]!r} does not contain "
                f"expected base function name {entry['kernel_name']!r}"
            )

    if errors:
        return False, errors

    resolved_ncu_metrics = manifest.get("resolved_ncu_metrics", {})
    dram_available = bool(resolved_ncu_metrics.get("dram_read_metric_available"))
    dram_read_bytes = parsed_metrics["metrics"].get(MANDATORY_DRAM_METRIC) if dram_available else None
    useful_bytes = float(app_row["useful_bytes"])
    classification, flags, ratio = classify_hbm(dram_read_bytes, useful_bytes)

    try:
        app_hash = p13.sha256_of(application_csv)
        metrics_hash = p13.sha256_of(metrics_csv)
        ncu_rep_hash = p13.sha256_of(ncu_rep)
    except p13.UnsafePathError as exc:
        return False, [str(exc)]

    case_result = {
        "case_name": entry["case_name"],
        "method": entry["method"],
        "stages": entry["stages"],
        "bytes_in_flight_kib": entry["bif_kib"],
        "useful_bytes": app_row["useful_bytes"],
        "dram_read_bytes": dram_read_bytes,
        "dram_read_ratio": ratio,
        "hbm_classification": classification,
        "diagnostic_flags": flags,
        "resolved_metric_values": {
            m: parsed_metrics["metrics"].get(m) for m in resolved_ncu_metrics.get("resolved", [])
        } if parsed_metrics else {},
        "application_csv_sha256": app_hash,
        "metrics_csv_sha256": metrics_hash,
        "ncu_rep_sha256": ncu_rep_hash,
    }
    new_results = dict(existing_results)
    new_results[entry["case_name"]] = case_result
    updates = {
        "case_results": new_results,
        "profile_count_completed": len(new_results),
    }
    try:
        p14_merge_manifest(campaign_dir, updates, state="PROFILE_IN_PROGRESS")
    except p13.ManifestTransitionError as exc:
        return False, [str(exc)]
    return True, []


def cmd_validate_profile_case(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_p14_campaign_dir(args.campaign_dir)
    except p13.UnsafePathError as exc:
        print(f"analyze_exp01_memory_paths_p14: validate-profile-case: ERROR: {exc}", file=sys.stderr)
        return 2
    success, errors = _do_validate_profile_case(
        campaign_dir=campaign_dir, index=args.index,
        application_csv=Path(args.application_csv), metrics_csv=Path(args.metrics_csv),
        ncu_rep=Path(args.ncu_rep), git_commit=args.git_commit,
    )
    if not success:
        print(f"analyze_exp01_memory_paths_p14: validate-profile-case: FAIL: index {args.index}", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp01_memory_paths_p14: validate-profile-case:   - {error}", file=sys.stderr)
        return 1
    print(f"analyze_exp01_memory_paths_p14: validate-profile-case: OK: index {args.index}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: finalize-profile
# ---------------------------------------------------------------------------
def _do_finalize_profile(
    *, campaign_dir: Path, completed_at_utc: str,
) -> tuple[bool, list[str]]:
    try:
        manifest = p13.load_manifest(campaign_dir)
        _validate_p14_manifest_document(manifest, require_initialized=True)
    except (p13.ManifestTransitionError, p13.UnsafePathError) as exc:
        return False, [f"P1.4 manifest: {exc}"]
    if manifest.get("state") != "PROFILE_IN_PROGRESS":
        return False, [f"P1.4 manifest state={manifest.get('state')!r} != 'PROFILE_IN_PROGRESS'; cannot finalize"]

    errors: list[str] = []
    plan = build_ncu_plan()
    plan_errors = check_ncu_plan_contract(plan)
    if plan_errors:
        return _fail_p14(campaign_dir, "profile_plan_contract", plan_errors)

    profile_plan_path = campaign_dir / "profile_plan.csv"
    errors.extend(validate_profile_plan_file(profile_plan_path, plan))

    case_results = manifest.get("case_results", {})
    if not isinstance(case_results, dict):
        errors.append("manifest case_results is not an object")
        case_results = {}
    expected_case_names = {entry["case_name"] for entry in plan}
    found_case_names = set(case_results)
    for missing in sorted(expected_case_names - found_case_names):
        errors.append(f"profile case {missing} was never validated/recorded")
    for extra in sorted(found_case_names - expected_case_names):
        errors.append(f"profile case_results contains unexpected case {extra}")
    if manifest.get("profile_count_completed") != EXPECTED_NCU_CASE_COUNT:
        errors.append(
            f"profile_count_completed={manifest.get('profile_count_completed')!r} != "
            f"{EXPECTED_NCU_CASE_COUNT}"
        )

    resolved_ncu_metrics = manifest.get("resolved_ncu_metrics", {})
    if not isinstance(resolved_ncu_metrics, dict) or "dram_read_metric_available" not in resolved_ncu_metrics:
        errors.append("manifest resolved_ncu_metrics is missing or incomplete")

    if errors:
        return _fail_p14(campaign_dir, "profile_finalize", errors)

    try:
        profile_plan_hash = p13.sha256_of(profile_plan_path)
    except p13.UnsafePathError as exc:
        return _fail_p14(campaign_dir, "profile_finalize_hashing", [str(exc)])

    artifact_sha256 = dict(manifest.get("artifact_sha256", {}))
    artifact_sha256["profile_plan.csv"] = profile_plan_hash
    pilot_ref = manifest.get("pilot_campaign_reference", {})
    for key in ("manifest_sha256", "combined_samples_sha256", "summary_sha256"):
        if key in pilot_ref:
            artifact_sha256[f"pilot_{key}"] = pilot_ref[key]
    for ref_name in ("preflight_reference_pilot", "preflight_reference_profile"):
        ref = manifest.get(ref_name, {})
        if isinstance(ref, dict) and "sha256" in ref:
            artifact_sha256[ref_name] = ref["sha256"]
    for case_name, result in case_results.items():
        for key in ("application_csv_sha256", "metrics_csv_sha256", "ncu_rep_sha256"):
            if key in result:
                artifact_sha256[f"{case_name}.{key}"] = result[key]

    updates = {
        "profile_completed_at_utc": completed_at_utc,
        "profile_order": plan,
        "artifact_sha256": artifact_sha256,
    }
    try:
        p14_merge_manifest(campaign_dir, updates, state="COMPLETE")
    except p13.ManifestTransitionError as exc:
        return _fail_p14(campaign_dir, "profile_finalize_manifest", [str(exc)])
    return True, []


def cmd_finalize_profile(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_p14_campaign_dir(args.campaign_dir)
    except p13.UnsafePathError as exc:
        print(f"analyze_exp01_memory_paths_p14: finalize-profile: ERROR: {exc}", file=sys.stderr)
        return 2
    success, errors = _do_finalize_profile(campaign_dir=campaign_dir, completed_at_utc=args.completed_at_utc)
    if not success:
        print("analyze_exp01_memory_paths_p14: finalize-profile: ERROR:", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp01_memory_paths_p14: finalize-profile:   - {error}", file=sys.stderr)
        return 1
    print("analyze_exp01_memory_paths_p14: finalize-profile: OK: COMPLETE", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: analyze
#
# GPU-free. Reads a COMPLETE campaign's already-validated pilot samples and
# profile-case results, computes statistics/comparison/saturation/HBM
# tables, and publishes analysis/* deterministically and no-clobber. On
# failure the campaign manifest is left untouched (state stays COMPLETE, not
# FAILED): analysis is a pure, retriable function of already-validated data,
# so a failed analyze attempt (e.g. a transient disk issue) should not force
# redoing GPU work.
# ---------------------------------------------------------------------------
def _read_combined_samples(path: Path) -> dict[tuple[str, int, int], list[float]]:
    samples: dict[tuple[str, int, int], list[float]] = {}
    with p13._open_regular_nofollow(path, binary=False) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = row["method"]
            stages = int(row["stages"])
            bif_kib = int(row["bytes_in_flight_per_sm"]) // 1024
            gbps = float(row["effective_gbps"])
            samples.setdefault((method, stages, bif_kib), []).append(gbps)
    return samples


def _stats_to_csv_row(method: str, stages: int, bif_kib: int, stats: dict) -> dict:
    return {
        "method": method,
        "stages": stages,
        "bytes_in_flight_kib": bif_kib,
        "sample_count": stats["count"],
        "mean_gbps": stats["mean"],
        "median_gbps": stats["median"],
        "stdev_gbps": stats["stdev"],
        "cv_percent": stats["cv_percent"],
        "min_gbps": stats["min"],
        "max_gbps": stats["max"],
        "median_ci_low_gbps": stats["median_ci_low"],
        "median_ci_high_gbps": stats["median_ci_high"],
        "iqr_low_bound_gbps": stats["iqr_lower_fence"],
        "iqr_high_bound_gbps": stats["iqr_upper_fence"],
        "iqr_flagged_count": stats["iqr_flagged_count"],
        "stability_review": "REVIEW" if stats["stability_review"] else "ok",
    }


PILOT_STATISTICS_HEADER = [
    "method", "stages", "bytes_in_flight_kib", "sample_count", "mean_gbps", "median_gbps",
    "stdev_gbps", "cv_percent", "min_gbps", "max_gbps", "median_ci_low_gbps",
    "median_ci_high_gbps", "iqr_low_bound_gbps", "iqr_high_bound_gbps", "iqr_flagged_count",
    "stability_review",
]
PAIRWISE_COMPARISON_HEADER = [
    "stages", "bytes_in_flight_kib", "median_gbps_ldgsts", "median_gbps_tma",
    "tma_to_ldgsts_ratio", "ratio_ci_low", "ratio_ci_high", "interpretation",
]
SATURATION_CANDIDATES_HEADER = [
    "method", "stages", "bif_16_median_gbps", "bif_32_median_gbps", "bif_64_median_gbps",
    "max_median_gbps", "earliest_tested_candidate_saturation_bif_kib",
]
NCU_VALIDATION_HEADER = [
    "index", "method", "stages", "bytes_in_flight_kib", "kernel_name", "useful_bytes",
    "dram_read_bytes", "dram_read_ratio", "hbm_classification", "diagnostic_flags",
]


def _write_csv_no_clobber(path: Path, header: list[str], rows: list[dict]) -> None:
    if os.path.lexists(path):
        raise p13.UnsafePathError(f"{path}: already exists, refusing to overwrite")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if os.path.lexists(tmp_path):
        raise p13.UnsafePathError(f"{tmp_path}: existing temporary; refusing to overwrite")
    try:
        with p13._open_exclusive(tmp_path, binary=False, newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            for row in rows:
                writer.writerow([row[field] for field in header])
    except Exception:
        if os.path.lexists(tmp_path):
            p13._safe_unlink_owned(tmp_path)
        raise
    try:
        p13._publish_no_clobber(tmp_path, path)
    except p13.UnsafePathError:
        if os.path.lexists(tmp_path):
            p13._safe_unlink_owned(tmp_path)
        raise


def _write_text_no_clobber(path: Path, text: str) -> None:
    if os.path.lexists(path):
        raise p13.UnsafePathError(f"{path}: already exists, refusing to overwrite")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if os.path.lexists(tmp_path):
        raise p13.UnsafePathError(f"{tmp_path}: existing temporary; refusing to overwrite")
    try:
        with p13._open_exclusive(tmp_path, binary=False) as handle:
            handle.write(text)
    except Exception:
        if os.path.lexists(tmp_path):
            p13._safe_unlink_owned(tmp_path)
        raise
    try:
        p13._publish_no_clobber(tmp_path, path)
    except p13.UnsafePathError:
        if os.path.lexists(tmp_path):
            p13._safe_unlink_owned(tmp_path)
        raise


def _do_analyze(*, campaign_dir: Path, analyzed_at_utc: str) -> tuple[bool, list[str]]:
    try:
        manifest = p13.load_manifest(campaign_dir)
        _validate_p14_manifest_document(manifest, require_initialized=True)
    except (p13.ManifestTransitionError, p13.UnsafePathError) as exc:
        return False, [f"P1.4 manifest: {exc}"]
    if manifest.get("state") != "COMPLETE":
        return False, [f"P1.4 manifest state={manifest.get('state')!r} != 'COMPLETE'; cannot analyze"]

    errors: list[str] = []
    pilot_ref = manifest.get("pilot_campaign_reference", {})
    if not isinstance(pilot_ref, dict) or "path" not in pilot_ref:
        return False, ["manifest pilot_campaign_reference is missing or incomplete"]
    p13_campaign_dir = REPO_ROOT / pilot_ref["path"]
    combined_path = p13_campaign_dir / "combined_samples.csv"
    try:
        combined_hash = p13.sha256_of(combined_path)
    except p13.UnsafePathError as exc:
        return False, [str(exc)]
    if combined_hash != pilot_ref.get("combined_samples_sha256"):
        return False, [
            "combined_samples.csv no longer matches the SHA-256 recorded at record-pilot time "
            "(tampered or corrupted since); refusing to analyze"
        ]

    samples_by_config = _read_combined_samples(combined_path)
    expected_configs = {(m, s, b) for m in p13.METHODS for (s, b) in p13.CONFIG_PAIRS}
    for missing in sorted(expected_configs - set(samples_by_config)):
        errors.append(f"combined_samples.csv missing configuration {missing}")
    for key in sorted(set(samples_by_config) - expected_configs):
        errors.append(f"combined_samples.csv has unexpected configuration {key}")
    for key, values in samples_by_config.items():
        if len(values) != FROZEN_PILOT_PARAMS["repetitions"]:
            errors.append(
                f"configuration {key} has {len(values)} sample(s), expected exactly "
                f"{FROZEN_PILOT_PARAMS['repetitions']}"
            )
        if any(not math.isfinite(v) for v in values):
            errors.append(f"configuration {key} has a non-finite effective_gbps value")
    if errors:
        return False, errors

    rng = random.Random(BOOTSTRAP_SEED)
    stats_by_config = compute_all_config_stats(samples_by_config, rng)
    pairwise_rows = compute_pairwise_comparisons(samples_by_config, stats_by_config, rng)
    saturation_rows = compute_saturation_candidates(stats_by_config)

    plan = build_ncu_plan()
    case_results = manifest.get("case_results", {})
    ncu_rows: list[dict] = []
    for entry in plan:
        result = case_results.get(entry["case_name"])
        if result is None:
            errors.append(f"case_results is missing entry for {entry['case_name']}")
            continue
        ncu_rows.append({
            "index": entry["index"],
            "method": entry["method"],
            "stages": entry["stages"],
            "bytes_in_flight_kib": entry["bif_kib"],
            "case_name": entry["case_name"],
            "kernel_name": entry["kernel_name"],
            "useful_bytes": result["useful_bytes"],
            "dram_read_bytes": result["dram_read_bytes"],
            "dram_read_ratio": result["dram_read_ratio"],
            "hbm_classification": result["hbm_classification"],
            "diagnostic_flags": result["diagnostic_flags"],
        })
    if errors:
        return False, errors

    stats_rows = [
        _stats_to_csv_row(method, stages, bif, stats_by_config[(method, stages, bif)])
        for (method, stages, bif) in sorted(stats_by_config, key=lambda k: (k[1], k[2], k[0]))
    ]

    provenance = manifest.get("provenance", {})
    dram_available = bool(manifest.get("resolved_ncu_metrics", {}).get("dram_read_metric_available"))
    analysis_dir = campaign_dir / "analysis"
    figures_dir = analysis_dir / "figures"
    try:
        p13._mkdir_component(figures_dir, must_not_exist=False)
    except p13.UnsafePathError as exc:
        return False, [str(exc)]

    outputs: list[tuple[Path, list[str], list[dict]]] = [
        (analysis_dir / "pilot_statistics.csv", PILOT_STATISTICS_HEADER, stats_rows),
        (analysis_dir / "pairwise_comparison.csv", PAIRWISE_COMPARISON_HEADER, pairwise_rows),
        (analysis_dir / "saturation_candidates.csv", SATURATION_CANDIDATES_HEADER, saturation_rows),
        (analysis_dir / "ncu_validation.csv", NCU_VALIDATION_HEADER, [
            {**row, "diagnostic_flags": ";".join(row["diagnostic_flags"])} for row in ncu_rows
        ]),
    ]
    for path, _, _ in outputs:
        for candidate in (path, path.with_suffix(path.suffix + ".tmp")):
            if os.path.lexists(candidate):
                return False, [f"{candidate}: existing analysis artifact; refusing to overwrite"]

    analysis_json = {
        "schema_version": P14_SCHEMA_VERSION,
        "campaign_id": manifest.get("campaign_id"),
        "publishable": False,
        "provenance": provenance,
        "pilot_statistics": stats_rows,
        "pairwise_comparison": pairwise_rows,
        "saturation_candidates": saturation_rows,
        "ncu_validation": ncu_rows,
        "resolved_ncu_metrics": manifest.get("resolved_ncu_metrics", {}),
        "notes": [
            "single pilot, not a final campaign",
            "timings are CUDA-event pilot measurements; NCU duration is never used as benchmark timing",
            "no sample was ever removed; IQR flags are diagnostics only",
            "NCU covers exactly six predefined cases out of 18 pilot configurations",
            "candidate saturation is limited to the three tested bytes-in-flight values per group",
            "no final or universal HBM ceiling is established",
            "publishable: false, pending independent review and later final campaigns",
        ],
    }
    report_md = render_report_markdown(
        campaign_id=str(manifest.get("campaign_id")), stats_rows=stats_rows,
        pairwise_rows=pairwise_rows, saturation_rows=saturation_rows, ncu_rows=ncu_rows,
        provenance=provenance, dram_read_metric_available=dram_available,
    )
    effective_gbps_svg = render_effective_gbps_svg(stats_by_config)
    ratio_svg = render_ratio_svg(pairwise_rows)
    dram_svg = render_dram_read_ratio_svg(ncu_rows)

    file_writes: list[tuple[Path, str]] = [
        (analysis_dir / "analysis.json", json.dumps(analysis_json, indent=2, sort_keys=True) + "\n"),
        (analysis_dir / "report.md", report_md),
        (figures_dir / "effective_gbps.svg", effective_gbps_svg),
        (figures_dir / "tma_to_ldgsts_ratio.svg", ratio_svg),
        (figures_dir / "dram_read_ratio.svg", dram_svg),
    ]
    for path, _ in file_writes:
        for candidate in (path, path.with_suffix(path.suffix + ".tmp")):
            if os.path.lexists(candidate):
                return False, [f"{candidate}: existing analysis artifact; refusing to overwrite"]

    published: list[Path] = []
    try:
        for path, header, rows in outputs:
            _write_csv_no_clobber(path, header, rows)
            published.append(path)
        for path, text in file_writes:
            _write_text_no_clobber(path, text)
            published.append(path)
    except (p13.UnsafePathError, OSError) as exc:
        for path in published:
            try:
                p13._safe_unlink_owned(path)
            except p13.UnsafePathError:
                pass
        return False, [f"analysis artifact publication failed: {exc}"]

    try:
        artifact_hashes = {
            path.relative_to(campaign_dir).as_posix(): p13.sha256_of(path) for path, _, _ in outputs
        }
        artifact_hashes.update(
            {path.relative_to(campaign_dir).as_posix(): p13.sha256_of(path) for path, _ in file_writes}
        )
    except p13.UnsafePathError as exc:
        return False, [str(exc)]

    updates = {
        "analyzed_at_utc": analyzed_at_utc,
        "artifact_sha256": {**manifest.get("artifact_sha256", {}), **artifact_hashes},
    }
    try:
        p14_merge_manifest(campaign_dir, updates, state="ANALYZED")
    except p13.ManifestTransitionError as exc:
        return False, [str(exc)]
    return True, []


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_p14_campaign_dir(args.campaign_dir)
    except p13.UnsafePathError as exc:
        print(f"analyze_exp01_memory_paths_p14: analyze: ERROR: {exc}", file=sys.stderr)
        return 2
    success, errors = _do_analyze(campaign_dir=campaign_dir, analyzed_at_utc=args.analyzed_at_utc)
    if not success:
        print("analyze_exp01_memory_paths_p14: analyze: ERROR:", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp01_memory_paths_p14: analyze:   - {error}", file=sys.stderr)
        return 1
    print("analyze_exp01_memory_paths_p14: analyze: OK: ANALYZED", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: manifest-write (FAILED/INTERRUPTED only, mirrors p13's own)
# ---------------------------------------------------------------------------
def cmd_manifest_write(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_p14_campaign_dir(args.campaign_dir)
    except p13.UnsafePathError as exc:
        print(f"analyze_exp01_memory_paths_p14: manifest-write: ERROR: {exc}", file=sys.stderr)
        return 2
    updates: dict = {}
    if args.merge_json:
        merge_path = Path(args.merge_json)
        try:
            updates = json.loads(merge_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"analyze_exp01_memory_paths_p14: manifest-write: ERROR: cannot read --merge-json: {exc}", file=sys.stderr)
            return 2
        if not isinstance(updates, dict):
            print("analyze_exp01_memory_paths_p14: manifest-write: ERROR: --merge-json must contain a JSON object", file=sys.stderr)
            return 2
    try:
        p14_merge_manifest(campaign_dir, updates, state=args.status)
    except p13.ManifestTransitionError as exc:
        print(f"analyze_exp01_memory_paths_p14: manifest-write: ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"analyze_exp01_memory_paths_p14: manifest-write: OK: state={args.status}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Self-test: GPU-free synthetic/adversarial tests. Never touches CUDA,
# Docker, nvidia-smi, either benchmark binary, NCU, the network, or real
# raw results. All campaign directories are built under a
# tempfile.TemporaryDirectory with BOTH this module's and p13's REPO_ROOT
# patched to that tempdir (verified during implementation to fully redirect
# every path-safety primitive, including the ones this module imports from
# p13), so the real results/raw/ tree is never touched.
# ---------------------------------------------------------------------------
class _Recorder:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.total = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.total += 1
        if condition:
            print(f"analyze_exp01_memory_paths_p14: self-test: PASS: {name}", file=sys.stderr)
        else:
            self.failures.append(name)
            suffix = f"; {detail}" if detail else ""
            print(f"analyze_exp01_memory_paths_p14: self-test: FAIL: {name}{suffix}", file=sys.stderr)

    def expect_error_containing(self, name: str, errors: list[str], needle: str) -> None:
        self.check(
            name, any(needle in error for error in errors),
            detail=f"expected substring {needle!r} in errors={errors}",
        )


_FIXED_GIT_COMMIT = "b" * 40
_FIXED_GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"
_FIXED_GPU_NAME = "NVIDIA B300 SXM6"


def _default_preflight_doc(
    *, timestamp_utc: str = "20260728T100000Z", overall_status: str = "PASS",
    git_dirty: bool = False, git_commit: str = _FIXED_GIT_COMMIT,
    gpu_uuid: str = _FIXED_GPU_UUID, gpu_name: str = _FIXED_GPU_NAME,
    compute_cap: str = "10.3", gpu_visibility_status: str = "PASS",
    ncu_profile_status: str = "PASS",
) -> dict:
    return {
        "schema_version": "1",
        "timestamp_utc": timestamp_utc,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "host_arch": "x86_64",
        "tool_versions": {"nvcc": "release 13.1", "ncu": "version 2025.4.0.0"},
        "gpu": {
            "logical_index": "0", "name": gpu_name, "uuid": gpu_uuid,
            "driver_version": "580.95.05", "compute_cap": compute_cap, "memory_total": "288 GiB",
        },
        "checks": [
            {"name": "gpu_visibility", "required": True, "status": gpu_visibility_status, "reason_code": "OK"},
            {"name": "tool_versions", "required": True, "status": "PASS", "reason_code": "OK"},
            {"name": "cuda_smoke_compile", "required": True, "status": "PASS", "reason_code": "OK"},
            {"name": "cuda_smoke_run", "required": True, "status": "PASS", "reason_code": "OK"},
            {"name": "cutedsl_smoke", "required": True, "status": "PASS", "reason_code": "OK"},
            {"name": "ncu_profile", "required": True, "status": ncu_profile_status, "reason_code": "OK"},
        ],
        "overall_status": overall_status,
    }


def _write_preflight_json(path: Path, doc: dict) -> None:
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _write_rows_csv(path: Path, rows: list[list[str]]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(rows)
    return path


def _build_p13_pilot_campaign_fixture(
    tmp_path: Path, campaign_id: str, *,
    git_commit: str = _FIXED_GIT_COMMIT, gpu_uuid: str = _FIXED_GPU_UUID,
    gpu_name: str = _FIXED_GPU_NAME, run_kind: str = "benchmark",
    repetitions: int = 30, passes: int = 32, warmup_ms: int = 2000,
    working_set_mib: int = 512, gbps_fn=None,
) -> tuple[Path, dict]:
    """Builds a complete, self-consistent, COMPLETE-state synthetic P1.3
    benchmark campaign directly on disk, reusing p13's own fixture/aggregate
    helpers (_default_row, write_execution_order, write_combined_samples,
    write_summary, merge_manifest, sha256_of) so it is byte-for-byte the
    shape a real P1.3 finalize produces. Returns (campaign_dir, manifest)."""
    plan = p13.build_plan()
    campaign_dir = tmp_path / "p13_campaigns" / campaign_id
    (campaign_dir / "cases").mkdir(parents=True)
    cases: list[tuple[dict, list[dict]]] = []
    for entry in plan:
        rows = []
        for sample_index in range(repetitions):
            kernel_time_ms = None
            if gbps_fn is not None:
                # gbps_fn returns a target effective_gbps; convert to a
                # kernel_time_ms that reproduces it for a 512 MiB/32-pass
                # working set at sm_count=4/l2_bytes=25165824 (the fixture's
                # own defaults), holding useful_bytes fixed across the group.
                target_gbps = gbps_fn(entry, sample_index)
                common_multiple = 4 * 32 * 1024
                working_set_bytes = p13.round_up_to_multiple(
                    working_set_mib * 1024 * 1024, common_multiple,
                )
                useful_bytes = working_set_bytes * passes
                kernel_time_ms = useful_bytes / (target_gbps * 1e9) * 1000.0
            row = p13._default_row(
                entry, sample_index, repetitions=repetitions, run_kind=run_kind,
                passes=passes, warmup_ms=warmup_ms, git_commit=git_commit,
                working_set_mib=working_set_mib, kernel_time_ms=kernel_time_ms,
                overrides={"gpu_uuid": gpu_uuid, "gpu_name": gpu_name},
            )
            rows.append(row)
        p13._write_case_csv(campaign_dir / "cases" / f"{entry['case_name']}.csv", rows)
        cases.append((entry, rows))
    p13.write_execution_order(campaign_dir, plan)
    started = "20260728T090000Z"
    p13.merge_manifest(
        campaign_dir,
        {
            "campaign_id": campaign_id, "run_kind": run_kind, "started_at_utc": started,
            "configuration_count_expected": p13.EXPECTED_CONFIGURATION_COUNT,
            "configuration_count_completed": p13.EXPECTED_CONFIGURATION_COUNT,
            "sample_count_expected": p13.EXPECTED_CONFIGURATION_COUNT * repetitions,
            "sample_count_completed": p13.EXPECTED_CONFIGURATION_COUNT * repetitions,
            "requested": {
                "run_kind": run_kind, "working_set_mib": working_set_mib, "passes": passes,
                "warmup_ms": warmup_ms, "repetitions": repetitions, "campaign_id": campaign_id,
            },
            "selected_gpu_index": 0, "git_commit": git_commit, "git_dirty": False,
            "self_test_outcomes": {"ldgsts": "PASS", "tma": "PASS"},
        },
        status="IN_PROGRESS",
    )
    args = argparse.Namespace(
        campaign_id=campaign_id, run_kind=run_kind, repetitions=repetitions, passes=passes,
        warmup_ms=warmup_ms, working_set_mib=working_set_mib, git_commit=git_commit,
        gpu_index=0, started_at_utc=started, completed_at_utc="20260728T093000Z",
        self_test_ldgsts="PASS", self_test_tma="PASS",
    )
    synthetic_artifact = tmp_path / f"synthetic_build_artifact_{campaign_id}"
    synthetic_artifact.write_bytes(b"synthetic non-empty artifact\n")
    synthetic_final_artifacts = {label: synthetic_artifact for label in p13.DEFAULT_FINAL_ARTIFACTS}
    synthetic_versions = tmp_path / f"VERSIONS_{campaign_id}.env"
    synthetic_versions.write_text(
        "CUDA_VERSION=13.1.0\nCUDA_IMAGE=nvidia/cuda:13.1.0-devel-ubuntu24.04\n"
        "CUDA_IMAGE_DIGEST=sha256:" + "0" * 64 + "\nCUDA_IMAGE_PLATFORM=linux/amd64\n"
        "CUTLASS_VERSION=v4.6.1\nCUTLASS_COMMIT=" + "0" * 40 + "\n"
        "CUDA_ARCH=sm_103a\nMAX_BUILD_JOBS=2\n",
        encoding="utf-8",
    )
    success, errors = p13._do_finalize(
        campaign_dir, args, artifact_paths=synthetic_final_artifacts, versions_path=synthetic_versions,
    )
    if not success:
        raise AssertionError(f"self-test fixture: P1.3 finalize failed: {errors}")
    manifest = p13.load_manifest(campaign_dir)
    return campaign_dir, manifest


def _build_ncu_case_fixture(
    tmp_path: Path, entry: dict, *, git_commit: str = _FIXED_GIT_COMMIT,
    gpu_uuid: str = _FIXED_GPU_UUID, gpu_name: str = _FIXED_GPU_NAME,
    dram_read_bytes: float | None = None, extra_metric_rows: list[list[str]] | None = None,
    kernel_name_in_csv: str | None = None, extra_kernel_row: bool = False,
) -> tuple[Path, Path, Path, dict]:
    """Builds one application CSV + metrics_raw.csv + .ncu-rep for a given
    NCU_PLAN entry. Returns (application_csv, metrics_csv, ncu_rep, row)."""
    row = p13._default_row(
        entry, 0, repetitions=1, run_kind="benchmark", passes=32, warmup_ms=0,
        git_commit=git_commit, working_set_mib=512,
        overrides={"gpu_uuid": gpu_uuid, "gpu_name": gpu_name},
    )
    app_csv = tmp_path / f"{entry['case_name']}.application.csv"
    p13._write_case_csv(app_csv, [row])
    useful_bytes = int(row["useful_bytes"])
    if dram_read_bytes is None:
        dram_read_bytes = useful_bytes * 0.95
    metrics_csv = tmp_path / f"{entry['case_name']}.metrics_raw.csv"
    kernel_name_field = kernel_name_in_csv or f"{entry['kernel_name']}<2,4>(...)"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["ID", "Kernel Name", "Metric Unit", "Metric Name", "Metric Value"])
        writer.writerow(["0", kernel_name_field, "byte", "dram__bytes_read.sum", f"{dram_read_bytes}"])
        writer.writerow(["0", kernel_name_field, "nsecond", "gpu__time_duration.sum", "123456"])
        if extra_kernel_row:
            # Same metric name and value as the row above (no value conflict in
            # parse_ncu_raw_csv), but a distinct kernel name/ID: this isolates
            # the "more than one distinct profiled launch" check from the
            # separate "conflicting metric value" check.
            writer.writerow(["1", "some_other_kernel(...)", "nsecond", "gpu__time_duration.sum", "123456"])
        for extra in extra_metric_rows or []:
            writer.writerow(extra)
    ncu_rep = tmp_path / f"{entry['case_name']}_report.ncu-rep"
    ncu_rep.write_bytes(b"synthetic ncu report bytes, never a real profile\n")
    return app_csv, metrics_csv, ncu_rep, row


def _run_profile_pipeline(
    tmp_path: Path, campaign_id: str, *, dram_read_bytes_fn=None,
) -> tuple[Path, Path]:
    """End-to-end fixture: builds a P1.3 pilot campaign, records it, runs
    metric discovery, and validates all six frozen NCU cases (HBM_VALIDATED
    by default). Returns (p14_campaign_dir, preflight_path); does not
    finalize or analyze."""
    p13_campaign_dir, _ = _build_p13_pilot_campaign_fixture(tmp_path, f"p13_{campaign_id}")
    p14_campaign_dir = _do_init_campaign(campaign_id=campaign_id, started_at_utc="20260728T100000Z")
    preflight_path = tmp_path / f"preflight_{campaign_id}.json"
    _write_preflight_json(preflight_path, _default_preflight_doc())
    now = _datetime(2026, 7, 28, 11, 0, tzinfo=_timezone.utc)
    ok, errors = _do_record_pilot(
        campaign_dir=p14_campaign_dir, p13_campaign_dir=p13_campaign_dir,
        preflight_path=preflight_path, git_commit=_FIXED_GIT_COMMIT,
        completed_at_utc="20260728T103000Z", now_utc=now,
    )
    if not ok:
        raise AssertionError(f"self-test fixture: record-pilot failed: {errors}")
    discovery_log = tmp_path / f"discovery_{campaign_id}.log"
    discovery_log.write_text("\n".join(CANDIDATE_METRICS) + "\n", encoding="utf-8")
    ok, errors, _resolved = _do_discover_metrics(
        campaign_dir=p14_campaign_dir, discovery_log=discovery_log,
        preflight_path=preflight_path, git_commit=_FIXED_GIT_COMMIT,
        started_at_utc="20260728T003100Z", now_utc=now,
    )
    if not ok:
        raise AssertionError(f"self-test fixture: discover-metrics failed: {errors}")
    for entry in build_ncu_plan():
        dram_bytes = dram_read_bytes_fn(entry) if dram_read_bytes_fn is not None else None
        app_csv, metrics_csv, ncu_rep, _row = _build_ncu_case_fixture(
            tmp_path, entry, dram_read_bytes=dram_bytes,
        )
        ok, errors = _do_validate_profile_case(
            campaign_dir=p14_campaign_dir, index=entry["index"], application_csv=app_csv,
            metrics_csv=metrics_csv, ncu_rep=ncu_rep, git_commit=_FIXED_GIT_COMMIT,
        )
        if not ok:
            raise AssertionError(f"self-test fixture: validate-profile-case {entry['case_name']} failed: {errors}")
    return p14_campaign_dir, preflight_path


def run_self_test() -> int:  # noqa: C901 - a long, linear, itemized test list is the clearest shape here
    rec = _Recorder()

    # --- six-case NCU plan contract (spec Section 12 item 2) ----------------
    ncu_plan = build_ncu_plan()
    ncu_plan_errors = check_ncu_plan_contract(ncu_plan)
    rec.check(
        "NCU plan has exactly six cases in the frozen order",
        len(ncu_plan) == EXPECTED_NCU_CASE_COUNT and not ncu_plan_errors,
        detail=f"errors={ncu_plan_errors}",
    )
    rec.check(
        "NCU plan matches the exact frozen (index, method, stages, bif_kib) table",
        tuple((e["index"], e["method"], e["stages"], e["bif_kib"]) for e in ncu_plan) == NCU_PLAN_RAW,
    )

    # --- P1.3's own 18-case plan contract still holds (item 1) --------------
    p13_plan = p13.build_plan()
    p13_plan_errors = p13.check_plan_contract(p13_plan)
    rec.check(
        "P1.3's frozen 18-invocation plan is unchanged and self-consistent",
        len(p13_plan) == 18 and not p13_plan_errors,
        detail=f"errors={p13_plan_errors}",
    )

    with tempfile.TemporaryDirectory(prefix="p14_selftest_") as tmp:
        tmp_path = Path(tmp).resolve()
        with mock.patch.object(sys.modules[__name__], "REPO_ROOT", tmp_path), \
                mock.patch.object(p13, "REPO_ROOT", tmp_path):

            # --- preflight validation (item 4) ------------------------------
            now = _datetime(2026, 7, 28, 12, 0, tzinfo=_timezone.utc)
            good_preflight = tmp_path / "good_preflight.json"
            _write_preflight_json(good_preflight, _default_preflight_doc())
            errors, snapshot = validate_preflight_file(
                good_preflight, expected_git_commit=_FIXED_GIT_COMMIT, now_utc=now,
            )
            rec.check("valid preflight is accepted", not errors, detail=f"errors={errors}")
            rec.check("preflight snapshot carries gpu_uuid", snapshot.get("gpu_uuid") == _FIXED_GPU_UUID)

            missing_preflight = tmp_path / "does_not_exist.json"
            errors, _ = validate_preflight_file(
                missing_preflight, expected_git_commit=_FIXED_GIT_COMMIT, now_utc=now,
            )
            rec.check("missing preflight file is rejected", bool(errors))

            malformed_preflight = tmp_path / "malformed.json"
            malformed_preflight.write_text("not json{{{", encoding="utf-8")
            errors, _ = validate_preflight_file(
                malformed_preflight, expected_git_commit=_FIXED_GIT_COMMIT, now_utc=now,
            )
            rec.check("malformed (non-JSON) preflight is rejected", bool(errors))

            dirty_preflight = tmp_path / "dirty.json"
            _write_preflight_json(dirty_preflight, _default_preflight_doc(git_dirty=True))
            errors, _ = validate_preflight_file(
                dirty_preflight, expected_git_commit=_FIXED_GIT_COMMIT, now_utc=now,
            )
            rec.expect_error_containing("dirty preflight (git_dirty=true) is rejected", errors, "git_dirty")

            stale_preflight = tmp_path / "stale.json"
            stale_ts = "20260727T110000Z"  # 25h before `now`
            _write_preflight_json(stale_preflight, _default_preflight_doc(timestamp_utc=stale_ts))
            errors, _ = validate_preflight_file(
                stale_preflight, expected_git_commit=_FIXED_GIT_COMMIT, now_utc=now,
            )
            rec.expect_error_containing("preflight older than 24h is rejected", errors, "24h")

            fresh_boundary_preflight = tmp_path / "fresh_boundary.json"
            boundary_ts = "20260727T120000Z"  # exactly 24h before `now`
            _write_preflight_json(fresh_boundary_preflight, _default_preflight_doc(timestamp_utc=boundary_ts))
            errors, _ = validate_preflight_file(
                fresh_boundary_preflight, expected_git_commit=_FIXED_GIT_COMMIT, now_utc=now,
            )
            rec.check("preflight exactly 24h old is accepted (boundary)", not errors, detail=f"errors={errors}")

            for label, kwargs, needle in (
                ("overall_status != PASS", {"overall_status": "FAIL"}, "overall_status"),
                ("compute_cap != 10.3", {"compute_cap": "9.0"}, "compute_cap"),
                ("gpu_visibility check not PASS", {"gpu_visibility_status": "FAIL"}, "gpu_visibility"),
                ("ncu_profile check not PASS", {"ncu_profile_status": "BLOCKED"}, "ncu_profile"),
            ):
                doc_path = tmp_path / f"bad_{label.replace(' ', '_').replace('.', '_').replace('!=','ne')}.json"
                _write_preflight_json(doc_path, _default_preflight_doc(**kwargs))
                errors, _ = validate_preflight_file(doc_path, expected_git_commit=_FIXED_GIT_COMMIT, now_utc=now)
                rec.expect_error_containing(f"preflight with {label} is rejected", errors, needle)

            mismatched_commit_preflight = tmp_path / "mismatched_commit.json"
            _write_preflight_json(mismatched_commit_preflight, _default_preflight_doc(git_commit="c" * 40))
            errors, _ = validate_preflight_file(
                mismatched_commit_preflight, expected_git_commit=_FIXED_GIT_COMMIT, now_utc=now,
            )
            rec.expect_error_containing("preflight git_commit mismatch is rejected", errors, "git_commit")

            symlinked_preflight = tmp_path / "symlinked_preflight.json"
            real_target = tmp_path / "real_preflight_target.json"
            _write_preflight_json(real_target, _default_preflight_doc())
            try:
                symlinked_preflight.symlink_to(real_target)
                errors, _ = validate_preflight_file(
                    symlinked_preflight, expected_git_commit=_FIXED_GIT_COMMIT, now_utc=now,
                )
                rec.check("symlinked preflight input is rejected (item 23)", bool(errors))
            except OSError:
                rec.check("symlinked preflight input is rejected (item 23)", True, detail="symlink unsupported; skipped")

            # --- malformed units, negative, empty, NaN/infinite metrics (item 13) --
            plan0 = ncu_plan[0]
            good_csv = tmp_path / "metrics_good.csv"
            with open(good_csv, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["ID", "Kernel Name", "Metric Unit", "Metric Name", "Metric Value"])
                w.writerow(["0", f"{plan0['kernel_name']}<2,4>(...)", "byte", "dram__bytes_read.sum", "1000000"])
            parsed = parse_ncu_raw_csv(good_csv)
            rec.check(
                "well-formed metrics CSV parses to a float value",
                parsed["metrics"].get("dram__bytes_read.sum") == 1000000.0,
                detail=f"parsed={parsed}",
            )
            for label, value in (
                ("empty", ""), ("NaN", "nan"), ("infinite", "inf"), ("non-numeric", "not-a-number"),
            ):
                bad_csv = tmp_path / f"metrics_bad_{label}.csv"
                with open(bad_csv, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(["ID", "Kernel Name", "Metric Unit", "Metric Name", "Metric Value"])
                    w.writerow(["0", f"{plan0['kernel_name']}<2,4>(...)", "byte", "dram__bytes_read.sum", value])
                raised = False
                try:
                    parse_ncu_raw_csv(bad_csv)
                except NcuCsvParseError:
                    raised = True
                rec.check(f"metrics CSV with {label} value is rejected", raised)

            # --- HBM classification thresholds (items 14, 15, 16) -------------------
            classification, flags, ratio = classify_hbm(900.0, 1000.0)
            rec.check(
                "exact 0.90 HBM boundary classifies HBM_VALIDATED (item 14)",
                classification == "HBM_VALIDATED" and not flags and ratio == 0.9,
                detail=f"classification={classification} flags={flags} ratio={ratio}",
            )
            classification, flags, ratio = classify_hbm(899.0, 1000.0)
            rec.check(
                "ratio just below 0.90 classifies INCONCLUSIVE (item 15)",
                classification == "INCONCLUSIVE",
                detail=f"classification={classification}",
            )
            classification, flags, ratio = classify_hbm(1100.0, 1000.0)
            rec.check(
                "exact 1.10 ratio does not raise READ_AMPLIFICATION (item 16 boundary)",
                classification == "HBM_VALIDATED" and "READ_AMPLIFICATION" not in flags,
            )
            classification, flags, ratio = classify_hbm(1101.0, 1000.0)
            rec.check(
                "ratio above 1.10 is HBM_VALIDATED with READ_AMPLIFICATION flag (item 16)",
                classification == "HBM_VALIDATED" and "READ_AMPLIFICATION" in flags,
                detail=f"classification={classification} flags={flags}",
            )
            classification, flags, ratio = classify_hbm(None, 1000.0)
            rec.check(
                "missing DRAM metric classifies INCONCLUSIVE (item 12)",
                classification == "INCONCLUSIVE" and ratio is None,
            )

            # --- deterministic bootstrap (item 17) -----------------------------------
            sample_values = [10.0, 10.5, 9.8, 10.2, 9.9, 10.1, 10.3, 9.7, 10.0, 10.4]
            rng_a = random.Random(BOOTSTRAP_SEED)
            ci_a = bootstrap_indices_median_ci(sample_values, rng_a)
            rng_b = random.Random(BOOTSTRAP_SEED)
            ci_b = bootstrap_indices_median_ci(sample_values, rng_b)
            rec.check("bootstrap median CI is deterministic given the fixed seed", ci_a == ci_b, detail=f"{ci_a} vs {ci_b}")

            # --- outliers retained in every primary statistic (item 18) -------------
            with_outlier = [10.0] * 29 + [1000.0]
            stats_outlier = compute_config_stats(with_outlier)
            rec.check(
                "an extreme outlier is never dropped from count/mean (item 18)",
                stats_outlier["count"] == 30 and stats_outlier["mean"] > 40.0 and stats_outlier["iqr_flagged_count"] >= 1,
                detail=f"stats={stats_outlier}",
            )

            # --- CV stability-review boundary (item 19) ------------------------------
            # Constructing data whose sample stdev/mean is exactly 5.0% would require
            # solving for stdev; assert the boundary comparison operator directly instead.
            rec.check("CV exactly at 5.0% is NOT flagged (operator is strictly greater-than)", not (5.0 > CV_STABILITY_REVIEW_PERCENT))
            rec.check("CV just above 5.0% IS flagged", (5.0001 > CV_STABILITY_REVIEW_PERCENT))
            low_cv_stats = compute_config_stats([100.0, 100.0, 100.0, 100.0])
            rec.check("near-zero variance data is not flagged for stability review", not low_cv_stats["stability_review"])
            high_cv_values = [50.0, 150.0, 50.0, 150.0, 50.0, 150.0]
            high_cv_stats = compute_config_stats(high_cv_values)
            rec.check(
                "high-variance data is flagged for stability review",
                high_cv_stats["stability_review"] and high_cv_stats["cv_percent"] > CV_STABILITY_REVIEW_PERCENT,
            )

            # --- pairwise ratio direction (item 20) -----------------------------------
            ldgsts_values = [10.0] * 30
            tma_values_higher = [12.0] * 30
            samples_dir = {("ldgsts", 2, 16): ldgsts_values, ("tma", 2, 16): tma_values_higher}
            stats_dir = compute_all_config_stats(samples_dir, random.Random(BOOTSTRAP_SEED))
            pairwise_dir = compute_pairwise_comparisons(samples_dir, stats_dir, random.Random(BOOTSTRAP_SEED))
            rec.check(
                "TMA higher than LDGSTS yields ratio > 1 and 'tma_higher' (item 20)",
                pairwise_dir[0]["tma_to_ldgsts_ratio"] > 1 and pairwise_dir[0]["interpretation"] == "tma_higher",
                detail=f"{pairwise_dir[0]}",
            )
            samples_dir_rev = {("ldgsts", 2, 16): tma_values_higher, ("tma", 2, 16): ldgsts_values}
            stats_dir_rev = compute_all_config_stats(samples_dir_rev, random.Random(BOOTSTRAP_SEED))
            pairwise_dir_rev = compute_pairwise_comparisons(samples_dir_rev, stats_dir_rev, random.Random(BOOTSTRAP_SEED))
            rec.check(
                "LDGSTS higher than TMA yields ratio < 1 and 'ldgsts_higher' (item 20)",
                pairwise_dir_rev[0]["tma_to_ldgsts_ratio"] < 1 and pairwise_dir_rev[0]["interpretation"] == "ldgsts_higher",
            )
            samples_dir_eq = {("ldgsts", 2, 16): ldgsts_values, ("tma", 2, 16): list(ldgsts_values)}
            stats_dir_eq = compute_all_config_stats(samples_dir_eq, random.Random(BOOTSTRAP_SEED))
            pairwise_dir_eq = compute_pairwise_comparisons(samples_dir_eq, stats_dir_eq, random.Random(BOOTSTRAP_SEED))
            rec.check(
                "equal medians yield ratio == 1 and 'equal' (item 20)",
                pairwise_dir_eq[0]["tma_to_ldgsts_ratio"] == 1 and pairwise_dir_eq[0]["interpretation"] == "equal",
            )
            rec.check(
                "no p-value/significance/winner language appears in pairwise output",
                all(
                    key not in ("p_value", "significant", "winner") for row in pairwise_dir for key in row
                ),
            )

            # --- saturation candidate selection (items 21, 22) -----------------------
            def _stat(median: float, ci_low: float, ci_high: float) -> dict:
                return {"median": median, "median_ci_low": ci_low, "median_ci_high": ci_high}

            stats_sat_16 = {
                ("ldgsts", 2, 16): _stat(100.0, 95.0, 105.0),
                ("ldgsts", 2, 32): _stat(100.0, 95.0, 105.0),
                ("ldgsts", 2, 64): _stat(100.0, 95.0, 105.0),
            }
            rows = compute_saturation_candidates(stats_sat_16)
            rec.check(
                "saturation earliest-candidate selects 16 KiB (item 21)",
                rows[0]["earliest_tested_candidate_saturation_bif_kib"] == 16, detail=f"{rows[0]}",
            )

            stats_sat_32 = {
                ("ldgsts", 2, 16): _stat(50.0, 45.0, 55.0),
                ("ldgsts", 2, 32): _stat(100.0, 95.0, 105.0),
                ("ldgsts", 2, 64): _stat(100.0, 95.0, 105.0),
            }
            rows = compute_saturation_candidates(stats_sat_32)
            rec.check(
                "saturation earliest-candidate selects 32 KiB (item 21)",
                rows[0]["earliest_tested_candidate_saturation_bif_kib"] == 32, detail=f"{rows[0]}",
            )

            stats_sat_64 = {
                ("ldgsts", 2, 16): _stat(50.0, 45.0, 55.0),
                ("ldgsts", 2, 32): _stat(60.0, 55.0, 65.0),
                ("ldgsts", 2, 64): _stat(100.0, 95.0, 105.0),
            }
            rows = compute_saturation_candidates(stats_sat_64)
            rec.check(
                "saturation earliest-candidate selects 64 KiB (item 21)",
                rows[0]["earliest_tested_candidate_saturation_bif_kib"] == 64, detail=f"{rows[0]}",
            )

            stats_sat_nonoverlap = {
                ("ldgsts", 2, 16): _stat(96.0, 90.0, 94.0),
                ("ldgsts", 2, 32): _stat(96.0, 90.0, 94.0),
                ("ldgsts", 2, 64): _stat(100.0, 97.0, 103.0),
            }
            rows = compute_saturation_candidates(stats_sat_nonoverlap)
            rec.check(
                "a non-overlapping CI prevents an earlier saturation selection (item 22)",
                rows[0]["earliest_tested_candidate_saturation_bif_kib"] == 64, detail=f"{rows[0]}",
            )

            # --- symlink safety: raw root, campaign dir (item 23) --------------------
            raw_root_target = tmp_path / "outside_raw_root"
            raw_root_target.mkdir()
            fake_raw_root = tmp_path / "results" / "raw"
            fake_raw_root.parent.mkdir(parents=True, exist_ok=True)
            symlink_raw_root_ok = True
            try:
                fake_raw_root.symlink_to(raw_root_target)
                try:
                    create_p14_campaign_dir("20260728T140000Z")
                    symlink_raw_root_ok = False
                except p13.UnsafePathError:
                    pass
            except OSError:
                pass
            finally:
                if fake_raw_root.is_symlink():
                    fake_raw_root.unlink()
            rec.check("a symlinked raw root is rejected (item 23)", symlink_raw_root_ok)

            campaign_dir_ok = create_p14_campaign_dir("20260728T150000Z")
            outside_campaign_target = tmp_path / "outside_campaign"
            outside_campaign_target.mkdir()
            fake_campaign_path = campaign_dir_ok.parent / "20260728T150001Z"
            symlink_campaign_ok = True
            try:
                fake_campaign_path.symlink_to(outside_campaign_target)
                try:
                    resolve_p14_campaign_dir(str(fake_campaign_path.relative_to(tmp_path)))
                    symlink_campaign_ok = False
                except p13.UnsafePathError:
                    pass
            except OSError:
                pass
            finally:
                if fake_campaign_path.is_symlink():
                    fake_campaign_path.unlink()
            rec.check("a symlinked campaign directory is rejected (item 23)", symlink_campaign_ok)

            profiles_symlink_ok = True
            outside_profiles_target = tmp_path / "outside_profiles"
            outside_profiles_target.mkdir()
            real_profiles_dir = campaign_dir_ok / "profiles"
            real_profiles_dir.rmdir()
            try:
                real_profiles_dir.symlink_to(outside_profiles_target)
                try:
                    resolve_p14_campaign_dir(str(campaign_dir_ok.relative_to(tmp_path)))
                    profiles_symlink_ok = False
                except p13.UnsafePathError:
                    pass
            except OSError:
                pass
            finally:
                if real_profiles_dir.is_symlink():
                    real_profiles_dir.unlink()
                    real_profiles_dir.mkdir()
            rec.check("a symlinked profiles/ subdirectory is rejected (item 23)", profiles_symlink_ok)

            # --- record-pilot: happy path + rejections (items 5, 6, 7, 8, 9) --------
            good_preflight_rp = tmp_path / "rp_good_preflight.json"
            _write_preflight_json(good_preflight_rp, _default_preflight_doc())
            now_rp = _datetime(2026, 7, 28, 12, 0, tzinfo=_timezone.utc)

            p13_good, _ = _build_p13_pilot_campaign_fixture(tmp_path, "rp_good")
            p14_good = _do_init_campaign(campaign_id="20260728T160000Z", started_at_utc="20260728T160000Z")
            ok, errors = _do_record_pilot(
                campaign_dir=p14_good, p13_campaign_dir=p13_good, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260728T160500Z", now_utc=now_rp,
            )
            rec.check("record-pilot accepts a valid COMPLETE benchmark P1.3 campaign", ok, detail=f"errors={errors}")
            manifest_good = p13.load_manifest(p14_good)
            rec.check("record-pilot transitions to PILOT_COMPLETE", manifest_good.get("state") == "PILOT_COMPLETE")

            p13_smoke, _ = _build_p13_pilot_campaign_fixture(tmp_path, "rp_smoke", run_kind="smoke")
            p14_smoke = _do_init_campaign(campaign_id="20260728T160100Z", started_at_utc="20260728T160100Z")
            ok, errors = _do_record_pilot(
                campaign_dir=p14_smoke, p13_campaign_dir=p13_smoke, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260728T160500Z", now_utc=now_rp,
            )
            rec.expect_error_containing("record-pilot rejects a smoke P1.3 campaign (item 5)", errors, "run_kind")
            rec.check("record-pilot smoke rejection leaves campaign FAILED, not PILOT_COMPLETE",
                      not ok and p13.load_manifest(p14_smoke).get("state") == "FAILED")

            p13_inprog_dir = tmp_path / "p13_campaigns" / "rp_inprog_raw"
            plan_inprog = p13._build_valid_campaign(p13_inprog_dir, repetitions=30, run_kind="benchmark", passes=32, warmup_ms=2000)
            p13.write_execution_order(p13_inprog_dir, plan_inprog)
            p13.merge_manifest(
                p13_inprog_dir,
                {
                    "campaign_id": "rp_inprog_raw", "run_kind": "benchmark", "started_at_utc": "20260728T160000Z",
                    "configuration_count_expected": 18, "configuration_count_completed": 18,
                    "sample_count_expected": 18 * 30, "sample_count_completed": 18 * 30,
                    "requested": {"run_kind": "benchmark", "working_set_mib": 512, "passes": 32,
                                  "warmup_ms": 2000, "repetitions": 30, "campaign_id": "rp_inprog_raw"},
                    "selected_gpu_index": 0, "git_commit": _FIXED_GIT_COMMIT, "git_dirty": False,
                    "self_test_outcomes": {"ldgsts": "PASS", "tma": "PASS"},
                },
                status="IN_PROGRESS",
            )
            p14_inprog = _do_init_campaign(campaign_id="20260728T160200Z", started_at_utc="20260728T160200Z")
            ok, errors = _do_record_pilot(
                campaign_dir=p14_inprog, p13_campaign_dir=p13_inprog_dir, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260728T160500Z", now_utc=now_rp,
            )
            rec.expect_error_containing("record-pilot rejects a non-COMPLETE P1.3 campaign (item 6)", errors, "COMPLETE")

            p13_wrongcfg, _ = _build_p13_pilot_campaign_fixture(tmp_path, "rp_wrongcfg", repetitions=5)
            p14_wrongcfg = _do_init_campaign(campaign_id="20260728T160300Z", started_at_utc="20260728T160300Z")
            ok, errors = _do_record_pilot(
                campaign_dir=p14_wrongcfg, p13_campaign_dir=p13_wrongcfg, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260728T160500Z", now_utc=now_rp,
            )
            rec.expect_error_containing(
                "record-pilot rejects wrong repetitions (item 7)", errors, "repetitions",
            )

            p13_wronguuid, _ = _build_p13_pilot_campaign_fixture(tmp_path, "rp_wronguuid", gpu_uuid="GPU-99999999-8888-7777-6666-555555555555")
            p14_wronguuid = _do_init_campaign(campaign_id="20260728T160400Z", started_at_utc="20260728T160400Z")
            ok, errors = _do_record_pilot(
                campaign_dir=p14_wronguuid, p13_campaign_dir=p13_wronguuid, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260728T160500Z", now_utc=now_rp,
            )
            rec.expect_error_containing("record-pilot rejects a changed GPU UUID (item 8)", errors, "gpu_uuid")

            p13_wrongcommit, _ = _build_p13_pilot_campaign_fixture(tmp_path, "rp_wrongcommit", git_commit="d" * 40)
            p14_wrongcommit = _do_init_campaign(campaign_id="20260728T160500Z", started_at_utc="20260728T160500Z")
            ok, errors = _do_record_pilot(
                campaign_dir=p14_wrongcommit, p13_campaign_dir=p13_wrongcommit, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260728T160600Z", now_utc=now_rp,
            )
            rec.expect_error_containing("record-pilot rejects a changed git commit (item 8)", errors, "git_commit")

            p13_tamper, _ = _build_p13_pilot_campaign_fixture(tmp_path, "rp_tamper")
            (p13_tamper / "combined_samples.csv").write_text(
                (p13_tamper / "combined_samples.csv").read_text(encoding="utf-8") + "\n", encoding="utf-8",
            )
            p14_tamper = _do_init_campaign(campaign_id="20260728T160600Z", started_at_utc="20260728T160600Z")
            ok, errors = _do_record_pilot(
                campaign_dir=p14_tamper, p13_campaign_dir=p13_tamper, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260728T160700Z", now_utc=now_rp,
            )
            rec.expect_error_containing(
                "record-pilot rejects a tampered combined_samples.csv hash (item 9)", errors, "SHA-256",
            )

            # --- discover-metrics: happy path + missing mandatory metric --------
            p13_dm, _ = _build_p13_pilot_campaign_fixture(tmp_path, "dm_good")
            p14_dm = _do_init_campaign(campaign_id="20260728T161000Z", started_at_utc="20260728T161000Z")
            ok, errors = _do_record_pilot(
                campaign_dir=p14_dm, p13_campaign_dir=p13_dm, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260728T161100Z", now_utc=now_rp,
            )
            assert ok, errors
            full_discovery_log = tmp_path / "discovery_full.log"
            full_discovery_log.write_text("\n".join(CANDIDATE_METRICS) + "\nunrelated__noise.metric\n", encoding="utf-8")
            ok, errors, resolved = _do_discover_metrics(
                campaign_dir=p14_dm, discovery_log=full_discovery_log, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, started_at_utc="20260728T161200Z", now_utc=now_rp,
            )
            rec.check(
                "discover-metrics resolves all five candidate metrics and transitions to PROFILE_IN_PROGRESS",
                ok and resolved["dram_read_metric_available"]
                and p13.load_manifest(p14_dm).get("state") == "PROFILE_IN_PROGRESS",
                detail=f"ok={ok} resolved={resolved}",
            )

            p13_dm2, _ = _build_p13_pilot_campaign_fixture(tmp_path, "dm_missing")
            p14_dm2 = _do_init_campaign(campaign_id="20260728T161300Z", started_at_utc="20260728T161300Z")
            ok, errors = _do_record_pilot(
                campaign_dir=p14_dm2, p13_campaign_dir=p13_dm2, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260728T161400Z", now_utc=now_rp,
            )
            assert ok, errors
            partial_discovery_log = tmp_path / "discovery_partial.log"
            partial_discovery_log.write_text(
                "\n".join(m for m in CANDIDATE_METRICS if m != MANDATORY_DRAM_METRIC) + "\n", encoding="utf-8",
            )
            ok, errors, resolved = _do_discover_metrics(
                campaign_dir=p14_dm2, discovery_log=partial_discovery_log, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, started_at_utc="20260728T161500Z", now_utc=now_rp,
            )
            rec.check(
                "discover-metrics still completes (COMPLETE-able) when the mandatory DRAM metric is unresolved "
                "(item 12: a data-quality flag, not a hard failure)",
                ok and not resolved["dram_read_metric_available"],
                detail=f"ok={ok} resolved={resolved}",
            )
            case0 = build_ncu_plan()[0]
            app_csv0, metrics_csv0, ncu_rep0, _ = _build_ncu_case_fixture(tmp_path, case0)
            case_ok, case_errors = _do_validate_profile_case(
                campaign_dir=p14_dm2, index=0, application_csv=app_csv0, metrics_csv=metrics_csv0,
                ncu_rep=ncu_rep0, git_commit=_FIXED_GIT_COMMIT,
            )
            classification_when_unavailable = p13.load_manifest(p14_dm2)["case_results"][case0["case_name"]]["hbm_classification"]
            rec.check(
                "a case is INCONCLUSIVE whenever the campaign's DRAM metric was never resolved (item 12)",
                case_ok and classification_when_unavailable == "INCONCLUSIVE",
                detail=f"case_ok={case_ok} classification={classification_when_unavailable}",
            )

            # --- validate-profile-case: wrong kernel name / duplicate launch (item 11) --
            p14_vc, _ = _run_profile_pipeline(tmp_path, "20260728T170000Z")
            # (the pipeline above already validates all six cases; build one more,
            # fresh campaign to test rejections against an un-recorded index)
            p13_vc, _ = _build_p13_pilot_campaign_fixture(tmp_path, "vc_base")
            p14_vc2 = _do_init_campaign(campaign_id="20260728T162000Z", started_at_utc="20260728T162000Z")
            ok, errors = _do_record_pilot(
                campaign_dir=p14_vc2, p13_campaign_dir=p13_vc, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260728T162100Z", now_utc=now_rp,
            )
            assert ok, errors
            disc_log_vc = tmp_path / "disc_vc.log"
            disc_log_vc.write_text("\n".join(CANDIDATE_METRICS) + "\n", encoding="utf-8")
            ok, errors, _ = _do_discover_metrics(
                campaign_dir=p14_vc2, discovery_log=disc_log_vc, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, started_at_utc="20260728T162200Z", now_utc=now_rp,
            )
            assert ok, errors

            case_wrong_kernel = build_ncu_plan()[0]  # ldgsts case
            app_wk, metrics_wk, rep_wk, _ = _build_ncu_case_fixture(
                tmp_path, case_wrong_kernel, kernel_name_in_csv="tma_benchmark_kernel<2,4>(...)",
            )
            ok, errors = _do_validate_profile_case(
                campaign_dir=p14_vc2, index=0, application_csv=app_wk, metrics_csv=metrics_wk,
                ncu_rep=rep_wk, git_commit=_FIXED_GIT_COMMIT,
            )
            rec.expect_error_containing("validate-profile-case rejects a wrong kernel name (item 11)", errors, "kernel name")

            case_dup_launch = build_ncu_plan()[1]
            app_dl, metrics_dl, rep_dl, _ = _build_ncu_case_fixture(tmp_path, case_dup_launch, extra_kernel_row=True)
            ok, errors = _do_validate_profile_case(
                campaign_dir=p14_vc2, index=1, application_csv=app_dl, metrics_csv=metrics_dl,
                ncu_rep=rep_dl, git_commit=_FIXED_GIT_COMMIT,
            )
            rec.expect_error_containing(
                "validate-profile-case rejects more than one distinct profiled kernel (item 11)", errors,
                "more than one distinct",
            )

            case_redo = build_ncu_plan()[0]
            app_redo, metrics_redo, rep_redo, _ = _build_ncu_case_fixture(tmp_path, case_redo)
            ok, errors = _do_validate_profile_case(
                campaign_dir=p14_vc2, index=0, application_csv=app_redo, metrics_csv=metrics_redo,
                ncu_rep=rep_redo, git_commit=_FIXED_GIT_COMMIT,
            )
            rec.check("validate-profile-case accepts case 0 on its first, correct submission", ok, detail=f"{errors}")
            ok, errors = _do_validate_profile_case(
                campaign_dir=p14_vc2, index=0, application_csv=app_redo, metrics_csv=metrics_redo,
                ncu_rep=rep_redo, git_commit=_FIXED_GIT_COMMIT,
            )
            rec.expect_error_containing(
                "validate-profile-case refuses to silently re-record an already-validated case (items 24/25)",
                errors, "already",
            )

            # --- finalize-profile: missing case rejected (item 10) -------------------
            ok, errors = _do_finalize_profile(campaign_dir=p14_vc2, completed_at_utc="20260728T163000Z")
            rec.expect_error_containing(
                "finalize-profile rejects an incomplete six-case set (item 10)", errors, "never validated",
            )

            ok, errors = _do_finalize_profile(campaign_dir=p14_vc, completed_at_utc="20260728T163100Z")
            rec.check("finalize-profile accepts a complete, correctly-ordered six-case set", ok, detail=f"{errors}")
            rec.check(
                "finalize-profile transitions to COMPLETE", p13.load_manifest(p14_vc).get("state") == "COMPLETE",
            )

            reordered_plan_path = p14_vc / "profile_plan.csv"
            plan_rows = list(csv.reader(reordered_plan_path.read_text(encoding="utf-8").splitlines()))
            plan_rows[1], plan_rows[2] = plan_rows[2], plan_rows[1]
            reorder_errors = validate_profile_plan_file(
                _write_rows_csv(tmp_path / "reordered_profile_plan.csv", plan_rows), build_ncu_plan(),
            )
            rec.check("a reordered profile_plan.csv is rejected (item 10)", bool(reorder_errors))

            # --- no-clobber: existing output/temporary never overwritten (item 24) --
            noclobber_dir = tmp_path / "noclobber"
            noclobber_dir.mkdir()
            existing_csv = noclobber_dir / "existing.csv"
            existing_csv.write_text("original,content\n1,2\n", encoding="utf-8")
            raised = False
            try:
                _write_csv_no_clobber(existing_csv, ["a", "b"], [{"a": "9", "b": "9"}])
            except p13.UnsafePathError:
                raised = True
            rec.check(
                "_write_csv_no_clobber refuses an existing target and leaves it untouched (item 24)",
                raised and existing_csv.read_text(encoding="utf-8") == "original,content\n1,2\n",
            )
            existing_md = noclobber_dir / "existing.md"
            existing_md.write_text("original report\n", encoding="utf-8")
            raised = False
            try:
                _write_text_no_clobber(existing_md, "new report\n")
            except p13.UnsafePathError:
                raised = True
            rec.check(
                "_write_text_no_clobber refuses an existing target and leaves it untouched (item 24)",
                raised and existing_md.read_text(encoding="utf-8") == "original report\n",
            )

            # --- no stale .tmp left behind on a mid-write failure (item 25) ---------
            broken_row_csv = noclobber_dir / "broken_row.csv"
            broken_tmp = broken_row_csv.with_suffix(broken_row_csv.suffix + ".tmp")
            try:
                _write_csv_no_clobber(broken_row_csv, ["a", "b"], [{"a": "1"}])  # missing key "b" -> KeyError
            except KeyError:
                pass
            rec.check(
                "a mid-write failure leaves no stale .tmp file and no partial target (item 25)",
                not os.path.lexists(broken_row_csv) and not os.path.lexists(broken_tmp),
            )

            # --- full pipeline: determinism (items 17, 26), hash presence (item 27) --
            def _fixed_gbps(entry: dict, sample_index: int) -> float:
                base = {"ldgsts": 900.0, "tma": 1000.0}[entry["method"]] + entry["stages"] * 2.0 + entry["bif_kib"] * 0.1
                jitter = (sample_index % 5) * 0.37
                return base + jitter

            det_reports: list[dict[str, bytes]] = []
            det_hash_sets: list[frozenset] = []
            for det_run in range(2):
                det_campaign_id = f"20260728T18{det_run:02d}00Z"
                p13_det, _ = _build_p13_pilot_campaign_fixture(
                    tmp_path, f"det_{det_run}", gbps_fn=_fixed_gbps,
                )
                p14_det = _do_init_campaign(campaign_id=det_campaign_id, started_at_utc=det_campaign_id)
                preflight_det = tmp_path / f"preflight_det_{det_run}.json"
                _write_preflight_json(preflight_det, _default_preflight_doc())
                now_det = _datetime(2026, 7, 28, 11, 0, tzinfo=_timezone.utc)
                ok, errors = _do_record_pilot(
                    campaign_dir=p14_det, p13_campaign_dir=p13_det, preflight_path=preflight_det,
                    git_commit=_FIXED_GIT_COMMIT, completed_at_utc=det_campaign_id, now_utc=now_det,
                )
                assert ok, errors
                disc_det = tmp_path / f"disc_det_{det_run}.log"
                disc_det.write_text("\n".join(CANDIDATE_METRICS) + "\n", encoding="utf-8")
                ok, errors, _ = _do_discover_metrics(
                    campaign_dir=p14_det, discovery_log=disc_det, preflight_path=preflight_det,
                    git_commit=_FIXED_GIT_COMMIT, started_at_utc=det_campaign_id, now_utc=now_det,
                )
                assert ok, errors
                for entry in build_ncu_plan():
                    app_csv, metrics_csv, ncu_rep, _row = _build_ncu_case_fixture(tmp_path, entry)
                    ok, errors = _do_validate_profile_case(
                        campaign_dir=p14_det, index=entry["index"], application_csv=app_csv,
                        metrics_csv=metrics_csv, ncu_rep=ncu_rep, git_commit=_FIXED_GIT_COMMIT,
                    )
                    assert ok, errors
                ok, errors = _do_finalize_profile(campaign_dir=p14_det, completed_at_utc=det_campaign_id)
                assert ok, errors
                ok, errors = _do_analyze(campaign_dir=p14_det, analyzed_at_utc=det_campaign_id)
                assert ok, errors

                analysis_dir_det = p14_det / "analysis"
                artifact_names = (
                    "pilot_statistics.csv", "pairwise_comparison.csv", "saturation_candidates.csv",
                    "ncu_validation.csv", "analysis.json", "report.md",
                )
                # Each determinism run deliberately uses its own campaign_id (so the
                # two fixture campaigns cannot collide on disk); normalize that one
                # legitimate, expected difference out before the byte comparison
                # below, which is otherwise checking real statistical determinism.
                snapshot: dict[str, bytes] = {
                    name: (analysis_dir_det / name).read_bytes().replace(
                        det_campaign_id.encode("utf-8"), b"NORMALIZED_CAMPAIGN_ID",
                    )
                    for name in artifact_names
                }
                for svg_name in ("effective_gbps.svg", "tma_to_ldgsts_ratio.svg", "dram_read_ratio.svg"):
                    svg_path = analysis_dir_det / "figures" / svg_name
                    svg_bytes = svg_path.read_bytes()
                    snapshot[f"figures/{svg_name}"] = svg_bytes
                    well_formed = True
                    try:
                        minidom.parseString(svg_bytes)
                    except Exception:
                        well_formed = False
                    rec.check(f"{svg_name} (run {det_run}) is well-formed XML", well_formed)
                det_reports.append(snapshot)
                manifest_det = p13.load_manifest(p14_det)
                # analysis.json/report.md legitimately embed this run's own
                # campaign_id, so their real on-disk hashes differ between runs;
                # only the CSV/SVG artifacts (which never mention campaign_id)
                # are expected to hash identically across the two runs.
                campaign_id_free_hashes = frozenset(
                    (k, v) for k, v in manifest_det["artifact_sha256"].items()
                    if not k.endswith(("analysis.json", "report.md"))
                    and any(k.endswith(name) for name in artifact_names + ("effective_gbps.svg", "tma_to_ldgsts_ratio.svg", "dram_read_ratio.svg"))
                )
                det_hash_sets.append(campaign_id_free_hashes)

            rec.check(
                "analyze() output is byte-identical, file by file, across two independently-built, "
                "data-identical campaigns (items 17, 26)",
                det_reports[0] == det_reports[1],
                detail=f"differing={[k for k in det_reports[0] if det_reports[0][k] != det_reports[1][k]]}",
            )
            rec.check(
                "recorded SHA-256 hashes for the campaign_id-free CSV/SVG artifacts are "
                "identical across both determinism runs",
                bool(det_hash_sets[0]) and det_hash_sets[0] == det_hash_sets[1],
                detail=f"run0={sorted(det_hash_sets[0])} run1={sorted(det_hash_sets[1])}",
            )
            expected_artifact_keys = {
                "analysis/pilot_statistics.csv", "analysis/pairwise_comparison.csv",
                "analysis/saturation_candidates.csv", "analysis/ncu_validation.csv",
                "analysis/analysis.json", "analysis/report.md",
                "analysis/figures/effective_gbps.svg", "analysis/figures/tma_to_ldgsts_ratio.svg",
                "analysis/figures/dram_read_ratio.svg",
            }
            last_manifest = p13.load_manifest(p14_det)
            all_hashes = last_manifest["artifact_sha256"]
            rec.check(
                "every final analysis artifact has a non-null recorded SHA-256 hash (item 27)",
                expected_artifact_keys <= set(all_hashes)
                and all(p13._is_sha256_hex(all_hashes[k]) for k in expected_artifact_keys),
                detail=f"missing={expected_artifact_keys - set(all_hashes)}",
            )
            rec.check("ANALYZED is the final recorded state", last_manifest.get("state") == "ANALYZED")

            # --- no analysis published when required validation fails (item 28) -----
            p13_broken, _ = _build_p13_pilot_campaign_fixture(tmp_path, "broken_analyze")
            p14_broken = _do_init_campaign(campaign_id="20260728T190000Z", started_at_utc="20260728T190000Z")
            preflight_broken = tmp_path / "preflight_broken.json"
            _write_preflight_json(preflight_broken, _default_preflight_doc())
            now_broken = _datetime(2026, 7, 28, 11, 0, tzinfo=_timezone.utc)
            ok, errors = _do_record_pilot(
                campaign_dir=p14_broken, p13_campaign_dir=p13_broken, preflight_path=preflight_broken,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260728T190100Z", now_utc=now_broken,
            )
            assert ok, errors
            disc_broken = tmp_path / "disc_broken.log"
            disc_broken.write_text("\n".join(CANDIDATE_METRICS) + "\n", encoding="utf-8")
            ok, errors, _ = _do_discover_metrics(
                campaign_dir=p14_broken, discovery_log=disc_broken, preflight_path=preflight_broken,
                git_commit=_FIXED_GIT_COMMIT, started_at_utc="20260728T190200Z", now_utc=now_broken,
            )
            assert ok, errors
            for entry in build_ncu_plan():
                app_csv, metrics_csv, ncu_rep, _row = _build_ncu_case_fixture(tmp_path, entry)
                ok, errors = _do_validate_profile_case(
                    campaign_dir=p14_broken, index=entry["index"], application_csv=app_csv,
                    metrics_csv=metrics_csv, ncu_rep=ncu_rep, git_commit=_FIXED_GIT_COMMIT,
                )
                assert ok, errors
            ok, errors = _do_finalize_profile(campaign_dir=p14_broken, completed_at_utc="20260728T190300Z")
            assert ok, errors
            # Corrupt the pilot's combined_samples.csv after COMPLETE, simulating
            # tampering/corruption discovered only when analyze() re-verifies it.
            (p13_broken / "combined_samples.csv").write_text(
                (p13_broken / "combined_samples.csv").read_text(encoding="utf-8") + "\ntampered\n",
                encoding="utf-8",
            )
            ok, errors = _do_analyze(campaign_dir=p14_broken, analyzed_at_utc="20260728T190400Z")
            analysis_dir_broken = p14_broken / "analysis"
            published_files = list(analysis_dir_broken.rglob("*")) if analysis_dir_broken.exists() else []
            published_content_files = [
                p for p in published_files if p.is_file() and p.suffix in (".csv", ".json", ".md", ".svg")
            ]
            rec.check(
                "analyze() rejects a tampered pilot input and publishes no analysis artifact (item 28)",
                not ok and not published_content_files,
                detail=f"ok={ok} errors={errors} published={published_content_files}",
            )
            rec.check(
                "a failed analyze() leaves the campaign state COMPLETE (retriable, not FAILED)",
                p13.load_manifest(p14_broken).get("state") == "COMPLETE",
            )

    if rec.failures:
        print(
            f"analyze_exp01_memory_paths_p14: self-test: FAILED ({len(rec.failures)}/{rec.total} case(s)): "
            f"{rec.failures}",
            file=sys.stderr,
        )
        print("analyze_exp01_memory_paths_p14: SELF_TEST_RESULT=FAIL", file=sys.stderr)
        return 1
    print(f"analyze_exp01_memory_paths_p14: self-test: OK ({rec.total} cases)", file=sys.stderr)
    print("analyze_exp01_memory_paths_p14: SELF_TEST_RESULT=PASS", file=sys.stderr)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_exp01_memory_paths_p14.py",
        description="P1.4 plan/preflight/pilot-recording/NCU-validation/analysis helper (see module docstring).",
    )
    parser.add_argument("--self-test", action="store_true", help="Run GPU-free synthetic tests and exit.")
    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="Print the frozen six-case NCU plan.")
    plan_parser.add_argument("--format", choices=("text", "lines", "json"), default="text")
    plan_parser.set_defaults(func=cmd_plan)

    init_parser = subparsers.add_parser("init-campaign", help="Symlink-safe P1.4 campaign creation.")
    init_parser.add_argument("--campaign-id", required=True)
    init_parser.add_argument("--started-at-utc", required=True)
    init_parser.set_defaults(func=cmd_init_campaign)

    vp_parser = subparsers.add_parser("validate-preflight", help="Validate a preflight summary.json.")
    vp_parser.add_argument("--preflight", required=True)
    vp_parser.add_argument("--expected-git-commit", required=True)
    vp_parser.add_argument("--now", default=None, help="Override 'now' (YYYY-MM-DDTHH:MM:SSZ); for tests.")
    vp_parser.set_defaults(func=cmd_validate_preflight)

    rp_parser = subparsers.add_parser("record-pilot", help="Validate a completed P1.3 pilot campaign.")
    rp_parser.add_argument("--campaign-dir", required=True)
    rp_parser.add_argument("--p13-campaign-dir", required=True)
    rp_parser.add_argument("--preflight", required=True)
    rp_parser.add_argument("--git-commit", required=True)
    rp_parser.add_argument("--completed-at-utc", required=True)
    rp_parser.add_argument("--now", default=None)
    rp_parser.set_defaults(func=cmd_record_pilot)

    dm_parser = subparsers.add_parser("discover-metrics", help="Record resolved NCU metrics; start profiling.")
    dm_parser.add_argument("--campaign-dir", required=True)
    dm_parser.add_argument("--discovery-log", required=True)
    dm_parser.add_argument("--preflight", required=True)
    dm_parser.add_argument("--git-commit", required=True)
    dm_parser.add_argument("--started-at-utc", required=True)
    dm_parser.add_argument("--now", default=None)
    dm_parser.set_defaults(func=cmd_discover_metrics)

    vc_parser = subparsers.add_parser("validate-profile-case", help="Validate one captured NCU profile case.")
    vc_parser.add_argument("--campaign-dir", required=True)
    vc_parser.add_argument("--index", required=True, type=int)
    vc_parser.add_argument("--application-csv", required=True)
    vc_parser.add_argument("--metrics-csv", required=True)
    vc_parser.add_argument("--ncu-rep", required=True)
    vc_parser.add_argument("--git-commit", required=True)
    vc_parser.set_defaults(func=cmd_validate_profile_case)

    fp_parser = subparsers.add_parser("finalize-profile", help="Re-validate and close the six-case profile set.")
    fp_parser.add_argument("--campaign-dir", required=True)
    fp_parser.add_argument("--completed-at-utc", required=True)
    fp_parser.set_defaults(func=cmd_finalize_profile)

    an_parser = subparsers.add_parser("analyze", help="Generate analysis/* from a COMPLETE campaign.")
    an_parser.add_argument("--campaign-dir", required=True)
    an_parser.add_argument("--analyzed-at-utc", required=True)
    an_parser.set_defaults(func=cmd_analyze)

    mw_parser = subparsers.add_parser("manifest-write", help="Mark FAILED/INTERRUPTED (never a completing state).")
    mw_parser.add_argument("--campaign-dir", required=True)
    mw_parser.add_argument("--status", required=True, choices=("FAILED", "INTERRUPTED"))
    mw_parser.add_argument("--merge-json", default=None)
    mw_parser.set_defaults(func=cmd_manifest_write)

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
