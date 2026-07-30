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
unmodified) as a library and reuses its generic path-safety primitives
(symlink rejection, no-clobber publish, exclusive creation, SHA-256
hashing), its 37-column CSV schema and per-field validators, and its geometry
formulas rather than reimplementing them. P1.4 deliberately does not reuse
P1.3's mutable manifest writer: its own state is an append-only immutable
revision chain. P1.4 does not modify, and does not need to modify, the P1.3
file.

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
  manifest-write        Mark FAILED/INTERRUPTED with required failure
                        stage/detail metadata (mirrors P1.3's own
                        manifest-write, but never accepts a completing
                        status).
  --self-test           GPU-free synthetic/adversarial tests. Prints
                        "analyze_exp01_memory_paths_p14: SELF_TEST_RESULT=PASS"
                        only if every case passes.

Exit codes: 0 on success (including --self-test passing); 1 on a validation
or analysis failure; 2 on a usage/precondition error.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import random
import re
import shutil
import stat
import statistics
import sys
import tempfile
import xml.dom.minidom as minidom
from unittest import mock
from datetime import datetime as _datetime
from datetime import timedelta as _timedelta
from datetime import timezone as _timezone
from pathlib import Path
from typing import Callable
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
# Task 4 remediation (Section 9): PILOT_IN_PROGRESS has no self-loop.
# Unlike PROFILE_IN_PROGRESS (which legitimately repeats itself once per
# validated case, six times, via _do_validate_profile_case), --pilot never
# reports incremental progress into the P1.4 manifest -- record-pilot goes
# directly from PILOT_IN_PROGRESS to PILOT_COMPLETE in one step. A same-
# state PILOT_IN_PROGRESS revision is therefore not a transition the actual
# workflow ever produces, so it is removed rather than left as an
# unnecessarily permissive extra edge (only FAILED/INTERRUPTED remain
# reachable from PILOT_IN_PROGRESS besides its one real next state).
ALLOWED_P14_TRANSITIONS: dict[str | None, frozenset[str]] = {
    None: frozenset({"PILOT_IN_PROGRESS"}),
    "PILOT_IN_PROGRESS": frozenset({"PILOT_COMPLETE", "FAILED", "INTERRUPTED"}),
    # --profile installs its signal/exit traps while the campaign is still
    # PILOT_COMPLETE, before metric discovery publishes PROFILE_IN_PROGRESS.
    # An interruption in that real runner interval must therefore be
    # recordable rather than silently discarded by `|| true`.
    "PILOT_COMPLETE": frozenset({"PROFILE_IN_PROGRESS", "FAILED", "INTERRUPTED"}),
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
    # Task 4 re-audit remediation (Group A): these four fields must never be
    # nullable. A premature key with value null is otherwise indistinguishable
    # from "not yet reached" by any check that tests truthiness/non-None
    # instead of presence -- absence is the only legal "not yet" placeholder.
    "pilot_completed_at_utc": str,
    "profile_started_at_utc": str,
    "profile_completed_at_utc": str,
    "analyzed_at_utc": str,
    "frozen_protocol": dict,
    "profile_plan_sha256": str,
    "pilot_campaign_reference": dict,
    "preflight_reference_pilot": dict,
    "preflight_reference_profile": dict,
    "provenance": dict,
    "resolved_ncu_metrics": dict,
    "profile_order": list,
    "profile_count_completed": int,
    "case_results": dict,
    "artifact_sha256": dict,
    # Task 4 re-audit remediation (Group A): failure_stage/failure_detail must
    # be absent (not merely null) outside FAILED/INTERRUPTED -- see the
    # comment above.
    "failure_stage": str,
    "failure_detail": list,
}

CASE_ARTIFACT_HASH_FIELDS: tuple[str, ...] = (
    "application_csv_sha256",
    "metrics_csv_sha256",
    "ncu_rep_sha256",
)

ANALYSIS_ARTIFACT_RELATIVE_PATHS: tuple[str, ...] = (
    "analysis/pilot_statistics.csv",
    "analysis/pairwise_comparison.csv",
    "analysis/saturation_candidates.csv",
    "analysis/ncu_validation.csv",
    "analysis/analysis.json",
    "analysis/report.md",
    "analysis/figures/effective_gbps.svg",
    "analysis/figures/tma_to_ldgsts_ratio.svg",
    "analysis/figures/dram_read_ratio.svg",
)


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


# Fields both preflight snapshots share, by construction, since both come
# from validate_preflight_file/validate_preflight_fields. gpu_driver_version
# here is always the NVIDIA display-driver string reported by
# scripts/preflight.sh's gpu.driver_version -- never compare it against the
# separate, integer-encoded cuda_driver_version/cuda_runtime_version fields
# recorded elsewhere in P1.3 provenance; those are a different representation
# of a different thing (the CUDA driver/runtime API version, not the NVIDIA
# kernel driver package) and crossing that boundary would be comparing
# incomparable values. This function only ever takes two same-shaped preflight
# snapshots, so that boundary cannot be crossed by construction.
PREFLIGHT_PROVENANCE_FIELDS = ("git_commit", "gpu_uuid", "gpu_name", "gpu_compute_cap", "gpu_driver_version")


def compare_preflight_provenance(pilot_snapshot: dict, profile_snapshot: dict) -> list[str]:
    """Fail-closed comparison of the pilot-phase and profiling-phase preflight
    snapshots recorded for the same P1.4 campaign (audit blocker #1: a
    profiling preflight with a different GPU UUID/name/driver from the pilot
    preflight must never be silently accepted). Returns one error per
    differing field; empty iff every field matches exactly."""
    errors: list[str] = []
    for field in PREFLIGHT_PROVENANCE_FIELDS:
        pilot_value = pilot_snapshot.get(field)
        profile_value = profile_snapshot.get(field)
        if pilot_value != profile_value:
            errors.append(
                f"profile preflight {field}={profile_value!r} != pilot preflight "
                f"{field}={pilot_value!r} (the pilot and profiling phases of one "
                f"campaign must run on the identical GPU/driver/commit)"
            )
    return errors


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
    results/raw/exp01_memory_paths_p14/<campaign_id>/{profiles,analysis,logs,manifest}
    one component at a time via lstat (p13._mkdir_component), refusing a
    symlink or wrong-type object at any level (including the raw root
    itself), and fails if the campaign directory already exists. `manifest/`
    holds the append-only, immutable manifest revision chain (see
    load_p14_manifest_chain / write_next_p14_manifest_revision)."""
    validate_p14_campaign_id(campaign_id)
    current = REPO_ROOT
    for part in RAW_ROOT_PARTS_P14:
        current = current / part
        p13._mkdir_component(current, must_not_exist=False)
    campaign_dir = current / campaign_id
    p13._mkdir_component(campaign_dir, must_not_exist=True)
    for sub in ("profiles", "analysis", "logs", "manifest"):
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
    for subdir_name in ("profiles", "analysis", "logs", "manifest"):
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
# P1.4 manifest: allowlisted keys/types, enforced state machine. P1.4's own
# manifest state is published as an append-only, immutable revision chain
# (campaign_dir/manifest/000000.json, 000001.json, ...; see
# load_p14_manifest_chain / write_next_p14_manifest_revision below) -- it
# never calls p13.write_manifest_atomic (which uses os.replace() to
# atomically *replace* manifest.json in place; P1.3's own campaign manifest
# is a separate, frozen, audited input that keeps using that mechanism
# unmodified). This module reuses p13._manifest_type_matches and
# p13._validate_compact_timestamp (generic schema-type helpers), and the
# generic lstat-based path-safety/no-clobber primitives (_open_exclusive,
# _publish_no_clobber, _reject_if_symlink_or_wrong_type, _mkdir_component,
# _open_regular_nofollow, sha256_of) for the revision files themselves.
# ---------------------------------------------------------------------------
def _expected_complete_artifact_sha256(manifest: dict) -> tuple[dict[str, str], list[str]]:
    """Derives COMPLETE's exact base-evidence hash map from the manifest's
    already-recorded canonical references and six case results.

    This is deliberately independent of artifact_sha256 itself: a terminal
    revision cannot make an incomplete or fabricated map self-consistent by
    merely asserting the same bad value twice. The evidence-integrity gate
    separately reopens and verifies these recorded source hashes against
    disk before production COMPLETE/ANALYZED transitions.
    """
    expected: dict[str, str] = {}
    errors: list[str] = []

    def _record(output_key: str, value: object, source: str) -> None:
        if not isinstance(value, str) or not p13._is_sha256_hex(value):
            errors.append(
                f"{source} must be a canonical 64-hex SHA-256 before "
                f"artifact_sha256[{output_key!r}] can be derived; got {value!r}"
            )
            return
        expected[output_key] = value

    _record(
        "profile_plan.csv",
        manifest.get("profile_plan_sha256"),
        "profile_plan_sha256",
    )

    pilot_ref = manifest.get("pilot_campaign_reference")
    if not isinstance(pilot_ref, dict):
        errors.append("pilot_campaign_reference must be an object for terminal artifact validation")
    else:
        for source_key, output_key in (
            ("manifest_sha256", "pilot_manifest_sha256"),
            ("combined_samples_sha256", "pilot_combined_samples_sha256"),
            ("summary_sha256", "pilot_summary_sha256"),
        ):
            _record(
                output_key,
                pilot_ref.get(source_key),
                f"pilot_campaign_reference.{source_key}",
            )

    for ref_name in ("preflight_reference_pilot", "preflight_reference_profile"):
        ref = manifest.get(ref_name)
        if not isinstance(ref, dict):
            errors.append(f"{ref_name} must be an object for terminal artifact validation")
            continue
        _record(ref_name, ref.get("sha256"), f"{ref_name}.sha256")

    case_results = manifest.get("case_results")
    if not isinstance(case_results, dict):
        errors.append("case_results must be an object for terminal artifact validation")
    else:
        for entry in build_ncu_plan():
            case_name = entry["case_name"]
            result = case_results.get(case_name)
            if not isinstance(result, dict):
                errors.append(
                    f"case_results.{case_name} must be an object for terminal artifact validation"
                )
                continue
            for field in CASE_ARTIFACT_HASH_FIELDS:
                _record(
                    f"{case_name}.{field}",
                    result.get(field),
                    f"case_results.{case_name}.{field}",
                )

    return expected, errors


def validate_terminal_manifest_content(manifest: dict) -> list[str]:
    """Requires canonical, complete terminal metadata.

    COMPLETE records exactly the frozen six-case plan plus every base
    evidence hash derived above. ANALYZED preserves that exact base map and
    adds exactly the nine deterministic analysis artifacts, each with a
    canonical SHA-256. No missing, extra, reordered, or internally
    inconsistent terminal field is tolerated.
    """
    state = manifest.get("state")
    if state not in ("COMPLETE", "ANALYZED"):
        return []

    errors: list[str] = []
    expected_plan = build_ncu_plan()
    if manifest.get("profile_order") != expected_plan:
        errors.append(
            "profile_order is not exactly the frozen six-case plan in its canonical order"
        )

    expected_base, source_errors = _expected_complete_artifact_sha256(manifest)
    errors.extend(source_errors)

    actual = manifest.get("artifact_sha256")
    if not isinstance(actual, dict):
        return errors + ["artifact_sha256 must be an object in COMPLETE/ANALYZED"]

    expected_keys = set(expected_base)
    if state == "ANALYZED":
        expected_keys.update(ANALYSIS_ARTIFACT_RELATIVE_PATHS)
    actual_keys = set(actual)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing:
        errors.append(f"artifact_sha256 is missing canonical key(s): {missing}")
    if unexpected:
        errors.append(f"artifact_sha256 contains unexpected key(s): {unexpected}")

    for key, expected_hash in expected_base.items():
        if key in actual and actual[key] != expected_hash:
            errors.append(
                f"artifact_sha256[{key!r}]={actual[key]!r} does not equal its "
                f"canonical recorded evidence hash {expected_hash!r}"
            )

    if state == "ANALYZED":
        for key in ANALYSIS_ARTIFACT_RELATIVE_PATHS:
            value = actual.get(key)
            if key in actual and (
                not isinstance(value, str) or not p13._is_sha256_hex(value)
            ):
                errors.append(
                    f"artifact_sha256[{key!r}] must be a canonical 64-hex SHA-256; "
                    f"got {value!r}"
                )

    return errors


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
        if key in manifest:
            p13._validate_compact_timestamp(manifest[key], key)
    if "profile_count_completed" in manifest and not (0 <= manifest["profile_count_completed"] <= EXPECTED_NCU_CASE_COUNT):
        raise p13.ManifestTransitionError(
            f"P1.4 manifest profile_count_completed must be in [0, {EXPECTED_NCU_CASE_COUNT}]"
        )
    if "failure_stage" in manifest and not manifest["failure_stage"]:
        raise p13.ManifestTransitionError("P1.4 manifest failure_stage must be non-empty when present")
    if "failure_detail" in manifest:
        if not all(isinstance(item, str) for item in manifest["failure_detail"]):
            raise p13.ManifestTransitionError("P1.4 manifest failure_detail must be a list of strings")

    required_by_state = {
        "PILOT_IN_PROGRESS": {
            "schema_version", "experiment_id", "campaign_id", "state", "publishable",
            "started_at_utc", "frozen_protocol", "profile_plan_sha256",
        },
        "PILOT_COMPLETE": {
            "pilot_completed_at_utc", "pilot_campaign_reference", "preflight_reference_pilot",
            "provenance",
        },
        "PROFILE_IN_PROGRESS": {"profile_started_at_utc", "resolved_ncu_metrics", "preflight_reference_profile"},
        "COMPLETE": {
            "profile_completed_at_utc", "profile_order", "profile_count_completed",
            "case_results", "artifact_sha256",
        },
        "ANALYZED": {"analyzed_at_utc"},
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

    if state in ("COMPLETE", "ANALYZED") and manifest.get(
        "profile_count_completed"
    ) != EXPECTED_NCU_CASE_COUNT:
        raise p13.ManifestTransitionError(
            f"P1.4 manifest state={state} requires profile_count_completed="
            f"{EXPECTED_NCU_CASE_COUNT}, got {manifest.get('profile_count_completed')!r}"
        )
    failure_keys = {"failure_stage", "failure_detail"}
    present_failure_keys = failure_keys & set(manifest)
    if state in ("FAILED", "INTERRUPTED"):
        missing_failure_keys = failure_keys - set(manifest)
        if missing_failure_keys:
            raise p13.ManifestTransitionError(
                f"P1.4 manifest state={state!r} missing failure telemetry field(s): "
                f"{sorted(missing_failure_keys)}"
            )
    elif present_failure_keys:
        raise p13.ManifestTransitionError(
            f"P1.4 manifest state={state!r} cannot contain failure telemetry field(s): "
            f"{sorted(present_failure_keys)}"
        )

    terminal_errors = validate_terminal_manifest_content(manifest)
    if terminal_errors:
        raise p13.ManifestTransitionError(
            f"P1.4 manifest state={state!r} has non-canonical terminal content: "
            f"{terminal_errors}"
        )


# ---------------------------------------------------------------------------
# Semantic manifest transition validation (Remediation C, second audit,
# blocker C). The hash chain alone only proves a revision was appended
# without altering an earlier byte -- it says nothing about whether the
# *content* of the new revision is a legitimate continuation of the old one.
# A syntactically valid, correctly re-hashed revision that simply changes
# campaign_id (or edits an earlier case_result, or jumps state illegally)
# previously passed unnoticed. Every top-level field is classified exactly
# once below; validate_manifest_revision_transition enforces the invariant
# each classification implies, and is applied to every adjacent revision
# pair while walking the chain (never only to the latest revision).
#
# Classification (documented here and in src/memory/P1_4_PROTOCOL.md):
#   immutable          -- present from revision 0 onward; the exact same
#                          value forever.
#   allowed timestamp   -- may be absent, then take a fixed value at its
#                          own specific transition; never changes once set.
#   set-once            -- (non-timestamp) may be absent, then take a fixed
#                          value at its own specific transition; never
#                          changes once set, and never reverts to absent.
#   state-derived       -- governed by the state machine itself (state's own
#                          allowlisted transitions; profile_count_completed's
#                          monotonic, never-decreasing count).
#   append-only         -- a collection that may only gain new entries; an
#                          existing entry is never edited, deleted, or
#                          reordered (case_results must also grow strictly
#                          in the frozen six-case order).
#   revision metadata    -- manifest_revision/previous_manifest_sha256 are
#                          not P1.4 manifest *content* fields at all (they
#                          are the chain wrapper); see MANIFEST_REVISION_KEYS.
# ---------------------------------------------------------------------------
P14_FIELD_IMMUTABLE = frozenset({
    "schema_version", "experiment_id", "campaign_id", "publishable",
    "frozen_protocol", "profile_plan_sha256",
})
P14_FIELD_ALLOWED_TIMESTAMP = frozenset({
    "started_at_utc", "pilot_completed_at_utc", "profile_started_at_utc",
    "profile_completed_at_utc", "analyzed_at_utc",
})
P14_FIELD_SET_ONCE = frozenset({
    "pilot_campaign_reference", "preflight_reference_pilot", "provenance",
    "preflight_reference_profile", "resolved_ncu_metrics", "profile_order",
})
P14_FIELD_STATE_DERIVED = frozenset({"state", "profile_count_completed"})
P14_FIELD_APPEND_ONLY = frozenset({"case_results", "artifact_sha256"})
# failure_stage/failure_detail: required together alongside a
# FAILED/INTERRUPTED state, and absent otherwise. Not immutable (they start
# absent), not set-once in the usual sense (a FAILED/INTERRUPTED campaign is
# always terminal in practice, so in effect they are set at most once, at
# the terminal transition), not state-derived in the state/count sense --
# validated by their own explicit rule in validate_manifest_revision_transition.
P14_FIELD_FAILURE = frozenset({"failure_stage", "failure_detail"})

_P14_FIELD_CLASSIFICATION_UNION = (
    P14_FIELD_IMMUTABLE | P14_FIELD_ALLOWED_TIMESTAMP | P14_FIELD_SET_ONCE
    | P14_FIELD_STATE_DERIVED | P14_FIELD_APPEND_ONLY | P14_FIELD_FAILURE
)
assert _P14_FIELD_CLASSIFICATION_UNION == frozenset(ALLOWED_P14_MANIFEST_KEYS), (
    "every P1.4 manifest field must be classified into exactly one of the "
    "P14_FIELD_* categories above -- an unclassified field must never be "
    "able to bypass transition validation: "
    f"missing={frozenset(ALLOWED_P14_MANIFEST_KEYS) - _P14_FIELD_CLASSIFICATION_UNION!r} "
    f"extra={_P14_FIELD_CLASSIFICATION_UNION - frozenset(ALLOWED_P14_MANIFEST_KEYS)!r}"
)

# Exact mutation matrix for every non-failure adjacent transition. Values
# are the content fields (excluding `state`) whose presence or value must
# change in that one revision. This is intentionally stricter than the
# broad immutable/set-once/append-only classifications above: those
# classifications say how a field behaves over its lifetime, while this
# table says the one exact transition on which it may mutate.
P14_EXACT_TRANSITION_MUTATIONS: dict[tuple[str, str], frozenset[str]] = {
    ("PILOT_IN_PROGRESS", "PILOT_COMPLETE"): frozenset({
        "pilot_completed_at_utc",
        "pilot_campaign_reference",
        "preflight_reference_pilot",
        "provenance",
    }),
    ("PILOT_COMPLETE", "PROFILE_IN_PROGRESS"): frozenset({
        "profile_started_at_utc",
        "resolved_ncu_metrics",
        "preflight_reference_profile",
    }),
    ("PROFILE_IN_PROGRESS", "PROFILE_IN_PROGRESS"): frozenset({
        "case_results",
        "profile_count_completed",
    }),
    ("PROFILE_IN_PROGRESS", "COMPLETE"): frozenset({
        "profile_completed_at_utc",
        "profile_order",
        "artifact_sha256",
    }),
    ("COMPLETE", "ANALYZED"): frozenset({
        "analyzed_at_utc",
        "artifact_sha256",
    }),
}


# ---------------------------------------------------------------------------
# Manifest state-shape validation (Task 4 remediation, Section 9). The
# broad set-once/timestamp classification above proves a field never
# *changes* once set, but never asserted that a field could not appear for
# the *first* time at the wrong state -- e.g. a manifest with
# state=PILOT_IN_PROGRESS that already carries resolved_ncu_metrics or
# profile_completed_at_utc previously passed unnoticed, since nothing
# checked that those fields were still absent this early. The 7-row
# state-availability matrix below bounds which fields may exist in each
# state. The adjacent-revision mutation matrix above then narrows that
# availability to the exact transition that may change each field (needed
# because PROFILE_IN_PROGRESS covers both its entering revision and its six
# case-result self-loops).
# validate_manifest_state_shape() is a pure function of one revision's own
# content -- it needs no "previous" revision, since "which fields are legal
# at state X" never depends on transition history, only on X itself.
# load_p14_manifest_chain() runs this alongside (never instead of)
# validate_manifest_revision_transition() for revision 0 and every later
# revision.
# ---------------------------------------------------------------------------
P14_STATE_ORDER: tuple[str, ...] = ("PILOT_IN_PROGRESS", "PILOT_COMPLETE", "PROFILE_IN_PROGRESS", "COMPLETE", "ANALYZED")

P14_FIELDS_INTRODUCED_BY_STATE: dict[str, frozenset[str]] = {
    # None -> PILOT_IN_PROGRESS (revision 0): ONLY these initialization
    # fields may be introduced; every later-phase/failure/analysis field
    # must be absent.
    "PILOT_IN_PROGRESS": frozenset({
        "schema_version", "experiment_id", "campaign_id", "state", "publishable",
        "started_at_utc", "frozen_protocol", "profile_plan_sha256",
    }),
    # PILOT_IN_PROGRESS -> PILOT_COMPLETE (record-pilot).
    "PILOT_COMPLETE": frozenset({
        "pilot_completed_at_utc", "pilot_campaign_reference",
        "preflight_reference_pilot", "provenance",
    }),
    # PILOT_COMPLETE -> PROFILE_IN_PROGRESS (discover-metrics) introduces
    # the first three. A manifest in PROFILE_IN_PROGRESS may also contain
    # case_results/profile_count_completed after one or more case-validation
    # self-loops, so the state-shape layer permits them here; the exact
    # transition-mutation matrix separately forbids them on the entering
    # transition and requires their first appearance on the first self-loop.
    "PROFILE_IN_PROGRESS": frozenset({
        "profile_started_at_utc", "resolved_ncu_metrics", "preflight_reference_profile",
        "case_results", "profile_count_completed",
    }),
    # PROFILE_IN_PROGRESS -> COMPLETE (finalize-profile).
    "COMPLETE": frozenset({
        "profile_completed_at_utc", "profile_order", "artifact_sha256",
    }),
    # COMPLETE -> ANALYZED (analyze). artifact_sha256 itself was already
    # introduced at COMPLETE; analyze() only ever *adds* more entries to it
    # (append-only, enforced by validate_manifest_revision_transition), so
    # it is not repeated here.
    "ANALYZED": frozenset({"analyzed_at_utc"}),
}
# failure_stage/failure_detail: the one pair of fields whose legal
# "introducing state" is not a fixed point in P14_STATE_ORDER at all, but
# "wherever the campaign transitions to FAILED or INTERRUPTED" -- which can
# happen from any non-terminal state. Their own state-based presence rule
# (required together in FAILED/INTERRUPTED and absent otherwise) is already enforced in
# _validate_p14_manifest_document(); no separate introducing-state entry is
# needed above because "the transition's target is FAILED/INTERRUPTED" IS
# their entire legality condition.
P14_FAILURE_ONLY_FIELDS = frozenset({"failure_stage", "failure_detail"})

_P14_FIELD_INTRODUCTION_UNION = frozenset().union(*P14_FIELDS_INTRODUCED_BY_STATE.values()) | P14_FAILURE_ONLY_FIELDS
assert _P14_FIELD_INTRODUCTION_UNION == frozenset(ALLOWED_P14_MANIFEST_KEYS), (
    "every P1.4 manifest field must be bound to a legal state-availability bucket (or the "
    "failure-only exception) in the Task 4 Section 9 matrix -- an unbound field must "
    "never be able to appear at an arbitrary state unnoticed: "
    f"missing={frozenset(ALLOWED_P14_MANIFEST_KEYS) - _P14_FIELD_INTRODUCTION_UNION!r} "
    f"extra={_P14_FIELD_INTRODUCTION_UNION - frozenset(ALLOWED_P14_MANIFEST_KEYS)!r}"
)
assert len(P14_FIELDS_INTRODUCED_BY_STATE) == 5 and len(ALLOWED_P14_STATES) == 7, (
    "the state-availability matrix is documented as exactly 7 rows (one per P1.4 state: the 5 "
    "non-terminal/analysis states above, plus FAILED and INTERRUPTED, whose shared rule is the "
    "P14_FAILURE_ONLY_FIELDS exception rather than a P14_STATE_ORDER entry) -- update this "
    "assertion deliberately if that count ever changes"
)


def _p14_cumulative_allowed_fields(state: str) -> frozenset[str]:
    """Fields permitted to be present (non-None) once a manifest revision's
    own state is `state`, per the mutation matrix above: the union of every
    state's own introduced fields up to and including `state` in the fixed
    linear progression P14_STATE_ORDER. FAILED/INTERRUPTED may retain any
    subset of the fields introduced through COMPLETE -- the campaign can
    fail at any point up to (but never after) ANALYZED, which is terminal
    with no outgoing FAILED/INTERRUPTED edge -- plus the two failure-only
    fields."""
    if state in ("FAILED", "INTERRUPTED"):
        cumulative: frozenset[str] = frozenset()
        for s in P14_STATE_ORDER:
            cumulative = cumulative | P14_FIELDS_INTRODUCED_BY_STATE[s]
            if s == "COMPLETE":
                break
        return cumulative | P14_FAILURE_ONLY_FIELDS
    cumulative = frozenset()
    for s in P14_STATE_ORDER:
        cumulative = cumulative | P14_FIELDS_INTRODUCED_BY_STATE[s]
        if s == state:
            return cumulative
    raise AssertionError(f"unreachable: state {state!r} is not in P14_STATE_ORDER or the failure states")


def validate_manifest_state_shape(current: dict, expected_campaign_id: str) -> list[str]:
    """Validates ONE manifest revision's shape in complete isolation from
    any other revision (Task 4, Section 9). Returns a list of violated-
    invariant messages, empty iff fully compliant. Complementary to
    validate_manifest_revision_transition(): that function checks a
    revision is a legitimate continuation of its predecessor; this function
    checks a revision is well-formed for its own declared state, with no
    knowledge of history at all."""
    errors: list[str] = []
    state = current.get("state")
    if state not in ALLOWED_P14_STATES:
        return [f"state={state!r} is not a recognized P1.4 state"]

    if current.get("campaign_id") != expected_campaign_id:
        errors.append(
            f"campaign_id={current.get('campaign_id')!r} != campaign directory basename "
            f"{expected_campaign_id!r}"
        )

    # Re-audit remediation (Group A): legality depends on key *presence*,
    # never on truthiness or non-null value. Before this fix, a premature key
    # holding an explicit `null` compared equal to "absent" here, so e.g. a
    # revision-0 manifest (state=PILOT_IN_PROGRESS) could carry
    # "profile_completed_at_utc": null / "analyzed_at_utc": null unnoticed.
    # ALLOWED_P14_MANIFEST_KEYS no longer declares any of the fields this
    # loop guards as nullable (see the type table above), so a present value
    # is always the field's real declared type by the time this runs; the
    # only question this loop asks is "is the key present at all".
    allowed = _p14_cumulative_allowed_fields(state)
    for key in ALLOWED_P14_MANIFEST_KEYS:
        present = key in current
        if present and key not in allowed:
            errors.append(
                f"{key} is present but state={state!r} has not yet reached the state that may "
                f"introduce it (Task 4 Section 9 mutation matrix)"
            )

    # Re-audit remediation (Group C): profile_count_completed and
    # case_results must always appear together -- neither may be present
    # without the other. Before this fix, only the *value* relationship
    # (count == len(case_results)) was checked, and only when case_results
    # happened to already be a dict; a manifest with profile_count_completed
    # present and case_results entirely absent (or vice versa) passed
    # unnoticed because the whole comparison lived inside
    # `if isinstance(case_results, dict):`.
    case_results_present = "case_results" in current
    count_present = "profile_count_completed" in current
    if case_results_present != count_present:
        errors.append(
            f"profile_count_completed present={count_present} but case_results "
            f"present={case_results_present} (the two fields must always appear together, "
            f"never one without the other)"
        )

    case_results = current.get("case_results")
    if isinstance(case_results, dict):
        frozen_order = [entry["case_name"] for entry in build_ncu_plan()]
        # The frozen order is checked as an exact ordered list, never
        # reduced to a set comparison: list(case_results) preserves the
        # dict's own JSON-loaded insertion order, so a *reordered* (but
        # otherwise identical) case_results -- same keys, different
        # sequence -- is caught here even though a naive
        # set(case_results) == set(frozen_order[:k]) check would consider
        # the two identical.
        actual_order = list(case_results)
        if actual_order != frozen_order[: len(actual_order)]:
            errors.append(
                f"case_results key order {actual_order!r} is not an exact ordered prefix of the "
                f"frozen six-case order {frozen_order!r} (checked as an ordered list, never a set)"
            )
        if count_present and current["profile_count_completed"] != len(case_results):
            errors.append(
                f"profile_count_completed={current.get('profile_count_completed')!r} != "
                f"len(case_results)={len(case_results)}"
            )

    return errors


def validate_manifest_revision_transition(
    previous: dict | None, current: dict, expected_campaign_id: str,
) -> list[str]:
    """Validates one manifest revision (previous=None: revision 0 in
    isolation) or one adjacent revision-to-revision transition. Never
    trusts campaign_id recorded inside a revision by itself:
    expected_campaign_id must come from the safely resolved campaign
    directory's own basename, supplied by the caller. Returns a list of
    violated-invariant messages, each identifying the specific field or
    rule broken; empty iff fully compliant."""
    errors: list[str] = []

    if current.get("campaign_id") != expected_campaign_id:
        errors.append(
            f"campaign_id={current.get('campaign_id')!r} != campaign directory "
            f"basename {expected_campaign_id!r}"
        )
    if current.get("publishable") is not False:
        errors.append(f"publishable={current.get('publishable')!r} != False (must always be false)")

    curr_state = current.get("state")
    curr_failure_stage = current.get("failure_stage")
    if curr_failure_stage is not None and curr_state not in ("FAILED", "INTERRUPTED"):
        errors.append(
            f"failure_stage={curr_failure_stage!r} is set but state={curr_state!r} "
            f"is neither FAILED nor INTERRUPTED"
        )
    if current.get("analyzed_at_utc") is not None and curr_state != "ANALYZED":
        errors.append(f"analyzed_at_utc is set but state={curr_state!r} != 'ANALYZED'")
    curr_artifact_sha256 = current.get("artifact_sha256")
    if isinstance(curr_artifact_sha256, dict):
        for name in curr_artifact_sha256:
            if name.startswith("analysis/") and curr_state != "ANALYZED":
                errors.append(
                    f"artifact_sha256 contains analysis artifact {name!r} but "
                    f"state={curr_state!r} != 'ANALYZED'"
                )

    if previous is None:
        prev_state = None
    else:
        for field in P14_FIELD_IMMUTABLE:
            if field in previous and previous.get(field) != current.get(field):
                errors.append(
                    f"{field} is immutable but changed: {previous.get(field)!r} -> {current.get(field)!r}"
                )
        for field in P14_FIELD_ALLOWED_TIMESTAMP | P14_FIELD_SET_ONCE:
            prev_value = previous.get(field)
            curr_value = current.get(field)
            if prev_value is not None and curr_value != prev_value:
                errors.append(f"{field} is set-once but changed: {prev_value!r} -> {curr_value!r}")
            if field in previous and prev_value is not None and field not in current:
                errors.append(f"{field} was present in the previous revision but is now absent")

        prev_artifacts = previous.get("artifact_sha256")
        prev_artifacts = prev_artifacts if isinstance(prev_artifacts, dict) else {}
        curr_artifacts = curr_artifact_sha256 if isinstance(curr_artifact_sha256, dict) else {}
        for name, prev_hash in prev_artifacts.items():
            if name not in curr_artifacts:
                errors.append(f"artifact_sha256.{name} was deleted")
            elif curr_artifacts[name] != prev_hash:
                errors.append(
                    f"artifact_sha256.{name} changed after being recorded: "
                    f"{prev_hash!r} -> {curr_artifacts[name]!r}"
                )

        prev_cases = previous.get("case_results")
        prev_cases = prev_cases if isinstance(prev_cases, dict) else {}
        curr_cases = current.get("case_results")
        curr_cases = curr_cases if isinstance(curr_cases, dict) else {}
        for case_name, prev_value in prev_cases.items():
            if case_name not in curr_cases:
                errors.append(f"case_results.{case_name} was deleted")
            elif curr_cases[case_name] != prev_value:
                errors.append(
                    f"case_results.{case_name} was modified after being recorded "
                    f"(existing case_results entries are append-only and immutable)"
                )
        frozen_order = [entry["case_name"] for entry in build_ncu_plan()]

        def _frozen_prefix_len(names: set) -> int | None:
            for k in range(len(frozen_order) + 1):
                if names == set(frozen_order[:k]):
                    return k
            return None

        prev_k = _frozen_prefix_len(set(prev_cases))
        curr_k = _frozen_prefix_len(set(curr_cases))
        if prev_k is None:
            errors.append(f"case_results in the previous revision is not a frozen-order prefix: {sorted(prev_cases)}")
        elif curr_k is None:
            errors.append(
                f"case_results gained an entry out of frozen order: now contains "
                f"{sorted(curr_cases)}, which is not a prefix of {frozen_order!r}"
            )
        elif curr_k < prev_k:
            errors.append(f"case_results shrank from {prev_k} to {curr_k} entrie(s)")

        prev_count = previous.get("profile_count_completed")
        curr_count = current.get("profile_count_completed")
        if isinstance(prev_count, int) and isinstance(curr_count, int) and curr_count < prev_count:
            errors.append(f"profile_count_completed regressed: {prev_count} -> {curr_count}")

        prev_state_for_loop = previous.get("state")
        curr_state_for_loop = current.get("state")
        # Task 4 (Section 9): the ONLY same-state revision the actual
        # workflow ever produces is PROFILE_IN_PROGRESS -> PROFILE_IN_PROGRESS,
        # once per validated case -- it must append *exactly* one new case
        # each time. Without this, neither a same-state "no-op" revision
        # (case_results unchanged) nor a revision that appends two cases at
        # once was previously rejected: curr_k >= prev_k alone (the check
        # above only rejects *shrinking*) permits both.
        if (
            prev_state_for_loop == "PROFILE_IN_PROGRESS" and curr_state_for_loop == "PROFILE_IN_PROGRESS"
            and prev_k is not None and curr_k is not None and curr_k != prev_k + 1
        ):
            errors.append(
                f"a PROFILE_IN_PROGRESS -> PROFILE_IN_PROGRESS revision must append exactly one "
                f"new case result; case_results went from {prev_k} to {curr_k} entrie(s)"
            )

        # Same-state ("self-loop") revisions are only legitimate where
        # ALLOWED_P14_TRANSITIONS itself lists the state as its own valid
        # next state (only PROFILE_IN_PROGRESS does, e.g. appending the next
        # validated profile case -- PILOT_IN_PROGRESS has no self-loop
        # either: the real --pilot workflow never reports incremental
        # progress into the P1.4 manifest); every other state (PILOT_COMPLETE,
        # COMPLETE, ANALYZED, FAILED, INTERRUPTED) has no self-loop entry and
        # so cannot legally repeat, exactly like any other disallowed
        # transition -- there is deliberately no special case for
        # curr_state == prev_state here.
        prev_state = previous.get("state")
        allowed_next_states = ALLOWED_P14_TRANSITIONS.get(prev_state, frozenset())
        if curr_state not in allowed_next_states:
            errors.append(
                f"illegal state transition: {prev_state!r} -> {curr_state!r} "
                f"(allowed next state(s) from {prev_state!r}: {sorted(allowed_next_states)!r})"
            )
        else:
            # Compare key presence as well as value. `dict.get()` would make
            # an absent key and a present key holding None look identical,
            # precisely the class of ambiguity the manifest schema forbids.
            changed_content_fields = {
                key
                for key in set(previous) | set(current)
                if key != "state"
                and (
                    (key in previous) != (key in current)
                    or (
                        key in previous
                        and key in current
                        and previous[key] != current[key]
                    )
                )
            }
            if curr_state in ("FAILED", "INTERRUPTED"):
                expected_changed_fields = P14_FIELD_FAILURE
            else:
                expected_changed_fields = P14_EXACT_TRANSITION_MUTATIONS.get(
                    (prev_state, curr_state)
                )
            if (
                expected_changed_fields is not None
                and changed_content_fields != expected_changed_fields
            ):
                missing_changes = sorted(expected_changed_fields - changed_content_fields)
                unexpected_changes = sorted(changed_content_fields - expected_changed_fields)
                errors.append(
                    f"exact transition mutation matrix violation for "
                    f"{prev_state!r} -> {curr_state!r}: changed content fields "
                    f"{sorted(changed_content_fields)!r}, expected exactly "
                    f"{sorted(expected_changed_fields)!r}; missing changes="
                    f"{missing_changes!r}; unexpected changes={unexpected_changes!r}"
                )

    return errors


# ---------------------------------------------------------------------------
# Chronological timestamp validation (re-audit remediation, Group B). Neither
# validate_manifest_state_shape() nor validate_manifest_revision_transition()
# reject a lifecycle timestamp that moves backwards: the former only checks
# *which* fields may appear in a given state, and the latter only checks that
# an *already-set* timestamp never changes -- neither compares two different
# timestamp fields' parsed values against each other. One reusable validator,
# run alongside the other two semantic layers for revision 0 and every later
# revision alike.
# ---------------------------------------------------------------------------
P14_LIFECYCLE_TIMESTAMP_ORDER: tuple[str, ...] = (
    "started_at_utc", "pilot_completed_at_utc", "profile_started_at_utc",
    "profile_completed_at_utc", "analyzed_at_utc",
)


def validate_manifest_timestamp_chronology(current: dict) -> list[str]:
    """Pure function of one revision's own content: requires the fixed,
    nondecreasing causal order started_at_utc <= pilot_completed_at_utc <=
    profile_started_at_utc <= profile_completed_at_utc <= analyzed_at_utc for
    every one of those fields actually present in `current` (a field's own
    requiredness -- whether it must be present at all in a given state -- is
    already enforced by _validate_p14_manifest_document/
    validate_manifest_state_shape; this only orders whichever subset is
    present). Parses each value with the existing canonical compact-UTC
    validator and compares the parsed datetimes, never the raw strings, so
    e.g. lexical ordering quirks can never matter. Because every P1.4
    manifest revision is a complete snapshot (never a diff), a newly
    introduced timestamp always appears in the very same revision as every
    earlier lifecycle timestamp already set, so a single per-revision
    ordered-pairwise scan is sufficient to catch a newly introduced
    timestamp that precedes an earlier one -- no history/previous-revision
    argument is needed. A malformed value is not re-reported here (the base
    schema validator already rejects it); this function silently skips a
    field it cannot parse rather than raising a second, redundant error."""
    errors: list[str] = []
    parsed: list[tuple[str, _datetime]] = []
    for field in P14_LIFECYCLE_TIMESTAMP_ORDER:
        value = current.get(field)
        if not isinstance(value, str):
            continue
        try:
            p13._validate_compact_timestamp(value, field)
            parsed.append((field, _datetime.strptime(value, "%Y%m%dT%H%M%SZ")))
        except p13.ManifestTransitionError:
            continue
    for (earlier_field, earlier_dt), (later_field, later_dt) in zip(parsed, parsed[1:]):
        if later_dt < earlier_dt:
            errors.append(
                f"{later_field}={current[later_field]!r} precedes {earlier_field}="
                f"{current[earlier_field]!r} (lifecycle timestamps must be nondecreasing: "
                f"{' <= '.join(P14_LIFECYCLE_TIMESTAMP_ORDER)})"
            )
    return errors


MANIFEST_REVISION_RE = re.compile(r"^(\d{6})\.json$")
MANIFEST_REVISION_TMP_NAME = ".manifest_revision.tmp"
MANIFEST_REVISION_KEYS = ("manifest_revision", "previous_manifest_sha256")


def _p14_manifest_dir(campaign_dir: Path) -> Path:
    return campaign_dir / "manifest"


def _manifest_revision_path(campaign_dir: Path, revision: int) -> Path:
    return _p14_manifest_dir(campaign_dir) / f"{revision:06d}.json"


def load_p14_manifest_chain(campaign_dir: Path) -> tuple[dict, int]:
    """Safely reopens, hash-chains, and validates every manifest revision
    under campaign_dir/manifest/. Returns (current_manifest_content,
    highest_valid_revision_index); ({}, -1) if no revision exists yet.

    Raises ManifestTransitionError on any chain defect: an extra, missing,
    reordered, malformed, or symlinked revision; a broken previous-hash
    link; or a revision whose content fails the ordinary P1.4 manifest
    schema (_validate_p14_manifest_document). Every revision is reopened
    and its hash is recomputed fresh from disk on every call -- nothing
    about an earlier revision is ever trusted from memory.

    Descriptor-anchored (Task 4, Section 7/13): manifest/ is opened exactly
    once, with O_DIRECTORY|O_NOFOLLOW relative to the campaign directory,
    and every subsequent enumeration/open/read/hash in this function uses
    that one held descriptor (or a file descriptor opened relative to it) --
    never a second, independent resolution of "campaign_dir/manifest" or
    "campaign_dir/manifest/<name>" by path string. This closes the same
    class of lstat-then-listdir gap Section 7 closes for profiles/: an
    earlier version of this function called
    p13._reject_if_symlink_or_wrong_type(manifest_dir, ...) (one lstat by
    path) and then a *separate* os.listdir(manifest_dir) (a second,
    independent resolution of the same path)."""
    manifest_dir = _p14_manifest_dir(campaign_dir)  # used only for diagnostic message text below
    if not os.path.lexists(manifest_dir):
        return {}, -1

    try:
        campaign_parts = campaign_dir.relative_to(REPO_ROOT).parts
        manifest_fd = _open_dir_component_chain(*campaign_parts, "manifest")
    except p13.UnsafePathError as exc:
        raise p13.ManifestTransitionError(str(exc)) from exc
    try:
        try:
            names = sorted(os.listdir(manifest_fd))
        except OSError as exc:
            raise p13.ManifestTransitionError(f"{manifest_dir}: cannot list revisions: {exc}") from exc
        revision_names = [n for n in names if MANIFEST_REVISION_RE.fullmatch(n)]
        other_names = [n for n in names if n not in revision_names and n != MANIFEST_REVISION_TMP_NAME]
        if other_names:
            raise p13.ManifestTransitionError(
                f"{manifest_dir}: unexpected entries in manifest revision directory: {sorted(other_names)}"
            )
        if not revision_names:
            return {}, -1
        expected_names = [f"{i:06d}.json" for i in range(len(revision_names))]
        if revision_names != expected_names:
            raise p13.ManifestTransitionError(
                f"{manifest_dir}: manifest revisions are not exactly contiguous "
                f"000000..{len(revision_names) - 1:06d}.json: found {revision_names!r}"
            )

        # Never trusts campaign_id recorded inside any revision by itself: the
        # expected value comes from the safely resolved campaign directory's
        # own basename, established once here, before any revision content is
        # read.
        expected_campaign_id = campaign_dir.name

        previous_hash: str | None = None
        previous_content: dict | None = None
        current_content: dict = {}
        for i, name in enumerate(expected_names):
            path = manifest_dir / name  # diagnostic message text only; never reopened by this path
            try:
                st = os.stat(name, dir_fd=manifest_fd, follow_symlinks=False)
                if stat.S_ISLNK(st.st_mode):
                    raise p13.UnsafePathError(f"{path}: is a symlink; refusing")
                if not stat.S_ISREG(st.st_mode):
                    raise p13.UnsafePathError(f"{path}: is not a regular file")
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=manifest_fd)
                try:
                    fst = os.fstat(fd)
                    if (fst.st_dev, fst.st_ino) != (st.st_dev, st.st_ino) or not stat.S_ISREG(fst.st_mode):
                        raise p13.UnsafePathError(f"{path}: changed identity while being opened")
                    raw = b"".join(iter(lambda: os.read(fd, 1 << 20), b""))
                finally:
                    os.close(fd)
            except (OSError, p13.UnsafePathError) as exc:
                raise p13.ManifestTransitionError(f"{path}: cannot load manifest revision: {exc}") from exc
            try:
                text = raw.decode("utf-8")
            except UnicodeError as exc:
                raise p13.ManifestTransitionError(f"{path}: cannot load manifest revision: {exc}") from exc
            try:
                doc = json.loads(text)
            except json.JSONDecodeError as exc:
                raise p13.ManifestTransitionError(f"{path}: invalid JSON: {exc}") from exc
            if not isinstance(doc, dict):
                raise p13.ManifestTransitionError(f"{path}: revision root is not a JSON object")
            if doc.get("manifest_revision") != i:
                raise p13.ManifestTransitionError(
                    f"{path}: manifest_revision={doc.get('manifest_revision')!r} != expected {i}"
                )
            expected_previous_hash = None if i == 0 else previous_hash
            if doc.get("previous_manifest_sha256") != expected_previous_hash:
                raise p13.ManifestTransitionError(
                    f"{path}: previous_manifest_sha256={doc.get('previous_manifest_sha256')!r} != "
                    f"expected {expected_previous_hash!r} (hash chain broken or tampered)"
                )
            content = {k: v for k, v in doc.items() if k not in MANIFEST_REVISION_KEYS}
            try:
                _validate_p14_manifest_document(content)
            except p13.ManifestTransitionError as exc:
                raise p13.ManifestTransitionError(f"{path}: {exc}") from exc
            # Three explicit semantic validation layers (Task 4, Section 9;
            # chronology added by the re-audit remediation, Group B), run for
            # revision 0 and every later revision alike: shape validation (is
            # this revision well-formed for its own declared state, taken in
            # isolation -- e.g. no field introduced before the one state that
            # may legally introduce it), transition validation (is this
            # revision a legitimate continuation of the previous one), and
            # chronology validation (do this revision's own lifecycle
            # timestamps agree with each other on a single nondecreasing
            # causal order). The hash chain alone only proves a revision was
            # appended without altering an earlier byte; none of the three
            # says anything the others already check, and together they are
            # what "semantically valid" means here.
            shape_errors = validate_manifest_state_shape(content, expected_campaign_id)
            if shape_errors:
                raise p13.ManifestTransitionError(
                    f"{path}: manifest state-shape validation failed: {shape_errors}"
                )
            transition_errors = validate_manifest_revision_transition(
                previous_content, content, expected_campaign_id,
            )
            if transition_errors:
                raise p13.ManifestTransitionError(
                    f"{path}: semantic transition validation failed: {transition_errors}"
                )
            chronology_errors = validate_manifest_timestamp_chronology(content)
            if chronology_errors:
                raise p13.ManifestTransitionError(
                    f"{path}: manifest timestamp chronology validation failed: {chronology_errors}"
                )
            previous_content = content
            current_content = content
            # Hashed from the exact same bytes already read via the held fd
            # above -- never a second, independent p13.sha256_of(path) open.
            previous_hash = hashlib.sha256(raw).hexdigest()

        return current_content, len(expected_names) - 1
    finally:
        os.close(manifest_fd)


def write_next_p14_manifest_revision(campaign_dir: Path, manifest_content: dict) -> dict:
    """Appends one new, immutable manifest revision under campaign_dir/manifest/.

    Never overwrites an existing revision and never calls os.replace(): the
    new revision is written to a fixed-name temporary via exclusive, no-follow
    creation, then published as the next contiguous revision number via
    hard-link-then-unlink no-clobber (mirrors every other P1.4 result
    artifact). Re-derives the next revision number and the previous
    revision's hash by re-reading the chain from disk immediately before
    writing, so a concurrent publication attempt for the same next revision
    fails closed at the final hard-link step (only one writer's os.link()
    can ever succeed for a given revision path); an interrupted write leaves
    only a harmless, unrecognized temporary name, never a partial or
    replaced revision."""
    manifest_dir = _p14_manifest_dir(campaign_dir)
    p13._mkdir_component(manifest_dir, must_not_exist=False)
    current, current_revision = load_p14_manifest_chain(campaign_dir)
    new_revision = current_revision + 1
    previous_hash = None
    if current_revision >= 0:
        previous_hash = p13.sha256_of(_manifest_revision_path(campaign_dir, current_revision))

    doc = dict(manifest_content)
    doc["manifest_revision"] = new_revision
    doc["previous_manifest_sha256"] = previous_hash
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"

    tmp_path = manifest_dir / MANIFEST_REVISION_TMP_NAME
    if os.path.lexists(tmp_path):
        raise p13.UnsafePathError(f"{tmp_path}: stale manifest-revision temporary already exists")

    # Defense in depth (Remediation C, extended by Task 4/Section 9 to both
    # semantic layers): re-validate the candidate revision's shape,
    # transition, and timestamp chronology here too, before ever writing it,
    # using the exact same functions load_p14_manifest_chain re-applies on
    # every future read. This never replaces the read-side check (an
    # attacker could always bypass this function and write a revision file
    # directly), but it does catch a bug in this codebase's own callers
    # before it is ever persisted. Runs after the stale-tmp check above: a
    # leftover temporary from an interrupted write is a filesystem-hygiene
    # defect independent of the new content's own semantic validity, and
    # should fail with that more specific diagnosis first.
    shape_errors = validate_manifest_state_shape(manifest_content, campaign_dir.name)
    if shape_errors:
        raise p13.ManifestTransitionError(
            f"refusing to write a manifest revision with an invalid shape: {shape_errors}"
        )
    transition_errors = validate_manifest_revision_transition(
        current if current_revision >= 0 else None, manifest_content, campaign_dir.name,
    )
    if transition_errors:
        raise p13.ManifestTransitionError(
            f"refusing to write a non-compliant manifest revision: {transition_errors}"
        )
    chronology_errors = validate_manifest_timestamp_chronology(manifest_content)
    if chronology_errors:
        raise p13.ManifestTransitionError(
            f"refusing to write a manifest revision with invalid timestamp chronology: {chronology_errors}"
        )

    final_path = _manifest_revision_path(campaign_dir, new_revision)
    try:
        with p13._open_exclusive(tmp_path, binary=False) as handle:
            handle.write(text)
    except Exception:
        if os.path.lexists(tmp_path):
            p13._safe_unlink_owned(tmp_path)
        raise
    try:
        p13._publish_no_clobber(tmp_path, final_path)
    except p13.UnsafePathError:
        if os.path.lexists(tmp_path):
            p13._safe_unlink_owned(tmp_path)
        raise
    return manifest_content


def p14_merge_manifest(campaign_dir: Path, updates: dict, state: str) -> dict:
    """Merges `updates` into the P1.4 manifest and sets `state`, enforcing
    the P1.4 state machine (ALLOWED_P14_TRANSITIONS): a terminal campaign
    can never be reopened/rewritten, and every key in `updates` must be
    allowlisted with the correct type. Publication uses the append-only,
    immutable revision chain (write_next_p14_manifest_revision) -- never
    p13.write_manifest_atomic / os.replace()."""
    _validate_p14_manifest_updates(updates)
    manifest, _revision = load_p14_manifest_chain(campaign_dir)
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
    write_next_p14_manifest_revision(campaign_dir, manifest)
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
# NCU 2025.4's `--page raw --csv --print-units base
# --print-kernel-base function` output is a *wide* table. The first row
# contains launch metadata columns (including literal "ID" and "Kernel
# Name") followed by one column per collected metric; the second row
# contains the unit for each metric column; every later row is one profiled
# launch. This is distinct from the Details-page long form whose rows carry
# "Metric Name", "Metric Unit", and "Metric Value" fields.
#
# The parser below is fail-closed: it requires unique columns, the two
# launch-identity columns, one unit row, exactly one launch row, non-empty
# launch/kernel identities, and one unambiguous column for every candidate
# metric present. Candidate values must be finite and non-negative and
# candidate units must match EXPECTED_METRIC_UNITS exactly after only
# case/whitespace normalization. Canonical and namespace-qualified GB300
# metric spellings are associated only through
# canonical_candidate_metric_name(); two columns mapping to the same
# canonical candidate are rejected as ambiguous. No unit scaling,
# aggregation, substring match, or structural fallback is permitted.
# ---------------------------------------------------------------------------
class NcuCsvParseError(ValueError):
    pass


REQUIRED_NCU_CSV_COLUMNS = ("ID", "Kernel Name")

# NCU 2025.4 unit spellings observed in the live GB300 raw-page export
# canary on 2026-07-29. Keep the exact profiler spellings here: in
# particular, `--print-units base` emits "ns", not "nsecond", for
# gpu__time_duration.sum.
EXPECTED_METRIC_UNITS = {
    "dram__bytes_read.sum": "byte",
    "dram__bytes_write.sum": "byte",
    "lts__t_bytes.sum": "byte",
    "gpu__time_duration.sum": "ns",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "%",
}


def _parse_ncu_raw_csv_rows(rows_raw: list[list[str]], *, label: str) -> dict:
    """Pure-data core of parse_ncu_raw_csv(): validates and parses an
    already-read NCU `--page raw --csv` export given as a list of row
    lists. `label` is used only for diagnostic messages (a path when the
    caller read from a path, or a descriptor-anchored "profiles/<case>/..."
    description when the caller read from an already-open fd -- see
    parse_ncu_raw_csv() and _reconstruct_case_result_from_fds()). Returns a
    dict with keys 'metrics' (actual metric-column name -> float value),
    'units' (actual metric-column name -> the exact unit string recorded for
    it), 'candidate_metric_names' (canonical candidate -> actual column
    name), 'kernel_name', and 'launch_id'. Raises NcuCsvParseError on any
    defect -- see the module comment above for the complete contract."""
    if not rows_raw:
        raise NcuCsvParseError(f"{label}: empty file (no header)")

    header = rows_raw[0]
    if not header:
        raise NcuCsvParseError(f"{label}: empty header row")
    if len(header) != len(set(header)):
        duplicates = sorted({h for h in header if header.count(h) > 1})
        raise NcuCsvParseError(f"{label}: duplicate header column name(s): {duplicates}")
    missing_columns = [c for c in REQUIRED_NCU_CSV_COLUMNS if c not in header]
    if missing_columns:
        raise NcuCsvParseError(
            f"{label}: missing required column(s) {missing_columns} in header {header!r}"
        )
    col_index = {name: header.index(name) for name in REQUIRED_NCU_CSV_COLUMNS}

    if len(rows_raw) < 2:
        raise NcuCsvParseError(f"{label}: missing metric-unit row")
    unit_row = rows_raw[1]
    if len(unit_row) != len(header):
        raise NcuCsvParseError(
            f"{label}: line 2 (metric units): expected {len(header)} field(s), "
            f"got {len(unit_row)}"
        )

    data_rows = rows_raw[2:]
    if len(data_rows) != 1:
        raise NcuCsvParseError(
            f"{label}: found {len(data_rows)} profiled launch row(s), expected exactly 1 "
            f"(--launch-count 1 should guarantee this)"
        )
    data_row = data_rows[0]
    if len(data_row) != len(header):
        raise NcuCsvParseError(
            f"{label}: line 3 (profiled launch): expected {len(header)} field(s), "
            f"got {len(data_row)}"
        )

    launch_id = data_row[col_index["ID"]].strip()
    kernel_name = data_row[col_index["Kernel Name"]].strip()
    if not launch_id:
        raise NcuCsvParseError(f"{label}: line 3: empty launch ID")
    if not kernel_name:
        raise NcuCsvParseError(f"{label}: line 3: empty kernel name")

    metrics: dict[str, float] = {}
    units: dict[str, str] = {}
    candidate_metric_names: dict[str, str] = {}
    for column_index, metric_name_raw in enumerate(header):
        metric_name = metric_name_raw.strip()
        canonical_metric_name = canonical_candidate_metric_name(metric_name)
        if canonical_metric_name is None:
            continue
        if canonical_metric_name in candidate_metric_names:
            previous = candidate_metric_names[canonical_metric_name]
            raise NcuCsvParseError(
                f"{label}: candidate metric {canonical_metric_name!r} is represented by "
                f"more than one column ({previous!r}, {metric_name!r}); refusing an "
                f"ambiguous canonical/qualified mapping"
            )

        metric_unit = unit_row[column_index].strip()
        raw_value = data_row[column_index].strip()
        if not metric_unit:
            raise NcuCsvParseError(
                f"{label}: line 2: empty metric unit for candidate column {metric_name!r}"
            )
        if not raw_value:
            raise NcuCsvParseError(
                f"{label}: line 3: empty metric value for candidate column {metric_name!r}"
            )

        try:
            value = float(raw_value)
        except ValueError as exc:
            raise NcuCsvParseError(
                f"{label}: line 3: metric {metric_name!r} value {raw_value!r} is not a number"
            ) from exc
        if not math.isfinite(value):
            raise NcuCsvParseError(
                f"{label}: line 3: metric {metric_name!r} value {raw_value!r} is not finite"
            )
        if value < 0:
            raise NcuCsvParseError(
                f"{label}: line 3: metric {metric_name!r} value {value!r} is negative"
            )

        expected_unit = EXPECTED_METRIC_UNITS.get(canonical_metric_name)
        assert expected_unit is not None
        if metric_unit.strip().lower() != expected_unit.strip().lower():
            raise NcuCsvParseError(
                f"{label}: line 2: metric {metric_name!r} has unit {metric_unit!r}, "
                f"expected exactly {expected_unit!r} (base units are matched case/whitespace-"
                f"insensitively but never scaled or converted)"
            )

        metrics[metric_name] = value
        units[metric_name] = metric_unit
        candidate_metric_names[canonical_metric_name] = metric_name

    return {
        "metrics": metrics,
        "units": units,
        "candidate_metric_names": candidate_metric_names,
        "kernel_name": kernel_name,
        "launch_id": launch_id,
        "launch_count": 1,
    }


def parse_ncu_raw_csv(path: Path) -> dict:
    """Path-based entry point: reads path (symlink-safe, non-empty-regular
    only) and hands the parsed rows to _parse_ncu_raw_csv_rows(). Used by
    validate-profile-case's one-shot validation of freshly-produced
    evidence via reconstruct_case_result()."""
    try:
        with p13._open_regular_nofollow(path, binary=False) as handle:
            rows_raw = list(csv.reader(handle))
    except (OSError, p13.UnsafePathError, UnicodeError) as exc:
        raise NcuCsvParseError(f"{path}: unable to read: {exc}") from exc
    return _parse_ncu_raw_csv_rows(rows_raw, label=str(path))


# ---------------------------------------------------------------------------
# Descriptor-anchored directory/file resolution (Task 4, Section 7). Every
# P1.4-owned raw-tree *read* used for a terminal decision (profiles/'s own
# inventory, and each case's application/metrics/.ncu-rep evidence) must
# hash/parse the exact bytes behind an already-open file descriptor, never
# re-open a pathname after a descriptor for it (or an ancestor of it) is
# already held. This mirrors scripts/p14_safe_capture.py's own
# O_DIRECTORY|O_NOFOLLOW discipline for writes, applied here to reads; it is
# intentionally duplicated in miniature rather than imported, since
# p14_safe_capture.py itself imports this module (importing it back would
# be circular) and the primitive is a handful of lines.
# ---------------------------------------------------------------------------
def _open_dir_nofollow_p14(name: str, *, dir_fd: int | None) -> int:
    flags = os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        raise p13.UnsafePathError(f"{name}: cannot open as a non-symlink directory: {exc}") from exc


def _open_dir_component_chain(*parts: str) -> int:
    """Opens REPO_ROOT/parts[0]/parts[1]/... as a directory, one component
    at a time (including the repository root itself), each via
    O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC relative to the previously opened
    descriptor -- never re-resolving a pathname afterward, so nothing that
    happens to an intermediate *name* after this call returns can affect
    the returned, already-open descriptor. Caller must os.close() the
    result."""
    fd = _open_dir_nofollow_p14(str(REPO_ROOT), dir_fd=None)
    try:
        for part in parts:
            next_fd = _open_dir_nofollow_p14(part, dir_fd=fd)
            os.close(fd)
            fd = next_fd
    except Exception:
        os.close(fd)
        raise
    return fd


def _open_profiles_fd_anchored(campaign_dir: Path) -> int:
    """Descriptor-anchored, no-follow open of campaign_dir/profiles, one
    path component at a time from the repository root. Caller must
    os.close() the result."""
    campaign_parts = campaign_dir.relative_to(REPO_ROOT).parts
    return _open_dir_component_chain(*campaign_parts, "profiles")


def _open_case_evidence_fds(profiles_fd: int, case_name: str) -> dict[str, int]:
    """Opens profiles_fd/<case_name>/ (O_DIRECTORY|O_NOFOLLOW) and then each
    of the three evidence files relative to *that* case descriptor
    (O_RDONLY|O_NOFOLLOW), never re-resolving a pathname. Verifies each
    evidence file is a non-symlink, non-empty regular file at open time.
    Returns a dict of open fds (including the case directory itself, under
    key "_case_dir") the caller must close; raises p13.UnsafePathError
    (closing any fd already opened in this call first) on any failure."""
    case_fd = None
    try:
        case_fd = _open_dir_nofollow_p14(case_name, dir_fd=profiles_fd)
    except p13.UnsafePathError as exc:
        raise p13.UnsafePathError(f"profiles/{case_name}: {exc}") from exc
    fds: dict[str, int] = {"_case_dir": case_fd}
    try:
        for label, filename in (
            ("application_csv", f"{case_name}.application.csv"),
            ("metrics_csv", f"{case_name}.metrics_raw.csv"),
            ("ncu_rep", f"{case_name}_report.ncu-rep"),
        ):
            file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
            try:
                fd = os.open(filename, file_flags, dir_fd=case_fd)
            except OSError as exc:
                raise p13.UnsafePathError(f"profiles/{case_name}/{filename}: cannot open: {exc}") from exc
            try:
                st = os.fstat(fd)
            except OSError:
                os.close(fd)
                raise
            if not stat.S_ISREG(st.st_mode) or st.st_size == 0:
                os.close(fd)
                raise p13.UnsafePathError(f"profiles/{case_name}/{filename}: not a non-empty regular file")
            fds[label] = fd
    except Exception:
        for fd in fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        raise
    return fds


def _sha256_of_fd(fd: int) -> str:
    """Hashes the exact bytes behind an already-open, read-only fd; never
    re-opens by name. Seeks to the start first so this can safely run more
    than once (or after _read_csv_rows_from_fd()) against the same fd."""
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _read_csv_rows_from_fd(fd: int) -> list[list[str]]:
    """Parses the exact bytes behind an already-open fd as CSV; never
    re-opens by name. Operates on a dup()'d fd so the text-mode wrapper's
    own close() (at the end of the "with" block) never closes the caller's
    original fd, which the caller may still need (e.g. to hash afterward)."""
    os.lseek(fd, 0, os.SEEK_SET)
    dup_fd = os.dup(fd)
    with os.fdopen(dup_fd, "r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


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
        # stats_rows is deliberately CSV-shaped, so this field is a string.
        # Never truth-test it: both "ok" and "REVIEW" are non-empty.
        stability_review = row["stability_review"]
        if stability_review not in {"ok", "REVIEW"}:
            raise ValueError(
                "pilot-statistics stability_review must be exactly 'ok' or "
                f"'REVIEW', got {stability_review!r}"
            )
        lines.append(
            f"| {row['method']} | {row['stages']} | {row['bytes_in_flight_kib']} | "
            f"{row['median_gbps']:.3f} | [{row['median_ci_low_gbps']:.3f}, "
            f"{row['median_ci_high_gbps']:.3f}] | {row['cv_percent']:.2f} | "
            f"{stability_review} | {row['iqr_flagged_count']} |"
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
    profile_plan_path = write_profile_plan(campaign_dir, ncu_plan)
    # Recorded once, here, as the trusted reference hash the central evidence-
    # integrity gate (verify_campaign_evidence_integrity) re-derives and
    # compares profile_plan.csv against before COMPLETE and before ANALYZED.
    profile_plan_sha256 = p13.sha256_of(profile_plan_path)
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
        "profile_plan_sha256": profile_plan_sha256,
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
        manifest, _revision = load_p14_manifest_chain(campaign_dir)
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
        manifest, _revision = load_p14_manifest_chain(campaign_dir)
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
def canonical_candidate_metric_name(metric_name: str) -> str | None:
    """Maps an exact or namespace-qualified NCU name to one frozen candidate.

    NCU 2025.4 on GB300 prints profiling metric identifiers such as
    ``FBSP.TriageCompute.dram__bytes_read.sum``. Older/synthetic output uses
    the canonical ``dram__bytes_read.sum`` form. A qualified identifier is a
    match only when it ends with a dot followed by the complete canonical
    candidate; arbitrary substring matches are never accepted.
    """
    if metric_name in CANDIDATE_METRICS:
        return metric_name
    matches = [
        candidate
        for candidate in CANDIDATE_METRICS
        if metric_name.endswith(f".{candidate}")
    ]
    return matches[0] if len(matches) == 1 else None


def parse_metric_discovery_log(path: Path) -> set[str]:
    """Parses `ncu --query-metrics --query-metrics-mode all --devices 0`.

    NCU may emit either one bare identifier per line or a table whose first
    whitespace-delimited column is the full identifier, followed by type,
    unit, and description columns. Preserve the exact identifier NCU reports
    so the same name can later be passed back to ``--metrics``.
    """
    try:
        with p13._open_regular_nofollow(path, binary=False) as handle:
            text = handle.read()
    except (OSError, p13.UnsafePathError, UnicodeError) as exc:
        raise ValueError(f"{path}: unable to read: {exc}") from exc
    names: set[str] = set()
    for line in text.splitlines():
        fields = line.split(maxsplit=1)
        if not fields:
            continue
        token = fields[0]
        if "__" in token:
            names.add(token)
    return names


def resolve_ncu_metrics(discovered: set[str]) -> dict:
    selected: dict[str, str] = {}
    ambiguous: dict[str, list[str]] = {}
    for candidate in CANDIDATE_METRICS:
        matches = sorted(
            name
            for name in discovered
            if canonical_candidate_metric_name(name) == candidate
        )
        if candidate in matches:
            # Prefer the canonical identifier if NCU reports both canonical
            # and qualified spellings for the same semantic metric.
            selected[candidate] = candidate
        elif len(matches) == 1:
            selected[candidate] = matches[0]
        elif len(matches) > 1:
            ambiguous[candidate] = matches

    if ambiguous:
        detail = "; ".join(
            f"{candidate}: {matches!r}"
            for candidate, matches in sorted(ambiguous.items())
        )
        raise ValueError(
            "metric discovery returned multiple qualified identifiers for "
            f"the same frozen candidate; refusing to guess ({detail})"
        )

    resolved = [
        selected[candidate]
        for candidate in CANDIDATE_METRICS
        if candidate in selected
    ]
    missing = [
        candidate
        for candidate in CANDIDATE_METRICS
        if candidate not in selected
    ]
    dram_read_metric = selected.get(MANDATORY_DRAM_METRIC)
    return {
        "requested": list(CANDIDATE_METRICS),
        "resolved": resolved,
        "missing": missing,
        "dram_read_metric": dram_read_metric,
        "dram_read_metric_available": dram_read_metric is not None,
    }


def _do_discover_metrics(
    *, campaign_dir: Path, discovery_log: Path, preflight_path: Path,
    git_commit: str, started_at_utc: str, now_utc: _datetime,
) -> tuple[bool, list[str], dict | None]:
    try:
        manifest, _revision = load_p14_manifest_chain(campaign_dir)
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

    # Audit blocker #1: a profiling preflight from a different GPU/driver/
    # commit than the one the pilot ran on must never be silently accepted,
    # regardless of whether the bash orchestrator already ran
    # validate-profile-preconditions first -- this is the hard gate at the
    # exact point preflight_reference_profile is recorded and the campaign
    # commits to PROFILE_IN_PROGRESS.
    if not preflight_errors:
        pilot_preflight_snapshot = manifest.get("preflight_reference_pilot")
        if not isinstance(pilot_preflight_snapshot, dict):
            errors.append("manifest preflight_reference_pilot is missing or incomplete")
        else:
            errors.extend(compare_preflight_provenance(pilot_preflight_snapshot, preflight_snapshot))

    resolved = None
    try:
        discovered = parse_metric_discovery_log(discovery_log)
        resolved = resolve_ncu_metrics(discovered)
    except ValueError as exc:
        errors.append(str(exc))

    if errors:
        success, fail_errors = _fail_p14(campaign_dir, "profile_start", errors)
        return success, fail_errors, None

    assert resolved is not None
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
# Subcommand: validate-profile-preconditions (Remediation A, audit blocker #1)
#
# GPU-free, side-effect-free precondition gate for entering the profiling
# phase: confirms the campaign is PILOT_COMPLETE, validates a freshly
# captured preflight snapshot, and fail-closed compares it against the
# pilot-phase preflight already recorded in the manifest. Writes nothing --
# run_exp01_memory_paths_p14.sh calls this immediately after capturing the
# fresh preflight and before any Docker/NCU invocation or raw-tree log write
# (Remediation D), so a GPU/driver swap between the pilot and profiling runs
# aborts before any expensive or irreversible work happens. discover-metrics
# also runs the same compare_preflight_provenance check itself as a hard gate
# at the point preflight_reference_profile is actually recorded, so this
# subcommand is an early, cheap check -- not the only enforcement point.
# ---------------------------------------------------------------------------
def _do_validate_profile_preconditions(
    *, campaign_dir: Path, preflight_path: Path, git_commit: str, now_utc: _datetime,
) -> tuple[bool, list[str]]:
    try:
        manifest, _revision = load_p14_manifest_chain(campaign_dir)
        _validate_p14_manifest_document(manifest, require_initialized=True)
    except (p13.ManifestTransitionError, p13.UnsafePathError) as exc:
        return False, [f"P1.4 manifest: {exc}"]
    if manifest.get("state") != "PILOT_COMPLETE":
        return False, [
            f"P1.4 manifest state={manifest.get('state')!r} != 'PILOT_COMPLETE'; "
            f"cannot check profiling preconditions"
        ]
    pilot_snapshot = manifest.get("preflight_reference_pilot")
    if not isinstance(pilot_snapshot, dict):
        return False, ["manifest preflight_reference_pilot is missing or incomplete"]

    preflight_errors, profile_snapshot = validate_preflight_file(
        preflight_path, expected_git_commit=git_commit, now_utc=now_utc,
    )
    if preflight_errors:
        return False, [f"preflight: {e}" for e in preflight_errors]

    provenance_errors = compare_preflight_provenance(pilot_snapshot, profile_snapshot)
    if provenance_errors:
        return False, provenance_errors
    return True, []


def cmd_validate_profile_preconditions(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_p14_campaign_dir(args.campaign_dir)
    except p13.UnsafePathError as exc:
        print(f"analyze_exp01_memory_paths_p14: validate-profile-preconditions: ERROR: {exc}", file=sys.stderr)
        return 2
    now_utc = _datetime.now(_timezone.utc)
    if args.now is not None:
        try:
            now_utc = parse_now_arg(args.now)
        except ValueError as exc:
            print(f"analyze_exp01_memory_paths_p14: validate-profile-preconditions: ERROR: {exc}", file=sys.stderr)
            return 2
    ok, errors = _do_validate_profile_preconditions(
        campaign_dir=campaign_dir, preflight_path=Path(args.preflight),
        git_commit=args.git_commit, now_utc=now_utc,
    )
    if not ok:
        print("analyze_exp01_memory_paths_p14: validate-profile-preconditions: ERROR:", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp01_memory_paths_p14: validate-profile-preconditions:   - {error}", file=sys.stderr)
        return 1
    print("analyze_exp01_memory_paths_p14: validate-profile-preconditions: OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Canonical case-result reconstruction (Remediation D, second audit,
# blocker D). The single function that reconstructs one profile case's
# complete, evidence-derived result from its authoritative sources alone --
# the frozen profile-plan row, the application CSV, the NCU raw metrics
# CSV, the .ncu-rep's own hash, campaign provenance, the resolved-metric
# definition, and the fixed HBM-classification rule -- never from a stored
# value. validate-profile-case calls this once to build the result it
# records; the central evidence-integrity gate calls it again, fresh from
# disk, immediately before both COMPLETE and ANALYZED, and compares the
# result to what is currently recorded as a complete typed structure (every
# key, not a hand-picked subset) so a change to any single derived field --
# including one nothing else happens to depend on, such as
# dram_read_bytes or a resolved_metric_values entry -- can never go
# undetected merely because some other field still happens to agree.
# ---------------------------------------------------------------------------
def p14_validate_case_rows(rows: list[list[str]], expect: dict, *, label: str) -> tuple[list[dict[str, str]], list[str]]:
    """P1.4-local adapter mirroring p13.validate_case_file()'s own
    validation body exactly (header/row-shape/per-field/sample-index
    checks via the frozen, unmodified p13.CSV_HEADER/p13.FIELD_VALIDATORS),
    but operating on already-parsed `rows` (e.g. from
    _read_csv_rows_from_fd()) instead of opening a path itself.
    scripts/aggregate_exp01_memory_paths.py only exposes a Path-based
    validate_case_file(); this adapter exists so the descriptor-anchored
    evidence layer (Task 4, Section 7) can validate bytes read from an
    already-open fd without ever reopening a pathname, while never
    modifying or weakening the frozen P1.3 module itself."""
    errors: list[str] = []
    if not rows:
        return [], [f"{label}: empty file (no header)"]
    header, data_rows_raw = rows[0], rows[1:]
    if header != p13.CSV_HEADER:
        return [], [f"{label}: header mismatch (wrong or reordered CSV header): {header!r}"]

    parsed_rows: list[dict[str, str]] = []
    for line_no, row in enumerate(data_rows_raw, start=2):
        if len(row) == 0:
            errors.append(f"{label}: line {line_no}: blank row")
            continue
        if row == p13.CSV_HEADER:
            errors.append(f"{label}: line {line_no}: repeated header row")
            continue
        if len(row) != len(p13.CSV_HEADER):
            errors.append(f"{label}: line {line_no}: expected {len(p13.CSV_HEADER)} fields, got {len(row)}")
            continue
        parsed_rows.append(dict(zip(p13.CSV_HEADER, row)))

    if errors:
        return parsed_rows, errors

    repetitions = expect["repetitions"]
    if len(parsed_rows) != repetitions:
        errors.append(f"{label}: has {len(parsed_rows)} data row(s), expected exactly repetitions={repetitions}")
        return parsed_rows, errors

    sample_indices: list[int] = []
    for row_num, row in enumerate(parsed_rows):
        ctx = f"{label}: data row {row_num} (line {row_num + 2})"
        validated_fields: set[str] = set()
        for field in p13.CSV_HEADER:
            validator = p13.FIELD_VALIDATORS[field]
            value = validator(row, expect, errors, ctx)
            validated_fields.add(field)
            if field == "sample_index" and value is not None:
                sample_indices.append(value)
        assert validated_fields == set(p13.CSV_HEADER), (
            f"internal error: row {row_num} did not run every field validator"
        )

    index_counts: dict[int, int] = {}
    for value in sample_indices:
        index_counts[value] = index_counts.get(value, 0) + 1
    expected_indices = set(range(repetitions))
    found_indices = set(index_counts)
    for missing in sorted(expected_indices - found_indices):
        errors.append(f"{label}: sample_index={missing} is missing")
    for unexpected in sorted(found_indices - expected_indices):
        errors.append(f"{label}: unexpected sample_index={unexpected} (expected 0..{repetitions - 1})")
    for value, count in sorted(index_counts.items()):
        if count > 1:
            errors.append(f"{label}: sample_index={value} appears {count} times, expected exactly once")

    return parsed_rows, errors


def _reconstruct_case_result_core(
    *, entry: dict, application_rows: list[list[str]], metrics_rows: list[list[str]],
    application_label: str, metrics_label: str,
    application_hash: str, metrics_hash: str, ncu_rep_hash: str,
    provenance: dict, resolved_ncu_metrics: dict, git_commit: str,
) -> tuple[dict | None, list[str]]:
    """Pure-data canonical core (Task 4, Section 7/8): every input is
    already-read content (rows, hashes) rather than a path or fd, so this
    one function is the single place both entry points below (path-based
    and descriptor-anchored) converge -- and the single place
    validate-profile-case, finalize-profile, and analyze ultimately share,
    since the central evidence-integrity gate calls the descriptor-anchored
    entry point immediately before both COMPLETE and ANALYZED."""
    errors: list[str] = []

    expect = {
        "method": entry["method"], "stages": entry["stages"], "bif_kib": entry["bif_kib"],
        "run_kind": FROZEN_PROFILE_PARAMS["run_kind"], "repetitions": FROZEN_PROFILE_PARAMS["repetitions"],
        "passes": FROZEN_PROFILE_PARAMS["passes"], "warmup_ms": FROZEN_PROFILE_PARAMS["warmup_ms"],
        "working_set_mib": FROZEN_PROFILE_PARAMS["working_set_mib"], "git_commit": git_commit,
    }
    app_rows, app_errors = p14_validate_case_rows(application_rows, expect, label=application_label)
    errors.extend(app_errors)
    app_row = app_rows[0] if (app_rows and not app_errors) else None

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
        parsed_metrics = _parse_ncu_raw_csv_rows(metrics_rows, label=metrics_label)
    except NcuCsvParseError as exc:
        errors.append(str(exc))

    resolved_names = resolved_ncu_metrics.get("resolved", [])
    resolved_to_actual: dict[str, str] = {}
    if parsed_metrics is not None:
        # Exact function-name match only -- never substring/regex matching.
        # The only accepted values are the two frozen benchmark kernel names.
        if parsed_metrics["kernel_name"] != entry["kernel_name"]:
            errors.append(
                f"metrics CSV kernel name {parsed_metrics['kernel_name']!r} != expected exact "
                f"function name {entry['kernel_name']!r} (no substring/regex matching is permitted)"
            )
        # Discovery and raw-page export may spell the same GB300 counter as
        # either its canonical identifier or a namespace-qualified identifier.
        # Associate them only through the exact suffix mapping shared with
        # discovery; the parser has already rejected two raw columns mapping
        # to the same candidate. A resolved semantic metric that is absent,
        # duplicated, malformed, or wrong-unit remains an integrity failure --
        # never silently reclassified as INCONCLUSIVE.
        for resolved_metric in resolved_names:
            canonical_name = canonical_candidate_metric_name(resolved_metric)
            actual_name = (
                parsed_metrics["candidate_metric_names"].get(canonical_name)
                if canonical_name is not None
                else None
            )
            if actual_name is None:
                errors.append(
                    f"metrics CSV is missing resolved metric {resolved_metric!r} "
                    f"(recorded as resolved during metric discovery; this is an evidence "
                    f"integrity failure, not an INCONCLUSIVE HBM classification)"
                )
            else:
                resolved_to_actual[resolved_metric] = actual_name

    if errors:
        return None, errors

    dram_available = bool(resolved_ncu_metrics.get("dram_read_metric_available"))
    dram_metric_name = resolved_ncu_metrics.get("dram_read_metric")
    dram_actual_name = resolved_to_actual.get(dram_metric_name) if dram_available else None
    dram_read_bytes = parsed_metrics["metrics"].get(dram_actual_name) if dram_actual_name else None
    useful_bytes = float(app_row["useful_bytes"])
    classification, flags, ratio = classify_hbm(dram_read_bytes, useful_bytes)

    case_result = {
        "case_name": entry["case_name"],
        "method": entry["method"],
        "stages": entry["stages"],
        "bytes_in_flight_kib": entry["bif_kib"],
        "useful_bytes": app_row["useful_bytes"],
        "launch_id": parsed_metrics["launch_id"],
        "launch_count": parsed_metrics["launch_count"],
        "dram_read_bytes": dram_read_bytes,
        "dram_read_ratio": ratio,
        "hbm_classification": classification,
        "diagnostic_flags": list(flags),
        "resolved_metric_values": {
            m: parsed_metrics["metrics"].get(resolved_to_actual[m])
            for m in resolved_names
        },
        "resolved_metric_units": {
            m: parsed_metrics["units"].get(resolved_to_actual[m])
            for m in resolved_names
        },
        "application_csv_sha256": application_hash,
        "metrics_csv_sha256": metrics_hash,
        "ncu_rep_sha256": ncu_rep_hash,
    }
    return case_result, []


def reconstruct_case_result(
    *, entry: dict, application_csv: Path, metrics_csv: Path, ncu_rep: Path,
    provenance: dict, resolved_ncu_metrics: dict, git_commit: str,
) -> tuple[dict | None, list[str]]:
    """Path-based entry point: used by validate-profile-case's one-shot
    validation of a case's freshly-produced evidence (this campaign's own
    just-created case directory, safely re-derived by the caller from
    --campaign-dir + --index via _resolve_case_evidence_paths(), never a
    caller-supplied path)."""
    ncu_rep_err = p13._verify_artifact(ncu_rep)
    if ncu_rep_err:
        return None, [ncu_rep_err]
    try:
        with p13._open_regular_nofollow(application_csv, binary=False) as handle:
            application_rows = list(csv.reader(handle))
    except (OSError, p13.UnsafePathError, UnicodeError) as exc:
        return None, [f"{application_csv}: unable to read: {exc}"]
    try:
        with p13._open_regular_nofollow(metrics_csv, binary=False) as handle:
            metrics_rows = list(csv.reader(handle))
    except (OSError, p13.UnsafePathError, UnicodeError) as exc:
        return None, [f"{metrics_csv}: unable to read: {exc}"]
    try:
        application_hash = p13.sha256_of(application_csv)
        metrics_hash = p13.sha256_of(metrics_csv)
        ncu_rep_hash = p13.sha256_of(ncu_rep)
    except p13.UnsafePathError as exc:
        return None, [str(exc)]
    return _reconstruct_case_result_core(
        entry=entry, application_rows=application_rows, metrics_rows=metrics_rows,
        application_label=str(application_csv), metrics_label=str(metrics_csv),
        application_hash=application_hash, metrics_hash=metrics_hash, ncu_rep_hash=ncu_rep_hash,
        provenance=provenance, resolved_ncu_metrics=resolved_ncu_metrics, git_commit=git_commit,
    )


def _reconstruct_case_result_from_fds(
    *, entry: dict, case_name: str, fds: dict[str, int],
    provenance: dict, resolved_ncu_metrics: dict, git_commit: str,
) -> tuple[dict | None, list[str]]:
    """Descriptor-anchored entry point (Task 4, Section 7): used
    exclusively by the central evidence-integrity gate
    (verify_campaign_evidence_integrity), whose caller keeps every relevant
    directory descriptor (the campaign directory, "profiles/", and this
    specific case directory) open for the *entire* inventory+evidence
    check. Reads bytes only from the exact fds that check already opened
    and verified as non-symlink non-empty regular files (see
    _open_case_evidence_fds()) -- never by reopening a pathname."""
    application_rows = _read_csv_rows_from_fd(fds["application_csv"])
    metrics_rows = _read_csv_rows_from_fd(fds["metrics_csv"])
    application_hash = _sha256_of_fd(fds["application_csv"])
    metrics_hash = _sha256_of_fd(fds["metrics_csv"])
    ncu_rep_hash = _sha256_of_fd(fds["ncu_rep"])
    return _reconstruct_case_result_core(
        entry=entry, application_rows=application_rows, metrics_rows=metrics_rows,
        application_label=f"profiles/{case_name}/{case_name}.application.csv",
        metrics_label=f"profiles/{case_name}/{case_name}.metrics_raw.csv",
        application_hash=application_hash, metrics_hash=metrics_hash, ncu_rep_hash=ncu_rep_hash,
        provenance=provenance, resolved_ncu_metrics=resolved_ncu_metrics, git_commit=git_commit,
    )


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
    *, campaign_dir: Path, index: int, git_commit: str,
) -> tuple[bool, list[str]]:
    plan = build_ncu_plan()
    entry = next((e for e in plan if e["index"] == index), None)
    if entry is None:
        return False, [f"index {index} is not one of the frozen NCU case indices 0..{EXPECTED_NCU_CASE_COUNT - 1}"]

    try:
        manifest, _revision = load_p14_manifest_chain(campaign_dir)
        _validate_p14_manifest_document(manifest, require_initialized=True)
    except (p13.ManifestTransitionError, p13.UnsafePathError) as exc:
        return False, [f"P1.4 manifest: {exc}"]
    if manifest.get("state") != "PROFILE_IN_PROGRESS":
        return False, [f"P1.4 manifest state={manifest.get('state')!r} != 'PROFILE_IN_PROGRESS'"]

    existing_results = manifest.get("case_results", {})
    if entry["case_name"] in existing_results:
        return False, [f"case {entry['case_name']} was already validated and recorded; refusing to redo it"]

    # Task 4 remediation (Section 7): evidence paths are re-derived from the
    # safely resolved campaign directory, the frozen plan, and this case's
    # own canonical name alone -- never trusted from a caller-supplied
    # --application-csv/--metrics-csv/--ncu-rep argument (the previous CLI
    # accepted, and used verbatim, any path the invoking shell happened to
    # pass). This is the exact same derivation
    # verify_campaign_evidence_integrity uses for its own re-check.
    path_errors: list[str] = []
    paths = _resolve_case_evidence_paths(campaign_dir, entry["case_name"], path_errors)
    if paths is None:
        return False, path_errors

    provenance = manifest.get("provenance", {}) if isinstance(manifest.get("provenance"), dict) else {}
    resolved_ncu_metrics = manifest.get("resolved_ncu_metrics", {})
    case_result, errors = reconstruct_case_result(
        entry=entry, application_csv=paths["application_csv"], metrics_csv=paths["metrics_csv"],
        ncu_rep=paths["ncu_rep"], provenance=provenance, resolved_ncu_metrics=resolved_ncu_metrics,
        git_commit=git_commit,
    )
    if case_result is None:
        return False, errors

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
        campaign_dir=campaign_dir, index=args.index, git_commit=args.git_commit,
    )
    if not success:
        print(f"analyze_exp01_memory_paths_p14: validate-profile-case: FAIL: index {args.index}", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp01_memory_paths_p14: validate-profile-case:   - {error}", file=sys.stderr)
        return 1
    print(f"analyze_exp01_memory_paths_p14: validate-profile-case: OK: index {args.index}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Central evidence-integrity gate (Remediation B). One function, used
# unmodified immediately before both PROFILE_IN_PROGRESS -> COMPLETE
# (finalize-profile) and COMPLETE -> ANALYZED (analyze): safely reopens,
# recomputes, and compares the hashes of every trusted input the P1.4
# manifest depends on (pilot preflight, profile preflight, P1.3 terminal
# manifest, P1.3 combined_samples.csv/summary.csv, this campaign's own
# profile_plan.csv, and each of the six frozen cases' application CSV /
# metrics_raw.csv / .ncu-rep), and reparses every application CSV and NCU
# metrics CSV to confirm the reconstructed values still match the recorded
# case_results. Never trusts a hash sitting in memory from an earlier
# manifest transition -- every hash is recomputed fresh from disk on every
# call. Per-case paths are derived only from the frozen NCU plan and
# canonical case names, never from any stored path string.
# ---------------------------------------------------------------------------
def _verify_hash(label: str, path: Path, expected_sha256: object, errors: list[str]) -> str | None:
    """Returns the freshly recomputed SHA-256 of `path` (re-audit
    remediation, Group D), or None if it could not be computed or compared.
    Appending to `errors` is unchanged; the return value lets a caller build
    a terminal manifest revision directly from what this function just
    recomputed, rather than from a value that was already sitting in memory
    before this check ran."""
    if not isinstance(expected_sha256, str) or not p13._is_sha256_hex(expected_sha256):
        errors.append(f"{label}: no valid recorded SHA-256 to verify against (got {expected_sha256!r})")
        return None
    try:
        actual = p13.sha256_of(path)
    except p13.UnsafePathError as exc:
        errors.append(f"{label}: {exc}")
        return None
    if actual != expected_sha256:
        errors.append(
            f"{label}: {path} SHA-256 {actual} != recorded {expected_sha256} "
            f"(tampered or corrupted since it was first validated)"
        )
        return None
    return actual


def _resolve_case_evidence_paths(campaign_dir: Path, case_name: str, errors: list[str]) -> dict[str, Path] | None:
    """Derives the three per-case artifact paths from the frozen plan and
    canonical case name alone (never a stored path), and verifies the case
    directory and each artifact are non-symlink regular files contained
    within campaign_dir/profiles/ before returning them."""
    profiles_dir = campaign_dir / "profiles"
    case_dir = profiles_dir / case_name
    try:
        p13._reject_if_symlink_or_wrong_type(case_dir, expect_dir=True)
        if not os.path.lexists(case_dir):
            errors.append(f"{case_name}: profile case directory does not exist: {case_dir}")
            return None
        p13._confirm_contained(case_dir, profiles_dir)
    except p13.UnsafePathError as exc:
        errors.append(f"{case_name}: {exc}")
        return None

    paths = {
        "application_csv": case_dir / f"{case_name}.application.csv",
        "metrics_csv": case_dir / f"{case_name}.metrics_raw.csv",
        "ncu_rep": case_dir / f"{case_name}_report.ncu-rep",
    }
    for label, path in paths.items():
        artifact_err = p13._verify_artifact(path)
        if artifact_err:
            errors.append(f"{case_name}.{label}: {artifact_err}")
            return None
        try:
            p13._confirm_contained(path, case_dir)
        except p13.UnsafePathError as exc:
            errors.append(f"{case_name}.{label}: {exc}")
            return None
    return paths


def _read_profile_plan_case_names(profile_plan_path: Path) -> tuple[list[str], list[str]]:
    """Reads campaign_dir/profile_plan.csv (already hash-verified by the time
    this runs) and returns (case_names_in_file_order, errors). Never trusts
    build_ncu_plan() for the expected-name set here -- the whole point of
    Remediation B is to derive the expected set from the frozen,
    already-on-disk plan file, not from an in-memory constant."""
    try:
        with p13._open_regular_nofollow(profile_plan_path, binary=False) as handle:
            rows = list(csv.reader(handle))
    except (OSError, p13.UnsafePathError, UnicodeError) as exc:
        return [], [f"{profile_plan_path}: unable to read: {exc}"]
    if not rows:
        return [], [f"{profile_plan_path}: empty file"]
    header, data_rows = rows[0], rows[1:]
    if header != PROFILE_PLAN_HEADER:
        return [], [f"{profile_plan_path}: header mismatch: {header!r}"]
    case_names = [row[-1] for row in data_rows if row]
    return case_names, []


def _list_and_check_profiles_inventory(profiles_fd: int, expected_names: set[str]) -> list[str]:
    """fd-anchored inventory check (Task 4, Section 7/blocker E-1): both
    os.listdir(profiles_fd) and every os.stat(name, dir_fd=profiles_fd,
    follow_symlinks=False) resolve against the *same* already-open
    descriptor, never a path string. The previous design called
    p13._reject_if_symlink_or_wrong_type(profiles_dir, ...) (one lstat by
    path) and, afterward, a *separate* os.listdir(profiles_dir) (a second,
    independent resolution of the same path) -- replacing profiles/ with a
    symlink in the gap between those two calls produced errors=[] (the
    listdir silently followed the new symlink's target, which could easily
    be empty or otherwise "look compliant" instead of failing closed). An
    fd, once open, is bound to the inode -- not the name -- so nothing that
    happens to the name "profiles" afterward can change what
    os.listdir(profiles_fd) enumerates."""
    errors: list[str] = []
    try:
        actual_names = set(os.listdir(profiles_fd))
    except OSError as exc:
        return [f"profiles/: cannot list directory: {exc}"]
    for extra in sorted(actual_names - expected_names):
        errors.append(
            f"profiles/{extra}: unplanned entry not present in profile_plan.csv "
            f"(exactly {sorted(expected_names)} are permitted)"
        )
    for missing in sorted(expected_names - actual_names):
        errors.append(f"profiles/{missing}: missing canonical case directory")
    for name in sorted(actual_names & expected_names):
        try:
            st = os.stat(name, dir_fd=profiles_fd, follow_symlinks=False)
        except OSError as exc:
            errors.append(f"profiles/{name}: cannot stat: {exc}")
            continue
        if stat.S_ISLNK(st.st_mode):
            errors.append(f"profiles/{name}: is a symlink; refusing")
        elif not stat.S_ISDIR(st.st_mode):
            errors.append(f"profiles/{name}: is not a directory")
    return errors


def verify_profiles_directory_inventory(campaign_dir: Path) -> list[str]:
    """Standalone, path-based-entry convenience wrapper around
    _list_and_check_profiles_inventory(): opens profiles/ fresh via a
    single descriptor-anchored resolution, runs the fd-anchored check, and
    closes it. verify_campaign_evidence_integrity() below does not call
    this function -- it performs the equivalent open+check itself so the
    same profiles_fd can stay open for the rest of the evidence check
    (Section 7's "keep descriptors open for the entire inventory+evidence
    check")."""
    profile_plan_path = campaign_dir / "profile_plan.csv"
    case_names, plan_errors = _read_profile_plan_case_names(profile_plan_path)
    if plan_errors:
        return plan_errors
    if len(set(case_names)) != len(case_names):
        return [f"{profile_plan_path}: contains a duplicate case name: {case_names!r}"]
    expected_names = set(case_names)

    try:
        profiles_fd = _open_profiles_fd_anchored(campaign_dir)
    except p13.UnsafePathError as exc:
        return [f"profiles/: {exc}"]
    try:
        return _list_and_check_profiles_inventory(profiles_fd, expected_names)
    finally:
        os.close(profiles_fd)


# ---------------------------------------------------------------------------
# Strict recursive case-result comparison (Task 4, Section 8; blocker C).
# Replaces the previous "for key in sorted(set(a)|set(b)): if a.get(key) ==
# b.get(key): continue" loop, whose dict.get()-based equality made {} and
# {"unexpected_evidence_field": None} compare as identical (an unexpected
# key with a null value was silently accepted) and never distinguished "key
# absent" from "key present with value None". This walks recorded (the
# manifest's own case_results entry) against canonical (freshly
# reconstructed from evidence) structurally: exact key sets first (missing
# vs. unexpected reported as distinct conditions), then exact list
# length/order, then exact scalar type (type(x) is type(y), so True is
# never accepted in place of canonical 1) and exact value, with NaN/
# infinity rejected outright on either side. There is no tolerance
# permitting a manifest value to differ from what its own evidence
# reconstructs.
# ---------------------------------------------------------------------------
# Small, explicit allowlist for genuinely non-reconstructable operational
# metadata that may be compared with a declared, immutable-once-recorded
# type instead of exact structural equality. Deliberately empty: every
# field of a P1.4 case_results entry is derived from its own evidence (the
# whole point of reconstruct_case_result()), so nothing here qualifies as
# "non-reconstructable" -- this constant exists so a future field that
# genuinely cannot be reconstructed (and only such a field) has one
# documented, reviewed place to be added, rather than a silent exception
# carved into the comparison logic itself.
CASE_RESULT_COMPARISON_ALLOWLIST: frozenset[str] = frozenset()


def _strict_compare_values(path: str, recorded: object, canonical: object, errors: list[str]) -> None:
    if isinstance(canonical, dict) or isinstance(recorded, dict):
        if not isinstance(canonical, dict) or not isinstance(recorded, dict):
            errors.append(
                f"{path}: recorded is {type(recorded).__name__} ({recorded!r}), reconstructed is "
                f"{type(canonical).__name__} ({canonical!r})"
            )
            return
        recorded_keys, canonical_keys = set(recorded), set(canonical)
        missing = sorted(canonical_keys - recorded_keys)
        unexpected = sorted(recorded_keys - canonical_keys)
        if missing:
            errors.append(
                f"{path}: missing key(s) {missing} (present in reconstructed evidence, absent from "
                f"the recorded manifest)"
            )
        if unexpected:
            errors.append(
                f"{path}: unexpected key(s) {unexpected} (present in the recorded manifest, absent "
                f"from reconstructed evidence)"
            )
        for key in sorted(recorded_keys & canonical_keys):
            _strict_compare_values(f"{path}.{key}", recorded[key], canonical[key], errors)
        return

    if isinstance(canonical, list) or isinstance(recorded, list):
        if not isinstance(canonical, list) or not isinstance(recorded, list):
            errors.append(
                f"{path}: recorded is {type(recorded).__name__} ({recorded!r}), reconstructed is "
                f"{type(canonical).__name__} ({canonical!r})"
            )
            return
        if len(recorded) != len(canonical):
            errors.append(
                f"{path}: recorded has {len(recorded)} entry(ies), reconstructed has "
                f"{len(canonical)} (exact list length/order is required): recorded={recorded!r} "
                f"reconstructed={canonical!r}"
            )
            return
        for i, (r_item, c_item) in enumerate(zip(recorded, canonical)):
            _strict_compare_values(f"{path}[{i}]", r_item, c_item, errors)
        return

    # Scalars. bool is a subclass of int in Python, so "type(x) is type(y)"
    # -- never isinstance() -- is required to catch True/False masquerading
    # as canonical 1/0, and to catch an int masquerading as a canonical
    # float (or vice versa).
    if type(recorded) is not type(canonical):
        errors.append(
            f"{path}: type mismatch: recorded is {type(recorded).__name__} ({recorded!r}), "
            f"reconstructed is {type(canonical).__name__} ({canonical!r})"
        )
        return
    if isinstance(canonical, float):
        if not math.isfinite(canonical) or not math.isfinite(recorded):
            errors.append(
                f"{path}: NaN/infinite value rejected outright: recorded={recorded!r} "
                f"reconstructed={canonical!r}"
            )
            return
    if recorded != canonical:
        suffix = "tampered or corrupted since it was first validated" if path.endswith("_sha256") else "recomputed fresh from disk"
        errors.append(f"{path}: recorded {recorded!r} != reconstructed {canonical!r} ({suffix})")


def _recheck_inode_identity(campaign_fd: int, profiles_fd: int, case_fds: dict[str, int]) -> list[str]:
    """Immediately before verify_campaign_evidence_integrity() returns its
    verdict -- which licenses the caller to publish a terminal manifest
    revision -- re-confirms that "profiles" and every case name this check
    trusted still refer to the exact same (device, inode) they did when
    first opened (Task 4, Section 7, step 9/10). A change here means a name
    was rebound (deleted and recreated, or symlink-swapped) during this
    very validation; this fails closed rather than let a later reader trust
    a name whose target has since moved -- it never re-opens or re-trusts
    the new target itself."""
    errors: list[str] = []
    try:
        fresh_profiles = os.stat("profiles", dir_fd=campaign_fd, follow_symlinks=False)
        held_profiles = os.fstat(profiles_fd)
    except OSError as exc:
        return [f"profiles/: cannot re-check identity before terminal publication: {exc}"]
    if stat.S_ISLNK(fresh_profiles.st_mode) or (fresh_profiles.st_dev, fresh_profiles.st_ino) != (
        held_profiles.st_dev, held_profiles.st_ino,
    ):
        errors.append(
            "profiles/: name-to-inode binding changed during validation; failing closed rather "
            "than trust a name whose target moved"
        )
        return errors  # profiles/ itself rebound: per-case rechecks below would be meaningless
    for case_name, case_fd in sorted(case_fds.items()):
        try:
            fresh_case = os.stat(case_name, dir_fd=profiles_fd, follow_symlinks=False)
            held_case = os.fstat(case_fd)
        except OSError as exc:
            errors.append(f"profiles/{case_name}: cannot re-check identity before terminal publication: {exc}")
            continue
        if stat.S_ISLNK(fresh_case.st_mode) or (fresh_case.st_dev, fresh_case.st_ino) != (
            held_case.st_dev, held_case.st_ino,
        ):
            errors.append(
                f"profiles/{case_name}: name-to-inode binding changed during validation; failing "
                f"closed rather than trust a name whose target moved"
            )
    return errors


def verify_campaign_evidence_integrity(
    campaign_dir: Path, manifest: dict,
) -> tuple[list[str], dict | None]:
    """Returns (errors, verified_snapshot). errors is empty iff every trusted
    artifact still matches its recorded evidence and every reparsed value
    still agrees with what was recorded; verified_snapshot is None whenever
    errors is non-empty, and otherwise a dict of exactly the hashes and
    per-case reconstructions this call just recomputed fresh from disk
    (re-audit remediation, Group D) -- callers building a terminal manifest
    revision's artifact_sha256 use *this* returned snapshot, never a value
    that was already sitting in the caller's in-memory `manifest` before
    this check ran, so the gate's own freshly-verified data is what
    actually gets published. Never mutates the manifest or the campaign.

    The profiles/ inventory and every case's evidence read (Task 4, Section
    7) are descriptor-anchored: the campaign directory, profiles/, and each
    of the six case directories are opened exactly once, with
    O_DIRECTORY|O_NOFOLLOW relative to the previously opened descriptor,
    and every one of those descriptors is kept open for the *entire*
    inventory+evidence check below -- including the final inode-identity
    re-check immediately before this function returns its verdict."""
    errors: list[str] = []
    snapshot: dict = {}

    pilot_ref = manifest.get("pilot_campaign_reference")
    if not isinstance(pilot_ref, dict) or "path" not in pilot_ref:
        return ["pilot_campaign_reference is missing or incomplete; cannot verify pilot evidence"], None
    try:
        p13_campaign_dir = resolve_p13_campaign_dir_arg(pilot_ref["path"])
    except p13.UnsafePathError as exc:
        return [f"pilot_campaign_reference.path: {exc}"], None

    snapshot["pilot_manifest_sha256"] = _verify_hash(
        "P1.3 manifest.json", p13_campaign_dir / "manifest.json",
        pilot_ref.get("manifest_sha256"), errors,
    )
    snapshot["pilot_combined_samples_sha256"] = _verify_hash(
        "P1.3 combined_samples.csv", p13_campaign_dir / "combined_samples.csv",
        pilot_ref.get("combined_samples_sha256"), errors,
    )
    snapshot["pilot_summary_sha256"] = _verify_hash(
        "P1.3 summary.csv", p13_campaign_dir / "summary.csv",
        pilot_ref.get("summary_sha256"), errors,
    )

    preflight_pilot = manifest.get("preflight_reference_pilot", {})
    if isinstance(preflight_pilot, dict) and preflight_pilot.get("path"):
        snapshot["preflight_reference_pilot_sha256"] = _verify_hash(
            "pilot preflight summary", Path(preflight_pilot["path"]),
            preflight_pilot.get("sha256"), errors,
        )
    else:
        errors.append("preflight_reference_pilot is missing or incomplete")

    preflight_profile = manifest.get("preflight_reference_profile", {})
    if isinstance(preflight_profile, dict) and preflight_profile.get("path"):
        snapshot["preflight_reference_profile_sha256"] = _verify_hash(
            "profile preflight summary", Path(preflight_profile["path"]),
            preflight_profile.get("sha256"), errors,
        )
    else:
        errors.append("preflight_reference_profile is missing or incomplete")

    snapshot["profile_plan_sha256"] = _verify_hash(
        "P1.4 profile_plan.csv", campaign_dir / "profile_plan.csv",
        manifest.get("profile_plan_sha256"), errors,
    )

    plan = build_ncu_plan()
    plan_errors = check_ncu_plan_contract(plan)
    if plan_errors:
        return errors + [f"internal NCU plan contract violation: {plan_errors}"], None

    case_results = manifest.get("case_results")
    if not isinstance(case_results, dict):
        return errors + ["case_results is missing or not an object"], None

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
        errors.append("provenance is missing or not an object")
    resolved_ncu_metrics = manifest.get("resolved_ncu_metrics")
    if not isinstance(resolved_ncu_metrics, dict):
        resolved_ncu_metrics = {}

    expected_names = {entry["case_name"] for entry in plan}
    snapshot_case_results: dict[str, dict] = {}

    # Descriptor-anchored inventory + evidence (Task 4, Section 7): open the
    # campaign directory and profiles/ exactly once, hold both descriptors
    # (plus one per case actually opened below) for the whole check, and
    # only release them after the final inode-identity re-check.
    try:
        campaign_parts = campaign_dir.relative_to(REPO_ROOT).parts
        campaign_fd = _open_dir_component_chain(*campaign_parts)
    except p13.UnsafePathError as exc:
        return errors + [f"campaign directory: {exc}"], None
    try:
        try:
            profiles_fd = _open_dir_nofollow_p14("profiles", dir_fd=campaign_fd)
        except p13.UnsafePathError as exc:
            return errors + [f"profiles/: {exc}"], None
        try:
            inventory_errors = _list_and_check_profiles_inventory(profiles_fd, expected_names)
            if inventory_errors:
                return errors + inventory_errors, None

            found_case_names = set(case_results)
            for missing in sorted(expected_names - found_case_names):
                errors.append(f"profile case {missing} is missing from case_results")
            for extra in sorted(found_case_names - expected_names):
                errors.append(f"case_results contains unexpected case {extra}")

            case_fds: dict[str, int] = {}
            try:
                for entry in plan:
                    case_name = entry["case_name"]
                    recorded = case_results.get(case_name)
                    if recorded is None:
                        continue  # already reported as missing above
                    if not isinstance(recorded, dict):
                        errors.append(f"{case_name}: recorded case_results entry is not an object")
                        continue

                    try:
                        evidence_fds = _open_case_evidence_fds(profiles_fd, case_name)
                    except p13.UnsafePathError as exc:
                        errors.append(f"{case_name}: {exc}")
                        continue
                    case_fds[case_name] = evidence_fds["_case_dir"]

                    reconstructed, case_errors = _reconstruct_case_result_from_fds(
                        entry=entry, case_name=case_name, fds=evidence_fds, provenance=provenance,
                        resolved_ncu_metrics=resolved_ncu_metrics, git_commit=provenance.get("git_commit"),
                    )
                    for label in ("application_csv", "metrics_csv", "ncu_rep"):
                        os.close(evidence_fds[label])
                    if reconstructed is None:
                        errors.extend(f"{case_name}: {e}" for e in case_errors)
                        continue

                    _strict_compare_values(case_name, recorded, reconstructed, errors)
                    snapshot_case_results[case_name] = reconstructed

                if not errors:
                    errors.extend(_recheck_inode_identity(campaign_fd, profiles_fd, case_fds))
            finally:
                for fd in case_fds.values():
                    os.close(fd)
        finally:
            os.close(profiles_fd)
    finally:
        os.close(campaign_fd)

    if errors:
        return errors, None
    snapshot["case_results"] = snapshot_case_results
    return errors, snapshot


# ---------------------------------------------------------------------------
# Subcommand: finalize-profile
# ---------------------------------------------------------------------------
def _do_finalize_profile(
    *, campaign_dir: Path, completed_at_utc: str,
) -> tuple[bool, list[str]]:
    try:
        manifest, _revision = load_p14_manifest_chain(campaign_dir)
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

    # Central evidence-integrity gate, run as the final filesystem evidence
    # read before COMPLETE (re-audit remediation, Group D): re-reads and
    # recomputes every trusted hash and reparses every application/metrics
    # CSV from disk, rejecting any tampering that happened after
    # validate-profile-case recorded it. Between this call and the manifest
    # publication below, only deterministic in-memory construction from its
    # returned `verified` snapshot happens -- no further evidence read, no
    # artifact write, no external process.
    integrity_errors, verified = verify_campaign_evidence_integrity(campaign_dir, manifest)
    if integrity_errors:
        return _fail_p14(campaign_dir, "profile_finalize_integrity", integrity_errors)

    # artifact_sha256 is built exclusively from `verified` -- the hashes and
    # per-case reconstructions the gate above just recomputed fresh from
    # disk -- never copied from `manifest`/`case_results`, which were loaded
    # before the gate ran and could otherwise carry a stale value forward
    # into COMPLETE even though the gate itself compared correctly.
    artifact_sha256: dict[str, str] = {}
    if verified["profile_plan_sha256"] is not None:
        artifact_sha256["profile_plan.csv"] = verified["profile_plan_sha256"]
    for pilot_key, snapshot_key in (
        ("manifest_sha256", "pilot_manifest_sha256"),
        ("combined_samples_sha256", "pilot_combined_samples_sha256"),
        ("summary_sha256", "pilot_summary_sha256"),
    ):
        if verified[snapshot_key] is not None:
            artifact_sha256[f"pilot_{pilot_key}"] = verified[snapshot_key]
    for ref_name, snapshot_key in (
        ("preflight_reference_pilot", "preflight_reference_pilot_sha256"),
        ("preflight_reference_profile", "preflight_reference_profile_sha256"),
    ):
        if verified[snapshot_key] is not None:
            artifact_sha256[ref_name] = verified[snapshot_key]
    for case_name, result in verified["case_results"].items():
        for key in ("application_csv_sha256", "metrics_csv_sha256", "ncu_rep_sha256"):
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


def _do_analyze(
    *, campaign_dir: Path, analyzed_at_utc: str,
    _test_hook_before_final_gate: Callable[[], None] | None = None,
) -> tuple[bool, list[str]]:
    """_test_hook_before_final_gate, if given, is called with no arguments
    immediately after the newly published analysis artifacts are hashed and
    immediately before the second (pre-ANALYZED) evidence-integrity gate --
    self-test only, never exposed as a production CLI option (mirrors
    scripts/p14_safe_capture.py's own `_test_hook_after_open` convention) --
    so an adversarial test can mutate original pilot/profile evidence at
    exactly that boundary and prove the second gate, not merely the first,
    is what catches it."""
    try:
        manifest, _revision = load_p14_manifest_chain(campaign_dir)
        _validate_p14_manifest_document(manifest, require_initialized=True)
    except (p13.ManifestTransitionError, p13.UnsafePathError) as exc:
        return False, [f"P1.4 manifest: {exc}"]
    if manifest.get("state") != "COMPLETE":
        return False, [f"P1.4 manifest state={manifest.get('state')!r} != 'COMPLETE'; cannot analyze"]

    # First evidence-integrity gate (re-audit remediation, Group D): run
    # before consuming any evidence. The exact same function finalize-profile
    # calls before COMPLETE, re-run here before analysis starts consuming
    # pilot/profile evidence. Re-reads and recomputes every trusted hash and
    # reparses every application/metrics CSV from disk; rejects any
    # tampering that happened after finalize-profile. Analysis legitimately
    # reads evidence *after* this point (it must, to compute statistics), so
    # a second gate immediately before publishing ANALYZED closes that
    # interval below.
    integrity_errors, _verified_before = verify_campaign_evidence_integrity(campaign_dir, manifest)
    if integrity_errors:
        return False, integrity_errors

    errors: list[str] = []
    pilot_ref = manifest.get("pilot_campaign_reference", {})
    if not isinstance(pilot_ref, dict) or "path" not in pilot_ref:
        return False, ["manifest pilot_campaign_reference is missing or incomplete"]
    p13_campaign_dir = REPO_ROOT / pilot_ref["path"]
    combined_path = p13_campaign_dir / "combined_samples.csv"

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
    planned_analysis_paths = tuple(
        path.relative_to(campaign_dir).as_posix()
        for path in [*(item[0] for item in outputs), *(item[0] for item in file_writes)]
    )
    if (
        len(planned_analysis_paths) != len(ANALYSIS_ARTIFACT_RELATIVE_PATHS)
        or set(planned_analysis_paths) != set(ANALYSIS_ARTIFACT_RELATIVE_PATHS)
    ):
        return False, [
            "internal analysis artifact inventory differs from the canonical terminal "
            f"manifest contract: planned={sorted(planned_analysis_paths)!r} "
            f"canonical={sorted(ANALYSIS_ARTIFACT_RELATIVE_PATHS)!r}"
        ]
    for path, _ in file_writes:
        for candidate in (path, path.with_suffix(path.suffix + ".tmp")):
            if os.path.lexists(candidate):
                return False, [f"{candidate}: existing analysis artifact; refusing to overwrite"]

    # published_identity records each artifact's (dev, inode) immediately
    # after this call publishes it, so any later cleanup can unlink it only
    # while its identity is still exactly what was just published --
    # ownership-safe cleanup (re-audit remediation, Group E), reusing the
    # existing, audited p13._safe_unlink_owned(path, identity) rather than a
    # bare unconditional unlink.
    published: list[Path] = []
    published_identity: dict[Path, tuple[int, int]] = {}

    def _cleanup_published() -> None:
        for cleanup_path in published:
            try:
                p13._safe_unlink_owned(cleanup_path, published_identity.get(cleanup_path))
            except p13.UnsafePathError:
                pass

    try:
        for path, header, rows in outputs:
            _write_csv_no_clobber(path, header, rows)
            published.append(path)
            published_identity[path] = p13._file_identity(path)
        for path, text in file_writes:
            _write_text_no_clobber(path, text)
            published.append(path)
            published_identity[path] = p13._file_identity(path)
    except (p13.UnsafePathError, OSError) as exc:
        _cleanup_published()
        return False, [f"analysis artifact publication failed: {exc}"]

    try:
        artifact_hashes = {
            path.relative_to(campaign_dir).as_posix(): p13.sha256_of(path) for path, _, _ in outputs
        }
        artifact_hashes.update(
            {path.relative_to(campaign_dir).as_posix(): p13.sha256_of(path) for path, _ in file_writes}
        )
    except p13.UnsafePathError as exc:
        _cleanup_published()
        return False, [str(exc)]

    if _test_hook_before_final_gate is not None:
        _test_hook_before_final_gate()

    # Second evidence-integrity gate, run as the final filesystem evidence
    # read before ANALYZED (re-audit remediation, Group D): confirms none of
    # the original pilot/profile evidence changed while this analysis was
    # being computed and published above. If it fails, ANALYZED is never
    # published; the analysis artifacts this call itself just published are
    # removed (ownership/identity-checked, never an unfamiliar replacement);
    # and the changed-artifact diagnostics from the gate are reported.
    # Between this gate and the manifest publication below, only
    # deterministic in-memory construction of `updates` happens.
    integrity_errors2, _verified_after = verify_campaign_evidence_integrity(campaign_dir, manifest)
    if integrity_errors2:
        _cleanup_published()
        return False, [
            "evidence changed while analysis was being computed/published, detected by the "
            "second (pre-ANALYZED) integrity gate; ANALYZED was not published and the "
            "analysis artifacts just written were removed:",
        ] + integrity_errors2

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
    compute_cap: str = "10.3", driver_version: str = "580.95.05",
    gpu_visibility_status: str = "PASS", ncu_profile_status: str = "PASS",
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
            "driver_version": driver_version, "compute_cap": compute_cap, "memory_total": "288 GiB",
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
    shape a real P1.3 finalize produces. Placed at the same canonical
    results/raw/exp01_memory_paths/<campaign_id> path a real P1.3 campaign
    occupies (relative to the patched REPO_ROOT) so that the recorded
    pilot_campaign_reference.path later re-resolves through the same strict
    resolve_p13_campaign_dir_arg the evidence-integrity gate uses -- this
    fixture must model production layout, not a synthetic shortcut. Returns
    (campaign_dir, manifest)."""
    plan = p13.build_plan()
    campaign_dir = tmp_path.joinpath(*p13.RAW_ROOT_PARTS, campaign_id)
    (campaign_dir / "cases").mkdir(parents=True)
    (campaign_dir / "logs").mkdir(parents=True)
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
    dram_read_bytes: float | None = None, extra_launch_rows: list[list[str]] | None = None,
    kernel_name_in_csv: str | None = None, extra_kernel_row: bool = False,
    extra_same_id_kernel_row: bool = False,
    omit_metrics: tuple[str, ...] = (), unit_overrides: dict[str, str] | None = None,
    value_overrides: dict[str, str] | None = None,
    resolved_metrics: tuple[str, ...] = CANDIDATE_METRICS,
    case_dir: Path | None = None,
) -> tuple[Path, Path, Path, dict]:
    """Builds one application CSV + metrics_raw.csv + .ncu-rep for a given
    NCU_PLAN entry. By default writes NCU's raw-page wide CSV shape: one
    header row containing launch metadata and one column per metric in
    `resolved_metrics`, one units row, and one launch row containing the
    values. The default covers all five candidates with exact base units
    and the exact un-templated kernel name --print-kernel-base function
    would produce.
    `case_dir`, when given, is the directory the three artifacts are written
    into -- pass campaign_dir/"profiles"/case_name so the fixture matches the
    real layout _resolve_case_evidence_paths re-derives paths from (required
    for any caller that goes on to finalize-profile/analyze successfully);
    omit it (default: tmp_path directly) for standalone parser-only cases
    that never reach finalize-profile. Returns
    (application_csv, metrics_csv, ncu_rep, row)."""
    row = p13._default_row(
        entry, 0, repetitions=1, run_kind="benchmark", passes=32, warmup_ms=0,
        git_commit=git_commit, working_set_mib=512,
        overrides={"gpu_uuid": gpu_uuid, "gpu_name": gpu_name},
    )
    out_dir = case_dir if case_dir is not None else tmp_path
    out_dir.mkdir(parents=True, exist_ok=True)
    app_csv = out_dir / f"{entry['case_name']}.application.csv"
    p13._write_case_csv(app_csv, [row])
    useful_bytes = int(row["useful_bytes"])
    if dram_read_bytes is None:
        dram_read_bytes = useful_bytes * 0.95
    default_values: dict[str, str] = {
        MANDATORY_DRAM_METRIC: f"{dram_read_bytes}",
        "dram__bytes_write.sum": f"{useful_bytes * 0.1}",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed": "72.5",
        "lts__t_bytes.sum": f"{useful_bytes * 1.02}",
        "gpu__time_duration.sum": "123456",
    }
    unit_overrides = unit_overrides or {}
    value_overrides = value_overrides or {}
    metrics_csv = out_dir / f"{entry['case_name']}.metrics_raw.csv"
    kernel_name_field = kernel_name_in_csv or entry["kernel_name"]
    metadata_header = [
        "ID", "Process ID", "Process Name", "Host Name", "Kernel Name",
        "Kernel Time", "Context", "Stream",
    ]
    metadata_units = [""] * len(metadata_header)
    metadata_values = [
        "0", "1234", "fixture", "fixture-host", kernel_name_field,
        "2026-Jul-28 00:00:00", "1", "7",
    ]
    metric_names: list[str] = []
    metric_units: list[str] = []
    metric_values: list[str] = []
    for metric_name in resolved_metrics:
        canonical_metric_name = canonical_candidate_metric_name(metric_name)
        if canonical_metric_name is None:
            raise AssertionError(
                f"self-test fixture received a non-candidate resolved metric: {metric_name!r}"
            )
        if metric_name in omit_metrics or canonical_metric_name in omit_metrics:
            continue
        metric_names.append(metric_name)
        metric_units.append(unit_overrides.get(
            metric_name,
            unit_overrides.get(
                canonical_metric_name,
                EXPECTED_METRIC_UNITS[canonical_metric_name],
            ),
        ))
        metric_values.append(value_overrides.get(
            metric_name,
            value_overrides.get(
                canonical_metric_name,
                default_values[canonical_metric_name],
            ),
        ))

    header = metadata_header + metric_names
    units = metadata_units + metric_units
    launch_values = metadata_values + metric_values
    with open(metrics_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerow(units)
        writer.writerow(launch_values)
        if extra_kernel_row:
            second_launch = launch_values.copy()
            second_launch[header.index("ID")] = "1"
            second_launch[header.index("Kernel Name")] = "some_other_kernel"
            writer.writerow(second_launch)
        if extra_same_id_kernel_row:
            second_kernel = launch_values.copy()
            second_kernel[header.index("Kernel Name")] = "some_other_kernel"
            writer.writerow(second_kernel)
        for extra in extra_launch_rows or []:
            writer.writerow(extra)
    ncu_rep = out_dir / f"{entry['case_name']}_report.ncu-rep"
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
        # One minute after this fixture's own pilot_completed_at_utc
        # ("20260728T103000Z" above) -- was previously "20260728T003100Z"
        # (00:31, i.e. *before* pilot completion), a latent fixture typo that
        # validate_manifest_timestamp_chronology (re-audit remediation,
        # Group B) now correctly catches.
        started_at_utc="20260728T103100Z", now_utc=now,
    )
    if not ok:
        raise AssertionError(f"self-test fixture: discover-metrics failed: {errors}")
    for entry in build_ncu_plan():
        dram_bytes = dram_read_bytes_fn(entry) if dram_read_bytes_fn is not None else None
        _build_ncu_case_fixture(
            tmp_path, entry, dram_read_bytes=dram_bytes,
            case_dir=p14_campaign_dir / "profiles" / entry["case_name"],
        )
        ok, errors = _do_validate_profile_case(
            campaign_dir=p14_campaign_dir, index=entry["index"], git_commit=_FIXED_GIT_COMMIT,
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

            # --- fail-closed NCU CSV parser (Remediation C / spec Section 7) -------
            plan0 = ncu_plan[0]
            _NCU_METADATA_HEADER = [
                "ID", "Process ID", "Process Name", "Host Name", "Kernel Name",
                "Kernel Time", "Context", "Stream",
            ]
            _NCU_METADATA_UNITS = [""] * len(_NCU_METADATA_HEADER)
            _NCU_METADATA_VALUES = [
                "0", "1234", "fixture", "fixture-host", plan0["kernel_name"],
                "2026-Jul-28 00:00:00", "1", "7",
            ]

            def _write_ncu_csv(
                path: Path, header: list[str], units: list[str], rows: list[list[str]],
            ) -> Path:
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(header)
                    w.writerow(units)
                    for row in rows:
                        w.writerow(row)
                return path

            one_metric_header = _NCU_METADATA_HEADER + [MANDATORY_DRAM_METRIC]
            one_metric_units = _NCU_METADATA_UNITS + ["byte"]
            one_metric_values = _NCU_METADATA_VALUES + ["1000000"]
            good_csv = _write_ncu_csv(
                tmp_path / "metrics_good.csv",
                one_metric_header, one_metric_units, [one_metric_values],
            )
            parsed = parse_ncu_raw_csv(good_csv)
            rec.check(
                "well-formed NCU wide raw CSV parses its metric column, units row, launch, and kernel",
                parsed["metrics"].get(MANDATORY_DRAM_METRIC) == 1000000.0
                and parsed["kernel_name"] == plan0["kernel_name"]
                and parsed["units"][MANDATORY_DRAM_METRIC] == "byte"
                and parsed["launch_id"] == "0"
                and parsed["launch_count"] == 1,
                detail=f"parsed={parsed}",
            )

            # Exact unit-row regression from the live GB300/Nsight Compute
            # 2025.4 export canary collected on 2026-07-29. These literals
            # are intentionally independent of EXPECTED_METRIC_UNITS so a
            # mistaken constant cannot make both fixture and parser agree.
            live_metric_header = _NCU_METADATA_HEADER + list(CANDIDATE_METRICS)
            live_metric_units = _NCU_METADATA_UNITS + [
                "byte", "byte", "%", "byte", "ns",
            ]
            live_metric_values = _NCU_METADATA_VALUES + [
                "17228217600", "5010176", "40.55", "25844105184", "5539840",
            ]
            live_units_csv = _write_ncu_csv(
                tmp_path / "metrics_live_gb300_units.csv",
                live_metric_header, live_metric_units, [live_metric_values],
            )
            live_units_parsed = parse_ncu_raw_csv(live_units_csv)
            rec.check(
                "live GB300 NCU 2025.4 unit row is accepted exactly, including gpu duration 'ns'",
                live_units_parsed["units"]
                == dict(zip(CANDIDATE_METRICS, live_metric_units[-5:])),
                detail=f"units={live_units_parsed['units']}",
            )

            for label, value in (
                ("empty", ""), ("NaN", "nan"), ("infinite", "inf"), ("negative-infinite", "-inf"),
                ("non-numeric", "not-a-number"), ("negative", "-5"),
            ):
                malformed_values = one_metric_values.copy()
                malformed_values[-1] = value
                bad_csv = _write_ncu_csv(
                    tmp_path / f"metrics_bad_{label}.csv",
                    one_metric_header, one_metric_units, [malformed_values],
                )
                raised = False
                try:
                    parse_ncu_raw_csv(bad_csv)
                except NcuCsvParseError:
                    raised = True
                rec.check(f"metrics CSV with {label} value is rejected", raised)

            # Launch identity columns are structural requirements.
            for missing_col in ("ID", "Kernel Name"):
                remove_at = one_metric_header.index(missing_col)
                header_without = one_metric_header[:remove_at] + one_metric_header[remove_at + 1:]
                units_without = one_metric_units[:remove_at] + one_metric_units[remove_at + 1:]
                row_without = one_metric_values[:remove_at] + one_metric_values[remove_at + 1:]
                missing_csv = _write_ncu_csv(
                    tmp_path / f"metrics_missing_{missing_col.replace(' ', '_')}.csv",
                    header_without, units_without, [row_without],
                )
                raised = False
                try:
                    parse_ncu_raw_csv(missing_csv)
                except NcuCsvParseError:
                    raised = True
                rec.check(f"metrics CSV missing the {missing_col!r} column is rejected", raised)

            missing_unit_row = one_metric_units.copy()
            missing_unit_row[-1] = ""
            missing_unit_csv = _write_ncu_csv(
                tmp_path / "metrics_missing_candidate_unit.csv",
                one_metric_header, missing_unit_row, [one_metric_values],
            )
            raised = False
            try:
                parse_ncu_raw_csv(missing_unit_csv)
            except NcuCsvParseError:
                raised = True
            rec.check("metrics CSV with an empty candidate unit in the units row is rejected", raised)

            # duplicate required header
            dup_header_csv = _write_ncu_csv(
                tmp_path / "metrics_dup_header.csv",
                one_metric_header + ["ID"],
                one_metric_units + [""],
                [one_metric_values + ["0"]],
            )
            raised = False
            try:
                parse_ncu_raw_csv(dup_header_csv)
            except NcuCsvParseError:
                raised = True
            rec.check("metrics CSV with a duplicate header column name is rejected", raised)

            # empty launch ID
            empty_id_values = one_metric_values.copy()
            empty_id_values[one_metric_header.index("ID")] = ""
            empty_id_csv = _write_ncu_csv(
                tmp_path / "metrics_empty_id.csv",
                one_metric_header, one_metric_units, [empty_id_values],
            )
            raised = False
            try:
                parse_ncu_raw_csv(empty_id_csv)
            except NcuCsvParseError:
                raised = True
            rec.check("metrics CSV with an empty launch ID is rejected", raised)

            # two launch IDs (unit-level, complementing the validate-profile-case test below)
            second_launch_values = one_metric_values.copy()
            second_launch_values[one_metric_header.index("ID")] = "1"
            two_launch_csv = _write_ncu_csv(
                tmp_path / "metrics_two_launch.csv",
                one_metric_header, one_metric_units,
                [one_metric_values, second_launch_values],
            )
            raised = False
            try:
                parse_ncu_raw_csv(two_launch_csv)
            except NcuCsvParseError:
                raised = True
            rec.check("metrics CSV with two distinct launch IDs is rejected (unit-level)", raised)

            # Canonical and qualified columns for the same candidate are
            # semantically duplicate and therefore ambiguous, even if values
            # agree and the literal header strings differ.
            qualified_dram = f"FBSP.TriageCompute.{MANDATORY_DRAM_METRIC}"
            dup_metric_csv = _write_ncu_csv(
                tmp_path / "metrics_dup_metric.csv",
                one_metric_header + [qualified_dram],
                one_metric_units + ["byte"],
                [one_metric_values + ["1000000"]],
            )
            raised = False
            try:
                parse_ncu_raw_csv(dup_metric_csv)
            except NcuCsvParseError:
                raised = True
            rec.check(
                "metrics CSV with canonical and qualified columns for one candidate is rejected as ambiguous",
                raised,
            )

            # dram unit = kilobyte / Kbyte -- rejected (never scaled/converted)
            for bad_unit in ("kilobyte", "Kbyte", "KB"):
                bad_units = one_metric_units.copy()
                bad_units[-1] = bad_unit
                bad_unit_csv = _write_ncu_csv(
                    tmp_path / f"metrics_unit_{bad_unit}.csv",
                    one_metric_header, bad_units, [one_metric_values],
                )
                raised = False
                try:
                    parse_ncu_raw_csv(bad_unit_csv)
                except NcuCsvParseError:
                    raised = True
                rec.check(f"metrics CSV with dram unit={bad_unit!r} is rejected", raised)
            # dram unit = byte (any case) -- accepted
            byte_case_units = one_metric_units.copy()
            byte_case_units[-1] = "Byte"
            byte_case_csv = _write_ncu_csv(
                tmp_path / "metrics_unit_Byte.csv",
                one_metric_header, byte_case_units, [one_metric_values],
            )
            byte_case_parsed = None
            try:
                byte_case_parsed = parse_ncu_raw_csv(byte_case_csv)
            except NcuCsvParseError:
                pass
            rec.check(
                "metrics CSV with dram unit='Byte' (case-normalized) is accepted",
                byte_case_parsed is not None
                and byte_case_parsed["metrics"][MANDATORY_DRAM_METRIC] == 1000000.0,
            )

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
            stability_stats_by_config = compute_all_config_stats(
                {
                    ("ldgsts", 2, 16): [100.0, 100.0, 100.0, 100.0],
                    ("tma", 2, 16): high_cv_values,
                },
                random.Random(BOOTSTRAP_SEED),
            )
            stability_report = render_report_markdown(
                campaign_id="20260730T073045Z",
                stats_rows=[
                    _stats_to_csv_row(
                        "ldgsts", 2, 16,
                        stability_stats_by_config[("ldgsts", 2, 16)],
                    ),
                    _stats_to_csv_row(
                        "tma", 2, 16,
                        stability_stats_by_config[("tma", 2, 16)],
                    ),
                ],
                pairwise_rows=[],
                saturation_rows=[],
                ncu_rows=[],
                provenance={},
                dram_read_metric_available=False,
            )
            low_stability_line = next(
                line for line in stability_report.splitlines()
                if line.startswith("| ldgsts | 2 | 16 |")
            )
            high_stability_line = next(
                line for line in stability_report.splitlines()
                if line.startswith("| tma | 2 | 16 |")
            )
            rec.check(
                "report renders the string label 'ok' as ok and 'REVIEW' as REVIEW "
                "instead of treating both non-empty strings as truthy",
                "| ok |" in low_stability_line
                and "| REVIEW |" not in low_stability_line
                and "| REVIEW |" in high_stability_line,
                detail=(
                    f"low={low_stability_line!r} "
                    f"high={high_stability_line!r}"
                ),
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
            manifest_good, _rev = load_p14_manifest_chain(p14_good)
            rec.check("record-pilot transitions to PILOT_COMPLETE", manifest_good.get("state") == "PILOT_COMPLETE")

            p13_smoke, _ = _build_p13_pilot_campaign_fixture(tmp_path, "rp_smoke", run_kind="smoke")
            p14_smoke = _do_init_campaign(campaign_id="20260728T160100Z", started_at_utc="20260728T160100Z")
            ok, errors = _do_record_pilot(
                campaign_dir=p14_smoke, p13_campaign_dir=p13_smoke, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260728T160500Z", now_utc=now_rp,
            )
            rec.expect_error_containing("record-pilot rejects a smoke P1.3 campaign (item 5)", errors, "run_kind")
            rec.check("record-pilot smoke rejection leaves campaign FAILED, not PILOT_COMPLETE",
                      not ok and load_p14_manifest_chain(p14_smoke)[0].get("state") == "FAILED")

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
            # NCU 2025.4 on the real GB300 emits a table and namespace-
            # qualified metric identifiers, not one bare canonical metric
            # per line. Reproduce that observed format exactly enough to
            # exercise both first-column extraction and qualified-name
            # resolution while retaining the full name for --metrics.
            qualified_prefix = "FBSP.TriageCompute."
            qualified_candidates = tuple(
                f"{qualified_prefix}{metric}"
                for metric in CANDIDATE_METRICS
            )
            full_discovery_log = tmp_path / "discovery_full.log"
            full_discovery_log.write_text(
                "Device NVIDIA B300 SXM6 AC (GB110)\n"
                + "\n".join(
                    f"{metric:<96} Counter         byte            # synthetic description"
                    for metric in qualified_candidates
                )
                + "\nunrelated__noise.metric                                      Counter\n",
                encoding="utf-8",
            )
            ok, errors, resolved = _do_discover_metrics(
                campaign_dir=p14_dm, discovery_log=full_discovery_log, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, started_at_utc="20260728T161200Z", now_utc=now_rp,
            )
            rec.check(
                "discover-metrics parses the real GB300 table format, preserves all five qualified "
                "NCU identifiers, and transitions to PROFILE_IN_PROGRESS",
                ok and resolved["dram_read_metric_available"]
                and resolved["resolved"] == list(qualified_candidates)
                and resolved["dram_read_metric"] == qualified_candidates[0]
                and load_p14_manifest_chain(p14_dm)[0].get("state") == "PROFILE_IN_PROGRESS",
                detail=f"ok={ok} resolved={resolved}",
            )
            qualified_case = build_ncu_plan()[0]
            _build_ncu_case_fixture(
                tmp_path,
                qualified_case,
                # Exact cross-spelling regression: discovery preserved the
                # GB300-qualified identifiers above, while the raw page uses
                # canonical metric column names. Both spellings must resolve
                # to one semantic candidate without weakening ambiguity
                # checks or changing the manifest's discovery identifiers.
                resolved_metrics=CANDIDATE_METRICS,
                case_dir=p14_dm / "profiles" / qualified_case["case_name"],
            )
            qualified_case_ok, qualified_case_errors = _do_validate_profile_case(
                campaign_dir=p14_dm,
                index=qualified_case["index"],
                git_commit=_FIXED_GIT_COMMIT,
            )
            qualified_case_result = load_p14_manifest_chain(p14_dm)[0].get(
                "case_results", {}
            ).get(qualified_case["case_name"], {})
            rec.check(
                "qualified GB300 discovery identifiers validate against canonical raw-page "
                "metric columns and classify the mandatory DRAM metric",
                qualified_case_ok
                and qualified_case_result.get("hbm_classification") == "HBM_VALIDATED"
                and qualified_candidates[0]
                in qualified_case_result.get("resolved_metric_values", {}),
                detail=(
                    f"ok={qualified_case_ok} errors={qualified_case_errors} "
                    f"result={qualified_case_result}"
                ),
            )

            ambiguous_metric_rejected = False
            ambiguous_metric_error = ""
            try:
                resolve_ncu_metrics({
                    f"DomainA.{MANDATORY_DRAM_METRIC}",
                    f"DomainB.{MANDATORY_DRAM_METRIC}",
                })
            except ValueError as exc:
                ambiguous_metric_rejected = True
                ambiguous_metric_error = str(exc)
            rec.check(
                "metric discovery rejects multiple qualified identifiers for one canonical "
                "candidate instead of guessing",
                ambiguous_metric_rejected
                and MANDATORY_DRAM_METRIC in ambiguous_metric_error,
                detail=f"error={ambiguous_metric_error!r}",
            )

            # --- pilot vs profile preflight provenance mismatch is rejected
            # (Remediation A, audit blocker #1) -----------------------------------
            def _build_mismatched_profile_case(suffix: str, campaign_id: str, **preflight_overrides):
                p13_c, _ = _build_p13_pilot_campaign_fixture(tmp_path, f"prov_{suffix}")
                p14_c = _do_init_campaign(campaign_id=campaign_id, started_at_utc=campaign_id)
                ok_inner, errors_inner = _do_record_pilot(
                    campaign_dir=p14_c, p13_campaign_dir=p13_c, preflight_path=good_preflight_rp,
                    git_commit=_FIXED_GIT_COMMIT, completed_at_utc=campaign_id, now_utc=now_rp,
                )
                assert ok_inner, errors_inner
                mismatched_preflight = tmp_path / f"preflight_prov_{suffix}.json"
                _write_preflight_json(mismatched_preflight, _default_preflight_doc(**preflight_overrides))
                disc_log = tmp_path / f"disc_prov_{suffix}.log"
                disc_log.write_text("\n".join(CANDIDATE_METRICS) + "\n", encoding="utf-8")
                return p14_c, mismatched_preflight, disc_log

            p14_prov_uuid, preflight_prov_uuid, disc_prov_uuid = _build_mismatched_profile_case(
                "uuid", "20260728T164000Z", gpu_uuid="GPU-99999999-8888-7777-6666-555555555555",
            )
            ok, errors, _ = _do_discover_metrics(
                campaign_dir=p14_prov_uuid, discovery_log=disc_prov_uuid, preflight_path=preflight_prov_uuid,
                git_commit=_FIXED_GIT_COMMIT, started_at_utc="20260728T164001Z", now_utc=now_rp,
            )
            rec.expect_error_containing(
                "discover-metrics rejects a profiling preflight with a different GPU UUID "
                "than the pilot (Remediation A)", errors, "gpu_uuid",
            )

            p14_prov_name, preflight_prov_name, disc_prov_name = _build_mismatched_profile_case(
                "name", "20260728T164100Z", gpu_name="NVIDIA B300 (impostor)",
            )
            ok, errors, _ = _do_discover_metrics(
                campaign_dir=p14_prov_name, discovery_log=disc_prov_name, preflight_path=preflight_prov_name,
                git_commit=_FIXED_GIT_COMMIT, started_at_utc="20260728T164101Z", now_utc=now_rp,
            )
            rec.expect_error_containing(
                "discover-metrics rejects a profiling preflight with a different GPU name "
                "than the pilot (Remediation A)", errors, "gpu_name",
            )

            p14_prov_drv, preflight_prov_drv, disc_prov_drv = _build_mismatched_profile_case(
                "driver", "20260728T164200Z", driver_version="570.00.01",
            )
            ok, errors, _ = _do_discover_metrics(
                campaign_dir=p14_prov_drv, discovery_log=disc_prov_drv, preflight_path=preflight_prov_drv,
                git_commit=_FIXED_GIT_COMMIT, started_at_utc="20260728T164201Z", now_utc=now_rp,
            )
            rec.expect_error_containing(
                "discover-metrics rejects a profiling preflight with a different driver version "
                "than the pilot (Remediation A)", errors, "gpu_driver_version",
            )

            # Direct unit coverage of compare_preflight_provenance for every field,
            # independent of whether a real captured preflight can currently
            # produce that particular mismatch (compute_cap is separately pinned
            # to '10.3' and git_commit to the caller's expected commit by
            # validate_preflight_fields itself -- this confirms the comparison
            # function still catches every field if either constraint is ever
            # relaxed).
            base_snapshot = {
                "git_commit": _FIXED_GIT_COMMIT, "gpu_uuid": _FIXED_GPU_UUID,
                "gpu_name": _FIXED_GPU_NAME, "gpu_compute_cap": "10.3",
                "gpu_driver_version": "580.95.05",
            }
            for field, other_value in (
                ("git_commit", "f" * 40), ("gpu_uuid", "GPU-99999999-8888-7777-6666-555555555555"),
                ("gpu_name", "NVIDIA B300 (impostor)"), ("gpu_compute_cap", "9.0"),
                ("gpu_driver_version", "570.00.01"),
            ):
                mismatched_snapshot = dict(base_snapshot, **{field: other_value})
                prov_errors = compare_preflight_provenance(base_snapshot, mismatched_snapshot)
                rec.check(
                    f"compare_preflight_provenance catches a {field} mismatch",
                    len(prov_errors) == 1 and field in prov_errors[0],
                    detail=f"errors={prov_errors}",
                )
            rec.check(
                "compare_preflight_provenance returns no errors when every field matches",
                compare_preflight_provenance(base_snapshot, dict(base_snapshot)) == [],
            )

            # --- validate-profile-preconditions: the standalone, side-effect-free
            # early gate (Remediation A) -------------------------------------------
            p13_vpp, _ = _build_p13_pilot_campaign_fixture(tmp_path, "vpp_base")
            p14_vpp = _do_init_campaign(campaign_id="20260728T164300Z", started_at_utc="20260728T164300Z")
            ok, errors = _do_record_pilot(
                campaign_dir=p14_vpp, p13_campaign_dir=p13_vpp, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260728T164301Z", now_utc=now_rp,
            )
            assert ok, errors
            manifest_before_vpp, revision_before_vpp = load_p14_manifest_chain(p14_vpp)

            ok, errors = _do_validate_profile_preconditions(
                campaign_dir=p14_vpp, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, now_utc=now_rp,
            )
            rec.check(
                "validate-profile-preconditions accepts a matching profiling preflight", ok, detail=f"{errors}",
            )
            manifest_after_vpp, revision_after_vpp = load_p14_manifest_chain(p14_vpp)
            rec.check(
                "validate-profile-preconditions writes nothing to the manifest (pure precondition check)",
                manifest_after_vpp == manifest_before_vpp and revision_after_vpp == revision_before_vpp,
            )

            mismatched_vpp_preflight = tmp_path / "preflight_vpp_mismatch.json"
            _write_preflight_json(
                mismatched_vpp_preflight, _default_preflight_doc(gpu_name="NVIDIA B300 (impostor)"),
            )
            ok, errors = _do_validate_profile_preconditions(
                campaign_dir=p14_vpp, preflight_path=mismatched_vpp_preflight,
                git_commit=_FIXED_GIT_COMMIT, now_utc=now_rp,
            )
            rec.expect_error_containing(
                "validate-profile-preconditions rejects a mismatched profiling preflight", errors, "gpu_name",
            )

            p14_vpp_wrongstate = _do_init_campaign(
                campaign_id="20260728T164400Z", started_at_utc="20260728T164400Z",
            )
            ok, errors = _do_validate_profile_preconditions(
                campaign_dir=p14_vpp_wrongstate, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, now_utc=now_rp,
            )
            rec.expect_error_containing(
                "validate-profile-preconditions refuses a campaign that is not yet PILOT_COMPLETE",
                errors, "PILOT_COMPLETE",
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
            _build_ncu_case_fixture(tmp_path, case0, case_dir=p14_dm2 / "profiles" / case0["case_name"])
            case_ok, case_errors = _do_validate_profile_case(
                campaign_dir=p14_dm2, index=0, git_commit=_FIXED_GIT_COMMIT,
            )
            classification_when_unavailable = load_p14_manifest_chain(p14_dm2)[0]["case_results"][case0["case_name"]]["hbm_classification"]
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
            _build_ncu_case_fixture(
                tmp_path, case_wrong_kernel, kernel_name_in_csv="tma_benchmark_kernel<2,4>(...)",
                case_dir=p14_vc2 / "profiles" / case_wrong_kernel["case_name"],
            )
            ok, errors = _do_validate_profile_case(
                campaign_dir=p14_vc2, index=0, git_commit=_FIXED_GIT_COMMIT,
            )
            rec.expect_error_containing("validate-profile-case rejects a wrong kernel name (item 11)", errors, "kernel name")

            case_dup_launch = build_ncu_plan()[1]
            _build_ncu_case_fixture(
                tmp_path, case_dup_launch, extra_kernel_row=True,
                case_dir=p14_vc2 / "profiles" / case_dup_launch["case_name"],
            )
            ok, errors = _do_validate_profile_case(
                campaign_dir=p14_vc2, index=1, git_commit=_FIXED_GIT_COMMIT,
            )
            rec.expect_error_containing(
                "validate-profile-case rejects more than one profiled launch row (item 11)", errors,
                "profiled launch row(s)",
            )

            case_dup_kernel = build_ncu_plan()[2]
            _build_ncu_case_fixture(
                tmp_path, case_dup_kernel, extra_same_id_kernel_row=True,
                case_dir=p14_vc2 / "profiles" / case_dup_kernel["case_name"],
            )
            ok, errors = _do_validate_profile_case(
                campaign_dir=p14_vc2, index=2, git_commit=_FIXED_GIT_COMMIT,
            )
            rec.expect_error_containing(
                "validate-profile-case rejects a second kernel row even when its launch ID is reused (item 11)",
                errors, "profiled launch row(s)",
            )

            # Kernel name containing the expected name plus extra text: exact
            # equality only, never substring/prefix/suffix matching.
            case_superset_kernel = build_ncu_plan()[3]
            _build_ncu_case_fixture(
                tmp_path, case_superset_kernel,
                kernel_name_in_csv=f"{case_superset_kernel['kernel_name']}_v2",
                case_dir=p14_vc2 / "profiles" / case_superset_kernel["case_name"],
            )
            ok, errors = _do_validate_profile_case(
                campaign_dir=p14_vc2, index=3, git_commit=_FIXED_GIT_COMMIT,
            )
            rec.check(
                "a kernel name containing the expected name plus extra text is rejected, not substring-matched",
                not ok and any("!= expected exact" in e for e in errors),
                detail=f"ok={ok} errors={errors}",
            )

            # A metric recorded as *resolved* during discovery but missing from
            # this case's own CSV is a hard integrity failure, never INCONCLUSIVE.
            case_missing_dram = build_ncu_plan()[4]
            _build_ncu_case_fixture(
                tmp_path, case_missing_dram, omit_metrics=(MANDATORY_DRAM_METRIC,),
                case_dir=p14_vc2 / "profiles" / case_missing_dram["case_name"],
            )
            ok, errors = _do_validate_profile_case(
                campaign_dir=p14_vc2, index=4, git_commit=_FIXED_GIT_COMMIT,
            )
            rec.expect_error_containing(
                "a resolved-but-missing DRAM metric is a hard validation failure, not INCONCLUSIVE",
                errors, "is missing resolved metric 'dram__bytes_read.sum'",
            )

            case_missing_optional = build_ncu_plan()[5]
            _build_ncu_case_fixture(
                tmp_path, case_missing_optional, omit_metrics=("lts__t_bytes.sum",),
                case_dir=p14_vc2 / "profiles" / case_missing_optional["case_name"],
            )
            ok, errors = _do_validate_profile_case(
                campaign_dir=p14_vc2, index=5, git_commit=_FIXED_GIT_COMMIT,
            )
            rec.expect_error_containing(
                "a resolved-but-missing optional metric is also a hard validation failure",
                errors, "is missing resolved metric 'lts__t_bytes.sum'",
            )

            case_redo = build_ncu_plan()[0]
            _build_ncu_case_fixture(
                tmp_path, case_redo, case_dir=p14_vc2 / "profiles" / case_redo["case_name"],
            )
            ok, errors = _do_validate_profile_case(
                campaign_dir=p14_vc2, index=0, git_commit=_FIXED_GIT_COMMIT,
            )
            rec.check("validate-profile-case accepts case 0 on its first, correct submission", ok, detail=f"{errors}")
            ok, errors = _do_validate_profile_case(
                campaign_dir=p14_vc2, index=0, git_commit=_FIXED_GIT_COMMIT,
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
                "finalize-profile transitions to COMPLETE",
                load_p14_manifest_chain(p14_vc)[0].get("state") == "COMPLETE",
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

            # --- append-only, hash-chained P1.4 manifest revisions (Remediation E) --
            # The second revision uses a real, legal transition
            # (PILOT_IN_PROGRESS -> FAILED) rather than an artificial
            # same-state repeat: PILOT_IN_PROGRESS has no self-loop (Task 4,
            # Section 9 -- the real --pilot workflow never reports
            # incremental progress into the P1.4 manifest, so that
            # same-state edge was removed from ALLOWED_P14_TRANSITIONS).
            # FAILED/INTERRUPTED are the only transitions reachable from
            # PILOT_IN_PROGRESS that require no additional manifest fields,
            # so they exercise the identical two-revision hash-chain
            # mechanics these tests are actually about.
            def _build_two_revision_campaign(campaign_id: str) -> Path:
                p14_c = _do_init_campaign(campaign_id=campaign_id, started_at_utc=campaign_id)
                p14_merge_manifest(
                    p14_c,
                    {"failure_stage": "self_test_synthetic", "failure_detail": []},
                    state="FAILED",
                )
                return p14_c

            p14_mrev = _do_init_campaign(campaign_id="20260728T170500Z", started_at_utc="20260728T170500Z")
            rev0_path = p14_mrev / "manifest" / "000000.json"
            rev0_bytes_at_write_time = rev0_path.read_bytes()

            atomic_write_called = False

            def _poison_write_manifest_atomic(*_args, **_kwargs):
                nonlocal atomic_write_called
                atomic_write_called = True
                raise AssertionError("p13.write_manifest_atomic must never be called for a P1.4 manifest")

            merge_raised = False
            with mock.patch.object(p13, "write_manifest_atomic", side_effect=_poison_write_manifest_atomic):
                try:
                    p14_merge_manifest(
                        p14_mrev,
                        {"failure_stage": "self_test_synthetic", "failure_detail": []},
                        state="FAILED",
                    )
                except AssertionError:
                    merge_raised = True
            rec.check(
                "p14_merge_manifest never calls p13.write_manifest_atomic, even when it "
                "would raise if called (Remediation E)",
                not merge_raised and not atomic_write_called,
            )
            rec.check(
                "revision 000000.json is still byte-for-byte identical after a later "
                "revision is published (Remediation E)",
                rev0_path.read_bytes() == rev0_bytes_at_write_time,
            )

            p14_tamper_prev = _build_two_revision_campaign("20260728T170600Z")
            rev0_tp = p14_tamper_prev / "manifest" / "000000.json"
            rev0_doc = json.loads(rev0_tp.read_text(encoding="utf-8"))
            rev0_doc["started_at_utc"] = "20260728T170601Z"
            rev0_tp.write_text(json.dumps(rev0_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            chain_raised = False
            try:
                load_p14_manifest_chain(p14_tamper_prev)
            except p13.ManifestTransitionError:
                chain_raised = True
            rec.check(
                "load_p14_manifest_chain rejects a campaign whose earlier revision "
                "content was modified after the fact (Remediation E)",
                chain_raised,
            )

            p14_broken_chain = _build_two_revision_campaign("20260728T170700Z")
            rev1_bc = p14_broken_chain / "manifest" / "000001.json"
            rev1_doc = json.loads(rev1_bc.read_text(encoding="utf-8"))
            rev1_doc["previous_manifest_sha256"] = "0" * 64
            rev1_bc.write_text(json.dumps(rev1_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            chain_raised = False
            try:
                load_p14_manifest_chain(p14_broken_chain)
            except p13.ManifestTransitionError:
                chain_raised = True
            rec.check(
                "load_p14_manifest_chain rejects a revision whose previous_manifest_sha256 "
                "does not match the actual previous revision's hash (Remediation E)",
                chain_raised,
            )

            p14_symlink_rev = _build_two_revision_campaign("20260728T170800Z")
            rev0_sr = p14_symlink_rev / "manifest" / "000000.json"
            decoy_target = tmp_path / "decoy_manifest_revision.json"
            decoy_target.write_text(rev0_sr.read_text(encoding="utf-8"), encoding="utf-8")
            rev0_sr.unlink()
            rev0_sr.symlink_to(decoy_target)
            chain_raised = False
            try:
                load_p14_manifest_chain(p14_symlink_rev)
            except p13.ManifestTransitionError:
                chain_raised = True
            rec.check(
                "load_p14_manifest_chain rejects a symlinked revision file, even one "
                "whose target has otherwise-valid content (Remediation E)",
                chain_raised,
            )

            p14_stale_tmp = _build_two_revision_campaign("20260728T170900Z")
            stale_tmp_path = p14_stale_tmp / "manifest" / MANIFEST_REVISION_TMP_NAME
            stale_tmp_path.write_text("stale leftover from an interrupted write\n", encoding="utf-8")
            stale_tmp_bytes = stale_tmp_path.read_bytes()
            next_revision_path = p14_stale_tmp / "manifest" / "000002.json"
            write_raised = False
            try:
                write_next_p14_manifest_revision(p14_stale_tmp, {"state": "PILOT_IN_PROGRESS"})
            except p13.UnsafePathError:
                write_raised = True
            rec.check(
                "write_next_p14_manifest_revision refuses to write over a stale "
                "temporary file and creates no new revision (Remediation E)",
                write_raised and stale_tmp_path.read_bytes() == stale_tmp_bytes
                and not next_revision_path.exists(),
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
                    _build_ncu_case_fixture(
                        tmp_path, entry, case_dir=p14_det / "profiles" / entry["case_name"],
                    )
                    ok, errors = _do_validate_profile_case(
                        campaign_dir=p14_det, index=entry["index"], git_commit=_FIXED_GIT_COMMIT,
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
                manifest_det, _rev = load_p14_manifest_chain(p14_det)
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
            last_manifest, _rev = load_p14_manifest_chain(p14_det)
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
                _build_ncu_case_fixture(
                    tmp_path, entry, case_dir=p14_broken / "profiles" / entry["case_name"],
                )
                ok, errors = _do_validate_profile_case(
                    campaign_dir=p14_broken, index=entry["index"], git_commit=_FIXED_GIT_COMMIT,
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
                load_p14_manifest_chain(p14_broken)[0].get("state") == "COMPLETE",
            )

            # --- evidence-integrity gate: independent tamper-then-reject coverage
            # for every artifact verify_campaign_evidence_integrity re-verifies
            # (Remediation B, audit blocker #2) --------------------------------
            p13_tp, _ = _build_p13_pilot_campaign_fixture(tmp_path, "tp_base")
            p14_tp = _do_init_campaign(campaign_id="20260728T200000Z", started_at_utc="20260728T200000Z")
            preflight_tp_pilot = tmp_path / "preflight_tp_pilot.json"
            preflight_tp_profile = tmp_path / "preflight_tp_profile.json"
            _write_preflight_json(preflight_tp_pilot, _default_preflight_doc())
            _write_preflight_json(preflight_tp_profile, _default_preflight_doc())
            now_tp = _datetime(2026, 7, 28, 11, 0, tzinfo=_timezone.utc)
            ok, errors = _do_record_pilot(
                campaign_dir=p14_tp, p13_campaign_dir=p13_tp, preflight_path=preflight_tp_pilot,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260728T200100Z", now_utc=now_tp,
            )
            assert ok, errors
            disc_tp = tmp_path / "disc_tp.log"
            disc_tp.write_text("\n".join(CANDIDATE_METRICS) + "\n", encoding="utf-8")
            ok, errors, _ = _do_discover_metrics(
                campaign_dir=p14_tp, discovery_log=disc_tp, preflight_path=preflight_tp_profile,
                git_commit=_FIXED_GIT_COMMIT, started_at_utc="20260728T200200Z", now_utc=now_tp,
            )
            assert ok, errors
            for entry in build_ncu_plan():
                _build_ncu_case_fixture(
                    tmp_path, entry, case_dir=p14_tp / "profiles" / entry["case_name"],
                )
                ok, errors = _do_validate_profile_case(
                    campaign_dir=p14_tp, index=entry["index"], git_commit=_FIXED_GIT_COMMIT,
                )
                assert ok, errors
            ok, errors = _do_finalize_profile(campaign_dir=p14_tp, completed_at_utc="20260728T200300Z")
            assert ok, errors

            case0_name = build_ncu_plan()[0]["case_name"]
            case0_dir = p14_tp / "profiles" / case0_name

            def _check_tamper_rejected(label: str, path: Path) -> None:
                original = path.read_bytes()
                path.write_bytes(original + b"\ntampered\n")
                try:
                    ok, errors = _do_analyze(campaign_dir=p14_tp, analyzed_at_utc="20260728T200400Z")
                    published = list((p14_tp / "analysis").rglob("*")) if (p14_tp / "analysis").exists() else []
                    rec.check(
                        f"analyze() rejects a tampered {label} and publishes nothing (Remediation B)",
                        not ok and not published,
                        detail=f"ok={ok} errors={errors} published={published}",
                    )
                finally:
                    path.write_bytes(original)

            _check_tamper_rejected("case metrics_raw.csv", case0_dir / f"{case0_name}.metrics_raw.csv")
            _check_tamper_rejected("case application.csv", case0_dir / f"{case0_name}.application.csv")
            _check_tamper_rejected("case .ncu-rep", case0_dir / f"{case0_name}_report.ncu-rep")
            _check_tamper_rejected("profile_plan.csv", p14_tp / "profile_plan.csv")
            _check_tamper_rejected("pilot preflight summary", preflight_tp_pilot)
            _check_tamper_rejected("profile preflight summary", preflight_tp_profile)
            _check_tamper_rejected("P1.3 manifest.json", p13_tp / "manifest.json")
            _check_tamper_rejected("P1.3 combined_samples.csv", p13_tp / "combined_samples.csv")
            _check_tamper_rejected("P1.3 summary.csv", p13_tp / "summary.csv")

            # Once every tampered artifact above is restored to its original
            # bytes, analyze() must succeed -- proving each rejection above was
            # caused by that iteration's own tamper, not leftover corruption.
            ok, errors = _do_analyze(campaign_dir=p14_tp, analyzed_at_utc="20260728T200400Z")
            rec.check("analyze() succeeds once all tampered artifacts are restored", ok, detail=f"{errors}")

            # --- the exact originally-audited sequence (audit blocker #2):
            # validate-profile-case -> modify metrics_raw.csv -> finalize-profile
            # rejected -> analyze cannot accept -----------------------------------
            p14_audit_seq, _ = _run_profile_pipeline(tmp_path, "20260728T210000Z")
            audit_case0_name = build_ncu_plan()[0]["case_name"]
            audit_metrics_path = (
                p14_audit_seq / "profiles" / audit_case0_name / f"{audit_case0_name}.metrics_raw.csv"
            )
            audit_metrics_path.write_bytes(audit_metrics_path.read_bytes() + b"\ntampered\n")
            ok, errors = _do_finalize_profile(campaign_dir=p14_audit_seq, completed_at_utc="20260728T210100Z")
            # The central gate now reconstructs each case via the exact same
            # canonical function validate-profile-case itself uses
            # (reconstruct_case_result), which reparses metrics_raw.csv
            # *before* it would ever re-hash it -- so an appended malformed
            # row is caught here as a structural parse failure, not a hash
            # mismatch; a tamper that preserved CSV structure would instead
            # be caught by the SHA-256 comparison a few lines later in the
            # same reconstruction. Both are the same evidence-integrity gate
            # rejecting the same tampered file; only the diagnostic differs.
            rec.check(
                "audited sequence: finalize-profile rejects a metrics_raw.csv modified "
                "after validate-profile-case recorded it",
                not ok and any("metrics_raw.csv" in e for e in errors),
                detail=f"ok={ok} errors={errors}",
            )
            rec.check(
                "audited sequence: the rejected campaign never reaches COMPLETE",
                load_p14_manifest_chain(p14_audit_seq)[0].get("state") != "COMPLETE",
            )
            ok, errors = _do_analyze(campaign_dir=p14_audit_seq, analyzed_at_utc="20260728T210200Z")
            rec.check(
                "audited sequence: analyze() cannot accept a campaign whose "
                "finalize-profile was rejected",
                not ok,
                detail=f"ok={ok} errors={errors}",
            )

            # --- Remediation B: exact profile-directory inventory (second
            # audit, blocker B) ---------------------------------------------------
            # The exact audit regression: an unplanned seventh profile
            # directory, never compared against anything by the previous gate.
            p14_inv, _ = _run_profile_pipeline(tmp_path, "20260728T220000Z")
            unplanned_dir = p14_inv / "profiles" / "99_unplanned_profile"
            unplanned_dir.mkdir()
            (unplanned_dir / "unplanned.ncu-rep").write_bytes(b"unplanned bytes\n")
            ok, errors = _do_finalize_profile(campaign_dir=p14_inv, completed_at_utc="20260728T220001Z")
            rec.check(
                "finalize-profile rejects an unplanned seventh profile directory "
                "(Remediation B, audit blocker B, exact regression)",
                not ok and any("99_unplanned_profile" in e for e in errors),
                detail=f"ok={ok} errors={errors}",
            )
            rec.check(
                "the campaign with an unplanned profile directory never reaches COMPLETE",
                load_p14_manifest_chain(p14_inv)[0].get("state") != "COMPLETE",
            )

            p14_inv_emptydir, _ = _run_profile_pipeline(tmp_path, "20260728T220200Z")
            (p14_inv_emptydir / "profiles" / "extra_empty_dir").mkdir()
            ok, errors = _do_finalize_profile(campaign_dir=p14_inv_emptydir, completed_at_utc="20260728T220201Z")
            rec.expect_error_containing(
                "finalize-profile rejects an extra empty profiles/ directory (Remediation B)",
                errors, "extra_empty_dir",
            )

            p14_inv_file, _ = _run_profile_pipeline(tmp_path, "20260728T220300Z")
            (p14_inv_file / "profiles" / "extra_file.txt").write_text("not a case\n", encoding="utf-8")
            ok, errors = _do_finalize_profile(campaign_dir=p14_inv_file, completed_at_utc="20260728T220301Z")
            rec.expect_error_containing(
                "finalize-profile rejects an extra regular file at profiles/ root (Remediation B)",
                errors, "extra_file.txt",
            )

            p14_inv_symlink, _ = _run_profile_pipeline(tmp_path, "20260728T220400Z")
            symlink_escape_dir = tmp_path / "profiles_symlink_escape_target"
            symlink_escape_dir.mkdir()
            (p14_inv_symlink / "profiles" / "extra_symlink").symlink_to(symlink_escape_dir)
            ok, errors = _do_finalize_profile(campaign_dir=p14_inv_symlink, completed_at_utc="20260728T220401Z")
            rec.expect_error_containing(
                "finalize-profile rejects an extra symlink at profiles/ root (Remediation B)",
                errors, "extra_symlink",
            )

            p14_inv_missing, _ = _run_profile_pipeline(tmp_path, "20260728T220500Z")
            missing_case_name = build_ncu_plan()[0]["case_name"]
            shutil.rmtree(p14_inv_missing / "profiles" / missing_case_name)
            ok, errors = _do_finalize_profile(campaign_dir=p14_inv_missing, completed_at_utc="20260728T220501Z")
            rec.expect_error_containing(
                "finalize-profile rejects a missing canonical profile directory (Remediation B)",
                errors, "missing canonical",
            )

            p14_inv_post, _ = _run_profile_pipeline(tmp_path, "20260728T220600Z")
            ok, errors = _do_finalize_profile(campaign_dir=p14_inv_post, completed_at_utc="20260728T220601Z")
            assert ok, errors
            (p14_inv_post / "profiles" / "post_complete_extra").mkdir()
            ok, errors = _do_analyze(campaign_dir=p14_inv_post, analyzed_at_utc="20260728T220700Z")
            analysis_dir_inv = p14_inv_post / "analysis"
            published_inv = list(analysis_dir_inv.rglob("*")) if analysis_dir_inv.exists() else []
            rec.check(
                "analyze() rejects a profiles/ entry that appeared after COMPLETE and "
                "publishes no analysis artifact (Remediation B)",
                not ok and not published_inv and any("post_complete_extra" in e for e in errors),
                detail=f"ok={ok} errors={errors} published={published_inv}",
            )

            # --- Remediation D: canonical case-result reconstruction (second
            # audit, blocker D) ---------------------------------------------------
            def _mutate_case_field(case_name: str, field: str, value):
                def _mutate(manifest: dict) -> None:
                    manifest["case_results"][case_name][field] = value
                return _mutate

            def _append_tampered_manifest_revision(campaign_dir: Path, mutate_fn) -> None:
                """Simulates an attacker with direct filesystem write access
                to manifest/ -- bypassing write_next_p14_manifest_revision
                and its own semantic-transition check entirely, exactly as
                the audited reproduction did ("a correctly hashed new
                revision... was accepted") -- by writing a syntactically
                valid, correctly hash-chained next revision file directly,
                whose content has been tampered. Never goes through this
                module's own writer, which would now (correctly) refuse it."""
                current, current_revision = load_p14_manifest_chain(campaign_dir)
                mutated = copy.deepcopy(current)
                mutate_fn(mutated)
                new_revision = current_revision + 1
                previous_hash = p13.sha256_of(_manifest_revision_path(campaign_dir, current_revision))
                doc = dict(mutated)
                doc["manifest_revision"] = new_revision
                doc["previous_manifest_sha256"] = previous_hash
                text = json.dumps(doc, indent=2, sort_keys=True) + "\n"
                _manifest_revision_path(campaign_dir, new_revision).write_text(text, encoding="utf-8")

            d_case_name = build_ncu_plan()[0]["case_name"]

            def _mentions_case_tamper(errors: list[str]) -> bool:
                # Once Remediation C exists, tampering *any* field inside an
                # already-recorded case_results entry is *also* an "existing
                # case_result was modified" append-only violation, which
                # load_p14_manifest_chain now catches even before
                # verify_campaign_evidence_integrity's own field-by-field
                # dram_read_bytes comparison would run. Both are correct,
                # independent confirmations that the exact same tamper is
                # rejected; accept either diagnosis.
                return any(
                    f"{d_case_name}.dram_read_bytes" in e
                    or (f"case_results.{d_case_name}" in e and "modified" in e)
                    for e in errors
                )

            # The exact audit regression: dram_read_bytes tampered to 0 while
            # metrics_raw.csv itself is left completely untouched.
            p14_d_pre, _ = _run_profile_pipeline(tmp_path, "20260728T221000Z")
            _append_tampered_manifest_revision(p14_d_pre, _mutate_case_field(d_case_name, "dram_read_bytes", 0.0))
            ok, errors = _do_finalize_profile(campaign_dir=p14_d_pre, completed_at_utc="20260728T221001Z")
            rec.check(
                "finalize-profile rejects a tampered dram_read_bytes even though "
                "metrics_raw.csv is untouched (Remediation D, audit blocker D, exact regression)",
                not ok and _mentions_case_tamper(errors),
                detail=f"ok={ok} errors={errors}",
            )
            try:
                pre_state = load_p14_manifest_chain(p14_d_pre)[0].get("state")
            except p13.ManifestTransitionError:
                pre_state = None  # the chain itself is now invalid -- certainly not COMPLETE
            rec.check(
                "the campaign with a tampered dram_read_bytes never reaches COMPLETE",
                pre_state != "COMPLETE",
            )

            p14_d_post, _ = _run_profile_pipeline(tmp_path, "20260728T221100Z")
            ok, errors = _do_finalize_profile(campaign_dir=p14_d_post, completed_at_utc="20260728T221101Z")
            assert ok, errors
            _append_tampered_manifest_revision(p14_d_post, _mutate_case_field(d_case_name, "dram_read_bytes", 0.0))
            ok, errors = _do_analyze(campaign_dir=p14_d_post, analyzed_at_utc="20260728T221200Z")
            analysis_dir_d = p14_d_post / "analysis"
            published_d = list(analysis_dir_d.rglob("*")) if analysis_dir_d.exists() else []
            rec.check(
                "analyze() rejects a post-COMPLETE tampered dram_read_bytes and "
                "publishes no analysis artifact (Remediation D)",
                not ok and not published_d and _mentions_case_tamper(errors),
                detail=f"ok={ok} errors={errors} published={published_d}",
            )

            # Table-driven: every remaining evidence-derived field, tested
            # directly against the pure verify_campaign_evidence_integrity
            # function (never mutates the manifest/campaign) for speed --
            # building a fresh six-case campaign per mutation would be
            # correct but needlessly slow; this exercises the exact same
            # comparison logic finalize-profile/analyze both call unmodified.
            p14_d_table, _ = _run_profile_pipeline(tmp_path, "20260728T221300Z")
            ok, errors = _do_finalize_profile(campaign_dir=p14_d_table, completed_at_utc="20260728T221301Z")
            assert ok, errors
            base_manifest_d, _rev = load_p14_manifest_chain(p14_d_table)
            baseline_errors, baseline_snapshot = verify_campaign_evidence_integrity(p14_d_table, base_manifest_d)
            rec.check(
                "verify_campaign_evidence_integrity finds no errors on an untampered COMPLETE campaign",
                baseline_errors == [],
                detail=f"errors={baseline_errors}",
            )
            rec.check(
                "verify_campaign_evidence_integrity returns a non-None verified snapshot with all six "
                "freshly reconstructed case results when it finds no errors (re-audit remediation, Group D)",
                baseline_snapshot is not None
                and set(baseline_snapshot.get("case_results", {})) == {e["case_name"] for e in build_ncu_plan()},
                detail=f"snapshot={baseline_snapshot}",
            )
            some_resolved_metric = next(iter(base_manifest_d["case_results"][d_case_name]["resolved_metric_values"]))

            def _check_d_mutation(label: str, field: str, mutate) -> None:
                mutated_manifest = copy.deepcopy(base_manifest_d)
                mutate(mutated_manifest["case_results"][d_case_name])
                mutation_errors, mutation_snapshot = verify_campaign_evidence_integrity(p14_d_table, mutated_manifest)
                rec.check(
                    f"canonical reconstruction detects a mutated {label} (Remediation D)",
                    any(f"{d_case_name}.{field}" in e for e in mutation_errors) and mutation_snapshot is None,
                    detail=f"errors={mutation_errors}",
                )

            _check_d_mutation(
                "resolved metric numeric value", "resolved_metric_values",
                lambda c: c["resolved_metric_values"].__setitem__(some_resolved_metric, -999.0),
            )
            _check_d_mutation(
                "resolved metric unit", "resolved_metric_units",
                lambda c: c["resolved_metric_units"].__setitem__(some_resolved_metric, "furlongs"),
            )
            _check_d_mutation(
                "missing resolved metric value", "resolved_metric_values",
                lambda c: c["resolved_metric_values"].pop(some_resolved_metric),
            )
            _check_d_mutation(
                "extra resolved metric value", "resolved_metric_values",
                lambda c: c["resolved_metric_values"].__setitem__("bogus_extra_metric", 1.0),
            )
            _check_d_mutation(
                "dram_read_ratio", "dram_read_ratio", lambda c: c.__setitem__("dram_read_ratio", 0.0),
            )
            _check_d_mutation(
                "HBM classification", "hbm_classification",
                lambda c: c.__setitem__("hbm_classification", "INCONCLUSIVE"),
            )
            _check_d_mutation(
                "diagnostic_flags (READ_AMPLIFICATION)", "diagnostic_flags",
                lambda c: c.__setitem__("diagnostic_flags", ["READ_AMPLIFICATION"]),
            )
            _check_d_mutation(
                "useful_bytes (expected DRAM-read bytes)", "useful_bytes",
                lambda c: c.__setitem__("useful_bytes", "1"),
            )
            _check_d_mutation("stages (stage count)", "stages", lambda c: c.__setitem__("stages", 999))
            _check_d_mutation(
                "bytes_in_flight_kib (bytes in flight)", "bytes_in_flight_kib",
                lambda c: c.__setitem__("bytes_in_flight_kib", 999),
            )
            _check_d_mutation("launch_id", "launch_id", lambda c: c.__setitem__("launch_id", "999"))
            _check_d_mutation("launch_count", "launch_count", lambda c: c.__setitem__("launch_count", 999))
            _check_d_mutation(
                "application_csv_sha256 (artifact hash)", "application_csv_sha256",
                lambda c: c.__setitem__("application_csv_sha256", "0" * 64),
            )
            _check_d_mutation(
                "metrics_csv_sha256 (artifact hash)", "metrics_csv_sha256",
                lambda c: c.__setitem__("metrics_csv_sha256", "0" * 64),
            )
            _check_d_mutation(
                "ncu_rep_sha256 (artifact hash)", "ncu_rep_sha256",
                lambda c: c.__setitem__("ncu_rep_sha256", "0" * 64),
            )

            # Paths and the frozen kernel name are re-derived, never trusted
            # from storage, so their integrity is tested directly against
            # reconstruct_case_result -- the exact function the gate and
            # validate-profile-case both call -- rather than as a case_results
            # mutation.
            d_entry = build_ncu_plan()[0]
            d_provenance = base_manifest_d["provenance"]
            d_resolved = base_manifest_d["resolved_ncu_metrics"]
            good_app_csv, good_metrics_csv, good_ncu_rep, _ = _build_ncu_case_fixture(tmp_path, d_entry)
            _, wrong_kernel_metrics_csv, _, _ = _build_ncu_case_fixture(
                tmp_path, d_entry, kernel_name_in_csv=f"{d_entry['kernel_name']}_v2",
            )
            reconstructed, recon_errors = reconstruct_case_result(
                entry=d_entry, application_csv=good_app_csv, metrics_csv=wrong_kernel_metrics_csv,
                ncu_rep=good_ncu_rep, provenance=d_provenance, resolved_ncu_metrics=d_resolved,
                git_commit=_FIXED_GIT_COMMIT,
            )
            rec.check(
                "reconstruct_case_result rejects a metrics CSV whose kernel name does not "
                "match the frozen plan entry (Remediation D, case order/identity)",
                reconstructed is None and any("kernel name" in e for e in recon_errors),
                detail=f"errors={recon_errors}",
            )

            missing_ncu_rep = tmp_path / "does_not_exist_report.ncu-rep"
            reconstructed, recon_errors = reconstruct_case_result(
                entry=d_entry, application_csv=good_app_csv, metrics_csv=good_metrics_csv,
                ncu_rep=missing_ncu_rep, provenance=d_provenance, resolved_ncu_metrics=d_resolved,
                git_commit=_FIXED_GIT_COMMIT,
            )
            rec.check(
                "reconstruct_case_result rejects a missing .ncu-rep artifact (Remediation D, "
                "artifact path)",
                reconstructed is None and bool(recon_errors),
                detail=f"errors={recon_errors}",
            )

            # --- Remediation C: semantic manifest transition validation
            # (second audit, blocker C) --------------------------------------
            def _mutate_top_level_field(field: str, value):
                def _mutate(manifest: dict) -> None:
                    manifest[field] = value
                return _mutate

            # The exact audit regression: a syntactically valid, correctly
            # hash-chained new revision that changes the immutable
            # campaign_id must still be rejected by semantic loading.
            p14_c_campaign_id, _ = _run_profile_pipeline(tmp_path, "20260728T222000Z")
            _append_tampered_manifest_revision(
                p14_c_campaign_id, _mutate_top_level_field("campaign_id", "20260728T235959Z"),
            )
            raised, error_text = False, ""
            try:
                load_p14_manifest_chain(p14_c_campaign_id)
            except p13.ManifestTransitionError as exc:
                raised, error_text = True, str(exc)
            rec.check(
                "load_p14_manifest_chain rejects a syntactically valid, correctly "
                "hash-chained revision that changes campaign_id (Remediation C, audit "
                "blocker C, exact regression)",
                raised and "campaign_id" in error_text,
                detail=f"raised={raised} error={error_text}",
            )

            # revision-zero campaign_id not matching the campaign directory.
            p14_c_rev0_id = "20260728T222100Z"
            p14_c_rev0 = _do_init_campaign(campaign_id=p14_c_rev0_id, started_at_utc=p14_c_rev0_id)
            rev0_path = p14_c_rev0 / "manifest" / "000000.json"
            rev0_doc = json.loads(rev0_path.read_text(encoding="utf-8"))
            rev0_doc["campaign_id"] = "20260728T235959Z"
            rev0_path.write_text(json.dumps(rev0_doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raised = False
            try:
                load_p14_manifest_chain(p14_c_rev0)
            except p13.ManifestTransitionError as exc:
                raised = "campaign_id" in str(exc)
            rec.check(
                "load_p14_manifest_chain rejects a revision 0 whose campaign_id does "
                "not match the campaign directory basename (Remediation C)",
                raised,
            )

            # Every other invariant: exercised directly against
            # validate_manifest_revision_transition, isolating each one from the
            # other, unrelated schema checks load_p14_manifest_chain also applies
            # (several of these values would also independently trip
            # _validate_p14_manifest_document's required-field/schema checks;
            # testing the pure transition function directly proves *this*
            # invariant specifically catches it too).
            # A same-document "unchanged pair" is not itself a meaningful check
            # for every state (e.g. COMPLETE has no self-loop, since
            # finalize-profile runs exactly once) -- so validate against a
            # real adjacent pair from p14_d_table's own untampered chain
            # instead (revision N-1 -> N, its last two real revisions).
            table_revision_files = sorted((p14_d_table / "manifest").glob("*.json"))
            assert len(table_revision_files) >= 2, table_revision_files
            real_prev_doc = json.loads(table_revision_files[-2].read_text(encoding="utf-8"))
            real_curr_doc = json.loads(table_revision_files[-1].read_text(encoding="utf-8"))
            real_prev_content = {k: v for k, v in real_prev_doc.items() if k not in MANIFEST_REVISION_KEYS}
            real_curr_content = {k: v for k, v in real_curr_doc.items() if k not in MANIFEST_REVISION_KEYS}
            base_transition_ok = validate_manifest_revision_transition(
                real_prev_content, real_curr_content, real_curr_content["campaign_id"],
            )
            rec.check(
                "validate_manifest_revision_transition finds no errors for a real, "
                "untampered adjacent revision pair",
                base_transition_ok == [],
                detail=f"errors={base_transition_ok}",
            )

            def _check_c_invariant(label: str, mutate_prev, mutate_curr, expect_substr: str) -> None:
                prev = copy.deepcopy(base_manifest_d)
                curr = copy.deepcopy(base_manifest_d)
                if mutate_prev is not None:
                    mutate_prev(prev)
                if mutate_curr is not None:
                    mutate_curr(curr)
                transition_errors = validate_manifest_revision_transition(prev, curr, prev["campaign_id"])
                rec.check(
                    f"validate_manifest_revision_transition detects {label} (Remediation C)",
                    any(expect_substr in e for e in transition_errors),
                    detail=f"errors={transition_errors}",
                )

            _check_c_invariant(
                "a changed schema_version", None,
                lambda m: m.__setitem__("schema_version", "999"), "schema_version",
            )
            _check_c_invariant(
                "a changed frozen_protocol", None,
                lambda m: m.__setitem__("frozen_protocol", {"tampered": True}), "frozen_protocol",
            )
            _check_c_invariant(
                "a changed pilot_campaign_reference", None,
                lambda m: m.__setitem__("pilot_campaign_reference", {"path": "tampered"}),
                "pilot_campaign_reference",
            )
            _check_c_invariant(
                "a changed resolved-metric definition (resolved_ncu_metrics)", None,
                lambda m: m.__setitem__(
                    "resolved_ncu_metrics", {"resolved": [], "dram_read_metric_available": False},
                ),
                "resolved_ncu_metrics",
            )
            _check_c_invariant(
                "an illegal state jump (COMPLETE -> PILOT_IN_PROGRESS)",
                lambda m: m.__setitem__("state", "COMPLETE"),
                lambda m: m.__setitem__("state", "PILOT_IN_PROGRESS"),
                "illegal state transition",
            )
            _check_c_invariant(
                "a state regression (ANALYZED -> COMPLETE)",
                lambda m: m.__setitem__("state", "ANALYZED"),
                lambda m: m.__setitem__("state", "COMPLETE"),
                "illegal state transition",
            )
            _check_c_invariant(
                "a modified earlier case_result", None,
                lambda m: m["case_results"][d_case_name].__setitem__("dram_read_bytes", -1.0),
                f"case_results.{d_case_name}",
            )
            _check_c_invariant(
                "a deleted case_result", None,
                lambda m: m["case_results"].pop(d_case_name),
                f"case_results.{d_case_name}",
            )

            frozen_order_names = [entry["case_name"] for entry in build_ncu_plan()]
            all_case_results = base_manifest_d["case_results"]
            prev_prefix_one = copy.deepcopy(base_manifest_d)
            prev_prefix_one["case_results"] = {frozen_order_names[0]: all_case_results[frozen_order_names[0]]}
            curr_skips_ahead = copy.deepcopy(base_manifest_d)
            curr_skips_ahead["case_results"] = {
                frozen_order_names[0]: all_case_results[frozen_order_names[0]],
                frozen_order_names[2]: all_case_results[frozen_order_names[2]],
            }
            reorder_errors = validate_manifest_revision_transition(
                prev_prefix_one, curr_skips_ahead, prev_prefix_one["campaign_id"],
            )
            rec.check(
                "validate_manifest_revision_transition detects case_results gaining an "
                "entry out of the frozen six-case order (Remediation C)",
                any("case_results" in e and "frozen" in e for e in reorder_errors),
                detail=f"errors={reorder_errors}",
            )

            # unknown top-level fields must not bypass transition validation --
            # confirmed at the full load_p14_manifest_chain level, since this is
            # (still, correctly) enforced by the pre-existing schema validator
            # this refactor must not weaken.
            p14_c_unknown, _ = _run_profile_pipeline(tmp_path, "20260728T222900Z")
            _append_tampered_manifest_revision(
                p14_c_unknown, _mutate_top_level_field("some_unclassified_field", "value"),
            )
            raised = False
            try:
                load_p14_manifest_chain(p14_c_unknown)
            except p13.ManifestTransitionError as exc:
                raised = "unknown" in str(exc).lower() or "some_unclassified_field" in str(exc)
            rec.check(
                "load_p14_manifest_chain still rejects an unknown top-level field -- "
                "it can never bypass transition validation (Remediation C)",
                raised,
            )

            # --- Task 4 remediation (Section 9): manifest state-shape validation
            # -- the exact one-legal-introducing-state mutation matrix, and the
            # explicit ordered-list case_results check ------------------------
            def _rewrite_revision_in_place(campaign_dir: Path, revision: int, mutate_fn) -> None:
                """Overwrites an *existing* revision file in place with a
                mutated copy of its own content (same revision number,
                same previous_manifest_sha256) -- used to introduce a pure
                *shape* defect (a field present too early for its own
                declared state) without needing a legal state transition to
                reach a new revision at all; contrast with
                _append_tampered_manifest_revision, which appends revision
                N+1 and therefore also exercises transition validation."""
                path = _manifest_revision_path(campaign_dir, revision)
                doc = json.loads(path.read_text(encoding="utf-8"))
                mutate_fn(doc)
                path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")

            def _check_shape_rejected_at_rev0(label: str, campaign_id: str, mutate_fn, expect_substr: str) -> None:
                p14_shape = _do_init_campaign(campaign_id=campaign_id, started_at_utc=campaign_id)
                _rewrite_revision_in_place(p14_shape, 0, mutate_fn)
                raised_shape, errmsg = False, ""
                try:
                    load_p14_manifest_chain(p14_shape)
                except p13.ManifestTransitionError as exc:
                    raised_shape, errmsg = True, str(exc)
                rec.check(
                    f"manifest state-shape validation rejects {label} (Task 4, Section 9)",
                    raised_shape and expect_substr in errmsg,
                    detail=f"raised={raised_shape} errmsg={errmsg}",
                )

            _check_shape_rejected_at_rev0(
                "resolved_ncu_metrics present while state=PILOT_IN_PROGRESS", "20260728T230100Z",
                lambda d: d.__setitem__("resolved_ncu_metrics", {
                    "requested": [], "resolved": [], "missing": [],
                    "dram_read_metric": None, "dram_read_metric_available": False,
                }),
                "resolved_ncu_metrics",
            )
            _check_shape_rejected_at_rev0(
                "profile_completed_at_utc present while state=PILOT_IN_PROGRESS", "20260728T230200Z",
                lambda d: d.__setitem__("profile_completed_at_utc", "20260728T230200Z"),
                "profile_completed_at_utc",
            )
            _check_shape_rejected_at_rev0(
                "preflight_reference_profile present while state=PILOT_IN_PROGRESS", "20260728T230300Z",
                lambda d: d.__setitem__("preflight_reference_profile", {}),
                "preflight_reference_profile",
            )
            _check_shape_rejected_at_rev0(
                "failure_stage/failure_detail present while state=PILOT_IN_PROGRESS (not FAILED/INTERRUPTED)",
                "20260728T230400Z",
                lambda d: d.__setitem__("failure_stage", "some_stage"),
                "failure_stage",
            )
            # case_results ordering is tested directly against
            # validate_manifest_state_shape() rather than through a JSON
            # round-trip: json.dumps(..., sort_keys=True) (used by both
            # write_next_p14_manifest_revision and this self-test's own
            # _rewrite_revision_in_place/_append_tampered_manifest_revision
            # helpers) recursively re-sorts every nested dict's keys,
            # including case_results itself -- and the frozen case names
            # ("00_...", "01_...", ..., "05_...") happen to already sort
            # alphabetically into the correct frozen order, so a reordering
            # attempt written through any of those helpers would be silently
            # undone before ever reaching disk. Calling the pure function
            # directly with an in-memory dict (never serialized) is the
            # only way to observe a genuinely reordered case_results at all.
            p14_reorder_base, _ = _run_profile_pipeline(tmp_path, "20260728T230500Z")
            reorder_manifest, _reorder_rev = load_p14_manifest_chain(p14_reorder_base)
            reordered_case_results = {
                name: reorder_manifest["case_results"][name]
                for name in reversed(list(reorder_manifest["case_results"]))
            }
            reordered_manifest = dict(reorder_manifest)
            reordered_manifest["case_results"] = reordered_case_results
            shape_errors = validate_manifest_state_shape(reordered_manifest, reorder_manifest["campaign_id"])
            rec.check(
                "validate_manifest_state_shape rejects a reordered (but set-identical) "
                "case_results -- list(case_results) is checked explicitly, never reduced to a "
                "set comparison (Task 4, Section 9)",
                any("case_results key order" in e for e in shape_errors),
                detail=f"errors={shape_errors}",
            )
            rec.check(
                "validate_manifest_state_shape finds no ordering error on the untampered, "
                "correctly-ordered manifest",
                not any("case_results key order" in e for e in validate_manifest_state_shape(
                    reorder_manifest, reorder_manifest["campaign_id"],
                )),
            )
            # End-to-end confirmation through the full load_p14_manifest_chain
            # pipeline: hand-write the revision file's raw JSON text with
            # sort_keys=False (an attacker with direct filesystem access is
            # not obligated to route through json.dumps(sort_keys=True) the
            # way this codebase's own writer does), so the on-disk bytes
            # genuinely preserve the reversed case_results order.
            reorder_rev_path = _manifest_revision_path(p14_reorder_base, _reorder_rev)
            reorder_doc = json.loads(reorder_rev_path.read_text(encoding="utf-8"))
            reorder_doc["case_results"] = reordered_case_results
            reorder_rev_path.write_text(
                json.dumps(reorder_doc, indent=2, sort_keys=False) + "\n", encoding="utf-8",
            )
            raised, errmsg = False, ""
            try:
                load_p14_manifest_chain(p14_reorder_base)
            except p13.ManifestTransitionError as exc:
                raised, errmsg = True, str(exc)
            rec.check(
                "load_p14_manifest_chain end-to-end rejects a manifest revision file whose "
                "on-disk case_results are genuinely reordered (Task 4, Section 9)",
                raised and "case_results key order" in errmsg,
                detail=f"raised={raised} errmsg={errmsg}",
            )

            p14_count_base, _ = _run_profile_pipeline(tmp_path, "20260728T230600Z")
            count_rev = load_p14_manifest_chain(p14_count_base)[1]
            _rewrite_revision_in_place(
                p14_count_base, count_rev,
                lambda d: d.__setitem__("profile_count_completed", 99),
            )
            raised, errmsg = False, ""
            try:
                load_p14_manifest_chain(p14_count_base)
            except p13.ManifestTransitionError as exc:
                raised, errmsg = True, str(exc)
            rec.check(
                "profile_count_completed disagreeing with len(case_results) is rejected "
                "(Task 4, Section 9)",
                raised and "profile_count_completed" in errmsg,
                detail=f"raised={raised} errmsg={errmsg}",
            )

            # "analysis hash before ANALYZED": artifact_sha256 gaining an
            # "analysis/"-prefixed key while state is still COMPLETE (not yet
            # ANALYZED) -- rewriting the COMPLETE campaign's own latest
            # revision in place, since COMPLETE has no self-loop to append a
            # same-state revision through.
            p14_analysis_early, _ = _run_profile_pipeline(tmp_path, "20260728T230700Z")
            ok, errors = _do_finalize_profile(campaign_dir=p14_analysis_early, completed_at_utc="20260728T230701Z")
            assert ok, errors
            _latest_manifest, _latest_rev = load_p14_manifest_chain(p14_analysis_early)
            _rewrite_revision_in_place(
                p14_analysis_early, _latest_rev,
                lambda d: d.setdefault("artifact_sha256", {}).__setitem__("analysis/early.csv", "0" * 64),
            )
            raised, errmsg = False, ""
            try:
                load_p14_manifest_chain(p14_analysis_early)
            except p13.ManifestTransitionError as exc:
                raised, errmsg = True, str(exc)
            rec.check(
                "an analysis/-prefixed artifact_sha256 entry while state=COMPLETE (not yet "
                "ANALYZED) is rejected (Task 4, Section 9: analysis hash before ANALYZED)",
                raised and "analysis" in errmsg,
                detail=f"raised={raised} errmsg={errmsg}",
            )

            # Same-state no-op revision, and two cases appended in one revision:
            # both would previously pass ("curr_k >= prev_k" alone permits
            # curr_k == prev_k and curr_k == prev_k + 2, not just the correct
            # curr_k == prev_k + 1); appended (not rewritten in place) since
            # PROFILE_IN_PROGRESS -> PROFILE_IN_PROGRESS *is* otherwise legal.
            p14_noop = _do_init_campaign(campaign_id="20260728T230800Z", started_at_utc="20260728T230800Z")
            ok, errors = _do_record_pilot(
                campaign_dir=p14_noop, p13_campaign_dir=_build_p13_pilot_campaign_fixture(tmp_path, "noop_base")[0],
                preflight_path=good_preflight_rp, git_commit=_FIXED_GIT_COMMIT,
                completed_at_utc="20260728T230801Z", now_utc=now_rp,
            )
            assert ok, errors
            disc_noop = tmp_path / "disc_noop.log"
            disc_noop.write_text("\n".join(CANDIDATE_METRICS) + "\n", encoding="utf-8")
            ok, errors, _ = _do_discover_metrics(
                campaign_dir=p14_noop, discovery_log=disc_noop, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, started_at_utc="20260728T230802Z", now_utc=now_rp,
            )
            assert ok, errors
            noop_entry0 = build_ncu_plan()[0]
            _build_ncu_case_fixture(tmp_path, noop_entry0, case_dir=p14_noop / "profiles" / noop_entry0["case_name"])
            ok, errors = _do_validate_profile_case(campaign_dir=p14_noop, index=0, git_commit=_FIXED_GIT_COMMIT)
            assert ok, errors

            _append_tampered_manifest_revision(p14_noop, lambda m: None)  # no-op: case_results unchanged
            raised, errmsg = False, ""
            try:
                load_p14_manifest_chain(p14_noop)
            except p13.ManifestTransitionError as exc:
                raised, errmsg = True, str(exc)
            rec.check(
                "a same-state PROFILE_IN_PROGRESS no-op revision (case_results unchanged) is "
                "rejected (Task 4, Section 9)",
                raised and "exactly one" in errmsg,
                detail=f"raised={raised} errmsg={errmsg}",
            )

            p14_twocase = _do_init_campaign(campaign_id="20260728T230900Z", started_at_utc="20260728T230900Z")
            ok, errors = _do_record_pilot(
                campaign_dir=p14_twocase,
                p13_campaign_dir=_build_p13_pilot_campaign_fixture(tmp_path, "twocase_base")[0],
                preflight_path=good_preflight_rp, git_commit=_FIXED_GIT_COMMIT,
                completed_at_utc="20260728T230901Z", now_utc=now_rp,
            )
            assert ok, errors
            disc_twocase = tmp_path / "disc_twocase.log"
            disc_twocase.write_text("\n".join(CANDIDATE_METRICS) + "\n", encoding="utf-8")
            ok, errors, _ = _do_discover_metrics(
                campaign_dir=p14_twocase, discovery_log=disc_twocase, preflight_path=good_preflight_rp,
                git_commit=_FIXED_GIT_COMMIT, started_at_utc="20260728T230902Z", now_utc=now_rp,
            )
            assert ok, errors
            twocase_entry0 = build_ncu_plan()[0]
            twocase_entry1 = build_ncu_plan()[1]
            twocase_dir0 = p14_twocase / "profiles" / twocase_entry0["case_name"]
            twocase_dir1 = p14_twocase / "profiles" / twocase_entry1["case_name"]
            _build_ncu_case_fixture(tmp_path, twocase_entry0, case_dir=twocase_dir0)
            _build_ncu_case_fixture(tmp_path, twocase_entry1, case_dir=twocase_dir1)
            # Starting from zero validated cases (no legitimate
            # validate-profile-case call at all), the tampered revision
            # below jumps directly from 0 to 2 case_results entries in one
            # step -- skipping the one legitimate intermediate revision the
            # real workflow always produces (one case at a time).
            twocase_manifest, _rev = load_p14_manifest_chain(p14_twocase)
            assert twocase_manifest.get("case_results", {}) == {}, "fixture must start with zero validated cases"

            def _add_two_cases_at_once(m: dict) -> None:
                for entry, case_dir in ((twocase_entry0, twocase_dir0), (twocase_entry1, twocase_dir1)):
                    result, recon_errs = reconstruct_case_result(
                        entry=entry,
                        application_csv=case_dir / f"{entry['case_name']}.application.csv",
                        metrics_csv=case_dir / f"{entry['case_name']}.metrics_raw.csv",
                        ncu_rep=case_dir / f"{entry['case_name']}_report.ncu-rep",
                        provenance=m["provenance"], resolved_ncu_metrics=m["resolved_ncu_metrics"],
                        git_commit=_FIXED_GIT_COMMIT,
                    )
                    assert result is not None, f"self-test fixture: reconstruct_case_result failed: {recon_errs}"
                    m.setdefault("case_results", {})[entry["case_name"]] = result
                m["profile_count_completed"] = len(m["case_results"])

            _append_tampered_manifest_revision(p14_twocase, _add_two_cases_at_once)
            raised, errmsg = False, ""
            try:
                load_p14_manifest_chain(p14_twocase)
            except p13.ManifestTransitionError as exc:
                raised, errmsg = True, str(exc)
            rec.check(
                "a PROFILE_IN_PROGRESS revision that appends two new cases at once (skipping the "
                "one-case-at-a-time intermediate revision) is rejected (Task 4, Section 9)",
                raised and "exactly one" in errmsg,
                detail=f"raised={raised} errmsg={errmsg}",
            )

            # =================================================================
            # Independent re-audit remediation (pragmatic, final scope): Groups
            # A-D. Each group closes exactly one blocker found against Git
            # commit a66d0fa8b37147eb4f237911c42b02e3c8cbed59.
            # =================================================================
            _reaudit_ts_state = {"n": 0}
            _reaudit_ts_base = _datetime(2026, 7, 29, 12, 0, 0, tzinfo=_timezone.utc)

            def _next_reaudit_ts() -> str:
                # A monotonically increasing real UTC instant, one second per
                # call -- avoids ever hand-typing an invalid HHMMSS literal
                # (e.g. minute/second fields >= 60) among the many distinct
                # campaign IDs and timestamp values Groups A-D need below.
                _reaudit_ts_state["n"] += 1
                return (
                    _reaudit_ts_base + _timedelta(seconds=_reaudit_ts_state["n"])
                ).strftime("%Y%m%dT%H%M%SZ")

            # --- Group A: premature null-valued manifest fields. Legality
            # must depend on key *presence*, not truthiness/non-null value --
            # `key in current and current[key] is not None` treated a
            # premature key holding an explicit `null` as absent. Every
            # revision constructed below is revision 0 of its own fresh
            # campaign, produced by the real, audited init-campaign path and
            # then mutated *in place* (same revision number, same
            # previous_manifest_sha256=null) -- so the hash chain itself
            # remains valid and any rejection proves semantic validation,
            # never merely a broken chain. ------------------------------
            _group_a_null_fields = [
                "pilot_completed_at_utc", "profile_started_at_utc",
                "profile_completed_at_utc", "analyzed_at_utc",
                "resolved_ncu_metrics", "case_results", "artifact_sha256",
                "failure_stage", "failure_detail", "unknown_field",
            ]
            for _field in _group_a_null_fields:
                _campaign_id = _next_reaudit_ts()
                _p14_null = _do_init_campaign(campaign_id=_campaign_id, started_at_utc=_campaign_id)
                _rewrite_revision_in_place(_p14_null, 0, lambda d, f=_field: d.__setitem__(f, None))
                _raised = False
                _errmsg = ""
                try:
                    load_p14_manifest_chain(_p14_null)
                except p13.ManifestTransitionError as _exc:
                    _raised, _errmsg = True, str(_exc)
                rec.check(
                    f"revision zero with {_field}=null is rejected, not treated as absent (Group A)",
                    _raised,
                    detail=f"errmsg={_errmsg}",
                )

            # A required timestamp (started_at_utc) set to null in its own
            # (revision-zero) state must also be rejected -- not merely a
            # premature field appearing too early, but a required field
            # whose declared type no longer tolerates null at all.
            _null_started_id = _next_reaudit_ts()
            _p14_null_started = _do_init_campaign(
                campaign_id=_null_started_id, started_at_utc=_null_started_id,
            )
            _rewrite_revision_in_place(_p14_null_started, 0, lambda d: d.__setitem__("started_at_utc", None))
            _raised = False
            try:
                load_p14_manifest_chain(_p14_null_started)
            except p13.ManifestTransitionError:
                _raised = True
            rec.check(
                "a required timestamp (started_at_utc) set to null in its own state is rejected "
                "(Group A)",
                _raised,
            )

            # Revision zero must contain *exactly* the None -> PILOT_IN_PROGRESS
            # initialization fields -- confirmed positively (the untampered
            # fixture above has no shape errors) and negatively (a premature,
            # non-null, otherwise well-typed later-phase field is already
            # covered by the existing "manifest state-shape validation
            # rejects ... while state=PILOT_IN_PROGRESS" checks above; this
            # positive check guards against a future over-broad fix that
            # rejected *everything*).
            _p14_a_baseline, _ = _run_profile_pipeline(tmp_path, _next_reaudit_ts())
            _baseline_rev0, _ = load_p14_manifest_chain(_p14_a_baseline)
            rec.check(
                "an untampered, fully-progressed manifest still passes state-shape validation "
                "(Group A fix is not over-broad)",
                validate_manifest_state_shape(_baseline_rev0, _baseline_rev0["campaign_id"]) == [],
                detail=f"errors={validate_manifest_state_shape(_baseline_rev0, _baseline_rev0['campaign_id'])}",
            )

            # --- Group B: chronological timestamp validation. Reuses
            # last_manifest (loaded above from the fully ANALYZED p14_det
            # campaign), which carries a real value for all five lifecycle
            # timestamps. -------------------------------------------------
            rec.check(
                "validate_manifest_timestamp_chronology finds no errors on an untampered, "
                "fully-progressed ANALYZED manifest",
                validate_manifest_timestamp_chronology(last_manifest) == [],
                detail=f"errors={validate_manifest_timestamp_chronology(last_manifest)}",
            )

            def _one_second_before(value: str) -> str:
                dt = _datetime.strptime(value, "%Y%m%dT%H%M%SZ") - _timedelta(seconds=1)
                return dt.strftime("%Y%m%dT%H%M%SZ")

            def _check_b_invariant(label: str, field: str) -> None:
                mutated = dict(last_manifest)
                mutated[field] = _one_second_before(last_manifest[field])
                chrono_errors = validate_manifest_timestamp_chronology(mutated)
                rec.check(
                    f"validate_manifest_timestamp_chronology rejects {label} (Group B)",
                    any(field in e for e in chrono_errors),
                    detail=f"errors={chrono_errors}",
                )

            _check_b_invariant("pilot_completed_at_utc < started_at_utc", "pilot_completed_at_utc")
            _check_b_invariant("profile_started_at_utc < pilot_completed_at_utc", "profile_started_at_utc")
            _check_b_invariant("profile_completed_at_utc < profile_started_at_utc", "profile_completed_at_utc")
            _check_b_invariant("analyzed_at_utc < profile_completed_at_utc", "analyzed_at_utc")

            # End-to-end: an earlier timestamp introduced in a later revision,
            # even with otherwise-valid syntax, must be rejected by
            # load_p14_manifest_chain itself, not merely by the pure
            # function in isolation -- and the mutated revision remains
            # cryptographically valid (same revision number, same
            # previous_manifest_sha256), so this proves semantic rejection.
            _p14_det_rev0, _p14_det_last_rev = load_p14_manifest_chain(p14_det)
            _rewrite_revision_in_place(
                p14_det, _p14_det_last_rev,
                lambda d: d.__setitem__(
                    "analyzed_at_utc", _one_second_before(d["profile_completed_at_utc"]),
                ),
            )
            _raised, _errmsg = False, ""
            try:
                load_p14_manifest_chain(p14_det)
            except p13.ManifestTransitionError as _exc:
                _raised, _errmsg = True, str(_exc)
            rec.check(
                "load_p14_manifest_chain rejects an earlier timestamp (analyzed_at_utc) "
                "introduced in the final revision, chained correctly but preceding an earlier "
                "lifecycle timestamp (Group B)",
                _raised and "chronology" in _errmsg,
                detail=f"raised={_raised} errmsg={_errmsg}",
            )

            # --- Group C: profile_count_completed / case_results must always
            # appear together, and their value relationship holds exactly.
            # Uses reorder_manifest (a real PROFILE_IN_PROGRESS-state
            # manifest with all six cases validated) as a base for pure-
            # function isolation tests, mirroring the existing reordering
            # test's own style. ---------------------------------------------
            _some_case_results = {
                k: v for i, (k, v) in enumerate(reorder_manifest["case_results"].items()) if i == 0
            }

            _count_absent = dict(reorder_manifest)
            _count_absent["case_results"] = _some_case_results
            del _count_absent["profile_count_completed"]
            _errs = validate_manifest_state_shape(_count_absent, reorder_manifest["campaign_id"])
            rec.check(
                "case_results present but profile_count_completed absent is rejected (Group C)",
                any("must always appear together" in e for e in _errs),
                detail=f"errors={_errs}",
            )

            _case_results_absent = dict(reorder_manifest)
            del _case_results_absent["case_results"]
            _case_results_absent["profile_count_completed"] = 1
            _errs = validate_manifest_state_shape(_case_results_absent, reorder_manifest["campaign_id"])
            rec.check(
                "profile_count_completed present but case_results absent is rejected (Group C)",
                any("must always appear together" in e for e in _errs),
                detail=f"errors={_errs}",
            )

            _count_one_empty = dict(reorder_manifest)
            _count_one_empty["case_results"] = {}
            _count_one_empty["profile_count_completed"] = 1
            _errs = validate_manifest_state_shape(_count_one_empty, reorder_manifest["campaign_id"])
            rec.check(
                "profile_count_completed=1 with empty case_results is rejected (Group C)",
                any("!= len(case_results)" in e for e in _errs),
                detail=f"errors={_errs}",
            )

            _count_zero_one_case = dict(reorder_manifest)
            _count_zero_one_case["case_results"] = _some_case_results
            _count_zero_one_case["profile_count_completed"] = 0
            _errs = validate_manifest_state_shape(_count_zero_one_case, reorder_manifest["campaign_id"])
            rec.check(
                "profile_count_completed=0 with one case_results entry is rejected (Group C)",
                any("!= len(case_results)" in e for e in _errs),
                detail=f"errors={_errs}",
            )

            # count=true/-1/7: end-to-end through load_p14_manifest_chain, on
            # a fresh six-case-validated campaign each, since these are base
            # schema/type/range violations (bool is never accepted in place
            # of a canonical int; the range is [0, 6]), not shape-in-
            # isolation violations.
            def _check_count_rejected(label: str, campaign_id: str, bad_count) -> None:
                p14_ct, _ = _run_profile_pipeline(tmp_path, campaign_id)
                ct_rev = load_p14_manifest_chain(p14_ct)[1]
                _rewrite_revision_in_place(
                    p14_ct, ct_rev, lambda d: d.__setitem__("profile_count_completed", bad_count),
                )
                raised_ct = False
                try:
                    load_p14_manifest_chain(p14_ct)
                except p13.ManifestTransitionError:
                    raised_ct = True
                rec.check(f"profile_count_completed={bad_count!r} is rejected ({label}) (Group C)", raised_ct)

            _check_count_rejected("count=true, a bool masquerading as int", _next_reaudit_ts(), True)
            _check_count_rejected("count=-1", _next_reaudit_ts(), -1)
            _check_count_rejected("count=7", _next_reaudit_ts(), 7)

            # COMPLETE/ANALYZED both require exactly six cases in frozen
            # order and count 6 -- COMPLETE via _validate_p14_manifest_document's
            # own explicit check; ANALYZED (which can never legally *shrink*
            # case_results below what COMPLETE already required six of, since
            # case_results is append-only) verified end-to-end by shrinking
            # the already-ANALYZED p14_det campaign's final revision in
            # place and confirming the chain-level append-only invariant
            # still catches it.
            _complete_five_cases = copy.deepcopy(base_manifest_d)
            _removed_case_name = next(iter(_complete_five_cases["case_results"]))
            _complete_five_cases["case_results"].pop(_removed_case_name)
            _complete_five_cases["profile_count_completed"] = 5
            _raised = False
            try:
                _validate_p14_manifest_document(_complete_five_cases, require_initialized=True)
            except p13.ManifestTransitionError:
                _raised = True
            rec.check("COMPLETE with only five (of six) cases is rejected (Group C)", _raised)

            p14_analyzed_shrink, _ = _run_profile_pipeline(tmp_path, _next_reaudit_ts())
            ok, errors = _do_finalize_profile(
                campaign_dir=p14_analyzed_shrink, completed_at_utc=_next_reaudit_ts(),
            )
            assert ok, errors
            ok, errors = _do_analyze(campaign_dir=p14_analyzed_shrink, analyzed_at_utc=_next_reaudit_ts())
            assert ok, errors
            _shrink_manifest, _shrink_rev = load_p14_manifest_chain(p14_analyzed_shrink)

            def _shrink_case_results(d: dict) -> None:
                removed_name = next(iter(d["case_results"]))
                d["case_results"].pop(removed_name)
                d["profile_count_completed"] = len(d["case_results"])

            _rewrite_revision_in_place(p14_analyzed_shrink, _shrink_rev, _shrink_case_results)
            _raised, _errmsg = False, ""
            try:
                load_p14_manifest_chain(p14_analyzed_shrink)
            except p13.ManifestTransitionError as _exc:
                _raised, _errmsg = True, str(_exc)
            rec.check(
                "ANALYZED with fewer than six cases (case_results shrunk on the final revision) "
                "is rejected end-to-end (Group C)",
                _raised,
                detail=f"raised={_raised} errmsg={_errmsg}",
            )

            # --- Group D: the second, pre-ANALYZED evidence-integrity gate
            # must catch evidence mutated *between* the first gate and
            # analysis publication -- not merely evidence already tampered
            # before analyze() starts (already covered by the Remediation B
            # _check_tamper_rejected table above). Uses the test-only
            # _test_hook_before_final_gate, fired after the analysis
            # artifacts are published and hashed but immediately before the
            # second integrity gate -- never exposed as a production CLI
            # option. ---------------------------------------------------
            p14_gate2, _ = _run_profile_pipeline(tmp_path, _next_reaudit_ts())
            ok, errors = _do_finalize_profile(campaign_dir=p14_gate2, completed_at_utc=_next_reaudit_ts())
            assert ok, errors
            gate2_manifest, _ = load_p14_manifest_chain(p14_gate2)
            gate2_p13_dir = REPO_ROOT / gate2_manifest["pilot_campaign_reference"]["path"]
            gate2_combined_path = gate2_p13_dir / "combined_samples.csv"
            gate2_original_combined = gate2_combined_path.read_bytes()
            gate2_hook_called = {"n": 0}

            def _tamper_between_gates() -> None:
                gate2_hook_called["n"] += 1
                gate2_combined_path.write_bytes(gate2_original_combined + b"\ntampered_between_gates\n")

            ok, errors = _do_analyze(
                campaign_dir=p14_gate2, analyzed_at_utc=_next_reaudit_ts(),
                _test_hook_before_final_gate=_tamper_between_gates,
            )
            gate2_analysis_dir = p14_gate2 / "analysis"
            gate2_published = (
                [p for p in gate2_analysis_dir.rglob("*") if p.is_file()] if gate2_analysis_dir.exists() else []
            )
            gate2_state_after = load_p14_manifest_chain(p14_gate2)[0].get("state")
            rec.check(
                "the second (pre-ANALYZED) integrity gate rejects evidence mutated after the "
                "first gate passed but before ANALYZED is published, detected via a deterministic "
                "test-only hook fired between analysis publication and the final gate; no "
                "analysis artifact survives and the campaign remains COMPLETE, not ANALYZED "
                "(Group D)",
                gate2_hook_called["n"] == 1 and not ok and not gate2_published
                and gate2_state_after == "COMPLETE",
                detail=f"ok={ok} errors={errors} published={gate2_published} state={gate2_state_after}",
            )
            gate2_combined_path.write_bytes(gate2_original_combined)

            # --- Final independent-audit regressions (commit 3d92a6b).
            # These are deliberately full-chain tests whose forged revisions
            # retain a correct manifest_revision/previous_manifest_sha256
            # relationship. Rejection must therefore come from the semantic
            # transition/content rules, never from a conveniently broken hash
            # chain. -------------------------------------------------------

            # The runner installs its interruption trap before --profile
            # leaves PILOT_COMPLETE. A signal in that interval must be
            # recordable with the strict list[str] failure_detail schema.
            _interrupt_id = _next_reaudit_ts()
            _interrupt_p13, _ = _build_p13_pilot_campaign_fixture(tmp_path, _interrupt_id)
            _interrupt_p14 = _do_init_campaign(
                campaign_id=_interrupt_id, started_at_utc="20260728T100000Z",
            )
            ok, errors = _do_record_pilot(
                campaign_dir=_interrupt_p14, p13_campaign_dir=_interrupt_p13,
                preflight_path=good_preflight_rp, git_commit=_FIXED_GIT_COMMIT,
                completed_at_utc="20260728T103000Z", now_utc=now_rp,
            )
            assert ok, errors
            _interrupt_recorded = False
            _interrupt_error = ""
            try:
                p14_merge_manifest(
                    _interrupt_p14,
                    {
                        "failure_stage": "signal_TERM",
                        "failure_detail": [],
                    },
                    state="INTERRUPTED",
                )
                _interrupt_recorded = (
                    load_p14_manifest_chain(_interrupt_p14)[0].get("state") == "INTERRUPTED"
                )
            except (p13.ManifestTransitionError, p13.UnsafePathError) as _exc:
                _interrupt_error = str(_exc)
            rec.check(
                "PILOT_COMPLETE -> INTERRUPTED records a runner signal with list[str] "
                "failure_detail (final audit: failure telemetry)",
                _interrupt_recorded,
                detail=f"error={_interrupt_error}",
            )

            # One real six-case PROFILE_IN_PROGRESS campaign supplies a
            # canonical case result and correctly hashed adjacent revisions.
            # Its immutable revision bytes are saved so three independent
            # forged-chain scenarios can be exercised without rebuilding the
            # expensive fixture.
            _matrix_campaign, _ = _run_profile_pipeline(tmp_path, _next_reaudit_ts())
            _matrix_paths = sorted((_matrix_campaign / "manifest").glob("*.json"))
            _matrix_bytes = {i: path.read_bytes() for i, path in enumerate(_matrix_paths)}
            _matrix_docs = [
                {k: v for k, v in json.loads(_matrix_bytes[i]).items() if k not in MANIFEST_REVISION_KEYS}
                for i in range(len(_matrix_paths))
            ]
            _matrix_plan = build_ncu_plan()
            _matrix_first_name = _matrix_plan[0]["case_name"]
            _matrix_sixth_name = _matrix_plan[-1]["case_name"]
            _matrix_first_result = copy.deepcopy(
                _matrix_docs[3]["case_results"][_matrix_first_name]
            )
            _matrix_sixth_result = copy.deepcopy(
                _matrix_docs[8]["case_results"][_matrix_sixth_name]
            )

            def _restore_matrix_revisions(first: int = 0) -> None:
                for _i in range(first, len(_matrix_paths)):
                    _manifest_revision_path(_matrix_campaign, _i).write_bytes(_matrix_bytes[_i])

            def _drop_matrix_revisions(first: int) -> None:
                for _i in range(first, len(_matrix_paths)):
                    _path = _manifest_revision_path(_matrix_campaign, _i)
                    if _path.exists():
                        _path.unlink()

            # PILOT_COMPLETE -> PROFILE_IN_PROGRESS may introduce only the
            # profiling preconditions. The first case belongs exclusively to
            # the first PROFILE_IN_PROGRESS self-loop.
            _drop_matrix_revisions(2)

            def _forge_profile_entry_with_case(d: dict) -> None:
                d["state"] = "PROFILE_IN_PROGRESS"
                for _field in (
                    "profile_started_at_utc",
                    "resolved_ncu_metrics",
                    "preflight_reference_profile",
                ):
                    d[_field] = copy.deepcopy(_matrix_docs[2][_field])
                d["case_results"] = {_matrix_first_name: copy.deepcopy(_matrix_first_result)}
                d["profile_count_completed"] = 1

            _append_tampered_manifest_revision(
                _matrix_campaign, _forge_profile_entry_with_case,
            )
            _raised, _errmsg = False, ""
            try:
                load_p14_manifest_chain(_matrix_campaign)
            except p13.ManifestTransitionError as _exc:
                _raised, _errmsg = True, str(_exc)
            rec.check(
                "PILOT_COMPLETE -> PROFILE_IN_PROGRESS cannot introduce the first case "
                "(final audit: exact transition matrix)",
                _raised and "transition" in _errmsg,
                detail=f"raised={_raised} errmsg={_errmsg}",
            )
            _drop_matrix_revisions(2)
            _restore_matrix_revisions(2)

            # PROFILE_IN_PROGRESS(5) -> COMPLETE may add terminal metadata
            # only; it must not smuggle in the sixth profile result.
            _drop_matrix_revisions(8)

            def _canonical_base_artifacts_for_test(d: dict) -> dict[str, str]:
                artifacts = {
                    "profile_plan.csv": d["profile_plan_sha256"],
                    "pilot_manifest_sha256": d["pilot_campaign_reference"]["manifest_sha256"],
                    "pilot_combined_samples_sha256": d["pilot_campaign_reference"][
                        "combined_samples_sha256"
                    ],
                    "pilot_summary_sha256": d["pilot_campaign_reference"]["summary_sha256"],
                    "preflight_reference_pilot": d["preflight_reference_pilot"]["sha256"],
                    "preflight_reference_profile": d["preflight_reference_profile"]["sha256"],
                }
                for _entry in build_ncu_plan():
                    _name = _entry["case_name"]
                    for _hash_field in (
                        "application_csv_sha256",
                        "metrics_csv_sha256",
                        "ncu_rep_sha256",
                    ):
                        artifacts[f"{_name}.{_hash_field}"] = d["case_results"][_name][
                            _hash_field
                        ]
                return artifacts

            def _forge_complete_with_sixth(d: dict) -> None:
                d["case_results"][_matrix_sixth_name] = copy.deepcopy(_matrix_sixth_result)
                d["profile_count_completed"] = 6
                d["profile_completed_at_utc"] = "20260728T103200Z"
                d["profile_order"] = copy.deepcopy(_matrix_plan)
                d["state"] = "COMPLETE"
                d["artifact_sha256"] = _canonical_base_artifacts_for_test(d)

            _append_tampered_manifest_revision(_matrix_campaign, _forge_complete_with_sixth)
            _raised, _errmsg = False, ""
            try:
                load_p14_manifest_chain(_matrix_campaign)
            except p13.ManifestTransitionError as _exc:
                _raised, _errmsg = True, str(_exc)
            rec.check(
                "PROFILE_IN_PROGRESS(5) -> COMPLETE cannot add the sixth case "
                "(final audit: exact transition matrix)",
                _raised and "transition" in _errmsg,
                detail=f"raised={_raised} errmsg={_errmsg}",
            )
            _drop_matrix_revisions(8)
            _restore_matrix_revisions(8)

            # A failure revision preserves the exact already-recorded prefix;
            # it may add only state/failure telemetry, never new progress.
            _drop_matrix_revisions(3)

            def _forge_failed_with_progress(d: dict) -> None:
                d["state"] = "FAILED"
                d["failure_stage"] = "synthetic_failure"
                d["failure_detail"] = ["synthetic"]
                d["case_results"] = {_matrix_first_name: copy.deepcopy(_matrix_first_result)}
                d["profile_count_completed"] = 1

            _append_tampered_manifest_revision(_matrix_campaign, _forge_failed_with_progress)
            _raised, _errmsg = False, ""
            try:
                load_p14_manifest_chain(_matrix_campaign)
            except p13.ManifestTransitionError as _exc:
                _raised, _errmsg = True, str(_exc)
            rec.check(
                "PROFILE_IN_PROGRESS -> FAILED cannot invent profile progress "
                "(final audit: exact transition matrix)",
                _raised and "transition" in _errmsg,
                detail=f"raised={_raised} errmsg={_errmsg}",
            )

            # COMPLETE must carry the exact frozen profile order and the
            # exact canonical base-evidence hash map. Exercise three
            # independent corruptions against the same valid terminal
            # revision, restoring its original immutable bytes after each.
            _terminal_campaign = p14_d_table
            _terminal_manifest, _terminal_revision = load_p14_manifest_chain(_terminal_campaign)
            _terminal_path = _manifest_revision_path(_terminal_campaign, _terminal_revision)
            _terminal_original = _terminal_path.read_bytes()

            def _terminal_mutation_rejected(label: str, mutate_fn, needle: str) -> None:
                _terminal_path.write_bytes(_terminal_original)
                _rewrite_revision_in_place(_terminal_campaign, _terminal_revision, mutate_fn)
                _bad_raised, _bad_error = False, ""
                try:
                    load_p14_manifest_chain(_terminal_campaign)
                except p13.ManifestTransitionError as _exc:
                    _bad_raised, _bad_error = True, str(_exc)
                rec.check(
                    f"{label} is rejected (final audit: canonical COMPLETE content)",
                    _bad_raised and needle in _bad_error,
                    detail=f"raised={_bad_raised} errmsg={_bad_error}",
                )
                _terminal_path.write_bytes(_terminal_original)

            _terminal_mutation_rejected(
                "a reversed profile_order",
                lambda d: d.__setitem__("profile_order", list(reversed(d["profile_order"]))),
                "profile_order",
            )
            _terminal_mutation_rejected(
                "an incomplete artifact_sha256 map",
                lambda d: d.__setitem__(
                    "artifact_sha256",
                    {"profile_plan.csv": d["artifact_sha256"]["profile_plan.csv"]},
                ),
                "artifact_sha256",
            )

            def _corrupt_one_terminal_hash(d: dict) -> None:
                _key = next(iter(d["artifact_sha256"]))
                d["artifact_sha256"][_key] = "0" * 64

            _terminal_mutation_rejected(
                "an internally incorrect artifact_sha256 value",
                _corrupt_one_terminal_hash,
                "artifact_sha256",
            )

            ok, errors = _do_analyze(
                campaign_dir=_terminal_campaign, analyzed_at_utc=_next_reaudit_ts(),
            )
            rec.check(
                "a canonical COMPLETE campaign still analyzes successfully after the stricter "
                "terminal validation",
                ok and load_p14_manifest_chain(_terminal_campaign)[0].get("state") == "ANALYZED",
                detail=f"ok={ok} errors={errors}",
            )
            _analyzed_manifest, _analyzed_revision = load_p14_manifest_chain(_terminal_campaign)
            _analyzed_path = _manifest_revision_path(_terminal_campaign, _analyzed_revision)
            _analyzed_doc = json.loads(_analyzed_path.read_text(encoding="utf-8"))
            _analysis_key = next(
                key for key in _analyzed_doc["artifact_sha256"] if key.startswith("analysis/")
            )
            del _analyzed_doc["artifact_sha256"][_analysis_key]
            _analyzed_path.write_text(
                json.dumps(_analyzed_doc, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            _raised, _errmsg = False, ""
            try:
                load_p14_manifest_chain(_terminal_campaign)
            except p13.ManifestTransitionError as _exc:
                _raised, _errmsg = True, str(_exc)
            rec.check(
                "ANALYZED requires the exact complete analysis-artifact inventory "
                "(final audit: canonical ANALYZED content)",
                _raised and "artifact_sha256" in _errmsg,
                detail=f"raised={_raised} errmsg={_errmsg}",
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

    vpp_parser = subparsers.add_parser(
        "validate-profile-preconditions",
        help="GPU-free check that a fresh profiling preflight matches the recorded pilot preflight.",
    )
    vpp_parser.add_argument("--campaign-dir", required=True)
    vpp_parser.add_argument("--preflight", required=True)
    vpp_parser.add_argument("--git-commit", required=True)
    vpp_parser.add_argument("--now", default=None)
    vpp_parser.set_defaults(func=cmd_validate_profile_preconditions)

    vc_parser = subparsers.add_parser("validate-profile-case", help="Validate one captured NCU profile case.")
    vc_parser.add_argument("--campaign-dir", required=True)
    vc_parser.add_argument("--index", required=True, type=int)
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

    mw_parser = subparsers.add_parser(
        "manifest-write",
        help="Mark FAILED/INTERRUPTED with required failure metadata (never a completing state).",
    )
    mw_parser.add_argument("--campaign-dir", required=True)
    mw_parser.add_argument("--status", required=True, choices=("FAILED", "INTERRUPTED"))
    mw_parser.add_argument("--merge-json", required=True)
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
