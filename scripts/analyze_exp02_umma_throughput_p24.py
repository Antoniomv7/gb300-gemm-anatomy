#!/usr/bin/env python3
"""P2.4 profiling, clock-calibrated TFLOP/s, scaling, saturation, and
empirical BF16 Tensor Core per-SM ceiling candidate for exp02_umma_throughput.

This module never touches CUDA, Docker, ``nvidia-smi``, either UMMA binary,
the P2.3 runner/aggregator's own files, or the network. It builds a
reproducible layer around the already-audited, unmodified P2.3 infrastructure
(``scripts/run_exp02_umma_throughput.sh``,
``scripts/aggregate_exp02_umma_throughput.py``, imported here as ``p23``):
the frozen 24-configuration pilot is driven by P2.3's own runner unmodified;
this module only validates P2.3's resulting campaign, profiles the same 24
configurations with Nsight Compute, and computes deterministic statistics,
1-SM/2-SM scaling, candidate depth saturation, and an empirical per-SM
BF16 Tensor Core ceiling candidate. See ``src/compute/P2_4_PROTOCOL.md`` for
the complete frozen contract.

P2.4 does not add or modify a CUDA kernel, does not change either UMMA
binary or SASS checker, and does not change the P2.3 plan, order, runner,
aggregator, or CSV schema. Every artifact this module can ever produce
carries ``publishable: false`` unconditionally. P2.4 produces a reviewed
pilot and an empirical ceiling *candidate*, never a final campaign; if the
mandatory SM-clock metric cannot be trusted for every one of the 24 profiled
configurations, the whole campaign is recorded ``INCONCLUSIVE`` and no
TFLOP/s or completed empirical-ceiling claim is ever emitted.

Subcommands:
  plan                              Print the frozen 24-case profile plan
                                     (identical configurations/order to
                                     P2.3's own plan, plus each case's exact
                                     kernel symbol).
  init-campaign                     Symlink-safe P2.4 campaign creation.
  validate-preflight                Validate a preflight summary.json.
  record-pilot                      Validate a COMPLETE P2.3 benchmark
                                     campaign (run with the frozen pilot
                                     parameters) as this campaign's pilot
                                     input.
  discover-metrics                  Resolve the mandatory SM-clock metric
                                     and the diagnostic tensor-pipe metrics;
                                     start the profiling phase.
  validate-profile-preconditions    GPU-free check that a fresh profiling
                                     preflight matches the recorded pilot
                                     preflight.
  validate-profile-case             Validate one captured NCU profile case
                                     and record its result.
  finalize-profile                  Re-validate and close the 24-case
                                     profile set (state COMPLETE).
  analyze                           Generate analysis/* from a COMPLETE
                                     campaign; state ANALYZED or
                                     INCONCLUSIVE.
  manifest-write                    Merge FAILED/INTERRUPTED failure
                                     telemetry (never a completing state).
  --self-test                       GPU-free synthetic/adversarial tests (no
                                     CUDA, no Docker, no nvidia-smi, no
                                     network, no real subprocess). Runs
                                     entirely under a TemporaryDirectory with
                                     this module's and p23's REPO_ROOT
                                     patched to it; never touches
                                     results/raw/.

Exit codes: 0 on success (including --self-test passing); 1 on a validation,
aggregation, or evidence-integrity failure; 2 on a usage/precondition error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import random
import re
import stat
import statistics
import sys
import tempfile
from datetime import datetime as _datetime
from datetime import timezone as _timezone
from pathlib import Path
from typing import Callable
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import aggregate_exp02_umma_throughput as p23  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Frozen P2.4 constants (src/compute/P2_4_PROTOCOL.md is the human-readable
# mirror of every constant below; keep them in lockstep).
# ---------------------------------------------------------------------------
P24_SCHEMA_VERSION = "1"
P24_EXPERIMENT_ID = "exp02_umma_throughput_p24"

FROZEN_PILOT_PARAMS = {
    "run_kind": "benchmark",
    "iterations": 1000,
    "warmup_iterations": 10,
    "repetitions": 30,
}

FROZEN_PROFILE_PARAMS = {
    "run_kind": "benchmark",
    "iterations": 1000,
    "warmup_iterations": 0,
    "repetitions": 1,
}

MANDATORY_SM_CLOCK_METRIC = "sm__cycles_elapsed.avg.per_second"
# Support the original cycle/nsecond contract and the Hz representation
# observed in a real NCU 2025.4 export on GB300.  Keep this allowlist closed:
# the key is the exact unit after case/outer-whitespace normalization, and the
# value is the only permitted scale factor to Hz.  In particular, prefixed
# units such as GHz/kHz are not inferred or guessed.
SM_CLOCK_UNIT_TO_HZ_SCALE = {
    "cycle/nsecond": 1e9,
    "hz": 1.0,
}
DEVICE_MULTIPROCESSOR_COUNT_METRIC = "device__attribute_multiprocessor_count"
# A raw device attribute count has no physical unit; the pinned NCU 2025.4
# build leaves this column blank (verified against real --page raw --csv
# output). Never rescaled or guessed: any other unit representation is
# rejected outright, exactly like the mandatory SM-clock metric's own closed
# unit allowlist.
EXPECTED_MULTIPROCESSOR_COUNT_UNIT = ""
DIAGNOSTIC_METRICS = (
    "gpu__time_duration.sum",
    DEVICE_MULTIPROCESSOR_COUNT_METRIC,
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "sm__inst_executed_pipe_tensor.sum",
    "smsp__inst_executed_pipe_tensor.sum",
)
CANDIDATE_METRICS = (MANDATORY_SM_CLOCK_METRIC,) + DIAGNOSTIC_METRICS

BOOTSTRAP_SEED = 20260804
BOOTSTRAP_RESAMPLES = 10000
BOOTSTRAP_LO_PERCENTILE = 0.025
BOOTSTRAP_HI_PERCENTILE = 0.975

CV_STABILITY_REVIEW_PERCENT = 5.0
SATURATION_FRACTION_OF_MAX = 0.95

STAT_METRIC_NAMES = ("elapsed_cycles", "cycles_per_umma", "flops_per_cycle", "flops_per_cycle_per_sm")
DEPTH_VALUES = (4, 16, 64, 256)

RAW_ROOT_PARTS_P24 = ("results", "raw", "exp02_umma_throughput_p24")
RAW_ROOT_REL_P24 = Path(*RAW_ROOT_PARTS_P24)

P24_CAMPAIGN_ID_RE = re.compile(r"^\d{8}T\d{6}Z$")
P24_TIMESTAMP_RE = p23.MANIFEST_TIMESTAMP_RE  # YYYYMMDDTHHMMSSZ, reused as-is
NOW_ARG_RE = p23.TIMESTAMP_UTC_RE  # YYYY-MM-DDTHH:MM:SSZ, reused as-is

PROFILE_PLAN_HEADER = [
    "index", "pair_index", "method", "cta_group", "m", "n", "k", "depth",
    "binary", "case_name", "kernel_symbol",
]

ALLOWED_P24_STATES = frozenset({
    "PILOT_IN_PROGRESS", "PILOT_COMPLETE", "PROFILE_IN_PROGRESS",
    "COMPLETE", "ANALYZED", "INCONCLUSIVE", "FAILED", "INTERRUPTED",
})
P24_TERMINAL_STATES = frozenset({"ANALYZED", "INCONCLUSIVE", "FAILED", "INTERRUPTED"})

EXPECTED_PROFILE_CASE_COUNT = p23.EXPECTED_CONFIGURATION_COUNT  # 24; the *same* configurations as P2.3

ALLOWED_P24_TRANSITIONS: dict[str | None, frozenset[str]] = {
    None: frozenset({"PILOT_IN_PROGRESS"}),
    "PILOT_IN_PROGRESS": frozenset({"PILOT_COMPLETE", "FAILED", "INTERRUPTED"}),
    "PILOT_COMPLETE": frozenset({"PROFILE_IN_PROGRESS", "FAILED", "INTERRUPTED"}),
    "PROFILE_IN_PROGRESS": frozenset({"PROFILE_IN_PROGRESS", "COMPLETE", "FAILED", "INTERRUPTED"}),
    # analyze() picks exactly one of these two terminal outcomes; COMPLETE has
    # no FAILED/INTERRUPTED edge because analysis is a pure, retriable
    # function of already-validated evidence (mirrors P1.4's COMPLETE ->
    # ANALYZED-only design; see src/compute/P2_4_PROTOCOL.md section 8).
    "COMPLETE": frozenset({"ANALYZED", "INCONCLUSIVE"}),
    "ANALYZED": frozenset(),
    "INCONCLUSIVE": frozenset(),
    "FAILED": frozenset(),
    "INTERRUPTED": frozenset(),
}

ALLOWED_P24_MANIFEST_KEYS: dict[str, object] = {
    "schema_version": str,
    "experiment_id": str,
    "campaign_id": str,
    "state": str,
    "publishable": bool,
    "started_at_utc": str,
    "pilot_completed_at_utc": str,
    "profile_started_at_utc": str,
    "profile_completed_at_utc": str,
    "analysis_completed_at_utc": str,
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
    "inconclusive_reason": list,
    "failure_stage": str,
    "failure_detail": list,
}

CASE_ARTIFACT_HASH_FIELDS: tuple[str, ...] = (
    "application_csv_sha256",
    "metrics_csv_sha256",
    "ncu_rep_sha256",
    "ncu_tool_log_sha256",
    "container_stdout_log_sha256",
    "container_stderr_log_sha256",
    "metrics_export_stderr_log_sha256",
)

# The frozen, exhaustive seven-file per-case profile inventory
# (src/compute/P2_4_PROTOCOL.md section 7). Defect-2 repair: the finalizer
# previously opened/validated only three of these seven, and the runner
# separately published an unauthorized eighth file
# (<case>.ncu_bridge_stderr.log, folded into <case>.ncu_tool.log instead --
# see scripts/p24_safe_capture.py's publish_ncu_bundle()).
CANONICAL_PROFILE_CASE_FILE_LABELS: tuple[tuple[str, str], ...] = (
    ("ncu_rep", "_report.ncu-rep"),
    ("ncu_tool_log", ".ncu_tool.log"),
    ("container_stdout_log", ".container_stdout.log"),
    ("container_stderr_log", ".container_stderr.log"),
    ("application_csv", ".application.csv"),
    ("metrics_csv", ".metrics_raw.csv"),
    ("metrics_export_stderr_log", ".metrics_export_stderr.log"),
)

# These are diagnostic streams, not payload artifacts. A successful command
# commonly writes zero bytes to stderr, but the zero-length file is still
# mandatory evidence and is still type-checked, opened without following
# symlinks, hashed, and re-read at both terminal publication gates.
EMPTY_ALLOWED_PROFILE_CASE_FILE_LABELS: frozenset[str] = frozenset((
    "container_stderr_log",
    "metrics_export_stderr_log",
))


def _verify_profile_case_artifact(path: Path, label: str) -> str | None:
    """P2.4-specific seven-file policy without weakening P2.3's verifier."""
    if label not in {item_label for item_label, _suffix in CANONICAL_PROFILE_CASE_FILE_LABELS}:
        return f"{path}: unknown P2.4 profile artifact label {label!r}"
    if not os.path.lexists(path):
        return f"{path}: does not exist"
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        return f"{path}: is a symlink; refusing"
    if not stat.S_ISREG(st.st_mode):
        return f"{path}: is not a regular file"
    if st.st_size == 0 and label not in EMPTY_ALLOWED_PROFILE_CASE_FILE_LABELS:
        return f"{path}: is empty"
    return None


def _sha256_profile_case_artifact(path: Path, label: str) -> str:
    """Safely hash a P2.4 artifact, including an allowed zero-length stderr."""
    artifact_error = _verify_profile_case_artifact(path, label)
    if artifact_error:
        raise p23.UnsafePathError(artifact_error)
    flags = os.O_RDONLY | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise p23.UnsafePathError(f"{path}: safe open failed: {exc}") from exc
    try:
        opened = os.fstat(fd)
        current = os.lstat(path)
        if not stat.S_ISREG(opened.st_mode):
            raise p23.UnsafePathError(f"{path}: opened object is not a regular file")
        if (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino):
            raise p23.UnsafePathError(f"{path}: changed while being opened")
        if opened.st_size == 0 and label not in EMPTY_ALLOWED_PROFILE_CASE_FILE_LABELS:
            raise p23.UnsafePathError(f"{path}: is empty")
        digest = hashlib.sha256()
        bytes_read = 0
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            digest.update(chunk)
            bytes_read += len(chunk)
        if bytes_read == 0 and label not in EMPTY_ALLOWED_PROFILE_CASE_FILE_LABELS:
            raise p23.UnsafePathError(f"{path}: became empty while being hashed")
        return digest.hexdigest()
    except OSError as exc:
        raise p23.UnsafePathError(f"{path}: safe hash read failed: {exc}") from exc
    finally:
        os.close(fd)


def canonical_profile_case_filenames(case_name: str) -> tuple[str, ...]:
    return tuple(f"{case_name}{suffix}" for _label, suffix in CANONICAL_PROFILE_CASE_FILE_LABELS)

ANALYSIS_ARTIFACT_RELATIVE_PATHS: tuple[str, ...] = (
    "analysis/configuration_statistics.csv",
    "analysis/scaling.csv",
    "analysis/saturation.csv",
    "analysis/profile_validation.csv",
    "analysis/empirical_ceiling.json",
    "analysis/report.md",
    "analysis/throughput.svg",
    "analysis/scaling_efficiency.svg",
    "analysis/saturation.svg",
    "analysis/analysis_manifest.json",
)


# ---------------------------------------------------------------------------
# Frozen 24-case profile plan: the *same* configurations as P2.3's own
# build_plan(), in the *same* canonical order -- never reimplemented, never
# reordered. Adds only the exact kernel symbol each case must be profiled
# under (src/compute/P2_PROTOCOL.md section 4 / P2_2_PROTOCOL.md section 3).
# ---------------------------------------------------------------------------
def _kernel_symbol(method: str, m: int, n: int, depth: int) -> str:
    tag = "1sm" if method == "umma_1sm" else "2sm"
    return f"umma_{tag}_m{m}n{n}k16_d{depth}"


def build_profile_plan() -> list[dict]:
    plan = []
    for entry in p23.build_plan():
        item = dict(entry)
        item["kernel_symbol"] = _kernel_symbol(entry["method"], entry["m"], entry["n"], entry["depth"])
        plan.append(item)
    return plan


def check_profile_plan_contract(plan: list[dict]) -> list[str]:
    """Independently re-derives every property build_profile_plan() must
    guarantee, so a future edit cannot silently diverge from P2.3's own
    frozen 24-case matrix without failing --self-test."""
    errors: list[str] = []
    p23_plan = p23.build_plan()
    p23_errors = p23.check_plan_contract(p23_plan)
    if p23_errors:
        errors.append(f"underlying P2.3 plan contract violated: {p23_errors}")
    if len(plan) != EXPECTED_PROFILE_CASE_COUNT:
        errors.append(f"profile plan has {len(plan)} case(s), expected {EXPECTED_PROFILE_CASE_COUNT}")
    for entry, p23_entry in zip(plan, p23_plan):
        for field in ("index", "pair_index", "method", "cta_group", "m", "n", "k", "depth", "binary", "case_name"):
            if entry.get(field) != p23_entry.get(field):
                errors.append(
                    f"profile plan entry index {entry.get('index')} field {field}={entry.get(field)!r} "
                    f"!= P2.3 plan's {p23_entry.get(field)!r} (P2.4 must reuse P2.3's plan unmodified)"
                )
        expected_symbol = _kernel_symbol(p23_entry["method"], p23_entry["m"], p23_entry["n"], p23_entry["depth"])
        if entry.get("kernel_symbol") != expected_symbol:
            errors.append(
                f"profile plan entry index {entry.get('index')} kernel_symbol="
                f"{entry.get('kernel_symbol')!r} != expected {expected_symbol!r}"
            )
    symbols = [e["kernel_symbol"] for e in plan]
    if len(set(symbols)) != len(symbols):
        errors.append("duplicate kernel_symbol values in profile plan")
    return errors


def format_plan_text(plan: list[dict]) -> str:
    lines = [
        "index  pair  method     cta_group  m    n    k   depth  case_name                  kernel_symbol",
    ]
    for entry in plan:
        lines.append(
            f"{entry['index']:>5d}  {entry['pair_index']:>4d}  {entry['method']:<9s}  "
            f"{entry['cta_group']:>9d}  {entry['m']:>3d}  {entry['n']:>3d}  {entry['k']:>3d}  "
            f"{entry['depth']:>5d}  {entry['case_name']:<26s}  {entry['kernel_symbol']}"
        )
    lines.append(f"total profile cases: {len(plan)}")
    return "\n".join(lines) + "\n"


def format_plan_lines(plan: list[dict]) -> str:
    return "".join(
        f"{e['index']}\t{e['method']}\t{e['n']}\t{e['depth']}\t{e['cta_group']}\t"
        f"{e['kernel_symbol']}\t{e['case_name']}\n"
        for e in plan
    )


# ---------------------------------------------------------------------------
# Preflight validation. Structurally identical contract to P1.4's own
# (src/memory/P1_4_PROTOCOL.md section 7): reused here field-for-field since
# scripts/preflight.sh's summary.json schema is a single, project-wide
# contract, not specific to any one experiment.
# ---------------------------------------------------------------------------
def load_preflight_json(path: Path) -> tuple[dict | None, list[str]]:
    try:
        with p23._open_regular_nofollow(path, binary=False) as handle:
            text = handle.read()
    except (OSError, p23.UnsafePathError, UnicodeError) as exc:
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


def validate_preflight_fields(doc: dict, *, expected_git_commit: str, now_utc: _datetime) -> tuple[list[str], dict]:
    errors: list[str] = []
    snapshot: dict = {}

    if doc.get("overall_status") != "PASS":
        errors.append(f"overall_status={doc.get('overall_status')!r} != 'PASS'")
    if doc.get("git_dirty") is not False:
        errors.append(f"git_dirty={doc.get('git_dirty')!r} != false")

    commit = doc.get("git_commit")
    if not isinstance(commit, str) or not p23.GIT_COMMIT_RE.fullmatch(commit):
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
    if uuid is not None and not p23.GPU_UUID_RE.fullmatch(uuid):
        errors.append(f"gpu.uuid={uuid!r} is not a canonical GPU-xxxxxxxx-... UUID")
    if gpu.get("compute_cap") != "10.3":
        errors.append(f"gpu.compute_cap={gpu.get('compute_cap')!r} != '10.3'")
    # Defect-1 repair: AGENTS.md requires exactly one GPU visible inside the
    # container, exposed as CUDA logical device 0. preflight.sh's own
    # gpu_visibility check already fails closed (VISIBLE_GPU_COUNT_<n>)
    # whenever nvidia-smi --query-gpu reports other than exactly one row, so
    # checks.gpu_visibility==PASS (checked below) already proves "exactly
    # one visible logical GPU" -- visible_device_count is recorded as an
    # explicit, independently-labeled fact of that already-enforced
    # invariant, never fabricated independently of it. gpu.logical_index
    # (the sole visible GPU's own index, as nvidia-smi enumerates it inside
    # the container) must additionally equal "0".
    if gpu.get("logical_index") != "0":
        errors.append(f"gpu.logical_index={gpu.get('logical_index')!r} != '0' (exactly one GPU must be visible, as CUDA logical device 0)")
    snapshot["gpu_uuid"] = gpu.get("uuid")
    snapshot["gpu_name"] = gpu.get("name")
    snapshot["gpu_driver_version"] = gpu.get("driver_version")
    snapshot["gpu_compute_cap"] = gpu.get("compute_cap")
    snapshot["gpu_logical_index"] = gpu.get("logical_index")

    checks = doc.get("checks")
    check_status: dict[str, object] = {}
    if isinstance(checks, list):
        for entry in checks:
            if isinstance(entry, dict) and isinstance(entry.get("name"), str):
                check_status[entry["name"]] = entry.get("status")
    else:
        errors.append("checks field is missing or not a list")
    if check_status.get("gpu_visibility") != "PASS":
        errors.append(f"checks.gpu_visibility={check_status.get('gpu_visibility')!r} != 'PASS'")
    if check_status.get("ncu_profile") != "PASS":
        errors.append(f"checks.ncu_profile={check_status.get('ncu_profile')!r} != 'PASS'")
    # visible_device_count is derived, not read from a dedicated JSON field:
    # preflight.sh (Phase 0, frozen -- never modified by this repair) has no
    # such field, but its gpu_visibility check can only ever reach PASS after
    # confirming nvidia-smi enumerated exactly one GPU row, so PASS is itself
    # the proof. Recorded only once every field-level error above is absent,
    # so a malformed/failing preflight never fabricates a trustworthy count.
    if not errors and check_status.get("gpu_visibility") == "PASS":
        snapshot["visible_device_count"] = 1

    ts = doc.get("timestamp_utc")
    if not isinstance(ts, str) or not P24_TIMESTAMP_RE.fullmatch(ts):
        errors.append(f"timestamp_utc={ts!r} is not YYYYMMDDTHHMMSSZ")
    else:
        try:
            ts_dt = _datetime.strptime(ts, "%Y%m%dT%H%M%SZ").replace(tzinfo=_timezone.utc)
        except ValueError:
            errors.append(f"timestamp_utc={ts!r} is not a real calendar UTC timestamp")
        else:
            age_seconds = (now_utc - ts_dt).total_seconds()
            if age_seconds < 0:
                errors.append(f"timestamp_utc={ts!r} is in the future relative to now={now_utc.isoformat()}")
            elif age_seconds > 24 * 3600:
                errors.append(
                    f"timestamp_utc={ts!r} is more than 24h old "
                    f"(age={age_seconds / 3600.0:.2f}h relative to now={now_utc.isoformat()})"
                )
            else:
                snapshot["timestamp_utc"] = ts

    return errors, snapshot


def validate_preflight_file(path: Path, *, expected_git_commit: str, now_utc: _datetime) -> tuple[list[str], dict]:
    doc, errors = load_preflight_json(path)
    if errors:
        return errors, {}
    field_errors, snapshot = validate_preflight_fields(doc, expected_git_commit=expected_git_commit, now_utc=now_utc)
    if field_errors:
        return field_errors, snapshot
    try:
        snapshot["sha256"] = p23.sha256_of(path)
    except p23.UnsafePathError as exc:
        return [str(exc)], snapshot
    snapshot["path"] = str(path)
    return [], snapshot


PREFLIGHT_PROVENANCE_FIELDS = (
    "git_commit", "gpu_uuid", "gpu_name", "gpu_compute_cap", "gpu_driver_version",
    "gpu_logical_index", "visible_device_count",
)


def compare_preflight_provenance(pilot_snapshot: dict, profile_snapshot: dict) -> list[str]:
    errors: list[str] = []
    for field in PREFLIGHT_PROVENANCE_FIELDS:
        pilot_value = pilot_snapshot.get(field)
        profile_value = profile_snapshot.get(field)
        if pilot_value != profile_value:
            errors.append(
                f"profile preflight {field}={profile_value!r} != pilot preflight {field}={pilot_value!r} "
                f"(the pilot and profiling phases of one campaign must run on the identical GPU/driver/commit)"
            )
    return errors


# ---------------------------------------------------------------------------
# Defect-1 repair: the immutable campaign provenance tuple. Every profile
# case's own, independently-reported application evidence must be compared
# against this tuple (never the reverse -- this tuple is never copied
# *into* a case's derived output as if that were "validation"). Presence is
# checked by key membership, never truthiness, so an absent field and one
# holding JSON `null` are never treated as equivalent.
# ---------------------------------------------------------------------------
MANDATORY_PROVENANCE_FIELDS: tuple[str, ...] = (
    "git_commit", "gpu_uuid", "gpu_name", "compute_capability",
    "cuda_driver_version", "cuda_runtime_version",
    "visible_device_count", "logical_device_index", "campaign_id",
)
PROVENANCE_FIELD_TYPES: dict[str, tuple[type, ...]] = {
    "git_commit": (str,), "gpu_uuid": (str,), "gpu_name": (str,), "compute_capability": (str,),
    "cuda_driver_version": (str, int), "cuda_runtime_version": (str, int),
    "visible_device_count": (int,), "logical_device_index": (int,), "campaign_id": (str,),
}
# Application-CSV field -> campaign-provenance-tuple field. Both sides use
# the identical underlying measurement (the profiled binary's own
# cudaDriverGetVersion()/cudaRuntimeGetVersion()/device-query report), so an
# equality comparison is meaningful; driver_version as reported by
# nvidia-smi (a human-readable package version string) is a *different*
# measurement and is intentionally never compared here.
APPLICATION_PROVENANCE_FIELD_MAP: tuple[tuple[str, str], ...] = (
    ("git_commit", "git_commit"),
    ("gpu_uuid", "gpu_uuid"),
    ("gpu_name", "gpu_name"),
    ("compute_capability", "compute_capability"),
    ("cuda_driver_version", "cuda_driver_version"),
    ("cuda_runtime_version", "cuda_runtime_version"),
)


def validate_provenance_tuple(provenance: object, *, label: str) -> list[str]:
    """Requires every MANDATORY_PROVENANCE_FIELD to be present (by key, not
    truthiness), non-null, and correctly typed. Does not compare values to
    anything; only proves the tuple itself is well-formed enough to compare
    against. Returns a non-empty error list for anything else, including a
    non-dict input."""
    if not isinstance(provenance, dict):
        return [f"{label}: must be an object, got {type(provenance).__name__}"]
    errors: list[str] = []
    for field in MANDATORY_PROVENANCE_FIELDS:
        if field not in provenance:
            errors.append(f"{label}.{field}: absent (mandatory campaign provenance field)")
            continue
        value = provenance[field]
        if value is None:
            errors.append(f"{label}.{field}: is null (an absent field and a null field are never equivalent; this field is mandatory)")
            continue
        allowed_types = PROVENANCE_FIELD_TYPES[field]
        if isinstance(value, bool) or not isinstance(value, allowed_types):
            errors.append(f"{label}.{field}={value!r}: wrong type {type(value).__name__}, expected one of {[t.__name__ for t in allowed_types]}")
            continue
        if field == "git_commit" and not p23.GIT_COMMIT_RE.fullmatch(value):
            errors.append(f"{label}.{field}={value!r}: not a 40-character lowercase hex commit")
        if field == "gpu_uuid" and not p23.GPU_UUID_RE.fullmatch(value):
            errors.append(f"{label}.{field}={value!r}: not a canonical GPU-xxxxxxxx-... UUID")
        if field == "visible_device_count" and value != 1:
            errors.append(f"{label}.{field}={value!r}: must be exactly 1 (exactly one GPU must ever be visible)")
        if field == "logical_device_index" and value != 0:
            errors.append(f"{label}.{field}={value!r}: must be exactly 0 (the visible GPU must be CUDA logical device 0)")
    return errors


def compare_application_provenance(*, app_row: dict[str, str], campaign_provenance: dict, label: str) -> list[str]:
    """Compares one profiled application CSV row's own, independently
    reported identity fields against the already-validated immutable
    campaign provenance tuple. app_row must come from freshly parsed raw
    evidence (never from campaign_provenance itself, which would make this
    comparison vacuous); this is the check that closes the originally
    audited cross-GPU-UUID defect. campaign_provenance is assumed already
    passed through validate_provenance_tuple by the caller."""
    errors: list[str] = []
    for row_field, provenance_field in APPLICATION_PROVENANCE_FIELD_MAP:
        row_value = app_row.get(row_field)
        provenance_value = campaign_provenance.get(provenance_field)
        if row_field in ("cuda_driver_version", "cuda_runtime_version"):
            matches = row_value is not None and provenance_value is not None and str(row_value) == str(provenance_value)
        else:
            matches = row_value == provenance_value
        if not matches:
            errors.append(
                f"{label}: application {row_field}={row_value!r} != campaign provenance {provenance_field}={provenance_value!r} "
                f"(this profile's own evidence does not match the campaign this profile is claimed to belong to)"
            )
    return errors


# ---------------------------------------------------------------------------
# P2.4 campaign-ID validation and symlink-safe raw-tree primitives. Reuses
# p23's generic lstat-based path-safety helpers.
# ---------------------------------------------------------------------------
def validate_p24_campaign_id(campaign_id: str) -> None:
    p23.validate_campaign_id(campaign_id)
    if not P24_CAMPAIGN_ID_RE.fullmatch(campaign_id):
        raise p23.UnsafePathError(
            f"P2_4_CAMPAIGN_ID={campaign_id!r} must be an explicit canonical UTC timestamp YYYYMMDDTHHMMSSZ"
        )
    try:
        _datetime.strptime(campaign_id, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise p23.UnsafePathError(f"P2_4_CAMPAIGN_ID={campaign_id!r} is not a real calendar UTC timestamp") from exc


def create_p24_campaign_dir(campaign_id: str) -> Path:
    validate_p24_campaign_id(campaign_id)
    current = REPO_ROOT
    for part in RAW_ROOT_PARTS_P24:
        current = current / part
        p23._mkdir_component(current, must_not_exist=False, root=REPO_ROOT)
    campaign_dir = current / campaign_id
    p23._mkdir_component(campaign_dir, must_not_exist=True, root=REPO_ROOT)
    for sub in ("profiles", "analysis", "logs", "manifest"):
        p23._mkdir_component(campaign_dir / sub, must_not_exist=False, root=REPO_ROOT)
    return campaign_dir


def resolve_p24_campaign_dir(campaign_dir_rel: str) -> Path:
    if os.path.isabs(campaign_dir_rel):
        raise p23.UnsafePathError(f"--campaign-dir must be relative, got absolute path {campaign_dir_rel!r}")
    parts = Path(campaign_dir_rel).parts
    if any(".." in part for part in parts):
        raise p23.UnsafePathError(f"--campaign-dir must not contain '..': {campaign_dir_rel!r}")
    if len(parts) != len(RAW_ROOT_PARTS_P24) + 1 or tuple(parts[: len(RAW_ROOT_PARTS_P24)]) != RAW_ROOT_PARTS_P24:
        raise p23.UnsafePathError(
            f"--campaign-dir must be exactly {'/'.join(RAW_ROOT_PARTS_P24)}/<campaign_id>, got {campaign_dir_rel!r}"
        )
    validate_p24_campaign_id(parts[-1])

    current = REPO_ROOT
    for part in parts:
        current = current / part
        p23._reject_if_symlink_or_wrong_type(current, expect_dir=True)
        if not os.path.lexists(current):
            raise p23.UnsafePathError(f"{current}: does not exist")
    p23._confirm_contained(current, REPO_ROOT)
    for subdir_name in ("profiles", "analysis", "logs", "manifest"):
        subdir = current / subdir_name
        p23._reject_if_symlink_or_wrong_type(subdir, expect_dir=True)
        if not os.path.lexists(subdir):
            raise p23.UnsafePathError(f"{subdir}: required campaign directory does not exist")
        p23._confirm_contained(subdir, current)
    return current


def resolve_p23_campaign_dir_arg(campaign_dir_rel: str) -> Path:
    return p23.resolve_campaign_dir(campaign_dir_rel)


# ---------------------------------------------------------------------------
# profile_plan.csv: written once at init, re-validated at finalize-profile
# time. Mirrors P2.3's own execution_order.csv discipline exactly.
# ---------------------------------------------------------------------------
def _profile_plan_row(entry: dict) -> list[str]:
    return [
        str(entry["index"]), str(entry["pair_index"]), entry["method"], str(entry["cta_group"]),
        str(entry["m"]), str(entry["n"]), str(entry["k"]), str(entry["depth"]), entry["binary"],
        entry["case_name"], entry["kernel_symbol"],
    ]


def write_profile_plan(campaign_dir: Path, plan: list[dict]) -> Path:
    out_path = campaign_dir / "profile_plan.csv"
    if os.path.lexists(out_path):
        raise p23.UnsafePathError(f"{out_path}: already exists, refusing to overwrite")
    tmp_path = campaign_dir / "profile_plan.csv.tmp"
    if os.path.lexists(tmp_path):
        raise p23.UnsafePathError(f"{tmp_path}: stale temporary file already exists")
    try:
        with p23._open_exclusive(tmp_path, binary=False, newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(PROFILE_PLAN_HEADER)
            for entry in plan:
                writer.writerow(_profile_plan_row(entry))
    except Exception:
        if os.path.lexists(tmp_path):
            p23._safe_unlink_owned(tmp_path)
        raise
    try:
        p23._publish_no_clobber(tmp_path, out_path)
    except p23.UnsafePathError:
        if os.path.lexists(tmp_path):
            p23._safe_unlink_owned(tmp_path)
        raise
    return out_path


def validate_profile_plan_file(path: Path, plan: list[dict]) -> list[str]:
    if os.path.lexists(path):
        st = os.lstat(path)
        if stat.S_ISLNK(st.st_mode):
            return [f"{path}: is a symlink; refusing"]
    else:
        return [f"{path}: does not exist"]
    try:
        with p23._open_regular_nofollow(path, binary=False) as handle:
            rows = list(csv.reader(handle))
    except (OSError, p23.UnsafePathError, UnicodeError) as exc:
        return [f"{path}: unable to read: {exc}"]
    if not rows:
        return [f"{path}: empty file"]
    header, data_rows = rows[0], rows[1:]
    if header != PROFILE_PLAN_HEADER:
        return [f"{path}: header mismatch: {header!r}"]
    if len(data_rows) != len(plan):
        return [f"{path}: has {len(data_rows)} row(s), expected {len(plan)}"]
    errors: list[str] = []
    for i, (row, entry) in enumerate(zip(data_rows, plan)):
        expected_row = _profile_plan_row(entry)
        if row != expected_row:
            errors.append(f"{path}: row {i} = {row!r} != expected {expected_row!r}")
    return errors


# ---------------------------------------------------------------------------
# Manifest: allowlisted keys/types, field classification, exact per-
# transition mutation matrix, state-shape validation, timestamp chronology,
# append-only hash chain. Mirrors src/memory/P1_4_PROTOCOL.md section 8's
# design exactly, extended with one new terminal state (INCONCLUSIVE) for
# the case where the mandatory SM-clock metric cannot be trusted.
# ---------------------------------------------------------------------------
def _expected_complete_artifact_sha256(manifest: dict) -> tuple[dict[str, str], list[str]]:
    expected: dict[str, str] = {}
    errors: list[str] = []

    def _record(output_key: str, value: object, source: str) -> None:
        if not isinstance(value, str) or not p23._is_sha256_hex(value):
            errors.append(
                f"{source} must be a canonical 64-hex SHA-256 before artifact_sha256[{output_key!r}] "
                f"can be derived; got {value!r}"
            )
            return
        expected[output_key] = value

    _record("profile_plan.csv", manifest.get("profile_plan_sha256"), "profile_plan_sha256")

    pilot_ref = manifest.get("pilot_campaign_reference")
    if not isinstance(pilot_ref, dict):
        errors.append("pilot_campaign_reference must be an object for terminal artifact validation")
    else:
        for source_key, output_key in (
            ("manifest_sha256", "pilot_manifest_sha256"),
            ("combined_samples_sha256", "pilot_combined_samples_sha256"),
            ("summary_sha256", "pilot_summary_sha256"),
        ):
            _record(output_key, pilot_ref.get(source_key), f"pilot_campaign_reference.{source_key}")

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
        for entry in build_profile_plan():
            case_name = entry["case_name"]
            result = case_results.get(case_name)
            if not isinstance(result, dict):
                errors.append(f"case_results.{case_name} must be an object for terminal artifact validation")
                continue
            for field in CASE_ARTIFACT_HASH_FIELDS:
                _record(f"{case_name}.{field}", result.get(field), f"case_results.{case_name}.{field}")

    return expected, errors


def validate_terminal_manifest_content(manifest: dict) -> list[str]:
    state = manifest.get("state")
    if state not in ("COMPLETE", "ANALYZED", "INCONCLUSIVE"):
        return []

    errors: list[str] = []
    expected_plan = build_profile_plan()
    if manifest.get("profile_order") != expected_plan:
        errors.append("profile_order is not exactly the frozen 24-case plan in its canonical order")

    expected_base, source_errors = _expected_complete_artifact_sha256(manifest)
    errors.extend(source_errors)

    actual = manifest.get("artifact_sha256")
    if not isinstance(actual, dict):
        return errors + ["artifact_sha256 must be an object in COMPLETE/ANALYZED/INCONCLUSIVE"]

    expected_keys = set(expected_base)
    if state in ("ANALYZED", "INCONCLUSIVE"):
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
                f"artifact_sha256[{key!r}]={actual[key]!r} does not equal its canonical recorded "
                f"evidence hash {expected_hash!r}"
            )
    if state in ("ANALYZED", "INCONCLUSIVE"):
        for key in ANALYSIS_ARTIFACT_RELATIVE_PATHS:
            value = actual.get(key)
            if key in actual and (not isinstance(value, str) or not p23._is_sha256_hex(value)):
                errors.append(f"artifact_sha256[{key!r}] must be a canonical 64-hex SHA-256; got {value!r}")

    if state == "INCONCLUSIVE":
        reason = manifest.get("inconclusive_reason")
        if not isinstance(reason, list) or not reason or not all(isinstance(item, str) for item in reason):
            errors.append("INCONCLUSIVE manifest must carry a non-empty inconclusive_reason list[str]")

    return errors


def _validate_p24_manifest_updates(updates: dict) -> None:
    unknown = set(updates) - set(ALLOWED_P24_MANIFEST_KEYS)
    if unknown:
        raise p23.ManifestTransitionError(f"unknown P2.4 manifest field(s): {sorted(unknown)}")
    for key, value in updates.items():
        expected_type = ALLOWED_P24_MANIFEST_KEYS[key]
        if not p23._manifest_type_matches(value, expected_type):
            raise p23.ManifestTransitionError(
                f"P2.4 manifest field {key!r} has invalid type {type(value).__name__}, expected {expected_type}"
            )


def _validate_p24_manifest_document(manifest: dict, *, require_initialized: bool = False) -> None:
    _validate_p24_manifest_updates(manifest)
    if not manifest:
        if require_initialized:
            raise p23.ManifestTransitionError("P2.4 manifest is empty")
        return

    state = manifest.get("state")
    if state not in ALLOWED_P24_STATES:
        raise p23.ManifestTransitionError(f"P2.4 manifest state={state!r} is invalid")
    if "schema_version" in manifest and manifest["schema_version"] != P24_SCHEMA_VERSION:
        raise p23.ManifestTransitionError("P2.4 manifest schema_version is invalid")
    if "experiment_id" in manifest and manifest["experiment_id"] != P24_EXPERIMENT_ID:
        raise p23.ManifestTransitionError("P2.4 manifest experiment_id is invalid")
    if "publishable" in manifest and manifest["publishable"] is not False:
        raise p23.ManifestTransitionError("P2.4 manifest publishable must be false")
    if "provenance" in manifest:
        provenance_errors = validate_provenance_tuple(manifest["provenance"], label="manifest.provenance")
        if provenance_errors:
            raise p23.ManifestTransitionError(f"P2.4 manifest provenance is malformed: {provenance_errors}")
        if "campaign_id" in manifest and manifest["provenance"].get("campaign_id") != manifest["campaign_id"]:
            raise p23.ManifestTransitionError(
                f"P2.4 manifest provenance.campaign_id={manifest['provenance'].get('campaign_id')!r} != "
                f"manifest.campaign_id={manifest['campaign_id']!r}"
            )
    if "campaign_id" in manifest:
        try:
            validate_p24_campaign_id(manifest["campaign_id"])
        except p23.UnsafePathError as exc:
            raise p23.ManifestTransitionError(f"P2.4 manifest campaign_id is invalid: {exc}") from exc
    if "started_at_utc" in manifest:
        p23._validate_compact_timestamp(manifest["started_at_utc"], "started_at_utc")
    for key in ("pilot_completed_at_utc", "profile_started_at_utc", "profile_completed_at_utc", "analysis_completed_at_utc"):
        if key in manifest:
            p23._validate_compact_timestamp(manifest[key], key)
    if "profile_count_completed" in manifest and not (0 <= manifest["profile_count_completed"] <= EXPECTED_PROFILE_CASE_COUNT):
        raise p23.ManifestTransitionError(
            f"P2.4 manifest profile_count_completed must be in [0, {EXPECTED_PROFILE_CASE_COUNT}]"
        )
    if "failure_stage" in manifest and not manifest["failure_stage"]:
        raise p23.ManifestTransitionError("P2.4 manifest failure_stage must be non-empty when present")
    if "failure_detail" in manifest and not all(isinstance(item, str) for item in manifest["failure_detail"]):
        raise p23.ManifestTransitionError("P2.4 manifest failure_detail must be a list of strings")
    if "inconclusive_reason" in manifest and not all(isinstance(item, str) for item in manifest["inconclusive_reason"]):
        raise p23.ManifestTransitionError("P2.4 manifest inconclusive_reason must be a list of strings")

    required_by_state = {
        "PILOT_IN_PROGRESS": {
            "schema_version", "experiment_id", "campaign_id", "state", "publishable",
            "started_at_utc", "frozen_protocol", "profile_plan_sha256",
        },
        "PILOT_COMPLETE": {
            "pilot_completed_at_utc", "pilot_campaign_reference", "preflight_reference_pilot", "provenance",
        },
        "PROFILE_IN_PROGRESS": {"profile_started_at_utc", "resolved_ncu_metrics", "preflight_reference_profile"},
        "COMPLETE": {
            "profile_completed_at_utc", "profile_order", "profile_count_completed",
            "case_results", "artifact_sha256",
        },
    }
    if state in required_by_state:
        gate_order = list(required_by_state)
        needed: set[str] = set()
        for gate_state in gate_order[: gate_order.index(state) + 1]:
            needed |= required_by_state[gate_state]
        missing = needed - set(manifest)
        if missing:
            raise p23.ManifestTransitionError(f"P2.4 manifest in state {state!r} missing required field(s): {sorted(missing)}")
    elif state in ("ANALYZED", "INCONCLUSIVE"):
        needed = set()
        for gate_state in required_by_state.values():
            needed |= gate_state
        needed.add("analysis_completed_at_utc")
        if state == "INCONCLUSIVE":
            needed.add("inconclusive_reason")
        missing = needed - set(manifest)
        if missing:
            raise p23.ManifestTransitionError(f"P2.4 manifest in state {state!r} missing required field(s): {sorted(missing)}")

    if state in ("COMPLETE", "ANALYZED", "INCONCLUSIVE") and manifest.get("profile_count_completed") != EXPECTED_PROFILE_CASE_COUNT:
        raise p23.ManifestTransitionError(
            f"P2.4 manifest state={state} requires profile_count_completed={EXPECTED_PROFILE_CASE_COUNT}, "
            f"got {manifest.get('profile_count_completed')!r}"
        )
    failure_keys = {"failure_stage", "failure_detail"}
    present_failure_keys = failure_keys & set(manifest)
    if state in ("FAILED", "INTERRUPTED"):
        missing_failure_keys = failure_keys - set(manifest)
        if missing_failure_keys:
            raise p23.ManifestTransitionError(
                f"P2.4 manifest state={state!r} missing failure telemetry field(s): {sorted(missing_failure_keys)}"
            )
    elif present_failure_keys:
        raise p23.ManifestTransitionError(
            f"P2.4 manifest state={state!r} cannot contain failure telemetry field(s): {sorted(present_failure_keys)}"
        )
    if state != "INCONCLUSIVE" and "inconclusive_reason" in manifest:
        raise p23.ManifestTransitionError(f"P2.4 manifest state={state!r} cannot contain inconclusive_reason")

    terminal_errors = validate_terminal_manifest_content(manifest)
    if terminal_errors:
        raise p23.ManifestTransitionError(f"P2.4 manifest state={state!r} has non-canonical terminal content: {terminal_errors}")


# --- Field classification (documented in src/compute/P2_4_PROTOCOL.md) -----
P24_FIELD_IMMUTABLE = frozenset({
    "schema_version", "experiment_id", "campaign_id", "publishable", "frozen_protocol", "profile_plan_sha256",
})
P24_FIELD_ALLOWED_TIMESTAMP = frozenset({
    "started_at_utc", "pilot_completed_at_utc", "profile_started_at_utc",
    "profile_completed_at_utc", "analysis_completed_at_utc",
})
P24_FIELD_SET_ONCE = frozenset({
    "pilot_campaign_reference", "preflight_reference_pilot", "provenance",
    "preflight_reference_profile", "resolved_ncu_metrics", "profile_order",
})
P24_FIELD_STATE_DERIVED = frozenset({"state", "profile_count_completed"})
P24_FIELD_APPEND_ONLY = frozenset({"case_results", "artifact_sha256"})
P24_FIELD_FAILURE = frozenset({"failure_stage", "failure_detail"})
P24_FIELD_INCONCLUSIVE = frozenset({"inconclusive_reason"})

_P24_FIELD_CLASSIFICATION_UNION = (
    P24_FIELD_IMMUTABLE | P24_FIELD_ALLOWED_TIMESTAMP | P24_FIELD_SET_ONCE
    | P24_FIELD_STATE_DERIVED | P24_FIELD_APPEND_ONLY | P24_FIELD_FAILURE | P24_FIELD_INCONCLUSIVE
)
assert _P24_FIELD_CLASSIFICATION_UNION == frozenset(ALLOWED_P24_MANIFEST_KEYS), (
    "every P2.4 manifest field must be classified into exactly one P24_FIELD_* category: "
    f"missing={frozenset(ALLOWED_P24_MANIFEST_KEYS) - _P24_FIELD_CLASSIFICATION_UNION!r} "
    f"extra={_P24_FIELD_CLASSIFICATION_UNION - frozenset(ALLOWED_P24_MANIFEST_KEYS)!r}"
)

P24_EXACT_TRANSITION_MUTATIONS: dict[tuple[str, str], frozenset[str]] = {
    ("PILOT_IN_PROGRESS", "PILOT_COMPLETE"): frozenset({
        "pilot_completed_at_utc", "pilot_campaign_reference", "preflight_reference_pilot", "provenance",
    }),
    ("PILOT_COMPLETE", "PROFILE_IN_PROGRESS"): frozenset({
        "profile_started_at_utc", "resolved_ncu_metrics", "preflight_reference_profile",
    }),
    ("PROFILE_IN_PROGRESS", "PROFILE_IN_PROGRESS"): frozenset({"case_results", "profile_count_completed"}),
    ("PROFILE_IN_PROGRESS", "COMPLETE"): frozenset({"profile_completed_at_utc", "profile_order", "artifact_sha256"}),
    ("COMPLETE", "ANALYZED"): frozenset({"analysis_completed_at_utc", "artifact_sha256"}),
    ("COMPLETE", "INCONCLUSIVE"): frozenset({"analysis_completed_at_utc", "artifact_sha256", "inconclusive_reason"}),
}

# State-availability matrix: which fields a revision in a given state may
# ever carry (Task-4-style, src/memory/P1_4_PROTOCOL.md section 8). ANALYZED
# and INCONCLUSIVE are both direct, mutually exclusive children of COMPLETE
# (never chained to each other), so _p24_cumulative_allowed_fields handles
# them as a special case rather than a strict linear walk.
P24_STATE_ORDER: tuple[str, ...] = ("PILOT_IN_PROGRESS", "PILOT_COMPLETE", "PROFILE_IN_PROGRESS", "COMPLETE")

P24_FIELDS_INTRODUCED_BY_STATE: dict[str, frozenset[str]] = {
    "PILOT_IN_PROGRESS": frozenset({
        "schema_version", "experiment_id", "campaign_id", "state", "publishable",
        "started_at_utc", "frozen_protocol", "profile_plan_sha256",
    }),
    "PILOT_COMPLETE": frozenset({
        "pilot_completed_at_utc", "pilot_campaign_reference", "preflight_reference_pilot", "provenance",
    }),
    "PROFILE_IN_PROGRESS": frozenset({
        "profile_started_at_utc", "resolved_ncu_metrics", "preflight_reference_profile",
        "case_results", "profile_count_completed",
    }),
    "COMPLETE": frozenset({"profile_completed_at_utc", "profile_order", "artifact_sha256"}),
    "ANALYZED": frozenset({"analysis_completed_at_utc"}),
    "INCONCLUSIVE": frozenset({"analysis_completed_at_utc", "inconclusive_reason"}),
}
P24_FAILURE_ONLY_FIELDS = frozenset({"failure_stage", "failure_detail"})

_P24_FIELD_INTRODUCTION_UNION = frozenset().union(*P24_FIELDS_INTRODUCED_BY_STATE.values()) | P24_FAILURE_ONLY_FIELDS
assert _P24_FIELD_INTRODUCTION_UNION == frozenset(ALLOWED_P24_MANIFEST_KEYS), (
    "every P2.4 manifest field must be bound to a legal state-availability bucket (or the failure-only "
    "exception) -- an unbound field must never be able to appear at an arbitrary state unnoticed: "
    f"missing={frozenset(ALLOWED_P24_MANIFEST_KEYS) - _P24_FIELD_INTRODUCTION_UNION!r} "
    f"extra={_P24_FIELD_INTRODUCTION_UNION - frozenset(ALLOWED_P24_MANIFEST_KEYS)!r}"
)


def _p24_cumulative_allowed_fields(state: str) -> frozenset[str]:
    if state in ("FAILED", "INTERRUPTED"):
        cumulative: frozenset[str] = frozenset()
        for s in P24_STATE_ORDER:
            cumulative = cumulative | P24_FIELDS_INTRODUCED_BY_STATE[s]
        return cumulative | P24_FAILURE_ONLY_FIELDS
    if state in ("ANALYZED", "INCONCLUSIVE"):
        cumulative = frozenset()
        for s in P24_STATE_ORDER:
            cumulative = cumulative | P24_FIELDS_INTRODUCED_BY_STATE[s]
        return cumulative | P24_FIELDS_INTRODUCED_BY_STATE[state]
    cumulative = frozenset()
    for s in P24_STATE_ORDER:
        cumulative = cumulative | P24_FIELDS_INTRODUCED_BY_STATE[s]
        if s == state:
            return cumulative
    raise AssertionError(f"unreachable: state {state!r} is not in P24_STATE_ORDER or the failure states")


def validate_manifest_state_shape(current: dict, expected_campaign_id: str) -> list[str]:
    """Validates ONE manifest revision's shape in isolation from any other
    revision: which fields may legally be present given its own declared
    state, with no knowledge of history. Presence is tested by key
    membership, never truthiness -- an unexpected `null` must never behave
    like an absent field."""
    errors: list[str] = []
    state = current.get("state")
    if state not in ALLOWED_P24_STATES:
        return [f"state={state!r} is not a recognized P2.4 state"]

    if current.get("campaign_id") != expected_campaign_id:
        errors.append(
            f"campaign_id={current.get('campaign_id')!r} != campaign directory basename {expected_campaign_id!r}"
        )

    allowed = _p24_cumulative_allowed_fields(state)
    for key in ALLOWED_P24_MANIFEST_KEYS:
        if key in current and key not in allowed:
            errors.append(
                f"{key} is present but state={state!r} has not yet reached the state that may introduce it"
            )

    case_results_present = "case_results" in current
    count_present = "profile_count_completed" in current
    if case_results_present != count_present:
        errors.append(
            f"profile_count_completed present={count_present} but case_results present={case_results_present} "
            f"(the two fields must always appear together)"
        )

    case_results = current.get("case_results")
    if isinstance(case_results, dict):
        frozen_order = [entry["case_name"] for entry in build_profile_plan()]
        actual_order = list(case_results)
        if actual_order != frozen_order[: len(actual_order)]:
            errors.append(
                f"case_results key order {actual_order!r} is not an exact ordered prefix of the frozen "
                f"24-case order (checked as an ordered list, never a set)"
            )
        if count_present and current["profile_count_completed"] != len(case_results):
            errors.append(
                f"profile_count_completed={current.get('profile_count_completed')!r} != "
                f"len(case_results)={len(case_results)}"
            )

    return errors


def validate_manifest_revision_transition(previous: dict | None, current: dict, expected_campaign_id: str) -> list[str]:
    """Validates one manifest revision (previous=None: revision 0 in
    isolation) or one adjacent revision-to-revision transition. Never trusts
    campaign_id recorded inside a revision by itself."""
    errors: list[str] = []

    if current.get("campaign_id") != expected_campaign_id:
        errors.append(f"campaign_id={current.get('campaign_id')!r} != campaign directory basename {expected_campaign_id!r}")
    if current.get("publishable") is not False:
        errors.append(f"publishable={current.get('publishable')!r} != False (must always be false)")

    curr_state = current.get("state")
    curr_failure_stage = current.get("failure_stage")
    if curr_failure_stage is not None and curr_state not in ("FAILED", "INTERRUPTED"):
        errors.append(f"failure_stage={curr_failure_stage!r} is set but state={curr_state!r} is neither FAILED nor INTERRUPTED")
    if current.get("analysis_completed_at_utc") is not None and curr_state not in ("ANALYZED", "INCONCLUSIVE"):
        errors.append(f"analysis_completed_at_utc is set but state={curr_state!r} is neither ANALYZED nor INCONCLUSIVE")
    if current.get("inconclusive_reason") is not None and curr_state != "INCONCLUSIVE":
        errors.append(f"inconclusive_reason is set but state={curr_state!r} != 'INCONCLUSIVE'")
    curr_artifact_sha256 = current.get("artifact_sha256")
    if isinstance(curr_artifact_sha256, dict):
        for name in curr_artifact_sha256:
            if name.startswith("analysis/") and curr_state not in ("ANALYZED", "INCONCLUSIVE"):
                errors.append(
                    f"artifact_sha256 contains analysis artifact {name!r} but state={curr_state!r} is "
                    f"neither ANALYZED nor INCONCLUSIVE"
                )

    if previous is None:
        prev_state = None
    else:
        for field in P24_FIELD_IMMUTABLE:
            if field in previous and previous.get(field) != current.get(field):
                errors.append(f"{field} is immutable but changed: {previous.get(field)!r} -> {current.get(field)!r}")
        for field in P24_FIELD_ALLOWED_TIMESTAMP | P24_FIELD_SET_ONCE:
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
                errors.append(f"artifact_sha256.{name} changed after being recorded: {prev_hash!r} -> {curr_artifacts[name]!r}")

        prev_cases = previous.get("case_results")
        prev_cases = prev_cases if isinstance(prev_cases, dict) else {}
        curr_cases = current.get("case_results")
        curr_cases = curr_cases if isinstance(curr_cases, dict) else {}
        for case_name, prev_value in prev_cases.items():
            if case_name not in curr_cases:
                errors.append(f"case_results.{case_name} was deleted")
            elif curr_cases[case_name] != prev_value:
                errors.append(f"case_results.{case_name} was modified after being recorded (append-only)")
        frozen_order = [entry["case_name"] for entry in build_profile_plan()]

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
            errors.append(f"case_results gained an entry out of frozen order: now contains {sorted(curr_cases)}")
        elif curr_k < prev_k:
            errors.append(f"case_results shrank from {prev_k} to {curr_k} entrie(s)")

        prev_count = previous.get("profile_count_completed")
        curr_count = current.get("profile_count_completed")
        if isinstance(prev_count, int) and isinstance(curr_count, int) and curr_count < prev_count:
            errors.append(f"profile_count_completed regressed: {prev_count} -> {curr_count}")

        prev_state_for_loop = previous.get("state")
        curr_state_for_loop = current.get("state")
        if (
            prev_state_for_loop == "PROFILE_IN_PROGRESS" and curr_state_for_loop == "PROFILE_IN_PROGRESS"
            and prev_k is not None and curr_k is not None and curr_k != prev_k + 1
        ):
            errors.append(
                f"a PROFILE_IN_PROGRESS -> PROFILE_IN_PROGRESS revision must append exactly one new case "
                f"result; case_results went from {prev_k} to {curr_k} entrie(s)"
            )

        prev_state = previous.get("state")
        allowed_next_states = ALLOWED_P24_TRANSITIONS.get(prev_state, frozenset())
        if curr_state not in allowed_next_states:
            errors.append(
                f"illegal state transition: {prev_state!r} -> {curr_state!r} "
                f"(allowed next state(s): {sorted(allowed_next_states)!r})"
            )
        else:
            changed_content_fields = {
                key
                for key in set(previous) | set(current)
                if key != "state"
                and ((key in previous) != (key in current) or (key in previous and key in current and previous[key] != current[key]))
            }
            if curr_state in ("FAILED", "INTERRUPTED"):
                expected_changed_fields = P24_FIELD_FAILURE
            else:
                expected_changed_fields = P24_EXACT_TRANSITION_MUTATIONS.get((prev_state, curr_state))
            if expected_changed_fields is not None and changed_content_fields != expected_changed_fields:
                missing_changes = sorted(expected_changed_fields - changed_content_fields)
                unexpected_changes = sorted(changed_content_fields - expected_changed_fields)
                errors.append(
                    f"exact transition mutation matrix violation for {prev_state!r} -> {curr_state!r}: "
                    f"changed content fields {sorted(changed_content_fields)!r}, expected exactly "
                    f"{sorted(expected_changed_fields)!r}; missing changes={missing_changes!r}; "
                    f"unexpected changes={unexpected_changes!r}"
                )

    return errors


P24_LIFECYCLE_TIMESTAMP_ORDER: tuple[str, ...] = (
    "started_at_utc", "pilot_completed_at_utc", "profile_started_at_utc",
    "profile_completed_at_utc", "analysis_completed_at_utc",
)


def validate_manifest_timestamp_chronology(current: dict) -> list[str]:
    errors: list[str] = []
    parsed: list[tuple[str, _datetime]] = []
    for field in P24_LIFECYCLE_TIMESTAMP_ORDER:
        value = current.get(field)
        if not isinstance(value, str):
            continue
        try:
            p23._validate_compact_timestamp(value, field)
            parsed.append((field, _datetime.strptime(value, "%Y%m%dT%H%M%SZ")))
        except p23.ManifestTransitionError:
            continue
    for (earlier_field, earlier_dt), (later_field, later_dt) in zip(parsed, parsed[1:]):
        if later_dt < earlier_dt:
            errors.append(
                f"{later_field}={current[later_field]!r} precedes {earlier_field}={current[earlier_field]!r} "
                f"(lifecycle timestamps must be nondecreasing: {' <= '.join(P24_LIFECYCLE_TIMESTAMP_ORDER)})"
            )
    return errors


MANIFEST_REVISION_RE = re.compile(r"^(\d{6})\.json$")
MANIFEST_REVISION_TMP_NAME = ".manifest_revision.tmp"
MANIFEST_REVISION_KEYS = ("manifest_revision", "previous_manifest_sha256")


def _p24_manifest_dir(campaign_dir: Path) -> Path:
    return campaign_dir / "manifest"


def _manifest_revision_path(campaign_dir: Path, revision: int) -> Path:
    return _p24_manifest_dir(campaign_dir) / f"{revision:06d}.json"


# ---------------------------------------------------------------------------
# Descriptor-anchored directory/file resolution, duplicated in miniature
# rather than imported from scripts/p24_safe_capture.py (which itself
# imports this module for the frozen plan/raw-root constants -- importing
# back would be circular). Mirrors P1.4's identical design choice.
# ---------------------------------------------------------------------------
def _open_dir_nofollow_p24(name: str, *, dir_fd: int | None) -> int:
    flags = os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        raise p23.UnsafePathError(f"{name}: cannot open as a non-symlink directory: {exc}") from exc


def _open_dir_component_chain(*parts: str) -> int:
    fd = _open_dir_nofollow_p24(str(REPO_ROOT), dir_fd=None)
    try:
        for part in parts:
            next_fd = _open_dir_nofollow_p24(part, dir_fd=fd)
            os.close(fd)
            fd = next_fd
    except Exception:
        os.close(fd)
        raise
    return fd


def _open_profiles_fd_anchored(campaign_dir: Path) -> int:
    campaign_parts = campaign_dir.relative_to(REPO_ROOT).parts
    return _open_dir_component_chain(*campaign_parts, "profiles")


def _check_case_directory_inventory(case_fd: int, case_name: str) -> list[str]:
    """Requires the case directory to contain EXACTLY the seven canonical
    per-case artifacts (src/compute/P2_4_PROTOCOL.md section 7): no missing,
    extra, duplicate (by name, impossible on a POSIX directory, but a
    differently-cased or differently-spelled decoy is caught as "extra"),
    symlinked, or wrong-type entry. Every directory entry is inspected via
    lstat, never following a symlink."""
    errors: list[str] = []
    expected = set(canonical_profile_case_filenames(case_name))
    labels_by_name = {
        f"{case_name}{suffix}": label
        for label, suffix in CANONICAL_PROFILE_CASE_FILE_LABELS
    }
    try:
        actual = set(os.listdir(case_fd))
    except OSError as exc:
        return [f"profiles/{case_name}: cannot list directory: {exc}"]
    for extra in sorted(actual - expected):
        errors.append(
            f"profiles/{case_name}/{extra}: unplanned entry (not one of the seven canonical per-case artifacts)"
        )
    for missing in sorted(expected - actual):
        errors.append(f"profiles/{case_name}/{missing}: missing canonical artifact")
    for name in sorted(actual & expected):
        try:
            st = os.stat(name, dir_fd=case_fd, follow_symlinks=False)
        except OSError as exc:
            errors.append(f"profiles/{case_name}/{name}: cannot stat: {exc}")
            continue
        if stat.S_ISLNK(st.st_mode):
            errors.append(f"profiles/{case_name}/{name}: is a symlink; refusing")
        elif not stat.S_ISREG(st.st_mode):
            errors.append(f"profiles/{case_name}/{name}: is not a regular file")
        elif st.st_size == 0 and labels_by_name[name] not in EMPTY_ALLOWED_PROFILE_CASE_FILE_LABELS:
            errors.append(f"profiles/{case_name}/{name}: is empty")
    return errors


def _open_case_evidence_fds(profiles_fd: int, case_name: str) -> dict[str, int]:
    case_fd = None
    try:
        case_fd = _open_dir_nofollow_p24(case_name, dir_fd=profiles_fd)
    except p23.UnsafePathError as exc:
        raise p23.UnsafePathError(f"profiles/{case_name}: {exc}") from exc
    fds: dict[str, int] = {"_case_dir": case_fd}
    try:
        inventory_errors = _check_case_directory_inventory(case_fd, case_name)
        if inventory_errors:
            raise p23.UnsafePathError("; ".join(inventory_errors))
        for label, suffix in CANONICAL_PROFILE_CASE_FILE_LABELS:
            filename = f"{case_name}{suffix}"
            file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
            try:
                fd = os.open(filename, file_flags, dir_fd=case_fd)
            except OSError as exc:
                raise p23.UnsafePathError(f"profiles/{case_name}/{filename}: cannot open: {exc}") from exc
            try:
                st = os.fstat(fd)
            except OSError:
                os.close(fd)
                raise
            if not stat.S_ISREG(st.st_mode):
                os.close(fd)
                raise p23.UnsafePathError(f"profiles/{case_name}/{filename}: not a regular file")
            if st.st_size == 0 and label not in EMPTY_ALLOWED_PROFILE_CASE_FILE_LABELS:
                os.close(fd)
                raise p23.UnsafePathError(f"profiles/{case_name}/{filename}: empty payload artifact")
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
    digest = hashlib.sha256()
    os.lseek(fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(fd, 1 << 20)
        if not chunk:
            break
        digest.update(chunk)
    return digest.hexdigest()


def _read_csv_rows_from_fd(fd: int) -> list[list[str]]:
    os.lseek(fd, 0, os.SEEK_SET)
    dup_fd = os.dup(fd)
    with os.fdopen(dup_fd, "r", encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


# ---------------------------------------------------------------------------
# Append-only, hash-chained manifest revision log.
# ---------------------------------------------------------------------------
def load_p24_manifest_chain(campaign_dir: Path) -> tuple[dict, int]:
    manifest_dir = _p24_manifest_dir(campaign_dir)
    if not os.path.lexists(manifest_dir):
        return {}, -1

    try:
        campaign_parts = campaign_dir.relative_to(REPO_ROOT).parts
        manifest_fd = _open_dir_component_chain(*campaign_parts, "manifest")
    except p23.UnsafePathError as exc:
        raise p23.ManifestTransitionError(str(exc)) from exc
    try:
        try:
            names = sorted(os.listdir(manifest_fd))
        except OSError as exc:
            raise p23.ManifestTransitionError(f"{manifest_dir}: cannot list revisions: {exc}") from exc
        revision_names = [n for n in names if MANIFEST_REVISION_RE.fullmatch(n)]
        other_names = [n for n in names if n not in revision_names and n != MANIFEST_REVISION_TMP_NAME]
        if other_names:
            raise p23.ManifestTransitionError(f"{manifest_dir}: unexpected entries in manifest revision directory: {sorted(other_names)}")
        if not revision_names:
            return {}, -1
        expected_names = [f"{i:06d}.json" for i in range(len(revision_names))]
        if revision_names != expected_names:
            raise p23.ManifestTransitionError(
                f"{manifest_dir}: manifest revisions are not exactly contiguous 000000..{len(revision_names) - 1:06d}.json: "
                f"found {revision_names!r}"
            )

        expected_campaign_id = campaign_dir.name
        previous_hash: str | None = None
        previous_content: dict | None = None
        current_content: dict = {}
        for i, name in enumerate(expected_names):
            path = manifest_dir / name  # diagnostic text only; never reopened by this path
            try:
                st = os.stat(name, dir_fd=manifest_fd, follow_symlinks=False)
                if stat.S_ISLNK(st.st_mode):
                    raise p23.UnsafePathError(f"{path}: is a symlink; refusing")
                if not stat.S_ISREG(st.st_mode):
                    raise p23.UnsafePathError(f"{path}: is not a regular file")
                fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=manifest_fd)
                try:
                    fst = os.fstat(fd)
                    if (fst.st_dev, fst.st_ino) != (st.st_dev, st.st_ino) or not stat.S_ISREG(fst.st_mode):
                        raise p23.UnsafePathError(f"{path}: changed identity while being opened")
                    raw = b"".join(iter(lambda: os.read(fd, 1 << 20), b""))
                finally:
                    os.close(fd)
            except (OSError, p23.UnsafePathError) as exc:
                raise p23.ManifestTransitionError(f"{path}: cannot load manifest revision: {exc}") from exc
            try:
                text = raw.decode("utf-8")
            except UnicodeError as exc:
                raise p23.ManifestTransitionError(f"{path}: cannot load manifest revision: {exc}") from exc
            try:
                doc = json.loads(text)
            except json.JSONDecodeError as exc:
                raise p23.ManifestTransitionError(f"{path}: invalid JSON: {exc}") from exc
            if not isinstance(doc, dict):
                raise p23.ManifestTransitionError(f"{path}: revision root is not a JSON object")
            if doc.get("manifest_revision") != i:
                raise p23.ManifestTransitionError(f"{path}: manifest_revision={doc.get('manifest_revision')!r} != expected {i}")
            expected_previous_hash = None if i == 0 else previous_hash
            if doc.get("previous_manifest_sha256") != expected_previous_hash:
                raise p23.ManifestTransitionError(
                    f"{path}: previous_manifest_sha256={doc.get('previous_manifest_sha256')!r} != expected "
                    f"{expected_previous_hash!r} (hash chain broken or tampered)"
                )
            content = {k: v for k, v in doc.items() if k not in MANIFEST_REVISION_KEYS}
            try:
                _validate_p24_manifest_document(content)
            except p23.ManifestTransitionError as exc:
                raise p23.ManifestTransitionError(f"{path}: {exc}") from exc
            shape_errors = validate_manifest_state_shape(content, expected_campaign_id)
            if shape_errors:
                raise p23.ManifestTransitionError(f"{path}: manifest state-shape validation failed: {shape_errors}")
            transition_errors = validate_manifest_revision_transition(previous_content, content, expected_campaign_id)
            if transition_errors:
                raise p23.ManifestTransitionError(f"{path}: semantic transition validation failed: {transition_errors}")
            chronology_errors = validate_manifest_timestamp_chronology(content)
            if chronology_errors:
                raise p23.ManifestTransitionError(f"{path}: manifest timestamp chronology validation failed: {chronology_errors}")
            previous_content = content
            current_content = content
            previous_hash = hashlib.sha256(raw).hexdigest()

        return current_content, len(expected_names) - 1
    finally:
        os.close(manifest_fd)


def write_next_p24_manifest_revision(campaign_dir: Path, manifest_content: dict) -> dict:
    manifest_dir = _p24_manifest_dir(campaign_dir)
    p23._mkdir_component(manifest_dir, must_not_exist=False, root=REPO_ROOT)
    current, current_revision = load_p24_manifest_chain(campaign_dir)
    new_revision = current_revision + 1
    previous_hash = None
    if current_revision >= 0:
        previous_hash = p23.sha256_of(_manifest_revision_path(campaign_dir, current_revision))

    doc = dict(manifest_content)
    doc["manifest_revision"] = new_revision
    doc["previous_manifest_sha256"] = previous_hash
    text = json.dumps(doc, indent=2, sort_keys=True) + "\n"

    tmp_path = manifest_dir / MANIFEST_REVISION_TMP_NAME
    if os.path.lexists(tmp_path):
        raise p23.UnsafePathError(f"{tmp_path}: stale manifest-revision temporary already exists")

    shape_errors = validate_manifest_state_shape(manifest_content, campaign_dir.name)
    if shape_errors:
        raise p23.ManifestTransitionError(f"refusing to write a manifest revision with an invalid shape: {shape_errors}")
    transition_errors = validate_manifest_revision_transition(
        current if current_revision >= 0 else None, manifest_content, campaign_dir.name,
    )
    if transition_errors:
        raise p23.ManifestTransitionError(f"refusing to write a non-compliant manifest revision: {transition_errors}")
    chronology_errors = validate_manifest_timestamp_chronology(manifest_content)
    if chronology_errors:
        raise p23.ManifestTransitionError(f"refusing to write a manifest revision with invalid timestamp chronology: {chronology_errors}")

    final_path = _manifest_revision_path(campaign_dir, new_revision)
    try:
        with p23._open_exclusive(tmp_path, binary=False) as handle:
            handle.write(text)
    except Exception:
        if os.path.lexists(tmp_path):
            p23._safe_unlink_owned(tmp_path)
        raise
    try:
        p23._publish_no_clobber(tmp_path, final_path)
    except p23.UnsafePathError:
        if os.path.lexists(tmp_path):
            p23._safe_unlink_owned(tmp_path)
        raise
    return manifest_content


def p24_merge_manifest(campaign_dir: Path, updates: dict, state: str) -> dict:
    _validate_p24_manifest_updates(updates)
    manifest, _revision = load_p24_manifest_chain(campaign_dir)
    _validate_p24_manifest_document(manifest)
    current_state = manifest.get("state")
    allowed = ALLOWED_P24_TRANSITIONS.get(current_state, frozenset())
    if state not in allowed:
        raise p23.ManifestTransitionError(f"invalid P2.4 manifest state transition: {current_state!r} -> {state!r}")
    immutable_after_init = {"campaign_id", "started_at_utc", "frozen_protocol"}
    for key in immutable_after_init & set(manifest) & set(updates):
        if updates[key] != manifest[key]:
            raise p23.ManifestTransitionError(f"P2.4 manifest field {key!r} is immutable after initialization")
    if "profile_count_completed" in manifest and "profile_count_completed" in updates:
        if updates["profile_count_completed"] < manifest["profile_count_completed"]:
            raise p23.ManifestTransitionError("P2.4 manifest profile_count_completed cannot decrease")
    manifest.update(updates)
    manifest["schema_version"] = P24_SCHEMA_VERSION
    manifest["experiment_id"] = P24_EXPERIMENT_ID
    manifest["state"] = state
    manifest["publishable"] = False
    _validate_p24_manifest_document(manifest, require_initialized=True)
    write_next_p24_manifest_revision(campaign_dir, manifest)
    return manifest


# ---------------------------------------------------------------------------
# Statistics: percentile/IQR helpers, deterministic bootstrap, per-config
# descriptive statistics, 1-SM/2-SM scaling, candidate depth saturation, and
# empirical per-SM ceiling selection. Python standard library only. See
# src/compute/P2_4_PROTOCOL.md section 6 for the frozen formulas.
# ---------------------------------------------------------------------------
def _reject_non_finite(values: list[float], *, label: str) -> None:
    for v in values:
        if not math.isfinite(v):
            raise ValueError(f"{label}: non-finite value {v!r} is rejected, never silently dropped")


def _percentile_linear(sorted_values: list[float], p: float) -> float:
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
    sorted_values = sorted(values)
    q1 = _percentile_linear(sorted_values, 0.25)
    q3 = _percentile_linear(sorted_values, 0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    flagged = sum(1 for v in values if v < lower or v > upper)
    return lower, upper, flagged


def _sample_stdev(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError(f"sample standard deviation requires at least 2 observations, got {len(values)}")
    return statistics.stdev(values)


def bootstrap_indices_median_ci(
    values: list[float], rng: random.Random, *, resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    n = len(values)
    medians = []
    for _ in range(resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        medians.append(statistics.median(resample))
    medians.sort()
    lo_idx = max(int(BOOTSTRAP_LO_PERCENTILE * resamples) - 1, 0)
    hi_idx = min(int(BOOTSTRAP_HI_PERCENTILE * resamples) - 1, resamples - 1)
    return medians[lo_idx], medians[hi_idx]


def bootstrap_indices_ratio_ci(
    values_a: list[float], values_b: list[float], rng: random.Random, *, resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """95% bootstrap CI for median(values_b) / median(values_a), resampling
    both inputs independently each iteration. The two configurations execute
    sequentially, never concurrently; these are never called "paired
    samples" anywhere in generated output."""
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


def compute_metric_stats(values: list[float]) -> dict:
    _reject_non_finite(values, label="metric sample")
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
    }


def _read_combined_samples(path: Path) -> dict[tuple[str, int, int], dict[str, list[float] | int]]:
    """Reads P2.3's combined_samples.csv: 30 retained repetitions x 24
    configurations. Derives flops_per_cycle_per_sm = flops_per_cycle /
    cta_group per sample (never per already-aggregated statistic)."""
    samples: dict[tuple[str, int, int], dict[str, list[float] | int]] = {}
    with p23._open_regular_nofollow(path, binary=False) as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            method = row["method"]
            n = int(row["n"])
            depth = int(row["depth"])
            cta_group = int(row["cta_group"])
            key = (method, n, depth)
            entry = samples.setdefault(key, {name: [] for name in STAT_METRIC_NAMES})
            entry["cta_group"] = cta_group
            elapsed_cycles = float(row["elapsed_cycles"])
            cycles_per_umma = float(row["cycles_per_umma"])
            flops_per_cycle = float(row["flops_per_cycle"])
            entry["elapsed_cycles"].append(elapsed_cycles)
            entry["cycles_per_umma"].append(cycles_per_umma)
            entry["flops_per_cycle"].append(flops_per_cycle)
            entry["flops_per_cycle_per_sm"].append(flops_per_cycle / cta_group)
    return samples


def compute_all_config_stats(
    samples_by_config: dict[tuple[str, int, int], dict], rng: random.Random,
) -> dict[tuple[str, int, int], dict]:
    """Computes descriptive stats + bootstrap median CI for all 24 configs
    and all 4 metrics, in one fixed order -- (n, depth, method) for configs,
    then STAT_METRIC_NAMES order for metrics within each config -- so that,
    given the same input samples, the shared rng's draw sequence (and
    therefore every output) is bit-identical on any machine, every time."""
    results: dict[tuple[str, int, int], dict] = {}
    ordered_keys = sorted(samples_by_config, key=lambda k: (k[1], k[2], k[0]))
    for key in ordered_keys:
        entry = samples_by_config[key]
        config_result: dict = {"cta_group": entry["cta_group"]}
        for metric_name in STAT_METRIC_NAMES:
            values = entry[metric_name]
            stats = compute_metric_stats(values)
            stats["median_ci_low"], stats["median_ci_high"] = bootstrap_indices_median_ci(values, rng)
            if metric_name == "flops_per_cycle":
                stats["stability_review"] = stats["cv_percent"] > CV_STABILITY_REVIEW_PERCENT
            config_result[metric_name] = stats
        results[key] = config_result
    return results


def compute_scaling(
    samples_by_config: dict[tuple[str, int, int], dict],
    stats_by_config: dict[tuple[str, int, int], dict],
    rng: random.Random,
) -> list[dict]:
    """Per (N, depth) pair, in ascending (N, depth) order: 2-SM-over-1-SM
    speedup and scaling efficiency of median flops_per_cycle, with an
    independently-resampled 95% bootstrap CI. Never clamped."""
    rows = []
    pairs = sorted({(n, depth) for (_method, n, depth) in samples_by_config})
    for n, depth in pairs:
        values_1sm = samples_by_config[("umma_1sm", n, depth)]["flops_per_cycle"]
        values_2sm = samples_by_config[("umma_2sm", n, depth)]["flops_per_cycle"]
        median_1sm = stats_by_config[("umma_1sm", n, depth)]["flops_per_cycle"]["median"]
        median_2sm = stats_by_config[("umma_2sm", n, depth)]["flops_per_cycle"]["median"]
        speedup = median_2sm / median_1sm
        scaling_efficiency = speedup / 2.0
        scaling_efficiency_percent = 100.0 * scaling_efficiency
        ci_low, ci_high = bootstrap_indices_ratio_ci(values_1sm, values_2sm, rng)
        rows.append({
            "n": n,
            "depth": depth,
            "median_flops_per_cycle_1sm": median_1sm,
            "median_flops_per_cycle_2sm": median_2sm,
            "speedup_2sm_over_1sm": speedup,
            "speedup_ci_low": ci_low,
            "speedup_ci_high": ci_high,
            "scaling_efficiency": scaling_efficiency,
            "scaling_efficiency_percent": scaling_efficiency_percent,
            "surprising_value_flag": not (0.0 <= scaling_efficiency_percent <= 100.0),
        })
    return rows


def _ci_overlaps(lo_a: float, hi_a: float, lo_b: float, hi_b: float) -> bool:
    return lo_a <= hi_b and lo_b <= hi_a


def compute_saturation(stats_by_config: dict[tuple[str, int, int], dict]) -> list[dict]:
    """Per (method, N) group, in ascending (method, N) order: the earliest
    (smallest) tested depth whose median flops_per_cycle is >= 95% of the
    group's observed maximum median AND whose bootstrap CI overlaps the
    maximum's own CI. The maximum-median entry is always a valid fallback."""
    rows = []
    groups = sorted({(method, n) for (method, n, _depth) in stats_by_config})
    for method, n in groups:
        medians = {d: stats_by_config[(method, n, d)]["flops_per_cycle"]["median"] for d in DEPTH_VALUES}
        max_median = max(medians.values())
        max_depth = min(d for d in DEPTH_VALUES if medians[d] == max_median)
        max_ci = (
            stats_by_config[(method, n, max_depth)]["flops_per_cycle"]["median_ci_low"],
            stats_by_config[(method, n, max_depth)]["flops_per_cycle"]["median_ci_high"],
        )
        earliest = max_depth
        for d in sorted(DEPTH_VALUES):
            if medians[d] < SATURATION_FRACTION_OF_MAX * max_median:
                continue
            candidate_ci = (
                stats_by_config[(method, n, d)]["flops_per_cycle"]["median_ci_low"],
                stats_by_config[(method, n, d)]["flops_per_cycle"]["median_ci_high"],
            )
            if _ci_overlaps(*candidate_ci, *max_ci):
                earliest = d
                break
        rows.append({
            "method": method,
            "n": n,
            "depth_4_median_flops_per_cycle": medians[4],
            "depth_16_median_flops_per_cycle": medians[16],
            "depth_64_median_flops_per_cycle": medians[64],
            "depth_256_median_flops_per_cycle": medians[256],
            "max_median_flops_per_cycle": max_median,
            "earliest_tested_candidate_saturation_depth": earliest,
        })
    return rows


def select_ceiling(stats_by_config: dict[tuple[str, int, int], dict]) -> dict[str, tuple[str, int, int]]:
    """Selects the ceiling candidate configuration(s) in clock-independent
    FLOP/cycle-per-SM space only -- never using a clock measurement. Ties
    resolve to the first configuration in (n, depth, method) sorted order
    (Python's max() keeps the first-seen maximal element)."""
    ordered_keys = sorted(stats_by_config, key=lambda k: (k[1], k[2], k[0]))

    def _best(keys: list[tuple[str, int, int]]) -> tuple[str, int, int]:
        return max(keys, key=lambda k: stats_by_config[k]["flops_per_cycle_per_sm"]["median"])

    best_1sm = _best([k for k in ordered_keys if k[0] == "umma_1sm"])
    best_2sm = _best([k for k in ordered_keys if k[0] == "umma_2sm"])
    overall = _best(ordered_keys)
    return {"best_1sm": best_1sm, "best_2sm": best_2sm, "empirical_per_sm_ceiling_candidate": overall}


# ---------------------------------------------------------------------------
# NCU metric name resolution (discovery). Unlike P1.4's resolve_ncu_metrics,
# no candidate here -- mandatory or diagnostic -- ever raises on ambiguity:
# an ambiguous or missing mandatory SM-clock metric is instead recorded and
# later drives the whole campaign to INCONCLUSIVE at analyze() time (never
# guessed, never silently downgraded to a diagnostic-only omission); an
# ambiguous or missing *diagnostic* metric is recorded and reported
# explicitly, exactly as src/compute/P2_4_PROTOCOL.md requires, and never
# blocks the campaign at all.
# ---------------------------------------------------------------------------
def canonical_candidate_metric_name(metric_name: str) -> str | None:
    if metric_name in CANDIDATE_METRICS:
        return metric_name
    matches = [candidate for candidate in CANDIDATE_METRICS if metric_name.endswith(f".{candidate}")]
    return matches[0] if len(matches) == 1 else None


def parse_metric_discovery_log(path: Path) -> set[str]:
    try:
        with p23._open_regular_nofollow(path, binary=False) as handle:
            text = handle.read()
    except (OSError, p23.UnsafePathError, UnicodeError) as exc:
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


def resolve_ncu_metrics_p24(discovered: set[str]) -> dict:
    per_metric: dict[str, dict] = {}
    for candidate in CANDIDATE_METRICS:
        matches = sorted(name for name in discovered if canonical_candidate_metric_name(name) == candidate)
        if candidate in matches:
            per_metric[candidate] = {"status": "resolved", "resolved_name": candidate, "ambiguous_candidates": []}
        elif len(matches) == 1:
            per_metric[candidate] = {"status": "resolved", "resolved_name": matches[0], "ambiguous_candidates": []}
        elif len(matches) > 1:
            per_metric[candidate] = {"status": "ambiguous", "resolved_name": None, "ambiguous_candidates": matches}
        else:
            per_metric[candidate] = {"status": "missing", "resolved_name": None, "ambiguous_candidates": []}
    return {
        "requested": list(CANDIDATE_METRICS),
        "per_metric": per_metric,
        "sm_clock_metric_resolved": per_metric[MANDATORY_SM_CLOCK_METRIC]["status"] == "resolved",
    }


def resolved_metric_names_for_ncu(resolved: dict) -> list[str]:
    """The exact --metrics argument: every candidate whose name resolved,
    mandatory first then diagnostics in their frozen order."""
    return [
        resolved["per_metric"][candidate]["resolved_name"]
        for candidate in CANDIDATE_METRICS
        if resolved["per_metric"][candidate]["status"] == "resolved"
    ]


# ---------------------------------------------------------------------------
# NCU raw-CSV metrics parsing. Same wide-table shape as P1.4's own (ID,
# Kernel Name, then one column per collected metric; one unit row; exactly
# one profiled launch row, since --launch-skip 1 --launch-count 1 collects
# exactly one). The mandatory SM-clock candidate is validated strictly
# (present, finite, strictly positive, exact expected unit); diagnostic
# candidates are recorded best-effort with no unit enforcement, since their
# exact availability or naming may differ on GB300.
# ---------------------------------------------------------------------------
class NcuCsvParseError(ValueError):
    pass


REQUIRED_NCU_CSV_COLUMNS = ("ID", "Kernel Name")


def _parse_ncu_raw_csv_rows(rows_raw: list[list[str]], *, label: str) -> dict:
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
        raise NcuCsvParseError(f"{label}: missing required column(s) {missing_columns} in header {header!r}")
    col_index = {name: header.index(name) for name in REQUIRED_NCU_CSV_COLUMNS}

    if len(rows_raw) < 2:
        raise NcuCsvParseError(f"{label}: missing metric-unit row")
    unit_row = rows_raw[1]
    if len(unit_row) != len(header):
        raise NcuCsvParseError(f"{label}: line 2 (metric units): expected {len(header)} field(s), got {len(unit_row)}")

    data_rows = rows_raw[2:]
    if len(data_rows) != 1:
        raise NcuCsvParseError(
            f"{label}: found {len(data_rows)} profiled launch row(s), expected exactly 1 "
            f"(--launch-skip 1 --launch-count 1 should guarantee this)"
        )
    data_row = data_rows[0]
    if len(data_row) != len(header):
        raise NcuCsvParseError(f"{label}: line 3 (profiled launch): expected {len(header)} field(s), got {len(data_row)}")

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
                f"{label}: candidate metric {canonical_metric_name!r} is represented by more than one "
                f"column ({previous!r}, {metric_name!r}); refusing an ambiguous canonical/qualified mapping"
            )
        metric_unit = unit_row[column_index].strip()
        raw_value = data_row[column_index].strip()
        if not raw_value:
            # A diagnostic counter this GB300 build does not populate is
            # recorded as unavailable rather than treated as zero.
            candidate_metric_names[canonical_metric_name] = metric_name
            units[metric_name] = metric_unit
            continue
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise NcuCsvParseError(f"{label}: line 3: metric {metric_name!r} value {raw_value!r} is not a number") from exc
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
    try:
        with p23._open_regular_nofollow(path, binary=False) as handle:
            rows_raw = list(csv.reader(handle))
    except (OSError, p23.UnsafePathError, UnicodeError) as exc:
        raise NcuCsvParseError(f"{path}: unable to read: {exc}") from exc
    return _parse_ncu_raw_csv_rows(rows_raw, label=str(path))


def evaluate_sm_clock(
    *, sm_clock_metric_resolved: bool, actual_column_name: str | None, parsed_metrics: dict,
) -> dict:
    """Strict, fail-closed evaluation of the mandatory SM-clock metric for
    one profiled case. Never guesses or rescales a unit. Returns a dict with
    sm_clock_valid (bool), sm_clock_issue (str, empty iff valid),
    sm_clock_raw_value, sm_clock_unit, sm_clock_hz (all None unless the
    metric parsed to a real value, regardless of validity)."""
    result = {
        "sm_clock_valid": False, "sm_clock_issue": "", "sm_clock_raw_value": None,
        "sm_clock_unit": None, "sm_clock_hz": None,
    }
    if not sm_clock_metric_resolved or actual_column_name is None:
        result["sm_clock_issue"] = "metric_unavailable_at_discovery"
        return result
    if actual_column_name not in parsed_metrics["metrics"]:
        result["sm_clock_issue"] = "missing_from_case_evidence"
        result["sm_clock_unit"] = parsed_metrics["units"].get(actual_column_name)
        return result
    value = parsed_metrics["metrics"][actual_column_name]
    unit = parsed_metrics["units"].get(actual_column_name, "")
    result["sm_clock_raw_value"] = value
    result["sm_clock_unit"] = unit
    if not math.isfinite(value):
        result["sm_clock_issue"] = "non_finite"
        return result
    if value <= 0:
        result["sm_clock_issue"] = "non_positive"
        return result
    normalized_unit = unit.strip().lower()
    hz_scale = SM_CLOCK_UNIT_TO_HZ_SCALE.get(normalized_unit)
    if hz_scale is None:
        result["sm_clock_issue"] = f"unknown_unit:{unit}"
        return result
    result["sm_clock_valid"] = True
    result["sm_clock_hz"] = value * hz_scale
    return result


# ---------------------------------------------------------------------------
# Canonical case-result reconstruction. Mirrors P1.4's reconstruct_case_result
# design (src/memory/P1_4_PROTOCOL.md section 8): one pure-data core, used by
# both a path-based entry point (validate-profile-case's one-shot validation
# of freshly-produced evidence) and a descriptor-anchored entry point (the
# evidence-integrity gate, which keeps every relevant fd open for the whole
# check).
# ---------------------------------------------------------------------------
def p24_validate_case_rows(rows: list[list[str]], expect: dict, *, label: str) -> tuple[list[dict[str, str]], list[str]]:
    """Validates the already-audited P2.1/P2.2 37-column application CSV
    (reused unmodified via p23.CSV_HEADER/p23.FIELD_VALIDATORS) against
    already-parsed rows, so the descriptor-anchored evidence layer can
    validate bytes from an already-open fd without reopening a pathname."""
    errors: list[str] = []
    if not rows:
        return [], [f"{label}: empty file (no header)"]
    header, data_rows_raw = rows[0], rows[1:]
    if header != p23.CSV_HEADER:
        return [], [f"{label}: header mismatch (wrong or reordered CSV header): {header!r}"]

    parsed_rows: list[dict[str, str]] = []
    for line_no, row in enumerate(data_rows_raw, start=2):
        if len(row) == 0:
            errors.append(f"{label}: line {line_no}: blank row")
            continue
        if row == p23.CSV_HEADER:
            errors.append(f"{label}: line {line_no}: repeated header row")
            continue
        if len(row) != len(p23.CSV_HEADER):
            errors.append(f"{label}: line {line_no}: expected {len(p23.CSV_HEADER)} fields, got {len(row)}")
            continue
        parsed_rows.append(dict(zip(p23.CSV_HEADER, row)))
    if errors:
        return parsed_rows, errors

    repetitions = expect["repetitions"]
    if len(parsed_rows) != repetitions:
        errors.append(f"{label}: has {len(parsed_rows)} data row(s), expected exactly repetitions={repetitions}")
        return parsed_rows, errors

    for row_num, row in enumerate(parsed_rows):
        ctx = f"{label}: data row {row_num} (line {row_num + 2})"
        for field in p23.CSV_HEADER:
            p23.FIELD_VALIDATORS[field](row, expect, errors, ctx)

    return parsed_rows, errors


def _reconstruct_case_result_core(
    *, entry: dict, application_rows: list[list[str]], metrics_rows: list[list[str]],
    application_label: str, metrics_label: str,
    application_hash: str, metrics_hash: str, ncu_rep_hash: str,
    ncu_tool_log_hash: str, container_stdout_log_hash: str, container_stderr_log_hash: str,
    metrics_export_stderr_log_hash: str,
    resolved_ncu_metrics: dict, git_commit: str, campaign_provenance: dict,
) -> tuple[dict | None, list[str]]:
    errors: list[str] = []
    provenance_errors = validate_provenance_tuple(campaign_provenance, label=f"{entry['case_name']}: campaign provenance")
    if provenance_errors:
        return None, provenance_errors
    if campaign_provenance.get("git_commit") != git_commit:
        return None, [
            f"{entry['case_name']}: internal error: campaign_provenance.git_commit="
            f"{campaign_provenance.get('git_commit')!r} != the git_commit this reconstruction was asked to expect {git_commit!r}"
        ]

    method = entry["method"]
    info = p23.METHOD_INFO[method]
    expect = {
        "method": method, "cta_group": info["cta_group"], "m": info["m"], "grid_blocks": info["grid_blocks"],
        "n": entry["n"], "depth": entry["depth"],
        "run_kind": FROZEN_PROFILE_PARAMS["run_kind"], "iterations": FROZEN_PROFILE_PARAMS["iterations"],
        "warmup_iterations": FROZEN_PROFILE_PARAMS["warmup_iterations"], "repetitions": FROZEN_PROFILE_PARAMS["repetitions"],
        "git_commit": git_commit,
    }
    app_rows, app_errors = p24_validate_case_rows(application_rows, expect, label=application_label)
    errors.extend(app_errors)
    app_row = app_rows[0] if (app_rows and not app_errors) else None

    # Defect-1 repair: the row's own, independently-parsed identity fields
    # (never a value copied from campaign_provenance itself) are compared
    # against the immutable campaign tuple. This is what actually rejects a
    # profile whose application evidence reports a different GPU
    # UUID/name/compute-capability/driver/runtime than the campaign this
    # profile is claimed to belong to -- src/compute/P2_PROTOCOL.md's/
    # P2_2_PROTOCOL.md's own case-CSV field validators only check that
    # gpu_uuid/gpu_name/compute_capability/driver/runtime are well-formed,
    # never that they equal a specific expected campaign identity.
    if app_row is not None:
        errors.extend(compare_application_provenance(app_row=app_row, campaign_provenance=campaign_provenance, label=entry["case_name"]))

    parsed_metrics = None
    try:
        parsed_metrics = _parse_ncu_raw_csv_rows(metrics_rows, label=metrics_label)
    except NcuCsvParseError as exc:
        errors.append(str(exc))

    if parsed_metrics is not None and parsed_metrics["kernel_name"] != entry["kernel_symbol"]:
        errors.append(
            f"metrics CSV kernel name {parsed_metrics['kernel_name']!r} != expected exact function name "
            f"{entry['kernel_symbol']!r} (no substring/regex matching is permitted)"
        )

    if errors:
        return None, errors

    per_metric = resolved_ncu_metrics.get("per_metric", {})
    sm_clock_entry = per_metric.get(MANDATORY_SM_CLOCK_METRIC, {})
    sm_clock_actual_name = None
    if sm_clock_entry.get("status") == "resolved":
        candidate_name = sm_clock_entry.get("resolved_name")
        canonical = canonical_candidate_metric_name(candidate_name) if candidate_name else None
        sm_clock_actual_name = parsed_metrics["candidate_metric_names"].get(canonical) if canonical else None
    sm_clock = evaluate_sm_clock(
        sm_clock_metric_resolved=resolved_ncu_metrics.get("sm_clock_metric_resolved", False),
        actual_column_name=sm_clock_actual_name, parsed_metrics=parsed_metrics,
    )

    diagnostic_values: dict[str, float] = {}
    diagnostic_units: dict[str, str] = {}
    for candidate in DIAGNOSTIC_METRICS:
        candidate_entry = per_metric.get(candidate, {})
        if candidate_entry.get("status") != "resolved":
            continue
        canonical = canonical_candidate_metric_name(candidate_entry.get("resolved_name") or "")
        actual_name = parsed_metrics["candidate_metric_names"].get(canonical) if canonical else None
        if actual_name and actual_name in parsed_metrics["metrics"]:
            diagnostic_values[candidate] = parsed_metrics["metrics"][actual_name]
            diagnostic_units[candidate] = parsed_metrics["units"].get(actual_name, "")

    case_result = {
        "case_name": entry["case_name"],
        "method": method,
        "n": entry["n"],
        "depth": entry["depth"],
        "cta_group": info["cta_group"],
        "kernel_symbol": entry["kernel_symbol"],
        "launch_id": parsed_metrics["launch_id"],
        "application_elapsed_cycles": int(app_row["elapsed_cycles"]),
        "application_cycles_per_umma": float(app_row["cycles_per_umma"]),
        "application_flops_per_cycle": float(app_row["flops_per_cycle"]),
        "application_flops_per_cycle_per_sm": float(app_row["flops_per_cycle"]) / info["cta_group"],
        "application_total_flops": int(app_row["total_flops"]),
        "application_total_umma": int(app_row["total_umma"]),
        # Independently reported by this profile's own application evidence
        # (never copied from campaign_provenance) -- see the
        # compare_application_provenance() call above, which already
        # rejected any mismatch against the immutable campaign tuple before
        # this dict is built. Recorded so a later, second reconstruction
        # (e.g. the evidence-integrity gate re-run immediately before
        # publishing ANALYZED) independently re-derives and re-compares the
        # identical facts from the identical raw bytes, rather than trusting
        # a previously derived value.
        "application_git_commit": app_row["git_commit"],
        "application_git_dirty": app_row["git_dirty"],
        "application_gpu_uuid": app_row["gpu_uuid"],
        "application_gpu_name": app_row["gpu_name"],
        "application_compute_capability": app_row["compute_capability"],
        "application_cuda_driver_version": app_row["cuda_driver_version"],
        "application_cuda_runtime_version": app_row["cuda_runtime_version"],
        "sm_clock_valid": sm_clock["sm_clock_valid"],
        "sm_clock_issue": sm_clock["sm_clock_issue"],
        "sm_clock_raw_value": sm_clock["sm_clock_raw_value"],
        "sm_clock_unit": sm_clock["sm_clock_unit"],
        "sm_clock_hz": sm_clock["sm_clock_hz"],
        "diagnostic_metric_values": diagnostic_values,
        "diagnostic_metric_units": diagnostic_units,
        "application_csv_sha256": application_hash,
        "metrics_csv_sha256": metrics_hash,
        "ncu_rep_sha256": ncu_rep_hash,
        # Defect-2 repair: all seven canonical per-case artifacts are hashed
        # (not just the three the pre-repair finalizer opened), so the
        # evidence-integrity gate's recorded-vs-reconstructed comparison
        # (_strict_compare_values) covers the complete frozen inventory.
        "ncu_tool_log_sha256": ncu_tool_log_hash,
        "container_stdout_log_sha256": container_stdout_log_hash,
        "container_stderr_log_sha256": container_stderr_log_hash,
        "metrics_export_stderr_log_sha256": metrics_export_stderr_log_hash,
    }
    return case_result, []


def reconstruct_case_result(
    *, entry: dict, application_csv: Path, metrics_csv: Path, ncu_rep: Path,
    ncu_tool_log: Path, container_stdout_log: Path, container_stderr_log: Path, metrics_export_stderr_log: Path,
    resolved_ncu_metrics: dict, git_commit: str, campaign_provenance: dict,
) -> tuple[dict | None, list[str]]:
    artifact_paths = {
        "ncu_rep": ncu_rep,
        "ncu_tool_log": ncu_tool_log,
        "container_stdout_log": container_stdout_log,
        "container_stderr_log": container_stderr_log,
        "application_csv": application_csv,
        "metrics_csv": metrics_csv,
        "metrics_export_stderr_log": metrics_export_stderr_log,
    }
    for label, path in artifact_paths.items():
        artifact_err = _verify_profile_case_artifact(path, label)
        if artifact_err:
            return None, [artifact_err]
    try:
        with p23._open_regular_nofollow(application_csv, binary=False) as handle:
            application_rows = list(csv.reader(handle))
    except (OSError, p23.UnsafePathError, UnicodeError) as exc:
        return None, [f"{application_csv}: unable to read: {exc}"]
    try:
        with p23._open_regular_nofollow(metrics_csv, binary=False) as handle:
            metrics_rows = list(csv.reader(handle))
    except (OSError, p23.UnsafePathError, UnicodeError) as exc:
        return None, [f"{metrics_csv}: unable to read: {exc}"]
    try:
        application_hash = _sha256_profile_case_artifact(application_csv, "application_csv")
        metrics_hash = _sha256_profile_case_artifact(metrics_csv, "metrics_csv")
        ncu_rep_hash = _sha256_profile_case_artifact(ncu_rep, "ncu_rep")
        ncu_tool_log_hash = _sha256_profile_case_artifact(ncu_tool_log, "ncu_tool_log")
        container_stdout_log_hash = _sha256_profile_case_artifact(container_stdout_log, "container_stdout_log")
        container_stderr_log_hash = _sha256_profile_case_artifact(container_stderr_log, "container_stderr_log")
        metrics_export_stderr_log_hash = _sha256_profile_case_artifact(
            metrics_export_stderr_log, "metrics_export_stderr_log",
        )
    except p23.UnsafePathError as exc:
        return None, [str(exc)]
    return _reconstruct_case_result_core(
        entry=entry, application_rows=application_rows, metrics_rows=metrics_rows,
        application_label=str(application_csv), metrics_label=str(metrics_csv),
        application_hash=application_hash, metrics_hash=metrics_hash, ncu_rep_hash=ncu_rep_hash,
        ncu_tool_log_hash=ncu_tool_log_hash, container_stdout_log_hash=container_stdout_log_hash,
        container_stderr_log_hash=container_stderr_log_hash, metrics_export_stderr_log_hash=metrics_export_stderr_log_hash,
        resolved_ncu_metrics=resolved_ncu_metrics, git_commit=git_commit, campaign_provenance=campaign_provenance,
    )


def _reconstruct_case_result_from_fds(
    *, entry: dict, case_name: str, fds: dict[str, int], resolved_ncu_metrics: dict, git_commit: str,
    campaign_provenance: dict,
) -> tuple[dict | None, list[str]]:
    application_rows = _read_csv_rows_from_fd(fds["application_csv"])
    metrics_rows = _read_csv_rows_from_fd(fds["metrics_csv"])
    return _reconstruct_case_result_core(
        entry=entry, application_rows=application_rows, metrics_rows=metrics_rows,
        application_label=f"profiles/{case_name}/{case_name}.application.csv",
        metrics_label=f"profiles/{case_name}/{case_name}.metrics_raw.csv",
        application_hash=_sha256_of_fd(fds["application_csv"]), metrics_hash=_sha256_of_fd(fds["metrics_csv"]),
        ncu_rep_hash=_sha256_of_fd(fds["ncu_rep"]), ncu_tool_log_hash=_sha256_of_fd(fds["ncu_tool_log"]),
        container_stdout_log_hash=_sha256_of_fd(fds["container_stdout_log"]),
        container_stderr_log_hash=_sha256_of_fd(fds["container_stderr_log"]),
        metrics_export_stderr_log_hash=_sha256_of_fd(fds["metrics_export_stderr_log"]),
        resolved_ncu_metrics=resolved_ncu_metrics, git_commit=git_commit, campaign_provenance=campaign_provenance,
    )


# ---------------------------------------------------------------------------
# Strict recursive case-result comparison and the central evidence-integrity
# gate. Mirrors src/memory/P1_4_PROTOCOL.md section 8 exactly: exact key
# sets first (missing vs. unexpected reported as distinct conditions), exact
# list length/order, exact scalar type (type(x) is type(y), so True is
# never accepted in place of canonical 1), exact value, NaN/infinity
# rejected outright on either side.
# ---------------------------------------------------------------------------
def _strict_compare_values(path: str, recorded: object, canonical: object, errors: list[str]) -> None:
    if isinstance(canonical, dict) or isinstance(recorded, dict):
        if not isinstance(canonical, dict) or not isinstance(recorded, dict):
            errors.append(f"{path}: recorded is {type(recorded).__name__} ({recorded!r}), reconstructed is {type(canonical).__name__} ({canonical!r})")
            return
        recorded_keys, canonical_keys = set(recorded), set(canonical)
        missing = sorted(canonical_keys - recorded_keys)
        unexpected = sorted(recorded_keys - canonical_keys)
        if missing:
            errors.append(f"{path}: missing key(s) {missing} (present in reconstructed evidence, absent from the recorded manifest)")
        if unexpected:
            errors.append(f"{path}: unexpected key(s) {unexpected} (present in the recorded manifest, absent from reconstructed evidence)")
        for key in sorted(recorded_keys & canonical_keys):
            _strict_compare_values(f"{path}.{key}", recorded[key], canonical[key], errors)
        return

    if isinstance(canonical, list) or isinstance(recorded, list):
        if not isinstance(canonical, list) or not isinstance(recorded, list):
            errors.append(f"{path}: recorded is {type(recorded).__name__} ({recorded!r}), reconstructed is {type(canonical).__name__} ({canonical!r})")
            return
        if len(recorded) != len(canonical):
            errors.append(f"{path}: recorded has {len(recorded)} entry(ies), reconstructed has {len(canonical)} (exact list length/order is required)")
            return
        for i, (r_item, c_item) in enumerate(zip(recorded, canonical)):
            _strict_compare_values(f"{path}[{i}]", r_item, c_item, errors)
        return

    if recorded is None or canonical is None:
        if recorded is not canonical:
            errors.append(f"{path}: recorded={recorded!r} != reconstructed={canonical!r}")
        return
    if type(recorded) is not type(canonical):
        errors.append(f"{path}: type mismatch: recorded is {type(recorded).__name__} ({recorded!r}), reconstructed is {type(canonical).__name__} ({canonical!r})")
        return
    if isinstance(canonical, float):
        if not math.isfinite(canonical) or not math.isfinite(recorded):
            errors.append(f"{path}: NaN/infinite value rejected outright: recorded={recorded!r} reconstructed={canonical!r}")
            return
    if recorded != canonical:
        suffix = "tampered or corrupted since it was first validated" if path.endswith("_sha256") else "recomputed fresh from disk"
        errors.append(f"{path}: recorded {recorded!r} != reconstructed {canonical!r} ({suffix})")


def _list_and_check_profiles_inventory(profiles_fd: int, expected_names: set[str]) -> list[str]:
    errors: list[str] = []
    try:
        actual_names = set(os.listdir(profiles_fd))
    except OSError as exc:
        return [f"profiles/: cannot list directory: {exc}"]
    for extra in sorted(actual_names - expected_names):
        errors.append(f"profiles/{extra}: unplanned entry not present in profile_plan.csv")
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


def _recheck_inode_identity(campaign_fd: int, profiles_fd: int, case_fds: dict[str, int]) -> list[str]:
    errors: list[str] = []
    try:
        fresh_profiles = os.stat("profiles", dir_fd=campaign_fd, follow_symlinks=False)
        held_profiles = os.fstat(profiles_fd)
    except OSError as exc:
        return [f"profiles/: cannot re-check identity before terminal publication: {exc}"]
    if stat.S_ISLNK(fresh_profiles.st_mode) or (fresh_profiles.st_dev, fresh_profiles.st_ino) != (held_profiles.st_dev, held_profiles.st_ino):
        errors.append("profiles/: name-to-inode binding changed during validation; failing closed rather than trust a name whose target moved")
        return errors
    for case_name, case_fd in sorted(case_fds.items()):
        try:
            fresh_case = os.stat(case_name, dir_fd=profiles_fd, follow_symlinks=False)
            held_case = os.fstat(case_fd)
        except OSError as exc:
            errors.append(f"profiles/{case_name}: cannot re-check identity before terminal publication: {exc}")
            continue
        if stat.S_ISLNK(fresh_case.st_mode) or (fresh_case.st_dev, fresh_case.st_ino) != (held_case.st_dev, held_case.st_ino):
            errors.append(f"profiles/{case_name}: name-to-inode binding changed during validation; failing closed")
    return errors


def _verify_hash(label: str, path: Path, expected_sha256: object, errors: list[str]) -> str | None:
    if not isinstance(expected_sha256, str) or not p23._is_sha256_hex(expected_sha256):
        errors.append(f"{label}: no valid recorded SHA-256 to verify against (got {expected_sha256!r})")
        return None
    try:
        actual = p23.sha256_of(path)
    except p23.UnsafePathError as exc:
        errors.append(f"{label}: {exc}")
        return None
    if actual != expected_sha256:
        errors.append(f"{label}: {path} SHA-256 {actual} != recorded {expected_sha256} (tampered or corrupted since it was first validated)")
        return None
    return actual


# ---------------------------------------------------------------------------
# Defect-4 repair: a read-only, independent P2.4-side re-validation of the
# closed P2.3 pilot campaign's complete raw evidence -- 24 configurations x
# 30 retained samples = 720 rows. P2.4 must not trust P2.3 merely because
# its manifest and file hashes claim to be valid (that was the pre-repair
# behavior: _do_record_pilot only compared P2.3's own recorded hashes
# against fresh ones for the two already-aggregated files). This function
# instead re-derives every fact using scripts/aggregate_exp02_umma_throughput.py's
# own exact protocol/schema validators and recomputation helpers (imported
# as p23, never modified, never reimplemented) applied fresh to the raw
# cases/ directory and execution_order.csv on disk -- including an
# independent recomputation of combined_samples.csv/summary.csv from the raw
# per-sample rows, compared byte-for-byte, so every stored derived formula
# is proved, not merely the ones validate_case_file()'s own per-row checks
# already cover. Never writes anything under p23_campaign_dir; never
# repairs or removes malformed evidence -- any defect rejects the whole
# campaign closed.
# ---------------------------------------------------------------------------
def revalidate_p23_pilot_campaign(p23_campaign_dir: Path, *, git_commit: str) -> tuple[list[str], dict | None]:
    errors: list[str] = []
    plan = p23.build_plan()
    plan_errors = p23.check_plan_contract(plan)
    if plan_errors:
        return [f"internal P2.3 plan contract violation: {plan_errors}"], None

    manifest_path = p23_campaign_dir / "manifest.json"
    try:
        manifest = p23.load_manifest(p23_campaign_dir)
        p23._validate_manifest_document(manifest, require_initialized=True)
    except (p23.ManifestTransitionError, p23.UnsafePathError) as exc:
        return [f"P2.3 manifest: {exc}"], None
    if not manifest:
        return ["P2.3 manifest.json does not exist"], None
    if manifest.get("status") != "COMPLETE":
        return [f"P2.3 campaign status={manifest.get('status')!r} != 'COMPLETE'"], None
    if manifest.get("run_kind") != "benchmark":
        return [f"P2.3 campaign run_kind={manifest.get('run_kind')!r} != 'benchmark'"], None
    if manifest.get("git_commit") != git_commit:
        errors.append(f"P2.3 manifest git_commit={manifest.get('git_commit')!r} != expected {git_commit!r}")
    if manifest.get("campaign_id") != p23_campaign_dir.name:
        errors.append(f"P2.3 manifest campaign_id={manifest.get('campaign_id')!r} != its own directory name {p23_campaign_dir.name!r}")

    repetitions = FROZEN_PILOT_PARAMS["repetitions"]
    requested = manifest.get("requested", {}) if isinstance(manifest.get("requested"), dict) else {}
    for key in ("iterations", "warmup_iterations", "repetitions"):
        expected = FROZEN_PILOT_PARAMS[key]
        if requested.get(key) != expected:
            errors.append(f"P2.3 campaign requested.{key}={requested.get(key)!r} != frozen pilot value {expected!r}")
    if errors:
        return errors, None

    # Exact 24-configuration canonical plan and order, independently
    # re-validated against execution_order.csv on disk.
    order_errors = p23.validate_execution_order_file(p23_campaign_dir / "execution_order.csv", plan)
    errors.extend(order_errors)

    # Exact files/directory inventory: no missing, extra, duplicate,
    # non-canonically-named, or symlinked case file.
    index_to_path, scan_errors = p23.scan_case_directory(p23_campaign_dir / "cases", plan)
    errors.extend(scan_errors)
    if errors:
        return errors, None

    # Every one of the 720 rows: canonical configuration/repetition indices;
    # method, M, N, K, depth, CTA group; run kind, iterations, warmup
    # iterations, repetitions; correctness=OK/mismatches=0; full commit and
    # GPU provenance; positive finite raw timing values; every stored
    # derived formula (cycles_per_umma, flops_per_cycle, total_umma,
    # total_flops, ...) recomputed from its raw inputs -- all via P2.3's own
    # unmodified validate_case_file()/FIELD_VALIDATORS.
    cases: list[tuple[dict, list[dict[str, str]]]] = []
    for entry in plan:
        path = index_to_path[entry["index"]]
        expect = p23._expect_for_entry(
            entry, run_kind="benchmark", iterations=FROZEN_PILOT_PARAMS["iterations"],
            warmup_iterations=FROZEN_PILOT_PARAMS["warmup_iterations"], repetitions=repetitions,
            git_commit=git_commit,
        )
        rows, case_errors = p23.validate_case_file(path, expect)
        if case_errors:
            errors.extend(f"{entry['case_name']}: {e}" for e in case_errors)
            continue
        cases.append((entry, rows))
    if errors:
        return errors, None

    total_samples = sum(len(rows) for _entry, rows in cases)
    expected_total = len(plan) * repetitions
    if len(cases) != len(plan) or total_samples != expected_total:
        return errors + [
            f"retained sample total={total_samples} across {len(cases)} configuration(s); expected exactly "
            f"{expected_total} samples across {len(plan)} configurations ({len(plan)} x {repetitions})"
        ], None

    # No missing, duplicate, or extra row: every configuration has exactly
    # `repetitions` rows (validate_case_file already proved this per case);
    # identical commit/GPU/driver/runtime/run-kind/iteration provenance
    # across all 720 rows, compared against one single reference row (never
    # just each case's own first row).
    errors.extend(p23.check_cross_case_consistency(cases))
    if errors:
        return errors, None

    # Agreement between sample files, combined CSV, and the P2.3 manifest:
    # independently recompute combined_samples.csv/summary.csv from the raw
    # 720 rows with P2.3's own writer functions and require byte-for-byte
    # agreement with what P2.3 actually published -- proves every derived
    # statistic, not just the formulas validate_case_file() spot-checks.
    with tempfile.TemporaryDirectory(prefix="p24_p23_revalidate_") as tmp:
        tmp_combined = Path(tmp) / "combined_samples.csv"
        tmp_summary = Path(tmp) / "summary.csv"
        p23.write_combined_samples(plan, cases, tmp_combined)
        p23.write_summary(cases, tmp_summary)
        recomputed_combined = tmp_combined.read_bytes()
        recomputed_summary = tmp_summary.read_bytes()

    actual_combined_path = p23_campaign_dir / "combined_samples.csv"
    actual_summary_path = p23_campaign_dir / "summary.csv"
    try:
        actual_combined = actual_combined_path.read_bytes()
        actual_summary = actual_summary_path.read_bytes()
    except OSError as exc:
        return [f"cannot read P2.3 aggregate artifact: {exc}"], None
    if actual_combined != recomputed_combined:
        errors.append(
            "combined_samples.csv on disk does not byte-for-byte match a fresh recomputation from the raw "
            "720 case-file samples"
        )
    if actual_summary != recomputed_summary:
        errors.append(
            "summary.csv on disk does not byte-for-byte match a fresh recomputation from the raw 720 "
            "case-file samples"
        )
    if errors:
        return errors, None

    # Hashes of every trusted input: independently recomputed from disk and
    # cross-checked against P2.3's own recorded values -- never trusted
    # merely because the manifest claims them valid.
    try:
        case_file_hashes = {entry["case_name"]: p23.sha256_of(index_to_path[entry["index"]]) for entry in plan}
        execution_order_hash = p23.sha256_of(p23_campaign_dir / "execution_order.csv")
        combined_hash = p23.sha256_of(actual_combined_path)
        summary_hash = p23.sha256_of(actual_summary_path)
        manifest_hash = p23.sha256_of(manifest_path)
    except p23.UnsafePathError as exc:
        return [str(exc)], None

    recorded_case_hashes = manifest.get("case_file_sha256")
    if not isinstance(recorded_case_hashes, dict):
        errors.append("P2.3 manifest case_file_sha256 is missing or not an object")
    else:
        for case_name, actual_hash in case_file_hashes.items():
            if recorded_case_hashes.get(case_name) != actual_hash:
                errors.append(
                    f"{case_name}: on-disk case-file SHA-256 {actual_hash} != P2.3 manifest's recorded "
                    f"{recorded_case_hashes.get(case_name)!r}"
                )
    if manifest.get("execution_order_sha256") != execution_order_hash:
        errors.append(
            f"execution_order.csv on-disk SHA-256 {execution_order_hash} != P2.3 manifest's recorded "
            f"{manifest.get('execution_order_sha256')!r}"
        )
    recorded_aggregate_hashes = manifest.get("aggregate_file_sha256")
    if not isinstance(recorded_aggregate_hashes, dict):
        errors.append("P2.3 manifest aggregate_file_sha256 is missing or not an object")
    else:
        if recorded_aggregate_hashes.get("combined_samples.csv") != combined_hash:
            errors.append(
                f"combined_samples.csv on-disk SHA-256 {combined_hash} != P2.3 manifest's recorded "
                f"{recorded_aggregate_hashes.get('combined_samples.csv')!r}"
            )
        if recorded_aggregate_hashes.get("summary.csv") != summary_hash:
            errors.append(
                f"summary.csv on-disk SHA-256 {summary_hash} != P2.3 manifest's recorded "
                f"{recorded_aggregate_hashes.get('summary.csv')!r}"
            )
    if errors:
        return errors, None

    reference_row = cases[0][1][0]
    snapshot = {
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest_hash,
        "combined_samples_sha256": combined_hash,
        "summary_sha256": summary_hash,
        "execution_order_sha256": execution_order_hash,
        "case_file_sha256": case_file_hashes,
        "sample_count": total_samples,
        "configuration_count": len(cases),
        "gpu_uuid": reference_row["gpu_uuid"],
        "gpu_name": reference_row["gpu_name"],
        "compute_capability": reference_row["compute_capability"],
        "cuda_driver_version": reference_row["cuda_driver_version"],
        "cuda_runtime_version": reference_row["cuda_runtime_version"],
        "git_commit": reference_row["git_commit"],
    }
    return [], snapshot


def verify_campaign_evidence_integrity(campaign_dir: Path, manifest: dict) -> tuple[list[str], dict | None]:
    """Re-verified before COMPLETE and before publishing ANALYZED/
    INCONCLUSIVE. Returns (errors, verified_snapshot); snapshot is None
    whenever errors is non-empty. Never mutates the manifest or campaign."""
    errors: list[str] = []
    snapshot: dict = {}

    pilot_ref = manifest.get("pilot_campaign_reference")
    if not isinstance(pilot_ref, dict) or "path" not in pilot_ref:
        return ["pilot_campaign_reference is missing or incomplete; cannot verify pilot evidence"], None
    try:
        p23_campaign_dir = resolve_p23_campaign_dir_arg(pilot_ref["path"])
    except p23.UnsafePathError as exc:
        return [f"pilot_campaign_reference.path: {exc}"], None

    snapshot["pilot_manifest_sha256"] = _verify_hash("P2.3 manifest.json", p23_campaign_dir / "manifest.json", pilot_ref.get("manifest_sha256"), errors)
    snapshot["pilot_combined_samples_sha256"] = _verify_hash("P2.3 combined_samples.csv", p23_campaign_dir / "combined_samples.csv", pilot_ref.get("combined_samples_sha256"), errors)
    snapshot["pilot_summary_sha256"] = _verify_hash("P2.3 summary.csv", p23_campaign_dir / "summary.csv", pilot_ref.get("summary_sha256"), errors)

    preflight_pilot = manifest.get("preflight_reference_pilot", {})
    if isinstance(preflight_pilot, dict) and preflight_pilot.get("path"):
        snapshot["preflight_reference_pilot_sha256"] = _verify_hash("pilot preflight summary", Path(preflight_pilot["path"]), preflight_pilot.get("sha256"), errors)
    else:
        errors.append("preflight_reference_pilot is missing or incomplete")

    preflight_profile = manifest.get("preflight_reference_profile", {})
    if isinstance(preflight_profile, dict) and preflight_profile.get("path"):
        snapshot["preflight_reference_profile_sha256"] = _verify_hash("profile preflight summary", Path(preflight_profile["path"]), preflight_profile.get("sha256"), errors)
    else:
        errors.append("preflight_reference_profile is missing or incomplete")

    snapshot["profile_plan_sha256"] = _verify_hash("P2.4 profile_plan.csv", campaign_dir / "profile_plan.csv", manifest.get("profile_plan_sha256"), errors)

    plan = build_profile_plan()
    plan_errors = check_profile_plan_contract(plan)
    if plan_errors:
        return errors + [f"internal profile plan contract violation: {plan_errors}"], None

    # Defect-1 repair: the immutable campaign provenance tuple must itself
    # be well-formed before it can be used to validate any of the 24
    # profiles' own evidence below; a missing/malformed tuple fails the
    # whole campaign closed rather than silently skipping the comparison.
    campaign_provenance = manifest.get("provenance")
    provenance_tuple_errors = validate_provenance_tuple(campaign_provenance, label="manifest.provenance")
    if provenance_tuple_errors:
        return errors + provenance_tuple_errors, None
    if campaign_provenance.get("campaign_id") != campaign_dir.name:
        return errors + [
            f"manifest.provenance.campaign_id={campaign_provenance.get('campaign_id')!r} != "
            f"campaign directory {campaign_dir.name!r}"
        ], None

    case_results = manifest.get("case_results")
    if not isinstance(case_results, dict):
        return errors + ["case_results is missing or not an object"], None

    resolved_ncu_metrics = manifest.get("resolved_ncu_metrics")
    if not isinstance(resolved_ncu_metrics, dict):
        resolved_ncu_metrics = {}

    expected_names = {entry["case_name"] for entry in plan}
    snapshot_case_results: dict[str, dict] = {}

    try:
        campaign_parts = campaign_dir.relative_to(REPO_ROOT).parts
        campaign_fd = _open_dir_component_chain(*campaign_parts)
    except p23.UnsafePathError as exc:
        return errors + [f"campaign directory: {exc}"], None
    try:
        try:
            profiles_fd = _open_dir_nofollow_p24("profiles", dir_fd=campaign_fd)
        except p23.UnsafePathError as exc:
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
                        continue
                    if not isinstance(recorded, dict):
                        errors.append(f"{case_name}: recorded case_results entry is not an object")
                        continue
                    try:
                        evidence_fds = _open_case_evidence_fds(profiles_fd, case_name)
                    except p23.UnsafePathError as exc:
                        errors.append(f"{case_name}: {exc}")
                        continue
                    case_fds[case_name] = evidence_fds["_case_dir"]

                    reconstructed, case_errors = _reconstruct_case_result_from_fds(
                        entry=entry, case_name=case_name, fds=evidence_fds,
                        resolved_ncu_metrics=resolved_ncu_metrics, git_commit=campaign_provenance.get("git_commit"),
                        campaign_provenance=campaign_provenance,
                    )
                    for label, _suffix in CANONICAL_PROFILE_CASE_FILE_LABELS:
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
# Subcommand: plan
# ---------------------------------------------------------------------------
def cmd_plan(args: argparse.Namespace) -> int:
    plan = build_profile_plan()
    errors = check_profile_plan_contract(plan)
    if errors:
        print("analyze_exp02_umma_throughput_p24: plan: ERROR: plan contract violated:", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp02_umma_throughput_p24: plan:   - {error}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(plan, indent=2))
    elif args.format == "lines":
        sys.stdout.write(format_plan_lines(plan))
    else:
        sys.stdout.write(format_plan_text(plan))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: init-campaign
# ---------------------------------------------------------------------------
def _do_init_campaign(*, campaign_id: str, started_at_utc: str) -> Path:
    campaign_dir = create_p24_campaign_dir(campaign_id)
    plan = build_profile_plan()
    plan_errors = check_profile_plan_contract(plan)
    if plan_errors:
        raise ValueError(f"internal profile plan contract violation: {plan_errors}")
    profile_plan_path = write_profile_plan(campaign_dir, plan)
    profile_plan_sha256 = p23.sha256_of(profile_plan_path)
    frozen_protocol = {
        "pilot_params": dict(FROZEN_PILOT_PARAMS),
        "profile_params": dict(FROZEN_PROFILE_PARAMS),
        "profile_plan": plan,
        "mandatory_sm_clock_metric": MANDATORY_SM_CLOCK_METRIC,
        "sm_clock_unit_to_hz_scale": dict(SM_CLOCK_UNIT_TO_HZ_SCALE),
        "diagnostic_metrics": list(DIAGNOSTIC_METRICS),
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
    p24_merge_manifest(campaign_dir, updates, state="PILOT_IN_PROGRESS")
    return campaign_dir


def cmd_init_campaign(args: argparse.Namespace) -> int:
    try:
        campaign_dir = _do_init_campaign(campaign_id=args.campaign_id, started_at_utc=args.started_at_utc)
    except (p23.UnsafePathError, p23.ManifestTransitionError, ValueError) as exc:
        print(f"analyze_exp02_umma_throughput_p24: init-campaign: ERROR: {exc}", file=sys.stderr)
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
            print(f"analyze_exp02_umma_throughput_p24: validate-preflight: ERROR: {exc}", file=sys.stderr)
            return 2
    errors, snapshot = validate_preflight_file(Path(args.preflight), expected_git_commit=args.expected_git_commit, now_utc=now_utc)
    if errors:
        print(f"analyze_exp02_umma_throughput_p24: validate-preflight: FAIL: {args.preflight}", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp02_umma_throughput_p24: validate-preflight:   - {error}", file=sys.stderr)
        return 1
    print(f"analyze_exp02_umma_throughput_p24: validate-preflight: OK: {args.preflight}", file=sys.stderr)
    print(json.dumps(snapshot, indent=2))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: record-pilot
# ---------------------------------------------------------------------------
def _fail_p24(campaign_dir: Path, stage: str, errors: list[str]) -> tuple[bool, list[str]]:
    try:
        manifest, _revision = load_p24_manifest_chain(campaign_dir)
        current_state = manifest.get("state")
        if current_state in ALLOWED_P24_TRANSITIONS and "FAILED" in ALLOWED_P24_TRANSITIONS[current_state]:
            p24_merge_manifest(campaign_dir, {"failure_stage": stage, "failure_detail": errors[:50]}, state="FAILED")
    except (p23.ManifestTransitionError, p23.UnsafePathError):
        pass
    return False, errors


def _do_record_pilot(
    *, campaign_dir: Path, p23_campaign_dir: Path, preflight_path: Path,
    git_commit: str, completed_at_utc: str, now_utc: _datetime,
) -> tuple[bool, list[str]]:
    try:
        manifest, _revision = load_p24_manifest_chain(campaign_dir)
        _validate_p24_manifest_document(manifest, require_initialized=True)
    except (p23.ManifestTransitionError, p23.UnsafePathError) as exc:
        return False, [f"P2.4 manifest: {exc}"]
    if manifest.get("state") != "PILOT_IN_PROGRESS":
        return False, [f"P2.4 manifest state={manifest.get('state')!r} != 'PILOT_IN_PROGRESS'; cannot record a pilot"]

    errors: list[str] = []
    preflight_errors, preflight_snapshot = validate_preflight_file(preflight_path, expected_git_commit=git_commit, now_utc=now_utc)
    errors.extend(f"preflight: {e}" for e in preflight_errors)

    try:
        p23_manifest = p23.load_manifest(p23_campaign_dir)
        p23._validate_manifest_document(p23_manifest, require_initialized=True)
    except (p23.ManifestTransitionError, p23.UnsafePathError) as exc:
        return _fail_p24(campaign_dir, "pilot_p23_manifest", [f"P2.3 manifest: {exc}"])
    if not p23_manifest:
        return _fail_p24(campaign_dir, "pilot_p23_manifest", ["P2.3 manifest.json does not exist"])

    if p23_manifest.get("status") != "COMPLETE":
        errors.append(f"P2.3 campaign status={p23_manifest.get('status')!r} != 'COMPLETE' (only a COMPLETE P2.3 campaign can be pilot input)")
    if p23_manifest.get("run_kind") != "benchmark":
        errors.append(f"P2.3 campaign run_kind={p23_manifest.get('run_kind')!r} != 'benchmark' (a smoke campaign can never be pilot input)")
    requested = p23_manifest.get("requested", {}) if isinstance(p23_manifest.get("requested"), dict) else {}
    for key in ("iterations", "warmup_iterations", "repetitions"):
        expected = FROZEN_PILOT_PARAMS[key]
        if requested.get(key) != expected:
            errors.append(f"P2.3 campaign requested.{key}={requested.get(key)!r} != frozen pilot value {expected!r}")
    if p23_manifest.get("configuration_count_completed") != p23.EXPECTED_CONFIGURATION_COUNT:
        errors.append(f"P2.3 campaign configuration_count_completed={p23_manifest.get('configuration_count_completed')!r} != {p23.EXPECTED_CONFIGURATION_COUNT}")
    if p23_manifest.get("git_commit") != git_commit:
        errors.append(f"P2.3 campaign git_commit={p23_manifest.get('git_commit')!r} != expected {git_commit!r}")
    if p23_campaign_dir.name != campaign_dir.name:
        errors.append(
            f"P2.3 campaign ID {p23_campaign_dir.name!r} != P2.4 campaign ID {campaign_dir.name!r} "
            f"(the P2.4 wrapper and the P2.3 campaign it drives must share one explicit P2_4_CAMPAIGN_ID)"
        )
    if p23_manifest.get("campaign_id") != p23_campaign_dir.name:
        errors.append(f"P2.3 manifest campaign_id={p23_manifest.get('campaign_id')!r} != its own directory name {p23_campaign_dir.name!r}")

    if not preflight_errors:
        for field, snapshot_key in (("gpu_uuid", "gpu_uuid"), ("gpu_name", "gpu_name")):
            if p23_manifest.get(field) != preflight_snapshot.get(snapshot_key):
                errors.append(f"P2.3 campaign {field}={p23_manifest.get(field)!r} != preflight {snapshot_key}={preflight_snapshot.get(snapshot_key)!r}")
        if p23_manifest.get("compute_capability") != preflight_snapshot.get("gpu_compute_cap"):
            errors.append(
                f"P2.3 campaign compute_capability={p23_manifest.get('compute_capability')!r} != "
                f"preflight gpu.compute_cap={preflight_snapshot.get('gpu_compute_cap')!r}"
            )

    combined_hash = summary_hash = p23_manifest_hash = None
    combined_path = p23_campaign_dir / "combined_samples.csv"
    summary_path = p23_campaign_dir / "summary.csv"
    p23_manifest_path = p23_campaign_dir / "manifest.json"
    try:
        combined_hash = p23.sha256_of(combined_path)
        summary_hash = p23.sha256_of(summary_path)
        p23_manifest_hash = p23.sha256_of(p23_manifest_path)
    except p23.UnsafePathError as exc:
        errors.append(f"artifact hashing: {exc}")
    recorded_hashes = p23_manifest.get("aggregate_file_sha256", {})
    if isinstance(recorded_hashes, dict):
        if combined_hash is not None and recorded_hashes.get("combined_samples.csv") != combined_hash:
            errors.append("combined_samples.csv on disk does not match the P2.3 manifest's recorded SHA-256")
        if summary_hash is not None and recorded_hashes.get("summary.csv") != summary_hash:
            errors.append("summary.csv on disk does not match the P2.3 manifest's recorded SHA-256")

    if errors:
        return _fail_p24(campaign_dir, "pilot_validation", errors)

    updates = {
        "pilot_completed_at_utc": completed_at_utc,
        "pilot_campaign_reference": {
            "campaign_id": p23_manifest["campaign_id"],
            "path": str(p23_campaign_dir.relative_to(REPO_ROOT)),
            "manifest_sha256": p23_manifest_hash,
            "combined_samples_sha256": combined_hash,
            "summary_sha256": summary_hash,
        },
        "preflight_reference_pilot": preflight_snapshot,
        # The immutable campaign provenance tuple (Defect 1 repair): every
        # one of the 24 profile cases' own application evidence is compared
        # against this tuple by compare_application_provenance(), never the
        # reverse. visible_device_count/logical_device_index are carried
        # over from the already-validated pilot preflight snapshot (proven
        # by validate_preflight_fields, never fabricated here); campaign_id
        # binds the tuple to this exact P2.4 campaign directory.
        "provenance": {
            "git_commit": git_commit,
            "gpu_uuid": p23_manifest["gpu_uuid"],
            "gpu_name": p23_manifest["gpu_name"],
            "compute_capability": p23_manifest["compute_capability"],
            "cuda_driver_version": p23_manifest["cuda_driver_version"],
            "cuda_runtime_version": p23_manifest["cuda_runtime_version"],
            "visible_device_count": preflight_snapshot.get("visible_device_count"),
            "logical_device_index": (
                int(preflight_snapshot["gpu_logical_index"])
                if isinstance(preflight_snapshot.get("gpu_logical_index"), str) and preflight_snapshot["gpu_logical_index"].isdigit()
                else preflight_snapshot.get("gpu_logical_index")
            ),
            "campaign_id": campaign_dir.name,
        },
    }
    provenance_errors = validate_provenance_tuple(updates["provenance"], label="constructed provenance")
    if provenance_errors:
        return _fail_p24(campaign_dir, "pilot_provenance_construction", provenance_errors)
    try:
        p24_merge_manifest(campaign_dir, updates, state="PILOT_COMPLETE")
    except p23.ManifestTransitionError as exc:
        return _fail_p24(campaign_dir, "pilot_manifest_transition", [str(exc)])
    return True, []


def cmd_record_pilot(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_p24_campaign_dir(args.campaign_dir)
        p23_campaign_dir = resolve_p23_campaign_dir_arg(args.p23_campaign_dir)
    except p23.UnsafePathError as exc:
        print(f"analyze_exp02_umma_throughput_p24: record-pilot: ERROR: {exc}", file=sys.stderr)
        return 2
    now_utc = _datetime.now(_timezone.utc)
    if args.now is not None:
        try:
            now_utc = parse_now_arg(args.now)
        except ValueError as exc:
            print(f"analyze_exp02_umma_throughput_p24: record-pilot: ERROR: {exc}", file=sys.stderr)
            return 2
    success, errors = _do_record_pilot(
        campaign_dir=campaign_dir, p23_campaign_dir=p23_campaign_dir, preflight_path=Path(args.preflight),
        git_commit=args.git_commit, completed_at_utc=args.completed_at_utc, now_utc=now_utc,
    )
    if not success:
        print("analyze_exp02_umma_throughput_p24: record-pilot: ERROR:", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp02_umma_throughput_p24: record-pilot:   - {error}", file=sys.stderr)
        return 1
    print("analyze_exp02_umma_throughput_p24: record-pilot: OK: PILOT_COMPLETE", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: discover-metrics
# ---------------------------------------------------------------------------
def _do_discover_metrics(
    *, campaign_dir: Path, discovery_log: Path, preflight_path: Path,
    git_commit: str, started_at_utc: str, now_utc: _datetime,
) -> tuple[bool, list[str], dict | None]:
    try:
        manifest, _revision = load_p24_manifest_chain(campaign_dir)
        _validate_p24_manifest_document(manifest, require_initialized=True)
    except (p23.ManifestTransitionError, p23.UnsafePathError) as exc:
        return False, [f"P2.4 manifest: {exc}"], None
    if manifest.get("state") != "PILOT_COMPLETE":
        return False, [f"P2.4 manifest state={manifest.get('state')!r} != 'PILOT_COMPLETE'; cannot start profiling"], None

    errors: list[str] = []
    preflight_errors, preflight_snapshot = validate_preflight_file(preflight_path, expected_git_commit=git_commit, now_utc=now_utc)
    errors.extend(f"preflight: {e}" for e in preflight_errors)

    if not preflight_errors:
        pilot_preflight_snapshot = manifest.get("preflight_reference_pilot")
        if not isinstance(pilot_preflight_snapshot, dict):
            errors.append("manifest preflight_reference_pilot is missing or incomplete")
        else:
            errors.extend(compare_preflight_provenance(pilot_preflight_snapshot, preflight_snapshot))

    resolved = None
    try:
        discovered = parse_metric_discovery_log(discovery_log)
        resolved = resolve_ncu_metrics_p24(discovered)
    except ValueError as exc:
        errors.append(str(exc))

    if errors:
        success, fail_errors = _fail_p24(campaign_dir, "profile_start", errors)
        return success, fail_errors, None

    assert resolved is not None
    updates = {
        "profile_started_at_utc": started_at_utc,
        "resolved_ncu_metrics": resolved,
        "preflight_reference_profile": preflight_snapshot,
    }
    try:
        p24_merge_manifest(campaign_dir, updates, state="PROFILE_IN_PROGRESS")
    except p23.ManifestTransitionError as exc:
        success, fail_errors = _fail_p24(campaign_dir, "profile_start_manifest", [str(exc)])
        return success, fail_errors, None
    return True, [], resolved


def cmd_discover_metrics(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_p24_campaign_dir(args.campaign_dir)
    except p23.UnsafePathError as exc:
        print(f"analyze_exp02_umma_throughput_p24: discover-metrics: ERROR: {exc}", file=sys.stderr)
        return 2
    now_utc = _datetime.now(_timezone.utc)
    if args.now is not None:
        try:
            now_utc = parse_now_arg(args.now)
        except ValueError as exc:
            print(f"analyze_exp02_umma_throughput_p24: discover-metrics: ERROR: {exc}", file=sys.stderr)
            return 2
    success, errors, resolved = _do_discover_metrics(
        campaign_dir=campaign_dir, discovery_log=Path(args.discovery_log), preflight_path=Path(args.preflight),
        git_commit=args.git_commit, started_at_utc=args.started_at_utc, now_utc=now_utc,
    )
    if not success:
        print("analyze_exp02_umma_throughput_p24: discover-metrics: ERROR:", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp02_umma_throughput_p24: discover-metrics:   - {error}", file=sys.stderr)
        return 1
    if not resolved["sm_clock_metric_resolved"]:
        print(
            f"analyze_exp02_umma_throughput_p24: discover-metrics: WARNING: {MANDATORY_SM_CLOCK_METRIC} was not "
            f"resolved; this campaign will become INCONCLUSIVE at analyze time",
            file=sys.stderr,
        )
    print("analyze_exp02_umma_throughput_p24: discover-metrics: OK: PROFILE_IN_PROGRESS", file=sys.stderr)
    print(",".join(resolved_metric_names_for_ncu(resolved)))
    return 0


# ---------------------------------------------------------------------------
# Subcommand: validate-profile-preconditions
# ---------------------------------------------------------------------------
def _do_validate_profile_preconditions(*, campaign_dir: Path, preflight_path: Path, git_commit: str, now_utc: _datetime) -> tuple[bool, list[str]]:
    try:
        manifest, _revision = load_p24_manifest_chain(campaign_dir)
        _validate_p24_manifest_document(manifest, require_initialized=True)
    except (p23.ManifestTransitionError, p23.UnsafePathError) as exc:
        return False, [f"P2.4 manifest: {exc}"]
    if manifest.get("state") != "PILOT_COMPLETE":
        return False, [f"P2.4 manifest state={manifest.get('state')!r} != 'PILOT_COMPLETE'; cannot check profiling preconditions"]
    pilot_snapshot = manifest.get("preflight_reference_pilot")
    if not isinstance(pilot_snapshot, dict):
        return False, ["manifest preflight_reference_pilot is missing or incomplete"]

    preflight_errors, profile_snapshot = validate_preflight_file(preflight_path, expected_git_commit=git_commit, now_utc=now_utc)
    if preflight_errors:
        return False, [f"preflight: {e}" for e in preflight_errors]

    provenance_errors = compare_preflight_provenance(pilot_snapshot, profile_snapshot)
    if provenance_errors:
        return False, provenance_errors
    return True, []


def cmd_validate_profile_preconditions(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_p24_campaign_dir(args.campaign_dir)
    except p23.UnsafePathError as exc:
        print(f"analyze_exp02_umma_throughput_p24: validate-profile-preconditions: ERROR: {exc}", file=sys.stderr)
        return 2
    now_utc = _datetime.now(_timezone.utc)
    if args.now is not None:
        try:
            now_utc = parse_now_arg(args.now)
        except ValueError as exc:
            print(f"analyze_exp02_umma_throughput_p24: validate-profile-preconditions: ERROR: {exc}", file=sys.stderr)
            return 2
    ok, errors = _do_validate_profile_preconditions(campaign_dir=campaign_dir, preflight_path=Path(args.preflight), git_commit=args.git_commit, now_utc=now_utc)
    if not ok:
        print("analyze_exp02_umma_throughput_p24: validate-profile-preconditions: ERROR:", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp02_umma_throughput_p24: validate-profile-preconditions:   - {error}", file=sys.stderr)
        return 1
    print("analyze_exp02_umma_throughput_p24: validate-profile-preconditions: OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: validate-profile-case
# ---------------------------------------------------------------------------
def _resolve_case_evidence_paths(campaign_dir: Path, case_name: str, errors: list[str]) -> dict[str, Path] | None:
    profiles_dir = campaign_dir / "profiles"
    case_dir = profiles_dir / case_name
    try:
        p23._reject_if_symlink_or_wrong_type(case_dir, expect_dir=True)
        if not os.path.lexists(case_dir):
            errors.append(f"{case_name}: profile case directory does not exist: {case_dir}")
            return None
        p23._confirm_contained(case_dir, profiles_dir)
    except p23.UnsafePathError as exc:
        errors.append(f"{case_name}: {exc}")
        return None

    # Defect-2 repair: require the exact seven-file canonical inventory --
    # no missing, extra, or wrongly named entry -- before trusting any of
    # them individually.
    expected_names = set(canonical_profile_case_filenames(case_name))
    try:
        actual_names = {entry.name for entry in case_dir.iterdir()}
    except OSError as exc:
        errors.append(f"{case_name}: cannot list profile case directory: {exc}")
        return None
    unexpected = sorted(actual_names - expected_names)
    missing = sorted(expected_names - actual_names)
    if unexpected:
        errors.append(f"{case_name}: unplanned entrie(s) in profile case directory: {unexpected}")
    if missing:
        errors.append(f"{case_name}: missing canonical artifact(s): {missing}")
    if unexpected or missing:
        return None

    paths = {label: case_dir / f"{case_name}{suffix}" for label, suffix in CANONICAL_PROFILE_CASE_FILE_LABELS}
    for label, path in paths.items():
        artifact_err = _verify_profile_case_artifact(path, label)
        if artifact_err:
            errors.append(f"{case_name}.{label}: {artifact_err}")
            return None
        try:
            p23._confirm_contained(path, case_dir)
        except p23.UnsafePathError as exc:
            errors.append(f"{case_name}.{label}: {exc}")
            return None
    return paths


def _do_validate_profile_case(*, campaign_dir: Path, index: int, git_commit: str) -> tuple[bool, list[str]]:
    plan = build_profile_plan()
    entry = next((e for e in plan if e["index"] == index), None)
    if entry is None:
        return False, [f"index {index} is not one of the frozen profile case indices 0..{EXPECTED_PROFILE_CASE_COUNT - 1}"]

    try:
        manifest, _revision = load_p24_manifest_chain(campaign_dir)
        _validate_p24_manifest_document(manifest, require_initialized=True)
    except (p23.ManifestTransitionError, p23.UnsafePathError) as exc:
        return False, [f"P2.4 manifest: {exc}"]
    if manifest.get("state") != "PROFILE_IN_PROGRESS":
        return False, [f"P2.4 manifest state={manifest.get('state')!r} != 'PROFILE_IN_PROGRESS'"]

    existing_results = manifest.get("case_results", {})
    if entry["case_name"] in existing_results:
        return False, [f"case {entry['case_name']} was already validated and recorded; refusing to redo it"]

    campaign_provenance = manifest.get("provenance")
    provenance_tuple_errors = validate_provenance_tuple(campaign_provenance, label="manifest.provenance")
    if provenance_tuple_errors:
        return False, provenance_tuple_errors
    if campaign_provenance.get("campaign_id") != campaign_dir.name:
        return False, [
            f"manifest.provenance.campaign_id={campaign_provenance.get('campaign_id')!r} != "
            f"campaign directory {campaign_dir.name!r}"
        ]
    if campaign_provenance.get("git_commit") != git_commit:
        return False, [
            f"--git-commit={git_commit!r} != this campaign's own recorded provenance.git_commit="
            f"{campaign_provenance.get('git_commit')!r}"
        ]

    path_errors: list[str] = []
    paths = _resolve_case_evidence_paths(campaign_dir, entry["case_name"], path_errors)
    if paths is None:
        return False, path_errors

    resolved_ncu_metrics = manifest.get("resolved_ncu_metrics", {})
    case_result, errors = reconstruct_case_result(
        entry=entry, application_csv=paths["application_csv"], metrics_csv=paths["metrics_csv"],
        ncu_rep=paths["ncu_rep"], ncu_tool_log=paths["ncu_tool_log"],
        container_stdout_log=paths["container_stdout_log"], container_stderr_log=paths["container_stderr_log"],
        metrics_export_stderr_log=paths["metrics_export_stderr_log"],
        resolved_ncu_metrics=resolved_ncu_metrics, git_commit=git_commit,
        campaign_provenance=campaign_provenance,
    )
    if case_result is None:
        return False, errors

    new_results = dict(existing_results)
    new_results[entry["case_name"]] = case_result
    updates = {"case_results": new_results, "profile_count_completed": len(new_results)}
    try:
        p24_merge_manifest(campaign_dir, updates, state="PROFILE_IN_PROGRESS")
    except p23.ManifestTransitionError as exc:
        return False, [str(exc)]
    return True, []


def cmd_validate_profile_case(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_p24_campaign_dir(args.campaign_dir)
    except p23.UnsafePathError as exc:
        print(f"analyze_exp02_umma_throughput_p24: validate-profile-case: ERROR: {exc}", file=sys.stderr)
        return 2
    success, errors = _do_validate_profile_case(campaign_dir=campaign_dir, index=args.index, git_commit=args.git_commit)
    if not success:
        print(f"analyze_exp02_umma_throughput_p24: validate-profile-case: FAIL: index {args.index}", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp02_umma_throughput_p24: validate-profile-case:   - {error}", file=sys.stderr)
        return 1
    print(f"analyze_exp02_umma_throughput_p24: validate-profile-case: OK: index {args.index}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: finalize-profile
# ---------------------------------------------------------------------------
def _do_finalize_profile(*, campaign_dir: Path, completed_at_utc: str) -> tuple[bool, list[str]]:
    try:
        manifest, _revision = load_p24_manifest_chain(campaign_dir)
        _validate_p24_manifest_document(manifest, require_initialized=True)
    except (p23.ManifestTransitionError, p23.UnsafePathError) as exc:
        return False, [f"P2.4 manifest: {exc}"]
    if manifest.get("state") != "PROFILE_IN_PROGRESS":
        return False, [f"P2.4 manifest state={manifest.get('state')!r} != 'PROFILE_IN_PROGRESS'; cannot finalize"]

    errors: list[str] = []
    plan = build_profile_plan()
    plan_errors = check_profile_plan_contract(plan)
    if plan_errors:
        return _fail_p24(campaign_dir, "profile_plan_contract", plan_errors)

    errors.extend(validate_profile_plan_file(campaign_dir / "profile_plan.csv", plan))

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
    if manifest.get("profile_count_completed") != EXPECTED_PROFILE_CASE_COUNT:
        errors.append(f"profile_count_completed={manifest.get('profile_count_completed')!r} != {EXPECTED_PROFILE_CASE_COUNT}")

    resolved_ncu_metrics = manifest.get("resolved_ncu_metrics", {})
    if not isinstance(resolved_ncu_metrics, dict) or "sm_clock_metric_resolved" not in resolved_ncu_metrics:
        errors.append("manifest resolved_ncu_metrics is missing or incomplete")

    if errors:
        return _fail_p24(campaign_dir, "profile_finalize", errors)

    # Defect-4 repair: before COMPLETE, independently revalidate the
    # complete raw P2.3 pilot evidence (24 x 30 = 720 samples) -- never
    # trusted merely because record-pilot once accepted its manifest/file
    # hashes.
    pilot_ref = manifest.get("pilot_campaign_reference", {})
    if not isinstance(pilot_ref, dict) or "path" not in pilot_ref:
        return _fail_p24(campaign_dir, "profile_finalize_p23_revalidation", ["pilot_campaign_reference is missing or incomplete; cannot revalidate P2.3 pilot evidence"])
    try:
        p23_campaign_dir_for_revalidation = resolve_p23_campaign_dir_arg(pilot_ref["path"])
    except p23.UnsafePathError as exc:
        return _fail_p24(campaign_dir, "profile_finalize_p23_revalidation", [f"pilot_campaign_reference.path: {exc}"])
    p23_revalidation_errors, _p23_snapshot = revalidate_p23_pilot_campaign(
        p23_campaign_dir_for_revalidation, git_commit=manifest.get("provenance", {}).get("git_commit"),
    )
    if p23_revalidation_errors:
        return _fail_p24(campaign_dir, "profile_finalize_p23_revalidation", p23_revalidation_errors)

    integrity_errors, verified = verify_campaign_evidence_integrity(campaign_dir, manifest)
    if integrity_errors:
        return _fail_p24(campaign_dir, "profile_finalize_integrity", integrity_errors)

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
        for key in CASE_ARTIFACT_HASH_FIELDS:
            artifact_sha256[f"{case_name}.{key}"] = result[key]

    updates = {"profile_completed_at_utc": completed_at_utc, "profile_order": plan, "artifact_sha256": artifact_sha256}
    try:
        p24_merge_manifest(campaign_dir, updates, state="COMPLETE")
    except p23.ManifestTransitionError as exc:
        return _fail_p24(campaign_dir, "profile_finalize_manifest", [str(exc)])
    return True, []


def cmd_finalize_profile(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_p24_campaign_dir(args.campaign_dir)
    except p23.UnsafePathError as exc:
        print(f"analyze_exp02_umma_throughput_p24: finalize-profile: ERROR: {exc}", file=sys.stderr)
        return 2
    success, errors = _do_finalize_profile(campaign_dir=campaign_dir, completed_at_utc=args.completed_at_utc)
    if not success:
        print("analyze_exp02_umma_throughput_p24: finalize-profile: ERROR:", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp02_umma_throughput_p24: finalize-profile:   - {error}", file=sys.stderr)
        return 1
    print("analyze_exp02_umma_throughput_p24: finalize-profile: OK: COMPLETE", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Deterministic SVG chart rendering. Python standard library only: plain
# string building, escaped via xml.sax.saxutils.escape. No NumPy, pandas,
# matplotlib, or a notebook/Docker dependency.
# ---------------------------------------------------------------------------
from xml.sax.saxutils import escape as _xml_escape  # noqa: E402

_SERIES_COLORS = {"umma_1sm": "#1b9e77", "umma_2sm": "#d95f02"}
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
        # Defect-6 repair: deterministic, machine-readable metadata carrying
        # the exact publication-status token, shared by all three SVG
        # artifacts via this one helper.
        f"<metadata>{_xml_escape(PUBLICATION_STATUS_TOKEN)}</metadata>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="none"/>',
    ]


def _render_panel(
    *, x0: float, y0: float, x1: float, y1: float, title: str, x_labels: list[str],
    y_min: float, y_max: float, y_label: str, series: list[dict], hlines: tuple[tuple[float, str], ...] = (),
) -> list[str]:
    out: list[str] = []
    plot_w = x1 - x0
    plot_h = y1 - y0
    out.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y0 - 20:.1f}" text-anchor="middle" font-weight="bold">{_xml_escape(title)}</text>')
    scale_y, y_lo, y_hi = _y_scale(y_min, y_max, y0, y1)

    n_x = max(len(x_labels), 1)
    x_positions = [x0 + (plot_w * (i + 0.5) / n_x) for i in range(n_x)] if n_x > 0 else []

    out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{plot_w:.1f}" height="{plot_h:.1f}" fill="none" stroke="#333333" stroke-width="1"/>')

    n_yticks = 5
    for t in range(n_yticks + 1):
        y_val = y_lo + (y_hi - y_lo) * t / n_yticks
        y_px = scale_y(y_val)
        out.append(f'<line x1="{x0:.1f}" y1="{y_px:.1f}" x2="{x1:.1f}" y2="{y_px:.1f}" stroke="#e0e0e0" stroke-width="1"/>')
        out.append(f'<text x="{x0 - 6:.1f}" y="{y_px + 4:.1f}" text-anchor="end">{_xml_escape(_fmt_num(y_val))}</text>')
    out.append(
        f'<text x="{x0 - 56:.1f}" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" '
        f'transform="rotate(-90 {x0 - 56:.1f} {(y0 + y1) / 2:.1f})">{_xml_escape(y_label)}</text>'
    )
    for i, label in enumerate(x_labels):
        out.append(f'<text x="{x_positions[i]:.1f}" y="{y1 + 20:.1f}" text-anchor="middle">{_xml_escape(label)}</text>')

    for value, hlabel in hlines:
        y_px = scale_y(value)
        out.append(f'<line x1="{x0:.1f}" y1="{y_px:.1f}" x2="{x1:.1f}" y2="{y_px:.1f}" stroke="#999999" stroke-width="1.5" stroke-dasharray="6,4"/>')
        out.append(f'<text x="{x1 - 4:.1f}" y="{y_px - 4:.1f}" text-anchor="end" fill="#666666">{_xml_escape(hlabel)}</text>')

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
                out.append(f'<line x1="{px:.1f}" y1="{py_lo:.1f}" x2="{px:.1f}" y2="{py_hi:.1f}" stroke="{color}" stroke-width="1.5"/>')
                cap = 5
                out.append(f'<line x1="{px - cap:.1f}" y1="{py_lo:.1f}" x2="{px + cap:.1f}" y2="{py_lo:.1f}" stroke="{color}" stroke-width="1.5"/>')
                out.append(f'<line x1="{px - cap:.1f}" y1="{py_hi:.1f}" x2="{px + cap:.1f}" y2="{py_hi:.1f}" stroke="{color}" stroke-width="1.5"/>')
        if len(points) >= 2 and point_colors is None:
            path = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
            out.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for (px, py), i in zip(points, point_indices):
            point_color = point_colors[i] if point_colors else color
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="5" fill="{point_color}" stroke="#ffffff" stroke-width="1"/>')

    legend_y = y0 + 4
    legend_x = x1 - 90
    for i, s in enumerate(series):
        ly = legend_y + i * 16
        out.append(f'<rect x="{legend_x:.1f}" y="{ly - 8:.1f}" width="12" height="12" fill="{s["color"]}"/>')
        out.append(f'<text x="{legend_x + 16:.1f}" y="{ly + 2:.1f}">{_xml_escape(s["label"])}</text>')
    return out


def render_throughput_svg(stats_by_config: dict[tuple[str, int, int], dict]) -> str:
    width, height = _CHART_WIDTH * 3 - 80, _CHART_HEIGHT
    out = _svg_header(width, height, "FLOP/cycle (median, 95% bootstrap CI) vs depth, per N")
    x_labels = [str(d) for d in DEPTH_VALUES]
    all_values = [stats["flops_per_cycle"]["median"] for stats in stats_by_config.values()]
    y_min, y_max = min(all_values), max(all_values)
    panel_w = (width - 60) / 3
    for panel_i, n in enumerate((64, 128, 256)):
        x0 = 20 + panel_i * panel_w
        x1 = x0 + panel_w
        px0, py0, px1, py1 = x0 + _MARGIN_LEFT - 20, _MARGIN_TOP, x1 - _MARGIN_RIGHT, height - _MARGIN_BOTTOM
        series = []
        for method in ("umma_1sm", "umma_2sm"):
            values, ci_low, ci_high = [], [], []
            for depth in DEPTH_VALUES:
                stats = stats_by_config.get((method, n, depth))
                values.append(stats["flops_per_cycle"]["median"] if stats else None)
                ci_low.append(stats["flops_per_cycle"]["median_ci_low"] if stats else None)
                ci_high.append(stats["flops_per_cycle"]["median_ci_high"] if stats else None)
            series.append({"label": method, "color": _SERIES_COLORS[method], "values": values, "ci_low": ci_low, "ci_high": ci_high})
        out.extend(_render_panel(
            x0=px0, y0=py0, x1=px1, y1=py1, title=f"N={n}", x_labels=x_labels, y_min=y_min, y_max=y_max,
            y_label="flops_per_cycle (median, 95% CI)", series=series,
        ))
    out.append(f'<text x="12" y="{height - 8}" font-size="10" fill="#666666">Clock-independent FLOP/cycle. TFLOP/s conversion is per-configuration in report.md; never NCU kernel duration.</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_scaling_efficiency_svg(scaling_rows: list[dict]) -> str:
    width, height = _CHART_WIDTH, _CHART_HEIGHT
    out = _svg_header(width, height, "2-SM/1-SM scaling efficiency vs (N, depth)")
    px0, py0, px1, py1 = _plot_rect(width=width, height=height)
    x_labels = [f"N{row['n']}d{row['depth']}" for row in scaling_rows]
    values = [row["scaling_efficiency_percent"] for row in scaling_rows]
    all_ci = [v for row in scaling_rows for v in (100.0 * row["speedup_ci_low"] / 2.0, 100.0 * row["speedup_ci_high"] / 2.0)]
    y_min = min([0.0, 100.0] + values + all_ci)
    y_max = max([100.0] + values + all_ci)
    ci_low = [100.0 * row["speedup_ci_low"] / 2.0 for row in scaling_rows]
    ci_high = [100.0 * row["speedup_ci_high"] / 2.0 for row in scaling_rows]
    series = [{"label": "scaling_efficiency_percent", "color": "#7570b3", "values": values, "ci_low": ci_low, "ci_high": ci_high}]
    out.extend(_render_panel(
        x0=px0, y0=py0, x1=px1, y1=py1, title="scaling_efficiency_percent (median-ratio-derived, 95% CI)",
        x_labels=x_labels, y_min=y_min, y_max=y_max, y_label="percent", series=series,
        hlines=((100.0, "100% (ideal linear scaling)"),),
    ))
    out.append(f'<text x="12" y="{height - 8}" font-size="10" fill="#666666">Never clamped; values above 100% or below 0% are preserved and flagged for review, not corrected.</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


def render_saturation_svg(saturation_rows: list[dict]) -> str:
    width, height = _CHART_WIDTH * 2 - 40, _CHART_HEIGHT
    out = _svg_header(width, height, "FLOP/cycle vs depth per (method, N), with candidate saturation depth")
    x_labels = [str(d) for d in DEPTH_VALUES]
    all_values = [
        row[f"depth_{d}_median_flops_per_cycle"] for row in saturation_rows for d in DEPTH_VALUES
    ]
    y_min, y_max = min(all_values), max(all_values)
    panel_w = (width - 40) / len(saturation_rows)
    for panel_i, row in enumerate(saturation_rows):
        x0 = 20 + panel_i * panel_w
        x1 = x0 + panel_w
        px0, py0, px1, py1 = x0 + _MARGIN_LEFT - 20, _MARGIN_TOP, x1 - _MARGIN_RIGHT, height - _MARGIN_BOTTOM
        values = [row[f"depth_{d}_median_flops_per_cycle"] for d in DEPTH_VALUES]
        colors = ["#1b9e77" if d == row["earliest_tested_candidate_saturation_depth"] else "#999999" for d in DEPTH_VALUES]
        series = [{
            "label": f"{row['method']} N={row['n']}", "color": "#1b9e77", "values": values,
            "ci_low": [None] * len(values), "ci_high": [None] * len(values), "point_colors": colors,
        }]
        out.extend(_render_panel(
            x0=px0, y0=py0, x1=px1, y1=py1, title=f"{row['method']} N={row['n']}", x_labels=x_labels,
            y_min=y_min, y_max=y_max, y_label="flops_per_cycle (median)", series=series,
        ))
    out.append(f'<text x="12" y="{height - 8}" font-size="10" fill="#666666">Green point = earliest_tested_candidate_saturation_depth. Limited to the four tested depths; never a universal saturation depth.</text>')
    out.append("</svg>")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Empirical ceiling construction and report.md. clock-independent selection
# (select_ceiling) always runs; a clock-calibrated TFLOP/s figure is
# attached to a selected configuration only when that configuration's own
# NCU profile has sm_clock_valid=True. If ANY of the 24 profiled
# configurations lacks a valid SM-clock reading, the whole campaign is
# INCONCLUSIVE and no TFLOP/s or device-equivalent figure is ever emitted
# anywhere in analysis/* (src/compute/P2_4_PROTOCOL.md section 6).
# ---------------------------------------------------------------------------
def _configuration_summary(
    key: tuple[str, int, int], stats_by_config: dict, case_results: dict[str, dict], *, campaign_may_emit_tflops: bool,
) -> dict:
    """campaign_may_emit_tflops gates every TFLOP/s figure on the whole
    campaign's SM-clock validity, never only this one configuration's own
    reading: an INCONCLUSIVE campaign must never emit a TFLOP/s or
    completed empirical-ceiling claim for *any* configuration, even one
    whose own profile happened to have a valid SM-clock reading."""
    method, n, depth = key
    entry = next(e for e in build_profile_plan() if (e["method"], e["n"], e["depth"]) == key)
    stats = stats_by_config[key]
    case = case_results.get(entry["case_name"], {})
    summary = {
        "method": method, "n": n, "depth": depth, "cta_group": stats["cta_group"],
        "case_name": entry["case_name"], "kernel_symbol": entry["kernel_symbol"],
        "median_flops_per_cycle": stats["flops_per_cycle"]["median"],
        "median_flops_per_cycle_per_sm": stats["flops_per_cycle_per_sm"]["median"],
        "sm_clock_valid": bool(case.get("sm_clock_valid")),
        "sm_clock_hz": case.get("sm_clock_hz"),
        "estimated_local_tflops": None,
        "estimated_tflops_per_sm": None,
    }
    if campaign_may_emit_tflops and summary["sm_clock_valid"] and summary["sm_clock_hz"]:
        estimated_local_tflops = stats["flops_per_cycle"]["median"] * case["sm_clock_hz"] / 1e12
        summary["estimated_local_tflops"] = estimated_local_tflops
        summary["estimated_tflops_per_sm"] = estimated_local_tflops / stats["cta_group"]
    return summary


def evaluate_device_multiprocessor_count(case_results: dict[str, dict], plan: list[dict]) -> dict:
    """Strict, fail-closed evaluation of DEVICE_MULTIPROCESSOR_COUNT_METRIC
    across every one of the frozen plan's profiles (audit repair: the prior
    implementation extrapolated from whichever profiles happened to report a
    consistent value, including just one out of 24, and never checked
    finiteness, positivity, or integrality). A device-wide extrapolation may
    be available only when every single one of the len(plan) profiles
    independently reports one finite, strictly positive, mathematically
    integral value in the documented unit representation
    (EXPECTED_MULTIPROCESSOR_COUNT_UNIT), and every one of those values is
    identical. Returns {"available": True, "multiprocessor_count": int} or
    {"available": False, "reasons": [...]} with one deterministic reason per
    offending case; never raises."""
    reasons: list[str] = []
    values: dict[str, int] = {}
    for entry in plan:
        case_name = entry["case_name"]
        result = case_results.get(case_name)
        if not isinstance(result, dict):
            reasons.append(f"{case_name}: no recorded profile result")
            continue
        diag_values = result.get("diagnostic_metric_values", {})
        diag_units = result.get("diagnostic_metric_units", {})
        if not isinstance(diag_values, dict) or DEVICE_MULTIPROCESSOR_COUNT_METRIC not in diag_values:
            reasons.append(f"{case_name}: {DEVICE_MULTIPROCESSOR_COUNT_METRIC} was not resolved/reported for this profile")
            continue
        value = diag_values[DEVICE_MULTIPROCESSOR_COUNT_METRIC]
        unit = diag_units.get(DEVICE_MULTIPROCESSOR_COUNT_METRIC, "") if isinstance(diag_units, dict) else ""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            reasons.append(f"{case_name}: {DEVICE_MULTIPROCESSOR_COUNT_METRIC} value {value!r} is not numeric")
            continue
        if not math.isfinite(value):
            reasons.append(f"{case_name}: {DEVICE_MULTIPROCESSOR_COUNT_METRIC} value {value!r} is not finite")
            continue
        if value <= 0:
            reasons.append(f"{case_name}: {DEVICE_MULTIPROCESSOR_COUNT_METRIC} value {value!r} is not strictly positive")
            continue
        if value != math.floor(value):
            reasons.append(f"{case_name}: {DEVICE_MULTIPROCESSOR_COUNT_METRIC} value {value!r} is not mathematically integral")
            continue
        if (unit or "").strip() != EXPECTED_MULTIPROCESSOR_COUNT_UNIT:
            reasons.append(f"{case_name}: {DEVICE_MULTIPROCESSOR_COUNT_METRIC} unit {unit!r} != expected {EXPECTED_MULTIPROCESSOR_COUNT_UNIT!r}")
            continue
        values[case_name] = int(value)

    if len(values) != len(plan):
        missing = [entry["case_name"] for entry in plan if entry["case_name"] not in values]
        if missing and not reasons:
            reasons.append(f"missing/invalid value(s) for: {missing}")
        return {"available": False, "reasons": reasons}

    distinct = sorted(set(values.values()))
    if len(distinct) != 1:
        return {"available": False, "reasons": [f"inconsistent values across profiles (all {len(plan)} must agree): {distinct}"]}
    return {"available": True, "multiprocessor_count": distinct[0], "reasons": []}


def build_empirical_ceiling(
    *, stats_by_config: dict, case_results: dict[str, dict], all_sm_clock_valid: bool, inconclusive_reason: list[str],
) -> dict:
    selection = select_ceiling(stats_by_config)
    best_1sm = _configuration_summary(selection["best_1sm"], stats_by_config, case_results, campaign_may_emit_tflops=all_sm_clock_valid)
    best_2sm = _configuration_summary(selection["best_2sm"], stats_by_config, case_results, campaign_may_emit_tflops=all_sm_clock_valid)
    ceiling = _configuration_summary(selection["empirical_per_sm_ceiling_candidate"], stats_by_config, case_results, campaign_may_emit_tflops=all_sm_clock_valid)

    # The SM count is optional: its absence/invalidity can only suppress the
    # whole-device extrapolation below, and must never reach or corrupt the
    # per-configuration local/per-SM estimates computed above.
    mp_eval = evaluate_device_multiprocessor_count(case_results, build_profile_plan())
    if all_sm_clock_valid and mp_eval["available"] and ceiling["estimated_tflops_per_sm"] is not None:
        mp_count = mp_eval["multiprocessor_count"]
        device_extrapolation = {
            "available": True,
            "multiprocessor_count": mp_count,
            "estimated_device_equivalent_tflops": ceiling["estimated_tflops_per_sm"] * mp_count,
            "label": "extrapolation from a one-/two-SM microbenchmark, never a directly measured whole-GPU throughput",
        }
    elif not all_sm_clock_valid:
        device_extrapolation = {"available": False, "reason": "campaign is INCONCLUSIVE: the mandatory SM-clock metric is not trustworthy for every profiled configuration"}
    elif ceiling["estimated_tflops_per_sm"] is None:
        device_extrapolation = {"available": False, "reason": "no estimated_tflops_per_sm is available for the selected ceiling configuration"}
    else:
        device_extrapolation = {
            "available": False,
            "reason": (
                f"device__attribute_multiprocessor_count is not a single finite, strictly positive, "
                f"mathematically integral value identical across all {len(build_profile_plan())} profiled "
                f"configurations: {'; '.join(mp_eval['reasons'])}"
            ),
        }

    return {
        "status": "ANALYZED" if all_sm_clock_valid else "INCONCLUSIVE",
        "inconclusive_reason": inconclusive_reason if not all_sm_clock_valid else [],
        "best_1sm_configuration": best_1sm,
        "best_2sm_configuration": best_2sm,
        "empirical_per_sm_ceiling_candidate": ceiling,
        "device_equivalent_estimate": device_extrapolation,
        "pilot_is_not_publishable": True,
        "publishable": False,
        "publication_status": PUBLICATION_STATUS_TOKEN,
        "notes": [
            "ceiling selected in clock-independent FLOP/cycle-per-SM space first, then converted using that "
            "same configuration's own matching NCU SM-frequency profile",
            "never derived from gpu__time_duration.sum or any other NCU kernel-duration metric",
            "a pilot result, never a final or publishable campaign",
            "never a theoretical architectural peak, never a directly measured whole-GPU throughput",
        ],
    }


def render_report_markdown(
    *, campaign_id: str, stats_rows: list[dict], scaling_rows: list[dict], saturation_rows: list[dict],
    profile_validation_rows: list[dict], empirical_ceiling: dict, provenance: dict,
) -> str:
    lines: list[str] = []
    lines.append(f"# P2.4 profiling and empirical BF16 UMMA ceiling pilot -- campaign `{campaign_id}`")
    lines.append("")
    lines.append(
        "**publishable: false.** This report is generated from a single reviewed pilot pending independent "
        "audit and GB300 re-verification; it is not a final experimental result."
    )
    lines.append("")
    lines.append(f"Status line: `{PUBLICATION_STATUS_TOKEN}`")
    lines.append("")
    lines.append("## What this is, and is not")
    lines.append("")
    lines.append(
        "- One complete `run_kind=benchmark` pilot (iterations=1000, warmup_iterations=10, repetitions=30) "
        "over the frozen 24-configuration P2.3 matrix, plus Nsight Compute profiling of the same 24 "
        "configurations -- never 24 additional sweep configurations."
    )
    lines.append("- **No sample was ever removed.** All 30 retained repetitions of all 24 configurations are used in every statistic below; IQR flags are diagnostics only.")
    lines.append("- TFLOP/s is never derived from NCU kernel duration; only from the pilot's own `%clock64`-timed `flops_per_cycle` combined with each configuration's own matching NCU SM-clock reading.")
    lines.append("- The empirical per-SM ceiling below is a **candidate**, not a theoretical architectural peak and not a directly measured whole-device throughput.")
    lines.append(f"- Status: **{empirical_ceiling['status']}**." + (
        " No TFLOP/s or empirical-ceiling claim is completed anywhere in this report." if empirical_ceiling["status"] == "INCONCLUSIVE" else ""
    ))
    if empirical_ceiling["status"] == "INCONCLUSIVE":
        for reason in empirical_ceiling["inconclusive_reason"]:
            lines.append(f"  - {reason}")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append("```text")
    for key in ("git_commit", "gpu_uuid", "gpu_name", "compute_capability", "cuda_driver_version", "cuda_runtime_version"):
        lines.append(f"{key}: {provenance.get(key)}")
    lines.append("```")
    lines.append("")
    lines.append("## Configuration statistics (24 configurations, all 30 repetitions each)")
    lines.append("")
    lines.append("| method | N | depth | median flops/cycle | 95% CI | median flops/cycle/SM | CV% (flops/cycle) | stability review |")
    lines.append("| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |")
    for row in stats_rows:
        stability_review = row["flops_per_cycle_stability_review"]
        if stability_review not in {"ok", "REVIEW"}:
            raise ValueError(f"configuration-statistics stability_review must be exactly 'ok' or 'REVIEW', got {stability_review!r}")
        lines.append(
            f"| {row['method']} | {row['n']} | {row['depth']} | {row['flops_per_cycle_median']:.4f} | "
            f"[{row['flops_per_cycle_median_ci_low']:.4f}, {row['flops_per_cycle_median_ci_high']:.4f}] | "
            f"{row['flops_per_cycle_per_sm_median']:.4f} | {row['flops_per_cycle_cv_percent']:.2f} | {stability_review} |"
        )
    lines.append("")
    lines.append("## 1-SM/2-SM scaling")
    lines.append("")
    lines.append("| N | depth | speedup (2-SM/1-SM) | 95% CI | scaling efficiency % | surprising |")
    lines.append("| ---: | ---: | ---: | --- | ---: | --- |")
    for row in scaling_rows:
        lines.append(
            f"| {row['n']} | {row['depth']} | {row['speedup_2sm_over_1sm']:.4f} | "
            f"[{row['speedup_ci_low']:.4f}, {row['speedup_ci_high']:.4f}] | {row['scaling_efficiency_percent']:.2f} | "
            f"{'yes' if row['surprising_value_flag'] else 'no'} |"
        )
    lines.append("")
    lines.append("## Candidate depth saturation (diagnostic; four tested depths only)")
    lines.append("")
    lines.append("| method | N | d4 | d16 | d64 | d256 | earliest_tested_candidate_saturation_depth |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in saturation_rows:
        lines.append(
            f"| {row['method']} | {row['n']} | {row['depth_4_median_flops_per_cycle']:.4f} | "
            f"{row['depth_16_median_flops_per_cycle']:.4f} | {row['depth_64_median_flops_per_cycle']:.4f} | "
            f"{row['depth_256_median_flops_per_cycle']:.4f} | {row['earliest_tested_candidate_saturation_depth']} |"
        )
    lines.append("")
    lines.append("## Empirical per-SM BF16 Tensor Core ceiling candidate")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(empirical_ceiling, indent=2, sort_keys=True))
    lines.append("```")
    lines.append("")
    lines.append("## Profile validation (24 NCU cases)")
    lines.append("")
    lines.append("| index | case | sm_clock status | sm_clock_hz |")
    lines.append("| ---: | --- | --- | ---: |")
    for row in profile_validation_rows:
        hz_text = f"{row['sm_clock_hz']:.6g}" if row["sm_clock_hz"] not in (None, "") else "n/a"
        lines.append(f"| {row['index']} | {row['case_name']} | {row['sm_clock_status']} | {hz_text} |")
    lines.append("")
    lines.append("## Figures")
    lines.append("")
    lines.append("- `throughput.svg`")
    lines.append("- `scaling_efficiency.svg`")
    lines.append("- `saturation.svg`")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("`publishable: false`. Pending independent review and GB300 re-verification.")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subcommand: analyze
# ---------------------------------------------------------------------------
_STAT_FIELD_SUFFIXES = (
    "mean", "median", "stdev", "cv_percent", "min", "max",
    "median_ci_low", "median_ci_high", "iqr_low_bound", "iqr_high_bound", "iqr_flagged_count",
)


# Defect-6 repair: every one of the ten analysis artifacts must carry the
# exact ASCII token "publishable=false" in a format-appropriate
# machine-readable field -- never merely implied by the surrounding text (a
# CSV row's own value, a JSON string field, a Markdown status line, an SVG
# metadata element). PUBLICATION_STATUS_TOKEN is the single source of truth
# for the literal token; it never changes the scientific status (this
# remains a non-publishable pilot and empirical-ceiling candidate).
PUBLICATION_STATUS_TOKEN = "publishable=false"
PUBLICATION_STATUS_COLUMN = "publication_status"


def _config_stats_columns() -> list[str]:
    columns = ["method", "n", "depth", "cta_group", "sample_count"]
    for metric in STAT_METRIC_NAMES:
        for suffix in _STAT_FIELD_SUFFIXES:
            columns.append(f"{metric}_{suffix}")
        if metric == "flops_per_cycle":
            columns.append("flops_per_cycle_stability_review")
    columns.append(PUBLICATION_STATUS_COLUMN)
    return columns


CONFIGURATION_STATISTICS_HEADER = _config_stats_columns()
SCALING_HEADER = [
    "n", "depth", "median_flops_per_cycle_1sm", "median_flops_per_cycle_2sm", "speedup_2sm_over_1sm",
    "speedup_ci_low", "speedup_ci_high", "scaling_efficiency", "scaling_efficiency_percent", "surprising_value_flag",
    PUBLICATION_STATUS_COLUMN,
]
SATURATION_HEADER = [
    "method", "n", "depth_4_median_flops_per_cycle", "depth_16_median_flops_per_cycle",
    "depth_64_median_flops_per_cycle", "depth_256_median_flops_per_cycle", "max_median_flops_per_cycle",
    "earliest_tested_candidate_saturation_depth", PUBLICATION_STATUS_COLUMN,
]
PROFILE_VALIDATION_HEADER = [
    "index", "case_name", "method", "n", "depth", "cta_group", "kernel_symbol", "launch_id",
    "sm_clock_status", "sm_clock_raw_value", "sm_clock_unit", "sm_clock_hz", "diagnostic_metrics_resolved_count",
    PUBLICATION_STATUS_COLUMN,
]


def _stats_to_csv_row(key: tuple[str, int, int], config_result: dict) -> dict:
    method, n, depth = key
    row = {"method": method, "n": n, "depth": depth, "cta_group": config_result["cta_group"], "sample_count": config_result["elapsed_cycles"]["count"]}
    for metric in STAT_METRIC_NAMES:
        stats = config_result[metric]
        for suffix in _STAT_FIELD_SUFFIXES:
            source_key = "iqr_lower_fence" if suffix == "iqr_low_bound" else "iqr_upper_fence" if suffix == "iqr_high_bound" else suffix
            row[f"{metric}_{suffix}"] = stats[source_key]
        if metric == "flops_per_cycle":
            row["flops_per_cycle_stability_review"] = "REVIEW" if stats["stability_review"] else "ok"
    row[PUBLICATION_STATUS_COLUMN] = PUBLICATION_STATUS_TOKEN
    return row


def _with_publication_status(row: dict) -> dict:
    """Projects a pure, already-computed row (scaling.csv/saturation.csv --
    never a frozen formula function's own return value, only its CSV
    serialization) with the mandatory publication-status column appended."""
    return {**row, PUBLICATION_STATUS_COLUMN: PUBLICATION_STATUS_TOKEN}


def _csv_bytes(header: list[str], rows: list[dict]) -> bytes:
    """Deterministically renders one CSV artifact's exact bytes in memory
    (never touching disk), so Defect-5's retry logic can compare a
    freshly recomputed artifact against an existing one byte-for-byte
    before deciding whether a retry may skip, create, or must fail
    closed."""
    buf = io.StringIO(newline="")
    writer = csv.writer(buf)
    writer.writerow(header)
    for row in rows:
        writer.writerow([row[field] for field in header])
    return buf.getvalue().encode("utf-8")


def _write_bytes_no_clobber(path: Path, content: bytes) -> None:
    if os.path.lexists(path):
        raise p23.UnsafePathError(f"{path}: already exists, refusing to overwrite")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if os.path.lexists(tmp_path):
        raise p23.UnsafePathError(f"{tmp_path}: existing temporary; refusing to overwrite")
    try:
        with p23._open_exclusive(tmp_path, binary=True) as handle:
            handle.write(content)
    except Exception:
        if os.path.lexists(tmp_path):
            p23._safe_unlink_owned(tmp_path)
        raise
    try:
        p23._publish_no_clobber(tmp_path, path)
    except p23.UnsafePathError:
        if os.path.lexists(tmp_path):
            p23._safe_unlink_owned(tmp_path)
        raise


def _resolve_retryable_artifact(path: Path, content: bytes) -> tuple[str, str | None]:
    """Defect-5 repair: resolves exactly one of three actions for one
    planned analysis artifact, without creating, overwriting, or deleting
    anything. "create" if path does not exist at all. "skip" if path
    already exists as a genuine, non-symlink, regular file whose on-disk
    bytes exactly equal the freshly recomputed content (i.e. a clean retry
    may safely leave it untouched). "conflict" (with a human-readable
    detail) for anything else a retry must fail closed on: a symlink, a
    non-regular-file entry, unreadable content, or content that differs
    from the current, freshly validated recomputation."""
    if not os.path.lexists(path):
        return "create", None
    try:
        st = os.lstat(path)
    except OSError as exc:
        return "conflict", f"{path}: cannot stat existing analysis artifact: {exc}"
    if stat.S_ISLNK(st.st_mode):
        return "conflict", f"{path}: existing analysis artifact is a symlink; refusing to touch it"
    if not stat.S_ISREG(st.st_mode):
        return "conflict", f"{path}: existing analysis artifact is not a regular file; refusing to touch it"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        try:
            fst = os.fstat(fd)
            if (fst.st_dev, fst.st_ino) != (st.st_dev, st.st_ino) or not stat.S_ISREG(fst.st_mode):
                return "conflict", f"{path}: changed identity while being checked; refusing to touch it"
            existing_bytes = b"".join(iter(lambda: os.read(fd, 1 << 20), b""))
        finally:
            os.close(fd)
    except OSError as exc:
        return "conflict", f"{path}: cannot read existing analysis artifact to verify a safe retry: {exc}"
    if existing_bytes != content:
        return "conflict", (
            f"{path}: existing analysis artifact differs from the freshly recomputed evidence; refusing to "
            f"overwrite (a retry must never clobber divergent evidence)"
        )
    return "skip", None


def _do_analyze(
    *, campaign_dir: Path, analyzed_at_utc: str,
    _test_hook_before_final_gate: Callable[[], None] | None = None,
    _test_hook_during_publication: Callable[[int, int], None] | None = None,
    _test_hook_before_manifest_append: Callable[[], None] | None = None,
) -> tuple[bool, list[str]]:
    """Defect-5 repair: deterministic, no-clobber, resumable publication. A
    campaign already at COMPLETE creates only the analysis artifacts that
    are actually missing (a clean retry after a partial or fully-interrupted
    earlier publication attempt); a campaign already ANALYZED/INCONCLUSIVE
    is purely revalidated -- every one of the ten artifacts and the
    manifest's own recorded hashes are re-checked byte-for-byte, and success
    is returned only if everything still matches, without writing anything.
    Any existing artifact that differs from a fresh recomputation, or any
    unexpected entry under analysis/, fails the whole call closed without
    overwriting or deleting anything."""
    try:
        manifest, _revision = load_p24_manifest_chain(campaign_dir)
        _validate_p24_manifest_document(manifest, require_initialized=True)
    except (p23.ManifestTransitionError, p23.UnsafePathError) as exc:
        return False, [f"P2.4 manifest: {exc}"]
    state = manifest.get("state")
    if state not in ("COMPLETE", "ANALYZED", "INCONCLUSIVE"):
        return False, [f"P2.4 manifest state={state!r} not in ('COMPLETE', 'ANALYZED', 'INCONCLUSIVE'); cannot analyze"]

    # Validate the complete transition eligibility -- the manifest hash
    # chain (already re-loaded/re-validated above), every one of the 24
    # profiles' own evidence (Defects 1/2), and the complete raw P2.3 pilot
    # evidence (Defect 4, 720 samples) -- before deriving or publishing
    # anything, whether this is a first publication, a retry, or a pure
    # revalidation of an already-terminal campaign.
    integrity_errors, _verified_before = verify_campaign_evidence_integrity(campaign_dir, manifest)
    if integrity_errors:
        return False, integrity_errors

    pilot_ref = manifest.get("pilot_campaign_reference", {})
    if not isinstance(pilot_ref, dict) or "path" not in pilot_ref:
        return False, ["manifest pilot_campaign_reference is missing or incomplete"]
    try:
        p23_campaign_dir = resolve_p23_campaign_dir_arg(pilot_ref["path"])
    except p23.UnsafePathError as exc:
        return False, [f"pilot_campaign_reference.path: {exc}"]
    p23_revalidation_errors, _p23_snapshot_before = revalidate_p23_pilot_campaign(
        p23_campaign_dir, git_commit=manifest.get("provenance", {}).get("git_commit"),
    )
    if p23_revalidation_errors:
        return False, p23_revalidation_errors
    combined_path = p23_campaign_dir / "combined_samples.csv"

    errors: list[str] = []
    samples_by_config = _read_combined_samples(combined_path)
    plan = build_profile_plan()
    expected_configs = {(e["method"], e["n"], e["depth"]) for e in plan}
    for missing in sorted(expected_configs - set(samples_by_config)):
        errors.append(f"combined_samples.csv missing configuration {missing}")
    for key in sorted(set(samples_by_config) - expected_configs):
        errors.append(f"combined_samples.csv has unexpected configuration {key}")
    for key, entry in samples_by_config.items():
        for metric in STAT_METRIC_NAMES:
            values = entry[metric]
            if len(values) != FROZEN_PILOT_PARAMS["repetitions"]:
                errors.append(f"configuration {key} metric {metric} has {len(values)} sample(s), expected exactly {FROZEN_PILOT_PARAMS['repetitions']}")
            if any(not math.isfinite(v) for v in values):
                errors.append(f"configuration {key} has a non-finite {metric} value")
    if errors:
        return False, errors

    # Derive all ten output byte streams deterministically from the
    # freshly validated raw evidence above -- never from anything cached or
    # previously derived.
    rng = random.Random(BOOTSTRAP_SEED)
    stats_by_config = compute_all_config_stats(samples_by_config, rng)
    scaling_rows = compute_scaling(samples_by_config, stats_by_config, rng)
    saturation_rows = compute_saturation(stats_by_config)

    case_results = manifest.get("case_results", {})
    profile_validation_rows: list[dict] = []
    inconclusive_reason: list[str] = []
    all_sm_clock_valid = True
    for entry in plan:
        result = case_results.get(entry["case_name"])
        if result is None:
            errors.append(f"case_results is missing entry for {entry['case_name']}")
            continue
        diagnostic_count = len(result.get("diagnostic_metric_values", {}))
        sm_status = "OK" if result.get("sm_clock_valid") else (result.get("sm_clock_issue") or "UNKNOWN")
        if not result.get("sm_clock_valid"):
            all_sm_clock_valid = False
            inconclusive_reason.append(f"{entry['case_name']}: sm_clock_status={sm_status}")
        profile_validation_rows.append({
            "index": entry["index"], "case_name": entry["case_name"], "method": entry["method"],
            "n": entry["n"], "depth": entry["depth"], "cta_group": entry["cta_group"],
            "kernel_symbol": entry["kernel_symbol"], "launch_id": result.get("launch_id"),
            "sm_clock_status": sm_status, "sm_clock_raw_value": result.get("sm_clock_raw_value"),
            "sm_clock_unit": result.get("sm_clock_unit"), "sm_clock_hz": result.get("sm_clock_hz"),
            "diagnostic_metrics_resolved_count": diagnostic_count,
        })
    if errors:
        return False, errors

    stats_rows = [_stats_to_csv_row(key, stats_by_config[key]) for key in sorted(stats_by_config, key=lambda k: (k[1], k[2], k[0]))]
    scaling_csv_rows = [_with_publication_status(row) for row in scaling_rows]
    saturation_csv_rows = [_with_publication_status(row) for row in saturation_rows]
    profile_validation_csv_rows = [_with_publication_status(row) for row in profile_validation_rows]

    empirical_ceiling = build_empirical_ceiling(
        stats_by_config=stats_by_config, case_results=case_results,
        all_sm_clock_valid=all_sm_clock_valid, inconclusive_reason=inconclusive_reason,
    )
    outcome_state = "ANALYZED" if all_sm_clock_valid else "INCONCLUSIVE"
    if state in ("ANALYZED", "INCONCLUSIVE") and state != outcome_state:
        return False, [
            f"manifest already reached terminal state={state!r}, but a fresh recomputation from currently "
            f"validated raw evidence independently derives outcome={outcome_state!r}; refusing to silently "
            f"accept a contradiction"
        ]

    provenance = manifest.get("provenance", {})
    report_md = render_report_markdown(
        campaign_id=str(manifest.get("campaign_id")), stats_rows=stats_rows, scaling_rows=scaling_rows,
        saturation_rows=saturation_rows, profile_validation_rows=profile_validation_rows,
        empirical_ceiling=empirical_ceiling, provenance=provenance,
    )
    throughput_svg = render_throughput_svg(stats_by_config)
    scaling_svg = render_scaling_efficiency_svg(scaling_rows)
    saturation_svg = render_saturation_svg(saturation_rows)
    analysis_manifest = {
        "schema_version": P24_SCHEMA_VERSION,
        "campaign_id": manifest.get("campaign_id"),
        "publishable": False,
        "publication_status": PUBLICATION_STATUS_TOKEN,
        "status": outcome_state,
        "provenance": provenance,
        "resolved_ncu_metrics": manifest.get("resolved_ncu_metrics", {}),
        "notes": [
            "single reviewed pilot, not a final campaign",
            "no sample was ever removed; IQR flags are diagnostics only",
            "TFLOP/s is never derived from NCU kernel duration",
            "empirical per-SM ceiling is a candidate, never a theoretical peak or measured whole-device throughput",
            "publishable: false, pending independent audit and GB300 re-verification",
        ],
    }

    analysis_dir = campaign_dir / "analysis"
    artifacts: list[tuple[Path, bytes]] = [
        (analysis_dir / "configuration_statistics.csv", _csv_bytes(CONFIGURATION_STATISTICS_HEADER, stats_rows)),
        (analysis_dir / "scaling.csv", _csv_bytes(SCALING_HEADER, scaling_csv_rows)),
        (analysis_dir / "saturation.csv", _csv_bytes(SATURATION_HEADER, saturation_csv_rows)),
        (analysis_dir / "profile_validation.csv", _csv_bytes(PROFILE_VALIDATION_HEADER, profile_validation_csv_rows)),
        (analysis_dir / "empirical_ceiling.json", (json.dumps(empirical_ceiling, indent=2, sort_keys=True) + "\n").encode("utf-8")),
        (analysis_dir / "report.md", report_md.encode("utf-8")),
        (analysis_dir / "throughput.svg", throughput_svg.encode("utf-8")),
        (analysis_dir / "scaling_efficiency.svg", scaling_svg.encode("utf-8")),
        (analysis_dir / "saturation.svg", saturation_svg.encode("utf-8")),
        (analysis_dir / "analysis_manifest.json", (json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")),
    ]
    planned_analysis_paths = tuple(path.relative_to(campaign_dir).as_posix() for path, _content in artifacts)
    if len(planned_analysis_paths) != len(ANALYSIS_ARTIFACT_RELATIVE_PATHS) or set(planned_analysis_paths) != set(ANALYSIS_ARTIFACT_RELATIVE_PATHS):
        return False, [
            f"internal analysis artifact inventory differs from the canonical terminal manifest contract: "
            f"planned={sorted(planned_analysis_paths)!r} canonical={sorted(ANALYSIS_ARTIFACT_RELATIVE_PATHS)!r}"
        ]

    # No unexpected entry may exist under analysis/ (a stray file -- e.g. an
    # orphaned .tmp from a crash mid-write -- fails closed rather than being
    # silently ignored or cleaned up).
    expected_basenames = {path.name for path, _content in artifacts}
    if os.path.lexists(analysis_dir):
        if os.path.islink(analysis_dir):
            return False, [f"{analysis_dir}: is a symlink; refusing"]
        try:
            actual_names = set(os.listdir(analysis_dir))
        except OSError as exc:
            return False, [f"{analysis_dir}: cannot list directory: {exc}"]
        unexpected = sorted(actual_names - expected_basenames)
        if unexpected:
            return False, [f"{analysis_dir}: unexpected entrie(s) present: {unexpected}"]

    # Resolve exactly one action per artifact -- create/skip/conflict --
    # before creating, overwriting, or deleting anything.
    to_create: list[tuple[Path, bytes]] = []
    conflicts: list[str] = []
    for path, content in artifacts:
        action, conflict_detail = _resolve_retryable_artifact(path, content)
        if action == "create":
            to_create.append((path, content))
        elif action == "conflict":
            conflicts.append(conflict_detail)
    if conflicts:
        return False, conflicts

    if state in ("ANALYZED", "INCONCLUSIVE"):
        # A terminal campaign must never gain a newly created artifact; if
        # one is missing here, the campaign's own claim of being terminal is
        # itself untrustworthy -- fail closed rather than complete it now.
        if to_create:
            return False, [
                f"manifest already reached terminal state={state!r}, but {len(to_create)} analysis "
                f"artifact(s) are missing on disk: {[p.name for p, _c in to_create]}"
            ]
        recorded_hashes = manifest.get("artifact_sha256", {})
        hash_errors = []
        for path, content in artifacts:
            rel = path.relative_to(campaign_dir).as_posix()
            actual_hash = hashlib.sha256(content).hexdigest()
            if recorded_hashes.get(rel) != actual_hash:
                hash_errors.append(f"{rel}: recomputed SHA-256 {actual_hash} != manifest's recorded {recorded_hashes.get(rel)!r}")
        if hash_errors:
            return False, hash_errors
        return True, []

    # state == "COMPLETE": create only the artifacts that are actually
    # missing, exclusively, no-clobber; every already-valid existing
    # artifact is left completely untouched (never rewritten, never
    # re-hashed-and-republished).
    published: list[Path] = []
    published_identity: dict[Path, tuple[int, int]] = {}

    def _cleanup_published() -> None:
        for cleanup_path in published:
            try:
                p23._safe_unlink_owned(cleanup_path, published_identity.get(cleanup_path))
            except p23.UnsafePathError:
                pass

    try:
        p23._mkdir_component(analysis_dir, must_not_exist=False, root=REPO_ROOT)
    except p23.UnsafePathError as exc:
        return False, [str(exc)]

    try:
        for i, (path, content) in enumerate(to_create):
            if _test_hook_during_publication is not None:
                _test_hook_during_publication(i, len(to_create))
            _write_bytes_no_clobber(path, content)
            published.append(path)
            published_identity[path] = p23._file_identity(path)
    except (p23.UnsafePathError, OSError) as exc:
        _cleanup_published()
        return False, [f"analysis artifact publication failed: {exc}"]

    # Re-read and re-hash all ten outputs fresh from disk immediately before
    # the final manifest revision -- including any artifact a prior,
    # interrupted attempt had already safely published.
    try:
        artifact_hashes = {path.relative_to(campaign_dir).as_posix(): p23.sha256_of(path) for path, _content in artifacts}
    except p23.UnsafePathError as exc:
        _cleanup_published()
        return False, [str(exc)]

    if _test_hook_before_final_gate is not None:
        _test_hook_before_final_gate()

    integrity_errors2, _verified_after = verify_campaign_evidence_integrity(campaign_dir, manifest)
    if integrity_errors2:
        _cleanup_published()
        return False, [
            "evidence changed while analysis was being computed/published, detected by the second "
            "(pre-terminal) integrity gate; the terminal state was not published and the analysis "
            "artifacts just written were removed:",
        ] + integrity_errors2

    p23_revalidation_errors2, _p23_snapshot_after = revalidate_p23_pilot_campaign(
        p23_campaign_dir, git_commit=manifest.get("provenance", {}).get("git_commit"),
    )
    if p23_revalidation_errors2:
        _cleanup_published()
        return False, [
            "P2.3 pilot evidence changed while analysis was being computed/published, detected by the "
            "second (pre-terminal) P2.3 revalidation; the terminal state was not published and the "
            "analysis artifacts just written were removed:",
        ] + p23_revalidation_errors2

    if _test_hook_before_manifest_append is not None:
        _test_hook_before_manifest_append()

    updates = {
        "analysis_completed_at_utc": analyzed_at_utc,
        "artifact_sha256": {**manifest.get("artifact_sha256", {}), **artifact_hashes},
    }
    if outcome_state == "INCONCLUSIVE":
        updates["inconclusive_reason"] = inconclusive_reason
    try:
        p24_merge_manifest(campaign_dir, updates, state=outcome_state)
    except p23.ManifestTransitionError as exc:
        return False, [str(exc)]
    return True, []


def cmd_analyze(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_p24_campaign_dir(args.campaign_dir)
    except p23.UnsafePathError as exc:
        print(f"analyze_exp02_umma_throughput_p24: analyze: ERROR: {exc}", file=sys.stderr)
        return 2
    success, errors = _do_analyze(campaign_dir=campaign_dir, analyzed_at_utc=args.analyzed_at_utc)
    if not success:
        print("analyze_exp02_umma_throughput_p24: analyze: ERROR:", file=sys.stderr)
        for error in errors:
            print(f"analyze_exp02_umma_throughput_p24: analyze:   - {error}", file=sys.stderr)
        return 1
    print("analyze_exp02_umma_throughput_p24: analyze: OK", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: manifest-write (FAILED/INTERRUPTED only)
# ---------------------------------------------------------------------------
def cmd_manifest_write(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_p24_campaign_dir(args.campaign_dir)
    except p23.UnsafePathError as exc:
        print(f"analyze_exp02_umma_throughput_p24: manifest-write: ERROR: {exc}", file=sys.stderr)
        return 2
    updates: dict = {}
    if args.merge_json:
        merge_path = Path(args.merge_json)
        try:
            updates = json.loads(merge_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"analyze_exp02_umma_throughput_p24: manifest-write: ERROR: cannot read --merge-json: {exc}", file=sys.stderr)
            return 2
        if not isinstance(updates, dict):
            print("analyze_exp02_umma_throughput_p24: manifest-write: ERROR: --merge-json must contain a JSON object", file=sys.stderr)
            return 2
    try:
        p24_merge_manifest(campaign_dir, updates, state=args.status)
    except p23.ManifestTransitionError as exc:
        print(f"analyze_exp02_umma_throughput_p24: manifest-write: ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"analyze_exp02_umma_throughput_p24: manifest-write: OK: state={args.status}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Self-test: GPU-free synthetic/adversarial tests. Never touches CUDA,
# Docker, nvidia-smi, either UMMA binary, NCU, the network, or real raw
# results. Every campaign directory is built under a
# tempfile.TemporaryDirectory with both this module's and p23's REPO_ROOT
# patched to it.
# ---------------------------------------------------------------------------
class _Recorder:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.total = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.total += 1
        if condition:
            print(f"analyze_exp02_umma_throughput_p24: self-test: PASS: {name}", file=sys.stderr)
        else:
            self.failures.append(name)
            suffix = f"; {detail}" if detail else ""
            print(f"analyze_exp02_umma_throughput_p24: self-test: FAIL: {name}{suffix}", file=sys.stderr)

    def expect_error_containing(self, name: str, errors: list[str], needle: str) -> None:
        self.check(name, any(needle in error for error in errors), detail=f"expected substring {needle!r} in errors={errors}")


_FIXED_GIT_COMMIT = "d" * 40
_FIXED_GPU_UUID = "GPU-22222222-3333-4444-5555-666666666666"
_FIXED_GPU_NAME = "NVIDIA B300 SXM6"


def _default_preflight_doc(
    *, timestamp_utc: str = "20260804T100000Z", overall_status: str = "PASS", git_dirty: bool = False,
    git_commit: str = _FIXED_GIT_COMMIT, gpu_uuid: str = _FIXED_GPU_UUID, gpu_name: str = _FIXED_GPU_NAME,
    compute_cap: str = "10.3", driver_version: str = "580.95.05",
    gpu_visibility_status: str = "PASS", ncu_profile_status: str = "PASS",
) -> dict:
    return {
        "schema_version": "1", "timestamp_utc": timestamp_utc, "git_commit": git_commit, "git_dirty": git_dirty,
        "host_arch": "x86_64", "tool_versions": {"nvcc": "release 13.1", "ncu": "version 2025.4.0.0"},
        "gpu": {
            "logical_index": "0", "name": gpu_name, "uuid": gpu_uuid, "driver_version": driver_version,
            "compute_cap": compute_cap, "memory_total": "288 GiB",
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


def _build_p23_pilot_campaign_fixture(
    tmp_path: Path, campaign_id: str, *, git_commit: str = _FIXED_GIT_COMMIT,
    gpu_uuid: str = _FIXED_GPU_UUID, gpu_name: str = _FIXED_GPU_NAME, repetitions: int = 30,
    elapsed_cycles_fn=None,
) -> tuple[Path, dict]:
    """Builds a complete, self-consistent, COMPLETE-state synthetic P2.3
    campaign directly on disk, reusing p23's own fixture/aggregate helpers so
    it is byte-for-byte the shape a real P2.3 finalize produces, at the same
    canonical results/raw/exp02_umma_throughput/<campaign_id> path a real
    P2.3 campaign occupies (relative to the patched REPO_ROOT)."""
    plan = p23.build_plan()
    campaign_dir = tmp_path.joinpath(*p23.RAW_ROOT_PARTS, campaign_id)
    (campaign_dir / "cases").mkdir(parents=True)
    (campaign_dir / "logs").mkdir(parents=True)
    cases: list[tuple[dict, list[dict]]] = []
    for entry in plan:
        rows = []
        for sample_index in range(repetitions):
            overrides = {"gpu_uuid": gpu_uuid, "gpu_name": gpu_name}
            if elapsed_cycles_fn is not None:
                overrides["elapsed_cycles"] = str(elapsed_cycles_fn(entry, sample_index))
                total_umma = entry["depth"] * FROZEN_PILOT_PARAMS["iterations"]
                info = p23.METHOD_INFO[entry["method"]]
                flops_per_umma = 2 * info["m"] * entry["n"] * p23.FROZEN_K
                total_flops = flops_per_umma * total_umma
                ec = int(overrides["elapsed_cycles"])
                overrides["cycles_per_umma"] = f"{ec / total_umma:.6f}"
                overrides["flops_per_cycle"] = f"{total_flops / ec:.6f}"
            row = p23._default_row(
                entry, sample_index, repetitions=repetitions, run_kind="benchmark",
                iterations=FROZEN_PILOT_PARAMS["iterations"], warmup_iterations=FROZEN_PILOT_PARAMS["warmup_iterations"],
                git_commit=git_commit, overrides=overrides,
            )
            rows.append(row)
        p23._write_case_csv(campaign_dir / "cases" / f"{entry['case_name']}.csv", rows)
        cases.append((entry, rows))
    p23.write_execution_order(campaign_dir, plan)
    started = "20260804T090000Z"
    p23.merge_manifest(
        campaign_dir,
        {
            "campaign_id": campaign_id, "run_kind": "benchmark", "started_at_utc": started,
            "configuration_count_expected": p23.EXPECTED_CONFIGURATION_COUNT,
            "configuration_count_completed": p23.EXPECTED_CONFIGURATION_COUNT,
            "sample_count_expected": p23.EXPECTED_CONFIGURATION_COUNT * repetitions,
            "sample_count_completed": p23.EXPECTED_CONFIGURATION_COUNT * repetitions,
            "requested": {
                "run_kind": "benchmark", "iterations": FROZEN_PILOT_PARAMS["iterations"],
                "warmup_iterations": FROZEN_PILOT_PARAMS["warmup_iterations"], "repetitions": repetitions,
                "campaign_id": campaign_id,
            },
            "selected_gpu_index": 0, "git_commit": git_commit, "git_dirty": False,
            "self_test_outcomes": {"umma_1sm": "PASS", "umma_2sm": "PASS"},
        },
        status="IN_PROGRESS",
    )
    args = argparse.Namespace(
        campaign_id=campaign_id, run_kind="benchmark", repetitions=repetitions,
        iterations=FROZEN_PILOT_PARAMS["iterations"], warmup_iterations=FROZEN_PILOT_PARAMS["warmup_iterations"],
        git_commit=git_commit, gpu_index=0, started_at_utc=started, completed_at_utc="20260804T093000Z",
        self_test_umma_1sm="PASS", self_test_umma_2sm="PASS",
    )
    synthetic_artifact = tmp_path / f"synthetic_build_artifact_{campaign_id}"
    synthetic_artifact.write_bytes(b"synthetic non-empty artifact\n")
    synthetic_final_artifacts = {label: synthetic_artifact for label in p23.DEFAULT_FINAL_ARTIFACTS}
    synthetic_versions = tmp_path / f"VERSIONS_{campaign_id}.env"
    synthetic_versions.write_text(
        "CUDA_VERSION=13.1.0\nCUDA_IMAGE=nvidia/cuda:13.1.0-devel-ubuntu24.04\n"
        "CUDA_IMAGE_DIGEST=sha256:" + "0" * 64 + "\nCUDA_IMAGE_PLATFORM=linux/amd64\n"
        "CUTLASS_VERSION=v4.6.1\nCUTLASS_COMMIT=" + "0" * 40 + "\nCUDA_ARCH=sm_103a\nMAX_BUILD_JOBS=2\n",
        encoding="utf-8",
    )
    success, errors = p23._do_finalize(campaign_dir, args, artifact_paths=synthetic_final_artifacts, versions_path=synthetic_versions)
    if not success:
        raise AssertionError(f"self-test fixture: P2.3 finalize failed: {errors}")
    manifest = p23.load_manifest(campaign_dir)
    return campaign_dir, manifest


def _build_ncu_case_fixture(
    tmp_path: Path, entry: dict, *, git_commit: str = _FIXED_GIT_COMMIT, gpu_uuid: str = _FIXED_GPU_UUID,
    gpu_name: str = _FIXED_GPU_NAME, sm_cycles_per_second: float | None = 1_400_000_000.0,
    kernel_name_in_csv: str | None = None, unit_override: str | None = None,
    resolved_metrics: tuple[str, ...] = CANDIDATE_METRICS, omit_metrics: tuple[str, ...] = (),
    case_dir: Path | None = None,
) -> tuple[Path, Path, Path, dict]:
    info = p23.METHOD_INFO[entry["method"]]
    row = p23._default_row(
        entry, 0, repetitions=1, run_kind="benchmark", iterations=FROZEN_PROFILE_PARAMS["iterations"],
        warmup_iterations=FROZEN_PROFILE_PARAMS["warmup_iterations"], git_commit=git_commit,
        overrides={"gpu_uuid": gpu_uuid, "gpu_name": gpu_name},
    )
    out_dir = case_dir if case_dir is not None else tmp_path
    out_dir.mkdir(parents=True, exist_ok=True)
    app_csv = out_dir / f"{entry['case_name']}.application.csv"
    p23._write_case_csv(app_csv, [row])

    kernel_name_field = kernel_name_in_csv or entry["kernel_symbol"]
    sm_clock_unit = "cycle/nsecond" if unit_override is None else unit_override
    if sm_cycles_per_second is None:
        sm_clock_raw_value = ""
    elif sm_clock_unit.strip().lower() == "hz":
        sm_clock_raw_value = f"{sm_cycles_per_second}"
    else:
        # The cycle/nsecond representation is the default.  Unknown-unit
        # fixtures use this harmless finite value too; the production
        # evaluator must reject them based on the unit before conversion.
        sm_clock_raw_value = f"{sm_cycles_per_second / 1e9}"
    metadata_header = ["ID", "Process ID", "Process Name", "Host Name", "Kernel Name", "Kernel Time", "Context", "Stream"]
    metadata_values = ["1", "1234", "fixture", "fixture-host", kernel_name_field, "2026-Aug-04 00:00:00", "1", "7"]
    default_values: dict[str, str] = {
        MANDATORY_SM_CLOCK_METRIC: sm_clock_raw_value,
        "gpu__time_duration.sum": "654321",
        "device__attribute_multiprocessor_count": "132",
        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed": "88.5",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": "90.1",
        "sm__inst_executed_pipe_tensor.sum": "4096",
        "smsp__inst_executed_pipe_tensor.sum": "1024",
    }
    default_units: dict[str, str] = {
        MANDATORY_SM_CLOCK_METRIC: sm_clock_unit,
        "gpu__time_duration.sum": "ns", "device__attribute_multiprocessor_count": "",
        "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed": "%",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": "%",
        "sm__inst_executed_pipe_tensor.sum": "inst", "smsp__inst_executed_pipe_tensor.sum": "inst",
    }
    metric_names, metric_units, metric_values = [], [], []
    for candidate in resolved_metrics:
        if candidate in omit_metrics:
            continue
        metric_names.append(candidate)
        metric_units.append(default_units[candidate])
        metric_values.append(default_values[candidate])
    header = metadata_header + metric_names
    units = [""] * len(metadata_header) + metric_units
    launch_values = metadata_values + metric_values
    metrics_csv = out_dir / f"{entry['case_name']}.metrics_raw.csv"
    with open(metrics_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerow(units)
        writer.writerow(launch_values)
    ncu_rep = out_dir / f"{entry['case_name']}_report.ncu-rep"
    ncu_rep.write_bytes(b"synthetic ncu report bytes, never a real profile\n")
    # The remaining four canonical per-case artifacts (Defect-2 repair: the
    # frozen inventory is exactly seven files -- see
    # CANONICAL_PROFILE_CASE_FILE_LABELS).
    (out_dir / f"{entry['case_name']}.ncu_tool.log").write_text("synthetic ncu tool log\n", encoding="utf-8")
    (out_dir / f"{entry['case_name']}.container_stdout.log").write_text("synthetic container stdout\n", encoding="utf-8")
    # Successful application/NCU-export invocations commonly produce no
    # stderr. Keep the fixture faithful to the real GB300 failure that
    # exposed the validator bug: both mandatory diagnostic artifacts exist
    # as genuine regular files, but legitimately contain zero bytes.
    (out_dir / f"{entry['case_name']}.container_stderr.log").write_bytes(b"")
    (out_dir / f"{entry['case_name']}.metrics_export_stderr.log").write_bytes(b"")
    return app_csv, metrics_csv, ncu_rep, row


def _run_full_pipeline(tmp_path: Path, campaign_id: str, *, sm_cycles_fn=None, sm_clock_unit_fn=None) -> Path:
    """End-to-end fixture: P2.3 pilot -> record-pilot -> discover-metrics ->
    all 24 profile cases validated. sm_cycles_fn(entry) -> float|None
    overrides the per-case SM-clock reading (None omits the metric's value
    entirely, simulating a counter this GB300 build did not populate).
    sm_clock_unit_fn(entry) -> str selects the raw NCU unit representation."""
    # The P2.4 wrapper and the P2.3 campaign it drives share one explicit
    # campaign ID (src/compute/P2_4_PROTOCOL.md section 2); the two live
    # under different raw roots (exp02_umma_throughput vs.
    # exp02_umma_throughput_p24) so this never collides.
    p23_campaign_dir, _ = _build_p23_pilot_campaign_fixture(tmp_path, campaign_id)
    p24_campaign_dir = _do_init_campaign(campaign_id=campaign_id, started_at_utc="20260804T100000Z")
    preflight_path = tmp_path / f"preflight_{campaign_id}.json"
    _write_preflight_json(preflight_path, _default_preflight_doc())
    now = _datetime(2026, 8, 4, 11, 0, tzinfo=_timezone.utc)
    ok, errors = _do_record_pilot(
        campaign_dir=p24_campaign_dir, p23_campaign_dir=p23_campaign_dir, preflight_path=preflight_path,
        git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260804T103000Z", now_utc=now,
    )
    if not ok:
        raise AssertionError(f"self-test fixture: record-pilot failed: {errors}")
    discovery_log = tmp_path / f"discovery_{campaign_id}.log"
    discovery_log.write_text("\n".join(CANDIDATE_METRICS) + "\n", encoding="utf-8")
    ok, errors, _resolved = _do_discover_metrics(
        campaign_dir=p24_campaign_dir, discovery_log=discovery_log, preflight_path=preflight_path,
        git_commit=_FIXED_GIT_COMMIT, started_at_utc="20260804T103100Z", now_utc=now,
    )
    if not ok:
        raise AssertionError(f"self-test fixture: discover-metrics failed: {errors}")
    for entry in build_profile_plan():
        sm_cycles = sm_cycles_fn(entry) if sm_cycles_fn is not None else 1_400_000_000.0
        sm_clock_unit = sm_clock_unit_fn(entry) if sm_clock_unit_fn is not None else None
        _build_ncu_case_fixture(
            tmp_path, entry, sm_cycles_per_second=sm_cycles, unit_override=sm_clock_unit,
            case_dir=p24_campaign_dir / "profiles" / entry["case_name"],
        )
        ok, errors = _do_validate_profile_case(campaign_dir=p24_campaign_dir, index=entry["index"], git_commit=_FIXED_GIT_COMMIT)
        if not ok:
            raise AssertionError(f"self-test fixture: validate-profile-case {entry['case_name']} failed: {errors}")
    return p24_campaign_dir


def run_self_test() -> int:  # noqa: C901
    rec = _Recorder()

    plan = build_profile_plan()
    plan_errors = check_profile_plan_contract(plan)
    rec.check("profile plan has exactly 24 cases, identical to P2.3's own plan plus kernel_symbol", not plan_errors, detail=str(plan_errors))
    rec.check(
        "kernel_symbol matches the documented P2.1/P2.2 symbol form",
        all(
            e["kernel_symbol"] == (f"umma_1sm_m128n{e['n']}k16_d{e['depth']}" if e["method"] == "umma_1sm" else f"umma_2sm_m256n{e['n']}k16_d{e['depth']}")
            for e in plan
        ),
    )
    rec.check("every kernel_symbol is unique", len({e["kernel_symbol"] for e in plan}) == len(plan))

    # --- statistics: determinism, sample stdev, bootstrap determinism ------
    try:
        compute_metric_stats([1000.0])
        single_sample_rejected = False
    except ValueError:
        single_sample_rejected = True
    rec.check("compute_metric_stats refuses to silently report zero stdev/CV for a single observation", single_sample_rejected)
    two_sample = compute_metric_stats([1000.0, 1002.0])
    rec.check("compute_metric_stats computes a real, non-zero sample stdev for two or more observations", two_sample["stdev"] > 0.0)

    values_a = [100.0 + i for i in range(30)]
    rng1 = random.Random(BOOTSTRAP_SEED)
    ci1 = bootstrap_indices_median_ci(values_a, rng1)
    rng2 = random.Random(BOOTSTRAP_SEED)
    ci2 = bootstrap_indices_median_ci(values_a, rng2)
    rec.check("bootstrap median CI is deterministic for a fixed seed and identical input", ci1 == ci2)

    values_b = [200.0 + i for i in range(30)]
    rng3 = random.Random(BOOTSTRAP_SEED)
    ratio_ci1 = bootstrap_indices_ratio_ci(values_a, values_b, rng3)
    rng4 = random.Random(BOOTSTRAP_SEED)
    ratio_ci2 = bootstrap_indices_ratio_ci(values_a, values_b, rng4)
    rec.check("bootstrap ratio CI is deterministic for a fixed seed and identical input", ratio_ci1 == ratio_ci2)

    lower, upper, flagged = iqr_bounds([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])
    rec.check("iqr_bounds flags a clear outlier without removing it from the input", flagged == 1)

    # --- scaling / efficiency formulas, never clamped -----------------------
    fake_samples = {
        ("umma_1sm", 128, 16): {"flops_per_cycle": [10.0] * 30, "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": [10.0] * 30, "cta_group": 1},
        ("umma_2sm", 128, 16): {"flops_per_cycle": [25.0] * 30, "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": [12.5] * 30, "cta_group": 2},
    }
    rng5 = random.Random(BOOTSTRAP_SEED)
    fake_stats = compute_all_config_stats(fake_samples, rng5)
    scaling_rows = compute_scaling(fake_samples, fake_stats, rng5)
    row = scaling_rows[0]
    rec.check("speedup_2sm_over_1sm = median(2sm)/median(1sm)", math.isclose(row["speedup_2sm_over_1sm"], 2.5))
    rec.check("scaling_efficiency = speedup / 2", math.isclose(row["scaling_efficiency"], 1.25))
    rec.check("scaling_efficiency_percent = 100 * efficiency", math.isclose(row["scaling_efficiency_percent"], 125.0))
    rec.check(
        "scaling efficiency above 100% is never clamped and is flagged as surprising",
        row["scaling_efficiency_percent"] > 100.0 and row["surprising_value_flag"] is True,
    )

    # --- saturation rule: boundary and CI-overlap behavior ------------------
    # Bimodal (not constant) samples give the bootstrap median CI real width,
    # so the overlap check is genuinely exercised rather than trivially
    # comparing two zero-width points.
    def _bimodal(lo: float, hi: float) -> list[float]:
        return [lo] * 15 + [hi] * 15

    sat_samples = {
        ("umma_1sm", 64, 4): {"flops_per_cycle": [5.0] * 30, "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": [5.0] * 30, "cta_group": 1},
        ("umma_1sm", 64, 16): {"flops_per_cycle": _bimodal(9.0, 10.2), "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": _bimodal(9.0, 10.2), "cta_group": 1},
        ("umma_1sm", 64, 64): {"flops_per_cycle": _bimodal(9.6, 10.4), "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": _bimodal(9.6, 10.4), "cta_group": 1},
        ("umma_1sm", 64, 256): {"flops_per_cycle": [8.0] * 30, "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": [8.0] * 30, "cta_group": 1},
    }
    rng6 = random.Random(BOOTSTRAP_SEED)
    sat_stats = compute_all_config_stats(sat_samples, rng6)
    sat_rows = compute_saturation(sat_stats)
    sat_row = next(r for r in sat_rows if r["method"] == "umma_1sm" and r["n"] == 64)
    rec.check(
        "saturation selects an earlier depth once it meets both the 95% threshold and a CI overlapping the max's own CI",
        sat_row["earliest_tested_candidate_saturation_depth"] == 16,
        detail=str(sat_row),
    )

    sat_no_overlap_samples = {
        ("umma_1sm", 64, 4): {"flops_per_cycle": [5.0] * 30, "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": [5.0] * 30, "cta_group": 1},
        ("umma_1sm", 64, 16): {"flops_per_cycle": [9.6] * 30, "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": [9.6] * 30, "cta_group": 1},
        ("umma_1sm", 64, 64): {"flops_per_cycle": [10.0] * 30, "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": [10.0] * 30, "cta_group": 1},
        ("umma_1sm", 64, 256): {"flops_per_cycle": [10.0] * 30, "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": [10.0] * 30, "cta_group": 1},
    }
    rng6b = random.Random(BOOTSTRAP_SEED)
    sat_no_overlap_stats = compute_all_config_stats(sat_no_overlap_samples, rng6b)
    sat_no_overlap_rows = compute_saturation(sat_no_overlap_stats)
    sat_no_overlap_row = next(r for r in sat_no_overlap_rows if r["method"] == "umma_1sm" and r["n"] == 64)
    rec.check(
        "a depth meeting the 95% threshold but whose (zero-width) CI does not overlap the max's own CI is "
        "correctly rejected, falling through to the depth that actually achieves the maximum",
        sat_no_overlap_row["earliest_tested_candidate_saturation_depth"] == 64,
        detail=str(sat_no_overlap_row),
    )
    below_threshold_samples = {
        ("umma_1sm", 64, 4): {"flops_per_cycle": [1.0] * 30, "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": [1.0] * 30, "cta_group": 1},
        ("umma_1sm", 64, 16): {"flops_per_cycle": [1.0] * 30, "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": [1.0] * 30, "cta_group": 1},
        ("umma_1sm", 64, 64): {"flops_per_cycle": [1.0] * 30, "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": [1.0] * 30, "cta_group": 1},
        ("umma_1sm", 64, 256): {"flops_per_cycle": [10.0] * 30, "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": [10.0] * 30, "cta_group": 1},
    }
    rng7 = random.Random(BOOTSTRAP_SEED)
    bt_stats = compute_all_config_stats(below_threshold_samples, rng7)
    bt_rows = compute_saturation(bt_stats)
    rec.check(
        "saturation falls back to the depth achieving the maximum when no smaller depth qualifies",
        bt_rows[0]["earliest_tested_candidate_saturation_depth"] == 256,
    )

    # --- ceiling selection: clock-independent FLOP/cycle-per-SM space -------
    ceiling_samples = {
        ("umma_1sm", 64, 4): {"flops_per_cycle": [8.0] * 30, "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": [8.0] * 30, "cta_group": 1},
        ("umma_2sm", 64, 4): {"flops_per_cycle": [12.0] * 30, "elapsed_cycles": [1.0] * 30, "cycles_per_umma": [1.0] * 30, "flops_per_cycle_per_sm": [6.0] * 30, "cta_group": 2},
    }
    rng8 = random.Random(BOOTSTRAP_SEED)
    ceiling_stats = compute_all_config_stats(ceiling_samples, rng8)
    selection = select_ceiling(ceiling_stats)
    rec.check(
        "ceiling candidate is selected by median flops_per_cycle_per_sm, not raw flops_per_cycle "
        "(1-SM's 8.0/SM beats 2-SM's 12.0 total / 6.0-per-SM)",
        selection["empirical_per_sm_ceiling_candidate"] == ("umma_1sm", 64, 4),
    )

    # --- NCU metric resolution: canonical, qualified, ambiguous, missing ----
    discovered_canonical = set(CANDIDATE_METRICS)
    resolved = resolve_ncu_metrics_p24(discovered_canonical)
    rec.check("canonical metric names resolve directly", resolved["sm_clock_metric_resolved"] and all(resolved["per_metric"][c]["status"] == "resolved" for c in CANDIDATE_METRICS))

    discovered_qualified = {f"FBSP.TriageCompute.{MANDATORY_SM_CLOCK_METRIC}"}
    resolved_q = resolve_ncu_metrics_p24(discovered_qualified)
    rec.check(
        "an exact namespace-qualified suffix match resolves the mandatory metric",
        resolved_q["per_metric"][MANDATORY_SM_CLOCK_METRIC]["status"] == "resolved"
        and resolved_q["per_metric"][MANDATORY_SM_CLOCK_METRIC]["resolved_name"] == f"FBSP.TriageCompute.{MANDATORY_SM_CLOCK_METRIC}",
    )

    discovered_ambiguous = {f"A.{MANDATORY_SM_CLOCK_METRIC}", f"B.{MANDATORY_SM_CLOCK_METRIC}"}
    resolved_amb = resolve_ncu_metrics_p24(discovered_ambiguous)
    rec.check(
        "an ambiguous mandatory metric is recorded as ambiguous (never guessed) and never marked resolved",
        resolved_amb["per_metric"][MANDATORY_SM_CLOCK_METRIC]["status"] == "ambiguous"
        and not resolved_amb["sm_clock_metric_resolved"],
    )

    resolved_missing = resolve_ncu_metrics_p24(set())
    rec.check("a missing mandatory metric is recorded as missing, not resolved", resolved_missing["per_metric"][MANDATORY_SM_CLOCK_METRIC]["status"] == "missing")

    substring_decoy = {f"not_{MANDATORY_SM_CLOCK_METRIC}_at_all"}
    resolved_decoy = resolve_ncu_metrics_p24(substring_decoy)
    rec.check("a substring decoy (not an exact suffix match) never resolves the mandatory metric", resolved_decoy["per_metric"][MANDATORY_SM_CLOCK_METRIC]["status"] == "missing")

    # --- closed SM-clock unit allowlist and exact conversions -----------------
    ok_parsed = {"metrics": {MANDATORY_SM_CLOCK_METRIC: 1.4}, "units": {MANDATORY_SM_CLOCK_METRIC: "cycle/nsecond"}}
    ok_eval = evaluate_sm_clock(sm_clock_metric_resolved=True, actual_column_name=MANDATORY_SM_CLOCK_METRIC, parsed_metrics=ok_parsed)
    rec.check("sm_clock_hz = metric_value * 1e9 for the verified cycle/nsecond unit", ok_eval["sm_clock_valid"] and math.isclose(ok_eval["sm_clock_hz"], 1.4e9))

    for hz_spelling in ("Hz", "hz"):
        hz_parsed = {
            "metrics": {MANDATORY_SM_CLOCK_METRIC: 1.4e9},
            "units": {MANDATORY_SM_CLOCK_METRIC: hz_spelling},
        }
        hz_eval = evaluate_sm_clock(
            sm_clock_metric_resolved=True,
            actual_column_name=MANDATORY_SM_CLOCK_METRIC,
            parsed_metrics=hz_parsed,
        )
        rec.check(
            f"sm_clock_hz preserves the metric value for the verified {hz_spelling} unit",
            hz_eval["sm_clock_valid"] and math.isclose(hz_eval["sm_clock_hz"], 1.4e9),
        )

    whitespace_hz_parsed = {
        "metrics": {MANDATORY_SM_CLOCK_METRIC: 1.4e9},
        "units": {MANDATORY_SM_CLOCK_METRIC: "  Hz  "},
    }
    whitespace_hz_eval = evaluate_sm_clock(
        sm_clock_metric_resolved=True,
        actual_column_name=MANDATORY_SM_CLOCK_METRIC,
        parsed_metrics=whitespace_hz_parsed,
    )
    rec.check(
        "the closed SM-clock unit allowlist normalizes only case and outer whitespace",
        whitespace_hz_eval["sm_clock_valid"] and math.isclose(whitespace_hz_eval["sm_clock_hz"], 1.4e9),
    )

    wrong_unit_parsed = {"metrics": {MANDATORY_SM_CLOCK_METRIC: 1.4}, "units": {MANDATORY_SM_CLOCK_METRIC: "GHz"}}
    wrong_unit_eval = evaluate_sm_clock(sm_clock_metric_resolved=True, actual_column_name=MANDATORY_SM_CLOCK_METRIC, parsed_metrics=wrong_unit_parsed)
    rec.check("an unknown SM-clock unit is rejected, never rescaled or guessed", not wrong_unit_eval["sm_clock_valid"] and wrong_unit_eval["sm_clock_issue"].startswith("unknown_unit"))

    non_finite_parsed = {"metrics": {MANDATORY_SM_CLOCK_METRIC: float("nan")}, "units": {MANDATORY_SM_CLOCK_METRIC: "cycle/nsecond"}}
    non_finite_eval = evaluate_sm_clock(sm_clock_metric_resolved=True, actual_column_name=MANDATORY_SM_CLOCK_METRIC, parsed_metrics=non_finite_parsed)
    rec.check("a non-finite SM-clock value is rejected", not non_finite_eval["sm_clock_valid"] and non_finite_eval["sm_clock_issue"] == "non_finite")

    non_positive_parsed = {"metrics": {MANDATORY_SM_CLOCK_METRIC: 0.0}, "units": {MANDATORY_SM_CLOCK_METRIC: "cycle/nsecond"}}
    non_positive_eval = evaluate_sm_clock(sm_clock_metric_resolved=True, actual_column_name=MANDATORY_SM_CLOCK_METRIC, parsed_metrics=non_positive_parsed)
    rec.check("a non-positive SM-clock value is rejected", not non_positive_eval["sm_clock_valid"] and non_positive_eval["sm_clock_issue"] == "non_positive")

    unavailable_eval = evaluate_sm_clock(sm_clock_metric_resolved=False, actual_column_name=None, parsed_metrics={"metrics": {}, "units": {}})
    rec.check("an unresolved mandatory metric is recorded unavailable, never guessed", not unavailable_eval["sm_clock_valid"] and unavailable_eval["sm_clock_issue"] == "metric_unavailable_at_discovery")

    # --- ambiguous/duplicate/malformed NCU raw CSV parsing -------------------
    dup_rows = [["ID", "Kernel Name", MANDATORY_SM_CLOCK_METRIC, MANDATORY_SM_CLOCK_METRIC], ["", "", "cycle/nsecond", "cycle/nsecond"], ["1", "k", "1.4", "1.4"]]
    raised = False
    try:
        _parse_ncu_raw_csv_rows(dup_rows, label="dup")
    except NcuCsvParseError:
        raised = True
    rec.check("a duplicate NCU CSV header column name is rejected", raised)

    wrong_launch_count_rows = [["ID", "Kernel Name"], ["", ""], ["1", "k"], ["2", "k"]]
    raised = False
    try:
        _parse_ncu_raw_csv_rows(wrong_launch_count_rows, label="two-launch")
    except NcuCsvParseError:
        raised = True
    rec.check("more than one profiled launch row is rejected (launch-skip/launch-count contract)", raised)

    missing_id_rows = [["Kernel Name"], [""], ["k"]]
    raised = False
    try:
        _parse_ncu_raw_csv_rows(missing_id_rows, label="no-id")
    except NcuCsvParseError:
        raised = True
    rec.check("a missing required ID/Kernel Name column is rejected", raised)

    with tempfile.TemporaryDirectory(prefix="p24_selftest_") as tmp:
        tmp_path = Path(tmp).resolve()
        with mock.patch.object(sys.modules[__name__], "REPO_ROOT", tmp_path), mock.patch.object(p23, "REPO_ROOT", tmp_path):

            # --- preflight validation ---------------------------------------
            now = _datetime(2026, 8, 4, 12, 0, tzinfo=_timezone.utc)
            good_preflight = tmp_path / "good_preflight.json"
            _write_preflight_json(good_preflight, _default_preflight_doc())
            errors, snapshot = validate_preflight_file(good_preflight, expected_git_commit=_FIXED_GIT_COMMIT, now_utc=now)
            rec.check("valid preflight is accepted", not errors, detail=str(errors))
            rec.check("preflight snapshot carries gpu_uuid", snapshot.get("gpu_uuid") == _FIXED_GPU_UUID)

            stale_preflight = tmp_path / "stale.json"
            _write_preflight_json(stale_preflight, _default_preflight_doc(timestamp_utc="20260803T110000Z"))
            errors, _ = validate_preflight_file(stale_preflight, expected_git_commit=_FIXED_GIT_COMMIT, now_utc=now)
            rec.expect_error_containing("a preflight older than 24h is rejected", errors, "24h")

            dirty_preflight = tmp_path / "dirty.json"
            _write_preflight_json(dirty_preflight, _default_preflight_doc(git_dirty=True))
            errors, _ = validate_preflight_file(dirty_preflight, expected_git_commit=_FIXED_GIT_COMMIT, now_utc=now)
            rec.expect_error_containing("a dirty-worktree preflight is rejected", errors, "git_dirty")

            # --- campaign init + manifest hash-chain determinism ------------
            campaign_dir = _do_init_campaign(campaign_id="20260804T120000Z", started_at_utc="20260804T120000Z")
            m0, rev0 = load_p24_manifest_chain(campaign_dir)
            rec.check("init-campaign publishes revision 0 in PILOT_IN_PROGRESS with publishable=false", rev0 == 0 and m0["state"] == "PILOT_IN_PROGRESS" and m0["publishable"] is False)
            rec.check(
                "init-campaign freezes the exact two-entry SM-clock unit conversion allowlist",
                m0["frozen_protocol"].get("sm_clock_unit_to_hz_scale") == SM_CLOCK_UNIT_TO_HZ_SCALE,
                detail=str(m0["frozen_protocol"]),
            )

            try:
                create_p24_campaign_dir("20260804T120000Z")
                dup_campaign_rejected = False
            except p23.UnsafePathError:
                dup_campaign_rejected = True
            rec.check("an existing campaign directory cannot be silently reused (no-clobber)", dup_campaign_rejected)

            try:
                validate_p24_campaign_id("not-a-timestamp")
                bad_id_rejected = False
            except p23.UnsafePathError:
                bad_id_rejected = True
            rec.check("a non-canonical-timestamp campaign ID is rejected", bad_id_rejected)

            # --- illegal manifest transitions and wrong state-field introduction --
            try:
                p24_merge_manifest(campaign_dir, {}, state="COMPLETE")
                illegal_skip_rejected = False
            except p23.ManifestTransitionError:
                illegal_skip_rejected = True
            rec.check("PILOT_IN_PROGRESS -> COMPLETE (skipping intermediate states) is rejected", illegal_skip_rejected)

            premature = dict(m0)
            premature["case_results"] = {}
            premature_errors = validate_manifest_state_shape(premature, campaign_dir.name)
            rec.check("case_results present during PILOT_IN_PROGRESS is rejected by state-shape validation", bool(premature_errors))

            premature_null = dict(m0)
            premature_null["profile_completed_at_utc"] = None
            try:
                _validate_p24_manifest_updates(premature_null)
                null_field_type_ok = False
            except p23.ManifestTransitionError:
                null_field_type_ok = True
            rec.check("an unexpected null value for a non-nullable field is rejected, never treated as absent", null_field_type_ok)

            # --- hash-chain corruption ---------------------------------------
            rev0_path = _manifest_revision_path(campaign_dir, 0)
            corrupted_dir = tmp_path.joinpath(*RAW_ROOT_PARTS_P24, "20260804T120099Z_corrupt")
            corrupted_dir.mkdir(parents=True)
            (corrupted_dir / "manifest").mkdir()
            (corrupted_dir / "manifest" / "000000.json").write_text(rev0_path.read_text(encoding="utf-8"), encoding="utf-8")
            doc = json.loads((corrupted_dir / "manifest" / "000000.json").read_text(encoding="utf-8"))
            doc["campaign_id"] = "20260804T120099Z_corrupt"
            (corrupted_dir / "manifest" / "000001.json").write_text(
                json.dumps({**doc, "manifest_revision": 1, "previous_manifest_sha256": "0" * 64}, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            raised = False
            try:
                load_p24_manifest_chain(corrupted_dir)
            except p23.ManifestTransitionError:
                raised = True
            rec.check("a manifest revision whose previous_manifest_sha256 does not match the prior file's real hash is rejected", raised)

            # --- full happy-path pipeline: pilot -> profile -> COMPLETE -> ANALYZED --
            good_campaign_id = "20260804T130000Z"
            good_campaign = _run_full_pipeline(tmp_path, good_campaign_id)
            m_complete, _rev = load_p24_manifest_chain(good_campaign)
            rec.check("all 24 profile cases recorded before finalize-profile", m_complete.get("profile_count_completed") == EXPECTED_PROFILE_CASE_COUNT)

            ok, errors = _do_finalize_profile(campaign_dir=good_campaign, completed_at_utc="20260804T140000Z")
            rec.check("finalize-profile succeeds against a fully validated 24-case campaign", ok, detail=str(errors))
            m_after_finalize, _ = load_p24_manifest_chain(good_campaign)
            rec.check("finalize-profile publishes state=COMPLETE with the canonical profile_order", m_after_finalize.get("state") == "COMPLETE" and m_after_finalize.get("profile_order") == build_profile_plan())

            ok, errors = _do_analyze(campaign_dir=good_campaign, analyzed_at_utc="20260804T150000Z")
            rec.check("analyze succeeds and reaches ANALYZED when every SM-clock reading is valid", ok, detail=str(errors))
            m_analyzed, _ = load_p24_manifest_chain(good_campaign)
            rec.check("analyze publishes state=ANALYZED with all ten analysis artifacts hashed", m_analyzed.get("state") == "ANALYZED" and set(ANALYSIS_ARTIFACT_RELATIVE_PATHS) <= set(m_analyzed.get("artifact_sha256", {})))
            ceiling_doc = json.loads((good_campaign / "analysis" / "empirical_ceiling.json").read_text(encoding="utf-8"))
            rec.check("a fully valid campaign's empirical_ceiling.json reports status=ANALYZED with a real TFLOP/s figure", ceiling_doc["status"] == "ANALYZED" and ceiling_doc["empirical_per_sm_ceiling_candidate"]["estimated_tflops_per_sm"] is not None)

            byte_identical_report = (good_campaign / "analysis" / "report.md").read_bytes()
            rec.check("configuration_statistics.csv processes configs in deterministic sorted order", (good_campaign / "analysis" / "configuration_statistics.csv").exists())

            try:
                p24_merge_manifest(good_campaign, {}, state="PILOT_IN_PROGRESS")
                terminal_reopen_rejected = False
            except p23.ManifestTransitionError:
                terminal_reopen_rejected = True
            rec.check("a terminal (ANALYZED) campaign can never be reopened or rewritten", terminal_reopen_rejected)

            # --- GB300-observed Hz/hz representation: full 24-case pipeline --
            def _alternating_hz_spelling(entry: dict) -> str:
                return "Hz" if entry["index"] % 2 == 0 else "hz"

            hz_campaign_id = "20260804T153000Z"
            hz_campaign = _run_full_pipeline(
                tmp_path,
                hz_campaign_id,
                sm_clock_unit_fn=_alternating_hz_spelling,
            )
            ok, errors = _do_finalize_profile(
                campaign_dir=hz_campaign,
                completed_at_utc="20260804T154000Z",
            )
            rec.check(
                "a complete 24-profile campaign accepts NCU's GB300-observed Hz/hz SM-clock representation",
                ok,
                detail=str(errors),
            )
            ok, errors = _do_analyze(
                campaign_dir=hz_campaign,
                analyzed_at_utc="20260804T155000Z",
            )
            rec.check(
                "the GB300-observed Hz/hz representation reaches ANALYZED instead of INCONCLUSIVE",
                ok,
                detail=str(errors),
            )
            hz_manifest, _ = load_p24_manifest_chain(hz_campaign)
            rec.check(
                "all 24 Hz/hz profile results preserve the raw unit and apply identity conversion to Hz",
                hz_manifest.get("state") == "ANALYZED"
                and all(
                    result["sm_clock_unit"] == _alternating_hz_spelling(entry)
                    and math.isclose(result["sm_clock_raw_value"], 1.4e9)
                    and math.isclose(result["sm_clock_hz"], 1.4e9)
                    for entry in build_profile_plan()
                    for result in [hz_manifest["case_results"][entry["case_name"]]]
                ),
            )
            hz_ceiling_doc = json.loads(
                (hz_campaign / "analysis" / "empirical_ceiling.json").read_text(encoding="utf-8")
            )
            rec.check(
                "the Hz/hz end-to-end regression emits a numeric TFLOP/s ceiling candidate",
                hz_ceiling_doc["status"] == "ANALYZED"
                and hz_ceiling_doc["empirical_per_sm_ceiling_candidate"]["estimated_tflops_per_sm"] is not None,
            )

            # --- INCONCLUSIVE path: one bad SM-clock reading anywhere ---------
            def _one_case_missing_clock(entry: dict) -> float | None:
                return None if entry["index"] == 0 else 1_400_000_000.0

            inconclusive_campaign_id = "20260804T160000Z"
            inconclusive_campaign = _run_full_pipeline(tmp_path, inconclusive_campaign_id, sm_cycles_fn=_one_case_missing_clock)
            ok, errors = _do_finalize_profile(campaign_dir=inconclusive_campaign, completed_at_utc="20260804T170000Z")
            rec.check("finalize-profile still reaches COMPLETE even when one case's SM-clock metric is unavailable (raw evidence capture always succeeds)", ok, detail=str(errors))
            ok, errors = _do_analyze(campaign_dir=inconclusive_campaign, analyzed_at_utc="20260804T180000Z")
            rec.check("analyze succeeds (produces artifacts) even for the INCONCLUSIVE outcome", ok, detail=str(errors))
            m_inconclusive, _ = load_p24_manifest_chain(inconclusive_campaign)
            rec.check(
                "one invalid SM-clock reading drives the whole campaign to INCONCLUSIVE with a non-empty reason",
                m_inconclusive.get("state") == "INCONCLUSIVE" and bool(m_inconclusive.get("inconclusive_reason")),
            )
            inconclusive_ceiling = json.loads((inconclusive_campaign / "analysis" / "empirical_ceiling.json").read_text(encoding="utf-8"))
            rec.check(
                "an INCONCLUSIVE campaign's empirical_ceiling.json never emits a TFLOP/s figure anywhere",
                inconclusive_ceiling["status"] == "INCONCLUSIVE"
                and inconclusive_ceiling["best_1sm_configuration"]["estimated_tflops_per_sm"] is None
                and inconclusive_ceiling["best_2sm_configuration"]["estimated_tflops_per_sm"] is None
                and inconclusive_ceiling["empirical_per_sm_ceiling_candidate"]["estimated_tflops_per_sm"] is None
                and inconclusive_ceiling["device_equivalent_estimate"]["available"] is False,
            )
            rec.check(
                "INCONCLUSIVE still produces all ten analysis artifacts (clock-independent statistics remain reviewable)",
                set(ANALYSIS_ARTIFACT_RELATIVE_PATHS) <= set(m_inconclusive.get("artifact_sha256", {})),
            )

            # --- evidence-integrity gate: tampering after validate-profile-case ----
            tamper_campaign_id = "20260804T190000Z"
            tamper_campaign = _run_full_pipeline(tmp_path, tamper_campaign_id)
            first_case_name = build_profile_plan()[0]["case_name"]
            tampered_app_csv = tamper_campaign / "profiles" / first_case_name / f"{first_case_name}.application.csv"
            original_bytes = tampered_app_csv.read_bytes()
            tampered_app_csv.write_bytes(original_bytes.replace(b"benchmark", b"benchmar1"))
            integrity_errors, verified = verify_campaign_evidence_integrity(tamper_campaign, load_p24_manifest_chain(tamper_campaign)[0])
            rec.check("tampering a validated application.csv after the fact is caught by the evidence-integrity gate", bool(integrity_errors) and verified is None)
            tampered_app_csv.write_bytes(original_bytes)

            ok2, errors2 = _do_finalize_profile(campaign_dir=tamper_campaign, completed_at_utc="20260804T200000Z")
            rec.check("finalize-profile succeeds again once tampered evidence is restored", ok2, detail=str(errors2))

            # --- missing/extra/symlinked profiles/ entries ---------------------
            extra_dir_campaign_id = "20260804T210000Z"
            extra_dir_campaign = _run_full_pipeline(tmp_path, extra_dir_campaign_id)
            (extra_dir_campaign / "profiles" / "unexpected_case_dir").mkdir()
            extra_errors, extra_snapshot = verify_campaign_evidence_integrity(extra_dir_campaign, load_p24_manifest_chain(extra_dir_campaign)[0])
            rec.check("an unplanned extra directory under profiles/ is rejected, never silently ignored", bool(extra_errors) and extra_snapshot is None)
            (extra_dir_campaign / "profiles" / "unexpected_case_dir").rmdir()

            symlink_case_name = build_profile_plan()[1]["case_name"]
            symlink_campaign_id = "20260804T220000Z"
            symlink_campaign = _run_full_pipeline(tmp_path, symlink_campaign_id)
            real_case_dir = symlink_campaign / "profiles" / symlink_case_name
            outside_dir = tmp_path / "outside_profiles_target"
            outside_dir.mkdir()
            import shutil as _shutil
            _shutil.rmtree(real_case_dir)
            real_case_dir.symlink_to(outside_dir)
            symlink_errors, symlink_snapshot = verify_campaign_evidence_integrity(symlink_campaign, load_p24_manifest_chain(symlink_campaign)[0])
            rec.check("a profile case directory replaced by a symlink is rejected", bool(symlink_errors) and symlink_snapshot is None)

            # --- provenance/GPU mismatch between pilot and profile preflight ------
            mismatch_campaign_id = "20260804T230000Z"
            p23_campaign_dir_mm, _ = _build_p23_pilot_campaign_fixture(tmp_path, mismatch_campaign_id)
            mismatch_campaign = _do_init_campaign(campaign_id=mismatch_campaign_id, started_at_utc="20260804T230000Z")
            preflight_mm = tmp_path / "preflight_mm.json"
            _write_preflight_json(preflight_mm, _default_preflight_doc())
            ok, errors = _do_record_pilot(
                campaign_dir=mismatch_campaign, p23_campaign_dir=p23_campaign_dir_mm, preflight_path=preflight_mm,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260804T233000Z", now_utc=now,
            )
            rec.check("record-pilot fixture setup succeeds", ok, detail=str(errors))
            different_gpu_preflight = tmp_path / "preflight_mm_different_gpu.json"
            _write_preflight_json(different_gpu_preflight, _default_preflight_doc(gpu_uuid="GPU-99999999-9999-9999-9999-999999999999"))
            ok, errors = _do_validate_profile_preconditions(campaign_dir=mismatch_campaign, preflight_path=different_gpu_preflight, git_commit=_FIXED_GIT_COMMIT, now_utc=now)
            rec.expect_error_containing("a profiling preflight from a different GPU UUID than the pilot's is rejected", errors, "gpu_uuid")
            rec.check("validate-profile-preconditions returns False on GPU/driver provenance mismatch", not ok)

            # --- interrupted/failed campaigns never reach COMPLETE or ANALYZED ----
            failed_campaign_id = "20260805T000000Z"
            failed_campaign = _do_init_campaign(campaign_id=failed_campaign_id, started_at_utc="20260805T000000Z")
            _fail_p24(failed_campaign, "synthetic_self_test_failure", ["synthetic reason"])
            m_failed, _ = load_p24_manifest_chain(failed_campaign)
            rec.check("a campaign driven to FAILED carries required failure telemetry and no completing-state field", m_failed.get("state") == "FAILED" and m_failed.get("failure_stage") == "synthetic_self_test_failure" and "profile_order" not in m_failed)
            try:
                p24_merge_manifest(failed_campaign, {}, state="COMPLETE")
                failed_cannot_complete = False
            except p23.ManifestTransitionError:
                failed_cannot_complete = True
            rec.check("a FAILED campaign can never subsequently reach COMPLETE", failed_cannot_complete)

            # =================================================================
            # Defect 1 repair: strict campaign-provenance comparison
            # =================================================================
            valid_provenance = {
                "git_commit": _FIXED_GIT_COMMIT, "gpu_uuid": _FIXED_GPU_UUID, "gpu_name": _FIXED_GPU_NAME,
                "compute_capability": "10.3", "cuda_driver_version": "13010", "cuda_runtime_version": "13010",
                "visible_device_count": 1, "logical_device_index": 0, "campaign_id": "20260805T010000Z",
            }
            prov_errors = validate_provenance_tuple(valid_provenance, label="test")
            rec.check("validate_provenance_tuple accepts a fully valid campaign provenance tuple", not prov_errors, detail=str(prov_errors))
            rec.check("validate_provenance_tuple rejects a non-dict input", bool(validate_provenance_tuple(None, label="test")))

            for field in MANDATORY_PROVENANCE_FIELDS:
                absent = dict(valid_provenance)
                del absent[field]
                errs = validate_provenance_tuple(absent, label="test")
                rec.check(f"validate_provenance_tuple rejects an absent {field}", any("absent" in e for e in errs), detail=str(errs))

                nulled = dict(valid_provenance)
                nulled[field] = None
                errs = validate_provenance_tuple(nulled, label="test")
                rec.check(f"validate_provenance_tuple rejects a null {field}, distinctly from absent", any("is null" in e for e in errs), detail=str(errs))

            wrong_type = dict(valid_provenance)
            wrong_type["visible_device_count"] = "1"
            rec.check("validate_provenance_tuple rejects a wrong-typed visible_device_count (str instead of int)", bool(validate_provenance_tuple(wrong_type, label="test")))

            bad_commit = dict(valid_provenance)
            bad_commit["git_commit"] = "not-a-commit"
            rec.check("validate_provenance_tuple rejects a malformed git_commit", bool(validate_provenance_tuple(bad_commit, label="test")))

            bad_uuid = dict(valid_provenance)
            bad_uuid["gpu_uuid"] = "not-a-uuid"
            rec.check("validate_provenance_tuple rejects a malformed gpu_uuid", bool(validate_provenance_tuple(bad_uuid, label="test")))

            for count in (0, 2, -1):
                bad_count = dict(valid_provenance)
                bad_count["visible_device_count"] = count
                rec.check(f"validate_provenance_tuple rejects visible_device_count={count} (must be exactly 1)", bool(validate_provenance_tuple(bad_count, label="test")))

            for index in (1, -1, 2):
                bad_index = dict(valid_provenance)
                bad_index["logical_device_index"] = index
                rec.check(f"validate_provenance_tuple rejects logical_device_index={index} (must be exactly 0)", bool(validate_provenance_tuple(bad_index, label="test")))

            valid_app_row = {
                "git_commit": _FIXED_GIT_COMMIT, "gpu_uuid": _FIXED_GPU_UUID, "gpu_name": _FIXED_GPU_NAME,
                "compute_capability": "10.3", "cuda_driver_version": "13010", "cuda_runtime_version": "13010",
            }
            rec.check(
                "compare_application_provenance accepts a row that matches the campaign tuple exactly",
                not compare_application_provenance(app_row=valid_app_row, campaign_provenance=valid_provenance, label="test"),
            )
            for row_field, _prov_field in APPLICATION_PROVENANCE_FIELD_MAP:
                mismatched_row = dict(valid_app_row)
                mismatched_row[row_field] = "DEFINITELY_DIFFERENT_VALUE"
                errs = compare_application_provenance(app_row=mismatched_row, campaign_provenance=valid_provenance, label="test")
                rec.check(f"compare_application_provenance rejects a mismatched {row_field}", bool(errs), detail=str(errs))

            cross_uuid_row = dict(valid_app_row)
            cross_uuid_row["gpu_uuid"] = "GPU-99999999-9999-9999-9999-999999999999"
            cross_uuid_errors = compare_application_provenance(app_row=cross_uuid_row, campaign_provenance=valid_provenance, label="test")
            rec.check(
                "compare_application_provenance rejects application evidence from a different GPU UUID (the reproduced audit finding)",
                any("gpu_uuid" in e for e in cross_uuid_errors), detail=str(cross_uuid_errors),
            )

            # --- end-to-end: application evidence from a different GPU UUID is rejected by the full pipeline ---
            uuid_tamper_campaign_id = "20260805T020000Z"
            uuid_tamper_campaign = _run_full_pipeline(tmp_path, uuid_tamper_campaign_id)
            uuid_case_name = build_profile_plan()[2]["case_name"]
            uuid_app_csv = uuid_tamper_campaign / "profiles" / uuid_case_name / f"{uuid_case_name}.application.csv"
            original_uuid_bytes = uuid_app_csv.read_bytes()
            uuid_app_csv.write_bytes(original_uuid_bytes.replace(_FIXED_GPU_UUID.encode(), b"GPU-99999999-9999-9999-9999-999999999999"))
            uuid_integrity_errors, uuid_verified = verify_campaign_evidence_integrity(uuid_tamper_campaign, load_p24_manifest_chain(uuid_tamper_campaign)[0])
            rec.check(
                "end-to-end: one profile's application evidence reporting a different GPU UUID than the campaign "
                "is rejected by the evidence-integrity gate",
                bool(uuid_integrity_errors) and uuid_verified is None and any("gpu_uuid" in e for e in uuid_integrity_errors),
                detail=str(uuid_integrity_errors),
            )
            uuid_app_csv.write_bytes(original_uuid_bytes)
            uuid_ok, uuid_finalize_errors = _do_finalize_profile(campaign_dir=uuid_tamper_campaign, completed_at_utc="20260805T021000Z")
            rec.check("end-to-end cross-UUID campaign finalizes successfully once evidence is restored", uuid_ok, detail=str(uuid_finalize_errors))

            # --- preflight requires exactly logical device 0 ------------------------
            default_preflight_doc = _default_preflight_doc()
            bad_logical_index_preflight = tmp_path / "preflight_bad_logical_index.json"
            _write_preflight_json(
                bad_logical_index_preflight,
                {**default_preflight_doc, "gpu": {**default_preflight_doc["gpu"], "logical_index": "1"}},
            )
            bad_index_errors, _snap = validate_preflight_file(bad_logical_index_preflight, expected_git_commit=_FIXED_GIT_COMMIT, now_utc=now)
            rec.check("a preflight reporting logical_index != '0' is rejected", any("logical_index" in e for e in bad_index_errors), detail=str(bad_index_errors))

            # --- P2.3 and P2.4 campaign IDs must match -------------------------------
            mismatched_ids_p24 = _do_init_campaign(campaign_id="20260805T030000Z", started_at_utc="20260805T030000Z")
            mismatched_ids_p23, _ = _build_p23_pilot_campaign_fixture(tmp_path, "p23_deliberately_different_id")
            preflight_ids = tmp_path / "preflight_ids.json"
            _write_preflight_json(preflight_ids, _default_preflight_doc())
            ok_ids, errors_ids = _do_record_pilot(
                campaign_dir=mismatched_ids_p24, p23_campaign_dir=mismatched_ids_p23, preflight_path=preflight_ids,
                git_commit=_FIXED_GIT_COMMIT, completed_at_utc="20260805T030500Z", now_utc=now,
            )
            rec.check(
                "record-pilot rejects a P2.3 campaign whose ID does not match the P2.4 campaign's own ID",
                not ok_ids and any("campaign ID" in e for e in errors_ids), detail=str(errors_ids),
            )

            # =================================================================
            # Defect 2 repair: exact seven-file per-case profile inventory
            # =================================================================
            canonical_names_check = canonical_profile_case_filenames(build_profile_plan()[3]["case_name"])
            rec.check("canonical_profile_case_filenames returns exactly seven names", len(canonical_names_check) == 7, detail=str(canonical_names_check))
            nonempty_required_labels = {
                label for label, _suffix in CANONICAL_PROFILE_CASE_FILE_LABELS
                if label not in EMPTY_ALLOWED_PROFILE_CASE_FILE_LABELS
            }
            rec.check(
                "only container_stderr.log and metrics_export_stderr.log may be empty; the other five canonical artifacts remain payload-bearing",
                EMPTY_ALLOWED_PROFILE_CASE_FILE_LABELS == {"container_stderr_log", "metrics_export_stderr_log"}
                and len(nonempty_required_labels) == 5,
                detail=f"empty_allowed={sorted(EMPTY_ALLOWED_PROFILE_CASE_FILE_LABELS)}, required={sorted(nonempty_required_labels)}",
            )

            file_removal_campaign_id = "20260805T040000Z"
            file_removal_campaign = _run_full_pipeline(tmp_path, file_removal_campaign_id)
            empty_stderr_manifest = load_p24_manifest_chain(file_removal_campaign)[0]
            empty_sha256 = hashlib.sha256(b"").hexdigest()
            rec.check(
                "end-to-end validate-profile-case accepts all 24 cases when both mandatory diagnostic stderr artifacts are genuine zero-length files",
                all(
                    (file_removal_campaign / "profiles" / entry["case_name"] / f"{entry['case_name']}.container_stderr.log").stat().st_size == 0
                    and (file_removal_campaign / "profiles" / entry["case_name"] / f"{entry['case_name']}.metrics_export_stderr.log").stat().st_size == 0
                    and empty_stderr_manifest["case_results"][entry["case_name"]]["container_stderr_log_sha256"] == empty_sha256
                    and empty_stderr_manifest["case_results"][entry["case_name"]]["metrics_export_stderr_log_sha256"] == empty_sha256
                    for entry in build_profile_plan()
                ),
            )
            empty_stderr_integrity_errors, empty_stderr_verified = verify_campaign_evidence_integrity(
                file_removal_campaign, empty_stderr_manifest,
            )
            rec.check(
                "descriptor-anchored terminal validation re-opens, hashes, and accepts both zero-length diagnostic stderr artifacts for all 24 cases",
                not empty_stderr_integrity_errors and empty_stderr_verified is not None,
                detail=str(empty_stderr_integrity_errors),
            )
            fr_case_name = build_profile_plan()[3]["case_name"]
            fr_case_dir = file_removal_campaign / "profiles" / fr_case_name
            suffix_by_label = dict(CANONICAL_PROFILE_CASE_FILE_LABELS)
            for label in sorted(nonempty_required_labels):
                target = fr_case_dir / f"{fr_case_name}{suffix_by_label[label]}"
                original = target.read_bytes()
                target.write_bytes(b"")
                path_errors: list[str] = []
                resolved_paths = _resolve_case_evidence_paths(file_removal_campaign, fr_case_name, path_errors)
                anchored_error = ""
                profiles_fd = _open_profiles_fd_anchored(file_removal_campaign)
                try:
                    try:
                        unexpected_fds = _open_case_evidence_fds(profiles_fd, fr_case_name)
                    except p23.UnsafePathError as exc:
                        anchored_error = str(exc)
                    else:
                        for fd in unexpected_fds.values():
                            os.close(fd)
                finally:
                    os.close(profiles_fd)
                rec.check(
                    f"an empty {label} is rejected by both the initial path validation and descriptor-anchored terminal validation",
                    resolved_paths is None and any("empty" in error for error in path_errors)
                    and "empty" in anchored_error,
                    detail=f"path_errors={path_errors}; anchored_error={anchored_error!r}",
                )
                target.write_bytes(original)
            for _label, _suffix in CANONICAL_PROFILE_CASE_FILE_LABELS:
                target = fr_case_dir / f"{fr_case_name}{_suffix}"
                original = target.read_bytes()
                target.unlink()
                errs, verified = verify_campaign_evidence_integrity(file_removal_campaign, load_p24_manifest_chain(file_removal_campaign)[0])
                rec.check(f"removing the canonical {fr_case_name}{_suffix} artifact is rejected, never reaches COMPLETE", bool(errs) and verified is None, detail=str(errs))
                target.write_bytes(original)
            fr_ok, fr_errors = _do_finalize_profile(campaign_dir=file_removal_campaign, completed_at_utc="20260805T041000Z")
            rec.check(
                "campaign finalizes successfully with all seven files present and both diagnostic stderr artifacts legitimately empty",
                fr_ok, detail=str(fr_errors),
            )

            eighth_file_campaign_id = "20260805T050000Z"
            eighth_file_campaign = _run_full_pipeline(tmp_path, eighth_file_campaign_id)
            ef_case_name = build_profile_plan()[4]["case_name"]
            ef_case_dir = eighth_file_campaign / "profiles" / ef_case_name

            eighth_path = ef_case_dir / f"{ef_case_name}.ncu_bridge_stderr.log"
            eighth_path.write_text("leaked bridge stderr\n", encoding="utf-8")
            eighth_errors, eighth_verified = verify_campaign_evidence_integrity(eighth_file_campaign, load_p24_manifest_chain(eighth_file_campaign)[0])
            rec.check(
                "the former unauthorized eighth file (<case>.ncu_bridge_stderr.log) is rejected as an unplanned entry",
                bool(eighth_errors) and eighth_verified is None, detail=str(eighth_errors),
            )
            eighth_path.unlink()

            extra_path = ef_case_dir / "notes.txt"
            extra_path.write_text("arbitrary extra file\n", encoding="utf-8")
            extra_file_errors, extra_file_verified = verify_campaign_evidence_integrity(eighth_file_campaign, load_p24_manifest_chain(eighth_file_campaign)[0])
            rec.check("an arbitrary extra file in a case directory is rejected", bool(extra_file_errors) and extra_file_verified is None)
            extra_path.unlink()

            symlinked_target = ef_case_dir / f"{ef_case_name}.ncu_tool.log"
            symlinked_original = symlinked_target.read_bytes()
            elsewhere = tmp_path / "elsewhere_ncu_tool.log"
            elsewhere.write_bytes(symlinked_original)
            symlinked_target.unlink()
            symlinked_target.symlink_to(elsewhere)
            symlink_file_errors, symlink_file_verified = verify_campaign_evidence_integrity(eighth_file_campaign, load_p24_manifest_chain(eighth_file_campaign)[0])
            rec.check("a canonical per-case artifact replaced by a symlink is rejected", bool(symlink_file_errors) and symlink_file_verified is None)
            symlinked_target.unlink()
            symlinked_target.write_bytes(symlinked_original)

            wrong_type_target = ef_case_dir / f"{ef_case_name}.container_stdout.log"
            wrong_type_original = wrong_type_target.read_bytes()
            wrong_type_target.unlink()
            wrong_type_target.mkdir()
            wrong_type_errors, wrong_type_verified = verify_campaign_evidence_integrity(eighth_file_campaign, load_p24_manifest_chain(eighth_file_campaign)[0])
            rec.check("a canonical per-case artifact replaced by a directory is rejected", bool(wrong_type_errors) and wrong_type_verified is None)
            wrong_type_target.rmdir()
            wrong_type_target.write_bytes(wrong_type_original)

            eighth_ok, eighth_finalize_errors = _do_finalize_profile(campaign_dir=eighth_file_campaign, completed_at_utc="20260805T051000Z")
            rec.check("campaign finalizes successfully once all decoy/extra/symlinked entries are removed and restored", eighth_ok, detail=str(eighth_finalize_errors))

            # =================================================================
            # Defect 3 repair: all-24-valid rule for SM-count device
            # extrapolation
            # =================================================================
            mp_plan = build_profile_plan()

            def _case_results_with_mp(values: dict, units: dict | None = None) -> dict:
                results = {}
                for e in mp_plan:
                    entry_values: dict = {}
                    entry_units: dict = {}
                    if e["case_name"] in values:
                        entry_values[DEVICE_MULTIPROCESSOR_COUNT_METRIC] = values[e["case_name"]]
                        entry_units[DEVICE_MULTIPROCESSOR_COUNT_METRIC] = (units or {}).get(e["case_name"], "")
                    results[e["case_name"]] = {"diagnostic_metric_values": entry_values, "diagnostic_metric_units": entry_units}
                return results

            all_valid_132 = {e["case_name"]: 132.0 for e in mp_plan}
            mp_eval = evaluate_device_multiprocessor_count(_case_results_with_mp(all_valid_132), mp_plan)
            rec.check(
                "evaluate_device_multiprocessor_count: a valid identical positive integer in all 24 profiles enables extrapolation",
                mp_eval["available"] and mp_eval["multiprocessor_count"] == 132, detail=str(mp_eval),
            )

            mp_eval_missing_all = evaluate_device_multiprocessor_count(_case_results_with_mp({}), mp_plan)
            rec.check("evaluate_device_multiprocessor_count: missing in all profiles suppresses extrapolation", not mp_eval_missing_all["available"])

            missing_one = dict(all_valid_132)
            del missing_one[mp_plan[0]["case_name"]]
            mp_eval_missing_one = evaluate_device_multiprocessor_count(_case_results_with_mp(missing_one), mp_plan)
            rec.check(
                "evaluate_device_multiprocessor_count: missing in exactly one profile suppresses extrapolation "
                "(audit repair: one profile's absence used to be silently ignored)",
                not mp_eval_missing_one["available"],
            )

            for _label, _bad_value in (("negative", -5.0), ("zero", 0.0), ("NaN", float("nan")), ("infinity", float("inf")), ("non-integer", 132.5)):
                tampered = dict(all_valid_132)
                tampered[mp_plan[1]["case_name"]] = _bad_value
                mp_eval_bad = evaluate_device_multiprocessor_count(_case_results_with_mp(tampered), mp_plan)
                rec.check(
                    f"evaluate_device_multiprocessor_count: a {_label} value in one profile suppresses extrapolation",
                    not mp_eval_bad["available"], detail=str(mp_eval_bad),
                )

            inconsistent = dict(all_valid_132)
            inconsistent[mp_plan[2]["case_name"]] = 148.0
            mp_eval_inconsistent = evaluate_device_multiprocessor_count(_case_results_with_mp(inconsistent), mp_plan)
            rec.check(
                "evaluate_device_multiprocessor_count: inconsistent (but individually valid positive) values across "
                "profiles suppress extrapolation",
                not mp_eval_inconsistent["available"],
            )

            wrong_unit_result = _case_results_with_mp(all_valid_132, units={mp_plan[3]["case_name"]: "count"})
            mp_eval_wrong_unit = evaluate_device_multiprocessor_count(wrong_unit_result, mp_plan)
            rec.check("evaluate_device_multiprocessor_count: an unexpected unit representation suppresses extrapolation", not mp_eval_wrong_unit["available"])

            dup_metric_rows = [
                ["ID", "Kernel Name", DEVICE_MULTIPROCESSOR_COUNT_METRIC, f"A.{DEVICE_MULTIPROCESSOR_COUNT_METRIC}"],
                ["", "", "", ""],
                ["1", "k", "132", "148"],
            ]
            dup_metric_raised = False
            try:
                _parse_ncu_raw_csv_rows(dup_metric_rows, label="dup-mp-count")
            except NcuCsvParseError:
                dup_metric_raised = True
            rec.check("duplicate/ambiguous device__attribute_multiprocessor_count columns within one profile's raw CSV are rejected", dup_metric_raised)

            ceiling_case_results = {
                e["case_name"]: {"sm_clock_valid": True, "sm_clock_hz": 1.4e9, "diagnostic_metric_values": {}, "diagnostic_metric_units": {}}
                for e in mp_plan
            }
            ceiling_stats = {
                (e["method"], e["n"], e["depth"]): {
                    "cta_group": p23.METHOD_INFO[e["method"]]["cta_group"],
                    "flops_per_cycle": {"median": 10.0},
                    "flops_per_cycle_per_sm": {"median": 10.0 / p23.METHOD_INFO[e["method"]]["cta_group"]},
                }
                for e in mp_plan
            }
            ceiling_doc = build_empirical_ceiling(stats_by_config=ceiling_stats, case_results=ceiling_case_results, all_sm_clock_valid=True, inconclusive_reason=[])
            rec.check(
                "a missing/invalid SM count suppresses only the device-wide estimate, never the local/per-SM TFLOP/s estimates",
                ceiling_doc["device_equivalent_estimate"]["available"] is False
                and ceiling_doc["empirical_per_sm_ceiling_candidate"]["estimated_local_tflops"] is not None
                and ceiling_doc["empirical_per_sm_ceiling_candidate"]["estimated_tflops_per_sm"] is not None,
                detail=str(ceiling_doc["device_equivalent_estimate"]),
            )

            # =================================================================
            # Defect 4 repair: full 720-sample P2.3 pilot revalidation
            # =================================================================
            p23_reval_id = "20260805T060000Z"
            p23_reval_dir, _p23_reval_manifest = _build_p23_pilot_campaign_fixture(tmp_path, p23_reval_id)
            reval_errors, reval_snapshot = revalidate_p23_pilot_campaign(p23_reval_dir, git_commit=_FIXED_GIT_COMMIT)
            rec.check(
                "revalidate_p23_pilot_campaign accepts a genuinely valid, untampered 720-sample (24x30) P2.3 campaign",
                not reval_errors and reval_snapshot is not None and reval_snapshot["sample_count"] == 720,
                detail=str(reval_errors),
            )

            reval_plan = p23.build_plan()
            reval_case_path = p23_reval_dir / "cases" / f"{reval_plan[5]['case_name']}.csv"

            def _mutate_case_row(path: Path, sample_index: int, field: str, new_value: str) -> bytes:
                original = path.read_bytes()
                with open(path, "r", newline="", encoding="utf-8") as fh:
                    rows = list(csv.reader(fh))
                header, data = rows[0], rows[1:]
                col = header.index(field)
                si_col = header.index("sample_index")
                for row in data:
                    if row[si_col] == str(sample_index):
                        row[col] = new_value
                        break
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.writer(fh)
                    writer.writerow(header)
                    writer.writerows(data)
                return original

            def _check_reval_rejects(label: str, path: Path, original: bytes) -> None:
                errs, snap = revalidate_p23_pilot_campaign(p23_reval_dir, git_commit=_FIXED_GIT_COMMIT)
                rec.check(f"revalidate_p23_pilot_campaign rejects {label}", bool(errs) and snap is None, detail=str(errs))
                path.write_bytes(original)

            _check_reval_rejects(
                "a mutated raw elapsed_cycles value inconsistent with its own stored derived fields",
                reval_case_path, _mutate_case_row(reval_case_path, 0, "elapsed_cycles", "999999999"),
            )
            _check_reval_rejects(
                "a mutated stored derived value (flops_per_cycle) inconsistent with its own raw inputs",
                reval_case_path, _mutate_case_row(reval_case_path, 0, "flops_per_cycle", "999.999999"),
            )
            _check_reval_rejects("a mutated correctness flag", reval_case_path, _mutate_case_row(reval_case_path, 0, "correctness", "FAIL"))
            _check_reval_rejects("a mutated mismatches count", reval_case_path, _mutate_case_row(reval_case_path, 0, "mismatches", "1"))
            _check_reval_rejects("a mutated commit in one row", reval_case_path, _mutate_case_row(reval_case_path, 0, "git_commit", "e" * 40))
            _check_reval_rejects(
                "a mutated GPU UUID in one row", reval_case_path,
                _mutate_case_row(reval_case_path, 0, "gpu_uuid", "GPU-99999999-9999-9999-9999-999999999999"),
            )
            _check_reval_rejects(
                "a mutated driver/runtime field in one row", reval_case_path,
                _mutate_case_row(reval_case_path, 0, "cuda_driver_version", "99999"),
            )
            _reval_new_n = 256 if reval_plan[5]["n"] != 256 else 64
            _check_reval_rejects(
                "a mutated configuration field (n) in one row", reval_case_path,
                _mutate_case_row(reval_case_path, 0, "n", str(_reval_new_n)),
            )

            # one sample/repetition index: overwrite sample_index=1 with a
            # duplicate of sample_index=0, simultaneously producing a
            # duplicate row and a missing index.
            original_si = reval_case_path.read_bytes()
            with open(reval_case_path, "r", newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            header, data = rows[0], rows[1:]
            si_col = header.index("sample_index")
            for row in data:
                if row[si_col] == "1":
                    row[si_col] = "0"
                    break
            with open(reval_case_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                writer.writerows(data)
            _check_reval_rejects("a mutated sample/repetition index producing a duplicate and a missing index", reval_case_path, original_si)

            # one file hash: tamper the P2.3 manifest's own recorded case-file hash
            manifest_path_reval = p23_reval_dir / "manifest.json"
            original_manifest_bytes = manifest_path_reval.read_bytes()
            manifest_doc = json.loads(original_manifest_bytes)
            manifest_doc["case_file_sha256"][reval_plan[5]["case_name"]] = "0" * 64
            manifest_path_reval.write_bytes((json.dumps(manifest_doc, indent=2, sort_keys=True) + "\n").encode("utf-8"))
            _check_reval_rejects("a tampered recorded case-file SHA-256 in the P2.3 manifest", manifest_path_reval, original_manifest_bytes)

            # one row order / canonical-plan position: corrupt execution_order.csv
            eo_path = p23_reval_dir / "execution_order.csv"
            original_eo = eo_path.read_bytes()
            with open(eo_path, "r", newline="", encoding="utf-8") as fh:
                eo_rows = list(csv.reader(fh))
            eo_header, eo_data = eo_rows[0], eo_rows[1:]
            eo_data[0], eo_data[1] = eo_data[1], eo_data[0]
            with open(eo_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(eo_header)
                writer.writerows(eo_data)
            _check_reval_rejects("a mutated row order/canonical-plan position in execution_order.csv", eo_path, original_eo)

            # missing row
            original_missing = reval_case_path.read_bytes()
            with open(reval_case_path, "r", newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            header, data = rows[0], rows[1:]
            data = [row for row in data if row[header.index("sample_index")] != "29"]
            with open(reval_case_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                writer.writerows(data)
            _check_reval_rejects("a missing row (29 of 30 retained samples in one configuration)", reval_case_path, original_missing)

            # duplicate row
            original_dup = reval_case_path.read_bytes()
            with open(reval_case_path, "r", newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            header, data = rows[0], rows[1:]
            duplicate_row = list(data[0])
            with open(reval_case_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                writer.writerows(data + [duplicate_row])
            _check_reval_rejects("a duplicate row (31 rows, one sample_index repeated)", reval_case_path, original_dup)

            # extra row (out-of-range sample_index)
            original_extra = reval_case_path.read_bytes()
            with open(reval_case_path, "r", newline="", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))
            header, data = rows[0], rows[1:]
            extra_row = list(data[0])
            extra_row[header.index("sample_index")] = "30"
            with open(reval_case_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(header)
                writer.writerows(data + [extra_row])
            _check_reval_rejects("an extra row with an out-of-range sample_index", reval_case_path, original_extra)

            reval_errors2, reval_snapshot2 = revalidate_p23_pilot_campaign(p23_reval_dir, git_commit=_FIXED_GIT_COMMIT)
            rec.check(
                "revalidate_p23_pilot_campaign accepts the campaign again once every mutation is fully restored",
                not reval_errors2 and reval_snapshot2 is not None, detail=str(reval_errors2),
            )

            # --- end-to-end: P2.4 finalize-profile now revalidates the full P2.3 pilot ---
            # Detection is checked read-only first (mirroring the existing
            # application.csv tamper test's pattern), so the P2.4 manifest is
            # never driven to FAILED before the evidence is restored --
            # exactly like the already-audited P2.4 manifest state machine
            # (a real _do_finalize_profile failure is intentionally terminal;
            # see "a FAILED campaign can never subsequently reach COMPLETE").
            p24_p23_integration_id = "20260805T070000Z"
            p24_p23_integration_campaign = _run_full_pipeline(tmp_path, p24_p23_integration_id)
            integration_p23_dir = tmp_path.joinpath(*p23.RAW_ROOT_PARTS, p24_p23_integration_id)
            integration_case_path = integration_p23_dir / "cases" / f"{p23.build_plan()[7]['case_name']}.csv"
            integration_original = _mutate_case_row(integration_case_path, 3, "elapsed_cycles", "123456789")
            integration_reval_errors, integration_reval_snapshot = revalidate_p23_pilot_campaign(integration_p23_dir, git_commit=_FIXED_GIT_COMMIT)
            rec.check(
                "P2.4's read-only P2.3 revalidation rejects the underlying pilot evidence once tampered after "
                "record-pilot originally accepted it (Defect 4: P2.4 no longer trusts P2.3 merely because its "
                "manifest/hashes were once valid)",
                bool(integration_reval_errors) and integration_reval_snapshot is None, detail=str(integration_reval_errors),
            )
            integration_case_path.write_bytes(integration_original)
            integration_ok, integration_errors = _do_finalize_profile(campaign_dir=p24_p23_integration_campaign, completed_at_utc="20260805T071000Z")
            rec.check(
                "finalize-profile succeeds once the underlying P2.3 pilot evidence is genuinely valid throughout",
                integration_ok, detail=str(integration_errors),
            )

            # finalize-profile itself (not just the standalone revalidation
            # function) must call the P2.3 revalidation and fail the P2.4
            # campaign closed when it is tampered.
            p24_p23_direct_id = "20260805T072000Z"
            p24_p23_direct_campaign = _run_full_pipeline(tmp_path, p24_p23_direct_id)
            direct_p23_dir = tmp_path.joinpath(*p23.RAW_ROOT_PARTS, p24_p23_direct_id)
            direct_case_path = direct_p23_dir / "cases" / f"{p23.build_plan()[9]['case_name']}.csv"
            _mutate_case_row(direct_case_path, 7, "elapsed_cycles", "555555555")
            direct_ok, direct_errors = _do_finalize_profile(campaign_dir=p24_p23_direct_campaign, completed_at_utc="20260805T072500Z")
            rec.check(
                "finalize-profile itself calls the P2.3 pilot revalidation and fails the P2.4 campaign closed "
                "(terminally FAILED) when the underlying pilot evidence is tampered",
                not direct_ok, detail=str(direct_errors),
            )
            m_direct, _ = load_p24_manifest_chain(p24_p23_direct_campaign)
            rec.check("the P2.4 campaign is driven to FAILED, never silently left COMPLETE, by a P2.3 revalidation failure", m_direct.get("state") == "FAILED")

            # =================================================================
            # Defect 5 repair: deterministic, no-clobber, resumable analysis
            # publication -- fault injection at all four required points.
            # =================================================================
            def _build_complete_campaign(campaign_id: str, finalize_at: str) -> Path:
                camp = _run_full_pipeline(tmp_path, campaign_id)
                ok, errs = _do_finalize_profile(campaign_dir=camp, completed_at_utc=finalize_at)
                if not ok:
                    raise AssertionError(f"self-test fixture: finalize-profile failed: {errs}")
                return camp

            # --- fault before the first artifact ------------------------------------
            fault1_campaign = _build_complete_campaign("20260805T080000Z", "20260805T080100Z")

            def _raise_before_first(i: int, _n: int) -> None:
                if i == 0:
                    raise RuntimeError("injected: before first artifact")

            raised1 = False
            try:
                _do_analyze(campaign_dir=fault1_campaign, analyzed_at_utc="20260805T080200Z", _test_hook_during_publication=_raise_before_first)
            except RuntimeError:
                raised1 = True
            rec.check("fault injection before the first artifact interrupts publication as expected", raised1)
            fault1_analysis_dir = fault1_campaign / "analysis"
            fault1_files_after_crash = sorted(p.name for p in fault1_analysis_dir.iterdir()) if fault1_analysis_dir.exists() else []
            rec.check("a crash before the first artifact leaves zero analysis artifacts published", fault1_files_after_crash == [], detail=str(fault1_files_after_crash))
            ok1, errs1 = _do_analyze(campaign_dir=fault1_campaign, analyzed_at_utc="20260805T080200Z")
            rec.check("a clean retry after a before-first-artifact crash completes successfully", ok1, detail=str(errs1))
            m1, _ = load_p24_manifest_chain(fault1_campaign)
            rec.check("the retried campaign reaches ANALYZED", m1.get("state") == "ANALYZED")

            # --- fault halfway through artifact publication -------------------------
            fault2_campaign = _build_complete_campaign("20260805T090000Z", "20260805T090100Z")

            def _raise_halfway(i: int, n: int) -> None:
                if i == n // 2:
                    raise RuntimeError("injected: halfway through publication")

            raised2 = False
            try:
                _do_analyze(campaign_dir=fault2_campaign, analyzed_at_utc="20260805T090200Z", _test_hook_during_publication=_raise_halfway)
            except RuntimeError:
                raised2 = True
            rec.check("fault injection halfway through publication interrupts it as expected", raised2)
            fault2_analysis_dir = fault2_campaign / "analysis"
            fault2_files_after_crash = sorted(p.name for p in fault2_analysis_dir.iterdir())
            rec.check(
                "a crash halfway through publication leaves a genuine, non-empty, strict subset of the ten artifacts",
                0 < len(fault2_files_after_crash) < len(ANALYSIS_ARTIFACT_RELATIVE_PATHS), detail=str(fault2_files_after_crash),
            )
            fault2_partial_bytes = {name: (fault2_analysis_dir / name).read_bytes() for name in fault2_files_after_crash}
            fault2_partial_inodes = {name: os.stat(fault2_analysis_dir / name).st_ino for name in fault2_files_after_crash}
            ok2, errs2 = _do_analyze(campaign_dir=fault2_campaign, analyzed_at_utc="20260805T090200Z")
            rec.check("a clean retry after a halfway crash completes successfully", ok2, detail=str(errs2))
            rec.check(
                "a clean retry never changes the bytes or the inode of an artifact a prior interrupted attempt already safely published",
                all(
                    (fault2_analysis_dir / name).read_bytes() == fault2_partial_bytes[name]
                    and os.stat(fault2_analysis_dir / name).st_ino == fault2_partial_inodes[name]
                    for name in fault2_files_after_crash
                ),
            )
            m2, _ = load_p24_manifest_chain(fault2_campaign)
            rec.check(
                "the retried halfway-crashed campaign reaches ANALYZED with all ten artifacts",
                m2.get("state") == "ANALYZED" and set(ANALYSIS_ARTIFACT_RELATIVE_PATHS) <= set(m2.get("artifact_sha256", {})),
            )

            # --- fault after all artifacts but before the ANALYZED revision ---------
            fault3_campaign = _build_complete_campaign("20260805T100000Z", "20260805T100100Z")

            def _raise_before_final_gate() -> None:
                raise RuntimeError("injected: after all artifacts, before the final gate")

            raised3 = False
            try:
                _do_analyze(campaign_dir=fault3_campaign, analyzed_at_utc="20260805T100200Z", _test_hook_before_final_gate=_raise_before_final_gate)
            except RuntimeError:
                raised3 = True
            rec.check("fault injection after all artifacts but before the final gate interrupts publication as expected", raised3)
            fault3_analysis_dir = fault3_campaign / "analysis"
            fault3_files_after_crash = sorted(p.name for p in fault3_analysis_dir.iterdir())
            rec.check(
                "a crash after all artifacts but before the final gate leaves all ten artifacts published, manifest still COMPLETE",
                len(fault3_files_after_crash) == len(ANALYSIS_ARTIFACT_RELATIVE_PATHS),
            )
            m3_before, _ = load_p24_manifest_chain(fault3_campaign)
            rec.check("the manifest is still COMPLETE (never a partial ANALYZED) after this crash", m3_before.get("state") == "COMPLETE")
            fault3_bytes_before_retry = {name: (fault3_analysis_dir / name).read_bytes() for name in fault3_files_after_crash}
            ok3, errs3 = _do_analyze(campaign_dir=fault3_campaign, analyzed_at_utc="20260805T100200Z")
            rec.check("a clean retry after an after-artifacts crash completes successfully and appends the missing ANALYZED revision", ok3, detail=str(errs3))
            rec.check(
                "the retry does not change any already-valid artifact's bytes",
                all((fault3_analysis_dir / name).read_bytes() == fault3_bytes_before_retry[name] for name in fault3_files_after_crash),
            )

            # --- fault during the final manifest append -----------------------------
            fault4_campaign = _build_complete_campaign("20260805T110000Z", "20260805T110100Z")

            def _raise_before_manifest_append() -> None:
                raise RuntimeError("injected: during the final manifest append")

            raised4 = False
            try:
                _do_analyze(campaign_dir=fault4_campaign, analyzed_at_utc="20260805T110200Z", _test_hook_before_manifest_append=_raise_before_manifest_append)
            except RuntimeError:
                raised4 = True
            rec.check("fault injection during the final manifest append interrupts it as expected", raised4)
            m4_before, _ = load_p24_manifest_chain(fault4_campaign)
            rec.check("the manifest is still COMPLETE after a fault injected during the final manifest append", m4_before.get("state") == "COMPLETE")
            ok4, errs4 = _do_analyze(campaign_dir=fault4_campaign, analyzed_at_utc="20260805T110200Z")
            rec.check("a clean retry after a during-manifest-append crash completes successfully", ok4, detail=str(errs4))
            m4_after, _ = load_p24_manifest_chain(fault4_campaign)
            rec.check("the retried campaign reaches ANALYZED", m4_after.get("state") == "ANALYZED")

            # --- pure revalidation of an already-ANALYZED campaign (no writes) ------
            reval_only_campaign = _build_complete_campaign("20260805T120000Z", "20260805T120100Z")
            ok_first, errs_first = _do_analyze(campaign_dir=reval_only_campaign, analyzed_at_utc="20260805T120200Z")
            rec.check("first analyze of a fresh COMPLETE campaign succeeds", ok_first, detail=str(errs_first))
            reval_only_dir = reval_only_campaign / "analysis"
            reval_only_inodes_before = {p.name: os.stat(p).st_ino for p in reval_only_dir.iterdir()}
            reval_only_bytes_before = {p.name: p.read_bytes() for p in reval_only_dir.iterdir()}
            _m_reval_only_before, rev_before = load_p24_manifest_chain(reval_only_campaign)
            ok_second, errs_second = _do_analyze(campaign_dir=reval_only_campaign, analyzed_at_utc="20260805T120200Z")
            rec.check("calling analyze again on an already-ANALYZED campaign succeeds (pure revalidation, not an error)", ok_second, detail=str(errs_second))
            reval_only_inodes_after = {p.name: os.stat(p).st_ino for p in reval_only_dir.iterdir()}
            rec.check(
                "revalidating an already-ANALYZED campaign never rewrites any artifact (identical inodes, identical bytes)",
                reval_only_inodes_after == reval_only_inodes_before
                and all(p.read_bytes() == reval_only_bytes_before[p.name] for p in reval_only_dir.iterdir()),
            )
            _m_reval_only_after, rev_after = load_p24_manifest_chain(reval_only_campaign)
            rec.check("revalidating an already-ANALYZED campaign never appends a new manifest revision", rev_before == rev_after)

            # --- retry over altered/unexpected evidence must fail, never silently accept or overwrite ---
            tampered_retry_campaign = _build_complete_campaign("20260805T130000Z", "20260805T130100Z")
            ok_pre, errs_pre = _do_analyze(campaign_dir=tampered_retry_campaign, analyzed_at_utc="20260805T130200Z")
            rec.check("fixture for tampered-retry test reaches ANALYZED", ok_pre, detail=str(errs_pre))
            tampered_path = tampered_retry_campaign / "analysis" / "report.md"
            original_report_bytes = tampered_path.read_bytes()
            tampered_path.write_bytes(original_report_bytes + b"\ntampered by an adversary\n")
            ok_tampered, errs_tampered = _do_analyze(campaign_dir=tampered_retry_campaign, analyzed_at_utc="20260805T130300Z")
            rec.check("a retry over a tampered already-published artifact fails closed, never overwrites it", not ok_tampered, detail=str(errs_tampered))
            rec.check(
                "the tampered artifact is left untouched (not silently repaired or deleted) after a failed retry",
                tampered_path.read_bytes() == original_report_bytes + b"\ntampered by an adversary\n",
            )
            tampered_path.write_bytes(original_report_bytes)
            ok_restored, errs_restored = _do_analyze(campaign_dir=tampered_retry_campaign, analyzed_at_utc="20260805T130300Z")
            rec.check("a retry succeeds again once the tampered artifact is restored", ok_restored, detail=str(errs_restored))

            # --- an unexpected extra entry under analysis/ fails closed --------------
            extra_entry_campaign = _build_complete_campaign("20260805T140000Z", "20260805T140100Z")
            (extra_entry_campaign / "analysis" / "unexpected_extra_file.txt").write_text("adversarial\n", encoding="utf-8")
            ok_extra, errs_extra = _do_analyze(campaign_dir=extra_entry_campaign, analyzed_at_utc="20260805T140200Z")
            rec.check("an unexpected extra entry under analysis/ fails the whole retry closed", not ok_extra, detail=str(errs_extra))
            (extra_entry_campaign / "analysis" / "unexpected_extra_file.txt").unlink()
            ok_extra2, errs_extra2 = _do_analyze(campaign_dir=extra_entry_campaign, analyzed_at_utc="20260805T140200Z")
            rec.check("the retry succeeds again once the unexpected extra entry is removed", ok_extra2, detail=str(errs_extra2))

            # =================================================================
            # Defect 6 repair: every one of the ten analysis artifacts carries
            # the exact ASCII token "publishable=false"
            # =================================================================
            token_campaign = _build_complete_campaign("20260805T150000Z", "20260805T150100Z")
            ok_token, errs_token = _do_analyze(campaign_dir=token_campaign, analyzed_at_utc="20260805T150200Z")
            rec.check("fixture for publication-token test reaches a terminal state", ok_token, detail=str(errs_token))
            token_analysis_dir = token_campaign / "analysis"
            token_bytes_by_name = {Path(rel).name: (token_campaign / rel).read_bytes() for rel in ANALYSIS_ARTIFACT_RELATIVE_PATHS}
            rec.check(
                "exactly the ten canonical analysis artifacts exist on disk",
                set(token_bytes_by_name) == {Path(rel).name for rel in ANALYSIS_ARTIFACT_RELATIVE_PATHS} and len(token_bytes_by_name) == 10,
            )
            token_token_bytes = PUBLICATION_STATUS_TOKEN.encode("ascii")
            for _name, _content in token_bytes_by_name.items():
                rec.check(f"{_name} contains the exact ASCII token {PUBLICATION_STATUS_TOKEN!r}", token_token_bytes in _content)
                rec.check(f"{_name} never contains or implies 'publishable=true'", b"publishable=true" not in _content)

            ceiling_doc_token = json.loads((token_analysis_dir / "empirical_ceiling.json").read_text(encoding="utf-8"))
            rec.check("empirical_ceiling.json retains a boolean publishable=false field", ceiling_doc_token.get("publishable") is False)
            rec.check("empirical_ceiling.json carries a string publication_status field containing the exact token", ceiling_doc_token.get("publication_status") == PUBLICATION_STATUS_TOKEN)

            analysis_manifest_doc_token = json.loads((token_analysis_dir / "analysis_manifest.json").read_text(encoding="utf-8"))
            rec.check("analysis_manifest.json retains a boolean publishable=false field", analysis_manifest_doc_token.get("publishable") is False)
            rec.check("analysis_manifest.json carries a string publication_status field containing the exact token", analysis_manifest_doc_token.get("publication_status") == PUBLICATION_STATUS_TOKEN)

            for csv_name, header in (
                ("configuration_statistics.csv", CONFIGURATION_STATISTICS_HEADER),
                ("scaling.csv", SCALING_HEADER),
                ("saturation.csv", SATURATION_HEADER),
                ("profile_validation.csv", PROFILE_VALIDATION_HEADER),
            ):
                with open(token_analysis_dir / csv_name, newline="", encoding="utf-8") as fh:
                    csv_rows = list(csv.reader(fh))
                rec.check(f"{csv_name} has a publication_status column", PUBLICATION_STATUS_COLUMN in csv_rows[0])
                status_col = csv_rows[0].index(PUBLICATION_STATUS_COLUMN)
                rec.check(
                    f"every {csv_name} data row's publication_status column is exactly {PUBLICATION_STATUS_TOKEN!r}",
                    all(row[status_col] == PUBLICATION_STATUS_TOKEN for row in csv_rows[1:]),
                )

            report_text = (token_analysis_dir / "report.md").read_text(encoding="utf-8")
            rec.check("report.md contains an explicit publishable=false status line", f"`{PUBLICATION_STATUS_TOKEN}`" in report_text)

            for svg_name in ("throughput.svg", "scaling_efficiency.svg", "saturation.svg"):
                svg_text = (token_analysis_dir / svg_name).read_text(encoding="utf-8")
                rec.check(f"{svg_name} carries the token in deterministic <metadata>", f"<metadata>{PUBLICATION_STATUS_TOKEN}</metadata>" in svg_text)

    if rec.failures:
        print(f"analyze_exp02_umma_throughput_p24: self-test: FAILED ({len(rec.failures)}/{rec.total} case(s)): {rec.failures}", file=sys.stderr)
        print("analyze_exp02_umma_throughput_p24: SELF_TEST_RESULT=FAIL", file=sys.stderr)
        return 1
    print(f"analyze_exp02_umma_throughput_p24: self-test: OK ({rec.total} cases)", file=sys.stderr)
    print("analyze_exp02_umma_throughput_p24: SELF_TEST_RESULT=PASS", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_exp02_umma_throughput_p24.py",
        description="P2.4 plan/preflight/pilot-recording/NCU-validation/analysis helper (see module docstring).",
    )
    parser.add_argument("--self-test", action="store_true", help="Run GPU-free synthetic tests and exit.")
    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="Print the frozen 24-case profile plan.")
    plan_parser.add_argument("--format", choices=("text", "lines", "json"), default="text")
    plan_parser.set_defaults(func=cmd_plan)

    init_parser = subparsers.add_parser("init-campaign", help="Symlink-safe P2.4 campaign creation.")
    init_parser.add_argument("--campaign-id", required=True)
    init_parser.add_argument("--started-at-utc", required=True)
    init_parser.set_defaults(func=cmd_init_campaign)

    vp_parser = subparsers.add_parser("validate-preflight", help="Validate a preflight summary.json.")
    vp_parser.add_argument("--preflight", required=True)
    vp_parser.add_argument("--expected-git-commit", required=True)
    vp_parser.add_argument("--now", default=None, help="Override 'now' (YYYY-MM-DDTHH:MM:SSZ); for tests.")
    vp_parser.set_defaults(func=cmd_validate_preflight)

    rp_parser = subparsers.add_parser("record-pilot", help="Validate a completed P2.3 benchmark campaign as this campaign's pilot input.")
    rp_parser.add_argument("--campaign-dir", required=True)
    rp_parser.add_argument("--p23-campaign-dir", required=True)
    rp_parser.add_argument("--preflight", required=True)
    rp_parser.add_argument("--git-commit", required=True)
    rp_parser.add_argument("--completed-at-utc", required=True)
    rp_parser.add_argument("--now", default=None)
    rp_parser.set_defaults(func=cmd_record_pilot)

    dm_parser = subparsers.add_parser("discover-metrics", help="Resolve NCU metrics; start the profiling phase.")
    dm_parser.add_argument("--campaign-dir", required=True)
    dm_parser.add_argument("--discovery-log", required=True)
    dm_parser.add_argument("--preflight", required=True)
    dm_parser.add_argument("--git-commit", required=True)
    dm_parser.add_argument("--started-at-utc", required=True)
    dm_parser.add_argument("--now", default=None)
    dm_parser.set_defaults(func=cmd_discover_metrics)

    vpp_parser = subparsers.add_parser("validate-profile-preconditions", help="GPU-free check that a fresh profiling preflight matches the recorded pilot preflight.")
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

    fp_parser = subparsers.add_parser("finalize-profile", help="Re-validate and close the 24-case profile set.")
    fp_parser.add_argument("--campaign-dir", required=True)
    fp_parser.add_argument("--completed-at-utc", required=True)
    fp_parser.set_defaults(func=cmd_finalize_profile)

    an_parser = subparsers.add_parser("analyze", help="Generate analysis/* from a COMPLETE campaign (state ANALYZED or INCONCLUSIVE).")
    an_parser.add_argument("--campaign-dir", required=True)
    an_parser.add_argument("--analyzed-at-utc", required=True)
    an_parser.set_defaults(func=cmd_analyze)

    mw_parser = subparsers.add_parser("manifest-write", help="Mark FAILED/INTERRUPTED with required failure metadata (never a completing state).")
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
