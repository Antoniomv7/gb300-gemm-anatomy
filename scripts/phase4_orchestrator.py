#!/usr/bin/env python3
"""P4.1 private helper for scripts/run_all.sh (the only public Phase 4 entry point).

This module owns deterministic planning, the append-only hash-chained
top-level manifest, descriptor-anchored safe capture, hashing, evidence
validation, and resume logic for one Phase 4 campaign. It composes the
already closed and audited experimental units without reimplementing any of
them: every GPU action is delegated to an existing Make target, and every
per-experiment verdict is delegated to that experiment's own validator.

    Experiment 1 (memory)  make memory-paths-p14-pilot / -profile / -analyze
    Experiment 2 (umma)    make compute-umma-p24-pilot / -profile / -analyze
    Experiment 3 (gemm)    make gemm-comparison-p35-smoke   (one atomic stage)

P4.1 is orchestration infrastructure only. It runs no pilot and no final
campaign of its own, computes no cross-experiment interpretation, creates no
variability/publication threshold, and marks nothing publishable: every
artifact it writes records `publishable=false` unconditionally. P4.2 will
execute one pilot and three independent final campaigns; P4.3 will perform the
integrated analysis, documentation, and final audit.

See src/phase4/P4_1_PROTOCOL.md for the complete frozen contract.

Trust model (mirrors src/memory/P1_4_PROTOCOL.md section 0): the campaign
filesystem is trusted and single-writer. The descriptor-anchored opens,
no-clobber publication, and re-hashing gates below make single-writer mistakes
and interruptions safe and auditable. They do not claim to withstand a hostile
co-resident process deliberately racing filesystem operations.

Subcommands:

    plan          Print the deterministic stage plan. No filesystem mutation,
                  no Git, no Docker, no GPU, no network.
    run           Execute (or resume) one campaign.
    --self-test   GPU-free synthetic checks.

Exit codes: 0 success (including a pure revalidation of a COMPLETE campaign);
1 execution/validation failure; 2 CLI, repository-state, or safety
precondition failure; 4 a terminal but non-complete INCONCLUSIVE outcome;
130 interrupted (the conventional SIGINT status).
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

SCHEMA_VERSION = "p41.v1"
UNIT = "P4.1"
PUBLISHABLE = False

EXIT_OK = 0
EXIT_RUN_FAILURE = 1
EXIT_PRECONDITION = 2
EXIT_INCONCLUSIVE = 4
# The conventional status for "terminated by SIGINT". It is both this process's
# own exit code on an interruption and the status recorded in the interrupted
# attempt, so an interruption is never confused with an ordinary stage failure.
EXIT_INTERRUPTED = 130

# --------------------------------------------------------------------------
# Frozen identifiers and the deterministic stage plans.
# --------------------------------------------------------------------------

CAMPAIGN_ID_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z$")
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
GPU_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F][0-9a-fA-F-]+$")
GPU_INDEX_RE = re.compile(r"^[0-9]+$")
BASENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

CAMPAIGN_KINDS = ("pilot", "final")
SCOPES = ("full", "memory", "umma", "gemm")
SELECTORS = ("memory", "umma", "gemm")

STAGE_PREFLIGHT = "preflight"
STAGE_MEMORY_PILOT = "memory.pilot"
STAGE_MEMORY_PROFILE = "memory.profile"
STAGE_MEMORY_ANALYZE = "memory.analyze"
STAGE_UMMA_PILOT = "umma.pilot"
STAGE_UMMA_PROFILE = "umma.profile"
STAGE_UMMA_ANALYZE = "umma.analyze"
STAGE_GEMM_CAPTURE = "gemm.capture"
STAGE_CAMPAIGN_VALIDATE = "campaign.validate"

# The exact orchestration order, frozen by src/phase4/P4_1_PROTOCOL.md
# section 4. Both experiments collect all of their GPU evidence before either
# GPU-free analyzer runs, so a campaign holds the selected GPU for the
# shortest contiguous span and no analysis interleaves with GPU work.
STAGE_PLANS: dict[str, tuple[str, ...]] = {
    "full": (
        STAGE_PREFLIGHT,
        STAGE_MEMORY_PILOT,
        STAGE_MEMORY_PROFILE,
        STAGE_UMMA_PILOT,
        STAGE_UMMA_PROFILE,
        STAGE_GEMM_CAPTURE,
        STAGE_MEMORY_ANALYZE,
        STAGE_UMMA_ANALYZE,
        STAGE_CAMPAIGN_VALIDATE,
    ),
    "memory": (
        STAGE_PREFLIGHT,
        STAGE_MEMORY_PILOT,
        STAGE_MEMORY_PROFILE,
        STAGE_MEMORY_ANALYZE,
        STAGE_CAMPAIGN_VALIDATE,
    ),
    "umma": (
        STAGE_PREFLIGHT,
        STAGE_UMMA_PILOT,
        STAGE_UMMA_PROFILE,
        STAGE_UMMA_ANALYZE,
        STAGE_CAMPAIGN_VALIDATE,
    ),
    "gemm": (
        STAGE_PREFLIGHT,
        STAGE_GEMM_CAPTURE,
        STAGE_CAMPAIGN_VALIDATE,
    ),
}

ALL_STAGES: tuple[str, ...] = (
    STAGE_PREFLIGHT,
    STAGE_MEMORY_PILOT,
    STAGE_MEMORY_PROFILE,
    STAGE_UMMA_PILOT,
    STAGE_UMMA_PROFILE,
    STAGE_GEMM_CAPTURE,
    STAGE_MEMORY_ANALYZE,
    STAGE_UMMA_ANALYZE,
    STAGE_CAMPAIGN_VALIDATE,
)

# Which existing Make target each stage delegates to. campaign.validate runs
# no child command at all: it is this module's own final integrity gate.
STAGE_MAKE_TARGET: dict[str, str] = {
    STAGE_PREFLIGHT: "preflight",
    STAGE_MEMORY_PILOT: "memory-paths-p14-pilot",
    STAGE_MEMORY_PROFILE: "memory-paths-p14-profile",
    STAGE_MEMORY_ANALYZE: "memory-paths-p14-analyze",
    STAGE_UMMA_PILOT: "compute-umma-p24-pilot",
    STAGE_UMMA_PROFILE: "compute-umma-p24-profile",
    STAGE_UMMA_ANALYZE: "compute-umma-p24-analyze",
    STAGE_GEMM_CAPTURE: "gemm-comparison-p35-smoke",
    STAGE_CAMPAIGN_VALIDATE: "",
}

# Stages that execute on the selected GPU. Their presence in the pending set
# is what makes a fresh preflight mandatory for this invocation.
GPU_STAGES = frozenset({
    STAGE_MEMORY_PILOT,
    STAGE_MEMORY_PROFILE,
    STAGE_UMMA_PILOT,
    STAGE_UMMA_PROFILE,
    STAGE_GEMM_CAPTURE,
})

# Stages that own no persistent per-experiment campaign state and may
# therefore be retried inside the same top-level campaign after a failure.
# Every other failure is terminal for the campaign, because the underlying
# P1.4/P2.4 campaign is itself terminally FAILED and its own frozen protocol
# never reopens a terminal campaign.
RESUMABLE_FAILURE_STAGES = frozenset({STAGE_PREFLIGHT, STAGE_GEMM_CAPTURE})

STAGE_KIND: dict[str, str] = {
    STAGE_PREFLIGHT: "gpu",
    STAGE_MEMORY_PILOT: "gpu",
    STAGE_MEMORY_PROFILE: "gpu",
    STAGE_MEMORY_ANALYZE: "gpu-free",
    STAGE_UMMA_PILOT: "gpu",
    STAGE_UMMA_PROFILE: "gpu",
    STAGE_UMMA_ANALYZE: "gpu-free",
    STAGE_GEMM_CAPTURE: "gpu",
    STAGE_CAMPAIGN_VALIDATE: "gpu-free",
}

PHASE4_RAW_PARTS = ("results", "raw", "phase4")
P14_RAW_REL = "results/raw/exp01_memory_paths_p14"
P24_RAW_REL = "results/raw/exp02_umma_throughput_p24"
PREFLIGHT_RAW_REL = "results/preflight"

GEMM_ARTIFACT_NAME = "gemm_comparison.csv"
PLAN_FILE_NAME = "plan.json"
MANIFEST_DIR_NAME = "manifest"
LOGS_DIR_NAME = "logs"
EXP03_DIR_NAME = "exp03"
CAMPAIGN_SUBDIRS = (MANIFEST_DIR_NAME, LOGS_DIR_NAME, EXP03_DIR_NAME)
GEMM_ARTIFACT_REL = f"{EXP03_DIR_NAME}/{GEMM_ARTIFACT_NAME}"

MANIFEST_REVISION_RE = re.compile(r"^(\d{6})\.json$")
MANIFEST_REVISION_KEYS = ("manifest_revision", "previous_manifest_sha256")

# The states of the two experiment campaigns P4.1 composes, taken verbatim
# from their own frozen protocols. P4.1 never redefines them.
P14_STATE_ORDER = ("PILOT_IN_PROGRESS", "PILOT_COMPLETE", "PROFILE_IN_PROGRESS",
                   "COMPLETE", "ANALYZED")
P24_STATE_ORDER = ("PILOT_IN_PROGRESS", "PILOT_COMPLETE", "PROFILE_IN_PROGRESS",
                   "COMPLETE", "ANALYZED")
P24_TERMINAL_INCONCLUSIVE = "INCONCLUSIVE"
UNIT_TERMINAL_STATES = frozenset({"ANALYZED", P24_TERMINAL_INCONCLUSIVE})

# The unit states at which that unit's own verify_campaign_evidence_integrity()
# applies: exactly the states its own frozen protocol re-verifies before
# publishing (before COMPLETE, and before ANALYZED/INCONCLUSIVE). P4.1 calls
# that function; it never reimplements any of the semantics behind it.
UNIT_INTEGRITY_STATES = frozenset({"COMPLETE", "ANALYZED", P24_TERMINAL_INCONCLUSIVE})

# The campaign-provenance fields both closed unit schemas expose (P1.4 records
# them in `provenance` when it records its pilot; P2.4 additionally requires
# them by name in MANDATORY_PROVENANCE_FIELDS). They are the only unit
# provenance P4.1 compares, and a missing or malformed one fails closed.
#
# `cuda_driver_version` and `cuda_runtime_version` are the integer-encoded CUDA
# API versions the profiled binary itself reports. The preflight's
# `gpu.driver_version` is the NVIDIA display-driver package string reported by
# the launcher: a different measurement of a different thing. P1.4 documents
# that boundary explicitly, and P4.1 never crosses it either -- the CUDA
# versions are compared between components, never against the driver string.
UNIT_PROVENANCE_FIELDS = ("git_commit", "gpu_uuid", "gpu_name", "compute_capability",
                          "cuda_driver_version", "cuda_runtime_version")
# unit provenance field -> the field of the campaign's validated preflight GPU
# identity that reports the same measurement.
UNIT_TO_PREFLIGHT_GPU_FIELDS = (("gpu_uuid", "uuid"), ("gpu_name", "name"),
                                ("compute_capability", "compute_capability"))
# The comparable fields every component of one campaign must agree on. They are
# collected from each stage's recorded evidence and compared as a set in the
# final integrity gate.
COMPARABLE_EVIDENCE_FIELDS = ("git_commit", "gpu_uuid", "gpu_name", "compute_capability",
                              "cuda_driver_version", "cuda_runtime_version")

STAGE_REQUIRED_UNIT_STATE = {
    STAGE_MEMORY_PILOT: "PILOT_COMPLETE",
    STAGE_MEMORY_PROFILE: "COMPLETE",
    STAGE_MEMORY_ANALYZE: "ANALYZED",
    STAGE_UMMA_PILOT: "PILOT_COMPLETE",
    STAGE_UMMA_PROFILE: "COMPLETE",
    STAGE_UMMA_ANALYZE: "ANALYZED",
}

# The six required preflight checks, restated here independently of
# scripts/preflight.sh so a silently shortened check list is rejected.
REQUIRED_PREFLIGHT_CHECKS = (
    "gpu_visibility",
    "tool_versions",
    "cuda_smoke_compile",
    "cuda_smoke_run",
    "cutedsl_smoke",
    "ncu_profile",
)
REQUIRED_COMPUTE_CAPABILITY = "10.3"

# scripts/run_container.sh's own allowlisted selection banner, and
# scripts/preflight.sh's own two output-location markers. These are the only
# evidence P4.1 ever uses to learn which preflight directory an invocation
# produced: never `ls -t`, never a "latest" symlink, never glob ordering, and
# never a modification time.
RUN_CONTAINER_SELECTION_RE = re.compile(
    r"^run_container: selected index=(?P<index>[0-9]+) uuid=(?P<uuid>GPU-[0-9a-fA-F-]+) "
    r"name='(?P<name>[^']*)' driver=(?P<driver>\S+)$"
)
PREFLIGHT_WRITING_RE = re.compile(
    r"^preflight: writing to (?P<dir>results/preflight/(?P<ts>[0-9]{8}T[0-9]{6}Z))$"
)
PREFLIGHT_SUMMARY_RE = re.compile(
    r"^preflight: summary written to (?P<path>results/preflight/"
    r"(?P<ts>[0-9]{8}T[0-9]{6}Z)/summary\.json)$"
)

# --------------------------------------------------------------------------
# The manifest schema. Every field is allowlisted by name and type and is
# classified into exactly one mutation category. Nothing outside this table
# may ever appear in a manifest revision.
# --------------------------------------------------------------------------

ALLOWED_MANIFEST_KEYS: dict[str, tuple[type, ...]] = {
    "schema_version": (str,),
    "unit": (str,),
    "campaign_id": (str,),
    "campaign_kind": (str,),
    "scope": (str,),
    "stage_order": (list,),
    "plan_sha256": (str,),
    "git_commit": (str,),
    "git_dirty": (bool,),
    "publishable": (bool,),
    "state": (str,),
    "created_at_utc": (str,),
    "updated_at_utc": (str,),
    "stage_attempts": (list,),
    "stage_results": (dict,),
    "preflight_reference": (dict,),
    "gpu": (dict,),
    "outcome": (str,),
    "failure_stage": (str,),
    "failure_detail": (list,),
}

MANIFEST_IMMUTABLE_FIELDS = frozenset({
    "schema_version", "unit", "campaign_id", "campaign_kind", "scope",
    "stage_order", "plan_sha256", "git_commit", "git_dirty", "publishable",
    "created_at_utc",
})
MANIFEST_STATE_DERIVED_FIELDS = frozenset({"state", "updated_at_utc"})
MANIFEST_APPEND_ONLY_FIELDS = frozenset({"stage_attempts", "stage_results"})
MANIFEST_SET_LATEST_FIELDS = frozenset({"preflight_reference", "gpu"})
MANIFEST_TERMINAL_FIELDS = frozenset({"outcome", "failure_stage", "failure_detail"})

_UNCLASSIFIED = set(ALLOWED_MANIFEST_KEYS) - (
    MANIFEST_IMMUTABLE_FIELDS
    | MANIFEST_STATE_DERIVED_FIELDS
    | MANIFEST_APPEND_ONLY_FIELDS
    | MANIFEST_SET_LATEST_FIELDS
    | MANIFEST_TERMINAL_FIELDS
)
assert not _UNCLASSIFIED, f"unclassified P4.1 manifest fields: {sorted(_UNCLASSIFIED)}"

MANIFEST_REQUIRED_FIELDS = (
    "schema_version", "unit", "campaign_id", "campaign_kind", "scope",
    "stage_order", "plan_sha256", "git_commit", "git_dirty", "publishable",
    "state", "created_at_utc", "updated_at_utc", "stage_attempts",
    "stage_results",
)

STATE_IN_PROGRESS = "IN_PROGRESS"
STATE_COMPLETE = "COMPLETE"
STATE_INCONCLUSIVE = "INCONCLUSIVE"
STATE_FAILED = "FAILED"
STATE_INTERRUPTED = "INTERRUPTED"

ALLOWED_STATES = frozenset({
    STATE_IN_PROGRESS, STATE_COMPLETE, STATE_INCONCLUSIVE,
    STATE_FAILED, STATE_INTERRUPTED,
})
TERMINAL_STATES = frozenset({
    STATE_COMPLETE, STATE_INCONCLUSIVE, STATE_FAILED, STATE_INTERRUPTED,
})

# The only legal state edges. A FAILED or INTERRUPTED campaign may be reopened
# by an explicit --resume (subject to the stage-level rule above); COMPLETE and
# INCONCLUSIVE never are.
ALLOWED_TRANSITIONS: dict[str | None, frozenset[str]] = {
    None: frozenset({STATE_IN_PROGRESS}),
    STATE_IN_PROGRESS: frozenset({
        STATE_IN_PROGRESS, STATE_COMPLETE, STATE_INCONCLUSIVE,
        STATE_FAILED, STATE_INTERRUPTED,
    }),
    STATE_FAILED: frozenset({STATE_IN_PROGRESS}),
    STATE_INTERRUPTED: frozenset({STATE_IN_PROGRESS}),
    STATE_COMPLETE: frozenset(),
    STATE_INCONCLUSIVE: frozenset(),
}

ATTEMPT_OUTCOMES = ("COMPLETE", "FAILED", "INTERRUPTED")
ATTEMPT_KEYS = ("stage", "attempt", "started_at_utc", "finished_at_utc",
                "outcome", "exit_status", "stdout_log", "stderr_log")
INTERRUPTED_EXIT_STATUS = EXIT_INTERRUPTED

# One attempt's two no-clobber log names, which are also how the next free
# attempt number is derived from disk (see Campaign._attempt_number).
LOG_NAME_RE = re.compile(r"^(?P<stage>[A-Za-z0-9_]+)\.(?P<attempt>\d{3})\.(stdout|stderr)\.log$")

# Every experimental Make target is invoked silently and without directory
# diagnostics: an echoed recipe line can carry the absolute bind-mount source of
# this checkout, which must never reach a durable Phase 4 log. The invoked
# scripts' and experiments' own output is passed through untouched.
MAKE_QUIET_FLAGS = ("--silent", "--no-print-directory")

# The stable replacement for this checkout's own absolute root in the durable
# text P4.1 writes itself (its campaign.validate logs and the manifest's
# failure detail). Nothing else about any message is rewritten.
REPO_ROOT_TOKEN = "<repo-root>"

# Allowlisted GPU identity. Nothing else about the device, the host, the
# operator, the environment, or dynamic telemetry is ever recorded.
GPU_IDENTITY_FIELDS = ("uuid", "name", "compute_capability", "driver_version")

PREFLIGHT_REFERENCE_KEYS = ("summary_path", "summary_sha256",
                            "preflight_timestamp_utc", "selected_gpu_index")

# Field-name tokens that must never appear anywhere in a manifest revision, at
# any nesting depth. Enforced structurally by validate_manifest_privacy().
FORBIDDEN_MANIFEST_KEY_TOKENS = (
    "user", "username", "home", "hostname", "env", "environ", "environment",
    "password", "passwd", "secret", "token", "credential", "ssh", "argv",
    "cmdline", "commandline", "pid", "power", "clock", "temperature",
    "utilization",
)


class OrchestratorError(Exception):
    """A fail-closed precondition, safety, or CLI error (exit 2)."""


class RunFailure(Exception):
    """A stage or validation failure of an otherwise well-formed run (exit 1)."""


class ManifestError(Exception):
    """The campaign manifest chain is missing, malformed, or not a legal
    continuation of itself."""


# --------------------------------------------------------------------------
# Deterministic plan construction and rendering.
# --------------------------------------------------------------------------


def validate_campaign_id(campaign_id: object) -> None:
    """An explicit canonical UTC timestamp. A campaign ID is never generated
    implicitly by this project, at any level."""
    if not isinstance(campaign_id, str) or not CAMPAIGN_ID_RE.fullmatch(campaign_id):
        raise OrchestratorError(
            f"--campaign-id {campaign_id!r} must match exactly YYYYMMDDTHHMMSSZ")
    try:
        datetime.strptime(campaign_id, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise OrchestratorError(
            f"--campaign-id {campaign_id!r} is not a real calendar UTC instant") from exc


def validate_campaign_kind(kind: object) -> None:
    if kind not in CAMPAIGN_KINDS:
        raise OrchestratorError(
            f"--campaign-kind {kind!r} must be one of {list(CAMPAIGN_KINDS)}")


def scope_for_selector(selector: object) -> str:
    """An invocation without --only uses the complete three-experiment plan."""
    if selector is None:
        return "full"
    if selector not in SELECTORS:
        raise OrchestratorError(
            f"--only {selector!r} must be exactly one of {list(SELECTORS)}")
    return str(selector)


def build_stage_plan(scope: str) -> tuple[str, ...]:
    if scope not in STAGE_PLANS:
        raise OrchestratorError(f"unknown scope {scope!r}")
    return STAGE_PLANS[scope]


def build_plan_document(campaign_id: str, campaign_kind: str, scope: str) -> dict:
    """The immutable, deterministic plan recorded once as plan.json."""
    stages = build_stage_plan(scope)
    return {
        "schema_version": SCHEMA_VERSION,
        "unit": UNIT,
        "campaign_id": campaign_id,
        "campaign_kind": campaign_kind,
        "scope": scope,
        "stage_order": list(stages),
        "stages": [
            {
                "index": index,
                "stage": stage,
                "kind": STAGE_KIND[stage],
                "make_target": STAGE_MAKE_TARGET[stage],
            }
            for index, stage in enumerate(stages, start=1)
        ],
        "publishable": PUBLISHABLE,
    }


def canonical_json_bytes(document: dict) -> bytes:
    """One canonical serialization, so a hash over it is reproducible.

    Key order is the document's own construction order, never sorted: the
    manifest's `stage_results` is an ordered prefix of the stage plan, and
    sorting would silently destroy that ordering on reload.
    """
    return (json.dumps(document, indent=2, sort_keys=False, ensure_ascii=True)
            + "\n").encode("utf-8")


def render_plan(document: dict) -> str:
    """The deterministic --dry-run rendering: stable and line-oriented, with
    no timestamp, absolute path, host name, or other non-reproducible text."""
    lines = [
        "phase4 orchestration plan (P4.1)",
        f"schema_version: {document['schema_version']}",
        f"campaign_id: {document['campaign_id']}",
        f"campaign_kind: {document['campaign_kind']}",
        f"scope: {document['scope']}",
        f"stage_count: {len(document['stages'])}",
    ]
    for entry in document["stages"]:
        target = f"make {entry['make_target']}" if entry["make_target"] else "(internal integrity gate)"
        lines.append(f"  {entry['index']}. {entry['stage']:<18} {entry['kind']:<8} {target}")
    lines.extend([
        "publishable: false",
        "P4.1 is orchestration infrastructure only: this plan was not executed.",
    ])
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Descriptor-anchored, no-clobber filesystem primitives.
#
# Every campaign path component is opened exactly once with
# O_DIRECTORY | O_NOFOLLOW relative to the previously opened descriptor and is
# never re-resolved by pathname afterwards, so a symlink swap of an ancestor
# (or of a final target) cannot redirect a later operation. Output files are
# created with O_CREAT | O_EXCL | O_NOFOLLOW, which is the sole no-clobber
# guarantee; os.replace() is never used anywhere in this module.
# --------------------------------------------------------------------------


def validate_basename(name: object) -> str:
    """A strict single path component. Empty, '.', '..', anything containing
    '/' or a control character, and any absolute path are rejected before the
    name ever reaches os.open(), os.link(), or os.stat()."""
    if not isinstance(name, str) or not name:
        raise OrchestratorError("a campaign file name must be a non-empty string")
    if name in (".", ".."):
        raise OrchestratorError(f"campaign file name {name!r} is not a real component")
    if "/" in name or "\0" in name or os.path.isabs(name):
        raise OrchestratorError(f"campaign file name {name!r} is not a single component")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in name):
        raise OrchestratorError(f"campaign file name {name!r} contains a control character")
    if not BASENAME_RE.fullmatch(name):
        raise OrchestratorError(f"campaign file name {name!r} is outside the allowed alphabet")
    return name


def open_dir_nofollow(name: str, *, dir_fd: int | None = None) -> int:
    """Open one directory component with no-follow semantics."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        if dir_fd is None:
            return os.open(name, flags)
        validate_basename(name)
        return os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise OrchestratorError(
                f"{name!r}: refusing a symlinked campaign path component") from exc
        if exc.errno == errno.ENOTDIR:
            raise OrchestratorError(
                f"{name!r}: campaign path component is not a directory") from exc
        raise OrchestratorError(
            f"{name!r}: cannot open campaign path component: {exc}") from exc


def open_dir_chain(root: Path, *parts: str) -> int:
    """Open root, then walk each component with open_dir_nofollow. The caller
    owns the returned descriptor; each step is anchored on the previous one."""
    current = open_dir_nofollow(str(root))
    try:
        for part in parts:
            nxt = open_dir_nofollow(part, dir_fd=current)
            os.close(current)
            current = nxt
    except BaseException:
        os.close(current)
        raise
    return current


def mkdir_component(path: Path, *, must_not_exist: bool) -> None:
    """Create one directory, rejecting a pre-existing symlink or wrong type.
    mkdir()'s own EEXIST is the only existence test performed."""
    try:
        os.mkdir(path)
        return
    except FileExistsError:
        if must_not_exist:
            raise OrchestratorError(f"{path}: already exists; refusing to reuse it") from None
    except OSError as exc:
        raise OrchestratorError(f"{path}: cannot create directory: {exc}") from exc
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        raise OrchestratorError(f"{path}: is a symlink; refusing")
    if not stat.S_ISDIR(info.st_mode):
        raise OrchestratorError(f"{path}: exists and is not a directory; refusing")


def reject_unsafe_path(path: Path, *, expect_dir: bool) -> None:
    """lstat-based type/symlink check for a path that must already exist. A
    dangling symlink is rejected exactly like a real one."""
    if not os.path.lexists(path):
        raise OrchestratorError(f"{path}: does not exist")
    info = os.lstat(path)
    if stat.S_ISLNK(info.st_mode):
        raise OrchestratorError(f"{path}: is a symlink; refusing (dangling or not)")
    if expect_dir and not stat.S_ISDIR(info.st_mode):
        raise OrchestratorError(f"{path}: is not a directory")
    if not expect_dir and not stat.S_ISREG(info.st_mode):
        raise OrchestratorError(f"{path}: is not a regular file")


def create_file_exclusive(name: str, *, dir_fd: int) -> int:
    """Create one output file, no-clobber, anchored on an open directory."""
    validate_basename(name)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        return os.open(name, flags, 0o644, dir_fd=dir_fd)
    except FileExistsError as exc:
        raise OrchestratorError(
            f"{name!r}: already exists; refusing to overwrite existing evidence") from exc
    except OSError as exc:
        raise OrchestratorError(f"{name!r}: cannot create evidence file: {exc}") from exc


def write_file_exclusive(name: str, payload: bytes, *, dir_fd: int) -> None:
    fd = create_file_exclusive(name, dir_fd=dir_fd)
    try:
        os.write(fd, payload)
        os.fsync(fd)
    finally:
        os.close(fd)


def read_file_nofollow(name: str, *, dir_fd: int) -> bytes:
    validate_basename(name)
    flags = os.O_RDONLY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        fd = os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise OrchestratorError(f"{name!r}: is a symlink; refusing to read") from exc
        raise OrchestratorError(f"{name!r}: cannot read: {exc}") from exc
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OrchestratorError(f"{name!r}: is not a regular file")
        chunks = []
        while True:
            block = os.read(fd, 1 << 20)
            if not block:
                break
            chunks.append(block)
        return b"".join(chunks)
    finally:
        os.close(fd)


def link_no_clobber(src_name: str, dst_name: str, *, src_dir_fd: int, dst_dir_fd: int) -> None:
    """Publish already-validated evidence under its accepted name via an
    in-place hard link. linkat()'s own EEXIST is the sole no-clobber
    guarantee, and the source is never unlinked, so the per-attempt capture an
    earlier attempt produced always survives."""
    validate_basename(src_name)
    validate_basename(dst_name)
    try:
        os.link(src_name, dst_name, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd,
                follow_symlinks=False)
    except FileExistsError as exc:
        raise OrchestratorError(
            f"{dst_name!r}: already exists; refusing to replace accepted evidence") from exc
    except OSError as exc:
        raise OrchestratorError(f"{dst_name!r}: cannot publish evidence: {exc}") from exc


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_of_path(path: Path) -> str:
    """Hash a regular, non-symlink file fresh from disk."""
    reject_unsafe_path(path, expect_dir=False)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_digest(document: object) -> str:
    """A reproducible SHA-256 over an already validated evidence snapshot.

    Keys are sorted here (unlike canonical_json_bytes, which preserves a
    manifest's own construction order) because this digest is only ever a pin
    over another unit's returned snapshot, whose key order P4.1 does not own.
    """
    payload = json.dumps(document, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=True, default=str)
    return sha256_bytes(payload.encode("utf-8"))


def repo_relative(path: Path, repo_root: Path) -> str:
    """Durable metadata always records repository-relative paths, never an
    absolute path that would leak a home directory or an operator name."""
    try:
        return str(Path(path).resolve().relative_to(Path(repo_root).resolve()))
    except ValueError as exc:
        raise OrchestratorError(f"{path}: is outside the repository root {repo_root}") from exc


def redact_repo_root(text: str, repo_root: Path) -> str:
    """Replace this checkout's own absolute root with a stable token.

    Applied only where P4.1 itself writes durable text: its campaign.validate
    logs and the manifest's failure detail. A diagnostic can legitimately quote
    an absolute path (a preflight summary, an evidence file), and that absolute
    prefix is a private host path. Nothing else in the message is rewritten: no
    general sanitization, no scientific content, no child-owned evidence, and
    no GPU identity, failure diagnostic, or reproducibility detail is removed.
    """
    root = str(Path(repo_root).resolve())
    if not root or root == os.sep:
        return text
    return text.replace(root, REPO_ROOT_TOKEN)


def write_all(fd: int, payload: bytes) -> None:
    """Write every byte to an already-open descriptor."""
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:  # pragma: no cover - os.write either progresses or raises
            raise OSError("short write while publishing a stage log")
        view = view[written:]


def copy_log_stream(read_fd: int, write_fd: int, *, repo_root: Path, redact: bool) -> None:
    """Copy one child stream, replacing only the exact checkout-root bytes.

    A small suffix is retained between reads so a path split across pipe chunks
    is still recognized. Streaming keeps partial, already-sanitized diagnostics
    in the no-clobber attempt log if the child is interrupted.
    """
    needle = str(Path(repo_root).resolve()).encode("utf-8")
    replacement = REPO_ROOT_TOKEN.encode("utf-8")
    if not needle or needle == os.sep.encode("utf-8"):
        redact = False
    pending = b""
    try:
        while True:
            chunk = os.read(read_fd, 1 << 16)
            if not chunk:
                break
            if not redact:
                write_all(write_fd, chunk)
                continue
            pending += chunk
            while True:
                index = pending.find(needle)
                if index >= 0:
                    write_all(write_fd, pending[:index])
                    write_all(write_fd, replacement)
                    pending = pending[index + len(needle):]
                    continue
                safe = len(pending) - len(needle) + 1
                if safe > 0:
                    write_all(write_fd, pending[:safe])
                    pending = pending[safe:]
                break
        if pending:
            write_all(write_fd, pending)
    finally:
        os.close(read_fd)


# --------------------------------------------------------------------------
# Manifest schema, privacy, state-shape, and transition validation.
# --------------------------------------------------------------------------


def validate_manifest_privacy(document: object, *, path: str = "manifest") -> list[str]:
    """Reject any key or string value that would carry host, operator,
    environment, credential, process, or dynamic-telemetry information."""
    errors: list[str] = []
    if isinstance(document, dict):
        for key, value in document.items():
            if not isinstance(key, str):
                errors.append(f"{path}: non-string key {key!r}")
                continue
            lowered = key.lower()
            for token in FORBIDDEN_MANIFEST_KEY_TOKENS:
                if (lowered == token or lowered.startswith(token + "_")
                        or lowered.endswith("_" + token)):
                    errors.append(
                        f"{path}.{key}: forbidden manifest field name (token {token!r})")
            errors.extend(validate_manifest_privacy(value, path=f"{path}.{key}"))
    elif isinstance(document, list):
        for index, value in enumerate(document):
            errors.extend(validate_manifest_privacy(value, path=f"{path}[{index}]"))
    elif isinstance(document, str):
        if document.startswith("/"):
            errors.append(f"{path}: absolute path {document!r} is never durable metadata")
    return errors


def validate_manifest_document(document: object, *, expected_campaign_id: str) -> list[str]:
    """Every field allowlisted by name and type; no unknown field; the
    immutable identity fields exactly what this campaign is."""
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["manifest revision is not a JSON object"]

    unknown = sorted(set(document) - set(ALLOWED_MANIFEST_KEYS))
    if unknown:
        errors.append(f"unknown manifest field(s): {unknown}")
    for field in MANIFEST_REQUIRED_FIELDS:
        if field not in document:
            errors.append(f"{field}: required manifest field is absent")
    for field, value in document.items():
        if field not in ALLOWED_MANIFEST_KEYS:
            continue
        allowed = ALLOWED_MANIFEST_KEYS[field]
        if value is None:
            errors.append(f"{field}: is null; no P4.1 manifest field is nullable")
            continue
        if isinstance(value, bool) and bool not in allowed:
            errors.append(f"{field}: a bool is not an allowed type here")
            continue
        if not isinstance(value, allowed):
            errors.append(
                f"{field}: wrong type {type(value).__name__}, expected "
                f"{[t.__name__ for t in allowed]}")
    if errors:
        return errors

    if document["schema_version"] != SCHEMA_VERSION:
        errors.append(f"schema_version={document['schema_version']!r} != {SCHEMA_VERSION!r}")
    if document["unit"] != UNIT:
        errors.append(f"unit={document['unit']!r} != {UNIT!r}")
    if document["campaign_id"] != expected_campaign_id:
        errors.append(
            f"campaign_id={document['campaign_id']!r} != the campaign directory's own name "
            f"{expected_campaign_id!r}")
    if document["campaign_kind"] not in CAMPAIGN_KINDS:
        errors.append(f"campaign_kind={document['campaign_kind']!r} is not a legal kind")
    if document["scope"] not in SCOPES:
        errors.append(f"scope={document['scope']!r} is not a legal scope")
    if document["publishable"] is not False:
        errors.append("publishable must be exactly false in every P4.1 manifest revision")
    if document["git_dirty"] is not False:
        errors.append("git_dirty must be exactly false: a campaign only runs on a clean tree")
    if not GIT_COMMIT_RE.fullmatch(document["git_commit"]):
        errors.append(
            f"git_commit={document['git_commit']!r} is not a 40-character lowercase hex commit")
    if document["state"] not in ALLOWED_STATES:
        errors.append(f"state={document['state']!r} is not a legal P4.1 state")
    if document["scope"] in STAGE_PLANS:
        expected_order = list(STAGE_PLANS[document["scope"]])
        if document["stage_order"] != expected_order:
            errors.append(
                f"stage_order={document['stage_order']!r} != the frozen plan for scope "
                f"{document['scope']!r}: {expected_order!r}")
    for field in ("created_at_utc", "updated_at_utc"):
        if not CAMPAIGN_ID_RE.fullmatch(document[field]):
            errors.append(f"{field}={document[field]!r} is not a canonical UTC timestamp")
    if document["created_at_utc"] > document["updated_at_utc"]:
        errors.append("updated_at_utc precedes created_at_utc")
    if not SHA256_RE.fullmatch(document["plan_sha256"]):
        errors.append("plan_sha256 is not a canonical SHA-256")

    errors.extend(_validate_stage_attempts(document))
    errors.extend(_validate_stage_results(document))
    errors.extend(_validate_terminal_fields(document))
    errors.extend(_validate_preflight_reference(document))
    errors.extend(validate_manifest_privacy(document))
    return errors


def _validate_stage_attempts(document: dict) -> list[str]:
    errors: list[str] = []
    order = document.get("stage_order", [])
    seen: dict[str, int] = {}
    for index, entry in enumerate(document.get("stage_attempts", [])):
        label = f"stage_attempts[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label}: not an object")
            continue
        if sorted(entry) != sorted(ATTEMPT_KEYS):
            errors.append(f"{label}: keys {sorted(entry)} != {sorted(ATTEMPT_KEYS)}")
            continue
        stage = entry["stage"]
        if stage not in order:
            errors.append(f"{label}: stage {stage!r} is not in this campaign's plan")
            continue
        # Strictly increasing, not necessarily contiguous: an attempt number is
        # claimed by creating its two no-clobber logs *before* the child runs,
        # and a hard interruption can leave those logs on disk with no attempt
        # record. The next attempt then skips that number rather than reusing
        # it, so a gap is legal but a repeat or a regression never is.
        attempt = entry["attempt"]
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            errors.append(f"{label}: attempt={attempt!r} is not a positive integer")
            continue
        if attempt <= seen.get(stage, 0):
            errors.append(
                f"{label}: attempt={attempt} does not follow the previous attempt "
                f"{seen[stage]} for stage {stage!r}")
        seen[stage] = attempt
        if entry["outcome"] not in ATTEMPT_OUTCOMES:
            errors.append(f"{label}: outcome={entry['outcome']!r} is not one of {list(ATTEMPT_OUTCOMES)}")
        if not isinstance(entry["exit_status"], int) or isinstance(entry["exit_status"], bool):
            errors.append(f"{label}: exit_status must be an integer")
        elif entry["outcome"] == "COMPLETE" and entry["exit_status"] != 0:
            errors.append(f"{label}: a COMPLETE attempt cannot record a non-zero exit status")
        elif entry["outcome"] in ("FAILED", "INTERRUPTED") and entry["exit_status"] == 0:
            errors.append(f"{label}: a {entry['outcome']} attempt cannot record exit status 0")
        for field in ("started_at_utc", "finished_at_utc"):
            if not isinstance(entry[field], str) or not CAMPAIGN_ID_RE.fullmatch(entry[field]):
                errors.append(f"{label}.{field}: not a canonical UTC timestamp")
        if (isinstance(entry["started_at_utc"], str) and isinstance(entry["finished_at_utc"], str)
                and entry["started_at_utc"] > entry["finished_at_utc"]):
            errors.append(f"{label}: finished_at_utc precedes started_at_utc")
        for field in ("stdout_log", "stderr_log"):
            value = entry[field]
            if not isinstance(value, str) or not value or value.startswith("/"):
                errors.append(f"{label}.{field}: must be a repository-relative path")
    return errors


def _validate_preflight_entry(entry: object, label: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(entry, dict):
        return [f"{label}: not an object"]
    if sorted(entry) != sorted(PREFLIGHT_REFERENCE_KEYS):
        return [f"{label}: keys {sorted(entry)} != {sorted(PREFLIGHT_REFERENCE_KEYS)}"]
    path = entry["summary_path"]
    if not isinstance(path, str) or not path.startswith(PREFLIGHT_RAW_REL + "/"):
        errors.append(f"{label}.summary_path={path!r} is not under {PREFLIGHT_RAW_REL}/")
    if not isinstance(entry["summary_sha256"], str) or not SHA256_RE.fullmatch(entry["summary_sha256"]):
        errors.append(f"{label}.summary_sha256 is not a canonical SHA-256")
    if (not isinstance(entry["preflight_timestamp_utc"], str)
            or not CAMPAIGN_ID_RE.fullmatch(entry["preflight_timestamp_utc"])):
        errors.append(f"{label}.preflight_timestamp_utc is not a canonical UTC timestamp")
    index = entry["selected_gpu_index"]
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        errors.append(f"{label}.selected_gpu_index must be a non-negative integer")
    return errors


def _validate_stage_results(document: dict) -> list[str]:
    errors: list[str] = []
    order = list(document.get("stage_order", []))
    results = document.get("stage_results", {})
    keys = list(results)
    if keys != order[: len(keys)]:
        errors.append(
            f"stage_results keys {keys!r} are not an ordered prefix of the plan {order!r}")
        return errors
    for stage, entry in results.items():
        label = f"stage_results.{stage}"
        if not isinstance(entry, dict):
            errors.append(f"{label}: not an object")
            continue
        expected_keys = {"attempt", "completed_at_utc", "evidence"}
        if set(entry) != expected_keys:
            errors.append(f"{label}: keys {sorted(entry)} != {sorted(expected_keys)}")
            continue
        if (not isinstance(entry["attempt"], int) or isinstance(entry["attempt"], bool)
                or entry["attempt"] < 1):
            errors.append(f"{label}.attempt: must be a positive integer")
        if (not isinstance(entry["completed_at_utc"], str)
                or not CAMPAIGN_ID_RE.fullmatch(entry["completed_at_utc"])):
            errors.append(f"{label}.completed_at_utc: not a canonical UTC timestamp")
        evidence = entry["evidence"]
        if not isinstance(evidence, dict):
            errors.append(f"{label}.evidence: not an object")
            continue
        if stage == STAGE_PREFLIGHT:
            history = evidence.get("preflights")
            if not isinstance(history, list) or not history:
                errors.append(f"{label}.evidence.preflights: must be a non-empty list")
                continue
            if set(evidence) != {"preflights"}:
                errors.append(f"{label}.evidence: keys {sorted(evidence)} != ['preflights']")
                continue
            for position, item in enumerate(history):
                errors.extend(_validate_preflight_entry(
                    item, f"{label}.evidence.preflights[{position}]"))
        elif stage == STAGE_GEMM_CAPTURE:
            for field in ("csv_path", "csv_sha256", "row_count"):
                if field not in evidence:
                    errors.append(f"{label}.evidence.{field}: absent")
            if isinstance(evidence.get("csv_sha256"), str) and not SHA256_RE.fullmatch(
                    evidence["csv_sha256"]):
                errors.append(f"{label}.evidence.csv_sha256 is not a canonical SHA-256")
    return errors


def _validate_terminal_fields(document: dict) -> list[str]:
    errors: list[str] = []
    state = document.get("state")
    has_failure = "failure_stage" in document or "failure_detail" in document
    if state in (STATE_FAILED, STATE_INTERRUPTED):
        if "failure_stage" not in document or "failure_detail" not in document:
            errors.append(f"state={state}: failure_stage and failure_detail are both required")
        else:
            if document["failure_stage"] not in document.get("stage_order", []):
                errors.append("failure_stage is not one of this campaign's own stages")
            if not all(isinstance(item, str) for item in document["failure_detail"]):
                errors.append("failure_detail must be a list of strings")
    elif has_failure:
        errors.append(
            f"state={state}: failure_stage/failure_detail may appear only on FAILED or INTERRUPTED")
    if state in (STATE_COMPLETE, STATE_INCONCLUSIVE):
        if document.get("outcome") != state:
            errors.append(f"state={state} requires outcome={state!r}")
    elif "outcome" in document:
        errors.append(f"state={state}: outcome may appear only on COMPLETE or INCONCLUSIVE")
    return errors


def _validate_preflight_reference(document: dict) -> list[str]:
    errors: list[str] = []
    if "preflight_reference" not in document:
        if "gpu" in document:
            errors.append("gpu identity recorded without a validated preflight reference")
        return errors
    errors.extend(_validate_preflight_entry(document["preflight_reference"], "preflight_reference"))
    if "gpu" not in document:
        errors.append("a validated preflight reference requires the allowlisted gpu identity")
        return errors
    gpu = document["gpu"]
    if sorted(gpu) != sorted(GPU_IDENTITY_FIELDS):
        errors.append(f"gpu keys {sorted(gpu)} != {sorted(GPU_IDENTITY_FIELDS)}")
        return errors
    if not isinstance(gpu["uuid"], str) or not GPU_UUID_RE.fullmatch(gpu["uuid"]):
        errors.append(f"gpu.uuid={gpu['uuid']!r} is not a canonical GPU UUID")
    for field in ("name", "compute_capability", "driver_version"):
        if not isinstance(gpu[field], str) or not gpu[field]:
            errors.append(f"gpu.{field} must be a non-empty string")
    return errors


def validate_manifest_state_shape(document: dict) -> list[str]:
    """A pure function of one revision, taken in isolation: which fields the
    declared state must have, and which it must still lack."""
    errors: list[str] = []
    state = document.get("state")
    results = document.get("stage_results", {})
    order = list(document.get("stage_order", []))

    if state == STATE_IN_PROGRESS:
        if order and list(results) == order:
            errors.append(
                "state=IN_PROGRESS while every planned stage is already recorded complete")
    elif state in (STATE_COMPLETE, STATE_INCONCLUSIVE):
        if list(results) != order:
            errors.append(
                f"state={state} requires every planned stage recorded complete; got "
                f"{list(results)!r}")
        if "preflight_reference" not in document:
            errors.append(f"state={state} requires a validated preflight reference")
    elif state in (STATE_FAILED, STATE_INTERRUPTED):
        if order and list(results) == order:
            errors.append(f"state={state} while every planned stage is recorded complete")
    return errors


def _preflight_history(results: dict, stage: str = STAGE_PREFLIGHT) -> list:
    entry = results.get(stage)
    if not isinstance(entry, dict):
        return []
    evidence = entry.get("evidence")
    if not isinstance(evidence, dict):
        return []
    history = evidence.get("preflights")
    return history if isinstance(history, list) else []


def validate_manifest_revision_transition(previous: dict | None, current: dict) -> list[str]:
    """Is `current` a legitimate continuation of `previous`? The hash chain
    alone only proves that nothing earlier was rewritten; it says nothing
    about whether this revision's content is a legal successor."""
    errors: list[str] = []
    previous_state = None if previous is None else previous.get("state")
    current_state = current.get("state")
    allowed = ALLOWED_TRANSITIONS.get(previous_state, frozenset())
    if current_state not in allowed:
        errors.append(
            f"illegal state transition {previous_state!r} -> {current_state!r} "
            f"(allowed: {sorted(allowed)})")
    if previous is None:
        if current_state != STATE_IN_PROGRESS:
            errors.append("revision 0 must be IN_PROGRESS")
        if current.get("stage_results"):
            errors.append("revision 0 cannot already record a completed stage")
        if current.get("stage_attempts"):
            errors.append("revision 0 cannot already record a stage attempt")
        return errors

    for field in sorted(MANIFEST_IMMUTABLE_FIELDS):
        if previous.get(field) != current.get(field):
            errors.append(
                f"{field}: immutable field changed from {previous.get(field)!r} to "
                f"{current.get(field)!r}")
    if current.get("updated_at_utc", "") < previous.get("updated_at_utc", ""):
        errors.append("updated_at_utc went backwards")

    old_attempts = previous.get("stage_attempts", [])
    new_attempts = current.get("stage_attempts", [])
    if new_attempts[: len(old_attempts)] != old_attempts:
        errors.append("stage_attempts is append-only: an existing entry was edited or removed")
    if len(new_attempts) > len(old_attempts) + 1:
        errors.append("stage_attempts grew by more than one entry in a single revision")

    old_results = previous.get("stage_results", {})
    new_results = current.get("stage_results", {})
    old_keys = list(old_results)
    new_keys = list(new_results)
    if new_keys[: len(old_keys)] != old_keys:
        errors.append("stage_results is append-only and ordered: an earlier stage moved or vanished")
    else:
        for key in old_keys:
            if key == STAGE_PREFLIGHT:
                errors.extend(_validate_preflight_stage_growth(
                    old_results[key], new_results.get(key)))
                continue
            if old_results[key] != new_results.get(key):
                errors.append(f"stage_results.{key}: an already recorded stage result was edited")
    if len(new_keys) > len(old_keys) + 1:
        errors.append("stage_results gained more than one stage in a single revision")

    if "preflight_reference" in previous and "preflight_reference" not in current:
        errors.append("preflight_reference disappeared")
    if "gpu" in previous and "gpu" in current and previous["gpu"] != current["gpu"]:
        errors.append(
            "gpu identity changed inside one campaign; every stage must run on the same device")
    return errors


def _validate_preflight_stage_growth(old_entry: object, new_entry: object) -> list[str]:
    """The one stage result that may legally grow: every --resume with pending
    GPU work creates and validates a fresh preflight, which is appended to this
    stage's own history. Earlier entries are never edited or removed."""
    errors: list[str] = []
    if not isinstance(old_entry, dict) or not isinstance(new_entry, dict):
        return ["stage_results.preflight: malformed stage record"]
    old_history = _preflight_history({STAGE_PREFLIGHT: old_entry})
    new_history = _preflight_history({STAGE_PREFLIGHT: new_entry})
    if new_history[: len(old_history)] != old_history:
        errors.append("stage_results.preflight: an earlier validated preflight was edited")
    if len(new_history) > len(old_history) + 1:
        errors.append("stage_results.preflight: more than one preflight appended in one revision")
    if new_entry.get("attempt", 0) < old_entry.get("attempt", 0):
        errors.append("stage_results.preflight: the attempt counter went backwards")
    if new_entry.get("completed_at_utc", "") < old_entry.get("completed_at_utc", ""):
        errors.append("stage_results.preflight: completed_at_utc went backwards")
    if len(new_history) != len(old_history) and new_entry.get("attempt") == old_entry.get("attempt"):
        errors.append("stage_results.preflight: a new preflight was appended without a new attempt")
    return errors


# --------------------------------------------------------------------------
# The append-only, hash-chained manifest.
# --------------------------------------------------------------------------


def load_manifest_chain(campaign_dir: Path, *, repo_root: Path) -> tuple[dict, int]:
    """Reopen, re-hash, and revalidate every revision from 000000.json
    forward, on every call. Returns (current_content, highest_revision), or
    ({}, -1) when no revision exists. Nothing about an earlier revision is
    ever trusted from memory."""
    expected_campaign_id = campaign_dir.name
    parts = Path(repo_relative(campaign_dir, repo_root)).parts
    manifest_fd = open_dir_chain(repo_root, *parts, MANIFEST_DIR_NAME)
    try:
        names = sorted(os.listdir(manifest_fd))
        revision_names = [n for n in names if MANIFEST_REVISION_RE.fullmatch(n)]
        other = [n for n in names if n not in revision_names]
        if other:
            raise ManifestError(f"unexpected entries in manifest/: {sorted(other)}")
        if not revision_names:
            return {}, -1
        expected_names = [f"{i:06d}.json" for i in range(len(revision_names))]
        if revision_names != expected_names:
            raise ManifestError(
                f"manifest revisions are not exactly contiguous 000000..: {revision_names!r}")
        previous_content: dict | None = None
        previous_hash: str | None = None
        current_content: dict = {}
        for index, name in enumerate(expected_names):
            raw = read_file_nofollow(name, dir_fd=manifest_fd)
            try:
                document = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ManifestError(f"manifest/{name}: invalid JSON: {exc}") from exc
            if not isinstance(document, dict):
                raise ManifestError(f"manifest/{name}: revision root is not a JSON object")
            if document.get("manifest_revision") != index:
                raise ManifestError(
                    f"manifest/{name}: manifest_revision="
                    f"{document.get('manifest_revision')!r} != {index}")
            expected_previous = None if index == 0 else previous_hash
            if document.get("previous_manifest_sha256") != expected_previous:
                raise ManifestError(
                    f"manifest/{name}: previous_manifest_sha256 does not match the freshly "
                    f"recomputed hash of the preceding revision (hash chain broken or tampered)")
            content = {k: v for k, v in document.items() if k not in MANIFEST_REVISION_KEYS}
            schema_errors = validate_manifest_document(
                content, expected_campaign_id=expected_campaign_id)
            if schema_errors:
                raise ManifestError(f"manifest/{name}: schema validation failed: {schema_errors}")
            shape_errors = validate_manifest_state_shape(content)
            if shape_errors:
                raise ManifestError(
                    f"manifest/{name}: state-shape validation failed: {shape_errors}")
            transition_errors = validate_manifest_revision_transition(previous_content, content)
            if transition_errors:
                raise ManifestError(
                    f"manifest/{name}: transition validation failed: {transition_errors}")
            previous_content = content
            current_content = content
            previous_hash = sha256_bytes(raw)
        return current_content, len(expected_names) - 1
    finally:
        os.close(manifest_fd)


def write_next_manifest_revision(campaign_dir: Path, content: dict, *, repo_root: Path) -> int:
    """Append exactly one new immutable revision. The next revision number and
    the previous hash are re-derived by reloading the whole chain immediately
    before writing; publication is exclusive-create, never os.replace()."""
    previous_content, previous_revision = load_manifest_chain(campaign_dir, repo_root=repo_root)
    expected_campaign_id = campaign_dir.name

    schema_errors = validate_manifest_document(content, expected_campaign_id=expected_campaign_id)
    if schema_errors:
        raise ManifestError(f"refusing to write an invalid manifest revision: {schema_errors}")
    shape_errors = validate_manifest_state_shape(content)
    if shape_errors:
        raise ManifestError(f"refusing to write an ill-shaped manifest revision: {shape_errors}")
    transition_errors = validate_manifest_revision_transition(
        previous_content if previous_revision >= 0 else None, content)
    if transition_errors:
        raise ManifestError(f"refusing an illegal manifest transition: {transition_errors}")

    revision = previous_revision + 1
    parts = Path(repo_relative(campaign_dir, repo_root)).parts
    manifest_fd = open_dir_chain(repo_root, *parts, MANIFEST_DIR_NAME)
    try:
        previous_hash = None
        if previous_revision >= 0:
            previous_hash = sha256_bytes(
                read_file_nofollow(f"{previous_revision:06d}.json", dir_fd=manifest_fd))
        document = dict(content)
        document["manifest_revision"] = revision
        document["previous_manifest_sha256"] = previous_hash
        write_file_exclusive(f"{revision:06d}.json", canonical_json_bytes(document),
                             dir_fd=manifest_fd)
    finally:
        os.close(manifest_fd)
    return revision


# --------------------------------------------------------------------------
# Campaign tree creation and resolution.
# --------------------------------------------------------------------------


def campaign_dir_for(campaign_id: str, *, repo_root: Path) -> Path:
    return Path(repo_root).joinpath(*PHASE4_RAW_PARTS, campaign_id)


def create_campaign_tree(campaign_id: str, *, repo_root: Path) -> Path:
    """Create the Phase 4 orchestration tree exactly once, refusing a symlink
    at any level and refusing to reuse an existing campaign directory."""
    validate_campaign_id(campaign_id)
    current = Path(repo_root)
    reject_unsafe_path(current, expect_dir=True)
    for part in PHASE4_RAW_PARTS:
        current = current / part
        mkdir_component(current, must_not_exist=False)
    campaign_dir = current / campaign_id
    mkdir_component(campaign_dir, must_not_exist=True)
    for sub in CAMPAIGN_SUBDIRS:
        mkdir_component(campaign_dir / sub, must_not_exist=False)
    return campaign_dir


def resolve_campaign_tree(campaign_id: str, *, repo_root: Path) -> Path:
    """Resolve an existing campaign directory, rejecting a symlink (dangling
    or not) or an unexpected file type at every level."""
    validate_campaign_id(campaign_id)
    current = Path(repo_root)
    reject_unsafe_path(current, expect_dir=True)
    for part in (*PHASE4_RAW_PARTS, campaign_id):
        current = current / part
        reject_unsafe_path(current, expect_dir=True)
    for sub in CAMPAIGN_SUBDIRS:
        reject_unsafe_path(current / sub, expect_dir=True)
    return current


def campaign_exists(campaign_id: str, *, repo_root: Path) -> bool:
    """Existence is tested with lexists, so a dangling symlink at the campaign
    path still counts as existing (and is rejected on resolution)."""
    return os.path.lexists(campaign_dir_for(campaign_id, repo_root=repo_root))


# --------------------------------------------------------------------------
# Preflight evidence: parsing, validation, provenance.
# --------------------------------------------------------------------------


def parse_preflight_stdout(text: str, *, selected_gpu_index: int) -> dict:
    """Learn which preflight directory this exact invocation produced from the
    invocation's own stdout: never `ls -t`, a 'latest' symlink, glob ordering,
    or a modification time. The launcher's own banner is also what binds the
    requested physical index to the GPU UUID that was actually exposed."""
    selections = []
    writings = []
    summaries = []
    for line in text.splitlines():
        stripped = line.strip()
        match = RUN_CONTAINER_SELECTION_RE.match(stripped)
        if match:
            selections.append(match)
            continue
        match = PREFLIGHT_WRITING_RE.match(stripped)
        if match:
            writings.append(match)
            continue
        match = PREFLIGHT_SUMMARY_RE.match(stripped)
        if match:
            summaries.append(match)
    if len(selections) != 1:
        raise RunFailure(
            f"the preflight stdout carries {len(selections)} launcher selection line(s); exactly "
            f"one is required to bind the requested physical index to a GPU UUID")
    if len(writings) != 1 or len(summaries) != 1:
        raise RunFailure(
            f"the preflight stdout carries {len(writings)} output-directory marker(s) and "
            f"{len(summaries)} summary marker(s); exactly one of each is required")
    selection, writing, summary = selections[0], writings[0], summaries[0]
    if int(selection.group("index")) != selected_gpu_index:
        raise RunFailure(
            f"the launcher selected physical index {selection.group('index')}, but this "
            f"invocation requested {selected_gpu_index}")
    if writing.group("ts") != summary.group("ts"):
        raise RunFailure(
            "the preflight output-directory and summary markers name different directories")
    return {
        "summary_path": summary.group("path"),
        "preflight_timestamp_utc": summary.group("ts"),
        "selected_gpu_uuid": selection.group("uuid"),
        "selected_gpu_name": selection.group("name"),
        "selected_gpu_driver_version": selection.group("driver"),
    }


def validate_preflight_summary(summary_path: Path, *, expected_git_commit: str,
                               expected_timestamp: str, expected_gpu_uuid: str) -> dict:
    """The complete preflight gate: a real, non-symlink, non-empty regular
    JSON file; overall PASS; the current clean commit; the explicitly selected
    GPU; and every one of the six required checks individually at PASS."""
    reject_unsafe_path(summary_path, expect_dir=False)
    if summary_path.stat().st_size == 0:
        raise RunFailure(f"{summary_path}: preflight summary is empty")
    try:
        document = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunFailure(f"{summary_path}: not readable JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise RunFailure(f"{summary_path}: preflight summary root is not a JSON object")

    errors: list[str] = []
    if document.get("overall_status") != "PASS":
        errors.append(f"overall_status={document.get('overall_status')!r} != 'PASS'")
    if document.get("git_dirty") is not False:
        errors.append(f"git_dirty={document.get('git_dirty')!r}; a campaign requires a clean tree")
    if document.get("git_commit") != expected_git_commit:
        errors.append(
            f"git_commit={document.get('git_commit')!r} != the current clean commit "
            f"{expected_git_commit!r}")
    if document.get("timestamp_utc") != expected_timestamp:
        errors.append(
            f"timestamp_utc={document.get('timestamp_utc')!r} != the directory this invocation "
            f"reported writing ({expected_timestamp!r})")
    gpu = document.get("gpu")
    if not isinstance(gpu, dict):
        errors.append("gpu: absent or not an object")
        gpu = {}
    if gpu.get("uuid") != expected_gpu_uuid:
        errors.append(
            f"gpu.uuid={gpu.get('uuid')!r} != the UUID the launcher resolved from the explicitly "
            f"selected physical index ({expected_gpu_uuid!r})")
    if gpu.get("compute_cap") != REQUIRED_COMPUTE_CAPABILITY:
        errors.append(
            f"gpu.compute_cap={gpu.get('compute_cap')!r} != {REQUIRED_COMPUTE_CAPABILITY!r}")
    if gpu.get("logical_index") != "0":
        errors.append(
            f"gpu.logical_index={gpu.get('logical_index')!r} != '0'; exactly one GPU must be "
            f"visible as CUDA logical device 0")
    for field in ("name", "driver_version"):
        value = gpu.get(field)
        if not isinstance(value, str) or not value:
            errors.append(f"gpu.{field}: absent or empty")

    checks = document.get("checks")
    if not isinstance(checks, list):
        errors.append("checks: absent or not a list")
    else:
        observed: dict[str, object] = {}
        for entry in checks:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                errors.append("checks: an entry is malformed")
                continue
            if entry["name"] in observed:
                errors.append(f"checks: duplicate check {entry['name']!r}")
            observed[entry["name"]] = entry.get("status")
        if sorted(observed) != sorted(REQUIRED_PREFLIGHT_CHECKS):
            errors.append(
                f"checks: {sorted(observed)} != the six required checks "
                f"{sorted(REQUIRED_PREFLIGHT_CHECKS)}")
        for name in REQUIRED_PREFLIGHT_CHECKS:
            if observed.get(name) != "PASS":
                errors.append(f"checks[{name}].status={observed.get(name)!r} != 'PASS'")
    if errors:
        raise RunFailure(f"{summary_path}: preflight validation failed: {errors}")

    return {
        "uuid": gpu["uuid"],
        "name": gpu["name"],
        "compute_capability": gpu["compute_cap"],
        "driver_version": gpu["driver_version"],
    }


# --------------------------------------------------------------------------
# Semantic loaders for the composed experiments. The real ones are the
# experiments' own validators; the acceptance tests inject synthetic
# equivalents so no real raw campaign is ever required.
# --------------------------------------------------------------------------


def _import_sibling(module_name: str):
    import importlib

    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    return importlib.import_module(module_name)


def _sha256_relative_regular_file(root: Path, relative_path: str) -> str:
    """Hash one validated relative file without following directory symlinks."""
    if (not isinstance(relative_path, str) or relative_path.startswith("/")
            or any(part in ("", ".", "..") for part in relative_path.split("/"))):
        raise RunFailure(f"invalid relative analysis artifact path {relative_path!r}")
    parts = relative_path.split("/")
    current_fd = open_dir_nofollow(str(root))
    try:
        for part in parts[:-1]:
            next_fd = open_dir_nofollow(part, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        payload = read_file_nofollow(parts[-1], dir_fd=current_fd)
    finally:
        os.close(current_fd)
    return sha256_bytes(payload)


def verify_terminal_analysis_artifacts(analyzer, campaign_dir: Path, manifest: dict) -> dict:
    """Re-hash the closed unit's canonical terminal ``analysis/`` outputs.

    The P1.4/P2.4 evidence-integrity functions reconstruct the raw pilot and
    profile evidence. Their terminal manifests additionally pin the derived
    analysis files. P4.1 verifies that small, explicit second set here using
    the canonical inventory exported by the owning analyzer; it does not
    reinterpret or recompute any scientific result.
    """
    state = manifest.get("state")
    if state not in UNIT_TERMINAL_STATES:
        return {}
    canonical = getattr(analyzer, "ANALYSIS_ARTIFACT_RELATIVE_PATHS", None)
    if (not isinstance(canonical, (tuple, list)) or not canonical
            or any(not isinstance(path, str) for path in canonical)):
        raise RunFailure("the unit analyzer exposes no usable canonical analysis inventory")
    expected = set(canonical)
    recorded = manifest.get("artifact_sha256")
    if not isinstance(recorded, dict):
        raise RunFailure("the terminal unit manifest has no artifact_sha256 object")
    observed = {
        path for path in recorded
        if isinstance(path, str) and path.startswith("analysis/")
    }
    if observed != expected:
        raise RunFailure(
            "the terminal unit manifest's analysis artifact inventory differs from its "
            f"canonical contract: missing={sorted(expected - observed)!r}, "
            f"unexpected={sorted(observed - expected)!r}")

    verified: dict[str, str] = {}
    for relative_path in sorted(expected):
        expected_hash = recorded.get(relative_path)
        if not isinstance(expected_hash, str) or not SHA256_RE.fullmatch(expected_hash):
            raise RunFailure(
                f"artifact_sha256[{relative_path!r}] is not a canonical SHA-256")
        actual_hash = _sha256_relative_regular_file(campaign_dir, relative_path)
        if actual_hash != expected_hash:
            raise RunFailure(
                f"{relative_path}: content changed after terminal analysis "
                f"(manifest records {expected_hash}, now {actual_hash})")
        verified[relative_path] = actual_hash
    return verified


def _load_unit_campaign(analyzer, campaign_rel: str, *, resolve, load) -> dict:
    """One composed unit's campaign, loaded through its own semantic loader and
    pinned cryptographically.

    Beyond the unit's own (manifest, revision) the payload carries the exact
    revision file that carried this state -- its repository-relative path and
    its SHA-256 -- plus, whenever the unit's own frozen protocol re-verifies its
    evidence at that state, a digest over the snapshot its own
    verify_campaign_evidence_integrity() just recomputed fresh from disk. Those
    three values are what a later revalidation re-derives and compares, so a
    synthetically valid *later* terminal revision can never quietly replace the
    revision this campaign originally accepted.

    P4.1 reimplements none of the semantics behind either function.
    """
    campaign_dir = resolve(campaign_rel)
    manifest, revision = load(campaign_dir)
    if revision < 0 or not manifest:
        raise RunFailure(f"{campaign_rel}: carries no manifest revision")
    manifest_rel = f"{campaign_rel}/{MANIFEST_DIR_NAME}/{revision:06d}.json"
    payload = {
        "manifest": manifest,
        "revision": revision,
        "campaign_dir": campaign_rel,
        "manifest_path": manifest_rel,
        "manifest_sha256": sha256_of_path(REPO_ROOT / manifest_rel),
    }
    if manifest.get("state") in UNIT_INTEGRITY_STATES:
        errors, snapshot = analyzer.verify_campaign_evidence_integrity(campaign_dir, manifest)
        if errors or snapshot is None:
            raise RunFailure(
                f"{campaign_rel}: the unit's own evidence-integrity gate failed: {errors}")
        evidence = {"unit_evidence": snapshot}
        if manifest.get("state") in UNIT_TERMINAL_STATES:
            evidence["terminal_analysis_sha256"] = verify_terminal_analysis_artifacts(
                analyzer, campaign_dir, manifest)
        payload["evidence_sha256"] = stable_digest(evidence)
    return payload


def default_p14_loader(campaign_id: str) -> dict:
    """Load the underlying P1.4 campaign through P1.4's own semantic loader,
    which re-opens, re-hashes, and revalidates its entire manifest chain."""
    analyzer = _import_sibling("analyze_exp01_memory_paths_p14")
    return _load_unit_campaign(
        analyzer, f"{P14_RAW_REL}/{campaign_id}",
        resolve=analyzer.resolve_p14_campaign_dir, load=analyzer.load_p14_manifest_chain)


def default_p24_loader(campaign_id: str) -> dict:
    """Load the underlying P2.4 campaign through P2.4's own semantic loader."""
    analyzer = _import_sibling("analyze_exp02_umma_throughput_p24")
    return _load_unit_campaign(
        analyzer, f"{P24_RAW_REL}/{campaign_id}",
        resolve=analyzer.resolve_p24_campaign_dir, load=analyzer.load_p24_manifest_chain)


def normalize_unit_provenance(provenance: object, *, label: str) -> dict:
    """The comparable provenance tuple of one composed unit, normalized.

    Every field both closed schemas expose is required by key and must be a
    non-empty scalar; a missing or malformed one fails closed. The two CUDA API
    versions are recorded as an int by one schema and as a string by the other,
    so they are normalized to their decimal text exactly the way P2.4's own
    compare_application_provenance() does -- a difference in textual formatting
    of the same version is never a mismatch, and a difference in the version
    itself always is.
    """
    if not isinstance(provenance, dict):
        raise RunFailure(f"{label}: the campaign provenance tuple is absent or not an object")
    snapshot: dict[str, str] = {}
    errors: list[str] = []
    for field in UNIT_PROVENANCE_FIELDS:
        if field not in provenance:
            errors.append(f"{field}: absent (mandatory campaign provenance field)")
            continue
        value = provenance[field]
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            errors.append(f"{field}={value!r}: must be a string or an integer")
            continue
        text = str(value).strip()
        if not text:
            errors.append(f"{field}: is empty")
            continue
        snapshot[field] = text
    if not errors and provenance.get("git_dirty") not in (None, False):
        # Only P4.1's own preflight and P3.5's rows expose a clean-tree flag
        # today; if a composed unit ever records one it must agree too.
        errors.append("git_dirty is not false; a campaign only ever runs on a clean tree")
    if errors:
        raise RunFailure(f"{label}: unusable campaign provenance: {errors}")
    return snapshot


def default_p35_validator(text: str) -> list[str]:
    """Validate a captured P3.5 CSV with P3.5's own frozen schema and its own
    validation function; never a second interpretation of the same contract."""
    checker = _import_sibling("check_gemm_comparison_p35")
    return list(checker.validate_serialized_output(text))


# --------------------------------------------------------------------------
# The P3.5 capture gate: P3.5's own validator, plus the Phase 4 bindings that
# only an enclosing campaign is in a position to check.
# --------------------------------------------------------------------------


def validate_gemm_capture(text: str, *, expected_git_commit: str, expected_gpu_uuid: str,
                          expected_gpu_name: str, expected_compute_capability: str,
                          expected_driver_version: str,
                          p35_validator=default_p35_validator) -> dict:
    """Fail closed unless the capture is exactly the frozen P3.5 evidence for
    this campaign. On any failure no Phase 4 GEMM artifact is ever accepted."""
    try:
        errors = list(p35_validator(text))
    except RunFailure:
        raise
    except Exception as exc:  # noqa: BLE001 - a capture P3.5's validator cannot even parse
        raise RunFailure(
            f"the captured P3.5 CSV is malformed: P3.5's own validator could not process it "
            f"({type(exc).__name__}: {exc})") from exc
    if errors:
        raise RunFailure(f"the captured P3.5 CSV failed P3.5's own validator: {errors}")

    rows = list(csv.DictReader(io.StringIO(text)))
    binding_errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        for field, expected in (
            ("git_commit", expected_git_commit),
            ("git_dirty", "false"),
            ("publishable", "false"),
            ("correctness", "PASS"),
            ("gpu_uuid", expected_gpu_uuid),
            ("gpu_name", expected_gpu_name),
            ("compute_capability", expected_compute_capability),
            ("driver_version", expected_driver_version),
        ):
            if row.get(field) != expected:
                binding_errors.append(
                    f"row {index}: {field}={row.get(field)!r} != the campaign value {expected!r}")
    if binding_errors:
        raise RunFailure(
            f"the captured P3.5 CSV does not belong to this Phase 4 campaign: {binding_errors}")
    return {"row_count": len(rows)}


# --------------------------------------------------------------------------
# Runtime context: everything the orchestrator touches outside pure logic.
# --------------------------------------------------------------------------

# Variables the orchestrator owns end to end. They are always removed from the
# inherited environment and re-set only for the stages that need them, so a
# stale operator export can never leak into a stage it does not belong to.
OWNED_ENV_VARS = (
    "BLACKWELL_GPU_INDEX",
    "P1_4_CAMPAIGN_ID", "P1_4_PREFLIGHT_SUMMARY",
    "P2_4_CAMPAIGN_ID", "P2_4_PREFLIGHT_SUMMARY",
    "RUN_CONTAINER_STDOUT_IS_DATA",
)


def _default_now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class Context:
    """Injectable runtime seams, so every acceptance test can exercise the real
    orchestration logic against temporary directories and stubbed children with
    no Docker, GPU, nvidia-smi, CUDA, or network involvement."""

    def __init__(self, *, repo_root: Path, run_command=None, now_utc=None, git_state=None,
                 p14_loader=None, p24_loader=None, p35_validator=None, stderr=None):
        self.repo_root = Path(repo_root)
        self.run_command = run_command if run_command is not None else self._default_run_command
        self.now_utc = now_utc if now_utc is not None else _default_now_utc
        self.git_state = git_state if git_state is not None else self._default_git_state
        self.p14_loader = p14_loader if p14_loader is not None else default_p14_loader
        self.p24_loader = p24_loader if p24_loader is not None else default_p24_loader
        self.p35_validator = p35_validator if p35_validator is not None else default_p35_validator
        self.stderr = stderr if stderr is not None else sys.stderr

    def log(self, message: str) -> None:
        print(f"run_all: {message}", file=self.stderr, flush=True)

    def _default_run_command(self, argv: list[str], *, env: dict[str, str],
                             stdout_fd: int, stderr_fd: int) -> int:
        """Run one child with its output written directly into already open,
        no-clobber descriptors: the child never performs its own path lookup
        for either output name."""
        child_env = {k: v for k, v in os.environ.items() if k not in OWNED_ENV_VARS}
        child_env.update(env)
        completed = subprocess.run(  # noqa: S603 - fixed argv, shell=False
            argv, cwd=str(self.repo_root), env=child_env,
            stdin=subprocess.DEVNULL, stdout=stdout_fd, stderr=stderr_fd,
            shell=False, check=False,
        )
        return completed.returncode

    def _default_git_state(self) -> tuple[str, bool]:
        try:
            commit = subprocess.run(  # noqa: S603 - fixed argv, shell=False
                ["git", "rev-parse", "--verify", "HEAD"], cwd=str(self.repo_root),
                capture_output=True, text=True, check=True, shell=False,
            ).stdout.strip()
            porcelain = subprocess.run(  # noqa: S603 - fixed argv, shell=False
                ["git", "status", "--porcelain"], cwd=str(self.repo_root),
                capture_output=True, text=True, check=True, shell=False,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise OrchestratorError(f"cannot determine the repository Git state: {exc}") from exc
        return commit, bool(porcelain.strip())


# --------------------------------------------------------------------------
# The orchestrator itself.
# --------------------------------------------------------------------------


class Campaign:
    """One top-level Phase 4 campaign: its plan, its manifest chain, its
    evidence, and the deterministic execution of its pending stages."""

    def __init__(self, context: Context, *, campaign_id: str, campaign_kind: str,
                 scope: str, gpu_index: int | None = None):
        self.context = context
        self.campaign_id = campaign_id
        self.campaign_kind = campaign_kind
        self.scope = scope
        self.gpu_index = gpu_index
        self.stage_order = build_stage_plan(scope)
        self.plan_document = build_plan_document(campaign_id, campaign_kind, scope)
        self.campaign_dir: Path | None = None
        self.manifest: dict = {}
        self.git_commit = ""

    # -- paths ----------------------------------------------------------

    @property
    def campaign_rel(self) -> str:
        return "/".join((*PHASE4_RAW_PARTS, self.campaign_id))

    def _campaign_fd(self) -> int:
        assert self.campaign_dir is not None
        parts = Path(repo_relative(self.campaign_dir, self.context.repo_root)).parts
        return open_dir_chain(self.context.repo_root, *parts)

    # -- manifest -------------------------------------------------------

    def _base_manifest(self, now: str) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "unit": UNIT,
            "campaign_id": self.campaign_id,
            "campaign_kind": self.campaign_kind,
            "scope": self.scope,
            "stage_order": list(self.stage_order),
            "plan_sha256": sha256_bytes(canonical_json_bytes(self.plan_document)),
            "git_commit": self.git_commit,
            "git_dirty": False,
            "publishable": PUBLISHABLE,
            "state": STATE_IN_PROGRESS,
            "created_at_utc": now,
            "updated_at_utc": now,
            "stage_attempts": [],
            "stage_results": {},
        }

    def _publish(self, content: dict) -> None:
        assert self.campaign_dir is not None
        revision = write_next_manifest_revision(
            self.campaign_dir, content, repo_root=self.context.repo_root)
        reloaded, reloaded_revision = load_manifest_chain(
            self.campaign_dir, repo_root=self.context.repo_root)
        if reloaded_revision != revision or reloaded != content:
            raise ManifestError("the manifest revision just published does not reload identically")
        self.manifest = reloaded

    def _next(self, *, drop: tuple[str, ...] = (), **updates) -> dict:
        content = json.loads(json.dumps(self.manifest))
        for key in drop:
            content.pop(key, None)
        content.update(updates)
        content["updated_at_utc"] = self.context.now_utc()
        return content

    # -- creation and resumption ----------------------------------------

    def create(self) -> None:
        self.campaign_dir = create_campaign_tree(
            self.campaign_id, repo_root=self.context.repo_root)
        campaign_fd = self._campaign_fd()
        try:
            write_file_exclusive(PLAN_FILE_NAME, canonical_json_bytes(self.plan_document),
                                 dir_fd=campaign_fd)
        finally:
            os.close(campaign_fd)
        now = self.context.now_utc()
        self._publish(self._base_manifest(now))
        self.context.log(
            f"campaign {self.campaign_id} created at {self.campaign_rel} "
            f"(kind={self.campaign_kind}, scope={self.scope}, publishable=false)")

    def load(self) -> None:
        self.campaign_dir = resolve_campaign_tree(
            self.campaign_id, repo_root=self.context.repo_root)
        manifest, revision = load_manifest_chain(
            self.campaign_dir, repo_root=self.context.repo_root)
        if revision < 0 or not manifest:
            raise OrchestratorError(
                f"{self.campaign_rel}: the campaign directory exists but carries no manifest "
                f"revision; refusing to adopt it")
        self.manifest = manifest
        self._check_immutable_agreement()
        self._check_plan_file()

    def _check_immutable_agreement(self) -> None:
        """A mismatch in campaign kind, scope, commit, or plan must abort."""
        mismatches = []
        if self.manifest["campaign_kind"] != self.campaign_kind:
            mismatches.append(
                f"campaign_kind: recorded {self.manifest['campaign_kind']!r}, requested "
                f"{self.campaign_kind!r}")
        if self.manifest["scope"] != self.scope:
            mismatches.append(
                f"scope: recorded {self.manifest['scope']!r}, requested {self.scope!r}")
        if self.manifest["git_commit"] != self.git_commit:
            mismatches.append(
                f"git_commit: recorded {self.manifest['git_commit']!r}, current "
                f"{self.git_commit!r}")
        if list(self.manifest["stage_order"]) != list(self.stage_order):
            mismatches.append("stage_order: the recorded plan differs from the requested plan")
        if mismatches:
            raise OrchestratorError(
                f"{self.campaign_rel}: refusing to resume; immutable campaign identity "
                f"mismatch: {mismatches}")

    def _check_plan_file(self) -> None:
        campaign_fd = self._campaign_fd()
        try:
            raw = read_file_nofollow(PLAN_FILE_NAME, dir_fd=campaign_fd)
        finally:
            os.close(campaign_fd)
        if sha256_bytes(raw) != self.manifest["plan_sha256"]:
            raise OrchestratorError(
                f"{self.campaign_rel}/{PLAN_FILE_NAME}: content no longer matches the hash "
                f"recorded in the manifest; refusing to regenerate it")
        if raw != canonical_json_bytes(self.plan_document):
            raise OrchestratorError(
                f"{self.campaign_rel}/{PLAN_FILE_NAME}: does not match the deterministic plan "
                f"for this campaign")

    # -- stage revalidation ---------------------------------------------

    def revalidate_stage(self, stage: str) -> None:
        """Re-hash and re-derive one recorded stage's evidence. A stage is
        never skipped merely because a directory or a filename exists."""
        record = self.manifest["stage_results"][stage]
        evidence = record["evidence"]
        if stage == STAGE_PREFLIGHT:
            self._revalidate_preflight(evidence)
        elif stage in (STAGE_MEMORY_PILOT, STAGE_MEMORY_PROFILE, STAGE_MEMORY_ANALYZE):
            self._revalidate_unit(stage, evidence, unit="P1.4")
        elif stage in (STAGE_UMMA_PILOT, STAGE_UMMA_PROFILE, STAGE_UMMA_ANALYZE):
            self._revalidate_unit(stage, evidence, unit="P2.4")
        elif stage == STAGE_GEMM_CAPTURE:
            self._revalidate_gemm(evidence)
        elif stage == STAGE_CAMPAIGN_VALIDATE:
            if evidence.get("outcome") not in (STATE_COMPLETE, STATE_INCONCLUSIVE):
                raise RunFailure("campaign.validate: the recorded outcome is not terminal")
        else:  # pragma: no cover - the plan tables are closed
            raise RunFailure(f"{stage}: no revalidation rule")

    def _revalidate_preflight(self, evidence: dict) -> None:
        for entry in evidence["preflights"]:
            path = self.context.repo_root / entry["summary_path"]
            digest = sha256_of_path(path)
            if digest != entry["summary_sha256"]:
                raise RunFailure(
                    f"{entry['summary_path']}: content changed since it was accepted "
                    f"(recorded {entry['summary_sha256']}, now {digest})")

    def _revalidate_unit(self, stage: str, evidence: dict, *, unit: str) -> None:
        """The underlying campaign legitimately advances along its own frozen
        state machine as later stages run (PILOT_COMPLETE -> COMPLETE ->
        ANALYZED), so a non-terminal recorded state is revalidated as "at or
        after", never as bit equality. A terminal recorded state must still be
        exactly what was accepted, and any state outside that unit's forward
        order -- a regression, FAILED, INTERRUPTED, or an unknown label -- is
        rejected."""
        loaded = self._load_unit(unit)
        state = loaded["manifest"].get("state")
        recorded_state = evidence.get("unit_state")
        if recorded_state in UNIT_TERMINAL_STATES:
            acceptable = state == recorded_state
        else:
            observed_rank = _unit_state_rank(unit, state)
            recorded_rank = _unit_state_rank(unit, recorded_state)
            acceptable = (observed_rank is not None and recorded_rank is not None
                          and observed_rank >= recorded_rank)
        if not acceptable:
            raise RunFailure(
                f"{stage}: the {unit} campaign is now in state {state!r}, but this campaign "
                f"accepted {recorded_state!r}")
        self._check_unit_pin(stage, evidence, loaded, unit=unit,
                             terminal=recorded_state in UNIT_TERMINAL_STATES)
        label = f"{stage}: the {unit} campaign"
        observed = normalize_unit_provenance(loaded["manifest"].get("provenance"), label=label)
        self._check_campaign_agreement(observed, label)
        changed = [f"{field}: accepted {evidence.get(field)!r}, now {value!r}"
                   for field, value in observed.items() if evidence.get(field) != value]
        if changed:
            raise RunFailure(f"{label} no longer reports the provenance accepted here: {changed}")

    def _check_unit_pin(self, stage: str, evidence: dict, loaded: dict, *, unit: str,
                        terminal: bool) -> None:
        """The exact manifest revision this campaign accepted must still be that
        exact file, byte for byte, and a unit accepted in a terminal state must
        still be at exactly that revision. A later terminal revision, however
        internally valid, is never silently adopted, and a changed accepted
        artifact is never regenerated."""
        recorded_path = evidence.get("unit_manifest_path")
        recorded_sha256 = evidence.get("unit_manifest_sha256")
        recorded_revision = evidence.get("unit_manifest_revision")
        if (not isinstance(recorded_path, str) or not recorded_path
                or not isinstance(recorded_sha256, str)
                or not SHA256_RE.fullmatch(recorded_sha256)
                or not isinstance(recorded_revision, int)
                or isinstance(recorded_revision, bool)):
            raise RunFailure(
                f"{stage}: the accepted unit evidence carries no usable manifest pin; refusing "
                f"to trust it")
        # The pin must name exactly this unit's own campaign manifest revision
        # for this campaign ID, so a recorded reference can never point at
        # another tree or outside the repository.
        unit_root = P14_RAW_REL if unit == "P1.4" else P24_RAW_REL
        expected_dir = f"{unit_root}/{self.campaign_id}"
        expected_path = f"{expected_dir}/{MANIFEST_DIR_NAME}/{recorded_revision:06d}.json"
        if evidence.get("unit_campaign_dir") != expected_dir or recorded_path != expected_path:
            raise RunFailure(
                f"{stage}: the accepted unit reference {recorded_path!r} is not this campaign's "
                f"own {unit} manifest revision {expected_path!r}")
        digest = sha256_of_path(self.context.repo_root / recorded_path)
        if digest != recorded_sha256:
            raise RunFailure(
                f"{recorded_path}: content changed since it was accepted "
                f"(recorded {recorded_sha256}, now {digest})")
        observed_revision = loaded.get("revision")
        if terminal:
            if observed_revision != recorded_revision or loaded.get("manifest_path") != recorded_path:
                raise RunFailure(
                    f"{stage}: the underlying campaign's terminal manifest is now revision "
                    f"{observed_revision!r} ({loaded.get('manifest_path')!r}), but this campaign "
                    f"accepted revision {recorded_revision} ({recorded_path}); a later terminal "
                    f"revision is never adopted silently")
            recorded_evidence = evidence.get("unit_evidence_sha256")
            if not isinstance(recorded_evidence, str) or not SHA256_RE.fullmatch(recorded_evidence):
                raise RunFailure(
                    f"{stage}: a terminal unit was accepted without an evidence-integrity pin")
            if loaded.get("evidence_sha256") != recorded_evidence:
                raise RunFailure(
                    f"{stage}: the underlying campaign's own evidence-integrity snapshot no "
                    f"longer matches the one accepted here; refusing to regenerate it")
        elif (not isinstance(observed_revision, int) or isinstance(observed_revision, bool)
                or observed_revision < recorded_revision):
            raise RunFailure(
                f"{stage}: the underlying campaign is now at manifest revision "
                f"{observed_revision!r}, before the revision {recorded_revision} accepted here")

    def _revalidate_gemm(self, evidence: dict) -> None:
        path = self.context.repo_root / evidence["csv_path"]
        digest = sha256_of_path(path)
        if digest != evidence["csv_sha256"]:
            raise RunFailure(
                f"{evidence['csv_path']}: content changed since it was accepted "
                f"(recorded {evidence['csv_sha256']}, now {digest})")
        gpu = self.manifest["gpu"]
        validate_gemm_capture(
            path.read_text(encoding="utf-8"),
            expected_git_commit=self.git_commit,
            expected_gpu_uuid=gpu["uuid"],
            expected_gpu_name=gpu["name"],
            expected_compute_capability=gpu["compute_capability"],
            expected_driver_version=gpu["driver_version"],
            p35_validator=self.context.p35_validator,
        )

    def _load_unit(self, unit: str) -> dict:
        loader = self.context.p14_loader if unit == "P1.4" else self.context.p24_loader
        try:
            return loader(self.campaign_id)
        except Exception as exc:  # noqa: BLE001 - any loader defect fails closed
            raise RunFailure(
                f"the {unit} campaign {self.campaign_id} is not loadable: {exc}") from exc

    # -- stage execution ------------------------------------------------

    def _attempt_number(self, stage: str) -> int:
        """The next free attempt number for one stage, derived from both the
        validated campaign state and the no-clobber log files already on disk.

        An attempt claims its number by creating its two logs *before* the child
        runs, so a hard interruption can leave a log set whose attempt record was
        never published. Taking the maximum of both sources means an earlier
        log is never reused or overwritten, and an interrupted stage stays
        eligible for a new attempt on --resume instead of colliding forever.
        """
        recorded = max((entry["attempt"] for entry in self.manifest["stage_attempts"]
                        if entry["stage"] == stage), default=0)
        return max(recorded, self._highest_log_attempt(stage)) + 1

    def _highest_log_attempt(self, stage: str) -> int:
        """The highest attempt number this stage already has a log for."""
        safe = stage.replace(".", "_")
        campaign_fd = self._campaign_fd()
        try:
            logs_fd = open_dir_nofollow(LOGS_DIR_NAME, dir_fd=campaign_fd)
            try:
                names = os.listdir(logs_fd)
            finally:
                os.close(logs_fd)
        finally:
            os.close(campaign_fd)
        highest = 0
        for name in names:
            match = LOG_NAME_RE.fullmatch(name)
            if match is not None and match.group("stage") == safe:
                highest = max(highest, int(match.group("attempt")))
        return highest

    def _log_names(self, stage: str, attempt: int) -> tuple[str, str]:
        safe = stage.replace(".", "_")
        return f"{safe}.{attempt:03d}.stdout.log", f"{safe}.{attempt:03d}.stderr.log"

    def _log_rel(self, name: str) -> str:
        return f"{self.campaign_rel}/{LOGS_DIR_NAME}/{name}"

    def _run_stage_command(self, argv: list[str], env: dict[str, str], *, stdout_name: str,
                           stderr_name: str, created: list[str]) -> tuple[int, bytes]:
        """Execute one child with descriptor-anchored, no-clobber capture.

        The child writes through two streaming filters. Before those bytes cross
        the durable log boundary, only this checkout's exact absolute root is
        replaced. P3.5's stdout is scientific CSV data and is copied byte for
        byte; its textual stderr still receives the narrow replacement. Already
        copied diagnostics remain available if the child is interrupted.
        """
        campaign_fd = self._campaign_fd()
        try:
            logs_fd = open_dir_nofollow(LOGS_DIR_NAME, dir_fd=campaign_fd)
        except BaseException:
            os.close(campaign_fd)
            raise
        try:
            stdout_fd = create_file_exclusive(stdout_name, dir_fd=logs_fd)
            created.append(stdout_name)
            try:
                stderr_fd = create_file_exclusive(stderr_name, dir_fd=logs_fd)
                created.append(stderr_name)
                try:
                    stdout_read, stdout_write = os.pipe()
                    stderr_read, stderr_write = os.pipe()
                    stream_errors: list[BaseException] = []

                    def pump(read_fd: int, write_fd: int, *, redact: bool) -> None:
                        try:
                            copy_log_stream(
                                read_fd, write_fd, repo_root=self.context.repo_root,
                                redact=redact)
                        except BaseException as exc:  # noqa: BLE001 - reported after both joins
                            stream_errors.append(exc)

                    threads = (
                        threading.Thread(
                            target=pump, args=(stdout_read, stdout_fd),
                            kwargs={"redact": env.get("RUN_CONTAINER_STDOUT_IS_DATA") != "1"}),
                        threading.Thread(
                            target=pump, args=(stderr_read, stderr_fd),
                            kwargs={"redact": True}),
                    )
                    for thread in threads:
                        thread.start()
                    try:
                        status = self.context.run_command(
                            argv, env=env, stdout_fd=stdout_write, stderr_fd=stderr_write)
                    finally:
                        os.close(stdout_write)
                        os.close(stderr_write)
                        for thread in threads:
                            thread.join()
                        os.fsync(stdout_fd)
                        os.fsync(stderr_fd)
                    if stream_errors:
                        raise RunFailure(
                            "failed while publishing a redacted child log: "
                            + "; ".join(
                                f"{type(exc).__name__}: {exc}" for exc in stream_errors))
                finally:
                    os.close(stderr_fd)
            finally:
                os.close(stdout_fd)
            captured = read_file_nofollow(stdout_name, dir_fd=logs_fd)
        finally:
            os.close(logs_fd)
            os.close(campaign_fd)
        if not isinstance(status, int) or isinstance(status, bool):
            raise RunFailure("the child command returned a non-integer exit status")
        return status, captured

    def _attempt_record(self, stage: str, attempt: int, started: str, finished: str,
                        outcome: str, exit_status: int, stdout_name: str,
                        stderr_name: str) -> dict:
        return {
            "stage": stage,
            "attempt": attempt,
            "started_at_utc": started,
            "finished_at_utc": finished,
            "outcome": outcome,
            "exit_status": exit_status,
            "stdout_log": self._log_rel(stdout_name),
            "stderr_log": self._log_rel(stderr_name),
        }

    def _stage_env(self, stage: str) -> dict[str, str]:
        """Every P1.4/P2.4 stage receives this campaign's own ID and the exact
        preflight summary that this invocation created and validated."""
        env: dict[str, str] = {}
        if stage in GPU_STAGES or stage == STAGE_PREFLIGHT:
            if self.gpu_index is None:
                raise OrchestratorError(
                    f"{stage}: refusing to run GPU work without an explicit BLACKWELL_GPU_INDEX")
            env["BLACKWELL_GPU_INDEX"] = str(self.gpu_index)
        reference = self.manifest.get("preflight_reference")
        preflight_path = reference.get("summary_path") if isinstance(reference, dict) else None
        if stage in (STAGE_MEMORY_PILOT, STAGE_MEMORY_PROFILE):
            env["P1_4_CAMPAIGN_ID"] = self.campaign_id
            env["P1_4_PREFLIGHT_SUMMARY"] = _require_preflight_path(preflight_path, stage)
        elif stage == STAGE_MEMORY_ANALYZE:
            env["P1_4_CAMPAIGN_ID"] = self.campaign_id
        elif stage in (STAGE_UMMA_PILOT, STAGE_UMMA_PROFILE):
            env["P2_4_CAMPAIGN_ID"] = self.campaign_id
            env["P2_4_PREFLIGHT_SUMMARY"] = _require_preflight_path(preflight_path, stage)
        elif stage == STAGE_UMMA_ANALYZE:
            env["P2_4_CAMPAIGN_ID"] = self.campaign_id
        if stage == STAGE_GEMM_CAPTURE:
            # The P3.5 smoke writes its CSV to stdout and everything else to
            # stderr; this keeps the launcher's own banner out of the CSV too.
            env["RUN_CONTAINER_STDOUT_IS_DATA"] = "1"
        return env

    def execute_stage(self, stage: str) -> None:
        attempt = self._attempt_number(stage)
        stdout_name, stderr_name = self._log_names(stage, attempt)
        # Silent and without directory diagnostics: Make never echoes a recipe
        # line, so an absolute bind-mount source can never reach a durable log.
        # The target's own behaviour and the invoked scripts' output are
        # unchanged.
        argv = ["make", *MAKE_QUIET_FLAGS, STAGE_MAKE_TARGET[stage]]
        env = self._stage_env(stage)
        started = self.context.now_utc()
        created: list[str] = []
        self.context.log(f"stage {stage}: attempt {attempt}: {' '.join(argv)}")
        try:
            status, captured = self._run_stage_command(
                argv, env, stdout_name=stdout_name, stderr_name=stderr_name, created=created)
        except KeyboardInterrupt:
            self._interrupt(
                stage, attempt, started, stdout_name, stderr_name,
                stdout_name in created and stderr_name in created)
            raise
        finished = self.context.now_utc()

        if status != 0:
            self._fail(stage, attempt, started, finished, status, stdout_name, stderr_name,
                       [f"{' '.join(argv)} exited with status {status}"])
            raise RunFailure(
                f"stage {stage} failed with exit status {status}; no downstream stage was "
                f"executed and every completed stage's evidence is preserved")

        try:
            accepted = self._accept_stage(stage, captured, stdout_name)
        except (RunFailure, OrchestratorError) as exc:
            self._fail(stage, attempt, started, finished, 1, stdout_name, stderr_name, [str(exc)])
            raise RunFailure(f"stage {stage} produced unacceptable evidence: {exc}") from exc

        attempts = list(self.manifest["stage_attempts"])
        attempts.append(self._attempt_record(
            stage, attempt, started, finished, "COMPLETE", 0, stdout_name, stderr_name))
        results = dict(self.manifest["stage_results"])
        if stage == STAGE_PREFLIGHT and stage in results:
            history = list(_preflight_history(results))
            history.append(accepted["preflight_entry"])
            results[stage] = {"attempt": attempt, "completed_at_utc": finished,
                              "evidence": {"preflights": history}}
        else:
            results[stage] = {"attempt": attempt, "completed_at_utc": finished,
                              "evidence": accepted["stage_evidence"]}
        updates = {"stage_attempts": attempts, "stage_results": results,
                   "state": STATE_IN_PROGRESS}
        updates.update(accepted.get("manifest_updates", {}))
        self._publish(self._next(drop=("failure_stage", "failure_detail"), **updates))
        self.context.log(f"stage {stage}: COMPLETE (publishable=false)")

    def _fail(self, stage: str, attempt: int, started: str, finished: str, status: int,
              stdout_name: str, stderr_name: str, detail: list[str]) -> None:
        attempts = list(self.manifest["stage_attempts"])
        attempts.append(self._attempt_record(
            stage, attempt, started, finished, "FAILED", status if status else 1,
            stdout_name, stderr_name))
        self._publish(self._next(
            stage_attempts=attempts, state=STATE_FAILED, failure_stage=stage,
            failure_detail=[redact_repo_root(str(item), self.context.repo_root)[:400]
                            for item in detail]))

    def _interrupt(self, stage: str, attempt: int, started: str, stdout_name: str,
                   stderr_name: str, complete_log_pair: bool) -> None:
        """Preserve every completed stage's evidence and record exactly where
        the campaign stopped. Nothing an earlier attempt produced is deleted."""
        if self.manifest.get("state") in TERMINAL_STATES:
            return
        updates: dict = {"state": STATE_INTERRUPTED, "failure_stage": stage,
                         "failure_detail": ["interrupted before the stage finished"]}
        if complete_log_pair:
            attempts = list(self.manifest["stage_attempts"])
            finished = self.context.now_utc()
            attempts.append(self._attempt_record(
                stage, attempt, started, finished, "INTERRUPTED", INTERRUPTED_EXIT_STATUS,
                stdout_name, stderr_name))
            updates["stage_attempts"] = attempts
        self._publish(self._next(**updates))

    # -- per-stage acceptance -------------------------------------------

    def _accept_stage(self, stage: str, captured: bytes, stdout_name: str) -> dict:
        if stage == STAGE_PREFLIGHT:
            return self._accept_preflight(captured)
        if stage in (STAGE_MEMORY_PILOT, STAGE_MEMORY_PROFILE, STAGE_MEMORY_ANALYZE):
            return self._accept_unit(stage, unit="P1.4")
        if stage in (STAGE_UMMA_PILOT, STAGE_UMMA_PROFILE, STAGE_UMMA_ANALYZE):
            return self._accept_unit(stage, unit="P2.4")
        if stage == STAGE_GEMM_CAPTURE:
            return self._accept_gemm(captured, stdout_name)
        raise RunFailure(f"{stage}: no acceptance rule")  # pragma: no cover

    def _accept_preflight(self, captured: bytes) -> dict:
        assert self.gpu_index is not None
        text = captured.decode("utf-8", errors="replace")
        parsed = parse_preflight_stdout(text, selected_gpu_index=self.gpu_index)
        summary_path = self.context.repo_root / parsed["summary_path"]
        gpu = validate_preflight_summary(
            summary_path,
            expected_git_commit=self.git_commit,
            expected_timestamp=parsed["preflight_timestamp_utc"],
            expected_gpu_uuid=parsed["selected_gpu_uuid"],
        )
        recorded_gpu = self.manifest.get("gpu")
        if isinstance(recorded_gpu, dict) and recorded_gpu != gpu:
            raise RunFailure(
                f"the fresh preflight ran on {gpu!r}, but this campaign already recorded "
                f"{recorded_gpu!r}; refusing to mix devices inside one campaign")
        entry = {
            "summary_path": parsed["summary_path"],
            "summary_sha256": sha256_of_path(summary_path),
            "preflight_timestamp_utc": parsed["preflight_timestamp_utc"],
            "selected_gpu_index": self.gpu_index,
        }
        return {
            "stage_evidence": {"preflights": [entry]},
            "preflight_entry": entry,
            "manifest_updates": {"preflight_reference": dict(entry), "gpu": gpu},
        }

    def _accept_unit(self, stage: str, *, unit: str) -> dict:
        loaded = self._load_unit(unit)
        manifest = loaded["manifest"]
        state = manifest.get("state")
        required = STAGE_REQUIRED_UNIT_STATE[stage]
        if stage == STAGE_UMMA_ANALYZE:
            if state not in (required, P24_TERMINAL_INCONCLUSIVE):
                raise RunFailure(
                    f"the {unit} campaign reached state {state!r}, not {required!r} or "
                    f"{P24_TERMINAL_INCONCLUSIVE!r}")
        elif state != required:
            raise RunFailure(
                f"the {unit} campaign reached state {state!r}, not the required {required!r}")
        label = f"{stage}: the {unit} campaign"
        provenance = normalize_unit_provenance(manifest.get("provenance"), label=label)
        self._check_campaign_agreement(provenance, label)
        evidence = {
            "unit": unit,
            "unit_campaign_dir": loaded["campaign_dir"],
            "unit_manifest_path": loaded["manifest_path"],
            "unit_manifest_revision": loaded["revision"],
            "unit_manifest_sha256": loaded["manifest_sha256"],
            "unit_state": state,
        }
        if state in UNIT_TERMINAL_STATES:
            pin = loaded.get("evidence_sha256")
            if not isinstance(pin, str) or not SHA256_RE.fullmatch(pin):
                raise RunFailure(
                    f"{label} reached the terminal state {state!r} without its own "
                    f"evidence-integrity snapshot; refusing to accept it")
            evidence["unit_evidence_sha256"] = pin
        evidence.update(provenance)
        return {"stage_evidence": evidence}

    def _check_campaign_agreement(self, provenance: dict, label: str) -> None:
        """Every component of one campaign must agree with this invocation's own
        validated preflight, and with every component already accepted, on all
        directly comparable fields."""
        commit = provenance["git_commit"]
        if commit != self.git_commit:
            raise RunFailure(
                f"{label}: git_commit={commit!r} != the campaign commit {self.git_commit!r}")
        gpu = self.manifest.get("gpu")
        if not isinstance(gpu, dict):
            raise RunFailure(
                f"{label}: the campaign carries no validated preflight GPU identity to compare "
                f"against")
        for field, gpu_key in UNIT_TO_PREFLIGHT_GPU_FIELDS:
            if provenance[field] != gpu[gpu_key]:
                raise RunFailure(
                    f"{label}: {field}={provenance[field]!r} != the campaign's validated preflight "
                    f"{gpu_key}={gpu[gpu_key]!r}")
        for stage, record in self.manifest["stage_results"].items():
            accepted = record["evidence"]
            for field in COMPARABLE_EVIDENCE_FIELDS:
                other = accepted.get(field)
                if field in provenance and other is not None and other != provenance[field]:
                    raise RunFailure(
                        f"{label}: {field}={provenance[field]!r} != {other!r}, already accepted "
                        f"from stage {stage!r}")

    def _accept_gemm(self, captured: bytes, stdout_name: str) -> dict:
        gpu = self.manifest.get("gpu")
        if not isinstance(gpu, dict):
            raise RunFailure("gemm.capture: no validated preflight GPU identity to bind to")
        try:
            text = captured.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise RunFailure(
                f"gemm.capture: the captured stdout is not valid UTF-8: {exc}") from exc
        summary = validate_gemm_capture(
            text,
            expected_git_commit=self.git_commit,
            expected_gpu_uuid=gpu["uuid"],
            expected_gpu_name=gpu["name"],
            expected_compute_capability=gpu["compute_capability"],
            expected_driver_version=gpu["driver_version"],
            p35_validator=self.context.p35_validator,
        )
        campaign_fd = self._campaign_fd()
        try:
            logs_fd = open_dir_nofollow(LOGS_DIR_NAME, dir_fd=campaign_fd)
            try:
                exp03_fd = open_dir_nofollow(EXP03_DIR_NAME, dir_fd=campaign_fd)
                try:
                    link_no_clobber(stdout_name, GEMM_ARTIFACT_NAME,
                                    src_dir_fd=logs_fd, dst_dir_fd=exp03_fd)
                finally:
                    os.close(exp03_fd)
            finally:
                os.close(logs_fd)
        finally:
            os.close(campaign_fd)
        csv_path = self.context.repo_root / self.campaign_rel / GEMM_ARTIFACT_REL
        digest = sha256_of_path(csv_path)
        if digest != sha256_bytes(captured):
            raise RunFailure(
                "gemm.capture: the published artifact does not hash to the validated capture")
        return {"stage_evidence": {
            "csv_path": f"{self.campaign_rel}/{GEMM_ARTIFACT_REL}",
            "csv_sha256": digest,
            "row_count": summary["row_count"],
            "git_commit": self.git_commit,
            "gpu_uuid": gpu["uuid"],
            "gpu_name": gpu["name"],
            "compute_capability": gpu["compute_capability"],
        }}

    # -- the final top-level integrity gate ------------------------------

    def run_campaign_validate(self) -> str:
        """Re-hash and revalidate every selected component immediately before
        the campaign is declared complete, and require all of them to agree on
        campaign ID, Git commit, GPU identity, and clean-tree status.

        An interruption is handled at exactly the same orchestration boundary as
        every other stage: the campaign is recorded INTERRUPTED, every artifact
        and log already created is preserved, and the next --resume claims a
        fresh attempt number rather than reusing this one.
        """
        attempt = self._attempt_number(STAGE_CAMPAIGN_VALIDATE)
        stdout_name, stderr_name = self._log_names(STAGE_CAMPAIGN_VALIDATE, attempt)
        started = self.context.now_utc()
        self.context.log(f"stage {STAGE_CAMPAIGN_VALIDATE}: attempt {attempt}: integrity gate")
        created: list[str] = []
        try:
            return self._campaign_validate(attempt, started, stdout_name, stderr_name, created)
        except KeyboardInterrupt:
            self._interrupt(STAGE_CAMPAIGN_VALIDATE, attempt, started, stdout_name, stderr_name,
                            stdout_name in created and stderr_name in created)
            raise

    def _campaign_validate(self, attempt: int, started: str, stdout_name: str, stderr_name: str,
                           created: list[str]) -> str:
        errors: list[str] = []
        report: list[str] = [
            f"campaign_id: {self.campaign_id}",
            f"campaign_kind: {self.campaign_kind}",
            f"scope: {self.scope}",
            f"git_commit: {self.git_commit}",
            "publishable: false",
        ]
        for stage in self.stage_order:
            if stage == STAGE_CAMPAIGN_VALIDATE:
                continue
            if stage not in self.manifest["stage_results"]:
                errors.append(f"{stage}: no completed stage record")
                continue
            try:
                self.revalidate_stage(stage)
                report.append(f"{stage}: revalidated")
            except (RunFailure, OrchestratorError) as exc:
                errors.append(f"{stage}: {exc}")

        gpu = self.manifest.get("gpu")
        if not isinstance(gpu, dict):
            errors.append("the campaign carries no validated preflight GPU identity")

        # Every directly comparable provenance field, collected from this
        # invocation's own validated preflight and from every component's
        # recorded evidence, must resolve to exactly one value.
        observed: dict[str, set] = {field: set() for field in COMPARABLE_EVIDENCE_FIELDS}
        observed["git_commit"].add(self.git_commit)
        if isinstance(gpu, dict):
            for field, gpu_key in UNIT_TO_PREFLIGHT_GPU_FIELDS:
                observed[field].add(gpu[gpu_key])
        for record in self.manifest["stage_results"].values():
            evidence = record["evidence"]
            for field in COMPARABLE_EVIDENCE_FIELDS:
                value = evidence.get(field)
                if value is not None:
                    observed[field].add(value)
        for field in COMPARABLE_EVIDENCE_FIELDS:
            if len(observed[field]) > 1:
                errors.append(
                    f"the composed experiments disagree on {field}: {sorted(observed[field])}")

        outcome = STATE_COMPLETE
        umma = self.manifest["stage_results"].get(STAGE_UMMA_ANALYZE)
        if umma and umma["evidence"].get("unit_state") == P24_TERMINAL_INCONCLUSIVE:
            outcome = STATE_INCONCLUSIVE
            report.append("umma.analyze: P2.4 reached INCONCLUSIVE")
            self.context.log(
                "the P2.4 analysis is INCONCLUSIVE; the top-level outcome is INCONCLUSIVE, "
                "never a complete campaign")
        report.append(f"outcome: {outcome}")

        # The durable log-write boundary of this unit's own output: a diagnostic
        # may legitimately quote an absolute evidence path, and only this
        # checkout's own root is replaced by a stable token on the way out.
        report_bytes = redact_repo_root("\n".join(report) + "\n",
                                        self.context.repo_root).encode("utf-8")
        errors_bytes = (redact_repo_root("\n".join(errors) + "\n",
                                         self.context.repo_root).encode("utf-8")
                        if errors else b"")
        campaign_fd = self._campaign_fd()
        try:
            logs_fd = open_dir_nofollow(LOGS_DIR_NAME, dir_fd=campaign_fd)
            try:
                write_file_exclusive(stdout_name, report_bytes, dir_fd=logs_fd)
                created.append(stdout_name)
                write_file_exclusive(stderr_name, errors_bytes, dir_fd=logs_fd)
                created.append(stderr_name)
            finally:
                os.close(logs_fd)
        finally:
            os.close(campaign_fd)

        finished = self.context.now_utc()
        if errors:
            self._fail(STAGE_CAMPAIGN_VALIDATE, attempt, started, finished, 1,
                       stdout_name, stderr_name, errors)
            raise RunFailure(f"the final campaign integrity gate failed: {errors}")

        attempts = list(self.manifest["stage_attempts"])
        attempts.append(self._attempt_record(
            STAGE_CAMPAIGN_VALIDATE, attempt, started, finished, "COMPLETE", 0,
            stdout_name, stderr_name))
        results = dict(self.manifest["stage_results"])
        results[STAGE_CAMPAIGN_VALIDATE] = {
            "attempt": attempt,
            "completed_at_utc": finished,
            "evidence": {"validated_at_utc": finished, "outcome": outcome},
        }
        self._publish(self._next(
            drop=("failure_stage", "failure_detail"),
            stage_attempts=attempts, stage_results=results, state=outcome, outcome=outcome))
        return outcome

    # -- the driver ------------------------------------------------------

    def pending_stages(self) -> list[str]:
        """A stage is pending unless it is already recorded complete. The
        preflight is always re-executed while any GPU work remains pending, so
        every GPU stage of every invocation runs behind a fresh, validated
        preflight taken from that same invocation."""
        done = self.manifest["stage_results"]
        pending = [stage for stage in self.stage_order if stage not in done]
        if any(stage in GPU_STAGES for stage in pending) and STAGE_PREFLIGHT not in pending:
            pending.insert(0, STAGE_PREFLIGHT)
        return pending

    def revalidate_completed(self) -> None:
        for stage in self.stage_order:
            if stage in self.manifest["stage_results"]:
                self.revalidate_stage(stage)
                self.context.log(f"stage {stage}: already complete and revalidated; skipping")

    def run(self) -> int:
        pending = self.pending_stages()
        if not pending:
            state = self.manifest["state"]
            self.context.log(
                f"campaign {self.campaign_id} is already {state}; nothing to execute")
            return EXIT_OK if state == STATE_COMPLETE else EXIT_INCONCLUSIVE

        if self.manifest["state"] in (STATE_FAILED, STATE_INTERRUPTED):
            self._publish(self._next(
                drop=("failure_stage", "failure_detail"), state=STATE_IN_PROGRESS))

        current_stage = pending[0]
        for stage in pending:
            current_stage = stage
            if stage == STAGE_CAMPAIGN_VALIDATE:
                outcome = self.run_campaign_validate()
                return EXIT_OK if outcome == STATE_COMPLETE else EXIT_INCONCLUSIVE
            self.execute_stage(stage)
        return EXIT_OK  # pragma: no cover - every plan ends with campaign.validate


def _unit_state_rank(unit: str, state: object) -> int | None:
    """Where a composed unit's state sits in its own frozen forward order.
    P2.4's INCONCLUSIVE is a sibling terminal of ANALYZED, so it ranks with it.
    A state outside that order (a regression, FAILED, INTERRUPTED, or an
    unknown label) has no rank at all and can never satisfy an "at or after"
    comparison."""
    order = P14_STATE_ORDER if unit == "P1.4" else P24_STATE_ORDER
    if unit == "P2.4" and state == P24_TERMINAL_INCONCLUSIVE:
        return len(order) - 1
    if not isinstance(state, str) or state not in order:
        return None
    return order.index(state)


def _require_preflight_path(path: object, stage: str) -> str:
    if not isinstance(path, str) or not path:
        raise OrchestratorError(
            f"{stage}: no validated preflight summary is recorded for this invocation")
    return path


def _require_gpu_index(text: object) -> int:
    """Reject a missing or non-numeric index before Docker, nvidia-smi, result
    creation, or any GPU-related subprocess."""
    if text is None or text == "":
        raise OrchestratorError(
            "BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index; this project "
            "never selects a GPU automatically")
    if not isinstance(text, str) or not GPU_INDEX_RE.fullmatch(text):
        raise OrchestratorError(
            f"BLACKWELL_GPU_INDEX must be a non-negative integer, got {text!r}")
    return int(text)


# --------------------------------------------------------------------------
# Public entry point used by scripts/run_all.sh.
# --------------------------------------------------------------------------


def run_campaign(context: Context, *, campaign_id: str, campaign_kind: str,
                 selector: str | None, gpu_index_text: str | None, resume: bool) -> int:
    """Execute (or resume) one top-level campaign. Every mutation happens only
    after the GPU-selection and repository-state preconditions have passed."""
    validate_campaign_id(campaign_id)
    validate_campaign_kind(campaign_kind)
    scope = scope_for_selector(selector)

    exists = campaign_exists(campaign_id, repo_root=context.repo_root)
    if exists and not resume:
        raise OrchestratorError(
            f"results/raw/phase4/{campaign_id} already exists; refusing to reuse it. Pass "
            f"--resume to continue that campaign, or choose a new --campaign-id.")
    if resume and not exists:
        raise OrchestratorError(
            f"results/raw/phase4/{campaign_id} does not exist; --resume requires an existing "
            f"matching top-level campaign.")

    commit, dirty = context.git_state()
    if not GIT_COMMIT_RE.fullmatch(commit or ""):
        raise OrchestratorError(f"HEAD {commit!r} is not a 40-character lowercase hex commit")
    if dirty:
        raise OrchestratorError(
            "the working tree is dirty; a Phase 4 campaign only ever runs on a clean commit")

    campaign = Campaign(context, campaign_id=campaign_id, campaign_kind=campaign_kind, scope=scope)
    campaign.git_commit = commit

    if resume:
        campaign.load()
        state = campaign.manifest["state"]
        if state in (STATE_COMPLETE, STATE_INCONCLUSIVE):
            campaign.revalidate_completed()
            context.log(
                f"campaign {campaign_id} is terminal {state}; pure revalidation succeeded, no "
                f"artifact was rewritten and no manifest revision was appended")
            return EXIT_OK if state == STATE_COMPLETE else EXIT_INCONCLUSIVE
        if state == STATE_FAILED:
            failure_stage = campaign.manifest.get("failure_stage")
            if failure_stage not in RESUMABLE_FAILURE_STAGES:
                raise OrchestratorError(
                    f"campaign {campaign_id} failed terminally at stage {failure_stage!r}; the "
                    f"underlying experiment campaign cannot be reopened by its own frozen "
                    f"protocol. Start a new campaign with a new --campaign-id.")
        campaign.revalidate_completed()
        pending = campaign.pending_stages()
    else:
        pending = list(campaign.stage_order)

    if any(stage in GPU_STAGES or stage == STAGE_PREFLIGHT for stage in pending):
        campaign.gpu_index = _require_gpu_index(gpu_index_text)

    if not resume:
        campaign.create()

    return campaign.run()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phase4_orchestrator.py",
        description="P4.1 private helper. The public Phase 4 entry point is scripts/run_all.sh.",
    )
    parser.add_argument("--self-test", action="store_true",
                        help="run GPU-free synthetic checks and exit")
    sub = parser.add_subparsers(dest="command")

    plan = sub.add_parser("plan", help="print the deterministic stage plan; no mutation")
    plan.add_argument("--campaign-id", required=True)
    plan.add_argument("--campaign-kind", required=True, choices=list(CAMPAIGN_KINDS))
    plan.add_argument("--only", choices=list(SELECTORS), default=None)

    run = sub.add_parser("run", help="execute or resume one campaign")
    run.add_argument("--campaign-id", required=True)
    run.add_argument("--campaign-kind", required=True, choices=list(CAMPAIGN_KINDS))
    run.add_argument("--only", choices=list(SELECTORS), default=None)
    run.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return _self_test()
    if args.command == "plan":
        validate_campaign_id(args.campaign_id)
        validate_campaign_kind(args.campaign_kind)
        scope = scope_for_selector(args.only)
        sys.stdout.write(render_plan(
            build_plan_document(args.campaign_id, args.campaign_kind, scope)))
        return EXIT_OK
    if args.command == "run":
        context = Context(repo_root=REPO_ROOT)
        return run_campaign(
            context, campaign_id=args.campaign_id, campaign_kind=args.campaign_kind,
            selector=args.only, gpu_index_text=os.environ.get("BLACKWELL_GPU_INDEX"),
            resume=args.resume)
    parser.print_help(sys.stderr)
    return EXIT_PRECONDITION


# --------------------------------------------------------------------------
# GPU-free self-test. The exhaustive acceptance suite lives in
# scripts/check_phase4_orchestrator_p41.py; these checks keep this module
# independently runnable and are re-run by that checker.
# --------------------------------------------------------------------------


def _self_test() -> int:
    failures: list[str] = []

    def check(label: str, condition: bool, detail: str = "") -> None:
        if condition:
            print(f"phase4_orchestrator: self-test: PASS: {label}")
        else:
            failures.append(f"{label}: {detail}")
            print(f"phase4_orchestrator: self-test: FAIL: {label}: {detail}", file=sys.stderr)

    check("the full plan is the frozen nine-stage order",
          STAGE_PLANS["full"] == (
              STAGE_PREFLIGHT, STAGE_MEMORY_PILOT, STAGE_MEMORY_PROFILE, STAGE_UMMA_PILOT,
              STAGE_UMMA_PROFILE, STAGE_GEMM_CAPTURE, STAGE_MEMORY_ANALYZE,
              STAGE_UMMA_ANALYZE, STAGE_CAMPAIGN_VALIDATE),
          str(STAGE_PLANS["full"]))
    for scope in ("memory", "umma", "gemm"):
        plan = STAGE_PLANS[scope]
        check(f"the --only {scope} plan starts with preflight and ends with campaign.validate",
              plan[0] == STAGE_PREFLIGHT and plan[-1] == STAGE_CAMPAIGN_VALIDATE, str(plan))
    check("no plan contains a separate cublaslt or cutedsl stage",
          not any("cublaslt" in stage or "cutedsl" in stage
                  for plan in STAGE_PLANS.values() for stage in plan),
          str(STAGE_PLANS))
    check("every stage has a kind and a target entry",
          all(stage in STAGE_KIND and stage in STAGE_MAKE_TARGET for stage in ALL_STAGES), "")

    for bad in ("2099-01-01", "20990101T000000", "20991301T000000Z", "", "20990101T000000Z "):
        try:
            validate_campaign_id(bad)
            check(f"campaign id {bad!r} is rejected", False, "accepted")
        except OrchestratorError:
            check(f"campaign id {bad!r} is rejected", True)
    validate_campaign_id("20990101T000000Z")
    check("a canonical campaign id is accepted", True)

    for bad in ("", "abc", "-1", "1x", " 1", None):
        try:
            _require_gpu_index(bad)
            check(f"BLACKWELL_GPU_INDEX={bad!r} is rejected", False, "accepted")
        except OrchestratorError:
            check(f"BLACKWELL_GPU_INDEX={bad!r} is rejected", True)
    check("BLACKWELL_GPU_INDEX='3' is accepted", _require_gpu_index("3") == 3)

    for bad in ("../escape", "a/b", "", ".", "..", "x\0y"):
        try:
            validate_basename(bad)
            check(f"basename {bad!r} is rejected", False, "accepted")
        except OrchestratorError:
            check(f"basename {bad!r} is rejected", True)

    rendered = render_plan(build_plan_document("20990101T000000Z", "pilot", "full"))
    check("the rendered plan is deterministic",
          rendered == render_plan(build_plan_document("20990101T000000Z", "pilot", "full")), "")
    check("the rendered plan states publishable: false", "publishable: false" in rendered, rendered)

    good = {"git_commit": "a" * 40, "gpu_uuid": "GPU-1", "gpu_name": "D",
            "compute_capability": "10.3", "cuda_driver_version": 13010,
            "cuda_runtime_version": "13010"}
    normalized = normalize_unit_provenance(good, label="self-test")
    check("an integer and a string CUDA version normalize to the same text",
          normalized["cuda_driver_version"] == normalized["cuda_runtime_version"] == "13010",
          str(normalized))
    for field in UNIT_PROVENANCE_FIELDS:
        try:
            normalize_unit_provenance({k: v for k, v in good.items() if k != field},
                                      label="self-test")
            check(f"unit provenance without {field} is rejected", False, "accepted")
        except RunFailure:
            check(f"unit provenance without {field} is rejected", True)
    try:
        normalize_unit_provenance(dict(good, git_dirty=True), label="self-test")
        check("unit provenance reporting a dirty tree is rejected", False, "accepted")
    except RunFailure:
        check("unit provenance reporting a dirty tree is rejected", True)

    redacted = redact_repo_root("read /checkout/results/x.json failed", Path("/checkout"))
    check("the durable log-write boundary redacts only the checkout root",
          redacted == f"read {REPO_ROOT_TOKEN}/results/x.json failed", redacted)
    check("the orchestrated Make invocation is silent and prints no directory",
          MAKE_QUIET_FLAGS == ("--silent", "--no-print-directory"), str(MAKE_QUIET_FLAGS))

    check("a manifest field named username is rejected",
          bool(validate_manifest_privacy({"username": "x"})), "")
    check("an absolute path in durable metadata is rejected",
          bool(validate_manifest_privacy({"path": "/home/someone/repo"})), "")
    check("an allowlisted manifest field is accepted",
          validate_manifest_privacy({"gpu": {"uuid": "GPU-1"}}) == [], "")

    if failures:
        print(f"phase4_orchestrator: self-test: FAILED ({len(failures)} check(s))", file=sys.stderr)
        return 1
    print("phase4_orchestrator: self-test: PASS")
    return EXIT_OK


def cli_main(argv: list[str] | None = None) -> int:
    """The process-level exit-code contract: every exception that can escape
    maps to exactly one documented status, and an interruption is never
    reported as an ordinary stage failure."""
    try:
        return main(argv)
    except OrchestratorError as error:
        print(f"run_all: ERROR: {error}", file=sys.stderr)
        return EXIT_PRECONDITION
    except (RunFailure, ManifestError) as error:
        print(f"run_all: ERROR: {error}", file=sys.stderr)
        return EXIT_RUN_FAILURE
    except KeyboardInterrupt:
        print("run_all: ERROR: interrupted; completed evidence is preserved and the interrupted "
              "stage stays eligible for a new attempt on --resume", file=sys.stderr)
        return EXIT_INTERRUPTED


if __name__ == "__main__":
    raise SystemExit(cli_main())
