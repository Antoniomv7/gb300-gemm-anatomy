#!/usr/bin/env python3
"""Fail-closed GPU-free checker for P4.1 -- the Phase 4 campaign orchestrator.

Two modes:

    python3 scripts/check_phase4_orchestrator_p41.py --self-test
        Synthetic acceptance tests over temporary directories with stubbed
        child commands. No Docker, no GPU, no nvidia-smi, no CUDA/PyTorch/
        CuTe DSL import, no network, and no real raw results are required.

    python3 scripts/check_phase4_orchestrator_p41.py <repo-root>
        The frozen P4.1 contract checked against the real repository: the
        required files, the public CLI surface, the forbidden shared-cluster
        patterns, the composition of the closed P1.4/P2.4/P3.5 entry points,
        the Make targets, and the truthful status documents.

The P3.5 capture is always validated with P3.5's own frozen schema and its own
`validate_serialized_output`, never a second interpretation; the synthetic
fixtures are likewise built from P3.5's own row factory.

Exit codes: 0 OK; 1 at least one check failed; 2 usage error.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import os
import re
import shutil
import sys
import tempfile
from contextlib import redirect_stderr
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent

ORCHESTRATOR_RELATIVE_PATH = "scripts/phase4_orchestrator.py"
RUN_ALL_RELATIVE_PATH = "scripts/run_all.sh"
CHECKER_RELATIVE_PATH = "scripts/check_phase4_orchestrator_p41.py"
PROTOCOL_RELATIVE_PATH = "src/phase4/P4_1_PROTOCOL.md"
P35_CHECKER_RELATIVE_PATH = "scripts/check_gemm_comparison_p35.py"

REQUIRED_P41_FILES = (
    RUN_ALL_RELATIVE_PATH,
    ORCHESTRATOR_RELATIVE_PATH,
    CHECKER_RELATIVE_PATH,
    PROTOCOL_RELATIVE_PATH,
)
EXECUTABLE_P41_FILES = (RUN_ALL_RELATIVE_PATH, ORCHESTRATOR_RELATIVE_PATH,
                        CHECKER_RELATIVE_PATH)

# The exact stage plans, restated independently of the orchestrator so a
# silent reordering, insertion, or deletion is rejected.
EXPECTED_FULL_PLAN = (
    "preflight", "memory.pilot", "memory.profile", "umma.pilot", "umma.profile",
    "gemm.capture", "memory.analyze", "umma.analyze", "campaign.validate",
)
EXPECTED_MEMORY_PLAN = (
    "preflight", "memory.pilot", "memory.profile", "memory.analyze", "campaign.validate",
)
EXPECTED_UMMA_PLAN = (
    "preflight", "umma.pilot", "umma.profile", "umma.analyze", "campaign.validate",
)
EXPECTED_GEMM_PLAN = ("preflight", "gemm.capture", "campaign.validate")

# The exact Make targets P4.1 is allowed to drive, restated independently.
EXPECTED_MAKE_TARGETS = {
    "preflight": "preflight",
    "memory.pilot": "memory-paths-p14-pilot",
    "memory.profile": "memory-paths-p14-profile",
    "memory.analyze": "memory-paths-p14-analyze",
    "umma.pilot": "compute-umma-p24-pilot",
    "umma.profile": "compute-umma-p24-profile",
    "umma.analyze": "compute-umma-p24-analyze",
    "gemm.capture": "gemm-comparison-p35-smoke",
    "campaign.validate": "",
}

# The binding shared-cluster prohibitions from AGENTS.md, restated here so the
# new P4.1 files are scanned by this unit's own gate too.
FORBIDDEN_CLUSTER_PATTERNS = (
    r"--gpus[ =]+all",
    r"NVIDIA_VISIBLE_DEVICES=all",
    r"--privileged",
    r"--pid[ =]+host",
    r"docker\.sock",
    r"--cap-add",
    r"SYS_ADMIN",
    r"set -x",
    r"\bs" + r"udo\b",
    r"\$\(np" + r"roc\)",
    r"nvidia-smi[^|]*(-pm|--persistence-mode|-lgc|--lock-gpu-clocks|-pl|--power-limit)",
    r"--force-overwrite",
    r"--set[ =]+full",
    r"clock-control[ =]+(base|boost|force-boost)",
)

# P4.1 adds no profiler route, no GEMM/CUDA implementation, and no direct
# container invocation of its own. These are scanned as *executed* tokens --
# anchored at a shell command position or a Python call -- so that honest help
# text naming a tool in order to say it is never used stays legal while an
# actual invocation does not.
FORBIDDEN_P41_COMMANDS = ("ncu", "nvidia-smi", "docker")
_COMMAND_POSITION = r"(?:^|[;|&(]|\$\(|`)\s*{name}\b"

# Tokens that must not appear at all: P4.1 owns no kernel and no library call.
FORBIDDEN_P41_TOKENS = (
    r"__global__",
    r"cublasLt",
    r"tcgen05",
    r"cp\.async",
)

# The Phase 4 status frontier this unit is allowed to advance, and the two
# rows that must stay unimplemented.
EXPECTED_P41_STATUS_LINE = "| P4.1 | Orchestrator | YES | NO | NO |"
FORBIDDEN_P41_STATUS_LINES = (
    "| P4.1 | Orchestrator | NO | NO | NO |",
    "| P4.1 | Orchestrator | YES | YES | NO |",
    "| P4.1 | Orchestrator | YES | NO | YES |",
    "| P4.1 | Orchestrator | YES | YES | YES |",
)
UNIMPLEMENTED_PHASE4_STATUS_LINES = (
    "| P4.2 | Pilot plus three final campaigns | NO | NO | NO |",
    "| P4.3 | Integrated analysis, documentation, audit | NO | NO | NO |",
)
CLOSED_STATUS_LINES = (
    "| P1.4 | Profiling, validation, analysis, pilot | YES | YES | YES |",
    "| P2.4 | Profiling and empirical ceiling | YES | YES | YES |",
    "| P3.5 | Five shapes and comparison | YES | YES | YES |",
)

REQUIRED_PROTOCOL_STATEMENTS = (
    "P4.1 = YES / NO / NO",
    "P4.1 is implemented infrastructure",
    "Independent audit: PENDING",
    "GB300 verification: PENDING",
    "No Phase 4 campaign was executed",
    "No publishable result exists",
    "P4.2 will execute one pilot and three independent final campaigns",
    "P4.3 will perform integrated analysis and publication review",
)
FORBIDDEN_PROTOCOL_STATEMENTS = (
    "P4.1 = YES / YES / YES",
    "P4.1 has been independently audited",
    "P4.1 was verified on GB300",
    "publishable=true",
    "publishable: true",
)

# Synthetic provenance for every fixture below. Nothing here is real.
FIXTURE_COMMIT = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
FIXTURE_OTHER_COMMIT = "0123456789abcdef0123456789abcdef01234567"
FIXTURE_GPU_UUID = "GPU-11111111-2222-3333-4444-555555555555"
FIXTURE_OTHER_GPU_UUID = "GPU-99999999-8888-7777-6666-555555555555"
FIXTURE_GPU_NAME = "SYNTHETIC TEST DEVICE"
FIXTURE_GPU_CC = "10.3"
FIXTURE_GPU_DRIVER = "610.43.02"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Synthetic fixtures: a temporary repository, stubbed child commands, and
# stubbed P1.4/P2.4 semantic loaders.
# ---------------------------------------------------------------------------


class Fixture:
    """One temporary repository plus a deterministic stubbed environment.

    The orchestration logic under test is the real one: only the seams that
    would otherwise touch Docker, a GPU, nvidia-smi, or a real raw campaign are
    replaced. The P3.5 validator is never stubbed -- the real P3.5 checker's
    own `validate_serialized_output` is used throughout.
    """

    def __init__(self, orchestrator, p35, root: Path, *, commit: str = FIXTURE_COMMIT,
                 dirty: bool = False):
        self.o = orchestrator
        self.p35 = p35
        self.root = Path(root)
        (self.root / "results").mkdir(parents=True, exist_ok=True)
        self.commit = commit
        self.dirty = dirty
        self.clock = 0
        self.preflight_seq = 0
        self.calls: list[tuple[str, dict]] = []
        self.fail_targets: dict[str, int] = {}
        self.interrupt_targets: set[str] = set()
        self.gemm_text_override: str | None = None
        self.gpu = {
            "uuid": FIXTURE_GPU_UUID,
            "name": FIXTURE_GPU_NAME,
            "compute_capability": FIXTURE_GPU_CC,
            "driver_version": FIXTURE_GPU_DRIVER,
        }
        self.p14_state: str | None = None
        self.p24_state: str | None = None
        self.p24_final_state = "ANALYZED"
        self.p14_revision = -1
        self.p24_revision = -1
        self.unit_commit = commit
        self.unit_gpu_uuid = FIXTURE_GPU_UUID
        self.unit_gpu_name = FIXTURE_GPU_NAME

    # -- seams ----------------------------------------------------------

    def now_utc(self) -> str:
        self.clock += 1
        minute, second = divmod(self.clock, 60)
        hour, minute = divmod(minute, 60)
        return f"20990101T{hour:02d}{minute:02d}{second:02d}Z"

    def git_state(self) -> tuple[str, bool]:
        return self.commit, self.dirty

    def context(self, **overrides):
        kwargs = dict(
            repo_root=self.root,
            run_command=self.run_command,
            now_utc=self.now_utc,
            git_state=self.git_state,
            p14_loader=self.p14_loader,
            p24_loader=self.p24_loader,
            p35_validator=self.p35.validate_serialized_output,
            stderr=io.StringIO(),
        )
        kwargs.update(overrides)
        return self.o.Context(**kwargs)

    # -- stubbed child commands -----------------------------------------

    def run_command(self, argv, *, env, stdout_fd, stderr_fd) -> int:
        assert argv[0] == "make", argv
        target = argv[1]
        self.calls.append((target, dict(env)))
        if target in self.interrupt_targets:
            raise KeyboardInterrupt
        os.write(stderr_fd, f"stub: {target}\n".encode())
        status = self.fail_targets.get(target, 0)
        if status == 0:
            handler = {
                "preflight": self._stub_preflight,
                "memory-paths-p14-pilot": lambda fd, e: self._advance_p14("PILOT_COMPLETE"),
                "memory-paths-p14-profile": lambda fd, e: self._advance_p14("COMPLETE"),
                "memory-paths-p14-analyze": lambda fd, e: self._advance_p14("ANALYZED"),
                "compute-umma-p24-pilot": lambda fd, e: self._advance_p24("PILOT_COMPLETE"),
                "compute-umma-p24-profile": lambda fd, e: self._advance_p24("COMPLETE"),
                "compute-umma-p24-analyze": lambda fd, e: self._advance_p24(self.p24_final_state),
                "gemm-comparison-p35-smoke": self._stub_gemm,
            }.get(target)
            if handler is None:
                raise AssertionError(f"the orchestrator invoked an unexpected target {target!r}")
            handler(stdout_fd, env)
        return status

    def _stub_preflight(self, stdout_fd, env) -> None:
        index = env["BLACKWELL_GPU_INDEX"]
        self.preflight_seq += 1
        timestamp = f"20990102T{self.preflight_seq:02d}0000Z"
        directory = self.root / "results" / "preflight" / timestamp
        directory.mkdir(parents=True, exist_ok=False)
        summary = {
            "schema_version": "1",
            "timestamp_utc": timestamp,
            "git_commit": self.commit,
            "git_dirty": self.dirty,
            "host_arch": "aarch64",
            "tool_versions": {"nvcc": "release 13.1"},
            "gpu": {
                "logical_index": "0",
                "name": self.gpu["name"],
                "uuid": self.gpu["uuid"],
                "driver_version": self.gpu["driver_version"],
                "compute_cap": self.gpu["compute_capability"],
                "memory_total": "186000 MiB",
            },
            "checks": [
                {"name": name, "required": True, "status": "PASS", "reason_code": "OK"}
                for name in self.o.REQUIRED_PREFLIGHT_CHECKS
            ],
            "overall_status": "PASS",
        }
        (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n",
                                                encoding="utf-8")
        lines = [
            f"run_container: selected index={index} uuid={self.gpu['uuid']} "
            f"name='{self.gpu['name']}' driver={self.gpu['driver_version']}",
            f"run_container: GPU {self.gpu['uuid']} has no active compute processes",
            f"preflight: writing to results/preflight/{timestamp}",
            "preflight: gpu_visibility PASS (OK)",
            f"preflight: summary written to results/preflight/{timestamp}/summary.json",
            "preflight: OVERALL=PASS",
        ]
        os.write(stdout_fd, ("\n".join(lines) + "\n").encode())

    def _advance_p14(self, state: str) -> None:
        self.p14_state = state
        self.p14_revision += 1

    def _advance_p24(self, state: str) -> None:
        self.p24_state = state
        self.p24_revision += 1

    def _stub_gemm(self, stdout_fd, env) -> None:
        text = self.gemm_text_override if self.gemm_text_override is not None else self.gemm_csv()
        os.write(stdout_fd, text.encode("utf-8"))

    # -- stubbed semantic loaders ---------------------------------------

    def _unit_payload(self, state, revision, rel):
        if state is None:
            raise RuntimeError("no such campaign")
        return {
            "manifest": {
                "state": state,
                "provenance": {
                    "git_commit": self.unit_commit,
                    "gpu_uuid": self.unit_gpu_uuid,
                    "gpu_name": self.unit_gpu_name,
                    "compute_capability": FIXTURE_GPU_CC,
                },
            },
            "revision": revision,
            "campaign_dir": rel,
        }

    def p14_loader(self, campaign_id: str) -> dict:
        return self._unit_payload(self.p14_state, self.p14_revision,
                                  f"{self.o.P14_RAW_REL}/{campaign_id}")

    def p24_loader(self, campaign_id: str) -> dict:
        return self._unit_payload(self.p24_state, self.p24_revision,
                                  f"{self.o.P24_RAW_REL}/{campaign_id}")

    # -- P3.5 fixtures --------------------------------------------------

    def gemm_rows(self) -> list:
        rows = self.p35._good_rows()
        for row in rows:
            row.update({
                "git_commit": self.commit,
                "git_dirty": "false",
                "publishable": "false",
                "gpu_uuid": self.gpu["uuid"],
                "gpu_name": self.gpu["name"],
                "compute_capability": self.gpu["compute_capability"],
                "driver_version": self.gpu["driver_version"],
            })
        return rows

    def gemm_csv(self, rows=None) -> str:
        return self.p35._serialize(self.gemm_rows() if rows is None else rows)

    # -- convenience ----------------------------------------------------

    def campaign_dir(self, campaign_id: str) -> Path:
        return self.root / "results" / "raw" / "phase4" / campaign_id

    def manifest_revisions(self, campaign_id: str) -> list[str]:
        directory = self.campaign_dir(campaign_id) / "manifest"
        if not directory.is_dir():
            return []
        return sorted(p.name for p in directory.iterdir())

    def load_manifest(self, campaign_id: str) -> dict:
        manifest, _revision = self.o.load_manifest_chain(
            self.campaign_dir(campaign_id), repo_root=self.root)
        return manifest

    def run(self, *, campaign_id: str, kind: str = "pilot", selector=None,
            gpu_index: str | None = "3", resume: bool = False, context=None) -> int:
        ctx = context if context is not None else self.context()
        return self.o.run_campaign(
            ctx, campaign_id=campaign_id, campaign_kind=kind, selector=selector,
            gpu_index_text=gpu_index, resume=resume)


# ---------------------------------------------------------------------------
# The synthetic acceptance suite.
# ---------------------------------------------------------------------------


class Reporter:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.failures: list[str] = []

    def check(self, label: str, condition: bool, detail: str = "") -> bool:
        if condition:
            print(f"{self.prefix}: PASS: {label}")
            return True
        self.failures.append(f"{label}: {detail}")
        print(f"{self.prefix}: FAIL: {label}: {detail}", file=sys.stderr)
        return False

    def rejects(self, label: str, callable_, expected_exception, fragment: str = "") -> bool:
        try:
            callable_()
        except expected_exception as exc:
            if fragment and fragment not in str(exc):
                return self.check(label, False, f"raised {exc!r} without {fragment!r}")
            return self.check(label, True)
        except Exception as exc:  # noqa: BLE001 - the wrong exception is a failure
            return self.check(label, False, f"raised the wrong exception {exc!r}")
        return self.check(label, False, "did not raise")


def run_self_test(orchestrator, p35) -> int:
    reporter = Reporter("check_phase4_orchestrator_p41: self-test")
    o = orchestrator

    _test_plans(reporter, o)
    _test_cli_validation(reporter, o)
    _test_gemm_capture_validation(reporter, o, p35)
    _test_manifest_rules(reporter, o)
    _test_happy_path(reporter, o, p35)
    _test_only_selectors(reporter, o, p35)
    _test_existence_rules(reporter, o, p35)
    _test_gpu_selection(reporter, o, p35)
    _test_failure_and_downstream(reporter, o, p35)
    _test_interruption_and_resume(reporter, o, p35)
    _test_tamper_detection(reporter, o, p35)
    _test_no_clobber(reporter, o, p35)
    _test_symlink_rejection(reporter, o, p35)
    _test_mismatch_rejection(reporter, o, p35)
    _test_inconclusive_propagation(reporter, o, p35)
    _test_bad_gemm_captures(reporter, o, p35)
    _test_privacy(reporter, o, p35)

    if reporter.failures:
        print(f"check_phase4_orchestrator_p41: self-test: FAILED "
              f"({len(reporter.failures)} check(s))", file=sys.stderr)
        return 1
    print("check_phase4_orchestrator_p41: self-test: OK")
    return 0


def _fixture(o, p35, stack, **kwargs) -> Fixture:
    root = Path(tempfile.mkdtemp(prefix="p41-"))
    stack.append(root)
    return Fixture(o, p35, root, **kwargs)


class _TempStack:
    def __init__(self):
        self.paths: list[Path] = []

    def append(self, path: Path) -> None:
        self.paths.append(path)

    def cleanup(self) -> None:
        for path in self.paths:
            shutil.rmtree(path, ignore_errors=True)


_STACK = _TempStack()


def _test_plans(reporter: Reporter, o) -> None:
    reporter.check("the full plan is exactly the frozen nine-stage order",
                   o.STAGE_PLANS["full"] == EXPECTED_FULL_PLAN, str(o.STAGE_PLANS["full"]))
    reporter.check("the --only memory plan is exactly the frozen order",
                   o.STAGE_PLANS["memory"] == EXPECTED_MEMORY_PLAN, str(o.STAGE_PLANS["memory"]))
    reporter.check("the --only umma plan is exactly the frozen order",
                   o.STAGE_PLANS["umma"] == EXPECTED_UMMA_PLAN, str(o.STAGE_PLANS["umma"]))
    reporter.check("the --only gemm plan is exactly the frozen order",
                   o.STAGE_PLANS["gemm"] == EXPECTED_GEMM_PLAN, str(o.STAGE_PLANS["gemm"]))
    reporter.check("an absent --only selects the complete three-experiment plan",
                   o.scope_for_selector(None) == "full", "")
    reporter.check("every stage maps to exactly the expected Make target",
                   o.STAGE_MAKE_TARGET == EXPECTED_MAKE_TARGETS, str(o.STAGE_MAKE_TARGET))
    reporter.check("there is no cublaslt or cutedsl selector",
                   set(o.SELECTORS) == {"memory", "umma", "gemm"}, str(o.SELECTORS))
    for selector in ("cublaslt", "cutedsl", "exp03", ""):
        reporter.rejects(f"--only {selector!r} is rejected",
                         lambda s=selector: o.scope_for_selector(s), o.OrchestratorError)
    rendered = o.render_plan(o.build_plan_document("20990101T000000Z", "final", "full"))
    reporter.check("the rendered plan is byte-identical across calls",
                   rendered == o.render_plan(
                       o.build_plan_document("20990101T000000Z", "final", "full")), "")
    reporter.check("campaign-kind=final does not make the plan publishable",
                   "publishable: false" in rendered, rendered)


def _test_cli_validation(reporter: Reporter, o) -> None:
    for bad in ("2099-01-01", "20990101T000000", "20991301T000000Z", "20990101T240000Z",
                "20990230T000000Z", "", " 20990101T000000Z", "20990101t000000z", 20990101, None):
        reporter.rejects(f"campaign id {bad!r} is rejected",
                         lambda b=bad: o.validate_campaign_id(b), o.OrchestratorError)
    o.validate_campaign_id("20990101T000000Z")
    reporter.check("a canonical campaign id is accepted", True)
    for bad in ("PILOT", "Final", "smoke", "", None, 1):
        reporter.rejects(f"campaign kind {bad!r} is rejected",
                         lambda b=bad: o.validate_campaign_kind(b), o.OrchestratorError)
    for good in ("pilot", "final"):
        o.validate_campaign_kind(good)
    reporter.check("both campaign kinds are accepted", True)


def _test_gemm_capture_validation(reporter: Reporter, o, p35) -> None:
    fixture = _fixture(o, p35, _STACK)
    good = fixture.gemm_csv()

    def validate(text):
        return o.validate_gemm_capture(
            text, expected_git_commit=fixture.commit, expected_gpu_uuid=fixture.gpu["uuid"],
            expected_gpu_name=fixture.gpu["name"],
            expected_compute_capability=fixture.gpu["compute_capability"],
            expected_driver_version=fixture.gpu["driver_version"],
            p35_validator=p35.validate_serialized_output)

    summary = validate(good)
    reporter.check("a well-formed P3.5 capture yields exactly 20 rows",
                   summary["row_count"] == p35.EXPECTED_ROW_COUNT, str(summary))

    lines = good.splitlines()
    reporter.rejects("a partial capture (19 rows) is rejected",
                     lambda: validate("\n".join(lines[:-1]) + "\n"), o.RunFailure)
    reporter.rejects("an extra 21st row is rejected",
                     lambda: validate(good + lines[-1] + "\n"), o.RunFailure)
    reordered = [lines[0], *lines[2:5], lines[1], *lines[5:]]
    reporter.rejects("a reordered capture is rejected",
                     lambda: validate("\n".join(reordered) + "\n"), o.RunFailure)
    reporter.rejects("a capture contaminated with a Make diagnostic is rejected",
                     lambda: validate("make: Entering directory\n" + good), o.RunFailure)
    reporter.rejects("a capture contaminated with a launcher notice is rejected",
                     lambda: validate(good + "run_container: selected index=3\n"), o.RunFailure)
    reporter.rejects("an empty capture is rejected", lambda: validate(""), o.RunFailure)
    reporter.rejects("a header-only capture is rejected",
                     lambda: validate(lines[0] + "\n"), o.RunFailure)

    header = lines[0].split(",")
    swapped = ",".join([header[1], header[0], *header[2:]])
    reporter.rejects("a reordered 100-field header is rejected",
                     lambda: validate("\n".join([swapped, *lines[1:]]) + "\n"), o.RunFailure)
    reporter.rejects("a header missing one field is rejected",
                     lambda: validate("\n".join([",".join(header[:-1]), *lines[1:]]) + "\n"),
                     o.RunFailure)

    for label, field, value in (
        ("a wrong commit", "git_commit", FIXTURE_OTHER_COMMIT),
        ("a wrong GPU UUID", "gpu_uuid", FIXTURE_OTHER_GPU_UUID),
        ("a wrong GPU name", "gpu_name", "SOME OTHER DEVICE"),
        ("a wrong driver version", "driver_version", "1.2.3"),
        ("a wrong compute capability", "compute_capability", "9.9"),
    ):
        rows = fixture.gemm_rows()
        for row in rows:
            row[field] = value
        reporter.rejects(f"{label} in every row is rejected",
                         lambda r=rows: validate(fixture.gemm_csv(r)), o.RunFailure)

    for label, field, value in (
        ("a failed correctness flag", "correctness", "FAIL"),
        ("a publishable=true flag", "publishable", "true"),
        ("a dirty tree flag", "git_dirty", "true"),
    ):
        rows = fixture.gemm_rows()
        rows[7][field] = value
        reporter.rejects(f"{label} on one row is rejected",
                         lambda r=rows: validate(fixture.gemm_csv(r)), o.RunFailure)

    rows = fixture.gemm_rows()
    rows[4]["variant"] = "persistent_2cta"
    reporter.rejects("a duplicated candidate inside one shape is rejected",
                     lambda r=rows: validate(fixture.gemm_csv(r)), o.RunFailure)
    rows = fixture.gemm_rows()
    rows[0]["shape_id"] = "1024x1024x1024x1"
    reporter.rejects("a shape outside the frozen five is rejected",
                     lambda r=rows: validate(fixture.gemm_csv(r)), o.RunFailure)
    rows = fixture.gemm_rows()
    rows[2]["kernel_time_ms"] = "0.000000"
    reporter.rejects("a non-positive kernel time is rejected",
                     lambda r=rows: validate(fixture.gemm_csv(r)), o.RunFailure)
    rows = fixture.gemm_rows()
    rows[2]["schema_version"] = "p34.v1"
    reporter.rejects("a wrong schema version is rejected",
                     lambda r=rows: validate(fixture.gemm_csv(r)), o.RunFailure)


def _test_manifest_rules(reporter: Reporter, o) -> None:
    reporter.check("every manifest field is classified exactly once",
                   set(o.ALLOWED_MANIFEST_KEYS) == (
                       o.MANIFEST_IMMUTABLE_FIELDS | o.MANIFEST_STATE_DERIVED_FIELDS
                       | o.MANIFEST_APPEND_ONLY_FIELDS | o.MANIFEST_SET_LATEST_FIELDS
                       | o.MANIFEST_TERMINAL_FIELDS), "")
    reporter.check("COMPLETE and INCONCLUSIVE are terminal",
                   o.ALLOWED_TRANSITIONS[o.STATE_COMPLETE] == frozenset()
                   and o.ALLOWED_TRANSITIONS[o.STATE_INCONCLUSIVE] == frozenset(), "")
    reporter.check("a campaign may only start IN_PROGRESS",
                   o.ALLOWED_TRANSITIONS[None] == frozenset({o.STATE_IN_PROGRESS}), "")

    base = {
        "schema_version": o.SCHEMA_VERSION, "unit": o.UNIT,
        "campaign_id": "20990101T000000Z", "campaign_kind": "pilot", "scope": "gemm",
        "stage_order": list(EXPECTED_GEMM_PLAN), "plan_sha256": "0" * 64,
        "git_commit": FIXTURE_COMMIT, "git_dirty": False, "publishable": False,
        "state": o.STATE_IN_PROGRESS, "created_at_utc": "20990101T000000Z",
        "updated_at_utc": "20990101T000000Z", "stage_attempts": [], "stage_results": {},
    }
    reporter.check("a well-formed revision-0 manifest is accepted",
                   o.validate_manifest_document(base, expected_campaign_id="20990101T000000Z") == [],
                   str(o.validate_manifest_document(base, expected_campaign_id="20990101T000000Z")))
    for label, mutation in (
        ("publishable=true", {"publishable": True}),
        ("git_dirty=true", {"git_dirty": True}),
        ("an unknown field", {"operator_home": "x"}),
        ("a wrong schema version", {"schema_version": "p41.v2"}),
        ("a nulled field", {"state": None}),
        ("a mismatched campaign id", {"campaign_id": "20990102T000000Z"}),
        ("a reordered stage plan", {"stage_order": list(reversed(EXPECTED_GEMM_PLAN))}),
        ("an illegal state", {"state": "DONE"}),
        ("an outcome without a terminal state", {"outcome": "COMPLETE"}),
        ("a failure stage without a failed state", {"failure_stage": "preflight"}),
    ):
        document = dict(base)
        document.update(mutation)
        errors = o.validate_manifest_document(document, expected_campaign_id="20990101T000000Z")
        reporter.check(f"a manifest carrying {label} is rejected", bool(errors), "accepted")

    reporter.check("revision 0 may not already record a stage",
                   bool(o.validate_manifest_revision_transition(
                       None, {**base, "stage_results": {"preflight": {}}})), "")
    previous = dict(base)
    current = dict(base)
    current["campaign_kind"] = "final"
    reporter.check("an immutable field change is rejected",
                   bool(o.validate_manifest_revision_transition(previous, current)), "")
    current = dict(base)
    current["state"] = o.STATE_COMPLETE
    current["outcome"] = o.STATE_COMPLETE
    reporter.check("declaring COMPLETE with no completed stage is rejected",
                   bool(o.validate_manifest_state_shape(current)), "")


def _drive_full_campaign(o, p35, reporter, *, campaign_id="20990301T000000Z", **fixture_kwargs):
    fixture = _fixture(o, p35, _STACK, **fixture_kwargs)
    status = fixture.run(campaign_id=campaign_id)
    return fixture, status


def _test_happy_path(reporter: Reporter, o, p35) -> None:
    campaign_id = "20990301T000000Z"
    fixture, status = _drive_full_campaign(o, p35, reporter, campaign_id=campaign_id)
    reporter.check("a complete full campaign exits 0", status == o.EXIT_OK, str(status))

    manifest = fixture.load_manifest(campaign_id)
    reporter.check("the terminal state is COMPLETE",
                   manifest["state"] == o.STATE_COMPLETE, manifest["state"])
    reporter.check("the terminal outcome is COMPLETE",
                   manifest.get("outcome") == o.STATE_COMPLETE, str(manifest.get("outcome")))
    reporter.check("every stage of the plan is recorded complete, in plan order",
                   list(manifest["stage_results"]) == list(EXPECTED_FULL_PLAN),
                   str(list(manifest["stage_results"])))
    reporter.check("the campaign remains publishable=false",
                   manifest["publishable"] is False, "")
    reporter.check("the campaign kind is recorded immutably",
                   manifest["campaign_kind"] == "pilot", manifest["campaign_kind"])
    reporter.check("the stages executed in exactly the frozen order",
                   [target for target, _env in fixture.calls]
                   == [EXPECTED_MAKE_TARGETS[stage] for stage in EXPECTED_FULL_PLAN[:-1]],
                   str([target for target, _env in fixture.calls]))

    envs = dict((target, env) for target, env in fixture.calls)
    reporter.check("the P1.4 stages receive the top-level campaign ID",
                   envs["memory-paths-p14-pilot"]["P1_4_CAMPAIGN_ID"] == campaign_id, "")
    reporter.check("the P2.4 stages receive the top-level campaign ID",
                   envs["compute-umma-p24-pilot"]["P2_4_CAMPAIGN_ID"] == campaign_id, "")
    preflight_path = manifest["preflight_reference"]["summary_path"]
    reporter.check("the P1.4 stages receive the exact validated preflight of this invocation",
                   envs["memory-paths-p14-pilot"]["P1_4_PREFLIGHT_SUMMARY"] == preflight_path,
                   envs["memory-paths-p14-pilot"]["P1_4_PREFLIGHT_SUMMARY"])
    reporter.check("the P2.4 stages receive the exact validated preflight of this invocation",
                   envs["compute-umma-p24-profile"]["P2_4_PREFLIGHT_SUMMARY"] == preflight_path,
                   envs["compute-umma-p24-profile"]["P2_4_PREFLIGHT_SUMMARY"])
    reporter.check("the GPU-free analyzers are never given a GPU index",
                   "BLACKWELL_GPU_INDEX" not in envs["memory-paths-p14-analyze"]
                   and "BLACKWELL_GPU_INDEX" not in envs["compute-umma-p24-analyze"], "")
    reporter.check("the P3.5 capture runs in the launcher's data-stream mode",
                   envs["gemm-comparison-p35-smoke"].get("RUN_CONTAINER_STDOUT_IS_DATA") == "1", "")

    artifact = fixture.campaign_dir(campaign_id) / "exp03" / "gemm_comparison.csv"
    reporter.check("the accepted GEMM artifact exists at the frozen path",
                   artifact.is_file() and not artifact.is_symlink(), str(artifact))
    reporter.check("the accepted GEMM artifact carries exactly 21 lines",
                   len(artifact.read_text(encoding="utf-8").splitlines()) == 21, "")
    reporter.check("the plan file was written once",
                   (fixture.campaign_dir(campaign_id) / "plan.json").is_file(), "")
    logs = sorted(p.name for p in (fixture.campaign_dir(campaign_id) / "logs").iterdir())
    reporter.check("one no-clobber log set exists per stage attempt",
                   len(logs) == 2 * len(EXPECTED_FULL_PLAN), str(logs))
    reporter.check("the manifest is an append-only chain, never one mutable file",
                   fixture.manifest_revisions(campaign_id)[0] == "000000.json"
                   and len(fixture.manifest_revisions(campaign_id)) >= 2,
                   str(fixture.manifest_revisions(campaign_id)))
    reporter.check("no manifest.json single mutable file exists",
                   not (fixture.campaign_dir(campaign_id) / "manifest.json").exists(), "")

    text = json.dumps(manifest)
    for token in ("/home/", "USER", "HOSTNAME", "PASSWORD", "ssh-rsa"):
        reporter.check(f"the manifest carries no {token!r}", token not in text, "")

    # A resume of a complete campaign is a pure, read-only revalidation.
    before = fixture.manifest_revisions(campaign_id)
    calls_before = len(fixture.calls)
    status = fixture.run(campaign_id=campaign_id, resume=True)
    reporter.check("resuming a COMPLETE campaign exits 0", status == o.EXIT_OK, str(status))
    reporter.check("resuming a COMPLETE campaign appends no manifest revision",
                   fixture.manifest_revisions(campaign_id) == before, "")
    reporter.check("resuming a COMPLETE campaign runs no child command",
                   len(fixture.calls) == calls_before, "")


def _test_only_selectors(reporter: Reporter, o, p35) -> None:
    for selector, plan in (("memory", EXPECTED_MEMORY_PLAN), ("umma", EXPECTED_UMMA_PLAN),
                           ("gemm", EXPECTED_GEMM_PLAN)):
        fixture = _fixture(o, p35, _STACK)
        campaign_id = "20990401T000000Z"
        status = fixture.run(campaign_id=campaign_id, selector=selector)
        reporter.check(f"an --only {selector} campaign completes", status == o.EXIT_OK, str(status))
        manifest = fixture.load_manifest(campaign_id)
        reporter.check(f"an --only {selector} campaign records exactly its own stages",
                       list(manifest["stage_results"]) == list(plan),
                       str(list(manifest["stage_results"])))
        reporter.check(f"an --only {selector} campaign records the immutable scope",
                       manifest["scope"] == selector, manifest["scope"])
        reporter.check(f"an --only {selector} campaign invokes only its own targets",
                       [target for target, _env in fixture.calls]
                       == [EXPECTED_MAKE_TARGETS[stage] for stage in plan[:-1]],
                       str([target for target, _env in fixture.calls]))


def _test_existence_rules(reporter: Reporter, o, p35) -> None:
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20990501T000000Z"
    fixture.run(campaign_id=campaign_id)
    reporter.rejects("an existing campaign without --resume is rejected",
                     lambda: fixture.run(campaign_id=campaign_id), o.OrchestratorError,
                     "already exists")

    other = _fixture(o, p35, _STACK)
    reporter.rejects("--resume for a missing campaign is rejected",
                     lambda: other.run(campaign_id="20990502T000000Z", resume=True),
                     o.OrchestratorError, "does not exist")

    dirty = _fixture(o, p35, _STACK, dirty=True)
    reporter.rejects("a dirty working tree is rejected before anything is created",
                     lambda: dirty.run(campaign_id="20990503T000000Z"),
                     o.OrchestratorError, "dirty")
    reporter.check("a rejected dirty run created no campaign tree",
                   not (dirty.root / "results" / "raw").exists(), "")


def _test_gpu_selection(reporter: Reporter, o, p35) -> None:
    for label, index in (("a missing", None), ("an empty", ""), ("a non-numeric", "abc"),
                         ("a negative", "-1"), ("a padded", " 3")):
        fixture = _fixture(o, p35, _STACK)
        reporter.rejects(f"{label} BLACKWELL_GPU_INDEX is rejected",
                         lambda f=fixture, i=index: f.run(campaign_id="20990601T000000Z",
                                                          gpu_index=i),
                         o.OrchestratorError)
        reporter.check(f"{label} BLACKWELL_GPU_INDEX created no campaign tree and ran no child",
                       not (fixture.root / "results" / "raw").exists() and not fixture.calls, "")

    fixture = _fixture(o, p35, _STACK)
    fixture.run(campaign_id="20990602T000000Z", gpu_index="5")
    reporter.check("the requested physical index reaches the launcher unchanged",
                   fixture.calls[0][1]["BLACKWELL_GPU_INDEX"] == "5", str(fixture.calls[0][1]))

    # A preflight whose launcher banner names a different index is rejected.
    fixture = _fixture(o, p35, _STACK)
    original = fixture._stub_preflight

    def lying_preflight(stdout_fd, env):
        original(stdout_fd, {**env, "BLACKWELL_GPU_INDEX": "7"})

    fixture._stub_preflight = lying_preflight
    reporter.rejects("a preflight that ran on another physical index is rejected",
                     lambda: fixture.run(campaign_id="20990603T000000Z", gpu_index="3"),
                     o.RunFailure)


def _test_failure_and_downstream(reporter: Reporter, o, p35) -> None:
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20990701T000000Z"
    fixture.fail_targets["compute-umma-p24-pilot"] = 7
    reporter.rejects("a failing child stops the campaign",
                     lambda: fixture.run(campaign_id=campaign_id), o.RunFailure,
                     "exit status 7")
    executed = [target for target, _env in fixture.calls]
    reporter.check("no downstream stage runs after a child failure",
                   executed == ["preflight", "memory-paths-p14-pilot",
                                "memory-paths-p14-profile", "compute-umma-p24-pilot"],
                   str(executed))
    manifest = fixture.load_manifest(campaign_id)
    reporter.check("the campaign is recorded FAILED",
                   manifest["state"] == o.STATE_FAILED, manifest["state"])
    reporter.check("the failure stage is recorded",
                   manifest["failure_stage"] == "umma.pilot", str(manifest.get("failure_stage")))
    failed = [entry for entry in manifest["stage_attempts"] if entry["outcome"] == "FAILED"]
    reporter.check("the child's exact exit status is preserved in the manifest",
                   len(failed) == 1 and failed[0]["exit_status"] == 7, str(failed))
    reporter.check("every stage completed before the failure is preserved",
                   list(manifest["stage_results"]) == ["preflight", "memory.pilot",
                                                       "memory.profile"],
                   str(list(manifest["stage_results"])))
    reporter.rejects("resuming a campaign that failed inside a P2.4 stage is refused",
                     lambda: fixture.run(campaign_id=campaign_id, resume=True),
                     o.OrchestratorError, "new --campaign-id")

    # A stateless stage (gemm.capture) may be retried inside the same campaign.
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20990702T000000Z"
    fixture.fail_targets["gemm-comparison-p35-smoke"] = 3
    reporter.rejects("a failing P3.5 capture stops the campaign",
                     lambda: fixture.run(campaign_id=campaign_id), o.RunFailure, "exit status 3")
    artifact = fixture.campaign_dir(campaign_id) / "exp03" / "gemm_comparison.csv"
    reporter.check("a failed capture publishes no accepted GEMM artifact",
                   not artifact.exists(), "")
    fixture.fail_targets.pop("gemm-comparison-p35-smoke")
    status = fixture.run(campaign_id=campaign_id, resume=True)
    reporter.check("a retried stateless stage lets the campaign complete",
                   status == o.EXIT_OK, str(status))
    manifest = fixture.load_manifest(campaign_id)
    attempts = [e for e in manifest["stage_attempts"] if e["stage"] == "gemm.capture"]
    reporter.check("both capture attempts are recorded explicitly",
                   [e["attempt"] for e in attempts] == [1, 2], str(attempts))
    logs = sorted(p.name for p in (fixture.campaign_dir(campaign_id) / "logs").iterdir())
    reporter.check("the first attempt's log was never overwritten",
                   "gemm_capture.001.stdout.log" in logs and "gemm_capture.002.stdout.log" in logs,
                   str(logs))
    reporter.check("a fresh preflight was created for the resumed GPU work",
                   len([e for e in manifest["stage_attempts"] if e["stage"] == "preflight"]) == 2,
                   str(manifest["stage_attempts"]))
    reporter.check("both validated preflights are recorded append-only",
                   len(manifest["stage_results"]["preflight"]["evidence"]["preflights"]) == 2, "")


def _test_interruption_and_resume(reporter: Reporter, o, p35) -> None:
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20990801T000000Z"
    fixture.interrupt_targets.add("gemm-comparison-p35-smoke")
    try:
        fixture.run(campaign_id=campaign_id)
        reporter.check("an interrupted campaign propagates KeyboardInterrupt", False, "no raise")
    except KeyboardInterrupt:
        reporter.check("an interrupted campaign propagates KeyboardInterrupt", True)
    manifest = fixture.load_manifest(campaign_id)
    reporter.check("an interrupted campaign is recorded INTERRUPTED",
                   manifest["state"] == o.STATE_INTERRUPTED, manifest["state"])
    reporter.check("an interruption preserves every completed stage",
                   list(manifest["stage_results"]) == ["preflight", "memory.pilot",
                                                       "memory.profile", "umma.pilot",
                                                       "umma.profile"],
                   str(list(manifest["stage_results"])))
    reporter.check("the interrupted attempt is recorded explicitly",
                   any(e["outcome"] == "INTERRUPTED" for e in manifest["stage_attempts"]), "")

    fixture.interrupt_targets.clear()
    calls_before = len(fixture.calls)
    status = fixture.run(campaign_id=campaign_id, resume=True)
    reporter.check("resuming an interrupted campaign completes it",
                   status == o.EXIT_OK, str(status))
    resumed = [target for target, _env in fixture.calls[calls_before:]]
    reporter.check("resume re-runs only the fresh preflight and the pending stages",
                   resumed == ["preflight", "gemm-comparison-p35-smoke",
                               "memory-paths-p14-analyze", "compute-umma-p24-analyze"],
                   str(resumed))
    manifest = fixture.load_manifest(campaign_id)
    reporter.check("the resumed campaign reaches COMPLETE",
                   manifest["state"] == o.STATE_COMPLETE, manifest["state"])


def _test_tamper_detection(reporter: Reporter, o, p35) -> None:
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20990901T000000Z"
    fixture.interrupt_targets.add("memory-paths-p14-analyze")
    try:
        fixture.run(campaign_id=campaign_id)
    except KeyboardInterrupt:
        pass
    fixture.interrupt_targets.clear()
    artifact = fixture.campaign_dir(campaign_id) / "exp03" / "gemm_comparison.csv"
    original = artifact.read_text(encoding="utf-8")
    artifact.write_text(original.replace("PASS", "FAIL", 1), encoding="utf-8")
    reporter.rejects("a tampered accepted artifact aborts the resume",
                     lambda: fixture.run(campaign_id=campaign_id, resume=True), o.RunFailure,
                     "changed since it was accepted")
    reporter.check("the tampered artifact was not regenerated automatically",
                   artifact.read_text(encoding="utf-8") != original, "")

    # A tampered manifest revision is rejected by the hash chain itself.
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20990902T000000Z"
    fixture.run(campaign_id=campaign_id)
    revision = fixture.campaign_dir(campaign_id) / "manifest" / "000001.json"
    document = json.loads(revision.read_text(encoding="utf-8"))
    document["campaign_kind"] = "final"
    revision.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    reporter.rejects("a tampered manifest revision breaks the hash chain",
                     lambda: fixture.load_manifest(campaign_id), o.ManifestError)

    # A tampered preflight summary is rejected on resume.
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20990903T000000Z"
    fixture.interrupt_targets.add("gemm-comparison-p35-smoke")
    try:
        fixture.run(campaign_id=campaign_id)
    except KeyboardInterrupt:
        pass
    fixture.interrupt_targets.clear()
    manifest = fixture.load_manifest(campaign_id)
    summary = fixture.root / manifest["preflight_reference"]["summary_path"]
    summary.write_text(summary.read_text(encoding="utf-8") + " ", encoding="utf-8")
    reporter.rejects("a tampered preflight summary aborts the resume",
                     lambda: fixture.run(campaign_id=campaign_id, resume=True), o.RunFailure,
                     "changed since it was accepted")

    # An underlying unit campaign that regressed is rejected on resume.
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20990904T000000Z"
    fixture.interrupt_targets.add("gemm-comparison-p35-smoke")
    try:
        fixture.run(campaign_id=campaign_id)
    except KeyboardInterrupt:
        pass
    fixture.interrupt_targets.clear()
    fixture.p14_state = "FAILED"
    reporter.rejects("an underlying P1.4 campaign that changed state aborts the resume",
                     lambda: fixture.run(campaign_id=campaign_id, resume=True), o.RunFailure,
                     "but this campaign accepted")


def _test_no_clobber(reporter: Reporter, o, p35) -> None:
    fixture = _fixture(o, p35, _STACK)
    directory = fixture.root / "no-clobber"
    directory.mkdir()
    fd = o.open_dir_nofollow(str(directory))
    try:
        o.write_file_exclusive("evidence.log", b"first", dir_fd=fd)
        reporter.rejects("an exclusive create never overwrites an existing name",
                         lambda: o.write_file_exclusive("evidence.log", b"second", dir_fd=fd),
                         o.OrchestratorError, "already exists")
        reporter.check("the original bytes survive the refused overwrite",
                       (directory / "evidence.log").read_bytes() == b"first", "")
        o.write_file_exclusive("published.csv", b"payload", dir_fd=fd)
        reporter.rejects("no-clobber publication refuses an occupied destination",
                         lambda: o.link_no_clobber("evidence.log", "published.csv",
                                                   src_dir_fd=fd, dst_dir_fd=fd),
                         o.OrchestratorError, "already exists")
        for bad in ("../escape.log", "a/b.log", "", ".", ".."):
            reporter.rejects(f"the capture layer rejects the file name {bad!r}",
                             lambda b=bad: o.write_file_exclusive(b, b"x", dir_fd=fd),
                             o.OrchestratorError)
    finally:
        os.close(fd)

    # An occupied next-attempt log name fails the whole stage closed.
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20991001T000000Z"
    fixture.fail_targets["gemm-comparison-p35-smoke"] = 4
    try:
        fixture.run(campaign_id=campaign_id)
    except o.RunFailure:
        pass
    fixture.fail_targets.pop("gemm-comparison-p35-smoke")
    squatter = fixture.campaign_dir(campaign_id) / "logs" / "preflight.002.stdout.log"
    squatter.write_text("pre-existing evidence", encoding="utf-8")
    reporter.rejects("an occupied next-attempt log name fails the stage closed",
                     lambda: fixture.run(campaign_id=campaign_id, resume=True),
                     o.OrchestratorError, "already exists")
    reporter.check("the squatting log file was never overwritten",
                   squatter.read_text(encoding="utf-8") == "pre-existing evidence", "")


def _test_symlink_rejection(reporter: Reporter, o, p35) -> None:
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20991101T000000Z"
    raw = fixture.root / "results" / "raw" / "phase4"
    raw.mkdir(parents=True)
    elsewhere = fixture.root / "elsewhere"
    elsewhere.mkdir()
    (raw / campaign_id).symlink_to(elsewhere, target_is_directory=True)
    reporter.rejects("a symlinked campaign directory is refused for a new campaign",
                     lambda: fixture.run(campaign_id=campaign_id), o.OrchestratorError,
                     "already exists")
    reporter.rejects("a symlinked campaign directory is refused on resume",
                     lambda: fixture.run(campaign_id=campaign_id, resume=True),
                     o.OrchestratorError, "symlink")

    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20991102T000000Z"
    dangling = fixture.root / "results" / "raw" / "phase4"
    dangling.mkdir(parents=True)
    (dangling / campaign_id).symlink_to(fixture.root / "nowhere")
    reporter.rejects("a dangling symlink still counts as an existing campaign",
                     lambda: fixture.run(campaign_id=campaign_id), o.OrchestratorError,
                     "already exists")

    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20991103T000000Z"
    fixture.interrupt_targets.add("memory-paths-p14-analyze")
    try:
        fixture.run(campaign_id=campaign_id)
    except KeyboardInterrupt:
        pass
    fixture.interrupt_targets.clear()
    artifact = fixture.campaign_dir(campaign_id) / "exp03" / "gemm_comparison.csv"
    payload = fixture.root / "outside.csv"
    payload.write_text(artifact.read_text(encoding="utf-8"), encoding="utf-8")
    artifact.unlink()
    artifact.symlink_to(payload)
    reporter.rejects("a symlinked accepted artifact is refused on resume",
                     lambda: fixture.run(campaign_id=campaign_id, resume=True),
                     o.OrchestratorError, "symlink")


def _test_mismatch_rejection(reporter: Reporter, o, p35) -> None:
    base_id = "20991201T000000Z"

    def interrupted_fixture(campaign_id=base_id, **kwargs):
        fixture = _fixture(o, p35, _STACK, **kwargs)
        fixture.interrupt_targets.add("gemm-comparison-p35-smoke")
        try:
            fixture.run(campaign_id=campaign_id)
        except KeyboardInterrupt:
            pass
        fixture.interrupt_targets.clear()
        return fixture

    fixture = interrupted_fixture()
    reporter.rejects("a mismatched campaign kind aborts the resume",
                     lambda: fixture.run(campaign_id=base_id, kind="final", resume=True),
                     o.OrchestratorError, "campaign_kind")

    fixture = interrupted_fixture()
    reporter.rejects("a mismatched scope aborts the resume",
                     lambda: fixture.run(campaign_id=base_id, selector="memory", resume=True),
                     o.OrchestratorError, "scope")

    fixture = interrupted_fixture()
    fixture.commit = FIXTURE_OTHER_COMMIT
    reporter.rejects("a changed Git commit aborts the resume",
                     lambda: fixture.run(campaign_id=base_id, resume=True),
                     o.OrchestratorError, "git_commit")

    fixture = interrupted_fixture()
    fixture.gpu = dict(fixture.gpu, uuid=FIXTURE_OTHER_GPU_UUID)
    reporter.rejects("a different GPU on the fresh preflight aborts the resume",
                     lambda: fixture.run(campaign_id=base_id, resume=True), o.RunFailure,
                     "refusing to mix devices")

    fixture = interrupted_fixture()
    plan = fixture.campaign_dir(base_id) / "plan.json"
    plan.write_text(plan.read_text(encoding="utf-8").replace("pilot", "final"), encoding="utf-8")
    reporter.rejects("a tampered plan.json aborts the resume",
                     lambda: fixture.run(campaign_id=base_id, resume=True), o.OrchestratorError,
                     "plan.json")

    # An underlying unit that reports a foreign commit is refused up front.
    fixture = _fixture(o, p35, _STACK)
    fixture.unit_commit = FIXTURE_OTHER_COMMIT
    reporter.rejects("an underlying unit reporting a foreign commit fails the stage",
                     lambda: fixture.run(campaign_id="20991202T000000Z"), o.RunFailure,
                     "the campaign commit")

    fixture = _fixture(o, p35, _STACK)
    fixture.unit_gpu_uuid = FIXTURE_OTHER_GPU_UUID
    reporter.rejects("an underlying unit reporting a foreign GPU fails the stage",
                     lambda: fixture.run(campaign_id="20991203T000000Z"), o.RunFailure,
                     "validated preflight")


def _test_inconclusive_propagation(reporter: Reporter, o, p35) -> None:
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20991210T000000Z"
    fixture.p24_final_state = "INCONCLUSIVE"
    status = fixture.run(campaign_id=campaign_id)
    reporter.check("a P2.4 INCONCLUSIVE result never yields a complete campaign",
                   status == o.EXIT_INCONCLUSIVE, str(status))
    manifest = fixture.load_manifest(campaign_id)
    reporter.check("the top-level state is INCONCLUSIVE",
                   manifest["state"] == o.STATE_INCONCLUSIVE, manifest["state"])
    reporter.check("the top-level outcome is INCONCLUSIVE",
                   manifest.get("outcome") == o.STATE_INCONCLUSIVE, str(manifest.get("outcome")))
    reporter.check("an INCONCLUSIVE campaign is still publishable=false",
                   manifest["publishable"] is False, "")
    status = fixture.run(campaign_id=campaign_id, resume=True)
    reporter.check("resuming an INCONCLUSIVE campaign revalidates and stays INCONCLUSIVE",
                   status == o.EXIT_INCONCLUSIVE, str(status))

    fixture = _fixture(o, p35, _STACK)
    fixture.p24_final_state = "FAILED"
    reporter.rejects("a P2.4 campaign that never reached a terminal analysis is rejected",
                     lambda: fixture.run(campaign_id="20991211T000000Z"), o.RunFailure,
                     "not 'ANALYZED'")


def _test_bad_gemm_captures(reporter: Reporter, o, p35) -> None:
    for label, mutate in (
        ("a truncated capture", lambda text: "\n".join(text.splitlines()[:5]) + "\n"),
        ("a contaminated capture", lambda text: "nvcc warning\n" + text),
        ("an empty capture", lambda text: ""),
    ):
        fixture = _fixture(o, p35, _STACK)
        campaign_id = "20991220T000000Z"
        fixture.gemm_text_override = mutate(fixture.gemm_csv())
        reporter.rejects(f"{label} fails the gemm stage closed",
                         lambda f=fixture, c=campaign_id: f.run(campaign_id=c), o.RunFailure)
        artifact = fixture.campaign_dir(campaign_id) / "exp03" / "gemm_comparison.csv"
        reporter.check(f"{label} publishes no accepted Phase 4 GEMM artifact",
                       not artifact.exists(), "")
        manifest = fixture.load_manifest(campaign_id)
        reporter.check(f"{label} leaves the campaign FAILED at gemm.capture",
                       manifest["state"] == o.STATE_FAILED
                       and manifest["failure_stage"] == "gemm.capture",
                       str(manifest.get("failure_stage")))
        reporter.check(f"{label} prevents any downstream stage from completing",
                       "memory.analyze" not in manifest["stage_results"], "")


def _test_privacy(reporter: Reporter, o, p35) -> None:
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20991230T000000Z"
    fixture.run(campaign_id=campaign_id)
    document = json.loads(
        (fixture.campaign_dir(campaign_id) / "manifest" / "000000.json").read_text(
            encoding="utf-8"))
    reporter.check("a manifest revision carries its own index and previous hash",
                   document["manifest_revision"] == 0
                   and document["previous_manifest_sha256"] is None, str(document.keys()))
    for forbidden in ({"username": "x"}, {"home_path": "/root"}, {"gpu_power": 1},
                      {"sm_clock": 1}, {"process_pid": 1}, {"env_dump": {}}):
        reporter.check(f"the privacy gate rejects {sorted(forbidden)[0]!r}",
                       bool(o.validate_manifest_privacy(forbidden)), "")
    reporter.check("the privacy gate rejects an absolute path value",
                   bool(o.validate_manifest_privacy({"summary_path": "/home/x/summary.json"})), "")
    manifest = fixture.load_manifest(campaign_id)
    reporter.check("every recorded evidence path is repository-relative",
                   all(not str(value).startswith("/")
                       for record in manifest["stage_results"].values()
                       for value in record["evidence"].values()
                       if isinstance(value, str)), "")


# ---------------------------------------------------------------------------
# The real-repository contract check.
# ---------------------------------------------------------------------------


def read_text(repo_root: Path, relative: str) -> str:
    return (repo_root / relative).read_text(encoding="utf-8")


def check_repository(repo_root: Path) -> int:
    reporter = Reporter("check_phase4_orchestrator_p41")

    for relative in REQUIRED_P41_FILES:
        path = repo_root / relative
        reporter.check(f"{relative} exists as a regular, non-symlink file",
                       path.is_file() and not path.is_symlink(), str(path))
    for relative in EXECUTABLE_P41_FILES:
        path = repo_root / relative
        reporter.check(f"{relative} is executable", os.access(path, os.X_OK), str(path))
    if reporter.failures:
        print("check_phase4_orchestrator_p41: FAILED (missing required files)", file=sys.stderr)
        return 1

    run_all = read_text(repo_root, RUN_ALL_RELATIVE_PATH)
    orchestrator_source = read_text(repo_root, ORCHESTRATOR_RELATIVE_PATH)
    checker_source = read_text(repo_root, CHECKER_RELATIVE_PATH)
    protocol = read_text(repo_root, PROTOCOL_RELATIVE_PATH)
    makefile = read_text(repo_root, "Makefile")
    plan_text = read_text(repo_root, "PLAN.md")
    readme = read_text(repo_root, "README.md")
    results_readme = read_text(repo_root, "results/README.md")

    _check_forbidden_patterns(reporter, run_all, orchestrator_source, checker_source)
    _check_public_surface(reporter, run_all, orchestrator_source)
    _check_composition(reporter, repo_root, orchestrator_source)
    _check_makefile(reporter, makefile)
    _check_status_documents(reporter, plan_text, protocol, readme, results_readme, makefile)

    if reporter.failures:
        print(f"check_phase4_orchestrator_p41: FAILED ({len(reporter.failures)} check(s))",
              file=sys.stderr)
        return 1
    print("check_phase4_orchestrator_p41: OK")
    return 0


def _strip_python_comments(source: str) -> str:
    """Drop '#' comments and triple-quoted strings so a docstring naming a
    forbidden token in order to ban it does not trip the scan."""
    import tokenize

    output = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            output.append(token.string)
    except (tokenize.TokenError, IndentationError):
        return source
    return "\n".join(output)


def _strip_shell_comments(source: str) -> str:
    lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        lines.append(line)
    return "\n".join(lines)


def _check_forbidden_patterns(reporter: Reporter, run_all: str, orchestrator: str,
                              checker: str) -> None:
    scans = (
        (RUN_ALL_RELATIVE_PATH, _strip_shell_comments(run_all)),
        (ORCHESTRATOR_RELATIVE_PATH, _strip_python_comments(orchestrator)),
        (CHECKER_RELATIVE_PATH, _strip_python_comments(checker)),
    )
    for label, source in scans:
        for pattern in FORBIDDEN_CLUSTER_PATTERNS:
            hits = re.findall(pattern, source)
            reporter.check(f"{label} carries no forbidden cluster pattern {pattern!r}",
                           not hits, str(hits[:3]))
        for pattern in FORBIDDEN_P41_TOKENS:
            hits = re.findall(pattern, source)
            reporter.check(f"{label} carries no {pattern!r} token",
                           not hits, str(hits[:3]))
        for name in FORBIDDEN_P41_COMMANDS:
            pattern = _COMMAND_POSITION.format(name=re.escape(name))
            hits = re.findall(pattern, source, re.MULTILINE)
            reporter.check(f"{label} never invokes {name} itself", not hits, str(hits[:3]))
    stripped_orchestrator = _strip_python_comments(orchestrator)
    for name in FORBIDDEN_P41_COMMANDS:
        reporter.check(f"the orchestrator never names {name} in executable code",
                       f'"{name}"' not in stripped_orchestrator
                       and f"'{name}'" not in stripped_orchestrator, "")


def _check_public_surface(reporter: Reporter, run_all: str, orchestrator: str) -> None:
    for fragment in ("--help", "--self-test", "--dry-run", "--campaign-id", "--campaign-kind",
                     "--only", "--resume"):
        reporter.check(f"run_all.sh implements {fragment}", fragment in run_all, "")
    reporter.check("run_all.sh rejects the forbidden per-library selectors explicitly",
                   "cublaslt|cutedsl)" in run_all, "")
    reporter.check("run_all.sh states that the supported selector is --only gemm",
                   "--only gemm" in run_all, "")
    reporter.check("run_all.sh requires an explicit BLACKWELL_GPU_INDEX",
                   "BLACKWELL_GPU_INDEX" in run_all
                   and "never selects a GPU automatically" in run_all, "")
    reporter.check("run_all.sh delegates to the private helper, never to a second entry point",
                   run_all.count("phase4_orchestrator.py") >= 2, "")
    reporter.check("the orchestrator declares publishable false, never true",
                   "PUBLISHABLE = False" in orchestrator
                   and "PUBLISHABLE = True" not in orchestrator, "")
    reporter.check("the orchestrator never uses os.replace for evidence",
                   "os.replace" not in _strip_python_comments(orchestrator), "")
    reporter.check("the orchestrator never discovers a preflight by listing or mtime",
                   not re.search(r"getmtime|st_mtime|glob\(|iglob|ls -t",
                                 _strip_python_comments(orchestrator)), "")
    reporter.check("the orchestrator never generates a campaign id implicitly",
                   "strftime" in orchestrator
                   and "_default_now_utc" in orchestrator
                   and "campaign_id=_default_now_utc" not in orchestrator, "")


def _check_composition(reporter: Reporter, repo_root: Path, orchestrator: str) -> None:
    module = load_module(repo_root / ORCHESTRATOR_RELATIVE_PATH, "_p41_orchestrator_repo")
    reporter.check("the full plan is the frozen nine-stage order",
                   module.STAGE_PLANS["full"] == EXPECTED_FULL_PLAN,
                   str(module.STAGE_PLANS["full"]))
    reporter.check("the --only plans are the frozen orders",
                   module.STAGE_PLANS["memory"] == EXPECTED_MEMORY_PLAN
                   and module.STAGE_PLANS["umma"] == EXPECTED_UMMA_PLAN
                   and module.STAGE_PLANS["gemm"] == EXPECTED_GEMM_PLAN, "")
    reporter.check("every stage delegates to exactly the expected closed Make target",
                   module.STAGE_MAKE_TARGET == EXPECTED_MAKE_TARGETS,
                   str(module.STAGE_MAKE_TARGET))
    reporter.check("the orchestrator reuses P1.4's own semantic manifest loader",
                   "load_p14_manifest_chain" in orchestrator, "")
    reporter.check("the orchestrator reuses P2.4's own semantic manifest loader",
                   "load_p24_manifest_chain" in orchestrator, "")
    reporter.check("the orchestrator reuses P3.5's own serialized-output validator",
                   "validate_serialized_output" in orchestrator, "")
    reporter.check("the orchestrator adds no P3.5 schema of its own",
                   "p35.v1" not in orchestrator and "EXPECTED_CSV_FIELDS" not in orchestrator, "")
    reporter.check("the orchestrator adds no scientific threshold",
                   not re.search(r"threshold|bootstrap|confidence|roofline|speedup|tflops",
                                 _strip_python_comments(orchestrator), re.IGNORECASE), "")

    p35 = load_module(repo_root / P35_CHECKER_RELATIVE_PATH, "_p35_checker_for_p41")
    reporter.check("the P3.5 row count P4.1 requires is P3.5's own",
                   p35.EXPECTED_ROW_COUNT == 20 and p35.EXPECTED_LINE_COUNT == 21,
                   str(p35.EXPECTED_ROW_COUNT))
    reporter.check("the four frozen candidates are unchanged",
                   p35.EXPECTED_CANDIDATE_ORDER == (
                       "nonpersistent_1cta", "persistent_1cta", "persistent_2cta",
                       "heuristic_first_supported"),
                   str(p35.EXPECTED_CANDIDATE_ORDER))
    reporter.check("the five frozen shapes are unchanged",
                   len(p35.EXPECTED_SHAPES) == 5, str(p35.EXPECTED_SHAPES))


def _check_makefile(reporter: Reporter, makefile: str) -> None:
    reporter.check("the Makefile declares phase4-p41-plan",
                   re.search(r"^phase4-p41-plan:", makefile, re.MULTILINE) is not None, "")
    reporter.check("the Makefile declares phase4-p41-check with the closed GPU-free gates",
                   re.search(
                       r"^phase4-p41-check: memory-paths-p14-check compute-umma-p24-check$",
                       makefile, re.MULTILINE) is not None, "")
    reporter.check("both P4.1 targets are declared .PHONY",
                   "phase4-p41-plan phase4-p41-check" in makefile, "")
    reporter.check("phase4-p41-check runs the real P4.1 checker",
                   "check_phase4_orchestrator_p41.py" in makefile, "")
    reporter.check("the Makefile resolves the public entry point to scripts/run_all.sh",
                   "PHASE4_P41_ENTRYPOINT := scripts/run_all.sh" in makefile, "")
    reporter.check("phase4-p41-check runs the public entry point's own self-test",
                   "$(PHASE4_P41_ENTRYPOINT) --self-test" in makefile, "")
    reporter.check("phase4-p41-check exercises the real --dry-run path",
                   "$(PHASE4_P41_ENTRYPOINT) --dry-run" in makefile, "")
    reporter.check("the Makefile adds no Phase 4 GPU target",
                   not re.search(r"^phase4-p41-(pilot|final|campaign|smoke|run):",
                                 makefile, re.MULTILINE), "")
    for target in ("memory-paths-p14-pilot", "memory-paths-p14-profile",
                   "memory-paths-p14-analyze", "compute-umma-p24-pilot",
                   "compute-umma-p24-profile", "compute-umma-p24-analyze",
                   "gemm-comparison-p35-smoke"):
        reporter.check(f"the closed target {target} still exists",
                       re.search(rf"^{re.escape(target)}:", makefile, re.MULTILINE) is not None, "")


def _check_status_documents(reporter: Reporter, plan_text: str, protocol: str, readme: str,
                            results_readme: str, makefile: str) -> None:
    reporter.check("PLAN.md records P4.1 as implemented only",
                   EXPECTED_P41_STATUS_LINE in plan_text, "")
    for wrong in FORBIDDEN_P41_STATUS_LINES:
        reporter.check(f"PLAN.md does not record the untrue P4.1 status {wrong!r}",
                       wrong not in plan_text, "")
    for line in UNIMPLEMENTED_PHASE4_STATUS_LINES:
        reporter.check(f"PLAN.md still records the unimplemented {line!r}",
                       line in plan_text, "")
    for line in CLOSED_STATUS_LINES:
        reporter.check(f"PLAN.md still records the closed {line!r}", line in plan_text, "")
    reporter.check("the Makefile's frontier assertion advanced to the truthful P4.1 row",
                   "P4.1 | Orchestrator | YES | NO | NO |" in makefile, "")
    reporter.check("the Makefile still requires P4.2 to be unimplemented",
                   "P4.2 | Pilot plus three final campaigns | NO | NO | NO |" in makefile, "")
    reporter.check("the Makefile still requires P4.3 to be unimplemented",
                   "P4.3 | Integrated analysis, documentation, audit | NO | NO | NO |" in makefile,
                   "")

    for statement in REQUIRED_PROTOCOL_STATEMENTS:
        reporter.check(f"{PROTOCOL_RELATIVE_PATH} states {statement!r}",
                       statement in protocol, "")
    for statement in FORBIDDEN_PROTOCOL_STATEMENTS:
        reporter.check(f"{PROTOCOL_RELATIVE_PATH} does not claim {statement!r}",
                       statement not in protocol, "")
    reporter.check(f"{PROTOCOL_RELATIVE_PATH} does not present author checks as an audit",
                   "not an independent audit" in protocol, "")

    reporter.check("README.md describes P4.1",
                   "P4.1 (Phase 4 campaign orchestrator)" in readme, "")
    reporter.check("README.md records P4.1 as pending audit and GB300 verification",
                   "P4.1: implemented; independent audit: PENDING; GB300 verification: PENDING"
                   in readme, "")
    reporter.check("README.md does not claim a Phase 4 result",
                   "Phase 4: CLOSED" not in readme, "")
    reporter.check("results/README.md documents the Phase 4 orchestration tree",
                   "results/raw/phase4/" in results_readme, "")
    reporter.check("results/README.md states no Phase 4 campaign has been executed",
                   "No Phase 4 campaign has been executed" in results_readme, "")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_phase4_orchestrator_p41.py",
        description="Fail-closed GPU-free checker for the P4.1 Phase 4 campaign orchestrator.",
    )
    parser.add_argument("repo_root", nargs="?", default=None,
                        help="repository root to check")
    parser.add_argument("--self-test", action="store_true",
                        help="run the synthetic acceptance suite and exit")
    args = parser.parse_args(argv)

    if args.self_test:
        if args.repo_root is not None:
            print("check_phase4_orchestrator_p41: --self-test is standalone", file=sys.stderr)
            return 2
        orchestrator = load_module(SCRIPT_DIR / "phase4_orchestrator.py", "_p41_orchestrator")
        p35 = load_module(SCRIPT_DIR / "check_gemm_comparison_p35.py", "_p35_checker")
        try:
            return run_self_test(orchestrator, p35)
        finally:
            _STACK.cleanup()

    if args.repo_root is None:
        print("check_phase4_orchestrator_p41: a repository root or --self-test is required",
              file=sys.stderr)
        return 2
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.is_dir():
        print(f"check_phase4_orchestrator_p41: {repo_root} is not a directory", file=sys.stderr)
        return 2
    return check_repository(repo_root)


if __name__ == "__main__":
    with redirect_stderr(sys.stderr):
        raise SystemExit(main())
