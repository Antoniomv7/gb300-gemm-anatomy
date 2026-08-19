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
import hashlib
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

# `make phase4-p41-check` is documented as runnable with no container runtime,
# no nvidia-smi, no CUDA compilation, no GPU, and no network. These are the
# commands that would break that promise; the check walks the target's real
# transitive Make prerequisites, so a Docker-backed gate cannot be reintroduced
# into the P4.1 check path by a dependency edge either.
FORBIDDEN_CHECK_PATH_COMMANDS = ("docker", "podman", "nvidia-smi", "ncu", "nvcc", "cuobjdump",
                                 "nsys")

# Tokens that must not appear at all: P4.1 owns no kernel and no library call.
FORBIDDEN_P41_TOKENS = (
    r"__global__",
    r"cublasLt",
    r"tcgen05",
    r"cp\.async",
)

# P4.1 and P4.2 are closed after independent audit and GB300 verification. The
# P4.2-owned row advanced without changing P4.1's execution path, and every
# stale or impossible P4.2 state stays rejected. P4.3 must still be recorded
# unimplemented. See
# src/phase4/P4_2_PROTOCOL.md section 9.1.
EXPECTED_P41_STATUS_LINE = "| P4.1 | Orchestrator | YES | YES | YES |"
FORBIDDEN_P41_STATUS_LINES = (
    "| P4.1 | Orchestrator | NO | NO | NO |",
    "| P4.1 | Orchestrator | YES | YES | NO |",
    "| P4.1 | Orchestrator | YES | NO | YES |",
    "| P4.1 | Orchestrator | YES | NO | NO |",
)
EXPECTED_P42_STATUS_LINE = "| P4.2 | Pilot plus three final campaigns | YES | YES | YES |"
FORBIDDEN_P42_STATUS_LINES = (
    "| P4.2 | Pilot plus three final campaigns | NO | NO | NO |",
    "| P4.2 | Pilot plus three final campaigns | NO | YES | NO |",
    "| P4.2 | Pilot plus three final campaigns | NO | NO | YES |",
    "| P4.2 | Pilot plus three final campaigns | NO | YES | YES |",
    "| P4.2 | Pilot plus three final campaigns | YES | NO | NO |",
    "| P4.2 | Pilot plus three final campaigns | YES | YES | NO |",
    "| P4.2 | Pilot plus three final campaigns | YES | NO | YES |",
)
# P4.3 (the offline integrated analysis) is implemented, independently audited,
# and verified against the real GB300 evidence, and its curated bundle is
# accepted by the external attestation. The row it owns advanced by exactly one
# step at implementation and by exactly one further step at acceptance; every
# other P4.3 state stays rejected, including the stale "NO | NO | NO" this tuple
# once required, which would have structurally forbidden P4.3 from existing at
# all, and the "YES | NO | NO" it required until acceptance.
# See src/phase4/P4_3_PROTOCOL.md sections 6.1 and 17.
EXPECTED_P43_STATUS_LINE = (
    "| P4.3 | Integrated analysis, documentation, audit | YES | YES | YES |"
)
FORBIDDEN_P43_STATUS_LINES = (
    "| P4.3 | Integrated analysis, documentation, audit | NO | NO | NO |",
    "| P4.3 | Integrated analysis, documentation, audit | NO | YES | NO |",
    "| P4.3 | Integrated analysis, documentation, audit | NO | NO | YES |",
    "| P4.3 | Integrated analysis, documentation, audit | NO | YES | YES |",
    "| P4.3 | Integrated analysis, documentation, audit | YES | NO | NO |",
    "| P4.3 | Integrated analysis, documentation, audit | YES | YES | NO |",
    "| P4.3 | Integrated analysis, documentation, audit | YES | NO | YES |",
)
CLOSED_STATUS_LINES = (
    "| P1.4 | Profiling, validation, analysis, pilot | YES | YES | YES |",
    "| P2.4 | Profiling and empirical ceiling | YES | YES | YES |",
    "| P3.5 | Five shapes and comparison | YES | YES | YES |",
)

REQUIRED_PROTOCOL_STATEMENTS = (
    "P4.1 = YES / YES / YES",
    "P4.1 is implemented infrastructure",
    "Independent audit: PASSED",
    "GB300 verification: PASSED",
    "Pilot campaign `20260812T013848Z`: COMPLETE",
    "No publishable result exists",
    "P4.2 is now closed as `YES / YES / YES`",
    "P4.3 will perform integrated analysis and publication review",
)
FORBIDDEN_PROTOCOL_STATEMENTS = (
    "P4.1 = YES / NO / NO",
    "Independent audit: PENDING",
    "GB300 verification: PENDING",
    "No Phase 4 campaign was executed",
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
# The integer-encoded CUDA API versions the closed unit schemas record. One
# component records them as an integer and another as a string; both must
# normalize to the same value.
FIXTURE_CUDA_DRIVER = 13010
FIXTURE_CUDA_RUNTIME = "13010"


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
        self.argvs: list[list[str]] = []
        self.fail_targets: dict[str, int] = {}
        self.interrupt_targets: set[str] = set()
        self.interrupt_validate = False
        self.watch_campaign_id: str | None = None
        self.child_noise = ""
        self.destroy_preflight_at: str | None = None
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
        self.unit_compute_capability = FIXTURE_GPU_CC
        self.unit_cuda_driver_version: object = FIXTURE_CUDA_DRIVER
        self.unit_cuda_runtime_version: object = FIXTURE_CUDA_RUNTIME
        self.unit_provenance_drop: str | None = None
        # Applied to the P2.4 payload only, so one component can be given
        # provenance that disagrees with the component already accepted.
        self.p24_provenance_overrides: dict = {}
        # The synthetic stand-ins for the two pins a real loader derives from
        # the unit's own terminal manifest revision and its own
        # verify_campaign_evidence_integrity() snapshot.
        self.unit_manifest_body = "synthetic-unit-manifest"
        self.unit_evidence_body = "synthetic-unit-evidence"

    # -- seams ----------------------------------------------------------

    def now_utc(self) -> str:
        # The clock is also the interruption seam: campaign.validate reads it
        # immediately after writing its two no-clobber logs, which is exactly
        # the window that used to leave a log with no attempt record.
        if self.interrupt_validate and self.validate_logs():
            self.interrupt_validate = False
            raise KeyboardInterrupt
        self.clock += 1
        minute, second = divmod(self.clock, 60)
        hour, minute = divmod(minute, 60)
        return f"20990101T{hour:02d}{minute:02d}{second:02d}Z"

    def validate_logs(self) -> list[str]:
        if self.watch_campaign_id is None:
            return []
        logs = self.campaign_dir(self.watch_campaign_id) / "logs"
        if not logs.is_dir():
            return []
        return sorted(p.name for p in logs.iterdir()
                      if p.name.startswith("campaign_validate."))

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
        # Recorded so the suite can prove every orchestrated invocation is
        # silent and prints no directory diagnostics.
        self.argvs.append(list(argv))
        target = argv[-1]
        self.calls.append((target, dict(env)))
        if self.child_noise:
            os.write(stderr_fd, self.child_noise.encode("utf-8"))
        if target in self.interrupt_targets:
            raise KeyboardInterrupt
        if target == self.destroy_preflight_at:
            # A validated preflight summary that disappears mid-campaign: the
            # final gate then reports it by its absolute path, which is exactly
            # the durable text that must be redacted before it is written.
            for directory in sorted((self.root / "results" / "preflight").iterdir()):
                (directory / "summary.json").unlink()
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

    def _unit_payload(self, state, revision, rel, overrides=None):
        """The payload a real semantic loader returns: the unit's manifest and
        revision, the exact revision file that carried this state (a real file
        under the temporary root, so the SHA-256 pin is a real hash of real
        bytes), and -- at the states the unit's own protocol re-verifies -- a
        digest standing in for its verify_campaign_evidence_integrity()
        snapshot."""
        if state is None:
            raise RuntimeError("no such campaign")
        manifest_rel = f"{rel}/manifest/{revision:06d}.json"
        path = self.root / manifest_rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(
                json.dumps({"state": state, "manifest_revision": revision,
                            "body": self.unit_manifest_body}, indent=2) + "\n",
                encoding="utf-8")
        provenance = {
            "git_commit": self.unit_commit,
            "gpu_uuid": self.unit_gpu_uuid,
            "gpu_name": self.unit_gpu_name,
            "compute_capability": self.unit_compute_capability,
            "cuda_driver_version": self.unit_cuda_driver_version,
            "cuda_runtime_version": self.unit_cuda_runtime_version,
        }
        provenance.update(overrides or {})
        if self.unit_provenance_drop is not None:
            provenance.pop(self.unit_provenance_drop, None)
        payload = {
            "manifest": {"state": state, "provenance": provenance},
            "revision": revision,
            "campaign_dir": rel,
            "manifest_path": manifest_rel,
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        if state in ("COMPLETE", "ANALYZED", "INCONCLUSIVE"):
            payload["evidence_sha256"] = hashlib.sha256(
                f"{rel}|{state}|{self.unit_evidence_body}".encode("utf-8")).hexdigest()
        return payload

    def p14_loader(self, campaign_id: str) -> dict:
        return self._unit_payload(self.p14_state, self.p14_revision,
                                  f"{self.o.P14_RAW_REL}/{campaign_id}")

    def p24_loader(self, campaign_id: str) -> dict:
        return self._unit_payload(self.p24_state, self.p24_revision,
                                  f"{self.o.P24_RAW_REL}/{campaign_id}",
                                  self.p24_provenance_overrides)

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
        self.watch_campaign_id = campaign_id
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
    _test_unit_evidence_pinning(reporter, o, p35)
    _test_terminal_analysis_hashing(reporter, o)
    _test_provenance_agreement(reporter, o, p35)
    _test_validate_interruption(reporter, o, p35)
    _test_durable_log_privacy(reporter, o, p35)
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

    # An already occupied attempt-log name is never reused or overwritten: the
    # next attempt number is derived from the logs actually on disk as well as
    # from the validated campaign state, so a log an interrupted attempt left
    # behind cannot block the stage forever either.
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
    status = fixture.run(campaign_id=campaign_id, resume=True)
    reporter.check("an occupied attempt-log name is skipped, not reused", status == o.EXIT_OK,
                   str(status))
    reporter.check("the pre-existing log file was never overwritten",
                   squatter.read_text(encoding="utf-8") == "pre-existing evidence", "")
    logs = sorted(p.name for p in (fixture.campaign_dir(campaign_id) / "logs").iterdir())
    reporter.check("the retried stage claimed the next free attempt number",
                   "preflight.003.stdout.log" in logs, str(logs))


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


def _test_unit_evidence_pinning(reporter: Reporter, o, p35) -> None:
    """A completed unit is pinned by the exact terminal manifest revision it was
    accepted at -- path, revision, and SHA-256 -- plus the digest of its own
    evidence-integrity snapshot. None of them may drift afterwards."""
    campaign_id = "20991221T000000Z"

    def completed():
        fixture = _fixture(o, p35, _STACK)
        status = fixture.run(campaign_id=campaign_id)
        reporter.check("a campaign that pins its units still completes", status == o.EXIT_OK,
                       str(status))
        return fixture

    fixture = completed()
    manifest = fixture.load_manifest(campaign_id)
    evidence = manifest["stage_results"]["memory.analyze"]["evidence"]
    reporter.check("an accepted terminal unit records its exact manifest revision path and hash",
                   evidence.get("unit_manifest_path", "").endswith(
                       f"manifest/{evidence.get('unit_manifest_revision'):06d}.json")
                   and bool(o.SHA256_RE.fullmatch(evidence.get("unit_manifest_sha256", ""))),
                   str(evidence))
    reporter.check("an accepted terminal unit records its own evidence-integrity pin",
                   bool(o.SHA256_RE.fullmatch(evidence.get("unit_evidence_sha256", ""))),
                   str(evidence))
    reporter.check("no pinned unit reference is an absolute path",
                   not str(evidence["unit_manifest_path"]).startswith("/"), "")

    # 1. A valid but LATER terminal revision must not replace the accepted one.
    fixture = completed()
    fixture.p14_revision += 1
    reporter.rejects("a later P1.4 terminal manifest revision is rejected",
                     lambda: fixture.run(campaign_id=campaign_id, resume=True), o.RunFailure,
                     "later terminal revision is never adopted silently")

    fixture = completed()
    fixture.p24_revision += 1
    reporter.rejects("a later P2.4 terminal manifest revision is rejected",
                     lambda: fixture.run(campaign_id=campaign_id, resume=True), o.RunFailure,
                     "later terminal revision is never adopted silently")

    # 2. The accepted revision file's own content may not change either.
    fixture = completed()
    pinned = fixture.root / fixture.load_manifest(campaign_id)[
        "stage_results"]["memory.analyze"]["evidence"]["unit_manifest_path"]
    pinned.write_text(pinned.read_text(encoding="utf-8") + " ", encoding="utf-8")
    reporter.rejects("a changed accepted unit manifest is rejected",
                     lambda: fixture.run(campaign_id=campaign_id, resume=True), o.RunFailure,
                     "changed since it was accepted")
    reporter.check("the changed unit manifest was not regenerated",
                   pinned.read_text(encoding="utf-8").endswith(" "), "")

    # 3. Changed referenced evidence: the unit's own integrity snapshot moves.
    fixture = completed()
    fixture.unit_evidence_body = "tampered"
    reporter.rejects("changed referenced unit evidence is rejected",
                     lambda: fixture.run(campaign_id=campaign_id, resume=True), o.RunFailure,
                     "evidence-integrity snapshot no longer matches")

    # 4. A unit whose own integrity gate fails is never accepted at all.
    fixture = _fixture(o, p35, _STACK)
    original = fixture.p14_loader

    def failing_gate(cid):
        payload = original(cid)
        if payload["manifest"]["state"] == "ANALYZED":
            raise RuntimeError("the unit's own evidence-integrity gate failed: ['synthetic']")
        return payload

    fixture.p14_loader = failing_gate
    reporter.rejects("a unit whose own evidence-integrity gate fails is never accepted",
                     lambda: fixture.run(campaign_id="20991222T000000Z"), o.RunFailure,
                     "evidence-integrity gate failed")


def _test_terminal_analysis_hashing(reporter: Reporter, o) -> None:
    """The terminal pin also covers the derived analysis files that the two
    unit evidence-integrity snapshots do not include."""
    root = Path(tempfile.mkdtemp(prefix="p41-analysis-"))
    _STACK.append(root)
    campaign_dir = root / "unit"
    (campaign_dir / "manifest").mkdir(parents=True)
    (campaign_dir / "analysis" / "figures").mkdir(parents=True)
    report_path = campaign_dir / "analysis" / "report.md"
    figure_path = campaign_dir / "analysis" / "figures" / "result.svg"
    report_path.write_text("terminal report\n", encoding="utf-8")
    figure_path.write_text("<svg/>\n", encoding="utf-8")
    inventory = ("analysis/report.md", "analysis/figures/result.svg")
    hashes = {
        relative: hashlib.sha256((campaign_dir / relative).read_bytes()).hexdigest()
        for relative in inventory
    }
    manifest = {"state": "ANALYZED", "artifact_sha256": hashes}
    (campaign_dir / "manifest" / "000000.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8")

    class Analyzer:
        ANALYSIS_ARTIFACT_RELATIVE_PATHS = inventory

        @staticmethod
        def verify_campaign_evidence_integrity(_campaign_dir, _manifest):
            return [], {"raw_evidence": "verified"}

    previous_root = o.REPO_ROOT
    o.REPO_ROOT = root
    try:
        def load(_campaign_dir):
            return manifest, 0

        def accepted():
            return o._load_unit_campaign(
                Analyzer, "unit", resolve=lambda _relative: campaign_dir, load=load)

        payload = accepted()
        reporter.check("terminal P1.4/P2.4 analysis artifacts contribute to the evidence pin",
                       bool(o.SHA256_RE.fullmatch(payload.get("evidence_sha256", ""))),
                       str(payload))
        report_path.write_text("tampered terminal report\n", encoding="utf-8")
        reporter.rejects("a changed terminal analysis artifact is rejected on revalidation",
                         accepted, o.RunFailure, "content changed after terminal analysis")
    finally:
        o.REPO_ROOT = previous_root


def _test_provenance_agreement(reporter: Reporter, o, p35) -> None:
    """Commit, GPU identity, compute capability, and both CUDA API versions must
    agree across every selected component and the current preflight."""
    for field, value in (("cuda_driver_version", 13020), ("cuda_runtime_version", 13020),
                         ("compute_capability", "9.9")):
        fixture = _fixture(o, p35, _STACK)
        fixture.p24_provenance_overrides = {field: value}
        reporter.rejects(f"a component disagreeing on {field} is rejected",
                         lambda f=fixture: f.run(campaign_id="20991223T000000Z"), o.RunFailure,
                         field)

    for field in o.UNIT_PROVENANCE_FIELDS:
        fixture = _fixture(o, p35, _STACK)
        fixture.unit_provenance_drop = field
        reporter.rejects(f"a unit whose provenance is missing {field} is rejected",
                         lambda f=fixture: f.run(campaign_id="20991224T000000Z"), o.RunFailure,
                         "unusable campaign provenance")

    for label, value in (("an empty", ""), ("a non-scalar", [13010]), ("a boolean", True)):
        fixture = _fixture(o, p35, _STACK)
        fixture.unit_cuda_driver_version = value
        reporter.rejects(f"{label} CUDA driver version is rejected",
                         lambda f=fixture: f.run(campaign_id="20991225T000000Z"), o.RunFailure,
                         "cuda_driver_version")

    # The same version in the two textual formats the closed schemas use is one
    # value, not a mismatch.
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20991226T000000Z"
    fixture.unit_cuda_driver_version = 13010
    fixture.p24_provenance_overrides = {"cuda_driver_version": "13010"}
    status = fixture.run(campaign_id=campaign_id)
    reporter.check("an int and a string spelling of the same CUDA version agree",
                   status == o.EXIT_OK, str(status))
    manifest = fixture.load_manifest(campaign_id)
    recorded = {record["evidence"].get("cuda_driver_version")
                for record in manifest["stage_results"].values()
                if record["evidence"].get("cuda_driver_version") is not None}
    reporter.check("the normalized CUDA driver version is recorded once, not twice",
                   recorded == {"13010"}, str(recorded))

    # Provenance is re-compared on resume, not only at acceptance.
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20991227T000000Z"
    fixture.run(campaign_id=campaign_id)
    fixture.unit_cuda_runtime_version = 13020
    reporter.rejects("changed unit provenance aborts a completed-campaign revalidation",
                     lambda: fixture.run(campaign_id=campaign_id, resume=True), o.RunFailure,
                     "cuda_runtime_version")


def _test_validate_interruption(reporter: Reporter, o, p35) -> None:
    """An interruption inside campaign.validate is handled at the same
    orchestration boundary as every other stage, and leaves the stage eligible
    for a new attempt."""
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20991228T000000Z"
    fixture.interrupt_validate = True
    try:
        fixture.run(campaign_id=campaign_id)
        reporter.check("an interrupted campaign.validate propagates KeyboardInterrupt", False,
                       "no raise")
    except KeyboardInterrupt:
        reporter.check("an interrupted campaign.validate propagates KeyboardInterrupt", True)

    manifest = fixture.load_manifest(campaign_id)
    reporter.check("an interrupted campaign.validate is recorded INTERRUPTED",
                   manifest["state"] == o.STATE_INTERRUPTED
                   and manifest.get("failure_stage") == "campaign.validate",
                   f"{manifest['state']} {manifest.get('failure_stage')}")
    interrupted = [e for e in manifest["stage_attempts"]
                   if e["stage"] == "campaign.validate" and e["outcome"] == "INTERRUPTED"]
    reporter.check("the interrupted attempt records the conventional interrupted status",
                   len(interrupted) == 1 and interrupted[0]["exit_status"] == o.EXIT_INTERRUPTED
                   and o.EXIT_INTERRUPTED == 130, str(interrupted))
    reporter.check("every stage completed before the interruption is preserved",
                   list(manifest["stage_results"]) == list(EXPECTED_FULL_PLAN[:-1]),
                   str(list(manifest["stage_results"])))
    first_logs = fixture.validate_logs()
    reporter.check("the interrupted attempt's own logs are preserved",
                   first_logs == ["campaign_validate.001.stderr.log",
                                  "campaign_validate.001.stdout.log"], str(first_logs))
    kept = (fixture.campaign_dir(campaign_id) / "logs"
            / "campaign_validate.001.stdout.log").read_text(encoding="utf-8")

    status = fixture.run(campaign_id=campaign_id, resume=True)
    reporter.check("resuming after an interrupted campaign.validate completes it",
                   status == o.EXIT_OK, str(status))
    reporter.check("the resumed validation used a new attempt log",
                   fixture.validate_logs() == ["campaign_validate.001.stderr.log",
                                               "campaign_validate.001.stdout.log",
                                               "campaign_validate.002.stderr.log",
                                               "campaign_validate.002.stdout.log"],
                   str(fixture.validate_logs()))
    reporter.check("the interrupted attempt's log was never overwritten",
                   (fixture.campaign_dir(campaign_id) / "logs"
                    / "campaign_validate.001.stdout.log").read_text(encoding="utf-8") == kept, "")
    _check_exit_code_contract(reporter, o)

    manifest = fixture.load_manifest(campaign_id)
    reporter.check("the resumed campaign reaches COMPLETE",
                   manifest["state"] == o.STATE_COMPLETE, manifest["state"])
    attempts = [e["attempt"] for e in manifest["stage_attempts"]
                if e["stage"] == "campaign.validate"]
    reporter.check("both validation attempts are recorded explicitly, in order",
                   attempts == [1, 2], str(attempts))
    reporter.check("no completed stage was re-executed by the resume",
                   [target for target, _env in fixture.calls].count(
                       "gemm-comparison-p35-smoke") == 1,
                   str([target for target, _env in fixture.calls]))

    # An attempt log left behind with no attempt record (a hard interruption)
    # is skipped rather than reused, so the stage stays resumable.
    revisions_before = fixture.manifest_revisions(campaign_id)
    calls_before = len(fixture.calls)
    status = fixture.run(campaign_id=campaign_id, resume=True)
    reporter.check("resuming the now COMPLETE campaign stays read-only",
                   status == o.EXIT_OK
                   and fixture.manifest_revisions(campaign_id) == revisions_before
                   and len(fixture.calls) == calls_before
                   and len(fixture.validate_logs()) == 4, str(status))

    # If interruption lands between the two exclusive log writes, the lone
    # file is preserved but no attempt may claim a stderr log that never
    # existed. The disk-derived attempt counter then skips the partial number.
    fixture = _fixture(o, p35, _STACK)
    campaign_id = "20991231T000000Z"
    original_write = o.write_file_exclusive
    interrupted_once = False

    def interrupt_between_logs(name, payload, *, dir_fd):
        nonlocal interrupted_once
        if not interrupted_once and name == "campaign_validate.001.stderr.log":
            interrupted_once = True
            raise KeyboardInterrupt
        return original_write(name, payload, dir_fd=dir_fd)

    o.write_file_exclusive = interrupt_between_logs
    try:
        try:
            fixture.run(campaign_id=campaign_id)
            reporter.check("an interruption between validation logs propagates", False,
                           "no raise")
        except KeyboardInterrupt:
            reporter.check("an interruption between validation logs propagates", True)
    finally:
        o.write_file_exclusive = original_write

    manifest = fixture.load_manifest(campaign_id)
    partial_attempts = [entry for entry in manifest["stage_attempts"]
                        if entry["stage"] == "campaign.validate"]
    reporter.check("a partial validation log pair is not recorded as an attempt",
                   partial_attempts == [], str(partial_attempts))
    reporter.check("the unpaired validation log is preserved without a false reference",
                   fixture.validate_logs() == ["campaign_validate.001.stdout.log"],
                   str(fixture.validate_logs()))
    status = fixture.run(campaign_id=campaign_id, resume=True)
    manifest = fixture.load_manifest(campaign_id)
    resumed_attempts = [entry["attempt"] for entry in manifest["stage_attempts"]
                        if entry["stage"] == "campaign.validate"]
    reporter.check("resume skips the partial log number and completes validation",
                   status == o.EXIT_OK and resumed_attempts == [2]
                   and fixture.validate_logs() == [
                       "campaign_validate.001.stdout.log",
                       "campaign_validate.002.stderr.log",
                       "campaign_validate.002.stdout.log",
                   ], f"status={status} attempts={resumed_attempts} logs={fixture.validate_logs()}")


def _check_exit_code_contract(reporter: Reporter, o) -> None:
    """The real process-level mapping, exercised directly: an interruption exits
    130, and every other escaping error keeps its own documented status."""
    original = o.main
    cases = (
        ("an interruption", KeyboardInterrupt(), o.EXIT_INTERRUPTED),
        ("a stage failure", o.RunFailure("synthetic"), o.EXIT_RUN_FAILURE),
        ("a manifest defect", o.ManifestError("synthetic"), o.EXIT_RUN_FAILURE),
        ("a precondition failure", o.OrchestratorError("synthetic"), o.EXIT_PRECONDITION),
    )
    try:
        for label, error, expected in cases:
            def raiser(argv=None, _error=error):
                raise _error

            o.main = raiser
            with redirect_stderr(io.StringIO()):
                status = o.cli_main([])
            reporter.check(f"{label} exits {expected}", status == expected, str(status))
        o.main = lambda argv=None: o.EXIT_OK
        with redirect_stderr(io.StringIO()):
            reporter.check("a successful run still exits 0", o.cli_main([]) == o.EXIT_OK, "")
    finally:
        o.main = original


def _test_durable_log_privacy(reporter: Reporter, o, p35) -> None:
    """No durable Phase 4 log or manifest field carries this checkout's absolute
    path, and the invoked children's own output is preserved untouched."""
    parent = Path(tempfile.mkdtemp(prefix="p41-"))
    _STACK.append(parent)
    root = parent / "work" / "gb300-gemm-anatomy"
    root.mkdir(parents=True)
    fixture = Fixture(o, p35, root)
    fixture.child_noise = (
        "stub: ordinary child diagnostic line\n"
        f"child error at {root}/results/bad.csv\n"
    )
    fixture.destroy_preflight_at = "compute-umma-p24-analyze"
    campaign_id = "20991229T000000Z"
    reporter.rejects("evidence that disappeared mid-campaign fails the final gate",
                     lambda: fixture.run(campaign_id=campaign_id), o.RunFailure,
                     "integrity gate failed")

    absolute = str(root)
    logs_dir = fixture.campaign_dir(campaign_id) / "logs"
    durable = sorted(p.name for p in logs_dir.iterdir())
    for name in durable:
        text = (logs_dir / name).read_text(encoding="utf-8", errors="replace")
        reporter.check(f"the durable log {name} carries no absolute checkout path",
                       absolute not in text, text[:200])
    validate_stderr = (logs_dir / "campaign_validate.001.stderr.log").read_text(encoding="utf-8")
    reporter.check("the redacted diagnostic keeps its stable token",
                   o.REPO_ROOT_TOKEN in validate_stderr, validate_stderr[:200])
    reporter.check("the redacted diagnostic keeps the failure information itself",
                   "preflight" in validate_stderr and "does not exist" in validate_stderr,
                   validate_stderr[:200])
    child_log = (logs_dir / "umma_analyze.001.stderr.log").read_text(encoding="utf-8")
    reporter.check("ordinary child output is preserved verbatim",
                   "ordinary child diagnostic line" in child_log, child_log[:200])
    reporter.check("a child-emitted checkout path is narrowly redacted before log publication",
                   f"{o.REPO_ROOT_TOKEN}/results/bad.csv" in child_log
                   and absolute not in child_log, child_log[:200])

    document = json.dumps(fixture.load_manifest(campaign_id))
    reporter.check("no manifest field carries the absolute checkout path",
                   absolute not in document, "")
    reporter.check("the manifest failure detail is redacted, not discarded",
                   o.REPO_ROOT_TOKEN in document, "")

    reporter.check("every orchestrated Make invocation is silent",
                   all(argv[:3] == ["make", "--silent", "--no-print-directory"]
                       for argv in fixture.argvs) and bool(fixture.argvs),
                   str(fixture.argvs[:1]))
    reporter.check("the invoked target itself is unchanged",
                   [argv[-1] for argv in fixture.argvs]
                   == [EXPECTED_MAKE_TARGETS[stage] for stage in EXPECTED_FULL_PLAN[:-1]],
                   str([argv[-1] for argv in fixture.argvs]))


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
    reporter.check("the orchestrator reuses each unit's own evidence-integrity gate",
                   orchestrator.count("verify_campaign_evidence_integrity") >= 1, "")
    reporter.check("the orchestrator pins the accepted unit manifest revision by hash",
                   all(field in orchestrator for field in
                       ("unit_manifest_path", "unit_manifest_sha256", "unit_evidence_sha256")), "")
    reporter.check("the terminal unit pin re-hashes the canonical analysis artifacts",
                   "verify_terminal_analysis_artifacts" in orchestrator
                   and "ANALYSIS_ARTIFACT_RELATIVE_PATHS" in orchestrator, "")
    reporter.check("the orchestrator exits 130 when it is interrupted",
                   "return EXIT_INTERRUPTED" in orchestrator
                   and "raise SystemExit(cli_main())" in orchestrator
                   and module.EXIT_INTERRUPTED == 130, str(module.EXIT_INTERRUPTED))
    reporter.check("the orchestrator invokes every Make target silently",
                   module.MAKE_QUIET_FLAGS == ("--silent", "--no-print-directory"),
                   str(module.MAKE_QUIET_FLAGS))
    reporter.check("child text crosses a narrow checkout-root redaction boundary",
                   "copy_log_stream" in orchestrator
                   and "RUN_CONTAINER_STDOUT_IS_DATA" in orchestrator, "")
    reporter.check("the orchestrator compares both CUDA API versions between components",
                   {"cuda_driver_version", "cuda_runtime_version"}
                   <= set(module.COMPARABLE_EVIDENCE_FIELDS)
                   and {"cuda_driver_version", "cuda_runtime_version"}
                   <= set(module.UNIT_PROVENANCE_FIELDS),
                   str(module.COMPARABLE_EVIDENCE_FIELDS))
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


def _parse_makefile_rules(makefile: str) -> dict[str, tuple[list[str], list[str]]]:
    """target -> (prerequisites, recipe lines). Simple `NAME := value` variables
    are expanded so a prerequisite written through one is still resolved."""
    variables = dict(re.findall(r"^([A-Za-z0-9_]+)\s*:=\s*(.*)$", makefile, re.MULTILINE))

    def expand(text: str) -> str:
        for _ in range(4):
            replaced = re.sub(r"\$\(([A-Za-z0-9_]+)\)",
                              lambda m: variables.get(m.group(1), m.group(0)), text)
            if replaced == text:
                break
            text = replaced
        return text

    rules: dict[str, tuple[list[str], list[str]]] = {}
    current: list[str] = []
    for line in makefile.splitlines():
        if line.startswith("\t"):
            for target in current:
                rules[target][1].append(line[1:])
            continue
        match = re.match(r"^([A-Za-z0-9._/$()-][^:=]*):(?!=)(.*)$", line)
        if match is None:
            # A comment or a blank line does not end a recipe; anything else at
            # column 0 does. Attributing too many lines to a target only makes
            # the scan below stricter, which is the fail-closed direction.
            if line.strip() and not line.startswith(("#", " ")):
                current = []
            continue
        targets = expand(match.group(1)).split()
        prerequisites = expand(match.group(2)).split()
        current = []
        for target in targets:
            rules.setdefault(target, ([], []))
            rules[target][0].extend(prerequisites)
            current.append(target)
    return rules


def _target_closure(rules: dict[str, tuple[list[str], list[str]]], root: str) -> list[str]:
    seen: list[str] = []
    pending = [root]
    while pending:
        target = pending.pop()
        if target in seen:
            continue
        seen.append(target)
        pending.extend(rules.get(target, ([], []))[0])
    return seen


def _check_p41_check_is_container_free(reporter: Reporter, makefile: str) -> None:
    """The documented P4.1 contract is that `make phase4-p41-check` runs with no
    container runtime, no nvidia-smi, no CUDA compilation, no GPU, and no
    network. This walks the target's real transitive prerequisites and fails if
    any recipe in that closure would invoke one of them."""
    rules = _parse_makefile_rules(makefile)
    reporter.check("the Makefile rule graph is parseable and declares phase4-p41-check",
                   "phase4-p41-check" in rules, str(sorted(rules)[:5]))
    if "phase4-p41-check" not in rules:
        return
    closure = _target_closure(rules, "phase4-p41-check")
    unknown = [target for target in closure if target not in rules]
    reporter.check("every prerequisite of phase4-p41-check resolves to a real rule",
                   not unknown, str(unknown))
    offenders: list[str] = []
    for target in closure:
        for line in rules.get(target, ([], []))[1]:
            command = line.lstrip().lstrip("@-+").lstrip()
            for name in FORBIDDEN_CHECK_PATH_COMMANDS:
                if re.search(_COMMAND_POSITION.format(name=re.escape(name)), command):
                    offenders.append(f"{target}: {line.strip()[:80]}")
    reporter.check("no recipe reachable from phase4-p41-check invokes a container runtime, "
                   "nvidia-smi, or a CUDA compiler", not offenders, str(offenders[:3]))
    reporter.check("the Docker-backed P2.4 and P3.5 gates are not reachable from "
                   "phase4-p41-check",
                   "compute-umma-p24-check" not in closure
                   and "gemm-comparison-p35-check" not in closure, str(sorted(closure)))
    reporter.check("phase4-p41-check still revalidates the GPU-free P1.4 and P2.4 gates",
                   "memory-paths-p14-check" in closure
                   and "compute-umma-p24-check-gpu-free" in closure, str(sorted(closure)))
    # The split is only legitimate while the full P2.4 gate still runs that same
    # GPU-free half plus its own three Docker-backed prerequisites.
    reporter.check("the full P2.4 gate still contains its GPU-free half and its own gates",
                   set(rules.get("compute-umma-p24-check", ([], []))[0])
                   == {"compute-umma-1sm-check", "compute-umma-2sm-check",
                       "compute-umma-sweep-check", "compute-umma-p24-check-gpu-free"},
                   str(rules.get("compute-umma-p24-check", ([], []))[0]))


def _check_makefile(reporter: Reporter, makefile: str) -> None:
    reporter.check("the Makefile declares phase4-p41-plan",
                   re.search(r"^phase4-p41-plan:", makefile, re.MULTILINE) is not None, "")
    reporter.check("the Makefile declares phase4-p41-check with the GPU-free gates only",
                   re.search(
                       r"^phase4-p41-check: memory-paths-p14-check "
                       r"compute-umma-p24-check-gpu-free$",
                       makefile, re.MULTILINE) is not None, "")
    _check_p41_check_is_container_free(reporter, makefile)
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
    reporter.check("PLAN.md records P4.1 as closed",
                   EXPECTED_P41_STATUS_LINE in plan_text, "")
    for wrong in FORBIDDEN_P41_STATUS_LINES:
        reporter.check(f"PLAN.md does not record the untrue P4.1 status {wrong!r}",
                       wrong not in plan_text, "")
    reporter.check("PLAN.md records the closed P4.2 status",
                   EXPECTED_P42_STATUS_LINE in plan_text, "")
    for wrong in FORBIDDEN_P42_STATUS_LINES:
        reporter.check(f"PLAN.md does not record the untrue P4.2 status {wrong!r}",
                       wrong not in plan_text, "")
    reporter.check("PLAN.md records the truthful implemented-only P4.3 status",
                   EXPECTED_P43_STATUS_LINE in plan_text, "")
    for wrong in FORBIDDEN_P43_STATUS_LINES:
        reporter.check(f"PLAN.md does not record the stale or untrue P4.3 status {wrong!r}",
                       wrong not in plan_text, "")
    for line in CLOSED_STATUS_LINES:
        reporter.check(f"PLAN.md still records the closed {line!r}", line in plan_text, "")
    reporter.check("the Makefile's frontier assertion records the closed P4.1 row",
                   "P4.1 | Orchestrator | YES | YES | YES |" in makefile, "")
    reporter.check("the Makefile records P4.2 as closed",
                   "P4.2 | Pilot plus three final campaigns | YES | YES | YES |" in makefile, "")
    # Presence only: the Makefile necessarily also NAMES the stale and
    # impossible rows in order to reject them with `! grep`, so a
    # "must not appear anywhere" scan of its text would be self-defeating.
    reporter.check("the Makefile asserts the truthful accepted P4.3 row",
                   "P4.3 | Integrated analysis, documentation, audit | YES | YES | YES |"
                   in makefile, "")
    reporter.check("P4.2 did not change the audited P4.1 runner or orchestrator",
                   "P4.1 is implemented infrastructure" in protocol
                   and "GB300 verification: PASSED" in protocol, "")

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
    reporter.check("README.md records P4.1 as audited and GB300-verified",
                   "P4.1: CLOSED; independent audit: YES; GB300 verification: YES"
                   in readme, "")
    # Phase 4 closure became truthful when P4.3 was accepted. P4.1 never
    # authorizes it on its own, so the claim and the external acceptance
    # attestation must travel together: a README that closes Phase 4 without
    # naming the attestation is still rejected.
    reporter.check("README.md closes Phase 4 only together with the P4.3 acceptance "
                   "attestation",
                   "Phase 4: CLOSED" not in readme
                   or "src/phase4/P4_3_ACCEPTANCE.json" in readme, "")
    reporter.check("results/README.md documents the Phase 4 orchestration tree",
                   "results/raw/phase4/" in results_readme, "")
    reporter.check("results/README.md records the completed Phase 4 pilot",
                   "20260812T013848Z" in results_readme
                   and re.search(r"No publishable\s+Phase 4 result exists",
                                 results_readme) is not None, "")


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
