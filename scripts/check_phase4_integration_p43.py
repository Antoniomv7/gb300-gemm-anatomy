#!/usr/bin/env python3
"""Fail-closed, GPU-free repository-contract checker for P4.3.

P4.3 is the offline integrated analysis over the frozen Phase 4 population. It
owns no experiment, no runner, and no evidence: it reads already accepted
artifacts and produces one small curated tree. This checker proves the parts of
that contract a reviewer should not have to re-derive by hand:

* the frozen population is exactly one excluded pilot plus three final
  campaigns, declared as constants and never discovered;
* the frozen cross-campaign statistical policy is the one the protocol states,
  exercised against the real analyzer rather than asserted in prose;
* the analyzer reuses P4.1's, P4.2's, and P3.5's audited code instead of
  writing a second interpretation of the same contracts;
* nothing in P4.3 can execute a GPU command, a container, Nsight Compute, a
  campaign, or any child process at all;
* nothing in P4.3 can write, repair, resume, or regenerate raw evidence;
* the artifact inventory is frozen and matches the protocol exactly;
* the status frontier is truthful: P4.3 is implemented, not audited, and its
  production analysis has not been run.

Two modes::

    python3 -I -B scripts/check_phase4_integration_p43.py --self-test
        Focused synthetic suite over temporary fixtures only.

    python3 -I -B scripts/check_phase4_integration_p43.py <repo-root>
        The frozen P4.3 repository contract. Needs no results/raw/, no cluster
        evidence, no container runtime, and no network.

Exit codes: 0 OK; 1 at least one check failed; 2 usage error.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import inspect
import io
import json
import linecache
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent

ORCHESTRATOR_RELATIVE_PATH = "scripts/phase4_orchestrator.py"
RUN_ALL_RELATIVE_PATH = "scripts/run_all.sh"
P41_CHECKER_RELATIVE_PATH = "scripts/check_phase4_orchestrator_p41.py"
P41_PROTOCOL_RELATIVE_PATH = "src/phase4/P4_1_PROTOCOL.md"
P42_CHECKER_RELATIVE_PATH = "scripts/check_phase4_campaigns_p42.py"
P42_PROTOCOL_RELATIVE_PATH = "src/phase4/P4_2_PROTOCOL.md"
P43_ANALYZER_RELATIVE_PATH = "scripts/analyze_phase4_p43.py"
P43_CHECKER_RELATIVE_PATH = "scripts/check_phase4_integration_p43.py"
P43_PROTOCOL_RELATIVE_PATH = "src/phase4/P4_3_PROTOCOL.md"
P43_ACCEPTANCE_RELATIVE_PATH = "src/phase4/P4_3_ACCEPTANCE.json"

# The accepted candidate bundle. It is small, curated, and committed, so the
# repository-contract check can bind the acceptance attestation to the exact
# reviewed bytes without reading one byte of raw campaign evidence. The order
# is the analyzer's frozen inventory order, and a check below proves it.
ACCEPTED_BUNDLE_ROOT = "results/phase4"
ACCEPTED_BUNDLE_ARTIFACTS = (
    "memory_paths.csv",
    "umma_throughput.csv",
    "gemm_comparison.csv",
    "integrated_summary.json",
    "report.md",
    "figures/memory_paths.svg",
    "figures/umma_throughput.svg",
    "figures/gemm_comparison.svg",
    "analysis_manifest.json",
)
ACCEPTED_BUNDLE_RELATIVE_PATHS = tuple(
    f"{ACCEPTED_BUNDLE_ROOT}/{name}" for name in ACCEPTED_BUNDLE_ARTIFACTS)

REQUIRED_P43_FILES = (
    P43_ANALYZER_RELATIVE_PATH,
    P43_CHECKER_RELATIVE_PATH,
    P43_PROTOCOL_RELATIVE_PATH,
    P43_ACCEPTANCE_RELATIVE_PATH,
    ORCHESTRATOR_RELATIVE_PATH,
    P42_CHECKER_RELATIVE_PATH,
    P42_PROTOCOL_RELATIVE_PATH,
    P41_CHECKER_RELATIVE_PATH,
    P41_PROTOCOL_RELATIVE_PATH,
    RUN_ALL_RELATIVE_PATH,
) + ACCEPTED_BUNDLE_RELATIVE_PATHS

# ---------------------------------------------------------------------------
# The truthful status frontier this unit is allowed to record. P4.3 is
# implemented, independently audited, and verified against the real GB300
# evidence: the production analysis ran from the audited analyzer commit, the
# curated bundle was recomputed byte for byte, an independent review accepted
# it, and the external acceptance attestation exists and validates. The row
# P4.3 owns advanced by exactly one step to "YES | YES | YES"; every other
# P4.3 state -- including the stale "YES | NO | NO" this checker used to
# require -- stays rejected, and no closed P1-P4.2 assertion was weakened.
# See src/phase4/P4_3_PROTOCOL.md section 17.
# ---------------------------------------------------------------------------
EXPECTED_STATUS_LINES = (
    "| P4.1 | Orchestrator | YES | YES | YES |",
    "| P4.2 | Pilot plus three final campaigns | YES | YES | YES |",
    "| P4.3 | Integrated analysis, documentation, audit | YES | YES | YES |",
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
FORBIDDEN_CLOSED_UNIT_REGRESSIONS = (
    "| P4.1 | Orchestrator | NO | NO | NO |",
    "| P4.1 | Orchestrator | YES | NO | NO |",
    "| P4.2 | Pilot plus three final campaigns | NO | NO | NO |",
    "| P4.2 | Pilot plus three final campaigns | YES | NO | NO |",
    "| P4.2 | Pilot plus three final campaigns | YES | YES | NO |",
)
CLOSED_STATUS_LINES = (
    "| P1.4 | Profiling, validation, analysis, pilot | YES | YES | YES |",
    "| P2.4 | Profiling and empirical ceiling | YES | YES | YES |",
    "| P3.5 | Five shapes and comparison | YES | YES | YES |",
)

STATUS_DOCUMENTS = ("PLAN.md", "README.md", "results/README.md",
                    P41_PROTOCOL_RELATIVE_PATH, P42_PROTOCOL_RELATIVE_PATH,
                    P43_PROTOCOL_RELATIVE_PATH)

# ---------------------------------------------------------------------------
# The frozen closure facts. P4.3 is accepted, and these are the exact
# identities the acceptance record and the status documents must carry. The
# three commits are three different facts about three different events and are
# never interchangeable:
#
#   ACCEPTED_EXECUTION_COMMIT  the commit the three GB300 campaigns RAN from,
#                              so it identifies the experimental evidence;
#   ACCEPTED_ANALYZER_COMMIT   the commit whose analysis code PRODUCED the
#                              candidate bundle;
#   ACCEPTED_CANDIDATE_COMMIT  the commit that CONTAINS the accepted candidate
#                              bytes.
#
# The later documentation/closure commit is none of the three, and it is
# deliberately absent from the attestation: a file cannot carry the hash of the
# commit that adds it.
# ---------------------------------------------------------------------------
ACCEPTED_EXECUTION_COMMIT = "b08e45c2636a3ac17c94ad8b1368084914196d7a"
ACCEPTED_ANALYZER_COMMIT = "2ef1ac52907c407dd43c41661382fc8d5673cce4"
ACCEPTED_CANDIDATE_COMMIT = "577fbe229eb1b857d82f23aacb305136014ec7b0"
ACCEPTED_MANIFEST_SHA256 = (
    "b95d17910f8384187ddc94afacc9081507858de1fb69292f5f3d73bf4cc2d6ac")
ACCEPTED_AUDIT_VERDICT = "ACCEPT WITH NON-BLOCKING OBSERVATIONS"
ACCEPTED_COMPARISON_METHOD = "byte-for-byte"
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# No document may claim a P4.3 status P4.3 does not have, promote the immutable
# candidate bytes themselves -- publication authority is the external
# attestation, never a rewritten candidate -- or declare the whole TFM
# finished: only its EXPERIMENTAL phase is closed, and the thesis analysis,
# writing, and defence remain.
FORBIDDEN_DOCUMENT_CLAIMS = (
    "P4.3 = YES / YES / NO",
    "P4.3 = YES / NO / YES",
    "P4.3 = YES / NO / NO",
    "P4.3 = NO / NO / NO",
    "the TFM is complete",
    "the complete TFM is closed",
    "the whole TFM is closed",
    "the TFM itself is closed",
    "TFM: CLOSED",
    "publishable=true",
    "publishable: true",
    "publishable = true",
    "publication_state: accepted",
    "publication_state=accepted",
)
# The stale pre-acceptance language. Every one of these was true while P4.3 was
# an unaccepted candidate and is false now, so a status document that still
# carries one contradicts the accepted state it also records.
FORBIDDEN_STALE_PRE_ACCEPTANCE_CLAIMS = (
    "awaiting a new independent audit",
    "has not been independently audited",
    "The P4.3 independent audit has not been performed",
    "Independent audit: NOT PERFORMED",
    "Production analysis: NOT RUN",
    "P4.3: IMPLEMENTED; independent audit: NO; production analysis: NO",
    "no acceptance attestation exists",
    "Phase 4 and the complete TFM are not closed",
    "Phase 4 and the complete TFM stay open",
    "no P4.3 result has been accepted for publication",
    "no P4.3 curated result has been accepted for publication",
    "no publishable result exists anywhere",
    "This tree does not exist yet",
    "not yet produced",
)
# Claims that are legitimate for the closed P4.1 and P4.2 protocols but untrue
# in the P4.3 protocol, which owns the P4.3 status. They are banned only there.
FORBIDDEN_P43_PROTOCOL_CLAIMS = (
    "Independent audit: PENDING",
    "Production analysis: PENDING",
    "P4.3 = YES / NO",
    "P4.3 remains YES / NO / NO",
)

# Every project-level narrative must carry the complete, checkable provenance
# chain and the acceptance evidence, not a bare assertion that P4.3 is closed.
# The three commits, the manifest digest, the comparison method, and the
# independent verdict must all be readable from each narrative document.
REQUIRED_DOCUMENT_STATEMENTS = (
    (r"P4\.3", "P4.3 is described"),
    (re.escape(P43_ACCEPTANCE_RELATIVE_PATH), "the acceptance record is linked"),
    (re.escape(ACCEPTED_EXECUTION_COMMIT),
     "the experimental execution commit is recorded"),
    (re.escape(ACCEPTED_ANALYZER_COMMIT), "the analyzer commit is recorded"),
    (re.escape(ACCEPTED_CANDIDATE_COMMIT),
     "the accepted candidate commit is recorded"),
    (re.escape(ACCEPTED_MANIFEST_SHA256),
     "the accepted manifest digest is recorded"),
    (re.escape(ACCEPTED_AUDIT_VERDICT), "the independent audit verdict is recorded"),
    (re.escape(ACCEPTED_COMPARISON_METHOD), "the comparison method is recorded"),
    (r"no\s+publishable\s+(phase\s+4\s+)?result",
     "the P4.2/raw non-publication boundary is retained"),
)
REQUIRED_P43_STATEMENTS = {
    "PLAN.md": (
        "| P4.3 | Integrated analysis, documentation, audit | YES | YES | YES |",
        "P4.3 = YES / YES / YES",
        "Phase 4: CLOSED",
        "the experimental phase of the TFM is closed",
    ),
    "README.md": (
        "P4.3: CLOSED; independent audit: YES; production analysis: YES",
        "Phase 4: CLOSED",
        "the experimental phase of the TFM is closed",
    ),
    "results/README.md": (
        "P4.3 is closed",
        "the P4.3 curated result is accepted for publication",
    ),
    P43_PROTOCOL_RELATIVE_PATH: (
        "P4.3 = YES / YES / YES",
        "Independent audit: ACCEPT WITH NON-BLOCKING OBSERVATIONS",
        "Production analysis: RUN",
        "the experimental phase of the TFM is closed",
        "P4.2 itself produced no publishable Phase 4 result",
    ),
}
# The P4.3 protocol must freeze the population, the statistical policy, and the
# artifact inventory by name.
REQUIRED_P43_PROTOCOL_TOKENS = (
    "20260812T013848Z",
    "20260817T110330Z",
    "20260817T111310Z",
    "20260817T112011Z",
    "b08e45c2636a3ac17c94ad8b1368084914196d7a",
    "p43.v1",
    "sample standard deviation",
    "coefficient of variation",
    "one complete final campaign",
)

# The Python standard library modules P4.3 may import. It adds no external
# dependency and no version pin.
STDLIB_IMPORT_ALLOWLIST = frozenset({
    "__future__", "argparse", "csv", "errno", "hashlib", "importlib", "inspect", "io",
    "json", "linecache", "math", "os", "re", "shutil", "stat", "struct", "sys",
    "tempfile", "zlib", "pathlib",
})

# Calls that would execute something. P4.3 runs no child process at all, so a
# GPU, a container, Nsight Compute, and a campaign are all structurally
# unreachable from it.
FORBIDDEN_EXECUTION_PATTERNS = (
    r"\bsubprocess\b",
    r"\bos\.(system|popen|fork|exec[lv]\w*|spawn\w+)\b",
    r"\bpty\b",
    r"\brun_campaign\b",
    r"\brun_all\b",
    r"\bCampaign\s*\([^)]*\)\s*\.\s*run\b",
)
# Calls that would mutate evidence. None may appear in an evidence-mode
# function of the analyzer.
FORBIDDEN_EVIDENCE_MODE_CALLS = (
    r"\bos\.(remove|unlink|rmdir|replace|rename|mkdir|makedirs|chmod|chown|symlink|"
    r"link|truncate|write)\b",
    r"\bshutil\.",
    r"\bwrite_text\b", r"\bwrite_bytes\b",
    r"\bcreate_file_exclusive\b", r"\bwrite_file_exclusive\b",
    r"\bmkdir_component\b", r"\blink_no_clobber\b",
    r"\bwrite_next_manifest_revision\b", r"\bcreate_campaign_tree\b",
    r"\bopen\s*\([^)]*[\"']w",
)
# Statistical operations the frozen policy forbids outright.
FORBIDDEN_STATISTICAL_TOKENS = (
    r"\bbootstrap\w*\b",
    r"\bp_?value\b",
    r"\bsignific\w*\b",
    r"\bt_?test\b",
    r"\bwilcoxon\b",
    r"\bmannwhitney\b",
    r"\bclamp\w*\b",
    r"\boutlier\w*\b",
    r"\bwinner\b",
)

# The evidence class each metric must carry. The pre-remediation implementation
# classified the first five of these as directly measured quantities.
EXPECTED_METRIC_EVIDENCE = {
    "median_effective_gbps": "within_campaign_derived_estimate",
    "dram_read_ratio": "within_campaign_derived_estimate",
    "hbm_classification": "within_campaign_derived_estimate",
    "median_flops_per_cycle": "within_campaign_derived_estimate",
    "median_flops_per_cycle_per_sm": "within_campaign_derived_estimate",
    "tma_to_ldgsts_ratio": "within_campaign_derived_estimate",
    "speedup_2sm_over_1sm": "within_campaign_derived_estimate",
    "scaling_efficiency_percent": "within_campaign_derived_estimate",
    "earliest_tested_candidate_saturation_bif_kib": "within_campaign_derived_estimate",
    "earliest_tested_candidate_saturation_depth": "within_campaign_derived_estimate",
    "tflops": "within_campaign_derived_estimate",
    "throughput_ratio_vs_cublaslt": "within_campaign_derived_estimate",
    "gap_to_cublaslt_pct": "within_campaign_derived_estimate",
    "best_cutedsl_variant": "within_campaign_derived_estimate",
    "estimated_tflops_per_sm": "modeled_estimate",
    "estimated_device_equivalent_tflops": "modeled_estimate",
    "kernel_time_ms": "measured_source_observation",
    "within_campaign_cv_percent": "source_diagnostic",
    "within_campaign_stability_review": "source_diagnostic",
    "within_campaign_sample_count": "source_diagnostic",
    "within_campaign_iqr_flagged_count": "source_diagnostic",
    "within_campaign_flops_per_cycle_per_sm_cv_percent": "source_diagnostic",
    "within_campaign_flops_per_cycle_iqr_flagged_count": "source_diagnostic",
    "within_campaign_flops_per_cycle_per_sm_iqr_flagged_count": "source_diagnostic",
    "profile_sm_clock_status": "source_diagnostic",
    "profile_diagnostic_metrics_resolved_count": "source_diagnostic",
    "surprising_value_flag": "source_diagnostic",
    "diagnostic_flags": "source_diagnostic",
    "ncu_coverage": "source_diagnostic",
}

# The frozen size of that taxonomy. Protocol section 5.1, the analyzer's
# METRIC_EVIDENCE, and the table above must all be exactly this many metrics.
EXPECTED_METRIC_COUNT = 29

# The deterministic monospace text metrics the figures are laid out with,
# restated here as frozen expectations so that a collision can never be
# "resolved" by quietly shrinking the constant the layout reserves space with.
EXPECTED_CHAR_ADVANCE = 0.62
EXPECTED_BASE_FONT = 11.0
EXPECTED_TEXT_ASCENT = 0.80
EXPECTED_TEXT_DESCENT = 0.20

P43_CHECK_TARGET = "phase4-p43-check"
P43_ANALYZE_TARGET = "phase4-p43-analyze"
P43_VERIFY_TARGET = "phase4-p43-verify"

EXPECTED_ANALYZER_CLI_OPTIONS = frozenset({
    "-h", "--help", "--self-test", "--analyze", "--verify", "--campaign-root",
    "--pilot-campaign-id", "--final-campaign-id", "--output-root",
})
EXPECTED_CHECKER_CLI_OPTIONS = frozenset({"-h", "--help", "--self-test"})


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


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


def read_text(repo_root: Path, relative: str) -> str:
    return (repo_root / relative).read_text(encoding="utf-8")


# ===========================================================================
# Repository-contract checks.
# ===========================================================================


def _check_required_files(reporter: Reporter, repo_root: Path) -> bool:
    for relative in REQUIRED_P43_FILES:
        path = repo_root / relative
        reporter.check(f"{relative} exists as a regular, non-symlink file",
                       path.is_file() and not path.is_symlink(), str(path))
    for relative in (P43_ANALYZER_RELATIVE_PATH, P43_CHECKER_RELATIVE_PATH):
        reporter.check(f"{relative} is executable",
                       os.access(repo_root / relative, os.X_OK), relative)
    reporter.check(f"{P43_PROTOCOL_RELATIVE_PATH} is documentation, not code",
                   not re.search(r"^(import|from|def|class) ",
                                 read_text(repo_root, P43_PROTOCOL_RELATIVE_PATH),
                                 re.MULTILINE), "")
    return not reporter.failures


def _check_status_frontier(reporter: Reporter, documents: dict[str, str]) -> None:
    plan_text = documents["PLAN.md"]
    for line in EXPECTED_STATUS_LINES:
        reporter.check(f"PLAN.md records the truthful frontier row {line!r}",
                       line in plan_text, "")
    for wrong in FORBIDDEN_P43_STATUS_LINES:
        reporter.check(f"PLAN.md does not record the stale or untrue P4.3 status {wrong!r}",
                       wrong not in plan_text, "")
    for wrong in FORBIDDEN_CLOSED_UNIT_REGRESSIONS:
        reporter.check(f"PLAN.md does not regress a closed unit to {wrong!r}",
                       wrong not in plan_text, "")
    for line in CLOSED_STATUS_LINES:
        reporter.check(f"PLAN.md still records the closed {line!r}", line in plan_text, "")


def _check_truthful_claims(reporter: Reporter, documents: dict[str, str]) -> None:
    for relative, text in documents.items():
        for claim in FORBIDDEN_DOCUMENT_CLAIMS:
            reporter.check(f"{relative} does not claim {claim!r}",
                           claim.lower() not in text.lower(), "")
    for relative in ("PLAN.md", "README.md", "results/README.md",
                     P43_PROTOCOL_RELATIVE_PATH):
        text = documents[relative]
        for claim in FORBIDDEN_STALE_PRE_ACCEPTANCE_CLAIMS:
            reporter.check(
                f"{relative} does not retain the stale pre-acceptance claim {claim!r}",
                claim.lower() not in text.lower(), "")
    protocol_text = documents[P43_PROTOCOL_RELATIVE_PATH]
    for claim in FORBIDDEN_P43_PROTOCOL_CLAIMS:
        reporter.check(f"{P43_PROTOCOL_RELATIVE_PATH} does not claim {claim!r}",
                       claim.lower() not in protocol_text.lower(), "")
    for relative in ("PLAN.md", "README.md", "results/README.md",
                     P43_PROTOCOL_RELATIVE_PATH):
        text = documents[relative]
        for pattern, label in REQUIRED_DOCUMENT_STATEMENTS:
            reporter.check(f"{relative} states that {label}",
                           re.search(pattern, text, re.IGNORECASE) is not None, "")
    for relative, statements in REQUIRED_P43_STATEMENTS.items():
        for statement in statements:
            reporter.check(f"{relative} records {statement!r}",
                           statement in documents[relative], "")
    for token in REQUIRED_P43_PROTOCOL_TOKENS:
        reporter.check(f"{P43_PROTOCOL_RELATIVE_PATH} freezes {token!r}",
                       token in documents[P43_PROTOCOL_RELATIVE_PATH], "")
    reporter.check(f"{P42_PROTOCOL_RELATIVE_PATH} keeps its P4.2 closure record",
                   "P4.2 = YES / YES / YES" in documents[P42_PROTOCOL_RELATIVE_PATH], "")
    reporter.check(f"{P41_PROTOCOL_RELATIVE_PATH} keeps its P4.1 closure record",
                   "P4.1 = YES / YES / YES" in documents[P41_PROTOCOL_RELATIVE_PATH], "")


def _check_frozen_population(reporter: Reporter, analyzer, p42) -> None:
    reporter.check("the accepted pilot is P4.2's own accepted pilot",
                   analyzer.PILOT_CAMPAIGN_ID == p42.PILOT_CAMPAIGN_ID,
                   f"{analyzer.PILOT_CAMPAIGN_ID} != {p42.PILOT_CAMPAIGN_ID}")
    reporter.check("exactly three final campaigns form the statistical population",
                   analyzer.CAMPAIGN_COUNT == p42.REQUIRED_FINAL_CAMPAIGN_COUNT == 3,
                   str(analyzer.CAMPAIGN_COUNT))
    reporter.check("the three declared final campaign IDs are distinct and exclude the pilot",
                   len(set(analyzer.FINAL_CAMPAIGN_IDS)) == 3
                   and analyzer.PILOT_CAMPAIGN_ID not in analyzer.FINAL_CAMPAIGN_IDS,
                   str(analyzer.FINAL_CAMPAIGN_IDS))
    reporter.check("every declared campaign ID is a canonical campaign identifier",
                   all(re.fullmatch(r"[0-9]{8}T[0-9]{6}Z", campaign_id)
                       for campaign_id in
                       (analyzer.PILOT_CAMPAIGN_ID,) + analyzer.FINAL_CAMPAIGN_IDS), "")
    reporter.check("the frozen final execution commit is a full Git commit",
                   re.fullmatch(r"[0-9a-f]{40}", analyzer.FINAL_EXECUTION_COMMIT)
                   is not None, analyzer.FINAL_EXECUTION_COMMIT)
    reporter.check("the pilot is recorded only as excluded qualification provenance",
                   "excluded" in analyzer.PILOT_ROLE
                   and "qualification" in analyzer.PILOT_ROLE, analyzer.PILOT_ROLE)
    body = _strip(inspect.getsource(analyzer))
    for pattern, label in ((r"\bglob\b", "glob expansion"),
                           (r"\bst_mtime\b", "a modification time"),
                           (r"\biglob\b", "glob expansion"),
                           (r"\brglob\b", "recursive glob expansion"),
                           (r"\blatest\b", "a 'latest' selection")):
        reporter.check(f"the analyzer never discovers a campaign through {label}",
                       re.search(pattern, body) is None, pattern)


def _check_no_new_dependency(reporter: Reporter, repo_root: Path, sources: dict[str, str]
                             ) -> None:
    for relative, source in sources.items():
        imports = set(re.findall(r"^(?:import|from)\s+([A-Za-z_][A-Za-z0-9_.]*)",
                                 source, re.MULTILINE))
        foreign = sorted(name for name in imports
                         if name.split(".")[0] not in STDLIB_IMPORT_ALLOWLIST)
        reporter.check(f"{relative} imports only the Python standard library",
                       not foreign, str(foreign))
    for relative in ("VERSIONS.env", "PHASE3_VERSIONS.env"):
        text = read_text(repo_root, relative)
        reporter.check(f"P4.3 adds no key to {relative}",
                       not re.search(r"^(PHASE4_|P41_|P42_|P43_)", text, re.MULTILINE), "")


def _check_never_executes(reporter: Reporter, sources: dict[str, str], p41) -> None:
    for relative, source in sources.items():
        body = _strip(source)
        hits = [pattern for pattern in FORBIDDEN_EXECUTION_PATTERNS
                if re.search(pattern, body)]
        reporter.check(f"{relative} starts no child process, campaign, or runner",
                       not hits, str(hits))
        offenders = [name for name in p41.FORBIDDEN_CHECK_PATH_COMMANDS
                     if re.search(p41._COMMAND_POSITION.format(name=re.escape(name)), body)]
        reporter.check(f"{relative} invokes no container runtime, nvidia-smi, Nsight "
                       f"Compute, or CUDA compiler", not offenders, str(offenders))


def _check_read_only_evidence(reporter: Reporter, analyzer) -> None:
    for function in analyzer.EVIDENCE_MODE_FUNCTIONS:
        source = inspect.getsource(function)
        hits = [pattern for pattern in FORBIDDEN_EVIDENCE_MODE_CALLS
                if re.search(pattern, source)]
        reporter.check(f"the analyzer's {function.__name__}() never writes, repairs, "
                       f"resumes, or regenerates evidence", not hits, str(hits))
    reporter.check("the analyzer refuses every output root under immutable raw evidence",
                   ("results", "raw") in analyzer.FORBIDDEN_OUTPUT_PREFIXES
                   and ("results", "preflight") in analyzer.FORBIDDEN_OUTPUT_PREFIXES,
                   str(analyzer.FORBIDDEN_OUTPUT_PREFIXES))
    body = _strip(inspect.getsource(analyzer.publish_documents))
    reporter.check("publication is no-clobber: it only ever creates a file exclusively",
                   "write_file_exclusive" in body and "os.replace" not in body, "")


def _check_reuses_closed_units(reporter: Reporter, analyzer_source: str) -> None:
    code = _strip(analyzer_source)
    for name, owner in (("check_campaign_evidence", "P4.2"),
                        ("campaign_root_to_repo_root", "P4.2"),
                        ("EvidenceError", "P4.2"),
                        ("load_manifest_chain", "P4.1"),
                        ("resolve_campaign_tree", "P4.1"),
                        ("validate_campaign_id", "P4.1"),
                        ("open_dir_chain", "P4.1"),
                        ("read_file_nofollow", "P4.1"),
                        ("write_file_exclusive", "P4.1"),
                        ("sha256_bytes", "P4.1"),
                        ("canonical_json_bytes", "P4.1"),
                        ("COMPARABLE_EVIDENCE_FIELDS", "P4.1"),
                        ("GPU_IDENTITY_FIELDS", "P4.1"),
                        ("validate_serialized_output", "P3.5"),
                        ("EXPECTED_SHAPES", "P3.5"),
                        ("EXPECTED_CANDIDATE_ORDER", "P3.5")):
        reporter.check(f"the analyzer reuses {owner}'s own {name}",
                       re.search(rf"^{name}$", code, re.MULTILINE) is not None, "")
    for name in ("load_manifest_chain", "resolve_campaign_tree", "validate_manifest_document",
                 "verify_terminal_analysis_artifacts", "check_campaign_evidence",
                 "validate_serialized_output", "summarize_campaign"):
        reporter.check(f"the analyzer defines no copy of {name}",
                       re.search(rf"^(?:{name}\s*[:=]|def {name}\b|class {name}\b)",
                                 analyzer_source, re.MULTILINE) is None, "")


def _check_statistical_policy(reporter: Reporter, analyzer) -> None:
    """The frozen policy, exercised against the real analyzer rather than
    asserted in prose."""
    body = _strip(inspect.getsource(analyzer))
    hits = [pattern for pattern in FORBIDDEN_STATISTICAL_TOKENS if re.search(pattern, body)]
    reporter.check("the analyzer computes no bootstrap, p-value, significance claim, "
                   "outlier filter, clamp, or winner", not hits, str(hits))
    reporter.check("the strict cross-campaign review threshold is CV > 5.0%",
                   analyzer.CV_REVIEW_THRESHOLD_PERCENT == 5.0,
                   str(analyzer.CV_REVIEW_THRESHOLD_PERCENT))

    summary = analyzer.summarize_metric([1.0, 2.0, 3.0], metric="median_effective_gbps")
    required = {"stdev_sample", "mean", "median", "minimum", "maximum",
                "cross_campaign_cv_percent", "cross_campaign_cv_review_flag",
                "campaign_values", "campaign_count"}
    if not required <= set(summary):
        reporter.check("a summarized metric carries every frozen statistic field, with the "
                       "cross-campaign ones unambiguously named", False,
                       str(sorted(required - set(summary))))
        return
    reporter.check("the sample standard deviation uses the n-1 denominator",
                   abs(summary["stdev_sample"] - 1.0) < 1e-12, str(summary["stdev_sample"]))
    reporter.check("mean, median, minimum, and maximum are the plain descriptive values",
                   (summary["mean"], summary["median"], summary["minimum"],
                    summary["maximum"]) == (2.0, 2.0, 1.0, 3.0), str(summary))
    reporter.check("the cross-campaign coefficient of variation is 100 x stdev / mean",
                   abs(summary["cross_campaign_cv_percent"] - 50.0) < 1e-12,
                   str(summary["cross_campaign_cv_percent"]))
    reporter.check("a cross-campaign coefficient of variation above the threshold flags "
                   "for review and keeps every campaign value",
                   summary["cross_campaign_cv_review_flag"] == "REVIEW"
                   and len(summary["campaign_values"]) == 3, str(summary))
    reporter.check("the cross-campaign statistic is named so it can never be read as a "
                   "within-campaign one",
                   "cv_percent" not in summary and "cv_review_flag" not in summary,
                   str(sorted(summary)))

    def rejects(label: str, call) -> None:
        try:
            call()
        except analyzer.P43Error:
            reporter.check(label, True)
        else:
            reporter.check(label, False, "no P43Error was raised")

    rejects("pooling a campaign's 30 internal repetitions is structurally rejected",
            lambda: analyzer.summarize_metric([1.0] * 90, metric="median_effective_gbps"))
    rejects("a sample of anything other than three campaigns is rejected",
            lambda: analyzer.summarize_metric([1.0, 2.0], metric="median_effective_gbps"))
    rejects("a non-finite campaign value is rejected",
            lambda: analyzer.summarize_metric([1.0, float("nan"), 3.0],
                                              metric="median_effective_gbps"))
    signed = analyzer.summarize_metric([-5.0, 0.0, 5.0], metric="gap_to_cublaslt_pct")
    reporter.check("no coefficient of variation is computed for the signed GEMM gap",
                   signed["cross_campaign_cv_percent"] is None
                   and signed["cross_campaign_cv_review_flag"] == analyzer.NOT_APPLICABLE,
                   str(signed))
    reporter.check("a negative GEMM gap is preserved without clamping",
                   signed["minimum"] == -5.0, str(signed))
    reporter.check("gap_to_cublaslt_pct is declared signed or zero-centred",
                   "gap_to_cublaslt_pct" in analyzer.SIGNED_OR_ZERO_CENTRED_METRICS, "")
    zero = analyzer.summarize_metric([0.0, 0.0, 0.0],
                                     metric="throughput_ratio_vs_cublaslt")
    reporter.check("a zero denominator never yields a coefficient of variation",
                   zero["cross_campaign_cv_percent"] is None, str(zero))
    agree = analyzer.consensus(["a", "a", "a"], label="x")
    disagree = analyzer.consensus(["a", "b", "a"], label="x")
    reporter.check("a consensus is reported only when all three campaigns agree, and a "
                   "disagreement keeps all three results",
                   agree["consensus"] == "a" and disagree["consensus"] is None
                   and disagree["campaign_values"] == ["a", "b", "a"], str(disagree))


def _check_artifact_contract(reporter: Reporter, analyzer, protocol: str) -> None:
    inventory = list(analyzer.ARTIFACT_RELATIVE_PATHS)
    expected = [
        "memory_paths.csv", "umma_throughput.csv", "gemm_comparison.csv",
        "integrated_summary.json", "report.md", "figures/memory_paths.svg",
        "figures/umma_throughput.svg", "figures/gemm_comparison.svg",
        "analysis_manifest.json",
    ]
    reporter.check("the analyzer's artifact inventory is exactly the frozen tree",
                   inventory == expected, str(inventory))
    for relative in inventory:
        reporter.check(f"{P43_PROTOCOL_RELATIVE_PATH} freezes the artifact {relative!r}",
                       relative in protocol, "")
    reporter.check("the schema version is p43.v1",
                   analyzer.SCHEMA_VERSION == "p43.v1", analyzer.SCHEMA_VERSION)
    reporter.check("every output records publishable=false",
                   analyzer.PUBLISHABLE is False
                   and "publishable=false" in analyzer.PUBLICATION_STATUS,
                   analyzer.PUBLICATION_STATUS)
    reporter.check(f"{P43_PROTOCOL_RELATIVE_PATH} names the default curated output tree",
                   analyzer.DEFAULT_OUTPUT_ROOT_REL in protocol,
                   analyzer.DEFAULT_OUTPUT_ROOT_REL)

    # A complete synthetic analysis: deterministic bytes, no pilot in the data,
    # and exactly the frozen inventory.
    orchestrator = load_module(DEFAULT_REPO_ROOT / ORCHESTRATOR_RELATIVE_PATH,
                               "_p43_check_orchestrator")
    try:
        first = analyzer.build_documents(orchestrator, analyzer._StubP35(),
                                         analyzer._fixture_records(),
                                         analyzer._fixture_provenance())
        second = analyzer.build_documents(orchestrator, analyzer._StubP35(),
                                          analyzer._fixture_records(),
                                          analyzer._fixture_provenance())
    except (analyzer.P43Error, KeyError, TypeError, AttributeError, ValueError) as exc:
        # A broken or tampered analyzer must be reported as a failure here, not
        # escape as a traceback that hides every remaining check.
        reporter.check("a complete synthetic analysis runs end to end", False,
                       f"{type(exc).__name__}: {exc}")
        return
    reporter.check("the analysis is deterministic: identical evidence gives identical bytes",
                   first == second, "")
    reporter.check("the produced documents are exactly the frozen inventory, in order",
                   [relative for relative, _ in first] == expected, "")
    payloads = dict(first)
    for relative in ("memory_paths.csv", "umma_throughput.csv", "gemm_comparison.csv"):
        reporter.check(f"{relative} never carries the excluded pilot as data",
                       analyzer.PILOT_CAMPAIGN_ID not in payloads[relative].decode("utf-8"),
                       "")
    for relative in ("figures/memory_paths.svg", "figures/umma_throughput.svg",
                     "figures/gemm_comparison.svg"):
        svg = payloads[relative].decode("utf-8")
        footer_lines = re.findall(r"<tspan\s+[^>]*>(.*?)</tspan>", svg)
        reporter.check(f"{relative} has a visible title and focused-scale subtitle",
                       '<text x="24" y="25"' in svg
                       and "Focused y-scale" in svg, "")
        reporter.check(f"{relative} reserves a multiline footer inside the 1080x480 canvas",
                       'viewBox="0 0 1080 480"' in svg
                       and len(footer_lines) >= 2
                       and all(len(line) <= analyzer._SVG_FOOTER_WRAP
                               for line in footer_lines), str(footer_lines))
        reporter.check(f"{relative} does not duplicate mutable publication progress",
                       analyzer.PUBLICATION_STATE not in svg
                       and "analysis_code_commit" not in svg
                       and "time-dependent audit" not in svg, "")
        # An independent spatial re-derivation of finding M1, computed here from
        # the emitted coordinates rather than by calling the analyzer's own
        # geometry helper, so that a defect in that helper cannot hide a defect
        # in the layout. Each panel's y-axis decorations must lie strictly
        # inside the clear gap between the preceding plot rectangle and its own.
        panels = sorted((float(match.group(1)),
                         float(match.group(1)) + float(match.group(2)))
                        for match in re.finditer(
                            r'<rect x="([\d.]+)" y="[\d.]+" width="([\d.]+)" '
                            r'height="[\d.]+" fill="none"', svg))
        ticks = [(float(x) - len(label) * EXPECTED_BASE_FONT * EXPECTED_CHAR_ADVANCE,
                  float(x))
                 for x, label in re.findall(
                     r'<text x="([\d.]+)" y="[\d.]+" text-anchor="end">([^<]*)</text>',
                     svg)]
        titles = [(float(x) - EXPECTED_TEXT_ASCENT * EXPECTED_BASE_FONT,
                   float(x) + EXPECTED_TEXT_DESCENT * EXPECTED_BASE_FONT)
                  for x in re.findall(
                      r'<text x="([\d.]+)" y="[\d.]+" text-anchor="middle" '
                      r'transform="rotate\(-90 ', svg)]
        reporter.check(f"{relative} emits one y-axis title and five tick labels per panel",
                       len(panels) >= 3 and len(titles) == len(panels)
                       and len(ticks) == 5 * len(panels),
                       f"panels={len(panels)} titles={len(titles)} ticks={len(ticks)}")
        intruding = [decoration for decoration in ticks + titles
                     for index, (left, right) in enumerate(panels)
                     if decoration[0] < right - 0.05 and decoration[1] > left + 0.05]
        reporter.check(f"{relative} draws no y-axis decoration inside any plot rectangle",
                       not intruding, str(intruding[:3]))
        outside = [decoration for decoration in ticks + titles
                   if decoration[0] < 0.0 or decoration[1] > float(analyzer._SVG_WIDTH)]
        reporter.check(f"{relative} clips no y-axis decoration at the canvas edge",
                       not outside, str(outside[:3]))
    reporter.check("the layout still reserves space with the frozen text metrics",
                   (analyzer._SVG_CHAR_ADVANCE, analyzer._SVG_BASE_FONT,
                    analyzer._SVG_TEXT_ASCENT, analyzer._SVG_TEXT_DESCENT)
                   == (EXPECTED_CHAR_ADVANCE, EXPECTED_BASE_FONT,
                       EXPECTED_TEXT_ASCENT, EXPECTED_TEXT_DESCENT),
                   f"{analyzer._SVG_CHAR_ADVANCE} {analyzer._SVG_BASE_FONT}")
    for relative in ("figures/memory_paths.svg", "figures/umma_throughput.svg",
                     "figures/gemm_comparison.svg"):
        reporter.check(f"{relative} passes the analyzer's own collision regression",
                       not analyzer.svg_geometry_findings(
                           payloads[relative].decode("utf-8")),
                       "; ".join(analyzer.svg_geometry_findings(
                           payloads[relative].decode("utf-8")))[:300])
    gemm_svg = payloads["figures/gemm_comparison.svg"].decode("utf-8")
    reporter.check("the GEMM SVG labels all four candidates directly",
                   all(label in gemm_svg for label in
                       ("NP1=nonpersistent_1cta", "P1=persistent_1cta",
                        "P2=persistent_2cta", "BL=cuBLASLt")), "")
    manifest = json.loads(payloads["analysis_manifest.json"].decode("utf-8"))
    reporter.check("analysis_manifest.json records all three final campaign IDs",
                   manifest["final_campaign_ids"] == list(analyzer.FINAL_CAMPAIGN_IDS), "")
    reporter.check("analysis_manifest.json records the pilot only as excluded provenance",
                   manifest["pilot_campaign_id_excluded"] == analyzer.PILOT_CAMPAIGN_ID
                   and analyzer.PILOT_CAMPAIGN_ID not in manifest["final_campaign_ids"], "")
    reporter.check("analysis_manifest.json pins a SHA-256 for every other output artifact",
                   set(manifest["artifact_sha256"])
                   == set(expected) - {analyzer.MANIFEST_RELATIVE_PATH}
                   and all(re.fullmatch(r"[0-9a-f]{64}", value)
                           for value in manifest["artifact_sha256"].values()),
                   str(sorted(manifest["artifact_sha256"])))
    reporter.check("analysis_manifest.json records the final execution commit and one GPU "
                   "identity",
                   manifest["final_execution_commit"] == analyzer.FINAL_EXECUTION_COMMIT
                   and sorted(manifest["gpu"]) == ["compute_capability", "driver_version",
                                                   "name", "uuid"], "")
    reporter.check("analysis_manifest.json records every source path and hash",
                   manifest["sources"] and all(
                       set(entry) >= {"campaign_id", "repo_relative_path", "sha256"}
                       and not entry["repo_relative_path"].startswith("/")
                       for entry in manifest["sources"]), "")
    blob = b"\n".join(payload for _, payload in first).decode("utf-8", errors="replace")
    for leak in (str(Path.home()), "/home/", "/Users/", "USER=", "HOSTNAME=", "SSH_"):
        reporter.check(f"no output artifact leaks {leak!r}", leak not in blob, "")


def _protocol_metric_table(protocol: str) -> dict[str, str]:
    """The frozen section 5.1 classification table, read from the protocol itself.

    Parsed rather than restated so that the bidirectional equality assertion
    below compares the document a reader actually sees against the
    classifications the analyzer actually applies (second independent audit,
    finding M5)."""
    marker = "### 5.1 The frozen classification"
    if marker not in protocol:
        return {}
    start = protocol.index(marker)
    opening = protocol.index("```text", start) + len("```text")
    closing = protocol.index("```", opening)
    table: dict[str, str] = {}
    for line in protocol[opening:closing].splitlines():
        parts = line.split()
        if len(parts) == 2:
            table[parts[0]] = parts[1]
    return table


def _check_evidence_taxonomy(reporter: Reporter, analyzer, protocol: str) -> None:
    """3.1 -- every reported quantity is classified, and no derived, modeled, or
    cross-campaign quantity is presented as a direct measurement."""
    classes = set(analyzer.EVIDENCE_CLASSES)
    reporter.check("the analyzer declares the frozen evidence taxonomy",
                   classes == {analyzer.EVIDENCE_MEASURED,
                               analyzer.EVIDENCE_WITHIN_CAMPAIGN,
                               analyzer.EVIDENCE_CROSS_CAMPAIGN,
                               analyzer.EVIDENCE_MODELED,
                               analyzer.EVIDENCE_INTERPRETATION,
                               analyzer.EVIDENCE_UNAVAILABLE,
                               analyzer.EVIDENCE_DIAGNOSTIC},
                   str(sorted(classes)))
    for name in sorted(classes):
        reporter.check(f"{P43_PROTOCOL_RELATIVE_PATH} freezes the evidence class {name!r}",
                       name in protocol, "")
    # The frozen table and the implementation must agree exactly, in both
    # directions. Before this assertion existed the protocol listed 23
    # classifications while the analyzer applied 29, and nothing noticed.
    table = _protocol_metric_table(protocol)
    implemented = {metric: entry[0]
                   for metric, entry in analyzer.METRIC_EVIDENCE.items()}
    reporter.check(f"{P43_PROTOCOL_RELATIVE_PATH} section 5.1 classifies exactly the "
                   f"{EXPECTED_METRIC_COUNT} metrics the analyzer classifies",
                   len(table) == len(implemented) == EXPECTED_METRIC_COUNT,
                   f"protocol={len(table)} implementation={len(implemented)} "
                   f"expected={EXPECTED_METRIC_COUNT}")
    reporter.check("no metric is classified in the implementation but missing from the "
                   "frozen protocol table",
                   not set(implemented) - set(table),
                   str(sorted(set(implemented) - set(table))))
    reporter.check("no metric is listed in the frozen protocol table but absent from the "
                   "implementation",
                   not set(table) - set(implemented),
                   str(sorted(set(table) - set(implemented))))
    mismatched = {metric: (table[metric], implemented[metric])
                  for metric in set(table) & set(implemented)
                  if table[metric] != implemented[metric]}
    reporter.check("every metric carries the same evidence class in the protocol and in "
                   "the implementation", not mismatched, str(mismatched))
    reporter.check("the checker's own frozen expectation is the same table",
                   EXPECTED_METRIC_EVIDENCE == table == implemented,
                   str(sorted(set(EXPECTED_METRIC_EVIDENCE) ^ set(implemented))))
    for metric, expected in EXPECTED_METRIC_EVIDENCE.items():
        declared = analyzer.METRIC_EVIDENCE.get(metric)
        reporter.check(f"{metric} is classified {expected}, not as a direct measurement",
                       declared is not None and declared[0] == expected,
                       str(declared))
    reporter.check("only the closed protocols' own measured inputs are called measured",
                   {metric for metric, entry in analyzer.METRIC_EVIDENCE.items()
                    if entry[0] == analyzer.EVIDENCE_MEASURED} == {"kernel_time_ms"},
                   str(sorted(metric for metric, entry in analyzer.METRIC_EVIDENCE.items()
                              if entry[0] == analyzer.EVIDENCE_MEASURED)))
    reporter.check("effective GB/s is documented as a timing-derived effective rate and "
                   "explicitly not HBM/DRAM bandwidth",
                   "not hbm/dram bandwidth"
                   in analyzer.METRIC_EVIDENCE["median_effective_gbps"][1].lower(), "")
    reporter.check("FLOP/cycle is documented as derived from a validated operation count "
                   "and measured cycles",
                   "operation count"
                   in analyzer.METRIC_EVIDENCE["median_flops_per_cycle"][1]
                   and "measured %clock64"
                   in analyzer.METRIC_EVIDENCE["median_flops_per_cycle"][1], "")
    reporter.check("dram_read_ratio and hbm_classification are documented as derived from "
                   "profiler evidence rather than as raw counters",
                   all("not a raw profiler counter"
                       in analyzer.METRIC_EVIDENCE[metric][1]
                       for metric in ("dram_read_ratio", "hbm_classification")), "")
    reporter.check("every metric emitted by the analyzer carries a classification and a "
                   "basis sentence",
                   all(entry[0] in classes and len(entry[1]) > 40
                       for entry in analyzer.METRIC_EVIDENCE.values()), "")
    try:
        analyzer.evidence_class_for("an_undeclared_metric")
    except analyzer.P43Error:
        reporter.check("an unclassified quantity can never be emitted", True)
    else:
        reporter.check("an unclassified quantity can never be emitted", False, "accepted")
    reporter.check("the analyzer states that the memory benchmark is a dedicated streaming "
                   "microbenchmark rather than GEMM traffic",
                   "streaming" in analyzer.METRIC_EVIDENCE["median_effective_gbps"][1]
                   or "streaming" in inspect.getsource(analyzer.build_interpretation), "")
    reporter.check("the analyzer names the min-max range a whisker, never a bar",
                   "whisker" in analyzer.WHISKER_CAPTION
                   and re.search(r"\bbars?\b", analyzer.WHISKER_CAPTION) is None,
                   analyzer.WHISKER_CAPTION)
    reporter.check("the figure caption states that it summarizes exactly three "
                   "campaign-level values",
                   f"exactly {analyzer.CAMPAIGN_COUNT} campaign-level values"
                   in analyzer.WHISKER_CAPTION, analyzer.WHISKER_CAPTION)


def _check_preserved_diagnostics(reporter: Reporter, analyzer, orchestrator) -> None:
    """3.2 -- diagnostics that are parsed must reach the curated artifacts, in
    the frozen campaign order, and must never be conflated with the
    cross-campaign statistics."""
    records = analyzer._fixture_records(**{
        analyzer.FINAL_CAMPAIGN_IDS[1]: {
            "ncu_diagnostic_flags": "READ_AMPLIFICATION",
            "memory_stability_review": "REVIEW", "memory_cv_percent": 9.5,
            "memory_iqr_flagged_count": 2,
            "umma_stability_review": "REVIEW",
            "umma_per_sm_cv_percent": 8.5,
            "umma_flops_iqr_flagged_count": 1,
            "umma_per_sm_iqr_flagged_count": 3,
            "profile_diagnostic_metrics_resolved_count": 4,
            "surprising_value_flag": "True"},
    })
    documents = dict(analyzer.build_documents(orchestrator, analyzer._StubP35(), records,
                                              analyzer._fixture_provenance()))
    memory = documents["memory_paths.csv"].decode("utf-8")
    umma = documents["umma_throughput.csv"].decode("utf-8")
    summary = documents["integrated_summary.json"].decode("utf-8")
    report = documents["report.md"].decode("utf-8")

    reporter.check("a parsed NCU diagnostic flag is never dropped between parsing and "
                   "publication",
                   all("READ_AMPLIFICATION" in text for text in (memory, summary, report)),
                   "")
    for metric, table in (("within_campaign_sample_count", memory),
                          ("within_campaign_cv_percent", memory),
                          ("within_campaign_stability_review", memory),
                          ("within_campaign_iqr_flagged_count", memory),
                          ("hbm_classification", memory),
                          ("diagnostic_flags", memory),
                          ("ncu_coverage", memory),
                          ("within_campaign_sample_count", umma),
                          ("within_campaign_cv_percent", umma),
                          ("within_campaign_stability_review", umma),
                          ("within_campaign_flops_per_cycle_per_sm_cv_percent", umma),
                          ("within_campaign_flops_per_cycle_iqr_flagged_count", umma),
                          ("within_campaign_flops_per_cycle_per_sm_iqr_flagged_count", umma),
                          ("profile_sm_clock_status", umma),
                          ("profile_diagnostic_metrics_resolved_count", umma),
                          ("surprising_value_flag", umma)):
        reporter.check(f"{metric} survives into the curated CSV representation",
                       f",{metric}," in table, metric)
        reporter.check(f"{metric} survives into the machine-readable JSON",
                       f'"{metric}"' in summary, metric)
    # The values themselves must survive, in the right campaign position: a row
    # that keeps the metric name but replaces every value with not_applicable
    # has still lost the diagnostic.
    for table, name, metric, expected in (
            (memory, "memory_paths.csv", "diagnostic_flags",
             (analyzer.NOT_APPLICABLE, "READ_AMPLIFICATION", analyzer.NOT_APPLICABLE)),
            (memory, "memory_paths.csv", "within_campaign_stability_review",
             ("ok", "REVIEW", "ok")),
            (memory, "memory_paths.csv", "within_campaign_cv_percent",
             ("0.1", "9.5", "0.1")),
            (memory, "memory_paths.csv", "within_campaign_iqr_flagged_count",
             ("0", "2", "0")),
            (umma, "umma_throughput.csv", "within_campaign_stability_review",
             ("ok", "REVIEW", "ok")),
            (umma, "umma_throughput.csv",
             "within_campaign_flops_per_cycle_per_sm_cv_percent",
             ("0.01", "8.5", "0.01")),
            (umma, "umma_throughput.csv",
             "within_campaign_flops_per_cycle_iqr_flagged_count", ("0", "1", "0")),
            (umma, "umma_throughput.csv",
             "within_campaign_flops_per_cycle_per_sm_iqr_flagged_count",
             ("0", "3", "0")),
            (umma, "umma_throughput.csv", "profile_sm_clock_status",
             ("OK", "OK", "OK")),
            (umma, "umma_throughput.csv",
             "profile_diagnostic_metrics_resolved_count", ("2", "4", "2")),
            (umma, "umma_throughput.csv", "surprising_value_flag",
             ("False", "True", "False"))):
        rows = [row for row in csv.DictReader(io.StringIO(table))
                if row["metric"] == metric]
        reporter.check(f"{name}: every {metric} value survives in the frozen campaign "
                       f"order",
                       rows and all((row["campaign_1_value"], row["campaign_2_value"],
                                     row["campaign_3_value"]) == expected
                                    for row in rows),
                       str(rows[:1]))
        reporter.check(f"{name}: {metric} never carries a cross-campaign statistic",
                       all(row["cross_campaign_cv_percent"] == analyzer.NOT_APPLICABLE
                           and row["mean"] == analyzer.NOT_APPLICABLE for row in rows), "")
    reporter.check("the twelve unprofiled configurations are marked not_profiled and their "
                   "HBM traffic is declared unavailable",
                   analyzer.NCU_NOT_PROFILED in memory
                   and "unavailable from the collected evidence" in summary, "")
    reporter.check("within-campaign and cross-campaign variability carry different names",
                   "within_campaign_cv_percent" in summary
                   and "cross_campaign_cv_percent" in summary
                   and "within_campaign_stability_review" in summary
                   and "cross_campaign_cv_review_flag" in summary, "")
    reporter.check("a cross-campaign review flag never excludes a campaign or changes a "
                   "value",
                   "never excludes a campaign" in summary, "")
    reporter.check("the report summarizes the source diagnostic warnings and the review "
                   "conditions",
                   "Source diagnostic warnings" in report
                   and "Cross-campaign variability review conditions" in report, "")


def _check_metadata_ownership(reporter: Reporter, analyzer, orchestrator,
                              documents: list, repo_root: Path) -> None:
    """3.3 -- the manifest is the authoritative envelope, and the documentation
    says exactly that instead of claiming every file embeds everything."""
    payloads = dict(documents)
    manifest = json.loads(payloads[analyzer.MANIFEST_RELATIVE_PATH].decode("utf-8"))
    siblings = [relative for relative in analyzer.ARTIFACT_RELATIVE_PATHS
                if relative != analyzer.MANIFEST_RELATIVE_PATH]
    reporter.check("exactly nine artifacts are generated",
                   len(documents) == 9, str(len(documents)))
    reporter.check("the manifest binds all eight siblings by path and SHA-256",
                   sorted(manifest["artifact_sha256"]) == sorted(siblings), "")
    reporter.check("every recomputed sibling hash matches the manifest",
                   all(manifest["artifact_sha256"][relative]
                       == orchestrator.sha256_bytes(payloads[relative])
                       for relative in siblings), "")
    reporter.check("the manifest states precisely that it cannot contain its own byte hash",
                   manifest["self_hash"]["value"] == analyzer.NOT_APPLICABLE
                   and "cannot contain its own byte hash"
                   in manifest["self_hash"]["reason"], "")
    reporter.check("the manifest maps every campaign value column to its campaign ID",
                   manifest["campaign_value_column_map"]
                   == {f"campaign_{index + 1}_value": campaign_id
                       for index, campaign_id
                       in enumerate(analyzer.FINAL_CAMPAIGN_IDS)}, "")
    reporter.check("the manifest records the analysis-code commit and its clean-worktree "
                   "verification, distinct from the final execution commit",
                   manifest["analysis_code_commit"] != manifest["final_execution_commit"]
                   and manifest["analysis_code_worktree_clean"] is True, "")
    reporter.check("the manifest records the candidate publication state",
                   manifest["publishable"] is False
                   and manifest["publication_state"] == analyzer.PUBLICATION_STATE, "")
    reporter.check("the reader-facing summary and report repeat the invariant candidate "
                   "state without becoming authoritative envelopes",
                   analyzer.PUBLICATION_STATE
                   in payloads["integrated_summary.json"].decode("utf-8")
                   and analyzer.PUBLICATION_STATE
                   in payloads["report.md"].decode("utf-8"), "")
    for relative in siblings:
        if relative.endswith((".csv", ".svg")):
            text = payloads[relative].decode("utf-8")
            reporter.check(f"{relative} is a data or visual artifact, not a duplicated "
                           f"provenance envelope",
                           all(campaign_id not in text
                               for campaign_id in analyzer.FINAL_CAMPAIGN_IDS)
                           and analyzer.PUBLICATION_STATE not in text
                           and "analysis_code_commit" not in text, relative)
    results_readme = read_text(repo_root, "results/README.md")
    reporter.check("results/README.md no longer claims that every file embeds the "
                   "campaigns, commit, provenance, and publishable flag",
                   "Every file carries schema version" not in results_readme, "")
    for document in ("results/README.md", P43_PROTOCOL_RELATIVE_PATH):
        text = read_text(repo_root, document)
        reporter.check(f"{document} states that a detached CSV or SVG is not a standalone "
                       f"provenance envelope",
                       "not a standalone provenance envelope" in text
                       or "not standalone provenance envelopes" in text, "")
        reporter.check(f"{document} names analysis_manifest.json the authoritative binding",
                       "authoritative" in text and "analysis_manifest.json" in text, "")


def _check_output_containment(reporter: Reporter, analyzer, orchestrator) -> None:
    """3.4 -- production output is limited to the one logical destination and is
    reached only through descriptor-anchored, symlink-rejecting traversal."""
    source = inspect.getsource(analyzer.open_output_tree)
    component = inspect.getsource(analyzer._open_output_component)
    # Scan executable code only: the docstrings name these flags in order to
    # explain them, and a docstring is not a guarantee.
    executable_tree = _strip(source)
    executable_component = _strip(component)
    for token in ("O_DIRECTORY", "O_NOFOLLOW"):
        reporter.check(f"the repository root is opened with {token}",
                       token in executable_tree, token)
    for token in ("O_DIRECTORY", "O_NOFOLLOW", "dir_fd"):
        reporter.check(f"every output path component is opened with {token}",
                       token in executable_component, token)
    reporter.check("a missing output directory is created relative to the validated "
                   "parent descriptor",
                   re.search(r"os\.mkdir\([^)]*dir_fd=parent_fd", component) is not None,
                   "")
    reporter.check("the output root components are frozen to results/phase4",
                   analyzer.OUTPUT_ROOT_COMPONENTS == ("results", "phase4"),
                   str(analyzer.OUTPUT_ROOT_COMPONENTS))
    executable = _strip(source) + _strip(inspect.getsource(analyzer._open_output_component))
    reporter.check("the safety decision is not a resolve(), abspath(), or string prefix",
                   not re.search(r"\bresolve\b|\babspath\b|\bstartswith\b|"
                                 r"\brelative_to\b", executable), executable[:200])
    reporter.check("the exact-tree verification never follows a symlink",
                   "follow_symlinks=False"
                   in inspect.getsource(analyzer._scan_output_tree), "")
    reporter.check("publication creates artifacts exclusively and never replaces one",
                   "write_file_exclusive" in inspect.getsource(analyzer.publish_documents)
                   and "os.replace" not in inspect.getsource(analyzer.publish_documents),
                   "")
    publication = inspect.getsource(analyzer.publish_documents)
    reporter.check("a partial retry validates every existing byte and unexpected path "
                   "before creating a missing artifact",
                   "assert_output_tree_compatible(tree)" in publication
                   and publication.index("for relative, payload in documents")
                   < publication.index("for relative, payload, directory_fd, name in missing"),
                   "")
    with tempfile.TemporaryDirectory(prefix="p43-check-containment-") as temporary:
        root = Path(temporary)
        outside = root / "outside"
        outside.mkdir()
        documents = analyzer.build_documents(orchestrator, analyzer._StubP35(),
                                             analyzer._fixture_records(),
                                             analyzer._fixture_provenance())

        def escapes(name: str, build) -> None:
            repo = root / name
            repo.mkdir()
            build(repo)
            try:
                analyzer.resolve_output_root(repo / analyzer.DEFAULT_OUTPUT_ROOT_REL, repo)
                tree = analyzer.open_output_tree(repo, create=True)
                try:
                    analyzer.publish_documents(orchestrator, tree, documents, write=True)
                finally:
                    analyzer.close_output_tree(tree)
            except analyzer.P43Error:
                reporter.check(f"the {name} escape is refused", True)
                return
            reporter.check(f"the {name} escape is refused", False, "it was accepted")

        def link_results(repo: Path) -> None:
            os.symlink(outside, repo / "results")

        def link_phase4(repo: Path) -> None:
            (repo / "results").mkdir()
            os.symlink(outside, repo / "results" / "phase4")

        def link_figures(repo: Path) -> None:
            (repo / "results" / "phase4").mkdir(parents=True)
            os.symlink(outside, repo / "results" / "phase4" / "figures")

        escapes("ancestor-results-symlink", link_results)
        escapes("phase4-symlink", link_phase4)
        escapes("figures-symlink", link_figures)
        reporter.check("no P4.3 artifact was written outside the fixture repositories",
                       not os.listdir(outside), str(os.listdir(outside)))
        arbitrary = root / "arbitrary"
        arbitrary.mkdir()
        for candidate in ("results/elsewhere", "analysis", "results/raw/phase4",
                          "results/preflight"):
            try:
                analyzer.resolve_output_root(arbitrary / candidate, arbitrary)
            except analyzer.P43Error:
                reporter.check(f"an output root at {candidate} is refused", True)
            else:
                reporter.check(f"an output root at {candidate} is refused", False,
                               "accepted")


def _check_candidate_and_acceptance(reporter: Reporter, analyzer, orchestrator,
                                    documents: list, repo_root: Path,
                                    protocol: str) -> None:
    """3.5 -- an immutable candidate plus the external acceptance attestation
    that is now the sole source of publication authority.

    The candidate model itself is unchanged by acceptance: the bytes still
    record `publishable=false` and the invariant candidate publication state,
    and nothing promotes, overwrites, or deletes them. What changed is that the
    separate, hash-bound attestation now exists; `_check_acceptance_record()`
    validates the real file against the real bundle.
    """
    payloads = dict(documents)
    reporter.check("candidate artifacts record publishable=false and the candidate "
                   "publication state",
                   analyzer.PUBLISHABLE is False
                   and analyzer.PUBLICATION_STATE
                   == "immutable_candidate_requires_external_attestation", "")
    reporter.check("the candidate publication state appears in the bundle",
                   analyzer.PUBLICATION_STATE
                   in payloads["integrated_summary.json"].decode("utf-8")
                   and analyzer.PUBLICATION_STATE
                   in payloads[analyzer.MANIFEST_RELATIVE_PATH].decode("utf-8"), "")
    report = payloads["report.md"].decode("utf-8")
    manifest = json.loads(payloads[analyzer.MANIFEST_RELATIVE_PATH].decode("utf-8"))
    reporter.check("immutable candidate bytes make no time-dependent audit, production, "
                   "review, or attestation-progress assertion",
                   "are all pending" not in analyzer.PUBLICATION_STATUS
                   and "No step after the production" not in report
                   and "P4.3 | Integrated analysis" not in report
                   and "exists" not in manifest["acceptance"], "")
    publication = inspect.getsource(analyzer.publish_documents)
    reporter.check("the analyzer has no promotion, overwrite, or delete route for a "
                   "candidate",
                   analyzer.PUBLISHABLE is False
                   and "refusing to overwrite" in publication
                   and not re.search(r"\bos\.(replace|rename|remove|unlink|truncate)\b|"
                                     r"\bshutil\.", _strip(publication)), "")
    reporter.check("the acceptance attestation is not part of the nine-artifact inventory",
                   analyzer.ACCEPTANCE_RELATIVE_PATH
                   not in analyzer.ARTIFACT_RELATIVE_PATHS, "")
    reporter.check("the analyzer never writes the acceptance attestation",
                   not re.search(
                       r"P4_3_ACCEPTANCE\.json[^\n]*(write|create|open)",
                       inspect.getsource(analyzer)), "")
    for token in (analyzer.ACCEPTANCE_SCHEMA_VERSION, analyzer.ACCEPTANCE_RELATIVE_PATH,
                  analyzer.ACCEPTANCE_STATUS_ACCEPTED,
                  "analysis_manifest_sha256", "accepted_for_publication"):
        reporter.check(f"{P43_PROTOCOL_RELATIVE_PATH} freezes the acceptance field {token!r}",
                       token in protocol, "")
    for step in analyzer.ACCEPTANCE_LIFECYCLE:
        reporter.check(f"{P43_PROTOCOL_RELATIVE_PATH} documents the lifecycle step {step!r}",
                       step in protocol, "")

    manifest_sha256 = orchestrator.sha256_bytes(
        payloads[analyzer.MANIFEST_RELATIVE_PATH])
    artifact_sha256 = {relative: orchestrator.sha256_bytes(payload)
                       for relative, payload in documents
                       if relative != analyzer.MANIFEST_RELATIVE_PATH}
    commit = json.loads(payloads[analyzer.MANIFEST_RELATIVE_PATH].decode(
        "utf-8"))["analysis_code_commit"]
    template = analyzer.build_acceptance_template(manifest_sha256, artifact_sha256, commit)
    reporter.check("a well-formed future acceptance attestation validates",
                   not analyzer.validate_acceptance_document(
                       template, manifest_sha256=manifest_sha256,
                       artifact_sha256=artifact_sha256, analysis_code_commit=commit), "")
    for label, mutate in (
            ("one wrong artifact hash",
             lambda doc: doc["artifact_sha256"].__setitem__("report.md", "0" * 64)),
            ("a wrong manifest hash",
             lambda doc: doc.__setitem__("analysis_manifest_sha256", "0" * 64)),
            ("a different analyzer commit",
             lambda doc: doc.__setitem__("analysis_code_commit", "f" * 40)),
            ("a missing field", lambda doc: doc.pop("artifact_sha256")),
            ("an unexpected field", lambda doc: doc.__setitem__("mutable_note", "x")),
            ("a substituted population",
             lambda doc: doc.__setitem__("final_campaign_ids", [])),
            ("a malformed status", lambda doc: doc.__setitem__("status", "MAYBE"))):
        document = json.loads(json.dumps(template))
        mutate(document)
        reporter.check(f"an acceptance attestation with {label} is rejected",
                       bool(analyzer.validate_acceptance_document(
                           document, manifest_sha256=manifest_sha256,
                           artifact_sha256=artifact_sha256,
                           analysis_code_commit=commit)), "")
    incomplete_reference = dict(artifact_sha256)
    incomplete_reference.pop("memory_paths.csv")
    reporter.check("an incomplete trusted reference hash map cannot validate an otherwise "
                   "well-formed attestation",
                   bool(analyzer.validate_acceptance_document(
                       template, manifest_sha256=manifest_sha256,
                       artifact_sha256=incomplete_reference,
                       analysis_code_commit=commit)), "")


def _check_acceptance_record(reporter: Reporter, analyzer, orchestrator,
                             repo_root: Path, documents: dict[str, str]) -> None:
    """3.5b -- the external acceptance attestation exists, is well formed, and
    binds exactly the accepted bytes.

    The file is parsed as JSON and validated structurally; it is never matched
    as text. It is checked twice over: field by field against the frozen
    closure facts, and then by the analyzer's own frozen
    ``validate_acceptance_document()`` against trusted inputs recomputed from
    the committed bundle. An attestation that differs in one campaign ID,
    commit, path, or hash is therefore rejected, and it can never authorize a
    modified or partially regenerated bundle.

    The closure/documentation commit that adds this file is deliberately not
    one of the recorded commits: it is not the experimental execution, it did
    not produce the bundle, and it does not contain the accepted candidate.
    """
    path = repo_root / P43_ACCEPTANCE_RELATIVE_PATH
    if not reporter.check(
            f"{P43_ACCEPTANCE_RELATIVE_PATH} exists as a regular, non-symlink file",
            path.is_file() and not path.is_symlink(), str(path)):
        return
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        reporter.check(f"{P43_ACCEPTANCE_RELATIVE_PATH} parses as JSON", False,
                       f"{type(exc).__name__}: {exc}")
        return
    reporter.check(f"{P43_ACCEPTANCE_RELATIVE_PATH} parses as JSON", True)
    if not reporter.check("the acceptance record is a JSON object",
                          isinstance(document, dict), type(document).__name__):
        return

    reporter.check("the acceptance record carries exactly the frozen "
                   f"{analyzer.ACCEPTANCE_SCHEMA_VERSION} top-level fields",
                   set(document) == set(analyzer.ACCEPTANCE_REQUIRED_FIELDS),
                   str(sorted(set(document) ^ set(analyzer.ACCEPTANCE_REQUIRED_FIELDS))))
    for field, expected in (
            ("schema_version", analyzer.ACCEPTANCE_SCHEMA_VERSION),
            ("unit", analyzer.UNIT),
            ("status", analyzer.ACCEPTANCE_STATUS_ACCEPTED),
            ("analysis_code_commit", ACCEPTED_ANALYZER_COMMIT),
            ("final_campaign_ids", list(analyzer.FINAL_CAMPAIGN_IDS)),
            ("pilot_campaign_id_excluded", analyzer.PILOT_CAMPAIGN_ID),
            ("analysis_manifest_sha256", ACCEPTED_MANIFEST_SHA256),
            ("verification_outcome", analyzer.ACCEPTANCE_VERIFICATION_OUTCOME),
            ("independent_output_review_outcome", analyzer.ACCEPTANCE_REVIEW_OUTCOME)):
        reporter.check(f"the acceptance record pins {field} to {expected!r}",
                       document.get(field) == expected, repr(document.get(field)))
    reporter.check("accepted_for_publication is exactly the boolean true",
                   document.get("accepted_for_publication") is True,
                   repr(document.get("accepted_for_publication")))
    commit = document.get("analysis_code_commit")
    reporter.check("the recorded analyzer commit is 40 lowercase hexadecimal characters",
                   isinstance(commit, str) and GIT_COMMIT_RE.fullmatch(commit) is not None,
                   repr(commit))
    digest = document.get("analysis_manifest_sha256")
    reporter.check("the recorded manifest digest is 64 lowercase hexadecimal characters",
                   isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None,
                   repr(digest))
    hashes = document.get("artifact_sha256")
    reporter.check("the acceptance record binds exactly the nine frozen artifacts",
                   isinstance(hashes, dict)
                   and set(hashes) == set(analyzer.ARTIFACT_RELATIVE_PATHS),
                   str(sorted(set(hashes) ^ set(analyzer.ARTIFACT_RELATIVE_PATHS)))
                   if isinstance(hashes, dict) else type(hashes).__name__)
    if isinstance(hashes, dict):
        malformed = sorted(relative for relative, value in hashes.items()
                           if not isinstance(value, str)
                           or SHA256_RE.fullmatch(value) is None)
        reporter.check("every bound artifact digest is 64 lowercase hexadecimal "
                       "characters", not malformed, str(malformed))

    # The three provenance commits are three different facts. Conflating any
    # two of them would let a different execution, analyzer, or candidate be
    # published under an accepted name.
    reporter.check("the experimental execution, analyzer, and candidate commits are "
                   "three distinct 40-character commits",
                   len({ACCEPTED_EXECUTION_COMMIT, ACCEPTED_ANALYZER_COMMIT,
                        ACCEPTED_CANDIDATE_COMMIT}) == 3
                   and all(GIT_COMMIT_RE.fullmatch(value) is not None
                           for value in (ACCEPTED_EXECUTION_COMMIT,
                                         ACCEPTED_ANALYZER_COMMIT,
                                         ACCEPTED_CANDIDATE_COMMIT)), "")
    reporter.check("the acceptance record does not record the execution or the candidate "
                   "commit as the analyzer commit",
                   commit not in (ACCEPTED_EXECUTION_COMMIT, ACCEPTED_CANDIDATE_COMMIT),
                   repr(commit))
    reporter.check("the analyzer's frozen execution commit is the accepted one",
                   analyzer.FINAL_EXECUTION_COMMIT == ACCEPTED_EXECUTION_COMMIT,
                   analyzer.FINAL_EXECUTION_COMMIT)
    reporter.check("this checker's bundle inventory equals the analyzer's frozen "
                   "inventory, in the frozen order",
                   ACCEPTED_BUNDLE_ARTIFACTS == tuple(analyzer.ARTIFACT_RELATIVE_PATHS),
                   str(ACCEPTED_BUNDLE_ARTIFACTS))

    # Bind the attestation to the bytes that are actually committed.
    payloads: dict[str, bytes] = {}
    unreadable: list[str] = []
    for relative in analyzer.ARTIFACT_RELATIVE_PATHS:
        artifact = repo_root / ACCEPTED_BUNDLE_ROOT / relative
        try:
            if artifact.is_symlink() or not artifact.is_file():
                unreadable.append(relative)
                continue
            payloads[relative] = artifact.read_bytes()
        except OSError:
            unreadable.append(relative)
    if not reporter.check("every accepted artifact is present as a regular, non-symlink "
                          "file", not unreadable, str(unreadable)):
        return
    recomputed = {relative: orchestrator.sha256_bytes(payload)
                  for relative, payload in payloads.items()}
    reporter.check("the accepted manifest's own SHA-256 is the frozen digest",
                   recomputed[analyzer.MANIFEST_RELATIVE_PATH] == ACCEPTED_MANIFEST_SHA256,
                   recomputed[analyzer.MANIFEST_RELATIVE_PATH])
    if isinstance(hashes, dict):
        mismatched = sorted(relative for relative, value in recomputed.items()
                            if hashes.get(relative) != value)
        reporter.check("the acceptance record binds the exact committed bytes of all "
                       "nine artifacts", not mismatched, str(mismatched))
    try:
        manifest = json.loads(
            payloads[analyzer.MANIFEST_RELATIVE_PATH].decode("utf-8"))
    except ValueError as exc:
        reporter.check("the accepted manifest parses as JSON", False, str(exc))
        return
    reporter.check("the accepted manifest records the analyzer commit the attestation "
                   "binds",
                   manifest.get("analysis_code_commit") == ACCEPTED_ANALYZER_COMMIT,
                   repr(manifest.get("analysis_code_commit")))
    reporter.check("the accepted manifest records the experimental execution commit the "
                   "campaigns ran from",
                   manifest.get("final_execution_commit") == ACCEPTED_EXECUTION_COMMIT,
                   repr(manifest.get("final_execution_commit")))
    reporter.check("the accepted manifest never conflates the execution and analyzer "
                   "commits",
                   manifest.get("final_execution_commit")
                   != manifest.get("analysis_code_commit"), "")

    siblings = {relative: value for relative, value in recomputed.items()
                if relative != analyzer.MANIFEST_RELATIVE_PATH}
    errors = analyzer.validate_acceptance_document(
        document, manifest_sha256=recomputed[analyzer.MANIFEST_RELATIVE_PATH],
        artifact_sha256=siblings,
        analysis_code_commit=manifest.get("analysis_code_commit", ""))
    reporter.check("the analyzer's own frozen acceptance validator accepts the real "
                   "attestation", not errors, str(errors[:3]))

    # The narrative documents must carry the same closure facts, so the record
    # and the documentation can never drift apart.
    for relative in ("PLAN.md", "README.md", "results/README.md",
                     P43_PROTOCOL_RELATIVE_PATH):
        text = documents[relative]
        for token in (ACCEPTED_EXECUTION_COMMIT, ACCEPTED_ANALYZER_COMMIT,
                      ACCEPTED_CANDIDATE_COMMIT, ACCEPTED_MANIFEST_SHA256,
                      ACCEPTED_AUDIT_VERDICT, ACCEPTED_COMPARISON_METHOD,
                      P43_ACCEPTANCE_RELATIVE_PATH):
            reporter.check(f"{relative} records the closure fact {token!r}",
                           token in text, "")


def _check_analysis_code_commit(reporter: Reporter, analyzer) -> None:
    """3.6 -- the analysis-code commit is resolved and verified at runtime,
    never hard-coded and never bypassed."""
    reporter.check("the analysis-code commit is not hard-coded anywhere",
                   not re.search(r"analysis_code_commit\s*=\s*[\"'][0-9a-f]{40}[\"']",
                                 inspect.getsource(analyzer)), "")
    reporter.check("the frozen final execution commit is never reused as the analysis-code "
                   "commit",
                   "FINAL_EXECUTION_COMMIT"
                   in _strip(inspect.getsource(analyzer.validate_analysis_provenance)), "")
    source = inspect.getsource(analyzer.resolve_git_provenance)
    for expectation in ("head", "cache-tree", "worktree"):
        reporter.check(f"the provenance reader verifies the {expectation}",
                       expectation in source.lower(), expectation)
    reporter.check("provenance resolution runs no child process and needs no network",
                   not re.search(r"\bsubprocess\b|\bos\.system\b|\bsocket\b|\burllib\b",
                                 _strip(inspect.getsource(analyzer))), "")
    reporter.check("every Git provenance helper is read-only",
                   not [name for name in analyzer.GIT_PROVENANCE_FUNCTION_NAMES
                        if any(re.search(pattern,
                                         inspect.getsource(getattr(analyzer, name)))
                               for pattern in FORBIDDEN_EVIDENCE_MODE_CALLS)],
                   str([name for name in analyzer.GIT_PROVENANCE_FUNCTION_NAMES
                        if any(re.search(pattern,
                                         inspect.getsource(getattr(analyzer, name)))
                               for pattern in FORBIDDEN_EVIDENCE_MODE_CALLS)]))
    reporter.check("untracked content is allowed only below the three frozen data roots",
                   analyzer.GIT_ALLOWED_UNTRACKED_ROOTS
                   == (("results", "raw"), ("results", "preflight"),
                       ("results", "phase4")),
                   str(analyzer.GIT_ALLOWED_UNTRACKED_ROOTS))
    reporter.check("the production interpreter is currently isolated from checkout and "
                   "environment import injection",
                   sys.flags.isolated == 1 and sys.flags.ignore_environment == 1
                   and sys.flags.no_user_site == 1 and sys.dont_write_bytecode, "")
    try:
        analyzer.validate_production_python_runtime()
    except analyzer.P43Error as exc:
        reporter.check("the analyzer accepts the required -I -B runtime", False, str(exc))
    else:
        reporter.check("the analyzer accepts the required -I -B runtime", True)

    with tempfile.TemporaryDirectory(prefix="p43-check-untracked-") as temporary:
        fixture = Path(temporary) / "repo"
        fixture.mkdir()
        analyzer._git_fixture_repository(
            fixture, files={"README.md": (b"fixture\n", analyzer.GIT_MODE_REGULAR)})
        (fixture / "scripts").mkdir()
        (fixture / "scripts" / "csv.py").write_text(
            "raise SystemExit(99)\n", encoding="utf-8")
        try:
            analyzer.resolve_git_provenance(fixture)
        except analyzer.GitProvenanceError as exc:
            reporter.check("an untracked scripts/csv.py cannot retain clean provenance",
                           "untracked path outside the allowed data roots" in str(exc),
                           str(exc))
        else:
            reporter.check("an untracked scripts/csv.py cannot retain clean provenance",
                           False, "accepted")

    def rejects(label: str, provenance) -> None:
        try:
            analyzer.validate_analysis_provenance(provenance)
        except analyzer.P43Error:
            reporter.check(label, True)
        else:
            reporter.check(label, False, "accepted")

    rejects("a missing analysis-code commit is rejected", {})
    rejects("an abbreviated analysis-code commit is rejected",
            analyzer._fixture_provenance(analysis_code_commit="1234567"))
    rejects("an uppercase analysis-code commit is rejected",
            analyzer._fixture_provenance(analysis_code_commit="A" * 40))
    rejects("a dirty analysis-code worktree is rejected",
            analyzer._fixture_provenance(worktree_clean=False))
    rejects("an analysis-code commit equal to the execution commit is rejected",
            analyzer._fixture_provenance(
                analysis_code_commit=analyzer.FINAL_EXECUTION_COMMIT))
    reporter.check("production analysis takes the provenance resolver by injection and "
                   "defaults to the real verifier",
                   "git_provenance=resolve_git_provenance"
                   in inspect.getsource(analyzer.run_analysis), "")
    run_source = inspect.getsource(analyzer.run_analysis)
    reporter.check("the clean checkout is verified before repository-owned dependency "
                   "modules or the evidence revalidator execute",
                   run_source.index("provenance = git_provenance")
                   < run_source.index("load_repository_modules")
                   < run_source.index("status = revalidator"), "")
    reporter.check("there is no production flag that skips provenance verification",
                   not any("provenance" in option.lower() or "commit" in option.lower()
                           for action in analyzer.build_parser()._actions
                           for option in action.option_strings), "")


def _check_make_targets(reporter: Reporter, repo_root: Path, makefile: str, p41) -> None:
    rules = p41._parse_makefile_rules(makefile)
    for target in (P43_CHECK_TARGET, P43_ANALYZE_TARGET, P43_VERIFY_TARGET):
        reporter.check(f"the Makefile declares {target}", target in rules, "")
    if P43_CHECK_TARGET not in rules:
        return
    closure = p41._target_closure(rules, P43_CHECK_TARGET)
    unknown = [target for target in closure if target not in rules]
    reporter.check(f"every prerequisite of {P43_CHECK_TARGET} resolves to a real rule",
                   not unknown, str(unknown))
    offenders: list[str] = []
    for target in closure:
        for line in rules.get(target, ([], []))[1]:
            command = line.lstrip().lstrip("@-+").lstrip()
            for name in p41.FORBIDDEN_CHECK_PATH_COMMANDS:
                if re.search(p41._COMMAND_POSITION.format(name=re.escape(name)), command):
                    offenders.append(f"{target}: {line.strip()[:80]}")
    reporter.check(f"no recipe reachable from {P43_CHECK_TARGET} invokes a container "
                   f"runtime, nvidia-smi, Nsight Compute, or a CUDA compiler",
                   not offenders, str(offenders[:3]))
    reporter.check(f"{P43_CHECK_TARGET} reaches no Docker-backed or GPU gate",
                   not ({"compute-umma-p24-check", "gemm-comparison-p35-check", "preflight",
                         "build-image", "check-env"} & set(closure)), str(sorted(closure)))
    recipe = "\n".join(rules[P43_CHECK_TARGET][1])
    reporter.check(f"{P43_CHECK_TARGET} runs both P4.3 self-tests and the repository check",
                   recipe.count("--self-test") >= 2
                   and ("$(PHASE4_P43_CHECKER)" in recipe
                        or P43_CHECKER_RELATIVE_PATH in recipe)
                   and ("$(PHASE4_P43_ANALYZER)" in recipe
                        or P43_ANALYZER_RELATIVE_PATH in recipe), recipe[:200])
    reporter.check(f"{P43_CHECK_TARGET} needs no raw campaign evidence",
                   "results/raw" not in recipe and "$(PHASE4_P41_RAW_ROOT)" not in recipe, "")
    for target in (P43_CHECK_TARGET, P43_ANALYZE_TARGET, P43_VERIFY_TARGET):
        if target not in rules:
            continue
        text = "\n".join(rules[target][1])
        reporter.check(f"{target} never starts or resumes a campaign",
                       "--resume" not in text
                       and not re.search(r"run_all\.sh", text), "")
        python_invocations = [line.strip() for line in rules[target][1]
                              if re.search(r"\bpython3\b", line)]
        reporter.check(f"{target} isolates every Python invocation with -I -B",
                       python_invocations
                       and all(re.search(r"\bpython3\s+-I\s+-B\b", line)
                               for line in python_invocations),
                       str(python_invocations))
    for target, flag in ((P43_ANALYZE_TARGET, "--analyze"), (P43_VERIFY_TARGET, "--verify")):
        if target not in rules:
            continue
        text = "\n".join(rules[target][1])
        reporter.check(f"{target} runs the analyzer with {flag} and explicit frozen IDs",
                       flag in text and text.count("--final-campaign-id") == 3
                       and "--pilot-campaign-id" in text, "")
    reporter.check("the Makefile adds no target that could start a Phase 4 campaign",
                   not re.search(r"^phase4-p43-(pilot|final|campaign|run|smoke):",
                                 makefile, re.MULTILINE), "")
    reporter.check("the closed P4.1 and P4.2 Make targets still exist",
                   all(re.search(rf"^{name}:", makefile, re.MULTILINE) is not None
                       for name in ("phase4-p41-plan", "phase4-p41-check",
                                    "phase4-p42-check")), "")
    for variable, expected in (("PHASE4_P43_ANALYZER", P43_ANALYZER_RELATIVE_PATH),
                               ("PHASE4_P43_CHECKER", P43_CHECKER_RELATIVE_PATH),
                               ("PHASE4_P43_PROTOCOL", P43_PROTOCOL_RELATIVE_PATH)):
        reporter.check(f"the Makefile binds {variable} to {expected}",
                       re.search(rf"^{variable}\s*:?=\s*{re.escape(expected)}\s*$",
                                 makefile, re.MULTILINE) is not None, "")
    reporter.check("the Makefile declares the P4.3 files as required files",
                   all(f"$({name})" in makefile for name in
                       ("PHASE4_P43_ANALYZER", "PHASE4_P43_CHECKER",
                        "PHASE4_P43_PROTOCOL")), "")


def _check_make_frozen_ids(reporter: Reporter, makefile: str, analyzer) -> None:
    """The production target must use exactly the frozen campaign IDs."""
    for variable, expected in (("PHASE4_P43_PILOT_CAMPAIGN_ID", analyzer.PILOT_CAMPAIGN_ID),
                               ("PHASE4_P43_FINAL_CAMPAIGN_1",
                                analyzer.FINAL_CAMPAIGN_IDS[0]),
                               ("PHASE4_P43_FINAL_CAMPAIGN_2",
                                analyzer.FINAL_CAMPAIGN_IDS[1]),
                               ("PHASE4_P43_FINAL_CAMPAIGN_3",
                                analyzer.FINAL_CAMPAIGN_IDS[2])):
        reporter.check(f"the Makefile pins {variable} to the frozen {expected}",
                       re.search(rf"^{variable}\s*:?=\s*{expected}\s*$", makefile,
                                 re.MULTILINE) is not None, "")
    reporter.check("the Makefile's curated output root is never under results/raw/",
                   re.search(r"^PHASE4_P43_OUTPUT_ROOT\s*:?=\s*results/raw", makefile,
                             re.MULTILINE) is None, "")


def _check_output_tree_committable(reporter: Reporter, repo_root: Path,
                                   analyzer) -> None:
    ignore = read_text(repo_root, ".gitignore")
    reporter.check("the curated P4.3 output tree is not blanket-ignored by Git",
                   not re.search(rf"^{re.escape(analyzer.DEFAULT_OUTPUT_ROOT_REL)}/?$",
                                 ignore, re.MULTILINE), "")
    reporter.check("raw campaign evidence stays ignored",
                   re.search(r"^results/raw/$", ignore, re.MULTILINE) is not None, "")


def _check_cli_surfaces(reporter: Reporter, analyzer) -> None:
    options = set()
    for action in analyzer.build_parser()._actions:
        options.update(action.option_strings)
    reporter.check("the analyzer's whole CLI surface is the three documented modes",
                   options == EXPECTED_ANALYZER_CLI_OPTIONS,
                   str(sorted(options ^ EXPECTED_ANALYZER_CLI_OPTIONS)))
    checker_options = set()
    for action in build_parser()._actions:
        checker_options.update(action.option_strings)
    reporter.check("this checker's whole CLI surface is the two documented modes",
                   checker_options == EXPECTED_CHECKER_CLI_OPTIONS,
                   str(sorted(checker_options ^ EXPECTED_CHECKER_CLI_OPTIONS)))
    reporter.check("neither mode can select a GPU",
                   not any("gpu" in option.lower()
                           for option in options | checker_options), "")
    reporter.check("the analyzer requires the real evidence and the frozen IDs explicitly",
                   analyzer.main(["--analyze"]) == 2, "")
    reporter.check("the analyzer refuses --analyze together with --verify",
                   analyzer.main(["--analyze", "--verify"]) == 2, "")
    reporter.check("the analyzer refuses a mode-less invocation",
                   analyzer.main([]) == 2, "")
    reporter.check("the analyzer's --self-test is standalone",
                   analyzer.main(["--self-test", "--analyze"]) == 2, "")


def _check_frozen_execution_path(reporter: Reporter, repo_root: Path, orchestrator,
                                 p42) -> None:
    for relative, expected in p42.FROZEN_EXECUTION_PATH_SHA256.items():
        digest = orchestrator.sha256_of_path(repo_root / relative)
        reporter.check(f"{relative} is still byte-identical to the audited execution path",
                       digest == expected,
                       f"{digest} != {expected}; P4.3 may not change how a campaign runs")


def _strip(source: str) -> str:
    """Drop comments and literal text so prose naming a forbidden spelling in
    order to ban it does not trip a scan of executable code."""
    import io as _io
    import tokenize

    dropped = {tokenize.COMMENT, tokenize.STRING}
    for optional in ("FSTRING_MIDDLE",):
        if hasattr(tokenize, optional):
            dropped.add(getattr(tokenize, optional))
    output = []
    try:
        for token in tokenize.generate_tokens(_io.StringIO(source).readline):
            if token.type in dropped:
                continue
            output.append(token.string)
    except (tokenize.TokenError, IndentationError):
        return source
    return "\n".join(output)


def check_repository(repo_root: Path) -> int:
    reporter = Reporter("check_phase4_integration_p43")
    if not _check_required_files(reporter, repo_root):
        print("check_phase4_integration_p43: FAILED (missing required files)",
              file=sys.stderr)
        return 1

    documents = {relative: read_text(repo_root, relative) for relative in STATUS_DOCUMENTS}
    makefile = read_text(repo_root, "Makefile")
    sources = {relative: read_text(repo_root, relative)
               for relative in (P43_ANALYZER_RELATIVE_PATH, P43_CHECKER_RELATIVE_PATH)}
    orchestrator = load_module(repo_root / ORCHESTRATOR_RELATIVE_PATH, "_p43_orchestrator_c")
    p41 = load_module(repo_root / P41_CHECKER_RELATIVE_PATH, "_p43_p41_checker")
    p42 = load_module(repo_root / P42_CHECKER_RELATIVE_PATH, "_p43_p42_checker")
    analyzer = load_module(repo_root / P43_ANALYZER_RELATIVE_PATH, "_p43_analyzer")

    _check_status_frontier(reporter, documents)
    _check_truthful_claims(reporter, documents)
    _check_frozen_population(reporter, analyzer, p42)
    _check_no_new_dependency(reporter, repo_root, sources)
    _check_never_executes(reporter, sources, p41)
    _check_read_only_evidence(reporter, analyzer)
    _check_reuses_closed_units(reporter, sources[P43_ANALYZER_RELATIVE_PATH])
    _check_statistical_policy(reporter, analyzer)
    _check_artifact_contract(reporter, analyzer,
                             documents[P43_PROTOCOL_RELATIVE_PATH])
    try:
        _check_evidence_taxonomy(reporter, analyzer,
                                 documents[P43_PROTOCOL_RELATIVE_PATH])
    except (analyzer.P43Error, KeyError, TypeError, AttributeError, ValueError) as exc:
        reporter.check("the scientific evidence taxonomy contract holds", False,
                       f"{type(exc).__name__}: {exc}")
    try:
        _check_preserved_diagnostics(reporter, analyzer, orchestrator)
        bundle = analyzer.build_documents(orchestrator, analyzer._StubP35(),
                                          analyzer._fixture_records(),
                                          analyzer._fixture_provenance())
    except (analyzer.P43Error, KeyError, TypeError, AttributeError, ValueError) as exc:
        reporter.check("the analyzer produces its own candidate bundle end to end",
                       False, f"{type(exc).__name__}: {exc}")
        bundle = None
    if bundle is not None:
        try:
            _check_metadata_ownership(reporter, analyzer, orchestrator, bundle, repo_root)
            _check_output_containment(reporter, analyzer, orchestrator)
            _check_candidate_and_acceptance(reporter, analyzer, orchestrator, bundle,
                                            repo_root,
                                            documents[P43_PROTOCOL_RELATIVE_PATH])
        except (analyzer.P43Error, KeyError, TypeError, AttributeError,
                ValueError, OSError) as exc:
            reporter.check("the bundle, containment, and acceptance contracts hold",
                           False, f"{type(exc).__name__}: {exc}")
    try:
        _check_acceptance_record(reporter, analyzer, orchestrator, repo_root, documents)
    except (analyzer.P43Error, KeyError, TypeError, AttributeError,
            ValueError, OSError) as exc:
        reporter.check("the external acceptance attestation contract holds", False,
                       f"{type(exc).__name__}: {exc}")
    try:
        _check_analysis_code_commit(reporter, analyzer)
    except (analyzer.P43Error, KeyError, TypeError, AttributeError, ValueError) as exc:
        reporter.check("the analysis-code provenance contract holds", False,
                       f"{type(exc).__name__}: {exc}")
    _check_cli_surfaces(reporter, analyzer)
    _check_make_targets(reporter, repo_root, makefile, p41)
    _check_make_frozen_ids(reporter, makefile, analyzer)
    _check_output_tree_committable(reporter, repo_root, analyzer)
    _check_frozen_execution_path(reporter, repo_root, orchestrator, p42)

    reporter.check("the repository contract check needed no raw campaign evidence", True)
    reporter.check("the repository contract check needed no container, GPU, or network",
                   True)

    if reporter.failures:
        print(f"check_phase4_integration_p43: FAILED ({len(reporter.failures)} check(s))",
              file=sys.stderr)
        return 1
    print("check_phase4_integration_p43: OK "
          "(P4.3 contract passed, and the external acceptance attestation binds the exact "
          "committed candidate bytes; this GPU-free check re-reads a recorded acceptance, "
          "it never performs an independent audit, a production analysis, or an output "
          "review of its own)")
    return 0


# ===========================================================================
# Self-test. Temporary directories only.
# ===========================================================================


def _copy_repository_for_contract_check(source: Path, destination: Path) -> None:
    """A minimal copy of exactly what repository-contract mode reads, with no
    results/raw/ tree, to prove the mode needs no raw campaign evidence."""
    needed = list(REQUIRED_P43_FILES) + list(STATUS_DOCUMENTS) + [
        "Makefile", "VERSIONS.env", "PHASE3_VERSIONS.env", ".gitignore",
        "scripts/check_gemm_comparison_p35.py"]
    destination.mkdir(parents=True)
    for relative in sorted(set(needed)):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
        os.chmod(target, os.stat(source / relative).st_mode)


def run_self_test() -> int:
    reporter = Reporter("check_phase4_integration_p43: self-test")
    analyzer = load_module(DEFAULT_REPO_ROOT / P43_ANALYZER_RELATIVE_PATH,
                           "_p43_selftest_analyzer")

    with tempfile.TemporaryDirectory(prefix="p43-check-selftest-") as temporary:
        root = Path(temporary)
        clone = root / "clone"
        _copy_repository_for_contract_check(DEFAULT_REPO_ROOT, clone)
        reporter.check("the bare clone really has no results/raw/ tree",
                       not (clone / "results" / "raw").exists(), "")
        reporter.check("repository-contract mode passes without any results/raw/ directory",
                       check_repository(clone) == 0, "")

        def rewrite(path: Path, text: str) -> None:
            """Rewrite a clone file and drop every cached view of it.

            A mutation and its restoration can land in the same second and
            produce the same file size, in which case a stale __pycache__ entry
            or a stale linecache line buffer would be reused and the adversarial
            test would silently examine the wrong source."""
            path.write_text(text, encoding="utf-8")
            shutil.rmtree(clone / "scripts" / "__pycache__", ignore_errors=True)
            importlib.invalidate_caches()
            linecache.clearcache()

        def mutate(relative: str, old: str, new: str, label: str,
                   every: bool = False) -> None:
            path = clone / relative
            original = path.read_text(encoding="utf-8")
            if old not in original:
                reporter.check(label, False, f"{old!r} not found in {relative}")
                return
            rewrite(path, original.replace(old, new) if every
                    else original.replace(old, new, 1))
            try:
                reporter.check(label, check_repository(clone) == 1, "")
            finally:
                rewrite(path, original)

        mutate("PLAN.md",
               "| P4.3 | Integrated analysis, documentation, audit | YES | YES | YES |",
               "| P4.3 | Integrated analysis, documentation, audit | YES | NO | NO |",
               "a PLAN.md that regresses accepted P4.3 to implemented-only is rejected")
        mutate("PLAN.md",
               "| P4.3 | Integrated analysis, documentation, audit | YES | YES | YES |",
               "| P4.3 | Integrated analysis, documentation, audit | NO | NO | NO |",
               "a PLAN.md that still calls P4.3 unimplemented is rejected")
        mutate("PLAN.md",
               "| P4.3 | Integrated analysis, documentation, audit | YES | YES | YES |",
               "| P4.3 | Integrated analysis, documentation, audit | YES | YES | NO |",
               "a PLAN.md that claims P4.3 was audited but not verified is rejected")
        mutate("PLAN.md",
               "| P4.2 | Pilot plus three final campaigns | YES | YES | YES |",
               "| P4.2 | Pilot plus three final campaigns | YES | NO | NO |",
               "a PLAN.md that regresses closed P4.2 is rejected")
        mutate("PLAN.md",
               "| P1.4 | Profiling, validation, analysis, pilot | YES | YES | YES |",
               "| P1.4 | Profiling, validation, analysis, pilot | YES | NO | NO |",
               "a PLAN.md that regresses closed P1.4 is rejected")
        mutate(P43_PROTOCOL_RELATIVE_PATH,
               "Independent audit: ACCEPT WITH NON-BLOCKING OBSERVATIONS",
               "Independent audit: NOT PERFORMED",
               "a protocol that reverts to the stale pre-acceptance audit state is "
               "rejected")
        mutate(P43_PROTOCOL_RELATIVE_PATH, "P4.3 = YES / YES / YES", "P4.3 = YES / NO / NO",
               "a protocol that reopens P4.3 is rejected", every=True)
        mutate(P43_PROTOCOL_RELATIVE_PATH,
               "P4.2 itself produced no publishable Phase 4 result",
               "P4.2 itself produced a publishable Phase 4 result",
               "a protocol that drops the P4.2 non-publication boundary is rejected")
        mutate("README.md",
               "P4.3: CLOSED; independent audit: YES; production analysis: YES",
               "P4.3: IMPLEMENTED; independent audit: NO; production analysis: NO",
               "a README that reverts to the stale implemented-only P4.3 header is "
               "rejected")
        mutate("results/README.md", "P4.3 is closed", "P4.3 is implemented",
               "a results/README.md that reopens P4.3 is rejected")
        mutate(P43_PROTOCOL_RELATIVE_PATH, "publishable=false", "publishable=true",
               "a protocol that promotes an artifact to publishable is rejected",
               every=True)
        mutate(P43_PROTOCOL_RELATIVE_PATH, "figures/memory_paths.svg", "figures/other.svg",
               "a protocol whose artifact inventory disagrees with the analyzer is rejected",
               every=True)
        mutate("Makefile", "PHASE4_P43_FINAL_CAMPAIGN_1 := 20260817T110330Z",
               "PHASE4_P43_FINAL_CAMPAIGN_1 := 20260818T000000Z",
               "a Makefile that analyses a campaign outside the frozen population is "
               "rejected")
        mutate("Makefile", "PHASE4_P43_OUTPUT_ROOT := results/phase4",
               "PHASE4_P43_OUTPUT_ROOT := results/raw/phase4-analysis",
               "a Makefile that writes the analysis under results/raw/ is rejected")

        # Second independent audit, finding M5: the frozen taxonomy table and
        # the implementation must stay equal in both directions.
        mutate(P43_PROTOCOL_RELATIVE_PATH,
               "profile_sm_clock_status                        source_diagnostic\n", "",
               "a protocol whose section 5.1 table omits a classified metric is rejected")
        mutate(P43_PROTOCOL_RELATIVE_PATH,
               "ncu_coverage                                   source_diagnostic",
               "ncu_coverage                                   source_diagnostic\n"
               "an_invented_metric                             source_diagnostic",
               "a protocol whose section 5.1 table invents a metric the analyzer does "
               "not classify is rejected")
        mutate(P43_PROTOCOL_RELATIVE_PATH,
               "profile_sm_clock_status                        source_diagnostic",
               "profile_sm_clock_status                        modeled_estimate",
               "a protocol whose section 5.1 table disagrees with the analyzer about an "
               "evidence class is rejected")

        analyzer_path = clone / P43_ANALYZER_RELATIVE_PATH
        original = analyzer_path.read_text(encoding="utf-8")
        for old, new, label in (
            ("import argparse", "import argparse\nimport subprocess",
             "an analyzer that could start a child process is rejected"),
            ("/ (count - 1)", "/ count",
             "an analyzer using the population standard deviation is rejected"),
            ("CV_REVIEW_THRESHOLD_PERCENT = 5.0", "CV_REVIEW_THRESHOLD_PERCENT = 50.0",
             "an analyzer that weakens the strict CV review threshold is rejected"),
            # Second independent audit, finding M1: a layout that lets a panel's
            # axis decorations reach back into the preceding panel, and the two
            # ways that defect could be hidden rather than fixed.
            ("_SVG_PANEL_SEPARATION = 14.0", "_SVG_PANEL_SEPARATION = -60.0",
             "an analyzer whose panels overlap their neighbour's gutter is rejected"),
            ("_SVG_GUTTER_PAD = 3.0", "_SVG_GUTTER_PAD = -55.0",
             "an analyzer that pushes an axis title back into the preceding panel is "
             "rejected"),
            ("_SVG_CHAR_ADVANCE = 0.62", "_SVG_CHAR_ADVANCE = 0.10",
             "an analyzer that under-reserves the gutter by shrinking its text metric "
             "is rejected"),
            ('SIGNED_OR_ZERO_CENTRED_METRICS = frozenset({"gap_to_cublaslt_pct"})',
             "SIGNED_OR_ZERO_CENTRED_METRICS = frozenset()",
             "an analyzer that would compute a CV for the signed GEMM gap is rejected"),
            (f'PILOT_CAMPAIGN_ID = "{analyzer.PILOT_CAMPAIGN_ID}"',
             f'PILOT_CAMPAIGN_ID = "{analyzer.FINAL_CAMPAIGN_IDS[0]}"',
             "an analyzer whose pilot is one of the final replicates is rejected"),
            ('FINAL_CAMPAIGN_IDS = (\n    "20260817T110330Z",',
             'FINAL_CAMPAIGN_IDS = (\n    "20260818T000000Z",',
             "an analyzer that substitutes a campaign outside the frozen population is "
             "rejected"),
            ('ARTIFACT_RELATIVE_PATHS = (\n    "memory_paths.csv",',
             'ARTIFACT_RELATIVE_PATHS = (\n    "memory_paths_extra.csv",',
             "an analyzer whose artifact inventory drifts from the frozen tree is "
             "rejected"),
        ):
            if old not in original:
                reporter.check(label, False, f"{old!r} not found in the analyzer")
                continue
            rewrite(analyzer_path, original.replace(old, new, 1))
            try:
                reporter.check(label, check_repository(clone) == 1, "")
            finally:
                rewrite(analyzer_path, original)
        # The seven audit findings, as mutations of the remediated analyzer.
        # Every one of them describes the pre-remediation implementation, so
        # each of these checks would have failed on that implementation.
        for old, new, label in (
            ('"median_effective_gbps": (\n        EVIDENCE_WITHIN_CAMPAIGN,',
             '"median_effective_gbps": (\n        EVIDENCE_MEASURED,',
             "an analyzer that classifies the timing-derived effective transfer rate as a "
             "direct measurement is rejected"),
            ('"median_flops_per_cycle": (\n        EVIDENCE_WITHIN_CAMPAIGN,',
             '"median_flops_per_cycle": (\n        EVIDENCE_MEASURED,',
             "an analyzer that presents FLOP/cycle as directly measured is rejected"),
            ('"dram_read_ratio": (\n        EVIDENCE_WITHIN_CAMPAIGN,',
             '"dram_read_ratio": (\n        EVIDENCE_MEASURED,',
             "an analyzer that presents the profiler-derived DRAM ratio as a raw "
             "measurement is rejected"),
            ('"estimated_tflops_per_sm": (\n        EVIDENCE_MODELED,',
             '"estimated_tflops_per_sm": (\n        EVIDENCE_MEASURED,',
             "an analyzer that presents the modeled clock conversion as measured is "
             "rejected"),
            ('("diagnostic_flags", [flag or NOT_APPLICABLE for flag in flags],',
             '("diagnostic_flags", [NOT_APPLICABLE for flag in flags],',
             "an analyzer that parses NCU diagnostic flags and then drops them is "
             "rejected"),
            ('[source["within_campaign_stability_review"] for source in sources]',
             '[NOT_APPLICABLE for source in sources]',
             "an analyzer that discards P2.4's within-campaign stability review is "
             "rejected"),
            ('"cross_campaign_cv_percent": cv_percent,',
             '"cv_percent": cv_percent,',
             "an analyzer that stops distinguishing cross-campaign from within-campaign "
             "variability is rejected"),
            ('"artifact_sha256": sibling_hashes,', '"artifact_sha256": {},',
             "an analyzer whose manifest does not bind its eight siblings is rejected"),
            ('"analysis_code_commit": provenance["analysis_code_commit"],',
             '"analysis_code_commit": records[0]["git_commit"],',
             "an analyzer that records the execution commit as the analysis-code commit "
             "is rejected"),
            ('    if provenance["worktree_clean"] is not True:',
             '    if provenance["worktree_clean"] is None:',
             "an analyzer that accepts a dirty analysis-code worktree is rejected"),
            ('OUTPUT_ROOT_COMPONENTS = ("results", "phase4")',
             'OUTPUT_ROOT_COMPONENTS = ("results",)',
             "an analyzer whose production output root drifts from results/phase4 is "
             "rejected"),
            ('    if parts != OUTPUT_ROOT_COMPONENTS:',
             '    if parts[:0] != OUTPUT_ROOT_COMPONENTS[:0]:',
             "an analyzer that accepts an arbitrary in-repository output directory is "
             "rejected"),
            ('    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW\n'
             '    if hasattr(os, "O_CLOEXEC"):\n'
             '        flags |= os.O_CLOEXEC\n'
             '    try:\n'
             '        root_fd = os.open(str(repo_root), flags)',
             '    flags = os.O_RDONLY | os.O_DIRECTORY\n'
             '    if hasattr(os, "O_CLOEXEC"):\n'
             '        flags |= os.O_CLOEXEC\n'
             '    try:\n'
             '        root_fd = os.open(str(repo_root), flags)',
             "an analyzer that opens the output tree without O_NOFOLLOW is rejected"),
            ('f"the vertical line is the min-max whisker over exactly {CAMPAIGN_COUNT} "',
             'f"the bar is min..max over exactly {CAMPAIGN_COUNT} "',
             "an analyzer that calls the min-max whisker a bar is rejected"),
            ('ACCEPTANCE_SCHEMA_VERSION = "p43.acceptance.v1"',
             'ACCEPTANCE_SCHEMA_VERSION = "p43.acceptance.v2"',
             "an analyzer whose acceptance schema drifts from the frozen version is "
             "rejected"),
            ('PUBLICATION_STATE = "immutable_candidate_requires_external_attestation"',
             'PUBLICATION_STATE = "accepted"',
             "an analyzer that promotes its candidate publication state is rejected"),
        ):
            if old not in original:
                reporter.check(label, False, f"{old!r} not found in the analyzer")
                continue
            rewrite(analyzer_path, original.replace(old, new, 1))
            try:
                reporter.check(label, check_repository(clone) == 1, "")
            finally:
                rewrite(analyzer_path, original)

        # Documentation regressions.
        mutate("results/README.md",
               "All nine artifacts record schema version",
               "Every file carries schema version",
               "a results/README.md that claims each individual file embeds all metadata "
               "is rejected")
        mutate("results/README.md", "not a standalone provenance envelope",
               "a complete standalone provenance envelope",
               "a results/README.md that calls a detached CSV a provenance envelope is "
               "rejected", every=True)
        mutate(P43_PROTOCOL_RELATIVE_PATH, "p43.acceptance.v1", "p43.acceptance.v9",
               "a protocol whose frozen acceptance schema disagrees with the analyzer is "
               "rejected", every=True)
        mutate(P43_PROTOCOL_RELATIVE_PATH, ACCEPTED_AUDIT_VERDICT,
               "ACCEPT WITH NO OBSERVATIONS",
               "a protocol that overstates the independent audit verdict is rejected",
               every=True)
        mutate(P43_PROTOCOL_RELATIVE_PATH, ACCEPTED_MANIFEST_SHA256, "0" * 64,
               "a protocol whose recorded manifest digest is not the accepted one is "
               "rejected", every=True)
        mutate(P43_PROTOCOL_RELATIVE_PATH, ACCEPTED_CANDIDATE_COMMIT,
               ACCEPTED_ANALYZER_COMMIT,
               "a protocol that collapses the candidate commit onto the analyzer commit "
               "is rejected", every=True)

        # The real acceptance attestation. P4.3 is accepted, so it must exist,
        # parse, and bind exactly the committed bytes. Every way it could be
        # wrong is exercised against the clone and then restored.
        acceptance = clone / P43_ACCEPTANCE_RELATIVE_PATH
        accepted_bytes = acceptance.read_text(encoding="utf-8")
        accepted = json.loads(accepted_bytes)

        def with_acceptance(payload: str, label: str) -> None:
            acceptance.write_text(payload, encoding="utf-8")
            try:
                reporter.check(label, check_repository(clone) == 1, "")
            finally:
                acceptance.write_text(accepted_bytes, encoding="utf-8")

        acceptance.unlink()
        try:
            reporter.check("a repository with no acceptance attestation is rejected while "
                           "P4.3 records the accepted state",
                           check_repository(clone) == 1, "")
        finally:
            acceptance.write_text(accepted_bytes, encoding="utf-8")
        with_acceptance("{not json", "an acceptance attestation that is not JSON is "
                                     "rejected")
        with_acceptance("[]\n", "an acceptance attestation that is not a JSON object is "
                                "rejected")
        for label, mutate_document in (
                ("a wrong manifest digest",
                 lambda doc: doc.__setitem__("analysis_manifest_sha256", "0" * 64)),
                ("a truncated manifest digest",
                 lambda doc: doc.__setitem__("analysis_manifest_sha256",
                                             ACCEPTED_MANIFEST_SHA256[:63])),
                ("an upper-case manifest digest",
                 lambda doc: doc.__setitem__("analysis_manifest_sha256",
                                             ACCEPTED_MANIFEST_SHA256.upper())),
                ("a wrong analyzer commit",
                 lambda doc: doc.__setitem__("analysis_code_commit", "f" * 40)),
                ("an abbreviated analyzer commit",
                 lambda doc: doc.__setitem__("analysis_code_commit",
                                             ACCEPTED_ANALYZER_COMMIT[:12])),
                ("the candidate commit substituted for the analyzer commit",
                 lambda doc: doc.__setitem__("analysis_code_commit",
                                             ACCEPTED_CANDIDATE_COMMIT)),
                ("the execution commit substituted for the analyzer commit",
                 lambda doc: doc.__setitem__("analysis_code_commit",
                                             ACCEPTED_EXECUTION_COMMIT)),
                ("one wrong artifact digest",
                 lambda doc: doc["artifact_sha256"].__setitem__("report.md", "0" * 64)),
                ("a missing artifact binding",
                 lambda doc: doc["artifact_sha256"].pop("report.md")),
                ("a missing required field",
                 lambda doc: doc.pop("verification_outcome")),
                ("an extra top-level field",
                 lambda doc: doc.__setitem__("closure_commit", "0" * 40)),
                ("a non-boolean acceptance flag",
                 lambda doc: doc.__setitem__("accepted_for_publication", "true")),
                ("a withdrawn status", lambda doc: doc.__setitem__("status", "PENDING")),
                ("a substituted population",
                 lambda doc: doc.__setitem__("final_campaign_ids", [])),
                ("a reordered population",
                 lambda doc: doc.__setitem__(
                     "final_campaign_ids",
                     list(reversed(accepted["final_campaign_ids"])))),
                ("the pilot promoted into the population",
                 lambda doc: doc.__setitem__("pilot_campaign_id_excluded",
                                             accepted["final_campaign_ids"][0])),
                ("an unverified byte-for-byte outcome",
                 lambda doc: doc.__setitem__("verification_outcome", "not_verified")),
                ("an unreviewed output review outcome",
                 lambda doc: doc.__setitem__("independent_output_review_outcome",
                                             "review_skipped")),
                ("a drifted schema version",
                 lambda doc: doc.__setitem__("schema_version", "p43.acceptance.v9")),
                ("a foreign unit", lambda doc: doc.__setitem__("unit", "P4.2"))):
            document = json.loads(accepted_bytes)
            mutate_document(document)
            with_acceptance(json.dumps(document, indent=2) + "\n",
                            f"an acceptance attestation with {label} is rejected")

        reporter.check("the restored clone passes again", check_repository(clone) == 0, "")

    reporter.check("the frozen status frontier records P4.3 as accepted and closed",
                   EXPECTED_STATUS_LINES[2].endswith("| YES | YES | YES |"),
                   EXPECTED_STATUS_LINES[2])
    reporter.check("no P4.3 row other than the accepted one is legal",
                   len(FORBIDDEN_P43_STATUS_LINES) == 7
                   and EXPECTED_STATUS_LINES[2] not in FORBIDDEN_P43_STATUS_LINES, "")
    reporter.check("this checker requires no raw evidence path at all",
                   not any("results/raw" in value for value in REQUIRED_P43_FILES), "")

    if reporter.failures:
        print(f"check_phase4_integration_p43: self-test: FAILED "
              f"({len(reporter.failures)} check(s))", file=sys.stderr)
        return 1
    print("check_phase4_integration_p43: self-test: OK")
    return 0


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_phase4_integration_p43.py",
        description="Fail-closed, GPU-free repository-contract checker for P4.3: the "
                    "offline integrated analysis over the frozen Phase 4 population.",
    )
    parser.add_argument("repo_root", nargs="?", default=None,
                        help="repository root to check against the frozen P4.3 contract")
    parser.add_argument("--self-test", action="store_true",
                        help="run the focused synthetic suite and exit; standalone only")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        if args.repo_root is not None:
            print("check_phase4_integration_p43: --self-test is standalone", file=sys.stderr)
            return 2
        return run_self_test()
    if args.repo_root is None:
        print("check_phase4_integration_p43: a repository root or --self-test is required",
              file=sys.stderr)
        return 2
    return check_repository(Path(args.repo_root))


if __name__ == "__main__":
    sys.exit(main())
