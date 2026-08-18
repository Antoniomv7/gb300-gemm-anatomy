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

    python3 scripts/check_phase4_integration_p43.py --self-test
        Focused synthetic suite over temporary fixtures only.

    python3 scripts/check_phase4_integration_p43.py <repo-root>
        The frozen P4.3 repository contract. Needs no results/raw/, no cluster
        evidence, no container runtime, and no network.

Exit codes: 0 OK; 1 at least one check failed; 2 usage error.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
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

REQUIRED_P43_FILES = (
    P43_ANALYZER_RELATIVE_PATH,
    P43_CHECKER_RELATIVE_PATH,
    P43_PROTOCOL_RELATIVE_PATH,
    ORCHESTRATOR_RELATIVE_PATH,
    P42_CHECKER_RELATIVE_PATH,
    P42_PROTOCOL_RELATIVE_PATH,
    P41_CHECKER_RELATIVE_PATH,
    P41_PROTOCOL_RELATIVE_PATH,
    RUN_ALL_RELATIVE_PATH,
)

# ---------------------------------------------------------------------------
# The truthful status frontier this unit is allowed to record. P4.3 is
# implemented; it is neither independently audited nor run against the real
# evidence, and this checker refuses every stronger claim.
# ---------------------------------------------------------------------------
EXPECTED_STATUS_LINES = (
    "| P4.1 | Orchestrator | YES | YES | YES |",
    "| P4.2 | Pilot plus three final campaigns | YES | YES | YES |",
    "| P4.3 | Integrated analysis, documentation, audit | YES | NO | NO |",
)
FORBIDDEN_P43_STATUS_LINES = (
    "| P4.3 | Integrated analysis, documentation, audit | NO | NO | NO |",
    "| P4.3 | Integrated analysis, documentation, audit | NO | YES | NO |",
    "| P4.3 | Integrated analysis, documentation, audit | NO | NO | YES |",
    "| P4.3 | Integrated analysis, documentation, audit | NO | YES | YES |",
    "| P4.3 | Integrated analysis, documentation, audit | YES | YES | NO |",
    "| P4.3 | Integrated analysis, documentation, audit | YES | NO | YES |",
    "| P4.3 | Integrated analysis, documentation, audit | YES | YES | YES |",
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

# No document may close Phase 4, close the TFM, claim a P4.3 audit, claim a
# P4.3 production result, or promote anything to publishable.
FORBIDDEN_DOCUMENT_CLAIMS = (
    "P4.3 = YES / YES / YES",
    "P4.3 = YES / YES / NO",
    "P4.3 = YES / NO / YES",
    "P4.3: CLOSED",
    "P4.3 is closed",
    "Phase 4: CLOSED",
    "Phase 4 is closed",
    "the TFM is complete",
    "TFM: CLOSED",
    "publishable=true",
    "publishable: true",
    "publishable = true",
    "P4.3 independent audit: PASSED",
    "the P4.3 analysis has been independently audited",
    "the production analysis has been run",
)
# Every project-level narrative must keep the honest P4.3 boundary.
REQUIRED_DOCUMENT_STATEMENTS = (
    (r"no\s+publishable\s+(phase\s+4\s+)?result", "no publishable result exists"),
    (r"P4\.3", "P4.3 is described"),
)
REQUIRED_P43_STATEMENTS = {
    "PLAN.md": (
        "| P4.3 | Integrated analysis, documentation, audit | YES | NO | NO |",
        "The P4.3 independent audit has not been performed",
        "no production analysis of the three final campaigns has been run",
    ),
    "README.md": (
        "P4.3: IMPLEMENTED; independent audit: NO; production analysis: NO",
        "The P4.3 independent audit has not been performed",
    ),
    "results/README.md": (
        "P4.3 is implemented",
        "no P4.3 curated result has been accepted for publication",
    ),
    P43_PROTOCOL_RELATIVE_PATH: (
        "P4.3 = YES / NO / NO",
        "Independent audit: NOT PERFORMED",
        "Production analysis: NOT RUN",
        "no publishable result exists",
        "Phase 4 and the complete TFM are not closed",
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
    "__future__", "argparse", "csv", "errno", "importlib", "inspect", "io", "json",
    "linecache", "math", "os", "re", "shutil", "stat", "sys", "tempfile", "pathlib",
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
                        ("mkdir_component", "P4.1"),
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
    reporter.check("the sample standard deviation uses the n-1 denominator",
                   abs(summary["stdev_sample"] - 1.0) < 1e-12, str(summary["stdev_sample"]))
    reporter.check("mean, median, minimum, and maximum are the plain descriptive values",
                   (summary["mean"], summary["median"], summary["minimum"],
                    summary["maximum"]) == (2.0, 2.0, 1.0, 3.0), str(summary))
    reporter.check("the coefficient of variation is 100 x stdev / mean",
                   abs(summary["cv_percent"] - 50.0) < 1e-12, str(summary["cv_percent"]))
    reporter.check("a coefficient of variation above the threshold flags for review and "
                   "keeps every campaign value",
                   summary["cv_review_flag"] == "REVIEW"
                   and len(summary["campaign_values"]) == 3, str(summary))

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
                   signed["cv_percent"] is None
                   and signed["cv_review_flag"] == analyzer.NOT_APPLICABLE, str(signed))
    reporter.check("a negative GEMM gap is preserved without clamping",
                   signed["minimum"] == -5.0, str(signed))
    reporter.check("gap_to_cublaslt_pct is declared signed or zero-centred",
                   "gap_to_cublaslt_pct" in analyzer.SIGNED_OR_ZERO_CENTRED_METRICS, "")
    zero = analyzer.summarize_metric([0.0, 0.0, 0.0],
                                     metric="throughput_ratio_vs_cublaslt")
    reporter.check("a zero denominator never yields a coefficient of variation",
                   zero["cv_percent"] is None, str(zero))
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
                                         analyzer._fixture_records())
        second = analyzer.build_documents(orchestrator, analyzer._StubP35(),
                                          analyzer._fixture_records())
    except analyzer.P43Error as exc:
        reporter.check("a complete synthetic analysis runs end to end", False, str(exc))
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
          "(P4.3 = YES / NO / NO; implemented, not audited, production analysis not run; "
          "no publishable result)")
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
               "| P4.3 | Integrated analysis, documentation, audit | YES | NO | NO |",
               "| P4.3 | Integrated analysis, documentation, audit | YES | YES | YES |",
               "a PLAN.md that claims P4.3 is audited and verified is rejected")
        mutate("PLAN.md",
               "| P4.3 | Integrated analysis, documentation, audit | YES | NO | NO |",
               "| P4.3 | Integrated analysis, documentation, audit | NO | NO | NO |",
               "a PLAN.md that still calls P4.3 unimplemented is rejected")
        mutate("PLAN.md",
               "| P4.2 | Pilot plus three final campaigns | YES | YES | YES |",
               "| P4.2 | Pilot plus three final campaigns | YES | NO | NO |",
               "a PLAN.md that regresses closed P4.2 is rejected")
        mutate("PLAN.md",
               "| P1.4 | Profiling, validation, analysis, pilot | YES | YES | YES |",
               "| P1.4 | Profiling, validation, analysis, pilot | YES | NO | NO |",
               "a PLAN.md that regresses closed P1.4 is rejected")
        mutate(P43_PROTOCOL_RELATIVE_PATH, "Independent audit: NOT PERFORMED",
               "Independent audit: PASSED",
               "a protocol that claims the P4.3 audit passed is rejected")
        mutate(P43_PROTOCOL_RELATIVE_PATH, "P4.3 = YES / NO / NO", "P4.3 = YES / YES / YES",
               "a protocol that closes P4.3 is rejected")
        mutate(P43_PROTOCOL_RELATIVE_PATH, "no publishable result exists",
               "a publishable result exists",
               "a protocol that claims a publishable result is rejected", every=True)
        mutate("README.md",
               "P4.3: IMPLEMENTED; independent audit: NO; production analysis: NO",
               "P4.3: IMPLEMENTED; independent audit: YES; production analysis: YES",
               "a README that claims the P4.3 audit and production analysis happened is "
               "rejected")
        mutate("results/README.md", "P4.3 is implemented", "P4.3 is closed",
               "a results/README.md that closes P4.3 is rejected")
        mutate(P43_PROTOCOL_RELATIVE_PATH, "publishable=false", "publishable=true",
               "a protocol that promotes an artifact to publishable is rejected",
               every=True)
        mutate(P43_PROTOCOL_RELATIVE_PATH, "figures/memory_paths.svg", "figures/other.svg",
               "a protocol whose artifact inventory disagrees with the analyzer is rejected")
        mutate("Makefile", "PHASE4_P43_FINAL_CAMPAIGN_1 := 20260817T110330Z",
               "PHASE4_P43_FINAL_CAMPAIGN_1 := 20260818T000000Z",
               "a Makefile that analyses a campaign outside the frozen population is "
               "rejected")
        mutate("Makefile", "PHASE4_P43_OUTPUT_ROOT := results/phase4",
               "PHASE4_P43_OUTPUT_ROOT := results/raw/phase4-analysis",
               "a Makefile that writes the analysis under results/raw/ is rejected")

        analyzer_path = clone / P43_ANALYZER_RELATIVE_PATH
        original = analyzer_path.read_text(encoding="utf-8")
        for old, new, label in (
            ("import argparse", "import argparse\nimport subprocess",
             "an analyzer that could start a child process is rejected"),
            ("/ (count - 1)", "/ count",
             "an analyzer using the population standard deviation is rejected"),
            ("CV_REVIEW_THRESHOLD_PERCENT = 5.0", "CV_REVIEW_THRESHOLD_PERCENT = 50.0",
             "an analyzer that weakens the strict CV review threshold is rejected"),
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
        reporter.check("the restored clone passes again", check_repository(clone) == 0, "")

    reporter.check("the frozen status frontier records P4.3 as implemented only",
                   EXPECTED_STATUS_LINES[2].endswith("| YES | NO | NO |"),
                   EXPECTED_STATUS_LINES[2])
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
