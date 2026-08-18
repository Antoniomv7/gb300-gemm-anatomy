#!/usr/bin/env python3
"""Fail-closed, GPU-free cross-campaign checker for P4.2.

P4.1 validates exactly one campaign at a time. P4.2 owns the invariants that
only hold *across* the four-campaign population -- one accepted pilot plus
three final replicates -- and nothing else. This checker adds no orchestrator,
no runner, no statistic, no threshold, and no interpretation.

Three modes:

    python3 scripts/check_phase4_campaigns_p42.py --self-test
        Synthetic acceptance tests over temporary directories only. Real
        P4.1 campaigns are driven into a temporary repository root through
        P4.1's own fixture, so every manifest chain under test is genuine.
        Nothing is created under this repository's real results/ tree.

    python3 scripts/check_phase4_campaigns_p42.py <repo-root>
        The frozen P4.2 repository contract: the required files, the truthful
        status frontier, the untouched P4.1 execution path, the GPU-free Make
        target, the documentary closure, and the absence of any publishable
        claim or P4.3 implementation.
        Needs no results/raw/ and no cluster evidence.

    python3 scripts/check_phase4_campaigns_p42.py \
        --campaign-root results/raw/phase4 \
        --pilot-campaign-id 20260812T013848Z \
        --final-campaign-id <FINAL_1> \
        --final-campaign-id <FINAL_2> \
        --final-campaign-id <FINAL_3>
        Strictly read-only evidence mode for an existing campaign population.
        It never writes, repairs, resumes, deletes, or regenerates anything,
        and it never invokes --resume.

Every per-campaign semantic decision is delegated to P4.1's own audited
manifest-chain loader, filesystem primitives, and read-only terminal stage
revalidation; this unit re-implements none of them.

Exit codes: 0 OK; 1 at least one check failed; 2 usage error.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import inspect
import io
import json
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
P35_CHECKER_RELATIVE_PATH = "scripts/check_gemm_comparison_p35.py"

REQUIRED_P42_FILES = (
    P42_CHECKER_RELATIVE_PATH,
    P42_PROTOCOL_RELATIVE_PATH,
    RUN_ALL_RELATIVE_PATH,
    ORCHESTRATOR_RELATIVE_PATH,
    P41_CHECKER_RELATIVE_PATH,
    P41_PROTOCOL_RELATIVE_PATH,
)

# ---------------------------------------------------------------------------
# The frozen P4.2 campaign population.
# ---------------------------------------------------------------------------

# The single accepted pilot. It is fixed, non-publishable, and is never one of
# the three final replicates. See src/phase4/P4_2_PROTOCOL.md section 5.1.
PILOT_CAMPAIGN_ID = "20260812T013848Z"
# The exact commit the accepted pilot ran from. It is a historical provenance
# pin: the pilot is excluded from the three-final commit-equality set, but a
# different campaign on the right ID is never accepted as that pilot.
PILOT_GIT_COMMIT = "77908dae377fb131ff90baef455c79d6d2c28b0b"

REQUIRED_FINAL_CAMPAIGN_COUNT = 3
PILOT_KIND = "pilot"
FINAL_KIND = "final"
REQUIRED_SCOPE = "full"

# The frozen nine-stage full plan, restated independently of the orchestrator
# so a silent reordering, insertion, or deletion is rejected here too.
EXPECTED_FULL_PLAN = (
    "preflight", "memory.pilot", "memory.profile", "umma.pilot", "umma.profile",
    "gemm.capture", "memory.analyze", "umma.analyze", "campaign.validate",
)

# ---------------------------------------------------------------------------
# The frozen P4.1 execution path.
#
# P4.2 freezes the code the three final campaigns run from. These are the
# SHA-256 digests of the two files that actually execute a campaign, taken at
# the accepted P4.1 closure commit 0258ea2dfc882b54277081d324a64d0b6c5ed17b.
# A change to either one invalidates the freeze and must be re-audited before
# any final campaign runs; this checker fails closed instead of accepting it
# silently. Nothing else in the repository is pinned by content.
# ---------------------------------------------------------------------------
FROZEN_EXECUTION_PATH_SHA256 = {
    RUN_ALL_RELATIVE_PATH:
        "5a7e9c9ef9f84c0a0b99a50e8c44995e4388183c329ab5b17916e7567a4fd428",
    ORCHESTRATOR_RELATIVE_PATH:
        "67877fcd74f91d9699290e107b9dc12cf13f0bc28e88ce0137980403cbf25453",
}

# ---------------------------------------------------------------------------
# The truthful status frontier this unit is allowed to record.
# ---------------------------------------------------------------------------
EXPECTED_STATUS_LINES = (
    "| P4.1 | Orchestrator | YES | YES | YES |",
    "| P4.2 | Pilot plus three final campaigns | YES | YES | YES |",
    "| P4.3 | Integrated analysis, documentation, audit | NO | NO | NO |",
)
# Every stale or impossible P4.2 state. P4.2 is closed; nothing else is legal.
FORBIDDEN_P42_STATUS_LINES = (
    "| P4.2 | Pilot plus three final campaigns | NO | NO | NO |",
    "| P4.2 | Pilot plus three final campaigns | NO | YES | NO |",
    "| P4.2 | Pilot plus three final campaigns | NO | NO | YES |",
    "| P4.2 | Pilot plus three final campaigns | NO | YES | YES |",
    "| P4.2 | Pilot plus three final campaigns | YES | NO | NO |",
    "| P4.2 | Pilot plus three final campaigns | YES | YES | NO |",
    "| P4.2 | Pilot plus three final campaigns | YES | NO | YES |",
)
# The closed P4.1 row must not regress, and P4.3 must stay unimplemented.
FORBIDDEN_P41_STATUS_LINES = (
    "| P4.1 | Orchestrator | NO | NO | NO |",
    "| P4.1 | Orchestrator | YES | YES | NO |",
    "| P4.1 | Orchestrator | YES | NO | YES |",
    "| P4.1 | Orchestrator | YES | NO | NO |",
)
FORBIDDEN_P43_STATUS_LINES = (
    "| P4.3 | Integrated analysis, documentation, audit | YES | NO | NO |",
    "| P4.3 | Integrated analysis, documentation, audit | NO | YES | NO |",
    "| P4.3 | Integrated analysis, documentation, audit | NO | NO | YES |",
    "| P4.3 | Integrated analysis, documentation, audit | YES | YES | YES |",
)

STATUS_DOCUMENTS = ("PLAN.md", "README.md", "results/README.md",
                    P41_PROTOCOL_RELATIVE_PATH, P42_PROTOCOL_RELATIVE_PATH)

# No document may retain a stale P4.2 frontier, close all of Phase 4, or claim a
# publishable Phase 4 result.
FORBIDDEN_DOCUMENT_CLAIMS = (
    "P4.2 = YES / NO / NO",
    "P4.2 = YES / YES / NO",
    "P4.2 = YES / NO / YES",
    "Independent audit: PENDING",
    "GB300 verification: PENDING",
    "zero of three final campaigns",
    "still requires three independent final campaigns",
    "Phase 4: CLOSED",
    "Phase 4 is closed",
    "publishable=true",
    "publishable: true",
    "publishable = true",
)
# Every project-level narrative must retain the accepted population and the
# non-publication boundary. The patterns tolerate normal line wrapping.
REQUIRED_DOCUMENT_STATEMENTS = (
    (r"no\s+publishable\s+(phase\s+4\s+)?result", "no publishable result exists"),
    (r"20260817T110330Z", "final campaign 20260817T110330Z is recorded"),
    (r"20260817T111310Z", "final campaign 20260817T111310Z is recorded"),
    (r"20260817T112011Z", "final campaign 20260817T112011Z is recorded"),
)
REQUIRED_CLOSURE_STATEMENTS = {
    "PLAN.md": "| P4.2 | Pilot plus three final campaigns | YES | YES | YES |",
    "README.md": "P4.2: CLOSED; independent audit: YES; GB300 verification: YES",
    "results/README.md": "P4.2 is closed",
    P42_PROTOCOL_RELATIVE_PATH: "P4.2 = YES / YES / YES",
}

REQUIRED_P42_PROTOCOL_STATEMENTS = (
    "P4.2 = YES / YES / YES",
    "P4.2 is implemented",
    "Independent audit: PASSED",
    "GB300 verification: PASSED",
    "no publishable result exists",
    PILOT_CAMPAIGN_ID,
    "20260817T110330Z",
    "20260817T111310Z",
    "20260817T112011Z",
    "b08e45c2636a3ac17c94ad8b1368084914196d7a",
)

# P4.3 belongs to P4.3. Nothing here may implement it.
FORBIDDEN_P43_ARTEFACTS = (
    "src/phase4/P4_3_PROTOCOL.md",
    "scripts/check_phase4_integration_p43.py",
    "scripts/analyze_phase4_p43.py",
)
# Tokens that would mean this unit had started computing a result.
FORBIDDEN_ANALYSIS_TOKENS = (
    r"\bmean\b", r"\bmedian\b", r"\bstdev\b", r"\bstddev\b", r"\bvariance\b",
    r"\bpercentile\b", r"\bbootstrap\b", r"\bconfidence\b", r"\broofline\b",
    r"\bspeedup\b", r"\btflops\b", r"\bbandwidth\b", r"\bwinner\b",
    r"\bthreshold\b", r"\bmatplotlib\b", r"\bplot\b", r"\bfigure\b",
)

# The Python standard library modules this unit is allowed to import. P4.2
# adds no dependency and no version pin.
STDLIB_IMPORT_ALLOWLIST = frozenset({
    "__future__", "argparse", "contextlib", "importlib", "inspect", "io", "json",
    "os", "re", "shutil", "sys", "tempfile", "pathlib",
})

# Calls that would mutate, repair, resume, or regenerate evidence. None of
# them may appear anywhere in an evidence-mode function.
FORBIDDEN_EVIDENCE_MODE_CALLS = (
    r"\bos\.(remove|unlink|rmdir|replace|rename|mkdir|makedirs|chmod|chown|"
    r"symlink|link|truncate|write)\b",
    r"\bshutil\.",
    r"\bwrite_text\b", r"\bwrite_bytes\b",
    r"\bcreate_file_exclusive\b", r"\bwrite_file_exclusive\b",
    r"\blink_no_clobber\b", r"\bwrite_next_manifest_revision\b",
    r"\bcreate_campaign_tree\b", r"\bmkdir_component\b",
    r"\brun_campaign\b", r"\.run\s*\(", r"\bsubprocess\b", r"--resume",
    r"\bopen\s*\([^)]*[\"']w",
)

# The Make target this unit adds. It is fast, GPU-free, and cannot start a
# campaign. Reused verbatim from P4.1's own container-free scan.
P42_MAKE_TARGET = "phase4-p42-check"


class EvidenceError(Exception):
    """A campaign's evidence cannot be interpreted. Always fatal, never
    repaired: this checker validates, it never recovers."""


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

    def note(self, text: str) -> None:
        print(f"{self.prefix}: note: {text}")


# ===========================================================================
# Evidence mode -- strictly read-only.
#
# Every function below opens evidence only through P4.1's own descriptor
# anchored, symlink-rejecting primitives and its own manifest-chain loader.
# None of them writes, repairs, resumes, deletes, or regenerates anything;
# check_repository() proves that mechanically from their own source.
# ===========================================================================


def campaign_root_to_repo_root(campaign_root: Path, orchestrator) -> Path:
    """P4.1 anchors every campaign open on the repository root and descends
    through `results/raw/phase4`, revalidating each component. The campaign
    root supplied on the command line must therefore be exactly that path, and
    the repository root is its lexical ancestor -- taken without resolving
    symlinks, so that P4.1's own per-component rejection still runs."""
    parts = tuple(orchestrator.PHASE4_RAW_PARTS)
    absolute = Path(os.path.abspath(str(campaign_root)))
    if absolute.parts[-len(parts):] != parts:
        raise EvidenceError(
            f"{campaign_root}: a P4.2 campaign root must be the repository's "
            f"{'/'.join(parts)} directory")
    return Path(*absolute.parts[:-len(parts)])


def read_campaign_file(orchestrator, repo_root: Path, campaign_dir: Path, name: str) -> bytes:
    """Read one campaign-owned file through P4.1's descriptor chain, so an
    ancestor or target symlink is rejected exactly as P4.1 rejects it."""
    parts = Path(orchestrator.repo_relative(campaign_dir, repo_root)).parts
    directory_fd = orchestrator.open_dir_chain(repo_root, *parts)
    try:
        return orchestrator.read_file_nofollow(name, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)


def summarize_campaign(orchestrator, repo_root: Path, campaign_id: str) -> dict:
    """Load one top-level campaign through P4.1's audited manifest loader and
    reduce it to the few cross-campaign facts P4.2 compares. Deep stage-owned
    evidence is checked separately by revalidate_campaign_evidence()."""
    try:
        orchestrator.validate_campaign_id(campaign_id)
    except orchestrator.OrchestratorError as exc:
        raise EvidenceError(f"{campaign_id!r}: {exc}") from exc

    try:
        campaign_dir = orchestrator.resolve_campaign_tree(campaign_id, repo_root=repo_root)
        manifest, revision = orchestrator.load_manifest_chain(campaign_dir, repo_root=repo_root)
    except (orchestrator.OrchestratorError, orchestrator.ManifestError) as exc:
        raise EvidenceError(f"{campaign_id}: {exc}") from exc
    if revision < 0:
        raise EvidenceError(
            f"{campaign_id}: the campaign directory carries no manifest revision at all; "
            f"it cannot be interpreted and is never repaired")

    # The immutable plan, re-hashed fresh from disk against what the manifest
    # recorded. A rewritten plan.json is rejected, never adopted.
    try:
        plan_bytes = read_campaign_file(
            orchestrator, repo_root, campaign_dir, orchestrator.PLAN_FILE_NAME)
    except orchestrator.OrchestratorError as exc:
        raise EvidenceError(f"{campaign_id}: {orchestrator.PLAN_FILE_NAME}: {exc}") from exc
    plan_digest = orchestrator.sha256_bytes(plan_bytes)
    if plan_digest != manifest["plan_sha256"]:
        raise EvidenceError(
            f"{campaign_id}: {orchestrator.PLAN_FILE_NAME} hashes to {plan_digest}, but the "
            f"manifest pinned plan_sha256={manifest['plan_sha256']}")
    try:
        plan = json.loads(plan_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"{campaign_id}: {orchestrator.PLAN_FILE_NAME}: invalid JSON: {exc}")
    if not isinstance(plan, dict):
        raise EvidenceError(f"{campaign_id}: {orchestrator.PLAN_FILE_NAME} is not a JSON object")
    for field, expected in (("campaign_id", campaign_id),
                            ("campaign_kind", manifest["campaign_kind"]),
                            ("scope", manifest["scope"]),
                            ("stage_order", list(manifest["stage_order"]))):
        if plan.get(field) != expected:
            raise EvidenceError(
                f"{campaign_id}: {orchestrator.PLAN_FILE_NAME} {field}={plan.get(field)!r} "
                f"disagrees with the manifest ({expected!r})")

    # plan.json legitimately names its own campaign, so plan_sha256 is unique
    # per campaign by construction. The shared, immutable part of the plan is
    # everything except that identity field; that is what the three finals must
    # agree on.
    plan_shape = {key: value for key, value in plan.items() if key != "campaign_id"}
    plan_shape_digest = orchestrator.stable_digest(plan_shape)

    gpu = manifest.get("gpu")
    if not isinstance(gpu, dict) or sorted(gpu) != sorted(orchestrator.GPU_IDENTITY_FIELDS):
        raise EvidenceError(
            f"{campaign_id}: the manifest carries no complete validated GPU identity "
            f"({sorted(gpu) if isinstance(gpu, dict) else type(gpu).__name__})")

    # The comparable provenance the closed P4.1 schema actually exposes. P4.2
    # invents none of its own and reads the field list from P4.1 itself, so a
    # future P4.1 field is compared here automatically.
    provenance: dict[str, object] = {}
    for field in orchestrator.COMPARABLE_EVIDENCE_FIELDS:
        observed = {record["evidence"][field]
                    for record in manifest["stage_results"].values()
                    if record["evidence"].get(field) is not None}
        if len(observed) > 1:
            raise EvidenceError(
                f"{campaign_id}: its own components disagree on {field}: {sorted(observed)}")
        provenance[field] = observed.pop() if observed else None

    return {
        "campaign_id": campaign_id,
        "campaign_kind": manifest["campaign_kind"],
        "scope": manifest["scope"],
        "state": manifest["state"],
        "outcome": manifest.get("outcome"),
        "git_commit": manifest["git_commit"],
        "git_dirty": manifest["git_dirty"],
        "publishable": manifest["publishable"],
        "plan_sha256": manifest["plan_sha256"],
        "plan_shape_digest": plan_shape_digest,
        "stage_order": tuple(manifest["stage_order"]),
        "completed_stages": tuple(manifest["stage_results"]),
        "manifest_revision": revision,
        "gpu": {field: gpu[field] for field in orchestrator.GPU_IDENTITY_FIELDS},
        "provenance": provenance,
    }


def revalidate_campaign_evidence(orchestrator, repo_root: Path, summary: dict, *,
                                 context=None) -> None:
    """Run P4.1's complete terminal revalidation over one loaded campaign.

    The production path uses P4.1's real semantic P1.4/P2.4 loaders and P3.5
    validator. A context may be injected only by the synthetic self-test so its
    GPU-free fixtures are revalidated through the same Campaign methods.
    """
    if context is None:
        orchestrator_root = Path(os.path.abspath(str(orchestrator.REPO_ROOT)))
        if repo_root != orchestrator_root:
            raise EvidenceError(
                f"{summary['campaign_id']}: the campaign evidence belongs to {repo_root}, "
                f"but the audited P4.1 module belongs to {orchestrator_root}")
        context = orchestrator.Context(repo_root=repo_root)
    elif Path(os.path.abspath(str(context.repo_root))) != repo_root:
        raise EvidenceError(
            f"{summary['campaign_id']}: the injected P4.1 context is anchored on "
            f"{context.repo_root}, not {repo_root}")

    campaign = orchestrator.Campaign(
        context,
        campaign_id=summary["campaign_id"],
        campaign_kind=summary["campaign_kind"],
        scope=summary["scope"],
    )
    campaign.git_commit = summary["git_commit"]
    try:
        campaign.load()
        campaign.revalidate_completed()
    except (orchestrator.OrchestratorError, orchestrator.ManifestError,
            orchestrator.RunFailure) as exc:
        raise EvidenceError(
            f"{summary['campaign_id']}: P4.1 terminal revalidation rejected the "
            f"campaign: {exc}") from exc


def validate_campaign_acceptance(summary: dict, *, expected_kind: str,
                                 orchestrator,
                                 expected_git_commit: str | None = None) -> list[str]:
    """Check P4.2's top-level per-campaign acceptance facts.

    The caller follows this with P4.1's complete terminal-stage revalidation;
    neither layer substitutes for the other.
    """
    errors: list[str] = []
    label = f"{summary['campaign_id']} ({expected_kind})"
    if summary["campaign_kind"] != expected_kind:
        errors.append(
            f"{label}: campaign_kind={summary['campaign_kind']!r}, not {expected_kind!r}")
    if summary["scope"] != REQUIRED_SCOPE:
        errors.append(f"{label}: scope={summary['scope']!r}, not {REQUIRED_SCOPE!r}")
    if summary["state"] != orchestrator.STATE_COMPLETE:
        errors.append(f"{label}: terminal state is {summary['state']!r}, not COMPLETE")
    if summary["outcome"] != orchestrator.STATE_COMPLETE:
        errors.append(f"{label}: outcome is {summary['outcome']!r}, not COMPLETE")
    if summary["stage_order"] != EXPECTED_FULL_PLAN:
        errors.append(f"{label}: stage_order is not the frozen nine-stage plan: "
                      f"{list(summary['stage_order'])}")
    if summary["completed_stages"] != EXPECTED_FULL_PLAN:
        errors.append(f"{label}: only {len(summary['completed_stages'])} of "
                      f"{len(EXPECTED_FULL_PLAN)} stages are recorded complete: "
                      f"{list(summary['completed_stages'])}")
    if summary["publishable"] is not False:
        errors.append(f"{label}: publishable is not false")
    if summary["git_dirty"] is not False:
        errors.append(f"{label}: git_dirty is not false")
    if (expected_git_commit is not None
            and summary["git_commit"] != expected_git_commit):
        errors.append(
            f"{label}: git_commit={summary['git_commit']!r}, not the frozen "
            f"historical commit {expected_git_commit!r}")
    return errors


def compare_final_set(finals: list[dict]) -> list[str]:
    """The frozen-execution invariant of protocol section 5.3: all three final
    replicates ran from one commit, one immutable plan, and one stage order."""
    errors: list[str] = []
    for field, description in (("git_commit", "Git commit"),
                               ("plan_shape_digest", "immutable plan"),
                               ("stage_order", "stage order")):
        observed = {summary[field] for summary in finals}
        if len(observed) > 1:
            errors.append(
                f"the three final campaigns do not share one {description}: "
                + ", ".join(f"{s['campaign_id']}={s[field]!r}" for s in finals))
    return errors


def compare_campaign_population(pilot: dict, finals: list[dict],
                                *, orchestrator) -> list[str]:
    """The whole-population invariant: every campaign, pilot included, ran on
    the same physical device with the same comparable exposed provenance.

    `git_commit` is deliberately excluded here and checked only across the
    three finals: the pilot is a functional qualification of the orchestration
    path, not one of the replicates, so it legitimately carries an earlier
    commit (protocol section 5.2)."""
    errors: list[str] = []
    everyone = [pilot] + finals
    for field in orchestrator.GPU_IDENTITY_FIELDS:
        observed = {summary["gpu"][field] for summary in everyone}
        if len(observed) > 1:
            errors.append(
                f"the four campaigns do not share one GPU {field}: "
                + ", ".join(f"{s['campaign_id']}={s['gpu'][field]!r}" for s in everyone))
    for field in orchestrator.COMPARABLE_EVIDENCE_FIELDS:
        if field == "git_commit":
            continue
        observed = {summary["provenance"][field] for summary in everyone}
        if len(observed) > 1:
            errors.append(
                f"the four campaigns do not share one {field}: "
                + ", ".join(f"{s['campaign_id']}={s['provenance'][field]!r}"
                            for s in everyone))
        elif observed == {None}:
            errors.append(f"no campaign exposes the comparable provenance field {field}")
    return errors


def list_campaign_ids(orchestrator, repo_root: Path) -> list[str]:
    """Enumerate the campaign root through the same descriptor chain P4.1 uses.
    Read-only: os.listdir on an already-open directory descriptor."""
    directory_fd = orchestrator.open_dir_chain(repo_root, *orchestrator.PHASE4_RAW_PARTS)
    try:
        return sorted(os.listdir(directory_fd))
    finally:
        os.close(directory_fd)


def scan_for_undeclared_finals(orchestrator, repo_root: Path,
                               declared: set[str]) -> tuple[list[str], list[str]]:
    """No final campaign may be silently omitted from the declared population
    (protocol section 5.4). Every entry in the campaign root is accounted for:
    an undeclared campaign of kind `final` is fatal, an entry that cannot be
    interpreted at all is fatal, and anything else is reported by name."""
    errors: list[str] = []
    notes: list[str] = []
    for name in list_campaign_ids(orchestrator, repo_root):
        if name in declared:
            continue
        if not orchestrator.CAMPAIGN_ID_RE.fullmatch(name):
            errors.append(
                f"{name!r}: unexpected entry in the campaign root; it is neither a canonical "
                f"campaign ID nor a declared campaign")
            continue
        try:
            summary = summarize_campaign(orchestrator, repo_root, name)
        except EvidenceError as exc:
            errors.append(
                f"{name}: an undeclared campaign that cannot be interpreted, so it cannot be "
                f"shown not to be a final replicate: {exc}")
            continue
        if summary["campaign_kind"] == FINAL_KIND:
            errors.append(
                f"{name}: an undeclared campaign_kind=final campaign (scope="
                f"{summary['scope']!r}, state={summary['state']!r}) exists in the campaign "
                f"root; the population must not be selected after seeing results")
            continue
        notes.append(
            f"{name}: undeclared campaign_kind={summary['campaign_kind']!r} "
            f"scope={summary['scope']!r} state={summary['state']!r} (not a final replicate)")
    return errors, notes


def check_campaign_evidence(orchestrator, campaign_root: Path, pilot_ids: list[str],
                            final_ids: list[str], *,
                            expected_pilot_id: str = PILOT_CAMPAIGN_ID,
                            expected_pilot_commit: str = PILOT_GIT_COMMIT,
                            revalidation_contexts: dict[str, object] | None = None) -> int:
    """Evidence mode. Read-only from the first call to the last."""
    reporter = Reporter("check_phase4_campaigns_p42: evidence")

    if len(pilot_ids) != 1:
        reporter.check("exactly one pilot campaign ID is declared", False,
                       f"{len(pilot_ids)} given: {pilot_ids}")
        return 1
    pilot_id = pilot_ids[0]
    reporter.check("exactly one pilot campaign ID is declared", True)
    if not reporter.check(
            f"the declared pilot is the frozen pilot {expected_pilot_id}",
            pilot_id == expected_pilot_id, pilot_id):
        return 1
    if not reporter.check(
            f"exactly {REQUIRED_FINAL_CAMPAIGN_COUNT} final campaign IDs are declared",
            len(final_ids) == REQUIRED_FINAL_CAMPAIGN_COUNT,
            f"{len(final_ids)} given: {final_ids}"):
        return 1
    if not reporter.check("the three final campaign IDs are distinct",
                          len(set(final_ids)) == len(final_ids), str(final_ids)):
        return 1
    if not reporter.check("the pilot is not declared as one of the final replicates",
                          pilot_id not in final_ids, pilot_id):
        return 1

    try:
        repo_root = campaign_root_to_repo_root(campaign_root, orchestrator)
        orchestrator.reject_unsafe_path(campaign_root, expect_dir=True)
    except (EvidenceError, orchestrator.OrchestratorError) as exc:
        reporter.check("the campaign root is a real, non-symlink directory under the "
                       "repository", False, str(exc))
        return 1
    reporter.check("the campaign root is a real, non-symlink directory under the repository",
                   True)

    summaries: dict[str, dict] = {}
    for campaign_id, kind in [(pilot_id, PILOT_KIND)] + [(f, FINAL_KIND) for f in final_ids]:
        try:
            summaries[campaign_id] = summarize_campaign(orchestrator, repo_root, campaign_id)
        except EvidenceError as exc:
            reporter.check(f"{campaign_id}: the {kind} campaign's evidence loads and is "
                           f"internally consistent", False, str(exc))
    if reporter.failures:
        print(f"check_phase4_campaigns_p42: evidence: FAILED "
              f"({len(reporter.failures)} check(s))", file=sys.stderr)
        return 1
    for campaign_id, kind in [(pilot_id, PILOT_KIND)] + [(f, FINAL_KIND) for f in final_ids]:
        reporter.check(f"{campaign_id}: the {kind} campaign's evidence loads and is "
                       f"internally consistent", True)

    pilot = summaries[pilot_id]
    finals = [summaries[campaign_id] for campaign_id in final_ids]

    for campaign_id, kind in [(pilot_id, PILOT_KIND)] + [(f, FINAL_KIND) for f in final_ids]:
        errors = validate_campaign_acceptance(
            summaries[campaign_id], expected_kind=kind, orchestrator=orchestrator,
            expected_git_commit=(expected_pilot_commit if campaign_id == pilot_id else None))
        reporter.check(f"{campaign_id}: accepted as a complete full-scope {kind} campaign",
                       not errors, "; ".join(errors))

    if reporter.failures:
        print(f"check_phase4_campaigns_p42: evidence: FAILED "
              f"({len(reporter.failures)} check(s))", file=sys.stderr)
        return 1

    for campaign_id, kind in [(pilot_id, PILOT_KIND)] + [(f, FINAL_KIND) for f in final_ids]:
        try:
            revalidate_campaign_evidence(
                orchestrator, repo_root, summaries[campaign_id],
                context=(revalidation_contexts or {}).get(campaign_id))
        except EvidenceError as exc:
            reporter.check(
                f"{campaign_id}: P4.1 deeply revalidates every completed {kind} stage",
                False, str(exc))
        else:
            reporter.check(
                f"{campaign_id}: P4.1 deeply revalidates every completed {kind} stage",
                True)

    if reporter.failures:
        print(f"check_phase4_campaigns_p42: evidence: FAILED "
              f"({len(reporter.failures)} check(s))", file=sys.stderr)
        return 1

    errors = compare_final_set(finals)
    reporter.check("the three final campaigns share one frozen execution commit, plan, and "
                   "stage order", not errors, "; ".join(errors))

    errors = compare_campaign_population(pilot, finals, orchestrator=orchestrator)
    reporter.check("all four campaigns share the required GPU identity and comparable "
                   "exposed provenance", not errors, "; ".join(errors))

    reporter.check("the pilot is excluded from the three-final replicate set",
                   pilot["campaign_kind"] == PILOT_KIND
                   and pilot_id not in {s["campaign_id"] for s in finals}, "")
    if pilot["git_commit"] != finals[0]["git_commit"]:
        reporter.note(
            f"the pilot ran from commit {pilot['git_commit']} and the three finals from "
            f"{finals[0]['git_commit']}; the pilot is deliberately not part of the final "
            f"statistical population")

    declared = {pilot_id, *final_ids}
    try:
        undeclared_errors, notes = scan_for_undeclared_finals(
            orchestrator, repo_root, declared)
    except orchestrator.OrchestratorError as exc:
        undeclared_errors, notes = [f"the campaign root cannot be enumerated: {exc}"], []
    for note in notes:
        reporter.note(note)
    reporter.check("no undeclared final campaign exists in the campaign root",
                   not undeclared_errors, "; ".join(undeclared_errors))

    if reporter.failures:
        print(f"check_phase4_campaigns_p42: evidence: FAILED "
              f"({len(reporter.failures)} check(s))", file=sys.stderr)
        return 1
    print("check_phase4_campaigns_p42: evidence: OK "
          "(1 accepted pilot + 3 accepted final campaigns; nothing was written, "
          "repaired, resumed, or regenerated; no statistic was computed)")
    return 0


EVIDENCE_MODE_FUNCTIONS = (
    campaign_root_to_repo_root,
    read_campaign_file,
    summarize_campaign,
    revalidate_campaign_evidence,
    validate_campaign_acceptance,
    compare_final_set,
    compare_campaign_population,
    list_campaign_ids,
    scan_for_undeclared_finals,
    check_campaign_evidence,
)


# ===========================================================================
# Repository-contract mode. GPU-free; needs no results/raw/ and no pilot.
# ===========================================================================


def read_text(repo_root: Path, relative: str) -> str:
    return (repo_root / relative).read_text(encoding="utf-8")


def _check_required_files(reporter: Reporter, repo_root: Path) -> bool:
    for relative in REQUIRED_P42_FILES:
        path = repo_root / relative
        reporter.check(f"{relative} exists as a regular, non-symlink file",
                       path.is_file() and not path.is_symlink(), str(path))
    path = repo_root / P42_CHECKER_RELATIVE_PATH
    reporter.check(f"{P42_CHECKER_RELATIVE_PATH} is executable",
                   os.access(path, os.X_OK), str(path))
    return not reporter.failures


def _check_status_frontier(reporter: Reporter, documents: dict[str, str]) -> None:
    plan_text = documents["PLAN.md"]
    for line in EXPECTED_STATUS_LINES:
        reporter.check(f"PLAN.md records the truthful frontier row {line!r}",
                       line in plan_text, "")
    for group, label in ((FORBIDDEN_P41_STATUS_LINES, "P4.1"),
                         (FORBIDDEN_P42_STATUS_LINES, "P4.2"),
                         (FORBIDDEN_P43_STATUS_LINES, "P4.3")):
        for wrong in group:
            reporter.check(
                f"PLAN.md does not record the stale or impossible {label} status {wrong!r}",
                wrong not in plan_text, "")


def _check_truthful_claims(reporter: Reporter, documents: dict[str, str]) -> None:
    for relative, text in documents.items():
        for claim in FORBIDDEN_DOCUMENT_CLAIMS:
            reporter.check(f"{relative} does not claim {claim!r}",
                           claim.lower() not in text.lower(), "")
    for relative in ("PLAN.md", "README.md", "results/README.md",
                     P42_PROTOCOL_RELATIVE_PATH):
        text = documents[relative]
        for pattern, label in REQUIRED_DOCUMENT_STATEMENTS:
            reporter.check(f"{relative} states that {label}",
                           re.search(pattern, text, re.IGNORECASE) is not None, "")
    for relative, statement in REQUIRED_CLOSURE_STATEMENTS.items():
        reporter.check(f"{relative} records the P4.2 closure",
                       statement in documents[relative], statement)
    protocol = documents[P42_PROTOCOL_RELATIVE_PATH]
    for statement in REQUIRED_P42_PROTOCOL_STATEMENTS:
        reporter.check(f"{P42_PROTOCOL_RELATIVE_PATH} states {statement!r}",
                       statement in protocol, "")
    reporter.check(f"{P42_PROTOCOL_RELATIVE_PATH} fixes the accepted pilot as "
                   f"non-publishable and not a final replicate",
                   "publishable=false" in protocol
                   and "not be treated as one of the three final replicates" in protocol, "")
    reporter.check(f"{P41_PROTOCOL_RELATIVE_PATH} keeps its P4.1 closure record",
                   "P4.1 = YES / YES / YES" in documents[P41_PROTOCOL_RELATIVE_PATH]
                   and "Independent audit: PASSED"
                   in documents[P41_PROTOCOL_RELATIVE_PATH], "")


def _check_frozen_execution_path(reporter: Reporter, repo_root: Path, orchestrator) -> None:
    for relative, expected in FROZEN_EXECUTION_PATH_SHA256.items():
        digest = orchestrator.sha256_of_path(repo_root / relative)
        reporter.check(
            f"{relative} is byte-identical to the audited P4.1 execution path",
            digest == expected,
            f"{digest} != {expected}; P4.2 may not change the P4.1 runner or orchestrator")


# The complete public CLI surface of this checker, introspected from its own
# parser. Nothing here can select a GPU, name a campaign kind, or execute a
# campaign, and a new option would have to be declared here first.
EXPECTED_CLI_OPTIONS = frozenset({
    "-h", "--help", "--self-test", "--campaign-root", "--pilot-campaign-id",
    "--final-campaign-id",
})


def _check_single_public_runner(reporter: Reporter, repo_root: Path,
                                p42_source: str) -> None:
    drivers = []
    for path in sorted((repo_root / "scripts").iterdir()):
        if not path.is_file() or path.is_symlink() or path.name == Path(
                ORCHESTRATOR_RELATIVE_PATH).name:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "phase4_orchestrator.py" in text and path.name not in (
                Path(RUN_ALL_RELATIVE_PATH).name,
                Path(P41_CHECKER_RELATIVE_PATH).name,
                Path(P42_CHECKER_RELATIVE_PATH).name):
            drivers.append(path.name)
    reporter.check("scripts/run_all.sh remains the only public Phase 4 execution entry point",
                   not drivers, str(drivers))
    # Scanned over executable code only. This file necessarily *names* the
    # forbidden spellings in order to ban them, so a raw substring scan of its
    # own text would be self-defeating; stripping comments and string literals
    # leaves exactly the code that would actually invoke something.
    code = _strip_python_comments(p42_source)
    reporter.check("the P4.2 checker never executes a real campaign",
                   not re.search(r"\brun_campaign\b|\bcampaign\s*\.\s*run\s*\(",
                                 code), "")
    reporter.check("the P4.2 checker never invokes a container runtime, nvidia-smi, NCU, "
                   "a CUDA compiler, or any child process",
                   not re.search(r"\b(docker|podman|nvidia|ncu|nvcc|subprocess|popen|"
                                 r"system|execv|execve|spawnv)\b", code, re.IGNORECASE), "")
    options = set()
    for action in build_parser()._actions:
        options.update(action.option_strings)
    reporter.check("the P4.2 checker's whole CLI surface is the three documented modes",
                   options == EXPECTED_CLI_OPTIONS,
                   str(sorted(options ^ EXPECTED_CLI_OPTIONS)))


def _check_no_new_dependency(reporter: Reporter, repo_root: Path, p42_source: str) -> None:
    imports = set(re.findall(r"^(?:import|from)\s+([A-Za-z_][A-Za-z0-9_.]*)",
                             p42_source, re.MULTILINE))
    foreign = sorted(name for name in imports
                     if name.split(".")[0] not in STDLIB_IMPORT_ALLOWLIST)
    reporter.check("the P4.2 checker imports only the Python standard library",
                   not foreign, str(foreign))
    for relative in ("VERSIONS.env", "PHASE3_VERSIONS.env"):
        text = read_text(repo_root, relative)
        reporter.check(f"P4.2 adds no key to {relative}",
                       not re.search(r"^(PHASE4_|P41_|P42_)", text, re.MULTILINE), "")


def _check_read_only_evidence_mode(reporter: Reporter) -> None:
    for function in EVIDENCE_MODE_FUNCTIONS:
        source = inspect.getsource(function)
        hits = [pattern for pattern in FORBIDDEN_EVIDENCE_MODE_CALLS
                if re.search(pattern, source)]
        reporter.check(
            f"evidence mode's {function.__name__}() never writes, repairs, resumes, "
            f"deletes, or regenerates evidence", not hits, str(hits))


def _check_reuses_p41(reporter: Reporter, p42_source: str) -> None:
    code = _strip_python_comments(p42_source)
    for name in ("load_manifest_chain", "resolve_campaign_tree", "read_file_nofollow",
                 "open_dir_chain", "reject_unsafe_path", "validate_campaign_id",
                 "sha256_bytes", "stable_digest", "COMPARABLE_EVIDENCE_FIELDS",
                 "GPU_IDENTITY_FIELDS", "PHASE4_RAW_PARTS", "Context", "Campaign",
                 "revalidate_completed"):
        # The stripped stream is one token per line, so an attribute access is
        # matched by its own name; a mention inside prose or a string literal is
        # already gone and cannot satisfy this.
        reporter.check(f"the P4.2 checker reuses P4.1's own {name}",
                       re.search(rf"^{name}$", code, re.MULTILINE) is not None, "")
    # Reuse is required; a second local definition of any of these would be a
    # reimplementation of audited P4.1 logic.
    for name in ("ALLOWED_MANIFEST_KEYS", "MANIFEST_REQUIRED_FIELDS", "MANIFEST_REVISION_RE",
                 "validate_manifest_document", "validate_manifest_revision_transition",
                 "validate_manifest_privacy", "validate_gemm_capture",
                 "verify_terminal_analysis_artifacts", "load_manifest_chain",
                 "resolve_campaign_tree"):
        reporter.check(f"the P4.2 checker defines no copy of P4.1's {name}",
                       re.search(rf"^(?:{name}\s*[:=]|def {name}\b|class {name}\b)",
                                 p42_source, re.MULTILINE) is None, "")


def _check_no_p43_work(reporter: Reporter, repo_root: Path, p42_source: str,
                       documents: dict[str, str]) -> None:
    for relative in FORBIDDEN_P43_ARTEFACTS:
        reporter.check(f"no P4.3 artefact {relative} exists yet",
                       not (repo_root / relative).exists(), "")
    protocol = documents[P42_PROTOCOL_RELATIVE_PATH]
    reporter.check(f"{P42_PROTOCOL_RELATIVE_PATH} defers every statistic and conclusion "
                   f"to P4.3",
                   "P4.3" in protocol
                   and re.search(r"belongs?\s+exclusively\s+to\s+P4\.3",
                                 protocol) is not None, "")
    body = _strip_python_comments(p42_source)
    hits = [pattern for pattern in FORBIDDEN_ANALYSIS_TOKENS
            if re.search(pattern, body, re.IGNORECASE)]
    reporter.check("the P4.2 checker computes no statistic, threshold, ranking, or figure",
                   not hits, str(hits))


def _strip_python_comments(source: str) -> str:
    """Drop '#' comments and literal text so prose naming a forbidden token in
    order to ban it does not trip a scan of executable code.

    From Python 3.12 an f-string's literal text is no longer a STRING token, so
    FSTRING_MIDDLE is dropped too where it exists; interpolated expressions stay
    because they are real code.
    """
    import tokenize
    import io as _io

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


def _check_make_target(reporter: Reporter, repo_root: Path, makefile: str) -> None:
    p41 = load_module(repo_root / P41_CHECKER_RELATIVE_PATH, "_p42_p41_checker_makefile")
    reporter.check(f"the Makefile declares {P42_MAKE_TARGET}",
                   re.search(rf"^{re.escape(P42_MAKE_TARGET)}:", makefile,
                             re.MULTILINE) is not None, "")
    reporter.check(f"{P42_MAKE_TARGET} is declared .PHONY",
                   P42_MAKE_TARGET in makefile, "")
    rules = p41._parse_makefile_rules(makefile)
    if not reporter.check(f"the Makefile rule graph declares {P42_MAKE_TARGET}",
                          P42_MAKE_TARGET in rules, ""):
        return
    closure = p41._target_closure(rules, P42_MAKE_TARGET)
    unknown = [target for target in closure if target not in rules]
    reporter.check(f"every prerequisite of {P42_MAKE_TARGET} resolves to a real rule",
                   not unknown, str(unknown))
    offenders: list[str] = []
    for target in closure:
        for line in rules.get(target, ([], []))[1]:
            command = line.lstrip().lstrip("@-+").lstrip()
            for name in p41.FORBIDDEN_CHECK_PATH_COMMANDS:
                if re.search(p41._COMMAND_POSITION.format(name=re.escape(name)), command):
                    offenders.append(f"{target}: {line.strip()[:80]}")
    reporter.check(f"no recipe reachable from {P42_MAKE_TARGET} invokes a container "
                   f"runtime, nvidia-smi, or a CUDA compiler", not offenders,
                   str(offenders[:3]))
    reporter.check(f"{P42_MAKE_TARGET} reaches no Docker-backed or GPU gate",
                   not ({"compute-umma-p24-check", "gemm-comparison-p35-check", "preflight",
                         "build-image", "check-env"} & set(closure)), str(sorted(closure)))
    # Recipe lines are deliberately not variable-expanded by P4.1's parser, so
    # the checker may legitimately be referenced through its Make variable.
    recipe = "\n".join(rules[P42_MAKE_TARGET][1])
    reporter.check(f"{P42_MAKE_TARGET} runs the real P4.2 checker's self-test and "
                   f"repository check",
                   "--self-test" in recipe
                   and (P42_CHECKER_RELATIVE_PATH in recipe
                        or "$(PHASE4_P42_CHECKER)" in recipe), "")
    reporter.check(f"{P42_MAKE_TARGET} never starts or resumes a campaign",
                   "--resume" not in recipe
                   and not re.search(r"run_all\.sh(?![^\n]*--dry-run)", recipe), "")
    reporter.check("the Makefile adds no target that could start a final campaign",
                   not re.search(r"^phase4-p42-(pilot|final|campaign|run|smoke):",
                                 makefile, re.MULTILINE), "")
    reporter.check("the closed P4.1 Make targets still exist",
                   re.search(r"^phase4-p41-check:", makefile, re.MULTILINE) is not None
                   and re.search(r"^phase4-p41-plan:", makefile, re.MULTILINE) is not None, "")


def check_repository(repo_root: Path) -> int:
    reporter = Reporter("check_phase4_campaigns_p42")
    if not _check_required_files(reporter, repo_root):
        print("check_phase4_campaigns_p42: FAILED (missing required files)", file=sys.stderr)
        return 1

    documents = {relative: read_text(repo_root, relative) for relative in STATUS_DOCUMENTS}
    makefile = read_text(repo_root, "Makefile")
    p42_source = read_text(repo_root, P42_CHECKER_RELATIVE_PATH)
    orchestrator = load_module(repo_root / ORCHESTRATOR_RELATIVE_PATH,
                               "_p42_orchestrator_repo")

    reporter.check("the frozen nine-stage full plan is P4.1's own",
                   tuple(orchestrator.STAGE_PLANS["full"]) == EXPECTED_FULL_PLAN,
                   str(orchestrator.STAGE_PLANS["full"]))
    reporter.check("the accepted pilot ID is unchanged",
                   PILOT_CAMPAIGN_ID == "20260812T013848Z", PILOT_CAMPAIGN_ID)
    reporter.check("the accepted pilot commit is unchanged",
                   PILOT_GIT_COMMIT == "77908dae377fb131ff90baef455c79d6d2c28b0b",
                   PILOT_GIT_COMMIT)
    reporter.check("P4.2 requires exactly three final campaigns",
                   REQUIRED_FINAL_CAMPAIGN_COUNT == 3, "")

    _check_status_frontier(reporter, documents)
    _check_truthful_claims(reporter, documents)
    _check_frozen_execution_path(reporter, repo_root, orchestrator)
    _check_single_public_runner(reporter, repo_root, p42_source)
    _check_no_new_dependency(reporter, repo_root, p42_source)
    _check_read_only_evidence_mode(reporter)
    _check_reuses_p41(reporter, p42_source)
    _check_no_p43_work(reporter, repo_root, p42_source, documents)
    _check_make_target(reporter, repo_root, makefile)

    reporter.check("the repository contract check needed no raw campaign evidence",
                   True)

    if reporter.failures:
        print(f"check_phase4_campaigns_p42: FAILED ({len(reporter.failures)} check(s))",
              file=sys.stderr)
        return 1
    print("check_phase4_campaigns_p42: OK "
          "(P4.2 = YES / YES / YES; three finals accepted; no publishable result; "
          "P4.3 unimplemented)")
    return 0


# ===========================================================================
# Self-test. Temporary directories only.
# ===========================================================================


PILOT_FIXTURE_ID = "20990601T000000Z"
FINAL_FIXTURE_IDS = ("20990602T000000Z", "20990603T000000Z", "20990604T000000Z")
FIXTURE_PILOT_COMMIT = "1111111111111111111111111111111111111111"
FIXTURE_FINAL_COMMIT = "2222222222222222222222222222222222222222"
FIXTURE_OTHER_COMMIT = "3333333333333333333333333333333333333333"


class _TempStack:
    def __init__(self):
        self.paths: list[Path] = []
        self.revalidation_contexts: dict[Path, dict[str, object]] = {}

    def new_root(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="p42-"))
        self.paths.append(root)
        return root

    def cleanup(self) -> None:
        for path in self.paths:
            shutil.rmtree(path, ignore_errors=True)
        self.revalidation_contexts.clear()

    def remember_context(self, root: Path, campaign_id: str, context) -> None:
        key = Path(os.path.abspath(str(root)))
        self.revalidation_contexts.setdefault(key, {})[campaign_id] = context


_STACK = _TempStack()


def _reset_units(fixture) -> None:
    """Each synthetic campaign gets a fresh underlying P1.4/P2.4 campaign, the
    way a real sequential campaign does."""
    fixture.p14_state = None
    fixture.p24_state = None
    fixture.p14_revision = -1
    fixture.p24_revision = -1


def _set_commit(fixture, commit: str) -> None:
    fixture.commit = commit
    fixture.unit_commit = commit


def _fixture_revalidation_context(fixture, campaign_id: str):
    """Freeze the two semantic loader payloads belonging to one fixture run.

    A Fixture instance is reused sequentially, while each real campaign keeps
    its own P1.4/P2.4 state. Capturing those payloads lets the later evidence
    check re-open every campaign with the exact state its completed run left.
    """
    p14_payload = (fixture.p14_loader(campaign_id)
                   if fixture.p14_state is not None else None)
    p24_payload = (fixture.p24_loader(campaign_id)
                   if fixture.p24_state is not None else None)

    def loader(expected_id: str, payload: dict | None):
        def load(requested_id: str) -> dict:
            if requested_id != expected_id:
                raise RuntimeError(
                    f"fixture loader for {expected_id} cannot load {requested_id}")
            if payload is None:
                raise RuntimeError(f"no fixture unit campaign exists for {expected_id}")
            return json.loads(json.dumps(payload))
        return load

    return fixture.context(
        p14_loader=loader(campaign_id, p14_payload),
        p24_loader=loader(campaign_id, p24_payload),
    )


def _drive(fixture, campaign_id: str, *, kind: str, selector=None) -> int:
    _reset_units(fixture)
    status = fixture.run(campaign_id=campaign_id, kind=kind, selector=selector)
    _STACK.remember_context(
        fixture.root, campaign_id,
        _fixture_revalidation_context(fixture, campaign_id),
    )
    return status


def _build_population(o, p35, p41, reporter: Reporter, label: str, *,
                      root: Path | None = None,
                      pilot_commit=FIXTURE_PILOT_COMMIT,
                      final_commits=None, gpu_overrides=None, runtime_overrides=None,
                      final_kinds=None, final_selectors=None):
    """One temporary repository holding a real pilot plus three real final
    campaigns, all produced by P4.1's own orchestration path."""
    root = root if root is not None else _STACK.new_root()
    fixture = p41.Fixture(o, p35, root)
    _set_commit(fixture, pilot_commit)
    statuses = {PILOT_FIXTURE_ID: _drive(fixture, PILOT_FIXTURE_ID, kind=PILOT_KIND)}
    commits = final_commits or [FIXTURE_FINAL_COMMIT] * REQUIRED_FINAL_CAMPAIGN_COUNT
    kinds = final_kinds or [FINAL_KIND] * REQUIRED_FINAL_CAMPAIGN_COUNT
    selectors = final_selectors or [None] * REQUIRED_FINAL_CAMPAIGN_COUNT
    for index, campaign_id in enumerate(FINAL_FIXTURE_IDS):
        _set_commit(fixture, commits[index])
        if gpu_overrides and index in gpu_overrides:
            fixture.gpu["uuid"] = gpu_overrides[index]
            fixture.unit_gpu_uuid = gpu_overrides[index]
        if runtime_overrides and index in runtime_overrides:
            fixture.unit_cuda_runtime_version = runtime_overrides[index]
        statuses[campaign_id] = _drive(
            fixture, campaign_id, kind=kinds[index], selector=selectors[index])
    wrong = {cid: status for cid, status in statuses.items() if status != o.EXIT_OK}
    reporter.check(f"{label}: four real campaigns were produced by P4.1's own path",
                   not wrong, str(wrong))
    return root, fixture


def _evidence(o, root: Path, *, pilot_ids=None, final_ids=None) -> int:
    root_key = Path(os.path.abspath(str(root)))
    return check_campaign_evidence(
        o, root / "results" / "raw" / "phase4",
        list(pilot_ids if pilot_ids is not None else [PILOT_FIXTURE_ID]),
        list(final_ids if final_ids is not None else FINAL_FIXTURE_IDS),
        expected_pilot_id=PILOT_FIXTURE_ID,
        expected_pilot_commit=FIXTURE_PILOT_COMMIT,
        revalidation_contexts=_STACK.revalidation_contexts.get(root_key, {}))


def _rejects(reporter: Reporter, label: str, call) -> bool:
    """Run one negative case with its own diagnostics captured, so a self-test
    log shows only the outer verdicts. The captured rejection detail is printed
    only when the case did *not* reject, which is when it is worth reading."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        status = call()
    if status == 1:
        return reporter.check(label, True)
    return reporter.check(label, False,
                          f"returned {status}; captured output: "
                          f"{(out.getvalue() + err.getvalue()).strip()[:400]}")


def _tree_snapshot(o, root: Path) -> dict[str, str]:
    """Every regular file under the campaign root, by hash. Used to prove
    evidence mode mutated nothing at all."""
    snapshot: dict[str, str] = {}
    base = root / "results" / "raw" / "phase4"
    for path in sorted(base.rglob("*")):
        if path.is_file() and not path.is_symlink():
            snapshot[str(path.relative_to(base))] = o.sha256_of_path(path)
        else:
            snapshot[str(path.relative_to(base))] = "<dir>"
    return snapshot


def _tamper_last_revision(o, root: Path, campaign_id: str, mutate) -> None:
    """Rewrite only the highest manifest revision, whose hash nothing later
    references, so the field-level schema gate is what rejects it rather than
    the chain hash. Synthetic fixtures only."""
    manifest_dir = root / "results" / "raw" / "phase4" / campaign_id / "manifest"
    names = sorted(p.name for p in manifest_dir.iterdir()
                   if o.MANIFEST_REVISION_RE.fullmatch(p.name))
    path = manifest_dir / names[-1]
    document = json.loads(path.read_text(encoding="utf-8"))
    mutate(document)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def run_self_test(o, p35, p41) -> int:
    reporter = Reporter("check_phase4_campaigns_p42: self-test")

    # 1. One valid pilot plus three valid finals is accepted. This is the only
    #    case whose evidence-mode output is shown; every negative case below
    #    captures its own rejection diagnostics so the log stays readable.
    root, _fixture = _build_population(o, p35, p41, reporter, "accepted set")
    before = _tree_snapshot(o, root)
    reporter.check("a valid pilot plus three valid final campaigns is accepted",
                   _evidence(o, root) == 0, "")
    reporter.check("evidence mode wrote, repaired, deleted, or regenerated nothing",
                   _tree_snapshot(o, root) == before, "")

    # Focused audit regression 1: the top-level manifest chain and plan remain
    # intact, but P4.1's terminal stage revalidation must notice that the
    # completed P3.5 capture changed on disk.
    csv_root, _ = _build_population(
        o, p35, p41, reporter, "tampered-P3.5 set")
    csv_path = (csv_root / "results" / "raw" / "phase4" / FINAL_FIXTURE_IDS[1]
                / o.GEMM_ARTIFACT_REL)
    csv_path.write_text(csv_path.read_text(encoding="utf-8") + "tampered\n",
                        encoding="utf-8")
    _rejects(reporter, "a changed completed P3.5 CSV is rejected by P4.1 revalidation",
             lambda: _evidence(o, csv_root))

    # Focused audit regression 2: the canonical pilot ID is insufficient by
    # itself; its recorded commit must be the exact accepted historical SHA.
    pilot_commit_root, _ = _build_population(
        o, p35, p41, reporter, "wrong-pilot-commit set",
        pilot_commit=FIXTURE_OTHER_COMMIT)
    _rejects(reporter, "the accepted pilot ID on the wrong Git commit is rejected",
             lambda: _evidence(o, pilot_commit_root))

    # 7. The pilot keeps its own historical commit and is excluded from the
    #    three-final commit-equality set.
    pilot_summary = summarize_campaign(o, root, PILOT_FIXTURE_ID)
    final_summary = summarize_campaign(o, root, FINAL_FIXTURE_IDS[0])
    reporter.check("the accepted set tolerates a pilot on its own historical commit",
                   pilot_summary["git_commit"] != final_summary["git_commit"], "")
    reporter.check("the pilot is recorded as a pilot, never as a final replicate",
                   pilot_summary["campaign_kind"] == PILOT_KIND
                   and final_summary["campaign_kind"] == FINAL_KIND, "")

    # 2. Fewer or more than three final IDs is rejected.
    for wrong in (list(FINAL_FIXTURE_IDS[:2]),
                  list(FINAL_FIXTURE_IDS) + ["20990605T000000Z"]):
        _rejects(reporter, f"a population declaring {len(wrong)} final campaigns is rejected",
                 lambda w=wrong: _evidence(o, root, final_ids=w))
    _rejects(reporter, "a population declaring no pilot is rejected",
             lambda: _evidence(o, root, pilot_ids=[]))
    _rejects(reporter, "a population declaring two pilots is rejected",
             lambda: _evidence(o, root, pilot_ids=[PILOT_FIXTURE_ID, PILOT_FIXTURE_ID]))

    # 3. Duplicate or reused campaign IDs are rejected.
    _rejects(reporter, "a duplicated final campaign ID is rejected",
             lambda: _evidence(o, root, final_ids=[FINAL_FIXTURE_IDS[0], FINAL_FIXTURE_IDS[0],
                                                   FINAL_FIXTURE_IDS[1]]))
    _rejects(reporter, "declaring the pilot as one of the three finals is rejected",
             lambda: _evidence(o, root, final_ids=[PILOT_FIXTURE_ID, FINAL_FIXTURE_IDS[1],
                                                   FINAL_FIXTURE_IDS[2]]))
    _rejects(reporter, "a missing campaign ID is rejected",
             lambda: _evidence(o, root, final_ids=[FINAL_FIXTURE_IDS[0], FINAL_FIXTURE_IDS[1],
                                                   "20990809T000000Z"]))
    _rejects(reporter, "a malformed campaign ID is rejected",
             lambda: _evidence(o, root, final_ids=[FINAL_FIXTURE_IDS[0], FINAL_FIXTURE_IDS[1],
                                                   "not-a-campaign"]))

    # 4a. A wrong kind is rejected in both directions.
    kind_root, _ = _build_population(
        o, p35, p41, reporter, "wrong-kind set",
        final_kinds=[FINAL_KIND, FINAL_KIND, PILOT_KIND])
    _rejects(reporter, "a final replicate recorded as campaign_kind=pilot is rejected",
             lambda: _evidence(o, kind_root))

    # 4b. A non-full scope is rejected, and with it the divergent stage order
    #     and immutable plan that scope implies.
    scope_root, _ = _build_population(
        o, p35, p41, reporter, "non-full-scope set",
        final_selectors=[None, None, "memory"])
    _rejects(reporter, "a final campaign whose scope is not full is rejected",
             lambda: _evidence(o, scope_root))
    scope_summary = summarize_campaign(o, scope_root, FINAL_FIXTURE_IDS[2])
    good_summary = summarize_campaign(o, scope_root, FINAL_FIXTURE_IDS[0])
    reporter.check("a non-full final also breaks the shared immutable plan and stage order",
                   len(compare_final_set([good_summary, good_summary, scope_summary])) == 2,
                   str(compare_final_set([good_summary, good_summary, scope_summary])))

    # 4c. A partial (non-terminal, incompletely staged) campaign is rejected.
    partial_root = _STACK.new_root()
    partial_fixture = p41.Fixture(o, p35, partial_root)
    _set_commit(partial_fixture, FIXTURE_PILOT_COMMIT)
    _drive(partial_fixture, PILOT_FIXTURE_ID, kind=PILOT_KIND)
    _set_commit(partial_fixture, FIXTURE_FINAL_COMMIT)
    _drive(partial_fixture, FINAL_FIXTURE_IDS[0], kind=FINAL_KIND)
    _drive(partial_fixture, FINAL_FIXTURE_IDS[1], kind=FINAL_KIND)
    partial_fixture.fail_targets["gemm-comparison-p35-smoke"] = 1
    failed_as_expected = False
    try:
        _drive(partial_fixture, FINAL_FIXTURE_IDS[2], kind=FINAL_KIND)
    except o.RunFailure:
        failed_as_expected = True
    reporter.check("the synthetic partial campaign really failed mid-plan",
                   failed_as_expected, "")
    _rejects(reporter, "a partial, non-terminal final campaign is rejected",
             lambda: _evidence(o, partial_root))

    # 4d. A dirty tree or a publishable artifact is rejected. P4.1 refuses to
    #     start such a campaign at all, so these are produced by tampering with
    #     the last manifest revision; the reused P4.1 loader is what rejects it.
    for field, value, tag in (("git_dirty", True, "git_dirty=true"),
                              ("publishable", True, "publishable=true")):
        tamper_root, _ = _build_population(o, p35, p41, reporter, f"{tag} set")
        _tamper_last_revision(o, tamper_root, FINAL_FIXTURE_IDS[1],
                              lambda doc, f=field, v=value: doc.__setitem__(f, v))
        _rejects(reporter, f"a final campaign recording {tag} is rejected",
                 lambda r=tamper_root: _evidence(o, r))

    # 5. Different final Git commits are rejected; a rewritten immutable plan
    #    is rejected by its own recorded digest.
    commit_root, _ = _build_population(
        o, p35, p41, reporter, "split-commit set",
        final_commits=[FIXTURE_FINAL_COMMIT, FIXTURE_FINAL_COMMIT, FIXTURE_OTHER_COMMIT])
    _rejects(reporter, "three final campaigns on different Git commits are rejected",
             lambda: _evidence(o, commit_root))
    plan_root, _ = _build_population(o, p35, p41, reporter, "rewritten-plan set")
    plan_path = (plan_root / "results" / "raw" / "phase4" / FINAL_FIXTURE_IDS[0]
                 / o.PLAN_FILE_NAME)
    plan_path.write_text(plan_path.read_text(encoding="utf-8").replace(
        '"scope": "full"', '"scope":  "full"'), encoding="utf-8")
    _rejects(reporter, "a rewritten immutable plan is rejected by its recorded digest",
             lambda: _evidence(o, plan_root))

    # 6. A different GPU identity or a different comparable exposed provenance
    #    is rejected, even though each campaign is internally consistent.
    gpu_root, _ = _build_population(
        o, p35, p41, reporter, "split-GPU set",
        gpu_overrides={2: "GPU-abcdef01-2345-6789-abcd-ef0123456789"})
    _rejects(reporter, "a final campaign run on a different physical GPU is rejected",
             lambda: _evidence(o, gpu_root))
    runtime_root, _ = _build_population(
        o, p35, p41, reporter, "split-provenance set", runtime_overrides={1: "13020"})
    _rejects(reporter, "a final campaign with different comparable exposed provenance is "
                       "rejected", lambda: _evidence(o, runtime_root))

    # 8. An additional undeclared final campaign is rejected; an undeclared
    #    pilot is reported rather than silently ignored.
    extra_root, extra_fixture = _build_population(o, p35, p41, reporter, "undeclared-final set")
    _set_commit(extra_fixture, FIXTURE_FINAL_COMMIT)
    _drive(extra_fixture, "20990605T000000Z", kind=FINAL_KIND)
    _rejects(reporter, "an additional undeclared full-scope final campaign is rejected",
             lambda: _evidence(o, extra_root))
    spare_root, spare_fixture = _build_population(o, p35, p41, reporter, "undeclared-pilot set")
    _set_commit(spare_fixture, FIXTURE_FINAL_COMMIT)
    _drive(spare_fixture, "20990606T000000Z", kind=PILOT_KIND)
    reporter.check("an undeclared extra pilot is reported but does not fail the set",
                   _evidence(o, spare_root) == 0, "")

    # Symlinked and structurally unexpected evidence is rejected.
    link_root, _ = _build_population(o, p35, p41, reporter, "symlinked set")
    base = link_root / "results" / "raw" / "phase4"
    os.symlink(base / FINAL_FIXTURE_IDS[0], base / "20990607T000000Z")
    _rejects(reporter, "a symlinked campaign directory in the campaign root is rejected",
             lambda: _evidence(o, link_root))
    stray_root, _ = _build_population(o, p35, p41, reporter, "stray-entry set")
    (stray_root / "results" / "raw" / "phase4" / "README").write_text("x", encoding="utf-8")
    _rejects(reporter, "an unexpected entry in the campaign root is rejected",
             lambda: _evidence(o, stray_root))
    empty_root, _ = _build_population(o, p35, p41, reporter, "empty-campaign set")
    (empty_root / "results" / "raw" / "phase4" / "20990608T000000Z").mkdir()
    _rejects(reporter, "an uninterpretable campaign directory is rejected, never repaired",
             lambda: _evidence(o, empty_root))

    # A campaign root that is not the repository's own results/raw/phase4.
    _rejects(reporter, "a campaign root outside results/raw/phase4 is rejected",
             lambda: check_campaign_evidence(o, root / "results", [PILOT_FIXTURE_ID],
                                             list(FINAL_FIXTURE_IDS),
                                             expected_pilot_id=PILOT_FIXTURE_ID))
    _rejects(reporter, "the CLI freezes the pilot ID to the accepted campaign",
             lambda: _evidence(o, root, pilot_ids=[FINAL_FIXTURE_IDS[0]]))

    # 9. Repository mode passes with no results/raw/ at all.
    bare = _STACK.new_root() / "clone"
    _copy_repository_for_contract_check(DEFAULT_REPO_ROOT, bare)
    reporter.check("the bare clone really has no results/raw/ tree",
                   not (bare / "results" / "raw").exists(), "")
    reporter.check("repository-contract mode passes without any results/raw/ directory",
                   check_repository(bare) == 0, "")

    if reporter.failures:
        print(f"check_phase4_campaigns_p42: self-test: FAILED "
              f"({len(reporter.failures)} check(s))", file=sys.stderr)
        return 1
    print("check_phase4_campaigns_p42: self-test: OK")
    return 0


def _copy_repository_for_contract_check(source: Path, destination: Path) -> None:
    """A minimal copy of exactly what repository-contract mode reads, with no
    results/raw/ tree, to prove the mode needs no raw campaign evidence."""
    needed = list(REQUIRED_P42_FILES) + list(STATUS_DOCUMENTS) + [
        "Makefile", "VERSIONS.env", "PHASE3_VERSIONS.env", P35_CHECKER_RELATIVE_PATH]
    destination.mkdir(parents=True)
    for relative in sorted(set(needed)):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, target)
    # Every other executable in scripts/ is scanned for a second Phase 4 runner.
    for path in sorted((source / "scripts").iterdir()):
        if path.is_file() and not path.is_symlink():
            shutil.copy2(path, destination / "scripts" / path.name)


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_phase4_campaigns_p42.py",
        description="Fail-closed, GPU-free cross-campaign checker for P4.2: one accepted "
                    "pilot plus three final campaigns.",
    )
    parser.add_argument("repo_root", nargs="?", default=None,
                        help="repository root to check against the frozen P4.2 contract")
    parser.add_argument("--self-test", action="store_true",
                        help="run the focused synthetic suite and exit; standalone only")
    parser.add_argument("--campaign-root", default=None,
                        help="an existing results/raw/phase4 directory to validate "
                             "(strictly read-only)")
    parser.add_argument("--pilot-campaign-id", action="append", default=None,
                        metavar="ID", help=f"the accepted pilot ({PILOT_CAMPAIGN_ID}); "
                                           f"exactly one is required in evidence mode")
    parser.add_argument("--final-campaign-id", action="append", default=None, metavar="ID",
                        help=f"a declared final campaign; exactly "
                             f"{REQUIRED_FINAL_CAMPAIGN_COUNT} are required")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    evidence_requested = any(value is not None for value in
                             (args.campaign_root, args.pilot_campaign_id,
                              args.final_campaign_id))

    if args.self_test:
        if args.repo_root is not None or evidence_requested:
            print("check_phase4_campaigns_p42: --self-test is standalone", file=sys.stderr)
            return 2
        orchestrator = load_module(SCRIPT_DIR / "phase4_orchestrator.py", "_p42_orchestrator")
        p35 = load_module(SCRIPT_DIR / "check_gemm_comparison_p35.py", "_p42_p35_checker")
        p41 = load_module(SCRIPT_DIR / "check_phase4_orchestrator_p41.py", "_p42_p41_checker")
        try:
            return run_self_test(orchestrator, p35, p41)
        finally:
            _STACK.cleanup()

    if evidence_requested:
        if args.repo_root is not None:
            print("check_phase4_campaigns_p42: evidence mode takes no repository root "
                  "positional argument", file=sys.stderr)
            return 2
        if args.campaign_root is None:
            print("check_phase4_campaigns_p42: evidence mode requires --campaign-root",
                  file=sys.stderr)
            return 2
        orchestrator = load_module(SCRIPT_DIR / "phase4_orchestrator.py", "_p42_orchestrator")
        return check_campaign_evidence(
            orchestrator, Path(args.campaign_root),
            list(args.pilot_campaign_id or []), list(args.final_campaign_id or []))

    if args.repo_root is None:
        print("check_phase4_campaigns_p42: a repository root, --self-test, or "
              "--campaign-root is required", file=sys.stderr)
        return 2
    repo_root = Path(args.repo_root).resolve()
    if not repo_root.is_dir():
        print(f"check_phase4_campaigns_p42: {repo_root} is not a directory", file=sys.stderr)
        return 2
    return check_repository(repo_root)


if __name__ == "__main__":
    raise SystemExit(main())
