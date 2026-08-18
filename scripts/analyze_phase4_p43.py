#!/usr/bin/env python3
"""P4.3 -- the offline, read-only integrated analysis over the frozen Phase 4
population.

P4.1 runs one campaign. P4.2 froze the population -- one accepted pilot plus
exactly three final campaigns -- and validates the invariants that only hold
across it. P4.3 is the smallest layer that turns that already accepted evidence
into the final curated tables, JSON summary, report, and figures.

It executes **no** GPU command. It never invokes Docker, ``nvidia-smi``, CUDA,
Nsight Compute, the preflight, ``scripts/run_all.sh``, or any child process at
all, and it never writes, repairs, resumes, or regenerates anything under
``results/raw/``. It adds no experimental parameter, shape, candidate, metric,
schema, or version pin.

Three modes::

    python3 scripts/analyze_phase4_p43.py --self-test
        Synthetic acceptance tests over temporary directories only. Nothing in
        this repository is read as evidence and nothing is written outside the
        temporary tree.

    python3 scripts/analyze_phase4_p43.py --analyze \\
        --campaign-root results/raw/phase4 \\
        --pilot-campaign-id <PILOT> \\
        --final-campaign-id <FINAL_1> --final-campaign-id <FINAL_2> \\
        --final-campaign-id <FINAL_3> \\
        --output-root results/phase4
        Deeply revalidate the whole population through P4.2's own read-only
        evidence mode (which itself delegates every per-campaign decision to
        P4.1), then read the canonical terminal P1.4/P2.4/P3.5 artifacts each
        final campaign's manifest pins, aggregate across the three campaigns,
        and publish the frozen artifact inventory no-clobber.

    python3 scripts/analyze_phase4_p43.py --verify   (same options)
        Recompute the complete analysis from the same evidence and compare
        every output byte for byte. Writes nothing.

The statistical unit is **one complete final campaign**, never one timing
repetition. The accepted pilot is orchestration qualification evidence only and
never enters a statistic, ranking, variability estimate, table, figure, or
conclusion.

Everything this module publishes is a **candidate**: the nine artifacts record
``publishable=false`` and ``publication_state=candidate_pending_independent_
output_review``. Nothing here promotes, overwrites, or deletes a candidate. The
later acceptance attestation (``src/phase4/P4_3_ACCEPTANCE.json``) is an
external file this module never writes and only knows how to validate.

Exit codes: 0 OK; 1 at least one check failed; 2 usage error.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import stat
import struct
import sys
import tempfile
import zlib
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent

ORCHESTRATOR_RELATIVE_PATH = "scripts/phase4_orchestrator.py"
P42_CHECKER_RELATIVE_PATH = "scripts/check_phase4_campaigns_p42.py"
P35_CHECKER_RELATIVE_PATH = "scripts/check_gemm_comparison_p35.py"

SCHEMA_VERSION = "p43.v1"
UNIT = "P4.3"
PUBLISHABLE = False
# Every artifact this module writes is a candidate awaiting an independent
# review of the outputs themselves. There is no route that promotes, rewrites,
# or deletes a candidate: acceptance is an external, separate attestation.
PUBLICATION_STATE = "candidate_pending_independent_output_review"
PUBLICATION_STATUS = (
    "publishable=false; publication_state="
    "candidate_pending_independent_output_review; the P4.3 independent audit, "
    "the production analysis of the three real final campaigns, and the "
    "independent review of its outputs are all pending"
)

# ---------------------------------------------------------------------------
# The external acceptance attestation. P4.3 never writes this file: it is
# created only by a later, explicitly authorized closing action, after an
# independent reviewer has inspected the complete candidate bundle. It is not
# part of the nine-artifact analysis inventory. It binds the manifest's own
# SHA-256 -- which analysis_manifest.json structurally cannot contain -- and so
# covers all nine artifacts without any self-reference.
# ---------------------------------------------------------------------------
ACCEPTANCE_SCHEMA_VERSION = "p43.acceptance.v1"
ACCEPTANCE_RELATIVE_PATH = "src/phase4/P4_3_ACCEPTANCE.json"
ACCEPTANCE_STATUS_ACCEPTED = "ACCEPTED"
ACCEPTANCE_VERIFICATION_OUTCOME = "byte_for_byte_recomputation_matched"
ACCEPTANCE_REVIEW_OUTCOME = "independent_output_review_passed"
ACCEPTANCE_REQUIRED_FIELDS = (
    "schema_version",
    "unit",
    "status",
    "accepted_for_publication",
    "analysis_code_commit",
    "final_campaign_ids",
    "pilot_campaign_id_excluded",
    "analysis_manifest_sha256",
    "artifact_sha256",
    "verification_outcome",
    "independent_output_review_outcome",
)
# The frozen lifecycle. Every step is a separate, explicitly authorized action;
# none of the steps after candidate production has been performed.
ACCEPTANCE_LIFECYCLE = (
    "an independently audited, clean analysis-code commit",
    "candidate production analysis from exactly that commit",
    "byte-for-byte verification of the candidate bundle",
    "independent scientific and output review of the complete bundle",
    f"an external acceptance attestation at {ACCEPTANCE_RELATIVE_PATH}",
    "a final documentation and status commit",
)

# ---------------------------------------------------------------------------
# The scientific evidence taxonomy (src/phase4/P4_3_PROTOCOL.md section 5).
#
# Every reported quantity carries exactly one of these classes, in the CSV
# tables, in the JSON summary, in the Markdown report, and in the figure
# captions. The classes are not decorative: they are what stops a deterministic
# derived rate, a modeled clock conversion, or a cross-campaign statistic from
# being read as a direct hardware measurement.
# ---------------------------------------------------------------------------
EVIDENCE_MEASURED = "measured_source_observation"
EVIDENCE_WITHIN_CAMPAIGN = "within_campaign_derived_estimate"
EVIDENCE_CROSS_CAMPAIGN = "cross_campaign_descriptive_statistic"
EVIDENCE_MODELED = "modeled_estimate"
EVIDENCE_INTERPRETATION = "interpretation"
EVIDENCE_UNAVAILABLE = "unavailable_from_collected_evidence"
EVIDENCE_DIAGNOSTIC = "source_diagnostic"
EVIDENCE_CLASSES = (
    EVIDENCE_MEASURED,
    EVIDENCE_WITHIN_CAMPAIGN,
    EVIDENCE_CROSS_CAMPAIGN,
    EVIDENCE_MODELED,
    EVIDENCE_INTERPRETATION,
    EVIDENCE_UNAVAILABLE,
    EVIDENCE_DIAGNOSTIC,
)

# The per-campaign value behind every emitted metric, classified once, with the
# exact provenance sentence the closed upstream protocol supports. The
# cross-campaign mean, median, sample standard deviation, coefficient of
# variation, minimum, and maximum computed beside it are *always*
# EVIDENCE_CROSS_CAMPAIGN, whatever the underlying quantity is.
METRIC_EVIDENCE: dict[str, tuple[str, str]] = {
    "median_effective_gbps": (
        EVIDENCE_WITHIN_CAMPAIGN,
        "campaign-level median of 30 timing-derived effective transfer rates; each "
        "repetition is the benchmark's logical useful_bytes divided by its own "
        "CUDA-event kernel time (P1.1/P1.2: effective copy bandwidth, explicitly NOT "
        "HBM/DRAM bandwidth)"),
    "tma_to_ldgsts_ratio": (
        EVIDENCE_WITHIN_CAMPAIGN,
        "derived within-campaign ratio of the two campaign-level median effective "
        "transfer rates at one identical configuration (P1.4 section 6)"),
    "dram_read_ratio": (
        EVIDENCE_WITHIN_CAMPAIGN,
        "derived from profiler evidence, not a raw profiler counter: Nsight Compute's "
        "dram__bytes_read.sum divided by that case's validated useful_bytes (P1.4 "
        "section 5)"),
    "hbm_classification": (
        EVIDENCE_WITHIN_CAMPAIGN,
        "derived classification of the profiler-derived dram_read_ratio against P1.4's "
        "frozen 0.90 rule; not a raw profiler counter"),
    "median_flops_per_cycle": (
        EVIDENCE_WITHIN_CAMPAIGN,
        "clock-independent operation-and-cycle-derived throughput: the campaign-level "
        "median of total_flops (a validated 2*M*N*K*depth*iterations operation count) "
        "divided by the measured %clock64 elapsed_cycles (P2.1/P2.2, P2.4 section 6)"),
    "median_flops_per_cycle_per_sm": (
        EVIDENCE_WITHIN_CAMPAIGN,
        "the same operation-and-cycle-derived throughput divided by the configuration's "
        "cta_group; clock-independent, still derived, never directly measured"),
    "speedup_2sm_over_1sm": (
        EVIDENCE_WITHIN_CAMPAIGN,
        "derived within-campaign ratio of two campaign-level median FLOP/cycle values "
        "(P2.4 section 6.1); the two configurations execute sequentially, never as "
        "paired samples"),
    "scaling_efficiency_percent": (
        EVIDENCE_WITHIN_CAMPAIGN,
        "derived within-campaign quantity: 100 * speedup_2sm_over_1sm / 2, never "
        "clamped to [0, 100]"),
    "earliest_tested_candidate_saturation_bif_kib": (
        EVIDENCE_WITHIN_CAMPAIGN,
        "derived within-campaign selection over exactly three tested bytes-in-flight "
        "values; never a universal HBM saturation threshold"),
    "earliest_tested_candidate_saturation_depth": (
        EVIDENCE_WITHIN_CAMPAIGN,
        "derived within-campaign selection over exactly four tested depths; never a "
        "universal architectural saturation depth"),
    "estimated_tflops_per_sm": (
        EVIDENCE_MODELED,
        "modeled clock conversion of a one-/two-SM microbenchmark result: the "
        "clock-independent median FLOP/cycle multiplied by that same configuration's "
        "own profiled SM clock and divided by cta_group; never an architectural peak"),
    "estimated_device_equivalent_tflops": (
        EVIDENCE_MODELED,
        "modeled whole-device extrapolation: the modeled per-SM microbenchmark "
        "estimate multiplied by a validated SM count; never a measured whole-GPU "
        "throughput"),
    "kernel_time_ms": (
        EVIDENCE_MEASURED,
        "measured input: CUDA-event kernel time on the candidate's own execution "
        "stream, divided by the measured iteration count, after correctness passed "
        "(P3.5 section 7 step 9)"),
    "tflops": (
        EVIDENCE_WITHIN_CAMPAIGN,
        "derived within-campaign throughput: the exact 2*M*N*K operation count divided "
        "by the measured kernel time (P3.5 section 8)"),
    "throughput_ratio_vs_cublaslt": (
        EVIDENCE_WITHIN_CAMPAIGN,
        "derived within-campaign ratio against that campaign's own cuBLASLt baseline "
        "row; never recomputed from cross-campaign aggregates"),
    "gap_to_cublaslt_pct": (
        EVIDENCE_WITHIN_CAMPAIGN,
        "derived within-campaign signed gap, 100 * (1 - throughput_ratio); negative "
        "means the candidate measured faster and is never clamped"),
    "best_cutedsl_variant": (
        EVIDENCE_WITHIN_CAMPAIGN,
        "derived within-campaign selection: the fastest of the three CuTe DSL "
        "candidates by full-precision kernel time in that campaign"),
    "within_campaign_sample_count": (
        EVIDENCE_DIAGNOSTIC,
        "the closed unit's own retained repetition count for this configuration"),
    "within_campaign_cv_percent": (
        EVIDENCE_DIAGNOSTIC,
        "the closed unit's own within-campaign coefficient of variation over that "
        "campaign's retained repetitions; never the cross-campaign CV"),
    "within_campaign_stability_review": (
        EVIDENCE_DIAGNOSTIC,
        "the closed unit's own within-campaign stability diagnostic (CV > 5% inside "
        "one campaign); a diagnostic only, it never filtered a sample"),
    "surprising_value_flag": (
        EVIDENCE_DIAGNOSTIC,
        "the closed unit's own diagnostic that a scaling efficiency fell outside "
        "[0, 100]; the value itself is preserved unclamped"),
    "diagnostic_flags": (
        EVIDENCE_DIAGNOSTIC,
        "the closed unit's own per-case Nsight Compute diagnostics, e.g. "
        "READ_AMPLIFICATION when dram_read_ratio > 1.10; reported, never hidden or "
        "normalized away"),
    "ncu_coverage": (
        EVIDENCE_DIAGNOSTIC,
        "whether this configuration is one of the six frozen Nsight Compute cases; "
        "where it is not, actual HBM/DRAM traffic is unavailable from the collected "
        "evidence"),
}

# The two Nsight Compute coverage states of a memory configuration.
NCU_PROFILED = "ncu_profiled"
NCU_NOT_PROFILED = "not_profiled"
NCU_UNAVAILABLE_NOTE = (
    "actual HBM/DRAM traffic is unavailable from the collected evidence for this "
    "configuration: it is not one of the six frozen Nsight Compute cases")


# ---------------------------------------------------------------------------
# The frozen Phase 4 population (src/phase4/P4_2_PROTOCOL.md section 3).
#
# These identifiers are constants, never discovered. Nothing in this module
# ever ranks a directory listing, consults a modification time, expands a glob,
# or picks a "latest" campaign, and there is no route that could create,
# replace, rerun, repair, resume, or add a fourth final campaign.
# ---------------------------------------------------------------------------
PILOT_CAMPAIGN_ID = "20260812T013848Z"
FINAL_CAMPAIGN_IDS = (
    "20260817T110330Z",
    "20260817T111310Z",
    "20260817T112011Z",
)
FINAL_EXECUTION_COMMIT = "b08e45c2636a3ac17c94ad8b1368084914196d7a"
CAMPAIGN_COUNT = len(FINAL_CAMPAIGN_IDS)

PILOT_ROLE = (
    "orchestration qualification evidence only; excluded from every P4.3 "
    "statistic, ranking, variability estimate, table, figure, and conclusion"
)

# The figure captions. The min-max range is drawn as a vertical line, so it is
# named a whisker; calling it a bar misdescribes the geometry and invites a
# bar-chart reading of a three-value range.
WHISKER_CAPTION = (
    f"the vertical line is the min-max whisker over exactly {CAMPAIGN_COUNT} "
    f"campaign-level values, one per final campaign")

# ---------------------------------------------------------------------------
# The frozen cross-campaign statistical policy (src/phase4/P4_3_PROTOCOL.md
# section 4). The independent replicate is one complete final campaign.
# ---------------------------------------------------------------------------
CV_REVIEW_THRESHOLD_PERCENT = 5.0
NOT_APPLICABLE = "not_applicable"
CV_FLAG_OK = "ok"
CV_FLAG_REVIEW = "REVIEW"

# Serialization precision. Full precision is retained throughout the
# computation; these decimals are applied only when a value is serialized.
DECIMALS_DEFAULT = 6
DECIMALS_BY_METRIC = {
    "median_effective_gbps": 6,
    "tma_to_ldgsts_ratio": 9,
    "dram_read_ratio": 9,
    "median_flops_per_cycle": 6,
    "median_flops_per_cycle_per_sm": 6,
    "estimated_tflops_per_sm": 9,
    "speedup_2sm_over_1sm": 9,
    "scaling_efficiency_percent": 6,
    "kernel_time_ms": 6,
    "tflops": 6,
    "throughput_ratio_vs_cublaslt": 9,
    "gap_to_cublaslt_pct": 6,
}
DECIMALS_CV = 6

# Metrics whose values are signed or may legitimately sit at or near zero. A
# coefficient of variation is meaningless for them and is never computed.
SIGNED_OR_ZERO_CENTRED_METRICS = frozenset({"gap_to_cublaslt_pct"})

# ---------------------------------------------------------------------------
# The frozen experimental grids, restated independently of the closed units --
# exactly as P4.2 restates P4.1's nine-stage plan -- so that a silently
# reordered, inserted, or dropped row in an already accepted artifact is
# rejected here too instead of being aggregated.
# ---------------------------------------------------------------------------
P14_METHODS = ("ldgsts", "tma")
P14_STAGES = (2, 4, 8)
P14_BIF_KIB = (16, 32, 64)
# analysis/pilot_statistics.csv row order.
P14_CONFIG_KEYS = tuple(
    (method, stages, bif)
    for stages in P14_STAGES for bif in P14_BIF_KIB for method in P14_METHODS
)
# analysis/pairwise_comparison.csv row order.
P14_PAIR_KEYS = tuple((stages, bif) for stages in P14_STAGES for bif in P14_BIF_KIB)
# analysis/saturation_candidates.csv row order.
P14_SATURATION_KEYS = tuple((method, stages) for method in P14_METHODS for stages in P14_STAGES)
# analysis/ncu_validation.csv: the frozen six-case Nsight Compute plan, in the
# order P1.4 profiles it. NCU covers exactly these six of the eighteen
# configurations and is never extrapolated to the other twelve.
P14_NCU_CASES = (
    (0, "ldgsts", 2, 16),
    (1, "tma", 2, 16),
    (2, "tma", 4, 32),
    (3, "ldgsts", 4, 32),
    (4, "ldgsts", 8, 64),
    (5, "tma", 8, 64),
)

P24_METHODS = ("umma_1sm", "umma_2sm")
P24_CTA_GROUP = {"umma_1sm": 1, "umma_2sm": 2}
P24_N_VALUES = (64, 128, 256)
P24_DEPTH_VALUES = (4, 16, 64, 256)
# analysis/configuration_statistics.csv row order.
P24_CONFIG_KEYS = tuple(
    (method, n, depth)
    for n in P24_N_VALUES for depth in P24_DEPTH_VALUES for method in P24_METHODS
)
# analysis/scaling.csv row order.
P24_SCALING_KEYS = tuple((n, depth) for n in P24_N_VALUES for depth in P24_DEPTH_VALUES)
# analysis/saturation.csv row order.
P24_SATURATION_KEYS = tuple((method, n) for method in P24_METHODS for n in P24_N_VALUES)
P24_PROFILE_CASE_COUNT = len(P24_CONFIG_KEYS)

# ---------------------------------------------------------------------------
# The canonical terminal artifacts P4.3 reads. Every one of them is pinned by
# the closed unit's own manifest and was already re-hashed by P4.1 during the
# terminal revalidation P4.2 drives; P4.3 re-verifies the bytes it actually
# reads against that same pin before parsing them.
# ---------------------------------------------------------------------------
P14_ARTIFACTS = (
    "analysis/pilot_statistics.csv",
    "analysis/pairwise_comparison.csv",
    "analysis/saturation_candidates.csv",
    "analysis/ncu_validation.csv",
)
P24_ARTIFACTS = (
    "analysis/configuration_statistics.csv",
    "analysis/scaling.csv",
    "analysis/saturation.csv",
    "analysis/profile_validation.csv",
    "analysis/empirical_ceiling.json",
)

STAGE_MEMORY_ANALYZE = "memory.analyze"
STAGE_UMMA_ANALYZE = "umma.analyze"
STAGE_GEMM_CAPTURE = "gemm.capture"
UNIT_TERMINAL_ANALYZED = "ANALYZED"

# ---------------------------------------------------------------------------
# The frozen output inventory. Nothing else is ever created.
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_ROOT_REL = "results/phase4"
# The one logical destination production output may ever have, walked component
# by component from an already opened repository descriptor.
OUTPUT_ROOT_COMPONENTS = ("results", "phase4")
OUTPUT_FIGURES_DIR = "figures"
ARTIFACT_RELATIVE_PATHS = (
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
MANIFEST_RELATIVE_PATH = "analysis_manifest.json"
# Output roots that would put a derived artifact inside immutable raw evidence.
FORBIDDEN_OUTPUT_PREFIXES = (("results", "raw"), ("results", "preflight"))

RESEARCH_QUESTION = (
    "How do HBM-to-SMEM data movement and fifth-generation Tensor Core "
    "throughput constrain BF16 GEMM performance on NVIDIA GB300, and how "
    "closely can the CuTe DSL implementation approach cuBLASLt?"
)

# ---------------------------------------------------------------------------
# Frozen CSV schemas. One long-format table per experiment, so that every
# reported quantity carries its own metric name, unit, the three preserved
# campaign-level values, and the cross-campaign statistics beside it.
# ---------------------------------------------------------------------------
# The four cross-campaign statistic columns are spelled `cross_campaign_*` so
# that they can never be confused with a closed unit's own within-campaign
# stability diagnostics, which travel beside them in their own rows.
_STAT_FIELDS = (
    "campaign_count",
    "campaign_1_value",
    "campaign_2_value",
    "campaign_3_value",
    "mean",
    "median",
    "stdev_sample",
    "cross_campaign_cv_percent",
    "minimum",
    "maximum",
    "cross_campaign_cv_review_flag",
    "notes",
)
MEMORY_CSV_FIELDS = (
    "schema_version", "section", "method", "stages", "bytes_in_flight_kib",
    "metric", "unit", "evidence_class",
) + _STAT_FIELDS
UMMA_CSV_FIELDS = (
    "schema_version", "section", "method", "n", "depth", "cta_group",
    "metric", "unit", "evidence_class",
) + _STAT_FIELDS
GEMM_CSV_FIELDS = (
    "schema_version", "section", "shape_index", "shape_id", "m", "n", "k", "l",
    "candidate_index", "variant", "method", "metric", "unit", "evidence_class",
) + _STAT_FIELDS

_DECIMAL_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_INTEGER_RE = re.compile(r"^-?\d+$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class P43Error(Exception):
    """Evidence, an artifact, or an output cannot be interpreted or accepted.

    Always fatal. P4.3 validates and reports; it never repairs, replaces,
    reruns, or works around anything."""


# ===========================================================================
# Module loading and small shared primitives.
# ===========================================================================


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise P43Error(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_repository_modules(repo_root: Path) -> tuple[object, object, object]:
    """The three audited repository modules P4.3 reuses instead of writing a
    second interpretation of the same contracts."""
    orchestrator = load_module(repo_root / ORCHESTRATOR_RELATIVE_PATH, "_p43_orchestrator")
    p42 = load_module(repo_root / P42_CHECKER_RELATIVE_PATH, "_p43_p42_checker")
    p35 = load_module(repo_root / P35_CHECKER_RELATIVE_PATH, "_p43_p35_checker")
    return orchestrator, p42, p35


def split_relative_path(relative: object) -> tuple[str, ...]:
    """A repository-relative path, validated before it reaches any syscall."""
    if not isinstance(relative, str) or not relative:
        raise P43Error(f"{relative!r}: not a repository-relative path")
    if relative.startswith("/") or "\\" in relative or "\0" in relative:
        raise P43Error(f"{relative!r}: not a repository-relative path")
    parts = tuple(relative.split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise P43Error(f"{relative!r}: contains an empty or traversing component")
    return parts


def read_repo_bytes(orchestrator, repo_root: Path, relative: str) -> bytes:
    """Read one repository-relative regular file through P4.1's own
    descriptor-anchored, symlink-rejecting primitives."""
    parts = split_relative_path(relative)
    try:
        directory_fd = orchestrator.open_dir_chain(repo_root, *parts[:-1])
    except orchestrator.OrchestratorError as exc:
        raise P43Error(f"{relative}: {exc}") from exc
    try:
        return orchestrator.read_file_nofollow(parts[-1], dir_fd=directory_fd)
    except orchestrator.OrchestratorError as exc:
        raise P43Error(f"{relative}: {exc}") from exc
    finally:
        os.close(directory_fd)


def decode_utf8(payload: bytes, label: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise P43Error(f"{label}: is not valid UTF-8: {exc}") from exc


def to_float(text: object, label: str, *, strictly_positive: bool = False) -> float:
    """A finite decimal. 'nan', 'inf', an empty cell, and any other
    non-decimal spelling are rejected rather than silently propagated."""
    if not isinstance(text, str) or not _DECIMAL_RE.fullmatch(text.strip()):
        raise P43Error(f"{label}: {text!r} is not a finite decimal value")
    value = float(text.strip())
    if not math.isfinite(value):
        raise P43Error(f"{label}: {text!r} is not finite")
    if strictly_positive and value <= 0.0:
        raise P43Error(f"{label}: {text!r} is not strictly positive")
    return value


def to_int(text: object, label: str) -> int:
    if not isinstance(text, str) or not _INTEGER_RE.fullmatch(text.strip()):
        raise P43Error(f"{label}: {text!r} is not an integer")
    return int(text.strip())


def normalize_zero(value: float) -> float:
    """A negative zero and a positive zero must serialize identically."""
    return 0.0 if value == 0.0 else value


def decimals_for(metric: str) -> int:
    return DECIMALS_BY_METRIC.get(metric, DECIMALS_DEFAULT)


def format_decimal(value: object, decimals: int) -> str:
    if value is None:
        return NOT_APPLICABLE
    return f"{normalize_zero(float(value)):.{decimals}f}"


def quantize(value: object, decimals: int):
    """The one place a computed value loses precision: serialization."""
    if value is None:
        return None
    return normalize_zero(float(f"{normalize_zero(float(value)):.{decimals}f}"))


# ===========================================================================
# The cross-campaign statistics. The independent replicate is one complete
# final campaign; internal timing repetitions are never pooled.
# ===========================================================================


def summarize_metric(values: list[float], *, metric: str, allow_cv: bool = True) -> dict:
    """Summarize exactly the three campaign-level values of one quantity.

    Only mean, median, sample standard deviation (n - 1), coefficient of
    variation where it is mathematically meaningful, minimum, and maximum are
    computed. No observation and no campaign is ever removed, no outlier filter
    runs, no p-value or significance claim is produced, and no confidence
    interval is bootstrapped from three campaigns.
    """
    if not isinstance(values, list) or len(values) != CAMPAIGN_COUNT:
        raise P43Error(
            f"{metric}: expected exactly {CAMPAIGN_COUNT} campaign-level values "
            f"(one per final campaign), got {len(values) if isinstance(values, list) else values!r}; "
            f"a campaign's internal timing repetitions are never pooled into the "
            f"cross-campaign sample")
    for index, value in enumerate(values, start=1):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise P43Error(f"{metric}: campaign {index} value {value!r} is not numeric")
        if not math.isfinite(float(value)):
            raise P43Error(f"{metric}: campaign {index} value {value!r} is not finite")
    numbers = [float(value) for value in values]
    count = len(numbers)
    mean = math.fsum(numbers) / count
    ordered = sorted(numbers)
    median = (ordered[count // 2] if count % 2
              else (ordered[count // 2 - 1] + ordered[count // 2]) / 2.0)
    # Sample standard deviation: the n - 1 denominator, never n.
    variance = math.fsum((value - mean) ** 2 for value in numbers) / (count - 1)
    stdev = math.sqrt(variance)

    cv_percent = None
    cv_flag = NOT_APPLICABLE
    cv_reason = "not computed"
    if metric in SIGNED_OR_ZERO_CENTRED_METRICS:
        cv_reason = ("a coefficient of variation is not meaningful for a signed or "
                     "zero-centred quantity")
    elif not allow_cv:
        cv_reason = "the metric is not a strictly positive performance quantity"
    elif all(value > 0.0 for value in numbers) and mean > 0.0:
        cv_percent = 100.0 * stdev / mean
        cv_flag = CV_FLAG_REVIEW if cv_percent > CV_REVIEW_THRESHOLD_PERCENT else CV_FLAG_OK
        cv_reason = "computed"
    else:
        cv_reason = "at least one campaign value is not strictly positive"

    return {
        "metric": metric,
        "campaign_count": count,
        "campaign_values": numbers,
        "mean": mean,
        "median": median,
        "stdev_sample": stdev,
        "cross_campaign_cv_percent": cv_percent,
        "cross_campaign_cv_reason": cv_reason,
        "minimum": min(numbers),
        "maximum": max(numbers),
        "cross_campaign_cv_review_flag": cv_flag,
    }


def consensus(values: list, *, label: str) -> dict:
    """Preserve every campaign's own result and report a single consensus only
    when all three campaigns agree. A disagreement is reported, never resolved
    by majority, by preference, or by rerunning anything."""
    if not isinstance(values, list) or len(values) != CAMPAIGN_COUNT:
        raise P43Error(f"{label}: expected exactly {CAMPAIGN_COUNT} campaign-level results")
    distinct = {json.dumps(value, sort_keys=True, default=str) for value in values}
    stable = len(distinct) == 1
    return {
        "campaign_values": list(values),
        "stable_across_campaigns": stable,
        "consensus": values[0] if stable else None,
        "note": ("all three final campaigns agree" if stable
                 else "no cross-campaign consensus exists; every campaign's own result is kept"),
    }


def evidence_class_for(metric: str) -> str:
    """The frozen evidence class of one metric's per-campaign value.

    A metric with no declared classification is a programming error, not a
    reason to publish an unclassified number."""
    if metric not in METRIC_EVIDENCE:
        raise P43Error(f"{metric}: has no declared scientific evidence classification; "
                       f"no quantity is ever emitted unclassified")
    return METRIC_EVIDENCE[metric][0]


def evidence_basis_for(metric: str) -> str:
    if metric not in METRIC_EVIDENCE:
        raise P43Error(f"{metric}: has no declared scientific evidence classification; "
                       f"no quantity is ever emitted unclassified")
    return METRIC_EVIDENCE[metric][1]


def evidence_json(metric: str) -> dict:
    """The classification carried beside every reported quantity."""
    return {
        "campaign_value_evidence_class": evidence_class_for(metric),
        "campaign_value_evidence_basis": evidence_basis_for(metric),
        "cross_campaign_statistics_evidence_class": EVIDENCE_CROSS_CAMPAIGN,
    }


def campaign_value_column_map() -> dict:
    """The deterministic mapping from every campaign value column to the exact
    campaign that produced it. Without it a `campaign_2_value` cell is an
    anonymous number."""
    return {f"campaign_{index + 1}_value": campaign_id
            for index, campaign_id in enumerate(FINAL_CAMPAIGN_IDS)}


def stat_cells(summary: dict | None, *, metric: str, notes: str,
               raw_values: list | None = None) -> list[str]:
    """The twelve shared statistic cells of one CSV row. A quantity for which a
    statistic is deliberately not computed carries the canonical
    ``not_applicable`` token, never a fabricated number."""
    decimals = decimals_for(metric)
    if summary is None:
        # A quantity that is deliberately not summarized still preserves every
        # campaign's own result; a campaign that recorded none carries the
        # canonical token rather than a fabricated or stringified null.
        values = [NOT_APPLICABLE if raw_values is None or raw_values[index] is None
                  else str(raw_values[index]) for index in range(CAMPAIGN_COUNT)]
        return ([str(CAMPAIGN_COUNT)] + values
                + [NOT_APPLICABLE] * 6 + [NOT_APPLICABLE, notes])
    return [
        str(summary["campaign_count"]),
        *[format_decimal(value, decimals) for value in summary["campaign_values"]],
        format_decimal(summary["mean"], decimals),
        format_decimal(summary["median"], decimals),
        format_decimal(summary["stdev_sample"], decimals),
        format_decimal(summary["cross_campaign_cv_percent"], DECIMALS_CV),
        format_decimal(summary["minimum"], decimals),
        format_decimal(summary["maximum"], decimals),
        summary["cross_campaign_cv_review_flag"],
        notes,
    ]


def diagnostic_cells(values: list, *, metric: str, notes: str) -> list[str]:
    """One preserved source diagnostic, in frozen campaign order.

    A diagnostic is never summarized: it carries each campaign's own value and
    the canonical `not_applicable` token in every statistic cell, so that a
    within-campaign flag can never be mistaken for a cross-campaign number."""
    if not isinstance(values, list) or len(values) != CAMPAIGN_COUNT:
        raise P43Error(f"{metric}: expected exactly {CAMPAIGN_COUNT} campaign-level "
                       f"diagnostic values, one per final campaign, in the frozen order")
    rendered = [NOT_APPLICABLE if value is None else str(value) for value in values]
    return ([str(CAMPAIGN_COUNT)] + rendered + [NOT_APPLICABLE] * 6
            + [NOT_APPLICABLE, notes])


def diagnostic_json(values: list, *, metric: str) -> dict:
    if not isinstance(values, list) or len(values) != CAMPAIGN_COUNT:
        raise P43Error(f"{metric}: expected exactly {CAMPAIGN_COUNT} campaign-level "
                       f"diagnostic values, one per final campaign, in the frozen order")
    return {
        "metric": metric,
        "campaign_count": CAMPAIGN_COUNT,
        "campaign_values": list(values),
        "statistics": NOT_APPLICABLE,
        **evidence_json(metric),
    }


def summary_json(summary: dict | None, *, metric: str,
                 raw_values: list | None = None) -> dict:
    decimals = decimals_for(metric)
    if summary is None:
        return {
            "metric": metric,
            "campaign_count": CAMPAIGN_COUNT,
            "campaign_values": list(raw_values or []),
            "statistics": NOT_APPLICABLE,
            **evidence_json(metric),
        }
    return {
        "metric": metric,
        "campaign_count": summary["campaign_count"],
        **evidence_json(metric),
        "campaign_values": [quantize(value, decimals) for value in summary["campaign_values"]],
        "mean": quantize(summary["mean"], decimals),
        "median": quantize(summary["median"], decimals),
        "stdev_sample": quantize(summary["stdev_sample"], decimals),
        "cross_campaign_cv_percent": quantize(summary["cross_campaign_cv_percent"], DECIMALS_CV),
        "cross_campaign_cv_status": summary["cross_campaign_cv_reason"],
        "cross_campaign_cv_review_flag": summary["cross_campaign_cv_review_flag"],
        "minimum": quantize(summary["minimum"], decimals),
        "maximum": quantize(summary["maximum"], decimals),
    }


# ===========================================================================
# Strict table reading. Missing, duplicate, reordered, or malformed rows are
# rejected; nothing is inferred, reordered, or filled in.
# ===========================================================================


def read_table(text: str, *, label: str, required_fields: tuple[str, ...]) -> list[dict]:
    reader = csv.reader(io.StringIO(text, newline=""))
    try:
        rows = list(reader)
    except csv.Error as exc:
        raise P43Error(f"{label}: malformed CSV: {exc}") from exc
    if not rows:
        raise P43Error(f"{label}: is empty")
    header = rows[0]
    if len(set(header)) != len(header):
        raise P43Error(f"{label}: the CSV header repeats a field name")
    missing = [field for field in required_fields if field not in header]
    if missing:
        raise P43Error(f"{label}: the CSV header is missing required field(s) {missing}")
    records = []
    for number, row in enumerate(rows[1:], start=2):
        if len(row) != len(header):
            raise P43Error(
                f"{label}: line {number} has {len(row)} cell(s), expected {len(header)}")
        records.append(dict(zip(header, row)))
    return records


def require_row_keys(records: list[dict], *, label: str, expected_keys: tuple,
                     key_fields: tuple[str, ...]) -> None:
    """The rows must be exactly the frozen keys, once each, in the frozen
    order. A missing, duplicated, reordered, or substituted row is fatal."""
    observed = []
    for number, record in enumerate(records, start=1):
        key = []
        for field in key_fields:
            raw = record.get(field)
            if raw is None:
                raise P43Error(f"{label}: row {number} has no {field!r} cell")
            text = raw.strip()
            key.append(int(text) if _INTEGER_RE.fullmatch(text) else text)
        observed.append(tuple(key))
    expected = [tuple(key) if isinstance(key, tuple) else (key,) for key in expected_keys]
    if observed != expected:
        raise P43Error(
            f"{label}: the rows are not the frozen population exactly once each in the "
            f"frozen order (got {observed[:4]}... expected {expected[:4]}...)")


# ===========================================================================
# The canonical terminal artifacts of the three closed units.
# ===========================================================================


def parse_p14_pilot_statistics(text: str, *, label: str) -> dict:
    records = read_table(text, label=label, required_fields=(
        "method", "stages", "bytes_in_flight_kib", "sample_count", "median_gbps",
        "cv_percent", "stability_review"))
    require_row_keys(records, label=label, expected_keys=P14_CONFIG_KEYS,
                     key_fields=("method", "stages", "bytes_in_flight_kib"))
    parsed = {}
    for record, key in zip(records, P14_CONFIG_KEYS):
        parsed[key] = {
            "median_gbps": to_float(record["median_gbps"], f"{label}: {key}: median_gbps",
                                    strictly_positive=True),
            "within_campaign_cv_percent": to_float(
                record["cv_percent"], f"{label}: {key}: cv_percent"),
            "within_campaign_sample_count": to_int(
                record["sample_count"], f"{label}: {key}: sample_count"),
            "within_campaign_stability_review": record["stability_review"].strip(),
        }
    return parsed


def parse_p14_pairwise(text: str, *, label: str) -> dict:
    records = read_table(text, label=label, required_fields=(
        "stages", "bytes_in_flight_kib", "tma_to_ldgsts_ratio", "interpretation",
        "median_gbps_ldgsts", "median_gbps_tma"))
    require_row_keys(records, label=label, expected_keys=P14_PAIR_KEYS,
                     key_fields=("stages", "bytes_in_flight_kib"))
    parsed = {}
    for record, key in zip(records, P14_PAIR_KEYS):
        parsed[key] = {
            "tma_to_ldgsts_ratio": to_float(
                record["tma_to_ldgsts_ratio"], f"{label}: {key}: tma_to_ldgsts_ratio",
                strictly_positive=True),
            "interpretation": record["interpretation"].strip(),
        }
    return parsed


def parse_p14_saturation(text: str, *, label: str) -> dict:
    field = "earliest_tested_candidate_saturation_bif_kib"
    records = read_table(text, label=label, required_fields=("method", "stages", field))
    require_row_keys(records, label=label, expected_keys=P14_SATURATION_KEYS,
                     key_fields=("method", "stages"))
    parsed = {}
    for record, key in zip(records, P14_SATURATION_KEYS):
        raw = record[field].strip()
        parsed[key] = None if raw == "" else to_int(raw, f"{label}: {key}: {field}")
    return parsed


def parse_p14_ncu(text: str, *, label: str) -> dict:
    records = read_table(text, label=label, required_fields=(
        "index", "method", "stages", "bytes_in_flight_kib", "dram_read_ratio",
        "hbm_classification", "diagnostic_flags"))
    require_row_keys(records, label=label, expected_keys=P14_NCU_CASES,
                     key_fields=("index", "method", "stages", "bytes_in_flight_kib"))
    parsed = {}
    for record, key in zip(records, P14_NCU_CASES):
        parsed[key] = {
            "dram_read_ratio": to_float(
                record["dram_read_ratio"], f"{label}: {key}: dram_read_ratio",
                strictly_positive=True),
            "hbm_classification": record["hbm_classification"].strip(),
            "diagnostic_flags": record["diagnostic_flags"].strip(),
        }
    return parsed


def parse_p24_configuration_statistics(text: str, *, label: str) -> dict:
    records = read_table(text, label=label, required_fields=(
        "method", "n", "depth", "cta_group", "sample_count",
        "flops_per_cycle_median", "flops_per_cycle_per_sm_median",
        "flops_per_cycle_cv_percent", "flops_per_cycle_stability_review"))
    require_row_keys(records, label=label, expected_keys=P24_CONFIG_KEYS,
                     key_fields=("method", "n", "depth"))
    parsed = {}
    for record, key in zip(records, P24_CONFIG_KEYS):
        cta_group = to_int(record["cta_group"], f"{label}: {key}: cta_group")
        if cta_group != P24_CTA_GROUP[key[0]]:
            raise P43Error(f"{label}: {key}: cta_group={cta_group} contradicts the method")
        parsed[key] = {
            "cta_group": cta_group,
            "median_flops_per_cycle": to_float(
                record["flops_per_cycle_median"], f"{label}: {key}: flops_per_cycle_median",
                strictly_positive=True),
            "median_flops_per_cycle_per_sm": to_float(
                record["flops_per_cycle_per_sm_median"],
                f"{label}: {key}: flops_per_cycle_per_sm_median", strictly_positive=True),
            "within_campaign_cv_percent": to_float(
                record["flops_per_cycle_cv_percent"],
                f"{label}: {key}: flops_per_cycle_cv_percent"),
            "within_campaign_sample_count": to_int(
                record["sample_count"], f"{label}: {key}: sample_count"),
            # P2.4's own within-campaign stability diagnostic. It is parsed and
            # carried through to the curated outputs; it is never conflated
            # with, replaced by, or overridden by the cross-campaign CV.
            "within_campaign_stability_review": record[
                "flops_per_cycle_stability_review"].strip(),
        }
    return parsed


def parse_p24_scaling(text: str, *, label: str) -> dict:
    records = read_table(text, label=label, required_fields=(
        "n", "depth", "speedup_2sm_over_1sm", "scaling_efficiency_percent",
        "surprising_value_flag"))
    require_row_keys(records, label=label, expected_keys=P24_SCALING_KEYS,
                     key_fields=("n", "depth"))
    parsed = {}
    for record, key in zip(records, P24_SCALING_KEYS):
        # Never clamped: a value outside [0, 100] is preserved exactly as the
        # closed unit recorded it, together with its surprising-value flag.
        parsed[key] = {
            "speedup_2sm_over_1sm": to_float(
                record["speedup_2sm_over_1sm"], f"{label}: {key}: speedup_2sm_over_1sm",
                strictly_positive=True),
            "scaling_efficiency_percent": to_float(
                record["scaling_efficiency_percent"],
                f"{label}: {key}: scaling_efficiency_percent"),
            "surprising_value_flag": record["surprising_value_flag"].strip(),
        }
    return parsed


def parse_p24_saturation(text: str, *, label: str) -> dict:
    field = "earliest_tested_candidate_saturation_depth"
    records = read_table(text, label=label, required_fields=("method", "n", field))
    require_row_keys(records, label=label, expected_keys=P24_SATURATION_KEYS,
                     key_fields=("method", "n"))
    parsed = {}
    for record, key in zip(records, P24_SATURATION_KEYS):
        raw = record[field].strip()
        parsed[key] = None if raw == "" else to_int(raw, f"{label}: {key}: {field}")
    return parsed


def parse_p24_profile_validation(text: str, *, label: str) -> dict:
    records = read_table(text, label=label, required_fields=(
        "index", "case_name", "method", "n", "depth", "sm_clock_status"))
    if len(records) != P24_PROFILE_CASE_COUNT:
        raise P43Error(f"{label}: has {len(records)} row(s), expected "
                       f"{P24_PROFILE_CASE_COUNT}")
    observed_configurations = []
    for number, record in enumerate(records):
        index = to_int(record["index"], f"{label}: row {number + 1}: index")
        if index != number:
            raise P43Error(f"{label}: row {number + 1} carries index {index}; the profile "
                           f"rows must be 0..{P24_PROFILE_CASE_COUNT - 1} in order")
        observed_configurations.append((
            record["method"].strip(),
            to_int(record["n"], f"{label}: row {number + 1}: n"),
            to_int(record["depth"], f"{label}: row {number + 1}: depth"),
        ))
    if sorted(observed_configurations) != sorted(P24_CONFIG_KEYS):
        raise P43Error(f"{label}: the profiled configurations are not the frozen "
                       f"{P24_PROFILE_CASE_COUNT}-configuration plan exactly once each")
    statuses = [record["sm_clock_status"].strip() for record in records]
    return {
        "case_count": len(records),
        "sm_clock_ok_count": sum(1 for status in statuses if status == "OK"),
        "sm_clock_statuses_distinct": sorted(set(statuses)),
    }


def parse_p24_empirical_ceiling(payload: bytes, *, label: str) -> dict:
    try:
        document = json.loads(decode_utf8(payload, label))
    except json.JSONDecodeError as exc:
        raise P43Error(f"{label}: invalid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise P43Error(f"{label}: is not a JSON object")
    if document.get("status") != UNIT_TERMINAL_ANALYZED:
        raise P43Error(f"{label}: status={document.get('status')!r}, not "
                       f"{UNIT_TERMINAL_ANALYZED!r}")
    if document.get("publishable") is not False:
        raise P43Error(f"{label}: publishable is not false")
    candidate = document.get("empirical_per_sm_ceiling_candidate")
    if not isinstance(candidate, dict):
        raise P43Error(f"{label}: carries no empirical_per_sm_ceiling_candidate object")
    for field in ("method", "n", "depth", "case_name", "estimated_tflops_per_sm",
                  "median_flops_per_cycle_per_sm", "sm_clock_valid"):
        if field not in candidate:
            raise P43Error(f"{label}: the ceiling candidate has no {field!r}")
    if candidate["sm_clock_valid"] is not True:
        raise P43Error(f"{label}: the selected ceiling candidate's SM-clock reading is not "
                       f"marked valid; no TFLOP/s conversion may be aggregated from it")
    tflops = candidate["estimated_tflops_per_sm"]
    if isinstance(tflops, bool) or not isinstance(tflops, (int, float)) \
            or not math.isfinite(float(tflops)) or float(tflops) <= 0.0:
        raise P43Error(f"{label}: estimated_tflops_per_sm={tflops!r} is not a finite "
                       f"strictly positive number")
    device = document.get("device_equivalent_estimate")
    if not isinstance(device, dict) or "available" not in device:
        raise P43Error(f"{label}: carries no device_equivalent_estimate object")
    return {
        "selected_configuration": {
            "method": str(candidate["method"]),
            "n": int(candidate["n"]),
            "depth": int(candidate["depth"]),
            "case_name": str(candidate["case_name"]),
        },
        "estimated_tflops_per_sm": float(tflops),
        "median_flops_per_cycle_per_sm": float(candidate["median_flops_per_cycle_per_sm"]),
        "sm_clock_valid": True,
        "device_equivalent_estimate": device,
    }


def parse_p35_capture(text: str, *, label: str, p35) -> dict:
    """Validate the captured P3.5 comparison through P3.5's own canonical
    validator, then read the four comparison quantities per row."""
    errors = list(p35.validate_serialized_output(text))
    if errors:
        raise P43Error(f"{label}: P3.5's own validator rejected the capture: {errors}")
    rows = [dict(row) for row in csv.DictReader(io.StringIO(text))]
    parsed: dict[tuple[int, int], dict] = {}
    best_by_shape: dict[int, str] = {}
    for row in rows:
        shape_index = to_int(row["shape_index"], f"{label}: shape_index")
        candidate_index = to_int(row["candidate_index"], f"{label}: candidate_index")
        key = (shape_index, candidate_index)
        for field, expected in (("schema_version", "p35.v1"), ("unit", "P3.5"),
                                ("run_kind", "smoke"), ("publishable", "false"),
                                ("correctness", "PASS"), ("git_dirty", "false")):
            if row.get(field) != expected:
                raise P43Error(f"{label}: {key}: {field}={row.get(field)!r}, expected "
                               f"{expected!r}")
        parsed[key] = {
            "shape_id": row["shape_id"],
            "m": to_int(row["m"], f"{label}: {key}: m"),
            "n": to_int(row["n"], f"{label}: {key}: n"),
            "k": to_int(row["k"], f"{label}: {key}: k"),
            "l": to_int(row["l"], f"{label}: {key}: l"),
            "variant": row["variant"],
            "method": row["method"],
            "kernel_time_ms": to_float(row["kernel_time_ms"],
                                       f"{label}: {key}: kernel_time_ms",
                                       strictly_positive=True),
            "tflops": to_float(row["tflops"], f"{label}: {key}: tflops",
                               strictly_positive=True),
            # The cuBLASLt-relative ratio and gap are taken from the campaign
            # that computed them; they are never re-derived from aggregates.
            "throughput_ratio_vs_cublaslt": to_float(
                row["throughput_ratio_vs_cublaslt"],
                f"{label}: {key}: throughput_ratio_vs_cublaslt", strictly_positive=True),
            # A negative gap means the candidate measured faster than cuBLASLt.
            # It is preserved exactly and never clamped.
            "gap_to_cublaslt_pct": to_float(row["gap_to_cublaslt_pct"],
                                            f"{label}: {key}: gap_to_cublaslt_pct"),
            "cache_mode": row["cache_mode"],
            "is_best_cutedsl": row["is_best_cutedsl"],
        }
        best_by_shape.setdefault(shape_index, row["best_cutedsl_variant"])
        if best_by_shape[shape_index] != row["best_cutedsl_variant"]:
            raise P43Error(f"{label}: shape {shape_index} disagrees with itself on "
                           f"best_cutedsl_variant")
    return {"rows": parsed, "best_cutedsl_variant": best_by_shape,
            "cache_mode": rows[0]["cache_mode"]}


# ===========================================================================
# Evidence collection. Read-only from the first call to the last.
# ===========================================================================


def load_phase4_manifest(orchestrator, repo_root: Path, campaign_id: str) -> dict:
    """The top-level campaign manifest, through P4.1's own audited loader."""
    try:
        orchestrator.validate_campaign_id(campaign_id)
        campaign_dir = orchestrator.resolve_campaign_tree(campaign_id, repo_root=repo_root)
        manifest, revision = orchestrator.load_manifest_chain(campaign_dir, repo_root=repo_root)
    except (orchestrator.OrchestratorError, orchestrator.ManifestError) as exc:
        raise P43Error(f"{campaign_id}: {exc}") from exc
    if revision < 0 or not manifest:
        raise P43Error(f"{campaign_id}: carries no manifest revision")
    return manifest


def require_terminal_unit_stage(manifest: dict, *, campaign_id: str, stage: str,
                                unit: str) -> dict:
    results = manifest.get("stage_results")
    if not isinstance(results, dict) or stage not in results:
        raise P43Error(f"{campaign_id}: has no completed {stage} stage; an incomplete or "
                       f"non-terminal campaign is never analysed")
    evidence = results[stage].get("evidence")
    if not isinstance(evidence, dict):
        raise P43Error(f"{campaign_id}: {stage}: carries no evidence object")
    if evidence.get("unit") != unit:
        raise P43Error(f"{campaign_id}: {stage}: unit={evidence.get('unit')!r}, expected "
                       f"{unit!r}")
    if evidence.get("unit_state") != UNIT_TERMINAL_ANALYZED:
        raise P43Error(f"{campaign_id}: {stage}: the {unit} campaign is in state "
                       f"{evidence.get('unit_state')!r}, not {UNIT_TERMINAL_ANALYZED!r}")
    for field in ("unit_campaign_dir", "unit_manifest_path", "unit_manifest_sha256"):
        if not isinstance(evidence.get(field), str) or not evidence[field]:
            raise P43Error(f"{campaign_id}: {stage}: {field} is absent or malformed")
    if not _SHA256_RE.fullmatch(evidence["unit_manifest_sha256"]):
        raise P43Error(f"{campaign_id}: {stage}: unit_manifest_sha256 is not canonical")
    return evidence


def read_pinned_artifact(orchestrator, repo_root: Path, *, relative: str,
                         expected_sha256: str, label: str) -> bytes:
    payload = read_repo_bytes(orchestrator, repo_root, relative)
    digest = orchestrator.sha256_bytes(payload)
    if digest != expected_sha256:
        raise P43Error(
            f"{label}: {relative} hashes to {digest}, but the accepted evidence pins "
            f"{expected_sha256}; the artifact changed after it was accepted")
    return payload


def read_unit_artifacts(orchestrator, repo_root: Path, evidence: dict, *,
                        campaign_id: str, unit: str,
                        artifacts: tuple[str, ...]) -> tuple[dict, list[dict]]:
    """Read one closed unit's canonical terminal artifacts through the exact
    manifest revision the Phase 4 campaign accepted."""
    label = f"{campaign_id}: {unit}"
    manifest_payload = read_pinned_artifact(
        orchestrator, repo_root, relative=evidence["unit_manifest_path"],
        expected_sha256=evidence["unit_manifest_sha256"], label=label)
    try:
        unit_manifest = json.loads(decode_utf8(manifest_payload, label))
    except json.JSONDecodeError as exc:
        raise P43Error(f"{label}: {evidence['unit_manifest_path']}: invalid JSON: {exc}") from exc
    if not isinstance(unit_manifest, dict):
        raise P43Error(f"{label}: the unit manifest is not a JSON object")
    if unit_manifest.get("campaign_id") != campaign_id:
        raise P43Error(f"{label}: the unit manifest names campaign "
                       f"{unit_manifest.get('campaign_id')!r}, not {campaign_id!r}")
    if unit_manifest.get("state") != evidence["unit_state"]:
        raise P43Error(f"{label}: the unit manifest state {unit_manifest.get('state')!r} "
                       f"disagrees with the accepted {evidence['unit_state']!r}")
    if unit_manifest.get("publishable") is not False:
        raise P43Error(f"{label}: the unit manifest does not record publishable=false")
    recorded = unit_manifest.get("artifact_sha256")
    if not isinstance(recorded, dict):
        raise P43Error(f"{label}: the unit manifest carries no artifact_sha256 object")

    payloads: dict[str, bytes] = {}
    sources: list[dict] = []
    for relative in artifacts:
        pinned = recorded.get(relative)
        if not isinstance(pinned, str) or not _SHA256_RE.fullmatch(pinned):
            raise P43Error(f"{label}: artifact_sha256[{relative!r}] is absent or not a "
                           f"canonical SHA-256")
        full = f"{evidence['unit_campaign_dir']}/{relative}"
        payloads[relative] = read_pinned_artifact(
            orchestrator, repo_root, relative=full, expected_sha256=pinned, label=label)
        sources.append({"campaign_id": campaign_id, "unit": unit, "artifact": relative,
                        "repo_relative_path": full, "sha256": pinned})
    return payloads, sources


def collect_campaign_evidence(orchestrator, p35, repo_root: Path, campaign_id: str,
                              manifest: dict) -> dict:
    """Everything P4.3 reads from one already accepted final campaign."""
    if manifest.get("campaign_id") != campaign_id:
        raise P43Error(f"{campaign_id}: the manifest names a different campaign")
    if manifest.get("campaign_kind") != "final":
        raise P43Error(f"{campaign_id}: campaign_kind="
                       f"{manifest.get('campaign_kind')!r}, not 'final'")
    if manifest.get("state") != "COMPLETE" or manifest.get("outcome") != "COMPLETE":
        raise P43Error(f"{campaign_id}: is not terminally COMPLETE")
    if manifest.get("publishable") is not False:
        raise P43Error(f"{campaign_id}: does not record publishable=false")
    gpu = manifest.get("gpu")
    if not isinstance(gpu, dict) or sorted(gpu) != sorted(orchestrator.GPU_IDENTITY_FIELDS):
        raise P43Error(f"{campaign_id}: carries no complete validated GPU identity")

    memory_evidence = require_terminal_unit_stage(
        manifest, campaign_id=campaign_id, stage=STAGE_MEMORY_ANALYZE, unit="P1.4")
    umma_evidence = require_terminal_unit_stage(
        manifest, campaign_id=campaign_id, stage=STAGE_UMMA_ANALYZE, unit="P2.4")

    memory_payloads, memory_sources = read_unit_artifacts(
        orchestrator, repo_root, memory_evidence, campaign_id=campaign_id, unit="P1.4",
        artifacts=P14_ARTIFACTS)
    umma_payloads, umma_sources = read_unit_artifacts(
        orchestrator, repo_root, umma_evidence, campaign_id=campaign_id, unit="P2.4",
        artifacts=P24_ARTIFACTS)

    results = manifest.get("stage_results", {})
    if STAGE_GEMM_CAPTURE not in results:
        raise P43Error(f"{campaign_id}: has no completed {STAGE_GEMM_CAPTURE} stage")
    gemm_evidence = results[STAGE_GEMM_CAPTURE].get("evidence")
    if not isinstance(gemm_evidence, dict):
        raise P43Error(f"{campaign_id}: {STAGE_GEMM_CAPTURE}: carries no evidence object")
    for field in ("csv_path", "csv_sha256"):
        if not isinstance(gemm_evidence.get(field), str) or not gemm_evidence[field]:
            raise P43Error(f"{campaign_id}: {STAGE_GEMM_CAPTURE}: {field} is absent")
    if not _SHA256_RE.fullmatch(gemm_evidence["csv_sha256"]):
        raise P43Error(f"{campaign_id}: {STAGE_GEMM_CAPTURE}: csv_sha256 is not canonical")
    gemm_payload = read_pinned_artifact(
        orchestrator, repo_root, relative=gemm_evidence["csv_path"],
        expected_sha256=gemm_evidence["csv_sha256"], label=f"{campaign_id}: P3.5")
    gemm_sources = [{"campaign_id": campaign_id, "unit": "P3.5",
                     "artifact": "exp03/gemm_comparison.csv",
                     "repo_relative_path": gemm_evidence["csv_path"],
                     "sha256": gemm_evidence["csv_sha256"]}]

    label14 = f"{campaign_id}: P1.4"
    label24 = f"{campaign_id}: P2.4"
    memory = {
        "pilot_statistics": parse_p14_pilot_statistics(
            decode_utf8(memory_payloads["analysis/pilot_statistics.csv"], label14),
            label=f"{label14}: pilot_statistics.csv"),
        "pairwise": parse_p14_pairwise(
            decode_utf8(memory_payloads["analysis/pairwise_comparison.csv"], label14),
            label=f"{label14}: pairwise_comparison.csv"),
        "saturation": parse_p14_saturation(
            decode_utf8(memory_payloads["analysis/saturation_candidates.csv"], label14),
            label=f"{label14}: saturation_candidates.csv"),
        "ncu": parse_p14_ncu(
            decode_utf8(memory_payloads["analysis/ncu_validation.csv"], label14),
            label=f"{label14}: ncu_validation.csv"),
    }
    umma = {
        "configuration": parse_p24_configuration_statistics(
            decode_utf8(umma_payloads["analysis/configuration_statistics.csv"], label24),
            label=f"{label24}: configuration_statistics.csv"),
        "scaling": parse_p24_scaling(
            decode_utf8(umma_payloads["analysis/scaling.csv"], label24),
            label=f"{label24}: scaling.csv"),
        "saturation": parse_p24_saturation(
            decode_utf8(umma_payloads["analysis/saturation.csv"], label24),
            label=f"{label24}: saturation.csv"),
        "profile_validation": parse_p24_profile_validation(
            decode_utf8(umma_payloads["analysis/profile_validation.csv"], label24),
            label=f"{label24}: profile_validation.csv"),
        "ceiling": parse_p24_empirical_ceiling(
            umma_payloads["analysis/empirical_ceiling.json"],
            label=f"{label24}: empirical_ceiling.json"),
    }
    gemm = parse_p35_capture(decode_utf8(gemm_payload, f"{campaign_id}: P3.5"),
                             label=f"{campaign_id}: P3.5: gemm_comparison.csv", p35=p35)

    provenance = {}
    for field in orchestrator.COMPARABLE_EVIDENCE_FIELDS:
        observed = {record["evidence"][field]
                    for record in manifest["stage_results"].values()
                    if isinstance(record.get("evidence"), dict)
                    and record["evidence"].get(field) is not None}
        if len(observed) > 1:
            raise P43Error(f"{campaign_id}: its own components disagree on {field}")
        provenance[field] = observed.pop() if observed else None

    return {
        "campaign_id": campaign_id,
        "git_commit": manifest["git_commit"],
        "gpu": {field: gpu[field] for field in orchestrator.GPU_IDENTITY_FIELDS},
        "provenance": provenance,
        "memory": memory,
        "umma": umma,
        "gemm": gemm,
        "sources": memory_sources + umma_sources + gemm_sources,
    }


def validate_declared_population(pilot_ids: list[str], final_ids: list[str]) -> None:
    """The population is declared explicitly and must be exactly the frozen
    one, in the frozen order. Nothing is discovered, substituted, or reordered."""
    if list(pilot_ids) != [PILOT_CAMPAIGN_ID]:
        raise P43Error(
            f"exactly one pilot campaign ID must be declared and it must be the accepted "
            f"pilot {PILOT_CAMPAIGN_ID}; got {list(pilot_ids)}")
    if len(final_ids) != CAMPAIGN_COUNT:
        raise P43Error(f"exactly {CAMPAIGN_COUNT} final campaign IDs must be declared; got "
                       f"{len(final_ids)}: {list(final_ids)}")
    if len(set(final_ids)) != len(final_ids):
        raise P43Error(f"the declared final campaign IDs repeat: {list(final_ids)}")
    if PILOT_CAMPAIGN_ID in final_ids:
        raise P43Error(
            f"the accepted pilot {PILOT_CAMPAIGN_ID} is not one of the three final "
            f"replicates and must never enter a statistic")
    if tuple(final_ids) != FINAL_CAMPAIGN_IDS:
        raise P43Error(
            f"the declared final campaigns {list(final_ids)} are not the frozen population "
            f"{list(FINAL_CAMPAIGN_IDS)} in the frozen order")


def compare_campaign_provenance(records: list[dict]) -> None:
    """The three replicates ran from one execution commit on one device."""
    commits = {record["git_commit"] for record in records}
    if len(commits) != 1:
        raise P43Error(f"the three final campaigns do not share one execution commit: "
                       f"{sorted(commits)}")
    commit = commits.pop()
    if commit != FINAL_EXECUTION_COMMIT:
        raise P43Error(f"the three final campaigns ran from {commit}, not the frozen "
                       f"execution commit {FINAL_EXECUTION_COMMIT}")
    for field in ("uuid", "name", "compute_capability", "driver_version"):
        observed = {record["gpu"][field] for record in records}
        if len(observed) != 1:
            raise P43Error(f"the three final campaigns do not share one GPU {field}: "
                           f"{sorted(observed)}")
    keys = {tuple(sorted(record["provenance"])) for record in records}
    if len(keys) != 1:
        raise P43Error("the three final campaigns expose different provenance fields")
    for field in sorted(next(iter(keys))):
        observed = {record["provenance"][field] for record in records}
        if len(observed) != 1:
            raise P43Error(f"the three final campaigns do not share one {field}: "
                           f"{sorted(str(value) for value in observed)}")


# Every function that touches already accepted evidence. All of them are
# strictly read-only; scripts/check_phase4_integration_p43.py proves that
# mechanically from their own source.
EVIDENCE_MODE_FUNCTIONS = (
    split_relative_path,
    read_repo_bytes,
    load_phase4_manifest,
    require_terminal_unit_stage,
    read_pinned_artifact,
    read_unit_artifacts,
    collect_campaign_evidence,
    parse_p14_pilot_statistics,
    parse_p14_pairwise,
    parse_p14_saturation,
    parse_p14_ncu,
    parse_p24_configuration_statistics,
    parse_p24_scaling,
    parse_p24_saturation,
    parse_p24_profile_validation,
    parse_p24_empirical_ceiling,
    parse_p35_capture,
    validate_declared_population,
    compare_campaign_provenance,
)

# The functions that compute every reported quantity. None of them may pool a
# campaign's internal repetitions, remove an observation, bootstrap, or clamp.
STATISTICAL_FUNCTIONS = (
    summarize_metric,
    consensus,
    stat_cells,
    summary_json,
)


# ===========================================================================
# Experiment 1 -- LDGSTS versus TMA.
# ===========================================================================


def aggregate_experiment_1(records: list[dict]) -> tuple[list[list[str]], dict]:
    rows: list[list[str]] = []
    warnings: list[dict] = []
    # The frozen six-case NCU plan, as a lookup over the 18-configuration grid,
    # so that every configuration states explicitly whether actual HBM/DRAM
    # traffic evidence exists for it at all.
    profiled = {(method, stages, bif) for _, method, stages, bif in P14_NCU_CASES}
    configurations = []
    for method, stages, bif in P14_CONFIG_KEYS:
        key = (method, stages, bif)
        entries = [record["memory"]["pilot_statistics"][key] for record in records]
        values = [entry["median_gbps"] for entry in entries]
        summary = summarize_metric(values, metric="median_effective_gbps")
        coverage = NCU_PROFILED if key in profiled else NCU_NOT_PROFILED
        prefix = [SCHEMA_VERSION, "configuration", method, str(stages), str(bif)]
        rows.append(prefix + ["median_effective_gbps", "GB/s",
                              evidence_class_for("median_effective_gbps")]
                    + stat_cells(summary, metric="median_effective_gbps",
                                 notes="campaign_level_median_of_that_campaign_own_"
                                       "timing_derived_effective_transfer_rates"))
        # Every within-campaign diagnostic the closed unit recorded travels
        # with the value it belongs to, in the frozen campaign order, and is
        # never replaced by the cross-campaign statistic beside it.
        diagnostics = {}
        for metric, cells in (
                ("within_campaign_sample_count",
                 [entry["within_campaign_sample_count"] for entry in entries]),
                ("within_campaign_cv_percent",
                 [entry["within_campaign_cv_percent"] for entry in entries]),
                ("within_campaign_stability_review",
                 [entry["within_campaign_stability_review"] for entry in entries]),
                ("ncu_coverage", [coverage] * CAMPAIGN_COUNT)):
            rows.append(prefix + [metric, NOT_APPLICABLE, evidence_class_for(metric)]
                        + diagnostic_cells(cells, metric=metric,
                                           notes="preserved_source_diagnostic"))
            diagnostics[metric] = diagnostic_json(cells, metric=metric)
        for campaign_index, entry in enumerate(entries):
            review = entry["within_campaign_stability_review"]
            if review and review != "ok":
                warnings.append({
                    "campaign_id": records[campaign_index]["campaign_id"],
                    "campaign_position": campaign_index + 1,
                    "section": "experiment_1_configuration",
                    "method": method, "stages": stages, "bytes_in_flight_kib": bif,
                    "within_campaign_stability_review": review,
                })
        configurations.append({
            "method": method, "stages": stages, "bytes_in_flight_kib": bif,
            **summary_json(summary, metric="median_effective_gbps"),
            "ncu_coverage": coverage,
            "hbm_traffic_evidence": (
                "profiler-derived DRAM traffic exists for this configuration; see the "
                "ncu_validation section" if coverage == NCU_PROFILED
                else NCU_UNAVAILABLE_NOTE),
            "within_campaign_diagnostics": diagnostics,
        })

    pair_ratios = []
    for stages, bif in P14_PAIR_KEYS:
        key = (stages, bif)
        values = [record["memory"]["pairwise"][key]["tma_to_ldgsts_ratio"]
                  for record in records]
        interpretations = [record["memory"]["pairwise"][key]["interpretation"]
                           for record in records]
        agreement = consensus(interpretations, label=f"interpretation {key}")
        summary = summarize_metric(values, metric="tma_to_ldgsts_ratio")
        note = (f"interpretation={agreement['consensus']}" if agreement["consensus"]
                else "interpretation=mixed")
        rows.append([SCHEMA_VERSION, "pair_ratio", NOT_APPLICABLE, str(stages), str(bif),
                     "tma_to_ldgsts_ratio", "ratio",
                     evidence_class_for("tma_to_ldgsts_ratio")]
                    + stat_cells(summary, metric="tma_to_ldgsts_ratio", notes=note))
        pair_ratios.append({
            "stages": stages, "bytes_in_flight_kib": bif,
            **summary_json(summary, metric="tma_to_ldgsts_ratio"),
            "campaign_interpretations": interpretations,
            "interpretation_consensus": agreement["consensus"],
        })

    saturation = []
    for method, stages in P14_SATURATION_KEYS:
        key = (method, stages)
        values = [record["memory"]["saturation"][key] for record in records]
        agreement = consensus(values, label=f"saturation {key}")
        note = (f"consensus={agreement['consensus']}" if agreement["stable_across_campaigns"]
                else "no_cross_campaign_consensus")
        rows.append([SCHEMA_VERSION, "saturation", method, str(stages), NOT_APPLICABLE,
                     "earliest_tested_candidate_saturation_bif_kib", "KiB",
                     evidence_class_for("earliest_tested_candidate_saturation_bif_kib")]
                    + stat_cells(None, metric="earliest_tested_candidate_saturation_bif_kib",
                                 notes=note, raw_values=values))
        saturation.append({"method": method, "stages": stages,
                           "campaign_values_kib": values,
                           "stable_across_campaigns": agreement["stable_across_campaigns"],
                           "consensus_kib": agreement["consensus"],
                           "note": agreement["note"]})

    ncu = []
    for index, method, stages, bif in P14_NCU_CASES:
        key = (index, method, stages, bif)
        cases = [record["memory"]["ncu"][key] for record in records]
        values = [case["dram_read_ratio"] for case in cases]
        classifications = [case["hbm_classification"] for case in cases]
        flags = [case["diagnostic_flags"] for case in cases]
        agreement = consensus(classifications, label=f"hbm_classification {key}")
        summary = summarize_metric(values, metric="dram_read_ratio")
        note = (f"hbm={agreement['consensus']}" if agreement["consensus"] else "hbm=mixed")
        prefix = [SCHEMA_VERSION, "ncu_validation", method, str(stages), str(bif)]
        rows.append(prefix + ["dram_read_ratio", "ratio",
                              evidence_class_for("dram_read_ratio")]
                    + stat_cells(summary, metric="dram_read_ratio", notes=note))
        # The HBM classification and the per-case Nsight Compute diagnostics
        # are the closed unit's own terminal evidence. They are preserved per
        # campaign, in the frozen campaign order, and a non-empty flag such as
        # READ_AMPLIFICATION is surfaced rather than discarded.
        diagnostics = {}
        for metric, cells in (("hbm_classification", classifications),
                              ("diagnostic_flags", [flag or NOT_APPLICABLE
                                                    for flag in flags])):
            rows.append(prefix + [metric, NOT_APPLICABLE, evidence_class_for(metric)]
                        + diagnostic_cells(cells, metric=metric,
                                           notes="preserved_source_diagnostic"))
            diagnostics[metric] = diagnostic_json(cells, metric=metric)
        for campaign_index, flag in enumerate(flags):
            if flag:
                warnings.append({
                    "campaign_id": records[campaign_index]["campaign_id"],
                    "campaign_position": campaign_index + 1,
                    "section": "experiment_1_ncu_validation",
                    "case_index": index, "method": method, "stages": stages,
                    "bytes_in_flight_kib": bif,
                    "diagnostic_flags": flag,
                })
        ncu.append({"index": index, "method": method, "stages": stages,
                    "bytes_in_flight_kib": bif,
                    **summary_json(summary, metric="dram_read_ratio"),
                    "campaign_hbm_classifications": classifications,
                    "hbm_classification_consensus": agreement["consensus"],
                    "campaign_diagnostic_flags": list(flags),
                    "source_diagnostics": diagnostics})

    saturation_stable = all(entry["stable_across_campaigns"] for entry in saturation)
    unprofiled = [{"method": entry["method"], "stages": entry["stages"],
                   "bytes_in_flight_kib": entry["bytes_in_flight_kib"]}
                  for entry in configurations
                  if entry["ncu_coverage"] == NCU_NOT_PROFILED]
    section = {
        "title": "Experiment 1 -- LDGSTS versus TMA HBM-to-SMEM data movement",
        "configuration_count": len(P14_CONFIG_KEYS),
        "primary_metric_scope": (
            "median_effective_gbps is a timing-derived effective transfer rate of a "
            "dedicated streaming HBM-to-SMEM microbenchmark: the benchmark's logical "
            "useful_bytes divided by its measured kernel time. It is not directly "
            "measured HBM/DRAM bandwidth and it is not GEMM memory traffic"),
        "configurations": configurations,
        "pair_ratios": pair_ratios,
        "pair_ratio_interpretation": (
            "a value above one means TMA reached the higher effective transfer rate in "
            "that campaign, and a value below one means LDGSTS did; this is a derived "
            "within-campaign ratio of two campaign-level medians, not a directly "
            "measured quantity, not a winner, and not a significance claim"),
        "saturation_candidates": saturation,
        "saturation_consensus_available": saturation_stable,
        "ncu_validation": ncu,
        "ncu_coverage": {
            "profiled_cases": len(P14_NCU_CASES),
            "total_configurations": len(P14_CONFIG_KEYS),
            "unprofiled_configuration_count": len(unprofiled),
            "unprofiled_configurations": unprofiled,
            "limitation": (
                "Nsight Compute HBM/DRAM traffic validation covers exactly these six "
                "predefined cases and is never extrapolated to the other twelve "
                "configurations"),
            "unprofiled_status": NCU_UNAVAILABLE_NOTE,
            "separation": (
                "the profiler-derived dram_read_ratio and hbm_classification of these six "
                "cases are kept separate from the timing-derived effective transfer rate; "
                "neither validates nor calibrates the other"),
        },
        "source_warnings": warnings,
    }
    return rows, section


# ===========================================================================
# Experiment 2 -- BF16 UMMA throughput.
# ===========================================================================


def aggregate_experiment_2(records: list[dict]) -> tuple[list[list[str]], dict]:
    rows: list[list[str]] = []
    warnings: list[dict] = []
    configurations = []
    for method, n, depth in P24_CONFIG_KEYS:
        key = (method, n, depth)
        cta_group = P24_CTA_GROUP[method]
        sources = [record["umma"]["configuration"][key] for record in records]
        prefix = [SCHEMA_VERSION, "configuration", method, str(n), str(depth),
                  str(cta_group)]
        entry = {"method": method, "n": n, "depth": depth, "cta_group": cta_group,
                 "metrics": {}}
        for metric, unit in (("median_flops_per_cycle", "FLOP/cycle"),
                             ("median_flops_per_cycle_per_sm", "FLOP/cycle/SM")):
            values = [source[metric] for source in sources]
            summary = summarize_metric(values, metric=metric)
            rows.append(prefix + [metric, unit, evidence_class_for(metric)]
                        + stat_cells(summary, metric=metric,
                                     notes="clock_independent_operation_and_cycle_"
                                           "derived_campaign_level_median"))
            entry["metrics"][metric] = summary_json(summary, metric=metric)
        # P2.4's own within-campaign stability evidence for flops_per_cycle,
        # preserved per campaign beside the value it qualifies.
        diagnostics = {}
        for metric, cells in (
                ("within_campaign_sample_count",
                 [source["within_campaign_sample_count"] for source in sources]),
                ("within_campaign_cv_percent",
                 [source["within_campaign_cv_percent"] for source in sources]),
                ("within_campaign_stability_review",
                 [source["within_campaign_stability_review"] for source in sources])):
            rows.append(prefix + [metric, NOT_APPLICABLE, evidence_class_for(metric)]
                        + diagnostic_cells(cells, metric=metric,
                                           notes="preserved_source_diagnostic"))
            diagnostics[metric] = diagnostic_json(cells, metric=metric)
        for campaign_index, source in enumerate(sources):
            review = source["within_campaign_stability_review"]
            if review and review != "ok":
                warnings.append({
                    "campaign_id": records[campaign_index]["campaign_id"],
                    "campaign_position": campaign_index + 1,
                    "section": "experiment_2_configuration",
                    "method": method, "n": n, "depth": depth,
                    "within_campaign_stability_review": review,
                })
        entry["within_campaign_diagnostics"] = diagnostics
        configurations.append(entry)

    scaling = []
    for n, depth in P24_SCALING_KEYS:
        key = (n, depth)
        flags = [record["umma"]["scaling"][key]["surprising_value_flag"]
                 for record in records]
        prefix = [SCHEMA_VERSION, "scaling", NOT_APPLICABLE, str(n), str(depth),
                  NOT_APPLICABLE]
        entry = {"n": n, "depth": depth, "campaign_surprising_value_flags": flags,
                 "metrics": {}}
        for metric, unit in (("speedup_2sm_over_1sm", "ratio"),
                             ("scaling_efficiency_percent", "percent")):
            values = [record["umma"]["scaling"][key][metric] for record in records]
            summary = summarize_metric(values, metric=metric)
            outside = any(value < 0.0 or value > 100.0 for value in values)
            note = ("value_outside_0_100_preserved_unclamped" if metric.endswith("percent")
                    and outside else "derived_within_campaign_value_never_pooled")
            rows.append(prefix + [metric, unit, evidence_class_for(metric)]
                        + stat_cells(summary, metric=metric, notes=note))
            entry["metrics"][metric] = summary_json(summary, metric=metric)
        rows.append(prefix + ["surprising_value_flag", NOT_APPLICABLE,
                              evidence_class_for("surprising_value_flag")]
                    + diagnostic_cells(list(flags), metric="surprising_value_flag",
                                       notes="preserved_source_diagnostic"))
        entry["source_diagnostics"] = {
            "surprising_value_flag": diagnostic_json(list(flags),
                                                     metric="surprising_value_flag")}
        for campaign_index, flag in enumerate(flags):
            if flag not in ("", "False", "false", NOT_APPLICABLE):
                warnings.append({
                    "campaign_id": records[campaign_index]["campaign_id"],
                    "campaign_position": campaign_index + 1,
                    "section": "experiment_2_scaling",
                    "n": n, "depth": depth, "surprising_value_flag": flag,
                })
        scaling.append(entry)

    saturation = []
    for method, n in P24_SATURATION_KEYS:
        key = (method, n)
        values = [record["umma"]["saturation"][key] for record in records]
        agreement = consensus(values, label=f"depth saturation {key}")
        note = (f"consensus={agreement['consensus']}" if agreement["stable_across_campaigns"]
                else "no_cross_campaign_consensus")
        rows.append([SCHEMA_VERSION, "saturation", method, str(n), NOT_APPLICABLE,
                     str(P24_CTA_GROUP[method]),
                     "earliest_tested_candidate_saturation_depth", "depth",
                     evidence_class_for("earliest_tested_candidate_saturation_depth")]
                    + stat_cells(None, metric="earliest_tested_candidate_saturation_depth",
                                 notes=note, raw_values=values))
        saturation.append({"method": method, "n": n, "campaign_values_depth": values,
                           "stable_across_campaigns": agreement["stable_across_campaigns"],
                           "consensus_depth": agreement["consensus"],
                           "note": agreement["note"]})

    selections = [record["umma"]["ceiling"]["selected_configuration"] for record in records]
    selection = consensus(selections, label="empirical per-SM ceiling selection")
    tflops_values = [record["umma"]["ceiling"]["estimated_tflops_per_sm"]
                     for record in records]
    clocks_ok = all(record["umma"]["profile_validation"]["sm_clock_ok_count"]
                    == P24_PROFILE_CASE_COUNT for record in records)
    ceiling_summary = None
    if selection["stable_across_campaigns"] and clocks_ok:
        ceiling_summary = summarize_metric(tflops_values, metric="estimated_tflops_per_sm")
        note = "selection_stable_and_all_clocks_valid"
    elif not selection["stable_across_campaigns"]:
        note = "selection_not_stable_across_campaigns"
    else:
        note = "sm_clock_conversion_not_trustworthy_in_every_campaign"
    rows.append([SCHEMA_VERSION, "ceiling", NOT_APPLICABLE, NOT_APPLICABLE, NOT_APPLICABLE,
                 NOT_APPLICABLE, "estimated_tflops_per_sm", "TFLOP/s/SM",
                 evidence_class_for("estimated_tflops_per_sm")]
                + stat_cells(ceiling_summary, metric="estimated_tflops_per_sm", notes=note,
                             raw_values=[format_decimal(value, decimals_for(
                                 "estimated_tflops_per_sm")) for value in tflops_values]))

    device = aggregate_device_equivalent(records)

    section = {
        "title": "Experiment 2 -- BF16 UMMA (fifth-generation Tensor Core) throughput",
        "configuration_count": len(P24_CONFIG_KEYS),
        "primary_metric_scope": (
            "median_flops_per_cycle and median_flops_per_cycle_per_sm are "
            "operation-and-cycle-derived throughputs: a validated operation count "
            "divided by the measured %clock64 cycle count. They are clock-independent, "
            "which does not make them directly measured"),
        "configurations": configurations,
        "scaling": scaling,
        "scaling_note": (
            "speedup and scaling efficiency are derived within-campaign quantities, "
            "summarized across the three campaigns; a value outside [0, 100] is "
            "preserved unclamped and keeps the closed unit's surprising-value "
            "diagnostic"),
        "depth_saturation": saturation,
        "depth_saturation_consensus_available": all(
            entry["stable_across_campaigns"] for entry in saturation),
        "empirical_per_sm_ceiling": {
            "campaign_selections": selections,
            "selection_stable_across_campaigns": selection["stable_across_campaigns"],
            "selected_configuration_consensus": selection["consensus"],
            "sm_clock_conversion_valid_in_every_campaign": clocks_ok,
            "estimated_tflops_per_sm": (
                summary_json(ceiling_summary, metric="estimated_tflops_per_sm")
                if ceiling_summary is not None else
                {"metric": "estimated_tflops_per_sm", "statistics": NOT_APPLICABLE,
                 "reason": note,
                 "campaign_values": [quantize(value, decimals_for("estimated_tflops_per_sm"))
                                     for value in tflops_values]}),
            "scope": (
                "a modeled clock conversion of a one-/two-SM microbenchmark result: the "
                "candidate is selected in clock-independent FLOP/cycle/SM space and only "
                "then multiplied by that same configuration's own profiled SM clock. It "
                "is never a theoretical architectural peak and never a measured "
                "whole-device throughput"),
            **evidence_json("estimated_tflops_per_sm"),
        },
        "device_equivalent_estimate": device,
        "profile_validation": [
            {"campaign_id": record["campaign_id"],
             "profiled_cases": record["umma"]["profile_validation"]["case_count"],
             "sm_clock_ok_count": record["umma"]["profile_validation"]["sm_clock_ok_count"],
             "sm_clock_statuses": record["umma"]["profile_validation"][
                 "sm_clock_statuses_distinct"]}
            for record in records],
        "source_warnings": warnings,
    }
    return rows, section


def aggregate_device_equivalent(records: list[dict]) -> dict:
    """A device-wide estimate is reported only when every final campaign
    independently contains a valid estimate built on a validated SM count and
    all three SM counts agree. Otherwise a structured `unavailable` result
    carries the exact per-campaign reason. The SM count is never taken from an
    external specification, hard-coded, inferred from another field, or used to
    convert the per-SM microbenchmark ceiling into a whole-GPU peak."""
    estimates = [record["umma"]["ceiling"]["device_equivalent_estimate"]
                 for record in records]
    per_campaign = []
    for record, estimate in zip(records, estimates):
        per_campaign.append({
            "campaign_id": record["campaign_id"],
            "available": bool(estimate.get("available")),
            "reason": (estimate.get("reason") if not estimate.get("available")
                       else "the closed unit resolved a validated SM count"),
            "multiprocessor_count": estimate.get("multiprocessor_count"),
        })
    if not all(entry["available"] for entry in per_campaign):
        return {
            "available": False,
            "reason": ("at least one final campaign contains no valid device-wide estimate, "
                       "so no whole-GPU throughput is reported"),
            "per_campaign": per_campaign,
        }
    counts = {entry["multiprocessor_count"] for entry in per_campaign}
    if len(counts) != 1 or not isinstance(next(iter(counts)), int) or next(iter(counts)) <= 0:
        return {
            "available": False,
            "reason": ("the three final campaigns do not agree on one validated, strictly "
                       "positive SM count"),
            "per_campaign": per_campaign,
        }
    values = []
    for estimate in estimates:
        value = estimate.get("estimated_device_equivalent_tflops")
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(float(value)) or float(value) <= 0.0:
            return {
                "available": False,
                "reason": "a campaign's device-equivalent estimate is not a finite positive value",
                "per_campaign": per_campaign,
            }
        values.append(float(value))
    summary = summarize_metric(values, metric="estimated_device_equivalent_tflops")
    return {
        "available": True,
        "multiprocessor_count": next(iter(counts)),
        "estimated_device_equivalent_tflops": summary_json(
            summary, metric="estimated_device_equivalent_tflops"),
        "status": "modeled estimate: per-SM ceiling multiplied by a validated SM count",
        "per_campaign": per_campaign,
    }


# ===========================================================================
# Experiment 3 -- CuTe DSL versus cuBLASLt.
# ===========================================================================


def aggregate_experiment_3(records: list[dict], *, p35) -> tuple[list[list[str]], dict]:
    rows: list[list[str]] = []
    shapes = []
    for shape_index in range(1, p35.EXPECTED_SHAPE_COUNT + 1):
        mnkl = p35.EXPECTED_SHAPES[shape_index - 1]
        shape_id = p35.EXPECTED_SHAPE_IDS[shape_index - 1]
        shape_entry = {"shape_index": shape_index, "shape_id": shape_id,
                       "m": mnkl[0], "n": mnkl[1], "k": mnkl[2], "l": mnkl[3],
                       "candidates": []}
        for candidate_index in range(1, p35.EXPECTED_CANDIDATE_COUNT + 1):
            key = (shape_index, candidate_index)
            first = records[0]["gemm"]["rows"][key]
            variant = first["variant"]
            method = first["method"]
            for record in records:
                row = record["gemm"]["rows"][key]
                if (row["variant"], row["method"], row["shape_id"]) != (
                        variant, method, shape_id):
                    raise P43Error(
                        f"{record['campaign_id']}: shape {shape_index} candidate "
                        f"{candidate_index} is not the same frozen candidate across "
                        f"campaigns")
            candidate_entry = {"candidate_index": candidate_index, "variant": variant,
                               "method": method, "metrics": {}}
            for metric, unit in (("kernel_time_ms", "ms"),
                                 ("tflops", "TFLOP/s"),
                                 ("throughput_ratio_vs_cublaslt", "ratio"),
                                 ("gap_to_cublaslt_pct", "percent")):
                values = [record["gemm"]["rows"][key][metric] for record in records]
                summary = summarize_metric(values, metric=metric)
                if metric == "gap_to_cublaslt_pct":
                    note = "signed_metric_cv_not_computed_negative_gaps_never_clamped"
                elif metric == "throughput_ratio_vs_cublaslt":
                    note = "derived_within_campaign_ratio_never_recomputed_from_aggregates"
                elif metric == "kernel_time_ms":
                    note = "measured_campaign_level_input"
                else:
                    note = "derived_within_campaign_value"
                rows.append([SCHEMA_VERSION, "candidate", str(shape_index), shape_id,
                             str(mnkl[0]), str(mnkl[1]), str(mnkl[2]), str(mnkl[3]),
                             str(candidate_index), variant, method, metric, unit,
                             evidence_class_for(metric)]
                            + stat_cells(summary, metric=metric, notes=note))
                candidate_entry["metrics"][metric] = summary_json(summary, metric=metric)
            shape_entry["candidates"].append(candidate_entry)

        best = [record["gemm"]["best_cutedsl_variant"][shape_index] for record in records]
        agreement = consensus(best, label=f"best CuTe DSL variant, shape {shape_index}")
        note = (f"stable_best={agreement['consensus']}" if agreement["stable_across_campaigns"]
                else "no_stable_best_cutedsl_variant_across_the_three_final_campaigns")
        rows.append([SCHEMA_VERSION, "best_cutedsl", str(shape_index), shape_id,
                     str(mnkl[0]), str(mnkl[1]), str(mnkl[2]), str(mnkl[3]),
                     NOT_APPLICABLE, NOT_APPLICABLE, "cutedsl", "best_cutedsl_variant",
                     "variant", evidence_class_for("best_cutedsl_variant")]
                    + stat_cells(None, metric="best_cutedsl_variant", notes=note,
                                 raw_values=best))
        shape_entry["best_cutedsl_variant"] = {
            "campaign_values": best,
            "stable_across_campaigns": agreement["stable_across_campaigns"],
            "stable_best_cutedsl_variant": agreement["consensus"],
            "note": (f"the same best CuTe DSL variant in all three final campaigns"
                     if agreement["stable_across_campaigns"] else
                     "no stable best CuTe DSL variant across the three final campaigns"),
        }
        shapes.append(shape_entry)

    cache_modes = consensus([record["gemm"]["cache_mode"] for record in records],
                            label="cache_mode")
    section = {
        "title": "Experiment 3 -- CuTe DSL BF16 GEMM versus cuBLASLt",
        "shape_count": p35.EXPECTED_SHAPE_COUNT,
        "candidate_count": p35.EXPECTED_CANDIDATE_COUNT,
        "candidate_order": list(p35.EXPECTED_CANDIDATE_ORDER),
        "row_count_per_campaign": p35.EXPECTED_ROW_COUNT,
        "source_row_kind": {"run_kind": "smoke", "publishable": "false",
                            "correctness": "PASS"},
        "cache_mode": cache_modes["consensus"],
        "primary_metric_scope": (
            "kernel_time_ms is the measured source observation (CUDA-event kernel time "
            "divided by the measured iteration count, after correctness passed). tflops, "
            "the cuBLASLt-relative ratio, the signed gap, and the best-variant selection "
            "are all derived within-campaign quantities; no GEMM kernel was profiled"),
        "shapes": shapes,
        "notes": [
            "beating cuBLASLt is not a success criterion",
            "a ratio above one and a negative gap mean the candidate reached the shorter "
            "measured kernel time; neither is clamped",
            "the cuBLASLt-relative ratio and gap are derived inside each campaign and only "
            "then summarized; they are never recomputed from aggregated means",
            "the best CuTe DSL variant per shape is a derived within-campaign selection, "
            "not a measured quantity",
        ],
    }
    return rows, section


# ===========================================================================
# The integrated interpretation.
# ===========================================================================


def build_interpretation(experiment_1: dict, experiment_2: dict, experiment_3: dict) -> dict:
    """The explicit scientific evidence taxonomy.

    Every reported quantity is placed in exactly one category, using the exact
    semantics of the closed upstream protocols rather than a convenient
    paraphrase. Nothing derived is described as directly measured; a
    campaign-level median is never presented as an individual raw observation;
    no roofline, architectural peak, GEMM bottleneck attribution, or causal
    conclusion is introduced.
    """
    stable_bests = {
        shape["shape_id"]: shape["best_cutedsl_variant"]["stable_best_cutedsl_variant"]
        for shape in experiment_3["shapes"]
    }
    coverage = experiment_1["ncu_coverage"]
    return {
        "research_question": RESEARCH_QUESTION,
        "taxonomy_definition": {
            EVIDENCE_MEASURED: (
                "a quantity an instrument recorded directly during the campaign, "
                "described as measured only where the closed upstream protocol supports "
                "that description"),
            EVIDENCE_WITHIN_CAMPAIGN: (
                "a deterministic quantity computed inside one campaign from measured "
                "inputs and validated constants; reproducible, but not itself measured"),
            EVIDENCE_CROSS_CAMPAIGN: (
                "a descriptive statistic P4.3 computed over exactly three campaign-level "
                "values; it describes agreement between campaigns, never a new "
                "measurement"),
            EVIDENCE_MODELED: (
                "a quantity produced by applying a model or a unit conversion to a "
                "derived estimate; its status as a model is always stated"),
            EVIDENCE_INTERPRETATION: (
                "a reading of the evidence, phrased as consistent-with rather than as a "
                "causal claim"),
            EVIDENCE_UNAVAILABLE: (
                "a question the collected evidence cannot answer; reported as such "
                "instead of being filled in"),
            EVIDENCE_DIAGNOSTIC: (
                "a trust signal the closed unit recorded about its own measurement; "
                "preserved verbatim and never converted into a result"),
        },
        "measured_source_observations": [
            "per-repetition CUDA-event kernel time of the LDGSTS and TMA streaming "
            "microbenchmark launches (P1.1/P1.2), which is the timing input of every "
            "effective-rate estimate below",
            "Nsight Compute's dram__bytes_read.sum for exactly "
            f"{coverage['profiled_cases']} of {coverage['total_configurations']} memory "
            "configurations, together with the validated useful_bytes of those same cases",
            "the raw %clock64 elapsed-cycle counts of the BF16 UMMA launches (P2.1/P2.2)",
            "the per-configuration Nsight Compute SM-clock readings of all 24 profiled "
            "UMMA configurations (P2.4)",
            "CUDA-event kernel time of the five frozen BF16 GEMM shapes for three CuTe DSL "
            "execution variants and one cuBLASLt baseline, hot-cache, recorded only after "
            "correctness passed (P3.5 section 7)",
        ],
        "within_campaign_derived_estimates": [
            "median_effective_gbps: a timing-derived effective transfer rate. Each "
            "repetition divides the benchmark's logical useful_bytes by its own measured "
            "kernel time, and the campaign reports the median of 30 such values. P1.1 and "
            "P1.2 label this effective copy bandwidth and state explicitly that it is not "
            "HBM/DRAM bandwidth",
            "tma_to_ldgsts_ratio: a derived within-campaign ratio of two campaign-level "
            "medians at one identical configuration",
            "dram_read_ratio and hbm_classification: derived from profiler evidence, not "
            "raw profiler counters -- dram__bytes_read.sum divided by validated "
            "useful_bytes, then classified against P1.4's frozen 0.90 rule",
            "median_flops_per_cycle and median_flops_per_cycle_per_sm: "
            "operation-and-cycle-derived throughputs, a validated 2*M*N*K*depth*iterations "
            "operation count divided by the measured elapsed cycles. They are "
            "clock-independent, which does not make them directly measured",
            "speedup_2sm_over_1sm and scaling_efficiency_percent: derived inside each "
            "campaign from two campaign-level medians",
            "GEMM tflops: the exact 2*M*N*K operation count divided by the measured kernel "
            "time",
            "throughput_ratio_vs_cublaslt and gap_to_cublaslt_pct: derived inside each "
            "campaign against that campaign's own cuBLASLt baseline row",
            "the earliest-tested candidate saturation selections and the best CuTe DSL "
            "variant per shape: derived within-campaign selections over the tested grid, "
            "never universal thresholds",
        ],
        "cross_campaign_descriptive_statistics": [
            f"every mean, median, sample standard deviation (n-1), coefficient of "
            f"variation where meaningful, minimum, and maximum in these artifacts is "
            f"computed by P4.3 over exactly {CAMPAIGN_COUNT} campaign-level values, one "
            f"per final campaign",
            "cross_campaign_cv_percent and cross_campaign_cv_review_flag describe "
            "agreement between campaigns only; they are never a within-campaign stability "
            "diagnostic and never replace one",
            "the cross-campaign consensus of a saturation candidate, a ceiling selection, "
            "or a best variant is a statement about agreement between the three campaigns, "
            "not a new measurement and not a majority vote",
        ],
        "modeled_estimates": [
            "estimated_tflops_per_sm: a modeled clock conversion of a microbenchmark "
            "result. The candidate is selected in clock-independent FLOP/cycle/SM space "
            "and only then multiplied by that same configuration's own profiled SM clock. "
            "It is a one-/two-SM empirical microbenchmark ceiling candidate, never an "
            "architectural peak",
            "estimated_device_equivalent_tflops, when it is available at all: the modeled "
            "per-SM estimate multiplied by a validated SM count, which is a whole-device "
            "extrapolation and never a measured whole-GPU throughput",
        ],
        "interpretations": [
            "the LDGSTS/TMA benchmark is a dedicated streaming HBM-to-SMEM data-movement "
            "microbenchmark. It does not directly measure the memory traffic a GEMM kernel "
            "generates, and its effective-rate estimates are consistent with, but not "
            "evidence of, GEMM-level memory behaviour",
            "where the TMA-to-LDGSTS ratio stays close to one across all three campaigns, "
            "the evidence is consistent with the two paths reaching a similar effective "
            "transfer rate at that configuration",
            "the distance between the best CuTe DSL variant and cuBLASLt per shape is a "
            "derived difference between measured kernel times; the collected evidence does "
            "not attribute it to any cause",
        ],
        "unavailable_from_the_collected_evidence": [
            f"actual HBM/DRAM traffic for the "
            f"{coverage['unprofiled_configuration_count']} memory configurations outside "
            f"the frozen {coverage['profiled_cases']}-case Nsight Compute plan: no "
            f"profiler evidence was collected for them, and the six profiled cases are "
            f"never extrapolated to them",
            "whether any specific GEMM shape is HBM-bound, Tensor-Core-bound, "
            "scheduler-bound, or limited by another implementation cost: P3.5 collected no "
            "Nsight Compute profile of a GEMM kernel, so no bottleneck attribution is made",
            "a numerical roofline, an architectural peak, or an arithmetic-intensity "
            "placement: the streaming microbenchmark and the GEMM measurements are not "
            "dimensionally comparable evidence of the same workload, and no compulsory-byte "
            "model was validated",
            "a cold-cache GEMM result: every GEMM measurement is hot-cache by construction",
            "a whole-device BF16 throughput figure whenever the modeled device-wide "
            "estimate is unavailable",
        ],
        "answer_summary": {
            "hbm_to_smem": (
                "derived: memory_paths.csv reports each campaign's median timing-derived "
                "effective transfer rate for both equivalent HBM-to-SMEM paths over the "
                "frozen grid, and the derived within-campaign TMA-to-LDGSTS ratios beside "
                "them. Profiler-derived DRAM traffic exists for six configurations only; "
                "for the other twelve, actual HBM traffic is unavailable. The saturation "
                "candidate is reported per group, and only as a cross-campaign consensus "
                "when all three campaigns agree, never as a universal HBM saturation "
                "threshold"),
            "tensor_core": (
                "derived: umma_throughput.csv reports the clock-independent, "
                "operation-and-cycle-derived FLOP/cycle and FLOP/cycle/SM values and each "
                "campaign's derived 1-SM/2-SM scaling. The per-SM ceiling candidate is a "
                "modeled clock conversion of a one-/two-SM microbenchmark and is "
                "summarized across campaigns only when all three select the same "
                "configuration"),
            "cutedsl_versus_cublaslt": (
                "measured input plus derived comparison: per shape and candidate, "
                "gemm_comparison.csv reports the campaign-level measured kernel time and "
                "the derived TFLOP/s, cuBLASLt-relative ratio, and signed gap, each "
                "summarized across the three campaigns. The stable best CuTe DSL variant "
                "per shape is " + json.dumps(stable_bests, sort_keys=True)
                + " (null means the three campaigns did not agree)"),
            "constraint_attribution": (
                "unavailable from the collected evidence: no GEMM-level profile exists, so "
                "the GEMM throughput is not attributed to the memory path, the Tensor Core "
                "ceiling, the scheduler, or any other single cost"),
        },
    }


def build_limitations(experiment_1: dict, experiment_2: dict, experiment_3: dict) -> list[str]:
    coverage = experiment_1["ncu_coverage"]
    return [
        f"the independent replicate is one complete final campaign; the cross-campaign "
        f"sample size is {CAMPAIGN_COUNT}, which is small and supports descriptive "
        f"statistics only",
        "no p-value, significance claim, or cross-campaign confidence interval is computed; "
        "the within-campaign confidence intervals the closed units recorded remain "
        "provenance and are never reinterpreted as cross-campaign intervals",
        "no observation and no campaign was removed; no outlier filter was applied; a "
        f"cross-campaign coefficient of variation above {CV_REVIEW_THRESHOLD_PERCENT:.1f}% "
        f"is a review diagnostic only. It never excludes a campaign, never changes a "
        f"result, and never replaces a closed unit's own within-campaign stability review",
        "within-campaign and cross-campaign variability are different quantities and are "
        "reported in separate, differently named fields; they may disagree, and a "
        "disagreement is reported rather than resolved",
        "a coefficient of variation is not computed for signed or zero-centred quantities "
        "such as gap_to_cublaslt_pct",
        "median_effective_gbps is a timing-derived effective transfer rate of a streaming "
        "microbenchmark -- the benchmark's logical useful_bytes divided by its measured "
        "kernel time -- and is explicitly not directly measured HBM/DRAM bandwidth",
        f"profiler-derived HBM/DRAM traffic exists for exactly {coverage['profiled_cases']} "
        f"of {coverage['total_configurations']} memory configurations; for the other "
        f"{coverage['unprofiled_configuration_count']}, actual HBM traffic is unavailable "
        f"from the collected evidence and the six profiled cases are never extrapolated to "
        f"them",
        "the dram_read_ratio and hbm_classification of those six cases are derived from "
        "profiler evidence, not raw profiler counters, and are kept separate from the "
        "timing-derived effective-rate metric",
        "the streaming memory microbenchmark does not directly measure the memory traffic a "
        "GEMM kernel generates",
        "flops_per_cycle and flops_per_cycle_per_sm are derived from validated operation "
        "counts and measured cycles; being clock-independent does not make them directly "
        "measured",
        "the BF16 UMMA ceiling is a modeled clock conversion of a one-/two-SM empirical "
        "microbenchmark result; it is not an architectural peak and not a measured "
        "whole-device throughput",
        "no SM count is imported from an external specification, hard-coded, or inferred; "
        "without validated agreeing SM-count evidence the modeled device-wide estimate "
        "stays structurally unavailable",
        f"every GEMM measurement is hot-cache ({experiment_3['cache_mode']}) and must not be "
        f"described as a cold-cache workload",
        "P3.5 collected no Nsight Compute profile of a GEMM kernel, so no GEMM bottleneck "
        "attribution, roofline placement, architectural peak, or arithmetic-intensity "
        "classification is made anywhere in these artifacts",
        "the source GEMM rows are run_kind=smoke evidence captured by the campaign; they "
        "carry publishable=false, and their kernel times are measured inputs to a "
        "comparison, not a validated publication-grade benchmark",
        "the sweep order inside each closed unit is fixed and non-randomized, a limitation "
        "the closed units already recorded",
        "the accepted pilot is excluded from every statistic here; it qualifies the "
        "orchestration path only",
        "these nine artifacts are a candidate bundle: they are bound together by "
        f"{MANIFEST_RELATIVE_PATH}, which is the authoritative provenance envelope. A "
        f"detached CSV or SVG is not a standalone provenance envelope and must be "
        f"distributed with the manifest",
        "no P4.3 result is publishable: the independent audit of this analysis layer, the "
        "production run against the three real final campaigns, the byte-for-byte "
        "verification, the independent review of the resulting outputs, and the external "
        "acceptance attestation are all still pending",
    ]


def collect_source_warnings(experiment_1: dict, experiment_2: dict) -> list[dict]:
    """Every non-empty terminal diagnostic the closed units recorded, in frozen
    campaign order. Nothing here is ever silently dropped: a warning that is not
    surfaced in the report would be a warning the reader never sees."""
    return list(experiment_1["source_warnings"]) + list(experiment_2["source_warnings"])


def collect_cross_campaign_reviews(sections: dict) -> list[dict]:
    """Every cross-campaign CV that crossed the review threshold. This is a
    diagnostic about agreement between campaigns; it never removes a campaign,
    never changes a value, and never stands in for a within-campaign flag."""
    reviews: list[dict] = []

    def scan(entry: object, path: str) -> None:
        if isinstance(entry, dict):
            if entry.get("cross_campaign_cv_review_flag") == CV_FLAG_REVIEW:
                reviews.append({
                    "location": path,
                    "metric": entry.get("metric"),
                    "cross_campaign_cv_percent": entry.get("cross_campaign_cv_percent"),
                    "campaign_values": entry.get("campaign_values"),
                    "effect": ("review diagnostic only: no campaign was excluded and no "
                               "value was changed"),
                })
            for key, value in entry.items():
                scan(value, f"{path}.{key}" if path else str(key))
        elif isinstance(entry, list):
            for index, value in enumerate(entry):
                scan(value, f"{path}[{index}]")

    for name in ("experiment_1", "experiment_2", "experiment_3"):
        scan(sections[name], name)
    return reviews


# ===========================================================================
# Rendering: CSV, JSON, Markdown, SVG.
# ===========================================================================


def render_csv(fields: tuple[str, ...], rows: list[list[str]]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(list(fields))
    for row in rows:
        if len(row) != len(fields):
            raise P43Error(f"a CSV row has {len(row)} cell(s), expected {len(fields)}")
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def build_summary_document(records: list[dict], sections: dict, interpretation: dict,
                           limitations: list[str], sources: list[dict],
                           provenance: dict, warnings: list[dict],
                           reviews: list[dict]) -> dict:
    reference = records[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "unit": UNIT,
        "analysis_kind": "cross_campaign_integrated_analysis",
        "research_question": RESEARCH_QUESTION,
        "population": {
            "campaign_count": CAMPAIGN_COUNT,
            "final_campaign_ids": list(FINAL_CAMPAIGN_IDS),
            "campaign_value_column_map": campaign_value_column_map(),
            "pilot_campaign_id_excluded": PILOT_CAMPAIGN_ID,
            "pilot_role": PILOT_ROLE,
            "final_execution_commit": reference["git_commit"],
            "gpu": dict(reference["gpu"]),
            "comparable_provenance": {key: reference["provenance"][key]
                                      for key in sorted(reference["provenance"])},
        },
        "analysis_provenance": dict(provenance),
        "bundle_contract": {
            "authoritative_envelope": MANIFEST_RELATIVE_PATH,
            "artifact_inventory": list(ARTIFACT_RELATIVE_PATHS),
            "note": (
                f"{MANIFEST_RELATIVE_PATH} is the authoritative binding for this bundle: "
                f"it records the population, the provenance, and a SHA-256 for each of "
                f"the other eight artifacts. An individual CSV or SVG is a deterministic "
                f"data or visual artifact, not a standalone provenance envelope, and must "
                f"be distributed together with the manifest. This document and report.md "
                f"carry the scientific context needed to interpret the bundle"),
        },
        "evidence_taxonomy": {
            "classes": list(EVIDENCE_CLASSES),
            "definitions": interpretation["taxonomy_definition"],
            "metric_classification": {
                metric: {"evidence_class": entry[0], "basis": entry[1]}
                for metric, entry in sorted(METRIC_EVIDENCE.items())},
            "cross_campaign_statistics": (
                "every mean, median, stdev_sample, cross_campaign_cv_percent, minimum, and "
                f"maximum in this bundle is a {EVIDENCE_CROSS_CAMPAIGN} over exactly "
                f"{CAMPAIGN_COUNT} campaign-level values, whatever the class of the "
                f"underlying quantity"),
        },
        "statistical_policy": {
            "independent_replicate": "one complete final campaign",
            "campaign_count": CAMPAIGN_COUNT,
            "statistics": ["mean", "median", "sample_standard_deviation_n_minus_1",
                           "coefficient_of_variation", "minimum", "maximum"],
            "cross_campaign_cv_review_threshold_percent": CV_REVIEW_THRESHOLD_PERCENT,
            "cross_campaign_cv_scope": "strictly positive performance metrics only",
            "cross_campaign_cv_effect": (
                "a review diagnostic about agreement between campaigns; it never excludes "
                "a campaign, never changes a result, and never replaces a closed unit's "
                "own within-campaign stability review"),
            "within_campaign_diagnostics": (
                "each closed unit's own sample_count, cv_percent, stability review, "
                "surprising-value flag, and Nsight Compute diagnostic flags are preserved "
                "per campaign, in the frozen campaign order, under their own "
                "within_campaign_* and source-diagnostic names"),
            "pooling_of_internal_repetitions": "forbidden",
            "ratio_policy": ("ratios are computed inside each campaign and only then "
                             "summarized; a ratio is never formed from two aggregates"),
            "outlier_policy": "no observation and no campaign is ever removed",
            "significance_testing": "none",
            "confidence_intervals": ("within-campaign intervals are preserved as provenance "
                                     "only; no cross-campaign interval is bootstrapped from "
                                     f"{CAMPAIGN_COUNT} campaigns"),
            "precision": ("full precision is retained during computation; decimals are "
                          "applied only at serialization"),
        },
        "experiment_1_memory_paths": sections["experiment_1"],
        "experiment_2_umma_throughput": sections["experiment_2"],
        "experiment_3_gemm_comparison": sections["experiment_3"],
        "integrated_interpretation": interpretation,
        "source_diagnostic_warnings": warnings,
        "cross_campaign_review_conditions": reviews,
        "limitations": limitations,
        "sources": sources,
        "publishable": PUBLISHABLE,
        "publication_state": PUBLICATION_STATE,
        "publication_status": PUBLICATION_STATUS,
    }


def _markdown_table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def render_report(document: dict) -> bytes:
    population = document["population"]
    exp1 = document["experiment_1_memory_paths"]
    exp2 = document["experiment_2_umma_throughput"]
    exp3 = document["experiment_3_gemm_comparison"]
    interpretation = document["integrated_interpretation"]

    lines: list[str] = []
    lines.append("# Phase 4 integrated analysis (P4.3)")
    lines.append("")
    lines.append(f"Schema version: `{document['schema_version']}`. "
                 f"Publication status: **{document['publication_status']}**.")
    lines.append("")
    lines.append(f"This report is one of {len(ARTIFACT_RELATIVE_PATHS)} artifacts in a "
                 f"candidate bundle. `{document['bundle_contract']['authoritative_envelope']}` "
                 f"is the authoritative provenance envelope and binds every other artifact "
                 f"by SHA-256; a detached CSV or SVG is not a standalone provenance "
                 f"envelope and must be distributed with the manifest.")
    lines.append("")
    lines.append("## 1. Population and provenance")
    lines.append("")
    lines.append(f"* Independent replicate: **one complete final campaign** "
                 f"(`campaign_count = {population['campaign_count']}`).")
    for campaign_id in population["final_campaign_ids"]:
        lines.append(f"* Final campaign `{campaign_id}`.")
    lines.append(f"* Accepted pilot `{population['pilot_campaign_id_excluded']}` is "
                 f"{population['pilot_role']}.")
    lines.append(f"* Final execution commit `{population['final_execution_commit']}` "
                 f"(the commit the three campaigns *ran* from).")
    provenance = document["analysis_provenance"]
    lines.append(f"* P4.3 analysis-code commit "
                 f"`{provenance['analysis_code_commit']}` (the commit whose code produced "
                 f"this bundle), worktree clean: "
                 f"`{str(provenance['worktree_clean']).lower()}`.")
    for column, campaign_id in population["campaign_value_column_map"].items():
        lines.append(f"* Column `{column}` is campaign `{campaign_id}`.")
    lines.append(f"* GPU `{population['gpu']['name']}` "
                 f"(`{population['gpu']['uuid']}`, compute capability "
                 f"`{population['gpu']['compute_capability']}`, driver "
                 f"`{population['gpu']['driver_version']}`).")
    lines.append("")
    lines.append("## 2. Frozen statistical policy")
    lines.append("")
    for key in sorted(document["statistical_policy"]):
        value = document["statistical_policy"][key]
        rendered = ", ".join(str(item) for item in value) if isinstance(value, list) else value
        lines.append(f"* `{key}`: {rendered}")
    lines.append("")

    lines.append("## 3. Experiment 1 — LDGSTS versus TMA")
    lines.append("")
    lines.append(exp1["primary_metric_scope"] + ".")
    lines.append("")
    lines.append("Campaign-level median timing-derived effective transfer rate "
                 "(`median_effective_gbps`, a "
                 f"`{evidence_class_for('median_effective_gbps')}`), summarized across the "
                 "three final campaigns (GB/s). This is **not** directly measured HBM/DRAM "
                 "bandwidth.")
    lines.append("")
    rows = []
    for entry in exp1["configurations"]:
        rows.append([entry["method"], str(entry["stages"]), str(entry["bytes_in_flight_kib"]),
                     format_decimal(entry["mean"], 6), format_decimal(entry["median"], 6),
                     format_decimal(entry["stdev_sample"], 6),
                     format_decimal(entry["cross_campaign_cv_percent"], DECIMALS_CV),
                     format_decimal(entry["minimum"], 6), format_decimal(entry["maximum"], 6),
                     entry["cross_campaign_cv_review_flag"]])
    lines.extend(_markdown_table(
        ["method", "stages", "bif_kib", "mean", "median", "stdev", "cross-campaign cv_%",
         "min", "max", "cross-campaign flag"], rows))
    lines.append("")
    lines.append("TMA-to-LDGSTS ratio per identical `(stages, bytes_in_flight_kib)` pair "
                 f"(a `{evidence_class_for('tma_to_ldgsts_ratio')}`). Above one means TMA "
                 "reached the higher effective transfer rate; below one means LDGSTS did. "
                 "This is a derived within-campaign ratio of two campaign-level medians, "
                 "not a directly measured quantity, not a winner, and not a significance "
                 "claim.")
    lines.append("")
    rows = []
    for entry in exp1["pair_ratios"]:
        rows.append([str(entry["stages"]), str(entry["bytes_in_flight_kib"]),
                     format_decimal(entry["mean"], 9), format_decimal(entry["median"], 9),
                     format_decimal(entry["stdev_sample"], 9),
                     format_decimal(entry["minimum"], 9), format_decimal(entry["maximum"], 9),
                     str(entry["interpretation_consensus"])])
    lines.extend(_markdown_table(
        ["stages", "bif_kib", "mean", "median", "stdev", "min", "max",
         "campaign interpretation"], rows))
    lines.append("")
    lines.append("Earliest tested candidate saturation point per group:")
    lines.append("")
    for entry in exp1["saturation_candidates"]:
        if entry["stable_across_campaigns"]:
            lines.append(f"* `{entry['method']}` stages `{entry['stages']}`: all three "
                         f"campaigns report `{entry['consensus_kib']}` KiB.")
        else:
            lines.append(f"* `{entry['method']}` stages `{entry['stages']}`: **no single "
                         f"cross-campaign consensus candidate exists**; the three campaigns "
                         f"report {entry['campaign_values_kib']} KiB.")
    lines.append("")
    coverage = exp1["ncu_coverage"]
    lines.append(f"Nsight Compute HBM/DRAM traffic validation covers exactly "
                 f"{coverage['profiled_cases']} of "
                 f"{coverage['total_configurations']} configurations. "
                 f"{coverage['limitation'].capitalize()}. "
                 f"For the remaining "
                 f"{coverage['unprofiled_configuration_count']} configurations, **actual "
                 f"HBM/DRAM traffic is unavailable from the collected evidence**; only the "
                 f"timing-derived effective transfer rate above exists for them. "
                 f"{coverage['separation'].capitalize()}.")
    lines.append("")
    lines.append(f"`dram_read_ratio` is a "
                 f"`{evidence_class_for('dram_read_ratio')}` derived from profiler "
                 f"evidence (`dram__bytes_read.sum / useful_bytes`), not a raw profiler "
                 f"counter, and `hbm_classification` is P1.4's frozen classification of "
                 f"it.")
    lines.append("")
    rows = []
    for entry in exp1["ncu_validation"]:
        rows.append([str(entry["index"]), entry["method"], str(entry["stages"]),
                     str(entry["bytes_in_flight_kib"]), format_decimal(entry["mean"], 9),
                     format_decimal(entry["minimum"], 9), format_decimal(entry["maximum"], 9),
                     str(entry["hbm_classification_consensus"]),
                     " / ".join(flag or "--"
                                for flag in entry["campaign_diagnostic_flags"])])
    lines.extend(_markdown_table(
        ["case", "method", "stages", "bif_kib", "dram_read_ratio mean", "min", "max",
         "classification", "diagnostic flags (c1 / c2 / c3)"], rows))
    lines.append("")

    lines.append("## 4. Experiment 2 — BF16 UMMA throughput")
    lines.append("")
    lines.append(exp2["primary_metric_scope"] + ".")
    lines.append("")
    lines.append("Clock-independent, operation-and-cycle-derived campaign-level medians "
                 f"(a `{evidence_class_for('median_flops_per_cycle')}`), summarized across "
                 "the three final campaigns.")
    lines.append("")
    rows = []
    for entry in exp2["configurations"]:
        per_sm = entry["metrics"]["median_flops_per_cycle_per_sm"]
        total = entry["metrics"]["median_flops_per_cycle"]
        rows.append([entry["method"], str(entry["n"]), str(entry["depth"]),
                     str(entry["cta_group"]), format_decimal(total["mean"], 6),
                     format_decimal(per_sm["mean"], 6),
                     format_decimal(per_sm["cross_campaign_cv_percent"], DECIMALS_CV),
                     per_sm["cross_campaign_cv_review_flag"]])
    lines.extend(_markdown_table(
        ["method", "N", "depth", "cta_group", "FLOP/cycle mean", "FLOP/cycle/SM mean",
         "cross-campaign cv_%", "cross-campaign flag"], rows))
    lines.append("")
    lines.append("1-SM/2-SM comparison. Speedup and scaling efficiency are derived "
                 f"within-campaign quantities (a "
                 f"`{evidence_class_for('speedup_2sm_over_1sm')}`) summarized across the "
                 "three campaigns; values outside `[0, 100]` are preserved unclamped and "
                 "keep the closed unit's surprising-value diagnostic.")
    lines.append("")
    rows = []
    for entry in exp2["scaling"]:
        speedup = entry["metrics"]["speedup_2sm_over_1sm"]
        efficiency = entry["metrics"]["scaling_efficiency_percent"]
        rows.append([str(entry["n"]), str(entry["depth"]),
                     format_decimal(speedup["mean"], 9),
                     format_decimal(speedup["minimum"], 9),
                     format_decimal(speedup["maximum"], 9),
                     format_decimal(efficiency["mean"], 6),
                     format_decimal(efficiency["minimum"], 6),
                     format_decimal(efficiency["maximum"], 6),
                     ",".join(entry["campaign_surprising_value_flags"])])
    lines.extend(_markdown_table(
        ["N", "depth", "speedup mean", "min", "max", "efficiency % mean", "min", "max",
         "surprising flags"], rows))
    lines.append("")
    lines.append("Depth saturation candidate per group:")
    lines.append("")
    for entry in exp2["depth_saturation"]:
        if entry["stable_across_campaigns"]:
            lines.append(f"* `{entry['method']}` N `{entry['n']}`: all three campaigns "
                         f"select depth `{entry['consensus_depth']}`.")
        else:
            lines.append(f"* `{entry['method']}` N `{entry['n']}`: **the selection is not "
                         f"stable across campaigns**; the three campaigns select "
                         f"{entry['campaign_values_depth']}.")
    lines.append("")
    ceiling = exp2["empirical_per_sm_ceiling"]
    lines.append("Empirical per-SM BF16 Tensor Core ceiling candidate "
                 f"({ceiling['scope']}):")
    lines.append("")
    if ceiling["selection_stable_across_campaigns"]:
        selected = ceiling["selected_configuration_consensus"]
        lines.append(f"* All three campaigns select `{selected['method']}` "
                     f"N `{selected['n']}` depth `{selected['depth']}` "
                     f"(`{selected['case_name']}`).")
    else:
        lines.append("* **The selection is not stable across the three final campaigns.** "
                     "Every campaign's own selection is preserved in "
                     "`integrated_summary.json`.")
    tflops = ceiling["estimated_tflops_per_sm"]
    if tflops.get("statistics") == NOT_APPLICABLE:
        lines.append(f"* No cross-campaign TFLOP/s/SM statistic is reported: "
                     f"`{tflops['reason']}`. The three campaign values are "
                     f"{tflops['campaign_values']}.")
    else:
        lines.append(f"* Modeled TFLOP/s/SM (a "
                 f"`{evidence_class_for('estimated_tflops_per_sm')}`) across the three "
                 f"campaigns: mean "
                     f"`{format_decimal(tflops['mean'], 9)}`, median "
                     f"`{format_decimal(tflops['median'], 9)}`, sample stdev "
                     f"`{format_decimal(tflops['stdev_sample'], 9)}`, min "
                     f"`{format_decimal(tflops['minimum'], 9)}`, max "
                     f"`{format_decimal(tflops['maximum'], 9)}`.")
    device = exp2["device_equivalent_estimate"]
    if device["available"]:
        lines.append(f"* Device-equivalent estimate (a **modeled** quantity): the per-SM "
                     f"ceiling multiplied by the validated SM count "
                     f"`{device['multiprocessor_count']}`, identical in all three campaigns.")
    else:
        lines.append(f"* Device-wide estimate: **unavailable**. Reason: {device['reason']}. "
                     f"No SM count is imported from an external specification, hard-coded, "
                     f"or inferred, so no whole-GPU peak is reported.")
    lines.append("")

    lines.append("## 5. Experiment 3 — CuTe DSL versus cuBLASLt")
    lines.append("")
    lines.append(exp3["primary_metric_scope"] + ".")
    lines.append("")
    lines.append(f"Five frozen shapes x {exp3['candidate_count']} frozen candidates, "
                 f"{exp3['row_count_per_campaign']} source rows per campaign, cache mode "
                 f"`{exp3['cache_mode']}`. Source rows carry "
                 f"`run_kind={exp3['source_row_kind']['run_kind']}` and "
                 f"`publishable={exp3['source_row_kind']['publishable']}`. Beating cuBLASLt "
                 f"is not a success criterion.")
    lines.append("")
    for shape in exp3["shapes"]:
        lines.append(f"### Shape `{shape['shape_id']}`")
        lines.append("")
        rows = []
        for candidate in shape["candidates"]:
            time_ms = candidate["metrics"]["kernel_time_ms"]
            tflops = candidate["metrics"]["tflops"]
            ratio = candidate["metrics"]["throughput_ratio_vs_cublaslt"]
            gap = candidate["metrics"]["gap_to_cublaslt_pct"]
            rows.append([candidate["variant"], candidate["method"],
                         format_decimal(time_ms["mean"], 6),
                         format_decimal(tflops["mean"], 6),
                         format_decimal(tflops["stdev_sample"], 6),
                         format_decimal(ratio["mean"], 9),
                         format_decimal(gap["mean"], 6),
                         format_decimal(gap["minimum"], 6),
                         format_decimal(gap["maximum"], 6)])
        lines.extend(_markdown_table(
            ["candidate", "method", "kernel_time_ms mean", "TFLOP/s mean", "TFLOP/s stdev",
             "ratio vs cuBLASLt mean", "gap % mean", "gap % min", "gap % max"], rows))
        lines.append("")
        best = shape["best_cutedsl_variant"]
        if best["stable_across_campaigns"]:
            lines.append(f"Stable best CuTe DSL variant: "
                         f"`{best['stable_best_cutedsl_variant']}`.")
        else:
            lines.append(f"**No stable best CuTe DSL variant across the three final "
                         f"campaigns**; the campaigns reported {best['campaign_values']}.")
        lines.append("")

    lines.append("## 6. Preserved source diagnostics and review conditions")
    lines.append("")
    lines.append("Every within-campaign diagnostic the closed units recorded is preserved "
                 "per campaign, in the frozen campaign order, in the CSV tables and in "
                 "`integrated_summary.json`. Nothing below excluded a campaign, removed an "
                 "observation, or changed a value.")
    lines.append("")
    warnings = document["source_diagnostic_warnings"]
    lines.append(f"### Source diagnostic warnings ({len(warnings)})")
    lines.append("")
    if not warnings:
        lines.append("* None: no campaign recorded a non-empty Nsight Compute diagnostic "
                     "flag, a within-campaign stability review, or a surprising-value flag "
                     "in any configuration.")
    else:
        for entry in warnings:
            detail = ", ".join(f"{key}={entry[key]}" for key in sorted(entry)
                               if key not in ("campaign_id", "campaign_position"))
            lines.append(f"* campaign `{entry['campaign_id']}` "
                         f"(campaign_{entry['campaign_position']}_value): {detail}")
    lines.append("")
    reviews = document["cross_campaign_review_conditions"]
    lines.append(f"### Cross-campaign variability review conditions ({len(reviews)})")
    lines.append("")
    lines.append(f"A cross-campaign coefficient of variation above "
                 f"{CV_REVIEW_THRESHOLD_PERCENT:.1f}% is a **review diagnostic only**. It "
                 f"is a different quantity from a closed unit's own within-campaign "
                 f"stability review, it never replaces one, and the two may disagree.")
    lines.append("")
    if not reviews:
        lines.append("* None: no reported quantity exceeded the cross-campaign review "
                     "threshold.")
    else:
        for entry in reviews:
            lines.append(f"* `{entry['location']}` metric `{entry['metric']}`: "
                         f"cross_campaign_cv_percent="
                         f"{format_decimal(entry['cross_campaign_cv_percent'], DECIMALS_CV)}, "
                         f"campaign values {entry['campaign_values']} "
                         f"({entry['effect']}).")
    lines.append("")

    lines.append("## 7. Integrated interpretation")
    lines.append("")
    lines.append(f"> {interpretation['research_question']}")
    lines.append("")
    lines.append("Each quantity below is placed in exactly one evidence class. A derived "
                 "or modeled quantity is never described as directly measured, and a "
                 "campaign-level median is never presented as an individual raw "
                 "observation.")
    lines.append("")
    for heading, key in (
            ("Measured source observations", "measured_source_observations"),
            ("Within-campaign derived estimates", "within_campaign_derived_estimates"),
            ("Cross-campaign descriptive statistics",
             "cross_campaign_descriptive_statistics"),
            ("Modeled estimates", "modeled_estimates"),
            ("Interpretations", "interpretations"),
            ("Unavailable from the collected evidence",
             "unavailable_from_the_collected_evidence")):
        lines.append(f"### {heading}")
        lines.append("")
        for item in interpretation[key]:
            lines.append(f"* {item}")
        lines.append("")
    lines.append("### Answer")
    lines.append("")
    for key in ("hbm_to_smem", "tensor_core", "cutedsl_versus_cublaslt",
                "constraint_attribution"):
        lines.append(f"* **{key}** — {interpretation['answer_summary'][key]}")
    lines.append("")

    lines.append("## 8. Limitations")
    lines.append("")
    for item in document["limitations"]:
        lines.append(f"* {item}")
    lines.append("")
    lines.append("## 9. Candidate status and the acceptance workflow")
    lines.append("")
    lines.append("These artifacts are a **candidate bundle**, not an accepted result:")
    lines.append("")
    lines.append("```text")
    lines.append(f"publishable        = {str(PUBLISHABLE).lower()}")
    lines.append(f"publication_state  = {PUBLICATION_STATE}")
    lines.append(f"analysis_code_commit = "
                 f"{document['analysis_provenance']['analysis_code_commit']}")
    lines.append("```")
    lines.append("")
    lines.append("The remaining lifecycle is, in this exact order:")
    lines.append("")
    for index, step in enumerate(ACCEPTANCE_LIFECYCLE, start=1):
        lines.append(f"{index}. {step}")
    lines.append("")
    lines.append(f"No step after the production of this candidate has been performed. "
                 f"Acceptance is an external attestation at `{ACCEPTANCE_RELATIVE_PATH}` "
                 f"that binds this bundle's `{MANIFEST_RELATIVE_PATH}` hash; it is never "
                 f"written by the analyzer, and no candidate artifact is ever promoted, "
                 f"rewritten, or deleted to record it.")
    lines.append("")
    lines.append("## 10. Status")
    lines.append("")
    lines.append("```text")
    lines.append("P4.1 | Orchestrator                              | YES | YES | YES")
    lines.append("P4.2 | Pilot plus three final campaigns          | YES | YES | YES")
    lines.append("P4.3 | Integrated analysis, documentation, audit | YES | NO  | NO")
    lines.append("```")
    lines.append("")
    lines.append(f"`publishable = {str(PUBLISHABLE).lower()}`. {PUBLICATION_STATUS}.")
    lines.append("")
    return ("\n".join(lines)).encode("utf-8")


# --- Deterministic SVG ------------------------------------------------------

_SVG_WIDTH = 1080
_SVG_HEIGHT = 400
_SVG_COLORS = ("#1f4e79", "#c05621", "#2f855a", "#6b46c1")


def _xml_escape(text: object) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;"))


def _svg_open(title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_SVG_WIDTH} {_SVG_HEIGHT}" '
        f'width="{_SVG_WIDTH}" height="{_SVG_HEIGHT}" font-family="monospace" '
        f'font-size="11">',
        f"<title>{_xml_escape(title)}</title>",
        f"<metadata>schema_version={_xml_escape(SCHEMA_VERSION)}; unit={_xml_escape(UNIT)}; "
        f"{_xml_escape(PUBLICATION_STATUS)}; this figure is a deterministic visual artifact "
        f"of the {MANIFEST_RELATIVE_PATH} bundle and is not a standalone provenance "
        f"envelope</metadata>",
        f'<rect x="0" y="0" width="{_SVG_WIDTH}" height="{_SVG_HEIGHT}" fill="#ffffff" '
        f'stroke="none"/>',
    ]


def _svg_panel(*, x0: float, y0: float, x1: float, y1: float, title: str,
               x_labels: list[str], y_label: str, series: list[dict],
               y_min: float, y_max: float) -> list[str]:
    out: list[str] = []
    if y_max <= y_min:
        y_max = y_min + 1.0
    span = y_max - y_min
    lo = y_min - 0.08 * span
    hi = y_max + 0.08 * span

    def scale_y(value: float) -> float:
        return y1 - (value - lo) / (hi - lo) * (y1 - y0)

    out.append(f'<text x="{(x0 + x1) / 2:.1f}" y="{y0 - 12:.1f}" text-anchor="middle" '
               f'font-weight="bold">{_xml_escape(title)}</text>')
    out.append(f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{x1 - x0:.1f}" '
               f'height="{y1 - y0:.1f}" fill="none" stroke="#333333" stroke-width="1"/>')
    for tick in range(5):
        value = lo + (hi - lo) * tick / 4.0
        y_px = scale_y(value)
        out.append(f'<line x1="{x0:.1f}" y1="{y_px:.1f}" x2="{x1:.1f}" y2="{y_px:.1f}" '
                   f'stroke="#e0e0e0" stroke-width="1"/>')
        out.append(f'<text x="{x0 - 5:.1f}" y="{y_px + 3:.1f}" text-anchor="end">'
                   f'{_xml_escape(f"{value:.4g}")}</text>')
    out.append(f'<text x="{x0 - 52:.1f}" y="{(y0 + y1) / 2:.1f}" text-anchor="middle" '
               f'transform="rotate(-90 {x0 - 52:.1f} {(y0 + y1) / 2:.1f})">'
               f'{_xml_escape(y_label)}</text>')

    count = max(len(x_labels), 1)
    positions = [x0 + (x1 - x0) * (index + 0.5) / count for index in range(count)]
    for index, label in enumerate(x_labels):
        out.append(f'<text x="{positions[index]:.1f}" y="{y1 + 16:.1f}" '
                   f'text-anchor="middle">{_xml_escape(label)}</text>')

    for order, entry in enumerate(series):
        color = _SVG_COLORS[order % len(_SVG_COLORS)]
        points = []
        for index, value in enumerate(entry["values"]):
            if value is None:
                continue
            px = positions[index]
            py = scale_y(value)
            points.append((px, py))
            low = entry["minimums"][index]
            high = entry["maximums"][index]
            if low is not None and high is not None:
                out.append(f'<line x1="{px:.1f}" y1="{scale_y(low):.1f}" x2="{px:.1f}" '
                           f'y2="{scale_y(high):.1f}" stroke="{color}" stroke-width="1.5"/>')
        if len(points) >= 2:
            path = " ".join(f"{px:.1f},{py:.1f}" for px, py in points)
            out.append(f'<polyline points="{path}" fill="none" stroke="{color}" '
                       f'stroke-width="2"/>')
        for px, py in points:
            out.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="4" fill="{color}" '
                       f'stroke="#ffffff" stroke-width="1"/>')
        out.append(f'<rect x="{x0 + 6:.1f}" y="{y0 + 6 + order * 14:.1f}" width="10" '
                   f'height="10" fill="{color}"/>')
        out.append(f'<text x="{x0 + 20:.1f}" y="{y0 + 15 + order * 14:.1f}">'
                   f'{_xml_escape(entry["label"])}</text>')
    return out


def _svg_close(footer: str) -> list[str]:
    return [f'<text x="12" y="{_SVG_HEIGHT - 8}" font-size="10" fill="#666666">'
            f'{_xml_escape(footer)}</text>', "</svg>"]


def _panel_geometry(count: int) -> list[tuple[float, float, float, float]]:
    panels = []
    usable = _SVG_WIDTH - 70
    width = usable / count
    for index in range(count):
        x0 = 62 + index * width
        panels.append((x0, 46.0, x0 + width - 22, float(_SVG_HEIGHT - 60)))
    return panels


def render_memory_svg(section: dict) -> bytes:
    lookup = {(entry["method"], entry["stages"], entry["bytes_in_flight_kib"]): entry
              for entry in section["configurations"]}
    values = [entry["mean"] for entry in section["configurations"]]
    minimums = [entry["minimum"] for entry in section["configurations"]]
    maximums = [entry["maximum"] for entry in section["configurations"]]
    out = _svg_open("Cross-campaign mean of the campaign-level median timing-derived "
                    "effective transfer rate")
    for index, (x0, y0, x1, y1) in enumerate(_panel_geometry(len(P14_STAGES))):
        stages = P14_STAGES[index]
        series = []
        for method in P14_METHODS:
            series.append({
                "label": method,
                "values": [lookup[(method, stages, bif)]["mean"] for bif in P14_BIF_KIB],
                "minimums": [lookup[(method, stages, bif)]["minimum"] for bif in P14_BIF_KIB],
                "maximums": [lookup[(method, stages, bif)]["maximum"] for bif in P14_BIF_KIB],
            })
        out.extend(_svg_panel(
            x0=x0, y0=y0, x1=x1, y1=y1, title=f"stages={stages}",
            x_labels=[f"{bif} KiB" for bif in P14_BIF_KIB],
            y_label="GB/s", series=series,
            y_min=min(minimums + values), y_max=max(maximums + values)))
    out.extend(_svg_close(
        f"n={CAMPAIGN_COUNT} final campaigns; {WHISKER_CAPTION}. Derived effective "
        f"transfer rate of a streaming microbenchmark (logical useful_bytes / measured "
        f"kernel time), NOT measured HBM/DRAM bandwidth and not GEMM traffic; not an HBM "
        f"saturation threshold."))
    return ("\n".join(out) + "\n").encode("utf-8")


def render_umma_svg(section: dict) -> bytes:
    lookup = {(entry["method"], entry["n"], entry["depth"]): entry
              for entry in section["configurations"]}
    metric = "median_flops_per_cycle_per_sm"
    values = [entry["metrics"][metric]["mean"] for entry in section["configurations"]]
    minimums = [entry["metrics"][metric]["minimum"] for entry in section["configurations"]]
    maximums = [entry["metrics"][metric]["maximum"] for entry in section["configurations"]]
    out = _svg_open("Cross-campaign mean of the campaign-level median "
                    "operation-and-cycle-derived FLOP/cycle/SM")
    for index, (x0, y0, x1, y1) in enumerate(_panel_geometry(len(P24_N_VALUES))):
        n_value = P24_N_VALUES[index]
        series = []
        for method in P24_METHODS:
            series.append({
                "label": method,
                "values": [lookup[(method, n_value, depth)]["metrics"][metric]["mean"]
                           for depth in P24_DEPTH_VALUES],
                "minimums": [lookup[(method, n_value, depth)]["metrics"][metric]["minimum"]
                             for depth in P24_DEPTH_VALUES],
                "maximums": [lookup[(method, n_value, depth)]["metrics"][metric]["maximum"]
                             for depth in P24_DEPTH_VALUES],
            })
        out.extend(_svg_panel(
            x0=x0, y0=y0, x1=x1, y1=y1, title=f"N={n_value}",
            x_labels=[f"d{depth}" for depth in P24_DEPTH_VALUES],
            y_label="FLOP/cycle/SM", series=series,
            y_min=min(minimums + values), y_max=max(maximums + values)))
    out.extend(_svg_close(
        f"n={CAMPAIGN_COUNT} final campaigns; {WHISKER_CAPTION}. Derived from validated "
        f"operation counts and measured cycles; clock-independent but not directly "
        f"measured. A one-/two-SM microbenchmark, never an architectural peak."))
    return ("\n".join(out) + "\n").encode("utf-8")


def render_gemm_svg(section: dict) -> bytes:
    values, minimums, maximums = [], [], []
    for shape in section["shapes"]:
        for candidate in shape["candidates"]:
            values.append(candidate["metrics"]["tflops"]["mean"])
            minimums.append(candidate["metrics"]["tflops"]["minimum"])
            maximums.append(candidate["metrics"]["tflops"]["maximum"])
    out = _svg_open("Cross-campaign mean of the campaign-level derived TFLOP/s per shape "
                    "and candidate")
    for index, (x0, y0, x1, y1) in enumerate(_panel_geometry(len(section["shapes"]))):
        shape = section["shapes"][index]
        series = [{
            "label": "candidates",
            "values": [candidate["metrics"]["tflops"]["mean"]
                       for candidate in shape["candidates"]],
            "minimums": [candidate["metrics"]["tflops"]["minimum"]
                         for candidate in shape["candidates"]],
            "maximums": [candidate["metrics"]["tflops"]["maximum"]
                         for candidate in shape["candidates"]],
        }]
        out.extend(_svg_panel(
            x0=x0, y0=y0, x1=x1, y1=y1, title=shape["shape_id"],
            x_labels=[str(candidate["candidate_index"])
                      for candidate in shape["candidates"]],
            y_label="TFLOP/s", series=series,
            y_min=min(minimums), y_max=max(maximums)))
    order = ", ".join(f"{index + 1}={name}"
                      for index, name in enumerate(section["candidate_order"]))
    out.extend(_svg_close(
        f"n={CAMPAIGN_COUNT} final campaigns; {WHISKER_CAPTION}. Candidate order: {order}. "
        f"TFLOP/s is derived from the exact operation count and the measured kernel time; "
        f"hot cache; beating cuBLASLt is not a success criterion."))
    return ("\n".join(out) + "\n").encode("utf-8")


# ===========================================================================
# The complete analysis: evidence in, deterministic documents out.
# ===========================================================================


def build_documents(orchestrator, p35, records: list[dict],
                    provenance: dict) -> list[tuple[str, bytes]]:
    """Every output artifact, in the frozen inventory order, as exact bytes."""
    if len(records) != CAMPAIGN_COUNT:
        raise P43Error(f"expected exactly {CAMPAIGN_COUNT} final campaign records")
    if [record["campaign_id"] for record in records] != list(FINAL_CAMPAIGN_IDS):
        raise P43Error("the campaign records are not the frozen final population in order")
    compare_campaign_provenance(records)
    validate_analysis_provenance(provenance)

    memory_rows, experiment_1 = aggregate_experiment_1(records)
    umma_rows, experiment_2 = aggregate_experiment_2(records)
    gemm_rows, experiment_3 = aggregate_experiment_3(records, p35=p35)
    sections = {"experiment_1": experiment_1, "experiment_2": experiment_2,
                "experiment_3": experiment_3}
    interpretation = build_interpretation(experiment_1, experiment_2, experiment_3)
    limitations = build_limitations(experiment_1, experiment_2, experiment_3)
    warnings = collect_source_warnings(experiment_1, experiment_2)
    reviews = collect_cross_campaign_reviews(sections)
    sources = [entry for record in records for entry in record["sources"]]

    summary = build_summary_document(records, sections, interpretation, limitations,
                                     sources, provenance, warnings, reviews)

    documents: list[tuple[str, bytes]] = [
        ("memory_paths.csv", render_csv(MEMORY_CSV_FIELDS, memory_rows)),
        ("umma_throughput.csv", render_csv(UMMA_CSV_FIELDS, umma_rows)),
        ("gemm_comparison.csv", render_csv(GEMM_CSV_FIELDS, gemm_rows)),
        ("integrated_summary.json", orchestrator.canonical_json_bytes(summary)),
        ("report.md", render_report(summary)),
        ("figures/memory_paths.svg", render_memory_svg(experiment_1)),
        ("figures/umma_throughput.svg", render_umma_svg(experiment_2)),
        ("figures/gemm_comparison.svg", render_gemm_svg(experiment_3)),
    ]
    documents.append((MANIFEST_RELATIVE_PATH, orchestrator.canonical_json_bytes(
        build_manifest(orchestrator, records, sources, provenance, documents))))
    if [relative for relative, _ in documents] != list(ARTIFACT_RELATIVE_PATHS):
        raise P43Error("the produced documents are not the frozen artifact inventory")
    return documents


def build_manifest(orchestrator, records: list[dict], sources: list[dict],
                   provenance: dict, siblings: list[tuple[str, bytes]]) -> dict:
    """The authoritative provenance envelope for the whole candidate bundle.

    Metadata ownership is deliberately central, not duplicated. The CSV files
    carry their own schema and deterministic data fields, the SVGs are
    deterministic visual artifacts, `integrated_summary.json` and `report.md`
    carry the scientific context -- and *this* file binds all eight of them by
    repository-relative path and SHA-256, together with the population, the
    provenance, and the candidate state.

    It structurally cannot contain its own byte hash: any value written into
    this document changes the very bytes that hash would describe. Its own hash
    is bound from outside, by `--verify`, which recomputes every byte of it, and
    later by the external acceptance attestation, which records
    `analysis_manifest_sha256` and thereby covers all nine artifacts without a
    self-reference.
    """
    sibling_hashes = {relative: orchestrator.sha256_bytes(payload)
                      for relative, payload in siblings}
    if set(sibling_hashes) != set(ARTIFACT_RELATIVE_PATHS) - {MANIFEST_RELATIVE_PATH}:
        raise P43Error("the manifest must bind exactly the eight non-manifest artifacts")
    return {
        "schema_version": SCHEMA_VERSION,
        "unit": UNIT,
        "analysis_kind": "cross_campaign_integrated_analysis",
        "role": (
            "the authoritative provenance envelope of this bundle: it binds every other "
            "artifact by repository-relative path and SHA-256. The individual CSV and SVG "
            "files are deterministic data and visual artifacts, not standalone provenance "
            "envelopes, and must be distributed together with this manifest"),
        "artifact_count": len(ARTIFACT_RELATIVE_PATHS),
        "artifact_inventory": list(ARTIFACT_RELATIVE_PATHS),
        "artifact_sha256": sibling_hashes,
        "self_hash": {
            "value": NOT_APPLICABLE,
            "reason": (
                "analysis_manifest.json cannot contain its own byte hash: writing the value "
                "would change the bytes it describes. This is a structural property, not an "
                "omission"),
            "bound_by": (
                f"'make phase4-p43-verify' recomputes every byte of this file from the "
                f"same evidence, and the later external acceptance attestation at "
                f"{ACCEPTANCE_RELATIVE_PATH} records analysis_manifest_sha256, which covers "
                f"all {len(ARTIFACT_RELATIVE_PATHS)} artifacts without any self-reference"),
        },
        "campaign_count": CAMPAIGN_COUNT,
        "final_campaign_ids": list(FINAL_CAMPAIGN_IDS),
        "campaign_value_column_order": [f"campaign_{index + 1}_value"
                                        for index in range(CAMPAIGN_COUNT)],
        "campaign_value_column_map": campaign_value_column_map(),
        "pilot_campaign_id_excluded": PILOT_CAMPAIGN_ID,
        "pilot_role": PILOT_ROLE,
        "final_execution_commit": records[0]["git_commit"],
        "analysis_code_commit": provenance["analysis_code_commit"],
        "analysis_code_worktree_clean": provenance["worktree_clean"],
        "analysis_code_clean_definition": provenance["clean_definition"],
        "analysis_code_verification_method": provenance["verification_method"],
        "analysis_code_tracked_paths_verified": provenance["tracked_paths_verified"],
        "gpu": dict(records[0]["gpu"]),
        "comparable_provenance": {key: records[0]["provenance"][key]
                                  for key in sorted(records[0]["provenance"])},
        "sources": sources,
        "publishable": PUBLISHABLE,
        "publication_state": PUBLICATION_STATE,
        "publication_status": PUBLICATION_STATUS,
        "acceptance": {
            "attestation_path": ACCEPTANCE_RELATIVE_PATH,
            "attestation_schema_version": ACCEPTANCE_SCHEMA_VERSION,
            "exists": False,
            "note": (
                "acceptance is an external attestation created only by a later, explicitly "
                "authorized closing action, after an independent reviewer has inspected "
                "this complete bundle. The analyzer never writes it, it is not part of the "
                "artifact inventory, and no candidate artifact is ever promoted, "
                "overwritten, or deleted to record it"),
            "lifecycle": list(ACCEPTANCE_LIFECYCLE),
        },
    }


def validate_analysis_provenance(provenance: object) -> None:
    """The analysis-code provenance a candidate is allowed to be built on."""
    if not isinstance(provenance, dict):
        raise P43Error("the analysis-code provenance is missing")
    for field in ("analysis_code_commit", "worktree_clean", "clean_definition",
                  "verification_method", "tracked_paths_verified"):
        if field not in provenance:
            raise P43Error(f"the analysis-code provenance has no {field!r}")
    commit = provenance["analysis_code_commit"]
    if not isinstance(commit, str) or not GIT_COMMIT_RE.fullmatch(commit):
        raise P43Error(f"analysis_code_commit={commit!r} is not a full 40-character "
                       f"lowercase Git commit")
    if provenance["worktree_clean"] is not True:
        raise P43Error("the analysis-code worktree is not verified clean; a candidate is "
                       "never produced from a dirty tree")
    if commit == FINAL_EXECUTION_COMMIT:
        # Not an error in principle, but it would mean the analysis code was
        # never committed after the campaigns ran, so the audited analysis
        # implementation cannot be the one that produced this bundle.
        raise P43Error(
            f"analysis_code_commit equals the frozen final execution commit "
            f"{FINAL_EXECUTION_COMMIT}; the P4.3 analysis code did not exist at that "
            f"commit, so this bundle cannot have been produced by it")


# ===========================================================================
# The external acceptance attestation (validation only).
#
# P4.3 never creates src/phase4/P4_3_ACCEPTANCE.json. This module only knows
# how to reject a wrong one, so that the later closing action has a frozen,
# reusable rule to be checked against instead of a prose description.
# ===========================================================================


def validate_acceptance_document(document: object, *, manifest_sha256: str,
                                 artifact_sha256: dict, analysis_code_commit: str
                                 ) -> list[str]:
    """Every way a future acceptance attestation can be wrong, as a list.

    An attestation that differs in any campaign ID, commit, path, inventory
    entry, or hash is rejected. It can therefore never authorize a modified,
    partially regenerated, or re-run bundle: the hashes it binds are the hashes
    of the exact reviewed bytes.
    """
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["the acceptance attestation is not a JSON object"]
    for field in ACCEPTANCE_REQUIRED_FIELDS:
        if field not in document:
            errors.append(f"the acceptance attestation has no {field!r}")
    if errors:
        return errors
    if document["schema_version"] != ACCEPTANCE_SCHEMA_VERSION:
        errors.append(f"schema_version={document['schema_version']!r}, expected "
                      f"{ACCEPTANCE_SCHEMA_VERSION!r}")
    if document["unit"] != UNIT:
        errors.append(f"unit={document['unit']!r}, expected {UNIT!r}")
    if document["status"] != ACCEPTANCE_STATUS_ACCEPTED:
        errors.append(f"status={document['status']!r}, expected "
                      f"{ACCEPTANCE_STATUS_ACCEPTED!r}")
    if document["accepted_for_publication"] is not True:
        errors.append("accepted_for_publication must be exactly true")
    if document["analysis_code_commit"] != analysis_code_commit:
        errors.append(
            f"analysis_code_commit={document['analysis_code_commit']!r} is not the commit "
            f"that produced this bundle ({analysis_code_commit}); an attestation never "
            f"transfers to a different analyzer commit")
    if document["final_campaign_ids"] != list(FINAL_CAMPAIGN_IDS):
        errors.append(f"final_campaign_ids={document['final_campaign_ids']!r} is not the "
                      f"frozen population in the frozen order")
    if document["pilot_campaign_id_excluded"] != PILOT_CAMPAIGN_ID:
        errors.append("pilot_campaign_id_excluded is not the accepted pilot")
    if document["analysis_manifest_sha256"] != manifest_sha256:
        errors.append(
            f"analysis_manifest_sha256={document['analysis_manifest_sha256']!r} does not "
            f"match the manifest of this bundle ({manifest_sha256}); the attestation binds "
            f"a different bundle")
    hashes = document["artifact_sha256"]
    if not isinstance(hashes, dict):
        errors.append("artifact_sha256 is not an object")
    else:
        expected = dict(artifact_sha256)
        expected[MANIFEST_RELATIVE_PATH] = manifest_sha256
        if set(hashes) != set(ARTIFACT_RELATIVE_PATHS):
            errors.append(
                f"artifact_sha256 must cover exactly the "
                f"{len(ARTIFACT_RELATIVE_PATHS)} P4.3 artifacts; got "
                f"{sorted(set(hashes) ^ set(ARTIFACT_RELATIVE_PATHS))} in symmetric "
                f"difference")
        for relative in sorted(set(hashes) & set(expected)):
            value = hashes[relative]
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                errors.append(f"artifact_sha256[{relative!r}] is not a canonical SHA-256")
            elif value != expected[relative]:
                errors.append(
                    f"artifact_sha256[{relative!r}] does not match the reviewed bytes; the "
                    f"attestation never authorizes a modified or partially regenerated "
                    f"artifact")
    if document["verification_outcome"] != ACCEPTANCE_VERIFICATION_OUTCOME:
        errors.append(f"verification_outcome={document['verification_outcome']!r}, expected "
                      f"{ACCEPTANCE_VERIFICATION_OUTCOME!r}")
    if document["independent_output_review_outcome"] != ACCEPTANCE_REVIEW_OUTCOME:
        errors.append(
            f"independent_output_review_outcome="
            f"{document['independent_output_review_outcome']!r}, expected "
            f"{ACCEPTANCE_REVIEW_OUTCOME!r}")
    return errors


def build_acceptance_template(manifest_sha256: str, artifact_sha256: dict,
                              analysis_code_commit: str) -> dict:
    """The exact shape a future acceptance attestation must have.

    This is a *shape*, produced only for validation self-tests and for the
    frozen schema in the protocol. Nothing in this module ever writes it to
    disk, and the repository checker requires the real file to be absent for as
    long as P4.3 is not accepted.
    """
    hashes = dict(artifact_sha256)
    hashes[MANIFEST_RELATIVE_PATH] = manifest_sha256
    return {
        "schema_version": ACCEPTANCE_SCHEMA_VERSION,
        "unit": UNIT,
        "status": ACCEPTANCE_STATUS_ACCEPTED,
        "accepted_for_publication": True,
        "analysis_code_commit": analysis_code_commit,
        "final_campaign_ids": list(FINAL_CAMPAIGN_IDS),
        "pilot_campaign_id_excluded": PILOT_CAMPAIGN_ID,
        "analysis_manifest_sha256": manifest_sha256,
        "artifact_sha256": {relative: hashes[relative]
                            for relative in ARTIFACT_RELATIVE_PATHS},
        "verification_outcome": ACCEPTANCE_VERIFICATION_OUTCOME,
        "independent_output_review_outcome": ACCEPTANCE_REVIEW_OUTCOME,
    }


# ===========================================================================
# Repository provenance: which commit's code produced this candidate.
#
# `final_execution_commit` (the commit the three GB300 campaigns RAN from) and
# `analysis_code_commit` (the commit whose analysis code produced this bundle)
# are different facts about different events. Confusing them would let a
# modified analyzer publish a candidate under an audited commit's name.
#
# This is a strict *reader* of an already existing Git repository, in pure
# Python. It starts no child process: P4.3 runs no `git`, no container, and no
# subprocess at all, and it needs no network. It never writes into `.git`.
#
# What it proves, exactly:
#
#   * HEAD resolves to one full 40-character lowercase commit;
#   * the index is byte-equal to that commit's tree (the index's own cache-tree
#     root OID equals the commit's tree OID), so nothing is staged;
#   * every tracked path exists in the worktree with exactly the indexed blob
#     content and mode, so nothing tracked is modified or deleted;
#   * no entry is unmerged, skip-worktree, or assume-unchanged, so nothing is
#     masked from that comparison;
#   * no repository operation (merge, rebase, cherry-pick, revert, bisect) is
#     in progress.
#
# What it deliberately does not treat as dirty: an *untracked* path. The
# analyzer's own output tree is untracked until someone commits it, so
# requiring its absence would make `--verify` unable to run after `--analyze`.
# Untracked files cannot change the content of any tracked file, which is the
# property that binds this bundle to the audited code.
# ===========================================================================

GIT_DIR_NAME = ".git"
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
GIT_INDEX_SIGNATURE = b"DIRC"
GIT_INDEX_SUPPORTED_VERSIONS = (2, 3, 4)
GIT_CACHE_TREE_EXTENSION = b"TREE"
# Index entry flag bits (see Git's Documentation/technical/index-format.txt).
GIT_FLAG_EXTENDED = 0x4000
GIT_FLAG_STAGE_MASK = 0x3000
GIT_EXTENDED_FLAG_SKIP_WORKTREE = 0x4000
GIT_EXTENDED_FLAG_INTENT_TO_ADD = 0x2000
GIT_MODE_REGULAR = 0o100644
GIT_MODE_EXECUTABLE = 0o100755
GIT_MODE_SYMLINK = 0o120000
GIT_IN_PROGRESS_MARKERS = (
    "MERGE_HEAD", "CHERRY_PICK_HEAD", "REVERT_HEAD", "BISECT_LOG",
    "rebase-merge", "rebase-apply",
)
GIT_PACK_IDX_MAGIC = b"\xfftOc"
GIT_PACK_MAGIC = b"PACK"
GIT_OBJ_COMMIT, GIT_OBJ_TREE, GIT_OBJ_BLOB, GIT_OBJ_TAG = 1, 2, 3, 4
GIT_OBJ_OFS_DELTA, GIT_OBJ_REF_DELTA = 6, 7
GIT_MAX_DELTA_CHAIN = 64

WORKTREE_CLEAN_DEFINITION = (
    "HEAD is one full 40-character commit; the index equals that commit's tree, so "
    "nothing is staged; every tracked path exists in the worktree with exactly the "
    "committed blob content and mode; no index entry is unmerged, skip-worktree, or "
    "assume-unchanged; and no merge, rebase, cherry-pick, revert, or bisect is in "
    "progress. Untracked paths are permitted and are recorded as such: they cannot "
    "change any tracked file, and the analyzer's own candidate output tree is itself "
    "untracked until it is committed"
)


class GitProvenanceError(P43Error):
    """The repository provenance of this analysis cannot be established.

    Always fatal in a production mode: a candidate that cannot name the exact
    clean commit whose code produced it is worse than no candidate."""


def _git_open_dir(root: Path, *parts: str) -> int:
    """Open a directory component by component with O_DIRECTORY | O_NOFOLLOW."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        current = os.open(str(root), flags)
    except OSError as exc:
        raise GitProvenanceError(f"{root}: cannot open the repository root: {exc}") from exc
    try:
        for part in parts:
            nxt = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = nxt
    except OSError as exc:
        os.close(current)
        raise GitProvenanceError(f"{'/'.join(parts)}: cannot open inside {root}: {exc}") from exc
    except BaseException:
        os.close(current)
        raise
    return current


def _git_read_regular(dir_fd: int, *parts: str) -> tuple[bytes, int] | None:
    """Read one regular file under an already opened directory, following no
    symlink at any component, and return its bytes together with its own
    st_mode. Returns None when the path simply does not exist."""
    current = os.dup(dir_fd)
    try:
        for part in parts[:-1]:
            flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            try:
                nxt = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                return None
            except OSError as exc:
                raise GitProvenanceError(f"{'/'.join(parts)}: {exc}") from exc
            os.close(current)
            current = nxt
        flags = os.O_RDONLY | os.O_NOFOLLOW
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            fd = os.open(parts[-1], flags, dir_fd=current)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise GitProvenanceError(f"{'/'.join(parts)}: {exc}") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode):
                raise GitProvenanceError(f"{'/'.join(parts)}: is not a regular file")
            chunks = []
            while True:
                block = os.read(fd, 1 << 20)
                if not block:
                    break
                chunks.append(block)
            return b"".join(chunks), info.st_mode
        finally:
            os.close(fd)
    finally:
        os.close(current)


def _git_read(dir_fd: int, *parts: str) -> bytes | None:
    found = _git_read_regular(dir_fd, *parts)
    return None if found is None else found[0]


def _git_exists(dir_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise GitProvenanceError(f"{name}: cannot inspect: {exc}") from exc
    return True


def _git_resolve_head(git_fd: int) -> str:
    """HEAD -> one full commit, through loose refs and packed-refs only."""
    payload = _git_read(git_fd, "HEAD")
    if payload is None:
        raise GitProvenanceError("the repository has no HEAD; provenance cannot be verified")
    text = decode_utf8(payload, "HEAD").strip()
    if GIT_COMMIT_RE.fullmatch(text):
        return text
    if not text.startswith("ref: "):
        raise GitProvenanceError(f"HEAD={text!r} is neither a full commit nor a symbolic ref")
    ref = text[len("ref: "):].strip()
    if not ref.startswith("refs/") or ".." in ref.split("/") or "" in ref.split("/"):
        raise GitProvenanceError(f"HEAD names the unsupported ref {ref!r}")
    loose = _git_read(git_fd, *ref.split("/"))
    if loose is not None:
        candidate = decode_utf8(loose, ref).strip()
        if not GIT_COMMIT_RE.fullmatch(candidate):
            raise GitProvenanceError(f"{ref} does not contain a full 40-character commit")
        return candidate
    packed = _git_read(git_fd, "packed-refs")
    if packed is None:
        raise GitProvenanceError(f"{ref} resolves to nothing; HEAD is unborn or detached "
                                 f"from any object")
    for line in decode_utf8(packed, "packed-refs").splitlines():
        if not line or line.startswith(("#", "^")):
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[1].strip() == ref:
            candidate = parts[0].strip()
            if not GIT_COMMIT_RE.fullmatch(candidate):
                raise GitProvenanceError(f"packed-refs holds a malformed OID for {ref}")
            return candidate
    raise GitProvenanceError(f"{ref} resolves to nothing; HEAD is unborn")


def _git_parse_varint_size(data: bytes, offset: int) -> tuple[int, int, int]:
    """The pack entry header: object type plus its inflated size."""
    byte = data[offset]
    offset += 1
    object_type = (byte >> 4) & 0x07
    size = byte & 0x0F
    shift = 4
    while byte & 0x80:
        byte = data[offset]
        offset += 1
        size |= (byte & 0x7F) << shift
        shift += 7
    return object_type, size, offset


def _git_apply_delta(base: bytes, delta: bytes) -> bytes:
    def read_size(position: int) -> tuple[int, int]:
        size, shift = 0, 0
        while True:
            byte = delta[position]
            position += 1
            size |= (byte & 0x7F) << shift
            shift += 7
            if not byte & 0x80:
                return size, position

    base_size, position = read_size(0)
    if base_size != len(base):
        raise GitProvenanceError("a packed delta does not match its base object size")
    result_size, position = read_size(position)
    out: list[bytes] = []
    while position < len(delta):
        opcode = delta[position]
        position += 1
        if opcode & 0x80:
            copy_offset = copy_size = 0
            for index in range(4):
                if opcode & (1 << index):
                    copy_offset |= delta[position] << (index * 8)
                    position += 1
            for index in range(3):
                if opcode & (1 << (4 + index)):
                    copy_size |= delta[position] << (index * 8)
                    position += 1
            if copy_size == 0:
                copy_size = 0x10000
            out.append(base[copy_offset:copy_offset + copy_size])
        elif opcode:
            out.append(delta[position:position + opcode])
            position += opcode
        else:
            raise GitProvenanceError("a packed delta contains a reserved opcode")
    payload = b"".join(out)
    if len(payload) != result_size:
        raise GitProvenanceError("a packed delta produced the wrong object size")
    return payload


def _git_pack_names(git_fd: int) -> list[str]:
    try:
        pack_fd = _git_open_dir_fd(git_fd, "objects", "pack")
    except FileNotFoundError:
        return []
    try:
        return sorted(name for name in os.listdir(pack_fd) if name.endswith(".idx"))
    finally:
        os.close(pack_fd)


def _git_open_dir_fd(dir_fd: int, *parts: str) -> int:
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    current = os.dup(dir_fd)
    try:
        for part in parts:
            nxt = os.open(part, flags, dir_fd=current)
            os.close(current)
            current = nxt
    except BaseException:
        os.close(current)
        raise
    return current


def _git_idx_lookup(index_payload: bytes, oid: str) -> int | None:
    """Find one object's pack offset in a version-2 pack index."""
    if index_payload[:4] != GIT_PACK_IDX_MAGIC or index_payload[4:8] != b"\x00\x00\x00\x02":
        raise GitProvenanceError("only version-2 pack indexes are supported")
    fanout_end = 8 + 256 * 4
    total = struct.unpack(">I", index_payload[fanout_end - 4:fanout_end])[0]
    raw = bytes.fromhex(oid)
    first = raw[0]
    low = 0 if first == 0 else struct.unpack(
        ">I", index_payload[8 + (first - 1) * 4:8 + first * 4])[0]
    high = struct.unpack(">I", index_payload[8 + first * 4:8 + (first + 1) * 4])[0]
    names = fanout_end
    position = None
    while low < high:
        middle = (low + high) // 2
        entry = index_payload[names + middle * 20:names + middle * 20 + 20]
        if entry == raw:
            position = middle
            break
        if entry < raw:
            low = middle + 1
        else:
            high = middle
    if position is None:
        return None
    offsets = names + total * 20 + total * 4
    value = struct.unpack(">I", index_payload[offsets + position * 4:
                                              offsets + position * 4 + 4])[0]
    if value & 0x80000000:
        large = offsets + total * 4
        slot = value & 0x7FFFFFFF
        return struct.unpack(">Q", index_payload[large + slot * 8:large + slot * 8 + 8])[0]
    return value


def _git_read_packed(pack_payload: bytes, offset: int, resolve, depth: int = 0
                     ) -> tuple[int, bytes]:
    if depth > GIT_MAX_DELTA_CHAIN:
        raise GitProvenanceError("a packed delta chain is longer than the accepted limit")
    object_type, size, position = _git_parse_varint_size(pack_payload, offset)
    if object_type in (GIT_OBJ_COMMIT, GIT_OBJ_TREE, GIT_OBJ_BLOB, GIT_OBJ_TAG):
        payload = zlib.decompressobj().decompress(pack_payload[position:], size)
        if len(payload) != size:
            raise GitProvenanceError("a packed object inflated to the wrong size")
        return object_type, payload
    if object_type == GIT_OBJ_OFS_DELTA:
        byte = pack_payload[position]
        position += 1
        base_offset = byte & 0x7F
        while byte & 0x80:
            byte = pack_payload[position]
            position += 1
            base_offset = ((base_offset + 1) << 7) | (byte & 0x7F)
        base_type, base = _git_read_packed(pack_payload, offset - base_offset, resolve,
                                           depth + 1)
    elif object_type == GIT_OBJ_REF_DELTA:
        base_oid = pack_payload[position:position + 20].hex()
        position += 20
        base_type, base = resolve(base_oid, depth + 1)
    else:
        raise GitProvenanceError(f"unsupported packed object type {object_type}")
    delta = zlib.decompressobj().decompress(pack_payload[position:], size)
    return base_type, _git_apply_delta(base, delta)


def _git_read_object(git_fd: int, oid: str, depth: int = 0) -> tuple[str, bytes]:
    """One Git object by OID, from a loose file or from a pack."""
    if not GIT_COMMIT_RE.fullmatch(oid):
        raise GitProvenanceError(f"{oid!r} is not a full object id")
    loose = _git_read(git_fd, "objects", oid[:2], oid[2:])
    if loose is not None:
        try:
            payload = zlib.decompress(loose)
        except zlib.error as exc:
            raise GitProvenanceError(f"object {oid} is not a readable loose object") from exc
        separator = payload.index(b"\0")
        header = payload[:separator].split(b" ")
        if len(header) != 2:
            raise GitProvenanceError(f"object {oid} has a malformed header")
        return header[0].decode("ascii"), payload[separator + 1:]
    for name in _git_pack_names(git_fd):
        index_payload = _git_read(git_fd, "objects", "pack", name)
        if index_payload is None:
            continue
        offset = _git_idx_lookup(index_payload, oid)
        if offset is None:
            continue
        pack_payload = _git_read(git_fd, "objects", "pack", name[:-4] + ".pack")
        if pack_payload is None or pack_payload[:4] != GIT_PACK_MAGIC:
            raise GitProvenanceError(f"pack {name[:-4]}.pack is missing or malformed")

        def resolve(base_oid: str, next_depth: int) -> tuple[int, bytes]:
            kind, body = _git_read_object(git_fd, base_oid, next_depth)
            return {"commit": GIT_OBJ_COMMIT, "tree": GIT_OBJ_TREE, "blob": GIT_OBJ_BLOB,
                    "tag": GIT_OBJ_TAG}[kind], body

        if depth > GIT_MAX_DELTA_CHAIN:
            raise GitProvenanceError("a delta chain is longer than the accepted limit")
        object_type, payload = _git_read_packed(pack_payload, offset, resolve, depth)
        return {GIT_OBJ_COMMIT: "commit", GIT_OBJ_TREE: "tree", GIT_OBJ_BLOB: "blob",
                GIT_OBJ_TAG: "tag"}[object_type], payload
    raise GitProvenanceError(f"object {oid} is not present in this repository")


def _git_commit_tree(git_fd: int, commit: str) -> str:
    kind, payload = _git_read_object(git_fd, commit)
    if kind != "commit":
        raise GitProvenanceError(f"{commit} is a {kind}, not a commit")
    for line in payload.split(b"\n"):
        if line.startswith(b"tree "):
            tree = line[5:].decode("ascii").strip()
            if not GIT_COMMIT_RE.fullmatch(tree):
                raise GitProvenanceError(f"commit {commit} names a malformed tree")
            return tree
        if not line:
            break
    raise GitProvenanceError(f"commit {commit} names no tree")


def _git_parse_index(payload: bytes) -> tuple[list[dict], str | None]:
    """Parse `.git/index`: every entry plus the root cache-tree OID."""
    if len(payload) < 12 or payload[:4] != GIT_INDEX_SIGNATURE:
        raise GitProvenanceError("the Git index is missing or malformed")
    version, count = struct.unpack(">II", payload[4:12])
    if version not in GIT_INDEX_SUPPORTED_VERSIONS:
        raise GitProvenanceError(f"Git index version {version} is not supported; "
                                 f"provenance cannot be verified")
    if version == 4:
        raise GitProvenanceError("Git index version 4 uses prefix-compressed paths, which "
                                 "this reader deliberately does not decode; provenance "
                                 "cannot be verified")
    if hashlib.sha1(payload[:-20]).digest() != payload[-20:]:
        raise GitProvenanceError("the Git index fails its own SHA-1 checksum")
    entries: list[dict] = []
    offset = 12
    for _ in range(count):
        start = offset
        mode = struct.unpack(">I", payload[offset + 24:offset + 28])[0]
        oid = payload[offset + 40:offset + 60].hex()
        flags = struct.unpack(">H", payload[offset + 60:offset + 62])[0]
        offset += 62
        extended = 0
        if flags & GIT_FLAG_EXTENDED:
            if version < 3:
                raise GitProvenanceError("a version-2 Git index cannot carry extended flags")
            extended = struct.unpack(">H", payload[offset:offset + 2])[0]
            offset += 2
        end = payload.index(b"\0", offset)
        path = payload[offset:end]
        offset = end + 1
        offset = start + ((offset - start + 7) // 8) * 8
        entries.append({
            "path": path,
            "mode": mode,
            "oid": oid,
            "stage": (flags & GIT_FLAG_STAGE_MASK) >> 12,
            "skip_worktree": bool(extended & GIT_EXTENDED_FLAG_SKIP_WORKTREE),
            "intent_to_add": bool(extended & GIT_EXTENDED_FLAG_INTENT_TO_ADD),
        })
    cache_tree = None
    while offset + 8 <= len(payload) - 20:
        name = payload[offset:offset + 4]
        length = struct.unpack(">I", payload[offset + 4:offset + 8])[0]
        body = payload[offset + 8:offset + 8 + length]
        if name == GIT_CACHE_TREE_EXTENSION:
            cache_tree = _git_parse_cache_tree_root(body)
        offset += 8 + length
    return entries, cache_tree


def _git_parse_cache_tree_root(body: bytes) -> str | None:
    """The root entry of the cache-tree extension, when it is valid.

    Layout (Git's index-format.txt): a NUL-terminated path component -- empty
    for the root -- then the ASCII entry count, a space, the subtree count, a
    newline, and, only when the entry count is not negative, the 20-byte tree
    object name. A negative entry count means the cached tree was invalidated,
    which is exactly what happens when something is staged; there is then
    nothing to trust and this returns None."""
    separator = body.index(b"\0")
    rest = body[separator + 1:]
    newline = rest.index(b"\n")
    header = rest[:newline].split(b" ")
    if len(header) != 2:
        return None
    try:
        entry_count = int(header[0])
    except ValueError:
        return None
    if entry_count < 0:
        return None
    oid = rest[newline + 1:newline + 21]
    if len(oid) != 20:
        return None
    return oid.hex()


def _git_blob_oid(payload: bytes) -> str:
    return hashlib.sha1(b"blob " + str(len(payload)).encode("ascii") + b"\0"
                        + payload).hexdigest()


def resolve_git_provenance(repo_root: Path) -> dict:
    """The analysis-code commit of this run, verified, never assumed."""
    try:
        git_fd = _git_open_dir(repo_root, GIT_DIR_NAME)
    except GitProvenanceError as exc:
        raise GitProvenanceError(
            f"{repo_root}: no readable .git directory; the analysis-code commit cannot be "
            f"resolved or verified, so no candidate is produced ({exc})") from exc
    try:
        head_commit = _git_resolve_head(git_fd)
        for marker in GIT_IN_PROGRESS_MARKERS:
            if _git_exists(git_fd, marker):
                raise GitProvenanceError(
                    f"a repository operation is in progress ({marker}); the worktree is not "
                    f"a clean checkout of {head_commit}")
        head_tree = _git_commit_tree(git_fd, head_commit)
        index_payload = _git_read(git_fd, "index")
        if index_payload is None:
            raise GitProvenanceError("the repository has no index; provenance cannot be "
                                     "verified")
        entries, cache_tree = _git_parse_index(index_payload)
        if cache_tree is None:
            raise GitProvenanceError(
                "the Git index carries no valid cache-tree, so it cannot be shown to equal "
                "any commit's tree and cleanliness cannot be proven. This is a refusal, "
                "never an assumption: check out or clone the audited commit afresh, or run "
                "'git write-tree' once to rebuild the index's cache-tree, and re-run")
        if cache_tree != head_tree:
            raise GitProvenanceError(
                f"the index tree {cache_tree} differs from the tree of HEAD "
                f"({head_tree}): the index is dirty (something is staged)")
        tracked = 0
        for entry in entries:
            label = entry["path"].decode("utf-8", errors="replace")
            if entry["stage"] != 0:
                raise GitProvenanceError(f"{label}: is unmerged; the index is dirty")
            if entry["skip_worktree"] or entry["intent_to_add"]:
                raise GitProvenanceError(
                    f"{label}: is marked skip-worktree or intent-to-add, which would hide a "
                    f"difference from the worktree comparison")
            parts = tuple(entry["path"].decode("utf-8").split("/"))
            if entry["mode"] == GIT_MODE_SYMLINK:
                raise GitProvenanceError(
                    f"{label}: is a tracked symlink; this repository tracks none, and "
                    f"following one to verify it is exactly what the analyzer refuses to do")
            if entry["mode"] not in (GIT_MODE_REGULAR, GIT_MODE_EXECUTABLE):
                raise GitProvenanceError(f"{label}: unsupported tracked mode "
                                         f"{entry['mode']:o}")
            found = _git_read_regular(_git_root_fd_cache(repo_root), *parts)
            if found is None:
                raise GitProvenanceError(f"{label}: is tracked at HEAD but missing from the "
                                         f"worktree; the worktree is dirty")
            payload, st_mode = found
            if _git_blob_oid(payload) != entry["oid"]:
                raise GitProvenanceError(f"{label}: differs from its committed content; the "
                                         f"worktree is dirty")
            executable = bool(st_mode & stat.S_IXUSR)
            if executable != (entry["mode"] == GIT_MODE_EXECUTABLE):
                raise GitProvenanceError(f"{label}: its executable bit differs from the "
                                         f"committed mode; the worktree is dirty")
            tracked += 1
        return {
            "analysis_code_commit": head_commit,
            "analysis_code_tree": head_tree,
            "worktree_clean": True,
            "clean_definition": WORKTREE_CLEAN_DEFINITION,
            "tracked_paths_verified": tracked,
            "verification_method": (
                "pure-Python read of .git: HEAD and refs resolved without a child process, "
                "the index's cache-tree root compared to the HEAD commit's tree, and every "
                "tracked path's blob SHA-1 and mode recomputed from the worktree"),
        }
    finally:
        os.close(git_fd)
        _git_root_fd_cache.cache_clear()


# Every function in the repository-provenance reader. All of them are strictly
# read-only and start no child process; the P4.3 checker proves that
# mechanically from their own source rather than from this comment.
GIT_PROVENANCE_FUNCTION_NAMES = (
    "_git_open_dir",
    "_git_open_dir_fd",
    "_git_read_regular",
    "_git_read",
    "_git_exists",
    "_git_resolve_head",
    "_git_parse_varint_size",
    "_git_apply_delta",
    "_git_pack_names",
    "_git_idx_lookup",
    "_git_read_packed",
    "_git_read_object",
    "_git_commit_tree",
    "_git_parse_index",
    "_git_parse_cache_tree_root",
    "_git_blob_oid",
    "resolve_git_provenance",
)


class _RootFdCache:
    """One descriptor for the repository root, reused for the tracked-file scan
    and closed when the provenance check finishes."""

    def __init__(self):
        self._entries: dict[str, int] = {}

    def __call__(self, repo_root: Path) -> int:
        key = str(repo_root)
        if key not in self._entries:
            self._entries[key] = _git_open_dir(repo_root)
        return self._entries[key]

    def cache_clear(self) -> None:
        for fd in self._entries.values():
            os.close(fd)
        self._entries.clear()


_git_root_fd_cache = _RootFdCache()


# ===========================================================================
# Output publication and verification.
# ===========================================================================


def resolve_output_root(output_root: Path, repo_root: Path) -> Path:
    """Validate the *declared* output root against the one legal destination.

    Production output is limited to exactly ``<repo-root>/results/phase4``. An
    arbitrary in-repository directory is refused, because "somewhere inside the
    repository" is not a destination a reviewer can verify.

    This is a check of what the operator *asked for*. It decides nothing about
    the filesystem: the containment guarantee comes from
    :func:`open_output_tree`, which walks the components from an already opened
    repository descriptor and never re-resolves a name. A lexical check alone is
    exactly the defect an ancestor symlink walks straight through.
    """
    absolute = Path(os.path.abspath(str(output_root)))
    try:
        relative = absolute.relative_to(Path(os.path.abspath(str(repo_root))))
    except ValueError as exc:
        raise P43Error(f"{output_root}: is outside the repository root {repo_root}") from exc
    parts = relative.parts
    if not parts:
        raise P43Error("the output root must not be the repository root itself")
    for forbidden in FORBIDDEN_OUTPUT_PREFIXES:
        if parts[:len(forbidden)] == forbidden:
            raise P43Error(
                f"{'/'.join(parts)}: P4.3 never writes under {'/'.join(forbidden)}; raw "
                f"evidence is immutable")
    if parts != OUTPUT_ROOT_COMPONENTS:
        raise P43Error(
            f"{'/'.join(parts)}: the only production output root is "
            f"{DEFAULT_OUTPUT_ROOT_REL}; an arbitrary in-repository output directory is "
            f"refused")
    return absolute


def _open_output_component(name: str, *, parent_fd: int, create: bool, label: str) -> int:
    """Open exactly one output path component, anchored on its parent.

    O_DIRECTORY | O_NOFOLLOW is the whole guarantee: a symlink at this component
    fails with ELOOP whatever it points at, and a non-directory fails with
    ENOTDIR. mkdir()'s own EEXIST is the only existence test, so a symlink that
    appears between the check and the create is refused rather than followed.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if "/" in name or name in ("", ".", "..") or "\0" in name:
        raise P43Error(f"{label}: {name!r} is not a single path component")
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise P43Error(f"{label}: does not exist; nothing to verify") from None
    except OSError as exc:
        raise P43Error(_component_refusal(label, exc, name, parent_fd)) from exc
    try:
        os.mkdir(name, 0o755, dir_fd=parent_fd)
    except OSError as exc:
        raise P43Error(f"{label}: cannot create the output directory: {exc}") from exc
    try:
        return os.open(name, flags, dir_fd=parent_fd)
    except OSError as exc:
        raise P43Error(_component_refusal(label, exc, name, parent_fd)) from exc


def _component_refusal(label: str, exc: OSError, name: str, parent_fd: int) -> str:
    """Name the refusal accurately.

    O_DIRECTORY combined with O_NOFOLLOW reports a symlinked component as
    ENOTDIR on Linux rather than ELOOP, so the errno alone cannot tell a
    symlink from an ordinary file. The open has already failed and nothing was
    followed; this lstat only decides what to call the thing that stopped it.
    """
    if exc.errno in (errno.ELOOP, errno.ENOTDIR):
        try:
            info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError:
            info = None
        if exc.errno == errno.ELOOP or (info is not None
                                        and stat.S_ISLNK(info.st_mode)):
            return (f"{label}: is a symlink; refusing to publish through a symlinked "
                    f"output path component")
        return f"{label}: exists and is not a directory; refusing"
    return f"{label}: cannot open the output path component: {exc}"


def open_output_tree(repo_root: Path, *, create: bool) -> dict:
    """Open the output tree component by component from the repository root.

    ``repo_root`` itself is opened once with O_DIRECTORY | O_NOFOLLOW, and every
    later component -- ``results``, ``phase4``, ``figures`` -- is opened
    relative to the previously opened descriptor and never by pathname again.
    A symlink at *any* of those levels, including an ancestor such as
    ``results`` pointing outside the repository, is rejected before a single
    byte is written. ``Path.resolve()``, ``abspath()``, a string prefix, and a
    lexical ``relative_to()`` are never the safety decision.
    """
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    try:
        root_fd = os.open(str(repo_root), flags)
    except OSError as exc:
        raise P43Error(f"{repo_root}: cannot open the repository root: {exc}") from exc
    opened: list[int] = [root_fd]
    try:
        current = root_fd
        walked: list[str] = []
        for component in OUTPUT_ROOT_COMPONENTS:
            walked.append(component)
            current = _open_output_component(component, parent_fd=current, create=create,
                                             label="/".join(walked))
            opened.append(current)
        figures_fd = _open_output_component(
            OUTPUT_FIGURES_DIR, parent_fd=current, create=create,
            label="/".join(walked + [OUTPUT_FIGURES_DIR]))
        opened.append(figures_fd)
    except BaseException:
        for fd in opened:
            os.close(fd)
        raise
    # Only the two descriptors the publication itself uses stay open.
    for fd in opened:
        if fd not in (current, figures_fd):
            os.close(fd)
    return {None: current, OUTPUT_FIGURES_DIR: figures_fd}


def close_output_tree(tree: dict) -> None:
    for fd in tree.values():
        os.close(fd)


def publish_documents(orchestrator, tree: dict, documents: list[tuple[str, bytes]],
                      *, write: bool) -> dict[str, str]:
    """Publish (or verify) the frozen inventory through the opened descriptors.

    Nothing is ever overwritten. Every artifact is created with
    ``O_CREAT | O_EXCL | O_NOFOLLOW``; an artifact that already exists must be
    byte-identical, in which case it is verified rather than rewritten, and a
    different existing artifact is fatal. There is no promotion path and no
    deletion path: a candidate is immutable once written.
    """
    outcomes: dict[str, str] = {}
    for relative, payload in documents:
        parts = split_relative_path(relative)
        if len(parts) == 1:
            subdirectory, name = None, parts[0]
        elif len(parts) == 2:
            subdirectory, name = parts[0], parts[1]
        else:
            raise P43Error(f"{relative}: the output inventory is at most one level deep")
        if subdirectory not in tree:
            raise P43Error(f"{relative}: names an output subdirectory outside the frozen "
                           f"tree")
        directory_fd = tree[subdirectory]
        try:
            existing = orchestrator.read_file_nofollow(name, dir_fd=directory_fd)
        except orchestrator.OrchestratorError as exc:
            if not _is_missing(exc):
                raise P43Error(f"{relative}: {exc}") from exc
            if not write:
                raise P43Error(f"{relative}: is missing; nothing to verify") from exc
            try:
                orchestrator.write_file_exclusive(name, payload, dir_fd=directory_fd)
            except orchestrator.OrchestratorError as inner:
                raise P43Error(f"{relative}: {inner}") from inner
            outcomes[relative] = "written"
            continue
        if existing != payload:
            raise P43Error(
                f"{relative}: already exists with different content; refusing to overwrite "
                f"a candidate artifact")
        outcomes[relative] = "verified_byte_identical"
    return outcomes


def _is_missing(exc: Exception) -> bool:
    """Distinguish "this artifact does not exist yet" from every other refusal.

    The orchestrator wraps the underlying OSError and chains it, so the errno is
    authoritative; the message scan is only a fallback for a future wrapper that
    does not chain."""
    cause = getattr(exc, "__cause__", None)
    if isinstance(cause, OSError):
        return cause.errno == errno.ENOENT
    return "No such file or directory" in str(exc)


def assert_output_tree_exact(tree: dict) -> None:
    """The output tree must contain exactly the frozen inventory.

    The scan runs entirely on the already validated descriptors and lstats every
    name relative to them, so no path is re-resolved and no symlink is ever
    followed. A partial, conflicting, or unexpected artifact, a symlink, a
    directory where a file belongs, and any other file type are all fatal.
    """
    expected_files = set(ARTIFACT_RELATIVE_PATHS)
    observed_files: set[str] = set()
    for subdirectory, directory_fd in tree.items():
        for entry in sorted(os.listdir(directory_fd)):
            label = entry if subdirectory is None else f"{subdirectory}/{entry}"
            info = os.stat(entry, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                if subdirectory is None and entry == OUTPUT_FIGURES_DIR:
                    continue
                raise P43Error(f"{label}/: unexpected directory in the output tree")
            if not stat.S_ISREG(info.st_mode):
                raise P43Error(f"{label}: is not a regular file (symlinks and special "
                               f"files are rejected)")
            observed_files.add(label)
    if observed_files != expected_files:
        raise P43Error(
            f"the output tree is not exactly the frozen inventory: "
            f"missing={sorted(expected_files - observed_files)}, "
            f"unexpected={sorted(observed_files - expected_files)}")


# ===========================================================================
# Drivers.
# ===========================================================================


def default_revalidator(orchestrator, p42, campaign_root: Path, pilot_ids: list[str],
                        final_ids: list[str]) -> int:
    """The deep population revalidation, delegated entirely to P4.2's own
    read-only evidence mode, which in turn delegates every per-campaign decision
    to P4.1. P4.3 implements no second interpretation of those contracts."""
    return p42.check_campaign_evidence(orchestrator, campaign_root, pilot_ids, final_ids)


def run_analysis(repo_root: Path, campaign_root: Path, pilot_ids: list[str],
                 final_ids: list[str], output_root: Path, *, write: bool,
                 revalidator=default_revalidator,
                 git_provenance=resolve_git_provenance) -> int:
    prefix = "analyze_phase4_p43"
    try:
        validate_declared_population(pilot_ids, final_ids)
    except P43Error as exc:
        print(f"{prefix}: FAILED: {exc}", file=sys.stderr)
        return 1
    orchestrator, p42, p35 = load_repository_modules(repo_root)

    # The authoritative gate runs first: not one scientific value is read and not
    # one byte is written until the whole frozen population has passed P4.2's
    # own strictly read-only revalidation.
    status = revalidator(orchestrator, p42, campaign_root, list(pilot_ids), list(final_ids))
    if status != 0:
        print(f"{prefix}: FAILED: the frozen Phase 4 population did not pass P4.2's "
              f"read-only revalidation; no value was read and nothing was written",
              file=sys.stderr)
        return 1

    tree = None
    try:
        # Which commit's code is producing this candidate, verified before any
        # value is read. There is no flag that skips this and no default that
        # stands in for it: a candidate that cannot name its own clean
        # analysis-code commit is never produced.
        provenance = git_provenance(repo_root)
        validate_analysis_provenance(provenance)
        resolve_output_root(output_root, repo_root)
        repo_from_campaigns = p42.campaign_root_to_repo_root(campaign_root, orchestrator)
        if Path(os.path.abspath(str(repo_from_campaigns))) != Path(
                os.path.abspath(str(repo_root))):
            raise P43Error(f"the campaign root belongs to {repo_from_campaigns}, not the "
                           f"repository being analysed ({repo_root})")
        records = []
        for campaign_id in final_ids:
            manifest = load_phase4_manifest(orchestrator, repo_root, campaign_id)
            records.append(collect_campaign_evidence(
                orchestrator, p35, repo_root, campaign_id, manifest))
        documents = build_documents(orchestrator, p35, records, provenance)
        # Every path component from the repository root down is opened here,
        # never resolved by name, so an ancestor symlink cannot redirect a byte.
        tree = open_output_tree(repo_root, create=write)
        outcomes = publish_documents(orchestrator, tree, documents, write=write)
        assert_output_tree_exact(tree)
    except (P43Error, p42.EvidenceError) as exc:
        print(f"{prefix}: FAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        if tree is not None:
            close_output_tree(tree)

    for relative in ARTIFACT_RELATIVE_PATHS:
        print(f"{prefix}: {outcomes[relative]}: {DEFAULT_OUTPUT_ROOT_REL}/{relative}")
    mode = "analyze" if write else "verify"
    print(f"{prefix}: {mode}: OK ({CAMPAIGN_COUNT} final campaigns; the accepted pilot "
          f"{PILOT_CAMPAIGN_ID} was excluded from every statistic; analysis-code commit "
          f"{provenance['analysis_code_commit']} on a verified clean worktree; no raw "
          f"evidence was modified; {PUBLICATION_STATUS})")
    return 0


# ===========================================================================
# Self-test. Temporary directories only; the repository is never modified.
# ===========================================================================


class _Reporter:
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

    def rejects(self, label: str, call, fragment: str = "") -> bool:
        try:
            call()
        except P43Error as exc:
            return self.check(label, fragment in str(exc), f"{fragment!r} not in {exc}")
        return self.check(label, False, "no P43Error was raised")


def _fixture_record(campaign_id: str, *, gbps_scale: float = 1.0,
                    ratio_scale: float = 1.0, saturation_kib: int = 64,
                    fpc_scale: float = 1.0, depth_saturation: int = 256,
                    ceiling_depth: int = 256, tflops: float = 16.0,
                    device_available: bool = False, sm_count: int | None = None,
                    gemm_scale: float = 1.0, best_variant: str = "persistent_2cta",
                    commit: str = FINAL_EXECUTION_COMMIT,
                    gpu_uuid: str = "GPU-11111111-2222-3333-4444-555555555555",
                    memory_cv_percent: float = 0.1,
                    memory_stability_review: str = "ok",
                    memory_sample_count: int = 30,
                    ncu_diagnostic_flags: str = "",
                    umma_cv_percent: float = 0.01,
                    umma_stability_review: str = "ok",
                    surprising_value_flag: str = "False") -> dict:
    """One synthetic, obviously fake campaign record shaped exactly like a
    parsed one. No real evidence is involved."""
    pilot_statistics = {}
    for index, key in enumerate(P14_CONFIG_KEYS):
        pilot_statistics[key] = {
            "median_gbps": (3000.0 + 100.0 * index) * gbps_scale,
            "within_campaign_cv_percent": memory_cv_percent,
            "within_campaign_sample_count": memory_sample_count,
            "within_campaign_stability_review": memory_stability_review,
        }
    pairwise = {key: {"tma_to_ldgsts_ratio": (0.97 + 0.001 * index) * ratio_scale,
                      "interpretation": "ldgsts_higher"}
                for index, key in enumerate(P14_PAIR_KEYS)}
    saturation = {key: saturation_kib for key in P14_SATURATION_KEYS}
    ncu = {key: {"dram_read_ratio": 1.0 + 0.0001 * index,
                 "hbm_classification": "HBM_VALIDATED",
                 "diagnostic_flags": ncu_diagnostic_flags}
           for index, key in enumerate(P14_NCU_CASES)}
    configuration = {}
    for index, key in enumerate(P24_CONFIG_KEYS):
        total = (2000.0 + 100.0 * index) * fpc_scale
        configuration[key] = {
            "cta_group": P24_CTA_GROUP[key[0]],
            "median_flops_per_cycle": total,
            "median_flops_per_cycle_per_sm": total / P24_CTA_GROUP[key[0]],
            "within_campaign_cv_percent": umma_cv_percent,
            "within_campaign_sample_count": 30,
            "within_campaign_stability_review": umma_stability_review,
        }
    scaling = {key: {"speedup_2sm_over_1sm": 1.5 + 0.01 * index,
                     "scaling_efficiency_percent": 75.0 + 0.5 * index,
                     "surprising_value_flag": surprising_value_flag}
               for index, key in enumerate(P24_SCALING_KEYS)}
    umma_saturation = {key: depth_saturation for key in P24_SATURATION_KEYS}
    device = ({"available": True, "multiprocessor_count": sm_count,
               "estimated_device_equivalent_tflops": tflops * (sm_count or 1)}
              if device_available else
              {"available": False, "reason": "the SM-count metric was not resolved"})
    gemm_rows = {}
    best_by_shape = {}
    # Candidate 3 is deliberately faster than the cuBLASLt baseline, so the
    # fixture exercises a ratio above one and an unclamped negative gap;
    # candidate 4 is the baseline itself, with ratio exactly one and gap zero.
    fixture_ratios = (0.5, 0.9, 1.2, 1.0)
    for shape_index in range(1, 6):
        for candidate_index in range(1, 5):
            base = 100.0 * shape_index + 10.0 * candidate_index
            ratio = fixture_ratios[candidate_index - 1]
            gemm_rows[(shape_index, candidate_index)] = {
                "shape_id": f"shape{shape_index}",
                "m": 4096, "n": 4096, "k": 4096, "l": 1,
                "variant": f"variant{candidate_index}",
                "method": "cublaslt" if candidate_index == 4 else "cutedsl",
                "kernel_time_ms": 0.1 * candidate_index,
                "tflops": base * gemm_scale,
                "throughput_ratio_vs_cublaslt": ratio,
                "gap_to_cublaslt_pct": 100.0 * (1.0 - ratio),
                "cache_mode": "hot",
                "is_best_cutedsl": "false",
            }
        best_by_shape[shape_index] = best_variant
    return {
        "campaign_id": campaign_id,
        "git_commit": commit,
        "gpu": {"uuid": gpu_uuid, "name": "SYNTHETIC TEST DEVICE",
                "compute_capability": "10.3", "driver_version": "610.43.02"},
        "provenance": {"git_commit": commit, "gpu_uuid": gpu_uuid,
                       "gpu_name": "SYNTHETIC TEST DEVICE",
                       "compute_capability": "10.3", "cuda_driver_version": "13030",
                       "cuda_runtime_version": "13010"},
        "memory": {"pilot_statistics": pilot_statistics, "pairwise": pairwise,
                   "saturation": saturation, "ncu": ncu},
        "umma": {"configuration": configuration, "scaling": scaling,
                 "saturation": umma_saturation,
                 "profile_validation": {"case_count": P24_PROFILE_CASE_COUNT,
                                        "sm_clock_ok_count": P24_PROFILE_CASE_COUNT,
                                        "sm_clock_statuses_distinct": ["OK"]},
                 "ceiling": {"selected_configuration": {
                                 "method": "umma_1sm", "n": 256, "depth": ceiling_depth,
                                 "case_name": f"23_umma_1sm_n256_d{ceiling_depth}"},
                             "estimated_tflops_per_sm": tflops,
                             "median_flops_per_cycle_per_sm": 8100.0,
                             "sm_clock_valid": True,
                             "device_equivalent_estimate": device}},
        "gemm": {"rows": gemm_rows, "best_cutedsl_variant": best_by_shape,
                 "cache_mode": "hot"},
        "sources": [{"campaign_id": campaign_id, "unit": "P1.4",
                     "artifact": "analysis/pilot_statistics.csv",
                     "repo_relative_path": f"results/raw/x/{campaign_id}/a.csv",
                     "sha256": "0" * 64}],
    }


def _fixture_records(**overrides) -> list[dict]:
    return [_fixture_record(campaign_id, **overrides.get(campaign_id, {}))
            for campaign_id in FINAL_CAMPAIGN_IDS]


# An obviously synthetic analysis-code provenance for the fixtures. It is
# injected; the production path never uses it and there is no flag that selects
# it. It is deliberately not the frozen final execution commit.
_FIXTURE_ANALYSIS_COMMIT = "1234567890abcdef1234567890abcdef12345678"


def _fixture_provenance(**overrides) -> dict:
    provenance = {
        "analysis_code_commit": _FIXTURE_ANALYSIS_COMMIT,
        "analysis_code_tree": "abcdef1234567890abcdef1234567890abcdef12",
        "worktree_clean": True,
        "clean_definition": WORKTREE_CLEAN_DEFINITION,
        "tracked_paths_verified": 64,
        "verification_method": "synthetic fixture provenance",
    }
    provenance.update(overrides)
    return provenance


class _StubP35:
    EXPECTED_SHAPES = ((4096, 4096, 4096, 1), (8192, 8192, 8192, 1), (16384, 512, 4096, 1),
                       (32768, 512, 4096, 1), (512, 16384, 4096, 1))
    EXPECTED_SHAPE_IDS = tuple(f"shape{index + 1}" for index in range(5))
    EXPECTED_SHAPE_COUNT = 5
    EXPECTED_CANDIDATE_COUNT = 4
    EXPECTED_CANDIDATE_ORDER = ("variant1", "variant2", "variant3", "variant4")
    EXPECTED_ROW_COUNT = 20

    @staticmethod
    def validate_serialized_output(text):
        return []


def _self_test_statistics(reporter: _Reporter) -> None:
    summary = summarize_metric([1.0, 2.0, 3.0], metric="median_effective_gbps")
    reporter.check("mean, median, min, and max over three campaign values",
                   (summary["mean"], summary["median"], summary["minimum"],
                    summary["maximum"]) == (2.0, 2.0, 1.0, 3.0), str(summary))
    reporter.check("the sample standard deviation uses the n-1 denominator",
                   abs(summary["stdev_sample"] - 1.0) < 1e-12
                   and abs(summary["stdev_sample"] - math.sqrt(2.0 / 3.0)) > 1e-6,
                   str(summary["stdev_sample"]))
    reporter.check("the coefficient of variation is 100*stdev/mean",
                   abs(summary["cross_campaign_cv_percent"] - 50.0) < 1e-12,
                   str(summary["cross_campaign_cv_percent"]))
    reporter.check("a coefficient of variation above the strict 5% threshold is flagged for "
                   "review and excludes nothing",
                   summary["cross_campaign_cv_review_flag"] == CV_FLAG_REVIEW
                   and summary["campaign_count"] == CAMPAIGN_COUNT
                   and len(summary["campaign_values"]) == CAMPAIGN_COUNT, str(summary))
    calm = summarize_metric([100.0, 100.5, 100.25], metric="median_effective_gbps")
    reporter.check("a low coefficient of variation is labelled ok",
                   calm["cross_campaign_cv_review_flag"] == CV_FLAG_OK, str(calm))
    signed = summarize_metric([-1.0, 0.0, 1.0], metric="gap_to_cublaslt_pct")
    reporter.check("no coefficient of variation is computed for a signed or zero-centred "
                   "metric",
                   signed["cross_campaign_cv_percent"] is None
                   and signed["cross_campaign_cv_review_flag"] == NOT_APPLICABLE,
                   str(signed))
    reporter.check("a negative gap is preserved without clamping",
                   signed["minimum"] == -1.0, str(signed))
    zero = summarize_metric([0.0, 0.0, 0.0], metric="throughput_ratio_vs_cublaslt")
    reporter.check("a zero denominator never produces a coefficient of variation",
                   zero["cross_campaign_cv_percent"] is None, str(zero))
    reporter.rejects("pooling a campaign's internal repetitions is rejected",
                     lambda: summarize_metric([1.0] * 90, metric="median_effective_gbps"),
                     "never pooled")
    reporter.rejects("fewer than three campaign values are rejected",
                     lambda: summarize_metric([1.0, 2.0], metric="median_effective_gbps"),
                     "exactly 3")
    reporter.rejects("a non-finite campaign value is rejected",
                     lambda: summarize_metric([1.0, float("inf"), 3.0],
                                              metric="median_effective_gbps"),
                     "not finite")
    reporter.rejects("a NaN campaign value is rejected",
                     lambda: summarize_metric([1.0, float("nan"), 3.0],
                                              metric="median_effective_gbps"),
                     "not finite")
    agreement = consensus([64, 64, 64], label="x")
    disagreement = consensus([64, 32, 64], label="x")
    reporter.check("a consensus is reported only when all three campaigns agree",
                   agreement["consensus"] == 64 and disagreement["consensus"] is None
                   and disagreement["campaign_values"] == [64, 32, 64], str(disagreement))

    # A ratio must be formed inside each campaign and only then summarized. The
    # fixture below makes the two policies numerically distinguishable.
    numerators, denominators = [10.0, 20.0, 30.0], [1.0, 4.0, 5.0]
    within = [n / d for n, d in zip(numerators, denominators)]
    aggregate_of_ratios = summarize_metric(within, metric="tma_to_ldgsts_ratio")["mean"]
    ratio_of_aggregates = (math.fsum(numerators) / 3) / (math.fsum(denominators) / 3)
    reporter.check("aggregate-of-within-campaign-ratios differs from ratio-of-aggregates, so "
                   "the policy is observable",
                   abs(aggregate_of_ratios - ratio_of_aggregates) > 0.5,
                   f"{aggregate_of_ratios} vs {ratio_of_aggregates}")


def _self_test_parsers(reporter: _Reporter) -> None:
    header = ("method,stages,bytes_in_flight_kib,sample_count,median_gbps,cv_percent,"
              "stability_review")
    good_lines = [header] + [
        f"{method},{stages},{bif},30,{3000 + index}.5,0.1,ok"
        for index, (method, stages, bif) in enumerate(P14_CONFIG_KEYS)]
    good = "\n".join(good_lines) + "\n"
    parsed = parse_p14_pilot_statistics(good, label="fixture")
    reporter.check("a well-formed pilot-statistics table parses to all 18 configurations",
                   len(parsed) == len(P14_CONFIG_KEYS), str(len(parsed)))
    reporter.rejects("a missing row is rejected",
                     lambda: parse_p14_pilot_statistics(
                         "\n".join(good_lines[:-1]) + "\n", label="fixture"),
                     "frozen population")
    reporter.rejects("a duplicated row is rejected",
                     lambda: parse_p14_pilot_statistics(
                         "\n".join(good_lines + [good_lines[-1]]) + "\n", label="fixture"),
                     "frozen population")
    reordered = [good_lines[0], good_lines[2], good_lines[1]] + good_lines[3:]
    reporter.rejects("a reordered row is rejected",
                     lambda: parse_p14_pilot_statistics("\n".join(reordered) + "\n",
                                                        label="fixture"),
                     "frozen order")
    reporter.rejects("a malformed row with the wrong cell count is rejected",
                     lambda: parse_p14_pilot_statistics(
                         "\n".join(good_lines[:-1] + [good_lines[-1] + ",extra"]) + "\n",
                         label="fixture"),
                     "cell(s), expected")
    reporter.rejects("a non-finite measured value is rejected",
                     lambda: parse_p14_pilot_statistics(
                         good.replace("3000.5", "nan"), label="fixture"),
                     "not a finite decimal")
    reporter.rejects("a header missing a consumed field is rejected",
                     lambda: parse_p14_pilot_statistics(
                         good.replace("median_gbps", "median_gbps_renamed"), label="fixture"),
                     "missing required field")
    reporter.rejects("a header that repeats a field name is rejected",
                     lambda: parse_p14_pilot_statistics(
                         good.replace("cv_percent", "median_gbps"), label="fixture"),
                     "repeats a field name")

    ceiling = {
        "status": "ANALYZED", "publishable": False,
        "empirical_per_sm_ceiling_candidate": {
            "method": "umma_1sm", "n": 256, "depth": 256, "case_name": "c",
            "estimated_tflops_per_sm": 16.0, "median_flops_per_cycle_per_sm": 8100.0,
            "sm_clock_valid": True},
        "device_equivalent_estimate": {"available": False, "reason": "unresolved"},
    }
    parsed_ceiling = parse_p24_empirical_ceiling(
        json.dumps(ceiling).encode("utf-8"), label="fixture")
    reporter.check("a terminal empirical-ceiling document parses",
                   parsed_ceiling["estimated_tflops_per_sm"] == 16.0, str(parsed_ceiling))
    for mutation, fragment in (
            ({"status": "INCONCLUSIVE"}, "not 'ANALYZED'"),
            ({"publishable": True}, "publishable is not false"),
    ):
        broken = dict(ceiling)
        broken.update(mutation)
        reporter.rejects(f"an empirical-ceiling document with {mutation} is rejected",
                         lambda payload=json.dumps(broken).encode("utf-8"):
                         parse_p24_empirical_ceiling(payload, label="fixture"), fragment)
    untrusted = json.loads(json.dumps(ceiling))
    untrusted["empirical_per_sm_ceiling_candidate"]["sm_clock_valid"] = False
    reporter.rejects("an untrustworthy SM-clock conversion is rejected",
                     lambda: parse_p24_empirical_ceiling(
                         json.dumps(untrusted).encode("utf-8"), label="fixture"),
                     "not marked valid")


def _self_test_population(reporter: _Reporter) -> None:
    finals = list(FINAL_CAMPAIGN_IDS)
    validate_declared_population([PILOT_CAMPAIGN_ID], finals)
    reporter.check("the frozen declared population is accepted", True)
    reporter.rejects("a missing final campaign ID is rejected",
                     lambda: validate_declared_population([PILOT_CAMPAIGN_ID], finals[:2]),
                     "exactly 3 final campaign IDs")
    reporter.rejects("a fourth final campaign ID is rejected",
                     lambda: validate_declared_population(
                         [PILOT_CAMPAIGN_ID], finals + ["20260818T000000Z"]),
                     "exactly 3 final campaign IDs")
    reporter.rejects("a duplicated final campaign ID is rejected",
                     lambda: validate_declared_population(
                         [PILOT_CAMPAIGN_ID], [finals[0], finals[0], finals[1]]),
                     "repeat")
    reporter.rejects("a reordered final population is rejected",
                     lambda: validate_declared_population(
                         [PILOT_CAMPAIGN_ID], [finals[1], finals[0], finals[2]]),
                     "frozen order")
    reporter.rejects("a substituted final campaign ID is rejected",
                     lambda: validate_declared_population(
                         [PILOT_CAMPAIGN_ID], [finals[0], finals[1], "20260818T000000Z"]),
                     "frozen population")
    reporter.rejects("the accepted pilot may never be one of the replicates",
                     lambda: validate_declared_population(
                         [PILOT_CAMPAIGN_ID], [finals[0], finals[1], PILOT_CAMPAIGN_ID]),
                     "never enter a statistic")
    reporter.rejects("a substituted pilot ID is rejected",
                     lambda: validate_declared_population(["20260818T000000Z"], finals),
                     "accepted pilot")

    compare_campaign_provenance(_fixture_records())
    reporter.check("three campaigns with one commit and one device are accepted", True)
    reporter.rejects("mixed final execution commits are rejected",
                     lambda: compare_campaign_provenance(_fixture_records(**{
                         FINAL_CAMPAIGN_IDS[2]: {"commit": "0" * 40}})),
                     "one execution commit")
    reporter.rejects("mixed GPU provenance is rejected",
                     lambda: compare_campaign_provenance(_fixture_records(**{
                         FINAL_CAMPAIGN_IDS[1]: {"gpu_uuid": "GPU-99999999-8888-7777-6666-"
                                                             "555555555555"}})),
                     "one GPU uuid")


def _self_test_aggregation(reporter: _Reporter, orchestrator) -> None:
    p35 = _StubP35()
    records = _fixture_records()
    provenance = _fixture_provenance()
    documents = build_documents(orchestrator, p35, records, provenance)
    inventory = [relative for relative, _ in documents]
    reporter.check("the analysis produces exactly the frozen artifact inventory",
                   inventory == list(ARTIFACT_RELATIVE_PATHS), str(inventory))
    again = build_documents(orchestrator, p35, _fixture_records(), _fixture_provenance())
    reporter.check("two independent runs over identical evidence are byte-identical",
                   [payload for _, payload in documents]
                   == [payload for _, payload in again], "")

    summary = json.loads(dict(documents)["integrated_summary.json"].decode("utf-8"))
    reporter.check("the pilot is recorded only as excluded qualification provenance",
                   summary["population"]["pilot_campaign_id_excluded"] == PILOT_CAMPAIGN_ID
                   and PILOT_CAMPAIGN_ID not in summary["population"]["final_campaign_ids"],
                   "")
    reporter.check("the candidate bundle records publishable=false and the candidate "
                   "publication state",
                   summary["publishable"] is False
                   and summary["publication_state"] == PUBLICATION_STATE, "")
    text = dict(documents)["memory_paths.csv"].decode("utf-8")
    reporter.check("no output artifact mentions the excluded pilot as data",
                   PILOT_CAMPAIGN_ID not in text, "")
    reporter.check("the memory table carries the value row plus every preserved diagnostic "
                   "row for each frozen configuration, ratio pair, saturation group, and "
                   "profiled case",
                   len(text.strip().split("\n")) == 1 + 5 * len(P14_CONFIG_KEYS)
                   + len(P14_PAIR_KEYS) + len(P14_SATURATION_KEYS)
                   + 3 * len(P14_NCU_CASES), str(len(text.strip().split("\n"))))

    gap_rows = [row for row in csv.DictReader(io.StringIO(
        dict(documents)["gemm_comparison.csv"].decode("utf-8")))
        if row["metric"] == "gap_to_cublaslt_pct"]
    reporter.check("the signed GEMM gap never carries a coefficient of variation",
                   all(row["cross_campaign_cv_percent"] == NOT_APPLICABLE
                       for row in gap_rows), "")
    reporter.check("negative GEMM gaps are preserved unclamped",
                   any(float(row["campaign_1_value"]) < 0 for row in gap_rows), "")

    # Disagreements are reported, never resolved.
    mixed = build_documents(orchestrator, p35, _fixture_records(**{
        FINAL_CAMPAIGN_IDS[1]: {"saturation_kib": 32, "depth_saturation": 64,
                                "ceiling_depth": 64, "best_variant": "persistent_1cta"}}),
        _fixture_provenance())
    mixed_summary = json.loads(dict(mixed)["integrated_summary.json"].decode("utf-8"))
    reporter.check("a disagreeing memory saturation candidate produces no consensus",
                   mixed_summary["experiment_1_memory_paths"][
                       "saturation_consensus_available"] is False, "")
    reporter.check("a disagreeing depth saturation produces no consensus",
                   mixed_summary["experiment_2_umma_throughput"][
                       "depth_saturation_consensus_available"] is False, "")
    ceiling = mixed_summary["experiment_2_umma_throughput"]["empirical_per_sm_ceiling"]
    reporter.check("an unstable ceiling selection suppresses the cross-campaign TFLOP/s "
                   "statistic and keeps every campaign's own selection",
                   ceiling["selection_stable_across_campaigns"] is False
                   and ceiling["estimated_tflops_per_sm"]["statistics"] == NOT_APPLICABLE
                   and len(ceiling["campaign_selections"]) == CAMPAIGN_COUNT, "")
    shapes = mixed_summary["experiment_3_gemm_comparison"]["shapes"]
    reporter.check("a disagreeing best CuTe DSL variant is reported as unstable, not resolved",
                   all(shape["best_cutedsl_variant"]["stable_across_campaigns"] is False
                       and shape["best_cutedsl_variant"]["stable_best_cutedsl_variant"] is None
                       for shape in shapes), "")

    # Device-wide estimates.
    unavailable = json.loads(dict(documents)["integrated_summary.json"].decode("utf-8"))[
        "experiment_2_umma_throughput"]["device_equivalent_estimate"]
    reporter.check("an unavailable device-wide estimate is structured and carries the exact "
                   "per-campaign reason",
                   unavailable["available"] is False
                   and len(unavailable["per_campaign"]) == CAMPAIGN_COUNT
                   and all(entry["reason"] for entry in unavailable["per_campaign"]), "")
    inconsistent = aggregate_device_equivalent(_fixture_records(**{
        campaign_id: {"device_available": True,
                      "sm_count": 132 if index < 2 else 148}
        for index, campaign_id in enumerate(FINAL_CAMPAIGN_IDS)}))
    reporter.check("disagreeing SM counts leave the device-wide estimate unavailable",
                   inconsistent["available"] is False
                   and "do not agree" in inconsistent["reason"], str(inconsistent))
    partial = aggregate_device_equivalent(_fixture_records(**{
        FINAL_CAMPAIGN_IDS[0]: {"device_available": True, "sm_count": 132},
        FINAL_CAMPAIGN_IDS[1]: {"device_available": True, "sm_count": 132}}))
    reporter.check("one campaign without a valid SM count leaves the estimate unavailable",
                   partial["available"] is False, str(partial))
    available = aggregate_device_equivalent(_fixture_records(**{
        campaign_id: {"device_available": True, "sm_count": 132}
        for campaign_id in FINAL_CAMPAIGN_IDS}))
    reporter.check("three agreeing validated SM counts do enable the modeled estimate",
                   available["available"] is True
                   and available["multiprocessor_count"] == 132, str(available))

    # High variability flags but never excludes.
    noisy = build_documents(orchestrator, p35, _fixture_records(**{
        FINAL_CAMPAIGN_IDS[2]: {"gbps_scale": 1.5}}), _fixture_provenance())
    noisy_summary = json.loads(dict(noisy)["integrated_summary.json"].decode("utf-8"))
    flagged = [entry for entry in noisy_summary["experiment_1_memory_paths"]["configurations"]
               if entry["cross_campaign_cv_review_flag"] == CV_FLAG_REVIEW]
    reporter.check("high cross-campaign variability is flagged for review and still keeps "
                   "all three campaign values",
                   flagged and all(len(entry["campaign_values"]) == CAMPAIGN_COUNT
                                   for entry in flagged), str(len(flagged)))

    reporter.rejects("a record set that is not the frozen final population is rejected",
                     lambda: build_documents(orchestrator, p35, records[:2],
                                             _fixture_provenance()),
                     "exactly 3 final campaign records")
    reporter.rejects("records in the wrong order are rejected",
                     lambda: build_documents(orchestrator, p35,
                                             [records[1], records[0], records[2]],
                                             _fixture_provenance()),
                     "frozen final population in order")
    pilot_record = _fixture_record(PILOT_CAMPAIGN_ID)
    reporter.rejects("the accepted pilot's record is never aggregated",
                     lambda: build_documents(orchestrator, p35,
                                             [pilot_record] + records[1:],
                                             _fixture_provenance()),
                     "frozen final population in order")

    _self_test_taxonomy(reporter, documents, summary)
    _self_test_diagnostics(reporter, orchestrator, p35)
    _self_test_metadata_contract(reporter, orchestrator, documents, summary)
    _self_test_analysis_provenance(reporter, orchestrator, p35, records)
    _self_test_acceptance(reporter, orchestrator, documents)


# --- 3.1 The scientific evidence taxonomy ----------------------------------

# The exact wordings the first independent audit rejected. Each one claimed a
# derived, modeled, or cross-campaign quantity as a direct measurement, or named
# a whisker a bar. They are matched as whole phrases, not as loose substrings,
# precisely so that a correct *negation* ("not a directly measured quantity")
# never trips the scan that bans the assertion.
FORBIDDEN_NARRATIVE_PHRASES = (
    "sustained HBM-to-SMEM bandwidth of the LDGSTS and TMA paths",
    "this is a measured ratio",
    "a measured throughput difference",
    "measured HBM-to-SMEM bandwidth",
    "measured FLOP/cycle",
    "means TMA measured higher sustained bandwidth",
    "campaign-level median effective bandwidth",
    "measured: both equivalent HBM-to-SMEM paths sustain the bandwidths",
    "measured: the clock-independent FLOP/cycle",
    "measured: per shape and candidate",
    "### directly measured",
    "deterministic derived quantities",
    "the bar is min",
    "min..max bar",
    "the bar is min..max",
)
# Phrases the corrected narrative must carry.
REQUIRED_NARRATIVE_PHRASES = (
    "timing-derived effective transfer rate",
    "not HBM/DRAM bandwidth",
    "min-max whisker",
    "operation-and-cycle-derived",
    "modeled clock conversion",
    "unavailable from the collected evidence",
)


def _self_test_taxonomy(reporter: _Reporter, documents: list[tuple[str, bytes]],
                        summary: dict) -> None:
    """Adversarial coverage for the corrected evidence taxonomy.

    Every check here fails on the pre-remediation wording, which classified
    derived rates, derived ratios, and derived FLOP/cycle values as directly
    measured quantities."""
    payloads = dict(documents)
    blob = b"\n".join(payload for _, payload in documents).decode("utf-8")
    for phrase in FORBIDDEN_NARRATIVE_PHRASES:
        reporter.check(f"no artifact presents a derived quantity as {phrase!r}",
                       phrase.lower() not in blob.lower(), phrase)
    for phrase in REQUIRED_NARRATIVE_PHRASES:
        reporter.check(f"the corrected narrative states {phrase!r}",
                       phrase.lower() in blob.lower(), phrase)

    interpretation = summary["integrated_interpretation"]
    reporter.check("the interpretation carries the six-category taxonomy and no "
                   "'directly_measured' bucket",
                   "directly_measured" not in interpretation
                   and "deterministic_derived" not in interpretation
                   and all(key in interpretation for key in (
                       "measured_source_observations",
                       "within_campaign_derived_estimates",
                       "cross_campaign_descriptive_statistics",
                       "modeled_estimates", "interpretations",
                       "unavailable_from_the_collected_evidence")),
                   str(sorted(interpretation)))

    classification = summary["evidence_taxonomy"]["metric_classification"]
    for metric, expected in (
            ("median_effective_gbps", EVIDENCE_WITHIN_CAMPAIGN),
            ("dram_read_ratio", EVIDENCE_WITHIN_CAMPAIGN),
            ("hbm_classification", EVIDENCE_WITHIN_CAMPAIGN),
            ("median_flops_per_cycle", EVIDENCE_WITHIN_CAMPAIGN),
            ("median_flops_per_cycle_per_sm", EVIDENCE_WITHIN_CAMPAIGN),
            ("tma_to_ldgsts_ratio", EVIDENCE_WITHIN_CAMPAIGN),
            ("speedup_2sm_over_1sm", EVIDENCE_WITHIN_CAMPAIGN),
            ("scaling_efficiency_percent", EVIDENCE_WITHIN_CAMPAIGN),
            ("tflops", EVIDENCE_WITHIN_CAMPAIGN),
            ("throughput_ratio_vs_cublaslt", EVIDENCE_WITHIN_CAMPAIGN),
            ("gap_to_cublaslt_pct", EVIDENCE_WITHIN_CAMPAIGN),
            ("best_cutedsl_variant", EVIDENCE_WITHIN_CAMPAIGN),
            ("estimated_tflops_per_sm", EVIDENCE_MODELED),
            ("estimated_device_equivalent_tflops", EVIDENCE_MODELED),
            ("kernel_time_ms", EVIDENCE_MEASURED)):
        reporter.check(f"{metric} is classified {expected}",
                       classification[metric]["evidence_class"] == expected,
                       str(classification.get(metric)))
    reporter.check("effective GB/s is never presented as actual HBM bandwidth",
                   "not hbm/dram bandwidth"
                   in classification["median_effective_gbps"]["basis"].lower(),
                   classification["median_effective_gbps"]["basis"])
    reporter.check("FLOP/cycle is never presented as directly measured",
                   "measured %clock64" in classification["median_flops_per_cycle"]["basis"]
                   and classification["median_flops_per_cycle"]["evidence_class"]
                   != EVIDENCE_MEASURED, "")

    for name, fields in (("memory_paths.csv", MEMORY_CSV_FIELDS),
                         ("umma_throughput.csv", UMMA_CSV_FIELDS),
                         ("gemm_comparison.csv", GEMM_CSV_FIELDS)):
        rows = list(csv.DictReader(io.StringIO(payloads[name].decode("utf-8"))))
        reporter.check(f"{name} classifies every row's campaign value",
                       rows and all(row["evidence_class"] in EVIDENCE_CLASSES
                                    for row in rows),
                       str({row["metric"] for row in rows
                            if row["evidence_class"] not in EVIDENCE_CLASSES}))
        reporter.check(f"{name} names its cross-campaign statistics unambiguously",
                       "cross_campaign_cv_percent" in fields
                       and "cross_campaign_cv_review_flag" in fields
                       and "cv_percent" not in fields, str(fields))
    memory_rows = list(csv.DictReader(io.StringIO(
        payloads["memory_paths.csv"].decode("utf-8"))))
    unprofiled = [row for row in memory_rows
                  if row["metric"] == "ncu_coverage"
                  and row["campaign_1_value"] == NCU_NOT_PROFILED]
    reporter.check("the twelve HBM-unvalidated configurations are marked not_profiled",
                   len(unprofiled) == len(P14_CONFIG_KEYS) - len(P14_NCU_CASES),
                   str(len(unprofiled)))
    report_text = payloads["report.md"].decode("utf-8")
    reporter.check("the report states that actual HBM traffic is unavailable for the "
                   "unprofiled configurations",
                   "actual HBM/DRAM traffic is unavailable" in report_text, "")
    reporter.check("no figure calls the min-max whisker a bar",
                   all(re.search(r"\bbars?\b", payloads[name].decode("utf-8")) is None
                       for name in ARTIFACT_RELATIVE_PATHS
                       if name.endswith(".svg")), "")
    reporter.check("every figure names the min-max range a whisker",
                   all("min-max whisker" in payloads[name].decode("utf-8")
                       for name in ARTIFACT_RELATIVE_PATHS
                       if name.endswith(".svg")), "")
    reporter.check("every figure states that it summarizes exactly three campaign-level "
                   "values",
                   all(f"exactly {CAMPAIGN_COUNT} campaign-level values".encode("utf-8")
                       in payloads[name] for name in ARTIFACT_RELATIVE_PATHS
                       if name.endswith(".svg")), "")
    reporter.rejects("an unclassified metric can never be emitted",
                     lambda: evidence_class_for("an_undeclared_metric"),
                     "no quantity is ever emitted unclassified")


# --- 3.2 Preserved diagnostics and within-campaign stability ----------------


def _self_test_diagnostics(reporter: _Reporter, orchestrator, p35) -> None:
    """Different campaigns carrying different diagnostics, one of them a
    READ_AMPLIFICATION, and a within/cross-campaign disagreement. Every value
    must survive in the correct campaign position."""
    records = _fixture_records(**{
        FINAL_CAMPAIGN_IDS[0]: {"ncu_diagnostic_flags": "",
                                "memory_stability_review": "ok",
                                "memory_cv_percent": 0.10,
                                "umma_stability_review": "ok",
                                "surprising_value_flag": "False"},
        FINAL_CAMPAIGN_IDS[1]: {"ncu_diagnostic_flags": "READ_AMPLIFICATION",
                                "memory_stability_review": "REVIEW",
                                "memory_cv_percent": 7.25,
                                "umma_stability_review": "REVIEW",
                                "surprising_value_flag": "True"},
        FINAL_CAMPAIGN_IDS[2]: {"ncu_diagnostic_flags": "SOME_OTHER_FLAG",
                                "memory_stability_review": "ok",
                                "memory_cv_percent": 0.30,
                                "umma_stability_review": "ok",
                                "surprising_value_flag": "False"},
    })
    documents = build_documents(orchestrator, p35, records, _fixture_provenance())
    payloads = dict(documents)
    summary = json.loads(payloads["integrated_summary.json"].decode("utf-8"))
    memory_rows = list(csv.DictReader(io.StringIO(
        payloads["memory_paths.csv"].decode("utf-8"))))

    flag_rows = [row for row in memory_rows if row["metric"] == "diagnostic_flags"]
    reporter.check("every profiled case keeps one diagnostic-flags row",
                   len(flag_rows) == len(P14_NCU_CASES), str(len(flag_rows)))
    reporter.check("NCU diagnostic flags survive in the correct frozen campaign order",
                   all((row["campaign_1_value"], row["campaign_2_value"],
                        row["campaign_3_value"])
                       == (NOT_APPLICABLE, "READ_AMPLIFICATION", "SOME_OTHER_FLAG")
                       for row in flag_rows),
                   str(flag_rows[:1]))
    reporter.check("READ_AMPLIFICATION is never dropped between parsing and publication",
                   "READ_AMPLIFICATION" in payloads["memory_paths.csv"].decode("utf-8")
                   and "READ_AMPLIFICATION"
                   in payloads["integrated_summary.json"].decode("utf-8")
                   and "READ_AMPLIFICATION" in payloads["report.md"].decode("utf-8"), "")
    classification_rows = [row for row in memory_rows
                           if row["metric"] == "hbm_classification"]
    reporter.check("every profiled case keeps its per-campaign HBM classification",
                   len(classification_rows) == len(P14_NCU_CASES)
                   and all(row["campaign_2_value"] == "HBM_VALIDATED"
                           for row in classification_rows), "")

    within_rows = [row for row in memory_rows
                   if row["metric"] == "within_campaign_stability_review"]
    reporter.check("P1.4's within-campaign stability review is preserved per campaign",
                   len(within_rows) == len(P14_CONFIG_KEYS)
                   and all((row["campaign_1_value"], row["campaign_2_value"],
                            row["campaign_3_value"]) == ("ok", "REVIEW", "ok")
                           for row in within_rows), "")
    sample_rows = [row for row in memory_rows
                   if row["metric"] == "within_campaign_sample_count"]
    reporter.check("P1.4's within-campaign sample count is preserved per campaign",
                   len(sample_rows) == len(P14_CONFIG_KEYS)
                   and all(row["campaign_1_value"] == "30" for row in sample_rows), "")
    cv_rows = [row for row in memory_rows if row["metric"] == "within_campaign_cv_percent"]
    reporter.check("P1.4's within-campaign CV is preserved per campaign and never carries "
                   "a cross-campaign statistic",
                   len(cv_rows) == len(P14_CONFIG_KEYS)
                   and all(row["campaign_2_value"] == "7.25"
                           and row["cross_campaign_cv_percent"] == NOT_APPLICABLE
                           for row in cv_rows), "")

    umma_rows = list(csv.DictReader(io.StringIO(
        payloads["umma_throughput.csv"].decode("utf-8"))))
    for metric, expected in (("within_campaign_sample_count", "30"),
                             ("within_campaign_cv_percent", "0.01"),
                             ("within_campaign_stability_review", "REVIEW")):
        rows = [row for row in umma_rows if row["metric"] == metric]
        reporter.check(f"P2.4's {metric} is preserved per campaign",
                       len(rows) == len(P24_CONFIG_KEYS)
                       and all(row["campaign_2_value"] == expected for row in rows),
                       str(rows[:1]))
    surprising = [row for row in umma_rows if row["metric"] == "surprising_value_flag"]
    reporter.check("P2.4's surprising-value flag is preserved per campaign",
                   len(surprising) == len(P24_SCALING_KEYS)
                   and all((row["campaign_1_value"], row["campaign_2_value"],
                            row["campaign_3_value"]) == ("False", "True", "False")
                           for row in surprising), "")

    warnings = summary["source_diagnostic_warnings"]
    kinds = {entry["section"] for entry in warnings}
    reporter.check("every non-empty source diagnostic reaches the machine-readable warning "
                   "list",
                   kinds == {"experiment_1_configuration", "experiment_1_ncu_validation",
                             "experiment_2_configuration", "experiment_2_scaling"},
                   str(sorted(kinds)))
    reporter.check("each warning names the campaign it belongs to",
                   all(entry["campaign_id"] in FINAL_CAMPAIGN_IDS for entry in warnings), "")
    report_text = payloads["report.md"].decode("utf-8")
    reporter.check("the report summarizes every source diagnostic warning",
                   f"Source diagnostic warnings ({len(warnings)})" in report_text
                   and "SOME_OTHER_FLAG" in report_text, "")

    # Within-campaign and cross-campaign statuses disagree: campaign 2 flags a
    # within-campaign REVIEW while the three campaign-level medians are almost
    # identical, so the cross-campaign CV is calm. Neither replaces the other.
    configuration = summary["experiment_1_memory_paths"]["configurations"][0]
    reporter.check("a within-campaign REVIEW does not become a cross-campaign REVIEW",
                   configuration["cross_campaign_cv_review_flag"] == CV_FLAG_OK
                   and configuration["within_campaign_diagnostics"][
                       "within_campaign_stability_review"]["campaign_values"]
                   == ["ok", "REVIEW", "ok"], str(configuration["cross_campaign_cv_review_flag"]))
    reporter.check("a within-campaign flag never removed a campaign or changed a value",
                   configuration["campaign_count"] == CAMPAIGN_COUNT
                   and len(configuration["campaign_values"]) == CAMPAIGN_COUNT, "")

    # The opposite disagreement: calm within-campaign reviews, loud
    # cross-campaign variability.
    noisy = build_documents(orchestrator, p35, _fixture_records(**{
        FINAL_CAMPAIGN_IDS[2]: {"gbps_scale": 1.5}}), _fixture_provenance())
    noisy_summary = json.loads(dict(noisy)["integrated_summary.json"].decode("utf-8"))
    entry = noisy_summary["experiment_1_memory_paths"]["configurations"][0]
    reviews = noisy_summary["cross_campaign_review_conditions"]
    reporter.check("a cross-campaign REVIEW coexists with calm within-campaign reviews and "
                   "removes nothing",
                   entry["cross_campaign_cv_review_flag"] == CV_FLAG_REVIEW
                   and entry["within_campaign_diagnostics"][
                       "within_campaign_stability_review"]["campaign_values"]
                   == ["ok", "ok", "ok"]
                   and len(entry["campaign_values"]) == CAMPAIGN_COUNT, "")
    reporter.check("every cross-campaign review condition is listed and marked diagnostic "
                   "only",
                   reviews and all("review diagnostic only" in item["effect"]
                                   for item in reviews), str(len(reviews)))
    reporter.check("the report summarizes the cross-campaign review conditions",
                   f"Cross-campaign variability review conditions ({len(reviews)})"
                   in dict(noisy)["report.md"].decode("utf-8"), "")


# --- 3.3 The metadata ownership contract -----------------------------------


def _self_test_metadata_contract(reporter: _Reporter, orchestrator,
                                 documents: list[tuple[str, bytes]], summary: dict) -> None:
    """The manifest is the authoritative envelope; the siblings are not."""
    payloads = dict(documents)
    reporter.check("exactly nine artifacts are generated",
                   len(documents) == 9
                   and [relative for relative, _ in documents]
                   == list(ARTIFACT_RELATIVE_PATHS), str(len(documents)))
    manifest = json.loads(payloads[MANIFEST_RELATIVE_PATH].decode("utf-8"))
    siblings = [relative for relative in ARTIFACT_RELATIVE_PATHS
                if relative != MANIFEST_RELATIVE_PATH]
    reporter.check("the manifest binds all eight siblings",
                   sorted(manifest["artifact_sha256"]) == sorted(siblings),
                   str(sorted(manifest["artifact_sha256"])))
    reporter.check("every recomputed sibling hash matches the manifest",
                   all(manifest["artifact_sha256"][relative]
                       == orchestrator.sha256_bytes(payloads[relative])
                       for relative in siblings), "")
    reporter.check("the manifest is reproduced byte for byte from the same evidence",
                   payloads[MANIFEST_RELATIVE_PATH]
                   == dict(build_documents(orchestrator, _StubP35(), _fixture_records(),
                                           _fixture_provenance()))[MANIFEST_RELATIVE_PATH],
                   "")
    reporter.check("every campaign value column has an explicit ordered campaign ID mapping",
                   manifest["campaign_value_column_map"]
                   == {f"campaign_{index + 1}_value": campaign_id
                       for index, campaign_id in enumerate(FINAL_CAMPAIGN_IDS)},
                   str(manifest.get("campaign_value_column_map")))
    reporter.check("the manifest states precisely that it cannot contain its own byte hash",
                   manifest["self_hash"]["value"] == NOT_APPLICABLE
                   and "cannot contain its own byte hash" in manifest["self_hash"]["reason"]
                   and ACCEPTANCE_RELATIVE_PATH in manifest["self_hash"]["bound_by"], "")
    for field in ("final_execution_commit", "analysis_code_commit",
                  "analysis_code_worktree_clean", "gpu", "comparable_provenance",
                  "sources", "artifact_inventory", "publishable", "publication_state",
                  "pilot_campaign_id_excluded", "pilot_role"):
        reporter.check(f"the manifest records {field}", field in manifest, "")
    reporter.check("the manifest distinguishes the execution commit from the analysis-code "
                   "commit",
                   manifest["final_execution_commit"] != manifest["analysis_code_commit"],
                   "")
    reporter.check("the manifest records the candidate publication state",
                   manifest["publishable"] is False
                   and manifest["publication_state"] == PUBLICATION_STATE
                   and manifest["acceptance"]["exists"] is False, "")

    # Metadata ownership: the detached data artifacts deliberately do NOT
    # duplicate the global provenance, and the documentation says so.
    for relative in ("memory_paths.csv", "umma_throughput.csv", "gemm_comparison.csv",
                     "figures/memory_paths.svg", "figures/umma_throughput.svg",
                     "figures/gemm_comparison.svg"):
        text = payloads[relative].decode("utf-8")
        reporter.check(f"{relative} is not claimed to embed the campaign IDs",
                       all(campaign_id not in text for campaign_id in FINAL_CAMPAIGN_IDS),
                       relative)
    reporter.check("the two context documents carry the scientific context needed to read "
                   "the bundle",
                   all(campaign_id in payloads["integrated_summary.json"].decode("utf-8")
                       and campaign_id in payloads["report.md"].decode("utf-8")
                       for campaign_id in FINAL_CAMPAIGN_IDS), "")
    reporter.check("the bundle documents that a detached CSV or SVG is not a standalone "
                   "provenance envelope",
                   "not a standalone provenance envelope"
                   in payloads["report.md"].decode("utf-8")
                   and "not a standalone provenance envelope"
                   in json.dumps(summary), "")
    reporter.rejects("a manifest that does not bind all eight siblings is rejected",
                     lambda: build_manifest(orchestrator, _fixture_records(), [],
                                            _fixture_provenance(),
                                            [("memory_paths.csv", b"x")]),
                     "exactly the eight non-manifest artifacts")


# --- 3.6 The analysis-code commit ------------------------------------------


def _self_test_analysis_provenance(reporter: _Reporter, orchestrator, p35,
                                   records: list[dict]) -> None:
    reporter.rejects("a candidate is never built without analysis-code provenance",
                     lambda: build_documents(orchestrator, p35, records, {}),
                     "has no 'analysis_code_commit'")
    reporter.rejects("a short or abbreviated analysis-code commit is rejected",
                     lambda: build_documents(orchestrator, p35, records,
                                             _fixture_provenance(
                                                 analysis_code_commit="1234567")),
                     "not a full 40-character")
    reporter.rejects("a dirty analysis-code worktree is rejected",
                     lambda: build_documents(orchestrator, p35, records,
                                             _fixture_provenance(worktree_clean=False)),
                     "not verified clean")
    reporter.rejects("the analysis-code commit is never the final execution commit",
                     lambda: build_documents(
                         orchestrator, p35, records,
                         _fixture_provenance(analysis_code_commit=FINAL_EXECUTION_COMMIT)),
                     "did not exist at that commit")


# --- 3.5 The external acceptance attestation -------------------------------


def _self_test_acceptance(reporter: _Reporter, orchestrator,
                          documents: list[tuple[str, bytes]]) -> None:
    payloads = dict(documents)
    manifest_sha256 = orchestrator.sha256_bytes(payloads[MANIFEST_RELATIVE_PATH])
    artifact_sha256 = {relative: orchestrator.sha256_bytes(payload)
                       for relative, payload in documents
                       if relative != MANIFEST_RELATIVE_PATH}
    commit = json.loads(
        payloads[MANIFEST_RELATIVE_PATH].decode("utf-8"))["analysis_code_commit"]
    template = build_acceptance_template(manifest_sha256, artifact_sha256, commit)

    def errors(mutate=None) -> list[str]:
        document = json.loads(json.dumps(template))
        if mutate is not None:
            mutate(document)
        return validate_acceptance_document(
            document, manifest_sha256=manifest_sha256,
            artifact_sha256=artifact_sha256, analysis_code_commit=commit)

    reporter.check("a correct future acceptance attestation validates", not errors(),
                   str(errors()))
    reporter.check("the attestation binds the manifest hash and therefore all nine "
                   "artifacts without self-reference",
                   template["analysis_manifest_sha256"] == manifest_sha256
                   and sorted(template["artifact_sha256"])
                   == sorted(ARTIFACT_RELATIVE_PATHS)
                   and template["artifact_sha256"][MANIFEST_RELATIVE_PATH]
                   == manifest_sha256, "")
    reporter.check("the frozen acceptance schema version is p43.acceptance.v1",
                   template["schema_version"] == ACCEPTANCE_SCHEMA_VERSION
                   and ACCEPTANCE_SCHEMA_VERSION == "p43.acceptance.v1", "")
    reporter.check("a malformed acceptance attestation is rejected",
                   validate_acceptance_document(
                       ["not", "an", "object"], manifest_sha256=manifest_sha256,
                       artifact_sha256=artifact_sha256, analysis_code_commit=commit), "")
    for label, mutate, fragment in (
        ("a missing required field", lambda doc: doc.pop("analysis_manifest_sha256"),
         "has no 'analysis_manifest_sha256'"),
        ("one wrong artifact hash",
         lambda doc: doc["artifact_sha256"].__setitem__("report.md", "0" * 64),
         "does not match the reviewed bytes"),
        ("a wrong manifest hash",
         lambda doc: doc.__setitem__("analysis_manifest_sha256", "0" * 64),
         "binds a different bundle"),
        ("a different analyzer commit",
         lambda doc: doc.__setitem__("analysis_code_commit", "f" * 40),
         "never transfers to a different analyzer commit"),
        ("an incomplete artifact inventory",
         lambda doc: doc["artifact_sha256"].pop("report.md"),
         "must cover exactly the"),
        ("a substituted campaign population",
         lambda doc: doc.__setitem__("final_campaign_ids",
                                     [FINAL_CAMPAIGN_IDS[0]] * CAMPAIGN_COUNT),
         "not the frozen population"),
        ("the pilot promoted into the population",
         lambda doc: doc.__setitem__("pilot_campaign_id_excluded", FINAL_CAMPAIGN_IDS[0]),
         "not the accepted pilot"),
        ("a status other than ACCEPTED", lambda doc: doc.__setitem__("status", "PENDING"),
         "expected 'ACCEPTED'"),
        ("accepted_for_publication that is not exactly true",
         lambda doc: doc.__setitem__("accepted_for_publication", "yes"),
         "must be exactly true"),
        ("a wrong schema version",
         lambda doc: doc.__setitem__("schema_version", "p43.acceptance.v2"),
         "expected 'p43.acceptance.v1'"),
        ("an unverified bundle",
         lambda doc: doc.__setitem__("verification_outcome", "skipped"),
         "verification_outcome="),
        ("an unreviewed bundle",
         lambda doc: doc.__setitem__("independent_output_review_outcome", "skipped"),
         "independent_output_review_outcome="),
    ):
        found = errors(mutate)
        reporter.check(f"an acceptance attestation with {label} is rejected",
                       any(fragment in item for item in found), str(found))
    reporter.check("the acceptance attestation is not part of the analysis inventory and "
                   "is never produced by the analyzer",
                   ACCEPTANCE_RELATIVE_PATH not in ARTIFACT_RELATIVE_PATHS
                   and isinstance(build_acceptance_template(
                       manifest_sha256, artifact_sha256, commit), dict), "")
    reporter.check("the frozen acceptance lifecycle keeps the external review before the "
                   "attestation",
                   [step for step in ACCEPTANCE_LIFECYCLE
                    if "review" in step].index(
                        [step for step in ACCEPTANCE_LIFECYCLE if "review" in step][0])
                   >= 0
                   and ACCEPTANCE_LIFECYCLE.index(
                       f"an external acceptance attestation at "
                       f"{ACCEPTANCE_RELATIVE_PATH}") == len(ACCEPTANCE_LIFECYCLE) - 2,
                   str(ACCEPTANCE_LIFECYCLE))


# --- 3.6 The pure-Python Git provenance reader -----------------------------
#
# The fixtures below build a real, minimal `.git` by hand: loose objects, a
# ref, and a version-2 index with a cache-tree extension. No `git` binary and
# no child process is involved, in the fixture or in the reader. The expected
# object ids are the ones Git itself computes for this content, so a bug that
# happened to be symmetric between the fixture writer and the reader would
# still be caught by the literal id assertions.


def _git_fixture_object(payload: bytes, kind: bytes) -> tuple[str, bytes]:
    body = kind + b" " + str(len(payload)).encode("ascii") + b"\0" + payload
    return hashlib.sha1(body).hexdigest(), body


def _git_fixture_write_object(root: Path, payload: bytes, kind: bytes) -> str:
    oid, body = _git_fixture_object(payload, kind)
    directory = root / ".git" / "objects" / oid[:2]
    directory.mkdir(parents=True, exist_ok=True)
    (directory / oid[2:]).write_bytes(zlib.compress(body))
    return oid


def _git_fixture_index(entries: list[tuple[str, str, int]], root_tree: str | None) -> bytes:
    """A version-2 index over (path, blob oid, mode), optionally with a valid
    root cache-tree entry."""
    body = struct.pack(">4sII", GIT_INDEX_SIGNATURE, 2, len(entries))
    for path, oid, mode in entries:
        entry = struct.pack(">10I", 0, 0, 0, 0, 0, 0, mode, 0, 0, 0)
        entry += bytes.fromhex(oid)
        raw = path.encode("utf-8")
        entry += struct.pack(">H", min(len(raw), 0x0FFF))
        entry += raw + b"\0"
        # Git pads each entry to a multiple of eight bytes measured from the
        # entry's own start, never from the file offset, and always keeps at
        # least one NUL terminator.
        body += entry + b"\0" * ((8 - (len(entry) % 8)) % 8)
    if root_tree is not None:
        payload = b"\0" + str(len(entries)).encode("ascii") + b" 0\n" \
            + bytes.fromhex(root_tree)
        body += GIT_CACHE_TREE_EXTENSION + struct.pack(">I", len(payload)) + payload
    return body + hashlib.sha1(body).digest()


def _git_fixture_repository(root: Path, *, files: dict, cache_tree: bool = True) -> str:
    """One synthetic repository with a single commit over `files`."""
    (root / ".git" / "refs" / "heads").mkdir(parents=True)
    (root / ".git" / "objects").mkdir(parents=True, exist_ok=True)
    entries = []
    tree_entries = []
    for path in sorted(files):
        payload, mode = files[path]
        oid = _git_fixture_write_object(root, payload, b"blob")
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        if mode == GIT_MODE_EXECUTABLE:
            os.chmod(target, 0o755)
        entries.append((path, oid, mode))
        tree_entries.append((path, oid, mode))
    # A flat tree is enough: every fixture path is a single component.
    tree_payload = b"".join(
        f"{mode:o} {path}".encode("utf-8") + b"\0" + bytes.fromhex(oid)
        for path, oid, mode in sorted(tree_entries))
    tree_oid = _git_fixture_write_object(root, tree_payload, b"tree")
    commit_payload = (f"tree {tree_oid}\n"
                      f"author T <t@example.invalid> 0 +0000\n"
                      f"committer T <t@example.invalid> 0 +0000\n\nfixture\n"
                      ).encode("utf-8")
    commit_oid = _git_fixture_write_object(root, commit_payload, b"commit")
    (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (root / ".git" / "refs" / "heads" / "main").write_text(commit_oid + "\n",
                                                           encoding="utf-8")
    (root / ".git" / "index").write_bytes(
        _git_fixture_index(entries, tree_oid if cache_tree else None))
    return commit_oid


def _self_test_git_provenance(reporter: _Reporter, root: Path) -> None:
    # A hand-built blob whose object id is the one Git itself computes for the
    # bytes b"hello\n"; it anchors the hashing against an external ground truth.
    reporter.check("the Git blob object id matches Git's own for known bytes",
                   _git_blob_oid(b"hello\n")
                   == "ce013625030ba8dba906f756967f9e9ca394464a",
                   _git_blob_oid(b"hello\n"))

    clean = root / "git-clean"
    clean.mkdir()
    commit = _git_fixture_repository(clean, files={
        "README.md": (b"one\n", GIT_MODE_REGULAR),
        "run.sh": (b"#!/bin/sh\n", GIT_MODE_EXECUTABLE),
    })
    provenance = resolve_git_provenance(clean)
    reporter.check("a clean fixture repository resolves its full HEAD commit",
                   provenance["analysis_code_commit"] == commit
                   and GIT_COMMIT_RE.fullmatch(commit) is not None
                   and provenance["worktree_clean"] is True
                   and provenance["tracked_paths_verified"] == 2, str(provenance))

    # An untracked file is deliberately not dirty: the candidate output tree is
    # itself untracked until someone commits it.
    (clean / "results").mkdir()
    (clean / "results" / "untracked.txt").write_text("x\n", encoding="utf-8")
    reporter.check("an untracked path does not make the analysis-code worktree dirty",
                   resolve_git_provenance(clean)["worktree_clean"] is True, "")

    (clean / "README.md").write_text("two\n", encoding="utf-8")
    reporter.rejects("a modified tracked file is a dirty worktree",
                     lambda: resolve_git_provenance(clean), "the worktree is dirty")
    (clean / "README.md").write_text("one\n", encoding="utf-8")
    resolve_git_provenance(clean)

    os.chmod(clean / "run.sh", 0o644)
    reporter.rejects("a changed executable bit is a dirty worktree",
                     lambda: resolve_git_provenance(clean), "executable bit differs")
    os.chmod(clean / "run.sh", 0o755)

    (clean / "README.md").unlink()
    reporter.rejects("a deleted tracked file is a dirty worktree",
                     lambda: resolve_git_provenance(clean), "missing from the worktree")
    (clean / "README.md").write_text("one\n", encoding="utf-8")

    for marker in ("MERGE_HEAD", "CHERRY_PICK_HEAD"):
        (clean / ".git" / marker).write_text("0" * 40 + "\n", encoding="utf-8")
        reporter.rejects(f"an in-progress {marker} refuses to claim a clean checkout",
                         lambda: resolve_git_provenance(clean), "is in progress")
        (clean / ".git" / marker).unlink()

    staged = root / "git-staged"
    staged.mkdir()
    _git_fixture_repository(staged, files={"README.md": (b"one\n", GIT_MODE_REGULAR)},
                            cache_tree=False)
    reporter.rejects("an index with no valid cache-tree is refused rather than assumed "
                     "clean",
                     lambda: resolve_git_provenance(staged), "no valid cache-tree")

    mismatched = root / "git-mismatched"
    mismatched.mkdir()
    _git_fixture_repository(mismatched, files={"README.md": (b"one\n", GIT_MODE_REGULAR)})
    index_path = mismatched / ".git" / "index"
    payload = index_path.read_bytes()
    entries, cache_tree = _git_parse_index(payload)
    forged = _git_fixture_index([(entries[0]["path"].decode(), entries[0]["oid"],
                                  entries[0]["mode"])], "0" * 40)
    index_path.write_bytes(forged)
    reporter.rejects("an index tree that differs from HEAD's tree is a dirty index",
                     lambda: resolve_git_provenance(mismatched), "the index is dirty")
    index_path.write_bytes(payload)
    resolve_git_provenance(mismatched)

    index_path.write_bytes(payload[:-1] + bytes([payload[-1] ^ 0xFF]))
    reporter.rejects("an index that fails its own checksum is refused",
                     lambda: resolve_git_provenance(mismatched), "SHA-1 checksum")
    index_path.write_bytes(payload)

    unborn = root / "git-unborn"
    (unborn / ".git" / "refs" / "heads").mkdir(parents=True)
    (unborn / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    reporter.rejects("an unborn HEAD cannot name an analysis-code commit",
                     lambda: resolve_git_provenance(unborn), "resolves to nothing")

    abbreviated = root / "git-abbrev"
    (abbreviated / ".git").mkdir(parents=True)
    (abbreviated / ".git" / "HEAD").write_text("1234567\n", encoding="utf-8")
    reporter.rejects("an abbreviated HEAD is not a full valid commit",
                     lambda: resolve_git_provenance(abbreviated),
                     "neither a full commit nor a symbolic ref")

    missing = root / "git-missing"
    missing.mkdir()
    reporter.rejects("a directory that is not a repository cannot supply provenance",
                     lambda: resolve_git_provenance(missing), "no readable .git directory")


# --- 3.4 The descriptor-anchored output tree -------------------------------


def _self_test_output_containment(reporter: _Reporter, orchestrator, root: Path) -> None:
    """Every escape the audit reproduced, plus the ones next to it.

    Each case must fail before a single byte is written outside the temporary
    fixture repository, which is proved by scanning the outside directory
    afterwards."""
    documents = build_documents(orchestrator, _StubP35(), _fixture_records(),
                                _fixture_provenance())
    outside = root / "outside"
    outside.mkdir()
    witness = sorted(os.listdir(outside))

    def fresh(name: str) -> Path:
        repo = root / name
        repo.mkdir()
        return repo

    def attempt(repo: Path, output: Path | None = None):
        target = repo / DEFAULT_OUTPUT_ROOT_REL if output is None else output
        resolve_output_root(target, repo)
        tree = open_output_tree(repo, create=True)
        try:
            publish_documents(orchestrator, tree, documents, write=True)
        finally:
            close_output_tree(tree)

    # 1. The exact case the audit reproduced: results itself is a symlink to a
    #    directory outside the repository.
    repo = fresh("escape-results")
    os.symlink(outside, repo / "results")
    reporter.rejects("an ancestor symlink at results/ cannot redirect the output",
                     lambda: attempt(repo), "symlink")

    # 2. results/phase4 is a symlink outside.
    repo = fresh("escape-phase4")
    (repo / "results").mkdir()
    os.symlink(outside, repo / "results" / "phase4")
    reporter.rejects("a symlink at results/phase4 cannot redirect the output",
                     lambda: attempt(repo), "symlink")

    # 3. results/phase4/figures is a symlink outside.
    repo = fresh("escape-figures")
    (repo / "results" / "phase4").mkdir(parents=True)
    os.symlink(outside, repo / "results" / "phase4" / "figures")
    reporter.rejects("a symlink at results/phase4/figures cannot redirect the output",
                     lambda: attempt(repo), "symlink")

    # 4. One individual output file is a symlink to a file outside.
    repo = fresh("escape-file")
    (repo / "results" / "phase4" / "figures").mkdir(parents=True)
    (outside / "captured.md").write_text("original\n", encoding="utf-8")
    os.symlink(outside / "captured.md", repo / "results" / "phase4" / "report.md")
    reporter.rejects("a symlinked individual artifact is never written through",
                     lambda: attempt(repo), "symlink")
    reporter.check("the symlink target outside the repository was not rewritten",
                   (outside / "captured.md").read_text(encoding="utf-8") == "original\n", "")

    # 5. An unexpected special file where a directory belongs.
    repo = fresh("escape-special")
    (repo / "results").write_text("not a directory\n", encoding="utf-8")
    reporter.rejects("a non-directory output path component is refused",
                     lambda: attempt(repo), "not a directory")

    # 6. An arbitrary in-repository output directory is not a legal destination.
    repo = fresh("escape-arbitrary")
    for other in ("results/elsewhere", "analysis", "results/phase4/nested"):
        reporter.rejects(f"an output root at {other} is refused in production mode",
                         lambda path=other: resolve_output_root(repo / path, repo),
                         "the only production output root is")

    # 7. Raw and preflight evidence remain unreachable.
    repo = fresh("escape-raw")
    for forbidden in ("results/raw/phase4", "results/raw", "results/preflight"):
        reporter.rejects(f"an output root under {forbidden} is refused",
                         lambda path=forbidden: resolve_output_root(repo / path, repo),
                         "never writes under")
    reporter.rejects("an output root outside the repository is refused",
                     lambda: resolve_output_root(root / "elsewhere", repo), "outside")

    reporter.check("no byte was written outside the temporary fixture repositories",
                   sorted(os.listdir(outside)) == sorted(witness + ["captured.md"]),
                   str(sorted(os.listdir(outside))))


def _self_test_publication(reporter: _Reporter, orchestrator, root: Path) -> None:
    p35 = _StubP35()
    documents = build_documents(orchestrator, p35, _fixture_records(),
                                _fixture_provenance())
    repo = root / "repo"
    (repo / "results").mkdir(parents=True)
    output = repo / "results" / "phase4"
    resolved = resolve_output_root(output, repo)
    reporter.check("the one legal production output root is accepted",
                   resolved == Path(os.path.abspath(str(output))), str(resolved))

    tree = open_output_tree(repo, create=True)
    try:
        publish_documents(orchestrator, tree, documents, write=True)
        assert_output_tree_exact(tree)
        reporter.check("the frozen inventory publishes into a clean output tree", True)

        outcomes = publish_documents(orchestrator, tree, documents, write=True)
        reporter.check("an existing byte-identical artifact is verified, never rewritten",
                       all(value == "verified_byte_identical"
                           for value in outcomes.values()), str(outcomes))
        reporter.check("a published candidate is immutable: the analyzer has no promotion, "
                       "overwrite, or delete route",
                       all(value != "written" for value in outcomes.values()), "")

        (output / "report.md").write_text("tampered\n", encoding="utf-8")
        reporter.rejects("a differing existing artifact is never overwritten",
                         lambda: publish_documents(orchestrator, tree, documents,
                                                   write=True),
                         "refusing to overwrite a candidate artifact")
        reporter.rejects("verification fails on a tampered artifact",
                         lambda: publish_documents(orchestrator, tree, documents,
                                                   write=False),
                         "different content")
        (output / "report.md").unlink()
        reporter.rejects("verification fails on a missing artifact",
                         lambda: publish_documents(orchestrator, tree, documents,
                                                   write=False),
                         "is missing")
        publish_documents(orchestrator, tree, documents, write=True)

        (output / "unexpected.csv").write_text("x\n", encoding="utf-8")
        reporter.rejects("an unexpected artifact in the output tree is rejected",
                         lambda: assert_output_tree_exact(tree), "unexpected=")
        (output / "unexpected.csv").unlink()
        assert_output_tree_exact(tree)

        os.symlink(output / "report.md", output / "figures" / "linked.svg")
        reporter.rejects("a symlink inside the output tree is rejected",
                         lambda: assert_output_tree_exact(tree), "not a regular file")
        (output / "figures" / "linked.svg").unlink()

        (output / "stray").mkdir()
        reporter.rejects("an unexpected directory in the output tree is rejected",
                         lambda: assert_output_tree_exact(tree), "unexpected directory")
        (output / "stray").rmdir()
        assert_output_tree_exact(tree)
    finally:
        close_output_tree(tree)

    missing_repo = root / "never-analysed"
    (missing_repo / "results").mkdir(parents=True)
    reporter.rejects("verification of an output tree that was never produced fails closed",
                     lambda: open_output_tree(missing_repo, create=False),
                     "does not exist; nothing to verify")
    reporter.check("no artifact was written under results/raw/",
                   not (repo / "results" / "raw").exists(), "")

    _self_test_output_containment(reporter, orchestrator, root)


def _self_test_pipeline(reporter: _Reporter, root: Path) -> None:
    """The evidence seam: a failed population revalidation must abort before a
    single scientific value is read or a single byte is written."""
    repo = root / "pipeline"
    (repo / "results" / "raw" / "phase4").mkdir(parents=True)
    calls: list[tuple] = []

    def refusing(orchestrator, p42, campaign_root, pilot_ids, final_ids):
        calls.append((tuple(pilot_ids), tuple(final_ids)))
        return 1

    def unreachable_provenance(_repo_root):
        raise AssertionError("provenance must not be consulted before revalidation passes")

    status = run_analysis(DEFAULT_REPO_ROOT, repo / "results" / "raw" / "phase4",
                          [PILOT_CAMPAIGN_ID], list(FINAL_CAMPAIGN_IDS),
                          repo / "results" / "phase4", write=True, revalidator=refusing,
                          git_provenance=unreachable_provenance)
    reporter.check("a failed P4.2 population revalidation aborts the whole analysis before "
                   "any provenance, evidence, or byte is touched",
                   status == 1, str(status))
    reporter.check("the revalidation received the declared pilot and the three finals",
                   calls == [((PILOT_CAMPAIGN_ID,), FINAL_CAMPAIGN_IDS)], str(calls))
    reporter.check("nothing was written when revalidation failed",
                   not (repo / "results" / "phase4").exists(), "")

    status = run_analysis(DEFAULT_REPO_ROOT, repo / "results" / "raw" / "phase4",
                          [PILOT_CAMPAIGN_ID], list(FINAL_CAMPAIGN_IDS)[:2],
                          repo / "results" / "phase4", write=True, revalidator=refusing,
                          git_provenance=unreachable_provenance)
    reporter.check("an incomplete declared population is refused before any revalidation",
                   status == 1 and len(calls) == 1, str(calls))

    def accepting(orchestrator, p42, campaign_root, pilot_ids, final_ids):
        return 0

    def dirty_provenance(_repo_root):
        raise GitProvenanceError("synthetic fixture: the worktree is dirty")

    status = run_analysis(DEFAULT_REPO_ROOT, repo / "results" / "raw" / "phase4",
                          [PILOT_CAMPAIGN_ID], list(FINAL_CAMPAIGN_IDS),
                          repo / "results" / "phase4", write=True, revalidator=accepting,
                          git_provenance=dirty_provenance)
    reporter.check("a dirty or unverifiable analysis-code provenance aborts before any "
                   "evidence is read or any byte is written",
                   status == 1 and not (repo / "results" / "phase4").exists(), str(status))

    status = run_analysis(DEFAULT_REPO_ROOT, repo / "results" / "raw" / "phase4",
                          [PILOT_CAMPAIGN_ID], list(FINAL_CAMPAIGN_IDS),
                          DEFAULT_REPO_ROOT / "results" / "elsewhere", write=True,
                          revalidator=accepting,
                          git_provenance=lambda _root: _fixture_provenance())
    reporter.check("an output root other than results/phase4 aborts production analysis "
                   "before anything is created",
                   status == 1
                   and not (DEFAULT_REPO_ROOT / "results" / "elsewhere").exists(),
                   str(status))


def _self_test_evidence(reporter: _Reporter, orchestrator, p35, root: Path) -> None:
    """collect_campaign_evidence() against hand-built temporary trees. No real
    campaign is read and nothing outside the temporary tree is touched."""
    repo = root / "evidence"
    campaign_id = FINAL_CAMPAIGN_IDS[0]
    unit_dir = f"results/raw/unit/{campaign_id}"
    (repo / unit_dir / "analysis").mkdir(parents=True)
    (repo / f"results/raw/phase4/{campaign_id}/exp03").mkdir(parents=True)

    artifact_rel = "analysis/pilot_statistics.csv"
    payload = b"method,stages\nldgsts,2\n"
    (repo / unit_dir / artifact_rel).write_bytes(payload)
    digest = orchestrator.sha256_bytes(payload)
    unit_manifest = {"campaign_id": campaign_id, "state": UNIT_TERMINAL_ANALYZED,
                     "publishable": False, "artifact_sha256": {artifact_rel: digest}}
    manifest_bytes = orchestrator.canonical_json_bytes(unit_manifest)
    (repo / unit_dir / "manifest.json").write_bytes(manifest_bytes)
    evidence = {
        "unit": "P1.4", "unit_state": UNIT_TERMINAL_ANALYZED,
        "unit_campaign_dir": unit_dir,
        "unit_manifest_path": f"{unit_dir}/manifest.json",
        "unit_manifest_sha256": orchestrator.sha256_bytes(manifest_bytes),
    }
    payloads, sources = read_unit_artifacts(
        orchestrator, repo, evidence, campaign_id=campaign_id, unit="P1.4",
        artifacts=(artifact_rel,))
    reporter.check("a pinned terminal artifact is read and re-verified against its hash",
                   payloads[artifact_rel] == payload and len(sources) == 1, "")

    (repo / unit_dir / artifact_rel).write_bytes(payload + b"tampered\n")
    reporter.rejects("a referenced artifact modified after acceptance is rejected",
                     lambda: read_unit_artifacts(
                         orchestrator, repo, evidence, campaign_id=campaign_id, unit="P1.4",
                         artifacts=(artifact_rel,)),
                     "changed after it was accepted")
    (repo / unit_dir / artifact_rel).write_bytes(payload)

    (repo / unit_dir / "manifest.json").write_bytes(manifest_bytes + b"\n")
    reporter.rejects("a tampered unit manifest revision is rejected",
                     lambda: read_unit_artifacts(
                         orchestrator, repo, evidence, campaign_id=campaign_id, unit="P1.4",
                         artifacts=(artifact_rel,)),
                     "hashes to")
    (repo / unit_dir / "manifest.json").write_bytes(manifest_bytes)

    (repo / unit_dir / artifact_rel).unlink()
    os.symlink(repo / unit_dir / "manifest.json", repo / unit_dir / artifact_rel)
    reporter.rejects("a symlinked referenced artifact is rejected",
                     lambda: read_unit_artifacts(
                         orchestrator, repo, evidence, campaign_id=campaign_id, unit="P1.4",
                         artifacts=(artifact_rel,)),
                     "symlink")
    (repo / unit_dir / artifact_rel).unlink()
    (repo / unit_dir / artifact_rel).write_bytes(payload)

    base_manifest = {
        "campaign_id": campaign_id, "campaign_kind": "final", "state": "COMPLETE",
        "outcome": "COMPLETE", "publishable": False, "git_commit": FINAL_EXECUTION_COMMIT,
        "gpu": {"uuid": "GPU-1", "name": "n", "compute_capability": "10.3",
                "driver_version": "610.43.02"},
        "stage_results": {},
    }
    for mutation, fragment in (
            ({"state": "IN_PROGRESS"}, "not terminally COMPLETE"),
            ({"campaign_kind": "pilot"}, "not 'final'"),
            ({"publishable": True}, "publishable=false"),
    ):
        broken = json.loads(json.dumps(base_manifest))
        broken.update(mutation)
        reporter.rejects(f"an incomplete or non-terminal campaign is rejected ({mutation})",
                         lambda doc=broken: collect_campaign_evidence(
                             orchestrator, p35, repo, campaign_id, doc), fragment)
    reporter.rejects("a campaign with no completed memory.analyze stage is rejected",
                     lambda: collect_campaign_evidence(
                         orchestrator, p35, repo, campaign_id,
                         json.loads(json.dumps(base_manifest))),
                     "has no completed memory.analyze")
    non_terminal = json.loads(json.dumps(base_manifest))
    non_terminal["stage_results"] = {STAGE_MEMORY_ANALYZE: {"evidence": dict(
        evidence, unit_state="COMPLETE")}}
    reporter.rejects("a unit campaign that never reached ANALYZED is rejected",
                     lambda: collect_campaign_evidence(
                         orchestrator, p35, repo, campaign_id, non_terminal),
                     "not 'ANALYZED'")


def run_self_test() -> int:
    reporter = _Reporter("analyze_phase4_p43: self-test")
    orchestrator, _p42, p35 = load_repository_modules(DEFAULT_REPO_ROOT)
    _self_test_statistics(reporter)
    _self_test_parsers(reporter)
    _self_test_population(reporter)
    _self_test_aggregation(reporter, orchestrator)
    with tempfile.TemporaryDirectory(prefix="p43-selftest-") as temporary:
        root = Path(temporary)
        _self_test_publication(reporter, orchestrator, root)
        _self_test_git_provenance(reporter, root)
        _self_test_pipeline(reporter, root)
        _self_test_evidence(reporter, orchestrator, p35, root)
    reporter.check("the frozen artifact inventory is exactly nine artifacts",
                   len(ARTIFACT_RELATIVE_PATHS) == 9, str(ARTIFACT_RELATIVE_PATHS))
    reporter.check("the frozen population is one excluded pilot plus three finals",
                   CAMPAIGN_COUNT == 3 and PILOT_CAMPAIGN_ID not in FINAL_CAMPAIGN_IDS, "")
    reporter.check("the acceptance attestation is not part of the analysis inventory",
                   ACCEPTANCE_RELATIVE_PATH not in ARTIFACT_RELATIVE_PATHS, "")
    reporter.check("every declared metric carries an evidence classification",
                   all(entry[0] in EVIDENCE_CLASSES and entry[1]
                       for entry in METRIC_EVIDENCE.values()), "")
    if reporter.failures:
        print(f"analyze_phase4_p43: self-test: FAILED ({len(reporter.failures)} check(s))",
              file=sys.stderr)
        return 1
    print("analyze_phase4_p43: self-test: OK (temporary fixtures only; no GPU, no container, "
          "no campaign, and no repository file was modified)")
    return 0


# ===========================================================================
# CLI
# ===========================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="analyze_phase4_p43.py",
        description="P4.3: the offline, read-only cross-campaign integrated analysis over "
                    "the frozen Phase 4 population. Executes no GPU command and never "
                    "starts, resumes, repairs, or creates a campaign.",
    )
    parser.add_argument("--self-test", action="store_true",
                        help="run the focused synthetic suite over temporary fixtures and "
                             "exit; standalone only")
    parser.add_argument("--analyze", action="store_true",
                        help="produce the curated artifacts from the real accepted evidence")
    parser.add_argument("--verify", action="store_true",
                        help="recompute the analysis and compare every output byte for byte")
    parser.add_argument("--campaign-root", default=None,
                        help="the existing results/raw/phase4 directory (read-only)")
    parser.add_argument("--pilot-campaign-id", action="append", default=None, metavar="ID",
                        help=f"the accepted pilot ({PILOT_CAMPAIGN_ID}); it is recorded as "
                             f"excluded provenance and never aggregated")
    parser.add_argument("--final-campaign-id", action="append", default=None, metavar="ID",
                        help=f"a declared final campaign; exactly {CAMPAIGN_COUNT} are "
                             f"required, in the frozen order")
    parser.add_argument("--output-root", default=None,
                        help=f"the curated output tree (repository-relative, e.g. "
                             f"{DEFAULT_OUTPUT_ROOT_REL}); never under results/raw/")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    production = args.analyze or args.verify
    if args.self_test:
        if production or any(value is not None for value in
                             (args.campaign_root, args.pilot_campaign_id,
                              args.final_campaign_id, args.output_root)):
            print("analyze_phase4_p43: --self-test is standalone", file=sys.stderr)
            return 2
        return run_self_test()
    if args.analyze and args.verify:
        print("analyze_phase4_p43: --analyze and --verify are mutually exclusive",
              file=sys.stderr)
        return 2
    if not production:
        print("analyze_phase4_p43: one of --self-test, --analyze, or --verify is required",
              file=sys.stderr)
        return 2
    missing = [name for name, value in (("--campaign-root", args.campaign_root),
                                        ("--pilot-campaign-id", args.pilot_campaign_id),
                                        ("--final-campaign-id", args.final_campaign_id),
                                        ("--output-root", args.output_root))
               if value is None]
    if missing:
        print(f"analyze_phase4_p43: the production modes require {missing}; the real raw "
              f"evidence and the frozen campaign IDs are always declared explicitly",
              file=sys.stderr)
        return 2
    return run_analysis(DEFAULT_REPO_ROOT, Path(args.campaign_root),
                        list(args.pilot_campaign_id), list(args.final_campaign_id),
                        Path(args.output_root), write=bool(args.analyze))


if __name__ == "__main__":
    sys.exit(main())
