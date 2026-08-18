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

Exit codes: 0 OK; 1 at least one check failed; 2 usage error.
"""

from __future__ import annotations

import argparse
import csv
import errno
import importlib.util
import io
import json
import math
import os
import re
import stat
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_REPO_ROOT = SCRIPT_DIR.parent

ORCHESTRATOR_RELATIVE_PATH = "scripts/phase4_orchestrator.py"
P42_CHECKER_RELATIVE_PATH = "scripts/check_phase4_campaigns_p42.py"
P35_CHECKER_RELATIVE_PATH = "scripts/check_gemm_comparison_p35.py"

SCHEMA_VERSION = "p43.v1"
UNIT = "P4.3"
PUBLISHABLE = False
PUBLICATION_STATUS = (
    "publishable=false; the P4.3 independent audit and the production analysis "
    "of the three real final campaigns are pending"
)

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
_STAT_FIELDS = (
    "campaign_count",
    "campaign_1_value",
    "campaign_2_value",
    "campaign_3_value",
    "mean",
    "median",
    "stdev_sample",
    "cv_percent",
    "minimum",
    "maximum",
    "cv_review_flag",
    "notes",
)
MEMORY_CSV_FIELDS = (
    "schema_version", "section", "method", "stages", "bytes_in_flight_kib",
    "metric", "unit",
) + _STAT_FIELDS
UMMA_CSV_FIELDS = (
    "schema_version", "section", "method", "n", "depth", "cta_group",
    "metric", "unit",
) + _STAT_FIELDS
GEMM_CSV_FIELDS = (
    "schema_version", "section", "shape_index", "shape_id", "m", "n", "k", "l",
    "candidate_index", "variant", "method", "metric", "unit",
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
        "cv_percent": cv_percent,
        "cv_reason": cv_reason,
        "minimum": min(numbers),
        "maximum": max(numbers),
        "cv_review_flag": cv_flag,
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
        format_decimal(summary["cv_percent"], DECIMALS_CV),
        format_decimal(summary["minimum"], decimals),
        format_decimal(summary["maximum"], decimals),
        summary["cv_review_flag"],
        notes,
    ]


def summary_json(summary: dict | None, *, metric: str,
                 raw_values: list | None = None) -> dict:
    decimals = decimals_for(metric)
    if summary is None:
        return {
            "metric": metric,
            "campaign_count": CAMPAIGN_COUNT,
            "campaign_values": list(raw_values or []),
            "statistics": NOT_APPLICABLE,
        }
    return {
        "metric": metric,
        "campaign_count": summary["campaign_count"],
        "campaign_values": [quantize(value, decimals) for value in summary["campaign_values"]],
        "mean": quantize(summary["mean"], decimals),
        "median": quantize(summary["median"], decimals),
        "stdev_sample": quantize(summary["stdev_sample"], decimals),
        "cv_percent": quantize(summary["cv_percent"], DECIMALS_CV),
        "cv_status": summary["cv_reason"],
        "cv_review_flag": summary["cv_review_flag"],
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
    configurations = []
    for method, stages, bif in P14_CONFIG_KEYS:
        key = (method, stages, bif)
        values = [record["memory"]["pilot_statistics"][key]["median_gbps"]
                  for record in records]
        summary = summarize_metric(values, metric="median_effective_gbps")
        rows.append([SCHEMA_VERSION, "configuration", method, str(stages), str(bif),
                     "median_effective_gbps", "GB/s"]
                    + stat_cells(summary, metric="median_effective_gbps",
                                 notes="campaign_level_median_of_30_repetitions"))
        configurations.append({
            "method": method, "stages": stages, "bytes_in_flight_kib": bif,
            **summary_json(summary, metric="median_effective_gbps"),
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
                     "tma_to_ldgsts_ratio", "ratio"]
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
                     "earliest_tested_candidate_saturation_bif_kib", "KiB"]
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
        values = [record["memory"]["ncu"][key]["dram_read_ratio"] for record in records]
        classifications = [record["memory"]["ncu"][key]["hbm_classification"]
                           for record in records]
        agreement = consensus(classifications, label=f"hbm_classification {key}")
        summary = summarize_metric(values, metric="dram_read_ratio")
        note = (f"hbm={agreement['consensus']}" if agreement["consensus"] else "hbm=mixed")
        rows.append([SCHEMA_VERSION, "ncu_validation", method, str(stages), str(bif),
                     "dram_read_ratio", "ratio"]
                    + stat_cells(summary, metric="dram_read_ratio", notes=note))
        ncu.append({"index": index, "method": method, "stages": stages,
                    "bytes_in_flight_kib": bif,
                    **summary_json(summary, metric="dram_read_ratio"),
                    "campaign_hbm_classifications": classifications,
                    "hbm_classification_consensus": agreement["consensus"]})

    saturation_stable = all(entry["stable_across_campaigns"] for entry in saturation)
    section = {
        "title": "Experiment 1 -- LDGSTS versus TMA HBM-to-SMEM data movement",
        "configuration_count": len(P14_CONFIG_KEYS),
        "configurations": configurations,
        "pair_ratios": pair_ratios,
        "pair_ratio_interpretation": (
            "a value above one means TMA measured higher sustained bandwidth in that "
            "campaign, and a value below one means LDGSTS measured higher; this is a "
            "measured ratio, not a winner and not a significance claim"),
        "saturation_candidates": saturation,
        "saturation_consensus_available": saturation_stable,
        "ncu_validation": ncu,
        "ncu_coverage": {
            "profiled_cases": len(P14_NCU_CASES),
            "total_configurations": len(P14_CONFIG_KEYS),
            "limitation": (
                "Nsight Compute/HBM validation covers exactly these six predefined cases "
                "and is never extrapolated to the other twelve configurations"),
        },
    }
    return rows, section


# ===========================================================================
# Experiment 2 -- BF16 UMMA throughput.
# ===========================================================================


def aggregate_experiment_2(records: list[dict]) -> tuple[list[list[str]], dict]:
    rows: list[list[str]] = []
    configurations = []
    for method, n, depth in P24_CONFIG_KEYS:
        key = (method, n, depth)
        cta_group = P24_CTA_GROUP[method]
        entry = {"method": method, "n": n, "depth": depth, "cta_group": cta_group,
                 "metrics": {}}
        for metric, unit in (("median_flops_per_cycle", "FLOP/cycle"),
                             ("median_flops_per_cycle_per_sm", "FLOP/cycle/SM")):
            values = [record["umma"]["configuration"][key][metric] for record in records]
            summary = summarize_metric(values, metric=metric)
            rows.append([SCHEMA_VERSION, "configuration", method, str(n), str(depth),
                         str(cta_group), metric, unit]
                        + stat_cells(summary, metric=metric,
                                     notes="clock_independent_campaign_level_median"))
            entry["metrics"][metric] = summary_json(summary, metric=metric)
        configurations.append(entry)

    scaling = []
    for n, depth in P24_SCALING_KEYS:
        key = (n, depth)
        flags = [record["umma"]["scaling"][key]["surprising_value_flag"]
                 for record in records]
        entry = {"n": n, "depth": depth, "campaign_surprising_value_flags": flags,
                 "metrics": {}}
        for metric, unit in (("speedup_2sm_over_1sm", "ratio"),
                             ("scaling_efficiency_percent", "percent")):
            values = [record["umma"]["scaling"][key][metric] for record in records]
            summary = summarize_metric(values, metric=metric)
            outside = any(value < 0.0 or value > 100.0 for value in values)
            note = ("value_outside_0_100_preserved_unclamped" if metric.endswith("percent")
                    and outside else "campaign_level_ratio_never_pooled")
            rows.append([SCHEMA_VERSION, "scaling", NOT_APPLICABLE, str(n), str(depth),
                         NOT_APPLICABLE, metric, unit]
                        + stat_cells(summary, metric=metric, notes=note))
            entry["metrics"][metric] = summary_json(summary, metric=metric)
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
                     "earliest_tested_candidate_saturation_depth", "depth"]
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
                 NOT_APPLICABLE, "estimated_tflops_per_sm", "TFLOP/s/SM"]
                + stat_cells(ceiling_summary, metric="estimated_tflops_per_sm", notes=note,
                             raw_values=[format_decimal(value, decimals_for(
                                 "estimated_tflops_per_sm")) for value in tflops_values]))

    device = aggregate_device_equivalent(records)

    section = {
        "title": "Experiment 2 -- BF16 UMMA (fifth-generation Tensor Core) throughput",
        "configuration_count": len(P24_CONFIG_KEYS),
        "configurations": configurations,
        "scaling": scaling,
        "scaling_note": (
            "speedup and scaling efficiency are each campaign's own values, summarized "
            "across the three campaigns; a value outside [0, 100] is preserved unclamped "
            "and keeps the closed unit's surprising-value diagnostic"),
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
                "an empirical one-/two-SM microbenchmark ceiling candidate, never a "
                "theoretical architectural peak and never a measured whole-device "
                "throughput"),
        },
        "device_equivalent_estimate": device,
        "profile_validation": [
            {"campaign_id": record["campaign_id"],
             "profiled_cases": record["umma"]["profile_validation"]["case_count"],
             "sm_clock_ok_count": record["umma"]["profile_validation"]["sm_clock_ok_count"],
             "sm_clock_statuses": record["umma"]["profile_validation"][
                 "sm_clock_statuses_distinct"]}
            for record in records],
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
                    note = "campaign_level_ratio_never_recomputed_from_aggregates"
                else:
                    note = "campaign_level_value"
                rows.append([SCHEMA_VERSION, "candidate", str(shape_index), shape_id,
                             str(mnkl[0]), str(mnkl[1]), str(mnkl[2]), str(mnkl[3]),
                             str(candidate_index), variant, method, metric, unit]
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
                     "variant"]
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
        "shapes": shapes,
        "notes": [
            "beating cuBLASLt is not a success criterion",
            "a ratio above one and a negative gap mean the candidate measured faster than "
            "cuBLASLt; neither is clamped",
            "the cuBLASLt-relative ratio and gap are each campaign's own values, summarized "
            "across the three campaigns; they are never recomputed from aggregated means",
        ],
    }
    return rows, section


# ===========================================================================
# The integrated interpretation.
# ===========================================================================


def build_interpretation(experiment_1: dict, experiment_2: dict, experiment_3: dict) -> dict:
    """Separate what was measured from what was derived, modeled, inferred, or
    is simply unavailable. No causal claim is made that the collected evidence
    cannot support, and no external architectural peak is ever imported."""
    stable_bests = {
        shape["shape_id"]: shape["best_cutedsl_variant"]["stable_best_cutedsl_variant"]
        for shape in experiment_3["shapes"]
    }
    return {
        "research_question": RESEARCH_QUESTION,
        "directly_measured": [
            "sustained HBM-to-SMEM bandwidth of the LDGSTS and TMA paths over the frozen "
            "18-configuration grid, as each campaign's median effective GB/s",
            "Nsight Compute DRAM read traffic for exactly six of those eighteen "
            "configurations, and their HBM classification",
            "clock-independent BF16 UMMA throughput (FLOP/cycle and FLOP/cycle/SM) over the "
            "frozen 24-configuration one-/two-SM grid",
            "per-configuration SM clock frequency for all profiled UMMA configurations",
            "kernel time of five frozen BF16 GEMM shapes for three CuTe DSL execution "
            "variants and one cuBLASLt baseline, hot-cache, correctness-checked first",
        ],
        "deterministic_derived": [
            "TMA-to-LDGSTS bandwidth ratios, computed inside each campaign and only then "
            "summarized across the three campaigns",
            "1-SM/2-SM speedup and scaling efficiency, computed inside each campaign",
            "GEMM TFLOP/s from the exact 2*M*N*K FLOP count and the measured kernel time",
            "the cuBLASLt-relative throughput ratio and signed gap, computed inside each "
            "campaign",
            "the cross-campaign mean, median, sample standard deviation (n-1), coefficient "
            "of variation where meaningful, minimum, and maximum over exactly three "
            "campaign-level values",
        ],
        "modeled_estimates": [
            "the empirical per-SM BF16 Tensor Core ceiling candidate: selected in "
            "clock-independent FLOP/cycle/SM space and only then converted with that same "
            "configuration's own measured SM clock; it is a one-/two-SM microbenchmark "
            "ceiling, not an architectural peak",
        ],
        "interpretations": [
            "the memory experiment measures a dedicated streaming microbenchmark, so it is "
            "consistent with, but is not a direct measurement of, the memory traffic a GEMM "
            "kernel generates",
            "where the TMA-to-LDGSTS ratio is close to one across all three campaigns, the "
            "evidence is consistent with the two paths reaching a similar sustained "
            "HBM-to-SMEM rate at that configuration",
            "the distance between the best CuTe DSL variant and cuBLASLt per shape is a "
            "measured throughput difference; the collected evidence does not attribute it "
            "to a specific cause",
        ],
        "unavailable_from_the_collected_evidence": [
            "whether any specific GEMM shape is HBM-bound, Tensor-Core-bound, "
            "scheduler-bound, or limited by another implementation cost: P3.5 collected no "
            "Nsight Compute profile of a GEMM kernel, so no bottleneck attribution is made",
            "a numerical roofline or arithmetic-intensity model: the memory benchmark and "
            "the GEMM measurements are not dimensionally comparable evidence of the same "
            "workload, and no compulsory-byte model was validated",
            "a cold-cache GEMM result: every GEMM measurement is hot-cache by construction",
            "a whole-device BF16 throughput figure whenever the device-wide estimate is "
            "unavailable",
            "any statement about the twelve HBM-unvalidated memory configurations beyond "
            "their measured bandwidth",
        ],
        "answer_summary": {
            "hbm_to_smem": (
                "measured: both equivalent HBM-to-SMEM paths sustain the bandwidths recorded "
                "in memory_paths.csv over the frozen grid, and the per-campaign "
                "TMA-to-LDGSTS ratios are summarized there; the saturation candidate is "
                "reported per group and only as a cross-campaign consensus when all three "
                "campaigns agree, never as a universal HBM saturation threshold"),
            "tensor_core": (
                "measured: the clock-independent FLOP/cycle and FLOP/cycle/SM values in "
                "umma_throughput.csv, plus each campaign's 1-SM/2-SM scaling; the empirical "
                "per-SM ceiling candidate is a microbenchmark ceiling and is reported as a "
                "cross-campaign statistic only when all three campaigns select the same "
                "configuration"),
            "cutedsl_versus_cublaslt": (
                "measured: per shape and candidate, gemm_comparison.csv reports the "
                "campaign-level kernel time, TFLOP/s, cuBLASLt-relative ratio, and signed "
                "gap, each summarized across the three campaigns; the stable best CuTe DSL "
                "variant per shape is " + json.dumps(stable_bests, sort_keys=True)
                + " (null means no stable best variant across the three campaigns)"),
            "constraint_attribution": (
                "cannot determine from the collected evidence: no GEMM-level profile exists, "
                "so the measured GEMM throughput is not attributed to the memory path, the "
                "Tensor Core ceiling, the scheduler, or any other single cost"),
        },
    }


def build_limitations(experiment_1: dict, experiment_2: dict, experiment_3: dict) -> list[str]:
    return [
        f"the independent replicate is one complete final campaign; the cross-campaign "
        f"sample size is {CAMPAIGN_COUNT}, which is small and supports descriptive "
        f"statistics only",
        "no p-value, significance claim, or cross-campaign confidence interval is computed; "
        "the within-campaign confidence intervals the closed units recorded remain "
        "provenance and are never reinterpreted as cross-campaign intervals",
        "no observation and no campaign was removed; no outlier filter was applied; a "
        f"coefficient of variation above {CV_REVIEW_THRESHOLD_PERCENT:.1f}% is a review "
        f"diagnostic only and never excludes anything",
        "a coefficient of variation is not computed for signed or zero-centred quantities "
        "such as gap_to_cublaslt_pct",
        f"Nsight Compute HBM validation covers exactly "
        f"{experiment_1['ncu_coverage']['profiled_cases']} of "
        f"{experiment_1['ncu_coverage']['total_configurations']} memory configurations and "
        f"is not extrapolated to the rest",
        "the memory microbenchmark is not a direct measurement of the memory traffic a GEMM "
        "kernel generates",
        "the BF16 UMMA ceiling is a one-/two-SM empirical microbenchmark ceiling; it is not "
        "an architectural peak and not a measured whole-device throughput",
        "no SM count is imported from an external specification, hard-coded, or inferred; "
        "without validated agreeing SM-count evidence the device-wide estimate stays "
        "structurally unavailable",
        f"every GEMM measurement is hot-cache ({experiment_3['cache_mode']}) and must not be "
        f"described as a cold-cache workload",
        "P3.5 collected no Nsight Compute profile of a GEMM kernel, so no GEMM bottleneck "
        "attribution, roofline placement, or arithmetic-intensity classification is made",
        "the source GEMM rows are run_kind=smoke evidence captured by the campaign; they "
        "carry publishable=false and are reported as measured values, not as a validated "
        "publication-grade benchmark",
        "the sweep order inside each closed unit is fixed and non-randomized, a limitation "
        "the closed units already recorded",
        "the accepted pilot is excluded from every statistic here; it qualifies the "
        "orchestration path only",
        "no P4.3 result is publishable: the independent audit of this analysis layer, the "
        "production run against the three real final campaigns, and the review of the "
        "resulting artifacts are all still pending",
    ]


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
                           limitations: list[str], sources: list[dict]) -> dict:
    reference = records[0]
    return {
        "schema_version": SCHEMA_VERSION,
        "unit": UNIT,
        "analysis_kind": "cross_campaign_integrated_analysis",
        "research_question": RESEARCH_QUESTION,
        "population": {
            "campaign_count": CAMPAIGN_COUNT,
            "final_campaign_ids": list(FINAL_CAMPAIGN_IDS),
            "pilot_campaign_id_excluded": PILOT_CAMPAIGN_ID,
            "pilot_role": PILOT_ROLE,
            "final_execution_commit": reference["git_commit"],
            "gpu": dict(reference["gpu"]),
            "comparable_provenance": {key: reference["provenance"][key]
                                      for key in sorted(reference["provenance"])},
        },
        "statistical_policy": {
            "independent_replicate": "one complete final campaign",
            "campaign_count": CAMPAIGN_COUNT,
            "statistics": ["mean", "median", "sample_standard_deviation_n_minus_1",
                           "coefficient_of_variation", "minimum", "maximum"],
            "cv_review_threshold_percent": CV_REVIEW_THRESHOLD_PERCENT,
            "cv_scope": "strictly positive performance metrics only",
            "cv_effect": "a diagnostic flag; it never excludes a campaign or changes a result",
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
        "limitations": limitations,
        "sources": sources,
        "publishable": PUBLISHABLE,
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
    lines.append("## 1. Population and provenance")
    lines.append("")
    lines.append(f"* Independent replicate: **one complete final campaign** "
                 f"(`campaign_count = {population['campaign_count']}`).")
    for campaign_id in population["final_campaign_ids"]:
        lines.append(f"* Final campaign `{campaign_id}`.")
    lines.append(f"* Accepted pilot `{population['pilot_campaign_id_excluded']}` is "
                 f"{population['pilot_role']}.")
    lines.append(f"* Final execution commit `{population['final_execution_commit']}`.")
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
    lines.append("Campaign-level median effective bandwidth, summarized across the three "
                 "final campaigns (GB/s).")
    lines.append("")
    rows = []
    for entry in exp1["configurations"]:
        rows.append([entry["method"], str(entry["stages"]), str(entry["bytes_in_flight_kib"]),
                     format_decimal(entry["mean"], 6), format_decimal(entry["median"], 6),
                     format_decimal(entry["stdev_sample"], 6),
                     format_decimal(entry["cv_percent"], DECIMALS_CV),
                     format_decimal(entry["minimum"], 6), format_decimal(entry["maximum"], 6),
                     entry["cv_review_flag"]])
    lines.extend(_markdown_table(
        ["method", "stages", "bif_kib", "mean", "median", "stdev", "cv_%", "min", "max",
         "flag"], rows))
    lines.append("")
    lines.append("TMA-to-LDGSTS ratio per identical `(stages, bytes_in_flight_kib)` pair. "
                 "Above one means TMA measured higher; below one means LDGSTS measured "
                 "higher. This is a measured ratio, not a winner and not a significance "
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
    lines.append(f"Nsight Compute HBM validation covers exactly "
                 f"{exp1['ncu_coverage']['profiled_cases']} of "
                 f"{exp1['ncu_coverage']['total_configurations']} configurations. "
                 f"{exp1['ncu_coverage']['limitation'].capitalize()}.")
    lines.append("")
    rows = []
    for entry in exp1["ncu_validation"]:
        rows.append([str(entry["index"]), entry["method"], str(entry["stages"]),
                     str(entry["bytes_in_flight_kib"]), format_decimal(entry["mean"], 9),
                     format_decimal(entry["minimum"], 9), format_decimal(entry["maximum"], 9),
                     str(entry["hbm_classification_consensus"])])
    lines.extend(_markdown_table(
        ["case", "method", "stages", "bif_kib", "dram_read_ratio mean", "min", "max",
         "classification"], rows))
    lines.append("")

    lines.append("## 4. Experiment 2 — BF16 UMMA throughput")
    lines.append("")
    lines.append("Clock-independent campaign-level medians, summarized across the three "
                 "final campaigns.")
    lines.append("")
    rows = []
    for entry in exp2["configurations"]:
        per_sm = entry["metrics"]["median_flops_per_cycle_per_sm"]
        total = entry["metrics"]["median_flops_per_cycle"]
        rows.append([entry["method"], str(entry["n"]), str(entry["depth"]),
                     str(entry["cta_group"]), format_decimal(total["mean"], 6),
                     format_decimal(per_sm["mean"], 6),
                     format_decimal(per_sm["cv_percent"], DECIMALS_CV),
                     per_sm["cv_review_flag"]])
    lines.extend(_markdown_table(
        ["method", "N", "depth", "cta_group", "FLOP/cycle mean", "FLOP/cycle/SM mean",
         "cv_%", "flag"], rows))
    lines.append("")
    lines.append("1-SM/2-SM comparison. Each campaign's own speedup and scaling efficiency "
                 "are summarized; values outside `[0, 100]` are preserved unclamped and keep "
                 "the closed unit's surprising-value diagnostic.")
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
        lines.append(f"* Estimated TFLOP/s/SM across the three campaigns: mean "
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

    lines.append("## 6. Integrated interpretation")
    lines.append("")
    lines.append(f"> {interpretation['research_question']}")
    lines.append("")
    for heading, key in (("Directly measured", "directly_measured"),
                         ("Deterministic derived quantities", "deterministic_derived"),
                         ("Modeled estimates", "modeled_estimates"),
                         ("Interpretations and inferences", "interpretations"),
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

    lines.append("## 7. Limitations")
    lines.append("")
    for item in document["limitations"]:
        lines.append(f"* {item}")
    lines.append("")
    lines.append("## 8. Status")
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
        f"<metadata>{_xml_escape(PUBLICATION_STATUS)}</metadata>",
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
    out = _svg_open("Cross-campaign mean of the campaign-level median effective bandwidth")
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
        f"n={CAMPAIGN_COUNT} final campaigns; the bar is min..max across campaigns. "
        f"Not a cold-cache GEMM workload and not an HBM saturation threshold."))
    return ("\n".join(out) + "\n").encode("utf-8")


def render_umma_svg(section: dict) -> bytes:
    lookup = {(entry["method"], entry["n"], entry["depth"]): entry
              for entry in section["configurations"]}
    metric = "median_flops_per_cycle_per_sm"
    values = [entry["metrics"][metric]["mean"] for entry in section["configurations"]]
    minimums = [entry["metrics"][metric]["minimum"] for entry in section["configurations"]]
    maximums = [entry["metrics"][metric]["maximum"] for entry in section["configurations"]]
    out = _svg_open("Cross-campaign mean of the campaign-level median FLOP/cycle/SM")
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
        f"n={CAMPAIGN_COUNT} final campaigns; the bar is min..max across campaigns. "
        f"Clock-independent; an empirical one-/two-SM ceiling, never an architectural peak."))
    return ("\n".join(out) + "\n").encode("utf-8")


def render_gemm_svg(section: dict) -> bytes:
    values, minimums, maximums = [], [], []
    for shape in section["shapes"]:
        for candidate in shape["candidates"]:
            values.append(candidate["metrics"]["tflops"]["mean"])
            minimums.append(candidate["metrics"]["tflops"]["minimum"])
            maximums.append(candidate["metrics"]["tflops"]["maximum"])
    out = _svg_open("Cross-campaign mean TFLOP/s per shape and candidate")
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
        f"n={CAMPAIGN_COUNT} final campaigns; the bar is min..max. Candidate order: {order}. "
        f"Hot cache; beating cuBLASLt is not a success criterion."))
    return ("\n".join(out) + "\n").encode("utf-8")


# ===========================================================================
# The complete analysis: evidence in, deterministic documents out.
# ===========================================================================


def build_documents(orchestrator, p35, records: list[dict]) -> list[tuple[str, bytes]]:
    """Every output artifact, in the frozen inventory order, as exact bytes."""
    if len(records) != CAMPAIGN_COUNT:
        raise P43Error(f"expected exactly {CAMPAIGN_COUNT} final campaign records")
    if [record["campaign_id"] for record in records] != list(FINAL_CAMPAIGN_IDS):
        raise P43Error("the campaign records are not the frozen final population in order")
    compare_campaign_provenance(records)

    memory_rows, experiment_1 = aggregate_experiment_1(records)
    umma_rows, experiment_2 = aggregate_experiment_2(records)
    gemm_rows, experiment_3 = aggregate_experiment_3(records, p35=p35)
    interpretation = build_interpretation(experiment_1, experiment_2, experiment_3)
    limitations = build_limitations(experiment_1, experiment_2, experiment_3)
    sources = [entry for record in records for entry in record["sources"]]

    summary = build_summary_document(
        records,
        {"experiment_1": experiment_1, "experiment_2": experiment_2,
         "experiment_3": experiment_3},
        interpretation, limitations, sources)

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
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "unit": UNIT,
        "analysis_kind": "cross_campaign_integrated_analysis",
        "campaign_count": CAMPAIGN_COUNT,
        "final_campaign_ids": list(FINAL_CAMPAIGN_IDS),
        "campaign_value_column_order": [f"campaign_{index + 1}_value"
                                        for index in range(CAMPAIGN_COUNT)],
        "pilot_campaign_id_excluded": PILOT_CAMPAIGN_ID,
        "pilot_role": PILOT_ROLE,
        "final_execution_commit": records[0]["git_commit"],
        "gpu": dict(records[0]["gpu"]),
        "comparable_provenance": {key: records[0]["provenance"][key]
                                  for key in sorted(records[0]["provenance"])},
        "sources": sources,
        "artifact_sha256": {relative: orchestrator.sha256_bytes(payload)
                            for relative, payload in documents},
        "artifact_inventory": list(ARTIFACT_RELATIVE_PATHS),
        "self_reference": (
            "analysis_manifest.json is the one artifact it cannot hash; --verify recomputes "
            "the complete analysis and compares every byte of it as well"),
        "publishable": PUBLISHABLE,
        "publication_status": PUBLICATION_STATUS,
    }
    documents.append((MANIFEST_RELATIVE_PATH, orchestrator.canonical_json_bytes(manifest)))
    if [relative for relative, _ in documents] != list(ARTIFACT_RELATIVE_PATHS):
        raise P43Error("the produced documents are not the frozen artifact inventory")
    return documents


# ===========================================================================
# Output publication and verification.
# ===========================================================================


def resolve_output_root(output_root: Path, repo_root: Path) -> Path:
    """The output tree must live inside the repository and must never be under
    immutable raw evidence."""
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
    return absolute


def _open_output_dir(orchestrator, root: Path, subdirectory: str | None, *,
                     create: bool) -> int:
    if create:
        try:
            orchestrator.mkdir_component(root, must_not_exist=False)
            if subdirectory is not None:
                orchestrator.mkdir_component(root / subdirectory, must_not_exist=False)
        except orchestrator.OrchestratorError as exc:
            raise P43Error(f"{root}: {exc}") from exc
    parts = () if subdirectory is None else (subdirectory,)
    try:
        return orchestrator.open_dir_chain(root, *parts)
    except orchestrator.OrchestratorError as exc:
        raise P43Error(f"{root}: {exc}") from exc


def publish_documents(orchestrator, output_root: Path, documents: list[tuple[str, bytes]],
                      *, write: bool) -> dict[str, str]:
    """Publish (or verify) the frozen inventory.

    Nothing is ever overwritten. An artifact that already exists must be
    byte-identical, in which case it is verified rather than rewritten; a
    different existing artifact is fatal.
    """
    outcomes: dict[str, str] = {}
    grouped: dict[str | None, list[tuple[str, bytes]]] = {}
    for relative, payload in documents:
        parts = split_relative_path(relative)
        if len(parts) == 1:
            grouped.setdefault(None, []).append((parts[0], payload))
        elif len(parts) == 2:
            grouped.setdefault(parts[0], []).append((parts[1], payload))
        else:
            raise P43Error(f"{relative}: the output inventory is at most one level deep")
    for subdirectory in sorted(grouped, key=lambda value: (value is not None, value or "")):
        directory_fd = _open_output_dir(orchestrator, output_root, subdirectory, create=write)
        try:
            for name, payload in grouped[subdirectory]:
                label = name if subdirectory is None else f"{subdirectory}/{name}"
                try:
                    existing = orchestrator.read_file_nofollow(name, dir_fd=directory_fd)
                except orchestrator.OrchestratorError as exc:
                    if not _is_missing(exc):
                        raise P43Error(f"{label}: {exc}") from exc
                    if not write:
                        raise P43Error(f"{label}: is missing; nothing to verify") from exc
                    try:
                        orchestrator.write_file_exclusive(name, payload, dir_fd=directory_fd)
                    except orchestrator.OrchestratorError as inner:
                        raise P43Error(f"{label}: {inner}") from inner
                    outcomes[label] = "written"
                    continue
                if existing != payload:
                    raise P43Error(
                        f"{label}: already exists with different content; refusing to "
                        f"overwrite a reviewed artifact")
                outcomes[label] = "verified_byte_identical"
        finally:
            os.close(directory_fd)
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


def assert_output_tree_exact(output_root: Path) -> None:
    """The output tree must contain exactly the frozen inventory: no partial,
    conflicting, or unexpected artifact, no symlink, and no other file type."""
    expected_files = set(ARTIFACT_RELATIVE_PATHS)
    expected_dirs = {OUTPUT_FIGURES_DIR}
    observed_files: set[str] = set()
    for entry in sorted(os.listdir(output_root)):
        info = os.lstat(output_root / entry)
        if stat.S_ISDIR(info.st_mode):
            if entry not in expected_dirs:
                raise P43Error(f"{entry}/: unexpected directory in the output tree")
            for nested in sorted(os.listdir(output_root / entry)):
                nested_info = os.lstat(output_root / entry / nested)
                if not stat.S_ISREG(nested_info.st_mode):
                    raise P43Error(f"{entry}/{nested}: is not a regular file")
                observed_files.add(f"{entry}/{nested}")
            continue
        if not stat.S_ISREG(info.st_mode):
            raise P43Error(f"{entry}: is not a regular file (symlinks and special files are "
                           f"rejected)")
        observed_files.add(entry)
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
                 revalidator=default_revalidator) -> int:
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

    try:
        resolved_output = resolve_output_root(output_root, repo_root)
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
        documents = build_documents(orchestrator, p35, records)
        outcomes = publish_documents(orchestrator, resolved_output, documents, write=write)
        assert_output_tree_exact(resolved_output)
    except (P43Error, p42.EvidenceError) as exc:
        print(f"{prefix}: FAILED: {exc}", file=sys.stderr)
        return 1

    relative_root = os.path.relpath(str(resolved_output), str(repo_root))
    for relative in ARTIFACT_RELATIVE_PATHS:
        print(f"{prefix}: {outcomes[relative]}: {relative_root}/{relative}")
    mode = "analyze" if write else "verify"
    print(f"{prefix}: {mode}: OK ({CAMPAIGN_COUNT} final campaigns; the accepted pilot "
          f"{PILOT_CAMPAIGN_ID} was excluded from every statistic; no raw evidence was "
          f"modified; {PUBLICATION_STATUS})")
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
                    gpu_uuid: str = "GPU-11111111-2222-3333-4444-555555555555") -> dict:
    """One synthetic, obviously fake campaign record shaped exactly like a
    parsed one. No real evidence is involved."""
    pilot_statistics = {}
    for index, key in enumerate(P14_CONFIG_KEYS):
        pilot_statistics[key] = {
            "median_gbps": (3000.0 + 100.0 * index) * gbps_scale,
            "within_campaign_cv_percent": 0.1,
            "within_campaign_sample_count": 30,
            "within_campaign_stability_review": "ok",
        }
    pairwise = {key: {"tma_to_ldgsts_ratio": (0.97 + 0.001 * index) * ratio_scale,
                      "interpretation": "ldgsts_higher"}
                for index, key in enumerate(P14_PAIR_KEYS)}
    saturation = {key: saturation_kib for key in P14_SATURATION_KEYS}
    ncu = {key: {"dram_read_ratio": 1.0 + 0.0001 * index,
                 "hbm_classification": "HBM_VALIDATED", "diagnostic_flags": ""}
           for index, key in enumerate(P14_NCU_CASES)}
    configuration = {}
    for index, key in enumerate(P24_CONFIG_KEYS):
        total = (2000.0 + 100.0 * index) * fpc_scale
        configuration[key] = {
            "cta_group": P24_CTA_GROUP[key[0]],
            "median_flops_per_cycle": total,
            "median_flops_per_cycle_per_sm": total / P24_CTA_GROUP[key[0]],
            "within_campaign_cv_percent": 0.01,
            "within_campaign_sample_count": 30,
        }
    scaling = {key: {"speedup_2sm_over_1sm": 1.5 + 0.01 * index,
                     "scaling_efficiency_percent": 75.0 + 0.5 * index,
                     "surprising_value_flag": "False"}
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
                   abs(summary["cv_percent"] - 50.0) < 1e-12, str(summary["cv_percent"]))
    reporter.check("a coefficient of variation above the strict 5% threshold is flagged for "
                   "review and excludes nothing",
                   summary["cv_review_flag"] == CV_FLAG_REVIEW
                   and summary["campaign_count"] == CAMPAIGN_COUNT
                   and len(summary["campaign_values"]) == CAMPAIGN_COUNT, str(summary))
    calm = summarize_metric([100.0, 100.5, 100.25], metric="median_effective_gbps")
    reporter.check("a low coefficient of variation is labelled ok",
                   calm["cv_review_flag"] == CV_FLAG_OK, str(calm))
    signed = summarize_metric([-1.0, 0.0, 1.0], metric="gap_to_cublaslt_pct")
    reporter.check("no coefficient of variation is computed for a signed or zero-centred "
                   "metric",
                   signed["cv_percent"] is None
                   and signed["cv_review_flag"] == NOT_APPLICABLE, str(signed))
    reporter.check("a negative gap is preserved without clamping",
                   signed["minimum"] == -1.0, str(signed))
    zero = summarize_metric([0.0, 0.0, 0.0], metric="throughput_ratio_vs_cublaslt")
    reporter.check("a zero denominator never produces a coefficient of variation",
                   zero["cv_percent"] is None, str(zero))
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
    documents = build_documents(orchestrator, p35, records)
    inventory = [relative for relative, _ in documents]
    reporter.check("the analysis produces exactly the frozen artifact inventory",
                   inventory == list(ARTIFACT_RELATIVE_PATHS), str(inventory))
    again = build_documents(orchestrator, p35, _fixture_records())
    reporter.check("two independent runs over identical evidence are byte-identical",
                   [payload for _, payload in documents]
                   == [payload for _, payload in again], "")

    summary = json.loads(dict(documents)["integrated_summary.json"].decode("utf-8"))
    reporter.check("the pilot is recorded only as excluded qualification provenance",
                   summary["population"]["pilot_campaign_id_excluded"] == PILOT_CAMPAIGN_ID
                   and PILOT_CAMPAIGN_ID not in summary["population"]["final_campaign_ids"],
                   "")
    reporter.check("every artifact records publishable=false",
                   summary["publishable"] is False, "")
    text = dict(documents)["memory_paths.csv"].decode("utf-8")
    reporter.check("no output artifact mentions the excluded pilot as data",
                   PILOT_CAMPAIGN_ID not in text, "")
    reporter.check("the memory table carries one row per frozen configuration, ratio pair, "
                   "saturation group, and profiled case",
                   len(text.strip().split("\n")) == 1 + len(P14_CONFIG_KEYS)
                   + len(P14_PAIR_KEYS) + len(P14_SATURATION_KEYS) + len(P14_NCU_CASES), "")

    gap_rows = [row for row in csv.DictReader(io.StringIO(
        dict(documents)["gemm_comparison.csv"].decode("utf-8")))
        if row["metric"] == "gap_to_cublaslt_pct"]
    reporter.check("the signed GEMM gap never carries a coefficient of variation",
                   all(row["cv_percent"] == NOT_APPLICABLE for row in gap_rows), "")
    reporter.check("negative GEMM gaps are preserved unclamped",
                   any(float(row["campaign_1_value"]) < 0 for row in gap_rows), "")

    # Disagreements are reported, never resolved.
    mixed = build_documents(orchestrator, p35, _fixture_records(**{
        FINAL_CAMPAIGN_IDS[1]: {"saturation_kib": 32, "depth_saturation": 64,
                                "ceiling_depth": 64, "best_variant": "persistent_1cta"}}))
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
        FINAL_CAMPAIGN_IDS[2]: {"gbps_scale": 1.5}}))
    noisy_summary = json.loads(dict(noisy)["integrated_summary.json"].decode("utf-8"))
    flagged = [entry for entry in noisy_summary["experiment_1_memory_paths"]["configurations"]
               if entry["cv_review_flag"] == CV_FLAG_REVIEW]
    reporter.check("high cross-campaign variability is flagged for review and still keeps "
                   "all three campaign values",
                   flagged and all(len(entry["campaign_values"]) == CAMPAIGN_COUNT
                                   for entry in flagged), str(len(flagged)))

    reporter.rejects("a record set that is not the frozen final population is rejected",
                     lambda: build_documents(orchestrator, p35, records[:2]),
                     "exactly 3 final campaign records")
    reporter.rejects("records in the wrong order are rejected",
                     lambda: build_documents(orchestrator, p35,
                                             [records[1], records[0], records[2]]),
                     "frozen final population in order")
    pilot_record = _fixture_record(PILOT_CAMPAIGN_ID)
    reporter.rejects("the accepted pilot's record is never aggregated",
                     lambda: build_documents(orchestrator, p35,
                                             [pilot_record] + records[1:]),
                     "frozen final population in order")


def _self_test_publication(reporter: _Reporter, orchestrator, root: Path) -> None:
    p35 = _StubP35()
    documents = build_documents(orchestrator, p35, _fixture_records())
    repo = root / "repo"
    (repo / "results").mkdir(parents=True)
    output = repo / "results" / "phase4"

    resolved = resolve_output_root(output, repo)
    publish_documents(orchestrator, resolved, documents, write=True)
    assert_output_tree_exact(resolved)
    reporter.check("the frozen inventory publishes into a clean output tree", True)

    outcomes = publish_documents(orchestrator, resolved, documents, write=True)
    reporter.check("an existing byte-identical artifact is verified, never rewritten",
                   all(value == "verified_byte_identical" for value in outcomes.values()),
                   str(outcomes))

    (resolved / "report.md").write_text("tampered\n", encoding="utf-8")
    reporter.rejects("a differing existing artifact is never overwritten",
                     lambda: publish_documents(orchestrator, resolved, documents, write=True),
                     "refusing to overwrite")
    reporter.rejects("verification fails on a tampered artifact",
                     lambda: publish_documents(orchestrator, resolved, documents,
                                               write=False),
                     "different content")
    (resolved / "report.md").unlink()
    reporter.rejects("verification fails on a missing artifact",
                     lambda: publish_documents(orchestrator, resolved, documents,
                                               write=False),
                     "is missing")
    publish_documents(orchestrator, resolved, documents, write=True)

    (resolved / "unexpected.csv").write_text("x\n", encoding="utf-8")
    reporter.rejects("an unexpected artifact in the output tree is rejected",
                     lambda: assert_output_tree_exact(resolved), "unexpected=")
    (resolved / "unexpected.csv").unlink()
    assert_output_tree_exact(resolved)

    os.symlink(resolved / "report.md", resolved / "figures" / "linked.svg")
    reporter.rejects("a symlink inside the output tree is rejected",
                     lambda: assert_output_tree_exact(resolved), "not a regular file")
    (resolved / "figures" / "linked.svg").unlink()

    linked_repo = root / "linked"
    (linked_repo / "results").mkdir(parents=True)
    os.symlink(resolved, linked_repo / "results" / "phase4")
    reporter.rejects("a symlinked output root is rejected",
                     lambda: publish_documents(
                         orchestrator, resolve_output_root(
                             linked_repo / "results" / "phase4", linked_repo),
                         documents, write=True),
                     "symlink")

    for forbidden in ("results/raw/phase4", "results/raw", "results/preflight"):
        reporter.rejects(f"an output root under {forbidden} is refused",
                         lambda path=forbidden: resolve_output_root(repo / path, repo),
                         "never writes under")
    reporter.rejects("an output root outside the repository is refused",
                     lambda: resolve_output_root(root / "elsewhere", repo), "outside")
    reporter.check("no artifact was written under results/raw/",
                   not (repo / "results" / "raw").exists(), "")


def _self_test_pipeline(reporter: _Reporter, root: Path) -> None:
    """The evidence seam: a failed population revalidation must abort before a
    single scientific value is read or a single byte is written."""
    repo = root / "pipeline"
    (repo / "results" / "raw" / "phase4").mkdir(parents=True)
    calls: list[tuple] = []

    def refusing(orchestrator, p42, campaign_root, pilot_ids, final_ids):
        calls.append((tuple(pilot_ids), tuple(final_ids)))
        return 1

    status = run_analysis(DEFAULT_REPO_ROOT, repo / "results" / "raw" / "phase4",
                          [PILOT_CAMPAIGN_ID], list(FINAL_CAMPAIGN_IDS),
                          repo / "results" / "phase4", write=True, revalidator=refusing)
    reporter.check("a failed P4.2 population revalidation aborts the whole analysis",
                   status == 1, str(status))
    reporter.check("the revalidation received the declared pilot and the three finals",
                   calls == [((PILOT_CAMPAIGN_ID,), FINAL_CAMPAIGN_IDS)], str(calls))
    reporter.check("nothing was written when revalidation failed",
                   not (repo / "results" / "phase4").exists(), "")

    status = run_analysis(DEFAULT_REPO_ROOT, repo / "results" / "raw" / "phase4",
                          [PILOT_CAMPAIGN_ID], list(FINAL_CAMPAIGN_IDS)[:2],
                          repo / "results" / "phase4", write=True, revalidator=refusing)
    reporter.check("an incomplete declared population is refused before any revalidation",
                   status == 1 and len(calls) == 1, str(calls))


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
        _self_test_pipeline(reporter, root)
        _self_test_evidence(reporter, orchestrator, p35, root)
    reporter.check("the frozen artifact inventory is exactly nine artifacts",
                   len(ARTIFACT_RELATIVE_PATHS) == 9, str(ARTIFACT_RELATIVE_PATHS))
    reporter.check("the frozen population is one excluded pilot plus three finals",
                   CAMPAIGN_COUNT == 3 and PILOT_CAMPAIGN_ID not in FINAL_CAMPAIGN_IDS, "")
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
