#!/usr/bin/env python3
"""P1.3 plan generation, CSV validation, consolidation, and aggregation for
the exp01_memory_paths joint LDGSTS/TMA sweep.

This script never touches CUDA, Docker, ``nvidia-smi``, either benchmark
binary, or the network. ``scripts/run_exp01_memory_paths.sh`` is the only
thing that invokes GPU work (exclusively through ``scripts/run_container.sh``
for anything that runs inside a container); this script only plans the
18-invocation sweep, validates the raw 37-column CSV the two P1.1/P1.2
binaries already emit, consolidates it losslessly, computes descriptive
per-configuration statistics, and reads/writes the campaign manifest.

P1.3 produces functional/descriptive infrastructure output only: it does not
compute LDGSTS/TMA speedups, run Nsight Compute, judge outliers, or draw any
performance conclusion. See src/memory/README.md and PLAN.md.

Subcommands:
  plan          Print the frozen deterministic 18-invocation plan.
  capture       Run one binary invocation inside the container, capturing its
                stdout to a temporary CSV and renaming it atomically only on
                success (used by run_exp01_memory_paths.sh via
                scripts/run_container.sh; never touches the network itself).
  validate-case Strictly validate one already-captured case CSV file.
  finalize      Re-validate an entire campaign, then write
                combined_samples.csv, summary.csv, and a COMPLETE manifest.
  manifest-write  Merge a small JSON fragment into manifest.json and set its
                  status (used for IN_PROGRESS/FAILED/INTERRUPTED updates).
  --self-test   GPU-free synthetic positive/negative tests (no subprocess,
                no CUDA, no Docker, no nvidia-smi, no network). Prints
                "aggregate_exp01_memory_paths: SELF_TEST_RESULT=PASS" only if
                every case passes.

Exit codes: 0 on success (including --self-test passing); 1 on a validation,
aggregation, or capture failure; 2 on a usage error.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SCHEMA_VERSION = "1"
MANIFEST_SCHEMA_VERSION = "1"
EXPERIMENT_ID = "exp01_memory_paths"

METHODS = ("ldgsts", "tma")
CONFIG_PAIRS = (
    (2, 16), (2, 32), (2, 64),
    (4, 16), (4, 32), (4, 64),
    (8, 16), (8, 32), (8, 64),
)
EXPECTED_CONFIGURATION_COUNT = 18

RAW_ROOT_REL = Path("results/raw/exp01_memory_paths")

MEMORY_LDGSTS_BIN = REPO_ROOT / "build/memory/ldgsts"
MEMORY_LDGSTS_SASS = REPO_ROOT / "build/memory/ldgsts.sass"
MEMORY_TMA_BIN = REPO_ROOT / "build/memory/tma"
MEMORY_TMA_SASS = REPO_ROOT / "build/memory/tma.sass"

# Frozen contract constants shared by both binaries (see src/memory/README.md).
FROZEN_THREADS_PER_CTA = 128
FROZEN_TARGET_CTAS_PER_SM = 1
FROZEN_OCCUPANCY_CTAS_PER_SM = 1
FROZEN_TILE_WIDTH_ELEMENTS = 128
FROZEN_TILE_WIDTH_BYTES = 256
FROZEN_VECTOR_BYTES = 16
FROZEN_COMPUTE_CAPABILITY = "10.3"

# Exact 37-column header both build/memory/ldgsts and build/memory/tma print.
CSV_HEADER = [
    "schema_version", "timestamp_utc", "run_kind", "method", "sample_index",
    "stages", "tile_width_elements", "tile_width_bytes", "tile_height",
    "stage_bytes", "bytes_in_flight_per_sm", "vector_bytes",
    "copies_per_thread_per_stage", "threads_per_cta", "target_ctas_per_sm",
    "occupancy_ctas_per_sm", "grid_blocks", "sm_count",
    "smem_reservation_bytes", "l2_bytes", "requested_working_set_bytes",
    "working_set_bytes", "working_set_l2_ratio", "passes", "useful_bytes",
    "warmup_ms", "kernel_time_ms", "effective_gbps", "correctness",
    "mismatches", "gpu_name", "gpu_uuid", "compute_capability",
    "cuda_driver_version", "cuda_runtime_version", "git_commit", "git_dirty",
]
assert len(CSV_HEADER) == 37, f"CSV_HEADER has {len(CSV_HEADER)} columns, expected 37"

CASE_NAME_RE = re.compile(r"^(\d{2})_(ldgsts|tma)_s(\d+)_bif(\d+)$")

# Documented tolerances for values that pass through the binaries' fixed
# six-decimal CSV formatting (std::fixed << std::setprecision(6)).
RATIO_ABS_TOL = 1e-6
GBPS_REL_TOL = 1e-3

INT64_MAX = 2**63 - 1


# ---------------------------------------------------------------------------
# Frozen geometry formulas (single source of truth, mirrors src/memory/*.cu).
# ---------------------------------------------------------------------------
def stage_bytes_of(stages: int, bif_kib: int) -> int:
    return (bif_kib * 1024) // stages


def bytes_in_flight_of(bif_kib: int) -> int:
    return bif_kib * 1024


def copies_per_thread_of(stages: int, bif_kib: int) -> int:
    return stage_bytes_of(stages, bif_kib) // (FROZEN_THREADS_PER_CTA * FROZEN_VECTOR_BYTES)


def tile_height_of(stages: int, bif_kib: int) -> int:
    return stage_bytes_of(stages, bif_kib) // FROZEN_TILE_WIDTH_BYTES


# ---------------------------------------------------------------------------
# Plan generation (frozen 18-invocation contract, section 6).
# ---------------------------------------------------------------------------
def build_plan() -> list[dict]:
    plan = []
    index = 0
    for pair_num, (stages, bif_kib) in enumerate(CONFIG_PAIRS):
        order = METHODS if pair_num % 2 == 0 else (METHODS[1], METHODS[0])
        for method in order:
            case_name = f"{index:02d}_{method}_s{stages}_bif{bif_kib}"
            plan.append({
                "index": index,
                "method": method,
                "stages": stages,
                "bif_kib": bif_kib,
                "case_name": case_name,
            })
            index += 1
    return plan


def plan_by_index(plan: list[dict]) -> dict[int, dict]:
    return {entry["index"]: entry for entry in plan}


def check_plan_contract(plan: list[dict]) -> list[str]:
    """Independently re-derives every property build_plan() is supposed to
    guarantee, so a future edit to build_plan() cannot silently break the
    frozen contract without failing --self-test."""
    errors: list[str] = []

    if len(plan) != EXPECTED_CONFIGURATION_COUNT:
        errors.append(f"plan has {len(plan)} invocations, expected {EXPECTED_CONFIGURATION_COUNT}")

    indices = [entry["index"] for entry in plan]
    if indices != list(range(len(plan))):
        errors.append(f"plan indices are not exactly 0..{len(plan) - 1} in order: {indices}")

    seen_keys: set[tuple[str, int, int]] = set()
    for entry in plan:
        key = (entry["method"], entry["stages"], entry["bif_kib"])
        if key in seen_keys:
            errors.append(f"duplicate invocation for {key}")
        seen_keys.add(key)

    for method in METHODS:
        count = sum(1 for entry in plan if entry["method"] == method)
        if count != 9:
            errors.append(f"method {method!r} appears {count} times, expected 9")

    for stages, bif_kib in CONFIG_PAIRS:
        methods_here = sorted(
            entry["method"] for entry in plan
            if entry["stages"] == stages and entry["bif_kib"] == bif_kib
        )
        if methods_here != sorted(METHODS):
            errors.append(
                f"config stages={stages} bif_kib={bif_kib} has methods={methods_here}, "
                f"expected exactly one of each method"
            )

    if len(plan) == EXPECTED_CONFIGURATION_COUNT:
        for pair_num, (stages, bif_kib) in enumerate(CONFIG_PAIRS):
            first, second = plan[pair_num * 2]["method"], plan[pair_num * 2 + 1]["method"]
            expected = ("ldgsts", "tma") if pair_num % 2 == 0 else ("tma", "ldgsts")
            if (first, second) != expected:
                errors.append(
                    f"pair {pair_num} (stages={stages}, bif_kib={bif_kib}) order "
                    f"{(first, second)} != expected {expected}"
                )
            pair_first_entry = plan[pair_num * 2]
            pair_second_entry = plan[pair_num * 2 + 1]
            if (pair_first_entry["stages"], pair_first_entry["bif_kib"]) != (stages, bif_kib):
                errors.append(f"pair {pair_num} first entry has wrong configuration")
            if (pair_second_entry["stages"], pair_second_entry["bif_kib"]) != (stages, bif_kib):
                errors.append(f"pair {pair_num} second entry has wrong configuration")

    return errors


def format_plan_text(plan: list[dict]) -> str:
    lines = [
        "index  method   stages  bif_kib  stage_bytes  tile_height  copies/thread/stage  case_name",
    ]
    for entry in plan:
        lines.append(
            f"{entry['index']:>5d}  {entry['method']:<7s}  {entry['stages']:>6d}  "
            f"{entry['bif_kib']:>7d}  {stage_bytes_of(entry['stages'], entry['bif_kib']):>11d}  "
            f"{tile_height_of(entry['stages'], entry['bif_kib']):>11d}  "
            f"{copies_per_thread_of(entry['stages'], entry['bif_kib']):>19d}  {entry['case_name']}"
        )
    lines.append(f"total invocations: {len(plan)}")
    return "\n".join(lines) + "\n"


def format_plan_lines(plan: list[dict]) -> str:
    return "".join(
        f"{entry['index']}\t{entry['method']}\t{entry['stages']}\t{entry['bif_kib']}\t{entry['case_name']}\n"
        for entry in plan
    )


# ---------------------------------------------------------------------------
# Path safety (capture subcommand): never escape the current campaign dir.
# ---------------------------------------------------------------------------
class UnsafePathError(ValueError):
    pass


def resolve_campaign_dir(campaign_dir_rel: str) -> Path:
    if os.path.isabs(campaign_dir_rel):
        raise UnsafePathError(f"--campaign-dir must be relative, got absolute path {campaign_dir_rel!r}")
    if ".." in Path(campaign_dir_rel).parts:
        raise UnsafePathError(f"--campaign-dir must not contain '..': {campaign_dir_rel!r}")
    raw_root = (REPO_ROOT / RAW_ROOT_REL).resolve()
    campaign_dir = (REPO_ROOT / campaign_dir_rel).resolve()
    if campaign_dir != raw_root and raw_root not in campaign_dir.parents:
        raise UnsafePathError(
            f"--campaign-dir {campaign_dir_rel!r} resolves outside {RAW_ROOT_REL}/"
        )
    if not campaign_dir.is_dir():
        raise UnsafePathError(f"campaign directory does not exist: {campaign_dir}")
    return campaign_dir


def resolve_capture_out_path(campaign_dir: Path, out_rel: str) -> Path:
    """Validates --out for the capture subcommand: relative, no traversal,
    resolves strictly beneath campaign_dir (no symlink escape), and must not
    already exist."""
    if os.path.isabs(out_rel):
        raise UnsafePathError(f"--out must be relative, got absolute path {out_rel!r}")
    if ".." in Path(out_rel).parts:
        raise UnsafePathError(f"--out must not contain '..': {out_rel!r}")
    campaign_dir_resolved = campaign_dir.resolve()
    out_path = (campaign_dir / out_rel).resolve()
    if out_path != campaign_dir_resolved and campaign_dir_resolved not in out_path.parents:
        raise UnsafePathError(f"--out {out_rel!r} resolves outside the current campaign directory")
    if out_path.exists():
        raise UnsafePathError(f"refusing to overwrite existing target: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Strict CSV validation (section 9).
# ---------------------------------------------------------------------------
def _parse_strict_int(raw: str, errors: list[str], ctx: str, field: str) -> int | None:
    try:
        value = int(raw)
    except ValueError:
        errors.append(f"{ctx}: {field}={raw!r} is not an integer")
        return None
    if not (-INT64_MAX - 1 <= value <= INT64_MAX):
        errors.append(f"{ctx}: {field}={value} is outside the int64 range")
        return None
    return value


def _parse_strict_float(raw: str, errors: list[str], ctx: str, field: str) -> float | None:
    try:
        value = float(raw)
    except ValueError:
        errors.append(f"{ctx}: {field}={raw!r} is not a float")
        return None
    if math.isnan(value):
        errors.append(f"{ctx}: {field} is NaN")
        return None
    if math.isinf(value):
        errors.append(f"{ctx}: {field} is infinite")
        return None
    return value


def read_case_rows(path: Path) -> tuple[list[list[str]], list[str]]:
    """Reads the raw CSV with csv.reader (never cut/awk/line-splitting).
    Returns (all_rows_including_header, errors). On a structural error the
    row list may be incomplete."""
    errors: list[str] = []
    try:
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
    except OSError as exc:
        return [], [f"{path}: unable to read: {exc}"]
    if not rows:
        return [], [f"{path}: file is empty (no header)"]
    return rows, errors


def validate_case_file(path: Path, expect: dict) -> tuple[list[dict[str, str]], list[str]]:
    """Validates one case CSV in full per section 9. `expect` must contain:
    method, stages, bif_kib, run_kind, repetitions, passes, warmup_ms,
    git_commit. Returns (parsed_data_rows, errors); errors is empty iff the
    file is fully valid."""
    errors: list[str] = []
    rows, read_errors = read_case_rows(path)
    if read_errors:
        return [], read_errors

    header, data_rows_raw = rows[0], rows[1:]
    if header != CSV_HEADER:
        return [], [f"{path}: header mismatch (wrong or reordered CSV header): {header!r}"]

    parsed_rows: list[dict[str, str]] = []
    for line_no, row in enumerate(data_rows_raw, start=2):
        if len(row) == 0:
            errors.append(f"{path}: line {line_no}: blank row")
            continue
        if row == CSV_HEADER:
            errors.append(f"{path}: line {line_no}: repeated header row")
            continue
        if len(row) != len(CSV_HEADER):
            errors.append(
                f"{path}: line {line_no}: expected {len(CSV_HEADER)} fields, got {len(row)}"
            )
            continue
        parsed_rows.append(dict(zip(CSV_HEADER, row)))

    if errors:
        return parsed_rows, errors

    repetitions = expect["repetitions"]
    if len(parsed_rows) != repetitions:
        errors.append(
            f"{path}: has {len(parsed_rows)} data row(s), expected exactly repetitions={repetitions}"
        )
        return parsed_rows, errors

    expected_stage_bytes = stage_bytes_of(expect["stages"], expect["bif_kib"])
    expected_bif_bytes = bytes_in_flight_of(expect["bif_kib"])
    expected_tile_height = tile_height_of(expect["stages"], expect["bif_kib"])
    expected_copies = copies_per_thread_of(expect["stages"], expect["bif_kib"])

    sample_indices: list[int] = []
    for row_num, row in enumerate(parsed_rows):
        ctx = f"{path}: data row {row_num} (line {row_num + 2})"

        if row["schema_version"] != SCHEMA_VERSION:
            errors.append(f"{ctx}: schema_version={row['schema_version']!r} != {SCHEMA_VERSION!r}")
        if row["method"] != expect["method"]:
            errors.append(f"{ctx}: method={row['method']!r} != {expect['method']!r}")
        if row["run_kind"] != expect["run_kind"]:
            errors.append(f"{ctx}: run_kind={row['run_kind']!r} != {expect['run_kind']!r}")
        if row["correctness"] != "OK":
            errors.append(f"{ctx}: correctness={row['correctness']!r} != 'OK'")

        sample_index = _parse_strict_int(row["sample_index"], errors, ctx, "sample_index")
        if sample_index is not None:
            sample_indices.append(sample_index)

        mismatches = _parse_strict_int(row["mismatches"], errors, ctx, "mismatches")
        if mismatches is not None and mismatches != 0:
            errors.append(f"{ctx}: mismatches={mismatches} != 0")

        stages = _parse_strict_int(row["stages"], errors, ctx, "stages")
        if stages is not None and stages != expect["stages"]:
            errors.append(f"{ctx}: stages={stages} != {expect['stages']}")

        target_ctas = _parse_strict_int(row["target_ctas_per_sm"], errors, ctx, "target_ctas_per_sm")
        if target_ctas is not None and target_ctas != FROZEN_TARGET_CTAS_PER_SM:
            errors.append(f"{ctx}: target_ctas_per_sm={target_ctas} != {FROZEN_TARGET_CTAS_PER_SM}")

        occupancy = _parse_strict_int(row["occupancy_ctas_per_sm"], errors, ctx, "occupancy_ctas_per_sm")
        if occupancy is not None and occupancy != FROZEN_OCCUPANCY_CTAS_PER_SM:
            errors.append(f"{ctx}: occupancy_ctas_per_sm={occupancy} != {FROZEN_OCCUPANCY_CTAS_PER_SM}")

        threads_per_cta = _parse_strict_int(row["threads_per_cta"], errors, ctx, "threads_per_cta")
        if threads_per_cta is not None and threads_per_cta != FROZEN_THREADS_PER_CTA:
            errors.append(f"{ctx}: threads_per_cta={threads_per_cta} != {FROZEN_THREADS_PER_CTA}")

        grid_blocks = _parse_strict_int(row["grid_blocks"], errors, ctx, "grid_blocks")
        sm_count = _parse_strict_int(row["sm_count"], errors, ctx, "sm_count")
        if grid_blocks is not None and sm_count is not None and grid_blocks != sm_count:
            errors.append(f"{ctx}: grid_blocks={grid_blocks} != sm_count={sm_count}")

        if row["compute_capability"] != FROZEN_COMPUTE_CAPABILITY:
            errors.append(
                f"{ctx}: compute_capability={row['compute_capability']!r} != {FROZEN_COMPUTE_CAPABILITY!r}"
            )

        tile_width_elements = _parse_strict_int(row["tile_width_elements"], errors, ctx, "tile_width_elements")
        if tile_width_elements is not None and tile_width_elements != FROZEN_TILE_WIDTH_ELEMENTS:
            errors.append(f"{ctx}: tile_width_elements={tile_width_elements} != {FROZEN_TILE_WIDTH_ELEMENTS}")

        tile_width_bytes = _parse_strict_int(row["tile_width_bytes"], errors, ctx, "tile_width_bytes")
        if tile_width_bytes is not None and tile_width_bytes != FROZEN_TILE_WIDTH_BYTES:
            errors.append(f"{ctx}: tile_width_bytes={tile_width_bytes} != {FROZEN_TILE_WIDTH_BYTES}")

        vector_bytes = _parse_strict_int(row["vector_bytes"], errors, ctx, "vector_bytes")
        if vector_bytes is not None and vector_bytes != FROZEN_VECTOR_BYTES:
            errors.append(f"{ctx}: vector_bytes={vector_bytes} != {FROZEN_VECTOR_BYTES}")

        stage_bytes = _parse_strict_int(row["stage_bytes"], errors, ctx, "stage_bytes")
        if stage_bytes is not None and stage_bytes != expected_stage_bytes:
            errors.append(f"{ctx}: stage_bytes={stage_bytes} != {expected_stage_bytes} (formula)")

        bif_bytes = _parse_strict_int(row["bytes_in_flight_per_sm"], errors, ctx, "bytes_in_flight_per_sm")
        if bif_bytes is not None and bif_bytes != expected_bif_bytes:
            errors.append(f"{ctx}: bytes_in_flight_per_sm={bif_bytes} != {expected_bif_bytes} (formula)")

        tile_height = _parse_strict_int(row["tile_height"], errors, ctx, "tile_height")
        if tile_height is not None and tile_height != expected_tile_height:
            errors.append(f"{ctx}: tile_height={tile_height} != {expected_tile_height} (formula)")

        copies = _parse_strict_int(row["copies_per_thread_per_stage"], errors, ctx, "copies_per_thread_per_stage")
        if copies is not None and copies != expected_copies:
            errors.append(f"{ctx}: copies_per_thread_per_stage={copies} != {expected_copies} (formula)")

        passes = _parse_strict_int(row["passes"], errors, ctx, "passes")
        if passes is not None and passes != expect["passes"]:
            errors.append(f"{ctx}: passes={passes} != requested {expect['passes']}")

        warmup_ms = _parse_strict_int(row["warmup_ms"], errors, ctx, "warmup_ms")
        if warmup_ms is not None and warmup_ms != expect["warmup_ms"]:
            errors.append(f"{ctx}: warmup_ms={warmup_ms} != requested {expect['warmup_ms']}")

        working_set_bytes = _parse_strict_int(row["working_set_bytes"], errors, ctx, "working_set_bytes")
        l2_bytes = _parse_strict_int(row["l2_bytes"], errors, ctx, "l2_bytes")
        useful_bytes = _parse_strict_int(row["useful_bytes"], errors, ctx, "useful_bytes")
        if working_set_bytes is not None and passes is not None and useful_bytes is not None:
            if useful_bytes != working_set_bytes * passes:
                errors.append(
                    f"{ctx}: useful_bytes={useful_bytes} != working_set_bytes*passes="
                    f"{working_set_bytes * passes} (formula)"
                )
        if expect["run_kind"] == "benchmark" and working_set_bytes is not None and l2_bytes is not None:
            if not (working_set_bytes > 2 * l2_bytes):
                errors.append(
                    f"{ctx}: working_set_bytes={working_set_bytes} is not > 2*l2_bytes={2 * l2_bytes} "
                    f"(required for run_kind=benchmark)"
                )

        ratio = _parse_strict_float(row["working_set_l2_ratio"], errors, ctx, "working_set_l2_ratio")
        if ratio is not None and working_set_bytes is not None and l2_bytes not in (None, 0):
            expected_ratio = working_set_bytes / l2_bytes
            if abs(ratio - expected_ratio) > RATIO_ABS_TOL:
                errors.append(
                    f"{ctx}: working_set_l2_ratio={ratio} inconsistent with "
                    f"working_set_bytes/l2_bytes={expected_ratio} (tol={RATIO_ABS_TOL})"
                )

        kernel_time_ms = _parse_strict_float(row["kernel_time_ms"], errors, ctx, "kernel_time_ms")
        if kernel_time_ms is not None and kernel_time_ms <= 0:
            errors.append(f"{ctx}: kernel_time_ms={kernel_time_ms} is not positive")

        effective_gbps = _parse_strict_float(row["effective_gbps"], errors, ctx, "effective_gbps")
        if effective_gbps is not None and effective_gbps <= 0:
            errors.append(f"{ctx}: effective_gbps={effective_gbps} is not positive")

        if (
            kernel_time_ms is not None and kernel_time_ms > 0
            and effective_gbps is not None and useful_bytes is not None
        ):
            recomputed = useful_bytes / (kernel_time_ms / 1000.0) / 1e9
            if recomputed > 0:
                rel_err = abs(recomputed - effective_gbps) / recomputed
                if rel_err > GBPS_REL_TOL:
                    errors.append(
                        f"{ctx}: effective_gbps={effective_gbps} inconsistent with "
                        f"useful_bytes/kernel_time={recomputed} (rel_err={rel_err}, tol={GBPS_REL_TOL})"
                    )

        if row["git_commit"] != expect["git_commit"]:
            errors.append(f"{ctx}: git_commit={row['git_commit']!r} != expected {expect['git_commit']!r}")
        if row["git_dirty"] != "false":
            errors.append(f"{ctx}: git_dirty={row['git_dirty']!r} != 'false'")

    index_counts = {}
    for value in sample_indices:
        index_counts[value] = index_counts.get(value, 0) + 1
    expected_indices = set(range(repetitions))
    found_indices = set(index_counts)
    for missing in sorted(expected_indices - found_indices):
        errors.append(f"{path}: sample_index={missing} is missing")
    for unexpected in sorted(found_indices - expected_indices):
        errors.append(f"{path}: unexpected sample_index={unexpected} (expected 0..{repetitions - 1})")
    for value, count in sorted(index_counts.items()):
        if count > 1:
            errors.append(f"{path}: sample_index={value} appears {count} times, expected exactly once")

    return parsed_rows, errors


COMMON_FIELDS = (
    "gpu_name", "gpu_uuid", "compute_capability", "cuda_driver_version",
    "cuda_runtime_version", "git_commit", "git_dirty", "sm_count", "l2_bytes",
    "working_set_bytes", "passes", "warmup_ms", "run_kind",
)


def check_cross_case_consistency(cases: list[tuple[dict, list[dict[str, str]]]]) -> list[str]:
    """cases: list of (plan_entry, parsed_rows), already individually valid.
    Checks the fields required to be identical across the whole campaign
    (section 9's "Across the complete campaign require identical" list),
    deliberately excluding smem_reservation_bytes."""
    errors: list[str] = []
    if not cases:
        return errors
    reference_entry, reference_rows = cases[0]
    reference_row = reference_rows[0]
    reference_repetitions = len(reference_rows)
    for entry, rows in cases[1:]:
        row = rows[0]
        for field in COMMON_FIELDS:
            if row[field] != reference_row[field]:
                errors.append(
                    f"case {entry['case_name']}: {field}={row[field]!r} != "
                    f"{reference_row['case_name'] if False else reference_entry['case_name']}'s "
                    f"{field}={reference_row[field]!r}"
                )
        if len(rows) != reference_repetitions:
            errors.append(
                f"case {entry['case_name']}: repetition count {len(rows)} != "
                f"{reference_entry['case_name']}'s {reference_repetitions}"
            )
    return errors


def scan_case_directory(cases_dir: Path, plan: list[dict]) -> tuple[dict[int, Path], list[str]]:
    """Scans cases_dir for files matching the canonical NN_method_sS_bifB.csv
    naming convention, cross-checks the parsed filename against the plan, and
    returns (index -> path for exactly the 18 expected indices, errors).
    Anything not matching the canonical pattern, or not present, is reported
    as an error rather than silently ignored or aggregated (section 9/12)."""
    errors: list[str] = []
    by_index = plan_by_index(plan)
    found: dict[int, Path] = {}

    if not cases_dir.is_dir():
        return {}, [f"{cases_dir}: cases directory does not exist"]

    for path in sorted(cases_dir.iterdir()):
        if path.suffix != ".csv":
            continue  # .partial / .invalid / anything else is never aggregated
        match = CASE_NAME_RE.match(path.stem)
        if not match:
            errors.append(f"{path}: does not match the canonical case-name pattern; unexpected file")
            continue
        index, method, stages, bif_kib = (
            int(match.group(1)), match.group(2), int(match.group(3)), int(match.group(4)),
        )
        expected_entry = by_index.get(index)
        if expected_entry is None:
            errors.append(f"{path}: index {index} is outside the expected 0..17 range; unexpected file")
            continue
        if index in found:
            errors.append(f"{path}: duplicate configuration for index {index} (already have {found[index]})")
            continue
        if (method, stages, bif_kib) != (
            expected_entry["method"], expected_entry["stages"], expected_entry["bif_kib"],
        ):
            errors.append(
                f"{path}: filename implies method={method} stages={stages} bif_kib={bif_kib}, "
                f"but plan index {index} expects method={expected_entry['method']} "
                f"stages={expected_entry['stages']} bif_kib={expected_entry['bif_kib']}"
            )
            continue
        found[index] = path

    for index in sorted(by_index):
        if index not in found:
            errors.append(
                f"{cases_dir}: missing configuration index {index} "
                f"({by_index[index]['case_name']}.csv)"
            )

    return found, errors


# ---------------------------------------------------------------------------
# Consolidation and aggregation (section 10).
# ---------------------------------------------------------------------------
def write_combined_samples(
    plan: list[dict], cases: list[tuple[dict, list[dict[str, str]]]], out_path: Path
) -> int:
    """Writes combined_samples.csv preserving the exact 37-column schema, one
    header, deterministic invocation order, and increasing sample_index
    within each invocation. Returns the number of data rows written."""
    rows_by_index = {entry["index"]: rows for entry, rows in cases}
    row_count = 0
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for entry in plan:
            rows = sorted(rows_by_index[entry["index"]], key=lambda r: int(r["sample_index"]))
            for row in rows:
                writer.writerow([row[field] for field in CSV_HEADER])
                row_count += 1
    os.replace(tmp_path, out_path)
    return row_count


SUMMARY_HEADER = [
    "method", "stages", "bytes_in_flight_per_sm", "stage_bytes", "tile_height",
    "copies_per_thread_per_stage", "run_kind", "sm_count", "working_set_bytes",
    "passes", "warmup_ms", "sample_count",
    "kernel_time_ms_mean", "kernel_time_ms_median", "kernel_time_ms_stdev",
    "kernel_time_ms_cv_percent",
    "effective_gbps_mean", "effective_gbps_median", "effective_gbps_stdev",
    "effective_gbps_cv_percent", "effective_gbps_min", "effective_gbps_max",
    "correctness", "mismatches",
]


def _sample_stdev(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_case(entry: dict, rows: list[dict[str, str]]) -> dict:
    kernel_times = [float(r["kernel_time_ms"]) for r in rows]
    gbps = [float(r["effective_gbps"]) for r in rows]
    kt_mean = statistics.mean(kernel_times)
    gbps_mean = statistics.mean(gbps)
    kt_stdev = _sample_stdev(kernel_times)
    gbps_stdev = _sample_stdev(gbps)
    row0 = rows[0]
    return {
        "method": entry["method"],
        "stages": entry["stages"],
        "bytes_in_flight_per_sm": bytes_in_flight_of(entry["bif_kib"]),
        "stage_bytes": stage_bytes_of(entry["stages"], entry["bif_kib"]),
        "tile_height": tile_height_of(entry["stages"], entry["bif_kib"]),
        "copies_per_thread_per_stage": copies_per_thread_of(entry["stages"], entry["bif_kib"]),
        "run_kind": row0["run_kind"],
        "sm_count": row0["sm_count"],
        "working_set_bytes": row0["working_set_bytes"],
        "passes": row0["passes"],
        "warmup_ms": row0["warmup_ms"],
        "sample_count": len(rows),
        "kernel_time_ms_mean": kt_mean,
        "kernel_time_ms_median": statistics.median(kernel_times),
        "kernel_time_ms_stdev": kt_stdev,
        "kernel_time_ms_cv_percent": (100.0 * kt_stdev / kt_mean) if kt_mean != 0 else 0.0,
        "effective_gbps_mean": gbps_mean,
        "effective_gbps_median": statistics.median(gbps),
        "effective_gbps_stdev": gbps_stdev,
        "effective_gbps_cv_percent": (100.0 * gbps_stdev / gbps_mean) if gbps_mean != 0 else 0.0,
        "effective_gbps_min": min(gbps),
        "effective_gbps_max": max(gbps),
        "correctness": "OK",
        "mismatches": 0,
    }


def format_summary_value(field: str, value) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_summary(plan: list[dict], cases: list[tuple[dict, list[dict[str, str]]]], out_path: Path) -> int:
    """Writes summary.csv: exactly 18 rows ordered by (stages,
    bytes_in_flight_per_sm, method), descriptive statistics only."""
    summaries = [summarize_case(entry, rows) for entry, rows in cases]
    summaries.sort(key=lambda s: (s["stages"], s["bytes_in_flight_per_sm"], s["method"]))
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SUMMARY_HEADER)
        for summary in summaries:
            writer.writerow([format_summary_value(field, summary[field]) for field in SUMMARY_HEADER])
    os.replace(tmp_path, out_path)
    return len(summaries)


# ---------------------------------------------------------------------------
# Manifest (section 11): allowlisted, hashed, atomic, never publishable.
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(campaign_dir: Path) -> dict:
    path = campaign_dir / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def write_manifest_atomic(campaign_dir: Path, manifest: dict) -> None:
    path = campaign_dir / "manifest.json"
    tmp_path = campaign_dir / "manifest.json.tmp"
    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def merge_manifest(campaign_dir: Path, updates: dict, status: str) -> dict:
    manifest = load_manifest(campaign_dir)
    manifest.update(updates)
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["experiment_id"] = EXPERIMENT_ID
    manifest["status"] = status
    manifest["publishable"] = False
    write_manifest_atomic(campaign_dir, manifest)
    return manifest


def parse_versions_env() -> dict[str, str]:
    values: dict[str, str] = {}
    versions_path = REPO_ROOT / "VERSIONS.env"
    if not versions_path.is_file():
        return values
    for line in versions_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


# ---------------------------------------------------------------------------
# Subcommand: capture
# ---------------------------------------------------------------------------
def cmd_capture(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_campaign_dir(args.campaign_dir)
        out_path = resolve_capture_out_path(campaign_dir, args.out)
    except UnsafePathError as exc:
        print(f"aggregate_exp01_memory_paths: capture: ERROR: {exc}", file=sys.stderr)
        return 2

    binary_argv = list(args.binary_args)
    if binary_argv and binary_argv[0] == "--":
        binary_argv = binary_argv[1:]  # argparse.REMAINDER keeps a literal '--' marker
    if not binary_argv:
        print("aggregate_exp01_memory_paths: capture: ERROR: no binary command given after '--'", file=sys.stderr)
        return 2

    tmp_path = out_path.with_name(out_path.name + ".tmp")
    print(
        f"aggregate_exp01_memory_paths: capture: running {binary_argv!r} -> {out_path.name}",
        file=sys.stderr,
    )
    try:
        with open(tmp_path, "wb") as csv_out:
            result = subprocess.run(binary_argv, stdout=csv_out, stderr=None)
    except OSError as exc:
        print(f"aggregate_exp01_memory_paths: capture: ERROR: unable to launch binary: {exc}", file=sys.stderr)
        return 1

    if result.returncode == 0:
        size = tmp_path.stat().st_size if tmp_path.exists() else 0
        if size == 0:
            failed_path = out_path.with_name(out_path.name + ".invalid")
            if tmp_path.exists():
                os.replace(tmp_path, failed_path)
            print(
                "aggregate_exp01_memory_paths: capture: ERROR: binary exited 0 but produced no stdout; "
                f"preserved as {failed_path.name}",
                file=sys.stderr,
            )
            return 1
        os.replace(tmp_path, out_path)
        print(f"aggregate_exp01_memory_paths: capture: OK: wrote {out_path}", file=sys.stderr)
        return 0

    suffix = ".partial" if result.returncode < 0 else ".invalid"
    if tmp_path.exists() and tmp_path.stat().st_size > 0:
        failed_path = out_path.with_name(out_path.name + suffix)
        os.replace(tmp_path, failed_path)
        print(
            f"aggregate_exp01_memory_paths: capture: ERROR: binary exited {result.returncode}; "
            f"preserved partial output as {failed_path.name}",
            file=sys.stderr,
        )
    else:
        if tmp_path.exists():
            tmp_path.unlink()
        print(
            f"aggregate_exp01_memory_paths: capture: ERROR: binary exited {result.returncode}; "
            "no output was produced",
            file=sys.stderr,
        )
    return 1


# ---------------------------------------------------------------------------
# Subcommand: validate-case
# ---------------------------------------------------------------------------
def cmd_validate_case(args: argparse.Namespace) -> int:
    plan = build_plan()
    entry = plan_by_index(plan).get(args.index)
    if entry is None:
        print(f"aggregate_exp01_memory_paths: validate-case: ERROR: index {args.index} is not in 0..17", file=sys.stderr)
        return 2

    try:
        campaign_dir = resolve_campaign_dir(args.campaign_dir)
    except UnsafePathError as exc:
        print(f"aggregate_exp01_memory_paths: validate-case: ERROR: {exc}", file=sys.stderr)
        return 2

    case_path = campaign_dir / "cases" / f"{entry['case_name']}.csv"
    if not case_path.is_file():
        print(f"aggregate_exp01_memory_paths: validate-case: ERROR: missing case file {case_path}", file=sys.stderr)
        return 1

    expect = {
        "method": entry["method"],
        "stages": entry["stages"],
        "bif_kib": entry["bif_kib"],
        "run_kind": args.run_kind,
        "repetitions": args.repetitions,
        "passes": args.passes,
        "warmup_ms": args.warmup_ms,
        "git_commit": args.git_commit,
    }
    _, errors = validate_case_file(case_path, expect)
    if errors:
        print(f"aggregate_exp01_memory_paths: validate-case: FAIL: {case_path}", file=sys.stderr)
        for error in errors:
            print(f"aggregate_exp01_memory_paths: validate-case:   - {error}", file=sys.stderr)
        return 1
    print(f"aggregate_exp01_memory_paths: validate-case: OK: {case_path}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: finalize
# ---------------------------------------------------------------------------
def cmd_finalize(args: argparse.Namespace) -> int:
    plan = build_plan()
    plan_errors = check_plan_contract(plan)
    if plan_errors:
        print("aggregate_exp01_memory_paths: finalize: ERROR: internal plan contract violation:", file=sys.stderr)
        for error in plan_errors:
            print(f"aggregate_exp01_memory_paths: finalize:   - {error}", file=sys.stderr)
        return 1

    try:
        campaign_dir = resolve_campaign_dir(args.campaign_dir)
    except UnsafePathError as exc:
        print(f"aggregate_exp01_memory_paths: finalize: ERROR: {exc}", file=sys.stderr)
        return 2

    def fail(stage: str, errors: list[str]) -> int:
        for error in errors:
            print(f"aggregate_exp01_memory_paths: finalize:   - {error}", file=sys.stderr)
        merge_manifest(
            campaign_dir,
            {
                "failure_stage": stage,
                "failure_detail": errors[:50],
                "configuration_count_expected": EXPECTED_CONFIGURATION_COUNT,
            },
            status="FAILED",
        )
        return 1

    cases_dir = campaign_dir / "cases"
    found, set_errors = scan_case_directory(cases_dir, plan)
    if set_errors:
        print("aggregate_exp01_memory_paths: finalize: ERROR: case-set validation failed:", file=sys.stderr)
        return fail("case_set", set_errors)

    common = {
        "run_kind": args.run_kind,
        "repetitions": args.repetitions,
        "passes": args.passes,
        "warmup_ms": args.warmup_ms,
        "git_commit": args.git_commit,
    }
    cases: list[tuple[dict, list[dict[str, str]]]] = []
    all_errors: list[str] = []
    for index in sorted(found):
        entry = plan_by_index(plan)[index]
        expect = {
            "method": entry["method"],
            "stages": entry["stages"],
            "bif_kib": entry["bif_kib"],
            **common,
        }
        rows, errors = validate_case_file(found[index], expect)
        if errors:
            all_errors.extend(errors)
        else:
            cases.append((entry, rows))
    if all_errors:
        print("aggregate_exp01_memory_paths: finalize: ERROR: per-case validation failed:", file=sys.stderr)
        return fail("per_case_validation", all_errors)

    cross_errors = check_cross_case_consistency(cases)
    if cross_errors:
        print("aggregate_exp01_memory_paths: finalize: ERROR: cross-case consistency failed:", file=sys.stderr)
        return fail("cross_case_consistency", cross_errors)

    cases.sort(key=lambda pair: pair[0]["index"])
    combined_path = campaign_dir / "combined_samples.csv"
    summary_path = campaign_dir / "summary.csv"
    combined_rows = write_combined_samples(plan, cases, combined_path)
    summary_rows = write_summary(plan, cases, summary_path)

    expected_rows = EXPECTED_CONFIGURATION_COUNT * args.repetitions
    if combined_rows != expected_rows:
        return fail(
            "consolidation",
            [f"combined_samples.csv has {combined_rows} rows, expected {expected_rows}"],
        )
    if summary_rows != EXPECTED_CONFIGURATION_COUNT:
        return fail("aggregation", [f"summary.csv has {summary_rows} rows, expected {EXPECTED_CONFIGURATION_COUNT}"])

    reference_row = cases[0][1][0]
    case_hashes = {
        plan_by_index(plan)[index]["case_name"]: sha256_of(found[index]) for index in sorted(found)
    }
    binary_hashes = {}
    for label, path in (
        ("ldgsts_bin", MEMORY_LDGSTS_BIN), ("ldgsts_sass", MEMORY_LDGSTS_SASS),
        ("tma_bin", MEMORY_TMA_BIN), ("tma_sass", MEMORY_TMA_SASS),
    ):
        binary_hashes[label] = sha256_of(path) if path.is_file() else None

    updates = {
        "campaign_id": args.campaign_id,
        "run_kind": args.run_kind,
        "started_at_utc": args.started_at_utc,
        "completed_at_utc": args.completed_at_utc,
        "configuration_count_expected": EXPECTED_CONFIGURATION_COUNT,
        "configuration_count_completed": len(cases),
        "sample_count_expected": expected_rows,
        "sample_count_completed": combined_rows,
        "requested": {
            "run_kind": args.run_kind,
            "working_set_mib": args.working_set_mib,
            "passes": args.passes,
            "warmup_ms": args.warmup_ms,
            "repetitions": args.repetitions,
            "campaign_id": args.campaign_id,
        },
        "observed_common": {
            "requested_working_set_bytes": reference_row["requested_working_set_bytes"],
            "working_set_bytes": reference_row["working_set_bytes"],
            "sm_count": reference_row["sm_count"],
            "l2_bytes": reference_row["l2_bytes"],
            "passes": reference_row["passes"],
            "warmup_ms": reference_row["warmup_ms"],
            "repetitions": len(cases[0][1]),
        },
        "invocation_order": [
            {"index": e["index"], "method": e["method"], "stages": e["stages"], "bif_kib": e["bif_kib"],
             "case_name": e["case_name"]}
            for e in plan
        ],
        "selected_gpu_index": args.gpu_index,
        "gpu_name": reference_row["gpu_name"],
        "gpu_uuid": reference_row["gpu_uuid"],
        "compute_capability": reference_row["compute_capability"],
        "cuda_driver_version": reference_row["cuda_driver_version"],
        "cuda_runtime_version": reference_row["cuda_runtime_version"],
        "git_commit": args.git_commit,
        "git_dirty": False,
        "versions_env": parse_versions_env(),
        "binary_and_sass_sha256": binary_hashes,
        "case_file_sha256": case_hashes,
        "aggregate_file_sha256": {
            "combined_samples.csv": sha256_of(combined_path),
            "summary.csv": sha256_of(summary_path),
        },
        "self_test_outcomes": {"ldgsts": args.self_test_ldgsts, "tma": args.self_test_tma},
        "failure_stage": None,
        "failure_detail": None,
    }
    merge_manifest(campaign_dir, updates, status="COMPLETE")
    print(
        f"aggregate_exp01_memory_paths: finalize: OK: campaign {args.campaign_id} COMPLETE "
        f"({len(cases)} configurations, {combined_rows} samples)",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# Subcommand: manifest-write
# ---------------------------------------------------------------------------
def cmd_manifest_write(args: argparse.Namespace) -> int:
    try:
        campaign_dir = resolve_campaign_dir(args.campaign_dir)
    except UnsafePathError as exc:
        print(f"aggregate_exp01_memory_paths: manifest-write: ERROR: {exc}", file=sys.stderr)
        return 2

    updates: dict = {}
    if args.merge_json:
        merge_path = Path(args.merge_json)
        try:
            updates = json.loads(merge_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"aggregate_exp01_memory_paths: manifest-write: ERROR: cannot read --merge-json: {exc}", file=sys.stderr)
            return 2
        if not isinstance(updates, dict):
            print("aggregate_exp01_memory_paths: manifest-write: ERROR: --merge-json must contain a JSON object", file=sys.stderr)
            return 2

    merge_manifest(campaign_dir, updates, status=args.status)
    print(f"aggregate_exp01_memory_paths: manifest-write: OK: status={args.status}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: plan
# ---------------------------------------------------------------------------
def cmd_plan(args: argparse.Namespace) -> int:
    plan = build_plan()
    errors = check_plan_contract(plan)
    if errors:
        print("aggregate_exp01_memory_paths: plan: ERROR: plan contract violated:", file=sys.stderr)
        for error in errors:
            print(f"aggregate_exp01_memory_paths: plan:   - {error}", file=sys.stderr)
        return 1
    if args.format == "json":
        print(json.dumps(plan, indent=2))
    elif args.format == "lines":
        sys.stdout.write(format_plan_lines(plan))
    else:
        sys.stdout.write(format_plan_text(plan))
    return 0


# ---------------------------------------------------------------------------
# Self-test (section 12): builds every fixture under a TemporaryDirectory and
# removes it afterward. Never calls CUDA, Docker, nvidia-smi, either
# benchmark binary, or the network.
# ---------------------------------------------------------------------------
def _default_row(
    entry: dict, sample_index: int, *, repetitions: int, run_kind: str = "smoke",
    sm_count: int = 4, l2_bytes: int = 25165824, passes: int = 1, warmup_ms: int = 0,
    git_commit: str = "a" * 40, git_dirty: str = "false",
    kernel_time_ms: float | None = None,
    working_set_common_multiple_units: int = 8,
    overrides: dict | None = None,
) -> dict[str, str]:
    stages, bif_kib = entry["stages"], entry["bif_kib"]
    stage_bytes = stage_bytes_of(stages, bif_kib)
    bif_bytes = bytes_in_flight_of(bif_kib)
    tile_height = tile_height_of(stages, bif_kib)
    copies = copies_per_thread_of(stages, bif_kib)
    working_set_bytes = sm_count * 32 * 1024 * working_set_common_multiple_units
    useful_bytes = working_set_bytes * passes
    if kernel_time_ms is None:
        kernel_time_ms = 1.0 + 0.25 * sample_index
    effective_gbps = useful_bytes / (kernel_time_ms / 1000.0) / 1e9

    row = {
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": "2026-07-27T00:00:00Z",
        "run_kind": run_kind,
        "method": entry["method"],
        "sample_index": str(sample_index),
        "stages": str(stages),
        "tile_width_elements": str(FROZEN_TILE_WIDTH_ELEMENTS),
        "tile_width_bytes": str(FROZEN_TILE_WIDTH_BYTES),
        "tile_height": str(tile_height),
        "stage_bytes": str(stage_bytes),
        "bytes_in_flight_per_sm": str(bif_bytes),
        "vector_bytes": str(FROZEN_VECTOR_BYTES),
        "copies_per_thread_per_stage": str(copies),
        "threads_per_cta": str(FROZEN_THREADS_PER_CTA),
        "target_ctas_per_sm": str(FROZEN_TARGET_CTAS_PER_SM),
        "occupancy_ctas_per_sm": str(FROZEN_OCCUPANCY_CTAS_PER_SM),
        "grid_blocks": str(sm_count),
        "sm_count": str(sm_count),
        "smem_reservation_bytes": str(stage_bytes * stages + 1024),
        "l2_bytes": str(l2_bytes),
        "requested_working_set_bytes": str(working_set_bytes),
        "working_set_bytes": str(working_set_bytes),
        "working_set_l2_ratio": f"{working_set_bytes / l2_bytes:.6f}",
        "passes": str(passes),
        "useful_bytes": str(useful_bytes),
        "warmup_ms": str(warmup_ms),
        "kernel_time_ms": f"{kernel_time_ms:.6f}",
        "effective_gbps": f"{effective_gbps:.6f}",
        "correctness": "OK",
        "mismatches": "0",
        "gpu_name": "NVIDIA B300 SXM6 AC",
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
        "compute_capability": FROZEN_COMPUTE_CAPABILITY,
        "cuda_driver_version": "13010",
        "cuda_runtime_version": "13010",
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }
    if overrides:
        row.update({k: (v if isinstance(v, str) else str(v)) for k, v in overrides.items()})
    return row


def _write_case_csv(path: Path, rows: list[dict[str, str]], *, header: list[str] | None = None) -> None:
    header = header if header is not None else CSV_HEADER
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row.get(field, "") for field in header])


def _build_valid_campaign(
    campaign_dir: Path, *, repetitions: int = 3, run_kind: str = "smoke",
    passes: int = 1, warmup_ms: int = 0, git_commit: str = "a" * 40,
    row_overrides_by_index: dict[int, dict[int, dict]] | None = None,
) -> list[dict]:
    plan = build_plan()
    (campaign_dir / "cases").mkdir(parents=True, exist_ok=True)
    row_overrides_by_index = row_overrides_by_index or {}
    for entry in plan:
        per_sample_overrides = row_overrides_by_index.get(entry["index"], {})
        rows = [
            _default_row(
                entry, sample_index, repetitions=repetitions, run_kind=run_kind,
                passes=passes, warmup_ms=warmup_ms, git_commit=git_commit,
                overrides=per_sample_overrides.get(sample_index),
            )
            for sample_index in range(repetitions)
        ]
        _write_case_csv(campaign_dir / "cases" / f"{entry['case_name']}.csv", rows)
    return plan


class _SelfTestRecorder:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.total = 0

    def check(self, name: str, condition: bool, detail: str = "") -> None:
        self.total += 1
        if condition:
            print(f"aggregate_exp01_memory_paths: self-test: PASS: {name}", file=sys.stderr)
        else:
            self.failures.append(name)
            suffix = f"; {detail}" if detail else ""
            print(f"aggregate_exp01_memory_paths: self-test: FAIL: {name}{suffix}", file=sys.stderr)

    def expect_valid(self, name: str, errors: list[str]) -> None:
        self.check(name, not errors, detail=f"errors={errors}")

    def expect_error_containing(self, name: str, errors: list[str], needle: str) -> None:
        self.check(
            name, any(needle in error for error in errors),
            detail=f"expected substring {needle!r} in errors={errors}",
        )


def run_self_test() -> int:
    rec = _SelfTestRecorder()
    plan = build_plan()

    # 1-4: plan contract.
    plan_errors = check_plan_contract(plan)
    rec.check("plan has exactly 18 unique invocations", len(plan) == 18 and not plan_errors,
              detail=f"errors={plan_errors}")
    rec.check("each method appears exactly nine times",
              all(sum(1 for e in plan if e["method"] == m) == 9 for m in METHODS))
    rec.check(
        "every frozen configuration appears exactly twice, once per method",
        all(
            sorted(e["method"] for e in plan if (e["stages"], e["bif_kib"]) == cfg) == sorted(METHODS)
            for cfg in CONFIG_PAIRS
        ),
    )
    alternation_ok = True
    for pair_num in range(9):
        first, second = plan[pair_num * 2]["method"], plan[pair_num * 2 + 1]["method"]
        expected = ("ldgsts", "tma") if pair_num % 2 == 0 else ("tma", "ldgsts")
        if (first, second) != expected:
            alternation_ok = False
    rec.check("pairwise method order alternates as specified", alternation_ok)

    with tempfile.TemporaryDirectory(prefix="exp01_selftest_") as tmp:
        tmp_path = Path(tmp)

        # 5-7: valid campaign accepted; combined output; descriptive stats.
        campaign = tmp_path / "campaign_valid"
        campaign.mkdir()
        _build_valid_campaign(campaign, repetitions=3)
        found, set_errors = scan_case_directory(campaign / "cases", plan)
        common = {"run_kind": "smoke", "repetitions": 3, "passes": 1, "warmup_ms": 0, "git_commit": "a" * 40}
        cases = []
        all_errors = list(set_errors)
        for index in sorted(found):
            entry = plan_by_index(plan)[index]
            expect = {"method": entry["method"], "stages": entry["stages"], "bif_kib": entry["bif_kib"], **common}
            rows, errors = validate_case_file(found[index], expect)
            all_errors.extend(errors)
            if not errors:
                cases.append((entry, rows))
        rec.check("a valid synthetic campaign is accepted", not all_errors, detail=f"errors={all_errors[:5]}")

        combined_path = campaign / "combined_samples.csv"
        combined_rows = write_combined_samples(plan, cases, combined_path)
        with open(combined_path, newline="", encoding="utf-8") as handle:
            combined_all = list(csv.reader(handle))
        header_count = sum(1 for r in combined_all if r == CSV_HEADER)
        rec.check(
            "valid combined output has one header and the exact expected row count",
            header_count == 1 and combined_rows == 18 * 3 and len(combined_all) == 1 + 18 * 3,
        )

        # Independently (by hand, not via statistics.*) computed descriptive
        # stats fixture for one config (kernel_time_ms = 1.0, 1.25, 1.5 ms).
        stats_entry = plan_by_index(plan)[0]
        stats_rows = [
            _default_row(stats_entry, i, repetitions=3, kernel_time_ms=v)
            for i, v in enumerate((1.0, 1.25, 1.5))
        ]
        summary = summarize_case(stats_entry, stats_rows)
        expected_mean = (1.0 + 1.25 + 1.5) / 3.0
        expected_median = 1.25
        expected_variance = ((1.0 - expected_mean) ** 2 + (1.25 - expected_mean) ** 2 + (1.5 - expected_mean) ** 2) / 2
        expected_stdev = math.sqrt(expected_variance)
        rec.check(
            "descriptive statistics match independently calculated fixture values",
            abs(summary["kernel_time_ms_mean"] - expected_mean) < 1e-9
            and abs(summary["kernel_time_ms_median"] - expected_median) < 1e-9
            and abs(summary["kernel_time_ms_stdev"] - expected_stdev) < 1e-9
            and summary["sample_count"] == 3,
            detail=f"summary={summary}",
        )

        # 8-10: missing / duplicate / unexpected configuration.
        campaign_missing = tmp_path / "campaign_missing"
        campaign_missing.mkdir()
        _build_valid_campaign(campaign_missing, repetitions=1)
        (campaign_missing / "cases" / f"{plan[0]['case_name']}.csv").unlink()
        _, missing_errors = scan_case_directory(campaign_missing / "cases", plan)
        rec.expect_error_containing("missing configuration is rejected", missing_errors, "missing configuration")

        campaign_dup = tmp_path / "campaign_dup"
        campaign_dup.mkdir()
        _build_valid_campaign(campaign_dup, repetitions=1)
        wrong_entry = plan_by_index(plan)[0]
        bogus_name = f"00_tma_s{wrong_entry['stages']}_bif{wrong_entry['bif_kib']}.csv"
        shutil_copy_src = campaign_dup / "cases" / f"{plan[1]['case_name']}.csv"
        (campaign_dup / "cases" / bogus_name).write_bytes(shutil_copy_src.read_bytes())
        _, dup_errors = scan_case_directory(campaign_dup / "cases", plan)
        rec.expect_error_containing("duplicate configuration is rejected", dup_errors, "duplicate configuration")

        campaign_unexpected = tmp_path / "campaign_unexpected"
        campaign_unexpected.mkdir()
        _build_valid_campaign(campaign_unexpected, repetitions=1)
        (campaign_unexpected / "cases" / "99_ldgsts_s2_bif16.csv").write_text("bogus\n")
        _, unexpected_errors = scan_case_directory(campaign_unexpected / "cases", plan)
        rec.expect_error_containing("unexpected configuration is rejected", unexpected_errors, "unexpected file")

        # 11: wrong or reordered CSV header.
        campaign_header = tmp_path / "campaign_header"
        campaign_header.mkdir(parents=True)
        entry0 = plan_by_index(plan)[0]
        bad_header = CSV_HEADER[1:] + [CSV_HEADER[0]]
        _write_case_csv(
            campaign_header / f"{entry0['case_name']}.csv",
            [_default_row(entry0, 0, repetitions=1)],
            header=bad_header,
        )
        _, header_errors = validate_case_file(
            campaign_header / f"{entry0['case_name']}.csv",
            {"method": entry0["method"], "stages": entry0["stages"], "bif_kib": entry0["bif_kib"], **common,
             "repetitions": 1},
        )
        rec.expect_error_containing("wrong or reordered CSV header is rejected", header_errors, "header mismatch")

        # 12: missing or duplicate sample index.
        campaign_idx = tmp_path / "campaign_idx"
        campaign_idx.mkdir(parents=True)
        rows_dup_idx = [
            _default_row(entry0, 0, repetitions=2),
            _default_row(entry0, 0, repetitions=2),  # duplicate sample_index=0, missing 1
        ]
        _write_case_csv(campaign_idx / f"{entry0['case_name']}.csv", rows_dup_idx)
        _, idx_errors = validate_case_file(
            campaign_idx / f"{entry0['case_name']}.csv",
            {"method": entry0["method"], "stages": entry0["stages"], "bif_kib": entry0["bif_kib"], **common,
             "repetitions": 2},
        )
        rec.check(
            "missing or duplicate sample index is rejected",
            any("appears 2 times" in e for e in idx_errors) and any("is missing" in e for e in idx_errors),
            detail=f"errors={idx_errors}",
        )

        # 13: wrong method/stages/BIF.
        for field, bad_value, label in (("method", "tma", "method"), ("stages", "4", "stages")):
            case_path = tmp_path / f"campaign_wrong_{label}"
            case_path.mkdir(parents=True)
            row = _default_row(entry0, 0, repetitions=1, overrides={field: bad_value})
            _write_case_csv(case_path / "case.csv", [row])
            _, errs = validate_case_file(
                case_path / "case.csv",
                {"method": entry0["method"], "stages": entry0["stages"], "bif_kib": entry0["bif_kib"], **common,
                 "repetitions": 1},
            )
            rec.expect_error_containing(f"wrong {label} is rejected", errs, f"{field}=")

        # 14: mismatch or non-OK correctness.
        case_mismatch = tmp_path / "campaign_mismatch"
        case_mismatch.mkdir(parents=True)
        row_mismatch = _default_row(entry0, 0, repetitions=1, overrides={"correctness": "MISMATCH", "mismatches": 3})
        _write_case_csv(case_mismatch / "case.csv", [row_mismatch])
        _, mismatch_errors = validate_case_file(
            case_mismatch / "case.csv",
            {"method": entry0["method"], "stages": entry0["stages"], "bif_kib": entry0["bif_kib"], **common,
             "repetitions": 1},
        )
        rec.check(
            "a mismatch or non-OK correctness value is rejected",
            any("correctness=" in e for e in mismatch_errors) and any("mismatches=" in e for e in mismatch_errors),
            detail=f"errors={mismatch_errors}",
        )

        # 15: inconsistent GPU UUID across cases.
        campaign_uuid = tmp_path / "campaign_uuid"
        campaign_uuid.mkdir()
        _build_valid_campaign(
            campaign_uuid, repetitions=1,
            row_overrides_by_index={plan[3]["index"]: {0: {"gpu_uuid": "GPU-FFFFFFFF-FFFF-FFFF-FFFF-FFFFFFFFFFFF"}}},
        )
        found_u, _ = scan_case_directory(campaign_uuid / "cases", plan)
        cases_u = []
        for index in sorted(found_u):
            e = plan_by_index(plan)[index]
            expect = {"method": e["method"], "stages": e["stages"], "bif_kib": e["bif_kib"], **common, "repetitions": 1}
            rows, errs = validate_case_file(found_u[index], expect)
            if not errs:
                cases_u.append((e, rows))
        uuid_errors = check_cross_case_consistency(cases_u)
        rec.expect_error_containing("inconsistent GPU UUID is rejected", uuid_errors, "gpu_uuid=")

        # 16: inconsistent git commit / dirty benchmark.
        case_dirty = tmp_path / "campaign_dirty"
        case_dirty.mkdir(parents=True)
        row_dirty = _default_row(entry0, 0, repetitions=1, overrides={"git_dirty": "true"})
        _write_case_csv(case_dirty / "case.csv", [row_dirty])
        _, dirty_errors = validate_case_file(
            case_dirty / "case.csv",
            {"method": entry0["method"], "stages": entry0["stages"], "bif_kib": entry0["bif_kib"], **common,
             "repetitions": 1},
        )
        rec.expect_error_containing("inconsistent Git commit or dirty benchmark is rejected", dirty_errors, "git_dirty=")

        # 17: inconsistent working set, passes, or warm-up.
        case_ws = tmp_path / "campaign_ws"
        case_ws.mkdir(parents=True)
        row_ws = _default_row(entry0, 0, repetitions=1, overrides={"passes": 99})
        _write_case_csv(case_ws / "case.csv", [row_ws])
        _, ws_errors = validate_case_file(
            case_ws / "case.csv",
            {"method": entry0["method"], "stages": entry0["stages"], "bif_kib": entry0["bif_kib"], **common,
             "repetitions": 1},
        )
        rec.expect_error_containing("inconsistent working set, passes, or warm-up is rejected", ws_errors, "passes=")

        # 18: incorrect geometry or useful-byte formula.
        case_geo = tmp_path / "campaign_geo"
        case_geo.mkdir(parents=True)
        row_geo = _default_row(entry0, 0, repetitions=1, overrides={"tile_height": 999})
        _write_case_csv(case_geo / "case.csv", [row_geo])
        _, geo_errors = validate_case_file(
            case_geo / "case.csv",
            {"method": entry0["method"], "stages": entry0["stages"], "bif_kib": entry0["bif_kib"], **common,
             "repetitions": 1},
        )
        rec.expect_error_containing("incorrect geometry or useful-byte formula is rejected", geo_errors, "tile_height=")

        # 19: NaN, infinity, zero, or negative timing.
        for bad_value, label in (("nan", "NaN"), ("inf", "infinite"), ("0.0", "zero"), ("-1.0", "negative")):
            case_bad_timing = tmp_path / f"campaign_timing_{label}"
            case_bad_timing.mkdir(parents=True)
            row_bad = _default_row(entry0, 0, repetitions=1, overrides={"kernel_time_ms": bad_value})
            _write_case_csv(case_bad_timing / "case.csv", [row_bad])
            _, timing_errors = validate_case_file(
                case_bad_timing / "case.csv",
                {"method": entry0["method"], "stages": entry0["stages"], "bif_kib": entry0["bif_kib"], **common,
                 "repetitions": 1},
            )
            rec.check(f"{label} timing is rejected", bool(timing_errors), detail=f"errors={timing_errors}")

        # 20-21: unsafe campaign IDs / output paths; never overwrite.
        campaign_safe = tmp_path / "campaign_safe"
        (campaign_safe / "cases").mkdir(parents=True)
        unsafe_cases = [
            ("absolute path", "/etc/passwd"),
            ("parent traversal", "../escape.csv"),
            ("nested parent traversal", "cases/../../escape.csv"),
        ]
        unsafe_ok = True
        for label, bad_out in unsafe_cases:
            try:
                resolve_capture_out_path(campaign_safe, bad_out)
                unsafe_ok = False
            except UnsafePathError:
                pass
        existing_target = campaign_safe / "cases" / "already_there.csv"
        existing_target.write_text("x\n")
        try:
            resolve_capture_out_path(campaign_safe, "cases/already_there.csv")
            unsafe_ok = False
        except UnsafePathError:
            pass
        try:
            good_path = resolve_capture_out_path(campaign_safe, "cases/new_case.csv")
            unsafe_ok = unsafe_ok and good_path == (campaign_safe / "cases" / "new_case.csv").resolve()
        except UnsafePathError:
            unsafe_ok = False
        rec.check("unsafe campaign IDs and output paths are rejected", unsafe_ok)

        symlink_ok = True
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        try:
            (campaign_safe / "cases" / "escape_link").symlink_to(outside_dir)
            resolve_capture_out_path(campaign_safe, "cases/escape_link/evil.csv")
            symlink_ok = False
        except UnsafePathError:
            pass
        except OSError:
            pass  # symlinks may be unavailable in some sandboxes; do not fail the suite for that
        rec.check("existing campaign/output targets are never overwritten", symlink_ok)

        # 22: partial case files are never aggregated.
        campaign_partial = tmp_path / "campaign_partial"
        campaign_partial.mkdir()
        _build_valid_campaign(campaign_partial, repetitions=1)
        good_case = campaign_partial / "cases" / f"{plan[0]['case_name']}.csv"
        good_case.unlink()
        (campaign_partial / "cases" / f"{plan[0]['case_name']}.csv.partial").write_text("incomplete\n")
        found_partial, partial_errors = scan_case_directory(campaign_partial / "cases", plan)
        rec.check(
            "partial case files are never aggregated",
            len(found_partial) == 17 and any("missing configuration" in e for e in partial_errors),
            detail=f"found={len(found_partial)} errors={partial_errors[:3]}",
        )

    if rec.failures:
        print(
            f"aggregate_exp01_memory_paths: self-test: FAILED ({len(rec.failures)}/{rec.total} case(s)): "
            f"{rec.failures}",
            file=sys.stderr,
        )
        print("aggregate_exp01_memory_paths: SELF_TEST_RESULT=FAIL", file=sys.stderr)
        return 1
    print(f"aggregate_exp01_memory_paths: self-test: OK ({rec.total} cases)", file=sys.stderr)
    print("aggregate_exp01_memory_paths: SELF_TEST_RESULT=PASS", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aggregate_exp01_memory_paths.py",
        description="P1.3 plan/validation/aggregation helper (see module docstring).",
    )
    parser.add_argument("--self-test", action="store_true", help="Run GPU-free synthetic tests and exit.")
    subparsers = parser.add_subparsers(dest="command")

    plan_parser = subparsers.add_parser("plan", help="Print the frozen 18-invocation plan.")
    plan_parser.add_argument("--format", choices=("text", "lines", "json"), default="text")
    plan_parser.set_defaults(func=cmd_plan)

    capture_parser = subparsers.add_parser("capture", help="Capture one binary invocation's stdout to a case CSV.")
    capture_parser.add_argument("--campaign-dir", required=True)
    capture_parser.add_argument("--out", required=True)
    capture_parser.add_argument("binary_args", nargs=argparse.REMAINDER)
    capture_parser.set_defaults(func=cmd_capture)

    validate_parser = subparsers.add_parser("validate-case", help="Strictly validate one captured case CSV.")
    validate_parser.add_argument("--campaign-dir", required=True)
    validate_parser.add_argument("--index", required=True, type=int)
    validate_parser.add_argument("--run-kind", required=True, choices=("smoke", "benchmark"))
    validate_parser.add_argument("--repetitions", required=True, type=int)
    validate_parser.add_argument("--passes", required=True, type=int)
    validate_parser.add_argument("--warmup-ms", required=True, type=int)
    validate_parser.add_argument("--git-commit", required=True)
    validate_parser.set_defaults(func=cmd_validate_case)

    finalize_parser = subparsers.add_parser("finalize", help="Validate, consolidate, aggregate, and close a campaign.")
    finalize_parser.add_argument("--campaign-dir", required=True)
    finalize_parser.add_argument("--campaign-id", required=True)
    finalize_parser.add_argument("--run-kind", required=True, choices=("smoke", "benchmark"))
    finalize_parser.add_argument("--repetitions", required=True, type=int)
    finalize_parser.add_argument("--passes", required=True, type=int)
    finalize_parser.add_argument("--warmup-ms", required=True, type=int)
    finalize_parser.add_argument("--working-set-mib", type=int, default=None)
    finalize_parser.add_argument("--git-commit", required=True)
    finalize_parser.add_argument("--gpu-index", required=True, type=int)
    finalize_parser.add_argument("--started-at-utc", required=True)
    finalize_parser.add_argument("--completed-at-utc", required=True)
    finalize_parser.add_argument("--self-test-ldgsts", required=True, choices=("PASS", "FAIL"))
    finalize_parser.add_argument("--self-test-tma", required=True, choices=("PASS", "FAIL"))
    finalize_parser.set_defaults(func=cmd_finalize)

    manifest_parser = subparsers.add_parser("manifest-write", help="Merge a JSON fragment into manifest.json.")
    manifest_parser.add_argument("--campaign-dir", required=True)
    manifest_parser.add_argument("--status", required=True, choices=("IN_PROGRESS", "COMPLETE", "FAILED", "INTERRUPTED"))
    manifest_parser.add_argument("--merge-json", default=None)
    manifest_parser.set_defaults(func=cmd_manifest_write)

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
