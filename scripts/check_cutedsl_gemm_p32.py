#!/usr/bin/env python3
"""GPU-free contract checker for the P3.2 one-shape CuTe DSL GEMM wrapper.

This checker is deliberately independent of ``src/gemm/cutedsl_gemm.py``: it
carries its own copy of the frozen P3.2 configuration, its own copy of the
frozen 47-field CSV schema, and its own row validator. A drift in either the
wrapper or the checker therefore shows up as a disagreement rather than as two
copies of the same mistake.

It uses only the Python standard library and never initializes CUDA. The only
subprocesses it starts are Python interpreters running the wrapper's
``--help`` and ``--self-test`` behind an import guard that makes any attempt to
import PyTorch, CuTe DSL, or the CUDA bindings a hard failure - which is how
"``--help`` and ``--self-test`` are GPU-free" is proved rather than assumed.

What it validates:

* the wrapper's frozen configuration matches the frozen P3.2 table exactly;
* the CSV field names and their order match the frozen schema exactly;
* one synthetic valid row serializes to exactly one header and one data row;
* missing, duplicate, unknown, non-finite, and wrongly typed fields are all
  rejected, by the wrapper's validator and by this checker's own;
* ``publishable`` is fixed to ``false`` and a successful row is always
  ``correctness=PASS``; a failed or skipped check cannot build a row at all;
* the command line exposes no shape, dtype, layout, tiler, cluster, TMA,
  scheduling, or MMA-group control, and no way to skip the reference check;
* the wrapper contains no persistent or 2-CTA configuration and no
  performance-metric arithmetic;
* the wrapper is syntactically valid and import-safe (importing it pulls in no
  GPU stack);
* every provenance value is read from the pinned version contracts instead of
  being silently redefined as a literal in the wrapper.

Usage:
  check_cutedsl_gemm_p32.py [repository-root]
  check_cutedsl_gemm_p32.py --self-test

Exit code: 0 only when the selected validation passes, 1 on a contract or
synthetic-test failure, and 2 on a usage error.
"""

import csv
import io
import math
import os
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path

# --- Independent frozen expectations ----------------------------------------

WRAPPER_RELATIVE_PATH = "src/gemm/cutedsl_gemm.py"
CHECKER_RELATIVE_PATH = "scripts/check_cutedsl_gemm_p32.py"
PROTOCOL_RELATIVE_PATH = "src/gemm/P3_2_PROTOCOL.md"
GLOBAL_CONTRACT_FILE = "VERSIONS.env"
PHASE3_CONTRACT_FILE = "PHASE3_VERSIONS.env"

EXPECTED_CSV_FIELDS = (
    "schema_version",
    "experiment",
    "unit",
    "run_kind",
    "method",
    "variant",
    "m",
    "n",
    "k",
    "l",
    "ab_dtype",
    "acc_dtype",
    "c_dtype",
    "a_major",
    "b_major",
    "c_major",
    "mma_tiler_m",
    "mma_tiler_n",
    "cluster_m",
    "cluster_n",
    "use_2cta_instrs",
    "use_tma_store",
    "seed",
    "reference",
    "atol",
    "rtol",
    "correctness",
    "max_abs_error",
    "max_rel_error",
    "compile_time_ms",
    "first_launch_ms",
    "kernel_time_ms",
    "warmup_iterations",
    "iterations",
    "cache_mode",
    "gpu_name",
    "gpu_uuid",
    "compute_capability",
    "driver_version",
    "cuda_toolkit_version",
    "torch_cuda_version",
    "cutedsl_version",
    "cutlass_commit",
    "upstream_example_sha256",
    "git_commit",
    "git_dirty",
    "publishable",
)

EXPECTED_FROZEN_CONFIG = {
    "schema_version": "p32.v1",
    "experiment": "exp03_cutedsl_vs_cublaslt",
    "unit": "P3.2",
    "run_kind": "smoke",
    "method": "cutedsl",
    "variant": "nonpersistent_1cta",
    "m": 4096,
    "n": 4096,
    "k": 4096,
    "l": 1,
    "ab_dtype": "BFloat16",
    "acc_dtype": "Float32",
    "c_dtype": "Float32",
    "a_major": "k",
    "b_major": "k",
    "c_major": "n",
    "mma_tiler_m": 128,
    "mma_tiler_n": 128,
    "cluster_m": 1,
    "cluster_n": 1,
    "use_2cta_instrs": False,
    "use_tma_store": True,
    "seed": 1111,
    "reference": "torch_cuda_fp32_ieee",
    "atol": 1e-1,
    "rtol": 1e-5,
    "cache_mode": "hot",
    "publishable": False,
}

# The pinned architecture P3.2 targets. This checker owns the expectation; the
# wrapper only derives a compute capability from whatever VERSIONS.env pins.
EXPECTED_CUDA_ARCH = "sm_103a"
EXPECTED_COMPUTE_CAPABILITY = "10.3"

EXPECTED_FIXED_ROW_VALUES = {
    "schema_version": "p32.v1",
    "experiment": "exp03_cutedsl_vs_cublaslt",
    "unit": "P3.2",
    "run_kind": "smoke",
    "method": "cutedsl",
    "variant": "nonpersistent_1cta",
    "m": "4096",
    "n": "4096",
    "k": "4096",
    "l": "1",
    "ab_dtype": "BFloat16",
    "acc_dtype": "Float32",
    "c_dtype": "Float32",
    "a_major": "k",
    "b_major": "k",
    "c_major": "n",
    "mma_tiler_m": "128",
    "mma_tiler_n": "128",
    "cluster_m": "1",
    "cluster_n": "1",
    "use_2cta_instrs": "false",
    "use_tma_store": "true",
    "seed": "1111",
    "reference": "torch_cuda_fp32_ieee",
    "atol": "0.100000000",
    "rtol": "0.000010000",
    "correctness": "PASS",
    "cache_mode": "hot",
    "publishable": "false",
}

TIMING_FIELDS = ("compile_time_ms", "first_launch_ms", "kernel_time_ms")
ERROR_FIELDS = ("max_abs_error", "max_rel_error")
COUNT_FIELDS = ("warmup_iterations", "iterations")
BOOL_FIELDS = ("use_2cta_instrs", "use_tma_store", "git_dirty", "publishable")
TIMING_DECIMALS = 6
ERROR_DECIMALS = 9

ALLOWED_CLI_OPTIONS = frozenset(
    {"--help", "--self-test", "--warmup-iterations", "--iterations"}
)

# Option spellings that would reopen a frozen property. Checked in both the
# dashed and the underscored upstream spelling.
FORBIDDEN_CLI_FRAGMENTS = (
    "mnkl",
    "shape",
    "dtype",
    "major",
    "tiler",
    "cluster",
    "tma",
    "persistent",
    "2cta",
    "cta-group",
    "cta_group",
    "skip-ref",
    "skip_ref",
    "cold-l2",
    "cold_l2",
    "tolerance",
    "atol",
    "rtol",
    "seed",
    "variant",
    "method",
    "gpu",
    "device",
)

# Identifier fragments that must not appear as code in the wrapper. Prose in
# docstrings and comments is exempt: the scan runs over Python NAME tokens
# only, so a sentence explaining that P3.2 computes no TFLOP/s and is not an
# experimental campaign is fine, while a tflops variable or a campaign_dir
# variable is not. Matching is by substring, so derived spellings are caught
# too.
FORBIDDEN_SOURCE_IDENTIFIERS = (
    "tflop",
    "speedup",
    "efficiency",
    "bandwidth",
    "utilization",
    "cublas",
    "nsight",
    "autotune",
    "campaign",
    "skip_ref_check",
    "use_cold_l2",
    "dense_gemm_persistent",
)

# Literal flag spellings that must not appear anywhere in the wrapper, not even
# in prose, because their presence would suggest the frozen contract can be
# reopened from the command line.
FORBIDDEN_SOURCE_LITERALS = (
    "--skip-ref-check",
    "--skip_ref_check",
    "--use_2cta_instrs",
    "--use-2cta-instrs",
    "--use_cold_l2",
    "--mnkl",
    "--ab_dtype",
    "--mma_tiler_mn",
    "--cluster_shape_mn",
)

# Modules whose import would mean the "GPU-free" claim is false.
GPU_STACK_MODULES = ("torch", "cutlass", "cuda", "numpy", "pynvml")

_RE_HEX40 = re.compile(r"\A[0-9a-f]{40}\Z")
_RE_HEX64 = re.compile(r"\A[0-9a-f]{64}\Z")
_RE_GPU_UUID = re.compile(r"\AGPU-[0-9a-fA-F][0-9a-fA-F-]+\Z")
_RE_DOTTED_VERSION = re.compile(r"\A[0-9]+(\.[0-9]+)*\Z")
_RE_COMPUTE_CAPABILITY = re.compile(r"\A[0-9]+\.[0-9]+\Z")
_RE_POSITIVE_INT = re.compile(r"\A[1-9][0-9]*\Z")
_RE_ENV_LINE = re.compile(r"\A([A-Z][A-Z0-9_]*)=(\S*)\Z")

# P3.2 uses exclusively the PyTorch 2.10 fp32_precision API for the CUDA
# matmul FP32 policy of its correctness oracle. In 2.10 the legacy allow_tf32
# property is an alias of the same setting, mixing the two is unsupported, and
# the last write silently wins, so any appearance of the legacy spellings - or
# of an overlapping global precision API - is a contract failure.
FORBIDDEN_FP32_API_SPELLINGS = (
    "allow_tf32",
    "set_float32_matmul_precision",
    "get_float32_matmul_precision",
    "cudnn",
)
REQUIRED_FP32_API_SPELLINGS = ("fp32_precision",)
FP32_PRECISION_IEEE = "ieee"
# The unset default. It proves nothing about IEEE behaviour and must never be
# accepted as evidence of it.
FP32_PRECISION_UNSET = "none"

# Constructs that would let the smoke target silently drop unexpected stdout
# instead of letting it surface as a contract violation.
FORBIDDEN_SMOKE_FILTERS = ("grep -v", "| sed", "| tail", "| head", "| awk", "| grep")

SMOKE_TARGET = "gemm-cutedsl-p32-smoke"
CHECK_TARGET = "gemm-cutedsl-p32-check"
LAUNCHER_RELATIVE_PATH = "scripts/run_container.sh"
# The opt-in launcher mode that keeps stdout a pure data stream.
LAUNCHER_DATA_MODE_VARIABLE = "RUN_CONTAINER_STDOUT_IS_DATA"
SMOKE_SUCCESS_SENTENCE = (
    "P3.2 smoke completed: correctness passed before warm-up and steady-state timing."
)

# The subprocess guard: any import of the GPU stack aborts the child.
_GUARD_PRELUDE = """
import sys

_BLOCKED = {blocked!r}


class _ImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _BLOCKED:
            raise AssertionError("GPU-free contract violated: import of " + fullname)
        return None


sys.meta_path.insert(0, _ImportGuard())
"""

GPU_FREE_GUARD = (
    _GUARD_PRELUDE
    + """
sys.argv = [{argv0!r}] + {argv!r}
import runpy

runpy.run_path({wrapper!r}, run_name="__main__")
"""
)

# Drives the real main() with execute_measurement replaced by a correctness
# failure, proving the no-CSV contract through the real descriptor plumbing.
CORRECTNESS_FAILURE_PROBE = (
    _GUARD_PRELUDE
    + """
import importlib.util

spec = importlib.util.spec_from_file_location("p32_probe", {wrapper!r})
module = importlib.util.module_from_spec(spec)
sys.modules["p32_probe"] = module
spec.loader.exec_module(module)


def _raise_correctness_failure(*args, **kwargs):
    raise module.CorrectnessError("synthetic mismatch: 1 element exceeds atol/rtol")


module.execute_measurement = _raise_correctness_failure
sys.exit(module.main([]))
"""
)


# --- Pure validators ---------------------------------------------------------


def validate_csv_schema(fields) -> list:
    """Check a CSV field sequence against the frozen schema, in order."""
    errors = []
    fields = tuple(fields)
    if len(fields) != len(set(fields)):
        duplicates = sorted({name for name in fields if list(fields).count(name) > 1})
        errors.append(f"duplicate CSV field name(s): {', '.join(duplicates)}")
    if fields != EXPECTED_CSV_FIELDS:
        missing = [name for name in EXPECTED_CSV_FIELDS if name not in fields]
        unknown = [name for name in fields if name not in EXPECTED_CSV_FIELDS]
        if missing:
            errors.append(f"missing CSV field(s): {', '.join(missing)}")
        if unknown:
            errors.append(f"unknown CSV field(s): {', '.join(unknown)}")
        if not missing and not unknown:
            errors.append("CSV field order does not match the frozen schema")
    return errors


def validate_frozen_config(config) -> list:
    """Check a frozen-configuration mapping against the frozen P3.2 table."""
    errors = []
    if not isinstance(config, dict):
        return ["the frozen configuration is not a mapping"]
    for key, expected in sorted(EXPECTED_FROZEN_CONFIG.items()):
        if key not in config:
            errors.append(f"frozen configuration is missing {key}")
            continue
        actual = config[key]
        if isinstance(expected, bool) or isinstance(actual, bool):
            if actual is not expected:
                errors.append(f"frozen {key}={actual!r} != expected {expected!r}")
        elif isinstance(expected, float):
            if not isinstance(actual, float) or not math.isclose(actual, expected, rel_tol=0.0):
                errors.append(f"frozen {key}={actual!r} != expected {expected!r}")
        elif actual != expected:
            errors.append(f"frozen {key}={actual!r} != expected {expected!r}")
    unknown = sorted(set(config) - set(EXPECTED_FROZEN_CONFIG))
    if unknown:
        errors.append(f"frozen configuration has unexpected key(s): {', '.join(unknown)}")
    return errors


def _validate_decimal_field(field, text, decimals, strictly_positive, errors) -> None:
    if not re.fullmatch(rf"(0|[1-9][0-9]*)\.[0-9]{{{decimals}}}", text):
        errors.append(
            f"{field}: {text!r} is not a fixed-point decimal with {decimals} fractional digits"
        )
        return
    value = float(text)
    if not math.isfinite(value):
        errors.append(f"{field}: {text!r} is not finite")
        return
    if strictly_positive and value <= 0.0:
        errors.append(f"{field}: {text!r} must be strictly positive")


def validate_row_mapping(row) -> list:
    """Independently validate one parsed CSV row against the frozen contract."""
    errors = []
    if not isinstance(row, dict):
        return ["the row is not a mapping"]

    missing = [name for name in EXPECTED_CSV_FIELDS if name not in row]
    unknown = sorted(set(row) - set(EXPECTED_CSV_FIELDS))
    if missing:
        errors.append(f"missing field(s): {', '.join(missing)}")
    if unknown:
        errors.append(f"unknown field(s): {', '.join(unknown)}")
    if missing:
        return errors

    for field in EXPECTED_CSV_FIELDS:
        value = row[field]
        if not isinstance(value, str):
            errors.append(f"{field}: value {value!r} is not a string")
        elif not value:
            errors.append(f"{field}: value is empty")
        elif re.search(r"[\x00-\x1f\x7f]", value):
            errors.append(f"{field}: value contains a control character")
    if errors:
        return errors

    for field, expected in sorted(EXPECTED_FIXED_ROW_VALUES.items()):
        if row[field] != expected:
            errors.append(f"{field}: {row[field]!r} != frozen {expected!r}")

    for field in BOOL_FIELDS:
        if row[field] not in ("true", "false"):
            errors.append(f"{field}: {row[field]!r} is not a canonical lowercase boolean")

    for field in COUNT_FIELDS:
        if not _RE_POSITIVE_INT.match(row[field]):
            errors.append(f"{field}: {row[field]!r} is not a positive integer")

    for field in TIMING_FIELDS:
        _validate_decimal_field(field, row[field], TIMING_DECIMALS, True, errors)
    for field in ERROR_FIELDS:
        _validate_decimal_field(field, row[field], ERROR_DECIMALS, False, errors)

    if not _RE_HEX40.match(row["cutlass_commit"]):
        errors.append(f"cutlass_commit: {row['cutlass_commit']!r} is not a 40-hex commit")
    if not _RE_HEX40.match(row["git_commit"]):
        errors.append(f"git_commit: {row['git_commit']!r} is not a 40-hex commit")
    if not _RE_HEX64.match(row["upstream_example_sha256"]):
        errors.append("upstream_example_sha256 is not a 64-hex digest")
    if not _RE_GPU_UUID.match(row["gpu_uuid"]):
        errors.append(f"gpu_uuid: {row['gpu_uuid']!r} is malformed")
    if not _RE_COMPUTE_CAPABILITY.match(row["compute_capability"]):
        errors.append(f"compute_capability: {row['compute_capability']!r} is malformed")
    for field in (
        "driver_version",
        "cuda_toolkit_version",
        "torch_cuda_version",
        "cutedsl_version",
    ):
        if not _RE_DOTTED_VERSION.match(row[field]):
            errors.append(f"{field}: {row[field]!r} is not a dotted version")
    return errors


def validate_serialized_output(text) -> list:
    """Validate a complete stdout payload: one header, one row, nothing else."""
    errors = []
    if not isinstance(text, str):
        return ["the serialized output is not text"]
    if not text.endswith("\n"):
        errors.append("the serialized output does not end with a newline")
    if "\r" in text:
        errors.append("the serialized output contains a carriage return")
    lines = text.splitlines()
    if len(lines) != 2:
        errors.append(f"expected exactly 2 lines (header + row), got {len(lines)}")
        return errors
    if lines[0] != ",".join(EXPECTED_CSV_FIELDS):
        errors.append("the CSV header does not match the frozen schema")
        return errors
    rows = list(csv.DictReader(io.StringIO(text)))
    if len(rows) != 1:
        errors.append(f"expected exactly 1 data row, parsed {len(rows)}")
        return errors
    errors.extend(validate_row_mapping(dict(rows[0])))
    return errors


def validate_cli_options(options) -> list:
    """Check the wrapper's whole option surface."""
    errors = []
    options = set(options)
    unknown = sorted(options - ALLOWED_CLI_OPTIONS)
    if unknown:
        errors.append(f"unexpected command-line option(s): {', '.join(unknown)}")
    missing = sorted(ALLOWED_CLI_OPTIONS - options)
    if missing:
        errors.append(f"missing permitted command-line option(s): {', '.join(missing)}")
    for option in sorted(options):
        normalized = option.lstrip("-").lower()
        for fragment in FORBIDDEN_CLI_FRAGMENTS:
            if fragment in normalized:
                errors.append(f"option {option} would reopen a frozen property ({fragment})")
    return errors


def validate_source(source) -> list:
    """Scan wrapper source for forbidden code identifiers and flag spellings."""
    import tokenize

    errors = []
    for literal in FORBIDDEN_SOURCE_LITERALS:
        if literal in source:
            errors.append(f"forbidden literal {literal!r} appears in the wrapper")

    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        return errors + [f"the wrapper could not be tokenized: {exc}"]

    names = {token.string.lower() for token in tokens if token.type == tokenize.NAME}
    for fragment in FORBIDDEN_SOURCE_IDENTIFIERS:
        offenders = sorted(name for name in names if fragment in name)
        if offenders:
            errors.append(
                f"forbidden identifier fragment {fragment!r} is used in the wrapper "
                f"({', '.join(offenders)})"
            )

    # A raw results tree would make P3.2 write a dataset of its own.
    for literal in ("results/raw", "results/preflight"):
        if literal in source:
            errors.append(f"the wrapper references {literal!r}; P3.2 writes no result file")
    return errors


def validate_fp32_precision_policy(source) -> list:
    """Check that only the PyTorch 2.10 fp32_precision API controls FP32.

    Scans Python NAME tokens (which is how attribute access such as
    ``matmul.allow_tf32`` appears) plus string literals whose whole content is
    a forbidden spelling (which is how ``getattr(matmul, "allow_tf32")``
    appears). Prose in docstrings and comments is exempt, so the wrapper may
    explain why the legacy API is never used.
    """
    import tokenize

    errors = []
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        return [f"the FP32 policy could not be tokenized: {exc}"]

    names = {token.string for token in tokens if token.type == tokenize.NAME}
    literals = set()
    for token in tokens:
        if token.type != tokenize.STRING:
            continue
        text = token.string.strip()
        for quote in ('"""', "'''", '"', "'"):
            if text.startswith(quote) and text.endswith(quote) and len(text) >= 2 * len(quote):
                literals.add(text[len(quote):-len(quote)])
                break

    for spelling in FORBIDDEN_FP32_API_SPELLINGS:
        if spelling in names or spelling in literals:
            errors.append(
                f"the legacy/overlapping FP32 API {spelling!r} appears in production code; "
                "P3.2 uses only torch.backends.cuda.matmul.fp32_precision"
            )
    for spelling in REQUIRED_FP32_API_SPELLINGS:
        if spelling not in names:
            errors.append(f"the FP32 policy does not use {spelling!r}")
    if FP32_PRECISION_IEEE not in literals:
        errors.append(f"the FP32 policy never names the required {FP32_PRECISION_IEEE!r} value")
    return errors


def extract_make_recipe(makefile_text, target) -> list:
    """Return the recipe lines of one Make target, without the target line."""
    lines = makefile_text.splitlines()
    recipe = []
    collecting = False
    for line in lines:
        if line.startswith(f"{target}:"):
            collecting = True
            continue
        if collecting:
            if line.startswith("\t"):
                recipe.append(line)
                continue
            if line.strip() == "":
                continue
            break
    return recipe


def validate_smoke_recipe(recipe) -> list:
    """Check the smoke target's stdout, exit-status, and reporting contract."""
    errors = []
    if not recipe:
        return [f"the {SMOKE_TARGET} recipe is empty or was not found"]

    joined = "\n".join(recipe)

    # 1. Make must not echo the recipe onto stdout. Only the first physical
    #    line of each logical line carries the @ prefix; a physical line that
    #    continues the previous one (backslash continuation) is part of it.
    logical_starts = []
    continuing = False
    for line in recipe:
        if not continuing:
            logical_starts.append(line)
        continuing = line.rstrip("\n").endswith("\\")
    for start in logical_starts:
        body = start.lstrip("\t")
        if not body.startswith("@"):
            errors.append(f"recipe line is not quiet (missing @): {body[:70]!r}")
    if not logical_starts:
        errors.append("no recipe line starts with @; Make would echo the whole recipe")

    # 2. Every human-readable echo goes to stderr.
    for line in recipe:
        for match in re.finditer(r"\becho\b", line):
            tail = line[match.end():]
            if ">&2" not in tail:
                errors.append(f"an echo is not redirected to stderr: {line.strip()[:70]!r}")
                break

    # 3. The launcher runs in its opt-in data-stream mode, through the audited
    #    launcher, and never through Docker directly.
    if f"{LAUNCHER_DATA_MODE_VARIABLE}=1" not in joined:
        errors.append(
            f"the smoke target does not set {LAUNCHER_DATA_MODE_VARIABLE}=1, so launcher "
            "and entrypoint text would contaminate stdout"
        )
    if LAUNCHER_RELATIVE_PATH not in joined:
        errors.append(f"the smoke target does not go through {LAUNCHER_RELATIVE_PATH}")
    if re.search(r"(?<![\w/-])docker\s+run\b", joined):
        errors.append("the smoke target invokes Docker directly")

    # 4. The exit status is captured and preserved.
    if "|| status=$$?" not in joined:
        errors.append("the smoke target does not capture the launcher exit status")
    if not joined.rstrip().endswith("exit $$status"):
        errors.append("the smoke target does not end by preserving the exit status")

    # 5. No unconditional success claim.
    if SMOKE_SUCCESS_SENTENCE not in joined:
        errors.append("the smoke target never reports success accurately")
    else:
        guard = re.search(r'if \[ "\$\$status" -eq 0 \]; then', joined)
        if guard is None:
            errors.append("the success statement is not guarded by a zero exit status")
        elif joined.index(SMOKE_SUCCESS_SENTENCE) < guard.end():
            errors.append("the success statement is printed before the exit-status guard")
    for claim in ("Correctness passed before any timing ran", "compilation succeeded"):
        if claim in joined:
            errors.append(f"the smoke target makes the unconditional claim {claim!r}")

    # 6. No filtering that could silently swallow unexpected stdout.
    for construct in FORBIDDEN_SMOKE_FILTERS:
        if construct in joined:
            errors.append(
                f"the smoke target filters its output with {construct!r}; unexpected stdout "
                "must surface, not be discarded"
            )
    return errors


def validate_launcher_data_mode(source) -> list:
    """Check the launcher's opt-in mode without changing its default."""
    errors = []
    if LAUNCHER_DATA_MODE_VARIABLE not in source:
        errors.append(f"{LAUNCHER_RELATIVE_PATH} has no {LAUNCHER_DATA_MODE_VARIABLE} mode")
        return errors
    if "--entrypoint" not in source:
        errors.append(
            "the launcher never bypasses the image entrypoint, whose banner would "
            "contaminate a data stream on stdout"
        )
    # The two informational lines must be emitted through the mode-aware helper
    # rather than an unconditional echo to stdout.
    for marker in ("run_container: selected index=", "run_container: GPU "):
        for line in source.splitlines():
            if marker in line and line.strip().startswith("echo "):
                errors.append(f"launcher line {marker!r} still echoes unconditionally to stdout")
    return errors


def validate_provenance_linkage(source, wrapper_contract, parsed_contract) -> list:
    """Prove provenance is read from the pinned contracts, not redefined."""
    errors = []
    for name in ('"VERSIONS.env"', '"PHASE3_VERSIONS.env"'):
        if name not in source:
            errors.append(f"the wrapper does not read {name}")

    for key, value in sorted(parsed_contract.items()):
        if len(value) < 3:
            continue
        if value in source:
            errors.append(
                f"the wrapper redefines the pinned {key} value as a literal instead of "
                "reading it from the version contract"
            )
        if key == "CUTLASS_VERSION" and value.lstrip("v") in source:
            errors.append("the wrapper redefines the pinned CuTe DSL version as a literal")

    if not isinstance(wrapper_contract, dict):
        return errors + ["the wrapper did not return a contract mapping"]
    for key, value in sorted(parsed_contract.items()):
        if wrapper_contract.get(key) != value:
            errors.append(
                f"the wrapper resolved {key}={wrapper_contract.get(key)!r}, "
                f"the contract file says {value!r}"
            )
    if wrapper_contract.get("CUDA_ARCH") != EXPECTED_CUDA_ARCH:
        errors.append(
            f"the pinned architecture is {wrapper_contract.get('CUDA_ARCH')!r}, "
            f"P3.2 targets {EXPECTED_CUDA_ARCH!r}"
        )
    if wrapper_contract.get("EXPECTED_COMPUTE_CAPABILITY") != EXPECTED_COMPUTE_CAPABILITY:
        errors.append(
            f"{EXPECTED_CUDA_ARCH} must derive compute capability "
            f"{EXPECTED_COMPUTE_CAPABILITY}, the wrapper derived "
            f"{wrapper_contract.get('EXPECTED_COMPUTE_CAPABILITY')!r}"
        )
    return errors


def parse_env_file(path) -> dict:
    """Independently parse a KEY=VALUE version contract."""
    values = {}
    text = Path(path).read_text(encoding="utf-8")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _RE_ENV_LINE.match(line)
        if match is None:
            raise ValueError(f"{path}:{lineno}: malformed contract line {raw!r}")
        key, value = match.group(1), match.group(2)
        if key in values:
            raise ValueError(f"{path}:{lineno}: duplicate contract key {key}")
        values[key] = value
    return values


# --- Checks against the real wrapper -----------------------------------------


class _FakeMatmulBackend:
    """Stand-in for torch.backends.cuda.matmul: no torch, no CUDA, no GPU.

    Any access to the legacy allow_tf32 property raises, so a single mixed-API
    read or write in production code fails this checker loudly.
    """

    def __init__(self, initial=FP32_PRECISION_UNSET, readback=None, reject=False):
        object.__setattr__(self, "_stored", initial)
        object.__setattr__(self, "_readback", readback)
        object.__setattr__(self, "_reject", reject)
        object.__setattr__(self, "writes", [])

    @property
    def fp32_precision(self):
        readback = object.__getattribute__(self, "_readback")
        if readback is not None:
            return readback
        return object.__getattribute__(self, "_stored")

    def __setattr__(self, name, value):
        if name == "allow_tf32":
            raise AssertionError("production code wrote the legacy allow_tf32 API")
        if name == "fp32_precision":
            if object.__getattribute__(self, "_reject"):
                raise RuntimeError(f"Unknown precision: {value}")
            object.__getattribute__(self, "writes").append(value)
            object.__setattr__(self, "_stored", value)
            return
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name == "allow_tf32":
            raise AssertionError("production code read the legacy allow_tf32 API")
        raise AttributeError(name)


class _FakeMatmulBackendWithoutApi:
    """A PyTorch too old to expose the fp32_precision API."""

    def __getattr__(self, name):
        if name == "allow_tf32":
            raise AssertionError("production code touched the legacy allow_tf32 API")
        raise AttributeError(name)


class _FakeTorch:
    """Minimal torch stand-in exposing only backends.cuda.matmul."""

    def __init__(self, matmul):
        if matmul is not None:
            cuda = types.SimpleNamespace(matmul=matmul)
        else:
            cuda = types.SimpleNamespace()
        object.__setattr__(self, "backends", types.SimpleNamespace(cuda=cuda))

    def __getattr__(self, name):
        if name in ("set_float32_matmul_precision", "get_float32_matmul_precision"):
            raise AssertionError(f"production code used the overlapping API torch.{name}")
        raise AttributeError(name)


def check_ieee_fp32_guard(module) -> list:
    """Drive the wrapper's IEEE FP32 guard with GPU-free mocks."""
    errors = []
    p32_error = getattr(module, "P32Error", None)
    guard = getattr(module, "ieee_fp32_matmul", None)
    require = getattr(module, "require_ieee_fp32_matmul_api", None)
    if guard is None or require is None or p32_error is None:
        return ["the wrapper does not expose the IEEE FP32 guard"]

    def expect_rejection(name, matmul):
        fake = _FakeTorch(matmul)
        try:
            with guard(fake):
                errors.append(f"the IEEE guard accepted {name}")
        except p32_error:
            return
        except AssertionError as exc:
            errors.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: unexpected {type(exc).__name__}: {exc}")

    # Accepts only an exact "ieee" readback, and restores the previous value.
    backend = _FakeMatmulBackend(initial="tf32")
    fake_torch = _FakeTorch(backend)
    try:
        with guard(fake_torch):
            if backend.fp32_precision != FP32_PRECISION_IEEE:
                errors.append("the IEEE guard did not establish fp32_precision='ieee'")
        if backend.fp32_precision != "tf32":
            errors.append(
                f"the IEEE guard did not restore the previous setting "
                f"(left {backend.fp32_precision!r})"
            )
        if FP32_PRECISION_IEEE not in backend.writes:
            errors.append("the IEEE guard never wrote fp32_precision='ieee'")
    except AssertionError as exc:
        errors.append(f"the IEEE guard touched a forbidden API: {exc}")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"the IEEE guard rejected a valid backend: {type(exc).__name__}: {exc}")

    # The unset default proves nothing and must be rejected.
    expect_rejection(
        f"a {FP32_PRECISION_UNSET!r} readback",
        _FakeMatmulBackend(readback=FP32_PRECISION_UNSET),
    )
    expect_rejection("a 'tf32' readback", _FakeMatmulBackend(readback="tf32"))
    expect_rejection("an empty readback", _FakeMatmulBackend(readback=""))
    expect_rejection("a 'highest' readback", _FakeMatmulBackend(readback="highest"))
    expect_rejection("a rejected assignment", _FakeMatmulBackend(reject=True))
    expect_rejection("a PyTorch without the API", _FakeMatmulBackendWithoutApi())
    expect_rejection("a PyTorch without backends.cuda.matmul", None)

    # The same rejections must also come from the up-front requirement check.
    for name, matmul in (
        ("a PyTorch without the API", _FakeMatmulBackendWithoutApi()),
        ("a PyTorch without backends.cuda.matmul", None),
    ):
        try:
            require(_FakeTorch(matmul))
        except p32_error:
            pass
        except Exception as exc:  # noqa: BLE001
            errors.append(f"require_ieee_fp32_matmul_api({name}): {type(exc).__name__}: {exc}")
        else:
            errors.append(f"require_ieee_fp32_matmul_api accepted {name}")
    return errors


def _write_launcher_fakes(directory, payload_path):
    """Create GPU-free, network-free fake docker/nvidia-smi executables."""
    binary_dir = directory / "bin"
    binary_dir.mkdir(parents=True, exist_ok=True)

    nvidia_smi = binary_dir / "nvidia-smi"
    nvidia_smi.write_text(
        "#!/bin/sh\n"
        "case \"$*\" in\n"
        "  *--query-compute-apps=pid*) exit 0 ;;\n"
        "  *--query-gpu=driver_version*) echo '999.99.99'; exit 0 ;;\n"
        "  *--query-gpu=index,uuid,name*)\n"
        "    echo '0, GPU-11111111-2222-3333-4444-555555555555, FAKE TEST DEVICE'; exit 0 ;;\n"
        "esac\n"
        "exit 1\n",
        encoding="utf-8",
    )
    nvidia_smi.chmod(0o755)

    # Simulates the NGC entrypoint banner on stdout unless the entrypoint was
    # overridden, then emits the simulated wrapper payload.
    docker = binary_dir / "docker"
    docker.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" > "${FAKE_DOCKER_ARGV_FILE}"\n'
        'case " $* " in\n'
        '  *" --entrypoint "*) ;;\n'
        '  *) echo "== CUDA =="; echo "Container image Copyright (c) NVIDIA"; echo ;;\n'
        "esac\n"
        'echo "fake docker: simulated container started" >&2\n'
        f'if [ -n "${{FAKE_DOCKER_EMIT_PAYLOAD}}" ]; then cat "{payload_path}"; fi\n'
        'if [ -n "${FAKE_DOCKER_EXTRA_STDOUT}" ]; then echo "${FAKE_DOCKER_EXTRA_STDOUT}"; fi\n'
        'exit "${FAKE_DOCKER_EXIT:-0}"\n',
        encoding="utf-8",
    )
    docker.chmod(0o755)
    return binary_dir


def _run_simulated_launcher(root, binary_dir, argv_file, environment):
    base = dict(os.environ)
    base["PATH"] = f"{binary_dir}{os.pathsep}{base.get('PATH', '')}"
    base["BLACKWELL_GPU_INDEX"] = "0"
    base["FAKE_DOCKER_ARGV_FILE"] = str(argv_file)
    base.pop("FAKE_DOCKER_EMIT_PAYLOAD", None)
    base.pop("FAKE_DOCKER_EXTRA_STDOUT", None)
    base.pop("FAKE_DOCKER_EXIT", None)
    base.pop(LAUNCHER_DATA_MODE_VARIABLE, None)
    base.update(environment)
    return subprocess.run(
        [str(root / LAUNCHER_RELATIVE_PATH), "bash", "-c", "true"],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env=base,
    )


def check_simulated_launcher(root) -> list:
    """Prove the smoke stdout contract with the real launcher and fake tools.

    No GPU, no network, no Docker: only the repository's own launcher script
    plus two stub executables. It exercises the real quiet-mode logic rather
    than a re-implementation of it.
    """
    errors = []
    root = Path(root).resolve()
    launcher = root / LAUNCHER_RELATIVE_PATH
    if not launcher.is_file():
        return [f"missing {LAUNCHER_RELATIVE_PATH}"]

    with tempfile.TemporaryDirectory() as directory:
        workspace = Path(directory)
        payload = workspace / "payload.csv"
        payload.write_text(_serialize(_good_row()), encoding="utf-8")
        expected_csv = payload.read_text(encoding="utf-8")
        binary_dir = _write_launcher_fakes(workspace, payload)
        argv_file = workspace / "argv.txt"

        # 1. Data-stream mode, successful run: stdout is exactly the CSV.
        run = _run_simulated_launcher(
            root,
            binary_dir,
            argv_file,
            {LAUNCHER_DATA_MODE_VARIABLE: "1", "FAKE_DOCKER_EMIT_PAYLOAD": "1"},
        )
        if run.returncode != 0:
            errors.append(f"simulated data-mode run exited {run.returncode}")
        if run.stdout != expected_csv:
            errors.append(
                "data-mode stdout is not exactly the payload CSV; it contained "
                f"{len(run.stdout.splitlines())} line(s)"
            )
        errors.extend(
            f"data-mode stdout: {error}" for error in validate_serialized_output(run.stdout)
        )
        if "run_container: selected index=" not in run.stderr:
            errors.append("data-mode launcher diagnostics did not reach stderr")
        argv = argv_file.read_text(encoding="utf-8").splitlines()
        if "--entrypoint" not in argv:
            errors.append("data mode did not bypass the image entrypoint")
        if "== CUDA ==" in run.stdout:
            errors.append("the simulated entrypoint banner reached stdout in data mode")

        # 2. Data-stream mode, failing run: no stdout at all, status preserved.
        run = _run_simulated_launcher(
            root, binary_dir, argv_file, {LAUNCHER_DATA_MODE_VARIABLE: "1", "FAKE_DOCKER_EXIT": "7"}
        )
        if run.returncode != 7:
            errors.append(f"a failing run returned {run.returncode}, not the inner status 7")
        if run.stdout != "":
            errors.append(f"a failing run wrote {run.stdout!r} to stdout; it must be empty")

        # 3. Unexpected extra stdout must surface, never be silently discarded.
        run = _run_simulated_launcher(
            root,
            binary_dir,
            argv_file,
            {
                LAUNCHER_DATA_MODE_VARIABLE: "1",
                "FAKE_DOCKER_EMIT_PAYLOAD": "1",
                "FAKE_DOCKER_EXTRA_STDOUT": "unexpected trailing line",
            },
        )
        if "unexpected trailing line" not in run.stdout:
            errors.append("unexpected stdout was silently discarded instead of surfacing")
        if not validate_serialized_output(run.stdout):
            errors.append("contaminated stdout was accepted as a valid P3.2 payload")

        # 4. The default mode is unchanged, which the closed P1/P2 callers rely on.
        run = _run_simulated_launcher(root, binary_dir, argv_file, {"FAKE_DOCKER_EMIT_PAYLOAD": "1"})
        if run.returncode != 0:
            errors.append(f"simulated default-mode run exited {run.returncode}")
        if "run_container: selected index=" not in run.stdout:
            errors.append(
                "the default mode no longer prints its banner lines to stdout; the closed "
                "Phase 1/Phase 2 callers depend on that behaviour"
            )
        argv = argv_file.read_text(encoding="utf-8").splitlines()
        if "--entrypoint" in argv:
            errors.append("the default mode changed the image entrypoint")
    return errors


def _load_wrapper_module(wrapper_path):
    import importlib.util

    # Never leave a __pycache__ behind: the repository may be mounted
    # read-only, and a checker must not modify what it is checking.
    sys.dont_write_bytecode = True

    spec = importlib.util.spec_from_file_location("p32_wrapper_under_test", str(wrapper_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot build an import spec for {wrapper_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["p32_wrapper_under_test"] = module
    spec.loader.exec_module(module)
    return module


def _run_guarded(wrapper_path, argv):
    """Run the wrapper in a child interpreter that forbids the GPU stack."""
    code = GPU_FREE_GUARD.format(
        blocked=set(GPU_STACK_MODULES),
        argv0=wrapper_path.name,
        argv=list(argv),
        wrapper=str(wrapper_path),
    )
    return subprocess.run(
        [sys.executable, "-B", "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )


def check_wrapper(repo_root) -> list:
    """Run every P3.2 contract check against the real repository files."""
    errors = []
    root = Path(repo_root).resolve()
    wrapper_path = root / WRAPPER_RELATIVE_PATH
    checker_path = root / CHECKER_RELATIVE_PATH

    for relative in (WRAPPER_RELATIVE_PATH, CHECKER_RELATIVE_PATH, PROTOCOL_RELATIVE_PATH):
        if not (root / relative).is_file():
            errors.append(f"missing required P3.2 file: {relative}")
    if errors:
        return errors

    source = wrapper_path.read_text(encoding="utf-8")

    # 1. Syntax.
    for path in (wrapper_path, checker_path):
        try:
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        except SyntaxError as exc:
            errors.append(f"{path.name} does not compile: {exc}")
    if errors:
        return errors

    # 2. Import safety: importing the wrapper must pull in no GPU stack.
    already_loaded = {name for name in GPU_STACK_MODULES if name in sys.modules}
    module = _load_wrapper_module(wrapper_path)
    for name in GPU_STACK_MODULES:
        if name in sys.modules and name not in already_loaded:
            errors.append(f"importing the wrapper imported {name}; it must stay GPU-free")

    # 3. Frozen configuration and CSV schema.
    errors.extend(validate_frozen_config(getattr(module, "FROZEN_CONFIG", None)))
    errors.extend(validate_csv_schema(getattr(module, "CSV_FIELDS", ())))

    # 4. One synthetic valid row serializes correctly and validates independently.
    synthetic_provenance = {
        "gpu_name": "SYNTHETIC CHECKER DEVICE",
        "gpu_uuid": "GPU-11111111-2222-3333-4444-555555555555",
        "compute_capability": "9.9",
        "driver_version": "999.99.99",
        "cuda_toolkit_version": "99.9.9",
        "torch_cuda_version": "98.7",
        "cutedsl_version": "97.6.5",
        "git_commit": "a" * 40,
        "git_dirty": "false",
    }
    synthetic_upstream = {"commit": "b" * 40, "sha256": "c" * 64}
    try:
        row = module.build_row(
            correctness="PASS",
            max_abs_error=0.0,
            max_rel_error=0.25,
            compile_time_ms=1234.5,
            first_launch_ms=12.25,
            kernel_time_ms=7.5,
            warmup_iterations=2,
            iterations=10,
            provenance=synthetic_provenance,
            upstream=synthetic_upstream,
        )
        errors.extend(validate_serialized_output(module.serialize_row(row)))
    except Exception as exc:  # noqa: BLE001 - any failure here is a contract failure
        errors.append(f"the wrapper could not serialize a valid synthetic row: {exc}")
        row = None

    # 5. Malformed rows are rejected by the wrapper's own validator.
    if row is not None:
        rejection_cases = (
            ("a missing field", {name: value for name, value in row.items() if name != "seed"}),
            ("an unknown field", {**row, "tflops": "1.0"}),
            ("a NaN timing", {**row, "kernel_time_ms": "nan"}),
            ("an infinite timing", {**row, "compile_time_ms": "inf"}),
            ("a zero kernel time", {**row, "kernel_time_ms": "0.000000"}),
            ("a wrongly typed count", {**row, "iterations": "many"}),
            ("a non-string value", {**row, "iterations": 10}),
            ("publishable=true", {**row, "publishable": "true"}),
            ("correctness=FAIL", {**row, "correctness": "FAIL"}),
            ("a 2-CTA row", {**row, "use_2cta_instrs": "true"}),
            ("a persistent-looking variant", {**row, "variant": "persistent_2cta"}),
            ("a changed shape", {**row, "k": "8192"}),
        )
        for description, bad_row in rejection_cases:
            try:
                module.validate_row(bad_row)
            except Exception:  # noqa: BLE001 - any rejection is acceptable
                pass
            else:
                errors.append(f"the wrapper accepted {description}")
            if not validate_row_mapping(bad_row if isinstance(bad_row, dict) else {}):
                errors.append(f"this checker accepted {description}")

    # 6. A failed or skipped correctness check can never build a row.
    for correctness in ("FAIL", "SKIPPED", "pass", ""):
        try:
            module.build_row(
                correctness=correctness,
                max_abs_error=0.0,
                max_rel_error=0.0,
                compile_time_ms=1.0,
                first_launch_ms=1.0,
                kernel_time_ms=1.0,
                warmup_iterations=2,
                iterations=10,
                provenance=synthetic_provenance,
                upstream=synthetic_upstream,
            )
        except Exception:  # noqa: BLE001 - any rejection is acceptable
            pass
        else:
            errors.append(f"the wrapper built a row with correctness={correctness!r}")

    # 7. The command-line surface.
    parser = module.build_arg_parser()
    help_text = parser.format_help()
    errors.extend(validate_cli_options(set(re.findall(r"--[a-z0-9][a-z0-9-]*", help_text))))

    # 8. Source scan, including the FP32 policy.
    errors.extend(validate_source(source))
    errors.extend(validate_fp32_precision_policy(source))
    errors.extend(check_ieee_fp32_guard(module))

    # 9. Provenance is read from the pinned contracts.
    parsed = {}
    for contract_file in (GLOBAL_CONTRACT_FILE, PHASE3_CONTRACT_FILE):
        try:
            parsed.update(parse_env_file(root / contract_file))
        except (OSError, ValueError) as exc:
            errors.append(f"cannot parse {contract_file}: {exc}")
    interesting = {
        key: value
        for key, value in parsed.items()
        if key
        in (
            "CUDA_VERSION",
            "CUTLASS_VERSION",
            "CUTLASS_COMMIT",
            "CUDA_ARCH",
            "PYTORCH_VERSION",
            "PYTORCH_CUDA_VERSION",
            "CUTEDSL_P31_EXAMPLE_PATH",
            "CUTEDSL_P31_EXAMPLE_GIT_BLOB",
            "CUTEDSL_P31_EXAMPLE_SHA256",
        )
    }
    try:
        wrapper_contract = module.load_pinned_contract(root)
    except Exception as exc:  # noqa: BLE001
        errors.append(f"the wrapper could not load the pinned contract: {exc}")
        wrapper_contract = None
    errors.extend(validate_provenance_linkage(source, wrapper_contract, interesting))

    # 10. --help and --self-test really are GPU-free.
    help_run = _run_guarded(wrapper_path, ["--help"])
    if help_run.returncode != 0:
        errors.append(f"--help exited {help_run.returncode} under the GPU-free guard")
    if "GPU-free contract violated" in help_run.stderr:
        errors.append("--help imported part of the GPU stack")
    errors.extend(validate_cli_options(set(re.findall(r"--[a-z0-9][a-z0-9-]*", help_run.stdout))))

    self_test_run = _run_guarded(wrapper_path, ["--self-test"])
    if self_test_run.returncode != 0:
        errors.append(
            f"--self-test exited {self_test_run.returncode} under the GPU-free guard: "
            f"{self_test_run.stderr.strip()[-400:]}"
        )
    if "GPU-free contract violated" in self_test_run.stderr:
        errors.append("--self-test imported part of the GPU stack")
    if self_test_run.stdout != "":
        errors.append("--self-test wrote to stdout; only the CSV row may appear there")
    if "SELF-TEST: PASS" not in self_test_run.stderr:
        errors.append("--self-test did not report SELF-TEST: PASS")

    # 11. A correctness failure emits no CSV, through the real main() plumbing.
    failure_run = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            CORRECTNESS_FAILURE_PROBE.format(
                blocked=set(GPU_STACK_MODULES), wrapper=str(wrapper_path)
            ),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if failure_run.returncode != 1:
        errors.append(f"a correctness failure exited {failure_run.returncode}, expected 1")
    if failure_run.stdout != "":
        errors.append(
            f"a correctness failure wrote {failure_run.stdout!r} to stdout; it must emit no CSV"
        )
    if "CORRECTNESS FAILED" not in failure_run.stderr:
        errors.append("a correctness failure printed no stderr diagnostic")
    if "no warm-up or steady-state timing ran" not in failure_run.stderr:
        errors.append("a correctness failure did not state that no timing ran")

    # 12. The smoke target's stdout, exit-status, and reporting contract.
    makefile_text = (root / "Makefile").read_text(encoding="utf-8")
    errors.extend(validate_smoke_recipe(extract_make_recipe(makefile_text, SMOKE_TARGET)))
    errors.extend(
        validate_launcher_data_mode((root / LAUNCHER_RELATIVE_PATH).read_text(encoding="utf-8"))
    )
    errors.extend(check_simulated_launcher(root))

    return errors


# --- Self-test ---------------------------------------------------------------


def _good_row() -> dict:
    row = dict(EXPECTED_FIXED_ROW_VALUES)
    row.update(
        {
            "max_abs_error": "0.000000000",
            "max_rel_error": "0.000000000",
            "compile_time_ms": "1234.500000",
            "first_launch_ms": "12.250000",
            "kernel_time_ms": "7.500000",
            "warmup_iterations": "2",
            "iterations": "10",
            "gpu_name": "SYNTHETIC CHECKER DEVICE",
            "gpu_uuid": "GPU-11111111-2222-3333-4444-555555555555",
            "compute_capability": "9.9",
            "driver_version": "999.99.99",
            "cuda_toolkit_version": "99.9.9",
            "torch_cuda_version": "98.7",
            "cutedsl_version": "97.6.5",
            "cutlass_commit": "b" * 40,
            "upstream_example_sha256": "c" * 64,
            "git_commit": "a" * 40,
            "git_dirty": "false",
        }
    )
    return row


def _serialize(row) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(EXPECTED_CSV_FIELDS),
        extrasaction="raise",
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerow(row)
    return buffer.getvalue()


def run_self_test() -> int:
    """Prove this checker rejects representative malformed cases."""
    failures = []

    def accepts(name, errors):
        if errors:
            failures.append(name)
            print(f"  FAIL {name}: unexpectedly rejected with {errors}", file=sys.stderr)
        else:
            print(f"  ok   {name}", file=sys.stderr)

    def rejects(name, errors, fragment):
        joined = " | ".join(errors)
        if errors and fragment in joined:
            print(f"  ok   {name}", file=sys.stderr)
        else:
            failures.append(name)
            print(f"  FAIL {name}: got {joined!r}, expected {fragment!r}", file=sys.stderr)

    print("check_cutedsl_gemm_p32 --self-test (GPU-free)", file=sys.stderr)

    good = _good_row()
    accepts("a valid synthetic row is accepted", validate_row_mapping(good))
    accepts("a valid serialized payload is accepted", validate_serialized_output(_serialize(good)))
    accepts("the frozen schema is accepted", validate_csv_schema(EXPECTED_CSV_FIELDS))
    accepts("the frozen configuration is accepted", validate_frozen_config(EXPECTED_FROZEN_CONFIG))
    accepts("the permitted option set is accepted", validate_cli_options(ALLOWED_CLI_OPTIONS))
    accepts("clean source is accepted", validate_source("VALUE = 1\n"))

    # Schema.
    rejects(
        "a missing schema field is rejected",
        validate_csv_schema(EXPECTED_CSV_FIELDS[:-1]),
        "missing CSV field",
    )
    rejects(
        "an unknown schema field is rejected",
        validate_csv_schema(EXPECTED_CSV_FIELDS + ("tflops",)),
        "unknown CSV field",
    )
    rejects(
        "a duplicate schema field is rejected",
        validate_csv_schema(EXPECTED_CSV_FIELDS + ("seed",)),
        "duplicate CSV field",
    )
    rejects(
        "a reordered schema is rejected",
        validate_csv_schema((EXPECTED_CSV_FIELDS[1], EXPECTED_CSV_FIELDS[0]) + EXPECTED_CSV_FIELDS[2:]),
        "order does not match",
    )

    # Frozen configuration.
    rejects(
        "another shape is rejected",
        validate_frozen_config({**EXPECTED_FROZEN_CONFIG, "k": 8192}),
        "frozen k",
    )
    rejects(
        "a 2-CTA MMA group is rejected",
        validate_frozen_config({**EXPECTED_FROZEN_CONFIG, "use_2cta_instrs": True}),
        "frozen use_2cta_instrs",
    )
    rejects(
        "a persistent variant is rejected",
        validate_frozen_config({**EXPECTED_FROZEN_CONFIG, "variant": "persistent_1cta"}),
        "frozen variant",
    )
    rejects(
        "a publishable configuration is rejected",
        validate_frozen_config({**EXPECTED_FROZEN_CONFIG, "publishable": True}),
        "frozen publishable",
    )
    rejects(
        "a cold-L2 cache model is rejected",
        validate_frozen_config({**EXPECTED_FROZEN_CONFIG, "cache_mode": "cold"}),
        "frozen cache_mode",
    )
    rejects(
        "a changed seed is rejected",
        validate_frozen_config({**EXPECTED_FROZEN_CONFIG, "seed": 7}),
        "frozen seed",
    )

    # Rows.
    rejects(
        "a missing row field is rejected",
        validate_row_mapping({name: value for name, value in good.items() if name != "seed"}),
        "missing field",
    )
    rejects(
        "an unknown row field is rejected",
        validate_row_mapping({**good, "tflops": "1.0"}),
        "unknown field",
    )
    rejects(
        "a NaN timing is rejected",
        validate_row_mapping({**good, "kernel_time_ms": "nan"}),
        "kernel_time_ms",
    )
    rejects(
        "an infinite timing is rejected",
        validate_row_mapping({**good, "first_launch_ms": "inf"}),
        "first_launch_ms",
    )
    rejects(
        "a non-finite error is rejected",
        validate_row_mapping({**good, "max_rel_error": "nan"}),
        "max_rel_error",
    )
    rejects(
        "a zero timing is rejected",
        validate_row_mapping({**good, "kernel_time_ms": "0.000000"}),
        "strictly positive",
    )
    rejects(
        "a negative timing is rejected",
        validate_row_mapping({**good, "compile_time_ms": "-1.000000"}),
        "compile_time_ms",
    )
    rejects(
        "the wrong decimal precision is rejected",
        validate_row_mapping({**good, "kernel_time_ms": "7.5"}),
        "fractional digits",
    )
    rejects(
        "a non-string value is rejected",
        validate_row_mapping({**good, "iterations": 10}),
        "not a string",
    )
    rejects(
        "a non-integer count is rejected",
        validate_row_mapping({**good, "warmup_iterations": "two"}),
        "warmup_iterations",
    )
    rejects(
        "publishable=true is rejected",
        validate_row_mapping({**good, "publishable": "true"}),
        "publishable",
    )
    rejects(
        "a capitalized boolean is rejected",
        validate_row_mapping({**good, "git_dirty": "False"}),
        "git_dirty",
    )
    rejects(
        "correctness=FAIL is rejected",
        validate_row_mapping({**good, "correctness": "FAIL"}),
        "correctness",
    )
    rejects(
        "a 2-CTA row is rejected",
        validate_row_mapping({**good, "use_2cta_instrs": "true"}),
        "use_2cta_instrs",
    )
    rejects(
        "a persistent variant row is rejected",
        validate_row_mapping({**good, "variant": "persistent_2cta"}),
        "variant",
    )
    rejects(
        "another problem shape is rejected",
        validate_row_mapping({**good, "m": "8192"}),
        "frozen",
    )
    rejects(
        "a malformed commit is rejected",
        validate_row_mapping({**good, "cutlass_commit": "deadbeef"}),
        "cutlass_commit",
    )
    rejects(
        "a malformed digest is rejected",
        validate_row_mapping({**good, "upstream_example_sha256": "c" * 63}),
        "upstream_example_sha256",
    )
    rejects(
        "a malformed GPU UUID is rejected",
        validate_row_mapping({**good, "gpu_uuid": "0000"}),
        "gpu_uuid",
    )
    rejects(
        "an embedded newline is rejected",
        validate_row_mapping({**good, "gpu_name": "A\nB"}),
        "control character",
    )

    # Serialized payloads.
    rejects(
        "a duplicated data row is rejected",
        validate_serialized_output(_serialize(good) + _serialize(good).splitlines()[1] + "\n"),
        "exactly 2 lines",
    )
    rejects(
        "a header-only payload is rejected",
        validate_serialized_output(",".join(EXPECTED_CSV_FIELDS) + "\n"),
        "exactly 2 lines",
    )
    rejects(
        "a wrong header is rejected",
        validate_serialized_output("a,b\n1,2\n"),
        "header does not match",
    )
    rejects(
        "a CRLF payload is rejected",
        validate_serialized_output(_serialize(good).replace("\n", "\r\n")),
        "carriage return",
    )

    # Command line.
    rejects(
        "a shape option is rejected",
        validate_cli_options(set(ALLOWED_CLI_OPTIONS) | {"--mnkl"}),
        "unexpected command-line option",
    )
    rejects(
        "a reference-skipping option is rejected",
        validate_cli_options(set(ALLOWED_CLI_OPTIONS) | {"--skip-ref-check"}),
        "unexpected command-line option",
    )
    rejects(
        "a missing permitted option is rejected",
        validate_cli_options(ALLOWED_CLI_OPTIONS - {"--self-test"}),
        "missing permitted command-line option",
    )

    # Source scanning.
    rejects(
        "a TFLOP/s computation is rejected",
        validate_source("tflops = 2 * m * n * k / t\n"),
        "tflops",
    )
    rejects(
        "a speedup computation is rejected",
        validate_source("speedup = a / b\n"),
        "speedup",
    )
    rejects(
        "a reference-skipping flag is rejected",
        validate_source('PARSER.add_argument("--skip-ref-check")\n'),
        "--skip-ref-check",
    )
    rejects(
        "a persistent upstream example is rejected",
        validate_source("import dense_gemm_persistent\n"),
        "dense_gemm_persistent",
    )
    rejects(
        "writing a raw results tree is rejected",
        validate_source('PATH = "results/raw/exp03"\n'),
        "results/raw",
    )
    rejects(
        "a campaign directory variable is rejected",
        validate_source("campaign_dir = 1\n"),
        "campaign",
    )
    rejects(
        "an Nsight Compute call is rejected",
        validate_source("nsight_report = 1\n"),
        "nsight",
    )
    accepts(
        "prose about TFLOP/s, cuBLASLt, and campaigns is allowed",
        validate_source(
            '"""P3.2 is not a campaign, computes no TFLOP/s, and has no cuBLASLt baseline."""\n'
        ),
    )

    # Provenance linkage.
    fake_contract = {"CUTLASS_COMMIT": "e" * 40, "CUDA_ARCH": EXPECTED_CUDA_ARCH}
    linked_source = 'A = "VERSIONS.env"\nB = "PHASE3_VERSIONS.env"\n'
    accepts(
        "a wrapper that reads both contracts is accepted",
        validate_provenance_linkage(
            linked_source,
            {
                **fake_contract,
                "EXPECTED_COMPUTE_CAPABILITY": EXPECTED_COMPUTE_CAPABILITY,
            },
            fake_contract,
        ),
    )
    rejects(
        "a hardcoded pinned commit is rejected",
        validate_provenance_linkage(
            linked_source + f'COMMIT = "{"e" * 40}"\n',
            {**fake_contract, "EXPECTED_COMPUTE_CAPABILITY": EXPECTED_COMPUTE_CAPABILITY},
            fake_contract,
        ),
        "redefines the pinned CUTLASS_COMMIT",
    )
    rejects(
        "a wrapper that never reads the global contract is rejected",
        validate_provenance_linkage(
            'B = "PHASE3_VERSIONS.env"\n',
            {**fake_contract, "EXPECTED_COMPUTE_CAPABILITY": EXPECTED_COMPUTE_CAPABILITY},
            fake_contract,
        ),
        'does not read "VERSIONS.env"',
    )
    rejects(
        "a mismatched resolved value is rejected",
        validate_provenance_linkage(
            linked_source,
            {
                "CUTLASS_COMMIT": "f" * 40,
                "CUDA_ARCH": EXPECTED_CUDA_ARCH,
                "EXPECTED_COMPUTE_CAPABILITY": EXPECTED_COMPUTE_CAPABILITY,
            },
            fake_contract,
        ),
        "the contract file says",
    )
    rejects(
        "another target architecture is rejected",
        validate_provenance_linkage(
            linked_source,
            {"CUTLASS_COMMIT": "e" * 40, "CUDA_ARCH": "sm_90a", "EXPECTED_COMPUTE_CAPABILITY": "9.0"},
            {"CUTLASS_COMMIT": "e" * 40, "CUDA_ARCH": "sm_90a"},
        ),
        "P3.2 targets",
    )

    # FP32 precision policy.
    good_fp32 = (
        'MATMUL = torch.backends.cuda.matmul\n'
        'MATMUL.fp32_precision = "ieee"\n'
    )
    accepts("an exclusive fp32_precision policy is accepted", validate_fp32_precision_policy(good_fp32))
    rejects(
        "the legacy allow_tf32 API is rejected",
        validate_fp32_precision_policy(good_fp32 + "MATMUL.allow_tf32 = False\n"),
        "allow_tf32",
    )
    rejects(
        "an overlapping global precision API is rejected",
        validate_fp32_precision_policy(good_fp32 + 'torch.set_float32_matmul_precision("highest")\n'),
        "set_float32_matmul_precision",
    )
    rejects(
        "the legacy cuDNN TF32 property is rejected",
        validate_fp32_precision_policy(good_fp32 + "torch.backends.cudnn.allow_tf32 = False\n"),
        "cudnn",
    )
    rejects(
        "a getattr-spelled legacy API is rejected",
        validate_fp32_precision_policy(good_fp32 + 'X = getattr(MATMUL, "allow_tf32")\n'),
        "allow_tf32",
    )
    accepts(
        "prose explaining why the legacy API is unused is allowed",
        validate_fp32_precision_policy(
            '"""P3.2 never reads allow_tf32 and never calls set_float32_matmul_precision."""\n'
            "# allow_tf32 is deliberately untouched.\n" + good_fp32
        ),
    )
    rejects(
        "a policy that never names ieee is rejected",
        validate_fp32_precision_policy("MATMUL = torch.backends.cuda.matmul\nMATMUL.fp32_precision = X\n"),
        "never names",
    )
    rejects(
        "a policy that does not use the new API is rejected",
        validate_fp32_precision_policy('X = "ieee"\n'),
        "fp32_precision",
    )

    # Smoke recipe.
    good_recipe = [
        "\t@if [ -z \"$${BLACKWELL_GPU_INDEX:-}\" ]; then \\",
        "\t\techo \"ERROR: set it\" >&2; \\",
        "\t\texit 2; \\",
        "\tfi",
        "\t@status=0; \\",
        "\tRUN_CONTAINER_STDOUT_IS_DATA=1 scripts/run_container.sh bash -c 'true' || status=$$?; \\",
        "\techo \"warning\" >&2; \\",
        "\tif [ \"$$status\" -eq 0 ]; then \\",
        f"\t\techo \"{SMOKE_SUCCESS_SENTENCE}\" >&2; \\",
        "\telse \\",
        "\t\techo \"failed\" >&2; \\",
        "\tfi; \\",
        "\texit $$status",
    ]
    accepts("a correct smoke recipe is accepted", validate_smoke_recipe(good_recipe))
    rejects("an empty smoke recipe is rejected", validate_smoke_recipe([]), "empty")
    rejects(
        "a non-quiet recipe line is rejected",
        validate_smoke_recipe([line.replace("\t@status=0", "\tstatus=0") for line in good_recipe]),
        "not quiet",
    )
    rejects(
        "an echo on stdout is rejected",
        validate_smoke_recipe([line.replace('echo "warning" >&2', 'echo "warning"') for line in good_recipe]),
        "not redirected to stderr",
    )
    rejects(
        "a recipe without the launcher data mode is rejected",
        validate_smoke_recipe(
            [line.replace(f"{LAUNCHER_DATA_MODE_VARIABLE}=1 ", "") for line in good_recipe]
        ),
        LAUNCHER_DATA_MODE_VARIABLE,
    )
    rejects(
        "a recipe that calls Docker directly is rejected",
        validate_smoke_recipe(
            [line.replace("scripts/run_container.sh bash", "docker run --gpus all bash") for line in good_recipe]
        ),
        "invokes Docker directly",
    )
    rejects(
        "a recipe that drops the exit status is rejected",
        validate_smoke_recipe([line for line in good_recipe if line.strip() != "exit $$status"]),
        "preserving the exit status",
    )
    rejects(
        "a recipe that does not capture the launcher status is rejected",
        validate_smoke_recipe([line.replace(" || status=$$?", "") for line in good_recipe]),
        "capture the launcher exit status",
    )
    rejects(
        "an unconditional success message is rejected",
        validate_smoke_recipe(
            [line for line in good_recipe if 'if [ "$$status" -eq 0 ]' not in line]
        ),
        "not guarded by a zero exit status",
    )
    rejects(
        "a stale unconditional correctness claim is rejected",
        validate_smoke_recipe(
            [line.replace('echo "warning" >&2', 'echo "Correctness passed before any timing ran" >&2')
             for line in good_recipe]
        ),
        "unconditional claim",
    )
    rejects(
        "an output filter is rejected",
        validate_smoke_recipe(
            [line.replace("|| status=$$?", "| grep -v run_container || status=$$?") for line in good_recipe]
        ),
        "filters its output",
    )

    # Launcher data mode.
    good_launcher = (
        f'STDOUT_IS_DATA=0\n[ "${{{LAUNCHER_DATA_MODE_VARIABLE}:-0}}" = "1" ] && STDOUT_IS_DATA=1\n'
        'info "run_container: selected index=1"\n'
        'info "run_container: GPU x has no active compute processes"\n'
        "entrypoint_args=(--entrypoint /bin/bash)\n"
    )
    accepts("a launcher with the opt-in data mode is accepted", validate_launcher_data_mode(good_launcher))
    rejects(
        "a launcher without the opt-in mode is rejected",
        validate_launcher_data_mode('echo "run_container: selected index=1"\n'),
        "no RUN_CONTAINER_STDOUT_IS_DATA mode",
    )
    rejects(
        "a launcher that never bypasses the entrypoint is rejected",
        validate_launcher_data_mode(good_launcher.replace("entrypoint_args=(--entrypoint /bin/bash)\n", "")),
        "never bypasses the image entrypoint",
    )
    rejects(
        "a launcher that still echoes to stdout is rejected",
        validate_launcher_data_mode(
            good_launcher.replace('info "run_container: selected', 'echo "run_container: selected')
        ),
        "echoes unconditionally to stdout",
    )

    # Make recipe extraction.
    synthetic_makefile = "other:\n\t@true\n\ngemm-cutedsl-p32-smoke:\n\t@echo one >&2\n\t@exit 0\n\nnext:\n\t@true\n"
    extracted = extract_make_recipe(synthetic_makefile, SMOKE_TARGET)
    accepts(
        "a Make recipe is extracted exactly",
        [] if extracted == ["\t@echo one >&2", "\t@exit 0"] else [f"extracted {extracted!r}"],
    )

    # Contract parsing.
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.env"
        path.write_text("# comment\n\nA_KEY=1\n", encoding="utf-8")
        accepts(
            "a well-formed contract file parses",
            [] if parse_env_file(path) == {"A_KEY": "1"} else ["unexpected parse result"],
        )
        path.write_text("A_KEY=1\nA_KEY=2\n", encoding="utf-8")
        try:
            parse_env_file(path)
        except ValueError as exc:
            rejects("a duplicate contract key is rejected", [str(exc)], "duplicate contract key")
        else:
            failures.append("a duplicate contract key is rejected")
            print("  FAIL a duplicate contract key is rejected", file=sys.stderr)

    if failures:
        print(f"SELF-TEST: FAIL ({len(failures)} case(s))", file=sys.stderr)
        return 1
    print("SELF-TEST: PASS", file=sys.stderr)
    return 0


# --- Entry point -------------------------------------------------------------


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if args == ["--self-test"]:
        return run_self_test()
    if len(args) > 1 or (args and args[0].startswith("-")):
        print(
            "usage: check_cutedsl_gemm_p32.py [repository-root]\n"
            "       check_cutedsl_gemm_p32.py --self-test",
            file=sys.stderr,
        )
        return 2

    root = Path(args[0]) if args else Path(__file__).resolve().parents[1]
    print(f"check_cutedsl_gemm_p32: validating P3.2 in {root}", file=sys.stderr)
    try:
        errors = check_wrapper(root)
    except Exception as exc:  # noqa: BLE001 - fail closed and report the cause
        print(f"check_cutedsl_gemm_p32: FAIL: {exc}", file=sys.stderr)
        return 1

    if errors:
        for error in errors:
            print(f"check_cutedsl_gemm_p32: FAIL: {error}", file=sys.stderr)
        print(f"check_cutedsl_gemm_p32: {len(errors)} contract failure(s)", file=sys.stderr)
        return 1
    print("check_cutedsl_gemm_p32: OK", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
