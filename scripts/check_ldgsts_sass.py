#!/usr/bin/env python3
"""GPU-free SASS verification for the P1.1 LDGSTS microbenchmark.

Disassemble the compiled binary with ``cuobjdump -sass`` (PTX is not accepted
as proof), identify all nine benchmark template specializations, and verify
for each one:

* one or more complete static groups of ``COPIES`` 16-byte LDGSTS operations;
* one LDGDEPBAR instruction per static group for ``cp.async.commit_group``;
* one ``wait_group<STAGES - 1>`` per static group; and
* exactly one final ``wait_group<0>`` drain.

``ptxas`` may unroll or peel the source loop and therefore emit more than one
static copy group. The checker validates every emitted group instead of
assuming that the total static LDGSTS count must equal the source-level
``COPIES`` value.

Usage:
  check_ldgsts_sass.py <binary> <output-sass-path>
  check_ldgsts_sass.py --self-test

Exit code: 0 only when the selected validation passes, 1 on a contract,
synthetic-test, I/O, or ``cuobjdump`` failure, and 2 on a usage error.
"""

from collections import Counter
import re
import subprocess
import sys


BENCHMARK_MARKER = "ldgsts_benchmark_kernel"
EXPECTED_SPECS = {
    (2, 4),
    (2, 8),
    (2, 16),
    (4, 2),
    (4, 4),
    (4, 8),
    (8, 1),
    (8, 2),
    (8, 4),
}

# Static group multiplicities observed from CUDA 13.1.80 for the nine frozen
# specializations. They are test fixtures, not production expectations: the
# production gate deliberately accepts any positive multiplicity whose groups
# remain complete and whose commit/wait counts match.
SELF_TEST_GROUPS = {
    (2, 4): 3,
    (2, 8): 3,
    (2, 16): 1,
    (4, 2): 7,
    (4, 4): 3,
    (4, 8): 3,
    (8, 1): 7,
    (8, 2): 7,
    (8, 4): 3,
}

# cuobjdump normally prints mangled CUDA symbols, but accepting a demangled
# spelling makes the checker resilient to output-mode changes.
SPEC_PATTERNS = (
    re.compile(r"ldgsts_benchmark_kernelILi(\d+)ELi(\d+)EE"),
    re.compile(r"ldgsts_benchmark_kernel<\s*(\d+)\s*,\s*(\d+)\s*>")
)

LDGSTS_PATTERN = re.compile(r"\bLDGSTS(?:\.[A-Z0-9_]+)*\b")
LDGDEPBAR_PATTERN = re.compile(r"\bLDGDEPBAR(?:\.[A-Z0-9_]+)*\b")
DEPBAR_PATTERN = re.compile(r"\bDEPBAR(?:\.[A-Z0-9_]+)*\b")
WAIT_VALUE_PATTERN = re.compile(
    r"\bDEPBAR\.LE\s+SB0\s*,\s*(0[xX][0-9A-Fa-f]+|\d+)\b"
)


def split_function_blocks(sass_text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for line in sass_text.splitlines():
        if "Function :" in line:
            if current:
                blocks.append(current)
            current = [line]
        elif current:
            current.append(line)
    if current:
        blocks.append(current)
    return blocks


def parse_specialization(header: str) -> tuple[int, int] | None:
    for pattern in SPEC_PATTERNS:
        match = pattern.search(header)
        if match:
            return int(match.group(1)), int(match.group(2))
    return None


def parse_wait_values(lines: list[str]) -> list[int]:
    values: list[int] = []
    for line in lines:
        match = WAIT_VALUE_PATTERN.search(line)
        if match:
            token = match.group(1)
            base = 16 if token.lower().startswith("0x") else 10
            values.append(int(token, base))
    return values


def analyze_sass(sass_text: str) -> tuple[list[str], list[str]]:
    """Return human-readable status lines and contract errors."""
    candidate_blocks = [
        block
        for block in split_function_blocks(sass_text)
        if BENCHMARK_MARKER in block[0]
    ]
    blocks_by_spec: dict[tuple[int, int], list[str]] = {}
    errors: list[str] = []

    for block in candidate_blocks:
        header = block[0].strip()
        spec = parse_specialization(header)
        if spec is None:
            errors.append(f"could not identify STAGES/COPIES in {header}")
            continue
        if spec in blocks_by_spec:
            errors.append(f"duplicate benchmark specialization {spec}: {header}")
            continue
        blocks_by_spec[spec] = block

    found_specs = set(blocks_by_spec)
    for spec in sorted(EXPECTED_SPECS - found_specs):
        errors.append(f"missing benchmark specialization STAGES={spec[0]} COPIES={spec[1]}")
    for spec in sorted(found_specs - EXPECTED_SPECS):
        errors.append(f"unexpected benchmark specialization STAGES={spec[0]} COPIES={spec[1]}")

    status_lines = [
        "found "
        f"{len(candidate_blocks)} benchmark function block(s); "
        f"identified {len(found_specs)}/{len(EXPECTED_SPECS)} expected specializations"
    ]

    for stages, copies in sorted(EXPECTED_SPECS):
        block = blocks_by_spec.get((stages, copies))
        if block is None:
            continue

        instruction_lines = block[1:]
        ldgsts_matches = [
            (line, match.group(0))
            for line in instruction_lines
            if (match := LDGSTS_PATTERN.search(line))
        ]
        commit_lines = [line for line in instruction_lines if LDGDEPBAR_PATTERN.search(line)]
        wait_lines = [line for line in instruction_lines if DEPBAR_PATTERN.search(line)]
        wait_counts = Counter(parse_wait_values(wait_lines))
        ldgsts_count = len(ldgsts_matches)

        spec_errors: list[str] = []
        static_groups: int | None = None
        if ldgsts_count == 0:
            spec_errors.append("LDGSTS missing")
        elif ldgsts_count % copies != 0:
            spec_errors.append(
                f"LDGSTS count={ldgsts_count} is not a whole number of "
                f"complete {copies}-copy group(s)"
            )
        else:
            static_groups = ldgsts_count // copies

        non_128_tokens = [
            token
            for _, token in ldgsts_matches
            if "128" not in token.split(".")[1:]
        ]
        if non_128_tokens:
            spec_errors.append(
                f"non-128-bit LDGSTS instruction(s)={len(non_128_tokens)}; "
                f"tokens={sorted(set(non_128_tokens))}"
            )

        steady_wait = stages - 1
        if static_groups is not None:
            if len(commit_lines) != static_groups:
                spec_errors.append(
                    f"LDGDEPBAR count={len(commit_lines)} expected={static_groups}"
                )
            if wait_counts[steady_wait] != static_groups:
                spec_errors.append(
                    f"wait_group<{steady_wait}> count={wait_counts[steady_wait]} "
                    f"expected={static_groups}"
                )
        if wait_counts[0] != 1:
            spec_errors.append(f"wait_group<0> count={wait_counts[0]} expected=1")

        unexpected_waits = sorted(set(wait_counts) - {0, steady_wait})
        if unexpected_waits:
            spec_errors.append(
                "unexpected DEPBAR.LE SB0 wait value(s): "
                + ", ".join(str(value) for value in unexpected_waits)
            )

        label = f"STAGES={stages} COPIES={copies}"
        if spec_errors:
            errors.extend(f"{label}: {detail}" for detail in spec_errors)
            status_lines.append(f"FAIL {label}")
        else:
            status_lines.append(
                f"OK   {label} LDGSTS={ldgsts_count} static_groups={static_groups} "
                f"LDGDEPBAR={len(commit_lines)} waits={dict(sorted(wait_counts.items()))}"
            )

    return status_lines, errors


def synthetic_block(
    stages: int,
    copies: int,
    *,
    groups: int,
    ldgsts_count: int | None = None,
    non_128_count: int = 0,
    commit_count: int | None = None,
    steady_wait_count: int | None = None,
    drain_wait_count: int = 1,
    extra_wait_values: tuple[int, ...] = (),
) -> str:
    """Build a minimal cuobjdump-like function block for checker self-tests."""
    if ldgsts_count is None:
        ldgsts_count = groups * copies
    if commit_count is None:
        commit_count = groups
    if steady_wait_count is None:
        steady_wait_count = groups

    lines = [
        "Function : "
        f"_Zsynthetic_ldgsts_benchmark_kernelILi{stages}ELi{copies}EEv"
    ]
    for index in range(ldgsts_count):
        width = 64 if index < non_128_count else 128
        lines.append(f"        LDGSTS.E.BYPASS.{width} [R0], [R2] ;")
    lines.extend("        LDGDEPBAR ;" for _ in range(commit_count))
    lines.extend(
        f"        DEPBAR.LE SB0, 0x{stages - 1:x} ;"
        for _ in range(steady_wait_count)
    )
    lines.extend("        DEPBAR.LE SB0, 0x0 ;" for _ in range(drain_wait_count))
    lines.extend(
        f"        DEPBAR.LE SB0, 0x{value:x} ;" for value in extra_wait_values
    )
    return "\n".join(lines)


def synthetic_sass(
    overrides: dict[tuple[int, int], dict[str, object]] | None = None,
) -> str:
    overrides = overrides or {}
    blocks = []
    for stages, copies in sorted(EXPECTED_SPECS):
        options = dict(overrides.get((stages, copies), {}))
        blocks.append(
            synthetic_block(
                stages,
                copies,
                groups=SELF_TEST_GROUPS[(stages, copies)],
                **options,
            )
        )
    return "\n\n".join(blocks) + "\n"


def run_self_test() -> int:
    cases: list[tuple[str, str, str | None]] = [
        ("accepts compiler-unrolled complete groups", synthetic_sass(), None),
        (
            "rejects an incomplete copy group",
            synthetic_sass({(2, 4): {"ldgsts_count": 11}}),
            "not a whole number of complete 4-copy group(s)",
        ),
        (
            "rejects a non-128-bit LDGSTS",
            synthetic_sass({(2, 8): {"non_128_count": 1}}),
            "non-128-bit LDGSTS",
        ),
        (
            "rejects a missing commit",
            synthetic_sass({(4, 2): {"commit_count": 6}}),
            "LDGDEPBAR count=6 expected=7",
        ),
        (
            "rejects a missing steady-state wait",
            synthetic_sass({(8, 1): {"steady_wait_count": 6}}),
            "wait_group<7> count=6 expected=7",
        ),
        (
            "rejects a missing final drain",
            synthetic_sass({(2, 16): {"drain_wait_count": 0}}),
            "wait_group<0> count=0 expected=1",
        ),
        (
            "rejects an unexpected wait value",
            synthetic_sass({(4, 4): {"extra_wait_values": (1,)}}),
            "unexpected DEPBAR.LE SB0 wait value(s): 1",
        ),
    ]

    failures: list[str] = []
    for name, sass_text, expected_error in cases:
        _, errors = analyze_sass(sass_text)
        if expected_error is None:
            passed = not errors
        else:
            passed = any(expected_error in error for error in errors)
        if passed:
            print(f"check_ldgsts_sass: self-test: PASS: {name}", file=sys.stderr)
        else:
            failures.append(name)
            print(
                f"check_ldgsts_sass: self-test: FAIL: {name}; errors={errors}",
                file=sys.stderr,
            )

    if failures:
        print(
            f"check_ldgsts_sass: self-test: FAILED ({len(failures)} case(s))",
            file=sys.stderr,
        )
        return 1
    print(
        f"check_ldgsts_sass: self-test: OK ({len(cases)} cases)",
        file=sys.stderr,
    )
    return 0


def check_binary(binary_path: str, out_path: str) -> int:
    try:
        result = subprocess.run(
            ["cuobjdump", "-sass", binary_path], capture_output=True, text=True
        )
    except OSError as exc:
        print(f"check_ldgsts_sass: unable to run cuobjdump: {exc}", file=sys.stderr)
        return 1
    if result.returncode != 0:
        print(
            f"check_ldgsts_sass: cuobjdump failed (rc={result.returncode}):\n"
            f"{result.stderr}",
            file=sys.stderr,
        )
        return 1

    sass_text = result.stdout
    try:
        with open(out_path, "w", encoding="utf-8") as output_file:
            output_file.write(sass_text)
    except OSError as exc:
        print(f"check_ldgsts_sass: unable to write {out_path}: {exc}", file=sys.stderr)
        return 1
    print(f"check_ldgsts_sass: wrote {out_path}", file=sys.stderr)

    status_lines, errors = analyze_sass(sass_text)
    for status in status_lines:
        print(f"check_ldgsts_sass: {status}", file=sys.stderr)

    if errors:
        print("check_ldgsts_sass: contract validation failed:", file=sys.stderr)
        for error in errors:
            print(f"check_ldgsts_sass:   - {error}", file=sys.stderr)
        return 1

    print(
        "check_ldgsts_sass: OK: all nine specializations have complete "
        "16-byte LDGSTS groups plus matching commit/wait dependency instructions",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args == ["--self-test"]:
        return run_self_test()
    if len(args) == 2 and all(not arg.startswith("-") for arg in args):
        return check_binary(args[0], args[1])
    print(
        "usage: check_ldgsts_sass.py <binary> <output-sass-path>\n"
        "       check_ldgsts_sass.py --self-test",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
