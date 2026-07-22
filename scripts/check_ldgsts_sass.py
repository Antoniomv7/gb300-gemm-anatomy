#!/usr/bin/env python3
"""GPU-free SASS verification for the P1.1 LDGSTS microbenchmark.

Disassemble the compiled binary with ``cuobjdump -sass`` (PTX is not accepted
as proof), identify all nine benchmark template specializations, and verify
for each one:

* the exact static LDGSTS count implied by ``COPIES``;
* an LDGDEPBAR instruction for ``cp.async.commit_group``; and
* DEPBAR.LE waits for both ``wait_group<STAGES - 1>`` and the final
  ``wait_group 0`` drain.

Usage: check_ldgsts_sass.py <binary> <output-sass-path>
Exit code: 0 only when the full contract passes, 1 on a validation or
``cuobjdump`` failure, and 2 on a usage error.
"""

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


def parse_wait_values(lines: list[str]) -> set[int]:
    values: set[int] = set()
    for line in lines:
        match = WAIT_VALUE_PATTERN.search(line)
        if match:
            token = match.group(1)
            base = 16 if token.lower().startswith("0x") else 10
            values.add(int(token, base))
    return values


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check_ldgsts_sass.py <binary> <output-sass-path>", file=sys.stderr)
        return 2
    binary_path, out_path = sys.argv[1], sys.argv[2]

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

    print(
        "check_ldgsts_sass: found "
        f"{len(candidate_blocks)} benchmark function block(s); "
        f"identified {len(found_specs)}/{len(EXPECTED_SPECS)} expected specializations",
        file=sys.stderr,
    )

    for stages, copies in sorted(EXPECTED_SPECS):
        block = blocks_by_spec.get((stages, copies))
        if block is None:
            continue

        instruction_lines = block[1:]
        ldgsts_lines = [line for line in instruction_lines if LDGSTS_PATTERN.search(line)]
        commit_lines = [line for line in instruction_lines if LDGDEPBAR_PATTERN.search(line)]
        wait_lines = [line for line in instruction_lines if DEPBAR_PATTERN.search(line)]
        wait_values = parse_wait_values(wait_lines)
        expected_wait_values = {0, stages - 1}

        spec_errors: list[str] = []
        if len(ldgsts_lines) != copies:
            spec_errors.append(f"LDGSTS count={len(ldgsts_lines)} expected={copies}")
        if not commit_lines:
            spec_errors.append("LDGDEPBAR missing (commit_group not demonstrated)")
        missing_waits = expected_wait_values - wait_values
        if missing_waits:
            spec_errors.append(
                "DEPBAR.LE SB0 wait value(s) missing: "
                + ", ".join(str(value) for value in sorted(missing_waits))
                + f"; observed={sorted(wait_values)}"
            )

        label = f"STAGES={stages} COPIES={copies}"
        if spec_errors:
            for detail in spec_errors:
                errors.append(f"{label}: {detail}")
            print(f"check_ldgsts_sass: FAIL {label}", file=sys.stderr)
        else:
            print(
                f"check_ldgsts_sass: OK   {label} LDGSTS={len(ldgsts_lines)} "
                f"LDGDEPBAR={len(commit_lines)} waits={sorted(wait_values)}",
                file=sys.stderr,
            )

    if errors:
        print("check_ldgsts_sass: contract validation failed:", file=sys.stderr)
        for error in errors:
            print(f"check_ldgsts_sass:   - {error}", file=sys.stderr)
        return 1

    print(
        "check_ldgsts_sass: OK: all nine specializations have exact LDGSTS "
        "counts plus commit/wait dependency instructions",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
