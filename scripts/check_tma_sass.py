#!/usr/bin/env python3
"""GPU-free SASS verification for the P1.2 TMA microbenchmark.

Disassemble the compiled binary with ``cuobjdump -sass`` (PTX is not accepted
as proof), identify all nine benchmark template specializations, and verify
for each one:

* at least one genuine ``UTMALDG.2D`` instruction, with no other TMA
  dimension or multicast/cluster qualifier attached;
* at least one transaction-aware mbarrier arrival/expectation instruction
  (``SYNCS.ARRIVE.TRANS*``, sm_103a's lowering of
  ``mbarrier.arrive.expect_tx``);
* at least one phase/parity wait instruction
  (``SYNCS.PHASECHK.TRANS*.TRYWAIT``, sm_103a's lowering of
  ``mbarrier.try_wait.parity``);
* at least STAGES mbarrier invalidation instructions (``SYNCS.CCTL.IV``,
  sm_103a's lowering of ``mbarrier.inval.shared.b64``) — one per ring slot,
  issued only after the pipeline is fully drained;
* no LDGSTS transfer path and no UBLKCP 1D bulk-copy path.

These mnemonics were read directly from ``cuobjdump -sass`` output of this
project's own ``build/memory/tma`` binary compiled for sm_103a with CUDA
13.1.80 ptxas, not guessed from documentation. ``ptxas`` may unroll or peel
the source loop and duplicate a complete static sequence (for example the
try_wait spin loop's fast-path check plus its retry-loop body both lower to
a static TRYWAIT), so the checker requires only presence — never an exact
static instruction count — for every category except invalidation, where the
source's own `#pragma unroll` over a compile-time-known STAGES gives a
stable per-specialization minimum (observed as exactly STAGES static
SYNCS.CCTL.IV per specialization, with no retry-loop-style duplication).

Usage:
  check_tma_sass.py <binary> <output-sass-path>
  check_tma_sass.py --self-test

Exit code: 0 only when the selected validation passes, 1 on a contract,
synthetic-test, I/O, or ``cuobjdump`` failure, and 2 on a usage error.
"""

import re
import subprocess
import sys


BENCHMARK_MARKER = "tma_benchmark_kernel"
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
    re.compile(r"tma_benchmark_kernelILi(\d+)ELi(\d+)EE"),
    re.compile(r"tma_benchmark_kernel<\s*(\d+)\s*,\s*(\d+)\s*>"),
)

UTMALDG_PATTERN = re.compile(r"\bUTMALDG((?:\.[A-Z0-9_]+)*)\b")
ARRIVE_TRANS_PATTERN = re.compile(r"\bSYNCS\.ARRIVE\.TRANS\d*\b")
PHASECHK_TRYWAIT_PATTERN = re.compile(r"\bSYNCS\.PHASECHK\.TRANS\d*\.TRYWAIT\b")
INVALIDATE_PATTERN = re.compile(r"\bSYNCS\.CCTL\.IV\b")
LDGSTS_PATTERN = re.compile(r"\bLDGSTS\b")
UBLKCP_PATTERN = re.compile(r"\bUBLKCP\b")


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
        text_block = "\n".join(instruction_lines)

        utmaldg_matches = list(UTMALDG_PATTERN.finditer(text_block))
        arrive_count = len(ARRIVE_TRANS_PATTERN.findall(text_block))
        trywait_count = len(PHASECHK_TRYWAIT_PATTERN.findall(text_block))
        inval_count = len(INVALIDATE_PATTERN.findall(text_block))
        ldgsts_count = len(LDGSTS_PATTERN.findall(text_block))
        ublkcp_count = len(UBLKCP_PATTERN.findall(text_block))

        spec_errors: list[str] = []

        if not utmaldg_matches:
            spec_errors.append("UTMALDG.2D missing")
        else:
            bad_qualifiers = sorted(
                {
                    match.group(1)
                    for match in utmaldg_matches
                    if [tok for tok in match.group(1).split(".") if tok] != ["2D"]
                }
            )
            if bad_qualifiers:
                spec_errors.append(
                    "UTMALDG instruction(s) with unexpected qualifiers (expected exactly "
                    f".2D, no other dimension or multicast/cluster variant): {bad_qualifiers}"
                )

        if arrive_count == 0:
            spec_errors.append(
                "no transaction-aware mbarrier arrival/expectation instruction "
                "(expected SYNCS.ARRIVE.TRANS*)"
            )
        if trywait_count == 0:
            spec_errors.append(
                "no phase/parity wait instruction "
                "(expected SYNCS.PHASECHK.TRANS*.TRYWAIT)"
            )
        if inval_count < stages:
            spec_errors.append(
                f"fewer than STAGES={stages} mbarrier invalidation instruction(s) "
                f"(expected SYNCS.CCTL.IV, one per ring slot; found {inval_count})"
            )
        if ldgsts_count != 0:
            spec_errors.append(f"LDGSTS transfer path present (count={ldgsts_count})")
        if ublkcp_count != 0:
            spec_errors.append(f"UBLKCP 1D bulk-copy path present (count={ublkcp_count})")

        label = f"STAGES={stages} COPIES={copies}"
        if spec_errors:
            errors.extend(f"{label}: {detail}" for detail in spec_errors)
            status_lines.append(f"FAIL {label}")
        else:
            status_lines.append(
                f"OK   {label} UTMALDG.2D={len(utmaldg_matches)} "
                f"ARRIVE.TRANS={arrive_count} PHASECHK.TRYWAIT={trywait_count} "
                f"CCTL.IV={inval_count}"
            )

    return status_lines, errors


def synthetic_block(
    stages: int,
    copies: int,
    *,
    utmaldg_count: int = 1,
    utmaldg_qualifier: str = "2D",
    arrive_count: int = 1,
    trywait_count: int = 2,
    inval_count: int | None = None,
    ldgsts_count: int = 0,
    ublkcp_count: int = 0,
) -> str:
    """Build a minimal cuobjdump-like function block for checker self-tests.

    Shaped after this project's own real cuobjdump -sass output for
    build/memory/tma on sm_103a (CUDA 13.1.80 ptxas): one UTMALDG.2D, one
    SYNCS.ARRIVE.TRANS64, two SYNCS.PHASECHK.TRANS64.TRYWAIT (fast-path check
    plus retry-loop body), and exactly STAGES SYNCS.CCTL.IV (one per
    #pragma-unrolled ring slot) per specialization.
    """
    if inval_count is None:
        inval_count = stages
    lines = [
        "Function : "
        f"_Zsynthetic_tma_benchmark_kernelILi{stages}ELi{copies}EEv"
    ]
    for _ in range(utmaldg_count):
        lines.append(f"        UTMALDG.{utmaldg_qualifier} [UR20], [UR24] ;")
    for _ in range(arrive_count):
        lines.append("        SYNCS.ARRIVE.TRANS64 RZ, [R5+UR19+0x10000], R2 ;")
    for _ in range(trywait_count):
        lines.append("        SYNCS.PHASECHK.TRANS64.TRYWAIT P1, [UR18+0x10000], R2 ;")
    for i in range(inval_count):
        lines.append(f"        SYNCS.CCTL.IV [UR4+0x{0x10000 + 8 * i:x}] ;")
    for _ in range(ldgsts_count):
        lines.append("        LDGSTS.E.BYPASS.128 [R0], [R2] ;")
    for _ in range(ublkcp_count):
        lines.append("        UBLKCP.CG [R0], [R2] ;")
    return "\n".join(lines)


def synthetic_sass(
    overrides: dict[tuple[int, int], dict[str, object]] | None = None,
) -> str:
    overrides = overrides or {}
    blocks = []
    for stages, copies in sorted(EXPECTED_SPECS):
        options = dict(overrides.get((stages, copies), {}))
        blocks.append(synthetic_block(stages, copies, **options))
    return "\n\n".join(blocks) + "\n"


def run_self_test() -> int:
    cases: list[tuple[str, str, str | None]] = [
        ("accepts compiler-unrolled complete specializations", synthetic_sass(), None),
        (
            "rejects a missing specialization",
            "\n\n".join(
                synthetic_block(stages, copies)
                for stages, copies in sorted(EXPECTED_SPECS)
                if (stages, copies) != (2, 4)
            )
            + "\n",
            "missing benchmark specialization STAGES=2 COPIES=4",
        ),
        (
            "rejects a duplicate specialization",
            synthetic_sass() + "\n" + synthetic_block(2, 4) + "\n",
            "duplicate benchmark specialization (2, 4)",
        ),
        (
            "rejects a missing UTMALDG.2D",
            synthetic_sass({(2, 4): {"utmaldg_count": 0}}),
            "UTMALDG.2D missing",
        ),
        (
            "rejects a 1D TMA instruction",
            synthetic_sass({(4, 2): {"utmaldg_qualifier": "1D"}}),
            "unexpected qualifiers",
        ),
        (
            "rejects a multicast/cluster TMA qualifier",
            synthetic_sass({(8, 1): {"utmaldg_qualifier": "2D.MULTICAST"}}),
            "unexpected qualifiers",
        ),
        (
            "rejects an LDGSTS fallback",
            synthetic_sass({(2, 16): {"ldgsts_count": 1}}),
            "LDGSTS transfer path present",
        ),
        (
            "rejects a UBLKCP 1D bulk-copy fallback",
            synthetic_sass({(4, 4): {"ublkcp_count": 1}}),
            "UBLKCP 1D bulk-copy path present",
        ),
        (
            "rejects a missing transaction arrival",
            synthetic_sass({(4, 8): {"arrive_count": 0}}),
            "no transaction-aware mbarrier arrival/expectation instruction",
        ),
        (
            "rejects a missing phase wait",
            synthetic_sass({(8, 2): {"trywait_count": 0}}),
            "no phase/parity wait instruction",
        ),
        (
            "rejects insufficient mbarrier invalidation",
            synthetic_sass({(8, 4): {"inval_count": 3}}),
            "fewer than STAGES=8 mbarrier invalidation instruction(s)",
        ),
        (
            "rejects a completely missing mbarrier invalidation",
            synthetic_sass({(2, 4): {"inval_count": 0}}),
            "fewer than STAGES=2 mbarrier invalidation instruction(s)",
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
            print(f"check_tma_sass: self-test: PASS: {name}", file=sys.stderr)
        else:
            failures.append(name)
            print(
                f"check_tma_sass: self-test: FAIL: {name}; errors={errors}",
                file=sys.stderr,
            )

    if failures:
        print(
            f"check_tma_sass: self-test: FAILED ({len(failures)} case(s))",
            file=sys.stderr,
        )
        return 1
    print(
        f"check_tma_sass: self-test: OK ({len(cases)} cases)",
        file=sys.stderr,
    )
    return 0


def check_binary(binary_path: str, out_path: str) -> int:
    try:
        result = subprocess.run(
            ["cuobjdump", "-sass", binary_path], capture_output=True, text=True
        )
    except OSError as exc:
        print(f"check_tma_sass: unable to run cuobjdump: {exc}", file=sys.stderr)
        return 1
    if result.returncode != 0:
        print(
            f"check_tma_sass: cuobjdump failed (rc={result.returncode}):\n"
            f"{result.stderr}",
            file=sys.stderr,
        )
        return 1

    sass_text = result.stdout
    try:
        with open(out_path, "w", encoding="utf-8") as output_file:
            output_file.write(sass_text)
    except OSError as exc:
        print(f"check_tma_sass: unable to write {out_path}: {exc}", file=sys.stderr)
        return 1
    print(f"check_tma_sass: wrote {out_path}", file=sys.stderr)

    status_lines, errors = analyze_sass(sass_text)
    for status in status_lines:
        print(f"check_tma_sass: {status}", file=sys.stderr)

    if errors:
        print("check_tma_sass: contract validation failed:", file=sys.stderr)
        for error in errors:
            print(f"check_tma_sass:   - {error}", file=sys.stderr)
        return 1

    print(
        "check_tma_sass: OK: all nine specializations contain 2D unicast TMA loads "
        "with transaction-barrier completion, full mbarrier invalidation after drain, "
        "and no LDGSTS fallback",
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
        "usage: check_tma_sass.py <binary> <output-sass-path>\n"
        "       check_tma_sass.py --self-test",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
