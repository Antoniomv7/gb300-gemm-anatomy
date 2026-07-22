#!/usr/bin/env python3
"""GPU-free SASS verification for the P1.1 LDGSTS microbenchmark.

Disassembles the compiled build/memory/ldgsts binary with `cuobjdump -sass`
(never accepting the .ptx as sufficient proof), then checks that the LDGSTS
opcode is present in every one of the nine benchmark-kernel specializations,
printing the dependency/barrier instructions found alongside each one.

Usage: check_ldgsts_sass.py <binary> <output-sass-path>
Exit code: 0 if LDGSTS is present in all nine benchmark specializations,
1 otherwise (missing opcode, wrong specialization count, or cuobjdump
failure), 2 on a usage error.
"""
import re
import subprocess
import sys

EXPECTED_SPECS = 9
BENCHMARK_MARKER = "ldgsts_benchmark_kernel"
BARRIER_PATTERN = re.compile(r"\b(DEPBAR|BAR\.\w+|MEMBAR\.\w*|ARRIVES\.\w*)\b")


def split_function_blocks(sass_text: str):
    blocks = []
    current = []
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


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: check_ldgsts_sass.py <binary> <output-sass-path>", file=sys.stderr)
        return 2
    binary_path, out_path = sys.argv[1], sys.argv[2]

    result = subprocess.run(["cuobjdump", "-sass", binary_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"check_ldgsts_sass: cuobjdump failed (rc={result.returncode}):\n{result.stderr}",
              file=sys.stderr)
        return 1
    sass_text = result.stdout
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(sass_text)
    print(f"check_ldgsts_sass: wrote {out_path}", file=sys.stderr)

    benchmark_blocks = [b for b in split_function_blocks(sass_text) if BENCHMARK_MARKER in b[0]]
    print(f"check_ldgsts_sass: found {len(benchmark_blocks)} benchmark kernel specialization(s) "
          f"in the disassembly (expected {EXPECTED_SPECS})", file=sys.stderr)

    missing = []
    for block in benchmark_blocks:
        header = block[0].strip()
        body = "\n".join(block)
        if "LDGSTS" not in body:
            missing.append(header)
            continue
        print(f"check_ldgsts_sass: OK  {header}", file=sys.stderr)
        for line in block:
            if BARRIER_PATTERN.search(line):
                print(f"check_ldgsts_sass:     barrier: {line.strip()}", file=sys.stderr)

    ok = True
    if len(benchmark_blocks) != EXPECTED_SPECS:
        print(f"check_ldgsts_sass: FAIL: expected {EXPECTED_SPECS} benchmark specializations, "
              f"found {len(benchmark_blocks)}", file=sys.stderr)
        ok = False
    if missing:
        print("check_ldgsts_sass: FAIL: LDGSTS opcode missing in:", file=sys.stderr)
        for header in missing:
            print(f"check_ldgsts_sass:   {header}", file=sys.stderr)
        ok = False

    if not ok:
        return 1
    print("check_ldgsts_sass: OK: LDGSTS present in all benchmark specializations", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
