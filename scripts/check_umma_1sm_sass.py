#!/usr/bin/env python3
"""GPU-free SASS verification for the P2.1 1-SM BF16 UMMA microbenchmark.

Disassemble the compiled binary with ``cuobjdump -sass`` (PTX is not accepted
as proof), identify all twelve ``umma_1sm_m128n{N}k16_d{DEPTH}`` symbols, and
verify for each one:

* exactly the twelve expected (N, depth) specializations exist, with no
  missing, extra, or duplicate configuration;
* the symbol contains ``UTCHMMA`` -- sm_103a's SASS lowering of
  ``tcgen05.mma.cta_group::1.kind::f16``, observed directly (see below), not
  the bare "UTCMMA" substring named informally in the P2.1 task brief;
* the static ``UTCHMMA`` count is exactly ``depth`` and the address spacing
  between consecutive occurrences is uniform, evidencing full compile-time
  unrolling rather than a runtime back-edge standing in for it;
* the burst ends with a genuine completion sequence: a ``UTCBAR`` (tcgen05
  .commit) after the last ``UTCHMMA``, followed by at least one
  ``SYNCS.PHASECHK.TRANS64.TRYWAIT`` (mbarrier.try_wait.parity);
* TMEM allocation (``UTCATOMSWS.FIND_AND_SET.ALIGN``) and deallocation
  (``UVIRTCOUNT.DEALLOC.SMPOOL``) are present, with deallocation ordered
  after the last TMEM read/write;
* TMEM-to-register loading (``LDTM.x32``) is present, with exactly
  N/32 occurrences (one per 32-column fragment);
* no 2-SM/cluster evidence is present in the disassembly or the source, and
  no WGMMA, mma.sync/HMMA, TMA, LDGSTS, FP8/FP4, or sparse-MMA instruction
  is present anywhere in the binary.

Mnemonic provenance (read directly from ``cuobjdump -sass`` output of this
project's own ``build/compute/umma_1sm`` binary compiled for sm_103a with
CUDA 13.1.80 ptxas, not guessed from documentation or the PTX ISA text):

  PTX (source)                                  SASS (sm_103a, this binary)
  tcgen05.mma.cta_group::1.kind::f16            UTCHMMA
  tcgen05.commit.cta_group::1.mbarrier...       UTCBAR
  tcgen05.alloc.cta_group::1...                 UTCATOMSWS.FIND_AND_SET.ALIGN (x2: ptxas
                                                 peels a fast-path check plus a retry-loop
                                                 body, the same duplication pattern already
                                                 documented for TMA's mbarrier wait in
                                                 check_tma_sass.py -- presence, not an exact
                                                 count, is required)
  tcgen05.dealloc.cta_group::1...               UVIRTCOUNT.DEALLOC.SMPOOL
  tcgen05.relinquish_alloc_permit...             (folded into UTCATOMSWS.AND; not checked)
  mbarrier.try_wait.parity                      SYNCS.PHASECHK.TRANS64.TRYWAIT
  mbarrier.inval.shared.b64                     SYNCS.CCTL.IV (same lowering documented for
                                                 P1.2's TMA mbarriers in check_tma_sass.py)
  tcgen05.ld.sync.aligned.32x32b.x32.b32        LDTM.x32
  tcgen05.wait::ld.sync.aligned                 (no distinct SASS instruction -- see below)
  tcgen05.fence::after_thread_sync               (no distinct SASS instruction -- see below)

``tcgen05.wait::ld`` and ``tcgen05.fence::after_thread_sync`` were confirmed
present in the compiled PTX (``nvcc -ptx``, one ``tcgen05.wait::ld`` per
``LDTM.x32`` fragment and one ``tcgen05.fence::after_thread_sync`` per
kernel) but ptxas emits no separate SASS instruction for either on this
pinned toolchain: register scoreboarding (the same mechanism an ordinary
load-then-use dependency relies on) already serializes the load, and the
fence is a pure code-motion constraint with no runtime effect once ptxas's
own scheduling already respects it. Per the P2.1 task brief's own guidance
("If the disassembler does not expose an attribute required to prove a
property, do not invent a check"), this checker proves both instructions'
presence with a static source check (--source, optional) instead of
inventing a SASS signal that does not exist; when --source is not given,
these two checks are skipped and reported as such, never silently assumed.

Usage:
  check_umma_1sm_sass.py --self-test

  check_umma_1sm_sass.py <binary> <output-sass-path> [--source <umma_1sm.cu>]

Exit code: 0 only when the selected validation passes, 1 on a contract,
synthetic-test, I/O, or ``cuobjdump``/source-check failure, and 2 on a usage
error.
"""

import re
import subprocess
import sys


FUNCTION_MARKER = "umma_1sm_m128n"
EXPECTED_NS = (64, 128, 256)
EXPECTED_DEPTHS = (4, 16, 64, 256)
EXPECTED_SPECS = {(n, d) for n in EXPECTED_NS for d in EXPECTED_DEPTHS}

SYMBOL_PATTERN = re.compile(r"\bumma_1sm_m128n(\d+)k16_d(\d+)\b")

UTCHMMA_PATTERN = re.compile(r"\bUTCHMMA\b")
UTCBAR_PATTERN = re.compile(r"\bUTCBAR\b")
TRYWAIT_PATTERN = re.compile(r"\bSYNCS\.PHASECHK\.TRANS\d*\.TRYWAIT\b")
ALLOC_PATTERN = re.compile(r"\bUTCATOMSWS\.FIND_AND_SET\.ALIGN\b")
DEALLOC_PATTERN = re.compile(r"\bUVIRTCOUNT\.DEALLOC\.SMPOOL\b")
INVALIDATE_PATTERN = re.compile(r"\bSYNCS\.CCTL\.IV\b")
LDTM_PATTERN = re.compile(r"\bLDTM\.x32\b")

# Forbidden instruction families (word-boundary anchored so "UTCHMMA" itself
# never matches "\bHMMA\b": there is no boundary before "HMMA" inside
# "UTCHMMA", both being contiguous identifier characters).
FORBIDDEN_PATTERNS = (
    (re.compile(r"\bHMMA\b"), "mma.sync/HMMA (non-tcgen05 Tensor Core MMA)"),
    (re.compile(r"\bWGMMA\b"), "WGMMA (Hopper warpgroup MMA)"),
    (re.compile(r"\bQGMMA\b"), "QGMMA"),
    (re.compile(r"\bIMMA\b"), "IMMA"),
    (re.compile(r"\bBMMA\b"), "BMMA"),
    (re.compile(r"\bUTMALDG\b"), "UTMALDG (TMA load)"),
    (re.compile(r"\bLDGSTS\b"), "LDGSTS"),
    (re.compile(r"\bUBLKCP\b"), "UBLKCP (1D bulk copy)"),
    (re.compile(r"UTCHMMA\.[A-Z0-9_.]*\bSP\b"), "sparse MMA (.sp) qualifier on UTCHMMA"),
)

# A cluster-scoped barrier or an explicit cluster-dimension header attribute
# would be the observable trace of a 2-SM/2-CTA (__cluster_dims__) kernel;
# this binary must show neither. No 2-SM UTCHMMA sample was available to
# compile for direct comparison, so this is combined with the --source
# static check below (documented limitation, not invented SASS evidence).
CLUSTER_BARRIER_PATTERN = re.compile(r"\bBAR\.SYNC\.[A-Z0-9_.]*CLUSTER\b|\bCLUSTER\b")

FORBIDDEN_SOURCE_PATTERNS = (
    (re.compile(r"cta_group::2"), "cta_group::2"),
    (re.compile(r"__cluster_dims__"), "__cluster_dims__"),
    (re.compile(r"multicast", re.IGNORECASE), "multicast"),
    (re.compile(r"\.kind::(?!f16\b)[a-z0-9_]+"), "a non-kind::f16 MMA kind"),
    (re.compile(r"\.sp\b"), "a sparse (.sp) MMA form"),
    (re.compile(r"block_scale"), "block_scale"),
)
REQUIRED_SOURCE_PATTERNS = (
    (re.compile(r"tcgen05\.wait::ld\.sync\.aligned"), "tcgen05.wait::ld.sync.aligned"),
    (re.compile(r"tcgen05\.fence::after_thread_sync"), "tcgen05.fence::after_thread_sync"),
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
    match = SYMBOL_PATTERN.search(header)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def instruction_addresses(pattern: re.Pattern, text_block: str) -> list[int]:
    addrs = []
    for line in text_block.splitlines():
        m = re.match(r"\s*/\*([0-9a-fA-F]+)\*/\s+(?:@!?U?P\d+\s+)?(\S.*)", line)
        if not m:
            continue
        addr_hex, rest = m.group(1), m.group(2)
        if pattern.search(rest):
            addrs.append(int(addr_hex, 16))
    return addrs


def analyze_sass(sass_text: str) -> tuple[list[str], list[str]]:
    """Return human-readable status lines and contract errors."""
    candidate_blocks = [
        block for block in split_function_blocks(sass_text) if FUNCTION_MARKER in block[0]
    ]
    blocks_by_spec: dict[tuple[int, int], list[str]] = {}
    errors: list[str] = []

    for block in candidate_blocks:
        header = block[0].strip()
        spec = parse_specialization(header)
        if spec is None:
            errors.append(f"could not identify N/depth in {header}")
            continue
        if spec in blocks_by_spec:
            errors.append(f"duplicate configuration N={spec[0]} depth={spec[1]}: {header}")
            continue
        blocks_by_spec[spec] = block

    found_specs = set(blocks_by_spec)
    for spec in sorted(EXPECTED_SPECS - found_specs):
        errors.append(f"missing specialization N={spec[0]} depth={spec[1]}")
    for spec in sorted(found_specs - EXPECTED_SPECS):
        errors.append(f"unexpected specialization N={spec[0]} depth={spec[1]}")

    status_lines = [
        f"found {len(candidate_blocks)} function block(s); "
        f"identified {len(found_specs)}/{len(EXPECTED_SPECS)} expected specializations"
    ]

    for n, depth in sorted(EXPECTED_SPECS):
        block = blocks_by_spec.get((n, depth))
        if block is None:
            continue
        text_block = "\n".join(block[1:])
        label = f"N={n} depth={depth}"
        spec_errors: list[str] = []

        utchmma_addrs = instruction_addresses(UTCHMMA_PATTERN, text_block)
        utcbar_addrs = instruction_addresses(UTCBAR_PATTERN, text_block)
        trywait_addrs = instruction_addresses(TRYWAIT_PATTERN, text_block)
        alloc_addrs = instruction_addresses(ALLOC_PATTERN, text_block)
        dealloc_addrs = instruction_addresses(DEALLOC_PATTERN, text_block)
        ldtm_addrs = instruction_addresses(LDTM_PATTERN, text_block)

        if len(utchmma_addrs) != depth:
            spec_errors.append(
                f"UTCHMMA count is {len(utchmma_addrs)}, expected exactly depth={depth}"
            )
        elif depth > 1:
            deltas = {b - a for a, b in zip(utchmma_addrs, utchmma_addrs[1:])}
            if len(deltas) != 1:
                spec_errors.append(
                    f"UTCHMMA occurrences are not uniformly spaced ({sorted(deltas)}); "
                    "a runtime back-edge may be standing in for compile-time unrolling"
                )

        if not alloc_addrs:
            spec_errors.append("no TMEM allocation instruction (expected UTCATOMSWS.FIND_AND_SET.ALIGN)")
        if not dealloc_addrs:
            spec_errors.append("no TMEM deallocation instruction (expected UVIRTCOUNT.DEALLOC.SMPOOL)")
        if not utcbar_addrs:
            spec_errors.append("no tcgen05.commit instruction (expected UTCBAR)")
        if not trywait_addrs:
            spec_errors.append("no mbarrier completion wait (expected SYNCS.PHASECHK.TRANS*.TRYWAIT)")

        if utchmma_addrs and utcbar_addrs:
            if max(utcbar_addrs) <= max(utchmma_addrs):
                spec_errors.append("no UTCBAR (commit) found after the last UTCHMMA")
            elif trywait_addrs and max(trywait_addrs) <= max(utcbar_addrs):
                spec_errors.append("no mbarrier wait found after the commit")

        expected_fragments = n // 32
        if len(ldtm_addrs) != expected_fragments:
            spec_errors.append(
                f"LDTM.x32 count is {len(ldtm_addrs)}, expected exactly N/32={expected_fragments}"
            )

        if dealloc_addrs:
            last_use = max(utchmma_addrs + ldtm_addrs) if (utchmma_addrs or ldtm_addrs) else -1
            if last_use >= 0 and min(dealloc_addrs) <= last_use:
                spec_errors.append("TMEM deallocation is not ordered after the last TMEM use")

        if label:
            pass  # keep label referenced for f-strings below regardless of branch taken

        if spec_errors:
            errors.extend(f"{label}: {detail}" for detail in spec_errors)
            status_lines.append(f"FAIL {label}")
        else:
            status_lines.append(
                f"OK   {label} UTCHMMA={len(utchmma_addrs)} LDTM.x32={len(ldtm_addrs)} "
                f"alloc={len(alloc_addrs)} dealloc={len(dealloc_addrs)}"
            )

    # Whole-binary forbidden-instruction and cluster-evidence checks.
    for pattern, description in FORBIDDEN_PATTERNS:
        if pattern.search(sass_text):
            errors.append(f"forbidden instruction present: {description}")
    if CLUSTER_BARRIER_PATTERN.search(sass_text):
        errors.append("cluster-scoped barrier or CLUSTER attribute present (2-SM evidence)")

    return status_lines, errors


def strip_line_comments(source_text: str) -> str:
    """Drop everything from '//' to end-of-line on every line.

    A deliberately simple heuristic (no string-literal awareness), sufficient
    for this one source file, which contains no '//' inside a string or
    inline-asm literal. This keeps forbidden-pattern matching scoped to code
    that could actually execute, so a comment explaining *why* a qualifier
    (e.g. ".multicast::cluster") is absent does not itself trip the check.
    """
    return "\n".join(line.split("//", 1)[0] for line in source_text.splitlines())


def check_source(source_text: str) -> list[str]:
    errors: list[str] = []
    code_only = strip_line_comments(source_text)
    for pattern, description in FORBIDDEN_SOURCE_PATTERNS:
        if pattern.search(code_only):
            errors.append(f"source contains forbidden pattern: {description}")
    for pattern, description in REQUIRED_SOURCE_PATTERNS:
        if not pattern.search(source_text):
            errors.append(f"source is missing required PTX instruction text: {description}")
    return errors


# ---------------------------------------------------------------------------
# Synthetic SASS for --self-test, shaped after this project's own real
# cuobjdump -sass output for build/compute/umma_1sm on sm_103a (CUDA 13.1.80
# ptxas): one ELECT+UTCHMMA pair per burst position at a uniform 0x60
# spacing, one UTCBAR (commit) plus two TRYWAITs (fast-path check plus
# retry-loop body, matching the same ptxas duplication pattern documented in
# check_tma_sass.py), two ALIGN allocs (same duplication reason), one
# DEALLOC.SMPOOL after N/32 LDTM.x32 fragments.
# ---------------------------------------------------------------------------
def synthetic_block(n: int, depth: int, *, utchmma_count: int | None = None, spacing: int = 0x60,
                     alloc_count: int = 2, dealloc_count: int = 1, commit_count: int = 1,
                     trywait_count: int = 2, ldtm_count: int | None = None,
                     dealloc_before_last_use: bool = False, extra_uneven_gap: bool = False,
                     cluster_marker: bool = False, forbidden_mnemonic: str | None = None,
                     symbol: str | None = None) -> str:
    if utchmma_count is None:
        utchmma_count = depth
    if ldtm_count is None:
        ldtm_count = n // 32

    lines = [f"\t\tFunction : {symbol or f'umma_1sm_m128n{n}k16_d{depth}'}"]
    addr = 0x0A00
    for i in range(alloc_count):
        lines.append(f"        /*{addr:04x}*/                   UTCATOMSWS.FIND_AND_SET.ALIGN UP0, UR5, UR5 ;")
        addr += 0x60

    utchmma_addrs = []
    for i in range(utchmma_count):
        lines.append(f"        /*{addr:04x}*/               @P0 ELECT P1, URZ, PT ;")
        addr += 0x10
        lines.append(
            f"        /*{addr:04x}*/                   UTCHMMA gdesc[UR12], gdesc[UR14], tmem[UR6], "
            f"tmem[UR4], idesc[UR5], {'!UPT' if i == 0 else 'UPT'} ;"
        )
        utchmma_addrs.append(addr)
        addr += spacing - 0x10
        if extra_uneven_gap and i == max(utchmma_count - 2, 0):
            addr += 0x20  # break uniform spacing to exercise the back-edge-evidence check

    if forbidden_mnemonic:
        lines.append(f"        /*{addr:04x}*/                   {forbidden_mnemonic} ;")
        addr += 0x10

    for _ in range(commit_count):
        lines.append(f"        /*{addr:04x}*/                   UTCBAR [UR4], URZ ;")
        addr += 0x10
    for _ in range(trywait_count):
        lines.append(f"        /*{addr:04x}*/                   SYNCS.PHASECHK.TRANS64.TRYWAIT P1, [R12+URZ], R3 ;")
        addr += 0x10

    ldtm_addrs = []
    for _ in range(ldtm_count):
        lines.append(f"        /*{addr:04x}*/                   LDTM.x32 R16, tmem[UR6] ;")
        ldtm_addrs.append(addr)
        addr += 0x10

    if cluster_marker:
        lines.append(f"        /*{addr:04x}*/                   BAR.SYNC.CLUSTER 0x0 ;")
        addr += 0x10

    dealloc_addr = addr
    if dealloc_before_last_use and (utchmma_addrs or ldtm_addrs):
        dealloc_addr = min(utchmma_addrs + ldtm_addrs)
    for _ in range(dealloc_count):
        lines.append(f"        /*{dealloc_addr:04x}*/                   UVIRTCOUNT.DEALLOC.SMPOOL 0x80 ;")
        dealloc_addr += 0x10

    return "\n".join(lines)


def synthetic_sass(overrides: dict[tuple[int, int], dict[str, object]] | None = None,
                    omit: set[tuple[int, int]] | None = None,
                    extra: list[str] | None = None) -> str:
    overrides = overrides or {}
    omit = omit or set()
    blocks = []
    for n, depth in sorted(EXPECTED_SPECS):
        if (n, depth) in omit:
            continue
        options = dict(overrides.get((n, depth), {}))
        blocks.append(synthetic_block(n, depth, **options))
    if extra:
        blocks.extend(extra)
    return "\n\n".join(blocks) + "\n"


def run_self_test() -> int:
    cases: list[tuple[str, str, str | None]] = [
        ("accepts a complete, correctly-shaped set of twelve specializations", synthetic_sass(), None),
        (
            "rejects a missing symbol",
            synthetic_sass(omit={(64, 4)}),
            "missing specialization N=64 depth=4",
        ),
        (
            "rejects an extra/unexpected symbol",
            synthetic_sass(extra=[synthetic_block(64, 4, symbol="umma_1sm_m128n64k16_d999")]),
            "unexpected specialization N=64 depth=999",
        ),
        (
            "rejects a duplicate configuration",
            synthetic_sass() + "\n" + synthetic_block(64, 4) + "\n",
            "duplicate configuration N=64 depth=4",
        ),
        (
            "rejects a missing UTCHMMA burst",
            synthetic_sass({(64, 4): {"utchmma_count": 0}}),
            "UTCHMMA count is 0, expected exactly depth=4",
        ),
        (
            "rejects an incorrect depth (fewer UTCHMMA than depth requires)",
            synthetic_sass({(128, 16): {"utchmma_count": 15}}),
            "UTCHMMA count is 15, expected exactly depth=16",
        ),
        (
            "rejects an incorrect depth (more UTCHMMA than depth requires)",
            synthetic_sass({(256, 64): {"utchmma_count": 65}}),
            "UTCHMMA count is 65, expected exactly depth=64",
        ),
        (
            "rejects a non-uniformly-spaced burst (possible back-edge standing in for unrolling)",
            synthetic_sass({(128, 64): {"extra_uneven_gap": True}}),
            "not uniformly spaced",
        ),
        (
            "rejects a missing commit",
            synthetic_sass({(64, 16): {"commit_count": 0}}),
            "no tcgen05.commit instruction",
        ),
        (
            "rejects a missing mbarrier wait",
            synthetic_sass({(64, 64): {"trywait_count": 0}}),
            "no mbarrier completion wait",
        ),
        (
            "rejects a missing TMEM allocation",
            synthetic_sass({(128, 4): {"alloc_count": 0}}),
            "no TMEM allocation instruction",
        ),
        (
            "rejects a missing TMEM deallocation",
            synthetic_sass({(128, 256): {"dealloc_count": 0}}),
            "no TMEM deallocation instruction",
        ),
        (
            "rejects deallocation ordered before the final TMEM use",
            synthetic_sass({(256, 4): {"dealloc_before_last_use": True}}),
            "TMEM deallocation is not ordered after the last TMEM use",
        ),
        (
            "rejects an incorrect LDTM.x32 fragment count",
            synthetic_sass({(256, 16): {"ldtm_count": 4}}),
            "LDTM.x32 count is 4, expected exactly N/32=8",
        ),
        (
            "rejects a forbidden HMMA instruction anywhere in the binary",
            synthetic_sass({(64, 4): {"forbidden_mnemonic": "HMMA.16816.F32 R0, R4, R8, R0"}}),
            "forbidden instruction present: mma.sync/HMMA",
        ),
        (
            "rejects a forbidden UTMALDG (TMA) instruction anywhere in the binary",
            synthetic_sass({(64, 16): {"forbidden_mnemonic": "UTMALDG.2D [UR20], [UR24]"}}),
            "forbidden instruction present: UTMALDG",
        ),
        (
            "rejects a forbidden LDGSTS instruction anywhere in the binary",
            synthetic_sass({(64, 64): {"forbidden_mnemonic": "LDGSTS.E.BYPASS.128 [R0], [R2]"}}),
            "forbidden instruction present: LDGSTS",
        ),
        (
            "rejects 2-SM/cluster evidence (cluster-scoped barrier)",
            synthetic_sass({(256, 256): {"cluster_marker": True}}),
            "cluster-scoped barrier or CLUSTER attribute present",
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
            print(f"check_umma_1sm_sass: self-test: PASS: {name}", file=sys.stderr)
        else:
            failures.append(name)
            print(f"check_umma_1sm_sass: self-test: FAIL: {name}; errors={errors}", file=sys.stderr)

    source_cases: list[tuple[str, str, str | None]] = [
        (
            "source check accepts required PTX text and no forbidden pattern",
            "tcgen05.wait::ld.sync.aligned;\ntcgen05.fence::after_thread_sync;\n"
            "tcgen05.mma.cta_group::1.kind::f16 [x], a, b, i, p;\n",
            None,
        ),
        (
            "source check rejects cta_group::2",
            "tcgen05.mma.cta_group::2.kind::f16 [x], a, b, i, p;\n",
            "cta_group::2",
        ),
        (
            "source check rejects __cluster_dims__",
            "__global__ __cluster_dims__(2,1,1) void k() {}\n",
            "__cluster_dims__",
        ),
        (
            "source check rejects a missing tcgen05.wait::ld",
            "tcgen05.fence::after_thread_sync;\n",
            "tcgen05.wait::ld.sync.aligned",
        ),
        (
            "source check rejects a missing tcgen05.fence::after_thread_sync",
            "tcgen05.wait::ld.sync.aligned;\n",
            "tcgen05.fence::after_thread_sync",
        ),
    ]
    for name, source_text, expected_error in source_cases:
        errors = check_source(source_text)
        if expected_error is None:
            passed = not errors
        else:
            passed = any(expected_error in error for error in errors)
        if passed:
            print(f"check_umma_1sm_sass: self-test: PASS: {name}", file=sys.stderr)
        else:
            failures.append(name)
            print(f"check_umma_1sm_sass: self-test: FAIL: {name}; errors={errors}", file=sys.stderr)

    total = len(cases) + len(source_cases)
    if failures:
        print(f"check_umma_1sm_sass: self-test: FAILED ({len(failures)}/{total} case(s))", file=sys.stderr)
        return 1
    print(f"check_umma_1sm_sass: self-test: OK ({total} cases)", file=sys.stderr)
    return 0


def check_binary(binary_path: str, out_path: str, source_path: str | None) -> int:
    try:
        result = subprocess.run(["cuobjdump", "-sass", binary_path], capture_output=True, text=True)
    except OSError as exc:
        print(f"check_umma_1sm_sass: unable to run cuobjdump: {exc}", file=sys.stderr)
        return 1
    if result.returncode != 0:
        print(f"check_umma_1sm_sass: cuobjdump failed (rc={result.returncode}):\n{result.stderr}", file=sys.stderr)
        return 1

    sass_text = result.stdout
    try:
        with open(out_path, "w", encoding="utf-8") as output_file:
            output_file.write(sass_text)
    except OSError as exc:
        print(f"check_umma_1sm_sass: unable to write {out_path}: {exc}", file=sys.stderr)
        return 1
    print(f"check_umma_1sm_sass: wrote {out_path}", file=sys.stderr)

    status_lines, errors = analyze_sass(sass_text)
    for status in status_lines:
        print(f"check_umma_1sm_sass: {status}", file=sys.stderr)

    if source_path is not None:
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                source_text = f.read()
        except OSError as exc:
            print(f"check_umma_1sm_sass: unable to read {source_path}: {exc}", file=sys.stderr)
            return 1
        source_errors = check_source(source_text)
        if source_errors:
            errors.extend(source_errors)
        else:
            print(
                "check_umma_1sm_sass: source check OK: tcgen05.wait::ld and "
                "tcgen05.fence::after_thread_sync are present; no forbidden pattern found",
                file=sys.stderr,
            )
    else:
        print(
            "check_umma_1sm_sass: LIMITATION: no --source given, so tcgen05.wait::ld and "
            "tcgen05.fence::after_thread_sync presence was NOT verified (ptxas emits no distinct "
            "SASS instruction for either on this toolchain; see this file's module docstring)",
            file=sys.stderr,
        )

    if errors:
        print("check_umma_1sm_sass: contract validation failed:", file=sys.stderr)
        for error in errors:
            print(f"check_umma_1sm_sass:   - {error}", file=sys.stderr)
        return 1

    print(
        "check_umma_1sm_sass: OK: all twelve specializations contain a genuine 1-SM UTCHMMA burst "
        "of exactly depth instructions, a real commit/wait completion sequence, a complete TMEM "
        "lifecycle, and no forbidden or 2-SM instruction",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args == ["--self-test"]:
        return run_self_test()

    source_path = None
    positional = []
    i = 0
    while i < len(args):
        if args[i] == "--source":
            if i + 1 >= len(args):
                print("check_umma_1sm_sass: --source requires a path argument", file=sys.stderr)
                return 2
            source_path = args[i + 1]
            i += 2
            continue
        positional.append(args[i])
        i += 1

    if len(positional) == 2 and all(not arg.startswith("-") for arg in positional):
        return check_binary(positional[0], positional[1], source_path)

    print(
        "usage: check_umma_1sm_sass.py <binary> <output-sass-path> [--source <umma_1sm.cu>]\n"
        "       check_umma_1sm_sass.py --self-test",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
