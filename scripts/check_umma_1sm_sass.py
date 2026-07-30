#!/usr/bin/env python3
"""GPU-free SASS and source verification for the P2.1 1-SM BF16 UMMA microbenchmark.

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
  tcgen05.dealloc.cta_group::1...                UVIRTCOUNT.DEALLOC.SMPOOL
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
own scheduling already respects it. This checker therefore proves both
instructions' presence with a mandatory static source check instead of
inventing a SASS signal that does not exist.

Source validation is mandatory, not optional: the two-positional-argument
invocation (``<binary> <output-sass-path>``) always validates the canonical
source ``src/compute/umma_1sm.cu``, resolved relative to this script (never
the caller's current directory). ``--source <path>`` may override which file
is checked (e.g. for testing), but omitting it never skips the check -- there
is no code path in which the real binary/SASS check can return success while
source validation was skipped. If the canonical source cannot be found,
opened, or safely lexically scanned (e.g. an unterminated block comment or
string literal), this checker exits 1.

The source scanner strips both ``//`` and ``/* ... */`` comments while
preserving the exact text of every string and character literal (required
inline PTX text lives inside C++ string literals passed to inline asm), so a
comment can satisfy neither a required-pattern check nor accidentally trip a
forbidden-pattern check. All forbidden- and required-pattern checks below run
against this comment-stripped, literal-preserving view of the source, never
against the raw text.

Beyond the original TMA/WGMMA/etc. forbidden-instruction checks, the source
gate structurally proves that the repaired per-warp TMEM load address is real
executable code: the helper's returned value must combine its allocation base
with both the shifted warp-lane contribution and the fragment-column
contribution, and that returned value must feed the actual
``tcgen05_ld_32x32b_x32`` call. It also proves that the launch-contract
predicate is negated on the rejection path before the first
``__syncthreads()``, and that the rejection path writes failure status and
returns. Finally, it locates both ``%clock64`` reads inside their actual
``timing_mode == TimingMode::kTimed`` lexical scopes and checks the self-test,
pre-validation, warm-up, and timed-repetition routes individually.

Usage:
  check_umma_1sm_sass.py --self-test

  check_umma_1sm_sass.py <binary> <output-sass-path> [--source <umma_1sm.cu>]

Exit code: 0 only when the selected validation passes, 1 on a contract,
synthetic-test, I/O, source-scan, or ``cuobjdump``/source-check failure, and
2 on a usage error.
"""

import re
import subprocess
import sys
from pathlib import Path


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
# compile for direct comparison, so this is combined with the source-level
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

# TMEM load address construction (repair brief section 7.3): the source gate
# must prove that the repaired per-warp lane/column addressing is real
# executable code, not merely a comment, and that the original defective
# direct operand is gone. Exact values are checked (not just presence of the
# constant names) so a regression that keeps the names but changes a value
# -- e.g. an incorrect lane shift -- is still caught.
REQUIRED_TMEM_ADDRESS_PATTERNS = (
    (re.compile(r"kTmemLaneShift\s*=\s*16\b"), "kTmemLaneShift defined as 16 (TMEM lane index occupies bits 31-16)"),
    (re.compile(r"kTmemRowsPerWarp\s*=\s*32\b"), "kTmemRowsPerWarp defined as 32 (rows per warp)"),
    (re.compile(r"kTmemColsPerFragment\s*=\s*32\b"), "kTmemColsPerFragment defined as 32 (columns per fragment)"),
    (re.compile(r"\bmake_tmem_load_address\s*\("), "an executable make_tmem_load_address(...) helper"),
    (re.compile(r"warp_id\)\s*\*\s*kTmemRowsPerWarp\b"),
     "warp contribution using warp_id * kTmemRowsPerWarp in the lane bits"),
    (re.compile(r"<<\s*kTmemLaneShift\b"), "the lane contribution shifted into bits 31-16 by kTmemLaneShift"),
    (re.compile(r"frag\)\s*\*\s*kTmemColsPerFragment\b"),
     "fragment contribution using frag * kTmemColsPerFragment in the column bits"),
    (re.compile(r"tcgen05_ld_32x32b_x32\(\s*make_tmem_load_address\("),
     "the TMEM load operand built by make_tmem_load_address(...)"),
)
FORBIDDEN_TMEM_ADDRESS_PATTERNS = (
    (re.compile(r"tmem_d\s*\+\s*frag\s*\*\s*32\b"), "the original defective direct operand tmem_d + frag * 32"),
)

DEFAULT_SOURCE_RELATIVE_PARTS = ("src", "compute", "umma_1sm.cu")


def resolve_default_source_path() -> Path:
    """The canonical P2.1 source, resolved relative to this checker script
    (never the caller's current working directory), so the two-positional-
    argument invocation always validates the real repository source.
    """
    root = Path(__file__).resolve().parent.parent
    return root.joinpath(*DEFAULT_SOURCE_RELATIVE_PARTS)


class SourceScanError(Exception):
    """Raised when the comment/literal scanner cannot safely determine which
    text is executable: an unterminated block comment or an unterminated
    string/character literal. Callers must treat this as a hard failure, not
    a skip.
    """


def strip_comments_preserving_literals(source_text: str) -> str:
    """Remove '//' and '/* ... */' comments while preserving the exact text
    of every string and character literal (escaped characters included).

    This is necessary because required inline PTX text lives inside C++
    string literals passed to inline asm, and because a comment must never
    be able to satisfy a required-pattern check or accidentally trip a
    forbidden-pattern check. Raises SourceScanError on an unterminated block
    comment or string/char literal, since the lexical state cannot then be
    safely determined.
    """
    out: list[str] = []
    i = 0
    n = len(source_text)
    while i < n:
        two = source_text[i:i + 2]
        if two == "//":
            newline = source_text.find("\n", i)
            if newline == -1:
                i = n
            else:
                out.append("\n")
                i = newline + 1
            continue
        if two == "/*":
            end = source_text.find("*/", i + 2)
            if end == -1:
                raise SourceScanError("unterminated /* block comment (no matching */)")
            out.append("\n" * source_text.count("\n", i, end + 2))
            i = end + 2
            continue
        c = source_text[i]
        if c in ("\"", "'"):
            quote = c
            j = i + 1
            closed = False
            while j < n:
                cj = source_text[j]
                if cj == "\\" and j + 1 < n:
                    j += 2
                    continue
                if cj == quote:
                    j += 1
                    closed = True
                    break
                if cj == "\n":
                    break
                j += 1
            if not closed:
                raise SourceScanError(f"unterminated {quote!r} literal")
            out.append(source_text[i:j])
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out)


class SourceStructureError(Exception):
    """Raised when a required function or braced scope cannot be identified
    uniquely and safely in the comment-free source.
    """


def find_matching_brace(code_only: str, open_brace: int) -> int:
    """Return the closing brace paired with ``open_brace``.

    ``code_only`` has already had comments removed, but inline PTX and
    diagnostics remain as C/C++ string literals. Braces inside those literals
    must not affect structural scope checks.
    """
    if open_brace < 0 or open_brace >= len(code_only) or code_only[open_brace] != "{":
        raise SourceStructureError("internal source-check error: expected an opening brace")

    depth = 0
    i = open_brace
    while i < len(code_only):
        c = code_only[i]
        if c in ("\"", "'"):
            quote = c
            i += 1
            while i < len(code_only):
                if code_only[i] == "\\" and i + 1 < len(code_only):
                    i += 2
                    continue
                if code_only[i] == quote:
                    i += 1
                    break
                i += 1
            else:
                raise SourceStructureError(f"unterminated {quote!r} literal while matching braces")
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
            if depth < 0:
                break
        i += 1

    raise SourceStructureError("unterminated braced scope")


def extract_single_function_body(
    code_only: str, definition_pattern: re.Pattern, description: str
) -> tuple[str, int, int]:
    """Extract one uniquely identifiable function body.

    The returned start/end offsets delimit the body contents (excluding the
    braces) in ``code_only``. Requiring exactly one definition prevents an
    unused duplicate helper from satisfying a gate while the live definition
    has regressed.
    """
    matches = list(definition_pattern.finditer(code_only))
    if len(matches) != 1:
        raise SourceStructureError(
            f"expected exactly one {description} definition, found {len(matches)}"
        )
    open_brace = matches[0].end() - 1
    close_brace = find_matching_brace(code_only, open_brace)
    return code_only[open_brace + 1:close_brace], open_brace + 1, close_brace


def extract_single_control_block(
    code_only: str,
    header_pattern: re.Pattern,
    description: str,
) -> tuple[str, int, int]:
    """Extract the body of one uniquely identifiable ``if``/``for`` block."""
    matches = list(header_pattern.finditer(code_only))
    if len(matches) != 1:
        raise SourceStructureError(
            f"expected exactly one {description} block, found {len(matches)}"
        )
    open_brace = matches[0].end() - 1
    close_brace = find_matching_brace(code_only, open_brace)
    return code_only[open_brace + 1:close_brace], open_brace + 1, close_brace


TMEM_HELPER_DEFINITION = re.compile(
    r"\buint32_t\s+make_tmem_load_address\s*\([^;{}]*\)\s*\{", re.DOTALL
)
LAUNCH_PREDICATE_DEFINITION = re.compile(
    r"\bbool\s+launch_contract_is_valid\s*\(\s*\)\s*\{", re.DOTALL
)
UMMA_BODY_DEFINITION = re.compile(
    r"\bvoid\s+umma_1sm_body\s*\([^;{}]*\)\s*\{", re.DOTALL
)
RUN_ONCE_DEFINITION = re.compile(
    r"\bRunResult\s+run_once\s*\([^;{}]*\)\s*\{", re.DOTALL
)
RUN_UNTIMED_DEFINITION = re.compile(
    r"\bvoid\s+run_untimed_or_die\s*\([^;{}]*\)\s*\{", re.DOTALL
)
RUN_TIMED_DEFINITION = re.compile(
    r"\bunsigned\s+long\s+long\s+run_timed_or_die\s*\([^;{}]*\)\s*\{", re.DOTALL
)
SELF_TEST_DEFINITION = re.compile(
    r"\bint\s+run_self_test\s*\(\s*\)\s*\{", re.DOTALL
)
MAIN_DEFINITION = re.compile(
    r"\bint\s+main\s*\([^;{}]*\)\s*\{", re.DOTALL
)


def check_tmem_address_construction(code_only: str) -> list[str]:
    """Prove that the live helper's returned address and the live load call
    both use the repaired per-warp/per-fragment mapping.
    """
    errors: list[str] = []
    try:
        helper_body, _, _ = extract_single_function_body(
            code_only, TMEM_HELPER_DEFINITION, "make_tmem_load_address"
        )
    except SourceStructureError as exc:
        return [f"invalid TMEM address construction: {exc}"]

    lane_definition = re.compile(
        r"\bconst\s+uint32_t\s+lane_contribution\s*=\s*"
        r"\(\s*static_cast<uint32_t>\s*\(\s*warp_id\s*\)\s*\*\s*"
        r"kTmemRowsPerWarp\s*\)\s*<<\s*kTmemLaneShift\s*;"
    )
    column_definition = re.compile(
        r"\bconst\s+uint32_t\s+column_contribution\s*=\s*"
        r"static_cast<uint32_t>\s*\(\s*frag\s*\)\s*\*\s*"
        r"kTmemColsPerFragment\s*;"
    )
    returned_address = re.compile(
        r"\breturn\s+tmem_base\s*\+\s*lane_contribution\s*\+\s*"
        r"column_contribution\s*;"
    )
    if not lane_definition.search(helper_body):
        errors.append(
            "TMEM helper does not define lane_contribution from "
            "(warp_id * kTmemRowsPerWarp) << kTmemLaneShift"
        )
    if not column_definition.search(helper_body):
        errors.append(
            "TMEM helper does not define column_contribution from "
            "frag * kTmemColsPerFragment"
        )
    if not returned_address.search(helper_body):
        errors.append(
            "TMEM helper return must combine tmem_base + lane_contribution + "
            "column_contribution"
        )

    try:
        umma_body, _, _ = extract_single_function_body(
            code_only, UMMA_BODY_DEFINITION, "umma_1sm_body"
        )
    except SourceStructureError as exc:
        errors.append(f"invalid TMEM load call site: {exc}")
        return errors

    all_load_calls = len(re.findall(r"\btcgen05_ld_32x32b_x32\s*\(", umma_body))
    repaired_load_calls = len(
        re.findall(
            r"\btcgen05_ld_32x32b_x32\s*\(\s*"
            r"make_tmem_load_address\s*\(\s*tmem_d\s*,\s*warp_id\s*,\s*frag\s*\)"
            r"\s*,\s*regs\s*\)",
            umma_body,
        )
    )
    if all_load_calls != 1 or repaired_load_calls != 1:
        errors.append(
            "umma_1sm_body must contain exactly one TMEM load call whose actual "
            "address operand is make_tmem_load_address(tmem_d, warp_id, frag)"
        )
    return errors


def check_launch_guard_ordering(code_only: str) -> list[str]:
    """Prove the predicate and its negative, observable rejection path."""
    errors: list[str] = []
    try:
        predicate_body, _, _ = extract_single_function_body(
            code_only, LAUNCH_PREDICATE_DEFINITION, "launch_contract_is_valid"
        )
    except SourceStructureError as exc:
        errors.append(f"invalid launch-contract predicate: {exc}")
        return errors

    return_matches = re.findall(r"\breturn\s+([^;]+);", predicate_body, re.DOTALL)
    expected_predicate = (
        "gridDim.x==kExpectedGridDim&&gridDim.y==kExpectedGridDim&&"
        "gridDim.z==kExpectedGridDim&&blockDim.x==kExpectedBlockDimX&&"
        "blockDim.y==1&&blockDim.z==1"
    )
    if len(return_matches) != 1 or re.sub(r"\s+", "", return_matches[0]) != expected_predicate:
        errors.append(
            "launch_contract_is_valid() must return the exact grid=(1,1,1), "
            "block=(128,1,1) conjunction"
        )

    try:
        umma_body, _, _ = extract_single_function_body(
            code_only, UMMA_BODY_DEFINITION, "umma_1sm_body"
        )
    except SourceStructureError as exc:
        errors.append(f"invalid launch-contract guard: {exc}")
        return errors

    first_sync = umma_body.find("__syncthreads()")
    if first_sync == -1:
        errors.append(
            "missing launch-contract ordering anchor: umma_1sm_body has no __syncthreads()"
        )
        return errors
    prefix = umma_body[:first_sync]
    negative_guard = re.compile(
        r"\bif\s*\(\s*!\s*launch_contract_is_valid\s*\(\s*\)\s*\)\s*\{",
        re.DOTALL,
    )
    try:
        rejected_body, _, rejected_end = extract_single_control_block(
            prefix, negative_guard, "negative launch-contract rejection"
        )
    except SourceStructureError as exc:
        errors.append(
            "missing launch-contract guard: expected "
            "if (!launch_contract_is_valid()) before the first __syncthreads(): "
            f"{exc}"
        )
        return errors

    if not re.search(r"\bg_launch_ok\s*\[\s*0\s*\]\s*=\s*0\s*;", rejected_body):
        errors.append("launch-contract rejection must write g_launch_ok[0] = 0")
    if not re.search(r"\breturn\s*;", rejected_body):
        errors.append("launch-contract rejection must return before synchronization")

    accepted_prefix = prefix[rejected_end + 1:]
    if not re.search(r"\bg_launch_ok\s*\[\s*0\s*\]\s*=\s*1\s*;", accepted_prefix):
        errors.append(
            "accepted launch path must write g_launch_ok[0] = 1 before the first "
            "__syncthreads()"
        )
    return errors


def check_timing_routing(code_only: str) -> list[str]:
    """Prove lexical clock guards and every host orchestration route.

    Counting guard tokens is insufficient: two empty timed guards elsewhere
    in the function must not legitimize unconditional clock reads. Likewise,
    one untimed call elsewhere must not legitimize a timed self-test.
    """
    errors: list[str] = []

    try:
        umma_body, _, _ = extract_single_function_body(
            code_only, UMMA_BODY_DEFINITION, "umma_1sm_body"
        )
    except SourceStructureError as exc:
        return [f"invalid timing structure: {exc}"]

    clock_positions = [match.start() for match in re.finditer(r"%%clock64\b", umma_body)]
    if len(clock_positions) != 2:
        errors.append(
            f"umma_1sm_body contains {len(clock_positions)} %clock64 read(s); expected exactly "
            "two (start and end)"
        )

    timed_guard_pattern = re.compile(
        r"\bif\s*\(\s*timing_mode\s*==\s*TimingMode::kTimed\s*\)\s*\{",
        re.DOTALL,
    )
    timed_guard_matches = list(timed_guard_pattern.finditer(umma_body))
    guard_scopes: list[tuple[int, int, str]] = []
    if len(timed_guard_matches) != 2:
        errors.append(
            f"umma_1sm_body contains {len(timed_guard_matches)} exact timed guard(s); "
            "expected exactly two"
        )
    for match in timed_guard_matches:
        try:
            close_brace = find_matching_brace(umma_body, match.end() - 1)
        except SourceStructureError as exc:
            errors.append(f"cannot validate timed guard scope: {exc}")
            continue
        guard_scopes.append(
            (match.end(), close_brace, umma_body[match.end():close_brace])
        )

    for clock_position in clock_positions:
        containing_scopes = [
            (start, end, body)
            for start, end, body in guard_scopes
            if start <= clock_position < end
        ]
        if len(containing_scopes) != 1:
            errors.append(
                "a %clock64 read is outside an exact "
                "if (timing_mode == TimingMode::kTimed) lexical scope"
            )

    if len(guard_scopes) == 2:
        if "start_clock" not in guard_scopes[0][2] or "end_clock" in guard_scopes[0][2]:
            errors.append("the first timed guard must contain only the start_clock read")
        if (
            "end_clock" not in guard_scopes[1][2]
            or "elapsed_cycles" not in guard_scopes[1][2]
        ):
            errors.append(
                "the second timed guard must contain the end_clock read and elapsed-cycle "
                "calculation"
            )

    wait_position = umma_body.find("mbarrier_try_wait_parity")
    if len(clock_positions) == 2 and not (
        clock_positions[0] < wait_position < clock_positions[1]
    ):
        errors.append(
            "clock ordering must be start read, real mbarrier completion wait, then end read"
        )

    try:
        run_once_body, _, _ = extract_single_function_body(
            code_only, RUN_ONCE_DEFINITION, "run_once"
        )
        run_untimed_body, _, _ = extract_single_function_body(
            code_only, RUN_UNTIMED_DEFINITION, "run_untimed_or_die"
        )
        run_timed_body, _, _ = extract_single_function_body(
            code_only, RUN_TIMED_DEFINITION, "run_timed_or_die"
        )
        self_test_body, _, _ = extract_single_function_body(
            code_only, SELF_TEST_DEFINITION, "run_self_test"
        )
        main_body, _, _ = extract_single_function_body(
            code_only, MAIN_DEFINITION, "main"
        )
    except SourceStructureError as exc:
        errors.append(f"cannot validate timing routes: {exc}")
        return errors

    kernel_mode_forward = re.compile(
        r"spec\.kernel\s*<<<.*?>>>\s*\(\s*iterations\s*,\s*mode\s*,\s*"
        r"d_out_device\s*,",
        re.DOTALL,
    )
    if not kernel_mode_forward.search(run_once_body):
        errors.append(
            "run_once must pass its TimingMode mode argument to the selected kernel launch"
        )
    if not re.search(
        r"\bumma_1sm_body\s*<\s*N\s*,\s*DEPTH\s*>\s*"
        r"\(\s*iterations\s*,\s*timing_mode\s*,",
        code_only,
    ):
        errors.append(
            "visible kernel wrappers must forward timing_mode to umma_1sm_body"
        )

    untimed_wrapper_call = re.compile(
        r"\brun_once\s*\(\s*spec\s*,\s*iterations\s*,\s*"
        r"TimingMode::kUntimed\s*\)\s*;"
    )
    if not untimed_wrapper_call.search(run_untimed_body) or "TimingMode::kTimed" in run_untimed_body:
        errors.append(
            "run_untimed_or_die must route exclusively through "
            "run_once(..., TimingMode::kUntimed)"
        )

    timed_wrapper_call = re.compile(
        r"\brun_once\s*\(\s*spec\s*,\s*iterations\s*,\s*"
        r"TimingMode::kTimed\s*\)\s*;"
    )
    if not timed_wrapper_call.search(run_timed_body) or "TimingMode::kUntimed" in run_timed_body:
        errors.append(
            "run_timed_or_die must route exclusively through "
            "run_once(..., TimingMode::kTimed)"
        )

    self_test_untimed_call = re.compile(
        r"\brun_once\s*\(\s*spec\s*,\s*kSelfTestIterations\s*,\s*"
        r"TimingMode::kUntimed\s*\)\s*;"
    )
    if not self_test_untimed_call.search(self_test_body) or "TimingMode::kTimed" in self_test_body:
        errors.append(
            "self-test must call run_once(..., TimingMode::kUntimed) and never use kTimed"
        )

    prevalidation_call = re.compile(
        r"\brun_untimed_or_die\s*\(\s*\*spec\s*,\s*cli\.iterations\s*\)\s*;"
    )
    timed_repetition_call = re.compile(
        r"\brun_timed_or_die\s*\(\s*\*spec\s*,\s*cli\.iterations\s*\)\s*;"
    )
    warmup_loop_pattern = re.compile(
        r"\bfor\s*\(\s*int64_t\s+w\s*=\s*0\s*;\s*"
        r"w\s*<\s*cli\.warmup_iterations\s*;\s*\+\+w\s*\)\s*\{",
        re.DOTALL,
    )
    repetition_loop_pattern = re.compile(
        r"\bfor\s*\(\s*int64_t\s+rep\s*=\s*0\s*;\s*"
        r"rep\s*<\s*cli\.repetitions\s*;\s*\+\+rep\s*\)\s*\{",
        re.DOTALL,
    )

    warmup_matches = list(warmup_loop_pattern.finditer(main_body))
    if len(warmup_matches) != 1:
        errors.append(
            f"main must contain exactly one warm-up loop, found {len(warmup_matches)}"
        )
    else:
        warmup_match = warmup_matches[0]
        try:
            warmup_close = find_matching_brace(main_body, warmup_match.end() - 1)
            warmup_body = main_body[warmup_match.end():warmup_close]
        except SourceStructureError as exc:
            errors.append(f"cannot validate warm-up timing route: {exc}")
        else:
            if not prevalidation_call.search(warmup_body) or "run_timed_or_die" in warmup_body:
                errors.append(
                    "every warm-up launch must route through run_untimed_or_die"
                )
            main_before_warmup = main_body[:warmup_match.start()]
            if not prevalidation_call.search(main_before_warmup):
                errors.append(
                    "pre-timing correctness validation must route through "
                    "run_untimed_or_die before the warm-up loop"
                )

    repetition_matches = list(repetition_loop_pattern.finditer(main_body))
    if len(repetition_matches) != 1:
        errors.append(
            f"main must contain exactly one timed-repetition loop, found "
            f"{len(repetition_matches)}"
        )
    else:
        repetition_match = repetition_matches[0]
        try:
            repetition_close = find_matching_brace(
                main_body, repetition_match.end() - 1
            )
            repetition_body = main_body[repetition_match.end():repetition_close]
        except SourceStructureError as exc:
            errors.append(f"cannot validate timed-repetition route: {exc}")
        else:
            if (
                not timed_repetition_call.search(repetition_body)
                or "run_untimed_or_die" in repetition_body
            ):
                errors.append(
                    "every measured repetition must route through run_timed_or_die"
                )
    return errors


def check_source(source_text: str) -> list[str]:
    """Full source-level contract: comment/literal-aware forbidden/required
    PTX text, the repaired TMEM address construction, the launch-contract
    guard, and timing-mode routing. Fails closed (non-empty list) if the
    lexical scan itself cannot be trusted.
    """
    try:
        code_only = strip_comments_preserving_literals(source_text)
    except SourceScanError as exc:
        return [f"cannot safely scan source: {exc}"]

    errors: list[str] = []
    for pattern, description in FORBIDDEN_SOURCE_PATTERNS:
        if pattern.search(code_only):
            errors.append(f"source contains forbidden pattern: {description}")
    for pattern, description in REQUIRED_SOURCE_PATTERNS:
        if not pattern.search(code_only):
            errors.append(f"source is missing required PTX instruction text: {description}")
    for pattern, description in REQUIRED_TMEM_ADDRESS_PATTERNS:
        if not pattern.search(code_only):
            errors.append(f"source is missing required TMEM address construction: {description}")
    for pattern, description in FORBIDDEN_TMEM_ADDRESS_PATTERNS:
        if pattern.search(code_only):
            errors.append(f"source contains forbidden pattern: {description}")
    errors.extend(check_tmem_address_construction(code_only))
    errors.extend(check_launch_guard_ordering(code_only))
    errors.extend(check_timing_routing(code_only))
    return errors


def validate_source_file(path: Path) -> list[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"cannot read canonical source {path}: {exc}"]
    return check_source(text)


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


# ---------------------------------------------------------------------------
# Synthetic source snippet for the source-level self-test cases below. Built
# from named fields so a single targeted override isolates exactly one
# defect, while every other required property (TMEM helper, launch guard,
# timing routing, required PTX text) stays intact -- mirroring the actual
# structure of src/compute/umma_1sm.cu closely enough to exercise the real
# regexes without needing the full file.
# ---------------------------------------------------------------------------
def golden_source_snippet(**overrides: str) -> str:
    fields = {
        "lane_shift_const": "constexpr uint32_t kTmemLaneShift = 16;",
        "rows_per_warp_const": "constexpr uint32_t kTmemRowsPerWarp = 32;",
        "cols_per_fragment_const": "constexpr uint32_t kTmemColsPerFragment = 32;",
        "lane_contribution": "(static_cast<uint32_t>(warp_id) * kTmemRowsPerWarp) << kTmemLaneShift",
        "column_contribution": "static_cast<uint32_t>(frag) * kTmemColsPerFragment",
        "tmem_return": "return tmem_base + lane_contribution + column_contribution;",
        "tmem_call_site": "tcgen05_ld_32x32b_x32(make_tmem_load_address(tmem_d, warp_id, frag), regs);",
        "launch_predicate": (
            "return gridDim.x == kExpectedGridDim && gridDim.y == kExpectedGridDim && "
            "gridDim.z == kExpectedGridDim && blockDim.x == kExpectedBlockDimX && "
            "blockDim.y == 1 && blockDim.z == 1;"
        ),
        "launch_guard": "if (!launch_contract_is_valid()) { g_launch_ok[0] = 0; return; }",
        "timing_guard_a": (
            'if (timing_mode == TimingMode::kTimed) { '
            'asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock)); }'
        ),
        "timing_guard_b": (
            'if (timing_mode == TimingMode::kTimed) { '
            'asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock)); '
            "elapsed_cycles = end_clock - start_clock; }"
        ),
        "kernel_mode_forward": (
            "umma_1sm_body<N, DEPTH>(iterations, timing_mode, g_d_out, "
            "g_elapsed_cycles, g_launch_ok);"
        ),
        "run_once_mode_forward": (
            "spec.kernel<<<1, 128>>>(iterations, mode, d_out_device, "
            "cycles_device, launch_ok_device);"
        ),
        "untimed_call_a": "run_once(spec, iterations, TimingMode::kUntimed);",
        "untimed_call_b": "run_once(spec, kSelfTestIterations, TimingMode::kUntimed);",
        "timed_call": "run_once(spec, iterations, TimingMode::kTimed);",
        "prevalidation_call": "run_untimed_or_die(*spec, cli.iterations);",
        "warmup_call": "run_untimed_or_die(*spec, cli.iterations);",
        "timed_repetition_call": "run_timed_or_die(*spec, cli.iterations);",
        "wait_ld_text": "tcgen05.wait::ld.sync.aligned;",
        "fence_text": "tcgen05.fence::after_thread_sync;",
        "extra_forbidden_line": "",
    }
    fields.update(overrides)
    return (
        f"{fields['lane_shift_const']}\n"
        f"{fields['rows_per_warp_const']}\n"
        f"{fields['cols_per_fragment_const']}\n"
        "__device__ uint32_t make_tmem_load_address(uint32_t tmem_base, int warp_id, int frag) {\n"
        f"    const uint32_t lane_contribution = {fields['lane_contribution']};\n"
        f"    const uint32_t column_contribution = {fields['column_contribution']};\n"
        f"    {fields['tmem_return']}\n"
        "}\n"
        "__device__ bool launch_contract_is_valid() {\n"
        f"    {fields['launch_predicate']}\n"
        "}\n"
        "__device__ void umma_1sm_body(int64_t iterations, TimingMode timing_mode) {\n"
        f"    {fields['launch_guard']}\n"
        "    g_launch_ok[0] = 1;\n"
        "    __syncthreads();\n"
        f"    {fields['timing_guard_a']}\n"
        "    while (!mbarrier_try_wait_parity()) {}\n"
        f"    {fields['timing_guard_b']}\n"
        f"    {fields['tmem_call_site']}\n"
        "}\n"
        "template <int N, int DEPTH>\n"
        "void visible_kernel(int64_t iterations, TimingMode timing_mode) {\n"
        f"    {fields['kernel_mode_forward']}\n"
        "}\n"
        "RunResult run_once(const Specialization& spec, int64_t iterations, TimingMode mode) {\n"
        f"    {fields['run_once_mode_forward']}\n"
        "}\n"
        "void run_untimed_or_die(const Specialization& spec, int64_t iterations) {\n"
        f"    {fields['untimed_call_a']}\n"
        "}\n"
        "unsigned long long run_timed_or_die(const Specialization& spec, int64_t iterations) {\n"
        f"    {fields['timed_call']}\n"
        "}\n"
        "int run_self_test() {\n"
        f"    {fields['untimed_call_b']}\n"
        "}\n"
        "int main(int argc, char** argv) {\n"
        f"    {fields['prevalidation_call']}\n"
        "    for (int64_t w = 0; w < cli.warmup_iterations; ++w) {\n"
        f"        {fields['warmup_call']}\n"
        "    }\n"
        "    for (int64_t rep = 0; rep < cli.repetitions; ++rep) {\n"
        f"        {fields['timed_repetition_call']}\n"
        "    }\n"
        "}\n"
        f"{fields['wait_ld_text']}\n"
        f"{fields['fence_text']}\n"
        f"{fields['extra_forbidden_line']}\n"
    )


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
            "source check accepts a fully valid source (TMEM helper, launch guard, timing "
            "routing, required PTX text, no forbidden pattern)",
            golden_source_snippet(),
            None,
        ),
        (
            "source check rejects cta_group::2",
            golden_source_snippet(extra_forbidden_line="tcgen05.mma.cta_group::2.kind::f16 [x], a, b, i, p;"),
            "cta_group::2",
        ),
        (
            "source check rejects __cluster_dims__",
            golden_source_snippet(extra_forbidden_line="__global__ __cluster_dims__(2,1,1) void k() {}"),
            "__cluster_dims__",
        ),
        (
            "source check rejects a missing tcgen05.wait::ld",
            golden_source_snippet(wait_ld_text=""),
            "tcgen05.wait::ld.sync.aligned",
        ),
        (
            "source check rejects a missing tcgen05.fence::after_thread_sync",
            golden_source_snippet(fence_text=""),
            "tcgen05.fence::after_thread_sync",
        ),
        (
            "source check rejects required PTX text present only in a // comment",
            golden_source_snippet(wait_ld_text="// tcgen05.wait::ld.sync.aligned"),
            "tcgen05.wait::ld.sync.aligned",
        ),
        (
            "source check rejects required PTX text present only in a /* */ comment",
            golden_source_snippet(fence_text="/* tcgen05.fence::after_thread_sync */"),
            "tcgen05.fence::after_thread_sync",
        ),
        (
            "source check accepts forbidden text present only inside a /* */ comment",
            golden_source_snippet(
                extra_forbidden_line="/* cta_group::2 __cluster_dims__ multicast block_scale */"
            ),
            None,
        ),
        (
            "source check rejects a missing warp-derived TMEM lane offset",
            golden_source_snippet(lane_contribution="kTmemRowsPerWarp << kTmemLaneShift"),
            "warp contribution using warp_id * kTmemRowsPerWarp",
        ),
        (
            "source check rejects an incorrect TMEM lane shift constant",
            golden_source_snippet(lane_shift_const="constexpr uint32_t kTmemLaneShift = 15;"),
            "kTmemLaneShift defined as 16",
        ),
        (
            "source check rejects a missing TMEM fragment column offset",
            golden_source_snippet(column_contribution="static_cast<uint32_t>(frag)"),
            "fragment contribution using frag * kTmemColsPerFragment",
        ),
        (
            "source check rejects a live TMEM return that omits lane_contribution",
            golden_source_snippet(
                tmem_return="return tmem_base + column_contribution;"
            ),
            "return must combine tmem_base + lane_contribution + column_contribution",
        ),
        (
            "source check rejects a live TMEM return that omits column_contribution",
            golden_source_snippet(
                tmem_return="return tmem_base + lane_contribution;"
            ),
            "return must combine tmem_base + lane_contribution + column_contribution",
        ),
        (
            "source check rejects the original defective tmem_d + frag * 32 read operand",
            golden_source_snippet(tmem_call_site="tcgen05_ld_32x32b_x32(tmem_d + frag * 32, regs);"),
            "tmem_d + frag * 32",
        ),
        (
            "source check rejects a missing launch-contract guard",
            golden_source_snippet(launch_guard=""),
            "missing launch-contract guard",
        ),
        (
            "source check rejects an inverted launch-contract guard",
            golden_source_snippet(
                launch_guard=(
                    "if (launch_contract_is_valid()) "
                    "{ g_launch_ok[0] = 0; return; }"
                )
            ),
            "if (!launch_contract_is_valid())",
        ),
        (
            "source check rejects missing timed clock guards and reads",
            golden_source_snippet(timing_guard_a="", timing_guard_b=""),
            "%clock64 read(s)",
        ),
        (
            "source check rejects unconditional clock reads hidden beside empty timed guards",
            golden_source_snippet(
                timing_guard_a=(
                    "if (timing_mode == TimingMode::kTimed) {} "
                    'asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock));'
                ),
                timing_guard_b=(
                    "if (timing_mode == TimingMode::kTimed) {} "
                    'asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock)); '
                    "elapsed_cycles = end_clock - start_clock;"
                ),
            ),
            "outside an exact if (timing_mode == TimingMode::kTimed) lexical scope",
        ),
        (
            "source check rejects a self-test routed through TimingMode::kTimed",
            golden_source_snippet(
                untimed_call_b=(
                    "run_once(spec, kSelfTestIterations, TimingMode::kTimed);"
                )
            ),
            "self-test must call run_once(..., TimingMode::kUntimed)",
        ),
        (
            "source check rejects a missing TimingMode::kUntimed call site",
            golden_source_snippet(untimed_call_a="", untimed_call_b=""),
            "TimingMode::kUntimed",
        ),
        (
            "source check fails closed on an unterminated /* block comment",
            "/* this block comment never closes\nint x = 1;\n",
            "cannot safely scan",
        ),
        (
            "source check fails closed on an unterminated string literal",
            'const char* s = "this string never closes;\n',
            "cannot safely scan",
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

    mandatory_validation_cases: list[tuple[str, bool]] = [
        (
            "mandatory source validation: default path resolves to src/compute/umma_1sm.cu",
            resolve_default_source_path().as_posix().endswith("src/compute/umma_1sm.cu"),
        ),
        (
            "mandatory source validation: a missing canonical source fails closed (non-empty errors)",
            bool(validate_source_file(Path("/nonexistent-path-should-never-exist/umma_1sm.cu"))),
        ),
    ]
    for name, ok in mandatory_validation_cases:
        if ok:
            print(f"check_umma_1sm_sass: self-test: PASS: {name}", file=sys.stderr)
        else:
            failures.append(name)
            print(f"check_umma_1sm_sass: self-test: FAIL: {name}", file=sys.stderr)

    total = len(cases) + len(source_cases) + len(mandatory_validation_cases)
    if failures:
        print(f"check_umma_1sm_sass: self-test: FAILED ({len(failures)}/{total} case(s))", file=sys.stderr)
        return 1
    print(f"check_umma_1sm_sass: self-test: OK ({total} cases)", file=sys.stderr)
    return 0


def check_binary(binary_path: str, out_path: str, explicit_source_path: str | None) -> int:
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

    # Source validation is mandatory: --source may override which file is
    # checked, but omitting it resolves the canonical repository path below
    # instead of skipping the check. There is no bypass.
    source_path = Path(explicit_source_path) if explicit_source_path is not None else resolve_default_source_path()
    source_errors = validate_source_file(source_path)
    if source_errors:
        errors.extend(f"[source {source_path}] {detail}" for detail in source_errors)
    else:
        print(
            f"check_umma_1sm_sass: source check OK ({source_path}): tcgen05.wait::ld and "
            "tcgen05.fence::after_thread_sync are present, the TMEM address helper/launch-contract "
            "guard/timing-mode routing are real executable code, and no forbidden pattern was found "
            "(comment- and string-literal-aware scan)",
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
        "lifecycle, correct per-warp TMEM addressing, and no forbidden or 2-SM instruction",
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
        "usage: check_umma_1sm_sass.py <binary> <output-sass-path> [--source <path>]\n"
        "       check_umma_1sm_sass.py --self-test\n"
        "the two-positional-argument form always validates the canonical source\n"
        "(src/compute/umma_1sm.cu, resolved relative to this script) even when\n"
        "--source is omitted; --source only overrides which file is checked.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
