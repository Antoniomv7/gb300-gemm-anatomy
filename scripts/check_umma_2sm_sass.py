#!/usr/bin/env python3
"""GPU-free SASS and source verification for the P2.2 2-SM BF16 UMMA microbenchmark.

Disassemble the compiled binary with ``cuobjdump -sass`` (PTX is not accepted
as proof), identify all twelve ``umma_2sm_m256n{N}k16_d{DEPTH}`` symbols, and
verify for each one:

* exactly the twelve expected (N, depth) specializations exist, with no
  missing, extra, or duplicate configuration;
* the symbol contains ``UTCHMMA.2CTA`` -- sm_103a's SASS lowering of
  ``tcgen05.mma.cta_group::2.kind::f16``, observed directly (see below);
* the static ``UTCHMMA.2CTA`` count is exactly ``depth`` and the address
  spacing between consecutive occurrences is uniform, evidencing full
  compile-time unrolling rather than a runtime back-edge standing in for it;
* the burst ends with a genuine completion sequence: ``UTCBAR.2CTA.
  MULTICAST`` (the multicast commit) after the last ``UTCHMMA.2CTA``,
  followed by at least one ``SYNCS.PHASECHK.TRANS*.TRYWAIT`` (mbarrier
  completion wait);
* collective TMEM allocation (``UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN``) and
  deallocation (``UVIRTCOUNT.DEALLOC.SMPOOL``) are present, with a
  cluster-barrier pair (``UCGABAR_ARV``/``UCGABAR_WAIT``) ordered strictly
  between the last TMEM use and the deallocation;
* TMEM-to-register loading (``LDTM.x32``) is present, with exactly N/32
  occurrences (one per 32-column fragment);
* cluster-rank evidence (``SR_CgaCtaId``) is present;
* the compiled ELF's ``.nv.info.<symbol>`` section for every specialization
  carries both ``EIATTR_EXPLICIT_CLUSTER`` and an ``EIATTR_CTA_PER_CLUSTER``
  value of exactly ``0x2 0x1 0x1`` (``cuobjdump -elf``'s direct, per-kernel,
  binary-level record of the compile-time two-CTA cluster declaration);
* no bare (non-``.2CTA``) ``UTCHMMA``/``UTCBAR``/``UTCATOMSWS.FIND_AND_SET.
  ALIGN`` (1-SM fallback evidence), and no WGMMA, mma.sync/HMMA, TMA,
  LDGSTS, FP8/FP4, or sparse-MMA instruction, is present anywhere in the
  binary.

Mnemonic provenance (read directly from ``cuobjdump -sass``/``-elf`` output
of this project's own ``build/compute/umma_2sm`` binary, real cta_group::2
kernels compiled for sm_103a with CUDA 13.1.80 ptxas -- not guessed from
documentation or the PTX ISA text, and cross-checked against isolated
single-instruction probes compiled the same way; see
src/compute/P2_2_PROTOCOL.md section 15 for the full evidence table):

  PTX (source)                                        SASS (sm_103a, this binary)
  tcgen05.mma.cta_group::2.kind::f16                  UTCHMMA.2CTA
  tcgen05.commit.cta_group::2.mbarrier::arrive::       UTCBAR.2CTA.MULTICAST
    one.shared::cluster.multicast::cluster.b64
  tcgen05.alloc.cta_group::2...                       UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN (x2: ptxas
                                                       peels a fast-path check plus a retry-loop body,
                                                       the same duplication pattern already documented
                                                       for P2.1's cta_group::1 alloc -- presence, not an
                                                       exact count, is required)
  tcgen05.dealloc.cta_group::2...                     UVIRTCOUNT.DEALLOC.SMPOOL (no distinct .2CTA
                                                       marker observed; same mnemonic as P2.1's
                                                       cta_group::1 form)
  tcgen05.relinquish_alloc_permit.cta_group::2...      (folded into UTCATOMSWS.AND; not checked, same
                                                       as P2.1's identical-mnemonic cta_group::1 form)
  mbarrier.try_wait.parity                            SYNCS.PHASECHK.TRANS64.TRYWAIT
  mbarrier.inval.shared.b64                            SYNCS.CCTL.IV
  tcgen05.ld.sync.aligned.32x32b.x32.b32               LDTM.x32
  tcgen05.wait::ld.sync.aligned                        (no distinct SASS instruction -- see below)
  tcgen05.fence::after_thread_sync                     (no distinct SASS instruction -- see below)
  %cluster_ctarank (cuda::ptx::get_sreg_cluster_ctarank) S2R/S2UR ..., SR_CgaCtaId
  %cluster_nctarank (cuda::ptx::get_sreg_cluster_nctarank) CS2R.32 ..., SR_CgaSize (combined with a
                                                       constant-bank lookup; not pattern-matched here,
                                                       SR_CgaCtaId alone is sufficient cluster-rank
                                                       evidence)
  barrier.cluster.arrive / barrier.cluster.wait        UCGABAR_ARV / UCGABAR_WAIT (plus a supporting
                                                       MEMBAR.ALL.CTA/MEMBAR.ALL.GPU/ERRBAR/CGAERRBAR/
                                                       CCTL.IVALL sequence; UCGABAR_ARV/WAIT alone are
                                                       used as the required evidence)
  __cluster_dims__(2, 1, 1)                            ELF attributes EIATTR_EXPLICIT_CLUSTER and
                                                       EIATTR_CTA_PER_CLUSTER (value 0x2 0x1 0x1) in
                                                       cuobjdump -elf's per-kernel .nv.info section

``tcgen05.wait::ld`` and ``tcgen05.fence::after_thread_sync`` were confirmed
present in the compiled PTX (one ``tcgen05.wait::ld`` per ``LDTM.x32``
fragment and one ``tcgen05.fence::after_thread_sync`` per kernel, mirroring
P2.1) but ptxas emits no separate SASS instruction for either on this pinned
toolchain, for the same reasons already documented for P2.1 (register
scoreboarding already serializes the load; the fence is a pure code-motion
constraint). This checker therefore proves both instructions' presence with
a mandatory static source check instead of inventing a SASS signal that does
not exist.

Source validation is mandatory, not optional: the two-positional-argument
invocation (``<binary> <output-sass-path>``) always validates the canonical
source ``src/compute/umma_2sm.cu``, resolved relative to this script (never
the caller's current working directory). ``--source <path>`` may override
which file is checked (used for ad hoc testing); omitting it never skips the
check. If the resolved source cannot be opened, this checker exits 1 --
there is no code path in which the real binary/SASS check can report success
while source validation was skipped or merely reported as a documented
limitation.

The source scanner (shared algorithm with, but an independent implementation
from, P2.1's checker -- see AGENTS.md/task instructions: umma_2sm.cu and this
checker must remain an independently auditable unit, not a refactor of
P2.1's files) strips both ``//`` line comments and ``/* ... */`` block
comments while preserving the exact text of every string and character
literal (required inline-PTX text lives inside C++ string literals passed to
inline asm). Every forbidden- and required-pattern check in this checker
runs against this comment-stripped, literal-preserving view, never against
the raw source text. The scanner fails closed on an unterminated ``/*``
block comment or an unterminated string/character literal.

Usage:
  check_umma_2sm_sass.py --self-test

  check_umma_2sm_sass.py <binary> <output-sass-path> [--source <umma_2sm.cu>]

Exit code: 0 only when the selected validation passes, 1 on a contract,
synthetic-test, I/O, source-scan, or ``cuobjdump``/source-check failure, and
2 on a usage error.
"""

import re
import subprocess
import sys
from pathlib import Path


FUNCTION_MARKER = "umma_2sm_m256n"
EXPECTED_NS = (64, 128, 256)
EXPECTED_DEPTHS = (4, 16, 64, 256)
EXPECTED_SPECS = {(n, d) for n in EXPECTED_NS for d in EXPECTED_DEPTHS}

SYMBOL_PATTERN = re.compile(r"\bumma_2sm_m256n(\d+)k16_d(\d+)\b")

UTCHMMA_2CTA_PATTERN = re.compile(r"\bUTCHMMA\.2CTA\b")
UTCHMMA_NON_2CTA_PATTERN = re.compile(r"\bUTCHMMA\b(?!\.2CTA\b)")
UTCBAR_MULTICAST_PATTERN = re.compile(r"\bUTCBAR\.2CTA\.MULTICAST\b")
UTCBAR_NON_MULTICAST_PATTERN = re.compile(r"\bUTCBAR\b(?!\.2CTA\.MULTICAST\b)")
TRYWAIT_PATTERN = re.compile(r"\bSYNCS\.PHASECHK\.TRANS\d*\.TRYWAIT\b")
ALLOC_2CTA_PATTERN = re.compile(r"\bUTCATOMSWS\.2CTA\.FIND_AND_SET\.ALIGN\b")
ALLOC_NON_2CTA_PATTERN = re.compile(r"\bUTCATOMSWS\.FIND_AND_SET\.ALIGN\b")
DEALLOC_PATTERN = re.compile(r"\bUVIRTCOUNT\.DEALLOC\.SMPOOL\b")
INVALIDATE_PATTERN = re.compile(r"\bSYNCS\.CCTL\.IV\b")
LDTM_PATTERN = re.compile(r"\bLDTM\.x32\b")
CGABAR_PATTERN = re.compile(r"\bUCGABAR_ARV\b|\bUCGABAR_WAIT\b")
CLUSTER_RANK_PATTERN = re.compile(r"\bSR_CgaCtaId\b")

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
    (UTCHMMA_NON_2CTA_PATTERN, "a non-.2CTA (1-SM fallback) UTCHMMA"),
    (UTCBAR_NON_MULTICAST_PATTERN, "a non-.2CTA.MULTICAST (1-SM/non-multicast fallback) UTCBAR"),
    (ALLOC_NON_2CTA_PATTERN, "a non-.2CTA (1-SM fallback) TMEM allocation"),
)

FORBIDDEN_SOURCE_PATTERNS = (
    (re.compile(r"cta_group::1\b"), "cta_group::1"),
    (re.compile(r"multicast::cluster", re.IGNORECASE), None),  # handled positively below; not forbidden
)
# The above placeholder is intentionally unused as a forbidden entry (P2.2
# REQUIRES multicast::cluster); keep FORBIDDEN_SOURCE_PATTERNS limited to
# genuinely forbidden text.
FORBIDDEN_SOURCE_PATTERNS = (
    (re.compile(r"cta_group::1\b"), "cta_group::1"),
    (re.compile(r"\.kind::(?!f16\b)[a-z0-9_]+"), "a non-kind::f16 MMA kind"),
    (re.compile(r"\.sp\b"), "a sparse (.sp) MMA form"),
    (re.compile(r"block_scale"), "block_scale"),
)
REQUIRED_SOURCE_PATTERNS = (
    (re.compile(r"tcgen05\.wait::ld\.sync\.aligned"), "tcgen05.wait::ld.sync.aligned"),
    (re.compile(r"tcgen05\.fence::after_thread_sync"), "tcgen05.fence::after_thread_sync"),
    (re.compile(r"tcgen05\.mma\.cta_group::2\.kind::f16"), "tcgen05.mma.cta_group::2.kind::f16"),
    (re.compile(r"tcgen05\.commit\.cta_group::2\.mbarrier::arrive::one\.shared::cluster\."
                r"multicast::cluster\.b64"),
     "tcgen05.commit.cta_group::2...shared::cluster.multicast::cluster.b64"),
    (re.compile(r"tcgen05\.alloc\.cta_group::2\.sync\.aligned\.shared::cta\.b32"),
     "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32"),
    (re.compile(r"tcgen05\.dealloc\.cta_group::2\.sync\.aligned\.b32"),
     "tcgen05.dealloc.cta_group::2.sync.aligned.b32"),
    (re.compile(r"tcgen05\.relinquish_alloc_permit\.cta_group::2\.sync\.aligned"),
     "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned"),
    (re.compile(r"__cluster_dims__\s*\(\s*2\s*,\s*1\s*,\s*1\s*\)"), "__cluster_dims__(2, 1, 1)"),
    (re.compile(r"get_sreg_cluster_ctarank"), "cuda::ptx::get_sreg_cluster_ctarank"),
    (re.compile(r"get_sreg_cluster_nctarank"), "cuda::ptx::get_sreg_cluster_nctarank"),
    (re.compile(r"barrier_cluster_arrive"), "cuda::ptx::barrier_cluster_arrive"),
    (re.compile(r"barrier_cluster_wait"), "cuda::ptx::barrier_cluster_wait"),
    (re.compile(r"0x0003u?\b"), "the exact multicast CTA mask 0x0003"),
)

DEFAULT_SOURCE_RELATIVE_PARTS = ("src", "compute", "umma_2sm.cu")


def resolve_default_source_path() -> Path:
    """The canonical P2.2 source, resolved relative to this checker script
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

    Independently implemented (same generic algorithm, fresh code) for this
    checker; required because inline PTX text lives inside C++ string
    literals passed to inline asm, and because a comment must never be able
    to satisfy a required-pattern check or accidentally trip a forbidden-
    pattern check. Raises SourceScanError on an unterminated block comment or
    string/char literal, since the lexical state cannot then be safely
    determined.
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


UMMA_BODY_DEFINITION = re.compile(
    r"\bvoid\s+umma_2sm_body\s*\([^;{}]*\)\s*\{", re.DOTALL
)
LAUNCH_PREDICATE_DEFINITION = re.compile(
    r"\bbool\s+launch_contract_is_valid\s*\([^;{}]*\)\s*\{", re.DOTALL
)
TMEM_HELPER_DEFINITION = re.compile(
    r"\buint32_t\s+make_tmem_load_address\s*\([^;{}]*\)\s*\{", re.DOTALL
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


def check_launch_guard_ordering(code_only: str) -> list[str]:
    """(task section 12: "Uniform launch guard") Prove the predicate depends
    only on cluster-uniform values plus the ISA-guaranteed ctarank<nctarank
    range fact, and that umma_2sm_body rejects before its first
    __syncthreads(), writing a per-rank status and returning.
    """
    errors: list[str] = []
    try:
        predicate_body, _, _ = extract_single_function_body(
            code_only, LAUNCH_PREDICATE_DEFINITION, "launch_contract_is_valid"
        )
    except SourceStructureError as exc:
        errors.append(f"invalid launch-contract predicate: {exc}")
        return errors

    if "cluster_nctarank" not in predicate_body or "cluster_ctarank" not in predicate_body:
        errors.append(
            "launch_contract_is_valid must depend on both cluster_nctarank and cluster_ctarank"
        )
    if "gridDim" not in predicate_body or "blockDim" not in predicate_body:
        errors.append("launch_contract_is_valid must check both gridDim and blockDim")

    try:
        umma_body, _, _ = extract_single_function_body(
            code_only, UMMA_BODY_DEFINITION, "umma_2sm_body"
        )
    except SourceStructureError as exc:
        errors.append(f"invalid launch-contract guard: {exc}")
        return errors

    first_sync = umma_body.find("__syncthreads()")
    if first_sync == -1:
        errors.append(
            "missing launch-contract ordering anchor: umma_2sm_body has no __syncthreads()"
        )
        return errors
    prefix = umma_body[:first_sync]
    negative_guard = re.compile(
        r"\bif\s*\(\s*!\s*launch_contract_is_valid\s*\([^)]*\)\s*\)\s*\{",
        re.DOTALL,
    )
    try:
        rejected_body, _, rejected_end = extract_single_control_block(
            prefix, negative_guard, "negative launch-contract rejection"
        )
    except SourceStructureError as exc:
        errors.append(
            "missing launch-contract guard: expected "
            "if (!launch_contract_is_valid(...)) before the first __syncthreads(): "
            f"{exc}"
        )
        return errors

    if not re.search(r"\bg_launch_ok\s*\[\s*0\s*\]\s*=\s*0\s*;", rejected_body):
        errors.append("launch-contract rejection must write g_launch_ok[0] = 0 for rank 0")
    if not re.search(r"\bg_launch_ok\s*\[\s*1\s*\]\s*=\s*0\s*;", rejected_body):
        errors.append("launch-contract rejection must write g_launch_ok[1] = 0 for rank 1")
    if not re.search(r"\breturn\s*;", rejected_body):
        errors.append("launch-contract rejection must return before synchronization")

    accepted_prefix = prefix[rejected_end + 1:]
    if not re.search(r"\bg_launch_ok\s*\[\s*0\s*\]\s*=\s*1\s*;", accepted_prefix):
        errors.append("accepted launch path must write g_launch_ok[0] = 1 for rank 0")
    if not re.search(r"\bg_launch_ok\s*\[\s*1\s*\]\s*=\s*1\s*;", accepted_prefix):
        errors.append("accepted launch path must write g_launch_ok[1] = 1 for rank 1")
    return errors


def check_rank_mapping(code_only: str) -> list[str]:
    """(task section 12: "Per-rank A and D mapping", "Identical B copies",
    "Correct local TMEM versus global output addressing") Prove that A's
    fill value and D's global write index both use
    cta_rank * kMLocal + local_row, that B's fill loop does not reference
    cta_rank at all, and that the TMEM load address helper is never given a
    rank-based offset.
    """
    errors: list[str] = []
    try:
        umma_body, _, _ = extract_single_function_body(
            code_only, UMMA_BODY_DEFINITION, "umma_2sm_body"
        )
    except SourceStructureError as exc:
        return [f"invalid rank-mapping check: {exc}"]

    global_row_expr = re.compile(r"\bglobal_row\s*=\s*cta_rank\s*\*\s*kMLocal\s*\+\s*local_row\s*;")
    global_row_matches = list(global_row_expr.finditer(umma_body))
    if len(global_row_matches) < 2:
        errors.append(
            "expected at least two uses of 'global_row = cta_rank * kMLocal + local_row' "
            f"(A initialization and D readback), found {len(global_row_matches)}"
        )

    a_loop_header = re.compile(
        r"\bfor\s*\(\s*int\s+idx\s*=\s*tid\s*;\s*idx\s*<\s*kMLocal\s*\*\s*kK\s*;\s*"
        r"idx\s*\+=\s*kThreadsPerCta\s*\)\s*\{",
        re.DOTALL,
    )
    try:
        a_loop_body, _, _ = extract_single_control_block(umma_body, a_loop_header, "A initialization loop")
    except SourceStructureError as exc:
        errors.append(f"cannot validate A initialization loop: {exc}")
    else:
        if "cta_rank" not in a_loop_body:
            errors.append("A initialization must depend on cta_rank (via global_row)")

    b_loop_header = re.compile(
        r"\bfor\s*\(\s*int\s+idx\s*=\s*tid\s*;\s*idx\s*<\s*N\s*\*\s*kK\s*;\s*"
        r"idx\s*\+=\s*kThreadsPerCta\s*\)\s*\{",
        re.DOTALL,
    )
    try:
        b_loop_body, _, _ = extract_single_control_block(umma_body, b_loop_header, "B initialization loop")
    except SourceStructureError as exc:
        errors.append(f"cannot validate B initialization loop: {exc}")
    else:
        if "cta_rank" in b_loop_body:
            errors.append("B initialization must NOT depend on cta_rank (identical in both CTAs)")

    forbidden_tmem_rank_offset = re.compile(r"tcgen05_ld_32x32b_x32\s*\([^;]*\bcta_rank\b", re.DOTALL)
    if forbidden_tmem_rank_offset.search(umma_body):
        errors.append(
            "TMEM load address must not be offset by cta_rank; the rank offset belongs only in "
            "the global output index"
        )
    if not re.search(r"make_tmem_load_address\s*\(\s*tmem_d\s*,\s*warp_id\s*,\s*frag\s*\)", umma_body):
        errors.append(
            "TMEM readback must call make_tmem_load_address(tmem_d, warp_id, frag) with no rank offset"
        )
    if not re.search(
        r"g_d_out\s*\[\s*static_cast<int64_t>\s*\(\s*global_row\s*\)\s*\*\s*N\s*\+\s*frag\s*\*\s*32\s*\+\s*i\s*\]",
        umma_body,
    ):
        errors.append("D readback must write g_d_out indexed by the GLOBAL row, not the local row")
    return errors


def check_collective_tmem_lifecycle(code_only: str) -> list[str]:
    """(task section 12: "Collective TMEM allocation/deallocation/
    relinquishing") Prove alloc/dealloc/relinquish are gated only by
    warp_id == 0 (never by cta_rank or a single elected lane).
    """
    errors: list[str] = []
    try:
        umma_body, _, _ = extract_single_function_body(
            code_only, UMMA_BODY_DEFINITION, "umma_2sm_body"
        )
    except SourceStructureError as exc:
        return [f"invalid TMEM lifecycle check: {exc}"]

    warp0_header = re.compile(r"\bif\s*\(\s*warp_id\s*==\s*0\s*\)\s*\{", re.DOTALL)
    warp0_blocks = []
    for match in warp0_header.finditer(umma_body):
        try:
            close = find_matching_brace(umma_body, match.end() - 1)
            warp0_blocks.append(umma_body[match.end():close])
        except SourceStructureError:
            continue
    if not any("tcgen05_alloc_2sm" in b for b in warp0_blocks):
        errors.append("tcgen05_alloc_2sm must be issued from an if (warp_id == 0) block")
    if not any("tcgen05_dealloc_2sm" in b and "tcgen05_relinquish_alloc_permit_2sm" in b for b in warp0_blocks):
        errors.append(
            "tcgen05_dealloc_2sm and tcgen05_relinquish_alloc_permit_2sm must be issued together "
            "from an if (warp_id == 0) block"
        )

    for forbidden_scope_pat, label in (
        (re.compile(r"\bif\s*\(\s*cta_rank\s*==\s*0\s*\)\s*\{", re.DOTALL), "cta_rank == 0"),
        (re.compile(r"\bif\s*\(\s*is_leader\s*\)\s*\{", re.DOTALL), "is_leader"),
    ):
        for match in forbidden_scope_pat.finditer(umma_body):
            try:
                close = find_matching_brace(umma_body, match.end() - 1)
            except SourceStructureError:
                continue
            scoped = umma_body[match.end():close]
            for fn in ("tcgen05_alloc_2sm", "tcgen05_dealloc_2sm", "tcgen05_relinquish_alloc_permit_2sm"):
                if fn in scoped:
                    errors.append(
                        f"{fn} must not be issued from inside an '{label}' conditional "
                        "(allocation/deallocation/relinquishing are warp-collective, not "
                        "single-lane or rank-0-only)"
                    )
    return errors


def check_rank0_only_issue(code_only: str) -> list[str]:
    """(task section 12: "Rank-0-only MMA and commit issue", "Exact
    multicast mask 0x0003", "Wait and readback in both CTAs") Prove
    issue_one_umma_2sm/commit_umma_2sm_multicast are lexically confined to an
    'if (cta_rank == 0)' block nested inside 'if (is_leader)', that the
    multicast mask is the literal 0x0003u, and that the mbarrier wait and
    the TMEM readback loop are NOT confined to any cta_rank == 0 block.
    """
    errors: list[str] = []
    try:
        umma_body, _, _ = extract_single_function_body(
            code_only, UMMA_BODY_DEFINITION, "umma_2sm_body"
        )
    except SourceStructureError as exc:
        return [f"invalid rank-0-issue check: {exc}"]

    leader_header = re.compile(r"\bif\s*\(\s*is_leader\s*\)\s*\{", re.DOTALL)
    try:
        leader_body, _, _ = extract_single_control_block(umma_body, leader_header, "leader timed-region")
    except SourceStructureError as exc:
        errors.append(f"cannot locate the leader-only timed region: {exc}")
        return errors

    rank0_header = re.compile(r"\bif\s*\(\s*cta_rank\s*==\s*0\s*\)\s*\{", re.DOTALL)
    rank0_matches = list(rank0_header.finditer(leader_body))
    issue_in_rank0 = False
    mask_ok = False
    for match in rank0_matches:
        try:
            close = find_matching_brace(leader_body, match.end() - 1)
        except SourceStructureError:
            continue
        scoped = leader_body[match.end():close]
        if "issue_one_umma_2sm" in scoped and "commit_umma_2sm_multicast" in scoped:
            issue_in_rank0 = True
            if re.search(r"commit_umma_2sm_multicast\s*\(\s*mbar_addr\s*,\s*/\*[^*]*\*/\s*0x0003u\s*\)", scoped) or \
               re.search(r"commit_umma_2sm_multicast\s*\([^;]*0x0003u?\b", scoped, re.DOTALL):
                mask_ok = True
    if not issue_in_rank0:
        errors.append(
            "issue_one_umma_2sm and commit_umma_2sm_multicast must both be issued from a single "
            "'if (cta_rank == 0)' block nested inside the leader-only region"
        )
    if not mask_ok:
        errors.append("commit_umma_2sm_multicast must be called with the exact literal CTA mask 0x0003")

    # The wait must be present in the leader body but OUTSIDE every
    # 'if (cta_rank == 0)' block (i.e. at the leader_body's own top level).
    stripped_of_rank0_blocks = leader_body
    for match in reversed(rank0_matches):
        try:
            close = find_matching_brace(leader_body, match.end() - 1)
        except SourceStructureError:
            continue
        stripped_of_rank0_blocks = (
            stripped_of_rank0_blocks[:match.start()] + stripped_of_rank0_blocks[close + 1:]
        )
    if "mbarrier_try_wait_parity" not in stripped_of_rank0_blocks:
        errors.append(
            "the mbarrier completion wait must not be enclosed in a cta_rank == 0 condition"
        )

    # The TMEM readback loop (after the leader block) must not be re-guarded
    # by cta_rank == 0 anywhere between the leader block and the end of the
    # function.
    after_leader = umma_body[umma_body.find(leader_body) + len(leader_body):]
    readback_rank0_guard = re.compile(
        r"\bif\s*\(\s*cta_rank\s*==\s*0\s*\)\s*\{[^{}]*tcgen05_ld_32x32b_x32", re.DOTALL
    )
    if readback_rank0_guard.search(after_leader):
        errors.append("the TMEM readback loop must not be enclosed in a cta_rank == 0 condition")
    if "tcgen05_ld_32x32b_x32" not in after_leader:
        errors.append("the TMEM readback loop must be present after the timed region")
    return errors


def check_cluster_sync_before_dealloc(code_only: str) -> list[str]:
    """(task section 12: "Cluster synchronization before deallocation")
    Prove a barrier_cluster_arrive()/barrier_cluster_wait() pair appears,
    textually, after the TMEM readback and before tcgen05_dealloc_2sm.
    """
    try:
        umma_body, _, _ = extract_single_function_body(
            code_only, UMMA_BODY_DEFINITION, "umma_2sm_body"
        )
    except SourceStructureError as exc:
        return [f"invalid cluster-sync-before-dealloc check: {exc}"]

    dealloc_pos = umma_body.find("tcgen05_dealloc_2sm")
    if dealloc_pos == -1:
        return ["tcgen05_dealloc_2sm call site not found"]
    readback_pos = umma_body.find("tcgen05_ld_32x32b_x32")
    if readback_pos == -1 or readback_pos >= dealloc_pos:
        return ["TMEM readback must precede tcgen05_dealloc_2sm"]

    between = umma_body[readback_pos:dealloc_pos]
    if "barrier_cluster_arrive" not in between or "barrier_cluster_wait" not in between:
        return [
            "a barrier_cluster_arrive()/barrier_cluster_wait() pair must appear between the final "
            "TMEM access and tcgen05_dealloc_2sm"
        ]
    return []


def check_timing_routing(code_only: str) -> list[str]:
    """(task section 12: "Timed and untimed route separation") Prove lexical
    clock guards (cta_rank == 0 AND timing_mode == kTimed, conjoined) and
    every host orchestration route, mirroring P2.1's equivalent check but
    adapted for P2.2's rank-0-only timing.
    """
    errors: list[str] = []

    try:
        umma_body, _, _ = extract_single_function_body(
            code_only, UMMA_BODY_DEFINITION, "umma_2sm_body"
        )
    except SourceStructureError as exc:
        return [f"invalid timing structure: {exc}"]

    clock_positions = [match.start() for match in re.finditer(r"%%clock64\b", umma_body)]
    if len(clock_positions) != 2:
        errors.append(
            f"umma_2sm_body contains {len(clock_positions)} %clock64 read(s); expected exactly "
            "two (start and end)"
        )

    timed_guard_pattern = re.compile(
        r"\bif\s*\(\s*cta_rank\s*==\s*0\s*&&\s*timing_mode\s*==\s*TimingMode::kTimed\s*\)\s*\{",
        re.DOTALL,
    )
    timed_guard_matches = list(timed_guard_pattern.finditer(umma_body))
    guard_scopes: list[tuple[int, int, str]] = []
    if len(timed_guard_matches) != 2:
        errors.append(
            f"umma_2sm_body contains {len(timed_guard_matches)} exact "
            "'cta_rank == 0 && timing_mode == TimingMode::kTimed' guard(s); expected exactly two"
        )
    for match in timed_guard_matches:
        try:
            close_brace = find_matching_brace(umma_body, match.end() - 1)
        except SourceStructureError as exc:
            errors.append(f"cannot validate timed guard scope: {exc}")
            continue
        guard_scopes.append((match.end(), close_brace, umma_body[match.end():close_brace]))

    for clock_position in clock_positions:
        containing_scopes = [
            (start, end, body) for start, end, body in guard_scopes if start <= clock_position < end
        ]
        if len(containing_scopes) != 1:
            errors.append(
                "a %clock64 read is outside an exact 'cta_rank == 0 && "
                "timing_mode == TimingMode::kTimed' lexical scope"
            )

    if len(guard_scopes) == 2:
        if "start_clock" not in guard_scopes[0][2] or "end_clock" in guard_scopes[0][2]:
            errors.append("the first timed guard must contain only the start_clock read")
        if "end_clock" not in guard_scopes[1][2] or "elapsed_cycles" not in guard_scopes[1][2]:
            errors.append("the second timed guard must contain the end_clock read and elapsed-cycle calculation")

    try:
        run_once_body, _, _ = extract_single_function_body(code_only, RUN_ONCE_DEFINITION, "run_once")
        run_untimed_body, _, _ = extract_single_function_body(
            code_only, RUN_UNTIMED_DEFINITION, "run_untimed_or_die"
        )
        run_timed_body, _, _ = extract_single_function_body(code_only, RUN_TIMED_DEFINITION, "run_timed_or_die")
        self_test_body, _, _ = extract_single_function_body(code_only, SELF_TEST_DEFINITION, "run_self_test")
        main_body, _, _ = extract_single_function_body(code_only, MAIN_DEFINITION, "main")
    except SourceStructureError as exc:
        errors.append(f"cannot validate timing routes: {exc}")
        return errors

    kernel_mode_forward = re.compile(
        r"spec\.kernel\s*<<<.*?>>>\s*\(\s*iterations\s*,\s*mode\s*,\s*d_out_device\s*,", re.DOTALL
    )
    if not kernel_mode_forward.search(run_once_body):
        errors.append("run_once must pass its TimingMode mode argument to the selected kernel launch")
    if not re.search(
        r"\bumma_2sm_body\s*<\s*N\s*,\s*DEPTH\s*>\s*\(\s*iterations\s*,\s*timing_mode\s*,", code_only
    ):
        errors.append("visible kernel wrappers must forward timing_mode to umma_2sm_body")

    untimed_wrapper_call = re.compile(r"\brun_once\s*\(\s*spec\s*,\s*iterations\s*,\s*TimingMode::kUntimed\s*\)\s*;")
    if not untimed_wrapper_call.search(run_untimed_body) or "TimingMode::kTimed" in run_untimed_body:
        errors.append("run_untimed_or_die must route exclusively through run_once(..., TimingMode::kUntimed)")

    timed_wrapper_call = re.compile(r"\brun_once\s*\(\s*spec\s*,\s*iterations\s*,\s*TimingMode::kTimed\s*\)\s*;")
    if not timed_wrapper_call.search(run_timed_body) or "TimingMode::kUntimed" in run_timed_body:
        errors.append("run_timed_or_die must route exclusively through run_once(..., TimingMode::kTimed)")

    self_test_untimed_call = re.compile(
        r"\brun_once\s*\(\s*spec\s*,\s*kSelfTestIterations\s*,\s*TimingMode::kUntimed\s*\)\s*;"
    )
    if not self_test_untimed_call.search(self_test_body) or "TimingMode::kTimed" in self_test_body:
        errors.append("self-test must call run_once(..., TimingMode::kUntimed) and never use kTimed")

    prevalidation_call = re.compile(r"\brun_untimed_or_die\s*\(\s*\*spec\s*,\s*cli\.iterations\s*\)\s*;")
    warmup_loop_pattern = re.compile(
        r"\bfor\s*\(\s*int64_t\s+w\s*=\s*0\s*;\s*w\s*<\s*cli\.warmup_iterations\s*;\s*\+\+w\s*\)\s*\{", re.DOTALL
    )
    repetition_loop_pattern = re.compile(
        r"\bfor\s*\(\s*int64_t\s+rep\s*=\s*0\s*;\s*rep\s*<\s*cli\.repetitions\s*;\s*\+\+rep\s*\)\s*\{", re.DOTALL
    )
    timed_repetition_call = re.compile(r"\brun_timed_or_die\s*\(\s*\*spec\s*,\s*cli\.iterations\s*\)\s*;")

    warmup_matches = list(warmup_loop_pattern.finditer(main_body))
    if len(warmup_matches) != 1:
        errors.append(f"main must contain exactly one warm-up loop, found {len(warmup_matches)}")
    else:
        warmup_match = warmup_matches[0]
        try:
            warmup_close = find_matching_brace(main_body, warmup_match.end() - 1)
            warmup_body = main_body[warmup_match.end():warmup_close]
        except SourceStructureError as exc:
            errors.append(f"cannot validate warm-up timing route: {exc}")
        else:
            if not prevalidation_call.search(warmup_body) or "run_timed_or_die" in warmup_body:
                errors.append("every warm-up launch must route through run_untimed_or_die")
            main_before_warmup = main_body[:warmup_match.start()]
            if not prevalidation_call.search(main_before_warmup):
                errors.append("pre-timing correctness validation must route through run_untimed_or_die before the warm-up loop")

    repetition_matches = list(repetition_loop_pattern.finditer(main_body))
    if len(repetition_matches) != 1:
        errors.append(f"main must contain exactly one timed-repetition loop, found {len(repetition_matches)}")
    else:
        repetition_match = repetition_matches[0]
        try:
            repetition_close = find_matching_brace(main_body, repetition_match.end() - 1)
            repetition_body = main_body[repetition_match.end():repetition_close]
        except SourceStructureError as exc:
            errors.append(f"cannot validate timed-repetition route: {exc}")
        else:
            if not timed_repetition_call.search(repetition_body) or "run_untimed_or_die" in repetition_body:
                errors.append("every measured repetition must route through run_timed_or_die")
    return errors


def check_source(source_text: str) -> list[str]:
    """Full source-level contract: comment/literal-aware forbidden/required
    PTX text, per-rank mapping, collective TMEM lifecycle, rank-0-only
    issue, cluster-sync-before-dealloc, and timing-mode routing. Fails
    closed (non-empty list) if the lexical scan itself cannot be trusted.
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
            errors.append(f"source is missing required text: {description}")
    errors.extend(check_launch_guard_ordering(code_only))
    errors.extend(check_rank_mapping(code_only))
    errors.extend(check_collective_tmem_lifecycle(code_only))
    errors.extend(check_rank0_only_issue(code_only))
    errors.extend(check_cluster_sync_before_dealloc(code_only))
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

        utchmma_addrs = instruction_addresses(UTCHMMA_2CTA_PATTERN, text_block)
        utcbar_addrs = instruction_addresses(UTCBAR_MULTICAST_PATTERN, text_block)
        trywait_addrs = instruction_addresses(TRYWAIT_PATTERN, text_block)
        alloc_addrs = instruction_addresses(ALLOC_2CTA_PATTERN, text_block)
        dealloc_addrs = instruction_addresses(DEALLOC_PATTERN, text_block)
        ldtm_addrs = instruction_addresses(LDTM_PATTERN, text_block)
        cgabar_addrs = instruction_addresses(CGABAR_PATTERN, text_block)
        cluster_rank_addrs = instruction_addresses(CLUSTER_RANK_PATTERN, text_block)

        if len(utchmma_addrs) != depth:
            spec_errors.append(f"UTCHMMA.2CTA count is {len(utchmma_addrs)}, expected exactly depth={depth}")
        elif depth > 1:
            deltas = {b - a for a, b in zip(utchmma_addrs, utchmma_addrs[1:])}
            if len(deltas) != 1:
                spec_errors.append(
                    f"UTCHMMA.2CTA occurrences are not uniformly spaced ({sorted(deltas)}); "
                    "a runtime back-edge may be standing in for compile-time unrolling"
                )

        if not alloc_addrs:
            spec_errors.append("no collective TMEM allocation instruction (expected UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN)")
        if not dealloc_addrs:
            spec_errors.append("no TMEM deallocation instruction (expected UVIRTCOUNT.DEALLOC.SMPOOL)")
        if not utcbar_addrs:
            spec_errors.append("no multicast commit instruction (expected UTCBAR.2CTA.MULTICAST)")
        if not trywait_addrs:
            spec_errors.append("no mbarrier completion wait (expected SYNCS.PHASECHK.TRANS*.TRYWAIT)")
        if len(cgabar_addrs) < 2:
            spec_errors.append(
                f"insufficient cluster-barrier evidence: {len(cgabar_addrs)} UCGABAR_ARV/WAIT "
                "occurrence(s), expected at least 2 (one arrive/wait pair)"
            )
        if not cluster_rank_addrs:
            spec_errors.append("no cluster-rank evidence (expected SR_CgaCtaId)")

        if utchmma_addrs and utcbar_addrs:
            if max(utcbar_addrs) <= max(utchmma_addrs):
                spec_errors.append("no multicast commit found after the last UTCHMMA.2CTA")
            elif trywait_addrs and max(trywait_addrs) <= max(utcbar_addrs):
                spec_errors.append("no mbarrier wait found after the commit")

        expected_fragments = n // 32
        if len(ldtm_addrs) != expected_fragments:
            spec_errors.append(f"LDTM.x32 count is {len(ldtm_addrs)}, expected exactly N/32={expected_fragments}")

        if dealloc_addrs:
            last_use = max(utchmma_addrs + ldtm_addrs) if (utchmma_addrs or ldtm_addrs) else -1
            if last_use >= 0 and min(dealloc_addrs) <= last_use:
                spec_errors.append("TMEM deallocation is not ordered after the last TMEM use")
            # Cluster synchronization before deallocation: at least one
            # UCGABAR_WAIT must sit strictly between the last TMEM use and
            # the (first) deallocation address.
            if last_use >= 0 and cgabar_addrs:
                sync_between = [a for a in cgabar_addrs if last_use < a < min(dealloc_addrs)]
                if not sync_between:
                    spec_errors.append(
                        "no cluster-barrier evidence (UCGABAR_ARV/WAIT) found between the last TMEM "
                        "use and TMEM deallocation"
                    )

        if spec_errors:
            errors.extend(f"{label}: {detail}" for detail in spec_errors)
            status_lines.append(f"FAIL {label}")
        else:
            status_lines.append(
                f"OK   {label} UTCHMMA.2CTA={len(utchmma_addrs)} LDTM.x32={len(ldtm_addrs)} "
                f"alloc={len(alloc_addrs)} dealloc={len(dealloc_addrs)} cgabar={len(cgabar_addrs)}"
            )

    # Whole-binary forbidden-instruction checks.
    for pattern, description in FORBIDDEN_PATTERNS:
        if pattern.search(sass_text):
            errors.append(f"forbidden instruction present: {description}")

    return status_lines, errors


ELF_SECTION_PATTERN = re.compile(r"^\.nv\.info\.(\S+)\s*$", re.MULTILINE)


def analyze_elf(elf_text: str) -> tuple[list[str], list[str]]:
    """Verify every expected specialization's own .nv.info.<symbol> section
    (cuobjdump -elf output) carries both EIATTR_EXPLICIT_CLUSTER and an
    EIATTR_CTA_PER_CLUSTER value of exactly "0x2 0x1 0x1" -- direct,
    per-kernel, binary-level evidence of the compile-time two-CTA cluster
    declaration (task section 11: "observable 2-CTA/CTA-pair ... evidence
    when exposed by the CUDA 13.1 disassembly").
    """
    matches = list(ELF_SECTION_PATTERN.finditer(elf_text))
    status_lines = [f"found {len(matches)} .nv.info.<symbol> ELF section(s)"]
    errors: list[str] = []

    sections: dict[str, str] = {}
    for i, m in enumerate(matches):
        name = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(elf_text)
        sections[name] = elf_text[start:end]

    found_specs: set[tuple[int, int]] = set()
    for symbol, body in sections.items():
        spec = parse_specialization(symbol)
        if spec is None:
            continue
        found_specs.add(spec)
        label = f"N={spec[0]} depth={spec[1]}"
        if "EIATTR_EXPLICIT_CLUSTER" not in body:
            errors.append(f"{label}: ELF section for {symbol} is missing EIATTR_EXPLICIT_CLUSTER")
        if not re.search(r"EIATTR_CTA_PER_CLUSTER[\s\S]*?Value:\s*0x2\s+0x1\s+0x1\b", body):
            errors.append(
                f"{label}: ELF section for {symbol} is missing EIATTR_CTA_PER_CLUSTER with value 0x2 0x1 0x1"
            )

    for spec in sorted(EXPECTED_SPECS - found_specs):
        errors.append(f"missing ELF .nv.info section for N={spec[0]} depth={spec[1]}")

    return status_lines, errors


# ---------------------------------------------------------------------------
# Synthetic SASS for --self-test, shaped after this project's own real
# cuobjdump -sass output for build/compute/umma_2sm on sm_103a (CUDA
# 13.1.80 ptxas): one ELECT+UTCHMMA.2CTA pair per burst position at a
# uniform 0x60 spacing, one UTCBAR.2CTA.MULTICAST (commit) plus TRYWAITs,
# two ALIGN.2CTA allocs (ptxas peels a fast-path check plus a retry-loop
# body, same duplication pattern documented for P2.1), a cluster-barrier
# pair before dealloc, one DEALLOC.SMPOOL after N/32 LDTM.x32 fragments.
# ---------------------------------------------------------------------------
def synthetic_block(n: int, depth: int, *, utchmma_count: int | None = None, spacing: int = 0x60,
                     alloc_count: int = 2, dealloc_count: int = 1, commit_count: int = 1,
                     trywait_count: int = 2, ldtm_count: int | None = None,
                     cgabar_count: int = 2, cgabar_before_dealloc: bool = True,
                     cluster_rank_count: int = 1,
                     dealloc_before_last_use: bool = False, extra_uneven_gap: bool = False,
                     forbidden_mnemonic: str | None = None, non_2cta_mma: bool = False,
                     non_multicast_commit: bool = False, non_2cta_alloc: bool = False,
                     symbol: str | None = None) -> str:
    if utchmma_count is None:
        utchmma_count = depth
    if ldtm_count is None:
        ldtm_count = n // 32

    lines = [f"\t\tFunction : {symbol or f'umma_2sm_m256n{n}k16_d{depth}'}"]
    addr = 0x0A00
    for i in range(alloc_count):
        mnemonic = "UTCATOMSWS.FIND_AND_SET.ALIGN" if non_2cta_alloc else "UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN"
        lines.append(f"        /*{addr:04x}*/                   {mnemonic} UP0, UR5, UR5 ;")
        addr += 0x60

    for i in range(cluster_rank_count):
        lines.append(f"        /*{addr:04x}*/                   S2R R5, SR_CgaCtaId ;")
        addr += 0x10

    if cgabar_count >= 1:
        lines.append(f"        /*{addr:04x}*/                   UCGABAR_ARV ;")
        addr += 0x10
        lines.append(f"        /*{addr:04x}*/                   UCGABAR_WAIT ;")
        addr += 0x10

    utchmma_addrs = []
    for i in range(utchmma_count):
        lines.append(f"        /*{addr:04x}*/               @P0 ELECT P1, URZ, PT ;")
        addr += 0x10
        mnemonic = "UTCHMMA" if non_2cta_mma else "UTCHMMA.2CTA"
        lines.append(
            f"        /*{addr:04x}*/                   {mnemonic} gdesc[UR12], gdesc[UR14], tmem[UR6], "
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
        mnemonic = "UTCBAR" if non_multicast_commit else "UTCBAR.2CTA.MULTICAST"
        lines.append(f"        /*{addr:04x}*/                   {mnemonic} [UR4], URZ, UR7 ;")
        addr += 0x10
    for _ in range(trywait_count):
        lines.append(f"        /*{addr:04x}*/                   SYNCS.PHASECHK.TRANS64.TRYWAIT P1, [R12+URZ], R3 ;")
        addr += 0x10

    ldtm_addrs = []
    for _ in range(ldtm_count):
        lines.append(f"        /*{addr:04x}*/                   LDTM.x32 R16, tmem[UR6] ;")
        ldtm_addrs.append(addr)
        addr += 0x10

    if cgabar_count >= 2 and cgabar_before_dealloc:
        lines.append(f"        /*{addr:04x}*/                   UCGABAR_ARV ;")
        addr += 0x10
        lines.append(f"        /*{addr:04x}*/                   UCGABAR_WAIT ;")
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


def synthetic_elf(omit_cluster_attrs: set[tuple[int, int]] | None = None,
                   omit_sections: set[tuple[int, int]] | None = None,
                   bad_cta_per_cluster: set[tuple[int, int]] | None = None) -> str:
    omit_cluster_attrs = omit_cluster_attrs or set()
    omit_sections = omit_sections or set()
    bad_cta_per_cluster = bad_cta_per_cluster or set()
    parts = [".nv.info\n\tAttribute:\tEIATTR_CUDA_API_VERSION\n"]
    for n, depth in sorted(EXPECTED_SPECS):
        if (n, depth) in omit_sections:
            continue
        symbol = f"umma_2sm_m256n{n}k16_d{depth}"
        body = "\t<0x1>\n\tAttribute:\tEIATTR_KPARAM_INFO\n"
        if (n, depth) not in omit_cluster_attrs:
            body += "\t<0x2>\n\tAttribute:\tEIATTR_EXPLICIT_CLUSTER\n\tFormat:\tEIFMT_NVAL\n"
            cta_per_cluster = "0x9 0x1 0x1" if (n, depth) in bad_cta_per_cluster else "0x2 0x1 0x1"
            body += (
                "\t<0x3>\n\tAttribute:\tEIATTR_CTA_PER_CLUSTER\n\tFormat:\tEIFMT_SVAL\n"
                f"\tValue:\t{cta_per_cluster} \n"
            )
        parts.append(f".nv.info.{symbol}\n{body}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Synthetic source snippet for the source-level self-test cases below. Built
# from named fields so a single targeted override isolates exactly one
# defect, while every other required property stays intact -- mirroring the
# actual structure of src/compute/umma_2sm.cu closely enough to exercise the
# real regexes without needing the full file.
# ---------------------------------------------------------------------------
def golden_source_snippet(**overrides: str) -> str:
    fields = {
        "wait_ld_text": "tcgen05.wait::ld.sync.aligned;",
        "fence_text": "tcgen05.fence::after_thread_sync;",
        "mma_text": "tcgen05.mma.cta_group::2.kind::f16 [x], a, b, i, p;",
        "commit_text": (
            "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster."
            "multicast::cluster.b64 [x], y;"
        ),
        "alloc_text": "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [x], y;",
        "dealloc_text": "tcgen05.dealloc.cta_group::2.sync.aligned.b32 x, y;",
        "relinquish_text": "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;",
        "cluster_dims_text": "__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128) void k() {}",
        "ctarank_text": "cuda::ptx::get_sreg_cluster_ctarank();",
        "nctarank_text": "cuda::ptx::get_sreg_cluster_nctarank();",
        "mask_text": "0x0003u",
        "extra_forbidden_line": "",
        "launch_predicate": (
            "return gridDim.x == kExpectedGridDim && gridDim.y == 1 && gridDim.z == 1 && "
            "blockDim.x == kExpectedBlockDimX && blockDim.y == 1 && blockDim.z == 1 && "
            "cluster_nctarank == kExpectedClusterCtas && cluster_ctarank < kExpectedClusterCtas;"
        ),
        "launch_guard": (
            "if (!launch_contract_is_valid(cluster_nctarank, cluster_ctarank)) { "
            "if (cluster_ctarank == 0) g_launch_ok[0] = 0; else if (cluster_ctarank == 1) "
            "g_launch_ok[1] = 0; return; }"
        ),
        "accepted_path": (
            "if (cluster_ctarank == 0) g_launch_ok[0] = 1; else if (cluster_ctarank == 1) "
            "g_launch_ok[1] = 1;"
        ),
        "a_loop": (
            "for (int idx = tid; idx < kMLocal * kK; idx += kThreadsPerCta) { "
            "const int local_row = idx / kK; const int k = idx % kK; "
            "const int global_row = cta_rank * kMLocal + local_row; "
            "A[smem_core_tile_index(local_row/8, local_row%8, k)] = __float2bfloat16(1.0f); }"
        ),
        "b_loop": (
            "for (int idx = tid; idx < N * kK; idx += kThreadsPerCta) { "
            "const int col = idx / kK; const int k = idx % kK; "
            "B[smem_core_tile_index(col/8, col%8, k)] = __float2bfloat16(1.0f); }"
        ),
        "alloc_call": "if (warp_id == 0) { tcgen05_alloc_2sm(x, N); }",
        "dealloc_call": "if (warp_id == 0) { tcgen05_dealloc_2sm(tmem_d, N); tcgen05_relinquish_alloc_permit_2sm(); }",
        "issue_call": (
            "if (cta_rank == 0) { issue_one_umma_2sm(tmem_d, a_desc, b_desc, idesc, 0); "
            "commit_umma_2sm_multicast(mbar_addr, 0x0003u); }"
        ),
        "wait_call": "while (!cuda::ptx::mbarrier_try_wait_parity(&mbar, parity)) {}",
        "cluster_sync_before_dealloc": "cuda::ptx::barrier_cluster_arrive(); cuda::ptx::barrier_cluster_wait();",
        "readback_global_row": (
            "const int global_row = cta_rank * kMLocal + local_row;"
        ),
        "readback_call": (
            "tcgen05_ld_32x32b_x32(make_tmem_load_address(tmem_d, warp_id, frag), regs); "
            "g_d_out[static_cast<int64_t>(global_row) * N + frag * 32 + i] = 0.0f;"
        ),
        "timing_guard_a": (
            'if (cta_rank == 0 && timing_mode == TimingMode::kTimed) { '
            'asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock)); }'
        ),
        "timing_guard_b": (
            'if (cta_rank == 0 && timing_mode == TimingMode::kTimed) { '
            'asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock)); '
            "elapsed_cycles = end_clock - start_clock; }"
        ),
        "kernel_mode_forward": (
            "umma_2sm_body<N, DEPTH>(iterations, timing_mode, g_d_out, g_elapsed_cycles, g_launch_ok);"
        ),
        "run_once_mode_forward": "spec.kernel<<<2, 128>>>(iterations, mode, d_out_device, cycles_device, launch_ok_device);",
        "untimed_call_a": "run_once(spec, iterations, TimingMode::kUntimed);",
        "untimed_call_b": "run_once(spec, kSelfTestIterations, TimingMode::kUntimed);",
        "timed_call": "run_once(spec, iterations, TimingMode::kTimed);",
        "prevalidation_call": "run_untimed_or_die(*spec, cli.iterations);",
        "warmup_call": "run_untimed_or_die(*spec, cli.iterations);",
        "timed_repetition_call": "run_timed_or_die(*spec, cli.iterations);",
    }
    fields.update(overrides)
    return (
        "__device__ bool launch_contract_is_valid(uint32_t cluster_nctarank, uint32_t cluster_ctarank) {\n"
        f"    {fields['launch_predicate']}\n"
        "}\n"
        "__device__ void umma_2sm_body(int64_t iterations, TimingMode timing_mode) {\n"
        f"    {fields['launch_guard']}\n"
        f"    {fields['accepted_path']}\n"
        "    const int cta_rank = static_cast<int>(cluster_ctarank);\n"
        f"    {fields['a_loop']}\n"
        f"    {fields['b_loop']}\n"
        "    __syncthreads();\n"
        f"    {fields['alloc_call']}\n"
        "    if (is_leader) {\n"
        f"        {fields['timing_guard_a']}\n"
        f"        {fields['issue_call']}\n"
        f"        {fields['wait_call']}\n"
        f"        {fields['timing_guard_b']}\n"
        "    }\n"
        f"    {fields['readback_global_row']}\n"
        f"    {fields['readback_call']}\n"
        f"    {fields['cluster_sync_before_dealloc']}\n"
        f"    {fields['dealloc_call']}\n"
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
        f"{fields['mma_text']}\n"
        f"{fields['commit_text']}\n"
        f"{fields['alloc_text']}\n"
        f"{fields['dealloc_text']}\n"
        f"{fields['relinquish_text']}\n"
        f"{fields['cluster_dims_text']}\n"
        f"{fields['ctarank_text']}\n"
        f"{fields['nctarank_text']}\n"
        f"{fields['mask_text']}\n"
        f"{fields['extra_forbidden_line']}\n"
    )


def run_self_test() -> int:
    cases: list[tuple[str, str, str | None]] = [
        ("accepts a complete, correctly-shaped set of twelve specializations", synthetic_sass(), None),
        ("rejects a missing symbol", synthetic_sass(omit={(64, 4)}), "missing specialization N=64 depth=4"),
        (
            "rejects an extra/unexpected symbol",
            synthetic_sass(extra=[synthetic_block(64, 4, symbol="umma_2sm_m256n64k16_d999")]),
            "unexpected specialization N=64 depth=999",
        ),
        (
            "rejects a duplicate configuration",
            synthetic_sass() + "\n" + synthetic_block(64, 4) + "\n",
            "duplicate configuration N=64 depth=4",
        ),
        (
            "rejects a missing UTCHMMA.2CTA burst",
            synthetic_sass({(64, 4): {"utchmma_count": 0}}),
            "UTCHMMA.2CTA count is 0, expected exactly depth=4",
        ),
        (
            "rejects an incorrect depth (fewer UTCHMMA.2CTA than depth requires)",
            synthetic_sass({(128, 16): {"utchmma_count": 15}}),
            "UTCHMMA.2CTA count is 15, expected exactly depth=16",
        ),
        (
            "rejects an incorrect depth (more UTCHMMA.2CTA than depth requires)",
            synthetic_sass({(256, 64): {"utchmma_count": 65}}),
            "UTCHMMA.2CTA count is 65, expected exactly depth=64",
        ),
        (
            "rejects a non-uniformly-spaced burst (possible back-edge standing in for unrolling)",
            synthetic_sass({(128, 64): {"extra_uneven_gap": True}}),
            "not uniformly spaced",
        ),
        (
            "rejects a missing multicast commit",
            synthetic_sass({(64, 16): {"commit_count": 0}}),
            "no multicast commit instruction",
        ),
        (
            "rejects a non-multicast commit (1-SM/fallback form)",
            synthetic_sass({(64, 16): {"non_multicast_commit": True}}),
            "forbidden instruction present: a non-.2CTA.MULTICAST",
        ),
        (
            "rejects a non-.2CTA MMA (1-SM fallback form)",
            synthetic_sass({(64, 4): {"non_2cta_mma": True}}),
            "forbidden instruction present: a non-.2CTA (1-SM fallback) UTCHMMA",
        ),
        (
            "rejects a non-.2CTA TMEM allocation (1-SM fallback form)",
            synthetic_sass({(128, 4): {"non_2cta_alloc": True}}),
            "forbidden instruction present: a non-.2CTA (1-SM fallback) TMEM allocation",
        ),
        (
            "rejects a missing mbarrier wait",
            synthetic_sass({(64, 64): {"trywait_count": 0}}),
            "no mbarrier completion wait",
        ),
        (
            "rejects a missing TMEM allocation",
            synthetic_sass({(128, 4): {"alloc_count": 0}}),
            "no collective TMEM allocation instruction",
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
            "rejects missing cluster-barrier evidence",
            synthetic_sass({(64, 4): {"cgabar_count": 0}}),
            "insufficient cluster-barrier evidence",
        ),
        (
            "rejects missing cluster-rank evidence",
            synthetic_sass({(64, 4): {"cluster_rank_count": 0}}),
            "no cluster-rank evidence",
        ),
        (
            "rejects a missing cluster-barrier pair between final TMEM use and dealloc",
            synthetic_sass({(64, 16): {"cgabar_before_dealloc": False, "cgabar_count": 1}}),
            "no cluster-barrier evidence (UCGABAR_ARV/WAIT) found between the last TMEM use and TMEM deallocation",
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
    ]

    failures: list[str] = []
    for name, sass_text, expected_error in cases:
        _, errors = analyze_sass(sass_text)
        if expected_error is None:
            passed = not errors
        else:
            passed = any(expected_error in error for error in errors)
        if passed:
            print(f"check_umma_2sm_sass: self-test: PASS: {name}", file=sys.stderr)
        else:
            failures.append(name)
            print(f"check_umma_2sm_sass: self-test: FAIL: {name}; errors={errors}", file=sys.stderr)

    elf_cases: list[tuple[str, str, str | None]] = [
        ("ELF check accepts a complete set of twelve cluster-declared sections", synthetic_elf(), None),
        (
            "ELF check rejects a missing EIATTR_EXPLICIT_CLUSTER attribute",
            synthetic_elf(omit_cluster_attrs={(64, 4)}),
            "missing EIATTR_EXPLICIT_CLUSTER",
        ),
        (
            "ELF check rejects a missing .nv.info section entirely",
            synthetic_elf(omit_sections={(128, 16)}),
            "missing ELF .nv.info section for N=128 depth=16",
        ),
        (
            "ELF check rejects an incorrect EIATTR_CTA_PER_CLUSTER value",
            synthetic_elf(bad_cta_per_cluster={(256, 64)}),
            "missing EIATTR_CTA_PER_CLUSTER with value 0x2 0x1 0x1",
        ),
    ]
    for name, elf_text, expected_error in elf_cases:
        _, errors = analyze_elf(elf_text)
        if expected_error is None:
            passed = not errors
        else:
            passed = any(expected_error in error for error in errors)
        if passed:
            print(f"check_umma_2sm_sass: self-test: PASS: {name}", file=sys.stderr)
        else:
            failures.append(name)
            print(f"check_umma_2sm_sass: self-test: FAIL: {name}; errors={errors}", file=sys.stderr)

    source_cases: list[tuple[str, str, str | None]] = [
        ("source check accepts a fully valid source", golden_source_snippet(), None),
        (
            "source check rejects cta_group::1",
            golden_source_snippet(extra_forbidden_line="tcgen05.mma.cta_group::1.kind::f16 [x], a, b, i, p;"),
            "cta_group::1",
        ),
        (
            "source check rejects a non-kind::f16 MMA kind",
            golden_source_snippet(extra_forbidden_line="tcgen05.mma.cta_group::2.kind::tf32 [x], a, b, i, p;"),
            "a non-kind::f16 MMA kind",
        ),
        (
            "source check rejects a sparse (.sp) MMA form",
            golden_source_snippet(extra_forbidden_line="tcgen05.mma.sp.cta_group::2.kind::f16 [x], a, b, m, i, p;"),
            "a sparse (.sp) MMA form",
        ),
        (
            "source check rejects block_scale",
            golden_source_snippet(extra_forbidden_line="tcgen05.mma.cta_group::2.kind::mxf4.block_scale [x];"),
            "block_scale",
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
            "source check rejects a missing tcgen05.mma.cta_group::2.kind::f16",
            golden_source_snippet(mma_text=""),
            "tcgen05.mma.cta_group::2.kind::f16",
        ),
        (
            "source check rejects a missing multicast commit form",
            golden_source_snippet(commit_text=""),
            "tcgen05.commit.cta_group::2...shared::cluster.multicast::cluster.b64",
        ),
        (
            "source check rejects a missing tcgen05.alloc.cta_group::2",
            golden_source_snippet(alloc_text=""),
            "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32",
        ),
        (
            "source check rejects a missing tcgen05.dealloc.cta_group::2",
            golden_source_snippet(dealloc_text=""),
            "tcgen05.dealloc.cta_group::2.sync.aligned.b32",
        ),
        (
            "source check rejects a missing tcgen05.relinquish_alloc_permit.cta_group::2",
            golden_source_snippet(relinquish_text=""),
            "tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned",
        ),
        (
            "source check rejects a missing __cluster_dims__(2, 1, 1)",
            golden_source_snippet(cluster_dims_text=""),
            "__cluster_dims__(2, 1, 1)",
        ),
        (
            "source check rejects a missing get_sreg_cluster_ctarank",
            golden_source_snippet(ctarank_text=""),
            "cuda::ptx::get_sreg_cluster_ctarank",
        ),
        (
            "source check rejects a missing get_sreg_cluster_nctarank",
            golden_source_snippet(nctarank_text=""),
            "cuda::ptx::get_sreg_cluster_nctarank",
        ),
        (
            "source check rejects a missing barrier_cluster_arrive",
            golden_source_snippet(cluster_sync_before_dealloc="cuda::ptx::barrier_cluster_wait();"),
            "cuda::ptx::barrier_cluster_arrive",
        ),
        (
            "source check rejects a missing barrier_cluster_wait",
            golden_source_snippet(cluster_sync_before_dealloc="cuda::ptx::barrier_cluster_arrive();"),
            "cuda::ptx::barrier_cluster_wait",
        ),
        (
            "source check rejects a missing exact multicast mask 0x0003",
            golden_source_snippet(mask_text="", issue_call=(
                "if (cta_rank == 0) { issue_one_umma_2sm(tmem_d, a_desc, b_desc, idesc, 0); "
                "commit_umma_2sm_multicast(mbar_addr, 0x0007u); }"
            )),
            "the exact multicast CTA mask 0x0003",
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
            golden_source_snippet(extra_forbidden_line="/* cta_group::1 block_scale */"),
            None,
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
                    "if (launch_contract_is_valid(cluster_nctarank, cluster_ctarank)) { "
                    "g_launch_ok[0] = 0; g_launch_ok[1] = 0; return; }"
                )
            ),
            "if (!launch_contract_is_valid",
        ),
        (
            "source check rejects a launch rejection that only writes rank 0's status",
            golden_source_snippet(
                launch_guard=(
                    "if (!launch_contract_is_valid(cluster_nctarank, cluster_ctarank)) { "
                    "if (cluster_ctarank == 0) g_launch_ok[0] = 0; return; }"
                )
            ),
            "g_launch_ok[1] = 0 for rank 1",
        ),
        (
            "source check rejects an acceptance path that only writes rank 0's status",
            golden_source_snippet(accepted_path="if (cluster_ctarank == 0) g_launch_ok[0] = 1;"),
            "g_launch_ok[1] = 1 for rank 1",
        ),
        (
            "source check rejects A initialization that does not depend on cta_rank",
            golden_source_snippet(
                a_loop=(
                    "for (int idx = tid; idx < kMLocal * kK; idx += kThreadsPerCta) { "
                    "const int local_row = idx / kK; const int k = idx % kK; "
                    "A[smem_core_tile_index(local_row/8, local_row%8, k)] = __float2bfloat16(1.0f); }"
                )
            ),
            "A initialization must depend on cta_rank",
        ),
        (
            "source check rejects B initialization that depends on cta_rank (should be identical)",
            golden_source_snippet(
                b_loop=(
                    "for (int idx = tid; idx < N * kK; idx += kThreadsPerCta) { "
                    "const int col = idx / kK; const int k = idx % kK; "
                    "const int v = cta_rank + col; "
                    "B[smem_core_tile_index(col/8, col%8, k)] = __float2bfloat16(1.0f); }"
                )
            ),
            "B initialization must NOT depend on cta_rank",
        ),
        (
            "source check rejects TMEM allocation gated by cta_rank instead of warp_id",
            golden_source_snippet(alloc_call="if (cta_rank == 0) { tcgen05_alloc_2sm(x, N); }"),
            "tcgen05_alloc_2sm must not be issued from inside an 'cta_rank == 0' conditional",
        ),
        (
            "source check rejects deallocation gated by is_leader instead of warp_id",
            golden_source_snippet(
                dealloc_call="if (is_leader) { tcgen05_dealloc_2sm(tmem_d, N); tcgen05_relinquish_alloc_permit_2sm(); }"
            ),
            "must not be issued from inside an 'is_leader' conditional",
        ),
        (
            "source check rejects an MMA/commit issue not confined to cta_rank == 0",
            golden_source_snippet(
                issue_call="issue_one_umma_2sm(tmem_d, a_desc, b_desc, idesc, 0); commit_umma_2sm_multicast(mbar_addr, 0x0003u);"
            ),
            "must both be issued from a single 'if (cta_rank == 0)' block",
        ),
        (
            "source check rejects a wait enclosed in cta_rank == 0",
            golden_source_snippet(
                issue_call=(
                    "if (cta_rank == 0) { issue_one_umma_2sm(tmem_d, a_desc, b_desc, idesc, 0); "
                    "commit_umma_2sm_multicast(mbar_addr, 0x0003u); "
                    "while (!cuda::ptx::mbarrier_try_wait_parity(&mbar, parity)) {} }"
                ),
                wait_call="",
            ),
            "mbarrier completion wait must not be enclosed in a cta_rank == 0 condition",
        ),
        (
            "source check rejects a TMEM load address offset by cta_rank",
            golden_source_snippet(
                readback_call=(
                    "tcgen05_ld_32x32b_x32(make_tmem_load_address(tmem_d, warp_id, frag) + cta_rank * kMLocal, regs); "
                    "g_d_out[static_cast<int64_t>(global_row) * N + frag * 32 + i] = 0.0f;"
                )
            ),
            "TMEM load address must not be offset by cta_rank",
        ),
        (
            "source check rejects D readback written by local_row instead of global_row",
            golden_source_snippet(
                readback_call=(
                    "tcgen05_ld_32x32b_x32(make_tmem_load_address(tmem_d, warp_id, frag), regs); "
                    "g_d_out[static_cast<int64_t>(local_row) * N + frag * 32 + i] = 0.0f;"
                )
            ),
            "D readback must write g_d_out indexed by the GLOBAL row",
        ),
        (
            "source check rejects a missing cluster sync before deallocation",
            golden_source_snippet(cluster_sync_before_dealloc=""),
            "barrier_cluster_arrive()/barrier_cluster_wait() pair must appear",
        ),
        (
            "source check rejects missing timed clock guards and reads",
            golden_source_snippet(timing_guard_a="", timing_guard_b=""),
            "%clock64 read(s)",
        ),
        (
            "source check rejects a timed guard missing the cta_rank == 0 conjunct",
            golden_source_snippet(
                timing_guard_a=(
                    'if (timing_mode == TimingMode::kTimed) { '
                    'asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock)); }'
                ),
            ),
            "guard(s); expected exactly two",
        ),
        (
            "source check rejects a self-test routed through TimingMode::kTimed",
            golden_source_snippet(untimed_call_b="run_once(spec, kSelfTestIterations, TimingMode::kTimed);"),
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
            print(f"check_umma_2sm_sass: self-test: PASS: {name}", file=sys.stderr)
        else:
            failures.append(name)
            print(f"check_umma_2sm_sass: self-test: FAIL: {name}; errors={errors}", file=sys.stderr)

    mandatory_validation_cases: list[tuple[str, bool]] = [
        (
            "mandatory source validation: default path resolves to src/compute/umma_2sm.cu",
            resolve_default_source_path().as_posix().endswith("src/compute/umma_2sm.cu"),
        ),
        (
            "mandatory source validation: a missing canonical source fails closed (non-empty errors)",
            bool(validate_source_file(Path("/nonexistent-path-should-never-exist/umma_2sm.cu"))),
        ),
    ]
    for name, ok in mandatory_validation_cases:
        if ok:
            print(f"check_umma_2sm_sass: self-test: PASS: {name}", file=sys.stderr)
        else:
            failures.append(name)
            print(f"check_umma_2sm_sass: self-test: FAIL: {name}", file=sys.stderr)

    total = len(cases) + len(elf_cases) + len(source_cases) + len(mandatory_validation_cases)
    if failures:
        print(f"check_umma_2sm_sass: self-test: FAILED ({len(failures)}/{total} case(s))", file=sys.stderr)
        return 1
    print(f"check_umma_2sm_sass: self-test: OK ({total} cases)", file=sys.stderr)
    return 0


def check_binary(binary_path: str, out_path: str, explicit_source_path: str | None) -> int:
    try:
        sass_result = subprocess.run(["cuobjdump", "-sass", binary_path], capture_output=True, text=True)
    except OSError as exc:
        print(f"check_umma_2sm_sass: unable to run cuobjdump -sass: {exc}", file=sys.stderr)
        return 1
    if sass_result.returncode != 0:
        print(f"check_umma_2sm_sass: cuobjdump -sass failed (rc={sass_result.returncode}):\n{sass_result.stderr}",
              file=sys.stderr)
        return 1

    sass_text = sass_result.stdout
    try:
        with open(out_path, "w", encoding="utf-8") as output_file:
            output_file.write(sass_text)
    except OSError as exc:
        print(f"check_umma_2sm_sass: unable to write {out_path}: {exc}", file=sys.stderr)
        return 1
    print(f"check_umma_2sm_sass: wrote {out_path}", file=sys.stderr)

    status_lines, errors = analyze_sass(sass_text)
    for status in status_lines:
        print(f"check_umma_2sm_sass: {status}", file=sys.stderr)

    try:
        elf_result = subprocess.run(["cuobjdump", "-elf", binary_path], capture_output=True, text=True)
    except OSError as exc:
        print(f"check_umma_2sm_sass: unable to run cuobjdump -elf: {exc}", file=sys.stderr)
        return 1
    if elf_result.returncode != 0:
        print(f"check_umma_2sm_sass: cuobjdump -elf failed (rc={elf_result.returncode}):\n{elf_result.stderr}",
              file=sys.stderr)
        return 1

    elf_status_lines, elf_errors = analyze_elf(elf_result.stdout)
    for status in elf_status_lines:
        print(f"check_umma_2sm_sass: {status}", file=sys.stderr)
    errors.extend(elf_errors)

    # Source validation is mandatory: --source may override which file is
    # checked, but omitting it resolves the canonical repository path below
    # instead of skipping the check. There is no bypass.
    source_path = Path(explicit_source_path) if explicit_source_path is not None else resolve_default_source_path()
    source_errors = validate_source_file(source_path)
    if source_errors:
        errors.extend(f"[source {source_path}] {detail}" for detail in source_errors)
    else:
        print(
            f"check_umma_2sm_sass: source check OK ({source_path}): every required cta_group::2 PTX "
            "form, the CTA-pair rank mapping, the collective TMEM lifecycle, rank-0-only issue with "
            "the exact multicast mask, and timed/untimed routing are all real executable code, and no "
            "forbidden pattern was found (comment- and string-literal-aware scan)",
            file=sys.stderr,
        )

    if errors:
        print("check_umma_2sm_sass: contract validation failed:", file=sys.stderr)
        for error in errors:
            print(f"check_umma_2sm_sass:   - {error}", file=sys.stderr)
        return 1

    print(
        "check_umma_2sm_sass: OK: all twelve specializations contain a genuine 2-SM UTCHMMA.2CTA "
        "burst of exactly depth instructions, a real multicast-commit/wait completion sequence, a "
        "complete collective TMEM lifecycle with cluster synchronization before deallocation, "
        "correct per-rank TMEM/global addressing, ELF-level two-CTA cluster attributes, and no "
        "forbidden or 1-SM-fallback instruction",
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
                print("check_umma_2sm_sass: --source requires a path argument", file=sys.stderr)
                return 2
            source_path = args[i + 1]
            i += 2
            continue
        positional.append(args[i])
        i += 1

    if len(positional) == 2 and all(not arg.startswith("-") for arg in positional):
        return check_binary(positional[0], positional[1], source_path)

    print(
        "usage: check_umma_2sm_sass.py <binary> <output-sass-path> [--source <path>]\n"
        "       check_umma_2sm_sass.py --self-test\n"
        "the two-positional-argument form always validates the canonical source\n"
        "(src/compute/umma_2sm.cu, resolved relative to this script) even when\n"
        "--source is omitted; --source only overrides which file is checked.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
