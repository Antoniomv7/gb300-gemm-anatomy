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
    (re.compile(r"\.kind::(?!f16\b)[a-z0-9_]+"), "a non-kind::f16 MMA kind"),
    (re.compile(r"\.sp\b"), "a sparse (.sp) MMA form"),
    (re.compile(r"block_scale"), "block_scale"),
)

# Required PTX-mnemonic text. Real occurrences of every one of these live
# inside a C++ string literal passed as the template operand of a genuine
# `asm`/`asm volatile(...)` statement. A mnemonic placed only in a comment
# (already stripped upstream) or an ordinary, non-asm string literal (e.g. a
# decoy `const char*`) must NOT satisfy the check -- so these are matched
# only against text that falls inside a real asm-statement operand span (see
# find_asm_string_operand_spans/pattern_has_asm_evidence below), never
# against the raw comment-stripped text directly.
REQUIRED_ASM_EVIDENCE_PATTERNS = (
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
)

# Required C++ identifiers/attributes (never meant to live inside a string
# literal in genuine code). Matched against the string-masked view (see
# mask_string_and_char_literals below) so a decoy string containing the same
# identifier text cannot satisfy the check.
REQUIRED_IDENTIFIER_PATTERNS = (
    (re.compile(r"__cluster_dims__\s*\(\s*2\s*,\s*1\s*,\s*1\s*\)"), "__cluster_dims__(2, 1, 1)"),
    (re.compile(r"get_sreg_cluster_ctarank"), "cuda::ptx::get_sreg_cluster_ctarank"),
    (re.compile(r"get_sreg_cluster_nctarank"), "cuda::ptx::get_sreg_cluster_nctarank"),
    (re.compile(r"barrier_cluster_arrive"), "cuda::ptx::barrier_cluster_arrive"),
    (re.compile(r"barrier_cluster_wait"), "cuda::ptx::barrier_cluster_wait"),
    (re.compile(r"0x0003u?\b"), "the exact multicast CTA mask 0x0003"),
    (re.compile(r"\bfence_mbarrier_init_release_cluster\s*\("),
     "an executable fence_mbarrier_init_release_cluster() call"),
)

# Exact geometry (task section 4, "Exact geometry"). Matched against the
# string-masked view for the same decoy-resistance reason as
# REQUIRED_IDENTIFIER_PATTERNS.
GEOMETRY_CONSTANT_PATTERNS = (
    (re.compile(r"\bconstexpr\s+int\s+kThreadsPerCta\s*=\s*128\s*;"), "constexpr int kThreadsPerCta = 128;"),
    (re.compile(r"\bconstexpr\s+int\s+kClusterCtas\s*=\s*2\s*;"), "constexpr int kClusterCtas = 2;"),
    (re.compile(r"\bconstexpr\s+int\s+kGridBlocks\s*=\s*2\s*;"), "constexpr int kGridBlocks = 2;"),
    (re.compile(r"__launch_bounds__\s*\(\s*128\s*\)"), "__launch_bounds__(128)"),
)
HOST_LAUNCH_PATTERN = re.compile(r"spec\.kernel\s*<<<\s*kGridBlocks\s*,\s*kThreadsPerCta\s*[,>]")

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


def mask_string_and_char_literals(code_with_strings: str) -> str:
    """Return ``code_with_strings`` (already comment-stripped) with the
    CONTENTS of every string/character literal replaced by a neutral filler
    character, same length and quote positions preserved, so a required- or
    forbidden- C++ identifier/attribute pattern can never be satisfied by a
    decoy string literal (e.g. ``const char* x = "get_sreg_cluster_ctarank";``
    or ``"fence_mbarrier_init_release_cluster()"``) while genuine code
    identifiers, control flow, and asm(...) call syntax are left completely
    untouched. Positions are therefore identical between this view and
    ``code_with_strings`` -- callers that need brace/structure matching may
    use either interchangeably.

    This view must NOT be used for checks that legitimately need to see PTX
    text inside an asm string operand (e.g. the %clock64 reads or any
    tcgen05.* mnemonic): use find_asm_string_operand_spans/
    pattern_has_asm_evidence for those instead.
    """
    chars = list(code_with_strings)
    i = 0
    n = len(code_with_strings)
    while i < n:
        c = code_with_strings[i]
        if c in ("\"", "'"):
            quote = c
            j = i + 1
            while j < n:
                if code_with_strings[j] == "\\" and j + 1 < n:
                    chars[j] = "#"
                    chars[j + 1] = "#"
                    j += 2
                    continue
                if code_with_strings[j] == quote:
                    j += 1
                    break
                chars[j] = "#"
                j += 1
            i = j
            continue
        i += 1
    return "".join(chars)


ASM_CALL_PATTERN = re.compile(r"\basm(?:\s+volatile)?\s*\(")


def find_asm_string_operand_spans(code_with_strings: str) -> list[tuple[int, int]]:
    """Return the (start, end) character spans, within ``code_with_strings``
    (comment-stripped, strings preserved), of every string-literal PTX
    template operand of a real ``asm``/``asm volatile(...)`` statement --
    i.e. the text ptxas/nvcc actually sees, not the surrounding C++ call
    syntax. Adjacent (C++-concatenated) string literals immediately after
    ``asm(``/``asm volatile(`` are all included. An ``asm(`` not
    immediately followed by a string literal contributes no span.
    """
    spans: list[tuple[int, int]] = []
    n = len(code_with_strings)
    for m in ASM_CALL_PATTERN.finditer(code_with_strings):
        i = m.end()
        while i < n and code_with_strings[i] in " \t\r\n":
            i += 1
        while i < n and code_with_strings[i] == '"':
            start = i
            j = i + 1
            while j < n:
                if code_with_strings[j] == "\\" and j + 1 < n:
                    j += 2
                    continue
                if code_with_strings[j] == '"':
                    j += 1
                    break
                j += 1
            spans.append((start, j))
            i = j
            while i < n and code_with_strings[i] in " \t\r\n":
                i += 1
    return spans


def pattern_has_asm_evidence(
    pattern: re.Pattern, code_with_strings: str, asm_spans: list[tuple[int, int]]
) -> bool:
    """True only if ``pattern`` matches text lying entirely within one of
    ``asm_spans`` -- i.e. genuinely inside the string-literal PTX-template
    operand of a real asm/asm volatile(...) statement. A match inside a
    comment (already stripped upstream) or an ordinary, non-asm string
    literal does not count.
    """
    for match in pattern.finditer(code_with_strings):
        for start, end in asm_spans:
            if start <= match.start() and match.end() <= end:
                return True
    return False


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
FENCE_MBARRIER_INIT_DEFINITION = re.compile(
    r"\bvoid\s+fence_mbarrier_init_release_cluster\s*\(\s*\)\s*\{", re.DOTALL
)
TCGEN05_WAIT_LD_DEFINITION = re.compile(
    r"\bvoid\s+tcgen05_wait_ld\s*\(\s*\)\s*\{", re.DOTALL
)

# Per-phase CTA-pair handshake structure (task sections 3-4).
ITERATION_LOOP_HEADER = re.compile(
    r"\bfor\s*\(\s*int64_t\s+it\s*=\s*0\s*;\s*it\s*<\s*iterations\s*;\s*\+\+it\s*\)\s*\{",
    re.DOTALL,
)
LEADER_BLOCK_HEADER = re.compile(r"\bif\s*\(\s*is_leader\s*\)\s*\{", re.DOTALL)
RANK0_BLOCK_HEADER = re.compile(r"\bif\s*\(\s*cta_rank\s*==\s*0\s*\)\s*\{", re.DOTALL)
TID0_BLOCK_HEADER = re.compile(r"\bif\s*\(\s*tid\s*==\s*0\s*\)\s*\{", re.DOTALL)
FRAGMENT_LOOP_HEADER = re.compile(
    r"\bfor\s*\(\s*int\s+frag\s*=\s*0\s*;\s*frag\s*<\s*kFragments\s*;\s*\+\+frag\s*\)\s*\{",
    re.DOTALL,
)
WAIT_LOOP_HEADER = re.compile(
    r"\bwhile\s*\(\s*!\s*(?:cuda::ptx::)?mbarrier_try_wait_parity\s*"
    r"\([^;{}]*\)\s*\)\s*\{",
    re.DOTALL,
)
SYNCTHREADS_PATTERN = re.compile(r"__syncthreads\s*\(\s*\)\s*;")
CLUSTER_ARRIVE_PATTERN = re.compile(
    r"(?:\bcuda::ptx::)?\bbarrier_cluster_arrive\s*\(\s*\)\s*;"
)
CLUSTER_WAIT_PATTERN = re.compile(
    r"(?:\bcuda::ptx::)?\bbarrier_cluster_wait\s*\(\s*\)\s*;"
)
PARITY_ADVANCE_PATTERN = re.compile(r"\bparity\s*\^=\s*1u?\s*;")


def _span_contains(span: tuple[int, int], pos: int) -> bool:
    return span[0] <= pos < span[1]


def brace_depth_within(code_only: str, scope_start: int, position: int) -> int:
    """Return the braced nesting depth at ``position`` within a known body.

    Callers pass the comment-stripped, string-masked view, so braces in
    comments or literals cannot affect the result.  A required operation at
    depth zero is a direct statement of the selected body; a positive depth
    proves that an additional braced conditional/loop can gate it.
    """
    if not (0 <= scope_start <= position <= len(code_only)):
        raise SourceStructureError("internal source-check error: invalid brace-depth range")
    depth = 0
    for char in code_only[scope_start:position]:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                raise SourceStructureError("unexpected closing brace while computing scope depth")
    return depth


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
    """Prove the CTA-level operand/output partitioning used by the 2-SM MMA.

    A and D split the global M=256 rows into 128 rows per CTA.  CUTLASS's
    official SM100 2x1SM BF16 MMA traits independently encode B as
    ``(_2, (N/2, K))``: each CTA therefore stores N/2 local columns, with
    CTA rank selecting the corresponding global-column half.  The SMEM
    address remains local while the validation value uses the global column.
    TMEM addressing likewise remains CTA-local; only the global D write gets
    the rank-based row offset.
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

    n_local_defs = list(
        re.finditer(
            r"\bconstexpr\s+int\s+kNLocal\s*=\s*N\s*/\s*kClusterCtas\s*;",
            umma_body,
        )
    )
    if len(n_local_defs) != 1:
        errors.append(
            "B initialization must define exactly one "
            "'constexpr int kNLocal = N / kClusterCtas;'"
        )
    if not re.search(
        r"\bstatic_assert\s*\(\s*N\s*%\s*kClusterCtas\s*==\s*0\s*(?:,[^;]*)?\)\s*;",
        umma_body,
        re.DOTALL,
    ):
        errors.append("B initialization must statically require N % kClusterCtas == 0")

    b_loop_header = re.compile(
        r"\bfor\s*\(\s*int\s+idx\s*=\s*tid\s*;\s*idx\s*<\s*kNLocal\s*\*\s*kK\s*;\s*"
        r"idx\s*\+=\s*kThreadsPerCta\s*\)\s*\{",
        re.DOTALL,
    )
    try:
        b_loop_body, _, _ = extract_single_control_block(umma_body, b_loop_header, "B initialization loop")
    except SourceStructureError as exc:
        errors.append(
            "B initialization loop must cover exactly kNLocal * kK elements: "
            f"{exc}"
        )
    else:
        local_col = re.search(
            r"\bconst\s+int\s+local_col\s*=\s*idx\s*/\s*kK\s*;",
            b_loop_body,
        )
        k_value = re.search(r"\bconst\s+int\s+k\s*=\s*idx\s*%\s*kK\s*;", b_loop_body)
        global_col = re.search(
            r"\bconst\s+int\s+global_col\s*=\s*cta_rank\s*\*\s*kNLocal\s*\+\s*local_col\s*;",
            b_loop_body,
        )
        value = re.search(
            r"\bconst\s+int\s+value\s*=\s*\(\(\s*2\s*\*\s*k\s*\+\s*global_col\s*\)\s*"
            r"%\s*5\s*\)\s*-\s*2\s*;",
            b_loop_body,
            re.DOTALL,
        )
        store = re.search(
            r"\bB\s*\[\s*smem_core_tile_index\s*\(\s*local_col\s*/\s*8\s*,\s*"
            r"local_col\s*%\s*8\s*,\s*k\s*\)\s*\]\s*=",
            b_loop_body,
            re.DOTALL,
        )

        if local_col is None:
            errors.append("B initialization must derive local_col = idx / kK")
        if k_value is None:
            errors.append("B initialization must derive k = idx % kK")
        if global_col is None:
            errors.append(
                "B initialization must map each local column with "
                "global_col = cta_rank * kNLocal + local_col"
            )
        if value is None:
            errors.append("B initialization value must use global_col, not a replicated local column")
        if store is None:
            errors.append(
                "B initialization must store the global-column value at the local_col SMEM position"
            )
        if all(match is not None for match in (local_col, k_value, global_col, value, store)):
            positions = [
                local_col.start(),
                k_value.start(),
                global_col.start(),
                value.start(),
                store.start(),
            ]
            if positions != sorted(positions):
                errors.append(
                    "B initialization must order local_col, k, global_col, value, then the local SMEM store"
                )
        if len(re.findall(r"\bB\s*\[", b_loop_body)) != 1:
            errors.append("B initialization loop must contain exactly one B SMEM store")

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


def check_exact_geometry(masked: str) -> list[str]:
    """(task section 4, "Exact geometry") Prove kGridBlocks=2,
    kThreadsPerCta=128, kClusterCtas=2, and __launch_bounds__(128) as real
    constants/attributes, and that the host launch (run_once) actually uses
    kGridBlocks/kThreadsPerCta as its <<<grid, block>>> launch configuration
    (grid=(2,1,1), block=(128,1,1)). __cluster_dims__(2, 1, 1) (the
    compile-time cluster=(2,1,1) declaration) is separately required by
    REQUIRED_IDENTIFIER_PATTERNS and the device-side launch guard's
    dependence on cluster_nctarank/cluster_ctarank is separately proved by
    check_launch_guard_ordering. Matched against the string-masked view so a
    decoy string cannot satisfy any of these.
    """
    errors: list[str] = []
    for pattern, description in GEOMETRY_CONSTANT_PATTERNS:
        if not pattern.search(masked):
            errors.append(f"source is missing required geometry constant: {description}")
    if not HOST_LAUNCH_PATTERN.search(masked):
        errors.append(
            "the host launch (run_once) must launch spec.kernel<<<kGridBlocks, "
            "kThreadsPerCta, ...>>> -- the exact grid=(2,1,1)/block=(128,1,1) launch contract"
        )
    return errors


def check_dynamic_smem_allocation(masked: str) -> list[str]:
    """Prove the host launch reserves exactly the CTA-local A and B slices.

    The repaired 2-SM operand mapping stores ``N / kClusterCtas`` B columns
    per CTA.  Checking only the device fill loop is insufficient: a host-side
    regression can still under-allocate dynamic shared memory while leaving
    the kernel mapping text intact.  Require the exact local-N derivation,
    the exact A-plus-local-B byte expression, and that this byte count is the
    dynamic-SMEM argument of the real kernel launch, in that order.
    """
    errors: list[str] = []
    try:
        run_once_body, _, _ = extract_single_function_body(
            masked, RUN_ONCE_DEFINITION, "run_once"
        )
    except SourceStructureError as exc:
        return [f"invalid dynamic shared-memory allocation check: {exc}"]

    n_local_pattern = re.compile(
        r"\bconst\s+int\s+n_local\s*=\s*spec\.n\s*/\s*kClusterCtas\s*;"
    )
    smem_bytes_pattern = re.compile(
        r"\bconst\s+int\s+smem_bytes\s*=\s*kMLocal\s*\*\s*kK\s*\*\s*2\s*\+\s*"
        r"n_local\s*\*\s*kK\s*\*\s*2\s*;",
        re.DOTALL,
    )
    launch_pattern = re.compile(
        r"\bspec\.kernel\s*<<<\s*kGridBlocks\s*,\s*kThreadsPerCta\s*,\s*"
        r"static_cast\s*<\s*size_t\s*>\s*\(\s*smem_bytes\s*\)\s*>>>",
        re.DOTALL,
    )

    n_local_matches = list(n_local_pattern.finditer(run_once_body))
    smem_bytes_matches = list(smem_bytes_pattern.finditer(run_once_body))
    launch_matches = list(launch_pattern.finditer(run_once_body))

    if len(n_local_matches) != 1:
        errors.append(
            "run_once must derive exactly one 'const int n_local = "
            "spec.n / kClusterCtas;' for the CTA-local B extent"
        )
    if len(smem_bytes_matches) != 1:
        errors.append(
            "run_once must reserve exactly kMLocal * kK * 2 + "
            "n_local * kK * 2 dynamic shared-memory bytes"
        )
    if len(launch_matches) != 1:
        errors.append(
            "run_once must pass static_cast<size_t>(smem_bytes) as the "
            "dynamic shared-memory argument of spec.kernel"
        )

    if len(n_local_matches) == len(smem_bytes_matches) == len(launch_matches) == 1:
        positions = (
            n_local_matches[0].start(),
            smem_bytes_matches[0].start(),
            launch_matches[0].start(),
        )
        if positions != tuple(sorted(positions)):
            errors.append(
                "run_once must derive n_local, compute smem_bytes, then launch the kernel"
            )
    return errors


def check_mbarrier_init_fence(masked: str) -> list[str]:
    """(task sections 2 and 4, "Fence ordering") Prove:
      * a single, real fence_mbarrier_init_release_cluster() helper is
        defined, and its own body genuinely calls
        cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release,
        cuda::ptx::scope_cluster) -- the official CUDA 13.1 wrapper that
        lowers unconditionally to "fence.mbarrier_init.release.cluster;" on
        sm_103a (see that helper's own comment in umma_2sm.cu);
      * umma_2sm_body calls this helper exactly once;
      * that call is program-ordered after mbarrier_init(&mbar, ...) and
        before the FIRST cluster barrier (the one that publishes CTA-local
        initialization to the pair, i.e. step 5's barrier_cluster_arrive());
      * a separate, real fence_proxy_async call is also present (the
        independently required async-proxy fence).
    Matched against the string-masked view so neither a comment nor a decoy
    string can satisfy any part of this.
    """
    errors: list[str] = []
    try:
        fence_body, _, _ = extract_single_function_body(
            masked, FENCE_MBARRIER_INIT_DEFINITION, "fence_mbarrier_init_release_cluster"
        )
    except SourceStructureError as exc:
        errors.append(f"invalid mbarrier-init fence helper: {exc}")
        return errors

    if not re.search(
        r"fence_mbarrier_init\s*\(\s*(?:cuda::ptx::)?sem_release\s*,\s*(?:cuda::ptx::)?scope_cluster\s*\)",
        fence_body,
    ):
        errors.append(
            "fence_mbarrier_init_release_cluster() must call cuda::ptx::fence_mbarrier_init"
            "(cuda::ptx::sem_release, cuda::ptx::scope_cluster) (the official wrapper that "
            "lowers to fence.mbarrier_init.release.cluster)"
        )

    try:
        umma_body, _, _ = extract_single_function_body(masked, UMMA_BODY_DEFINITION, "umma_2sm_body")
    except SourceStructureError as exc:
        errors.append(f"invalid mbarrier-init fence ordering check: {exc}")
        return errors

    fence_calls = list(
        re.finditer(r"\bfence_mbarrier_init_release_cluster\s*\(\s*\)\s*;", umma_body)
    )
    if len(fence_calls) != 1:
        errors.append(
            "umma_2sm_body must call fence_mbarrier_init_release_cluster() exactly once, "
            f"found {len(fence_calls)}"
        )
        return errors

    mbarrier_init_calls = list(
        re.finditer(r"\bmbarrier_init\s*\(\s*&mbar\b[^;]*\)\s*;", umma_body)
    )
    if len(mbarrier_init_calls) != 1:
        errors.append(
            "umma_2sm_body must call mbarrier_init(&mbar, ...) exactly once, "
            f"found {len(mbarrier_init_calls)}"
        )
        return errors
    mbarrier_init_match = mbarrier_init_calls[0]

    proxy_calls = list(
        re.finditer(r"(?:\bcuda::ptx::)?\bfence_proxy_async\s*\([^;]*\)\s*;", umma_body)
    )
    if len(proxy_calls) != 1:
        errors.append(
            "umma_2sm_body must separately call fence_proxy_async(...) exactly once "
            f"(the async-proxy fence), found {len(proxy_calls)}"
        )
        return errors
    proxy_match = proxy_calls[0]

    arrive_match = CLUSTER_ARRIVE_PATTERN.search(umma_body)
    if not arrive_match:
        errors.append("umma_2sm_body must call barrier_cluster_arrive() at least once")
        return errors

    fence_match = fence_calls[0]
    if not (
        mbarrier_init_match.start()
        < fence_match.start()
        < proxy_match.start()
        < arrive_match.start()
    ):
        errors.append(
            "fence_mbarrier_init_release_cluster() must be called after mbarrier_init(&mbar, ...) "
            "and before the separate fence_proxy_async(...) and first cluster barrier that publish "
            "CTA-local initialization to the pair"
        )

    # Presence and ordering are insufficient: ``if (false) { fence(); }``
    # is textually ordered but unreachable.  Identify the unique tid==0
    # initialization block containing mbarrier_init and require init, both
    # fences, and no intervening executable syntax to form one direct
    # statement sequence in that block.
    init_blocks: list[tuple[int, int]] = []
    for header in TID0_BLOCK_HEADER.finditer(umma_body):
        try:
            close = find_matching_brace(umma_body, header.end() - 1)
        except SourceStructureError:
            continue
        span = (header.end(), close)
        if _span_contains(span, mbarrier_init_match.start()):
            init_blocks.append(span)
    if len(init_blocks) != 1:
        errors.append(
            "mbarrier_init(&mbar, ...), fence_mbarrier_init_release_cluster(), and "
            "fence_proxy_async(...) must share one 'if (tid == 0)' initialization block"
        )
        return errors

    init_span = init_blocks[0]
    required_in_init = (
        (mbarrier_init_match, "mbarrier_init(&mbar, ...)"),
        (fence_match, "fence_mbarrier_init_release_cluster()"),
        (proxy_match, "fence_proxy_async(...)"),
    )
    for match, label in required_in_init:
        if not _span_contains(init_span, match.start()):
            errors.append(
                f"{label} must be a direct, unconditionally reachable statement in the "
                "same 'if (tid == 0)' initialization block"
            )
            continue
        try:
            depth = brace_depth_within(umma_body, init_span[0], match.start())
        except SourceStructureError as exc:
            errors.append(f"cannot validate mbarrier initialization reachability: {exc}")
            continue
        if depth != 0:
            errors.append(
                f"{label} must be a direct, unconditionally reachable statement in the "
                "'if (tid == 0)' initialization block; an additional braced condition "
                "must not gate it"
            )

    if umma_body[mbarrier_init_match.end():fence_match.start()].strip() or \
       umma_body[fence_match.end():proxy_match.start()].strip():
        errors.append(
            "mbarrier_init(&mbar, ...), fence_mbarrier_init_release_cluster(), and "
            "fence_proxy_async(...) must be one direct statement sequence with no "
            "intervening conditional or executable statement"
        )

    return errors


def check_iteration_structure(masked: str) -> list[str]:
    """(task sections 3-4: per-phase CTA-pair handshake; MMA/commit
    dominance) All checks anchored on the single runtime outer iteration
    loop in umma_2sm_body, working in that function's own absolute text
    coordinates (regex ``pos``/``endpos``, never re-slicing) so nested spans
    can be compared directly:

      * the loop is executed uniformly by the whole cluster (never nested
        inside is_leader or cta_rank == 0);
      * issue_one_umma_2sm/commit_umma_2sm_multicast each appear EXACTLY
        ONCE in the whole function, both confined to a single
        'if (cta_rank == 0) { ... }' block nested inside the loop's single
        'if (is_leader) { ... }' block, with the exact literal mask 0x0003;
      * the mbarrier wait is present inside that is_leader block but NOT
        nested inside any cta_rank == 0 block;
      * neither __syncthreads() nor the cluster arrive/wait may appear
        inside the is_leader block;
      * __syncthreads(), then barrier_cluster_arrive(), then
        barrier_cluster_wait(), appear (in that order) inside the loop body,
        after the is_leader block;
      * the TMEM readback loop, after the iteration loop, is not re-confined
        to a cta_rank == 0 block.
    Matched against the string-masked view for decoy resistance.
    """
    errors: list[str] = []
    try:
        umma_body, _, _ = extract_single_function_body(masked, UMMA_BODY_DEFINITION, "umma_2sm_body")
    except SourceStructureError as exc:
        return [f"invalid iteration-structure check: {exc}"]

    loop_headers = list(ITERATION_LOOP_HEADER.finditer(umma_body))
    if len(loop_headers) != 1:
        errors.append(
            "umma_2sm_body must contain exactly one runtime outer iteration loop "
            f"('for (int64_t it = 0; it < iterations; ++it)'), found {len(loop_headers)}"
        )
        return errors
    loop_open = loop_headers[0].end() - 1
    try:
        loop_close = find_matching_brace(umma_body, loop_open)
    except SourceStructureError as exc:
        errors.append(f"cannot validate the runtime outer iteration loop body: {exc}")
        return errors
    loop_body_start, loop_body_end = loop_open + 1, loop_close

    for header_pat, label in ((LEADER_BLOCK_HEADER, "is_leader"), (RANK0_BLOCK_HEADER, "cta_rank == 0")):
        for match in header_pat.finditer(umma_body, 0, loop_open):
            try:
                close = find_matching_brace(umma_body, match.end() - 1)
            except SourceStructureError:
                continue
            if close > loop_open:
                errors.append(
                    "the runtime outer iteration loop must not be nested inside an "
                    f"'{label}' conditional; it must be executed uniformly by the whole cluster"
                )

    leader_headers = list(LEADER_BLOCK_HEADER.finditer(umma_body, loop_body_start, loop_body_end))
    if len(leader_headers) != 1:
        errors.append(
            "the runtime outer iteration loop must contain exactly one "
            f"'if (is_leader) {{ ... }}' block, found {len(leader_headers)}"
        )
        return errors
    leader_open = leader_headers[0].end() - 1
    try:
        leader_close = find_matching_brace(umma_body, leader_open)
    except SourceStructureError as exc:
        errors.append(f"cannot validate the per-iteration leader block: {exc}")
        return errors
    leader_span = (leader_open + 1, leader_close)

    # ---- MMA/commit dominance (task section 4; mutations 6-7) ------------
    # issue_one_umma_2sm legitimately appears at TWO call sites in the
    # canonical source (the enable-input-d=false UMMA 0, then the
    # #pragma-unrolled UMMA 1..DEPTH-1 loop), both inside the same
    # rank-0-nested-in-leader block; commit_umma_2sm_multicast appears at
    # exactly one. The dominance requirement is therefore "every call site
    # (whatever its count) lies inside that single block", not "exactly
    # one call site" -- an extra call ANYWHERE outside that block is what
    # must be rejected (mutations 6-7).
    all_issue = list(re.finditer(r"\bissue_one_umma_2sm\s*\(", umma_body))
    all_commit = list(re.finditer(r"\bcommit_umma_2sm_multicast\s*\(", umma_body))
    if not all_issue:
        errors.append("umma_2sm_body must contain at least one issue_one_umma_2sm(...) call site")
    if len(all_commit) != 1:
        errors.append(
            f"umma_2sm_body must contain exactly one commit_umma_2sm_multicast(...) call "
            f"site, found {len(all_commit)}"
        )

    rank0_span: tuple[int, int] | None = None
    mask_ok = False
    for match in RANK0_BLOCK_HEADER.finditer(umma_body, leader_span[0], leader_span[1]):
        try:
            close = find_matching_brace(umma_body, match.end() - 1)
        except SourceStructureError:
            continue
        span = (match.end(), close)
        if any(_span_contains(span, m.start()) for m in all_issue) and \
           any(_span_contains(span, m.start()) for m in all_commit):
            rank0_span = span
            if re.search(r"commit_umma_2sm_multicast\s*\([^;]*0x0003u?\b", umma_body[span[0]:span[1]], re.DOTALL):
                mask_ok = True
            break
    if rank0_span is None:
        errors.append(
            "issue_one_umma_2sm and commit_umma_2sm_multicast must both be issued from a "
            "single 'if (cta_rank == 0) { ... }' block nested inside the per-iteration "
            "'if (is_leader)' block"
        )
    else:
        if not mask_ok:
            errors.append("commit_umma_2sm_multicast must be called with the exact literal CTA mask 0x0003")
        if any(not _span_contains(rank0_span, m.start()) for m in all_issue):
            errors.append(
                "issue_one_umma_2sm must be confined to the rank-0-nested-in-leader block; "
                "no additional issue call site is permitted"
            )
        if any(not _span_contains(rank0_span, m.start()) for m in all_commit):
            errors.append(
                "commit_umma_2sm_multicast must be confined to the rank-0-nested-in-leader "
                "block; no additional commit call site is permitted"
            )

    # ---- Wait must remain reachable in both CTA ranks (mutation 14) ------
    wait_loop_close: int | None = None
    wait_matches = list(re.finditer(r"\bmbarrier_try_wait_parity\s*\(", umma_body))
    if len(wait_matches) != 1:
        errors.append(
            "umma_2sm_body must contain exactly one mbarrier_try_wait_parity(...) call "
            f"site, found {len(wait_matches)}"
        )
    leader_waits = [m for m in wait_matches if _span_contains(leader_span, m.start())]
    if not leader_waits:
        errors.append(
            "the mbarrier completion wait must be present inside the per-iteration "
            "'if (is_leader)' block"
        )
    else:
        wait_match = leader_waits[0]
        for rank0_header in RANK0_BLOCK_HEADER.finditer(
            umma_body, leader_span[0], leader_span[1]
        ):
            try:
                rank0_close = find_matching_brace(umma_body, rank0_header.end() - 1)
            except SourceStructureError:
                continue
            if _span_contains((rank0_header.end(), rank0_close), wait_match.start()):
                errors.append(
                    "the mbarrier completion wait must not be enclosed in a cta_rank == 0 "
                    "condition (CTA rank 1's leader must wait too)"
                )
                break
        try:
            wait_depth = brace_depth_within(umma_body, leader_span[0], wait_match.start())
        except SourceStructureError as exc:
            errors.append(f"cannot validate mbarrier-wait reachability: {exc}")
        else:
            if wait_depth != 0:
                errors.append(
                    "the mbarrier completion wait must be directly reachable by both CTA "
                    "leaders; no additional braced conditional may enclose it"
                )

        wait_loop_headers = list(
            WAIT_LOOP_HEADER.finditer(umma_body, leader_span[0], leader_span[1])
        )
        if len(wait_loop_headers) != 1:
            errors.append(
                "the leader block must contain exactly one braced while-loop around "
                f"mbarrier_try_wait_parity(...), found {len(wait_loop_headers)}"
            )
        else:
            wait_loop_header = wait_loop_headers[0]
            try:
                wait_loop_close = find_matching_brace(
                    umma_body, wait_loop_header.end() - 1
                )
            except SourceStructureError as exc:
                errors.append(f"cannot validate the mbarrier wait loop: {exc}")
            if rank0_span is not None and \
               umma_body[rank0_span[1] + 1:wait_loop_header.start()].strip():
                errors.append(
                    "the both-CTA mbarrier wait loop must directly follow the rank-0 issue "
                    "block; no additional conditional or executable statement may gate it"
                )

    # ``mbarrier_try_wait_parity`` only follows successive primary phases
    # when the expected parity advances once after every successful wait.
    # Require the advance as a direct statement of the same leader block,
    # after the wait, so a decoy or conditionally executed update cannot
    # satisfy the gate.
    parity_advances = list(PARITY_ADVANCE_PATTERN.finditer(umma_body))
    if len(parity_advances) != 1:
        errors.append(
            "umma_2sm_body must advance mbarrier parity exactly once per iteration with "
            f"'parity ^= 1u;' after the successful wait, found {len(parity_advances)}"
        )
    else:
        parity_match = parity_advances[0]
        if not _span_contains(leader_span, parity_match.start()):
            errors.append(
                "'parity ^= 1u;' must be inside the per-iteration 'if (is_leader)' block"
            )
        else:
            try:
                parity_depth = brace_depth_within(
                    umma_body, leader_span[0], parity_match.start()
                )
            except SourceStructureError as exc:
                errors.append(f"cannot validate mbarrier parity reachability: {exc}")
            else:
                if parity_depth != 0:
                    errors.append(
                        "'parity ^= 1u;' must be a direct, unconditionally reachable statement "
                        "of the leader block after the successful wait"
                    )
        if leader_waits and parity_match.start() <= leader_waits[0].start():
            errors.append("'parity ^= 1u;' must occur after mbarrier_try_wait_parity succeeds")
        if wait_loop_close is not None and \
           umma_body[wait_loop_close + 1:parity_match.start()].strip():
            errors.append(
                "'parity ^= 1u;' must directly follow the successful mbarrier wait loop; "
                "no additional conditional or executable statement may gate it"
            )

    # ---- Neither may live inside is_leader (mutation 13) -----------------
    for pattern, label in (
        (SYNCTHREADS_PATTERN, "__syncthreads()"),
        (CLUSTER_ARRIVE_PATTERN, "barrier_cluster_arrive()"),
        (CLUSTER_WAIT_PATTERN, "barrier_cluster_wait()"),
    ):
        if pattern.search(umma_body, leader_span[0], leader_span[1]):
            errors.append(
                f"{label} must not be issued from inside the per-iteration 'if (is_leader)' "
                "block; it must be reachable by every thread"
            )

    # ---- CTA sync, then full cluster rendezvous, in order, inside the ----
    # ---- loop body, after the leader block (mutations 9, 10, 11, 12). ----
    sync_match = SYNCTHREADS_PATTERN.search(umma_body, leader_close + 1, loop_body_end)
    if not sync_match:
        errors.append(
            "the runtime outer iteration loop must contain a __syncthreads() call, after the "
            "per-iteration leader block, inside every outer iteration -- publishing the "
            "successful local wait to the whole CTA"
        )
        return errors
    if umma_body[leader_close + 1:sync_match.start()].strip():
        errors.append(
            "the post-wait __syncthreads() must directly follow the leader block; no "
            "additional conditional or executable statement may gate it"
        )
    try:
        sync_depth = brace_depth_within(umma_body, loop_body_start, sync_match.start())
    except SourceStructureError as exc:
        errors.append(f"cannot validate post-wait CTA synchronization reachability: {exc}")
    else:
        if sync_depth != 0:
            errors.append(
                "the post-wait __syncthreads() must be a direct statement of the runtime "
                "outer iteration loop, reachable by every thread in both CTA ranks"
            )
    arrive_match = CLUSTER_ARRIVE_PATTERN.search(umma_body, sync_match.end(), loop_body_end)
    if not arrive_match:
        errors.append(
            "the runtime outer iteration loop must contain a barrier_cluster_arrive() call, "
            "after the post-wait __syncthreads(), inside every outer iteration"
        )
        return errors
    if umma_body[sync_match.end():arrive_match.start()].strip():
        errors.append(
            "barrier_cluster_arrive() must directly follow the post-wait __syncthreads(); "
            "no additional conditional or executable statement may gate it"
        )
    try:
        arrive_depth = brace_depth_within(umma_body, loop_body_start, arrive_match.start())
    except SourceStructureError as exc:
        errors.append(f"cannot validate per-phase cluster-arrive reachability: {exc}")
    else:
        if arrive_depth != 0:
            errors.append(
                "the per-phase barrier_cluster_arrive() must be a direct statement of the "
                "runtime outer iteration loop, reachable by every thread in both CTA ranks"
            )
    cluster_wait_match = CLUSTER_WAIT_PATTERN.search(umma_body, arrive_match.end(), loop_body_end)
    if not cluster_wait_match:
        errors.append(
            "the runtime outer iteration loop must contain a barrier_cluster_wait() call, "
            "after barrier_cluster_arrive(), inside every outer iteration"
        )
        return errors
    if umma_body[arrive_match.end():cluster_wait_match.start()].strip():
        errors.append(
            "barrier_cluster_wait() must directly follow barrier_cluster_arrive(); no "
            "additional conditional or executable statement may gate it"
        )
    try:
        cluster_wait_depth = brace_depth_within(
            umma_body, loop_body_start, cluster_wait_match.start()
        )
    except SourceStructureError as exc:
        errors.append(f"cannot validate per-phase cluster-wait reachability: {exc}")
    else:
        if cluster_wait_depth != 0:
            errors.append(
                "the per-phase barrier_cluster_wait() must be a direct statement of the "
                "runtime outer iteration loop, reachable by every thread in both CTA ranks"
            )

    # ---- Readback, after the loop, must remain reachable by both ranks. --
    after_loop_start = loop_body_end + 1
    readback_match = re.search(r"\btcgen05_ld_32x32b_x32\s*\(", umma_body[after_loop_start:])
    if not readback_match:
        errors.append("the TMEM readback loop must be present after the runtime outer iteration loop")
    else:
        readback_pos = after_loop_start + readback_match.start()
        for match in RANK0_BLOCK_HEADER.finditer(umma_body, after_loop_start):
            try:
                close = find_matching_brace(umma_body, match.end() - 1)
            except SourceStructureError:
                continue
            if _span_contains((match.end(), close), readback_pos):
                errors.append("the TMEM readback loop must not be enclosed in a cta_rank == 0 condition")
                break

    return errors


def check_tmem_load_completion(code_only: str, masked: str) -> list[str]:
    """Prove every compile-time TMEM fragment performs load -> wait -> use.

    The PTX mnemonic alone is insufficient because an unused helper still
    compiles as source evidence while the live readback path can omit its
    call.  Require the mnemonic in the unique helper's real asm operand and
    require exactly one live helper call, directly after the single load in
    the canonical ``N/32`` fragment loop and before the first global-output
    use of the loaded registers.
    """
    errors: list[str] = []
    wait_ptx_pattern = re.compile(r"tcgen05\.wait::ld\.sync\.aligned")

    try:
        wait_helper_body, _, _ = extract_single_function_body(
            code_only, TCGEN05_WAIT_LD_DEFINITION, "tcgen05_wait_ld"
        )
    except SourceStructureError as exc:
        return [f"invalid tcgen05.wait::ld helper: {exc}"]
    helper_asm_spans = find_asm_string_operand_spans(wait_helper_body)
    if not pattern_has_asm_evidence(wait_ptx_pattern, wait_helper_body, helper_asm_spans):
        errors.append(
            "tcgen05_wait_ld() must contain tcgen05.wait::ld.sync.aligned as real asm evidence"
        )

    try:
        umma_body, _, _ = extract_single_function_body(
            masked, UMMA_BODY_DEFINITION, "umma_2sm_body"
        )
    except SourceStructureError as exc:
        errors.append(f"invalid TMEM-load completion check: {exc}")
        return errors

    fragment_count = re.search(
        r"\bconstexpr\s+int\s+kFragments\s*=\s*N\s*/\s*32\s*;", umma_body
    )
    if not fragment_count:
        errors.append("TMEM readback must derive exactly kFragments = N / 32")

    fragment_headers = list(FRAGMENT_LOOP_HEADER.finditer(umma_body))
    if len(fragment_headers) != 1:
        errors.append(
            "umma_2sm_body must contain exactly one 'for (int frag = 0; "
            f"frag < kFragments; ++frag)' TMEM readback loop, found {len(fragment_headers)}"
        )
        return errors
    fragment_open = fragment_headers[0].end() - 1
    try:
        fragment_close = find_matching_brace(umma_body, fragment_open)
    except SourceStructureError as exc:
        errors.append(f"cannot validate the TMEM fragment loop: {exc}")
        return errors
    fragment_span = (fragment_open + 1, fragment_close)

    all_loads = list(re.finditer(r"\btcgen05_ld_32x32b_x32\s*\([^;]*\)\s*;", umma_body))
    all_wait_calls = list(re.finditer(r"\btcgen05_wait_ld\s*\(\s*\)\s*;", umma_body))
    if len(all_loads) != 1:
        errors.append(
            "umma_2sm_body must contain exactly one tcgen05_ld_32x32b_x32(...) call "
            f"site in the N/32 fragment loop, found {len(all_loads)}"
        )
    if len(all_wait_calls) != 1:
        errors.append(
            "umma_2sm_body must call tcgen05_wait_ld() exactly once after the fragment "
            f"load, found {len(all_wait_calls)}"
        )
    if len(all_loads) != 1 or len(all_wait_calls) != 1:
        return errors

    load_match = all_loads[0]
    wait_match = all_wait_calls[0]
    if not _span_contains(fragment_span, load_match.start()):
        errors.append("tcgen05_ld_32x32b_x32(...) must execute inside the N/32 fragment loop")
    if not _span_contains(fragment_span, wait_match.start()):
        errors.append("tcgen05_wait_ld() must execute inside the N/32 fragment loop")
    if wait_match.start() <= load_match.end():
        errors.append("tcgen05_wait_ld() must execute after tcgen05_ld_32x32b_x32(...)")
    elif umma_body[load_match.end():wait_match.start()].strip():
        errors.append(
            "tcgen05_wait_ld() must directly follow tcgen05_ld_32x32b_x32(...); no "
            "additional conditional or executable statement may gate it"
        )

    for match, label in (
        (load_match, "tcgen05_ld_32x32b_x32(...)"),
        (wait_match, "tcgen05_wait_ld()"),
    ):
        try:
            depth = brace_depth_within(umma_body, fragment_span[0], match.start())
        except SourceStructureError as exc:
            errors.append(f"cannot validate {label} reachability: {exc}")
        else:
            if depth != 0:
                errors.append(
                    f"{label} must be a direct, unconditionally reachable statement of the "
                    "N/32 fragment loop"
                )

    first_output_use = re.compile(r"\bg_d_out\s*\[").search(
        umma_body, load_match.end(), fragment_span[1]
    )
    if not first_output_use:
        errors.append("the N/32 fragment loop must write the loaded registers to g_d_out")
    elif wait_match.start() >= first_output_use.start():
        errors.append("tcgen05_wait_ld() must execute before the loaded registers are written")

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
        r"\bif\s*\(\s*is_leader\s*&&\s*cta_rank\s*==\s*0\s*&&\s*timing_mode\s*==\s*TimingMode::kTimed\s*\)\s*\{",
        re.DOTALL,
    )
    timed_guard_matches = list(timed_guard_pattern.finditer(umma_body))
    guard_scopes: list[tuple[int, int, str]] = []
    if len(timed_guard_matches) != 2:
        errors.append(
            f"umma_2sm_body contains {len(timed_guard_matches)} exact "
            "'is_leader && cta_rank == 0 && timing_mode == TimingMode::kTimed' guard(s); "
            "expected exactly two"
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
                "a %clock64 read is outside an exact 'is_leader && cta_rank == 0 && "
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
    PTX text (as genuine asm evidence), exact geometry, the
    mbarrier-initialization fence and its ordering, per-rank mapping,
    collective TMEM lifecycle, the per-phase CTA-pair handshake and
    MMA/commit dominance, cluster-sync-before-dealloc, and timing-mode
    routing. Fails closed (non-empty list) if the lexical scan itself
    cannot be trusted.

    Two decoy-resistant views are derived from the comment-stripped,
    literal-preserving ``code_only`` text (see module docstring):
    ``masked`` blanks out every string/char literal's CONTENTS (so a
    required/forbidden C++ identifier or attribute can never be satisfied by
    a decoy string) and ``asm_spans`` locates the genuine string-literal
    operand of every real asm/asm volatile(...) statement (so a required PTX
    mnemonic can never be satisfied by a comment or an ordinary, non-asm
    string literal). Checks that legitimately need to see PTX text inside an
    asm operand (e.g. %clock64) still use ``code_only`` directly.
    """
    try:
        code_only = strip_comments_preserving_literals(source_text)
    except SourceScanError as exc:
        return [f"cannot safely scan source: {exc}"]

    masked = mask_string_and_char_literals(code_only)
    asm_spans = find_asm_string_operand_spans(code_only)

    errors: list[str] = []
    for pattern, description in FORBIDDEN_SOURCE_PATTERNS:
        if pattern.search(code_only):
            errors.append(f"source contains forbidden pattern: {description}")
    for pattern, description in REQUIRED_ASM_EVIDENCE_PATTERNS:
        if not pattern_has_asm_evidence(pattern, code_only, asm_spans):
            errors.append(
                "source is missing required text as real asm evidence (a comment or an "
                f"ordinary, non-asm string literal does not count): {description}"
            )
    for pattern, description in REQUIRED_IDENTIFIER_PATTERNS:
        if not pattern.search(masked):
            errors.append(f"source is missing required text: {description}")
    errors.extend(check_exact_geometry(masked))
    errors.extend(check_dynamic_smem_allocation(masked))
    errors.extend(check_launch_guard_ordering(code_only))
    errors.extend(check_rank_mapping(masked))
    errors.extend(check_collective_tmem_lifecycle(code_only))
    errors.extend(check_mbarrier_init_fence(masked))
    errors.extend(check_iteration_structure(masked))
    errors.extend(check_tmem_load_completion(code_only, masked))
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
        # ---- PTX-mnemonic text: wrapped in a real asm volatile("...") by
        # ---- the template below, so it only counts as evidence when it is
        # ---- genuinely an asm operand (never a bare/comment/string decoy).
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
        # ---- geometry / identifiers: rendered as bare (real) code. --------
        "geometry_consts": (
            "constexpr int kThreadsPerCta = 128;\n"
            "constexpr int kClusterCtas = 2;\n"
            "constexpr int kGridBlocks = 2;\n"
        ),
        "cluster_dims_text": "__global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128) void k() {}",
        "ctarank_text": "cuda::ptx::get_sreg_cluster_ctarank();",
        "nctarank_text": "cuda::ptx::get_sreg_cluster_nctarank();",
        "mask_text": "0x0003u",
        "extra_forbidden_line": "",
        "fence_helper_def": (
            "void fence_mbarrier_init_release_cluster() { "
            "cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release, cuda::ptx::scope_cluster); }"
        ),
        "fence_helper_call": "fence_mbarrier_init_release_cluster();",
        "mbarrier_init_call": "cuda::ptx::mbarrier_init(&mbar, 1u);",
        "fence_proxy_call": "cuda::ptx::fence_proxy_async(cuda::ptx::space_cluster);",
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
        "b_local_extent": (
            "constexpr int kNLocal = N / kClusterCtas; "
            "static_assert(N % kClusterCtas == 0, \"N must divide evenly\");"
        ),
        "a_loop": (
            "for (int idx = tid; idx < kMLocal * kK; idx += kThreadsPerCta) { "
            "const int local_row = idx / kK; const int k = idx % kK; "
            "const int global_row = cta_rank * kMLocal + local_row; "
            "A[smem_core_tile_index(local_row/8, local_row%8, k)] = __float2bfloat16(1.0f); }"
        ),
        "b_loop": (
            "for (int idx = tid; idx < kNLocal * kK; idx += kThreadsPerCta) { "
            "const int local_col = idx / kK; const int k = idx % kK; "
            "const int global_col = cta_rank * kNLocal + local_col; "
            "const int value = ((2 * k + global_col) % 5) - 2; "
            "B[smem_core_tile_index(local_col/8, local_col%8, k)] = "
            "__float2bfloat16(static_cast<float>(value)); }"
        ),
        "alloc_call": "if (warp_id == 0) { tcgen05_alloc_2sm(x, N); }",
        "dealloc_call": "if (warp_id == 0) { tcgen05_dealloc_2sm(tmem_d, N); tcgen05_relinquish_alloc_permit_2sm(); }",
        # ---- the per-iteration handshake structure. ------------------
        "loop_open": "for (int64_t it = 0; it < iterations; ++it) {",
        "loop_close": "}",
        "leader_wrapper_open": "if (is_leader) {",
        "leader_wrapper_close": "}",
        "rank0_wrapper_open": "if (cta_rank == 0) {",
        "rank0_wrapper_close": "}",
        "issue_call": (
            "issue_one_umma_2sm(tmem_d, a_desc, b_desc, idesc, 0); "
            "commit_umma_2sm_multicast(mbar_addr, 0x0003u);"
        ),
        "extra_issue_call": "",
        "extra_commit_call": "",
        "wait_call": "while (!cuda::ptx::mbarrier_try_wait_parity(&mbar, parity)) {}",
        "parity_advance": "parity ^= 1u;",
        "leader_extra": "",
        "post_wait_syncthreads": "__syncthreads();",
        "loop_cluster_arrive": "cuda::ptx::barrier_cluster_arrive();",
        "loop_cluster_wait": "cuda::ptx::barrier_cluster_wait();",
        "after_loop_extra": "",
        "cluster_sync_before_dealloc": "cuda::ptx::barrier_cluster_arrive(); cuda::ptx::barrier_cluster_wait();",
        "readback_global_row": (
            "const int global_row = cta_rank * kMLocal + local_row;"
        ),
        "readback_load_call": (
            "tcgen05_ld_32x32b_x32(make_tmem_load_address(tmem_d, warp_id, frag), regs);"
        ),
        "readback_wait_call": "tcgen05_wait_ld();",
        "readback_store_call": (
            "g_d_out[static_cast<int64_t>(global_row) * N + frag * 32 + i] = 0.0f;"
        ),
        "timing_guard_a": (
            'if (is_leader && cta_rank == 0 && timing_mode == TimingMode::kTimed) { '
            'asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock)); }'
        ),
        "timing_guard_b": (
            'if (is_leader && cta_rank == 0 && timing_mode == TimingMode::kTimed) { '
            'asm volatile("mov.u64 %0, %%clock64;" : "=l"(end_clock)); '
            "elapsed_cycles = end_clock - start_clock; }"
        ),
        "kernel_mode_forward": (
            "umma_2sm_body<N, DEPTH>(iterations, timing_mode, g_d_out, g_elapsed_cycles, g_launch_ok);"
        ),
        "run_once_mode_forward": (
            "spec.kernel<<<kGridBlocks, kThreadsPerCta, static_cast<size_t>(smem_bytes)>>>("
            "iterations, mode, d_out_device, cycles_device, launch_ok_device);"
        ),
        "host_smem_setup": (
            "const int n_local = spec.n / kClusterCtas; "
            "const int smem_bytes = kMLocal * kK * 2 + n_local * kK * 2;"
        ),
        "untimed_call_a": "run_once(spec, iterations, TimingMode::kUntimed);",
        "untimed_call_b": "run_once(spec, kSelfTestIterations, TimingMode::kUntimed);",
        "timed_call": "run_once(spec, iterations, TimingMode::kTimed);",
        "prevalidation_call": "run_untimed_or_die(*spec, cli.iterations);",
        "warmup_call": "run_untimed_or_die(*spec, cli.iterations);",
        "timed_repetition_call": "run_timed_or_die(*spec, cli.iterations);",
    }
    fields.update(overrides)

    def asm(text: str) -> str:
        return f'asm volatile("{text}");' if text else ""

    return (
        f"{fields['geometry_consts']}\n"
        "__device__ bool launch_contract_is_valid(uint32_t cluster_nctarank, uint32_t cluster_ctarank) {\n"
        f"    {fields['launch_predicate']}\n"
        "}\n"
        f"__device__ void tcgen05_wait_ld() {{ {asm(fields['wait_ld_text'])} }}\n"
        f"__device__ {fields['fence_helper_def']}\n"
        "__device__ void umma_2sm_body(int64_t iterations, TimingMode timing_mode) {\n"
        f"    {fields['launch_guard']}\n"
        f"    {fields['accepted_path']}\n"
        "    const int cta_rank = static_cast<int>(cluster_ctarank);\n"
        f"    {fields['b_local_extent']}\n"
        f"    {fields['a_loop']}\n"
        f"    {fields['b_loop']}\n"
        "    __syncthreads();\n"
        "    if (tid == 0) {\n"
        f"        {fields['mbarrier_init_call']}\n"
        f"        {fields['fence_helper_call']}\n"
        f"        {fields['fence_proxy_call']}\n"
        "    }\n"
        "    __syncthreads();\n"
        "    cuda::ptx::barrier_cluster_arrive();\n"
        "    cuda::ptx::barrier_cluster_wait();\n"
        f"    {fields['alloc_call']}\n"
        "    uint64_t start_clock = 0, end_clock = 0;\n"
        f"    {fields['timing_guard_a']}\n"
        "    uint32_t parity = 0;\n"
        f"    {fields['loop_open']}\n"
        f"        {fields['leader_wrapper_open']}\n"
        f"            {fields['rank0_wrapper_open']}\n"
        f"                {fields['issue_call']}\n"
        f"            {fields['rank0_wrapper_close']}\n"
        f"            {fields['extra_issue_call']}\n"
        f"            {fields['extra_commit_call']}\n"
        f"            {fields['wait_call']}\n"
        f"            {fields['parity_advance']}\n"
        f"            {fields['leader_extra']}\n"
        f"        {fields['leader_wrapper_close']}\n"
        f"        {fields['post_wait_syncthreads']}\n"
        f"        {fields['loop_cluster_arrive']}\n"
        f"        {fields['loop_cluster_wait']}\n"
        f"    {fields['loop_close']}\n"
        f"    {fields['after_loop_extra']}\n"
        f"    {fields['timing_guard_b']}\n"
        f"    {fields['readback_global_row']}\n"
        "    constexpr int kFragments = N / 32;\n"
        "    for (int frag = 0; frag < kFragments; ++frag) {\n"
        f"        {fields['readback_load_call']}\n"
        f"        {fields['readback_wait_call']}\n"
        f"        {fields['readback_store_call']}\n"
        "    }\n"
        f"    {fields['cluster_sync_before_dealloc']}\n"
        f"    {fields['dealloc_call']}\n"
        "}\n"
        "template <int N, int DEPTH>\n"
        "void visible_kernel(int64_t iterations, TimingMode timing_mode) {\n"
        f"    {fields['kernel_mode_forward']}\n"
        "}\n"
        "RunResult run_once(const Specialization& spec, int64_t iterations, TimingMode mode) {\n"
        f"    {fields['host_smem_setup']}\n"
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
        f"{asm(fields['fence_text'])}\n"
        f"{asm(fields['mma_text'])}\n"
        f"{asm(fields['commit_text'])}\n"
        f"{asm(fields['alloc_text'])}\n"
        f"{asm(fields['dealloc_text'])}\n"
        f"{asm(fields['relinquish_text'])}\n"
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
            "source check rejects a missing barrier_cluster_arrive before deallocation",
            golden_source_snippet(cluster_sync_before_dealloc="cuda::ptx::barrier_cluster_wait();"),
            "barrier_cluster_arrive()/barrier_cluster_wait() pair must appear",
        ),
        (
            "source check rejects a missing barrier_cluster_wait before deallocation",
            golden_source_snippet(cluster_sync_before_dealloc="cuda::ptx::barrier_cluster_arrive();"),
            "barrier_cluster_arrive()/barrier_cluster_wait() pair must appear",
        ),
        (
            "source check rejects a missing exact multicast mask 0x0003",
            golden_source_snippet(
                mask_text="",
                issue_call=(
                    "issue_one_umma_2sm(tmem_d, a_desc, b_desc, idesc, 0); "
                    "commit_umma_2sm_multicast(mbar_addr, 0x0007u);"
                ),
            ),
            "the exact multicast CTA mask 0x0003",
        ),
        (
            "source check rejects required PTX text present only in a // comment",
            golden_source_snippet(wait_ld_text="", extra_forbidden_line="// tcgen05.wait::ld.sync.aligned"),
            "tcgen05.wait::ld.sync.aligned",
        ),
        (
            "source check rejects required PTX text present only in a /* */ comment",
            golden_source_snippet(fence_text="", extra_forbidden_line="/* tcgen05.fence::after_thread_sync */"),
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
            "GPU regression: rejects a full replicated B tile in both CTAs",
            golden_source_snippet(
                b_loop=(
                    "for (int idx = tid; idx < N * kK; idx += kThreadsPerCta) { "
                    "const int col = idx / kK; const int k = idx % kK; "
                    "const int value = ((2 * k + col) % 5) - 2; "
                    "B[smem_core_tile_index(col/8, col%8, k)] = "
                    "__float2bfloat16(static_cast<float>(value)); }"
                )
            ),
            "B initialization loop must cover exactly kNLocal * kK elements",
        ),
        (
            "GPU regression: rejects a local B extent that is not N / kClusterCtas",
            golden_source_snippet(
                b_local_extent=(
                    "constexpr int kNLocal = N; "
                    "static_assert(N % kClusterCtas == 0, \"N must divide evenly\");"
                )
            ),
            "constexpr int kNLocal = N / kClusterCtas;",
        ),
        (
            "GPU regression: a string decoy cannot hide the wrong local B extent",
            golden_source_snippet(
                b_local_extent=(
                    "constexpr int kNLocal = N; "
                    "static_assert(N % kClusterCtas == 0, \"N must divide evenly\");"
                ),
                extra_forbidden_line=(
                    'const char* b_extent_decoy = '
                    '"constexpr int kNLocal = N / kClusterCtas;";'
                ),
            ),
            "constexpr int kNLocal = N / kClusterCtas;",
        ),
        (
            "GPU regression: rejects a rank-independent B global column",
            golden_source_snippet(
                b_loop=(
                    "for (int idx = tid; idx < kNLocal * kK; idx += kThreadsPerCta) { "
                    "const int local_col = idx / kK; const int k = idx % kK; "
                    "const int global_col = local_col; "
                    "const int value = ((2 * k + global_col) % 5) - 2; "
                    "B[smem_core_tile_index(local_col/8, local_col%8, k)] = "
                    "__float2bfloat16(static_cast<float>(value)); }"
                )
            ),
            "global_col = cta_rank * kNLocal + local_col",
        ),
        (
            "GPU regression: rejects B values computed from local_col despite a global_col decoy",
            golden_source_snippet(
                b_loop=(
                    "for (int idx = tid; idx < kNLocal * kK; idx += kThreadsPerCta) { "
                    "const int local_col = idx / kK; const int k = idx % kK; "
                    "const int global_col = cta_rank * kNLocal + local_col; "
                    "const int value = ((2 * k + local_col) % 5) - 2; "
                    "B[smem_core_tile_index(local_col/8, local_col%8, k)] = "
                    "__float2bfloat16(static_cast<float>(value)); }"
                )
            ),
            "B initialization value must use global_col",
        ),
        (
            "GPU regression: rejects using global_col as the local B SMEM address",
            golden_source_snippet(
                b_loop=(
                    "for (int idx = tid; idx < kNLocal * kK; idx += kThreadsPerCta) { "
                    "const int local_col = idx / kK; const int k = idx % kK; "
                    "const int global_col = cta_rank * kNLocal + local_col; "
                    "const int value = ((2 * k + global_col) % 5) - 2; "
                    "B[smem_core_tile_index(global_col/8, global_col%8, k)] = "
                    "__float2bfloat16(static_cast<float>(value)); }"
                )
            ),
            "store the global-column value at the local_col SMEM position",
        ),
        (
            "GPU regression: rejects a host local-B extent smaller than N / kClusterCtas",
            golden_source_snippet(
                host_smem_setup=(
                    "const int n_local = spec.n / 4; "
                    "const int smem_bytes = kMLocal * kK * 2 + n_local * kK * 2;"
                )
            ),
            "const int n_local = spec.n / kClusterCtas;",
        ),
        (
            "GPU regression: rejects host dynamic SMEM that omits the local B slice",
            golden_source_snippet(
                host_smem_setup=(
                    "const int n_local = spec.n / kClusterCtas; "
                    "const int smem_bytes = kMLocal * kK * 2;"
                )
            ),
            "n_local * kK * 2 dynamic shared-memory bytes",
        ),
        (
            "GPU regression: rejects a kernel launch that omits the computed dynamic SMEM size",
            golden_source_snippet(
                run_once_mode_forward=(
                    "spec.kernel<<<kGridBlocks, kThreadsPerCta>>>(iterations, mode, "
                    "d_out_device, cycles_device, launch_ok_device);"
                )
            ),
            "pass static_cast<size_t>(smem_bytes) as the dynamic shared-memory argument",
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
            golden_source_snippet(rank0_wrapper_open="", rank0_wrapper_close=""),
            "must both be issued from a single 'if (cta_rank == 0) { ... }' block",
        ),
        (
            "source check rejects a TMEM load address offset by cta_rank",
            golden_source_snippet(
                readback_load_call=(
                    "tcgen05_ld_32x32b_x32(make_tmem_load_address(tmem_d, warp_id, frag) "
                    "+ cta_rank * kMLocal, regs);"
                )
            ),
            "TMEM load address must not be offset by cta_rank",
        ),
        (
            "source check rejects D readback written by local_row instead of global_row",
            golden_source_snippet(
                readback_store_call=(
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
                    'if (is_leader && timing_mode == TimingMode::kTimed) { '
                    'asm volatile("mov.u64 %0, %%clock64;" : "=l"(start_clock)); }'
                ),
            ),
            "guard(s); expected exactly two",
        ),
        (
            "source check rejects a timed guard missing the is_leader conjunct",
            golden_source_snippet(
                timing_guard_a=(
                    'if (cta_rank == 0 && timing_mode == TimingMode::kTimed) { '
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
        (
            "the mbarrier-init fence helper must genuinely call the official wrapper",
            golden_source_snippet(fence_helper_def="void fence_mbarrier_init_release_cluster() { /* no-op */ }"),
            "must call cuda::ptx::fence_mbarrier_init",
        ),
        (
            "source check rejects a missing fence_mbarrier_init_release_cluster() helper definition",
            golden_source_snippet(fence_helper_def=""),
            "invalid mbarrier-init fence helper",
        ),
        (
            "the mbarrier-init fence call must be ordered after mbarrier_init, not before",
            golden_source_snippet(
                mbarrier_init_call="",
                fence_helper_call="fence_mbarrier_init_release_cluster(); cuda::ptx::mbarrier_init(&mbar, 1u);",
            ),
            "must be called after mbarrier_init(&mbar, ...)",
        ),

        # ---- Task section 5: fourteen independent adversarial mutations. -
        # ---- Each isolates exactly one defect; every one must be rejected
        # ---- for the intended reason (mutation 5 is folded in above,
        # ---- immediately after the "missing tcgen05.wait::ld" case, since
        # ---- it is a variant of the same required-PTX-evidence family). --
        (
            "mutation 1/14: changes kGridBlocks from 2 to 3",
            golden_source_snippet(geometry_consts=(
                "constexpr int kThreadsPerCta = 128;\n"
                "constexpr int kClusterCtas = 2;\n"
                "constexpr int kGridBlocks = 3;\n"
            )),
            "constexpr int kGridBlocks = 2;",
        ),
        (
            "mutation 2/14: changes kThreadsPerCta from 128 to 64",
            golden_source_snippet(geometry_consts=(
                "constexpr int kThreadsPerCta = 64;\n"
                "constexpr int kClusterCtas = 2;\n"
                "constexpr int kGridBlocks = 2;\n"
            )),
            "constexpr int kThreadsPerCta = 128;",
        ),
        (
            "mutation 3/14: changes kClusterCtas from 2 to 4",
            golden_source_snippet(geometry_consts=(
                "constexpr int kThreadsPerCta = 128;\n"
                "constexpr int kClusterCtas = 4;\n"
                "constexpr int kGridBlocks = 2;\n"
            )),
            "constexpr int kClusterCtas = 2;",
        ),
        (
            "mutation 4/14: changes __cluster_dims__(2,1,1) to another value",
            golden_source_snippet(
                cluster_dims_text="__global__ __cluster_dims__(4, 1, 1) __launch_bounds__(128) void k() {}"
            ),
            "__cluster_dims__(2, 1, 1)",
        ),
        (
            "mutation 5/14: removes the executable tcgen05.wait::ld while leaving the mnemonic in a normal string",
            golden_source_snippet(
                wait_ld_text="",
                extra_forbidden_line='const char* decoy = "tcgen05.wait::ld.sync.aligned";',
            ),
            "tcgen05.wait::ld.sync.aligned",
        ),
        (
            "mutation 6/14: adds an extra UMMA issue outside the rank-0 condition",
            golden_source_snippet(extra_issue_call="issue_one_umma_2sm(tmem_d, a_desc, b_desc, idesc, 1);"),
            "issue_one_umma_2sm must be confined to the rank-0-nested-in-leader block",
        ),
        (
            "mutation 7/14: adds an extra commit outside the rank-0 condition",
            golden_source_snippet(extra_commit_call="commit_umma_2sm_multicast(mbar_addr, 0x0003u);"),
            "commit_umma_2sm_multicast must be confined to the rank-0-nested-in-leader block",
        ),
        (
            "mutation 8a/14: removes fence.mbarrier_init.release.cluster's call, leaving a comment decoy",
            golden_source_snippet(fence_helper_call="// fence_mbarrier_init_release_cluster();"),
            "must call fence_mbarrier_init_release_cluster() exactly once, found 0",
        ),
        (
            "mutation 8b/14: removes fence.mbarrier_init.release.cluster's call, leaving a string decoy",
            golden_source_snippet(
                fence_helper_call='const char* decoy = "fence_mbarrier_init_release_cluster();";'
            ),
            "must call fence_mbarrier_init_release_cluster() exactly once, found 0",
        ),
        (
            "mutation 9/14: removes the CTA-wide synchronization after the local mbarrier wait",
            golden_source_snippet(post_wait_syncthreads=""),
            "must contain a __syncthreads() call, after the per-iteration leader block",
        ),
        (
            "mutation 10/14: removes the per-phase cluster arrive",
            golden_source_snippet(loop_cluster_arrive=""),
            "must contain a barrier_cluster_arrive() call, after the post-wait __syncthreads()",
        ),
        (
            "mutation 11/14: removes the per-phase cluster wait",
            golden_source_snippet(loop_cluster_wait=""),
            "must contain a barrier_cluster_wait() call, after barrier_cluster_arrive()",
        ),
        (
            "mutation 12/14: moves the cluster rendezvous outside the runtime iteration loop",
            golden_source_snippet(
                post_wait_syncthreads="",
                loop_cluster_arrive="",
                loop_cluster_wait="",
                after_loop_extra=(
                    "__syncthreads(); cuda::ptx::barrier_cluster_arrive(); cuda::ptx::barrier_cluster_wait();"
                ),
            ),
            "must contain a __syncthreads() call, after the per-iteration leader block",
        ),
        (
            "mutation 13/14: places the cluster rendezvous inside is_leader",
            golden_source_snippet(
                post_wait_syncthreads="",
                loop_cluster_arrive="",
                loop_cluster_wait="",
                leader_extra=(
                    "__syncthreads(); cuda::ptx::barrier_cluster_arrive(); cuda::ptx::barrier_cluster_wait();"
                ),
            ),
            "must not be issued from inside the per-iteration 'if (is_leader)' block",
        ),
        (
            "mutation 14/14: places the mbarrier wait inside cta_rank == 0",
            golden_source_snippet(
                issue_call=(
                    "issue_one_umma_2sm(tmem_d, a_desc, b_desc, idesc, 0); "
                    "commit_umma_2sm_multicast(mbar_addr, 0x0003u); "
                    "while (!cuda::ptx::mbarrier_try_wait_parity(&mbar, parity)) {}"
                ),
                wait_call="",
            ),
            "the mbarrier completion wait must not be enclosed in a cta_rank == 0 condition",
        ),

        # ---- Independent re-audit regressions found after the original --
        # ---- 88-case repair.  Each reproduces one source mutation that ---
        # ---- the old checker incorrectly accepted with zero errors. ------
        (
            "re-audit regression: removes the live tcgen05_wait_ld() call but keeps its PTX helper",
            golden_source_snippet(readback_wait_call=""),
            "must call tcgen05_wait_ld() exactly once after the fragment load",
        ),
        (
            "re-audit regression: wraps both-CTA mbarrier wait in a second cta_rank == 0 condition",
            golden_source_snippet(
                wait_call=(
                    "if (cta_rank == 0) { "
                    "while (!cuda::ptx::mbarrier_try_wait_parity(&mbar, parity)) {} }"
                )
            ),
            "the mbarrier completion wait must not be enclosed in a cta_rank == 0 condition",
        ),
        (
            "re-audit regression: wraps the per-phase CTA/cluster rendezvous in cta_rank == 0",
            golden_source_snippet(
                post_wait_syncthreads="if (cta_rank == 0) { __syncthreads();",
                loop_cluster_wait="cuda::ptx::barrier_cluster_wait(); }",
            ),
            "post-wait __syncthreads() must be a direct statement of the runtime outer iteration loop",
        ),
        (
            "re-audit regression: removes the mbarrier parity phase advance",
            golden_source_snippet(parity_advance=""),
            "must advance mbarrier parity exactly once per iteration",
        ),
        (
            "re-audit regression: makes the mbarrier-init fence call unreachable",
            golden_source_snippet(
                fence_helper_call=(
                    "if (false) { fence_mbarrier_init_release_cluster(); }"
                )
            ),
            "must be a direct, unconditionally reachable statement",
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
        (
            "the repaired canonical source (src/compute/umma_2sm.cu) is accepted with zero errors",
            validate_source_file(resolve_default_source_path()) == [],
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
