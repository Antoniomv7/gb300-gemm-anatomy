#!/usr/bin/env bash
# P4.1 -- the single public Phase 4 campaign orchestration entry point.
#
# One reproducible top-level campaign across the three closed experiments:
#
#   1. LDGSTS versus TMA                 (P1.4, results/raw/exp01_memory_paths_p14)
#   2. BF16 UMMA throughput              (P2.4, results/raw/exp02_umma_throughput_p24)
#   3. CuTe DSL BF16 GEMM vs cuBLASLt    (P3.5, one atomic 5x4 comparison)
#
# This script composes those already audited units; it never reimplements a
# kernel, a scientific matrix, a statistic, a profiler plan, a correctness
# rule, a schema, or an execution parameter. Every GPU action is delegated to
# an existing Make target and, through it, to scripts/run_container.sh. This
# script never calls Docker or nvidia-smi itself, never selects a GPU
# automatically, never uses --gpus all, privileged mode, host PID, added
# capabilities, a Docker socket, MPS, sudo, or multiple GPUs, never changes
# clocks/power/persistence/compute mode, and never uses $(nproc).
#
# Experiment 3 is ONE atomic `gemm` stage. P3.5 enforces shared operands,
# shape-major ordering, all-or-nothing output, and one common comparison
# contract over its five frozen shapes and four candidates, so there is
# deliberately no `--only cublaslt` and no `--only cutedsl`.
#
# P4.1 is orchestration infrastructure only. It runs no pilot and no final
# campaign, computes no cross-experiment interpretation, creates no
# variability or publication threshold, and produces NO publishable result:
# every artifact it writes records publishable=false. P4.2 will execute one
# pilot plus three independent final campaigns; P4.3 will perform the
# integrated analysis, documentation, and final audit.
#
# See src/phase4/P4_1_PROTOCOL.md for the complete frozen contract.
#
# Exit codes:
#   0  success, --help, --self-test, --dry-run, or a pure revalidation of an
#      already COMPLETE campaign
#   1  a stage or validation failure
#   2  a CLI, repository-state, or safety-precondition failure
#   4  a terminal but non-complete INCONCLUSIVE campaign outcome

set -Eeuo pipefail
set -o noclobber

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SELF_PATH="${REPO_ROOT}/scripts/run_all.sh"
ORCHESTRATOR="${REPO_ROOT}/scripts/phase4_orchestrator.py"
PHASE4_RAW_ROOT="${REPO_ROOT}/results/raw/phase4"

usage() {
    cat <<'EOF'
Usage:
  scripts/run_all.sh --help
  scripts/run_all.sh --self-test

  scripts/run_all.sh --dry-run \
      --campaign-id <YYYYMMDDTHHMMSSZ> \
      --campaign-kind <pilot|final> \
      [--only <memory|umma|gemm>]

  BLACKWELL_GPU_INDEX=<physical-index> scripts/run_all.sh \
      --campaign-id <YYYYMMDDTHHMMSSZ> \
      --campaign-kind <pilot|final> \
      [--only <memory|umma|gemm>] \
      [--resume]

P4.1 Phase 4 campaign orchestrator. Composes the closed P1.4, P2.4, and P3.5
units into one reproducible top-level campaign. See src/phase4/P4_1_PROTOCOL.md.

Options:
  --help              Show this help and exit 0. Standalone only.
  --self-test         Run GPU-free synthetic checks and exit. No Docker, no
                      GPU, no nvidia-smi, no CUDA/PyTorch/CuTe DSL import, no
                      network, no results directory, no manifest or log write,
                      no Git mutation. Standalone only.
  --dry-run           Print the deterministic stage plan and exit 0 without
                      executing it. Performs no mutation whatsoever.
  --campaign-id ID    REQUIRED. An explicit canonical UTC timestamp matching
                      exactly YYYYMMDDTHHMMSSZ. A campaign ID is never
                      generated implicitly.
  --campaign-kind K   REQUIRED and immutable: 'pilot' or 'final'. It records
                      whether P4.2 is using this invocation as its pilot or as
                      one of its final replicates. campaign-kind=final does
                      NOT make any raw evidence publishable.
  --only SELECTOR     Exactly one of memory, umma, or gemm. Without --only the
                      complete three-experiment plan runs. There is no
                      --only cublaslt and no --only cutedsl: P3.5 is one
                      atomic comparison.
  --resume            Continue an existing matching top-level campaign. Every
                      apparently complete stage is re-validated from its own
                      evidence before it is skipped; a fresh preflight is
                      created and validated whenever GPU work is still pending.

An existing campaign without --resume is rejected; --resume for a missing
campaign is rejected. Duplicate options, missing values, unexpected positional
arguments, and unknown options exit 2.

Every real invocation with pending GPU work requires an explicit numeric
BLACKWELL_GPU_INDEX. This project never selects a GPU automatically.

NOTHING this script produces is publishable: every artifact and manifest
records publishable=false. P4.1 is implemented infrastructure whose
independent audit and GB300 verification are both PENDING.
EOF
}

fail_cli() {
    echo "run_all: ERROR: $*" >&2
    usage >&2
    exit 2
}

fail_precondition() {
    echo "run_all: ERROR: $*" >&2
    exit 2
}

# ---------------------------------------------------------------------------
# GPU-free self-test.
#
# Every check below runs against this script's own real CLI and a temporary
# directory. Nothing here starts Docker, queries nvidia-smi, imports CUDA,
# PyTorch, or CuTe DSL, touches the network, creates a results directory,
# writes a manifest or a log, or mutates Git. The exhaustive acceptance suite
# lives in scripts/check_phase4_orchestrator_p41.py.
# ---------------------------------------------------------------------------

ST_FAILURES=0

st_check() {
    # $1 label, $2 condition result (0 = pass), $3 detail
    if [ "$2" -eq 0 ]; then
        echo "run_all: self-test: PASS: $1"
    else
        echo "run_all: self-test: FAIL: $1: ${3:-}" >&2
        ST_FAILURES=$((ST_FAILURES + 1))
    fi
}

st_expect_status() {
    # $1 expected status, then the argv to pass to this script
    local expected="$1"; shift
    local status=0
    "${SELF_PATH}" "$@" >/dev/null 2>&1 || status=$?
    if [ "${status}" -eq "${expected}" ]; then
        st_check "'$*' exits ${expected}" 0
    else
        st_check "'$*' exits ${expected}" 1 "got ${status}"
    fi
}

self_test() {
    if [ "${RUN_ALL_SELF_TEST_ACTIVE:-0}" = "1" ]; then
        echo "run_all: ERROR: --self-test must not recurse" >&2
        return 2
    fi
    export RUN_ALL_SELF_TEST_ACTIVE=1

    local phase4_existed=0
    [ -e "${PHASE4_RAW_ROOT}" ] && phase4_existed=1

    local tmp_dir
    tmp_dir="$(mktemp -d)" || { echo "run_all: self-test: cannot create a temporary directory" >&2; return 1; }
    # shellcheck disable=SC2064
    trap "rm -rf -- '${tmp_dir}'" RETURN

    echo "run_all: self-test: the private helper's own GPU-free checks"
    local status=0
    python3 "${ORCHESTRATOR}" --self-test >"${tmp_dir}/helper.log" 2>&1 || status=$?
    st_check "scripts/phase4_orchestrator.py --self-test passes" "${status}" "see ${tmp_dir}/helper.log"
    [ "${status}" -eq 0 ] || sed -n '1,40p' "${tmp_dir}/helper.log" >&2

    echo "run_all: self-test: standalone options"
    st_expect_status 0 --help
    st_expect_status 2 --help --dry-run
    st_expect_status 2 --help --campaign-id 20990101T000000Z
    st_expect_status 2 --help --help

    echo "run_all: self-test: missing, invalid, duplicated, and unknown options"
    st_expect_status 2
    st_expect_status 2 --campaign-id
    st_expect_status 2 --campaign-kind
    st_expect_status 2 --only
    st_expect_status 2 --campaign-id 20990101T000000Z
    st_expect_status 2 --campaign-kind pilot
    st_expect_status 2 --dry-run --campaign-id 20990101T000000Z
    st_expect_status 2 --dry-run --campaign-kind pilot
    st_expect_status 2 --dry-run --campaign-id 20990101T000000Z --campaign-id 20990101T000001Z --campaign-kind pilot
    st_expect_status 2 --dry-run --campaign-id 20990101T000000Z --campaign-kind pilot --campaign-kind final
    st_expect_status 2 --dry-run --campaign-id 20990101T000000Z --campaign-kind pilot --only memory --only umma
    st_expect_status 2 --dry-run --dry-run --campaign-id 20990101T000000Z --campaign-kind pilot
    st_expect_status 2 --dry-run --campaign-id 20990101T000000Z --campaign-kind pilot --unknown
    st_expect_status 2 --dry-run --campaign-id 20990101T000000Z --campaign-kind pilot extra
    st_expect_status 2 --dry-run --campaign-id 20990101T000000Z --campaign-kind pilot -- extra
    st_expect_status 2 --dry-run --campaign-id 20990101T000000Z --campaign-kind maybe
    st_expect_status 2 --dry-run --campaign-id 2099-01-01 --campaign-kind pilot
    st_expect_status 2 --dry-run --campaign-id 20991301T000000Z --campaign-kind pilot
    st_expect_status 2 --dry-run --campaign-id 20990101T000000Z --campaign-kind pilot --only everything
    st_expect_status 2 --dry-run --campaign-id 20990101T000000Z --campaign-kind pilot --resume

    echo "run_all: self-test: the forbidden per-library selectors stay forbidden"
    st_expect_status 2 --dry-run --campaign-id 20990101T000000Z --campaign-kind pilot --only cublaslt
    st_expect_status 2 --dry-run --campaign-id 20990101T000000Z --campaign-kind pilot --only cutedsl

    echo "run_all: self-test: a missing or non-numeric GPU index is rejected before any side effect"
    status=0
    env -u BLACKWELL_GPU_INDEX "${SELF_PATH}" --campaign-id 20990101T000000Z \
        --campaign-kind pilot >/dev/null 2>&1 || status=$?
    st_check "a real run without BLACKWELL_GPU_INDEX exits 2" "$([ "${status}" -eq 2 ] && echo 0 || echo 1)" "got ${status}"
    status=0
    BLACKWELL_GPU_INDEX=abc "${SELF_PATH}" --campaign-id 20990101T000000Z \
        --campaign-kind pilot >/dev/null 2>&1 || status=$?
    st_check "a real run with a non-numeric BLACKWELL_GPU_INDEX exits 2" "$([ "${status}" -eq 2 ] && echo 0 || echo 1)" "got ${status}"

    echo "run_all: self-test: deterministic --dry-run plans"
    local expected_full="preflight memory.pilot memory.profile umma.pilot umma.profile gemm.capture memory.analyze umma.analyze campaign.validate"
    local observed
    observed="$("${SELF_PATH}" --dry-run --campaign-id 20990101T000000Z --campaign-kind pilot \
        | awk '/^  [0-9]+\./ {print $2}' | tr '\n' ' ' | sed 's/ $//')"
    st_check "the full plan is the frozen nine-stage order" \
        "$([ "${observed}" = "${expected_full}" ] && echo 0 || echo 1)" "got '${observed}'"
    observed="$("${SELF_PATH}" --dry-run --campaign-id 20990101T000000Z --campaign-kind final \
        --only memory | awk '/^  [0-9]+\./ {print $2}' | tr '\n' ' ' | sed 's/ $//')"
    st_check "the --only memory plan is the frozen five-stage order" \
        "$([ "${observed}" = "preflight memory.pilot memory.profile memory.analyze campaign.validate" ] && echo 0 || echo 1)" \
        "got '${observed}'"
    observed="$("${SELF_PATH}" --dry-run --campaign-id 20990101T000000Z --campaign-kind final \
        --only umma | awk '/^  [0-9]+\./ {print $2}' | tr '\n' ' ' | sed 's/ $//')"
    st_check "the --only umma plan is the frozen five-stage order" \
        "$([ "${observed}" = "preflight umma.pilot umma.profile umma.analyze campaign.validate" ] && echo 0 || echo 1)" \
        "got '${observed}'"
    observed="$("${SELF_PATH}" --dry-run --campaign-id 20990101T000000Z --campaign-kind final \
        --only gemm | awk '/^  [0-9]+\./ {print $2}' | tr '\n' ' ' | sed 's/ $//')"
    st_check "the --only gemm plan is the frozen three-stage order" \
        "$([ "${observed}" = "preflight gemm.capture campaign.validate" ] && echo 0 || echo 1)" \
        "got '${observed}'"

    "${SELF_PATH}" --dry-run --campaign-id 20990101T000000Z --campaign-kind pilot \
        >"${tmp_dir}/plan.a" 2>/dev/null
    "${SELF_PATH}" --dry-run --campaign-id 20990101T000000Z --campaign-kind pilot \
        >"${tmp_dir}/plan.b" 2>/dev/null
    status=0
    cmp -s "${tmp_dir}/plan.a" "${tmp_dir}/plan.b" || status=1
    st_check "two --dry-run invocations print byte-identical plans" "${status}"
    status=0
    grep -Fq 'publishable: false' "${tmp_dir}/plan.a" || status=1
    st_check "the printed plan states publishable: false" "${status}"

    echo "run_all: self-test: side-effect freedom"
    if [ "${phase4_existed}" -eq 0 ] && [ -e "${PHASE4_RAW_ROOT}" ]; then
        st_check "--help/--self-test/--dry-run create no results/raw/phase4 tree" 1 "the tree appeared"
    else
        st_check "--help/--self-test/--dry-run create no results/raw/phase4 tree" 0
    fi

    if [ "${ST_FAILURES}" -ne 0 ]; then
        echo "run_all: self-test: FAILED (${ST_FAILURES} check(s))" >&2
        return 1
    fi
    echo "run_all: self-test: PASS"
    return 0
}

# ---------------------------------------------------------------------------
# Strict CLI parsing. Duplicate options, missing values, unexpected positional
# arguments, and unknown options all fail closed with exit code 2.
# ---------------------------------------------------------------------------

OPT_HELP=0
OPT_SELF_TEST=0
OPT_DRY_RUN=0
OPT_RESUME=0
OPT_CAMPAIGN_ID=""
OPT_CAMPAIGN_KIND=""
OPT_ONLY=""
SEEN_CAMPAIGN_ID=0
SEEN_CAMPAIGN_KIND=0
SEEN_ONLY=0

require_value() {
    # $1 option name, $2 remaining argument count, $3 candidate value
    [ "$2" -ge 2 ] || fail_cli "$1 requires a value"
    [ -n "$3" ] || fail_cli "$1 requires a non-empty value"
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --help)
            [ "${OPT_HELP}" -eq 0 ] || fail_cli "duplicate --help"
            OPT_HELP=1; shift ;;
        --self-test)
            [ "${OPT_SELF_TEST}" -eq 0 ] || fail_cli "duplicate --self-test"
            OPT_SELF_TEST=1; shift ;;
        --dry-run)
            [ "${OPT_DRY_RUN}" -eq 0 ] || fail_cli "duplicate --dry-run"
            OPT_DRY_RUN=1; shift ;;
        --resume)
            [ "${OPT_RESUME}" -eq 0 ] || fail_cli "duplicate --resume"
            OPT_RESUME=1; shift ;;
        --campaign-id)
            [ "${SEEN_CAMPAIGN_ID}" -eq 0 ] || fail_cli "duplicate --campaign-id"
            require_value "--campaign-id" "$#" "${2:-}"
            OPT_CAMPAIGN_ID="$2"; SEEN_CAMPAIGN_ID=1; shift 2 ;;
        --campaign-id=*)
            [ "${SEEN_CAMPAIGN_ID}" -eq 0 ] || fail_cli "duplicate --campaign-id"
            OPT_CAMPAIGN_ID="${1#--campaign-id=}"
            [ -n "${OPT_CAMPAIGN_ID}" ] || fail_cli "--campaign-id requires a non-empty value"
            SEEN_CAMPAIGN_ID=1; shift ;;
        --campaign-kind)
            [ "${SEEN_CAMPAIGN_KIND}" -eq 0 ] || fail_cli "duplicate --campaign-kind"
            require_value "--campaign-kind" "$#" "${2:-}"
            OPT_CAMPAIGN_KIND="$2"; SEEN_CAMPAIGN_KIND=1; shift 2 ;;
        --campaign-kind=*)
            [ "${SEEN_CAMPAIGN_KIND}" -eq 0 ] || fail_cli "duplicate --campaign-kind"
            OPT_CAMPAIGN_KIND="${1#--campaign-kind=}"
            [ -n "${OPT_CAMPAIGN_KIND}" ] || fail_cli "--campaign-kind requires a non-empty value"
            SEEN_CAMPAIGN_KIND=1; shift ;;
        --only)
            [ "${SEEN_ONLY}" -eq 0 ] || fail_cli "duplicate --only"
            require_value "--only" "$#" "${2:-}"
            OPT_ONLY="$2"; SEEN_ONLY=1; shift 2 ;;
        --only=*)
            [ "${SEEN_ONLY}" -eq 0 ] || fail_cli "duplicate --only"
            OPT_ONLY="${1#--only=}"
            [ -n "${OPT_ONLY}" ] || fail_cli "--only requires a non-empty value"
            SEEN_ONLY=1; shift ;;
        --)
            fail_cli "unexpected '--' separator: this orchestrator takes no positional argument" ;;
        -*)
            fail_cli "unknown option '$1'" ;;
        *)
            fail_cli "unexpected positional argument '$1'" ;;
    esac
done

OTHER_OPTIONS=$((OPT_DRY_RUN + OPT_RESUME + SEEN_CAMPAIGN_ID + SEEN_CAMPAIGN_KIND + SEEN_ONLY))

if [ "${OPT_HELP}" -eq 1 ]; then
    [ "${OPT_SELF_TEST}" -eq 0 ] || fail_cli "--help and --self-test are each standalone"
    [ "${OTHER_OPTIONS}" -eq 0 ] || fail_cli "--help is standalone and takes no other option"
    usage
    exit 0
fi

if [ "${OPT_SELF_TEST}" -eq 1 ]; then
    [ "${OTHER_OPTIONS}" -eq 0 ] || fail_cli "--self-test is standalone and takes no other option"
    self_test
    exit $?
fi

[ "${SEEN_CAMPAIGN_ID}" -eq 1 ] || fail_cli "--campaign-id is required (an explicit canonical UTC timestamp YYYYMMDDTHHMMSSZ)"
[ "${SEEN_CAMPAIGN_KIND}" -eq 1 ] || fail_cli "--campaign-kind is required ('pilot' or 'final')"
case "${OPT_CAMPAIGN_KIND}" in
    pilot|final) ;;
    *) fail_cli "--campaign-kind must be exactly 'pilot' or 'final', got '${OPT_CAMPAIGN_KIND}'" ;;
esac
if [ "${SEEN_ONLY}" -eq 1 ]; then
    case "${OPT_ONLY}" in
        memory|umma|gemm) ;;
        cublaslt|cutedsl)
            fail_cli "--only ${OPT_ONLY} is forbidden: experiment 3 is ONE atomic P3.5 comparison of five shapes x four candidates; the supported selector is --only gemm" ;;
        *) fail_cli "--only must be exactly one of memory, umma, or gemm, got '${OPT_ONLY}'" ;;
    esac
fi
if [ "${OPT_DRY_RUN}" -eq 1 ] && [ "${OPT_RESUME}" -eq 1 ]; then
    fail_cli "--dry-run performs no mutation and therefore never resumes; use one or the other"
fi

[ -f "${ORCHESTRATOR}" ] || fail_precondition "missing ${ORCHESTRATOR}"

ONLY_ARGS=()
if [ "${SEEN_ONLY}" -eq 1 ]; then
    ONLY_ARGS=(--only "${OPT_ONLY}")
fi

# --- --dry-run: deterministic plan, zero mutation ---------------------------
# No Docker, no nvidia-smi, no CUDA/PyTorch/CuTe DSL import, no network, no
# results directory, no manifest, no log, and no Git invocation at all.
if [ "${OPT_DRY_RUN}" -eq 1 ]; then
    exec python3 "${ORCHESTRATOR}" plan \
        --campaign-id "${OPT_CAMPAIGN_ID}" \
        --campaign-kind "${OPT_CAMPAIGN_KIND}" \
        ${ONLY_ARGS[@]+"${ONLY_ARGS[@]}"}
fi

# --- real execution ---------------------------------------------------------
# A new campaign always has pending GPU work, so its explicit numeric GPU index
# is required here, before any subprocess, Docker invocation, nvidia-smi query,
# or results directory is created. A --resume may legitimately be a pure
# read-only revalidation of an already complete campaign, so the orchestrator
# decides there -- still before any mutation or GPU-related subprocess.
if [ -n "${BLACKWELL_GPU_INDEX:-}" ]; then
    case "${BLACKWELL_GPU_INDEX}" in
        '' | *[!0-9]*)
            fail_precondition "BLACKWELL_GPU_INDEX must be a non-negative integer, got '${BLACKWELL_GPU_INDEX}'" ;;
    esac
fi
if [ "${OPT_RESUME}" -eq 0 ] && [ -z "${BLACKWELL_GPU_INDEX:-}" ]; then
    fail_precondition "BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index; this project never selects a GPU automatically"
fi

RESUME_ARGS=()
if [ "${OPT_RESUME}" -eq 1 ]; then
    RESUME_ARGS=(--resume)
fi

exec python3 "${ORCHESTRATOR}" run \
    --campaign-id "${OPT_CAMPAIGN_ID}" \
    --campaign-kind "${OPT_CAMPAIGN_KIND}" \
    ${ONLY_ARGS[@]+"${ONLY_ARGS[@]}"} \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"}
