#!/usr/bin/env bash
# P2.3 joint 1-SM/2-SM BF16 UMMA sweep runner (exp02_umma_throughput).
#
# Runs the frozen 24-invocation plan (12 logical (N,depth) pairs x 2 methods,
# alternating which method runs first per pair; see
# scripts/aggregate_exp02_umma_throughput.py) through scripts/run_container.sh,
# one GPU process at a time, reusing the already-audited P2.1
# (build/compute/umma_1sm) and P2.2 (build/compute/umma_2sm) binaries and
# their existing command-line interfaces completely unmodified -- this
# script introduces no CUDA kernel of its own. It only orchestrates: plan
# generation, symlink-safe campaign initialization, CSV parsing/validation,
# manifest handling, consolidation, and aggregation all live in
# scripts/aggregate_exp02_umma_throughput.py (Python standard library only).
# The 24-case loop below is a strictly sequential `while read` over one GPU
# process at a time, so the two methods never run concurrently.
#
# Output is functional/descriptive infrastructure only: no TFLOP/s, no
# empirical Tensor Core ceiling, no 1-SM/2-SM speedup, no scaling
# efficiency, no saturation, no winning configuration, no Nsight Compute,
# no publishable performance result. See src/compute/P2_3_PROTOCOL.md (the
# scientific boundary with P2.4) and PLAN.md.
#
# Exit codes: 0 success/--help/--print-plan/--self-test; 1 execution,
# correctness, CSV-validation, or aggregation failure; 2 CLI,
# repository-state, or safety-precondition failure.
set -Eeuo pipefail
set -o noclobber

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SELF_PATH="${REPO_ROOT}/scripts/run_exp02_umma_throughput.sh"
AGGREGATOR_HOST="${REPO_ROOT}/scripts/aggregate_exp02_umma_throughput.py"
AGGREGATOR_IN_CONTAINER="scripts/aggregate_exp02_umma_throughput.py"
RUN_CONTAINER="${REPO_ROOT}/scripts/run_container.sh"

usage() {
    cat <<'EOF'
Usage:
  run_exp02_umma_throughput.sh --help
  run_exp02_umma_throughput.sh --print-plan
  run_exp02_umma_throughput.sh --self-test
  run_exp02_umma_throughput.sh --run-kind {smoke,benchmark} [--campaign-id ID] \
      --iterations N --warmup-iterations N --repetitions N

P2.3 joint 1-SM/2-SM BF16 UMMA sweep runner (exp02_umma_throughput): the
frozen 24-invocation plan (12 logical (N,depth) pairs x umma_1sm/umma_2sm,
alternating method order per pair) through scripts/run_container.sh, reusing
the already-audited P2.1/P2.2 binaries and command-line interfaces
unmodified, with strict CSV validation and descriptive aggregation.
Functional/descriptive output only: no TFLOP/s, no empirical Tensor Core
ceiling, no 1-SM/2-SM speedup, no scaling efficiency, no saturation, no
winning configuration, no Nsight Compute, no publishable performance result
(that is P2.4).

Options:
  --help                        Show this help and exit 0. Standalone only.
  --print-plan                  Print the deterministic 24-invocation plan
                                 and exit 0. No GPU, no Docker. Standalone.
  --self-test                   Run GPU-free synthetic checks and exit. No
                                 GPU, no Docker, no nvidia-smi, no network.
                                 Standalone.
  --run-kind {smoke,benchmark}   Required for a campaign. Labels the CSV
                                 rows; forwarded verbatim to both binaries.
  --campaign-id ID                Optional; default is the current UTC
                                 timestamp (YYYYMMDDTHHMMSSZ). Must match
                                 [A-Za-z0-9][A-Za-z0-9._-]{0,63} and contain
                                 no ".." substring.
  --iterations N                  Required for a campaign, in [1, 1000000].
                                 Forwarded verbatim to both binaries.
  --warmup-iterations N            Required for a campaign, in [0, 1000000].
                                 Forwarded verbatim to both binaries.
  --repetitions N                  Required for a campaign, in [2, 1000000].
                                 Forwarded verbatim to both binaries. Real
                                 campaigns require at least two repetitions
                                 so summary.csv's sample standard deviation
                                 and coefficient of variation are never
                                 silently reported as zero for a single
                                 observation.

--help, --print-plan, and --self-test are mutually exclusive with each other
and with every campaign option, and each may be given at most once. Every
other flag may likewise be given at most once; unknown flags are rejected.

A real campaign additionally requires BLACKWELL_GPU_INDEX=<physical-index>
in the environment (never selected automatically) and a clean Git worktree
at a full 40-character commit. Every GPU invocation goes through
scripts/run_container.sh independently; the two methods never run
concurrently -- the 24-case loop is strictly sequential. Campaign
initialization (directory creation, execution_order.csv, the initial
manifest) is centralized and symlink-safe in
scripts/aggregate_exp02_umma_throughput.py's init-campaign subcommand.

This runner reuses the already-audited P2.1 (build/compute/umma_1sm) and
P2.2 (build/compute/umma_2sm) binaries and their existing command-line
interfaces unmodified; it never introduces another CUDA kernel.

Exit codes: 0 success/--help/--print-plan/--self-test; 1 execution,
correctness, CSV-validation, or aggregation failure; 2 CLI,
repository-state, or safety-precondition failure.
EOF
}

fail_cli() {
    echo "run_exp02_umma_throughput: ERROR: $*" >&2
    usage >&2
    exit 2
}

fail_precondition() {
    echo "run_exp02_umma_throughput: ERROR: $*" >&2
    exit 2
}

is_bounded_uint() {
    [[ "$1" =~ ^[0-9]{1,15}$ ]]
}

validate_range() {
    local name="$1" value="$2" min="$3" max="$4"
    is_bounded_uint "$value" || fail_cli "${name} must be a non-negative integer, got '${value}'"
    if [ "${value}" -lt "${min}" ] || [ "${value}" -gt "${max}" ]; then
        fail_cli "${name} must be in [${min}, ${max}], got '${value}'"
    fi
}

# require_regular_nonsymlink_artifact ABS_PATH LABEL: succeeds only if
# ABS_PATH exists as a regular, non-symlink file. Checks -L (lstat; true for
# a symlink regardless of whether its target exists or is itself valid)
# BEFORE -f (which follows symlinks), so a symlinked "binary" or SASS
# evidence file is refused outright rather than silently accepted via a
# test -f that resolved through it (audit repair). LABEL is a
# repository-relative name used only for the diagnostic message, so a
# rejection never prints this host's absolute repository path.
require_regular_nonsymlink_artifact() {
    local abs_path="$1" label="$2"
    if [ -L "${abs_path}" ]; then
        echo "run_exp02_umma_throughput: ERROR: ${label} is a symlink, refusing" >&2
        return 1
    fi
    if [ ! -f "${abs_path}" ]; then
        echo "run_exp02_umma_throughput: ERROR: ${label} missing or not a regular file" >&2
        return 1
    fi
    return 0
}

# _check_rejected_cli LABEL ARGS...: recursively invokes this same script
# with ARGS and asserts it exits 2 (a CLI-rejection path only) without
# touching Docker, nvidia-smi, or results/raw/. Used exclusively by the
# GPU-free self-test below for duplicate/mutually-exclusive/out-of-range
# special-mode checks; every ARGS combination it is called with is one this
# script's own parser rejects before any GPU/Docker/results-directory code
# runs.
_check_rejected_cli() {
    local label="$1"
    shift
    "${SELF_PATH}" "$@" >/dev/null 2>&1
    local rc=$?
    if [ "${rc}" -eq 2 ]; then
        echo "run_exp02_umma_throughput: self-test: PASS: rejects ${label} (exit 2)" >&2
        return 0
    fi
    echo "run_exp02_umma_throughput: self-test: FAIL: ${label} exited ${rc}, expected 2" >&2
    return 1
}

run_self_test() {
    local failures=0

    if [ -f "${REPO_ROOT}/VERSIONS.env" ]; then
        echo "run_exp02_umma_throughput: self-test: PASS: repo root resolves to ${REPO_ROOT}" >&2
    else
        echo "run_exp02_umma_throughput: self-test: FAIL: VERSIONS.env not found at resolved repo root" >&2
        failures=$((failures + 1))
    fi

    local plan_line_count
    plan_line_count="$(python3 "${AGGREGATOR_HOST}" plan --format lines | wc -l | tr -d ' ')"
    if [ "${plan_line_count}" -eq 24 ]; then
        echo "run_exp02_umma_throughput: self-test: PASS: --print-plan has exactly 24 invocations" >&2
    else
        echo "run_exp02_umma_throughput: self-test: FAIL: plan has ${plan_line_count} lines, expected 24" >&2
        failures=$((failures + 1))
    fi

    local symlink_test_dir
    symlink_test_dir="$(mktemp -d)"
    : >"${symlink_test_dir}/real_target"
    ln -s "${symlink_test_dir}/real_target" "${symlink_test_dir}/symlinked_binary"
    if require_regular_nonsymlink_artifact "${symlink_test_dir}/symlinked_binary" "test artifact" 2>/dev/null; then
        echo "run_exp02_umma_throughput: self-test: FAIL: a symlinked execution artifact was NOT rejected before any external launcher could run" >&2
        failures=$((failures + 1))
    else
        echo "run_exp02_umma_throughput: self-test: PASS: a symlinked execution artifact is rejected before any external launcher could run" >&2
    fi
    : >"${symlink_test_dir}/real_regular_file"
    if require_regular_nonsymlink_artifact "${symlink_test_dir}/real_regular_file" "test artifact"; then
        echo "run_exp02_umma_throughput: self-test: PASS: a genuine regular, non-symlink file is accepted" >&2
    else
        echo "run_exp02_umma_throughput: self-test: FAIL: a genuine regular, non-symlink file was rejected" >&2
        failures=$((failures + 1))
    fi
    if require_regular_nonsymlink_artifact "${symlink_test_dir}/does_not_exist" "test artifact" 2>/dev/null; then
        echo "run_exp02_umma_throughput: self-test: FAIL: a missing artifact was NOT rejected" >&2
        failures=$((failures + 1))
    else
        echo "run_exp02_umma_throughput: self-test: PASS: a missing artifact is rejected" >&2
    fi
    rm -rf "${symlink_test_dir}"

    echo "run_exp02_umma_throughput: self-test: delegating to the Python aggregator's synthetic test suite" >&2
    if python3 "${AGGREGATOR_HOST}" --self-test; then
        echo "run_exp02_umma_throughput: self-test: PASS: aggregate_exp02_umma_throughput.py --self-test" >&2
    else
        echo "run_exp02_umma_throughput: self-test: FAIL: aggregate_exp02_umma_throughput.py --self-test" >&2
        failures=$((failures + 1))
    fi

    _check_rejected_cli "'--help --help'" --help --help || failures=$((failures + 1))
    _check_rejected_cli "'-h --help'" -h --help || failures=$((failures + 1))
    _check_rejected_cli "'--print-plan --print-plan'" --print-plan --print-plan || failures=$((failures + 1))
    _check_rejected_cli "'--self-test --self-test'" --self-test --self-test || failures=$((failures + 1))
    _check_rejected_cli "'--help --print-plan'" --help --print-plan || failures=$((failures + 1))
    _check_rejected_cli "'--print-plan --self-test'" --print-plan --self-test || failures=$((failures + 1))
    _check_rejected_cli "'--self-test --run-kind smoke'" --self-test --run-kind smoke || failures=$((failures + 1))
    _check_rejected_cli "'--print-plan --iterations 1'" --print-plan --iterations 1 || failures=$((failures + 1))
    _check_rejected_cli "'--help --self-test'" --help --self-test || failures=$((failures + 1))
    _check_rejected_cli "duplicate --run-kind" --run-kind smoke --run-kind smoke \
        --iterations 1 --warmup-iterations 0 --repetitions 1 || failures=$((failures + 1))
    _check_rejected_cli "duplicate --iterations" --run-kind smoke \
        --iterations 1 --iterations 2 --warmup-iterations 0 --repetitions 1 || failures=$((failures + 1))
    _check_rejected_cli "duplicate --campaign-id" --run-kind smoke --campaign-id a --campaign-id b \
        --iterations 1 --warmup-iterations 0 --repetitions 1 || failures=$((failures + 1))
    _check_rejected_cli "unknown flag" --run-kind smoke --iterations 1 --warmup-iterations 0 \
        --repetitions 1 --bogus-flag || failures=$((failures + 1))
    _check_rejected_cli "'--campaign-id a..b'" --campaign-id a..b \
        --run-kind smoke --iterations 1 --warmup-iterations 0 --repetitions 2 || failures=$((failures + 1))
    _check_rejected_cli "--iterations below range" --run-kind smoke \
        --iterations 0 --warmup-iterations 0 --repetitions 1 || failures=$((failures + 1))
    _check_rejected_cli "--iterations above range" --run-kind smoke \
        --iterations 1000001 --warmup-iterations 0 --repetitions 1 || failures=$((failures + 1))
    _check_rejected_cli "--warmup-iterations negative (non-numeric)" --run-kind smoke \
        --iterations 1 --warmup-iterations -1 --repetitions 1 || failures=$((failures + 1))
    _check_rejected_cli "--repetitions below range" --run-kind smoke \
        --iterations 1 --warmup-iterations 0 --repetitions 0 || failures=$((failures + 1))
    _check_rejected_cli "--repetitions=1 (below the 2-repetition minimum)" --run-kind smoke \
        --iterations 1 --warmup-iterations 0 --repetitions 1 || failures=$((failures + 1))
    _check_rejected_cli "missing --run-kind" --iterations 1 --warmup-iterations 0 --repetitions 1 \
        || failures=$((failures + 1))
    _check_rejected_cli "missing --iterations" --run-kind smoke --warmup-iterations 0 --repetitions 1 \
        || failures=$((failures + 1))
    _check_rejected_cli "invalid --run-kind value" --run-kind bogus \
        --iterations 1 --warmup-iterations 0 --repetitions 1 || failures=$((failures + 1))

    if [ "${failures}" -eq 0 ]; then
        echo "run_exp02_umma_throughput: SELF_TEST_RESULT=PASS" >&2
        return 0
    fi
    echo "run_exp02_umma_throughput: SELF_TEST_RESULT=FAIL (${failures} failure(s))" >&2
    return 1
}

# ---------------------------------------------------------------------------
# CLI parsing: argument arrays only, no eval, no unquoted command
# construction. Every branch below must be reachable without touching the
# GPU, Docker, or nvidia-smi.
# ---------------------------------------------------------------------------
HELP_COUNT=0
PRINT_PLAN_COUNT=0
SELF_TEST_COUNT=0
HAS_RUN_KIND=0; RUN_KIND=""
HAS_CAMPAIGN_ID=0; CAMPAIGN_ID=""
HAS_ITERATIONS=0; ITERATIONS=""
HAS_WARMUP_ITERATIONS=0; WARMUP_ITERATIONS=""
HAS_REPETITIONS=0; REPETITIONS=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --help|-h)
            HELP_COUNT=$((HELP_COUNT + 1)); shift ;;
        --print-plan)
            PRINT_PLAN_COUNT=$((PRINT_PLAN_COUNT + 1)); shift ;;
        --self-test)
            SELF_TEST_COUNT=$((SELF_TEST_COUNT + 1)); shift ;;
        --run-kind)
            [ "${HAS_RUN_KIND}" -eq 0 ] || fail_cli "duplicate --run-kind"
            [ "$#" -ge 2 ] || fail_cli "--run-kind requires a value"
            RUN_KIND="$2"; HAS_RUN_KIND=1; shift 2 ;;
        --campaign-id)
            [ "${HAS_CAMPAIGN_ID}" -eq 0 ] || fail_cli "duplicate --campaign-id"
            [ "$#" -ge 2 ] || fail_cli "--campaign-id requires a value"
            CAMPAIGN_ID="$2"; HAS_CAMPAIGN_ID=1; shift 2 ;;
        --iterations)
            [ "${HAS_ITERATIONS}" -eq 0 ] || fail_cli "duplicate --iterations"
            [ "$#" -ge 2 ] || fail_cli "--iterations requires a value"
            ITERATIONS="$2"; HAS_ITERATIONS=1; shift 2 ;;
        --warmup-iterations)
            [ "${HAS_WARMUP_ITERATIONS}" -eq 0 ] || fail_cli "duplicate --warmup-iterations"
            [ "$#" -ge 2 ] || fail_cli "--warmup-iterations requires a value"
            WARMUP_ITERATIONS="$2"; HAS_WARMUP_ITERATIONS=1; shift 2 ;;
        --repetitions)
            [ "${HAS_REPETITIONS}" -eq 0 ] || fail_cli "duplicate --repetitions"
            [ "$#" -ge 2 ] || fail_cli "--repetitions requires a value"
            REPETITIONS="$2"; HAS_REPETITIONS=1; shift 2 ;;
        -*)
            fail_cli "unknown option: $1" ;;
        *)
            fail_cli "unexpected positional argument: $1" ;;
    esac
done

if [ "${HELP_COUNT}" -gt 1 ] || [ "${PRINT_PLAN_COUNT}" -gt 1 ] || [ "${SELF_TEST_COUNT}" -gt 1 ]; then
    fail_cli "--help, --print-plan, and --self-test may each be given at most once"
fi
SPECIAL_MODE_COUNT=$((HELP_COUNT + PRINT_PLAN_COUNT + SELF_TEST_COUNT))
if [ "${SPECIAL_MODE_COUNT}" -gt 1 ]; then
    fail_cli "--help, --print-plan, and --self-test are mutually exclusive"
fi
if [ "${SPECIAL_MODE_COUNT}" -eq 1 ]; then
    if [ "${HAS_RUN_KIND}" -eq 1 ] || [ "${HAS_CAMPAIGN_ID}" -eq 1 ] || [ "${HAS_ITERATIONS}" -eq 1 ] \
       || [ "${HAS_WARMUP_ITERATIONS}" -eq 1 ] || [ "${HAS_REPETITIONS}" -eq 1 ]; then
        fail_cli "--help, --print-plan, and --self-test cannot be combined with campaign options"
    fi
fi

if [ "${HELP_COUNT}" -eq 1 ]; then
    usage
    exit 0
fi

if [ "${PRINT_PLAN_COUNT}" -eq 1 ]; then
    python3 "${AGGREGATOR_HOST}" plan --format text
    exit 0
fi

if [ "${SELF_TEST_COUNT}" -eq 1 ]; then
    if run_self_test; then
        exit 0
    fi
    exit 1
fi

[ "${HAS_RUN_KIND}" -eq 1 ] || fail_cli "--run-kind is required"
case "${RUN_KIND}" in
    smoke|benchmark) ;;
    *) fail_cli "--run-kind must be 'smoke' or 'benchmark', got '${RUN_KIND}'" ;;
esac
[ "${HAS_ITERATIONS}" -eq 1 ] || fail_cli "--iterations is required"
[ "${HAS_WARMUP_ITERATIONS}" -eq 1 ] || fail_cli "--warmup-iterations is required"
[ "${HAS_REPETITIONS}" -eq 1 ] || fail_cli "--repetitions is required"

# Bounds chosen so all existing signed 64-bit FLOP/UMMA formulas (see
# check_int64_safety() in scripts/aggregate_exp02_umma_throughput.py) stay
# far inside the int64 range for every one of the 24 frozen configurations.
validate_range "--iterations" "${ITERATIONS}" 1 1000000
validate_range "--warmup-iterations" "${WARMUP_ITERATIONS}" 0 1000000
# Real campaigns require repetitions >= 2 (audit repair): summary.csv's
# sample standard deviation and coefficient of variation are only
# statistically meaningful with at least two observations, and must never be
# silently reported as zero for a single sample. Matches
# aggregate_exp02_umma_throughput.py's REPETITIONS_MIN.
validate_range "--repetitions" "${REPETITIONS}" 2 1000000

if [ "${HAS_CAMPAIGN_ID}" -eq 1 ]; then
    [[ "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] \
        || fail_cli "--campaign-id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}, got '${CAMPAIGN_ID}'"
    case "${CAMPAIGN_ID}" in
        *..*) fail_cli "--campaign-id must not contain '..', got '${CAMPAIGN_ID}'" ;;
    esac
else
    CAMPAIGN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
fi

# ---------------------------------------------------------------------------
# Everything below this point is a real campaign attempt: GPU selection,
# repository state, Docker, and the results tree are now all in scope.
# ---------------------------------------------------------------------------
[ -n "${BLACKWELL_GPU_INDEX:-}" ] \
    || fail_precondition "BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index; this runner never selects a GPU automatically"
[[ "${BLACKWELL_GPU_INDEX}" =~ ^[0-9]+$ ]] \
    || fail_precondition "BLACKWELL_GPU_INDEX must be a non-negative integer, got '${BLACKWELL_GPU_INDEX}'"

GIT_STATUS="$(cd "${REPO_ROOT}" && git status --porcelain)"
[ -z "${GIT_STATUS}" ] || fail_precondition "worktree is not clean; commit or stash changes before running a campaign"
GIT_COMMIT="$(cd "${REPO_ROOT}" && git rev-parse HEAD)"
[[ "${GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
    || fail_precondition "unable to resolve a full 40-character Git commit SHA (got '${GIT_COMMIT}')"

STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
init_args=(init-campaign --campaign-id "${CAMPAIGN_ID}" --run-kind "${RUN_KIND}"
    --iterations "${ITERATIONS}" --warmup-iterations "${WARMUP_ITERATIONS}" --repetitions "${REPETITIONS}"
    --git-commit "${GIT_COMMIT}" --gpu-index "${BLACKWELL_GPU_INDEX}" --started-at-utc "${STARTED_AT}")
if ! CAMPAIGN_REL="$(python3 "${AGGREGATOR_HOST}" "${init_args[@]}")"; then
    fail_precondition "campaign initialization failed (unsafe campaign ID, symlink, or existing campaign directory)"
fi
CAMPAIGN_DIR="${REPO_ROOT}/${CAMPAIGN_REL}"

CAMPAIGN_OUTCOME=""
on_exit() {
    local rc=$?
    if [ -z "${CAMPAIGN_OUTCOME}" ] && [ -d "${CAMPAIGN_DIR}" ]; then
        echo "run_exp02_umma_throughput: unexpected termination (rc=${rc}); marking campaign INTERRUPTED" >&2
        CAMPAIGN_OUTCOME=INTERRUPTED
        write_manifest_status INTERRUPTED "unexpected_termination" "${rc}" || true
    fi
}
on_signal() {
    local sig="$1"
    trap - EXIT INT TERM
    if [ -z "${CAMPAIGN_OUTCOME}" ] && [ -d "${CAMPAIGN_DIR}" ]; then
        echo "run_exp02_umma_throughput: received ${sig}; marking campaign INTERRUPTED" >&2
        CAMPAIGN_OUTCOME=INTERRUPTED
        write_manifest_status INTERRUPTED "signal_${sig}" "" || true
    fi
    exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap on_exit EXIT

write_manifest_status() {
    local status="$1" failure_stage="${2:-}" failure_exit_code="${3:-}" failure_log_rel="${4:-}"
    local merge_file
    merge_file="$(mktemp "${CAMPAIGN_DIR}/manifest_merge.XXXXXX")"
    if [ -n "${failure_stage}" ]; then
        if [ -n "${failure_log_rel}" ]; then
            printf '{"failure_stage": "%s", "failure_exit_code": %s, "failure_detail": ["%s"]}\n' \
                "${failure_stage}" "${failure_exit_code:-null}" "${failure_log_rel}" >| "${merge_file}"
        else
            printf '{"failure_stage": "%s", "failure_exit_code": %s}\n' \
                "${failure_stage}" "${failure_exit_code:-null}" >| "${merge_file}"
        fi
    else
        printf '{}' >| "${merge_file}"
    fi
    python3 "${AGGREGATOR_HOST}" manifest-write --campaign-dir "${CAMPAIGN_REL}" \
        --status "${status}" --merge-json "${merge_file}" >/dev/null
    rm -f "${merge_file}"
}

# write_self_test_outcomes UMMA_1SM_OUTCOME UMMA_2SM_OUTCOME: records both
# methods' self-test outcome together (the manifest schema always requires
# both keys). Used to record PENDING/STARTED immediately before a self-test
# runs and PASS/FAIL/NOT_RUN immediately after, so the manifest is always
# truthful about what has actually happened, including for an interrupted
# run (audit repair -- see aggregate_exp02_umma_throughput.py's
# SELF_TEST_TRANSITIONS for the enforced per-method lifecycle).
write_self_test_outcomes() {
    local umma_1sm_outcome="$1" umma_2sm_outcome="$2"
    local merge_file
    merge_file="$(mktemp "${CAMPAIGN_DIR}/manifest_merge.XXXXXX")"
    printf '{"self_test_outcomes": {"umma_1sm": "%s", "umma_2sm": "%s"}}\n' \
        "${umma_1sm_outcome}" "${umma_2sm_outcome}" >| "${merge_file}"
    python3 "${AGGREGATOR_HOST}" manifest-write --campaign-dir "${CAMPAIGN_REL}" \
        --status IN_PROGRESS --merge-json "${merge_file}" >/dev/null
    local rc=$?
    rm -f "${merge_file}"
    return "${rc}"
}

write_manifest_progress() {
    local configurations_completed="$1"
    local samples_completed="$2"
    local merge_file
    merge_file="$(mktemp "${CAMPAIGN_DIR}/manifest_merge.XXXXXX")"
    printf '{"configuration_count_completed": %s, "sample_count_completed": %s}\n' \
        "${configurations_completed}" "${samples_completed}" >| "${merge_file}"
    python3 "${AGGREGATOR_HOST}" manifest-write --campaign-dir "${CAMPAIGN_REL}" \
        --status IN_PROGRESS --merge-json "${merge_file}" >/dev/null
    local rc=$?
    rm -f "${merge_file}"
    return "${rc}"
}

echo "run_exp02_umma_throughput: campaign ${CAMPAIGN_ID} IN_PROGRESS at ${CAMPAIGN_REL}" >&2

# Step 3-4: existing GPU-free compilation and SASS gates (P2.1/P2.2, both
# unmodified by P2.3); require the resulting artifacts.
if ! ( cd "${REPO_ROOT}" && make compute-umma-1sm-sass compute-umma-2sm-sass ); then
    write_manifest_status FAILED "compile_or_sass_gate" 1
    CAMPAIGN_OUTCOME=FAILED
    echo "run_exp02_umma_throughput: ERROR: GPU-free compilation/SASS gate failed" >&2
    exit 1
fi
# Symlink-safe (audit repair): require every execution artifact to be a
# regular, non-symlink file before the first scripts/run_container.sh
# invocation below (the self-tests). A plain `test -f` would follow a
# symlink and accept it; require_regular_nonsymlink_artifact rejects it
# outright via -L (lstat) first.
for artifact in build/compute/umma_1sm build/compute/umma_1sm.sass build/compute/umma_2sm build/compute/umma_2sm.sass; do
    if ! require_regular_nonsymlink_artifact "${REPO_ROOT}/${artifact}" "${artifact}"; then
        write_manifest_status FAILED "invalid_artifact" 1
        CAMPAIGN_OUTCOME=FAILED
        exit 1
    fi
done

# Step 5-6: both full binary self-tests, independently and sequentially,
# through the launcher. Continue only if both pass. Each method's outcome is
# recorded in the manifest immediately -- STARTED right before it runs, then
# PASS/FAIL right after -- so an interrupted or failed self-test is never
# later reported as passed, and a method that never runs because its
# sibling failed first is truthfully recorded as NOT_RUN rather than
# omitted or invented (audit repair).
SELF_TEST_UMMA_1SM_LOG_REL="${CAMPAIGN_REL}/logs/self_test_umma_1sm.stderr.log"
SELF_TEST_UMMA_2SM_LOG_REL="${CAMPAIGN_REL}/logs/self_test_umma_2sm.stderr.log"

if ! write_self_test_outcomes STARTED PENDING; then
    write_manifest_status FAILED "self_test_outcome_record" 1
    CAMPAIGN_OUTCOME=FAILED
    echo "run_exp02_umma_throughput: ERROR: could not record umma_1sm self-test start" >&2
    exit 1
fi
if ! "${RUN_CONTAINER}" build/compute/umma_1sm --self-test \
        >"${CAMPAIGN_DIR}/logs/self_test_umma_1sm.launcher.log" \
        2>"${CAMPAIGN_DIR}/logs/self_test_umma_1sm.stderr.log"; then
    write_self_test_outcomes FAIL NOT_RUN || true
    write_manifest_status FAILED "self_test_umma_1sm" 1 "${SELF_TEST_UMMA_1SM_LOG_REL}"
    CAMPAIGN_OUTCOME=FAILED
    echo "run_exp02_umma_throughput: ERROR: umma_1sm --self-test failed; see ${SELF_TEST_UMMA_1SM_LOG_REL%.stderr.log}.*.log" >&2
    exit 1
fi
SELF_TEST_UMMA_1SM=PASS
write_self_test_outcomes PASS PENDING

write_self_test_outcomes PASS STARTED
if ! "${RUN_CONTAINER}" build/compute/umma_2sm --self-test \
        >"${CAMPAIGN_DIR}/logs/self_test_umma_2sm.launcher.log" \
        2>"${CAMPAIGN_DIR}/logs/self_test_umma_2sm.stderr.log"; then
    write_self_test_outcomes PASS FAIL
    write_manifest_status FAILED "self_test_umma_2sm" 1 "${SELF_TEST_UMMA_2SM_LOG_REL}"
    CAMPAIGN_OUTCOME=FAILED
    echo "run_exp02_umma_throughput: ERROR: umma_2sm --self-test failed; see ${SELF_TEST_UMMA_2SM_LOG_REL%.stderr.log}.*.log" >&2
    exit 1
fi
SELF_TEST_UMMA_2SM=PASS
write_self_test_outcomes PASS PASS
echo "run_exp02_umma_throughput: both full self-tests PASS; starting the 24-configuration sweep" >&2

# Step 7-10: the 24 deterministic configurations, one GPU process at a time
# (never concurrent). Capture stdout as the case CSV; diagnostics/launcher
# output go to separate log files. Validate strictly immediately after
# capture, then record progress -- only after each case has passed.
PLAN_TSV="$(python3 "${AGGREGATOR_HOST}" plan --format lines)"
while IFS=$'\t' read -r p_index p_pair p_method p_cta p_m p_n p_k p_depth p_binary p_case_name; do
    [ -n "${p_index}" ] || continue

    bench_args=(--run-kind "${RUN_KIND}" --n "${p_n}" --depth "${p_depth}"
        --iterations "${ITERATIONS}" --warmup-iterations "${WARMUP_ITERATIONS}" --repetitions "${REPETITIONS}")

    launcher_log="${CAMPAIGN_DIR}/logs/${p_case_name}.launcher.log"
    stderr_log="${CAMPAIGN_DIR}/logs/${p_case_name}.stderr.log"
    launcher_log_rel="${CAMPAIGN_REL}/logs/${p_case_name}.launcher.log"
    stderr_log_rel="${CAMPAIGN_REL}/logs/${p_case_name}.stderr.log"

    echo "run_exp02_umma_throughput: [${p_index}/23] ${p_case_name} (pair ${p_pair}, ${p_method})" >&2
    if ! "${RUN_CONTAINER}" python3 "${AGGREGATOR_IN_CONTAINER}" capture \
            --campaign-dir "${CAMPAIGN_REL}" --out "cases/${p_case_name}.csv" -- \
            "${p_binary}" "${bench_args[@]}" \
            >"${launcher_log}" 2>"${stderr_log}"; then
        write_manifest_status FAILED "capture_${p_case_name}" 1 "${stderr_log_rel}"
        CAMPAIGN_OUTCOME=FAILED
        echo "run_exp02_umma_throughput: ERROR: capture failed at index ${p_index} (${p_case_name}); see ${launcher_log_rel} / ${stderr_log_rel}" >&2
        exit 1
    fi

    validate_args=(validate-case --campaign-dir "${CAMPAIGN_REL}" --index "${p_index}"
        --run-kind "${RUN_KIND}" --iterations "${ITERATIONS}" --warmup-iterations "${WARMUP_ITERATIONS}"
        --repetitions "${REPETITIONS}" --git-commit "${GIT_COMMIT}")
    if ! python3 "${AGGREGATOR_HOST}" "${validate_args[@]}" 2>>"${stderr_log}"; then
        write_manifest_status FAILED "validate_${p_case_name}" 1 "${stderr_log_rel}"
        CAMPAIGN_OUTCOME=FAILED
        echo "run_exp02_umma_throughput: ERROR: validation failed at index ${p_index} (${p_case_name}); see ${stderr_log_rel}" >&2
        exit 1
    fi

    configurations_completed=$((p_index + 1))
    samples_completed=$((configurations_completed * REPETITIONS))
    if ! write_manifest_progress "${configurations_completed}" "${samples_completed}"; then
        write_manifest_status FAILED "progress_${p_case_name}" 1 || true
        CAMPAIGN_OUTCOME=FAILED
        echo "run_exp02_umma_throughput: ERROR: could not record progress after ${p_case_name}" >&2
        exit 1
    fi
done <<< "${PLAN_TSV}"

# Step 11-12: consolidate, aggregate, and close the campaign. finalize
# re-validates the complete evidence set (execution_order.csv, every case
# file, cross-case consistency) before it may set status=COMPLETE.
COMPLETED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
finalize_args=(finalize --campaign-dir "${CAMPAIGN_REL}" --campaign-id "${CAMPAIGN_ID}"
    --run-kind "${RUN_KIND}" --iterations "${ITERATIONS}" --warmup-iterations "${WARMUP_ITERATIONS}"
    --repetitions "${REPETITIONS}" --git-commit "${GIT_COMMIT}" --gpu-index "${BLACKWELL_GPU_INDEX}"
    --started-at-utc "${STARTED_AT}" --completed-at-utc "${COMPLETED_AT}"
    --self-test-umma-1sm "${SELF_TEST_UMMA_1SM}" --self-test-umma-2sm "${SELF_TEST_UMMA_2SM}")

if ! python3 "${AGGREGATOR_HOST}" "${finalize_args[@]}"; then
    CAMPAIGN_OUTCOME=FAILED
    echo "run_exp02_umma_throughput: ERROR: finalize failed; campaign marked FAILED" >&2
    exit 1
fi

CAMPAIGN_OUTCOME=COMPLETE
echo "run_exp02_umma_throughput: campaign ${CAMPAIGN_ID} COMPLETE at ${CAMPAIGN_REL}" >&2
echo "run_exp02_umma_throughput: functional/descriptive output only; not a publishable performance result (P2.4 owns interpretation)" >&2
exit 0
