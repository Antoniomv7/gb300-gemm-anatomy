#!/usr/bin/env bash
# P1.3 joint LDGSTS/TMA sweep runner (exp01_memory_paths).
#
# Runs the frozen 18-invocation plan (2 methods x 3 stage counts x 3
# bytes-in-flight values, see scripts/aggregate_exp01_memory_paths.py)
# through scripts/run_container.sh, one GPU process at a time, then
# validates and aggregates the raw CSV. This script only orchestrates: plan
# generation, CSV parsing/validation, manifest handling, consolidation, and
# aggregation all live in scripts/aggregate_exp01_memory_paths.py (Python
# standard library only).
#
# Output is functional/descriptive infrastructure only: no speedups, no
# Nsight Compute, no performance conclusions. See src/memory/README.md and
# PLAN.md (P1.4 owns interpretation).
#
# Exit codes: 0 success/--help/--print-plan/--self-test; 1 execution,
# correctness, CSV-validation, or aggregation failure; 2 CLI,
# repository-state, or safety-precondition failure.
set -Eeuo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
AGGREGATOR_HOST="${REPO_ROOT}/scripts/aggregate_exp01_memory_paths.py"
AGGREGATOR_IN_CONTAINER="scripts/aggregate_exp01_memory_paths.py"
RUN_CONTAINER="${REPO_ROOT}/scripts/run_container.sh"
CASES_ROOT_REL="results/raw/exp01_memory_paths"

usage() {
    cat <<'EOF'
Usage:
  run_exp01_memory_paths.sh --help
  run_exp01_memory_paths.sh --print-plan
  run_exp01_memory_paths.sh --self-test
  run_exp01_memory_paths.sh --run-kind {smoke,benchmark} [--campaign-id ID] \
      [--working-set-mib N] --passes N --warmup-ms N --repetitions N

P1.3 joint LDGSTS/TMA sweep runner (exp01_memory_paths): the frozen
18-invocation plan (2 methods x 3 stage counts x 3 bytes-in-flight values)
through scripts/run_container.sh, with strict CSV validation and descriptive
aggregation. Functional/descriptive output only: no speedups, no Nsight
Compute, no performance conclusions (that is P1.4).

Options:
  --help                        Show this help and exit 0.
  --print-plan                  Print the deterministic 18-invocation plan
                                 and exit 0. No GPU, no Docker.
  --self-test                   Run GPU-free synthetic checks and exit. No
                                 GPU, no Docker, no nvidia-smi, no network.
  --run-kind {smoke,benchmark}   Required for a campaign.
  --campaign-id ID               Optional; default is the current UTC
                                 timestamp (YYYYMMDDTHHMMSSZ). Must match
                                 [A-Za-z0-9][A-Za-z0-9._-]{0,63}.
  --working-set-mib N             Optional, forwarded verbatim to both
                                 binaries, in [1, 1048576]. If omitted, both
                                 binaries apply their own default (>= 4x
                                 queried L2).
  --passes N                     Required for a campaign, in [1, 1000000].
  --warmup-ms N                  Required for a campaign, in [0, 3600000].
  --repetitions N                Required for a campaign, in [1, 1000000].

A real campaign additionally requires BLACKWELL_GPU_INDEX=<physical-index>
in the environment (never selected automatically) and a clean Git worktree
at a full 40-character commit. Every GPU invocation goes through
scripts/run_container.sh independently; methods never run concurrently.

Exit codes: 0 success/--help/--print-plan/--self-test; 1 execution,
correctness, CSV-validation, or aggregation failure; 2 CLI,
repository-state, or safety-precondition failure.
EOF
}

fail_cli() {
    echo "run_exp01_memory_paths: ERROR: $*" >&2
    usage >&2
    exit 2
}

fail_precondition() {
    echo "run_exp01_memory_paths: ERROR: $*" >&2
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

run_self_test() {
    local failures=0

    if [ -f "${REPO_ROOT}/VERSIONS.env" ]; then
        echo "run_exp01_memory_paths: self-test: PASS: repo root resolves to ${REPO_ROOT}" >&2
    else
        echo "run_exp01_memory_paths: self-test: FAIL: VERSIONS.env not found at resolved repo root" >&2
        failures=$((failures + 1))
    fi

    local plan_line_count
    plan_line_count="$(python3 "${AGGREGATOR_HOST}" plan --format lines | wc -l | tr -d ' ')"
    if [ "${plan_line_count}" -eq 18 ]; then
        echo "run_exp01_memory_paths: self-test: PASS: --print-plan has exactly 18 invocations" >&2
    else
        echo "run_exp01_memory_paths: self-test: FAIL: plan has ${plan_line_count} lines, expected 18" >&2
        failures=$((failures + 1))
    fi

    echo "run_exp01_memory_paths: self-test: delegating to the Python aggregator's synthetic test suite" >&2
    if python3 "${AGGREGATOR_HOST}" --self-test; then
        echo "run_exp01_memory_paths: self-test: PASS: aggregate_exp01_memory_paths.py --self-test" >&2
    else
        echo "run_exp01_memory_paths: self-test: FAIL: aggregate_exp01_memory_paths.py --self-test" >&2
        failures=$((failures + 1))
    fi

    if [ "${failures}" -eq 0 ]; then
        echo "run_exp01_memory_paths: SELF_TEST_RESULT=PASS" >&2
        return 0
    fi
    echo "run_exp01_memory_paths: SELF_TEST_RESULT=FAIL (${failures} failure(s))" >&2
    return 1
}

# ---------------------------------------------------------------------------
# CLI parsing: argument arrays only, no eval, no unquoted command
# construction. Every branch below must be reachable without touching the
# GPU, Docker, or nvidia-smi.
# ---------------------------------------------------------------------------
HELP=0
PRINT_PLAN=0
SELF_TEST=0
HAS_RUN_KIND=0; RUN_KIND=""
HAS_CAMPAIGN_ID=0; CAMPAIGN_ID=""
HAS_WORKING_SET_MIB=0; WORKING_SET_MIB=""
HAS_PASSES=0; PASSES=""
HAS_WARMUP_MS=0; WARMUP_MS=""
HAS_REPETITIONS=0; REPETITIONS=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --help|-h)
            HELP=1; shift ;;
        --print-plan)
            PRINT_PLAN=1; shift ;;
        --self-test)
            SELF_TEST=1; shift ;;
        --run-kind)
            [ "${HAS_RUN_KIND}" -eq 0 ] || fail_cli "duplicate --run-kind"
            [ "$#" -ge 2 ] || fail_cli "--run-kind requires a value"
            RUN_KIND="$2"; HAS_RUN_KIND=1; shift 2 ;;
        --campaign-id)
            [ "${HAS_CAMPAIGN_ID}" -eq 0 ] || fail_cli "duplicate --campaign-id"
            [ "$#" -ge 2 ] || fail_cli "--campaign-id requires a value"
            CAMPAIGN_ID="$2"; HAS_CAMPAIGN_ID=1; shift 2 ;;
        --working-set-mib)
            [ "${HAS_WORKING_SET_MIB}" -eq 0 ] || fail_cli "duplicate --working-set-mib"
            [ "$#" -ge 2 ] || fail_cli "--working-set-mib requires a value"
            WORKING_SET_MIB="$2"; HAS_WORKING_SET_MIB=1; shift 2 ;;
        --passes)
            [ "${HAS_PASSES}" -eq 0 ] || fail_cli "duplicate --passes"
            [ "$#" -ge 2 ] || fail_cli "--passes requires a value"
            PASSES="$2"; HAS_PASSES=1; shift 2 ;;
        --warmup-ms)
            [ "${HAS_WARMUP_MS}" -eq 0 ] || fail_cli "duplicate --warmup-ms"
            [ "$#" -ge 2 ] || fail_cli "--warmup-ms requires a value"
            WARMUP_MS="$2"; HAS_WARMUP_MS=1; shift 2 ;;
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

if [ "${HELP}" -eq 1 ]; then
    usage
    exit 0
fi

if [ "${PRINT_PLAN}" -eq 1 ]; then
    python3 "${AGGREGATOR_HOST}" plan --format text
    exit 0
fi

if [ "${SELF_TEST}" -eq 1 ]; then
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
[ "${HAS_PASSES}" -eq 1 ] || fail_cli "--passes is required"
[ "${HAS_WARMUP_MS}" -eq 1 ] || fail_cli "--warmup-ms is required"
[ "${HAS_REPETITIONS}" -eq 1 ] || fail_cli "--repetitions is required"

validate_range "--passes" "${PASSES}" 1 1000000
validate_range "--warmup-ms" "${WARMUP_MS}" 0 3600000
validate_range "--repetitions" "${REPETITIONS}" 1 1000000
if [ "${HAS_WORKING_SET_MIB}" -eq 1 ]; then
    validate_range "--working-set-mib" "${WORKING_SET_MIB}" 1 1048576
fi

if [ "${HAS_CAMPAIGN_ID}" -eq 1 ]; then
    [[ "${CAMPAIGN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$ ]] \
        || fail_cli "--campaign-id must match [A-Za-z0-9][A-Za-z0-9._-]{0,63}, got '${CAMPAIGN_ID}'"
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

CAMPAIGN_REL="${CASES_ROOT_REL}/${CAMPAIGN_ID}"
CAMPAIGN_DIR="${REPO_ROOT}/${CAMPAIGN_REL}"
mkdir -p "${REPO_ROOT}/${CASES_ROOT_REL}"
if ! mkdir "${CAMPAIGN_DIR}" 2>/dev/null; then
    fail_precondition "campaign directory already exists, refusing to overwrite: ${CAMPAIGN_DIR}"
fi
mkdir -p "${CAMPAIGN_DIR}/cases" "${CAMPAIGN_DIR}/logs"

CAMPAIGN_OUTCOME=""
on_exit() {
    local rc=$?
    if [ -z "${CAMPAIGN_OUTCOME}" ] && [ -d "${CAMPAIGN_DIR}" ]; then
        echo "run_exp01_memory_paths: unexpected termination (rc=${rc}); marking campaign INTERRUPTED" >&2
        CAMPAIGN_OUTCOME=INTERRUPTED
        write_manifest_status INTERRUPTED "unexpected_termination" "${rc}" || true
    fi
}
on_signal() {
    local sig="$1"
    trap - EXIT INT TERM
    if [ -z "${CAMPAIGN_OUTCOME}" ] && [ -d "${CAMPAIGN_DIR}" ]; then
        echo "run_exp01_memory_paths: received ${sig}; marking campaign INTERRUPTED" >&2
        CAMPAIGN_OUTCOME=INTERRUPTED
        write_manifest_status INTERRUPTED "signal_${sig}" "" || true
    fi
    exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap on_exit EXIT

write_manifest_status() {
    local status="$1" failure_stage="${2:-}" failure_exit_code="${3:-}"
    local merge_file
    merge_file="$(mktemp "${CAMPAIGN_DIR}/manifest_merge.XXXXXX")"
    if [ -n "${failure_stage}" ]; then
        printf '{"failure_stage": "%s", "failure_exit_code": %s}\n' \
            "${failure_stage}" "${failure_exit_code:-null}" > "${merge_file}"
    else
        printf '{}' > "${merge_file}"
    fi
    python3 "${AGGREGATOR_HOST}" manifest-write --campaign-dir "${CAMPAIGN_REL}" \
        --status "${status}" --merge-json "${merge_file}" >/dev/null
    rm -f "${merge_file}"
}

if [ "${HAS_WORKING_SET_MIB}" -eq 1 ]; then
    WORKING_SET_MIB_JSON="${WORKING_SET_MIB}"
else
    WORKING_SET_MIB_JSON="null"
fi
STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
INIT_MERGE_FILE="$(mktemp "${CAMPAIGN_DIR}/manifest_merge.XXXXXX")"
cat > "${INIT_MERGE_FILE}" <<EOF
{
  "campaign_id": "${CAMPAIGN_ID}",
  "run_kind": "${RUN_KIND}",
  "started_at_utc": "${STARTED_AT}",
  "configuration_count_expected": 18,
  "configuration_count_completed": 0,
  "sample_count_expected": $((18 * REPETITIONS)),
  "sample_count_completed": 0,
  "requested": {
    "run_kind": "${RUN_KIND}",
    "working_set_mib": ${WORKING_SET_MIB_JSON},
    "passes": ${PASSES},
    "warmup_ms": ${WARMUP_MS},
    "repetitions": ${REPETITIONS},
    "campaign_id": "${CAMPAIGN_ID}"
  },
  "selected_gpu_index": ${BLACKWELL_GPU_INDEX},
  "git_commit": "${GIT_COMMIT}",
  "git_dirty": false
}
EOF
python3 "${AGGREGATOR_HOST}" manifest-write --campaign-dir "${CAMPAIGN_REL}" \
    --status IN_PROGRESS --merge-json "${INIT_MERGE_FILE}" >/dev/null
rm -f "${INIT_MERGE_FILE}"

echo "run_exp01_memory_paths: campaign ${CAMPAIGN_ID} IN_PROGRESS at ${CAMPAIGN_DIR}" >&2

# Step 3-4: existing GPU-free compilation and SASS gates; require artifacts.
if ! ( cd "${REPO_ROOT}" && make memory-ldgsts-sass memory-tma-sass ); then
    write_manifest_status FAILED "compile_or_sass_gate" 1
    CAMPAIGN_OUTCOME=FAILED
    echo "run_exp01_memory_paths: ERROR: GPU-free compilation/SASS gate failed" >&2
    exit 1
fi
for artifact in build/memory/ldgsts build/memory/ldgsts.sass build/memory/tma build/memory/tma.sass; do
    if [ ! -f "${REPO_ROOT}/${artifact}" ]; then
        write_manifest_status FAILED "missing_artifact" 1
        CAMPAIGN_OUTCOME=FAILED
        echo "run_exp01_memory_paths: ERROR: expected artifact missing: ${artifact}" >&2
        exit 1
    fi
done

# Step 5-6: both full binary self-tests, independently, through the launcher.
if ! "${RUN_CONTAINER}" build/memory/ldgsts --self-test \
        >"${CAMPAIGN_DIR}/logs/self_test_ldgsts.launcher.log" \
        2>"${CAMPAIGN_DIR}/logs/self_test_ldgsts.stderr.log"; then
    write_manifest_status FAILED "self_test_ldgsts" 1
    CAMPAIGN_OUTCOME=FAILED
    echo "run_exp01_memory_paths: ERROR: ldgsts --self-test failed; see ${CAMPAIGN_DIR}/logs/self_test_ldgsts.*.log" >&2
    exit 1
fi
SELF_TEST_LDGSTS=PASS

if ! "${RUN_CONTAINER}" build/memory/tma --self-test \
        >"${CAMPAIGN_DIR}/logs/self_test_tma.launcher.log" \
        2>"${CAMPAIGN_DIR}/logs/self_test_tma.stderr.log"; then
    write_manifest_status FAILED "self_test_tma" 1
    CAMPAIGN_OUTCOME=FAILED
    echo "run_exp01_memory_paths: ERROR: tma --self-test failed; see ${CAMPAIGN_DIR}/logs/self_test_tma.*.log" >&2
    exit 1
fi
SELF_TEST_TMA=PASS

SELFTEST_MERGE_FILE="$(mktemp "${CAMPAIGN_DIR}/manifest_merge.XXXXXX")"
printf '{"self_test_outcomes": {"ldgsts": "%s", "tma": "%s"}}\n' \
    "${SELF_TEST_LDGSTS}" "${SELF_TEST_TMA}" > "${SELFTEST_MERGE_FILE}"
python3 "${AGGREGATOR_HOST}" manifest-write --campaign-dir "${CAMPAIGN_REL}" \
    --status IN_PROGRESS --merge-json "${SELFTEST_MERGE_FILE}" >/dev/null
rm -f "${SELFTEST_MERGE_FILE}"
echo "run_exp01_memory_paths: both full self-tests PASS; starting the 18-configuration sweep" >&2

# Step 7: the 18 deterministic configurations, one GPU process at a time.
PLAN_TSV="$(python3 "${AGGREGATOR_HOST}" plan --format lines)"
while IFS=$'\t' read -r p_index p_method p_stages p_bif p_case_name; do
    [ -n "${p_index}" ] || continue

    case "${p_method}" in
        ldgsts) bin_rel="build/memory/ldgsts" ;;
        tma) bin_rel="build/memory/tma" ;;
        *) echo "run_exp01_memory_paths: internal error: unknown method '${p_method}'" >&2; exit 1 ;;
    esac

    bench_args=(--stages "${p_stages}" --bytes-in-flight-kib "${p_bif}" --run-kind "${RUN_KIND}")
    if [ "${HAS_WORKING_SET_MIB}" -eq 1 ]; then
        bench_args+=(--working-set-mib "${WORKING_SET_MIB}")
    fi
    bench_args+=(--passes "${PASSES}" --warmup-ms "${WARMUP_MS}" --repetitions "${REPETITIONS}")

    launcher_log="${CAMPAIGN_DIR}/logs/${p_case_name}.launcher.log"
    stderr_log="${CAMPAIGN_DIR}/logs/${p_case_name}.stderr.log"

    echo "run_exp01_memory_paths: [${p_index}/17] ${p_case_name}" >&2
    if ! "${RUN_CONTAINER}" python3 "${AGGREGATOR_IN_CONTAINER}" capture \
            --campaign-dir "${CAMPAIGN_REL}" --out "cases/${p_case_name}.csv" -- \
            "${bin_rel}" "${bench_args[@]}" \
            >"${launcher_log}" 2>"${stderr_log}"; then
        write_manifest_status FAILED "capture_${p_case_name}" 1
        CAMPAIGN_OUTCOME=FAILED
        echo "run_exp01_memory_paths: ERROR: capture failed at index ${p_index} (${p_case_name}); see ${launcher_log} / ${stderr_log}" >&2
        exit 1
    fi

    if ! python3 "${AGGREGATOR_HOST}" validate-case \
            --campaign-dir "${CAMPAIGN_REL}" --index "${p_index}" \
            --run-kind "${RUN_KIND}" --repetitions "${REPETITIONS}" \
            --passes "${PASSES}" --warmup-ms "${WARMUP_MS}" --git-commit "${GIT_COMMIT}" \
            2>>"${stderr_log}"; then
        write_manifest_status FAILED "validate_${p_case_name}" 1
        CAMPAIGN_OUTCOME=FAILED
        echo "run_exp01_memory_paths: ERROR: validation failed at index ${p_index} (${p_case_name}); see ${stderr_log}" >&2
        exit 1
    fi
done <<< "${PLAN_TSV}"

# Step: consolidate, aggregate, and close the campaign.
COMPLETED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
finalize_args=(finalize --campaign-dir "${CAMPAIGN_REL}" --campaign-id "${CAMPAIGN_ID}"
    --run-kind "${RUN_KIND}" --repetitions "${REPETITIONS}" --passes "${PASSES}"
    --warmup-ms "${WARMUP_MS}" --git-commit "${GIT_COMMIT}" --gpu-index "${BLACKWELL_GPU_INDEX}"
    --started-at-utc "${STARTED_AT}" --completed-at-utc "${COMPLETED_AT}"
    --self-test-ldgsts "${SELF_TEST_LDGSTS}" --self-test-tma "${SELF_TEST_TMA}")
if [ "${HAS_WORKING_SET_MIB}" -eq 1 ]; then
    finalize_args+=(--working-set-mib "${WORKING_SET_MIB}")
fi

if ! python3 "${AGGREGATOR_HOST}" "${finalize_args[@]}"; then
    CAMPAIGN_OUTCOME=FAILED
    echo "run_exp01_memory_paths: ERROR: finalize failed; campaign marked FAILED" >&2
    exit 1
fi

CAMPAIGN_OUTCOME=COMPLETE
echo "run_exp01_memory_paths: campaign ${CAMPAIGN_ID} COMPLETE at ${CAMPAIGN_DIR}" >&2
echo "run_exp01_memory_paths: functional/descriptive output only; not a publishable performance result (P1.4 owns interpretation)" >&2
exit 0
