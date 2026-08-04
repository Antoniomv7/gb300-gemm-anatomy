#!/usr/bin/env bash
# P2.4 pilot/profile orchestrator for exp02_umma_throughput (BF16 UMMA
# throughput: profiling and empirical ceiling candidate).
#
# Adds a reproducible layer around the audited, unmodified P2.3 infrastructure
# (scripts/run_exp02_umma_throughput.sh, scripts/aggregate_exp02_umma_throughput.py)
# and the new, separately-audited P2.4 helper
# (scripts/analyze_exp02_umma_throughput_p24.py). This script never
# reimplements the P2.3 sweep: --pilot shells out to the unmodified P2.3
# runner with frozen parameters. It never runs Nsight Compute with a
# clock-controlling default, never selects a GPU automatically, and never
# collects a second, independent sweep. See src/compute/P2_4_PROTOCOL.md for
# the complete frozen protocol this file encodes.
#
# GPU-free modes (no Docker, no GPU, no nvidia-smi, no network):
#   --help, --print-plan, --self-test
#
# GPU-executing modes (go exclusively through scripts/run_container.sh for
# anything that touches the selected GPU; the GPU-free NCU-help-capability
# probe and .ncu-rep -> exported CSV step happen entirely inside
# scripts/p24_ncu_bridge.py, itself launched via scripts/run_container.sh):
#   --pilot   (requires BLACKWELL_GPU_INDEX, P2_4_CAMPAIGN_ID, P2_4_PREFLIGHT_SUMMARY)
#   --profile (requires BLACKWELL_GPU_INDEX, P2_4_CAMPAIGN_ID, P2_4_PREFLIGHT_SUMMARY)
#
# Exit codes: 0 success/--help/--print-plan/--self-test; 1 execution,
# validation, or NCU-collection failure; 2 CLI, repository-state, or
# safety-precondition failure.
set -Eeuo pipefail
set -o noclobber

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SELF_PATH="${REPO_ROOT}/scripts/run_exp02_umma_throughput_p24.sh"
P23_RUNNER="${REPO_ROOT}/scripts/run_exp02_umma_throughput.sh"
P23_AGGREGATOR="${REPO_ROOT}/scripts/aggregate_exp02_umma_throughput.py"
P24_ANALYZER_HOST="${REPO_ROOT}/scripts/analyze_exp02_umma_throughput_p24.py"
P24_SAFE_CAPTURE="${REPO_ROOT}/scripts/p24_safe_capture.py"
P24_NCU_BRIDGE_IN_CONTAINER="scripts/p24_ncu_bridge.py"
RUN_CONTAINER="${REPO_ROOT}/scripts/run_container.sh"
IMAGE_TAG="${IMAGE_TAG:-gb300-gemm-anatomy:phase0}"

usage() {
    cat <<'EOF'
Usage:
  run_exp02_umma_throughput_p24.sh --help
  run_exp02_umma_throughput_p24.sh --print-plan
  run_exp02_umma_throughput_p24.sh --self-test
  BLACKWELL_GPU_INDEX=<i> P2_4_CAMPAIGN_ID=<YYYYMMDDTHHMMSSZ> \
      P2_4_PREFLIGHT_SUMMARY=<path> run_exp02_umma_throughput_p24.sh --pilot
  BLACKWELL_GPU_INDEX=<i> P2_4_CAMPAIGN_ID=<YYYYMMDDTHHMMSSZ> \
      P2_4_PREFLIGHT_SUMMARY=<path> run_exp02_umma_throughput_p24.sh --profile

P2.4 profiling/TFLOP/s/scaling/saturation/empirical-ceiling orchestrator.
Reuses the audited P2.3 runner unmodified for the frozen 24-configuration
pilot; adds Nsight Compute profiling of the same 24 configurations. See
src/compute/P2_4_PROTOCOL.md for the complete frozen protocol.

Options:
  --help                Show this help and exit 0. Standalone only.
  --print-plan           Print the frozen P2.3 24-invocation pilot plan and
                          the frozen P2.4 24-case profile plan, then exit 0.
                          No GPU, no Docker. Standalone.
  --self-test             Run GPU-free synthetic checks and exit. No GPU, no
                          Docker, no nvidia-smi, no network, no real raw
                          results. Standalone.
  --pilot                 Run the frozen 24-configuration run_kind=benchmark
                          pilot through the unmodified P2.3 runner. Requires
                          BLACKWELL_GPU_INDEX, P2_4_CAMPAIGN_ID (an explicit
                          canonical UTC timestamp YYYYMMDDTHHMMSSZ), and
                          P2_4_PREFLIGHT_SUMMARY (a fresh, matching preflight
                          summary.json).
  --profile               Profile exactly the same 24 configurations with
                          Nsight Compute against an already-PILOT_COMPLETE
                          P2.4 campaign. Same three required inputs as
                          --pilot.

--help, --print-plan, and --self-test are mutually exclusive with each other
and with --pilot/--profile, and each may be given at most once.

Exit codes: 0 success/--help/--print-plan/--self-test; 1 execution,
validation, or NCU-collection failure; 2 CLI, repository-state, or
safety-precondition failure.
EOF
}

fail_cli() {
    echo "run_exp02_umma_throughput_p24: ERROR: $*" >&2
    usage >&2
    exit 2
}

fail_precondition() {
    echo "run_exp02_umma_throughput_p24: ERROR: $*" >&2
    exit 2
}

fail_run() {
    echo "run_exp02_umma_throughput_p24: ERROR: $*" >&2
    exit 1
}

# Failure metadata is rendered in one place and exercised by --self-test.
render_p24_failure_merge_json() {
    local failure_stage="${1:-}"
    if [ -z "${failure_stage}" ]; then
        printf '{}\n'
        return 0
    fi
    [[ "${failure_stage}" =~ ^[A-Za-z0-9_.-]+$ ]] \
        || return 2
    printf '{"failure_stage": "%s", "failure_detail": []}\n' "${failure_stage}"
}

# ---------------------------------------------------------------------------
# NCU CLI capability gate: a pure function over already-captured `ncu --help`
# text (a file path), so --self-test can exercise it against synthetic
# fixtures without ever invoking Docker or NCU. Only --profile mode ever
# calls this against a real captured `ncu --help`. Extends P1.4's identical
# gate with the two flags the frozen P2.4 profile protocol additionally
# requires: --launch-skip and --kernel-name (exact-symbol filtering).
# ---------------------------------------------------------------------------
check_ncu_help_capability() {
    local help_file="$1"
    local missing=0
    local pat flag val

    for pat in \
        -- '--clock-control' '--pipeline-boost-state' '--cache-control' \
        '--kernel-name-base' '--kernel-name' '--launch-skip' '--launch-count' \
        '--devices' '--replay-mode' '--print-summary' '--query-metrics' '--metrics' \
        '--csv' '--page' '--print-units' \
        '--print-kernel-base' '--log-file' '--import' '--export'
    do
        [ "${pat}" = "--" ] && continue
        if ! grep -qF -- "${pat}" "${help_file}"; then
            echo "run_exp02_umma_throughput_p24: NCU capability gate: MISSING flag '${pat}'" >&2
            missing=1
        fi
    done

    for pat in \
        "clock-control:none" "pipeline-boost-state:dynamic" \
        "cache-control:none" "kernel-name-base:function"
    do
        flag="${pat%%:*}"
        val="${pat##*:}"
        if ! grep -A 10 -F -- "--${flag} arg" "${help_file}" | grep -qw -- "${val}"; then
            echo "run_exp02_umma_throughput_p24: NCU capability gate: --${flag} does not list required value '${val}'" >&2
            missing=1
        fi
    done

    if ! grep -A 10 -F -- "--print-kernel-base arg" "${help_file}" | grep -qF -- "--kernel-name-base"; then
        echo "run_exp02_umma_throughput_p24: NCU capability gate: --print-kernel-base does not reference --kernel-name-base" >&2
        missing=1
    fi

    [ "${missing}" -eq 0 ]
}

# ---------------------------------------------------------------------------
# CLI parsing: special modes only. --pilot/--profile take no flags of their
# own; every input is an explicit, operator-provided environment variable.
# ---------------------------------------------------------------------------
HELP_COUNT=0
PRINT_PLAN_COUNT=0
SELF_TEST_COUNT=0
PILOT_COUNT=0
PROFILE_COUNT=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --help|-h) HELP_COUNT=$((HELP_COUNT + 1)); shift ;;
        --print-plan) PRINT_PLAN_COUNT=$((PRINT_PLAN_COUNT + 1)); shift ;;
        --self-test) SELF_TEST_COUNT=$((SELF_TEST_COUNT + 1)); shift ;;
        --pilot) PILOT_COUNT=$((PILOT_COUNT + 1)); shift ;;
        --profile) PROFILE_COUNT=$((PROFILE_COUNT + 1)); shift ;;
        -*) fail_cli "unknown option: $1" ;;
        *) fail_cli "unexpected positional argument: $1" ;;
    esac
done

MODE_COUNT=$((HELP_COUNT + PRINT_PLAN_COUNT + SELF_TEST_COUNT + PILOT_COUNT + PROFILE_COUNT))
if [ "${HELP_COUNT}" -gt 1 ] || [ "${PRINT_PLAN_COUNT}" -gt 1 ] || [ "${SELF_TEST_COUNT}" -gt 1 ] \
        || [ "${PILOT_COUNT}" -gt 1 ] || [ "${PROFILE_COUNT}" -gt 1 ]; then
    fail_cli "--help, --print-plan, --self-test, --pilot, and --profile may each be given at most once"
fi
if [ "${MODE_COUNT}" -eq 0 ]; then
    fail_cli "exactly one of --help, --print-plan, --self-test, --pilot, --profile is required"
fi
if [ "${MODE_COUNT}" -gt 1 ]; then
    fail_cli "--help, --print-plan, --self-test, --pilot, and --profile are mutually exclusive"
fi

if [ "${HELP_COUNT}" -eq 1 ]; then
    usage
    exit 0
fi

if [ "${PRINT_PLAN_COUNT}" -eq 1 ]; then
    echo "== frozen P2.3 24-invocation pilot plan (reused unmodified) =="
    python3 "${P23_AGGREGATOR}" plan --format text
    echo
    echo "== frozen P2.4 24-case profile plan (same configurations, plus kernel_symbol) =="
    python3 "${P24_ANALYZER_HOST}" plan --format text
    exit 0
fi

# ---------------------------------------------------------------------------
# --self-test: GPU-free only.
# ---------------------------------------------------------------------------
_check_rejected_cli() {
    local label="$1"
    shift
    "${SELF_PATH}" "$@" >/dev/null 2>&1
    local rc=$?
    if [ "${rc}" -eq 2 ]; then
        echo "run_exp02_umma_throughput_p24: self-test: PASS: rejects ${label} (exit 2)" >&2
        return 0
    fi
    echo "run_exp02_umma_throughput_p24: self-test: FAIL: ${label} exited ${rc}, expected 2" >&2
    return 1
}

run_self_test() {
    local failures=0

    if [ -f "${REPO_ROOT}/VERSIONS.env" ]; then
        echo "run_exp02_umma_throughput_p24: self-test: PASS: repo root resolves to ${REPO_ROOT}" >&2
    else
        echo "run_exp02_umma_throughput_p24: self-test: FAIL: VERSIONS.env not found at resolved repo root" >&2
        failures=$((failures + 1))
    fi

    local failure_payload
    failure_payload="$(render_p24_failure_merge_json "signal_TERM")"
    if python3 -c '
import json
import sys
doc = json.load(sys.stdin)
raise SystemExit(0 if doc == {"failure_stage": "signal_TERM", "failure_detail": []} else 1)
' <<< "${failure_payload}"; then
        echo "run_exp02_umma_throughput_p24: self-test: PASS: failure telemetry renders failure_detail as list[str]" >&2
    else
        echo "run_exp02_umma_throughput_p24: self-test: FAIL: failure telemetry JSON is malformed" >&2
        failures=$((failures + 1))
    fi
    if render_p24_failure_merge_json '../unsafe' >/dev/null 2>&1; then
        echo "run_exp02_umma_throughput_p24: self-test: FAIL: failure telemetry accepted an unsafe stage label" >&2
        failures=$((failures + 1))
    else
        echo "run_exp02_umma_throughput_p24: self-test: PASS: failure telemetry rejects an unsafe stage label" >&2
    fi

    local p23_plan_lines p24_plan_lines
    p23_plan_lines="$(python3 "${P23_AGGREGATOR}" plan --format lines | wc -l | tr -d ' ')"
    if [ "${p23_plan_lines}" -eq 24 ]; then
        echo "run_exp02_umma_throughput_p24: self-test: PASS: P2.3 pilot plan has exactly 24 invocations" >&2
    else
        echo "run_exp02_umma_throughput_p24: self-test: FAIL: P2.3 plan has ${p23_plan_lines} lines, expected 24" >&2
        failures=$((failures + 1))
    fi
    p24_plan_lines="$(python3 "${P24_ANALYZER_HOST}" plan --format lines | wc -l | tr -d ' ')"
    if [ "${p24_plan_lines}" -eq 24 ]; then
        echo "run_exp02_umma_throughput_p24: self-test: PASS: P2.4 profile plan has exactly 24 cases" >&2
    else
        echo "run_exp02_umma_throughput_p24: self-test: FAIL: P2.4 profile plan has ${p24_plan_lines} lines, expected 24" >&2
        failures=$((failures + 1))
    fi

    echo "run_exp02_umma_throughput_p24: self-test: delegating to the Python analyzer's synthetic test suite" >&2
    if python3 "${P24_ANALYZER_HOST}" --self-test; then
        echo "run_exp02_umma_throughput_p24: self-test: PASS: analyze_exp02_umma_throughput_p24.py --self-test" >&2
    else
        echo "run_exp02_umma_throughput_p24: self-test: FAIL: analyze_exp02_umma_throughput_p24.py --self-test" >&2
        failures=$((failures + 1))
    fi

    local ncu_tmp
    ncu_tmp="$(mktemp -d)"
    trap 'rm -rf "${ncu_tmp}"' RETURN
    cat > "${ncu_tmp}/good_help.txt" <<'EOF'
  --cache-control arg (=all)            Control the behavior of the GPU caches during profiling. Allowed values:
                                          all
                                          none
  --clock-control arg (=base)           Control the behavior of the GPU clocks during profiling. Allowed values:
                                          base
                                          boost
                                          force-boost
                                          none
                                          reset
  --pipeline-boost-state arg (=stable)  Control the Tensor Core boost state.
                                          stable
                                          dynamic
  --devices arg                         Specify the devices to enable profiling on.
  -k [ --kernel-name ] arg              Filter the kernel in one of the following ways:
  --kernel-name-base arg (=function)    Set the basis for --kernel-name:
                                          function
                                          demangled
                                          mangled
  -c [ --launch-count ] arg             Limit the number of collected profile results.
  -s [ --launch-skip ] arg              Skip the number of launches before starting to profile.
  --replay-mode arg (=kernel)           Mechanism used for replaying a kernel launch multiple times.
  --print-summary arg (=per-kernel)     Adds a summary at the end of the report.
  --query-metrics                       Query available metrics for devices on the system.
  --metrics arg                         Specify all metrics to be profiled, separated by comma.
  --log-file arg                        Send all tool output to the specified file.
  -o [ --export ] arg                   Set the output file for writing the profile results.
  -i [ --import ] arg                   Set the input file for reading profile results.
  --csv                                 Use comma-separated values in the output.
  --page arg (=details)                 Select report page to output.
  --print-units arg (=auto)             Set scaling of metric units.
  --print-kernel-base arg (=demangled)  Set the basis for kernel name output. See
                                          --kernel-name-base for available options.
EOF
    if check_ncu_help_capability "${ncu_tmp}/good_help.txt"; then
        echo "run_exp02_umma_throughput_p24: self-test: PASS: NCU capability gate accepts a complete synthetic --help" >&2
    else
        echo "run_exp02_umma_throughput_p24: self-test: FAIL: NCU capability gate rejected a complete synthetic --help" >&2
        failures=$((failures + 1))
    fi
    grep -v -- '--launch-skip' "${ncu_tmp}/good_help.txt" > "${ncu_tmp}/missing_launch_skip.txt"
    if check_ncu_help_capability "${ncu_tmp}/missing_launch_skip.txt"; then
        echo "run_exp02_umma_throughput_p24: self-test: FAIL: NCU capability gate accepted a --help missing --launch-skip" >&2
        failures=$((failures + 1))
    else
        echo "run_exp02_umma_throughput_p24: self-test: PASS: NCU capability gate rejects a --help missing --launch-skip" >&2
    fi
    grep -v -- '--kernel-name-base' "${ncu_tmp}/good_help.txt" > "${ncu_tmp}/missing_flag.txt"
    if check_ncu_help_capability "${ncu_tmp}/missing_flag.txt"; then
        echo "run_exp02_umma_throughput_p24: self-test: FAIL: NCU capability gate accepted a --help missing --kernel-name-base" >&2
        failures=$((failures + 1))
    else
        echo "run_exp02_umma_throughput_p24: self-test: PASS: NCU capability gate rejects a --help missing --kernel-name-base" >&2
    fi
    sed 's/dynamic/xyz/' "${ncu_tmp}/good_help.txt" > "${ncu_tmp}/missing_value.txt"
    if check_ncu_help_capability "${ncu_tmp}/missing_value.txt"; then
        echo "run_exp02_umma_throughput_p24: self-test: FAIL: NCU capability gate accepted a --help missing the 'dynamic' value" >&2
        failures=$((failures + 1))
    else
        echo "run_exp02_umma_throughput_p24: self-test: PASS: NCU capability gate rejects a --help missing the 'dynamic' value" >&2
    fi
    sed 's/--kernel-name-base for available options/kernel-name output options/' \
        "${ncu_tmp}/good_help.txt" > "${ncu_tmp}/missing_print_kernel_base_reference.txt"
    if check_ncu_help_capability "${ncu_tmp}/missing_print_kernel_base_reference.txt"; then
        echo "run_exp02_umma_throughput_p24: self-test: FAIL: NCU capability gate accepted --print-kernel-base without its --kernel-name-base reference" >&2
        failures=$((failures + 1))
    else
        echo "run_exp02_umma_throughput_p24: self-test: PASS: NCU capability gate rejects --print-kernel-base without its --kernel-name-base reference" >&2
    fi
    rm -rf "${ncu_tmp}"
    trap - RETURN

    echo "run_exp02_umma_throughput_p24: self-test: delegating to the safe-capture module's own synthetic/adversarial test suite" >&2
    if python3 "${P24_SAFE_CAPTURE}" --self-test; then
        echo "run_exp02_umma_throughput_p24: self-test: PASS: p24_safe_capture.py --self-test" >&2
    else
        echo "run_exp02_umma_throughput_p24: self-test: FAIL: p24_safe_capture.py --self-test" >&2
        failures=$((failures + 1))
    fi

    echo "run_exp02_umma_throughput_p24: self-test: delegating to the NCU bridge's own synthetic test suite" >&2
    if python3 "${REPO_ROOT}/scripts/p24_ncu_bridge.py" --self-test; then
        echo "run_exp02_umma_throughput_p24: self-test: PASS: p24_ncu_bridge.py --self-test" >&2
    else
        echo "run_exp02_umma_throughput_p24: self-test: FAIL: p24_ncu_bridge.py --self-test" >&2
        failures=$((failures + 1))
    fi

    _check_rejected_cli "'--help --help'" --help --help || failures=$((failures + 1))
    _check_rejected_cli "'-h --help'" -h --help || failures=$((failures + 1))
    _check_rejected_cli "'--print-plan --print-plan'" --print-plan --print-plan || failures=$((failures + 1))
    _check_rejected_cli "'--self-test --self-test'" --self-test --self-test || failures=$((failures + 1))
    _check_rejected_cli "'--self-test' followed by positional argument 'plan'" --self-test plan || failures=$((failures + 1))
    _check_rejected_cli "'--pilot --pilot'" --pilot --pilot || failures=$((failures + 1))
    _check_rejected_cli "'--profile --profile'" --profile --profile || failures=$((failures + 1))
    _check_rejected_cli "'--help --print-plan'" --help --print-plan || failures=$((failures + 1))
    _check_rejected_cli "'--pilot --profile'" --pilot --profile || failures=$((failures + 1))
    _check_rejected_cli "'--self-test --pilot'" --self-test --pilot || failures=$((failures + 1))
    _check_rejected_cli "'--print-plan --profile'" --print-plan --profile || failures=$((failures + 1))
    _check_rejected_cli "no arguments" || failures=$((failures + 1))
    _check_rejected_cli "unknown option" --bogus-option || failures=$((failures + 1))
    _check_rejected_cli "unexpected positional argument" positional-arg || failures=$((failures + 1))

    if [ "${failures}" -eq 0 ]; then
        echo "run_exp02_umma_throughput_p24: SELF_TEST_RESULT=PASS" >&2
        return 0
    fi
    echo "run_exp02_umma_throughput_p24: SELF_TEST_RESULT=FAIL (${failures} failure(s))" >&2
    return 1
}

if [ "${SELF_TEST_COUNT}" -eq 1 ]; then
    if run_self_test; then
        exit 0
    fi
    exit 1
fi

# ---------------------------------------------------------------------------
# Everything below this point is a real --pilot or --profile attempt.
# ---------------------------------------------------------------------------
[ -n "${BLACKWELL_GPU_INDEX:-}" ] \
    || fail_precondition "BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index; this script never selects a GPU automatically"
[[ "${BLACKWELL_GPU_INDEX}" =~ ^[0-9]+$ ]] \
    || fail_precondition "BLACKWELL_GPU_INDEX must be a non-negative integer, got '${BLACKWELL_GPU_INDEX}'"
[ -n "${P2_4_CAMPAIGN_ID:-}" ] \
    || fail_precondition "P2_4_CAMPAIGN_ID must be set explicitly to a canonical UTC timestamp YYYYMMDDTHHMMSSZ"
[[ "${P2_4_CAMPAIGN_ID}" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] \
    || fail_precondition "P2_4_CAMPAIGN_ID='${P2_4_CAMPAIGN_ID}' must match YYYYMMDDTHHMMSSZ (a canonical UTC timestamp)"
[ -n "${P2_4_PREFLIGHT_SUMMARY:-}" ] \
    || fail_precondition "P2_4_PREFLIGHT_SUMMARY must be set explicitly to a preflight summary.json path"
[ -f "${P2_4_PREFLIGHT_SUMMARY}" ] \
    || fail_precondition "P2_4_PREFLIGHT_SUMMARY='${P2_4_PREFLIGHT_SUMMARY}' does not exist or is not a regular file"

GIT_STATUS="$(cd "${REPO_ROOT}" && git status --porcelain)"
[ -z "${GIT_STATUS}" ] || fail_precondition "worktree is not clean; commit or stash changes before running --pilot/--profile"
GIT_COMMIT="$(cd "${REPO_ROOT}" && git rev-parse HEAD)"
[[ "${GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
    || fail_precondition "unable to resolve a full 40-character Git commit SHA (got '${GIT_COMMIT}')"

CAMPAIGN_REL="results/raw/exp02_umma_throughput_p24/${P2_4_CAMPAIGN_ID}"
P23_CAMPAIGN_REL="results/raw/exp02_umma_throughput/${P2_4_CAMPAIGN_ID}"

readonly CSV_HEADER_LITERAL="schema_version,timestamp_utc,run_kind,publishable,method,sample_index,cta_group,m,n,k,depth,iterations,warmup_iterations,repetitions,umma_per_iteration,total_umma,flops_per_umma,total_flops,elapsed_cycles,cycles_per_umma,flops_per_cycle,threads_per_cta,grid_blocks,tmem_columns,operand_path,input_type,accumulator_type,correctness,mismatches,max_abs_error,gpu_name,gpu_uuid,compute_capability,cuda_driver_version,cuda_runtime_version,git_commit,git_dirty"

# Recovers the one clean application CSV (header + single data row) a
# profiled UMMA binary printed to its inherited stdout, which -- because
# `ncu --log-file` isolates NCU's own tool output elsewhere -- shares that
# stream only with scripts/run_container.sh's own allowlisted banner lines.
extract_application_csv() {
    local captured_stdout="$1" campaign_rel="$2" rel_case_dir="$3" out_name="$4"
    local header_line_no total_lines data_line_no
    header_line_no="$(grep -nFx -- "${CSV_HEADER_LITERAL}" "${captured_stdout}" | head -n1 | cut -d: -f1)"
    if [ -z "${header_line_no}" ]; then
        echo "run_exp02_umma_throughput_p24: could not find the expected CSV header in ${captured_stdout}" >&2
        return 1
    fi
    total_lines="$(wc -l < "${captured_stdout}" | tr -d ' ')"
    data_line_no=$((header_line_no + 1))
    if [ "${data_line_no}" -gt "${total_lines}" ]; then
        echo "run_exp02_umma_throughput_p24: CSV header found but no data row follows it in ${captured_stdout}" >&2
        return 1
    fi
    sed -n "${header_line_no},${data_line_no}p" "${captured_stdout}" \
        | python3 "${P24_SAFE_CAPTURE}" write \
            --campaign-dir "${campaign_rel}" --rel-dir "${rel_case_dir}" --name "${out_name}"
}

CAMPAIGN_DIR="${REPO_ROOT}/${CAMPAIGN_REL}"
CAMPAIGN_OUTCOME=""

write_p24_manifest_status() {
    local status="$1" failure_stage="${2:-}"
    local merge_file rc
    merge_file="$(mktemp)"
    render_p24_failure_merge_json "${failure_stage}" >| "${merge_file}"
    python3 "${P24_ANALYZER_HOST}" manifest-write --campaign-dir "${CAMPAIGN_REL}" --status "${status}" --merge-json "${merge_file}" >/dev/null
    rc=$?
    rm -f "${merge_file}"
    return "${rc}"
}

on_exit() {
    local rc=$?
    if [ -z "${CAMPAIGN_OUTCOME}" ] && [ -d "${CAMPAIGN_DIR}" ]; then
        echo "run_exp02_umma_throughput_p24: unexpected termination (rc=${rc}); marking P2.4 campaign INTERRUPTED" >&2
        CAMPAIGN_OUTCOME=INTERRUPTED
        write_p24_manifest_status INTERRUPTED "unexpected_termination" || true
    fi
}
on_signal() {
    local sig="$1"
    trap - EXIT INT TERM
    if [ -z "${CAMPAIGN_OUTCOME}" ] && [ -d "${CAMPAIGN_DIR}" ]; then
        echo "run_exp02_umma_throughput_p24: received ${sig}; marking P2.4 campaign INTERRUPTED" >&2
        CAMPAIGN_OUTCOME=INTERRUPTED
        write_p24_manifest_status INTERRUPTED "signal_${sig}" || true
    fi
    exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap on_exit EXIT

# ---------------------------------------------------------------------------
# --pilot: the frozen 24-configuration run_kind=benchmark pilot, run entirely
# through the unmodified P2.3 runner.
# ---------------------------------------------------------------------------
if [ "${PILOT_COUNT}" -eq 1 ]; then
    echo "run_exp02_umma_throughput_p24: --pilot: validating preflight ${P2_4_PREFLIGHT_SUMMARY}" >&2
    if ! python3 "${P24_ANALYZER_HOST}" validate-preflight \
            --preflight "${P2_4_PREFLIGHT_SUMMARY}" --expected-git-commit "${GIT_COMMIT}" >&2; then
        fail_run "preflight validation failed; see errors above; refusing to start the pilot"
    fi

    STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
    if ! CAMPAIGN_REL_OUT="$(python3 "${P24_ANALYZER_HOST}" init-campaign \
            --campaign-id "${P2_4_CAMPAIGN_ID}" --started-at-utc "${STARTED_AT}")"; then
        fail_precondition "P2.4 campaign initialization failed (unsafe campaign ID, symlink, or an existing campaign directory)"
    fi
    [ "${CAMPAIGN_REL_OUT}" = "${CAMPAIGN_REL}" ] \
        || fail_run "internal error: init-campaign returned '${CAMPAIGN_REL_OUT}', expected '${CAMPAIGN_REL}'"

    echo "run_exp02_umma_throughput_p24: --pilot: campaign ${P2_4_CAMPAIGN_ID} PILOT_IN_PROGRESS at ${CAMPAIGN_DIR}" >&2
    echo "run_exp02_umma_throughput_p24: --pilot: running the frozen 24-configuration benchmark pilot through scripts/run_exp02_umma_throughput.sh" >&2
    export BLACKWELL_GPU_INDEX
    if ! "${P23_RUNNER}" --run-kind benchmark --campaign-id "${P2_4_CAMPAIGN_ID}" \
            --iterations 1000 --warmup-iterations 10 --repetitions 30; then
        write_p24_manifest_status FAILED "p23_pilot_run" || true
        CAMPAIGN_OUTCOME=FAILED
        fail_run "the P2.3 pilot run failed; P2.4 campaign marked FAILED (the P2.3 campaign's own raw evidence, if any, is preserved under results/raw/exp02_umma_throughput/${P2_4_CAMPAIGN_ID}/)"
    fi

    COMPLETED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
    if ! python3 "${P24_ANALYZER_HOST}" record-pilot --campaign-dir "${CAMPAIGN_REL}" \
            --p23-campaign-dir "${P23_CAMPAIGN_REL}" --preflight "${P2_4_PREFLIGHT_SUMMARY}" \
            --git-commit "${GIT_COMMIT}" --completed-at-utc "${COMPLETED_AT}"; then
        CAMPAIGN_OUTCOME=FAILED
        fail_run "record-pilot validation failed; P2.4 campaign marked FAILED; see errors above"
    fi

    CAMPAIGN_OUTCOME=PILOT_COMPLETE
    echo "run_exp02_umma_throughput_p24: --pilot: campaign ${P2_4_CAMPAIGN_ID} PILOT_COMPLETE at ${CAMPAIGN_DIR}" >&2
    echo "run_exp02_umma_throughput_p24: functional/pilot output only; publishable=false; not yet independently audited or GB300-verified" >&2
    echo "run_exp02_umma_throughput_p24: next (after independent audit): BLACKWELL_GPU_INDEX=<i> P2_4_CAMPAIGN_ID=${P2_4_CAMPAIGN_ID} P2_4_PREFLIGHT_SUMMARY=<fresh-preflight> ${SELF_PATH} --profile" >&2
    exit 0
fi

# ---------------------------------------------------------------------------
# --profile: Nsight Compute on the same 24 configurations, against an
# already-PILOT_COMPLETE P2.4 campaign.
# ---------------------------------------------------------------------------
if [ "${PROFILE_COUNT}" -eq 1 ]; then
    echo "run_exp02_umma_throughput_p24: --profile: safely resolving the P2.4 campaign tree and checking profiling preconditions (GPU-free)" >&2
    VPP_RC=0
    python3 "${P24_ANALYZER_HOST}" validate-profile-preconditions \
        --campaign-dir "${CAMPAIGN_REL}" --preflight "${P2_4_PREFLIGHT_SUMMARY}" \
        --git-commit "${GIT_COMMIT}" >&2 || VPP_RC=$?
    if [ "${VPP_RC}" -eq 2 ]; then
        fail_precondition "P2.4 campaign ${P2_4_CAMPAIGN_ID} at ${CAMPAIGN_REL} does not safely resolve (missing, never completed --pilot, or an unsafe/symlinked campaign path); see errors above"
    elif [ "${VPP_RC}" -ne 0 ]; then
        fail_run "profiling preconditions not met (preflight validation failed, or its GPU/driver/commit does not match the pilot's); see errors above; refusing to start profiling"
    fi
    echo "run_exp02_umma_throughput_p24: --profile: campaign tree and profiling preconditions verified" >&2

    echo "run_exp02_umma_throughput_p24: --profile: verifying NCU CLI capability (GPU-free)" >&2
    NCU_HELP_LOG_NAME="ncu_help_capability_probe.log"
    if ! python3 "${P24_SAFE_CAPTURE}" run \
            --campaign-dir "${CAMPAIGN_REL}" --rel-dir logs \
            --stdout-name "${NCU_HELP_LOG_NAME}" --combine-stderr \
            -- docker run --rm \
                --network none \
                --security-opt no-new-privileges \
                --cap-drop ALL \
                --user "$(id -u):$(id -g)" \
                -e HOME=/tmp \
                "${IMAGE_TAG}" \
                ncu --help; then
        fail_run "could not run 'ncu --help' inside the pinned image; see ${CAMPAIGN_DIR}/logs/${NCU_HELP_LOG_NAME}"
    fi
    NCU_HELP_LOG="${CAMPAIGN_DIR}/logs/${NCU_HELP_LOG_NAME}"
    if ! check_ncu_help_capability "${NCU_HELP_LOG}"; then
        fail_run "the installed NCU CLI does not support a control/flag/value the frozen protocol requires; see ${NCU_HELP_LOG}; never falling back to an NCU default that could control clocks"
    fi
    echo "run_exp02_umma_throughput_p24: --profile: NCU CLI capability verified" >&2

    echo "run_exp02_umma_throughput_p24: --profile: discovering supported NCU metrics on logical device 0" >&2
    DISCOVERY_STDOUT_NAME="metric_discovery.stdout.log"
    DISCOVERY_STDERR_NAME="metric_discovery.stderr.log"
    export BLACKWELL_GPU_INDEX
    if ! python3 "${P24_SAFE_CAPTURE}" run \
            --campaign-dir "${CAMPAIGN_REL}" --rel-dir logs \
            --stdout-name "${DISCOVERY_STDOUT_NAME}" --stderr-name "${DISCOVERY_STDERR_NAME}" \
            -- "${RUN_CONTAINER}" ncu --query-metrics --query-metrics-mode all --devices 0; then
        write_p24_manifest_status FAILED "metric_discovery" || true
        CAMPAIGN_OUTCOME=FAILED
        fail_run "NCU metric discovery failed; see ${CAMPAIGN_DIR}/logs/${DISCOVERY_STDERR_NAME}"
    fi
    DISCOVERY_STDOUT="${CAMPAIGN_DIR}/logs/${DISCOVERY_STDOUT_NAME}"

    PROFILE_STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
    DISCOVER_METRICS_STDERR_NAME="discover_metrics.stderr.log"
    DISCOVER_METRICS_STDERR="${CAMPAIGN_DIR}/logs/${DISCOVER_METRICS_STDERR_NAME}"
    if ! RESOLVED_METRICS="$(python3 "${P24_SAFE_CAPTURE}" run \
            --campaign-dir "${CAMPAIGN_REL}" --rel-dir logs \
            --stderr-name "${DISCOVER_METRICS_STDERR_NAME}" \
            -- python3 "${P24_ANALYZER_HOST}" discover-metrics \
                --campaign-dir "${CAMPAIGN_REL}" --discovery-log "${DISCOVERY_STDOUT}" \
                --preflight "${P2_4_PREFLIGHT_SUMMARY}" --git-commit "${GIT_COMMIT}" \
                --started-at-utc "${PROFILE_STARTED_AT}")"; then
        cat "${DISCOVER_METRICS_STDERR}" >&2 2>/dev/null || true
        CAMPAIGN_OUTCOME=FAILED
        fail_run "discover-metrics failed; P2.4 campaign marked FAILED; see ${DISCOVER_METRICS_STDERR}"
    fi
    cat "${DISCOVER_METRICS_STDERR}" >&2
    [ -n "${RESOLVED_METRICS}" ] \
        || fail_run "discover-metrics resolved zero metrics; cannot proceed with NCU collection"
    echo "run_exp02_umma_throughput_p24: --profile: resolved metrics: ${RESOLVED_METRICS}" >&2

    PLAN_TSV="$(python3 "${P24_ANALYZER_HOST}" plan --format lines)"
    while IFS=$'\t' read -r p_index p_method p_n p_depth p_cta_group p_kernel_symbol p_case_name; do
        [ -n "${p_index}" ] || continue

        case "${p_method}" in
            umma_1sm) bin_rel="build/compute/umma_1sm" ;;
            umma_2sm) bin_rel="build/compute/umma_2sm" ;;
            *) fail_run "internal error: unknown method '${p_method}'" ;;
        esac

        rel_case_dir="profiles/${p_case_name}"
        case_dir="${CAMPAIGN_DIR}/${rel_case_dir}"
        if ! python3 "${P24_SAFE_CAPTURE}" mkdir-case \
                --campaign-dir "${CAMPAIGN_REL}" --case-name "${p_case_name}"; then
            fail_run "profile case directory already exists or could not be safely created: ${case_dir}"
        fi

        container_stdout_name="${p_case_name}.container_stdout.log"
        container_stderr_name="${p_case_name}.container_stderr.log"
        application_csv_name="${p_case_name}.application.csv"
        bundle_name="${p_case_name}.ncu_bridge_bundle.bin"
        bridge_stderr_name="${p_case_name}.ncu_bridge_stderr.log"

        # NCU never receives a campaign-relative pathname of any kind, for
        # either the collection or the metrics-export step:
        # scripts/p24_ncu_bridge.py runs entirely inside the container,
        # stages every NCU "-o"/"--log-file"/"--import" argument inside its
        # own private, non-host-mounted "/tmp", and emits a single
        # versioned, length-delimited bundle on its own stdout -- captured
        # here into an anchored partial, then decoded/republished by
        # "publish-bundle" below.
        echo "run_exp02_umma_throughput_p24: --profile: [${p_index}/23] collecting ${p_case_name} (${p_kernel_symbol}) via the NCU bridge" >&2
        if ! python3 "${P24_SAFE_CAPTURE}" run \
                --campaign-dir "${CAMPAIGN_REL}" --rel-dir "${rel_case_dir}" \
                --stdout-name "${bundle_name}" --stderr-name "${bridge_stderr_name}" \
                -- "${RUN_CONTAINER}" python3 "${P24_NCU_BRIDGE_IN_CONTAINER}" \
                    --metrics "${RESOLVED_METRICS}" --kernel-name "${p_kernel_symbol}" \
                    -- \
                    "${bin_rel}" --run-kind benchmark --n "${p_n}" --depth "${p_depth}" \
                        --iterations 1000 --warmup-iterations 0 --repetitions 1; then
            write_p24_manifest_status FAILED "profile_collect_${p_case_name}" || true
            CAMPAIGN_OUTCOME=FAILED
            fail_run "NCU bridge failed for ${p_case_name}; see ${case_dir}/${bridge_stderr_name}"
        fi

        # --bridge-stderr-name folds this already-captured orchestration
        # stderr into the published <case>.ncu_tool.log and removes the
        # standalone file, so the case directory ends up with exactly the
        # seven canonical artifacts -- never an eighth
        # <case>.ncu_bridge_stderr.log (audit repair).
        if ! python3 "${P24_SAFE_CAPTURE}" publish-bundle \
                --campaign-dir "${CAMPAIGN_REL}" --rel-dir "${rel_case_dir}" \
                --bundle-name "${bundle_name}" --bridge-stderr-name "${bridge_stderr_name}" \
                --names "${container_stdout_name}" "${container_stderr_name}" \
                        "${p_case_name}.ncu_tool.log" "${p_case_name}_report.ncu-rep" \
                        "${p_case_name}.metrics_raw.csv" "${p_case_name}.metrics_export_stderr.log"; then
            write_p24_manifest_status FAILED "profile_publish_bundle_${p_case_name}" || true
            CAMPAIGN_OUTCOME=FAILED
            fail_run "could not decode/publish the NCU bridge bundle for ${p_case_name}"
        fi
        container_stdout="${case_dir}/${container_stdout_name}"

        if ! extract_application_csv "${container_stdout}" "${CAMPAIGN_REL}" "${rel_case_dir}" "${application_csv_name}"; then
            write_p24_manifest_status FAILED "profile_extract_csv_${p_case_name}" || true
            CAMPAIGN_OUTCOME=FAILED
            fail_run "could not extract the application CSV for ${p_case_name} from ${container_stdout}"
        fi

        if ! python3 "${P24_ANALYZER_HOST}" validate-profile-case \
                --campaign-dir "${CAMPAIGN_REL}" --index "${p_index}" --git-commit "${GIT_COMMIT}"; then
            write_p24_manifest_status FAILED "profile_validate_${p_case_name}" || true
            CAMPAIGN_OUTCOME=FAILED
            fail_run "validate-profile-case failed for ${p_case_name}; P2.4 campaign marked FAILED"
        fi
    done <<< "${PLAN_TSV}"

    PROFILE_COMPLETED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
    if ! python3 "${P24_ANALYZER_HOST}" finalize-profile \
            --campaign-dir "${CAMPAIGN_REL}" --completed-at-utc "${PROFILE_COMPLETED_AT}"; then
        CAMPAIGN_OUTCOME=FAILED
        fail_run "finalize-profile failed; P2.4 campaign marked FAILED; see errors above"
    fi

    CAMPAIGN_OUTCOME=COMPLETE
    echo "run_exp02_umma_throughput_p24: --profile: campaign ${P2_4_CAMPAIGN_ID} COMPLETE at ${CAMPAIGN_DIR}" >&2
    echo "run_exp02_umma_throughput_p24: functional/pilot+profile output only; publishable=false" >&2
    echo "run_exp02_umma_throughput_p24: next: P2_4_CAMPAIGN_ID=${P2_4_CAMPAIGN_ID} make compute-umma-p24-analyze" >&2
    exit 0
fi

fail_cli "internal error: no mode matched after CLI parsing"
