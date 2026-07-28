#!/usr/bin/env bash
# P1.4 pilot/profile orchestrator for exp01_memory_paths (LDGSTS versus TMA).
#
# Adds a reproducible layer around the audited P1.3 infrastructure
# (scripts/run_exp01_memory_paths.sh, scripts/aggregate_exp01_memory_paths.py)
# and the new, separately-audited P1.4 helper
# (scripts/analyze_exp01_memory_paths_p14.py). This script never
# reimplements the P1.3 sweep: --pilot shells out to the unmodified P1.3
# runner with frozen parameters. It never runs Nsight Compute with a
# clock-controlling default, never selects a GPU automatically, and never
# collects a final (non-pilot) campaign. See src/memory/P1_4_PROTOCOL.md for
# the complete frozen protocol this file encodes.
#
# GPU-free modes (no Docker, no GPU, no nvidia-smi, no network):
#   --help, --print-plan, --self-test
#
# GPU-executing modes (go exclusively through scripts/run_container.sh for
# anything that touches the selected GPU; the one GPU-free Nsight Compute
# post-processing step -- .ncu-rep -> exported CSV -- instead uses a plain,
# unprivileged, network-disabled, non---gpus `docker run`, exactly like this
# repository's own check-env/memory-*-build targets):
#   --pilot   (requires BLACKWELL_GPU_INDEX, P1_4_CAMPAIGN_ID, P1_4_PREFLIGHT_SUMMARY)
#   --profile (requires BLACKWELL_GPU_INDEX, P1_4_CAMPAIGN_ID, P1_4_PREFLIGHT_SUMMARY)
#
# Exit codes: 0 success/--help/--print-plan/--self-test; 1 execution,
# validation, or NCU-collection failure; 2 CLI, repository-state, or
# safety-precondition failure.
set -Eeuo pipefail
set -o noclobber

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
SELF_PATH="${REPO_ROOT}/scripts/run_exp01_memory_paths_p14.sh"
P13_RUNNER="${REPO_ROOT}/scripts/run_exp01_memory_paths.sh"
P13_AGGREGATOR="${REPO_ROOT}/scripts/aggregate_exp01_memory_paths.py"
P14_ANALYZER_HOST="${REPO_ROOT}/scripts/analyze_exp01_memory_paths_p14.py"
P14_SAFE_CAPTURE="${REPO_ROOT}/scripts/p14_safe_capture.py"
P14_NCU_BRIDGE_IN_CONTAINER="scripts/p14_ncu_bridge.py"
RUN_CONTAINER="${REPO_ROOT}/scripts/run_container.sh"
IMAGE_TAG="${IMAGE_TAG:-gb300-gemm-anatomy:phase0}"

usage() {
    cat <<'EOF'
Usage:
  run_exp01_memory_paths_p14.sh --help
  run_exp01_memory_paths_p14.sh --print-plan
  run_exp01_memory_paths_p14.sh --self-test
  BLACKWELL_GPU_INDEX=<i> P1_4_CAMPAIGN_ID=<YYYYMMDDTHHMMSSZ> \
      P1_4_PREFLIGHT_SUMMARY=<path> run_exp01_memory_paths_p14.sh --pilot
  BLACKWELL_GPU_INDEX=<i> P1_4_CAMPAIGN_ID=<YYYYMMDDTHHMMSSZ> \
      P1_4_PREFLIGHT_SUMMARY=<path> run_exp01_memory_paths_p14.sh --profile

P1.4 profiling/HBM-validation/analysis/pilot orchestrator. Reuses the
audited P1.3 runner unmodified for the frozen 18-configuration pilot; adds
Nsight Compute validation of exactly six frozen cases. See
src/memory/P1_4_PROTOCOL.md for the complete frozen protocol.

Options:
  --help                Show this help and exit 0. Standalone only.
  --print-plan           Print the frozen 18-invocation P1.3 pilot plan and
                          the frozen six-case NCU plan, then exit 0. No GPU,
                          no Docker. Standalone.
  --self-test             Run GPU-free synthetic checks and exit. No GPU, no
                          Docker, no nvidia-smi, no network, no real raw
                          results. Standalone.
  --pilot                 Run the frozen 18-configuration run_kind=benchmark
                          pilot through the unmodified P1.3 runner. Requires
                          BLACKWELL_GPU_INDEX, P1_4_CAMPAIGN_ID (an explicit
                          canonical UTC timestamp YYYYMMDDTHHMMSSZ), and
                          P1_4_PREFLIGHT_SUMMARY (a fresh, matching preflight
                          summary.json).
  --profile               Profile exactly the six frozen cases with Nsight
                          Compute against an already-PILOT_COMPLETE P1.4
                          campaign. Same three required inputs as --pilot.

--help, --print-plan, and --self-test are mutually exclusive with each other
and with --pilot/--profile, and each may be given at most once.

Exit codes: 0 success/--help/--print-plan/--self-test; 1 execution,
validation, or NCU-collection failure; 2 CLI, repository-state, or
safety-precondition failure.
EOF
}

fail_cli() {
    echo "run_exp01_memory_paths_p14: ERROR: $*" >&2
    usage >&2
    exit 2
}

fail_precondition() {
    echo "run_exp01_memory_paths_p14: ERROR: $*" >&2
    exit 2
}

fail_run() {
    echo "run_exp01_memory_paths_p14: ERROR: $*" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# NCU CLI capability gate: a pure function over already-captured `ncu --help`
# text (a file path), so --self-test can exercise it against synthetic
# fixtures without ever invoking Docker or NCU. Only --profile mode ever
# calls this against a real captured `ncu --help`.
# ---------------------------------------------------------------------------
check_ncu_help_capability() {
    local help_file="$1"
    local missing=0
    local pat flag val

    # Required flags: a literal, unambiguous substring match is enough here.
    for pat in \
        -- '--clock-control' '--pipeline-boost-state' '--cache-control' \
        '--kernel-name-base' '--kernel-name' '--launch-count' \
        '--devices' '--replay-mode' '--query-metrics' '--metrics' \
        '--csv' '--page' '--print-metric-name' '--print-units' \
        '--print-kernel-base' '--log-file' '--import' '--export'
    do
        [ "${pat}" = "--" ] && continue
        if ! grep -qF -- "${pat}" "${help_file}"; then
            echo "run_exp01_memory_paths_p14: NCU capability gate: MISSING flag '${pat}'" >&2
            missing=1
        fi
    done

    # Required values: ncu --help wraps each flag's allowed-value list across
    # several lines below the flag's own line, so look within a fixed window
    # of lines following the flag (verified against the pinned NCU
    # 2025.4.0.0 --help output during implementation) rather than assuming a
    # single-line match or relying on grep-implementation-specific
    # multi-line dot-matches-newline behavior.
    for pat in \
        "clock-control:none" "pipeline-boost-state:dynamic" \
        "cache-control:none" "kernel-name-base:function" \
        "print-kernel-base:function"
    do
        flag="${pat%%:*}"
        val="${pat##*:}"
        if ! grep -A 10 -F -- "--${flag} arg" "${help_file}" | grep -qw -- "${val}"; then
            echo "run_exp01_memory_paths_p14: NCU capability gate: --${flag} does not list required value '${val}'" >&2
            missing=1
        fi
    done

    [ "${missing}" -eq 0 ]
}

# ---------------------------------------------------------------------------
# Safe capture (Remediation A, second audit): every raw-campaign write this
# script performs -- every NCU-help/metric-discovery/collection/export log
# and the extracted application CSV -- goes exclusively through
# scripts/p14_safe_capture.py, never a plain `>`/`>>`/`2>`/`2>>` shell
# redirection into results/raw/exp01_memory_paths_p14/. A precheck (even an
# `-L`-aware one) immediately before an ordinary redirection still leaves a
# TOCTOU window between the check and the later open(); p14_safe_capture.py
# closes that window structurally by opening every directory component with
# O_NOFOLLOW, relative to the previously opened descriptor, and never
# re-resolving a pathname afterwards -- see its own module docstring and
# self-test for the full design and the two adversarial race
# reproductions. See "P1.4 GPU-free synthetic/adversarial tests" below for
# where this module's own --self-test is exercised.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CLI parsing: special modes only. --pilot/--profile take no flags of their
# own; every input is an explicit, operator-provided environment variable
# (BLACKWELL_GPU_INDEX, P1_4_CAMPAIGN_ID, P1_4_PREFLIGHT_SUMMARY), exactly
# like scripts/run_exp01_memory_paths.sh's own BLACKWELL_GPU_INDEX contract.
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
    echo "== frozen P1.3 18-invocation pilot plan (reused unmodified) =="
    python3 "${P13_AGGREGATOR}" plan --format text
    echo
    echo "== frozen P1.4 six-case Nsight Compute plan =="
    python3 "${P14_ANALYZER_HOST}" plan --format text
    exit 0
fi

# ---------------------------------------------------------------------------
# --self-test: GPU-free only. Never invokes Docker, NCU, nvidia-smi, a CUDA
# binary, or creates/modifies results/raw/.
# ---------------------------------------------------------------------------
_check_rejected_cli() {
    local label="$1"
    shift
    "${SELF_PATH}" "$@" >/dev/null 2>&1
    local rc=$?
    if [ "${rc}" -eq 2 ]; then
        echo "run_exp01_memory_paths_p14: self-test: PASS: rejects ${label} (exit 2)" >&2
        return 0
    fi
    echo "run_exp01_memory_paths_p14: self-test: FAIL: ${label} exited ${rc}, expected 2" >&2
    return 1
}

run_self_test() {
    local failures=0

    if [ -f "${REPO_ROOT}/VERSIONS.env" ]; then
        echo "run_exp01_memory_paths_p14: self-test: PASS: repo root resolves to ${REPO_ROOT}" >&2
    else
        echo "run_exp01_memory_paths_p14: self-test: FAIL: VERSIONS.env not found at resolved repo root" >&2
        failures=$((failures + 1))
    fi

    local p13_plan_lines p14_plan_lines
    p13_plan_lines="$(python3 "${P13_AGGREGATOR}" plan --format lines | wc -l | tr -d ' ')"
    if [ "${p13_plan_lines}" -eq 18 ]; then
        echo "run_exp01_memory_paths_p14: self-test: PASS: P1.3 pilot plan has exactly 18 invocations" >&2
    else
        echo "run_exp01_memory_paths_p14: self-test: FAIL: P1.3 plan has ${p13_plan_lines} lines, expected 18" >&2
        failures=$((failures + 1))
    fi
    p14_plan_lines="$(python3 "${P14_ANALYZER_HOST}" plan --format lines | wc -l | tr -d ' ')"
    if [ "${p14_plan_lines}" -eq 6 ]; then
        echo "run_exp01_memory_paths_p14: self-test: PASS: P1.4 NCU plan has exactly 6 cases" >&2
    else
        echo "run_exp01_memory_paths_p14: self-test: FAIL: P1.4 NCU plan has ${p14_plan_lines} lines, expected 6" >&2
        failures=$((failures + 1))
    fi

    echo "run_exp01_memory_paths_p14: self-test: delegating to the Python analyzer's synthetic test suite" >&2
    if python3 "${P14_ANALYZER_HOST}" --self-test; then
        echo "run_exp01_memory_paths_p14: self-test: PASS: analyze_exp01_memory_paths_p14.py --self-test" >&2
    else
        echo "run_exp01_memory_paths_p14: self-test: FAIL: analyze_exp01_memory_paths_p14.py --self-test" >&2
        failures=$((failures + 1))
    fi

    # --- NCU capability gate: pure-function tests against synthetic fixture
    # text, never against a real `ncu --help` (which would require Docker).
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
  --replay-mode arg (=kernel)           Mechanism used for replaying a kernel launch multiple times.
  --query-metrics                       Query available metrics for devices on the system.
  --metrics arg                         Specify all metrics to be profiled, separated by comma.
  --log-file arg                        Send all tool output to the specified file.
  -o [ --export ] arg                   Set the output file for writing the profile results.
  -i [ --import ] arg                   Set the input file for reading profile results.
  --csv                                 Use comma-separated values in the output.
  --page arg (=details)                 Select report page to output.
  --print-metric-name arg (=label)      Select one of the option to show it in the Metric Name column.
  --print-units arg (=auto)             Set scaling of metric units.
  --print-kernel-base arg (=demangled)  Set the basis for kernel name output. See --kernel-name-base:
                                          function
                                          demangled
                                          mangled
EOF
    if check_ncu_help_capability "${ncu_tmp}/good_help.txt"; then
        echo "run_exp01_memory_paths_p14: self-test: PASS: NCU capability gate accepts a complete synthetic --help" >&2
    else
        echo "run_exp01_memory_paths_p14: self-test: FAIL: NCU capability gate rejected a complete synthetic --help" >&2
        failures=$((failures + 1))
    fi
    grep -v -- '--kernel-name-base' "${ncu_tmp}/good_help.txt" > "${ncu_tmp}/missing_flag.txt"
    if check_ncu_help_capability "${ncu_tmp}/missing_flag.txt"; then
        echo "run_exp01_memory_paths_p14: self-test: FAIL: NCU capability gate accepted a --help missing --kernel-name-base" >&2
        failures=$((failures + 1))
    else
        echo "run_exp01_memory_paths_p14: self-test: PASS: NCU capability gate rejects a --help missing --kernel-name-base" >&2
    fi
    sed 's/dynamic/xyz/' "${ncu_tmp}/good_help.txt" > "${ncu_tmp}/missing_value.txt"
    if check_ncu_help_capability "${ncu_tmp}/missing_value.txt"; then
        echo "run_exp01_memory_paths_p14: self-test: FAIL: NCU capability gate accepted a --help missing the 'dynamic' value" >&2
        failures=$((failures + 1))
    else
        echo "run_exp01_memory_paths_p14: self-test: PASS: NCU capability gate rejects a --help missing the 'dynamic' value" >&2
    fi
    grep -v -- 'print-kernel-base' "${ncu_tmp}/good_help.txt" > "${ncu_tmp}/missing_print_kernel_base.txt"
    if check_ncu_help_capability "${ncu_tmp}/missing_print_kernel_base.txt"; then
        echo "run_exp01_memory_paths_p14: self-test: FAIL: NCU capability gate accepted a --help missing --print-kernel-base" >&2
        failures=$((failures + 1))
    else
        echo "run_exp01_memory_paths_p14: self-test: PASS: NCU capability gate rejects a --help missing --print-kernel-base" >&2
    fi
    rm -rf "${ncu_tmp}"
    trap - RETURN

    echo "run_exp01_memory_paths_p14: self-test: delegating to the safe-capture module's own synthetic/adversarial test suite" >&2
    if python3 "${P14_SAFE_CAPTURE}" --self-test; then
        echo "run_exp01_memory_paths_p14: self-test: PASS: p14_safe_capture.py --self-test" >&2
    else
        echo "run_exp01_memory_paths_p14: self-test: FAIL: p14_safe_capture.py --self-test" >&2
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
        echo "run_exp01_memory_paths_p14: SELF_TEST_RESULT=PASS" >&2
        return 0
    fi
    echo "run_exp01_memory_paths_p14: SELF_TEST_RESULT=FAIL (${failures} failure(s))" >&2
    return 1
}

if [ "${SELF_TEST_COUNT}" -eq 1 ]; then
    if run_self_test; then
        exit 0
    fi
    exit 1
fi

# ---------------------------------------------------------------------------
# Everything below this point is a real --pilot or --profile attempt: GPU
# selection, repository state, Docker, and the results tree are now all in
# scope.
# ---------------------------------------------------------------------------
[ -n "${BLACKWELL_GPU_INDEX:-}" ] \
    || fail_precondition "BLACKWELL_GPU_INDEX must be set explicitly to a physical GPU index; this script never selects a GPU automatically"
[[ "${BLACKWELL_GPU_INDEX}" =~ ^[0-9]+$ ]] \
    || fail_precondition "BLACKWELL_GPU_INDEX must be a non-negative integer, got '${BLACKWELL_GPU_INDEX}'"
[ -n "${P1_4_CAMPAIGN_ID:-}" ] \
    || fail_precondition "P1_4_CAMPAIGN_ID must be set explicitly to a canonical UTC timestamp YYYYMMDDTHHMMSSZ"
[[ "${P1_4_CAMPAIGN_ID}" =~ ^[0-9]{8}T[0-9]{6}Z$ ]] \
    || fail_precondition "P1_4_CAMPAIGN_ID='${P1_4_CAMPAIGN_ID}' must match YYYYMMDDTHHMMSSZ (a canonical UTC timestamp)"
[ -n "${P1_4_PREFLIGHT_SUMMARY:-}" ] \
    || fail_precondition "P1_4_PREFLIGHT_SUMMARY must be set explicitly to a preflight summary.json path"
[ -f "${P1_4_PREFLIGHT_SUMMARY}" ] \
    || fail_precondition "P1_4_PREFLIGHT_SUMMARY='${P1_4_PREFLIGHT_SUMMARY}' does not exist or is not a regular file"

GIT_STATUS="$(cd "${REPO_ROOT}" && git status --porcelain)"
[ -z "${GIT_STATUS}" ] || fail_precondition "worktree is not clean; commit or stash changes before running --pilot/--profile"
GIT_COMMIT="$(cd "${REPO_ROOT}" && git rev-parse HEAD)"
[[ "${GIT_COMMIT}" =~ ^[0-9a-f]{40}$ ]] \
    || fail_precondition "unable to resolve a full 40-character Git commit SHA (got '${GIT_COMMIT}')"

CAMPAIGN_REL="results/raw/exp01_memory_paths_p14/${P1_4_CAMPAIGN_ID}"
P13_CAMPAIGN_REL="results/raw/exp01_memory_paths/${P1_4_CAMPAIGN_ID}"

# The exact 37-column CSV header both build/memory/ldgsts and build/memory/tma
# print, mirrored from aggregate_exp01_memory_paths.CSV_HEADER (never changed
# independently of it -- see check-static's cross-file consistency style).
readonly CSV_HEADER_LITERAL="schema_version,timestamp_utc,run_kind,method,sample_index,stages,tile_width_elements,tile_width_bytes,tile_height,stage_bytes,bytes_in_flight_per_sm,vector_bytes,copies_per_thread_per_stage,threads_per_cta,target_ctas_per_sm,occupancy_ctas_per_sm,grid_blocks,sm_count,smem_reservation_bytes,l2_bytes,requested_working_set_bytes,working_set_bytes,working_set_l2_ratio,passes,useful_bytes,warmup_ms,kernel_time_ms,effective_gbps,correctness,mismatches,gpu_name,gpu_uuid,compute_capability,cuda_driver_version,cuda_runtime_version,git_commit,git_dirty"

# Recovers the one clean application CSV (header + single data row) a
# profiled benchmark binary printed to its inherited stdout, which -- because
# `ncu --log-file` isolates NCU's own tool output elsewhere -- shares that
# stream only with scripts/run_container.sh's own allowlisted banner lines.
# Scans for the exact literal header rather than assuming the stream is
# otherwise pure. The extracted bytes are piped into p14_safe_capture.py's
# "write" subcommand -- never a shell "> out_path" -- so the actual
# publication into the anchored profiles/<case>/ directory is descriptor-
# anchored and no-clobber, exactly like every other P1.4 raw-tree write.
extract_application_csv() {
    local captured_stdout="$1" campaign_rel="$2" rel_case_dir="$3" out_name="$4"
    local header_line_no total_lines data_line_no
    header_line_no="$(grep -nFx -- "${CSV_HEADER_LITERAL}" "${captured_stdout}" | head -n1 | cut -d: -f1)"
    if [ -z "${header_line_no}" ]; then
        echo "run_exp01_memory_paths_p14: could not find the expected CSV header in ${captured_stdout}" >&2
        return 1
    fi
    total_lines="$(wc -l < "${captured_stdout}" | tr -d ' ')"
    data_line_no=$((header_line_no + 1))
    if [ "${data_line_no}" -gt "${total_lines}" ]; then
        echo "run_exp01_memory_paths_p14: CSV header found but no data row follows it in ${captured_stdout}" >&2
        return 1
    fi
    sed -n "${header_line_no},${data_line_no}p" "${captured_stdout}" \
        | python3 "${P14_SAFE_CAPTURE}" write \
            --campaign-dir "${campaign_rel}" --rel-dir "${rel_case_dir}" --name "${out_name}"
}

CAMPAIGN_DIR="${REPO_ROOT}/${CAMPAIGN_REL}"
CAMPAIGN_OUTCOME=""

write_p14_manifest_status() {
    local status="$1" failure_stage="${2:-}"
    local merge_file rc
    # Task 4 remediation (Section 10): this temporary previously lived
    # *inside* the campaign path (mktemp "${CAMPAIGN_DIR}/manifest_merge.
    # XXXXXX"), created and written to (">|") by ordinary path-string
    # operations rather than the descriptor-anchored primitives every other
    # P1.4 raw-tree write goes through -- a symlink swap of any ancestor
    # component between the mktemp and the later open() could redirect
    # either. The manifest-merge content itself is not campaign evidence; it
    # is a transient argument-passing mechanism between this script and
    # "manifest-write" (which reads it once via a plain path and then
    # publishes the *real* manifest revision through the existing
    # descriptor-anchored, hash-chained, no-clobber writer). Moving the
    # temporary to the system default temporary directory (plain "mktemp",
    # no directory argument) takes it out of the raw tree entirely, so no
    # raw-campaign path is ever mktemp'd or shell-redirected into.
    merge_file="$(mktemp)"
    if [ -n "${failure_stage}" ]; then
        printf '{"failure_stage": "%s", "failure_detail": null}\n' "${failure_stage}" >| "${merge_file}"
    else
        printf '{}' >| "${merge_file}"
    fi
    python3 "${P14_ANALYZER_HOST}" manifest-write --campaign-dir "${CAMPAIGN_REL}" \
        --status "${status}" --merge-json "${merge_file}" >/dev/null
    rc=$?
    rm -f "${merge_file}"
    return "${rc}"
}

on_exit() {
    local rc=$?
    if [ -z "${CAMPAIGN_OUTCOME}" ] && [ -d "${CAMPAIGN_DIR}" ]; then
        echo "run_exp01_memory_paths_p14: unexpected termination (rc=${rc}); marking P1.4 campaign INTERRUPTED" >&2
        CAMPAIGN_OUTCOME=INTERRUPTED
        write_p14_manifest_status INTERRUPTED "unexpected_termination" || true
    fi
}
on_signal() {
    local sig="$1"
    trap - EXIT INT TERM
    if [ -z "${CAMPAIGN_OUTCOME}" ] && [ -d "${CAMPAIGN_DIR}" ]; then
        echo "run_exp01_memory_paths_p14: received ${sig}; marking P1.4 campaign INTERRUPTED" >&2
        CAMPAIGN_OUTCOME=INTERRUPTED
        write_p14_manifest_status INTERRUPTED "signal_${sig}" || true
    fi
    exit 130
}
trap 'on_signal INT' INT
trap 'on_signal TERM' TERM
trap on_exit EXIT

# ---------------------------------------------------------------------------
# --pilot: the frozen 18-configuration run_kind=benchmark pilot, run entirely
# through the unmodified P1.3 runner. This script never reimplements the
# P1.3 sweep, CSV validation, correctness checks, aggregation, or manifest
# rules.
# ---------------------------------------------------------------------------
if [ "${PILOT_COUNT}" -eq 1 ]; then
    echo "run_exp01_memory_paths_p14: --pilot: validating preflight ${P1_4_PREFLIGHT_SUMMARY}" >&2
    if ! python3 "${P14_ANALYZER_HOST}" validate-preflight \
            --preflight "${P1_4_PREFLIGHT_SUMMARY}" --expected-git-commit "${GIT_COMMIT}" >&2; then
        fail_run "preflight validation failed; see errors above; refusing to start the pilot"
    fi

    STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
    if ! CAMPAIGN_REL_OUT="$(python3 "${P14_ANALYZER_HOST}" init-campaign \
            --campaign-id "${P1_4_CAMPAIGN_ID}" --started-at-utc "${STARTED_AT}")"; then
        fail_precondition "P1.4 campaign initialization failed (unsafe campaign ID, symlink, or an existing campaign directory)"
    fi
    [ "${CAMPAIGN_REL_OUT}" = "${CAMPAIGN_REL}" ] \
        || fail_run "internal error: init-campaign returned '${CAMPAIGN_REL_OUT}', expected '${CAMPAIGN_REL}'"

    echo "run_exp01_memory_paths_p14: --pilot: campaign ${P1_4_CAMPAIGN_ID} PILOT_IN_PROGRESS at ${CAMPAIGN_DIR}" >&2
    echo "run_exp01_memory_paths_p14: --pilot: running the frozen 18-configuration benchmark pilot through scripts/run_exp01_memory_paths.sh" >&2
    export BLACKWELL_GPU_INDEX
    if ! "${P13_RUNNER}" --run-kind benchmark --campaign-id "${P1_4_CAMPAIGN_ID}" \
            --working-set-mib 512 --passes 32 --warmup-ms 2000 --repetitions 30; then
        write_p14_manifest_status FAILED "p13_pilot_run" || true
        CAMPAIGN_OUTCOME=FAILED
        fail_run "the P1.3 pilot run failed; P1.4 campaign marked FAILED (the P1.3 campaign's own raw evidence, if any, is preserved under results/raw/exp01_memory_paths/${P1_4_CAMPAIGN_ID}/)"
    fi

    COMPLETED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
    if ! python3 "${P14_ANALYZER_HOST}" record-pilot --campaign-dir "${CAMPAIGN_REL}" \
            --p13-campaign-dir "${P13_CAMPAIGN_REL}" --preflight "${P1_4_PREFLIGHT_SUMMARY}" \
            --git-commit "${GIT_COMMIT}" --completed-at-utc "${COMPLETED_AT}"; then
        CAMPAIGN_OUTCOME=FAILED
        fail_run "record-pilot validation failed; P1.4 campaign marked FAILED; see errors above"
    fi

    CAMPAIGN_OUTCOME=PILOT_COMPLETE
    echo "run_exp01_memory_paths_p14: --pilot: campaign ${P1_4_CAMPAIGN_ID} PILOT_COMPLETE at ${CAMPAIGN_DIR}" >&2
    echo "run_exp01_memory_paths_p14: functional/pilot output only; publishable=false; not yet independently audited or NCU-validated" >&2
    echo "run_exp01_memory_paths_p14: next (after independent audit): BLACKWELL_GPU_INDEX=<i> P1_4_CAMPAIGN_ID=${P1_4_CAMPAIGN_ID} P1_4_PREFLIGHT_SUMMARY=<fresh-preflight> ${SELF_PATH} --profile" >&2
    exit 0
fi

# ---------------------------------------------------------------------------
# --profile: Nsight Compute on exactly the six frozen cases, against an
# already-PILOT_COMPLETE P1.4 campaign. Every GPU-touching NCU invocation
# goes through scripts/run_container.sh; the GPU-free .ncu-rep -> CSV export
# and the GPU-free `ncu --help` capability probe instead use a plain,
# unprivileged, network-disabled, non---gpus `docker run`, mirroring
# check-env/memory-*-build.
# ---------------------------------------------------------------------------
if [ "${PROFILE_COUNT}" -eq 1 ]; then
    # Safely resolve the complete P1.4 campaign tree (symlink-safe at every
    # component, exactly as the Python analyzer's own resolve_p14_campaign_dir
    # requires: campaign_dir plus profiles/, analysis/, logs/, manifest/) and
    # confirm the campaign is PILOT_COMPLETE with a profiling preflight whose
    # GPU/driver/commit matches the pilot's -- all *before* any Docker/NCU
    # invocation or raw-tree log write (audit blockers #1 and #4). A plain
    # `[ -d ... ]` check would follow a symlinked campaign directory instead
    # of rejecting it, so this reuses the same audited, symlink-safe
    # resolution and preflight-provenance comparison the Python analyzer
    # itself enforces again (as a hard gate) inside discover-metrics.
    echo "run_exp01_memory_paths_p14: --profile: safely resolving the P1.4 campaign tree and checking profiling preconditions (GPU-free)" >&2
    VPP_RC=0
    python3 "${P14_ANALYZER_HOST}" validate-profile-preconditions \
        --campaign-dir "${CAMPAIGN_REL}" --preflight "${P1_4_PREFLIGHT_SUMMARY}" \
        --git-commit "${GIT_COMMIT}" >&2 || VPP_RC=$?
    if [ "${VPP_RC}" -eq 2 ]; then
        fail_precondition "P1.4 campaign ${P1_4_CAMPAIGN_ID} at ${CAMPAIGN_REL} does not safely resolve (missing, never completed --pilot, or an unsafe/symlinked campaign path); see errors above"
    elif [ "${VPP_RC}" -ne 0 ]; then
        fail_run "profiling preconditions not met (preflight validation failed, or its GPU/driver/commit does not match the pilot's); see errors above; refusing to start profiling"
    fi
    echo "run_exp01_memory_paths_p14: --profile: campaign tree and profiling preconditions verified" >&2

    echo "run_exp01_memory_paths_p14: --profile: verifying NCU CLI capability (GPU-free)" >&2
    NCU_HELP_LOG_NAME="ncu_help_capability_probe.log"
    if ! python3 "${P14_SAFE_CAPTURE}" run \
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
    echo "run_exp01_memory_paths_p14: --profile: NCU CLI capability verified" >&2

    echo "run_exp01_memory_paths_p14: --profile: discovering supported NCU metrics on logical device 0" >&2
    DISCOVERY_STDOUT_NAME="metric_discovery.stdout.log"
    DISCOVERY_STDERR_NAME="metric_discovery.stderr.log"
    export BLACKWELL_GPU_INDEX
    if ! python3 "${P14_SAFE_CAPTURE}" run \
            --campaign-dir "${CAMPAIGN_REL}" --rel-dir logs \
            --stdout-name "${DISCOVERY_STDOUT_NAME}" --stderr-name "${DISCOVERY_STDERR_NAME}" \
            -- "${RUN_CONTAINER}" ncu --query-metrics --query-metrics-mode all --devices 0; then
        write_p14_manifest_status FAILED "metric_discovery" || true
        CAMPAIGN_OUTCOME=FAILED
        fail_run "NCU metric discovery failed; see ${CAMPAIGN_DIR}/logs/${DISCOVERY_STDERR_NAME}"
    fi
    DISCOVERY_STDOUT="${CAMPAIGN_DIR}/logs/${DISCOVERY_STDOUT_NAME}"

    PROFILE_STARTED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
    DISCOVER_METRICS_STDERR_NAME="discover_metrics.stderr.log"
    DISCOVER_METRICS_STDERR="${CAMPAIGN_DIR}/logs/${DISCOVER_METRICS_STDERR_NAME}"
    if ! RESOLVED_METRICS="$(python3 "${P14_SAFE_CAPTURE}" run \
            --campaign-dir "${CAMPAIGN_REL}" --rel-dir logs \
            --stderr-name "${DISCOVER_METRICS_STDERR_NAME}" \
            -- python3 "${P14_ANALYZER_HOST}" discover-metrics \
                --campaign-dir "${CAMPAIGN_REL}" --discovery-log "${DISCOVERY_STDOUT}" \
                --preflight "${P1_4_PREFLIGHT_SUMMARY}" --git-commit "${GIT_COMMIT}" \
                --started-at-utc "${PROFILE_STARTED_AT}")"; then
        cat "${DISCOVER_METRICS_STDERR}" >&2 2>/dev/null || true
        CAMPAIGN_OUTCOME=FAILED
        fail_run "discover-metrics failed; P1.4 campaign marked FAILED; see ${DISCOVER_METRICS_STDERR}"
    fi
    cat "${DISCOVER_METRICS_STDERR}" >&2
    [ -n "${RESOLVED_METRICS}" ] \
        || fail_run "discover-metrics resolved zero metrics; cannot proceed with NCU collection"
    echo "run_exp01_memory_paths_p14: --profile: resolved metrics: ${RESOLVED_METRICS}" >&2

    PLAN_TSV="$(python3 "${P14_ANALYZER_HOST}" plan --format lines)"
    while IFS=$'\t' read -r p_index p_method p_stages p_bif p_kernel p_case_name; do
        [ -n "${p_index}" ] || continue

        case "${p_method}" in
            ldgsts) bin_rel="build/memory/ldgsts" ;;
            tma) bin_rel="build/memory/tma" ;;
            *) fail_run "internal error: unknown method '${p_method}'" ;;
        esac

        # Descriptor-anchored replacement for the old
        # "[ -L case_dir ] || [ -e case_dir ]; mkdir case_dir" pair, which was
        # itself racy against profiles/ (case_dir's *parent*) being swapped
        # for a symlink between the check and the mkdir.
        rel_case_dir="profiles/${p_case_name}"
        case_dir="${CAMPAIGN_DIR}/${rel_case_dir}"
        if ! python3 "${P14_SAFE_CAPTURE}" mkdir-case \
                --campaign-dir "${CAMPAIGN_REL}" --case-name "${p_case_name}"; then
            fail_run "profile case directory already exists or could not be safely created: ${case_dir}"
        fi

        ncu_tool_log_name="${p_case_name}.ncu_tool.log"
        container_stdout_name="${p_case_name}.container_stdout.log"
        container_stderr_name="${p_case_name}.container_stderr.log"
        application_csv_name="${p_case_name}.application.csv"
        metrics_csv_name="${p_case_name}.metrics_raw.csv"
        metrics_export_stderr_name="${p_case_name}.metrics_export_stderr.log"
        ncu_rep_name="${p_case_name}_report.ncu-rep"
        bundle_name="${p_case_name}.ncu_bridge_bundle.bin"
        bridge_stderr_name="${p_case_name}.ncu_bridge_stderr.log"

        # Task 4 remediation (blockers A/B): NCU never receives a
        # campaign-relative pathname of any kind, for either the collection
        # or the metrics-export step. scripts/p14_ncu_bridge.py runs
        # entirely inside the container, stages every NCU "-o"/
        # "--log-file"/"--import" argument inside its own private,
        # non-host-mounted "/tmp", and emits a single versioned,
        # length-delimited bundle on its own stdout -- captured here into
        # an anchored partial exactly like any other P1.4 child-process
        # output, then decoded/republished by "publish-bundle" below.
        # (This replaces the previous design, which built
        # "profiles/<case>/<case>_report" relative to /workspace instead of
        # the campaign directory and handed that path to NCU's own "-o"/
        # "--log-file" -- and separately handed "--import" a raw campaign
        # ".ncu-rep" path in a second `docker run` -- so NCU itself opened
        # a raw-tree path for writing in three different places.)
        echo "run_exp01_memory_paths_p14: --profile: [${p_index}/5] collecting ${p_case_name} (${p_kernel}) via the NCU bridge" >&2
        if ! python3 "${P14_SAFE_CAPTURE}" run \
                --campaign-dir "${CAMPAIGN_REL}" --rel-dir "${rel_case_dir}" \
                --stdout-name "${bundle_name}" --stderr-name "${bridge_stderr_name}" \
                -- "${RUN_CONTAINER}" python3 "${P14_NCU_BRIDGE_IN_CONTAINER}" \
                    --metrics "${RESOLVED_METRICS}" --kernel-name "${p_kernel}" \
                    -- \
                    "${bin_rel}" --stages "${p_stages}" --bytes-in-flight-kib "${p_bif}" \
                    --run-kind benchmark --working-set-mib 512 --passes 32 --warmup-ms 0 --repetitions 1; then
            write_p14_manifest_status FAILED "profile_collect_${p_case_name}" || true
            CAMPAIGN_OUTCOME=FAILED
            fail_run "NCU bridge failed for ${p_case_name}; see ${case_dir}/${bridge_stderr_name}"
        fi

        if ! python3 "${P14_SAFE_CAPTURE}" publish-bundle \
                --campaign-dir "${CAMPAIGN_REL}" --rel-dir "${rel_case_dir}" \
                --bundle-name "${bundle_name}" \
                --names "${container_stdout_name}" "${container_stderr_name}" \
                        "${ncu_tool_log_name}" "${ncu_rep_name}" \
                        "${metrics_csv_name}" "${metrics_export_stderr_name}"; then
            write_p14_manifest_status FAILED "profile_publish_bundle_${p_case_name}" || true
            CAMPAIGN_OUTCOME=FAILED
            fail_run "could not decode/publish the NCU bridge bundle for ${p_case_name}"
        fi
        container_stdout="${case_dir}/${container_stdout_name}"

        if ! extract_application_csv "${container_stdout}" "${CAMPAIGN_REL}" "${rel_case_dir}" "${application_csv_name}"; then
            write_p14_manifest_status FAILED "profile_extract_csv_${p_case_name}" || true
            CAMPAIGN_OUTCOME=FAILED
            fail_run "could not extract the application CSV for ${p_case_name} from ${container_stdout}"
        fi

        # validate-profile-case derives every evidence path itself, from
        # --campaign-dir and --index alone (the frozen plan's own case
        # name), rather than trusting a caller-supplied --application-csv/
        # --metrics-csv/--ncu-rep string (Task 4, Section 7).
        if ! python3 "${P14_ANALYZER_HOST}" validate-profile-case \
                --campaign-dir "${CAMPAIGN_REL}" --index "${p_index}" --git-commit "${GIT_COMMIT}"; then
            write_p14_manifest_status FAILED "profile_validate_${p_case_name}" || true
            CAMPAIGN_OUTCOME=FAILED
            fail_run "validate-profile-case failed for ${p_case_name}; P1.4 campaign marked FAILED"
        fi
    done <<< "${PLAN_TSV}"

    PROFILE_COMPLETED_AT="$(date -u +%Y%m%dT%H%M%SZ)"
    if ! python3 "${P14_ANALYZER_HOST}" finalize-profile \
            --campaign-dir "${CAMPAIGN_REL}" --completed-at-utc "${PROFILE_COMPLETED_AT}"; then
        CAMPAIGN_OUTCOME=FAILED
        fail_run "finalize-profile failed; P1.4 campaign marked FAILED; see errors above"
    fi

    CAMPAIGN_OUTCOME=COMPLETE
    echo "run_exp01_memory_paths_p14: --profile: campaign ${P1_4_CAMPAIGN_ID} COMPLETE at ${CAMPAIGN_DIR}" >&2
    echo "run_exp01_memory_paths_p14: functional/pilot+profile output only; publishable=false" >&2
    echo "run_exp01_memory_paths_p14: next: P1_4_CAMPAIGN_ID=${P1_4_CAMPAIGN_ID} make memory-paths-p14-analyze" >&2
    exit 0
fi

fail_cli "internal error: no mode matched after CLI parsing"
