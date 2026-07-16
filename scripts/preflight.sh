#!/usr/bin/env bash
# Phase 0 preflight for gb300-gemm-anatomy.
#
# Runs INSIDE the pinned container, launched by scripts/run_container.sh
# (which guarantees exactly one GPU is visible as CUDA logical device 0 and
# exports EXPECTED_GPU_UUID; this script refuses to start without it).
#
# Check gating (dependent checks are SKIPped with an explicit reason):
#   cuda_smoke_compile  requires tool_versions PASS
#   cuda_smoke_run      requires gpu_visibility PASS and cuda_smoke_compile PASS
#   cutedsl_smoke       requires gpu_visibility PASS and tool_versions PASS
#   ncu_profile         requires gpu_visibility PASS and cuda_smoke_run PASS
# No CUDA, CuTe DSL, or NCU execution happens unless gpu_visibility is PASS.
#
# Checks (all required):
#   gpu_visibility      exactly one GPU, allowlisted fields, CC 10.3
#   tool_versions       nvcc, ptxas, cuobjdump, nvdisasm, ncu, python3, CuTe DSL
#   cuda_smoke_compile  nvcc -arch=sm_103a on smoke/cuda_smoke.cu
#   cuda_smoke_run      execute + numeric correctness + CC 10.3
#   cutedsl_smoke       JIT + execute + validate a real minimal CuTe DSL kernel
#   ncu_profile         ncu --set basic on the CUDA smoke binary
#
# Statuses: PASS | FAIL | BLOCKED | SKIP.
#   BLOCKED_NCU     -> ERR_NVGPUCTRPERM (profiling counters not permitted)
#   BLOCKED_DRIVER  -> cudaErrorUnsupportedPtxVersion or
#                      cudaErrorCallRequiresNewerDriver (driver too old for
#                      the pinned toolkit; never "fixed" automatically)
# Overall: any FAIL fails; otherwise any required BLOCKED blocks; PASS only
# if every required check passed. Unreached checks fail the run (fail closed).
#
# A summary (results/preflight/<UTC>/summary.json) is written by an EXIT trap
# once execution has started, even when a check fails mid-run.

set -u -o pipefail

# The launcher must have pinned the exact device; refuse to run without it.
if [ -z "${EXPECTED_GPU_UUID:-}" ]; then
    echo "preflight: ERROR: EXPECTED_GPU_UUID is not set." >&2
    echo "preflight: run this script only via scripts/run_container.sh." >&2
    exit 2
fi

SCHEMA_VERSION="1"
CUDA_ARCH="sm_103a"
REQUIRED_CC="10.3"
TS_UTC="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="results/preflight/${TS_UTC}"

CHECK_NAMES=(gpu_visibility tool_versions cuda_smoke_compile cuda_smoke_run cutedsl_smoke ncu_profile)
declare -A STATUS REASON
for c in "${CHECK_NAMES[@]}"; do
    STATUS[$c]="SKIP"
    REASON[$c]="NOT_REACHED"
done

mkdir -p "${OUT_DIR}" || { echo "preflight: cannot create ${OUT_DIR}" >&2; exit 2; }
GPU_KV="${OUT_DIR}/gpu.kv"
TOOLS_KV="${OUT_DIR}/tools.kv"
: > "${GPU_KV}"
: > "${TOOLS_KV}"

echo "preflight: writing to ${OUT_DIR}"

# --- finalization: always write summary.json once execution has started -----
finalize() {
    local overall="PASS" c
    for c in "${CHECK_NAMES[@]}"; do
        if [ "${STATUS[$c]}" = "FAIL" ]; then overall="FAIL"; fi
    done
    if [ "${overall}" != "FAIL" ]; then
        for c in "${CHECK_NAMES[@]}"; do
            if [ "${STATUS[$c]}" = "BLOCKED" ]; then overall="BLOCKED"; fi
        done
    fi
    if [ "${overall}" = "PASS" ]; then
        # Fail closed: a check that never ran (and was not skipped because a
        # dependency failed or blocked) must not yield an overall PASS.
        for c in "${CHECK_NAMES[@]}"; do
            if [ "${STATUS[$c]}" != "PASS" ]; then overall="FAIL"; fi
        done
    fi

    local git_commit git_dirty
    git_commit="$(git rev-parse --verify HEAD 2>/dev/null)" || git_commit="NO_COMMITS_YET"
    if git_status_out="$(git status --porcelain 2>/dev/null)"; then
        if [ -n "${git_status_out}" ]; then git_dirty="true"; else git_dirty="false"; fi
    else
        git_dirty="unknown"
    fi

    {
        for c in "${CHECK_NAMES[@]}"; do
            printf '%s\t%s\t%s\n' "$c" "${STATUS[$c]}" "${REASON[$c]}"
        done
    } > "${OUT_DIR}/checks.tsv"

    SCHEMA_VERSION="${SCHEMA_VERSION}" TS_UTC="${TS_UTC}" GIT_COMMIT="${git_commit}" \
    GIT_DIRTY="${git_dirty}" HOST_ARCH="$(uname -m)" OVERALL="${overall}" \
    OUT_DIR="${OUT_DIR}" GPU_KV="${GPU_KV}" TOOLS_KV="${TOOLS_KV}" \
    python3 - <<'PY'
import json, os

def read_kv(path):
    data = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if "=" in line:
                key, value = line.split("=", 1)
                data[key] = value
    return data

checks = []
with open(os.path.join(os.environ["OUT_DIR"], "checks.tsv"), encoding="utf-8") as fh:
    for line in fh:
        name, status, reason = line.rstrip("\n").split("\t")
        checks.append({"name": name, "required": True,
                       "status": status, "reason_code": reason})

git_dirty = {"true": True, "false": False}.get(os.environ["GIT_DIRTY"],
                                               os.environ["GIT_DIRTY"])
summary = {
    "schema_version": os.environ["SCHEMA_VERSION"],
    "timestamp_utc": os.environ["TS_UTC"],
    "git_commit": os.environ["GIT_COMMIT"],
    "git_dirty": git_dirty,
    "host_arch": os.environ["HOST_ARCH"],
    "tool_versions": read_kv(os.environ["TOOLS_KV"]),
    "gpu": read_kv(os.environ["GPU_KV"]),
    "checks": checks,
    "overall_status": os.environ["OVERALL"],
}
path = os.path.join(os.environ["OUT_DIR"], "summary.json")
with open(path, "w", encoding="utf-8") as fh:
    json.dump(summary, fh, indent=2)
    fh.write("\n")
print(f"preflight: summary written to {path}")
PY

    echo "preflight: ---- summary ----"
    for c in "${CHECK_NAMES[@]}"; do
        printf 'preflight:   %-20s %-8s %s\n' "$c" "${STATUS[$c]}" "${REASON[$c]}"
    done
    echo "preflight: OVERALL=${overall}"
    case "${overall}" in
        PASS) exit 0 ;;
        BLOCKED) exit 3 ;;
        *) exit 1 ;;
    esac
}
trap finalize EXIT

# Detects driver-vs-toolkit incompatibility markers in a log file.
log_has_driver_error() {
    grep -qE 'cudaErrorUnsupportedPtxVersion|cudaErrorCallRequiresNewerDriver|CUDA_ERROR_UNSUPPORTED_PTX_VERSION|CUDA_ERROR_CALL_REQUIRES_NEWER_DRIVER' "$1"
}

# --- check 1: gpu_visibility -------------------------------------------------
if ! command -v nvidia-smi >/dev/null 2>&1; then
    STATUS[gpu_visibility]="FAIL"; REASON[gpu_visibility]="NVIDIA_SMI_MISSING"
elif ! gpu_query="$(nvidia-smi --query-gpu=index,name,uuid,driver_version,compute_cap,memory.total \
        --format=csv,noheader 2>"${OUT_DIR}/gpu_query.err")" || [ -z "${gpu_query}" ]; then
    STATUS[gpu_visibility]="FAIL"; REASON[gpu_visibility]="NVIDIA_SMI_FAILED"
else
    n_gpus="$(grep -c . <<<"${gpu_query}")"
    if [ "${n_gpus}" -ne 1 ]; then
        STATUS[gpu_visibility]="FAIL"; REASON[gpu_visibility]="VISIBLE_GPU_COUNT_${n_gpus}"
    else
        IFS=',' read -r g_index g_name g_uuid g_driver g_cc g_mem <<<"${gpu_query}"
        g_index="${g_index## }"; g_name="${g_name## }"; g_uuid="${g_uuid## }"
        g_driver="${g_driver## }"; g_cc="${g_cc## }"; g_mem="${g_mem## }"
        {
            echo "logical_index=${g_index}"
            echo "name=${g_name}"
            echo "uuid=${g_uuid}"
            echo "driver_version=${g_driver}"
            echo "compute_cap=${g_cc}"
            echo "memory_total=${g_mem}"
        } > "${GPU_KV}"
        if [ "${g_uuid}" != "${EXPECTED_GPU_UUID}" ]; then
            STATUS[gpu_visibility]="FAIL"; REASON[gpu_visibility]="UUID_MISMATCH"
        elif [ "${g_cc}" != "${REQUIRED_CC}" ]; then
            STATUS[gpu_visibility]="FAIL"; REASON[gpu_visibility]="UNEXPECTED_COMPUTE_CAP_${g_cc}"
        else
            STATUS[gpu_visibility]="PASS"; REASON[gpu_visibility]="OK"
        fi
    fi
fi
echo "preflight: gpu_visibility ${STATUS[gpu_visibility]} (${REASON[gpu_visibility]})"

# --- check 2: tool_versions --------------------------------------------------
# Version queries only; nothing here executes GPU work. A tool is rejected if
# it is absent, its version command fails, or its output is empty/malformed
# (must match the expected version pattern).
tools_ok=1
record_tool() {
    local name="$1" pattern="$2"
    shift 2
    local out line
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "${name}=MISSING" >> "${TOOLS_KV}"
        tools_ok=0
        return
    fi
    if ! out="$("$@" 2>&1)"; then
        echo "${name}=COMMAND_FAILED" >> "${TOOLS_KV}"
        tools_ok=0
        return
    fi
    line="$(grep -iE -- "${pattern}" <<<"${out}" | head -n1 | tr -d '\r')"
    if [ -z "${line}" ]; then
        echo "${name}=MALFORMED_VERSION_OUTPUT" >> "${TOOLS_KV}"
        tools_ok=0
        return
    fi
    echo "${name}=${line}" >> "${TOOLS_KV}"
}
record_tool "nvcc"      'release [0-9]+\.[0-9]+'  nvcc --version
record_tool "ptxas"     'release [0-9]+\.[0-9]+'  ptxas --version
record_tool "cuobjdump" 'release [0-9]+\.[0-9]+'  cuobjdump --version
record_tool "nvdisasm"  'release [0-9]+\.[0-9]+'  nvdisasm --version
record_tool "ncu"       'version [0-9]+'          ncu --version
record_tool "python3"   'python [0-9]+\.[0-9]+'   python3 --version
# CuTe DSL is a module, not an executable; validate a real version string.
if dsl_out="$(python3 -c 'import cutlass; print(cutlass.__version__)' 2>&1)" \
        && [[ "${dsl_out}" =~ ^[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
    echo "cutedsl=${dsl_out}" >> "${TOOLS_KV}"
else
    echo "cutedsl=MISSING_OR_MALFORMED" >> "${TOOLS_KV}"
    tools_ok=0
fi
if [ "${tools_ok}" -eq 1 ]; then
    STATUS[tool_versions]="PASS"; REASON[tool_versions]="OK"
else
    STATUS[tool_versions]="FAIL"; REASON[tool_versions]="TOOL_MISSING_OR_MALFORMED"
fi
echo "preflight: tool_versions ${STATUS[tool_versions]} (${REASON[tool_versions]})"

# --- check 3: cuda_smoke_compile ---------------------------------------------
SMOKE_BIN="${OUT_DIR}/cuda_smoke"
if [ "${STATUS[tool_versions]}" != "PASS" ]; then
    STATUS[cuda_smoke_compile]="SKIP"
    REASON[cuda_smoke_compile]="DEPENDENCY_TOOL_VERSIONS_${STATUS[tool_versions]}"
elif nvcc -arch=${CUDA_ARCH} -O2 -o "${SMOKE_BIN}" smoke/cuda_smoke.cu \
        > "${OUT_DIR}/cuda_smoke_compile.log" 2>&1; then
    STATUS[cuda_smoke_compile]="PASS"; REASON[cuda_smoke_compile]="OK"
else
    STATUS[cuda_smoke_compile]="FAIL"; REASON[cuda_smoke_compile]="NVCC_COMPILE_FAILED"
fi
echo "preflight: cuda_smoke_compile ${STATUS[cuda_smoke_compile]} (${REASON[cuda_smoke_compile]})"

# --- check 4: cuda_smoke_run --------------------------------------------------
RUN_LOG="${OUT_DIR}/cuda_smoke_run.log"
if [ "${STATUS[gpu_visibility]}" != "PASS" ]; then
    STATUS[cuda_smoke_run]="SKIP"
    REASON[cuda_smoke_run]="DEPENDENCY_GPU_VISIBILITY_${STATUS[gpu_visibility]}"
elif [ "${STATUS[cuda_smoke_compile]}" != "PASS" ]; then
    STATUS[cuda_smoke_run]="SKIP"
    REASON[cuda_smoke_run]="DEPENDENCY_CUDA_SMOKE_COMPILE_${STATUS[cuda_smoke_compile]}"
else
    "${SMOKE_BIN}" > "${RUN_LOG}" 2>&1
    rc=$?
    if log_has_driver_error "${RUN_LOG}"; then
        STATUS[cuda_smoke_run]="BLOCKED"; REASON[cuda_smoke_run]="BLOCKED_DRIVER"
    elif [ "${rc}" -eq 0 ] \
            && grep -q 'CUDA_SMOKE_RESULT=PASS' "${RUN_LOG}" \
            && grep -q "compute_capability=${REQUIRED_CC}" "${RUN_LOG}"; then
        STATUS[cuda_smoke_run]="PASS"; REASON[cuda_smoke_run]="OK"
    elif [ "${rc}" -eq 0 ]; then
        STATUS[cuda_smoke_run]="FAIL"; REASON[cuda_smoke_run]="UNEXPECTED_OUTPUT"
    else
        STATUS[cuda_smoke_run]="FAIL"; REASON[cuda_smoke_run]="SMOKE_RUN_FAILED_RC_${rc}"
    fi
fi
echo "preflight: cuda_smoke_run ${STATUS[cuda_smoke_run]} (${REASON[cuda_smoke_run]})"

# --- check 5: cutedsl_smoke ----------------------------------------------------
DSL_LOG="${OUT_DIR}/cutedsl_smoke.log"
if [ "${STATUS[gpu_visibility]}" != "PASS" ]; then
    STATUS[cutedsl_smoke]="SKIP"
    REASON[cutedsl_smoke]="DEPENDENCY_GPU_VISIBILITY_${STATUS[gpu_visibility]}"
elif [ "${STATUS[tool_versions]}" != "PASS" ]; then
    STATUS[cutedsl_smoke]="SKIP"
    REASON[cutedsl_smoke]="DEPENDENCY_TOOL_VERSIONS_${STATUS[tool_versions]}"
else
    python3 smoke/cutedsl_smoke.py > "${DSL_LOG}" 2>&1
    rc=$?
    if log_has_driver_error "${DSL_LOG}"; then
        STATUS[cutedsl_smoke]="BLOCKED"; REASON[cutedsl_smoke]="BLOCKED_DRIVER"
    elif [ "${rc}" -eq 0 ] && grep -q 'CUTEDSL_SMOKE_RESULT=PASS' "${DSL_LOG}"; then
        STATUS[cutedsl_smoke]="PASS"; REASON[cutedsl_smoke]="OK"
    else
        STATUS[cutedsl_smoke]="FAIL"; REASON[cutedsl_smoke]="SMOKE_RUN_FAILED_RC_${rc}"
    fi
fi
echo "preflight: cutedsl_smoke ${STATUS[cutedsl_smoke]} (${REASON[cutedsl_smoke]})"

# --- check 6: ncu_profile ------------------------------------------------------
NCU_LOG="${OUT_DIR}/ncu_profile.log"
if [ "${STATUS[gpu_visibility]}" != "PASS" ]; then
    STATUS[ncu_profile]="SKIP"
    REASON[ncu_profile]="DEPENDENCY_GPU_VISIBILITY_${STATUS[gpu_visibility]}"
elif [ "${STATUS[cuda_smoke_run]}" != "PASS" ]; then
    STATUS[ncu_profile]="SKIP"
    REASON[ncu_profile]="DEPENDENCY_CUDA_SMOKE_RUN_${STATUS[cuda_smoke_run]}"
else
    ncu --set basic --force-overwrite -o "${OUT_DIR}/cuda_smoke_profile" \
        "${SMOKE_BIN}" > "${NCU_LOG}" 2>&1
    rc=$?
    if grep -q 'ERR_NVGPUCTRPERM' "${NCU_LOG}"; then
        STATUS[ncu_profile]="BLOCKED"; REASON[ncu_profile]="BLOCKED_NCU"
    elif log_has_driver_error "${NCU_LOG}"; then
        STATUS[ncu_profile]="BLOCKED"; REASON[ncu_profile]="BLOCKED_DRIVER"
    elif [ "${rc}" -eq 0 ] && [ -f "${OUT_DIR}/cuda_smoke_profile.ncu-rep" ]; then
        STATUS[ncu_profile]="PASS"; REASON[ncu_profile]="OK"
    else
        STATUS[ncu_profile]="FAIL"; REASON[ncu_profile]="NCU_FAILED_RC_${rc}"
    fi
fi
echo "preflight: ncu_profile ${STATUS[ncu_profile]} (${REASON[ncu_profile]})"

# finalize() runs via the EXIT trap and sets the process exit code.
exit 0
