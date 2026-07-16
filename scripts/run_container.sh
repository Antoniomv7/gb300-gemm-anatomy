#!/usr/bin/env bash
# Safe single-GPU container launcher for gb300-gemm-anatomy.
#
# Fail-closed contract:
#   - Requires an explicit numeric BLACKWELL_GPU_INDEX; never auto-selects.
#   - Resolves the physical index to its GPU UUID and exposes exactly that
#     UUID to the container; exposing every GPU is forbidden.
#   - Refuses to launch unless an immediate compute-process query proves the
#     selected device has no active compute processes. Ambiguous or
#     incomplete visibility aborts.
#   - Inside the container, verifies exactly one GPU is visible and that its
#     UUID matches before executing the inner command (CUDA logical device 0).
#   - No privileged mode, host PID, added capabilities, or Docker socket.
#   - Preserves the inner command's exit code.
#   - Logs only allowlisted device data (index, UUID, name, driver version).

set -Eeuo pipefail

usage() {
    echo "Usage: BLACKWELL_GPU_INDEX=<physical-index> $0 <command> [args...]" >&2
    echo "Runs <command> inside the pinned container with exactly one GPU visible." >&2
}

fail() {
    echo "run_container: ERROR: $*" >&2
    exit 2
}

[ "$#" -ge 1 ] || { usage; fail "no command given"; }

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "${REPO_ROOT}/VERSIONS.env" ] || fail "repository root not found at ${REPO_ROOT}"
IMAGE_TAG="${IMAGE_TAG:-gb300-gemm-anatomy:phase0}"

command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found on host"
command -v docker >/dev/null 2>&1 || fail "docker not found on host"

# --- Explicit GPU selection (never automatic) -------------------------------
[ -n "${BLACKWELL_GPU_INDEX:-}" ] \
    || fail "BLACKWELL_GPU_INDEX is not set; this launcher never selects a GPU automatically"
[[ "${BLACKWELL_GPU_INDEX}" =~ ^[0-9]+$ ]] \
    || fail "BLACKWELL_GPU_INDEX must be a non-negative integer, got '${BLACKWELL_GPU_INDEX}'"

# --- Resolve physical index -> UUID (allowlisted fields only) ---------------
gpu_table="$(nvidia-smi --query-gpu=index,uuid,name --format=csv,noheader)" \
    || fail "nvidia-smi GPU query failed; GPU visibility is ambiguous"
[ -n "${gpu_table}" ] || fail "nvidia-smi reported no GPUs"

row="$(awk -F', *' -v idx="${BLACKWELL_GPU_INDEX}" '$1 == idx' <<<"${gpu_table}")"
[ -n "${row}" ] || fail "no GPU with physical index ${BLACKWELL_GPU_INDEX} exists on this host"
[ "$(wc -l <<<"${row}")" -eq 1 ] \
    || fail "physical index ${BLACKWELL_GPU_INDEX} matched more than one GPU; refusing"

GPU_UUID="$(awk -F', *' '{print $2}' <<<"${row}")"
GPU_NAME="$(awk -F', *' '{print $3}' <<<"${row}")"
[[ "${GPU_UUID}" =~ ^GPU-[0-9a-fA-F][0-9a-fA-F-]+$ ]] \
    || fail "resolved UUID has unexpected format '${GPU_UUID}'; refusing"

DRIVER_VERSION="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "${GPU_UUID}")" \
    || fail "driver version query failed for ${GPU_UUID}"

echo "run_container: selected index=${BLACKWELL_GPU_INDEX} uuid=${GPU_UUID} name='${GPU_NAME}' driver=${DRIVER_VERSION}"

# --- Prove the device is idle immediately before launch (fail closed) -------
# Any query failure, permission gap, or unexpected content counts as
# "cannot prove idle" and aborts. The window between this check and container
# start is unavoidable but minimized by checking last.
if ! apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "${GPU_UUID}" 2>&1)"; then
    fail "compute-process query failed for ${GPU_UUID}; cannot prove the device is free"
fi
if grep -qiE 'insufficient|not supported|unknown|error|n/a' <<<"${apps}"; then
    fail "compute-process visibility is incomplete for ${GPU_UUID}; refusing to share the device"
fi
if [ -n "${apps//[[:space:]]/}" ]; then
    n_procs="$(grep -c . <<<"${apps}")"
    fail "GPU ${GPU_UUID} has ${n_procs} active compute process(es); refusing to run"
fi
echo "run_container: GPU ${GPU_UUID} has no active compute processes"

# --- Launch: only this UUID visible, unprivileged, repo-only mount ----------
# The in-container guard re-verifies that exactly one GPU is visible and that
# it is the requested UUID (as CUDA logical device 0) before exec'ing the
# inner command; 'exec' everywhere preserves the inner exit code.
exec docker run \
    --rm \
    --gpus "device=${GPU_UUID}" \
    --user "$(id -u):$(id -g)" \
    --network none \
    --security-opt no-new-privileges \
    --cap-drop ALL \
    -e HOME=/tmp \
    -e EXPECTED_GPU_UUID="${GPU_UUID}" \
    -v "${REPO_ROOT}:/workspace" \
    -w /workspace \
    "${IMAGE_TAG}" \
    bash -c '
        set -euo pipefail
        mapfile -t uuids < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
        if [ "${#uuids[@]}" -ne 1 ]; then
            echo "container guard: expected exactly 1 visible GPU, saw ${#uuids[@]}; aborting" >&2
            exit 66
        fi
        if [ "${uuids[0]}" != "${EXPECTED_GPU_UUID}" ]; then
            echo "container guard: visible GPU UUID does not match the requested UUID; aborting" >&2
            exit 66
        fi
        exec "$@"
    ' guard "$@"
