# AGENTS.md — rules for humans and AI agents working in this repository

This file is binding for every agent (human or AI) that edits or runs anything
in `gb300-gemm-anatomy`. When in doubt, stop and ask; fail closed.

## Scientific scope (frozen)

Research question: how do HBM-to-SMEM data movement and fifth-generation
Tensor Core throughput constrain BF16 GEMM performance on NVIDIA GB300, and
how closely can a CuTe DSL implementation approach cuBLASLt?

Exactly three experiments:

1. **LDGSTS versus TMA** — equivalent HBM-to-SMEM paths; sustained traffic and
   in-flight bytes needed for saturation (≤18 configurations).
2. **BF16 UMMA throughput** — fifth-generation Tensor Core ceiling and 2-SM
   scaling (BF16×BF16, FP32 accumulation, ≤24 configurations).
3. **CuTe DSL BF16 GEMM versus cuBLASLt** — configure a GEMM from experiments
   1–2 and explain the remaining gap on five fixed `(M,N,K)` shapes.

Out of scope: Hopper; FP8/FP4/NVFP4/MXFP4; multi-GPU, NVLink, or
Grace-Blackwell coherence; attention/convolution/elementwise studies; a
general instruction catalogue; a CUDA-core roofline; exhaustive sweeps;
beating cuBLASLt as a success criterion.

Pinned target: CUDA 13.1.0, CUTLASS/CuTe DSL v4.6.1, `sm_103a` (see
`VERSIONS.env`). Never change pinned versions automatically.

## Shared cluster rules (mandatory)

The target is a shared node with eight GB300-class GPUs. These rules are not
negotiable:

- **Never select a GPU automatically.** Every GPU run requires an explicit
  `BLACKWELL_GPU_INDEX=<physical-index>` from the operator.
- **Refuse a GPU run unless the selected physical device can be conservatively
  shown to have no active compute processes.** Ambiguous or incomplete
  process visibility means stop, not proceed.
- **Resolve the requested physical index to its UUID and expose exactly that
  UUID** to the container. Inside the container, exactly one matching GPU must
  be visible as CUDA logical device 0; otherwise abort before running work.
- **Never** use `--gpus all`, `NVIDIA_VISIBLE_DEVICES=all`, privileged mode,
  host PID namespace, added capabilities, `SYS_ADMIN`, a Docker socket mount,
  `sudo`, MPS, or a multi-GPU workload.
- **Never change clocks, persistence/compute modes, or power limits.**
- **Never use `$(nproc)`.** Compilation and builds are capped at **two**
  parallel jobs (`MAX_BUILD_JOBS=2`).
- **Never use `set -x`, dump the environment, or store secrets, credentials,
  SSH material, usernames, home paths, or unrelated host metadata** in logs,
  results, or committed files. Log only allowlisted device data (GPU index,
  name, UUID, driver version, compute capability, memory size, tool versions).
- **Correctness must pass before timing or profiling.** No performance number
  is reported from a run whose correctness check did not pass.
- **Preserve existing user work and fail closed on uncertainty.** Never
  delete or overwrite files you did not create in the current task.

## Working conventions

- Work incrementally; every unit must be independently audited before it is
  trusted (static self-checks are not an audit) and separately verified on
  GB300 hardware. `PLAN.md` tracks `Implemented / Audited / Verified on GB300`
  per unit.
- Do not commit, stage, branch, push, or open pull requests unless the
  operator explicitly requests it.
- Documentation must never claim results, performance numbers, or
  compatibility that has not actually been measured in this repository.
- Raw run output goes under `results/preflight/` or future raw areas (ignored
  by Git); only small, curated, secret-free processed results are committed.
