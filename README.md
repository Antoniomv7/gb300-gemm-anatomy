# gb300-gemm-anatomy

Anatomy of BF16 GEMM performance on NVIDIA GB300: a small, reproducible,
auditable measurement study.

**Status: `Phase 0 — audited and verified on GB300`.**

The Phase 0 environment, single-GPU launcher, CUDA smoke test, CuTe DSL smoke
test, and Nsight Compute access were successfully verified on the target
hardware on 20 July 2026. No experimental performance results exist yet.

## Research question

How do HBM-to-SMEM data movement and fifth-generation Tensor Core throughput
constrain BF16 GEMM performance on NVIDIA GB300, and how closely can a CuTe
DSL implementation approach cuBLASLt?

## Experiments (complete list)

1. **LDGSTS versus TMA** — compare equivalent HBM-to-SMEM paths (vectorized
   LDGSTS/`cp.async` versus 2D unicast TMA) and determine sustained traffic
   and the in-flight bytes needed for saturation. 2/4/8 stages, three byte
   volumes, at most 18 configurations, one CTA per SM, working set above 2× L2,
   with selected SASS and Nsight Compute checks.
2. **BF16 UMMA throughput** — estimate the fifth-generation Tensor Core
   ceiling and 2-SM scaling. BF16×BF16 with FP32 accumulation; 1-SM M=128,
   2-SM M=256; N ∈ {64, 128, 256}; depth ∈ {4, 16, 64, 256}; at most 24
   configurations; tcgen05/UTCMMA usage to be verified in SASS.
3. **CuTe DSL BF16 GEMM versus cuBLASLt** — use experiments 1–2 to configure a
   GEMM and explain the remaining gap. Variants: non-persistent 1-CTA,
   persistent 1-CTA, persistent 2-CTA; at most six candidates per shape; an
   equivalent cuBLASLt baseline.

Final `(M,N,K)` shapes for experiment 3:

- `(4096,4096,4096)`
- `(8192,8192,8192)`
- `(16384,512,4096)`
- `(32768,512,4096)`
- `(512,16384,4096)`

## Out of scope

Hopper; FP8, FP4, NVFP4, and MXFP4; multi-GPU execution, NVLink, or
Grace–Blackwell coherence; attention, convolution, or elementwise studies; a
general instruction catalogue; a CUDA-core roofline; exhaustive sweeps; and
beating cuBLASLt as a success criterion.

## Verified target environment

Phase 0 was verified with the following environment:

- Shared node containing eight NVIDIA B300 SXM6 AC GPUs.
- Exactly one explicitly selected physical GPU exposed to each container run.
- Selected physical GPU mapped to logical device 0 inside the container.
- Compute capability 10.3 with compilation target `sm_103a`.
- NVIDIA driver 580.95.05.
- CUDA Toolkit 13.1.0:
  - `nvcc` 13.1.80
  - `ptxas` 13.1.80
  - `cuobjdump` 13.1.80
  - `nvdisasm` 13.1.80
- Nsight Compute 2025.4.0.0.
- Python 3.12.3.
- CUTLASS/CuTe DSL 4.6.1.
- Docker with the NVIDIA Container Toolkit.

The successful smoke tests establish compatibility for the Phase 0 checks.
Each later experimental phase must still validate correctness and the required
Blackwell instructions before collecting performance measurements.

## Phase 0 verification record

The executable Phase 0 implementation was verified at Git commit:

```text
7bb553fe7df95daf7a8ee07a4cd4cf5cc0824fb7
```

The preflight ran with a clean Git worktree and produced:

```text
Timestamp:               20260720T161935Z
GPU visibility:          PASS
Tool versions:           PASS
CUDA smoke compilation:  PASS
CUDA smoke execution:    PASS
CuTe DSL smoke:          PASS
Nsight Compute profile:  PASS
Overall status:          PASS
Exit code:               0
```

The run used physical GPU index 4, whose UUID was verified against logical
device 0 inside the container. No active compute processes were present when
the launcher performed its pre-execution check.

Raw diagnostic output is stored locally under:

```text
results/preflight/20260720T161935Z/
```

This directory contains logs, the smoke binary, the Nsight Compute report, and
`summary.json`. Raw preflight output is intentionally ignored by Git.

## Repository contents after Phase 0

```text
AGENTS.md                 Binding rules for agents and shared-cluster safety
README.md                 Project scope and current verified status
PLAN.md                   Phase plan with per-unit audit/verification status
LICENSE                   BSD 3-Clause
.gitignore                Ignore rules for raw outputs, caches, and secrets
VERSIONS.env              Immutable version contract
Dockerfile                Reproducible CUDA 13.1 and CuTe DSL environment
Makefile                  Phase 0 build and validation entry points
scripts/run_container.sh  Fail-closed single-GPU container launcher
scripts/preflight.sh      In-container preflight and JSON summary generation
smoke/cuda_smoke.cu       Deterministic CUDA smoke test
smoke/cutedsl_smoke.py    Minimal real CuTe DSL kernel smoke test
results/README.md         Result storage and publication policy
```

## Phase 0 validation workflow

The completed Phase 0 workflow is:

```bash
make check-static
make build-image
make check-env

# Select a physical GPU only after confirming that it is available.
BLACKWELL_GPU_INDEX=<physical-index> make preflight
```

`BLACKWELL_GPU_INDEX` is mandatory. The project never selects a GPU
automatically and never exposes all GPUs to a container.

Phase 0 provides environment and tooling validation only. Experiments 1–3 have
not started, and the repository contains no bandwidth, throughput, GEMM
performance, or cuBLASLt comparison results yet. See `PLAN.md` for the
remaining schedule and `AGENTS.md` for the mandatory shared-cluster rules.
