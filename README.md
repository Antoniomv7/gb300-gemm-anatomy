# gb300-gemm-anatomy

Anatomy of BF16 GEMM performance on NVIDIA GB300: a small, reproducible,
auditable measurement study.

**Status: `Phase 0 implementation — pending audit`.** Nothing in this
repository has been built, executed, or verified on hardware yet. There are
no results and no performance or compatibility claims.

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
`(4096,4096,4096)`, `(8192,8192,8192)`, `(16384,512,4096)`,
`(32768,512,4096)`, `(512,16384,4096)`.

## Out of scope

Hopper; FP8/FP4/NVFP4/MXFP4; multi-GPU, NVLink, or Grace-Blackwell coherence;
attention, convolution, or elementwise studies; a general instruction
catalogue; a CUDA-core roofline; exhaustive sweeps; and beating cuBLASLt as a
success criterion.

## Target environment — pending verification

The following is **expected but not yet verified** in this repository; the
preflight (not yet run) is what will verify it:

- Shared node with eight NVIDIA GB300/B300 GPUs, exactly one physical GPU per run.
- Expected compute capability 10.3 (`sm_103a`).
- Previously observed driver 580.95.05 — note that CUDA 13.1 ships with driver
  590.44.01 and 580.x relies on CUDA 13.x minor-version compatibility; this is
  an open risk the preflight must resolve, not a verified fact.
- Docker with the NVIDIA Container Toolkit.
- Pinned software: CUDA 13.1.0, CUTLASS/CuTe DSL v4.6.1 (see `VERSIONS.env`).

## Repository contents (Phase 0)

```text
AGENTS.md                 Binding rules for agents and shared-cluster safety
README.md                 This file
PLAN.md                   Phase plan with per-unit audit/verification status
LICENSE                   BSD 3-Clause
.gitignore                Ignore rules (raw outputs, caches, secrets)
VERSIONS.env              Immutable version contract (image digest, commit)
Dockerfile                Reproducible CUDA 13.1 + CuTe DSL environment
Makefile                  help / check-static / build-image / check-env / preflight
scripts/run_container.sh  Safe fail-closed one-GPU container launcher
scripts/preflight.sh      In-container preflight (smokes + ncu, JSON summary)
smoke/cuda_smoke.cu       Deterministic CUDA smoke test (no benchmarking)
smoke/cutedsl_smoke.py    Real minimal CuTe DSL kernel smoke test (non-GEMM)
results/README.md         Policy for result storage and publication
```

## Intended usage — not yet run

None of the following has been executed yet; they require an independent
audit first, then the shared GB300 node:

```bash
make check-static                              # static validation only (safe anywhere)
make build-image                               # build pinned image, no GPU
make check-env                                 # tool checks in a GPU-less container
BLACKWELL_GPU_INDEX=<physical-index> make preflight   # explicit single-GPU preflight
```

There are no results yet. See `PLAN.md` for schedule and per-unit status and
`AGENTS.md` for the mandatory shared-cluster rules.
