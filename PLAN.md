# PLAN.md — schedule, units, and audit status

Schedule: 17 July through 15 August 2026.

Field semantics (per unit):

- **Implemented** — the code/definition exists in this repository. It says
  nothing about having been built, executed, or exercised.
- **Audited** — an independent reviewer (not the author, and not a static
  self-check) has audited the unit. Static self-checks such as
  `make check-static` are **not** an audit.
- **Verified on GB300** — the unit has actually been exercised successfully on
  the target GB300 hardware.

No unit in this repository has been audited or verified yet.

## Phase 0 — Contract, environment, launcher, smoke (17–19 July 2026)

Gate: every P0 unit implemented, independently audited, and verified on GB300
before Phase 1 begins.

| Unit | Description | Implemented | Audited | Verified on GB300 |
|------|-------------|-------------|---------|-------------------|
| P0.1 | Contract and repository (AGENTS.md, README.md, PLAN.md, LICENSE, .gitignore, VERSIONS.env) | YES | NO | NO |
| P0.2 | Reproducible CUDA 13.1 + CuTe DSL environment (Dockerfile, image pinning) | YES (definition only; image not built) | NO | NO |
| P0.3 | Safe one-GPU launcher and preflight (run_container.sh, preflight.sh, Makefile) | YES (static only; never run) | NO | NO |
| P0.4 | CUDA, CuTe DSL, and NCU smoke checks (cuda_smoke.cu, cutedsl_smoke.py, ncu step in preflight) | YES (source only; never compiled or run) | NO | NO |

## Phase 1 — LDGSTS versus TMA (20–26 July 2026)

Gate: Phase 0 gate passed; correctness validated before any timing/profiling.

| Unit | Description | Implemented | Audited | Verified on GB300 |
|------|-------------|-------------|---------|-------------------|
| P1.1 | Minimal LDGSTS path | NO | NO | NO |
| P1.2 | Equivalent TMA path | NO | NO | NO |
| P1.3 | Joint sweep (≤18 configurations) | NO | NO | NO |
| P1.4 | Profiling, validation, analysis, pilot | NO | NO | NO |

## Phase 2 — BF16 UMMA throughput (27 July–2 August 2026)

Gate: Phase 1 gate passed.

| Unit | Description | Implemented | Audited | Verified on GB300 |
|------|-------------|-------------|---------|-------------------|
| P2.1 | 1-SM UMMA | NO | NO | NO |
| P2.2 | 2-SM UMMA | NO | NO | NO |
| P2.3 | Sweep (≤24 configurations) | NO | NO | NO |
| P2.4 | Profiling and empirical ceiling | NO | NO | NO |

## Phase 3 — CuTe DSL GEMM versus cuBLASLt (3–9 August 2026)

Gate: Phase 2 gate passed.

| Unit | Description | Implemented | Audited | Verified on GB300 |
|------|-------------|-------------|---------|-------------------|
| P3.1 | Pinned official CuTe DSL example | NO | NO | NO |
| P3.2 | One-shape wrapper | NO | NO | NO |
| P3.3 | cuBLASLt baseline | NO | NO | NO |
| P3.4 | Three execution variants | NO | NO | NO |
| P3.5 | Five shapes and comparison | NO | NO | NO |

## Phase 4 — Campaigns and integration (10–15 August 2026)

Gate: Phase 3 gate passed.

| Unit | Description | Implemented | Audited | Verified on GB300 |
|------|-------------|-------------|---------|-------------------|
| P4.1 | Orchestrator | NO | NO | NO |
| P4.2 | Pilot plus three final campaigns | NO | NO | NO |
| P4.3 | Integrated analysis, documentation, audit | NO | NO | NO |
