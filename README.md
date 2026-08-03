# gb300-gemm-anatomy

Anatomy of BF16 GEMM performance on NVIDIA GB300: a small, reproducible,
auditable measurement study.

**Status: `Phase 0 — audited and verified on GB300`. `P1.1 (standalone LDGSTS
baseline) — implemented, audited, functionally verified on GB300`. `P1.2
(standalone 2D unicast TMA path) — implemented, audited, functionally
verified on GB300`. `P1.3 (joint LDGSTS/TMA sweep infrastructure) —
implemented, remediation completed, independently audited, and functionally
verified on GB300`. `P1.4 (profiling, HBM validation, analysis, pilot) —
implemented, remediated after FIVE independent GPU-free audits; final
post-remediation review PASS; verified on GB300: YES; fresh preflight: PASS;
pilot executed: YES; NCU/HBM validation: six of six predefined cases
`HBM_VALIDATED`; publishable results: NONE; Phase 1: CLOSED`. `P2.1 (1-SM
BF16 UMMA) — implemented, independently audited, and functionally verified
on GB300; publishable results: NONE`. `P2.2 (2-SM BF16 UMMA) — implemented;
independently audited: YES; verified on GB300: YES; publishable results:
NONE`. `P2.3 (joint 1-SM/2-SM UMMA sweep infrastructure) — implemented;
independently audited: YES; verified on GB300: YES; publishable results:
NONE`.**

The Phase 0 environment, single-GPU launcher, CUDA smoke test, CuTe DSL smoke
test, and Nsight Compute access were successfully verified on the target
hardware on 20 July 2026. The P1.1 and P1.2 GB300 runs were functional
verification of the binaries and container plumbing (all nine
specializations' `--self-test` correctness, plus one short `run_kind=smoke`
measurement each). The P1.3 joint smoke campaign `20260728T103315Z` completed
on GB300 at Git commit `59777406b9454f00799c48bff8fa85cb03625cb6`, with
both full-binary self-tests and all 18 planned invocations passing. These
smoke runs establish functional verification only: their bandwidth values are
not experimental results. No publishable experimental performance results
exist yet.

P1.4 campaign `20260730T073045Z` was executed on GB300 at Git commit
`e2d01b86f53177bd48d18b215be48b422dc3c53b`, after fresh preflight
`20260730T072946Z` passed on the same GPU and driver. The frozen pilot reached
`ANALYZED`: all 18 configurations completed all 30 retained repetitions
(540 samples), and all six predefined Nsight Compute cases were classified
`HBM_VALIDATED` with no diagnostic flags. The final integrity validator
reloaded the append-only manifest chain (revision 10), re-hashed the
evidence, and reported `CIERRE TÉCNICO P1.4 / FASE 1: PASS`. This is one
reviewed pilot, not a final campaign; every artifact remains
`publishable: false`.

P1.1, the standalone LDGSTS arm of the "LDGSTS versus TMA" experiment
(`src/memory/ldgsts.cu`), is implemented as a global-memory-to-SMEM effective
copy benchmark. Its GPU-free SASS gate requires complete 16-byte LDGSTS groups
and matching commit/wait dependency instructions for all nine frozen
specializations, while allowing `ptxas` to duplicate whole groups when it
unrolls or peels the loop (see `src/memory/README.md`). P1.1 has been
independently audited and functionally verified on GB300, so `PLAN.md`
records Audited=YES and Verified on GB300=YES for P1.1.

P1.2, the standalone 2D unicast TMA arm (`src/memory/tma.cu`), is implemented
as the TMA counterpart: it moves the exact same logical tiles as P1.1 through
a host-encoded rank-2 `CUtensorMap` descriptor and an mbarrier-tracked
pipeline (`cp.async.bulk.tensor.2d.shared::cta.global`), with the same
128-threads/CTA, grid-equals-SM-count, and one-CTA-per-SM occupancy contract.
Its GPU-free SASS gate requires a genuine `UTMALDG.2D` load, transaction-aware
mbarrier arrival, phase/parity waits, and full mbarrier invalidation after the
pipeline drains for all nine frozen specializations, with no LDGSTS, 1D, or
multicast/cluster fallback (see `src/memory/README.md`). P1.2 has likewise
been independently audited and functionally verified on GB300, so `PLAN.md`
records Audited=YES and Verified on GB300=YES for P1.2 as well.

P1.3, the joint LDGSTS/TMA sweep infrastructure
(`scripts/run_exp01_memory_paths.sh`, `scripts/aggregate_exp01_memory_paths.py`),
is implemented: a deterministic 18-invocation runner (2 methods x 3 stage
counts x 3 bytes-in-flight values, alternating which method runs first per
configuration pair), strict validation of every field of every repetition of
both binaries' raw 37-column CSV, symlink-safe centralized campaign
initialization, no-clobber publication of every result/log/evidence file,
lossless consolidation into `combined_samples.csv`, and purely descriptive
per-configuration statistics in `summary.csv` (mean/median/sample
stdev/coefficient of variation — no speedups, no outlier filtering, no
significance testing). A first independent audit found fifteen confirmed
defects; a second audit found remaining blockers in synthetic-test
isolation, symlink-safe finalization, loaded-manifest validation,
no-clobber/rollback behavior, canonical CSV parsing, and progress telemetry.
The second remediation is completed in the implementation and adversarial
GPU-free tests (see `src/memory/README.md`), and the remediated implementation
subsequently passed a new independent GPU-free audit. Campaign
`20260728T103315Z` then reached `COMPLETE` on GB300 after both binaries'
full self-tests and all 18 smoke configurations passed. `PLAN.md` therefore
records Implemented=YES, Audited=YES, and Verified on GB300=YES for P1.3.
This closes the P1.3 infrastructure gate but does not create a publishable
performance result.

P1.4 (profiling, Nsight Compute HBM validation, the pilot benchmark
campaign, statistics, and comparative LDGSTS/TMA interpretation) is
implemented as a GPU-free layer around the audited P1.3 infrastructure:
`scripts/run_exp01_memory_paths_p14.sh` reuses the unmodified P1.3 runner,
under frozen parameters, for the one 18-configuration `run_kind=benchmark`
pilot, and profiles exactly six predefined `(method, stages,
bytes_in_flight_kib)` cases with Nsight Compute 2025.4.0.0 under
clock-control-disabled, non-defaulting profiler controls that were verified
against the pinned image's real `ncu --help` output during implementation.
`scripts/analyze_exp01_memory_paths_p14.py` validates the frozen
preflight/provenance contract, classifies each profiled case's DRAM-read
ratio as `HBM_VALIDATED` or `INCONCLUSIVE`, computes deterministic
bootstrap statistics (fixed seed, 10,000 resamples, standard library only)
over all 30 retained repetitions of all 18 pilot configurations, a paired
LDGSTS/TMA ratio comparison, and a candidate-saturation search limited to
the three tested bytes-in-flight values, then generates CSV/JSON/Markdown/
SVG artifacts, all `publishable: false`. See `src/memory/P1_4_PROTOCOL.md`
for the complete frozen protocol. A first independent GPU-free audit found
five blockers (preflight/provenance comparison, post-validation tamper
detection, NCU raw-CSV parsing, raw-tree/log write ordering and safety, and
the manifest's overwrite-based publication); a second independent GPU-free
audit of the remediated implementation then found four further blockers
(a residual raw-tree-write TOCTOU window, an uncompared/unplanned extra
profile directory, cryptographically-chained-but-semantically-unvalidated
manifest revisions, and an incomplete evidence-integrity comparison that
missed some derived per-case fields); a **third** independent GPU-free audit
of that twice-remediated implementation then found five further blockers
(NCU itself still received a raw-campaign pathname for both profile
collection and metrics export, via a path built relative to `/workspace`
instead of the campaign directory; the evidence-integrity gate's field
comparison used `dict.get()`-based equality, under which an unexpected
null-valued field compared as identical to its absence; manifest fields
were classified broadly but never bound to the one specific transition
legally allowed to introduce them; and an unvalidated capture filename, a
launch-failure cleanup gap, and a still-path-based profile inventory in
`scripts/p14_safe_capture.py`); a **fourth** independent GPU-free audit of
that thrice-remediated implementation then found six further blockers
(a manifest field's premature presence was tested by truthiness/non-null
value instead of key presence, so a `null` placeholder evaded detection;
lifecycle timestamps were never compared against each other for
chronological order; `profile_count_completed` and `case_results` could
each appear without the other; `finalize-profile`/`analyze` built their
terminal manifest's `artifact_sha256` from values already in memory before
the evidence-integrity gate ran, and `analyze` never re-ran that gate a
second time immediately before publishing `ANALYZED`; the NCU bundle
publisher's cleanup could unlink a file it no longer owned; and `PLAN.md`
stated a phase gate had passed when it had not). A **fifth** independent
GPU-free audit of commit `3d92a6b375ce3d0e803afd3e62723b08e471f3c8`
found three final functional blockers: failure telemetry used
`failure_detail: null` and could not record an interruption while still
`PILOT_COMPLETE`; manifest revisions lacked an exact per-transition mutation
matrix; and `COMPLETE` did not require the frozen `profile_order` plus the
canonical complete evidence-hash map. All twenty-three blockers, across all
five rounds, were remediated GPU-free, each with a new
adversarial test that first demonstrably failed against the original
behavior and then passed against the fix — the third round's fix for the
NCU path-escape blockers is a new container-side bridge
(`scripts/p14_ncu_bridge.py`) that runs NCU entirely inside the container's
own private, non-host-mounted `/tmp` and hands the host only a versioned
bundle over its own stdout, so NCU never receives a campaign-relative
pathname at all; the fourth round documents a trusted, single-writer
filesystem model explicitly, rather than expanding P1.4 into a
hostile-concurrency-resistant design (see `src/memory/P1_4_PROTOCOL.md`
for the closed design of every blocker and the trust model); the fifth adds
typed failure telemetry, the exact transition-mutation matrix, and canonical
terminal-content validation. The final post-remediation review additionally
covered the four live-NCU compatibility corrections required by NCU 2025.4
on GB300 (help capability, qualified metric identifiers, wide raw-page CSV,
and the `ns` duration unit). The live campaign above then verified the
complete pilot/profile/analyze path. Review of its generated Markdown found
one presentation-only defect: the already-computed string label `ok` was
tested for truthiness and therefore rendered as `REVIEW`. The closing fix
renders the exact `ok`/`REVIEW` value and adds a regression test; it changes
no sample, statistic, profiler evidence, HBM classification, kernel, or GPU
path. No P1.4 result is publishable yet.

## Phase 2 status: P2.1 (1-SM BF16 UMMA), P2.2 (2-SM BF16 UMMA), P2.3 (joint sweep)

P2.1, the 1-SM arm of the "BF16 UMMA throughput" experiment
(`src/compute/umma_1sm.cu`), is implemented: twelve `tcgen05.mma.
cta_group::1.kind::f16` (BF16 x BF16 -> FP32) specializations, one CTA of
128 threads each, N in `{64,128,256}` and depth in `{4,16,64,256}`, with a
GPU-free SASS gate (`scripts/check_umma_1sm_sass.py`) that disassembles the
real compiled binary and requires exactly those twelve symbols, a
compile-time-unrolled `UTCHMMA` burst of exactly `depth` instructions per
symbol, a complete TMEM allocate/commit/wait/load/deallocate lifecycle, and
the absence of any WGMMA, `mma.sync`, TMA, LDGSTS, FP8/FP4, sparse, or 2-SM
instruction. See `src/compute/P2_PROTOCOL.md` for the full frozen contract,
the PTX ISA citations behind every descriptor bit, and the audit of the
pinned secondary reference. The build and SASS checks above may only be
declared successful when they have actually been executed (`make
compute-umma-1sm-build`, `make compute-umma-1sm-sass`,
`make compute-umma-1sm-check`); nothing here is simulated. The remediated
implementation at Git commit
`1004666db7a2eef1ec499c60740cafc1e2f41328` passed an independent audit and
was functionally verified on a physical NVIDIA B300 on 30 July 2026. The
real `sm_103a` binary passed the twelve-specialization SASS contract; the
full device self-test reported `SELF_TEST: PASS (12/12)` with zero
mismatches; and both a short `run_kind=smoke` and a short
`run_kind=benchmark` routing check completed with `correctness=OK`,
`git_dirty=false`, and three rows each. These brief runs validate the
implementation and both CLI routes; their cycle values are not experimental
performance results, and every row is unconditionally `publishable=false`.
P2.2, the 2-SM arm of the "BF16 UMMA throughput" experiment
(`src/compute/umma_2sm.cu`), is implemented: twelve `tcgen05.mma.
cta_group::2.kind::f16` (BF16 x BF16 -> FP32) specializations, one static
two-CTA cluster (128 threads per CTA, 128 local output rows per CTA, joint
M=256), N in `{64,128,256}` and depth in `{4,16,64,256}`, with a GPU-free
SASS/ELF/source gate (`scripts/check_umma_2sm_sass.py`) that disassembles
the real compiled binary and requires exactly those twelve symbols, a
compile-time-unrolled `UTCHMMA.2CTA` burst of exactly `depth` instructions
per symbol, a real `UTCBAR.2CTA.MULTICAST` completion sequence with the
exact `0x0003` cluster mask, a complete collective (both-CTA, warp-0-only)
TMEM allocate/commit/wait/load/deallocate/relinquish lifecycle with cluster
synchronization before deallocation, ELF-level `EIATTR_EXPLICIT_CLUSTER`/
`EIATTR_CTA_PER_CLUSTER` evidence of the compile-time two-CTA cluster
declaration, and the absence of any non-`.2CTA` (1-SM-fallback), WGMMA,
`mma.sync`, TMA, LDGSTS, FP8/FP4, or sparse instruction. See
`src/compute/P2_2_PROTOCOL.md` for the full frozen contract, the PTX ISA
citations behind every descriptor and synchronization step, and the audit
of the pinned secondary reference. The build and SASS checks above may only
be declared successful when they have actually been executed (`make
compute-umma-2sm-build`, `make compute-umma-2sm-sass`, `make
compute-umma-2sm-check`); nothing here is simulated. The real `sm_103a`
binary passed the twelve-specialization SASS/ELF contract during
implementation. The repaired implementation at Git commit
`637b6a7e2efbe77b1c9c5d3dfc7ece527f522bba` passed an independent audit,
the 101-case checker, the pinned CUDA 13.1 `sm_103a` build, and the complete
real-cubin SASS/ELF contract for all twelve specializations. It was
functionally verified on a physical NVIDIA B300 on 31 July 2026: fresh
preflight campaign `20260731T115848Z` passed, the device self-test reported
`SELF_TEST: PASS (12/12)`, and the short smoke route emitted three rows with
`correctness=OK`, `mismatches=0`, the expected commit, and
`git_dirty=false`. These are functional checks only; every CSV row remains
unconditionally `publishable=false`, and no throughput or 1-SM/2-SM scaling
result is claimed.

P2.3, the joint 1-SM/2-SM sweep infrastructure
(`scripts/run_exp02_umma_throughput.sh`,
`scripts/aggregate_exp02_umma_throughput.py`), is implemented: a
deterministic 24-invocation runner (12 logical `(N, depth)` pairs x
`umma_1sm`/`umma_2sm`, alternating which method runs first per pair),
reusing the audited P2.1/P2.2 binaries and command-line interfaces
completely unmodified, strict validation of every field of every repetition
of both binaries' raw 37-column CSV, symlink-safe centralized campaign
initialization, no-clobber publication of every result/log/evidence file,
lossless consolidation into `combined_samples.csv`, and purely descriptive
per-configuration statistics in `summary.csv` (mean/median/sample
stdev/coefficient of variation for `elapsed_cycles`, `cycles_per_umma`, and
`flops_per_cycle` -- no TFLOP/s, no empirical ceiling, no 1-SM/2-SM speedup,
no scaling efficiency, no saturation, no Nsight Compute). See
`src/compute/P2_3_PROTOCOL.md` for the full frozen contract. The final
implementation at Git commit `7a7cc2ab83197376720f030ba2e990092c3ada40`
passed the independent audit and was functionally verified on a physical
NVIDIA B300 on 3 August 2026. Fresh preflight campaign `20260803T141347Z`
reported `OVERALL=PASS`; both complete device self-tests passed; and smoke
campaign `20260803T141410Z` validated all 24 frozen invocations before
reaching `status=COMPLETE`. This closes P2.3 as infrastructure only: every
row remains `publishable=false`, and the smoke cycle values are not
experimental performance results.
P2.4 (profiling and empirical ceiling) remains entirely **not implemented**.

## Research question

How do HBM-to-SMEM data movement and fifth-generation Tensor Core throughput
constrain BF16 GEMM performance on NVIDIA GB300, and how closely can a CuTe
DSL implementation approach cuBLASLt?

## Experiments (complete list)

1. **LDGSTS versus TMA** — compare equivalent HBM-to-SMEM paths (vectorized
   LDGSTS/`cp.async` versus 2D unicast TMA) and determine sustained traffic
   and the in-flight bytes needed for saturation. 2/4/8 stages, three byte
   volumes, at most 18 configurations, maximum active residency of one CTA per
   SM, grid equal to the SM count, working set above 2× L2, with selected SASS
   and Nsight Compute checks.
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

Phase 0 provides environment and tooling validation only. Experiment 1 has
completed P1.1–P1.4: every unit is implemented, independently audited, and
verified on GB300. P1.1/P1.2 and P1.3 campaign `20260728T103315Z` are
functional checks only. P1.4 campaign `20260730T073045Z` is a reviewed,
HBM-validated pilot with 18 configurations, 540 retained samples, and six
successful NCU cases; it closes the Phase 1 technical gate but remains
`publishable: false` pending later final campaigns. Experiment 2 is
underway: P2.1 is implemented, independently audited, and functionally
verified on GB300; P2.2 is likewise implemented, independently audited, and
functionally verified on GB300; P2.3 (joint sweep infrastructure) is
implemented, independently audited, and functionally verified on GB300;
P2.4 and
experiment 3 remain unimplemented. The repository still contains no
publishable bandwidth, throughput, GEMM-performance, or cuBLASLt-comparison
result. The pinned
CUDA 13.1 container contract remains unchanged. See `PLAN.md` for the
remaining schedule and `AGENTS.md` for the mandatory shared-cluster rules.
