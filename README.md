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
NONE`. `P2.4 (profiling and empirical BF16 UMMA per-SM ceiling) —
implemented; independently audited: YES; verified on GB300: YES; campaign
executed: YES; 24/24 profiles validated; analysis: ANALYZED; empirical
per-SM ceiling candidate: 16.37244853848296 TFLOP/s/SM; publishable results:
NONE; Phase 2: CLOSED`. `P3.1 (pinned official CuTe DSL example) —
implemented; independently audited: YES; verified on GB300: YES; executes an
exact pinned official NVIDIA example unchanged; publishable results: NONE;
P3.1: CLOSED`. `P3.2 (one-shape CuTe DSL wrapper) — implemented;
independently audited: YES; verified on GB300: YES; complete-result
correctness: PASS; publishable results: NONE; P3.2: CLOSED. P3.3 (cuBLASLt
baseline) — implemented; independently audited: YES; verified on GB300: YES;
complete-result correctness: PASS; publishable results: NONE; P3.3: CLOSED;
P3.3 itself produces no CuTe-versus-cuBLASLt comparison. P3.4 (three CuTe DSL execution
variants) — implemented; independently audited: YES; verified on GB300: YES;
complete-result correctness: PASS for all three variants; publishable results:
NONE; P3.4: CLOSED; P3.4 itself produces no variant or cuBLASLt comparison.
P3.5 (five shapes and comparison) — implemented; independently audited: YES;
verified on GB300: YES; complete-result correctness: PASS for all 20
shape/candidate rows; publishable results: NONE; P3.5: CLOSED; Phase 3:
CLOSED. P4.1 (Phase 4 campaign orchestrator) — P4.1: implemented; independent
audit: YES; GB300 verification: YES; pilot campaign `20260812T013848Z`:
COMPLETE; publishable results: NONE; P4.1: CLOSED. P4.2 (one accepted pilot
plus three final campaigns) — implemented; independently audited: YES;
verified on GB300: YES; three of three final campaigns terminally revalidated;
publishable results: NONE; P4.2: CLOSED. P4.3 (integrated analysis,
documentation, closing audit preparation) — P4.3: IMPLEMENTED; independent
audit: NO; production analysis: NO; publishable results: NONE.`**

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

P2.4 campaign `20260805T102759Z` was executed on GB300 at Git commit
`65f14d1069f0f04cb591ccdb9262c6222797042e`. Profiling preflight
`20260805T102944Z` reported `OVERALL=PASS`; the frozen pilot completed all
24 configurations and 720 retained samples, all 24 Nsight Compute profiles
were captured and validated, and the analysis reached `ANALYZED` with every
mandatory SM-clock check at `OK`. The independently audited empirical
per-SM ceiling candidate is `16.37244853848296 TFLOP/s/SM`, selected from
the 1-SM `N=256`, `depth=256` configuration. The best 2-SM configuration
reached `16.220558567678513 TFLOP/s/SM` and 99.16% scaling efficiency at
the same `N` and `depth`. This closes P2.4 and Phase 2, but it remains one
reviewed pilot: every artifact is `publishable: false` and no final
publishable throughput result is claimed.

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

## Phase 2 status: CLOSED (P2.1–P2.4)

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

P2.4, profiling and the empirical BF16 UMMA per-SM ceiling
(`scripts/run_exp02_umma_throughput_p24.sh`,
`scripts/analyze_exp02_umma_throughput_p24.py`,
`scripts/p24_safe_capture.py`, `scripts/p24_ncu_bridge.py`), is implemented:
a reproducible layer around the unmodified P2.3 infrastructure that drives
one frozen 24-configuration `run_kind=benchmark` pilot through the
unmodified P2.3 runner, profiles the identical 24 configurations with
Nsight Compute (an exact kernel-symbol filter with `--launch-skip 1
--launch-count 1`, profiling only the second, timed launch; clock-control-
disabled, non-defaulting profiler controls), and computes deterministic
statistics, clock-calibrated TFLOP/s, 1-SM/2-SM speedup and scaling
efficiency (never clamped), candidate depth saturation per `(method, N)`
group, and an empirical per-SM ceiling candidate selected in
clock-independent FLOP/cycle-per-SM space before any clock conversion. If
the mandatory SM-clock metric cannot be trusted for any of the 24 profiled
configurations, the campaign becomes `INCONCLUSIVE` and no TFLOP/s or
completed empirical-ceiling claim is emitted anywhere. See
`src/compute/P2_4_PROTOCOL.md` for the complete frozen contract. P2.4
introduces no new CUDA kernel and modifies no P2.1/P2.2/P2.3 file. This
implementation at Git commit
`65f14d1069f0f04cb591ccdb9262c6222797042e` passed an independent audit and
was verified end-to-end by GB300 campaign `20260805T102759Z`: the pilot
completed all 24 configurations and 720 retained samples, all 24 NCU
profiles validated, and the analysis reached `ANALYZED` with 24/24 SM-clock
checks at `OK`. The empirical per-SM ceiling candidate is
`16.37244853848296 TFLOP/s/SM`; the best 2-SM configuration reached
`16.220558567678513 TFLOP/s/SM` and 99.16% scaling efficiency at `N=256`,
`depth=256`. The optional device-wide extrapolation was not emitted because
NCU did not resolve the SM-count metric. Every artifact remains
`publishable: false` unconditionally. P2.4 and Phase 2 are closed.

## Phase 3 status: CLOSED — P3.1–P3.5 are YES / YES / YES

Phase 2 is closed and the Phase 3 gate has passed. P3.1, P3.2, P3.3, and P3.4
are closed as `YES / YES / YES`. P3.5 is also implemented, independently
audited, and functionally verified on GB300, so P3.5 and Phase 3 are closed.
No publishable Experiment 3 result exists; the P3.5 smoke remains functional
comparison evidence only. Status flags: P3.5: CLOSED; Phase 3: CLOSED.

P3.1, the pinned official CuTe DSL example, is implemented. It executes one
exact, unmodified, official NVIDIA example — `NVIDIA/cutlass` v4.6.1, commit
`e05f953a5b3d38adc240df2ff928e0421c2abba3`,
`examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py`,
BSD-3-Clause — in place from the pinned `/opt/cutlass` checkout inside the
image, checked against the upstream commit, Git blob SHA, and SHA-256. That
file is never copied, vendored, forked, reformatted, or patched into this
repository; P3.1 adds no GEMM source of its own. The two files it creates are
`src/gemm/P3_1_PROTOCOL.md` and `PHASE3_VERSIONS.env`. The frozen functional
configuration is
BF16 × BF16 → FP32 with FP32 accumulation at `(M,N,K,L) = (256,256,512,1)`,
non-persistent, 1-CTA MMA group, MMA tiler `(128,128)`, cluster `(1,1)`, TMA
loads, TMA store, and mandatory reference validation performed by the unchanged
example. The shape is deliberately small: this is a functional compatibility
check, not one of the five final shapes.

**P3.1 creates no performance result.** Any timing the example computes
internally is discarded and explicitly classified as non-publishable
functional-smoke output. No TFLOP/s, no comparison, and no cuBLASLt baseline
exists yet, and nothing here says or implies that a CuTe DSL GEMM approaches
cuBLASLt. P3.1 introduces no wrapper, no persistent variant, no 2-CTA
instruction, no sweep, no autotuning, no Nsight Compute, and no result file.

Two Make targets were added:

```bash
make gemm-cutedsl-p31-check   # GPU-free, network-free, unprivileged: verifies
                              # the upstream commit, checkout cleanliness, the
                              # example's regular-file identity, Git blob SHA,
                              # SHA-256, the CuTe DSL and PyTorch pins, and the
                              # example's own GPU-free --help.

BLACKWELL_GPU_INDEX=<physical-index> make gemm-cutedsl-p31-smoke
                              # The only P3.1 GPU target. Validates the index
                              # first, runs exclusively through
                              # scripts/run_container.sh, re-checks the
                              # upstream commit and SHA-256 inside that same
                              # container, then runs the frozen command with
                              # reference checking enabled.
```

Phase 3 has its own version contract. The global `VERSIONS.env` is unchanged —
byte-for-byte identical to `main`, as `git diff --exit-code main --
VERSIONS.env` proves — because the closed, audited P1/P2 aggregators parse it
against a closed key allowlist and reject any unknown key. Every Phase 3-only
pin therefore lives in the new root-level `PHASE3_VERSIONS.env`, which extends
that global contract and redefines nothing in it: the auxiliary PyTorch pins
(`2.10.0+cu130` from the official cu130 index, `torch.version.cuda == 13.0`),
the `cuda-python`/`cuda-bindings` pins, and the example's path, blob SHA, and
SHA-256. PyTorch is used only by NVIDIA's example for allocation, DLPack
interoperability, CUDA stream access, and the CPU reference; it does **not**
replace the pinned CUDA 13.1.0 CuTe DSL toolchain, and no existing CUDA,
digest, CUTLASS, architecture, or build-job pin changed.

The image's Python dependency graph is consistent. `torch 2.10.0+cu130`
requires `cuda-bindings==13.0.3`, so `cuda-python` and `cuda-bindings` are both
pinned to `13.0.3` — a combination that also satisfies CuTe DSL 4.6.1's own
`cuda-python>=12.8` constraint. Nothing is uninstalled, excluded, or
allowlisted to mask a conflict: `python3 -m pip check` must report
`No broken requirements found.`, and it is a hard, unsuppressed gate during the
image build, in `make check-env`, and in `make gemm-cutedsl-p31-check`. The
same three places verify exact versions: the build gate reads all four pinned
distributions (`torch`, `cuda-python`, `cuda-bindings`, `nvidia-cutlass-dsl`)
through `importlib.metadata`, and the two check targets read `cuda-python` and
`cuda-bindings` through `importlib.metadata` while re-reading
`torch.__version__`, `torch.version.cuda`, and `cutlass.__version__` at
runtime. `make check-static` additionally imports both real
aggregator modules and runs their real `parse_versions_env()` against the
repository's actual `VERSIONS.env`, so the closed P1/P2 contract cannot regress
unnoticed.

The remediated implementation at Git commit
`f34cb33a9456ba011feb0a5a35910bbd00f9a9e6` passed an independent audit and
was functionally verified on a physical NVIDIA B300 on 6 August 2026. Fresh
preflight campaign `20260806T101657Z` reported `OVERALL=PASS` on physical GPU
index `3` (UUID `GPU-90fb226c-3937-2448-1052-2e12282a61b9`); the frozen smoke
then re-checked the upstream provenance, kept reference checking enabled, and
ended with `PASS`. This closes P3.1 as `YES / YES / YES` without creating a
performance result. See `src/gemm/P3_1_PROTOCOL.md` for the frozen protocol and
the exact verification commands.

### P3.2 (one-shape wrapper) — closed (`YES / YES / YES`)

**P3.2 is implemented, independently audited, and functionally verified on
GB300.** It is closed as `YES / YES / YES` without creating a publishable
performance result.

P3.2 adds a thin, repository-owned CuTe DSL wrapper
(`src/gemm/cutedsl_gemm.py`) around the *same* pinned, unmodified official
NVIDIA example that P3.1 froze. It loads that file read-only and in place from
the pinned `/opt/cutlass` checkout after revalidating the pinned commit, Git
blob SHA, and SHA-256, and reuses `DenseGemmKernel`, `can_implement()`, and the
upstream deterministic tensor factory. It never calls the upstream `run()`,
because that function fuses compilation, the first launch, correctness, and
benchmarking into a single returned number and so cannot provide the separation
this unit exists to establish. Nothing is copied, vendored, forked, reformatted,
or patched, `/opt/cutlass` is never written to, and no key is added to
`VERSIONS.env` or `PHASE3_VERSIONS.env`: P3.2 executes P3.1's file and reuses
P3.1's pins, reading every provenance value from those two contracts at run
time.

Exactly one frozen configuration exists — BF16 × BF16 → FP32 with FP32
accumulation in TMEM at `(M,N,K,L) = (4096,4096,4096,1)` (the first of the five
final shapes), `a_major=k`, `b_major=k`, `c_major=n`, MMA tiler `(128,128)`,
cluster `(1,1)`, one-CTA MMA group, non-persistent, TMA loads, TMA store, seed
`1111`, hot reused operands, `sm_103a` — and none of it is reachable from the
command line. The only runtime controls are `--warmup-iterations`,
`--iterations`, `--self-test`, and `--help`, all bounded, and the reference
check cannot be skipped.

The wrapper separates three costs and validates correctness before timing any
of the steady state:

```text
compile_time_ms    monotonic host clock around cute.compile alone
first_launch_ms    the same clock around the first launch, whose output is
                   the tensor that gets validated
kernel_time_ms     CUDA events on the kernel's own stream after warm-up,
                   divided by the iteration count
```

Correctness compares the complete result against an untimed PyTorch CUDA
oracle (`atol=1e-1`, `rtol=1e-5`). Its FP32 policy is set through the PyTorch
2.10 `torch.backends.cuda.matmul.fp32_precision` API and nothing else — the
legacy `allow_tf32` flag and `set_float32_matmul_precision()` are never touched,
because in 2.10 those are views of the same setting and mixing them is
unsupported — and the property must read back as exactly `ieee`; the unset
`none` default is rejected, and an unavailable API fails closed with no
fallback. A failure exits non-zero with a stderr diagnostic and emits no CSV at
all, and no warm-up or steady-state timing runs.

On a successful run the smoke target's entire stdout is one CSV header and one
data row; on a failure it is empty. Every launcher line, Make diagnostic, and
progress message goes to stderr.

**These three timings are non-publishable diagnostic fields, not a result.**
Every emitted row carries `publishable=false`; no TFLOP/s, speedup, efficiency,
utilization, or bandwidth is computed anywhere; no result file or campaign
directory is written; and **no cuBLASLt comparison and no Phase 3 experimental
result exists yet**. The untimed PyTorch oracle is a correctness reference only
— it is explicitly *not* the P3.3 baseline. Nothing here says or implies that a
CuTe DSL GEMM approaches cuBLASLt.

Successful runs write exactly one CSV header and one data row to stdout under a
frozen 47-field `schema_version=p32.v1` contract produced with Python's `csv`
module; every human-readable message, including native compiler writes to
descriptor 1, goes to stderr.

The remediation at Git commit
`c8b3e2ee57e0297940e0fd5864583ec12dfb23e3` passed an independent technical
re-audit; the only remaining findings were stale status statements in the
documentation, corrected by this closure update. Fresh preflight campaign
`20260806T163806Z` then reported `OVERALL=PASS` on physical GPU index `7`
(UUID `GPU-40e00845-d89c-1393-2c32-a2dca3ee9442`), an NVIDIA B300 SXM6 AC
with compute capability 10.3 and driver 610.43.02. The frozen P3.2 smoke ran
the same clean repository commit, re-checked CUTLASS commit
`e05f953a5b3d38adc240df2ff928e0421c2abba3` and upstream SHA-256
`f99bc4cc1e0aea8990e2929d7c703dfc8196d797b7c9f5a889eabcd3c4ff67ec`,
reported `can_implement: OK`, validated the complete result with
`max_abs_error=0.0` and `max_rel_error=0.0`, completed two warm-ups and ten
measured launches, and emitted its one `p32.v1` row with `git_dirty=false` and
`publishable=false`. All three timing fields were finite and positive; they
remain non-publishable diagnostics and are not an experimental result.

Two Make targets were added:

```bash
make gemm-cutedsl-p32-check   # GPU-free, network-free, unprivileged. Runs the
                              # existing P3.1 gate first, then re-verifies the
                              # upstream identity, the pinned versions and
                              # dependency consistency, and runs the wrapper's
                              # --help/--self-test plus the checker and its own
                              # self-test inside the pinned image with the
                              # repository mounted read-only and no GPU exposed.

BLACKWELL_GPU_INDEX=<physical-index> make gemm-cutedsl-p32-smoke
                              # The only P3.2 GPU target. Verified on B300 on
                              # 6 August 2026; every rerun still requires an
                              # explicitly selected idle physical GPU.
                              # Validates the index first, runs exclusively
                              # through scripts/run_container.sh, re-checks the
                              # upstream commit and SHA-256 inside that same
                              # container, then runs the frozen one-shape
                              # configuration with 2 warm-ups and 10 measured
                              # launches.
```

See `src/gemm/P3_2_PROTOCOL.md` for the frozen protocol, the exact CSV schema,
and the verification commands.

### P3.3 (cuBLASLt baseline) — closed (`YES / YES / YES`)

**P3.3 is implemented, independently audited, and functionally verified on
GB300.** It is closed as `YES / YES / YES` without creating a publishable
result or a CuTe-versus-cuBLASLt performance comparison.

P3.3 is the vendor-library counterpart of P3.2: the same frozen BF16 geometry,
`(M,N,K,L) = (4096,4096,4096,1)` computing `C = A × Bᵀ` with FP32 accumulation,
on the same operand bytes, issued through a direct, explicit `cublasLtMatmul`
call. It adds four files — `src/gemm/cublaslt_gemm.py` (orchestration),
`src/gemm/cublaslt_bridge.cu` (a small C-ABI bridge),
`scripts/check_cublaslt_gemm_p33.py` (the fail-closed checker), and
`src/gemm/P3_3_PROTOCOL.md` (the frozen protocol).

cuBLASLt already ships inside the pinned CUDA 13.1 development image, so P3.3
adds no package, does not change the `Dockerfile`, and adds no key to
`VERSIONS.env` or `PHASE3_VERSIONS.env`; the library's runtime version is read
with `cublasLtGetVersion()` and recorded rather than pinned. No NVIDIA GEMM
implementation is copied, forked, patched, or vendored: the bridge calls only
the public cuBLASLt API of the pinned headers, defines no CUDA kernel of its
own, writes nothing to stdout, and lets no C++ exception cross the C boundary.

The explicit descriptor contract is frozen in the bridge *and* in the wrapper,
independently, and the run aborts unless the two agree exactly: A row-major
`M × K` with `lda = K`, B row-major `N × K` with `ldb = K`, C and D row-major
`M × N` with `ldc = ldd = N`, `transa = CUBLAS_OP_N`, `transb = CUBLAS_OP_T`,
`CUDA_R_16BF` in, `CUDA_R_32F` out, `CUBLAS_COMPUTE_32F`, `CUDA_R_32F` scale,
`CUBLASLT_POINTER_MODE_HOST`, `CUBLASLT_EPILOGUE_DEFAULT`, `alpha = 1`,
`beta = 0`, no bias, seed `1111`, hot reused operands. Nothing is silently
transposed, relaid out, retyped, or diverted to another GEMM interface.

Operand equivalence with P3.2 is structural, not asserted. P3.3 replicates the
pinned upstream `create_tensors()` sequence call for call — the same
`cutlass.torch.matrix` factory, the same seed applied once, the same A/B/C call
order that fixes the RNG stream, the same dtypes and strides — because the
upstream factory discards the device tensors for A and B while P3.3 must retain
every allocation to pass its pointer to cuBLASLt. A parser over the verified
upstream file fails the run if that factory ever diverges from what P3.3
replicates.

The algorithm policy is fixed and never autotuned: a 64 MiB workspace limit,
exactly 32 requested heuristic results, `CUBLASLT_SEARCH_BEST_FIT`, the first
entry whose state is `CUBLAS_STATUS_SUCCESS`, re-validated with
`cublasLtMatmulAlgoCheck()`, rejected if it needs more workspace than the
limit, given exactly the workspace it requires, and executed alone. No
candidate is ever benchmarked: the bridge has exactly one `cublasLtMatmul` call
site and no timing facility at all.

Correctness is mandatory and always precedes any timing, using the identical
untimed PyTorch CUDA oracle P3.2 uses (`fp32_precision` only, read back as
exactly `ieee`, `atol=1e-1`, `rtol=1e-5`, complete result). A failure exits
non-zero, emits no CSV at all, and runs neither warm-up nor steady state. The
wrapper separates `setup_time_ms`, `first_launch_ms`, and `kernel_time_ms`;
`setup_time_ms` is deliberately *not* `compile_time_ms`, because nothing is
compiled at run time, and the P3.2 field name is never reused. Successful runs
write exactly one CSV header and one data row to stdout under a new frozen
77-field `schema_version=p33.v1` contract; the P3.2 `p32.v1` schema is neither
modified nor reinterpreted.

```bash
make gemm-cublaslt-p33-check  # GPU-free, network-free, unprivileged. Runs the
                              # existing P3.2 gate first, then compiles the
                              # bridge for sm_103a into container-private /tmp,
                              # inspects its ELF symbols to prove the measured
                              # path references cublasLtMatmul and no fallback
                              # GEMM API, and runs the wrapper's --help and
                              # --self-test plus the checker and its own
                              # self-test, with the repository mounted
                              # read-only and no GPU exposed.

BLACKWELL_GPU_INDEX=<physical-index> make gemm-cublaslt-p33-smoke
                              # The only P3.3 GPU target. Verified on B300 on
                              # 7 August 2026; every rerun still requires an
                              # explicitly selected idle physical GPU.
                              # Validates the index first, runs exclusively
                              # through scripts/run_container.sh, compiles the
                              # bridge and re-checks the upstream commit and
                              # SHA-256 inside that same container, then runs
                              # the frozen configuration with 2 warm-ups and
                              # 10 measured launches.
```

The first independent audit of implementation commit
`bb66e3275d2f5bf1addbd14c84596b1edede977f` found two blockers. The wrapper
rejected the valid cuBLASLt `split_k=0` value and read `SPLITK_NUM` with a
signed width instead of the documented `uint32_t`; separately, an obsolete
P3.2 status assertion caused `make check-static` to fail. Remediation commit
`1c3ade8a39ae1e19882514e2b06094a418eb70bf` fixed both findings, added
adversarial regression coverage, and corrected the associated stale P3.2
documentation. The remediated tree passed `make check-static`, the audit
findings were rechecked with no remaining blocker, and the operator confirmed
that the Docker-backed `make gemm-cublaslt-p33-check` gate passed on the same
clean commit.

Fresh preflight campaign `20260807T144123Z` then reported `OVERALL=PASS` on
physical GPU index `7`, UUID `GPU-40e00845-d89c-1393-2c32-a2dca3ee9442`, an
NVIDIA B300 SXM6 AC with compute capability 10.3 and driver 610.43.02. An
initial invocation with an unset GPU-index variable exited with status 2 before
using a GPU, demonstrating the fail-closed launcher path; it is not the smoke
evidence. The valid rerun with `BLACKWELL_GPU_INDEX=7` executed clean commit
`1c3ade8a39ae1e19882514e2b06094a418eb70bf`, revalidated the upstream
CUTLASS provenance, compiled the bridge in the selected GPU container, called
`cublasLtMatmul` directly, and passed complete-result correctness with
`max_abs_error=0.0` and `max_rel_error=0.0`. The heuristic returned eight
supported entries from 32 requested and selected index 0 (`algo_id=66`,
`tile_id=23`, `stages_id=35`, `split_k=1`, workspace 0 bytes). The frozen two
warm-ups and ten measured launches completed, and stdout contained exactly one
77-field `p33.v1` row with `git_dirty=false` and `publishable=false`. Its three
finite positive timing fields are non-publishable diagnostics, not an
experimental result.

**Every emitted row carries `publishable=false`.** No TFLOP/s, speedup,
efficiency, utilization, bandwidth, or winner label is computed anywhere, no
result file or campaign directory is written, and P3.3 makes no comparison
against P3.2 — that comparison is P3.5's. P3.5 is now closed, but its smoke
remains non-publishable functional comparison evidence. The GPU-free checks listed in
`src/gemm/P3_3_PROTOCOL.md` section 13 were run by
the author and passed; those are the author's own self-checks, **not** an
independent audit. See `src/gemm/P3_3_PROTOCOL.md` for the frozen protocol, the
exact CSV schema, the recorded algorithm metadata, and the verification
commands.

### P3.4 (three execution variants) — closed as YES / YES / YES

**P3.4 is implemented, independently audited, and functionally verified on
GB300. All three frozen variants passed complete-result correctness. There is
still no variant-versus-variant or CuTe-versus-cuBLASLt performance comparison
anywhere in this repository, and no publishable result.**

P3.4 adds the two remaining frozen CuTe DSL execution variants alongside the one
P3.2 established, so that all three exist under one identical operand set, one
identical correctness oracle, and one identical timing discipline, at the same
single shape `(M,N,K,L) = (4096,4096,4096,1)`:

| Variant | Upstream class | Scheduler | MMA tiler | Cluster | `use_2cta_instrs` |
|---------|----------------|-----------|-----------|---------|-------------------|
| `nonpersistent_1cta` | `DenseGemmKernel` | non-persistent | `(128,128)` | `(1,1)` | `false` |
| `persistent_1cta` | `PersistentDenseGemmKernel` | static persistent | `(128,128)` | `(1,1)` | `false` |
| `persistent_2cta` | `PersistentDenseGemmKernel` | static persistent | `(256,128)` | `(2,1)` | `true` |

The 2-CTA row deliberately uses an M tile of 256 so each of the two
participating CTAs keeps a local M extent of 128 — P2.2's two-SM geometry, and
the shape NVIDIA's own persistent example documents for `use_2cta_instrs=True`.
Exactly three fixed candidates run, always in that order: no autotuning, no
candidate search, no fourth candidate, and none of the other four final shapes.

It adds three files — `src/gemm/cutedsl_variants.py` (the wrapper),
`scripts/check_cutedsl_variants_p34.py` (the fail-closed checker), and
`src/gemm/P3_4_PROTOCOL.md` (the frozen protocol).

This repository still owns no GEMM kernel. The non-persistent variant keeps
using P3.1's pinned example, and the two persistent variants use the official
static-persistent example `dense_gemm_persistent.py` from the **same** already
pinned CUTLASS commit (BSD-3-Clause). Both files are loaded read-only and in
place from `/opt/cutlass`, under private module names, after HEAD, checkout
cleanliness, regular-file identity, Git blob, and SHA-256 are verified for each
file; neither upstream `run()` nor either upstream benchmarking helper is ever
called. The only new pins are the three `CUTEDSL_P34_*` keys in
`PHASE3_VERSIONS.env` — `VERSIONS.env`, the `Dockerfile`, and every closed
P3.1/P3.2/P3.3 interface are untouched, and the checker re-runs both closed
P1/P2 aggregator parsers to prove `VERSIONS.env` still satisfies their
allowlist.

All three variants consume byte-identical A and B: the operands are built once
by the pinned non-persistent example's own `create_tensors()` — same factory,
same seed `1111`, same A/B/C order, same dtypes and strides as P3.2 and P3.3 —
and are never mutated. Only C is reset between candidates, to NaN, outside every
timer, so an element a kernel fails to write stays non-finite and is rejected
rather than surviving as a stale value. `max_active_clusters` comes from the
official pinned hardware helper for each variant's own cluster size and is never
guessed; the non-persistent row records the canonical `not_applicable`.

Per variant, the wrapper separates `compile_time_ms`, `first_launch_ms`, and
`kernel_time_ms`, validates the complete result against the identical untimed
PyTorch CUDA oracle P3.2 and P3.3 use, and only then runs that variant's warm-up
and steady state. The whole output is buffered: a successful run writes exactly
one CSV header and exactly three rows — four lines — under a new frozen
51-field `schema_version=p34.v1` contract, and a failure in **any** of the three
positions emits no CSV at all, including no rows from variants that already
passed.

```bash
make gemm-cutedsl-p34-check   # GPU-free, network-free, unprivileged. Runs the
                              # existing P3.3 gate first, then revalidates the
                              # CUTLASS checkout and BOTH pinned official
                              # sources, asserts the persistent file really
                              # carries StaticPersistentTileScheduler and
                              # CtaGroup.TWO, checks the pinned package versions
                              # and pip check, and runs the wrapper's --help and
                              # --self-test plus the checker and its own
                              # self-test, with the repository mounted read-only
                              # and no GPU exposed.

BLACKWELL_GPU_INDEX=<physical-index> make gemm-cutedsl-p34-smoke
                              # The only P3.4 GPU target. Verified on B300 on
                              # 7 August 2026.
                              # Validates the index first, runs exclusively
                              # through scripts/run_container.sh, re-checks both
                              # upstream sources inside that same container,
                              # then runs all three variants with 2 warm-ups and
                              # 10 measured launches each.
```

**Every emitted row carries `publishable=false`.** No TFLOP/s, speedup,
efficiency, utilization, bandwidth, ranking, or winner label is computed
anywhere, no result file or campaign directory is written, and P3.4 compares
nothing — neither the variants against each other nor against the P3.3 cuBLASLt
baseline. That comparison is P3.5's. P3.5 is now closed, but its smoke remains
non-publishable functional comparison evidence. The GPU-free acceptance commands listed in
`src/gemm/P3_4_PROTOCOL.md` section 12 were run by
the author and passed; those are the author's own self-checks, **not** an
independent audit, and GPU-free checks are **not** GB300 verification. An
independent audit of implementation commit `bb8cdc5b` found no blocking defect.
The operator subsequently ran a fresh passing preflight and the frozen smoke on
an explicitly selected idle physical NVIDIA B300 at index 4. Both upstream
sources were revalidated; all three candidates passed `can_implement()`,
compiled and launched, completed two warm-ups and ten measured launches, and
reported zero maximum absolute and relative error. The official helper returned
148 active clusters for `persistent_1cta` and 74 for `persistent_2cta`. The
four-line `p34.v1` CSV contained the three variants in frozen order with
`correctness=PASS` and `publishable=false` throughout. Those timings remain
functional diagnostics only.

### P3.5 (five shapes and comparison) — closed as YES / YES / YES

**P3.5 is implemented, independently audited, and functionally verified on
GB300. All five shapes and all four candidates passed complete-result
correctness. P3.5 creates no publishable result, and nothing here claims that
any CuTe DSL variant approaches or beats cuBLASLt.**

P3.5 extends the already verified P3.3/P3.4 infrastructure to all five final
Experiment 3 shapes and performs the first explicit, purely **descriptive**
comparison among four candidates per shape.

| # | Shape `(M,N,K,L)` | `shape_id` | `flop_count` = 2·M·N·K |
|---|-------------------|-----------|------------------------|
| 1 | `(4096, 4096, 4096, 1)` | `4096x4096x4096x1` | 137,438,953,472 |
| 2 | `(8192, 8192, 8192, 1)` | `8192x8192x8192x1` | 1,099,511,627,776 |
| 3 | `(16384, 512, 4096, 1)` | `16384x512x4096x1` | 68,719,476,736 |
| 4 | `(32768, 512, 4096, 1)` | `32768x512x4096x1` | 137,438,953,472 |
| 5 | `(512, 16384, 4096, 1)` | `512x16384x4096x1` | 68,719,476,736 |

| # | Method | Variant | Notes |
|---|--------|---------|-------|
| 1 | `cutedsl` | `nonpersistent_1cta` | `DenseGemmKernel`, tiler `(128,128)`, cluster `(1,1)` |
| 2 | `cutedsl` | `persistent_1cta` | `PersistentDenseGemmKernel`, tiler `(128,128)`, cluster `(1,1)` |
| 3 | `cutedsl` | `persistent_2cta` | `PersistentDenseGemmKernel`, tiler `(256,128)`, cluster `(2,1)` |
| 4 | `cublaslt` | `heuristic_first_supported` | the comparison baseline |

Output is **shape-major**: the five shapes in that order, the four candidates in
that order inside each, for exactly 20 rows and 21 lines. **No arbitrary shape
is reachable** from the command line, the environment, a configuration file, or
an input CSV: the Python wrapper and the C bridge freeze the same five
geometries independently, the bridge exposes its own allowlist through
`p35_shape_count()` / `p35_shape_at()`, and the wrapper reads it back and
requires the two to be identical before any measurement runs. A geometry outside
that allowlist never reaches a cuBLASLt descriptor, a heuristic query, or a
launch.

It adds four files — `src/gemm/gemm_comparison.py` (the wrapper),
`src/gemm/cublaslt_bridge_p35.cu` (the P3.5 C-ABI bridge),
`scripts/check_gemm_comparison_p35.py` (the fail-closed checker), and
`src/gemm/P3_5_PROTOCOL.md` (the frozen protocol).

This repository still owns no GEMM kernel, and **P3.5 adds no pin**: the three
CuTe DSL candidates use the same two pinned official NVIDIA examples P3.4 uses,
and the cuBLASLt candidate uses the library that already ships inside the pinned
CUDA 13.1 image. `VERSIONS.env`, `PHASE3_VERSIONS.env`, the `Dockerfile`, and
`scripts/run_container.sh` are untouched, and every closed P3.1–P3.4 executable
keeps its CLI, schema, field order, Make targets, one-shape restriction, output
behaviour, correctness and provenance checks, and smoke semantics unchanged.

The cuBLASLt policy is exactly the closed P3.3 policy — 64 MiB workspace limit,
32 requested heuristic results, `CUBLASLT_SEARCH_BEST_FIT`, the first supported
entry, re-validated with `cublasLtMatmulAlgoCheck()`, no fallback GEMM API, and
no autotuning by execution. A different supported algorithm may naturally be
selected for each shape; the *selection policy* never changes. The bridge has
exactly one `cublasLtMatmul` call site, contains no CUDA kernel and no timing
facility, prints nothing, lets no C++ exception cross the C boundary, and
validates every dimension and derived byte size against overflow before creating
a descriptor.

Per shape the operands are built once by the pinned non-persistent example's own
`create_tensors()` — same factory, seed `1111`, A/B/C order, dtypes and strides
as P3.2–P3.4 — and are never mutated; the untimed IEEE-FP32 CUDA oracle is
computed once and reused by all four candidates; the output buffer is reset to
NaN outside every timer before each candidate; and **no candidate runs warm-up
or steady-state timing until its complete result has passed**. Per candidate the
wrapper separates `compile_time_ms` (CuTe DSL JIT only) *or* `setup_time_ms`
(cuBLASLt plan creation only) — never both, each carrying the canonical
`not_applicable` on the other method's rows, and never compared against each
other — plus `first_launch_ms` and, from CUDA events on that candidate's own
stream, `kernel_time_ms`.

Only `kernel_time_ms` participates in the comparison:

```text
flop_count                   = 2 × M × N × K                     (exact integer)
tflops                       = flop_count / (kernel_time_ms × 1e9)
throughput_ratio_vs_cublaslt = candidate_tflops / cublaslt_tflops
gap_to_cublaslt_pct          = 100 × (1 − throughput_ratio_vs_cublaslt)
```

A **positive** gap means the candidate is slower than cuBLASLt, **zero** means
equal, and a **negative** gap means the candidate is faster — and negative values
are **never clamped**. Beating cuBLASLt is **not** a success criterion. The
cuBLASLt row is the baseline and carries a ratio of exactly 1 and a gap of
exactly 0. Candidates are ranked by full-precision `kernel_time_ms` with an exact
tie broken by the frozen candidate order; `best_cutedsl_variant` is selected from
the three CuTe DSL candidates only and repeated identically on all four rows of
the shape; exactly one CuTe DSL row carries `is_best_cutedsl=true`. **No
confidence interval, p-value, outlier removal, roofline efficiency,
empirical-ceiling utilization, bandwidth, arithmetic-intensity classification, or
causal interpretation is computed anywhere.**

The whole output is buffered under a new frozen 100-field
`schema_version=p35.v1` contract — the closed `p32.v1`, `p33.v1`, and `p34.v1`
schemas are neither modified nor reinterpreted — and a failure at **any** shape
or candidate emits no CSV at all, including rows already completed.

```bash
make gemm-comparison-p35-check   # GPU-free, network-free, unprivileged. Runs the
                                 # existing P3.4 gate first, then revalidates the
                                 # CUTLASS checkout and BOTH pinned official
                                 # sources, checks the pinned package versions and
                                 # pip check, compiles the P3.5 bridge into
                                 # container-private /tmp and inspects its ELF
                                 # symbols and dynamic dependencies (cublasLtMatmul
                                 # present, no fallback GEMM API present), and runs
                                 # the wrapper's --help and --self-test plus the
                                 # checker and its own self-test, with the
                                 # repository mounted read-only and no GPU exposed.

BLACKWELL_GPU_INDEX=<physical-index> make gemm-comparison-p35-smoke
                                 # The only P3.5 GPU target. Verified on B300 on
                                 # 8 August 2026.
                                 # Validates the index first, runs exclusively
                                 # through scripts/run_container.sh, re-checks both
                                 # upstream sources and compiles the bridge inside
                                 # that same container, then runs all five shapes ×
                                 # four candidates with 2 warm-ups and 10 measured
                                 # launches each.
```

The first independent technical audit of implementation commit `61d17845`
found two fail-closed cleanup paths that could warn and continue after native
plan-destruction or shape-resource cleanup failure. Remediation commit
`b76c774473b85a498d8e8872296594cae472d498` propagated those failures, checked
all native destructor statuses, preserved an existing primary exception, and
added adversarial regression coverage. Post-remediation review found no
remaining blocker. The full GPU-free, network-free
`make gemm-comparison-p35-check` gate then passed on the corrected tree: both
pinned sources and the dependency graph were revalidated, the bridge compiled
in private `/tmp`, and ELF inspection confirmed `cublasLtMatmul` with no
fallback GEMM API. These checks support the audit but are not themselves an
independent audit or GB300 verification.

On 8 August 2026, the operator ran a fresh passing preflight and the frozen
smoke on an explicitly selected idle physical NVIDIA B300 at index 4 (UUID
`GPU-4ae7e013-1aac-31d8-8b8e-c27530f1c6ed`, driver `610.43.02`). The run used
the clean corrected commit `b76c774473b85a498d8e8872296594cae472d498`,
revalidated both pinned upstream sources, compiled the bridge in private
`/tmp`, and executed all five shapes × four candidates with two warm-ups and
ten measured launches each. All 20 complete-result checks passed with zero
maximum absolute and relative error. Stdout contained exactly one 100-field
header and 20 rows in frozen shape-major order; every row recorded
`git_dirty=false`, `correctness=PASS`, and `publishable=false`. The comparison
fields remain arithmetic, non-publishable smoke diagnostics: no final campaign,
statistical conclusion, Nsight Compute analysis, or Phase 4 interpretation was
performed. The `NamedBarrier ID 0` warning was non-fatal. P3.5 and Phase 3 are
therefore closed as `YES / YES / YES`.

Implementing P3.5 also required correcting three stale frontier guards that the
P3.4 closure had already superseded: the `Makefile` and
`scripts/check_cublaslt_gemm_p33.py` still demanded that P3.4 be *unclosed*, and
the `Makefile`, `scripts/check_cublaslt_gemm_p33.py`, and
`scripts/check_cutedsl_variants_p34.py` all required the literal PLAN.md row
`P3.5 | Five shapes and comparison | NO | NO | NO`, which structurally forbade
P3.5 from ever being implemented. At the P3.5 baseline commit `b50dca3` both
`make check-static` and `python3 scripts/check_cublaslt_gemm_p33.py .` therefore
already failed, before any P3.5 file existed. At implementation time all three
guards were advanced to the then-truthful state — P3.4 closed and P3.5
implemented but not yet audited or verified. This closure advances the
P3.5-owned status guard to `YES / YES / YES`; none is weakened, and all still
reject impossible partial states. See `src/gemm/P3_5_PROTOCOL.md` section 12.3.

## Phase 4 status: P4.1 and P4.2 closed as YES / YES / YES

### P4.1 (Phase 4 campaign orchestrator) — closed

**P4.1: CLOSED; independent audit: YES; GB300 verification: YES.** Pilot
campaign `20260812T013848Z` completed all nine stages on GB300 and remains
`publishable=false`. P4.2's frozen protocol, three final campaigns, terminal
revalidations, and cross-campaign validator have since closed independently as
`YES / YES / YES`. No publishable result exists. P4.3 (integrated analysis,
documentation, and the closing audit preparation) is implemented but neither
independently audited nor run against the real evidence.

`scripts/run_all.sh` is the **only** public Phase 4 orchestration entry point.
It coordinates one reproducible top-level campaign across the three closed
experiments by *composing* their already audited entry points — it reimplements
no kernel, scientific matrix, statistic, profiler plan, correctness rule,
schema, or execution parameter, adds no external dependency, no version pin,
and no Nsight Compute case:

| Experiment | Composed unit | Driven through |
|------------|---------------|----------------|
| 1. LDGSTS versus TMA | P1.4 | `make memory-paths-p14-pilot` / `-profile` / `-analyze` |
| 2. BF16 UMMA throughput | P2.4 | `make compute-umma-p24-pilot` / `-profile` / `-analyze` |
| 3. CuTe DSL GEMM versus cuBLASLt | P3.5 | `make gemm-comparison-p35-smoke` |

Experiment 3 stays **one atomic `gemm` stage**: P3.5 is a single comparison of
five frozen shapes × four candidates (exactly 20 rows) over shared, immutable
operands with shape-major ordering and all-or-nothing output, so a separate
`--only cublaslt` or `--only cutedsl` mode is **forbidden** and rejected by
name. The supported selector is `--only gemm`.

```text
scripts/run_all.sh --help
scripts/run_all.sh --self-test

scripts/run_all.sh --dry-run \
    --campaign-id <YYYYMMDDTHHMMSSZ> --campaign-kind <pilot|final> \
    [--only <memory|umma|gemm>]

BLACKWELL_GPU_INDEX=<physical-index> scripts/run_all.sh \
    --campaign-id <YYYYMMDDTHHMMSSZ> --campaign-kind <pilot|final> \
    [--only <memory|umma|gemm>] [--resume]
```

The deterministic full plan is `preflight`, `memory.pilot`, `memory.profile`,
`umma.pilot`, `umma.profile`, `gemm.capture`, `memory.analyze`, `umma.analyze`,
`campaign.validate`; `--dry-run` prints it byte-identically and executes
nothing. `--campaign-id` is a required, explicit canonical UTC timestamp — a
campaign ID is never generated implicitly — and `--campaign-kind`
(`pilot`/`final`) is required and immutable. **`campaign-kind=final` does not
make any raw evidence publishable**: it is a label for P4.2's bookkeeping.
Duplicate options, missing values, unexpected positional arguments, and unknown
options exit 2. `--help`, `--self-test`, and a new-campaign `--dry-run` are
genuinely GPU-free: no Docker, no `nvidia-smi`, no CUDA/PyTorch/CuTe DSL
import, no network, no results directory, no manifest or log write, and no Git
mutation.

Every real invocation with pending GPU work requires an explicit numeric
`BLACKWELL_GPU_INDEX`, rejected before Docker, `nvidia-smi`, result creation, or
any GPU-related subprocess. The preflight runs only through
`BLACKWELL_GPU_INDEX=<i> make preflight`, and the exact
`results/preflight/<TS>/summary.json` **that invocation** produced is identified
from its own stdout markers — never `ls -t`, a "latest" symlink, glob ordering,
or a modification time — then validated as a non-symlink, non-empty regular
JSON file with `overall_status=PASS`, `git_dirty=false`, the current clean
commit, the explicitly selected GPU's UUID, compute capability `10.3`, logical
index `0`, and all six required checks at `PASS`. That exact summary path is
then passed to every remaining P1.4/P2.4 stage. This unit never calls Docker or
`nvidia-smi` itself, never selects a GPU automatically, and adds no profiler
route.

The top-level campaign tree lives at `results/raw/phase4/<campaign_id>/`
(`manifest/`, `plan.json`, `logs/`, `exp03/gemm_comparison.csv`) under the
existing blanket `results/raw/` Git-ignore rule. The manifest is an
**append-only, hash-chained revision history**, never one mutable
`manifest.json`, and records only allowlisted information — schema version,
campaign ID and immutable kind, immutable scope, the clean Git commit, stage
order and per-stage status, repository-relative evidence paths, SHA-256 hashes,
allowlisted GPU identity, the validated preflight reference, timestamps, any
failure or interruption stage, and `publishable=false`. It never stores
usernames, home paths, host names, environment dumps, credentials, SSH
material, process information, host command lines, or dynamic power, clock,
temperature, or utilization telemetry. Every experimental Make target is invoked
with `--silent --no-print-directory` so no recipe line — and therefore no
absolute bind-mount source — is ever echoed into a captured log, and the durable
text logs and failure details replace this checkout's own root with the stable
token `<repo-root>`; ordinary child diagnostics remain unchanged and P3.5's
scientific CSV stdout is copied byte for byte. The experiment-owned raw trees are
referenced by validated repository-relative path and hash rather than copied,
and all five trees share one campaign ID. Symlinks and unexpected file types are
rejected everywhere; evidence is never overwritten; publication is no-clobber;
valid partial evidence survives an interruption or a child failure; and
everything is re-hashed and revalidated immediately before a stage or campaign
is declared complete.

`--resume` is evidence-driven, not existence-driven: each apparently complete
stage is re-loaded through the underlying unit's **own** semantic loader
(`load_p14_manifest_chain` / `load_p24_manifest_chain`) and its own
`verify_campaign_evidence_integrity()`, or P3.5's own
`validate_serialized_output`, re-hashed, and rechecked against the immutable
campaign identity before it is skipped, and a fresh preflight is created and
validated whenever GPU work is still pending. A unit accepted in a terminal
state stays **pinned** to the exact manifest revision it was accepted at — its
repository-relative path, revision number, SHA-256, and the digest of its own
evidence-integrity snapshot plus freshly checked hashes of every canonical
terminal `analysis/` artifact — so a later terminal revision, a changed
revision, or changed raw or derived evidence is rejected instead of being
adopted silently.
Every component must also agree with the current preflight and with each other
on Git commit, clean-tree status, GPU UUID and name, compute capability, and —
where the closed schemas expose them — the CUDA driver and runtime API
versions, compared after normalizing the two textual formats those schemas use.
An existing campaign without `--resume` is rejected, `--resume` for a missing
campaign is rejected, a tampered artifact aborts rather than being regenerated,
and a resume of an already complete campaign is a pure read-only revalidation.
An interruption — including one inside `campaign.validate` — preserves every
artifact and log already created, exits 130, and leaves that stage eligible for
a new attempt, which always claims the next free attempt number rather than
reusing an existing log. An attempt is recorded only when both named logs
exist; a lone log left by an interruption remains preserved but unreferenced.
A P2.4
`INCONCLUSIVE` analysis propagates to a non-complete top-level outcome and is
never accepted as a complete final campaign. **P4.1 creates no Phase 4
variability threshold, publication threshold, or scientific acceptance rule**;
integrated tables, interpretation, publication decisions, and final figures
belong exclusively to P4.3.

```bash
make phase4-p41-plan    # GPU-free. Prints the deterministic full-campaign plan
                        # through the real --dry-run path; creates nothing.
make phase4-p41-check   # GPU-free, container-free, network-free: no Docker, no
                        # nvidia-smi, no CUDA compilation, no GPU. Runs the
                        # container-free memory-paths-p14-check and
                        # compute-umma-p24-check-gpu-free gates first, then the
                        # shell/Python syntax checks, the public CLI surface,
                        # the synthetic acceptance suite, and the full
                        # repository contract check. The checker walks this
                        # target's real transitive Make prerequisites and fails
                        # if any recipe in that closure would invoke a container
                        # runtime, nvidia-smi, or a CUDA compiler.
```

Implementing P4.1 required advancing one stale frontier assertion that the P4.1
row itself owned: the `Makefile` and `scripts/check_gemm_comparison_p35.py`
both required the literal `PLAN.md` row `P4.1 | Orchestrator | NO | NO | NO`,
which structurally forbade P4.1 from ever being implemented. Both were advanced
to the then-truthful `YES | NO | NO`. This closure advances the P4.1-owned
status assertions in the `Makefile`, the P3.5 checker, and the P4.1 checker to
`YES | YES | YES`; none is weakened and stale and impossible partial states
remain rejected. At that P4.1 closure P4.2 and P4.3 were still required to
remain unimplemented; both rows have since advanced under their own units. The
author's GPU-free checks in `src/phase4/P4_1_PROTOCOL.md` were not treated as an
independent audit or as GB300 verification.

The independent review covered implementation commit
`129c20c1eeb11b11076e765fa1e59a73831d6f2d` and its two remediation rounds. The
final re-audit accepted corrected commit
`77908dae377fb131ff90baef455c79d6d2c28b0b` with no remaining blocker. On 12
August 2026, after both `make phase4-p41-check` and `make check-static` passed,
the operator selected idle physical GPU index 7 (NVIDIA B300 SXM6 AC, driver
`610.43.02`) and ran the full nine-stage pilot `20260812T013848Z` from that
same clean commit. The process exited 0. A subsequent `--resume` revalidated
and skipped every stage, reported terminal state `COMPLETE`, exited 0, and
confirmed that no artifact was rewritten and no manifest revision was
appended. This closes P4.1 as `YES / YES / YES` and supplies P4.2's required
pilot. At that P4.1 closure, all evidence remained `publishable=false` and the
three P4.2 final campaigns had not yet run. P4.2 has since closed independently,
and P4.3's offline analysis layer is now implemented but not yet audited or run.

### P4.2 (one accepted pilot plus three final campaigns) — closed

**P4.2: CLOSED; independent audit: YES; GB300 verification: YES.** P4.2 froze
the final-campaign policy before collecting the final population and added one
small GPU-free checker for the invariants P4.1 cannot see when validating one
campaign at a time. It added no runner: `scripts/run_all.sh` remains the
**only** public Phase 4 execution entry point, and no Make target can start a
campaign. See `src/phase4/P4_2_PROTOCOL.md` for the complete frozen contract
and closure record.

The final independent review accepted execution commit
`b08e45c2636a3ac17c94ad8b1368084914196d7a` with no remaining blocker. The
accepted pilot `20260812T013848Z` remains `publishable=false`, is not a final
replicate, and contributes nothing to P4.3's aggregation. On 17 August 2026,
the three declared finals ran sequentially from that same clean commit and
completed the frozen full nine-stage plan:

```text
20260817T110330Z
20260817T111310Z
20260817T112011Z
```

Each campaign used `campaign_kind=final` and `scope=full`; no `--only` selector,
replacement campaign, or fourth final was used. Terminal `--resume`
revalidated and skipped every completed stage for all three finals, exited
successfully, and confirmed that no artifact was rewritten and no manifest
revision was appended.

The final read-only population check was:

```bash
python3 scripts/check_phase4_campaigns_p42.py \
  --campaign-root results/raw/phase4 \
  --pilot-campaign-id 20260812T013848Z \
  --final-campaign-id 20260817T110330Z \
  --final-campaign-id 20260817T111310Z \
  --final-campaign-id 20260817T112011Z
```

It deeply revalidated every completed stage through P4.1, confirmed the three
finals share one frozen execution commit, plan, stage order, physical GPU
identity, and comparable exposed provenance, excluded the pilot from the final
replicate set, and found no undeclared final campaign. It ended with:

```text
check_phase4_campaigns_p42: evidence: OK (1 accepted pilot + 3 accepted final campaigns; nothing was written, repaired, resumed, or regenerated; no statistic was computed)
```

The repository gate remains fast and GPU-free:

```text
python3 scripts/check_phase4_campaigns_p42.py --self-test
python3 scripts/check_phase4_campaigns_p42.py .
make phase4-p42-check
```

The checker reuses P4.1's audited manifest-chain loader, descriptor-anchored
opens, hashing primitives, and terminal stage revalidation. It adds no
dependency or version pin, and it keeps the SHA-256 pins for
`scripts/run_all.sh` and `scripts/phase4_orchestrator.py`; the documentary
closure changes neither execution file.

The P4.2-owned status guards in the `Makefile`, the P3.5 checker, the P4.1
checker, and the P4.2 checker now require `YES / YES / YES` and reject every
stale or impossible partial state. Every campaign remains `publishable=false`;
P4.2 computed no statistic, threshold, ranking, table, figure, or conclusion,
and no publishable Phase 4 result exists.

### P4.3 (integrated analysis, documentation, closing audit preparation)

**P4.3: IMPLEMENTED; independent audit: NO; production analysis: NO.**
**The P4.3 independent audit has not been performed**, and no production
analysis of the three final campaigns has been run in this repository. No
curated P4.3 artifact exists, no P4.3 result has been accepted for publication,
and no publishable result exists anywhere here. Phase 4 and the complete TFM
stay open.

P4.3 (`scripts/analyze_phase4_p43.py`,
`scripts/check_phase4_integration_p43.py`, `src/phase4/P4_3_PROTOCOL.md`) is an
offline, **read-only** analysis layer over the already accepted GB300 evidence.
It executes no GPU command and starts no Docker container, `nvidia-smi`, CUDA
compilation, Nsight Compute run, preflight, or campaign — it starts no child
process at all. It adds no CUDA, CuTe DSL, or cuBLASLt code, no shape,
candidate, matrix, layout, dtype, tile, cluster, or algorithm, no runner or
resume path, no measurement parameter, no profiler case, no raw or existing
analysis schema change, no external dependency, and no version pin;
`scripts/run_all.sh` and `scripts/phase4_orchestrator.py` stay byte-identical to
the content P4.2 pinned by SHA-256, and nothing under `results/raw/` is ever
written.

The frozen population is restated as literal constants and never discovered:
the accepted pilot `20260812T013848Z` is recorded **only** as excluded
qualification provenance, and the three final campaigns `20260817T110330Z`,
`20260817T111310Z`, and `20260817T112011Z` are the statistical population.
Before a single scientific value is read, the whole population is revalidated
through P4.2's own read-only `check_campaign_evidence()`, which delegates every
per-campaign decision to P4.1's audited loader, hashing primitives, and terminal
revalidation; each captured GEMM table is validated by P3.5's own canonical
validator. The canonical terminal P1.4, P2.4, and P3.5 artifacts each campaign's
manifest pins are then read and re-verified against that pin.

**The independent replicate is one complete final campaign.** Across the three
campaign-level values only `campaign_count = 3`, mean, median, sample standard
deviation (`n - 1`), coefficient of variation where mathematically meaningful,
minimum, and maximum are computed. Internal timing repetitions are never pooled,
nothing is ever excluded, no outlier filter runs, no p-value or significance
claim is produced, and no confidence interval is bootstrapped from three
campaigns. The strict `CV > 5.0%` threshold is a review diagnostic only; no
coefficient of variation is computed for a signed or zero-centred quantity such
as `gap_to_cublaslt_pct`; and every ratio is formed inside a campaign before it
is summarized, never from two aggregates. Saturation candidates, ceiling
selections, and best CuTe DSL variants are reported as a consensus only when all
three campaigns agree, and otherwise as not stable. A device-wide estimate is
emitted only with validated, agreeing SM-count evidence in all three campaigns;
otherwise a structured `unavailable` result carries the exact reason and no SM
count is imported from a specification, hard-coded, or inferred. Negative GEMM
gaps and scaling values outside `[0, 100]` are never clamped.

The generated report separates directly measured results, deterministic derived
quantities, modeled estimates, interpretations, and quantities that are
unavailable from the collected evidence. It states explicitly that the memory
benchmark is not a direct measurement of GEMM memory traffic, that the UMMA
ceiling is a one-/two-SM empirical microbenchmark ceiling rather than an
architectural peak, that the GEMM timing is hot-cache, and that P3.5 contains no
GEMM Nsight Compute profile — so no bottleneck attribution, roofline placement,
or arithmetic-intensity classification is made.

Three GPU-free Make targets were added: `phase4-p43-check` (fast,
container-free, network-free, no prerequisites, and independent of
`results/raw/`), `phase4-p43-analyze`, and `phase4-p43-verify`. The latter two
require the real accepted evidence explicitly, name only the frozen campaign
IDs, and can neither start nor resume a campaign. The curated tree
(`results/phase4/`) is written no-clobber, never overwrites a differing
artifact, verifies an existing byte-identical one instead of rewriting it, and
is checked byte for byte by a full recomputation.

The implementation-time GPU-free checks in `src/phase4/P4_3_PROTOCOL.md`
section 12 are the author's own self-checks. **They are not an independent
audit, and a GPU-free check is never GB300 verification.**

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
`publishable: false` pending later final campaigns. Experiment 2 is closed:
P2.1, P2.2, P2.3, and P2.4 are implemented, independently audited, and
verified on GB300. Reviewed P2.4 pilot `20260805T102759Z` reached
`ANALYZED` and established the non-publishable empirical per-SM ceiling
candidate described above, so the Phase 2 gate has passed and Phase 3 has
completed. Of experiment 3, P3.1 (executing the pinned official NVIDIA CuTe DSL
example unchanged) is implemented, independently audited, and functionally
verified on GB300; P3.2 (the frozen one-shape wrapper) is likewise implemented,
independently audited, and functionally verified on GB300. P3.3 (the equivalent
cuBLASLt baseline) is also implemented, independently audited, and functionally
verified on GB300. P3.4 (the three frozen CuTe DSL execution variants) is also
implemented, independently audited, and functionally verified on GB300. P3.5
(the five final shapes and the first descriptive four-candidate comparison) is
also implemented, independently audited, and functionally verified on GB300.
All five units and Phase 3 are closed.
The repository still contains no
publishable bandwidth, throughput, GEMM-performance, or cuBLASLt-comparison
result. The pinned CUDA 13.1, CUTLASS v4.6.1, and `sm_103a` contract in
`VERSIONS.env` remains unchanged and untouched; P3.1's own pins — the exactly
pinned auxiliary Python dependencies and the upstream example's provenance
values — live in the separate `PHASE3_VERSIONS.env`. See `PLAN.md` for the
remaining schedule and `AGENTS.md` for the mandatory shared-cluster rules.
