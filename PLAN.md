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

Phase 0 has been independently audited. Its executable units were
successfully verified on the target GB300 hardware on 20 July 2026.

## Phase 0 — Contract, environment, launcher, smoke (17–19 July 2026)

Gate: every P0 unit implemented and independently audited, and every
executable P0 unit verified on GB300 before Phase 1 begins.

| Unit | Description | Implemented | Audited | Verified on GB300 |
|------|-------------|-------------|---------|-------------------|
| P0.1 | Contract and repository (AGENTS.md, README.md, PLAN.md, LICENSE, .gitignore, VERSIONS.env) | YES | YES | N/A |
| P0.2 | Reproducible CUDA 13.1 + CuTe DSL environment (Dockerfile, image pinning) | YES | YES | YES |
| P0.3 | Safe one-GPU launcher and preflight (run_container.sh, preflight.sh, Makefile) | YES | YES | YES |
| P0.4 | CUDA, CuTe DSL, and NCU smoke checks (cuda_smoke.cu, cutedsl_smoke.py, ncu step in preflight) | YES | YES | YES |

## Phase 1 — LDGSTS versus TMA (20–26 July 2026)

Gate: Phase 0 gate passed; correctness validated before any timing/profiling.

| Unit | Description | Implemented | Audited | Verified on GB300 |
|------|-------------|-------------|---------|-------------------|
| P1.1 | Standalone LDGSTS baseline | YES | YES | YES |
| P1.2 | Equivalent TMA path | YES | YES | YES |
| P1.3 | Joint sweep (≤18 configurations) | YES | YES | YES |
| P1.4 | Profiling, validation, analysis, pilot | YES | YES | YES |

P1.3's remediated implementation passed a new independent GPU-free audit and
was functionally verified on GB300 on 28 July 2026. At Git commit
`59777406b9454f00799c48bff8fa85cb03625cb6`, smoke campaign
`20260728T103315Z` completed both full-binary self-tests and all 18 planned
invocations with `status=COMPLETE` and `publishable=false`. This closes P1.3
without creating an experimental performance result. P1.4 (profiling,
Nsight Compute validation, the pilot benchmark campaign, statistics, and
comparative interpretation) is now implemented: a GPU-free layer
(`scripts/run_exp01_memory_paths_p14.sh`,
`scripts/analyze_exp01_memory_paths_p14.py`, see
`src/memory/P1_4_PROTOCOL.md`) that reuses the audited P1.3 runner unmodified
for the frozen 18-configuration pilot and adds Nsight Compute validation of
six frozen cases, deterministic statistics/bootstrap/saturation analysis,
and report/figure generation. An independent GPU-free audit of that first
implementation found five blockers (a profiling preflight from a different
GPU/driver than the pilot's could be accepted; a validated `metrics_raw.csv`
could be modified after validation and still reach `COMPLETE`/`ANALYZED`; the
NCU raw-CSV parser accepted malformed/wrong-unit/substring-matched evidence;
`--profile` wrote diagnostic logs before safely resolving the campaign tree;
and P1.4 manifest updates delegated to P1.3's overwrite-based writer). All
five were remediated GPU-free, each with a new adversarial test that first
demonstrably failed against the original behavior and then passed against
the fix: preflight provenance is now cross-checked field-by-field before any
Docker/NCU work; a central evidence-integrity gate re-hashes and reparses
every trusted input immediately before both `COMPLETE` and `ANALYZED`; the
NCU CSV parser is now fail-closed on schema/unit/kernel-identity; raw-tree
log writes are symlink-safe and ordered after the safety check; and the P1.4
manifest now publishes as an append-only, hash-chained revision history
(`campaign_dir/manifest/000000.json`, `000001.json`, ...) that never calls
`os.replace()`. A second independent GPU-free audit of that remediated
implementation found four further blockers: (A) a precheck immediately
before a shell redirection into the raw campaign still left a TOCTOU window
between the check and the later `open()`; (B) an unplanned extra
`profiles/<name>/` directory was never compared against anything and so was
silently ignored; (C) a syntactically valid, correctly re-hashed manifest
revision that changed the immutable `campaign_id` (or edited an earlier
`case_result`, or jumped state illegally) previously passed unnoticed; (D)
the evidence-integrity gate never compared its own `dram_read_bytes`
reconstruction (or any `resolved_metric_values` entry) against what was
recorded, so either could be silently tampered. All four were remediated
GPU-free, each with a new adversarial test that first demonstrably failed
and then passed: a descriptor-anchored safe-capture module
(`scripts/p14_safe_capture.py`) replaces every raw-campaign shell redirection
with `O_NOFOLLOW`/`O_EXCL` operations on already-open directory descriptors;
`profiles/`'s actual contents are now compared exactly against the frozen
plan before both `COMPLETE` and `ANALYZED`; every manifest revision is now
semantically validated against an explicit per-field classification, not
just cryptographically chained; and one canonical
`reconstruct_case_result` function, shared by `validate-profile-case` and
the gate, is now compared as a complete structure. See
`src/memory/P1_4_PROTOCOL.md` for the full design of each. A **third**
independent GPU-free audit of that twice-remediated implementation found
five further blockers: (A) the runner built
`profiles/<case>/<case>_report` relative to `/workspace` instead of the
campaign directory, handing that path directly to NCU's own `-o`/
`--log-file`; (B) a corrected path string alone would not fix the
underlying problem, since NCU still opened a raw campaign path (including a
second one for metrics-export `--import`) for writing itself; (C) the
evidence-integrity gate's field comparison used `dict.get()`-based
equality, under which an unexpected `null`-valued field compared as
identical to its absence; (D) manifest fields were classified broadly
(set-once/timestamp) but never bound to the one specific transition legally
allowed to introduce them, so e.g. `resolved_ncu_metrics` could appear
during `PILOT_IN_PROGRESS`; (E) several descriptor/helper defects — an
unvalidated filename could escape the anchored capture directory, a launch
failure orphaned an empty partial file, and `profiles/`'s own inventory and
evidence reads were still lstat-then-listdir/open-by-path rather than
descriptor-anchored. All five were remediated GPU-free, each with a new
adversarial test that first demonstrably failed and then passed: a new
container-side bridge (`scripts/p14_ncu_bridge.py`) runs NCU collection and
metrics export entirely inside the container's own private, non-host-mounted
`/tmp` and hands the host only a versioned, length-delimited bundle over its
own stdout, so NCU never receives a campaign-relative pathname at all; the
evidence-integrity gate now uses a strict recursive structural comparison
(exact key sets, exact types, no `dict.get()` equality); a new
`validate_manifest_state_shape` function binds every manifest field to the
one state that may legally introduce it; and `scripts/p14_safe_capture.py`
now validates every accepted filename as a strict single-component
basename, fixes its failure-cleanup control flow, and — together with the
central evidence-integrity gate and `load_p14_manifest_chain` — extends
descriptor-anchored, no-follow discipline to profile-directory inventory,
per-case evidence reads, and the manifest revision directory itself. See
`src/memory/P1_4_PROTOCOL.md` for the full design of each. A fourth audit
then closed presence-vs-null state shape, timestamp chronology,
count/result co-occurrence, final evidence-gate placement, inode-owned
cleanup, and status/trust-model defects. A fifth audit of `3d92a6b` found
the final three functional blockers: invalid runner failure telemetry, no
exact per-transition mutation matrix, and non-canonical terminal
`profile_order`/`artifact_sha256`. Those blockers are now covered by eight
new full-chain regressions and the final GPU-free acceptance suite passes.
The final post-remediation review also covered the four NCU-2025.4
compatibility fixes made after the fifth audit: the real help layout,
namespace-qualified metric identifiers, the wide raw-page CSV schema, and
the live `ns` duration unit. Fresh preflight `20260730T072946Z` then passed,
and GB300 campaign `20260730T073045Z` at commit
`e2d01b86f53177bd48d18b215be48b422dc3c53b` reached `ANALYZED`: 18/18
pilot configurations, 540/540 retained samples, six/six predefined NCU
profiles, and six `HBM_VALIDATED` classifications with no diagnostic flags.
The final evidence/hash validator reported
`CIERRE TÉCNICO P1.4 / FASE 1: PASS`.

Review of the generated Markdown exposed one presentation-only defect:
string label `ok` was truth-tested and rendered as `REVIEW`. The closing
GPU-free fix now renders the exact label and adds a regression test; it
changes no GPU path, measurement, statistic, or HBM classification. P1.4 is
therefore audited and verified on GB300, and Phase 1 is closed. The campaign
is still a single pilot and remains `publishable: false`; final experimental
campaigns remain Phase 4 work.

## Phase 2 — BF16 UMMA throughput (27 July–2 August 2026)

Entry condition for Phase 2: the Phase 1 gate must pass. Current status:
Phase 1 gate passed; P2.1, P2.2, P2.3, and P2.4 are implemented,
independently audited, and verified on GB300. The Phase 2 gate has passed
and Phase 2 is closed.

| Unit | Description | Implemented | Audited | Verified on GB300 |
|------|-------------|-------------|---------|-------------------|
| P2.1 | 1-SM UMMA | YES | YES | YES |
| P2.2 | 2-SM UMMA | YES | YES | YES |
| P2.3 | Sweep (≤24 configurations) | YES | YES | YES |
| P2.4 | Profiling and empirical ceiling | YES | YES | YES |

P2.1 at Git commit
`1004666db7a2eef1ec499c60740cafc1e2f41328` passed an independent audit
after its source-validation repairs and was functionally verified on a
physical NVIDIA B300 on 30 July 2026. The real `sm_103a` binary contained
all twelve frozen specializations with the exact depth-dependent `UTCHMMA`
and N-dependent `LDTM.x32` counts and the required TMEM lifecycle. The full
GPU self-test reported `SELF_TEST: PASS (12/12)`, and short `smoke` and
`benchmark` routing checks both completed with `correctness=OK`,
`mismatches=0`, the expected commit, and `git_dirty=false`. These checks
close P2.1 as audited, functionally verified infrastructure only: every row
remained `publishable=false`, and no throughput or empirical-ceiling result
is claimed before P2.4.

P2.2 (`src/compute/umma_2sm.cu`, `scripts/check_umma_2sm_sass.py`,
`src/compute/P2_2_PROTOCOL.md`) is implemented: twelve `tcgen05.mma.
cta_group::2.kind::f16` (BF16 x BF16 -> FP32, joint M=256 across one static
two-CTA cluster, 128 local rows per CTA) specializations, N in
{64,128,256} and depth in {4,16,64,256}, with a GPU-free SASS/ELF/source
gate that disassembles the real compiled binary and requires exactly those
twelve symbols, a compile-time-unrolled `UTCHMMA.2CTA` burst of exactly
`depth` instructions per symbol, a real `UTCBAR.2CTA.MULTICAST` completion
sequence, a complete collective (both-CTA, warp-0-only) TMEM
allocate/commit/wait/load/deallocate/relinquish lifecycle with cluster
synchronization before deallocation, ELF-level `EIATTR_EXPLICIT_CLUSTER`/
`EIATTR_CTA_PER_CLUSTER` evidence of the compile-time two-CTA cluster, and
the absence of any non-`.2CTA` (1-SM-fallback-shaped) UTCHMMA/UTCBAR/
allocation, WGMMA, `mma.sync`, TMA, LDGSTS, FP8/FP4, or sparse instruction.
The repaired implementation at Git commit
`637b6a7e2efbe77b1c9c5d3dfc7ece527f522bba` passed an independent audit,
the 101-case fail-closed checker, the pinned CUDA 13.1 `sm_103a` build, and
the full real-cubin SASS/ELF contract for all twelve specializations. It was
then functionally verified on a physical NVIDIA B300 on 31 July 2026: fresh
preflight campaign `20260731T115848Z` reported `OVERALL=PASS`, the device
self-test reported `SELF_TEST: PASS (12/12)`, and the short smoke route
completed with `correctness=OK`, `mismatches=0`, the expected commit, and
`git_dirty=false`. These checks close P2.2 as audited, functionally verified
infrastructure only. No publishable result exists or is claimed; every CSV
row remains `publishable=false`, and no throughput, ceiling, or 1-SM/2-SM
scaling claim is made before P2.3-P2.4. See
`src/compute/P2_2_PROTOCOL.md` for the full audit and validation history.

P2.3 (joint 1-SM/2-SM sweep infrastructure,
`scripts/run_exp02_umma_throughput.sh`,
`scripts/aggregate_exp02_umma_throughput.py`) is implemented: a deterministic
24-invocation runner (12 logical `(N, depth)` pairs x `umma_1sm`/`umma_2sm`,
alternating which method runs first per pair), reusing the already-audited
P2.1/P2.2 binaries and their existing command-line interfaces completely
unmodified, strict validation of every field of every repetition of both
binaries' raw 37-column CSV, symlink-safe centralized campaign
initialization, no-clobber publication of every result/log/evidence file, a
fail-closed manifest state machine, lossless consolidation into
`combined_samples.csv`, and purely descriptive per-configuration statistics
in `summary.csv` (mean/median/sample stdev/coefficient of variation for
`elapsed_cycles`, `cycles_per_umma`, and `flops_per_cycle` -- no TFLOP/s, no
empirical ceiling, no 1-SM/2-SM speedup, no scaling efficiency, no
saturation, no winning configuration, no Nsight Compute). See
`src/compute/P2_3_PROTOCOL.md` for the full frozen contract. P2.3 introduces
no new CUDA kernel and does not modify `src/compute/umma_1sm.cu`,
`src/compute/umma_2sm.cu`, or either SASS checker. The final implementation
at Git commit `7a7cc2ab83197376720f030ba2e990092c3ada40` passed the
independent audit and was functionally verified on a physical NVIDIA B300
on 3 August 2026. Preflight campaign `20260803T141347Z` reported
`OVERALL=PASS`; both complete device self-tests passed; and smoke campaign
`20260803T141410Z` executed and validated the exact 24-invocation plan before
reaching `status=COMPLETE`. All evidence remained `publishable=false`; this
was infrastructure verification, not a performance result.

P2.4 (profiling and empirical ceiling,
`scripts/run_exp02_umma_throughput_p24.sh`,
`scripts/analyze_exp02_umma_throughput_p24.py`,
`scripts/p24_safe_capture.py`, `scripts/p24_ncu_bridge.py`) is implemented: a
reproducible layer around the unmodified, already-audited P2.3 infrastructure
that drives one frozen 24-configuration `run_kind=benchmark` pilot
(iterations=1000, warmup_iterations=10, repetitions=30) through the
unmodified P2.3 runner, profiles the same 24 configurations with Nsight
Compute (an exact kernel-symbol filter, `--launch-skip 1 --launch-count 1`
to profile only the second, timed launch, and clock-control-disabled,
non-defaulting profiler controls), and computes deterministic statistics
(count/mean/median/sample stdev/CV/min/max, a 95% bootstrap CI for the
median, and Tukey-IQR diagnostics) for `elapsed_cycles`, `cycles_per_umma`,
`flops_per_cycle`, and `flops_per_cycle_per_sm` over all 30 retained
repetitions of all 24 configurations, 1-SM/2-SM speedup and scaling
efficiency (never clamped), candidate depth saturation per `(method, N)`
group, and an empirical per-SM BF16 Tensor Core ceiling candidate selected
in clock-independent FLOP/cycle-per-SM space and only then converted with
that same configuration's own matching NCU SM-clock measurement. If the
mandatory `sm__cycles_elapsed.avg.per_second` metric cannot be trusted
(unavailable, ambiguous, malformed, non-finite, non-positive, or an unknown
unit) for any of the 24 profiled configurations, the campaign's terminal
state is `INCONCLUSIVE` rather than `ANALYZED` and no TFLOP/s or completed
empirical-ceiling claim is ever emitted; every other clock-independent
statistic and artifact is still produced. See `src/compute/P2_4_PROTOCOL.md`
for the complete frozen contract. P2.4 introduces no new CUDA kernel and
does not modify `src/compute/umma_1sm.cu`, `src/compute/umma_2sm.cu`, either
SASS checker, or any P2.3 file. The final implementation at Git commit
`65f14d1069f0f04cb591ccdb9262c6222797042e` passed an independent audit and
was verified end-to-end on a physical NVIDIA B300 on 5 August 2026.
Campaign `20260805T102759Z`, with profiling preflight
`20260805T102944Z` reporting `OVERALL=PASS`, completed all 24 frozen
profiles and reached `ANALYZED`; all 24 mandatory SM-clock readings were
`OK`. The selected empirical per-SM ceiling candidate was
`16.37244853848296 TFLOP/s/SM` (`umma_1sm`, `N=256`, `depth=256`). The
corresponding best 2-SM case delivered `16.220558567678513 TFLOP/s/SM` and
99.16% scaling efficiency at `N=256`, `depth=256`. NCU did not resolve the
optional SM-count metric, so no device-wide extrapolation was emitted. The
campaign is one reviewed pilot and every artifact remains
`publishable: false`; it is not a final publishable campaign. These checks
close P2.4 and Phase 2.

## Phase 3 — CuTe DSL GEMM versus cuBLASLt (3–9 August 2026)

Gate: Phase 2 gate passed. P2.1, P2.2, P2.3, and P2.4 are implemented,
independently audited, and verified on GB300. Phase 3 is in progress.

| Unit | Description | Implemented | Audited | Verified on GB300 |
|------|-------------|-------------|---------|-------------------|
| P3.1 | Pinned official CuTe DSL example | YES | YES | YES |
| P3.2 | One-shape wrapper | YES | YES | YES |
| P3.3 | cuBLASLt baseline | NO | NO | NO |
| P3.4 | Three execution variants | NO | NO | NO |
| P3.5 | Five shapes and comparison | NO | NO | NO |

P3.1 (`src/gemm/P3_1_PROTOCOL.md`) is implemented: it executes one pinned,
unmodified, official NVIDIA CuTe DSL example — `NVIDIA/cutlass` v4.6.1,
commit `e05f953a5b3d38adc240df2ff928e0421c2abba3`,
`examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py`,
Git blob `6c6144bc88896cffb3c8c4692ca915f993c71e1d`, SHA-256
`f99bc4cc1e0aea8990e2929d7c703dfc8196d797b7c9f5a889eabcd3c4ff67ec`,
BSD-3-Clause — in place from the pinned `/opt/cutlass` checkout. That
approximately 1,800-line file is never copied, vendored, forked, reformatted,
or patched into this repository, and P3.1 adds no GEMM source of its own. The
frozen functional configuration is BF16 × BF16 → FP32 with FP32 accumulation,
`(M,N,K,L) = (256,256,512,1)`, `a_major=k`, `b_major=k`, `c_major=n`, MMA tiler
`(128,128)`, cluster `(1,1)`, non-persistent, 1-CTA MMA group, TMA loads,
TMA store, `tcgen05`/UMMA with the FP32 accumulator in TMEM, an identity FP32
epilogue, and mandatory reference validation performed by the unchanged
example. Two Make targets were added: GPU-free `gemm-cutedsl-p31-check`
(fails closed unless the checkout HEAD, cleanliness, regular-file identity,
Git blob SHA, SHA-256, CuTe DSL version, PyTorch version, `torch.version.cuda`,
and the example's own GPU-free `--help` and frozen options all match the pinned
contract) and `gemm-cutedsl-p31-smoke`, which validates `BLACKWELL_GPU_INDEX`
before any Docker work, runs exclusively through `scripts/run_container.sh`,
re-checks the upstream commit and SHA-256 inside that same GPU container, runs
exactly the frozen command with reference checking enabled, preserves the
example's exit code, and prints an explicit non-performance notice.
The global `VERSIONS.env` is unchanged and byte-for-byte identical to `main`;
every Phase 3-only pin lives in the new root-level `PHASE3_VERSIONS.env`, which
extends that contract without redefining anything in it: the exact auxiliary
PyTorch pins (`2.10.0+cu130` from the official cu130 index,
`torch.version.cuda == 13.0`), the coherent `cuda-python==13.0.3` /
`cuda-bindings==13.0.3` pins, and the example's path/blob/SHA-256. No existing
pin changed. The small shape is deliberate:
P3.1 is a functional compatibility check, not one of the five final shapes.
P3.1 introduces no wrapper, no persistent variant, no 2-CTA instruction, no
cuBLASLt baseline, no sweep, no autotuning, no Nsight Compute, no campaign
infrastructure, and no result file, and it **produces no experimental result**;
any timing the example computes internally is discarded and explicitly
classified as non-publishable functional-smoke output. The image's Python
dependency graph is coherent rather than merely documented: `torch
2.10.0+cu130` requires `cuda-bindings==13.0.3`, so `cuda-python` and
`cuda-bindings` are both pinned to `13.0.3` (which also satisfies CuTe DSL
4.6.1's own `cuda-python>=12.8` constraint), and `python3 -m pip check` is a
hard, unsuppressed gate during the image build, in `make check-env`, and in
`make gemm-cutedsl-p31-check`; CuTe DSL is still re-verified as `4.6.1` in each
of them. Because the audited P1.3/P2.3 aggregators parse `VERSIONS.env` against
a closed key allowlist, no Phase 3 key was added there and neither aggregator
was modified; `make check-static` runs their real `parse_versions_env()`
against the real `VERSIONS.env` as a regression gate.
The remediated P3.1 implementation at Git commit
`f34cb33a9456ba011feb0a5a35910bbd00f9a9e6` passed an independent audit. It
was then functionally verified on a physical NVIDIA B300 on 6 August 2026:
fresh preflight campaign `20260806T101657Z` reported `OVERALL=PASS` on physical
GPU index `3` (UUID `GPU-90fb226c-3937-2448-1052-2e12282a61b9`), and the frozen
official-example smoke retained reference validation and ended with `PASS`.
P3.1 is therefore closed as `YES / YES / YES`; this functional check creates
no performance result.

P3.2 (`src/gemm/cutedsl_gemm.py`, `scripts/check_cutedsl_gemm_p32.py`,
`src/gemm/P3_2_PROTOCOL.md`) is **implemented, independently audited, and
verified on GB300**. It is a thin, repository-owned orchestration wrapper
around the very same pinned, unmodified upstream example: it loads that file
read-only and in place from `/opt/cutlass` after revalidating the pinned
commit, Git blob SHA, and SHA-256, reuses `DenseGemmKernel`, `can_implement()`,
and the upstream deterministic tensor factory, and deliberately never calls the
upstream `run()`, which fuses compilation, the first launch, correctness, and
benchmarking into a single returned number and therefore cannot provide the
separation P3.2 exists to establish. No NVIDIA source is copied, vendored,
forked, reformatted, or patched, `/opt/cutlass` is never written to, and no key
is added to `VERSIONS.env` or `PHASE3_VERSIONS.env`: P3.2 executes P3.1's file
and so reuses P3.1's pins, reading every provenance value from those two
contracts at run time. The frozen configuration is BF16 × BF16 → FP32 with FP32
accumulation in TMEM at `(M,N,K,L) = (4096,4096,4096,1)` — the first of the five
final shapes — `a_major=k`, `b_major=k`, `c_major=n`, MMA tiler `(128,128)`,
cluster `(1,1)`, one-CTA MMA group, non-persistent, TMA loads, TMA store, seed
`1111`, hot reused operands, `sm_103a`. None of it is reachable from the command
line: the only runtime controls are `--warmup-iterations`, `--iterations`,
`--self-test`, and `--help`, all bounded, and there is no way to skip the
reference check. The wrapper separates `compile_time_ms` (a monotonic host
clock around `cute.compile` alone), `first_launch_ms` (the same clock around the
first launch, whose output is the tensor that gets validated), and
`kernel_time_ms` (CUDA events on the kernel's own stream after warm-up, divided
by the iteration count), validates the complete result against an untimed
PyTorch CUDA oracle (`atol=1e-1`, `rtol=1e-5`) whose FP32 policy is set through
the PyTorch 2.10 `torch.backends.cuda.matmul.fp32_precision` API exclusively —
never the legacy `allow_tf32` flag or `set_float32_matmul_precision()`, which in
2.10 are views of the same setting that must not be mixed — and which must read
back as exactly `ieee`, with the unset `none` default rejected and an absent API
failing closed, and only then runs warm-up
and steady state. A correctness failure exits non-zero with a stderr diagnostic
and emits no CSV row at all, and a row can only be constructed through a
function that refuses any `correctness` value other than `PASS`. Successful
runs write exactly one CSV header and one data row on stdout under a frozen
47-field `schema_version=p32.v1` contract, produced with Python's `csv` module,
with everything else — progress, warnings, and compiler output, including native
writes to descriptor 1 — redirected to stderr. Two Make targets were added:
GPU-free, network-free `gemm-cutedsl-p32-check` (which runs the existing,
unmodified P3.1 gate first, then re-verifies the upstream identity, the pinned
versions and dependency consistency, and runs both GPU-free self-tests plus the
checker inside the pinned image with the repository mounted read-only and no
GPU exposed) and `gemm-cutedsl-p32-smoke`, which validates
`BLACKWELL_GPU_INDEX` in its first recipe step before any Docker work, runs
exclusively through `scripts/run_container.sh`, re-checks the upstream commit
and SHA-256 inside that same GPU container, runs exactly the frozen one-shape
configuration with two warm-ups and ten measured launches, preserves the
wrapper's exit code, and prints an explicit stderr notice. **P3.2 produces no
publishable performance result**: every row carries `publishable=false`, no
TFLOP/s, speedup, efficiency, utilization, or bandwidth is computed anywhere,
no result file or campaign directory is written, and the untimed PyTorch oracle
is a correctness reference only — it is explicitly not the P3.3 cuBLASLt
baseline, which does not exist. The GPU-free checks listed in
`src/gemm/P3_2_PROTOCOL.md` section 10 were run by the author and passed. The
first independent audit of commit
`ea501d4c43b2cf364ac419ddefa3ae84b564581e` found two blockers: mixed
PyTorch FP32-control APIs, and contamination plus incorrect success reporting
in the smoke target. Both were remediated at commit
`c8b3e2ee57e0297940e0fd5864583ec12dfb23e3`; an independent technical
re-audit confirmed both fixes and found no remaining code blocker. Its three
remaining findings were stale documentary statements, corrected by the P3.2
closure update. Fresh preflight `20260806T163806Z` then reported
`OVERALL=PASS` on an NVIDIA B300 SXM6 AC at physical index `7` (UUID
`GPU-40e00845-d89c-1393-2c32-a2dca3ee9442`, compute capability 10.3, driver
610.43.02). The frozen smoke ran that same clean commit, revalidated the pinned
CUTLASS commit and example SHA-256, reported `can_implement: OK`, passed the
complete-result check with zero maximum absolute and relative error, completed
two warm-ups and ten measured launches, and emitted one `p32.v1` row with
`publishable=false`. Its three finite positive timings remain non-publishable
diagnostics, not an experimental result. P3.2 is therefore closed as
`YES / YES / YES`; P3.3–P3.5 remain unimplemented.

## Phase 4 — Campaigns and integration (10–15 August 2026)

Gate: Phase 3 gate passed.

| Unit | Description | Implemented | Audited | Verified on GB300 |
|------|-------------|-------------|---------|-------------------|
| P4.1 | Orchestrator | NO | NO | NO |
| P4.2 | Pilot plus three final campaigns | NO | NO | NO |
| P4.3 | Integrated analysis, documentation, audit | NO | NO | NO |
