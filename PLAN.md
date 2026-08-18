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
independently audited, and verified on GB300. Phase 3 is closed: P3.1–P3.5 are
implemented, independently audited, and verified on GB300.

| Unit | Description | Implemented | Audited | Verified on GB300 |
|------|-------------|-------------|---------|-------------------|
| P3.1 | Pinned official CuTe DSL example | YES | YES | YES |
| P3.2 | One-shape wrapper | YES | YES | YES |
| P3.3 | cuBLASLt baseline | YES | YES | YES |
| P3.4 | Three execution variants | YES | YES | YES |
| P3.5 | Five shapes and comparison | YES | YES | YES |

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
baseline, which is a separate unit (`src/gemm/P3_3_PROTOCOL.md`). No
P3.2-owned comparison against P3.3 is performed; the descriptive comparison
belongs to P3.5, which is now closed but remains non-publishable functional
evidence. The
GPU-free checks listed in
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
`YES / YES / YES`.

P3.3 (`src/gemm/cublaslt_gemm.py`, `src/gemm/cublaslt_bridge.cu`,
`scripts/check_cublaslt_gemm_p33.py`, `src/gemm/P3_3_PROTOCOL.md`) is
**implemented, independently audited, and verified on GB300**. It is the
vendor-library counterpart of P3.2: the same frozen BF16 geometry,
`(M,N,K,L) = (4096,4096,4096,1)` computing `C = A × Bᵀ` with FP32 accumulation,
on the same operand bytes, issued through a direct, explicit `cublasLtMatmul`
call. cuBLASLt already ships inside the pinned CUDA 13.1 development image, so
P3.3 adds no package, does not change `Dockerfile`, and adds no key to
`VERSIONS.env` or `PHASE3_VERSIONS.env`; the library's runtime version is read
with `cublasLtGetVersion()` and recorded rather than pinned. No NVIDIA GEMM
implementation is copied, forked, patched, or vendored: the small C-ABI bridge
calls only the public cuBLASLt API of the pinned headers, defines no CUDA
kernel of its own, prints nothing, and lets no C++ exception cross the C
boundary. The explicit descriptor contract is A row-major `M × K` with
`lda = K`, B row-major `N × K` with `ldb = K`, C and D row-major `M × N` with
`ldc = ldd = N`, `transa = CUBLAS_OP_N`, `transb = CUBLAS_OP_T`, `CUDA_R_16BF`
inputs, `CUDA_R_32F` output, `CUBLAS_COMPUTE_32F`, `CUDA_R_32F` scale,
`CUBLASLT_POINTER_MODE_HOST`, `CUBLASLT_EPILOGUE_DEFAULT`, `alpha = 1`,
`beta = 0`, no bias, seed `1111`, hot reused operands — and nothing is silently
transposed, relaid out, retyped, or diverted to another GEMM interface.
Operand equivalence with P3.2 is structural rather than asserted: P3.3
replicates the pinned upstream `create_tensors()` sequence call for call — the
same `cutlass.torch.matrix` factory, the same seed applied once, the same A/B/C
call order that fixes the RNG stream, the same dtypes and strides — because the
upstream factory discards the device tensors for A and B while P3.3 must retain
every allocation to hand its pointer to cuBLASLt, and a parser over the
verified upstream file fails the run if that factory ever diverges from what
P3.3 replicates. The algorithm policy is fixed and never autotuned: a
`67,108,864`-byte (64 MiB) workspace limit, exactly 32 requested heuristic
results, `CUBLASLT_SEARCH_BEST_FIT`, the first entry whose state is
`CUBLAS_STATUS_SUCCESS`, re-validated with `cublasLtMatmulAlgoCheck()`,
rejected if it needs more workspace than the limit, given exactly the workspace
it requires (a null pointer only when that is zero), and executed alone; no
candidate is ever benchmarked, and the bridge contains exactly one
`cublasLtMatmul` call site and no timing facility at all. The selected
algorithm's ID, tile, stages, split-K, reduction scheme, CTA swizzling, custom
option, inner shape, cluster shape, waves count, required workspace, heuristic
counts and index, and the four real pointer alignments are all recorded. The
wrapper separates `setup_time_ms` (a monotonic host clock around plan creation
alone), `first_launch_ms` (the same clock around one `cublasLtMatmul`, whose
output is the tensor that gets validated), and `kernel_time_ms` (CUDA events on
the same stream after warm-up, divided by the iteration count); `setup_time_ms`
is deliberately not `compile_time_ms`, because nothing is compiled at run time
and the P3.2 field name is never reused. Correctness uses the identical untimed
PyTorch CUDA oracle P3.2 uses, with the same PyTorch 2.10
`fp32_precision`-only policy that must read back as exactly `ieee`, the same
`atol = 1e-1` / `rtol = 1e-5`, and the same complete-result criterion; a
failure exits non-zero, emits no CSV at all, and runs neither warm-up nor
steady state. Successful runs write exactly one CSV header and one data row on
stdout under a new frozen 77-field `schema_version=p33.v1` contract — the P3.2
`p32.v1` schema is neither modified nor reinterpreted — with everything else on
stderr. Two Make targets were added: GPU-free, network-free
`gemm-cublaslt-p33-check` (which runs the existing, unmodified P3.2 gate first,
then compiles the bridge for `sm_103a` into container-private `/tmp` and
inspects its ELF symbols to prove the measured path references
`cublasLtMatmul` and references no fallback GEMM entry point) and
`gemm-cublaslt-p33-smoke`, which validates `BLACKWELL_GPU_INDEX` in its first
recipe step before any Docker work, runs exclusively through
`scripts/run_container.sh`, and executes the frozen configuration with two
warm-ups and ten measured launches. **P3.3 produces no publishable performance
result**: every row carries `publishable=false`, no TFLOP/s, speedup,
efficiency, utilization, bandwidth, or winner label is computed anywhere, no
result file or campaign directory is written, and **P3.3 itself performs no
CuTe-versus-cuBLASLt comparison** — that comparison is P3.5's, which is now
closed and remains non-publishable functional evidence. The GPU-free checks listed in
`src/gemm/P3_3_PROTOCOL.md` section 13 were run by the
author and passed. An independent audit of implementation commit
`bb66e3275d2f5bf1addbd14c84596b1edede977f` found two blockers: valid
`split_k=0` metadata was rejected and read with the wrong signed width, and an
obsolete P3.2 status assertion made `make check-static` fail. Remediation
commit `1c3ade8a39ae1e19882514e2b06094a418eb70bf` corrected both findings,
added adversarial regression coverage, and removed the associated stale P3.2
documentation. The remediated tree passed `make check-static`; the audit
findings were then rechecked with no remaining blocker, and the operator
confirmed the Docker-backed `make gemm-cublaslt-p33-check` gate also passed on
that clean commit.

Fresh preflight campaign `20260807T144123Z` reported `OVERALL=PASS` on an
NVIDIA B300 SXM6 AC at physical index `7` (UUID
`GPU-40e00845-d89c-1393-2c32-a2dca3ee9442`, compute capability 10.3, driver
610.43.02). An initial smoke invocation with an unset operator variable exited
with status 2 before exposing or using a GPU, as required. The valid rerun with
`BLACKWELL_GPU_INDEX=7` executed the same clean remediation commit through
direct `cublasLtMatmul`, revalidated the pinned upstream identity, returned
eight supported heuristic entries from 32 requested, selected index 0
(`algo_id=66`, `tile_id=23`, `stages_id=35`, `split_k=1`, zero workspace),
passed the complete-result check with zero maximum absolute and relative error,
completed two warm-ups and ten measured launches, and emitted exactly one
77-field `p33.v1` row with `git_dirty=false` and `publishable=false`. Its three
finite positive timings remain non-publishable diagnostics, not an experimental
result. P3.3 is therefore closed as `YES / YES / YES`.

P3.4 (`src/gemm/cutedsl_variants.py`, `scripts/check_cutedsl_variants_p34.py`,
`src/gemm/P3_4_PROTOCOL.md`) is **implemented, independently audited, and
verified on GB300**. It adds the two remaining frozen CuTe DSL execution
variants alongside the one P3.2 established, so that all three exist under one
identical operand set, one identical correctness oracle, and one identical
timing discipline, at the same single shape `(M,N,K,L) = (4096,4096,4096,1)`:
`nonpersistent_1cta` (`DenseGemmKernel`, non-persistent, MMA tiler `(128,128)`,
cluster `(1,1)`, `use_2cta_instrs=false`), `persistent_1cta`
(`PersistentDenseGemmKernel`, static persistent, tiler `(128,128)`, cluster
`(1,1)`, `use_2cta_instrs=false`), and `persistent_2cta`
(`PersistentDenseGemmKernel`, static persistent, tiler `(256,128)`, cluster
`(2,1)`, `use_2cta_instrs=true`). The 2-CTA row deliberately uses an M tile of
256 so each of the two participating CTAs keeps a local M extent of 128 —
P2.2's two-SM geometry, and the shape NVIDIA's own persistent example documents
for `use_2cta_instrs=True`; the checker rejects any table whose 2-CTA row does
not satisfy `tiler_M / cluster_M == 128`. Exactly three fixed candidates run,
always in that order, with no autotuning, no candidate search, no fourth
candidate, and no additional shape. This repository still owns no GEMM kernel:
the non-persistent variant keeps using P3.1's pinned example and the two
persistent variants use the official static-persistent example
`dense_gemm_persistent.py` from the **same** already pinned CUTLASS commit
(BSD-3-Clause, Git blob `10d62d239457748372a522488ee23bc3df5f346d`, SHA-256
`d59344faf902cb215a2cee3f2ae6415a14589c6ad8f93e5e74e2612c1e6a0810`), both
loaded read-only and in place from `/opt/cutlass` under private module names
after HEAD, checkout cleanliness, regular-file identity, Git blob, and SHA-256
are verified for each file — and neither upstream `run()` nor either upstream
benchmarking helper is ever called, because they fuse compilation, first
launch, correctness, and benchmarking into one number. The only new pins are
the three `CUTEDSL_P34_*` keys in `PHASE3_VERSIONS.env`; `VERSIONS.env`, the
`Dockerfile`, and every closed P3.1/P3.2/P3.3 interface are untouched, and the
checker re-runs both closed P1/P2 aggregator parsers to prove `VERSIONS.env`
still satisfies their allowlist. All three variants consume byte-identical A
and B: the operands are built once by the pinned non-persistent example's own
`create_tensors()` — same factory, same seed `1111`, same A/B/C order, same
dtypes and strides as P3.2 and P3.3 — and are never mutated; the persistent
example's independent tensor path is deliberately unused. Only C is reset
between candidates, to NaN, outside every timer, so an element a kernel fails to
write stays non-finite and is rejected instead of surviving as a stale value.
`max_active_clusters` comes from the official pinned hardware helper
(`cutlass.utils.HardwareInfo().get_max_active_clusters`) for each variant's own
cluster size, is required to be a finite positive integer, and is recorded; the
non-persistent row records the canonical `not_applicable`. Per variant the
wrapper separates `compile_time_ms` (a monotonic host clock around
`cute.compile` alone), `first_launch_ms` (the same clock around the first
launch, whose output is validated), and `kernel_time_ms` (CUDA events on the
kernel's own stream after warm-up, divided by the iteration count), validates
the complete result against the identical untimed PyTorch CUDA oracle P3.2 and
P3.3 use (`fp32_precision` only, read back as exactly `ieee`, `atol=1e-1`,
`rtol=1e-5`), and only then runs that variant's warm-up and steady state. The
whole output is buffered: successful runs write exactly one CSV header and
exactly three rows — four lines — under a new frozen 51-field
`schema_version=p34.v1` contract, and a failure in **any** of the three
positions emits no CSV at all, including no rows from variants that already
passed. Two Make targets were added: GPU-free, network-free
`gemm-cutedsl-p34-check` (which runs the existing, unmodified P3.3 gate first,
then revalidates the checkout and both official sources, asserts the persistent
file really carries `StaticPersistentTileScheduler` and `CtaGroup.TWO`, checks
the pinned package versions and `pip check`, and runs both GPU-free self-tests
plus the full contract check) and `gemm-cutedsl-p34-smoke`, which rejects a
missing or non-numeric `BLACKWELL_GPU_INDEX` in its first recipe action before
any Docker work, runs exclusively through `scripts/run_container.sh`,
revalidates both sources inside that same GPU container, and runs all three
variants with two warm-ups and ten measured launches each. **P3.4 produces no
publishable performance result**: every row carries `publishable=false`, no
TFLOP/s, speedup, efficiency, utilization, bandwidth, ranking, or winner is
computed anywhere, no result file or campaign directory is written, and
**P3.4 itself performs no variant-versus-variant or CuTe-versus-cuBLASLt
comparison** — that comparison is P3.5's. An independent technical audit of
implementation commit
`bb8cdc5b` found no blocking defect and approved the unit for GB300 execution.
On 7 August 2026, the operator ran a fresh passing preflight and the frozen
`gemm-cutedsl-p34-smoke` on an explicitly selected idle physical NVIDIA B300
at index 4. Both pinned upstream sources were revalidated; all three variants
passed `can_implement()`, compiled, launched, completed two warm-ups and ten
measured launches, and passed complete-result correctness with zero maximum
absolute and relative error. The official occupancy helper returned 148 active
clusters for `persistent_1cta` and 74 for `persistent_2cta`. The smoke emitted
exactly the required four CSV lines in frozen order, with every row marked
`correctness=PASS` and `publishable=false`; its timings remain functional,
non-publishable diagnostics. P3.4 is therefore closed as `YES / YES / YES`.

P3.5 (`src/gemm/gemm_comparison.py`, `src/gemm/cublaslt_bridge_p35.cu`,
`scripts/check_gemm_comparison_p35.py`, `src/gemm/P3_5_PROTOCOL.md`) is
**implemented, independently audited, and verified on GB300**. It extends the
already verified P3.3/P3.4 infrastructure to all five
final Experiment 3 shapes — `(4096,4096,4096,1)`, `(8192,8192,8192,1)`,
`(16384,512,4096,1)`, `(32768,512,4096,1)`, `(512,16384,4096,1)`, in that frozen
order — and performs the first explicit, purely descriptive comparison among
four candidates per shape, always in the frozen order `nonpersistent_1cta`,
`persistent_1cta`, `persistent_2cta`, and cuBLASLt `heuristic_first_supported`.
Output is shape-major: exactly 5 × 4 = 20 rows and 21 lines. No arbitrary shape
is reachable from the command line, the environment, a configuration file, or an
input CSV: the Python wrapper and the C bridge freeze the same five geometries
independently, the bridge exposes its own allowlist through `p35_shape_count()` /
`p35_shape_at()`, and the wrapper reads it back and requires the two to be
identical before any measurement runs; a geometry outside the allowlist never
reaches a descriptor, a heuristic query, or a launch. The three CuTe DSL rows are
byte-for-byte the closed P3.4 table (including the `(256,128)` tiler over a
`(2,1)` cluster that keeps the per-CTA M extent at 128), and the cuBLASLt policy
is exactly the closed P3.3 policy — 64 MiB workspace limit, 32 requested
heuristic results, `CUBLASLT_SEARCH_BEST_FIT`, the first supported entry,
re-validated with `cublasLtMatmulAlgoCheck()`, no fallback GEMM API, and no
autotuning by execution — so a different supported algorithm may naturally win
per shape while the *selection policy* never changes. This repository still owns
no GEMM kernel and P3.5 adds **no pin**: it reuses the same two already pinned
official NVIDIA examples P3.4 uses and the cuBLASLt library that already ships in
the pinned CUDA 13.1 image, so `VERSIONS.env`, `PHASE3_VERSIONS.env`, the
`Dockerfile`, and `scripts/run_container.sh` are untouched. Per shape the
operands are built once by the pinned non-persistent example's own
`create_tensors()` (same factory, seed `1111`, A/B/C order, dtypes and strides as
P3.2–P3.4), never mutated, and the untimed IEEE-FP32 CUDA oracle is computed once
and reused by all four candidates; the cuBLASLt candidate receives its own device
copies of A and B made from those same immutable host tensors, each proved
byte-identical before anything runs and each layout-checked against the frozen
descriptor contract. The output buffer is reset to NaN outside every timer before
every candidate, and every candidate's complete result is validated before any
warm-up or steady-state timing runs for it. Per candidate the wrapper separates
`compile_time_ms` (CuTe DSL JIT only) *or* `setup_time_ms` (cuBLASLt plan
creation only) — never both, each carrying the canonical `not_applicable` on the
other method's rows, and never compared against each other — plus
`first_launch_ms` and, from CUDA events on that candidate's own stream,
`kernel_time_ms`. Only `kernel_time_ms` participates in the comparison:
`flop_count = 2·M·N·K` exactly, `tflops = flop_count / (kernel_time_ms × 1e9)`,
`throughput_ratio_vs_cublaslt = candidate_tflops / cublaslt_tflops`, and
`gap_to_cublaslt_pct = 100 × (1 − ratio)` — positive means slower, zero equal,
and **negative means faster and is never clamped**. Candidates are ranked by
full-precision `kernel_time_ms` with an exact tie broken by the frozen candidate
order, `best_cutedsl_variant` is chosen among the three CuTe DSL candidates only
and repeated on all four rows of the shape, and exactly one CuTe DSL row carries
`is_best_cutedsl=true`. No confidence interval, p-value, outlier removal,
roofline efficiency, empirical-ceiling utilization, bandwidth,
arithmetic-intensity classification, or causal interpretation is computed
anywhere. The whole output is buffered under a new frozen 100-field
`schema_version=p35.v1` contract — the closed `p32.v1`, `p33.v1`, and `p34.v1`
schemas are neither modified nor reinterpreted — and a failure at **any** shape
or candidate emits no CSV at all, including rows already completed. Two Make
targets were added: GPU-free, network-free `gemm-comparison-p35-check` (which
runs the existing, unmodified P3.4 gate first, then revalidates the checkout and
both official sources, checks the pinned versions and `pip check`, compiles the
P3.5 bridge into container-private `/tmp` and inspects its ELF symbols and
dynamic dependencies to prove it references `cublasLtMatmul` and no fallback GEMM
API, and runs both GPU-free self-tests plus the full contract check) and
`gemm-comparison-p35-smoke`, which rejects a missing or non-numeric
`BLACKWELL_GPU_INDEX` in its first recipe action before any Docker work and then
runs exclusively through `scripts/run_container.sh`. **P3.5 produces no
publishable performance result**: every row carries `publishable=false`, the
comparison fields are arithmetic rather than a conclusion, no result file or
campaign directory is written, and beating cuBLASLt is not a success criterion.
Implementing P3.5 also required correcting three stale frontier guards that the
P3.4 closure had already superseded — the `Makefile` and
`scripts/check_cublaslt_gemm_p33.py` still demanded that P3.4 be unclosed, and
the `Makefile`, `scripts/check_cublaslt_gemm_p33.py`, and
`scripts/check_cutedsl_variants_p34.py` all required the literal row
`P3.5 | Five shapes and comparison | NO | NO | NO`, which structurally forbade
P3.5 from existing; at the P3.5 baseline commit `b50dca3` both `make
check-static` and `python3 scripts/check_cublaslt_gemm_p33.py .` therefore
already failed. All three guards were advanced to the truthful state and none was
weakened; every closed unit's CLI, schema, field order, Make targets, one-shape
restriction, output behaviour, correctness and provenance checks, and smoke
semantics are unchanged. See `src/gemm/P3_5_PROTOCOL.md` for the full frozen
contract. The GPU-free checks listed in that protocol's section 12 were run by
the author and passed; **those are the author's own self-checks, not an
independent audit, and GPU-free checks are not GB300 verification**. The first
independent technical audit of implementation commit `61d17845` found two
fail-closed cleanup routes that could warn and continue. Remediation commit
`b76c774473b85a498d8e8872296594cae472d498` propagated cleanup failures,
checked every native destructor status, preserved existing primary exceptions,
and added adversarial regressions; post-remediation review found no remaining
blocker. The Docker-backed, network-free `make gemm-comparison-p35-check` gate
then passed on the corrected tree, including bridge compilation and ELF proof of
`cublasLtMatmul` with no fallback GEMM API.

On 8 August 2026, a fresh preflight passed on an explicitly selected idle
physical NVIDIA B300 at index 4 (UUID
`GPU-4ae7e013-1aac-31d8-8b8e-c27530f1c6ed`, driver `610.43.02`). The frozen
smoke ran the clean corrected commit, revalidated both pinned upstream sources,
compiled the bridge into private `/tmp`, and completed all five shapes × four
candidates with two warm-ups and ten measured launches each. All 20
complete-result checks passed with zero maximum absolute and relative error.
The output contained the exact 100-field header and 20 rows in frozen order,
with `git_dirty=false`, `correctness=PASS`, and `publishable=false` throughout.
The comparison quantities remain arithmetic functional-smoke diagnostics, not
an experimental conclusion; no final campaign, statistical treatment, Nsight
Compute analysis, or Phase 4 interpretation was performed. P3.5 and Phase 3 are
therefore closed as `YES / YES / YES`.

## Phase 4 — Campaigns and integration (10–15 August 2026)

Gate: Phase 3 gate passed. P3.1–P3.5 are closed as `YES / YES / YES`. P4.1 is
implemented, independently audited, and verified on GB300. P4.2 is also
implemented, independently audited, and verified on GB300: accepted pilot
`20260812T013848Z` plus final campaigns `20260817T110330Z`,
`20260817T111310Z`, and `20260817T112011Z` passed terminal revalidation and the
read-only cross-campaign validator. All evidence remains non-publishable and
P4.3 remains unimplemented.

| Unit | Description | Implemented | Audited | Verified on GB300 |
|------|-------------|-------------|---------|-------------------|
| P4.1 | Orchestrator | YES | YES | YES |
| P4.2 | Pilot plus three final campaigns | YES | YES | YES |
| P4.3 | Integrated analysis, documentation, audit | NO | NO | NO |

P4.1 (`scripts/run_all.sh`, `scripts/phase4_orchestrator.py`,
`scripts/check_phase4_orchestrator_p41.py`, `src/phase4/P4_1_PROTOCOL.md`) is
**implemented, independently audited, and verified on GB300**.
`scripts/run_all.sh` is the single public Phase 4 orchestration entry point; it
coordinates one reproducible top-level
campaign across the three closed experiments by **composing** their existing
audited entry points — `make memory-paths-p14-pilot` / `-profile` / `-analyze`
for experiment 1, `make compute-umma-p24-pilot` / `-profile` / `-analyze` for
experiment 2, and `make gemm-comparison-p35-smoke` for experiment 3 — passing
the same top-level campaign ID through `P1_4_CAMPAIGN_ID`/`P2_4_CAMPAIGN_ID`
and the exact preflight summary that invocation created and validated through
`P1_4_PREFLIGHT_SUMMARY`/`P2_4_PREFLIGHT_SUMMARY`. It reimplements no kernel,
scientific matrix, statistic, profiler plan, correctness rule, schema, or
execution parameter, adds no external dependency and no version pin, and adds
no Nsight Compute case or profiler route. Experiment 3 remains **one atomic
`gemm` stage** — five frozen shapes × four candidates, exactly 20 rows — because
P3.5 enforces shared operands, shape-major ordering, all-or-nothing output, and
one common comparison contract; a separate `--only cublaslt` or `--only
cutedsl` mode is forbidden and rejected by name, and the supported selector is
`--only gemm`. The deterministic full plan is `preflight`, `memory.pilot`,
`memory.profile`, `umma.pilot`, `umma.profile`, `gemm.capture`,
`memory.analyze`, `umma.analyze`, `campaign.validate`, with the three `--only`
plans given in `src/phase4/P4_1_PROTOCOL.md` section 3. Every real invocation
with pending GPU work requires an explicit numeric `BLACKWELL_GPU_INDEX`,
rejected before Docker, `nvidia-smi`, result creation, or any GPU-related
subprocess; the preflight runs only through `BLACKWELL_GPU_INDEX=<i> make
preflight`, and the exact `results/preflight/<TS>/summary.json` it produced is
identified from that invocation's own stdout markers — never `ls -t`, a
"latest" symlink, glob ordering, or a modification time — then validated as a
non-symlink, non-empty regular JSON file with `overall_status=PASS`, the
current clean commit, the explicitly selected GPU, and all six required checks
at `PASS`. The top-level campaign tree
(`results/raw/phase4/<campaign_id>/manifest/`, `plan.json`, `logs/`,
`exp03/gemm_comparison.csv`) carries an append-only, hash-chained manifest
history rather than one mutable `manifest.json`; it records only allowlisted
information (schema version, campaign ID and immutable kind, immutable scope,
the clean Git commit, stage order and per-stage status, repository-relative
evidence paths, SHA-256 hashes, allowlisted GPU identity, the validated
preflight reference, timestamps, any failure or interruption stage, and
`publishable=false`), never usernames, home paths, host names, environment
dumps, credentials, SSH material, process information, host command lines, or
dynamic power/clock/temperature/utilization telemetry. The experiment-owned raw
trees are referenced by validated repository-relative path and hash rather than
copied. Symlinks and unexpected file types are rejected everywhere, evidence is
never overwritten, publication is no-clobber, valid partial evidence survives an
interruption or a child failure, and everything is re-hashed and revalidated
immediately before a stage or campaign is declared complete. `--resume` is
evidence-driven rather than existence-driven: each apparently complete stage is
re-loaded through the underlying unit's **own** semantic loader
(`load_p14_manifest_chain` / `load_p24_manifest_chain`) and its own
`verify_campaign_evidence_integrity()`, or P3.5's own
`validate_serialized_output`, re-hashed, and rechecked against the immutable
campaign identity before it is skipped, and a fresh preflight is created and
validated whenever GPU work is still pending. A unit accepted in a terminal
state stays pinned to the exact manifest revision it was accepted at (path,
revision, SHA-256, and a digest covering its own evidence-integrity snapshot and
fresh hashes of every canonical terminal `analysis/` artifact), so a later
terminal revision, a changed revision, or changed raw or derived evidence is
rejected rather than adopted silently; every component must also agree with the
current preflight and with each other on Git commit, clean-tree status, GPU UUID
and name, compute capability, and — where the closed schemas expose them — the
CUDA driver and runtime API versions, compared after normalizing the two textual
formats those schemas use. An interruption, including one inside
`campaign.validate`, preserves every artifact and log already created, exits
130, and leaves that stage eligible for a new attempt under the next free
attempt number; an attempt is recorded only once both named logs exist. Every
experimental Make target is invoked with
`--silent --no-print-directory` so no echoed recipe line can put an absolute
bind-mount source into a durable log, and the exact checkout root emitted by a
child is replaced with `<repo-root>` only at the textual-log boundary. A P2.4
`INCONCLUSIVE` analysis
propagates to a non-complete top-level outcome and is never accepted as a
complete campaign. P4.1 creates no Phase 4 variability threshold, publication
threshold, or scientific acceptance rule; those belong to P4.2/P4.3. Two
GPU-free Make targets were added, `phase4-p41-plan` and `phase4-p41-check`, the
latter depending only on container-free gates (`memory-paths-p14-check` and
`compute-umma-p24-check-gpu-free`, the GPU-free half split out of
`compute-umma-p24-check`, which still depends on it and is otherwise unchanged),
so the P4.1 gate runs with no container runtime at all. Implementing P4.1 also required advancing one
stale frontier assertion that the P4.1 row itself owned — the `Makefile` and
`scripts/check_gemm_comparison_p35.py` both demanded the literal row
`P4.1 | Orchestrator | NO | NO | NO`, which structurally forbade P4.1 from
being implemented at all; both were advanced to the then-truthful
`YES | NO | NO`. This closure advances the P4.1-owned assertions in the
`Makefile`, the P3.5 checker, and the P4.1 checker to `YES | YES | YES` while
continuing to reject every stale or impossible partial state. P4.2/P4.3 are
still required to remain unimplemented. The GPU-free checks in
`src/phase4/P4_1_PROTOCOL.md` section 12 were run by the author and passed;
**those checks were not treated as an independent audit or as GB300
verification.**

The independent review covered implementation commit
`129c20c1eeb11b11076e765fa1e59a73831d6f2d` and its two remediation rounds,
ending at commit `77908dae377fb131ff90baef455c79d6d2c28b0b`. The final re-audit
accepted that corrected commit with no remaining blocker. On 12 August 2026,
the operator then ran the full nine-stage pilot campaign `20260812T013848Z` on
an explicitly selected idle physical NVIDIA B300 at index 7, with driver
`610.43.02`, from that same clean commit. All stages reached `COMPLETE`; the
process exited 0. A subsequent `--resume` revalidated and skipped all nine
stages, reported the campaign terminally `COMPLETE`, exited 0, and confirmed
that no artifact was rewritten and no manifest revision was appended. This is
the P4.1 GB300 verification and also satisfies the single P4.2 pilot. Every
artifact remained `publishable=false`; at P4.1 closure no final campaign,
cross-campaign statistic, integrated interpretation, final table, or figure
had been produced. P4.1 is therefore closed as `YES / YES / YES`. P4.2 has
since completed the three final campaigns and closed independently; P4.3
remains unimplemented.

P4.2 (`src/phase4/P4_2_PROTOCOL.md`,
`scripts/check_phase4_campaigns_p42.py`) is **implemented, independently
audited, and verified on GB300**. It is
the smallest layer that freezes the final-campaign policy *before* any result
is collected and then verifies the invariants P4.1 structurally cannot see,
because P4.1 validates exactly one campaign at a time. The frozen population is
one accepted pilot plus three final campaigns: pilot `20260812T013848Z` remains
accepted and `publishable=false`, is a functional qualification of the
orchestration path rather than a measurement, and is never one of the three
replicates or part of any P4.3 aggregation. Each final campaign used
`--campaign-kind final` with `scope=full` — the `--only` selectors were
forbidden for P4.2 finals — and ran all nine stages. All three share
one clean Git commit, one immutable plan and stage order, one physical GPU
UUID, name, compute capability and driver version, the same campaign scope, the
same audited execution path, and the same provenance fields the accepted P4.1
schema already exposes; P4.2 invents no provenance field and changes no part of
the P4.1 schema. They run sequentially, never concurrently, and no code,
documentation, configuration, version, or tracked-file change is allowed
between the start of final campaign 1 and the terminal revalidation of final
campaign 3. The preferred device is the pilot's own GPU index 7 and its
validated UUID; if it is unavailable the sequence stops for an explicit
protocol decision rather than silently switching devices. An interrupted or
resumable campaign keeps its own ID and uses the existing P4.1 `--resume` path;
a terminally `INCONCLUSIVE`, integrity-violating, provenance-incompatible, or
unresumable campaign stops the sequence, preserves all evidence, and produces
**no** fourth or replacement final campaign — it leaves P4.2 incomplete and
requires a documented protocol amendment plus a new audit before further GPU
work. No failed, interrupted, inconclusive, or inconvenient campaign may be
silently omitted, and the checker enforces that structurally by enumerating the
whole campaign root and failing on any undeclared `campaign_kind=final`
campaign or any entry it cannot interpret. `scripts/check_phase4_campaigns_p42.py`
reuses P4.1's audited manifest-chain loader, descriptor-anchored opens, and
hashing primitives rather than reimplementing them, adds no dependency and no
version pin, and has three surfaces: a synthetic `--self-test` over temporary
fixtures, a repository-contract mode that needs no `results/raw/`, and a
strictly read-only evidence mode for cluster use that never writes,
repairs, resumes, deletes, or regenerates evidence and never invokes
`--resume`. Because the pilot is outside the final statistical population, the
three finals may run from a later P4.2 commit; to bound that, the checker pins
the SHA-256 of `scripts/run_all.sh` and `scripts/phase4_orchestrator.py` to
their content at the accepted P4.1 closure commit, so any change to the
execution path fails closed and must be re-audited. One fast GPU-free Make
target was added, `phase4-p42-check`, with no prerequisites and no ability to
start or resume a campaign; `scripts/run_all.sh` remains the only public Phase 4
execution entry point. Implementing P4.2 required advancing one stale frontier
assertion that the P4.2 row itself owned — the `Makefile`,
`scripts/check_gemm_comparison_p35.py`, and
`scripts/check_phase4_orchestrator_p41.py` all demanded the literal row
`P4.2 | Pilot plus three final campaigns | NO | NO | NO`, which structurally
forbade P4.2 from being implemented at all. The implementation row first
advanced to the then-truthful `YES | NO | NO`. This documentary closure
advances the same P4.2-owned guards to `YES | YES | YES`; every closed P1–P4.1
assertion is preserved, P4.3 must still be recorded unimplemented, and every
stale or impossible P4.2 partial state is rejected. The implementation-time
GPU-free checks in `src/phase4/P4_2_PROTOCOL.md` section 11.1 remain author
self-checks; the later independent audit and GB300 evidence are recorded
separately.

The final independent review accepted frozen execution commit
`b08e45c2636a3ac17c94ad8b1368084914196d7a` with no remaining blocker. On
17 August 2026, final campaigns `20260817T110330Z`, `20260817T111310Z`, and
`20260817T112011Z` ran sequentially from that same clean commit, completed the
full nine-stage plan, and then passed terminal `--resume` revalidation without
rewriting an artifact or appending a manifest revision. Evidence mode accepted
the real four-campaign population, confirmed the shared final commit, plan,
stage order, GPU identity, and comparable provenance, excluded the pilot from
the replicate set, and found no undeclared final campaign. P4.2 is therefore
closed as `YES / YES / YES`.

P4.2 computes no statistic, winner, threshold, roofline interpretation,
architectural conclusion, table, or figure; those belong exclusively to P4.3,
which remains unimplemented. Every campaign remains `publishable=false`, and
no publishable Phase 4 result exists. The next work is P4.3 integrated analysis,
documentation, and audit.
