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
Phase 1 gate passed; Phase 2 may begin. P2.1 and P2.2 are closed. P2.3
(joint sweep infrastructure) is implemented; independent audit and GB300
verification are pending. P2.4 remains unimplemented.

| Unit | Description | Implemented | Audited | Verified on GB300 |
|------|-------------|-------------|---------|-------------------|
| P2.1 | 1-SM UMMA | YES | YES | YES |
| P2.2 | 2-SM UMMA | YES | YES | YES |
| P2.3 | Sweep (≤24 configurations) | YES | NO | NO |
| P2.4 | Profiling and empirical ceiling | NO | NO | NO |

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
`src/compute/umma_2sm.cu`, or either SASS checker. Independent audit and
GB300 verification are both pending. P2.4 (profiling and empirical ceiling)
remains entirely unimplemented.

## Phase 3 — CuTe DSL GEMM versus cuBLASLt (3–9 August 2026)

Gate: Phase 2 gate remains pending. P2.1 and P2.2 are audited and verified on
GB300, but P2.3 still needs an independent audit and GB300 verification, and
P2.4 remains unimplemented. Phase 3 cannot begin until all four P2 units are
implemented, audited, and verified.

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
