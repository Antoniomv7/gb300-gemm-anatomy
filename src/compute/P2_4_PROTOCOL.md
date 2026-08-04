# P2.4 frozen protocol -- profiling and empirical BF16 UMMA per-SM ceiling

This document freezes the P2.4 contract: a reproducible profiling and
analysis layer built entirely on top of the already-implemented,
independently audited, and GB300-verified P2.1 (`src/compute/umma_1sm.cu`),
P2.2 (`src/compute/umma_2sm.cu`), and P2.3
(`scripts/run_exp02_umma_throughput.sh`,
`scripts/aggregate_exp02_umma_throughput.py`) infrastructure. P2.4
introduces no CUDA kernel, no change to either UMMA binary, no change to
either SASS checker, and no change to the P2.3 plan, order, runner,
aggregator, or CSV schema; it only drives one complete pilot through the
unmodified P2.3 runner, profiles the same 24 configurations with Nsight
Compute, and computes deterministic statistics, clock-calibrated TFLOP/s,
1-SM/2-SM scaling, candidate depth saturation, and an empirical per-SM BF16
Tensor Core ceiling *candidate*.

**Status: implemented. Independent audit: pending. GB300 verification:
pending. No P2.4 campaign has been executed. No empirical ceiling has been
measured. Publishable results: NONE. Phase 2: not closed.**

## 0. Trust model (binding on this document, mirrors P1_4_PROTOCOL.md section 0)

The campaign filesystem under `results/raw/exp02_umma_throughput_p24/` is
trusted and single-writer, exactly like every other raw campaign tree in
this repository. P2.4's manifest chain, no-clobber publication, and
evidence-integrity gates protect against accidental corruption, malformed
or stale evidence, interrupted execution, pre-existing unsafe paths,
accidental overwrites, and ordinary recovery failures. They do not claim to
defend against a malicious concurrent process running with the same
filesystem permissions, or against deliberate path or inode replacement
after validation within one operation. A future auditor should evaluate
every claim in this document against this scope, not against a general
adversarial-filesystem threat model.

## 1. Scope boundary: P2.3 versus P2.4

`src/compute/P2_3_PROTOCOL.md` section 1 lists exactly what P2.3 explicitly
excludes and defers to P2.4: TFLOP/s, an empirical Tensor Core ceiling,
1-SM versus 2-SM speedup, scaling efficiency, saturation, a winning
configuration, Nsight Compute conclusions, and any publishable performance
result. P2.4 implements all of these, and only these:

1. one complete 24-configuration `run_kind=benchmark` pilot, driven through
   the unmodified P2.3 runner with frozen parameters;
2. Nsight Compute profiling of the same 24 configurations, in the exact
   canonical P2.3 order -- never 24 additional sweep configurations;
3. clock-calibrated TFLOP/s estimates;
4. 1-SM versus 2-SM speedup and scaling efficiency;
5. candidate depth-saturation analysis;
6. an empirical per-SM BF16 Tensor Core ceiling candidate;
7. deterministic CSV, JSON, Markdown, and SVG analysis artifacts;
8. the provenance, preflight, manifest, and evidence-integrity gates needed
   to audit all of the above.

P2.4 does not add a CUDA kernel, does not change either UMMA binary or SASS
checker, does not change the P2.3 plan/order/runner/aggregator/CSV schema,
does not add another N/depth/datatype/operand-path/CTA-group variant, does
not profile FP8/FP4/sparse/block-scaled/non-BF16 instructions, does not
modify pinned versions or `sm_103a`, does not use NCU kernel duration as the
primary UMMA throughput timing, and does not start CuTe DSL GEMM, cuBLASLt,
or any Phase 3 work. Every artifact P2.4 can ever produce carries
`publishable: false` unconditionally. P2.4 produces a reviewed pilot and an
empirical ceiling *candidate*, never a final campaign; final publishable
campaigns remain Phase 4 work.

| Unit | Scope | Status in this document |
|------|-------|--------------------------|
| P2.1 | 1-SM UMMA. | **Implemented, independently audited, functionally verified on GB300.** Unmodified by P2.4. |
| P2.2 | 2-SM UMMA. | **Implemented, independently audited, functionally verified on GB300.** Unmodified by P2.4. |
| P2.3 | Joint 1-SM/2-SM sweep infrastructure, exactly 24 configurations. | **Implemented, independently audited, functionally verified on GB300.** Unmodified by P2.4. |
| P2.4 | Profiling and empirical ceiling: Nsight Compute, TFLOP/s, scaling, saturation. | **Implemented. Independently audited: NO. Verified on GB300: NO.** |

## 2. Frozen pilot protocol

`--pilot` shells out to `scripts/run_exp02_umma_throughput.sh`, unmodified,
with:

```text
run_kind:            benchmark
iterations:           1000
warmup_iterations:    10
repetitions:           30
configuration_count:   24
```

using the exact P2.3 execution order, including its alternating method order
within each `(N, depth)` pair (`src/compute/P2_3_PROTOCOL.md` section 2).
The P2.4 wrapper and the underlying P2.3 campaign share one explicit
`P2_4_CAMPAIGN_ID` (a canonical UTC timestamp `YYYYMMDDTHHMMSSZ`, checked
independently of P2.3's own more permissive campaign-ID pattern), creating
two independently inspectable raw roots:

```text
results/raw/exp02_umma_throughput/<campaign-id>/       (owned and validated by P2.3, unmodified)
results/raw/exp02_umma_throughput_p24/<campaign-id>/   (owned by P2.4)
```

`record-pilot` validates the P2.3 campaign's manifest (`status=COMPLETE`,
`run_kind=benchmark`, `requested.iterations=1000`,
`requested.warmup_iterations=10`, `requested.repetitions=30`,
`configuration_count_completed=24`, matching `git_commit`, and matching
`gpu_uuid`/`gpu_name`/`compute_capability` against the pilot preflight) and
records a reference to it (campaign ID, path, and SHA-256 of
`manifest.json`, `combined_samples.csv`, and `summary.csv`) -- it never
reimplements or rewrites P2.3's own files.

## 3. Frozen 24-case profile plan and kernel-symbol derivation

`scripts/analyze_exp02_umma_throughput_p24.py build_profile_plan()` calls
`scripts/aggregate_exp02_umma_throughput.py build_plan()` directly and adds
exactly one derived field, `kernel_symbol`, resolved from the canonical
symbol tables already audited and GB300-verified in
`src/compute/P2_PROTOCOL.md` section 4 and `src/compute/P2_2_PROTOCOL.md`
section 3:

```text
umma_1sm_m128n<N>k16_d<DEPTH>
umma_2sm_m256n<N>k16_d<DEPTH>
```

`check_profile_plan_contract()` independently re-derives every field of
every one of the 24 entries against P2.3's own `build_plan()`/
`check_plan_contract()` output and fails closed on any divergence, so a
future edit to either module cannot silently break the "same 24
configurations, same order" guarantee.

## 4. Frozen profile protocol: NCU invocation and CLI surface

For each of the 24 profile application invocations:

```text
run_kind:            benchmark
iterations:           1000
warmup_iterations:    0
repetitions:           1
```

Each invocation of `build/compute/umma_1sm` or `build/compute/umma_2sm`
performs one untimed pre-timing correctness-validation launch of the exact
kernel symbol, followed by the one timed launch (`warmup_iterations=0`
means no warm-up launches intervene). The NCU invocation therefore uses an
exact function-name filter and profiles the *second* matching launch:

```text
--kernel-name-base function
--kernel-name <exact-symbol>
--launch-skip 1
--launch-count 1
```

Frozen profiler controls (verified against the pinned NCU 2025.4 `--help`
output at runtime, before any GPU/Docker/raw-tree work, by
`check_ncu_help_capability` in `scripts/run_exp02_umma_throughput_p24.sh`):

```text
--clock-control none
--pipeline-boost-state dynamic
--cache-control none
--devices 0
--replay-mode kernel
--print-summary none
```

Never used: `--force-overwrite`, `--set full`, `--clock-control base`,
`--clock-control boost`, `--clock-control force-boost`.

### 4.1 Metric discovery and resolution

Run once per profile campaign (GPU-touching, no clock control):

```bash
scripts/run_container.sh ncu --query-metrics --query-metrics-mode all --devices 0
```

`resolve_ncu_metrics_p24()` resolves every candidate metric name only by an
exact canonical-name match or an exact `.<canonical-name>` suffix
(namespace-qualified GB300 identifiers such as
`FBSP.TriageCompute.sm__cycles_elapsed.avg.per_second`). Two or more
qualified identifiers mapping to the same candidate are recorded
`ambiguous`, never guessed. The full identifier NCU reports is preserved and
passed back to `--metrics` verbatim.

Unlike P1.4's `resolve_ncu_metrics` (which raises and fails the whole
profiling step on any ambiguous candidate), no P2.4 candidate -- mandatory
or diagnostic -- ever raises at discovery time. This is a deliberate,
documented difference: P2.4 has an explicit `INCONCLUSIVE` terminal outcome
(section 8) that exists precisely to record "the raw evidence was captured,
but the mandatory clock metric could not be trusted" without discarding the
23 other configurations' evidence. A missing or ambiguous *mandatory*
metric is recorded and later drives the whole campaign's `analyze` step to
`INCONCLUSIVE`; a missing or ambiguous *diagnostic* metric is recorded and
reported explicitly, and never blocks the campaign at all.

Mandatory conversion metric:

```text
sm__cycles_elapsed.avg.per_second
```

Diagnostic (collected when available, never gating the campaign):

```text
gpu__time_duration.sum
device__attribute_multiprocessor_count
sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed
sm__inst_executed_pipe_tensor.sum
smsp__inst_executed_pipe_tensor.sum
```

If no suitable tensor activity counter resolves, this is reported
explicitly (`resolved_ncu_metrics.per_metric.<candidate>.status`); the
campaign relies only on the already-audited real-cubin SASS evidence
(`scripts/check_umma_1sm_sass.py`, `scripts/check_umma_2sm_sass.py`) for
positive `UTCHMMA`/`UTCHMMA.2CTA` identification, exactly as the task brief
requires. The SM-clock metric is the only metric this module ever treats as
mandatory.

### 4.2 SM-clock unit policy

For the verified `cycle/nsecond` representation:

```text
sm_clock_hz = metric_value * 1e9
```

`evaluate_sm_clock()` requires, for every profiled case, in order: the
metric resolved at discovery; a present value in that case's own
`metrics_raw.csv`; a finite value; a strictly positive value; and an exact
(case/whitespace-normalized) match against `cycle/nsecond`. Any other unit
is rejected outright -- never rescaled, never guessed. A case failing any
of these checks is recorded `sm_clock_valid=False` with the specific reason
(`metric_unavailable_at_discovery`, `missing_from_case_evidence`,
`non_finite`, `non_positive`, or `unknown_unit:<unit>`); raw evidence
capture for that case still succeeds (section 8).

### 4.3 Container-side bridge

`scripts/p24_ncu_bridge.py` adapts P1.4's proven private-`/tmp`,
length-delimited NCU bridge design (`scripts/p14_ncu_bridge.py`,
`src/memory/P1_4_PROTOCOL.md` section 4): both the collection and the
GPU-free metrics-export invocations run entirely inside the container's own
private, non-host-mounted `/tmp`, and NCU never receives a path under the
host raw campaign tree for `-o`, `--log-file`, or `--import`. The bridge
emits one versioned, length-delimited bundle (six fixed segments:
`app_stdout`, `app_stderr`, `ncu_tool_log`, `ncu_rep`, `metrics_csv`,
`metrics_export_stderr`) on its own stdout; the host captures it through
`scripts/p24_safe_capture.py run` and republishes it through
`publish-bundle`. The application CSV (the profiled UMMA binary's own
37-column stdout row) is recovered separately from the published
`container_stdout.log`, exactly as P1.4 recovers its own application CSV.

## 5. Preflight and provenance gate

Both `--pilot` and `--profile` require `BLACKWELL_GPU_INDEX`,
`P2_4_CAMPAIGN_ID`, and `P2_4_PREFLIGHT_SUMMARY` before any GPU,
Docker/NCU, or raw-campaign write. The preflight file must be, at the
moment of the check: a non-empty, non-symlink regular JSON file;
`overall_status == "PASS"`; fresh within 24 hours; `git_dirty == false`; a
full 40-character commit equal to the current, clean `HEAD`; compute
capability `10.3`; exactly one visible logical GPU
(`checks.gpu_visibility.status == "PASS"`); `checks.ncu_profile.status ==
"PASS"`. `compare_preflight_provenance()` requires the same `git_commit`,
GPU UUID, GPU name, compute capability, and driver version across the
pilot-phase and profiling-phase preflight snapshots, enforced twice: once
by the GPU-free `validate-profile-preconditions` subcommand before any
Docker/NCU invocation, and again by `discover-metrics` itself as a hard
gate at the exact point the campaign commits to `PROFILE_IN_PROGRESS`.
Never stores secrets, unrelated host metadata, usernames, or absolute
home/repository paths.

## 6. Required calculations (frozen formulas)

All 30 retained pilot samples of all 24 configurations are used in every
statistic below; no sample is ever removed automatically. Configurations
are processed in one fixed `(N, depth, method)` sorted order, and the four
per-configuration metrics in one fixed order
(`elapsed_cycles, cycles_per_umma, flops_per_cycle,
flops_per_cycle_per_sm`), so that, given the same input evidence, the exact
same sequence of `random.Random` draws -- and therefore byte-identical
output -- is reproduced on any machine, any number of times.

```text
flops_per_cycle_per_sm = flops_per_cycle / cta_group
```

Per configuration, per metric: count, mean, median, sample standard
deviation (`n-1`), coefficient of variation, minimum, maximum, a
deterministic 95% bootstrap confidence interval for the median, and
Tukey-IQR diagnostic bounds plus flagged-sample count. IQR-flagged samples
remain in every primary calculation. A stability-review diagnostic is added
when `flops_per_cycle`'s CV exceeds 5%.

Bootstrap: Python standard library only (`random`, `statistics`, `math`).
Frozen seed `20260804`, `10000` resamples, nearest-rank 2.5th/97.5th
percentiles (indices `int(0.025*resamples)-1` / `int(0.975*resamples)-1`,
clamped to `[0, resamples-1]`).

### 6.1 1-SM/2-SM scaling

For every one of the 12 `(N, depth)` pairs:

```text
speedup_2sm_over_1sm     = median_flops_per_cycle_2sm / median_flops_per_cycle_1sm
scaling_efficiency        = speedup_2sm_over_1sm / 2
scaling_efficiency_percent = 100 * scaling_efficiency
```

with a 95% bootstrap interval for the ratio of medians, independently
resampling the 1-SM and 2-SM sample sets each of the 10,000 iterations
(never described as "paired samples": the two configurations execute
sequentially, never concurrently). Scaling efficiency is never clamped;
values outside `[0, 100]` are preserved and flagged
(`surprising_value_flag`) for review.

### 6.2 Candidate depth saturation

For every one of the 6 `(method, N)` groups, over `depth in {4, 16, 64,
256}`:

1. find the largest median `flops_per_cycle`;
2. scan depths ascending; select the first depth whose median is `>= 0.95 *
   max` **and** whose 95% bootstrap interval overlaps the maximum
   configuration's own interval;
3. if no smaller depth qualifies, select the depth that achieves the
   maximum (this always exists, since it trivially satisfies both
   conditions against itself).

Reported as `earliest_tested_candidate_saturation_depth` -- never a
universal architectural saturation depth; limited to the four tested
depths per group.

### 6.3 Empirical per-SM ceiling selection

`select_ceiling()` chooses the best 1-SM configuration, the best 2-SM
configuration, and the overall empirical per-SM ceiling candidate
exclusively by the largest median `flops_per_cycle_per_sm` -- clock-
independent FLOP/cycle space -- before any clock is ever consulted. Ties
resolve to the first configuration in the fixed `(N, depth, method)` sorted
order.

Each selected configuration is then converted using *that same
configuration's own* matching NCU SM-clock measurement:

```text
estimated_local_tflops  = median_flops_per_cycle * sm_clock_hz / 1e12
estimated_tflops_per_sm = estimated_local_tflops / cta_group
```

A device-wide extrapolation is emitted only when a trustworthy SM-count
attribute (`device__attribute_multiprocessor_count`) resolves and is
identical across every profiled configuration that reported it:

```text
estimated_device_equivalent_tflops = estimated_tflops_per_sm * multiprocessor_count
```

always labelled an extrapolation from a one-/two-SM microbenchmark, never a
directly measured whole-GPU throughput, and never a theoretical
architectural peak.

### 6.4 The INCONCLUSIVE outcome

If the mandatory SM-clock metric is unavailable, ambiguous, malformed,
non-finite, non-positive, or has an unknown unit for **any** of the 24
profiled configurations, `analyze()` still computes and publishes every
clock-independent artifact (configuration statistics, scaling, saturation,
`profile_validation.csv`, both figures that do not require a clock, and
`report.md`) -- raw evidence capture and the science that does not depend
on a wall-clock conversion are never discarded -- but the campaign state
becomes `INCONCLUSIVE` rather than `ANALYZED`, `empirical_ceiling.json`
records `status: "INCONCLUSIVE"` plus the specific per-case reasons, and
**no** configuration's `estimated_local_tflops`,
`estimated_tflops_per_sm`, or `estimated_device_equivalent_tflops` is ever
populated anywhere in `analysis/*` -- including for a configuration whose
own individual SM-clock reading happened to be valid, since a completed
empirical-ceiling claim requires the *whole* campaign's clock evidence to
be trustworthy, not just the winning configuration's.

## 7. Campaign layout, manifest, and state machine

```text
results/raw/exp02_umma_throughput_p24/<campaign-id>/
├── manifest/                  (append-only, hash-chained revisions: 000000.json, 000001.json, ...)
├── profile_plan.csv           (the frozen 24-case plan, written once)
├── logs/
├── profiles/<24 canonical case names>/
│   ├── <case>_report.ncu-rep
│   ├── <case>.ncu_tool.log
│   ├── <case>.container_stdout.log
│   ├── <case>.container_stderr.log
│   ├── <case>.application.csv
│   ├── <case>.metrics_raw.csv
│   └── <case>.metrics_export_stderr.log
└── analysis/
    ├── configuration_statistics.csv
    ├── scaling.csv
    ├── saturation.csv
    ├── profile_validation.csv
    ├── empirical_ceiling.json
    ├── report.md
    ├── throughput.svg
    ├── scaling_efficiency.svg
    ├── saturation.svg
    └── analysis_manifest.json
```

State machine (`ALLOWED_P24_TRANSITIONS` in
`scripts/analyze_exp02_umma_throughput_p24.py`):

```text
None               -> PILOT_IN_PROGRESS
PILOT_IN_PROGRESS  -> PILOT_COMPLETE | FAILED | INTERRUPTED
PILOT_COMPLETE     -> PROFILE_IN_PROGRESS | FAILED | INTERRUPTED
PROFILE_IN_PROGRESS -> PROFILE_IN_PROGRESS | COMPLETE | FAILED | INTERRUPTED
COMPLETE           -> ANALYZED | INCONCLUSIVE
ANALYZED           -> (terminal)
INCONCLUSIVE       -> (terminal)
FAILED             -> (terminal)
INTERRUPTED        -> (terminal)
```

`COMPLETE` has no `FAILED`/`INTERRUPTED` edge: `analyze()` is a pure,
retriable function of already-validated evidence (mirrors P1.4's identical
`COMPLETE -> ANALYZED`-only design), so a failure during analysis leaves the
campaign at `COMPLETE` rather than discarding validated profile evidence.
`PILOT_IN_PROGRESS` has no self-loop (`--pilot` never reports incremental
progress into the manifest); `PROFILE_IN_PROGRESS` self-loops exactly once
per validated case, appending exactly one new `case_results` entry each
time.

Every manifest field is classified into exactly one of six categories
(immutable, allowed-timestamp, set-once, state-derived, append-only,
failure-only, plus the `INCONCLUSIVE`-only `inconclusive_reason`), bound to
the exact one state that may first introduce it, and bound to the exact one
adjacent-revision transition that may change it
(`P24_EXACT_TRANSITION_MUTATIONS`). Presence is validated separately from a
value of `null`: no manifest field in this contract is ever nullable, so an
unexpected `null` fails the type check outright rather than being
mistakable for an absent field. `validate_manifest_timestamp_chronology`
requires every lifecycle timestamp present in a revision to be
nondecreasing (`started_at_utc <= pilot_completed_at_utc <=
profile_started_at_utc <= profile_completed_at_utc <=
analysis_completed_at_utc`). The manifest is never a single mutable file:
each transition appends one complete, immutable snapshot to `manifest/` as
the next contiguous revision (`load_p24_manifest_chain` re-opens,
re-hashes, and re-validates every revision from `000000.json` forward on
every call; `write_next_p24_manifest_revision` publishes via
exclusive-create-to-a-temporary followed by hard-link-then-unlink
no-clobber -- never `os.replace()`).

Before both `COMPLETE` and publishing `ANALYZED`/`INCONCLUSIVE`,
`verify_campaign_evidence_integrity()` re-opens the campaign directory,
`profiles/`, and every case directory with descriptor-anchored, no-follow
resolution (held open for the whole check), confirms `profiles/` contains
exactly the 24 canonical case directories, re-hashes every trusted input
fresh from disk, reconstructs every case's complete result from its raw
evidence alone, and compares the reconstruction against what is recorded
via a strict recursive structural comparison (exact key sets, exact types,
never `dict.get()`-based equality) -- so a validated artifact, or any single
recorded derived value, modified after the fact is rejected rather than
silently accepted. `analyze()` re-runs this gate a second time, immediately
before publishing the terminal state, so evidence that changed *while
analysis itself was running* is also caught; on failure, the analysis
artifacts just written are removed (ownership/identity-checked, never an
unfamiliar replacement) and no terminal state is published.

## 8. Safety and evidence integrity

Follows every rule in `AGENTS.md`: explicit operator-selected physical GPU;
conservatively verified idle GPU; exactly one UUID exposed as logical
device 0; no automatic GPU selection; no `--gpus all`; no privileged mode,
host PID namespace, added capabilities, `SYS_ADMIN`, Docker socket mount,
`sudo`, or MPS; no multi-GPU execution; no clock, persistence, compute-mode,
or power-limit changes; no `$(nproc)`; correctness before timing or
profiling; preserve existing work and fail closed. All GPU work runs
sequentially through `scripts/run_container.sh`. `scripts/p24_safe_capture.py`
(adapted from the audited `scripts/p14_safe_capture.py`) provides exclusive,
no-clobber publication for every raw-tree write this project performs;
child processes write through already-open safe-capture descriptors or
through the private-container NCU bridge, never ordinary shell
redirections into the raw tree. `os.replace()` is never used for evidence
or manifest revisions.

## 9. Commands

GPU-free (no Docker, no GPU, no network; used to produce and validate this
implementation):

```bash
bash -n scripts/run_exp02_umma_throughput_p24.sh
python3 -m py_compile \
    scripts/analyze_exp02_umma_throughput_p24.py \
    scripts/p24_safe_capture.py \
    scripts/p24_ncu_bridge.py

scripts/run_exp02_umma_throughput_p24.sh --help
scripts/run_exp02_umma_throughput_p24.sh --print-plan
scripts/run_exp02_umma_throughput_p24.sh --self-test

python3 scripts/analyze_exp02_umma_throughput_p24.py --self-test
python3 scripts/p24_safe_capture.py --self-test
python3 scripts/p24_ncu_bridge.py --self-test

make compute-umma-p24-plan
make compute-umma-p24-check
```

GB300-executing (not yet run; requires an explicit, conservatively verified
free physical GPU and a fresh preflight):

```bash
BLACKWELL_GPU_INDEX=<i> make preflight
BLACKWELL_GPU_INDEX=<i> P2_4_CAMPAIGN_ID=<YYYYMMDDTHHMMSSZ> \
    P2_4_PREFLIGHT_SUMMARY=results/preflight/<TS>/summary.json \
    make compute-umma-p24-pilot

BLACKWELL_GPU_INDEX=<i> P2_4_CAMPAIGN_ID=<same-id> \
    P2_4_PREFLIGHT_SUMMARY=results/preflight/<TS-or-fresher>/summary.json \
    make compute-umma-p24-profile

P2_4_CAMPAIGN_ID=<same-id> make compute-umma-p24-analyze
```

## 10. Status

```text
P2.4 | Profiling and empirical ceiling | YES | NO | NO |
```

That is: Implemented = **YES**; Independently audited = **NO** (pending);
Verified on GB300 = **NO** (pending). No P2.4 campaign has been executed on
real hardware. No empirical ceiling has been measured. No publishable
result exists. Phase 2 remains **not closed** (P2.4's own audit and GB300
verification are the remaining gate items). Phase 3 remains gated on the
Phase 2 gate passing.

## 11. Verification and scientific limitations (recorded in advance)

* `elapsed_cycles`, `cycles_per_umma`, and `flops_per_cycle` are exactly the
  raw, unconverted per-sample quantities P2.1/P2.2/P2.3 already produce and
  have already audited; P2.4 adds no new correctness check to the UMMA
  binaries themselves and trusts only their existing `--self-test` and
  per-repetition `correctness=OK`/`mismatches=0`/`max_abs_error=0`
  contract.
* NCU kernel duration (`gpu__time_duration.sum`) is collected only as a
  diagnostic; it is never used to derive TFLOP/s, which comes exclusively
  from the pilot's own `%clock64`-timed `flops_per_cycle` combined with the
  matching profile's SM-clock reading.
* The empirical per-SM ceiling is a **candidate** from a one-pilot,
  one-profile-pass campaign -- not a final architectural peak, and a
  one-/two-SM device-wide extrapolation (when emitted at all) is explicitly
  labelled as such, never a directly measured whole-device throughput.
* This document, the five new files it describes, and the Make/documentation
  updates around them have not yet been independently audited or exercised
  on GB300 hardware. Every claim of correctness above is a design claim
  backed by GPU-free synthetic/adversarial self-tests only.
