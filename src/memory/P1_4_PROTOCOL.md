# P1.4 frozen protocol — profiling, HBM validation, analysis, pilot

**Status: implemented, independent audit PENDING, GB300 verification NO, pilot
NOT executed, NCU/HBM validation NO. No performance result exists yet.**

This document is the single frozen reference for P1.4
(`scripts/run_exp01_memory_paths_p14.sh`,
`scripts/analyze_exp01_memory_paths_p14.py`). It adds a reproducible layer
around the audited P1.1/P1.2/P1.3 infrastructure. P1.4 does not add or modify
CUDA kernels, does not change the frozen 18-configuration matrix or its
execution order, does not modify P1.3 validation/aggregation semantics, does
not create a new memory-copy method, and does not run a final campaign. All
timing values remain named **effective copy bandwidth**, never automatically
HBM/DRAM bandwidth — that determination is the narrow, six-case NCU
validation below, and it is scoped to exactly those six cases.

See `AGENTS.md` for the binding shared-cluster rules and `PLAN.md` for the
per-unit audit ledger.

## 1. Scope boundary

P1.4 implements, but this implementation has not executed:

1. one 18-configuration `run_kind=benchmark` pilot, run through the audited
   P1.3 runner unmodified;
2. Nsight Compute profiling of six predefined representative cases;
3. an HBM/DRAM-traffic validation classification for those six cases only;
4. descriptive statistics (mean/median/stdev/CV, bootstrap CI, IQR
   diagnostics) over all 30 retained repetitions per configuration;
5. a paired LDGSTS/TMA ratio comparison per identical `(stages,
   bytes_in_flight_kib)` pair;
6. a "candidate saturation" search over the three tested bytes-in-flight
   values per `(method, stages)` group;
7. deterministic CSV/JSON/Markdown/SVG artifacts, all `publishable: false`.

P1.4 explicitly does **not**: claim a universal HBM ceiling; call NCU-profiled
kernel duration a benchmark timing; declare a "winner"; run a final
(non-pilot) campaign; or start P2/experiment 2.

## 2. Frozen pilot protocol

The pilot is **not** a new sweep implementation. `--pilot` mode shells out to
the already-audited `scripts/run_exp01_memory_paths.sh`, unmodified, with the
frozen parameters below — it reuses P1.3's exact 18-invocation order (2
methods x 3 stage counts x 3 bytes-in-flight values, methods alternating lead
per configuration pair), CSV validation, correctness checks, aggregation, and
manifest rules. P1.4 adds no reimplementation of any of that.

```text
run_kind:        benchmark
working_set_mib: 512
passes:          32
warmup_ms:       2000
repetitions:     30
```

These five values are frozen constants in `analyze_exp01_memory_paths_p14.py`
(`FROZEN_PILOT_PARAMS`) and are never adjusted by a CLI flag. The existing
P1.3 benchmark-mode validation (`working_set_bytes > 2 * l2_bytes`) still
applies unmodified inside the P1.3 runner; if the selected GB300's L2 makes
512 MiB fail that check, the pilot stops (P1.3 already fails closed here) —
P1.4 never substitutes a larger working set automatically.

The P1.3 campaign the pilot drives shares its campaign ID with the P1.4
wrapper campaign that records it (`results/raw/exp01_memory_paths/<id>/` for
the P1.3 data, `results/raw/exp01_memory_paths_p14/<id>/` for the P1.4
wrapper); this keeps the two raw trees independently inspectable while making
the relationship explicit and traceable by a shared identifier.

## 3. Frozen six-case Nsight Compute plan

Exactly six cases, in this fixed order (diagnostic low/centre/high sample; it
never adapts to pilot results and never selects only favourable
configurations):

```text
index  method   stages  bytes_in_flight_kib  kernel
0      ldgsts   2       16                   ldgsts_benchmark_kernel
1      tma      2       16                   tma_benchmark_kernel
2      tma      4       32                   tma_benchmark_kernel
3      ldgsts   4       32                   ldgsts_benchmark_kernel
4      ldgsts   8       64                   ldgsts_benchmark_kernel
5      tma      8       64                   tma_benchmark_kernel
```

`ldgsts_benchmark_kernel<STAGES, COPIES>` and `tma_benchmark_kernel<STAGES,
COPIES>` are the exact template names in `src/memory/ldgsts.cu` /
`src/memory/tma.cu`; `--kernel-name-base function` matches against the
un-templated function name, which is why the exact literal names above are
sufficient as `--kernel-name` filters without a `regex:` prefix.

Each profiled invocation of the benchmark binary uses these frozen
parameters (`FROZEN_PROFILE_PARAMS`), deliberately different from the pilot's:

```text
run_kind:        benchmark
working_set_mib: 512
passes:          32
warmup_ms:       0
repetitions:     1
```

`warmup_ms=0` and `repetitions=1` are intentional: NCU replay already
re-executes the kernel as needed for multi-pass metric collection, and the
pilot (not the profile step) is what characterizes steady-state variability.
The binary's own correctness validation (over the full working set) still
always runs before its benchmark kernel, unconditionally, exactly as it does
for every other invocation of these binaries.

## 4. Verified NCU 2025.4.0.0 CLI surface

The exact flags below were confirmed against the pinned image's real
`ncu --version` / `ncu --help` (NVIDIA (R) Nsight Compute Command Line
Profiler, Version 2025.4.0.0, build 36690805, public-release) during
implementation — not assumed. `run_exp01_memory_paths_p14.sh --profile` also
re-verifies this at run time (see "Runtime NCU capability gate" below) so a
future NCU upgrade cannot silently change behaviour underneath the frozen
protocol.

| Flag | Confirmed allowed values | Frozen choice |
| --- | --- | --- |
| `--clock-control` | `base`, `boost`, `force-boost`, `none`, `reset` | `none` |
| `--pipeline-boost-state` | `stable`, `dynamic` | `dynamic` |
| `--cache-control` | `all`, `none` | `none` |
| `--kernel-name-base` | `function`, `demangled`, `mangled` | `function` |
| `--replay-mode` | `kernel`, ... | `kernel` |
| `--devices` | comma-separated device list | `0` |

Never used, and mechanically grepped for in the diff (`git diff --check`
target, see `Makefile`/CI note): `--force-overwrite`/`-f`, `--set full`,
`--clock-control base|boost|force-boost`.

### Runtime NCU capability gate

Before doing anything else, `--profile` mode runs a GPU-free capability probe
(no `--gpus`, no `BLACKWELL_GPU_INDEX` needed for this specific step):

```bash
docker run --rm --network none --security-opt no-new-privileges --cap-drop ALL \
    gb300-gemm-anatomy:phase0 ncu --help
```

and greps the captured text for every flag name and every required value
listed above, plus `--query-metrics`, `--query-metrics-mode`, `--metrics`,
`--csv`, `--page`, `--print-metric-name`, `--print-units`, `--log-file`,
`--import`, `-o`/`--export`, `--launch-count`, `--kernel-name`. If any is
missing, `--profile` stops immediately (exit 1) and reports exactly which
flag/value is unavailable; it never falls back to an NCU default that could
control clocks.

### Metric discovery (once per profile campaign, GPU-touching)

```bash
scripts/run_container.sh ncu --query-metrics --query-metrics-mode all --devices 0
```

captured to `logs/metric_discovery.{stdout,stderr}.log`. The candidate metric
list, in preference order:

```text
dram__bytes_read.sum                                    (mandatory)
dram__bytes_write.sum
dram__throughput.avg.pct_of_peak_sustained_elapsed
lts__t_bytes.sum
gpu__time_duration.sum
```

`dram__bytes_read.sum` is the semantic measurement "DRAM bytes read by the
profiled benchmark kernel." A metric is "resolved" only if its exact string
appears verbatim in the `--query-metrics-mode all` output for logical device
0 — never inferred, never substituted from a different architecture. If
`dram__bytes_read.sum` does not resolve, the whole six-case HBM
classification becomes `INCONCLUSIVE` (recorded explicitly in the manifest as
`resolved_ncu_metrics.dram_read_metric_available: false`); the other
resolved metrics (if any) are still collected and recorded, since they may
still be diagnostically useful, but no HBM claim is permitted. This is a
data-quality outcome, not a hard failure of the raw collection workflow —
`--profile` can still reach `COMPLETE` with `dram_read_metric_available:
false` recorded honestly.

### Collection command (per case, GPU-touching, via `scripts/run_container.sh`)

```bash
ncu \
    --clock-control none \
    --pipeline-boost-state dynamic \
    --cache-control none \
    --kernel-name-base function \
    --kernel-name <ldgsts_benchmark_kernel|tma_benchmark_kernel> \
    --launch-count 1 \
    --devices 0 \
    --replay-mode kernel \
    --metrics <comma-joined resolved metrics> \
    --print-summary none \
    --log-file <profiles>/<case>.ncu_tool.log \
    -o <profiles>/<case>_report \
    -- \
    build/memory/<ldgsts|tma> --stages <S> --bytes-in-flight-kib <B> \
        --run-kind benchmark --working-set-mib 512 --passes 32 \
        --warmup-ms 0 --repetitions 1
```

`--log-file` isolates NCU's own tool/progress output from the profiled
binary's inherited stdout/stderr (confirmed by `ncu --help`: "Send all tool
output to the specified file"), so the binary's own single CSV header+row
still lands cleanly on the invocation's stdout, alongside
`scripts/run_container.sh`'s own allowlisted banner lines. The application
CSV is recovered by scanning the captured stdout for the exact 37-column
`CSV_HEADER` line (reusing `aggregate_exp01_memory_paths.CSV_HEADER`) and
taking that line plus the one row that follows — never by assuming the
stream is otherwise clean. `-o` is never given `--force-overwrite`, so NCU
itself refuses to clobber an existing report; `run_exp01_memory_paths_p14.sh`
additionally pre-checks the target path with the same symlink/no-clobber
primitives P1.3 uses (imported, not reimplemented).

### Metrics export (per case, GPU-free — pure `.ncu-rep` post-processing, plain `docker run`, no `--gpus`, no `BLACKWELL_GPU_INDEX`)

```bash
docker run --rm --network none --security-opt no-new-privileges --cap-drop ALL \
    -v "$PWD:/workspace" -w /workspace gb300-gemm-anatomy:phase0 \
    ncu --import <profiles>/<case>_report.ncu-rep \
        --csv --page raw --print-metric-name name --print-units base \
    > <profiles>/<case>.metrics_raw.csv
```

`--print-metric-name name` selects the raw metric identifier (e.g.
`dram__bytes_read.sum`) as the CSV column header instead of NCU's
human-readable label; `--print-units base` keeps units unscaled. This step
never touches a GPU (it reads an already-collected `.ncu-rep`), so it does
not need `BLACKWELL_GPU_INDEX` or the idle-device proof — only the
GPU-touching collection step above does.

### Preserved, never-overwritten artifacts per case

```text
profiles/<case>/<case>_report.ncu-rep
profiles/<case>/<case>.ncu_tool.log
profiles/<case>/<case>.container_stdout.log
profiles/<case>/<case>.container_stderr.log
profiles/<case>/<case>.application.csv
profiles/<case>/<case>.metrics_raw.csv
profiles/<case>/<case>.metrics_export_stderr.log
```

plus one campaign-level `logs/metric_discovery.{stdout,stderr}.log` and
`logs/ncu_help_capability_probe.log`.

## 5. HBM validation rule (six profiled cases only)

```text
dram_read_ratio = dram_read_bytes / useful_bytes
```

`useful_bytes` is the already-validated field from that case's own
application CSV row (`working_set_bytes * passes`, `working_set_mib=512`,
`passes=32`); `dram_read_bytes` is `dram__bytes_read.sum` from that case's
`metrics_raw.csv`.

```text
HBM_VALIDATED   :  dram_read_ratio >= 0.90
INCONCLUSIVE    :  dram_read_ratio <  0.90, missing, non-finite, malformed,
                    or dram__bytes_read.sum unsupported
```

If `dram_read_ratio > 1.10`, the case keeps `HBM_VALIDATED` but gains a
`READ_AMPLIFICATION` diagnostic flag — the excess traffic is reported, never
hidden or normalized away. NCU's
`dram__throughput.avg.pct_of_peak_sustained_elapsed`, if resolved, is recorded
verbatim and labelled exactly "NCU DRAM peak-sustained utilization" —
never called an empirical HBM ceiling.

This establishes predominant DRAM traffic for the six profiled cases only. It
is never extrapolated to the other twelve configurations in the pilot.

## 6. Statistical policy

All 30 retained repetitions of each of the 18 pilot configurations are used;
none are ever automatically removed. Per configuration:
count, mean, median, sample standard deviation (`n-1`), coefficient of
variation (`100*stdev/mean`), min, max, a 95% bootstrap CI for the median,
IQR diagnostic bounds (Tukey fences, linear/`PERCENTILE.INC`-style
interpolated quartiles), and the count of IQR-flagged samples (diagnostic
only — flagged samples remain in every primary statistic). A configuration
is flagged for stability review when `effective_gbps_cv_percent > 5.0`; this
is a diagnostic, not a filtering rule.

Bootstrap is deterministic: Python's `random.Random(20260728)` (the literal
seed `20260728`), 10,000 resamples, nearest-rank 2.5th/97.5th percentiles
(zero-based indices 249 and 9749 of the sorted 10,000 resample statistics),
standard-library `random` module only. All 18 configurations, all 9 pairwise
comparisons, and all 6 `(method, stages)` saturation groups are processed in
one fixed, explicitly sorted order — `(stages, bytes_in_flight_kib, method)`
for configurations, `(stages, bytes_in_flight_kib)` for pairs, `(method,
stages)` for saturation groups — so that, given the same input samples, the
exact same sequence of `random.Random` draws (and therefore bit-identical
output) is reproduced on any machine, any number of times. Non-finite
`effective_gbps` values are rejected outright (P1.3 already guarantees this
on a `COMPLETE` campaign; P1.4 re-checks defensively rather than trusting
that silently).

### Pairwise LDGSTS/TMA comparison

For each of the 9 identical `(stages, bytes_in_flight_kib)` pairs:

```text
tma_to_ldgsts_ratio = median_effective_gbps_tma / median_effective_gbps_ldgsts
```

with an independently-resampled 95% bootstrap CI (LDGSTS's 30 samples and
TMA's 30 samples are resampled separately in each of the 10,000 iterations).
Interpretation is fixed to: ratio > 1 means TMA measured higher effective
copy bandwidth in this pilot; ratio < 1 means LDGSTS did; ratio = 1 means
equal medians. No p-value, significance claim, or the word "winner" is ever
emitted. The fixed, non-randomized P1.3 execution order is carried into the
report as a named limitation of this single pilot.

### Candidate-saturation rule

Per `(method, stages)` group (6 groups total), over the three tested
`bytes_in_flight_kib` values (16, 32, 64):

1. `max_median` = the largest of the three observed medians;
2. scan 16, 32, 64 in ascending order; select the first (smallest) value
   whose median is `>= 0.95 * max_median` **and** whose bootstrap CI overlaps
   the bootstrap CI of the value that achieved `max_median`;
3. if no smaller candidate satisfies both conditions, the value that achieved
   `max_median` itself always qualifies (trivially: ratio 1.0, CI overlaps
   itself), guaranteeing a result exists.

The result is reported only as `earliest_tested_candidate_saturation_bif_kib`
— never a "universal architectural saturation threshold"; only three BIF
values are ever tested per group.

## 7. Preflight and provenance gate

`--pilot` and `--profile` each independently require `P1_4_PREFLIGHT_SUMMARY`
to point at a file that is, at the moment of the check:

* a non-symlink, non-empty regular JSON file (`lstat`-based, mirrors
  `aggregate_exp01_memory_paths._reject_if_symlink_or_wrong_type` /
  `_verify_artifact`, imported rather than reimplemented);
* `overall_status == "PASS"` (`scripts/preflight.sh`'s own top-level field);
* `git_dirty == false`;
* `git_commit` a full 40-character hex string equal to the current, clean
  `git rev-parse HEAD`;
* `gpu.compute_cap == "10.3"`;
* exactly one logical GPU — `scripts/preflight.sh` only ever populates its
  `gpu.*` fields, and only ever reports `checks[gpu_visibility].status ==
  "PASS"`, when its own `nvidia-smi` query returned exactly one row; P1.4
  treats that check's `PASS` plus a fully-populated `gpu` object as the proof
  (`scripts/preflight.sh` records no separate raw GPU count field — this is a
  documented reading of its existing, frozen output schema, not an
  invention);
* `checks[ncu_profile].status == "PASS"`;
* `timestamp_utc` no more than 24 hours before the moment of the check.

`--profile` accepts a fresh preflight summary independent of the one
`--pilot` used (operationally, profiling may happen hours or days after the
pilot) — but the P1.4 manifest then requires that every one of the following
agree across whichever preflight(s) were used, the P1.3 pilot campaign, and
all six profile application CSVs:

```text
git commit
GPU UUID
GPU name
compute capability
driver version        (compared only within like representations — see below)
CUDA runtime version
working-set parameters (working_set_mib=512, and the resulting working_set_bytes)
passes                 (32)
```

`scripts/preflight.sh`'s `gpu.driver_version` is the NVIDIA display-driver
package string (e.g. `580.95.05`, from `nvidia-smi`); a benchmark binary's
own `cuda_driver_version` CSV column is `cudaDriverGetVersion()`'s integer
encoding (`MAJOR*1000+MINOR*10`) — a different representation of a related
but distinct value. P1.4 never compares these two representations for
bit/string equality; it treats `cuda_driver_version` (and
`cuda_runtime_version`, which has no preflight equivalent at all) as
internally cross-checked only among the pilot's `combined_samples.csv` rows
and the six profile application CSVs, and treats the preflight's own
`driver_version` string as self-consistent only across whichever preflight
summary(ies) were actually used. Any disagreement anywhere in this list
makes the affected step `INCONCLUSIVE`/`FAILED`; nothing is ever silently
reconciled.

## 8. Raw output, state machine, no-clobber

Raw root (ignored by Git via the existing blanket `results/raw/` rule —
no `.gitignore` change needed):

```text
results/raw/exp01_memory_paths_p14/<campaign_id>/
    manifest.json
    profile_plan.csv
    profiles/
    analysis/
    logs/
```

`<campaign_id>` must match `^[0-9]{8}T[0-9]{6}Z$` (a real calendar UTC
instant) — stricter than P1.3's general campaign-id pattern, on top of which
it is still checked (imported `validate_campaign_id`). Every path component
from the raw root down is created/resolved with the same `lstat`-based,
symlink-refusing, no-`resolve()`-alone primitives P1.3 uses (imported, not
reimplemented): a real or dangling symlink at any level — including the raw
root itself, `profiles/<case>/`, or any individual artifact path — is
refused. No result, report, CSV, log, manifest temporary, analysis file,
figure, partial file, or failure artifact is ever overwritten; publication
uses hard-link-then-unlink no-clobber (never `os.replace()`) except for
`manifest.json` itself, whose atomic-replace lifecycle mirrors P1.3's (a
`.tmp` created exclusively, never following a symlink, replacing only after
verifying the prior file's identity was unchanged). NCU's own
`--force-overwrite` is never used. A failed launch leaves no stale `.tmp`;
non-empty partial evidence is preserved under a fresh `.invalid`/`.partial`
name, exactly like P1.3's capture step.

### State machine

```text
None              -> PILOT_IN_PROGRESS
PILOT_IN_PROGRESS -> PILOT_IN_PROGRESS | PILOT_COMPLETE | FAILED | INTERRUPTED
PILOT_COMPLETE     -> PROFILE_IN_PROGRESS | FAILED
PROFILE_IN_PROGRESS -> PROFILE_IN_PROGRESS | COMPLETE | FAILED | INTERRUPTED
COMPLETE           -> ANALYZED
ANALYZED           -> (terminal)
FAILED             -> (terminal)
INTERRUPTED        -> (terminal)
```

`COMPLETE` means the raw pilot-plus-six-profile collection workflow finished
successfully — it never means the result is publishable (`publishable` is a
separate field, always `false`, at every state). `ANALYZED` means
`analysis/*` was generated from a `COMPLETE` campaign; it is still not
publishable. A terminal state (`FAILED`, `INTERRUPTED`, `ANALYZED`) is never
reopened or rewritten. `failure_stage`/`failure_detail` record where and why
a campaign stopped, exactly like P1.3's manifest.

## 9. Analysis artifacts

Generated only from a `COMPLETE` P1.4 campaign, deterministically, via
no-clobber publish, standard library only (no NumPy/pandas/matplotlib/
notebook/Docker dependency in the analysis code path):

```text
analysis/pilot_statistics.csv
analysis/pairwise_comparison.csv
analysis/saturation_candidates.csv
analysis/ncu_validation.csv
analysis/analysis.json
analysis/report.md
analysis/figures/effective_gbps.svg
analysis/figures/tma_to_ldgsts_ratio.svg
analysis/figures/dram_read_ratio.svg
```

`report.md` always states plainly: this is a single pilot; timings are
CUDA-event pilot measurements, never NCU durations; no sample was ever
removed; NCU covers exactly six predefined cases, never the other twelve;
any candidate-saturation point is limited to the three tested BIF values; no
final or universal HBM ceiling is established; every artifact remains
`publishable: false` pending independent review and later final campaigns.

## 10. What an operator runs, and in what order, after independent audit

```bash
# Fresh preflight (mandatory; host driver changed since Phase 0):
BLACKWELL_GPU_INDEX=<physical-index> make preflight
# -> results/preflight/<TS>/summary.json

P1_4_CAMPAIGN_ID=$(date -u +%Y%m%dT%H%M%SZ)
BLACKWELL_GPU_INDEX=<physical-index> \
P1_4_CAMPAIGN_ID="${P1_4_CAMPAIGN_ID}" \
P1_4_PREFLIGHT_SUMMARY=results/preflight/<TS>/summary.json \
    make memory-paths-p14-pilot

BLACKWELL_GPU_INDEX=<physical-index> \
P1_4_CAMPAIGN_ID="${P1_4_CAMPAIGN_ID}" \
P1_4_PREFLIGHT_SUMMARY=results/preflight/<TS-or-fresher>/summary.json \
    make memory-paths-p14-profile

P1_4_CAMPAIGN_ID="${P1_4_CAMPAIGN_ID}" \
    make memory-paths-p14-analyze
```

None of these five commands were executed by this implementation task; it
stopped at the GPU-free plan/check layer.
