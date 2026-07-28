# P1.4 frozen protocol — profiling, HBM validation, analysis, pilot

**Status: implemented, remediated after two independent GPU-free audits;
a further independent re-audit is PENDING, GB300 verification NO, pilot NOT
executed, NCU/HBM validation NO. No performance result exists yet.**

A first independent GPU-free audit of the initial implementation found five
blockers, each closed with a GPU-free fix plus a new adversarial test that
first demonstrably failed against the original behavior and then passed
against the fix: (1) a profiling preflight from a different GPU/driver/commit
than the pilot's could be accepted — closed by `compare_preflight_provenance`
(Section 7); (2) a validated `metrics_raw.csv` (or any other trusted input)
could be modified after validation and still reach `COMPLETE`/`ANALYZED` —
closed by the central evidence-integrity gate (Section 8); (3) the NCU
raw-CSV parser accepted malformed, wrong-unit, or substring-matched evidence
— closed by the fail-closed parser (Section 4); (4) `--profile` wrote
diagnostic logs before safely resolving the campaign tree — closed by
reordering `--profile` and adding symlink-safe capture checks (Section 8);
(5) P1.4 manifest updates delegated to P1.3's overwrite-based writer,
contrary to the no-overwrite requirement — closed by the append-only,
hash-chained manifest revision design (Section 8).

A second independent GPU-free audit of that remediated implementation found
four further blockers, each closed the same way (a new adversarial test that
first demonstrably failed, then passed): (A) a precheck (even a symlink-aware
one) immediately before an ordinary shell redirection into the raw campaign
still left a TOCTOU window between the check and the later `open()` —
closed by `scripts/p14_safe_capture.py`, a P1.4-only descriptor-anchored
capture module that opens every directory component exactly once with Linux
no-follow semantics and never re-resolves a pathname afterward (Section 4);
(B) an unplanned extra `profiles/<name>/` directory was never compared
against anything and so was silently ignored regardless of state — closed by
`verify_profiles_directory_inventory`, which requires `profiles/`'s actual
contents to equal exactly the six canonical names in `profile_plan.csv`
(Section 8); (C) a syntactically valid, correctly re-hashed manifest revision
that changed the immutable `campaign_id` (or edited an earlier `case_result`,
or jumped state illegally) previously passed unnoticed, since the hash chain
alone only proves a revision was appended without altering an earlier byte,
never that its *content* is a legitimate continuation — closed by
`validate_manifest_revision_transition`, an explicit per-field classification
and transition validator applied to every adjacent revision pair (Section 8);
(D) the evidence-integrity gate recomputed several derived per-case values
but never compared its own `dram_read_bytes` reconstruction (or any
`resolved_metric_values` entry) against what was actually recorded, so either
could be silently tampered without the classification/ratio checks (computed
from the untouched CSV) ever disagreeing with anything — closed by
`reconstruct_case_result`, one canonical function shared by
`validate-profile-case` and the gate, compared as a complete structure, key
for key, never a hand-picked subset (Section 8).

None of these nine fixes, across both rounds, changed the frozen pilot
matrix, the six-case NCU plan, the statistical calculations, the bootstrap
seed/resample count, the outlier-retention policy, the saturation rule, or
the HBM thresholds below.

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
| `--print-kernel-base` | `function`, `demangled`, `mangled` | `function` |
| `--replay-mode` | `kernel`, ... | `kernel` |
| `--devices` | comma-separated device list | `0` |

`--print-kernel-base function` (metrics export step, below) makes the
exported CSV's `Kernel Name` column the same un-templated function name
`--kernel-name-base function` filters on at collection time, so the parser
can require exact string equality against the frozen kernel name — never a
substring/prefix/suffix match — with no risk of the two flags disagreeing on
representation.

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
taking that line plus the one row that follows, then publishing it via
`scripts/p14_safe_capture.py write` (below) — never by assuming the stream is
otherwise clean, and never via a shell `>` redirect. `-o`/`--log-file` are
NCU's own arguments; NCU resolves and writes those two paths itself (never
given `--force-overwrite`, so NCU itself refuses to clobber an existing
report), which this script cannot intercept — `run_exp01_memory_paths_p14.sh`
re-verifies both landed strictly inside the anchored `profiles/<case>/`
directory afterward via `scripts/p14_safe_capture.py verify` (Section 4a),
instead of trusting a plain `test -f`.

### Metrics export (per case, GPU-free — pure `.ncu-rep` post-processing, plain `docker run`, no `--gpus`, no `BLACKWELL_GPU_INDEX`)

```bash
docker run --rm --network none --security-opt no-new-privileges --cap-drop ALL \
    -v "$PWD:/workspace" -w /workspace gb300-gemm-anatomy:phase0 \
    ncu --import <profiles>/<case>_report.ncu-rep \
        --csv --page raw --print-metric-name name --print-units base \
        --print-kernel-base function
```

(stdout/stderr captured via `scripts/p14_safe_capture.py run`, below — never
a shell `>`/`2>` redirect into `<profiles>/<case>.metrics_raw.csv`.)
`--print-metric-name name` selects the raw metric identifier (e.g.
`dram__bytes_read.sum`) as the CSV column header instead of NCU's
human-readable label; `--print-units base` keeps units unscaled;
`--print-kernel-base function` makes the `Kernel Name` column the bare
function name (see the flag table above). This step never touches a GPU (it
reads an already-collected `.ncu-rep`), so it does not need
`BLACKWELL_GPU_INDEX` or the idle-device proof — only the GPU-touching
collection step above does.

### 4a. Descriptor-anchored safe capture (`scripts/p14_safe_capture.py`)

Every raw-campaign write `run_exp01_memory_paths_p14.sh --profile` performs —
the NCU-help-capability-probe log, both metric-discovery logs,
`discover-metrics`' own stderr log, each case's container stdout/stderr, the
extracted application CSV, and the exported metrics CSV/stderr — goes
exclusively through this P1.4-only module, never a plain
`>`/`>>`/`2>`/`2>>` shell redirection into `results/raw/exp01_memory_paths_p14/`
(mechanically confirmed: `rg -n '(^|[[:space:]])[0-9]*>>?'
scripts/run_exp01_memory_paths_p14.sh` and inspect every match). A precheck
immediately before an ordinary redirection — even a symlink-aware one — still
leaves a window between the check and the later `open()`; this module closes
that window structurally instead of narrowing it:

* every directory component from the repository root down to `logs/` or
  `profiles/<case>/` is opened exactly once, with `O_DIRECTORY | O_NOFOLLOW`,
  relative to the previously opened descriptor (`os.open(name, flags,
  dir_fd=parent_fd)`) — never re-resolved by pathname afterward, so nothing
  that later happens to the *name* (a symlink swap of an ancestor, or of the
  final target) can redirect any subsequent operation;
* the output file is created the same way, `O_EXCL | O_NOFOLLOW`, so it can
  never already exist (as a directory, regular file, or symlink, dangling or
  not) and can never be a symlink itself;
* the child process (`subprocess.run(argv, stdout=<fd>, stderr=<fd>,
  shell=False)`) writes directly to the already-open descriptor — it never
  performs its own path lookup for the output name at all;
* on a zero exit, the output is published via an in-directory hard link
  (`os.link(partial, final, src_dir_fd=fd, dst_dir_fd=fd)`) then unlink of the
  partial — `linkat()`'s own `EEXIST` is the sole, atomic no-clobber
  guarantee; never `os.replace()`;
* on a non-zero exit, a non-empty partial is preserved under its own unique
  name (never renamed or clobbered); an empty owned partial is removed after
  re-confirming its identity by descriptor, never by a second, racy path
  lookup.

`--rel-dir` is restricted to an explicit allowlist (`logs`, or one of the six
frozen `profiles/<case>` names); `mkdir-case` (the safe replacement for the
old `[ -L case_dir ] || [ -e case_dir ]; mkdir case_dir` precheck-then-create
pair, which was itself racy against `profiles/` — the parent — being
swapped) creates exactly one of those six directories via `mkdirat()`'s own
`EEXIST`, never a separate check. `write` publishes already-extracted bytes
(the application CSV) the same no-clobber way; `verify` confirms NCU's own
`-o`/`--log-file` outputs are genuine non-symlink files strictly inside the
anchored directory. See the module's own docstring and `--self-test` (two
dedicated adversarial reproductions: a symlink swap of `logs/` itself, and a
swap performed via an injectable hook after the directory descriptor is
already open but before the child writes) for the complete design.

### Fail-closed raw-CSV parsing

`metrics_raw.csv` is read with `csv.reader` (never `DictReader`, so a
duplicate header column name is itself detected) and rejected outright,
before any value is trusted, unless all of the following hold:

* the header contains exactly the five required columns `ID`, `Kernel Name`,
  `Metric Name`, `Metric Unit`, `Metric Value`, with no duplicate column name;
* every row has that same column count;
* the file contains exactly one distinct `ID` (launch) and exactly one
  distinct `Kernel Name` — never invented as `launch_count=1`, and never
  averaged/summed across multiple launches or kernels;
* the one `Kernel Name` present equals the frozen case's kernel name exactly
  (see above) — never a substring, prefix, or regex match;
* every metric's `Metric Unit` equals its expected unit exactly, after only
  case/whitespace normalization — `byte`, `Byte`, and `BYTE` are the same
  unit; `kilobyte`/`Kbyte`/`KB` are a **different**, rejected unit, never
  silently rescaled to bytes;
* every metric value is present, numeric, and finite (no empty/NaN/±infinity
  value is ever treated as zero or missing-but-ignorable);
* no metric name appears more than once, even with an identical value.

A metric NCU discovery recorded as *resolved* (`resolved_ncu_metrics.resolved`
in the manifest) but absent from this case's own `metrics_raw.csv` is a hard
validation failure, never a silent downgrade to `INCONCLUSIVE` — that
downgrade path is reserved exclusively for `dram__bytes_read.sum` never
having resolved for the whole campaign in the first place (Section 5).

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

The pilot-versus-profile preflight fields (`git_commit`, `gpu_uuid`,
`gpu_name`, `gpu_compute_cap`, `gpu_driver_version`) are compared by one
reusable function, `compare_preflight_provenance`, enforced at two points:
first, `--profile` calls the GPU-free, side-effect-free
`validate-profile-preconditions` subcommand immediately after safely
resolving the campaign tree and before any Docker/NCU invocation or raw-tree
log write, so a GPU/driver swap between the pilot and profiling runs aborts
before any expensive or irreversible work happens; second, `discover-metrics`
itself re-runs the identical comparison as a hard gate at the exact point
`preflight_reference_profile` is recorded and the campaign commits to
`PROFILE_IN_PROGRESS`, so the guarantee holds even if the orchestrating shell
script were bypassed. `compare_preflight_provenance` only ever takes two
preflight snapshots of identical shape (both produced by the same
`validate_preflight_file`), so it can never cross into the CUDA
driver/runtime integer encoding described next.

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
    manifest/
        000000.json
        000001.json
        ...
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
root itself, `profiles/<case>/`, `manifest/`, or any individual artifact path
— is refused. No result, report, CSV, log, manifest revision, analysis file,
figure, partial file, or failure artifact is ever overwritten; every
publication uses hard-link-then-unlink no-clobber (`_publish_no_clobber`) —
**never** `os.replace()`, anywhere, including for the manifest. NCU's own
`--force-overwrite` is never used. A failed launch leaves no stale `.tmp`;
non-empty partial evidence is preserved under a fresh `.invalid`/`.partial`
name, exactly like P1.3's capture step.

### Append-only, hash-chained manifest (never `os.replace()`)

Unlike P1.3's own `manifest.json` (a separate, frozen, already-audited input
this file never modifies, and which correctly keeps its own
atomic-replace-in-place lifecycle), the P1.4 manifest is never a single
mutable file. Each state transition appends one complete, immutable snapshot
to `manifest/` as the next contiguous revision file
(`000000.json`, `000001.json`, ...); nothing already on disk is ever edited
or replaced. Every revision document carries two extra fields beyond the
ordinary P1.4 manifest schema: `manifest_revision` (its own index) and
`previous_manifest_sha256` (the SHA-256 of the immediately preceding revision
file, or `null` for revision 0). Loading the manifest
(`load_p14_manifest_chain`) re-opens and re-validates *every* revision from
`000000.json` forward on every call — never trusting anything about an
earlier revision from memory — and rejects the whole campaign as invalid if:
any revision is a symlink (dangling or not); the revision filenames are not
exactly contiguous `000000.json..NNNNNN.json` with no extra or missing
entries; a revision's `manifest_revision` field does not match its own
position; a revision's `previous_manifest_sha256` does not match the
freshly-recomputed hash of the file that precedes it; or a revision's content
fails the ordinary P1.4 manifest schema. Writing the next revision
(`write_next_p14_manifest_revision`) re-derives the next revision number and
the previous revision's hash by re-reading the chain immediately before
writing, then publishes the new revision file via exclusive-create-to-a-
fixed-name-temporary followed by hard-link-then-unlink no-clobber — so a
concurrent writer racing for the same next revision number fails closed at
the final hard-link step, and a stale leftover temporary from an interrupted
write blocks (rather than silently overwrites) the next attempt. This is
strictly additive to P1.3's own manifest discipline: P1.4 never calls
P1.3's `write_manifest_atomic`/`os.replace()`-based writer for its own
manifest.

### Semantic manifest transition validation

The hash chain above proves a revision was appended without altering an
earlier byte; it says nothing about whether the new revision's *content* is
a legitimate continuation of the previous one. A syntactically valid,
correctly re-hashed revision that simply changed `campaign_id` (or edited an
earlier `case_result`, or jumped state illegally) previously passed
unnoticed. `validate_manifest_revision_transition(previous, current,
expected_campaign_id)` closes this: `load_p14_manifest_chain` applies it to
every adjacent revision pair while walking the chain (never only to the
latest revision), and `write_next_p14_manifest_revision` applies it once
more, defensively, immediately before writing. `expected_campaign_id` always
comes from the safely resolved campaign directory's own basename — a
revision's own stored `campaign_id` is never trusted by itself.

Every top-level P1.4 manifest field is classified into exactly one category
(a module-level assertion in `analyze_exp01_memory_paths_p14.py` fails at
import time if any field is ever left unclassified, so an unclassified field
can never silently bypass transition validation):

| Category | Fields | Rule |
| --- | --- | --- |
| immutable | `schema_version`, `experiment_id`, `campaign_id`, `publishable`, `frozen_protocol`, `profile_plan_sha256` | present from revision 0 onward; the exact same value forever |
| allowed timestamp metadata | `started_at_utc`, `pilot_completed_at_utc`, `profile_started_at_utc`, `profile_completed_at_utc`, `analyzed_at_utc` | may be absent, then take a fixed value at its own specific transition; never changes once set |
| set-once | `pilot_campaign_reference`, `preflight_reference_pilot`, `provenance`, `preflight_reference_profile`, `resolved_ncu_metrics`, `profile_order` | may be absent, then take a fixed value at its own specific transition; never changes once set (non-timestamp analogue of the row above) |
| state-derived | `state`, `profile_count_completed` | governed by the state machine below (state) or monotonically non-decreasing (the count) |
| append-only | `case_results`, `artifact_sha256` | may only gain new entries; an existing entry is never edited, deleted, or reordered; `case_results` must also grow strictly in the frozen six-case order (its key set is always exactly a prefix of the frozen order) |
| (failure fields) | `failure_stage`, `failure_detail` | may be non-`None` only alongside `state in (FAILED, INTERRUPTED)` |

`state` transitions (including a same-state "self-loop," such as appending
the next validated profile case) are legal only when
`ALLOWED_P14_TRANSITIONS` itself lists the target as reachable from the
current state — there is no separate "same state is always fine" exception,
since most states (`PILOT_COMPLETE`, `COMPLETE`, `ANALYZED`, `FAILED`,
`INTERRUPTED`) have no self-loop at all and a repeat is exactly as illegal as
any other disallowed jump; since the table only ever adds edges forward,
this same check also rejects any state regression. `analyzed_at_utc` and any
`artifact_sha256` key beginning `analysis/` may appear only once
`state == "ANALYZED"`.

### Evidence-integrity gate (re-verified before `COMPLETE` and before `ANALYZED`)

A validated artifact (any of: a case's `application.csv`, `metrics_raw.csv`,
or `.ncu-rep`; `profile_plan.csv`; either preflight summary; the P1.3
campaign's `manifest.json`, `combined_samples.csv`, or `summary.csv`) must
never be modifiable after validation and still reach a completing state, and
every profiled case's *complete* recorded result must still match what its
authoritative evidence alone reconstructs. `finalize-profile` (before
`PROFILE_IN_PROGRESS -> COMPLETE`) and `analyze` (before `COMPLETE ->
ANALYZED`) both call the same function, `verify_campaign_evidence_integrity`,
unmodified. It:

1. re-derives every artifact's path from the frozen NCU plan and canonical
   case names alone (never from a stored path string) and recomputes every
   SHA-256 fresh from disk (pilot/profile preflights, the P1.3 terminal
   manifest and its aggregate CSVs, this campaign's own `profile_plan.csv`);
2. confirms `profiles/` contains *exactly* the six canonical case
   directories in `profile_plan.csv` — no unplanned extra entry (directory,
   regular file, or symlink, dangling or not), none missing, none the wrong
   type (`verify_profiles_directory_inventory`; filesystem enumeration order
   is irrelevant) — before ever trusting `case_results` at all;
3. for each of the six cases, calls `reconstruct_case_result` — the exact
   same canonical function `validate-profile-case` itself calls to build the
   result it records — fresh from disk (the frozen plan row, the application
   CSV, the raw metrics CSV, the `.ncu-rep` hash, campaign provenance, the
   resolved-metric definition, and the fixed HBM-classification rule), and
   compares the reconstruction against what is currently recorded as a
   complete typed structure: every key (`case_name`, `method`, `stages`,
   `bytes_in_flight_kib`, `useful_bytes`, `launch_id`, `launch_count`,
   `dram_read_bytes`, `dram_read_ratio`, `hbm_classification`,
   `diagnostic_flags`, `resolved_metric_values`, `resolved_metric_units`, and
   the three artifact hashes), never a hand-picked subset — so a change to
   any single derived field, including one nothing else happens to depend
   on, can never go undetected merely because some other field still happens
   to agree.

Any mismatch — however it arose — fails the transition closed; nothing about
an earlier validation is ever trusted without re-verification.

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
