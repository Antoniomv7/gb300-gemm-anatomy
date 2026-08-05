# P2.3 frozen protocol -- BF16 UMMA throughput, joint 1-SM/2-SM sweep infrastructure

This document freezes the P2.3 contract: a deterministic, reproducible,
fail-closed runner and aggregator for the complete frozen Phase 2 matrix,
built entirely on top of the already-implemented, independently audited, and
GB300-verified P2.1 (`src/compute/umma_1sm.cu`) and P2.2
(`src/compute/umma_2sm.cu`) binaries. P2.3 introduces no CUDA kernel, no
change to either binary, and no change to either binary's SASS checker; it
only orchestrates, validates, and aggregates.

## 1. Scientific boundary: P2.3 versus P2.4

AGENTS.md experiment 2 asks for the fifth-generation Tensor Core throughput
ceiling and 2-SM scaling. P2.1 and P2.2 established functional correctness
of the two arms in isolation. P2.3 is the **joint sweep infrastructure**
that runs both arms, back to back, across the complete frozen matrix, and
produces the complete raw and validated evidence that P2.4 will use. P2.3
itself calculates and claims none of the following; every one of them is
explicitly out of scope here and remains P2.4 work:

* TFLOP/s
* an empirical Tensor Core ceiling
* 1-SM versus 2-SM speedup
* scaling efficiency
* saturation
* a winning configuration
* Nsight Compute conclusions
* any publishable performance result

Every row P2.3 can ever produce carries `publishable=false` unconditionally,
inherited unchanged from the audited P2.1/P2.2 CSV schema. `summary.csv`
(section 9) contains only descriptive statistics (count, mean, median,
sample standard deviation, coefficient of variation, minimum, maximum) of
raw per-sample fields already present in that schema -- never a ratio
between two configurations, a ranking, or a saturation classification.

| Unit | Scope | Status in this document |
|------|-------|--------------------------|
| P2.1 | 1-SM UMMA: single CTA, `cta_group::1`, M=128. | **Implemented, independently audited, and functionally verified on GB300** (see `src/compute/P2_PROTOCOL.md`). Unmodified by P2.3. |
| P2.2 | 2-SM UMMA: CTA pair, `cta_group::2`, M=256. | **Implemented, independently audited, and functionally verified on GB300** (see `src/compute/P2_2_PROTOCOL.md`). Unmodified by P2.3. |
| P2.3 | Joint 1-SM/2-SM sweep infrastructure, exactly 24 configurations. | **Implemented, independently audited, and functionally verified on GB300.** |
| P2.4 | Profiling and empirical ceiling: Nsight Compute, TFLOP/s and saturation analysis. | **Implemented, independently audited, and verified on GB300.** See `src/compute/P2_4_PROTOCOL.md`. |

## 2. The frozen 24-case matrix and its execution order

AGENTS.md caps experiment 2 at 24 configurations: `cta_group::1` (M=128,
`build/compute/umma_1sm`) and `cta_group::2` (M=256,
`build/compute/umma_2sm`), each with N in `{64,128,256}` and depth in
`{4,16,64,256}` -- 2 x 3 x 4 = 24. This is exactly P2.1's twelve
configurations plus exactly P2.2's twelve configurations; P2.3 adds no new
`(N, depth)` pair and no third method.

The twelve logical `(N, depth)` pairs, in the one frozen, canonical order
(`PAIR_ORDER` in `scripts/aggregate_exp02_umma_throughput.py`):

```
pair 0: (64, 4)      pair 1: (64, 16)     pair 2: (64, 64)     pair 3: (64, 256)
pair 4: (128, 4)     pair 5: (128, 16)    pair 6: (128, 64)    pair 7: (128, 256)
pair 8: (256, 4)     pair 9: (256, 16)    pair 10: (256, 64)   pair 11: (256, 256)
```

Within each pair, method order alternates to reduce systematic temporal
bias:

* even zero-based pair index: `umma_1sm`, then `umma_2sm`
* odd zero-based pair index: `umma_2sm`, then `umma_1sm`

This produces exactly 24 unique invocations, indexed `0..23`, generated from
one single canonical definition (`build_plan()`), never randomized, never
reordered, and independently re-derived and cross-checked by
`check_plan_contract()` on every `plan`/`--self-test`/`--print-plan`
invocation so a future edit cannot silently break the frozen contract. The
exact 24-row table (`scripts/run_exp02_umma_throughput.sh --print-plan`):

```
index  pair  method     cta_group  m    n    k   depth  binary                     case_name
    0     0  umma_1sm           1  128   64   16      4  build/compute/umma_1sm     00_umma_1sm_n64_d4
    1     0  umma_2sm           2  256   64   16      4  build/compute/umma_2sm     01_umma_2sm_n64_d4
    2     1  umma_2sm           2  256   64   16     16  build/compute/umma_2sm     02_umma_2sm_n64_d16
    3     1  umma_1sm           1  128   64   16     16  build/compute/umma_1sm     03_umma_1sm_n64_d16
    4     2  umma_1sm           1  128   64   16     64  build/compute/umma_1sm     04_umma_1sm_n64_d64
    5     2  umma_2sm           2  256   64   16     64  build/compute/umma_2sm     05_umma_2sm_n64_d64
    6     3  umma_2sm           2  256   64   16    256  build/compute/umma_2sm     06_umma_2sm_n64_d256
    7     3  umma_1sm           1  128   64   16    256  build/compute/umma_1sm     07_umma_1sm_n64_d256
    8     4  umma_1sm           1  128  128   16      4  build/compute/umma_1sm     08_umma_1sm_n128_d4
    9     4  umma_2sm           2  256  128   16      4  build/compute/umma_2sm     09_umma_2sm_n128_d4
   10     5  umma_2sm           2  256  128   16     16  build/compute/umma_2sm     10_umma_2sm_n128_d16
   11     5  umma_1sm           1  128  128   16     16  build/compute/umma_1sm     11_umma_1sm_n128_d16
   12     6  umma_1sm           1  128  128   16     64  build/compute/umma_1sm     12_umma_1sm_n128_d64
   13     6  umma_2sm           2  256  128   16     64  build/compute/umma_2sm     13_umma_2sm_n128_d64
   14     7  umma_2sm           2  256  128   16    256  build/compute/umma_2sm     14_umma_2sm_n128_d256
   15     7  umma_1sm           1  128  128   16    256  build/compute/umma_1sm     15_umma_1sm_n128_d256
   16     8  umma_1sm           1  128  256   16      4  build/compute/umma_1sm     16_umma_1sm_n256_d4
   17     8  umma_2sm           2  256  256   16      4  build/compute/umma_2sm     17_umma_2sm_n256_d4
   18     9  umma_2sm           2  256  256   16     16  build/compute/umma_2sm     18_umma_2sm_n256_d16
   19     9  umma_1sm           1  128  256   16     16  build/compute/umma_1sm     19_umma_1sm_n256_d16
   20    10  umma_1sm           1  128  256   16     64  build/compute/umma_1sm     20_umma_1sm_n256_d64
   21    10  umma_2sm           2  256  256   16     64  build/compute/umma_2sm     21_umma_2sm_n256_d64
   22    11  umma_2sm           2  256  256   16    256  build/compute/umma_2sm     22_umma_2sm_n256_d256
   23    11  umma_1sm           1  128  256   16    256  build/compute/umma_1sm     23_umma_1sm_n256_d256
total invocations: 24
```

`K=16` is constant throughout (implied by `.kind::f16` dense BF16, exactly as
documented in `src/compute/P2_PROTOCOL.md` section 7 and
`src/compute/P2_2_PROTOCOL.md` section 6; never a variable or a CLI option
here).

## 3. Runner CLI

```text
scripts/run_exp02_umma_throughput.sh --help
scripts/run_exp02_umma_throughput.sh --print-plan
scripts/run_exp02_umma_throughput.sh --self-test
scripts/run_exp02_umma_throughput.sh \
  --run-kind {smoke,benchmark} \
  [--campaign-id ID] \
  --iterations N \
  --warmup-iterations N \
  --repetitions N
```

* `--help`, `--print-plan`, and `--self-test` are standalone, mutually
  exclusive with each other and with every campaign option, and each may be
  given at most once. `--print-plan` and `--self-test` never require
  Docker, a GPU, `nvidia-smi`, or network access.
* Every other flag may be given at most once; unknown flags are rejected.
  All bounded integer arguments are validated before any Docker, GPU, or
  raw-results access.
* `--campaign-id` is optional (default: the current UTC timestamp,
  `YYYYMMDDTHHMMSSZ`); when given it must match
  `[A-Za-z0-9][A-Za-z0-9._-]{0,63}` and must not contain `..`.
* `--iterations` in `[1, 1000000]`, `--warmup-iterations` in
  `[0, 1000000]`, `--repetitions` in `[2, 1000000]` (audit repair: real P2.3
  campaigns require at least two repetitions per configuration, since
  `summary.csv`'s sample standard deviation and coefficient of variation are
  only statistically meaningful with two or more observations and must
  never be silently reported as zero for a single sample; `--repetitions 1`
  is rejected as a CLI/precondition error before any Docker, GPU, or
  raw-results access) -- forwarded verbatim to both binaries' own
  identically-named flags. The iterations/warmup-iterations bounds are
  chosen so that every existing signed 64-bit FLOP/UMMA formula
  (`flops_per_umma = 2*M*N*K`, `total_umma = depth*iterations`,
  `total_flops = flops_per_umma*total_umma`) stays far inside the int64
  range for every one of the 24 frozen configurations -- proved by
  `check_int64_safety()` in `scripts/aggregate_exp02_umma_throughput.py` and
  exercised by `--self-test`. At the worst case (M=256, N=256,
  depth=256, iterations=1000000), `total_flops` is about `5.4e14`, roughly
  four orders of magnitude below `INT64_MAX` (`9.22e18`).
* A real campaign additionally requires `BLACKWELL_GPU_INDEX` in the
  environment (never selected automatically), a clean Git worktree, and a
  full 40-character commit. Every GPU invocation goes exclusively through
  `scripts/run_container.sh`. The two methods never run concurrently: the
  24-case loop is a single strictly sequential `while read`, one
  `run_container.sh` invocation at a time.
* `set -Eeuo pipefail` throughout; `set -x` is never used.
* Exit codes: `0` success/`--help`/`--print-plan`/`--self-test`; `1`
  execution, correctness, or CSV-validation/aggregation failure; `2` CLI or
  precondition failure.

P2.3 reuses P2.1's and P2.2's audited command-line interfaces completely
unmodified: `--run-kind {smoke,benchmark} --n {64,128,256} --depth
{4,16,64,256} --iterations N --warmup-iterations N --repetitions N` (see
`src/compute/P2_PROTOCOL.md` section 16 and
`src/compute/P2_2_PROTOCOL.md` section 13). The runner's own
`--iterations`/`--warmup-iterations`/`--repetitions` map one-to-one onto
those binary flags; `--n`/`--depth` come from the frozen plan, never from
the runner's CLI.

## 4. Campaign lifecycle and raw layout

```text
results/raw/exp02_umma_throughput/<campaign-id>/
├── manifest.json
├── execution_order.csv
├── combined_samples.csv
├── summary.csv
├── cases/
│   └── <24 case CSV files>
└── logs/
    └── <self-test, launcher, and stderr logs>
```

A real campaign proceeds, fail-closed at every step:

1. Validate CLI and repository state (clean worktree, resolvable 40-character
   commit, explicit `BLACKWELL_GPU_INDEX`).
2. `init-campaign`: symlink-safe, no-clobber campaign directory creation,
   `execution_order.csv` written once, initial `IN_PROGRESS` manifest.
3. `make compute-umma-1sm-sass compute-umma-2sm-sass` (existing, unmodified
   P2.1/P2.2 GPU-free build/SASS gates).
4. Confirm both binaries and their `.sass` evidence files exist as regular
   files.
5. Both binaries' complete device `--self-test` runs, sequentially, each
   through `scripts/run_container.sh`.
6. Continue only if both self-tests pass; their `PASS`/`FAIL` outcome is
   recorded in the manifest and can never change afterward.
7. Execute the frozen 24-case plan, one process at a time.
8. Capture stdout as the case CSV (`capture`, no-clobber, symlink-safe);
   diagnostics/launcher output go to separate log files under `logs/`.
9. Strictly validate each case (`validate-case`) immediately after capture.
10. Record progress (`manifest-write`, `configuration_count_completed`/
    `sample_count_completed`) after every validated case.
11. `finalize`: re-validate the *entire* evidence set from scratch (manifest
    preconditions, `execution_order.csv`, every one of the 24 case files,
    cross-case consistency), then produce `combined_samples.csv` and
    `summary.csv` deterministically, no-clobber.
12. Mark the campaign `COMPLETE` only after that complete revalidation and
    after every mandatory hash (binaries, SASS, case files, execution order,
    aggregates) has been recorded.
13. On any error, signal (`INT`/`TERM`), or unexpected termination, the
    campaign is marked `FAILED`/`INTERRUPTED` with a recorded failure stage;
    partial evidence is never presented as `COMPLETE`.

Manifest state machine (`scripts/aggregate_exp02_umma_throughput.py`):
`None -> IN_PROGRESS -> {IN_PROGRESS, COMPLETE, FAILED, INTERRUPTED}`; every
one of `COMPLETE`/`FAILED`/`INTERRUPTED` is terminal and can never be
reopened or silently replaced. Only `finalize` may set `COMPLETE`. Every
manifest key is allowlisted with an exact required Python type
(`ALLOWED_MANIFEST_KEYS`); `campaign_id`, `run_kind`, `started_at_utc`,
`configuration_count_expected`, `sample_count_expected`, `requested`,
`selected_gpu_index`, `git_commit`, and `git_dirty` are immutable once set;
`configuration_count_completed`/`sample_count_completed` can never decrease.
`self_test_outcomes` starts as `PENDING` for both methods and follows the
per-method legal transitions `PENDING -> STARTED -> PASS/FAIL`, with
`PENDING -> NOT_RUN` only when the sibling fails first; terminal outcomes
cannot change. The manifest contains
only experiment-relevant, secret-free metadata: schema/experiment
identifiers, campaign ID, status, `publishable=false` (unconditional), the
frozen plan/configuration count (24), completed configuration/sample
counts, requested run parameters, the full Git commit and clean-state
evidence, pinned `VERSIONS.env` contents, allowlisted GPU/CUDA provenance,
the P2.1/P2.2 self-test outcomes, SHA-256 hashes of both binaries, both
`.sass` files, all 24 case files, `execution_order.csv`, and both final
aggregate artifacts, and failure/interruption information when applicable.
Adapted directly from the audited P1.3 campaign patterns
(`scripts/aggregate_exp01_memory_paths.py`): symlink-safe containment
(lstat-based, never following a symlink at any path component), no-clobber
publication (hard-link-then-unlink, never `os.replace()` except for the
manifest's own atomic update), progress recorded after every case, strict
CSV parsing, and deterministic aggregation. P2.3 does not copy P1.4's
Nsight Compute bridge or its profiling-specific manifest chain -- P2.3 never
invokes NCU.

`create_campaign_dir`/`resolve_campaign_dir` accept an optional `raw_root`
(defaulting to the real repository root for every production call site);
`--self-test` injects a `TemporaryDirectory` there so every symlink-escape
adversarial case is exercised without ever creating, following, or touching
anything under the real `results/raw/` tree.

## 5. Exact case-CSV validation

Both `build/compute/umma_1sm` and `build/compute/umma_2sm` emit the
identical, already-audited 37-column schema (`src/compute/P2_PROTOCOL.md`
section 14, `src/compute/P2_2_PROTOCOL.md` section 12):

```text
schema_version,timestamp_utc,run_kind,publishable,method,sample_index,cta_group,m,n,k,depth,iterations,warmup_iterations,repetitions,umma_per_iteration,total_umma,flops_per_umma,total_flops,elapsed_cycles,cycles_per_umma,flops_per_cycle,threads_per_cta,grid_blocks,tmem_columns,operand_path,input_type,accumulator_type,correctness,mismatches,max_abs_error,gpu_name,gpu_uuid,compute_capability,cuda_driver_version,cuda_runtime_version,git_commit,git_dirty
```

`validate_case_file()` requires the exact header (no reordering, no missing
or extra columns) and, for every one of the `repetitions` rows:

* Exactly `repetitions` rows; `sample_index=0..repetitions-1`, each exactly
  once.
* Exact requested `run_kind`, `iterations`, `warmup_iterations`,
  `repetitions`, and the case's own `n`/`depth`.
* `schema_version=1`; `publishable=` the literal string `false` (rejected
  for any other spelling, including `False`/`0`/`true`).
* `k=16`; `n` in `{64,128,256}`; `depth` in `{4,16,64,256}`.
* `umma_per_iteration=depth`.
* `total_umma = depth * iterations`.
* `flops_per_umma = 2 * m * n * 16`.
* `total_flops = flops_per_umma * total_umma`.
* `elapsed_cycles` is a positive canonical integer.
* `cycles_per_umma` and `flops_per_cycle` use the binary's canonical fixed
  six-decimal syntax, are finite and positive, and agree with the
  independently reconstructed `elapsed_cycles/total_umma` and
  `total_flops/elapsed_cycles` values within the half-unit-in-the-last-decimal
  interval implied by `std::fixed << std::setprecision(6)`
  (`FIXED6_HALF_ULP = 0.5e-6`).
* `threads_per_cta=128`; `tmem_columns=n`; `operand_path=smem_smem`;
  `input_type=bf16`; `accumulator_type=fp32`.
* `correctness=OK`; `mismatches=0`; `max_abs_error=0` (both printed, and
  required, as the literal integer `0`, not a six-decimal float -- the
  binaries' own `<< 0` stream insertion is unaffected by the row's
  `std::fixed`/`setprecision(6)` manipulators, which apply only to
  floating-point insertions).
* `compute_capability=10.3`.
* Non-empty, control-character-free `gpu_name`; `gpu_uuid` matching
  `GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`; positive
  `cuda_driver_version`/`cuda_runtime_version`.
* `git_commit` equal to the exact 40-character campaign commit;
  `git_dirty=false`.
* `timestamp_utc` a valid, parseable UTC timestamp
  (`YYYY-MM-DDTHH:MM:SSZ`).

Method-specific requirements (independently checked per row, never assumed
by analogy):

| Field         | `umma_1sm` | `umma_2sm` |
| ------------- | ---------: | ---------: |
| `cta_group`   |          1 |          2 |
| `m`           |        128 |        256 |
| `grid_blocks` |          1 |          2 |
| `method`      | `umma_1sm` | `umma_2sm` |

Across all repetitions and cases, `check_cross_case_consistency()` requires
identical `gpu_name`, `gpu_uuid`, `compute_capability`,
`cuda_driver_version`, `cuda_runtime_version`, `git_commit`, `git_dirty`,
`run_kind`, `iterations`, `warmup_iterations`, and `repetitions` -- compared
against one single reference row (the very first row of the very first
case), not merely each case's own first row, so a field that drifts only in
a later repetition is still caught.

`scan_case_directory()` requires the literal set of 24 filenames generated
by `build_plan()` and referenced by `execution_order.csv`; it does not parse
names back into integers. A spelling such as `n064` therefore cannot stand
in for canonical `n64`. It also rejects malformed CSV quoting/field counts,
missing columns, duplicate rows (duplicate `sample_index`), NaN/Inf numeric
fields, every wrong formula above, mismatched provenance, symlinked case
files, and every extra directory entry regardless of name, extension, or
type. Non-`.csv` artifacts such as salvaged `.invalid`/`.partial` capture
evidence prevent completion and are never aggregated.

## 6. Aggregated artifacts

`combined_samples.csv` is a deterministic, lossless consolidation of every
validated row, in execution-plan order, with increasing `sample_index`
within each invocation. Every row is prefixed with three stable
traceability fields and then carries every one of the original 37 fields
unchanged:

```text
invocation_index,pair_index,case_name,schema_version,timestamp_utc,run_kind,publishable,method,sample_index,cta_group,m,n,k,depth,iterations,warmup_iterations,repetitions,umma_per_iteration,total_umma,flops_per_umma,total_flops,elapsed_cycles,cycles_per_umma,flops_per_cycle,threads_per_cta,grid_blocks,tmem_columns,operand_path,input_type,accumulator_type,correctness,mismatches,max_abs_error,gpu_name,gpu_uuid,compute_capability,cuda_driver_version,cuda_runtime_version,git_commit,git_dirty
```

`summary.csv` contains exactly 24 rows, one per configuration, sorted by
`(n, depth, method)` (independent of the alternating execution order), with
purely descriptive statistics -- `sample_count`, mean, median, sample
standard deviation, coefficient of variation (percent), minimum, and
maximum -- for `elapsed_cycles`, `cycles_per_umma`, and `flops_per_cycle`
only. It never computes a pairwise speedup, a scaling ratio, TFLOP/s, a
ranking, a saturation classification, a confidence interval, outlier
removal, or any scientific conclusion; those are P2.4's work. Both files are
published no-clobber (`_open_exclusive` + hard-link-then-unlink) only after
all 24 cases have independently passed validation and cross-case
consistency.

## 7. Commands

GPU-free (no Docker GPU, no network; used to produce and validate this
implementation):

```bash
bash -n scripts/run_exp02_umma_throughput.sh
python3 -m py_compile scripts/aggregate_exp02_umma_throughput.py
scripts/run_exp02_umma_throughput.sh --help
scripts/run_exp02_umma_throughput.sh --print-plan
scripts/run_exp02_umma_throughput.sh --self-test
make check-static
make compute-umma-sweep-plan
make compute-umma-sweep-check
```

GB300 functional-verification commands (executed successfully on 3 August
2026 against Git commit `7a7cc2ab83197376720f030ba2e990092c3ada40`; see
section 8.3):

```bash
BLACKWELL_GPU_INDEX=<physical-index> make preflight
BLACKWELL_GPU_INDEX=<physical-index> make compute-umma-sweep-smoke
```

`compute-umma-sweep-smoke` requires an explicit `BLACKWELL_GPU_INDEX`, then
invokes the runner with `run_kind=smoke`, `iterations=20`,
`warmup_iterations=5`, `repetitions=3`. It is functional verification
infrastructure only; its cycle values are not publishable results. No
`compute-umma-sweep-benchmark` or P2.4 target exists.

## 8. Limitations and current status

* P2.3 is **implemented, independently audited, and functionally verified on
  GB300**. Section 8.3 records the exact implementation commit and campaign
  identifiers.
* `elapsed_cycles`, `cycles_per_umma`, and `flops_per_cycle` are exactly the
  raw, unconverted quantities P2.1/P2.2 already produce (see
  `src/compute/P2_PROTOCOL.md` section 17 and
  `src/compute/P2_2_PROTOCOL.md` section 17 for their own documented
  limitations, unchanged here): a raw `%clock64` delta, never wall-clock
  time, never corrected for clock throttling/boost state, and never a
  throughput or saturation claim.
  `summary.csv`'s descriptive statistics do not change this.
* P2.3 introduces no new correctness check beyond what P2.1/P2.2's own
  `--self-test` and per-repetition `correctness=OK`/`mismatches=0`/
  `max_abs_error=0` contract already guarantees; P2.3 only orchestrates and
  validates the already-audited evidence.
* P2.3 itself computes no TFLOP/s, empirical ceiling, 1-SM/2-SM speedup,
  scaling efficiency, saturation, or winning configuration. P2.4 owns that
  interpretation and is implemented, independently audited, and verified on
  GB300 by campaign `20260805T102759Z` (see
  `src/compute/P2_4_PROTOCOL.md`). Its empirical per-SM ceiling remains a
  `publishable=false` pilot candidate, not a final publishable result.

### 8.1 First independent audit and GPU-free repair (evidence integrity and truthfulness)

A first independent GPU-free audit of the initial implementation (commit
`bbd6371eb1c40357e60eb843acfabea3f00e1366`) found eight defects, all in
evidence integrity and campaign truthfulness rather than in the frozen
24-case matrix, the P2.1/P2.2 binaries, or either SASS checker (none of
which this repair touches):

1. **Malformed CSV quoting.** `read_case_rows()`/`validate_execution_order_file()`
   parsed campaign CSVs with Python's lenient (non-strict) `csv.reader`,
   which can silently reinterpret broken or unterminated quoting instead of
   rejecting it.
2. **Unenforced `cases/` inventory.** `scan_case_directory()` silently
   skipped any regular file whose name did not end in `.csv`
   (`.partial`/`.invalid` salvage evidence, `notes.txt`, hidden files, ...)
   instead of rejecting it, contrary to its own docstring.
3. **Symlinked execution artifacts.** The runner's pre-self-test artifact
   check used `test -f`, which follows symlinks, so a symlinked binary or
   SASS evidence file could pass before the first `scripts/run_container.sh`
   invocation.
4. **Untruthful self-test outcomes.** `self_test_outcomes` was written to
   the manifest only after *both* device self-tests passed; a failing or
   never-run self-test left the manifest with no outcome recorded at all.
5. **Absolute host paths.** Validation-error strings (built from absolute
   `Path` objects) could reach `failure_detail` and campaign logs verbatim,
   including this host's absolute repository path.
6. **`make compute-umma-sweep-smoke` prerequisite ordering.** Listing
   `compute-umma-1sm-sass compute-umma-2sm-sass` as Make prerequisites ran
   Docker/compilation/SASS checks before the recipe's own
   `BLACKWELL_GPU_INDEX` check.
7. **Repetitions floor.** `--repetitions 1` was accepted, under which
   `summary.csv`'s sample standard deviation and coefficient of variation
   are silently reported as `0.000000` rather than being statistically
   meaningless.
8. **`PLAN.md` presentation.** The Phase 3 heading stated "Gate: Phase 2
   gate passed." while P2.3 was unaudited/unverified and P2.4 was
   unimplemented.

All eight were remediated GPU-free, each with new adversarial self-test
coverage: strict-mode `csv.Error` regressions for both the case-CSV and
`execution_order.csv` readers; unexpected-`cases/`-entry regressions for
`notes.txt`, a `.partial` file, and a subdirectory, alongside the
pre-existing symlinked-case-file regression; a GPU-free synthetic proof that
a symlinked execution artifact is rejected before any external launcher
runs; manifest-lifecycle regressions for a first-method self-test failure, a
second-method failure, and both methods passing, using an explicit
PENDING/STARTED/PASS/FAIL/NOT_RUN per-method transition table
(`SELF_TEST_TRANSITIONS`) in place of the previous whole-dict immutability
rule; a synthetic-absolute-repository-path regression proving the redacted
root never reaches the manifest, `failure_detail`, or P2.3-generated logs;
and a `--repetitions 1`/`requested.repetitions=1` rejection regression at
both the runner CLI and the manifest-validation layer. No CUDA kernel, SASS
checker, or element of the frozen 24-case matrix changed. This repair does
not itself constitute an audit: P2.3 remains independently unaudited and
unverified on GB300, and P2.4 remains entirely unimplemented.

### 8.2 Focused re-audit repair (canonical identity and failure telemetry)

A focused re-audit of the first repair (commit
`6bea37cc7641f1de813ffe74806ce7dfdec0f1c5`) found that filenames were still
parsed semantically: replacing canonical `00_umma_1sm_n64_d4.csv` with
`00_umma_1sm_n064_d4.csv` could satisfy the configuration checks even though
`execution_order.csv` referenced a missing path. The finalizer now compares
the literal filename inventory with the canonical plan and has both a
direct scan regression and an end-to-end regression proving that this
campaign becomes `FAILED`, never `COMPLETE`.

The same focused repair initializes both manifest self-test outcomes as
`PENDING`, so failures before device self-tests truthfully show that neither
ran, and records the launcher's real nonzero status in `failure_exit_code`
instead of the previous constant `1`. No kernel, SASS checker, plan entry,
performance calculation, or P2.4 analysis changed. These repairs still
required independent audit and GB300 verification at that point.

### 8.3 Independent audit and GB300 functional verification

The final implementation at Git commit
`7a7cc2ab83197376720f030ba2e990092c3ada40` passed the independent audit.
The cluster reran the pinned `sm_103a` build and both real-cubin SASS gates,
all 127 synthetic aggregator regressions, the runner's CLI and artifact
self-tests, and the complete `make compute-umma-sweep-check` target; every
gate passed.

Functional verification then ran on 3 August 2026 on physical GPU index 4,
an NVIDIA B300 SXM6 AC with UUID
`GPU-4ae7e013-1aac-31d8-8b8e-c27530f1c6ed`. Fresh preflight campaign
`20260803T141347Z` reported `OVERALL=PASS`. Both complete device self-tests
passed sequentially, after which smoke campaign `20260803T141410Z`
executed and validated the frozen 24-invocation order with
`iterations=20`, `warmup_iterations=5`, and `repetitions=3`; the finalizer
revalidated the complete evidence set and recorded `status=COMPLETE`, and
the Make target returned `smoke_rc=0`.

This closes P2.3 as audited and GB300-verified infrastructure. It does not
create a publishable performance result: every row remains
`publishable=false`, and no cycle value from the smoke campaign is cited as
a Tensor Core ceiling, speedup, scaling, or saturation result. Those tasks
remain exclusively in P2.4, which is now implemented, independently audited,
and verified on GB300 (see `src/compute/P2_4_PROTOCOL.md`).

## 9. Status

```text
P2.3 = YES / YES / YES
```

That is: Implemented = **YES**; Independently audited = **YES**; Verified on
GB300 = **YES**. P2.1 and P2.2 remain `YES / YES / YES`, unchanged and
unaltered by this document. P2.4 is also `YES / YES / YES` (see
`src/compute/P2_4_PROTOCOL.md`). Phase 2 is **closed**.
