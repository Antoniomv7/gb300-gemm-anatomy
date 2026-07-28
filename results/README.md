# results/

No publishable experimental performance results exist yet. Phase 0's
diagnostic preflight, and the P1.1/P1.2 GB300 self-test plus one-shot
`run_kind=smoke` runs, completed successfully as *functional* verification —
none of those smoke bandwidth values are experimental results.

## Future contents

### `results/preflight/<UTC timestamp>/` (raw, not committed)

Each preflight run (`BLACKWELL_GPU_INDEX=<i> make preflight`) will create one
directory named with a UTC timestamp in `YYYYMMDDTHHMMSSZ` format, containing:

- `summary.json` — machine-readable summary (`schema_version`,
  `timestamp_utc`, `git_commit`, `git_dirty`, `host_arch`, `tool_versions`,
  allowlisted `gpu` fields, per-check statuses, `overall_status`).
- Per-check logs (compiler output, smoke-test output, `ncu` output).
- The compiled smoke binary and the `.ncu-rep` profile.

`results/preflight/` is ignored by Git: raw and temporary output is never
committed.

### `results/raw/exp01_memory_paths/<campaign_id>/` (raw, not committed)

Each P1.3 sweep (`scripts/run_exp01_memory_paths.sh --run-kind ... --campaign-id ...`,
default campaign ID is the current UTC timestamp) creates exactly one
directory, once, via `aggregate_exp01_memory_paths.py`'s centralized
`init-campaign` subcommand, which walks every path component
(`results/`, `raw/`, `exp01_memory_paths/`, `<campaign_id>/`, `cases/`,
`logs/`) with `lstat` — never a `resolve()`/`is_dir()` check alone — refusing
a symlink (including a dangling one) at any level, including the raw root
itself, and refuses to overwrite an existing campaign directory:

```
manifest.json           # schema/status/provenance, see below
execution_order.csv     # the exact 18 deterministic invocation indices
cases/                  # one raw CSV per invocation, e.g. 00_ldgsts_s2_bif16.csv
logs/                   # one launcher log + one stderr log per invocation,
                         # plus the two full-binary self-tests
combined_samples.csv    # lossless union of all 18 raw cases, one header,
                         # exactly 18*repetitions rows, original 37-column
                         # schema (see src/memory/README.md), deterministic
                         # invocation order, increasing sample_index
summary.csv             # exactly 18 rows, one per configuration, ordered by
                         # (stages, bytes_in_flight_per_sm, method)
```

`manifest.json.status` is one of `IN_PROGRESS`, `COMPLETE`, `FAILED`, or
`INTERRUPTED`. A campaign is only ever `COMPLETE` once all 18 case files have
been strictly validated (37-column header and order; exact `repetitions`
rows with `sample_index=0..repetitions-1` each exactly once; `schema_version`,
`method`, `stages`, `run_kind`, `correctness=OK`, `mismatches=0`, and the
frozen occupancy/tile/vector constants; the stage/BIF/tile-height/useful-bytes
formulas; canonical fixed-six-decimal positive finite
`kernel_time_ms`/`effective_gbps`, with the latter inside the mathematical
interval implied by independent half-ULP rounding of both values;
`working_set_bytes > 2*l2_bytes` for `run_kind=benchmark`; the exact runner
Git commit with `git_dirty=false`; and, across the whole campaign, identical
`gpu_name`/`gpu_uuid`/`compute_capability`/driver+runtime versions/`git_commit`/
`git_dirty`/`sm_count`/`l2_bytes`/`working_set_bytes`/`passes`/`warmup_ms`/
`run_kind`/repetition count — deliberately excluding `smem_reservation_bytes`,
since TMA also reserves mbarrier storage; this comparison covers every
repetition of every case against one single validated reference row, not
just each case's first sample, so a value that only changes in a later
repetition is caught too). `execution_order.csv` must also independently
re-validate exactly (see `src/memory/README.md`), and all four build
artifacts (`build/memory/{ldgsts,tma}` and their `.sass` disassembly) must
exist as non-symlink, non-empty regular files with a real SHA-256 hash —
never `null`. On any invocation, validation, aggregation, signal, or I/O
failure the campaign is marked `FAILED` or `INTERRUPTED`, completed raw cases
and logs are preserved, and no `summary.csv` is produced. `manifest.json`'s
`status` follows an enforced state machine: the only legal transitions are
unset→`IN_PROGRESS`, `IN_PROGRESS`→`IN_PROGRESS`/`COMPLETE`/`FAILED`/
`INTERRUPTED`; a terminal campaign (`COMPLETE`, `FAILED`, or `INTERRUPTED`)
can never be reopened or rewritten, and only the validated `finalize`
subcommand — never the generic manifest-update path — may set `COMPLETE`.
Every field in the complete loaded manifest is allowlisted by name and type,
nested objects have exact schemas, immutable provenance cannot change,
progress counters cannot decrease, and an unrecognized field or a value of
the wrong type is rejected. Both self-test values must be `PASS`; the pinned
`VERSIONS.env` must be present, non-empty, non-symlink, and contain every
required key. Configuration/sample counters are updated after each validated
case, so a failure or interruption records actual progress.

`manifest.json` contains only safe, experiment-relevant, non-publishable
metadata: schema/experiment/campaign identifiers, status, requested and
observed common values, the exact invocation order, the selected physical GPU
index, allowlisted GPU/toolchain identity already reported by the binaries
themselves, the pinned `VERSIONS.env` contract, SHA-256 hashes of the
binaries/SASS/raw case files/`execution_order.csv`/aggregate files, self-test
outcomes, and `publishable: false`. It never stores full environment dumps,
usernames, home paths, SSH material, credentials, hostnames, process command
lines, or dynamic GPU telemetry (power/clock/temperature/utilization, Nsight
counters) — P1.3 "telemetry" means allowlisted provenance and execution
outcomes, not performance monitoring.

`summary.csv` is purely descriptive: arithmetic mean, median, sample standard
deviation (`n-1`, zero when `n=1`), and coefficient of variation
(`100*stdev/mean`) for `kernel_time_ms` and `effective_gbps`, plus
`effective_gbps_min`/`max`. It never filters outliers, computes confidence
intervals or significance, or compares LDGSTS against TMA — comparative
interpretation, speedups, and any outlier policy are P1.4. A `run_kind=smoke`
summary is functional/non-publishable by definition; a `run_kind=benchmark`
summary produced by P1.3 is still unreviewed raw input for P1.4, not a
publishable result, and does not by itself establish that the measured bytes
came from DRAM/HBM rather than L2 (that requires Nsight Compute, P1.4).

No result, log, or failure-evidence path is ever silently overwritten.
`combined_samples.csv`, `summary.csv`, `execution_order.csv`, and each
captured case `.csv` are all published with a hard-link-then-unlink
no-clobber operation (never `os.replace()`, which would overwrite): if the
final name already exists, publication fails outright rather than replacing
it. A failed or interrupted capture preserves any non-empty partial stdout
under a fresh `.invalid` or `.partial` name — never overwriting earlier
evidence — and a launch failure (e.g. an `OSError` starting the binary)
leaves no stale temporary file behind. Finalization checks both aggregate
targets and their temporaries before creating either output and removes only
its own new aggregate files if the final `COMPLETE` manifest update fails.
`manifest.json` is the sole intentional replacement lifecycle, but its
temporary is created exclusively with no symlink following.

To reproduce aggregation from an existing campaign's raw `cases/` directory
without rerunning any GPU work, see `scripts/aggregate_exp01_memory_paths.py`'s
`finalize` subcommand (invoked automatically by
`scripts/run_exp01_memory_paths.sh` at the end of a successful sweep).

`results/raw/` is ignored by Git: raw campaign output is never committed
automatically. P1.4 decides which small, curated, reviewed results (if any)
are suitable for publication under a future `results/` subdirectory.

## Safe public metadata

Anything stored here must contain only allowlisted device and tool data: GPU
index, name, UUID, driver version, compute capability, memory size, tool
versions, and check outcomes. Never store secrets, credentials, SSH material,
usernames, home paths, full environment dumps, or unrelated host metadata.

## Selected processed results (committed deliberately)

Small, curated, secret-free processed result files (e.g. per-experiment CSV or
JSON summary tables produced by later phases) may be committed under future
`results/` subdirectories so they remain publishable with the thesis. This is
always a deliberate, reviewed action — never an automatic copy of raw output.
CSV/JSON files are intentionally not blanket-ignored for this reason.

## Naming

All timestamps in file and directory names are UTC (`YYYYMMDDTHHMMSSZ`).
