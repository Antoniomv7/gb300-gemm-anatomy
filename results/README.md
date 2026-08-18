# results/

No publishable experimental performance results exist yet. Phase 0's
diagnostic preflight, and the P1.1/P1.2 GB300 self-test plus one-shot
`run_kind=smoke` runs, completed successfully as *functional* verification.
P1.3 joint smoke campaign `20260728T103315Z` also completed both full-binary
self-tests and all 18 planned invocations on GB300. None of those smoke
bandwidth values are experimental results. P1.4 pilot campaign
`20260730T073045Z` completed its frozen benchmark, six-case NCU validation,
analysis, integrity check, and independent review. It is reviewed pilot
evidence, but remains `publishable: false` and is not a final campaign.
P2.4 campaign `20260805T102759Z` completed its frozen 24-configuration
pilot, all 24 Nsight Compute profiles, deterministic analysis, evidence-
integrity checks, and independent review. It reached `ANALYZED` and produced
an empirical per-SM ceiling candidate of `16.37244853848296 TFLOP/s/SM`.
This closes P2.4 and Phase 2, but the campaign remains reviewed pilot
evidence with `publishable: false`, not a final campaign. See
`src/compute/P2_4_PROTOCOL.md` for the frozen contract and limitations.

P4.1 (the campaign orchestrator, `scripts/run_all.sh`) is implemented,
independently audited, and verified on GB300. Pilot campaign
`20260812T013848Z` completed all nine stages on 12 August 2026 and a subsequent
read-only `--resume` revalidated every stage without rewriting an artifact or
appending a manifest revision. The raw tree remains deliberately ignored and
uncommitted, and every artifact records `publishable=false`. P4.2 has since
completed and independently validated its three final campaigns, and P4.3's
offline integrated-analysis layer is implemented.
No publishable Phase 4 result exists.

P4.2's frozen protocol (`src/phase4/P4_2_PROTOCOL.md`) and its GPU-free
cross-campaign validator (`scripts/check_phase4_campaigns_p42.py`) are
implemented, independently audited, and verified on GB300. Pilot
`20260812T013848Z` remains accepted, untouched, and outside the replicate set.
Final campaigns `20260817T110330Z`, `20260817T111310Z`, and
`20260817T112011Z` each completed the full nine-stage plan from frozen commit
`b08e45c2636a3ac17c94ad8b1368084914196d7a` and passed terminal
revalidation without an artifact rewrite or a new manifest revision. The
strictly read-only population check accepted one pilot plus exactly those
three finals, their shared plan, commit, stage order, GPU identity, and
comparable provenance, and found no undeclared final. All four campaign trees
remain raw, ignored, and uncommitted; every artifact records
`publishable=false`. P4.2 is closed and no publishable Phase 4 result exists.

P4.3 is implemented (`scripts/analyze_phase4_p43.py`,
`scripts/check_phase4_integration_p43.py`, `src/phase4/P4_3_PROTOCOL.md`): an
offline, read-only analysis layer that revalidates the frozen four-campaign
population through P4.2's own checker, reads the canonical terminal P1.4, P2.4,
and P3.5 artifacts each final campaign's manifest pins, and aggregates across
the three finals with **one complete final campaign as the independent
replicate**. The accepted pilot `20260812T013848Z` is recorded only as excluded
qualification provenance and never enters a statistic. The implementation has
since received a remediation after a first independent audit — a corrected
scientific evidence taxonomy, preserved NCU and within-campaign stability
diagnostics, a truthful central metadata contract, descriptor-anchored output
containment, an immutable candidate-to-acceptance workflow, a verified
analysis-code commit, and corrected figure terminology — and **that remediation
is itself awaiting a new independent audit**. **The P4.3 independent audit has
not been performed**, no production analysis of the three final campaigns has
been run, and **no P4.3 curated result has been accepted for publication**. The
directory `results/phase4/` described below is therefore the declared
destination of that future analysis, not an existing result.

## Trust model

The campaign filesystem under `results/raw/` is trusted and single-writer.
P1.4's manifest chain, no-clobber publication, and evidence-integrity gates
(see `src/memory/P1_4_PROTOCOL.md`) protect against accidental corruption,
malformed or stale evidence, interrupted execution, pre-existing unsafe
paths, accidental overwrites, and ordinary recovery failures. They do not
claim to defend against a malicious concurrent process running with the
same filesystem permissions, or against deliberate path or inode
replacement after validation within one operation.

## Recorded P1.3 functional verification

```text
Git commit:               59777406b9454f00799c48bff8fa85cb03625cb6
Campaign:                 20260728T103315Z
Run kind:                 smoke
Configurations completed: 18
Samples completed:        36
Status:                   COMPLETE
Publishable:              false
```

The raw campaign is stored locally at
`results/raw/exp01_memory_paths/20260728T103315Z/` and is intentionally
ignored by Git. This record closes P1.3's functional GB300 verification only;
it must not be cited as a performance measurement or used to compare LDGSTS
against TMA. P1.4 owns the pilot benchmark campaign, Nsight Compute/HBM
validation, figures, and interpretation; it is implemented — and the
twenty-three blockers found by five independent GPU-free audits of that
implementation have all been remediated GPU-free (see
`src/memory/P1_4_PROTOCOL.md`). Its post-remediation review and GB300
verification are recorded below.

## Recorded P1.4 pilot and HBM validation

```text
Date:                     2026-07-30
Git commit:               e2d01b86f53177bd48d18b215be48b422dc3c53b
Preflight:                20260730T072946Z (OVERALL=PASS)
Campaign:                 20260730T073045Z
GPU:                      NVIDIA B300 SXM6 AC
P1.3 pilot:               COMPLETE (18 configurations, 540 samples)
P1.4 profiles:            COMPLETE (6/6)
P1.4 analysis:            ANALYZED (manifest revision 10)
HBM validation:           6/6 HBM_VALIDATED, no diagnostic flags
Technical closure:        PASS
Publishable:              false
```

The final validator reloaded the append-only manifest chain, verified every
analysis-artifact hash, checked the linked P1.3 campaign, and printed
`CIERRE TÉCNICO P1.4 / FASE 1: PASS`. The six NCU classifications apply only
to the six frozen profiled cases and are not extrapolated to the other
twelve configurations. The candidate-saturation result is limited to the
tested 16/32/64 KiB points, and the fixed non-randomized sweep order remains
a named limitation.

Independent review found one presentation-only erratum in this campaign's
immutable `analysis/report.md`: all 18 stability cells render as `REVIEW`
even though their CV values are 0.01–0.18%, below the strict `>5%` review
threshold. The CSV/JSON statistics path already carries the correct `ok`
labels; the error was caused only by truth-testing the non-empty string
`"ok"` while rendering Markdown. The closing code fix renders the exact
`ok`/`REVIEW` label and adds a regression test. It changes no samples,
statistics, profiler evidence, HBM classifications, kernels, or GPU
execution. The original raw campaign remains untouched and hash-valid.

## Recorded P2.4 pilot and empirical per-SM ceiling

```text
Date:                     2026-08-05
Git commit:               65f14d1069f0f04cb591ccdb9262c6222797042e
Profiling preflight:      20260805T102944Z (OVERALL=PASS)
Campaign:                 20260805T102759Z
GPU:                      NVIDIA B300 SXM6 AC (compute capability 10.3)
P2.3 pilot:               COMPLETE (24 configurations, 720 samples)
P2.4 profiles:            COMPLETE (24/24)
P2.4 analysis:            ANALYZED
SM-clock validation:      24/24 OK
Per-SM ceiling candidate: 16.37244853848296 TFLOP/s/SM
Best 1-SM case:           umma_1sm_m128n256k16_d256
Best 2-SM result:         16.220558567678513 TFLOP/s/SM
2-SM scaling efficiency:  99.16% (N=256, depth=256)
Device-wide estimate:     unavailable (optional SM-count metric unresolved)
P2.4 / Phase 2:           CLOSED
Publishable:              false
```

The campaign's append-only manifest reached `ANALYZED`; all 24 mandatory
SM-clock readings were valid, and independent recomputation matched the
recorded ceiling exactly. The device-wide extrapolation is intentionally
absent because NCU did not resolve
`device__attribute_multiprocessor_count`; the protocol makes that estimate
optional, so this does not invalidate the per-SM result. Scaling efficiency
of 104.83% at `N=64`, `depth=256` remains preserved and explicitly flagged
as surprising, but it did not determine the selected ceiling. The raw
campaign remains immutable and ignored by Git. This is one independently
reviewed pilot and must not be presented as a final architectural peak or a
publishable whole-GPU throughput result.

### `results/raw/exp01_memory_paths_p14/<campaign_id>/` (raw, not committed)

Each P1.4 campaign (`scripts/run_exp01_memory_paths_p14.sh --pilot` /
`--profile`, an explicit canonical UTC timestamp `--campaign-id`) creates
exactly one directory, symlink-safe like the P1.3 raw tree above, containing
`manifest/` (an append-only, hash-chained sequence of complete manifest
snapshots — `000000.json`, `000001.json`, ... — never a single mutable
`manifest.json`; each revision is published once, hard-link-then-unlink,
and never edited or replaced), `profile_plan.csv` (the frozen six-case
Nsight Compute plan, written once), `profiles/` (one subdirectory per
profiled case: its `.ncu-rep`, NCU tool log, captured container
stdout/stderr, extracted application CSV, and exported raw metrics CSV — NCU
itself never receives any of these as a pathname to open; a container-side
bridge, `scripts/p14_ncu_bridge.py`, stages every NCU output inside the
container's own private, non-host-mounted `/tmp` and hands the host a single
versioned bundle over its own stdout, which the host decodes and publishes
into this directory), `logs/` (metric-discovery and
NCU-help-capability-probe logs), and, once `make memory-paths-p14-analyze`
has run against a `COMPLETE` campaign, `analysis/` (deterministic
statistics/comparison/saturation/HBM-validation CSVs, `analysis.json`,
`report.md`, and three SVG figures under `analysis/figures/`). The
manifest's `state` field follows its own state machine
(`PILOT_IN_PROGRESS`→`PILOT_COMPLETE`→`PROFILE_IN_PROGRESS`→`COMPLETE`→`ANALYZED`,
or `FAILED`/`INTERRUPTED`; `PILOT_IN_PROGRESS` has no self-loop, since
`--pilot` never reports incremental progress into this manifest);
`COMPLETE` means the raw pilot-plus-profile collection succeeded, not that
the result is publishable — `publishable` is always `false`. Every manifest
revision is validated cryptographically (the hash chain) and through two
explicit semantic layers: a state-shape check (every field is bound to the
one specific state that may first introduce it — not just a broad
immutable/set-once/append-only/state-derived/timestamp classification, and
`case_results`'s key order is checked as an exact list, never reduced to a
set comparison) and a transition check (a correctly-hashed revision that
changes an immutable field, edits an earlier profiled case's result, or
appends more than one case at once is still rejected). The transition check
also enforces an exact per-edge mutation set: profile cases appear only one
at a time on `PROFILE_IN_PROGRESS` self-loops, failure/interruption preserves
the exact recorded prefix, and finalization cannot introduce its sixth case.
`COMPLETE`/`ANALYZED` require the frozen profile order and exact canonical
artifact-hash inventories. Before both
`COMPLETE` and `ANALYZED`, the campaign directory, `profiles/`, and each
case directory are opened once with descriptor-anchored, no-follow
resolution and held open for the whole check; `profiles/` is confirmed to
contain exactly the six canonical case directories the frozen plan expects,
every trusted artifact recorded in the manifest is re-hashed from disk, and
every profiled case's complete recorded result is compared — via a strict
recursive structural comparison, never `dict.get()`-based equality — against
what its raw evidence alone reconstructs, so a validated artifact, or any
single recorded derived value (or an unexpected extra field, however it was
set), modified after the fact is rejected rather than silently accepted.
This raw tree is covered by the same blanket `results/raw/` Git-ignore rule
as P1.3's own raw tree.

## Storage layout

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

### `results/raw/exp02_umma_throughput_p24/<campaign_id>/` (raw, not committed)

Each P2.4 campaign (`scripts/run_exp02_umma_throughput_p24.sh --pilot` /
`--profile`, an explicit canonical UTC timestamp `--campaign-id` shared with
the P2.3 campaign it drives) creates exactly one directory, symlink-safe
like every other raw tree in this repository, containing `manifest/` (an
append-only, hash-chained sequence of complete manifest snapshots, never a
single mutable `manifest.json`), `profile_plan.csv` (the frozen 24-case
Nsight Compute plan -- the same 24 configurations as P2.3's own plan, plus
each case's exact kernel symbol -- written once), `profiles/` (one
subdirectory per profiled case: its `.ncu-rep`, NCU tool log, captured
container stdout/stderr, extracted application CSV, and exported raw
metrics CSV -- NCU itself never receives any of these as a pathname to
open; a container-side bridge, `scripts/p24_ncu_bridge.py`, stages every
NCU output inside the container's own private, non-host-mounted `/tmp` and
hands the host a single versioned bundle over its own stdout, decoded and
published by `scripts/p24_safe_capture.py`), `logs/` (metric-discovery and
NCU-help-capability-probe logs), and, once `make compute-umma-p24-analyze`
has run against a `COMPLETE` campaign, `analysis/` (deterministic
configuration-statistics/scaling/saturation/profile-validation CSVs,
`empirical_ceiling.json`, `report.md`, three SVG figures, and
`analysis_manifest.json`). The manifest's `state` field follows
`PILOT_IN_PROGRESS`→`PILOT_COMPLETE`→`PROFILE_IN_PROGRESS`→`COMPLETE`→
(`ANALYZED` or `INCONCLUSIVE`), or terminal `FAILED`/`INTERRUPTED`.
`COMPLETE` means the raw pilot-plus-profile collection succeeded, not that
the result is publishable -- `publishable` is always `false`. `ANALYZED`
means every one of the 24 profiled configurations' mandatory SM-clock
reading was trustworthy and a full TFLOP/s/empirical-ceiling analysis was
produced; `INCONCLUSIVE` means the raw evidence and every clock-independent
statistic were still produced, but at least one configuration's SM-clock
reading could not be trusted, so no TFLOP/s or completed empirical-ceiling
claim was ever emitted. Before both `COMPLETE` and publishing
`ANALYZED`/`INCONCLUSIVE`, the campaign directory, `profiles/`, and each
case directory are opened with descriptor-anchored, no-follow resolution
and held open for the whole check; every trusted artifact recorded in the
manifest is re-hashed from disk, and every profiled case's complete
recorded result is compared -- via strict recursive structural comparison,
never `dict.get()`-based equality -- against what its raw evidence alone
reconstructs. This raw tree is covered by the same blanket `results/raw/`
Git-ignore rule as every other campaign tree in this repository. Campaign
`20260805T102759Z` executed this complete path on GB300 and reached
`ANALYZED`; its closure record appears above. See
`src/compute/P2_4_PROTOCOL.md` for the complete frozen contract.

### `results/raw/phase4/<campaign_id>/` (raw, not committed)

Pilot campaign `20260812T013848Z` and final campaigns `20260817T110330Z`,
`20260817T111310Z`, and `20260817T112011Z` each created one such tree in the
operator's cluster checkout and reached terminal state `COMPLETE`. All four are
intentionally absent from Git under the same blanket `results/raw/` ignore rule
as every other raw campaign tree. P4.1 (`scripts/run_all.sh`, see
`src/phase4/P4_1_PROTOCOL.md`) uses the following layout for each top-level
campaign:

```text
results/raw/phase4/<campaign_id>/
├── manifest/            # append-only, hash-chained revisions: 000000.json, ...
├── plan.json            # the immutable deterministic stage plan, written once
├── logs/                # one no-clobber log set per stage/attempt
└── exp03/
    └── gemm_comparison.csv   # the accepted P3.5 capture (21 lines, 20 rows)
```

The three experiments keep their own raw trees; P4.1 does **not** copy them,
and instead records validated repository-relative references plus SHA-256
hashes. For a P1.4 or P2.4 stage that reference **pins** the exact manifest
revision the campaign accepted — its repository-relative path, its revision
number, and its SHA-256 — and, once that unit is terminal, a digest over the
snapshot the unit's own `verify_campaign_evidence_integrity()` recomputed fresh
from disk together with fresh hashes of every canonical terminal `analysis/`
artifact. A later terminal revision, a changed revision, or changed raw or
derived evidence is therefore rejected on revalidation rather than adopted
silently.
All five trees share one explicit campaign ID:

```text
results/raw/phase4/<id>/
results/raw/exp01_memory_paths/<id>/
results/raw/exp01_memory_paths_p14/<id>/
results/raw/exp02_umma_throughput/<id>/
results/raw/exp02_umma_throughput_p24/<id>/
```

The top-level manifest is never a single mutable file: each transition appends
one complete, immutable snapshot whose `previous_manifest_sha256` is the
freshly recomputed hash of the preceding revision, and loading re-opens,
re-hashes, and revalidates every revision from `000000.json` forward on every
call. `os.replace()` is never used. Its `state` field follows
`IN_PROGRESS` → `COMPLETE` / `INCONCLUSIVE` / `FAILED` / `INTERRUPTED`, with
`FAILED` and `INTERRUPTED` reopenable only by an explicit `--resume` and only
for the two stages that own no persistent per-experiment state
(`preflight`, `gemm.capture`). `COMPLETE` means every selected component
independently passed its own existing validator **and** the final top-level
integrity gate; it never means the result is publishable — `publishable` is
always `false`. A P2.4 `INCONCLUSIVE` analysis propagates to a non-complete
top-level outcome and is never accepted as a complete campaign.

The manifest records only allowlisted information: schema version, the
campaign ID and its immutable kind (`pilot`/`final`), the immutable selected
scope (`full`/`memory`/`umma`/`gemm`), the current clean Git commit, the stage
order and per-stage status, repository-relative evidence paths, SHA-256
hashes, allowlisted GPU identity (UUID, name, compute capability, driver
version) taken from validated evidence, the validated preflight reference,
timestamps, any failure or interruption stage, and `publishable=false`. It
never stores usernames, home paths, host names, full environment dumps,
credentials or tokens, SSH material, unrelated process information, complete
host command lines, or dynamic power, clock, temperature, or utilization
telemetry; a structural privacy gate rejects any offending field name at any
nesting depth and any absolute path value. The campaign logs are held to the
same standard: every experimental Make target is invoked with
`--silent --no-print-directory` so no echoed recipe line can carry the absolute
bind-mount source of the checkout, and durable textual logs and failure details
replace this checkout's exact root with the stable token `<repo-root>`. Ordinary
child diagnostics remain unchanged; P3.5's scientific CSV stdout is copied byte
for byte and no scientific content is rewritten.

### `results/phase4/` (curated P4.3 output — declared, not yet produced)

This is the only tree P4.3 may write, and it is deliberately **not** under
`results/raw/`: the analyzer refuses any output root under `results/raw/` or
`results/preflight/`, so raw evidence stays immutable. The frozen inventory is
exactly:

```text
results/phase4/
├── memory_paths.csv
├── umma_throughput.csv
├── gemm_comparison.csv
├── integrated_summary.json
├── report.md
├── analysis_manifest.json
└── figures/
    ├── memory_paths.svg
    ├── umma_throughput.svg
    └── gemm_comparison.svg
```

All nine artifacts record schema version `p43.v1` and are byte-deterministic:
row order, JSON key order, decimal formatting, Markdown, and SVG bytes are
reproducible, and two runs over identical evidence are byte-identical. Beyond
that, **metadata ownership is central, not duplicated**:

* `analysis_manifest.json` is the **authoritative binding** for the whole
  bundle. It alone records the three ordered final campaign IDs and their
  explicit mapping to the `campaign_1_value` / `campaign_2_value` /
  `campaign_3_value` columns, the pilot ID as excluded qualification provenance,
  the final execution commit, the P4.3 analysis-code commit and its
  clean-worktree verification, the common GPU identity, the comparable
  provenance, the exact repository-relative source paths and SHA-256 hashes of
  everything read, the candidate publication state, and an exact SHA-256 for
  each of the other eight artifacts.
* It **cannot contain its own byte hash** — writing the value would change the
  bytes it describes. It says so explicitly in a `self_hash` object.
  `make phase4-p43-verify` recomputes every byte of it, and the later external
  acceptance attestation records `analysis_manifest_sha256`, which covers all
  nine artifacts without any self-reference.
* The three CSV files carry their own schema, key and data fields, the
  `evidence_class` of every row, and the three campaign-level values in the
  frozen campaign order. The three SVG files are deterministic visual artifacts.
  Neither duplicates the global provenance.
* `integrated_summary.json` and `report.md` carry the scientific context needed
  to interpret the bundle.
* **A detached CSV or SVG is not a standalone provenance envelope.** These files
  must be distributed together with `analysis_manifest.json`; on their own they
  carry no campaign identity, no commit, and no publication state.

Production output is limited to the single logical destination
`<repo-root>/results/phase4`, reached by opening `results`, `phase4`, and
`figures` component by component from an already opened repository descriptor
with `O_DIRECTORY | O_NOFOLLOW`; a symlink at any level, including an ancestor,
is rejected before a byte is written. Publication is no-clobber: an existing
byte-identical artifact is verified rather than rewritten, an existing different
artifact is fatal, and a symlink, an unexpected file type, or any artifact
outside this inventory fails closed.

Every artifact is a **candidate**, recording `publishable=false` and
`publication_state=candidate_pending_independent_output_review`. Nothing
promotes, overwrites, or deletes a candidate: acceptance is a separate external
attestation at `src/phase4/P4_3_ACCEPTANCE.json`, written only after an
independent review of the complete bundle. That file does not exist, and the
P4.3 checker fails if it does.

**This tree does not exist yet.** No production P4.3 analysis has been run, and
no P4.3 curated result has been accepted for publication.

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
