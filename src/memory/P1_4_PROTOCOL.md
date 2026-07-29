# P1.4 frozen protocol — profiling, HBM validation, analysis, pilot

**Status: implemented, remediated after FIVE independent GPU-free audits.
The final remediation passes the repository's GPU-free acceptance suite;
independent post-remediation sign-off remains pending. P1.4 Implemented:
YES — remediated. Independent audit: PENDING SIGN-OFF. Verified on GB300:
NO. Fresh preflight: PENDING. Pilot executed: NO. NCU/HBM validation: NO.
Publishable results: NONE. Phase 1: OPEN. No performance result exists yet.
GPU-free self-tests passing in this repository are not, and are never
described here as, an independent audit — only a reviewer who did not author
this remediation can close the "Independent audit" line above.**

## 0. Trust model (binding on every remediation in this document)

The campaign filesystem is trusted and single-writer. P1.4 protects against
accidental corruption, malformed or stale evidence, interrupted execution,
pre-existing unsafe paths, accidental overwrites, and ordinary recovery
failures. It does not claim to defend against a malicious concurrent process
running with the same filesystem permissions, or against deliberate path or
inode replacement after validation within one operation. Every
descriptor-anchored check, no-clobber publish, and ownership-checked cleanup
described below exists to make single-writer mistakes and interruptions
safe and auditable — not to withstand a hostile co-resident process
deliberately racing filesystem operations, holding every campaign descriptor
for an entire campaign, or replacing an inode between validation and use
within one operation. A future auditor should evaluate every claim in this
document against this scope, not against a general adversarial-filesystem
threat model.

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

A **third** independent GPU-free audit of that twice-remediated
implementation found five further blockers, each closed the same way (a new
adversarial test that first demonstrably failed against the pre-remediation
behavior, then passed): (A) the runner built
`profiles/<case>/<case>_report` relative to `/workspace` instead of the
campaign directory, and handed that (and a second, raw campaign `.ncu-rep`
path for metrics export) directly to NCU's own `-o`/`--log-file`/`--import`
arguments; (B) even a corrected path would still let NCU itself open a raw
campaign path for writing, structurally, regardless of whether the specific
path string were right; both closed by `scripts/p14_ncu_bridge.py`, a new
container-side bridge that runs NCU collection *and* metrics export entirely
inside the container's own private, non-host-mounted `/tmp` and hands the
host only a versioned, length-delimited bundle over its own stdout (Section
4a); (C) the evidence-integrity gate's field comparison used
`dict.get()`-based equality, under which `{}` and
`{"unexpected_evidence_field": null}` compared as identical — closed by a
strict recursive structural comparison that reports missing/unexpected keys
separately and requires `type(x) is type(y)` (Section 8); (D) manifest
fields were classified broadly (set-once/timestamp) but never bound to the
one specific transition legally allowed to introduce them, so e.g.
`resolved_ncu_metrics` could appear while `state=PILOT_IN_PROGRESS` — closed
by `validate_manifest_state_shape`, a second explicit validation layer run
alongside `validate_manifest_revision_transition` (Section 8a); (E) several
descriptor/helper defects in `scripts/p14_safe_capture.py` and the
profile-inventory/evidence-reading path: a filename such as `../escape.bin`
reached `os.link()`/`os.stat()` unvalidated, a launch failure left an
orphaned empty `.partial` file, and `profiles/`'s own inventory (and each
case's evidence reads) were still lstat-then-listdir/open-by-path instead of
descriptor-anchored — closed by strict single-component basename validation,
corrected failure-cleanup control flow, and extending the descriptor-
anchored discipline to profile inventory, evidence reads, and the manifest
revision directory itself (Sections 4a, 8).

A **fourth** independent GPU-free audit of that thrice-remediated
implementation found six further blockers (Groups A-F), each closed the
same way (a new adversarial test that first demonstrably failed against
`a66d0fa8b37147eb4f237911c42b02e3c8cbed59`, then passed): (A) the manifest
state-shape check tested `key in current and current[key] is not None`, so
a premature key holding an explicit `null` compared as absent, letting
revision 0 carry later-phase fields such as `profile_completed_at_utc:
null` unnoticed — closed by testing presence alone (`key in current`) and
by removing `type(None)` from every manifest field's declared type that had
allowed a premature null in the first place (Section 9); (B) lifecycle
timestamps were syntax- and immutability-checked but never compared against
each other, so e.g. a `profile_started_at_utc` earlier than
`pilot_completed_at_utc` passed unnoticed — closed by
`validate_manifest_timestamp_chronology`, one reusable validator enforcing
`started_at_utc <= pilot_completed_at_utc <= profile_started_at_utc <=
profile_completed_at_utc <= analyzed_at_utc` on every revision (Section 9);
(C) `profile_count_completed` and `case_results` could each appear without
the other, since their relationship was only checked once both already
existed — closed by an explicit co-occurrence check inside
`validate_manifest_state_shape` (Section 9); (D) `finalize-profile` and
`analyze` each performed the central evidence-integrity gate correctly but
then built `artifact_sha256` from values already sitting in memory before
the gate ran, and `analyze` never re-ran the gate a second time immediately
before publishing `ANALYZED` — closed by having
`verify_campaign_evidence_integrity` return the exact hashes/reconstructions
it just recomputed for direct use by the terminal manifest revision, and by
adding a second gate call in `analyze`, immediately before publication,
whose failure removes only the analysis artifacts just published and never
publishes `ANALYZED` (Section 8); (E) `publish_ncu_bundle()`'s rollback and
final bundle unlink removed any current regular file at a recorded name,
without checking it was still the same file this call itself had published
or opened — closed by `unlink_if_same_owned_inode`, a small dir_fd-anchored
helper that unlinks a name only while its `(st_dev, st_ino)` still matches
what was recorded at publish/open time (`scripts/p14_safe_capture.py`); (F)
`PLAN.md` stated "Gate: Phase 1 gate passed" as if already true — closed by
rephrasing it as an entry condition with an explicit current-status line,
and by adding this trust-model section (Section 0) to this document and to
`results/README.md`.

A **fifth** independent GPU-free audit of commit
`3d92a6b375ce3d0e803afd3e62723b08e471f3c8` found three final functional
blockers: (1) the runner wrote `failure_detail: null`, which the corrected
manifest schema rejects, and it could not record a signal received while a
profile campaign was still `PILOT_COMPLETE`; (2) the semantic loader did not
enforce an exact mutation set per transition, so profile progress could be
introduced on entry to profiling, while finalizing, or while failing; (3)
`COMPLETE` did not require the exact frozen `profile_order` and canonical
base-evidence `artifact_sha256` map, allowing a malformed `COMPLETE` revision
to be analyzed. These are closed respectively by typed runner failure
telemetry plus the real `PILOT_COMPLETE -> INTERRUPTED` edge, an exact
per-transition mutation matrix, and strict canonical `COMPLETE`/`ANALYZED`
terminal-content validation. Eight new full-chain adversarial regressions
first fail on `3d92a6b` and pass after this remediation.

None of these twenty-three fixes, across all five rounds, changed the frozen
pilot matrix, the six-case NCU plan, the statistical calculations, the
bootstrap seed/resample count, the outlier-retention policy, the saturation
rule, or the HBM thresholds below.

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
profiled benchmark kernel." NCU may print either that canonical identifier
or a namespace-qualified full identifier whose final component is exactly
that candidate (for example,
`FBSP.TriageCompute.dram__bytes_read.sum` on GB300). The discovery parser
reads the first whitespace-delimited table column and resolves a candidate
only on an exact whole-name match or an exact `.<candidate>` suffix. It
preserves the full identifier NCU reported and passes that same identifier
back to `--metrics`; if more than one qualified identifier matches a
candidate, discovery fails closed rather than guessing. No substring match,
architecture substitution, or inferred metric is accepted. If
`dram__bytes_read.sum` does not resolve under those rules, the whole six-case HBM
classification becomes `INCONCLUSIVE` (recorded explicitly in the manifest as
`resolved_ncu_metrics.dram_read_metric_available: false`); the other
resolved metrics (if any) are still collected and recorded, since they may
still be diagnostically useful, but no HBM claim is permitted. This is a
data-quality outcome, not a hard failure of the raw collection workflow —
`--profile` can still reach `COMPLETE` with `dram_read_metric_available:
false` recorded honestly.

### Collection and metrics export (per case, GPU-touching, entirely inside `scripts/p14_ncu_bridge.py`)

Both NCU invocations for a case — collection and the GPU-free `.ncu-rep`
metrics export — run inside a single container invocation
(`scripts/run_container.sh python3 scripts/p14_ncu_bridge.py`), and NCU is
never given a raw-campaign pathname for either one. This replaced an earlier
design (see the status note above, blockers A/B) that built
`profiles/<case>/<case>_report` relative to `/workspace` instead of the
campaign directory and handed that — and a second, separate `.ncu-rep` path
— directly to NCU's own `-o`/`--log-file`/`--import` arguments; a corrected
path string alone would not have fixed the underlying problem, since NCU
would still be opening a raw campaign path for writing itself.

```bash
scripts/run_container.sh python3 scripts/p14_ncu_bridge.py \
    --metrics <comma-joined resolved metrics> \
    --kernel-name <ldgsts_benchmark_kernel|tma_benchmark_kernel> \
    -- \
    build/memory/<ldgsts|tma> --stages <S> --bytes-in-flight-kib <B> \
        --run-kind benchmark --working-set-mib 512 --passes 32 \
        --warmup-ms 0 --repetitions 1
```

Inside the container, the bridge:

1. creates a private directory under the container's own, non-host-mounted
   `/tmp` (`scripts/run_container.sh` only ever bind-mounts the repository at
   `/workspace`; nothing under `/tmp` is shared with the host, and the
   container itself is destroyed on exit — `docker run --rm`);
2. runs the collection invocation with the same profiler controls as before
   (`--clock-control none`, `--pipeline-boost-state dynamic`,
   `--cache-control none`, `--kernel-name-base function`, `--launch-count 1`,
   `--devices 0`, `--replay-mode kernel`, `--print-summary none`), but with
   `-o`/`--log-file` pointed *only* inside that private directory, and the
   profiled binary's own inherited stdout/stderr captured to two more files
   in the same private directory;
3. runs the metrics-export invocation (`--import`, `--csv --page raw
   --print-metric-name name --print-units base --print-kernel-base
   function`) against the private `.ncu-rep`, likewise entirely inside the
   private directory;
4. verifies every output that must be non-empty (`.ncu-rep`, the NCU tool
   log, the application stdout, the exported metrics CSV) is a genuine
   non-symlink regular file;
5. emits exactly one versioned, length-delimited bundle — application
   stdout, application stderr, the NCU tool log, the raw `.ncu-rep` bytes,
   the exported `metrics_raw.csv`, and the metric-export step's stderr, in
   that fixed order — to its own stdout, and deletes the private directory
   before exiting;
6. on any collection or export failure, emits nothing to stdout at all (only
   diagnostics on stderr), so no bundle is ever produced from a partial or
   failed run.

The host side never talks to the bridge directly: `run_exp01_memory_paths_p14.sh`
captures the bridge's stdout through `scripts/p14_safe_capture.py run`
(an already-open, descriptor-anchored partial file, exactly like every other
P1.4 child-process capture — this is also where
`scripts/run_container.sh`'s own two allowlisted host-side banner lines
land, ahead of the bridge's real output) and then decodes/republishes it
through `scripts/p14_safe_capture.py publish-bundle` (Section 4a), which
splits the bundle back into its six named artifacts and publishes each one
into the anchored case directory, no-clobber. The bundle format is
length-prefixed (never delimiter-based), so arbitrary binary content (the
`.ncu-rep` bytes, or a metrics CSV containing any byte sequence) can never
be misparsed, and decoding tolerates the leading host-side banner bytes by
scanning for the format's own versioned magic marker.

### 4a. Descriptor-anchored safe capture and bundle publication (`scripts/p14_safe_capture.py`)

Every raw-campaign write `run_exp01_memory_paths_p14.sh --profile` performs —
the NCU-help-capability-probe log, both metric-discovery logs,
`discover-metrics`' own stderr log, each case's captured NCU-bridge bundle
and bridge stderr, and (after decoding) the six per-case artifacts the
bundle carries — goes exclusively through this P1.4-only module, never a
plain `>`/`>>`/`2>`/`2>>` shell redirection into
`results/raw/exp01_memory_paths_p14/` (mechanically confirmed: `rg -n
'(^|[[:space:]])[0-9]*>>?' scripts/run_exp01_memory_paths_p14.sh` and inspect
every match). A precheck immediately before an ordinary redirection — even a
symlink-aware one — still leaves a window between the check and the later
`open()`; this module closes that window structurally instead of narrowing
it:

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
* on a non-zero exit, *or any failure before or during the child's own
  launch* (a nonexistent executable, a failure creating the second of two
  outputs, or the self-test's own injectable pre-launch hook), a non-empty
  partial is preserved under its own unique name (never renamed or
  clobbered); an empty owned partial is removed after re-confirming its
  identity by descriptor, never by a second, racy path lookup — an earlier
  version of this function let a pre-launch failure propagate straight past
  this cleanup entirely, orphaning an empty partial file forever.

Every filename this module accepts from a caller (`--stdout-name`,
`--stderr-name`, `write`/`publish-bundle`'s `--name`/`--names`, and
`--bundle-name`) is validated as a strict single-component basename before
anything is created or any child is launched: empty, `.`, `..`, any name
containing `/` or NUL/control characters, and any absolute path are all
rejected outright. An earlier version accepted a caller-supplied name
containing `../`, which `os.link()`'s own `dst_dir_fd`-relative path
resolution then walked one level above the anchored directory exactly like
any other relative path with `..` components would — reproduced directly
against `publish_no_clobber()` during this remediation (a file was written
one level above the anchored `logs/` directory) and against
`run_capturing_outputs()`'s handling of an absolute `--stdout-name` (which
wrote outside the campaign entirely, since `linkat()`/`os.link()` ignore
`dst_dir_fd` altogether for an absolute `newpath`).

`--rel-dir` is restricted to an explicit allowlist (`logs`, or one of the six
frozen `profiles/<case>` names); `mkdir-case` (the safe replacement for the
old `[ -L case_dir ] || [ -e case_dir ]; mkdir case_dir` precheck-then-create
pair, which was itself racy against `profiles/` — the parent — being
swapped) creates exactly one of those six directories via `mkdirat()`'s own
`EEXIST`, never a separate check. `write` publishes already-extracted bytes
(the application CSV) the same no-clobber way. `publish-bundle` decodes an
already-captured `scripts/p14_ncu_bridge.py` bundle (itself captured via a
prior `run --stdout-name`) and republishes its six fixed-order segments
under caller-given names, no-clobber, then removes the raw transport bundle.
See the module's own docstring and `--self-test` (45 cases, including two
dedicated race reproductions — a symlink swap of `logs/` itself, and a swap
performed via an injectable hook after the directory descriptor is already
open but before the child writes — plus the traversal, launch-failure, and
bundle-format regressions above) for the complete design.

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
`logs/ncu_help_capability_probe.log`. The raw NCU-bridge bundle and the
bridge's own stderr log (`<case>.ncu_bridge_bundle.bin`,
`<case>.ncu_bridge_stderr.log`) are transient transport artifacts: the
bundle is deleted by `publish-bundle` once its six segments are
successfully republished under the names above, and only survives on disk
if publication itself failed partway (in which case it is preserved as
failure evidence, exactly like any other partial).

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
name, exactly like P1.3's capture step. `write_p14_manifest_status`, the
bash helper that records `FAILED`/`INTERRUPTED` outcomes, passes its merge
JSON to `manifest-write` through a system-default temporary file (plain
`mktemp`, no directory argument) rather than one created inside the
campaign path — an earlier version used `mktemp
"${CAMPAIGN_DIR}/manifest_merge.XXXXXX"` plus a shell redirect, both
operating on a campaign-relative pathname with no descriptor anchoring; the
merge file is a transient argument-passing mechanism between the shell and
`manifest-write` (which reads it once via a plain path and republishes the
*real* manifest revision through the descriptor-anchored, hash-chained,
no-clobber writer below), not campaign evidence, so moving it outside the
raw tree removes the concern entirely rather than narrowing it. The helper
always emits `failure_stage` plus `failure_detail: []` (a JSON list, never
`null`) and rejects a stage label outside its small internal basename-safe
alphabet before invoking `manifest-write`.

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
(`load_p14_manifest_chain`) opens `manifest/` exactly once, with
`O_DIRECTORY | O_NOFOLLOW` relative to the campaign directory, and re-opens
and re-validates *every* revision (each as a file descriptor opened
relative to that one held directory descriptor, never a second, independent
path resolution) from `000000.json` forward on every call — never trusting
anything about an earlier revision from memory — and rejects the whole
campaign as invalid if:
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

### Semantic manifest validation

The hash chain above proves a revision was appended without altering an
earlier byte; it says nothing about whether the new revision's *content* is
a legitimate continuation of the previous one, nor whether a field appeared
for the first time at the *wrong* state altogether. The schema/terminal
content validator and the following two complementary functions close this,
run together for revision 0 and every later revision alike:

* `validate_manifest_state_shape(current, expected_campaign_id)` — a pure
  function of *one* revision's own content, taken in isolation, with no
  knowledge of any earlier revision. Given `current["state"]` alone, both
  which fields must be present (already enforced by the schema validator's
  cumulative required-field gate) and which fields must still be *absent*
  are intrinsic properties of that one state. Before this function existed,
  nothing checked the second half: a manifest with `state=PILOT_IN_PROGRESS`
  that already carried `resolved_ncu_metrics` or `profile_completed_at_utc`
  passed unnoticed. It also checks `list(case_results)` against the frozen
  six-case order as an exact ordered list (never reduced to a set
  comparison — a *reordered* but set-identical `case_results` is otherwise
  indistinguishable from a correctly-ordered one) and that
  `profile_count_completed == len(case_results)`.
* `validate_manifest_revision_transition(previous, current,
  expected_campaign_id)` — checks that `current` is a legitimate
  continuation of `previous` (or, for revision 0, of the campaign directory
  itself): immutable fields unchanged, set-once fields unchanged once set,
  `case_results`/`artifact_sha256` append-only and never reordered/edited,
  `case_results` growing by exactly one entry per
  `PROFILE_IN_PROGRESS -> PROFILE_IN_PROGRESS` self-loop revision (neither a
  same-state no-op nor two cases appended at once), the state transition
  itself legal per `ALLOWED_P14_TRANSITIONS`, and the set of content fields
  changed by that revision exactly equal to the transition-specific
  allowlist below. This last condition is presence-aware and distinguishes
  an absent key from a present null-valued key.

`load_p14_manifest_chain` applies the schema, canonical terminal-content,
state-shape, transition, and timestamp-chronology checks to every revision
while walking the chain (never only to the latest revision), and
`write_next_p14_manifest_revision` applies the same checks once more,
defensively, immediately before writing. `expected_campaign_id` always comes
from the safely resolved campaign directory's own basename — a revision's
own stored `campaign_id` is never trusted by itself.

Every top-level P1.4 manifest field is classified into exactly one category
for `validate_manifest_revision_transition`'s purposes (a module-level
assertion fails at import time if any field is ever left unclassified):

| Category | Fields | Rule |
| --- | --- | --- |
| immutable | `schema_version`, `experiment_id`, `campaign_id`, `publishable`, `frozen_protocol`, `profile_plan_sha256` | present from revision 0 onward; the exact same value forever |
| allowed timestamp metadata | `started_at_utc`, `pilot_completed_at_utc`, `profile_started_at_utc`, `profile_completed_at_utc`, `analyzed_at_utc` | may be absent, then take a fixed value at its own specific transition; never changes once set |
| set-once | `pilot_campaign_reference`, `preflight_reference_pilot`, `provenance`, `preflight_reference_profile`, `resolved_ncu_metrics`, `profile_order` | may be absent, then take a fixed value at its own specific transition; never changes once set (non-timestamp analogue of the row above) |
| state-derived | `state`, `profile_count_completed` | governed by the state machine below (state) or monotonically non-decreasing (the count) |
| append-only | `case_results`, `artifact_sha256` | may only gain new entries; an existing entry is never edited, deleted, or reordered; `case_results` must also grow strictly in the frozen six-case order (its key set is always exactly a prefix of the frozen order) |
| (failure fields) | `failure_stage`, `failure_detail` | required together on a transition to `FAILED`/`INTERRUPTED`; absent everywhere else |

`validate_manifest_state_shape` independently classifies every field by
*which state may first introduce it* — a second, separate 7-row matrix (one
row per P1.4 state; FAILED/INTERRUPTED share the rule that they may retain
any subset of the fields introduced through `COMPLETE`, plus
`failure_stage`/`failure_detail`, since a campaign can fail at any point up
to — but never after — the terminal `ANALYZED` state): `None ->
PILOT_IN_PROGRESS` may introduce only `schema_version`, `experiment_id`,
`campaign_id`, `state`, `publishable`, `started_at_utc`, `frozen_protocol`,
`profile_plan_sha256`; `-> PILOT_COMPLETE` adds `pilot_completed_at_utc`,
`pilot_campaign_reference`, `preflight_reference_pilot`, `provenance`; `->
PROFILE_IN_PROGRESS` adds `profile_started_at_utc`, `resolved_ncu_metrics`,
`preflight_reference_profile`; later revisions in that same state may carry
`case_results`/`profile_count_completed`, but only after the first
`PROFILE_IN_PROGRESS -> PROFILE_IN_PROGRESS` self-loop introduces them; `->
COMPLETE` adds `profile_completed_at_utc`,
`profile_order`, `artifact_sha256`; `-> ANALYZED` adds `analyzed_at_utc`
(and permits `artifact_sha256` to gain `analysis/`-prefixed keys, which is
also independently checked by `validate_manifest_revision_transition`). This
is strictly more precise than the broad category table above: e.g.
`resolved_ncu_metrics` is "set-once" in that table (never *changes* once
set), but only the state-shape matrix asserts it cannot appear *at all*
before `PROFILE_IN_PROGRESS`.

The adjacent-revision mutation matrix is exact:

| Transition | Content fields that must change |
| --- | --- |
| `PILOT_IN_PROGRESS -> PILOT_COMPLETE` | `pilot_completed_at_utc`, `pilot_campaign_reference`, `preflight_reference_pilot`, `provenance` |
| `PILOT_COMPLETE -> PROFILE_IN_PROGRESS` | `profile_started_at_utc`, `resolved_ncu_metrics`, `preflight_reference_profile` |
| `PROFILE_IN_PROGRESS -> PROFILE_IN_PROGRESS` | `case_results`, `profile_count_completed` (exactly the next one-case prefix) |
| `PROFILE_IN_PROGRESS -> COMPLETE` | `profile_completed_at_utc`, `profile_order`, `artifact_sha256` |
| `COMPLETE -> ANALYZED` | `analyzed_at_utc`, `artifact_sha256` |
| any legal `-> FAILED/INTERRUPTED` | `failure_stage`, `failure_detail` only; all prior progress is preserved exactly |

`COMPLETE` additionally requires `profile_order == build_ncu_plan()` and an
`artifact_sha256` map whose keys and values exactly match the canonical
profile plan, both preflights, the three P1.3 pilot artifacts, and all three
evidence hashes for each of the six frozen profile cases. `ANALYZED`
preserves that base map and adds exactly the nine deterministic
`analysis/` artifacts, with no missing or extra key and a canonical SHA-256
for every value. Thus terminal metadata is validated as content, not merely
as an append-only dictionary.

`state` transitions (including a same-state "self-loop," which only
`PROFILE_IN_PROGRESS` has — once per validated case, appending exactly one
new `case_results` entry each time; `PILOT_IN_PROGRESS` has no self-loop,
since `--pilot` never reports incremental progress into the P1.4 manifest)
are legal only when `ALLOWED_P14_TRANSITIONS` itself lists the target as
reachable from the current state — there is no separate "same state is
always fine" exception, since most states (`PILOT_COMPLETE`, `COMPLETE`,
`ANALYZED`, `FAILED`, `INTERRUPTED`) have no self-loop at all and a repeat is
exactly as illegal as any other disallowed jump; since the table only ever
adds edges forward, this same check also rejects any state regression.
`analyzed_at_utc` and any `artifact_sha256` key beginning `analysis/` may
appear only once `state == "ANALYZED"`.

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

1. re-derives every non-`profiles/` artifact's path from the frozen NCU plan
   and canonical case names alone (never from a stored path string) and
   recomputes every SHA-256 fresh from disk (pilot/profile preflights, the
   P1.3 terminal manifest and its aggregate CSVs, this campaign's own
   `profile_plan.csv`);
2. opens the campaign directory and `profiles/` exactly once each, with
   `O_DIRECTORY | O_NOFOLLOW` relative to the previously opened descriptor,
   and confirms `profiles/` contains *exactly* the six canonical case
   directories in `profile_plan.csv` — no unplanned extra entry (directory,
   regular file, or symlink, dangling or not), none missing, none the wrong
   type — using that one held descriptor for both the listing
   (`os.listdir(profiles_fd)`) and every entry's type check
   (`os.stat(name, dir_fd=profiles_fd, follow_symlinks=False)`), never a
   separate lstat-then-listdir pair on the path string (an earlier version
   did exactly that, and a symlink swap of `profiles/` in the gap between
   the two calls produced `errors=[]`) — before ever trusting `case_results`
   at all;
3. for each of the six cases, opens that case's directory (again
   `O_DIRECTORY | O_NOFOLLOW`, relative to the still-held `profiles/`
   descriptor) and its three evidence files (`O_RDONLY | O_NOFOLLOW`,
   relative to the case descriptor), keeping every one of these descriptors
   open for the entire inventory-plus-evidence check; reads and hashes the
   evidence only from those exact file descriptors (never by reopening a
   pathname), and calls `reconstruct_case_result`'s descriptor-anchored
   entry point — built on the exact same canonical comparison core
   `validate-profile-case` itself uses — to rebuild the case's complete
   result fresh from those bytes;
4. compares the reconstruction against what is currently recorded with a
   strict recursive structural comparison, never `dict.get()`-based
   equality (which previously let `{}` and
   `{"unexpected_evidence_field": null}` compare as identical): exact key
   sets first (missing and unexpected keys reported as distinct
   conditions, so an absent key is never confused with one present and
   explicitly `null`), then exact list length/order, then exact scalar type
   (`type(x) is type(y)`, so `True` is never accepted in place of a
   canonical `1`, nor an int in place of a canonical float) and exact value,
   with NaN/infinity rejected outright on either side — covering every key
   (`case_name`, `method`, `stages`, `bytes_in_flight_kib`, `useful_bytes`,
   `launch_id`, `launch_count`, `dram_read_bytes`, `dram_read_ratio`,
   `hbm_classification`, `diagnostic_flags`, `resolved_metric_values`,
   `resolved_metric_units`, and the three artifact hashes), never a
   hand-picked subset, and recursing into `resolved_metric_values`/
   `resolved_metric_units` themselves rather than treating them as opaque
   blobs;
5. immediately before returning its verdict — which licenses the caller to
   publish a terminal manifest revision — re-confirms that `profiles/` and
   every case name this check just trusted still refer to the exact same
   `(device, inode)` they did when first opened, using the still-open
   descriptors; only then are all descriptors closed. A name-to-inode
   binding that changed during validation fails the check closed rather than
   silently trusting whatever the name now points to.

Any mismatch — however it arose — fails the transition closed; nothing about
an earlier validation is ever trusted without re-verification.
`validate-profile-case` itself re-derives its one case's evidence paths from
`--campaign-dir` and `--index` alone (the frozen plan's own case name), the
same way this gate does — it never trusts a caller-supplied
`--application-csv`/`--metrics-csv`/`--ncu-rep` path (an earlier version of
its CLI accepted, and used verbatim, exactly such an argument).

### State machine

```text
None              -> PILOT_IN_PROGRESS
PILOT_IN_PROGRESS -> PILOT_COMPLETE | FAILED | INTERRUPTED
PILOT_COMPLETE     -> PROFILE_IN_PROGRESS | FAILED | INTERRUPTED
PROFILE_IN_PROGRESS -> PROFILE_IN_PROGRESS | COMPLETE | FAILED | INTERRUPTED
COMPLETE           -> ANALYZED
ANALYZED           -> (terminal)
FAILED             -> (terminal)
INTERRUPTED        -> (terminal)
```

`PILOT_IN_PROGRESS` has no self-loop: unlike `PROFILE_IN_PROGRESS` (which
legitimately repeats once per validated case), `--pilot` never reports
incremental progress into the P1.4 manifest — `record-pilot` goes directly
from `PILOT_IN_PROGRESS` to `PILOT_COMPLETE` in one step, so a same-state
`PILOT_IN_PROGRESS` revision is not a transition the actual workflow ever
produces and is rejected like any other edge the table does not list.

`COMPLETE` means the raw pilot-plus-six-profile collection workflow finished
successfully — it never means the result is publishable (`publishable` is a
separate field, always `false`, at every state). `ANALYZED` means
`analysis/*` was generated from a `COMPLETE` campaign; it is still not
publishable. A terminal state (`FAILED`, `INTERRUPTED`, `ANALYZED`) is never
reopened or rewritten. `failure_stage`/`failure_detail` record where and why
a campaign stopped; both fields are required together, with
`failure_detail` always a JSON list of strings.

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
