# P4.1 — Phase 4 campaign orchestrator (frozen protocol)

Status: `P4.1 = YES / YES / YES` (Implemented / Audited / Verified on GB300).

* **P4.1 is implemented infrastructure.**
* **Independent audit: PASSED.**
* **GB300 verification: PASSED.**
* **Pilot campaign `20260812T013848Z`: COMPLETE.** P4.2 later completed its
  three separate final campaigns without changing the P4.1 execution path.
* **No publishable result exists**, and none is claimed. Every artifact this
  unit can ever write records `publishable=false`.
* **P4.2 is now closed as `YES / YES / YES`.** Its three final campaigns and
  read-only cross-campaign validation passed after this P4.1 closure.
* **P4.3 will perform integrated analysis and publication review**, the final
  tables and figures, and the closing audit.

The GPU-free checks in section 12 were run by the author. **They are the
author's own self-checks; they are not an independent audit, and GPU-free
checks are not GB300 verification.**

## 0. Trust model (mirrors `src/memory/P1_4_PROTOCOL.md` section 0)

The campaign filesystem under `results/raw/phase4/` is trusted and
single-writer, exactly like every other raw campaign tree in this repository.
P4.1's manifest chain, descriptor-anchored opens, no-clobber publication, and
re-hashing gates protect against accidental corruption, malformed or stale
evidence, interrupted execution, pre-existing unsafe paths, accidental
overwrites, and ordinary recovery failures. They do not claim to defend
against a malicious concurrent process running with the same filesystem
permissions, or against deliberate path or inode replacement after validation
within one operation. A future auditor should evaluate every claim in this
document against that scope.

## 1. Purpose and scope

Phases 1–3 closed three independent experimental units. P4.1 adds the single
public entry point that coordinates one reproducible top-level campaign across
all three:

| # | Experiment | Composed unit | Entry points P4.1 drives |
|---|------------|---------------|--------------------------|
| 1 | LDGSTS versus TMA | P1.4 | `make memory-paths-p14-pilot` / `-profile` / `-analyze` |
| 2 | BF16 UMMA throughput | P2.4 | `make compute-umma-p24-pilot` / `-profile` / `-analyze` |
| 3 | CuTe DSL BF16 GEMM versus cuBLASLt | P3.5 | `make gemm-comparison-p35-smoke` |

P4.1 **composes**; it never reimplements. It adds no CUDA, CuTe DSL, or
cuBLASLt code, no shape, candidate, layout, dtype, tile, cluster, or
algorithm, no autotuning, no Nsight Compute case, no profiler route, no
statistic, no threshold, no figure, and no interpretation. It does not modify
any kernel, scientific matrix, statistical rule, profiler plan, correctness
rule, schema, or execution parameter of the units it drives.

**P4.1 executes nothing by itself.** Implementing this unit ran no pilot, no
final campaign, and no GPU command; it produced no experimental result and no
publishable artifact.

## 2. Public interface

`scripts/run_all.sh` is the **only** public Phase 4 orchestration entry point.
`scripts/phase4_orchestrator.py` is a private helper that owns deterministic
planning, manifest validation, safe capture, hashing, and resume logic;
`scripts/check_phase4_orchestrator_p41.py` is the fail-closed checker.

```text
scripts/run_all.sh --help
scripts/run_all.sh --self-test

scripts/run_all.sh \
  --dry-run \
  --campaign-id <YYYYMMDDTHHMMSSZ> \
  --campaign-kind <pilot|final> \
  [--only <memory|umma|gemm>]

BLACKWELL_GPU_INDEX=<physical-index> scripts/run_all.sh \
  --campaign-id <YYYYMMDDTHHMMSSZ> \
  --campaign-kind <pilot|final> \
  [--only <memory|umma|gemm>] \
  [--resume]
```

Rules, all enforced and all covered by acceptance tests:

* `--campaign-id` is **required** and is an explicit canonical UTC timestamp
  matching exactly `YYYYMMDDTHHMMSSZ` that is also a real calendar instant. The
  orchestrator **never** generates a campaign ID implicitly, at any level.
* `--campaign-kind` is **required** and **immutable**. It records whether P4.2
  is using the invocation as its pilot or as one of its final replicates.
  **`campaign-kind=final` does not make any raw evidence publishable**; it is a
  label for P4.2's own bookkeeping and nothing else.
* `--only` accepts exactly one of `memory`, `umma`, or `gemm`. An invocation
  without `--only` uses the complete three-experiment plan. The selected scope
  is immutable for the life of the campaign.
* `--resume` requires an existing matching top-level campaign; an existing
  campaign without `--resume` is rejected, and `--resume` for a missing
  campaign is rejected.
* Duplicate options, missing values, unexpected positional arguments, and
  unknown options fail with exit code **2**.
* `--help` and `--self-test` are standalone: neither accepts any other option.
* `--dry-run` may be combined with the campaign and selector options and
  performs no mutation. It is rejected together with `--resume`, which has no
  meaning without execution.

`--help`, `--self-test`, and a new-campaign `--dry-run` are genuinely GPU-free:
no Docker, no `nvidia-smi`, no CUDA/PyTorch/CuTe DSL import, no network, no
results directory creation, no manifest or log write, and no Git mutation. The
`--dry-run` path does not invoke Git at all.

### 2.1 Why there is no `--only cublaslt` and no `--only cutedsl`

Experiment 3 is **one atomic comparison**. P3.5 runs five frozen shapes × four
candidates — `cutedsl/nonpersistent_1cta`, `cutedsl/persistent_1cta`,
`cutedsl/persistent_2cta`, and `cublaslt/heuristic_first_supported` — over
shared, immutable operands and one untimed oracle per shape, in shape-major
order, with all-or-nothing output under one common comparison contract.
Splitting it would break exactly those guarantees. A separate `--only cublaslt`
or `--only cutedsl` mode is therefore **forbidden**; the supported selector is
`--only gemm`, and `scripts/run_all.sh` rejects the two obsolete spellings by
name with an explicit diagnostic.

### 2.2 Exit codes

```text
0    success, --help, --self-test, --dry-run, or a pure revalidation of an
     already COMPLETE campaign
1    a stage or validation failure
2    a CLI, repository-state, or safety-precondition failure
4    a terminal but non-complete INCONCLUSIVE campaign outcome
130  interrupted (the conventional SIGINT status); every completed stage's
     evidence is preserved and the interrupted stage stays eligible for a new
     attempt on --resume
```

A failing child command's **exact** exit status is preserved as evidence: it is
recorded verbatim in the manifest attempt record and printed on stderr. The
orchestrator's own process exit code stays inside the fixed contract above so
that a child's status can never be confused with `2` (a precondition failure)
or `4` (an inconclusive outcome).

## 3. The exact stage plans

Full campaign (no `--only`):

```text
1. preflight
2. memory.pilot
3. memory.profile
4. umma.pilot
5. umma.profile
6. gemm.capture
7. memory.analyze
8. umma.analyze
9. campaign.validate
```

`--only memory`:

```text
1. preflight   2. memory.pilot   3. memory.profile   4. memory.analyze   5. campaign.validate
```

`--only umma`:

```text
1. preflight   2. umma.pilot   3. umma.profile   4. umma.analyze   5. campaign.validate
```

`--only gemm`:

```text
1. preflight   2. gemm.capture   3. campaign.validate
```

Both experiments collect **all** of their GPU evidence before either GPU-free
analyzer runs, so a campaign holds the selected shared GPU for the shortest
contiguous span and no analysis interleaves with GPU work.

The analysis stages mean exactly: invoke the existing P1.4 and P2.4 analyzers,
and validate the P3.5 capture. **They do not mean integrated analysis across
experiments.** Integrated tables, scientific interpretation, publication
decisions, and final figures belong exclusively to P4.3.

`--dry-run` prints this plan deterministically — the same bytes on every
invocation for the same arguments — and exits 0 without executing it.

### 3.1 Stage-to-target mapping

| Stage | Kind | Delegated to |
|-------|------|--------------|
| `preflight` | gpu | `BLACKWELL_GPU_INDEX=<i> make preflight` |
| `memory.pilot` | gpu | `make memory-paths-p14-pilot` |
| `memory.profile` | gpu | `make memory-paths-p14-profile` |
| `umma.pilot` | gpu | `make compute-umma-p24-pilot` |
| `umma.profile` | gpu | `make compute-umma-p24-profile` |
| `gemm.capture` | gpu | `make gemm-comparison-p35-smoke` |
| `memory.analyze` | gpu-free | `make memory-paths-p14-analyze` |
| `umma.analyze` | gpu-free | `make compute-umma-p24-analyze` |
| `campaign.validate` | gpu-free | this unit's own final integrity gate |

Every one of those Make invocations is made **silently and without directory
diagnostics** (`make --silent --no-print-directory <target>`), so Make never
echoes a recipe line. A recipe line can carry the absolute bind-mount source of
this checkout, and a private host path must never reach a durable Phase 4 log.
The targets themselves are unchanged. Before child output becomes a durable
text log, only this checkout's exact absolute root is replaced with
`<repo-root>`; ordinary diagnostics remain unchanged. P3.5 stdout is scientific
CSV data and is copied byte for byte, while its textual stderr uses the same
narrow replacement.

Every P1.4 stage receives `P1_4_CAMPAIGN_ID` = the top-level campaign ID and
`P1_4_PREFLIGHT_SUMMARY` = the exact preflight summary this invocation created
and validated. Every P2.4 stage receives `P2_4_CAMPAIGN_ID` and
`P2_4_PREFLIGHT_SUMMARY` the same way. The two GPU-free analyzers receive only
the campaign ID and never a GPU index. Every variable the orchestrator owns is
removed from the inherited environment first, so a stale operator export can
never leak into a stage it does not belong to.

## 4. Preflight handling

Every real invocation with pending GPU work requires an explicit **numeric**
`BLACKWELL_GPU_INDEX`. The orchestrator:

1. rejects a missing or non-numeric index **before** Docker, `nvidia-smi`,
   result creation, or any GPU-related subprocess. For a new campaign the
   public entry point itself rejects it before invoking anything; for a
   `--resume` the decision is taken after the read-only manifest reload that
   determines whether any GPU work is pending at all, and still before any
   mutation or subprocess;
2. runs the preflight only through `BLACKWELL_GPU_INDEX=<i> make preflight`;
3. captures the exact `results/preflight/<timestamp>/summary.json` **that
   invocation** produced. The directory is learned from the invocation's own
   stdout: `scripts/preflight.sh` prints exactly one
   `preflight: writing to results/preflight/<TS>` and exactly one
   `preflight: summary written to results/preflight/<TS>/summary.json`, and the
   two must name the same directory. `scripts/run_container.sh` prints exactly
   one allowlisted `run_container: selected index=<i> uuid=<UUID> name='<NAME>'
   driver=<DRV>` banner, which is what binds the requested **physical** index
   to the UUID that was actually exposed;
4. **never** discovers a preflight through `ls -t`, a "latest" symlink, glob
   ordering, or a modification time;
5. validates the captured summary: a non-symlink, non-empty, regular JSON file;
   `overall_status=PASS`; `git_dirty=false`; `git_commit` equal to the current
   clean `HEAD`; `gpu.uuid` equal to the UUID the launcher resolved from the
   explicitly selected physical index; `gpu.compute_cap = 10.3`;
   `gpu.logical_index = "0"`; and a `checks` list that is **exactly** the six
   required checks (`gpu_visibility`, `tool_versions`, `cuda_smoke_compile`,
   `cuda_smoke_run`, `cutedsl_smoke`, `ncu_profile`), each at `PASS`, with no
   duplicate and none missing;
6. passes that exact summary path to every remaining P1.4/P2.4 stage of the
   invocation;
7. on `--resume`, creates and validates a **fresh** preflight whenever GPU work
   is still pending, then requires its commit and GPU identity to match what
   the campaign already recorded. A campaign never mixes devices: a fresh
   preflight naming a different GPU aborts the resume.

The orchestrator never calls Docker or `nvidia-smi` for experimental work: all
GPU execution continues through the existing Make targets and ultimately
`scripts/run_container.sh`. It never selects a GPU automatically, never uses
`--gpus all`, privileged mode, host PID, added capabilities, a Docker socket,
MPS, `sudo`, or multiple GPUs, never changes clocks, power limits, persistence
mode, or compute mode, never uses `$(nproc)`, and adds no profiler route.

## 5. Top-level raw campaign layout

Created only for a real invocation:

```text
results/raw/phase4/<campaign_id>/
├── manifest/
│   ├── 000000.json
│   ├── 000001.json
│   └── ...
├── plan.json
├── logs/
│   └── <stage>.<attempt>.stdout.log and <stage>.<attempt>.stderr.log
└── exp03/
    └── gemm_comparison.csv
```

The experiment-owned trees stay where their own protocols put them. P4.1 does
**not** copy them: it records validated repository-relative references and
SHA-256 hashes instead. All five trees share one campaign ID:

```text
results/raw/phase4/<id>/
results/raw/exp01_memory_paths/<id>/
results/raw/exp01_memory_paths_p14/<id>/
results/raw/exp02_umma_throughput/<id>/
results/raw/exp02_umma_throughput_p24/<id>/
```

The existing blanket `results/raw/` Git-ignore rule already covers the new
tree; it is neither weakened nor amended, and no raw output is ever committed.

## 6. Manifest and evidence rules

The top-level manifest is an **append-only, hash-chained revision history** —
never one mutable `manifest.json`. Each transition appends one complete,
immutable snapshot as the next contiguous revision (`000000.json`,
`000001.json`, ...). Every revision carries `manifest_revision` (its own index)
and `previous_manifest_sha256` (the freshly recomputed hash of the preceding
revision, or `null` for revision 0). Loading re-opens, re-reads, re-hashes, and
re-validates **every** revision from `000000.json` forward on every call;
nothing about an earlier revision is trusted from memory. Publication is
exclusive-create; `os.replace()` is never used anywhere in this unit.

### 6.1 Allowlisted content

A revision may carry only these fields, each typed and classified:

| Field | Category |
|-------|----------|
| `schema_version` (`p41.v1`), `unit`, `campaign_id`, `campaign_kind`, `scope`, `stage_order`, `plan_sha256`, `git_commit`, `git_dirty` (always `false`), `publishable` (always `false`), `created_at_utc` | immutable |
| `state`, `updated_at_utc` | state-derived |
| `stage_attempts`, `stage_results` | append-only |
| `preflight_reference`, `gpu` | latest validated value |
| `outcome`, `failure_stage`, `failure_detail` | terminal-only |

`stage_results` records per-stage evidence: repository-relative paths, SHA-256
hashes, and the normalized comparable provenance of that component. For a P1.4
or P2.4 stage that is the unit's own campaign directory plus the **pin** of the
exact manifest revision this campaign accepted — `unit_manifest_path` (a
repository-relative path), `unit_manifest_revision`, `unit_manifest_sha256`, and
`unit_state` — plus, once that state is terminal, `unit_evidence_sha256`: a
digest over the snapshot the unit's **own** `verify_campaign_evidence_integrity()`
just recomputed fresh from disk, combined with fresh SHA-256 hashes of every
canonical terminal `analysis/` artifact checked against that unit manifest's
`artifact_sha256`. `gpu` carries exactly `uuid`, `name`, `compute_capability`,
`driver_version`.
`stage_attempts` records every attempt explicitly: stage, attempt number, start
and finish timestamps, outcome (`COMPLETE`/`FAILED`/`INTERRUPTED`), the child's
exact exit status, and both log paths.

**Never stored:** usernames, home paths, host names, full environment dumps,
credentials or tokens, SSH material, unrelated process information, complete
host command lines, or dynamic power, clock, temperature, or utilization
telemetry. A structural privacy gate rejects any manifest key whose name
matches a forbidden token at any nesting depth, and any string value that is an
absolute path. Durable textual child logs, P4.1's `campaign.validate` logs, and
the manifest's `failure_detail` pass through one narrow redaction at their write
boundary that replaces this checkout's own absolute root with the stable token
`<repo-root>`, because a diagnostic may legitimately quote an absolute evidence
path. Nothing else is rewritten: no general sanitization, no scientific CSV,
no experiment-owned evidence, and no GPU identity, failure diagnostic, or
reproducibility detail is removed.

### 6.1.1 Comparable provenance

Every selected component must agree with this invocation's own validated
preflight, and with every component already accepted, on all directly
comparable fields: Git commit, clean-tree status, GPU UUID, GPU name, compute
capability, and — between the closed unit schemas that expose them — the CUDA
driver and CUDA runtime API versions. Those two are recorded as an integer by
one schema and as a string by the other, so they are normalized to their decimal
text exactly the way P2.4's own `compare_application_provenance()` does: a
difference in textual formatting of the same version is never a mismatch, and a
difference in the version itself always is. A field an applicable closed schema
requires but that is absent or malformed fails closed.

The preflight's `gpu.driver_version` is the NVIDIA display-driver package string
reported by the launcher, a different measurement of a different thing from the
integer-encoded CUDA API versions; P1.4 documents that boundary explicitly and
P4.1 never crosses it either. Clean-tree status is enforced where each schema
exposes it: the campaign refuses to start on a dirty tree, every accepted
preflight summary must record `git_dirty=false`, every P3.5 row must record
`git_dirty=false`, and a composed unit that ever reported a dirty tree in its
own provenance is rejected.

The same comparison runs at initial acceptance, in the final `campaign.validate`
gate, and on every `--resume` revalidation.

### 6.2 Fail-closed filesystem handling

* Symlinks are rejected at every level, dangling or not, for the campaign root,
  every component, every subdirectory, and every artifact.
* Unexpected file types are rejected.
* Existing evidence is never overwritten: every write is
  `O_CREAT | O_EXCL | O_NOFOLLOW` on an already-open, `O_NOFOLLOW`-anchored
  directory descriptor, and the accepted GEMM artifact is published with an
  in-place hard link whose own `EEXIST` is the no-clobber guarantee.
* Valid partial evidence survives an interruption or a child failure; nothing
  an earlier attempt created is ever deleted. A retry uses the next free
  attempt number — derived from the validated campaign state **and** from the
  no-clobber logs already on disk — so an earlier stage log can never be
  overwritten or reused. An attempt claims its number by creating its two logs
  before the child runs, so a hard interruption can leave a log set whose
  attempt record was never published; the next attempt skips that number
  instead of colliding with it forever. Attempt numbers are therefore strictly
  increasing per stage, and a gap is legal evidence of exactly that. An attempt
  is recorded only after both named logs exist; if interruption lands between
  their exclusive writes, the lone log remains unmodified but is not referenced
  as a complete pair.
* Durable metadata always uses repository-relative paths.
* Evidence is re-hashed and revalidated immediately before a stage or a
  campaign is declared complete.

### 6.3 State machine

```text
None          -> IN_PROGRESS
IN_PROGRESS   -> IN_PROGRESS | COMPLETE | INCONCLUSIVE | FAILED | INTERRUPTED
FAILED        -> IN_PROGRESS      (only by an explicit --resume, see section 8)
INTERRUPTED   -> IN_PROGRESS      (only by an explicit --resume)
COMPLETE      -> (terminal)
INCONCLUSIVE  -> (terminal)
```

A top-level campaign may reach `COMPLETE` only when every selected component
has independently passed its existing validator **and** the final top-level
integrity gate. For a full campaign that means P1.4 with valid terminal
`ANALYZED` evidence, P2.4 with valid terminal `ANALYZED` evidence, a P3.5
capture passing the checks in section 7, every accepted unit still pinned to the
exact terminal manifest revision it was accepted at, and agreement across all
three on campaign ID and on every comparable provenance field of section 6.1.1.

A P2.4 `INCONCLUSIVE` analysis propagates: the top-level outcome becomes
`INCONCLUSIVE`, never `COMPLETE`, and the process exits 4. It is never silently
accepted as a complete final campaign.

**P4.1 creates no Phase 4 variability threshold, publication threshold, or
scientific acceptance rule.** Those policies belong to P4.2 and P4.3.

## 7. P3.5 capture validation

`gemm.capture` runs `make gemm-comparison-p35-smoke` with the P3.5 stdout
captured into an already-open, no-clobber descriptor and stderr captured into a
separate one, and with `RUN_CONTAINER_STDOUT_IS_DATA=1` so the launcher's own
banner cannot contaminate the CSV either. Make diagnostics, compiler output,
and launcher notices therefore never reach the captured stream.

Before `gemm.capture` is recorded `COMPLETE`, the captured file must pass
**P3.5's own** `validate_serialized_output` from
`scripts/check_gemm_comparison_p35.py` — never a second interpretation of the
same contract — plus the Phase 4 bindings only an enclosing campaign can check.
Together they require:

* exactly one header and 20 data rows (21 lines);
* exactly the frozen 100-field `p35.v1` schema, in the frozen field order;
* exact shape-major and candidate-major ordering;
* all five frozen shapes;
* exactly `cutedsl/nonpersistent_1cta`, `cutedsl/persistent_1cta`,
  `cutedsl/persistent_2cta`, and `cublaslt/heuristic_first_supported`;
* `correctness=PASS` on every row;
* `publishable=false` on every row;
* `git_dirty=false` on every row;
* the current campaign Git commit on every row;
* the same GPU UUID, GPU name, compute capability, and driver version as the
  Phase 4 preflight and the other experiments;
* valid finite positive timing and comparison fields under the existing P3.5
  rules;
* no additional, missing, duplicate, reordered, partial, or malformed row.

Any failure emits **no** accepted Phase 4 GEMM artifact — the raw capture stays
in `logs/` as failure evidence and nothing is linked into `exp03/` — and
prevents every later stage from being marked complete. A capture so malformed
that P3.5's own validator cannot even process it is also a hard failure, never
an unhandled crash.

## 8. Resume semantics

`--resume` is evidence-driven, not existence-driven. For every stage that
appears complete the orchestrator:

1. loads the top-level manifest chain (re-hashing every revision);
2. loads the underlying P1.4/P2.4 campaign through **its own** semantic loader
   (`load_p14_manifest_chain` / `load_p24_manifest_chain`) and, at the states
   that unit's own protocol re-verifies, its own
   `verify_campaign_evidence_integrity()`; or re-parses the P3.5 CSV through
   P3.5's own validator;
3. re-hashes all referenced evidence, including the exact unit manifest
   revision file this campaign pinned;
4. rechecks the immutable campaign ID, kind, scope, and the full comparable
   provenance of section 6.1.1;
5. skips the stage **only** if all of those still pass.

A unit accepted in a terminal state stays pinned to exactly the revision that
was accepted: its recorded path, revision number, SHA-256, and evidence-integrity
digest must all still match. A **later** terminal revision, however internally
valid, is never silently adopted; the same revision with changed content is
rejected by its hash; and changed referenced evidence is rejected by the unit's
own integrity gate. A unit still legitimately advancing (`PILOT_COMPLETE` →
`COMPLETE` → `ANALYZED`) may move forward, but the earlier revision file it was
accepted at must still exist unchanged, byte for byte.

A stage is never skipped merely because a directory or a filename exists. The
underlying unit's state is revalidated as "at or after" what was accepted while
that state is non-terminal — a P1.4 campaign legitimately advances
`PILOT_COMPLETE` → `COMPLETE` → `ANALYZED` as later stages run — and as exact
equality once terminal. A regression, `FAILED`, `INTERRUPTED`, or an unknown
label is always rejected.

Additional rules:

* an earlier stage log is never overwritten; each retry uses the next free
  attempt number and every attempt is recorded explicitly. An interruption is
  handled at exactly the same orchestration boundary for every stage, including
  `campaign.validate`: the campaign is recorded `INTERRUPTED` where that is
  safe, the process exits 130, every artifact and log already created is
  preserved, and that stage stays eligible for a new attempt on `--resume`;
* execution stops immediately when a child command fails, preserving its exit
  status and all evidence, and no downstream stage runs;
* a resume of an already `COMPLETE` (or `INCONCLUSIVE`) campaign performs pure
  revalidation: it rewrites no artifact, appends no manifest revision, and runs
  no child command;
* a mismatch in campaign kind, scope, commit, or GPU UUID aborts;
* a changed or corrupted previously accepted artifact aborts rather than being
  regenerated automatically;
* a campaign that failed inside a P1.4 or P2.4 stage is **not** resumable: that
  unit's own campaign is terminally `FAILED` and its frozen protocol never
  reopens a terminal campaign, so P4.1 fails closed and instructs the operator
  to start a new campaign with a new `--campaign-id`. Only the two stages that
  own no persistent unit state — `preflight` and `gemm.capture` — may be
  retried inside the same campaign.

## 9. Files

Added by P4.1:

```text
scripts/run_all.sh                          the only public entry point
scripts/phase4_orchestrator.py              the private helper
scripts/check_phase4_orchestrator_p41.py    the fail-closed checker
src/phase4/P4_1_PROTOCOL.md                 this document
```

Updated minimally: `Makefile`, `PLAN.md`, `README.md`, `results/README.md`. The
only P2.4-owned change is the split of `compute-umma-p24-check`'s GPU-free half
into `compute-umma-p24-check-gpu-free`, which that gate still depends on, so its
meaning is unchanged (see section 10).

Unchanged: every CUDA and CuTe DSL kernel, `VERSIONS.env`,
`PHASE3_VERSIONS.env`, the `Dockerfile`, `scripts/run_container.sh`,
`scripts/preflight.sh`, every P1.4/P2.4/P3.5 schema, and every existing
smoke/check target's semantics. No new external dependency and no version-pin
change: the orchestrator uses only the Python standard library.

### 9.1 The one stale frontier assertion this unit advanced

Two closed-unit guards required the literal `PLAN.md` row
`P4.1 | Orchestrator | NO | NO | NO`, which structurally forbade P4.1 from ever
being implemented:

1. `Makefile`'s `check-static` Phase 4 assertion;
2. `scripts/check_gemm_comparison_p35.py`'s `PHASE4_STATUS_LINES`, together
   with the self-test case that used the P4.1 row to prove "a PLAN.md that
   implements Phase 4 is rejected".

At implementation time only that exact assertion was advanced to the
then-truthful frontier `P4.1 | Orchestrator | YES | NO | NO`. This documentary
closure advances the P4.1-owned assertions in the `Makefile`, the P3.5 checker,
and the P4.1 checker to `YES | YES | YES`. Nothing is weakened: every
closed-unit regression is preserved, stale and impossible partial states are
rejected, and **at that P4.1 closure P4.2 and P4.3 were still recorded
unimplemented**. Later units advance only their own rows.

## 10. Make targets

Both are GPU-free, and neither executes a campaign.

### `phase4-p41-plan`

Prints the deterministic full-campaign plan through the real public entry
point's `--dry-run`, using a fixed placeholder campaign ID. No mutation.

### `phase4-p41-check`

Runs with **no container runtime, no `nvidia-smi`, no CUDA compilation, no GPU,
and no network**, and creates no campaign. Its complete prerequisite closure is:

```text
phase4-p41-check
├── memory-paths-p14-check            (already container-free in full)
└── compute-umma-p24-check-gpu-free   (the GPU-free semantic/schema half of
                                       compute-umma-p24-check)
```

`compute-umma-p24-check` itself is **not** a prerequisite: its own P2.1/P2.2/P2.3
prerequisites compile and disassemble inside the pinned image and therefore need
Docker. Its GPU-free half was split out verbatim into
`compute-umma-p24-check-gpu-free`, and `compute-umma-p24-check` now depends on
that half plus the same three gates as before, so P2.4's own gate keeps exactly
its previous meaning. The Docker-backed P3.5 gate (`gemm-comparison-p35-check`)
is likewise not a prerequisite.

That dependency graph is enforced mechanically rather than by comment: the
checker parses the Makefile, walks the real transitive prerequisites of
`phase4-p41-check`, and fails if any recipe in that closure invokes a container
runtime, `nvidia-smi`, or a CUDA compiler, or if a Docker-backed gate becomes
reachable again through a new dependency edge.

On top of its prerequisites the target runs `bash -n`, `python3 -m py_compile`,
`scripts/run_all.sh --help`, `scripts/run_all.sh --self-test`, the two
representative `--dry-run` plans,
`python3 scripts/check_phase4_orchestrator_p41.py --self-test`, and
`python3 scripts/check_phase4_orchestrator_p41.py .`. `make check-static` runs
the same P4.1 checker, and the checker itself revalidates P3.5's frozen row
count, candidate order, and shape count directly from
`scripts/check_gemm_comparison_p35.py`.

## 11. Security model

* No GPU is ever selected automatically; every GPU run requires an explicit
  `BLACKWELL_GPU_INDEX` and goes exclusively through the audited launcher, via
  an existing Make target.
* This unit never invokes Docker or `nvidia-smi` itself. The checker proves
  both files carry neither token.
* No forbidden cluster pattern appears in any P4.1 file:
  `--gpus all`, `NVIDIA_VISIBLE_DEVICES=all`, `--privileged`, `--pid host`, a
  Docker socket, `--cap-add`, `SYS_ADMIN`, shell tracing, `sudo`, a
  processor-count expansion, a persistence/clock/power `nvidia-smi` mutation,
  `--force-overwrite`, `--set full`, and every clock-controlling profiler mode
  are all absent and mechanically scanned for.
* Only allowlisted device data is recorded: GPU UUID, name, compute capability,
  and driver version, all taken from already-validated evidence.
* Child processes write into already-open, no-clobber descriptors and never
  perform their own path lookup for an output name.
* No durable Phase 4 log carries a private host path. Every experimental Make
  target is invoked with `--silent --no-print-directory`, so no recipe line —
  and therefore no absolute bind-mount source — is ever echoed into a captured
  stream, and the narrow redaction of section 6.1 replaces this checkout's own
  root in the durable text P4.1 writes itself. The children's own output is
  preserved verbatim and no scientific content is ever rewritten.

## 12. What was and was not run

### 12.1 GPU-free acceptance commands performed by the author

```bash
git diff --check
bash -n scripts/run_all.sh
python3 -m py_compile scripts/phase4_orchestrator.py scripts/check_phase4_orchestrator_p41.py
scripts/run_all.sh --help
scripts/run_all.sh --self-test
scripts/run_all.sh --dry-run --campaign-id 20990101T000000Z --campaign-kind pilot
scripts/run_all.sh --dry-run --campaign-id 20990101T000000Z --campaign-kind pilot --only memory
python3 scripts/check_phase4_orchestrator_p41.py --self-test
python3 scripts/check_phase4_orchestrator_p41.py .
make -n phase4-p41-check    # the dependency graph itself, before running it
make phase4-p41-check
make check-static
```

**These are the author's own self-checks. They are not an independent audit,
and GPU-free checks are not GB300 verification.**

### 12.2 Independent audit and remediation

The independent review covered implementation commit
`129c20c1eeb11b11076e765fa1e59a73831d6f2d` and two focused remediation
rounds. The first correction was published as
`0605b48d03361763cbea92c269c1a79775401d84`; the final correction was published
as `77908dae377fb131ff90baef455c79d6d2c28b0b`. The final independent re-audit
accepted `77908dae377fb131ff90baef455c79d6d2c28b0b` with no remaining blocker.
P4.1 was independently audited.

### 12.3 GB300 verification and P4.2 pilot

On 12 August 2026 the operator ran `make phase4-p41-check` and
`make check-static`; both completed successfully. The operator then selected
idle physical GPU index 7 (NVIDIA B300 SXM6 AC, driver `610.43.02`) and ran the
full deterministic nine-stage plan from clean commit
`77908dae377fb131ff90baef455c79d6d2c28b0b`:

```bash
BLACKWELL_GPU_INDEX=7 scripts/run_all.sh \
  --campaign-id 20260812T013848Z \
  --campaign-kind pilot
```

The process exited 0 and the campaign reached terminal state `COMPLETE`. A
subsequent invocation with `--resume`, after unsetting `BLACKWELL_GPU_INDEX`,
revalidated and skipped all nine stages, exited 0, and reported that no artifact
was rewritten and no manifest revision was appended. P4.1 was verified on
GB300. The same campaign satisfies P4.2's single pilot requirement. Every
artifact remains `publishable=false`.

### 12.4 State at P4.1 closure

At P4.1 closure no final campaign had been executed. No cross-campaign
statistical treatment,
cross-experiment interpretation, integrated table, final figure, or publication
review has been produced, and no existing result row has been promoted out of
its recorded `publishable=false` status. The pilot raw tree remains ignored and
uncommitted under `results/raw/`.

## 13. Non-goals

P4.1 adds none of: a new CUDA, CuTe DSL, or cuBLASLt implementation; a new
shape, candidate, layout, dtype, tile, cluster, or algorithm; autotuning or
algorithm benchmarking; a new Nsight Compute case; a separate cuBLASLt or
CuTe DSL execution path; multi-GPU execution; clock or power control; automatic
GPU selection; raw-result publication; final statistical aggregation; a
high-variability rejection threshold; integrated roofline analysis; a final
conclusion, table, or figure; P4.2 final-campaign policy or completion; P4.3
documentation or audit closure; a new external dependency; a version-pin
change; or any commit, push, merge, or pull request.

## 14. Status

```text
P4.1 | Orchestrator                                | YES | YES | YES
P4.2 | Pilot plus three final campaigns            | YES | YES | YES
P4.3 | Integrated analysis, documentation, audit   | NO  | NO  | NO
```

`P4.1 = YES / YES / YES`. The independent audit and GB300 verification passed.
Pilot campaign `20260812T013848Z` is terminally `COMPLETE` and remains
non-publishable. P4.2 has since completed three independent final campaigns and
closed as `YES / YES / YES`; P4.3 remains unimplemented, and no publishable
result exists anywhere in this repository.

P4.2 has since landed its frozen protocol and its GPU-free cross-campaign
checker (`src/phase4/P4_2_PROTOCOL.md`,
`scripts/check_phase4_campaigns_p42.py`). Its final independent review accepted
execution commit `b08e45c2636a3ac17c94ad8b1368084914196d7a`; final campaigns
`20260817T110330Z`, `20260817T111310Z`, and `20260817T112011Z` then completed
and passed terminal and cross-campaign revalidation, which is why the P4.2 row
above now reads `YES | YES | YES`. **Nothing in this document's P4.1
implementation, audit, remediation, or GB300 record changed**, and P4.2 changed
neither `scripts/run_all.sh` nor `scripts/phase4_orchestrator.py`.
