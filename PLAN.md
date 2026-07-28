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
| P1.4 | Profiling, validation, analysis, pilot | YES | NO | NO |

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
`src/memory/P1_4_PROTOCOL.md` for the full design of each. Independent
re-audit, GB300 verification, and pilot execution are all still pending; no
performance result exists yet. A fresh preflight is required before any
P1.4 GPU work because the host driver changed after the Phase 0
verification.

## Phase 2 — BF16 UMMA throughput (27 July–2 August 2026)

Gate: Phase 1 gate passed.

| Unit | Description | Implemented | Audited | Verified on GB300 |
|------|-------------|-------------|---------|-------------------|
| P2.1 | 1-SM UMMA | NO | NO | NO |
| P2.2 | 2-SM UMMA | NO | NO | NO |
| P2.3 | Sweep (≤24 configurations) | NO | NO | NO |
| P2.4 | Profiling and empirical ceiling | NO | NO | NO |

## Phase 3 — CuTe DSL GEMM versus cuBLASLt (3–9 August 2026)

Gate: Phase 2 gate passed.

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
