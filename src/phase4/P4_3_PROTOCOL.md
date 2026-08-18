# P4.3 — integrated analysis, documentation, and closing audit preparation

Status: `P4.3 = YES / NO / NO` (Implemented / Audited / Verified on GB300).

* **P4.3 is an implemented, offline, read-only analysis layer** over already
  accepted GB300 evidence. It executes no GPU command and starts no Docker
  container, `nvidia-smi`, CUDA compilation, Nsight Compute run, preflight, or
  campaign.
* **Independent audit: NOT PERFORMED.**
* **Production analysis: NOT RUN.** No P4.3 run against the three real final
  campaigns has been performed in this repository.
* **No curated P4.3 result has been accepted for publication**, and
  **no publishable result exists** anywhere in this repository.
* **Phase 4 and the complete TFM are not closed.** They stay open until the
  later independent audit, the production run from that audited commit, and the
  review of its outputs all pass.

The GPU-free implementation checks in section 12 were run by the author. **They
are self-checks, not an independent audit, and a GPU-free check is never GB300
verification.**

## 0. Trust model

P4.3 inherits P4.1's trust model unchanged (`src/phase4/P4_1_PROTOCOL.md`
section 0) and P4.2's population model unchanged
(`src/phase4/P4_2_PROTOCOL.md` section 0). The campaign filesystem under
`results/raw/` is trusted and single-writer. P4.3 adds no new defence against a
malicious concurrent process, and it adds no second interpretation of the
existing evidence contracts: every per-campaign decision is delegated to P4.1
through P4.2's own strictly read-only evidence mode, and every P3.5 capture is
validated by P3.5's own canonical validator.

## 1. Purpose and scope

P4.2 closed as `YES / YES / YES` with an immutable population. P4.3 is the
smallest auditable layer that:

1. deeply revalidates that frozen population before reading any value;
2. reads the canonical terminal P1.4, P2.4, and P3.5 artifacts each final
   campaign's manifest pins;
3. performs cross-campaign descriptive aggregation with **the campaign** as the
   independent replicate;
4. produces one small curated tree of tables, a JSON summary, a Markdown
   report, and SVG figures;
5. answers the repository's research question using only conclusions the
   collected evidence supports;
6. records every limitation and every unavailable quantity explicitly;
7. provides a deterministic verifier for its own outputs;
8. prepares — and does not pre-empt — the independent closing audit.

P4.3 adds **no** CUDA, CuTe DSL, or cuBLASLt code; no shape, candidate, matrix,
layout, dtype, tile, cluster, or algorithm; no campaign runner and no resume
path; no measurement parameter and no execution-order change; no Nsight Compute
case or metric; no raw schema and no change to any existing analysis schema; no
external Python dependency; no version pin; and no automatic GPU selection. It
creates no new experimental evidence. `scripts/run_all.sh` and
`scripts/phase4_orchestrator.py` are untouched and remain byte-identical to the
content P4.2 pinned by SHA-256.

## 2. Frozen experimental population

The accepted Phase 4 population is immutable and is restated here as literal
constants, never discovered:

```text
Accepted pilot (excluded from every statistic):
20260812T013848Z

Final campaign 1: 20260817T110330Z
Final campaign 2: 20260817T111310Z
Final campaign 3: 20260817T112011Z

Frozen final execution commit:
b08e45c2636a3ac17c94ad8b1368084914196d7a
```

Binding rules:

* the pilot is **orchestration qualification evidence only**; it never enters a
  statistic, ranking, variability estimate, table, figure, or scientific
  conclusion, and appears in the outputs solely as excluded provenance;
* exactly those three declared final campaigns form the statistical population;
* a campaign is never discovered through "latest", a timestamp ranking, glob
  ordering, a modification time, or convenient selection — the analyzer contains
  no such route at all;
* no campaign is ever omitted, replaced, rerun, repaired, resumed, or added;
* nothing under `results/raw/` is ever modified, and the analyzer refuses any
  output root under `results/raw/` or `results/preflight/`;
* P4.3 executes no GPU command and invokes no Docker, `nvidia-smi`, CUDA, NCU,
  preflight, or `scripts/run_all.sh`.

Before a scientific value is read, the whole population is revalidated by
calling P4.2's own `check_campaign_evidence()`, which loads every manifest chain
through P4.1's audited loader, re-hashes every pinned artifact, runs P4.1's
complete terminal-stage revalidation, compares the shared commit, plan, stage
order, GPU identity, and comparable provenance, and fails on any undeclared
final campaign. If that gate does not pass, nothing is read and nothing is
written.

## 3. Statistical unit and cross-campaign policy

**The independent replicate is one complete final campaign**, never one timing
repetition.

For every reported configuration or comparison the analyzer takes the canonical
campaign-level estimate from that campaign's already validated terminal
artifact, preserves the three individual campaign values, and computes across
those three values only:

```text
campaign_count = 3
mean
median
sample standard deviation (n - 1)
coefficient of variation, where mathematically meaningful
minimum
maximum
```

Frozen rules:

* the three campaigns' internal timing repetitions are **never pooled**; a
  sample of anything other than three campaign-level values is rejected;
* no observation and no campaign is ever removed, and no outlier filter runs;
* no p-value and no statistical-significance claim is computed;
* no confidence interval is bootstrapped from three campaigns; the closed
  units' within-campaign intervals are preserved as provenance only and are
  never presented as cross-campaign intervals;
* the existing strict `CV > 5.0%` review threshold is applied only as a
  cross-campaign **diagnostic** for strictly positive performance metrics; a
  flag never excludes a campaign, changes a result, or triggers replacement GPU
  work;
* no coefficient of variation is computed for a signed or near-zero quantity
  such as `gap_to_cublaslt_pct`; its individual values, mean, median, sample
  standard deviation, minimum, and maximum are reported instead;
* every ratio is computed **inside** each campaign first and only then
  summarized across the three campaign-level ratios; a ratio is never formed
  from two independently aggregated means or medians;
* full precision is retained throughout the computation; decimals are applied
  only when a value is serialized.

A quantity for which a statistic is deliberately not computed carries the
canonical `not_applicable` token, never a fabricated number.

## 4. Experiment-specific analysis

### 4.1 Experiment 1 — LDGSTS versus TMA

Source: each final campaign's canonical P1.4 terminal artifacts
(`analysis/pilot_statistics.csv`, `analysis/pairwise_comparison.csv`,
`analysis/saturation_candidates.csv`, `analysis/ncu_validation.csv`).

* for each of the 18 frozen `(method, stages, bytes_in_flight_kib)`
  configurations, the campaign-level **median** `effective_gbps` is aggregated;
* for each of the 9 identical `(stages, bytes_in_flight_kib)` pairs, that
  campaign's own `tma_to_ldgsts_ratio` is taken and the three campaign-level
  ratios are summarized. A value above one means **TMA measured higher** and a
  value below one means **LDGSTS measured higher**. The word "winner" is not
  used and no significance claim is made;
* each campaign's `earliest_tested_candidate_saturation_bif_kib` is preserved
  per group. One final consensus candidate is reported **only** if all three
  campaigns agree; otherwise the analysis states that no single cross-campaign
  consensus candidate exists and lists all three results. It is never called a
  universal HBM saturation threshold;
* the existing limitation is preserved verbatim in the outputs: NCU/HBM
  validation covers exactly **six predefined cases** and is never extrapolated
  to the other twelve configurations.

### 4.2 Experiment 2 — BF16 UMMA throughput

Source: each final campaign's canonical P2.4 terminal artifacts
(`analysis/configuration_statistics.csv`, `analysis/scaling.csv`,
`analysis/saturation.csv`, `analysis/profile_validation.csv`,
`analysis/empirical_ceiling.json`).

* campaign-level medians of the clock-independent metrics `flops_per_cycle` and
  `flops_per_cycle_per_sm` are aggregated over the 24 frozen configurations;
* `estimated_tflops_per_sm` is summarized only where all three campaigns carry
  trustworthy clock conversions (every one of the 24 profiled configurations
  reporting a valid SM-clock reading, and the selected candidate's own reading
  marked valid);
* for the 1-SM/2-SM comparison each campaign's own `speedup_2sm_over_1sm` and
  `scaling_efficiency_percent` are summarized. Values outside `[0, 100]` are
  preserved **without clamping** and keep the closed unit's surprising-value
  diagnostic;
* the depth-saturation selection and the empirical-ceiling selection of every
  final campaign are preserved. A final consensus is reported only when all
  three campaigns select the same result; otherwise the analysis states that the
  selection is not stable across campaigns;
* a device-wide estimate is reported **only** if every final campaign
  independently contains a valid estimate based on a validated SM count and all
  three SM counts agree. Otherwise a structured `unavailable` result carries the
  exact per-campaign reason. The B300 SM count is never taken from an external
  specification, hard-coded, or inferred from another field, and the per-SM
  microbenchmark ceiling is never converted into a whole-GPU peak without
  validated evidence.

### 4.3 Experiment 3 — CuTe DSL versus cuBLASLt

Source: each final campaign's accepted P3.5 capture
(`exp03/gemm_comparison.csv`), validated through **P3.5's own canonical
validator** (`validate_serialized_output`), which enforces exactly five frozen
shapes, exactly four candidates per shape, exactly 20 rows in frozen order, the
unchanged `p35.v1` schema, and correctness `PASS`. P4.3 additionally requires
every source row to record `run_kind=smoke` and `publishable=false`.

Per shape and candidate the following are summarized across the three final
campaigns:

```text
kernel_time_ms
tflops
throughput_ratio_vs_cublaslt
gap_to_cublaslt_pct
```

The cuBLASLt-relative ratio and gap are taken from each campaign **before**
cross-campaign aggregation. Per shape the best CuTe DSL variant reported by each
final campaign is preserved; one stable best CuTe DSL variant is declared only
if all three campaigns agree, otherwise the analysis reports "no stable best
CuTe DSL variant across the three final campaigns". Negative gaps and ratios
above one are preserved without clamping. **Beating cuBLASLt is not a success
requirement.**

## 5. Integrated interpretation

The report answers:

> How do HBM-to-SMEM data movement and fifth-generation Tensor Core throughput
> constrain BF16 GEMM performance on NVIDIA GB300, and how closely can the CuTe
> DSL implementation approach cuBLASLt?

It separates, explicitly and in named sections, what was **directly measured**,
what is a **deterministic derived quantity**, what is a **modeled estimate**,
what is an **interpretation or inference**, and what is **unavailable**.

No numerical roofline or bottleneck attribution is forced where the evidence is
not dimensionally comparable. In particular the outputs state that:

* the memory benchmark is not a direct measurement of GEMM memory traffic;
* the UMMA ceiling is a one-/two-SM empirical microbenchmark ceiling unless
  valid whole-device evidence exists;
* P3.5 contains no GEMM Nsight Compute profile and therefore cannot prove
  whether a specific GEMM shape is HBM-bound, Tensor-Core-bound,
  scheduler-bound, or affected by another implementation cost;
* the GEMM timing is hot-cache and is never described as a cold-cache workload;
* arithmetic intensity, compulsory bytes, and roofline quantities are absent;
  they may be added later only with explicit formulas, assumptions, units, an
  explicit "this is a model" status, and validated inputs;
* no external architectural peak or vendor specification is ever imported to
  fill an evidence gap.

The wording is "measured", "consistent with", "suggests", or "cannot
determine" — never an unsupported causal claim. **A scientifically honest
"unavailable from the collected evidence" result is preferred to an invented
conclusion.**

## 6. Files

Added by P4.3:

```text
scripts/analyze_phase4_p43.py           the offline read-only analyzer
scripts/check_phase4_integration_p43.py the GPU-free repository-contract checker
src/phase4/P4_3_PROTOCOL.md             this document
```

Updated minimally: `Makefile`, `PLAN.md`, `README.md`, `results/README.md`,
`src/phase4/P4_1_PROTOCOL.md`, and `src/phase4/P4_2_PROTOCOL.md` — in each case
only where a current-status assertion about P4.3 would otherwise have become
false — plus the stale status-frontier guards described in section 6.1.

Unchanged: `scripts/run_all.sh`, `scripts/phase4_orchestrator.py`, every CUDA
and CuTe DSL kernel, `src/memory/`, `src/compute/`, `src/gemm/`, every
P1.4/P2.4/P3.5 runner, analyzer, schema, and checker contract, `VERSIONS.env`,
`PHASE3_VERSIONS.env`, the `Dockerfile`, `scripts/run_container.sh`,
`scripts/preflight.sh`, and every raw or processed scientific result.

### 6.1 The stale frontier assertions this unit advanced

Four closed-unit guards required the literal `PLAN.md` row
`P4.3 | Integrated analysis, documentation, audit | NO | NO | NO`, which
structurally forbade P4.3 from ever being implemented — the same stale-frontier
situation P3.5 had to correct for P3.4, P4.1 for itself, and P4.2 for itself.
The guards in `Makefile`, `scripts/check_gemm_comparison_p35.py`,
`scripts/check_phase4_orchestrator_p41.py`, and
`scripts/check_phase4_campaigns_p42.py` were advanced by exactly one step, to
the now-truthful `YES | NO | NO`, and each of them now rejects every *other*
P4.3 state, including all four that would claim an audit or a verification that
has not happened. `scripts/check_phase4_campaigns_p42.py` additionally stopped
asserting that the three P4.3 files are absent and instead asserts that they
exist, are owned by P4.3, and start no campaign — while continuing to prove, on
its own source, that **P4.2 itself computes no statistic, threshold, ranking, or
figure**. Nothing was weakened: every closed P1–P4.2 assertion and every
impossible-state rejection is preserved.

## 7. Frozen artifact inventory

The production analysis creates exactly this tree and nothing else:

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

As an explicit flat inventory, in the exact order the analyzer produces and
publishes them:

```text
memory_paths.csv
umma_throughput.csv
gemm_comparison.csv
integrated_summary.json
report.md
figures/memory_paths.svg
figures/umma_throughput.svg
figures/gemm_comparison.svg
analysis_manifest.json
```

Requirements, all enforced by the analyzer:

* deterministic row order, JSON key order, decimal formatting, Markdown, and
  SVG bytes; two runs over identical evidence are byte-identical;
* schema version `p43.v1` on every table and document;
* all three final campaign IDs recorded explicitly, in the frozen order that the
  `campaign_1_value` / `campaign_2_value` / `campaign_3_value` columns follow;
* the pilot ID recorded only as excluded qualification provenance;
* the final execution commit, the common GPU identity, and the comparable
  provenance recorded;
* exact repository-relative source paths and SHA-256 hashes for every artifact
  read;
* an exact SHA-256 in `analysis_manifest.json` for every other output artifact.
  The manifest is the one artifact it cannot hash from inside itself; `--verify`
  recomputes and compares every byte of it as well;
* no absolute path, username, home directory, environment dump, credential,
  hostname, or unrelated metadata;
* no mutation of raw evidence and no output root under `results/raw/` or
  `results/preflight/`;
* no overwrite of an existing different artifact — publication is
  `O_CREAT | O_EXCL | O_NOFOLLOW` and never `os.replace()`;
* an existing byte-identical output is verified rather than rewritten;
* symlinks and unexpected file types fail closed, and the output tree must
  contain exactly this inventory — a partial, conflicting, or unexpected
  artifact is fatal;
* output verification recomputes the complete analysis and compares byte for
  byte.

### 7.1 The publication gate

The curated artifacts may be marked publishable only after **all** of the
following have happened, in this order:

1. the P4.3 implementation passes an independent audit;
2. the analyzer is run from that independently audited clean commit;
3. all three final campaigns pass fresh read-only validation;
4. deterministic recomputation matches every output byte and hash;
5. the final output itself is independently reviewed.

**None of those five conditions has been met at the time of this
implementation, and this document does not claim otherwise.**

## 8. Public interface

```text
python3 scripts/analyze_phase4_p43.py --self-test

python3 scripts/analyze_phase4_p43.py --analyze \
  --campaign-root results/raw/phase4 \
  --pilot-campaign-id 20260812T013848Z \
  --final-campaign-id 20260817T110330Z \
  --final-campaign-id 20260817T111310Z \
  --final-campaign-id 20260817T112011Z \
  --output-root results/phase4

python3 scripts/analyze_phase4_p43.py --verify   (identical options)

python3 scripts/check_phase4_integration_p43.py --self-test
python3 scripts/check_phase4_integration_p43.py .
```

Exit codes: `0` OK, `1` at least one check failed, `2` a usage error. There is
no option that selects a GPU, names a campaign kind, or executes a campaign, and
the declared campaign IDs must equal the frozen population exactly, in the
frozen order.

## 9. Make targets

| Target | What it does |
|--------|--------------|
| `phase4-p43-check` | Fast, GPU-free, container-free, network-free, and independent of `results/raw/`: syntax checks, both temporary-fixture self-tests, and the repository-contract check. No prerequisites. |
| `phase4-p43-analyze` | The GPU-free production analysis of exactly the three frozen final campaigns. Requires the real raw evidence explicitly and starts nothing. |
| `phase4-p43-verify` | Deterministic byte-for-byte verification of the curated artifacts. Writes nothing. |

`scripts/run_all.sh` remains the only public Phase 4 execution entry point.
None of the three targets can start or resume a campaign, and the
repository-contract checker succeeds with no cluster evidence present.

## 10. Adversarial coverage

The two self-tests use temporary directories only and leave the repository
unchanged. Between them they cover: missing, duplicate, reordered, and
substituted final campaign IDs; the pilot being declared as a replicate; an
undeclared fourth final campaign; mixed final execution commits and mixed GPU
provenance; an incomplete or non-terminal campaign; a tampered manifest revision
and a tampered referenced artifact; a symlinked artifact and a symlinked output
root; missing, duplicate, reordered, and malformed CSV rows; non-finite values
and zero denominators; pooling a campaign's internal repetitions; the observable
difference between aggregate-of-within-campaign-ratios and
ratio-of-aggregates; the `n - 1` versus `n` standard-deviation denominator; a
coefficient of variation on a signed or zero-centred metric; disagreement in
saturation candidates, ceiling selections, and best variants; unavailable and
inconsistent SM-count evidence; negative GEMM gaps being clamped; high
variability causing exclusion; output nondeterminism; partial, conflicting, and
unexpected output artifacts; any attempted write under `results/raw/`; and any
route that could invoke a campaign, GPU, Docker, NCU, or `nvidia-smi`.

## 11. Status

```text
P4.1 | Orchestrator                              | YES | YES | YES
P4.2 | Pilot plus three final campaigns          | YES | YES | YES
P4.3 | Integrated analysis, documentation, audit | YES | NO  | NO
```

`P4.3 = YES / NO / NO`. The implementation exists. **Independent audit: NOT
PERFORMED. Production analysis: NOT RUN.** No curated P4.3 artifact has been
produced from the real evidence, no P4.3 result has been accepted for
publication, **no publishable result exists** anywhere in this repository, and
**Phase 4 and the complete TFM are not closed**.

## 12. Implementation-time GPU-free checks performed by the author

```bash
python3 -m py_compile \
  scripts/analyze_phase4_p43.py \
  scripts/check_phase4_integration_p43.py
python3 scripts/analyze_phase4_p43.py --self-test
python3 scripts/check_phase4_integration_p43.py --self-test
python3 scripts/check_phase4_integration_p43.py .
make phase4-p43-check
make phase4-p42-check
make phase4-p41-check
make check-static
git diff --check
```

**These are the author's own self-checks. They are not an independent audit,
and no GPU command and no production analysis against the real campaigns was
run.**

## 13. Non-goals

P4.3 adds none of: a campaign runner, a second public execution entry point, or
a Make target that could start or resume a campaign; a new CUDA, CuTe DSL, or
cuBLASLt implementation; a new shape, candidate, layout, dtype, tile, cluster,
or algorithm; a new Nsight Compute case, metric, or profiler route; a change to
any raw or existing analysis schema; automatic GPU selection; a new external
dependency or version pin; a p-value, significance test, cross-campaign
bootstrap, or outlier filter; a fourth or replacement final campaign; a
publication decision; and any commit, push, merge, or pull request.
