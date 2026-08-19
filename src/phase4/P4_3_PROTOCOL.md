# P4.3 — integrated analysis, documentation, and closing audit preparation

Status: `P4.3 = YES / NO / NO` (Implemented / Audited / Verified on GB300).

* **P4.3 is an implemented, offline, read-only analysis layer** over already
  accepted GB300 evidence. It executes no GPU command and starts no Docker
  container, `nvidia-smi`, CUDA compilation, Nsight Compute run, preflight, or
  campaign, and it starts no child process at all.
* **The implementation has received remediation after two independent
  audits.** The first audit found seven defects: an incorrect scientific evidence
  taxonomy, parsed-then-dropped NCU diagnostics and within-campaign stability
  evidence, an untruthful metadata contract, an ancestor-symlink escape from the
  output tree, no immutable candidate-to-acceptance workflow, a missing
  analysis-code commit, and incorrect figure terminology. All seven are
  remediated here; section 14 records them. A second audit found seven further
  release blockers in that remediation: time-dependent lifecycle metadata,
  incomplete Python/import provenance, an incomplete trusted acceptance map,
  documentation/metadata contradictions, clipped or overlapping SVG content,
  omitted terminal diagnostics, and an unsafe partial-output contract. Section
  15 records the corrections.
* **The present remediation is awaiting a new independent audit.**
* **Independent audit: NOT PERFORMED for the present remediation.**
* **Production analysis: NOT RUN.** No P4.3 run against the three real final
  campaigns has been performed in this repository, and `results/phase4/` does
  not exist.
* **No acceptance attestation exists.** `src/phase4/P4_3_ACCEPTANCE.json` is
  absent, and the repository checker requires it to stay absent.
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

#### 4.1.1 Reading a candidate that sits at the tested grid boundary

The bytes-in-flight grid is exactly `16, 32, 64` KiB and the depth grid of
experiment 2 is exactly `4, 16, 64, 256`. A *candidate saturation point* is an
upstream selection made strictly **within** that grid; it is never a measured
architectural limit and never a universal saturation threshold. When the
selected candidate equals the largest tested value, the selection has reached
the boundary of the evaluated range, and the tested grid cannot distinguish a
genuine saturation point there from one lying beyond the largest value tested.

Because every campaign evaluates the same frozen grid, agreement between the
three campaigns on a boundary candidate is **not by itself** evidence that a
saturation point was independently located. Both experiments therefore emit,
beside every candidate list, a deterministic `saturation_boundary_interpretation`
recording the tested grid, its upper bound, whether every candidate sits at that
bound, and whether the summarized quantity was still rising across the final
tested step — that is, whether any plateau was observed at all. The report
repeats that reading in prose next to the candidates. This is the second
independent audit's finding M2. It adds no measurement, no metric name, and no
evidence class; the candidate values, metric names, and CSV columns are
unchanged.

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

## 5. The scientific evidence taxonomy (frozen)

The report answers:

> How do HBM-to-SMEM data movement and fifth-generation Tensor Core throughput
> constrain BF16 GEMM performance on NVIDIA GB300, and how closely can the CuTe
> DSL implementation approach cuBLASLt?

Every reported quantity carries **exactly one** of these classes, consistently
in the CSV tables (an `evidence_class` column on every row), in
`integrated_summary.json` (an `evidence_taxonomy.metric_classification` map plus
a class beside each summarized quantity), in `report.md` (one named section per
class), and in the SVG captions:

```text
measured_source_observation
within_campaign_derived_estimate
cross_campaign_descriptive_statistic
modeled_estimate
interpretation
unavailable_from_collected_evidence
source_diagnostic
```

The classes mean:

| Class | Meaning |
|-------|---------|
| `measured_source_observation` | A quantity an instrument recorded directly during the campaign. Used **only** where the closed upstream protocol explicitly supports that description. |
| `within_campaign_derived_estimate` | A deterministic quantity computed inside one campaign from measured inputs and validated constants. Reproducible, but not measured. |
| `cross_campaign_descriptive_statistic` | A statistic P4.3 computed over exactly three campaign-level values. It describes agreement between campaigns, never a new measurement. |
| `modeled_estimate` | A model or unit conversion applied to a derived estimate; its status as a model is always stated. |
| `interpretation` | A reading of the evidence, phrased as consistent-with, never as a causal claim. |
| `unavailable_from_collected_evidence` | A question the evidence cannot answer, reported as such instead of being filled in. |
| `source_diagnostic` | A trust signal a closed unit recorded about its own measurement; preserved verbatim and never converted into a result. |

### 5.1 The frozen classification

```text
kernel_time_ms                                 measured_source_observation
median_effective_gbps                          within_campaign_derived_estimate
tma_to_ldgsts_ratio                            within_campaign_derived_estimate
dram_read_ratio                                within_campaign_derived_estimate
hbm_classification                             within_campaign_derived_estimate
median_flops_per_cycle                         within_campaign_derived_estimate
median_flops_per_cycle_per_sm                  within_campaign_derived_estimate
speedup_2sm_over_1sm                           within_campaign_derived_estimate
scaling_efficiency_percent                     within_campaign_derived_estimate
earliest_tested_candidate_saturation_bif_kib   within_campaign_derived_estimate
earliest_tested_candidate_saturation_depth     within_campaign_derived_estimate
tflops                                         within_campaign_derived_estimate
throughput_ratio_vs_cublaslt                   within_campaign_derived_estimate
gap_to_cublaslt_pct                            within_campaign_derived_estimate
best_cutedsl_variant                           within_campaign_derived_estimate
estimated_tflops_per_sm                        modeled_estimate
estimated_device_equivalent_tflops             modeled_estimate
within_campaign_sample_count                   source_diagnostic
within_campaign_cv_percent                     source_diagnostic
within_campaign_stability_review               source_diagnostic
within_campaign_iqr_flagged_count              source_diagnostic
within_campaign_flops_per_cycle_per_sm_cv_percent  source_diagnostic
within_campaign_flops_per_cycle_iqr_flagged_count  source_diagnostic
within_campaign_flops_per_cycle_per_sm_iqr_flagged_count  source_diagnostic
profile_sm_clock_status                        source_diagnostic
profile_diagnostic_metrics_resolved_count      source_diagnostic
surprising_value_flag                          source_diagnostic
diagnostic_flags                               source_diagnostic
ncu_coverage                                   source_diagnostic
```

This table is the frozen taxonomy: it lists **exactly 29 metrics**, the same 29
that `METRIC_EVIDENCE` classifies in `scripts/analyze_phase4_p43.py`, with the
same class for every one. `check_phase4_integration_p43.py` asserts that
equality in both directions -- no metric classified only in the code, none
listed only here, and no class mismatch -- so the table cannot silently drift
behind the implementation again, as it had for the six `source_diagnostic`
entries added by remediation 13 and recorded by the second independent audit as
finding M5.

`kernel_time_ms` is the only metric described as a measured input, because P3.5
section 7 step 9 explicitly measures it with CUDA events on the candidate's own
execution stream, divided by the measured iteration count, after correctness
passes. Every cross-campaign mean, median, sample standard deviation,
coefficient of variation, minimum, and maximum computed beside any of these is
always a `cross_campaign_descriptive_statistic`, whatever the class of the
underlying quantity. A median or any other campaign-level estimate is **never**
presented as an individual raw observation.

### 5.2 The binding statements

* **`effective_gbps` is not HBM/DRAM bandwidth.** It is a deterministic
  effective-rate estimate: the benchmark's own logical `useful_bytes` divided by
  its measured CUDA-event kernel time. P1.1 and P1.2 already label it *effective
  copy bandwidth* and state explicitly that it is **not** HBM/DRAM bandwidth.
  The outputs say "timing-derived effective transfer rate", never "measured
  HBM-to-SMEM bandwidth".
* **The LDGSTS/TMA benchmark is a dedicated streaming data-movement
  microbenchmark.** It does not directly measure the memory traffic a GEMM
  kernel generates.
* **Only the six frozen Nsight Compute cases have HBM/DRAM traffic
  validation.** That evidence is kept separate from the effective-rate metric,
  neither validates nor calibrates the other, and it is never extrapolated to
  the other twelve configurations. Each of the eighteen configurations records
  an explicit `ncu_coverage` of `ncu_profiled` or `not_profiled`, and for every
  `not_profiled` configuration the outputs state that **actual HBM/DRAM traffic
  is unavailable from the collected evidence**.
* **`dram_read_ratio` and `hbm_classification` are derived from profiler
  evidence, not raw profiler counters.** `dram_read_ratio` is
  `dram__bytes_read.sum / useful_bytes`; `hbm_classification` is P1.4's frozen
  0.90 classification of that ratio.
* **`flops_per_cycle` and `flops_per_cycle_per_sm` are derived** from a
  validated `2*M*N*K*depth*iterations` operation count and the measured
  `%clock64` elapsed cycles. Being clock-independent does not make them directly
  measured. The outputs say "operation-and-cycle-derived throughput", never
  "measured FLOP/cycle".
* **`estimated_tflops_per_sm` is a modeled clock conversion** of a one-/two-SM
  microbenchmark result, never an architectural peak.
* **GEMM TFLOP/s, the TMA/LDGSTS ratios, the 1-SM/2-SM speedup and scaling
  efficiency, the cuBLASLt-relative ratios and gaps, the saturation selections,
  the best-variant selections, and every cross-campaign statistic are derived
  quantities**, and are described as such.
* Ratio language is "derived within-campaign ratio", never "measured ratio".

### 5.3 What is never introduced

No numerical roofline, no architectural peak imported from a vendor
specification, no arithmetic-intensity placement, no GEMM bottleneck
attribution, and no causal conclusion. In particular the outputs state that:

* the memory benchmark is not a direct measurement of GEMM memory traffic;
* the UMMA ceiling is a modeled conversion of a one-/two-SM empirical
  microbenchmark result unless valid whole-device evidence exists;
* P3.5 contains no GEMM Nsight Compute profile and therefore cannot prove
  whether a specific GEMM shape is HBM-bound, Tensor-Core-bound,
  scheduler-bound, or affected by another implementation cost;
* the GEMM timing is hot-cache and is never described as a cold-cache workload;
* arithmetic intensity, compulsory bytes, and roofline quantities are absent;
  they may be added later only with explicit formulas, assumptions, units, an
  explicit "this is a model" status, and validated inputs;
* no external architectural peak or vendor specification is ever imported to
  fill an evidence gap.

The wording is "measured", "derived", "modeled", "consistent with", or "cannot
determine" — never an unsupported causal claim. **A scientifically honest
"unavailable from the collected evidence" result is preferred to an invented
conclusion.**

### 5.4 Preserved diagnostics and the two kinds of variability

Every terminal diagnostic that P4.3 relies on to interpret a reported quantity
is preserved in the curated outputs, per campaign, **in the frozen campaign
order**, as its own long-format row and as a machine-readable JSON object:

```text
P1.4  within_campaign_sample_count, within_campaign_cv_percent,
      within_campaign_stability_review,
      within_campaign_iqr_flagged_count             per configuration (18)
P1.4  hbm_classification, diagnostic_flags          per profiled case (6)
P1.4  ncu_coverage (ncu_profiled | not_profiled)    per configuration (18)
P2.4  within_campaign_sample_count,
      within_campaign_cv_percent (flops_per_cycle),
      within_campaign_stability_review,
      within_campaign_flops_per_cycle_per_sm_cv_percent,
      within_campaign_flops_per_cycle_iqr_flagged_count,
      within_campaign_flops_per_cycle_per_sm_iqr_flagged_count,
      profile_sm_clock_status,
      profile_diagnostic_metrics_resolved_count     per configuration (24)
P2.4  surprising_value_flag                         per scaling pair (12)
```

No diagnostic is invented. P3.5 records no per-row stability diagnostic, so none
is fabricated for it; where information is unavailable it is marked
`not_applicable` explicitly rather than left blank or guessed.

Four names are kept unambiguously distinct, and are never interchanged:

```text
within_campaign_cv_percent          a closed unit's own CV inside one campaign
within_campaign_stability_review    that unit's own ok | REVIEW flag
cross_campaign_cv_percent           P4.3's CV over the three campaign values
cross_campaign_cv_review_flag       P4.3's ok | REVIEW | not_applicable flag
```

A cross-campaign CV above `5.0%` is a **review diagnostic only**. It never
replaces a within-campaign flag, never removes a campaign, and never alters a
result. The two may disagree in either direction; a disagreement is reported,
not resolved.

`report.md` summarizes **every** non-empty warning (for example
`READ_AMPLIFICATION`) and **every** cross-campaign review condition in two
dedicated sections; nothing is silently discarded.

#### 5.4.1 A zero cross-campaign spread is degenerate, not remarkable

The experiment 2 primary metric is derived from integer-cycle-quantized
`%clock64` measurements whose within-campaign variation is approximately zero.
The campaign-level medians therefore frequently land on identical values, and
the cross-campaign sample standard deviation and coefficient of variation are
frequently **exactly zero**, with a calm `ok` review flag and a zero-length
min-max whisker.

Those statistics are mathematically valid and are preserved exactly as
computed. What the outputs must not let a reader infer is the stronger claim:
a zero coefficient of variation here reports that **no cross-campaign review
threshold was exceeded**, and it is a property of a cycle-quantized instrument
— not a demonstration of extraordinary independent reproducibility. The UMMA
section of `report.md`, the `cross_campaign_variability_interpretation` field of
`integrated_summary.json`, and the UMMA figure caption all state this, and all
three add that a whisker spanning zero, or spanning less than the plotted
marker, is still drawn at its true length and is simply hidden by the marker.

This is the second independent audit's finding M3. No artificial uncertainty,
jitter, minimum whisker length, invented precision, or new review flag is
introduced; every statistic, flag, threshold, marker, and zero span is
unchanged.

#### 5.4.2 An empty diagnostic set is an observed result, not missing data

For a profiled Nsight Compute case the upstream `diagnostic_flags` field is
always present. An empty flag set therefore means the profiler recorded the
field and raised nothing — a clean result — which differs categorically from a
quantity that could not be collected at all. The canonical CSV serialization of
an empty set stays `not_applicable`, because the serialized representation is
frozen, so three states are named explicitly instead:

```text
present_and_empty          the field was present and its flag set was empty;
                           rendered "none recorded" in report.md
present_and_non_empty      a flag such as READ_AMPLIFICATION was recorded and
                           is surfaced verbatim
unavailable_not_profiled   one of the twelve never-profiled configurations,
                           whose actual HBM/DRAM traffic is unavailable from
                           the collected evidence
```

The per-campaign state accompanies every profiled case in
`integrated_summary.json`, the `diagnostic_flags` CSV row note records that the
upstream field was present, and `report.md` renders `none recorded` with an
adjacent footnote. An empty flag set is **never** converted into a warning, and
the regressions assert that the three states stay mutually distinguishable. This
is the second independent audit's finding M4; no scientific result changes.


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

## 7. Frozen artifact inventory and the metadata ownership model

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

### 7.1 Where the metadata lives

Provenance is **central, not duplicated**. `analysis_manifest.json` is the
**authoritative binding** for the complete output bundle, and it records:

```text
schema_version p43.v1
the exact artifact inventory (all nine paths) and artifact_count
the three ordered final campaign IDs
campaign_value_column_map: campaign_1_value -> 20260817T110330Z, and so on
pilot_campaign_id_excluded plus its qualification-only role
final_execution_commit          (the commit the campaigns ran from)
analysis_code_commit            (the commit whose code produced the bundle)
analysis_code_worktree_clean, its definition, and the verification method
the common GPU identity and the comparable provenance
repository-relative source paths and SHA-256 hashes of every artifact read
publishable=false, publication_state, publication_status
SHA-256 of each of the eight sibling output artifacts
```

`analysis_manifest.json` **cannot contain its own byte hash**: any value written
into the document changes the very bytes that hash would describe. This is a
structural property, not an omission, and the manifest states it precisely in a
`self_hash` object. Its own hash is bound from outside — by
`make phase4-p43-verify`, which recomputes every byte of it, and later by the
independent acceptance attestation of section 7.3, which records
`analysis_manifest_sha256` and thereby covers all nine artifacts without any
self-reference.

The other artifacts deliberately do **not** duplicate the global provenance:

* the three CSV files carry their own schema version, their deterministic key
  and data fields, the `evidence_class` of every row, and the three
  campaign-level values in the frozen campaign order;
* the three SVG files are deterministic visual artifacts;
* `integrated_summary.json` and `report.md` carry the scientific context needed
  to interpret the bundle: population, provenance, statistical policy, the
  evidence taxonomy, the preserved diagnostics, the limitations, and the
  candidate status;
* **every one of the eight non-manifest artifacts is path-and-hash bound by
  `analysis_manifest.json`.**

**A detached CSV or SVG is not a standalone provenance envelope.** These files
must be distributed together with `analysis_manifest.json`; on their own they
carry no campaign identity, no commit, and no publication state.

### 7.2 Requirements, all enforced by the analyzer

* deterministic row order, JSON key order, decimal formatting, Markdown, and
  SVG bytes; two runs over identical evidence are byte-identical;
* schema version `p43.v1` on every table and document;
* exactly nine artifacts, and the manifest binds all eight siblings;
* every campaign value column mapped to its campaign ID explicitly;
* no absolute path, username, home directory, environment dump, credential,
  hostname, or unrelated metadata;
* no mutation of raw evidence, and production output limited to the single
  logical destination `<repo-root>/results/phase4` (section 7.4);
* no overwrite of an existing different artifact — publication is
  `O_CREAT | O_EXCL | O_NOFOLLOW` and never `os.replace()`;
* an existing byte-identical output is verified rather than rewritten;
* symlinks and unexpected file types fail closed, and the completed output tree
  must contain exactly this inventory;
* a partial retry is safe only after a first, write-free pass has verified
  every existing artifact byte for byte and rejected every unexpected path;
  only then may a second pass create missing artifacts exclusively. A
  conflicting or unsafe partial tree fails before any missing artifact is
  written, while verification always rejects an incomplete tree;
* output verification recomputes the complete analysis and compares byte for
  byte.

### 7.3 The immutable candidate-to-acceptance workflow

Publication is **never** solved by overwriting, deleting, or regenerating a
candidate artifact. The analyzer writes nine **immutable candidate** artifacts.
`analysis_manifest.json` records authoritatively, and the JSON summary and
Markdown report repeat for readers:

```text
publishable=false
publication_state=immutable_candidate_requires_external_attestation
analysis_code_commit=<full audited commit>
```

The CSV and SVG siblings deliberately carry no mutable publication progress or
commit claim; the manifest binds them by path and hash. The candidate state is
an invariant property of the bytes, not a clock-dependent statement about
whether verification, review, or attestation is currently pending or complete.
Those facts live outside the immutable bundle.

The lifecycle, in this exact order, is:

```text
audited clean analysis-code commit
-> candidate production analysis
-> byte-for-byte verification
-> independent scientific/output review
-> external acceptance attestation
-> final documentation/status commit
```

The analyzer records the required order verbatim as a lifecycle contract, not
as a progress log:

```text
an independently audited, clean analysis-code commit
candidate production analysis from exactly that commit
byte-for-byte verification of the candidate bundle
independent scientific and output review of the complete bundle
an external acceptance attestation at src/phase4/P4_3_ACCEPTANCE.json
a final documentation and status commit
```

The repository's current status is recorded separately in sections 11 and 12:
the production candidate has not been produced and no P4.3 result is accepted.
Future candidate bytes will not need to change as those external steps occur.

Only after the independent output review may a **separate** file be created:

```text
src/phase4/P4_3_ACCEPTANCE.json
```

Its schema is frozen **now**:

```json
{
  "schema_version": "p43.acceptance.v1",
  "unit": "P4.3",
  "status": "ACCEPTED",
  "accepted_for_publication": true,
  "analysis_code_commit": "<the 40-character commit that produced the bundle>",
  "final_campaign_ids": ["20260817T110330Z", "20260817T111310Z", "20260817T112011Z"],
  "pilot_campaign_id_excluded": "20260812T013848Z",
  "analysis_manifest_sha256": "<sha256 of results/phase4/analysis_manifest.json>",
  "artifact_sha256": {
    "memory_paths.csv": "<sha256>",
    "umma_throughput.csv": "<sha256>",
    "gemm_comparison.csv": "<sha256>",
    "integrated_summary.json": "<sha256>",
    "report.md": "<sha256>",
    "figures/memory_paths.svg": "<sha256>",
    "figures/umma_throughput.svg": "<sha256>",
    "figures/gemm_comparison.svg": "<sha256>",
    "analysis_manifest.json": "<sha256>"
  },
  "verification_outcome": "byte_for_byte_recomputation_matched",
  "independent_output_review_outcome": "independent_output_review_passed"
}
```

Frozen validation rules, implemented by
`validate_acceptance_document()` in `scripts/analyze_phase4_p43.py` and
exercised against temporary fixtures by both self-tests:

* the object must contain exactly the eleven top-level fields above; a missing
  or additional field is fatal;
* `schema_version` must be exactly `p43.acceptance.v1`, `unit` exactly `P4.3`,
  `status` exactly `ACCEPTED`, and `accepted_for_publication` exactly `true`;
* `analysis_code_commit` must equal the commit recorded in the bundle's own
  manifest — an attestation never transfers to a different analyzer commit;
* `final_campaign_ids` must be the frozen population in the frozen order, and
  `pilot_campaign_id_excluded` the accepted pilot;
* `analysis_manifest_sha256` must equal the manifest's own SHA-256 — this is
  what makes the attestation cover all nine artifacts without self-reference;
* the trusted validator inputs are exactly a canonical manifest hash, the full
  lowercase analysis-code commit, and canonical hashes for exactly the eight
  non-manifest siblings; an incomplete or extra trusted map is rejected before
  it can authorize anything;
* the attestation's `artifact_sha256` must cover exactly the nine inventory
  paths, each a canonical SHA-256, each equal to the reviewed bytes; its
  `analysis_manifest.json` entry must equal the separately trusted
  `analysis_manifest_sha256`;
* `verification_outcome` and `independent_output_review_outcome` must be exactly
  the frozen tokens above;
* any difference in a campaign ID, a commit, a path, an inventory entry, or a
  hash rejects the attestation, so it can **never** authorize a modified or
  partially regenerated bundle.

The acceptance file:

* is **not** generated by the analyzer;
* is **not** part of the nine-artifact analysis inventory;
* **must not exist** in this remediation, and the repository checker fails if it
  does, because P4.3 remains `YES / NO / NO`;
* will be created only by a later, explicitly authorized closing action.

### 7.4 Output containment: descriptor-anchored traversal

The first independent audit reproduced an escape in which
`repo/results` was a symlink to a directory outside the repository, and
`report.md` was written outside the repository. Lexical containment
(`Path.resolve()`, `abspath()`, a string prefix, or `relative_to()`) is not a
containment guarantee, because it inspects a name rather than the filesystem.

Production output is therefore limited to the exact logical destination

```text
<repo-root>/results/phase4
```

and no other in-repository output directory is accepted in production mode.
The analyzer:

* opens the repository root once with `O_DIRECTORY | O_NOFOLLOW`;
* walks `results`, then `phase4`, then `figures` **component by component**,
  each opened with `O_DIRECTORY | O_NOFOLLOW` relative to the previously opened
  descriptor, and never re-resolved by pathname afterwards;
* rejects a symlink or a non-directory at **every** level, ancestors included;
* creates a missing component with `mkdir` relative to the validated parent
  descriptor, whose own `EEXIST` is the sole existence test;
* creates every artifact with `O_CREAT | O_EXCL | O_NOFOLLOW` anchored on that
  descriptor;
* verifies an existing byte-identical artifact instead of rewriting it, and
  refuses an existing different artifact;
* performs the exact-tree verification on the same descriptors with
  `follow_symlinks=False`;
* never bases the safety decision on `Path.resolve()`, `abspath()`, a string
  prefix, or a lexical `relative_to()`.

`results/raw/` and `results/preflight/` remain structurally unreachable.

### 7.5 The analysis-code commit

`final_execution_commit` and `analysis_code_commit` are different facts about
different events and are never conflated:

```text
final_execution_commit  b08e45c2636a3ac17c94ad8b1368084914196d7a
                        the commit the three GB300 campaigns RAN from
analysis_code_commit    resolved and verified at production runtime
                        the commit whose analysis code PRODUCED the bundle
```

It cannot be hard-coded, because the audited remediation commit does not exist
yet. The analyzer resolves and verifies it at production runtime with a strict,
pure-Python reader of `.git` that **starts no child process** and needs no
network: HEAD and refs are resolved from loose refs and `packed-refs`, and Git
objects from loose files or from a version-2 pack index.

This provenance gate runs before the analyzer imports or executes any other
repository-owned Python module and before P4.2 evidence revalidation. Thus a
dirty dependency cannot execute before the clean-commit refusal; the later P4.2
gate still runs before any scientific value is read or output byte is written.

Production analysis fails if:

* HEAD does not resolve to one full 40-character lowercase commit;
* the index does not provably equal that commit's tree (something is staged);
* any tracked path is missing, modified, or has a changed executable bit;
* any index entry is unmerged, skip-worktree, or intent-to-add;
* a merge, rebase, cherry-pick, revert, or bisect is in progress;
* the repository provenance cannot be verified at all;
* the declared analysis-code commit is not that verified HEAD, is not a full
  commit, or equals the frozen final execution commit (at which the P4.3
  analysis code did not exist).

The exact clean-worktree definition is recorded inside the manifest. Untracked
paths are rejected everywhere except the three descriptor-relative data roots
`results/raw`, `results/preflight`, and `results/phase4`. Their contents are
validated later by the evidence or candidate contracts; allowing them is what
permits accepted raw evidence and the analyzer's own output tree to coexist
with a clean tracked-code claim. An untracked importable file such as
`scripts/csv.py`, a repository-root `sitecustomize.py`, or any other untracked
path outside those roots is fatal.

Production additionally requires an isolated Python runtime: `python3 -I -B`
(`-I` implies isolated environment/import handling and user-site suppression;
`-B` forbids bytecode writes). The analyzer checks those runtime flags itself,
so calling its production modes without them is a usage error. Together with
the untracked-path policy, this prevents repository-local import shadowing from
escaping the analysis-code provenance claim.

There is **no production bypass**: no flag skips or supplies this verification.
The self-tests stay deterministic by injecting the resolver
(`run_analysis(..., git_provenance=...)`) and by exercising the real reader
against hand-built temporary Git fixtures.

### 7.6 Figure terminology

The min-max range in every figure is rendered as a vertical line, so it is
named a **min-max whisker**, never a bar. Each figure states that it summarizes
exactly three campaign-level values, one per final campaign, and a regression
check prevents the incorrect caption from returning.


## 8. Public interface

```text
python3 -I -B scripts/analyze_phase4_p43.py --self-test

python3 -I -B scripts/analyze_phase4_p43.py --analyze \
  --campaign-root results/raw/phase4 \
  --pilot-campaign-id 20260812T013848Z \
  --final-campaign-id 20260817T110330Z \
  --final-campaign-id 20260817T111310Z \
  --final-campaign-id 20260817T112011Z \
  --output-root results/phase4

python3 -I -B scripts/analyze_phase4_p43.py --verify   (identical options)

python3 -I -B scripts/check_phase4_integration_p43.py --self-test
python3 -I -B scripts/check_phase4_integration_p43.py .
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
unchanged. Between them they cover, in addition to every check that existed
before the first independent audit:

**Population and statistics.** Missing, duplicate, reordered, and substituted
final campaign IDs; the pilot declared as a replicate; an undeclared fourth
final campaign; mixed final execution commits and mixed GPU provenance; an
incomplete or non-terminal campaign; a tampered manifest revision and a tampered
referenced artifact; a symlinked referenced artifact; missing, duplicate,
reordered, and malformed CSV rows; non-finite values and zero denominators;
pooling a campaign's internal repetitions; the observable difference between
aggregate-of-within-campaign-ratios and ratio-of-aggregates; the `n - 1` versus
`n` standard-deviation denominator; a coefficient of variation on a signed or
zero-centred metric; unclamped negative gaps; disagreement in saturation
candidates, ceiling selections, and best variants; unavailable and inconsistent
SM-count evidence; output nondeterminism; and any route that could invoke a
campaign, GPU, Docker, NCU, or `nvidia-smi`.

**Evidence taxonomy (audit finding 1).** The exact pre-remediation wordings that
claimed a derived, modeled, or cross-campaign quantity as a direct measurement
are banned by phrase in every generated artifact; effective GB/s presented as
actual HBM bandwidth; FLOP/cycle presented as directly measured; an
unclassified metric reaching an output at all; a metric classified in the wrong
class; a missing `evidence_class` column; the absence of the explicit
`not_profiled` status or of the statement that actual HBM traffic is unavailable
for the twelve unprofiled configurations.

**Diagnostics (audit findings 2 and 13).** NCU diagnostics being parsed and
then dropped; `READ_AMPLIFICATION` disappearing from the CSV, the JSON, or the
report; different campaigns carrying different diagnostic flags; loss of a
within-campaign sample count, either reported within-campaign CV, any reported
IQR-flag count, a stability review, a profile SM-clock status, a profile
diagnostic-resolution count, or the surprising-value flag; conflation of
within-campaign and cross-campaign variability; a campaign-order permutation of
any preserved diagnostic; a cross-campaign REVIEW coexisting with calm
within-campaign reviews and the reverse.

**Metadata contract (audit findings 3 and 11).** A bundle that is not exactly nine
artifacts; a manifest that does not bind all eight siblings; a recomputed
sibling hash that does not match; a manifest that is not reproduced byte for
byte; a campaign value column with no campaign ID mapping; documentation that
claims each individual file embeds the campaigns, commit, provenance, and
publishable flag; an SVG that duplicates mutable lifecycle or commit metadata.

**Output containment (audit finding 4).** `repo/results` symlinked to a
directory outside the repository; `repo/results/phase4` symlinked outside;
`repo/results/phase4/figures` symlinked outside; an individual output file
symlinked to a file outside; an unexpected special file where a directory
belongs; an output path other than `results/phase4`; an output root under
`results/raw` or `results/preflight`; an unexpected file or directory inside the
output tree. Every case proves that the analyzer fails **before writing any byte
outside the temporary fixture repository**.

**Candidate and acceptance (audit findings 5, 8, 10, and 14).** An attempt to overwrite a
differing candidate artifact; verification against a tampered or missing
artifact; a missing or malformed future acceptance attestation; an attestation
with one wrong artifact hash; an attestation with a wrong manifest hash; an
attestation bound to a different analyzer commit; an incomplete inventory; a
substituted population; a status other than `ACCEPTED`; the real acceptance file
being present at all; any extra top-level attestation field; a malformed or
incomplete trusted sibling-hash map; a time-dependent progress claim inside
candidate bytes; and a partial retry that writes a missing early artifact before
discovering a later conflict.

**Analysis-code commit (audit findings 6 and 9).** A missing analysis-code commit; an
abbreviated or uppercase commit; a dirty worktree; a modified, deleted, or
mode-changed tracked file; a staged change; an index with no valid cache-tree;
an index failing its own checksum; an index tree that differs from HEAD's tree;
an unborn HEAD; an in-progress merge or cherry-pick; a directory that is not a
repository; an analysis-code commit equal to the frozen execution commit; an
untracked `scripts/csv.py`, repository-root `sitecustomize.py`, or other path
outside the three frozen data roots; and a production invocation without the
isolated, bytecode-free Python runtime. The Git blob object id is anchored
against Git's own value for known bytes so that the fixture writer and the
reader cannot be symmetrically wrong.

**Figures (audit findings 7 and 12, and second-audit finding M1).** The
incorrect "bar" caption returning in any figure; the absence of the "summarizes
exactly three campaign-level values" statement; clipped title/caption content;
footer lines beyond the frozen wrap limit; an undersized canvas; a missing
visible title, focused-scale cue, or direct abbreviation mapping.

Those checks are textual. The overlap coverage is **spatial**: for each of the
three figures the regression re-parses the emitted SVG, reconstructs a real
bounding box for every plot rectangle, tick label, rotated axis title, panel
title, x-axis label, caption, footer, data marker, min-max whisker, and series
line, and then asserts, by rectangle intersection, that

* every plot rectangle lies inside the 1080x480 canvas;
* consecutive panels are separated by a positive gutter;
* all six y-axis decorations of panels 2..N lie inside that panel's **own**
  allocated gutter;
* no axis decoration intersects any plot rectangle, and in particular none
  reaches back into a preceding one;
* no axis decoration, panel title, or axis label overprints a data marker,
  min-max whisker, or series line;
* no text of any kind is clipped by the canvas edge; and
* the base font stays at its readable size, so a collision can never be
  "resolved" by shrinking, hiding, or clipping a label.

A frozen pre-remediation fixture -- three panels at the old 22 px inter-panel
gap with tick labels at `x0 - 5` and a rotated axis title at `x0 - 52` -- is
asserted to be **rejected** by that regression, so the geometric coverage cannot
become vacuous. The bounding boxes come from one deterministic monospace advance
constant shared by the layout and the regression; no font library, renderer, or
other new dependency is involved.


## 11. Status

```text
P4.1 | Orchestrator                              | YES | YES | YES
P4.2 | Pilot plus three final campaigns          | YES | YES | YES
P4.3 | Integrated analysis, documentation, audit | YES | NO  | NO
```

`P4.3 = YES / NO / NO`. The implementation exists and has been remediated after
two independent audits; **the present remediation is awaiting a new independent
audit**. **Independent audit: NOT PERFORMED for this revision. Production analysis:
NOT RUN.** `results/phase4/` does not exist, `src/phase4/P4_3_ACCEPTANCE.json`
does not exist, no curated P4.3 artifact has been produced from the real
evidence, no P4.3 result has been accepted for publication, **no publishable
result exists** anywhere in this repository, and **Phase 4 and the complete TFM
are not closed**.

## 12. Implementation-time GPU-free checks performed by the author

```bash
python3 -I -B -m py_compile \
  scripts/analyze_phase4_p43.py \
  scripts/check_phase4_integration_p43.py
python3 -I -B scripts/analyze_phase4_p43.py --self-test
python3 -I -B scripts/check_phase4_integration_p43.py --self-test
python3 -I -B scripts/check_phase4_integration_p43.py .
make phase4-p43-check
make phase4-p42-check
make phase4-p41-check
make check-static
git diff --check
```

**These are the author's own self-checks. They are not an independent audit,
and no GPU command and no production analysis against the real campaigns was
run.** `make phase4-p43-analyze` and `make phase4-p43-verify` were deliberately
**not** executed against the real evidence.

## 13. Non-goals

P4.3 adds none of: a campaign runner, a second public execution entry point, or
a Make target that could start or resume a campaign; a new CUDA, CuTe DSL, or
cuBLASLt implementation; a new shape, candidate, layout, dtype, tile, cluster,
or algorithm; a new Nsight Compute case, metric, or profiler route; a change to
any raw or existing analysis schema; automatic GPU selection; a new external
dependency or version pin; a p-value, significance test, cross-campaign
bootstrap, or outlier filter; a fourth or replacement final campaign; a
publication decision; a merge or pull request.

## 14. Remediation after the first independent audit

The first independent audit of the P4.3 implementation found seven defects that
the then-passing test suite did not detect. All seven were remediated in commit
`3b101c2cfb45ffbd50910cb108d2dabffb26c081`, and each carries regression
coverage that fails on the pre-remediation implementation:

| # | Defect | Remediation |
|---|--------|-------------|
| 1 | Derived, modeled, and cross-campaign quantities were classified as directly measured, across CSV, JSON, Markdown, and SVG | The frozen seven-class evidence taxonomy of section 5, an `evidence_class` on every CSV row, a `metric_classification` map in the JSON, one report section per class, and corrected figure captions |
| 2 | `parse_p14_ncu()` read `diagnostic_flags` and `aggregate_experiment_1()` dropped them; within-campaign stability evidence was lost and conflated with the cross-campaign CV | Section 5.4: every diagnostic preserved per campaign in the frozen order, `cross_campaign_*` naming, warning and review-condition sections in the report |
| 3 | The documentation claimed every individual file embedded the campaigns, commit, provenance, and `publishable=false`, and one self-test asserted it | Section 7.1's central envelope model, the manifest's `self_hash` statement, corrected `results/README.md`, and six real self-tests replacing the misleading one |
| 4 | `resolve_output_root()` performed only lexical containment, so `repo/results -> outside` let `report.md` be written outside the repository | Section 7.4's descriptor-anchored traversal, the single legal destination, and seven escape regressions that prove no byte is written outside the fixture |
| 5 | Publication had no immutable candidate model and no defined acceptance route | Section 7.3's candidate state, frozen `p43.acceptance.v1` schema, reusable validator, and the requirement that the real attestation stay absent |
| 6 | The bundle recorded no analysis-code commit and could not distinguish it from the execution commit | Section 7.5's runtime resolution and verification, with no production bypass and injected resolvers in the self-tests |
| 7 | The SVG captions called the min-max whisker a "bar" | Section 7.6, plus a regression check |

That remediation was committed and pushed for a separate audit. It did not
create or rerun a campaign, touch `results/raw/`, change a kernel, runner,
shape, candidate, measurement parameter, closed-unit schema, dependency, or
version pin, or execute the production analysis.

## 15. Remediation after the second independent audit

The second independent audit examined commit
`3b101c2cfb45ffbd50910cb108d2dabffb26c081` and found seven additional release
blockers. The present implementation revision corrects all seven and adds
regressions that exercise their failure modes:

| # | Defect | Remediation |
|---|--------|-------------|
| 8 | Immutable candidate bytes encoded a clock-dependent “pending review” state | The invariant `immutable_candidate_requires_external_attestation` state; candidate bytes never claim external workflow progress |
| 9 | The clean tracked tree did not cover untracked Python/import shadowing | The Git provenance gate now precedes every other repository-module import; descriptor-relative rejection covers every untracked path outside the three frozen data roots; production requires `python3 -I -B` |
| 10 | Acceptance validation trusted a caller-supplied hash map without proving it was complete | Exact canonical trusted inputs: one manifest hash, one analysis-code commit, and exactly eight non-manifest sibling hashes; the attestation itself still binds all nine artifacts |
| 11 | Metadata ownership contradicted itself across manifest, report, CSV, and SVG | The manifest is authoritative; summary/report repeat reader context; CSV/SVG remain detached siblings with no mutable publication or commit claim |
| 12 | SVG titles/captions could be clipped and GEMM labels overlapped | A larger frozen canvas, visible neutral title and focused-scale subtitle, wrapped reserved footer, short direct labels, and a complete abbreviation mapping |
| 13 | P1.4/P2.4 terminal diagnostics were still omitted | P1.4 IQR counts; P2.4 per-SM CV, both reported IQR counts, profile SM-clock status, and resolved-profile-diagnostic counts are validated and preserved per campaign |
| 14 | The prose declared every partial tree fatal while the writer could mutate an incomplete tree before discovering a later conflict | A two-pass retry: validate every existing byte and all paths first, then exclusively create missing artifacts; verification remains exact and rejects partial output |

**This remediation has not been independently audited.** It is prepared for a
new separate audit and for nothing else. No campaign was created or rerun, no
file under `results/raw/` was touched, no kernel, runner, shape, candidate,
measurement parameter, closed-unit schema, dependency, or version pin was
changed, and no production analysis was executed.


## 16. Remediation after the second independent audit

The second independent audit returned **ACCEPT WITH NON-BLOCKING
OBSERVATIONS**: zero BLOCKER findings, zero MAJOR findings, five MINOR findings,
and eleven observations. It confirmed artifact integrity, provenance, and every
one of the 186 recomputed cross-campaign statistics. The five MINOR findings
were presentation, semantic-documentation, and regression-coverage gaps that
changed no reported value; they are corrected here.

| # | Second-audit finding | Correction in this remediation |
|---|----------------------|--------------------------------|
| M1 | The y-axis tick labels and rotated axis titles of panels 2..N were drawn inside the preceding panel's plot rectangle, overprinting data in two figures, because a 22 px inter-panel gap was set against a 52 px axis-title offset; the figure regressions were textual and could not see it | Every panel now owns a cell whose leftmost region is a deterministic gutter sized from that figure's own widest tick label, with clear separation after each plot rectangle; a real bounding-box collision regression covers all three figures and is itself pinned against a frozen pre-remediation fixture |
| M2 | The saturation candidates (64 KiB, depth 256) coincide with the largest tested grid points and no output disclosed it | `saturation_boundary_interpretation` in both experiments, prose beside every candidate list in `report.md`, and section 4.1.1 above state the tested grid, its upper bound, that a candidate is an in-grid selection rather than a measured limit, whether any plateau was observed, and that agreement at a grid boundary does not locate a saturation point |
| M3 | 72 of 73 UMMA cross-campaign rows report `SD = 0`, `CV = 0.000000%`, `ok`, and a zero-length whisker, presented without explanation | `cross_campaign_variability_interpretation`, a paragraph in the UMMA section of `report.md`, the UMMA figure caption, and section 5.4.1 above explain the cycle quantization, state that no review threshold was exceeded rather than that reproducibility was proved, and warn that a sub-marker whisker is hidden by the marker |
| M4 | An empty upstream NCU `diagnostic_flags` set serialized to the same `not_applicable` token used for genuinely unavailable quantities | The canonical token is preserved; the three states `present_and_empty`, `present_and_non_empty`, and `unavailable_not_profiled` are named in the summary, recorded in the CSV row note, rendered as `none recorded` with a footnote in `report.md`, and frozen in section 5.4.2 above |
| M5 | Section 5.1 listed 23 classifications while `METRIC_EVIDENCE` classified 29 | The six `source_diagnostic` metrics added by remediation 13 are listed; the checker now asserts exact bidirectional equality between the table and `METRIC_EVIDENCE` |

Nothing scientific moved. The excluded pilot, the three final campaign IDs, the
frozen execution commit, every formula, statistical population, sample-SD
convention, CV threshold, rounding rule, unit, row ordering, missing-data
policy, evidence class, and every numerical measurement are unchanged, as is the
nine-artifact candidate contract and the deterministic byte-for-byte
verification design. The corrections are confined to figure layout, generated
explanatory text, structured explanatory metadata, protocol documentation, and
regression coverage.

Three of the corrections change **generated bytes** — the three SVGs (layout),
`report.md` and `integrated_summary.json` (explanatory text and metadata), and
`memory_paths.csv` (the `diagnostic_flags` row note only, never a value) — so
`analysis_manifest.json` will pin different sibling hashes. The candidate
delivered to the second audit is therefore superseded and **must be
regenerated from the frozen campaigns** before any acceptance attestation is
created.

**This remediation has not been independently audited.** It is prepared for a
new separate audit and for nothing else. No campaign was created or rerun, no
file under `results/raw/` was touched, no kernel, runner, shape, candidate,
measurement parameter, closed-unit schema, dependency, or version pin was
changed, and no production analysis or verification was executed.
