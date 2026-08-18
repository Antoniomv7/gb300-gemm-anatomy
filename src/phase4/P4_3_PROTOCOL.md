# P4.3 — integrated analysis, documentation, and closing audit preparation

Status: `P4.3 = YES / NO / NO` (Implemented / Audited / Verified on GB300).

* **P4.3 is an implemented, offline, read-only analysis layer** over already
  accepted GB300 evidence. It executes no GPU command and starts no Docker
  container, `nvidia-smi`, CUDA compilation, Nsight Compute run, preflight, or
  campaign, and it starts no child process at all.
* **The implementation has received remediation after a first independent
  audit.** That audit found seven defects: an incorrect scientific evidence
  taxonomy, parsed-then-dropped NCU diagnostics and within-campaign stability
  evidence, an untruthful metadata contract, an ancestor-symlink escape from the
  output tree, no immutable candidate-to-acceptance workflow, a missing
  analysis-code commit, and incorrect figure terminology. All seven are
  remediated here; section 14 records them.
* **The remediation itself is awaiting a new independent audit.**
* **Independent audit: NOT PERFORMED.**
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
surprising_value_flag                          source_diagnostic
diagnostic_flags                               source_diagnostic
ncu_coverage                                   source_diagnostic
```

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

Every terminal trust signal the closed units recorded is preserved in the
curated outputs, per campaign, **in the frozen campaign order**, as its own
long-format row and as a machine-readable JSON object:

```text
P1.4  within_campaign_sample_count, within_campaign_cv_percent,
      within_campaign_stability_review              per configuration (18)
P1.4  hbm_classification, diagnostic_flags          per profiled case (6)
P1.4  ncu_coverage (ncu_profiled | not_profiled)    per configuration (18)
P2.4  within_campaign_sample_count,
      within_campaign_cv_percent (flops_per_cycle),
      within_campaign_stability_review              per configuration (24)
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
* symlinks and unexpected file types fail closed, and the output tree must
  contain exactly this inventory — a partial, conflicting, or unexpected
  artifact is fatal;
* output verification recomputes the complete analysis and compares byte for
  byte.

### 7.3 The immutable candidate-to-acceptance workflow

Publication is **never** solved by overwriting, deleting, or regenerating a
candidate artifact. The analyzer writes nine **immutable candidate** artifacts,
each recording:

```text
publishable=false
publication_state=candidate_pending_independent_output_review
analysis_code_commit=<full audited commit>
```

The lifecycle, in this exact order, is:

```text
audited clean analysis-code commit
-> candidate production analysis
-> byte-for-byte verification
-> independent scientific/output review
-> external acceptance attestation
-> final documentation/status commit
```

which the analyzer records verbatim as:

```text
an independently audited, clean analysis-code commit
candidate production analysis from exactly that commit
byte-for-byte verification of the candidate bundle
independent scientific and output review of the complete bundle
an external acceptance attestation at src/phase4/P4_3_ACCEPTANCE.json
a final documentation and status commit
```

**None of the steps after candidate production has been performed, and the
candidate itself has not been produced either.** No P4.3 result is accepted.

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

* every field above is mandatory; a missing field is fatal;
* `schema_version` must be exactly `p43.acceptance.v1`, `unit` exactly `P4.3`,
  `status` exactly `ACCEPTED`, and `accepted_for_publication` exactly `true`;
* `analysis_code_commit` must equal the commit recorded in the bundle's own
  manifest — an attestation never transfers to a different analyzer commit;
* `final_campaign_ids` must be the frozen population in the frozen order, and
  `pilot_campaign_id_excluded` the accepted pilot;
* `analysis_manifest_sha256` must equal the manifest's own SHA-256 — this is
  what makes the attestation cover all nine artifacts without self-reference;
* `artifact_sha256` must cover exactly the nine inventory paths, each a
  canonical SHA-256, each equal to the reviewed bytes;
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
paths are deliberately **not** treated as dirty, and the manifest says so: they
cannot change the content of any tracked file, and the analyzer's own candidate
output tree is itself untracked until it is committed.

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

**Diagnostics (audit finding 2).** NCU diagnostics being parsed and then
dropped; `READ_AMPLIFICATION` disappearing from the CSV, the JSON, or the
report; different campaigns carrying different diagnostic flags; loss of a
within-campaign sample count, CV, or stability review; loss of the
surprising-value flag; conflation of within-campaign and cross-campaign
variability; a campaign-order permutation of any preserved diagnostic; a
cross-campaign REVIEW coexisting with calm within-campaign reviews and the
reverse.

**Metadata contract (audit finding 3).** A bundle that is not exactly nine
artifacts; a manifest that does not bind all eight siblings; a recomputed
sibling hash that does not match; a manifest that is not reproduced byte for
byte; a campaign value column with no campaign ID mapping; documentation that
claims each individual file embeds the campaigns, commit, provenance, and
publishable flag.

**Output containment (audit finding 4).** `repo/results` symlinked to a
directory outside the repository; `repo/results/phase4` symlinked outside;
`repo/results/phase4/figures` symlinked outside; an individual output file
symlinked to a file outside; an unexpected special file where a directory
belongs; an output path other than `results/phase4`; an output root under
`results/raw` or `results/preflight`; an unexpected file or directory inside the
output tree. Every case proves that the analyzer fails **before writing any byte
outside the temporary fixture repository**.

**Candidate and acceptance (audit finding 5).** An attempt to overwrite a
differing candidate artifact; verification against a tampered or missing
artifact; a missing or malformed future acceptance attestation; an attestation
with one wrong artifact hash; an attestation with a wrong manifest hash; an
attestation bound to a different analyzer commit; an incomplete inventory; a
substituted population; a status other than `ACCEPTED`; the real acceptance file
being present at all.

**Analysis-code commit (audit finding 6).** A missing analysis-code commit; an
abbreviated or uppercase commit; a dirty worktree; a modified, deleted, or
mode-changed tracked file; a staged change; an index with no valid cache-tree;
an index failing its own checksum; an index tree that differs from HEAD's tree;
an unborn HEAD; an in-progress merge or cherry-pick; a directory that is not a
repository; an analysis-code commit equal to the frozen execution commit; and
the proof that an untracked path is *not* dirty. The Git blob object id is
anchored against Git's own value for known bytes so that the fixture writer and
the reader cannot be symmetrically wrong.

**Figure terminology (audit finding 7).** The incorrect "bar" caption returning
in any figure, and the absence of the "summarizes exactly three campaign-level
values" statement.


## 11. Status

```text
P4.1 | Orchestrator                              | YES | YES | YES
P4.2 | Pilot plus three final campaigns          | YES | YES | YES
P4.3 | Integrated analysis, documentation, audit | YES | NO  | NO
```

`P4.3 = YES / NO / NO`. The implementation exists and has been remediated after
a first independent audit; **that remediation is itself awaiting a new
independent audit**. **Independent audit: NOT PERFORMED. Production analysis:
NOT RUN.** `results/phase4/` does not exist, `src/phase4/P4_3_ACCEPTANCE.json`
does not exist, no curated P4.3 artifact has been produced from the real
evidence, no P4.3 result has been accepted for publication, **no publishable
result exists** anywhere in this repository, and **Phase 4 and the complete TFM
are not closed**.

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
run.** `make phase4-p43-analyze` and `make phase4-p43-verify` were deliberately
**not** executed against the real evidence.

## 14. Remediation after the first independent audit

The first independent audit of the P4.3 implementation found seven defects that
the then-passing test suite did not detect. All seven are remediated in the
working tree, and each carries new regression coverage that fails on the
pre-remediation implementation:

| # | Defect | Remediation |
|---|--------|-------------|
| 1 | Derived, modeled, and cross-campaign quantities were classified as directly measured, across CSV, JSON, Markdown, and SVG | The frozen seven-class evidence taxonomy of section 5, an `evidence_class` on every CSV row, a `metric_classification` map in the JSON, one report section per class, and corrected figure captions |
| 2 | `parse_p14_ncu()` read `diagnostic_flags` and `aggregate_experiment_1()` dropped them; within-campaign stability evidence was lost and conflated with the cross-campaign CV | Section 5.4: every diagnostic preserved per campaign in the frozen order, `cross_campaign_*` naming, warning and review-condition sections in the report |
| 3 | The documentation claimed every individual file embedded the campaigns, commit, provenance, and `publishable=false`, and one self-test asserted it | Section 7.1's central envelope model, the manifest's `self_hash` statement, corrected `results/README.md`, and six real self-tests replacing the misleading one |
| 4 | `resolve_output_root()` performed only lexical containment, so `repo/results -> outside` let `report.md` be written outside the repository | Section 7.4's descriptor-anchored traversal, the single legal destination, and seven escape regressions that prove no byte is written outside the fixture |
| 5 | Publication had no immutable candidate model and no defined acceptance route | Section 7.3's candidate state, frozen `p43.acceptance.v1` schema, reusable validator, and the requirement that the real attestation stay absent |
| 6 | The bundle recorded no analysis-code commit and could not distinguish it from the execution commit | Section 7.5's runtime resolution and verification, with no production bypass and injected resolvers in the self-tests |
| 7 | The SVG captions called the min-max whisker a "bar" | Section 7.6, plus a regression check |

**This remediation has not been independently audited.** It is prepared for that
separate audit and for nothing else: nothing was committed, pushed, merged, or
proposed as a pull request, no campaign was created or rerun, no file under
`results/raw/` was touched, no kernel, runner, shape, candidate, measurement
parameter, closed-unit schema, dependency, or version pin was changed, and no
production analysis was executed.


## 13. Non-goals

P4.3 adds none of: a campaign runner, a second public execution entry point, or
a Make target that could start or resume a campaign; a new CUDA, CuTe DSL, or
cuBLASLt implementation; a new shape, candidate, layout, dtype, tile, cluster,
or algorithm; a new Nsight Compute case, metric, or profiler route; a change to
any raw or existing analysis schema; automatic GPU selection; a new external
dependency or version pin; a p-value, significance test, cross-campaign
bootstrap, or outlier filter; a fourth or replacement final campaign; a
publication decision; and any commit, push, merge, or pull request.
