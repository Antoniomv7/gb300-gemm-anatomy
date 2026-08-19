# Phase 4 integrated analysis (P4.3)

Schema version: `p43.v1`. Publication status: **publishable=false; publication_state=immutable_candidate_requires_external_attestation; candidate bytes do not record time-dependent audit, verification, review, or attestation progress; publication authority requires a separate hash-bound external acceptance attestation**.

This report is one of 9 artifacts in a candidate bundle. `analysis_manifest.json` is the authoritative provenance envelope and binds every other artifact by SHA-256; a detached CSV or SVG is not a standalone provenance envelope and must be distributed with the manifest.

## 1. Population and provenance

* Independent replicate: **one complete final campaign** (`campaign_count = 3`).
* Final campaign `20260817T110330Z`.
* Final campaign `20260817T111310Z`.
* Final campaign `20260817T112011Z`.
* Accepted pilot `20260812T013848Z` is orchestration qualification evidence only; excluded from every P4.3 statistic, ranking, variability estimate, table, figure, and conclusion.
* Final execution commit `b08e45c2636a3ac17c94ad8b1368084914196d7a` (the commit the three campaigns *ran* from).
* P4.3 analysis-code commit `dba7bfc5aca7f750ba9e6f8ac4e26b26e540f711` (the commit whose code produced this bundle), worktree clean: `true`.
* Column `campaign_1_value` is campaign `20260817T110330Z`.
* Column `campaign_2_value` is campaign `20260817T111310Z`.
* Column `campaign_3_value` is campaign `20260817T112011Z`.
* GPU `NVIDIA B300 SXM6 AC` (`GPU-40e00845-d89c-1393-2c32-a2dca3ee9442`, compute capability `10.3`, driver `610.43.02`).

## 2. Frozen statistical policy

* `campaign_count`: 3
* `confidence_intervals`: within-campaign intervals are preserved as provenance only; no cross-campaign interval is bootstrapped from 3 campaigns
* `cross_campaign_cv_effect`: a review diagnostic about agreement between campaigns; it never excludes a campaign, never changes a result, and never replaces a closed unit's own within-campaign stability review
* `cross_campaign_cv_review_threshold_percent`: 5.0
* `cross_campaign_cv_scope`: strictly positive performance metrics only
* `independent_replicate`: one complete final campaign
* `outlier_policy`: no observation and no campaign is ever removed
* `pooling_of_internal_repetitions`: forbidden
* `precision`: full precision is retained during computation; decimals are applied only at serialization
* `ratio_policy`: ratios are computed inside each campaign and only then summarized; a ratio is never formed from two aggregates
* `significance_testing`: none
* `statistics`: mean, median, sample_standard_deviation_n_minus_1, coefficient_of_variation, minimum, maximum
* `within_campaign_diagnostics`: each reported configuration preserves the closed unit's own sample count, CV, stability review, Tukey-IQR flagged count, profile SM-clock status, resolved-diagnostic count, surprising-value flag, and Nsight Compute diagnostic flags where those fields exist upstream; all remain per campaign, in the frozen order, under source-diagnostic names

## 3. Experiment 1 — LDGSTS versus TMA

median_effective_gbps is a timing-derived effective transfer rate of a dedicated streaming HBM-to-SMEM microbenchmark: the benchmark's logical useful_bytes divided by its measured kernel time. It is not directly measured HBM/DRAM bandwidth and it is not GEMM memory traffic.

Campaign-level median timing-derived effective transfer rate (`median_effective_gbps`, a `within_campaign_derived_estimate`), summarized across the three final campaigns (GB/s). This is **not** directly measured HBM/DRAM bandwidth.

| method | stages | bif_kib | mean | median | stdev | cross-campaign cv_% | min | max | cross-campaign flag |
|---|---|---|---|---|---|---|---|---|---|
| ldgsts | 2 | 16 | 3127.104065 | 3126.771032 | 0.592718 | 0.018954 | 3126.752766 | 3127.788395 | ok |
| tma | 2 | 16 | 3038.522613 | 3037.804856 | 1.302931 | 0.042880 | 3037.736394 | 3040.026589 | ok |
| ldgsts | 2 | 32 | 5167.620524 | 5167.562212 | 1.675034 | 0.032414 | 5165.975406 | 5169.323952 | ok |
| tma | 2 | 32 | 5052.975138 | 5052.634237 | 2.802322 | 0.055459 | 5050.358861 | 5055.932315 | ok |
| ldgsts | 2 | 64 | 6949.559148 | 6951.711137 | 4.162539 | 0.059896 | 6944.761133 | 6952.205174 | ok |
| tma | 2 | 64 | 6956.278332 | 6957.011919 | 2.748446 | 0.039510 | 6953.237526 | 6958.585551 | ok |
| ldgsts | 4 | 16 | 3234.236153 | 3233.925036 | 1.017774 | 0.031469 | 3233.410247 | 3235.373174 | ok |
| tma | 4 | 16 | 2393.862588 | 2392.495583 | 2.409280 | 0.100644 | 2392.447733 | 2396.644447 | ok |
| ldgsts | 4 | 32 | 5130.640057 | 5130.378926 | 1.799227 | 0.035068 | 5128.985664 | 5132.555580 | ok |
| tma | 4 | 32 | 4674.206073 | 4673.705371 | 1.311792 | 0.028064 | 4673.218373 | 4675.694475 | ok |
| ldgsts | 4 | 64 | 7023.684148 | 7024.142369 | 0.874107 | 0.012445 | 7022.676201 | 7024.233876 | ok |
| tma | 4 | 64 | 6961.856254 | 6960.475281 | 2.913271 | 0.041846 | 6959.890271 | 6965.203210 | ok |
| ldgsts | 8 | 16 | 2001.935820 | 2002.372106 | 2.347347 | 0.117254 | 1999.400938 | 2004.034416 | ok |
| tma | 8 | 16 | 1202.043731 | 1202.306051 | 0.771506 | 0.064183 | 1201.175270 | 1202.649872 | ok |
| ldgsts | 8 | 32 | 3672.451051 | 3671.707344 | 1.739902 | 0.047377 | 3671.206601 | 3674.439207 | ok |
| tma | 8 | 32 | 2400.705734 | 2399.411189 | 2.344806 | 0.097672 | 2399.293582 | 2403.412431 | ok |
| ldgsts | 8 | 64 | 6698.140023 | 6697.222691 | 2.152280 | 0.032133 | 6696.598393 | 6700.598986 | ok |
| tma | 8 | 64 | 4799.801060 | 4801.754745 | 3.420841 | 0.071270 | 4795.851091 | 4801.797346 | ok |

TMA-to-LDGSTS ratio per identical `(stages, bytes_in_flight_kib)` pair (a `within_campaign_derived_estimate`). Above one means TMA reached the higher effective transfer rate; below one means LDGSTS did. This is a derived within-campaign ratio of two campaign-level medians, not a directly measured quantity, not a winner, and not a significance claim.

| stages | bif_kib | mean | median | stdev | min | max | campaign interpretation |
|---|---|---|---|---|---|---|---|
| 2 | 16 | 0.971673058 | 0.971552624 | 0.000534500 | 0.971209049 | 0.972257501 | ldgsts_higher |
| 2 | 32 | 0.977814618 | 0.977759731 | 0.000227474 | 0.977619610 | 0.978064513 | ldgsts_higher |
| 2 | 64 | 1.000967135 | 1.000988881 | 0.000807989 | 1.000148493 | 1.001764033 | tma_higher |
| 4 | 16 | 0.740163100 | 0.739914688 | 0.000521997 | 0.739811701 | 0.740762910 | ldgsts_higher |
| 4 | 32 | 0.911037728 | 0.910891464 | 0.000526281 | 0.910600051 | 0.911621670 | ldgsts_higher |
| 4 | 64 | 0.991197272 | 0.990935963 | 0.000538061 | 0.990839769 | 0.991816084 | ldgsts_higher |
| 8 | 16 | 0.600441513 | 0.600440871 | 0.001063271 | 0.599378564 | 0.601505105 | ldgsts_higher |
| 8 | 32 | 0.653706565 | 0.653575636 | 0.000337195 | 0.653454471 | 0.654089589 | ldgsts_higher |
| 8 | 64 | 0.716587130 | 0.716622104 | 0.000408499 | 0.716162268 | 0.716977017 | ldgsts_higher |

Earliest tested candidate saturation point per group:

* `ldgsts` stages `2`: all three campaigns report `64` KiB.
* `ldgsts` stages `4`: all three campaigns report `64` KiB.
* `ldgsts` stages `8`: all three campaigns report `64` KiB.
* `tma` stages `2`: all three campaigns report `64` KiB.
* `tma` stages `4`: all three campaigns report `64` KiB.
* `tma` stages `8`: all three campaigns report `64` KiB.

Nsight Compute HBM/DRAM traffic validation covers exactly 6 of 18 configurations. Nsight compute hbm/dram traffic validation covers exactly these six predefined cases and is never extrapolated to the other twelve configurations. For the remaining 12 configurations, **actual HBM/DRAM traffic is unavailable from the collected evidence**; only the timing-derived effective transfer rate above exists for them. The profiler-derived dram_read_ratio and hbm_classification of these six cases are kept separate from the timing-derived effective transfer rate; neither validates nor calibrates the other.

`dram_read_ratio` is a `within_campaign_derived_estimate` derived from profiler evidence (`dram__bytes_read.sum / useful_bytes`), not a raw profiler counter, and `hbm_classification` is P1.4's frozen classification of it.

| case | method | stages | bif_kib | dram_read_ratio mean | min | max | classification | diagnostic flags (c1 / c2 / c3) |
|---|---|---|---|---|---|---|---|---|
| 0 | ldgsts | 2 | 16 | 1.000040180 | 1.000008917 | 1.000095929 | HBM_VALIDATED | -- / -- / -- |
| 1 | tma | 2 | 16 | 1.000055527 | 1.000036559 | 1.000092229 | HBM_VALIDATED | -- / -- / -- |
| 2 | tma | 4 | 32 | 1.000020400 | 1.000004057 | 1.000047913 | HBM_VALIDATED | -- / -- / -- |
| 3 | ldgsts | 4 | 32 | 1.000007936 | 1.000003002 | 1.000011532 | HBM_VALIDATED | -- / -- / -- |
| 4 | ldgsts | 8 | 64 | 1.000024481 | 1.000004399 | 1.000059891 | HBM_VALIDATED | -- / -- / -- |
| 5 | tma | 8 | 64 | 1.000031372 | 1.000012706 | 1.000049859 | HBM_VALIDATED | -- / -- / -- |

## 4. Experiment 2 — BF16 UMMA throughput

median_flops_per_cycle and median_flops_per_cycle_per_sm are operation-and-cycle-derived throughputs: a validated operation count divided by the measured %clock64 cycle count. They are clock-independent, which does not make them directly measured.

Clock-independent, operation-and-cycle-derived campaign-level medians (a `within_campaign_derived_estimate`), summarized across the three final campaigns.

| method | N | depth | cta_group | FLOP/cycle mean | FLOP/cycle/SM mean | cross-campaign cv_% | cross-campaign flag |
|---|---|---|---|---|---|---|---|
| umma_1sm | 64 | 4 | 1 | 2361.317288 | 2361.317288 | 0.000000 | ok |
| umma_2sm | 64 | 4 | 2 | 2249.858925 | 1124.929462 | 0.000000 | ok |
| umma_1sm | 64 | 16 | 1 | 3721.409597 | 3721.409597 | 0.000000 | ok |
| umma_2sm | 64 | 16 | 2 | 5745.130040 | 2872.565020 | 0.000000 | ok |
| umma_1sm | 64 | 64 | 1 | 4941.666873 | 4941.666873 | 0.000000 | ok |
| umma_2sm | 64 | 64 | 2 | 9361.982704 | 4680.991352 | 0.000000 | ok |
| umma_1sm | 64 | 256 | 1 | 5321.432661 | 5321.432661 | 0.000000 | ok |
| umma_2sm | 64 | 256 | 2 | 11154.956285 | 5577.478143 | 0.000000 | ok |
| umma_1sm | 128 | 4 | 1 | 3489.025423 | 3489.025423 | 0.000000 | ok |
| umma_2sm | 128 | 4 | 2 | 4248.988482 | 2124.494241 | 0.000000 | ok |
| umma_1sm | 128 | 16 | 1 | 6651.960672 | 6651.960672 | 0.000000 | ok |
| umma_2sm | 128 | 16 | 2 | 9531.859191 | 4765.929595 | 0.000000 | ok |
| umma_1sm | 128 | 64 | 1 | 7706.483983 | 7706.483983 | 0.000000 | ok |
| umma_2sm | 128 | 64 | 2 | 13813.777413 | 6906.888706 | 0.000000 | ok |
| umma_1sm | 128 | 256 | 1 | 8064.974680 | 8064.974680 | 0.000000 | ok |
| umma_2sm | 128 | 256 | 2 | 15655.763339 | 7827.881669 | 0.000000 | ok |
| umma_1sm | 256 | 4 | 1 | 5599.318625 | 5599.318625 | 0.000000 | ok |
| umma_2sm | 256 | 4 | 2 | 6802.734196 | 3401.367098 | 0.000000 | ok |
| umma_1sm | 256 | 16 | 1 | 7007.810118 | 7007.810118 | 0.000000 | ok |
| umma_2sm | 256 | 16 | 2 | 12051.978652 | 6025.989326 | 0.000000 | ok |
| umma_1sm | 256 | 64 | 1 | 7840.687983 | 7840.687983 | 0.000000 | ok |
| umma_2sm | 256 | 64 | 2 | 15014.614718 | 7507.307359 | 0.000000 | ok |
| umma_1sm | 256 | 256 | 1 | 8101.277950 | 8101.277950 | 0.000000 | ok |
| umma_2sm | 256 | 256 | 2 | 16064.781112 | 8032.390556 | 0.000000 | ok |

1-SM/2-SM comparison. Speedup and scaling efficiency are derived within-campaign quantities (a `within_campaign_derived_estimate`) summarized across the three campaigns; values outside `[0, 100]` are preserved unclamped and keep the closed unit's surprising-value diagnostic.

| N | depth | speedup mean | min | max | efficiency % mean | min | max | surprising flags |
|---|---|---|---|---|---|---|---|---|
| 64 | 4 | 0.952798227 | 0.952798227 | 0.952798227 | 47.639911 | 47.639911 | 47.639911 | False,False,False |
| 64 | 16 | 1.543804811 | 1.543804811 | 1.543804811 | 77.190241 | 77.190241 | 77.190241 | False,False,False |
| 64 | 64 | 1.894498950 | 1.894498950 | 1.894498950 | 94.724948 | 94.724948 | 94.724948 | False,False,False |
| 64 | 256 | 2.096231785 | 2.096231785 | 2.096231785 | 104.811589 | 104.811589 | 104.811589 | True,True,True |
| 128 | 4 | 1.217815283 | 1.217815283 | 1.217815283 | 60.890764 | 60.890764 | 60.890764 | False,False,False |
| 128 | 16 | 1.432939799 | 1.432939799 | 1.432939799 | 71.646990 | 71.646990 | 71.646990 | False,False,False |
| 128 | 64 | 1.792487656 | 1.792487656 | 1.792487656 | 89.624383 | 89.624383 | 89.624383 | False,False,False |
| 128 | 256 | 1.941204277 | 1.941204277 | 1.941204277 | 97.060214 | 97.060214 | 97.060214 | False,False,False |
| 256 | 4 | 1.214921788 | 1.214921788 | 1.214921788 | 60.746089 | 60.746089 | 60.746089 | False,False,False |
| 256 | 16 | 1.719792410 | 1.719792410 | 1.719792410 | 85.989621 | 85.989621 | 85.989621 | False,False,False |
| 256 | 64 | 1.914961385 | 1.914961385 | 1.914961385 | 95.748069 | 95.748069 | 95.748069 | False,False,False |
| 256 | 256 | 1.982993450 | 1.982993450 | 1.982993450 | 99.149673 | 99.149673 | 99.149673 | False,False,False |

Depth saturation candidate per group:

* `umma_1sm` N `64`: all three campaigns select depth `256`.
* `umma_1sm` N `128`: all three campaigns select depth `256`.
* `umma_1sm` N `256`: all three campaigns select depth `256`.
* `umma_2sm` N `64`: all three campaigns select depth `256`.
* `umma_2sm` N `128`: all three campaigns select depth `256`.
* `umma_2sm` N `256`: all three campaigns select depth `256`.

Empirical per-SM BF16 Tensor Core ceiling candidate (a modeled clock conversion of a one-/two-SM microbenchmark result: the candidate is selected in clock-independent FLOP/cycle/SM space and only then multiplied by that same configuration's own profiled SM clock. It is never a theoretical architectural peak and never a measured whole-device throughput):

* All three campaigns select `umma_1sm` N `256` depth `256` (`23_umma_1sm_n256_d256`).
* Modeled TFLOP/s/SM (a `modeled_estimate`) across the three campaigns: mean `16.358109340`, median `16.356605557`, sample stdev `0.002910033`, min `16.356258873`, max `16.361463590`.
* Device-wide estimate: **unavailable**. Reason: at least one final campaign contains no valid device-wide estimate, so no whole-GPU throughput is reported. No SM count is imported from an external specification, hard-coded, or inferred, so no whole-GPU peak is reported.

## 5. Experiment 3 — CuTe DSL versus cuBLASLt

kernel_time_ms is the measured source observation (CUDA-event kernel time divided by the measured iteration count, after correctness passed). tflops, the cuBLASLt-relative ratio, the signed gap, and the best-variant selection are all derived within-campaign quantities; no GEMM kernel was profiled.

Five frozen shapes x 4 frozen candidates, 20 source rows per campaign, cache mode `hot`. Source rows carry `run_kind=smoke` and `publishable=false`. Beating cuBLASLt is not a success criterion.

### Shape `4096x4096x4096x1`

| candidate | method | kernel_time_ms mean | TFLOP/s mean | TFLOP/s stdev | ratio vs cuBLASLt mean | gap % mean | gap % min | gap % max |
|---|---|---|---|---|---|---|---|---|
| nonpersistent_1cta | cutedsl | 0.144531 | 950.930070 | 1.037207 | 0.542901653 | 45.709835 | 45.547432 | 45.830381 |
| persistent_1cta | cutedsl | 0.093982 | 1462.411761 | 5.371745 | 0.834916584 | 16.508342 | 16.023517 | 16.768464 |
| persistent_2cta | cutedsl | 0.082526 | 1665.416315 | 5.424877 | 0.950811624 | 4.918838 | 4.571100 | 5.145407 |
| heuristic_first_supported | cublaslt | 0.078466 | 1751.575992 | 3.831810 | 1.000000000 | 0.000000 | 0.000000 | 0.000000 |

Stable best CuTe DSL variant: `persistent_2cta`.

### Shape `8192x8192x8192x1`

| candidate | method | kernel_time_ms mean | TFLOP/s mean | TFLOP/s stdev | ratio vs cuBLASLt mean | gap % mean | gap % min | gap % max |
|---|---|---|---|---|---|---|---|---|
| nonpersistent_1cta | cutedsl | 1.229272 | 894.441722 | 0.271137 | 0.422170501 | 57.782950 | 57.772744 | 57.797333 |
| persistent_1cta | cutedsl | 0.883862 | 1243.985198 | 0.492370 | 0.587152702 | 41.284730 | 41.257056 | 41.317543 |
| persistent_2cta | cutedsl | 0.761616 | 1443.659446 | 2.661194 | 0.681397636 | 31.860237 | 31.768186 | 32.008237 |
| heuristic_first_supported | cublaslt | 0.518962 | 2118.674163 | 0.399950 | 1.000000000 | 0.000000 | 0.000000 | 0.000000 |

Stable best CuTe DSL variant: `persistent_2cta`.

### Shape `16384x512x4096x1`

| candidate | method | kernel_time_ms mean | TFLOP/s mean | TFLOP/s stdev | ratio vs cuBLASLt mean | gap % mean | gap % min | gap % max |
|---|---|---|---|---|---|---|---|---|
| nonpersistent_1cta | cutedsl | 0.113899 | 603.366079 | 4.965062 | 0.421137884 | 57.886212 | 57.640775 | 58.297149 |
| persistent_1cta | cutedsl | 0.088334 | 777.986520 | 6.365032 | 0.543019902 | 45.698010 | 45.382709 | 46.231967 |
| persistent_2cta | cutedsl | 0.084723 | 811.123455 | 4.633565 | 0.566148410 | 43.385159 | 43.176997 | 43.780722 |
| heuristic_first_supported | cublaslt | 0.047965 | 1432.706728 | 0.833034 | 1.000000000 | 0.000000 | 0.000000 | 0.000000 |

Stable best CuTe DSL variant: `persistent_2cta`.

### Shape `32768x512x4096x1`

| candidate | method | kernel_time_ms mean | TFLOP/s mean | TFLOP/s stdev | ratio vs cuBLASLt mean | gap % mean | gap % min | gap % max |
|---|---|---|---|---|---|---|---|---|
| nonpersistent_1cta | cutedsl | 0.222304 | 618.254297 | 2.458771 | 0.410956947 | 58.904305 | 58.841836 | 59.023420 |
| persistent_1cta | cutedsl | 0.186242 | 737.968072 | 3.263370 | 0.490531794 | 50.946820 | 50.772067 | 51.123990 |
| persistent_2cta | cutedsl | 0.181746 | 756.220595 | 2.754861 | 0.502664635 | 49.733536 | 49.582576 | 49.870254 |
| heuristic_first_supported | cublaslt | 0.091357 | 1504.422838 | 2.736202 | 1.000000000 | 0.000000 | 0.000000 | 0.000000 |

Stable best CuTe DSL variant: `persistent_2cta`.

### Shape `512x16384x4096x1`

| candidate | method | kernel_time_ms mean | TFLOP/s mean | TFLOP/s stdev | ratio vs cuBLASLt mean | gap % mean | gap % min | gap % max |
|---|---|---|---|---|---|---|---|---|
| nonpersistent_1cta | cutedsl | 0.090559 | 758.837748 | 0.864650 | 0.507555849 | 49.244415 | 48.969782 | 49.437566 |
| persistent_1cta | cutedsl | 0.062188 | 1105.036110 | 2.387918 | 0.739118214 | 26.088179 | 25.660303 | 26.622172 |
| persistent_2cta | cutedsl | 0.054434 | 1262.501446 | 11.312691 | 0.844416942 | 15.558306 | 15.011245 | 15.974165 |
| heuristic_first_supported | cublaslt | 0.045964 | 1495.105892 | 7.559372 | 1.000000000 | 0.000000 | 0.000000 | 0.000000 |

Stable best CuTe DSL variant: `persistent_2cta`.

## 6. Preserved source diagnostics and review conditions

Every terminal diagnostic enumerated by this P4.3 contract is preserved per campaign, in the frozen campaign order, in the CSV tables and in `integrated_summary.json`: sample counts, relevant within-campaign CVs and stability reviews, Tukey-IQR flagged counts, per-case SM-clock and resolved-profiler-metric diagnostics, scaling flags, NCU coverage, HBM classification, and NCU diagnostic flags. Nothing below excluded a campaign, removed an observation, or changed a value.

### Source diagnostic warnings (71)

* campaign `20260817T110330Z` (campaign_1_value): bytes_in_flight_kib=16, effect=diagnostic_only_no_observation_removed, method=ldgsts, section=experiment_1_configuration, stages=2, within_campaign_iqr_flagged_count=1
* campaign `20260817T111310Z` (campaign_2_value): bytes_in_flight_kib=16, effect=diagnostic_only_no_observation_removed, method=ldgsts, section=experiment_1_configuration, stages=2, within_campaign_iqr_flagged_count=1
* campaign `20260817T111310Z` (campaign_2_value): bytes_in_flight_kib=16, effect=diagnostic_only_no_observation_removed, method=tma, section=experiment_1_configuration, stages=2, within_campaign_iqr_flagged_count=1
* campaign `20260817T112011Z` (campaign_3_value): bytes_in_flight_kib=16, effect=diagnostic_only_no_observation_removed, method=tma, section=experiment_1_configuration, stages=2, within_campaign_iqr_flagged_count=1
* campaign `20260817T110330Z` (campaign_1_value): bytes_in_flight_kib=32, effect=diagnostic_only_no_observation_removed, method=tma, section=experiment_1_configuration, stages=2, within_campaign_iqr_flagged_count=3
* campaign `20260817T110330Z` (campaign_1_value): bytes_in_flight_kib=64, effect=diagnostic_only_no_observation_removed, method=ldgsts, section=experiment_1_configuration, stages=2, within_campaign_iqr_flagged_count=3
* campaign `20260817T112011Z` (campaign_3_value): bytes_in_flight_kib=64, effect=diagnostic_only_no_observation_removed, method=ldgsts, section=experiment_1_configuration, stages=2, within_campaign_iqr_flagged_count=1
* campaign `20260817T110330Z` (campaign_1_value): bytes_in_flight_kib=64, effect=diagnostic_only_no_observation_removed, method=tma, section=experiment_1_configuration, stages=2, within_campaign_iqr_flagged_count=1
* campaign `20260817T112011Z` (campaign_3_value): bytes_in_flight_kib=64, effect=diagnostic_only_no_observation_removed, method=tma, section=experiment_1_configuration, stages=2, within_campaign_iqr_flagged_count=1
* campaign `20260817T111310Z` (campaign_2_value): bytes_in_flight_kib=16, effect=diagnostic_only_no_observation_removed, method=ldgsts, section=experiment_1_configuration, stages=4, within_campaign_iqr_flagged_count=1
* campaign `20260817T112011Z` (campaign_3_value): bytes_in_flight_kib=16, effect=diagnostic_only_no_observation_removed, method=ldgsts, section=experiment_1_configuration, stages=4, within_campaign_iqr_flagged_count=2
* campaign `20260817T110330Z` (campaign_1_value): bytes_in_flight_kib=32, effect=diagnostic_only_no_observation_removed, method=ldgsts, section=experiment_1_configuration, stages=4, within_campaign_iqr_flagged_count=2
* campaign `20260817T110330Z` (campaign_1_value): bytes_in_flight_kib=32, effect=diagnostic_only_no_observation_removed, method=tma, section=experiment_1_configuration, stages=4, within_campaign_iqr_flagged_count=1
* campaign `20260817T111310Z` (campaign_2_value): bytes_in_flight_kib=32, effect=diagnostic_only_no_observation_removed, method=tma, section=experiment_1_configuration, stages=4, within_campaign_iqr_flagged_count=1
* campaign `20260817T112011Z` (campaign_3_value): bytes_in_flight_kib=32, effect=diagnostic_only_no_observation_removed, method=tma, section=experiment_1_configuration, stages=4, within_campaign_iqr_flagged_count=1
* campaign `20260817T110330Z` (campaign_1_value): bytes_in_flight_kib=64, effect=diagnostic_only_no_observation_removed, method=ldgsts, section=experiment_1_configuration, stages=4, within_campaign_iqr_flagged_count=5
* campaign `20260817T110330Z` (campaign_1_value): bytes_in_flight_kib=64, effect=diagnostic_only_no_observation_removed, method=tma, section=experiment_1_configuration, stages=4, within_campaign_iqr_flagged_count=1
* campaign `20260817T110330Z` (campaign_1_value): bytes_in_flight_kib=16, effect=diagnostic_only_no_observation_removed, method=tma, section=experiment_1_configuration, stages=8, within_campaign_iqr_flagged_count=2
* campaign `20260817T112011Z` (campaign_3_value): bytes_in_flight_kib=16, effect=diagnostic_only_no_observation_removed, method=tma, section=experiment_1_configuration, stages=8, within_campaign_iqr_flagged_count=1
* campaign `20260817T110330Z` (campaign_1_value): bytes_in_flight_kib=32, method=ldgsts, section=experiment_1_configuration, stages=8, within_campaign_stability_review=REVIEW
* campaign `20260817T110330Z` (campaign_1_value): bytes_in_flight_kib=32, effect=diagnostic_only_no_observation_removed, method=ldgsts, section=experiment_1_configuration, stages=8, within_campaign_iqr_flagged_count=1
* campaign `20260817T110330Z` (campaign_1_value): bytes_in_flight_kib=64, effect=diagnostic_only_no_observation_removed, method=ldgsts, section=experiment_1_configuration, stages=8, within_campaign_iqr_flagged_count=2
* campaign `20260817T110330Z` (campaign_1_value): bytes_in_flight_kib=64, effect=diagnostic_only_no_observation_removed, method=tma, section=experiment_1_configuration, stages=8, within_campaign_iqr_flagged_count=1
* campaign `20260817T112011Z` (campaign_3_value): bytes_in_flight_kib=64, effect=diagnostic_only_no_observation_removed, method=tma, section=experiment_1_configuration, stages=8, within_campaign_iqr_flagged_count=1
* campaign `20260817T110330Z` (campaign_1_value): depth=4, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=1
* campaign `20260817T110330Z` (campaign_1_value): depth=4, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=1
* campaign `20260817T111310Z` (campaign_2_value): depth=4, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=1
* campaign `20260817T111310Z` (campaign_2_value): depth=4, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=1
* campaign `20260817T112011Z` (campaign_3_value): depth=4, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=1
* campaign `20260817T112011Z` (campaign_3_value): depth=4, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=1
* campaign `20260817T110330Z` (campaign_1_value): depth=16, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=1
* campaign `20260817T110330Z` (campaign_1_value): depth=16, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=1
* campaign `20260817T111310Z` (campaign_2_value): depth=16, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=1
* campaign `20260817T111310Z` (campaign_2_value): depth=16, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=1
* campaign `20260817T112011Z` (campaign_3_value): depth=16, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=1
* campaign `20260817T112011Z` (campaign_3_value): depth=16, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=1
* campaign `20260817T110330Z` (campaign_1_value): depth=64, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=1
* campaign `20260817T110330Z` (campaign_1_value): depth=64, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=1
* campaign `20260817T111310Z` (campaign_2_value): depth=64, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=1
* campaign `20260817T111310Z` (campaign_2_value): depth=64, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=1
* campaign `20260817T112011Z` (campaign_3_value): depth=64, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=1
* campaign `20260817T112011Z` (campaign_3_value): depth=64, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=1
* campaign `20260817T110330Z` (campaign_1_value): depth=256, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=7
* campaign `20260817T110330Z` (campaign_1_value): depth=256, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=64, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=7
* campaign `20260817T110330Z` (campaign_1_value): depth=4, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=4
* campaign `20260817T110330Z` (campaign_1_value): depth=4, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=4
* campaign `20260817T111310Z` (campaign_2_value): depth=4, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=4
* campaign `20260817T111310Z` (campaign_2_value): depth=4, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=4
* campaign `20260817T110330Z` (campaign_1_value): depth=16, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=4
* campaign `20260817T110330Z` (campaign_1_value): depth=16, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=4
* campaign `20260817T111310Z` (campaign_2_value): depth=16, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=4
* campaign `20260817T111310Z` (campaign_2_value): depth=16, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=4
* campaign `20260817T112011Z` (campaign_3_value): depth=16, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=4
* campaign `20260817T112011Z` (campaign_3_value): depth=16, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=4
* campaign `20260817T110330Z` (campaign_1_value): depth=64, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=4
* campaign `20260817T110330Z` (campaign_1_value): depth=64, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=4
* campaign `20260817T111310Z` (campaign_2_value): depth=64, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=4
* campaign `20260817T111310Z` (campaign_2_value): depth=64, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=4
* campaign `20260817T112011Z` (campaign_3_value): depth=64, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=4
* campaign `20260817T112011Z` (campaign_3_value): depth=64, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=4
* campaign `20260817T111310Z` (campaign_2_value): depth=256, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=7
* campaign `20260817T111310Z` (campaign_2_value): depth=256, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=7
* campaign `20260817T112011Z` (campaign_3_value): depth=256, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=7
* campaign `20260817T112011Z` (campaign_3_value): depth=256, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=128, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=7
* campaign `20260817T111310Z` (campaign_2_value): depth=256, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=256, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=7
* campaign `20260817T111310Z` (campaign_2_value): depth=256, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=256, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=7
* campaign `20260817T112011Z` (campaign_3_value): depth=256, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=256, section=experiment_2_configuration, within_campaign_flops_per_cycle_iqr_flagged_count=7
* campaign `20260817T112011Z` (campaign_3_value): depth=256, effect=diagnostic_only_no_observation_removed, method=umma_2sm, n=256, section=experiment_2_configuration, within_campaign_flops_per_cycle_per_sm_iqr_flagged_count=7
* campaign `20260817T110330Z` (campaign_1_value): depth=256, n=64, section=experiment_2_scaling, surprising_value_flag=True
* campaign `20260817T111310Z` (campaign_2_value): depth=256, n=64, section=experiment_2_scaling, surprising_value_flag=True
* campaign `20260817T112011Z` (campaign_3_value): depth=256, n=64, section=experiment_2_scaling, surprising_value_flag=True

### Cross-campaign variability review conditions (0)

A cross-campaign coefficient of variation above 5.0% is a **review diagnostic only**. It is a different quantity from a closed unit's own within-campaign stability review, it never replaces one, and the two may disagree.

* None: no reported quantity exceeded the cross-campaign review threshold.

## 7. Integrated interpretation

> How do HBM-to-SMEM data movement and fifth-generation Tensor Core throughput constrain BF16 GEMM performance on NVIDIA GB300, and how closely can the CuTe DSL implementation approach cuBLASLt?

Each quantity below is placed in exactly one evidence class. A derived or modeled quantity is never described as directly measured, and a campaign-level median is never presented as an individual raw observation.

### Measured source observations

* per-repetition CUDA-event kernel time of the LDGSTS and TMA streaming microbenchmark launches (P1.1/P1.2), which is the timing input of every effective-rate estimate below
* Nsight Compute's dram__bytes_read.sum for exactly 6 of 18 memory configurations, together with the validated useful_bytes of those same cases
* the raw %clock64 elapsed-cycle counts of the BF16 UMMA launches (P2.1/P2.2)
* the per-configuration Nsight Compute SM-clock readings of all 24 profiled UMMA configurations (P2.4)
* CUDA-event kernel time of the five frozen BF16 GEMM shapes for three CuTe DSL execution variants and one cuBLASLt baseline, hot-cache, recorded only after correctness passed (P3.5 section 7)

### Within-campaign derived estimates

* median_effective_gbps: a timing-derived effective transfer rate. Each repetition divides the benchmark's logical useful_bytes by its own measured kernel time, and the campaign reports the median of 30 such values. P1.1 and P1.2 label this effective copy bandwidth and state explicitly that it is not HBM/DRAM bandwidth
* tma_to_ldgsts_ratio: a derived within-campaign ratio of two campaign-level medians at one identical configuration
* dram_read_ratio and hbm_classification: derived from profiler evidence, not raw profiler counters -- dram__bytes_read.sum divided by validated useful_bytes, then classified against P1.4's frozen 0.90 rule
* median_flops_per_cycle and median_flops_per_cycle_per_sm: operation-and-cycle-derived throughputs, a validated 2*M*N*K*depth*iterations operation count divided by the measured elapsed cycles. They are clock-independent, which does not make them directly measured
* speedup_2sm_over_1sm and scaling_efficiency_percent: derived inside each campaign from two campaign-level medians
* GEMM tflops: the exact 2*M*N*K operation count divided by the measured kernel time
* throughput_ratio_vs_cublaslt and gap_to_cublaslt_pct: derived inside each campaign against that campaign's own cuBLASLt baseline row
* the earliest-tested candidate saturation selections and the best CuTe DSL variant per shape: derived within-campaign selections over the tested grid, never universal thresholds

### Cross-campaign descriptive statistics

* every mean, median, sample standard deviation (n-1), coefficient of variation where meaningful, minimum, and maximum in these artifacts is computed by P4.3 over exactly 3 campaign-level values, one per final campaign
* cross_campaign_cv_percent and cross_campaign_cv_review_flag describe agreement between campaigns only; they are never a within-campaign stability diagnostic and never replace one
* the cross-campaign consensus of a saturation candidate, a ceiling selection, or a best variant is a statement about agreement between the three campaigns, not a new measurement and not a majority vote

### Modeled estimates

* estimated_tflops_per_sm: a modeled clock conversion of a microbenchmark result. The candidate is selected in clock-independent FLOP/cycle/SM space and only then multiplied by that same configuration's own profiled SM clock. It is a one-/two-SM empirical microbenchmark ceiling candidate, never an architectural peak
* estimated_device_equivalent_tflops, when it is available at all: the modeled per-SM estimate multiplied by a validated SM count, which is a whole-device extrapolation and never a measured whole-GPU throughput

### Interpretations

* the LDGSTS/TMA benchmark is a dedicated streaming HBM-to-SMEM data-movement microbenchmark. It does not directly measure the memory traffic a GEMM kernel generates, and its effective-rate estimates are consistent with, but not evidence of, GEMM-level memory behaviour
* where the TMA-to-LDGSTS ratio stays close to one across all three campaigns, the evidence is consistent with the two paths reaching a similar effective transfer rate at that configuration
* the distance between the best CuTe DSL variant and cuBLASLt per shape is a derived difference between measured kernel times; the collected evidence does not attribute it to any cause

### Unavailable from the collected evidence

* actual HBM/DRAM traffic for the 12 memory configurations outside the frozen 6-case Nsight Compute plan: no profiler evidence was collected for them, and the six profiled cases are never extrapolated to them
* whether any specific GEMM shape is HBM-bound, Tensor-Core-bound, scheduler-bound, or limited by another implementation cost: P3.5 collected no Nsight Compute profile of a GEMM kernel, so no bottleneck attribution is made
* a numerical roofline, an architectural peak, or an arithmetic-intensity placement: the streaming microbenchmark and the GEMM measurements are not dimensionally comparable evidence of the same workload, and no compulsory-byte model was validated
* a cold-cache GEMM result: every GEMM measurement is hot-cache by construction
* a whole-device BF16 throughput figure whenever the modeled device-wide estimate is unavailable

### Answer

* **hbm_to_smem** — derived: memory_paths.csv reports each campaign's median timing-derived effective transfer rate for both equivalent HBM-to-SMEM paths over the frozen grid, and the derived within-campaign TMA-to-LDGSTS ratios beside them. Profiler-derived DRAM traffic exists for six configurations only; for the other twelve, actual HBM traffic is unavailable. The saturation candidate is reported per group, and only as a cross-campaign consensus when all three campaigns agree, never as a universal HBM saturation threshold
* **tensor_core** — derived: umma_throughput.csv reports the clock-independent, operation-and-cycle-derived FLOP/cycle and FLOP/cycle/SM values and each campaign's derived 1-SM/2-SM scaling. The per-SM ceiling candidate is a modeled clock conversion of a one-/two-SM microbenchmark and is summarized across campaigns only when all three select the same configuration
* **cutedsl_versus_cublaslt** — measured input plus derived comparison: per shape and candidate, gemm_comparison.csv reports the campaign-level measured kernel time and the derived TFLOP/s, cuBLASLt-relative ratio, and signed gap, each summarized across the three campaigns. The stable best CuTe DSL variant per shape is {"16384x512x4096x1": "persistent_2cta", "32768x512x4096x1": "persistent_2cta", "4096x4096x4096x1": "persistent_2cta", "512x16384x4096x1": "persistent_2cta", "8192x8192x8192x1": "persistent_2cta"} (null means the three campaigns did not agree)
* **constraint_attribution** — unavailable from the collected evidence: no GEMM-level profile exists, so the GEMM throughput is not attributed to the memory path, the Tensor Core ceiling, the scheduler, or any other single cost

## 8. Limitations

* the independent replicate is one complete final campaign; the cross-campaign sample size is 3, which is small and supports descriptive statistics only
* no p-value, significance claim, or cross-campaign confidence interval is computed; the within-campaign confidence intervals the closed units recorded remain provenance and are never reinterpreted as cross-campaign intervals
* no observation and no campaign was removed; no outlier filter was applied; a cross-campaign coefficient of variation above 5.0% is a review diagnostic only. It never excludes a campaign, never changes a result, and never replaces a closed unit's own within-campaign stability review
* within-campaign and cross-campaign variability are different quantities and are reported in separate, differently named fields; they may disagree, and a disagreement is reported rather than resolved
* a coefficient of variation is not computed for signed or zero-centred quantities such as gap_to_cublaslt_pct
* median_effective_gbps is a timing-derived effective transfer rate of a streaming microbenchmark -- the benchmark's logical useful_bytes divided by its measured kernel time -- and is explicitly not directly measured HBM/DRAM bandwidth
* profiler-derived HBM/DRAM traffic exists for exactly 6 of 18 memory configurations; for the other 12, actual HBM traffic is unavailable from the collected evidence and the six profiled cases are never extrapolated to them
* the dram_read_ratio and hbm_classification of those six cases are derived from profiler evidence, not raw profiler counters, and are kept separate from the timing-derived effective-rate metric
* the streaming memory microbenchmark does not directly measure the memory traffic a GEMM kernel generates
* flops_per_cycle and flops_per_cycle_per_sm are derived from validated operation counts and measured cycles; being clock-independent does not make them directly measured
* the BF16 UMMA ceiling is a modeled clock conversion of a one-/two-SM empirical microbenchmark result; it is not an architectural peak and not a measured whole-device throughput
* no SM count is imported from an external specification, hard-coded, or inferred; without validated agreeing SM-count evidence the modeled device-wide estimate stays structurally unavailable
* every GEMM measurement is hot-cache (hot) and must not be described as a cold-cache workload
* P3.5 collected no Nsight Compute profile of a GEMM kernel, so no GEMM bottleneck attribution, roofline placement, architectural peak, or arithmetic-intensity classification is made anywhere in these artifacts
* the source GEMM rows are run_kind=smoke evidence captured by the campaign; they carry publishable=false, and their kernel times are measured inputs to a comparison, not a validated publication-grade benchmark
* the sweep order inside each closed unit is fixed and non-randomized, a limitation the closed units already recorded
* the accepted pilot is excluded from every statistic here; it qualifies the orchestration path only
* these nine artifacts are a candidate bundle: they are bound together by analysis_manifest.json, which is the authoritative provenance envelope. A detached CSV or SVG is not a standalone provenance envelope and must be distributed with the manifest
* no candidate artifact self-authorizes publication: candidate bytes remain publishable=false and publication authority, if later granted, exists only in a separate attestation that binds the exact manifest and sibling hashes after the required verification and independent review

## 9. Candidate status and the acceptance workflow

These artifacts are a **candidate bundle**, not an accepted result:

```text
publishable        = false
publication_state  = immutable_candidate_requires_external_attestation
analysis_code_commit = dba7bfc5aca7f750ba9e6f8ac4e26b26e540f711
```

The complete required lifecycle is, in this exact order:

1. an independently audited, clean analysis-code commit
2. candidate production analysis from exactly that commit
3. byte-for-byte verification of the candidate bundle
4. independent scientific and output review of the complete bundle
5. an external acceptance attestation at src/phase4/P4_3_ACCEPTANCE.json
6. a final documentation and status commit

Candidate bytes deliberately make no claim about which external lifecycle steps have occurred. Acceptance is an external attestation at `src/phase4/P4_3_ACCEPTANCE.json` that binds this bundle's `analysis_manifest.json` hash; it is never written by the analyzer, and no candidate artifact is ever promoted, rewritten, or deleted to record workflow progress.

`publishable = false` inside the immutable candidate. publishable=false; publication_state=immutable_candidate_requires_external_attestation; candidate bytes do not record time-dependent audit, verification, review, or attestation progress; publication authority requires a separate hash-bound external acceptance attestation.
