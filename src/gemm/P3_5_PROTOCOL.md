# P3.5 — Five shapes and comparison (frozen protocol)

Status: `P3.5 = YES / NO / NO` (Implemented / Audited / Verified on GB300).
The author's own GPU-free checks are **not** an independent audit, and GPU-free
checks are **not** GB300 verification. Section 12 records exactly what was and
was not run.

## 1. Purpose and scope

P3.2 established one CuTe DSL execution variant at the first of the five final
shapes. P3.3 established the cuBLASLt baseline for exactly that geometry. P3.4
added the two remaining CuTe DSL execution variants, still at that one shape.
P3.5 extends the same already verified infrastructure to **all five** final
Experiment 3 shapes and performs the first explicit, descriptive comparison
among the four candidates.

P3.5 must prove that all four candidates execute correctly and comparably on all
five frozen shapes, and must compute deterministic steady-state comparison
fields from those runs.

**P3.5 creates no publishable performance result.** It is infrastructure and
functional-comparison evidence, not an experimental campaign. Every emitted row
carries `publishable=false`. Pilot and final campaigns, statistical treatment,
experiment integration, and final interpretation are Phase 4 work. Nothing in
this unit claims — or may be read as claiming — that a CuTe DSL variant
approaches or beats cuBLASLt. The comparison fields are arithmetic, not a
conclusion.

## 2. The exact five-shape table

The five and only five `(M,N,K,L)` shapes, in this exact order:

| # | M | N | K | L | `shape_id` | `flop_count` = 2·M·N·K |
|---|---|---|---|---|-----------|------------------------|
| 1 | 4096 | 4096 | 4096 | 1 | `4096x4096x4096x1` | 137,438,953,472 |
| 2 | 8192 | 8192 | 8192 | 1 | `8192x8192x8192x1` | 1,099,511,627,776 |
| 3 | 16384 | 512 | 4096 | 1 | `16384x512x4096x1` | 68,719,476,736 |
| 4 | 32768 | 512 | 4096 | 1 | `32768x512x4096x1` | 137,438,953,472 |
| 5 | 512 | 16384 | 4096 | 1 | `512x16384x4096x1` | 68,719,476,736 |

Shape 1 is the same shape P3.2, P3.3, and P3.4 used, so the new unit remains
directly comparable with the closed ones.

**No arbitrary shape may be supplied through the CLI, an environment variable, a
configuration file, or an input CSV.** The table is frozen in
`src/gemm/gemm_comparison.py` and, independently, in the C bridge
`src/gemm/cublaslt_bridge_p35.cu`. The bridge exposes its own allowlist through
`p35_shape_count()` / `p35_shape_at()`, and the wrapper reads it back and
requires the two to be identical before any measurement. A geometry that is not
in the allowlist never reaches a cuBLASLt descriptor, a heuristic query, or a
launch.

## 3. The exact four-candidate table

| # | Method | Variant | Upstream class | Scheduler | MMA tiler | Cluster | 2-CTA |
|---|--------|---------|----------------|-----------|-----------|---------|-------|
| 1 | `cutedsl` | `nonpersistent_1cta` | `DenseGemmKernel` | `nonpersistent` | `(128,128)` | `(1,1)` | `false` |
| 2 | `cutedsl` | `persistent_1cta` | `PersistentDenseGemmKernel` | `static_persistent` | `(128,128)` | `(1,1)` | `false` |
| 3 | `cutedsl` | `persistent_2cta` | `PersistentDenseGemmKernel` | `static_persistent` | `(256,128)` | `(2,1)` | `true` |
| 4 | `cublaslt` | `heuristic_first_supported` | — | — | — | — | — |

They always run in exactly this order, for every shape. There is no fifth
candidate, no autotuning, no candidate search, no alternative tile or cluster,
no dynamic scheduler selection, no split-K experiment, no layout or dtype
variant, and no fallback kernel.

The three CuTe DSL rows are byte-for-byte the closed P3.4 table. The 2-CTA row
keeps `tiler_M / cluster_M = 256 / 2 = 128`, so each participating CTA retains
the local M extent of 128 that P2.2 measured and that NVIDIA's own persistent
example documents for `use_2cta_instrs=True`.

## 4. Frozen scientific contract

```text
Operation:       C[m,n,l] = sum_k A[m,k,l] * B[n,k,l]   (C = A x B^T)
Input dtype:     BF16 x BF16
Accumulation:    FP32
Output dtype:    FP32
A major:         k
B major:         k
C major:         n
L:               1
TMA loads:       enabled   (CuTe DSL candidates)
TMA store:       enabled   (CuTe DSL candidates)
Epilogue:        identity
Seed:            1111
Cache model:     hot, reused operands
Target:          sm_103a
Reference:       untimed PyTorch CUDA FP32 IEEE oracle
atol:            1e-1
rtol:            1e-5
```

None of it is reachable from a CLI argument, an environment variable, or a
configuration file.

### 4.1 cuBLASLt policy (exactly the closed P3.3 policy)

```text
API:                         direct cublasLtMatmul
workspace limit:             67,108,864 bytes (64 MiB)
heuristic results requested: 32
search mode:                 CUBLASLT_SEARCH_BEST_FIT
selection:                   first supported heuristic result
validation:                  cublasLtMatmulAlgoCheck
fallback GEMM API:           forbidden
autotuning by execution:     forbidden
```

Descriptors: A row-major `M × K` with `lda = K`, B row-major `N × K` with
`ldb = K`, C and D row-major `M × N` with `ldc = ldd = N`;
`transa = CUBLAS_OP_N`, `transb = CUBLAS_OP_T`; `CUDA_R_16BF` inputs,
`CUDA_R_32F` output; `CUBLAS_COMPUTE_32F`; `CUDA_R_32F` scale;
`CUBLASLT_POINTER_MODE_HOST`; `CUBLASLT_EPILOGUE_DEFAULT`; `alpha = 1`;
`beta = 0`; no bias.

The leading dimensions are **derived** from the validated shape on both sides of
the ABI — never supplied by the caller — and the two derivations must agree.

A different supported algorithm may naturally be selected for each shape. **The
selection policy never changes.** No candidate algorithm is ever executed,
timed, compared, or ranked in order to choose it.

## 5. Pinned sources and pins

P3.5 uses the same two already pinned official CUTLASS sources P3.4 uses and the
same cuBLASLt library that already ships in the pinned CUDA 13.1 image.
Therefore it needs **no new external pin, dependency, image package, or Docker
change**, and it adds **no key** to `VERSIONS.env` or `PHASE3_VERSIONS.env`.

```text
Repository: NVIDIA/cutlass @ e05f953a5b3d38adc240df2ff928e0421c2abba3 (BSD-3-Clause)
Non-persistent: .../dense_gemm/dense_gemm.py             (CUTEDSL_P31_EXAMPLE_*)
Persistent:     .../dense_gemm/dense_gemm_persistent.py  (CUTEDSL_P34_PERSISTENT_EXAMPLE_*)
```

Both files are loaded read-only and in place from `/opt/cutlass` under private
module names, after HEAD, checkout cleanliness, regular-file identity, Git blob,
and SHA-256 are verified for each. The whole verification is repeated after
provenance collection and the two observations must agree. Neither upstream
`run()` nor either upstream benchmarking helper is ever called: P3.5 owns every
timer. Nothing is copied, vendored, forked, reformatted, or patched, and
`/opt/cutlass` is never written to.

The `cublasLtGetVersion()` runtime version is recorded, not pinned.

## 6. Operand and correctness equivalence

Per shape, in this order and entirely outside every timer:

1. A, B, and the output storage are created **once** by the pinned
   non-persistent example's own `create_tensors()` — the same factory, the same
   seed `1111`, the same A/B/C creation order, the same dtypes and strides that
   P3.2, P3.3, and P3.4 use. The persistent example's independent
   operand-generation path is deliberately never used.
2. A parser over the verified upstream file fails the run if that factory ever
   stops seeding with `1111` or stops building A, then B, then C in that order,
   because P3.5's cuBLASLt leading dimensions assume exactly the shapes and
   strides it produces.
3. A and B are never mutated.
4. The reference is computed **once** per shape, untimed, and reused by all four
   candidates — which is correct precisely because A and B are identical and
   immutable across them.
5. Before every candidate, the output buffer is reset **to NaN**, outside every
   timer, followed by a synchronize, so any element a candidate fails to write
   stays non-finite and is rejected instead of surviving as a stale value.
6. The complete output of every candidate is validated.
7. Shape-owned tensors, the cuBLASLt plan, its workspace, and every descriptor
   are released before the next shape begins.

### 6.1 How cuBLASLt receives byte-identical operands

The upstream factory keeps its device tensors for A and B private and returns
only the host tensors and the device C buffer, so a pointer into the CuTe DSL
candidates' own device A/B cannot be handed to cuBLASLt. The cuBLASLt candidate
is therefore given its own device copies, made from the very same immutable host
tensors with the same `empty_like`/`copy_` pair the closed P3.3 unit uses, and:

* each copy is proved byte-identical to that host tensor before anything runs;
* each copy's shape, strides, dtype, device, and non-null pointer are checked
  against exactly what the frozen descriptor contract assumes — nothing is
  transposed, re-laid out, retyped, or made contiguous;
* byte-identity is additionally enforced end to end, because **all four**
  candidates are validated against the one reference computed from those same
  host tensors, so an operand that differed at all could not pass.

### 6.2 Correctness policy (unchanged from P3.2/P3.3/P3.4)

* the CUDA matmul FP32 policy is set **only** through
  `torch.backends.cuda.matmul.fp32_precision`;
* it is set to `ieee` and must read back as exactly `ieee`;
* the unset `none` default proves nothing and is rejected;
* an absent API fails closed;
* `allow_tf32` is never read or written, and
  `torch.set_float32_matmul_precision()` is never used — in PyTorch 2.10 they
  are aliases of one setting and the last write silently wins;
* `atol = 1e-1`, `rtol = 1e-5`;
* the **entire** result is validated elementwise at full precision against
  `|result − reference| ≤ atol + rtol·|reference|`;
* non-finite results and non-finite references are rejected;
* `max_abs_error` and `max_rel_error` are finite, non-negative diagnostics;
  `max_rel_error` uses the same safe denominator floor the closed units use and
  never participates in the pass/fail decision;
* **no candidate whose correctness check failed ever runs warm-up or
  steady-state timing.**

## 7. Execution order and timer boundaries

Output order is **shape-major**: the five shapes in the frozen order, and inside
each shape the four candidates in the frozen order. For every candidate:

1. validate the environment and the repository/upstream provenance;
2. build the candidate's exact frozen configuration;
3. run the corresponding official `can_implement()` check (each upstream class
   in its own official form) or the cuBLASLt algorithm validation;
4. reset the output to NaN and synchronize — outside all timers;
5. measure **either** `compile_time_ms` around `cute.compile()` only, **or**
   `setup_time_ms` around cuBLASLt plan creation only;
6. measure `first_launch_ms` with a synchronized monotonic host interval;
7. validate the complete first-launch result;
8. only after correctness passes, execute identical warm-up counts;
9. measure `kernel_time_ms` with CUDA events on the candidate's own execution
   stream, divided by the measured iteration count;
10. buffer the validated measurement without emitting anything.

### 7.1 Compilation time and setup time are different concepts

`compile_time_ms` is CuTe DSL JIT compilation. `setup_time_ms` is cuBLASLt plan
creation, during which **nothing is compiled**. Both fields exist in the schema,
each carries the canonical `not_applicable` on the rows of the other method, and
**they are never compared against each other**.

Only `kernel_time_ms` participates in the P3.5 comparison.

## 8. Exact comparison definitions

For every candidate and shape:

```text
flop_count = 2 × M × N × K
tflops     = flop_count / (kernel_time_ms × 1e9)
```

The factor of two counts one multiplication plus one addition per
multiply-accumulate. `flop_count` is an exact integer property of the problem;
setup, compilation, the first launch, correctness, the output reset, and
epilogue bookkeeping are excluded by construction.

For each shape the cuBLASLt row is the baseline:

```text
throughput_ratio_vs_cublaslt = candidate_tflops / cublaslt_tflops
                             ≡ cublaslt_kernel_time_ms / candidate_kernel_time_ms

gap_to_cublaslt_pct          = 100 × (1 − throughput_ratio_vs_cublaslt)
```

The interpretation of the gap is fixed:

* **positive** — the candidate is slower than cuBLASLt;
* **zero** — equal;
* **negative** — the candidate is faster.

Negative values are **never clamped**, and beating cuBLASLt is **not** a success
requirement. For the cuBLASLt row itself,
`throughput_ratio_vs_cublaslt = 1` and `gap_to_cublaslt_pct = 0`, exactly.

Per shape:

* all four candidates are ranked by full-precision `kernel_time_ms`, ascending;
* an exact tie is broken by the frozen candidate order;
* `rank_within_shape` is 1 through 4;
* `best_cutedsl_variant` is selected from the three CuTe DSL candidates only,
  under the same rule, and is repeated identically on all four rows of the shape;
* exactly one CuTe DSL row carries `is_best_cutedsl=true`; the cuBLASLt row
  always carries `false`.

Every calculation and decision uses full-precision values. Deterministic decimal
formatting is applied only during serialization.

**Not computed anywhere:** confidence intervals, p-values, outlier removal,
roofline efficiency, empirical-ceiling utilization, memory bandwidth,
arithmetic-intensity classification, or any causal interpretation.

## 9. CSV output contract

```text
schema_version = p35.v1
experiment     = exp03_cutedsl_vs_cublaslt
unit           = P3.5
run_kind       = smoke
publishable    = false
```

A successful normal run writes to stdout **exactly 1 CSV header, exactly 20 CSV
rows, exactly 21 lines**, and nothing else. The closed `p32.v1`, `p33.v1`, and
`p34.v1` schemas are neither modified nor reinterpreted.

### 9.1 Exact ordered 100-field schema

```text
schema_version                  order_a                    seed
experiment                      order_b                    reference
unit                            order_c                    atol
run_kind                        order_d                    rtol
shape_index                     transa                     correctness
shape_id                        transb                     max_abs_error
candidate_index                 lda                        max_rel_error
method                          ldb                        compile_time_ms
variant                         ldc                        setup_time_ms
m                               ldd                        first_launch_ms
n                               compute_type               kernel_time_ms
k                               scale_type                 warmup_iterations
l                               pointer_mode               iterations
ab_dtype                        epilogue                   cache_mode
acc_dtype                       alpha                      flop_count
c_dtype                         beta                       tflops
a_major                         search_mode                throughput_ratio_vs_cublaslt
b_major                         workspace_limit_bytes      gap_to_cublaslt_pct
c_major                         workspace_bytes            rank_within_shape
scheduler                       alignment_a_bytes          best_cutedsl_variant
mma_tiler_m                     alignment_b_bytes          is_best_cutedsl
mma_tiler_n                     alignment_c_bytes          gpu_name
cluster_m                       alignment_d_bytes          gpu_uuid
cluster_n                       heuristic_requested        compute_capability
use_2cta_instrs                 heuristic_returned         driver_version
use_tma_store                   heuristic_index            cuda_toolkit_version
max_active_clusters             algo_id                    torch_cuda_version
                                tile_id                    cutedsl_version
                                stages_id                  cutlass_commit
                                split_k                    operand_factory_sha256
                                reduction_scheme           upstream_kernel_file
                                cta_swizzling              upstream_kernel_git_blob
                                custom_option              upstream_kernel_sha256
                                inner_shape_id             git_commit
                                cluster_shape_id           git_dirty
                                waves_count                publishable
                                cublaslt_version
```

The authoritative ordering is the single `CSV_FIELDS` tuple in
`src/gemm/gemm_comparison.py`; `scripts/check_gemm_comparison_p35.py`
independently restates the same 100 names in the same order and rejects any
missing, duplicate, additional, or reordered field.

### 9.2 Frozen categorical values

```text
experiment=exp03_cutedsl_vs_cublaslt   l=1                 reference=torch_cuda_fp32_ieee
unit=P3.5                              ab_dtype=BFloat16   correctness=PASS
run_kind=smoke                         acc_dtype=Float32   cache_mode=hot
seed=1111                              c_dtype=Float32     publishable=false
a_major=k   b_major=k   c_major=n
```

`method` is `cutedsl` on candidates 1–3 and `cublaslt` on candidate 4.
`scheduler` is `nonpersistent`, `static_persistent`, `static_persistent`, and
`not_applicable` respectively.

### 9.3 Method-specific `not_applicable` fields

The canonical value is the string `not_applicable`. It is never a number, never
zero, and never an empty field.

* **cuBLASLt rows** carry `not_applicable` for: `scheduler`, `mma_tiler_m`,
  `mma_tiler_n`, `cluster_m`, `cluster_n`, `use_2cta_instrs`, `use_tma_store`,
  `max_active_clusters`, `compile_time_ms`, `upstream_kernel_file`,
  `upstream_kernel_git_blob`, `upstream_kernel_sha256`.
* **CuTe DSL rows** carry `not_applicable` for the 38 cuBLASLt descriptor,
  heuristic, algorithm, workspace, alignment, scalar, runtime-version, and
  `setup_time_ms` fields.
* `max_active_clusters` is additionally `not_applicable` on
  `nonpersistent_1cta`, which has no cluster scheduler, and is a positive
  decimal integer from the official pinned hardware helper on both persistent
  rows.

`operand_factory_sha256` is applicable on **all** rows: it is the SHA-256 of the
pinned non-persistent example that built every operand of every shape, and it is
identical across the whole run.

### 9.4 Serialization rules

* Python's `csv` module, never string concatenation.
* Missing, duplicate, unknown, and reordered fields are rejected.
* Booleans are canonical lowercase `true` / `false`.
* Timing fields: fixed-point milliseconds with exactly 6 fractional digits.
* Tolerance, error, `alpha`, and `beta` fields: exactly 9 fractional digits.
* `waves_count` and `tflops`: exactly 6 fractional digits.
* `throughput_ratio_vs_cublaslt`: exactly 9 fractional digits, strictly positive.
* `gap_to_cublaslt_pct`: **signed** fixed-point with exactly 6 fractional
  digits; a value that rounds to zero from below is normalised to `0.000000`, so
  "equal" has exactly one spelling and a negative zero is rejected.
* `flop_count` is an exact decimal integer.
* `first_launch_ms` and `kernel_time_ms` must be finite and strictly positive on
  every row; `compile_time_ms` / `setup_time_ms` likewise on their applicable rows.
* No field is empty and none contains exponent notation or locale-dependent
  formatting.
* A row can only be built through a function that refuses any `correctness`
  value other than `PASS`, and it takes the shape, method, variant, scheduler,
  tiler, cluster, and 2-CTA flag from the frozen tables rather than from the
  caller.

### 9.5 stdout discipline and all-or-nothing output

All human-readable progress, compiler output, warnings, library output, and
diagnostics go to **stderr**. Descriptor 1 is redirected to descriptor 2 for the
whole measurement — so native writes from the JIT toolchain or from cuBLASLt
cannot corrupt the CSV — and the real stdout is restored only after all 20 rows
have passed validation.

The output is **all-or-nothing**. Any failure at any shape or candidate:

* exits non-zero;
* emits no CSV header;
* emits no CSV row, **including rows already completed**;
* prints a concise diagnostic to stderr;
* never describes the run as successful or comparable.

The checker proves this by injecting a synthetic failure at each of the four
candidate positions of an early shape (1), a middle shape (3), and the final
shape (5), and by writing directly to descriptor 1 during a successful
measurement.

## 10. Command line

```text
--warmup-iterations   integer 1..100, default 5
--iterations          integer 1..100, default 20
--self-test
--help
```

Normal execution always runs all five shapes and all four candidates. There is
no shape, method, variant, dtype, layout, scheduler, tile, cluster, seed,
tolerance, workspace, algorithm, source-path, publication, correctness-skip,
partial-run, or output-file control.

`--help` and `--self-test` are genuinely GPU-free: they import neither PyTorch,
nor CuTe DSL, nor the CUDA bindings, nor `ctypes`, nor either upstream example,
nor the native bridge. The checker proves this by running both behind an import
guard that turns any such import into a hard failure.

## 11. Make targets

### `gemm-comparison-p35-check` (GPU-free)

Depends on `gemm-cutedsl-p34-check`, preserving the existing chain (which in
turn runs the unmodified P3.3, P3.2, and P3.1 gates). It runs inside the pinned
image with no GPU, `--network none`, `--security-opt no-new-privileges`,
`--cap-drop ALL`, the invoking UID/GID, the repository mounted read-only, and
Python caches plus every build artifact under container-private `/tmp`. It:

* revalidates the pinned CUTLASS checkout (HEAD and clean working tree) and
  **both** official sources by Git blob and SHA-256;
* verifies the pinned CuTe DSL / PyTorch / cuda-python / cuda-bindings versions
  and runs `python3 -m pip check`;
* compiles `src/gemm/cublaslt_bridge_p35.cu` for `sm_103a` into `/tmp`;
* inspects the resulting ELF's dynamic symbols and dependencies;
* proves it references `cublasLtMatmul`, `cublasLtMatmulAlgoCheck`, and
  `cublasLtMatmulAlgoGetHeuristic`, and exports all ten `p35_*` entry points;
* proves it references **no** fallback GEMM API (`cublasGemmEx`,
  `cublasGemmStridedBatchedEx`, `cublasGemmBatchedEx`, `cublasSgemm`,
  `cublasHgemm`) and no alternative algorithm-enumeration path;
* syntax-checks all P3.5 Python files;
* runs the wrapper's GPU-free `--help` and `--self-test`;
* runs the checker's `--self-test` and the full repository contract check.

### `gemm-comparison-p35-smoke` (GPU) — the only P3.5 GPU target

Its **first recipe action** rejects a missing or non-numeric
`BLACKWELL_GPU_INDEX` before Docker, compilation, Make prerequisites, or any
other work — which is why it deliberately has no Make prerequisite that could
run first. It then uses only `scripts/run_container.sh`, which alone resolves
the physical index to a UUID, proves the device has no active compute processes,
exposes exactly that one UUID, and re-verifies inside the container that exactly
one matching GPU is visible as CUDA logical device 0. `--gpus all` is never
used and a GPU is never selected automatically.

Inside that same GPU container it revalidates **both** pinned upstream sources,
compiles the P3.5 bridge into private `/tmp`, runs the wrapper with exactly
`--warmup-iterations 2 --iterations 10`, preserves the wrapper's exit status,
preserves the CSV-only stdout contract, and prints a conspicuous stderr notice
stating that this is P3.5 functional comparison evidence; that all five shapes
and four candidates were required; that all rows are non-publishable; and that
no final campaign, statistical conclusion, Nsight analysis, or Phase 4
interpretation has been performed.

## 12. What was and was not run

### 12.1 GPU-free acceptance commands performed by the author

```bash
git diff --check
python3 -m py_compile src/gemm/gemm_comparison.py scripts/check_gemm_comparison_p35.py
python3 src/gemm/gemm_comparison.py --help
python3 src/gemm/gemm_comparison.py --self-test
python3 scripts/check_gemm_comparison_p35.py --self-test
python3 scripts/check_gemm_comparison_p35.py .
make check-static
make gemm-comparison-p35-check
```

**These are the author's own self-checks. They are not an independent audit, and
GPU-free checks are not GB300 verification.**

### 12.2 Not run

`make preflight` and `make gemm-comparison-p35-smoke` were **not** run: they
require later, operator-controlled GB300 execution on an explicitly selected
idle physical GPU:

```bash
BLACKWELL_GPU_INDEX=<idle-physical-index> make preflight
BLACKWELL_GPU_INDEX=<same-idle-physical-index> make gemm-comparison-p35-smoke
```

No independent audit of P3.5 has been performed, no GB300 run of P3.5 exists,
and no P3.5 measurement of any kind exists in this repository.

### 12.3 Two stale frontier guards this unit necessarily corrected

At the P3.5 baseline commit `b50dca3`, three closed-unit guards still described
a repository state that the P3.4 closure had already superseded, so
`make check-static` and `python3 scripts/check_cublaslt_gemm_p33.py .` both
failed **before** any P3.5 file existed:

1. `Makefile` asserted `P3.4 | Three execution variants | YES | NO | NO` and
   rejected `YES | YES | YES`, while `PLAN.md`, `README.md`, and
   `src/gemm/P3_4_PROTOCOL.md` all correctly recorded the closed
   `YES / YES / YES`.
2. `scripts/check_cublaslt_gemm_p33.py` rejected any PLAN.md in which a later
   unit was closed while P3.3 was "the frontier" — which P3.4's closure made
   permanently false.
3. `Makefile`, `scripts/check_cublaslt_gemm_p33.py`, and
   `scripts/check_cutedsl_variants_p34.py` all required the literal row
   `P3.5 | Five shapes and comparison | NO | NO | NO`, which structurally
   forbade P3.5 from ever being implemented.

All three were advanced to the truthful state — P3.4 closed, P3.5 implemented
but neither audited nor GB300-verified — and nothing was weakened: each guard
still rejects an overstated status, and the closed units' CLIs, schemas, field
orders, Make targets, one-shape restrictions, output behaviour, correctness and
provenance checks, and smoke semantics are byte-for-byte unchanged. P3.4 had to
make the same kind of correction to P3.3's checker when it landed.

## 13. Security model

* No GPU is ever selected automatically; every GPU run requires an explicit
  `BLACKWELL_GPU_INDEX` and goes exclusively through the audited launcher.
* The GPU-free gate runs with no GPU, no network, no added capabilities, no
  privilege escalation, the invoking UID/GID, and a read-only repository mount.
* The bridge is built only inside container-private `/tmp`, never in the
  repository.
* The bridge prints nothing, lets no C++ exception cross the C boundary,
  contains no CUDA kernel and no timing facility, has exactly one
  `cublasLtMatmul` call site, and validates every dimension and derived byte
  size against overflow before creating a descriptor.
* Only allowlisted device data is recorded: GPU index-free UUID, name, driver
  version, compute capability, and tool versions. No host name, user, path, or
  environment dump is ever read or emitted.
* No result file, campaign directory, manifest, or persistent artifact is
  written anywhere.

## 14. Non-goals

P3.5 adds none of: Nsight Compute; SASS inspection of proprietary library
kernels; HBM or UMMA attribution; roofline or arithmetic-intensity analysis;
plots, dashboards, or narrative interpretation; confidence intervals or
statistical significance; cold-cache experiments; arbitrary shapes or sweeps;
more than four candidates per shape; kernel autotuning; a new GEMM kernel;
copied or patched NVIDIA GEMM source; new dependencies or version pins; a
persistent campaign directory; final or publishable results; Phase 4
orchestration; automatic GPU selection; multi-GPU execution; or any commit,
push, merge, or pull request.

Phase 3 remains **open** until P3.5 is independently audited and verified on
GB300.
