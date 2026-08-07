# P3.3 — Equivalent cuBLASLt BF16 GEMM baseline (frozen protocol)

Status: `P3.3 = YES / NO / NO` (Implemented / Audited / Verified on GB300).
The author's own GPU-free checks are **not** an independent audit, and no
GB300 execution of this unit has happened yet. Section 13 records exactly what
was and was not run.

## 1. Objective

Provide the vendor-library counterpart of the P3.2 one-shape CuTe DSL wrapper:
exactly the same GEMM geometry, on exactly the same operand bytes, computed by
a direct, explicit `cublasLtMatmul` call.

P3.3 exists to establish six things:

1. direct, explicit use of `cublasLtMatmul` — not `cublasGemmEx`, not an
   ordinary cuBLAS GEMM, not `torch.matmul`, and not any other framework
   operation;
2. an auditable vendor-heuristic algorithm policy, fixed in advance and never
   autotuned;
3. complete correctness validation *before* any warm-up or steady-state timing;
4. separated setup, first-launch, and steady-state timings;
5. a strict machine-readable CSV contract suitable for later P3.5 normalization
   and comparison;
6. non-publishable functional evidence only.

**P3.3 creates no publishable performance result.** It is an implementation and
functional-verification unit, not an experimental campaign, and it makes no
performance claim of any kind.

## 2. Non-objectives

P3.3 deliberately does **not** compare itself against P3.2. No CuTe-versus-
cuBLASLt table, ratio, speedup, efficiency, utilization, bandwidth, TFLOP/s
figure, winner label, plot, or narrative conclusion exists anywhere in this
unit. That comparison is P3.5's job and P3.5 does not exist.

P3.3 also introduces no persistent 1-CTA or 2-CTA CuTe DSL variant, no
additional CuTe execution variant, none of the remaining four final shapes, no
shape / tile / cluster / stage / variant sweep, no algorithm autotuning, no
candidate benchmarking, no Nsight Compute, no SASS analysis of proprietary
cuBLASLt kernels, no campaign directory, manifest, aggregation, statistics, or
report, no Phase 4 orchestration, no cold-L2 mode, and no final-campaign
requirement such as three repetitions, three campaigns, a three-second
frequency-stabilization warm-up, or at least 1000 measured launches.

No FP16, TF32, FP8, FP4, NVFP4, or MXFP4 path, no Hopper target, no multi-GPU,
no NVLink, no MPS, no clock, power, or persistence-mode change, and no
privileged execution is introduced. No pinned version changes and no closed
Phase 1 / Phase 2 / P3.1 / P3.2 interface is touched.

The untimed PyTorch FP32 oracle in section 8 is a **correctness reference
only**. It is never timed and never reported as a competing method.

## 3. Files

| File | Role |
|------|------|
| `src/gemm/cublaslt_gemm.py` | Repository-owned orchestration: tensor generation, provenance, correctness, timing, CSV validation and serialization, bounded CLI, GPU-free self-test |
| `src/gemm/cublaslt_bridge.cu` | Small C-compatible shared-library bridge: cuBLASLt handle, descriptors, layouts, preference, heuristic selection, algorithm validation, workspace, metadata, launch, cleanup |
| `scripts/check_cublaslt_gemm_p33.py` | Fail-closed source, schema, CLI, Makefile, shared-library, and status checker with a GPU-free adversarial self-test |
| `src/gemm/P3_3_PROTOCOL.md` | This document |

No NVIDIA GEMM implementation is copied, forked, patched, or vendored. The
bridge calls only the public cuBLASLt API declared in the pinned CUDA 13.1
headers, defines no CUDA kernel of its own (`__global__` and `__device__` are
absent), and is proved to contain no custom kernel by both `make check-static`
and the checker.

## 4. Pinned contract (unchanged)

P3.3 changes no pin and adds no package. cuBLASLt is already supplied by the
pinned CUDA 13.1 development image, so `Dockerfile`, `VERSIONS.env`, and
`PHASE3_VERSIONS.env` are all untouched, and no floating cuBLAS/cuBLASLt
version is introduced anywhere.

| Property | Pinned value | Source |
|----------|--------------|--------|
| CUDA Toolkit | `13.1.0` | `VERSIONS.env` |
| Architecture | `sm_103a` | `VERSIONS.env` |
| CUTLASS / CuTe DSL | `v4.6.1` | `VERSIONS.env` |
| CUTLASS commit | `e05f953a5b3d38adc240df2ff928e0421c2abba3` | `VERSIONS.env` |
| PyTorch | `2.10.0+cu130` | `PHASE3_VERSIONS.env` |
| `torch.version.cuda` | `13.0` | `PHASE3_VERSIONS.env` |
| `cuda-python` / `cuda-bindings` | `13.0.3` | `PHASE3_VERSIONS.env` |
| Maximum build concurrency | 2 | `VERSIONS.env` |

The cuBLASLt **runtime** version is deliberately *not* pinned: it is obtained
at run time with `cublasLtGetVersion()` and recorded in the CSV as
`cublaslt_version`. This is the version that actually executed, which is the
only version worth recording — and it is not necessarily derivable from the
toolkit version.

## 5. Exact frozen configuration

None of the following is configurable through a CLI argument, an environment
variable, or a configuration file. Every value is a compile-time constant in
the bridge **and** an immutable constant in the wrapper. Neither declaration is
derived from the other, and the wrapper refuses to measure anything unless the
bridge reports back exactly the same contract (section 7).

| Property | Frozen value |
|----------|--------------|
| Operation | `C = A × Bᵀ` |
| Problem | `(M,N,K,L) = (4096,4096,4096,1)` |
| `A` | BF16, logical `M × K`, K-contiguous |
| `B` | BF16, logical `N × K`, K-contiguous |
| `C` / `D` | FP32, logical `M × N`, N-contiguous |
| Accumulation | FP32 |
| `alpha` | `1.0` |
| `beta` | `0.0` |
| Bias | none |
| Epilogue | default identity |
| Seed | `1111` |
| Operands | hot, reused |
| Devices | one visible GPU, CUDA logical device 0 |
| Streams | one CUDA stream |
| Batching | none beyond `L = 1` |

### 5.1 Explicit cuBLASLt descriptor contract

| Descriptor | Frozen value |
|------------|--------------|
| A layout | row-major, rows `M`, columns `K`, `lda = K` |
| B layout | row-major, rows `N`, columns `K`, `ldb = K` |
| C layout | row-major, rows `M`, columns `N`, `ldc = N` |
| D layout | row-major, rows `M`, columns `N`, `ldd = N` |
| `transa` | `CUBLAS_OP_N` |
| `transb` | `CUBLAS_OP_T` |
| A/B type | `CUDA_R_16BF` |
| C/D type | `CUDA_R_32F` |
| Compute type | `CUBLAS_COMPUTE_32F` |
| Scale type | `CUDA_R_32F` |
| Pointer mode | `CUBLASLT_POINTER_MODE_HOST` |
| Epilogue | `CUBLASLT_EPILOGUE_DEFAULT` |
| Batch count | `1` |

With `Adesc` describing an `M × K` matrix and `transa = CUBLAS_OP_N`,
`op(A) = A` is `M × K`; with `Bdesc` describing an `N × K` matrix and
`transb = CUBLAS_OP_T`, `op(B) = Bᵀ` is `K × N`. The product is therefore the
`M × N` matrix `A × Bᵀ`, which is exactly the mathematical operation P3.2
computes.

Nothing is silently transposed, no physical layout is changed, no output type
is changed, and there is no fallback to another GEMM interface. The leading
dimensions above are exactly the strides the operands already have, so the
descriptors describe the memory as it is rather than requiring a repack.

C and D are the same FP32 buffer. `beta` is exactly `0.0`, so C is never read;
the descriptors for C and D are identical, which is the documented in-place
form.

## 6. Exact operand equivalence with P3.2

This is the point of the unit, so it is established structurally, not asserted.

P3.2 calls the pinned upstream example's own `create_tensors()`. That function
seeds once with `1111` and then calls `cutlass.torch.matrix` three times — for
A, then B, then C — before copying each to the device. It cannot be reused
verbatim by P3.3, because it returns the device tensors only for C and discards
those for A and B (`a_tensor, _ = cutlass_torch.cute_tensor_like(...)`), while
P3.3 must retain every allocation in order to hand its device pointer to
cuBLASLt.

P3.3 therefore replicates that sequence exactly:

* the same pinned CUTLASS/CuTe installation;
* the same `cutlass.torch.matrix` factory;
* the same seed `1111`, applied once, before the first call;
* the same call order — A, then B, then C — which is what fixes the RNG stream;
* the same major flags, dtypes, shapes, and physical strides;
* the same `torch.empty_like(..., device="cuda")` / `copy_` device transfer, in
  the same order.

A different C++ RNG using the same numeric seed would **not** be equivalent and
is never used. No random number is generated anywhere in the bridge.

`verify_upstream_tensor_factory()` parses the verified upstream file with
Python's `ast` module — it is never imported, so roughly 1,800 lines of
unrelated module-level code never execute — locates `create_tensors`, and fails
closed unless the factory still seeds with `1111` and still performs exactly
the three `matrix()` calls, in the order and with the argument roles P3.3
replicates. A divergence between upstream and P3.3 is a hard error, not a
silent difference in operands.

The resulting physical layouts, verified explicitly on the device tensors
before any cuBLASLt object exists:

| Operand | dtype | shape | strides |
|---------|-------|-------|---------|
| A | `torch.bfloat16` | `(4096, 4096, 1)` | `(4096, 1, 16777216)` |
| B | `torch.bfloat16` | `(4096, 4096, 1)` | `(4096, 1, 16777216)` |
| C/D | `torch.float32` | `(4096, 4096, 1)` | `(4096, 1, 16777216)` |

`require_operand_layout()` rejects any mismatch of dtype, device, shape,
stride, or a null pointer. Nothing is corrected, made contiguous, or
transposed to fit: a mismatch fails the run.

Allocation, host-to-device conversion, tensor creation, and reference
construction are all outside every timer.

## 7. cuBLASLt algorithm policy

One fixed, non-autotuned policy, applied exactly once:

| Step | Frozen behaviour |
|------|------------------|
| Workspace limit | exactly `67,108,864` bytes (64 MiB), set as `CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES` |
| Heuristic request | exactly 32 results |
| Search mode | `CUBLASLT_SEARCH_BEST_FIT` |
| Selection | the first returned entry whose `state == CUBLAS_STATUS_SUCCESS` |
| Validation | `cublasLtMatmulAlgoCheck()` on that algorithm, against the same descriptors it will run with |
| Workspace rejection | the algorithm is rejected if its required workspace exceeds the fixed limit |
| Workspace allocation | exactly the required size; a null pointer only when the requirement is zero |
| Execution | only that one selected algorithm |

No candidate is benchmarked, executed for comparison, timed, or ranked by this
repository. There is no retry with another workspace limit, type, layout,
compute mode, or API. If no valid result exists, the bridge fails with a clear
diagnostic naming the returned count or the rejecting status, and **no CSV is
emitted**.

The bridge contains exactly one `cublasLtMatmul` call site, uses no timing
facility at all (`cudaEventRecord`, `cudaEventElapsedTime`, `std::chrono`, and
`clock_gettime` are all absent), and therefore cannot benchmark anything. Both
`make check-static` and the checker enforce this structurally.

### 7.1 Pointer alignment preferences

`CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_{A,B,C,D}_BYTES` are set from the actual
device-pointer alignments, never overstated: for each pointer the bridge
computes the largest power of two, capped at cuBLASLt's own 256-byte default,
that genuinely divides the address. Claiming more than a buffer satisfies would
let the heuristic return an algorithm the data cannot legally feed.

### 7.2 Recorded metadata

Every value below is read from the library at run time and serialized:

* returned cuBLASLt version (`cublasLtGetVersion()`);
* requested and returned heuristic counts;
* zero-based selected heuristic index;
* required workspace;
* waves count (from the `cublasLtMatmulAlgoCheck` result for the selected
  algorithm, which is the post-validation authoritative figure);
* actual pointer-alignment preferences for A, B, C, and D;
* every available selected-algorithm configuration attribute, read with
  `cublasLtMatmulAlgoConfigGetAttribute` at its documented width:
  `CUBLASLT_ALGO_CONFIG_ID` (algorithm ID, `int32_t`),
  `TILE_ID` (`uint32_t`), `STAGES_ID` (`uint32_t`),
  `SPLITK_NUM` (split-K count, `int32_t`),
  `REDUCTION_SCHEME` (`uint32_t`), `CTA_SWIZZLING` (`uint32_t`),
  `CUSTOM_OPTION` (`uint32_t`), `INNER_SHAPE_ID` (`uint16_t`), and
  `CLUSTER_SHAPE_ID` (`uint16_t`).

If any of the nine attributes cannot be read, or the library writes an
unexpected number of bytes, the run fails: missing or invalid algorithm
metadata is never serialized as a default.

## 8. Correctness policy

Correctness is mandatory, complete, and always precedes any warm-up or
steady-state timing. The oracle is exactly the P3.2 oracle, unchanged:

* the CUDA matmul FP32 policy is set **only** through
  `torch.backends.cuda.matmul.fp32_precision`;
* it is set to `ieee` and read back, and must read back as exactly `ieee`;
* the unset `none` default proves nothing and is rejected like any other value;
* an absent API fails closed;
* the legacy `allow_tf32` property is never read or written — in PyTorch 2.10
  the two are aliases of one setting, mixing them is unsupported, and the last
  write silently wins;
* `torch.set_float32_matmul_precision()` is never used;
* `atol = 1e-1`, `rtol = 1e-5`;
* the **entire** result is validated, elementwise, at full precision, against
  `|d − ref| ≤ atol + rtol·|ref|`;
* non-finite results and non-finite references are rejected;
* `max_abs_error` is the finite maximum absolute difference, and
  `max_rel_error` is the same safe diagnostic P3.2 reports,
  `max(|d − ref| / max(|ref|, 1.0))`, which stays finite where the reference is
  zero. It is a diagnostic only; the pass/fail decision never uses it.

The reference is `torch.einsum("mkl,nkl->mnl", A_f32, B_f32)`, computed on CUDA
over the same host operands under the IEEE guard, and it is never timed.

A correctness failure:

* returns non-zero;
* prints a diagnostic to stderr;
* emits **no** CSV header and **no** CSV row;
* executes **no** warm-up;
* executes **no** steady-state timing.

There is no CLI option to skip, weaken, or bypass it, and a CSV row can only be
constructed through `build_row()`, which refuses any `correctness` value other
than `PASS`.

## 9. Execution and timing order

The order is fixed:

1. validate the pinned environment and exactly one visible B300 GPU (compute
   capability derived from the pinned `sm_103a`, `nvidia-smi` reporting exactly
   one device, CUDA logical device 0);
2. verify repository and upstream provenance (commit, working-tree
   cleanliness, regular-file identity, Git blob SHA, SHA-256, and the upstream
   tensor-factory equivalence of section 6);
3. create all tensors — outside every timer;
4. create the cuBLASLt plan: handle, operation descriptor, four layouts,
   preference, heuristic query, `cublasLtMatmulAlgoCheck`, workspace
   allocation;
5. measure **`setup_time_ms`** with a monotonic host clock (`perf_counter_ns`)
   around step 4 only;
6. synchronize;
7. measure one first `cublasLtMatmul` invocation as **`first_launch_ms`**, with
   the same host-clock method P3.2 uses (synchronize, start clock, launch,
   synchronize, stop clock);
8. synchronize;
9. validate that complete output against the untimed oracle of section 8;
10. only if correctness passes, execute the warm-up launches;
11. measure steady state with CUDA events recorded on the exact stream
    `cublasLtMatmul` runs on;
12. compute **`kernel_time_ms` = `total_event_ms` / `iterations`**;
13. validate and emit one CSV row.

`setup_time_ms` is **not** `compile_time_ms`. Nothing is compiled at run time
in P3.3: the bridge is compiled ahead of time by the Make targets, and
cuBLASLt performs descriptor construction and a heuristic query, not JIT
compilation. The P3.2 field name is deliberately not reused, the wrapper
contains no occurrence of `compile_time_ms`, and both `make check-static` and
the checker reject any attempt to reintroduce it.

## 10. CSV contract

Frozen schema `schema_version=p33.v1`. The P3.2 `p32.v1` schema is neither
modified nor reinterpreted; the common comparison fields carry the same names
and semantics as in P3.2, and the P3.3-only fields are explicit cuBLASLt
metadata.

### 10.1 Exact ordered field list (77 fields)

```text
schema_version,experiment,unit,run_kind,method,variant,
m,n,k,l,ab_dtype,acc_dtype,c_dtype,a_major,b_major,c_major,
order_a,order_b,order_c,order_d,transa,transb,lda,ldb,ldc,ldd,
compute_type,scale_type,pointer_mode,epilogue,alpha,beta,seed,reference,
atol,rtol,correctness,max_abs_error,max_rel_error,
setup_time_ms,first_launch_ms,kernel_time_ms,warmup_iterations,iterations,
cache_mode,workspace_limit_bytes,workspace_bytes,
alignment_a_bytes,alignment_b_bytes,alignment_c_bytes,alignment_d_bytes,
heuristic_requested,heuristic_returned,heuristic_index,
algo_id,tile_id,stages_id,split_k,reduction_scheme,cta_swizzling,
custom_option,inner_shape_id,cluster_shape_id,waves_count,
gpu_name,gpu_uuid,compute_capability,driver_version,cuda_toolkit_version,
torch_cuda_version,cutedsl_version,cublaslt_version,cutlass_commit,
upstream_example_sha256,git_commit,git_dirty,publishable
```

### 10.2 Frozen categorical values

| Field | Frozen value |
|-------|--------------|
| `experiment` | `exp03_cutedsl_vs_cublaslt` |
| `unit` | `P3.3` |
| `run_kind` | `smoke` |
| `method` | `cublaslt` |
| `variant` | `heuristic_first_supported` |
| `reference` | `torch_cuda_fp32_ieee` |
| `cache_mode` | `hot` |
| `correctness` | `PASS` |
| `publishable` | `false` |

### 10.3 Output rules

* stdout on success: exactly one CSV header line and one CSV data row;
* stdout on any failure: empty;
* all diagnostics, build output, launcher output, warnings, and notices go to
  stderr. Descriptor 1 is redirected to descriptor 2 for the whole measurement
  and the real stdout is restored only to emit the two CSV lines, so even a
  native write to descriptor 1 cannot contaminate the data stream;
* serialization uses Python's `csv.DictWriter`, never string concatenation;
* booleans are canonical lowercase `true` / `false`;
* no field is empty, and no field contains `NaN`, infinity, exponent notation,
  or locale-dependent formatting;
* timing fields: fixed-point milliseconds with exactly six fractional digits;
* tolerance and error fields: fixed-point with exactly nine fractional digits;
* `alpha` and `beta`: fixed-point with exactly nine fractional digits;
* `waves_count`: fixed-point with exactly six fractional digits, finite and
  non-negative;
* integer and enum metadata: canonical decimal strings (no leading zeros);
* setup, first-launch, and kernel timings: finite and strictly positive;
* a row may only be constructed when correctness is exactly `PASS`.

No TFLOP/s, FLOP/s, speedup, efficiency, utilization, bandwidth, winner label,
or CuTe-versus-cuBLASLt comparison is computed or emitted anywhere.

## 11. Command line

The only runtime controls are:

| Option | Bounds |
|--------|--------|
| `--warmup-iterations N` | `1..100`, default `5` |
| `--iterations N` | `1..100`, default `20` |
| `--self-test` | GPU-free contract self-test |
| `--help` | usage |

The Make smoke target uses exactly 2 warm-ups and 10 measured launches.

There is no CLI option for shape, types, layouts, transposes, leading
dimensions, alpha, beta, epilogue, workspace, heuristic count, algorithm,
correctness skipping, cache mode, or publication, and no environment variable
that reopens any of them. The compiled bridge's location is a constant, not a
control.

`--help` and `--self-test` are completely GPU-free: they import neither
PyTorch, nor CuTe DSL, nor the CUDA bindings, and they never load the shared
bridge — `ctypes` itself is not imported on those paths. This is proved, not
assumed: the checker runs both behind an import guard that turns any such
attempt into a hard failure.

## 12. Make targets

### `gemm-cublaslt-p33-check` (GPU-free)

Depends on the existing `gemm-cutedsl-p32-check` (which in turn runs the
unmodified `gemm-cutedsl-p31-check`). It uses the pinned image, exposes no
GPU, and runs with `--network none`, `--cap-drop ALL`, `no-new-privileges`, the
invoking UID/GID, and the repository mounted read-only; bytecode and the bridge
build output both go to the container's own `/tmp`. Inside it:

* confirms nvcc reports the pinned CUDA major.minor;
* re-verifies the pinned CUTLASS commit and the upstream example SHA-256;
* compiles the bridge with CUDA 13.1, C++17, `-O3`, `-lineinfo`,
  `-Xcompiler -fPIC -shared`, and the pinned `sm_103a` target, linking directly
  against `libcublasLt` and `libcudart`;
* inspects the resulting shared object with `nm -D` and `readelf -d`: the
  measured path must reference `cublasLtMatmul`,
  `cublasLtMatmulAlgoGetHeuristic`, and `cublasLtMatmulAlgoCheck`; the six
  exported `p33_*` entry points must be present; `cublasGemmEx`,
  `cublasGemmStridedBatchedEx`, `cublasGemmBatchedEx`, `cublasSgemm`,
  `cublasHgemm`, `cublasLtMatmulAlgoGetIds`, and `cublasLtMatmulAlgoInit` must
  all be absent; and `libcublasLt.so` and `libcudart.so` must both be linked;
* runs Python syntax checks;
* runs the wrapper's `--help` and `--self-test`;
* runs the checker's `--self-test` and the complete repository contract check.

### `gemm-cublaslt-p33-smoke` (GPU)

It has no Make prerequisite by design: its **first** recipe step validates that
`BLACKWELL_GPU_INDEX` was explicitly provided, before Docker, compilation, or
any other work begins. It then:

* runs exclusively through `scripts/run_container.sh` with
  `RUN_CONTAINER_STDOUT_IS_DATA=1`, so the launcher's own informational lines
  and the image entrypoint banner stay off stdout;
* lets that existing, audited launcher resolve the physical index to a UUID,
  prove the device has no active compute processes, expose exactly that UUID,
  and re-verify inside the container that exactly one matching GPU is visible
  as CUDA logical device 0;
* compiles the bridge inside that already-selected container into private
  `/tmp`;
* revalidates the pinned CUTLASS commit and the upstream SHA-256 there;
* executes the wrapper with exactly 2 warm-ups and 10 measured launches;
* preserves the wrapper's exit status;
* keeps stdout empty on failure and exactly two CSV lines on success;
* prints an explicit stderr notice that this is non-publishable P3.3
  functional verification and **not** a performance comparison.

`scripts/run_container.sh` is not modified: its audited data-stream mode
already exists and P3.3 only opts into it.

## 13. What was and was not run

### 13.1 GPU-free checks performed by the author

All of the following were executed on the development host and passed:

```bash
git diff --check
python3 -m py_compile src/gemm/cublaslt_gemm.py scripts/check_cublaslt_gemm_p33.py
python3 src/gemm/cublaslt_gemm.py --help
python3 src/gemm/cublaslt_gemm.py --self-test
python3 scripts/check_cublaslt_gemm_p33.py --self-test
python3 scripts/check_cublaslt_gemm_p33.py .
make check-static
make gemm-cublaslt-p33-check
```

The bridge was additionally compiled inside the pinned image with the pinned
toolchain, and its ELF symbols were inspected, as part of
`make gemm-cublaslt-p33-check`.

**These are the author's own self-checks. They are not an independent audit.**

### 13.2 GB300 commands not yet performed

Neither of the following has been run, and no P3.3 GPU result of any kind
exists:

```bash
BLACKWELL_GPU_INDEX=<operator-supplied-index> make preflight
BLACKWELL_GPU_INDEX=<operator-supplied-index> make gemm-cublaslt-p33-smoke
```

Consequently the following remain unproven on hardware and must be treated as
open until an explicitly authorized GB300 run closes them:

* that `cublasLtMatmulAlgoGetHeuristic` returns at least one supported
  algorithm for this exact row-major BF16 descriptor contract on `sm_103a`
  (P3.3 fails closed with a diagnostic and emits no CSV if it does not, and
  never retries with a different layout, type, compute mode, or API);
* the actual selected-algorithm metadata, workspace requirement, waves count,
  and pointer alignments;
* that correctness passes against the untimed IEEE-FP32 oracle;
* the three timings.

### 13.3 Separation from P3.4, P3.5, and Phase 4

P3.4 (three execution variants), P3.5 (five shapes and comparison), and Phase 4
(campaigns and integration) remain unimplemented and are unaffected by this
unit. In particular, the CuTe-versus-cuBLASLt comparison that experiment 3
ultimately reports belongs to P3.5 and does not exist: P3.2 and P3.3 each emit
independent, non-publishable, single-row functional evidence, and nothing in
this repository joins, normalizes, ranks, or compares them.

## 14. Failure behaviour summary

| Condition | Behaviour |
|-----------|-----------|
| Missing or modified upstream checkout / example | non-zero exit, stderr diagnostic, no CSV |
| Upstream tensor factory diverged from P3.3's replication | non-zero exit, stderr diagnostic, no CSV |
| Wrong device, wrong compute capability, more than one GPU | non-zero exit, stderr diagnostic, no CSV |
| Pinned version mismatch | non-zero exit, stderr diagnostic, no CSV |
| Missing or unloadable compiled bridge | non-zero exit, stderr diagnostic, no CSV |
| Bridge/wrapper contract disagreement (any frozen value) | non-zero exit, stderr diagnostic, no CSV |
| Operand shape, stride, dtype, or device mismatch | non-zero exit, stderr diagnostic, no CSV |
| No supported heuristic result | non-zero exit, stderr diagnostic, no CSV |
| Selected algorithm needs more than 64 MiB workspace | non-zero exit, stderr diagnostic, no CSV |
| Unreadable algorithm metadata | non-zero exit, stderr diagnostic, no CSV |
| `cublasLtMatmul` returns a non-success status | non-zero exit, stderr diagnostic, no CSV |
| Correctness failure | non-zero exit, stderr diagnostic, no CSV, no warm-up, no steady-state timing |
| Non-finite or non-positive timing | non-zero exit, stderr diagnostic, no CSV |
| Any row-contract violation | non-zero exit, stderr diagnostic, no CSV |
