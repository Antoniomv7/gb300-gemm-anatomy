# P3.2 — One-shape CuTe DSL GEMM wrapper (frozen protocol)

Status: `P3.2 = YES / YES / YES` (Implemented / Audited / Verified on GB300).
The author's own GPU-free checks alone are **not** an independent audit or a
GB300 verification; the completed independent audit and GB300 execution that
support this closed status are recorded in section 14.

## 1. Objective

Provide one small, repository-owned CuTe DSL wrapper that executes exactly one
frozen BF16 GEMM configuration and cleanly separates three costs that the
pinned upstream example fuses together:

```text
compile_time_ms    JIT compilation / load only
first_launch_ms    the first launch of the compiled kernel
kernel_time_ms     steady-state per-launch time after warm-up
```

The wrapper compiles the kernel, executes its first launch, validates the
complete result against an untimed FP32 oracle, and only then performs warm-up
and steady-state timing. It emits exactly one machine-readable CSV data row and
classifies every timing as non-publishable P3.2 infrastructure evidence.

**P3.2 creates no publishable performance result.** It is an implementation and
functional-verification unit, not an experimental campaign, and it makes no
performance claim of any kind.

## 2. Non-objectives

P3.2 deliberately does **not** introduce a cuBLAS or cuBLASLt baseline,
persistent scheduling, 2-CTA MMA instructions, any other GEMM shape, a shape /
tile / cluster / stage / variant sweep, autotuning, candidate selection,
TFLOP/s or speedup arithmetic, Nsight Compute, SASS or profiler analysis,
campaign directories, manifests, aggregation, final datasets, figures, tables,
comparative analysis, a cold-L2 mode, or any final-campaign requirement such as
three repetitions, three campaigns, a three-second frequency-stabilization
warm-up, or at least 1000 measured launches. Those belong to P3.3–P3.5 and
Phase 4 (section 11).

No FP16, FP8, FP4, NVFP4, or MXFP4 path, no Hopper target, no multi-GPU, no
NVLink, no MPS, no clock or power-limit change, and no privileged execution is
introduced. No pinned version and no closed Phase 1 / Phase 2 interface changes.

The untimed PyTorch FP32 oracle in section 6 is a **correctness reference
only**. It is never timed, never reported as a competing method, and it is
explicitly **not** the P3.3 cuBLASLt baseline, which is a separate unit
(`src/gemm/P3_3_PROTOCOL.md`). No P3.2-versus-P3.3 comparison exists anywhere;
that comparison belongs to P3.5, which is unimplemented.

## 3. Exact frozen configuration

| Property | Frozen value |
|----------|--------------|
| Operation | `C = A × B` |
| Problem | `(M,N,K,L) = (4096,4096,4096,1)` |
| Input dtype | BF16 × BF16 |
| Accumulation | FP32 in TMEM |
| Output dtype | FP32 |
| A major | `k` |
| B major | `k` |
| C major | `n` |
| MMA tiler | `(M,N) = (128,128)` |
| Cluster shape | `(M,N) = (1,1)` |
| MMA group | one CTA |
| Scheduler | non-persistent |
| A/B movement | TMA |
| Output path | TMA store |
| Seed | `1111` |
| Cache model | hot / reused operands |
| Target | `sm_103a` |

These are immutable constants of the unit. The command line exposes **no**
shape, dtype, layout, tiler, cluster, TMA, persistence, or 2-CTA control, and
no way to skip the reference check. The only runtime controls that exist are:

```text
--warmup-iterations   positive integer, 1..100, default 5
--iterations          positive integer, 1..100, default 20
--self-test           GPU-free contract self-test
--help
```

The defaults are deliberately small and explicitly non-publishable. The GPU
smoke target uses two warm-ups and ten measured launches. The upper bound of
100 is itself a guard: a final campaign would need at least 1000 measured
launches, so P3.2 cannot accidentally masquerade as one.

Mathematically, with the pinned example's own conventions: `A` is `(M,K,L)`
with K contiguous, `B` is `(N,K,L)` with K contiguous, `C` is `(M,N,L)` with N
contiguous, and `C[m,n,l] = Σ_k A[m,k,l] · B[n,k,l]` accumulated in FP32.

## 4. Why `(4096,4096,4096,1)`

It is the **first of the five final `(M,N,K)` shapes** of experiment 3, which
`README.md` and `AGENTS.md` already fix as `(4096,4096,4096)`,
`(8192,8192,8192)`, `(16384,512,4096)`, `(32768,512,4096)`, and
`(512,16384,4096)`. Choosing the first final shape rather than inventing a
sixth one means P3.2 exercises the wrapper on geometry the later units will
actually use, without introducing any configuration that has to be retired.

It is also the smallest of the five and the only square one, so it is the
cheapest shape that is still large enough for the three timers to be clearly
separable, and its operands (2 × 32 MiB BF16 inputs plus a 64 MiB FP32 output)
comfortably fit in device memory alongside the FP32 reference.

**P3.2 must not expose or execute the other four shapes.** They belong to P3.5.

## 5. Reuse and provenance of the pinned NVIDIA implementation

The wrapper owns no GEMM kernel. It loads the exact file P3.1 pinned:

```text
Repository:    NVIDIA/cutlass
Tag:           v4.6.1
Relative path: examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py
License:       BSD-3-Clause (NVIDIA CORPORATION & AFFILIATES)
```

read-only and in place from the pinned `/opt/cutlass` checkout inside the
image. The commit, Git blob SHA, and SHA-256 are **not** restated here as
literals: they live in `VERSIONS.env` (`CUTLASS_COMMIT`) and
`PHASE3_VERSIONS.env` (`CUTEDSL_P31_EXAMPLE_PATH`,
`CUTEDSL_P31_EXAMPLE_GIT_BLOB`, `CUTEDSL_P31_EXAMPLE_SHA256`), and the wrapper
reads them from those two files at run time. `scripts/check_cutedsl_gemm_p32.py`
fails if any of those pinned values ever appears as a literal in the wrapper,
so provenance cannot silently drift out of the contracts. P3.2 adds **no key**
to either contract file: it executes the same upstream file as P3.1 and
therefore reuses P3.1's existing pins.

The approximately 1800-line upstream file is never copied, vendored, forked,
reformatted, or patched into this repository, and `/opt/cutlass` is never
written to — Git is queried read-only, with a per-invocation
`-c safe.directory=/opt/cutlass` because the checkout is root-owned inside the
image while the container runs as the invoking user.

What the wrapper reuses from that module:

* `DenseGemmKernel` — the kernel class, constructed with the frozen
  accumulator dtype, one-CTA MMA group, `(128,128)` tiler, `(1,1)` cluster, and
  TMA store;
* `DenseGemmKernel.can_implement()` — checked before compilation;
* `create_tensors()` — the deterministic tensor factory, which applies the
  frozen seed `1111` itself.

The module is imported under a private module name, so the upstream
`if __name__ == "__main__"` block never runs.

### Why the upstream `run()` is not called

`run()` performs tensor creation, `cute.compile`, the reference-checked launch,
and its own benchmarking helper in one call and returns a single number. It
cannot provide the required separation of compilation, first launch, and
steady-state kernel time, and its benchmarking helper regenerates workspaces
between iterations. The wrapper therefore drives the same upstream objects
directly and never calls `run()`. Nothing in the upstream file is modified to
make this possible.

If a future upstream pin made these interfaces unusable without copying or
patching the kernel, that is a blocker to report — not a licence to vendor the
example.

### Lazy imports

Every heavy import (PyTorch, CuTe DSL, the CUDA bindings) is deferred into the
measurement path, so `--help` and `--self-test` are GPU-free. The checker
proves this rather than assuming it: it runs both in a child interpreter whose
own import hook turns any attempt to load `torch`, `cutlass`, `cuda`, `numpy`,
or `pynvml` into a hard failure.

## 6. Complete execution sequence

The wrapper executes in exactly this order.

### 6.1 Environment and provenance (before any tensor exists)

1. Exactly one CUDA-visible GPU is required; any other count aborts.
2. It must be logical CUDA device `0`.
3. Only allowlisted provenance is collected: GPU name, GPU UUID, compute
   capability, driver version, CUDA toolkit version, PyTorch CUDA version,
   CuTe DSL version, CUTLASS commit, upstream example SHA-256, repository Git
   commit, and repository dirty state. Nothing else — no host name, no user,
   no path, no environment dump.
4. The pinned upstream checkout and file identity are revalidated: checkout
   present, HEAD equal to the pinned commit, no tracked or untracked
   modification, the example a non-symlink regular file, matching Git blob SHA,
   matching SHA-256. This runs once **before** the module is imported and again
   as part of provenance collection.
5. Everything fails closed. A missing, ambiguous, malformed, or mismatched
   value aborts before any tensor is allocated. In particular: the device
   compute capability must equal the capability derived from the pinned
   `CUDA_ARCH`, the installed toolkit's `release X.Y` must match the pinned
   `CUDA_VERSION`, and `torch.__version__`, `torch.version.cuda`, and
   `cutlass.__version__` must match their pins.

Physical GPU selection and the idle-device proof remain **exclusively** owned
by `scripts/run_container.sh`. The wrapper never selects a GPU, never enumerates
the host's devices, and never runs Docker.

### 6.2 Tensor preparation

Seed `1111` is applied, then the pinned upstream `create_tensors()` builds the
BF16 A/B and FP32 C tensors in the frozen layouts. All allocation happens
**outside every timer**. `can_implement()` is checked before compilation, and a
negative answer aborts: P3.2 never silently falls back to another
configuration.

### 6.3 Compilation timing

`compile_time_ms` is measured with the monotonic host clock
`time.perf_counter_ns()`:

1. synchronize the selected CUDA device;
2. start the host timer;
3. call `cute.compile(...)`;
4. synchronize again;
5. stop the timer.

It therefore contains only the JIT compilation/load step — not tensor
allocation, not the reference computation, not the first launch, not warm-up,
and not kernel benchmarking.

### 6.4 First launch and correctness

The first invocation of the compiled kernel is also the launch whose output is
validated:

1. synchronize;
2. start a monotonic host timer;
3. invoke the compiled GEMM once;
4. synchronize;
5. stop the timer and record `first_launch_ms`;
6. compute the complete FP32 reference;
7. compare the **entire** result, before any warm-up or steady-state timing.

The reference is the pinned PyTorch installation used purely as an untimed
correctness oracle, evaluated on the GPU in IEEE FP32. This is why the CSV
records `reference=torch_cuda_fp32_ieee`. It is never timed and never reported
as a competing method.

#### The IEEE FP32 guarantee (PyTorch 2.10 API only)

The FP32 policy for CUDA matrix multiplication is established through the
PyTorch 2.10 API and **nothing else**:

```text
torch.backends.cuda.matmul.fp32_precision = "ieee"
```

The property is then **read back, and must be exactly the string `ieee`**.

* The legacy `torch.backends.cuda.matmul.allow_tf32` property is never read and
  never written, and neither is the legacy cuDNN TF32 property. In PyTorch 2.10
  the legacy flag and `fp32_precision` are two views of one setting, mixing the
  two APIs is unsupported, and the last write silently wins — writing
  `allow_tf32 = True` after `fp32_precision = "ieee"` rewrites the policy to
  `tf32` with no error at all. Mixing is therefore forbidden outright.
* `torch.set_float32_matmul_precision()` and any other overlapping
  precision-control API are likewise never combined with it.
* The API must exist. A PyTorch without `torch.backends.cuda.matmul` or without
  `fp32_precision` fails closed **before** the reference is computed; there is
  no fallback.
* `none` — the unset default — is **rejected**. It records that no explicit
  policy was chosen and proves nothing about IEEE behaviour. So are an empty
  value, an absent attribute, and every other alternative.
* A rejected assignment, or any readback that is not exactly `ieee`, produces a
  clear stderr diagnostic and **no CSV**.
* The previous value of the same new API is restored afterwards. No legacy
  setting is ever read or restored.

The requirement is checked twice: once up front, so a PyTorch that cannot
guarantee a trustworthy verdict fails before a JIT compilation is spent on it,
and once as the guard that wraps the reference computation itself.

Tolerances:

```text
atol = 1e-1
rtol = 1e-5
```

The decision rule is the elementwise criterion
`|c - ref| <= atol + rtol * |ref|`, evaluated at full precision over the whole
tensor, together with a shape check and a finiteness check on both tensors.

Two diagnostics are recorded:

```text
max_abs_error = max |c - ref|
max_rel_error = max ( |c - ref| / max(|ref|, 1.0) )
```

The denominator floor of `1.0` keeps the quotient finite and well defined where
the reference is exactly zero. Both diagnostics are finite by construction and
neither participates in the pass/fail decision.

If correctness fails, the wrapper exits non-zero, prints the diagnostic to
stderr, emits **no CSV header and no CSV data row**, performs **no warm-up**,
and performs **no steady-state timing**. Compilation and the first launch are
observed before correctness is known, but their values may only be emitted
after correctness has passed.

### 6.5 Warm-up and steady-state kernel timing

Only after full correctness passes:

1. the requested warm-up launches execute;
2. the device is synchronized;
3. the requested launches are measured with CUDA events recorded on the same
   stream the kernel runs on;
4. the device is synchronized once after the ending event;
5. `kernel_time_ms = total CUDA-event elapsed milliseconds / iterations`.

`kernel_time_ms` therefore excludes JIT compilation, first-launch
initialization, tensor allocation, the reference computation, and CSV
serialization. All three timings are validated as finite and strictly positive
before a row can be built.

No TFLOP/s, speedup, efficiency, utilization, bandwidth, or comparison of any
kind is computed or printed.

## 7. Timer boundaries at a glance

| Timer | Clock | Starts after | Stops after | Excludes |
|-------|-------|--------------|-------------|----------|
| `compile_time_ms` | host `perf_counter_ns` | device sync, immediately before `cute.compile` | device sync after `cute.compile` | allocation, first launch, reference, warm-up, steady state |
| `first_launch_ms` | host `perf_counter_ns` | device sync, immediately before the first launch | device sync after that launch | compilation, reference, warm-up, steady state |
| `kernel_time_ms` | CUDA events on the kernel's stream | warm-up complete and synchronized | sync after the ending event, divided by `iterations` | compilation, first launch, allocation, reference, serialization |

## 8. CSV contract

Normal GPU execution writes to **stdout**: exactly one CSV header line, exactly
one CSV data row, and nothing else. Every human-readable message — progress,
warnings, compiler output, diagnostics — goes to **stderr**.

To make that true even when the JIT toolchain writes to file descriptor 1 from
native code, descriptor 1 is redirected to descriptor 2 for the whole
measurement, and the real stdout is restored only to emit the two CSV lines,
after correctness has already passed. Rebinding `sys.stdout` alone would not
cover native writes.

The frozen ordered schema (47 fields):

```text
schema_version
experiment
unit
run_kind
method
variant
m
n
k
l
ab_dtype
acc_dtype
c_dtype
a_major
b_major
c_major
mma_tiler_m
mma_tiler_n
cluster_m
cluster_n
use_2cta_instrs
use_tma_store
seed
reference
atol
rtol
correctness
max_abs_error
max_rel_error
compile_time_ms
first_launch_ms
kernel_time_ms
warmup_iterations
iterations
cache_mode
gpu_name
gpu_uuid
compute_capability
driver_version
cuda_toolkit_version
torch_cuda_version
cutedsl_version
cutlass_commit
upstream_example_sha256
git_commit
git_dirty
publishable
```

Frozen categorical values:

```text
schema_version=p32.v1
experiment=exp03_cutedsl_vs_cublaslt
unit=P3.2
run_kind=smoke
method=cutedsl
variant=nonpersistent_1cta
reference=torch_cuda_fp32_ieee
correctness=PASS
cache_mode=hot
publishable=false
```

Formatting rules:

* the row is produced with Python's `csv` module (`csv.DictWriter`), never by
  string concatenation, with `\n` line endings and minimal quoting;
* booleans use exactly one canonical lowercase spelling, `true` or `false`,
  in `use_2cta_instrs`, `use_tma_store`, `git_dirty`, and `publishable`;
* the three timings are plain fixed-point decimals with exactly **6**
  fractional digits (milliseconds, i.e. nanosecond resolution);
* `atol`, `rtol`, `max_abs_error`, and `max_rel_error` are plain fixed-point
  decimals with exactly **9** fractional digits;
* no exponent notation, no locale dependence, no shortest-round-trip
  ambiguity; a magnitude below half of the last retained digit therefore
  serializes as zero, which is intentional because these fields are
  diagnostics and every decision is taken on the full-precision value first;
* `NaN` and infinity are forbidden and cannot be serialized;
* the three timings must be strictly positive.

A row can only be built through one function, and that function refuses any
`correctness` value other than `PASS`. A successful row with failed or skipped
correctness is therefore not merely unlikely, it is unconstructible.

P3.2 does **not** automatically write a result file and does **not** create a
campaign directory. Redirecting stdout is the operator's choice, and no
generated CSV is committed.

## 9. Files P3.2 owns

```text
src/gemm/cutedsl_gemm.py             the wrapper
scripts/check_cutedsl_gemm_p32.py    the GPU-free checker
src/gemm/P3_2_PROTOCOL.md            this document
```

Its remaining footprint is the `gemm-cutedsl-p32-check` /
`gemm-cutedsl-p32-smoke` targets plus their validation in `Makefile`, and the
truthful status text in `PLAN.md` and `README.md`. `VERSIONS.env`,
`PHASE3_VERSIONS.env`, and `Dockerfile` are unchanged, and P3.1's files and
status are untouched. `scripts/run_container.sh` gains only the opt-in
`RUN_CONTAINER_STDOUT_IS_DATA=1` path documented below; its default interface
and behaviour for every closed Phase 1/Phase 2 caller remain unchanged.

## 10. Verification commands

### GPU-free (no GPU, no network, no elevated privileges)

```bash
python3 -m py_compile src/gemm/cutedsl_gemm.py scripts/check_cutedsl_gemm_p32.py
python3 src/gemm/cutedsl_gemm.py --self-test
python3 scripts/check_cutedsl_gemm_p32.py --self-test
make check-static
make gemm-cutedsl-p32-check
```

`make gemm-cutedsl-p32-check` runs inside the pinned image with `--network
none`, `--security-opt no-new-privileges`, `--cap-drop ALL`, the invoking
UID/GID, **no GPU exposed**, and the repository mounted read-only. It fails
closed unless all of the following hold:

1. `/opt/cutlass` exists and its HEAD is exactly the pinned `CUTLASS_COMMIT`.
2. The checkout has no tracked or untracked modification.
3. The example is a non-symlink regular file.
4. Its Git blob SHA matches `CUTEDSL_P31_EXAMPLE_GIT_BLOB`.
5. Its SHA-256 matches `CUTEDSL_P31_EXAMPLE_SHA256`.
6. CuTe DSL, PyTorch, and `torch.version.cuda` report their pinned versions.
7. `importlib.metadata` reports the pinned `cuda-python` and `cuda-bindings`.
8. `python3 -m pip check` reports no broken requirements (never suppressed).
9. Both P3.2 Python files compile.
10. The wrapper's `--help` and `--self-test` run GPU-free and pass.
11. The checker and the checker's own `--self-test` pass.

The existing P3.1 checks are untouched and still run separately.

### GB300 verification (performed 6 August 2026)

The following command sequence was executed with physical index `7` substituted
for `<physical-index>` and an explicitly confirmed idle device — the project did
not and does not select a GPU automatically:

```bash
BLACKWELL_GPU_INDEX=<physical-index> make preflight
BLACKWELL_GPU_INDEX=<same-physical-index> make gemm-cutedsl-p32-smoke
```

`make gemm-cutedsl-p32-smoke` validates `BLACKWELL_GPU_INDEX` in its first
recipe step, before any Docker work — which is why it deliberately has no Make
prerequisite. It then runs exclusively through `scripts/run_container.sh`,
never invoking Docker itself, re-checks the upstream commit and SHA-256 inside
that same GPU container immediately before running the wrapper, runs exactly
the frozen one-shape configuration with two warm-ups and ten measured launches,
preserves the wrapper's exit code, and prints an explicit stderr notice that
the emitted timings are P3.2 non-publishable functional evidence.

Fresh preflight campaign `20260806T163806Z` reported `OVERALL=PASS` on physical
GPU index `7`, UUID `GPU-40e00845-d89c-1393-2c32-a2dca3ee9442`, an NVIDIA
B300 SXM6 AC with compute capability 10.3 and driver 610.43.02. The P3.2 smoke
then executed clean repository commit
`c8b3e2ee57e0297940e0fd5864583ec12dfb23e3`, revalidated CUTLASS commit
`e05f953a5b3d38adc240df2ff928e0421c2abba3` and upstream SHA-256
`f99bc4cc1e0aea8990e2929d7c703dfc8196d797b7c9f5a889eabcd3c4ff67ec`,
reported `can_implement: OK`, and passed the complete-result check with
`max_abs_error=0.0` and `max_rel_error=0.0`. It completed the frozen two
warm-ups and ten measured launches and emitted exactly one `p32.v1` data row
with `git_dirty=false` and `publishable=false`. The three timing fields were
finite and strictly positive; they are non-publishable functional diagnostics,
not a performance result.

### The smoke target's stdout is exactly the CSV

On a successful run, the **whole stdout of `make gemm-cutedsl-p32-smoke`** is
exactly one CSV header line and one CSV data row — nothing else, from any
source. On a failed run, its stdout is **empty**. Every launcher line, Make
diagnostic, container message, wrapper progress line, and compiler warning goes
to stderr.

Three contamination sources are closed structurally, never by filtering:

1. **Make echoing the recipe.** Every recipe line of the target is quiet (`@`).
2. **The launcher's own informational lines.** `scripts/run_container.sh` gains
   one opt-in mode, `RUN_CONTAINER_STDOUT_IS_DATA=1`, which sends its two
   allowlisted device-selection lines to stderr. The smoke target is its only
   user. Unset — as in every closed Phase 1/Phase 2 caller — the launcher's
   behaviour is byte-for-byte unchanged, which those callers depend on.
3. **The image entrypoint banner.** The NGC entrypoint unconditionally writes a
   copyright/licence banner to stdout before the command starts. In the same
   opt-in mode the launcher bypasses it with `--entrypoint /bin/bash`. The
   entrypoint parts are purely informational; the only side effect any of them
   has is exporting `NVIDIA_CPU_ONLY=1` when no NVIDIA driver is present, a
   state this launcher already refuses. The executed command, the in-container
   guard, GPU selection, the idle-device proof, every security flag, and the
   preserved exit status are identical in both modes.

Nothing greps, deletes, or rewrites lines: unexpected stdout is meant to
surface as a contract violation, not to be silently discarded, and the checker
tests exactly that.

### Exit status and success reporting

The launcher's exit status is captured and re-raised by the recipe, so a
failure at provenance validation, launcher setup, container startup, import,
compilation, first launch, correctness, warm-up, or timing returns non-zero.
(GNU Make itself then exits `2` for the failed target, as it does for every
target in this repository, and reports the recipe's own status in its
`Error <status>` line.)

The non-publishable warning is always printed to stderr. The success
statement —

```text
P3.2 smoke completed: correctness passed before warm-up and steady-state timing.
```

— is printed **only after a zero exit status**, and claims nothing about the
order beyond what actually holds. A failing run instead states that no CSV
header and no CSV row were emitted. The real order is always:

```text
compile -> first launch -> full correctness validation
        -> warm-up -> steady-state timing -> CSV
```

## 11. Expected successful evidence

* the safe launcher proves the selected GPU has no active compute process;
* exactly one device is visible inside the container and its UUID matches;
* the upstream commit and SHA-256 re-check passes inside that same GPU
  container, immediately before execution;
* stderr shows provenance collection, `can_implement: OK`, compilation, the
  first launch, `correctness: PASS`, warm-up, and steady state, in that order;
* stdout carries exactly one CSV header and one data row — and nothing else —
  with `correctness=PASS`, `run_kind=smoke`, `variant=nonpersistent_1cta`,
  `m=n=k=4096`, `l=1`, `use_2cta_instrs=false`, `use_tma_store=true`,
  `cache_mode=hot`, and `publishable=false`;
* all three timings are finite, strictly positive, and separated;
* the command exits `0`, and only then is the success statement printed to
  stderr.

## 12. Failure behaviour

| Failure | Behaviour |
|---------|-----------|
| More or fewer than one visible GPU | abort before any tensor, exit 1 |
| Wrong compute capability, toolkit, torch, or CuTe DSL version | abort, exit 1 |
| Missing/dirty `/opt/cutlass`, wrong HEAD, blob, or SHA-256 | abort before import, exit 1 |
| `can_implement()` false | abort, exit 1, no fallback configuration |
| `fp32_precision` unavailable, rejected, or not read back as exactly `ieee` | stderr diagnostic, exit 1, **no CSV**, no fallback reference |
| Correctness mismatch | stderr diagnostic, exit 1, **no CSV**, no warm-up, no steady state |
| A timing not finite and strictly positive | abort, exit 1, no row |
| Any row-contract violation | abort, exit 1, no row |
| Out-of-range `--warmup-iterations` / `--iterations` | argparse usage error, exit 2 |

## 13. Separation from P3.3–P3.5 and Phase 4

* **P3.3** adds the cuBLASLt baseline. The untimed PyTorch FP32 oracle used
  here is *not* that baseline and must never be described as one.
* **P3.4** adds the three execution variants (non-persistent 1-CTA, persistent
  1-CTA, persistent 2-CTA). P3.2 has exactly one variant,
  `nonpersistent_1cta`, and can express no other.
* **P3.5** adds the remaining four final shapes and the comparison. P3.2
  executes only the first shape and produces no comparison.
* **Phase 4** owns campaign orchestration, the pilot and three final
  campaigns, aggregation, manifests, statistics, figures, and the integrated
  analysis. P3.2 writes no result file and creates no campaign directory.

## 14. Status and verification record

```text
Implemented:        YES
Audited:            YES   (independent technical re-audit PASS)
Verified on GB300:  YES   (preflight and frozen smoke PASS)
```

An independent audit of the first implementation (project commit
`ea501d4c43b2cf364ac419ddefa3ae84b564581e`) found two blockers, both since
remediated:

1. **Mixed PyTorch FP32/TF32 control APIs.** The original guard set the legacy
   `allow_tf32` flag, the new `fp32_precision` property, *and*
   `set_float32_matmul_precision()`, then accepted a readback of `none` as if
   it proved IEEE FP32. In PyTorch 2.10 those APIs are views of one setting,
   mixing them is unsupported and order-dependent, and `none` is merely the
   unset default. The guard now uses the new API exclusively, requires an exact
   `ieee` readback, rejects `none`, fails closed when the API is absent, and
   restores only the new-API value (section 6.4).
2. **Contaminated smoke stdout and an unconditional success claim.** The target
   echoed its own recipe, the launcher wrote two lines to stdout, the NGC
   entrypoint banner went to stdout, and a "correctness passed" line was
   printed even when the run failed before the wrapper started. All four are
   fixed structurally (section 10), with the success statement now gated on a
   zero exit status.

The remediation at commit
`c8b3e2ee57e0297940e0fd5864583ec12dfb23e3` was then independently
re-audited. Both technical blockers were closed, and no remaining code blocker
was found in the timer boundaries, complete-result validation, CSV contract,
provenance, or launcher safety. The re-audit identified only three stale
documentary statements — two in `README.md` and one in this protocol — all
corrected by the closure update containing this record.

The independent audit environment could not repeat the Docker-backed check and
could not finish one unrelated legacy P1.4 self-test because Docker was absent
and `/dev/urandom` was unavailable there. Direct syntax checks, wrapper and
checker self-tests, the adversarial contract review, source inspection, and the
simulated success/failure launcher paths passed. The later real GB300 run above
then exercised the pinned container and complete GPU path successfully.

Executed GPU-free: `py_compile` of both Python files, the wrapper's `--help`
and `--self-test`, the checker and its `--self-test`, `make check-static`,
`make gemm-cutedsl-p32-check` in the pinned image, and a simulated
failure-path and success-path run of the smoke target against stub
`docker`/`nvidia-smi` executables (no GPU, no network) proving empty stdout on
failure and exactly two valid CSV lines on success. Those are author
self-checks and are not, by themselves, the independent audit or GB300
verification recorded above.

Executed on GB300: fresh preflight `20260806T163806Z` and
`make gemm-cutedsl-p32-smoke` on the explicitly selected physical GPU index
`7`. Both completed successfully. The smoke ran the exact frozen shape with
full correctness before warm-up and steady-state timing, and every emitted
field remained `publishable=false`.

**P3.2 creates no publishable performance result.** No TFLOP/s, no speedup, no
efficiency, no comparison, and no cuBLASLt baseline exists at this point in the
project, and nothing in this unit says or implies that a CuTe DSL GEMM
approaches cuBLASLt.
