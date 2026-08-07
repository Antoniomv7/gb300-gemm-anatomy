# P3.4 — Three CuTe DSL execution variants (frozen protocol)

Status: `P3.4 = YES / NO / NO` (Implemented / Audited / Verified on GB300).
The author's own GPU-free checks are **not** an independent audit, and no
GB300 execution of this unit has happened yet. Section 12 records exactly what
was and was not run.

## 1. Purpose and scope

P3.2 established one CuTe DSL execution variant at the first of the five final
shapes. P3.3 established the cuBLASLt baseline for exactly the same geometry.
P3.4 adds the two remaining execution variants the project plan froze, so that
all three exist under one identical operand set, one identical correctness
oracle, and one identical timing discipline.

P3.4 exercises exactly one shape:

```text
(M,N,K,L) = (4096,4096,4096,1)
```

That is the same first final shape P3.2 and P3.3 use. P3.4 adds none of the
other four shapes, and it runs exactly three fixed candidates — one per
execution variant — with no autotuning and no candidate search.

**P3.4 creates no publishable performance result.** It is an implementation
and functional-verification unit, not an experimental campaign, and it makes no
performance claim of any kind.

## 2. The exact three-candidate table

| Variant | Upstream class | Scheduler | MMA tiler `(M,N)` | Cluster `(M,N)` | `use_2cta_instrs` | Upstream source |
|---------|----------------|-----------|------------------:|----------------:|-------------------|-----------------|
| `nonpersistent_1cta` | `DenseGemmKernel` | non-persistent | `(128,128)` | `(1,1)` | `false` | P3.1's pinned `dense_gemm.py` |
| `persistent_1cta` | `PersistentDenseGemmKernel` | static persistent | `(128,128)` | `(1,1)` | `false` | pinned `dense_gemm_persistent.py` |
| `persistent_2cta` | `PersistentDenseGemmKernel` | static persistent | `(256,128)` | `(2,1)` | `true` | pinned `dense_gemm_persistent.py` |

They always run in exactly this order. There is no fourth candidate, no
variant selection, and no way to run fewer than all three.

### 2.1 Why the 2-CTA row uses a `(256,128)` tiler

The 2-CTA configuration deliberately uses an M tile of 256 so that each of the
two participating CTAs retains a **local M extent of 128**:

```text
tiler M / cluster M = 256 / 2 = 128
```

This matches the project's own P2.2 two-SM geometry, and it is the shape
NVIDIA's persistent example documents for `use_2cta_instrs=True`: that file
states that the MMA tiler M must be 128 or 256 when `use_2cta_instrs=True`, and
that the cluster M must be a multiple of 2. Substituting `(128,128)` here would
halve the per-CTA M extent and stop being the two-SM geometry the rest of the
project measured, so no other tiler or cluster is ever silently substituted —
the checker rejects any table whose 2-CTA row does not satisfy
`tiler_M / cluster_M == 128`.

## 3. Frozen scientific contract

All three variants share this, unchanged from P3.2 and P3.3:

```text
Operation:       C[m,n,l] = sum_k A[m,k,l] * B[n,k,l]
Input dtype:     BF16 x BF16
Accumulation:    FP32 in TMEM
Output dtype:    FP32
A major:         k
B major:         k
C major:         n
TMA loads:       enabled
TMA store:       enabled
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

## 4. The two pinned upstream sources

This repository owns no GEMM kernel. Nothing is written, copied, vendored,
forked, patched, or reformatted. Both files are loaded read-only and in place
out of the pinned `/opt/cutlass` checkout, under private module names so that
neither `if __name__ == "__main__"` block ever executes, and **neither upstream
`run()` is ever called**.

### 4.1 Non-persistent source (unchanged from P3.1)

```text
Repository: NVIDIA/cutlass
Commit:     e05f953a5b3d38adc240df2ff928e0421c2abba3
Path:       examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py
Git blob:   6c6144bc88896cffb3c8c4692ca915f993c71e1d
SHA-256:    f99bc4cc1e0aea8990e2929d7c703dfc8196d797b7c9f5a889eabcd3c4ff67ec
License:    BSD-3-Clause
```

### 4.2 Persistent source (new in P3.4)

```text
Repository: NVIDIA/cutlass
Commit:     e05f953a5b3d38adc240df2ff928e0421c2abba3   (the SAME pinned commit)
Path:       examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py
Git blob:   10d62d239457748372a522488ee23bc3df5f346d
SHA-256:    d59344faf902cb215a2cee3f2ae6415a14589c6ad8f93e5e74e2612c1e6a0810
License:    BSD-3-Clause
```

Both pins were resolved on 2026-08-07 by read-only query against the pinned
checkout, and they live in `PHASE3_VERSIONS.env` under the P3.4 keys
`CUTEDSL_P34_PERSISTENT_EXAMPLE_PATH`,
`CUTEDSL_P34_PERSISTENT_EXAMPLE_GIT_BLOB`, and
`CUTEDSL_P34_PERSISTENT_EXAMPLE_SHA256`. **No Phase 3 key is added to
`VERSIONS.env`**, which the closed P1/P2 aggregators parse against their own
closed allowlist, and the checker re-runs both of those parsers to prove it.
The wrapper and the checker duplicate none of these values as literals: every
one is read from the two contracts at run time.

Before either module is imported, P3.4 revalidates the CUTLASS `HEAD`, the
clean working tree, and — for **each** file — the non-symlink regular-file
identity, the Git blob, and the SHA-256. The whole verification is repeated
after provenance collection and the two observations must agree.

P3.4 also proves the two sources are genuinely different files (different path,
different blob, different digest) and that the two kernel classes are distinct
objects, so the persistent rows cannot silently measure the non-persistent
scheduler.

## 5. Operand equivalence

All three variants consume **byte-identical** A and B.

The operands are built once, by the pinned non-persistent example's own
`create_tensors()` — the same factory, the same seed `1111`, the same A/B/C
creation order, the same dtypes, and the same layouts and strides that P3.2 and
P3.3 use. A and B are allocated once, reused by all three variants, and never
mutated.

The persistent example's own independent tensor-generation path
(`prepare_tensors`) is deliberately **not** used: it would weaken equivalence
with P3.2 and P3.3.

Only the output buffer is reset between candidates. It is reset **to NaN**,
outside every timer and followed by a synchronize, so that any element a kernel
fails to write remains non-finite and is rejected by the complete-result check
instead of surviving as a stale value from the previous variant that would
silently pass.

The reference is computed once, outside every timer, and reused — which is
correct precisely because A and B are identical and immutable across the three
variants.

## 6. Persistent scheduler and `max_active_clusters`

The two persistent variants use the official static persistent tile scheduler
of the pinned persistent example (`utils.StaticPersistentTileScheduler`), and
the 2-CTA row reaches its `tcgen05.CtaGroup.TWO` selection through the
upstream `use_2cta_instrs` path. The checker asserts both markers are present
in the pinned file, and asserts the non-persistent file contains neither.

`max_active_clusters` is obtained from the official pinned hardware helper, for
that variant's own cluster size:

```python
cutlass.utils.HardwareInfo().get_max_active_clusters(cluster_m * cluster_n)
```

This is exactly the helper the pinned persistent example itself uses. It is
never guessed, hard-coded, exposed as a CLI option, or read from an environment
override. The returned value must be a finite positive integer, and it is
recorded per row. The non-persistent row records the canonical string
`not_applicable` — never a number, never zero, never an empty field.

## 7. Execution order and timer boundaries

For **every** variant, in this exact order:

1. verify the environment, the repository provenance, and **both** upstream
   source identities;
2. allocate the shared operands — outside every timer;
3. build the exact frozen variant object from the source the frozen table names;
4. run that class's own official `can_implement()` check (the two upstream
   classes deliberately expose different signatures:
   `DenseGemmKernel.can_implement(a, b, c)` versus
   `PersistentDenseGemmKernel.can_implement(mnkl, a_dtype, b_dtype, c_dtype,
   a_major, b_major, c_major)`, and each is called in its own official form);
5. for a persistent variant, query and validate `max_active_clusters`;
6. reset C and synchronize — outside the timers;
7. measure **`compile_time_ms`** with a monotonic host clock around
   `cute.compile()` only;
8. measure **`first_launch_ms`** with a synchronized monotonic host interval
   around the first launch;
9. validate that complete first-launch result against the untimed IEEE-FP32
   CUDA oracle;
10. only after that variant passes correctness, run its warm-up launches;
11. measure steady-state **`kernel_time_ms`** with CUDA events on the kernel's
    own stream, divided by the measured iteration count;
12. build — but do not yet emit — that variant's row.

### 7.1 Launch signature

`cute.compile` bakes every `cutlass.Constexpr` parameter in at compile time and
drops it from the compiled callable, which therefore takes only the dynamic
arguments `(a, b, c, stream)`. Both pinned examples demonstrate exactly this:
the non-persistent one compiles `(gemm, a, b, c, stream)` and calls
`(a, b, c, stream)`; the persistent one compiles
`(bmm, gemm, a, b, c, max_active_clusters, stream, epilogue_op)` and also calls
`(a, b, c, stream)`. P3.4 therefore compiles the persistent kernel with
`max_active_clusters` and launches without it. A `TypeError` at that boundary
is treated as a hard failure with an explicit diagnostic, never worked around
by guessing another signature.

## 8. Correctness oracle

Identical to the closed P3.2/P3.3 policy:

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
  `|c − ref| ≤ atol + rtol·|ref|`;
* non-finite results and non-finite references are rejected;
* `max_abs_error` and `max_rel_error` are finite, non-negative diagnostics;
  `max_rel_error` uses the same safe denominator floor P3.2 and P3.3 use and
  never participates in the pass/fail decision.

Neither upstream benchmarking helper and neither upstream `run()` is used: they
fuse compilation, the first launch, correctness, and benchmarking into one
number and therefore cannot provide the separation this unit exists to give.

### 8.1 Failure behaviour

A failed provenance check, unsupported configuration, compilation error, launch
error, non-finite result, shape mismatch, or correctness mismatch:

* exits non-zero;
* prints a concise diagnostic to stderr;
* skips warm-up and steady-state timing for the failed variant;
* emits **no** CSV header and **no** CSV row — **including no rows from
  variants that already passed**.

The complete output is buffered and emitted only after all three variants have
passed, so a failure in any of the three positions yields an empty stdout
rather than a truncated table. The checker proves this by injecting a synthetic
failure at each of the three positions in turn.

## 9. CSV output contract

Frozen schema `schema_version=p34.v1`. The closed `p32.v1` and `p33.v1` schemas
are neither modified nor reinterpreted.

A normal successful run writes to stdout exactly one CSV header, exactly three
CSV rows in the frozen variant order, and exactly four lines in total.
Everything else — including JIT and compiler output — goes to stderr, using the
same file-descriptor-level redirection P3.2 uses so that native writes to
descriptor 1 cannot corrupt the CSV.

### 9.1 Exact ordered 51-field schema

```text
schema_version
experiment
unit
run_kind
method
variant
scheduler
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
max_active_clusters
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
upstream_kernel_file
upstream_kernel_git_blob
upstream_kernel_sha256
git_commit
git_dirty
publishable
```

### 9.2 Frozen categorical values

```text
experiment=exp03_cutedsl_vs_cublaslt
unit=P3.4
run_kind=smoke
method=cutedsl
reference=torch_cuda_fp32_ieee
correctness=PASS
cache_mode=hot
publishable=false
```

`scheduler=nonpersistent` for `nonpersistent_1cta`, and
`scheduler=static_persistent` for both persistent rows.

`upstream_kernel_file`, `upstream_kernel_git_blob`, and
`upstream_kernel_sha256` identify the source **actually used by that row**, so
the non-persistent row names a different file from the two persistent rows, and
the two persistent rows name the same one.

### 9.3 Serialization rules

* Python's `csv` module, never string concatenation.
* Missing, duplicate, unknown, and reordered fields are rejected.
* Booleans are canonical lowercase `true` / `false`.
* Timing fields: fixed-point milliseconds with exactly six fractional digits.
* Tolerance and error fields: fixed-point with exactly nine fractional digits.
* All three timings must be finite and strictly positive.
* `max_abs_error` and `max_rel_error` must be finite and non-negative.
* `max_active_clusters` is `not_applicable` for the non-persistent row only,
  and a positive decimal integer for both persistent rows.
* No field is empty and none contains exponent notation or locale-dependent
  formatting.
* A row can only be built through a function that refuses any `correctness`
  value other than `PASS`, and it takes the variant's tiler, cluster,
  scheduler, and 2-CTA flag from the frozen table rather than from the caller.

No TFLOP/s, FLOP/s, speedup, efficiency, bandwidth, utilization, ranking,
winner, or comparison field exists. Every row is `publishable=false`. **These
timings are functional diagnostics only.**

## 10. Command line

```text
--warmup-iterations   integer 1..100, default 5
--iterations          integer 1..100, default 20
--self-test
--help
```

The normal execution path always runs all three variants, in the frozen order.
There is no shape, dtype, layout, variant, scheduler, tiler, cluster,
persistence, 2-CTA, seed, tolerance, source-path, or correctness control.

`--help` and `--self-test` are genuinely GPU-free: they import neither PyTorch,
nor CuTe DSL, nor the CUDA bindings, nor either upstream example. The checker
proves this by running both behind an import guard that turns any such import
into a hard failure.

## 11. Make targets

### `gemm-cutedsl-p34-check` (GPU-free)

Depends on `gemm-cublaslt-p33-check`, preserving the existing chain (which in
turn runs the unmodified P3.2 and P3.1 gates). It runs inside the pinned image
with no GPU, `--network none`, `--security-opt no-new-privileges`,
`--cap-drop ALL`, the invoking UID/GID, the repository mounted read-only, and
Python cache plus every temporary file under container-private `/tmp`. It:

* revalidates the CUTLASS checkout (HEAD and clean working tree);
* validates **both** official source files — regular file, not a symlink, Git
  blob, SHA-256 — and asserts they are two different files;
* asserts the persistent source really contains
  `StaticPersistentTileScheduler`, `CtaGroup.TWO`, and
  `class PersistentDenseGemmKernel`, and that the non-persistent source
  contains `class DenseGemmKernel`;
* verifies the pinned CuTe DSL / PyTorch / cuda-python / cuda-bindings
  versions;
* runs `python3 -m pip check`;
* compiles the P3.4 Python files;
* runs the wrapper's `--help` and `--self-test`;
* runs the checker's `--self-test` and the full repository check.

### `gemm-cutedsl-p34-smoke` (GPU)

The only P3.4 GPU target. Its first recipe action rejects a missing or
non-numeric `BLACKWELL_GPU_INDEX` before Docker, compilation, or any other work
— which is why it deliberately has no Make prerequisite that could run first.
It then uses only `scripts/run_container.sh`, which alone resolves the physical
index to a UUID, proves the device has no active compute processes, exposes
exactly that one UUID, and re-verifies inside the container that exactly one
matching GPU is visible as CUDA logical device 0. `--gpus all` is never used
and a GPU is never selected automatically.

Inside that same GPU container it revalidates the pinned commit and **both**
source SHA-256 digests, then runs the wrapper with exactly two warm-ups and ten
measured launches per variant, preserves the wrapper's exit status, preserves
the four-line CSV-only stdout contract, and prints a conspicuous stderr notice
that the run is functional verification only, that all timings are
non-publishable, and that no variant or cuBLASLt comparison has been performed.

It never invokes the P3.3 cuBLASLt executable.

## 12. What was and was not run

### 12.1 GPU-free acceptance commands performed by the author

```bash
git diff --check
python3 -m py_compile \
  src/gemm/cutedsl_variants.py \
  scripts/check_cutedsl_variants_p34.py
python3 src/gemm/cutedsl_variants.py --self-test
python3 scripts/check_cutedsl_variants_p34.py --self-test
python3 scripts/check_cutedsl_variants_p34.py .
make check-static
make gemm-cutedsl-p34-check
```

**These are the author's own self-checks. They are not an independent audit,
and GPU-free checks are not GB300 verification.**

### 12.2 GB300 commands not yet performed

Neither of the following has been run, and no P3.4 GPU result of any kind
exists:

```bash
BLACKWELL_GPU_INDEX=<physical-index> make preflight
BLACKWELL_GPU_INDEX=<same-physical-index> make gemm-cutedsl-p34-smoke
```

The following therefore remain unproven on hardware and must be treated as open
until an explicitly authorized GB300 run closes them:

* that all three variants compile and launch on `sm_103a`, in particular that
  `PersistentDenseGemmKernel.can_implement()` accepts the `(256,128)` tiler
  with the `(2,1)` cluster at this shape;
* the value the official helper returns for `max_active_clusters` at cluster
  sizes 1 and 2;
* that the compiled persistent callable really takes the dynamic-only
  `(a, b, c, stream)` signature (section 7.1);
* that every variant passes the complete-result correctness check;
* all nine timings.

## 13. Non-goals and publication policy

P3.4 adds none of: the other four final shapes; cuBLASLt execution or
comparison; TFLOP/s or any derived performance claim; winner selection or
ranking; autotuning; more than the three frozen candidates; a campaign
directory, manifest, aggregator, report, plot, or result file; Nsight Compute;
SASS extraction or proprietary-kernel analysis; cold-cache experiments; new
CUDA kernels; copied NVIDIA source; new package or image versions; automatic
GPU selection; or any commit, push, branch, or pull request.

Comparing the three variants against each other, or against the P3.3 cuBLASLt
baseline, is **P3.5's** job. P3.5 is unimplemented and Phase 3 remains open. No
publishable P3.4 result and no CuTe-versus-cuBLASLt comparison exists anywhere
in this repository.
