# P3.1 — Pinned official CuTe DSL example (frozen protocol)

Status: `P3.1 = YES / YES / YES` (Implemented / Audited / Verified on GB300).
The independent audit and physical verification record are in section 11;
the author's own GPU-free checks alone are **not** an independent audit.

## 1. Objective

Establish that one pinned, unmodified, official NVIDIA CuTe DSL dense GEMM
example can be located, provenance-checked, and executed correctly in this
repository's pinned environment on the target hardware.

That is the whole unit. **P3.1 produces no experimental result.**

## 2. Non-objectives

P3.1 deliberately does **not** introduce a repository-owned CuTe GEMM
implementation, a reusable one-shape wrapper, persistent variants, 2-CTA MMA
instructions, a cuBLASLt baseline, a shape or parameter sweep, autotuning,
Nsight Compute profiling, SASS performance analysis, campaign infrastructure,
CSV/JSON result datasets, performance comparisons, or publishable
measurements. Those belong to P3.2–P3.5 and Phase 4.

No Phase 3 performance or compatibility claim may be made from this unit, and
none is made here.

## 3. Exact upstream source

```text
Repository:   NVIDIA/cutlass
Tag:          v4.6.1
Commit:       e05f953a5b3d38adc240df2ff928e0421c2abba3
Relative path: examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py
Git blob SHA: 6c6144bc88896cffb3c8c4692ca915f993c71e1d
SHA-256:      f99bc4cc1e0aea8990e2929d7c703dfc8196d797b7c9f5a889eabcd3c4ff67ec
License:      BSD-3-Clause (NVIDIA CORPORATION & AFFILIATES)
URL:          https://github.com/NVIDIA/cutlass/blob/v4.6.1/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py
```

The same values are pinned in the two version contracts: `CUTLASS_COMMIT` in
the global `VERSIONS.env`, and `CUTEDSL_P31_EXAMPLE_PATH`,
`CUTEDSL_P31_EXAMPLE_GIT_BLOB`, and `CUTEDSL_P31_EXAMPLE_SHA256` in the
Phase 3-only `PHASE3_VERSIONS.env` (see section 9). Every P3.1 check reads
them from there; no provenance constant is duplicated anywhere else.

`dense_gemm_persistent.py` and every other example in that directory are out
of scope for P3.1. Persistent execution is P3.4.

## 4. Why the source is executed from `/opt/cutlass`, not copied

The pinned image already checks out CUTLASS at exactly `CUTLASS_COMMIT` under
`/opt/cutlass` (see `Dockerfile`), so the example is executed in place at

```text
/opt/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py
```

Executing in place rather than vendoring approximately 1,800 lines of NVIDIA
code:

- keeps the file **byte-identical** to the tagged upstream release, which is
  what the Git blob SHA and SHA-256 checks actually prove;
- removes any possibility of an accidental local patch, reformat, or drift;
- keeps authorship unambiguous — this repository claims none of that code;
- keeps the P3.1 diff small enough to audit in one sitting.

The upstream file is never modified, and the checkout is never written to:
Git is queried read-only, with a per-invocation `-c safe.directory=/opt/cutlass`
because the checkout is root-owned inside the image while the container runs
as the invoking user.

## 5. Exact frozen command

```bash
python3 /opt/cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py \
  --mnkl 256,256,512,1 \
  --ab_dtype BFloat16 \
  --c_dtype Float32 \
  --acc_dtype Float32 \
  --a_major k \
  --b_major k \
  --c_major n \
  --mma_tiler_mn 128,128 \
  --cluster_shape_mn 1,1 \
  --use_tma_store \
  --warmup_iterations 0 \
  --iterations 1
```

`--use_2cta_instrs`, `--skip_ref_check`, and `--use_cold_l2` are **never**
passed. `make gemm-cutedsl-p31-smoke` runs exactly this command and cannot
disable reference checking; `make check-static` fails if the frozen options
change or if any of those three flags appears in the Makefile.

## 6. Numeric and layout contract

| Aspect | Frozen value |
|--------|--------------|
| Operation | `C = A × B`, batched dense GEMM |
| Input dtype | BF16 × BF16 (`--ab_dtype BFloat16`) |
| Accumulation | FP32 in TMEM (`--acc_dtype Float32`) |
| Output dtype | FP32 (`--c_dtype Float32`) |
| Problem | `(M,N,K,L) = (256,256,512,1)` |
| MMA tiler | `(M,N) = (128,128)` |
| Cluster | `(M,N) = (1,1)` |
| MMA group | 1 CTA (no `use_2cta_instrs`) |
| Execution | non-persistent (this example has no persistent scheduler) |
| A/B movement | TMA, as implemented by the official example |
| Output path | TMA store (`--use_tma_store`) |
| Tensor Core path | Blackwell `tcgen05`/UMMA |
| Epilogue | identity conversion to FP32 |
| Reference check | mandatory, performed by the unchanged example |

Mathematical interpretation, using the example's own conventions:

- `A` is `(M,K,L) = (256,512,1)`. `--a_major k` means K is contiguous
  (row-major in `M×K`).
- `B` is represented by the example as `(N,K,L) = (256,512,1)`. `--b_major k`
  means K is contiguous, i.e. the operand is stored "N-rows by K-columns" and
  the contraction runs over the contiguous K axis.
- `C` is `(M,N,L) = (256,256,1)`. `--c_major n` means N is contiguous
  (row-major in `M×N`).
- The contraction is `C[m,n,l] = Σ_k A[m,k,l] · B[n,k,l]`, computed in FP32.

The shape is deliberately small: this is a functional compatibility check, not
one of the five final `(M,N,K)` shapes of experiment 3.

## 7. What the official example does on the device

Taken from the unchanged upstream implementation (nothing here is a claim
about performance):

1. **TMA loads** move A and B tiles from GMEM to SMEM through a multi-stage
   pipeline. With `--cluster_shape_mn 1,1` no multicast partner exists, so the
   loads are effectively unicast.
2. **`tcgen05.mma` (UMMA)** performs the BF16 × BF16 multiply-accumulate. The
   MMA reads both operands from SMEM and writes the accumulator to **TMEM**;
   with one CTA per MMA group the instruction is the `cta_group::1` form.
3. **TMEM → RMEM**: the completed FP32 accumulator is loaded to registers with
   `tcgen05.ld`.
4. **Epilogue**: the accumulator is type-converted to the output type. Because
   `acc_dtype` and `c_dtype` are both `Float32`, this conversion is the
   identity; no elementwise `epilogue_op` is supplied.
5. **TMA store**: with `--use_tma_store`, C goes RMEM → SMEM → GMEM through
   TMA rather than a direct RMEM → GMEM store.

The example refuses configurations its own `can_implement()` rejects (dtype,
tiler/cluster shape, alignment, epilogue-store legality), so an invalid frozen
configuration would fail loudly rather than silently degrade.

## 8. Correctness, JIT compilation, and timing are separate

The unchanged example performs, in this order:

1. **Tensor creation** — PyTorch CPU tensors, converted to CuTe tensors via
   DLPack (`cutlass.torch`), with a fixed seed (`torch.manual_seed(1111)`).
2. **JIT compilation** — `cute.compile(...)` builds the kernel for the current
   device. Compilation cost is not a measurement and is not reported.
3. **Correctness execution** — because `--skip_ref_check` is never passed, the
   compiled kernel runs once and its result is compared against a CPU
   `torch.einsum` FP32 reference converted to `c_dtype`, using
   `torch.testing.assert_close(atol=<tolerance>, rtol=1e-05)` with the
   example's default tolerance (`1e-01`). A mismatch raises and the process
   exits non-zero.
4. **Internal timing** — the example then calls its own benchmarking helper
   (`cutlass.cute.testing.benchmark`) with `warmup_iterations=0` and
   `iterations=1` and returns the execution time in microseconds. Its
   `__main__` block discards that return value and prints `PASS`.

The official script is **not** patched to remove step 4. Any timing produced
or internally computed during P3.1 is explicitly classified as
**non-publishable functional-smoke output**: it is one un-warmed iteration at a
tiny shape, it is not recorded, not aggregated, not converted to TFLOP/s, and
must never be cited.

Correctness therefore precedes any timing, as required by `AGENTS.md`.

## 9. Runtime dependencies (auxiliary, exactly pinned) and the two contracts

### 9.0 Where Phase 3 pins live

The global version contract `VERSIONS.env` is **closed and unchanged**. It is
byte-for-byte identical to the version on `main`, because two audited, closed
P1/P2 files parse it against a closed key allowlist and reject any unknown
key:

```text
scripts/aggregate_exp01_memory_paths.py   parse_versions_env()
scripts/aggregate_exp02_umma_throughput.py parse_versions_env()
```

Every Phase 3-only pin therefore lives in a separate root-level file,
`PHASE3_VERSIONS.env`, which **extends but never alters** the global contract:
it redefines no CUDA, image, CUTLASS, architecture, or build-job value, and
those remain defined exclusively in `VERSIONS.env`. Both files use plain
`KEY=VALUE` syntax and are `include`d by the `Makefile`; the `Dockerfile`
receives every value as an explicit build argument.

`make check-static` enforces this in both directions: it fails if a Phase 3 key
appears in `VERSIONS.env` or if a global key is redefined in
`PHASE3_VERSIONS.env`, and it imports both real aggregator modules and calls
their real `parse_versions_env()` against the repository's actual
`VERSIONS.env`, so a future regression is caught GPU-free rather than during a
P1/P2 campaign finalize.

### 9.1 The pinned Python dependencies

The official example uses PyTorch for allocation, DLPack interoperability,
CUDA stream access (`torch.cuda.current_stream()`), and the CPU reference.
`PHASE3_VERSIONS.env` pins the auxiliary Python dependencies exactly:

```text
PYTORCH_VERSION=2.10.0+cu130
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
PYTORCH_CUDA_VERSION=13.0
CUDA_PYTHON_VERSION=13.0.3
CUDA_BINDINGS_VERSION=13.0.3
```

They are installed into the existing `/opt/venv` virtual environment, PyTorch
coming from the official PyTorch index — never `latest`, never an unversioned
`pip install torch`, never a nightly, never Conda, never a second Python
environment.

`torch 2.10.0+cu130` requires `cuda-bindings==13.0.3`, whereas the CuTe DSL
v4.6.1 installer resolves `cuda-python 13.3.1` (which requires
`cuda-bindings~=13.3.1`). Pinning `cuda-python` to its matching `13.0.3`
release resolves that dependency graph coherently: `13.0.3` satisfies torch's
exact requirement and CuTe DSL 4.6.1's own `cuda-python>=12.8` constraint at
the same time. Nothing is uninstalled, excluded, or allowlisted to hide a
conflict, and no pin in `VERSIONS.env` is weakened or changed.

The resulting environment must satisfy

```bash
python3 -m pip check
```

```text
No broken requirements found.
```

and this is a hard, unsuppressed gate in three places: during the image build
(after every Python package is installed), in `make check-env`, and in
`make gemm-cutedsl-p31-check`. It is never run as `pip check || true`, never
filtered, and never downgraded to a warning.

The same three places also verify exact versions. The build gate reads all four
pinned distributions through `importlib.metadata` (`torch`, `cuda-python`,
`cuda-bindings`, `nvidia-cutlass-dsl`); `check-env` and
`gemm-cutedsl-p31-check` read `cuda-python` and `cuda-bindings` through
`importlib.metadata` and re-read `torch.__version__`, `torch.version.cuda`, and
`cutlass.__version__` at runtime. In the image, the coherent
`cuda-python`/`cuda-bindings` pair is installed before the pinned PyTorch
build, so the environment is never left transiently inconsistent during the
build either. No GPU is used or required for any of it.

**The `cu130` PyTorch wheel is an auxiliary allocation/reference dependency
only. It does not replace the pinned CuTe DSL toolchain**, which remains:

```text
CUDA Toolkit:       13.1.0
CUTLASS/CuTe DSL:   v4.6.1
Architecture:       sm_103a
```

No existing CUDA, image-digest, CUTLASS, architecture, or build-job pin is
changed by P3.1, and neither audited P1/P2 aggregator is modified or given an
extended allowlist: `VERSIONS.env` is simply left alone.

## 10. Verification commands

GPU-free (no GPU, no network at runtime, no elevated privileges):

```bash
git diff --check
git diff --exit-code main -- VERSIONS.env
make check-static
make build-image
make check-env
make gemm-cutedsl-p31-check
```

`make check-static` additionally exercises the two real, unmodified P1/P2
version parsers against the real global version file:

```bash
python3 -c 'import sys; sys.path.insert(0, "scripts"); import aggregate_exp01_memory_paths as p1, aggregate_exp02_umma_throughput as p2; p1.parse_versions_env(); p2.parse_versions_env(); print("P1/P2 VERSIONS.env compatibility: PASS")'
```

`make gemm-cutedsl-p31-check` fails closed unless all of the following hold:

1. `/opt/cutlass` exists.
2. Its HEAD is exactly `e05f953a5b3d38adc240df2ff928e0421c2abba3`.
3. The checkout has no tracked or untracked modification.
4. The example is a non-symlink regular file.
5. Its Git blob SHA is `6c6144bc88896cffb3c8c4692ca915f993c71e1d`.
6. Its SHA-256 is
   `f99bc4cc1e0aea8990e2929d7c703dfc8196d797b7c9f5a889eabcd3c4ff67ec`.
7. CuTe DSL reports `4.6.1`.
8. PyTorch reports `2.10.0+cu130`.
9. `torch.version.cuda` reports `13.0`.
10. `importlib.metadata` reports `cuda-python 13.0.3` and
    `cuda-bindings 13.0.3`.
11. `python3 -m pip check` reports no broken requirements.
12. The example's own `--help` exits successfully without touching a GPU.
13. Every frozen CLI option is present in that help output.

GB300 verification (only with an explicitly supplied, free physical device —
the project never selects a GPU automatically):

```bash
BLACKWELL_GPU_INDEX=<physical-index> make preflight
BLACKWELL_GPU_INDEX=<same-physical-index> make gemm-cutedsl-p31-smoke
```

Expected successful evidence:

- the safe launcher proves the selected GPU has no active compute process;
- exactly one device is visible inside the container and its UUID matches;
- the upstream commit and SHA-256 re-check passes **inside that same GPU
  container**, immediately before execution;
- the example prints the frozen BF16/FP32 configuration
  (`mnkl: (256, 256, 512, 1)`, `AB dtype: BFloat16`, `C dtype: Float32`,
  `Acc dtype: Float32`, `2CTA MMA instructions: False`,
  `Use TMA Store: True`);
- `Skip reference checking: False`;
- the official example exits `0` and its last line is `PASS`.

`make gemm-cutedsl-p31-smoke` preserves the official program's exit code and
then prints an explicit notice that the run was a functional smoke check and
not a performance result.

## 11. Status and verification record

```text
Implemented:        YES
Audited:            YES   (independent review: PASS)
Verified on GB300:  YES
```

The remediated implementation at project Git commit
`f34cb33a9456ba011feb0a5a35910bbd00f9a9e6` passed an independent audit before
the physical verification. The published branch still pointed to that exact
commit when this closing record was prepared.

At `2026-08-06T10:16:57Z`, the operator ran

```bash
BLACKWELL_GPU_INDEX=3 make preflight
```

on an explicitly selected physical `NVIDIA B300 SXM6 AC`. The safe launcher
resolved physical index `3` to UUID
`GPU-90fb226c-3937-2448-1052-2e12282a61b9`, reported driver `610.43.02`, and
proved that the device had no active compute processes before exposing it to
the container. The container reported CUDA `13.1.0`. Preflight campaign
`20260806T101657Z`, recorded under
`results/preflight/20260806T101657Z/summary.json`, passed all six checks:
`gpu_visibility`, `tool_versions`, `cuda_smoke_compile`,
`cuda_smoke_run`, `cutedsl_smoke`, and `ncu_profile`; its final status was
`OVERALL=PASS`.

Immediately afterwards, on the same explicitly selected device, the operator
ran

```bash
BLACKWELL_GPU_INDEX=3 make gemm-cutedsl-p31-smoke
```

The GPU container re-checked upstream CUTLASS commit
`e05f953a5b3d38adc240df2ff928e0421c2abba3` and example SHA-256
`f99bc4cc1e0aea8990e2929d7c703dfc8196d797b7c9f5a889eabcd3c4ff67ec`.
The unchanged official example reported the frozen configuration:

```text
mnkl: (256, 256, 512, 1)
AB dtype: BFloat16, C dtype: Float32, Acc dtype: Float32
Mma Tiler (M, N): (128, 128), Cluster Shape (M, N): (1, 1)
2CTA MMA instructions: False
Use TMA Store: True
Warmup iterations: 0
Iterations: 1
Skip reference checking: False
Use cold L2: False
PASS
```

The command exited successfully. The warnings emitted by the unchanged
upstream example were non-fatal API deprecation/named-barrier warnings and did
not disable reference validation or alter the final `PASS`.

This closes P3.1 as `YES / YES / YES`. It remains a functional compatibility
check only: no timing is retained, no TFLOP/s is computed, no cuBLASLt baseline
exists, and no publishable performance result is created.

## 12. Files P3.1 owns

```text
src/gemm/P3_1_PROTOCOL.md   this document
PHASE3_VERSIONS.env         the Phase 3-only version contract (section 9.0)
```

P3.1 adds no source, no runner, no wrapper, no campaign directory, and no
result file. Its remaining footprint is confined to the auxiliary-dependency
installs and their verification in `Dockerfile`, the
`gemm-cutedsl-p31-check` / `gemm-cutedsl-p31-smoke` targets plus their
validation in `Makefile`, and the truthful status text in `PLAN.md` and
`README.md`. `VERSIONS.env` is **not** part of that footprint: it is identical
to the version on `main`, which `git diff --exit-code main -- VERSIONS.env`
proves.
