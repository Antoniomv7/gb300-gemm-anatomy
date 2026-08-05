# P3.1 — Pinned official CuTe DSL example (frozen protocol)

Status: `P3.1 = YES / NO / NO` (Implemented / Audited / Verified on GB300).
The author's own GPU-free checks are **not** an independent audit.

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

The same values are pinned in `VERSIONS.env` as `CUTLASS_COMMIT`,
`CUTEDSL_P31_EXAMPLE_PATH`, `CUTEDSL_P31_EXAMPLE_GIT_BLOB`, and
`CUTEDSL_P31_EXAMPLE_SHA256`. Every P3.1 check reads them from there; no
provenance constant is duplicated anywhere else.

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

## 9. Runtime dependency (auxiliary, exactly pinned)

The official example uses PyTorch for allocation, DLPack interoperability,
CUDA stream access (`torch.cuda.current_stream()`), and the CPU reference. The
version contract pins it exactly:

```text
PYTORCH_VERSION=2.10.0+cu130
PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
PYTORCH_CUDA_VERSION=13.0
```

It is installed into the existing `/opt/venv` virtual environment from the
official PyTorch index — never `latest`, never an unversioned `pip install
torch`, never a nightly, never Conda, never a second Python environment.

**The `cu130` PyTorch wheel is an auxiliary allocation/reference dependency
only. It does not replace the pinned CuTe DSL toolchain**, which remains:

```text
CUDA Toolkit:       13.1.0
CUTLASS/CuTe DSL:   v4.6.1
Architecture:       sm_103a
```

No existing CUDA, image-digest, CUTLASS, architecture, or build-job pin is
changed by P3.1.

### 9.1 Recorded dependency consequence (not silent, not a weakened pin)

`torch 2.10.0+cu130` hard-requires `cuda-bindings==13.0.3`. The CuTe DSL v4.6.1
installer had resolved `cuda-bindings 13.3.1` (via `cuda-python 13.3.1`), so
installing the pinned PyTorch build **replaces `cuda-bindings 13.3.1` with
`13.0.3`** inside the image. Afterwards `pip check` reports exactly one
unsatisfied requirement:

```text
cuda-python 13.3.1 has requirement cuda-bindings~=13.3.1, but you have cuda-bindings 13.0.3.
```

This is recorded rather than hidden or worked around:

- no pin in `VERSIONS.env` is weakened, relaxed, or silently changed, and no
  package version is pinned or unpinned to make the conflict disappear;
- the image build and `make gemm-cutedsl-p31-check` both re-verify that CuTe
  DSL still reports exactly `4.6.1` after the PyTorch install, so a broken or
  downgraded CuTe DSL fails closed;
- CuTe DSL 4.6.1 itself only requires `cuda-python>=12.8`, so the downgrade
  does not violate its own declared constraint — but the resulting combination
  is *not* the one the CuTe DSL installer selected, and that difference is a
  legitimate audit finding, not a formality;
- the practical consequence is that the **GB300 verification below is what
  establishes that the CuTe DSL stack still works under this dependency
  state**; `make preflight` (which runs the Phase 0 CuTe DSL smoke) and
  `make gemm-cutedsl-p31-smoke` both exercise it.

An auditor may legitimately decide that this conflict must be resolved
differently (for example by pinning a PyTorch build compatible with
`cuda-bindings 13.3.1`, or by re-resolving the CuTe DSL stack). P3.1 does not
make that decision unilaterally.

### 9.2 Recorded interaction with the P1/P2 version-contract parsers

`scripts/aggregate_exp01_memory_paths.py` and
`scripts/aggregate_exp02_umma_throughput.py` parse `VERSIONS.env` against a
closed allowlist (`REQUIRED_VERSION_KEYS`) and reject **any** unknown key. The
six keys P3.1 adds are therefore rejected by those two audited P1/P2 files:
a *future* P1.3 or P2.3 campaign `finalize` (and so a future P1.4/P2.4 pilot)
would fail at its `versions_env` stage.

- Already-completed campaigns, their manifests, and their recorded evidence
  are unaffected: they are immutable and were finalized before this change.
- `make check-static` is unaffected: both aggregator self-tests build their own
  synthetic `VERSIONS.env` fixtures.
- P3.1 does **not** modify those two files, because extending an audited P1/P2
  parser is out of this unit's scope and would require its own re-audit.

This is reported as a known cross-unit interaction for the P3.1 audit to rule
on, not as an accepted defect.

## 10. Verification commands

GPU-free (no GPU, no network at runtime, no elevated privileges):

```bash
git diff --check
make check-static
make build-image
make check-env
make gemm-cutedsl-p31-check
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
10. The example's own `--help` exits successfully without touching a GPU.
11. Every frozen CLI option is present in that help output.

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
Audited:            NO   (requires an independent reviewer; static self-checks
                          by the author are not an audit)
Verified on GB300:  NOT RUN
```

No genuine GB300 execution of `make gemm-cutedsl-p31-smoke` has been performed
for P3.1. No operator GPU index was supplied during implementation, so no GPU
command was run, and this section must stay `NOT RUN` until a real run on a
physical B300/GB300 with an explicitly supplied `BLACKWELL_GPU_INDEX`
succeeds. When that happens, record here: the UTC timestamp, the Git commit,
the preflight campaign identifier and status, the physical GPU index and UUID,
the observed configuration lines, and the final `PASS`.

Even after a successful run, `Audited` stays `NO` until an independent review
is completed.

## 12. Files P3.1 owns

```text
src/gemm/P3_1_PROTOCOL.md   this document (the only file P3.1 adds)
```

P3.1 adds no source, no runner, no wrapper, no campaign directory, and no
result file. Its remaining footprint is confined to the pinned dependency and
provenance values in `VERSIONS.env`, the PyTorch install in `Dockerfile`, the
`gemm-cutedsl-p31-check` / `gemm-cutedsl-p31-smoke` targets plus their
validation in `Makefile`, and the truthful status text in `PLAN.md` and
`README.md`.
