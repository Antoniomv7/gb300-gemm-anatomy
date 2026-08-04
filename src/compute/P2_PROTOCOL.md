# P2 frozen protocol -- BF16 UMMA throughput, P2.1 (1-SM UMMA)

This document freezes the P2.1 contract: an instruction-level microbenchmark
of `tcgen05.mma.cta_group::1.kind::f16` (BF16 x BF16 -> FP32) on a single CTA
of a single SM. It also records, for provenance, the full future matrix of
Phase 2 (P2.1-P2.4) and states explicitly which parts remain unimplemented.

## 1. Phase 2 scientific question

AGENTS.md experiment 2: estimate the fifth-generation Tensor Core throughput
ceiling and 2-SM scaling for BF16 x BF16 -> FP32 matrix-multiply-accumulate
on NVIDIA GB300 (`sm_103a`), using the `tcgen05` ("UMMA") instruction family,
with `tcgen05.mma`/`UTCMMA` usage verified in SASS -- not inferred, not
estimated from documentation.

## 2. P2.1 vs P2.2 vs P2.3 vs P2.4

| Unit | Scope | Status in this document |
|------|-------|--------------------------|
| P2.1 | 1-SM UMMA: single CTA, `cta_group::1`, M=128, N in {64,128,256}, depth in {4,16,64,256}, 12 configurations. | **Implemented, independently audited, and functionally verified on GB300.** |
| P2.2 | 2-SM UMMA: CTA pair, `cta_group::2`, M=256, cluster of 2 CTAs. | **Implemented, independently audited, and functionally verified on GB300** at commit `637b6a7e2efbe77b1c9c5d3dfc7ece527f522bba`. This P2.1 document's own closed scope and evidence (sections 4-20 below) remain unchanged and describe only the `cta_group::1` arm. See `src/compute/P2_2_PROTOCOL.md` for the complete, independently frozen P2.2 contract and closure evidence. |
| P2.3 | Joint 1-SM/2-SM sweep infrastructure, at most 24 configurations (AGENTS.md ceiling). | **Implemented, independently audited, and functionally verified on GB300** at commit `7a7cc2ab83197376720f030ba2e990092c3ada40` on 3 August 2026. See `src/compute/P2_3_PROTOCOL.md`. Reuses the P2.1/P2.2 binaries above unmodified; introduces no new CUDA kernel. |
| P2.4 | Profiling and empirical ceiling: Nsight Compute, TFLOP/s and saturation analysis. | **Implemented; independently audited: NO; verified on GB300: NO.** See `src/compute/P2_4_PROTOCOL.md`. `elapsed_cycles` in the P2.1 CSV remains a raw `%clock64` delta, never converted to seconds or FLOP/s here -- that conversion happens exclusively in P2.4's own analyzer, against P2.4's own profiled evidence. |

## 3. Complete future Phase 2 matrix (24 configurations, for provenance only)

AGENTS.md caps experiment 2 at 24 configurations: 1-SM (M=128) and 2-SM
(M=256), each with N in {64,128,256} and depth in {4,16,64,256} -- 2 x 3 x 4
= 24. P2.1 implements exactly the 1-SM half (12 of the 24); the 2-SM half is
P2.2's scope, implemented in `src/compute/umma_2sm.cu`, independently
audited, and functionally verified on GB300 (see
`src/compute/P2_2_PROTOCOL.md`).

| cta_group | M | N | depth | Unit |
|-----------|---|---|-------|------|
| ::1 | 128 | {64,128,256} | {4,16,64,256} | P2.1 (this document, 12 configs) |
| ::2 | 256 | {64,128,256} | {4,16,64,256} | P2.2 (implemented, 12 configs; see P2_2_PROTOCOL.md) |

## 4. The exact twelve P2.1 configurations

```
umma_1sm_m128n64k16_d4      umma_1sm_m128n64k16_d16      umma_1sm_m128n64k16_d64      umma_1sm_m128n64k16_d256
umma_1sm_m128n128k16_d4     umma_1sm_m128n128k16_d16     umma_1sm_m128n128k16_d64     umma_1sm_m128n128k16_d256
umma_1sm_m128n256k16_d4     umma_1sm_m128n256k16_d16     umma_1sm_m128n256k16_d64     umma_1sm_m128n256k16_d256
```

Every kernel: `extern "C" __global__ __launch_bounds__(128)`, exactly 128
threads, exactly 1 CTA (`grid_blocks=1`), compiled for `sm_103a`.

## 5. Definition of 1-SM and `cta_group::1`

"1-SM" means: a single CTA (`grid_blocks=1`), 128 threads, `cta_group::1`
throughout, and a `tcgen05.mma` operation that reads/writes the Tensor
Memory (TMEM) of the one SM the CTA is scheduled on. It does **not** mean
"one CTA per SM on the device" -- the grid is exactly one block regardless
of how many SMs the target GPU has, and no per-sample work is multiplied by
the SM count. The CTA is never pinned to a physical SM id; the scheduler may
place it anywhere.

Per PTX ISA 9.3 section 9.7.17.5 (Issue Granularity, Table 51): `.mma`,
`.commit` with `cta_group::1` are issued by a single thread; `.alloc`,
`.dealloc`, `.relinquish_alloc_permit` with `cta_group::1` are issued
collectively by a single warp. `cta_group::2` (CTA pairs, clusters,
multicast) is out of scope for P2.1 entirely (see section 17).

## 6. Definition of `depth`

`depth` is the number of consecutive `tcgen05.mma` instructions issued
during one outer-loop iteration before one `tcgen05.commit`:

```
UMMA 0:           enable-input-d = false   (D = A*B)
UMMA 1..depth-1:  enable-input-d = true    (D = A*B + D)
one tcgen05.commit
one mbarrier completion wait (mbarrier.try_wait.parity)
```

The `depth`-many instructions are specialized per `(N, depth)` pair and
completely unrolled at compile time (`#pragma unroll` over a template
non-type parameter, verified in SASS -- section 15). `iterations` (a
runtime CLI value) is a plain `for` loop around this fully-unrolled burst;
every outer iteration restarts with `enable-input-d = false`, so the final
TMEM value of D never accumulates across outer iterations and is
independent of `iterations` (see section 11).

## 7. Types, shapes, descriptors

Frozen contract: `dtype = FP32` (accumulator D), `atype = BF16` (A),
`btype = BF16` (B), dense (no sparsity, no block scaling, no integer
saturation), `M = 128`, `N` in `{64,128,256}`, `K = 16` (implied by
`.kind::f16` dense BF16, not separately encodable -- see below), no
transpose, no negate.

**K = 16 is a documented fact, not a choice.** PTX ISA 9.3 section 9.7.17.2.1
(Table 44, "Various combinations of .kind and shapes"), row
`.kind::f16 / no .ws / cta_group 1 / Dense`, lists shapes `64xNxK` /
`128xNxK` with `N = {8,16,...,256}` steps of 8 and `K = 16` for
`atype/btype in {.f16, .bf16}`. The same section states: "K can be specified
explicitly if there are multiple values of K supported for a given MMA
variant. Otherwise, if K can be uniquely determined ..., then K cannot be
explicitly specified." Only K=16 is listed for this exact combination, so K
is not an operand or a descriptor field anywhere in this implementation.

### 7.1 Instruction descriptor (32-bit, PTX ISA 9.3 Table 47, `.kind::f16` column)

| Bits | Field | Value used |
|------|-------|------------|
| 0-1 | Sparsity selector | 0 |
| 2 | Sparsity | 0 (dense) |
| 3 | Saturate (n/a for `.kind::f16`) | 0 |
| 4-5 | dtype (D) | 1 (F32) |
| 6 | reserved | 0 |
| 7-9 | atype (A) | 1 (BF16) |
| 10-12 | btype (B) | 1 (BF16) |
| 13 | Negate A | 0 |
| 14 | Negate B | 0 |
| 15 | Transpose A | 0 |
| 16 | Transpose B | 0 |
| 17-22 | N (3 LSBs not included) | `N >> 3` |
| 23 | reserved | 0 |
| 24-28 | M (4 LSBs not included) | `128 >> 4 = 8` |
| 29 | reserved | 0 |
| 30-31 | `.ws` B-reuse max shift (n/a, no `.ws`) | 0 |

Built by `make_instruction_descriptor<N>()` and independently re-derived
field-by-field by `validate_instruction_descriptor<N>()`
(`src/compute/umma_1sm.cu`); `static_assert`ed for all three N values.

### 7.2 Shared memory descriptor (64-bit, PTX ISA 9.3 Table 45)

| Bits | Field | Value used |
|------|-------|------------|
| 0-13 | `matrix-descriptor-encode(start address)` = `(addr & 0x3FFFF) >> 4` | per-operand SMEM address |
| 14-15 | reserved | 0 |
| 16-29 | `matrix-descriptor-encode(LBO)` | `128 >> 4 = 8` (LBO = 128 bytes) |
| 30-31 | reserved | 0 |
| 32-45 | `matrix-descriptor-encode(SBO)` | `256 >> 4 = 16` (SBO = 256 bytes) |
| 46-48 | fixed constant | `0b001` |
| 49-51 | matrix base offset | 0 (no swizzle: PTX ISA 9.3 9.7.17.4.1 defines base offset as 0 "when the repeating pattern of the specified swizzling mode starts"; the no-swizzle case has no such pattern) |
| 52 | leading-dimension stride mode | 0 (relative byte offset) |
| 53-60 | fixed constant | 0 |
| 61-63 | swizzle mode | 0 (no swizzling) |

Built by `make_smem_descriptor()`. Section 9.7.17.4.1's closing note requires
the matrix start address, LBO, and SBO to be 16-byte aligned; the dynamic
SMEM buffer is `__align__(128)` and A/B occupy `M*K*2` / `N*K*2` bytes
respectively (both multiples of 16), so every operand address, and the
constant LBO/SBO below, satisfy this trivially.

## 8. The single A/B layout

**Major-ness.** PTX ISA 9.3 Table 58 ("Major-ness for different matrices")
and Table 59 ("Valid Combinations of Type-Size, Major-ness and Swizzling")
establish that for 16-bit operands, non-transposed (`Row A` / `Column B`,
matching Transpose A = Transpose B = 0 above) is valid for "all swizzling
modes", including none. This is the layout used here: **K-major** for both A
and B (A row-major with K contiguous; B "column"-major with K contiguous per
N-column), no swizzling (swizzle mode 0, section 7.2).

**Byte-exact packing (derived, not copied -- see section 19 audit note).**
PTX ISA 9.3 section 9.7.17.3.3 gives the canonical K-major, no-swizzle
layout as a nested CuTe-style shape:stride expression,
`((8,m),(T,2k)):((1*T,SBO),(1,LBO))`, where `T = 128 / bits-per-element`
(the "128-bit normalization" factor) and `m`, `k` are repeat counts. For
BF16, `T = 128/16 = 8`. Because K=16 exactly equals `2*T` (=16), the whole K
extent fits in **one** elementary K-major "core tile": 8 rows (or columns)
by 16 K-elements (two T=8 chunks), so `k=1` always, and only `m` (= number
of 8-row/column groups) varies with M or N.

Choosing the tightest possible (fully contiguous, gapless) placement of
these 8x16 core tiles fixes:

```
LBO = 64 elements  = 128 bytes   (stride between a tile's two 8-element K-chunks)
SBO = 128 elements = 256 bytes   (stride between successive 8-row/column groups)
```

which matches `A_bytes = M*K*2 = 4096` and `B_bytes = N*K*2` exactly, with
zero padding, for every N in {64,128,256}. The mapping from a logical
`(row_or_col, k)` index to its physical flat BF16-element offset is
`smem_core_tile_index(group_idx, pos_in_group, k)` in `umma_1sm.cu`:

```
chunk = k / 8;  t = k % 8
offset = group_idx * 128 + chunk * 64 + pos_in_group * 8 + t
```

used identically for A (`group_idx = row/8`, `pos_in_group = row%8`) and B
(`group_idx = col/8`, `pos_in_group = col%8`). LBO and SBO are therefore
**constants shared by every specialization** (they depend only on K=16 and
BF16's T=8, never on M or N); only the descriptor's base-address field
differs between the A and B descriptors, and between specializations.

## 9. TMEM lifecycle

Before any of the numbered steps below, every kernel first evaluates an
explicit launch-contract guard (`launch_contract_is_valid()` in
`umma_1sm.cu`): a launch that is not exactly `grid=(1,1,1)`,
`block=(128,1,1)` writes `0` to a host-visible `g_launch_ok` output and
returns immediately, before touching `__syncthreads()`, mbarrier
initialization, TMEM allocation, or any UMMA instruction. A launch that
passes writes `1` to the same location before continuing to step 1, so the
host can distinguish "the kernel ran and confirmed its own launch shape"
from every other outcome -- a silent early return alone would not be
observable by the host. `run_once()` always allocates and zero-initializes
`g_launch_ok` before the launch and treats any value other than `1` as
fatal, checked before any correctness validation of `D`. The frozen host
launcher (`kGridBlocks=1`, `kThreadsPerCta=128`) always satisfies this
guard; it exists so a regression to a different launch shape fails loudly
instead of allocating TMEM against an unexpected thread/warp count or
silently computing a wrong `D`. `__launch_bounds__(128)` alone is not
sufficient here since it only constrains the maximum thread count, not the
actual launch configuration.

Per kernel, in order (PTX ISA 9.3 sections 9.7.17.1.2, 9.7.17.7.1):

1. All 128 threads fill A and B directly into the physical layout above
   (section 11's validation pattern), then `__syncthreads()`.
2. Thread 0: `mbarrier.init` (`cuda::ptx::mbarrier_init`, expected count 1),
   then `fence.proxy.async` (`cuda::ptx::fence_proxy_async(space_shared)`),
   publishing both the mbarrier initialization and every thread's A/B writes
   (gathered by step 1's barrier) to the async proxy that `tcgen05.mma` reads
   through (PTX ISA 9.3 section 9.7.17.6.5). Then `__syncthreads()` again,
   publishing the fence's effect to whichever thread `elect_sync` selects as
   leader next.
3. Warp 0 (all 32 threads, collectively): `tcgen05.alloc.cta_group::1`,
   exactly N columns (64, 128, or 256; section 10.4's exact-N-columns
   requirement). Then `__syncthreads()`.
4. Leader thread only: the timed region (section 10) -- `depth`-unrolled
   `tcgen05.mma` bursts, `tcgen05.commit`, `mbarrier.try_wait.parity`, for
   `iterations` outer-loop repeats.
5. `__syncthreads()`, then every thread: `tcgen05.fence::after_thread_sync`
   (required before the new asynchronous `tcgen05.ld` -- PTX ISA 9.3 section
   9.7.17.6.4.2's Example 2, and composes with the preceding barrier per
   section 9.7.17.6.3, matching the canonical "different-thread" pattern of
   section 9.7.17.6.4.4), then, for each 32-column fragment `frag` (N/32
   fragments; 8 for N=256): `tcgen05.ld.sync.aligned.32x32b.x32.b32` at the
   collective address `make_tmem_load_address(tmem_d, warp_id, frag)` +
   `tcgen05.wait::ld.sync.aligned`, storing D to global memory. See section
   9.1 below for the corrected address construction (repair, 2026-07-30).
6. `__syncthreads()`.
7. Warp 0 (the same warp that allocated): `tcgen05.dealloc.cta_group::1`,
   then `tcgen05.relinquish_alloc_permit.cta_group::1`.
8. Thread 0: `mbarrier.inval.shared.b64`, ending the mbarrier's lifetime.

There are exactly two exit paths: the launch-contract rejection above
(before step 1, so TMEM is never allocated and no barrier is ever reached on
this path) and the function's natural end after step 8. Every accepted
launch reaches all of steps 1-8 unconditionally, so no accepted-launch path
leaves TMEM allocated. All `tcgen05` instructions use `cta_group::1`
throughout.

### 9.1 TMEM load address construction (repair, 2026-07-30)

PTX ISA 9.3 section 9.7.17.1.1 ("Tensor Memory Addressing") specifies that a
Tensor Memory address is 32 bits wide with two components: bits 31-16 are
the lane index, bits 15-0 are the column index. Section 9.7.17.8.1 ("Access
restrictions") further specifies that the Tensor Memory of a CTA is split
into four equal 32-lane chunks, one per warp of the warpgroup, and that
every thread of the issuing warp must supply the identical collective
`taddr`:

| Warp ID within the warpgroup | Accessible lanes |
|-------------------------------|-------------------|
| 0 | 0-31 |
| 1 | 32-63 |
| 2 | 64-95 |
| 3 | 96-127 |

`tcgen05.alloc` returns a base address at lane 0 of the allocated columns (a
column allocation spans all 128 lanes, PTX ISA 9.3 section 9.7.17.1.2), so
every warp other than warp 0 must add its own lane contribution before
issuing `tcgen05.ld`, or it reads warp 0's lanes instead of its own. The
originally audited defect used `tmem_d + frag * 32` for all four warps -- a
plain column offset with no lane contribution at all, so warps 1-3 all read
the wrong 32-row band of `D` (warp 0's rows 0-31, repeated four times, in
place of their own rows 32-63/64-95/96-127).

The fix, `make_tmem_load_address()` in `umma_1sm.cu`:

```cpp
constexpr uint32_t kTmemLaneShift = 16;        // bits 31-16
constexpr uint32_t kTmemRowsPerWarp = 32;      // one warpgroup chunk
constexpr uint32_t kTmemColsPerFragment = 32;  // this kernel's fragment width

uint32_t make_tmem_load_address(uint32_t tmem_base, int warp_id, int frag) {
    const uint32_t lane_contribution = (static_cast<uint32_t>(warp_id) * kTmemRowsPerWarp) << kTmemLaneShift;
    const uint32_t column_contribution = static_cast<uint32_t>(frag) * kTmemColsPerFragment;
    return tmem_base + lane_contribution + column_contribution;
}
```

`tmem_base`'s lane bits are always 0 (lane 0 of the allocation), and
`lane_contribution`/`column_contribution` occupy disjoint bit ranges (31-16
and 15-0 respectively -- `frag * 32` never exceeds `7 * 32 = 224`, the
largest fragment index for N=256, so it never carries into bit 16), so plain
addition is equivalent to a bitwise OR of the three components. The
resulting mapping is exactly the access-restriction table above: warp 0 ->
D rows 0-31, warp 1 -> rows 32-63, warp 2 -> rows 64-95, warp 3 -> rows
96-127, matching the row index (`warp_id * 32 + lane`) already used for the
global-store side of the same readback loop. `scripts/check_umma_1sm_sass.py`'s
source gate (section 15.1) proves this helper is real executable code, that
the warp/lane/column contributions use the named constants with their
correct values, that the call site feeding `tcgen05_ld_32x32b_x32` is built
by this helper, and that the original defective `tmem_d + frag * 32`
operand is absent.

## 10. Synchronization and completion (timed region)

`%clock64` is read only by the elected leader thread (`cuda::ptx::elect_sync`
from warp 0), immediately before the outer `iterations` loop and immediately
after the final `mbarrier.try_wait.parity` of the last iteration confirms
completion. Both `mov.u64 %0, %%clock64;` reads use an inline-asm `"memory"`
clobber so the compiler cannot reorder surrounding code across them. Reading
D from TMEM, global-memory stores, TMEM allocation/deallocation, A/B
initialization, descriptor construction, and mbarrier init/invalidate are
all outside this region (see section 12 for the exact included/excluded
list, reproduced from the task brief).

### 10.1 Timing mode: validation-only, warm-up, and timed execution (repair, 2026-07-30)

An explicit `TimingMode` (`kUntimed` / `kTimed`) kernel argument, propagated
through `run_once()` and `umma_1sm_body()`, decides at each launch whether
either `%clock64` read may execute: each read is guarded by
`if (timing_mode == TimingMode::kTimed)`. `TimingMode` is a launch argument
(uniform across the whole grid, not a per-thread value), so this guard is
never warp-divergent. Three call sites always launch with
`TimingMode::kUntimed` and therefore never execute a clock64 read:
`--self-test` (`run_self_test()`), the pre-timing correctness validation,
and every warm-up iteration (the latter two both routed through
`run_untimed_or_die()` in `main()`). Only the per-repetition timed loop
launches with `TimingMode::kTimed` (`run_timed_or_die()`). Correctness is
still required on every launch regardless of mode; only the cycle-count
expectation differs: `run_untimed_or_die()` requires validation to pass but
makes no claim about `elapsed_cycles` (legitimately `0` when untimed), while
`run_timed_or_die()` requires both validation to pass and
`elapsed_cycles > 0`. This closes the originally audited defect, in which
every launch -- including `--self-test`, pre-timing validation, and warm-up
-- unconditionally executed both clock64 reads despite being described as
untimed.

## 11. Validation pattern and CPU reference

Per element, a small integer pattern exactly representable in BF16:

```
A(row,k) = ((row + 3*k) % 7) - 3      row in [0,128), k in [0,16)
B(k,col) = ((2*k + col) % 5) - 2      k in [0,16),  col in [0,N)

reference(row,col) = depth * sum_{k=0..15} A(row,k) * B(k,col)
```

`iterations` never appears in `reference(...)`: every outer iteration
restarts with `enable-input-d=false` (D = A*B, not D += A*B), so the TMEM
value of D at kernel exit is exactly the *last* iteration's own
`depth`-deep burst result, identical to every other iteration's result.
Products and the depth-scaled sum stay far inside FP32's exact-integer
range (`|A|<=3`, `|B|<=2`, `|sum_16 terms|<=16*6=96`, `|reference|<=256*96=
24576`, versus FP32's exact range of +-2^24), so the contract is
`GPU result == CPU reference` bit-for-bit, `max_abs_error == 0`, checked for
all `128*N` elements on the host after one `cudaMemcpy` of D back (not a
sample, not a checksum). A mismatch reports the first failing flat index,
expected value, obtained value, and total mismatch count; prevents any
subsequent timing or CSV output for that repetition; and yields a nonzero
process exit code once no repetition has produced a publishable row.

## 12. Timed region boundaries (reproduced from the frozen task contract)

Included: issuing `depth` UMMA instructions per iteration, one
`tcgen05.commit` per iteration, the completion wait for every iteration.
Excluded: kernel launch, A/B initialization, descriptor construction,
mbarrier initialization, TMEM allocation, warm-up, reading D from TMEM,
global-memory stores, device-to-host copy, CPU validation, TMEM
deallocation. Cycles are never converted to seconds or TFLOP/s here (P2.4
work).

## 13. FLOP/UMMA accounting

```
flops_per_umma = 2 * M * N * K
total_umma     = depth * iterations
total_flops    = 2 * M * N * K * depth * iterations
```

with `M=128`, `K=16`, `grid_blocks=1`, `operations=1`; never multiplied by
threads/warps/SM count/D-element count/an extra factor of 2/`repetitions`.
Computed with `checked_mul_i64` (`__int128`-checked 64-bit multiplication,
host-only) in `umma_1sm.cu`.

## 14. CSV schema

`smoke`/`benchmark` stdout carries exactly one header line plus one row per
repetition; all diagnostics go to stderr.

```
schema_version,timestamp_utc,run_kind,publishable,method,sample_index,cta_group,m,n,k,depth,iterations,warmup_iterations,repetitions,umma_per_iteration,total_umma,flops_per_umma,total_flops,elapsed_cycles,cycles_per_umma,flops_per_cycle,threads_per_cta,grid_blocks,tmem_columns,operand_path,input_type,accumulator_type,correctness,mismatches,max_abs_error,gpu_name,gpu_uuid,compute_capability,cuda_driver_version,cuda_runtime_version,git_commit,git_dirty
```

Frozen values: `schema_version=1`, `publishable=false`, `method=umma_1sm`,
`cta_group=1`, `m=128`, `k=16`, `umma_per_iteration=depth`,
`threads_per_cta=128`, `grid_blocks=1`, `tmem_columns=n`,
`operand_path=smem_smem`, `input_type=bf16`, `accumulator_type=fp32`,
`correctness=OK`, `mismatches=0`, `max_abs_error=0`,
`compute_capability=10.3`. `elapsed_cycles>0`; `cycles_per_umma` and
`flops_per_cycle` are `double`s computed from the exact integer counters.
These are technical evidence from a functional pilot, not publishable
results (`publishable=false` on every row, unconditionally).

## 15. SASS contract

Enforced by `scripts/check_umma_1sm_sass.py` against real
`cuobjdump -sass` output of `build/compute/umma_1sm` compiled for
`sm_103a` with CUDA 13.1.80 `ptxas` (mnemonics observed directly, never
guessed -- see that script's module docstring for the full PTX-to-SASS
mapping table and the two instructions, `tcgen05.wait::ld` and
`tcgen05.fence::after_thread_sync`, that ptxas emits with no distinct SASS
footprint on this toolchain and are instead proven present via a mandatory
source check of the compiled `.cu` file -- see 15.1 below). Summary of what
is proved for every one of the twelve symbols:

1. Exactly the twelve expected symbols exist, no missing/extra/duplicate.
2. Every symbol's SASS contains `UTCHMMA` (sm_103a's lowering of
   `tcgen05.mma.cta_group::1.kind::f16`; the task brief's shorthand
   "UTCMMA" refers to this observed mnemonic).
3. The static `UTCHMMA` count equals `depth` exactly, and consecutive
   occurrences sit at a single uniform address spacing -- evidence of full
   compile-time unrolling, not a runtime back-edge standing in for it.
4. The burst ends with a real completion sequence: `UTCBAR` (commit) after
   the last `UTCHMMA`, then at least one `SYNCS.PHASECHK.TRANS64.TRYWAIT`
   (mbarrier wait) after the commit.
5. TMEM allocation (`UTCATOMSWS.FIND_AND_SET.ALIGN`) and deallocation
   (`UVIRTCOUNT.DEALLOC.SMPOOL`) are present, with deallocation ordered
   after the last `UTCHMMA`/`LDTM.x32` use.
6. `LDTM.x32` (TMEM-to-register load) appears exactly `N/32` times.
7. No `HMMA`/`WGMMA`/`QGMMA`/`IMMA`/`BMMA` (non-tcgen05 MMA), `UTMALDG`
   (TMA), `LDGSTS`, `UBLKCP`, or sparse (`.sp`) `UTCHMMA` qualifier appears
   anywhere in the binary; no cluster-scoped barrier or `CLUSTER` header
   attribute appears (2-SM evidence).
8. (mandatory source check, section 15.1) the compiled `.cu` text contains
   `tcgen05.wait::ld.sync.aligned` and `tcgen05.fence::after_thread_sync` as
   executable code (not merely inside a comment), and contains none of
   `cta_group::2`, `__cluster_dims__`, a real (non-comment) `multicast`
   qualifier, a non-`.kind::f16` MMA kind, a `.sp` sparse form, or
   `block_scale`, likewise as executable code.
9. (source check) the corrected TMEM load address is real executable code:
   `make_tmem_load_address()` exists, its warp contribution uses
   `warp_id * kTmemRowsPerWarp` shifted by `kTmemLaneShift` (=16, the lane
   bits), its fragment contribution uses `frag * kTmemColsPerFragment`
   (=32, the column bits), the `tcgen05_ld_32x32b_x32` call site is built by
   this helper, and the original defective `tmem_d + frag * 32` operand is
   absent (section 9.1).
10. (source check) an explicit launch-contract guard
    (`launch_contract_is_valid()`) is checked inside `umma_1sm_body` before
    its first `__syncthreads()`.
11. (source check) every timed `%clock64` read is guarded by
    `timing_mode == TimingMode::kTimed`, and at least one call site uses
    `TimingMode::kUntimed` (section 10.1).

### 15.1 Mandatory, fail-closed source validation (repair, 2026-07-30)

The two-positional-argument invocation
(`check_umma_1sm_sass.py <binary> <sass-path>`) always validates the
canonical source `src/compute/umma_1sm.cu`, resolved relative to the
checker script itself (never the caller's current working directory) by
`resolve_default_source_path()`. `--source <path>` may override which file
is checked (used for ad hoc testing); omitting it never skips the check. If
the resolved source cannot be opened, this checker exits 1 -- there is no
code path in which the real binary/SASS check can report success while
source validation was skipped or merely reported as a documented
limitation.

The source scanner (`strip_comments_preserving_literals()`) strips both
`//` line comments and `/* ... */` block comments while preserving the
exact text of every string and character literal (required inline-PTX text
lives inside C++ string literals passed to inline asm, so it must never be
mistaken for a comment). Every forbidden- and required-pattern check in
this checker runs against this comment-stripped, literal-preserving view,
never against the raw source text -- closing the originally audited defect
in which a file containing only comments such as

```cpp
// tcgen05.wait::ld.sync.aligned
// tcgen05.fence::after_thread_sync
```

passed the required-source checks (the comparison used the raw text, not a
comment-stripped view). The scanner fails closed (raises, and the checker
reports a "cannot safely scan" error) on an unterminated `/*` block comment
or an unterminated string/character literal, since the lexical state cannot
then be safely determined.

`scripts/check_umma_1sm_sass.py --self-test` exercises all of the above (42
cases total): the original eighteen SASS-contract cases (missing symbol,
extra symbol, duplicate configuration, missing `UTCHMMA`, incorrect depth
in both directions, a non-uniformly spaced burst, missing commit, missing
wait, missing alloc/dealloc, deallocation before final use, incorrect
LDTM.x32 count, and every forbidden instruction/2-SM marker) plus
source-level positive and negative cases for: required PTX text present
only in a `//` or `/* */` comment (reject), forbidden text present only in
a `/* */` comment (accept), a missing warp-derived TMEM lane offset, an
incorrect TMEM lane shift constant, a missing TMEM fragment column offset,
a live helper return that omits either the lane or column contribution, the
original defective `tmem_d + frag * 32` operand, a missing or inverted
launch-contract guard, unconditional clock reads hidden beside empty timed
guards, a self-test routed through `TimingMode::kTimed`, a missing
`TimingMode::kUntimed` call site, an unterminated block comment, an
unterminated string literal, and the mandatory default-source-path
resolution and fail-closed-on-missing-file behavior itself.

## 16. Commands

GPU-free (no Docker GPU, no network, used to produce and validate this
implementation):

```bash
python3 -m py_compile scripts/check_umma_1sm_sass.py
python3 scripts/check_umma_1sm_sass.py --self-test
make check-static
make compute-umma-1sm-build
make compute-umma-1sm-sass
make compute-umma-1sm-check
```

GB300 functional-verification commands (executed successfully on 30 July
2026 with an explicitly selected free physical device):

```bash
BLACKWELL_GPU_INDEX=<physical-index> make preflight
BLACKWELL_GPU_INDEX=<physical-index> make compute-umma-1sm-self-test
BLACKWELL_GPU_INDEX=<physical-index> make compute-umma-1sm-smoke

BLACKWELL_GPU_INDEX=<physical-index> scripts/run_container.sh \
  build/compute/umma_1sm \
  --run-kind benchmark --n 128 --depth 16 \
  --iterations 20 --warmup-iterations 5 --repetitions 3
```

## 17. Verification and scientific limitations

* GB300 functional verification completed on 30 July 2026 at Git commit
  `1004666db7a2eef1ec499c60740cafc1e2f41328`. The K-major SMEM byte layout
  (section 8), derived from the PTX ISA's documented formula rather than
  copied from a reference, passed the full numerical device self-test for
  all twelve `(N, depth)` specializations (`SELF_TEST: PASS (12/12)`, zero
  mismatches). Short `smoke` and `benchmark` routing checks also passed.
  This proves functional correctness for the frozen P2.1 matrix; it does
  not establish a throughput ceiling.
* `tcgen05.wait::ld` and `tcgen05.fence::after_thread_sync` have no
  distinct SASS footprint on the observed toolchain (section 15); their
  presence is proved only via a static source check, not SASS evidence.
* No 2-SM UTCHMMA sample was available to compile for direct SASS
  comparison; "no 2-SM instruction present" is evidence from the absence of
  cluster-related SASS/header markers plus a static source check, not a
  positive identification of a distinguishing 2-SM SASS qualifier.
* `elapsed_cycles` is a raw `%clock64` delta on one thread of one SM; it is
  not wall-clock time, not corrected for clock throttling/boost state, and
  not a throughput or saturation claim. No TFLOP/s conversion exists in
  this unit (P2.4 work).
* P2.2 (2-SM) is implemented, independently audited, and functionally
  verified on GB300 at commit `637b6a7e2efbe77b1c9c5d3dfc7ece527f522bba`
  (see `src/compute/P2_2_PROTOCOL.md`). P2.3 (joint sweep infrastructure) is
  implemented, independently audited, and functionally verified on GB300 at
  commit `7a7cc2ab83197376720f030ba2e990092c3ada40` (see
  `src/compute/P2_3_PROTOCOL.md`); P2.4 (profiling/ceiling) is implemented
  but not yet independently audited or verified on GB300 (see
  `src/compute/P2_4_PROTOCOL.md`). Nothing in this document or in
  `PLAN.md`'s P2.1 row changes P2.1's own closed scope.

## 18. Status

A first independent review of this implementation found seven defects: the
TMEM readback used the wrong lane address for warps 1-3 (section 9.1); the
SASS checker could pass without mandatory source validation and required
PTX-text checks could be satisfied by a comment (section 15.1);
`--self-test`, pre-timing validation, and warm-up executed `%clock64` reads
despite being described as untimed (section 10.1); kernels did not reject
an invalid grid/CTA size (section 9); the CLI accepted an unauthorized `-h`
alias and arbitrary upper limits absent from the frozen contract; and the
non-literal architecture-flag workaround lacked recorded evidence (section
20). All seven were repaired on 2026-07-30. A subsequent review found that
five of those properties could still be disabled while leaving superficial
source patterns behind: either TMEM return contribution could be omitted,
the launch guard could be inverted, the self-test could be routed through
the timed path, or the clock reads could be moved outside empty timed
guards. Commit `1004666db7a2eef1ec499c60740cafc1e2f41328` made those checks
structural and fail-closed and expanded the synthetic suite to 42 cases.
That commit then passed an independent audit and the real-GB300
compilation/SASS, twelve-case device self-test, `smoke`, and `benchmark`
routing checks described in sections 16-17:

* P2.1: **implemented and closed**.
* Independent audit: **passed**.
* GB300 verification: **passed**.
* Publishable result: **none**. Every CSV row P2.1 can ever emit carries
  `publishable=false` unconditionally.
* P2.2: **implemented and closed** (`src/compute/umma_2sm.cu`; independent
  audit and GB300 functional verification passed at commit `637b6a7` -- see
  `src/compute/P2_2_PROTOCOL.md`).
* P2.3: **implemented, independently audited, and functionally verified on
  GB300** at commit `7a7cc2ab83197376720f030ba2e990092c3ada40` (see
  `src/compute/P2_3_PROTOCOL.md`).
* P2.4: **implemented; independently audited: NO; verified on GB300: NO**
  (see `src/compute/P2_4_PROTOCOL.md`). No P2.4 campaign has been executed;
  no empirical ceiling has been measured.

## 19. References

Primary (normative): NVIDIA PTX ISA 9.3,
<https://docs.nvidia.com/cuda/parallel-thread-execution/>, chapter
"9.7.17. TensorCore 5th Generation Family Instructions" (sections
9.7.17.1-9.7.17.12 cited by number throughout this document). Read from the
official PDF (`https://docs.nvidia.com/cuda/pdf/ptx_isa_9.3.pdf`) rather
than the paginated HTML, so every table and worked example cited above was
read in full, not summarized.

Secondary (conceptual, adapted and audited, not copied): pinned commit
`9a068d853d5c3676939eb46fe21ff6d6a2a4133b` of
`SemiAnalysisAI/microbench-blackwell/umma_throughput/umma_tput.cu`. Audit
findings, and how this implementation diverges:

* That file never validates numerical correctness (it fills A/B with an
  index-independent nonzero pattern purely to avoid degenerate all-zero
  operands, and never reads D back at all) -- consistent with it being a
  pure throughput probe, but it means its shared-memory descriptor LBO/SBO
  values could not be trusted as a K-major-correct reference. This
  implementation's LBO/SBO (section 8) are derived independently from the
  PTX ISA's own canonical-layout formula and validated end-to-end by the
  CPU-reference numerical check (section 11), which that file has no
  equivalent of.
* Its bit-level instruction-descriptor and shared-memory-descriptor field
  placement (which fields occupy which bits, the `0b001`/base-offset-0/
  no-swizzle constants) matches this document's independent reading of PTX
  ISA Tables 45 and 47 exactly, and was used as a cross-check.
* It omits `fence.proxy.async` between writing A/B (generic proxy) and
  `tcgen05.mma` reading them (async proxy), despite PTX ISA 9.3 section
  9.7.17.6.5 stating that a cross-proxy fence is required. This
  implementation includes it (section 9, step 2), reusing the same
  `cuda::ptx::fence_proxy_async` idiom already audited and GB300-verified
  for TMA in `src/memory/tma.cu` (P1.2).
* Its `tcgen05.commit`/wait pattern uses `.multicast::cluster` (a
  `cta_group::2`/cluster-only feature); this implementation uses the
  plain, non-cluster `tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64`
  form instead, matching the official PTX ISA worked example in section
  9.7.17.10.9.1 and appropriate for a genuinely single-CTA kernel.
* Its `enable-input-d` predicate construction (`setp.ne.b32` inside inline
  asm on a plain integer argument) and its omission of the optional
  `disable-output-lane` operand (also shown valid by an official PTX ISA
  worked example, section 9.7.17.6.4.2) were adopted directly, since both
  match the PTX ISA and introduce no correctness risk.

Also consulted (conceptual only, no bit-level detail found there for
tcgen05 descriptors): NVIDIA CUTLASS Blackwell functionality documentation,
<https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html>.

## 20. Architecture-flag evidence (repair, 2026-07-30)

The frozen contract requested the literal single-flag form
`nvcc -std=c++17 -O3 -lineinfo -arch=$(CUDA_ARCH)`. Before this repair, the
Makefile instead derived `-arch=compute_103a -code=sm_103a` with only a
comment asserting the literal form fails; that assertion was not backed by
recorded command output. Both forms were directly executed in the pinned,
networkless `gb300-gemm-anatomy:phase0` container, with no GPU, to settle
this with evidence rather than memory.

Literal form (`CUDA_ARCH=sm_103a`, single flag):

```bash
docker run --rm --network none --security-opt no-new-privileges --cap-drop ALL \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$(pwd):/workspace" -w /workspace \
  gb300-gemm-anatomy:phase0 \
  nvcc -std=c++17 -O3 -lineinfo -arch=sm_103a -o build/compute/umma_1sm_probe_literal src/compute/umma_1sm.cu
```

Result: `nvcc` exits **255**. No binary is produced. `ptxas` reports 3336
lines of the form `error: Instruction 'tcgen05.alloc' not supported on
.target 'sm_103'` / `error: Feature '.cta_group::1' not supported on
.target 'sm_103'` / `error: Instruction 'tcgen05.mma' not supported on
.target 'sm_103'` / `error: Feature '.kind::f16' not supported on .target
'sm_103'`, one set per `tcgen05` instruction across all twelve
specializations. `nvcc`'s own intermediate PTX file for this invocation is
named `.../tmpxft_...-7_umma_1sm.compute_103.ptx` -- **not**
`compute_103a.ptx` -- confirming the "a" (architecture-specific) suffix is
already lost before `ptxas` ever runs, exactly as the pre-repair comment
claimed but had not recorded.

Split form (`CUDA_ARCH=sm_103a`, explicit virtual/real pair, unchanged from
before this repair):

```bash
docker run --rm --network none --security-opt no-new-privileges --cap-drop ALL \
  --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$(pwd):/workspace" -w /workspace \
  gb300-gemm-anatomy:phase0 \
  nvcc -std=c++17 -O3 -lineinfo -arch=compute_103a -code=sm_103a -o build/compute/umma_1sm_probe_split src/compute/umma_1sm.cu
```

Result: `nvcc` exits **0**. The resulting binary disassembles (`cuobjdump
-sass`) to all twelve expected `umma_1sm_m128n{N}k16_d{DEPTH}` symbols, each
with the expected exact `UTCHMMA` count (`depth`) and `LDTM.x32` count
(`N/32`), a real TMEM allocate/commit/wait/deallocate lifecycle, and no
forbidden or 2-SM instruction -- the same real-cubin evidence table produced
by `make compute-umma-1sm-sass` (section 15).

Decision: the pinned CUDA 13.1.80 toolchain reproducibly requires the
explicit virtual/real pair for this file; the Makefile retains
`COMPUTE_UMMA_1SM_ARCH_FLAGS := -arch=compute_$(patsubst sm_%,%,$(CUDA_ARCH)) -code=$(CUDA_ARCH)`,
derived from the same pinned `CUDA_ARCH` value in `VERSIONS.env` (unchanged),
never a hardcoded literal independent of the pinned contract.
