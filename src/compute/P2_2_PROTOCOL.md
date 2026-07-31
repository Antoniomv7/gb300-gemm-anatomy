# P2.2 frozen protocol -- BF16 UMMA throughput, 2-SM CTA-pair arm

This document freezes the P2.2 contract: an instruction-level microbenchmark
of `tcgen05.mma.cta_group::2.kind::f16` (BF16 x BF16 -> FP32) on a static
two-CTA cluster (one CTA pair), joint M=256. It is independent of P2.1
(`src/compute/umma_1sm.cu` and `scripts/check_umma_1sm_sass.py` are not
modified, included, or refactored into a shared helper); every descriptor,
synchronization step, TMEM address, and completion mechanism below was
independently re-derived from the PTX ISA text and validated by compiling
isolated probes against the pinned CUDA 13.1.80 toolchain (section 15).

## 1. Scientific question

AGENTS.md experiment 2: estimate the fifth-generation Tensor Core throughput
ceiling and 2-SM scaling for BF16 x BF16 -> FP32 matrix-multiply-accumulate
on NVIDIA GB300 (`sm_103a`), using the `tcgen05` ("UMMA") instruction family,
with `tcgen05.mma`/`UTCHMMA` usage verified in SASS -- not inferred, not
estimated from documentation. P2.2 answers the "2-SM scaling" half: does a
single CTA pair, cooperating through the CTA-pair mechanism described in PTX
ISA 9.3 section 9.7.17.5.1, correctly execute a joint M=256 operation split
128 rows per CTA. P2.2 establishes functional correctness only; it makes no
throughput, ceiling, or scaling claim (that is P2.4's work, gated on P2.3's
sweep infrastructure).

## 2. P2.1 vs P2.2 vs P2.3 vs P2.4

| Unit | Scope | Status in this document |
|------|-------|--------------------------|
| P2.1 | 1-SM UMMA: single CTA, `cta_group::1`, M=128, N in {64,128,256}, depth in {4,16,64,256}, 12 configurations. | **Implemented, independently audited, and functionally verified on GB300** (see `src/compute/P2_PROTOCOL.md`). |
| P2.2 | 2-SM UMMA: CTA pair, `cta_group::2`, M=256, cluster of 2 CTAs, 12 configurations. | **Implemented in this document/commit. Not yet independently audited. Not yet verified on GB300.** |
| P2.3 | Joint 1-SM/2-SM sweep infrastructure, at most 24 configurations (AGENTS.md ceiling). | **Not implemented.** No runner, no campaign, no sweep script exists. |
| P2.4 | Profiling and empirical ceiling: Nsight Compute, TFLOP/s and saturation analysis. | **Not implemented.** No profiling script, no TFLOP/s conversion, no saturation claim exists. `elapsed_cycles` in the P2.2 CSV is a raw `%clock64` delta, never converted to seconds or FLOP/s here. |

Together with P2.1, P2.2 completes AGENTS.md's full 24-configuration Phase 2
matrix (2 cta_group values x 3 N values x 4 depth values); no configuration
outside this frozen matrix is implemented in either file.

## 3. The exact twelve P2.2 configurations

```
umma_2sm_m256n64k16_d4      umma_2sm_m256n64k16_d16      umma_2sm_m256n64k16_d64      umma_2sm_m256n64k16_d256
umma_2sm_m256n128k16_d4     umma_2sm_m256n128k16_d16     umma_2sm_m256n128k16_d64     umma_2sm_m256n128k16_d256
umma_2sm_m256n256k16_d4     umma_2sm_m256n256k16_d16     umma_2sm_m256n256k16_d64     umma_2sm_m256n256k16_d256
```

Every kernel: `extern "C" __global__ __cluster_dims__(2, 1, 1)
__launch_bounds__(128)`, exactly 128 threads per CTA, exactly one static
two-CTA cluster (`grid_blocks=2`), compiled for `sm_103a`.

## 4. Definition of 2-SM, CTA pair, and `cta_group::2`

"2-SM" means: one static cluster of exactly two CTAs (`grid_blocks=2`,
`__cluster_dims__(2, 1, 1)`), 128 threads per CTA, `cta_group::2` throughout,
and a single `tcgen05.mma` operation whose joint M=256 result is split
across the two CTAs' own Tensor Memory, 128 rows each. It does **not** mean
choosing two physical SM identifiers: neither CTA is ever pinned to a
physical SM id, and scheduling is never inferred from `%smid`. This mirrors
P2.1's own definition of "1-SM" (`src/compute/P2_PROTOCOL.md` section 5).

PTX ISA 9.3 section 9.7.17.5.1 ("CTA Pair"): "Any 2 CTAs within the cluster
whose `%cluster_ctarank` differs by the last bit only is said to form a CTA
pair. Within a CTA pair, the CTA whose last bit in the `%cluster_ctarank` is
0 is termed the even numbered CTA ...; 1 is termed the odd numbered CTA."
Section 9.7.17.5.2 ("Peer CTA"): "The peer CTA of the odd CTA within the CTA
pair is the even CTA in the same pair" (and vice versa). This document and
the source code use "CTA rank 0" / "CTA rank 1" for the even/odd CTA
respectively, matching `%cluster_ctarank` directly (`cta_rank =
static_cast<int>(cluster_ctarank)`, valid only after the launch-contract
guard confirms `cluster_nctarank == 2`).

Per PTX ISA 9.3 section 9.7.17.5 (Issue Granularity, Table 51), for
`cta_group::2`: `.mma`/`.commit` are issued by "a single thread from the
CTA-Pair"; `.alloc`/`.dealloc`/`.relinquish_alloc_permit` are issued
"collectively" by one warp in **each** of the current CTA and its peer CTA;
`.ld`/`.st`/`.wait::{ld,st}`/`.fence::*` take no `.cta_group` qualifier at
all and always operate on "the current CTA['s]" own Tensor Memory. Every one
of these facts is reflected directly in the synchronization design (section
7) and the source-level checker contract (section 14).

## 5. Definition of `depth`

Identical in form to P2.1's definition (`src/compute/P2_PROTOCOL.md` section
6), independently re-derived here:

```
UMMA 0:           enable-input-d = false   (D = A*B)
UMMA 1..depth-1:  enable-input-d = true    (D = A*B + D)
one tcgen05.commit (cta_group::2, multicast)
one mbarrier completion wait, by both CTAs, on their own local mbarrier
```

The `depth`-many instructions are specialized per `(N, depth)` pair and
completely unrolled at compile time (`#pragma unroll`, verified in SASS --
section 14). `iterations` (a runtime CLI value) is a plain `for` loop around
this fully-unrolled burst; every outer iteration restarts with
`enable-input-d = false`, so the final TMEM value of D never accumulates
across outer iterations and is independent of `iterations`.

## 6. Types, shapes, descriptors

Frozen contract: `dtype = FP32` (accumulator D), `atype = BF16` (A),
`btype = BF16` (B), dense (no sparsity, no block scaling, no integer
saturation), joint `M = 256` (128 local rows per CTA), `N` in
`{64,128,256}`, `K = 16` (implied by `.kind::f16` dense BF16), no transpose,
no negate.

**K = 16 and the M=256/cta_group::2 shape are documented facts, re-checked
for this exact combination, not assumed by analogy with P2.1's M=128.** PTX
ISA 9.3 section 9.7.17.2.1 (Table 44, "Various combinations of .kind and
shapes"), row `.kind::f16 / no .ws / CTA Group 2 / Dense`, lists shapes
`128xNxK` / `256xNxK` with `N = {16,32,...,256}` steps of 16 and `K = 16`
for `atype/btype in {.f16, .bf16}, dtype = .f32`. `N in {64,128,256}` (the
frozen P2.2 matrix, section 3) is a subset of this valid step-16 range, so
every specialization is a documented, valid shape for `cta_group::2`
`.kind::f16` dense.

### 6.1 Instruction descriptor (32-bit, PTX ISA 9.3 Table 47, `.kind::f16` column)

Identical bit layout to P2.1's descriptor (Table 47 does not vary by
`.cta_group`; `.cta_group` is a mnemonic qualifier, not a descriptor bit
field) -- only the encoded **value** of the M field differs:

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
| 24-28 | M (4 LSBs not included) | `256 >> 4 = 16` |
| 29 | reserved | 0 |
| 30-31 | `.ws` B-reuse max shift (n/a, no `.ws`) | 0 |

Built by `make_instruction_descriptor<N>()` and independently re-derived
field-by-field by `validate_instruction_descriptor<N>()`
(`src/compute/umma_2sm.cu`), `static_assert`ed for all three N values plus an
explicit non-zero-M-field assertion (task requirement: re-check the M=256
encoding, not assume it from the M=128 case).

### 6.2 Shared memory descriptor (64-bit, PTX ISA 9.3 Table 45)

Bit-for-bit identical formula to P2.1's (Table 45 depends only on the
operand's own address/LBO/SBO, never on M or `.cta_group`), independently
re-derived in `make_smem_descriptor()`: LBO = 128 bytes, SBO = 256 bytes (see
section 6.3 for the K-major derivation), matrix start address per operand.
Built once per CTA from that CTA's own local A/B shared-memory pointers.

### 6.3 The K-major layout

Identical derivation to P2.1's (`src/compute/P2_PROTOCOL.md` section 8):
PTX ISA 9.3 section 9.7.17.3.3's canonical K-major, no-swizzle layout
`((8,m),(T,2k)):((1T,SBO),(1,LBO))` with `T = 128/16 = 8` for BF16 and
`K=16 = 2T`, giving `LBO = 64 elements = 128 bytes`, `SBO = 128 elements =
256 bytes`, implemented by `smem_core_tile_index()`. This layout depends
only on K=16 and BF16's T=8, never on M, `.cta_group`, or which CTA rank is
executing, so it is re-derived (not merely copied) for P2.2 and is
byte-identical in form to P2.1's -- exactly what section 6's task
requirement to "independently re-derive and validate the descriptors" asks
for, arriving at the same LBO/SBO because the underlying operand shape (128
local rows/N columns, K=16) is genuinely unchanged per CTA.

## 7. Operand distribution and CTA-pair synchronization

Each CTA owns, in its own local shared/Tensor Memory:

```
A: 128 local rows x 16 BF16 values   (value depends on the GLOBAL row)
B: a complete 16 x N BF16 copy       (identical in both CTAs)
D: 128 local output rows x N FP32 values, in local TMEM
```

Global output mapping: `global_row = cta_rank * 128 + local_row`. CTA rank 0
owns global rows 0-127; CTA rank 1 owns global rows 128-255.

**Why B is replicated and why A/B/mbarrier/TMEM-address must share the same
relative SMEM offset.** PTX ISA 9.3 section 9.7.17.10's introduction states:
"the B matrix has shape KxN, in Shared Memory of the current CTA and
*optionally* in peer CTA." A 64-bit shared-memory descriptor (Table 45)
encodes an address that is always relative to the *issuing* thread's own
CTA (section 9.7.17.4.1: "location in the shared memory of the current
CTA"); only CTA rank 0's leader thread ever builds and passes `a_desc`/
`b_desc`. For the CTA-pair hardware to locate CTA rank 1's own local
contribution to the joint M=256 operation (A rows 128-255) and its own local
copy of B without a second, explicit descriptor, it must apply the *same*
descriptor -- interpreted as a relative offset -- to each CTA's own local
shared memory bank. This is why task section 6 requires A, B, the mbarrier,
and the TMEM-address shared variable to occupy **identical relative
shared-memory offsets in both CTAs**: since both CTAs execute the textually
identical kernel body (ordinary SPMD CUDA C++, no per-rank divergent
`__shared__`/`extern __shared__` declaration), the compiler lays these
variables out identically for every CTA instance, satisfying the requirement
by construction. B is therefore filled identically and redundantly in both
CTAs (task section 6's explicit requirement), and A's *value* -- never its
physical SMEM position -- depends on the global row, so the two halves can
never be silently exchanged or duplicated (section 11).

Before any of the numbered steps below, every kernel first reads
`%cluster_ctarank`/`%cluster_nctarank` (`cuda::ptx::get_sreg_cluster_ctarank
()`/`get_sreg_cluster_nctarank()`) and evaluates an explicit launch-contract
guard (`launch_contract_is_valid()`): a launch that is not exactly
`grid=(2,1,1)`, `cluster=(2,1,1)`, `block=(128,1,1)` writes `0` to that CTA's
own slot of a host-visible two-element `g_launch_ok` array and returns
immediately, before touching `__syncthreads()`, a cluster barrier, mbarrier
initialization, TMEM allocation, or any UMMA instruction. A launch that
passes writes `1` to that same slot before continuing to step 1. Both CTAs'
slots let the host confirm the guard's outcome was uniform across the pair.
The predicate depends only on values that are identical for the whole
cluster (`gridDim`, `blockDim`, `%cluster_nctarank`) plus the ISA-guaranteed
range fact `0 <= %cluster_ctarank < %cluster_nctarank` (PTX ISA 9.3 section
10.16), so both CTAs independently compute the same accept/reject verdict --
no accepted launch ever lets one CTA proceed into a collective operation
while its peer has rejected and returned (task section 5's explicit
requirement).

Per kernel, in order (PTX ISA 9.3 sections 9.7.17.1.2, 9.7.17.5.1,
9.7.17.6.5, 9.7.17.7.1, 9.7.17.12.1):

1. All 128 threads of both CTAs fill their own local A and B directly into
   the fixed K-major physical layout (section 11's validation pattern).
2. Local `__syncthreads()` after initialization.
3. One local mbarrier per CTA (`mbarrier_init`, expected count 1), at the
   same relative SMEM offset in both CTAs by construction.
4. Thread 0 of each CTA, in order: `fence.mbarrier_init.release.cluster`
   (publishing THIS CTA's own mbarrier initialization to the whole cluster,
   distinct from and required in addition to the next fence -- see the new
   section 7.1 below), then `fence.proxy.async.shared::cluster`
   (`cuda::ptx::fence_proxy_async(cuda::ptx::space_cluster)`, publishing
   that CTA's A/B writes to the async proxy at **cluster** scope -- section
   8), then local `__syncthreads()` again.
5. Cluster synchronization (`barrier_cluster_arrive()`/`_wait()`, called by
   all 128 threads of both CTAs, mirroring `cooperative_groups::cluster_
   group::sync()`'s full-block-collective semantics) before any collective
   TMEM operation.
6. `tcgen05.alloc.cta_group::2` issued collectively by warp 0 of **both**
   CTAs (never a single elected lane, never only rank 0 -- PTX ISA 9.3
   Table 51 requires one warp from each CTA of the pair), exactly N
   columns.
7. Cluster synchronization before using the allocated TMEM (both CTAs' own
   `tmem_d` must be locally published and cross-CTA-visible before rank 0's
   leader issues the joint MMA).
8. UMMA issue by one elected thread in CTA rank 0 only (`cuda::ptx::
   elect_sync`, gated additionally by `cta_rank == 0`).
9. Completion multicast to the two CTA-local mbarriers (`tcgen05.commit.
   cta_group::2...multicast::cluster.b64`, ctaMask `0x0003`; section 9).
10. Completion wait, CTA-wide publish, and cluster-wide rendezvous, once
    per mbarrier phase (i.e. inside every outer iteration, before the next
    commit -- see the new section 10.1 below): each CTA's own elected
    leader waits on that CTA's own local mbarrier
    (`mbarrier_try_wait_parity`), never enclosed in a `cta_rank == 0`
    condition; that successful wait is then published to the whole CTA
    with `__syncthreads()`; then every non-exited thread of BOTH CTAs
    (never just leaders, never just rank 0) rendezvouses with a full
    cluster arrive/wait (`barrier_cluster_arrive()`/`_wait()`) before the
    loop's back-edge.
11. TMEM readback executed by both CTAs, every thread, never enclosed in a
    `cta_rank == 0` condition; each CTA reads only its own local 128 TMEM
    rows (section 10).
12. Cluster synchronization after the final TMEM access.
13. `tcgen05.dealloc.cta_group::2` issued collectively by warp 0 of both
    CTAs.
14. `tcgen05.relinquish_alloc_permit.cta_group::2` issued collectively by
    warp 0 of both CTAs, as required by the ISA (section 9.7.17.7.1: "it is
    illegal for a CTA to perform tcgen05.alloc after any of its constituent
    threads execute tcgen05.relinquish_alloc_permit").
15. Final mbarrier invalidation (`mbarrier.inval.shared.b64`), by thread 0
    of each CTA, after it can no longer be referenced by any pending
    collective operation.

There are exactly two exit paths: the launch-contract rejection above
(before step 1, so neither TMEM nor a cluster barrier is ever reached on
this path) and the function's natural end after step 15. Every accepted
launch reaches all fifteen steps unconditionally. All `tcgen05` instructions
in `umma_2sm.cu` use `cta_group::2`; no executable `cta_group::1` instruction
exists anywhere in the file (verified by the source checker, section 14).

### 7.1 The mbarrier-initialization fence: `fence.mbarrier_init.release.cluster` (repair)

Step 4 issues a fence that the pre-repair implementation omitted entirely:
`fence.mbarrier_init.release.cluster` (PTX ISA 9.3, "Parallel
Synchronization and Communication Instructions", Membar/Fence
Instructions). It is issued BEFORE `fence.proxy.async` (section 8), and the
two are not interchangeable -- neither substitutes for the other:

* `fence.proxy.async` publishes ORDINARY (generic-proxy) memory writes --
  the A/B shared-memory fill of step 1 -- to the ASYNC proxy that
  `tcgen05.mma` reads through (PTX ISA 9.3 section 9.7.17.6.5).
* `fence.mbarrier_init.release.cluster` instead publishes the
  INITIALIZATION performed by `mbarrier.init` itself (step 3) to every
  thread of the cluster, so that a later arrive-on operation targeting this
  mbarrier from the PEER CTA -- CTA rank 0's multicast
  `tcgen05.commit...multicast::cluster` (step 9), which arrives on CTA rank
  1's own local mbarrier at the identical relative SMEM offset (section 7)
  -- is guaranteed to observe a fully initialized barrier object rather
  than racing its initialization.

Implementation: `fence_mbarrier_init_release_cluster()`
(`src/compute/umma_2sm.cu`) calls the pinned CUDA 13.1.80 toolchain's
official wrapper, `cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release,
cuda::ptx::scope_cluster)`
(`cuda/__ptx/instructions/generated/fence_mbarrier_init.h`, included
transitively by the top-level `<cuda/ptx>` header via
`cuda/__ptx/instructions/fence.h`; confirmed present in the pinned image
and unconditionally lowering to `fence.mbarrier_init.release.cluster;` for
`__CUDA_ARCH__ >= 900`, which `sm_103a` satisfies) -- matching this file's
existing convention of using the official wrapper for every primitive
genuinely covered by `<cuda/ptx>` (`mbarrier_init`, `fence_proxy_async`,
`elect_sync`, `get_sreg_cluster_ctarank`/`_nctarank`,
`barrier_cluster_arrive`/`_wait`, `mbarrier_try_wait_parity`); only the
tcgen05 family (unwrapped by this pinned toolchain, Blackwell-only) uses
hand-written inline PTX in this file. `tcgen05.wait::ld` and
`tcgen05.fence::after_thread_sync` have no distinct SASS footprint on this
toolchain (section 15); `fence.mbarrier_init.release.cluster` is a
comparable pure ordering/visibility fence with no data movement, so its
presence is proved via a mandatory, structural source check of the real
executable call site -- never a comment or an ordinary (non-asm) string
literal (`scripts/check_umma_2sm_sass.py`, section 14).

Ordering (source-checker-enforced): `fence_mbarrier_init_release_cluster()`
must be called after `mbarrier_init(&mbar, ...)` (step 3) and before the
first cluster barrier that publishes CTA-local initialization to the pair
(step 5's `barrier_cluster_arrive()`/`_wait()`).

## 8. Memory ordering: why `fence.proxy.async` is issued at cluster scope

PTX ISA 9.3 section 9.7.17.6.5 ("Shared Memory Accesses"): "The shared
memory accesses by `tcgen05.mma` and `tcgen05.cp` operations are performed
in the asynchronous proxy (async proxy). Accessing the same memory location
across multiple proxies needs a cross-proxy fence. For the async proxy,
`fence.proxy.async` should be used to synchronize memory between generic
proxy and the async proxy." P2.1 uses `cuda::ptx::fence_proxy_async(cuda::
ptx::space_shared)`, which the pinned CUDA 13.1 `<cuda/ptx>` header lowers
to `fence.proxy.async.shared::cta` -- a **CTA-scoped** fence, sufficient
because P2.1's single-CTA MMA only ever reads that same CTA's own SMEM.

For `cta_group::2`, CTA rank 0's issued MMA reads CTA rank 1's own local A
(and, under either plausible reading of the CTA-pair hardware mechanism --
a genuine remote read over the inter-SM interconnect, or an
identically-applied local read on each CTA's own datapath -- CTA rank 1's
own local B) at the identical relative SMEM offset (section 7). The public
PTX ISA text does not fully disambiguate which of those two physical
mechanisms is used. This implementation therefore uses `cuda::ptx::
fence_proxy_async(cuda::ptx::space_cluster)`, confirmed by inspection of the
pinned toolchain's `<cuda/ptx>` header to lower to `fence.proxy.async.
shared::cluster` (verified to compile under CUDA 13.1.80, section 15): a
cluster-scoped fence is a strict superset of a CTA-scoped one (anywhere the
narrower fence would suffice, the wider one also suffices), so this is the
safe choice under either interpretation of the hardware mechanism, and it
costs nothing measurable since it executes entirely outside the timed
region (section 12).

## 9. TMEM addressing: per-CTA, not a joint 256-lane space

PTX ISA 9.3 section 9.7.17.1 ("Tensor Memory"): "the 5th generation
TensorCore's Tensor Memory has a two-dimensional structure of 512 columns
and **128 rows per CTA**." Section 9.7.17.8.1 ("Access restrictions")
splits those 128 rows into four 32-lane chunks, one per warp of the
warpgroup -- identically to P2.1's single-CTA case. There is no such thing
as "TMEM lanes 128-255" to address: Tensor Memory is per-CTA, so CTA rank
1's own local M-rows 0-127 (its own warp-rank 0-3) map to CTA rank 1's own
local TMEM lanes 0-127, exactly like CTA rank 0's. `make_tmem_load_address
(tmem_base, warp_id, frag)` (`src/compute/umma_2sm.cu`) is therefore
byte-identical in form to P2.1's helper of the same name -- independently
re-derived here, not shared code -- and is called with `tmem_d` alone,
**never** offset by `cta_rank`.

PTX ISA 9.3 section 9.7.17.10.5.1 ("Layout A (M = 256)"), Figure 205 and
Figure 206, cross-checks this decomposition directly: it maps the joint
M=0..255 output as (`warp-rank % 4`, even/odd CTA in the CTA pair), and its
corresponding address table shows the even CTA's four warp-rank bands at
lane offsets `0x0000, 0x0020, 0x0040, 0x0060` and the odd CTA's four
warp-rank bands at `0x0080, 0x00A0, 0x00C0, 0x00E0` -- i.e. the *global*
M-index decomposes exactly into (CTA rank, local warp-rank), with each rank
covering the *same* four local lane offsets `{0x0000..0x0060}` from its own
CTA's perspective. Table 51 additionally confirms `.ld`/`.st` are "N/A"
for `.cta_group` and always access "the current CTA['s]" own Tensor Memory,
so this decomposition is documentation of the global numbering convention,
not a literal unified-256-lane address a single CTA's `tcgen05.ld` may use.
The rank offset therefore belongs only in the **global output index**:

```
global_row = cta_rank * 128 + local_row
```

used solely when writing `g_d_out` (host-visible global memory), never in
the TMEM address itself. `scripts/check_umma_2sm_sass.py`'s source gate
(section 14) proves both halves of this separation structurally: the TMEM
load call site must be exactly `make_tmem_load_address(tmem_d, warp_id,
frag)` with no `cta_rank` token anywhere in that call's argument list, and
the `g_d_out` write must be indexed by `global_row`, not `local_row`.

## 10. UMMA burst semantics

For every runtime outer iteration:

```
depth x tcgen05.mma.cta_group::2.kind::f16   (issued by CTA rank 0's elected leader only)
1     x tcgen05.commit.cta_group::2          (multicast, ctaMask = 0x0003)
1     x completion phase, waited by both CTAs' own elected leader on its own local mbarrier
```

The burst is completely unrolled at compile time (`#pragma unroll` over a
template non-type parameter, verified in SASS -- section 14). Within each
burst: `UMMA 0` has `enable-input-d = false` (D = A*B); `UMMA 1..depth-1`
have `enable-input-d = true` (D = A*B + D). Each outer iteration therefore
restarts D rather than accumulating across separate iterations, identically
to P2.1's semantic (`src/compute/P2_PROTOCOL.md` section 6).

The commit uses the ISA-supported multicast form (PTX ISA 9.3 section
9.7.17.12.1): `tcgen05.commit.cta_group::2.mbarrier::arrive::one.
shared::cluster.multicast::cluster.b64 [mbar], ctaMask;`. "The optional
qualifier `.multicast::cluster` allows signaling on the mbarrier objects of
multiple CTAs in the cluster. Operand `ctaMask` specifies the CTAs in the
cluster such that each bit position in the 16-bit `ctaMask` operand
corresponds to the `%cluster_ctarank` of the destination CTA. The mbarrier
signal is multicast to the **same offset** as `mbar` in the shared memory of
each destination CTA." `ctaMask = 0x0003` (bits 0 and 1) selects exactly
`%cluster_ctarank` 0 and 1 -- both CTAs of this exactly-two-CTA cluster, no
more, no fewer. Because A, B, and the mbarrier all occupy identical relative
SMEM offsets in both CTAs (section 7), "the same offset ... in the shared
memory of each destination CTA" correctly resolves to each CTA's own local
mbarrier.

`total_umma`/`flops_per_umma` accounting (section 12) is never doubled to
"account for two CTAs": one `cta_group::2` instruction represents the whole
joint M=256 operation, issued once per depth-position per outer iteration,
regardless of the fact that its effects land in two CTAs' Tensor Memory.

### 10.1 Why the rendezvous is required once per mbarrier phase (repair)

PTX requires at least one successful `mbarrier.test_wait`/`mbarrier.
try_wait` observation of a given mbarrier phase before any later arrive-on
operation targets that same mbarrier again. Because
`tcgen05.commit...multicast::cluster` (this section's completion multicast)
arrives on BOTH CTAs' local mbarriers from CTA rank 0's leader alone, CTA
rank 0 must not be allowed to begin mbarrier phase P+1's commit until it is
certain CTA rank 1's leader has already SUCCESSFULLY observed phase P's
completion -- not merely that rank 0 itself issued it. A single elected
leader's own local `mbarrier_try_wait_parity` call proves only that ITS OWN
CTA's local mbarrier reached the expected phase; it says nothing about the
PEER CTA's own local mbarrier or its leader's progress.

The fix (task section 3): after each outer iteration's local wait succeeds,
that success is published to the whole CTA with `__syncthreads()`, and then
EVERY thread of BOTH CTAs -- never just elected leaders, never just rank 0
-- rendezvouses with a full cluster arrive/wait
(`barrier_cluster_arrive()`/`_wait()`) before the loop's back-edge. This
rendezvous therefore runs once per mbarrier phase (every outer iteration),
not once after the whole loop: rank 0's leader cannot proceed past the
cluster barrier until rank 1's leader has also reached it, which cannot
happen until rank 1's leader has itself completed its own successful local
wait for that same phase. Because this sequence sits between the two
`%clock64` reads (section 12), its cost is included in the measured timed
region -- it is a real, unavoidable part of every measured iteration, not
an artifact excluded from the numbers.

The pre-repair implementation instead placed a single
`barrier_cluster_arrive()`/`_wait()` pair only once, AFTER the entire
`iterations` loop had already completed, with only the elected leaders
(not the whole cluster) executing the loop at all -- so CTA rank 0's
leader could freely begin the next iteration's commit as soon as its OWN
local wait succeeded, with no guarantee that CTA rank 1's leader had
observed the PRECEDING phase's completion first. `scripts/
check_umma_2sm_sass.py`'s source gate (section 14) now proves structurally
that the runtime outer loop is executed uniformly by the whole cluster,
that the CTA sync/cluster rendezvous sequence sits INSIDE that loop (never
after it), that it is reachable by every thread (never confined to
`is_leader` or `cta_rank == 0`), and that it follows the leader block's
successful wait.

## 11. Correctness method

Per-element validation pattern, deterministic, chosen so a wrong
global-row mapping, rank duplication, missing rank offset, wrong B
replication, column-addressing error, TMEM fragment-offset error, missing
depth accumulation, or cross-CTA synchronization error all manifest as a
numerical mismatch:

```
A(global_row, k) = ((global_row + 3*k) % 7) - 3      global_row in [0,256), k in [0,16)
B(k, col)         = ((2*k + col) % 5) - 2             k in [0,16),  col in [0,N)

reference(global_row, col) = depth * sum_{k=0..15} A(global_row,k) * B(k,col)
```

Identical mathematical form to P2.1's pattern (`src/compute/P2_PROTOCOL.md`
section 11), independently re-derived here for the 256-row global range: the
device-side A initialization uses `global_row = cta_rank * 128 + local_row`
exactly as the CPU reference does, so any of the following device-side
defects produces a nonzero `max_abs_error`:

* **Rank-0/rank-1 duplication** -- if CTA rank 1 mistakenly used its own
  `local_row` (0-127) instead of `global_row` (128-255) for A's value, since
  `128 mod 7 = 2`, the resulting pattern for every one of CTA rank 1's rows
  would be shifted by a residue of 2 from the correct value, a detectable
  mismatch for generic `k`/`col`.
* **Incorrect global-row mapping** (off-by-128, wrong stride, forgetting the
  offset) -- any deviation from `cta_rank * 128 + local_row` changes A's
  value pattern by the same residue-shift argument.
* **Incorrect B replication** -- both CTAs fill B identically by construction
  (section 7); the source checker additionally proves B's fill loop body
  never references `cta_rank` (section 14).
* **Column-addressing / TMEM fragment-offset errors** -- caught the same way
  P2.1's identical readback loop catches them (`src/compute/P2_PROTOCOL.md`
  section 9.1's repair history is the precedent for why this class of bug is
  realistic and must be checked structurally, not just numerically).
* **Missing depth accumulation** -- `reference(...)` scales by `depth`;
  `iterations` never appears in it, matching the "restart-per-outer-
  iteration" semantic (section 10).
* **Cross-CTA synchronization errors** -- if CTA rank 1's readback raced
  ahead of CTA rank 0's completed MMA (missing/incorrect multicast, missing
  cluster barrier), the mbarrier wait -- signaled only by the real hardware
  completion of the tracked asynchronous `tcgen05.mma` operations, section
  9.7.17.6.2.2 -- would simply not observe the correct phase in time, and the
  read TMEM value would not match a fully-completed computation.

`|A| <= 3`, `|B| <= 2`, `|sum_16 terms| <= 96`, `|reference| <= 256 * 96 =
24576` (depth max 256), versus FP32's exact-integer range of +-2^24, so the
contract is `GPU result == CPU reference` bit-for-bit, `max_abs_error == 0`,
checked for **all `256 * N` elements** (task section 9: "Validate all
`256 x N` FP32 outputs, not only CTA rank 0"), not a sample, not a checksum.
Correctness must pass before any timing (AGENTS.md, section 12).

## 12. Timing and CSV semantics

Timed region: identical exclusion list to P2.1's (`src/compute/P2_PROTOCOL.md`
section 12), plus the CTA-pair-specific steps: included are the `depth`
UMMA issues per iteration (rank 0 leader only), one multicast commit per
iteration, the completion wait for every iteration (both CTAs' elected
leaders), and -- required by the per-phase handshake (section 10.1) -- the
CTA-wide `__syncthreads()` and the full cluster `barrier_cluster_arrive()`/
`_wait()` rendezvous that follow it every iteration, since that rendezvous
is what actually gates when rank 0's leader may issue the next commit.
Excluded: kernel launch, A/B initialization, descriptor construction,
mbarrier initialization and its dedicated fence (section 7.1), cluster
synchronization around TMEM allocation, TMEM allocation, warm-up, reading D
from TMEM, global-memory stores, device-to-host copy, CPU validation,
cluster synchronization before deallocation, TMEM deallocation. Cycles are
never converted to seconds or TFLOP/s here (P2.4 work).

`%clock64` is read only by CTA rank 0's elected leader thread, and only when
`timing_mode == TimingMode::kTimed` (identical `TimingMode` untimed/timed
split to P2.1's -- section 10.1 of `src/compute/P2_PROTOCOL.md` -- with
added `is_leader` and `cta_rank == 0` conjuncts, both explicit in the exact
guard `if (is_leader && cta_rank == 0 && timing_mode == TimingMode::
kTimed)`): `--self-test`, pre-timing correctness validation, and every
warm-up iteration always launch with `TimingMode::kUntimed`; only the
per-repetition timed loop launches with `TimingMode::kTimed`. CTA rank 1
never reads `%clock64` and never writes `g_elapsed_cycles`. Both CTAs still
participate in every collective operation, completion wait, and per-phase
cluster rendezvous regardless of timing mode.

FLOP/UMMA accounting:

```
umma_per_iteration = depth
total_umma          = depth * iterations
flops_per_umma      = 2 * 256 * N * 16
total_flops          = total_umma * flops_per_umma
```

`total_umma` is **not** multiplied by two: one `cta_group::2` instruction
represents the joint M=256 operation (section 10). Computed with
`checked_mul_i64` (`__int128`-checked 64-bit multiplication, host-only,
independently rewritten from P2.1's identical-purpose helper) in
`umma_2sm.cu`.

CSV schema: identical column order to P2.1's audited schema
(`src/compute/P2_PROTOCOL.md` section 14), one header line plus one row per
repetition on stdout, diagnostics on stderr:

```
schema_version,timestamp_utc,run_kind,publishable,method,sample_index,cta_group,m,n,k,depth,iterations,warmup_iterations,repetitions,umma_per_iteration,total_umma,flops_per_umma,total_flops,elapsed_cycles,cycles_per_umma,flops_per_cycle,threads_per_cta,grid_blocks,tmem_columns,operand_path,input_type,accumulator_type,correctness,mismatches,max_abs_error,gpu_name,gpu_uuid,compute_capability,cuda_driver_version,cuda_runtime_version,git_commit,git_dirty
```

Frozen values: `schema_version=1`, `publishable=false`, `method=umma_2sm`,
`cta_group=2`, `m=256`, `k=16`, `umma_per_iteration=depth`,
`threads_per_cta=128`, `grid_blocks=2`, `tmem_columns=n`,
`operand_path=smem_smem`, `input_type=bf16`, `accumulator_type=fp32`,
`correctness=OK`, `mismatches=0`, `max_abs_error=0`,
`compute_capability=10.3`. `elapsed_cycles>0`; `cycles_per_umma` and
`flops_per_cycle` are `double`s computed from the exact integer counters.
These are technical evidence from a functional implementation, not
publishable results (`publishable=false` on every row, unconditionally).

## 13. `--self-test` and CLI contract

`--self-test` exercises all twelve specializations, untimed, and finishes
only with `SELF_TEST: PASS (12/12)` on stdout (diagnostics per
specialization on stderr). CLI flags are identical to P2.1's audited surface
(task section 10: "adapt the audited P2.1 host interface"): `--help`,
`--self-test`, `--run-kind {smoke,benchmark}`, `--n {64,128,256}`,
`--depth {4,16,64,256}`, `--iterations`, `--warmup-iterations`,
`--repetitions`. `--help` succeeds without CUDA initialization (confirmed:
no `cudaGetDeviceCount`/`cudaSetDevice` call executes before `--help`
returns); `-h` and any other unrecognized flag are rejected with exit code 2.
Before any GPU work: exactly one visible CUDA device, `EXPECTED_GPU_UUID`
match, compute capability 10.3, and a clean correctness run before any
timing -- all identical in form to P2.1's contract, independently
implemented.

## 14. SASS and source contract

Enforced by `scripts/check_umma_2sm_sass.py` against real `cuobjdump -sass`
**and** `cuobjdump -elf` output of `build/compute/umma_2sm` compiled for
`sm_103a` with CUDA 13.1.80 `ptxas`/`nvcc` (mnemonics and ELF attributes
observed directly from this project's own compiled binary and from isolated
single-instruction probes compiled the same way -- never guessed -- see that
script's module docstring for the full PTX-to-SASS/ELF mapping table).
Summary of what is proved for every one of the twelve symbols:

1. Exactly the twelve expected symbols exist, no missing/extra/duplicate.
2. Every symbol's SASS contains `UTCHMMA.2CTA` (sm_103a's lowering of
   `tcgen05.mma.cta_group::2.kind::f16`); the static count equals `depth`
   exactly, at a uniform address spacing (compile-time unrolling evidence,
   not a runtime back-edge).
3. The burst ends with a real completion sequence: `UTCBAR.2CTA.MULTICAST`
   (the multicast commit) after the last `UTCHMMA.2CTA`, then at least one
   `SYNCS.PHASECHK.TRANS64.TRYWAIT` (mbarrier wait) after the commit.
4. Collective TMEM allocation (`UTCATOMSWS.2CTA.FIND_AND_SET.ALIGN`) and
   deallocation (`UVIRTCOUNT.DEALLOC.SMPOOL`) are present, with a
   cluster-barrier pair (`UCGABAR_ARV`/`UCGABAR_WAIT`) ordered strictly
   between the last TMEM use and the deallocation.
5. `LDTM.x32` (TMEM-to-register load) appears exactly `N/32` times.
6. Cluster-rank evidence (`SR_CgaCtaId`, sm_103a's lowering of
   `%cluster_ctarank`) is present.
7. The compiled ELF's `.nv.info.<symbol>` section carries both
   `EIATTR_EXPLICIT_CLUSTER` and `EIATTR_CTA_PER_CLUSTER` with value
   `0x2 0x1 0x1` -- a direct, per-kernel, binary-level record of the
   compile-time two-CTA cluster declaration.
8. No non-`.2CTA` (1-SM-fallback-shaped) `UTCHMMA`, no non-`.2CTA.
   MULTICAST` `UTCBAR`, no non-`.2CTA` TMEM allocation, and no
   `HMMA`/`WGMMA`/`QGMMA`/`IMMA`/`BMMA` (non-tcgen05 MMA), `UTMALDG` (TMA),
   `LDGSTS`, `UBLKCP`, or sparse (`.sp`) `UTCHMMA` qualifier appears anywhere
   in the binary.
9. (mandatory source check) the canonical source contains every required
   `cta_group::2` PTX form (`tcgen05.mma`, the multicast `tcgen05.commit`,
   `tcgen05.alloc`, `tcgen05.dealloc`, `tcgen05.relinquish_alloc_permit`),
   `tcgen05.wait::ld.sync.aligned` and `tcgen05.fence::after_thread_sync` as
   executable code (no distinct SASS footprint on this toolchain, same as
   P2.1), `__cluster_dims__(2, 1, 1)`, `get_sreg_cluster_ctarank`/
   `_nctarank`, `barrier_cluster_arrive`/`_wait`, and the exact literal
   multicast mask `0x0003`; and contains **no** `cta_group::1`, no non-
   `kind::f16` MMA kind, no `.sp` sparse form, and no `block_scale`, all as
   executable code (comment- and string-literal-aware scan).
10. (source check) the launch-contract predicate depends on `gridDim`,
    `blockDim`, `cluster_nctarank`, and `cluster_ctarank`, is evaluated
    (negated) before `umma_2sm_body`'s first `__syncthreads()`, and both the
    rejection and acceptance paths write **both** ranks' `g_launch_ok` slots.
11. (source check) A's fill loop references `cta_rank`; B's fill loop does
    not; the TMEM load call site is exactly `make_tmem_load_address(tmem_d,
    warp_id, frag)` with no `cta_rank` token in its arguments; the `g_d_out`
    write is indexed by `global_row`, not `local_row`.
12. (source check) `tcgen05_alloc_2sm`/`tcgen05_dealloc_2sm`/
    `tcgen05_relinquish_alloc_permit_2sm` are issued only from an
    `if (warp_id == 0)` block, never from inside a `cta_rank == 0` or
    `is_leader` conditional.
13. (source check) `issue_one_umma_2sm`/`commit_umma_2sm_multicast` are
    confined to a single `if (cta_rank == 0)` block nested inside the
    per-iteration `if (is_leader)` block (section 7 step 10), with the exact
    literal mask `0x0003`, and no additional issue/commit call site exists
    anywhere else in the function; the mbarrier wait and the TMEM readback
    loop are present but **not** confined to any `cta_rank == 0` block.
14. (source check) a `barrier_cluster_arrive()`/`barrier_cluster_wait()`
    pair appears, textually, between the final TMEM access and
    `tcgen05_dealloc_2sm`.
15. (source check) every timed `%clock64` read is guarded by the exact
    conjunction `is_leader && cta_rank == 0 && timing_mode ==
    TimingMode::kTimed`, and at least one call site uses
    `TimingMode::kUntimed`.
16. (source check, repair) exact geometry: `kThreadsPerCta = 128`,
    `kClusterCtas = 2`, `kGridBlocks = 2`, and `__launch_bounds__(128)` are
    real constants/attributes (`__cluster_dims__(2, 1, 1)` is independently
    required by item 9 above), and the host launch (`run_once`) actually
    launches `spec.kernel<<<kGridBlocks, kThreadsPerCta, ...>>>` -- proving
    `grid=(2,1,1)`/`block=(128,1,1)` from the real launch call, not merely
    from the constants' own declared values.
17. (source check, repair) the mbarrier-initialization fence (section 7.1):
    a single, real `fence_mbarrier_init_release_cluster()` helper is defined
    and its own body genuinely calls the official
    `cuda::ptx::fence_mbarrier_init(cuda::ptx::sem_release,
    cuda::ptx::scope_cluster)` wrapper; `umma_2sm_body` calls this helper
    exactly once as a direct, unconditionally reachable statement in the
    same `if (tid == 0)` initialization block, program-ordered immediately
    after `mbarrier_init(&mbar, ...)` and before the separate, real
    `fence_proxy_async` call and the first cluster barrier that publishes
    CTA-local initialization to the pair.
18. (source check, repair) the per-phase CTA-pair handshake (section 10.1):
    the runtime outer iteration loop (`for (int64_t it = 0; it <
    iterations; ++it)`) is executed uniformly by the whole cluster -- never
    nested inside `is_leader` or `cta_rank == 0`; neither `__syncthreads()`
    nor `barrier_cluster_arrive()`/`_wait()` may appear inside the loop's
    per-iteration `if (is_leader)` block; and, inside that same loop body,
    after the leader block, `__syncthreads()`, then
    `barrier_cluster_arrive()`, then `barrier_cluster_wait()`, appear in
    that exact order as direct statements reachable by every thread in both
    CTA ranks. The local wait and its single `parity ^= 1u` phase advance
    must likewise be direct statements reachable by both CTA leaders, never
    hidden in an additional rank condition.
19. (source check, re-audit repair) the live TMEM load-completion route: the
    unique `tcgen05_wait_ld()` helper contains real inline-PTX evidence, and
    the canonical `kFragments = N / 32` readback loop contains exactly one
    direct `tcgen05_ld_32x32b_x32(...)` call followed by exactly one live
    `tcgen05_wait_ld()` call before the loaded registers are written.

`scripts/check_umma_2sm_sass.py --self-test` exercises all of the above (93
cases total): 23 SASS-contract cases (missing/extra/duplicate symbol,
missing/incorrect-depth/non-uniformly-spaced burst, missing commit/wait/
alloc/dealloc, non-`.2CTA` fallback forms for MMA/commit/alloc, incorrect
`LDTM.x32` count, missing cluster-barrier evidence in two distinct forms,
every forbidden whole-binary instruction), 4 ELF-attribute cases (accept,
missing attribute, missing section, wrong `EIATTR_CTA_PER_CLUSTER` value),
63 source-level positive and negative cases (every required/forbidden
pattern individually, comment- and ordinary-non-asm-string-literal-only
placement of required PTX text -- so a decoy never counts as evidence --
launch-guard structure and both ranks' status writes, A/B rank-dependence in
both directions, collective-vs-single-lane/rank-0-only TMEM lifecycle gating
in both directions, rank-0-confinement and the exact mask for MMA/commit
issue, the wait/readback non-confinement, TMEM-address-vs-global-index
separation in both directions, cluster-sync-before-dealloc, every timing-
route defect, both lexical-scan failure modes, exact-geometry regressions,
mbarrier-init-fence presence/body-content/ordering defects including
comment and string decoys, and every independent per-phase-handshake
mutation from the fourteen listed in task section 5, plus five independent
follow-up regressions covering an unused live TMEM-wait helper, a second
rank-0 condition around the local wait, rank-0-only CTA/cluster rendezvous,
a missing parity phase advance, and a conditionally unreachable
mbarrier-init fence), and 3 mandatory-
source-path-resolution cases (including that the repaired canonical source
itself is accepted with zero errors).

## 15. Compile-time validation methodology (this implementation's process)

Before writing `src/compute/umma_2sm.cu`, every instruction form used in it
(cluster-rank reads, cluster barriers, the CTA-scoped vs. cluster-scoped
`fence.proxy.async` forms, `tcgen05.alloc/mma/commit/dealloc/
relinquish_alloc_permit.cta_group::2`, the multicast commit with an explicit
`ctaMask`, `extern "C" __cluster_dims__(2, 1, 1) __launch_bounds__(128)`
combined with the same macro-stamped-kernel pattern used for the twelve
symbols) was first compiled in isolation, GPU-free, against the pinned
`gb300-gemm-anatomy:phase0` image's real CUDA 13.1.80 `nvcc`/`ptxas`, using
the same split `-arch=compute_103a -code=sm_103a` flags P2.1's Makefile
target already established as required for this toolchain
(`src/compute/P2_PROTOCOL.md` section 20). Each probe's real `cuobjdump
-sass`/`-elf` output was inspected directly to confirm the SASS mnemonic
table in section 14 and this document's citations, before any regular
expression in `scripts/check_umma_2sm_sass.py` was written. The real,
complete twelve-specialization `build/compute/umma_2sm` binary was then
compiled and disassembled the same way, and every SASS/ELF check in section
14 was independently cross-checked against its real output (exact per-
specialization instruction counts and address orderings, not representative
samples) before this document and the checker's synthetic self-test data
were finalized. This mirrors, and independently repeats for cta_group::2,
the evidence-gathering discipline `src/compute/P2_PROTOCOL.md` section 20
records for P2.1's own architecture-flag decision.

## 16. Commands

GPU-free (no Docker GPU, no network, used to produce and validate this
implementation):

```bash
python3 -m py_compile scripts/check_umma_2sm_sass.py
python3 scripts/check_umma_2sm_sass.py --self-test
make check-static
make compute-umma-2sm-build
make compute-umma-2sm-sass
make compute-umma-2sm-check
```

GB300 functional-verification commands (**not executed by this
implementation task**; recorded here only as the commands an operator should
later run -- see section 17):

```bash
BLACKWELL_GPU_INDEX=<physical-index> make preflight
BLACKWELL_GPU_INDEX=<physical-index> make compute-umma-2sm-self-test
BLACKWELL_GPU_INDEX=<physical-index> make compute-umma-2sm-smoke

BLACKWELL_GPU_INDEX=<physical-index> scripts/run_container.sh \
  build/compute/umma_2sm \
  --run-kind benchmark --n 128 --depth 16 \
  --iterations 20 --warmup-iterations 5 --repetitions 3
```

## 17. Verification status and scientific limitations

* P2.2 is **implemented**: the real `sm_103a` binary compiled cleanly under
  the pinned CUDA 13.1.80 toolchain and passed the full SASS/ELF/source
  contract above for all twelve specializations (section 14, section 15).
* A first independent audit of commit `e00046a415eec77663867dfd2c6691a1ab5a26d2`
  found four blockers, all in synchronization correctness, the source
  checker, and documentation honesty: (1) `mbarrier_init` was never followed
  by `fence.mbarrier_init.release.cluster` (section 7.1); (2) the runtime
  outer loop's inter-phase CTA-pair handshake was invalid -- only elected
  leaders executed the loop, and a single cluster barrier ran once after the
  whole loop instead of once per mbarrier phase, so CTA rank 0 could begin a
  new commit before CTA rank 1 had observed the preceding phase's completion
  (section 10.1); (3) `scripts/check_umma_2sm_sass.py` accepted source that
  violated (1) and (2) and could be satisfied by non-executable decoy text;
  (4) this document, `src/compute/P2_PROTOCOL.md`, and the `Makefile` help
  text still described P2.2 as unimplemented. All four were remediated
  GPU-free: the dedicated fence helper and its ordering (section 7.1), the
  restructured per-phase handshake (section 10.1), a hardened,
  comment/string-decoy-resistant source checker with 88 self-test cases
  including fourteen independent adversarial mutations of the handshake and
  geometry (section 14), and the corrected documentation here and in the
  files above.
* A follow-up independent audit of commit
  `b78695848bb73e46aa4f6f53cab155cc3375fea9` confirmed the canonical CUDA
  synchronization repair but found five remaining fail-open checker paths:
  it could accept removal of the live `tcgen05_wait_ld()` call, a second
  rank-0 condition around the local wait, a rank-0-only per-phase
  rendezvous, removal of the parity phase advance, and an unreachable
  mbarrier-init fence call. The 93-case checker described in section 14
  repairs those five paths with direct-scope, reachability, sequence, and
  live-call checks. This checker-only remediation is GPU-free and has
  **not** itself been independently audited.
* P2.2 has **not** been independently audited (a static self-check, however
  thorough, is not an audit -- AGENTS.md, `PLAN.md`). This includes the
  repair above: a fresh independent audit of the repaired implementation
  remains pending.
* P2.2 has **not** been verified on GB300 hardware. No `--self-test`,
  `smoke`, or `benchmark` invocation of `build/compute/umma_2sm` has been
  executed on a physical device by this implementation task; no GPU command
  of any kind (`nvidia-smi`, a CUDA kernel launch, Nsight Compute,
  `scripts/run_container.sh`, or any target requiring
  `BLACKWELL_GPU_INDEX`) was run.
* No publishable result exists or is claimed. Every CSV row P2.2 can ever
  emit carries `publishable=false` unconditionally; `elapsed_cycles` is a
  raw `%clock64` delta on CTA rank 0's one thread, not wall-clock time, not
  corrected for clock throttling/boost state, and not a throughput or
  saturation claim (P2.4 work).
* `tcgen05.wait::ld` and `tcgen05.fence::after_thread_sync` have no distinct
  SASS footprint on the observed toolchain (same as P2.1); their presence is
  proved only via a static source check.
* `UVIRTCOUNT.DEALLOC.SMPOOL` and the relinquish-permit lowering
  (`UTCATOMSWS.AND`) show no distinct `.2CTA` SASS qualifier on this
  toolchain (unlike alloc/mma/commit, which do); their presence, not an
  exact instruction-level 2-SM marker, is what the checker requires for
  these two operations specifically, matching P2.1's identical documented
  limitation for its own `cta_group::1` relinquish lowering.
* The exact physical mechanism by which CTA rank 0's issued MMA reads CTA
  rank 1's local A/B (a genuine remote interconnect read vs. an identically-
  applied local read on each CTA's own datapath) is not fully disambiguated
  by the publicly available PTX ISA text; section 8 documents the safe,
  strictly-sufficient choice made under this ambiguity (cluster-scoped
  `fence.proxy.async`).
* P2.3 (joint sweep) and P2.4 (profiling/ceiling) remain entirely
  unimplemented; nothing in this document or in `PLAN.md`'s P2.2 row claims
  otherwise.

## 18. Status

* P2.2: **implemented**. A first independent audit found four blockers
  (missing mbarrier-initialization fence, invalid per-phase CTA-pair
  handshake, a source checker that accepted both defects, and stale
  "unimplemented" documentation); all four were remediated GPU-free
  (sections 7.1, 10.1, 14, and this document/`src/compute/P2_PROTOCOL.md`/
  `Makefile`).
* Independent audit: **pending**. The audit above found and this round
  fixed four blockers; a fresh independent audit of the repaired
  implementation has not yet been performed.
* GB300 verification: **pending**.
* Publishable result: **none**. Every CSV row P2.2 can ever emit carries
  `publishable=false` unconditionally.
* P2.3, P2.4: **not implemented**.

## 19. References

Primary (normative): NVIDIA PTX ISA 9.3,
<https://docs.nvidia.com/cuda/parallel-thread-execution/>, chapter
"9.7.17. TensorCore 5th Generation Family Instructions" (sections
9.7.17.1-9.7.17.12 cited by number throughout this document), read from the
official PDF (`https://docs.nvidia.com/cuda/pdf/ptx_isa_9.3.pdf`) so every
table, figure, and worked example cited above was read in full, not
summarized. Sections and artifacts specifically relied on beyond P2.1's own
citation set: 9.7.17.5/9.7.17.5.1/9.7.17.5.2 (Issue Granularity, CTA Pair,
Peer CTA), 9.7.17.6.5 (cross-proxy fence), 9.7.17.7.1 (collective alloc/
dealloc/relinquish, including its `.2cta` deallocation-ordering worked
examples), 9.7.17.10.5/9.7.17.10.5.1 (Data Path Layout Organization, Layout
A for M=256, Figures 205-206), 9.7.17.10.9.1 (the `tcgen05.mma` syntax
block and its cta_group::2/kind::tf32 worked example), 9.7.17.12.1
(`tcgen05.commit`'s multicast form and its cta_group::2 worked example),
and chapter 10 sections 10.16-10.17 (`%cluster_ctarank`/`%cluster_nctarank`).

Secondary (conceptual, adapted and independently audited against the PTX
ISA, not copied): pinned commit `9a068d853d5c3676939eb46fe21ff6d6a2a4133b`
of `SemiAnalysisAI/microbench-blackwell/umma_throughput/umma_tput.cu`. Audit
findings, and how this implementation diverges:

* That file's inline-PTX templates for `tcgen05.mma.cta_group::%N.kind::...`
  and `tcgen05.commit.cta_group::%N.mbarrier::arrive::one.shared::cluster.
  multicast::cluster.b64` were used as an independent cross-check of the
  operand order and qualifier spelling this document derived from the PTX
  ISA text directly (section 6.1, section 10) -- both sources agree exactly.
* That file never validates numerical correctness for any configuration and
  never reads D back for the 2-SM path either, consistent with it being a
  pure throughput probe; this implementation's CPU reference (section 11)
  and its extension of the global-row-dependent validation pattern to the
  full 256-row range have no equivalent there.
* That file's `enable-input-d` predicate construction (`setp.ne.b32` inside
  inline asm on a plain integer argument) matches the PTX ISA and P2.1's own
  audited idiom; adopted directly for `issue_one_umma_2sm`.
* Its shared-memory descriptor bit placement (matching PTX ISA Table 45
  exactly) was used only as a cross-check, not as the source of this
  implementation's LBO/SBO values, which are independently re-derived from
  the PTX ISA's own K-major layout formula (section 6.3) and validated
  end-to-end by the CPU-reference numerical check, exactly as documented for
  P2.1 (`src/compute/P2_PROTOCOL.md` section 19).

Also consulted (conceptual only, no bit-level or CTA-pair-specific detail
found there beyond what the PTX ISA text itself already provides): NVIDIA
CUTLASS Blackwell functionality documentation,
<https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html>.
