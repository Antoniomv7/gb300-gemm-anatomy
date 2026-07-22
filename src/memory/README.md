# src/memory — P1.1 LDGSTS baseline and P1.2 TMA path

This directory holds both arms of the "LDGSTS versus TMA" experiment
(experiment 1 in `AGENTS.md`): the P1.1 LDGSTS baseline
(`src/memory/ldgsts.cu`) and the P1.2 2D unicast TMA path
(`src/memory/tma.cu`). Each measures the effective copy bandwidth of its own
global-memory-to-shared-memory software pipeline. Dynamic shared memory caps
the maximum active residency at one CTA per SM for both; neither observes or
guarantees runtime block placement.

**Status: P1.1 implemented, pending audit and GB300 verification. P1.2
implemented, pending audit and GB300 verification (see `PLAN.md`).** No
experimental numbers from either binary have been published in `README.md`
or anywhere else in the repository. P1.3 (the joint sweep) and P1.4
(profiling/analysis) have not started and remain blocked until P1.1 and P1.2
are both independently audited and verified on GB300.

## P1.1 — standalone LDGSTS baseline

## What P1.1 measures

- The *effective copy bandwidth* (`effective_gbps`) of a fixed, 16-byte
  vectorized `cp.async.cg.shared.global` pipeline moving data from a large
  global-memory working set into a per-SM shared-memory ring buffer, for
  each of nine frozen `(stages, bytes_in_flight_per_sm)` specializations.
- Whether every copied 16-byte vector, for the selected configuration, lands
  in shared memory with the exact bytes the deterministic source pattern
  predicts (`correctness`, `mismatches`).
- Whether the launch configuration and shared-memory reservation limit the
  occupancy API's maximum active residency to one CTA per SM
  (`occupancy_ctas_per_sm`).

## What it cannot yet claim

- **Not HBM/DRAM bandwidth.** `effective_gbps` is *effective copy bandwidth*:
  `useful_bytes` (derived from the frozen working-set/pass formula) divided
  by measured kernel time. It says nothing about where those bytes actually
  came from (L2 vs DRAM) — that requires Nsight Compute, which is out of
  scope until P1.4.
- **Not yet a TMA comparison.** TMA / `cp.async.bulk.tensor` is now
  implemented in `src/memory/tma.cu` (P1.2, `method=tma`), which expresses
  this file's exact 2D tile layout as 2D unicast TMA loads (see "P1.2 —
  standalone 2D unicast TMA path" below). The two arms are still not
  compared, aggregated, or run jointly: P1.2 is implemented but unaudited
  and unverified on GB300 (see `PLAN.md`), and the joint sweep that would
  actually compare them is P1.3, which has not started.
- **Not a sweep or an analysis.** This binary runs one specialization (or,
  under `--self-test`, validates all nine) per invocation. Aggregation,
  statistics, plots, and conclusions are P1.3/P1.4.
- **Not a final result.** `run_kind=smoke` output exists only to prove the
  binary and container plumbing work end to end; it is never a publishable
  measurement. `run_kind=benchmark` output is raw, per-repetition CSV with
  no statistics computed — still not a "result" until P1.3/P1.4 process it.

## Frozen contract

- Method: `ldgsts`. PTX instruction: `cp.async.cg.shared.global` (the `.cg`
  qualifier requires exactly 16-byte copies, which is why 16 bytes is fixed
  rather than configurable).
- 128 threads/CTA, grid = SM count, maximum active residency = 1 CTA/SM,
  stages ∈ {2, 4, 8}, bytes-in-flight/SM ∈ {16, 32, 64} KiB.
- One pipeline stage = one `cp.async` group = one `cp.async.commit_group`.
  Steady state waits with `cp.async.wait_group<Stages-1>` before reusing a
  ring slot; draining the pipeline waits with `cp.async.wait_group 0`.

### Bytes-in-flight formulas

```
stage_bytes                 = bytes_in_flight_per_sm / stages
copies_per_thread_per_stage = stage_bytes / (128 threads * 16 bytes)
bytes_in_flight_per_sm      = stages * stage_bytes
```

These formulas are the single source of truth in `src/memory/ldgsts.cu`
(`make_spec()`); the table below is their result, not an independent
hand-entry.

| Stages | BIF KiB | Stage KiB | 16B copies/thread/stage | Tile height (rows) |
| -----: | ------: | --------: | -----------------------: | ------------------: |
|      2 |      16 |         8 |                         4 |                   32 |
|      2 |      32 |        16 |                         8 |                   64 |
|      2 |      64 |        32 |                        16 |                  128 |
|      4 |      16 |         4 |                         2 |                   16 |
|      4 |      32 |         8 |                         4 |                   32 |
|      4 |      64 |        16 |                         8 |                   64 |
|      8 |      16 |         2 |                         1 |                    8 |
|      8 |      32 |         4 |                         2 |                   16 |
|      8 |      64 |         8 |                         4 |                   32 |

Runtime dispatch (`dispatch_run()`) selects the matching `<STAGES, COPIES>`
template instantiation of two kernel templates (`ldgsts_validate_kernel`,
`ldgsts_benchmark_kernel`); the kernel body is written once, not duplicated
nine times.

### 2D layout (shared with the future TMA arm)

- 128 BF16-width (2-byte) elements per row = 256 bytes/row
  (`tile_width_elements` / `tile_width_bytes`).
- `tile_height = stage_bytes / 256` rows — 8, 16, 32, 64, or 128 depending on
  the specialization.
- LDGSTS copies each tile as one contiguous linear region (row-major, no
  padding, so "2D tile" and "contiguous 16-byte-vector range" are the same
  bytes). P1.2 must be able to express exactly the same tiles as 2D TMA
  copies.
- The underlying buffer is raw 16-byte vectors (`uint4`), not a typed BF16
  array: this microbenchmark measures copied bytes, not arithmetic, so no
  BF16 arithmetic type is needed in memory.

### Maximum-active-CTA occupancy cap

`sharedMemPerMultiprocessor`, the max opt-in shared memory per block
(`cudaDevAttrMaxSharedMemoryPerBlockOptin`), and the SM count are all queried
at runtime — never hardcoded. For each specialization the binary reserves
dynamic shared memory strictly larger than half of
`sharedMemPerMultiprocessor` (so a second CTA can never fit), at least
`bytes_in_flight_per_sm`, aligned to 128 bytes, and capped at the max
opt-in value. It configures
`cudaFuncAttributeMaxDynamicSharedMemorySize` to that reservation and then
calls `cudaOccupancyMaxActiveBlocksPerMultiprocessor` for both the validate
and benchmark kernels; if either reports anything other than exactly 1, the
binary aborts with a diagnostic instead of continuing silently. Only the
first `bytes_in_flight_per_sm` bytes of the reservation are the actual ring
buffer; the rest is deliberate, untouched padding whose only purpose is
capping maximum active residency at one CTA per SM.

The occupancy API reports a resource-based upper bound; it does not inspect
where the scheduler places blocks during a launch. Likewise, launching
`grid_blocks = sm_count` does not by itself prove that every SM concurrently
receives one block. P1.1 therefore claims only
`max_active_ctas_per_sm = 1`, recorded in the frozen CSV column
`occupancy_ctas_per_sm`, and makes no stronger placement claim.

### Working set and rotation

- `l2CacheSize` and `multiProcessorCount` are queried at runtime, never
  hardcoded for a specific B200/GB300 SKU.
- The working set is rounded up to a common multiple of
  `sm_count * 32 KiB`, which guarantees it is evenly divisible by every
  possible `stage_bytes` value (2/4/8/16/32 KiB all divide 32 KiB) and that
  every specialization sees the exact same working set.
- `--working-set-mib` requests a size in MiB; if omitted, the default is at
  least 4x the queried L2 size. Both `requested_working_set_bytes` (before
  rounding) and `working_set_bytes` (after rounding) are recorded in the
  CSV. `run_kind=benchmark` rejects a rounded working set that is not
  strictly larger than 2xL2; `run_kind=smoke` has no such requirement.
- Each CTA owns a disjoint, contiguous slice of the working set
  (`per_cta_bytes = working_set_bytes / sm_count`, split into whole
  `stage_bytes` tiles); the grid covers the whole working set exactly once
  per pass, with no overlap and no out-of-bounds access.
- `useful_bytes = working_set_bytes * passes`. Each measured repetition is
  one kernel launch that internally performs `passes` full traversals; the
  starting tile rotates (in whole-tile units, always inside the CTA's own
  region) both between passes within a launch and between repetitions, so
  the same tile is not always issued first.
- A large working set reduces artificial L2 reuse but does not prove all
  traffic came from HBM — that is an NCU question for P1.4, not something
  P1.1 claims.

### Correctness before timing

Both kernels share one `__device__` helper (`emit_stage_cp_async`) that
issues the actual `cp.async.cg.shared.global` instructions, so the timed
path runs the identical PTX sequence the validated path already proved
correct:

- **Validation** initializes the working set on-device with a deterministic
  pattern that mixes all 64 bits of each 16-byte vector's global index
  (`expected_vector()`), runs the same pipeline once over the whole working
  set, and compares every copied vector — not a sample — against the
  recomputed expected value. Mismatches accumulate in per-thread registers
  and reach the host as a single `atomicAdd`'d counter, so the working set
  never has to travel back to the host. Any mismatch, or any CUDA error before
  warm-up, prevents timing; later CUDA errors are reported separately from
  correctness failures and make any partial CSV invalid.
- **Benchmark** only runs after that exact configuration's validation
  passed. It contains only pipeline issue/wait/rotate logic plus a minimal
  sink: each thread XORs one 32-bit lane per consumed slot into a
  per-thread sink buffer (`grid_blocks * 128 * 4` bytes, independent of the
  working set) so the compiler cannot discard the copies; the sink's own
  traffic is excluded from `useful_bytes` / `effective_gbps`.
- `--self-test` runs the validation path only, for all nine specializations,
  on a small fixed working set (`sm_count * 32 KiB * 8`); it never touches
  timing or CSV output. Every normal (non-self-test) invocation still
  validates its one selected configuration, over the actual configured
  working set, before any timing happens.

### Timing

CUDA events, not CPU timers, bound each measured kernel launch. Warm-up
repeatedly launches the benchmark kernel (synchronizing after each launch)
until at least `--warmup-ms` of wall-clock time has elapsed. Each
`--repetitions` measured sample is a separate kernel launch with its own
`cudaEventElapsedTime`; all individual repetition times are preserved as
separate CSV rows — no in-binary median/mean. `effective_gbps` is decimal
GB/s: `useful_bytes / kernel_time_seconds / 1e9`.

## Build, SASS, self-test, and smoke

```bash
make check-static             # no Docker, no GPU, no network
make memory-ldgsts-build       # compile inside the pinned image; no GPU
make memory-ldgsts-sass        # verify complete 16B groups + commit/wait SASS; no GPU

# GPU-executing targets require an explicit, operator-provided physical
# index and go exclusively through scripts/run_container.sh:
BLACKWELL_GPU_INDEX=<physical-index> make memory-ldgsts-self-test
BLACKWELL_GPU_INDEX=<physical-index> make memory-ldgsts-smoke
```

Direct CLI usage (inside the container, e.g. via
`scripts/run_container.sh build/memory/ldgsts ...`):

```bash
build/memory/ldgsts --help
build/memory/ldgsts --self-test
build/memory/ldgsts --stages 4 --bytes-in-flight-kib 32 --run-kind benchmark \
    --working-set-mib 512 --passes 4 --warmup-ms 200 --repetitions 20
```

## P1.2 — standalone 2D unicast TMA path

This is the TMA arm of the same experiment. It measures the effective copy
bandwidth of a 2D unicast Tensor Memory Accelerator pipeline
(`cp.async.bulk.tensor.2d.shared::cta.global` with mbarrier transaction
completion) moving the exact same logical tiles as P1.1, from a host-encoded
rank-2 `CUtensorMap` descriptor, into the same per-SM shared-memory ring
buffer shape.

**Status: implemented, pending audit and GB300 verification (see
`PLAN.md`).** No experimental numbers from this code have been published in
`README.md` or anywhere else in the repository.

### What P1.2 measures

- The *effective copy bandwidth* (`effective_gbps`) of a 2D unicast TMA
  pipeline for each of the same nine frozen `(stages, bytes_in_flight_per_sm)`
  specializations as P1.1.
- Whether every copied 16-byte vector lands in shared memory with the exact
  bytes the same deterministic source pattern predicts (`correctness`,
  `mismatches`), for all nine specializations under `--self-test`.
- The same `occupancy_ctas_per_sm = 1` residency cap as P1.1, verified with
  the occupancy API for both the validate and benchmark kernels.

### What it cannot yet claim

Identical caveats to P1.1: not HBM/DRAM bandwidth (`effective_gbps` is
*effective copy bandwidth*, not a claim about L2 vs DRAM traffic — that is
an NCU question for P1.4); not a comparison against LDGSTS (P1.3 is the
joint sweep; it has not started); not a sweep or an analysis (this binary
runs one specialization, or under `--self-test` validates all nine, per
invocation); not a final result (`run_kind=smoke` is a functional check
only, never a publishable measurement).

### Frozen contract

- Method: `tma`. PTX: `cp.async.bulk.tensor.2d.shared::cta.global...
  mbarrier::complete_tx::bytes`, issued through the CUDA 13.1 `cuda::ptx`
  low-level wrappers (`cp_async_bulk_tensor`, `mbarrier_arrive_expect_tx`,
  `mbarrier_try_wait_parity`, `elect_sync`).
- Same threads/CTA, grid, occupancy cap, stages, and bytes-in-flight sets as
  P1.1: 128 threads/CTA, grid = SM count, maximum active residency = 1
  CTA/SM, stages ∈ {2, 4, 8}, bytes-in-flight/SM ∈ {16, 32, 64} KiB.
- One pipeline stage = one 2D TMA tile transfer of exactly `stage_bytes`,
  tracked by one distinct shared-memory mbarrier with explicit phase/parity
  reuse (`mbarrier.try_wait.parity`), not `cp.async`'s commit/wait-group
  counters.
- **Single geometry source of truth.** `stage_bytes`, and therefore
  `tile_height = stage_bytes / 256`, are computed once
  (`compute_stage_bytes()` / `compute_tile_height()` in `src/memory/tma.cu`)
  and reused everywhere that value is needed: the host-side `Specialization`
  table, the `CUtensorMap` descriptor's `boxDim[1]`, each kernel's Y-coordinate
  stride, the expected-source-index computation, the shared-memory payload
  sizing, and a dispatch-time runtime check that the caller-selected
  `Specialization` matches the `<STAGES, COPIES>` template it is about to
  launch. `static_assert`s pin all five distinct `COPIES` values
  (1, 2, 4, 8, 16) and the full nine-specialization table to their frozen
  expected numbers, so a regression to an independently-computed (and
  therefore driftable) tile height fails to compile.

#### Bytes-in-flight formulas

```
stage_bytes                 = bytes_in_flight_per_sm / stages
copies_per_thread_per_stage = stage_bytes / (128 threads * 16 bytes)
bytes_in_flight_per_sm      = stages * stage_bytes
tile_height                 = stage_bytes / 256
```

Identical formulas and identical resulting nine-row table to P1.1's (see
above); `copies_per_thread_per_stage` and `vector_bytes=16` describe the
logical LDGSTS-equivalent tile decomposition retained for CSV/validation
compatibility — TMA itself issues exactly one 2D tile operation per stage,
not one operation per 16-byte vector.

### Tensor map descriptor

A host-encoded rank-2 `CUtensorMap`, built once per specialization via
`cuTensorMapEncodeTiled` (obtained at run time through
`cudaGetDriverEntryPointByVersion`, so the GPU-free build never links a
driver stub):

| Field | Value |
| --- | --- |
| Element type | `CU_TENSOR_MAP_DATA_TYPE_UINT16` (opaque BF16-width storage; no arithmetic) |
| Rank | 2 |
| `globalDim[0]` | 128 elements |
| `globalDim[1]` | `working_set_bytes / 256` |
| `globalStrides[0]` | 256 bytes (no padding between rows) |
| `boxDim[0]` | 128 elements |
| `boxDim[1]` | `tile_height` (the corrected, shared-geometry value) |
| `elementStrides` | `{1, 1}` |
| Interleave / swizzle / L2 promotion / OOB fill | all `NONE` |

Coordinates are `{x = 0, y = global_tile_index * tile_height}`
(fastest-moving dimension first). The descriptor is passed to both kernel
templates as a `const __grid_constant__ CUtensorMap` parameter.

### Mbarrier pipeline: issue, wait, and invalidation

Both kernel templates share the same TMA issue and synchronization helpers
(`tma_elect_leader`, `tma_init_barriers`, `tma_issue_stage`,
`tma_wait_stage`, `tma_invalidate_barriers`), so the timed benchmark path is
the one `--self-test` validates:

- **Issue.** Exactly one compiler-elected thread in warp 0 (`elect.sync`)
  registers the expected transaction byte count
  (`mbarrier.arrive.expect_tx`) and issues the 2D TMA tile transfer
  (`cp.async.bulk.tensor.2d...mbarrier::complete_tx::bytes`) for one ring
  slot.
- **Wait.** Every thread independently spins
  `mbarrier.try_wait.parity` on that slot's mbarrier, tracking its own
  per-slot phase parity; a successful wait already guarantees full
  visibility of the TMA-written bytes to that thread.
- **Pipeline shape.** Fill `STAGES` distinct ring slots before the first
  wait, wait for the oldest slot, consume it, synchronize the CTA before
  reusing that slot, refill it on the next iteration, and drain every
  outstanding stage after the main loop (or after the last pass, for the
  benchmark kernel's per-launch pass loop).
- **Invalidation.** After the drain loop's final `__syncthreads()` — i.e.
  only once every outstanding TMA transaction on every stage has completed
  and every consumer has finished reading its payload — the same elected
  leader thread invalidates all `STAGES` mbarriers
  (`mbarrier.inval.shared.b64`, sm_103a SASS: `SYNCS.CCTL.IV`, one
  instruction per ring slot), followed by one more `__syncthreads()` so the
  invalidation is visible to every thread before kernel exit. This never
  races a pending transaction: invalidation only ever follows a full drain.

### Build, SASS, self-test, and smoke

```bash
make check-static           # no Docker, no GPU, no network
make memory-tma-build       # compile inside the pinned image; no GPU
make memory-tma-sass        # verify 2D unicast TMA loads, transaction-barrier
                             # completion, and full mbarrier invalidation; no GPU

# GPU-executing targets require an explicit, operator-provided physical
# index and go exclusively through scripts/run_container.sh:
BLACKWELL_GPU_INDEX=<physical-index> make memory-tma-self-test
BLACKWELL_GPU_INDEX=<physical-index> make memory-tma-smoke
```

Direct CLI usage (inside the container, e.g. via
`scripts/run_container.sh build/memory/tma ...`):

```bash
build/memory/tma --help
build/memory/tma --self-test
build/memory/tma --stages 4 --bytes-in-flight-kib 32 --run-kind benchmark \
    --working-set-mib 512 --passes 4 --warmup-ms 200 --repetitions 20
```

## LDGSTS/TMA equivalence table

| Dimension | P1.1 LDGSTS | P1.2 TMA |
| --- | --- | --- |
| Threads/CTA, grid | 128 threads/CTA, grid = SM count | Identical |
| Occupancy cap | `occupancy_ctas_per_sm = 1`, enforced identically | Identical |
| Stages | `{2, 4, 8}` | Identical |
| Bytes in flight/SM | `{16, 32, 64}` KiB | Identical |
| Stage bytes | `bytes_in_flight_per_sm / stages` | Identical formula |
| Tile width | 128 BF16-width elements = 256 bytes | Identical |
| Tile height | `stage_bytes / 256` | Identical formula, single shared source of truth (see "Frozen contract" above) |
| Working-set partition | Disjoint per-CTA slice, `sm_count*32KiB` common multiple, `>2xL2` for `benchmark` | Identical |
| Passes / rotation | `useful_bytes = working_set_bytes * passes`; starting tile rotates between passes and repetitions | Identical |
| Timing | CUDA events around one kernel launch per CSV sample; warm-up outside timed events | Identical |
| Correctness | Full-working-set validation before any timing; device-side mismatch accumulation; no timed run after a mismatch | Identical |
| CSV schema | `schema_version="1"`, `method=ldgsts` | Same schema, `method=tma`; `vector_bytes`/`copies_per_thread_per_stage` describe the logical LDGSTS-equivalent decomposition, not TMA's issuing granularity |
| Issuing mechanism | Every thread issues its own 16-byte `cp.async.cg.shared.global` copies | One elected thread issues one 2D tile TMA operation per stage |
| Completion mechanism | `cp.async.commit_group` / `cp.async.wait_group<N>` per-thread counters | One mbarrier per ring slot; `mbarrier.arrive.expect_tx` + `mbarrier.try_wait.parity`, explicit phase/parity tracking, explicit `mbarrier.inval` after drain |

## CSV schema

Emitted only for `--stages`/`--bytes-in-flight-kib`/`--run-kind` invocations
(never for `--self-test`), one header line plus one row per repetition, on
stdout; everything else goes to stderr.

| Column | Unit / format | Meaning |
| --- | --- | --- |
| `schema_version` | string | CSV schema version (`"1"`), shared by both `build/memory/ldgsts` and `build/memory/tma`. |
| `timestamp_utc` | ISO 8601, UTC | Wall-clock time the row was produced. |
| `run_kind` | `smoke` \| `benchmark` | As passed on the CLI. |
| `method` | string | `ldgsts` from `build/memory/ldgsts`; `tma` from `build/memory/tma`. |
| `sample_index` | integer | 0-based repetition index within this run. |
| `stages` | integer | Ring buffer depth (2, 4, or 8). |
| `tile_width_elements` | BF16-width elements | Always 128. |
| `tile_width_bytes` | bytes | Always 256. |
| `tile_height` | rows | `stage_bytes / 256`. |
| `stage_bytes` | bytes | One pipeline stage's transfer size. |
| `bytes_in_flight_per_sm` | bytes | `stages * stage_bytes`. |
| `vector_bytes` | bytes | Always 16 (the `cp.async.cg` fixed vector width for `ldgsts`; the logical LDGSTS-equivalent decomposition unit for `tma`, not its issuing granularity). |
| `copies_per_thread_per_stage` | count | `stage_bytes / (128 * 16)`. For `tma`, describes the logical tile decomposition used for validation/comparison; TMA itself issues one 2D tile operation per stage, not one operation per vector. |
| `threads_per_cta` | count | Always 128. |
| `target_ctas_per_sm` | count | Always 1: the frozen target for maximum active residency. |
| `occupancy_ctas_per_sm` | count | Maximum active blocks/SM reported by `cudaOccupancyMaxActiveBlocksPerMultiprocessor`; always 1 or the run aborts. This is not an observed block-placement count. |
| `grid_blocks` | count | Equals `sm_count`. |
| `sm_count` | count | Queried `multiProcessorCount`. |
| `smem_reservation_bytes` | bytes | Actual dynamic shared memory reserved (> half of `sharedMemPerMultiprocessor`, includes padding beyond `bytes_in_flight_per_sm`; for `tma`, also includes mbarrier storage). |
| `l2_bytes` | bytes | Queried `l2CacheSize`. |
| `requested_working_set_bytes` | bytes | Before rounding (CLI value or the 4xL2 default). |
| `working_set_bytes` | bytes | After rounding to the `sm_count*32KiB` common multiple. |
| `working_set_l2_ratio` | ratio | `working_set_bytes / l2_bytes`. |
| `passes` | count | Full working-set traversals per measured kernel launch. |
| `useful_bytes` | bytes | `working_set_bytes * passes`; excludes sink/init/validation traffic. |
| `warmup_ms` | milliseconds | Requested warm-up floor. |
| `kernel_time_ms` | milliseconds | CUDA-event-measured time for this one repetition's kernel launch. |
| `effective_gbps` | decimal GB/s | `useful_bytes / (kernel_time_ms/1000) / 1e9`. Effective copy bandwidth — **not** HBM/DRAM bandwidth. |
| `correctness` | `OK` \| `MISMATCH` | Result of the one-time pre-timing validation for this configuration (repeated on every row of the run; the benchmark never runs after a `MISMATCH`). |
| `mismatches` | count | Mismatch count from that same one-time validation. |
| `gpu_name` | string (CSV-quoted) | `cudaDeviceProp.name`. |
| `gpu_uuid` | string | `EXPECTED_GPU_UUID`, as set by `scripts/run_container.sh`; no other environment variable is read. |
| `compute_capability` | `MAJOR.MINOR` | Always `10.3` (enforced at startup). |
| `cuda_driver_version` | integer | Raw `cudaDriverGetVersion` value (`MAJOR*1000 + MINOR*10`). |
| `cuda_runtime_version` | integer | Raw `cudaRuntimeGetVersion` value, same encoding. |
| `git_commit` | hex string \| `UNKNOWN` | `git rev-parse HEAD` at run time. |
| `git_dirty` | `true` \| `false` \| `unknown` | Whether `git status --porcelain` reported any changes. |
