# src/memory — P1.1 standalone LDGSTS baseline

This is the LDGSTS arm of the "LDGSTS versus TMA" experiment (experiment 1 in
`AGENTS.md`). It measures the effective copy bandwidth of a vectorized
`cp.async.cg.shared.global` (LDGSTS) software pipeline from global memory to
shared memory. Dynamic shared memory caps the maximum active residency at one
CTA per SM; it does not observe or guarantee runtime block placement.

**Status: implemented, pending audit and GB300 verification (see `PLAN.md`).**
No experimental numbers from this code have been published in `README.md` or
anywhere else in the repository.

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
- **Not a TMA comparison.** TMA / `cp.async.bulk` is P1.2. The 2D tile
  layout here is deliberately chosen so P1.2 can express the identical
  tiles, but no TMA code exists yet.
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
make memory-ldgsts-sass        # verify LDGSTS counts + commit/wait SASS; no GPU

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

## CSV schema

Emitted only for `--stages`/`--bytes-in-flight-kib`/`--run-kind` invocations
(never for `--self-test`), one header line plus one row per repetition, on
stdout; everything else goes to stderr.

| Column | Unit / format | Meaning |
| --- | --- | --- |
| `schema_version` | string | CSV schema version (`"1"`). P1.2 reuses this schema. |
| `timestamp_utc` | ISO 8601, UTC | Wall-clock time the row was produced. |
| `run_kind` | `smoke` \| `benchmark` | As passed on the CLI. |
| `method` | string | Always `ldgsts` in this file; P1.2 will emit `tma`. |
| `sample_index` | integer | 0-based repetition index within this run. |
| `stages` | integer | Ring buffer depth (2, 4, or 8). |
| `tile_width_elements` | BF16-width elements | Always 128. |
| `tile_width_bytes` | bytes | Always 256. |
| `tile_height` | rows | `stage_bytes / 256`. |
| `stage_bytes` | bytes | One pipeline stage's transfer size. |
| `bytes_in_flight_per_sm` | bytes | `stages * stage_bytes`. |
| `vector_bytes` | bytes | Always 16 (the `cp.async.cg` fixed vector width). |
| `copies_per_thread_per_stage` | count | `stage_bytes / (128 * 16)`. |
| `threads_per_cta` | count | Always 128. |
| `target_ctas_per_sm` | count | Always 1: the frozen target for maximum active residency. |
| `occupancy_ctas_per_sm` | count | Maximum active blocks/SM reported by `cudaOccupancyMaxActiveBlocksPerMultiprocessor`; always 1 or the run aborts. This is not an observed block-placement count. |
| `grid_blocks` | count | Equals `sm_count`. |
| `sm_count` | count | Queried `multiProcessorCount`. |
| `smem_reservation_bytes` | bytes | Actual dynamic shared memory reserved (> half of `sharedMemPerMultiprocessor`, includes padding beyond `bytes_in_flight_per_sm`). |
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
