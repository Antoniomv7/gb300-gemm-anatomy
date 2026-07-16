# results/

There are **no results yet**. This directory currently contains only this
policy file.

## Future contents

### `results/preflight/<UTC timestamp>/` (raw, not committed)

Each preflight run (`BLACKWELL_GPU_INDEX=<i> make preflight`) will create one
directory named with a UTC timestamp in `YYYYMMDDTHHMMSSZ` format, containing:

- `summary.json` — machine-readable summary (`schema_version`,
  `timestamp_utc`, `git_commit`, `git_dirty`, `host_arch`, `tool_versions`,
  allowlisted `gpu` fields, per-check statuses, `overall_status`).
- Per-check logs (compiler output, smoke-test output, `ncu` output).
- The compiled smoke binary and the `.ncu-rep` profile.

`results/preflight/` is ignored by Git: raw and temporary output is never
committed.

## Safe public metadata

Anything stored here must contain only allowlisted device and tool data: GPU
index, name, UUID, driver version, compute capability, memory size, tool
versions, and check outcomes. Never store secrets, credentials, SSH material,
usernames, home paths, full environment dumps, or unrelated host metadata.

## Selected processed results (committed deliberately)

Small, curated, secret-free processed result files (e.g. per-experiment CSV or
JSON summary tables produced by later phases) may be committed under future
`results/` subdirectories so they remain publishable with the thesis. This is
always a deliberate, reviewed action — never an automatic copy of raw output.
CSV/JSON files are intentionally not blanket-ignored for this reason.

## Naming

All timestamps in file and directory names are UTC (`YYYYMMDDTHHMMSSZ`).
