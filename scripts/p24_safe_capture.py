#!/usr/bin/env python3
"""P2.4-only descriptor-anchored safe command capture.

Adapts the audited P1.4 design (``scripts/p14_safe_capture.py``,
``src/memory/P1_4_PROTOCOL.md`` section 4a) for the P2.4 raw tree
(``results/raw/exp02_umma_throughput_p24/``). Every raw-campaign write the
P2.4 wrapper performs -- the NCU-help-capability-probe log, the
metric-discovery logs, each case's captured NCU-bridge bundle and bridge
stderr, and (after decoding) the seven per-case artifacts the bundle
carries -- goes exclusively through this module, never a plain
``>``/``>>``/``2>``/``2>>`` shell redirection into the raw tree. Every
directory component from the repository root down to ``logs/`` or
``profiles/<case>/`` is opened exactly once with Linux no-follow semantics
(``os.open(..., dir_fd=parent_fd)`` with ``O_DIRECTORY | O_NOFOLLOW``); the
resulting descriptor is used for every subsequent operation, so nothing that
happens to the *name* afterward (a symlink swap of the directory itself, or
of the output name) can redirect a later operation elsewhere -- the output
name is also created via ``O_EXCL | O_NOFOLLOW`` and only ever published by
an in-directory hard link, never ``os.replace()``, never following a
symlink.

This module never touches Docker, CUDA, NCU, ``nvidia-smi``, or GPU
hardware itself: it only orchestrates already-decided argv vectors
(``shell=False``) and safely captures their stdout/stderr. It never modifies
``scripts/aggregate_exp02_umma_throughput.py`` (P2.3, frozen) or
``scripts/analyze_exp02_umma_throughput_p24.py`` and is not imported by
either.

Every filename this module accepts from a caller (``--stdout-name``,
``--stderr-name``, ``write``/``publish-bundle``'s ``--name``/``--names``,
and ``--bundle-name``) is validated as a strict single-component basename
before anything is created or any child is launched.

Subcommands:
  run             Execute ARGV (after "--", shell=False) with stdout/stderr
                  connected directly to newly created, descriptor-anchored,
                  exclusive, no-follow partial files under
                  <campaign-dir>/<rel-dir>/; on a zero exit, publishes each
                  partial to its final name via no-clobber hard link; on a
                  non-zero exit -- or any failure before or during the
                  child's own launch -- preserves a non-empty partial under
                  its unique name and removes only an empty owned partial.
  write           Reads bytes from stdin and safely publishes them as one
                  new, no-clobber file under <campaign-dir>/<rel-dir>/.
  verify          Confirms that one or more names already exist as genuine
                  non-symlink, non-empty regular files strictly within
                  <campaign-dir>/<rel-dir>/.
  mkdir-case      Safely creates exactly one of the 24 frozen
                  profiles/<case>/ directories via mkdirat()'s own EEXIST.
  publish-bundle  Decodes an already-captured scripts/p24_ncu_bridge.py
                  bundle (itself captured via a prior "run --stdout-name")
                  and republishes its seven fixed-order segments under
                  caller-given names, no-clobber, then removes the raw
                  transport bundle.

Exit codes: 0 success; the child's own exit code for a "run" whose command
ran but exited non-zero; 1 for "write"/"verify"/"publish-bundle"
content/evidence failures; 2 for a path-safety, argument, or
capture-mechanism failure.
"""

from __future__ import annotations

import argparse
import os
import re
import stat
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Callable, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import aggregate_exp02_umma_throughput as p23  # noqa: E402
import analyze_exp02_umma_throughput_p24 as p24  # noqa: E402


class UnsafeCaptureError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Strict single-component basename validation. Every filename this module
# accepts from a caller must be a single path component: os.link()'s
# dst_dir_fd-relative resolution otherwise walks a multi-component relative
# name (including one containing "..") exactly like any other relative path,
# and ignores dst_dir_fd entirely for an absolute newpath.
# ---------------------------------------------------------------------------
_BASENAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _validate_basename(name: str, *, what: str) -> None:
    if not isinstance(name, str) or not name:
        raise UnsafeCaptureError(f"{what}: must be a non-empty string, got {name!r}")
    if "\x00" in name or any(ord(ch) < 0x20 for ch in name):
        raise UnsafeCaptureError(f"{what}: must not contain NUL or control characters, got {name!r}")
    if os.path.isabs(name):
        raise UnsafeCaptureError(f"{what}: must not be an absolute path, got {name!r}")
    if name in (".", ".."):
        raise UnsafeCaptureError(f"{what}: must not be '.' or '..', got {name!r}")
    if Path(name).parts != (name,):
        raise UnsafeCaptureError(f"{what}: must be a single path component, not {name!r}")
    if not _BASENAME_RE.fullmatch(name):
        raise UnsafeCaptureError(
            f"{what}: must match {_BASENAME_RE.pattern} (start with an alphanumeric character; "
            f"only alphanumerics, '.', '_', '-' afterward), got {name!r}"
        )


# ---------------------------------------------------------------------------
# Descriptor-anchored directory resolution. Never trusts an earlier
# validation of the same pathname: every call re-opens every component
# itself, one at a time, with O_NOFOLLOW.
# ---------------------------------------------------------------------------
def _rel_dir_allowlist() -> tuple[str, ...]:
    plan = p24.build_profile_plan()
    return ("logs",) + tuple(f"profiles/{entry['case_name']}" for entry in plan)


def _validate_campaign_dir_rel(campaign_dir_rel: str) -> tuple[str, ...]:
    if os.path.isabs(campaign_dir_rel):
        raise UnsafeCaptureError(f"--campaign-dir must be relative, got absolute path {campaign_dir_rel!r}")
    parts = Path(campaign_dir_rel).parts
    if any(".." in part for part in parts):
        raise UnsafeCaptureError(f"--campaign-dir must not contain '..': {campaign_dir_rel!r}")
    if (
        len(parts) != len(p24.RAW_ROOT_PARTS_P24) + 1
        or tuple(parts[: len(p24.RAW_ROOT_PARTS_P24)]) != p24.RAW_ROOT_PARTS_P24
    ):
        raise UnsafeCaptureError(
            f"--campaign-dir must be exactly {'/'.join(p24.RAW_ROOT_PARTS_P24)}/<campaign_id>, got {campaign_dir_rel!r}"
        )
    try:
        p24.validate_p24_campaign_id(parts[-1])
    except p23.UnsafePathError as exc:
        raise UnsafeCaptureError(str(exc)) from exc
    return parts


def _validate_rel_dir(rel_dir: str) -> tuple[str, ...]:
    allowed = _rel_dir_allowlist()
    if rel_dir not in allowed:
        raise UnsafeCaptureError(
            f"--rel-dir must be one of {sorted(allowed)!r} (the frozen logs/ directory or one of the "
            f"24 canonical profiles/<case> directories), got {rel_dir!r}"
        )
    parts = Path(rel_dir).parts
    if any(".." in part for part in parts):
        raise UnsafeCaptureError(f"--rel-dir must not contain '..': {rel_dir!r}")
    return parts


def _open_dir_nofollow(name: str, *, dir_fd: int | None) -> int:
    flags = os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        raise UnsafeCaptureError(f"{name}: cannot open as a non-symlink directory: {exc}") from exc


def _open_repo_relative_dir_fd(*parts: str) -> int:
    root = str(p24.REPO_ROOT)
    fd = _open_dir_nofollow(root, dir_fd=None)
    try:
        for part in parts:
            next_fd = _open_dir_nofollow(part, dir_fd=fd)
            os.close(fd)
            fd = next_fd
    except Exception:
        os.close(fd)
        raise
    return fd


def resolve_campaign_rel_dir_fd(campaign_dir_rel: str, rel_dir: str) -> int:
    campaign_parts = _validate_campaign_dir_rel(campaign_dir_rel)
    rel_parts = _validate_rel_dir(rel_dir)
    return _open_repo_relative_dir_fd(*campaign_parts, *rel_parts)


def resolve_profiles_root_fd(campaign_dir_rel: str) -> int:
    campaign_parts = _validate_campaign_dir_rel(campaign_dir_rel)
    return _open_repo_relative_dir_fd(*campaign_parts, "profiles")


def mkdir_case_dir(profiles_dir_fd: int, case_name: str) -> None:
    valid_names = {entry["case_name"] for entry in p24.build_profile_plan()}
    if case_name not in valid_names:
        raise UnsafeCaptureError(f"case_name must be one of the frozen case names, got {case_name!r}")
    try:
        os.mkdir(case_name, 0o700, dir_fd=profiles_dir_fd)
    except FileExistsError as exc:
        raise UnsafeCaptureError(f"{case_name}: profile case directory already exists") from exc
    except OSError as exc:
        raise UnsafeCaptureError(f"{case_name}: cannot create profile case directory: {exc}") from exc


# ---------------------------------------------------------------------------
# Exclusive-create / publish / cleanup primitives, all dir_fd-relative.
# ---------------------------------------------------------------------------
def _partial_name(final_name: str) -> str:
    return f".{final_name}.p24capture-{uuid.uuid4().hex}.partial"


def create_partial(dir_fd: int, final_name: str) -> tuple[int, str]:
    _validate_basename(final_name, what="output name")
    partial = _partial_name(final_name)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(partial, flags, 0o600, dir_fd=dir_fd)
    except OSError as exc:
        raise UnsafeCaptureError(f"{final_name}: cannot create owned partial output: {exc}") from exc
    return fd, partial


def publish_no_clobber(dir_fd: int, partial_name: str, final_name: str) -> None:
    _validate_basename(final_name, what="output name")
    try:
        os.link(partial_name, final_name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except FileExistsError as exc:
        raise UnsafeCaptureError(f"{final_name}: already exists; refusing to overwrite") from exc
    except OSError as exc:
        raise UnsafeCaptureError(f"{final_name}: could not publish without clobbering: {exc}") from exc
    os.unlink(partial_name, dir_fd=dir_fd)


def discard_if_empty_owned(dir_fd: int, partial_name: str, partial_stat: os.stat_result) -> bool:
    if not stat.S_ISREG(partial_stat.st_mode) or partial_stat.st_size != 0:
        return False
    st_name = os.stat(partial_name, dir_fd=dir_fd, follow_symlinks=False)
    if (st_name.st_dev, st_name.st_ino) != (partial_stat.st_dev, partial_stat.st_ino):
        raise UnsafeCaptureError(f"{partial_name}: changed identity before cleanup; refusing to unlink")
    os.unlink(partial_name, dir_fd=dir_fd)
    return True


def unlink_if_same_owned_inode(dir_fd: int, name: str, expected_stat: os.stat_result) -> bool:
    """Unlinks `name`, strictly within dir_fd and without ever following a
    symlink, only if a fresh, non-following stat still shows the exact same
    (device, inode) and regular-file type as expected_stat -- i.e. only if
    the name still refers to the very entry this call site itself is
    responsible for. A missing name is treated as already absent (returns
    False). A name that currently identifies something else is left
    completely untouched and reported via a raised UnsafeCaptureError."""
    try:
        current = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(expected_stat.st_mode):
        raise UnsafeCaptureError(f"{name}: expected_stat is not a regular file; refusing to unlink anything")
    if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
        raise UnsafeCaptureError(f"{name}: no longer a regular file; the entry at this name was replaced -- leaving it untouched")
    if (current.st_dev, current.st_ino) != (expected_stat.st_dev, expected_stat.st_ino):
        raise UnsafeCaptureError(f"{name}: identity changed since it was last trusted; a different file now occupies this name -- leaving it untouched")
    os.unlink(name, dir_fd=dir_fd)
    return True


def verify_regular_file_in_dir(dir_fd: int, name: str) -> str | None:
    try:
        _validate_basename(name, what="name")
    except UnsafeCaptureError as exc:
        return str(exc)
    try:
        st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    except FileNotFoundError:
        return f"{name}: does not exist in the anchored directory"
    except OSError as exc:
        return f"{name}: cannot stat in the anchored directory: {exc}"
    if stat.S_ISLNK(st.st_mode):
        return f"{name}: is a symlink; refusing"
    if not stat.S_ISREG(st.st_mode):
        return f"{name}: is not a regular file"
    if st.st_size == 0:
        return f"{name}: is empty"
    return None


# ---------------------------------------------------------------------------
# NCU bundle format. scripts/p24_ncu_bridge.py runs entirely inside the
# container's own private, non-host-mounted /tmp and never receives a
# campaign-relative pathname; it hands its results back to the host as one
# versioned, length-delimited bundle on its own stdout, captured here (via
# the ordinary "run" subcommand) into an anchored partial exactly like any
# other P2.4 child stdout. "publish-bundle" then decodes that already-
# anchored file and republishes its six segments under their real names via
# the same no-clobber primitives every other P2.4 artifact uses. The bash
# wrapper separately extracts <case>.application.csv from the published
# container_stdout.log (the profiled binary's own inherited stdout).
#
# Length-prefixed, not delimiter-based, so arbitrary binary content (the
# .ncu-rep bytes, or a metrics CSV containing any byte sequence) can never be
# misparsed. Decoding tolerates leading bytes before the magic marker:
# scripts/run_container.sh prints allowlisted banner lines to its own stdout
# before exec'ing the container command.
# ---------------------------------------------------------------------------
NCU_BUNDLE_MAGIC = b"P24NCUBUNDLE1\n"
NCU_BUNDLE_SEGMENT_NAMES = (
    "app_stdout", "app_stderr", "ncu_tool_log", "ncu_rep", "metrics_csv", "metrics_export_stderr",
)
_NCU_BUNDLE_LENGTH_WIDTH = 20


class NcuBundleParseError(ValueError):
    pass


def encode_ncu_bundle(segments: dict[str, bytes]) -> bytes:
    missing = [name for name in NCU_BUNDLE_SEGMENT_NAMES if name not in segments]
    if missing:
        raise ValueError(f"encode_ncu_bundle: missing segment(s): {missing}")
    parts = [NCU_BUNDLE_MAGIC]
    for name in NCU_BUNDLE_SEGMENT_NAMES:
        data = segments[name]
        if not isinstance(data, (bytes, bytearray)):
            raise ValueError(f"encode_ncu_bundle: segment {name!r} must be bytes, got {type(data).__name__}")
        parts.append(f"{len(data):0{_NCU_BUNDLE_LENGTH_WIDTH}d}\n".encode("ascii"))
        parts.append(bytes(data))
    return b"".join(parts)


def decode_ncu_bundle(raw: bytes) -> dict[str, bytes]:
    idx = raw.find(NCU_BUNDLE_MAGIC)
    if idx == -1:
        raise NcuBundleParseError("bundle magic marker not found (captured stream is not a valid NCU bundle)")
    pos = idx + len(NCU_BUNDLE_MAGIC)
    segments: dict[str, bytes] = {}
    for name in NCU_BUNDLE_SEGMENT_NAMES:
        header_end = pos + _NCU_BUNDLE_LENGTH_WIDTH
        if header_end + 1 > len(raw) or raw[header_end:header_end + 1] != b"\n":
            raise NcuBundleParseError(f"segment {name!r}: malformed or truncated length header at offset {pos}")
        length_field = raw[pos:header_end]
        if not length_field.isdigit():
            raise NcuBundleParseError(f"segment {name!r}: non-numeric length header {length_field!r}")
        length = int(length_field)
        data_start = header_end + 1
        data_end = data_start + length
        if data_end > len(raw):
            raise NcuBundleParseError(f"segment {name!r}: truncated (expected {length} bytes, only {len(raw) - data_start} available)")
        segments[name] = raw[data_start:data_end]
        pos = data_end
    return segments


def publish_ncu_bundle(*, campaign_dir_rel: str, rel_dir: str, bundle_name: str, output_names: Sequence[str]) -> None:
    _validate_basename(bundle_name, what="--bundle-name")
    if len(output_names) != len(NCU_BUNDLE_SEGMENT_NAMES):
        raise UnsafeCaptureError(
            f"--names requires exactly {len(NCU_BUNDLE_SEGMENT_NAMES)} value(s) in fixed order {NCU_BUNDLE_SEGMENT_NAMES!r}, got {len(output_names)}"
        )
    for output_name in output_names:
        _validate_basename(output_name, what="--names")

    dir_fd = resolve_campaign_rel_dir_fd(campaign_dir_rel, rel_dir)
    try:
        verify_err = verify_regular_file_in_dir(dir_fd, bundle_name)
        if verify_err:
            raise UnsafeCaptureError(f"{bundle_name}: {verify_err}")
        bundle_fd = os.open(bundle_name, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC, dir_fd=dir_fd)
        try:
            bundle_identity = os.fstat(bundle_fd)
            chunks = []
            while True:
                chunk = os.read(bundle_fd, 1 << 20)
                if not chunk:
                    break
                chunks.append(chunk)
            raw = b"".join(chunks)
        finally:
            os.close(bundle_fd)

        try:
            segments = decode_ncu_bundle(raw)
        except NcuBundleParseError as exc:
            raise UnsafeCaptureError(f"{bundle_name}: {exc}") from exc

        published: dict[str, os.stat_result] = {}
        try:
            for segment_name, output_name in zip(NCU_BUNDLE_SEGMENT_NAMES, output_names):
                partial_fd, partial = create_partial(dir_fd, output_name)
                handle = os.fdopen(partial_fd, "wb")
                write_ok = False
                try:
                    handle.write(segments[segment_name])
                    handle.flush()
                    write_ok = True
                finally:
                    partial_stat = os.fstat(partial_fd)
                    handle.close()
                if not write_ok:
                    try:
                        discard_if_empty_owned(dir_fd, partial, partial_stat)
                    except UnsafeCaptureError:
                        pass
                    raise UnsafeCaptureError(f"{output_name}: failed to write bundle segment {segment_name!r}")
                publish_no_clobber(dir_fd, partial, output_name)
                published[output_name] = partial_stat
        except Exception:
            for name, identity in published.items():
                try:
                    unlink_if_same_owned_inode(dir_fd, name, identity)
                except UnsafeCaptureError:
                    pass
            raise

        unlink_if_same_owned_inode(dir_fd, bundle_name, bundle_identity)
    finally:
        os.close(dir_fd)


def cmd_publish_bundle(args: argparse.Namespace) -> int:
    try:
        publish_ncu_bundle(campaign_dir_rel=args.campaign_dir, rel_dir=args.rel_dir, bundle_name=args.bundle_name, output_names=args.names)
    except UnsafeCaptureError as exc:
        print(f"p24_safe_capture: publish-bundle: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"p24_safe_capture: publish-bundle: OK: {args.rel_dir}/{{{','.join(args.names)}}}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------
def run_capturing_outputs(
    *, campaign_dir_rel: str, rel_dir: str, argv: Sequence[str], stdout_name: str | None, stderr_name: str | None,
    combine_stderr: bool, _test_hook_after_open: Callable[[int], None] | None = None,
) -> int:
    """Resolves <campaign_dir_rel>/<rel_dir>, then runs argv (shell=False)
    with stdout/stderr connected directly to open, already-created output
    descriptors. Returns the child's own exit code on a clean run; raises
    UnsafeCaptureError for any pre-launch or publish failure.

    _test_hook_after_open, if given, is called with the resolved directory
    fd immediately after every output descriptor has been created but
    before the child is launched -- self-test only, never a production CLI
    option."""
    if combine_stderr and stderr_name is not None:
        raise UnsafeCaptureError("--combine-stderr and --stderr-name are mutually exclusive")
    if combine_stderr and stdout_name is None:
        raise UnsafeCaptureError("--combine-stderr requires --stdout-name")
    if stdout_name is not None:
        _validate_basename(stdout_name, what="--stdout-name")
    if stderr_name is not None:
        _validate_basename(stderr_name, what="--stderr-name")
    if not argv:
        raise UnsafeCaptureError("no command given after '--'")

    dir_fd = resolve_campaign_rel_dir_fd(campaign_dir_rel, rel_dir)
    try:
        stdout_fd = stdout_partial = None
        stderr_fd = stderr_partial = None
        stdout_stat = stderr_stat = None
        result = None
        pre_launch_error: Exception | None = None
        try:
            if stdout_name is not None:
                stdout_fd, stdout_partial = create_partial(dir_fd, stdout_name)
            if combine_stderr:
                stderr_fd = stdout_fd
            elif stderr_name is not None:
                stderr_fd, stderr_partial = create_partial(dir_fd, stderr_name)

            if _test_hook_after_open is not None:
                _test_hook_after_open(dir_fd)

            result = subprocess.run(list(argv), stdout=stdout_fd, stderr=stderr_fd, stdin=subprocess.DEVNULL, shell=False)
        except OSError as exc:
            pre_launch_error = UnsafeCaptureError(f"could not launch command: {exc}")
        except UnsafeCaptureError as exc:
            pre_launch_error = exc
        finally:
            if stdout_fd is not None:
                stdout_stat = os.fstat(stdout_fd)
                os.close(stdout_fd)
            if stderr_fd is not None and not combine_stderr:
                stderr_stat = os.fstat(stderr_fd)
                os.close(stderr_fd)

        errors: list[str] = []
        if pre_launch_error is None and result.returncode == 0:
            if stdout_partial is not None:
                publish_no_clobber(dir_fd, stdout_partial, stdout_name)
            if stderr_partial is not None:
                publish_no_clobber(dir_fd, stderr_partial, stderr_name)
        else:
            for partial, partial_stat in ((stdout_partial, stdout_stat), (stderr_partial, stderr_stat)):
                if partial is None:
                    continue
                try:
                    discard_if_empty_owned(dir_fd, partial, partial_stat)
                except UnsafeCaptureError as exc:
                    errors.append(str(exc))
        if pre_launch_error is not None:
            if errors:
                raise UnsafeCaptureError(f"{pre_launch_error}; additionally: {'; '.join(errors)}")
            raise pre_launch_error
        if errors:
            raise UnsafeCaptureError("; ".join(errors))
        return result.returncode
    finally:
        os.close(dir_fd)


def _split_trailing_argv(raw: list[str]) -> list[str]:
    if raw and raw[0] == "--":
        return raw[1:]
    return raw


def cmd_run(args: argparse.Namespace) -> int:
    argv = _split_trailing_argv(args.command)
    try:
        rc = run_capturing_outputs(
            campaign_dir_rel=args.campaign_dir, rel_dir=args.rel_dir, argv=argv,
            stdout_name=args.stdout_name, stderr_name=args.stderr_name, combine_stderr=args.combine_stderr,
        )
    except UnsafeCaptureError as exc:
        print(f"p24_safe_capture: run: ERROR: {exc}", file=sys.stderr)
        return 2
    if rc != 0:
        print(f"p24_safe_capture: run: command exited {rc}", file=sys.stderr)
        return rc if 0 < rc < 126 else 1
    return 0


# ---------------------------------------------------------------------------
# Subcommand: write (stdin -> one new, no-clobber, anchored file)
# ---------------------------------------------------------------------------
def write_stdin_safely(*, campaign_dir_rel: str, rel_dir: str, name: str, content: bytes, allow_empty: bool = False) -> None:
    _validate_basename(name, what="--name")
    if not content and not allow_empty:
        raise UnsafeCaptureError(f"{name}: refusing to publish empty content")
    dir_fd = resolve_campaign_rel_dir_fd(campaign_dir_rel, rel_dir)
    try:
        fd, partial = create_partial(dir_fd, name)
        handle = os.fdopen(fd, "wb")
        write_ok = False
        try:
            handle.write(content)
            handle.flush()
            write_ok = True
        finally:
            partial_stat = os.fstat(fd)
            handle.close()
        if not write_ok:
            try:
                discard_if_empty_owned(dir_fd, partial, partial_stat)
            except UnsafeCaptureError:
                pass
            raise UnsafeCaptureError(f"{name}: failed to write content to partial output")
        publish_no_clobber(dir_fd, partial, name)
    finally:
        os.close(dir_fd)


def cmd_write(args: argparse.Namespace) -> int:
    content = sys.stdin.buffer.read()
    try:
        write_stdin_safely(campaign_dir_rel=args.campaign_dir, rel_dir=args.rel_dir, name=args.name, content=content)
    except UnsafeCaptureError as exc:
        print(f"p24_safe_capture: write: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"p24_safe_capture: write: OK: {args.rel_dir}/{args.name}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: verify
# ---------------------------------------------------------------------------
def cmd_verify(args: argparse.Namespace) -> int:
    try:
        dir_fd = resolve_campaign_rel_dir_fd(args.campaign_dir, args.rel_dir)
    except UnsafeCaptureError as exc:
        print(f"p24_safe_capture: verify: ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        errors = [err for name in args.name if (err := verify_regular_file_in_dir(dir_fd, name))]
    finally:
        os.close(dir_fd)
    if errors:
        for err in errors:
            print(f"p24_safe_capture: verify: ERROR: {err}", file=sys.stderr)
        return 1
    print(f"p24_safe_capture: verify: OK: {args.rel_dir}/{{{','.join(args.name)}}}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: mkdir-case
# ---------------------------------------------------------------------------
def cmd_mkdir_case(args: argparse.Namespace) -> int:
    try:
        fd = resolve_profiles_root_fd(args.campaign_dir)
        try:
            mkdir_case_dir(fd, args.case_name)
        finally:
            os.close(fd)
    except UnsafeCaptureError as exc:
        print(f"p24_safe_capture: mkdir-case: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"p24_safe_capture: mkdir-case: OK: profiles/{args.case_name}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# --self-test: GPU-free, Docker-free, fully isolated in a temporary
# directory standing in for REPO_ROOT. Never touches results/raw/.
# ---------------------------------------------------------------------------
def _fake_campaign_dir(tmp_path: Path, campaign_id: str = "20260804T230000Z") -> str:
    rel = tmp_path.joinpath(*p24.RAW_ROOT_PARTS_P24, campaign_id)
    (rel / "logs").mkdir(parents=True)
    case_name = p24.build_profile_plan()[0]["case_name"]
    (rel / "profiles" / case_name).mkdir(parents=True)
    return str(Path(*p24.RAW_ROOT_PARTS_P24) / campaign_id)


class _Recorder:
    def __init__(self) -> None:
        self.total = 0
        self.failures: list[str] = []

    def check(self, label: str, condition: bool, *, detail: str = "") -> None:
        self.total += 1
        if condition:
            print(f"p24_safe_capture: self-test: PASS: {label}", file=sys.stderr)
        else:
            print(f"p24_safe_capture: self-test: FAIL: {label}; {detail}", file=sys.stderr)
            self.failures.append(label)


def run_self_test() -> int:
    import shutil
    import tempfile
    from unittest import mock

    rec = _Recorder()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with mock.patch.object(p24, "REPO_ROOT", tmp_path), mock.patch.object(p23, "REPO_ROOT", tmp_path):
            campaign_rel = _fake_campaign_dir(tmp_path, "20260804T230000Z")
            case_name = p24.build_profile_plan()[0]["case_name"]
            logs_dir = tmp_path / campaign_rel / "logs"

            # --- happy path -----------------------------------------------
            rc = run_capturing_outputs(
                campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="ok.log", stderr_name=None,
                combine_stderr=False, argv=[sys.executable, "-c", "print('hello')"],
            )
            entries = sorted(p.name for p in logs_dir.iterdir())
            rec.check(
                "a successful command publishes exactly one final log",
                rc == 0 and entries == ["ok.log"] and (logs_dir / "ok.log").read_text() == "hello\n",
                detail=f"rc={rc} entries={entries}",
            )

            # --- combine-stderr ---------------------------------------------
            rc = run_capturing_outputs(
                campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="combined.log", stderr_name=None,
                combine_stderr=True, argv=[sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
            )
            combined_text = (logs_dir / "combined.log").read_text()
            rec.check("--combine-stderr merges both streams", rc == 0 and "out" in combined_text and "err" in combined_text)

            # --- failed command, non-empty output: partial preserved -------
            before = set(logs_dir.iterdir())
            rc = run_capturing_outputs(
                campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="fail_nonempty.log", stderr_name=None,
                combine_stderr=False, argv=[sys.executable, "-c", "print('partial output'); import sys; sys.exit(3)"],
            )
            after = set(logs_dir.iterdir()) - before
            partials = [p for p in after if p.name != "fail_nonempty.log"]
            rec.check(
                "a failed command preserves non-empty partial evidence and never publishes the final name",
                rc == 3 and not (logs_dir / "fail_nonempty.log").exists() and len(partials) == 1
                and partials[0].read_text() == "partial output\n",
                detail=f"rc={rc} after={sorted(p.name for p in after)}",
            )

            # --- failed command, empty output: no stale temporary ----------
            before = set(logs_dir.iterdir())
            rc = run_capturing_outputs(
                campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="fail_empty.log", stderr_name=None,
                combine_stderr=False, argv=[sys.executable, "-c", "import sys; sys.exit(1)"],
            )
            after = set(logs_dir.iterdir()) - before
            rec.check("an empty failed capture leaves no stale temporary and no final name", rc == 1 and not after)

            # --- existing regular target: publish refused -------------------
            existing = logs_dir / "existing.log"
            existing.write_text("original content\n")
            rc_err = None
            try:
                run_capturing_outputs(
                    campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="existing.log", stderr_name=None,
                    combine_stderr=False, argv=[sys.executable, "-c", "print('new content')"],
                )
            except UnsafeCaptureError as exc:
                rc_err = str(exc)
            rec.check("an existing regular target is left unchanged and publish is refused", rc_err is not None and existing.read_text() == "original content\n")

            # --- symlinked final target: publish refused ---------------------
            outside_dir = tmp_path / "outside"
            outside_dir.mkdir()
            symlink_target = outside_dir / "escape.log"
            (logs_dir / "symlinked.log").symlink_to(symlink_target)
            rc_err = None
            try:
                run_capturing_outputs(
                    campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="symlinked.log", stderr_name=None,
                    combine_stderr=False, argv=[sys.executable, "-c", "print('should not land here')"],
                )
            except UnsafeCaptureError as exc:
                rc_err = str(exc)
            rec.check("a symlinked final target is rejected; its external target is never written", rc_err is not None and not symlink_target.exists())

            # --- logs/ itself replaced by a symlink ---------------------------
            escape_dir = tmp_path / "escape_root"
            escape_dir.mkdir()
            shutil.rmtree(logs_dir)
            logs_dir.symlink_to(escape_dir)
            rc_err = None
            try:
                run_capturing_outputs(
                    campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="x.log", stderr_name=None,
                    combine_stderr=False, argv=[sys.executable, "-c", "print('should not land here')"],
                )
            except UnsafeCaptureError as exc:
                rc_err = str(exc)
            rec.check("a symlinked logs/ directory is rejected; nothing is written to its target", rc_err is not None and not any(escape_dir.iterdir()))
            logs_dir.unlink()
            logs_dir.mkdir()

            # --- race window: name swapped for a symlink after the fd is already open --
            escape_dir2 = tmp_path / "escape_root2"
            escape_dir2.mkdir()
            backup_dir = tmp_path / "logs_original_after_swap"
            swap_done = {"done": False}

            def _swap_logs_for_symlink(_dir_fd: int) -> None:
                logs_dir.rename(backup_dir)
                logs_dir.symlink_to(escape_dir2)
                swap_done["done"] = True

            rc = run_capturing_outputs(
                campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="race.log", stderr_name=None,
                combine_stderr=False, argv=[sys.executable, "-c", "print('race payload')"],
                _test_hook_after_open=_swap_logs_for_symlink,
            )
            published_in_original = backup_dir / "race.log"
            rec.check(
                "swapping logs/ for a symlink after the directory descriptor is already open does not "
                "redirect the child's write to the symlink target",
                swap_done["done"] and rc == 0 and not any(escape_dir2.iterdir())
                and published_in_original.read_text() == "race payload\n",
            )
            logs_dir.unlink()
            shutil.rmtree(backup_dir, ignore_errors=True)
            logs_dir.mkdir()

            # --- traversal/invalid names rejected everywhere ------------------
            for bad_name in ("../escape.bin", "subdir/escape.bin", "/absolute/escape.bin", ".", "..", "", "name\x00nul"):
                outside_root = tmp_path / "traversal_outside_root"
                outside_root.mkdir(exist_ok=True)
                before_outside = set(outside_root.iterdir())
                before_logs = set(logs_dir.iterdir())
                raised = False
                try:
                    write_stdin_safely(campaign_dir_rel=campaign_rel, rel_dir="logs", name=bad_name, content=b"attacker payload")
                except UnsafeCaptureError:
                    raised = True
                rec.check(
                    f"write_stdin_safely rejects traversal/invalid name {bad_name!r}",
                    raised and set(outside_root.iterdir()) == before_outside and set(logs_dir.iterdir()) - before_logs == set(),
                )

            # --- write: happy path + no-clobber + empty rejection -------------
            write_stdin_safely(campaign_dir_rel=campaign_rel, rel_dir="logs", name="written.csv", content=b"a,b\n1,2\n")
            rec.check("write_stdin_safely publishes stdin content to a new anchored file", (logs_dir / "written.csv").read_bytes() == b"a,b\n1,2\n")
            raised = False
            try:
                write_stdin_safely(campaign_dir_rel=campaign_rel, rel_dir="logs", name="written.csv", content=b"tampered\n")
            except UnsafeCaptureError:
                raised = True
            rec.check("write_stdin_safely refuses to overwrite an existing file", raised and (logs_dir / "written.csv").read_bytes() == b"a,b\n1,2\n")
            raised = False
            try:
                write_stdin_safely(campaign_dir_rel=campaign_rel, rel_dir="logs", name="empty.csv", content=b"")
            except UnsafeCaptureError:
                raised = True
            rec.check("write_stdin_safely refuses to publish empty content", raised and not (logs_dir / "empty.csv").exists())

            # --- verify: genuine, missing, empty, symlinked -------------------
            profiles_dir = tmp_path / campaign_rel / "profiles" / case_name
            (profiles_dir / f"{case_name}_report.ncu-rep").write_bytes(b"synthetic ncu-rep bytes\n")
            dir_fd = resolve_campaign_rel_dir_fd(campaign_rel, f"profiles/{case_name}")
            try:
                err = verify_regular_file_in_dir(dir_fd, f"{case_name}_report.ncu-rep")
            finally:
                os.close(dir_fd)
            rec.check("verify_regular_file_in_dir accepts a genuine non-symlink non-empty file", err is None)

            (profiles_dir / "empty.bin").touch()
            (profiles_dir / "escape_target.bin").write_bytes(b"outside content\n")
            (profiles_dir / "sneaky_link.bin").symlink_to(profiles_dir / "escape_target.bin")
            dir_fd = resolve_campaign_rel_dir_fd(campaign_rel, f"profiles/{case_name}")
            try:
                err_missing = verify_regular_file_in_dir(dir_fd, "does_not_exist.bin")
                err_empty = verify_regular_file_in_dir(dir_fd, "empty.bin")
                err_symlink = verify_regular_file_in_dir(dir_fd, "sneaky_link.bin")
            finally:
                os.close(dir_fd)
            rec.check("verify_regular_file_in_dir rejects missing/empty/symlinked names", err_missing is not None and err_empty is not None and err_symlink is not None)

            raised = False
            try:
                resolve_campaign_rel_dir_fd(campaign_rel, "not_a_real_subdir")
            except UnsafeCaptureError:
                raised = True
            rec.check("resolve_campaign_rel_dir_fd rejects a rel-dir outside the frozen allowlist", raised)

            # --- mkdir-case ------------------------------------------------
            second_case_name = p24.build_profile_plan()[1]["case_name"]
            profiles_root = tmp_path / campaign_rel / "profiles"
            fd = resolve_profiles_root_fd(campaign_rel)
            try:
                mkdir_case_dir(fd, second_case_name)
            finally:
                os.close(fd)
            rec.check("mkdir_case_dir creates exactly the named frozen case directory", (profiles_root / second_case_name).is_dir() and not (profiles_root / second_case_name).is_symlink())
            raised = False
            fd = resolve_profiles_root_fd(campaign_rel)
            try:
                mkdir_case_dir(fd, second_case_name)
            except UnsafeCaptureError:
                raised = True
            finally:
                os.close(fd)
            rec.check("mkdir_case_dir refuses to recreate an already-existing case directory", raised)
            raised = False
            fd = resolve_profiles_root_fd(campaign_rel)
            try:
                mkdir_case_dir(fd, "not_one_of_the_24_frozen_cases")
            except UnsafeCaptureError:
                raised = True
            finally:
                os.close(fd)
            rec.check("mkdir_case_dir refuses a case name outside the frozen 24-case plan", raised)

            # --- NCU bundle round-trip and publish-bundle ---------------------
            sample_segments = {
                "app_stdout": b"application stdout bytes\n", "app_stderr": b"",
                "ncu_tool_log": b"fake ncu tool log\n",
                "ncu_rep": b"\x00\x01FAKE_NCU_REP\xff\xfe" + NCU_BUNDLE_MAGIC,
                "metrics_csv": b"ID,Kernel Name\n0,k\n", "metrics_export_stderr": b"",
            }
            encoded = encode_ncu_bundle(sample_segments)
            decoded = decode_ncu_bundle(encoded)
            rec.check(
                "encode_ncu_bundle/decode_ncu_bundle round-trip is exact, even with a segment containing the magic marker as content",
                decoded == sample_segments,
            )
            noisy = b"run_container: selected index=3 uuid=GPU-xxxx name='fake' driver=1.0\n" + encoded
            rec.check("decode_ncu_bundle tolerates leading bytes before the magic marker", decode_ncu_bundle(noisy) == sample_segments)
            raised = False
            try:
                decode_ncu_bundle(encoded[: len(encoded) - 5])
            except NcuBundleParseError:
                raised = True
            rec.check("decode_ncu_bundle rejects a truncated bundle", raised)

            bundle_case_name = second_case_name
            bundle_case_dir = tmp_path / campaign_rel / "profiles" / bundle_case_name
            bundle_case_rel = f"profiles/{bundle_case_name}"
            bundle_script = tmp_path / "fake_bundle_emitter.py"
            bundle_script.write_text("import sys\nsys.stdout.buffer.write(" + repr(encoded) + ")\n", encoding="utf-8")
            run_capturing_outputs(
                campaign_dir_rel=campaign_rel, rel_dir=bundle_case_rel, stdout_name="raw_bundle.bin", stderr_name=None,
                combine_stderr=False, argv=[sys.executable, str(bundle_script)],
            )
            output_names = [
                f"{bundle_case_name}.container_stdout.log", f"{bundle_case_name}.container_stderr.log",
                f"{bundle_case_name}.ncu_tool.log", f"{bundle_case_name}_report.ncu-rep",
                f"{bundle_case_name}.metrics_raw.csv", f"{bundle_case_name}.metrics_export_stderr.log",
            ]
            publish_ncu_bundle(campaign_dir_rel=campaign_rel, rel_dir=bundle_case_rel, bundle_name="raw_bundle.bin", output_names=output_names)
            published_contents = {name: (bundle_case_dir / name).read_bytes() for name in output_names}
            rec.check(
                "publish-bundle republishes all six segments under their real names with exact byte-for-byte content",
                published_contents[output_names[0]] == sample_segments["app_stdout"]
                and published_contents[output_names[3]] == sample_segments["ncu_rep"]
                and published_contents[output_names[4]] == sample_segments["metrics_csv"],
            )
            rec.check("publish-bundle removes the raw transport bundle after successfully republishing it", not (bundle_case_dir / "raw_bundle.bin").exists())

            malformed_script = tmp_path / "fake_malformed_bundle_emitter.py"
            malformed_script.write_text("import sys\nsys.stdout.buffer.write(b'not a real bundle')\n", encoding="utf-8")
            run_capturing_outputs(
                campaign_dir_rel=campaign_rel, rel_dir=bundle_case_rel, stdout_name="raw_bundle_bad.bin", stderr_name=None,
                combine_stderr=False, argv=[sys.executable, str(malformed_script)],
            )
            before = set(bundle_case_dir.iterdir())
            raised = False
            try:
                publish_ncu_bundle(campaign_dir_rel=campaign_rel, rel_dir=bundle_case_rel, bundle_name="raw_bundle_bad.bin", output_names=output_names)
            except UnsafeCaptureError:
                raised = True
            after = set(bundle_case_dir.iterdir())
            rec.check("publish-bundle rejects a malformed bundle and publishes none of the six outputs", raised and after == before)

    if rec.failures:
        print(f"p24_safe_capture: self-test: FAILED ({len(rec.failures)}/{rec.total} case(s)): {rec.failures}", file=sys.stderr)
        print("p24_safe_capture: SELF_TEST_RESULT=FAIL", file=sys.stderr)
        return 1
    print(f"p24_safe_capture: self-test: OK ({rec.total} cases)", file=sys.stderr)
    print("p24_safe_capture: SELF_TEST_RESULT=PASS", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="p24_safe_capture.py", description="P2.4-only descriptor-anchored safe command capture (see module docstring).")
    parser.add_argument("--self-test", action="store_true", help="Run GPU-free synthetic tests and exit.")
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Safely capture one command's stdout/stderr.")
    run_parser.add_argument("--campaign-dir", required=True)
    run_parser.add_argument("--rel-dir", required=True)
    run_parser.add_argument("--stdout-name", default=None)
    run_parser.add_argument("--stderr-name", default=None)
    run_parser.add_argument("--combine-stderr", action="store_true")
    run_parser.add_argument("command", nargs=argparse.REMAINDER)
    run_parser.set_defaults(func=cmd_run)

    write_parser = subparsers.add_parser("write", help="Safely publish stdin as one new anchored file.")
    write_parser.add_argument("--campaign-dir", required=True)
    write_parser.add_argument("--rel-dir", required=True)
    write_parser.add_argument("--name", required=True)
    write_parser.set_defaults(func=cmd_write)

    verify_parser = subparsers.add_parser("verify", help="Confirm one or more names are genuine files strictly inside the anchored directory.")
    verify_parser.add_argument("--campaign-dir", required=True)
    verify_parser.add_argument("--rel-dir", required=True)
    verify_parser.add_argument("--name", action="append", required=True, dest="name")
    verify_parser.set_defaults(func=cmd_verify)

    mkdir_parser = subparsers.add_parser("mkdir-case", help="Safely create one of the 24 frozen profiles/<case> directories.")
    mkdir_parser.add_argument("--campaign-dir", required=True)
    mkdir_parser.add_argument("--case-name", required=True)
    mkdir_parser.set_defaults(func=cmd_mkdir_case)

    bundle_parser = subparsers.add_parser("publish-bundle", help="Decode an already-captured scripts/p24_ncu_bridge.py bundle and publish its six segments under their real names, no-clobber.")
    bundle_parser.add_argument("--campaign-dir", required=True)
    bundle_parser.add_argument("--rel-dir", required=True)
    bundle_parser.add_argument("--bundle-name", required=True)
    bundle_parser.add_argument(
        "--names", required=True, nargs=len(NCU_BUNDLE_SEGMENT_NAMES), metavar=tuple(n.upper() for n in NCU_BUNDLE_SEGMENT_NAMES),
        help="Exactly " + str(len(NCU_BUNDLE_SEGMENT_NAMES)) + " output names, in fixed order " + str(NCU_BUNDLE_SEGMENT_NAMES) + ".",
    )
    bundle_parser.set_defaults(func=cmd_publish_bundle)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--self-test"]:
        return run_self_test()

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return run_self_test()
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
