#!/usr/bin/env python3
"""P1.4-only descriptor-anchored safe command capture (Remediation A, second
independent audit, blocker A).

The first remediation added a precheck-then-open guard
(``capture_target_is_safe``) immediately before each shell redirection in
``scripts/run_exp01_memory_paths_p14.sh``. That still leaves a TOCTOU window:
anything can replace the checked path between the check and the later
``open()`` a shell redirection performs. This module closes that window by
never resolving a campaign-relative pathname more than once. Every directory
component from the repository root down to ``logs/`` or
``profiles/<case>/`` is opened exactly once with Linux no-follow semantics
(``os.open(..., dir_fd=parent_fd)`` with ``O_DIRECTORY | O_NOFOLLOW``); the
resulting open directory file descriptor is then used for every subsequent
operation (creating the output file, publishing it, and connecting the
child's stdout/stderr to it). Because the kernel resolves ``*at()`` calls
against the file descriptor's underlying inode rather than re-walking a
pathname, nothing that happens to the *name* ``logs`` (or any ancestor)
after this module opens it can redirect a later operation elsewhere -- not a
symlink swap of the directory itself, and not a symlink swap of the output
name, since the output name is also created via ``O_EXCL | O_NOFOLLOW``
relative to the same held descriptor and only ever published by an
in-directory hard link (never ``os.replace()``, never a rename over an
existing name, never following a symlink).

This module never touches Docker, CUDA, NCU, ``nvidia-smi``, or GPU
hardware itself: it only orchestrates already-decided argv vectors
(``shell=False``) and safely captures their stdout/stderr. Every check here
is a P1.4-only addition; it never modifies
``scripts/aggregate_exp01_memory_paths.py`` (P1.3, frozen) and is not
imported by it.

Every filename this module accepts from a caller (--stdout-name,
--stderr-name, "write"/"publish-bundle"'s --name/--names, and --bundle-name)
is validated as a strict single-component basename before anything is
created or any child is launched: empty, ".", "..", any name containing "/"
or NUL/control characters, and any absolute path are all rejected outright
(Task 4 remediation E; a prior version accepted a caller-supplied name
containing "../", which os.link()'s own dst_dir_fd-relative path resolution
then walked one level above the anchored directory exactly like any other
relative path with ".." components would).

Subcommands:
  run             Execute ARGV (after "--", shell=False) with stdout/stderr
                  connected directly to newly created, descriptor-anchored,
                  exclusive, no-follow partial files under
                  <campaign-dir>/<rel-dir>/; on a zero exit, publishes each
                  partial to its final name via no-clobber hard link; on a
                  non-zero exit -- or any failure before or during the
                  child's own launch -- preserves a non-empty partial under
                  its unique name and removes only an empty owned partial.
                  Exits with the child's own return code (or 2 for a
                  tool-level path-safety or publish failure).
  write           Reads bytes from stdin and safely publishes them as one
                  new, no-clobber file under <campaign-dir>/<rel-dir>/.
                  Used for the one P1.4 output that is not literally a
                  command's own stdout/stderr (the application CSV
                  recovered from a profiled binary's captured stdout).
  verify          Confirms that one or more names already exist as genuine
                  non-symlink, non-empty regular files strictly within
                  <campaign-dir>/<rel-dir>/, reopening every directory
                  component the same descriptor-anchored way.
  mkdir-case      Safely creates exactly one of the six frozen
                  profiles/<case>/ directories via mkdirat()'s own EEXIST.
  publish-bundle  Decodes an already-captured scripts/p14_ncu_bridge.py
                  bundle (itself captured via a prior "run --stdout-name")
                  and republishes its six fixed-order segments (Task 4
                  remediation A/B) under caller-given names, no-clobber,
                  then removes the raw transport bundle.

Exit codes: 0 success; the child's own exit code for a "run" whose command
ran but exited non-zero; 1 for "write"/"verify"/"publish-bundle"
content/evidence failures; 2 for a path-safety, argument, or
capture-mechanism failure.
"""

from __future__ import annotations

import argparse
import errno
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

import aggregate_exp01_memory_paths as p13  # noqa: E402
import analyze_exp01_memory_paths_p14 as p14  # noqa: E402


class UnsafeCaptureError(ValueError):
    pass


# ---------------------------------------------------------------------------
# Strict single-component basename validation (Task 4 remediation,
# Remediation E). Every filename this module accepts from a caller --
# --stdout-name, --stderr-name, the "write"/"verify" --name -- must be a
# single path component, never a multi-component relative path. Before this
# check existed, publish_no_clobber(dir_fd, partial_name, final_name) passed
# final_name directly to os.link(..., dst_dir_fd=dir_fd); linkat() resolves a
# multi-component relative newpath against dst_dir_fd exactly like any other
# relative path, including ".." components, so a final_name of
# "../escape.bin" walked one level *above* the anchored directory (out of
# logs/ into the campaign root, or out of profiles/<case>/ into profiles/) --
# the directory-descriptor anchoring closes a *symlink* race on the
# anchored directory itself, but never protected against a multi-component
# *name* handed to it. verify_regular_file_in_dir and write_stdin_safely's
# create_partial/publish_no_clobber calls had the identical exposure.
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
    # Path(name).parts == (name,) rejects every multi-component relative
    # path ("subdir/escape.bin", "../escape.bin", "a/../b", trailing/leading
    # slashes, etc.) in addition to the conservative allowlist regex below;
    # requiring *both* means a change to either check alone can never
    # silently widen what is accepted.
    if Path(name).parts != (name,):
        raise UnsafeCaptureError(f"{what}: must be a single path component, not {name!r}")
    if not _BASENAME_RE.fullmatch(name):
        raise UnsafeCaptureError(
            f"{what}: must match {_BASENAME_RE.pattern} (start with an alphanumeric character; "
            f"only alphanumerics, '.', '_', '-' afterward), got {name!r}"
        )


# ---------------------------------------------------------------------------
# Descriptor-anchored directory resolution. Never trusts an earlier
# validation of the same pathname (e.g. validate-profile-preconditions, or
# an earlier check_ncu_help_capability run): every call re-opens every
# component itself, one at a time, with O_NOFOLLOW.
# ---------------------------------------------------------------------------
def _rel_dir_allowlist() -> tuple[str, ...]:
    plan = p14.build_ncu_plan()
    return ("logs",) + tuple(f"profiles/{entry['case_name']}" for entry in plan)


def _validate_campaign_dir_rel(campaign_dir_rel: str) -> tuple[str, ...]:
    if os.path.isabs(campaign_dir_rel):
        raise UnsafeCaptureError(f"--campaign-dir must be relative, got absolute path {campaign_dir_rel!r}")
    parts = Path(campaign_dir_rel).parts
    if any(".." in part for part in parts):
        raise UnsafeCaptureError(f"--campaign-dir must not contain '..': {campaign_dir_rel!r}")
    if (
        len(parts) != len(p14.RAW_ROOT_PARTS_P14) + 1
        or tuple(parts[: len(p14.RAW_ROOT_PARTS_P14)]) != p14.RAW_ROOT_PARTS_P14
    ):
        raise UnsafeCaptureError(
            f"--campaign-dir must be exactly {'/'.join(p14.RAW_ROOT_PARTS_P14)}/<campaign_id>, "
            f"got {campaign_dir_rel!r}"
        )
    try:
        p14.validate_p14_campaign_id(parts[-1])
    except p13.UnsafePathError as exc:
        raise UnsafeCaptureError(str(exc)) from exc
    return parts


def _validate_rel_dir(rel_dir: str) -> tuple[str, ...]:
    allowed = _rel_dir_allowlist()
    if rel_dir not in allowed:
        raise UnsafeCaptureError(
            f"--rel-dir must be one of {sorted(allowed)!r} (the frozen logs/ directory or one "
            f"of the six canonical profiles/<case> directories), got {rel_dir!r}"
        )
    parts = Path(rel_dir).parts
    if any(".." in part for part in parts):
        raise UnsafeCaptureError(f"--rel-dir must not contain '..': {rel_dir!r}")
    return parts


def _open_dir_nofollow(name: str, *, dir_fd: int | None) -> int:
    """Opens exactly one directory component relative to dir_fd (or, if
    dir_fd is None, as an absolute/cwd-relative path -- used only for the
    very first component, the repository root). O_NOFOLLOW makes the
    kernel itself refuse a symlink at this exact name, dangling or not,
    instead of relying on a separate lstat() that a later open() could
    race past."""
    flags = os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        return os.open(name, flags, dir_fd=dir_fd)
    except OSError as exc:
        raise UnsafeCaptureError(f"{name}: cannot open as a non-symlink directory: {exc}") from exc


def _open_repo_relative_dir_fd(*parts: str) -> int:
    """Descriptor-anchored, no-follow open of REPO_ROOT/<parts...> as a
    directory: every component, including the repository root itself, is
    opened exactly once via O_NOFOLLOW relative to the previous, already-open
    descriptor. Returns an open fd the caller must close via os.close()."""
    root = str(p14.REPO_ROOT)
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
    """Descriptor-anchored, no-follow open of
    REPO_ROOT/campaign_dir_rel/rel_dir as a directory. Re-validates the
    exact canonical campaign-directory shape and campaign ID, and that
    rel_dir is exactly "logs" or one of the six frozen "profiles/<case>"
    directories -- never trusting that an earlier validation of the same
    strings still holds -- then opens every component from the repository
    root down, one at a time, closing each parent fd once its child is
    open. Returns an open fd the caller must close via os.close()."""
    campaign_parts = _validate_campaign_dir_rel(campaign_dir_rel)
    rel_parts = _validate_rel_dir(rel_dir)
    return _open_repo_relative_dir_fd(*campaign_parts, *rel_parts)


def resolve_profiles_root_fd(campaign_dir_rel: str) -> int:
    """Descriptor-anchored, no-follow open of
    REPO_ROOT/campaign_dir_rel/profiles itself (the fixed parent of all six
    canonical case directories), for safely creating one of those case
    directories. Returns an open fd the caller must close via os.close()."""
    campaign_parts = _validate_campaign_dir_rel(campaign_dir_rel)
    return _open_repo_relative_dir_fd(*campaign_parts, "profiles")


def mkdir_case_dir(profiles_dir_fd: int, case_name: str) -> None:
    """Creates exactly one of the six frozen case directories strictly
    within profiles_dir_fd. mkdirat()'s own EEXIST is the sole, atomic
    guarantee that an existing entry at this name -- directory, regular
    file, or symlink -- is never silently reused or followed; the
    directory this fd anchors cannot be swapped out from under the caller
    between an earlier check and this call, because there is no earlier
    check -- this *is* the check."""
    valid_names = {entry["case_name"] for entry in p14.build_ncu_plan()}
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
    return f".{final_name}.p14capture-{uuid.uuid4().hex}.partial"


def create_partial(dir_fd: int, final_name: str) -> tuple[int, str]:
    """Creates a uniquely-named partial output file strictly within dir_fd,
    O_EXCL | O_NOFOLLOW (so it can never already exist and can never be a
    symlink), and returns (fd, partial_name). The caller must eventually
    publish_no_clobber() or discard_if_empty_owned() it."""
    _validate_basename(final_name, what="output name")
    partial = _partial_name(final_name)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(partial, flags, 0o600, dir_fd=dir_fd)
    except OSError as exc:
        raise UnsafeCaptureError(f"{final_name}: cannot create owned partial output: {exc}") from exc
    return fd, partial


def publish_no_clobber(dir_fd: int, partial_name: str, final_name: str) -> None:
    """Publishes partial_name as final_name within the same directory via a
    hard link, then unlinks partial_name. Never os.replace(); linkat()'s own
    EEXIST is the sole, atomic no-clobber guarantee -- final_name is refused
    whether it is already a regular file, a directory, or a symlink
    (dangling or not), since all of those already occupy the name."""
    _validate_basename(final_name, what="output name")
    try:
        os.link(partial_name, final_name, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    except FileExistsError as exc:
        raise UnsafeCaptureError(f"{final_name}: already exists; refusing to overwrite") from exc
    except OSError as exc:
        raise UnsafeCaptureError(f"{final_name}: could not publish without clobbering: {exc}") from exc
    os.unlink(partial_name, dir_fd=dir_fd)


def discard_if_empty_owned(dir_fd: int, partial_name: str, partial_stat: os.stat_result) -> bool:
    """Removes partial_name only if partial_stat -- captured via os.fstat()
    on the partial's own fd *before* it was closed -- shows an empty
    regular file, and a fresh, non-following stat of the name still
    identifies the same inode (defends against the name being swapped for a
    symlink between the write and this cleanup). Returns True if removed;
    otherwise the partial is left in place as non-empty failure evidence."""
    if not stat.S_ISREG(partial_stat.st_mode) or partial_stat.st_size != 0:
        return False
    st_name = os.stat(partial_name, dir_fd=dir_fd, follow_symlinks=False)
    if (st_name.st_dev, st_name.st_ino) != (partial_stat.st_dev, partial_stat.st_ino):
        raise UnsafeCaptureError(f"{partial_name}: changed identity before cleanup; refusing to unlink")
    os.unlink(partial_name, dir_fd=dir_fd)
    return True


def verify_regular_file_in_dir(dir_fd: int, name: str) -> str | None:
    """Returns an error string, or None if name exists strictly within
    dir_fd as a non-symlink, non-empty regular file. Never follows a
    symlink and never resolves name via any other path."""
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
# NCU bundle format (Task 4, blockers A/B). scripts/p14_ncu_bridge.py runs
# entirely inside the container's own private, non-host-mounted /tmp and
# never receives a campaign-relative pathname; it hands its results back to
# the host as one versioned, length-delimited bundle on its own stdout,
# which is captured here (via the ordinary "run" subcommand below) into an
# anchored partial exactly like any other P1.4 child stdout. "publish-bundle"
# then decodes that already-anchored file and republishes its six segments
# under their real names via the same no-clobber primitives every other
# P1.4 artifact uses.
#
# Length-prefixed, not delimiter-based, so arbitrary binary content (the
# ".ncu-rep" bytes, or a metrics CSV containing any byte sequence at all)
# can never be misparsed. Decoding tolerates leading bytes before the magic
# marker: scripts/run_container.sh prints two allowlisted banner lines to
# its own stdout before exec'ing the container command, which lands in the
# same captured stream ahead of the bridge's real output -- this is the
# same, already-accepted tolerance
# scripts/run_exp01_memory_paths_p14.sh's extract_application_csv() already
# relies on for the (unmodified) existing per-case application-CSV recovery.
# ---------------------------------------------------------------------------
NCU_BUNDLE_MAGIC = b"P14NCUBUNDLE1\n"
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
            raise NcuBundleParseError(
                f"segment {name!r}: truncated (expected {length} bytes, only "
                f"{len(raw) - data_start} available)"
            )
        segments[name] = raw[data_start:data_end]
        pos = data_end
    return segments


def publish_ncu_bundle(
    *, campaign_dir_rel: str, rel_dir: str, bundle_name: str, output_names: Sequence[str],
) -> None:
    """Decodes the already-captured, already-anchored bundle_name and
    republishes its six fixed-order segments as output_names (same order as
    NCU_BUNDLE_SEGMENT_NAMES), each via create_partial/publish_no_clobber,
    then removes the raw bundle (its only purpose was transport). Every
    name is validated as a strict basename before anything is opened;
    directory resolution is re-done from scratch (never trusts an earlier
    "run" call's resolution)."""
    _validate_basename(bundle_name, what="--bundle-name")
    if len(output_names) != len(NCU_BUNDLE_SEGMENT_NAMES):
        raise UnsafeCaptureError(
            f"--names requires exactly {len(NCU_BUNDLE_SEGMENT_NAMES)} value(s) in fixed order "
            f"{NCU_BUNDLE_SEGMENT_NAMES!r}, got {len(output_names)}"
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

        published: list[str] = []
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
                published.append(output_name)
        except Exception:
            for name in published:
                try:
                    st = os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
                    if stat.S_ISREG(st.st_mode):
                        os.unlink(name, dir_fd=dir_fd)
                except OSError:
                    pass
            raise

        os.unlink(bundle_name, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)


def cmd_publish_bundle(args: argparse.Namespace) -> int:
    try:
        publish_ncu_bundle(
            campaign_dir_rel=args.campaign_dir, rel_dir=args.rel_dir,
            bundle_name=args.bundle_name, output_names=args.names,
        )
    except UnsafeCaptureError as exc:
        print(f"p14_safe_capture: publish-bundle: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"p14_safe_capture: publish-bundle: OK: {args.rel_dir}/{{{','.join(args.names)}}}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: run
# ---------------------------------------------------------------------------
def run_capturing_outputs(
    *,
    campaign_dir_rel: str,
    rel_dir: str,
    argv: Sequence[str],
    stdout_name: str | None,
    stderr_name: str | None,
    combine_stderr: bool,
    _test_hook_after_open: Callable[[int], None] | None = None,
) -> int:
    """Resolves <campaign_dir_rel>/<rel_dir> via resolve_campaign_rel_dir_fd,
    then runs argv (shell=False) with stdout/stderr connected directly to
    open, already-created output descriptors (never a shell redirection, so
    the child never re-resolves a campaign-relative pathname at all). Omit
    stdout_name/stderr_name to inherit this process's own stream (used so
    the caller can still capture it, e.g. via a shell command
    substitution). Returns the child's own exit code on a clean run;
    raises UnsafeCaptureError for any pre-launch or publish failure.

    _test_hook_after_open, if given, is called with the resolved directory
    fd immediately after every output descriptor has been created but
    before the child is launched -- self-test only, never exposed as a
    production CLI option -- so an adversarial test can replace the
    on-disk name of the resolved directory (or of an output name) at
    exactly that point and prove the already-open descriptors are immune
    to it.
    """
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
        # pre_launch_error covers every failure that can happen before or
        # during the child's own launch: creating the first output,
        # creating the second output (after the first already exists on
        # disk), the injectable test hook, or subprocess.run() itself
        # failing to start the child (e.g. a nonexistent executable). Each
        # of these previously propagated straight out of this function
        # without ever reaching the cleanup logic below, so any partial
        # already created (necessarily empty, since nothing had a chance to
        # write to it) was silently orphaned rather than removed -- this
        # single except clause routes every one of them through the exact
        # same "did we already create an output? then discard it if still
        # empty" cleanup a non-zero child exit already used, before
        # re-raising the original failure.
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

            result = subprocess.run(
                list(argv),
                stdout=stdout_fd,
                stderr=stderr_fd,
                stdin=subprocess.DEVNULL,
                shell=False,
            )
        except OSError as exc:
            pre_launch_error = UnsafeCaptureError(f"could not launch command: {exc}")
        except UnsafeCaptureError as exc:
            pre_launch_error = exc
        finally:
            # fstat before close: a later "was the partial left empty?" check
            # (relevant on a non-zero exit or a pre-launch failure) must not
            # depend on a fd that is no longer open.
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
            campaign_dir_rel=args.campaign_dir,
            rel_dir=args.rel_dir,
            argv=argv,
            stdout_name=args.stdout_name,
            stderr_name=args.stderr_name,
            combine_stderr=args.combine_stderr,
        )
    except UnsafeCaptureError as exc:
        print(f"p14_safe_capture: run: ERROR: {exc}", file=sys.stderr)
        return 2
    if rc != 0:
        print(f"p14_safe_capture: run: command exited {rc}", file=sys.stderr)
        return rc if 0 < rc < 126 else 1
    return 0


# ---------------------------------------------------------------------------
# Subcommand: write (stdin -> one new, no-clobber, anchored file)
# ---------------------------------------------------------------------------
def write_stdin_safely(
    *, campaign_dir_rel: str, rel_dir: str, name: str, content: bytes, allow_empty: bool = False,
) -> None:
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
        write_stdin_safely(
            campaign_dir_rel=args.campaign_dir, rel_dir=args.rel_dir, name=args.name, content=content,
        )
    except UnsafeCaptureError as exc:
        print(f"p14_safe_capture: write: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"p14_safe_capture: write: OK: {args.rel_dir}/{args.name}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Subcommand: verify (confirm NCU's own direct -o/--log-file writes landed
# strictly inside the anchored directory)
# ---------------------------------------------------------------------------
def cmd_verify(args: argparse.Namespace) -> int:
    try:
        dir_fd = resolve_campaign_rel_dir_fd(args.campaign_dir, args.rel_dir)
    except UnsafeCaptureError as exc:
        print(f"p14_safe_capture: verify: ERROR: {exc}", file=sys.stderr)
        return 2
    try:
        errors = [err for name in args.name if (err := verify_regular_file_in_dir(dir_fd, name))]
    finally:
        os.close(dir_fd)
    if errors:
        for err in errors:
            print(f"p14_safe_capture: verify: ERROR: {err}", file=sys.stderr)
        return 1
    print(f"p14_safe_capture: verify: OK: {args.rel_dir}/{{{','.join(args.name)}}}", file=sys.stderr)
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
        print(f"p14_safe_capture: mkdir-case: ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"p14_safe_capture: mkdir-case: OK: profiles/{args.case_name}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# --self-test: GPU-free, Docker-free, fully isolated in a temporary
# directory standing in for REPO_ROOT. Never touches results/raw/ or
# results/preflight/.
# ---------------------------------------------------------------------------
def _fake_campaign_dir(tmp_path: Path, campaign_id: str = "20260728T230000Z") -> str:
    rel = tmp_path.joinpath(*p14.RAW_ROOT_PARTS_P14, campaign_id)
    (rel / "logs").mkdir(parents=True)
    case_name = p14.build_ncu_plan()[0]["case_name"]
    (rel / "profiles" / case_name).mkdir(parents=True)
    return str(Path(*p14.RAW_ROOT_PARTS_P14) / campaign_id)


class _Recorder:
    def __init__(self) -> None:
        self.total = 0
        self.failures: list[str] = []

    def check(self, label: str, condition: bool, *, detail: str = "") -> None:
        self.total += 1
        if condition:
            print(f"p14_safe_capture: self-test: PASS: {label}", file=sys.stderr)
        else:
            print(f"p14_safe_capture: self-test: FAIL: {label}; {detail}", file=sys.stderr)
            self.failures.append(label)


def run_self_test() -> int:
    import tempfile
    from unittest import mock

    rec = _Recorder()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with mock.patch.object(p14, "REPO_ROOT", tmp_path), mock.patch.object(p13, "REPO_ROOT", tmp_path):
            campaign_rel = _fake_campaign_dir(tmp_path, "20260728T230000Z")
            case_name = p14.build_ncu_plan()[0]["case_name"]

            # --- happy path: exactly one final file, no leftover partial ---
            rc = run_capturing_outputs(
                campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="ok.log",
                stderr_name=None, combine_stderr=False,
                argv=[sys.executable, "-c", "print('hello')"],
            )
            logs_dir = tmp_path / campaign_rel / "logs"
            entries = sorted(p.name for p in logs_dir.iterdir())
            rec.check(
                "a successful fake command publishes exactly one final log",
                rc == 0 and entries == ["ok.log"] and (logs_dir / "ok.log").read_text() == "hello\n",
                detail=f"rc={rc} entries={entries}",
            )

            # --- combine-stderr: one file receives both streams ---
            rc = run_capturing_outputs(
                campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="combined.log",
                stderr_name=None, combine_stderr=True,
                argv=[sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
            )
            combined_text = (logs_dir / "combined.log").read_text()
            rec.check(
                "--combine-stderr merges both streams into the one published file",
                rc == 0 and "out" in combined_text and "err" in combined_text,
                detail=f"rc={rc} content={combined_text!r}",
            )

            # --- stdout inherited (omitted), stderr captured ---
            rc = run_capturing_outputs(
                campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name=None,
                stderr_name="stderr_only.log", combine_stderr=False,
                argv=[sys.executable, "-c", "import sys; print('to-stderr', file=sys.stderr)"],
            )
            rec.check(
                "omitting --stdout-name inherits the parent stream and still captures stderr",
                rc == 0 and (logs_dir / "stderr_only.log").read_text().strip() == "to-stderr",
                detail=f"rc={rc}",
            )

            # --- failed command, non-empty output: partial preserved ---
            before = set(logs_dir.iterdir())
            rc = run_capturing_outputs(
                campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="fail_nonempty.log",
                stderr_name=None, combine_stderr=False,
                argv=[sys.executable, "-c", "import sys; print('partial output'); sys.exit(3)"],
            )
            after = set(logs_dir.iterdir()) - before
            partials = [p for p in after if p.name != "fail_nonempty.log"]
            rec.check(
                "a failed fake command preserves non-empty partial evidence and never "
                "publishes the final name",
                rc == 3 and not (logs_dir / "fail_nonempty.log").exists()
                and len(partials) == 1 and partials[0].read_text() == "partial output\n",
                detail=f"rc={rc} after={sorted(p.name for p in after)}",
            )

            # --- failed command, empty output: no stale temporary ---
            before = set(logs_dir.iterdir())
            rc = run_capturing_outputs(
                campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="fail_empty.log",
                stderr_name=None, combine_stderr=False,
                argv=[sys.executable, "-c", "import sys; sys.exit(1)"],
            )
            after = set(logs_dir.iterdir()) - before
            rec.check(
                "an empty failed capture leaves no stale temporary and no final name",
                rc == 1 and not after,
                detail=f"rc={rc} after={sorted(p.name for p in after)}",
            )

            # --- existing regular target: publish refused, target unchanged ---
            existing = logs_dir / "existing.log"
            existing.write_text("original content\n")
            rc_err = None
            try:
                run_capturing_outputs(
                    campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="existing.log",
                    stderr_name=None, combine_stderr=False,
                    argv=[sys.executable, "-c", "print('new content')"],
                )
            except UnsafeCaptureError as exc:
                rc_err = str(exc)
            rec.check(
                "an existing regular target is left byte-for-byte unchanged and publish is refused",
                rc_err is not None and existing.read_text() == "original content\n",
                detail=f"rc_err={rc_err}",
            )

            # --- symlinked final target: publish refused, symlink target untouched ---
            outside_dir = tmp_path / "outside"
            outside_dir.mkdir()
            symlink_target = outside_dir / "escape.log"
            (logs_dir / "symlinked.log").symlink_to(symlink_target)
            rc_err = None
            try:
                run_capturing_outputs(
                    campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="symlinked.log",
                    stderr_name=None, combine_stderr=False,
                    argv=[sys.executable, "-c", "print('should not land here')"],
                )
            except UnsafeCaptureError as exc:
                rc_err = str(exc)
            rec.check(
                "a symlinked final target is rejected and its external target is never written",
                rc_err is not None and not symlink_target.exists(),
                detail=f"rc_err={rc_err}",
            )

            # --- logs/ itself replaced by a symlink to an outside directory ---
            escape_dir = tmp_path / "escape_root"
            escape_dir.mkdir()
            import shutil
            shutil.rmtree(logs_dir)
            logs_dir.symlink_to(escape_dir)
            rc_err = None
            try:
                run_capturing_outputs(
                    campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="x.log",
                    stderr_name=None, combine_stderr=False,
                    argv=[sys.executable, "-c", "print('should not land here')"],
                )
            except UnsafeCaptureError as exc:
                rc_err = str(exc)
            rec.check(
                "a symlinked logs/ directory is rejected and nothing is written to its target "
                "(item: symlinked campaign/logs directory)",
                rc_err is not None and not any(escape_dir.iterdir()),
                detail=f"rc_err={rc_err} escape_dir_contents={list(escape_dir.iterdir())}",
            )
            logs_dir.unlink()
            logs_dir.mkdir()

            # --- broken (dangling) symlink in place of logs/ ---
            logs_dir.rmdir()
            logs_dir.symlink_to(tmp_path / "does_not_exist_at_all")
            rc_err = None
            try:
                run_capturing_outputs(
                    campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="x.log",
                    stderr_name=None, combine_stderr=False,
                    argv=[sys.executable, "-c", "print('unreachable')"],
                )
            except UnsafeCaptureError as exc:
                rc_err = str(exc)
            rec.check(
                "a broken (dangling) symlink in place of logs/ is rejected",
                rc_err is not None,
                detail=f"rc_err={rc_err}",
            )
            logs_dir.unlink()
            logs_dir.mkdir()

            # --- race window 2: the name is swapped for a symlink *after* the
            # directory descriptor and output descriptors are already open, but
            # *before* the (fake) child writes. An injectable, synchronous hook
            # performs the swap deterministically -- no sleep, no thread. The
            # swap renames (never deletes) the original directory so its
            # already-created partial file survives under a different path,
            # exactly like a real attacker replacing only the *name* "logs"
            # would -- the held dir_fd/output fds are anchored to the inode,
            # not the name, and must be unaffected either way. ---
            escape_dir2 = tmp_path / "escape_root2"
            escape_dir2.mkdir()
            backup_dir = tmp_path / "logs_original_after_swap"
            swap_done = {"done": False}

            def _swap_logs_for_symlink(_dir_fd: int) -> None:
                logs_dir.rename(backup_dir)
                logs_dir.symlink_to(escape_dir2)
                swap_done["done"] = True

            rc = run_capturing_outputs(
                campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="race2.log",
                stderr_name=None, combine_stderr=False,
                argv=[sys.executable, "-c", "print('race window 2 payload')"],
                _test_hook_after_open=_swap_logs_for_symlink,
            )
            published_in_original = backup_dir / "race2.log"
            escape_received_anything = any(escape_dir2.iterdir())
            rec.check(
                "swapping logs/ for a symlink after the directory descriptor is already open "
                "does not redirect the child's write to the symlink target "
                "(race window 2, injectable hook)",
                swap_done["done"] and rc == 0 and not escape_received_anything
                and published_in_original.exists()
                and published_in_original.read_text() == "race window 2 payload\n",
                detail=(
                    f"swap_done={swap_done['done']} rc={rc} "
                    f"escape_received_anything={escape_received_anything} "
                    f"published_in_original.exists()={published_in_original.exists()}"
                ),
            )
            logs_dir.unlink()
            shutil.rmtree(backup_dir, ignore_errors=True)
            logs_dir.mkdir()

            # --- write subcommand: happy path + no-clobber ---
            write_stdin_safely(
                campaign_dir_rel=campaign_rel, rel_dir="logs", name="written.csv",
                content=b"a,b\n1,2\n",
            )
            rec.check(
                "write_stdin_safely publishes stdin content to a new anchored file",
                (logs_dir / "written.csv").read_bytes() == b"a,b\n1,2\n",
            )
            raised = False
            try:
                write_stdin_safely(
                    campaign_dir_rel=campaign_rel, rel_dir="logs", name="written.csv",
                    content=b"tampered\n",
                )
            except UnsafeCaptureError:
                raised = True
            rec.check(
                "write_stdin_safely refuses to overwrite an existing file, unchanged",
                raised and (logs_dir / "written.csv").read_bytes() == b"a,b\n1,2\n",
            )
            raised = False
            try:
                write_stdin_safely(
                    campaign_dir_rel=campaign_rel, rel_dir="logs", name="empty.csv", content=b"",
                )
            except UnsafeCaptureError:
                raised = True
            rec.check(
                "write_stdin_safely refuses to publish empty content",
                raised and not (logs_dir / "empty.csv").exists(),
            )

            # --- verify subcommand: genuine file, missing file, empty file, symlink ---
            profiles_dir = tmp_path / campaign_rel / "profiles" / case_name
            (profiles_dir / f"{case_name}_report.ncu-rep").write_bytes(b"synthetic ncu-rep bytes\n")
            (profiles_dir / f"{case_name}.ncu_tool.log").write_text("tool log\n")
            dir_fd = resolve_campaign_rel_dir_fd(campaign_rel, f"profiles/{case_name}")
            try:
                errs = [
                    verify_regular_file_in_dir(dir_fd, n)
                    for n in (f"{case_name}_report.ncu-rep", f"{case_name}.ncu_tool.log")
                ]
            finally:
                os.close(dir_fd)
            rec.check(
                "verify_regular_file_in_dir accepts genuine non-symlink non-empty files",
                errs == [None, None],
                detail=f"errs={errs}",
            )
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
            rec.check(
                "verify_regular_file_in_dir rejects a missing, an empty, and a symlinked name",
                err_missing is not None and err_empty is not None and err_symlink is not None,
                detail=f"missing={err_missing!r} empty={err_empty!r} symlink={err_symlink!r}",
            )

            # --- rel-dir allowlist: an unlisted directory name is rejected outright ---
            raised = False
            try:
                resolve_campaign_rel_dir_fd(campaign_rel, "not_a_real_subdir")
            except UnsafeCaptureError:
                raised = True
            rec.check(
                "resolve_campaign_rel_dir_fd rejects a rel-dir outside the frozen allowlist",
                raised,
            )

            # --- mkdir-case: the descriptor-anchored replacement for the old
            # "[ -L ] || [ -e ]; mkdir" precheck-then-create pattern, which was
            # itself racy against profiles/ (the *parent*) being swapped ---
            second_case_name = p14.build_ncu_plan()[1]["case_name"]
            profiles_root = tmp_path / campaign_rel / "profiles"
            fd = resolve_profiles_root_fd(campaign_rel)
            try:
                mkdir_case_dir(fd, second_case_name)
            finally:
                os.close(fd)
            rec.check(
                "mkdir_case_dir creates exactly the named frozen case directory",
                (profiles_root / second_case_name).is_dir()
                and not (profiles_root / second_case_name).is_symlink(),
            )
            raised = False
            fd = resolve_profiles_root_fd(campaign_rel)
            try:
                mkdir_case_dir(fd, second_case_name)
            except UnsafeCaptureError:
                raised = True
            finally:
                os.close(fd)
            rec.check(
                "mkdir_case_dir refuses to recreate an already-existing case directory",
                raised,
            )
            raised = False
            fd = resolve_profiles_root_fd(campaign_rel)
            try:
                mkdir_case_dir(fd, "not_one_of_the_six_frozen_cases")
            except UnsafeCaptureError:
                raised = True
            finally:
                os.close(fd)
            rec.check(
                "mkdir_case_dir refuses a case name outside the frozen six-case plan",
                raised,
            )
            # profiles/ itself replaced by a symlink: resolve_profiles_root_fd
            # must reject it exactly like resolve_campaign_rel_dir_fd does for
            # logs/, closing the same class of race for case-directory
            # creation.
            profiles_escape = tmp_path / "profiles_escape_root"
            profiles_escape.mkdir()
            profiles_backup = tmp_path / "profiles_backup"
            profiles_root.rename(profiles_backup)
            profiles_root.symlink_to(profiles_escape)
            raised = False
            try:
                resolve_profiles_root_fd(campaign_rel)
            except UnsafeCaptureError:
                raised = True
            rec.check(
                "resolve_profiles_root_fd rejects a symlinked profiles/ directory",
                raised,
            )
            profiles_root.unlink()
            profiles_backup.rename(profiles_root)

            # -----------------------------------------------------------
            # Task 4 remediation (Remediation E): strict single-component
            # basename validation. Every one of these traversal attempts
            # previously reached publish_no_clobber()/verify_regular_file_
            # in_dir() with no rejection at all -- publish_no_clobber() in
            # particular passed final_name straight to os.link(dst_dir_fd=
            # dir_fd), and linkat() resolves a relative newpath with ".."
            # components, or ignores dst_dir_fd outright for an absolute
            # newpath, exactly like any other path argument. Demonstrated
            # against the pre-remediation module (git commit 49be85b): a
            # direct publish_no_clobber(dir_fd, <owned partial>,
            # "../escape_relative.bin") call wrote a file one level *above*
            # the anchored logs/ directory, and a --stdout-name of an
            # absolute path wrote outside the campaign entirely -- both
            # reproduced and confirmed during this remediation before the
            # _validate_basename() checks below were added. For every case
            # here, the assertion itself also proves no file was created
            # anywhere outside the anchored directory.
            # -----------------------------------------------------------
            traversal_names = [
                "../escape.bin",
                "subdir/escape.bin",
                "/absolute/escape.bin",
                ".",
                "..",
                "",
                "name\x00withnul",
                "name\nwithnewline",
            ]
            outside_root = tmp_path / "traversal_outside_root"
            outside_root.mkdir()
            for bad_name in traversal_names:
                before_outside = set(outside_root.iterdir())
                before_logs = set(logs_dir.iterdir())
                raised = False
                try:
                    write_stdin_safely(
                        campaign_dir_rel=campaign_rel, rel_dir="logs", name=bad_name,
                        content=b"attacker payload",
                    )
                except UnsafeCaptureError:
                    raised = True
                after_outside = set(outside_root.iterdir())
                after_logs = set(logs_dir.iterdir()) - before_logs
                rec.check(
                    f"write_stdin_safely rejects traversal/invalid name {bad_name!r} and creates "
                    f"nothing outside logs/ or anywhere unexpected inside it",
                    raised and after_outside == before_outside and after_logs == set(),
                    detail=f"raised={raised} after_outside={after_outside} after_logs={after_logs}",
                )

            # publish_no_clobber() itself (not just the higher-level
            # write_stdin_safely/run_capturing_outputs wrappers) must
            # refuse a traversal final_name -- this is the exact function
            # and exact call shape the original audit finding reproduced.
            dir_fd = resolve_campaign_rel_dir_fd(campaign_rel, "logs")
            try:
                _partial_fd, _partial_name_ = create_partial(dir_fd, "safe_output_for_traversal_test.log")
                os.write(_partial_fd, b"attacker payload via relative traversal\n")
                os.close(_partial_fd)
                raised = False
                try:
                    publish_no_clobber(dir_fd, _partial_name_, "../escape_relative_direct.bin")
                except UnsafeCaptureError:
                    raised = True
            finally:
                os.close(dir_fd)
            rec.check(
                "publish_no_clobber() itself rejects a '../...' final_name and writes nothing "
                "one level above the anchored directory",
                raised and not (tmp_path / campaign_rel / "escape_relative_direct.bin").exists(),
            )
            # The owned (still-partial) evidence from the attempt above must
            # not itself be treated as silently lost: it remains under its
            # own partial name inside logs/, available as forensic evidence,
            # since publish_no_clobber() only ever unlinks the partial on
            # its own successful publish path.
            leftover_partials = [p for p in logs_dir.iterdir() if "safe_output_for_traversal_test" in p.name]
            rec.check(
                "the owned partial from a rejected publish attempt is preserved, not silently lost",
                len(leftover_partials) == 1,
                detail=f"leftover_partials={leftover_partials}",
            )
            for p in leftover_partials:
                p.unlink()

            # An absolute --stdout-name reaches run_capturing_outputs's own
            # basename check before resolve_campaign_rel_dir_fd or any
            # output is ever created; linkat()/os.link() ignore dst_dir_fd
            # entirely for an absolute newpath, so this was reachable even
            # though create_partial()'s own accidental "." + name mangling
            # happens to make some (not all) relative ".." attempts fail
            # for an unrelated reason (a nonexistent intermediate
            # directory component) rather than by design.
            absolute_evil = outside_root / "escape_via_stdout_name.bin"
            raised = False
            try:
                run_capturing_outputs(
                    campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name=str(absolute_evil),
                    stderr_name=None, combine_stderr=False,
                    argv=[sys.executable, "-c", "print('attacker payload')"],
                )
            except UnsafeCaptureError:
                raised = True
            rec.check(
                "run_capturing_outputs rejects an absolute --stdout-name before creating anything",
                raised and not absolute_evil.exists(),
            )

            # -----------------------------------------------------------
            # Task 4 remediation (Remediation E): failure-cleanup edge
            # cases. Before this remediation, any exception raised before
            # or during the child's own launch (a nonexistent executable,
            # a failure creating the *second* output after the first
            # already existed on disk, or the injectable test hook itself
            # raising) propagated straight out of run_capturing_outputs
            # without ever reaching the "was the partial left empty?"
            # cleanup logic -- reproduced and confirmed against the
            # pre-remediation module below via a nonexistent executable,
            # which left a stale, empty, uniquely-named ".partial" file
            # behind forever.
            # -----------------------------------------------------------
            before = set(logs_dir.iterdir())
            raised = False
            try:
                run_capturing_outputs(
                    campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="never_launched.log",
                    stderr_name="never_launched.stderr.log", combine_stderr=False,
                    argv=["/definitely/does/not/exist/p14-self-test-binary-xyz"],
                )
            except UnsafeCaptureError:
                raised = True
            after = set(logs_dir.iterdir()) - before
            rec.check(
                "a nonexistent executable raises UnsafeCaptureError and leaves no stale partial "
                "for either stdout or stderr",
                raised and after == set(),
                detail=f"after={sorted(p.name for p in after)}",
            )

            before = set(logs_dir.iterdir())
            hook_calls = {"n": 0}

            def _raising_hook(_dir_fd: int) -> None:
                hook_calls["n"] += 1
                raise UnsafeCaptureError("synthetic pre-launch hook failure (self-test)")

            raised = False
            try:
                run_capturing_outputs(
                    campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="hook_failure.log",
                    stderr_name=None, combine_stderr=False,
                    argv=[sys.executable, "-c", "print('should never run')"],
                    _test_hook_after_open=_raising_hook,
                )
            except UnsafeCaptureError:
                raised = True
            after = set(logs_dir.iterdir()) - before
            rec.check(
                "a pre-launch test hook that raises leaves no stale partial and the child never runs",
                raised and hook_calls["n"] == 1 and after == set(),
                detail=f"after={sorted(p.name for p in after)}",
            )

            real_create_partial = create_partial
            flaky_state = {"n": 0}

            def _flaky_create_partial(dir_fd_arg: int, final_name_arg: str):
                flaky_state["n"] += 1
                if flaky_state["n"] == 2:
                    raise UnsafeCaptureError("synthetic second-output creation failure (self-test)")
                return real_create_partial(dir_fd_arg, final_name_arg)

            before = set(logs_dir.iterdir())
            raised = False
            with mock.patch.object(sys.modules[__name__], "create_partial", side_effect=_flaky_create_partial):
                try:
                    run_capturing_outputs(
                        campaign_dir_rel=campaign_rel, rel_dir="logs", stdout_name="first_output_ok.log",
                        stderr_name="second_output_fails.log", combine_stderr=False,
                        argv=[sys.executable, "-c", "print('should never run')"],
                    )
                except UnsafeCaptureError:
                    raised = True
            after = set(logs_dir.iterdir()) - before
            rec.check(
                "a failure creating the second output after the first already exists cleans up "
                "the first (now-empty) partial and leaves no stale file",
                raised and flaky_state["n"] == 2 and after == set(),
                detail=f"after={sorted(p.name for p in after)}",
            )

            # Concurrent final-target creation: something else creates the
            # final name *after* this call's own descriptors are already
            # open (via the same injectable, synchronous hook the race-
            # window-2 test above uses) but *before* the child writes.
            # publish_no_clobber()'s linkat()-based EEXIST is the sole,
            # atomic guarantee here -- no separate existence pre-check to
            # race against.
            concurrent_target = logs_dir / "concurrent_final_target.log"

            def _create_concurrent_target(_dir_fd: int) -> None:
                concurrent_target.write_text("written by a concurrent process\n")

            before = set(logs_dir.iterdir())
            raised = False
            try:
                run_capturing_outputs(
                    campaign_dir_rel=campaign_rel, rel_dir="logs",
                    stdout_name="concurrent_final_target.log", stderr_name=None, combine_stderr=False,
                    argv=[sys.executable, "-c", "print('our own successful output')"],
                    _test_hook_after_open=_create_concurrent_target,
                )
            except UnsafeCaptureError:
                raised = True
            after = set(logs_dir.iterdir()) - before
            surviving_partials = [p for p in after if p.name != "concurrent_final_target.log"]
            rec.check(
                "publication refused when the final target appeared concurrently; the "
                "concurrently-written target is untouched and our own non-empty output survives "
                "as partial evidence",
                raised and concurrent_target.read_text() == "written by a concurrent process\n"
                and len(surviving_partials) == 1
                and surviving_partials[0].read_text() == "our own successful output\n",
                detail=f"after={sorted(p.name for p in after)}",
            )
            for p in after:
                p.unlink()

            # -----------------------------------------------------------
            # Task 4 remediation (Remediation A/B): the NCU bundle format
            # and the "publish-bundle" subcommand that decodes an already-
            # captured scripts/p14_ncu_bridge.py bundle and republishes its
            # six segments under their real names, no-clobber.
            # -----------------------------------------------------------
            sample_segments = {
                "app_stdout": b"application stdout bytes\n",
                "app_stderr": b"",
                "ncu_tool_log": b"fake ncu tool log\n",
                "ncu_rep": b"\x00\x01FAKE_NCU_REP\xff\xfe" + NCU_BUNDLE_MAGIC,
                "metrics_csv": b"ID,Kernel Name,Metric Name,Metric Unit,Metric Value\n0,k,m,byte,1\n",
                "metrics_export_stderr": b"",
            }
            encoded = encode_ncu_bundle(sample_segments)
            decoded = decode_ncu_bundle(encoded)
            rec.check(
                "encode_ncu_bundle/decode_ncu_bundle round-trip is exact, including an empty "
                "segment and a segment whose own content contains the magic marker",
                decoded == sample_segments, detail=f"decoded={decoded!r}",
            )
            noisy = b"run_container: selected index=3 uuid=GPU-xxxx name='fake' driver=1.0\n" + encoded
            rec.check(
                "decode_ncu_bundle tolerates leading bytes before the magic marker",
                decode_ncu_bundle(noisy) == sample_segments,
            )
            raised = False
            try:
                decode_ncu_bundle(b"not a bundle at all")
            except NcuBundleParseError:
                raised = True
            rec.check("decode_ncu_bundle rejects a stream with no magic marker at all", raised)
            raised = False
            try:
                decode_ncu_bundle(encoded[: len(encoded) - 3])
            except NcuBundleParseError:
                raised = True
            rec.check("decode_ncu_bundle rejects a truncated bundle", raised)

            # Uses second_case_name's directory (created empty, above, by the
            # mkdir_case_dir tests) rather than the first case_name's, whose
            # directory already holds artifacts from the verify_regular_
            # file_in_dir tests earlier in this self-test.
            bundle_case_name = second_case_name
            bundle_case_dir = tmp_path / campaign_rel / "profiles" / bundle_case_name
            bundle_case_rel = f"profiles/{bundle_case_name}"
            bundle_script = tmp_path / "fake_bundle_emitter.py"
            bundle_script.write_text(
                "import sys\n"
                "sys.stdout.buffer.write(" + repr(encoded) + ")\n",
                encoding="utf-8",
            )
            rc = run_capturing_outputs(
                campaign_dir_rel=campaign_rel, rel_dir=bundle_case_rel, stdout_name="raw_bundle.bin",
                stderr_name=None, combine_stderr=False, argv=[sys.executable, str(bundle_script)],
            )
            output_names = [
                f"{bundle_case_name}.container_stdout.log", f"{bundle_case_name}.container_stderr.log",
                f"{bundle_case_name}.ncu_tool.log", f"{bundle_case_name}_report.ncu-rep",
                f"{bundle_case_name}.metrics_raw.csv", f"{bundle_case_name}.metrics_export_stderr.log",
            ]
            publish_ncu_bundle(
                campaign_dir_rel=campaign_rel, rel_dir=bundle_case_rel,
                bundle_name="raw_bundle.bin", output_names=output_names,
            )
            published_contents = {name: (bundle_case_dir / name).read_bytes() for name in output_names}
            rec.check(
                "publish-bundle republishes all six segments under their real names with exact "
                "byte-for-byte content",
                rc == 0
                and published_contents[output_names[0]] == sample_segments["app_stdout"]
                and published_contents[output_names[1]] == sample_segments["app_stderr"]
                and published_contents[output_names[2]] == sample_segments["ncu_tool_log"]
                and published_contents[output_names[3]] == sample_segments["ncu_rep"]
                and published_contents[output_names[4]] == sample_segments["metrics_csv"]
                and published_contents[output_names[5]] == sample_segments["metrics_export_stderr"],
            )
            rec.check(
                "publish-bundle removes the raw transport bundle after successfully republishing it",
                not (bundle_case_dir / "raw_bundle.bin").exists(),
            )

            # Malformed bundle: no output is created at all.
            malformed_script = tmp_path / "fake_malformed_bundle_emitter.py"
            malformed_script.write_text(
                "import sys\nsys.stdout.buffer.write(b'not a real bundle')\n", encoding="utf-8",
            )
            run_capturing_outputs(
                campaign_dir_rel=campaign_rel, rel_dir=bundle_case_rel, stdout_name="raw_bundle_bad.bin",
                stderr_name=None, combine_stderr=False, argv=[sys.executable, str(malformed_script)],
            )
            before = set(bundle_case_dir.iterdir())
            raised = False
            try:
                publish_ncu_bundle(
                    campaign_dir_rel=campaign_rel, rel_dir=bundle_case_rel,
                    bundle_name="raw_bundle_bad.bin", output_names=output_names,
                )
            except UnsafeCaptureError:
                raised = True
            after = set(bundle_case_dir.iterdir())
            rec.check(
                "publish-bundle rejects a malformed bundle and publishes none of the six outputs",
                raised and after == before,
                detail=f"before={sorted(p.name for p in before)} after={sorted(p.name for p in after)}",
            )
            (bundle_case_dir / "raw_bundle_bad.bin").unlink()

            # --names must be exactly six values, and every bundle_name/
            # output_name is itself validated as a strict basename.
            raised = False
            try:
                publish_ncu_bundle(
                    campaign_dir_rel=campaign_rel, rel_dir=bundle_case_rel,
                    bundle_name="raw_bundle.bin", output_names=output_names[:5],
                )
            except UnsafeCaptureError:
                raised = True
            rec.check("publish-bundle rejects a --names list of the wrong length", raised)
            raised = False
            try:
                publish_ncu_bundle(
                    campaign_dir_rel=campaign_rel, rel_dir=bundle_case_rel,
                    bundle_name="../escape_bundle.bin", output_names=output_names,
                )
            except UnsafeCaptureError:
                raised = True
            rec.check("publish-bundle rejects a traversal --bundle-name", raised)
            raised = False
            try:
                publish_ncu_bundle(
                    campaign_dir_rel=campaign_rel, rel_dir=bundle_case_rel,
                    bundle_name="raw_bundle.bin",
                    output_names=["../escape.log"] + output_names[1:],
                )
            except UnsafeCaptureError:
                raised = True
            rec.check("publish-bundle rejects a traversal entry inside --names", raised)

    if rec.failures:
        print(
            f"p14_safe_capture: self-test: FAILED ({len(rec.failures)}/{rec.total} case(s)): {rec.failures}",
            file=sys.stderr,
        )
        print("p14_safe_capture: SELF_TEST_RESULT=FAIL", file=sys.stderr)
        return 1
    print(f"p14_safe_capture: self-test: OK ({rec.total} cases)", file=sys.stderr)
    print("p14_safe_capture: SELF_TEST_RESULT=PASS", file=sys.stderr)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="p14_safe_capture.py",
        description="P1.4-only descriptor-anchored safe command capture (see module docstring).",
    )
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

    verify_parser = subparsers.add_parser(
        "verify", help="Confirm one or more names are genuine files strictly inside the anchored directory.",
    )
    verify_parser.add_argument("--campaign-dir", required=True)
    verify_parser.add_argument("--rel-dir", required=True)
    verify_parser.add_argument("--name", action="append", required=True, dest="name")
    verify_parser.set_defaults(func=cmd_verify)

    mkdir_parser = subparsers.add_parser(
        "mkdir-case", help="Safely create one of the six frozen profiles/<case> directories.",
    )
    mkdir_parser.add_argument("--campaign-dir", required=True)
    mkdir_parser.add_argument("--case-name", required=True)
    mkdir_parser.set_defaults(func=cmd_mkdir_case)

    bundle_parser = subparsers.add_parser(
        "publish-bundle",
        help="Decode an already-captured scripts/p14_ncu_bridge.py bundle and publish its "
             "six segments under their real names, no-clobber.",
    )
    bundle_parser.add_argument("--campaign-dir", required=True)
    bundle_parser.add_argument("--rel-dir", required=True)
    bundle_parser.add_argument("--bundle-name", required=True)
    bundle_parser.add_argument(
        "--names", required=True, nargs=len(NCU_BUNDLE_SEGMENT_NAMES),
        metavar=tuple(n.upper() for n in NCU_BUNDLE_SEGMENT_NAMES),
        help="Exactly " + str(len(NCU_BUNDLE_SEGMENT_NAMES)) + " output names, in fixed order "
             + str(NCU_BUNDLE_SEGMENT_NAMES) + ".",
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
