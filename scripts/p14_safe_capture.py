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

Subcommands:
  run     Execute ARGV (after "--", shell=False) with stdout/stderr
          connected directly to newly created, descriptor-anchored,
          exclusive, no-follow partial files under
          <campaign-dir>/<rel-dir>/; on a zero exit, publishes each
          partial to its final name via no-clobber hard link; on a
          non-zero exit, preserves a non-empty partial under its unique
          name and removes only an empty owned partial. Exits with the
          child's own return code (or 2 for a tool-level path-safety or
          publish failure).
  write   Reads bytes from stdin and safely publishes them as one new,
          no-clobber file under <campaign-dir>/<rel-dir>/. Used for the
          one P1.4 output that is not literally a command's own
          stdout/stderr (the application CSV recovered from a profiled
          binary's captured stdout).
  verify  Confirms that one or more names already exist as genuine
          non-symlink, non-empty regular files strictly within
          <campaign-dir>/<rel-dir>/, reopening every directory component
          the same descriptor-anchored way. Used after Nsight Compute
          itself writes ".ncu-rep"/".ncu_tool.log" directly (via its own
          "-o"/"--log-file" arguments, which this module cannot intercept
          -- NCU resolves those paths itself) to confirm neither escaped
          the anchored directory, instead of trusting a plain "test -f".

Exit codes: 0 success; the child's own exit code for a "run" whose command
ran but exited non-zero; 1 for "write"/"verify" content/evidence failures;
2 for a path-safety, argument, or capture-mechanism failure.
"""

from __future__ import annotations

import argparse
import errno
import os
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
    if not argv:
        raise UnsafeCaptureError("no command given after '--'")

    dir_fd = resolve_campaign_rel_dir_fd(campaign_dir_rel, rel_dir)
    try:
        stdout_fd = stdout_partial = None
        stderr_fd = stderr_partial = None
        stdout_stat = stderr_stat = None
        try:
            if stdout_name is not None:
                stdout_fd, stdout_partial = create_partial(dir_fd, stdout_name)
            if combine_stderr:
                stderr_fd = stdout_fd
            elif stderr_name is not None:
                stderr_fd, stderr_partial = create_partial(dir_fd, stderr_name)

            if _test_hook_after_open is not None:
                _test_hook_after_open(dir_fd)

            try:
                result = subprocess.run(
                    list(argv),
                    stdout=stdout_fd,
                    stderr=stderr_fd,
                    stdin=subprocess.DEVNULL,
                    shell=False,
                )
            except OSError as exc:
                raise UnsafeCaptureError(f"could not launch command: {exc}") from exc
        finally:
            # fstat before close: a later "was the partial left empty?" check
            # (only relevant on a non-zero exit) must not depend on a fd that
            # is no longer open.
            if stdout_fd is not None:
                stdout_stat = os.fstat(stdout_fd)
                os.close(stdout_fd)
            if stderr_fd is not None and not combine_stderr:
                stderr_stat = os.fstat(stderr_fd)
                os.close(stderr_fd)

        errors: list[str] = []
        if result.returncode == 0:
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
