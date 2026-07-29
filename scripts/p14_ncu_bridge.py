#!/usr/bin/env python3
"""P1.4-only container-side NCU bridge (Task 4 remediation, blockers A/B).

Runs INSIDE the pinned container, launched through the unmodified
scripts/run_container.sh exactly like every other GPU-touching P1.4 step.
Never receives, constructs, or opens a campaign-relative or repository
raw-tree pathname: every NCU "-o" / "--log-file" / "--import" argument this
script builds points exclusively inside a private directory freshly created
under the container's own, non-host-mounted "/tmp" (scripts/run_container.sh
only ever bind-mounts the repository itself at "/workspace"; nothing under
"/tmp" is shared with the host or with any other container invocation, and
the container is destroyed on exit -- "docker run --rm" -- so nothing here
outlives the single collection it performs).

This is the structural fix for the finding that the previous design let NCU
itself resolve and write "-o"/"--log-file" pathnames built from
"results/raw/exp01_memory_paths_p14/<campaign_id>/profiles/<case>/...", and
let a second, separate `docker run` pass that same campaign's ".ncu-rep"
path to NCU's own "--import" for the metrics-export step. Whatever NCU did
with those arguments -- including a bug in this project's own path
construction, such as building "profiles/<case>/<case>_report" relative to
"/workspace" instead of the campaign directory -- was a path NCU itself
opened for writing. This script removes that possibility structurally: NCU
is never given any argument derived from the host raw tree, ever, in either
of the two invocations it runs.

Design (matches src/memory/P1_4_PROTOCOL.md's "container-private NCU output
staging" section):
  1. Runs inside the container (via scripts/run_container.sh, unmodified).
  2. Creates a private directory in the container's own, non-mounted "/tmp".
  3. Gives NCU only paths inside that private directory for "-o",
     "--log-file", and "--import".
  4. Verifies every expected output is a genuine, non-empty regular file.
  5. Emits a versioned, length-delimited bundle (see p14_safe_capture's
     NCU_BUNDLE_* helpers) containing exactly: application stdout,
     application stderr, the NCU tool log, the raw ".ncu-rep" bytes, the
     exported "metrics_raw.csv", and the metric-export step's stderr.
  6/7/8. The host side (scripts/run_exp01_memory_paths_p14.sh) captures
     this script's stdout through p14_safe_capture.py's "run" subcommand
     (an already-open, descriptor-anchored partial file, exactly like every
     other P1.4 child-process capture) and then decodes/publishes it via
     "publish-bundle", which republishes every artifact into the anchored
     case-directory descriptor with no-follow/no-clobber operations.
  9. This script always deletes its own private directory before exiting
     (belt-and-suspenders: the container itself is destroyed on exit by
     scripts/run_container.sh's "--rm", which already destroys "/tmp" with
     it).

This script takes no "--campaign-dir"-shaped argument at all, by design, so
it cannot structurally be pointed at a raw-tree path even if every
argument it is given were adversarial.

Usage:
  p14_ncu_bridge.py --metrics M1,M2,... --kernel-name NAME -- <bin> [args...]
  p14_ncu_bridge.py --self-test
  p14_ncu_bridge.py --help

Exit codes: 0 on success (the bundle is written to stdout, and stdout
contains nothing else); 1 if NCU collection or export failed, or an
expected private-directory output was missing/empty (nothing is written to
stdout in this case -- only diagnostics on stderr); 2 on a usage error.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# Only the bundle encode/decode helpers are used from here (pure functions,
# no path-safety calls against any repository tree are ever made from
# inside the container by this script).
import p14_safe_capture as capture_lib  # noqa: E402

# Verified against the pinned image's own NCU 2025.4.0.0 --help output
# during the original P1.4 implementation (see src/memory/P1_4_PROTOCOL.md
# Section 4). Never a forced-overwrite flag, never the full metric set,
# never a clock-controlling default.
NCU_COLLECTION_FLAGS = (
    "--clock-control", "none",
    "--pipeline-boost-state", "dynamic",
    "--cache-control", "none",
    "--kernel-name-base", "function",
    "--launch-count", "1",
    "--devices", "0",
    "--replay-mode", "kernel",
    "--print-summary", "none",
)
# `--print-metric-name` is intentionally absent. NCU 2025.4 accepts that
# option only for the details page and exits 1 when it is combined with
# `--page raw`; the raw page already emits actual metric identifiers as
# wide-table column names. `--csv` implies base units, but keep
# `--print-units base` explicit because the parser's unit contract is
# intentionally frozen.
NCU_EXPORT_FLAGS = (
    "--csv", "--page", "raw",
    "--print-units", "base",
    "--print-kernel-base", "function",
)


class BridgeError(RuntimeError):
    pass


def _bounded_diagnostic(path: Path, *, limit: int = 4096) -> str:
    """Returns a bounded, escaped rendering of an NCU diagnostic stream.

    NCU can report CLI errors on stdout even when stderr is empty. The
    private directory is always deleted, so include both streams in the
    bridge's own stderr before cleanup. `repr` keeps control characters from
    altering the surrounding log, while the byte limit prevents an
    unexpectedly large profiler response from flooding it.
    """
    try:
        data = path.read_bytes()
    except OSError as exc:
        return f"<unreadable: {exc}>"
    if not data:
        return "<empty>"
    truncated = len(data) > limit
    data = data[:limit]
    text = data.decode("utf-8", errors="backslashreplace")
    suffix = f" <truncated after {limit} bytes>" if truncated else ""
    return f"{text!r}{suffix}"


def _require_nonempty_file(path: Path, label: str) -> None:
    """Uses lstat-equivalent semantics (Path.is_symlink()/is_file() do not
    themselves follow a symlink into acceptance -- is_file() on a symlink
    reports the *target*, so a symlink is explicitly rejected first) since
    this private directory, although container-local and short-lived, is
    still the boundary between "NCU wrote this" and "this script trusts
    it": NCU's own -o/--log-file targets are always freshly created here,
    never pre-existing, but failing closed on an unexpected symlink costs
    nothing and assumes nothing about NCU's own behavior."""
    if path.is_symlink():
        raise BridgeError(f"{label}: expected NCU output is a symlink, refusing: {path}")
    if not path.is_file():
        raise BridgeError(f"{label}: expected NCU output does not exist as a regular file: {path}")
    if path.stat().st_size == 0:
        raise BridgeError(f"{label}: expected NCU output is empty: {path}")


def run_bridge(
    *, ncu_binary: str, metrics: str, kernel_name: str, benchmark_argv: list[str],
    tmp_root: str = "/tmp",
) -> bytes:
    """Runs NCU collection then a GPU-free metrics export entirely inside a
    fresh private directory under tmp_root (always the container's own
    "/tmp" in real use; overridable only for --self-test, never via a CLI
    flag), and returns the encoded bundle bytes. Raises BridgeError with a
    human-readable reason on any failure; never partially writes a bundle
    -- either every one of the six segments is present and this function
    returns normally, or it raises and the caller (main()) prints nothing
    to stdout at all."""
    if not benchmark_argv:
        raise BridgeError("no benchmark command given after '--'")
    if not metrics:
        raise BridgeError("--metrics must be a non-empty comma-separated metric list")
    if not kernel_name:
        raise BridgeError("--kernel-name must be non-empty")

    private_dir = Path(tempfile.mkdtemp(prefix="p14_ncu_bridge_", dir=tmp_root))
    try:
        report_base = private_dir / "report"
        ncu_rep_path = private_dir / "report.ncu-rep"
        ncu_tool_log_path = private_dir / "report.ncu_tool.log"
        app_stdout_path = private_dir / "app_stdout.bin"
        app_stderr_path = private_dir / "app_stderr.bin"
        metrics_csv_path = private_dir / "metrics_raw.csv"
        metrics_export_stderr_path = private_dir / "metrics_export_stderr.bin"

        collect_argv = [
            ncu_binary, *NCU_COLLECTION_FLAGS,
            "--kernel-name", kernel_name,
            "--metrics", metrics,
            "--log-file", str(ncu_tool_log_path),
            "-o", str(report_base),
            "--", *benchmark_argv,
        ]
        with open(app_stdout_path, "wb") as out_f, open(app_stderr_path, "wb") as err_f:
            try:
                collect_result = subprocess.run(
                    collect_argv, stdout=out_f, stderr=err_f, stdin=subprocess.DEVNULL, shell=False,
                )
            except OSError as exc:
                raise BridgeError(f"could not launch NCU collection ({ncu_binary!r}): {exc}") from exc
        if collect_result.returncode != 0:
            raise BridgeError(
                f"NCU collection exited {collect_result.returncode}; not bundling any output for "
                f"this case (application stdout/stderr and the NCU tool log are only ever "
                f"emitted together with a successful collection and export)"
            )
        _require_nonempty_file(ncu_rep_path, "collection: .ncu-rep")
        _require_nonempty_file(ncu_tool_log_path, "collection: NCU tool log")
        _require_nonempty_file(app_stdout_path, "collection: application stdout")

        export_argv = [ncu_binary, "--import", str(ncu_rep_path), *NCU_EXPORT_FLAGS]
        with open(metrics_csv_path, "wb") as out_f, open(metrics_export_stderr_path, "wb") as err_f:
            try:
                export_result = subprocess.run(
                    export_argv, stdout=out_f, stderr=err_f, stdin=subprocess.DEVNULL, shell=False,
                )
            except OSError as exc:
                raise BridgeError(f"could not launch NCU metrics export ({ncu_binary!r}): {exc}") from exc
        if export_result.returncode != 0:
            raise BridgeError(
                f"NCU metrics export exited {export_result.returncode}; "
                f"stdout={_bounded_diagnostic(metrics_csv_path)}; "
                f"stderr={_bounded_diagnostic(metrics_export_stderr_path)}; "
                f"not bundling any output for this case"
            )
        _require_nonempty_file(metrics_csv_path, "export: metrics_raw.csv")

        segments = {
            "app_stdout": app_stdout_path.read_bytes(),
            "app_stderr": app_stderr_path.read_bytes(),
            "ncu_tool_log": ncu_tool_log_path.read_bytes(),
            "ncu_rep": ncu_rep_path.read_bytes(),
            "metrics_csv": metrics_csv_path.read_bytes(),
            "metrics_export_stderr": metrics_export_stderr_path.read_bytes(),
        }
        return capture_lib.encode_ncu_bundle(segments)
    finally:
        shutil.rmtree(private_dir, ignore_errors=True)


def usage() -> str:
    return (
        "Usage:\n"
        "  p14_ncu_bridge.py --metrics M1,M2,... --kernel-name NAME -- <bin> [args...]\n"
        "  p14_ncu_bridge.py --self-test\n"
        "  p14_ncu_bridge.py --help\n"
    )


def run_self_test() -> int:
    """GPU-free, Docker-free: exercises run_bridge() against fake "ncu"
    stand-ins (never the real NCU binary, never a real GPU or container)
    under an isolated tmp_root. ncu_binary/tmp_root are ordinary function
    parameters used only via direct in-process calls here -- never exposed
    as a production CLI flag -- exactly like check_ncu_help_capability()'s
    own synthetic-fixture self-test convention in
    scripts/run_exp01_memory_paths_p14.sh."""
    import tempfile as _tempfile

    total = 0
    failures: list[str] = []

    def check(label: str, condition: bool, *, detail: str = "") -> None:
        nonlocal total
        total += 1
        if condition:
            print(f"p14_ncu_bridge: self-test: PASS: {label}", file=sys.stderr)
        else:
            print(f"p14_ncu_bridge: self-test: FAIL: {label}; {detail}", file=sys.stderr)
            failures.append(label)

    fake_ncu_good = SCRIPT_DIR.parent / "__does_not_exist__"  # placeholder, replaced below
    with _tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        tmp_root = tmp_path / "container_tmp"
        tmp_root.mkdir()

        fake_ncu_good = tmp_path / "fake_ncu_good.py"
        fake_ncu_good.write_text(_FAKE_NCU_GOOD_SOURCE, encoding="utf-8")
        fake_ncu_missing_rep = tmp_path / "fake_ncu_missing_rep.py"
        fake_ncu_missing_rep.write_text(_FAKE_NCU_MISSING_REP_SOURCE, encoding="utf-8")
        fake_ncu_export_fails = tmp_path / "fake_ncu_export_fails.py"
        fake_ncu_export_fails.write_text(_FAKE_NCU_EXPORT_FAILS_SOURCE, encoding="utf-8")
        fake_ncu_collect_fails = tmp_path / "fake_ncu_collect_fails.py"
        fake_ncu_collect_fails.write_text(_FAKE_NCU_COLLECT_FAILS_SOURCE, encoding="utf-8")
        for p in (fake_ncu_good, fake_ncu_missing_rep, fake_ncu_export_fails, fake_ncu_collect_fails):
            p.chmod(0o700)

        fake_bin = tmp_path / "fake_benchmark.py"
        fake_bin.write_text(
            "#!/usr/bin/env python3\nprint('schema_version,fake\\n1,ok')\n", encoding="utf-8",
        )
        fake_bin.chmod(0o700)

        # --- happy path: a complete bundle with the exact six segments ---
        bundle = run_bridge(
            ncu_binary=str(fake_ncu_good), metrics="dram__bytes_read.sum", kernel_name="fake_kernel",
            benchmark_argv=[sys.executable, str(fake_bin)], tmp_root=str(tmp_root),
        )
        segments = capture_lib.decode_ncu_bundle(bundle)
        must_be_nonempty = {"app_stdout", "ncu_tool_log", "ncu_rep", "metrics_csv"}
        check(
            "a successful bridge run emits a decodable bundle with exactly the six fixed segments",
            set(segments) == set(capture_lib.NCU_BUNDLE_SEGMENT_NAMES)
            and all(segments[k] for k in must_be_nonempty),
            detail=f"keys={sorted(segments)} empty={[k for k, v in segments.items() if not v]}",
        )
        check(
            "the bundled .ncu-rep segment is the fake NCU's own report bytes",
            segments["ncu_rep"] == b"FAKE_NCU_REP_BYTES",
        )
        check(
            "the bundled metrics_csv segment is the fake NCU's own exported CSV",
            b"dram__bytes_read.sum" in segments["metrics_csv"],
        )
        check(
            "the raw-page export omits details-only --print-metric-name while retaining "
            "the frozen CSV/page/unit/kernel-name controls",
            "--print-metric-name" not in NCU_EXPORT_FLAGS
            and NCU_EXPORT_FLAGS == (
                "--csv", "--page", "raw",
                "--print-units", "base",
                "--print-kernel-base", "function",
            ),
            detail=f"flags={NCU_EXPORT_FLAGS!r}",
        )
        private_dirs_left = list(tmp_root.iterdir())
        check(
            "the private directory is removed before run_bridge() returns",
            private_dirs_left == [],
            detail=f"leftover={private_dirs_left}",
        )

        # --- collection failure: no bundle, private dir still cleaned up ---
        raised = None
        try:
            run_bridge(
                ncu_binary=str(fake_ncu_collect_fails), metrics="dram__bytes_read.sum",
                kernel_name="fake_kernel", benchmark_argv=[sys.executable, str(fake_bin)],
                tmp_root=str(tmp_root),
            )
        except BridgeError as exc:
            raised = str(exc)
        check(
            "a fake NCU that fails during collection raises BridgeError and leaves no bundle",
            raised is not None, detail=f"raised={raised!r}",
        )
        check(
            "the private directory is still removed after a collection failure",
            list(tmp_root.iterdir()) == [],
        )

        # --- a fake ncu that "succeeds" (exit 0) but never wrote .ncu-rep ---
        raised = None
        try:
            run_bridge(
                ncu_binary=str(fake_ncu_missing_rep), metrics="dram__bytes_read.sum",
                kernel_name="fake_kernel", benchmark_argv=[sys.executable, str(fake_bin)],
                tmp_root=str(tmp_root),
            )
        except BridgeError as exc:
            raised = str(exc)
        check(
            "a zero-exit NCU stand-in that never produced .ncu-rep is still rejected",
            raised is not None and ".ncu-rep" in raised, detail=f"raised={raised!r}",
        )

        # --- export step fails: no bundle produced ---
        raised = None
        try:
            run_bridge(
                ncu_binary=str(fake_ncu_export_fails), metrics="dram__bytes_read.sum",
                kernel_name="fake_kernel", benchmark_argv=[sys.executable, str(fake_bin)],
                tmp_root=str(tmp_root),
            )
        except BridgeError as exc:
            raised = str(exc)
        check(
            "a metrics-export failure raises BridgeError and preserves bounded stdout/stderr "
            "diagnostics in the bridge error",
            raised is not None
            and "export" in raised
            and "fake export stdout diagnostic" in raised
            and "fake export stderr diagnostic" in raised,
            detail=f"raised={raised!r}",
        )

        # --- no benchmark command at all ---
        raised = None
        try:
            run_bridge(
                ncu_binary=str(fake_ncu_good), metrics="dram__bytes_read.sum",
                kernel_name="fake_kernel", benchmark_argv=[], tmp_root=str(tmp_root),
            )
        except BridgeError as exc:
            raised = str(exc)
        check("an empty benchmark_argv is rejected before any process is launched", raised is not None)

        # --- nonexistent ncu binary: OSError is converted to BridgeError ---
        raised = None
        try:
            run_bridge(
                ncu_binary=str(tmp_path / "does_not_exist_ncu"), metrics="dram__bytes_read.sum",
                kernel_name="fake_kernel", benchmark_argv=[sys.executable, str(fake_bin)],
                tmp_root=str(tmp_root),
            )
        except BridgeError as exc:
            raised = str(exc)
        check(
            "a nonexistent NCU executable raises BridgeError rather than an uncaught OSError",
            raised is not None,
        )

        # --- bundle round-trip: encode then decode reproduces every segment byte-for-byte,
        # including a segment that itself contains the magic marker as *content* (proves
        # length-prefixing, not a delimiter scan, is what actually bounds each segment) ---
        tricky = {name: f"segment-{name}-payload".encode() for name in capture_lib.NCU_BUNDLE_SEGMENT_NAMES}
        tricky["ncu_rep"] = capture_lib.NCU_BUNDLE_MAGIC + b"\x00\x01binary\xffbytes" + capture_lib.NCU_BUNDLE_MAGIC
        encoded = capture_lib.encode_ncu_bundle(tricky)
        decoded = capture_lib.decode_ncu_bundle(encoded)
        check(
            "encode_ncu_bundle/decode_ncu_bundle round-trip is exact, even when a segment's own "
            "content contains the magic marker",
            decoded == tricky, detail=f"decoded={decoded!r}",
        )
        leading_noise = b"run_container: selected index=3 uuid=GPU-xxxx name='fake' driver=1.0\n" + encoded
        check(
            "decode_ncu_bundle tolerates arbitrary bytes (e.g. run_container.sh's own banner "
            "lines) before the magic marker",
            capture_lib.decode_ncu_bundle(leading_noise) == tricky,
        )
        truncated = encoded[: len(encoded) - 5]
        raised = None
        try:
            capture_lib.decode_ncu_bundle(truncated)
        except capture_lib.NcuBundleParseError as exc:
            raised = str(exc)
        check("decode_ncu_bundle rejects a truncated bundle rather than returning partial data", raised is not None)

    if failures:
        print(f"p14_ncu_bridge: self-test: FAILED ({len(failures)}/{total} case(s)): {failures}", file=sys.stderr)
        print("p14_ncu_bridge: SELF_TEST_RESULT=FAIL", file=sys.stderr)
        return 1
    print(f"p14_ncu_bridge: self-test: OK ({total} cases)", file=sys.stderr)
    print("p14_ncu_bridge: SELF_TEST_RESULT=PASS", file=sys.stderr)
    return 0


_FAKE_NCU_GOOD_SOURCE = """#!/usr/bin/env python3
import subprocess
import sys
args = sys.argv[1:]
if "--import" in args:
    rep_path = args[args.index("--import") + 1]
    with open(rep_path, "rb") as f:
        if not f.read():
            sys.exit(9)
    # Exact regression for NCU 2025.4: this option is details-page-only and
    # must never be passed with the raw page.
    if "--print-metric-name" in args:
        sys.stdout.write("==ERROR== Option '--print-metric-name' is only supported for the details page.\\n")
        sys.exit(1)
    required = ["--csv", "--page", "raw", "--print-units", "base", "--print-kernel-base", "function"]
    if any(item not in args for item in required):
        sys.stderr.write("fake export is missing a required raw-page flag\\n")
        sys.exit(8)
    sys.stdout.write("ID,Process ID,Process Name,Host Name,Kernel Name,Kernel Time,Context,Stream,dram__bytes_read.sum\\n")
    sys.stdout.write(",,,,,,,,byte\\n")
    sys.stdout.write("0,1234,fake,fake-host,fake_kernel,2026-Jul-28 00:00:00,1,7,12345\\n")
    sys.exit(0)
o_idx = args.index("-o")
report_base = args[o_idx + 1]
log_idx = args.index("--log-file")
log_path = args[log_idx + 1]
with open(report_base + ".ncu-rep", "wb") as f:
    f.write(b"FAKE_NCU_REP_BYTES")
with open(log_path, "w") as f:
    f.write("fake ncu tool log\\n")
# Real NCU launches the profiled binary as its own child, inheriting this
# process's stdout/stderr; simulate that so the bridge's own capture of
# *this* fake ncu's stdout/stderr picks up the "benchmark"'s output, exactly
# like the real collection step.
sep = args.index("--")
child_argv = args[sep + 1:]
result = subprocess.run(child_argv)
sys.exit(result.returncode)
"""

_FAKE_NCU_MISSING_REP_SOURCE = """#!/usr/bin/env python3
import sys
args = sys.argv[1:]
log_idx = args.index("--log-file")
with open(args[log_idx + 1], "w") as f:
    f.write("fake tool log, but .ncu-rep is deliberately never written\\n")
sys.exit(0)
"""

_FAKE_NCU_EXPORT_FAILS_SOURCE = """#!/usr/bin/env python3
import subprocess
import sys
args = sys.argv[1:]
if "--import" in args:
    sys.stdout.write("fake export stdout diagnostic\\n")
    sys.stderr.write("fake export stderr diagnostic\\n")
    sys.exit(7)
o_idx = args.index("-o")
report_base = args[o_idx + 1]
log_idx = args.index("--log-file")
with open(report_base + ".ncu-rep", "wb") as f:
    f.write(b"FAKE_NCU_REP_BYTES")
with open(args[log_idx + 1], "w") as f:
    f.write("fake ncu tool log\\n")
sep = args.index("--")
result = subprocess.run(args[sep + 1:])
sys.exit(result.returncode)
"""

_FAKE_NCU_COLLECT_FAILS_SOURCE = """#!/usr/bin/env python3
import sys
sys.stderr.write("fake collection failure\\n")
sys.exit(3)
"""


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--self-test"]:
        return run_self_test()
    if argv == ["--help"] or argv == ["-h"]:
        sys.stdout.write(usage())
        return 0

    parser = argparse.ArgumentParser(prog="p14_ncu_bridge.py", add_help=False)
    parser.add_argument("--help", "-h", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--metrics")
    parser.add_argument("--kernel-name")
    parser.add_argument("benchmark_argv", nargs=argparse.REMAINDER)
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        return 2

    if args.help:
        sys.stdout.write(usage())
        return 0
    if args.self_test:
        return run_self_test()

    if not args.metrics:
        print("p14_ncu_bridge: ERROR: --metrics is required", file=sys.stderr)
        return 2
    if not args.kernel_name:
        print("p14_ncu_bridge: ERROR: --kernel-name is required", file=sys.stderr)
        return 2
    benchmark_argv = args.benchmark_argv
    if benchmark_argv and benchmark_argv[0] == "--":
        benchmark_argv = benchmark_argv[1:]
    if not benchmark_argv:
        print("p14_ncu_bridge: ERROR: no benchmark command given after '--'", file=sys.stderr)
        return 2

    try:
        bundle = run_bridge(
            ncu_binary="ncu", metrics=args.metrics, kernel_name=args.kernel_name,
            benchmark_argv=benchmark_argv,
        )
    except BridgeError as exc:
        print(f"p14_ncu_bridge: ERROR: {exc}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(bundle)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
