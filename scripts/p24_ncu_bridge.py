#!/usr/bin/env python3
"""P2.4-only container-side NCU bridge.

Runs INSIDE the pinned container, launched through the unmodified
``scripts/run_container.sh`` exactly like every other GPU-touching P2.4
step. Never receives, constructs, or opens a campaign-relative or
repository raw-tree pathname: every NCU ``-o`` / ``--log-file`` /
``--import`` argument this script builds points exclusively inside a
private directory freshly created under the container's own,
non-host-mounted ``/tmp`` (``scripts/run_container.sh`` only ever
bind-mounts the repository at ``/workspace``; nothing under ``/tmp`` is
shared with the host, and the container itself is destroyed on exit --
``docker run --rm``). Adapts the audited P1.4 bridge design
(``scripts/p14_ncu_bridge.py``) with the P2.4-frozen profiler controls:
``--kernel-name-base function --kernel-name <exact-symbol> --launch-skip 1
--launch-count 1``, so the untimed pre-timing correctness-validation launch
is skipped and only the second, timed launch of the same kernel symbol is
profiled -- never silently falling back to profiling the wrong launch.

Design:
  1. Runs inside the container (via scripts/run_container.sh, unmodified).
  2. Creates a private directory in the container's own, non-mounted /tmp.
  3. Gives NCU only paths inside that private directory for "-o",
     "--log-file", and "--import".
  4. Verifies every expected output is a genuine, non-empty regular file.
  5. Emits a versioned, length-delimited bundle (see p24_safe_capture's
     NCU_BUNDLE_* helpers) containing exactly: application stdout,
     application stderr, the NCU tool log, the raw ".ncu-rep" bytes, the
     exported "metrics_raw.csv", and the metric-export step's stderr.
  6/7/8. The host side (scripts/run_exp02_umma_throughput_p24.sh) captures
     this script's stdout through p24_safe_capture.py's "run" subcommand and
     then decodes/publishes it via "publish-bundle".
  9. This script always deletes its own private directory before exiting.

This script takes no "--campaign-dir"-shaped argument at all, by design, so
it cannot structurally be pointed at a raw-tree path even if every argument
it is given were adversarial.

Usage:
  p24_ncu_bridge.py --metrics M1,M2,... --kernel-name NAME -- <bin> [args...]
  p24_ncu_bridge.py --self-test
  p24_ncu_bridge.py --help

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

import p24_safe_capture as capture_lib  # noqa: E402

# Frozen profiler controls (src/compute/P2_4_PROTOCOL.md). Never a
# forced-overwrite flag, never the full metric set, never a
# clock-controlling default. --launch-skip 1 --launch-count 1 profiles
# exactly the second matching
# launch of the exact kernel symbol (the timed one; the first is the
# untimed pre-timing correctness-validation launch) -- never guessed.
NCU_COLLECTION_FLAGS = (
    "--clock-control", "none",
    "--pipeline-boost-state", "dynamic",
    "--cache-control", "none",
    "--kernel-name-base", "function",
    "--launch-skip", "1",
    "--launch-count", "1",
    "--devices", "0",
    "--replay-mode", "kernel",
    "--print-summary", "none",
)
# --print-metric-name is intentionally absent: NCU 2025.4 accepts it only
# for the details page and exits 1 when combined with --page raw; the raw
# page already emits actual metric identifiers as wide-table column names.
NCU_EXPORT_FLAGS = (
    "--csv", "--page", "raw",
    "--print-units", "base",
    "--print-kernel-base", "function",
)


class BridgeError(RuntimeError):
    pass


def _bounded_diagnostic(path: Path, *, limit: int = 4096) -> str:
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
    if path.is_symlink():
        raise BridgeError(f"{label}: expected NCU output is a symlink, refusing: {path}")
    if not path.is_file():
        raise BridgeError(f"{label}: expected NCU output does not exist as a regular file: {path}")
    if path.stat().st_size == 0:
        raise BridgeError(f"{label}: expected NCU output is empty: {path}")


def run_bridge(
    *, ncu_binary: str, metrics: str, kernel_name: str, benchmark_argv: list[str], tmp_root: str = "/tmp",
) -> bytes:
    """Runs NCU collection then a GPU-free metrics export entirely inside a
    fresh private directory under tmp_root, and returns the encoded bundle
    bytes. Raises BridgeError on any failure; never partially writes a
    bundle."""
    if not benchmark_argv:
        raise BridgeError("no benchmark command given after '--'")
    if not metrics:
        raise BridgeError("--metrics must be a non-empty comma-separated metric list")
    if not kernel_name:
        raise BridgeError("--kernel-name must be non-empty")

    private_dir = Path(tempfile.mkdtemp(prefix="p24_ncu_bridge_", dir=tmp_root))
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
                collect_result = subprocess.run(collect_argv, stdout=out_f, stderr=err_f, stdin=subprocess.DEVNULL, shell=False)
            except OSError as exc:
                raise BridgeError(f"could not launch NCU collection ({ncu_binary!r}): {exc}") from exc
        if collect_result.returncode != 0:
            raise BridgeError(
                f"NCU collection exited {collect_result.returncode}; not bundling any output for this case "
                f"(application stdout/stderr and the NCU tool log are only ever emitted together with a "
                f"successful collection and export)"
            )
        _require_nonempty_file(ncu_rep_path, "collection: .ncu-rep")
        _require_nonempty_file(ncu_tool_log_path, "collection: NCU tool log")
        _require_nonempty_file(app_stdout_path, "collection: application stdout")

        export_argv = [ncu_binary, "--import", str(ncu_rep_path), *NCU_EXPORT_FLAGS]
        with open(metrics_csv_path, "wb") as out_f, open(metrics_export_stderr_path, "wb") as err_f:
            try:
                export_result = subprocess.run(export_argv, stdout=out_f, stderr=err_f, stdin=subprocess.DEVNULL, shell=False)
            except OSError as exc:
                raise BridgeError(f"could not launch NCU metrics export ({ncu_binary!r}): {exc}") from exc
        if export_result.returncode != 0:
            raise BridgeError(
                f"NCU metrics export exited {export_result.returncode}; stdout={_bounded_diagnostic(metrics_csv_path)}; "
                f"stderr={_bounded_diagnostic(metrics_export_stderr_path)}; not bundling any output for this case"
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
        "  p24_ncu_bridge.py --metrics M1,M2,... --kernel-name NAME -- <bin> [args...]\n"
        "  p24_ncu_bridge.py --self-test\n"
        "  p24_ncu_bridge.py --help\n"
    )


def run_self_test() -> int:
    import tempfile as _tempfile

    total = 0
    failures: list[str] = []

    def check(label: str, condition: bool, *, detail: str = "") -> None:
        nonlocal total
        total += 1
        if condition:
            print(f"p24_ncu_bridge: self-test: PASS: {label}", file=sys.stderr)
        else:
            print(f"p24_ncu_bridge: self-test: FAIL: {label}; {detail}", file=sys.stderr)
            failures.append(label)

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
        fake_bin.write_text("#!/usr/bin/env python3\nprint('schema_version,fake\\n1,ok')\n", encoding="utf-8")
        fake_bin.chmod(0o700)

        bundle = run_bridge(
            ncu_binary=str(fake_ncu_good), metrics="sm__cycles_elapsed.avg.per_second",
            kernel_name="umma_1sm_m128n64k16_d4", benchmark_argv=[sys.executable, str(fake_bin)], tmp_root=str(tmp_root),
        )
        segments = capture_lib.decode_ncu_bundle(bundle)
        must_be_nonempty = {"app_stdout", "ncu_tool_log", "ncu_rep", "metrics_csv"}
        check(
            "a successful bridge run emits a decodable bundle with exactly the six fixed segments",
            set(segments) == set(capture_lib.NCU_BUNDLE_SEGMENT_NAMES) and all(segments[k] for k in must_be_nonempty),
            detail=f"keys={sorted(segments)}",
        )
        check("the bundled .ncu-rep segment is the fake NCU's own report bytes", segments["ncu_rep"] == b"FAKE_NCU_REP_BYTES")
        check("the bundled metrics_csv segment is the fake NCU's own exported CSV", b"sm__cycles_elapsed" in segments["metrics_csv"])
        check(
            "the collection flags include --launch-skip 1 --launch-count 1 so only the second (timed) "
            "launch of the exact kernel symbol is ever profiled",
            "--launch-skip" in NCU_COLLECTION_FLAGS and NCU_COLLECTION_FLAGS[NCU_COLLECTION_FLAGS.index("--launch-skip") + 1] == "1"
            and "--launch-count" in NCU_COLLECTION_FLAGS and NCU_COLLECTION_FLAGS[NCU_COLLECTION_FLAGS.index("--launch-count") + 1] == "1",
        )
        check(
            "the raw-page export omits details-only --print-metric-name while retaining the frozen controls",
            "--print-metric-name" not in NCU_EXPORT_FLAGS
            and NCU_EXPORT_FLAGS == ("--csv", "--page", "raw", "--print-units", "base", "--print-kernel-base", "function"),
        )
        check("never a clock-controlling default", "base" not in NCU_COLLECTION_FLAGS and "boost" not in NCU_COLLECTION_FLAGS)
        check("the private directory is removed before run_bridge() returns", list(tmp_root.iterdir()) == [])

        raised = None
        try:
            run_bridge(
                ncu_binary=str(fake_ncu_collect_fails), metrics="sm__cycles_elapsed.avg.per_second",
                kernel_name="umma_1sm_m128n64k16_d4", benchmark_argv=[sys.executable, str(fake_bin)], tmp_root=str(tmp_root),
            )
        except BridgeError as exc:
            raised = str(exc)
        check("a fake NCU that fails during collection raises BridgeError and leaves no bundle", raised is not None)
        check("the private directory is still removed after a collection failure", list(tmp_root.iterdir()) == [])

        raised = None
        try:
            run_bridge(
                ncu_binary=str(fake_ncu_missing_rep), metrics="sm__cycles_elapsed.avg.per_second",
                kernel_name="umma_1sm_m128n64k16_d4", benchmark_argv=[sys.executable, str(fake_bin)], tmp_root=str(tmp_root),
            )
        except BridgeError as exc:
            raised = str(exc)
        check("a zero-exit NCU stand-in that never produced .ncu-rep is still rejected", raised is not None and ".ncu-rep" in raised)

        raised = None
        try:
            run_bridge(
                ncu_binary=str(fake_ncu_export_fails), metrics="sm__cycles_elapsed.avg.per_second",
                kernel_name="umma_1sm_m128n64k16_d4", benchmark_argv=[sys.executable, str(fake_bin)], tmp_root=str(tmp_root),
            )
        except BridgeError as exc:
            raised = str(exc)
        check(
            "a metrics-export failure raises BridgeError and preserves bounded stdout/stderr diagnostics",
            raised is not None and "export" in raised and "fake export stdout diagnostic" in raised and "fake export stderr diagnostic" in raised,
        )

        raised = None
        try:
            run_bridge(ncu_binary=str(fake_ncu_good), metrics="sm__cycles_elapsed.avg.per_second", kernel_name="k", benchmark_argv=[], tmp_root=str(tmp_root))
        except BridgeError as exc:
            raised = str(exc)
        check("an empty benchmark_argv is rejected before any process is launched", raised is not None)

        raised = None
        try:
            run_bridge(
                ncu_binary=str(tmp_path / "does_not_exist_ncu"), metrics="sm__cycles_elapsed.avg.per_second",
                kernel_name="k", benchmark_argv=[sys.executable, str(fake_bin)], tmp_root=str(tmp_root),
            )
        except BridgeError as exc:
            raised = str(exc)
        check("a nonexistent NCU executable raises BridgeError rather than an uncaught OSError", raised is not None)

        tricky = {name: f"segment-{name}-payload".encode() for name in capture_lib.NCU_BUNDLE_SEGMENT_NAMES}
        tricky["ncu_rep"] = capture_lib.NCU_BUNDLE_MAGIC + b"\x00\x01binary\xffbytes" + capture_lib.NCU_BUNDLE_MAGIC
        encoded = capture_lib.encode_ncu_bundle(tricky)
        decoded = capture_lib.decode_ncu_bundle(encoded)
        check("encode/decode round-trip is exact even when content contains the magic marker", decoded == tricky)

    if failures:
        print(f"p24_ncu_bridge: self-test: FAILED ({len(failures)}/{total} case(s)): {failures}", file=sys.stderr)
        print("p24_ncu_bridge: SELF_TEST_RESULT=FAIL", file=sys.stderr)
        return 1
    print(f"p24_ncu_bridge: self-test: OK ({total} cases)", file=sys.stderr)
    print("p24_ncu_bridge: SELF_TEST_RESULT=PASS", file=sys.stderr)
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
    if "--print-metric-name" in args:
        sys.stdout.write("==ERROR== Option '--print-metric-name' is only supported for the details page.\\n")
        sys.exit(1)
    required = ["--csv", "--page", "raw", "--print-units", "base", "--print-kernel-base", "function"]
    if any(item not in args for item in required):
        sys.stderr.write("fake export is missing a required raw-page flag\\n")
        sys.exit(8)
    sys.stdout.write("ID,Process ID,Process Name,Host Name,Kernel Name,Kernel Time,Context,Stream,sm__cycles_elapsed.avg.per_second\\n")
    sys.stdout.write(",,,,,,,,cycle/nsecond\\n")
    sys.stdout.write("0,1234,fake,fake-host,umma_1sm_m128n64k16_d4,2026-Aug-04 00:00:00,1,7,1.4\\n")
    sys.exit(0)
o_idx = args.index("-o")
report_base = args[o_idx + 1]
log_idx = args.index("--log-file")
log_path = args[log_idx + 1]
with open(report_base + ".ncu-rep", "wb") as f:
    f.write(b"FAKE_NCU_REP_BYTES")
with open(log_path, "w") as f:
    f.write("fake ncu tool log\\n")
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

    parser = argparse.ArgumentParser(prog="p24_ncu_bridge.py", add_help=False)
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
        print("p24_ncu_bridge: ERROR: --metrics is required", file=sys.stderr)
        return 2
    if not args.kernel_name:
        print("p24_ncu_bridge: ERROR: --kernel-name is required", file=sys.stderr)
        return 2
    benchmark_argv = args.benchmark_argv
    if benchmark_argv and benchmark_argv[0] == "--":
        benchmark_argv = benchmark_argv[1:]
    if not benchmark_argv:
        print("p24_ncu_bridge: ERROR: no benchmark command given after '--'", file=sys.stderr)
        return 2

    try:
        bundle = run_bridge(ncu_binary="ncu", metrics=args.metrics, kernel_name=args.kernel_name, benchmark_argv=benchmark_argv)
    except BridgeError as exc:
        print(f"p24_ncu_bridge: ERROR: {exc}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(bundle)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
