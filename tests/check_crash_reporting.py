"""
Run deliberate failures in subprocesses; never attach to a production server.

Usage: python3 tests/check_crash_reporting.py target/debug/examples/crash_probe
"""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

probe = str(Path(sys.argv[1]).resolve())
cases = [("normal", 0), ("panic", 101), ("caught-panic", 0)]
if sys.platform == "linux":
    cases += [("segv", -signal.SIGSEGV), ("abort", -signal.SIGABRT),
              ("worker-segv", -signal.SIGSEGV), ("helper-dead", -signal.SIGSEGV)]
temporary = tempfile.TemporaryDirectory()
if sys.platform == "linux":
    library = str(Path(temporary.name) / "native_probe.so")
    subprocess.run(["c++", "-shared", "-fPIC", "-g", "-O2", "-fno-omit-frame-pointer",
                    str(Path(__file__).with_name("native_crash_probe.cpp")), "-o", library], check=True)
    cases += [("native-segv", -signal.SIGSEGV)]
for mode, expected in cases:
    env = dict(os.environ, RUST_BACKTRACE="0")
    result = subprocess.run([probe, mode] + ([library] if mode == "native-segv" else []), capture_output=True, text=True,
                            env=env, timeout=35)
    assert result.returncode == expected, (mode, result.returncode, result.stderr)
    if "panic" in mode:
        assert "MUNA RUST PANIC" in result.stderr, result.stderr
        assert "deliberate panic" in result.stderr, result.stderr
        assert "rust_panic_probe" in result.stderr, result.stderr
        assert "crash_probe.rs:" in result.stderr, result.stderr
        assert "name=" in result.stderr and "pid=" in result.stderr
    elif mode != "normal":
        assert "cpptrace fatal-signal reporting enabled" in result.stderr, result.stderr
        assert "MUNA FATAL SIGNAL" in result.stderr, result.stderr
        assert "fault_address=0x" in result.stderr and "pc=0x" in result.stderr
        assert "module=" in result.stderr and "offset=0x" in result.stderr
        if mode == "helper-dead":
            assert "symbolication helper unavailable" in result.stderr, result.stderr
            assert "END MUNA FATAL SIGNAL" in result.stderr, result.stderr
            print(f"PASS {mode} (exit={result.returncode}, raw fallback)", flush=True)
            continue
        assert "symbolicated crashing-thread stack" in result.stderr, result.stderr
        if mode == "native-segv":
            assert "native_crash_probe" in result.stderr, result.stderr
            assert "native_crash_probe.cpp:" in result.stderr, result.stderr
            assert "native_library_probe" in result.stderr, result.stderr
            assert "fault_address=0x0 " in result.stderr, result.stderr
        else:
            assert "fatal_probe" in result.stderr, result.stderr
        assert "crash_probe.rs:" in result.stderr, result.stderr
        assert "/maps ---" in result.stderr and "/status ---" in result.stderr
        assert "END MUNA FATAL SIGNAL" in result.stderr, result.stderr
        assert "timed out" not in result.stderr, result.stderr
    print(f"PASS {mode} (exit={result.returncode})", flush=True)
