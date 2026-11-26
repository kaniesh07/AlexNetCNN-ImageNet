#!/usr/bin/env python3
"""
comparison_test.py -- Stage B/C cross-checks, driven from Python (HOST).

C1  HLS C++ kernel (compiled with g++ + open-source ap_int headers) runs its
    unit tests AND the full 8-layer AlexNet; every layer output must equal the
    bit-exact Python model (build/deploy/golden).   [needs g++ and build/hlsap]
C2  Handwritten RTL (Icarus Verilog) unit + stream tests, including a packet of
    real conv5 activations whose expected values are the HLS golden output.
    [needs iverilog]; with Vivado use rtl/testbench/run_xsim.sh instead.
C3  Board driver against the mock PYNQ runtime (tests/driver_mock_test.py).
Any missing tool is reported as SKIP, never as PASS.
"""
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run(cmd, cwd, env=None):
    p = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True,
                       env={**os.environ, **(env or {})})
    return p.returncode, p.stdout + p.stderr


def main():
    results = {}
    golden = os.path.join(ROOT, "build", "deploy", "golden")
    if shutil.which("g++") and os.path.isdir(os.path.join(ROOT, "build", "hlsap")):
        rc, out = run("make -s -f Makefile.gcc run-full", os.path.join(ROOT, "hls"),
                      {"GOLDEN": golden})
        print("\n".join(l for l in out.splitlines() if l.startswith("[") or "====" in l))
        results["C1 HLS C++ vs Python golden"] = rc == 0 and "ALL TESTS PASSED" in out \
            and "[full L7]" in out
    else:
        results["C1 HLS C++ vs Python golden"] = None
    if shutil.which("iverilog"):
        rc, out = run("./run_iverilog.sh", os.path.join(ROOT, "rtl", "testbench"))
        print("\n".join(l for l in out.splitlines() if "PASS" in l or "FAIL" in l or "ERROR" in l))
        results["C2 RTL vs golden model"] = rc == 0 and "ALL RTL TESTS PASSED" in out
    else:
        results["C2 RTL vs golden model"] = None
    rc, out = run(f"{sys.executable} tests/driver_mock_test.py", ROOT)
    print(out.strip())
    results["C3 driver vs mock IP"] = rc == 0
    print("\nSUMMARY")
    for k, v in results.items():
        print(f"  {k:32s} {'SKIP' if v is None else ('PASS' if v else 'FAIL')}")
    sys.exit(0 if all(v is not False for v in results.values()) else 1)


if __name__ == "__main__":
    main()
