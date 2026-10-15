"""Compile/run the real CUDA PR14 cold initialization regression (no CPU shim)."""
from pathlib import Path as _ValidationPath
import sys as _validation_sys
_validation_sys.path.insert(0, str(next(p.parent for p in _ValidationPath(__file__).resolve().parents if p.name == "validation")))
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import argparse
import os
import resource
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("app", choices=["CHT", "FSH"])
    parser.add_argument("--nvcc", default=os.environ.get("NVCC", "/usr/local/cuda/bin/nvcc"))
    parser.add_argument("--arch", default="sm_89")
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    backend = root / "applications" / args.app / ("gpu" if args.app == "CHT" else "private_backend")
    command = [args.nvcc, "-std=c++17", "-O3", "-arch=" + args.arch,
               "--fmad=" + ("false" if args.app == "CHT" else "true"),
               "-DPR14_COLD_CHT=" + str(int(args.app == "CHT")),
               "-I" + str(backend), "-I" + str(root / "common"),
               "-I" + str(root / "common/wall"),
               "-I" + str(root / "applications" / args.app / "gpu"),
               str(VALIDATION_ROOT / "tests/fixtures/pr14_fast_cold/initialization.cu"),
               "-o", str(out / "initialization")]
    if args.app == "CHT":
        command.append(str(backend / "GpuWallEnergy64.cu"))
    soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK,
                       (64 * 1024 * 1024 if hard == resource.RLIM_INFINITY
                        else min(64 * 1024 * 1024, hard), hard))
    (out / "command.txt").write_text(" ".join(command) + "\n")
    with (out / "build.log").open("w") as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print((out / "build.log").read_text()[-8000:])
        return result.returncode
    if args.compile_only:
        print("Compiled actual CUDA fixture; GPU runtime was not run.")
        return 0
    with (out / "run.log").open("w") as log:
        result = subprocess.run([str(out / "initialization")], stdout=log,
                                stderr=subprocess.STDOUT)
    print((out / "run.log").read_text())
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
