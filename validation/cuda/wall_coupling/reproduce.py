#!/usr/bin/env python3
"""Reproduce current coherent-profile/directional-solver scalar and CUDA checks.

Self-contained coherent checks need only this checkout. --cpu-only runs no CUDA
and no CPU cooperative tests. Optional --archive retains the five historical
BVP snapshot checks; it is NOT the oracle for the coherent profile. That path
expects the common extraction root of the previously delivered evidence ZIPs.
The profile oracle uses the same current shared math as CUDA, not an independent
frozen mathematical reference. The separate directional fixture checks scalar
roots against pinned numeric evidence, then compares native CUDA Engine<24> to
the current scalar engine. Full coupled qualification remains separate.
"""
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SNAPSHOTS = ["original-failures/fail-2.bin", "original-failures/fail-3.bin",
             "history-only-fail.bin", "l2-last-fail.bin", "adaptive-valid-history-fail.bin"]


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--archive", type=Path,
                    help="also run the original, separate legacy BVP snapshot checks")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"))
    ap.add_argument("--arch", default="sm_89")
    ap.add_argument("--cpu-only", action="store_true",
                    help="build/run only scalar host checks; GPU results remain UNRUN")
    ap.add_argument("--sanitize", action="store_true",
                    help="run CUDA memcheck/racecheck/synccheck (not used with --cpu-only)")
    ap.add_argument("--benchmark", action="store_true",
                    help="GPU-only single-lane vs warp32 control (not CPU parallelism)")
    args = ap.parse_args()
    archive = args.archive.resolve() if args.archive else None
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs, binaries = [], []
    scope = {
        "oracle": "current shared coherent Profile::fitTeam with SerialWallTeam",
        "independent_mathematical_reference": False,
        "cpu_parallel_tests": False,
        "profile_inputs": "original eight 24/128-node cases plus three manufactured affine/pure cases",
        "profile_seed": "original cases use legacy BVP seeds; manufactured cases use analytic constant seeds",
        "profile_solver_qualification": False,
        "directional_solver_cases": "24-node captured723 accepted seed, failed seed with fresh cache, and cold start",
        "directional_scalar_oracle": "pinned numeric roots in coherent_directional_723.H",
        "directional_gpu_oracle": "current shared scalar Engine<24>, not a CPU cooperative test",
        "legacy_archive_checks": archive is not None,
        "gpu_execution": "UNRUN",  # coherent profile
        "gpu_directional_execution": "UNRUN",
        "gpu_sanitizers": "UNRUN",
        "gpu_benchmark": "UNRUN",
    }

    def save_scope():
        (out / "scope.json").write_text(json.dumps(scope, indent=2) + "\n")

    save_scope()

    def run(cmd, name, env=None, status=()):
        command = [str(x) for x in cmd]
        for key in status:
            scope[key] = "running"
        save_scope()
        begin = time.monotonic()
        try:
            p = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        except OSError as error:
            for key in status:
                scope[key] = "UNRUN"  # process could not be started
            save_scope()
            (out / (name + ".stderr")).write_text(str(error) + "\n")
            runs.append(dict(name=name, command=command, exit=None,
                             seconds=time.monotonic() - begin, error=str(error)))
            (out / "commands.json").write_text(json.dumps(runs, indent=2) + "\n")
            raise
        (out / (name + ".stdout")).write_text(p.stdout)
        (out / (name + ".stderr")).write_text(p.stderr)
        runs.append(dict(name=name, command=command, exit=p.returncode,
                         seconds=time.monotonic() - begin))
        (out / "commands.json").write_text(json.dumps(runs, indent=2) + "\n")
        print(name, p.returncode, flush=True)
        for key in status:
            scope[key] = "passed" if p.returncode == 0 else "failed"
        save_scope()
        p.check_returncode()

    source = {}
    for folder in ["common", "applications/gasUGKP/private_backend",
                   "validation/cuda/wall_coupling", "validation/tests/gas_wall/fixtures"]:
        for p in (ROOT / folder).rglob("*"):
            if (p.is_file() and p.suffix in {".H", ".cuh", ".cu", ".inc", ".cpp", ".py"}
                    and not any(x in p.parts for x in ["build", "build_logs"])):
                source[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out / "source.json").write_text(json.dumps({
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "files": source,
    }, indent=2) + "\n")
    include = ["-I" + str(ROOT), "-I" + str(ROOT / "common")]
    host = ["g++", "-std=c++17", "-O2", "-ffp-contract=off", *include]
    run([*host, "-x", "c++", "-DUGKP_PROFILE_CPU_ONLY=1", HERE / "profile.cu",
         "-o", out / "profile-cpu"], "build-profile-cpu")
    binaries.append("profile-cpu")
    run([out / "profile-cpu"], "profile-cpu")
    scope["scalar_profile"] = "passed"
    save_scope()
    run([*host, "-x", "c++", "-DUGKP_COHERENT_CPU_ONLY=1", HERE / "coherent_directional.cu",
         "-o", out / "coherent-directional-cpu"], "build-coherent-directional-cpu")
    binaries.append("coherent-directional-cpu")
    run([out / "coherent-directional-cpu"], "coherent-directional-cpu",
        status=("scalar_directional",))

    inputs = []
    if archive:
        inputs = [archive / "complete-coupling/reference-debug" / x for x in SNAPSHOTS]
        # Historical snapshots/headers remain a separately labelled legacy test.
        legacy_headers = list((archive / "complete-coupling/twod").glob("*.hpp"))
        legacy_headers += list((archive / "complete-coupling/augmented").glob("*.hpp"))
        manifest = {str(p.relative_to(archive)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [*inputs, *legacy_headers]}
        (out / "legacy-inputs.json").write_text(json.dumps(manifest, indent=2) + "\n")
        run([*host, "-I" + str(archive / "complete-coupling/twod"),
             "-I" + str(archive / "complete-coupling/augmented"), HERE / "reference_cpu.cpp",
             "-o", out / "reference-cpu"], "build-legacy-reference-cpu")
        binaries.append("reference-cpu")
        run([out / "reference-cpu", *inputs], "legacy-reference-cpu")

    if not args.cpu_only:
        nvcc = [args.cuda / "bin/nvcc", "-std=c++17", "-O3", "-arch=" + args.arch,
                "--expt-extended-lambda", "--expt-relaxed-constexpr", "--fmad=false",
                "-Xptxas=-v", *include]
        # No external ConservativeProfile.hpp or archive include enters this build.
        run([*nvcc, HERE / "profile.cu", "-o", out / "profile"], "build-profile")
        binaries.append("profile")
        env = dict(os.environ)
        env.pop("UGKP_PROFILE_BENCH", None)
        if args.benchmark:
            env["UGKP_PROFILE_BENCH"] = "1"
        run([out / "profile"], "profile", env,
            status=("gpu_execution", "gpu_benchmark") if args.benchmark else ("gpu_execution",))
        run([*nvcc, HERE / "coherent_directional.cu", "-o", out / "coherent-directional"],
            "build-coherent-directional")
        binaries.append("coherent-directional")
        run([out / "coherent-directional"], "coherent-directional",
            status=("gpu_directional_execution",))
        programs = ["profile", "coherent-directional"]
        # Preserve the old six cold-start cases even without an archive; adding
        # --archive also supplies all five original snapshot inputs unchanged.
        run([*nvcc, HERE / "reference.cu", "-o", out / "reference"], "build-legacy-reference")
        binaries.append("reference")
        run([out / "reference", *inputs], "legacy-reference")
        programs.append("reference")
        if args.sanitize:
            for tool in ["memcheck", "racecheck", "synccheck"]:
                for program in programs:
                    extra = [inputs[3]] if program == "reference" and inputs else []
                    run([args.cuda / "bin/compute-sanitizer", "--tool", tool,
                         "--error-exitcode", "99", out / program, *extra],
                        program + "-" + tool, env, status=("gpu_sanitizers",))
            scope["gpu_sanitizers"] = "passed"
            save_scope()
    hashes = {name: hashlib.sha256((out / name).read_bytes()).hexdigest() for name in binaries}
    (out / "binaries.json").write_text(json.dumps(hashes, indent=2) + "\n")


if __name__ == "__main__":
    main()
