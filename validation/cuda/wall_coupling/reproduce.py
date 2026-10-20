#!/usr/bin/env python3
"""Reproduce only the CUDA port checks against the delivered CPU evidence.
The --archive directory is the common extraction root of the three delivered ZIPs.
"""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SNAPSHOTS = ["original-failures/fail-2.bin", "original-failures/fail-3.bin",
             "history-only-fail.bin", "l2-last-fail.bin", "adaptive-valid-history-fail.bin"]

def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cuda", type=Path, default=Path("/usr/local/cuda"))
    ap.add_argument("--arch", default="sm_89")
    ap.add_argument("--sanitize", action="store_true")
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()
    archive, out = args.archive.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    runs = []
    def run(cmd, name, env=None):
        begin = time.monotonic()
        p = subprocess.run([str(x) for x in cmd], cwd=ROOT, env=env, capture_output=True, text=True)
        (out/(name+".stdout")).write_text(p.stdout)
        (out/(name+".stderr")).write_text(p.stderr)
        runs.append(dict(command=[str(x) for x in cmd], exit=p.returncode, seconds=time.monotonic()-begin))
        (out/"commands.json").write_text(json.dumps(runs, indent=2))
        print(name, p.returncode, flush=True)
        p.check_returncode()
    inputs = [archive/"complete-coupling/reference-debug"/x for x in SNAPSHOTS]
    manifest = {str(p.relative_to(archive)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (out/"inputs.json").write_text(json.dumps(manifest, indent=2))
    source = {}
    for folder in ["common", "applications/gasUGKP/private_backend", "validation/cuda/wall_coupling"]:
        for p in (ROOT/folder).rglob("*"):
            if p.is_file() and p.suffix in {".H", ".cuh", ".cu", ".inc", ".cpp", ".py"} and not any(x in p.parts for x in ["build", "build_logs"]):
                source[str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    (out/"source.json").write_text(json.dumps({"head": subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(), "files":source},indent=2))
    include = ["-I"+str(ROOT), "-I"+str(ROOT/"common")]
    run(["g++", "-std=c++17", "-O2", "-ffp-contract=off", *include,
         "-I"+str(archive/"complete-coupling/twod"), "-I"+str(archive/"complete-coupling/augmented"),
         HERE/"reference_cpu.cpp", "-o", out/"reference-cpu"], "build-reference-cpu")
    run([out/"reference-cpu", *inputs], "reference-cpu")
    nvcc = [args.cuda/"bin/nvcc", "-std=c++17", "-O3", "-arch="+args.arch,
            "--expt-extended-lambda", "--expt-relaxed-constexpr", "--fmad=false", "-Xptxas=-v", *include]
    run([*nvcc, HERE/"reference.cu", "-o", out/"reference"], "build-reference")
    run([out/"reference", *inputs], "reference")
    run([*nvcc, "-I"+str(archive/"complete-coupling/reconstruction"),
         "-I"+str(archive/"full-physics-audit"), HERE/"profile.cu", "-o", out/"profile"], "build-profile")
    env = dict(os.environ)
    if args.benchmark: env["UGKP_PROFILE_BENCH"] = "1"
    run([out/"profile"], "profile", env)
    if args.sanitize:
        for tool in ["memcheck", "racecheck", "synccheck"]:
            for program in ["reference", "profile"]:
                extra = [inputs[3]] if program == "reference" else []
                run([args.cuda/"bin/compute-sanitizer", "--tool", tool, "--error-exitcode", "99",
                     out/program, *extra], program+"-"+tool)
    binaries = {name: hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ["reference-cpu","reference","profile"]}
    (out/"binaries.json").write_text(json.dumps(binaries, indent=2))

if __name__ == "__main__": main()
