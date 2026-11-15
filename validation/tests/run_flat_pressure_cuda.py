#!/usr/bin/env python3
import argparse,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def main():
    p=argparse.ArgumentParser();p.add_argument("--work",type=Path,required=True)
    p.add_argument("--compile-only",action="store_true");p.add_argument("--nvcc",default="/usr/local/cuda/bin/nvcc")
    a=p.parse_args();a.work.mkdir(parents=True,exist_ok=False)
    for bits in (32,64):
        exe=a.work/f"pressure{bits}"
        # Match the production materialized tree: overlay-first -I paths alone
        # allow quoted nested includes to load both physical headers.
        includes=[ROOT/".build/precision/fp32/source/common"] if bits==32 else [ROOT/"common"]
        command=[a.nvcc,"-std=c++17","-O2","--expt-relaxed-constexpr",
                 "-DTEST_REAL="+("float" if bits==32 else "double"),
                 *[item for path in includes for item in ("-I",str(path))],
                 str(ROOT/"validation/tests/fixtures/pressure/flat_pressure_cuda.cu"),"-o",str(exe)]
        with (a.work/f"build{bits}.log").open("w") as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        if not a.compile_only:
            with (a.work/f"run{bits}.log").open("w") as log:
                subprocess.run([str(exe)],stdout=log,stderr=subprocess.STDOUT,check=True)
if __name__=="__main__":main()
