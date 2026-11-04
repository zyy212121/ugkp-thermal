"""Build real CUDA mirror kernels; optionally verify the pre-fix counterexample."""
import argparse
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--before-header', type=Path)
    parser.add_argument('--expect-failure', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    header = args.before_header or ROOT/'common/operators/updateLegacyGasBoundaryMirrorKernel.cuh'
    # Copy unchanged kernel definitions; exclude unrelated reconstruction helpers.
    text = header.read_text()
    begin = text.index('template<class GasState>')
    end = text.index('#include "reconstructGasMixtureFace.cuh"')
    (out/'mirror_kernels.inc').write_text(text[begin:end])
    exe = out/'dynamic-inlet-mirror'
    command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-O2', '--fmad=false',
               '-arch=sm_89', '-I'+str(ROOT/'common'), '-I'+str(out),
               str(ROOT/'validation/pr36/dynamic_inlet_mirror.cu'), '-o', str(exe)]
    (out/'build-command.txt').write_text(' '.join(command)+'\n')
    with (out/'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (out/'run.log').open('w') as log:
        result = subprocess.run([exe], stdout=log, stderr=subprocess.STDOUT)
    print((out/'run.log').read_text(), end='')
    assert result.returncode == (1 if args.expect_failure else 0), result.returncode

if __name__ == '__main__':
    main()
