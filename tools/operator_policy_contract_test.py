#!/usr/bin/env python3
"""Compile real entry adapter definitions without imposing an inventory policy.

This is an explicit offline compiler smoke test. It does not establish CUDA
operator execution, and normal builds do not invoke it.
"""
from pathlib import Path
import argparse
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    tests = 0
    with tempfile.TemporaryDirectory(prefix='operator-adapters-') as tmp:
        source = Path(tmp) / 'entry.cpp'

        def compile_source(text):
            source.write_text(text)
            return subprocess.run(['g++', '-std=c++17', '-fsyntax-only', '-I'+str(root/'common'), str(source)], capture_output=True, text=True)

        empty = compile_source('#include "GpuOperatorContract.cuh"\nint main(){}\n')
        if empty.returncode:
            print(empty.stderr)
            return 1
        tests += 1
        for app in ('gasUGKP', 'FSH', 'CHT'):
            path = root/'applications'/app/('gpu' if app == 'CHT' else 'private_backend')/'GpuResidentStrict.cu'
            if not path.is_file():
                continue
            prefix = path.read_text().split('#include <cuda_runtime.h>', 1)[0]
            defines = []
            lines = iter(prefix.splitlines())
            for line in lines:
                if line.startswith('#define '):
                    while line.endswith('\\'):
                        line += '\n' + next(lines)
                    defines.append(line)
            setup = f''
            if app == 'CHT':
                setup += '#include "GpuPrecisionTypes.H"\n'
            text = setup+'\n'.join(defines)+'\n#include "GpuOperatorContract.cuh"\n'

            text += 'GPU_OPERATOR_REAL value=GPU_OPERATOR_R(1.25);\nGPU_OPERATOR_TIME interval=0.5;\nint main(){return value<=0 || interval<=0;}\n'
            result = compile_source(text)
            if result.returncode:
                print(app, 64, result.stderr)
                return 1
            tests += 1

        missing = compile_source('#include "GpuOperatorContract.cuh"\nGPU_OPERATOR_REAL value;\n')
        if missing.returncode == 0 or 'GPU_OPERATOR_REAL' not in missing.stderr:
            print('FAIL compiler did not report the undefined used type')
            return 1
        tests += 1
    print(f'PASS {tests} actual entry adapter compile cases; no policy inventory admission')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
