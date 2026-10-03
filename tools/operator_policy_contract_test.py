#!/usr/bin/env python3
"""Compile actual entry configuration and reject missing/invalid policy flags."""
from pathlib import Path
import argparse
import re
import subprocess
import tempfile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = ap.parse_args()
    root = args.root.resolve()
    contract = root / 'common/GpuOperatorContract.cuh'
    if not contract.is_file():
        print('FAIL missing executable entry policy contract')
        return 1
    required = re.findall(r'^#ifndef (GPU_\w+)', contract.read_text(), re.M)
    tests = 0
    with tempfile.TemporaryDirectory(prefix='operator-policy-') as tmp:
        source = Path(tmp) / 'contract.cpp'
        for app in ('gasUGKP', 'FSH', 'CHT'):
            path = root / 'applications' / app / ('gpu' if app == 'CHT' else 'private_backend') / 'GpuResidentStrict.cu'
            if not path.is_file():
                continue
            prefix = path.read_text().split('#include <cuda_runtime.h>', 1)[0]
            defines = []
            source_lines = iter(prefix.splitlines())
            for line in source_lines:
                if line.startswith('#define '):
                    while line.endswith('\\'):
                        line += '\n' + next(source_lines)
                    defines.append(line)
            for bits in ((32, 64) if app == 'CHT' else (64,)):
                setup = f'#define UGKWP_GPU_REAL_BITS {bits}\n'
                if app == 'CHT':
                    setup += '#include "GpuPrecisionTypes.H"\n'
                def compile_with(items):
                    source.write_text(setup + '\n'.join(items) + '\n#include "GpuOperatorContract.cuh"\nint main(){}\n')
                    result = subprocess.run(['g++', '-std=c++17', '-fsyntax-only', '-I'+str(root/'common'), str(source)], capture_output=True, text=True)
                    if result.returncode not in (0, 1) or 'internal compiler error' in result.stderr.lower():
                        raise RuntimeError(f'compiler execution failed, not a policy rejection: exit={result.returncode}\n{result.stderr}')
                    return result
                result = compile_with(defines)
                if result.returncode:
                    print(app, bits, result.stderr)
                    return 1
                tests += 1
                present = set(re.findall(r'^#define (GPU_\w+)', '\n'.join(defines), re.M))
                for macro in required:
                    if macro not in present:
                        continue
                    result = compile_with([line for line in defines if not re.match(r'#define '+macro+r'\b', line)])
                    if result.returncode == 0 or macro not in result.stderr:
                        print('FAIL expected missing-policy diagnostic:', app, bits, macro, 'compiler exit', result.returncode)
                        print(result.stderr)
                        return 1
                    tests += 1
                for macro, value in [('GPU_OPERATOR_THERMAL', '2'), ('GPU_POOL_STATIC_DIRECTORY', '2'), ('GPU_DIRECTORY_OWNS_TILE_POLICY', '2')]:
                    result = compile_with([re.sub(r'^#define '+macro+r'\b.*', '#define '+macro+' '+value, line) for line in defines])
                    if result.returncode == 0:
                        print('FAIL invalid flag accepted:', app, bits, macro)
                        return 1
                    tests += 1
                for changes in [
                    {'GPU_POOL_STATIC_DIRECTORY':'0','GPU_POOL_CACHED_PROBABILITY':'1'},
                    {'GPU_POOL_STATIC_DIRECTORY':'0','GPU_DIRECTORY_HAS_BASE_ONLY':'1'},
                    {'GPU_DIRECTORY_HAS_BASE_ONLY':'0','GPU_DIRECTORY_OWNS_TILE_POLICY':'1'},
                ]:
                    items = defines[:]
                    for macro, value in changes.items():
                        items = [re.sub(r'^#define '+macro+r'\b.*', '#define '+macro+' '+value, line) for line in items]
                    result = compile_with(items)
                    if result.returncode == 0:
                        print('FAIL invalid policy combination accepted:', changes)
                        return 1
                    tests += 1
                for macro, typ in [('GPU_OPERATOR_REAL','int'),('GPU_OPERATOR_TIME','long long')]:
                    result = compile_with([re.sub(r'^#define '+macro+r'\b.*', '#define '+macro+' '+typ, line) for line in defines])
                    if result.returncode == 0:
                        print('FAIL invalid numeric type accepted:', app, bits, macro, typ)
                        return 1
                    tests += 1
        # No application identifiers are supplied: invalid policy combinations are rejected structurally.
    print(f'PASS {tests} entry policy compile cases')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
