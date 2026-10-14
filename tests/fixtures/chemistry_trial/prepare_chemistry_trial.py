"""Generate/compile a real-CUDA chemistry owner regression; never emulate CUDA.

Execution is explicit (--run) and requires an actual supported CUDA device.
The generated translation unit includes the unchanged production native owner.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--nvcc', default=shutil.which('nvcc'))
    parser.add_argument('--arch', default=os.environ.get('UGKWP_CUDA_ARCH', 'sm_89'))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = root / 'applications/gasUGKP/private_backend'
    native = (source / 'GpuResidentStrict.cu').read_text()
    begin = native.index('struct DeviceState')
    end = native.index('\n};', begin)
    state = native[begin:end].replace(
        '#include "GpuAutomaticCsrScheduleFields.inl"',
        (root / 'common/GpuAutomaticCsrScheduleFields.inl').read_text())
    fields = re.findall(
        r'^\s*((?:unsigned\s+)?(?:long long|char)|double|float|int|GpuReal|GpuTime|GpuWallEnergy)\*\s+(\w+)\s*=',
        state, re.M)
    fields = [(ty, name) for ty, name in fields
              if name not in ('diagnosticPreTransportParticleCount', 'sourceInjectedCount')
              and not name.startswith('flatPressure')]
    template = Path(__file__).with_name('chemistry_trial.cu.in').read_text()
    generated = template.replace('ALLOCATE_NATIVE_FIELDS', '\n'.join(
        f'    testMem(s->{name});' for _, name in fields))
    cu = output / 'chemistry_trial.cu'
    cu.write_text(generated)
    binary = output / 'chemistry_trial'
    command = [args.nvcc or 'nvcc', '-std=c++17', '-O3', '-arch=' + args.arch,
               '--fmad=true', '-I' + str(source), '-I' + str(root / 'common'),
               '-I' + str(root / 'common/gasNumerics'),
               '-I' + str(root / 'applications/gasUGKP/gpu'), str(cu), '-o', str(binary)]
    (output / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
    if args.generate_only:
        print(cu)
        return
    if not args.nvcc:
        parser.error('nvcc is unavailable; use --generate-only to prepare the reviewer fixture')
    with (output / 'build.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise SystemExit(result.returncode)
    print(binary)
    if args.run:
        with (output / 'run.log').open('w') as log:
            result = subprocess.run([str(binary)], stdout=log, stderr=subprocess.STDOUT)
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
