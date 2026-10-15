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


def validate_storage_output(output):
    rows = re.findall(
        r'^PASS storage-failure entry=(positivity|euler|sst) native=(\d+) '
        r'code=1 unchangedState=1 offsetRestored=1$', output, re.M)
    if sorted((entry, int(native)) for entry, native in rows) != sorted(
            (entry, native) for entry in ('positivity', 'euler', 'sst') for native in (0, 1)):
        raise ValueError('incomplete actual storage producer/consumer matrix')
    return dict(storage_controls=6)


def validate_run_output(output):
    controls = (
        'PASS requested owner accepted trace',
        'PASS terminal owner completion',
        'PASS original successful requested interval',
        'PASS original direct chemistry success',
    )
    for marker in controls:
        if sum(line.startswith(marker) for line in output.splitlines()) != 1:
            raise ValueError('missing or duplicate chemistry control: ' + marker)
    for label in ('failed-cell', 'failed-transaction'):
        kinds = re.findall(r'^PASS ' + label + r' kind=(\d+) ', output, re.M)
        if sorted(map(int, kinds)) != list(range(16)):
            raise ValueError('incomplete chemistry ' + label + ' matrix')
    statuses = re.findall(
        r'^FAIL_STATUS kind=(\d+) cell=(\d+) code=-?\d+ thermo=-?\d+ '
        r'reaction=-?\d+ attempted=-?\d+ rejected=-?\d+ cellStatus=-?\d+$',
        output, re.M)
    if sorted((int(kind), int(cell)) for kind, cell in statuses) != [
            (kind, cell) for kind in range(16) for cell in range(2)]:
        raise ValueError('incomplete five-member chemistry status matrix')
    replays = re.findall(r'^PASS public-replay kind=(\d+) ', output, re.M)
    if sorted(map(int, replays)) != [7, 15]:
        raise ValueError('incomplete public owner replay controls')
    for label in ('missing-face-owner', 'missing-periodic-pair'):
        native = re.findall(r'^PASS ' + label + r' native=(\d+) actualMissingResource=1 ', output, re.M)
        if sorted(map(int, native)) != [0, 1]:
            raise ValueError('incomplete actual resource control: ' + label)
    final = ('PASS real CUDA chemistry Trial routing, owner retry, terminal completion '
             'and unavailable-resource controls')
    if output.splitlines().count(final) != 1:
        raise ValueError('chemistry fixture did not finish cleanup')
    return dict(controls=4, failed_cells=16, failed_transactions=16,
                status_rows=32, public_replays=2, address_controls=4,
                **validate_storage_output(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--generate-only', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--storage-only', action='store_true')
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
            result = subprocess.run([str(binary)]+(['--storage-only'] if args.storage_only else []),
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise SystemExit(result.returncode)
        try:
            validate = validate_storage_output if args.storage_only else validate_run_output
            summary = validate((output / 'run.log').read_text())
        except ValueError as error:
            parser.exit(1, 'FAIL: ' + str(error) + '\n')
        (output / 'run-summary.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
