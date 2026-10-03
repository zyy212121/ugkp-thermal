#!/usr/bin/env python3
"""Native moment-path fixture runner: SOURCE OUT APP BITS.

SOURCE is one solver library root. Requires a CUDA compiler and device.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 5:
        print('usage: moment_publication.py SOURCE OUT APP BITS', file=sys.stderr)
        return 2
    root = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    app = sys.argv[3]
    try:
        bits = int(sys.argv[4])
    except ValueError:
        print('BITS must be 32 or 64', file=sys.stderr)
        return 2
    if (app, bits) not in {('gasUGKP', 64), ('FSH', 64), ('CHT', 64), ('CHT', 32)}:
        print('supported configurations: gasUGKP64, FSH64, CHT64, CHT32', file=sys.stderr)
        return 2
    native = root / 'applications' / app / ('gpu' if app == 'CHT' else 'private_backend')
    fixture = Path(__file__).resolve().with_name('moment_publication_paths.cu')
    source_files = [fixture, native / 'GpuResidentStrict.cu'] + [root / 'common' / name for name in (
        'GpuParticleMoments.cuh', 'GpuSegmentedMomentWorkers.cuh',
        'GpuParticleMomentRange.inl', 'GpuParticleMomentContribution.inl',
        'CsrPersistentQueue.cuh',
    )]
    missing = [str(path) for path in source_files if not path.is_file()]
    if missing:
        print('missing fixture/source file(s): ' + ', '.join(missing), file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)
    generated = out / 'moment_publication_paths.cu'
    generated.write_bytes(fixture.read_bytes())
    executable = out / 'moment_publication'
    command = [
        str(Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')) / 'bin' / 'nvcc'),
        '-std=c++17', '-O3', '-arch=' + os.environ.get('UGKWP_CUDA_ARCH', 'sm_89'),
        '--fmad=' + ('false' if app == 'CHT' else 'true'),
        '-DUGKP_DEVELOPMENT_PROBES=1', f'-DUGKWP_GPU_REAL_BITS={bits}',
        f'-DMH04_THERMAL={int(app != "gasUGKP")}', f'-DMH04_CHT={int(app == "CHT")}',
        '-I' + str(native), '-I' + str(root / 'common'),
        '-I' + str(root / 'applications' / app / 'gpu'),
        str(generated), '-o', str(executable),
    ]
    if app == 'CHT':
        wall_source = native / 'GpuWallEnergy64.cu'
        command.append(str(wall_source))
        source_files.append(wall_source)
    manifest = {
        'app': app, 'bits': bits, 'source_root': str(root),
        'source_files': [{'path': str(path), 'sha256': sha256(path)} for path in source_files],
        'compile_command': command, 'run_command': [str(executable)],
        'compile_returncode': None, 'run_returncode': None,
    }
    manifest_path = out / 'fixture-manifest.json'

    def save_manifest():
        manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    save_manifest()
    try:
        with (out / 'build.log').open('w', encoding='utf-8') as log:
            built = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        manifest['compile_returncode'] = built.returncode
        save_manifest()
        if built.returncode:
            print((out / 'build.log').read_text(encoding='utf-8', errors='replace')[-6000:])
            return built.returncode
        ran = subprocess.run([str(executable)], capture_output=True, text=True, check=False)
        (out / 'run.log').write_text(ran.stdout + ran.stderr, encoding='utf-8')
        manifest['run_returncode'] = ran.returncode
        save_manifest()
        print(app, bits, ran.stdout, ran.stderr)
        return ran.returncode
    except OSError as error:
        manifest['error'] = str(error)
        save_manifest()
        print(f'fixture command failed to start: {error}', file=sys.stderr)
        return 127


if __name__ == '__main__':
    sys.exit(main())
