#!/usr/bin/env python3
"""Generate compiler/build metadata without imposing source identity admission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def command_version(command):
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    return (result.stdout + result.stderr).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--species-count', type=int, default=2,
                        help='Shared UGKWP_GAS_SPECIES storage count; runtime model owns names/order')
    parser.add_argument('--arch')
    parser.add_argument('--nvcc', default='nvcc')
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'g++'))
    parser.add_argument('--compile-only', action='store_true', help='Record frontend-only compilation without CUDA')
    args = parser.parse_args()
    host = command_version([args.cxx, '--version'])
    cuda = 'COMPILE_ONLY_NO_CUDA' if args.compile_only else command_version([args.nvcc, '--version'])
    arch = 'COMPILE_ONLY' if args.compile_only else (args.arch if args.arch is not None else 'UNKNOWN')
    data = {'artifact_kind': 'COMPILE_ONLY' if args.compile_only else 'CHMT_BUILD_MANIFEST',
            'species_identity': 'runtime name/order hash', 'Ns': args.species_count,
            'precision': 'FP64', 'cuda_arch': arch, 'cuda_compiler': cuda,
            'host_compiler': host, 'openfoam_version': os.environ.get('WM_PROJECT_VERSION', 'UNKNOWN'),
            'openfoam_directory': os.environ.get('WM_PROJECT_DIR', 'UNKNOWN'),
            'source_fingerprint': 'UNKNOWN', 'upstream_base': 'UNKNOWN', 'solver_commit': 'UNKNOWN'}
    # This identifies the recorded metadata only; it is not a verified source hash.
    data['build_id'] = hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    macros = {'CHMT_BUILD_ID': data['build_id'], 'CHMT_SOURCE_FINGERPRINT': data['source_fingerprint'],
              'CHMT_UPSTREAM_BASE': data['upstream_base'], 'CHMT_SOLVER_COMMIT': data['solver_commit'],
              'CHMT_CUDA_COMPILER': cuda, 'CHMT_HOST_COMPILER': host, 'CHMT_CUDA_ARCH': arch,
              'CHMT_OPENFOAM_VERSION': data['openfoam_version'],
              'CHMT_BUILD_MANIFEST_JSON': json.dumps(data, sort_keys=True, separators=(',', ':'))}
    generated = '#pragma once\n' + ''.join('#define ' + key + ' ' + json.dumps(value) + '\n' for key, value in macros.items())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'CHMTBuildIdentity.H').write_text(generated)
    (args.output / 'build.json').write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    print(data['build_id'])


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print('CHMT build metadata: ' + str(exc), file=sys.stderr)
        sys.exit(2)
