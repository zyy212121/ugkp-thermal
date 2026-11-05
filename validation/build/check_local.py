#!/usr/bin/env python3
"""Exercise every locally built application using repository-owned fixtures."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'validation/transient_particles'))
import check_native as native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    bindir = args.bin_dir.resolve()
    env = dict(os.environ, OMP_NUM_THREADS='4', GAS_UGKP_CUDA_STACK_BYTES='65536',
               CHMT_CUDA_STACK_BYTES='65536',
               FSH_CUDA_BACKEND=str(bindir / 'FSHCudaBackend'))
    report = {'binaries': {}, 'checks': []}
    for name in ['gasUGKP', 'gasUGKPSteady', 'FSH', 'FSHCudaBackend', 'CHT', 'CHMT']:
        p = bindir / name
        assert p.is_file() and os.access(p, os.X_OK), p
        report['binaries'][name] = {'path': str(p), 'sha256': native.sha(p)}

    def save():
        (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')

    def command(name, argv):
        with (out / (name + '.log')).open('w') as log:
            result = subprocess.run(list(map(str, argv)), env=env, stdout=log, stderr=subprocess.STDOUT)
        report['checks'].append({'name': name, 'command': list(map(str, argv)), 'exit': result.returncode})
        save()
        assert result.returncode == 0, (name, result.returncode)
        print(name, 'PASS', flush=True)

    command('empty-restart', [sys.executable, ROOT / 'validation/transient_particles/check_empty_restart.py',
                            '--transient', bindir / 'gasUGKP', '--steady', bindir / 'gasUGKPSteady',
                            '--output', out / 'empty-restart'])
    command('coupled', [sys.executable, ROOT / 'validation/gpu_coupled_pipeline/layer1.py',
                        '--baseline', bindir, '--work', out / 'coupled', '--cases',
                        'gas-wave', 'cht-contact', 'cht-mixture', 'chmt-mixture', 'chmt-moving-short'])
    command('heat-controls', [sys.executable, ROOT / 'validation/steady_cuda/check_heat_failure.py',
                              '--binary', bindir / 'gasUGKPSteady', '--output', out / 'heat-controls'])
    # An ordinary single-gas parcel fixture exercises FSH's real IPC backend.
    fsh = out / 'fsh-particles'
    native.copy_inputs(ROOT / 'validation/steady_cuda/evidence/cases/particles', fsh)
    native.control(fsh, 200, write=20)
    native.set_entry(fsh / 'system/controlDict', 'application', 'FSH')
    command('fsh', [bindir / 'FSH', '-case', fsh])
    final = native.dirs(fsh)[-1]
    assert abs(float(final.name) - .004) < 1e-12
    with (final / 'gpuResidentStrictParticles.dat').open('rb') as stream:
        header = stream.readline().decode().split()
    assert header[0] == 'UGKP_FSH_PARTICLES_SCHEMA8_FP64_FIELDS_BIN'
    assert int(header[1]) > 0
    report['fsh_live_parcels'] = int(header[1])
    # Static boundaries and matching physical dt: compare the actual two drivers.
    cases = []
    for name in ['gasUGKP', 'gasUGKPSteady']:
        case = out / ('mixture-particles-' + name)
        native.copy_inputs(ROOT / 'validation/steady_cuda/evidence/cases/mixture_particles', case)
        native.control(case, 200, steady=name == 'gasUGKPSteady', write=20)
        command(name + '-mixture-particles', [bindir / name, '-case', case])
        cases.append(case)
    report['physical_driver_comparisons'] = [native.compare(a, b, 32)
        for a, b in zip(native.dirs(cases[0])[1:], native.dirs(cases[1])[1:])]
    assert len(report['physical_driver_comparisons']) == 10
    report['pass'] = True
    save()


if __name__ == '__main__':
    main()
