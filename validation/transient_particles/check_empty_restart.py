#!/usr/bin/env python3
"""Native empty-parcel checkpoint regression for both physical-time drivers.

Source OpenFOAM bashrc first. --expect-before checks the historical silent
mismatch; the default requires restart equivalence and independent porosity
source values. No production code or checkpoint history is synthesized.
"""
from pathlib import Path
import argparse
import json
import shutil
import struct
import sys
import check_native as native


def eps_prev(directory):
    values = (directory / 'gpuResidentStrictEpsGPrev.dat').read_text().split()
    assert int(values[0]) == 1
    return float(values[1])


def scalar(directory, name):
    return native.fields.internal(directory / name, 1)[0][0]


def prepare(directory, kind, steady):
    native.failure.prepare(directory, .175, .001, 2, 100)
    native.control(directory, 2, steady=steady, write=1)
    # No inlet source: initial boundary velocity is zero on both open faces.
    native.set_entry(directory / 'constant/particleProperties',
                     'particleGasHeatTransferModel', 'none')
    particle = directory / '0/gpuResidentStrictParticles.dat'
    if kind == 'empty':
        particle.write_bytes(b'UGKP_PARTICLES_SCHEMA1_BIN 0 1\n')
        # Initial moments must also describe an empty state.
        p = directory / '0/epsilonS'
        native.set_entry(p, 'internalField', 'uniform 0')
    else:
        data = bytearray(particle.read_bytes())
        start = data.index(b'\n') + 1 + 4
        struct.pack_into('<d', data, start, .699999 if kind == 'exit' else .35)
        struct.pack_into('<d', data, start + 3*8, 1.)
        struct.pack_into('<d', data, start + 6*8, 1000.)
        particle.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['transient', 'steady', 'output']:
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--expect-before', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    report = {'expect_before': args.expect_before, 'runs': [], 'comparisons': [],
              'binaries': {k: {'path': str(getattr(args, k).resolve()),
                               'sha256': native.sha(getattr(args, k))}
                           for k in ['transient', 'steady']}}

    def save():
        (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')

    def execute(driver, case, mesh=False, failure=False):
        row = native.run(getattr(args, driver).resolve(), case, mesh)
        row.update(case=case.name, driver=driver)
        report['runs'].append(row)
        save()
        assert (row['exit'] != 0) == failure, row
        return row

    for driver in ['transient', 'steady']:
        steady = driver == 'steady'
        for kind in (['exit'] if args.expect_before else ['exit', 'nonempty', 'empty']):
            case = out / (driver + '-' + kind)
            prepare(case, kind, steady)
            execute(driver, case, mesh=True)
            times = native.dirs(case)
            assert len(times) == 3
            checkpoint, final = times[1:]
            count = len(native.parcel(checkpoint / 'gpuResidentStrictParticles.dat'))
            history = eps_prev(checkpoint)
            assert count == (1 if kind == 'nonempty' else 0), (kind, count)
            assert abs(history - (1. if kind == 'empty' else .9999)) < 1e-15
            assert abs(scalar(checkpoint, 'epsilonS') - (0 if kind != 'nonempty' else .0001)) < 1e-15
            restart = out / (case.name + '-restart')
            native.copy_inputs(case, restart)
            shutil.copytree(checkpoint, restart / checkpoint.name)
            native.set_entry(restart / 'system/controlDict', 'startFrom', 'latestTime')
            execute(driver, restart)
            actual = native.dirs(restart)[-1]
            assert actual.name == final.name
            row = {'driver': driver, 'kind': kind, 'checkpoint': checkpoint.name,
                   'checkpoint_parcels': count, 'checkpoint_epsGPrev': history,
                   'continuous': {name: scalar(final, name) for name in ['rho', 'rhoY_A', 'rhoY_B', 'rhoE']},
                   'restart': {name: scalar(actual, name) for name in ['rho', 'rhoY_A', 'rhoY_B', 'rhoE']},
                   'continuous_epsGPrev': eps_prev(final), 'restart_epsGPrev': eps_prev(actual)}
            report['comparisons'].append(row)
            save()
            if args.expect_before:
                assert abs(row['continuous']['rho'] - row['restart']['rho']) > 9e-5
                assert abs(eps_prev(actual) - .9999) < 1e-15
                assert eps_prev(final) == 1.
            else:
                row['comparison'] = native.compare(final, actual, 1)
                assert abs(eps_prev(final) - eps_prev(actual)) < 1e-15
            if kind == 'exit':
                # Uniform one-cell open gas: paired boundary fluxes cancel.
                # Independently evaluate the existing explicit porosity source.
                scale = 1. - (1. - history) * 2e-5 / (2e-5 + sys.float_info.epsilon)
                expected = {name: scalar(checkpoint, name)*scale
                            for name in ['rho', 'rhoY_A', 'rhoY_B']}
                expected['rhoE'] = scalar(checkpoint, 'rhoE')*scale + (scale-1.)*scalar(checkpoint, 'p')
                row['expected_source_result'] = expected
                row['source_error'] = {name: abs(row['continuous'][name]-value) for name, value in expected.items()}
                assert all(row['source_error'][name] < (1e-6 if name == 'rhoE' else 1e-12) for name in expected), row
            print(driver, kind, 'PASS', flush=True)
            save()
        if not args.expect_before:
            case = out / (driver + '-heat-failure')
            native.failure.prepare(case, 1., 1e-6, 1, 100)
            native.control(case, 1, steady=steady, write=1)
            execute(driver, case, mesh=True, failure=True)
            log = (case / 'run.log').read_text()
            assert 'site=30700' in log and 'detailCode=4' in log and len(native.dirs(case)) == 1
            print(driver, 'heat failure rejected before output', flush=True)
    report['pass'] = True
    save()


if __name__ == '__main__':
    main()
