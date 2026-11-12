"""CHT rest-to-flow regression, using the real production executable and restart.

Complements pr34/cht_injection_native.py: its original default-velocity fixtures
already started with nonzero gas velocity and cannot catch a stale zero mirror.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'validation/molecular_transport'), str(ROOT/'tool/restart')]
import native_matrix as matrix
import thermal_particle_restart_current as wire

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--solver', type=Path, default=ROOT/'bin/CHT')
    parser.add_argument('--expect-stale', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    meta = json.loads((ROOT/'validation/molecular_transport/fixtures.json').read_text())['cht-contact']
    report = []
    for varying in (False, True):
        case = out/('varying' if varying else 'constant')
        matrix.prepare(ROOT/'validation/molecular_transport/fixtures/cht-contact', case, meta)
        # The parent contact fixture is a closed box; turn its left wall into
        # an actual inlet so the imposed pressure can establish gas flow.
        path = case/'constant/polyMesh/boundary'
        text, changed = re.subn(r'(left\s*\{\s*type\s+)wall', r'\1patch', path.read_text())
        assert changed == 1
        path.write_text(text)
        path = case/'constant/schedulingProperties'
        matrix.entry(path, 'gpuResidentPureGasOnly', 'false')
        matrix.entry(path, 'gpuResidentParticleCapacity', 8192)
        matrix.entry(path, 'gpuResidentDynamicInlet', 'true')
        path = case/'constant/particleProperties'
        matrix.entry(path, 'parcelMass', 1e-9)
        path.write_text(path.read_text()+'\ngpuResidentInletPatch left;\n')
        # Keep U at rest, with zeroGradient inlet and no particle velocity override.
        path = case/'0/U'
        path.write_text(path.read_text().replace(
            'left{type fixedValue;value uniform (0 0 0);}',
            'left{type zeroGradient;}'))
        path = case/'0/epsilonS'
        path.write_text(path.read_text().replace('left{type zeroGradient;}',
                        'left{type fixedValue;value uniform .002;}'))
        path = case/'0/p'
        path.write_text(path.read_text().replace('left{type zeroGradient;}',
                        'left{type fixedValue;value uniform 356334.11220656736;}'))
        pressure = 400000 if varying else 356334.11220656736
        fraction = .0022 if varying else .002
        (case/'constant/inletPressure.table').write_text(
            f'((0 356334.11220656736)(0.000002 {pressure}))\n')
        (case/'constant/inletVolumeFraction.table').write_text(
            f'((0 .002)(0.000002 {fraction}))\n')
        for key, value in [('endTime', 2e-6), ('deltaT', 1e-6),
                           ('writeInterval', 2e-6)]:
            matrix.entry(case/'system/controlDict', key, value)
        matrix.entry(case/'constant/scheme', 'maxDeltaT', 1e-6)
        matrix.run([args.solver.resolve(), '-case', case], case/'run.log')
        final = matrix.times(case)[-1]
        # A zero-particle result is only a useful counterexample if gas flow
        # really developed; reject a closed-wall or unchanged-rest fixture.
        velocity = (final/'U').read_text().split('internalField', 1)[1].split('boundaryField', 1)[0]
        vectors = re.findall(r'\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)', velocity)
        maximum_gas_ux = max(float(v[0]) for v in vectors)
        assert maximum_gas_ux > .001
        chunks = [data for _, data in wire.iter_chunks(final/'gpuResidentStrictParticles.dat')]
        count = sum(len(data['pm']) for data in chunks)
        mass = sum(float(data['pm'].sum()) for data in chunks)
        residual = (final/'gpuResidentStrictSourceResidual.dat').read_text().split()
        assert residual[0] == 'UGKP_SOURCE_RESIDUAL_SCHEMA1'
        remaining = sum(float(x) for x in residual[3::2])
        if args.expect_stale:
            assert count == 0 and mass == 0 and remaining == 0
        else:
            assert 0 < count < 8192 and mass > 0 and remaining >= 0
            for data in chunks:
                assert (data['pux'] > 0).all()
        report.append(dict(case=case.name, count=count, mass=mass, residual_mass=remaining, maximum_gas_ux=maximum_gas_ux))
    (out/'results.json').write_text(json.dumps(dict(
        solver=str(args.solver.resolve()),
        binary_sha256=hashlib.sha256(args.solver.read_bytes()).hexdigest(),
        expected_stale=args.expect_stale, cases=report), indent=2))
    print(json.dumps(report))

if __name__ == '__main__':
    main()
