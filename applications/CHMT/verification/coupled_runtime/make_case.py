#!/usr/bin/env python3
"""Generate the small real two-region acceptance case, without running a solver."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

APP = Path(__file__).resolve().parents[2]
ROOT = APP.parents[1]


def initial_scalar(path, value):
    text = path.read_text()
    if isinstance(value, list):
        entry = 'nonuniform List<scalar>\n8\n(\n' + '\n'.join(format(v, '.17g') for v in value) + '\n)'
    else:
        entry = 'uniform ' + format(value, '.17g')
    path.write_text(re.sub(r'internalField\s+uniform\s+[^;]+;', 'internalField ' + entry + ';', text))


def make_case(path, scenario='thermal', end_time=None, restart=None, refine=False):
    path = Path(path)
    if path.exists():
        raise ValueError('case destination already exists; use a fresh directory')
    chemical = scenario in ('chemistry', 'chemistry_control')
    base_interval = 1e-5 if chemical else .0005
    interval = base_interval/(2 if refine else 1)
    end_time = 2*base_interval if end_time is None else end_time
    if not 0 < end_time <= 2*base_interval:
        raise ValueError('end time must be positive and at most two coupling intervals')
    subprocess.run([sys.executable, str(APP/'verification/coupled_input/make_case.py'), str(path)] + (['--h2o2'] if chemical else []), check=True)
    gas_path = path/'constant/gasModelProperties'
    properties = (path/'constant/chmtProperties').read_text()
    properties = re.sub(r'modelFingerprint\s+"[^"]+";\s*', '', properties)
    properties = properties.replace('coupledInputFixture', 'coupledNativeAcceptance')
    properties = properties.replace('couplingInterval 0.01', f'couplingInterval {interval:.17g}')
    micro_dt = (1e-6 if chemical else 1e-5)/(2 if refine else 1)
    properties = properties.replace('gasMaxDt 0.00001', f'gasMaxDt {micro_dt:.17g}')
    control = (path/'system/controlDict').read_text()
    control = re.sub(r'endTime\s+[^;]+;', f'endTime {end_time:.17g};', control)
    control = re.sub(r'deltaT\s+[^;]+;', f'deltaT {micro_dt:.17g};', control)
    (path/'system/controlDict').write_text(control)
    spec = dict(scenario=scenario, end_time=end_time, coupling_interval=interval, gas_max_dt=micro_dt,
                gas_cells=8, material_cells=8, interface_faces=4,
                initial_gas_temperature=600., initial_solid_temperature=300.,
                initial_gas_mass=1., initial_material_mass=1000.,
                initial_gas_energy=(1040-8.31446261815324/.028)*600,
                initial_material_energy=3e8, species=['S0', 'S1'])
    if scenario in ('diffusion', 'diffusion_control'):
        # Identical caloric and molecular properties suppress composition-driven
        # pressure differences; the zero-D companion isolates actual diffusion.
        gas = gas_path.read_text().replace('molarMass 0.032', 'molarMass 0.028')
        coefficient = '0.02' if scenario == 'diffusion' else '0'
        gas_path.write_text(gas.replace('diffusion { model none; }', f'diffusion {{ model constant; coefficients ({coefficient} {coefficient}); }}'))
        initial_scalar(path/'0/Y_S0', [.8, .2]*4)
        initial_scalar(path/'0/Y_S1', [.2, .8]*4)
    if chemical:
        data = ROOT/'common/chemistry/mechanisms'
        reference = json.loads((data/'h2o2.cantera-reactor-reference.json').read_text())
        initial = next(c['states'][0] for c in reference['cases'] if c['name'] == 'nitrogen_1200K_1atm')
        spec['species'] = json.loads((data/'h2o2.manifest.json').read_text())['species']
        spec.update(initial_gas_mass=initial['density'], initial_gas_temperature=1200.,
                    initial_gas_energy=initial['density']*initial['specificInternalEnergy'])
        initial_scalar(path/'0/rho', initial['density'])
        initial_scalar(path/'0/T', 1200.)
        for species, fraction in zip(spec['species'], initial['massFractions']):
            initial_scalar(path/('0/Y_'+species), fraction)
        # Record only atomic accounting, independently evaluated from the pinned
        # species molecular weights and atom counts; no CHMT math is imported.
        gas = gas_path.read_text()
        elements = {}
        for name in spec['species']:
            body = re.search(r'\b'+name+r'\s*\{([^{}]+)\}', gas).group(1)
            mw = float(re.search(r'molarMass\s+([^;]+);', body).group(1))
            atoms = [float(v) for v in re.search(r'atoms\s*\(([^)]+)\)', body).group(1).split()]
            elements[name] = [atom/mw for atom in atoms]
        spec['element_moles_per_kg'] = elements
    if scenario == 'chemistry_control':
        gas = gas_path.read_text().replace('gasMode mixtureChemistry;', 'gasMode mixtureFrozen;')
        gas = re.sub(r'(?m)^mechanism\s+[^;]+;\s*|^phase\s+[^;]+;\s*', '', gas)
        gas_path.write_text(gas)
    thermo = {}
    gas = gas_path.read_text()
    for name in spec['species']:
        body = re.search(r'\b'+name+r'\s*\{([^{}]+)\}', gas).group(1)
        scalar = lambda key: float(re.search(key+r'\s+([^;]+);', body).group(1))
        model = re.search(r'model\s+([^;]+);', body).group(1).strip()
        thermo[name] = dict(model=model, molar_mass=scalar('molarMass'),
                            min_temperature=scalar('minTemperature'), max_temperature=scalar('maxTemperature'),
                            coefficients=[float(v) for v in re.search(r'coefficients\s*\(([^)]+)\)', body).group(1).split()])
        if model == 'NASA7':
            thermo[name]['mid_temperature'] = scalar('midTemperature')
    spec['gas_thermo'] = thermo
    if scenario == 'single':
        (path/'constant/materialGasProperties').write_text(gas_path.read_text().replace('object gasModelProperties;', 'object materialGasProperties;'))
        gas_path.write_text('FoamFile { version 2.0; format ascii; class dictionary; object gasModelProperties; }\ngasMode single;\n')
        properties += 'singleGasSpecies S0;\n'
        for name in spec['species']:
            (path/('0/Y_'+name)).unlink()
    digest = hashlib.sha256((properties+gas_path.read_text()+(gas if scenario == 'single' else '')).encode()).hexdigest()
    properties += f'modelFingerprint "{digest}";\n'
    if restart is not None:
        restart = Path(restart).resolve()
        if '"' in str(restart) or '\n' in str(restart):
            raise ValueError('restart path contains an unsupported character')
        properties += f'restartDirectory "{restart}";\n'
        spec['restart_directory'] = str(restart)
    (path/'constant/chmtProperties').write_text(properties)
    (path/'verification.json').write_text(json.dumps(spec, indent=2, sort_keys=True)+'\n')
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--single', action='store_true')
    mode.add_argument('--diffusion', action='store_true')
    mode.add_argument('--diffusion-control', action='store_true')
    mode.add_argument('--chemistry', action='store_true')
    mode.add_argument('--chemistry-control', action='store_true')
    parser.add_argument('--refine', action='store_true', help='Halve macro interval and gas time-step cap at the same final time')
    parser.add_argument('--end-time', type=float)
    parser.add_argument('--restart', type=Path)
    args = parser.parse_args()
    scenario = 'single' if args.single else 'chemistry_control' if args.chemistry_control else 'chemistry' if args.chemistry else 'diffusion' if args.diffusion else 'diffusion_control' if args.diffusion_control else 'thermal'
    try:
        make_case(args.case, scenario, args.end_time, args.restart, args.refine)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(2, f'coupled case generation: {exc}\n')


if __name__ == '__main__':
    main()
