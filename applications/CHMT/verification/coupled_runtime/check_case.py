#!/usr/bin/env python3
"""Independent CSV/wire-format checks. Alone this does not prove native execution.

No CHMT/common implementation is imported. Cell inventories are summed with
math.fsum; static closed walls imply delta(E_gas + E_material) = 0 and zero
mass loss. Signed GasSolid packet energy must equal delta(E_gas), and its
negative must equal delta(E_material). Restart wire clock/counters and FNV-1a
are decoded independently. The native launcher supplies actual CUDA provenance.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import struct


# Schema 5 includes active single species and SST integrals before the fixed tail.
FIELD_SCHEMA_HASH = 12202687703557820755
HEADER = struct.Struct('<8sQIIIIIIQQQQ')
ENERGY_ABS = 2e-5  # J; well below the prescribed millijoule-or-larger exchange
MASS_ABS = 1e-10  # kg


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, absolute=1e-10, relative=1e-10):
    return abs(actual-expected) <= absolute + relative*max(abs(actual), abs(expected))


def csv_rows(path):
    with Path(path).open(newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames and len(set(reader.fieldnames)) == len(reader.fieldnames), f'{path}: invalid CSV columns')
        rows = []
        for row in reader:
            require(None not in row and None not in row.values(), f'{path}: incomplete CSV row')
            values = {key: float(value) for key, value in row.items()}
            require(all(math.isfinite(value) for value in values.values()), f'{path}: nonfinite value')
            rows.append(values)
        return rows


def checkpoint_clock(path, species_count):
    wire = Path(path).read_bytes()
    require(len(wire) >= HEADER.size, 'truncated checkpoint header')
    magic, schema, endian, real_size, ns, nc, nr, ne, field_hash, config_size, state_size, digest = HEADER.unpack_from(wire)
    require((magic, schema, endian, real_size, ns, nc, nr, ne) == (b'CHMTCP2\0', 5, 0x01020304, 8, species_count, 2, 8, 8), 'checkpoint schema/precision/species mismatch')
    require(field_hash == FIELD_SCHEMA_HASH, 'checkpoint field schema differs from the independently supported layout')
    require(config_size > 0 and state_size > 0 and HEADER.size+config_size+state_size == len(wire), 'checkpoint payload length mismatch')
    checksum = 14695981039346656037
    for byte in wire[HEADER.size:]:
        checksum = ((checksum ^ byte) * 1099511628211) & ((1 << 64)-1)
    require(checksum == digest, 'checkpoint checksum mismatch')
    # Schema-5 fixed tail: time, three uint64 counters, then Budget. Budget
    # stores (45 + 5*Ns + Ne) binary64 values and a consumedPackets uint64.
    budget_size = (45+5*ns+ne)*8+8
    tail_size = 32+budget_size
    require(state_size >= tail_size, 'truncated checkpoint state tail')
    time, accepted, rejected, commit = struct.unpack_from('<dQQQ', wire, len(wire)-tail_size)
    require(math.isfinite(time) and time >= 0 and accepted == commit, 'checkpoint synchronized clock/counters invalid')
    return dict(time=time, accepted_steps=accepted, commit_sequence=commit, rejected_steps=rejected)


def total(rows, key):
    return math.fsum(row[key] for row in rows)


def validate_cells(rows, count, region, species):
    require(len(rows) == count and [r['cell'] for r in rows] == list(range(count)), f'{region} cell count/order mismatch')
    require(all(r['volume'] > 0 and r['temperature'] > 0 for r in rows), f'{region} volume/temperature invalid')
    mass_keys = ['mass_'+s for s in species] if region == 'gas' else ['mass_C0', 'mass_C1']+['mass_pore_'+s for s in species]
    require(all(r[k] >= 0 for r in rows for k in mass_keys), f'{region} negative species mass')
    if region == 'gas':
        require(all(r['mass'] > 0 for r in rows), 'gas mass is not positive')
        require(all(close(math.fsum(r[k] for k in mass_keys), r['mass'], MASS_ABS) for r in rows), 'gas species mass sum mismatch')
    else:
        require(all(r['porosity'] == 0 for r in rows), 'unexpected material porosity in dry case')
        require(all(close(r['total_energy'], 1000*(r['mass_C0']+r['mass_C1'])*r['temperature'], 1e-6, 1e-10) for r in rows), 'material energy/temperature caloric mismatch')
    return math.fsum(total(rows, key) for key in mass_keys)


def check_gas_thermodynamics(cells, species, tables):
    for row in cells:
        temperature = row['temperature']
        energies, pressure_terms = [], []
        for name in species:
            table = tables[name]
            require(table['min_temperature'] <= temperature <= table['max_temperature'], 'gas temperature outside species range')
            gas_constant = 8.31446261815324/table['molar_mass']
            c = table['coefficients']
            if table['model'] == 'linearCp':
                internal = c[2]+(c[0]-gas_constant)*temperature+c[1]*temperature**2/2
            else:
                require(table['model'] == 'NASA7', 'unknown independent gas thermo model')
                a = c[:7] if temperature <= table['mid_temperature'] else c[7:]
                internal = gas_constant*((a[0]-1)*temperature+a[1]*temperature**2/2+a[2]*temperature**3/3+a[3]*temperature**4/4+a[4]*temperature**5/5+a[5])
            mass = row['mass_'+name]
            energies.append(mass*internal)
            pressure_terms.append(mass*gas_constant*temperature/row['volume'])
        kinetic = math.fsum(row[k]**2 for k in ('momentum_x', 'momentum_y', 'momentum_z'))/(2*row['mass'])
        require(close(row['total_energy'], math.fsum(energies)+kinetic, 1e-7, 2e-10), 'gas energy/temperature caloric mismatch')
        require(close(row['pressure'], math.fsum(pressure_terms), 1e-7, 2e-10), 'gas pressure/temperature equation of state mismatch')


def snapshots(case):
    out = Path(case)/'chmtOutput'
    files = sorted(out.glob('accepted-*.csv'), key=lambda p: int(p.stem.split('-')[-1]))
    require(files, 'no synchronized accepted output')
    steps = {int(path.stem.split('-')[-1]) for path in files}
    for prefix in ('gas', 'material', 'exchange', 'checkpoint'):
        found = {int(path.stem.split('-')[-1]) for path in out.glob(prefix+'-*')}
        require(found == steps, 'missing/orphan '+prefix+' output outside accepted endpoints')
    result = []
    for path in files:
        n = int(path.stem.split('-')[-1])
        summary = csv_rows(path)
        require(len(summary) == 1, 'accepted summary must contain one row')
        result.append(dict(step=n, summary=summary[0], gas=csv_rows(out/f'gas-{n}.csv'),
                           material=csv_rows(out/f'material-{n}.csv'), exchange=csv_rows(out/f'exchange-{n}.csv'),
                           checkpoint=out/f'checkpoint-{n}/state.chmt'))
    return result


def check_case(case):
    case = Path(case)
    spec = json.loads((case/'verification.json').read_text())
    states = snapshots(case)
    require(len(states) >= 2, 'no accepted coupled time advancement')
    species = spec['species']
    initial = states[0]
    if 'restart_directory' not in spec:
        require(initial['step'] == 0 and initial['summary']['time'] == 0, 'initial synchronized state is missing')
    else:
        saved = checkpoint_clock(Path(spec['restart_directory'])/'state.chmt', len(species))
        require(initial['step'] == saved['accepted_steps'] and close(initial['summary']['time'], saved['time'], 1e-15, 1e-13), 'restart initial synchronized state differs from checkpoint')
    initial_gas = total(initial['gas'], 'total_energy')
    initial_material = total(initial['material'], 'total_energy')
    initial_energy = initial_gas+initial_material
    energy_tolerance = max(ENERGY_ABS, abs(initial_energy)*2e-13)
    initial_mass = total(initial['gas'], 'mass')
    initial_species = {s: total(initial['gas'], 'mass_'+s) for s in species}
    maximum_residual = 0.
    previous = None
    exchanged = 0.
    for state in states:
        n, summary = state['step'], state['summary']
        require(summary['accepted_steps'] == n and summary['commit_sequence'] == n, 'output is not synchronized to accepted commit')
        require(summary['gas_mode'] == (2 if spec['scenario'] == 'chemistry' else 0 if spec['scenario'] == 'single' else 1) and summary['thermo_hash'] > 0, 'wrong accepted gas model identity')
        for key in ('gas_cells', 'material_cells', 'interface_faces'):
            require(summary[key] == spec[key], key+' mismatch')
        clock = checkpoint_clock(state['checkpoint'], len(species))
        require(close(clock['time'], summary['time'], 1e-15, 1e-13), 'checkpoint/output time mismatch')
        require(clock['accepted_steps'] == n and clock['commit_sequence'] == n, 'checkpoint/output synchronized counters mismatch')
        require(close(total(state['gas'], 'mass'), initial_mass, MASS_ABS), 'closed gas mass balance failed')
        validate_cells(state['gas'], spec['gas_cells'], 'gas', species)
        check_gas_thermodynamics(state['gas'], species, spec['gas_thermo'])
        material_mass = validate_cells(state['material'], spec['material_cells'], 'material', species)
        require(close(material_mass, spec['initial_material_mass'], MASS_ABS), 'closed material mass balance failed')
        require(close(total(state['gas'], 'volume'), 1.) and close(total(state['material'], 'volume'), 1.), 'static region volume changed')
        gas_energy = total(state['gas'], 'total_energy')
        material_energy = total(state['material'], 'total_energy')
        residual = math.fsum([r['total_energy'] for region in ('gas','material') for r in state[region]]
                             + [-r['total_energy'] for region in ('gas','material') for r in initial[region]])
        maximum_residual = max(maximum_residual, abs(residual))
        require(abs(residual) <= energy_tolerance, 'closed total energy balance failed')
        if spec['scenario'] != 'chemistry':
            for s in species:
                require(close(total(state['gas'], 'mass_'+s), initial_species[s], MASS_ABS), 'closed species mass balance failed: '+s)
        else:
            for e in range(4):
                actual = math.fsum(total(state['gas'], 'mass_'+s)*spec['element_moles_per_kg'][s][e] for s in species)
                expected = math.fsum(initial_species[s]*spec['element_moles_per_kg'][s][e] for s in species)
                require(close(actual, expected, 1e-9, 1e-9), 'chemical element conservation failed')
        if previous is None and n == 0:
            require(close(initial_mass, spec['initial_gas_mass'], MASS_ABS), 'initial gas mass differs from prescribed case')
            require(close(initial_gas, spec['initial_gas_energy'], 1e-5, 1e-10), 'initial gas energy differs from independent thermodynamic reference')
            require(close(initial_material, spec['initial_material_energy'], ENERGY_ABS, 1e-13), 'initial material energy differs from prescribed case')
            require(all(close(r['temperature'], spec['initial_gas_temperature'], 1e-6) for r in state['gas']), 'initial gas temperature differs from case')
            require(all(close(r['temperature'], spec['initial_solid_temperature'], 1e-6) for r in state['material']), 'initial material temperature differs from case')
        if previous is not None:
            require(n == previous['step']+1 and summary['time'] > previous['summary']['time'], 'accepted sequence/time did not advance exactly once')
            require(summary['time']-previous['summary']['time'] <= spec['coupling_interval']*(1+1e-10), 'accepted interval exceeds configured window')
            require(state['exchange'], 'accepted window has no nonzero coupling exchange')
            for packet in state['exchange']:
                require(packet['kind'] in (0, 3) and packet['consumer_mask'] == 3, 'exchange consumer/kind mismatch')
                if packet['kind'] == 3:
                    require(packet['energy'] == packet['conductive'] == 0, 'dry contact has nonzero pore exchange')
                require(packet['gas_cell'].is_integer() and 0 <= packet['gas_cell'] < spec['gas_cells'] and packet['solid_cell'].is_integer() and 0 <= packet['solid_cell'] < spec['material_cells'], 'exchange cell address invalid')
                require(packet['mass'] == 0, 'thermal contact exchanged mass')
                require(all(abs(packet[k]) < 1e-12 for k in ('advective', 'pressure_work', 'viscous_work', 'radiation', 'liquid_kinetic_advection')), 'unexpected nonconductive exchange')
                require(close(packet['energy'], packet['conductive'], 1e-12, 1e-12), 'exchange energy decomposition mismatch')
            energy = total(state['exchange'], 'energy')
            require(energy < -1e-6, 'nonzero hot-gas-to-solid exchange missing or reversed')
            # Four quarter-square-metre conformal faces; both adjacent cell
            # centres are 0.25 m from contact, with k_gas=k_solid=1 W/(m K).
            # Thus UA=1/(0.25/1+0.25/1)=2 W/K. Endpoint extrema bound this
            # short monotone dry-contact case (5% slack covers temporal lag).
            dt = summary['time']-previous['summary']['time']
            gas_temperatures = [r['temperature'] for snap in (previous,state) for r in snap['gas']]
            solid_temperatures = [r['temperature'] for snap in (previous,state) for r in snap['material']]
            lower = 2*dt*(min(gas_temperatures)-max(solid_temperatures))
            upper = 2*dt*(max(gas_temperatures)-min(solid_temperatures))
            require(.95*lower-energy_tolerance <= -energy <= 1.05*upper+energy_tolerance,
                    'thermal exchange disagrees with independent contact conductance')
            gas_change = gas_energy-total(previous['gas'], 'total_energy')
            solid_change = material_energy-total(previous['material'], 'total_energy')
            require(abs(gas_change-energy) <= energy_tolerance, 'gas energy change disagrees with exchange')
            require(abs(solid_change+energy) <= energy_tolerance, 'material energy change disagrees with exchange')
            exchanged += energy
        previous = state
    final = states[-1]
    require(close(final['summary']['time'], spec['end_time'], 1e-15, 1e-12), 'final accepted time differs from configured end')
    solid_change = total(final['material'], 'total_energy')-initial_material
    gas_change = total(final['gas'], 'total_energy')-initial_gas
    require(solid_change > energy_tolerance and gas_change < -energy_tolerance, 'both participants must show resolved nonzero heat transfer')
    require(max(r['temperature'] for r in final['material']) > max(r['temperature'] for r in initial['material']), 'material temperature did not rise')
    if spec['scenario'] == 'chemistry':
        response = max(abs(total(final['gas'], 'mass_'+s)-initial_species[s]) for s in species)
        require(response > 1e-10, 'reactive mixture produced no resolved chemical species response')
    return dict(evidence='independent-output-checks-only', scenario=spec['scenario'], accepted_windows=len(states)-1,
                final_time=final['summary']['time'], gas_energy_change=gas_change, material_energy_change=solid_change,
                signed_exchange_energy=exchanged, maximum_total_energy_residual=maximum_residual,
                energy_tolerance=energy_tolerance)


def compare_restart(reference, resumed, source):
    full, continuation = snapshots(reference), snapshots(resumed)
    source = Path(source)/'state.chmt'
    require(source.read_bytes() == continuation[0]['checkpoint'].read_bytes(), 'restart initial checkpoint is not a bitwise reserialization')
    require(full[-1]['step'] == continuation[-1]['step'], 'restart accepted count differs')
    for region in ('gas', 'material'):
        left, right = full[-1][region], continuation[-1][region]
        require(len(left) == len(right), 'restart final cell count differs')
        for a, b in zip(left, right):
            require(a.keys() == b.keys(), 'restart output columns differ')
            for key in a:
                require(close(a[key], b[key], 1e-10, 5e-12), f'restart final {region} {key} differs from uninterrupted run')
    require(full[-1]['summary'] == continuation[-1]['summary'], 'restart final synchronized summary differs')
    return 'restart continuation matches uninterrupted inventories'


def compare_diffusion(diffusive, control):
    def variance(case):
        cells = snapshots(case)[-1]['gas']
        mass = total(cells, 'mass')
        mean = total(cells, 'mass_S0')/mass
        return math.fsum(r['mass']*(r['mass_S0']/r['mass']-mean)**2 for r in cells)/mass
    active, disabled = variance(diffusive), variance(control)
    require(active < disabled-1e-10, 'configured molecular diffusion has no resolved mixing beyond zero-D control')
    return dict(diffusive_variance=active, zero_diffusion_variance=disabled)


def compare_chemistry(reacting, control):
    active, frozen = snapshots(reacting), snapshots(control)
    species = [key for key in active[0]['gas'][0] if key.startswith('mass_')]
    changes = {key: total(active[-1]['gas'], key)-total(active[0]['gas'], key) for key in species}
    drift = max(abs(total(frozen[-1]['gas'], key)-total(frozen[0]['gas'], key)) for key in species)
    response = max(abs(value) for value in changes.values())
    require(response > max(1e-10, 100*drift), 'chemical source response is unresolved relative to disabled-chemistry control')
    require(changes.get('mass_H2', 0) < -1e-10 and changes.get('mass_O2', 0) < -1e-10
            and changes.get('mass_H2O', 0) > 1e-10, 'chemical source did not consume H2/O2 and form water')
    require(drift <= MASS_ABS, 'disabled-chemistry control has a net species source')
    return dict(maximum_species_response=response, disabled_chemistry_species_drift=drift, species_mass_changes=changes)


def compare_refinement(coarse, fine):
    spec = json.loads((Path(coarse)/'verification.json').read_text())
    refined = json.loads((Path(fine)/'verification.json').read_text())
    require(close(refined['coupling_interval']*2, spec['coupling_interval'], 1e-15, 1e-13)
            and close(refined['gas_max_dt']*2, spec.get('gas_max_dt', 1e-5), 1e-15, 1e-13), 'refinement must halve both H and gas dt')
    original, smaller = snapshots(coarse), snapshots(fine)
    require(len(smaller) > len(original), 'refinement did not produce more accepted macro intervals')
    require(original[-1]['summary']['time'] == smaller[-1]['summary']['time'], 'refinement final times differ')
    heat = total(original[-1]['material'], 'total_energy')-total(original[0]['material'], 'total_energy')
    fine_heat = total(smaller[-1]['material'], 'total_energy')-total(smaller[0]['material'], 'total_energy')
    # An acceptance stability bound, not an assertion of a measured convergence
    # order: halving H and dt must preserve physical heat to 0.5% plus binary64
    # inventory-subtraction allowance for this large solid heat capacity.
    allowance = .005*abs(fine_heat)+max(2*ENERGY_ABS, abs(total(original[0]['material'], 'total_energy'))*4e-13)
    require(heat > 0 and fine_heat > 0 and abs(heat-fine_heat) <= allowance, 'refinement changed heat transfer beyond the 0.5% stability bound')
    for a, b in zip(original[-1]['gas'], smaller[-1]['gas']):
        require(close(a['mass'], b['mass'], 1e-9, 1e-6) and close(a['temperature'], b['temperature'], .01, 0), 'refinement changed gas inventories/temperature beyond tolerance')
    return dict(coarse_heat=heat, refined_heat=fine_heat, heat_difference=abs(heat-fine_heat),
                heat_difference_allowance=allowance, expectation='H/2 and dt/2: heat agrees within 0.5% plus roundoff; no convergence-order claim')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--restart-from', type=Path)
    parser.add_argument('--diffusion-control', type=Path)
    parser.add_argument('--chemistry-control', type=Path)
    parser.add_argument('--refinement', type=Path)
    args = parser.parse_args()
    try:
        report = check_case(args.case)
        if args.reference or args.restart_from:
            require(args.reference and args.restart_from, 'restart comparison requires both reference and restart-from')
            report['restart'] = compare_restart(args.reference, args.case, args.restart_from)
        if args.diffusion_control:
            check_case(args.diffusion_control)
            report['diffusion'] = compare_diffusion(args.case, args.diffusion_control)
        if args.chemistry_control:
            check_case(args.chemistry_control)
            report['chemistry'] = compare_chemistry(args.case, args.chemistry_control)
        if args.refinement:
            check_case(args.refinement)
            report['refinement'] = compare_refinement(args.case, args.refinement)
        print(json.dumps(report, indent=2, sort_keys=True))
    except (ValueError, OSError, KeyError, struct.error) as exc:
        parser.exit(1, f'FAIL: coupled independent check: {exc}\n')


if __name__ == '__main__':
    main()
