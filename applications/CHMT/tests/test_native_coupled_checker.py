"""Deliberately synthetic checker self-tests; never native CUDA evidence."""
from pathlib import Path
import csv
import importlib.util
import json
import struct
import subprocess
import sys
import pytest

APP = Path(__file__).resolve().parents[1]
CHECKER = APP / 'verification/coupled_runtime/check_case.py'


def load_checker():
    assert CHECKER.is_file(), 'independent coupled runtime checker is missing'
    spec = importlib.util.spec_from_file_location('coupled_checker', CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_csv(path, rows, columns=None):
    with path.open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def checkpoint(path, time, step):
    path.parent.mkdir(parents=True)
    # Minimal synthetic body sufficient for the independent wire header/tail
    # checks. It is intentionally NOT a loadable production checkpoint.
    model = b'synthetic-model'
    state = b'synthetic-state' + struct.pack('<dQQQ', time, step, 0, step) + bytes((45 + 5*2 + 8)*8 + 8)
    body = model + state
    digest = 14695981039346656037
    for byte in body:
        digest = ((digest ^ byte) * 1099511628211) & ((1 << 64) - 1)
    path.write_bytes(struct.pack('<8sQIIIIIIQQQQ', b'CHMTCP2\0', 5, 0x01020304, 8, 2, 2, 8, 8, 12202687703557820755, len(model), len(state), digest) + body)


def synthetic_case(tmp_path, intervals=2):
    case = tmp_path / 'synthetic'
    out = case / 'chmtOutput'
    out.mkdir(parents=True)
    (case / 'verification.json').write_text(json.dumps({'scenario': 'thermal', 'end_time': .001, 'coupling_interval': .001/intervals, 'gas_max_dt': 2e-5/intervals, 'gas_cells': 8, 'material_cells': 8, 'interface_faces': 4, 'species': ['S0', 'S1'], 'initial_gas_temperature': 600., 'initial_solid_temperature': 300., 'initial_gas_mass': 1., 'initial_material_mass': 1000., 'initial_gas_energy': (1040 - 8.31446261815324/.028)*600, 'initial_material_energy': 3e8, 'gas_thermo': {name: dict(model='linearCp', molar_mass=.028, min_temperature=100, max_temperature=3000, coefficients=[1040, 0, 0]) for name in ('S0','S1')}}))
    gas_energy = (1040 - 8.31446261815324/.028)*600
    for n in range(intervals+1):
        time = n * .001/intervals
        write_csv(out / f'accepted-{n}.csv', [dict(time=time, accepted_steps=n, commit_sequence=n, gas_cells=8, material_cells=8, interface_faces=4, gas_mode=1, thermo_hash=123)])
        write_csv(out / f'gas-{n}.csv', [dict(cell=i, volume=.125, mass=.125, momentum_x=0., momentum_y=0., momentum_z=0., total_energy=(gas_energy-.6*n/intervals)/8, temperature=600-.6*n/intervals/(gas_energy/600), pressure=8.31446261815324/.028*(600-.6*n/intervals/(gas_energy/600)), mass_S0=.125, mass_S1=0.) for i in range(8)])
        write_csv(out / f'material-{n}.csv', [dict(cell=i, volume=.125, total_energy=(3e8+.6*n/intervals)/8, temperature=300+.6*n/intervals/1e6, porosity=0., mass_C0=125., mass_C1=0., mass_pore_S0=0., mass_pore_S1=0.) for i in range(8)])
        rows = [dict(packet=i, step=n-1, geometry=0, face=i, stage=0, kind=0, gas_cell=i, solid_cell=i, mass=0., energy=-.15/intervals, conductive=-.15/intervals, advective=0., pressure_work=0., viscous_work=0., radiation=0., liquid_kinetic_advection=0., consumer_mask=3) for i in range(4)] if n else []
        columns = list(rows[0]) if rows else ['packet', 'step', 'geometry', 'face', 'stage', 'kind', 'gas_cell', 'solid_cell', 'mass', 'energy', 'conductive', 'advective', 'pressure_work', 'viscous_work', 'radiation', 'liquid_kinetic_advection', 'consumer_mask']
        write_csv(out / f'exchange-{n}.csv', rows, columns)
        checkpoint(out / f'checkpoint-{n}/state.chmt', time, n)
    return case


def mutate_csv(path, column, value):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    rows[0][column] = value
    write_csv(path, rows)


def test_accepts_balanced_nonzero_exchange(tmp_path):
    checker = load_checker()
    report = checker.check_case(synthetic_case(tmp_path))
    assert report['final_time'] == .001
    assert report['gas_energy_change'] == pytest.approx(-.6, abs=1e-8)
    assert report['material_energy_change'] == pytest.approx(.6, abs=1e-6)
    assert report['evidence'] == 'independent-output-checks-only'
    assert report['maximum_total_energy_residual'] > 0  # retain sub-ULP total-inventory cancellation


@pytest.mark.parametrize('file,column,value,reason', [
    ('gas-2.csv', 'total_energy', '90000000', 'energy'),
    ('gas-2.csv', 'mass', '.126', 'mass'),
    ('gas-2.csv', 'mass_S0', '.1', 'species'),
    ('gas-2.csv', 'temperature', 'nan', 'finite'),
    ('gas-2.csv', 'cell', '1', 'cell'),
    ('material-2.csv', 'temperature', '299', 'temperature'),
    ('accepted-2.csv', 'commit_sequence', '1', 'synchronized'),
    ('exchange-2.csv', 'consumer_mask', '1', 'consumer'),
    ('exchange-2.csv', 'energy', '0', 'exchange'),
    ('exchange-2.csv', 'gas_cell', '999', 'cell'),
])
def test_rejects_corrupt_evidence(tmp_path, file, column, value, reason):
    checker = load_checker()
    case = synthetic_case(tmp_path)
    mutate_csv(case/'chmtOutput'/file, column, value)
    with pytest.raises(ValueError, match=reason):
        checker.check_case(case)


def test_rejects_no_exchange_and_missing_checkpoint(tmp_path):
    checker = load_checker()
    case = synthetic_case(tmp_path)
    (case/'chmtOutput/checkpoint-2/state.chmt').unlink()
    with pytest.raises((ValueError, OSError)):
        checker.check_case(case)


def test_rejects_checkpoint_checksum_and_clock(tmp_path):
    checker = load_checker()
    case = synthetic_case(tmp_path)
    path = case/'chmtOutput/checkpoint-2/state.chmt'
    checkpoint_path = path.read_bytes()
    path.write_bytes(checkpoint_path[:-1] + b'\x01')
    with pytest.raises(ValueError, match='checksum'):
        checker.check_case(case)
    path.unlink(); path.parent.rmdir()
    checkpoint(path, .00075, 2)
    with pytest.raises(ValueError, match='checkpoint.*time'):
        checker.check_case(case)


def test_generator_creates_physical_case_and_refuses_overwrite(tmp_path):
    script = APP/'verification/coupled_runtime/make_case.py'
    case = tmp_path/'case'
    run = subprocess.run([sys.executable, str(script), str(case), '--diffusion'], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    manifest = json.loads((case/'verification.json').read_text())
    assert manifest['scenario'] == 'diffusion'
    assert 'model constant' in (case/'constant/gasModelProperties').read_text()
    assert 'nonuniform List<scalar>' in (case/'0/Y_S0').read_text()
    again = subprocess.run([sys.executable, str(script), str(case)], capture_output=True, text=True)
    assert again.returncode != 0


def test_native_launcher_refuses_missing_cuda(tmp_path):
    script = APP/'devtools/check_native_coupled.sh'
    import os
    environment = dict(os.environ, NVCC='chmt-deliberately-missing-nvcc', WM_PROJECT_VERSION='10', CHMT_CUDA_ARCH='sm_80')
    run = subprocess.run(['bash', str(script), str(tmp_path/'native-output')], env=environment, capture_output=True, text=True)
    assert run.returncode == 2
    assert 'BLOCKED' in run.stderr and 'nvcc' in run.stderr
    assert not (tmp_path/'native-output').exists()


def test_rejects_finite_but_wrong_gas_temperature(tmp_path):
    checker = load_checker()
    case = synthetic_case(tmp_path)
    mutate_csv(case/'chmtOutput/gas-2.csv', 'temperature', '550')
    with pytest.raises(ValueError, match='gas.*temperature'):
        checker.check_case(case)


def test_restart_comparison_detects_changed_inventory(tmp_path):
    import shutil
    checker = load_checker()
    case = synthetic_case(tmp_path)
    resumed = tmp_path/'resumed'
    shutil.copytree(case, resumed)
    assert checker.compare_restart(case, resumed, case/'chmtOutput/checkpoint-0')
    mutate_csv(resumed/'chmtOutput/gas-2.csv', 'momentum_x', '1')
    with pytest.raises(ValueError, match='restart final gas momentum'):
        checker.compare_restart(case, resumed, case/'chmtOutput/checkpoint-0')


def test_diffusion_control_requires_additional_mixing(tmp_path):
    import shutil
    checker = load_checker()
    case = synthetic_case(tmp_path)
    control = tmp_path/'control'
    shutil.copytree(case, control)
    with pytest.raises(ValueError, match='diffusion.*no resolved'):
        checker.compare_diffusion(case, control)
    with (control/'chmtOutput/gas-2.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    for i, row in enumerate(rows):
        row['mass_S0'] = float(row['mass'])*(.8 if i%2 else .2)
        row['mass_S1'] = float(row['mass'])-float(row['mass_S0'])
    write_csv(control/'chmtOutput/gas-2.csv', rows)
    report = checker.compare_diffusion(case, control)
    assert report['diffusive_variance'] < report['zero_diffusion_variance']


def test_chemistry_generator_starts_reactive_not_inert(tmp_path):
    case = tmp_path/'reactive'
    run = subprocess.run([sys.executable, str(APP/'verification/coupled_runtime/make_case.py'), str(case), '--chemistry'], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    spec = json.loads((case/'verification.json').read_text())
    assert len(spec['species']) == 10
    assert spec['initial_gas_temperature'] == 1200
    assert spec['initial_gas_mass'] == pytest.approx(.21236802713789663)
    assert 'internalField uniform 0.028522' in (case/'0/Y_H2').read_text()
    assert 'internalField uniform 0.226354' in (case/'0/Y_O2').read_text()
    assert spec['element_moles_per_kg']['H2'][1] > 0
    assert spec['gas_thermo']['H2']['model'] == 'NASA7'


def test_independent_clock_reads_production_checkpoint(tmp_path):
    checker = load_checker()
    source = tmp_path/'write.cpp'
    source.write_text(r'''
#define main original_geometry_test_main
#include "devtools/multirate/test_sweep_constraints.cpp"
#undef main
#include "tests/TestSupport.H"
#include "restart/Checkpoint.H"
#include <iostream>
int main(int argc,char** argv){
    if(argc!=2)return 2;
    chmt::ModelConfig model;model.physics=chmt_test::physics();model.physics.enableGas=false;model.physics.modelFingerprint=17;
    chmt::HostState h;h.solidMesh=twoHexahedra(0);auto& m=h.solidMesh;
    m.oldPoints=m.referencePoints=m.points;m.oldVolumes=m.volumes;
    m.boundaryPrimitive.resize(m.owner.size());m.boundarySst.resize(m.owner.size());
    h.solid.resize(m.volumes.size());
    for(std::size_t c=0;c<h.solid.size();++c){h.solid[c].condensed[0]=1000*m.volumes[c];h.solid[c].energy=h.solid[c].condensed[0]*chmt::condensedE(model.physics.condensed[0],300);}
    h.time=.125;h.acceptedSteps=h.commitSequence=1;h.rejectedSteps=7;h.nextDt=.001;h.lastAcceptedDt=.125;
    for(auto& g:h.solidStages){g.interval=.125;g.geometryVersion=m.geometryVersion;g.topologyHash=m.topologyHash;
        g.oldVolume=g.newVolume=g.evaluationVolume=m.volumes;g.sweptVolume.assign(m.owner.size(),0);g.areaVector=m.areaVectors;
        g.cellCentre=m.cellCentres;g.faceCentre=m.faceCentres;g.oldPoints=g.newPoints=m.points;}
    std::string error;if(!chmt::writeCheckpoint(argv[1],model,h,error)){std::cerr<<error;return 1;}
}
''')
    binary = tmp_path/'write'
    build = subprocess.run(['g++', '-std=c++17', '-O1', '-I'+str(APP), str(source), str(APP/'restart/Checkpoint.C'), str(APP/'mesh/Geometry.C'), '-o', str(binary)], capture_output=True, text=True)
    assert build.returncode == 0, build.stderr
    directory = tmp_path/'checkpoint'
    run = subprocess.run([str(binary), str(directory)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    assert checker.checkpoint_clock(directory/'state.chmt', 2) == dict(time=.125, accepted_steps=1, commit_sequence=1, rejected_steps=7)


def test_rejects_balanced_but_wrong_contact_conductance(tmp_path):
    checker = load_checker()
    case = synthetic_case(tmp_path)
    for n in (1,2):
        for region in ('gas', 'material'):
            path = case/f'chmtOutput/{region}-{n}.csv'
            with path.open() as stream:
                rows = list(csv.DictReader(stream))
            with (case/f'chmtOutput/{region}-0.csv').open() as stream:
                base = list(csv.DictReader(stream))
            for row, original in zip(rows, base):
                for key in ('total_energy', 'temperature') + (('pressure',) if region == 'gas' else ()):
                    row[key] = float(original[key])+.1*(float(row[key])-float(original[key]))
            write_csv(path, rows)
        path=case/f'chmtOutput/exchange-{n}.csv'
        with path.open() as stream:
            rows=list(csv.DictReader(stream))
        for row in rows:
            row['conductive']=row['energy']=float(row['energy'])*.1
        write_csv(path,rows)
    with pytest.raises(ValueError, match='contact conductance'):
        checker.check_case(case)


def test_rejects_missing_initial_or_orphan_trial_output(tmp_path):
    checker = load_checker()
    case = synthetic_case(tmp_path)
    orphan = case/'chmtOutput/gas-99.csv'
    orphan.write_text('cell,mass\n0,1\n')
    with pytest.raises(ValueError, match='orphan'):
        checker.check_case(case)
    orphan.unlink()
    for path in (case/'chmtOutput').glob('*-0.csv'):
        path.unlink()
    import shutil
    shutil.rmtree(case/'chmtOutput/checkpoint-0')
    with pytest.raises(ValueError, match='initial synchronized'):
        checker.check_case(case)


def test_refinement_and_frozen_chemistry_generators(tmp_path):
    script=APP/'verification/coupled_runtime/make_case.py'
    refined=tmp_path/'refined'
    run=subprocess.run([sys.executable,str(script),str(refined),'--refine'],capture_output=True,text=True)
    assert run.returncode == 0, run.stderr
    spec=json.loads((refined/'verification.json').read_text())
    assert spec['end_time'] == .001 and spec['coupling_interval'] == .00025
    assert spec['gas_max_dt'] == 5e-6
    frozen=tmp_path/'frozen'
    run=subprocess.run([sys.executable,str(script),str(frozen),'--chemistry-control'],capture_output=True,text=True)
    assert run.returncode == 0, run.stderr
    spec=json.loads((frozen/'verification.json').read_text())
    assert spec['scenario'] == 'chemistry_control' and len(spec['species']) == 10
    gas=(frozen/'constant/gasModelProperties').read_text()
    assert 'gasMode mixtureFrozen;' in gas and 'mechanism "' not in gas and 'phase ohmech' not in gas


def test_refinement_rejects_changed_heat_transfer(tmp_path):
    import shutil
    checker=load_checker()
    coarse=synthetic_case(tmp_path);fine=tmp_path/'fine'
    shutil.copytree(coarse,fine)
    spec=json.loads((fine/'verification.json').read_text());spec.update(coupling_interval=.00025,gas_max_dt=5e-6)
    (fine/'verification.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError, match='refinement'):
        checker.compare_refinement(coarse,fine)


def test_chemical_control_rejects_inert_copy(tmp_path):
    checker=load_checker()
    case=synthetic_case(tmp_path)
    with pytest.raises(ValueError, match='chemical source'):
        checker.compare_chemistry(case,case)


def test_refinement_accepts_identical_resolved_heat(tmp_path):
    checker=load_checker()
    coarse=synthetic_case(tmp_path/'coarse')
    fine=synthetic_case(tmp_path/'fine',intervals=4)
    checker.check_case(coarse)
    checker.check_case(fine)
    report=checker.compare_refinement(coarse,fine)
    assert report['heat_difference']==0
    assert 'no convergence-order claim' in report['expectation']


def test_chemical_control_detects_nonzero_reaction_products(tmp_path):
    checker=load_checker()
    reacting=synthetic_case(tmp_path/'reacting')
    frozen=synthetic_case(tmp_path/'frozen')
    for case in (reacting,frozen):
        for n in range(3):
            path=case/f'chmtOutput/gas-{n}.csv'
            with path.open() as stream:
                rows=list(csv.DictReader(stream))
            extent=1e-6*n if case==reacting else 0
            for row in rows:
                row.pop('mass_S0');row.pop('mass_S1')
                row.update(mass_H2=.01-extent,mass_O2=.04-8*extent,mass_H2O=.075+9*extent)
            write_csv(path,rows)
    report=checker.compare_chemistry(reacting,frozen)
    assert report['maximum_species_response']>1e-5
    assert report['disabled_chemistry_species_drift']==0


def test_accepts_native_zero_pore_bookkeeping_packets(tmp_path):
    checker=load_checker()
    case=synthetic_case(tmp_path)
    for n in (1,2):
        path=case/f'chmtOutput/exchange-{n}.csv'
        with path.open() as stream:
            rows=list(csv.DictReader(stream))
        pore=dict(rows[0]);pore.update(packet=4,kind=3,energy=0,conductive=0)
        rows.append(pore);write_csv(path,rows)
    checker.check_case(case)


def test_single_generator_keeps_real_metadata_and_omits_species_fields(tmp_path):
    case=tmp_path/'single'
    run=subprocess.run([sys.executable,str(APP/'verification/coupled_runtime/make_case.py'),str(case),'--single'],capture_output=True,text=True)
    assert run.returncode==0,run.stderr
    spec=json.loads((case/'verification.json').read_text())
    assert spec['scenario']=='single' and spec['species']==['S0','S1']
    assert 'gasMode single;' in (case/'constant/gasModelProperties').read_text()
    assert 'species (S0 S1);' in (case/'constant/materialGasProperties').read_text()
    assert 'singleGasSpecies S0;' in (case/'constant/chmtProperties').read_text()
    assert not (case/'0/Y_S0').exists() and not (case/'0/Y_S1').exists()


def test_single_output_requires_mode_zero(tmp_path):
    checker=load_checker();case=synthetic_case(tmp_path)
    spec=json.loads((case/'verification.json').read_text());spec['scenario']='single'
    (case/'verification.json').write_text(json.dumps(spec))
    for n in range(3):
        mutate_csv(case/f'chmtOutput/accepted-{n}.csv','gas_mode','0')
    checker.check_case(case)
