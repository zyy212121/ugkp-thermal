"""Tests exercise real generated inputs/readback; never manufacture CUDA results."""
import csv
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent

def module(name):
    spec=importlib.util.spec_from_file_location(name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def test_materialized_reservoir_and_ascii_roundtrip(tmp_path):
    create=module('of_cases');export=module('export_of');compare=module('compare_application')
    case=create.materialize(HERE/'cases/reservoir.cfg',tmp_path/'case');output=tmp_path/'initial.csv'
    report=export.export(case,output,[0]);rows=list(csv.DictReader(output.open()))
    assert report['rows']==32 and len(rows)==32
    assert abs(sum(float(r['volume']) for r in rows)-2)<1e-14
    assert all(abs(float(r['T'])-100000/287)<1e-10 for r in rows)
    assert 'rhoU' in report['missing_actual_fields']['0']
    assert 'rhoUx' not in rows[0]
    assert compare.comparison(output,output,['rho','rhoE','Ux','p','T'],1e-8,1e-10)['pass']
    assert not compare.comparison(output,output,['rhoUx'],1e-8,1e-10)['pass']

def test_materialized_diffusion_species(tmp_path):
    create=module('of_cases');export=module('export_of')
    case=create.materialize(HERE/'cases/mixture_diffusion.cfg',tmp_path/'case')
    y=export.internal(case/'0/Y_A',32);z=export.internal(case/'0/Y_B',32)
    assert max(r[0] for r in y)-min(r[0] for r in y)>.09
    assert all(abs(a[0]+b[0]-1)<1e-14 for a,b in zip(y,z))
    assert 'coefficients (0.001 0.003)' in (case/'constant/gasModelProperties').read_text()

def test_vector_parser_and_missing_comparison(tmp_path):
    export=module('export_of');compare=module('compare_application')
    f=tmp_path/'U';f.write_text('internalField nonuniform List<vector> 2 ((1 2 3) (4 5 6));')
    assert export.internal(f,2)==[[1,2,3],[4,5,6]]
    a=tmp_path/'a.csv';b=tmp_path/'b.csv';a.write_text('iteration,cell,rho\n0,0,1\n');b.write_text('iteration,cell,rho\n0,0,nan\n')
    assert not compare.comparison(a,b,['rho'],1e-8,1e-10)['pass']
    b.write_text('iteration,cell,rho\n0,0,1\n0,0,1\n')
    assert not compare.comparison(a,b,['rho'],1e-8,1e-10)['pass']

def test_window_csv_format_only(tmp_path):
    # Synthetic parser unit fixture, not simulation evidence.
    export=module('export_of');directory=tmp_path/'1000';directory.mkdir()
    (directory/'steadyWindowMean.csv').write_text('cell,rho,rhoUx\n0,1,20\n1,2,30\n')
    output=tmp_path/'windows.csv';result=export.export_windows(tmp_path,output)
    rows=list(csv.DictReader(output.open()))
    assert result['rows']==2 and rows[0]['q0Mean']=='1' and rows[1]['q1Mean']=='30'
    assert rows[0]['iteration']=='1000'

def test_budget_log_parser_fixture(tmp_path):
    # Synthetic parser unit fixture, not simulation evidence.
    budgets=module('compare_window_budgets');ref=tmp_path/'cpu.csv';log=tmp_path/'app.log'
    ref.write_text('iteration,cell,q0Mean,q0MeanChange,q0Fluctuation,q0InventoryBegin,q0InventoryEnd\n1000,0,1,0.1,0.2,0.5,0.6\n1000,1,2,0.3,0.1,0.7,0.8\n')
    log.write_text('Steady sampling window 1 physicalDuration=1 totalSamplingTime=1\n  rho meanChange=0.3 relativeRms=0.2 inventoryBegin=1.2 inventoryEnd=1.4 inventoryDrift=0.1\n')
    assert budgets.compare(ref,log)['pass']
    log.write_text(log.read_text().replace('inventoryEnd=1.4','inventoryEnd=2.4'))
    assert not budgets.compare(ref,log)['pass']


def test_explicit_openfoam_constant_and_escape_particle_boundaries(tmp_path):
    from of_cases import materialize
    import json,re
    rr=8314.4700665054497
    case=materialize(Path(__file__).parent/'cases/particles.cfg',tmp_path/'matched',True,rr)
    properties=(case/'constant/particleProperties').read_text()
    assert 'particleWallCoeffs { default 1; left 0; right 0; }' in properties
    mass=float(re.search(r'molWeight\s+([^;]+)',(case/'constant/fluidProperties').read_text())[1])
    assert rr/mass==287
    manifest=json.loads((case/'reference-input.json').read_text())
    assert manifest['foam_universal_gas_constant']==rr
    assert manifest['particle_escape_boundaries']==['left','right']
