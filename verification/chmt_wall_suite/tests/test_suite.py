import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import pytest
ROOT=Path(__file__).resolve().parents[3]
SUITE=ROOT/'verification/chmt_wall_suite'
sys.path.insert(0,str(SUITE))

def load(name):
    spec=importlib.util.spec_from_file_location('suite_'+name,SUITE/(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def test_catalog_has_three_classes_and_supported_pairing():
    c=load('catalog').cases()
    assert {x['category'] for x in c.values()}=={1,2,3}
    assert all(x['gate']=='report_only' for x in c.values() if x['category'] in (2,3))
    assert c['small_couette']['gate']=='hard'
    assert set(c['flatplate_fixed']['wall_variants'])=={'lowRe','wallFunction','boundaryLayer_reactingSst'}
    assert c['wall_constant_transport']['models']['turbulence']=='laminar'

def test_fresh_generation_refuses_existing_target(tmp_path):
    g=load('generate');p=tmp_path/'run';p.mkdir();(p/'keep').write_text('mine')
    with pytest.raises(FileExistsError):g.prepare('gas_species_wave',p,{})
    assert (p/'keep').read_text()=='mine'

def test_wave_case_generates_production_input_and_complete_manifest(tmp_path):
    g=load('generate');p=tmp_path/'run';out=g.prepare('gas_species_wave',p,{'cells':32})
    assert 'gasMode mixtureFrozen;' in (p/'constant/gasModelProperties').read_text()
    assert out['native_execution']=='NOT_RUN'
    assert out['solver']=='gasUGKP'
    assert out['inputs_sha256']
    assert not list(p.rglob('README*'))

def test_chmt_moving_has_reaction_motion_and_complete_material(tmp_path):
    g=load('generate');p=tmp_path/'run';m=g.prepare('chmt_receding_slab',p,{'cells':8})
    s=(p/'constant/chmtProperties').read_text()
    assert 'CoupledRecession' in s and 'surfaceReactions' in s
    assert 'materialSource syntheticControlledV1' in s
    assert 'material_card' in m and m['material_card']['classification']=='controlled_synthetic'
    assert (p/'0/solid/rhoPore_S0').exists()
    assert m['gate']=='report_only'

def test_wave_oracle_is_exact_cell_average():
    x=load('metrics').wave_average(0,.25,0,length=1,velocity=0,diffusivity=0)
    assert x==pytest.approx(.5+.4/math.pi,abs=1e-15)

def test_couette_oracle_is_transient_not_incorrect_steady_limit():
    f=load('metrics').couette_average
    assert f(.45,.55,0,1,1,.1)==pytest.approx(0,abs=1e-10)
    assert f(.45,.55,100,1,1,.1)==pytest.approx(.5,abs=1e-10)
    assert 0<f(.85,.95,.05,1,1,.1)<.9

def test_nonfinite_scalar_and_lfs_are_rejected(tmp_path):
    f=load('metrics').read_field;p=tmp_path/'T'
    p.write_text('internalField uniform nan;')
    with pytest.raises(ValueError):f(p,2)
    p.write_text('version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 3\n')
    with pytest.raises(ValueError):f(p,2)

def test_empty_native_log_cannot_pass(tmp_path):
    r=load('run');p=tmp_path/'case';load('generate').prepare('gas_species_wave',p,{'cells':8})
    (p/'log.solver').write_text('')
    with pytest.raises(ValueError):r.validate_completion(p,json.loads((p/'suite_case.json').read_text()))

def test_mesh_missing_remains_not_run(tmp_path,monkeypatch):
    r=load('run');monkeypatch.setattr(r.shutil,'which',lambda _:None)
    p=tmp_path/'run';load('generate').prepare('gas_species_wave',p,{'cells':8})
    s=r.execute(p,mode='mesh')
    assert s['native_execution']=='NOT_RUN' and s['mesh']=='NOT_RUN'
    assert s['validation']=='NOT_RUN'

def test_external_sources_nc_not_vendored_and_hashes_available():
    sources=json.loads((SUITE/'references/sources.json').read_text())
    assert any(s.get('license')=='CC-BY-NC-4.0' and s.get('redistribute') is False for s in sources)
    assert all(len(s['sha256'])==64 for s in sources if s.get('sha256'))

def test_metric_gate_does_not_turn_report_only_objective_into_failure():
    r=load('metrics')
    assert r.apply_gate({'x':9},{'x':1},'report_only')['acceptance']=='REPORT_ONLY'
    assert r.apply_gate({'x':9},{'x':1},'hard')['acceptance']=='FAIL'
    with pytest.raises(ValueError):r.apply_gate({'x':float('nan')},{'x':1},'report_only')

@pytest.mark.parametrize('name',['small_cht_contact','gas_chemistry_reactor'])
def test_existing_production_generator_contracts(tmp_path,name):
    out=load('generate').prepare(name,tmp_path/name,{})
    assert out['end_time']>0 and out['cells']>0
    assert (tmp_path/name/'suite_case.json').exists()

@pytest.mark.parametrize('name',['flatplate_fixed','flatplate_moving','chmt_receding_slab'])
def test_native_fields_have_seven_dimensions_and_reactions_have_elements(tmp_path,name):
    import re
    out=load('generate').prepare(name,tmp_path/name,{'nx':4,'ny':4,'solid_ny':4})
    for p in (tmp_path/name/'0').rglob('*'):
        if p.is_file():
            d=re.search(r'dimensions\s*\[([^]]+)\]',p.read_text())
            assert len(d.group(1).split())==7,str(p)
    if out['solver']=='CHMT' and name!='small_cht_contact':
        assert 'elements (X);' in (tmp_path/name/'constant/chmtProperties').read_text()
        assert 'atoms (1);' in (tmp_path/name/'constant/gasModelProperties').read_text()

def test_reactive_transport_and_chmt_produce_actual_mechanism(tmp_path):
    for name in ('gas_reacting_wave','chmt_reacting_receding_slab'):
        out=load('generate').prepare(name,tmp_path/name,{})
        assert 'gasMode mixtureChemistry;' in (tmp_path/name/'constant/gasModelProperties').read_text()
        assert (tmp_path/name/'constant/synthetic.mechanism').exists()
        assert out['parameters']['reaction_rate_per_s']==2

def test_reacting_wave_exact_zero_gradient_limit():
    f=load('metrics').reacting_wave_average
    assert f(0,.2,.25,length=1,velocity=1,diffusivity=.02,amplitude=0)==pytest.approx(.5*math.exp(-.5))

def test_sod_independent_star_and_no_output_self_pass(tmp_path):
    f=load('metrics').riemann_state
    rho,u,p=f(.5,.2,interface=.5,gamma=1.4,left=[1,0,1],right=[.125,0,.1])
    assert rho==pytest.approx(.4263194282,rel=1e-8)
    assert u==pytest.approx(.9274526200,rel=1e-8)
    assert p==pytest.approx(.3031301781,rel=1e-8)
    case=tmp_path/'wave';load('generate').prepare('gas_species_wave',case,{})
    with pytest.raises(ValueError):load('metrics').compare(case)

def test_cli_required_missing_backend_returns_nonzero(tmp_path):
    c=tmp_path/'case';load('generate').prepare('gas_species_wave',c,{})
    import os
    env=dict(os.environ,PATH='/nonexistent')
    run=subprocess.run([sys.executable,str(SUITE/'run.py'),str(c),'--mode','execute','--required'],env=env,capture_output=True,text=True)
    assert run.returncode==3
    assert json.loads((c/'run_status.json').read_text())['native_execution']=='NOT_RUN'

def test_pair_generation_has_actual_low_high_inputs_and_unsupported_record(tmp_path):
    g=load('suite');summary=g.prepare_suite(tmp_path/'suite',['flatplate_fixed','flatplate_moving'],{})
    ids=[x['id'] for x in summary['cases']]
    assert 'flatplate_fixed__lowRe' in ids and 'flatplate_fixed__wallFunction' in ids
    assert 'flatplate_moving__boundaryLayer_reactingSst' in ids
    assert any(x['case']=='flatplate_moving' and x['family']=='wallFunction' and x['status']=='UNSUPPORTED' for x in summary['unsupported'])
    for item in summary['cases']:assert (Path(item['path'])/'suite_case.json').exists()

def test_pair_comparison_refuses_preparation_only(tmp_path):
    p=load('compare_pair');g=load('generate')
    a=tmp_path/'a';b=tmp_path/'b';g.prepare('flatplate_fixed',a,{'nx':4,'ny':4});g.prepare('flatplate_fixed',b,{'nx':4,'ny':4,'wall':'wallFunction'})
    with pytest.raises(ValueError):p.compare_pair(a,b)

def test_piecewise_linear_interpolation_is_exact_and_refuses_extrapolation():
    p=load('compare_pair');assert p.linear([0,1,2],[1,3,5],.5)==pytest.approx(2)
    with pytest.raises(ValueError):p.linear([0,1,2],[1,3,5],-1)

def test_gas_wall_controls_are_in_real_fluid_properties(tmp_path):
    g=load('generate')
    for name,opts in [('wall_constant_transport',{}),('flatplate_fixed',{'nx':4,'ny':4,'wall':'boundaryLayer_reactingSst'})]:
        g.prepare(name,tmp_path/name,opts)
        text=(tmp_path/name/'constant/fluidProperties').read_text()
        assert 'wallTreatment boundaryLayer;' in text
        assert 'model '+('constantTransport' if name=='wall_constant_transport' else 'reactingSst')+';' in text
        assert not (tmp_path/name/'constant/momentumTransport').exists()

def test_two_material_demos_differ_in_physical_parameters(tmp_path):
    g=load('generate');a=g.prepare('chmt_receding_slab',tmp_path/'a',{'material':'controlled_v1'});b=g.prepare('chmt_receding_slab',tmp_path/'b',{'material':'controlled_v2'})
    assert a['material_card']['condensed'][0]['rho']!=b['material_card']['condensed'][0]['rho']
    for card in (a['material_card'],b['material_card']):
        r=card['surface_reactions'][0]
        assert sum(r['condensedNu'])+sum(r['gasNu'])==pytest.approx(0)
        assert -r['condensedNu'][0]==pytest.approx(.028)

def test_host_metric_limits_are_fixed_and_nonfinite_is_rejected():
    h=load('host_cases');m=h.discrepancy([1.,2.],[1.,2.001],limit=.01)
    assert m['status']=='PASS' and m['target']==[1.,2.001] and m['linf']>0
    with pytest.raises(ValueError):h.discrepancy(float('nan'),1.)

def test_host_cuda_request_never_uses_cpu_as_native(tmp_path,monkeypatch):
    import run
    monkeypatch.setattr(run,'cuda_probe',lambda:{'status':'UNAVAILABLE','reason':'unit test simulates no device'})
    r=load('host_cases').run(tmp_path/'host',names=['wall_constant_limit'],backend='cuda')
    assert r['counts']['NOT_RUN']==1 and r['counts']['PASS']==0

def test_independent_bvp_is_well_conditioned_and_satisfies_boundaries():
    h=load('host_cases');ref,meta=h.bvp_reference()
    assert meta['max_rms_residual']<=meta['tolerance']
    assert abs(ref.y[1,0])<1e-12 and ref.y[2,0]==pytest.approx(500)
    assert ref.y[0,-1]==pytest.approx(.7) and ref.y[2,-1]==pytest.approx(700)

def test_wall_diagnostic_yplus_uses_actual_tau_rho_mu_and_numeric_time(tmp_path):
    for time,tau in [('2',4.),('10',9.)]:
        d=tmp_path/time/'uniform';d.mkdir(parents=True)
        (d/'gasBoundaryLayer.csv').write_text('face,traceDensity,viscosity,tractionX,tractionY,tractionZ,ownerDistance,stageTime\n0,1,.01,'+str(tau)+',0,0,.1,'+time+'\n')
    r=load('metrics').wall_diagnostics(tmp_path,{})
    assert r['directory_time']==10 and r['yplus']['max']==pytest.approx(30.)

def test_reference_import_rejects_untested_or_modified_data(tmp_path):
    mod=load('import_reference');p=tmp_path/'data';p.write_text('0 1\n')
    with pytest.raises(ValueError,match='SHA256'):mod.import_source('nasa_sst_cf.dat',p,tmp_path/'x.json')
    with pytest.raises(ValueError,match='noncommercial'):mod.import_source('ablantis_exp2_tc.dat',p,tmp_path/'nc.json')

def test_laminar_reactive_wall_has_explicit_finite_rate_mode(tmp_path):
    out=load('generate').prepare('chmt_laminar_reacting_wall',tmp_path/'case',{'wall_nodes':96})
    text=(tmp_path/'case/constant/chmtProperties').read_text()
    assert 'enableSst false;' in text and 'model finiteRate;' in text and 'nodes 96;' in text
    assert out['models']['turbulence']=='laminar' and out['time_controls']['endTime']
    with pytest.raises(ValueError):load('generate').wall_config('boundaryLayer_reactingSst',nodes=129)

def test_reactive_case_cannot_accept_a_frozen_backend_log(tmp_path):
    p=tmp_path/'reactive';out=load('generate').prepare('gas_reacting_wave',p,{})
    (p/format(out['end_time'],'.12g')).mkdir()
    (p/'log.solver').write_text('Shared gas model: api=1 Ns=2 mode=1\nEnd\n')
    with pytest.raises(ValueError,match='identity'):load('run').validate_completion(p,out)

def test_legacy_missing_shared_metadata_is_not_an_unsupported_case(tmp_path,monkeypatch):
    r=load('run');case=tmp_path/'legacy';load('generate').prepare('small_couette',case,{})
    monkeypatch.setattr(r.shutil,'which',lambda _: '/fake/tool')
    def executed(args,cwd,stdout,stderr):
        stdout.write('Mesh OK.\n');return type('Completed',(),{'returncode':0})()
    monkeypatch.setattr(r.subprocess,'run',executed)
    out=r.execute(case,'check-input')
    assert out['input_import']=='NOT_APPLICABLE' and out['native_execution']=='NOT_RUN'
    assert out['outcome']=='NOT_RUN'

def test_contact_fixture_rejects_unimplemented_material_override(tmp_path):
    with pytest.raises(ValueError, match='controlled_v1'):
        load('generate').prepare('small_cht_contact',tmp_path/'contact',{'material':'controlled_v2'})

def test_reactor_missing_evidence_is_not_report_only(tmp_path,monkeypatch):
    import foam
    from types import SimpleNamespace
    (tmp_path/'suite_case.json').write_text(json.dumps(dict(metric='reactor',parameters={},cells=4,end_time=1)))
    (tmp_path/'case_contract.json').write_text(json.dumps(dict(check_times=[0,1])))
    monkeypatch.setattr(foam,'module',lambda _:SimpleNamespace(compare_case=lambda case:dict(passed=False,failures=['missing history file'],checked_times=[])))
    with pytest.raises(ValueError,match='reactor evidence'):
        load('metrics').compare(tmp_path)

def test_flatplate_coordinates_preserve_affine_cartesian_field():
    import numpy as np
    c=load('compare_pair')
    points=np.array([[0.,.001,.005],[1.,.003,.005]])
    actual=c.profile_coordinates(points,'flatplate_fixed')
    assert np.array_equal(actual,points[:,:2])
    radial=c.profile_coordinates(points,'mss7_fixed')
    assert radial[0,1]==pytest.approx(math.hypot(.001,.005))

def test_surface_rate_reference_detects_frozen_noop_and_matches_exact_history():
    f=load('metrics').surface_rate_history
    times=[0.,.01,.02]
    inactive=dict(removed_mass_kg=[0.]*3,volume_loss_m3=[0.]*3,
        gas_S0_mass_kg=[1.]*3,gas_S1_mass_kg=[0.]*3)
    values,samples=f(times,inactive,area=1.,mass_flux=.001,density=1000.,rate=2.)
    assert values['surface_rate_removed_mass_kg_linf']==pytest.approx(1.)
    assert values['surface_rate_volume_loss_m3_linf']==pytest.approx(1.)
    assert values['surface_rate_gas_S1_mass_kg_linf']==pytest.approx(1.)
    exact={key:value['reference'] for key,value in samples.items() if key!='time_s'}
    residuals,_=f(times,exact,area=1.,mass_flux=.001,density=1000.,rate=2.)
    assert max(residuals.values())<1e-15

def test_reacting_slab_declares_independent_constant_rate_reference(tmp_path):
    s=load('generate').prepare('chmt_reacting_receding_slab',tmp_path/'case',{})
    assert s['parameters']['surface_area_m2']==1.
    assert s['parameters']['surface_mass_flux_kg_m2_s']==pytest.approx(.001)
    assert s['parameters']['reaction_rate_per_s']==2.

def test_suite_selects_distinct_species_count_binaries():
    choose=load('suite').select_solver
    assert choose({'solver':'gasUGKP','species_count':2},'chmt','gas2','gas10')=='gas2'
    assert choose({'solver':'gasUGKP','species_count':10},'chmt','gas2','gas10')=='gas10'
    assert choose({'solver':'CHMT','species_count':2},'chmt','gas2','gas10')=='chmt'
    assert choose({'solver':'gasUGKP','species_count':10},'chmt','gas2',None) is None

@pytest.mark.parametrize('rows', [[], [[0,1,0,1,float('nan'),float('nan')]], [[0,1,0,1,2,3]]])
def test_sst_probe_rejects_incomplete_or_nonfinite_rows(tmp_path,monkeypatch,rows):
    h=load('host_cases')
    monkeypatch.setattr(h.runpy,'run_path',lambda path:{'BASE':''})
    monkeypatch.setattr(h,'_build_run',lambda *a,**k:(rows,{}))
    with pytest.raises(ValueError):h.wall_sst_robustness(ROOT,tmp_path)

def _write_chmt_wall(path,time,available=True):
    import csv
    keys='accepted_time stage_time face status species wall_temperature wall_density wall_mass_fraction wall_species_flux matching_species_flux reaction_integral species_balance_residual chemistry_mismatch_available chemistry_mismatch conductive_heat_flux traction_x traction_y traction_z wall_k_flux owner_k_source_integral owner_omega iterations residual'.split()
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for face in (7,9):
            for species in ('A','B'):
                row=dict.fromkeys(keys,'');row.update(accepted_time=time,face=face,status='AVAILABLE' if available else 'NOT_AVAILABLE',species=species)
                if available:
                    row.update({key:1 for key in keys[5:]});row.update(stage_time=time-.1,chemistry_mismatch_available=0,chemistry_mismatch='',conductive_heat_flux=face,wall_mass_fraction=.5)
                writer.writerow(row)

def test_chmt_wall_export_uses_accepted_time_and_unique_faces(tmp_path):
    p=tmp_path/'chmtOutput';_write_chmt_wall(p/'wall-layer-99.csv',2);_write_chmt_wall(p/'wall-layer-100.csv',10)
    r=load('metrics').wall_diagnostics(tmp_path,{'species_count':2})
    assert r['status']=='AVAILABLE' and r['accepted_time']==10 and r['stage_time']==9.9
    assert r['faces']==2 and r['rows']==4 and r['yplus']=='UNAVAILABLE'
    assert r['statistics']['conductive_heat_flux']['mean_face_count_weighted']==8
    assert set(r['species_statistics'])=={'A','B'}

def test_chmt_latest_unavailable_does_not_reuse_old_stage(tmp_path):
    p=tmp_path/'chmtOutput';_write_chmt_wall(p/'wall-layer-99.csv',2);_write_chmt_wall(p/'wall-layer-100.csv',10,False)
    r=load('metrics').wall_diagnostics(tmp_path,{'species_count':2})
    assert r['status']=='UNAVAILABLE' and r['accepted_time']==10

@pytest.mark.parametrize('fault',['nonfinite','duplicate','missing_species','inconsistent_face','bad_stage','missing_column'])
def test_chmt_wall_export_rejects_invalid_evidence(tmp_path,fault):
    import csv
    path=tmp_path/'chmtOutput/wall-layer-1.csv';_write_chmt_wall(path,1)
    with path.open() as f:reader=csv.DictReader(f);keys=reader.fieldnames;rows=list(reader)
    if fault=='nonfinite':rows[0]['residual']='nan'
    elif fault=='duplicate':rows.append(rows[0])
    elif fault=='missing_species':rows.pop()
    elif fault=='inconsistent_face':rows[0]['conductive_heat_flux']='100'
    elif fault=='bad_stage':rows[0]['stage_time']='2'
    else:keys.remove('wall_density');[r.pop('wall_density') for r in rows]
    with path.open('w') as f:writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    with pytest.raises(ValueError):load('metrics').wall_diagnostics(tmp_path,{'species_count':2})

def test_runner_records_process_timing_without_extra_solver_pass(tmp_path,monkeypatch):
    g=load('generate');case=tmp_path/'case';g.prepare('small_couette',case,{})
    r=load('run');calls=[]
    monkeypatch.setattr(r.shutil,'which',lambda name:'/fake/'+name)
    def command(args,**kw):
        calls.append(args)
        kw['stdout'].write('Mesh OK.\n')
        return subprocess.CompletedProcess(args,0)
    monkeypatch.setattr(r.subprocess,'run',command)
    result=r.execute(case,'mesh')
    assert result['mesh']=='PASS'
    assert len(result['command_measurements'])==len(calls)==2
    assert all(x['wall_seconds']>=0 and x['returncode']==0 for x in result['command_measurements'])
    assert result['native_execution']=='NOT_RUN'
