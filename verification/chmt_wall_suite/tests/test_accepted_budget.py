"""Synthetic ledger receipts exercise reader guards, not native physics."""
import json
import pytest
from test_suite import load


def segment(path,start,initial,end,final,flux):
    header={'schema':1,'record':'NEW_SEGMENT','continuity':'UNVERIFIED_NEW_SEGMENT',
        'accepted_initial_time':start,'initial_inventory':initial,'species':['A','B'],'elements':['X'],
        'species_order_hash':'11','thermo_hash':'22','mechanism_hash':'0','element_mass_factors_mol_per_kg':[2.,2.]}
    row={'schema':1,'record':'ACCEPTED_INTERVAL','sequence':1,'interval_start':start,'accepted_time':end,
        'inventory':final,'cumulative_boundary_transport':flux,'cumulative_chemistry_species':[0.,0.]}
    path.write_text(json.dumps(header)+'\n'+json.dumps(row)+'\n');return header,row


def test_reader_recomputes_actual_accepted_flux_balances(tmp_path):
    path=tmp_path/'one.jsonl';segment(path,0.,[2.,10.,.8,1.2],.1,[2.1,12.,.84,1.26],[.1,2.,.04,.06])
    report=load('budget').read_segments([path],expected_end=.1)
    assert report['status']=='AVAILABLE'
    assert report['continuity']=='SINGLE_SEGMENT_FROM_ZERO'
    assert max(map(abs,report['constraint_or_roundoff_residual']))<1e-14
    assert max(map(abs,report['element_conservation_residual_mol']))<1e-14
    assert report['mass_elements_physical_energy_acceptance']=='NOT_ESTABLISHED'


def test_matching_restart_segments_are_reconciled_without_resetting_totals(tmp_path):
    first=tmp_path/'first.jsonl';second=tmp_path/'second.jsonl'
    segment(first,0.,[2.,10.,.8,1.2],.1,[2.1,12.,.84,1.26],[.1,2.,.04,.06])
    segment(second,.1,[2.1,12.,.84,1.26],.2,[2.2,14.,.88,1.32],[.1,2.,.04,.06])
    report=load('budget').read_segments([first,second],expected_end=.2)
    assert report['continuity']=='VERIFIED_SEGMENT_CHAIN'
    assert report['cumulative_boundary_transport']==pytest.approx([.2,4.,.08,.12])


@pytest.mark.parametrize('fault',['time','inventory','identity','species_order','missing_interval','final_time'])
def test_invalid_restart_or_incomplete_chain_rejects(tmp_path,fault):
    first=tmp_path/'first.jsonl';second=tmp_path/'second.jsonl'
    segment(first,0.,[2.,10.,.8,1.2],.1,[2.1,12.,.84,1.26],[.1,2.,.04,.06])
    h,r=segment(second,.1,[2.1,12.,.84,1.26],.2,[2.2,14.,.88,1.32],[.1,2.,.04,.06])
    if fault=='time':h['accepted_initial_time']=.09;r['interval_start']=.09
    if fault=='inventory':h['initial_inventory'][1]=99.
    if fault=='identity':h['thermo_hash']='99'
    if fault=='species_order':h['species'].reverse()
    if fault=='missing_interval':r['sequence']=2
    second.write_text(json.dumps(h)+'\n'+json.dumps(r)+'\n')
    if fault in ('time','inventory'):
        report=load('budget').read_segments([first,second],expected_end=.2)
        assert report['continuity']=='UNVERIFIED_SEGMENT_CHAIN'
        assert report['constraint_or_roundoff_residual'] is None
    else:
        with pytest.raises(ValueError):load('budget').read_segments([first,second],expected_end=.3 if fault=='final_time' else .2)


def test_chemistry_element_drift_is_not_subtracted_from_conservation(tmp_path):
    path=tmp_path/'bad.jsonl';h,r=segment(path,0.,[2.,10.,.8,1.2],.1,[2.,10.,.9,1.2],[0.,0.,0.,0.])
    r['cumulative_chemistry_species']=[.1,0.];path.write_text(json.dumps(h)+'\n'+json.dumps(r)+'\n')
    report=load('budget').read_segments([path],expected_end=.1)
    assert report['constraint_or_roundoff_residual'][2]==pytest.approx(0,abs=1e-15)
    assert report['element_conservation_residual_mol']==pytest.approx([.2])
    assert report['chemistry_element_drift_mol']==pytest.approx([.2])


def test_restart_segment_alone_never_claims_history_from_zero(tmp_path):
    path=tmp_path/'restart.jsonl';segment(path,.5,[2.,10.,.8,1.2],.6,[2.,10.,.8,1.2],[0.,0.,0.,0.])
    report=load('budget').read_segments([path],expected_end=.6)
    assert report['continuity']=='RESTART_SEGMENT_ONLY'


def test_opt_in_manifest_requires_a_real_final_budget_file(tmp_path):
    case=tmp_path/'case';spec=load('generate').prepare('gas_species_wave',case,{'budget_audit':True})
    assert spec['budget_audit_requested'] is True
    with pytest.raises(ValueError,match='budget'):
        load('metrics').accepted_cumulative_budget_evidence(case,spec)
    segment(case/'gasAcceptedBudget.jsonl',0.,[2.,10.,.8,1.2],spec['end_time'],[2.,10.,.8,1.2],[0.,0.,0.,0.])
    assert load('metrics').accepted_cumulative_budget_evidence(case,spec)['status']=='AVAILABLE'


@pytest.mark.parametrize('name',['small_couette','small_sod','chmt_receding_slab'])
def test_opt_in_does_not_silently_claim_unsupported_application_coverage(tmp_path,name):
    with pytest.raises(ValueError,match='budget'):
        load('generate').prepare(name,tmp_path/name,{'budget_audit':True})


def test_reader_preserves_serialized_reduction_precision(tmp_path):
    path=tmp_path/'precision.jsonl'
    segment(path,0,[10000000000000001,10,.8,1.2],.1,[10000000000000002,10,.8,1.2],[1,0,0,0])
    report=load('budget').read_segments([path])
    assert report['constraint_or_roundoff_residual'][0]==0


@pytest.mark.parametrize('fault',['species_object','element_object','negative_initial','negative_species'])
def test_malformed_segment_header_fails_cleanly(tmp_path,fault):
    path=tmp_path/'bad-header.jsonl';h,r=segment(path,0,[2,10,.8,1.2],.1,[2,10,.8,1.2],[0,0,0,0])
    if fault=='species_object':h['species']=[{},'B']
    if fault=='element_object':h['elements']=[{}]
    if fault=='negative_initial':h['initial_inventory'][0]=-2
    if fault=='negative_species':h['initial_inventory'][2]=-.8
    path.write_text(json.dumps(h)+'\n'+json.dumps(r)+'\n')
    with pytest.raises(ValueError):load('budget').read_segments([path])


@pytest.mark.parametrize('fault',['restart_inventory','restart_time','interval_time'])
def test_large_origin_cannot_hide_restart_or_interval_gap(tmp_path,fault):
    first=tmp_path/'first.jsonl';second=tmp_path/'second.jsonl'
    segment(first,0,[10000000000000000,10,.8,1.2],1000000.,[10000000000000000,10,.8,1.2],[0,0,0,0])
    h,r=segment(second,1000000.,[10000000000000000,10,.8,1.2],1000001.,[10000000000000000,10,.8,1.2],[0,0,0,0])
    if fault=='restart_inventory':h['initial_inventory'][0]+=100
    if fault=='restart_time':h['accepted_initial_time']+=1e-8;r['interval_start']=h['accepted_initial_time']
    if fault=='interval_time':r['interval_start']+=1e-8
    second.write_text(json.dumps(h)+'\n'+json.dumps(r)+'\n')
    if fault=='interval_time':
        with pytest.raises(ValueError):load('budget').read_segments([first,second])
    else:
        report=load('budget').read_segments([first,second])
        assert report['continuity']=='UNVERIFIED_SEGMENT_CHAIN'
        assert report['complete_from_time_zero'] is False
        assert report['constraint_or_roundoff_residual'] is None
        assert report['segments'][1]['raw_records']==second.read_text().splitlines()
        if fault=='restart_inventory':
            assert report['restart_discontinuities'][0]['inventory_jump'][0]==100
            assert report['segments'][1]['constraint_or_roundoff_residual'][0]==-100
        else:assert report['restart_discontinuities'][0]['time_jump']>0
