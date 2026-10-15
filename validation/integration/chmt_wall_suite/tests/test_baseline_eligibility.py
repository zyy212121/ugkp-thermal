"""A wall-treatment label and a completed run do not prove resolved lowRe."""
import json
import pytest
from test_suite import load
from test_native_completion_contract import final_fields


def pair(tmp_path,yplus=None,time=.2,stage=.2):
    paths=[]
    for family in ('lowRe','wallFunction'):
        case=tmp_path/family;spec=load('generate').prepare('flatplate_fixed',case,{'nx':4,'ny':4,'wall':family})
        final_fields(case,spec)
        mesh=case/'constant/polyMesh';mesh.mkdir()
        for name in ('points','faces','owner','neighbour'):(mesh/name).write_text('identical reader fixture')
        (case/'run_status.json').write_text(json.dumps({'native_execution':'COMPLETED'}))
        if yplus is not None:
            directory=load('metrics').final_directory(case,spec['end_time']) if time==spec['end_time'] else case/str(time)
            output=directory/'uniform';output.mkdir(parents=True,exist_ok=True)
            (output/'gasBoundaryLayer.csv').write_text('face,traceDensity,viscosity,tractionX,tractionY,tractionZ,ownerDistance,stageTime\n'+f'0,1,1,{yplus*yplus},0,0,1,{stage}\n')
        paths.append(case)
    return paths


@pytest.mark.parametrize('measured,status',[(51.5,'INELIGIBLE'),(.5,'ELIGIBLE_BY_MEASURED_YPLUS'),(None,'UNAVAILABLE')])
def test_pair_requires_actual_lowre_measurement_before_reference_eligibility(tmp_path,measured,status):
    report=load('compare_pair').compare_pair(*pair(tmp_path,measured))
    assert report['reference_eligibility']['status']==status
    assert report['reference_eligibility']['target']['maximum_yplus']==1.
    assert report['acceptance']=='REPORT_ONLY'
    assert report['reference_eligibility']['physical_accuracy_acceptance']=='NOT_ESTABLISHED'


@pytest.mark.parametrize('time,stage',[(.1,.1),(.3,.2),(.2,.01),(.2,.3)])
def test_stale_future_or_outside_final_window_yplus_is_unavailable(tmp_path,time,stage):
    report=load('compare_pair').compare_pair(*pair(tmp_path,.5,time,stage))
    assert report['reference_eligibility']['status']=='UNAVAILABLE'


def test_missing_yplus_target_is_not_inferred_from_a_run_label(tmp_path):
    low,high=pair(tmp_path,.5)
    spec=json.loads((low/'suite_case.json').read_text());spec.pop('resolved_lowre_target',None)
    (low/'suite_case.json').write_text(json.dumps(spec))
    report=load('compare_pair').compare_pair(low,high)
    assert report['reference_eligibility']['status']=='UNAVAILABLE'


def test_ineligible_target_cannot_be_loosened_in_saved_manifest(tmp_path):
    low,high=pair(tmp_path,51.5)
    spec=json.loads((low/'suite_case.json').read_text())
    spec['resolved_lowre_target']={'maximum_yplus':100.,'final_window_fraction':.1}
    (low/'suite_case.json').write_text(json.dumps(spec))
    with pytest.raises(ValueError,match='target'):load('compare_pair').compare_pair(low,high)
