"""Reject requested controls that a canonical/pinned case would silently ignore."""
import math
import pytest
from test_suite import load


@pytest.mark.parametrize('name',['small_couette','small_sod','small_cht_contact','gas_chemistry_reactor'])
def test_fixed_reference_endpoint_override_is_not_silently_ignored(tmp_path,name):
    with pytest.raises(ValueError,match='endpoint'):
        load('generate').prepare(name,tmp_path/name,{'end_time':123.})


@pytest.mark.parametrize('end',[0.,-1.,math.nan,math.inf,-math.inf,None])
@pytest.mark.parametrize('name',['gas_species_wave','chmt_receding_slab','flatplate_fixed'])
def test_invalid_requested_endpoint_rejected_before_generation(tmp_path,name,end):
    with pytest.raises(ValueError,match='end_time'):
        load('generate').prepare(name,tmp_path/name,{'end_time':end})
    assert not (tmp_path/name/'suite_case.json').exists()


def test_same_canonical_endpoint_and_small_nonround_supported_endpoint(tmp_path):
    fixed=load('generate').prepare('small_couette',tmp_path/'fixed',{'end_time':.05})
    assert fixed['end_time']==.05
    free=load('generate').prepare('gas_species_wave',tmp_path/'free',{'end_time':.037})
    assert free['end_time']==.037 and float(free['time_controls']['endTime'])==.037
    assert free['time_controls']['writeControl']=='adjustableRunTime'
