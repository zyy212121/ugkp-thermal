"""Partial performance samples and instantaneous CSVs cannot become acceptance."""
import json
import math
import subprocess
import sys
import pytest
from test_suite import load,SUITE
from test_review_runner_regressions import _exact_couette_output,_wall_csv


def test_instantaneous_or_missing_wall_csv_does_not_become_cumulative_budget(tmp_path):
    case=tmp_path/'case';_exact_couette_output(case);_wall_csv(case)
    report=load('metrics').compare(case)
    ledger=report['accepted_cumulative_budget']
    assert ledger['status']=='UNAVAILABLE'
    assert ledger['mass_elements_physical_energy_acceptance']=='NOT_ESTABLISHED'
    assert 'instantaneous' in ledger['reason']


def sample(**changes):
    values={'physical_start_s':0.,'physical_end_s':.001,'target_end_s':.2,
            'wall_seconds':2.,'concurrent':True,'instrumented':True}
    values.update(changes);return values


def test_short_concurrent_sample_estimate_is_explicitly_diagnostic():
    result=load('performance').estimate(sample())
    assert result['status']=='DIAGNOSTIC_LINEAR_EXTRAPOLATION'
    assert result['estimated_remaining_wall_seconds']==pytest.approx(398.)
    assert set(result['limitations'])=={'PARTIAL_TRAJECTORY','CONCURRENT_WORK','INSTRUMENTED_PROCESS'}
    assert result['isolated_gpu_throughput']=='NOT_MEASURED'
    assert result['eta_reliability']=='NOT_ESTABLISHED'
    assert result['authorizes_full_run'] is False


def test_unverified_isolation_is_not_silently_assumed_exclusive():
    values=sample();del values['concurrent'];del values['instrumented']
    result=load('performance').estimate(values)
    assert 'CONCURRENCY_UNVERIFIED' in result['limitations']
    assert 'INSTRUMENTATION_UNVERIFIED' in result['limitations']


def test_even_exclusive_process_timing_is_not_kernel_timing():
    result=load('performance').estimate(sample(concurrent=False,instrumented=False))
    assert result['isolated_gpu_throughput']=='NOT_MEASURED'
    assert result['timing_scope']=='PROCESS_WALL_TIME_INCLUDING_STARTUP_AND_IO'


@pytest.mark.parametrize('changes',[{'physical_end_s':0.},{'physical_end_s':math.nan},
    {'wall_seconds':0.},{'wall_seconds':math.inf},{'physical_start_s':-.1},
    {'target_end_s':.0001},{'concurrent':'false'},{'instrumented':1}])
def test_invalid_sample_cannot_produce_a_finite_estimate(changes):
    with pytest.raises(ValueError):load('performance').estimate(sample(**changes))


def test_performance_cli_writes_only_a_diagnostic_report(tmp_path):
    source=tmp_path/'sample.json';source.write_text(json.dumps(sample()))
    output=tmp_path/'report.json'
    result=subprocess.run([sys.executable,str(SUITE/'performance.py'),str(source),'--output',str(output)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(output.read_text())['authorizes_full_run'] is False
