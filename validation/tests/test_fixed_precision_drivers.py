"""Execute changed CLI generation paths; intercept compilation before launching it."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import resource
import runpy
import subprocess
import sys

import pytest

ROOT = SOURCE_ROOT
DRIVERS = [
 'gas_auto/gas_auto_pool.py', 'pressure/limited_flux_balance.py',
 'shared_operators/collision_behavior.py', 'shared_operators/collision_split_behavior.py',
 'shared_operators/contact_age_theta.py', 'shared_operators/contact_conservation.py',
 'shared_operators/extended_operators.py', 'shared_operators/moment_publication.py',
 'shared_operators/operator_contract.py', 'shared_operators/payload_commit.py',
 'shared_operators/source_balance.py', 'shared_operators/tracking_grid.py',
 'thermal_workers/common_pipeline.py', 'thermal_workers/s1_pipeline.py',
 'thermal_workers/fused_pipeline.py',
]


@pytest.mark.parametrize('driver', DRIVERS)
def test_current_driver_generates_without_precision_argument(tmp_path, monkeypatch, driver):
    command = []
    class CompileReached(Exception):
        pass
    def capture(argv, *args, **kwargs):
        command.extend(map(str, argv))
        raise CompileReached()
    def forward(argv, *args, **kwargs):
        monkeypatch.setattr(sys, 'argv', list(map(str, argv[1:])))
        runpy.run_path(str(argv[1]), run_name='__main__')
    monkeypatch.setattr(subprocess, 'run', capture)
    monkeypatch.setattr(subprocess, 'call', forward)
    monkeypatch.setattr(resource, 'setrlimit', lambda *args: None)
    path = (VALIDATION_ROOT / 'tools/tests/fixtures') / driver
    monkeypatch.setattr(sys, 'argv', [str(path), str(ROOT), str(tmp_path), 'CHT'])
    with pytest.raises(CompileReached):
        runpy.run_path(str(path), run_name='__main__')
    assert Path(command[0]).name == 'nvcc'
    assert not any('REAL_BITS' in arg for arg in command)
    if driver != 'shared_operators/payload_commit.py':
        assert str(ROOT/'applications/CHT/gpu/GpuWallEnergy64.cu') in command
    assert list(tmp_path.rglob('*.cu'))


@pytest.mark.parametrize('driver,mode,needle', [
 ('gas_auto/gas_auto_pool.py', 'production-directory', '--wrap=cudaLaunchKernel'),
 ('pressure/limited_flux_balance.py', 'unsorted', 'ConstrainedPressureLaunch<true>::projectUnsorted(s,1.,1,64)'),
 ('pressure/limited_flux_balance.py', 'split', 'ConstrainedPressureLaunch<true>::project(s,1.,1,64,64,true,false)'),
 ('thermal_workers/common_pipeline.py', 'L1', 'L1 shared moment kernel'),
])
def test_mode_is_first_argument_after_application(tmp_path, monkeypatch, driver, mode, needle):
    commands=[]
    class CompileReached(Exception): pass
    def capture(argv,*args,**kwargs):
        commands.extend(map(str,argv));raise CompileReached()
    monkeypatch.setattr(subprocess,'run',capture)
    monkeypatch.setattr(resource,'setrlimit',lambda *args:None)
    path=(VALIDATION_ROOT / 'tools/tests/fixtures')/driver
    monkeypatch.setattr(sys,'argv',[str(path),str(ROOT),str(tmp_path),'CHT',mode])
    with pytest.raises(CompileReached):runpy.run_path(str(path),run_name='__main__')
    generated='\n'.join(p.read_text() for p in tmp_path.rglob('*.cu'))
    assert needle in generated+'\n'+' '.join(commands)
