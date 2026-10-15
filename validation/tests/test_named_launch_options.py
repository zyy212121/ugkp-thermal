"""Compile public launch options: callers cannot interchange independent policies."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
ROOT = SOURCE_ROOT
def test_named_launch_options(tmp_path):
    cases = [
        ('SegmentedMomentOptions x{MomentPayload::gatherSurvivors, MomentRecovery::deferToAdvance};', True),
        ('SegmentedMomentOptions x{true, false};', False),
        ('SegmentedMomentOptions x{MomentRecovery::completeHere, MomentPayload::momentsOnly};', False),
    ]
    for n,(body,valid) in enumerate(cases):
        source=tmp_path/f'options{n}.cpp'
        source.write_text('#include "GpuLaunchOptions.cuh"\nint main(){'+body+'}\n')
        result=subprocess.run(['g++','-std=c++17','-fsyntax-only','-I'+str(ROOT/'common'),str(source)],capture_output=True,text=True)
        assert (result.returncode==0)==valid,result.stderr
