"""Compile public launch options: callers cannot interchange independent policies."""
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[1]
def test_named_launch_options(tmp_path):
    cases = [
        ('SegmentedMomentOptions x{MomentPayload::gatherSurvivors, MomentRecovery::deferToAdvance};', True),
        ('SegmentedMomentOptions x{true, false};', False),
        ('SegmentedMomentOptions x{MomentRecovery::completeHere, MomentPayload::momentsOnly};', False),
        ('FlatPressureSegment x = 1;', False),
        ('FlatPressureSegment x = FlatPressureSegment::base;', True),
    ]
    for n,(body,valid) in enumerate(cases):
        source=tmp_path/f'options{n}.cpp'
        source.write_text('#include "GpuLaunchOptions.cuh"\n#include "GpuPressureFlatLayout.cuh"\nint main(){'+body+'}\n')
        result=subprocess.run(['g++','-std=c++17','-fsyntax-only','-I'+str(ROOT/'common'),str(source)],capture_output=True,text=True)
        assert (result.returncode==0)==valid,result.stderr
