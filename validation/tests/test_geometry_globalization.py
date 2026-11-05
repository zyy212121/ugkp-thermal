"""Real fixed-interval geometry globalization and positive wall quadrature regressions."""
from pathlib import Path
import subprocess
import pytest
from validation.tools.paths import SOURCE_ROOT
ROOT=SOURCE_ROOT
FIXTURE=Path(__file__).parent/'fixtures/geometry_globalization'
@pytest.mark.parametrize('name', ['sweep_manufactured','kernel','production_geometry','contracts','kernel_scan'])
def test_fixed_equation_geometry(tmp_path,name):
    command=['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(ROOT),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasWall'),'-I'+str(ROOT/'applications/CHMT'),str(FIXTURE/(name+'.cpp'))]
    if name=='sweep_manufactured': command.append(str(ROOT/'applications/CHMT/mesh/Geometry.C'))
    exe=tmp_path/name;command+=['-o',str(exe)]
    built=subprocess.run(command,capture_output=True,text=True)
    assert built.returncode==0,built.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
