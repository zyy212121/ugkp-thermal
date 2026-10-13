"""Actual predictor body, with host-only dependencies and explicit no-GPU scope."""
from pathlib import Path
import hashlib
import io
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
PIN='a999cde5fd2736bb269f9f0a9c792e565d3d7e18'

def owner(root):
    source=(root/'applications/CHMT/materials/CpuMaterialDriver.C').read_text()
    headers=source[:source.index('#include "fvCFD.H"')]
    begin=source.index('// Host predictor-only completion;') if '// Host predictor-only completion;' in source else source.index('bool baseTrace(')
    end=source.index('\nbool CpuMaterialDriver::advanceCandidate(')
    body=source[begin:end].replace('bulk.resize(base.surface.area.size());','++traceCalls;bulk.resize(base.surface.area.size());')
    body=body.replace('std::vector<GasWallClosureContext> contexts;', '++surfaceCalls;std::vector<GasWallClosureContext> contexts;')
    return headers+'\nstatic unsigned long long traceCalls=0,surfaceCalls=0;\nnamespace chmt { namespace {\n'+body+'\n}\n'

def run(root,tmp,terminal):
    app=root/'applications/CHMT';tmp.mkdir()
    (tmp/'owner.C').write_text(owner(root))
    fixture=ROOT/'tests/fixtures/predictor_terminal/probe.cpp'
    cmd=['g++','-std=c++14','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-pedantic','-Wno-misleading-indentation','-Wno-mismatched-new-delete', '-DTERMINAL_TEST='+str(int(terminal)), '-I'+str(tmp),'-I'+str(app),'-I'+str(root/'common'),'-I'+str(ROOT/'tests/fixtures/surface_terminal'),str(fixture),str(app/'ablation/CpuSurfaceInterface.C'),str(app/'mesh/Geometry.C'),'-o',str(tmp/'probe')]
    build=subprocess.run(cmd,text=True,capture_output=True)
    (tmp/'build.log').write_text(' '.join(cmd)+'\n'+build.stdout+build.stderr)
    assert build.returncode==0,build.stderr
    result=subprocess.run([str(tmp/'probe')],text=True,capture_output=True,timeout=60)
    (tmp/'run.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0,result.stderr
    return result.stdout

def test_predictor_original_fields_costs_and_terminal(tmp_path):
    baseline=tmp_path/'base';baseline.mkdir()
    archive=subprocess.check_output(['git','archive',PIN,'applications/CHMT','common'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(baseline,filter='data')
    old=run(baseline,tmp_path/'original',False)
    current=run(ROOT,tmp_path/'current',False)
    assert old==current
    assert old.count('COUNT ')==3
    terminal=run(ROOT,tmp_path/'terminal',True)
    assert 'actual predictor EOS/surface/prescribed tails and missing storage passed' in terminal
