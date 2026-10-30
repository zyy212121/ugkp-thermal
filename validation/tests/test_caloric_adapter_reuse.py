"""Actual CHMT caloric adapter secants, exact outputs and source-call counts."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import io
import tarfile
import pytest

ROOT = SOURCE_ROOT
BASE = '60ddb604dc25e0e451c8d46848a043f831189d6e'
OWNER = 'applications/CHMT/materials/GasThermoAdapter.H'


def test_adapter_reuse_exact_secants(tmp_path):
    base = subprocess.check_output(['git', 'show', BASE + ':' + OWNER], cwd=ROOT, text=True)
    baseline=tmp_path/'frozen';baseline.mkdir()
    archive=subprocess.check_output(['git','archive',BASE,'applications/CHMT','common'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(baseline,filter='data')
    outputs = []
    counts = []
    for label, source in [('base', base), ('current', (ROOT / OWNER).read_text())]:
        header = tmp_path / (label + '.H')
        needle = ('CHMT_HD explicit GasThermoAdapter(const SpeciesThermo* input) {' if label == 'base'
                  else 'CHMT_HD explicit GasThermoAdapter(const SpeciesThermo* input,bool terminal=false) {')
        assert source.count(needle) == 1
        header.write_text(source.replace(needle, needle + ' ++adapterConstructions;'))


        overlay = tmp_path / label / 'materials'
        overlay.mkdir(parents=True)
        if label == 'base':
            for dependency in ('MaterialCaloric.H', 'Thermodynamics.H', 'Reaction.H'):
                old = subprocess.check_output(['git', 'show', BASE + ':applications/CHMT/materials/' + dependency],
                                              cwd=ROOT, text=True)
                (overlay / dependency).write_text(old)
        probe = tmp_path / (label + '.cpp')
        probe.write_text(((VALIDATION_ROOT / 'tests/fixtures/caloric_reuse/probe.cpp')).read_text().replace('ADAPTER_HEADER', str(header)))
        exe = tmp_path / (label + '-probe')
        frozen_includes = ['-I'+str(baseline/'applications/CHMT'),'-I'+str(baseline/'common')] if label=='base' else []
        cmd = ['g++', '-O2', '-std=c++17', '-ffp-contract=off', '-D_GLIBCXX_ASSERTIONS',
               '-I' + str(tmp_path / label), *frozen_includes, '-I' + str(ROOT / 'applications/CHMT'), '-I' + str(ROOT / 'applications/CHMT/materials'),
               '-I' + str(ROOT / 'common'), '-MMD', '-MF', str(tmp_path / (label + '.d')), str(probe), '-o', str(exe)]
        build = subprocess.run(cmd, capture_output=True, text=True)
        (tmp_path / (label + '-build.log')).write_text(build.stdout + build.stderr)
        assert build.returncode == 0, build.stderr
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=30)
        (tmp_path / (label + '.out')).write_text(run.stdout)
        (tmp_path / (label + '.counts')).write_text(run.stderr)
        assert run.returncode == 0, run.stderr
        outputs.append(run.stdout)
        counts.append([list(map(int, row.split())) for row in run.stderr.splitlines()])
    assert outputs[0] == outputs[1]
    assert len(counts[0]) == len(counts[1]) == 48
    for before, after in zip(*counts):
        model, same, crossing, constructors = after
        assert before[:3] == after[:3]
        assert constructors == (1 if same else 4 if crossing else 1)
        assert before[3] == (1 if same else 8 if crossing else 3)



@pytest.mark.parametrize('conflicting_ref',[False,True])
def test_offline_history_is_automatic_and_preserves_checkout(tmp_path,conflicting_ref):
    import hashlib
    import os
    import shutil
    import sys
    checkout=tmp_path/'checkout';checkout.mkdir()
    def git(*args):
        return subprocess.check_output(['git','-C',str(checkout),*args],text=True).strip()
    git('init','--initial-branch=main')
    git('-c','gc.auto=0','fetch','--no-tags','--no-write-fetch-head','--no-recurse-submodules',
        str(ROOT),'82f9c016704c9822b16b87cdeeb11f5bbd44d06f')
    missing=subprocess.run(['git','-C',str(checkout),'cat-file','-e',BASE+'^{commit}'],
        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    assert missing.returncode!=0
    anchor=checkout/'anchor';anchor.write_text('committed\n')
    git('add','anchor')
    git('-c','user.name=Offline fixture','-c','user.email=fixture@example.invalid',
        'commit','--no-gpg-sign','-m','fixture')
    anchor.write_text('staged\n');git('add','anchor');anchor.write_text('working\n')
    validation=checkout/'validation'
    tests=validation/'tests';tests.mkdir(parents=True)
    for relative in ('__init__.py','conftest.py','tools/__init__.py',
                     'tools/paths.py','tests/conftest.py'):
        target=validation/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(VALIDATION_ROOT/relative,target)
    for name in ('first_round_history','continuation_history'):
        for relative in ('tools/tests/fixtures/'+name+'/prepare.py',
                         'tests/fixtures/'+name+'/history.gitbundle'):
            target=validation/relative;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(VALIDATION_ROOT/relative,target)
    (tests/'test_frozen_history.py').write_text(
        "import subprocess\n"
        "def test_actual_frozen_objects():\n"
        "    for pin in ('60ddb604dc25e0e451c8d46848a043f831189d6e','d7d954cf31da97c587ce8b086cd41834377aae8c','dc79925991e8dd4f4fc8f339942a3b9e85e0c633'):\n"
        "        subprocess.check_call(['git','cat-file','-e',pin+'^{commit}'])\n")
    fetch_head=checkout/'.git/FETCH_HEAD';fetch_head.write_bytes(b'preserved offline fetch head\n')
    head,branch=git('rev-parse','HEAD'),git('symbolic-ref','HEAD')
    if conflicting_ref:git('update-ref','refs/audit/first-round-original-history',head)
    index=(checkout/'.git/index').read_bytes()
    def inventory():
        return {str(p.relative_to(checkout)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in checkout.rglob('*') if p.is_file() and '.git' not in p.relative_to(checkout).parts}
    before=inventory()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',GIT_ALLOW_PROTOCOL='file',GIT_TERMINAL_PROMPT='0')
    command=[sys.executable,'-m','pytest','-q','--prepare-audit-history','validation/tests/test_frozen_history.py',
        '-o','cache_dir='+str(tmp_path/'cache'),'--basetemp',str(tmp_path/'nested')]
    for repeat in range(1 if conflicting_ref else 2):
        run=subprocess.run(command,cwd=checkout,env=env,capture_output=True,text=True)
        (tmp_path/('offline-history-'+str(repeat)+'.log')).write_text(run.stdout+run.stderr)
        if conflicting_ref:
            assert run.returncode!=0
            assert git('rev-parse','refs/audit/first-round-original-history')==head
        else:
            assert run.returncode==0,run.stdout+run.stderr
        assert git('rev-parse','HEAD')==head and git('symbolic-ref','HEAD')==branch
        assert (checkout/'.git/index').read_bytes()==index
        assert fetch_head.read_bytes()==b'preserved offline fetch head\n'
        assert inventory()==before
