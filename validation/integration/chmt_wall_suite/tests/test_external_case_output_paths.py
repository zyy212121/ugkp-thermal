"""Host-only CLI/path checks; the runner fixture never launches a real solver."""
from pathlib import Path
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[4]
CASES_ROOT = Path(os.environ.get('UGKP_CASES_ROOT', SOURCE_ROOT.parent / 'ugkp-thermal-cases')).resolve()
GAS_TOOLS = CASES_ROOT / 'test/tools/consistency/gasUGKP'


def load_tool(relative):
    path = GAS_TOOLS / relative
    spec = importlib.util.spec_from_file_location('output_path_' + path.parent.name + '_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cli_defaults(module, monkeypatch):
    class Parsed(Exception):
        pass
    captured = {}
    original = argparse.ArgumentParser.parse_args
    def capture(parser, *args, **kwargs):
        captured['args'] = original(parser, [])
        raise Parsed
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', capture)
    with pytest.raises(Parsed):
        module.main()
    return captured['args']


@pytest.mark.parametrize('relative,argument,expected', [
    ('mixtureTransport/make_case.py', 'case', 'test/consistency/gasUGKP/mixtureTransport'),
    ('mixtureTransport/reference.py', 'case', 'test/results/consistency/gasUGKP/mixtureTransport'),
    ('chemistryReactor/make_case.py', 'case', 'test/consistency/gasUGKP/chemistryReactor'),
    ('chemistryReactor/run.py', 'output', 'test/results/consistency/gasUGKP/chemistryReactor/runs'),
])
def test_input_and_runtime_cli_defaults_are_separated(relative, argument, expected, monkeypatch):
    monkeypatch.setenv('UGKP_SOURCE_ROOT', str(SOURCE_ROOT))
    args = cli_defaults(load_tool(relative), monkeypatch)
    assert getattr(args, argument) == CASES_ROOT / expected


@pytest.mark.parametrize('explicit_case', [False, True])
def test_mixture_runner_uses_one_result_case_for_every_command(tmp_path, explicit_case):
    isolated = tmp_path / 'cases'
    tools = isolated / 'test/tools/consistency/gasUGKP/mixtureTransport'
    tools.mkdir(parents=True)
    shutil.copyfile(GAS_TOOLS / 'mixtureTransport/Allrun', tools / 'Allrun')
    inputs = isolated / 'test/consistency/gasUGKP/mixtureTransport'
    inputs.mkdir(parents=True)
    (inputs / 'keep').write_text('unchanged input\n')
    binaries = tmp_path / 'bin'
    binaries.mkdir()
    log = tmp_path / 'commands.jsonl'
    stub = '#!' + sys.executable + '\n' + '''import json,os,sys
from pathlib import Path
with open(os.environ['COMMAND_LOG'],'a') as stream:
    stream.write(json.dumps({'command':Path(sys.argv[0]).name,'arguments':sys.argv[1:],'cwd':os.getcwd()})+'\\n')
if Path(sys.argv[0]).name=='python3' and sys.argv[1].endswith('reference.py'):
    print('{}')
'''
    for name in ('python3', 'blockMesh', 'gasUGKP'):
        path = binaries / name
        path.write_text(stub)
        path.chmod(0o755)
    case = tmp_path / 'explicit-results' if explicit_case else isolated / 'test/results/consistency/gasUGKP/mixtureTransport'
    command = ['bash', str(tools / 'Allrun'), '--cells', '8']
    if explicit_case:
        command += ['--case', str(case)]
    environment = dict(os.environ, PATH=str(binaries) + os.pathsep + os.environ['PATH'], WM_PROJECT_DIR=str(tmp_path), COMMAND_LOG=str(log))
    subprocess.run(command, cwd=tmp_path, env=environment, check=True, capture_output=True, text=True)
    calls = [json.loads(row) for row in log.read_text().splitlines()]
    assert [row['command'] for row in calls] == ['python3', 'blockMesh', 'gasUGKP', 'python3']
    assert all(Path(row['cwd']) == case for row in calls)
    for row, script in ((calls[0], 'make_case.py'), (calls[-1], 'reference.py')):
        assert Path(row['arguments'][0]) == tools / script
        index = row['arguments'].index('--case')
        assert Path(row['arguments'][index + 1]) == case
    assert (case / 'comparison.json').is_file()
    assert sorted(path.name for path in inputs.iterdir()) == ['keep']
    assert (inputs / 'keep').read_text() == 'unchanged input\n'
    assert sorted(path.name for path in tools.iterdir()) == ['Allrun']
