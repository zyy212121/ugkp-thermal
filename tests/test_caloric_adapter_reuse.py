"""Actual CHMT caloric adapter secants, exact outputs and source-call counts."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '60ddb604dc25e0e451c8d46848a043f831189d6e'
OWNER = 'applications/CHMT/materials/GasThermoAdapter.H'


def test_adapter_reuse_exact_secants(tmp_path):
    base = subprocess.check_output(['git', 'show', BASE + ':' + OWNER], cwd=ROOT, text=True)
    outputs = []
    counts = []
    for label, source in [('base', base), ('current', (ROOT / OWNER).read_text())]:
        header = tmp_path / (label + '.H')
        needle = 'CHMT_HD explicit GasThermoAdapter(const SpeciesThermo* input) {'
        assert source.count(needle) == 1
        header.write_text(source.replace(needle, needle + ' ++adapterConstructions;'))
        # The historical adapter is paired with its historical caloric users;
        # current users also reference the new private terminal helper templates.
        overlay = tmp_path / label / 'materials'
        overlay.mkdir(parents=True)
        if label == 'base':
            for dependency in ('MaterialCaloric.H', 'Thermodynamics.H', 'Reaction.H'):
                old = subprocess.check_output(['git', 'show', BASE + ':applications/CHMT/materials/' + dependency],
                                              cwd=ROOT, text=True)
                (overlay / dependency).write_text(old)
        probe = tmp_path / (label + '.cpp')
        probe.write_text((ROOT / 'tests/fixtures/caloric_reuse/probe.cpp').read_text().replace('ADAPTER_HEADER', str(header)))
        exe = tmp_path / (label + '-probe')
        cmd = ['g++', '-O2', '-std=c++14', '-ffp-contract=off', '-D_GLIBCXX_ASSERTIONS',
               '-I' + str(tmp_path / label), '-I' + str(ROOT / 'applications/CHMT'), '-I' + str(ROOT / 'applications/CHMT/materials'),
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
