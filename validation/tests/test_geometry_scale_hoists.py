"""Actual coupled geometry output/refinement parity and invariant scan counts."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import io
import tarfile

ROOT = SOURCE_ROOT
BASE = '528f8937f109cec51fb9016145f565f78e09d19e'
OWNER = 'applications/CHMT/mesh/Geometry.C'


def instrument(source):
    source = '#include <iostream>\nunsigned long long edgeScaleCalls=0,materialScaleCalls=0,allScaleCalls=0,pointVisits=0;\n' + source
    source = source.replace('Real meshScale(const std::vector<Vec3>& p) {',
                            'Real meshScale(const std::vector<Vec3>& p) {++allScaleCalls;pointVisits+=p.size();')
    signature='    bool rebuildTrajectorySurfaceImpl(' if '    bool rebuildTrajectorySurfaceImpl(' in source else '    bool rebuildTrajectorySurface(const HostMesh&'
    a=source.index(signature);b=source.index('{',a)+1;depth=1
    while depth:
        depth+=(source[b]=='{')-(source[b]=='}');b+=1
    body = source[a:b].replace('meshScale(oldSolid.points)', '(++edgeScaleCalls,meshScale(oldSolid.points))')
    generic='<Terminal>' if 'edgeSweepQuadrature<Terminal>' in body else ''
    needle = 'Real refined = edgeSweepQuadrature'+generic+'(oldSolid, solid.points, reference, edge, segments,emptySolid);'
    assert body.count(needle) == 1
    body = body.replace(needle, needle + '\n                std::cout<<"QUAD "<<edge.owner<<" "<<segments<<" "<<refined<<"\\n";')
    source = source[:a] + body + source[b:]
    a = source.index('        const auto physicalSolveTargets=solveTargets;')
    call='        if (!constrainedMotion'+('<Terminal>' if 'constrainedMotion<Terminal>(' in source[a:] else '')+'('
    b = source.index(call, a)
    source = source[:a] + source[a:b].replace('meshScale(oldSolid.points)', '(++materialScaleCalls,meshScale(oldSolid.points))') + source[b:]
    return source


def test_geometry_scale_exact_outputs_and_counts(tmp_path):
    before = subprocess.check_output(['git', 'show', BASE + ':' + OWNER], cwd=ROOT, text=True)
    baseline=tmp_path/'frozen';baseline.mkdir()
    archive=subprocess.check_output(['git','archive',BASE,'applications/CHMT','common'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(baseline,filter='data')
    results = []
    for label, source in [('base', before), ('current', (ROOT / OWNER).read_text())]:
        root=baseline if label=='base' else ROOT
        owner = tmp_path / (label + '.C');owner.write_text(instrument(source))
        probe = tmp_path / (label + '.cpp')
        probe.write_text(((VALIDATION_ROOT / 'tests/fixtures/geometry_hoists/probe.cpp')).read_text().replace('GEOMETRY_FIXTURE', str(root / 'applications/CHMT/devtools/multirate/test_sweep_constraints.cpp' if label == 'base' else VALIDATION_ROOT / 'solvers/CHMT/devtools/multirate/test_sweep_constraints.cpp')))
        exe = tmp_path / label
        command = ['g++', '-O2', '-std=c++14', '-ffp-contract=off', '-D_GLIBCXX_ASSERTIONS',
                   '-I' + str(root / 'applications/CHMT'), '-I' + str(root / 'applications/CHMT/mesh'), '-I' + str(root / 'common'),
                   '-MMD', '-MF', str(tmp_path / (label + '.d')), str(probe), str(owner), '-o', str(exe)]
        build = subprocess.run(command, capture_output=True, text=True)
        (tmp_path / (label + '-build.log')).write_text(build.stdout + build.stderr)
        assert build.returncode == 0, build.stderr
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
        (tmp_path / (label + '.out')).write_text(run.stdout);(tmp_path / (label + '.counts')).write_text(run.stderr)
        assert run.returncode == 0, run.stderr
        results.append(run)
    assert results[0].stdout == results[1].stdout
    old_rows = [list(map(int, row.split())) for row in results[0].stderr.splitlines()]
    new_rows = [list(map(int, row.split())) for row in results[1].stderr.splitlines()]
    assert len(old_rows) == len(new_rows) == 13
    for old, new in zip(old_rows, new_rows):
        case, edges, targets, edge_calls, material_calls, total, points = new
        assert old[:3] == new[:3]
        assert edge_calls == bool(edges)
        assert material_calls == bool(targets)
        assert old[3] == 2 * edges and old[4] == targets
        assert old[5] - total == (old[3] - edge_calls) + (old[4] - material_calls)
        assert old[6] - points == 12 * (old[5] - total)

    levels = [int(row.split()[2]) for row in results[1].stdout.splitlines() if row.startswith('QUAD ')]
    assert max(levels) > 16
