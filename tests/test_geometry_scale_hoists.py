"""Actual coupled geometry output/refinement parity and invariant scan counts."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = '528f8937f109cec51fb9016145f565f78e09d19e'
OWNER = 'applications/CHMT/mesh/Geometry.C'


def instrument(source):
    source = '#include <iostream>\nunsigned long long edgeScaleCalls=0,materialScaleCalls=0,allScaleCalls=0,pointVisits=0;\n' + source
    source = source.replace('Real meshScale(const std::vector<Vec3>& p) {',
                            'Real meshScale(const std::vector<Vec3>& p) {++allScaleCalls;pointVisits+=p.size();')
    a = source.index('    bool rebuildTrajectorySurface(const HostMesh&')
    b = source.index('    bool rebuildTrajectorySurface(const HostState&', a)
    body = source[a:b].replace('meshScale(oldSolid.points)', '(++edgeScaleCalls,meshScale(oldSolid.points))')
    needle = 'Real refined = edgeSweepQuadrature(oldSolid, solid.points, reference, edge, segments,emptySolid);'
    assert body.count(needle) == 1
    body = body.replace(needle, needle + '\n                std::cout<<"QUAD "<<edge.owner<<" "<<segments<<" "<<refined<<"\\n";')
    source = source[:a] + body + source[b:]
    a = source.index('        const auto physicalSolveTargets=solveTargets;')
    b = source.index('        if (!constrainedMotion(', a)
    source = source[:a] + source[a:b].replace('meshScale(oldSolid.points)', '(++materialScaleCalls,meshScale(oldSolid.points))') + source[b:]
    return source


def test_geometry_scale_exact_outputs_and_counts(tmp_path):
    before = subprocess.check_output(['git', 'show', BASE + ':' + OWNER], cwd=ROOT, text=True)
    results = []
    for label, source in [('base', before), ('current', (ROOT / OWNER).read_text())]:
        owner = tmp_path / (label + '.C');owner.write_text(instrument(source))
        probe = tmp_path / (label + '.cpp')
        probe.write_text((ROOT / 'tests/fixtures/geometry_hoists/probe.cpp').read_text().replace('GEOMETRY_FIXTURE', str(ROOT / 'applications/CHMT/devtools/multirate/test_sweep_constraints.cpp')))
        exe = tmp_path / label
        command = ['g++', '-O2', '-std=c++14', '-ffp-contract=off', '-D_GLIBCXX_ASSERTIONS',
                   '-I' + str(ROOT / 'applications/CHMT'), '-I' + str(ROOT / 'common'),
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
    # Actual large-deformation fixture must preserve a successful refinement >16.
    levels = [int(row.split()[2]) for row in results[1].stdout.splitlines() if row.startswith('QUAD ')]
    assert max(levels) > 16
