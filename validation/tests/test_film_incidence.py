"""Actual CPU film source parity and stable-neighbour traversal counts (host only)."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import io
import tarfile

ROOT = SOURCE_ROOT
BASE = '6dd13b85287435ff36b0d4d5cc951ba36385f71b'
OWNER = 'applications/CHMT/film/CpuFilmDriver.C'
APP = ROOT / 'applications/CHMT'


def instrument(source):
    source = source.replace('namespace chmt { namespace {',
        'unsigned long long gradientVisits=0,limiterVisits=0,profileCalls=0,acceptedTrial=0,buildEdgeVisits=0,buildEntries=0,buildFaces=0;\nnamespace chmt { namespace {', 1)
    scan = 'for(std::size_t edge=0;edge<m.edgeOwner.size();++edge) {'
    indexed = 'for(std::size_t entry=e.incidentOffsets[f];entry<e.incidentOffsets[f+1];++entry) {'
    for start, end, counter in [('Vec3 surfaceGradient(', 'FilmProfile faceProfile(', 'gradientVisits'),
                                 ('bool reconstruct(', 'bool recoverAndProfile(', 'limiterVisits')]:
        a, b = source.index(start), source.index(end)
        body = source[a:b]
        loop = indexed if indexed in body else scan
        assert body.count(loop) == 1
        source = source[:a] + body.replace(loop, loop + ' ++' + counter + ';') + source[b:]
    source = source.replace('e.q=h.film;e.aux=h.filmAux;', '++profileCalls;e.q=h.film;e.aux=h.filmAux;')
    source = source.replace('if(recoverFilm(q,1,e.aux[f].pressure,p,aux)){result=q;',
                            'if(recoverFilm(q,1,e.aux[f].pressure,p,aux)){acceptedTrial=trial;result=q;')
    a, b = source.index('bool recoverAndProfile('), source.index('void addRate(')
    body = source[a:b]
    body = body.replace(scan, scan + ' ++buildEdgeVisits;')
    body = body.replace('e.incident[next[m.edgeOwner[edge]]++]=edge;', 'e.incident[next[m.edgeOwner[edge]]++]=edge;++buildEntries;')
    body = body.replace('e.incident[next[neighbour]++]=edge;', 'e.incident[next[neighbour]++]=edge;++buildEntries;')
    body = body.replace('for(std::size_t f=0;f<n;++f)e.incidentOffsets[f+1]+=e.incidentOffsets[f];',
        'for(std::size_t f=0;f<n;++f){++buildFaces;e.incidentOffsets[f+1]+=e.incidentOffsets[f];}')
    return source[:a] + body + source[b:]


def legacy_transport_adapter(header):
    """Bridge only the test call shape; execute the historical RHS unchanged."""
    if 'std::vector<FilmRateBudget>& budgets' not in header:
        return ''
    return r'''#include "film/CpuFilmDriver.H"
namespace chmt {
inline bool evaluateCpuFilmTransport(const HostState& h,const PhysicsConfig& p,Real dt,
 const std::vector<CpuFilmDrive>& d,std::vector<FilmQ>& rates,std::string& error,
 std::vector<FilmAux>* aux=nullptr,std::vector<Vec3>* gradients=nullptr,std::vector<Real>* side=nullptr){
 std::vector<FilmRateBudget> oldOutput;
 return evaluateCpuFilmTransport(h,p,dt,d,rates,oldOutput,error,aux,gradients,side);
}
}
'''


def physical_film_probe(source):
    source=source.replace("<<a.kineticEnergy<<' '",'')
    source=source.replace('  require(near(c.filmAux[f].kineticEnergy,p.liquid.rho*profile.kineticPerArea,1e-15),"endpoint kinetic profile");','')
    source=source.replace(',beforeK=0,afterK=0','')
    source=source.replace('beforeK+=h.filmAux[f].kineticEnergy;afterK+=c.filmAux[f].kineticEnergy;','')
    source=source.replace("<<' '<<beforeK<<' '<<afterK",'')
    return source


def test_film_exact_outputs_and_linear_incidence(tmp_path):
    original = subprocess.check_output(['git', 'show', BASE + ':' + OWNER], cwd=ROOT, text=True)
    current = (ROOT / OWNER).read_text()


    baseline = tmp_path / 'baseline'
    baseline.mkdir()
    archive = subprocess.check_output(['git', 'archive', BASE, 'applications/CHMT', 'common'], cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(baseline, filter='data')
    results = []
    for label, source in [('base', original), ('current', current)]:
        owner = tmp_path / (label + '.C')
        owner.write_text(instrument(source))
        body = physical_film_probe(((VALIDATION_ROOT / 'tests/fixtures/film_incidence/probe.cpp')).read_text())
        owner_app = baseline / 'applications/CHMT' if label == 'base' else APP
        body = legacy_transport_adapter((owner_app/'film/CpuFilmDriver.H').read_text()) + body
        support = owner_app if label == 'base' else VALIDATION_ROOT / 'solvers/CHMT'
        body = body.replace('FILM_FIXTURE', str(support / 'devtools/multirate/test_cpu_film.cpp'))
        body = body.replace('FILM_OWNER', str(owner))
        probe = tmp_path / (label + '.cpp')
        probe.write_text(body)
        exe = tmp_path / label
        includes = ['-I' + str(baseline / 'applications/CHMT'), '-I' + str(baseline / 'common')] if label == 'base' else []
        command = ['g++', '-std=c++14', '-O2', '-ffp-contract=off', '-D_GLIBCXX_ASSERTIONS',
                   *includes, '-I' + str(APP), '-I' + str(support), '-I' + str(ROOT / 'common'), '-MMD', '-MF', str(tmp_path / (label + '.d')), str(probe), '-o', str(exe)]
        build = subprocess.run(command, text=True, capture_output=True)
        (tmp_path / (label + '-build.log')).write_text(build.stdout + build.stderr)
        assert build.returncode == 0, build.stderr
        result = subprocess.run([str(exe)], text=True, capture_output=True, timeout=60)
        (tmp_path / (label + '.out')).write_text(result.stdout)
        (tmp_path / (label + '.counts')).write_text(result.stderr)
        assert result.returncode == 0, result.stderr
        results.append(result)
    assert results[0].stdout == results[1].stdout
    before = [list(map(int, line.split()[1:])) for line in results[0].stderr.splitlines() if line.startswith('COUNT ')]
    after = [list(map(int, line.split()[1:])) for line in results[1].stderr.splitlines() if line.startswith('COUNT ')]
    assert len(before) == len(after) == 4
    for old, new in zip(before, after):
        faces, edges, grad, limiter, allocations, allocated_bytes = new
        assert old[:2] == new[:2]
        assert grad == 36 * faces and limiter == 32 * faces
        assert grad < old[2] and limiter < old[3]

        assert allocations == old[4] + 1
        assert allocated_bytes == old[5] + 8 * (2 * (faces + 1) + 2 * edges) - 64 * faces - 32 * edges - 8 * faces


    for label, expected_delta in [('ADVCOUNT', 2), ('ISOLATED', 0)]:
        old_rows = [list(map(int, line.split()[1:])) for line in results[0].stderr.splitlines() if line.startswith(label + ' ')]
        new_rows = [list(map(int, line.split()[1:])) for line in results[1].stderr.splitlines() if line.startswith(label + ' ')]
        assert len(old_rows) == len(new_rows) == (4 if label == 'ADVCOUNT' else 2)
        for old, new in zip(old_rows, new_rows):
            assert new[0] == old[0]
            assert new[1] == old[1] + expected_delta
            if label == 'ISOLATED':
                assert new[2] == old[2] - 8

    construction = [list(map(int, line.split()[1:])) for line in results[1].stderr.splitlines() if line.startswith('BUILD ')]
    assert len(construction) == 4
    for faces, edges, edge_visits, entries, prefix_faces in construction:
        assert (edge_visits, entries, prefix_faces) == (2 * edges, 2 * edges, faces)
