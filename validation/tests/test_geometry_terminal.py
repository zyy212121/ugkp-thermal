"""Bounded actual geometry producer continuation and original output/cost parity."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess

ROOT = SOURCE_ROOT
BASE = 'f23ab41d726bed9f53a36779ec0799ca77f5e0d7'
OWNER = 'applications/CHMT/mesh/Geometry.C'

def build_run(tmp, name, probe, owner=None, include_root=ROOT):
    source = tmp / (name + '.cpp'); source.write_text(probe)
    cmd = ['g++', '-std=c++14', '-O2', '-ffp-contract=off', '-Wall', '-Wextra', '-Werror', '-pedantic', '-Wno-mismatched-new-delete',
           '-I' + str(include_root / 'applications/CHMT'), '-I' + str(include_root / 'common'), str(source)]
    if owner: cmd.append(str(owner))
    cmd += ['-o', str(tmp / name)]
    run = subprocess.run(cmd, text=True, capture_output=True)
    (tmp / (name + '.build.log')).write_text(' '.join(cmd) + '\n' + run.stdout + run.stderr)
    assert run.returncode == 0, run.stderr
    run = subprocess.run([str(tmp / name)], text=True, capture_output=True, timeout=90)
    (tmp / (name + '.run.log')).write_text(run.stdout + run.stderr)
    assert run.returncode == 0, run.stderr
    return run.stdout, run.stderr

def test_terminal_geometry_actual_fields(tmp_path):
    owner = tmp_path / 'geometry.C'
    source = (ROOT / OWNER).read_text()
    source = 'bool forceQuad=false;unsigned long long quadCalls=0,harmonicIterations=0;\n' + source
    source = source.replace('for (int iter = 0; iter<20000 && !converged; ++iter) {',
                            'for (int iter = 0; iter<20000 && !converged; ++iter) {++harmonicIterations;')
    source = source.replace('int segments,bool empty) {', 'int segments,bool empty) {++quadCalls;')
    source = source.replace('if (absValue(refined-sweep) <= tolerance)', 'if (!forceQuad && absValue(refined-sweep) <= tolerance)')
    source=source.replace('!usedSolid.insert(f).second', 'false /* isolate edge producer */')
    owner.write_text(source)
    probe = ((VALIDATION_ROOT / 'tests/fixtures/geometry_terminal/terminal.cpp')).read_text()
    probe = probe.replace('GEOMETRY_OWNER', str(owner)).replace('GEOMETRY_FIXTURE', str((VALIDATION_ROOT / 'solvers/CHMT/devtools/multirate/test_sweep_constraints.cpp')))
    build_run(tmp_path, 'terminal', probe)

def test_normal_geometry_exact_fields_and_costs(tmp_path):

    import importlib.util
    spec = importlib.util.spec_from_file_location('hoists', (VALIDATION_ROOT / 'tests/test_geometry_scale_hoists.py'))
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    import io, tarfile
    historical=tmp_path/'historical';historical.mkdir()
    archive=subprocess.check_output(['git','archive',BASE,'applications/CHMT','common'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar: tar.extractall(historical, filter='data')
    outputs=[]
    for label, source in [('old', subprocess.check_output(['git','show',BASE+':'+OWNER],cwd=ROOT,text=True)), ('current',(ROOT/OWNER).read_text())]:

        source=source.replace('rebuildTrajectorySurfaceImpl(', 'rebuildTrajectorySurface(') if label=='old' else source
        if label=='current':
            a=source.index('    bool rebuildTrajectorySurfaceImpl(');b=source.index('    bool rebuildGeometry(HostMesh& m',a)
            body=source[a:b].replace('meshScale(oldSolid.points)', '(++edgeScaleCalls,meshScale(oldSolid.points))')
            needle='Real refined = edgeSweepQuadrature<Terminal>(oldSolid, solid.points, reference, edge, segments,emptySolid);'
            body=body.replace(needle,needle+'\n                std::cout<<"QUAD "<<edge.owner<<" "<<segments<<" "<<refined<<"\\n";')
            source=source[:a]+body+source[b:]
            a=source.index('        const auto physicalSolveTargets=solveTargets;');b=source.index('        if (!constrainedMotion',a)
            source=source[:a]+source[a:b].replace('meshScale(oldSolid.points)','(++materialScaleCalls,meshScale(oldSolid.points))')+source[b:]
            source='#include <iostream>\nunsigned long long edgeScaleCalls=0,materialScaleCalls=0,allScaleCalls=0,pointVisits=0;\n'+source
            source=source.replace('Real meshScale(const std::vector<Vec3>& p) {','Real meshScale(const std::vector<Vec3>& p) {++allScaleCalls;pointVisits+=p.size();')
        else: source=module.instrument(source)
        import re
        source='unsigned long long publicJacobi=0,modeJacobi=0,geometryCalls=0,stageCalls=0,meshCopies=0;\n'+source
        source=source.replace('for (int iter = 0; iter<20000 && !converged; ++iter) {','for (int iter = 0; iter<20000 && !converged; ++iter) {++publicJacobi;')
        source=re.sub(r'(bool geometryInPlace\([^)]*\) \{)',r'\1++geometryCalls;',source)
        source=re.sub(r'(bool makeStageGeometry(?:Impl)?\(const HostMesh& accepted,[^{]+\{)',r'\1++stageCalls;',source)
        source=re.sub(r'HostMesh \w+\s*=[^;]+;',lambda m:m[0]+'meshCopies+='+str(m[0].count('='))+';',source)
        modes=((historical if label=='old' else ROOT)/'applications/CHMT/mesh/SweepWorkspace.H').read_text()
        modes=modes.replace('for(int iteration=0;iteration<20000 && !converged;++iteration) {','for(int iteration=0;iteration<20000 && !converged;++iteration) {++modeJacobi;')
        modes_path=tmp_path/(label+'-modes.H');modes_path.write_text(modes)
        source=source.replace('"mesh/SweepWorkspace.H"','"'+str(modes_path)+'"')
        owner=tmp_path/(label+'.C');owner.write_text(source)
        probe=((VALIDATION_ROOT / 'tests/fixtures/geometry_hoists/probe.cpp')).read_text().replace('GEOMETRY_FIXTURE',str(historical/'applications/CHMT/devtools/multirate/test_sweep_constraints.cpp' if label=='old' else VALIDATION_ROOT/'solvers/CHMT/devtools/multirate/test_sweep_constraints.cpp'))
        allocation = """
#include <new>
#include <cstdlib>
extern unsigned long long publicJacobi,modeJacobi,geometryCalls,stageCalls,meshCopies;
unsigned long long allocations=0,allocationBytes=0;
void* operator new(std::size_t n){++allocations;allocationBytes+=n;if(void* p=std::malloc(n))return p;throw std::bad_alloc();}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
"""
        probe=allocation+probe
        probe=probe.replace('void reset(){','void reset(){publicJacobi=modeJacobi=geometryCalls=stageCalls=meshCopies=allocations=allocationBytes=0;')
        probe=probe.replace("<<pointVisits<<'\\n';", "<<pointVisits<<' '<<allocations<<' '<<allocationBytes<<' '<<publicJacobi<<' '<<modeJacobi<<' '<<geometryCalls<<' '<<stageCalls<<' '<<meshCopies<<'\\n';")
        outputs.append(build_run(tmp_path,label,probe,owner,historical if label=='old' else ROOT))
    assert outputs[0][0] == outputs[1][0]
    for before,after in zip(outputs[0][1].splitlines(),outputs[1][1].splitlines()):
        old=list(map(int,before.split()));new=list(map(int,after.split()))
        assert old[:7]==new[:7] and old[9:]==new[9:]
        assert new[7]<=old[7] and new[8]<=old[8]
    assert all(len(row.split()) == 14 for row in outputs[1][1].splitlines())
    levels=[int(row.split()[2]) for row in outputs[1][0].splitlines() if row.startswith('QUAD ')]
    assert max(levels)>16
