"""Actual reaction/film terminal producers, with normal-prefix cost witnesses."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess

ROOT = SOURCE_ROOT
APP = ROOT / 'applications/CHMT'


def build_run(tmp_path, source, includes=(), flags=()):
    exe = tmp_path / (source.stem + ".exe")
    cmd = ['g++', '-std=c++14', '-O2', '-ffp-contract=off', '-Wall', '-Wextra',
           '-Werror', '-pedantic', *flags, *['-I'+str(p) for p in includes],
           '-I'+str(APP), '-I'+str(VALIDATION_ROOT/'solvers/CHMT'), '-I'+str(ROOT/'common'), str(source), '-o', str(exe)]
    build = subprocess.run(cmd, text=True, capture_output=True)
    (tmp_path/(source.stem+'.build.log')).write_text(' '.join(cmd)+'\n'+build.stdout+build.stderr)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(exe)], text=True, capture_output=True, timeout=60)
    (tmp_path/(source.stem+'.run.log')).write_text(run.stdout+run.stderr)
    assert run.returncode == 0, run.stderr
    return run


def test_reaction_terminal_original_requested_euler(tmp_path):
    overlay = tmp_path/'overlay/materials'; overlay.mkdir(parents=True)
    source = (APP/'materials/Reaction.H').read_text()
    needle = '    Real localRates[Nr]{};'
    assert source.count(needle) == 1
    source = source.replace('namespace chmt {',
        'static unsigned long long strictEuler=0,terminalEuler=0;\nnamespace chmt {', 1)
    source = source.replace(needle, '    if(Terminal)++terminalEuler;else ++strictEuler;\n'+needle)
    (overlay/'Reaction.H').write_text(source)
    (overlay/'ReactionStepControl.H').write_text((APP/'materials/ReactionStepControl.H').read_text())
    probe=tmp_path/'reaction.cpp'
    probe.write_text(((VALIDATION_ROOT / 'tests/fixtures/terminal_sources/reaction.cpp')).read_text())
    run=build_run(tmp_path,probe,[overlay.parent])
    assert 'terminal_prefix_euler 120 extra_euler 1 paid_dt 0.01' in run.stdout


def test_film_last_evaluated_and_complete_tails(tmp_path):
    fixture=tmp_path/'film-fixture.cpp'
    fixture.write_text(((VALIDATION_ROOT / 'solvers/CHMT/devtools/multirate/test_cpu_film.cpp')).read_text().replace('std::cout<<"CPU film tests passed\\n";}','std::cout<<"CPU film tests passed\\n";return 0;}'))
    owner=tmp_path/'film-owner.C'
    source=(APP/'film/CpuFilmDriver.C').read_text()
    assert 'if(!Terminal&&(owner<0||static_cast<std::size_t>(owner)>=n||neighbour< -1' in source
    source=source.replace('namespace chmt { namespace {',
        'static unsigned long long filmTrials=0;\nnamespace chmt { namespace {',1)
    source=source.replace('for(int trial=0;trial<48;++trial) {',
        'for(int trial=0;trial<48;++trial) { ++filmTrials;')
    owner.write_text(source)
    probe=tmp_path/'film.cpp'
    probe.write_text(((VALIDATION_ROOT / 'tests/fixtures/terminal_sources/film.cpp')).read_text()
        .replace(' assert(candidate.filmAux[0].kineticEnergy==h.filmAux[0].kineticEnergy);','')
        .replace('FILM_OWNER',str(owner))
        .replace('FILM_FIXTURE',str(fixture)))
    build_run(tmp_path,probe)


def test_original_film_bytes_and_costs(tmp_path):
    import importlib.util
    import io
    import tarfile
    baseline=tmp_path/'baseline';baseline.mkdir()
    archive=subprocess.check_output(['git','archive','8ed8a306cbd7eab5f2210341b45b6bf078b57861',
        'applications/CHMT','common'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(baseline,filter='data')
    spec=importlib.util.spec_from_file_location('film_counts',(VALIDATION_ROOT / 'tests/test_film_incidence.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    results=[]
    for label,root in [('base',baseline),('current',ROOT)]:
        app=root/'applications/CHMT'
        support=app if label=='base' else VALIDATION_ROOT/'solvers/CHMT'
        owner=tmp_path/(label+'-owner.C')
        owner.write_text(module.instrument((app/'film/CpuFilmDriver.C').read_text()))
        fixture=tmp_path/(label+'-fixture.cpp')
        fixture.write_text((support/'devtools/multirate/test_cpu_film.cpp').read_text().replace(
            'std::cout<<"CPU film tests passed\\n";}','std::cout<<"CPU film tests passed\\n";return 0;}'))
        probe=tmp_path/(label+'-parity.cpp')
        probe.write_text(module.legacy_transport_adapter((app/'film/CpuFilmDriver.H').read_text())+module.physical_film_probe(((VALIDATION_ROOT / 'tests/fixtures/film_incidence/probe.cpp')).read_text())
            .replace('FILM_OWNER',str(owner)).replace('FILM_FIXTURE',str(fixture)))

        results.append(build_run(tmp_path,probe,[app,root/'common'],['-Wno-misleading-indentation','-Wno-mismatched-new-delete']))
    assert results[0].stdout==results[1].stdout


    old_lines=results[0].stderr.splitlines();new_lines=results[1].stderr.splitlines()
    assert len(old_lines)==len(new_lines)
    edges_by_faces={}
    for old,new in zip(old_lines,new_lines):
        if old.startswith('COUNT '):
            before=list(map(int,old.split()[1:]));after=list(map(int,new.split()[1:]))
            faces,edges=before[:2];edges_by_faces[faces]=edges
            assert after[:4]==before[:4]
            assert after[4]==before[4]-2
            assert after[5]==before[5]-64*faces-32*edges-8*faces
        elif old.startswith('ADVCOUNT '):
            before=list(map(int,old.split()[1:]));after=list(map(int,new.split()[1:]))
            faces=before[0];assert after[0]==faces
            assert after[1]==before[1]-4
            assert after[2]==before[2]-2*(64*faces+32*edges_by_faces[faces])-32*faces
        else:
            if old.startswith('ISOLATED '):
                before=list(map(int,old.split()[1:]));after=list(map(int,new.split()[1:]))
                assert after[:-1]==before[:-1] and after[-1]==before[-1]-8
            else:
                assert old==new

    reaction=[]
    for label,root in [('base',baseline),('current',ROOT)]:
        app=root/'applications/CHMT'
        overlay=tmp_path/(label+'-reaction/materials');overlay.mkdir(parents=True)
        source=(app/'materials/Reaction.H').read_text()
        source=source.replace('namespace chmt {','static unsigned long long eulerCalls=0;\nnamespace chmt {',1)
        source=source.replace('    Real localRates[Nr]{};','    ++eulerCalls;Real localRates[Nr]{};')
        (overlay/'Reaction.H').write_text(source)
        (overlay/'ReactionStepControl.H').write_text((app/'materials/ReactionStepControl.H').read_text())
        probe=tmp_path/(label+'-reaction.cpp')
        probe.write_text(((VALIDATION_ROOT / 'tests/fixtures/terminal_sources/reaction_parity.cpp')).read_text())
        reaction.append(build_run(tmp_path,probe,[overlay.parent,app,root/'common']))
    assert reaction[0].stdout==reaction[1].stdout


def test_interval_progress_refusal_and_actual_packet_tail(tmp_path):



    source=(APP/'materials/MaterialTransport.C').read_text()


    assert 'if(!Terminal&&(packet.solidCell<0||static_cast<std::size_t>(packet.solidCell)>=base.size())){if(numericalFailure)*numericalFailure=true;' in source
    hook=r'''
static int proposalCalls=0;
namespace chmt {
template<bool Terminal> bool stagnationCandidate(const SolidQ& q,Real v,const PhysicsConfig& p,
 Real,Real,SolidQ& out,Real& dt,bool& active,bool* numerical){
 dt=proposalCalls++==0?.5:1e-20;active=true;
 return advanceLocalReactions<true>(q,v,p,dt,out,nullptr,nullptr,-1,numerical);
}
}
'''
    source=source.replace('namespace chmt { namespace {',hook+'namespace chmt { namespace {',1)
    source=source.replace('reactionStepCandidate<Terminal>(', 'stagnationCandidate<Terminal>(')
    owner=tmp_path/'transport.C';owner.write_text(source)
    probe=tmp_path/'interval.cpp'
    probe.write_text(((VALIDATION_ROOT / 'tests/fixtures/terminal_sources/interval.cpp')).read_text().replace('TRANSPORT_OWNER',str(owner)))
    build_run(tmp_path,probe)
