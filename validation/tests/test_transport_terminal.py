"""Material transport real last-candidate and complete consumer tails."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import io
import subprocess
import tarfile
ROOT=SOURCE_ROOT
PIN='d7d954cf31da97c587ce8b086cd41834377aae8c'

def build_run(root,dest,terminal):
    dest.mkdir();app=root/'applications/CHMT'
    source=(app/'materials/MaterialTransport.C').read_text()
    source=source.replace('namespace chmt { namespace {','static unsigned long long trials=0;static int rejectBefore=0;static bool rejectTrials=false;\nnamespace chmt { namespace {',1)
    if terminal:
        source=source.replace('namespace chmt { namespace {','namespace chmt { namespace {\nstatic SolidQ lastCandidate;static Real lastBound=0,nextEnergy=0;',1)
        source=source.replace('if(recoverMaterial(candidate,1,p,state))', 'lastCandidate=candidate;lastBound=bound;nextEnergy=addSolid(density,increment,bound*.5).energy;\n        if(!rejectTrials&&trial>=rejectBefore&&recoverMaterial(candidate,1,p,state))',1)
    source=source.replace('if(recoverMaterial(candidate,1,p,state))', 'if(trial>=rejectBefore&&recoverMaterial(candidate,1,p,state))',1)
    source=source.replace('for(int trial=0;trial<48;++trial){','for(int trial=0;trial<48;++trial){++trials;',1)

    source=source.replace('static bool rejectTrials=false;','static bool rejectTrials __attribute__((unused))=false;')
    (dest/'owner.C').write_text(source)
    probe=dest/'probe.cpp'
    probe.write_text(((VALIDATION_ROOT / 'tests/fixtures/transport_terminal/probe.cpp')).read_text().replace("<<out.bodyPower<<' '",'').replace('&&out.bodyPower==normal.bodyPower',''))
    cmd=['g++','-std=c++14','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-pedantic','-Wno-misleading-indentation','-Wno-mismatched-new-delete','-DTERMINAL_TEST='+str(int(terminal)),'-I'+str(dest),'-I'+str(app),'-I'+str((VALIDATION_ROOT / 'tests/fixtures/transport_terminal')),str(probe),'-o',str(dest/'probe')]
    b=subprocess.run(cmd,text=True,capture_output=True);(dest/'build.log').write_text(' '.join(cmd)+'\n'+b.stdout+b.stderr);assert b.returncode==0,b.stderr
    r=subprocess.run([str(dest/'probe')],text=True,capture_output=True,timeout=60);(dest/'run.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
    return r.stdout

def test_transport_original_fields_costs_and_terminal(tmp_path):
    baseline=tmp_path/'baseline';baseline.mkdir()
    archive=subprocess.check_output(['git','archive',PIN,'applications/CHMT','common'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(baseline,filter='data')
    old=build_run(baseline,tmp_path/'old',False);current=build_run(ROOT,tmp_path/'current',False)
    assert old==current;assert old.count('COUNT ')==6
    assert 'trial48 and missing input passed' in build_run(ROOT,tmp_path/'terminal',True)

def test_actual_owner_signed_film_geometry_proposal(tmp_path):
    app=ROOT/'applications/CHMT';text=(app/'materials/CpuMaterialDriver.C').read_text()
    begin=text.index('                std::vector<FilmAux> aux=evaluation.filmAux;')
    end=text.index('                if(p.meshMotion.policy==MeshMotionPolicy::CoupledRecession',begin)
    actual=text[begin:end]
    source='''#include "materials/MaterialTransport.H"
#include "materials/CpuMaterialDriver.H"
#include "film/CpuFilmDriver.H"
#include "ablation/CpuSurfaceInterface.H"
#include <cassert>
#include <iostream>
using namespace chmt;
namespace chmt {
bool proposal(bool terminalMaterial,std::vector<FilmAux>& output,MaterialFailureOrigin& lastFailure,std::string& error){
 PhysicsConfig p;p.enableFilm=true;p.liquid.rho=1000;const Real dt=.01;
 HostState state;state.surface.gasFace={0};state.surface.solidFace={0};state.surface.area={1};state.gasMesh.areaVectors={{1,0,0}};state.solidMesh.areaVectors={{1,0,0}};
 state.film.resize(1);state.film[0].mass=1;state.filmAux.resize(1);state.filmAux[0].solidFront=2;state.filmAux[0].gasFront=3;
 HostState evaluation=state;std::vector<FilmQ> filmRate(1);filmRate[0].mass=-300;
 std::vector<CpuFilmForcing> forcing(1);std::vector<Real> sideVolume{.3},target{-.02};CpuSurfaceResult surface;surface.physics.resize(1);surface.physics[0].wet=true;
 do {
'''+actual+'''
 output=std::move(aux);return true;
 }while(false);
 return false;
}
}
int main(){std::string error;std::vector<FilmAux> output;MaterialFailureOrigin origin=MaterialFailureOrigin::None;
 assert(!proposal(false,output,origin,error)&&origin==MaterialFailureOrigin::GeometryNumerical);
 error.clear();assert(proposal(true,output,origin,error));assert(output.size()==1);
 assert(output[0].thickness==-2./1000&&output[0].area==1);
 const Real solid=-.02/(1*.01),top=(-.02/.01+-3./(.01*1000)-.3)/1;
 assert(output[0].solidNormalVelocity==solid&&output[0].normalVelocity==top);
 assert(output[0].solidFront==2+.01*solid&&output[0].gasFront==3+.01*top);
 std::cout<<"actual owner signed-mass thickness/front/velocity tail passed\\n";
}
'''
    probe=tmp_path/'proposal.cpp';probe.write_text(source)
    cmd=['g++','-std=c++14','-O2','-ffp-contract=off','-Wall','-Wextra','-Werror','-pedantic','-Wno-misleading-indentation','-I'+str(app),'-I'+str(ROOT/'common'),str(probe),str(app/'materials/MaterialTransport.C'),'-o',str(tmp_path/'proposal')]
    b=subprocess.run(cmd,text=True,capture_output=True);(tmp_path/'build.log').write_text(' '.join(cmd)+'\n'+b.stdout+b.stderr);assert b.returncode==0,b.stderr
    r=subprocess.run([str(tmp_path/'proposal')],text=True,capture_output=True);(tmp_path/'run.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
