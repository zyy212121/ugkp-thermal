"""Film reconstruction evaluates once; forcing conservation is warning-only."""
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[2]
APP=ROOT/'applications/CHMT'
def test_film_single_pass_and_warning(tmp_path):
    source=(APP/'film/CpuFilmDriver.C').read_text()
    assert 'scale*=.5' not in source
    assert 'completeCpuFilm' not in source
    assert 'hasExactEvent' not in source
    source=source.replace('namespace chmt { namespace {','namespace chmt { int recoveryCalls=0; namespace {',1)
    source=source.replace('if(!recoverFilm(q,1,e.aux[f].pressure,p,aux))','++recoveryCalls; if(!recoverFilm(q,1,e.aux[f].pressure,p,aux))')
    owner=tmp_path/'film.C';owner.write_text(source)
    probe=tmp_path/'probe.cpp';probe.write_text('#include "'+str(owner)+'"\n'+r'''
#include <cassert>
using namespace chmt;
int main(){
 ugkp::diagnostics::DeviceContext ctx{};PhysicsConfig p;p.diagnostics=&ctx;
 p.enableFilm=true;p.spatialOrder=2;p.reconstruction=ReconstructionMode::SmoothVerification;
 p.liquid.rho=1000;p.liquid.cp0=1000;p.liquid.Tmin=100;p.liquid.Tmax=1000;p.liquidViscosity=1;
 HostState h;h.surface.area={1,1};h.surface.normal={{0,0,1},{0,0,1}};h.film.resize(2);h.filmAux.resize(2);
 h.surface.edgeOwner={0};h.surface.edgeNeighbour={1};h.surface.edgeLength={1};h.surface.edgeConormal={{1,0,0}};
 h.surface.edgeOwnerOffset={{.5,0,0}};h.surface.edgeNeighbourOffset={{-.5,0,0}};
 std::vector<CpuFilmForcing> f(2);std::vector<CpuFilmDrive>d(2);for(int i=0;i<2;++i){d[i].pressure=0;h.filmAux[i].pressure=0;}f[0].species[0]=.1;
 std::string error;bool numerical=false;assert(validate(h,p,.01,f,d,error,&numerical));
 assert(ctx.warnings[0].count==1&&!ugkp::diagnostics::hasFatal(&ctx));
 Evaluation e;e.q.resize(2);e.aux.resize(2);e.incidentOffsets={0,1,2};e.incident={0,0};
 for(int i=0;i<2;++i){e.aux[i].pressure=0;e.q[i].mass=1;e.q[i].species[0]=1;e.q[i].enthalpy=1000*(300+100*i);}
 FilmQ out;out.mass=77;
 recoveryCalls=0;assert(!reconstruct(0,{-10,0,0},e,h.surface,p,out));assert(recoveryCalls==1&&out.mass==77);
 recoveryCalls=0;assert(reconstruct(0,{.1,0,0},e,h.surface,p,out));assert(recoveryCalls==1&&out.mass==1);
 assert(out.enthalpy==310000);
}
''')
    exe=tmp_path/'probe'
    subprocess.run(['g++','-std=c++17','-Wall','-Wextra','-Werror','-I'+str(APP),'-I'+str(ROOT/'common'),str(probe),'-o',str(exe)],check=True,capture_output=True,text=True)
    subprocess.run([str(exe)],check=True,capture_output=True,text=True)
