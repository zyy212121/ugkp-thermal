"""Exercise actual shared wall operations and count thermodynamic evaluator work."""
import shutil
import subprocess
from validation.tools.paths import SOURCE_ROOT

def test_point_uses_one_species_evaluation_and_shared_velocity_factorization(tmp_path):
    shutil.copytree(SOURCE_ROOT/'common',tmp_path/'common')
    h=tmp_path/'common/gasTransport/MixtureThermo.H'
    text=h.read_text(); start=text.index('inline ThermoStatus evaluateSpecies')
    body=text.index('    using namespace ugkp::diagnostics;',start)
    text=text[:body]+'    ++::speciesEvaluations;\n'+text[body:];h.write_text(text)
    cpp=tmp_path/'work.cpp'
    cpp.write_text(r"""
long long speciesEvaluations=0;
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include <cassert>
#include <cstring>
#include <iostream>
int main(){
 using namespace ugkwp::gaswall; using namespace ugkwp::gaswall::detail;
 CapturedSstWall m;LayerContext<double,2> ctx;assert(makeContext(m.input,m.config,ctx));
 double x[6]={1,2,100,.25,.5,.1};LayerPoint<double,2> p;
 speciesEvaluations=0;assert(pointState(m.input,m.config,ctx,x,p));
 std::cout<<"point_species_evaluations="<<speciesEvaluations<<std::endl;
 assert(speciesEvaluations==2);
 // Vary both right-hand sides, advection, mesh spacing and positive diffusion.
 for(int n: {3,7,24})for(int trial=0;trial<41;++trial){
   auto c=m.config;c.nodes=n;WallWorkspace<double,2,24> a{},b{};
   for(int i=0;i<n;++i){a.y[i]=b.y[i]=.001*i*(i+1);a.diagonal[i][0][0]=b.diagonal[i][0][0]=.00001*(1+(i+trial)%7);}
   double mass=trial*.00003,u0=trial*.73-4,u1=trial*-.31+7;
   assert(seedScalar(c,a,0,0,-1,-1,mass,0.,u0));
   assert(seedScalar(c,a,1,0,-1,-1,mass,0.,u1));
   assert(seedVelocityPair(c,b,mass,u0,u1));
   for(int i=0;i<n;++i)for(int d=0;d<2;++d)assert(std::memcmp(&a.delta[i][d],&b.delta[i][d],sizeof(double))==0);
   b.diagonal[1][0][0]=-100; a=b;
   bool old=seedScalar(c,a,0,0,-1,-1,mass,0.,u0)&&seedScalar(c,a,1,0,-1,-1,mass,0.,u1);
   assert(seedVelocityPair(c,b,mass,u0,u1)==old);
 }
}
""")
    exe=tmp_path/'work'
    b=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(tmp_path),'-I'+str(SOURCE_ROOT),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert b.returncode==0,b.stdout+b.stderr
    q=subprocess.run([str(exe)],capture_output=True,text=True)
    assert q.returncode==0,q.stdout+q.stderr

def test_seed_reuses_only_invariant_thermodynamics(tmp_path):
    shutil.copytree(SOURCE_ROOT/'common',tmp_path/'common')
    h=tmp_path/'common/gasWall/ReactingWallLayer.H';text=h.read_text()
    start=text.index(' bool pointState(');body=text.index('){',start)+2
    text=text[:body]+'if(::insideSeed)++::seedPoints;'+text[body:]
    start=text.index(' bool seedSstLayer(');body=text.index('){',start)+2
    text=text[:body]+'::insideSeed=true;'+text[body:]
    end=text.index('template<class R,int N,int M,class TableReal>',body)
    tail=text.rfind('    return true;',body,end)
    text=text[:tail]+'    ::insideSeed=false;'+text[tail:];h.write_text(text)
    cpp=tmp_path/'seed.cpp';cpp.write_text(r"""
bool insideSeed=false;long long seedPoints=0;
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include <cassert>
#include <iostream>
int main(){
 using namespace ugkwp::gaswall;
 CapturedSstWall m;WallWorkspace<double,2,24> w;WallOutput<double,2> o;WallStatus status;
 assert(evaluateReactingWallLayer(m.input,m.config,w,o,status));
 std::cout<<"seed_point_evaluations="<<seedPoints<<std::endl;
 assert(o.iterations>0&&o.iterations<m.config.maxIterations);assert(seedPoints<=m.config.nodes+3);
}
""")
    exe=tmp_path/'seed'
    q=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(tmp_path),'-I'+str(SOURCE_ROOT),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert q.returncode==0,q.stdout+q.stderr
    q=subprocess.run([str(exe)],capture_output=True,text=True)
    assert q.returncode==0,q.stdout+q.stderr

def test_trial_residual_shares_intervals_without_changing_node_sources(tmp_path):
    shutil.copytree(SOURCE_ROOT/'common',tmp_path/'common')
    h=tmp_path/'common/gasWall/ReactingWallLayer.H';text=h.read_text()
    start=text.index(' bool intervalFlux(');body=text.index('){',start)+2
    text=text[:body]+'++::intervalCalls;'+text[body:];h.write_text(text)
    cpp=tmp_path/'flux.cpp';cpp.write_text(r"""
long long intervalCalls=0;
#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include <cassert>
#include <cstring>
#include <iostream>
int main(){
 using namespace ugkwp::gaswall;using namespace ugkwp::gaswall::detail;
 CapturedSstWall m;WallWorkspace<double,2,24> w;WallOutput<double,2> o;WallStatus status;
 assert(evaluateReactingWallLayer(m.input,m.config,w,o,status));
 LayerContext<double,2> ctx;assert(makeContext(m.input,m.config,ctx));
 double original[24][6]{},shared[24][6]{},n0=0,n1=0;
 intervalCalls=0;
 assert(allResidual(m.input,m.config,ctx,w.state,w.y,original,n0));
 assert(intervalCalls==45);intervalCalls=0;
 assert(allResidual(m.input,m.config,ctx,w.state,w.y,shared,n1,SerialWallTeam{},static_cast<const WallInput<double,2>*>(nullptr),nullptr,true,w.upper));
 std::cout<<"shared_interval_evaluations="<<intervalCalls<<std::endl;
 assert(intervalCalls==23);assert(n0==n1);
 assert(std::memcmp(original,shared,sizeof(original))==0);
 // Live diagnostics preserve the uncached row context and warning multiplicity.
 w.state[6][3]=-.1;
 ugkp::diagnostics::DeviceContext d0{},d1{};
 m.input.diagnostics=&d0;intervalCalls=0;
 bool ok0=allResidual(m.input,m.config,ctx,w.state,w.y,original,n0);
 auto count0=intervalCalls;m.input.diagnostics=&d1;intervalCalls=0;
 bool ok1=allResidual(m.input,m.config,ctx,w.state,w.y,shared,n1,SerialWallTeam{},static_cast<const WallInput<double,2>*>(nullptr),nullptr,true,w.upper);
 assert(ok0==ok1&&intervalCalls==count0);
 assert(std::memcmp(&d0,&d1,sizeof(d0))==0);
 assert(std::memcmp(original,shared,sizeof(original))==0);
 // An incomplete H channel must still pass through the owner's storage check.
 m.input.model.thermo.coefficientCount=5;
 auto trial=m.input;trial.diagnostics=nullptr;
 bool available=true;m.input.diagnostics=&d0;d0=ugkp::diagnostics::DeviceContext{};
 assert(!allResidual(trial,m.config,ctx,w.state,w.y,shared,n1,SerialWallTeam{},&m.input,&available,true,w.upper));
 assert(!available&&d0.fatalClaim);

}
""")
    exe=tmp_path/'flux'
    q=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-I'+str(tmp_path),'-I'+str(SOURCE_ROOT),str(cpp),'-o',str(exe)],capture_output=True,text=True)
    assert q.returncode==0,q.stdout+q.stderr
    q=subprocess.run([str(exe)],capture_output=True,text=True)
    assert q.returncode==0,q.stdout+q.stderr
