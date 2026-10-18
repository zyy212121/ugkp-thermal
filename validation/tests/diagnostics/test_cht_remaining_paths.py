"""Execute production gas gravity and CHT auxiliary output/property bodies on host."""
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def run_cpp(tmp_path, source):
    cpp = tmp_path / 'probe.cpp'
    cpp.write_text(source)
    exe = tmp_path / 'probe'
    p = subprocess.run(['g++', '-std=c++17', '-I'+str(ROOT/'common'), '-I'+str(tmp_path), str(cpp), '-o', str(exe)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout+p.stderr
    p = subprocess.run([str(exe)], capture_output=True, text=True)
    assert p.returncode == 0, p.stdout+p.stderr


def test_resident_gravity_including_zero_particle_bridge(tmp_path):
    run_cpp(tmp_path, r'''
#include <cmath>
#include <cassert>
#include <limits>
#include "Diagnostics.H"
#define __device__
#define __global__
#define __forceinline__ inline
#define GPU_OPERATOR_REAL double
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) (x)
struct Id { int x=0; } blockIdx, threadIdx, blockDim;
struct DeviceState {
  int nCells=1, nParticles=0;
  double rho[1]={2},rhoUx[1]={3},rhoUy[1]={4},rhoUz[1]={5},rhoE[1]={20};
  double rhoMin=.1,gravityX=1,gravityY=2,gravityZ=3;
  ugkp::diagnostics::DeviceContext* diagnostics;
};
double sqr3(double x,double y,double z){return x*x+y*y+z*z;}
#include "operators/clampMin.cuh"
#include "operators/applyGasGravityKernel.cuh"
int main(){
 using namespace ugkp::diagnostics;
 DeviceContext d{};DeviceState s;s.diagnostics=&d;
 applyGasGravityKernel(&s,.5);assert(!hasFatal(&d));
 assert(s.rhoUx[0]==4&&s.rhoUy[0]==6&&s.rhoUz[0]==8&&s.rhoE[0]==36.5);
 for(int particles: {0,7})for(int component=0;component<9;++component){
  s=DeviceState{};s.diagnostics=&d;s.nParticles=particles;reset(d,1,2,3,4);
  double dt=.5;double* inputs[]={s.rho,s.rhoUx,s.rhoUy,s.rhoUz,s.rhoE,&s.gravityX,&s.gravityY,&s.gravityZ,&dt};
  *inputs[component]=std::numeric_limits<double>::quiet_NaN();
  applyGasGravityKernel(&s,dt);
  assert(hasFatal(&d)&&d.firstFatal.sourceSiteId==30500&&d.firstFatal.component==component);
  assert(std::isnan(d.firstFatal.observed)&&d.firstFatal.entityIndex==0);
  assert(component==4?std::isnan(s.rhoE[0]):s.rhoE[0]==20);
  assert(component==1?std::isnan(s.rhoUx[0]):s.rhoUx[0]==3);
 }
 for(double energy:{-1.1e300,1.1e300}){
  reset(d,1,2,3,4);s=DeviceState{};s.diagnostics=&d;s.rhoE[0]=energy;
  applyGasGravityKernel(&s,0);assert(!hasFatal(&d)&&s.rhoE[0]==energy);
 }
 reset(d,1,2,3,4);s=DeviceState{};s.diagnostics=&d;s.gravityX=std::numeric_limits<double>::max();
 applyGasGravityKernel(&s,2);assert(hasFatal(&d)&&d.firstFatal.sourceSiteId==30501);
 assert(s.rhoUx[0]==3&&s.rhoE[0]==20);
 reset(d,1,2,3,4);s=DeviceState{};s.diagnostics=&d;s.rho[0]=.01;
 applyGasOnlyGravityKernel(&s,1);assert(!hasFatal(&d)&&s.rhoUx[0]==3.1);
}
'''.replace('#include <limits>', '#include <limits>\n#include <initializer_list>'))


def test_wall_workspace_has_no_free_memory_policy_cap(tmp_path):
    run_cpp(tmp_path, r'''
#include "gasTransport/GasBoundaryLayerWorkspace.H"
#include <cassert>
int main(){using namespace ugkwp;
 const std::size_t bytes=gasBoundaryLayerWorkspaceBytes<double,2>(24);
 auto a=resolveBoundaryLayerWorkspace(10000,0,200,0,bytes);
 auto b=resolveBoundaryLayerWorkspace(10000,0,200,std::size_t(1)<<40,bytes);
 assert(a.slots==6400&&a.slots==b.slots&&a.bytes==6400*bytes);
 assert(resolveBoundaryLayerWorkspace(10000,5000,200,0,bytes).slots==5000);
 assert(!resolveBoundaryLayerWorkspace(12,12,80,0,std::numeric_limits<std::size_t>::max()).slots);
 assert(!gasBoundaryLayerWorkspaceCapacity(129));
}
''')


def test_solid_extrapolation_warns_but_keeps_finite_value(tmp_path):
    production = (ROOT/'applications/CHT/thermal/GpuSolidThermalProperties.C').read_text()
    evaluate = production.split('scalar GpuSolidThermalProperties::evaluate', 1)[1].split('scalarField GpuSolidThermalProperties::buildRhoCpKnotUnion', 1)[0]
    (tmp_path/'error.H').write_text('')
    source = r'''
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>
#include <sstream>
#include <cassert>
#include <limits>
#include "THERMAL_DIAGNOSTICS"
using Foam::gpuThermal::hostDiagnosticWarningsEnabled;
using scalar=double;using label=int;
struct Values:std::vector<double>{using std::vector<double>::vector;double first()const{return front();}double last()const{return back();}int size()const{return int(std::vector<double>::size());}};
struct PropertyModel{std::string name="kappa";bool constant=false,clampOutOfBounds=false;double constantValue=1;Values temperatures{300,400},values{10,20};};
bool finiteScalar(double v){return std::isfinite(v);}
std::ostringstream warnings;
#define WarningInFunction warnings
struct Fatal{};struct FatalStream{template<class T>FatalStream& operator<<(const T&){return *this;}} fatal;
FatalStream& operator<<(FatalStream&,Fatal){throw Fatal{};}
#define FatalErrorInFunction fatal
#define FatalError Fatal{}
Fatal exit(Fatal f){return f;}
scalar evaluate EVALUATE
int main(){PropertyModel p;unsetenv("UGKP_DIAGNOSTIC_WARNINGS");
 assert(evaluate(p,450)==25);assert(warnings.str().find("extrapolation")!=std::string::npos);
 warnings.str("");setenv("UGKP_DIAGNOSTIC_WARNINGS","0",1);
 assert(evaluate(p,450)==25&&warnings.str().empty());
 p.clampOutOfBounds=true;assert(evaluate(p,450)==20);
 p.clampOutOfBounds=false;p.values={10,1};bool failed=false;
 try{evaluate(p,450);}catch(Fatal){failed=true;}assert(failed);
 p.constant=true;p.constantValue=std::numeric_limits<double>::quiet_NaN();failed=false;
 try{evaluate(p,350);}catch(Fatal){failed=true;}assert(failed);
}
'''.replace('THERMAL_DIAGNOSTICS',str(ROOT/'applications/CHT/thermal/ThermalHostDiagnostics.H')).replace('EVALUATE',evaluate)
    run_cpp(tmp_path, source)


def test_auxiliary_restart_writes_detect_flush_and_close_failures(tmp_path):
    production=(ROOT/'applications/CHT/gpu/GpuResidentStrict.H').read_text()
    methods=production.split('    void writeEpsGPrevRestartMirror\n',1)[1].split('\nprotected:',1)[0]
    source=r'''
#include <fstream>
#include <vector>
#include <string>
#include <limits>
#include <cassert>
#include <cstdio>
using scalar=double;using label=int;using fileName=std::string;
struct scalarField:std::vector<double>{using std::vector<double>::vector;double* begin(){return data();}};
struct labelList:std::vector<int>{using std::vector<int>::vector;int* begin(){return data();}};
#define forAll(v,i) for(int i=0;i<int(v.size());++i)
struct Time{std::string path;int controlDict()const{return 0;}};
struct IoError{};struct OldError{};struct Stream{template<class T>Stream& operator<<(const T&){return *this;}} stream;
Stream& operator<<(Stream&,IoError){throw IoError{};}
Stream& operator<<(Stream&,OldError){throw OldError{};}
#define FatalErrorInFunction stream
#define FatalIOErrorInFunction(x) stream
#define FatalError OldError{}
#define FatalIOError IoError{}
IoError exit(IoError e){return e;}OldError exit(OldError e){return e;}
int ugkwpGpuResidentStrictDownloadEpsGPrev(void*,double* values){values[0]=.3;return 0;}
int ugkwpGpuResidentStrictDownloadSourceResidualMass(void*,int* n,int capacity,int* face,double* mass){*n=1;if(capacity){face[0]=2;mass[0]=.7;}return 0;}
struct Owner{int hostMirrorCells_=1;void* handle_=nullptr;
 void requireInitialised(const char*)const{}void check(int,const char*)const{}
 static fileName epsGPrevRestartFile(const Time&t){return t.path;}
 static fileName sourceResidualRestartFile(const Time&t){return t.path;}
 void writeEpsGPrevRestartMirror METHODS
};
int main(){Owner o;
 for(bool residual:{false,true}){
  for(const char* path:{"/dev/full","/no/such/ugkp-directory/file"}){
   bool failed=false;try{if(residual)o.writeSourceResidualRestartMirror(Time{path});else o.writeEpsGPrevRestartMirror(Time{path});}catch(IoError){failed=true;}
   assert(failed);
  }
  std::string path="OUTPUT";if(residual)o.writeSourceResidualRestartMirror(Time{path});else o.writeEpsGPrevRestartMirror(Time{path});
  std::ifstream in(path);std::string text((std::istreambuf_iterator<char>(in)),{});assert(!text.empty());std::remove(path.c_str());
 }
}
'''.replace('METHODS',methods).replace('OUTPUT',str(tmp_path/'restart.dat'))
    run_cpp(tmp_path,source)


def test_chmt_workspace_configuration_has_no_4096_slot_gate(tmp_path):
    run_cpp(tmp_path, r'''
#include "CHMT_CONFIGURATION"
#include <cassert>
int main(){chmt::PhysicsConfig p;std::string error;
 p.gasMode=ugkwp::GasMode::MixtureFrozen;p.gasViscosity=.02;p.gasConductivity=1;
 p.wallModel.family=ugkwp::gaswall::WallFamily::BoundaryLayer;
 p.wallModel.model=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
 p.wallWorkspaceSlots=5000;
 assert(chmt::validateWallModelConfig(p,error));
 p.wallModel.nodes=129;assert(!chmt::validateWallModelConfig(p,error));
}
'''.replace('CHMT_CONFIGURATION',str(ROOT/'applications/CHMT/core/Configuration.H')))
