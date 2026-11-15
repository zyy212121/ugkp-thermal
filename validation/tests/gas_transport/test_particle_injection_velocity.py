pass
import ast
import subprocess
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT

def injection_fixture():
    tree=ast.parse((VALIDATION_ROOT/"tests/gas_transport/test_particle_thermal_carry_lifecycle.py").read_text())
    function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=="test_actual_injection_initializes_reused_slot_without_other_solver_storage")
    assignment=next(x for x in function.body if isinstance(x,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="code" for t in x.targets))
    code=ast.literal_eval(assignment.value).split("int main(){",1)[0]
    code=code.replace('#include "GpuFrozenParticleRelease.cuh"', "")
    code=code.replace("struct DeviceState {", '#include "gasBoundary/Definition.H"\nstruct DeviceState {\n ugkp::boundary::View gasBoundaryProgram;\n double Ux[1]={0},Uy[1]={.4},Uz[1]={0},epsS[1]={.2};')
    code=code.replace("blockIdx, threadIdx, blockDim;", "blockIdx, threadIdx, blockDim, gridDim;")
    return code

def test_actual_injection_velocity_and_mass(tmp_path):
    code=injection_fixture()+r"""
int main(){
 using namespace ugkp::boundary;
 blockDim.x=1;
 // Independent expected values: fixed, interpolated table, gas-stage mirror (not post-drag owner).
 for(int mode=0;mode<3;++mode)for(int overrideMode=0;overrideMode<4;++overrideMode) {
  DeviceState s;Face face;Function functions[2];Term terms[4]={{0,.1},{1,.2},{0,.2},{1,.4}};
  face.solidMode=mode==2?0:1;face.uMode=mode==2?0:1;
  face.value[SolidFraction]=mode==2?.99:.1;face.value[Uy]=mode==2?9:.2;
  if(mode==1){
   functions[0]={FunctionKind::Table,Bounds::Clamp,0,2,0};
   functions[1]={FunctionKind::Table,Bounds::Clamp,2,2,0};
   face.function[SolidFraction]=0;face.function[Uy]=1;
  }
  s.gasBoundaryProgram={&face,functions,terms,1};
  s.particleInjectionVelocitySpecified=overrideMode!=0;
  s.particleInjectionVelocity[0]=overrideMode? .03:0;
  s.particleInjectionVelocity[1]=overrideMode==1?.1:overrideMode==3?-.1:0;
  s.particleInjectionVelocity[2]=overrideMode?-.02:0;
  s.gasBoundaryUy[0]=.4;s.Uy[0]=7;
  s.sourceTheta[0]=mode==2?-1:.03;s.theta[0]=.07;
  s.rhoSolid=2800;s.Sfy[0]=-1;s.sourceUy[0]=9;s.sourceMassRate[0]=999;
  const double fraction=mode==0?.1:mode==1?.15:.2;
  const double gasVy=mode==0?.2:mode==1?.3:.4;
  const double vy=overrideMode?s.particleInjectionVelocity[1]:gasVy;
  const double rate=fraction*2800*std::max(0.,vy);
  const double dt=rate>0?1.5/rate:1.;
  injectBoundaryParticlesKernel(&s,dt,.5);
  if(rate==0){assert(s.n==0&&s.sourceResidualMass[0]==0);continue;}
  assert(s.n==1&&s.pStatus[0]==1);
  assert(std::abs(s.puy[0]-vy)<1e-15);
  assert(std::abs(s.pux[0]-(overrideMode?.03:0))<1e-15);
  assert(std::abs(s.puz[0]-(overrideMode?-.02:0))<1e-15);
  assert(std::abs(s.pm[0]+s.sourceResidualMass[0]-rate*dt)<1e-14);
  assert(std::abs(s.pm[0]*s.puy[0]-vy)<1e-15);
  assert(s.pT[0]==300);
  assert(s.pTheta[0]==(mode==2?.07:.03));
 }
}
"""
    (tmp_path/"cuda_runtime.h").write_text("// CPU CUDA stand-in; qualifiers and dimensions are supplied by fixture.\n")
    cpp=tmp_path/"injection.cpp";exe=tmp_path/"injection";cpp.write_text(code)
    subprocess.run(["g++","-std=c++17","-O2","-I"+str(tmp_path),"-I"+str(SOURCE_ROOT/"common"),str(cpp),"-o",str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

def test_physical_open_patch_source_enumeration(tmp_path):
    pass
    import re
    for app in ("CHT", "gasUGKP"):
        header=(SOURCE_ROOT/f"applications/{app}/gpu/GpuResidentStrict.H").read_text()
        body=header.split("void uploadBoundarySources",1)[1]
        predicate=body.split("if (",1)[1].split("\n            {",1)[0]
        assert predicate.endswith(")")
        predicate=predicate[:-1]
        predicate=re.sub(r"//[^\n]*", "", predicate)
        predicate=predicate.replace("mesh.boundaryMesh()[patchI]", "patch")
        code=r"""
#include <cassert>
struct Patch {int kind; bool coupled()const{return kind==6;}};
struct wallPolyPatch{};struct emptyPolyPatch{};struct wedgePolyPatch{};
struct symmetryPolyPatch{};struct symmetryPlanePolyPatch{};
template<class T> bool isA(const Patch&);
template<> bool isA<wallPolyPatch>(const Patch&p){return p.kind==1;}
template<> bool isA<emptyPolyPatch>(const Patch&p){return p.kind==2;}
template<> bool isA<wedgePolyPatch>(const Patch&p){return p.kind==3;}
template<> bool isA<symmetryPolyPatch>(const Patch&p){return p.kind==4;}
template<> bool isA<symmetryPlanePolyPatch>(const Patch&p){return p.kind==5;}
bool excluded(const Patch&patch){return PREDICATE;}
int main(){
 for(int kind=0;kind<=6;++kind){
  Patch patch{kind};int sources=0;
  // Enrollment is geometric, not based on startup velocity or epsilonS.
  for(int startupFlow=0;startupFlow<2;++startupFlow)
   if(!excluded(patch))++sources;
  assert(sources==(kind==0?2:0));
 }
}
""".replace("PREDICATE",predicate)
        cpp=tmp_path/(app+".cpp");exe=tmp_path/app;cpp.write_text(code)
        subprocess.run(["g++","-std=c++17",str(cpp),"-o",str(exe)],check=True)
        subprocess.run([str(exe)],check=True)
