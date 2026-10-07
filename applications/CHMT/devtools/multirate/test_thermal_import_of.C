// Real production field importer + native implicit matrix. No gas/device emulation.
#include "io/MeshIO.H"
#include "gas/BoundarySample.H"
#include "gas/SstMath.H"
#include "materials/CpuMaterialDriver.H"
#include "tests/TestSupport.H"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <unistd.h>
namespace {
using namespace chmt;
void scalarInput(const Foam::fvMesh& mesh,const std::string& name,const std::string& dimensions,
    Real internal,const std::string& type,Real boundary) {
 std::ofstream out(std::string(mesh.time().timePath())+"/"+name);
 out<<std::setprecision(17)<<"FoamFile { version 2.0; format ascii; class volScalarField; object "<<name<<"; }\n"
    <<"dimensions "<<dimensions<<";\ninternalField uniform "<<internal<<";\nboundaryField {\n";
 for(const auto& patch:mesh.boundaryMesh()) {
  out<<patch.name()<<" { type "<<type<<"; value uniform "<<boundary<<";";
  if(type=="fixedGradient")out<<" gradient uniform 1;";
  out<<" }\n";
 }
 out<<"}\n";chmt_test::check(bool(out),"write native scalar fixture");
}
ModelConfig model(){ModelConfig m;m.physics=chmt_test::physics();m.physics.modelFingerprint=1;m.physics.spatialOrder=1;m.physics.maxDt=10000;m.physics.enableGas=false;
 for(int c=0;c<Nc;++c){m.condensedNames[c]="C"+std::to_string(c);auto& t=m.physics.condensed[c];t.rho=1000;t.cp0=1000;t.cp1=0;t.e0=-1e8;t.conductivity=1000;t.Tmin=100;t.Tmax=2000;}return m;}
void solidInputs(const Foam::fvMesh& mesh,const ModelConfig& m){
 scalarInput(mesh,"solidEnergyDensity","[1 -1 -2 0 0 0 0]",1000*condensedE(m.physics.condensed[0],400),"zeroGradient",0);
 scalarInput(mesh,"porosity","[0 0 0 0 0 0 0]",0,"zeroGradient",0);
 for(int c=0;c<Nc;++c)scalarInput(mesh,"rho_"+m.condensedNames[c],"[1 -3 0 0 0 0 0]",c?0:1000,"zeroGradient",0);
 for(int s=0;s<Ns;++s)scalarInput(mesh,"rhoPore_"+m.speciesNames[s],"[1 -3 0 0 0 0 0]",0,"zeroGradient",0);
}
IntervalHistory history(){IntervalHistory h;CouplingInterval interval;interval.identity.sequence=1;interval.end=10000;std::string error;
 chmt_test::check(h.begin(interval,error),error);GasIntervalRecord r;r.begin=0;r.end=10000;chmt_test::check(h.appendAccepted(r,error),error);return h;}
void solidCase(Foam::fvMesh& mesh,const ModelConfig& m,bool fixed) {
 HostState base;base.solidMesh=io::importMesh(mesh,Foam::dictionary{});io::importSolidFields(mesh,m,base);
 HostState endpoint=base;endpoint.time=10000;CpuMaterialDriver driver(m,&mesh);CpuMaterialControls controls;HostState result;WallProgram wall;CpuMaterialReport report;std::string error;
 chmt_test::check(driver.advanceCandidate(base,endpoint,history(),controls,result,wall,report,error),error);
 // 2x2x2 cube: each corner has three .25 m2 boundary faces, distance .25 m.
 const Real ratio=10000*3*1000*.25/.25/(125*1000),exact=fixed?(400+ratio*650)/(1+ratio):400;
 long double change=0;
 for(std::size_t c=0;c<result.solid.size();++c){Real T=0;chmt_test::check(recoverSolid(result.solid[c],m.physics,T),"native solid EOS");
  chmt_test::near(T,exact,"production-import fixed/zeroGradient backward-Euler temperature",1e-10,1e-7);change+=125000.L*(T-400);}
 chmt_test::check(report.linearSolves>0,"real sparse solve executed");
 chmt_test::check(std::abs(change+report.budgetDelta.boundaryEnergy)<1e-3L,"independent signed boundary-energy closure");
 chmt_test::check(fixed?report.budgetDelta.boundaryEnergy<0:report.budgetDelta.boundaryEnergy==0,"thermal boundary heat sign");
 std::cout<<"NATIVE_OF10 production solid import "<<(fixed?"fixedValue heating":"zeroGradient insulation")<<" passed; linear solves="<<report.linearSolves<<'\n';
}
void gasInputs(const Foam::fvMesh& mesh,const ModelConfig& m,const std::string& type){
 scalarInput(mesh,"rho","[1 -3 0 0 0 0 0]",1.3,"zeroGradient",1.3);
 scalarInput(mesh,"T","[0 0 0 1 0 0 0]",730,type,650);
 for(int s=0;s<Ns;++s)scalarInput(mesh,"Y_"+m.speciesNames[s],"[0 0 0 0 0 0 0]",s?0:1,"zeroGradient",s?0:1);
 scalarInput(mesh,"k","[0 2 -2 0 0 0 0]",.01,"fixedValue",0);
 scalarInput(mesh,"omega","[0 0 -1 0 0 0 0]",1,"fixedValue",1);
 std::ofstream out(std::string(mesh.time().timePath())+"/U");out<<"FoamFile { version 2.0; format ascii; class volVectorField; object U; }\ndimensions [0 1 -1 0 0 0 0]; internalField uniform (0 0 0); boundaryField {\n";
 for(const auto& patch:mesh.boundaryMesh())out<<patch.name()<<" { type fixedValue; value uniform (0 0 0); }\n";out<<"}\n";
}
void gasCase(Foam::fvMesh& mesh,ModelConfig m,bool fixed){
 m.physics.enableGas=true;m.physics.enableSst=true;m.physics.gasViscosity=.001;m.physics.gasConductivity=5;
 gasInputs(mesh,m,fixed?"fixedValue":"zeroGradient");HostState state;state.gasMesh=io::importMesh(mesh,Foam::dictionary{});io::importGasFields(mesh,m,state);
 const int face=mesh.boundaryMesh()[0].start(),cell=state.gasMesh.owner[face];GasPrimitive owner;
 chmt_test::check(recoverGas(state.gas[cell],state.gasMesh.volumes[cell],m.physics,owner),"import gas EOS");
 owner.temperature=900;GasPrimitive updated;chmt_test::check(recoverGas(conservativeGas(owner,1,m.physics),1,m.physics,updated),"evolved gas owner EOS");
 const auto& prescribed=state.gasMesh.boundaryPrimitive[face];chmt_test::check(prescribed.temperature>0,"import retains positive wall primitive for SST");
 BoundaryKind kind=BoundaryKind::NoSlip;ThermalBoundaryKind policy=state.gasMesh.thermalBoundary[face];GasView gas;gas.primitive=&updated;
 Vec3 area{1,0,0};Real swept=0;GeometryView geometry;geometry.boundaryKind=&kind;geometry.thermalBoundary=&policy;geometry.boundaryPrimitive=&prescribed;geometry.areaVector=&area;geometry.sweptVolume=&swept;
 chmt_test::near(boundarySample(0,0,gas,geometry,1).temperature,fixed?400:900,"actual T-patch policy controls evolving wall sample");
 GasPrimitive trace;chmt_test::check(wallThermodynamicTrace(updated,prescribed,fixed,m.physics,trace),"native imported endpoint EOS trace");
 chmt_test::near(trace.temperature,fixed?650:900,"SST trace uses current zeroGradient or fixedValue");
 Real omega;chmt_test::check(wallOmegaTarget(trace,.25,m.physics,omega),"native imported SST trace remains valid");
 GasGradient gradient;gradient.temperature={(650.-900)/.25,0,0};
 chmt_test::near(impermeableWallDiffusiveFlux(updated,prescribed,{},gradient,area,fixed,true,m.physics,.1).energy,fixed?5000:0,"actual imported policy controls independent heat flux");
 std::cout<<"NATIVE_OF10 gas T import "<<(fixed?"fixedValue":"zeroGradient")<<" evolving owner/SST host boundary math passed (CUDA not run)\n";
}
}
int main(int argc,char** argv){
 Foam::argList::noParallel();Foam::argList args(argc,argv);if(!args.checkRootCase())return 2;
 Foam::Time time(Foam::Time::controlDictName,args);Foam::fvMesh mesh(Foam::IOobject(Foam::polyMesh::defaultRegion,time.timeName(),time,Foam::IOobject::MUST_READ));
 chmt_test::check(mesh.nCells()==8,"native 2x2x2 fixture");Foam::mkDir(time.timePath());auto m=model();solidInputs(mesh,m);
 solidCase(mesh,m,false); // Optional solidTemperature absent: backward compatible.
 scalarInput(mesh,"solidTemperature","[0 0 0 1 0 0 0]",1234,"fixedValue",650);solidCase(mesh,m,true);
 scalarInput(mesh,"solidTemperature","[0 0 0 1 0 0 0]",1234,"zeroGradient",650);solidCase(mesh,m,false);
 auto rejectSolid=[&](const std::string& expected){HostState state;state.solidMesh=io::importMesh(mesh,Foam::dictionary{});bool refused=false;
  try{io::importSolidFields(mesh,m,state);}catch(const std::runtime_error& error){refused=std::string(error.what()).find(expected)!=std::string::npos;}chmt_test::check(refused,"native solid importer rejects "+expected);};
 scalarInput(mesh,"solidTemperature","[0 0 0 1 0 0 0]",400,"fixedGradient",650);rejectSolid("unsupported thermal patch operator");
 scalarInput(mesh,"solidTemperature","[0 0 0 0 0 0 0]",400,"fixedValue",650);rejectSolid("dimensions mismatch");
 scalarInput(mesh,"solidTemperature","[0 0 0 1 0 0 0]",400,"fixedValue",2500);rejectSolid("caloric range");
 scalarInput(mesh,"solidEnergyDensity","[1 -1 -2 0 0 0 0]",1000*condensedE(m.physics.condensed[0],400),"fixedValue",650);rejectSolid("solidEnergyDensity boundary operator");
 gasCase(mesh,m,false);gasCase(mesh,m,true);
 gasInputs(mesh,m,"fixedGradient");HostState state;state.gasMesh=io::importMesh(mesh,Foam::dictionary{});bool refused=false;
 try{io::importGasFields(mesh,m,state);}catch(const std::runtime_error& error){refused=std::string(error.what()).find("unsupported thermal patch operator")!=std::string::npos;}
 chmt_test::check(refused,"native gas importer rejects unsupported thermal patch operator");
 std::cout<<"NATIVE_OF10 thermal import negative cases passed\n";
}
