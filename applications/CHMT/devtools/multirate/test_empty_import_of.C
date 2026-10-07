// Import-only native regression: geometric empty faces have zero field entries.
// No 2-D gas/material evolution or CUDA execution is claimed here.
#include "io/MeshIO.H"
#include "tests/TestSupport.H"
#include <fstream>
#include <iomanip>
using namespace chmt;
namespace {
void scalarInput(const Foam::fvMesh& mesh,const std::string& name,const std::string& dimensions,
    Real value,const std::string& wallType="zeroGradient",Real wall=0) {
 std::ofstream out(std::string(mesh.time().timePath())+"/"+name);
 out<<std::setprecision(17)<<"FoamFile { version 2.0; format ascii; class volScalarField; object "<<name<<"; }\n"
    <<"dimensions "<<dimensions<<"; internalField uniform "<<value<<"; boundaryField {\n";
 for(const auto& patch:mesh.boundaryMesh())out<<patch.name()<<" { type "<<(patch.type()=="empty"?"empty":wallType)<<"; value uniform "<<wall<<"; }\n";
 out<<"}\n";chmt_test::check(bool(out),"write empty-patch scalar fixture");
}
}
int main(int argc,char** argv){
 Foam::argList::noParallel();Foam::argList args(argc,argv);if(!args.checkRootCase())return 2;
 Foam::Time time(Foam::Time::controlDictName,args);Foam::fvMesh mesh(Foam::IOobject(Foam::polyMesh::defaultRegion,time.timeName(),time,Foam::IOobject::MUST_READ));
 Foam::mkDir(time.timePath());ModelConfig model;model.physics=chmt_test::physics();model.physics.enableGas=true;model.physics.enableSst=true;
 for(int c=0;c<Nc;++c)model.condensedNames[c]="C"+std::to_string(c);
 scalarInput(mesh,"rho","[1 -3 0 0 0 0 0]",1.3);
 scalarInput(mesh,"T","[0 0 0 1 0 0 0]",730);
 for(int s=0;s<Ns;++s)scalarInput(mesh,"Y_"+model.speciesNames[s],"[0 0 0 0 0 0 0]",s?0:1);
 scalarInput(mesh,"k","[0 2 -2 0 0 0 0]",.01);
 scalarInput(mesh,"omega","[0 0 -1 0 0 0 0]",1);
 {std::ofstream out(std::string(time.timePath())+"/U");out<<"FoamFile { version 2.0; format ascii; class volVectorField; object U; }\ndimensions [0 1 -1 0 0 0 0]; internalField uniform (0 0 0); boundaryField {\n";
  for(const auto& patch:mesh.boundaryMesh())out<<patch.name()<<" { type "<<(patch.type()=="empty"?"empty":"fixedValue")<<"; value uniform (0 0 0); }\n";out<<"}\n";}
 int emptyFaces=0;
 {auto temperature=io::scalarField(mesh,"T",Foam::dimTemperature);
  for(int patch=0;patch<mesh.boundaryMesh().size();++patch)if(mesh.boundaryMesh()[patch].type()=="empty"){
   chmt_test::check(mesh.boundaryMesh()[patch].size()>0&&temperature.boundaryField()[patch].size()==0,"real geometric empty faces have zero field values");
   emptyFaces+=mesh.boundaryMesh()[patch].size();}}
 chmt_test::check(emptyFaces>0,"fixture includes empty patches");
 std::cout<<"Actual empty-patch import: "<<emptyFaces<<" geometric faces, zero stored T/rho/U/Y/k/omega values"<<std::endl;
 HostState state;state.gasMesh=io::importMesh(mesh,Foam::dictionary{});io::importGasFields(mesh,model,state);
 for(std::size_t f=0;f<state.gasMesh.owner.size();++f)if(state.gasMesh.boundaryKind[f]==BoundaryKind::Empty){
  chmt_test::near(state.gasMesh.boundaryPrimitive[f].temperature,730,"unused Empty gas primitive uses valid owner EOS");
  chmt_test::near(state.gasMesh.boundarySst[f].k,.01,"unused Empty SST k uses owner");
  chmt_test::near(state.gasMesh.boundarySst[f].omega,1,"unused Empty SST omega uses owner");
  chmt_test::check(state.gasMesh.thermalBoundary[f]==ThermalBoundaryKind::ZeroGradient,"Empty thermal policy is nonconducting");}
 scalarInput(mesh,"solidEnergyDensity","[1 -1 -2 0 0 0 0]",model.physics.condensed[0].rho*condensedE(model.physics.condensed[0],400));
 scalarInput(mesh,"porosity","[0 0 0 0 0 0 0]",0);
 for(int c=0;c<Nc;++c)scalarInput(mesh,"rho_"+model.condensedNames[c],"[1 -3 0 0 0 0 0]",c?0:model.physics.condensed[0].rho);
 for(int s=0;s<Ns;++s)scalarInput(mesh,"rhoPore_"+model.speciesNames[s],"[1 -3 0 0 0 0 0]",0);
 scalarInput(mesh,"solidTemperature","[0 0 0 1 0 0 0]",1234,"fixedValue",650);
 state.solidMesh=io::importMesh(mesh,Foam::dictionary{});io::importSolidFields(mesh,model,state);
 for(std::size_t f=0;f<state.solidMesh.owner.size();++f)if(state.solidMesh.boundaryKind[f]==BoundaryKind::Empty){
  chmt_test::near(state.solidMesh.boundaryPrimitive[f].temperature,400,"Empty solid thermal import avoids absent patch values");
  chmt_test::check(state.solidMesh.thermalBoundary[f]==ThermalBoundaryKind::ZeroGradient,"Empty solid thermal policy remains nonconducting");}
 std::cout<<"NATIVE_OF10 Empty gas/SST/solid thermal field import passed; no 2-D evolution claim\n";
}
