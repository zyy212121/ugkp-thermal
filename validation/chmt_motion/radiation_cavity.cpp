// Full shared coupler -> native CUDA DO -> material root -> CHT publication.
#include "fvCFD.H"
#include "GpuResidentStrict.H"
#include "GpuParticleRadiationCoupler.H"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
using namespace Foam;using namespace Foam::gpuThermal;
extern "C" void* makeCavityOwner();
extern "C" void resetCavityOwner(void*,double,double,double,double,int);
extern "C" void readCavityOwner(void*,double*);
struct Resident:GpuResidentStrictBase {
 Resident(const fvMesh& mesh){handle_=makeCavityOwner();residentMesh_=&mesh;hostMirrorCells_=1;hostMirrorFaces_=6;residentTpMin_=100;residentTpMax_=5000;}
 void reset(double a,double b,double ca=0,double cb=0,int csr=1){resetCavityOwner(handle_,a,b,ca,cb,csr);}
 void read(double* x){readCavityOwner(handle_,x);}
};
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(int argc,char**argv)try{
 #include "setRootCase.H"
 #include "createTime.H"
 #include "createMesh.H"
 (void)mesh.cells();require(mesh.nCells()==1,"one-cell fixture");
 MieTable table(runTime.path()/"constant/mie.table");
 labelList patches(mesh.boundary().size());forAll(patches,p)patches[p]=p;
 volScalarField wall(IOobject("wallTemperature",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,dimensionedScalar("T",dimTemperature,4500),"fixedValue");
 List<scalarField> emissivity(mesh.boundary().size());forAll(emissivity,p)emissivity[p]=scalarField(mesh.boundary()[p].size(),1.);
 dictionary config;config.add("IBoundaryCondition",word("diffuseGrey"));
 MieDoControls controls{2,4,300,1e-13,1e-13,200,0,1e-12,100,1};
 MieDoRadiationSolver solver(mesh,config,table,patches,controls);
 auto workspace=solver.makeWorkspace();
 detail::ParticleRadiationTransactionContext context{mesh,table,patches,100,5000,0};
 Resident resident(mesh);const ugkwp::ParticleCaloric law;
 auto step=[&](double tw,double dt,bool cold){
  forAll(wall.boundaryField(),p)wall.boundaryFieldRef()[p]==tw;
  scalarField mass,temperature,diameter;resident.downloadMobileParticleRadiationMaterial(mesh,mass,temperature,diameter);
  scalarField eps(mesh.nCells(),.001);
  ParticleRadiationMaterialSnapshot snapshot{eps,mass,temperature,diameter,wall,emissivity,1542,dt,resident.particleRadiationResponse()};
  if(cold)workspace=solver.makeWorkspace();
  double before[4];resident.read(before);
  double doTemperature=0,doPower=0,incident=0;
  auto result=detail::solveParticleRadiationTransaction(context,snapshot,[&]{return &workspace;},[&](MieDoRadiationWorkspace*& w,const MieDoSnapshot& s){
   doTemperature=s.particleTemperatureK[0];auto out=solver.solve(s,*w);doPower=out.particlePowerDensityWm3[0];incident=out.incidentRadiationG_Wm2[0];return out;
  });
  require(doTemperature==temperature[0],"DO received a clipped physical temperature");
  const double sigma=5.670374419e-8;
  require(std::abs(doPower-(incident-4*sigma*std::pow(temperature[0],4)))<1e-6,"initial material/DO power mismatch");
  resident.applyParticleRadiationEnergy(mesh,result.deltaParticleRadiationEnergyJ,mass,temperature,1542);
  double after[4];resident.read(after);
  double actual=0,mean=0;
  for(int i=0;i<2;++i){const double weight=i?.7:.3;actual+=weight*((law.enthalpy(after[i])-law.enthalpy(before[i]))+(after[2+i]-before[2+i]));mean+=weight*law.temperature(law.enthalpy(after[i])+after[2+i]);}
  double boundary=result.escapedBoundaryRadiationEnergyJ;forAll(result.solidWallRadiationEnergyJ,p)for(double q:result.solidWallRadiationEnergyJ[p])boundary+=q;
  require(std::abs(actual-result.deltaParticleRadiationEnergyJ[0])<2e-8,"actual CHT enthalpy publication differs");
  require(std::abs(actual+boundary)<2e-8,"particle/wall heat mismatch");
  require(std::abs(mean-result.finalMeanTemperatureK[0])<1e-9,"published mean differs from material root");
  std::cout<<std::setprecision(17)<<"CAVITY old="<<temperature[0]<<" wall="<<tw<<" doT="<<doTemperature<<" doPower="<<doPower<<" heat="<<actual<<" final="<<mean<<" boundary="<<boundary<<"\n";
  return actual;
 };
 for(int csr:{0,1})for(double t:{500.,1000.,2500.,4000.,4500.}){resident.reset(t,t,0,0,csr);require(std::abs(step(t,.001,true))<1e-7,"isothermal cavity transferred heat");double x[4];resident.read(x);require(std::abs(x[0]-t)<1e-9&&std::abs(x[1]-t)<1e-9,"isothermal temperature changed");}
 for(bool heating:{false,true}){resident.reset(heating?3900:1100,heating?3900:1100);step(heating?5000:500,100,true);double x[4];resident.read(x);require(heating?x[0]>4000:x[0]<1000,"did not cross Mie endpoint");require(std::abs(step(x[0],.001,false))<1e-7,"post-crossing equilibrium lost");}
 resident.reset(1300,2700,.03125,-.0625);step(4000,.001,true);double saved[4];resident.read(saved);require(saved[2]!=0||saved[3]!=0,"carry not represented");step(3500,.001,false);double continuous[4];resident.read(continuous);
 resident.reset(saved[0],saved[1],saved[2],saved[3]);step(3500,.001,true);double resumed[4];resident.read(resumed);
 for(int i=0;i<4;++i)require(std::abs(continuous[i]-resumed[i])<1e-8,"restored owner/cold DO differs");
 std::cout<<"CAVITY_PASS equilibrium=10 crossing=2 heterogeneous_carry_restoration=1\n";
 return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
