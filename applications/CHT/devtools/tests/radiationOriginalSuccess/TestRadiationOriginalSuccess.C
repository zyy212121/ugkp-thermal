#include "fvCFD.H"
#include "GpuParticleRadiationCoupler.H"
#include "GpuParticleCaloric.H"
#include "GpuParticleThermalConsumers.H"
#include "GpuParticleRadiationMath.H"
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cfenv>
#include <iomanip>
#include <iostream>
// The actual bounded-reader production dependency and its small storage adapter
// are inserted from the already reviewed C03 fixture by the Python harness.
// C04_READER_ADAPTER
using namespace Foam;
using namespace Foam::gpuThermal;
void value(const char* name,double x){std::uint64_t bits;std::memcpy(&bits,&x,sizeof bits);std::cout<<"DATA "<<name<<' '<<std::hex<<bits<<std::dec<<'\n';}
void dump(const ParticleRadiationMaterialResult& r){
 forAll(r.finalMeanTemperatureK,i){value("absorption",r.absorptionCoefficientInvM[i]);value("scattering",r.scatteringCoefficientInvM[i]);value("extinction",r.extinctionCoefficientInvM[i]);value("asymmetry",r.asymmetryFactor[i]);value("temperature",r.finalMeanTemperatureK[i]);value("energy",r.deltaParticleRadiationEnergyJ[i]);}
 forAll(r.solidWallRadiationEnergyJ,p)forAll(r.solidWallRadiationEnergyJ[p],f)value("wall",r.solidWallRadiationEnergyJ[p][f]);
 value("escaped",r.escapedBoundaryRadiationEnergyJ);value("residual",r.globalConservationResidualJ);
}
int main(int argc,char**argv) try {
 fedisableexcept(FE_ALL_EXCEPT);
 const std::string mode=std::getenv("C04_MODE");
 if(mode.find("scale_")==0){
  const double a=std::getenv("C04_A")?std::strtod(std::getenv("C04_A"),nullptr):mode=="scale_ordinary"?-60:1e-20;
  const double b=std::getenv("C04_B")?std::strtod(std::getenv("C04_B"),nullptr):mode=="scale_zero"?0:mode=="scale_ordinary"?100:1e-20;
  const double gross=std::getenv("C04_G")?std::strtod(std::getenv("C04_G"),nullptr):std::abs(a)+std::abs(b);
  const double tolerance=std::getenv("C04_TOL")?std::strtod(std::getenv("C04_TOL"),nullptr):1e-10;
  auto r=detail::conservativeBoundaryEnergyScale(a,b,gross,tolerance);
  value("scale",r.scale);value("residual",r.residualJ);
 }else if(mode=="query"){
  MieTable table(std::getenv("C04_TABLE"));
  const double t=std::strtod(std::getenv("C04_T"),nullptr),d=std::strtod(std::getenv("C04_D"),nullptr),mu=std::strtod(std::getenv("C04_MU"),nullptr);
  const auto q=table.query(d,t);value("qabs",q.Qabs);value("qsca",q.Qsca);value("qext",q.Qext);value("band",q.planckBandFraction);value("phase",table.phase(d,t,mu));
 }else{
  #include "setRootCase.H"
  #include "createTime.H"
  #include "createMesh.H"
  fedisableexcept(FE_ALL_EXCEPT);
  MieTable table(std::getenv("C04_TABLE"));
  volScalarField wall(IOobject("T",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,dimensionedScalar("T",dimTemperature,300));
  List<scalarField> emissivity(mesh.boundary().size());forAll(emissivity,p)emissivity[p]=scalarField(mesh.boundary()[p].size(),1.);
  scalarField eps(mesh.nCells(),1e-6),mass(mesh.nCells(),.01),temperature(mesh.nCells(),3000),diameter(mesh.nCells(),1e-4);
  const double input=mode=="above"?4100:mode=="below"?100:mode=="reader"?3800:3000;
  temperature=input;
  ugkwp::ParticleCaloric law;
  const double cp=law.specificHeat(input);
  const double dt=(mode=="below" || mode=="inactive")?1.:.01;
  if(mode=="below")forAll(wall.boundaryField(),p)wall.boundaryFieldRef()[p]==1000.;
  if(mode=="inactive"){eps[1]=0;temperature[1]=250;}
  const double carry=.03125;
  if(mode=="reader"){
   DeviceState state;state.pT[0]=GpuReal(input);state.pThermalEnthalpyResidual[0]=carry;
   const auto r=particleRadiationContribution(state,0);temperature=r.temperatureMass/r.mass;
   value("reader_mean",temperature[0]);value("reader_carry",state.pThermalEnthalpyResidual[0]);
  }
  dictionary config;config.add("IBoundaryCondition",word("diffuseGrey"));config.add("linearSolverTolerance",scalar(1e-12));config.add("linearSolverMaxIterations",label(200));
  labelList patches(1,0);MieDoControls controls{2,4,200,1e-12,1e-12,200,0,1e-10,100,mode=="parallel"?2:1};
  MieDoRadiationSolver solver(mesh,config,table,patches,controls);
  if(mode=="do"){
   auto r=solver.solve(MieDoSnapshot{eps,temperature,diameter,wall,emissivity});
   forAll(r.incidentRadiationG_Wm2,i){value("incident",r.incidentRadiationG_Wm2[i]);value("source",r.particlePowerDensityWm3[i]);}
   value("do_residual",r.conservationResidualW);
  }else{
   auto workspace=solver.makeWorkspace();
   detail::ParticleRadiationTransactionContext context{mesh,table,patches,200,4000,0};
   ParticleRadiationMaterialSnapshot snapshot{eps,mass,temperature,diameter,wall,emissivity,cp,dt};
   const scalarField beforeTemperature(temperature),beforeDiameter(diameter);
   auto result=detail::solveParticleRadiationTransaction(context,snapshot,[&]{return &workspace;},[&](MieDoRadiationWorkspace*& w,const MieDoSnapshot& s){
    forAll(s.particleTemperatureK,i)value("initial_do_temperature",s.particleTemperatureK[i]);
    auto actual=solver.solve(s,*w);
    forAll(actual.incidentRadiationG_Wm2,i){value("incident",actual.incidentRadiationG_Wm2[i]);value("source",actual.particlePowerDensityWm3[i]);}
    forAll(actual.boundaryQrOutwardWm2,p)forAll(actual.boundaryQrOutwardWm2[p],f)value("do_boundary",actual.boundaryQrOutwardWm2[p][f]);
    value("integrated_particle",actual.integratedParticlePowerW);value("integrated_boundary",actual.integratedBoundaryPowerW);value("do_residual",actual.conservationResidualW);value("iterations",actual.sourceIterations);value("threads",actual.angleThreadsUsed);
    if(mode=="cancellation"){
     const double energies[]={1e16,-1e16+2,1};int index=0;
     forAll(actual.boundaryQrOutwardWm2,p)forAll(actual.boundaryQrOutwardWm2[p],f){const double energy=index<3?energies[index]:0;++index;actual.boundaryQrOutwardWm2[p][f]=energy/(.01*mag(mesh.Sf().boundaryField()[p][f]));}
    }
    return actual;
   });
   if(std::memcmp(beforeTemperature.begin(),temperature.begin(),temperature.size()*sizeof(scalar)) || std::memcmp(beforeDiameter.begin(),diameter.begin(),diameter.size()*sizeof(scalar)))throw std::runtime_error("caller storage changed");
   dump(result);
   if(mode=="reader" || mode=="ordinary" || mode=="parallel")forAll(mass,i){
    auto u=radiationAffineTemperatureUpdate(result.deltaParticleRadiationEnergyJ[i],mass[i],cp,temperature[i],200.,4000.);
    ugkwp::ParticleThermalCarry<GpuReal> state{GpuReal(input),carry};
    auto p=ugkwp::publishAffineRadiationThermalHeat(state,u.scale,u.offset,GpuReal(200),GpuReal(4000),law);
    value("affine_scale",u.scale);value("affine_offset",u.offset);value("valid",p.valid);value("limited",p.limited);value("accepted_heat",p.acceptedSpecificHeat);value("cache",state.temperature);value("carry",state.specificEnthalpyResidual);
   }
  }
 }
 std::cout<<"PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
