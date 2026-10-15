// Real OpenFOAM thermal owners; only the native resident/device boundary records calls.
#include "fvCFD.H"
#include "GpuSolidThermalCoupler.H"
#include "GpuThermalExchangeState.H"
#include "CompatibilityResident.H"
#include "IStringStream.H"
#include "IFstream.H"
#include "fixedGradientFvPatchFields.H"
#include <fstream>
#include <iostream>
#include <iomanip>
#include <limits>
using namespace Foam; using namespace Foam::gpuThermal;
void emitData(const std::string& name, scalar v){std::cout<<"DATA "<<name<<' '<<std::hexfloat<<v<<'\n';}
void value(const std::string& name, scalar v){emitData(name,v);}
void value(const std::string& name, const vector& v){for(direction c=0;c<3;++c)emitData(name+"."+std::to_string(c),v[c]);}
template<class Field> void field(const std::string& name,const Field& f){
 forAll(f,i)value(name+".cell."+std::to_string(i),f[i]);
 forAll(f.boundaryField(),p)forAll(f.boundaryField()[p],i)value(name+".patch."+std::to_string(p)+"."+std::to_string(i),f.boundaryField()[p][i]);
}
void solid(const std::string& name,const volScalarField& t){field(name,t);forAll(t.boundaryField(),p)if(isA<fixedGradientFvPatchScalarField>(t.boundaryField()[p])){const auto& g=refCast<const fixedGradientFvPatchScalarField>(t.boundaryField()[p]).gradient();forAll(g,i)emitData(name+".gradient."+std::to_string(p)+"."+std::to_string(i),g[i]);}}
std::string fullState(label v,scalar clock){std::ostringstream s;s<<"formatVersion "<<v<<"; initialState true; exchangeSequence 0; completedTimeIndex 0; completedSimulationTimeS 0; completedRadiationSimulationTimeS "<<clock<<"; previousExchangeSimulationTimeS 0; gasWallLedgerConsumed false; particleWallLedgerConsumed false; particleRadiationApplied false; solidStateUpdated false; wallTemperatureUploaded false; particleMomentsRebuilt false; particleContactEnergyJ 0; ";for(const char* k:{"fluidMeshTopologySha1","solidMeshTopologySha1","couplingConfigurationSha1","wallTemperatureSha1UsedForCompletedInterval","newlyUploadedWallTemperatureSha1"})s<<k<<" \"0000000000000000000000000000000000000000\"; ";s<<"auxiliarySolidMeshTopologySha1 \""<<(v>=2?"0000000000000000000000000000000000000000":"")<<"\";";return s.str();}
int main(int argc,char**argv){
#include "setRootCase.H"
#include "createTime.H"
 fvMesh mesh(IOobject(fvMesh::defaultRegion,runTime.timeName(),runTime,IOobject::MUST_READ));
 IOdictionary config(IOobject("exchangeConfig",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
 const word mode=config.lookup<word>("c05Mode");
 FatalError.throwExceptions();FatalIOError.throwExceptions();
 try{
  if(mode.find("surface_")==0){
   volScalarField t(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
   const label pi=mesh.boundaryMesh().findPatchID("left");auto& p=refCast<fixedGradientFvPatchScalarField>(t.boundaryFieldRef()[pi]);
   if(mode=="surface_owner")t[p.patch().faceCells()[0]]=std::numeric_limits<scalar>::quiet_NaN();
   if(mode=="surface_gradient")p.gradient()[0]=std::numeric_limits<scalar>::quiet_NaN();
   if(mode=="surface_delta")const_cast<scalarField&>(p.patch().deltaCoeffs())[0]=0;
   const scalarField saved(p);const scalarField got=coupledPatchSurfaceTemperature(t,pi);
   if(got.size()!=saved.size()||std::memcmp(got.begin(),saved.begin(),got.size()*sizeof(scalar)))throw std::runtime_error("saved surface bytes changed");
   forAll(got,i)emitData("savedSurface."+std::to_string(i),got[i]);return 0;
  }
  if(mode.find("decode_")==0){
   std::ofstream state((runTime.timePath()/ThermalRestartPreflight::stateObjectName()).c_str());
   state<<"initialState false; exchangeSequence 5; completedSimulationTimeS -0.2; ";
   if(config.found("stateVersion"))state<<"formatVersion "<<config.lookup<word>("stateVersion")<<"; ";
   if(config.lookupOrDefault<bool>("radiationClock",true))state<<"completedRadiationSimulationTimeS -0.3; ";state.close();
   for(const word& n:wordList{"gpuResidentStrictParticles.dat","gpuResidentStrictEpsGPrev.dat","gpuResidentStrictSourceResidual.dat","rho","rhoU","rhoE"}){if(config.lookupOrDefault<bool>("missingRhoE",false)&&n=="rhoE")continue;std::ofstream f((runTime.timePath()/n).c_str());f<<"recording boundary presence fixture\n";}
   const auto& patch=mesh.boundaryMesh()[mesh.boundaryMesh().findPatchID("left")];
   if(config.lookupOrDefault<bool>("pending",true)){std::ofstream f((runTime.timePath()/ThermalRestartPreflight::pendingWallEnergyObjectName()).c_str());f<<"formatVersion futureTag; checkpointSimulationTimeS 123; completedExchangeSimulationTimeS -99; nEntries 99; entries (("<<patch.start()<<" 123 456 789)";forAll(patch,i){if(config.lookupOrDefault<bool>("missingSlot",false)&&i==patch.size()-1)continue;f<<"("<<patch.start()+i<<" 50 20 -5)";}f<<");\n";}
   const auto initial=ThermalRestartPreflight::validateBeforeResidentInitialise(runTime,mesh,config);
   emitData("decodedClock",initial.completedRadiationSimulationTimeS);
   GpuParticleResidentSolver resident(mesh);resident.zeroAtConfigure=true;
   GpuSolidThermalCoupler coupler(runTime,mesh,resident,config,initial);
   emitData("restoreCalls",resident.restoreCalls);forAll(patch,i){emitData("gas."+std::to_string(i),resident.gas[patch.start()+i]);emitData("deposited."+std::to_string(i),resident.deposited[patch.start()+i]);emitData("reflected."+std::to_string(i),resident.reflected[patch.start()+i]);}return 0;
  }
  const label version=config.lookupOrDefault<label>("reviewStateVersion",4);const scalar clock=config.lookupOrDefault<scalar>("reviewExtraRadiationClock",0);
  IStringStream st(fullState(version,clock));dictionary sd(st);const auto initial=ThermalRestartPreflight::stateFromDictionary(sd);emitData("decodedClock",initial.completedRadiationSimulationTimeS);
  GpuParticleResidentSolver resident(mesh);resident.heatEnabled=mode!="flags_disabled";
  volScalarField T(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
#define SF(n) volScalarField n(IOobject(#n,runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),mesh,dimensionedScalar(#n,dimless,1))
#define VF(n) volVectorField n(IOobject(#n,runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),mesh,dimensionedVector(#n,dimless,vector::zero))
  SF(rho);VF(rhoU);SF(rhoE);VF(U);SF(p);SF(epsilonS);VF(rhoUs);SF(rhoEs);SF(rhoDs);SF(rhoHp);VF(Us);SF(theta);SF(Tp);SF(dMeanCell);
  GpuSolidThermalCoupler coupler(runTime,mesh,resident,config,initial);coupler.initialiseWallTemperatureBeforeFirstAdvance(T);runTime.setTime(.1,1);
  const auto r=coupler.exchangeIfDue(runTime,rho,rhoU,rhoE,U,p,T,epsilonS,rhoUs,rhoEs,rhoDs,rhoHp,Us,theta,Tp,dMeanCell);
  emitData("coupled",r.coupled);emitData("radiationCoupled",r.radiationCoupled);emitData("elapsedCoupling",r.elapsedCouplingTimeS);emitData("elapsedRadiation",r.elapsedRadiationTimeS);emitData("particleCount",r.particleCount);
#define DUMP(n) field(#n,n)
  DUMP(rho);DUMP(rhoU);DUMP(rhoE);DUMP(U);DUMP(p);DUMP(T);DUMP(epsilonS);DUMP(rhoUs);DUMP(rhoEs);DUMP(rhoDs);DUMP(rhoHp);DUMP(Us);DUMP(theta);DUMP(Tp);DUMP(dMeanCell);
  const auto& sm=runTime.lookupObject<fvMesh>("solid");solid("solid",sm.lookupObject<volScalarField>("T"));emitData("hasAuxiliary",runTime.foundObject<fvMesh>("aux"));if(runTime.foundObject<fvMesh>("aux"))solid("aux",runTime.lookupObject<fvMesh>("aux").lookupObject<volScalarField>("T"));
  for(const word& n:wordList{"gasConvectiveWallHeatFlux","particleStuckWallHeatFlux","particleReflectedWallHeatFlux","particleRadiationWallHeatFlux"})if(mesh.foundObject<surfaceScalarField>(n))field(n,mesh.lookupObject<surfaceScalarField>(n));
  for(const word& n:wordList{"particleRadiationAbsorptionCoefficient","particleRadiationScatteringCoefficient","particleRadiationExtinctionCoefficient","particleRadiationAsymmetryFactor"})if(mesh.foundObject<volScalarField>(n))field(n,mesh.lookupObject<volScalarField>(n));
  forAll(resident.radiationEnergy,i){emitData("particleRadiationEnergy."+std::to_string(i),resident.radiationEnergy[i]);emitData("particleRadiationMean."+std::to_string(i),resident.radiationMean[i]);}
  emitData("gasResets",resident.gasResets);emitData("particleResets",resident.particleResets);emitData("radiationApplications",resident.radiationApplications);emitData("enthalpyRefreshes",resident.enthalpyRefreshes);
  coupler.persistAtWriteTime(runTime,rho,rhoU,rhoE,U,p,T,epsilonS,rhoUs,rhoEs,rhoDs,rhoHp,Us,theta,Tp,dMeanCell,true);
  IFstream in(runTime.timePath()/ThermalRestartPreflight::stateObjectName());dictionary state(in);
  for(const word& n:wordList{"gasWallLedgerConsumed","particleWallLedgerConsumed","particleRadiationApplied","solidStateUpdated","wallTemperatureUploaded","particleMomentsRebuilt"})emitData(n,bool(state.lookup<Switch>(n)));
  for(const word& n:wordList{"completedSimulationTimeS","completedRadiationSimulationTimeS","previousExchangeSimulationTimeS"})emitData(n,state.lookup<scalar>(n));
  scalar contactEnergy=0;
  const auto& depositedFlux=mesh.lookupObject<surfaceScalarField>("particleStuckWallHeatFlux");const auto& reflectedFlux=mesh.lookupObject<surfaceScalarField>("particleReflectedWallHeatFlux");
  forAll(depositedFlux.boundaryField(),pi)forAll(depositedFlux.boundaryField()[pi],f)contactEnergy+=(depositedFlux.boundaryField()[pi][f]+reflectedFlux.boundaryField()[pi][f])*mesh.magSf().boundaryField()[pi][f]*r.elapsedCouplingTimeS;
  emitData("transferredContactEnergyJ",contactEnergy);
  const auto noWork=coupler.exchangeIfDue(runTime,rho,rhoU,rhoE,U,p,T,epsilonS,rhoUs,rhoEs,rhoDs,rhoHp,Us,theta,Tp,dMeanCell);emitData("repeatedCoupled",noWork.coupled);emitData("finalGasResets",resident.gasResets);return 0;
 }catch(const Foam::error& e){std::cerr<<"ERROR "<<e.message()<<'\n';return 2;}catch(const std::exception& e){std::cerr<<"ERROR "<<e.what()<<'\n';return 2;}
}
