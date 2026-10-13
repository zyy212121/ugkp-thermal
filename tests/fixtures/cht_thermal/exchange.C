#include "fvCFD.H"
#include "GpuSolidThermalCoupler.H"
#include "GpuThermalExchangeState.H"
#include "RecordingResident.H"
#include "IStringStream.H"
#include "fixedGradientFvPatchFields.H"
#include <cmath>
#include <iostream>
#include <fstream>
extern int t14DictionaryOpens,t14HashCalls,t14FieldScans;
using namespace Foam;using namespace Foam::gpuThermal;
void ck(bool a,const char* m){if(!a)throw std::runtime_error(m);}void near(scalar a,scalar b,scalar e=1e-10){if(!std::isfinite(a)||std::abs(a-b)>e*std::max(scalar(1),std::abs(b))){std::cerr<<a<<" != "<<b<<'\n';throw std::runtime_error("value mismatch");}}
int main(int argc,char**argv){
 #include "setRootCase.H"
 #include "createTime.H"
 fvMesh mesh(IOobject(fvMesh::defaultRegion,runTime.timeName(),runTime,IOobject::MUST_READ));
 IOdictionary config(IOobject("exchangeConfig",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
 const word mode=config.lookup<word>("testMode");GpuParticleResidentSolver resident(mesh);resident.mismatch=mode=="mismatch";resident.failConsume=mode=="consume_error";
 volScalarField T(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::AUTO_WRITE),mesh);
 #define SF(n) volScalarField n(IOobject(#n,runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),mesh,dimensionedScalar(#n,dimless,1))
 #define VF(n) volVectorField n(IOobject(#n,runTime.timeName(),mesh,IOobject::NO_READ,IOobject::AUTO_WRITE),mesh,dimensionedVector(#n,dimless,vector::zero))
 SF(rho);VF(rhoU);SF(rhoE);VF(U);SF(p);SF(epsilonS);VF(rhoUs);SF(rhoEs);SF(rhoDs);SF(rhoHp);VF(Us);SF(theta);SF(Tp);SF(dMeanCell);
 ThermalExchangeRestartState initial;initial.formatVersion=4;initial.initialState=true;
 FatalError.throwExceptions();FatalIOError.throwExceptions();
 try{
  if(mode.find("legacy_")==0||mode=="missing_current"||mode=="missing_unknown"){
   const bool legacy=mode.find("legacy_")==0;
   initial.initialState=false;initial.formatVersion=legacy?(mode=="legacy_one_absent"?1:mode=="legacy_two_absent"?2:3):(mode=="missing_unknown"?99:4);initial.completedSimulationTimeS=legacy?-.2:0;resident.zeroAtConfigure=true;
   bool failed=false;try {GpuSolidThermalCoupler decoded(runTime,mesh,resident,config,initial);}catch(const GpuSolidThermalSolveError& e){if(legacy)throw;failed=std::string(e.what()).find("pending wall-energy ledger")!=std::string::npos;}
   if(!legacy)ck(failed,"required current ledger fabricated");else {ck(resident.restoreCalls==0,"invented legacy ledger upload");forAll(resident.gas,i)near(resident.gas[i],0,0);}
   Info<<"PASS "<<mode<<nl;return 0;
  }
  const bool pending=mode.find("pending_")==0;
  if(pending){
   const auto& patch=mesh.boundaryMesh()[mesh.boundaryMesh().findPatchID("left")];ck(patch.size()==3,"three actual ledger faces required");
   std::ofstream os((runTime.timePath()/ThermalRestartPreflight::pendingWallEnergyObjectName()).c_str());
   os<<"formatVersion 99; checkpointSimulationTimeS 123; completedExchangeSimulationTimeS -4; nEntries 999; gasConvectiveEnergyJ 999; entries (("<<patch.start()+2<<" 50 20 -5)("<<patch.start()<<" 123 456 789)("<<patch.start()<<" 50 20 -5)";
   if(mode!="pending_missing")os<<"("<<patch.start()+1<<" 50 20 -5)";os<<");\n";os.close();initial.initialState=false;initial.formatVersion=mode=="pending_legacy_present"?2:99;
  }
  if(mode=="pending_missing") {bool failed=false;try {GpuSolidThermalCoupler missing(runTime,mesh,resident,config,initial);}catch(const GpuSolidThermalSolveError& e){failed=std::string(e.what()).find("missing a required configured face")!=std::string::npos;}ck(failed,"missing pending face fabricated");Info<<"PASS "<<mode<<nl;return 0;}
  GpuSolidThermalCoupler coupler(runTime,mesh,resident,config,initial);coupler.initialiseWallTemperatureBeforeFirstAdvance(T);
  runTime.setTime(.1,1);
  auto advance=[&](){return coupler.exchangeIfDue(runTime,rho,rhoU,rhoE,U,p,T,epsilonS,rhoUs,rhoEs,rhoDs,rhoHp,Us,theta,Tp,dMeanCell);};
  if(mode=="outer_zero_k") {bool failed=false;try{advance();}catch(const GpuSolidThermalSolveError& e){failed=std::string(e.what()).find("conductance divisor is zero")!=std::string::npos;}ck(failed,"required zero divisor hidden");ck(resident.gasResets==0&&resident.particleResets==0,"arithmetic error consumed ledgers");Info<<"PASS "<<mode<<nl;return 0;}
  if(mode=="consume_error") {bool fail=false;try{advance();}catch(const std::runtime_error& e){fail=std::string(e.what()).find("consume error")!=std::string::npos;}ck(fail,"actual failure hidden");const auto& sm=runTime.lookupObject<fvMesh>("solid");near(sm.lookupObject<volScalarField>("T")[0],300,0);ck(resident.gasResets==0,"reset on failed consume");Info<<"PASS "<<mode<<nl;return 0;}
  auto result=advance();ck(result.coupled,"no actual exchange");near(result.elapsedCouplingTimeS,.1,0);
  const auto& sm=runTime.lookupObject<fvMesh>("solid");const auto& ts=sm.lookupObject<volScalarField>("T");
  scalar energy=0;forAll(ts,i)energy+=1000*sm.V()[i]*(ts[i]-300);
  scalar radiativeWallEnergy=0;const bool radiating=mode=="radiation";
  if(radiating){const label left=mesh.boundaryMesh().findPatchID("left");const auto& flux=mesh.lookupObject<surfaceScalarField>("particleRadiationWallHeatFlux");forAll(flux.boundaryField()[left],f)radiativeWallEnergy+=flux.boundaryField()[left][f]*mesh.magSf().boundaryField()[left][f]*.1;ck(radiativeWallEnergy!=0,"radiation energy discarded");ck(resident.radiationApplications==1&&resident.enthalpyRefreshes==1,"radiation publication incomplete");forAll(resident.radiationMean,i){near(resident.radiationMean[i],4100,1e-13);ck(std::isfinite(resident.radiationEnergy[i])&&resident.radiationEnergy[i]!=0,"actual particle energy discarded");}near(result.elapsedRadiationTimeS,.1,0);}
  const bool three=config.subDict("solidThermalCoupling").found("solidSolidInterfaces");
  if(three){const auto& am=runTime.lookupObject<fvMesh>("aux");const auto& ta=am.lookupObject<volScalarField>("T");forAll(ta,i)energy+=1000*am.V()[i]*(ta[i]-100);near(ta[0],100.12);near(ts[0],299.945);near(ta.boundaryField()[am.boundaryMesh().findPatchID("right")][0],200.12);near(refCast<const fixedGradientFvPatchScalarField>(ta.boundaryField()[am.boundaryMesh().findPatchID("right")]).gradient()[0],200);near(ts.boundaryField()[sm.boundaryMesh().findPatchID("left")][0],199.945);}
  else near(ts[0],300+(pending?195:65)/1000.0+radiativeWallEnergy/1000);
  near(energy,(pending?195:65)+radiativeWallEnergy,1e-9);ck(resident.gasResets==1&&resident.particleResets==1,"wrong reset count");ck(resident.gasDownloads==0&&resident.particleDownloads==0,"discarded consume transfers retained");
  auto noWork=advance();ck(!noWork.coupled&&resident.gasResets==1,"repeated same time consumed");
  label left=mesh.boundaryMesh().findPatchID("left");near(mesh.lookupObject<surfaceScalarField>("gasConvectiveWallHeatFlux").boundaryField()[left][0],pending?1500:500);near(mesh.lookupObject<surfaceScalarField>("particleStuckWallHeatFlux").boundaryField()[left][0],pending?600:200);near(mesh.lookupObject<surfaceScalarField>("particleReflectedWallHeatFlux").boundaryField()[left][0],pending?-150:-50);
  near(T.boundaryField()[left][0],ts.boundaryField()[sm.boundaryMesh().findPatchID("right")][0]);
  forAll(ts.boundaryField(),pi)forAll(ts.boundaryField()[pi],f)ck(std::isfinite(ts.boundaryField()[pi][f]),"unpublished face");
  coupler.persistAtWriteTime(runTime,rho,rhoU,rhoE,U,p,T,epsilonS,rhoUs,rhoEs,rhoDs,rhoHp,Us,theta,Tp,dMeanCell,true);
  ThermalRestartPreflightExpectations expected;auto state=ThermalRestartPreflight::validateRestartDirectory(runTime.timePath(),expected);near(state.completedSimulationTimeS,.1,0);near(state.completedRadiationSimulationTimeS,radiating?.1:0,0);near(state.particleContactEnergyJ,pending?45:15,0);
  auto ledger=ThermalRestartPreflight::readPendingWallEnergyLedgers(runTime.timePath(),.1,.1,"",mesh.nFaces());ck(ledger.faceIds.size()==(pending?3:1),"missing persisted face");forAll(ledger.faceIds,i){near(ledger.gasConvectiveEnergyJ[i],0,0);near(ledger.particleDepositedEnergyJ[i],0,0);near(ledger.particleReflectedEnergyJ[i],0,0);}
  ck(t14HashCalls==0&&t14FieldScans==0,"runtime certificate work remains");ck(t14DictionaryOpens==(pending?3:2),"unexpected state/ledger reread");Info<<"COST dictionary_opens="<<t14DictionaryOpens<<" hash_calls="<<t14HashCalls<<" field_scans="<<t14FieldScans<<nl;
  Info<<"PASS "<<mode<<" energy="<<energy<<" primary="<<ts[0]<<" transfers="<<resident.gasDownloads+resident.particleDownloads<<nl;return 0;
 }catch(const Foam::error& e){std::cerr<<e.message()<<'\n';return 2;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}
}
