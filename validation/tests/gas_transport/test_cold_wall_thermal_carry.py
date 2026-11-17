"""CPU execution of actual cold profile and coupled thermal transaction paths.

Eight synchronized host lanes adapt CUDA collectives. Both complete cold wrappers,
actual paired source/reduction/acceptance/settlement, gas primitive recovery and
caloric helpers are production code. The callback wrapper only counts invocations;
it delegates all physics to the real publisher. No numeric physics stubs are used.

Mixed cases run one closed stationary cell with three independently represented
ordinary/1D/2D particles. The wall-bound directory selects each eligible cold
particle for its actual wrapper. Transport, pressure, collision, tracking,
injection and full strictAdvance dispatch are outside this thermal-only fixture.
CUDA code generation, warp scheduling/FMA and performance are not established.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import pytest

ROOT = SOURCE_ROOT

HARNESS = r'''
#include "GpuColdWallThermalCarry.cuh"
#include <random>
#include <iostream>
#include <type_traits>
ugkwp::ParticleCaloric law(const DeviceState& s) {
 return ugkwp::coldWallCarryCaloric(s.coldWallSolidificationParameters);
}
template<class Node> double inventory(const Node* h,int n,double r) {double a=r;for(int j=0;j<n;++j)a+=double(h[j])/n;return a;}
void pure_profile(int n) {
 double h[64],old[64];for(int j=0;j<n;++j)h[j]=float(4000000.+j*1000.);
 for(double heat:{0.,1e-3,-7.25,1e6,-1e12}){
  std::copy(h,h+n,old);double expected=0.;
  for(int j=0;j<n;++j)expected+=((double(old[j])+heat)-double(old[j]))/n;
  auto q=ugkwp::publishColdWallUniformSpecificHeat(h,n,heat);
  require(q.valid,"finite raw publication");near(q.acceptedSpecificHeat,expected,0.,"accepted heat equals actual stored difference");
  for(int j=0;j<n;++j)require(h[j]==double(old[j])+heat,"ordinary nearest float, no mean correction");
  if(heat==1e-3)require(q.acceptedSpecificHeat!=0.,"native64 resolves its own ULP");
 }
}
template<bool twoD> void kernel_profile(double dt,double temperature,bool gradient,bool gas,bool cooling,bool nullBuffers) {
 DeviceState s;seed(s);s.pT[0]=GpuReal(temperature);
 if(nullBuffers){s.solveParticleTemperature=0;s.particleHeatBudget=nullptr;s.particleHeatCoefficient=nullptr;}
 s.pThermalEnthalpyResidual[0]=twoD?.07125:0.;
 const auto c=law(s);auto* h=[&]{if constexpr(twoD)return s.cold2DH;else return s.coldH;}();const int n=twoD?64:8;
 for(int j=0;j<n;++j)h[j]=float(c.enthalpy(temperature+(gradient?40.*double(j)/(n-1):0.)));
 const double oldR=s.pThermalEnthalpyResidual[0];
 std::remove_pointer_t<decltype(h)> old[64];std::copy(h,h+n,old);const double before=inventory(h,n,oldR);
 s.gasBoundaryT[0]=GpuReal(temperature-100.);
 const GpuReal volume=(finiteContactPi/GPU_R(6))*s.pd[0]*s.pd[0]*s.pd[0];
 const GpuReal physicalMass=liquidAluminaProperties(s.pT[0]).densityKgM3*volume;
 s.pm[0]=GPU_R(7)*physicalMass;
 GpuReal wall=0,mean=0,surface=0,frozen=0;
 const int failures=launch8([&]{
  GpuReal w=0,m=0,f=0,st=0;bool ok;
  if(twoD)ok=advanceColdWall2DThermalGroup(s,0,threadIdx.x,255,volume,physicalMass,
    GPU_R(2.2e-8),GPU_R(1.e-8),dt,s.gasBoundaryT[0],GPU_R(10000),GPU_R(1.e-5),
    cooling?GPU_R(1000):GPU_R(4200),gas?GPU_R(1.e-5):GPU_R(0),m,st,f,w);
  else ok=advanceColdWall1DThermalGroup(s,0,threadIdx.x,255,volume,physicalMass,
    GPU_R(2.2e-8),GPU_R(1.e-8),.01,GPU_R(.4),dt,s.gasBoundaryT[0],GPU_R(10000),
    GPU_R(1.e-5),GPU_R(4200),GPU_R(0),m,f,w);
  if(!ok)throw DeviceTrap{};
  if(threadIdx.x==0){wall=w;mean=m;surface=st;frozen=f;}
 });
 require(failures==0,"actual group accepts cold carry state");
 const double after=inventory(h,n,s.pThermalEnthalpyResidual[0]);
 const GpuReal published=GpuReal((s.pm[0]/physicalMass)*wall);
 const double expected=(s.heatBudget[0]-double(published))/double(s.pm[0]);
 if(twoD)near(after-before,expected,3.e-9,"legacy 2D profile increment pairs published wall energy");
 else {require(std::isfinite(after-before-expected),"raw 1D balance is finite");std::cout<<"RAW1D_STORAGE_RESIDUAL "<<after-before-expected<<"\n";}
 require(std::abs(s.pThermalEnthalpyResidual[0])<=.500001,"bounded published carry");
 near(double(mean),double(GpuReal(c.temperature(after))),1.e-9,"published mean uses authority");
 if(dt==0.)require(oldR==s.pThermalEnthalpyResidual[0]&&std::equal(h,h+n,old),"zero-time exact profile and carry identity");
 else require(expected!=0.,"nonzero wall exchange not rounded away");
 if(!gas)near(s.heatBudget[0],0.,2.e-18,"wall-only update contributes zero gas heat");
 if(gas&&dt==1.e-10&&!gradient) {
  const double initialTop=c.temperature(double(old[7])+oldR);
  const double analytic=double(s.pm[0])/double(physicalMass)*double(GPU_R(1.e-5))
    *((cooling?1000.:4200.)-initialTop)*dt;
  near(s.heatBudget[0],analytic,2.e-19,"single substep top boundary analytic gas work");
 }
 if(gas){require(cooling?s.heatBudget[0]<0.:s.heatBudget[0]>0.,"signed top gas heat published to cell actual budget");if(dt>=1.e-5)require(cooling?h[7]<h[0]:h[7]>h[0],"2D top source remains spatially nonuniform");}
 if(std::abs(temperature-2336.9999)<1.e-5)require(c.temperature(double(h[7])+s.pThermalEnthalpyResidual[0])>c.liquidus(),"gas heat crosses liquidus");
 if(std::abs(temperature-2317.0001)<1.e-5)require(c.temperature(double(h[7])+s.pThermalEnthalpyResidual[0])<c.solidus(),"gas cooling crosses solidus");
 (void)surface;(void)frozen;
}
void coldest_node_roundoff() {pure_profile(8);}
void coldest_positive_roundoff() {pure_profile(8);}
void publication_stress() {
 std::mt19937_64 rng(453156);std::uniform_real_distribution<double> u(0.,1.);
 for(int trial=0;trial<200000;++trial){
  double h[8],old[8];for(int j=0;j<8;++j)h[j]=old[j]=float(std::pow(10.,-20.+40.*u(rng)));
  double q=(2*u(rng)-1)*1e6,expected=0.;
  for(int j=0;j<8;++j)expected+=((double(old[j])+q)-double(old[j]))/8.;
  auto result=ugkwp::publishColdWallUniformSpecificHeat(h,8,q);
  require(result.valid,"finite raw node target");near(result.acceptedSpecificHeat,expected,0.,"random stored-energy difference");
  for(int j=0;j<8;++j)require(h[j]==double(old[j])+q,"random ordinary float target");
 }
}
void no_duplicate_gas() {
 DeviceState s;seed(s);s.pThermalEnthalpyResidual[0]=.03125;
 double old[8];std::copy(s.coldH,s.coldH+8,old);
 require(launch8([&]{GpuReal mean=0;
   if(!advanceColdWall1DGasOnlyGroup(s,0,threadIdx.x,255,GPU_R(1.e-9),GPU_R(4200),GPU_R(1),1.e-4,0.,mean))throw DeviceTrap{};
 })==0,"completed deposit gas-only group");
 require(std::equal(s.coldH,s.coldH+8,old),"paired gas is not exchanged twice");
 clearColdWallParticleState(s,0);require(s.pThermalEnthalpyResidual[0]==0.,"1D clear removes unused residual");
 const double initializedEnthalpy=coldWallSpecificEnthalpyJkg(GPU_R(3200),s.coldWallSolidificationParameters);
 const double castLoss=initializedEnthalpy-double(float(initializedEnthalpy));
 initialiseColdWallParticleState(s,0,GPU_R(3200));
 require(s.pThermalEnthalpyResidual[0]==0.,"1D activation has no carry");
 for(int j=0;j<8;++j)require(s.coldH[j]==initializedEnthalpy,"1D each initialized native node is exact");
 clearColdWall2DParticleState(s,0);require(s.pThermalEnthalpyResidual[0]==0.,"2D clear retains preexisting residual");
 const double before2d=s.pThermalEnthalpyResidual[0];
 initialiseColdWall2DParticleState(s,0,GPU_R(3200));
 require(s.pThermalEnthalpyResidual[0]==before2d+castLoss,"2D activation retains previous carry and cast loss");
 near(inventory(s.cold2DH,64,s.pThermalEnthalpyResidual[0]),initializedEnthalpy+before2d,0.,"2D initial represented enthalpy");
}

void full_2d_wrapper(double scale) {
 DeviceState s;seed(s);s.coldWallSolidificationEnabled=0;
 s.pStuck[0]=particleWallDeposited;s.pDepositionArea[0]=1.e-8f;
 s.particleStuckCandidateMask[0]=particleWallColdWall2D;
 s.particleHeatCoefficient[0]=scale;s.pThermalEnthalpyResidual[0]=.07125;
 const auto c=law(s);for(int j=0;j<64;++j)s.cold2DH[j]=float(c.enthalpy(s.pT[0]));
 s.gasBoundaryT[0]=GPU_R(3000);const double before=inventory(s.cold2DH,64,s.pThermalEnthalpyResidual[0]);
 const double topT=c.temperature(double(s.cold2DH[7])+s.pThermalEnthalpyResidual[0]);
 const GpuReal volume=(finiteContactPi/GPU_R(6))*s.pd[0]*s.pd[0]*s.pd[0];
 const GpuReal physicalMass=liquidAluminaProperties(s.pT[0]).densityKgM3*volume;
 const GpuReal cp=particleSpecificHeatDevice(s.pT[0]);
 const GpuReal rate=GPU_R(6)*GPU_R(2)*molecularGasConductivity(s)/(s.rhoSolid*cp*s.pd[0]*s.pd[0]+GPU_TINY(1e-300));
 GpuReal g=rate*physicalMass*cp;const double target=double(g)*scale;g=GpuReal(target);
 if(double(g)>target)g=std::nextafter(g,GpuReal(0));
 const double expected=double(s.pm[0])/double(physicalMass)*double(g)*(double(s.couplingTgasOld[0])-topT)*1.e-10;
 require(launch8([&]{relaxColdWall2DParticlesToResidentGasKernel(&s,1.e-10);})==0,"full 2D production wrapper succeeds");
 const double delta=(inventory(s.cold2DH,64,s.pThermalEnthalpyResidual[0])-before)*double(s.pm[0]);
 near(delta+double(s.deposited[0])-s.heatBudget[0],0.,4.e-17,"full 2D actual particle wall and gas budget");
 near(s.heatBudget[0],expected,2.e-19,"full 2D accepted native conductance scaled gas work");
 require(s.pStuck[0]==particleWallDeposited,"2D deposit retained");
}
void prepare_1d_wrapper(DeviceState& s) {
 seed(s);s.coldWall2DEnabled=0;s.pStuck[0]=particleWallDeposited;
 s.pDepositionArea[0]=1.e-8f;s.particleHeatCoefficient[0]=.3;
 s.pThermalEnthalpyResidual[0]=.07125;s.gasBoundaryT[0]=GPU_R(2200);
 s.particleWallDepositionHeatTransferEfficiency=GPU_R(.2);
}
void full_1d_wrapper(double dt,bool wide=false) {
 DeviceState actual,reference;prepare_1d_wrapper(actual);prepare_1d_wrapper(reference);
 if(wide){actual.particleWallDepositionHeatTransferEfficiency=GPU_R(1.01);reference.particleWallDepositionHeatTransferEfficiency=GPU_R(1.01);}
 const double before=particleEffectiveSpecificEnthalpyDevice(actual,0);
 const auto proposal=pairedParticleHeatCandidate(actual,0,dt);
 actual.particleContactHeatCandidate[0]=reference.particleContactHeatCandidate[0]=proposal.specificHeat;
 const GpuReal volume=(finiteContactPi/GPU_R(6))*actual.pd[0]*actual.pd[0]*actual.pd[0];
 const GpuReal oldPhysicalMass=liquidAluminaProperties(actual.pT[0]).densityKgM3*volume;
 const GpuReal oldMultiplicity=actual.pm[0]/oldPhysicalMass;
 // Numeric reference uses exactly the production publisher, then exactly the
 // production cold group, with material mass frozen before the publication.
 actualPublishPairedParticleGasHeat(reference,0,dt);
 const double published=reference.heatBudget[0];
 const GpuReal effusivity=std::sqrt(reference.particleWallDensityKgM3*reference.particleWallSpecificHeatJkgK*reference.particleWallConductivityWmK);
 GpuReal refWall=0,refMean=0;
 require(launch8([&]{GpuReal mean=0,frozen=0,wall=0;
  bool ok=advanceColdWall1DThermalGroup(reference,0,threadIdx.x,255,volume,oldPhysicalMass,
   GpuReal(reference.pContactMaximumArea[0]),GpuReal(reference.pDepositionArea[0]),
   GpuTime(reference.pContactDuration[0]),GpuReal(reference.pContactPeakFraction[0]),dt,
   reference.gasBoundaryT[0],effusivity,reference.particleWallDepositionHeatTransferEfficiency,
   reference.couplingTgasOld[0],GPU_R(0),mean,frozen,wall);
  if(!ok)throw DeviceTrap{};if(threadIdx.x==0){refWall=wall;refMean=mean;}
 })==0,"reference actual publisher then group accepted");
 pairedCallbackCalls=0;
 require(launch8([&]{relaxColdWall1DParticlesToResidentGasKernel(&actual,dt);})==0,"complete 1D paired wrapper accepted");
 require(pairedCallbackCalls==1,"1D paired callback called exactly once");
 require(published>0.&&proposal.specificHeat>0.,"nonzero paired gas source accepted");
 near(actual.heatBudget[0],published,2.e-20,"complete 1D callback publishes one actual gas budget");
 for(int j=0;j<8;++j)near(actual.coldH[j],reference.coldH[j],0.,"1D gas source precedes same actual wall solve");
 near(actual.pThermalEnthalpyResidual[0],reference.pThermalEnthalpyResidual[0],0.,"1D source order preserves identical carry");
 near(actual.deposited[0],double(GpuReal(oldMultiplicity*refWall)),0.,"1D wall ledger uses original frozen material mass");
 near(actual.pT[0],refMean,0.,"1D complete wrapper publishes true mean cache");
 const double change=double(actual.pm[0])*(particleEffectiveSpecificEnthalpyDevice(actual,0)-before);
 near(change+actual.deposited[0]-actual.heatBudget[0],0.,4.e-17,"1D physical particle wall and gas inventory closure");
}
long double mixed_inventory(const DeviceState& s) {
 long double e=static_cast<long double>(pairedGasPhysicalVolume(s,0))*s.rhoE[0];
 for(int i=0;i<3;++i)e+=static_cast<long double>(s.pm[i])*(static_cast<long double>(particleMaterialEnthalpyBaseDevice(s,i))+(particleColdProfileNodeCountDevice(s,i)==8?0.:s.pThermalEnthalpyResidual[i]));
 for(int f=0;f<3;++f)e+=s.reflected[f]+s.deposited[f];
 return e;
}
void mixed_cell(double dt,bool nearFloor=false,bool counterflow=false,int coldest=0) {
 DeviceState s;s.particleCount=3;s.particleCapacity=3;s.nFaces=3;
 s.couplingTgasOld[0]=GPU_R(4200);s.V[0]=GPU_R(1.e-8);s.epsGPrev[0]=GPU_R(.9);
 const double cv=double(s.Rgas)/(double(s.gammaGas)-1.);
 s.rhoE[0]=double(s.rho[0])*cv*double(s.couplingTgasOld[0]);
 if(nearFloor)s.rhoE[0]=pairedGasEnergyFloor(s,0)+1.e-3;
 for(int i=0;i<3;++i){
  s.pStatus[i]=1;s.pCellId[i]=0;s.pStuck[i]=particleWallDeposited;
  s.pStuckFaceId[i]=i==0?-1:i;s.particleStuckCandidateMask[i]=i==2?particleWallColdWall2D:particleWallSolidifyingDeposition;
  s.pT[i]=counterflow?(i==0?GPU_R(5000):(i==1?GPU_R(4800):GPU_R(2800))):(i==0?GPU_R(2800):(i==1?GPU_R(3200):GPU_R(2336.9)));s.pd[i]=GPU_R(1.e-4)*(GPU_R(1)+GPU_R(.1)*i);
  const GpuReal vol=(finiteContactPi/GPU_R(6))*s.pd[i]*s.pd[i]*s.pd[i];s.pm[i]=GPU_R(7)*liquidAluminaProperties(s.pT[i]).densityKgM3*vol;
  s.pThermalEnthalpyResidual[i]=.07125;s.pContactMaximumArea[i]=2.2e-8f;s.pContactDuration[i]=.01f;
  s.pContactPeakFraction[i]=.4f;s.pDepositionArea[i]=i?1.e-8f:0.f;s.pContactAge[i]=.008;
  s.particleWallContactAreaScale[i]=GPU_R(1);s.gasBoundaryT[i]=GPU_R(2200);
 }
 initialiseColdWallParticleState(s,1,s.pT[1]);initialiseColdWall2DParticleState(s,2,s.pT[2]);
 if(coldest){
  s.couplingTgasOld[0]=GPU_R(3200);s.V[0]=GPU_R(1);s.rhoE[0]=double(s.rho[0])*cv*double(s.couplingTgasOld[0]);
  if(coldest==1)s.coldH[8]=1.f;
  else {s.coldH[8]=.10001f;for(int j=1;j<8;++j)s.coldH[8+j]=4000000.f;s.pThermalEnthalpyResidual[1]=-.1;}
  s.pT[1]=GpuReal(particleEffectiveTemperatureDevice(s,1));
 }
 const double oldGas=s.rhoE[0];
 const long double before=mixed_inventory(s);double base[3];for(int i=0;i<3;++i)base[i]=particleEffectiveSpecificEnthalpyDevice(s,i);
 threadIdx.x=0;clearPairedParticleHeatKernel(&s);
 for(int i=0;i<3;++i){threadIdx.x=i;reducePairedParticleHeatKernel(&s,dt);}
 if(counterflow){
  require(pairedParticleHeatCandidate(s,0,dt).specificHeat<0.&&pairedParticleHeatCandidate(s,1,dt).specificHeat<0.,"actual ordinary and1D negative heat credit");
  require(pairedParticleHeatCandidate(s,2,dt).specificHeat>0.&&s.heatBudget[0]>0.,"positive2D reservation leaves net withdrawal limited by current gas floor");
 }
 const auto coldCandidate=pairedParticleHeatCandidate(s,1,dt);
 if(coldest==1)near(coldCandidate.specificHeat,ugkwp::coldWallUniformCoolingBound(s.coldH+8,8),1.e-15,"negative credit uses quantization-safe cooling bound");
 threadIdx.x=0;acceptPairedParticleHeatKernel(&s);
 require(s.heatCoefficient[0]>0.&&s.heatCoefficient[0]<=1.,"mixed cell shared scale accepted");
 if(nearFloor)require(s.heatCoefficient[0]<1.e-3,"current gas floor limits old-cache hot proposals");
 actualPublishPairedParticleGasHeat(s,0,dt);const double ordinaryBudget=s.heatBudget[0];
 require(counterflow?ordinaryBudget<0.:ordinaryBudget>0.,"ordinary particle accepted signed gas heat");
 // Generic publication deliberately skips both cold routes here: 1D publishes
 // inside its wrapper; the actual generic 2D publisher must be an identity.
 const double twoBefore=particleEffectiveSpecificEnthalpyDevice(s,2);
 actualPublishPairedParticleGasHeat(s,2,dt);near(s.heatBudget[0],ordinaryBudget,0.,"2D generic callback is identity");
 near(particleEffectiveSpecificEnthalpyDevice(s,2),twoBefore,0.,"2D generic callback preserves physical profile authority");
 double rawUniformAccepted=0.;
 for(int j=0;j<8;++j){const double old=s.coldH[8+j],stored=old+s.heatCoefficient[0]*coldCandidate.specificHeat;rawUniformAccepted+=(stored-old)/8.;}
 s.wallBoundCount=1;s.wallBoundParticleIndex[0]=1;pairedCallbackCalls=0;
 require(launch8([&]{relaxColdWall1DParticlesToResidentGasKernel(&s,dt);})==0,"mixed cell full 1D wrapper");
 require(pairedCallbackCalls==1,"mixed cell 1D callback once");const double after1d=s.heatBudget[0];require(counterflow?after1d<ordinaryBudget:after1d>ordinaryBudget,"1D contributes accepted signed gas heat");
 if(coldest)near(after1d-ordinaryBudget,double(s.pm[1])*rawUniformAccepted,2.e-20,"raw rounded node change publishes in mixed cell");
 s.wallBoundParticleIndex[0]=2;
 require(launch8([&]{relaxColdWall2DParticlesToResidentGasKernel(&s,dt);})==0,"mixed cell full 2D wrapper");
 require(s.heatBudget[0]>after1d,"2D contributes actual top-boundary gas heat");
 long double accepted=0.;for(int i=0;i<3;++i)accepted+=static_cast<long double>(s.pm[i])*(particleEffectiveSpecificEnthalpyDevice(s,i)-base[i]);
 for(int f=0;f<3;++f)accepted+=s.reflected[f]+s.deposited[f];
 const double rawResidual=double(accepted)-s.heatBudget[0];require(std::isfinite(rawResidual),"mixed raw balance finite");std::cout<<"MIXED_RAW_RESIDUAL "<<rawResidual<<"\n";
 threadIdx.x=0;settlePairedParticleHeatKernel(&s);
 const double gasRoundoff=2.*(std::nextafter(oldGas,INFINITY)-oldGas)*pairedGasPhysicalVolume(s,0);
 near(double(mixed_inventory(s)-before),rawResidual,gasRoundoff+3.e-16,"mixed total inventory exposes actual raw residual");
 if(coldest)for(int j=0;j<8;++j)require(double(s.coldH[8+j])+s.pThermalEnthalpyResidual[1]>0.,"mixed coldest effective node remains positive");
 require(s.rhoE[0]>=pairedGasEnergyFloor(s,0),"mixed gas floor preserved");
 require(s.Tgas[0]>s.TgasMin,"mixed actual gas recovery ran");
}

void extended_carry() {
 for(double sign:{-1.,1.}) {
  double nodes[8];std::fill(nodes,nodes+8,sign*4000000.f);
  auto tiny=ugkwp::publishColdWallUniformSpecificHeat(nodes,8,1.e-10);
  require(tiny.valid&&tiny.acceptedSpecificHeat==0.,"raw sub-ULP heat is honestly zero");
  std::fill(nodes,nodes+8,sign*double(DBL_MAX));auto bad=ugkwp::publishColdWallUniformSpecificHeat(nodes,8,sign*double(DBL_MAX));
  require(!bad.valid,"unrepresentable node target follows finite-storage policy");
 }
 require(!ugkwp::publishColdWallUniformSpecificHeat(nullptr,8,1.).valid,"missing nodes rejected by existing publisher");
 double nodes[8]{};nodes[3]=NAN;
 require(!ugkwp::publishColdWallUniformSpecificHeat(nodes,8,0.).valid,"nonfinite node remains visible");
}
void finite_policy_paths() {
 DeviceState s;seed(s);s.pColdFrozenArea[0]=3.e-8f;
 s.pThermalEnthalpyResidual[0]=-.125;
 for(int j=0;j<8;++j)s.coldH[j]=-140000.f;
 GpuReal mean=-999.,frozen=-999.,wall=-999.;
 require(launch8([&]{GpuReal m=-999.,f=-999.,w=-999.;
  bool ok=advanceColdWall1DThermalGroup(s,0,threadIdx.x,255,GPU_R(1.e-12),GPU_R(3.e-9),
    GPU_R(2.e-8),GPU_R(1.e-8),.01,GPU_R(1.1),0.,GPU_R(-300),GPU_R(10000),GPU_R(1.01),GPU_R(-10),GPU_R(0),m,f,w);
  if(!ok)throw DeviceTrap{};if(threadIdx.x==0){mean=m;frozen=f;wall=w;}
 })==0,"finite outside-old zero interval publishes complete actual output");
 require(mean<0.&&frozen==s.pColdFrozenArea[0]&&wall==0.,"actual outside profile/history retained");
 s.coldWallSolidificationEnabled=0;initialiseColdWallParticleState(s,0,GPU_R(-100));
 for(int j=0;j<8;++j)near(s.coldH[j],double(GPU_R(-100)*s.coldWallSolidificationParameters.solidSpecificHeatJkgK),.02,"existing destination initialized despite enabled identity");
 const auto cap=evaluateCapillaryDetachmentState(GPU_R(-100),GPU_R(1.e-4),GPU_R(-1),GPU_R(.5));
 require(cap.valid&&cap.adhesionSpecificEnergyJkg<0.&&cap.equilibriumContactAreaM2>0.,"capillary complete finite signed energy");
 const auto damage=applyCapillaryContactDamage(cap,cap.equilibriumContactAreaM2,GPU_R(1.01),GPU_R(1));
 require(damage.valid&&damage.detached&&damage.residualSpecificEnergyJkg>GPU_R(1),"signed capillary actual damage transition");
}

void contact_continuation() {
 const GpuReal a=GPU_R(2.e-8),r=GPU_R(1.e-7),eff=GPU_R(10000);
 const GpuReal unit=wallInterfaceConductanceTimeIntegral(a,GPU_R(1),.001,.002,r,GPU_R(0),eff,false);
 const GpuReal wide=wallInterfaceConductanceTimeIntegral(a,GPU_R(1.01),.001,.002,r,GPU_R(0),eff,false);
 near(wide,GPU_R(1.01)*unit,1.e-10,"interface finite efficiency actual equation");
 const GpuReal signedQ=wallInterfaceConductanceTimeIntegral(a,GPU_R(-.5),.001,.002,r,GPU_R(0),eff,false);
 near(signedQ,-GPU_R(.5)*unit,1.e-10,"signed interface conductance");
 const GpuReal integral=finiteContactWallConductanceTimeIntegral(a,a,.01,GPU_R(1.1),.009,.0001,GPU_R(1.01),r,GPU_R(0),eff,false);
 require(std::isfinite(integral)&&integral>0.,"finite contact complete ring integral outside old metadata");
 auto impact=evaluateFiniteWallContactImpact(GPU_R(-100),GPU_R(1.e-4),GPU_R(-10),GPU_R(.5));
 require(impact.valid&&impact.betaMax>0.&&impact.maximumAreaM2>0.&&impact.physicalMassKg>0.&&impact.contactDurationS!=0.,"finite signed impact publishes actual full fields");
 const auto material=liquidAluminaProperties(GPU_R(-100));
 near(impact.weber,material.densityKgM3*GPU_R(100)*GPU_R(1.e-4)/material.surfaceTensionNM,GPU_R(.001),"impact actual Weber");
 const auto cap=evaluateCapillaryDetachmentState(GPU_R(-100),GPU_R(1.e-4),GPU_R(-1),GPU_R(.5));
 require(cap.valid&&cap.adhesionSpecificEnergyJkg<0.,"signed capillary producer");
 DeviceState s;seed(s);s.pStuck[0]=particleWallMobile;s.pT[0]=GPU_R(-100);s.pm[0]=GPU_R(2.1e-8);s.pThermalEnthalpyResidual[0]=.07125;
 auto candidate=particleFiniteContactThermalCandidateDevice(s,0,GPU_R(-200),GPU_R(3.e-9),GPU_R(2.1e-8),GPU_R(1.e-10),0);
 require(candidate.valid&&std::isfinite(candidate.wallEnergyJ),"carried finite-wall signed old state continues");
 const double old=particleEffectiveSpecificEnthalpyDevice(s,0);
 commitParticleFiniteContactThermalCandidateDevice(s,0,candidate,0);
 near(double(s.pm[0])*(particleEffectiveSpecificEnthalpyDevice(s,0)-old),0.,1.e-12,"outside-old outward limiter retains zero-containing interval");
 require(!std::isfinite(wallInterfaceConductanceTimeIntegral(a,GPU_R(1),.001,.002,GPU_R(0),GPU_R(0),eff,false)),"actual zero resistance division error");
 require(!evaluateFiniteWallContactImpact(GPU_R(3000),GPU_R(1.e-4),GPU_R(0),GPU_R(.5)).valid,"actual undefined zero-speed duration error");
}



void geometry_arithmetic() {
 for(GpuReal area:{GPU_R(0),GPU_R(-1)}) {
  const auto g=coldWall2DGeometry(GPU_R(1),GPU_R(1),area);
  require(g.valid,"finite geometry inputs retain original formula outcome");
  for(double dt:{0.,1.e-10}) {
   DeviceState s;seed(s);std::fill(s.cold2DH,s.cold2DH+64,4000000.f);const double beforeCarry=s.pThermalEnthalpyResidual[0];
   const int failures=launch8([&]{GpuReal m=-999,t=-999,f=-999,w=-999;
    if(!advanceColdWall2DThermalGroup(s,0,threadIdx.x,255,GPU_R(1.e-12),GPU_R(3.e-9),area,GPU_R(1.e-8),dt,GPU_R(300),GPU_R(10000),GPU_R(1.e-5),GPU_R(4200),GPU_R(0),m,t,f,w))throw DeviceTrap{};
   });require(failures==0,"consumed backing permits terminal geometry arithmetic");
   if(dt==0.)require(s.cold2DH[0]==4000000.f&&s.pThermalEnthalpyResidual[0]==beforeCarry,"zero interval preserves original profile");
   else require(!std::isfinite(s.pThermalEnthalpyResidual[0]),"undefined geometry remains visible in represented enthalpy");
  }
 }
 const auto signedGeometry=coldWall2DGeometry(GPU_R(-1),GPU_R(-1),GPU_R(1));
 require(signedGeometry.valid&&signedGeometry.filmThicknessM<0&&signedGeometry.nodeMassKg<0,"finite signed geometry executes existing divisions");
}
void resistance_arithmetic() {
 for(int which=0;which<6;++which) {
  DeviceState s;seed(s);auto c=law(s);for(int j=0;j<64;++j)s.cold2DH[j]=float(c.enthalpy(1000.));
  for(int r=0;r<8;++r)s.cold2DAge[r]=.001;
  GpuReal eff=GPU_R(10000),area=GPU_R(1.e-8);double dt=1.e-10;
  if(which==1||which==4||which==5)s.coldWallSolidificationParameters.solidThermalConductivityWmK=GPU_R(0);
  if(which==2||which==4||which==5){s.coldWallSolidificationParameters.wallTransientResistance=1;eff=GPU_R(0);}
  if(which==3){s.coldWallSolidificationParameters.wallTransientResistance=1;s.cold2DAge[0]=-.001;}
  if(which==4)area=GPU_R(0);if(which==5)dt=0.;
  GpuReal wall=-999;
  const int failures=launch8([&]{GpuReal m=-999,t=-999,f=-999,w=-999;
   if(!advanceColdWall2DThermalGroup(s,0,threadIdx.x,255,GPU_R(1.e-12),GPU_R(3.e-9),GPU_R(2.e-8),area,dt,GPU_R(900),eff,GPU_R(1.e-5),GPU_R(1100),GPU_R(0),m,t,f,w))throw DeviceTrap{};
   if(threadIdx.x==0)wall=w;
  });
  require(failures==0,"terminal resistance arithmetic consumes existing backing",failures,which);
  if(which==3)require(!std::isfinite(wall)&&!std::isfinite(s.pThermalEnthalpyResidual[0]),"invalid active square root propagates to physical outputs");
  if(which==0)require(wall>0,"ordinary actual wet wall heat remains nonzero");
  if(which>=4)require(wall==0.,"unused wall operator has actual zero wall heat");
 }
}
template<bool two> void complete_group_layout_one() {
 {
  std::vector<double> reference;
  for(int block:{32,8,16,24}) {
   DeviceState s;seed(s);blockDim.x=block;
   s.particleCount=s.particleCapacity=s.wallBoundCount=s.nFaces=3;s.particleHeatCoefficient[0]=.3;
   const auto c=law(s);double old[3];
   for(int i=0;i<3;++i){
    s.wallBoundParticleIndex[i]=i;s.pCellId[i]=0;s.pStuckFaceId[i]=i;s.pStatus[i]=1;
    s.pStuck[i]=particleWallDeposited;s.pDepositionArea[i]=1.e-8f;
    s.particleStuckCandidateMask[i]=two?particleWallColdWall2D:particleWallSolidifyingDeposition;
    s.particleWallContactAreaScale[i]=GPU_R(1);s.gasBoundaryT[i]=GPU_R(2200);
    s.pd[i]=s.pd[0];s.pm[i]=s.pm[0];s.pT[i]=GPU_R(3200);s.pThermalEnthalpyResidual[i]=two?.07125:0.;
    s.pContactDuration[i]=.01f;s.pContactMaximumArea[i]=2.2e-8f;s.pContactPeakFraction[i]=.4f;
    s.pContactAge[i]=.008;s.pColdContactAge[i]=.008;
    for(int j=0;j<(two?64:8);++j)([&]{if constexpr(two)return s.cold2DH;else return s.coldH;}())[i*(two?64:8)+j]=float(c.enthalpy(3200.));
    old[i]=inventory(([&]{if constexpr(two)return s.cold2DH;else return s.coldH;}())+i*(two?64:8),two?64:8,s.pThermalEnthalpyResidual[i]);
   }
   int traps=0;
   for(int group=0;group<block/8;++group)traps+=launch8([&]{threadIdx.x+=group*8;if(two)relaxColdWall2DParticlesToResidentGasKernel(&s,1.e-10);else relaxColdWall1DParticlesToResidentGasKernel(&s,1.e-10);});
   require(traps==0,"every complete eight-lane group executes",traps,block);
   std::vector<double> fields;double ledger=0.;
   for(int i=0;i<3;++i){
    auto* h=([&]{if constexpr(two)return s.cold2DH;else return s.coldH;}())+i*(two?64:8);const int n=two?64:8;
    const double dh=inventory(h,n,s.pThermalEnthalpyResidual[i])-old[i];
    require(dh!=0.&&s.deposited[i]!=0.,"every directory particle receives its actual interval and wall heat");
    ledger+=double(s.pm[i])*dh+double(s.deposited[i]);
    for(int j=0;j<n;++j)fields.push_back(h[j]);
    for(int j=0;j<8;++j)fields.push_back(two?s.cold2DAge[i*8+j]:double(s.coldRing[i*8+j]));
    for(double v:{double(s.pT[i]),s.pThermalEnthalpyResidual[i],double(s.coldFrozen[i]),double(s.pCold2DFrozenArea[i]),s.coldAge[i],s.pContactAge[i],double(s.deposited[i]),double(s.reflected[i]),double(s.pux[i]),double(s.puy[i]),double(s.puz[i]),double(s.pStuck[i])})fields.push_back(v);
   }
   if(two)near(ledger,s.heatBudget[0],1.2e-16,"legacy2D inventory/wall changes settle gas once");else {require(std::isfinite(ledger-s.heatBudget[0]),"raw1D layout residual finite");fields.push_back(ledger-s.heatBudget[0]);}fields.push_back(s.heatBudget[0]);
   if(block==32)reference=fields;else require(fields==reference,"complete layout fullfield parity against block32");
  }
 }
 blockDim.x=32;
}
void missing_gas_ledger() {
 DeviceState s;seed(s);s.particleHeatBudget=nullptr;
 for(int j=0;j<64;++j)s.cold2DH[j]=4000000.f;
 const int traps=launch8([&]{GpuReal m=-999.,t=-999.,f=-999.,w=-999.;
  advanceColdWall2DThermalGroup(s,0,threadIdx.x,255,GPU_R(1.e-12),GPU_R(3.e-9),GPU_R(2.e-8),GPU_R(1.e-8),1.e-10,GPU_R(300),GPU_R(10000),GPU_R(1.e-5),GPU_R(4200),GPU_R(1.e-5),m,t,f,w);
 });
 require(traps==1,"actual required gas settlement address fails on its leader lane");
}
template<bool twoD> void large_group_identity_one() {
 for(double sign:{-1.,1.}) {
  DeviceState s;seed(s);s.pThermalEnthalpyResidual[0]=twoD?sign*4.*double(FLT_MAX):0.;
  auto* nodes=[&]{if constexpr(twoD)return s.cold2DH;else return s.coldH;}();int count=twoD?64:8;
  std::fill(nodes,nodes+count,sign*4000000.f);
  const double before=inventory(nodes,count,s.pThermalEnthalpyResidual[0]);
  const double oldCarry=s.pThermalEnthalpyResidual[0];
  require(launch8([&]{GpuReal m=-999.,t=-999.,f=-999.,w=-999.;bool ok;
   if(twoD)ok=advanceColdWall2DThermalGroup(s,0,threadIdx.x,255,GPU_R(1.e-12),GPU_R(3.e-9),GPU_R(2.e-8),GPU_R(1.e-8),0.,GPU_R(-300),GPU_R(10000),GPU_R(1.01),GPU_R(4200),GPU_R(0),m,t,f,w);
   else ok=advanceColdWall1DThermalGroup(s,0,threadIdx.x,255,GPU_R(1.e-12),GPU_R(3.e-9),GPU_R(2.e-8),GPU_R(1.e-8),.01,GPU_R(.4),0.,GPU_R(-300),GPU_R(10000),GPU_R(1.01),GPU_R(4200),GPU_R(0),m,f,w);
   if(!ok)throw DeviceTrap{};
   near(m,GpuReal(law(s).temperature(before)),0.,"large actual effective group temperature");require(w==0.,"zero interval actual zero wall heat");
  })==0,"large signed effective identity actual group");
  require(s.pThermalEnthalpyResidual[0]==oldCarry&&inventory(nodes,count,oldCarry)==before,"zero interval preserves full large signed inventory");
 }
 DeviceState missing;seed(missing);missing.pColdNodeSpecificEnthalpy=nullptr;
 bool trapped=false;try{initialiseColdWallParticleState(missing,0,GPU_R(300));}catch(DeviceTrap&){trapped=true;}
 require(trapped,"actual absent initializer destination fails");
}
#include "cold_wall_2d_continuation.hpp"
int main(int argc,char** argv) {
 require(argc==2,"case required");std::string k=argv[1];
 if(k=="geometry_arithmetic")geometry_arithmetic();
 else if(k=="resistance_arithmetic")resistance_arithmetic();
 else if(k=="complete_group_layout"){complete_group_layout_one<false>();complete_group_layout_one<true>();}
 else if(k=="missing_gas_ledger")missing_gas_ledger();
 else if(k=="wide1d")full_1d_wrapper(1.e-10,true);
 else if(k=="large_group_identity"){large_group_identity_one<false>();large_group_identity_one<true>();}
 else if(k=="signedwall2d")full_2d_interval(false,false,false,true);
 else if(k=="signedgas2d")full_2d_interval(false,false,true);
 else if(k=="reverse2d")full_2d_interval(false,true);
 else if(k=="contact_continuation")contact_continuation();
 else if(k=="interval2d")full_2d_interval(false);
 else if(k=="tail2d")full_2d_interval(true);
 else if(k=="extended_carry")extended_carry();
 else if(k=="finite_policy_paths")finite_policy_paths();
 else if(k=="publication_stress")publication_stress();
 else if(k=="coldest_positive")coldest_positive_roundoff();
 else if(k=="coldest_roundoff")coldest_node_roundoff();
 else if(k=="mixed_coldest")mixed_cell(1.e-5,false,true,1);
 else if(k=="mixed_coldest_positive")mixed_cell(1.e-10,false,false,2);
 else if(k=="wrapper1d_micro")full_1d_wrapper(1.e-10);
 else if(k=="wrapper1d_finite")full_1d_wrapper(1.e-3);
 else if(k=="mixed_micro")mixed_cell(1.e-10);
 else if(k=="mixed_finite")mixed_cell(1.e-5);
 else if(k=="mixed_gas_floor")mixed_cell(1.e-5,true);
 else if(k=="mixed_counterflow_floor")mixed_cell(1.e-5,true,true);
 else if(k=="wrapper2d_zero")full_2d_wrapper(0.);
 else if(k=="wrapper2d_scaled")full_2d_wrapper(.3);
 else if(k=="wrapper2d_full")full_2d_wrapper(1.);
 else if(k=="uniform8")pure_profile(8);
 else if(k=="uniform64")pure_profile(64);
 else if(k=="no_duplicate")no_duplicate_gas();
 else {bool two=k.find("2d")!=std::string::npos;double dt=k.find("zero")!=std::string::npos?0.:
  (k.find("micro")!=std::string::npos?1.e-10:1.e-5);
  double t=k.find("mushy")!=std::string::npos?2327.:3200.;
  if(k.find("liquidus")!=std::string::npos)t=2336.9999;
  if(k.find("solidus")!=std::string::npos)t=2317.0001;
  if(two)kernel_profile<true>(dt,t,k.find("gradient")!=std::string::npos,k.find("gas")!=std::string::npos,k.find("cooling")!=std::string::npos,k.find("null")!=std::string::npos);else kernel_profile<false>(dt,t,k.find("gradient")!=std::string::npos,k.find("gas")!=std::string::npos,k.find("cooling")!=std::string::npos,k.find("null")!=std::string::npos);}
}
'''

@pytest.fixture(scope="module")
def cold_carry_binary(tmp_path_factory):
    build=tmp_path_factory.mktemp(f"cold_carry_64")
    (build/'cuda_runtime.h').write_text('#pragma once\n#define __host__\n')
    fixture=((VALIDATION_ROOT / 'tests/fixtures/cold_wall_contact_routes.cpp')).read_text()
    prefix=fixture[:fixture.index('#include "GpuThermalParticleRelaxation.cuh"')]
    prefix=prefix.replace('#include <thread>','#include <thread>\n#include <string>\n#include "GpuColdWall2DSolidification.H"\n#define GPU_PAIRED_PARTICLE_HEAT 1')
    prefix=prefix.replace('struct DeviceState {','struct DeviceState {\n double particleContactHeatCandidate[1]{};unsigned char particleContactHeatEpoch[1]{};\n double pThermalEnthalpyResidual[1]{}; double heatBudget[1]{}; double heatCoefficient[1]{1.}; double* particleHeatBudget=heatBudget; double* particleHeatCoefficient=heatCoefficient;\n int coldWall2DEnabled=1;\n float cold2DH[64]{}; GpuTime cold2DAge[8]{};\n float* pCold2DNodeSpecificEnthalpy=cold2DH;\n GpuTime* pCold2DRingContactAge=cold2DAge;')
    prefix=prefix.replace('using ProfileStorage=float;', 'using ProfileStorage=double;')
    prefix=prefix.replace('float coldH[8]', 'double coldH[8]').replace('float* pColdNodeSpecificEnthalpy', 'double* pColdNodeSpecificEnthalpy')
    prefix=prefix.replace('#include "operators/clearColdWallParticleState.cuh"', '#include "wall/GpuColdWall1DHandles.cuh"\n#include "ParticleResidentInterval.H"\n#include "operators/clearColdWallParticleState.cuh"')
    prefix=prefix.replace('struct DeviceTrap {};','struct DeviceTrap {};\nvoid __syncwarp(unsigned){lanes.arrive_and_wait();}')

    prefix=prefix.replace('T atomicAdd(T* target,T value) {', 'T atomicAdd(T* target,T value) {if(!target)throw DeviceTrap{};')
    prefix=prefix.replace('struct DeviceState {', 'struct DeviceState {\n GpuReal epsGPrev[1]{GPU_R(.9)},V[1]{GPU_R(1)},rho[1]{GPU_R(1)},rhoUx[1]{},rhoUy[1]{},rhoUz[1]{};\n GpuReal Tgas[1]{GPU_R(4200)},p[1]{},Ux[1]{},Uy[1]{},Uz[1]{};\n GpuReal Rgas=GPU_R(287),gammaGas=GPU_R(1.4);\n double rhoE[1]{3.e6},rhoENext[1]{};\n unsigned char* compactPStuck=pStuck;int* compactPStuckFaceId=pStuckFaceId;GpuReal* compactPT=pT;\n double* compactPColdNodeSpecificEnthalpy=coldH;float* compactPCold2DNodeSpecificEnthalpy=cold2DH;\n double* compactPThermalEnthalpyResidual=pThermalEnthalpyResidual;\n')
    # All three particle kinds share one cell; face and parcel arrays have
    # independent slots. Default single-particle cases still use slot zero.
    prefix=prefix.replace('laneValues[threadIdx.x]','laneValues[threadIdx.x&7]').replace('laneValues[target]','laneValues[target&7]')
    prefix=prefix.replace('threadIdx.x >= offset ? threadIdx.x-offset : threadIdx.x','(threadIdx.x&7) >= offset ? (threadIdx.x&7)-offset : (threadIdx.x&7)')
    prefix=prefix.replace('threadIdx.x+offset < 8 ? threadIdx.x+offset : threadIdx.x','(threadIdx.x&7)+offset < 8 ? (threadIdx.x&7)+offset : (threadIdx.x&7)')
    prefix=prefix.replace('    return result;\n}\nbool __all_sync','    return result << (threadIdx.x&24);\n}\nbool __all_sync')
    prefix=prefix.replace('[1]','[3]').replace('coldH[8]','coldH[24]').replace('coldRing[8]','coldRing[24]').replace('cold2DH[64]','cold2DH[192]').replace('cold2DAge[8]','cold2DAge[24]')
    prefix=prefix.replace('#include \"GpuColdWall1DDevice.cuh\"', '#include "wall/GpuColdWall1DHandles.cuh"\n#include "GpuParticleThermalCarry.H"\n#include "GpuPairedParticleHeat.H"\n#include "GpuColdWallThermalCarry.cuh"\n#include "GpuParticleCaloricDevice.cuh"\nconstexpr GpuReal OfSmall=GPU_R(2.22044604925031308085e-16);\n#include "actual_gas_recovery.cuh"\n#define publishPairedParticleGasHeat actualPublishPairedParticleGasHeat\n#include "GpuPairedParticleHeat.cuh"\n#include "paired_heat_reference.cuh"\n#undef publishPairedParticleGasHeat\nstd::atomic<int> pairedCallbackCalls{0};\nvoid publishPairedParticleGasHeat(DeviceState& s,int i,GpuTime dt) {\n ++pairedCallbackCalls;actualPublishPairedParticleGasHeat(s,i,dt);\n}\n#include "GpuColdWall1DDevice.cuh"')
    prefix = prefix.replace('struct DeviceState {', '#include "Diagnostics.H"\nstruct DeviceState {\n ugkp::diagnostics::DeviceContext* diagnostics=nullptr;\n int particleFiniteContactMaximumIterations=64;\n')
    recovery=(ROOT/'common/operators/computeSstFaceFluxKernel.cuh').read_text()
    start=recovery.index('template<class GasState>\n__device__ void recoverGasPrimitiveCell')
    body=recovery.index('{',start);depth=1;end=body+1
    while depth:
        depth+=(recovery[end]=='{')-(recovery[end]=='}');end+=1
    (build/'actual_gas_recovery.cuh').write_text('#include \"gasTransport/GasCapabilities.H\"\n#include \"gasTransport/MixtureThermo.H\"\n#include \"gasTransport/GasGeometryValidation.H\"\n'+'#include "operators/gasStencilDiagnostics.cuh"\n'+recovery[recovery.index('template<class GasState,class Scalar>'):recovery.index('template<class GasState>\n__device__ void sstInternalRhoDiffusivities')]+recovery[start:end])
    prefix=prefix.replace('void clearColdWall2DParticleState(DeviceState&,int) {}','void clearColdWall2DParticleState(DeviceState&,int);')
    two=(ROOT/'legacy/common/wall/GpuColdWall2DDevice.cuh').read_text()

    two=two.replace('        ColdWallThermalReal power[', '        if(radialNode==0)observedStepTimes.push_back(stepTime);\n        ColdWallThermalReal power[')
    prefix += '\n#include "GpuFrozenParticleRelease.cuh"\n#include <vector>\nstd::vector<double> observedStepTimes;\n'
    # Compile the complete production 2D wrapper as well as its group.
    (build/'actual_2d_group.cuh').write_text(two)
    helpers=fixture[fixture.index('void require('):fixture.index('double meanH(')]
    (build/'test.cpp').write_text(prefix+'namespace ugkwp {\ninline float coldWallStoredEnthalpyFloor(double target) {\n if(target>double(FLT_MAX))return FLT_MAX;if(target<-double(FLT_MAX))return -FLT_MAX;\n float stored=static_cast<float>(target);if(double(stored)>target)stored=::nextafterf(stored,-INFINITY);return stored;\n}\n}\n'+'\n#include "actual_2d_group.cuh"\n#include "GpuParticleThermalConsumerDevice.cuh"\n'+helpers+HARNESS)
    (build/'route_conductivity.cuh').write_text('#include "gasTransport/GasMolecularConductivity.cuh"\n')
    binary=build/'test'
    run=subprocess.run(['g++','-std=c++20','-O2','-pthread','-DROUTE_FSH=0','-I',str((VALIDATION_ROOT / 'tests/fixtures')),'-I',str(build),'-I',str(ROOT/'common'),'-I',str(ROOT/'common/wall'),'-I',str((VALIDATION_ROOT / 'tests/gas_transport')),str(build/'test.cpp'),'-o',str(binary)],capture_output=True,text=True,timeout=90)
    assert run.returncode==0,run.stdout+run.stderr
    return binary

@pytest.mark.parametrize('case',['geometry_arithmetic','resistance_arithmetic','complete_group_layout','missing_gas_ledger','wide1d','large_group_identity','signedwall2d','signedgas2d','reverse2d','contact_continuation','interval2d','tail2d','extended_carry','finite_policy_paths','publication_stress','coldest_positive','coldest_roundoff','mixed_coldest','mixed_coldest_positive','wrapper1d_micro','wrapper1d_finite','mixed_micro','mixed_finite','mixed_gas_floor','mixed_counterflow_floor','wrapper2d_zero','wrapper2d_scaled','wrapper2d_full','uniform8','uniform64','no_duplicate',
 '1d_zero','1d_micro','1d_finite','1d_mushy','1d_gradient','2d_zero','2d_null','2d_micro','2d_finite','2d_mushy','2d_gradient','2d_gas','2d_micro_gas','2d_mushy_gas','2d_liquidus_gas','2d_solidus_gas_cooling'])
def test_cold_wall_mean_authority(cold_carry_binary,case):
    run=subprocess.run([str(cold_carry_binary),case],capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stdout+run.stderr


def test_cold_carry_schedule_source_contract():
    inputs=(ROOT/"common/wall/GpuColdWall1DInputs.inl").read_text()
    assert inputs.count("publishPairedParticleGasHeat(s, i, dt)")==1
    first_sync=inputs.index("__syncwarp(mask)")
    second_sync=inputs.index("__syncwarp(mask)",first_sync+1)
    assert inputs.index("const GpuReal physicalMass")<first_sync<inputs.index("publishPairedParticleGasHeat(s, i, dt)")<second_sync<inputs.index("GpuReal gasConductanceWK")
    two=(ROOT/"legacy/common/wall/GpuColdWall2DDevice.cuh").read_text()
    assert "localGasEnergyJ += gasPower*stepTime" in two
    assert "s.particleHeatCoefficient[cellI]" in two
    assert "double(s.pm[particleI])*actualSpecificChange" in two
    assert "publishPairedParticleGasHeat" not in two
