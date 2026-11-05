#include "fvCFD.H"
#include "GpuParticleRadiationCoupler.H"
#include "MieDoRadiationMath.H"
#include "GpuParticleCaloric.H"
#include "GpuParticleThermalConsumers.H"
#include "GpuParticleRadiationMath.H"
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
using namespace Foam;
using namespace Foam::gpuThermal;
void check(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
bool near(double a,double b,double tol=1e-11){return std::abs(a-b)<=tol*std::max({1.,std::abs(a),std::abs(b)});}
template<class F> void fails(F f){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed,"required error was swallowed");}
void dumpMieResult(const char* tag,const MieDoResult& r){
 forAll(r.incidentRadiationG_Wm2,i)std::cout<<tag<<"_CELL "<<i<<' '<<r.absorptionCoefficientInvM[i]<<' '<<r.scatteringCoefficientInvM[i]<<' '<<r.extinctionCoefficientInvM[i]<<' '<<r.asymmetryFactor[i]<<' '<<r.incidentRadiationG_Wm2[i]<<' '<<r.particlePowerDensityWm3[i]<<'\n';
 forAll(r.boundaryQrOutwardWm2,p)forAll(r.boundaryQrOutwardWm2[p],f)std::cout<<tag<<"_FACE "<<p<<' '<<f<<' '<<r.boundaryQrOutwardWm2[p][f]<<'\n';
 std::cout<<tag<<"_STATS "<<r.integratedParticlePowerW<<' '<<r.integratedBoundaryPowerW<<' '<<r.conservationResidualW<<' '<<r.sourceIterations<<' '<<r.angleThreadsUsed<<'\n';
}
int main(int argc,char**argv) try {
 const std::string mode=std::getenv("T13_MODE");
 std::cout<<std::setprecision(17);
 if(mode=="root_ordinary"){
  auto f=[](double t){return 1e-3*(std::pow(1200.,4)-std::pow(t,4));};
  auto r=detail::solveBoundedImplicitMaterialTemperature(3600,300,5000,300,.01,200,1e-10,1e-9,f,f);
  std::cout<<"ROOT "<<r.temperatureK<<' '<<r.residualJ<<' '<<r.iterations<<'\n';
 } else if(mode=="root_one" || mode=="root_zero"){
  std::vector<double> evaluated;
  auto power=[&](double t){evaluated.push_back(t);return 1e6-t*t;};
  auto r=detail::solveImplicitMaterialTemperature(300,1000,200,2000,1,1,mode=="root_zero"?0:1,1e-16,1e-16,power);
  check(r.temperatureK==650 && r.residualJ==-577150 && r.iterations==1,"root candidate/residual lineage");
  check(evaluated.size()==3 && evaluated.back()==r.temperatureK,"evaluated root lineage");
 } else if(mode=="root_same_sign"){
  auto power=[](double t){return 1e6+t;};
  auto r=detail::solveImplicitMaterialTemperature(300,1000,200,2000,1,1,1,1e-16,1e-16,power);
  check((r.temperatureK==300 || r.temperatureK==1000) && r.residualJ==(r.temperatureK-300)-power(r.temperatureK),"same-sign endpoint must be measured");
  check(r.residualJ!=0 && r.iterations==0,"same-sign result must not claim exact root");
 } else if(mode=="root_equilibrium"){
  std::vector<double> material;
  auto r=detail::solveBoundedImplicitMaterialTemperature(300,200,1000,1,1,1,1e-30,1e-30,[](double t){return 800-t;},[&](double t){material.push_back(t);return 800-t;});
  check(material.size()==3 && r.iterations==1,"equilibrium exhaustion skipped material solve");
  check(r.temperatureK==650 && r.residualJ==200,"bounded material evaluated candidate");
 } else if(mode=="root_signed"){
  auto r=detail::solveImplicitMaterialTemperature(-300,1000,200,2000,-1,-1,1,-1,-1,[](double t){return 1e6-t*t;});
  check(r.temperatureK==350 && r.residualJ==876850 && r.iterations==1,"defined signed root arithmetic");
 } else if(mode=="root_nonfinite_tolerance"){
  auto power=[](double t){return 1e6-t*t;};
  auto r=detail::solveImplicitMaterialTemperature(300,1000,200,2000,1,1,1,std::numeric_limits<double>::quiet_NaN(),1e-10,power);
  check(r.temperatureK==650 && r.residualJ==-577150 && r.iterations==1,"nonfinite tolerance changed evaluated candidate");
 } else if(mode=="root_errors"){
  auto nanRoot=detail::solveImplicitMaterialTemperature(300,1000,200,2000,1,1,1,1e-10,1e-10,[](double){return std::numeric_limits<double>::quiet_NaN();});check(nanRoot.temperatureK==1000 && std::isnan(nanRoot.residualJ),"actual nonfinite endpoint residual");
  auto overflowRoot=detail::solveImplicitMaterialTemperature(300,1000,200,2000,1e308,1,1,1e-10,1e-10,[](double){return 1.;});check(overflowRoot.temperatureK==650 && std::isinf(overflowRoot.residualJ),"actual overflowing midpoint residual");
 } else if(mode.find("scale_")==0){
  if(mode=="scale_errors"){
   auto zero=detail::conservativeBoundaryEnergyScale(1,0,1,1e-10);check(std::isinf(zero.scale) && std::isnan(zero.residualJ),"actual zero-divisor scale");
   auto overflow=detail::conservativeBoundaryEnergyScale(1,1e-320,1,1e-10);check(std::isinf(overflow.scale) && std::isinf(overflow.residualJ),"actual overflowing scale");
   auto z=detail::conservativeBoundaryEnergyScale(0,0,0,1e-10);check(z.scale==1 && z.residualJ==0,"zero identity");
  }else{
   double a=mode=="scale_negative"?2:mode=="scale_tiny"?1e-20:-2;
   double b=mode=="scale_tiny"?1e-20:1;
   auto r=detail::conservativeBoundaryEnergyScale(a,b,std::abs(a)+std::abs(b),1e-10);
   if(mode=="scale_tiny")check(r.scale==1 && r.residualJ==a+b,"original successful near-zero identity");
   else check(r.scale==-a/b && r.residualJ==a+r.scale*b,"actual boundary division");
  }
 } else if(mode.find("phase_")==0){
  List<scalarField> k(2); k[0]=scalarField(2);k[1]=scalarField(2);
  k[0][0]=4;k[0][1]=1;k[1][0]=mode=="phase_asymmetric"?2:1;k[1][1]=1;
  scalarField w(2,1.),s,n;List<scalarField> p;
  if(mode=="phase_negative_weights"){
   w=-1.;k[0][0]=-4;k[0][1]=-1;k[1][0]=-1;k[1][1]=-1;
   auto m=balanceSymmetricPhaseKernelInPlace(k,w,1e-30,1,s,n,p);
   const double expected[]={.8,1/std::sqrt(10.),1/std::sqrt(10.),.5};
   double rowError=0,columnError=0;
   for(label i=0;i<2;++i){
    double row=0,column=0;
    for(label j=0;j<2;++j){check(near(p[i][j],expected[2*i+j]),"signed-weight actual probability");row+=p[i][j];column+=w[j]*p[j][i];}
    rowError=std::max(rowError,std::abs(row-1));
    columnError=std::max(columnError,std::abs(column-w[i])/std::abs(w[i]));
   }
   check(near(columnError,.18377223398316223),"independent signed-weight witness");
   check(m.maxWeightedColumnRelativeError==columnError && m.maxRowError==rowError && m.iterations==1,"signed-weight metrics hide actual error");
   check(m.maxRelativeScalingCorrection==std::max(std::abs(s[0]-1),std::abs(s[1]-1)),"signed-weight scaling metric");
   std::cout<<"PHASE_SIGNED "<<rowError<<' '<<columnError<<' '<<m.maxWeightedColumnRelativeError<<'\n';
  }else if(mode=="phase_signed"){
   k[0][0]=4;k[0][1]=-1;k[1][0]=-1;k[1][1]=4;
   auto m=balanceSymmetricPhaseKernelInPlace(k,w,1e-12,200,s,n,p);
   check(near(p[0][0],4./3.) && near(p[0][1],-1./3.) && m.maxRowError<1e-12,"finite signed phase computation");
  }else if(mode=="phase_ordinary"){
   auto m=balanceSymmetricPhaseKernelInPlace(k,w,1e-12,200,s,n,p);
   std::cout<<"PHASE "<<p[0][0]<<' '<<p[0][1]<<' '<<p[1][0]<<' '<<p[1][1]<<' '<<m.maxRowError<<' '<<m.maxWeightedColumnRelativeError<<' '<<m.maxRelativeScalingCorrection<<' '<<m.iterations<<'\n';
  }else if(mode=="phase_errors"){
   k[0][0]=std::numeric_limits<double>::quiet_NaN();balanceSymmetricPhaseKernelInPlace(k,w,1e-30,1,s,n,p);check(std::isnan(p[0][0]),"actual NaN phase probability");
   k[0][0]=0;k[0][1]=0;balanceSymmetricPhaseKernelInPlace(k,w,1e-30,1,s,n,p);check(std::isnan(p[0][0]),"actual zero-row probability");
   k[0][0]=4;k[0][1]=1;w[0]=0;balanceSymmetricPhaseKernelInPlace(k,w,1e-30,1,s,n,p);check(p[0][0]==0,"actual zero-weight probability");
   w[0]=std::numeric_limits<double>::quiet_NaN();balanceSymmetricPhaseKernelInPlace(k,w,1e-30,1,s,n,p);check(std::isnan(p[0][0]),"actual NaN-weight probability");
   k[0].setSize(1);fails([&]{balanceSymmetricPhaseKernelInPlace(k,w,1e-30,1,s,n,p);});
  }else if(mode=="phase_nonfinite_tolerance"){
   auto m=balanceSymmetricPhaseKernelInPlace(k,w,std::numeric_limits<double>::quiet_NaN(),1,s,n,p);
   check(m.iterations==1 && near(p[0][0],.8),"nonfinite tolerance withheld actual phase sweep");
  }else{
   auto m=balanceSymmetricPhaseKernelInPlace(k,w,1e-30,mode=="phase_zero"?0:1,s,n,p);
   auto measured=measurePhaseBalance(p,w);
   check(m.iterations==1 && m.maxRowError==measured.maxRowError && m.maxWeightedColumnRelativeError==measured.maxWeightedColumnRelativeError,"phase last metrics");
   check(near(p[0][0],4./5.) && near(p[0][1],1/std::sqrt(5.*(k[1][0]+1))) && near(p[1][1],1./(k[1][0]+1)),"actual phase probabilities");
   check(m.maxRowError>1e-4 && m.maxRelativeScalingCorrection==std::max(std::abs(s[0]-1),std::abs(s[1]-1)),"phase residual was falsified");
  }
 } else if(mode=="table" || mode=="table_interior" || mode=="table_nonfinite"){
  MieTable table(std::getenv("T13_TABLE"));const double queryT=mode=="table_nonfinite"?std::numeric_limits<double>::quiet_NaN():mode=="table_interior"?2500.:4100.;auto v=table.query(1.5e-4,queryT);
  std::cout<<"TABLE "<<v.Qabs<<' '<<v.Qsca<<' '<<v.Qext<<' '<<v.planckBandFraction<<' '<<table.phase(1.5e-4,queryT,0)<<'\n';
 } else {
  #include "setRootCase.H"
  #include "createTime.H"
  #include "createMesh.H"

  (void)mesh.cells();
  MieTable table(std::getenv("T13_TABLE"));
  volScalarField wall(IOobject("T",runTime.timeName(),mesh,IOobject::NO_READ,IOobject::NO_WRITE),mesh,dimensionedScalar("T",dimTemperature,300));
  List<scalarField> emissivity(mesh.boundary().size());forAll(emissivity,i)emissivity[i]=scalarField(mesh.boundary()[i].size(),1.);
  scalarField eps(mesh.nCells(),1e-6),temp(mesh.nCells(),1000),diam(mesh.nCells(),1e-4);
  labelList patches(1,0);
  const bool firstSweep=mode=="do_one" || mode=="do_one_negative_diameter";
  if(mode=="do_one_negative_diameter")diam=-1e-4;
  dictionary config;
  config.add("IBoundaryCondition",word("diffuseGrey"));
  config.add("linearSolverMaxIterations",label(firstSweep?1:mode=="do_zero"?0:200));
  config.add("linearSolverTolerance",scalar(firstSweep?1e-30:1e-12));
  MieDoControls controls{2,4,firstSweep?1:mode=="do_zero"?0:200,1e-12,1e-12,200,0,1e-10,100,1};
  if(mode=="do_negative" || mode=="do_roundoff"){
   config.set("IBoundaryCondition",word("fixedI"));config.add("IFixed",scalar(mode=="do_negative"?-2:-1e-15));eps=0;patches.clear();
  }
  if(mode=="do_signed"){diam=-1e-4;forAll(emissivity,p)emissivity[p]=2.;}
  if(mode=="do_parallel_ordinary")controls.angleParallelThreads=2;
  if(mode=="do_parallel"){controls.angleParallelThreads=2;controls.maxSourceIterations=1;config.set("linearSolverMaxIterations",label(1));config.set("linearSolverTolerance",scalar(1e-30));}
  if(mode=="do_inactive_nan"){eps=0;diam=std::numeric_limits<double>::quiet_NaN();}
  if(mode=="do_nonfinite")temp[0]=std::numeric_limits<double>::quiet_NaN();
  if(mode=="do_missing")diam.setSize(1);
  if(mode=="do_zero_divisor")diam=0;
  if(mode=="do_unknown")config.set("IBoundaryCondition",word("missingSelectedOperator"));
  if(mode=="do_phase_one"){controls.maxPhaseBalanceIterations=1;controls.phaseBalanceTolerance=1e-30;controls.maxPhaseScalingCorrection=0;}
  if(mode=="do_relaxation"){config.add("sourceRelaxation",scalar(1.2));controls.sourceRelativeTolerance=-1;controls.maxSourceIterations=1;}
  if(mode=="transaction_inactive_nan" || mode=="transaction" || mode=="ordinary" || mode=="transaction_padded" || mode=="transaction_exact_diameter" || mode=="transaction_missing" || mode.find("boundary_")==0){
   ugkwp::ParticleCaloric law;ugkwp::ParticleThermalCarry<GpuReal> carried{GpuReal(4000),law.enthalpy(4100)-law.enthalpy(4000)};
   const bool inRange=mode=="transaction_inactive_nan" || mode=="ordinary" || mode.find("boundary_")==0;
   const double old=inRange?3000.:ugkwp::effectiveParticleTemperature(carried,law);
   if(inRange){carried.temperature=3000;carried.specificEnthalpyResidual=0;}
   temp=old;
   const double cp=law.specificHeat(old),massValue=.01,dt=.01;
   scalarField mass(mesh.nCells(),massValue);
   if(mode=="transaction_inactive_nan"){eps=0;diam=std::numeric_limits<double>::quiet_NaN();}
   if(mode=="transaction_exact_diameter")diam=3e-4;
   if(mode=="transaction_missing")mass.setSize(mesh.nCells()-1);
   if(mode=="transaction_padded"){
    eps.setSize(mesh.nCells()+1);temp.setSize(mesh.nCells()+1);diam.setSize(mesh.nCells()+1);mass.setSize(mesh.nCells()+1);
    diam=3e-4;eps[mesh.nCells()]=temp[mesh.nCells()]=diam[mesh.nCells()]=mass[mesh.nCells()]=std::numeric_limits<double>::quiet_NaN();
   }
   const scalarField beforeDiameter(diam),beforeTemperature(temp),beforeMass(mass),beforeEpsilon(eps);
   MieDoRadiationSolver solver(mesh,config,table,patches,controls);
   auto workspace=solver.makeWorkspace();
   detail::ParticleRadiationTransactionContext context{mesh,table,patches,200,4000,0};
   ParticleRadiationMaterialSnapshot snapshot{eps,mass,temp,diam,wall,emissivity,cp,dt};
   double observed=-999;
   double suppliedBoundarySum=0;List<scalarField> suppliedBoundary;
   auto result=detail::solveParticleRadiationTransaction(context,snapshot,[&]{return &workspace;},[&](MieDoRadiationWorkspace*& w,const MieDoSnapshot& s){
    observed=s.particleTemperatureK[0];
    if(mode=="transaction_padded")check(s.representativeDiameterM[0]==table.maxDiameterM(),"available-prefix shape changed diameter clamping");
    auto actual=solver.solve(s,*w);
    if(mode.find("boundary_")==0){


     int index=0;const double energy[]={1e16,-1e16+2,1};
     forAll(actual.boundaryQrOutwardWm2,p)forAll(actual.boundaryQrOutwardWm2[p],f){
      double e=mode=="boundary_overflow"?(index==0?1e-320:0):mode=="boundary_tiny"?(index==0?1e-20:0):mode=="boundary_negative"?(index==0?-1e-6:0):(index<3?energy[index]:0);
      actual.boundaryQrOutwardWm2[p][f]=e/(dt*mag(mesh.Sf().boundaryField()[p][f]));
      suppliedBoundarySum+=dt*mag(mesh.Sf().boundaryField()[p][f])*actual.boundaryQrOutwardWm2[p][f];++index;
     }
     suppliedBoundary=actual.boundaryQrOutwardWm2;
    }
    return actual;
   });
   const double canonicalOld=std::max(300.,std::min(4000.,old));
   check(observed==canonicalOld,"original DO/material canonical temperature");
   check(std::memcmp(diam.begin(),beforeDiameter.begin(),sizeof(scalar)*diam.size())==0 && std::memcmp(temp.begin(),beforeTemperature.begin(),sizeof(scalar)*temp.size())==0 && std::memcmp(mass.begin(),beforeMass.begin(),sizeof(scalar)*mass.size())==0 && std::memcmp(eps.begin(),beforeEpsilon.begin(),sizeof(scalar)*eps.size())==0,"caller prefix/trailing storage was modified");
   check(result.finalMeanTemperatureK.size()==mesh.nCells() && result.solidWallRadiationEnergyJ.size()==mesh.boundary().size(),"complete transaction");
   double accepted=0,boundary=result.escapedBoundaryRadiationEnergyJ;
   for(label i=0;i<mesh.nCells();++i){
    const double energy=result.deltaParticleRadiationEnergyJ[i];
    check(energy==massValue*cp*(result.finalMeanTemperatureK[i]-canonicalOld),"accepted energy canonical old-state base");
    auto state=carried;
    if(inRange){
     const auto update=radiationAffineTemperatureUpdate(energy,massValue,cp,old,200,4000);
     const auto pub=ugkwp::publishAffineRadiationThermalHeat(state,update.scale,update.offset,GpuReal(200),GpuReal(4000),law);
     check(pub.valid && near(ugkwp::effectiveParticleTemperature(state,law),result.finalMeanTemperatureK[i]),"real affine/carry applied candidate");
     check(near(pub.acceptedSpecificHeat,law.enthalpyIncrement(old,std::fma(update.scale-1.,old,update.offset))),"actual enthalpy increment");
     std::cout<<"APPLY "<<i<<' '<<update.scale<<' '<<update.offset<<' '<<pub.acceptedSpecificHeat<<' '<<ugkwp::effectiveParticleTemperature(state,law)<<'\n';
    }


    accepted+=energy;
    std::cout<<"DIAG "<<i<<' '<<result.absorptionCoefficientInvM[i]<<' '<<result.scatteringCoefficientInvM[i]<<' '<<result.extinctionCoefficientInvM[i]<<' '<<result.asymmetryFactor[i]<<'\n';
    std::cout<<"CELL "<<i<<' '<<observed<<' '<<result.finalMeanTemperatureK[i]<<' '<<energy<<' '<<state.temperature<<' '<<state.specificEnthalpyResidual<<'\n';
   }
   forAll(result.solidWallRadiationEnergyJ,p)forAll(result.solidWallRadiationEnergyJ[p],f)boundary+=result.solidWallRadiationEnergyJ[p][f];
   if(mode=="boundary_overflow"){
    check(std::isinf(result.solidWallRadiationEnergyJ[patches[0]][0]),"overflow scale never reached physical face publication");
    check(std::isnan(result.escapedBoundaryRadiationEnergyJ),"zero faces must retain actual infinite-scale IEEE product");
   }else if(mode.find("boundary_")==0){
    const double scale=mode=="boundary_cancellation"?1.:-accepted/suppliedBoundarySum;double escaped=0;
    forAll(suppliedBoundary,p)forAll(suppliedBoundary[p],f){
     const double e=scale*(dt*mag(mesh.Sf().boundaryField()[p][f])*suppliedBoundary[p][f]);
     if(p==patches[0])check(result.solidWallRadiationEnergyJ[p][f]==e,"actual per-face boundary scale");else escaped+=e;
    }
    check(result.escapedBoundaryRadiationEnergyJ==escaped,"actual escaped energy after one per-face application");
    double expectedBoundary=escaped;
    forAll(suppliedBoundary[patches[0]],f)expectedBoundary+=scale*(dt*mag(mesh.Sf().boundaryField()[patches[0]][f])*suppliedBoundary[patches[0]][f]);
    check(boundary==expectedBoundary,"returned physical boundary energy");
    if(mode=="boundary_cancellation")check(accepted+boundary!=0,"returned-energy cancellation witness must retain its nonzero residual");
   }else{
    check(near(accepted+boundary,0),"actual transferred boundary energy");
   }
   forAll(result.solidWallRadiationEnergyJ,p)forAll(result.solidWallRadiationEnergyJ[p],f)std::cout<<"BOUNDARY "<<p<<' '<<f<<' '<<result.solidWallRadiationEnergyJ[p][f]<<'\n';
   std::cout<<"ESCAPED "<<result.escapedBoundaryRadiationEnergyJ<<'\n';
   std::cout<<"BALANCE "<<accepted<<' '<<boundary<<' '<<accepted+boundary<<'\n';
  }else{
   MieDoRadiationSolver solver(mesh,config,table,patches,controls);
   auto workspace=solver.makeWorkspace();
   MieDoSnapshot snapshot{eps,temp,diam,wall,emissivity};
   auto result=solver.solve(snapshot,workspace);
   check(result.sourceIterations>=1,"actual initial source sweep missing");
   if(mode=="do_parallel")check(result.angleThreadsUsed>=1 && result.sourceIterations==1,"actual parallel initial sweep");
   if(firstSweep || mode=="do_zero")check(result.sourceIterations==1,"one completed source sweep");
   check(result.incidentRadiationG_Wm2.size()==mesh.nCells() && result.absorptionCoefficientInvM.size()==mesh.nCells() && result.scatteringCoefficientInvM.size()==mesh.nCells() && result.extinctionCoefficientInvM.size()==mesh.nCells() && result.asymmetryFactor.size()==mesh.nCells(),"complete DO fields");
   if(mode=="do_zero_divisor" || mode=="do_nonfinite"){
    if(mode=="do_zero_divisor")check(std::isinf(result.absorptionCoefficientInvM[0]),"actual zero-diameter absorption missing");
    check(std::isnan(result.incidentRadiationG_Wm2[0]),"nonfinite residual skipped native RHS/diagonal sweep and published cached intensity");
    check(!std::isfinite(result.particlePowerDensityWm3[0]),"actual IEEE particle power missing");
    check(result.boundaryQrOutwardWm2[0].size()==mesh.boundary()[0].size(),"IEEE boundary publication missing");
    std::cout<<"PASS "<<mode<<'\n';return 0;
   }
   double power=0,boundary=0;
   forAll(temp,i){
    const double g=result.incidentRadiationG_Wm2[i];
    check(std::isfinite(g) && std::isfinite(result.particlePowerDensityWm3[i]),"actual finite moments");
    if(mode=="do_negative")check(near(g,-2*fourPiConstant()),"large negative intensity lost before moment");
    if(mode=="do_roundoff")check(g==0,"small roundoff clamp changed");
    const auto optical=table.query(diam[i],temp[i]);
    check(result.particlePowerDensityWm3[i]==particleRadiationPowerDensity(result.absorptionCoefficientInvM[i],g,eps[i]==0?0:bandBlackBodyIntensity(temp[i],optical.planckBandFraction)),"source consumed actual moment");
    power+=result.particlePowerDensityWm3[i]*mesh.V()[i];
    std::cout<<"DO "<<i<<' '<<g<<' '<<result.particlePowerDensityWm3[i]<<' '<<result.absorptionCoefficientInvM[i]<<' '<<result.scatteringCoefficientInvM[i]<<'\n';
   }
   forAll(mesh.boundary(),p){check(result.boundaryQrOutwardWm2[p].size()==mesh.boundary()[p].size(),"complete DO boundary");forAll(result.boundaryQrOutwardWm2[p],f){check(std::isfinite(result.boundaryQrOutwardWm2[p][f]),"finite boundary");boundary+=result.boundaryQrOutwardWm2[p][f]*mag(mesh.Sf().boundaryField()[p][f]);std::cout<<"FACE "<<p<<' '<<f<<' '<<result.boundaryQrOutwardWm2[p][f]<<' '<<mesh.Sf().boundaryField()[p][f].x()/mag(mesh.Sf().boundaryField()[p][f])<<' '<<mesh.Sf().boundaryField()[p][f].y()/mag(mesh.Sf().boundaryField()[p][f])<<' '<<mesh.Sf().boundaryField()[p][f].z()/mag(mesh.Sf().boundaryField()[p][f])<<' '<<mesh.boundary()[p].faceCells()[f]<<'\n';}}
   check(result.integratedParticlePowerW==power && result.integratedBoundaryPowerW==boundary && result.conservationResidualW==power+boundary,"DO integrated outputs");
   dumpMieResult("RESULT",result);
   if(mode=="do_warm") {auto next=solver.solve(snapshot,workspace);check(near(next.integratedParticlePowerW,power,1e-8),"warm physical result");dumpMieResult("WARM",next);}
   if(mode=="do_identity") {MieDoRadiationSolver other(mesh,config,table,patches,controls);auto next=other.solve(snapshot,workspace);check(near(next.integratedParticlePowerW,power,1e-8),"reusable workspace identity");}
  }
 }
 std::cout<<"PASS "<<mode<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<"FAIL "<<e.what()<<'\n';return 1;}
