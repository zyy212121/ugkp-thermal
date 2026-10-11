"""Retained blowing states through actual wall publication and SST owner update.

One owner control volume with a fixed external donor and imposed compensating
mass outflow. Bulk U/T/rho are prescribed, not advanced. Every accepted step
solves the actual nonlinear wall again and advances actual coarse rho*k.
"""
from pathlib import Path
import json,sys,subprocess
ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).parent; OUT=HERE/'results'
OUT.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT/'tests/gas_transport'))
from test_legacy_ale_transport import fixture
from test_sst_ale_audit import AUDIT
body=fixture()+AUDIT+r'''
#include "research/sst_wall_function/owner32_input.hpp"
#include "gasTransport/GasBoundaryLayerEvaluation.H"
#include <iomanip>
#include <chrono>
int main(){std::cout<<std::setprecision(17)<<"[";int which=0;
for(int nodes:{32,100})for(int variant=0;variant<6;++variant){
 const double mass=variant==0?0:variant==1?1e-9:variant==4?1e-12:variant==5?1e-15:.001,yp=.004,area=5e-5,volume=4e-7;
 Model m;m.in.integrateOwnerKSource=false;m.in.temperature=m.in.matching.temperature=500;m.in.matching.velocity[0]=30;m.in.matching.k=.5;m.in.matching.omega=200;
 m.in.model.mode=variant==3?ugkwp::GasMode::MixtureChemistry:ugkwp::GasMode::MixtureFrozen;
 if(variant==3){m.in.matching.temperature=700;m.coef[2]=100000;m.rx[0].highRate.preExponential=20;m.in.model.diffusivity[0]=1e-4;m.in.model.diffusivity[1]=3e-4;}
 m.in.massFlux[0]=.7*mass;m.in.massFlux[1]=.3*mass;
 double qy[2]={.002,.006},qw[2]={2e-7,2e-7};m.in.quadrature={qy,qw,2,volume,1.6e-9};
 WallModelConfig<double> cfg;cfg.enableSst=true;cfg.nodes=nodes;cfg.maxIterations=100;
 static WallWorkspace<double,2,128> work;WallOutput<double,2> out;WallStatus status;
 State s;initialise(s);s.nFaces=2;s.nInternalFaces=1;s.sstConfigured=1;s.turbulenceModel=3;s.sstWallTreatment=2;s.sstWallCmu=.09;s.sstWallKappa=.41;s.sstWallE=9.8;s.gasMu=2e-5;s.Rgas=ugkwp::universalGasConstant<Real>()/.028;s.gasPrClamped=.72;
 for(int c=0;c<2;++c){s.Cx[c]=s.Cz[c]=0;s.Cy[c]=c==0?yp:.01;s.V[c]=volume;s.sstWallDistance[c]=s.Cy[c];s.Tgas[c]=c==0?(500+m.in.matching.temperature)/2:m.in.matching.temperature;s.rho[c]=101325/(s.Rgas*s.Tgas[c]);s.p[c]=101325;s.Ux[c]=c==0?25:30;s.Uy[c]=mass/s.rho[c];s.Uz[c]=0;s.k[c]=.5;s.omega[c]=200;s.rhoK[c]=s.rho[c]*s.k[c];s.rhoOmega[c]=s.rho[c]*s.omega[c];s.nut[c]=.5/200;}
 s.cellPlaneStart[0]=0;s.cellPlaneCount[0]=2;s.cellFaceId[0]=0;s.cellFaceId[1]=1;s.cellPlaneCount[1]=0;
 for(int f=0;f<2;++f){s.Sfx[f]=s.Sfz[f]=0;s.Sfy[f]=f==0?area:-area;s.magSf[f]=area;s.faceCx[f]=s.faceCz[f]=0;s.faceCy[f]=f==0?.007:0;s.deltaCoeffs[f]=f==0?1/.006:1/yp;s.faceWeight[f]=.5;s.riemannBoundaryKind[f]=f==0?0:2;}
 auto&w=s.gasBoundaryLayer;w.enabled=true;w.count=1;w.faceSlot=new int[2]{-1,0};w.ownerSlot=new int[2]{0,-1};w.status=new int[1]{};w.exchange=new ugkwp::GasBoundaryLayerExchange<Real>[1];w.sst=new ugkwp::GasBoundaryLayerSstClosure<Real>[1];w.speciesFlux=new Real[2]{};
 double time=0,sourceIntegral=0,faceIntegral=0,maxBudget=0,maxNumber=0,minK=.5,maxSpeciesResidual=0,maxWallEnergyResidual=0,maxWallMomentumResidual=0,maxWallMassResidual=0,maxWallK=0,maxWallOmega=0,maxBvpResidual=0;
 audit(s);const double initial=s.rhoK[0]*volume,initialOmega=s.rhoOmega[0]*volume;int steps=0,newton=0;const auto start=std::chrono::steady_clock::now();
 while(time<.005){
  ck(evaluateWallModel(m.in,cfg,work,out,status),"nonlinear blowing wall failed");newton+=out.iterations;maxBvpResidual=std::max(maxBvpResidual,out.residual);
  ck(ugkwp::publishGasBoundaryLayerOutput(s,0,m.in,out,area,0.),"wall publish failed");
  threadIdx.x=1;computeGasInternalFaceFluxKernel<false>(&s,2e-5);ck(!s.gasSpecies.faceStatus[1],"actual physical wall flux failed");
  // Prescribed external outflow conserves the fixed bulk mass exactly.
  s.gasPhiRho[0]=-s.gasPhiRho[1];
  maxWallMassResidual=std::max(maxWallMassResidual,std::abs(s.gasPhiRho[1]+area*mass));
  maxWallMomentumResidual=std::max(maxWallMomentumResidual,std::abs(s.gasPhiRhoUx[1]-area*out.traction[0]));
  double energy=out.conductiveHeatFlux+.5*mass*(out.traceVelocity[0]*out.traceVelocity[0]+out.traceVelocity[1]*out.traceVelocity[1]+out.traceVelocity[2]*out.traceVelocity[2]);
  for(int j=0;j<2;++j){energy+=out.wallSpeciesFlux[j]*speciesH(j,out.traceTemperature,m.in.model.thermo);maxSpeciesResidual=std::max(maxSpeciesResidual,std::abs(out.speciesBalanceResidual[j]));near(w.speciesFlux[j],-area*m.in.massFlux[j],"species injection was changed by auxiliary reaction");}
  maxWallEnergyResidual=std::max(maxWallEnergyResidual,std::abs(s.gasPhiRhoE[1]+area*energy));
  threadIdx.x=0;applySstWallFunctionStateKernel(&s);computeGasPrimitiveGradientsKernel(&s);computeGasEddyViscosityKernel(&s);computeSstGradientsKernel(&s);
  double dt=std::min(2e-5,.005-time);computeSstStabilityNumberKernel(&s,dt,.5);if(s.sstSourceNumber[0]>.5)dt*=.45/s.sstSourceNumber[0];
  for(int f=0;f<2;++f){threadIdx.x=f;computeSstFaceFluxKernel(&s);}
  maxWallK=std::max(maxWallK,std::abs(s.sstPhiRhoK[1]));maxWallOmega=std::max(maxWallOmega,std::abs(s.sstPhiRhoOmega[1]));
  const double flux=-s.sstPhiRhoK[0]-s.sstPhiRhoK[1],div=(s.gasPhiRho[0]/sstFaceDensity(s,0)+s.gasPhiRho[1]/sstFaceDensity(s,1))/volume;
  Real sk,so;sstSourcesForCell(s,0,div,sk,so);const double old=s.rhoK[0];threadIdx.x=0;applySstFluxAndSourceKernel(&s,dt);ck(!s.gasSpecies.cellStatus[0],"owner update failed");ck(s.rhoK[0]>s.rho[0]*s.sstKMin,"owner floor activated");
  maxBudget=std::max(maxBudget,std::abs((s.rhoK[0]-old)*volume-dt*(sk*volume+flux)));sourceIntegral+=dt*volume*sk;faceIntegral+=dt*flux;maxNumber=std::max(maxNumber,double(s.sstSourceNumber[0]));recoverSstPrimitivesKernel(&s);minK=std::min(minK,double(s.k[0]));time+=dt;++steps;ck(steps<100000,"source time-step collapse");
 }
 const auto&a=s.gasSstAudit;
 const double omegaBudget=s.rhoOmega[0]*volume-initialOmega-a.transportOmega[0]-a.sourceOmega[0]-a.constraintOmega[0];
 if(which++)std::cout<<",";
 std::cout<<"{\"nodes\":"<<nodes<<",\"chemistry\":"<<(variant==3?"true":"false")<<",\"mass_flux\":"<<mass<<",\"time\":"<<time<<",\"steps\":"<<steps<<",\"newton_iterations\":"<<newton<<",\"seconds\":"<<std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<<",\"tau\":"<<out.traction[0]<<",\"heat\":"<<out.conductiveHeatFlux<<",\"owner_omega\":"<<out.ownerOmega<<",\"final_k\":"<<s.k[0]<<",\"minimum_k\":"<<minK<<",\"source_impulse\":"<<sourceIntegral<<",\"face_impulse\":"<<faceIntegral<<",\"audit_source_k\":"<<a.sourceK[0]<<",\"audit_transport_k\":"<<a.transportK[0]<<",\"audit_constraint_k\":"<<a.constraintK[0]<<",\"audit_source_omega\":"<<a.sourceOmega[0]<<",\"audit_transport_omega\":"<<a.transportOmega[0]<<",\"audit_constraint_omega\":"<<a.constraintOmega[0]<<",\"omega_budget_residual\":"<<omegaBudget<<",\"global_k_budget_residual\":"<<s.rhoK[0]*volume-initial-sourceIntegral-faceIntegral<<",\"max_step_budget_residual\":"<<maxBudget<<",\"max_source_number\":"<<maxNumber<<",\"wall_mass_flux_residual\":"<<maxWallMassResidual<<",\"wall_tangent_momentum_residual\":"<<maxWallMomentumResidual<<",\"wall_energy_flux_residual\":"<<maxWallEnergyResidual<<",\"wall_k_flux\":"<<maxWallK<<",\"wall_omega_flux\":"<<maxWallOmega<<",\"aux_species_balance_residual\":"<<maxSpeciesResidual<<",\"aux_reaction_integral\":"<<out.reactionIntegral[0]<<",\"aux_bvp_residual\":"<<maxBvpResidual<<"}";
 }std::cout<<"]\n";}
'''
pre=(ROOT/'tests/gas_transport/gas_state_probe.cpp').read_text().split('void hashBytes')[0]
source=OUT/'blowing.cpp';source.write_text(pre+body);exe=OUT/'blowing'
subprocess.run(['g++','-O2','-std=c++17','-I'+str(ROOT),'-I'+str(ROOT/'tests/gas_transport'),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(source),'-o',str(exe)],check=True)
p=subprocess.run([str(exe)],capture_output=True,text=True);(OUT/'blowing.log').write_text(p.stdout+p.stderr);print(p.stdout+p.stderr);p.check_returncode()
(OUT/'blowing.json').write_text(json.dumps(json.loads(p.stdout),indent=2))
