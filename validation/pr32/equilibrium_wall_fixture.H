#pragma once
#include "gasWall/WallModel.H"
#include "gasTransport/GasBoundaryLayerEvaluation.H"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <algorithm>
#include <cstdlib>
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif
using namespace ugkwp;
using namespace ugkwp::gaswall;
struct Fixture {
 SpeciesThermoData<double> species[2];
 double coefficients[6]={1004.675,0,0,819.27,0,0};
 PowerLawDiffusivity<double> diffusion[2];
 WallInput<double,2> in;
 UGKWP_GAS_HD Fixture(){
  for(int j=0;j<2;++j){species[j].coefficientOffset=3*j;species[j].molarMass=j?.0440095:.02896520682164515;species[j].minTemperature=1;species[j].maxTemperature=5000;
   diffusion[j]={5.75e-10,1,101325,1.81,-1};}
  in.model.thermo.species=species;in.model.thermo.coefficients=coefficients;in.model.thermo.coefficientCount=6;
  auto& v=in.model.molecularViscosity;
  v.model=MolecularViscosityModel::MassFractionSutherland;v.compositionSpecies=1;v.compositionTRef=273;
  v.muCoefficients[0]=17.1e-6;v.muCoefficients[1]=-4.65e-6;v.muCoefficients[2]=1.2e-6;
  v.sCoefficients[0]=114;v.sCoefficients[1]=220;
  v.prandtlSpecies=1;v.prandtlCoefficients[0]=.72;v.prandtlCoefficients[1]=.064;
  v.diffusion=diffusion;v.diffusionCount=2;in.model.molecularPrandtl=.72;in.model.viscosity=1.71e-5;
  in.temperature=281.175969323;in.normal[1]=1;in.matchingDistance=.0004;in.ownerDistance=.0002;
  in.matching.massFraction[0]=1;
 }
};

using Workspace=WallWorkspace<double,2,32>;
UGKWP_GAS_HD void configure(int index,Fixture& f,WallModelConfig<double>& c){
 auto& in=f.in;c.model=BoundaryLayerModel::EquilibriumOde;c.nodes=8;c.stretch=2.5;c.maxIterations=60;
 in.pressure=index<6?10154.:101325.;in.matching.temperature=(index%6)<3?83.:200.;
 in.matching.velocity[0]=650.;in.massFlux[1]=index%3==0?0.:(index%3==1?.02:.22);
 if(index==12){
  // Captured failure: the conservative matching extrapolation approaches its
  // positive-energy boundary while the wall remains hot. The old edge-only
  // residual scale amplified roundoff and stalled this otherwise finite BVP.
  in.pressure=1.038123550185038e-11;in.temperature=204.89045186139472;
  in.matching.temperature=5.6843418860808015e-13;
  in.matchingDistance=.00040000000000003927;in.ownerDistance=.00020000000000003348;
  in.matching.velocity[0]=294.72677217253636;in.matching.velocity[1]=457.65401340161969;
  in.matching.velocity[2]=-4.2457267064962132e-18;in.normal[1]=-1;in.massFlux[1]=0;
 }
 if(index==13)c.maxIterations=1;
 if(index>=15){in.adiabatic=true;in.pressure=10154;in.matching.temperature=83;in.massFlux[1]=index==16?.02:0;}
 if(index>=17){
  c.relativeTolerance=1e-10;c.absoluteTolerance=1e-12;
  in.pressure=101325;in.temperature=300;in.matching.temperature=200;in.matching.velocity[0]=100;
  f.coefficients[0]=f.coefficients[3]=1000;f.species[0].molarMass=f.species[1].molarMass=.028;
  in.model.molecularViscosity={};in.model.viscosity=2e-5;in.model.molecularPrandtl=.72;
  in.model.diffusivity[0]=in.model.diffusivity[1]=1e-4;c.equilibriumKappa=0;c.nodes=32;c.stretch=1;
  if(index==18)c.model=BoundaryLayerModel::ConstantTransport;
  if(index==19){c.model=BoundaryLayerModel::ReactingSst;c.enableSst=true;c.maxIterations=200;
   in.matchingDistance=.01;in.ownerDistance=.004;in.matching.temperature=in.temperature=500;
   in.matching.velocity[0]=30;in.matching.k=.5;in.matching.omega=200;}
 }
 if(index==14){
  // Independent isothermal, constant-property advection/diffusion solution:
  // Y_CO2(y)=1-exp(m*(y-L)/(rho*D)), J_Air=0, J_CO2=m.
  in.pressure=101325;in.temperature=in.matching.temperature=300;
  in.matching.velocity[0]=0;in.matchingDistance=.01;in.ownerDistance=.004;
  in.massFlux[1]=.002;f.coefficients[3]=f.coefficients[0]=1000;
  f.species[1].molarMass=f.species[0].molarMass=.028;
  in.model.molecularViscosity={};in.model.viscosity=2e-5;
  in.model.diffusivity[0]=in.model.diffusivity[1]=1e-4;
  c.equilibriumKappa=0;c.nodes=32;c.stretch=1;
 }
}
