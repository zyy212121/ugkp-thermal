#pragma once
#include "common/gasWall/WallModel.H"
using namespace ugkwp;using namespace ugkwp::gaswall;
struct CompositionWallCase {
 SpeciesThermoData<double> sp[2]; double coef[6]={1000,0,0,1000,0,0};
 double elements[2]={1,1}, basis[2]={-1,1};
 GasReactionData<double> rx[1]; GasStoichTerm<double> re[1],pr[1];
 WallInput<double,2> in;
 CompositionWallCase(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=200;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 auto& t=in.model.thermo;t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 rx[0].reactantCount=1;rx[0].productCount=1;rx[0].highRate.preExponential=2;re[0].species=0;re[0].coefficient=1;pr[0].species=1;pr[0].coefficient=1;
 auto& m=in.model.mechanism;m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=1;m.products=pr;m.productTermCount=1;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1;
 in.integrateOwnerKSource=true; // This fixture explicitly tests auxiliary source integrals.
 in.pressure=101325;in.temperature=300;in.normal[1]=1;in.matchingDistance=.01;in.ownerDistance=.004;
 in.matching.temperature=500;in.matching.velocity[0]=10;in.matching.massFraction[0]=.7;in.matching.massFraction[1]=.3;
 in.model.viscosity=2e-5;in.model.conductivity=.04;in.model.diffusivity[0]=in.model.diffusivity[1]=1e-5;
 }
};

inline void configureCompositionCase(CompositionWallCase& m,int index){
if(index==11){m.in.matchingDistance=0.017551894257351978;m.in.ownerDistance=0.0070207577029407914;m.in.temperature=811.06245592236519;m.in.matching.temperature=812.43478723336011;m.in.matching.velocity[0]=20.678745171782722;m.in.matching.k=3.1383821261466879e-08;m.in.matching.omega=2.4335784832997751;m.in.massFlux[0]=.7*0.033123945517176005;m.in.massFlux[1]=.3*0.033123945517176005;m.coef[2]=300763.30746524036;m.rx[0].highRate.preExponential=141.83705673354405;}
if(index==25){m.in.matchingDistance=0.042158321148868173;m.in.ownerDistance=0.016863328459547269;m.in.temperature=942.31747165322304;m.in.matching.temperature=1665.6711737858132;m.in.matching.velocity[0]=127.5637975848754;m.in.matching.k=5.4043890780066887e-05;m.in.matching.omega=22212.191989440555;m.in.massFlux[0]=.7*0.0025689486407775367;m.in.massFlux[1]=.3*0.0025689486407775367;m.coef[2]=213700.72844438255;m.rx[0].highRate.preExponential=5.7324830220009204;}
m.in.model.mode=GasMode::MixtureChemistry;m.in.model.diffusivity[0]=1e-4;m.in.model.diffusivity[1]=3e-4;
}
