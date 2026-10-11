
#include "common/gasWall/WallModel.H"
#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>
using namespace ugkwp;
using namespace ugkwp::gaswall;
struct Model {
 SpeciesThermoData<double> sp[2]; double coef[6]={1040,0,0,1040,0,0};
 double elements[2]={1,1}, basis[2]={-1,1};
 GasReactionData<double> rx[1]; GasStoichTerm<double> re[1],pr[1];
 WallInput<double,2> in;
 Model(){for(int s=0;s<2;++s){sp[s].coefficientOffset=3*s;sp[s].molarMass=.028;sp[s].minTemperature=100;sp[s].maxTemperature=4000;sp[s].referencePressure=101325;sp[s].referenceTemperature=300;sp[s].referenceEntropy=1500;sp[s].hasEntropyReference=true;}
 auto& t=in.model.thermo;t.species=sp;t.coefficients=coef;t.coefficientCount=6;t.elementCount=1;t.elementComposition=elements;t.speciesOrderHash=1;
 rx[0].reactantCount=1;rx[0].productCount=1;rx[0].highRate.preExponential=2;re[0].species=0;re[0].coefficient=1;pr[0].species=1;pr[0].coefficient=1;
 auto& m=in.model.mechanism;m.reactions=rx;m.reactionCount=1;m.reactants=re;m.reactantTermCount=1;m.products=pr;m.productTermCount=1;m.referencePressure=101325;m.speciesOrderHash=1;m.stoichiometricBasis=basis;m.independentRank=1;
 in.integrateOwnerKSource=true; // Retain original auxiliary diagnostic comparison.
 in.pressure=101325;in.temperature=300;in.normal[1]=1;in.matchingDistance=.01;in.ownerDistance=.004;
 in.matching.temperature=500;in.matching.velocity[0]=10;in.matching.massFraction[0]=.7;in.matching.massFraction[1]=.3;
 in.model.viscosity=2e-5;in.model.conductivity=.04;in.model.diffusivity[0]=in.model.diffusivity[1]=1e-5;
 }
};
bool near(double a,double b,double r=1e-10){return std::abs(a-b)<1e-10+r*std::abs(b);}


#include <algorithm>
static constexpr int NN=24, BB=6;
void configure(Model&m){
m.in.temperature=m.in.matching.temperature=300;m.in.pressure=101325;
 m.in.matching.velocity[0]=70;m.in.matching.massFraction[0]=m.in.matching.massFraction[1]=.5;
 m.in.matching.k=1e-4;m.in.matching.omega=1;
 m.in.ownerDistance=0.00061447192990219334;m.in.matchingDistance=0.0018712125170887821;
 m.in.model.viscosity=1.8e-5;m.in.model.conductivity=1.8e-5*1040/.72;m.in.model.molecularPrandtl=.72;
 m.in.model.diffusivity[0]=m.in.model.diffusivity[1]=.02;
 static double distance[]={1.3944738147928374e-09,3.6561619078705297e-08,1.990767557126556e-07,5.8964327942529685e-07,1.2384999885095418e-06,2.0580234921422729e-06,2.8545810604424219e-06,3.3981929606041689e-06,5.7797990870390286e-06,1.1906519600000498e-05,1.7886613865683493e-05,2.8097814720619279e-05,3.7067956119143774e-05,5.1363637316053863e-05,6.3323825847419858e-05,8.1703987386304256e-05,9.6654223050511763e-05,0.00011911886493137048,0.00013705914772841948,0.00016360826995125252,0.00018453859988114301,0.00021517220244595032,0.0002390925795086823,0.00027381066241546399,0.00030072108661103751,0.00033952364985979341,0.00036942412118820843,0.00041231116477893876,0.00044520168324019526,0.00049217320717289987,0.00052805377276699775,0.00057910977704167651,0.00060132127814122916,0.00061094822338290011,0.00063113104152958028,0.00067664458090456274,0.00071498153424505049,0.0007742064992036779,0.00081905720619630037,0.0008823666514969022,0.00093020740562236605,0.00099760133126494221,0.0010484321325232478,0.0011199105385077983,0.0011635860289236936,0.0012114312818008559},weight[]={1.4811553874765256e-12,1.6661071935312081e-11,5.4843666364243152e-11,1.0912274885094559e-10,1.5814979179284293e-10,1.7633615502069745e-10,1.4721817161868379e-10,7.3117184323112895e-11,1.1053949179399717e-09,1.1053949179399717e-09,1.8423248632332855e-09,1.8423248632332861e-09,2.5792548085265985e-09,2.5792548085265985e-09,3.3161847538199137e-09,3.3161847538199129e-09,4.0531146991132269e-09,4.0531146991132278e-09,4.7900446444065406e-09,4.7900446444065406e-09,5.5269745896998476e-09,5.5269745896998476e-09,6.2639045349931686e-09,6.2639045349931678e-09,7.000834480286474e-09,7.0008344802864706e-09,7.737764425579795e-09,7.7377644255797934e-09,8.4746943708730946e-09,8.4746943708730913e-09,9.2116243161663908e-09,9.2116243161663892e-09,1.7369123208601431e-09,1.7369123208601427e-09,8.2116419405995938e-09,8.2116419405995905e-09,1.0685484206753021e-08,1.0685484206753021e-08,1.1422414152046314e-08,1.1422414152046313e-08,1.215934409733962e-08,1.215934409733962e-08,1.2896274042632973e-08,1.2896274042632966e-08,8.6323342587735315e-09,8.6323342587735299e-09};
 m.in.quadrature.distance=distance;m.in.quadrature.volumeWeight=weight;m.in.quadrature.count=46;
 m.in.quadrature.volume=2.5602997079258032e-07;m.in.quadrature.firstMoment=1.5732323026571902e-10;
}
