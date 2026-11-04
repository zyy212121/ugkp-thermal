#include "configuration/SharedGasModel.H"
#include <cassert>
#include <iostream>
using namespace chmt;
ugkwp::GasModelConfiguration valid(){
 ugkwp::GasModelConfiguration g;g.species.resize(Ns);g.speciesNames.resize(Ns);g.diffusionCoefficients.resize(Ns);
 for(int s=0;s<Ns;++s){g.speciesNames[s]="S"+std::to_string(s);g.species[s].molarMass=.028;
 g.species[s].coefficientOffset=3*s;g.coefficients.insert(g.coefficients.end(),{1000,0,0});}
 return g;
}
int main(int argc,char** argv){
 const int only=argc>1?std::stoi(argv[1]):-1;
 for(int test=0;test<12;++test){if(only>=0&&only!=test)continue;auto g=valid();
 if(test==1){g.species.resize(Ns-1);g.species.shrink_to_fit();}
 if(test==2){g.elementNames.resize(Ne+1);g.elementComposition.resize(Ns*(Ne+1));}
 if(test==3)g.speciesNames.resize(Ns-1);
 if(test==4)g.diffusionCoefficients.clear();
 if(test==5){g.elementNames.resize(1);g.elementComposition.resize(Ns-1);}
 if(test==6)g.species[0].coefficientOffset=-1;
 if(test==7)g.coefficients.pop_back();
 if(test==8)g.species[0].model=static_cast<ugkwp::SpeciesThermoModel>(255);
 if(test==9||test==10){g.species[0].model=ugkwp::SpeciesThermoModel::NASA7;
 g.species[0].coefficientOffset=int(g.coefficients.size());g.coefficients.resize(g.coefficients.size()+(test==9?14:13));}
 if(test==11){g.elementNames.resize(Ne);g.elementComposition.resize(Ns*Ne);}
 ModelConfig m; m.gasPhase="unchanged";
 if(test==11){m.physics.nElements=Ne;for(int e=0;e<Ne;++e)m.elementNames[e]=g.elementNames[e]="E"+std::to_string(e);}std::string error;
 const bool ok=bindSharedGasModel(g,m,error);assert(ok==(test==0||test==9||test==11));
 if(!ok){assert(!error.empty());assert(m.gasPhase=="unchanged");
 assert(!bindSingleGasModel(g,"S0",m,error));assert(m.gasPhase=="unchanged");}
 std::cout<<"PASS metadata "<<test<<" accepted="<<ok<<'\n';
 }
}
