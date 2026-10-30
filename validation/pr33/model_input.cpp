#include "configuration/ModelIO.H"
#include "IStringStream.H"
#include <cassert>
#include <iostream>
chmt::ModelConfig parse(int count,bool surface,const std::string& wall){
 std::string input="modelName test;materialSource test;mechanismSource test;condensedNames (C0 C1);elements ();minDt 1e-12;maxDt .001;cfl .4;";
 input+=surface?"material { surfaceReactions {":"reactions {";
 for(int i=0;i<count;++i)input+="r"+std::to_string(i)+" { A 1; temperaturePower 0; activationEnergy 0; condensedNu (-1 1); gasNu (0 0); order (1 0); gasOrder (0 0); }";
 input+=surface?"}}":"}";input+=wall;
 Foam::IStringStream stream(input);Foam::dictionary d(stream);
 auto gas=ugkwp::parseGasModelProperties(R"(gasMode mixtureFrozen;species(A B);speciesThermo{
 A{model linearCp;molarMass 0.028;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);}
 B{model linearCp;molarMass 0.032;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);}}
 diffusion{model none;})");return chmt::io::readModel(d,gas);
}
int main(){Foam::FatalError.throwExceptions();Foam::FatalIOError.throwExceptions();
 for(bool surface:{false,true})for(int n:{0,8,9}){bool refused=false;try{auto m=parse(n,surface,"");assert((surface?m.physics.material.nSurfaceReactions:m.physics.nReactions)==n);}catch(const std::exception&){refused=true;}assert(refused==(n==9));}
 for(const char* model:{"constantTransport","reactingSst","equilibriumOde"}){
  auto m=parse(0,false,std::string("sst {wallTreatment boundaryLayer;boundaryLayer{model ")+model+";}}");
  int want=std::string(model)=="constantTransport"?0:std::string(model)=="reactingSst"?1:2;assert(int(m.physics.wallModel.model)==want);
 }
 for(const char* config:{"sst {wallTreatment wallFunction;}","sst {wallTreatment typo;}","sst {boundaryLayer{model typo;}}"}){bool refused=false;try{parse(0,false,config);}catch(const std::exception&){refused=true;}assert(refused);}
 std::cout<<"native OpenFOAM model parsing: capacity and explicit wall selection passed\n";
}
