#include "configuration/ModelIO.H"
#include "IStringStream.H"
#include <cassert>
#include <iostream>
chmt::ModelConfig parse(int count,bool surface,const std::string& wall,int metadata=0){
 std::string input="modelName test;materialSource test;mechanismSource test;condensedNames (C0 C1);elements ();minDt 1e-12;maxDt .001;cfl .4;";
 input+=surface?"material { surfaceReactions {":"reactions {";
 for(int i=0;i<count;++i)input+="r"+std::to_string(i)+" { A 1; temperaturePower 0; activationEnergy 0; condensedNu (-1 1); gasNu (0 0); order (1 0); gasOrder (0 0); }";
 input+=surface?"}}":"}";input+=wall;
 Foam::IStringStream stream(input);Foam::dictionary d(stream);
 auto gas=ugkwp::parseGasModelProperties(R"(gasMode mixtureFrozen;species(A B);speciesThermo{
 A{model linearCp;molarMass 0.028;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);}
 B{model linearCp;molarMass 0.032;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);}}
 diffusion{model none;})");
 if(metadata==1)gas.species.resize(chmt::Ns-1);
 if(metadata==2){gas.elementNames.resize(chmt::Ne+1);gas.elementComposition.resize(chmt::Ns*(chmt::Ne+1));}
 return chmt::io::readModel(d,gas);
}
void elementMapping(bool malformedChecks=true){
 const std::string material=R"(modelName test;materialSource test;mechanismSource test;condensedNames (C0 C1);elements (C H);minDt 1e-12;maxDt .001;cfl .4;
 condensedThermo{C0{rho 1000;cp0 1000;cp1 0;e0 0;conductivity 1;Tmin 100;Tmax 3000;element (83.333333333333333 0);}
 C1{rho 1000;cp0 1000;cp1 0;e0 0;conductivity 1;Tmin 100;Tmax 3000;element (0 1000);}})";
 for(bool reverse:{false,true}){
  const std::string gasText=std::string("gasMode mixtureFrozen;species(A B);elements(")+(reverse?"H C":"C H")+R"();speciesThermo{
 A{model linearCp;molarMass .012;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);atoms()"+(reverse?"0 1":"1 0")+R"();}
 B{model linearCp;molarMass .001;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);atoms()"+(reverse?"1 0":"0 1")+R"();}}diffusion{model none;})";
  const auto gas=ugkwp::parseGasModelProperties(gasText);
  Foam::IStringStream stream(material);Foam::dictionary d(stream);auto model=chmt::io::readModel(d,gas);
  chmt::Reaction reaction;reaction.A=1;reaction.condensedNu[0]=-.012;reaction.gasNu[0]=.012;
  std::string error;assert(chmt::validateReaction(reaction,model.physics,0,error));
  for(int e=0;e<2;++e){double balance=.012*(model.physics.species[0].element[e]-model.physics.condensed[0].element[e]);std::cout<<"reverse="<<reverse<<" element="<<e<<" balance="<<balance<<std::endl;assert(std::abs(balance)<1e-14);}
  for(int defect=0;malformedChecks&&defect<5;++defect){auto malformed=gas;
   auto input=material;
   if(defect==0)malformed.elementNames[1]=malformed.elementNames[0];
   if(defect==1)malformed.elementNames[0]="O";
   if(defect==2){malformed.elementNames.pop_back();malformed.elementComposition.resize(chmt::Ns);}
   if(defect==3)input.replace(input.find("elements (C H)"),14,"elements (C C)");
   if(defect==4)input.replace(input.find("elements (C H)"),14,"elements ()");
   bool refused=false;try{Foam::IStringStream bad(input);Foam::dictionary dictionary(bad);chmt::io::readModel(dictionary,malformed);}catch(const std::exception&){refused=true;}assert(refused);
  }
 }
 std::cout<<"PASS native element-name mapping, carbon reaction closure and malformed tables\n";
}
int main(int argc,char**){std::cout<<std::unitbuf;Foam::FatalError.throwExceptions();Foam::FatalIOError.throwExceptions();
 elementMapping(argc==1);if(argc>1)return 0;
 for(bool surface:{false,true})for(int n:{0,8,9}){bool refused=false;try{auto m=parse(n,surface,"");assert((surface?m.physics.material.nSurfaceReactions:m.physics.nReactions)==n);}catch(const std::exception&){refused=true;}assert(refused==(n==9));}
 for(int malformed:{1,2}){bool refused=false;try{parse(0,false,"",malformed);}catch(const std::exception&){refused=true;}assert(refused);}
 for(const char* model:{"constantTransport","reactingSst","equilibriumOde"}){
  auto m=parse(0,false,std::string("sst {wallTreatment boundaryLayer;boundaryLayer{model ")+model+";}}");
  int want=std::string(model)=="constantTransport"?0:std::string(model)=="reactingSst"?1:2;assert(int(m.physics.wallModel.model)==want);
 }
 for(const char* config:{"sst {wallTreatment wallFunction;}","sst {wallTreatment typo;}","sst {boundaryLayer{model typo;}}"}){bool refused=false;try{parse(0,false,config);}catch(const std::exception&){refused=true;}assert(refused);}
 std::cout<<"native OpenFOAM model parsing: capacity and explicit wall selection passed\n";
}
