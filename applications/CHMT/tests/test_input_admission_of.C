// Actual OpenFOAM dictionary/token input and CHMT serializers; no solver execution.
#include "configuration/ModelIO.H"
#include "restart/Checkpoint.C"
#include "IStringStream.H"
#include <cassert>
#include <iostream>
#include <iomanip>
Foam::dictionary parseDictionary(const std::string& text){Foam::IStringStream input(text);return Foam::dictionary(input);}
ugkwp::GasModelConfiguration inputGas(){return ugkwp::parseGasModelProperties(R"(gasMode mixtureFrozen;species(S0 S1);speciesThermo{S0{model linearCp;molarMass .028;minTemperature 100;maxTemperature 3000;coefficients(1040 0 700);}S1{model linearCp;molarMass .032;minTemperature 100;maxTemperature 3000;coefficients(950 .1 -300);}}diffusion{model constant;coefficients(.001 .002);turbulentSchmidt .8;})");}
std::string modelText(bool bad){return std::string("modelName sample; materialSource supplied; mechanismSource canonical; condensedNames (C0 C1 ")+(bad?"extra":"")+"); elements (E0); minDt 0.01; maxDt 0.1; cfl 0.5; singleGasSpecies S0; filmThermalMode "+(bad?"alien":"ResolvedNormal")+"; materialReconstruction "+(bad?"alien":"SmoothVerification")+"; material { evaporation "+(bad?"alien":"HertzKnudsen")+"; } sst { wallTreatment "+(bad?"alien":"boundaryLayer")+"; boundaryLayer { model "+(bad?"alien":"constantTransport")+"; } } meshMotion { policy "+(bad?"alien":"Static")+"; } particle { condensedIndex 0; heatTransferCoefficient 1; contactHeatTransferCoefficient 2; contactDuration 3; contact "+(bad?"alien":"Deposit")+"; } liquid { rho 1000; cp0 1000; cp1 0; e0 0; conductivity 1; Tmin 100; Tmax 3000; element (1); }";}
void output(const chmt::ModelConfig&m){chmt::Encoder bytes;bytes(m);std::cout<<"MODEL "<<std::dec<<bytes.data.size()<<" ";for(auto b:bytes.data)std::cout<<std::hex<<std::setw(2)<<std::setfill('0')<<int(b);std::cout<<"\n";}
int main(int argc,char**argv){const std::string mode=argc>1?argv[1]:"normal";
 if(mode=="missing"){auto d=parseDictionary("x 1;");(void)chmt::io::readNumber(d,"missing");return 99;}
 if(mode=="token"){auto d=parseDictionary("x 42;");try{(void)chmt::io::text(d,"x");}catch(const std::runtime_error&e){std::cout<<e.what()<<"\n";return 7;}return 99;}
 if(mode=="short"){auto d=parseDictionary("x (1 2);");(void)chmt::io::readVector(d,"x");return 99;}
 if(mode=="normal"){for(bool single:{false,true}){auto d=parseDictionary(modelText(false));auto gas=inputGas();if(single){gas.mode=ugkwp::GasMode::SingleLegacy;for(auto& v:gas.diffusionCoefficients)v=0;gas.coefficients[1]=gas.coefficients[2]=gas.coefficients[4]=gas.coefficients[5]=0;}output(chmt::io::readModel(d,gas));}return 0;}
 assert(mode=="admitted");auto d=parseDictionary("x 3.5; v (1 2 3); a (4 5); element (6);");
 double a[3]={91,92,93};chmt::io::array(d,"a",a);assert(a[0]==4&&a[1]==5&&a[2]==93);double e[2]={94,95};chmt::io::elements(d,e,2);assert(e[0]==6&&e[1]==95);
 const double nan=std::numeric_limits<double>::quiet_NaN(),inf=std::numeric_limits<double>::infinity();
 // Supply complete typed OpenFOAM tokens, without relying on textual nan spelling.
 d.lookup("x")[0]=Foam::token(nan);assert(std::isnan(chmt::io::readNumber(d,"x")));
 d.lookup("v")[1]=Foam::token(inf);auto v=chmt::io::readVector(d,"v");assert(std::isinf(v.x)&&v.y==2&&v.z==3);
 d.lookup("a")[1]=Foam::token(-inf);auto numbers=chmt::io::scalars(d,"a");assert(std::isinf(numbers[0])&&numbers[0]<0&&numbers[1]==5);
 auto input=parseDictionary(modelText(true));auto m=chmt::io::readModel(input,inputGas());
 assert(m.physics.filmThermalMode==chmt::FilmThermalMode::ThicknessAveraged&&m.physics.reconstruction==chmt::ReconstructionMode::LimitedLinear);
 assert(m.physics.material.evaporation==chmt::EvaporationLaw::Disabled&&m.physics.wallModel.family==ugkwp::gaswall::WallFamily::LowRe&&m.physics.wallModel.model==ugkwp::gaswall::BoundaryLayerModel::ReactingSst);
 assert(m.physics.meshMotion.policy==chmt::MeshMotionPolicy::PrescribedSinusoidal&&m.physics.particle.contact==chmt::ParticleContactMode::ElasticRebound);
 auto badThermo=parseDictionary("rho -1; cp0 -2; cp1 0; e0 9; conductivity -1; Tmin 1000; Tmax 100; element (1);");auto t=chmt::io::condensed(badThermo,1);assert(t.rho==-1&&t.cp0==-2&&t.Tmin==1000&&t.Tmax==100);
 std::cout<<"ADMITTED typed nonfinite, supplied short arrays, foreign labels, actual invalid thermo\n";
}
