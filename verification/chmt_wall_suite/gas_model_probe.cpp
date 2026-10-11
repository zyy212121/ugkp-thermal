// Real production gas-model/chemistry parser. Import evidence only, never CUDA.
#include "common/gasTransport/GasMechanismIO.H"
#include <iostream>
#include <fstream>
#include <sstream>
static std::string read(const std::string&p){std::ifstream f(p);if(!f)throw std::runtime_error("cannot read "+p);std::ostringstream s;s<<f.rdbuf();return s.str();}
int main(int argc,char**argv){try{
 if(argc!=2){std::cerr<<"usage: gas-model-input-check CASE\n";return 2;}
 const std::string root=argv[1];auto model=ugkwp::parseGasModelProperties(read(root+"/constant/gasModelProperties"));
 if(model.speciesNames.size()!=UGKWP_GAS_SPECIES)throw std::runtime_error("compiled species mismatch");
 if(model.mode==ugkwp::GasMode::MixtureChemistry){auto mechanism=ugkwp::parseGasMechanismProperties<UGKWP_GAS_SPECIES>(read(root+"/constant/"+model.mechanism),model.thermoView<UGKWP_GAS_SPECIES>(),model.phase);std::cout<<"mechanismHash="<<mechanism.mechanismHash<<'\n';}
 std::cout<<"PRODUCTION_GAS_MODEL_IMPORT_ONLY Ns="<<UGKWP_GAS_SPECIES<<" speciesOrderHash="<<model.speciesOrderHash<<" thermoHash="<<model.thermoHash<<'\n';return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
