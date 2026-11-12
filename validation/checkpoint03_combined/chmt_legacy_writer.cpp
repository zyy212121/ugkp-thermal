// Compile this translation unit separately against EACH historical FP64
// source closure, with namespaces renamed. Never include candidate headers.
// CHMT_LEGACY_CHECKPOINT is an absolute, frozen Checkpoint.C include.
#define CHMT_LEGACY_SEED 1
#include CHMT_LEGACY_CHECKPOINT
#include "chmt_seed.H"
extern "C" int CHMT_LEGACY_EXPORT(const char* directory,int carry){
 chmt::ModelConfig model;chmt::HostState h;chmtSeed(model,h,carry!=0);
 std::string error;
 if(!chmt::writeCheckpoint(directory,model,h,error)){std::cerr<<error<<'\n';return 1;}
 return 0;
}

#define CHMT_JOIN_I(a,b) a##b
#define CHMT_JOIN(a,b) CHMT_JOIN_I(a,b)
extern "C" int CHMT_JOIN(CHMT_LEGACY_EXPORT,_inspect)(const char* directory,double* carry){
 // Inspect wire compatibility using the archive's actual configuration, not
 // old double defaults that differ from widened FP32 configuration values.
 // The real historical readCheckpoint still validates and decodes the state.
 chmt::ModelConfig model;chmt::HostState read;std::string error;
 std::ifstream input(std::string(directory)+"/state.chmt",std::ios::binary|std::ios::ate);
 if(!input || input.tellg()<72)return 2;
 chmt::Bytes data(static_cast<std::size_t>(input.tellg()));input.seekg(0);
 input.read(reinterpret_cast<char*>(data.data()),data.size());if(!input)return 2;
 chmt::Decoder decoder(data);decoder.offset=72;decoder(model);
 if(decoder.integer(8)!=0x31474643504b4755ULL)return 3;
 ugkp::thermal::visitParticlePhaseConfiguration(decoder,model.particlePhase);
 ugkp::thermal::visitParticleContactConfiguration(decoder,model.particleContact);
 if(!chmt::readCheckpoint(directory,model,read,error)||read.particles.size()!=1){
  std::cerr<<error<<'\n';return 1;
 }
 *carry=read.particles[0].thermal.specificEnthalpyResidual;return 0;
}
