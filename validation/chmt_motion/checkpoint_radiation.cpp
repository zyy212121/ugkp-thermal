#include "restart/Checkpoint.C"
#include <cassert>
#include <iostream>
int main(int argc,char** argv){
 if(argc!=3)return 2;using namespace chmt;
 std::ifstream f(std::string(argv[1])+"/state.chmt",std::ios::binary|std::ios::ate);assert(f);
 Bytes bytes(size_t(f.tellg()));f.seekg(0);f.read(reinterpret_cast<char*>(bytes.data()),bytes.size());
 Decoder decoder(bytes);decoder.offset=72;ModelConfig config;decoder(config);
 assert(decoder.integer(8)==0x31474643504b4755ULL);
 ugkp::thermal::visitParticlePhaseConfiguration(decoder,config.particlePhase);
 ugkp::thermal::visitParticleContactConfiguration(decoder,config.particleContact);
 HostState state;std::string error;
 if(!readCheckpoint(argv[1],config,state,error)){std::cerr<<error;return 3;}
 assert(state.radiationTime>=0&&state.radiationTime<=state.time);
 state.radiationTime=state.time*.75;
 CountingEncoder extent;encodeStatePayload(extent,state);Encoder payload;encodeStatePayload(payload,state);
 assert(extent.bytes==payload.data.size());
 if(!writeCheckpoint(argv[2],config,state,error)){std::cerr<<error;return 4;}
 HostState resumed;if(!readCheckpoint(argv[2],config,resumed,error)){std::cerr<<error;return 5;}
 assert(state.radiationTime==resumed.radiationTime);
 Encoder before,after;before(state);after(resumed);assert(before.data==after.data);
 std::cout<<"CHECKPOINT_RADIATION_PASS schema="<<checkpointSchema<<" time="<<state.time<<" radiationTime="<<resumed.radiationTime<<"\n";
}
