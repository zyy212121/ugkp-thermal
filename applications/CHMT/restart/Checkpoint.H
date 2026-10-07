#ifndef CHMT_RESTART_CHECKPOINT_H
#define CHMT_RESTART_CHECKPOINT_H
#include "core/HostState.H"
namespace chmt {
// Version/schema 2 persists bounded numerical solid sweep remainders exactly.
// Older files are rejected rather than silently creating a zero remainder.
bool writeCheckpoint(const std::string&,const ModelConfig&,const HostState&,std::string&);
bool readCheckpoint(const std::string&,const ModelConfig&,HostState&,std::string&);
}
#endif
