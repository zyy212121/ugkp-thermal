#ifndef CHMT_RESTART_CHECKPOINT_H
#define CHMT_RESTART_CHECKPOINT_H
#include "core/HostState.H"
namespace chmt {
// Schema 3 persists thermal policy and bounded solid sweep remainders exactly.
// Older files are rejected rather than guessing a thermal operator from values.
bool writeCheckpoint(const std::string&,const ModelConfig&,const HostState&,std::string&);
bool readCheckpoint(const std::string&,const ModelConfig&,HostState&,std::string&);
}
#endif
