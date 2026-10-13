#ifndef CHMT_RESTART_CHECKPOINT_H
#define CHMT_RESTART_CHECKPOINT_H
#include "core/HostState.H"
namespace chmt {
// Persist the current fixed-width layout, including thermal policy and carry.
// Metadata tags do not select historical conversions: reads retain positional
// decoding and require actual bytes for every consumed field.
bool writeCheckpoint(const std::string&,const ModelConfig&,const HostState&,std::string&);
bool readCheckpoint(const std::string&,const ModelConfig&,HostState&,std::string&);
}
#endif
