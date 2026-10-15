#ifndef CHMT_CONFIGURATION_COUPLINGENTRY_H
#define CHMT_CONFIGURATION_COUPLINGENTRY_H
#include "core/HostState.H"
namespace chmt {

inline bool validateCouplingEntry(const std::string&,const HostState&,std::string& error) {
    error.clear();return true;
}
}
#endif
