#pragma once
#include "../gasTransport/GasModel.H"
namespace ugkwp { namespace gaswall { namespace detail {
// Shared host/device launch contract; device primitive implementation is separate.
constexpr int wallWarpWidth=32;
// Host API and one-thread callers retain the original execution order.
// A cooperative team supplies rank/size, a full-team barrier, all and broadcast.
// Every member must enter and leave each collective in the same order.
struct SerialWallTeam {
    UGKWP_GAS_HD int rank()const{return 0;}
    UGKWP_GAS_HD int size()const{return 1;}
    UGKWP_GAS_HD bool leader()const{return true;}
    UGKWP_GAS_HD void sync()const{}
    UGKWP_GAS_HD bool all(bool value)const{return value;}
    template<class T> UGKWP_GAS_HD T broadcast(T value)const{return value;}
};
}}}
