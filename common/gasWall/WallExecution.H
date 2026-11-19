#pragma once
#include "../gasTransport/GasModel.H"
namespace ugkwp { namespace gaswall { namespace detail {

constexpr int wallWarpWidth=32;



struct SerialWallTeam {
    UGKWP_GAS_HD int rank()const{return 0;}
    UGKWP_GAS_HD int size()const{return 1;}
    UGKWP_GAS_HD bool leader()const{return true;}
    UGKWP_GAS_HD void sync()const{}
    UGKWP_GAS_HD bool all(bool value)const{return value;}
    template<class T> UGKWP_GAS_HD T sum(T value)const{return value;}
    template<class T> UGKWP_GAS_HD T maximum(T value)const{return value;}
    template<class T> UGKWP_GAS_HD T broadcast(T value)const{return value;}
};
}}}
