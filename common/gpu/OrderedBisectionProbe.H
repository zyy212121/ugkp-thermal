#pragma once
#include <limits>
#include "gasTransport/GasModel.H"
namespace ugkp { namespace gpu {
// The midpoint and every possible midpoint in the next Depth-1 rounds.
// A collective evaluator may
// evaluate these independent probes together; consumption still follows the
// original bisection decisions, including invalid states and early stopping.
template<class Real,int Depth=2> struct OrderedBisectionProbe {
    static_assert(Depth>0&&Depth<std::numeric_limits<unsigned>::digits-1,"dyadic index must fit int");
    static constexpr int count=(1<<Depth)-1;
    Real temperature[count]{};bool ready=false;
    UGKWP_GAS_HD int find(Real value)const {
        if(ready)for(int i=0;i<count;++i)if(value==temperature[i])return i;
        return -1;
    }
    UGKWP_GAS_HD void prepare(Real lo,Real hi){
        Real left[count],right[count];left[0]=lo;right[0]=hi;
        for(int i=0;i<count;++i){
            temperature[i]=Real(.5)*(left[i]+right[i]);
            if(2*i+1<count){
                left[2*i+1]=left[i];right[2*i+1]=temperature[i];
                left[2*i+2]=temperature[i];right[2*i+2]=right[i];
            }
        }
        ready=true;
    }
};
}}
