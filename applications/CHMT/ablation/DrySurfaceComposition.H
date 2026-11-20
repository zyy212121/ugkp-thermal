#pragma once
#include "core/Types.H"

namespace chmt {
template<class Scalar>
CHMT_HD Scalar relaxDrySurfaceComposition(Scalar (&current)[Ns],const Scalar (&closed)[Ns],
    Scalar relaxation,bool acceptedFlux){
    if(acceptedFlux)return 0;
    Scalar change=0;
    for(int s=0;s<Ns;++s){
        const Scalar difference=closed[s]-current[s];
        const Scalar magnitude=difference<0?-difference:difference;
        if(magnitude>change)change=magnitude;
        current[s]=relaxation==Scalar(.5)?Scalar(.5)*(current[s]+closed[s])
            :(Scalar(1)-relaxation)*current[s]+relaxation*closed[s];
    }
    return change;
}
}
