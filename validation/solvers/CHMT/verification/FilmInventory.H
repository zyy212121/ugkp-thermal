#pragma once
#include "core/Types.H"
namespace chmt { namespace verification {
inline long double filmKineticEnergy(const FilmQ& q,const FilmAux& a) {
    if(q.mass==0)return 0;
    long double base[3]={a.baseVelocity.x,a.baseVelocity.y,a.baseVelocity.z};
    const long double mean[3]={a.meanVelocity.x,a.meanVelocity.y,a.meanVelocity.z};
    const long double top[3]={a.topVelocity.x,a.topVelocity.y,a.topVelocity.z};
    const long double normal[3]={a.normal.x,a.normal.y,a.normal.z};
    long double bottomNormal=0,meanNormal=0,topNormal=0;
    for(int axis=0;axis<3;++axis) {
        bottomNormal+=base[axis]*normal[axis];
        meanNormal+=mean[axis]*normal[axis];
        topNormal+=top[axis]*normal[axis];
    }
    for(int axis=0;axis<3;++axis)
        base[axis]+=normal[axis]*(2*meanNormal-topNormal-bottomNormal);
    const long double weights[5]={7,32,12,32,7};
    long double integral=0;
    for(int axis=0;axis<3;++axis) {
        const long double linear=6*mean[axis]-4*base[axis]-2*top[axis];
        const long double quadratic=3*(base[axis]+top[axis]-2*mean[axis]);
        for(int i=0;i<5;++i) {
            const long double z=static_cast<long double>(i)/4;
            const long double velocity=base[axis]+linear*z+quadratic*z*z;
            integral+=weights[i]*velocity*velocity;
        }
    }
    return static_cast<long double>(q.mass)*integral/180;
}
}}
