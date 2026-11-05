#pragma once
#if defined(__CUDACC__)
#define UGKP_TRACKING_HD __host__ __device__
#else
#define UGKP_TRACKING_HD
#endif
namespace ugkp { namespace thermal {
template<class Real> struct ParticleTrackingPlane {Real x,y,z,d;};
// The gasUGKP/CHT face-plane convention, including the existing SMALL term.
// A cell's neighbour-side normal is opposite the owner's normal.
template<class Real>
UGKP_TRACKING_HD inline ParticleTrackingPlane<Real> particleTrackingPlane(
    int cell,int owner,Real sx,Real sy,Real sz,Real area,
    Real cx,Real cy,Real cz,Real small) {
    const Real sign=owner==cell?Real(1):Real(-1),denominator=area+small;
    const Real nx=sign*(sx/denominator),ny=sign*(sy/denominator),nz=sign*(sz/denominator);
    return {nx,ny,nz,nx*cx+ny*cy+nz*cz};
}
}}
#undef UGKP_TRACKING_HD
