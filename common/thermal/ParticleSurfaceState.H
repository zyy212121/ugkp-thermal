#pragma once
#if defined(__CUDACC__)
#define UGKP_SURFACE_HD __host__ __device__
#else
#define UGKP_SURFACE_HD
#endif
namespace ugkp { namespace thermal {
// Borrowed material surface fields, independent of gas injection/ejection.
// A null view retains the existing gas-wall fields for applications whose
// impermeable wall uses the same temperature and mechanical velocity.
struct ParticleSurfaceState {
    const double* temperature=nullptr;
    const double* velocityX=nullptr;
    const double* velocityY=nullptr;
    const double* velocityZ=nullptr;
    const double* effusivity=nullptr; // sqrt(k rho Cp), W s^(1/2)/(m^2 K).
};
// Borrowed cumulative heat ledgers [J] and contact area data. A consumer
// snapshots/differences the ledgers on its stream; reading does not reset them.
struct ParticleWallExchangeView {
    const double* depositedEnergy=nullptr;
    const double* reflectedEnergy=nullptr;
    const double* representedContactArea=nullptr;
    const double* contactAreaScale=nullptr;
};
ParticleWallExchangeView particleWallExchange(void* owner);
int bindParticleSurface(void* owner,int nFaces,const ParticleSurfaceState& surface);
}}
namespace ugkwp { namespace particleSurfaceDetail {
template<class S,class Real> UGKP_SURFACE_HD auto effusivity(const S& s,int f,Real legacy,int)
 ->decltype(s.particleSurface.effusivity,Real{}) {return s.particleSurface.effusivity?s.particleSurface.effusivity[f]:legacy;}
template<class S,class Real> UGKP_SURFACE_HD Real effusivity(const S&,int,Real legacy,long){return legacy;}

#define UGKP_SURFACE_ACCESSOR(Name,Field,GasField) \
 template<class S> UGKP_SURFACE_HD auto Name(const S& s,int f,int) \
 ->decltype(s.particleSurface.Field,s.GasField[f]+s.GasField[f]) \
 {return s.particleSurface.Field?s.particleSurface.Field[f]:s.GasField[f];} \
 template<class S> UGKP_SURFACE_HD auto Name(const S& s,int f,long) \
 ->decltype(s.GasField[f]+s.GasField[f]) {return s.GasField[f];}
UGKP_SURFACE_ACCESSOR(temperature,temperature,gasBoundaryT)
UGKP_SURFACE_ACCESSOR(velocityX,velocityX,gasBoundaryUx)
UGKP_SURFACE_ACCESSOR(velocityY,velocityY,gasBoundaryUy)
UGKP_SURFACE_ACCESSOR(velocityZ,velocityZ,gasBoundaryUz)
#undef UGKP_SURFACE_ACCESSOR
}
template<class S,class Real> UGKP_SURFACE_HD Real particleWallEffusivity(const S& s,int f,Real legacy){
    return particleSurfaceDetail::effusivity(s,f,legacy,0);
}
#define UGKP_SURFACE_PUBLIC(Name,Impl) \
 template<class S> UGKP_SURFACE_HD auto Name(const S& s,int f) \
 ->decltype(particleSurfaceDetail::Impl(s,f,0)) {return particleSurfaceDetail::Impl(s,f,0);}
UGKP_SURFACE_PUBLIC(particleWallTemperature,temperature)
UGKP_SURFACE_PUBLIC(particleWallVelocityX,velocityX)
UGKP_SURFACE_PUBLIC(particleWallVelocityY,velocityY)
UGKP_SURFACE_PUBLIC(particleWallVelocityZ,velocityZ)
#undef UGKP_SURFACE_PUBLIC
}
#undef UGKP_SURFACE_HD
