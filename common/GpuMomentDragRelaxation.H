#pragma once
#if defined(__CUDACC__)
#define UGKWP_MOMENT_HD __host__ __device__
#else
#define UGKWP_MOMENT_HD
#endif
#include <cmath>
namespace ugkwp {
template<class R> struct MomentDragRelaxationResult {
 R velocity[3]{};R intrinsicGasEnergy=0,alphaInternal=1;
};
// Shared source-only finite-time Eulerian moment closure. Inputs are extensive
// phase masses/momenta/energies (or densities per common volume); no state is
// mutated, so the same operator serves macro updates and frozen wall-node CVs.
template<class R> UGKWP_MOMENT_HD inline MomentDragRelaxationResult<R>
relaxGasFromParticleMoments(R mg,R ms,R epsG,R epsGsafe,R gasEnergyIntrinsic,
 R gasInternalFloor,const R ug[3],const R particleMomentum[3],R particleEnergy,
 R betaDt){
 MomentDragRelaxationResult<R> out;out.intrinsicGasEnergy=gasEnergyIntrinsic;
 for(int d=0;d<3;++d)out.velocity[d]=ug[d];
 if(!(mg>R(0))||!(ms>R(0))||!(epsG>R(0))||!(epsGsafe>R(0))||!(betaDt>R(0)))return out;
 const R us[3]={particleMomentum[0]/ms,particleMomentum[1]/ms,particleMomentum[2]/ms};
 const R kg0=R(.5)*mg*(ug[0]*ug[0]+ug[1]*ug[1]+ug[2]*ug[2]);
 const R ks0=R(.5)*ms*(us[0]*us[0]+us[1]*us[1]+us[2]*us[2]);
 R ig0=epsG*gasEnergyIntrinsic-kg0;
 bool repair=!std::isfinite(double(ig0))||ig0<gasInternalFloor;
 if(repair)ig0=gasInternalFloor;
 R is0=particleEnergy-ks0;if(!std::isfinite(double(is0))||is0<R(0))is0=R(0);
 const R alphaI=R(::exp(double(-R(2)*betaDt)));
 out.alphaInternal=alphaI;
 const R is1=is0*alphaI,ig1=ig0+(is0-is1);
 const R wx=ug[0]-us[0],wy=ug[1]-us[1],wz=ug[2]-us[2];
 if(!repair&&wx==R(0)&&wy==R(0)&&wz==R(0)&&is0==is1)return out;
 const R alphaW=R(::exp(double(-betaDt*(R(1)+ms/(mg+R(1e-300))))));
 const R w1[3]={wx*alphaW,wy*alphaW,wz*alphaW};
 const R p[3]={mg*ug[0]+particleMomentum[0],mg*ug[1]+particleMomentum[1],mg*ug[2]+particleMomentum[2]};
 const R mt=mg+ms;
 const R us1[3]={(p[0]-mg*w1[0])/mt,(p[1]-mg*w1[1])/mt,(p[2]-mg*w1[2])/mt};
 for(int d=0;d<3;++d)out.velocity[d]=us1[d]+w1[d];
 const R kg1=R(.5)*mg*(out.velocity[0]*out.velocity[0]+out.velocity[1]*out.velocity[1]+out.velocity[2]*out.velocity[2]);
 const R ks1=R(.5)*ms*(us1[0]*us1[0]+us1[1]*us1[1]+us1[2]*us1[2]);
 R diss=(kg0+ks0)-(kg1+ks1);if(!std::isfinite(double(diss))||diss<R(0))diss=R(0);
 out.intrinsicGasEnergy=(kg1+ig1+diss)/epsGsafe;
 return out;
}
}
#undef UGKWP_MOMENT_HD
