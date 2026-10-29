#pragma once
// Shared reduced-BVP energy identity. Species flux is supplied by its operator.
#include "../ReactingWallLayer.H"
#include "PhysicalPrimitiveFlux.H"
namespace ugkwp { namespace gaswall { namespace coherent_energy {
template<class R> struct Terms {R conductivity=0,advective=0,stress=0;};
template<class R> struct Integral {
 R resistance=0,advective=0,stress=0;
 UGKWP_GAS_HD void add(R weight,const Terms<R>& t){R r=weight/t.conductivity;resistance+=r;advective+=r*t.advective;stress+=r*t.stress;}
 UGKWP_GAS_HD R energy(R temperatureJump)const{return (advective+stress-temperatureJump)/resistance;}
};
// Species flux is explicit: caller can supply the SAME flux representation used
// by its species operator. This avoids silently imposing constant species flux
// on a reacting interval. No extra material evaluations or rejection policy.
template<class R,int N,class T>
UGKWP_GAS_HD Terms<R> terms(const ugkwp::gaswall::WallInput<R,N,T>&in,
 const ugkwp::gaswall::WallModelConfig<R>&cfg,
 const ugkwp::gaswall::detail::LayerContext<R,N>&ctx,
 const ugkwp::gaswall::physical::Primitive<R,N>&v,R nut,
 const R (&momentum)[2],const R (&speciesFlux)[N]){
 Terms<R> out;R cp=0,kinetic=0,normal=0;
 for(int j=0;j<N;++j){R c,h;ugkwp::speciesCpH(j,v.T,in.model.thermo,c,h);cp+=v.Y[j]*c;out.advective+=speciesFlux[j]*((ctx.wallEnthalpy[j]-ctx.energyReference)+ugkwp::gaswall::detail::speciesEnthalpyIncrement(j,in.temperature,v.T-in.temperature,in.model.thermo,ctx.enthalpyJump[j]));}
 for(int a=0;a<3;++a)normal+=v.U[a]*in.normal[a];
 kinetic=normal*normal/R(2);
 for(int d=0;d<2;++d){R u=-ctx.wallU[d];for(int a=0;a<3;++a)u+=v.U[a]*ctx.tangent[d][a];kinetic+=u*u/R(2);out.stress-=(ctx.mass*u-momentum[d])*u;}
 out.advective+=ctx.mass*kinetic;
 out.conductivity=(in.model.molecularPrandtl>0?wallMolecularViscosity(in,v.T,v.Y)*cp/wallMolecularPrandtl(in,v.Y):in.model.conductivity)+v.rho*nut*cp/cfg.turbulentPrandtl;
 return out;
}
// Optional pointwise physical species flux, using exactly the existing corrected
// diffusion helper. Return value forwards its existing availability semantics.
template<class R,int N,class T>
UGKWP_GAS_HD bool physicalSpecies(const ugkwp::gaswall::WallInput<R,N,T>&in,
 const ugkwp::gaswall::WallModelConfig<R>&cfg,
 const ugkwp::gaswall::detail::LayerContext<R,N>&ctx,
 const ugkwp::gaswall::physical::Primitive<R,N>&v,R nut,R (&flux)[N]){
 R grad[N]{},diff[N]{},diffH=0;for(int j=0;j<N;++j)for(int a=0;a<3;++a)grad[j]+=v.gradY[j][a]*in.normal[a];bool available=false;
 R gasR=0,localD[N];for(int j=0;j<N;++j)gasR+=v.Y[j]*ugkwp::universalGasConstant<R>()/in.model.thermo.species[j].molarMass;
 for(int j=0;j<N;++j)localD[j]=wallMolecularDiffusivity(in,j,v.T,v.rho*gasR*v.T);
 bool valid=ugkwp::correctedSpeciesDiffusion(v.Y,grad,v.rho,localD,nut,cfg.turbulentSchmidt,v.T,in.model.thermo,diff,diffH,true,&available);
 for(int j=0;j<N;++j)flux[j]=ctx.mass*v.Y[j]+diff[j];
 return valid&&available;
}
template<class R,int N,class T>
UGKWP_GAS_HD R wallTransport(const ugkwp::gaswall::WallInput<R,N,T>&in,
 const ugkwp::gaswall::detail::LayerContext<R,N>&ctx,R wallNormalVelocity){
 R result=ctx.mass*wallNormalVelocity*wallNormalVelocity/R(2);
 for(int j=0;j<N;++j)result+=in.massFlux[j]*(ctx.wallEnthalpy[j]-ctx.energyReference);
 return result;
}
}}} // namespace ugkwp::gaswall::coherent_energy
