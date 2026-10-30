#pragma once
#include "WallReferenceSolver.H"
namespace ugkwp { namespace gaswall {
// Metadata belongs to a physical wall, never to a pooled Newton worker.
template<int N> struct WallHistoryFrame {
 bool valid=false,adiabatic=false;int nodes=0;double distance=0,stretch=0,temperature=0;
 double normal[3]{},tangent[2][3]{},wallU[2]{};int independent[N-1]{},dependent=0;
 unsigned long long speciesOrder=0,thermoHash=0;
 int model=0;bool enableSst=false,localStressCorrection=false;double wallStress=0;
 double turbulentPrandtl=0,turbulentSchmidt=0,equilibriumKappa=0,equilibriumAPlus=0;
};
template<class R,int N,class T> UGKWP_GAS_HD inline WallHistoryFrame<N> historyFrame(
 const WallInput<R,N,T>&in,const WallModelConfig<R>&c,const detail::LayerContext<R,N>&ctx){
 WallHistoryFrame<N> h;h.valid=true;
 h.localStressCorrection=c.localStressCorrection;h.model=int(c.model);h.enableSst=c.enableSst;h.thermoHash=in.model.thermo.thermoHash;
 h.turbulentPrandtl=c.turbulentPrandtl;h.turbulentSchmidt=c.turbulentSchmidt;
 h.equilibriumKappa=c.equilibriumKappa;h.equilibriumAPlus=c.equilibriumAPlus;h.adiabatic=in.adiabatic;h.nodes=c.nodes;h.distance=in.matchingDistance;h.stretch=c.stretch;h.temperature=in.temperature;h.speciesOrder=in.model.thermo.speciesOrderHash;h.dependent=ctx.dependent;
 for(int d=0;d<3;++d)h.normal[d]=in.normal[d];
 for(int a=0;a<2;++a){h.wallU[a]=ctx.wallU[a];for(int d=0;d<3;++d)h.tangent[a][d]=ctx.tangent[a][d];}
 for(int j=0;j<N-1;++j)h.independent[j]=ctx.independent[j];
 return h;
}
template<int N> UGKWP_GAS_HD inline bool compatibleHistory(const WallHistoryFrame<N>&old,const WallHistoryFrame<N>&now){
 for(int d=0;d<3;++d)if(old.normal[d]!=now.normal[d])return false;
 return old.valid&&old.localStressCorrection==now.localStressCorrection&&old.model==now.model&&old.enableSst==now.enableSst&&old.thermoHash==now.thermoHash
     &&old.turbulentPrandtl==now.turbulentPrandtl&&old.turbulentSchmidt==now.turbulentSchmidt
     &&old.equilibriumKappa==now.equilibriumKappa&&old.equilibriumAPlus==now.equilibriumAPlus&&old.adiabatic==now.adiabatic&&old.nodes==now.nodes&&old.distance==now.distance&&old.stretch==now.stretch&&old.speciesOrder==now.speciesOrder;
}
// Preserve physical T, tangential velocity and composition when the algebraic
// coordinates change. This is not an endpoint lift or a Picard iteration.
template<class R,int N> UGKWP_GAS_HD inline void transformHistoryRow(
 const R*src,R*dst,const WallHistoryFrame<N>&old,const WallHistoryFrame<N>&now){
 for(int j=0;j<N+4;++j)dst[j]=src[j];
 bool identical=old.temperature==now.temperature&&old.dependent==now.dependent;
 for(int a=0;a<2;++a){identical=identical&&old.wallU[a]==now.wallU[a];for(int d=0;d<3;++d)identical=identical&&old.tangent[a][d]==now.tangent[a][d];}
 for(int j=0;j<N-1;++j)identical=identical&&old.independent[j]==now.independent[j];
 if(identical)return;
 for(int a=0;a<2;++a){R u=0;for(int d=0;d<3;++d)for(int b=0;b<2;++b)u+=(src[b]+old.wallU[b])*old.tangent[b][d]*now.tangent[a][d];dst[a]=u-now.wallU[a];}
 dst[2]=src[2]+old.temperature-now.temperature;
 R y[N]{};y[old.dependent]=1;for(int j=0;j<N-1;++j){y[old.independent[j]]=src[3+j];y[old.dependent]-=src[3+j];}
 for(int j=0;j<N-1;++j)dst[3+j]=y[now.independent[j]];
}
}}
