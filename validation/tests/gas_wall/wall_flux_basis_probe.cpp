#include "validation/tests/gas_wall/fixtures/sst_restart004_capture.H"
#include "common/gasWall/reconstruction/ConservativeProfile.H"
#include <cassert>
#include <cstring>
#include <iostream>
int main(){using namespace ugkwp::gaswall;using namespace ugkwp::gaswall::reconstruction;int cases=0;
 for(bool sst:{false,true})for(bool prandtl:{false,true})for(bool diffusion:{false,true})for(bool blowing:{false,true}){CapturedSstWall m;auto cfg=m.config;cfg.coherentDiscretization=true;cfg.enableSst=sst;if(!prandtl)m.input.model.molecularPrandtl=0;if(!diffusion)m.input.model.diffusivity[0]=m.input.model.diffusivity[1]=0;if(blowing){m.input.massFlux[0]=.0001;m.input.massFlux[1]=.0002;}WallWorkspace<double,2,24>w{};loadCapturedSstIterate(w);Profile direct(m.input,cfg,w),cached(m.input,cfg,w);Segment seg{};seg.lo=0;seg.hi=w.y[23];direct.segments.push_back(seg);cached.segments.push_back(seg);direct.wallSpeciesSlopeDelta0=cached.wallSpeciesSlopeDelta0=.123;auto basis=cached.wallFluxBasis();
  for(int i=0;i<20;++i){WallOutput<double,2>out;out.conductiveHeatFlux=-1200+147*i;out.traction[0]=.01*i;out.traction[1]=-.001*i;out.traction[2]=-.003*i;direct.setWallFlux(out);cached.setWallFlux(out,basis);for(int j=0;j<3;++j)assert(direct.wallSlopeDelta[j]==cached.wallSlopeDelta[j]);assert(direct.wallSpeciesSlopeDelta0==cached.wallSpeciesSlopeDelta0);for(double q:{.1,.25,.63,1.}){auto y=q*w.y[1];auto a=direct.base(y),b=cached.base(y);assert(std::memcmp(&a,&b,sizeof a)==0);auto c=direct.physicalPrimitive(y),d=cached.physicalPrimitive(y);assert(std::memcmp(&c,&d,sizeof c)==0);}++cases;}}
 std::cout<<cases<<" cached flux closures match direct FP64 primitives bitwise\n";
}
