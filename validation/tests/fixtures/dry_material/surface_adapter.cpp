#include "../surface_terminal/fixture.hpp"
#include "ablation/DrySurfaceInterfaceMath.H"
#include <cmath>
#include <cstring>
void near(Real a,Real b){if(std::abs(a-b)>1e-10*std::max(Real(1),std::max(std::abs(a),std::abs(b)))){std::cerr<<a<<" != "<<b<<'\n';std::abort();}}
int main(){
 auto p=physics();auto h=host(p);auto in=input(h,p);
 MaterialPrimitive material;assert(recoverMaterial(in.solid,in.solidVolume,p,material));
 SurfacePhysicsResult old,dry;assert(solveSurfaceInterface(in,p,old));assert(solveDrySurfaceInterface(in,p,dry));
 near(old.topTemperature,dry.topTemperature);near(old.gasEnergy,dry.gasEnergy);near(old.thermalResidual,dry.thermalResidual);
 // Primary chemistry is per physical material area, not the reference area.
 p.material.enableSurfaceReactions=true;p.material.nSurfaceReactions=1;
 auto& reaction=p.material.surfaceReactions[0];reaction.A=2;reaction.condensedNu[0]=-.012;reaction.gasNu[0]=.012;
 in.dt=0;in.area=2;in.gasArea=in.solidArea=5;
 assert(evaluateDrySurfaceTemperature(in,p,material,0,700,false,dry));
 near(dry.condensed[0],.06);near(dry.gasSpecies[0],.06);near(dry.solidSpeed,-.024/1000);
 near(dry.poreSpecies[0],0);near(dry.gasSpecies[0]*in.area,.024*in.solidArea);
 assert(evaluateSurfaceTemperature(in,p,material,0,700,false,old));
 near(old.topTemperature,dry.topTemperature);near(old.gasEnergy,dry.gasEnergy);near(old.thermalResidual,dry.thermalResidual);
 // Signed reactant/product and unchanged core kinetics at nontrivial orders.
 if(Ns>1){reaction.gasNu[0]=-.016;reaction.gasNu[1]=.028;p.material.surfaceGasOrder[0][0]=1;
 auto provisional=in.bulk;auto production=drySurfaceProduction(in,provisional,700,p);
 const auto rate=surfaceArrheniusRate(reaction,p.material.surfaceGasOrder[0],in,provisional,700);
 near(production.condensed[0],-.012*rate);near(production.gas[0],-.016*rate);near(production.gas[1],.028*rate);
 p.material.surfaceGasOrder[0][0]=0;reaction.gasNu[0]=.012;reaction.gasNu[1]=0;}
 // Pore sweep out, Darcy influx: opposing channels retain separate donor energy.
 in.solid.condensed[0]=.5;in.solid.pore[0]=.0001;in.solidBase=in.solid;
 in.solid.energy=.5*condensedE(p.condensed[0],600)+.0001*speciesE(p.species[0],600);
 assert(recoverMaterial(in.solid,in.solidVolume,p,material));
 p.material.enablePoreOutflow=true;p.permeability=1e-12;p.poreViscosity=1e-5;
 in.useSweptGeometry=true;in.solidSweptRate=in.gasSweptRate=-1e-4;
 assert(evaluateDrySurfaceTemperature(in,p,material,0,700,false,dry));
 assert(dry.poreDarcy[0]<0&&dry.poreSweep[0]>0);
 near(dry.poreSpecies[0],dry.poreDarcy[0]+dry.poreSweep[0]);
 const Real donor=dry.poreDarcy[0]*speciesH(p.species[0],700)+dry.poreSweep[0]*speciesE(p.species[0],material.temperature);
 const Real solidDonor=dry.condensed[0]*condensedE(p.condensed[0],700);
 const Real ks=1/(in.solidDistance/material.conductivity+p.material.solidContactResistance);
 near(dry.thermalResidual,dry.gasEnergy+dry.poreEnergy-solidDonor-donor-(in.solidArea/in.area)*ks*(material.temperature-700)-dry.radiationFlux);
 near(dry.gasSpecies[0],.06); // no pore double counting in primary slot
 // Unequal-D material diffusion uses trial wall composition, mass-average
 // correction, and face-temperature h, not sign-selected net-flux donor h.
 if(Ns>1){
   p.material.enableSurfaceReactions=false;p.permeability=0;
   in.useSweptGeometry=false;in.solid.pore[0]=.00008;in.solid.pore[1]=.00002;
   in.solid.energy=.5*condensedE(p.condensed[0],600);
   for(int species=0;species<Ns;++species)in.solid.energy+=in.solid.pore[species]*speciesE(p.species[species],600);
   in.solidBase=in.solid;assert(recoverMaterial(in.solid,in.solidVolume,p,material));
   in.bulk.Y[0]=.2;in.bulk.Y[1]=.8;
   p.poreEffectiveDiffusivity[0]=1e-5;p.poreEffectiveDiffusivity[1]=2e-5;
   p.gasDiffusivity[0]=p.gasDiffusivity[1]=1e-3;
   assert(evaluateDrySurfaceTemperature(in,p,material,0,700,false,dry));
   near(dry.poreSpecies[0]+dry.poreSpecies[1],0);
   assert(dry.poreSpecies[0]>0&&dry.poreSpecies[1]<0);
   near(dry.gasSpecies[0],0);near(dry.gasSpecies[1],0);
   Real diffusionEnergy=0;for(int species=0;species<Ns;++species)
      diffusionEnergy+=dry.poreSpecies[species]*speciesH(p.species[species],650);
   near(dry.thermalResidual,dry.gasEnergy+dry.poreEnergy-diffusionEnergy
      -(in.solidArea/in.area)*ks*(material.temperature-700)-dry.radiationFlux);
 }
#if UGKWP_GAS_SPECIES >= 3
 // Three-species zero-support case: centered correction formerly exported
 // species 0 from a material cell containing none. Exercise the actual adapter.
 p=physics();p.material.enablePoreOutflow=true;p.permeability=0;
 h=host(p);in=input(h,p);in.dt=0;in.solid.condensed[0]=.5;
 in.solid.pore[0]=0;in.solid.pore[1]=in.solid.pore[2]=.00025;
 in.solid.energy=.5*condensedE(p.condensed[0],700);
 for(int species=0;species<Ns;++species)in.solid.energy+=in.solid.pore[species]*speciesE(p.species[species],700);
 in.solidBase=in.solid;in.solidDistance=1;
 in.bulk.temperature=700;in.bulk.pressure=287*700;
 in.bulk.Y[0]=.1;in.bulk.Y[1]=.9;in.bulk.Y[2]=0;
 p.poreEffectiveDiffusivity[0]=1e-6;p.poreEffectiveDiffusivity[1]=100e-6;p.poreEffectiveDiffusivity[2]=1e-6;
 for(int species=0;species<Ns;++species)p.gasDiffusivity[species]=1;
 assert(recoverMaterial(in.solid,in.solidVolume,p,material));
 assert(evaluateDrySurfaceTemperature(in,p,material,0,700,false,dry));
 assert(dry.poreSpecies[0]<0);Real sum=0;for(auto rate:dry.poreSpecies)sum+=rate;near(sum,0);
#endif
 // Actual host adapter ignores all retired film state and never manufactures packets.
 p=physics();h=host(p);h.filmAux.clear();p.enableFilm=true;p.material.enableMelting=true;
 CpuSurfaceResult cpu;std::string error;
 assert(evaluateCpuSurface(h,p,{bulk(h,p)},{},{},{},.01,1,cpu,error));
 assert(cpu.phasePackets.empty()&&!cpu.physics[0].wet);near(cpu.wall[0].temperature,cpu.physics[0].topTemperature);
 // Full host solve, not just fixed-T evaluator, accepts scalar area normalization.
 p.material.enableSurfaceReactions=true;p.material.nSurfaceReactions=1;
 p.material.surfaceReactions[0].A=2;p.material.surfaceReactions[0].condensedNu[0]=-.012;
 p.material.surfaceReactions[0].gasNu[0]=.012;
 h.surface.area[0]=2;h.gasMesh.areaVectors[0]={-5,0,0};h.solidMesh.areaVectors[0]={5,0,0};
 assert(evaluateCpuSurface(h,p,{bulk(h,p)},{},{},{},.001,2,cpu,error));
 near(cpu.wall[0].speciesRate[0],.06);near(cpu.wall[0].condensedRate[0],.06);
 near(cpu.wall[0].normalVelocity,-.024/1000);
 std::cout<<"dry surface host adapter: PASS\n";
}
