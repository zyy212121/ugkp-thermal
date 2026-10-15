#include "fixture.hpp"
#include <sstream>
std::string emitted(const SurfacePhysicsResult& r){std::ostringstream text;auto* previous=std::cout.rdbuf(text.rdbuf());std::cout<<std::hexfloat;resultOut(r);std::cout.rdbuf(previous);return text.str();}
int main(){auto p=physics();auto h=host(p);auto in=input(h,p);SurfacePhysicsResult result,reference;bool numerical=false;
 // All grid evaluations fail ordinary numerical admission. The actual final
 // requested T is 3000; terminal computes its whole 60-iteration composition tail.
 p.tolerances.relativeMass=-1;surfaceCalls=0;
 assert(!solveSurfaceInterface(in,p,result));const auto strict=surfaceCalls;
 surfaceCalls=0;assert(solveSurfaceInterface<true>(in,p,result,nullptr,&numerical));
 assert(surfaceCalls==2*strict+1&&result.topTemperature==3000);
 MaterialPrimitive material;assert(recoverMaterial<true>(in.solid,in.solidVolume,p,material));
 assert(evaluateSurfaceTemperature<true>(in,p,material,p.meltTemperature,3000,false,reference));
 assert(emitted(result)==emitted(reference));assert(result.gasEnergy!=0&&result.thermalResidual!=0);
 std::cout<<"all_failed_search "<<strict<<" terminal_calls "<<2*strict+1<<" actual_T "<<result.topTemperature<<'\n';
 // Earlier completed candidate beats later incomplete attempts, after every
 // original grid search has run. No criterion-valid result is required.
 p=physics();forceNoBracket=true;failAbove=2000;surfaceCalls=0;
 assert(!solveSurfaceInterface(in,p,result));const auto failed=surfaceCalls;
 surfaceCalls=0;assert(solveSurfaceInterface<true>(in,p,result));
 assert(surfaceCalls==2*failed&&result.topTemperature<2000&&result.topTemperature>1990);
 assert(evaluateSurfaceTemperature(in,p,material,p.meltTemperature,result.topTemperature,false,reference));
 assert(emitted(result)==emitted(reference));std::cout<<"last_complete_T "<<result.topTemperature<<" last_attempt_T 3000\n";
 forceNoBracket=false;failAbove=1e100;
 // Last residual-evaluated, non-improving trial is not the pre-search Y or
 // the unevaluated proposal following the final alpha halving.
 p.gasDiffusivity[0]=p.gasDiffusivity[1]=.01;Real Y[Ns]{},J[Ns]{},out[Ns]{};Y[0]=Y[1]=.5;J[0]=.1;
 forceFlatResidual=true;residualCalls=0;assert(!interfaceComposition(J,Y,100000,500,.5,p,1e-12,out));
 assert(interfaceComposition<true>(J,Y,100000,500,.5,p,1e-12,out));
 bool differs=false;for(int s=0;s<Ns;++s){assert(out[s]==lastResidualY[s]);differs=differs||out[s]!=Y[s];}assert(differs);
 forceFlatResidual=false;std::cout<<"last_line_trial actual_residual_calls "<<residualCalls<<'\n';
 Real negativeY[Ns]{},flux[Ns]{};negativeY[0]=-1;FilmQ liquid;liquid.mass=.2;liquid.species[0]=.2;
 p.material.evaporationAccommodation[0]=1;p.material.saturationPressureScale[0]=200000;
 assert(hertzKnudsenFlux<true>(liquid,negativeY,100000,500,p,flux));
 assert(flux[0]==100000/::sqrt(2*3.14159265358979323846*p.species[0].R*500));
 // Actual CPU proposals, surplus metadata and missing consumed input.
 p=physics();p.tolerances.relativeMass=-1;CpuSurfaceResult cpu;std::string error;
 assert(detail::completeCpuSurface(h,p,{in.bulk},{},{},{},.01,7,cpu,error,nullptr,nullptr,nullptr,&numerical));
 assert(cpu.wall.size()==1&&cpu.physics.size()==1&&cpu.wall[0].temperature==cpu.physics[0].topTemperature);
 // Wet phase production must execute its actual packet tail too.
 auto wetPhysics=physics();wetPhysics.enableFilm=true;wetPhysics.material.enableMelting=true;
 wetPhysics.material.phaseCondensed=0;wetPhysics.material.phaseFilmY[0]=1;wetPhysics.meltTemperature=500;wetPhysics.liquid.e0=200000;
 auto wetHost=host(wetPhysics);wetHost.film.resize(1);wetHost.film[0].mass=.2;wetHost.film[0].species[0]=.2;
 wetHost.film[0].enthalpy=.2*liquidH(wetPhysics,650,in.bulk.pressure);
 assert(recoverFilm(wetHost.film[0],1,in.bulk.pressure,wetPhysics,wetHost.filmAux[0]));wetPhysics.tolerances.relativeMass=-1;
 assert(detail::completeCpuSurface(wetHost,wetPhysics,{in.bulk},{},{},{},.01,9,cpu,error,nullptr,nullptr,nullptr,&numerical));
 assert(cpu.phasePackets.size()==1);const auto& phase=cpu.phasePackets[0];const auto& actual=cpu.physics[0];
 assert(phase.mass==.01*actual.phaseMass&&phase.energy==.01*actual.phaseEnergy);
 assert(phase.conductive==.01*actual.phaseConductive&&phase.advective==.01*actual.phaseAdvective);
 assert(phase.liquidKineticAdvection==.01*actual.bottomKineticAdvection&&phase.mass!=0);
 assert(cpu.wall[0].temperature==actual.topTemperature&&phase.step==9&&phase.face==7);
 h.filmAux.clear();assert(!detail::completeCpuSurface(h,p,{in.bulk},{},{},{},.01,7,cpu,error,nullptr,nullptr,nullptr,&numerical)&&!numerical);
 p.species[0].model=static_cast<ugkwp::SpeciesThermoModel>(99);assert(!solveSurfaceInterface<true>(in,p,result,nullptr,&numerical)&&!numerical);
}
