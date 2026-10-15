#include <new>
#include <cstdlib>
#include <cmath>
static unsigned long long allocs=0,bytes=0;
void* operator new(std::size_t n){++allocs;bytes+=n;if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept{std::free(p);}void operator delete(void* p,std::size_t)noexcept{std::free(p);}
#include "fixture.hpp"
int main(){auto p=physics();auto h=host(p);auto in=input(h,p);std::cout<<std::hexfloat;
 for(bool late:{false,true}){forceWindow=late;surfaceCalls=0;maxGrid=0;allocs=bytes=0;
  SurfacePhysicsResult result;assert(solveSurfaceInterface(in,p,result));resultOut(result);
  std::cout<<"COUNT "<<surfaceCalls<<' '<<maxGrid<<' '<<allocs<<' '<<bytes<<'\n';if(late)assert(maxGrid==512);
 }
 forceWindow=false;std::string error;CpuSurfaceResult result;surfaceCalls=0;maxGrid=0;allocs=bytes=0;
 assert(evaluateCpuSurface(h,p,{in.bulk},{},{},{},.01,1,result,error));resultOut(result.physics[0]);
 std::cout<<"CPUCOUNT "<<surfaceCalls<<' '<<maxGrid<<' '<<allocs<<' '<<bytes<<'\n';
 auto wet=in;p.enableFilm=true;p.material.enableMelting=true;p.material.phaseCondensed=0;p.material.phaseFilmY[0]=1;p.meltTemperature=500;p.liquid.e0=200000;
 wet.hasFilm=true;wet.film.mass=.2;wet.film.species[0]=.2;wet.film.enthalpy=.2*liquidH(p,650,in.bulk.pressure);wet.filmBase=wet.film;
 assert(recoverFilm(wet.film,1,in.bulk.pressure,p,wet.aux));surfaceCalls=0;maxGrid=0;allocs=bytes=0;
 SurfacePhysicsResult wetResult;assert(solveSurfaceInterface(wet,p,wetResult));assert(wetResult.phaseMass!=0&&wetResult.phaseEnergy!=0);resultOut(wetResult);
 std::cout<<"WETCOUNT "<<surfaceCalls<<' '<<maxGrid<<' '<<allocs<<' '<<bytes<<'\n';
 p=physics();
 MaterialPrimitive material;assert(recoverMaterial(in.solid,in.solidVolume,p,material));
 for(auto& species:p.species)species.Tmax=1e50;
 SurfacePhysicsResult left,right,rejected;assert(evaluateSurfaceTemperature(in,p,material,500,599.31,false,left));
 assert(evaluateSurfaceTemperature(in,p,material,500,1e40,false,right));
 surfaceCalls=0;rejected.topTemperature=-17;
 assert(!bisectFeasibleSurface(in,p,material,500,599.31,1e40,left,right,rejected));
 Real expected=1e40;for(int i=0;i<100;++i)expected=.5*(599.31+expected);
 assert(surfaceCalls==100&&rejected.topTemperature==-17&&lastCompleteT==expected);
 std::cout<<"BISECT100 "<<surfaceCalls<<' '<<lastCompleteT<<'\n';

}
