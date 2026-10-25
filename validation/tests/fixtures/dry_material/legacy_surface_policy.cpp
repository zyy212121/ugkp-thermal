#include "../surface_terminal/fixture.hpp"
int main(){auto p=physics();auto h=host(p);auto in=input(h,p);std::cout<<std::hexfloat;
 for(Real temperature:{300.,813.37,1500.}){
  in.bulk.temperature=temperature;in.bulk.pressure=287*temperature;
  SurfacePhysicsResult result;DeviceStatus status;bool numerical=false;
  const bool success=solveSurfaceInterface(in,p,result,&status,&numerical);
  std::cout<<success<<' '<<status.code<<' '<<numerical<<' ';if(success)resultOut(result);
 }
 // Retained legacy wet device path is not replaced by the host dry evaluator.
 p.enableFilm=true;p.material.enableMelting=true;p.material.phaseCondensed=0;
 p.material.phaseFilmY[0]=1;p.meltTemperature=500;p.liquid.e0=200000;
 in.hasFilm=true;in.film.mass=.2;in.film.species[0]=.2;
 in.film.enthalpy=.2*liquidH(p,650,in.bulk.pressure);in.filmBase=in.film;
 assert(recoverFilm(in.film,1,in.bulk.pressure,p,in.aux));
 SurfacePhysicsResult result;DeviceStatus status;bool numerical=false;
 const bool success=solveSurfaceInterface(in,p,result,&status,&numerical);
 std::cout<<success<<' '<<status.code<<' '<<numerical<<' ';if(success)resultOut(result);
}
