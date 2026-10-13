#include <new>
#include <cstdlib>
static unsigned long long allocations=0,allocatedBytes=0;
void* operator new(std::size_t n){++allocations;allocatedBytes+=n;if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void* p)noexcept{std::free(p);}void operator delete(void* p,std::size_t)noexcept{std::free(p);}
#include "owner.C"
#include "materials/ConductionContinuation.H"
#include "fixture.hpp"
template<class T>void emit(T x){std::cout<<x<<' ';}void emit(Vec3 v){emit(v.x);emit(v.y);emit(v.z);}
template<class T>void values(const std::vector<T>& v){std::cout<<v.size()<<' ';for(auto x:v)emit(x);}
void programOut(const WallProgram& program){
 std::cout<<program.interval.identity.sequence<<' '<<program.interval.identity.epoch<<' ';emit(program.interval.begin);emit(program.interval.end);
 const auto& s=program.surface;
 values(s.prescribedTopTraction);
 values(s.prescribedPressureGradient);
 values(s.edgeOwnerOffset);
 values(s.edgeNeighbourOffset);
 values(s.gasDistance);
 values(s.solidDistance);
 values(s.gasFace);
 values(s.solidFace);
 values(s.solidCell);
 values(s.edgeOwner);
 values(s.edgeNeighbour);
 values(s.persistentId);
 values(s.edgeLength);
 values(s.area);
 values(s.oldArea);
 values(s.sweptEdgeArea);
 values(s.edgeConormal);
 values(s.centre);
 values(s.normal);
 values(s.meshVelocity);
 values(s.baseVelocity);
 values(s.gasMapOffsets);
 values(s.gasMapFaces);
 values(s.gasMapWeights);
 values(s.normalOffsets);
 values(s.normalCoordinates);
 values(s.normalLayerMass);
 values(s.normalLayerVolume);
 values(s.oldNormalLayerVolume);
 values(s.normalThickness);
 values(s.normalPressure);
 values(s.normalPressureRate);
 values(s.bottomTemperature);
 values(s.topTemperature);
 values(s.bottomHeatFlux);
 values(s.topHeatFlux);
 std::cout<<program.knots.size()<<' ';
 for(const auto& k:program.knots){emit(k.time);std::cout<<k.localTime.sum<<' '<<k.localTime.correction<<' ';values(k.gasPoints);values(k.solidPoints);std::cout<<k.faces.size()<<' ';
  for(const auto& f:k.faces){emit(f.temperature);emit(f.velocity);std::cout<<int(f.primaryKind)<<' ';for(auto x:f.speciesRate)emit(x);for(auto x:f.condensedRate)emit(x);for(auto x:f.poreRate)emit(x);for(auto x:f.poreSweepRate)emit(x);emit(f.radiationFlux);emit(f.solidNormalVelocity);emit(f.normalVelocity);}}
 std::cout<<program.donorPlan.interval.identity.sequence<<' '<<program.donorPlan.interval.identity.epoch<<' ';emit(program.donorPlan.interval.begin);emit(program.donorPlan.interval.end);assert(program.donorPlan.knots.empty());std::cout<<" donor 0\n";
}
int main(){auto p=physics();ModelConfig model;model.physics=p;auto base=host(p);base.gasMesh.points={{0,0,0},{1,2,3}};base.solidMesh.points={{4,5,6}};base.gasMesh.referencePoints=base.gasMesh.points;
 CouplingInterval interval;interval.identity.sequence=7;interval.end=.01;std::string error;CpuMaterialReport report;std::cout<<std::hexfloat;
 for(int mode=0;mode<3;++mode){ModelConfig m=model;HostState b=base;
  if(mode==1){m.physics.meshMotion.policy=MeshMotionPolicy::PrescribedSinusoidal;b.surface=SurfaceMesh{};b.filmAux.clear();m.physics.meshMotion.amplitude={.01,.02,.03};m.physics.meshMotion.angularFrequency=1;m.physics.meshMotion.spatialWaveNumber={1,2,3};}
  if(mode==2)interval.end=.005;
  CpuMaterialDriver driver(m,nullptr);WallProgram out;allocations=allocatedBytes=traceCalls=surfaceCalls=0;
  if(!driver.predictWall(b,interval,out,error,&report)){std::cerr<<"mode "<<mode<<" "<<error<<"\n";return 2;}auto a=allocations,z=allocatedBytes;programOut(out);std::cout<<"COUNT "<<a<<' '<<z<<' '<<traceCalls<<' '<<surfaceCalls<<'\n';
 }
#if TERMINAL_TEST
 interval.end=.01;CpuMaterialDriver driver(model,nullptr);HostState bad=base;bad.gas[0].energy=1e10;WallProgram out;
 assert(!driver.predictWall(bad,interval,out,error,&report));assert(report.failureOrigin==MaterialFailureOrigin::SurfaceNumerical);
 assert(detail::ConductionContinuation::predict(driver,bad,interval,out,error,&report));assert(out.knots.size()==2&&out.knots.back().time==.01&&out.knots.back().faces.size()==1);
 std::vector<GasPrimitive> actual;bool numerical=false;assert(baseTrace<true>(bad,p,actual,error,&numerical));
 std::cerr<<"EOS actual "<<actual[0].temperature<<" "<<actual[0].rho<<" "<<actual[0].Y[0]<<" "<<actual[0].pressure<<"\n";
 assert(actual[0].rho==1&&actual[0].Y[0]==1&&actual[0].pressure==287*actual[0].temperature);
 const Real cv=1000-287;assert(actual[0].soundSpeed==::sqrt((1+287/cv)*287*actual[0].temperature));
 bad=base;bad.surface.gasFace.clear();assert(!detail::ConductionContinuation::predict(driver,bad,interval,out,error,&report));assert(report.failureOrigin==MaterialFailureOrigin::None);
 bad=base;bad.gas.clear();assert(!detail::ConductionContinuation::predict(driver,bad,interval,out,error,&report));assert(report.failureOrigin==MaterialFailureOrigin::None);
 bad=base;bad.gas[0].energy=1e10;ModelConfig unavailable=model;unavailable.physics.species[0].model=static_cast<ugkwp::SpeciesThermoModel>(99);CpuMaterialDriver noFormula(unavailable,nullptr);
 assert(!detail::ConductionContinuation::predict(noFormula,bad,interval,out,error,&report));assert(report.failureOrigin==MaterialFailureOrigin::None);
 ModelConfig prescribed=model;prescribed.physics.meshMotion.policy=MeshMotionPolicy::PrescribedSinusoidal;prescribed.physics.meshMotion.amplitude={.01,0,0};prescribed.physics.meshMotion.angularFrequency=1;prescribed.physics.meshMotion.spatialWaveNumber={1,0,0};CpuMaterialDriver motion(prescribed,nullptr);
 assert(!motion.predictWall(base,interval,out,error,&report));assert(report.failureOrigin==MaterialFailureOrigin::GeometryNumerical);
 assert(detail::ConductionContinuation::predict(motion,base,interval,out,error,&report));assert(out.knots.back().gasPoints[1].x==1+.01*::sin(1.)*::sin(.01));
 bad=base;bad.gasMesh.referencePoints.clear();assert(!detail::ConductionContinuation::predict(motion,bad,interval,out,error,&report));assert(report.failureOrigin==MaterialFailureOrigin::None);
 std::cout<<"actual predictor EOS/surface/prescribed tails and missing storage passed\n";
#endif
}
