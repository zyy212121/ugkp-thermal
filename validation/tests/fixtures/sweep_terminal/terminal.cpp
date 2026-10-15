#include <new>
#include <cstdlib>
long failAfter=-1;bool forceCorrectionFault=false,forcePublicationFault=false;
void* operator new(std::size_t n){if(failAfter==0){failAfter=-1;throw std::bad_alloc();}if(failAfter>0)--failAfter;if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void* p) noexcept {std::free(p);}
void operator delete(void* p,std::size_t) noexcept {std::free(p);}
#include "GEOMETRY_OWNER"
#define main original_fixture_main
#include "GEOMETRY_FIXTURE"
#undef main
#include <cassert>
#include <iostream>
static void same(const std::vector<Vec3>& a,const std::vector<Vec3>& b){assert(a.size()==b.size());for(std::size_t i=0;i<a.size();++i)assert(a[i].x==b[i].x&&a[i].y==b[i].y&&a[i].z==b[i].z);}
int main(){
 auto mesh=tetrahedralBulk();std::vector<int> faces;for(int f=0;f<int(mesh.owner.size());++f)if(mesh.neighbour[f]<0)faces.push_back(f);
 std::vector<Vec3> prescribed(mesh.points.size()),directions(mesh.points.size());std::vector<unsigned char>fixed(mesh.points.size(),1);fixed[4]=0;
 for(int i=0;i<4;++i)directions[i]=normalized(mesh.points[i]-mesh.points[4]);prescribed[0]=directions[0]*.005;
 std::vector<Real> target(faces.size(),-.01);HostMesh out;std::vector<Real>actual;SweepConstraintReport report;std::string error;bool numerical=false;
 // Independent initial candidate and original local-to-world expression.
 auto old=mesh;const Vec3 origin=old.points[0];for(auto& p:old.points)p-=origin;assert(rebuildGeometry(old,error));std::vector<Vec3>start;assert(harmonicPointMotion(old,prescribed,fixed,start,error));
 auto world=mesh.points;for(std::size_t p=0;p<world.size();++p){const Vec3 d=start[p]-old.points[p];if(d.x!=0||d.y!=0||d.z!=0)world[p]+=d;}
 HostMesh reference;std::vector<Real>referenceSweep;assert(detail::completeStageGeometry(mesh,world,1,reference,referenceSweep,error));
 for(int policy=0;policy<6;++policy){SweepConstraintControls c;
  if(policy==0)c.maximumIterations=0;if(policy==1)c.maximumLineSearch=0;if(policy==2)c.maximumDegreesOfFreedom=1;if(policy==3)c.maximumConstraints=1;if(policy==4)c.maximumModeEntries=1;if(policy==5)c.maximumWorkspaceBytes=1;
  assert(!constrainFaceSweeps(mesh,faces,target,prescribed,fixed,directions,1,out,actual,report,error,c));
  assert(detail::completeFaceSweeps(mesh,faces,target,prescribed,fixed,directions,1,out,actual,report,error,c,Tolerances{},&numerical));same(out.points,world);assert(actual==referenceSweep);assert(report.maximumSweepResidual>0);
 }
 // All original line-search trials run; final publication uses retained points, not rejected trial.
 SweepConstraintControls c;c.maximumLineSearch=4;forceTrialReject=true;lastTrial.clear();lastRetained.clear();
 assert(detail::completeFaceSweeps(mesh,faces,target,prescribed,fixed,directions,1,out,actual,report,error,c));assert(report.lineSearchTrials==4);same(out.points,world);assert(!lastTrial.empty());bool different=false;for(std::size_t i=0;i<start.size();++i)different=different||mag(lastTrial[i]-start[i])>0;assert(different);same(lastRetained,start);forceTrialReject=false;
 c=SweepConstraintControls{};c.maximumIterations=1;lastAccepted.clear();
 assert(detail::completeFaceSweeps(mesh,faces,target,prescribed,fixed,directions,1,out,actual,report,error,c));assert(report.iterations==1&&!lastAccepted.empty());
 auto retainedWorld=mesh.points;different=false;for(std::size_t i=0;i<start.size();++i){const Vec3 d=lastAccepted[i]-old.points[i];if(d.x!=0||d.y!=0||d.z!=0)retainedWorld[i]+=d;different=different||mag(lastAccepted[i]-start[i])>0;}
 assert(different);same(out.points,retainedWorld);
 // Rank-incompatible constraints have a complete initial state but no valid correction proposal.
 std::vector<Vec3> zeroDirections(directions.size());assert(detail::completeFaceSweeps(mesh,faces,target,prescribed,fixed,zeroDirections,1,out,actual,report,error));same(out.points,world);
 auto missing=prescribed;missing.clear();assert(!detail::completeFaceSweeps(mesh,faces,target,missing,fixed,directions,1,out,actual,report,error,c,Tolerances{},&numerical)&&!numerical);
 // Actual allocator failure in initial geometry, correction workspace and final publication stays failure.
 for(int fault=0;fault<3;++fault){out=HostMesh{};out.geometryVersion=777;actual={123};c=SweepConstraintControls{};
  if(fault==0)failAfter=0;if(fault==1)forceCorrectionFault=true;if(fault==2){c.maximumIterations=0;forcePublicationFault=true;}
  numerical=true;assert(!detail::completeFaceSweeps(mesh,faces,target,prescribed,fixed,directions,1,out,actual,report,error,c,Tolerances{},&numerical));
  forceCorrectionFault=forcePublicationFault=false;failAfter=-1;assert(!numerical&&out.geometryVersion==777&&out.points.empty()&&actual==std::vector<Real>{123});
 }
 // Coupled actual gas/surface tail runs even when the target cannot be met in zero correction steps.
 auto base=coupledPair();auto film=base.filmAux;film[0].solidFront=-.03;film[1].solidFront=-.07;HostMesh gas,solid;SurfaceMesh surface;c=SweepConstraintControls{};c.maximumIterations=0;
 assert(detail::completeCoupledMeshes(base,film,{-.04,-.1},1,gas,solid,surface,report,error,c,Tolerances{},&numerical));
 assert(!gas.points.empty()&&!solid.points.empty()&&surface.sweptEdgeArea.size()==surface.edgeOwner.size());
 HostMesh measured;std::vector<Real> swept;assert(detail::completeStageGeometry(base.solidMesh,solid.points,1,measured,swept,error));
 base.solidSweepRemainder={10,-20};std::vector<Real> carry;assert(updateMaterialSweepRemainder<true>(base,{-.04,-.1},swept,solid,carry,error));
 for(std::size_t f=0;f<carry.size();++f){const Real expected=Real(static_cast<long double>(base.solidSweepRemainder[f])+static_cast<long double>(f==0?-.04:-.1)-swept[base.surface.solidFace[f]]);assert(carry[f]==expected);}
 assert(!validateMaterialSweepRemainder(base,error));assert(validateMaterialSweepRemainder<true>(base,error));
 base.solidSweepRemainder={10};numerical=true;
 assert(!detail::completeCoupledMeshes(base,film,{-.04,-.1},1,gas,solid,surface,report,error,c,Tolerances{},&numerical)&&!numerical);
 auto prefix=base;prefix.surface.solidFace={0};prefix.solidSweepRemainder.clear();
 assert(updateMaterialSweepRemainder<true>(prefix,{0},{.125},solid,carry,error));assert(carry[0]==-.125);
 auto reversedAreas=base.gasMesh;for(auto& area:reversedAreas.areaVectors)area=area*(-1);
 std::vector<int> mapped,vertices;std::vector<Real> thickness(base.surface.solidFace.size(),0);
 numerical=true;assert(!mapConformalInterfaceImpl<false>(reversedAreas,base.solidMesh,base.surface.solidFace,thickness,base.surface.gasFace,mapped,vertices,error,&numerical)&&numerical);
 assert(mapConformalInterfaceImpl<true>(reversedAreas,base.solidMesh,base.surface.solidFace,thickness,base.surface.gasFace,mapped,vertices,error,&numerical));assert(mapped==base.surface.gasFace);
 // Empty mate is actually produced when present; no guessed mate when exact pairing fails.
 auto empty=base.solidMesh;for(std::size_t f=0;f<empty.owner.size();++f)if(chmt::absValue(empty.areaVectors[f].y)>0)empty.boundaryKind[f]=BoundaryKind::Empty;
 empty_detail::Extrusion ex;assert(empty_detail::build<true>(empty,ex,error));assert(ex.mate.size()==empty.points.size());
 empty.points[0].x+=.0123;assert(!empty_detail::build<true>(empty,ex,error));
 std::vector<Vec3> d(empty.points.size()),dir(empty.points.size());std::vector<unsigned char>all(empty.points.size(),1);d[0].z=.01;
 assert(detail::completeFaceSweeps(empty,{}, {},d,all,dir,1,out,actual,report,error,c));assert(out.points[0].z==empty.points[0].z+.01);
 std::cout<<"retained world/stage; six policies; four rejected trials; real coupled gas/surface; signed carry; actual Empty pairing\n";
}
