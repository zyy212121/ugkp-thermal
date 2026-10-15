#include "GEOMETRY_OWNER"
#define main original_fixture_main
#include "GEOMETRY_FIXTURE"
#undef main
#include <cassert>
#include <iomanip>
static void equal(Vec3 a,Vec3 b){assert(a.x==b.x&&a.y==b.y&&a.z==b.z);}
int main(){
 std::string error;bool numerical=false;auto base=coupledPair();HostMesh out;std::vector<Real>sweeps;Tolerances tol;tol.absoluteGeometry=-1;
 assert(!makeStageGeometry(base.solidMesh,base.solidMesh.points,1,out,sweeps,error,tol));
 assert(detail::completeStageGeometry(base.solidMesh,base.solidMesh.points,1,out,sweeps,error,tol,&numerical));
 assert(out.points.size()==base.solidMesh.points.size());assert(out.volumes==base.solidMesh.volumes);
 assert(out.oldVolumes==base.solidMesh.volumes);assert(out.geometryVersion==base.solidMesh.geometryVersion+1);
 std::vector<Vec3> area;assert(detail::completeStageAreaVectors(base.solidMesh,base.solidMesh.points,area,error));
 for(std::size_t f=0;f<area.size();++f){equal(area[f],base.solidMesh.areaVectors[f]);assert(sweeps[f]==0);equal(out.meshVelocity[f],Vec3{});}
 auto reversed=base.solidMesh.points;for(auto& p:reversed)p.x=-p.x;
 assert(!makeStageGeometry(base.solidMesh,reversed,1,out,sweeps,error));
 assert(detail::completeStageGeometry(base.solidMesh,reversed,1,out,sweeps,error,Tolerances{},&numerical));
 for(std::size_t c=0;c<out.volumes.size();++c){assert(out.volumes[c]==-base.solidMesh.volumes[c]);equal(out.cellCentres[c],Vec3{-base.solidMesh.cellCentres[c].x,base.solidMesh.cellCentres[c].y,base.solidMesh.cellCentres[c].z});}
 for(std::size_t f=0;f<sweeps.size();++f)assert(absValue(sweeps[f]-quadratureSweep(base.solidMesh,reversed,f))<1e-14);
 assert(detail::completeStageGeometry(base.solidMesh,reversed,0,out,sweeps,error));assert(!chmt::finite(out.meshVelocity[0]));
 auto duplicate=base.solidMesh;duplicate.faceIds.assign(duplicate.faceIds.size(),7);duplicate.neighbour.push_back(-1);
 assert(detail::completeGeometry(duplicate,error,&numerical));assert(duplicate.faceIds[0]==7);
 auto missing=base.solidMesh;missing.neighbour.clear();assert(!detail::completeGeometry(missing,error,&numerical)&&!numerical);
 auto shortPoints=base.solidMesh.points;shortPoints.pop_back();assert(!detail::completeStageGeometry(base.solidMesh,shortPoints,1,out,sweeps,error,Tolerances{},&numerical)&&!numerical);
 auto surplus=base.solidMesh.points;surplus.push_back(Vec3{3,4,5});
 assert(detail::completeStageGeometry(base.solidMesh,surplus,1,out,sweeps,error));assert(out.points.size()==surplus.size());
 // Actual slow two-free-node Jacobi graph: no forced iteration limit or new threshold.
 HostMesh mesh;mesh.points={{-1,0,0},{0,0,0},{.0001,0,0},{1,0,0}};mesh.owner={0};mesh.boundaryKind={BoundaryKind::Slip};mesh.faceOffsets={0,4};mesh.facePoints={0,1,2,3};
 std::vector<unsigned char> fixed={1,0,0,1};std::vector<Vec3>d(4);d[0]={1,0,0};std::vector<Vec3>motion;
 harmonicIterations=0;assert(!harmonicPointMotion(mesh,d,fixed,motion,error));assert(harmonicIterations==20000);
 harmonicIterations=0;assert(detail::completeHarmonicPointMotion(mesh,d,fixed,motion,error,&numerical));assert(harmonicIterations==20000);
 std::vector<Vec3> current=d,next(4);std::vector<std::set<int>>adj(4);for(int i=0;i<4;++i){adj[i].insert((i+1)%4);adj[(i+1)%4].insert(i);}
 for(int iteration=0;iteration<20000;++iteration){for(int i=0;i<4;++i){if(fixed[i]){next[i]=current[i];continue;}Vec3 sum{};Real total=0;for(int other:adj[i]){Real weight=1/mag(mesh.points[i]-mesh.points[other]);sum+=current[other]*weight;total+=weight;}next[i]=sum/total;}current.swap(next);}
 assert(current[1].x!=next[1].x || current[2].x!=next[2].x);for(int i=0;i<4;++i)equal(motion[i],mesh.points[i]+current[i]);
 std::size_t peak=0;sweep_detail::Budget budget(1<<20,peak);sweep_detail::HarmonicModes modes(mesh,fixed,budget,-1);
 sweep_detail::Points boundary(4,Vec3{},sweep_detail::Allocator<Vec3>(budget)),result{ sweep_detail::Allocator<Vec3>(budget)};
 boundary[0]=d[0];assert(!modes.solve(boundary,result));assert(modes.solve<true>(boundary,result));
 for(int i=0;i<4;++i){Vec3 expected=mesh.points[i]+current[i];expected-=mesh.points[i];if(fixed[i])expected=d[i];equal(result[i],expected);}
 // Degenerate numeric edge still evaluates existing arithmetic.
 mesh.points[2]=mesh.points[1];assert(detail::completeHarmonicPointMotion(mesh,d,fixed,motion,error));assert(!chmt::finite(motion[1]));
 // Test-only bypass of the preceding duplicate-face proof isolates actual edge-origin typing.
 auto repeated=base.surface;repeated.solidFace={base.surface.solidFace[0],base.surface.solidFace[0]};repeated.gasFace={-1,-1};SurfaceMesh repeatedOutput;
 numerical=true;assert(!rebuildTrajectorySurface(HostMesh{},base.solidMesh,repeated,HostMesh{},base.solidMesh,1,repeatedOutput,error,Tolerances{},&numerical));assert(numerical);
 assert(detail::completeTrajectorySurface(HostMesh{},base.solidMesh,repeated,HostMesh{},base.solidMesh,1,repeatedOutput,error,Tolerances{},&numerical));
 // Whole terminal edge loop: actual 1024 integral, all later edges populated.
 auto endpoint=base.solidMesh;for(auto& p:endpoint.points)p={.02*p.x,p.y,p.z+.1*p.x};base.surface.gasFace={-1,-1};SurfaceMesh surface;
 forceQuad=true;quadCalls=0;assert(!rebuildTrajectorySurface(HostMesh{},base.solidMesh,base.surface,HostMesh{},endpoint,1,surface,error));
 quadCalls=0;assert(detail::completeTrajectorySurface(HostMesh{},base.solidMesh,base.surface,HostMesh{},endpoint,1,surface,error,Tolerances{},&numerical));
 std::vector<SurfaceEdge>edges;assert(surfaceEdges(base.solidMesh,base.surface,edges,error));assert(surface.sweptEdgeArea.size()==edges.size());assert(quadCalls==8*edges.size());
 for(std::size_t i=0;i<edges.size();++i){const Real actual=edgeSweepQuadrature<true>(base.solidMesh,endpoint.points,base.surface,edges[i],1024,false);assert(actual==surface.sweptEdgeArea[i]);}
 auto flat=base.solidMesh.points;flat[edges[0].b]=flat[edges[0].a];
 const Real value=edgeSweepIntegrand<true>(base.solidMesh,flat,base.surface,edges[0],1,false);assert(value==0); // real dot with zero tangent, not undefinedValue sentinel
 assert(!chmt::finite(edgeSweepIntegrand(base.solidMesh,flat,base.surface,edges[0],1,false)));
 // Axis-only continuation does not create a mate map.
 auto empty=base.solidMesh;for(std::size_t f=0;f<empty.owner.size();++f)if(absValue(empty.areaVectors[f].y)>0)empty.boundaryKind[f]=BoundaryKind::Empty;
 empty_detail::Extrusion extrusion;assert(empty_detail::axisGeometry(empty,extrusion,error,&numerical));assert(extrusion.axis==1&&extrusion.mate.empty());
 for(std::size_t f=0;f<empty.owner.size();++f)if(empty.boundaryKind[f]==BoundaryKind::Empty){empty.areaVectors[f]=Vec3{};break;}
 assert(empty_detail::axisGeometry(empty,extrusion,error,&numerical));assert(extrusion.axis==1&&extrusion.mate.empty());
 for(auto& area:empty.areaVectors)area=Vec3{};
 assert(empty_detail::axisGeometry(empty,extrusion,error,&numerical));assert(extrusion.axis==-1&&extrusion.mate.empty());
 auto collapsed=empty;for(auto& point:collapsed.points)point=Vec3{};
 assert(detail::completeGeometry(collapsed,error,&numerical));assert(collapsed.volumes[0]==0&&!chmt::finite(collapsed.cellCentres[0]));
 // Supplied periodic topology exists, but the requested surface has no searched counterpart.
 auto periodic=base.solidMesh;int left=-1,right=-1;
 for(int f=0;f<int(periodic.owner.size());++f){if(periodic.faceCentres[f].x==0)left=f;if(periodic.faceCentres[f].x==2)right=f;}
 assert(left>=0&&right>=0);periodic.boundaryKind[left]=periodic.boundaryKind[right]=BoundaryKind::Periodic;
 periodic.periodicPartner[left]=right;periodic.periodicPartner[right]=left;
 SurfaceMesh partial;partial.solidFace={base.surface.solidFace[0]};partial.gasFace={-1};
 numerical=true;assert(!detail::completeTrajectorySurface(HostMesh{},periodic,partial,HostMesh{},periodic,1,surface,error,Tolerances{},&numerical)&&!numerical);
 assert(error.find("unpaired periodic surface edge")!=std::string::npos);
 std::cout<<"actual stage arrays; Jacobi 20000 swapped state; mode output; terminal edges "<<edges.size()<<" at1024; missing input propagated\n";
}
