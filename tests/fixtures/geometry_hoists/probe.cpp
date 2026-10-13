#define main original_sweep_main
#include "GEOMETRY_FIXTURE"
#undef main
#include <iomanip>
extern unsigned long long edgeScaleCalls,materialScaleCalls,allScaleCalls,pointVisits;
template<class T> void dump(T x){std::cout<<x<<'\n';}
void dump(Vec3 x){dump(x.x);dump(x.y);dump(x.z);}
void dump(BoundaryKind x){dump(static_cast<int>(x));}
void dump(ThermalBoundaryKind x){dump(static_cast<int>(x));}
void dump(GasPrimitive x){dump(x.rho);dump(x.velocity);dump(x.temperature);dump(x.pressure);dump(x.soundSpeed);for(auto y:x.Y)dump(y);}
void dump(SstPrimitive x){dump(x.k);dump(x.omega);dump(x.eddyViscosity);dump(x.blend1);dump(x.blend2);}
template<class T> void dump(const std::vector<T>& x){dump(x.size());for(const auto& y:x)dump(y);}
void dump(const HostMesh& x){
 dump(x.referencePoints);dump(x.points);dump(x.oldPoints);dump(x.cellCentres);dump(x.faceCentres);dump(x.areaVectors);dump(x.meshVelocity);
 dump(x.volumes);dump(x.oldVolumes);dump(x.wallDistance);dump(x.faceOffsets);dump(x.facePoints);dump(x.owner);dump(x.neighbour);dump(x.periodicPartner);dump(x.cellFaceOffsets);dump(x.cellFaces);dump(x.cellFaceSigns);
 dump(x.boundaryKind);dump(x.boundaryPrimitive);dump(x.thermalBoundary);dump(x.boundarySst);dump(x.faceIds);dump(x.geometryVersion);dump(x.topologyHash);
}
void dump(const SurfaceMesh& x){
 dump(x.prescribedTopTraction);dump(x.prescribedPressureGradient);dump(x.edgeOwnerOffset);dump(x.edgeNeighbourOffset);dump(x.gasDistance);dump(x.solidDistance);
 dump(x.gasFace);dump(x.solidFace);dump(x.solidCell);dump(x.edgeOwner);dump(x.edgeNeighbour);dump(x.persistentId);dump(x.edgeLength);dump(x.area);dump(x.oldArea);dump(x.sweptEdgeArea);
 dump(x.edgeConormal);dump(x.centre);dump(x.normal);dump(x.meshVelocity);dump(x.baseVelocity);dump(x.gasMapOffsets);dump(x.gasMapFaces);dump(x.gasMapWeights);
 dump(x.normalOffsets);dump(x.normalCoordinates);dump(x.normalLayerMass);dump(x.normalLayerVolume);dump(x.oldNormalLayerVolume);dump(x.normalThickness);dump(x.normalPressure);dump(x.normalPressureRate);dump(x.bottomTemperature);dump(x.topTemperature);dump(x.bottomHeatFlux);dump(x.topHeatFlux);
}
void dump(const SweepConstraintReport& x){dump(static_cast<int>(x.status));dump(x.iterations);dump(x.rank);dump(x.constraints);dump(x.degreesOfFreedom);dump(x.lineSearchTrials);dump(x.maximumSweepResidual);dump(x.maximumPlanarityResidual);dump(x.streamedModes);dump(x.peakWorkspaceBytes);dump(x.modeSolves);}
void reset(){edgeScaleCalls=materialScaleCalls=allScaleCalls=pointVisits=0;}
void counts(int id,int edges,int targets){std::cerr<<id<<' '<<edges<<' '<<targets<<' '<<edgeScaleCalls<<' '<<materialScaleCalls<<' '<<allScaleCalls<<' '<<pointVisits<<'\n';}
int main(){std::cout<<std::hexfloat;int id=0;
 // Same successful dense small-cell fixtures, without unrelated test execution.
 for(Real scale:{1e-4,1e-3,1.})for(Real offset:{0.,.25}){
  auto base=coupledPair();for(auto* m:{&base.gasMesh,&base.solidMesh}){for(auto& p:m->points)p=(p+Vec3{offset,offset,offset})*scale;std::string e;require(rebuildGeometry(*m,e),e);}
  base.solid.resize(2);for(int c=0;c<2;++c)base.solid[c].condensed[0]=1000*base.solidMesh.volumes[c];
  auto candidate=base.filmAux;candidate[0].solidFront=-.03e-8*scale;candidate[1].solidFront=-.12e-8*scale;
  HostMesh gas,solid;SurfaceMesh surface;SweepConstraintReport report;std::string error;reset();
  require(moveCoupledMeshesConstrained(base,candidate,{-.04e-8*scale*scale*scale,-.1e-8*scale*scale*scale},1,gas,solid,surface,report,error),error);
  counts(id++,surface.edgeOwner.size(),2);dump(gas);dump(solid);dump(surface);dump(report);dump(error);
 }
 for(int mode:{0,1,2}){
  auto base=coupledPair();std::string error;
  if(mode==1)for(auto* m:{&base.gasMesh,&base.solidMesh}){for(std::size_t f=0;f<m->owner.size();++f)if(m->neighbour[f]<0&&absValue(m->areaVectors[f].y)>0)m->boundaryKind[f]=BoundaryKind::Empty;require(rebuildGeometry(*m,error),error);}
  if(mode==2){base.surface=SurfaceMesh{};base.filmAux.clear();base.solid.resize(2);}
  HostMesh gas,solid;SurfaceMesh surface;SweepConstraintReport report;reset();
  std::vector<Real> targets=mode==2?std::vector<Real>{}:std::vector<Real>{-.03,-.08};
  require(moveCoupledMeshesConstrained(base,base.filmAux,targets,1,gas,solid,surface,report,error),error);
  counts(id++,surface.edgeOwner.size(),0);dump(gas);dump(solid);dump(surface);dump(report);dump(error);
 }
 // Periodic surface incidence: original global topology and metric outputs.
 {
  auto periodic=coupledPair();std::string error;
  for(auto* m:{&periodic.gasMesh,&periodic.solidMesh}){
   int left=-1,right=-1;for(int f=0;f<static_cast<int>(m->owner.size());++f){if(m->faceCentres[f].x==0)left=f;if(m->faceCentres[f].x==2)right=f;}
   require(left>=0&&right>=0,"periodic opposing faces");m->boundaryKind[left]=m->boundaryKind[right]=BoundaryKind::Periodic;
   m->periodicPartner[left]=right;m->periodicPartner[right]=left;require(rebuildGeometry(*m,error),error);
  }
  SurfaceMesh surface;reset();require(rebuildTrajectorySurface(periodic,periodic.gasMesh,periodic.solidMesh,1,surface,error),error);
  counts(id++,surface.edgeOwner.size(),0);dump(surface);dump(error);
 }
 // Direct surface rebuild: strong, nonsingular affine compression requires
 // more than the first quadrature refinement; no gas mapping is needed.
 auto base=coupledPair();auto endpoint=base.solidMesh;
 for(auto& p:endpoint.points)p={.02*p.x,p.y,p.z+.1*p.x};
 base.surface.gasFace={-1,-1};SurfaceMesh surface;std::string error;reset();
 require(rebuildTrajectorySurface(HostMesh{},base.solidMesh,base.surface,HostMesh{},endpoint,1,surface,error),error);
 counts(id++,surface.edgeOwner.size(),0);dump(surface);dump(error);
 SurfaceMesh empty;reset();require(rebuildTrajectorySurface(HostState{},HostMesh{},HostMesh{},1,empty,error),error);counts(id++,0,0);dump(empty);dump(error);
 HostMesh gas,solid;SweepConstraintReport report;reset();require(moveCoupledMeshesConstrained(HostState{},{},{},1,gas,solid,empty,report,error),error);counts(id++,0,0);dump(gas);dump(solid);dump(empty);dump(report);dump(error);
 return 0;}
