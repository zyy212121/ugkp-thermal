"""Native common-owner mapping: active attachments move; lab-frame histories do not.
This tests the resident API, not a complete CHMT recession calculation.
"""
import argparse
import json
from pathlib import Path
import subprocess
from particle_state_native import source as state_source, ROOT

MAPPING=r'''
 using Point=ugkp::thermal::ParticleGeometryPoint;
 const int faceMap[6]={1,2,0,3,4,5},offsets[7]={0,4,8,12,16,20,24},owners[6]={};
 int oldIds[24],newIds[24];Point oldPoints[24],newPoints[24];
 for(int f=0;f<6;++f)for(int j=0;j<4;++j){
  const int k=4*f+j;oldIds[k]=k;newIds[4*faceMap[f]+j]=k;
  oldPoints[k]={j==1||j==2?1.:0.,j>=2?1.:0.,double(f)};
  newPoints[k]={-oldPoints[k].y+.3,oldPoints[k].x-.2,oldPoints[k].z-.1};
 }
 int *dOffsets=nullptr,*dOldIds=nullptr,*dNewIds=nullptr,*dOwners=nullptr,*dMap=nullptr;
 Point *dOldPoints=nullptr,*dNewPoints=nullptr;
#define UPLOAD(P,H,N) CHECK(allocate(P,N,"geometry fixture")==0);CHECK(copyToDevice(P,H,N,"geometry fixture upload")==0)
 UPLOAD(dOffsets,offsets,7);UPLOAD(dOldIds,oldIds,24);UPLOAD(dNewIds,newIds,24);
 UPLOAD(dOwners,owners,6);UPLOAD(dMap,faceMap,6);UPLOAD(dOldPoints,oldPoints,24);UPLOAD(dNewPoints,newPoints,24);
#undef UPLOAD
 ugkp::thermal::ParticleSurfaceGeometry previous{6,24,dOffsets,dOldIds,dOwners,dOldPoints};
 ugkp::thermal::ParticleSurfaceGeometry endpoint{6,24,dOffsets,dNewIds,dOwners,dNewPoints};
 cudaStream_t stream;CUDA_OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 CHECK(ugkp::thermal::mapAttachedParticles(s,previous,endpoint,dMap,stream)==0);
 CUDA_OK(cudaStreamSynchronize(stream));CUDA_OK(cudaStreamDestroy(stream));
 // The application also remaps its boundary model directory with the mesh.
 const unsigned char reorderedMask[6]={4,1,3,0,0,0};
 CHECK(copyToDevice(s->particleStuckCandidateMask,reorderedMask,6,"remapped wall model")==0);
 std::vector<ugkp::thermal::ParticleState<>> mapped;
 CHECK(ugkp::thermal::downloadParticleState(s,mapped)==0);CHECK(mapped.size()==first.size());
 int activeAttached=0,unchanged=0;
 for(size_t i=0;i<first.size();++i) {
  auto a=first[i];const auto& b=mapped[i];
  if(a.thermal.status!=0&&a.state!=0){
   const double x=a.position.x,y=a.position.y;
   a.position={-y+.3,x-.2,a.position.z-.1};a.face=faceMap[a.face];++activeAttached;
  }else ++unchanged;
  CHECK(std::abs(a.position.x-b.position.x)<1e-14&&std::abs(a.position.y-b.position.y)<1e-14&&std::abs(a.position.z-b.position.z)<1e-14);
  a.position=b.position;
#define UGKP_PARTICLE_FIELD(Field,Value) CHECK(a.Value==b.Value);
#define UGKP_PARTICLE_OPTIONAL(Field,Value) CHECK(a.Value==b.Value);
#define UGKP_PARTICLE_ARRAY(Field,Value,N) for(int j=0;j<N;++j)CHECK(a.Value[j]==b.Value[j]);
#include "ParticleStateFields.inc"
#undef UGKP_PARTICLE_FIELD
#undef UGKP_PARTICLE_OPTIONAL
#undef UGKP_PARTICLE_ARRAY
  // Inactive slots retain stale face IDs and are excluded from inventories.
  // Their complete stored history, checked above, must still be unchanged.
  if(a.thermal.status!=0)CHECK(a.energy==b.energy);
 }
 CHECK(activeAttached>0&&unchanged>0);
 CUDA_OK(cudaFree(dOffsets));CUDA_OK(cudaFree(dOldIds));CUDA_OK(cudaFree(dNewIds));CUDA_OK(cudaFree(dOwners));
 CUDA_OK(cudaFree(dMap));CUDA_OK(cudaFree(dOldPoints));CUDA_OK(cudaFree(dNewPoints));

 // Exercise the public resident tracking refresh on a two-cell view, including
 // the neighbour side of an internal face and a periodic boundary neighbour.
 DeviceState tracking=*s;tracking.nCells=2;tracking.nFaces=4;tracking.nInternalFaces=1;tracking.nCellPlanes=5;
 tracking.deviceState=nullptr;
 const int starts[2]={0,2},counts[2]={2,3},ids[5]={0,1,0,2,3},owner[4]={0,0,1,1},neighbour[4]={1,-1,0,-1};
 const int kinds[4]={0,2,6,5};
 const double sx[4]={1.,-2.,0.,0.},sy[4]={0.,0.,3.,0.},sz[4]={0.,0.,0.,-4.},area[4]={1.,2.,3.,4.};
 const double cx[4]={.5,-.5,1.,2.},cy[4]={1.,2.,3.,4.},cz[4]={-1.,-2.,-3.,-4.};
 const double rest[4]={.11,.8,.9,.7},tangent[4]={.22,.6,.5,.4};
#define FIELD(Name,H,N) tracking.Name=nullptr;CHECK(allocate(tracking.Name,N,"tracking fixture")==0);CHECK(copyToDevice(tracking.Name,H,N,"tracking fixture upload")==0)
 FIELD(cellPlaneStart,starts,2);FIELD(cellPlaneCount,counts,2);FIELD(cellFaceId,ids,5);
 FIELD(faceOwner,owner,4);FIELD(faceNeighbour,neighbour,4);
 FIELD(Sfx,sx,4);FIELD(Sfy,sy,4);FIELD(Sfz,sz,4);FIELD(magSf,area,4);
 FIELD(faceCx,cx,4);FIELD(faceCy,cy,4);FIELD(faceCz,cz,4);
#undef FIELD
#define OUTPUT(Name) tracking.Name=nullptr;CHECK(allocate(tracking.Name,5,"tracking output")==0)
 OUTPUT(planeNx);OUTPUT(planeNy);OUTPUT(planeNz);OUTPUT(planeD);
 OUTPUT(cellFaceNeighbor);OUTPUT(cellFaceKind);OUTPUT(cellFaceRestitution);OUTPUT(cellFaceTangential);
#undef OUTPUT
 int* dKind=nullptr;double *dRest=nullptr,*dTangent=nullptr;
 CHECK(allocate(dKind,4,"tracking kind")==0);CHECK(copyToDevice(dKind,kinds,4,"tracking kind")==0);
 CHECK(allocate(dRest,4,"tracking rest")==0);CHECK(copyToDevice(dRest,rest,4,"tracking rest")==0);
 CHECK(allocate(dTangent,4,"tracking tangent")==0);CHECK(copyToDevice(dTangent,tangent,4,"tracking tangent")==0);
 CHECK(allocate(tracking.deviceState,1,"tracking view")==0);CHECK(syncDeviceState(&tracking,"tracking view init")==0);
 CUDA_OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 CHECK(ugkp::thermal::refreshParticleTrackingGeometry(&tracking,{4,dKind,dRest,dTangent},stream)==0);
 CUDA_OK(cudaStreamSynchronize(stream));CUDA_OK(cudaStreamDestroy(stream));
 double nx[5],ny[5],nz[5],distance[5],reflection[5],friction[5];int neighbors[5],types[5];
#define READ(H,Name) CHECK(copyToHost(H,tracking.Name,5,"tracking result")==0)
 READ(nx,planeNx);READ(ny,planeNy);READ(nz,planeNz);READ(distance,planeD);
 READ(reflection,cellFaceRestitution);READ(friction,cellFaceTangential);READ(neighbors,cellFaceNeighbor);READ(types,cellFaceKind);
#undef READ
 const int expectedNeighbor[5]={1,-1,0,0,-1},expectedType[5]={0,2,0,6,5};
 for(int j=0;j<5;++j){const int f=ids[j],cell=j<2?0:1;const double sign=owner[f]==cell?1.:-1.;
  CHECK(nx[j]==sign*sx[f]/(area[f]+OfSmall));CHECK(ny[j]==sign*sy[f]/(area[f]+OfSmall));CHECK(nz[j]==sign*sz[f]/(area[f]+OfSmall));
  CHECK(distance[j]==nx[j]*cx[f]+ny[j]*cy[f]+nz[j]*cz[f]);
  CHECK(neighbors[j]==expectedNeighbor[j]&&types[j]==expectedType[j]);
  CHECK(reflection[j]==(f==0?1.:rest[f])&&friction[j]==(f==0?1.:tangent[f]));
 }
#define FREE(Name) CUDA_OK(cudaFree(tracking.Name))
 FREE(deviceState);FREE(cellPlaneStart);FREE(cellPlaneCount);FREE(cellFaceId);FREE(faceOwner);FREE(faceNeighbour);
 FREE(Sfx);FREE(Sfy);FREE(Sfz);FREE(magSf);FREE(faceCx);FREE(faceCy);FREE(faceCz);
 FREE(planeNx);FREE(planeNy);FREE(planeNz);FREE(planeD);FREE(cellFaceNeighbor);FREE(cellFaceKind);FREE(cellFaceRestitution);FREE(cellFaceTangential);
#undef FREE
 CUDA_OK(cudaFree(dKind));CUDA_OK(cudaFree(dRest));CUDA_OK(cudaFree(dTangent));
 std::printf("{\"attached_mapped\":%d,\"free_or_inactive_unchanged\":%d,\"lab_velocity_and_history_exact\":true,\"tracking_planes_and_boundary_routes_exact\":true}\n",activeAttached,unchanged);
'''

def source():
    text=state_source()
    text=text.replace("h.status=(i%7==6)?0:1;","h.status=(i%7==6)?0:((i%5==4)?2:1);")
    text=text.replace('q.position={.1,.2,.3};','q.position={.1,.2,q.state?double(q.face)-1e-9:.3};')
    point=text.index(' std::printf("{\\\"particles')
    return text[:point]+MAPPING+'\n releaseState(s);\n}\n'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cu=out/'particle-geometry.cu';cu.write_text(source());exe=out/'particle-geometry'
    command=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','--fmad=false','--expt-relaxed-constexpr','--Ofast-compile=min',
        '-arch=sm_89','-I'+str(ROOT/'common/thermal'),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),str(cu),str(ROOT/'common/thermal/GpuWallEnergy64.cu'),'-o',str(exe)]
    with (out/'build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    results=[]
    for count in (12,65537):
        run=subprocess.run([str(exe),str(count)],capture_output=True,text=True)
        (out/f'run-{count}.log').write_text(run.stdout+run.stderr);run.check_returncode()
        results.append(json.loads(run.stdout))
    with (out/'memcheck.log').open('w') as log:
        subprocess.run(['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','88',str(exe),'12'],stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results))
if __name__=='__main__':main()
