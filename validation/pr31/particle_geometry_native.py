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
  if(a.thermal.status==1&&a.state!=0){
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
  if(a.thermal.status==1)CHECK(a.energy==b.energy);
 }
 CHECK(activeAttached>0&&unchanged>0);
 CUDA_OK(cudaFree(dOffsets));CUDA_OK(cudaFree(dOldIds));CUDA_OK(cudaFree(dNewIds));CUDA_OK(cudaFree(dOwners));
 CUDA_OK(cudaFree(dMap));CUDA_OK(cudaFree(dOldPoints));CUDA_OK(cudaFree(dNewPoints));
 std::printf("{\"attached_mapped\":%d,\"free_or_inactive_unchanged\":%d,\"lab_velocity_and_history_exact\":true}\n",activeAttached,unchanged);
'''

def source():
    text=state_source()
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
