from pathlib import Path
import re,subprocess,sys,json
ROOT=Path(sys.argv[1]);OUT=Path(sys.argv[2]);OUT.mkdir(parents=True,exist_ok=True)
def func(text,name):
 m=re.search(r'__device__\s+\w+\s+'+name+r'\s*\(',text);assert m,name
 i=text.index('{',m.start());j=i+1;depth=1
 while depth:depth+=(text[j]=='{')-(text[j]=='}');j+=1
 return text[m.start():j]
prefix=r"""
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#define __forceinline__ inline
#define __device__
#include "PRECISION"
#define GPU_OPERATOR_REAL GpuReal
#define GPU_OPERATOR_TIME GpuTime
#define GPU_OPERATOR_R(x) GPU_R(x)
#define GPU_OPERATOR_TINY(x) GPU_TINY(x)
#define GPU_OPERATOR_THERMAL 0
#define GPU_PERIODIC_FACE_KIND 6
#define GPU_PARTICLE_EXIT_MASS_FALLBACK s.injectionParcelMass
#define GPU_PERIODIC_COORDINATE(hit,shift,eps,normal) ((hit)-(shift)+(eps)*(normal))
#define GPU_WALL_COORDINATE(hit,eps,normal) ((hit)-(eps)*(normal))
template<class T>T clampMin(T a,T b){return std::max(a,b);}
template<class T>T clampRange(T a,T b,T c){return std::min(std::max(a,b),c);}
template<class T>T finiteOr(T a,T b){return std::isfinite(a)?a:b;}
struct DeviceState {
 int particleCapacity=1,nCells=1,maxFaceWalkHops=512;
 int pStatus[1]={1},pCellId[1]={0},cellPlaneStart[1]={0},cellPlaneCount[1]={2};
 int cellFaceKind[4]={1,1,1,1},cellFaceNeighbor[4]={-1,-1,-1,-1},cellFaceId[4]={0,1,2,3};
 GpuReal px[1]={},py[1]={},pz[1]={},pux[1]={},puy[1]={},puz[1]={},puxOld[1]={},puyOld[1]={},puzOld[1]={},pm[1]={GPU_R(2)};
 GpuReal planeNx[4]={},planeNy[4]={},planeNz[4]={},planeD[4]={},cellLength[1]={GPU_R(.001)};
 GpuReal cellFaceRestitution[4]={GPU_R(1),GPU_R(1),GPU_R(1),GPU_R(1)},faceCx[4]={},faceCy[4]={},faceCz[4]={};
 GpuReal facePeriodicDx[4]={},facePeriodicDy[4]={},facePeriodicDz[4]={},injectionParcelMass=GPU_R(2);
};
bool isPeriodicFace(const DeviceState& s,int f){return s.cellFaceKind[f]==6;}
"""
main=r"""
int main(int argc,char**argv){
 const char* mode=argc>1?argv[1]:"axis";DeviceState s;
 if (!std::strcmp(mode,"axis")){
  const GpuReal a=GPU_R(2.5*3.14159265358979323846/180.0);
  s.cellFaceRestitution[0]=s.cellFaceRestitution[1]=wedgeRestitution();
  s.planeNy[0]=s.planeNy[1]=-sin(a);s.planeNz[0]=cos(a);s.planeNz[1]=-cos(a);
  s.py[0]=GPU_R(2e-6);s.puy[0]=s.puyOld[0]=GPU_R(-100);
  trackOneParticleLocalFaceWalk(s,0,4e-8);
  const double speed2=double(s.puy[0])*s.puy[0]+double(s.puz[0])*s.puz[0];
  printf("axis live=%d y=%.17g z=%.17g vy=%.17g vz=%.17g speed2=%.17g mass=%.17g\n",s.pStatus[0],double(s.py[0]),double(s.pz[0]),double(s.puy[0]),double(s.puz[0]),speed2,double(s.pm[0]));
  if(s.pStatus[0]!=1){fprintf(stderr,"FAIL: axis-crossing particle was deleted despite512 hops\n");return 1;}
  const double posTol=sizeof(GpuReal)==4?2e-10:2e-10;
  if(fabs(double(s.py[0])-2e-6)>posTol||fabs(double(s.pz[0]))>posTol){fprintf(stderr,"FAIL: specular wedge axis-crossing final position\n");return 2;}
  if(fabs(speed2-10000)>(sizeof(GpuReal)==4?.1:1e-7)||s.pm[0]!=GPU_R(2)){fprintf(stderr,"FAIL: elastic reflection mass/energy\n");return 3;}
 }else if (!std::strcmp(mode,"planar")){
  s.cellPlaneCount[0]=1;s.planeNy[0]=GPU_R(-1);s.cellFaceRestitution[0]=GPU_R(.5);
  s.py[0]=GPU_R(.25);s.puy[0]=s.puyOld[0]=GPU_R(-1);trackOneParticleLocalFaceWalk(s,0,.5);
  if(s.pStatus[0]!=1||fabs(double(s.py[0])-.125)>1e-7||fabs(double(s.puy[0])-.5)>1e-7||s.pm[0]!=GPU_R(2))return 4;
 }else if (!std::strcmp(mode,"outflow")){
  s.cellPlaneCount[0]=1;s.planeNx[0]=GPU_R(1);s.planeD[0]=GPU_R(1);s.faceCx[0]=GPU_R(1);s.cellFaceKind[0]=3;
  s.px[0]=GPU_R(.9);s.pux[0]=s.puxOld[0]=GPU_R(1);trackOneParticleLocalFaceWalk(s,0,.2);if(s.pStatus[0]!=0)return 5;
 }else return 6;
 printf("PASS %s\n",mode);return 0;
}
"""
results=[]
for app,bits in [('gas',64),('FSH',64),('CHT',64),('CHT',32)]:
 if not (ROOT/f'applications/{"gasUGKP" if app=="gas" else app}').exists():continue
 pfx=prefix.replace('PRECISION',str(ROOT/'common/GpuPrecisionTypes.H'))
 header=(ROOT/f'applications/{"gasUGKP" if app=="gas" else app}/gpu/GpuResidentStrict.H').read_text()
 pos=header.index('isA<wedgePolyPatch>(pp)');left=header.index('{',pos);right=header.index('}',left)
 boundary_body=header[left+1:right]
 if (ROOT/'common/GpuParticleBoundaryModel.H').exists():pfx+='\n#include "'+str(ROOT/'common/GpuParticleBoundaryModel.H')+'"\nusing namespace Foam;\n'
 pfx+='\nGpuReal wedgeRestitution(){ int kind=-1; GpuReal restitution=GPU_R(-1); '+boundary_body+' return restitution; }\n'
 pfx+='\n#define GPU_GEOMETRY_ROUNDING_AWARE '+('1' if app=='CHT' and bits==32 else '0')+'\n'
 geometry='#include "'+str(ROOT/'common/operators/pointInsideCell.cuh')+'"\n'
 if app=='CHT':pfx+='\n#undef GPU_WALL_COORDINATE\n#define GPU_WALL_COORDINATE(hit,eps,normal) insideFaceCoordinate(hit,normal,eps)\n'
 cpp=OUT/f'{app}{bits}.cpp';exe=OUT/f'{app}{bits}';cpp.write_text(pfx+'\n'+geometry+'\n'+(ROOT/'common/GpuParticleTransport.cuh').read_text()+'\n'+main)
 q=subprocess.run(['g++','-std=c++17','-O2','-ffp-contract=off','-DUGKWP_GPU_REAL_BITS='+str(bits),'-I'+str(ROOT/'common'),str(cpp),'-o',str(exe)],capture_output=True,text=True);assert q.returncode==0,q.stderr
 for mode in ['axis','planar','outflow']:
  q=subprocess.run([str(exe),mode],capture_output=True,text=True);r={'app':app,'bits':bits,'mode':mode,'returncode':q.returncode,'stdout':q.stdout,'stderr':q.stderr};results.append(r);print(r,flush=True)
(OUT/'results.json').write_text(json.dumps(results,indent=2));sys.exit(int(any(x['returncode'] for x in results)))
