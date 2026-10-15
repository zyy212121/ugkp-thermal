"""Real injection, gather, copy, pointer commit and carry contribution. Host only."""
from pathlib import Path
import json
import re
import subprocess
import pytest
ROOT=Path(__file__).resolve().parents[1]
C=ROOT/'common'
def text(p):return (C/p).read_text().replace('asm("trap;");','throw std::runtime_error("actual address or arithmetic");')
def function(src,name):
 a=src.index(name);a=src.rfind('\n',0,a)+1;e=src.index('{',a)+1;d=1
 while d:d+=(src[e]=='{')-(src[e]=='}');e+=1
 return src[a:e]

@pytest.mark.parametrize('app',['CHT', 'FSH'])
@pytest.mark.parametrize('route',['selected','payload','survivor'])
def test_complete_inject_copy_swap_energy_rng_lineage(tmp_path,app,route):
 fields=[f for f in json.loads((C/'ParticleFieldManifest.json').read_text())['fields'] if app in f['storage']]
 decl=[];init=[];copycheck=[];swapcheck=[]
 for k,f in enumerate(fields):
  name,compact,w=f['name'],f['compact'],f['width'];ty=f['storage'][app].replace('GpuReal','Real').replace('GpuTime','double')
  decl.append(f'alignas(16) {ty} {name}Storage[{w*4}]={{}};alignas(16) {ty} {compact}Storage[{w*4}]={{}}; {ty} *{name}={name}Storage,*{compact}={compact}Storage;')
  init.append(f'for(int j=0;j<{4*w};++j){{s.{name}[j]=static_cast<{ty}>(j+{k+1});s.{compact}[j]=static_cast<{ty}>(-99);}}')
  if name=='pStatus': expected='1'
  elif name=='pCellId':expected='0'
  else:expected=f's.{name}[2*{w}+j]'
  copycheck.append(f'for(int j=0;j<{w};++j){{assert(s.{compact}[j]=={expected});assert(s.{compact}[{w}+j]==static_cast<{ty}>(-99));}}')
  swapcheck.append(f'assert(s.{name}==s.{compact}Storage&&s.{compact}==s.{name}Storage);')
 prefix=r'''
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#define __device__
#define __global__
#define __host__
#define __forceinline__ inline
#define GPU_OPERATOR_THERMAL 1
#define GPU_OPERATOR_REAL Real
#define GPU_OPERATOR_TIME double
#define GPU_OPERATOR_R(x) Real(x)
#include "GpuPrecisionTypes.H"
using Real=GpuReal;
struct Dim{unsigned x=0;}blockIdx,threadIdx,blockDim{1},gridDim{1};
struct alignas(16) float4{float x,y,z,w;};
using std::min;using std::max;
template<class T,class U>auto clampMin(T x,U l)->typename std::common_type<T,U>::type {using V=typename std::common_type<T,U>::type;return max(V(x),V(l));}template<class T,class U,class W>auto clampRange(T x,U l,W h)->typename std::common_type<T,U,W>::type {using V=typename std::common_type<T,U,W>::type;return min(max(V(x),V(l)),V(h));}
template<class T>T finiteOr(T x,T d){return std::isfinite(x)?x:d;}template<class T>bool finiteDevice(T x){return std::isfinite(x);}
double sqr3(double x,double y,double z){return x*x+y*y+z*z;}
template<class T>T atomicAdd(T*p,T d){T old=*p;*p+=d;return old;}template<class T>T atomicSub(T*p,T d){T old=*p;*p-=d;return old;}
constexpr Real OfSmall=Real(1e-12);
namespace ugkwp {inline Real radialDistributionG0FromRatio(Real r){return 1/(1-r);}}
namespace Foam{namespace gpuThermal{enum{particleWallMobile=0,coldWallAxialNodeCount=8,coldWallRadialRingCount=8,coldWall2DNodeCount=64,coldWall2DRadialNodeCount=8};}namespace gpuWall{template<class S>void publishWallBoundParticleIndex(S&s,int){++s.published;}}}
struct DeviceState{
FIELDS
 int nCells=1,nFaces=1,nInternalFaces=0,particleCapacity=4,nBoundarySources=1,coldWallSolidificationEnabled=1,coldWall2DEnabled=1,published=0;
 int count=2,compactCount=1,*particleCountDevice=&count,*compactCountDevice=&compactCount,compactCellOffset[2]={0,1},sortedParticleIndex[4]={2,0,0,0};
 int sourceFace[1]={0},sourceCell[1]={0},nScheduledInletFaces=0,*scheduledInletFaceMask=nullptr;
 Real injectionParcelMass=Real(2),particleDiameterMin=Real(.001),particleDiameterMax=Real(.002),particleDiameterSigma=0,particleDiameterFallback=Real(.001),TpMin=200,TpMax=4000,rhoSolid=3000;
 unsigned long long rngSeed=31;
 Real gasBoundaryUx[1]={},gasBoundaryUy[1]={},gasBoundaryUz[1]={},sourceUx[1]={1},sourceUy[1]={2},sourceUz[1]={3},sourceTheta[1]={4},theta[1]={},Sfx[1]={-1},Sfy[1]={},Sfz[1]={},sourceMassRate[1]={2},sourceResidualMass[1]={0},sourcePx[1]={.1},sourcePy[1]={.2},sourcePz[1]={.3},Cx[1]={},Cy[1]={},Cz[1]={},sourceT[1]={3000},sourceD[1]={.001};
};
'''.replace('FIELDS','\n'.join(decl))
 carry='CHT'==app
 extra=r'''
#include "GpuParticleThermalCarryStorage.cuh"
template<class S>auto resetSeparateContactAge(S&s,int i,int)->decltype(s.pContactAge[i]=0,void()){s.pContactAge[i]=0;}
template<class S>void resetSeparateContactAge(S&,int,long){}
Real scheduledSolidVolumeFractionDevice(DeviceState&,double){return 0;}
'''
 extra+=function(text('operators/clearColdWallParticleState.cuh'),'clearColdWallParticleState')+'\n'+function(text('wall/GpuColdWall2DDevice.cuh'),'clearColdWall2DParticleState')+'\n'
 extra+=function(text('operators/snapshotParticleGasCouplingStateKernel.cuh'),'mixSeed')+'\n'+text('operators/uniform01Device.cuh')+'\n'+text('operators/normalDevice.cuh')+'\n'+text('operators/linearScheduledValueDevice.cuh')+'\n'+text('operators/injectBoundaryParticlesKernel.cuh')
 extra+='\n'+text('GpuCellLocalThermalFields.cuh')+'\n#define GPU_PARTICLE_EXTRA_FIELDS CellLocalThermalExtraFields<'+('true,true' if carry else 'false,false')+'>\n'+text('GpuCellLocalPrimary.cuh')+'\n'+text('operators/gatherSelectedParticlesKernel.cuh')+'\n'+text('GpuParticlePayload.cuh').split('// Both scheduling policies',1)[0]+'\n'+text('GpuParticleBufferCommit.cuh')

 extra+='\n#include "GpuParticleCaloric.H"\n'
 extra+='double particleSpecificEnthalpyDevice(Real t){return ugkwp::ParticleCaloric{}.enthalpy(double(t));}\n#include "GpuMaterialEnthalpyMoment.cuh"\n'
 extra+='Real particleMomentThetaDevice(const DeviceState&s,int i){return s.pTheta[i];}\n#define GPU_MOMENT_REAL Real\n#define GPU_MOMENT_R(x) Real(x)\n#define GPU_MOMENT_THERMAL 1\ndouble heat(DeviceState&s,int i){\n#include "GpuParticleMomentContribution.inl"\nreturn particleHeat;}\n'
 main='int main(){DeviceState s;'+''.join(init)+r'''
 const auto expectedRng=mixSeed(s.rngSeed^(0xd1b54a32d192ed03ULL+2ULL));
 injectBoundaryParticlesKernel(&s,1.,0.);assert(s.count==3&&s.sourceResidualMass[0]==0&&s.pRng[2]==expectedRng&&s.pOrigId[2]==2&&s.pT[2]==3000&&s.pStatus[2]==1&&s.pCellId[2]==0);
'''
 if carry:main+='assert(s.pThermalEnthalpyResidual[2]==0);s.pThermalEnthalpyResidual[2]=.125;'
 main+='const double oldHeat=heat(s,2);const double authority=materialEnthalpyMomentFromParticle(s,2,s.pm[2],s.pT[2],0);s.pStatus[2]=2;'
 if carry:main+='assert(authority==double(materialEnthalpyMoment(s.pm[2],s.pT[2]))+.25);'
 if route=='selected':main+='gatherSelectedParticlesKernel(&s);'
 elif route=='payload':main+='s.compactPStatus[0]=2;gatherCellLocalParticlePayloadKernel<false>(&s);'
 else:main+='gatherCellLocalParticlePayloadKernel<true>(&s);'
 main+=''.join(copycheck)+'commitCellLocalParticleBuffersKernel(&s);assert(s.count==1);'+''.join(swapcheck)+'assert(s.pRng[0]==expectedRng&&s.pOrigId[0]==2&&heat(s,0)==oldHeat&&materialEnthalpyMomentFromParticle(s,0,s.pm[0],s.pT[0],0)==authority);}\n'
 cpp=tmp_path/'lineage.cpp';cpp.write_text(prefix+extra+main);exe=tmp_path/'lineage'
 q=subprocess.run(['g++','-std=c++17','-O2','-fno-strict-aliasing','-I'+str(C),str(cpp),'-o',str(exe)],capture_output=True,text=True)
 assert q.returncode==0,q.stderr
 q=subprocess.run([str(exe)],capture_output=True,text=True);assert q.returncode==0,q.stdout+q.stderr
