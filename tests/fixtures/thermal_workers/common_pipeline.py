"""Exercise actual CUDA thermal kernels against independent expectations."""
from pathlib import Path
import re,sys,subprocess
root=Path(sys.argv[1]).resolve()
P=Path(sys.argv[2]).resolve()
branch,bits=sys.argv[3],int(sys.argv[4]);mode='candidate'
for sub in ['tests','bin','logs']:(P/sub).mkdir(parents=True,exist_ok=True)
folder='private_backend' if branch=='FSH' else 'gpu';src=root/'applications'/branch/folder/'GpuResidentStrict.cu'
s=src.read_text().replace('#include "GpuAutomaticCsrScheduleFields.inl"', (root/"common/GpuAutomaticCsrScheduleFields.inl").read_text() if (root/"common/GpuAutomaticCsrScheduleFields.inl").is_file() else "");a=s.index('struct DeviceState');b=s.index('\n};',a);d=s[a:b]
ptr=re.findall(r'^\s*((?:unsigned\s+)?(?:long long|char)|double|float|int|GpuReal|GpuTime|GpuWallEnergy)\*\s+(\w+)\s*=',d,re.M)
ptr=[(t,n) for t,n in ptr if n not in ['diagnosticPreTransportParticleCount','sourceInjectedCount','flatPressureParameters','flatPressureFlags']]
pairs=[]
for typ,n in ptr:
 if n.startswith('compactP'):
  original='p'+n[len('compactP'):]
  if original in dict((n,t) for t,n in ptr):pairs.append((typ,original,n))
def width(n):
 if 'Cold2DNodeSpecificEnthalpy' in n:return 'Foam::gpuThermal::coldWall2DNodeCount'
 if 'Cold2DRingContactAge' in n:return 'Foam::gpuThermal::coldWall2DRadialNodeCount'
 if 'ColdNodeSpecificEnthalpy' in n:return 'Foam::gpuThermal::coldWallAxialNodeCount'
 if 'ColdRingSolidMass' in n:return 'Foam::gpuThermal::coldWallRadialRingCount'
 return '1'
text=r"""
#include "GpuResidentStrict.cu"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
template<class T> void mem(T*&p,int n){if(cudaMallocManaged(&p,sizeof(T)*n)!=cudaSuccess)std::abort();for(int i=0;i<n;++i)p[i]=T{};}
#define CHECK(x) do{if(!(x)){printf("FAIL line%d %s\n",__LINE__,#x);return 1;}}while(0)
#ifdef FUSED13
__global__ void projectBoth(DeviceState* s){
 for(int pos=threadIdx.x;pos<s->cellParticleOffset[s->nCells];pos+=blockDim.x){
  int i=s->sortedParticleIndex[pos];
  applyPressureParticleStateDevice<false>(*s,i,1,2,3,2,4,6,.125,.5,.25,true);
  applyPressureParticleStateDevice<true>(*s,pos,1,2,3,2,4,6,.125,.5,.25,true);
 }
}
#endif
int main(){
 setvbuf(stdout,nullptr,_IONBF,0);printf("fixture start\n");
 DeviceState* s;mem(s,1);s->deviceState=s;s->nCells=4;s->particleCapacity=129;s->rhoMin=1e-20;s->TpMin=1;s->TpMax=5000;s->particleDiameterFallback=1e-6;s->csrHeavyReductionEnabled=1;s->csrHeavyWorkerGrid=3;s->multiprocessorCount=24;
"""
text+='\n'.join('mem(s->'+n+',129*('+width(n)+'));' for typ,n in ptr)
text+='\nmem(s->csrReductionTasks,129);printf("allocated\\n");\n'
text+='for(int block:{32,64,128,256})for(int scenario=0;scenario<5;++scenario)for(int flags=0;flags<4;++flags){\n'
text+='s->coldWallSolidificationEnabled=flags&1;s->coldWall2DEnabled=flags&2;*s->particleCountDevice=129;*s->wallBoundParticleCountDevice=0;\n'
for j,(typ,n,c) in enumerate(pairs):text+=f'for(int x=0;x<129*({width(n)});++x){{s->{n}[x]=static_cast<{typ}>(x%13+{j+1});s->{c}[x]=0;}}\n'
text+=r"""
  int expected[4]={},wall=0;
 for(int i=0;i<129;++i){
  s->pStatus[i]=scenario==0?1:scenario==1?(i%4):scenario==2?0:scenario==3?(i%2):(i==128);
  s->pCellId[i]=i<100?0:2;if(scenario==3 && i%11==0)s->pCellId[i]=-1;
  s->pStuck[i]=i%3;s->puxOld[i]=1000+i;s->puyOld[i]=2000+i;s->puzOld[i]=3000+i;s->pm[i]=2;s->pRng[i]=1234+i;s->pOrigId[i]=4321+i;
  if(s->pStatus[i]==1&&s->pCellId[i]>=0){++expected[s->pCellId[i]];wall+=s->pStuck[i]!=0;}
 }
  clearParticleCellBinsKernel<<<1,32>>>(s);
#ifdef FUSED13
 countParticlesByCellKernel<true,true><<<5,32>>>(s);
#else
 countParticlesByCellKernel<true><<<5,32>>>(s);
#endif
 CHECK(cudaDeviceSynchronize()==cudaSuccess);
 for(int c=0;c<4;++c)CHECK(s->cellParticleCount[c]==expected[c]);
 s->cellParticleOffset[0]=0;
 for(int c=0;c<4;++c){s->V[c]=1;s->cellParticleOffset[c+1]=s->cellParticleOffset[c]+expected[c];}
 initialiseParticleCellWritesKernel<<<1,32>>>(s);
#ifdef FUSED13
 scatterParticlesByCellKernel<true,true><<<5,32>>>(s);
#else
 scatterParticlesByCellKernel<true><<<5,32>>>(s);
#endif
 CHECK(cudaDeviceSynchronize()==cudaSuccess);
 int nt=0,nh=0;
 for(int c=0;c<4;++c){s->csrCellTaskOffset[c]=nt;s->csrCellTaskCount[c]=0;
  for(int start=s->cellParticleOffset[c];start<s->cellParticleOffset[c+1];start+=17){s->csrReductionTasks[nt++]={c,start,std::min(start+17,s->cellParticleOffset[c+1]),0};++s->csrCellTaskCount[c];}
  if(s->csrCellTaskCount[c]>1)s->csrMultiTaskCellList[nh++]=c;
 }
 s->csrCellTaskOffset[4]=nt;*s->csrHeavyTaskCount=nt;*s->csrHeavyCellCount=nh;
 clearParticleMomentsKernel<<<1,32>>>(s);
 CHECK(cudaDeviceSynchronize()==cudaSuccess);DeviceState host=*s;
#ifdef FUSED13
 CHECK(launchCsrSegmentedMomentReduction(&host,block,true)==0);
#else
 CHECK(launchCsrSegmentedMomentReduction(&host,block)==0);
#endif
 CHECK(cudaDeviceSynchronize()==cudaSuccess);
 for(int c=0;c<4;++c)CHECK(s->momRhoP[c]==2*expected[c]);
 CHECK(*s->wallBoundParticleCountDevice==wall);
 std::vector<int> seen(129,0);
 for(int j=0;j<wall;++j){int dst=s->wallBoundParticleIndex[j];CHECK(dst>=0&&dst<s->cellParticleOffset[4]);++seen[dst];}
#ifdef FUSED13
 projectBoth<<<1,64>>>(s);CHECK(cudaDeviceSynchronize()==cudaSuccess);
#endif
 for(int c=0;c<4;++c)for(int pos=s->cellParticleOffset[c];pos<s->cellParticleOffset[c+1];++pos){int i=s->sortedParticleIndex[pos];
 CHECK(seen[pos]==(s->pStuck[i]!=0));
"""
for typ,n,c in pairs:
 guard='s->coldWall2DEnabled' if 'Cold2D' in n else 's->coldWallSolidificationEnabled' if 'Cold' in n else 'true'
 text+=f'if({guard})for(int x=0;x<({width(n)});++x)CHECK(s->{c}[pos*({width(n)})+x]==s->{n}[i*({width(n)})+x]);\n'
text+=r"""
 }
 for(int c=0;c<=4;++c)s->compactCellOffset[c]=s->cellParticleOffset[c];
 commitCellLocalParticleBuffersKernel<<<1,1>>>(s);CHECK(cudaDeviceSynchronize()==cudaSuccess);
 CHECK(*s->particleCountDevice==s->cellParticleOffset[4]);
 }
 printf("PASS actual survivor bins, shared moment queue, all thermal payloads, wall publication, compact pressure and commit; 32/64/128/256 threads; full/filter/empty/sparse/single; cold1D/2D on/off\n");
}
"""
if branch=='FSH':
 text=text.replace('  applyPressureParticleStateDevice<false>(*s,i,1,2,3,2,4,6,.125,.5,.25,true);','')
 text=text.replace('#endif\nint main()',r"""
__global__ void projectOriginal(DeviceState* s){
 for(int pos=threadIdx.x;pos<s->cellParticleOffset[s->nCells];pos+=blockDim.x){
  int i=s->sortedParticleIndex[pos];
  applyPressureParticleStateDevice<false>(*s,i,1,2,3,2,4,6,.125,.5,.25,true);
 }
}
#endif
int main()""")
 text=text.replace(' projectBoth<<<1,64>>>(s);CHECK(cudaDeviceSynchronize()==cudaSuccess);',r"""
 projectBoth<<<1,64>>>(s);CHECK(cudaDeviceSynchronize()==cudaSuccess);
 for(int i=0;i<129;++i){bool deposited=s->pStatus[i]==1 && s->pCellId[i]>=0 && s->pStuck[i]==Foam::gpuThermal::particleWallDeposited;
  CHECK(s->puxOld[i]==(deposited?0:1000+i));CHECK(s->puyOld[i]==(deposited?0:2000+i));CHECK(s->puzOld[i]==(deposited?0:3000+i));}
 projectOriginal<<<1,64>>>(s);CHECK(cudaDeviceSynchronize()==cudaSuccess);""")
if branch=='FSH':
 text=text.replace(' projectOriginal<<<1,64>>>(s);CHECK(cudaDeviceSynchronize()==cudaSuccess);',r"""
 projectOriginal<<<1,64>>>(s);CHECK(cudaDeviceSynchronize()==cudaSuccess);
 for(int cutoff=0;cutoff<2;++cutoff){
  s->epsSMin=cutoff?1e10:0;s->nFaces=4;
  for(int c=0;c<4;++c){s->cellPlaneStart[c]=c;s->cellPlaneCount[c]=1;s->cellFaceId[c]=c;s->faceOwner[c]=c;s->solidPressurePhiMomX[c]=0.01;s->solidPressurePhiEnergy[c]=0.1;}
  double input[4][4];double* moment[4]={s->momRhoUPx,s->momRhoUPy,s->momRhoUPz,s->momRhoEP};for(int c=0;c<4;++c)for(int j=0;j<4;++j)input[c][j]=moment[j][c];
  applyCollisionalPressureProjectionKernel<false,false><<<4,block,8*(block/32)*sizeof(double)>>>(s,.01);CHECK(cudaDeviceSynchronize()==cudaSuccess);
  double ref[4][4];int counts[4];for(int c=0;c<4;++c){counts[c]=s->cellParticleCount[c];for(int j=0;j<4;++j){ref[c][j]=moment[j][c];moment[j][c]=input[c][j];}}
  applyCollisionalPressureProjectionKernel<false,true><<<4,block,8*(block/32)*sizeof(double)>>>(s,.01);CHECK(cudaDeviceSynchronize()==cudaSuccess);
  for(int c=0;c<4;++c){CHECK(counts[c]==s->cellParticleCount[c]);for(int j=0;j<4;++j)CHECK(ref[c][j]==moment[j][c]);}
 }
""")
text=text.replace(' CHECK(launchCsrSegmentedMomentReduction(&host,block,true)==0);',r"""
 CHECK(launchCsrSegmentedMomentReduction(&host,block)==0);
 CHECK(cudaDeviceSynchronize()==cudaSuccess);CHECK(*s->wallBoundParticleCountDevice==0);
 for(int pos=0;pos<s->cellParticleOffset[4];++pos)CHECK(s->compactPStatus[pos]==0);
 clearParticleMomentsKernel<<<1,32>>>(s);CHECK(cudaDeviceSynchronize()==cudaSuccess);
 CHECK(launchCsrSegmentedMomentReduction(&host,block,true)==0);""")
if branch=='CHT':
 if bits==32:text=text.replace('#include <cstdio>', 'using namespace ugkwpCudaFp32;\n#include <cstdio>')
 text=text.replace('  applyPressureParticleStateDevice<false>(*s,i,','  applyPressureParticleStateDevice(*s,i,')
 text=text.replace('  applyPressureParticleStateDevice<true>(*s,pos,1,2,3,2,4,6,.125,.5,.25,true);','')
 text=text.replace(' CHECK(*s->wallBoundParticleCountDevice==wall);',r"""
 for(int pos=0;pos<s->cellParticleOffset[4];++pos)CHECK(s->compactPStatus[pos]==0);
#ifdef FUSED13
 projectBoth<<<1,64>>>(s);CHECK(cudaDeviceSynchronize()==cudaSuccess);
#endif
 for(int c=0;c<=4;++c)s->compactCellOffset[c]=s->cellParticleOffset[c];
 gatherCellLocalParticlePayloadKernel<true><<<5,32>>>(s);CHECK(cudaDeviceSynchronize()==cudaSuccess);
 CHECK(*s->wallBoundParticleCountDevice==wall);""")
 text=text.replace('#ifdef FUSED13\n projectBoth<<<1,64>>>(s);CHECK(cudaDeviceSynchronize()==cudaSuccess);\n#endif\n for(int c=0;c<4;', ' for(int c=0;c<4;')

level=sys.argv[5] if len(sys.argv)>5 else 'L2'
if level=='S1':
 text=text.replace('s->csrHeavyReductionEnabled=1;', 's->csrHeavyReductionEnabled=0;')
 text=text.replace('CHECK(launchCsrSegmentedMomentReduction(&host,block)==0);','accumulateParticleMomentsSegmentedKernel<false,postTransportFusePayload><<<4,block,8*(block/32)*sizeof(GpuReal)>>>(s);')
 text=text.replace('CHECK(launchCsrSegmentedMomentReduction(&host,block,true)==0);','accumulateParticleMomentsSegmentedKernel<false,postTransportFusePayload><<<4,block,8*(block/32)*sizeof(GpuReal)>>>(s);')
text=text.replace('shared moment queue',level+' shared moment kernel')
mode=mode+'_'+level
f=P/'tests'/f'exact13_{branch}_{bits}_{mode}.cu';f.write_text(text);exe=P/'bin'/f'exact13_{branch}_{bits}_{mode}'
cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','-arch='+__import__('os').environ.get('UGKWP_CUDA_ARCH','sm_89'),'--fmad=false','-DUGKWP_GPU_REAL_BITS='+str(bits),'-I'+str(src.parent),'-I'+str(root/'common'),'-I'+str(root/'applications/FSH/gpu'),'-I'+str(root/'applications/CHT/thermal'),str(f),'-o',str(exe)]
if branch=='CHT':cmd.append(str(root/'applications/CHT/gpu/GpuWallEnergy64.cu'))
if mode!='baseline':cmd.append('-DFUSED13')
log=P/'logs'/f'fixture_{branch}_{bits}_{mode}_build.log'
with log.open('w') as out:q=subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT)
print('BUILD',branch,bits,mode,q.returncode,flush=True)
if q.returncode:print(log.read_text()[-5000:]);sys.exit(q.returncode)
if __import__('os').environ.get('UGKP_CUDA_BUILD_ONLY')=='1':print('BUILD ONLY: GPU execution deferred');sys.exit(0)
q=subprocess.run([str(exe)],capture_output=True,text=True);(P/'logs'/f'fixture_{branch}_{bits}_{mode}_run.log').write_text(q.stdout+q.stderr);print(q.stdout+q.stderr);sys.exit(q.returncode)
