"""Real CUDA dispatcher switches plus frozen ordered-face/cold-wrapper oracles."""
from pathlib import Path
import json,re,resource,subprocess,sys

def main():
    root,out,app,bits=Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3],int(sys.argv[4])
    out.mkdir(parents=True,exist_ok=True)
    source=root/'applications'/app/('gpu' if app=='CHT' else 'private_backend')
    raw=(source/'GpuResidentStrict.cu').read_text();start=raw.index('struct DeviceState');end=raw.index('\n};',start)
    fields=re.findall(r'^\s*((?:unsigned\s+)?(?:long long|char)|double|float|int|GpuReal|GpuTime|GpuWallEnergy)\*\s+(\w+)\s*=',raw[start:end],re.M)
    fields=[(ty,name) for ty,name in fields if name not in ['diagnosticPreTransportParticleCount','sourceInjectedCount'] and (bits==32 or not name.startswith('flatPressure'))]
    frozen=json.loads(Path(__file__).with_name('cleanup_baseline.json').read_text())
    oldface=frozen['ordered_face'].replace('pressureDeltaFromLimitedFaces','baseline_pressureDeltaFromLimitedFaces')
    oldcold='' if app=='gasUGKP' else frozen['cold'+str(bits)].replace('relaxColdWall1DParticlesToResidentGasKernel','baseline_relaxColdWall')
    if app=='FSH':oldcold=oldcold.replace('s.pContactAge[','s.pTheta[')
    code='#include "GpuResidentStrict.cu"\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n#include <new>\n'
    if app=='CHT' and bits==32:code+='using namespace ugkwpCudaFp32;\n'
    code+=oldface+'\n'+oldcold+r'''
void ck(bool x,const char*m){if(!x){printf("FAIL %s\n",m);exit(2);}}
void check(cudaError_t e){ck(e==cudaSuccess,cudaGetErrorString(e));}
void finish(){check(cudaDeviceSynchronize());check(cudaGetLastError());}
template<class T> __attribute__((noinline)) void mem(T*&p,int n){check(cudaMallocManaged(&p,n*sizeof(T)));for(int i=0;i<n;i++)::new(static_cast<void*>(p+i)) T{};}
template<class T>void equal(T a,T b,const char*m){ck(!memcmp(&a,&b,sizeof(T)),m);}
__attribute__((noinline)) DeviceState* make(){DeviceState*s;mem(s,1);ALLOC return s;}
__global__ void facePair(DeviceState*s,DeviceState*r){int c=threadIdx.x+blockDim.x*blockIdx.x;if(c>=s->nCells)return;PressureReal a[4],b[4];pressureDeltaFromLimitedFaces(*s,c,PressureTime(1.23456789e-6),a);baseline_pressureDeltaFromLimitedFaces(*r,c,PressureTime(1.23456789e-6),b);for(int i=0;i<4;i++)if(a[i]!=b[i])asm("trap;");}
int main(){setvbuf(stdout,nullptr,_IONBF,0);DeviceState*s=make(),*r=make();
for(auto v:{s,r}){v->nCells=512;v->nFaces=1024;for(int c=0;c<512;c++){v->V[c]=.125+(c%7)*.03125;v->cellPlaneStart[c]=2*c;v->cellPlaneCount[c]=2;v->cellFaceId[2*c]=2*c;v->cellFaceId[2*c+1]=(c%13==0)?-1:2*c+1;}for(int f=0;f<1024;f++){v->faceOwner[f]=(f%2)?(f/2+1):f/2;v->solidPressurePhiMomX[f]=((f%19)-9)*.03125;v->solidPressurePhiMomY[f]=((f%17)-8)*1.1234567e-5;v->solidPressurePhiMomZ[f]=(f%3)*-.0625;v->solidPressurePhiEnergy[f]=((f%23)-11)*.00390625;}}
facePair<<<16,32>>>(s,r);finish();for(int c=0;c<512;c++){equal(s->pressureDeltaMomX[c],r->pressureDeltaMomX[c],"ordered face publication x");equal(s->pressureDeltaMomY[c],r->pressureDeltaMomY[c],"ordered face publication y");equal(s->pressureDeltaMomZ[c],r->pressureDeltaMomZ[c],"ordered face publication z");equal(s->pressureDeltaEnergy[c],r->pressureDeltaEnergy[c],"ordered face publication energy");}
puts("PASS ordered pressure accumulation and publication bitwise against frozen body");

// The host descriptor must remain ordinary host memory, as in production.
// WSL does not permit CPU descriptor reads while managed device state is busy.
DeviceState host=*s;s=&host;
s->nCells=4;s->particleCapacity=1024;s->reductionBlockThreads=32;s->multiprocessorCount=2;RESIDENCY
mem(s->csrReductionTasks,4096);s->csrHeavyTaskCapacity=4096;check(cudaFree(s->cellScanTempStorage));s->cellScanTempStorage=nullptr;check(cub::DeviceScan::ExclusiveSum(s->cellScanTempStorage,s->cellScanTempBytes,s->csrCellTaskCount,s->csrCellTaskOffset,s->nCells+1));check(cudaMalloc(&s->cellScanTempStorage,s->cellScanTempBytes));
DIRECTORY_KIND
s->deviceState=r;s->csrHeavyReductionMode=2;s->csrHeavyAutoInterval=3;s->schedulingAdvanceCount=0;s->csrHeavyReductionEnabled=1;SPLIT
auto offsets=[&](bool heavy){s->cellParticleOffset[0]=0;for(int c=0;c<4;c++)s->cellParticleOffset[c+1]=s->cellParticleOffset[c]+(heavy?(c==0?33:(c==3?31:32)):32);};
auto call=[&](){*r=*s;CALL;finish();ck(s->csrHeavyReductionEnabled==r->csrHeavyReductionEnabled,"host/device automatic decision agreement");ck(s->csrHeavyReductionActive==r->csrHeavyReductionActive,"active publication agreement");};
offsets(false);call();ck(s->csrHeavyReductionEnabled==0,"equality at threshold uses L1");
offsets(true);call();ck(s->csrHeavyReductionEnabled==0,"inspection interval retains prior L1 decision");
call();ck(s->csrHeavyReductionEnabled==1,"above threshold activates L2");
offsets(false);call();ck(s->csrHeavyReductionEnabled==1,"interval retains prior L2 decision");call();call();ck(s->csrHeavyReductionEnabled==0,"decreasing occupancy deactivates L2");
for(int c=0;c<=4;c++)s->cellParticleOffset[c]=0;call();call();call();ck(s->csrHeavyReductionEnabled==0,"empty directory uses L1");
s->csrHeavyAutoInterval=1;s->schedulingAdvanceCount=0;
ENABLE_COMBINED_DIRECTORY
s->preBaseCellOffset[0]=s->cellParticleOffset[0]=0;
for(int c=0;c<4;c++){s->preBaseCellOffset[c+1]=s->preBaseCellOffset[c]+16;s->cellParticleOffset[c+1]=s->cellParticleOffset[c]+16;}
*s->preBaseParticleCountDevice=64;call();ck(s->csrHeavyReductionEnabled==0,"combined base/injection equality stays L1");
s->preBaseCellOffset[1]=17;call();ck(s->csrHeavyReductionEnabled==1,"combined base/injection above threshold enables L2");
s->preBaseCellOffset[1]=16;call();ck(s->csrHeavyReductionEnabled==0,"combined directory decrease disables L2");
s->csrHeavyReductionMode=1;s->csrHeavyReductionActive=1;s->csrHeavyReductionEnabled=1;auto count=s->schedulingAdvanceCount;call();ck(s->csrHeavyReductionEnabled==1&&s->schedulingAdvanceCount==count,"explicit L2 never automatically disabled");
puts("PASS actual automatic low-high-low-empty transitions, interval, strict threshold and fixed L2");
COLD_TEST
return 0;}
'''
    code=code.replace('ALLOC','\n'.join(f'mem(s->{name},4096);' for _,name in fields))
    code=code.replace('RESIDENCY','s->lightBlocksPerSm=2;' if app=='gasUGKP' else 's->lightResidentBlocksPerSm=2;')
    code=code.replace('SPLIT','' if app=='gasUGKP' else 's->splitPreDirectoryActive=0;')
    code=code.replace('DIRECTORY_KIND','auto kind=HeavyDirectoryKind::full;' if app=='gasUGKP' else '')
    code=code.replace('ENABLE_COMBINED_DIRECTORY','kind=HeavyDirectoryKind::splitBaseAndInjection;' if app=='gasUGKP' else 's->splitPreDirectoryActive=1;')
    code=code.replace('CALL','ck(runToolB3(s,32,kind)==0,"automatic dispatch status");' if app=='gasUGKP' else 'ck(runToolB3(s,32)==0,"automatic dispatch status");')
    cold=''
    if app!='gasUGKP':
        cold=r'''
s=make();r=make();
for(auto v:{s,r}){
 v->nCells=1;v->nFaces=1;v->particleCapacity=1;*v->wallBoundParticleCountDevice=1;v->wallBoundParticleIndex[0]=0;
 v->pStatus[0]=1;v->pStuck[0]=Foam::gpuThermal::particleWallTransientDeposit;v->pStuckFaceId[0]=0;v->pCellId[0]=0;
 v->particleWallHeatTransferEnabled=1;v->particleStuckCandidateMask[0]=Foam::gpuThermal::particleWallSolidifyingDeposition;
 v->pd[0]=1e-4;v->pT[0]=3500;v->pm[0]=3.2e-9;v->TpMin=300;v->TpMax=5000;v->TgasMin=300;v->rhoMin=1e-12;v->rhoSolid=3990;
 v->couplingTgasOld[0]=3200;v->couplingRhoOld[0]=1;v->couplingUxOld[0]=2;v->gasMu=1e-5;v->gasPrOneThird=.9;v->gammaGas=1.4;v->gasCp=1000;
 v->gasBoundaryT[0]=300;v->particleWallEffusivityByFace[0]=13000;v->particleWallContactAreaScale[0]=1;
 v->particleWallReflectionHeatTransferEfficiency=.1;v->particleWallDepositionHeatTransferEfficiency=.1;
 v->pContactMaximumArea[0]=2.2e-8;v->pContactDuration[0]=8e-6;v->pContactPeakFraction[0]=.42;v->pDepositionArea[0]=1.7e-8;
 v->coldWallSolidificationEnabled=1;v->coldWallSolidificationParameters={2327.,20.,1.16e6,3990.,1273.,5.9,.25,0.,1,4};
 for(int n=0;n<8;n++)v->pColdNodeSpecificEnthalpy[n]=float(Foam::gpuThermal::coldWallSpecificEnthalpyJkg(3500.,v->coldWallSolidificationParameters));
}
for(int heat:{0,1})for(int state:{int(Foam::gpuThermal::particleWallTransientDeposit),int(Foam::gpuThermal::particleWallDeposited)}){
 s->solveParticleTemperature=r->solveParticleTemperature=heat;s->particleGasHeatTransferModelId=r->particleGasHeatTransferModelId=heat;
 s->pStuck[0]=r->pStuck[0]=state;
 for(int step=0;step<16;step++){
  relaxColdWall1DParticlesToResidentGasKernel<<<1,32>>>(s,GpuTime(7.5e-8));baseline_relaxColdWall<<<1,32>>>(r,GpuTime(7.5e-8));finish();
  equal(s->pT[0],r->pT[0],"outer cold temperature");equal(s->pColdContactAge[0],r->pColdContactAge[0],"outer cold age");
  equal(s->particleWallReflectedEnergy[0],r->particleWallReflectedEnergy[0],"outer reflected wall heat");equal(s->particleWallDepositedEnergy[0],r->particleWallDepositedEnergy[0],"outer deposited wall heat");
  for(int n=0;n<8;n++){equal(s->pColdNodeSpecificEnthalpy[n],r->pColdNodeSpecificEnthalpy[n],"outer node enthalpy");equal(s->pColdRingSolidMass[n],r->pColdRingSolidMass[n],"outer ring mass");}
 }
}
puts("PASS full cold-wall outer kernel bitwise, finite/deposited contacts, gas heat on/off");
'''
    code=code.replace('COLD_TEST',cold)
    cu=out/'auto_cleanup.cu';cu.write_text(code);exe=out/'auto_cleanup'
    soft,hard=resource.getrlimit(resource.RLIMIT_STACK);resource.setrlimit(resource.RLIMIT_STACK,(min(128*1024*1024,hard) if hard!=resource.RLIM_INFINITY else 128*1024*1024,hard));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','-arch=sm_89','--fmad='+('false' if app=='CHT' else 'true'),'-DUGKWP_GPU_REAL_BITS='+str(bits),'-I'+str(source),'-I'+str(root/'common'),'-I'+str(root/'applications'/app/'gpu'),str(cu),'-o',str(exe)]
    if app=='CHT':cmd.append(str(source/'GpuWallEnergy64.cu'))
    with (out/'build.log').open('w') as log:q=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if q.returncode:print((out/'build.log').read_text()[-5000:]);return q.returncode
    q=subprocess.run([str(exe)],capture_output=True,text=True);(out/'run.log').write_text(q.stdout+q.stderr);print(q.stdout+q.stderr);return q.returncode
if __name__=='__main__':raise SystemExit(main())
