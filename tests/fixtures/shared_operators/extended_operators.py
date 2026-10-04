import os
from pathlib import Path
import re,subprocess,sys
root=Path(sys.argv[1]);out=Path(sys.argv[2]);app=sys.argv[3];bits=int(sys.argv[4]);out.mkdir(parents=True,exist_ok=True)
src=root/f'applications/{app}/{"gpu" if app=="CHT" else "private_backend"}'
t=(src/'GpuResidentStrict.cu').read_text().replace('#include "GpuAutomaticCsrScheduleFields.inl"', (root/"common/GpuAutomaticCsrScheduleFields.inl").read_text() if (root/"common/GpuAutomaticCsrScheduleFields.inl").is_file() else "");a=t.index('struct DeviceState');b=t.index('\n};',a)
fields=re.findall(r'^\s*((?:unsigned\s+)?(?:long long|char)|double|float|int|GpuReal|GpuTime|GpuWallEnergy)\*\s+(\w+)\s*=',t[a:b],re.M)
fields=[(ty,n) for ty,n in fields if not n.startswith('flatPressure') and n not in ['diagnosticPreTransportParticleCount','sourceInjectedCount']]
code='#define UGKP_DEVELOPMENT_PROBES 1\n#include "GpuResidentStrict.cu"\n#include <cstdio>\n#include <new>\n#include <cstdlib>\n#include <cmath>\n#include <set>\n'
if app=='CHT' and bits==32:code+='using namespace ugkwpCudaFp32;\n'
code+=r'''
template<class T>void mem(T*&p){if(cudaMallocManaged(&p,256*sizeof(T))!=cudaSuccess)abort();for(int i=0;i<256;i++)::new(static_cast<void*>(p+i)) T{};}
void deviceSync(){auto e=cudaDeviceSynchronize();if(e!=cudaSuccess){printf("CUDA %s\n",cudaGetErrorString(e));exit(10);}}
void ck(bool x,const char*m){if(!x){printf("FAIL %s\n",m);exit(20);}}
void near(double a,double b,const char*m){if(!(std::isfinite(a)&&fabs(a-b)<=TOL*fmax(1.,fabs(b)))){printf("MISMATCH %s actual=%.17g expected=%.17g\n",m,a,b);ck(false,m);}}
__global__ void geometry(DeviceState*s,double*out){GPU_OPERATOR_REAL ht;out[0]=pointInsideCell(*s,0,0,0,0);out[1]=pointInsideCell(*s,0,2,0,0);out[2]=firstSegmentIntersection(*s,0,0,0,0,2,0,0,ht);out[3]=ht;}
int main(){setvbuf(stdout,nullptr,_IONBF,0);DeviceState*s;mem(s);
ALLOC
s->nCells=2;s->nFaces=2;s->particleCapacity=128;s->rhoMin=1e-12;s->rhoSolid=3000;s->TpMin=1;s->TpMax=5000;s->particleDiameterFallback=1e-4;
s->cellPlaneStart[0]=0;s->cellPlaneCount[0]=2;s->cellLength[0]=1;s->cellFaceId[0]=0;s->cellFaceId[1]=1;
s->planeNx[0]=-1;s->planeNx[1]=1;s->planeD[0]=s->planeD[1]=1;s->faceCx[0]=-1;s->faceCx[1]=1;
double*geom;mem(geom);geometry<<<1,1>>>(s,geom);deviceSync();ck(geom[0]==1&&geom[1]==0&&geom[2]==1,"geometry membership/intersection");near(geom[3],.5,"geometry intersection fraction");
puts("PASS geometry membership and crossing plane");
for(int i=0;i<37;i++){s->pStatus[i]=i%7?1:0;s->pCellId[i]=i%2;s->pm[i]=.5;s->pd[i]=1e-4;s->pT[i]=300;s->pux[i]=2;s->puxOld[i]=4;STUCK_INIT}
s->pCellId[5]=-1;*s->particleCountDevice=37;*s->preBaseParticleCountDevice=3;
for(int warp=0;warp<2;warp++){
 for(int c=0;c<3;c++)s->cellParticleCount[c]=0;
 if(warp)countSplitPreInjectionParticlesKernel<true><<<2,32>>>(s);else countSplitPreInjectionParticlesKernel<false><<<2,32>>>(s);deviceSync();
 int expect[2]={};std::set<int> expected;
 for(int i=3;i<37;i++)if(s->pStatus[i]&&s->pCellId[i]>=0){expect[s->pCellId[i]]++;expected.insert(i);}
 ck(s->cellParticleCount[0]==expect[0]&&s->cellParticleCount[1]==expect[1],"injection directory count");
 s->cellParticleOffset[0]=0;s->cellParticleOffset[1]=expect[0];s->cellParticleOffset[2]=expect[0]+expect[1];s->cellParticleWrite[0]=0;s->cellParticleWrite[1]=expect[0];
 if(warp)scatterSplitPreInjectionParticlesKernel<true><<<2,32>>>(s);else scatterSplitPreInjectionParticlesKernel<false><<<2,32>>>(s);deviceSync();
 std::set<int> got;for(int c=0;c<2;c++)for(int j=s->cellParticleOffset[c];j<s->cellParticleOffset[c+1];j++){int i=s->sortedParticleIndex[j];ck(s->pCellId[i]==c,"directory cell");ck(got.insert(i).second,"directory duplicate ID");}ck(got==expected,"directory exact coverage");
}
puts("PASS injection-directory atomic and warp count/scatter exact coverage");
accumulateMobilePackingMomentsKernel<<<2,32>>>(s);deviceSync();
for(int c=0;c<2;c++){double mass=0,stuck=0;for(int i=0;i<37;i++)if(s->pStatus[i]==1&&s->pCellId[i]==c){if(IS_STUCK)stuck+=.5;else mass+=.5;}near(s->mobilePackingRho[c],mass,"packing mobile mass");near(s->mobilePackingMomX[c],3*mass,"packing momentum");STUCK_CHECK}
puts("PASS packing mobile/constraint mass and averaged momentum");
// Injection: fractional mass balance and capacity saturation use the same source ledger.
s->particleCapacity=3;*s->particleCountDevice=0;s->nBoundarySources=1;s->injectionParcelMass=1;s->sourceFace[0]=0;s->sourceCell[0]=0;s->sourceMassRate[0]=2.5;s->sourceUx[0]=3;s->sourceT[0]=400;s->sourceTheta[0]=.25;s->sourceD[0]=1e-4;
injectBoundaryParticlesKernel<<<1,32>>>(s,1.,0.);deviceSync();ck(*s->particleCountDevice==2,"injection number");near(s->sourceResidualMass[0],.5,"fractional residual");
injectBoundaryParticlesKernel<<<1,32>>>(s,1.,1.);deviceSync();ck(*s->particleCountDevice==3,"capacity respected");near(s->sourceResidualMass[0],2.,"saturated residual");
std::set<unsigned long long> ids;for(int i=0;i<3;i++){near(s->pm[i],1,"parcel mass");near(s->pux[i],3,"injected velocity");near(s->pT[i],400,"injected temperature");near(s->pTheta[i],.25,"injected theta");ck(s->pCellId[i]==0&&s->pStatus[i]==1,"injected metadata");ck(ids.insert(s->pOrigId[i]).second,"injection ID");INJECT_CONTACT}
near(3+s->sourceResidualMass[0],5,"total supplied injection mass");puts("PASS injection mass residual, momentum/state and capacity saturation");
DevelopmentProbeDeviceSummary*summary;mem(summary);s->particleCapacity=3;*s->particleCountDevice=3;
validateDevelopmentProbeParticlesKernel<<<1,32>>>(s,summary);deviceSync();ck(summary->badParticles==0,"diagnostic valid fixture");
s->px[0]=NAN;s->pCellId[1]=-1;s->pm[2]=-1;validateDevelopmentProbeParticlesKernel<<<1,32>>>(s,summary);deviceSync();ck(summary->badParticles==3,"diagnostic bad count");ck((summary->badFieldMask&(ProbeBadParticlePosition|ProbeBadParticleMetadata|ProbeBadParticleThermal))==(ProbeBadParticlePosition|ProbeBadParticleMetadata|ProbeBadParticleThermal),"diagnostic bad masks");
puts("PASS diagnostic detects position, metadata and mass faults");
RADIATION
return 0;}
'''
code=code.replace('ALLOC','\n'.join('mem(s->'+n+');' for _,n in fields)).replace('TOL','2e-6' if bits==32 else '2e-9')
code=code.replace('STUCK_INIT','s->pStuck[i]=(i%3==0);' if app!='gasUGKP' else '').replace('IS_STUCK','s->pStuck[i]!=0' if app!='gasUGKP' else 'false').replace('STUCK_CHECK','near(s->packingStuckRho[c],stuck,"packing constraint mass");' if app!='gasUGKP' else '').replace('INJECT_CONTACT','ck(s->pStuck[i]==0&&s->pStuckFaceId[i]==-1,"injected contact reset");' if app!='gasUGKP' else '')
radiation=r'''
s->solveParticleTemperature=1;s->particleCapacity=8;*s->particleCountDevice=8;double expected[2][4]={};
for(int c=0;c<2;c++){s->V[c]=1;s->preBaseCellOffset[c]=4*c;s->momRhoP[c]=2;s->momRhoHpP[c]=0;}
s->preBaseCellOffset[2]=8;
for(int i=0;i<8;i++){int c=i/4;s->pStatus[i]=1;s->pCellId[i]=c;s->pStuck[i]=(i%3==0);s->pm[i]=.5;s->pT[i]=300+10*i;s->pd[i]=1e-4;
 expected[c][3]+=.5*Foam::gpuThermal::aluminaSpecificEnthalpyJkg(s->pT[i]);if(!s->pStuck[i]){expected[c][0]+=.5;expected[c][1]+=.5*s->pT[i];expected[c][2]+=.5*s->pd[i];}}
clearMobileParticleRadiationSumsKernel<<<1,32>>>(s);accumulateMobileParticleRadiationSumsAtomicKernel<<<1,32>>>(s);accumulateParticleEnthalpyMomentAtomicKernel<<<1,32>>>(s);deviceSync();
for(int c=0;c<2;c++){near(s->radiationMobileMass[c],expected[c][0],"atomic radiation mass");near(s->radiationMobileTemperatureMass[c],expected[c][1],"atomic radiation temperature");near(s->radiationMobileDiameterMass[c],expected[c][2],"atomic radiation diameter");near(s->momRhoHpP[c],expected[c][3],"atomic enthalpy");}
accumulatePackedMobileParticleRadiationSumsKernel<<<2,32,3*sizeof(GpuReal)>>>(s);refreshPackedParticleEnthalpyKernel<<<2,32,sizeof(GpuReal)>>>(s);deviceSync();
for(int c=0;c<2;c++){near(s->radiationMobileMass[c],expected[c][0],"packed radiation mass");near(s->radiationMobileTemperatureMass[c],expected[c][1],"packed radiation temperature");near(s->radiationMobileDiameterMass[c],expected[c][2],"packed radiation diameter");near(s->momRhoHpP[c],expected[c][3],"packed enthalpy");}
puts("PASS atomic/packed radiation mobile membership and all-alive material enthalpy");
'''
code=code.replace('RADIATION',radiation if app=='CHT' else '')
f=out/'extended.cu';f.write_text(code);exe=out/'extended';cmd=[str(Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin/nvcc'),'-std=c++17','-O3','-arch='+os.environ.get('UGKWP_CUDA_ARCH','sm_89'),'--fmad='+('false' if app=='CHT' else 'true'),'-DUGKWP_GPU_REAL_BITS='+str(bits),'-I'+str(src),'-I'+str(root/'common'),'-I'+str(root/'applications'/app/'gpu'),str(f),'-o',str(exe)]
if app=='CHT':cmd.append(str(src/'GpuWallEnergy64.cu'))
with (out/'build.log').open('w') as log:q=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
if q.returncode:print((out/'build.log').read_text()[-6000:]);sys.exit(q.returncode)
q=subprocess.run([str(exe)],capture_output=True,text=True);(out/'run.log').write_text(q.stdout+q.stderr);print(app,bits,q.stdout,q.stderr);sys.exit(q.returncode)
