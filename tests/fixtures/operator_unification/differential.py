"""Run actual CUDA implementations against frozen pre-extraction device bodies.

No timings or throughput measurements are collected. Fixtures exercise precision
boundaries, periodic/internal faces, sparse wall channels and cold-wall updates.
"""
from pathlib import Path
import argparse, json, os, re, resource, subprocess

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('app',choices=['gasUGKP','FSH','CHT']);parser.add_argument('bits',type=int,choices=[32,64])
    args=parser.parse_args();root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    app,bits=args.app,args.bits
    source=root/'applications'/app/('gpu' if app=='CHT' else 'private_backend')
    frozen=json.loads((root/'tests/fixtures/operator_unification/baseline.json').read_text())
    code='#include "GpuResidentStrict.cu"\n#include <cstdio>\n#include <new>\n#include <cstdlib>\n#include <cmath>\n#include <cstring>\n'
    if app=='CHT' and bits==32:code+='using namespace ugkwpCudaFp32;\n'
    code+='using Real=GPU_OPERATOR_REAL;\n'
    names=['solidEpsFromMomentDevice','solidPressureFromMomentsDevice','granularCollisionTauFromCellDevice','riemannFacePrimitiveForGradient']
    if app!='gasUGKP':names+=['atomicAddParticleWallEnergyByFace']
    else:names+=['preparePressureProjectionCell']
    for name in names:
        code+=frozen[app+':'+name].replace(name,'baseline_'+name)+'\n'
    code+=frozen['drag'].replace('UGKWP_GPU_DRAG_ALGEBRA_CUH','BASELINE_GPU_DRAG_ALGEBRA_CUH').replace('ugkwpGpuDragAlgebra','baselineDrag')+'\n'
    if app!='gasUGKP':
        cold=frozen['cold'+str(bits)].replace('advanceColdWall1DThermalGroup','baselineColdWall')
        if bits==32:cold='template<bool StabilizePcr=false>\n'+cold
        code+=cold+'\n'
        code+=frozen['thermalPressureCell'].replace('applyCollisionalPressureProjectionCellAtomicKernel','baselinePressureCell')+'\n'
        code+=frozen['thermalPressureParticles'].replace('applyCollisionalPressureProjectionParticlesAtomicKernel','baselinePressureParticles')+'\n'
    raw=(source/'GpuResidentStrict.cu').read_text();a=raw.index('struct DeviceState');b=raw.index('\n};',a)
    fields=re.findall(r'^\s*((?:unsigned\s+)?(?:long long|char)|double|float|int|GpuReal|GpuTime|GpuWallEnergy)\*\s+(\w+)\s*=',raw[a:b],re.M)
    fields=[(ty,n) for ty,n in fields if n not in ['diagnosticPreTransportParticleCount','sourceInjectedCount'] and (bits==32 or not n.startswith('flatPressure'))]
    code+=r'''
template<class T>void mem(T*&p){if(cudaMallocManaged(&p,1024*sizeof(T))!=cudaSuccess)abort();for(int i=0;i<1024;i++)::new(static_cast<void*>(p+i)) T{};}
void sync(){auto e=cudaDeviceSynchronize();if(e!=cudaSuccess){printf("FAIL CUDA %s\n",cudaGetErrorString(e));exit(10);}}
void ck(bool x,const char*m){if(!x){printf("FAIL %s\n",m);exit(11);}}
void equal(double a,double b,const char*m){if(std::memcmp(&a,&b,sizeof(double))){printf("FAIL bits %s %.17g != %.17g\n",m,a,b);exit(12);}}
__global__ void primitives(DeviceState*s,double*values){int i=threadIdx.x+blockDim.x*blockIdx.x;if(i>=512)return;
 int c=i%18-1; int f=i%40;int fc=s->faceOwner[f];if(f<s->nInternalFaces && (i&1))fc=s->faceNeighbour[f];
 Real cur[]={solidEpsFromMomentDevice(*s,c),solidPressureFromMomentsDevice(*s,c),granularCollisionTauFromCellDevice(*s,c)};
 Real old[]={baseline_solidEpsFromMomentDevice(*s,c),baseline_solidPressureFromMomentsDevice(*s,c),baseline_granularCollisionTauFromCellDevice(*s,c)};
 auto face=riemannFacePrimitiveForGradient(*s,fc,f);auto bf=baseline_riemannFacePrimitiveForGradient(*s,fc,f);
 Real faces[]={face.rho,face.ux,face.uy,face.uz,face.p,face.T},refs[]={bf.rho,bf.ux,bf.uy,bf.uz,bf.p,bf.T};
 for(int j=0;j<3;j++){values[32*i+2*j]=cur[j];values[32*i+2*j+1]=old[j];}
 for(int j=0;j<6;j++){values[32*i+6+2*j]=faces[j];values[32*i+7+2*j]=refs[j];}
}
__global__ void dragRates(double*values){int i=threadIdx.x+blockDim.x*blockIdx.x;if(i>=512)return;
 const Real slips[]={Real(0),Real(1e-32),Real(1e-12),Real(.1),Real(999.999),Real(1000),Real(1000.001),Real(-1)};
 const Real densities[]={Real(0),Real(1),Real(-1),Real(1e-30)};
 const Real diameters[]={Real(1),Real(1e-4),Real(1e-31),Real(0)};
 const Real alphas[]={Real(.799999),Real(.8),Real(.800001),Real(1)};
 Real slip=slips[i%8],rho=densities[(i/8)%4],d=diameters[(i/32)%4],alpha=alphas[(i/128)%4],mu=Real(1),rs=Real(3000),reg=Real(1e-30),rr=Real(1e-12);
 Real cur[]={ugkwpGpuDragAlgebra::gasUgkpSchillerNaumannInverseResponseTime(rho,mu,rs,d,slip,reg),ugkwpGpuDragAlgebra::gasUgkpGidaspowInverseResponseTime(rho,mu,alpha,rs,d,slip,reg,rr),ugkwpGpuDragAlgebra::fshChtSchillerNaumannInverseRelaxationTime(rho,mu,rs,d,slip),ugkwpGpuDragAlgebra::fshChtGidaspowInverseRelaxationTime(rho,alpha,mu,rs,d,slip,rr)};
 Real old[]={baselineDrag::gasUgkpSchillerNaumannInverseResponseTime(rho,mu,rs,d,slip,reg),baselineDrag::gasUgkpGidaspowInverseResponseTime(rho,mu,alpha,rs,d,slip,reg,rr),baselineDrag::fshChtSchillerNaumannInverseRelaxationTime(rho,mu,rs,d,slip),baselineDrag::fshChtGidaspowInverseRelaxationTime(rho,alpha,mu,rs,d,slip,rr)};
 for(int j=0;j<4;j++){values[8*i+2*j]=cur[j];values[8*i+2*j+1]=old[j];}
}
EXTRA_KERNELS
int main(){setvbuf(stdout,nullptr,_IONBF,0);DeviceState*s,*reference;mem(s);mem(reference);
ALLOC
s->nCells=16;s->nFaces=40;s->nInternalFaces=8;s->rhoSolid=3000;s->epsSMin=1e-12;s->thetaMin=1e-12;s->rhoMin=1e-12;s->Rgas=1;s->TgasMin=.001;s->particleDiameterFallback=1e-4;s->collisionalPressureEnabled=1;s->collisionalRestitution=.9;
for(int c=0;c<16;c++){
 s->rho[c]=1+.125*c;s->Ux[c]=2+.25*c;s->Uy[c]=-.5;s->Uz[c]=.25;s->p[c]=2+.25*c;s->Tgas[c]=300+c;
 s->momRhoP[c]=(c==0?0:100+.125*c);s->momRhoUPx[c]=.25*s->momRhoP[c];s->momRhoUPy[c]=-.125*s->momRhoP[c];s->momRhoUPz[c]=0;s->momRhoEP[c]=(c==2?1e-30:5)*s->momRhoP[c];s->momRhoPD[c]=1e-4*s->momRhoP[c];
 s->V[c]=1+.0625*c;s->cellPlaneStart[c]=3*c;s->cellPlaneCount[c]=3;s->cellFaceId[3*c]=c%8;s->cellFaceId[3*c+1]=-1;s->cellFaceId[3*c+2]=(c+2)%8;
}
s->momRhoP[3]=NAN;s->momRhoPD[4]=NAN;s->momRhoEP[5]=INFINITY;
for(int f=0;f<40;f++){
 s->faceOwner[f]=f%16;s->faceNeighbour[f]=(f+1)%16;s->facePeriodicPair[f]=f>=8&&f<16?(f^1):-1;s->faceWeight[f]=.25*(f%5);s->riemannBoundaryKind[f]=f%5;s->riemannBoundaryUFix[f]=f%2;s->riemannBoundaryTFix[f]=f%2;
 s->magSf[f]=1;s->Sfx[f]=1;s->riemannBoundaryUx[f]=.5;s->riemannBoundaryT[f]=500;s->solidPressurePhiMomX[f]=.125*(f-4);s->solidPressurePhiEnergy[f]=.25*(f-2);
}
double*values; if(cudaMallocManaged(&values,512*32*sizeof(double))!=cudaSuccess)abort();
primitives<<<4,128>>>(s,values);sync();for(int i=0;i<512;i++)for(int j=0;j<9;j++)equal(values[32*i+2*j],values[32*i+2*j+1],"moment/face primitive");
puts("PASS bitwise moments and face states: invalid cells, nonfinite inputs, internal faces from both sides, periodic owners and wall kinds");
dragRates<<<4,128>>>(values);sync();for(int i=0;i<512*4;i++)equal(values[2*i],values[2*i+1],"drag policy rate");
puts("PASS 2048 bitwise drag rates including zero slip, Re/alpha thresholds and denominator bounds");
// Pressure publication uses a finite positive fixture; primitive tests above
// separately cover nonfinite moment sanitization.
for(int c=0;c<16;c++){s->momRhoP[c]=100;s->momRhoEP[c]=500;s->momRhoUPx[c]=25;s->momRhoUPy[c]=-12.5;s->momRhoUPz[c]=0;}
PRESSURE_TEST
WALL_TEST
COLD_TEST
puts("PASS all frozen-implementation differential checks");return 0;}
'''
    allocations='\n'.join(f'mem(s->{n});mem(reference->{n});' for _,n in fields)
    code=code.replace('ALLOC',allocations)
    copy='{DeviceState copy=*s;\n'+''.join(f'copy.{n}=reference->{n};std::memcpy(copy.{n},s->{n},1024*sizeof(*s->{n}));\n' for _,n in fields)+'*reference=copy;}\n'
    common_compare='\n'.join(f'for(int c=0;c<16;c++)equal(s->{n}[c],reference->{n}[c],"pressure {n}");' for n in 'pressureDeltaMomX pressureDeltaMomY pressureDeltaMomZ pressureDeltaEnergy momRhoUPx momRhoUPy momRhoUPz momRhoEP rhoUsx rhoUsy rhoUsz rhoEs Usx Usy Usz theta'.split())
    extra=''
    if app=='gasUGKP':
        extra+=r'''
__global__ void pressurePair(DeviceState*s,DeviceState*r,double*out,double dt){int c=threadIdx.x;if(c>=16)return;auto a=preparePressureProjectionCell(*s,c,dt);auto b=baseline_preparePressureProjectionCell(*r,c,dt);
 const double x[]={a.ux0,a.uy0,a.uz0,a.ux1,a.uy1,a.uz1,a.theta1,a.thermalScale,a.thetaScale,double(a.resolved)},y[]={b.ux0,b.uy0,b.uz0,b.ux1,b.uy1,b.uz1,b.theta1,b.thermalScale,b.thetaScale,double(b.resolved)};
 for(int j=0;j<10;j++){out[20*c+2*j]=x[j];out[20*c+2*j+1]=y[j];}}
'''
        pressure=copy+'for(int mode:{0,1}){s->csrCellLocalPathEnabled=mode;reference->csrCellLocalPathEnabled=mode;pressurePair<<<1,32>>>(s,reference,values,.000100000000000003);sync();for(int i=0;i<160;i++)equal(values[2*i],values[2*i+1],"pressure kinematics");'+common_compare+'}\n'
    else:
        pressure=copy+'applyCollisionalPressureProjectionCellAtomicKernel<<<1,32>>>(s,.000100000000000003);baselinePressureCell<<<1,32>>>(reference,.000100000000000003);sync();'+common_compare+'\n'
        pressure+=r'''
s->particleCapacity=64;*s->particleCountDevice=64;
for(int i=0;i<64;i++){
 s->pStatus[i]=1;s->pCellId[i]=i%16;s->pm[i]=.5;s->pd[i]=1e-4;s->pT[i]=1000;s->pTheta[i]=.25;
 s->pux[i]=.25+(i%3)*.125;s->puy[i]=-.125;s->puz[i]=0;s->puxOld[i]=2;s->puyOld[i]=-1;s->puzOld[i]=.5;
 s->pStuck[i]=(i/16==0)?Foam::gpuThermal::particleWallMobile:(i/16==1)?Foam::gpuThermal::particleWallTransientRebound:Foam::gpuThermal::particleWallDeposited;
}
COPY_THERMAL
applyCollisionalPressureProjectionParticlesAtomicKernel<true><<<2,32>>>(s);baselinePressureParticles<true><<<2,32>>>(reference);sync();
for(int i=0;i<64;i++){
 equal(s->pux[i],reference->pux[i],"unsorted constrained particle X");equal(s->puy[i],reference->puy[i],"unsorted constrained particle Y");equal(s->puz[i],reference->puz[i],"unsorted constrained particle Z");equal(s->pTheta[i],reference->pTheta[i],"unsorted constrained theta");
 equal(s->puxOld[i],reference->puxOld[i],"constraint old velocity");
 if(s->pStuck[i])ck(s->pux[i]==0&&s->puy[i]==0&&s->puz[i]==0,"wall constraint velocity");
 if(s->pStuck[i]==Foam::gpuThermal::particleWallDeposited)ck(s->puxOld[i]==0,"deposited old velocity reset");
}
for(int j=0;j<7*16;j++){double a=s->pressureParticleMoments[j],b=reference->pressureParticleMoments[j];ck(std::isfinite(a)&&fabs(a-b)<=PRESSURE_TOL*fmax(1.,fabs(b)),"atomic constrained moments");}
puts("PASS bitwise unsorted particle update with mobile, transient and deposited wall constraints");
'''.replace('COPY_THERMAL',copy).replace('PRESSURE_TOL','2e-6' if bits==32 else '2e-12')
    pressure+='puts("PASS bitwise pressure face deltas and published cell moments");'
    code=code.replace('PRESSURE_TEST',pressure)
    wall=cold=''
    if app!='gasUGKP':
        extra+=r'''
__global__ void wallChannels(DeviceState*s,DeviceState*r){int i=threadIdx.x+blockDim.x*blockIdx.x;if(i>=129)return;int f=i%3;Real e=(i%7==0)?Real(0):Real((i%2?-1:1)*(1+i%5)*.125);auto a=i%2?s->particleWallReflectedEnergy:s->particleWallDepositedEnergy;auto b=i%2?r->particleWallReflectedEnergy:r->particleWallDepositedEnergy;atomicAddParticleWallEnergyByFace(*s,a,f,e);baseline_atomicAddParticleWallEnergyByFace(*r,b,f,e);}
__global__ void coldPair(DeviceState*s,DeviceState*r,double*out,int stable,double dt,double gasConductance){int lane=threadIdx.x;Real ma=0,fa=0,ea=0,mb=0,fb=0,eb=0;bool failA=false,failB=false;
 const Real volume=Real(1e-12),mass=Real(3.2e-9),area=Real(2.2e-8),contact=Real(1.7e-8);
 bool a,b;
 COLD_CALLS
 if(lane==0){out[0]=a;out[1]=b;out[2]=ma;out[3]=mb;out[4]=fa;out[5]=fb;out[6]=ea;out[7]=eb;out[8]=failA;out[9]=failB;}}
'''
        callargs='volume,mass,area,contact,GpuTime(8e-6),Real(.42),GpuTime(dt),Real(300),Real(13000),Real(.1),Real(3200),Real(gasConductance)'
        calls=f'a=advanceColdWall1DThermalGroup(*s,0,lane,0xffu,{callargs},ma,fa,ea);b=baselineColdWall(*r,0,lane,0xffu,{callargs},mb,fb,eb);'
        if bits==32:
            calls=f'if(stable){{a=advanceColdWall1DThermalGroup<true>(*s,0,lane,0xffu,{callargs},ma,fa,ea,&failA);b=baselineColdWall<true>(*r,0,lane,0xffu,{callargs},mb,fb,eb,&failB);}}else{{a=advanceColdWall1DThermalGroup<false>(*s,0,lane,0xffu,{callargs},ma,fa,ea,&failA);b=baselineColdWall<false>(*r,0,lane,0xffu,{callargs},mb,fb,eb,&failB);}}'
        extra=extra.replace('COLD_CALLS',calls)
        wall='for(int block:{32,64,128,256}){for(int f=0;f<3;f++)s->particleWallReflectedEnergy[f]=s->particleWallDepositedEnergy[f]=reference->particleWallReflectedEnergy[f]=reference->particleWallDepositedEnergy[f]=0;wallChannels<<<(129+block-1)/block,block>>>(s,reference);sync();for(int f=0;f<3;f++){equal(s->particleWallReflectedEnergy[f],reference->particleWallReflectedEnergy[f],"wall reflected");equal(s->particleWallDepositedEnergy[f],reference->particleWallDepositedEnergy[f],"wall deposited");}}puts("PASS bitwise signed wall ledger with partial warps, multiple faces/channels and block sizes");'
        cold=r'''
s->coldWallSolidificationEnabled=reference->coldWallSolidificationEnabled=1;
s->coldWallSolidificationParameters=reference->coldWallSolidificationParameters={2327.,20.,1.16e6,3990.,1273.,5.9,.25,0.,1,4};
double largestColdResidual=0,largestColdBound=0;
for(int closed:{0,1})for(int stable=0;stable<STABLE_MODES;stable++)for(int transient:{0,1}){
 s->coldWallSolidificationParameters.wallTransientResistance=reference->coldWallSolidificationParameters.wallTransientResistance=transient;
 for(int i=0;i<8;i++){s->pColdNodeSpecificEnthalpy[i]=reference->pColdNodeSpecificEnthalpy[i]=float(Foam::gpuThermal::coldWallSpecificEnthalpyJkg(3500.,s->coldWallSolidificationParameters));s->pColdRingSolidMass[i]=reference->pColdRingSolidMass[i]=0;}
 s->pColdContactAge[0]=reference->pColdContactAge[0]=0;s->pColdFrozenArea[0]=reference->pColdFrozenArea[0]=0;
 for(int step=0;step<64;step++){
  double before=0;for(int i=0;i<8;i++)before+=double(Real(3.2e-9)/Real(8))*s->pColdNodeSpecificEnthalpy[i];
  coldPair<<<1,8>>>(s,reference,values,stable,7.5e-8,closed?0:2e-4);sync();for(int j=0;j<5;j++)equal(values[2*j],values[2*j+1],"cold-wall output");ck(values[0]==1,"cold-wall valid step");
  double after=0,storageBound=0;
  for(int i=0;i<8;i++){
    equal(s->pColdNodeSpecificEnthalpy[i],reference->pColdNodeSpecificEnthalpy[i],"cold-wall node enthalpy");equal(s->pColdRingSolidMass[i],reference->pColdRingSolidMass[i],"cold-wall ring mass");
    float node=s->pColdNodeSpecificEnthalpy[i];double nodeMass=double(Real(3.2e-9)/Real(8));after+=nodeMass*node;
    storageBound+=.5*nodeMass*fabs(double(std::nextafter(node,INFINITY))-node);
  }
  equal(s->pColdContactAge[0],reference->pColdContactAge[0],"cold-wall time64");
  if(closed){double residual=after-before+values[6];double bound=storageBound+64*GPU_REAL_EPSILON*(fabs(before)+fabs(after)+fabs(values[6]));largestColdResidual=fmax(largestColdResidual,fabs(residual));largestColdBound=fmax(largestColdBound,bound);ck(fabs(residual)<=bound,"closed cold-wall particle enthalpy + wall heat conservation within storage rounding");}
 }
 float saved=s->pColdNodeSpecificEnthalpy[0];double age=s->pColdContactAge[0];coldPair<<<1,8>>>(s,reference,values,stable,-1,0);sync();ck(values[0]==0&&values[1]==0,"invalid cold-wall step rejected");equal(s->pColdNodeSpecificEnthalpy[0],saved,"rejected enthalpy unchanged");equal(s->pColdContactAge[0],age,"rejected age unchanged");
}
printf("COLD_ENERGY maximum_abs_residual=%.17g maximum_rounding_bound=%.17g\n",largestColdResidual,largestColdBound);
puts("PASS bitwise cold-wall temperature, wall heat, enthalpy and ring mass across precision policies");
'''.replace('STABLE_MODES','2' if bits==32 else '1')
    code=code.replace('EXTRA_KERNELS',extra).replace('WALL_TEST',wall).replace('COLD_TEST',cold)
    code=code.replace('void sync()','void deviceSync()').replace('sync();','deviceSync();')
    cu=out/'differential.cu';cu.write_text(code);exe=out/'differential'
    cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','-arch=sm_89','--fmad='+('false' if app=='CHT' else 'true'),'-DUGKWP_GPU_REAL_BITS='+str(bits),'-I'+str(source),'-I'+str(root/'common'),'-I'+str(root/'common/wall'),'-I'+str(root/'applications'/app/'gpu'),str(cu),'-o',str(exe)]
    if app=='CHT':cmd.append(str(source/'GpuWallEnergy64.cu'))
    soft,hard=resource.getrlimit(resource.RLIMIT_STACK)
    requested=64*1024*1024
    resource.setrlimit(resource.RLIMIT_STACK,(requested if hard==resource.RLIM_INFINITY else min(requested,hard),hard))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    with (out/'build.log').open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:print((out/'build.log').read_text()[-6000:]);return result.returncode
    result=subprocess.run([str(exe)],capture_output=True,text=True);(out/'run.log').write_text(result.stdout+result.stderr)
    print(app,bits,result.stdout+result.stderr,flush=True);return result.returncode

if __name__=='__main__':raise SystemExit(main())
