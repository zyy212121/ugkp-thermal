#!/usr/bin/env python3
"""Paired kernel regression measurements for operator extraction.

These synthetic kernel loads do not substitute for whole-application speedups.
Both versions use the same mesh, particles, inputs and launch geometry. No
case-specific parameter is written to production code.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,resource,statistics,subprocess

def build(root,out,app,bits):
    source=root/'applications'/app/('gpu' if app=='CHT' else 'private_backend')
    raw=(source/'GpuResidentStrict.cu').read_text().replace('#include "GpuAutomaticCsrScheduleFields.inl"', (root/"common/GpuAutomaticCsrScheduleFields.inl").read_text() if (root/"common/GpuAutomaticCsrScheduleFields.inl").is_file() else "");a=raw.index('struct DeviceState');b=raw.index('\n};',a)
    fields=re.findall(r'^\s*((?:unsigned\s+)?(?:long long|char)|double|float|int|GpuReal|GpuTime|GpuWallEnergy)\*\s+(\w+)\s*=',raw[a:b],re.M)
    fields=[(ty,n) for ty,n in fields if n not in ['diagnosticPreTransportParticleCount','sourceInjectedCount'] and (bits==32 or not n.startswith('flatPressure'))]
    code='#include "GpuResidentStrict.cu"\n#include <cstdio>\n#include <cstdlib>\n#include <new>\n#include <cmath>\n'
    if app=='CHT' and bits==32:code+='using namespace ugkwpCudaFp32;\n'
    code+=r'''
template<class T>void mem(T*&p,int n){if(cudaMallocManaged(&p,n*sizeof(T))!=cudaSuccess)abort();for(int i=0;i<n;i++)::new(static_cast<void*>(p+i)) T{};}
void check(cudaError_t e){if(e!=cudaSuccess){printf("FAIL CUDA %s\n",cudaGetErrorString(e));exit(10);}}
int main(int argc,char**argv){int cells=argc>1?atoi(argv[1]):512;bool graphTiming=argc>2;int particles=128*cells;int faces=cells+1;int capacity=particles+32;
DeviceState*s;mem(s,1);ALLOC
s->deviceState=s;s->nCells=cells;s->nFaces=faces;s->nInternalFaces=cells-1;s->nCellPlanes=2*cells;s->particleCapacity=particles;s->particleWorkGrid=(particles+127)/128;s->particleBlockThreads=128;*s->particleCountDevice=particles;
s->rhoSolid=3000;s->epsSMin=1e-12;s->thetaMin=1e-12;s->rhoMin=1e-12;s->Rgas=1;s->TgasMin=1;s->gammaGas=1.4;s->gasMu=1e-5;s->particleDiameterFallback=1e-4;s->TpMin=1;s->TpMax=5000;s->csrCellLocalPathEnabled=0;
for(int c=0;c<cells;c++){
 s->rho[c]=1+.01*(c%7);s->p[c]=400*s->rho[c];s->Tgas[c]=400;s->Ux[c]=.25;s->Uy[c]=-.125;s->Uz[c]=0;s->V[c]=1;s->cellLength[c]=1;s->Cx[c]=c;
 s->momRhoP[c]=64;s->momRhoUPx[c]=16;s->momRhoUPy[c]=-8;s->momRhoUPz[c]=0;s->momRhoEP[c]=100;s->cellPlaneStart[c]=2*c;s->cellPlaneCount[c]=2;s->cellFaceId[2*c]=c?c-1:cells-1;s->cellFaceId[2*c+1]=c+1<cells?c:cells;
}
for(int f=0;f<faces;f++){
 s->faceOwner[f]=f<cells-1?f:(f==cells-1?0:cells-1);s->faceNeighbour[f]=f<cells-1?f+1:-1;s->facePeriodicPair[f]=-1;s->faceWeight[f]=.5;s->Sfx[f]=f==cells-1?-1:1;s->magSf[f]=1;s->riemannBoundaryKind[f]=3;s->faceCx[f]=f<cells-1?f+.5:(f==cells-1?-.5:cells-.5);
 s->solidPressurePhiMomX[f]=.125;s->solidPressurePhiMomY[f]=-.0625;s->solidPressurePhiMomZ[f]=0;s->solidPressurePhiEnergy[f]=0;
}
for(int i=0;i<particles;i++){
 s->pStatus[i]=1;s->pCellId[i]=i/128;s->pm[i]=.5;s->pT[i]=3500;s->pd[i]=1e-4;s->pux[i]=.25+(i%3)*.125;s->puy[i]=-.125;s->puz[i]=0;s->pTheta[i]=.25;THERMAL
}
const int block=128,cellGrid=(cells+block-1)/block,particleGrid=(particles+block-1)/block;
cudaStream_t stream=nullptr;if(graphTiming)check(cudaStreamCreate(&stream));
PRESSURE_PREPARE
check(cudaDeviceSynchronize());
cudaEvent_t start,stop;check(cudaEventCreate(&start));check(cudaEventCreate(&stop));
for(int stage=0;stage<5;stage++){
 auto launch=[&](){switch(stage){
 case 0:computeGasPrimitiveGradientsKernel<<<cellGrid,block,0,stream>>>(s);break;
 case 1:computeGasGradientLimiterKernel<<<cellGrid,block,0,stream>>>(s);break;
 case 2:PRESSURE_CELL;break;
 case 3:PRESSURE_PARTICLES;break;
 case 4:computeGasDiffusionNumberKernel<<<cellGrid,block,0,stream>>>(s,1e-6,1.);break;
 }};
 for(int i=0;i<20;i++)launch();check(cudaDeviceSynchronize());
 cudaGraph_t graph=nullptr;cudaGraphExec_t graphExec=nullptr;
 if(graphTiming){
  check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));for(int i=0;i<64;i++)launch();check(cudaStreamEndCapture(stream,&graph));
  check(cudaGraphInstantiate(&graphExec,graph,nullptr,nullptr,0));
  for(int i=0;i<3;i++)check(cudaGraphLaunch(graphExec,stream));check(cudaDeviceSynchronize());
 }
 check(cudaEventRecord(start,stream));if(graphTiming)check(cudaGraphLaunch(graphExec,stream));else for(int i=0;i<64;i++)launch();check(cudaEventRecord(stop,stream));check(cudaEventSynchronize(stop));float ms=0;check(cudaEventElapsedTime(&ms,start,stop));
 const char*names[]={"gas_gradients","gas_limiter","pressure_cell","pressure_particles","gas_diffusion"};printf("TIME %s %.9g\n",names[stage],ms/64);
 if(graphTiming){check(cudaGraphExecDestroy(graphExec));check(cudaGraphDestroy(graph));}
}
check(cudaEventDestroy(start));check(cudaEventDestroy(stop));if(graphTiming)check(cudaStreamDestroy(stream));return 0;}
'''
    alloc='\n'.join(f'mem(s->{n},capacity);' for _,n in fields)
    if app=='gasUGKP':alloc+='\nmem(s->pressureProjectionCache,cells);'
    code=code.replace('ALLOC',alloc).replace('THERMAL','s->pStuck[i]=(i%4==0)?Foam::gpuThermal::particleWallTransientRebound:Foam::gpuThermal::particleWallMobile;' if app!='gasUGKP' else '')
    if app=='gasUGKP':
        prepare='preparePressureProjectionCacheKernel<<<cellGrid,block,0,stream>>>(s,1e-6);'
        cell='preparePressureProjectionCacheKernel<<<cellGrid,block,0,stream>>>(s,1e-6)'
        particle='applyCachedPressureProjectionParticlesKernel<<<particleGrid,block,0,stream>>>(s)'
    else:
        prepare='applyCollisionalPressureProjectionCellAtomicKernel<<<cellGrid,block,0,stream>>>(s,1e-6);'
        cell='applyCollisionalPressureProjectionCellAtomicKernel<<<cellGrid,block,0,stream>>>(s,1e-6)'
        particle='applyCollisionalPressureProjectionParticlesAtomicKernel<true><<<particleGrid,block,0,stream>>>(s)'
    code=code.replace('PRESSURE_PREPARE',prepare).replace('PRESSURE_CELL',cell).replace('PRESSURE_PARTICLES',particle)
    out.mkdir(parents=True,exist_ok=True);cu=out/'probe.cu';cu.write_text(code);exe=out/'probe'
    digest=hashlib.sha256(code.encode())
    digest.update(raw.encode())
    for header in sorted((root/'common').rglob('*')):
        if header.is_file() and header.suffix in ('.H','.cuh','.inl'):digest.update(header.read_bytes())
    signature=digest.hexdigest();stamp=out/'source.sha256'
    if exe.is_file() and stamp.is_file() and stamp.read_text()==signature:return exe
    cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O3','-arch=sm_89','--fmad='+('false' if app=='CHT' else 'true'),'-DUGKWP_GPU_REAL_BITS='+str(bits),'-I'+str(source),'-I'+str(root/'common'),'-I'+str(root/'applications'/app/'gpu'),str(cu),'-o',str(exe)]
    if app=='CHT':cmd.append(str(source/'GpuWallEnergy64.cu'))
    with (out/'build.log').open('w') as log:q=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if q.returncode:raise RuntimeError((out/'build.log').read_text()[-5000:])
    stamp.write_text(signature)
    return exe

def run(exe,cells,log,graph=False):
    q=subprocess.run([str(exe),str(cells)]+(['graph'] if graph else []),capture_output=True,text=True,check=True)
    log.write_text(q.stdout+q.stderr)
    return {parts[1]:float(parts[2]) for parts in (line.split() for line in q.stdout.splitlines()) if len(parts)==3 and parts[0]=='TIME'}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',required=True,type=Path);p.add_argument('--after',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--repeats',type=int,default=10);p.add_argument('--loads',type=int,nargs=2,default=[512,1024]);p.add_argument('--build-only',action='store_true');p.add_argument('--graph',action='store_true',help='Replay captured kernels to reduce CPU launch starvation');args=p.parse_args()
    soft,hard=resource.getrlimit(resource.RLIMIT_STACK)
    requested=64*1024*1024
    resource.setrlimit(resource.RLIMIT_STACK,(requested if hard==resource.RLIM_INFINITY else min(requested,hard),hard))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True);rows=[];summary=[]
    for app,bits in [('gasUGKP',64),('FSH',64),('CHT',64),('CHT',32)]:
        binaries={version:build(getattr(args,version).resolve(),output/(app+str(bits))/version,app,bits) for version in ('before','after')}
        print('built',app,bits,flush=True)
        if args.build_only:continue
        for cells in args.loads:
            for rep in range(args.repeats):
                values={}
                for version in (('before','after') if rep%2==0 else ('after','before')):
                    values[version]=run(binaries[version],cells,output/(app+str(bits))/f'{cells}-{rep}-{version}.log',args.graph)
                for name in values['before']:
                    old,new=values['before'][name],values['after'][name]
                    rows.append(dict(app=app,bits=bits,cells=cells,particles=128*cells,repeat=rep,kernel=name,timing='graph' if args.graph else 'launch',before_ms=old,after_ms=new,speedup=old/new))
                (output/'samples.json').write_text(json.dumps(rows,indent=2)+'\n')
            for name in values['before']:
                samples=[r for r in rows if r['app']==app and r['bits']==bits and r['cells']==cells and r['kernel']==name]
                ratios=[s['speedup'] for s in samples]
                result=dict(app=app,bits=bits,cells=cells,particles=128*cells,kernel=name,pairs=len(samples),median_speedup=statistics.median(ratios),minimum=min(ratios),maximum=max(ratios))
                summary.append(result);print(json.dumps(result),flush=True)
            (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return 0

if __name__=='__main__':raise SystemExit(main())
