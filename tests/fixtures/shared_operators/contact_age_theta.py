#!/usr/bin/env python3
"""Build/run the native correction-path contact-age invariant witness.

Usage: python3 contact_age_theta.py SOURCE OUT FSH 64 [--prepare-only]
       python3 contact_age_theta.py SOURCE OUT CHT 32|64 [--prepare-only]
OUT must be outside SOURCE and may be reused. Generated files are replaced.
Exit 2 means mobile theta or detached stored contact age was not reset.
This tests local native kernels, not end-to-end solver acceptance.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import sys


def main():
    if len(sys.argv) < 5:
        raise SystemExit(__doc__)
    root, out = map(lambda x: Path(x).resolve(), sys.argv[1:3])
    app, bits = sys.argv[3], int(sys.argv[4])
    if app not in ('FSH', 'CHT') or bits not in (32, 64) or (app == 'FSH' and bits != 64):
        raise SystemExit('Supported states: FSH64, CHT64, CHT32')
    if out == root or root in out.parents:
        raise SystemExit('Output must be separate from the source checkout')
    out.mkdir(parents=True, exist_ok=True)
    source = root / 'applications' / app / ('gpu' if app == 'CHT' else 'private_backend') / 'GpuResidentStrict.cu'
    text = source.read_text()
    a = text.index('struct DeviceState')
    b = text.index('\n};', a)
    fields = re.findall(r'^\s*((?:unsigned\s+)?(?:long long|char)|double|float|int|GpuReal|GpuTime|GpuWallEnergy)\*\s+(\w+)\s*=', text[a:b], re.M)
    fields = [(t, n) for t, n in fields if n not in ('diagnosticPreTransportParticleCount', 'sourceInjectedCount') and (bits == 32 or not n.startswith('flatPressure'))]
    code = '#include "GpuResidentStrict.cu"\n#include <cstdio>\n#include <cstdlib>\n#include <new>\n#include <cmath>\n'
    if app == 'CHT' and bits == 32:
        code += 'using namespace ugkwpCudaFp32;\n'
    code += r'''
template<class T> void alloc(T*& p,int n) {
    if(cudaMallocManaged(&p,n*sizeof(T))!=cudaSuccess) std::abort();
    for(int i=0;i<n;++i) ::new(static_cast<void*>(p+i)) T{};
}
void deviceSynchronizeChecked() {
    cudaError_t e=cudaDeviceSynchronize();
    if(e!=cudaSuccess){std::printf("CUDA_ERROR %s\n",cudaGetErrorString(e));std::exit(3);}
}
__global__ void observe(DeviceState* s,double* out) {
    if(threadIdx.x || blockIdx.x) return;
    out[0]=particleMomentThetaDevice(*s,0);
    out[1]=double(s->pm[0])*.5*(double(s->pux[0])*s->pux[0]+double(s->puy[0])*s->puy[0]+double(s->puz[0])*s->puz[0]);
    out[2]=out[1]+1.5*double(s->pm[0])*out[0];
    out[3]=1.5*double(s->pm[0])*out[0];
}
int main() {
    setvbuf(stdout,nullptr,_IONBF,0);
    DeviceState* s; alloc(s,1);
'''
    code += '\n'.join(f'    alloc(s->{name},16);' for _, name in fields)
    code += r'''
    s->deviceState=s;s->particleCapacity=2;*s->particleCountDevice=2;
    s->nCells=1;s->nFaces=1;s->nInternalFaces=0;
    s->rhoMin=1e-12;s->rhoSolid=3000;s->gasMu=1e-5;
    s->particleDiameterFallback=1e-4;s->TpMin=1;s->TpMax=5000;
    s->TgasMin=1;s->thetaMin=1e-12;s->epsSMin=1e-12;
    s->dragModelId=0;s->solveParticleTemperature=0;s->particleGasHeatTransferModelId=0;
    s->particleWallHeatTransferEnabled=0;s->coldWallSolidificationEnabled=0;s->coldWall2DEnabled=0;
    s->particleWallAdhesionEnergyScale=1;s->particleWallContactAngleCosine=0;
    s->particleStuckModelConfigured=1;s->particleWallMaximumCoverage=1;
    s->coldWallSolidificationParameters={2327,20,1e6,3900,1000,30,.1,1e-5,1,4};
    s->magSf[0]=1;s->Sfx[0]=0;s->Sfy[0]=1;s->Sfz[0]=0;
    s->gasBoundaryUx[0]=s->gasBoundaryUy[0]=s->gasBoundaryUz[0]=0;
    s->particleWallContactAreaScale[0]=1;
    s->couplingRhoOld[0]=1;s->couplingUxOld[0]=s->couplingUyOld[0]=s->couplingUzOld[0]=0;
    s->couplingTgasOld[0]=2800;
    double* result;alloc(result,4);
    bool foundLeak=false;
    // Feed a consistent post-sampling state into the actual native correction kernels.
    // Two equal masses, sampled velocities +/-speed, zero pool mean, scale=1.
    // Cases 0,1 are the two transient states; 2 is a long-deposit negative control;
    // case 3 is transient damage below the detachment threshold (age must survive).
    for(int scenario=0;scenario<4;++scenario) {
        const bool deposited=scenario==2;
        const bool shouldDetach=scenario!=3;
        s->coldWallSolidificationEnabled=(scenario==1 || deposited)?1:0;
        s->particleStuckCandidateMask[0]=scenario==1 || deposited
          ? Foam::gpuThermal::particleWallSolidifyingDeposition
          : Foam::gpuThermal::particleWallReboundContact;
        for(int i=0;i<2;++i) {
            s->pStatus[i]=2;s->pCellId[i]=0;s->pm[i]=1e-9;s->pd[i]=1e-4;s->pT[i]=2800;
            s->pTheta[i]=0;s->puy[i]=s->puz[i]=0;
            s->pStuck[i]=i==0 ? (deposited ? Foam::gpuThermal::particleWallDeposited
              : scenario==1 ? Foam::gpuThermal::particleWallTransientDeposit
                            : Foam::gpuThermal::particleWallTransientRebound)
              : Foam::gpuThermal::particleWallMobile;
            s->pStuckFaceId[i]=i==0?0:-1;
            s->pContactDuration[i]=deposited?0:.001;s->pContactPeakFraction[i]=deposited?0:.5;
            s->pColdFrozenArea[i]=s->pCold2DFrozenArea[i]=0;
            s->puxOld[i]=s->puyOld[i]=s->puzOld[i]=0;
            s->pColdContactAge[i]=0;
            for(int j=0;j<8;++j){s->pColdNodeSpecificEnthalpy[8*i+j]=Foam::gpuThermal::coldWallSpecificEnthalpyJkg(2800,s->coldWallSolidificationParameters);s->pColdRingSolidMass[8*i+j]=0;}
        }
        const double age=deposited?0:.0002;
        AGE_WRITE
        auto cap=Foam::gpuThermal::evaluateCapillaryDetachmentState(s->pT[0],s->pd[0],s->particleWallAdhesionEnergyScale,s->particleWallContactAngleCosine);
        if(!cap.valid) return 4;
        s->pContactMaximumArea[0]=deposited?0:cap.equilibriumContactAreaM2;
        s->pDepositionArea[0]=deposited?cap.equilibriumContactAreaM2:0;
        const double area=deposited ? double(s->pDepositionArea[0])
          : double(s->pContactMaximumArea[0])*Foam::gpuThermal::normalizedKinematicArea(age/double(s->pContactDuration[0]),double(s->pContactPeakFraction[0]));
        const double required=double(cap.adhesionSpecificEnergyJkg)/double(cap.equilibriumContactAreaM2)*area;
        const double sampleSpecific=shouldDetach?required+1:required*.25;
        const double speed=std::sqrt(2*sampleSpecific);
        s->pux[0]=speed;s->pux[1]=-speed;
        const double poolMass=double(s->pm[0])+double(s->pm[1]);
        s->poissonPoolSampleTargetCount[0]=2;s->poissonPoolMass[0]=poolMass;
        s->poissonPoolMomX[0]=s->poissonPoolMomY[0]=s->poissonPoolMomZ[0]=0;
        s->poissonPoolEnergy[0]=poolMass*sampleSpecific;
        s->poolThermalSumUx[0]=s->poolThermalSumUy[0]=s->poolThermalSumUz[0]=0;
        s->poolThermalSumU2[0]=poolMass*speed*speed;
        *s->compactCountDevice=1;s->compactPStatus[0]=0;
        correctPoissonThermalizedMobileParticlesKernel<<<1,32>>>(s,0);
        correctPoissonThermalizedStuckParticlesKernel<<<1,32>>>(s,0);deviceSynchronizeChecked();
        const bool detached=s->pStuck[0]==Foam::gpuThermal::particleWallMobile;
        if(detached!=shouldDetach){std::printf("UNEXPECTED_TRANSITION case=%d detached=%d required=%.17g sample=%.17g\n",scenario,detached,required,sampleSpecific);return 4;}
        // Same next consumer as the full step; drag/particle-gas heat disabled by valid configuration.
        relaxMobileParticlesToResidentGasKernelStatic<<<1,32>>>(s,1e-6,ugkwpGpuDrag::SchillerNaumannDrag{});deviceSynchronizeChecked();
        observe<<<1,1>>>(s,result);deviceSynchronizeChecked();
        const double storedAge=AGE_READ;
        std::printf("WITNESS case=%d detach=%d status=%d ageBefore=%.17g storedAgeAfter=%.17g thetaRaw=%.17g thetaForMoment=%.17g kinetic=%.17g mechanicalMoment=%.17g excessThetaEnergy=%.17g expectedAgeLeak=%.17g requiredSpecific=%.17g\n",scenario,detached,s->pStatus[0],age,storedAge,double(s->pTheta[0]),result[0],result[1],result[2],result[3],1.5*double(s->pm[0])*age,required);
        if(detached && (result[0]!=0 || storedAge!=0)) {
            std::puts("DETACHED_CONTACT_AGE_OR_THETA_NOT_ZERO");foundLeak=true;
        }
        if(!detached && storedAge!=age){std::puts("CONTACT_AGE_CHANGED_IN_DAMAGE_CONTROL");return 4;}
    }
    if(foundLeak){std::puts("FAIL detached contact age or mobile mechanical theta is not zero");return 2;}
    std::puts("PASS detached contact age and mobile theta are zero after native correction and relaxation");return 0;
}
'''
    code = code.replace('AGE_WRITE', 's->pContactAge[0]=age;' if app == 'CHT' else 's->pTheta[0]=age;')
    code = code.replace('AGE_READ', 'double(s->pContactAge[0])' if app == 'CHT' else 'double(s->pTheta[0])')
    cu = out / 'contact_age_theta.cu'
    cu.write_text(code)
    exe = out / 'contact_age_theta'
    cmd = [str(Path(os.environ.get('CUDA_HOME','/usr/local/cuda'))/'bin/nvcc'), '-std=c++17', '-O3', '-arch='+os.environ.get('UGKWP_CUDA_ARCH','sm_89'), '--fmad='+('false' if app=='CHT' else 'true'), '-DUGKP_DEVELOPMENT_PROBES=1', '-DUGKWP_GPU_REAL_BITS='+str(bits), '-I'+str(source.parent), '-I'+str(root/'common'), '-I'+str(root/'applications'/app/'gpu'), str(cu)]
    if app == 'CHT':
        cmd.append(str(root/'applications/CHT/gpu/GpuWallEnergy64.cu'))
    cmd += ['-o',str(exe)]
    paths=[source,root/'common/GpuThermalParticleFinalization.cuh',root/'common/GpuCollisionPoolCorrection.cuh',root/'common/GpuThermalParticleRelaxation.cuh',root/'common/operators/correctOnePoissonThermalizedParticlePath.cuh',root/'common/operators/relaxMobileParticlesToResidentGasKernelStatic.cuh',Path(__file__).resolve()]
    provenance={'scope':'Local invariant witness only; post-sampling state is constructed; neither RNG distribution nor end-to-end solver is tested.', 'source_root':str(root),'app':app,'bits':bits,'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'generated_sha256':hashlib.sha256(cu.read_bytes()).hexdigest(),'command':cmd,'execution':'not-run'}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    if '--prepare-only' in sys.argv:
        print(json.dumps(provenance,indent=2));return 0
    with (out/'build.log').open('w') as stream:
        build=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT)
    if build.returncode:
        print((out/'build.log').read_text()[-6000:]);return build.returncode
    run=subprocess.run([str(exe)],capture_output=True,text=True,cwd=out)
    (out/'run.log').write_text(run.stdout+run.stderr)
    provenance.update(execution='finished',exit=run.returncode,binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(run.stdout+run.stderr);return run.returncode


if __name__=='__main__':
    raise SystemExit(main())
