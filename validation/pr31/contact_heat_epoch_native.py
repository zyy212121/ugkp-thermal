"""Native production collision/capillary/heat stages across contact detachment.

The controlled pool replaces stochastic sampling, not collision finalization or
heat mathematics. The other mobile parcel is at gas temperature, so the earlier
single-gas macro heat source is zero. This is a stage probe, not a full Advance
or performance benchmark. Mode 2 deliberately isolates the original free heat
relaxation without a macro update; it checks dispatch, not energy closure.
--source-root permits the same probe on old sources.
"""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'validation/tests/gas_transport'))
sys.path.insert(0,str(ROOT))
from test_full_heat_advance import native_cuda_fixture

def source():
    text=native_cuda_fixture().replace('s->particleCapacity=1;', 's->particleCapacity=2;')
    text=text.replace('s->rhoSolid=1000;s->invRhoSolid=GPU_R(.001);','s->rhoSolid=2500;s->invRhoSolid=GPU_R(.0004);')
    text=text.replace('s->particleGasHeatTransferModelId=scenario==0?0:1;', 's->particleGasHeatTransferModelId=1;')
    text=text.replace('const double dt=scenario==1?1e-8:1e-4;', 'const double dt=argc>1?std::atof(argv[1]):1e-4;')
    text=text.replace('int main(int argc,char**argv){', r"""
__global__ void setupHeat(DeviceState* sp,int mode){
 auto& s=*sp;*s.particleCountDevice=2;
 for(int i=0;i<2;++i){s.pm[i]=.002;s.pd[i]=.001;s.pT[i]=i?3200:2800;
  s.pStatus[i]=1;s.pCellId[i]=0;s.pStuck[i]=0;s.pStuckFaceId[i]=-1;
  s.pThermalEnthalpyResidual[i]=0;s.pux[i]=s.puy[i]=s.puz[i]=0;}
 s.pStuck[0]=mode==2?0:1;s.pStuckFaceId[0]=mode==2?-1:0;
 s.pDepositionArea[0]=mode==1?1e-8:0;
 s.couplingTgasOld[0]=3200;s.couplingRhoOld[0]=1;
 s.couplingUxOld[0]=s.couplingUyOld[0]=s.couplingUzOld[0]=0;
 s.epsGPrev[0]=1-.004/s.rhoSolid;
 clearPairedParticleHeatCell(s,0);
}
__global__ void prepareControlledPool(DeviceState*sp,int mode){
 auto& s=*sp;
 // Actual selected-contact correction receives zero random energy. Zero
 // remaining contact area releases; nonzero adhesion area remains attached.
 s.pStatus[0]=mode==2?1:2;*s.compactCountDevice=mode==2?0:1;s.compactPStatus[0]=0;
 s.poissonPoolSampleTargetCount[0]=1;s.poissonPoolMass[0]=s.pm[0];
 s.poissonPoolMomX[0]=s.poissonPoolMomY[0]=s.poissonPoolMomZ[0]=s.poissonPoolEnergy[0]=0;
}
__global__ void relaxProbe(DeviceState*sp,double dt){
 auto& s=*sp;for(int i=0;i<2;++i)relaxOneParticleToResidentGas(s,i,dt,ugkwpGpuDrag::SchillerNaumannDrag{});
}
double particleHeat(DeviceState*s){
 double e=0;for(int i=0;i<2;++i)e+=readOne(s->pm+i)*(ugkwp::ParticleCaloric{}.enthalpy(readOne(s->pT+i))+readOne(s->pThermalEnthalpyResidual+i));return e;
}
int main(int argc,char**argv){
""")
    start=text.index(' CHECK(rebuildResidentParticleMomentsFromParticles(s,1)==0);')
    return text[:start]+r"""
 const int mode=argc>2?std::atoi(argv[2]):0;
 const unsigned char mask[6]={1,0,0,0,0,0};
 CHECK(ugkwpGpuResidentStrictConfigureParticleStuckModel(s,6,mask,100.,1,1.,1.,1.,1.,90.,0,8,2327.,20.,1.1e6,3950.,1000.,5.,.01,0.)==0);
 s->particleGasHeatTransferModelId=1;s->particleWallHeatTransferEnabled=0;
 CHECK(syncDeviceState(s,"heat configuration")==0);
 setupHeat<<<1,1>>>(s->deviceState,mode);CUDA_OK(cudaDeviceSynchronize());
 CHECK(rebuildResidentParticleMomentsFromParticles(s,2)==0);
 prepareMobileParticleHeatMomentsKernel<<<1,1>>>(s->deviceState);
 subtractContactParticleHeatMomentsKernel<<<1,1>>>(s->deviceState);
 CUDA_OK(cudaDeviceSynchronize());
 const double epochBefore=readOne(s->particleContactHeatEpoch);
 const double gasBefore=readOne(s->rhoE)*readOne(s->epsGPrev),heatBefore=particleHeat(s);
 if(mode!=2)applyEulerianParticleMaterialHeatKernel<<<1,1>>>(s->deviceState,dt);
 prepareControlledPool<<<1,1>>>(s->deviceState,mode);
 correctPoissonThermalizedMobileParticlesKernel<true><<<1,32>>>(s->deviceState,0,dt);
 correctPoissonThermalizedStuckParticlesKernel<true><<<1,32>>>(s->deviceState,0,dt);
 acceptPairedParticleHeatKernel<<<1,1>>>(s->deviceState);
 relaxProbe<<<1,1>>>(s->deviceState,dt);
 settlePairedParticleHeatKernel<<<1,1>>>(s->deviceState);
 CUDA_OK(cudaDeviceSynchronize());
 const double dg=readOne(s->rhoE)*readOne(s->epsGPrev)-gasBefore,dp=particleHeat(s)-heatBefore;
 const int stuck=int(readOne(s->pStuck)),epoch=int(readOne(s->particleContactHeatEpoch));
 CHECK(stuck==(mode==1?1:0));CHECK(readOne(s->pStatus)==1);
 // Next source formation must retire the transaction after detachment.
 subtractContactParticleHeatMomentsKernel<<<1,1>>>(s->deviceState);CUDA_OK(cudaDeviceSynchronize());
 std::printf("{\"dt\":%.17g,\"mode\":%d,\"gas_J\":%.17g,\"particle_J\":%.17g,\"residual_J\":%.17g,\"state\":%d,\"epoch_before\":%.0f,\"epoch_settlement\":%d,\"epoch_next\":%.0f}\n",dt,mode,dg,dp,dg+dp,stuck,epochBefore,epoch,readOne(s->particleContactHeatEpoch));
 releaseState(s);
}
"""

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-root',type=Path,default=ROOT);p.add_argument('--record-only',action='store_true');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);r=a.source_root.resolve()
    cu=out/'contact-heat.cu';cu.write_text(source());exe=out/'contact-heat'
    cmd=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','--fmad=false','--expt-relaxed-constexpr','--Ofast-compile=min','-arch=sm_89',
         '-I'+str(r/'common/thermal'),'-I'+str(r/'common'),'-I'+str(r/'common/gasNumerics'),str(cu),str(r/'common/thermal/GpuWallEnergy64.cu'),'-o',str(exe)]
    (out/'build-command.json').write_text(json.dumps(cmd,indent=2)+'\n')
    with (out/'build.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
    results=[]
    for mode in (0,1,2):
        for dt in (0.,1e-4,1e-3):
            run=subprocess.run([str(exe),str(dt),str(mode)],capture_output=True,text=True)
            (out/f'run-{mode}-{dt}.log').write_text(run.stdout+run.stderr);run.check_returncode()
            q=json.loads(run.stdout);results.append(q)
            if not a.record_only:
                assert q['epoch_next']==(mode==1),q
                if mode!=2:
                    assert q['epoch_before']==q['epoch_settlement']==1,q
                    assert abs(q['residual_J'])<1e-8,q
                    if dt>0:assert q['particle_J']>0 and q['gas_J']<0,q
                else:assert q['epoch_before']==q['epoch_settlement']==0 and q['gas_J']==0,q
                if dt==0:assert q['particle_J']==q['gas_J']==0,q
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    if not a.record_only:
        with (out/'memcheck.log').open('w') as log:
            subprocess.run(['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','88',str(exe),'0.0001','0'],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(json.dumps(results))
if __name__=='__main__':main()
