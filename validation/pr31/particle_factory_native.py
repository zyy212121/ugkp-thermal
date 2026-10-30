"""Public shared particle factory + borrowed mixture gas + real phase advance.
The reference owner supplies only the mesh/gas and initial restart snapshot;
the particle owner is created/configured using the API used by CHMT.
"""
import argparse,json,subprocess
from pathlib import Path
from mixture_thermal_native import source as mixture_source,ROOT,FIELDS,caloric_inventory,Decimal

def source():
    s=mixture_source()
    needle=' std::printf("{вы");printState("before",s);'.replace('вы','')
    assert needle in s
    s=s.replace(needle,r'''
 DeviceState* gasOwner=s;
 auto gas=ugkp::thermal::particleGasBinding(*gasOwner);
 ugkp::thermal::ParticlePhaseConfiguration config;
 config.capacity=1;config.rhoSolid=gasOwner->rhoSolid;
 config.minimumDiameter=config.maximumDiameter=config.diameter=.001;
 config.minimumTemperature=1;config.maximumTemperature=10000;config.minimumTheta=0;
 config.dragModel=0;config.heatTransferModel=gasOwner->particleGasHeatTransferModelId;
 config.csrCellLocalPath=csr;
 ugkp::thermal::ParticleContactConfiguration contact;
 contact.model=argc>4?std::atoi(argv[4]):1;contact.heatTransfer=0;
 const unsigned char mask[6]={1,1,1,1,1,1};const double gravity[3]={};void* handle=nullptr;
 CHECK(ugkp::thermal::createBoundParticlePhase(config,contact,gas,6,mask,gravity,&handle)==0);
 s=asState(handle);
 CHECK(s->coldWallSolidificationEnabled==(contact.model==3));
 CHECK(s->coldWall2DEnabled==(contact.model==4));
 std::vector<ugkp::thermal::ParticleState<>> particles;
 CHECK(ugkp::thermal::downloadParticleState(gasOwner,particles)==0);
 CHECK(ugkp::thermal::uploadParticleState(s,int(particles.size()),particles.data())==0);
 const double eps=readOne(gasOwner->epsGPrev);
 CHECK(ugkwpGpuResidentStrictUploadEpsGPrev(s,&eps)==0);
 CHECK(ugkp::thermal::particleGasVolumeFraction(s)==s->epsGPrev);
 cudaStream_t stream;CUDA_OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
 const ugkp::thermal::ParticleTrackingBoundaryView boundary{6,gasOwner->cellFaceKind,gasOwner->cellFaceRestitution,gasOwner->cellFaceTangential};
 CHECK(ugkp::thermal::refreshParticleTrackingGeometry(s,boundary,stream)==0);
'''+needle)
    s=s.replace('CHECK(advanceResidentThermalParticles(s,dt,dt)==0);','CHECK(ugkp::thermal::advanceBoundParticlePhase(s,dt,dt,stream)==0);')
    s=s.replace(' releaseState(s);',r'''
 ugkwpGpuResidentStrictRelease(s);
 // The external owner is still live; borrowed pointers were not freed.
 CHECK(readOne(gasOwner->rho)==1);CUDA_OK(cudaStreamDestroy(stream));releaseState(gasOwner);
''')
    return s

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    cu=out/'particle-factory.cu';cu.write_text(source());exe=out/'particle-factory'
    command=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','--fmad=false','--expt-relaxed-constexpr','--Ofast-compile=min','-arch=sm_89',
        '-I'+str(ROOT/'common/thermal'),'-I'+str(ROOT/'common'),str(cu),str(ROOT/'common/thermal/GpuWallEnergy64.cu'),'-o',str(exe)]
    with (out/'build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=[]
    for model in (1,3,4):
        for scenario in (0,1,2):
            for csr in (0,1):
                run=subprocess.run([str(exe),str(scenario),str(csr),'1',str(model)],capture_output=True,text=True)
                (out/f'run-{model}-{scenario}-{csr}.log').write_text(run.stdout+run.stderr);run.check_returncode()
                raw=json.loads(run.stdout);before=dict(zip(FIELDS,raw['before']));after=dict(zip(FIELDS,raw['after']))
                residual=caloric_inventory(after,raw['caloric'])-caloric_inventory(before,raw['caloric'])
                residual+=Decimal.from_float(after['mass'])*Decimal.from_float(raw['carry_after'])-Decimal.from_float(before['mass'])*Decimal.from_float(raw['carry_before'])
                assert abs(residual)<Decimal('3e-8'),residual
                if scenario:assert after['particle_T']<before['particle_T'] and after['gas_T']>before['gas_T']
                rows.append(dict(model=model,scenario=scenario,csr=csr,residual_J=str(residual),raw=raw))
    with (out/'memcheck.log').open('w') as log:
        subprocess.run(['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','88',str(exe),'2','1','1','4'],stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(dict(runs=len(rows),maximum_residual_J=max(abs(float(r['residual_J'])) for r in rows))))
if __name__=='__main__':main()
