"""Actual shared contact heat transaction: gas caloric capacity and energy datum.
No CHMT application enablement and no wall-conduction/2D-domain-bound claim.
"""
import argparse,json,math,subprocess
from pathlib import Path
from mixture_thermal_native import source as mixture_source, ROOT,FIELDS,caloric_inventory,Decimal

def source():
    text=mixture_source()
    text=text.replace("readOne(s->pStuck)==0","readOne(s->pStuck)==1")
    text=text.replace('s.pT[0]=10000;', 's.pT[0]=500;s.pStuck[0]=1;s.pStuckFaceId[0]=0;')
    text=text.replace('const double masses[2]={.9,.1},coefficients[6]={1000,-.22,-1e7,1000,-.22,-1e7};',
        'const double formation=argc>4?std::atof(argv[4]):-1e7,slope=argc>5?std::atof(argv[5]):-.22;\n const double masses[2]={.9,.1},coefficients[6]={1000,slope,formation,1000,slope,formation};')
    needle=' initialiseClosedCube<<<1,1>>>(s->deviceState);'
    assert needle in text
    text=text.replace(needle,r'''
 const unsigned char mask[6]={1,0,0,0,0,0};
 CHECK(ugkwpGpuResidentStrictConfigureParticleStuckModel(s,6,mask,100.,0,1.,1.,1.,1.,90.,0,8,2327.,20.,1.1e6,3950.,1000.,5.,.01,0.)==0);
'''+needle)
    text=text.replace(' CHECK(rebuildResidentParticleMomentsFromParticles(s,1)==0);',
        ' rebuildWallBoundParticleDirectoryKernel<<<1,128>>>(s->deviceState);CUDA_OK(cudaDeviceSynchronize());CHECK(readOne(s->wallBoundParticleCountDevice)==1);\n CHECK(rebuildResidentParticleMomentsFromParticles(s,1)==0);')
    return text

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--common-root',type=Path,default=ROOT/'common');p.add_argument('--observe-baseline',action='store_true');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True);common=a.common_root.resolve()
    cu=out/'mixture-contact.cu';cu.write_text(source());exe=out/'mixture-contact'
    command=['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','--fmad=false','--expt-relaxed-constexpr','--Ofast-compile=min','-arch=sm_89',
        '-I'+str(common/'thermal'),'-I'+str(common),str(cu),str(common/'thermal/GpuWallEnergy64.cu'),'-o',str(exe)]
    with (out/'build.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    rows=[]
    for slope in (0.,-.22):
        for formation in (0.,-1e7,1e7):
            for csr in (0,1):
                run=subprocess.run([str(exe),'2',str(csr),'1',str(formation),str(slope)],capture_output=True,text=True)
                (out/f'run-{slope}-{formation}-{csr}.log').write_text(run.stdout+run.stderr);run.check_returncode()
                raw=json.loads(run.stdout);before=dict(zip(FIELDS,raw['before']));after=dict(zip(FIELDS,raw['after']))
                for key in ('eps','rho','gx','gy','gz','mass','px','py','pz','theta'):assert before[key]==after[key],key
                mass=Decimal.from_float(before['mass']);inventory=caloric_inventory(after,raw['caloric'])-caloric_inventory(before,raw['caloric'])
                inventory+=mass*(Decimal.from_float(raw['carry_after'])-Decimal.from_float(raw['carry_before']))
                row={'slope':slope,'formation':formation,'csr':csr,'particle_temperature_gain':after['particle_T']-before['particle_T'],
                    'inventory_residual_J':str(inventory),'raw':raw};rows.append(row)
                if not a.observe_baseline:
                    assert row['particle_temperature_gain']>0,row
                    cp=raw['caloric'][2];R=8.31446261815324/.028
                    C=before['mass']*cp;Cg=before['eps']*before['rho']*(1000+slope*before['gas_T']-R)
                    rate=12*.06/(2500*cp*.001**2)
                    expected=Cg/(Cg+C)*(before['gas_T']-before['particle_T'])*(-math.expm1(-rate*(1+C/Cg)*raw['dt']))
                    row['analytic_temperature_gain']=expected
                    assert abs(row['particle_temperature_gain']-expected)<2e-10,row
                    assert abs(inventory)<Decimal('4e-8'),row
    if not a.observe_baseline:
        for slope in (0.,-.22):
            values=[r['particle_temperature_gain'] for r in rows if r['slope']==slope]
            assert max(values)-min(values)<1e-8,(slope,values)
        with (out/'memcheck.log').open('w') as log:
            subprocess.run(['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','88',str(exe),'2','1','1','-10000000','-.22'],stdout=log,stderr=subprocess.STDOUT,check=True)
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='raw'} for r in rows]))
if __name__=='__main__':main()
