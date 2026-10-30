"""Real CHMT frozen-mixture particle/contact cases; local evidence only."""
import argparse,csv,importlib.util,json,re,subprocess,shutil,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('native_matrix',ROOT/'validation/molecular_transport/native_matrix.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def prepare(path,model,impact,end_time=0.001,aligned_calorics=False):
    meta=json.loads((ROOT/'validation/molecular_transport/fixtures.json').read_text())['chmt-mixture']
    m.prepare(ROOT/'validation/molecular_transport/fixtures/chmt-mixture',path,meta)
    control=path/'system/controlDict'
    control.write_text(re.sub(r'endTime [^;]+;',f'endTime {end_time:.17g};',control.read_text()))
    d=path/'constant/chmtProperties'
    # A separate conservation fixture aligns the liquid datum of the two
    # existing caloric laws. Default-model runs remain available and explicit.
    cp=lambda t:(153.5+.0031*(min(4000.,max(2327.,t))-2327.))/.101961276
    liquid_h=lambda t:cp(2327.)*t+.5*(.0031/.101961276)*max(0.,t-2327.)**2
    cold_without_latent=1273.*2317.+.5*(1273.+cp(2327.))*20.
    latent=liquid_h(2337.)-cold_without_latent
    calorics=f'latentHeat {latent:.17g};' if aligned_calorics else ''
    position='0.000005 0.25 0.25' if impact else '0.25 0.25 0.25'
    velocity='-1 0 0' if impact else '0 0 0'
    d.write_text(d.read_text()+f"""
enableParticles true;
particle {{
 volumeMode Coupled; capacity 64; rhoSolid 2500;
 dragModel none; heatTransferModel RanzMarshall; contactModel {model};
 contactCoeffs {{ sommerfeldThreshold 100; interfaceResistance 0.00001; {calorics} }}
}}
particles {{ parcel {{
 id \"42\"; rng \"1234567\"; cell 0; position ({position}); velocity ({velocity});
 mass 0.002; diameter 0.001; temperature 2500;
}} }}
""")
    return path

def table(path):
    with path.open() as f:return list(csv.DictReader(f))

def inventory(path,index):
    g=table(path/f'gas-{index}.csv');s=table(path/f'material-{index}.csv');p=table(path/f'particles-{index}.csv')
    active=[q for q in p if int(q['status'])]
    particle=sum(float(q['thermal_energy'])+float(q['contact_mechanical_energy'])+float(q['mass'])*(.5*sum(float(q[k])**2 for k in ('ux','uy','uz'))+1.5*float(q['theta'])) for q in active)
    return dict(gas=sum(float(q['total_energy']) for q in g),solid=sum(float(q['total_energy']) for q in s),particle=particle,
        gas_mass=sum(float(q['mass']) for q in g),particle_mass=sum(float(q['mass']) for q in active),parcels=p)

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--prepare-only',action='store_true')
    a.add_argument('--model',default='reboundContact',choices=['reboundContact','coldWall1D','coldWall2D']);a.add_argument('--impact',action='store_true')
    a.add_argument('--end-time',type=float,default=.001);a.add_argument('--aligned-calorics',action='store_true');a.add_argument('--restart-check',action='store_true')
    a.add_argument('--solver',type=Path,default=ROOT/'applications/CHMT/.build/native/CHMT');args=a.parse_args()
    case=prepare(args.output.resolve(),args.model,args.impact,args.end_time,args.aligned_calorics)
    if args.prepare_only:return
    seconds=m.run([args.solver,'-case',case],case/'run.log')
    output=case/'chmtOutput';indices=sorted(int(p.stem.split('-')[1]) for p in output.glob('particles-*.csv'))
    first,last=inventory(output,indices[0]),inventory(output,indices[-1]);assert len(last['parcels'])==1 and int(last['parcels'][0]['status'])==1
    assert first['parcels'][0]['temperature']!=last['parcels'][0]['temperature']
    heat=sum(float(q['energy']) for i in indices[1:] for q in table(output/f'exchange-{i}.csv') if q['kind']=='5')
    report=dict(elapsed_seconds=seconds,model=args.model,impact=args.impact,first=first,last=last,particle_wall_energy_J=heat,
        total_energy_change_J=sum(last[k]-first[k] for k in ('gas','solid','particle')))
    if args.impact:assert heat!=0,report
    report['aligned_calorics']=args.aligned_calorics
    assert all(math.isfinite(float(v)) for q in last['parcels'] for k,v in q.items() if k not in ('id',))
    report['energy_absolute_below_2e_4_J']=abs(report['total_energy_change_J'])<2e-4
    # Report the physical residual, including the accepted macro/micro error;
    # successful execution or restart parity alone is not a conservation pass.
    if args.restart_check:
        restart=case.parent/(case.name+'-restart')
        assert not restart.exists(),restart
        shutil.copytree(case,restart,ignore=shutil.ignore_patterns("chmtOutput"))
        shutil.copytree(output/"checkpoint-1",restart/"initial-checkpoint")
        properties=restart/'constant/chmtProperties'
        properties.write_text(properties.read_text()+'\nrestartDirectory "initial-checkpoint";\n')
        m.run([args.solver,'-case',restart],restart/'restart.log')
        errors={}
        for name in ('gas','material','particles'):
            before=table(output/f'{name}-{indices[-1]}.csv');after=table(restart/'chmtOutput'/f'{name}-{indices[-1]}.csv')
            assert len(before)==len(after)
            maximum=0.
            for left,right in zip(before,after):
                assert left.keys()==right.keys()
                for key in left:
                    try:x,y=float(left[key]),float(right[key])
                    except ValueError:assert left[key]==right[key];continue
                    assert math.isfinite(x) and math.isfinite(y)
                    maximum=max(maximum,abs(x-y)/max(1.,abs(x),abs(y)))
            assert maximum<2e-13,(name,maximum)
            errors[name]=maximum
        report['restart_max_relative_difference']=errors
        reference=output/f'checkpoint-{indices[-1]}/state.chmt'
        resumed=restart/'chmtOutput'/f'checkpoint-{indices[-1]}/state.chmt'
        report['restart_checkpoint_byte_identical']=reference.read_bytes()==resumed.read_bytes()
        assert report['restart_checkpoint_byte_identical']
    (case/'particle-results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
