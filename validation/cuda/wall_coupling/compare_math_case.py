#!/usr/bin/env python3
"""Compare all 17 accepted fields and published wall fluxes with frozen CPU data.

The wall CSV's heatFluxIntoGas is the BVP flux. Physical coupled heat/shear
come from wall-work.csv after conservative reconstruction; these must not be
interchanged. --wall-function uses the independent archived wall-function
trajectory at the requested checkpoint, with no wall exclusion or tuning.
"""
import argparse,csv,json,re
from pathlib import Path
import numpy as np
from compare_case import field

def metrics(got,want):
    diff=np.abs(got-want);normalized=diff/np.maximum(1.,np.abs(want))
    return {'max_abs':float(diff.max()),'mae':float(diff.mean()),
            'max_normalized_floor_1':float(normalized.max()),
            'relative_L2':float(np.linalg.norm(got-want)/max(1e-300,np.linalg.norm(want))),
            'worst_index':int(diff.argmax()),'max_abs_worst_index':int(diff.argmax()),
            'max_normalized_worst_index':int(normalized.argmax()),'points_above_1e_5':int(np.sum(normalized>1e-5)),
            'points':int(got.size),'finite':bool(np.isfinite(got).all() and np.isfinite(want).all())}

def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--case',type=Path,required=True);p.add_argument('--cpu',type=Path,required=True)
    p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--wall-function',type=Path);p.add_argument('--step',type=int,default=1000)
    a=p.parse_args()
    time=max((d for d in a.case.iterdir() if d.is_dir() and re.fullmatch('[0-9eE.+-]+',d.name)),key=lambda d:float(d.name))
    manifest=json.loads((a.case/'input-manifest.json').read_text())
    run=json.loads((a.audit/'run.json').read_text());events=run['events']
    expected_steps=int(manifest['steps']);expected_dt=float(manifest['dt'])
    # Fixed-step clock additions accumulate FP64 roundoff over long runs.
    time_roundoff_limit=max(1e-18,4*np.finfo(float).eps*expected_steps**2*expected_dt)
    schedule_passed=(run['exit']==0 and len(events)==expected_steps and abs(float(time.name)-expected_steps*expected_dt)<=time_roundoff_limit
        and all(row['step']==index and np.isfinite(row['dt']) and abs(row['dt']-expected_dt)<=1e-20
                and np.isfinite(row['time']) and abs(row['time']-index*expected_dt)<=time_roundoff_limit
                for index,row in enumerate(events,1)))
    rho=field(time/'rho');velocity=field(time/'U',True);momentum=field(time/'rhoU',True)
    data={name:field(time/name) for name in ['rho','rhoE','p','k','omega','nut']};data['Tgas']=field(time/'T')
    for j,d in enumerate('xyz'):data['U'+d]=velocity[:,j];data['rhoU'+d]=momentum[:,j]
    for species in ['Air','CO2']:data['speciesRho'+species]=field(time/('rhoY_'+species))
    data['rhoK']=rho*data['k'];data['rhoOmega']=rho*data['omega']
    result={'time':float(time.name),'fields':{name:metrics(values,np.fromfile(a.cpu/(name+'.bin'),dtype='<f8')) for name,values in data.items()},
            'physical_comparison_limit':1e-5,'normalization':'abs(gpu-cpu)/max(1,abs(cpu))'}
    result['schedule']={'passed':schedule_passed,'expected_steps':expected_steps,'observed_steps':len(events),'expected_dt':expected_dt,'integrator':manifest['integrator'],'clock_roundoff_limit_seconds':time_roundoff_limit}
    result['field_cpu_parity']=all(row['finite'] and row['max_normalized_floor_1']<=1e-5 for row in result['fields'].values())
    work=list(csv.DictReader((a.audit/'wall-work.csv').open()));gpu={int(row['owner']):row for row in work}
    cpu=list(csv.DictReader((a.cpu/'walls.csv').open()))
    if set(gpu)!={int(row['owner']) for row in cpu}:raise SystemExit('Wall owner sets differ')
    owners=[int(row['owner']) for row in cpu]
    for dest,g,c in [('wall_heat','diffusiveEnergy','q'),('wall_shear','tractionX','tau')]:
        result[dest]=metrics(np.array([float(gpu[owner][g]) for owner in owners]),np.array([float(row[c]) for row in cpu]))
    result['wall_cpu_parity']=all(result[name]['finite'] and result[name]['max_normalized_floor_1']<=1e-5 for name in ['wall_heat','wall_shear'])
    result['work']={key:sum(int(row[key]) for row in work) for key in ['solves','failures','iterations','residuals','jacobians','history','fallbacks','active']}
    result['schedule']['expected_wall_solves']=len(owners)*expected_steps*(1 if manifest['integrator']=='Euler' else 2)
    result['schedule']['wall_solves_passed']=result['work']['solves']==result['schedule']['expected_wall_solves']
    result['work'].update(max_residual=max(float(row['maxResidual']) for row in work),max_moment=max(float(row['momentResidual']) for row in work))
    result['species_closure_max']=float(np.max(np.abs(data['speciesRhoAir']+data['speciesRhoCO2']-rho)))
    result['minimums']={name:float(data[name].min()) for name in ['rho','p','Tgas','k','omega','speciesRhoAir','speciesRhoCO2']}
    result['species_closure_limit']=1e-12*max(1.,float(np.max(np.abs(rho))))
    result['physical_state_passed']=(result['species_closure_max']<=result['species_closure_limit']
        and all(result['minimums'][name]>0 for name in ['rho','p','Tgas','omega'])
        and all(result['minimums'][name]>=0 for name in ['k','speciesRhoAir','speciesRhoCO2']))
    faces=list(csv.DictReader((a.audit/'wall-faces.csv').open()))
    wall=[row for row in faces if int(row['index'])==0]
    result['wall_flux_max_abs']={name:max(abs(float(row[name])) for row in wall) for name in ['mass','species0','species1']}
    if a.wall_function:
        rows=[row for row in csv.DictReader(a.wall_function.open()) if int(row['step'])==a.step]
        if {int(row['owner']) for row in rows}!=set(gpu):raise SystemExit('Wall-function owner set differs')
        q=np.array([float(gpu[int(row['owner'])]['diffusiveEnergy']) for row in rows]);tau=np.array([float(gpu[int(row['owner'])]['tractionX']) for row in rows])
        wq=np.array([float(row['wfQ']) for row in rows]);wtau=np.array([float(row['wfTau']) for row in rows]);area=np.array([float(row['area']) for row in rows])
        activity=np.array([float(row['wfActivity']) for row in rows]);weights=area/area.sum()
        result['wall_function']={'reference':'independently evolved frozen CPU wall-function trajectory','walls':len(rows),
          'heat_area_mae_W_m2':float(np.sum(weights*np.abs(q-wq))),
          'shear_area_mae_Pa':float(np.sum(weights*np.abs(tau-wtau))),
          'heat_sign_disagreements':int(np.sum(q*wq<0)),
          'shear_area_absolute_error_over_absolute_reference':float(np.sum(area*np.abs(tau-wtau))/np.sum(area*np.abs(wtau))),
          'absolute_heat_error_over_integrated_absolute_activity':float(np.sum(area*np.abs(q-wq))/np.sum(area*activity)),
          'activity_ratio_is_not_net_heat_relative_error':True,
          'entrance':{'owner':int(rows[0]['owner']),'gpu_heat':float(q[0]),'wall_function_heat':float(wq[0]),'gpu_shear':float(tau[0]),'wall_function_shear':float(wtau[0])}}
        with a.output.with_suffix('.walls.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['owner','x','area','gpuQ','cpuQ','wfQ','gpuTau','cpuTau','wfTau'])
            for row,gq,gt in zip(rows,q,tau):writer.writerow([row['owner'],row['x'],row['area'],gq,row['newQ'],row['wfQ'],gt,row['newTau'],row['wfTau']])
    interior_species=[abs(float(row[name])) for row in faces if int(row['index'])!=0 for name in ['species0','species1']]
    finite_scale=all(np.isfinite(value) for value in interior_species)
    result['wall_species_roundoff_scale_kg_s']=max(interior_species,default=0.)
    result['wall_species_roundoff_limit_kg_s']=64*np.finfo(float).eps*result['wall_species_roundoff_scale_kg_s']
    result['wall_species_roundoff_criterion']='64*FP64 epsilon*max absolute sampled interior species face flux; zero scale gives zero limit'
    result['no_blow_fluxes_passed']=(finite_scale and len(wall)==len(owners)
        and all(np.isfinite(float(row['mass'])) and float(row['mass'])==0.
                and all(np.isfinite(float(row[name])) and abs(float(row[name]))<=result['wall_species_roundoff_limit_kg_s'] for name in ['species0','species1'])
                and float(row['species0'])+float(row['species1'])==0. for row in wall))
    result['work_finite']=all(np.isfinite(float(row[name])) for row in work for name in ['maxResidual','momentResidual'])
    result['passed']=schedule_passed and result['schedule']['wall_solves_passed'] and result['physical_state_passed'] and result['work_finite'] and result['no_blow_fluxes_passed'] and result['field_cpu_parity'] and result['wall_cpu_parity'] and result['work']['failures']==0 and result['work']['fallbacks']==0 and result['work']['max_residual']<=1.01e-8 and result['work']['max_moment']<=1e-9
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return 0 if result['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
