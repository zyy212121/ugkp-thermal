#!/usr/bin/env python3
"""Full CHT radiation/solid/particle parity and closed physical energy inventory.

All variants select existing models through input dictionaries. This runner has
no production-code hooks. Baseline and candidate own independent fresh cases.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import re
import shutil
import numpy as np
import particle_contact as contact
layer=contact.layer
matrix=contact.matrix

def prepare(case,grid,threads,mobile):
    contact.prepare(case,grid,.00004)
    path=case/'constant/solidRegionProperties'
    path.write_text(path.read_text().replace('emissivity {type constant;value 0;}','emissivity {type constant;value .8;}'))
    (case/'constant/radiationProperties').write_text('''FoamFile {version 2.0; format ascii; class dictionary; object radiationProperties;}
schemaVersion 1; enabled true; couplingFrequency 1;
model {mieTable "constant/mie.table"; do {
 nTheta 3; nPhi 8; maxSourceIterations 300; sourceRelativeTolerance 1e-10;
 phaseBalanceTolerance 1e-12; maxPhaseBalanceIterations 200;
 radiationActiveEpsilonMin 0; conservationRelativeTolerance 1e-8; maxPhaseScalingCorrection 100;
 IBoundaryCondition specular; boundaryRadiation {left {type diffuseGrey;}}
 angleParallelThreads '''+str(threads)+'''; }}
''')
    spec=importlib.util.spec_from_file_location('radiation_fixture',layer.SOURCE/'validation/solvers/CHT/devtools/tests/test_radiation_continuation.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture);fixture.table(case/'constant/mie.table')
    if mobile:
        data=contact.particles(case/'0');data['pStuck'][:]=0;data['pStuckFaceId'][:]=-1;data['pm'][:]=1e-4
        # Place the stationary mobile particles inside their original owner cells.
        data['px'][:]=.5/grid
        contact.wire.write_restart(case/'0/mobile.dat',[data],len(data['pm']),262144,cht=True)
        (case/'0/mobile.dat').replace(case/'0/gpuResidentStrictParticles.dat')

def inventory(case,grid,mobile,*,enforce_absolute_bounds=True):
    states=[]
    for snapshot in matrix.times(case):
        gm,ge=contact.gas_inventory(snapshot,grid)
        if mobile:
            enthalpy=layer.scalar_sum(snapshot/'rhoHp',grid**3)
            fields=contact.particles(snapshot);mp=math.fsum(fields['pm']);assert np.all(fields['pStuck']==0)
            temperature=layer.field(snapshot/'Tp',grid**3)
            # The existing radiation transaction uses the configured constant
            # heat factor (1000 J/kg/K in this fixture). Also retain the actual
            # nonlinear enthalpy inventory; its baseline discrepancy is evidence,
            # not a tolerance change or a claim of enthalpy conservation.
            ep=math.fsum(float(mass)*1000*temperature[int(cell)] for mass,cell in zip(fields['pm'],fields['cell']))
        else:
            ep,mp=contact.particle_inventory(snapshot);enthalpy=ep
        se=1e6*layer.scalar_sum(snapshot/'solid/T',grid**3)
        qr=contact.patch_flux(snapshot,'particleRadiationWallHeatFlux',grid)
        qg=contact.patch_flux(snapshot,'gasConvectiveWallHeatFlux',grid)
        qp=contact.patch_flux(snapshot,'particleStuckWallHeatFlux',grid)+contact.patch_flux(snapshot,'particleReflectedWallHeatFlux',grid)
        states.append(dict(gas_wall_flux=qg,contact_wall_flux=qp,time=float(snapshot.name),gas=ge,solid=se,particle=ep,total=ge+se+ep,particle_nonlinear_enthalpy=enthalpy,total_using_nonlinear_enthalpy=ge+se+enthalpy,mass=gm+mp,radiation_wall_flux=qr))
    assert len(states)>=2
    energy=max(abs(row['total']-states[0]['total']) for row in states)
    mass=max(abs(row['mass']-states[0]['mass']) for row in states)
    if enforce_absolute_bounds:
        assert energy<2e-5,(case,'closed gas+solid+particle energy',energy)
        assert mass<1e-10,(case,'closed mass',mass)
    if mobile:assert max(abs(row['radiation_wall_flux']) for row in states)>1e-9
    nonlinear=max(abs(row['total_using_nonlinear_enthalpy']-states[0]['total_using_nonlinear_enthalpy']) for row in states)
    exchange=dict(gas=0.,solid=0.,particle=0.)
    for old,new in zip(states,states[1:]):
        dt=new['time']-old['time']
        exchange['gas']=max(exchange['gas'],abs(new['gas']-old['gas']+dt*new['gas_wall_flux']))
        exchange['solid']=max(exchange['solid'],abs(new['solid']-old['solid']-dt*(new['gas_wall_flux']+new['contact_wall_flux']+new['radiation_wall_flux'])))
        exchange['particle']=max(exchange['particle'],abs(new['particle']-old['particle']+dt*(new['contact_wall_flux']+new['radiation_wall_flux'])))
    assert exchange['solid']<2e-5 and exchange['particle']<2e-5,(case,exchange)
    return dict(states=states,maximum_exchange_residual_J=exchange,absolute_energy_bound_pass=energy<2e-5,absolute_mass_bound_pass=mass<1e-10,maximum_transaction_model_energy_residual_J=energy,maximum_nonlinear_enthalpy_inventory_drift_J=nonlinear,
        maximum_closed_mass_residual_kg=mass,energy_scope='Configured constant particle heat factor for mobile radiation; nonlinear enthalpy inventory reported independently.')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True);p.add_argument('--grid',type=int,default=2)
    p.add_argument('--mobility',choices=['both','mobile','deposited'],default='both');p.add_argument('--angle-threads',type=int,nargs='+',default=[1,4])
    p.add_argument('--record-baseline-conservation',action='store_true',help='Retain absolute closure pass/fail and require unchanged baseline residuals; do not mark a failed absolute bound as passed.')
    a=p.parse_args();a.work.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',cases=[],binaries={key:layer.digest(binary) for key,binary in [('baseline',a.baseline),('candidate',a.candidate)]})
    def save():(a.work/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        for mobile in ([False,True] if a.mobility=='both' else [a.mobility=='mobile']):
            for threads in a.angle_threads:
                item=dict(mobile=mobile,angle_threads=threads,runs={});report['cases'].append(item);save();old=None
                for key,binary in [('baseline',a.baseline),('candidate',a.candidate)]:
                    case=a.work/f'mobile{int(mobile)}-threads{threads}-{key}';prepare(case,a.grid,threads,mobile)
                    elapsed=matrix.run([binary.resolve(),'-case',case],case/'run.log')
                    item['runs'][key]=dict(elapsed_seconds=elapsed,audit=inventory(case,a.grid,mobile,enforce_absolute_bounds=not a.record_baseline_conservation))
                    if old:
                        item['parity']=layer.compare_fields(old,case)
                        left,right=contact.particles(matrix.times(old)[-1]),contact.particles(matrix.times(case)[-1])
                        for field in left:assert np.array_equal(left[field],right[field],equal_nan=True),(case,field)
                        item['particle_restart_fields_bitwise_equal']=True
                        old_audit=item['runs']['baseline']['audit'];new_audit=item['runs']['candidate']['audit']
                        for metric in ('maximum_transaction_model_energy_residual_J','maximum_nonlinear_enthalpy_inventory_drift_J','maximum_closed_mass_residual_kg'):
                            assert abs(old_audit[metric]-new_audit[metric])<2e-10,(case,metric)
                    else:old=case
                    save()
        report['status']='PASS'
        if any(not row['audit'][key] for item in report['cases'] for row in item['runs'].values() for key in ['absolute_energy_bound_pass','absolute_mass_bound_pass']):
            report['status']='PASS_PARITY_WITH_BASELINE_CONSERVATION_LIMIT'
        save()
    except BaseException as error:report.update(status='FAILED',error=repr(error));save();raise
if __name__=='__main__':main()
