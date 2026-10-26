#!/usr/bin/env python3
"""Native layer-one evidence using existing model fixtures and isolated binaries.

Run from sourced OpenFOAM 10. --candidate is optional for baseline characterization.
No solver tolerances or fixture physics are changed. Repeated timing alternates
binaries after one warmup; process startup and normal output are included.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import re
import statistics
import struct
from pathlib import Path

SOURCE=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('molecular_matrix',SOURCE/'validation/molecular_transport/native_matrix.py')
matrix=importlib.util.module_from_spec(spec);spec.loader.exec_module(matrix)
NUMBER=r'[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?'
DEFAULT_CASES=['chmt-single','chmt-mixture','chmt-diffusion','chmt-moving-short','cht-contact','cht-mixture','cht-pure','cht-wall','cht-ode','cht-wallfunction']


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:return list(csv.DictReader(stream))


def physical_rows(path):
    return [{k:float(v) for k,v in row.items() if k!='sst_chemistry_integral_audit_collection'} for row in rows(path)]


def compare_csv(left,right):
    a,b=rows(left),rows(right)
    assert len(a)==len(b),(left,'row count')
    maximum={}
    discrete={'packet','step','geometry','face','stage','kind','gas_cell','solid_cell','consumer_mask','cell','accepted_steps','gas_cells','material_cells','interface_faces','gas_mode','thermo_hash','commit_sequence'}
    for x,y in zip(a,b):
        assert x.keys()==y.keys(),left
        for key,value in x.items():
            if key in discrete or key=='sst_chemistry_integral_audit_collection' or value=='':assert value==y[key];continue
            try:u,v=float(value),float(y[key])
            except ValueError:assert value==y[key];continue
            if left.name.startswith('wall-layer-') and key=='owner_k_source_integral' and math.isnan(u) and math.isnan(v):continue
            assert math.isfinite(u) and math.isfinite(v),(left,key,u,v)
            error=abs(u-v);scale=max(abs(u),abs(v))
            maximum[key]=max(maximum.get(key,0.),error/max(1.,scale))
            # Packet energy subtraction may cancel; use a small absolute floor.
            assert error<=2e-10+2e-10*scale,(left,key,u,v,error)
    return maximum


def compare_chmt(left,right):
    result={};paths=sorted((left/'chmtOutput').glob('*.csv'))
    assert {p.name for p in paths}=={p.name for p in (right/'chmtOutput').glob('*.csv')}
    for path in paths:
        errors=compare_csv(path,right/'chmtOutput'/path.name)
        for key,error in errors.items():result[key]=max(error,result.get(key,0.))
    def payload(case):
        wire=matrix.terminal(case).read_bytes();header=matrix.HEADER.unpack_from(wire)
        return wire[matrix.HEADER.size+header[-3]:]
    return dict(maximum_scaled_csv_errors=result,checkpoint_payload_bitwise_equal=payload(left)==payload(right))


def field(path,count=8):
    text=path.read_text();match=re.search(r'internalField\s+uniform\s+([^;]+);',text)
    if match:return [float(match[1])]*count
    match=re.search(r'internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
    assert match,(path,'expected scalar field')
    values=[float(x) for x in re.findall(NUMBER,match[2])]
    assert len(values)==int(match[1])==count;return values


def scalar_sum(path,count=8):return math.fsum(field(path,count))/count


def audit_chmt(case,closed):
    out=case/'chmtOutput';steps=sorted(int(p.stem.split('-')[1]) for p in out.glob('gas-*.csv'))
    result=dict(snapshots=len(steps),closed_domain=closed,max_mass_residual=0.,max_energy_residual=0.,max_species_sum_residual=0.,max_packet_energy_residual=0.,max_gas_exchange_residual=0.,max_material_exchange_residual=0.)
    initial=None;previous=None;initial_components=None;component_residuals={}
    for step in steps:
        g=physical_rows(out/f'gas-{step}.csv');s=physical_rows(out/f'material-{step}.csv');exchange=physical_rows(out/f'exchange-{step}.csv')
        assert all(math.isfinite(v) for row in g+s+exchange for v in row.values())
        gm=math.fsum(r['mass'] for r in g);sm=math.fsum(v for r in s for k,v in r.items() if k.startswith('mass_'))
        ge=math.fsum(r['total_energy'] for r in g);se=math.fsum(r['total_energy'] for r in s)
        current=(gm+sm,ge+se,ge,se)
        if initial is None:initial=current
        components={key:math.fsum(row.get(key,0.) for row in g+s) for key in set().union(*(row.keys() for row in g+s)) if key.startswith('mass_')}
        if initial_components is None:initial_components=components
        for key,value in components.items():
            residual=abs(value-initial_components[key]);component_residuals[key]=max(component_residuals.get(key,0.),residual)
            if closed:assert residual<=1e-10*max(1.,abs(initial_components[key])),(case,'closed nonreacting component',key,residual)
        for row in g:
            residual=abs(math.fsum(v for k,v in row.items() if k.startswith('mass_'))-row['mass'])
            result['max_species_sum_residual']=max(result['max_species_sum_residual'],residual)
            assert residual<=1e-10+1e-10*abs(row['mass'])
        for row in exchange:
            assert row['consumer_mask']==3,(case,'packet not consumed by both participants')
            residual=abs(row['energy']-math.fsum(row[k] for k in ('conductive','advective','pressure_work','viscous_work','radiation','liquid_kinetic_advection')))
            result['max_packet_energy_residual']=max(result['max_packet_energy_residual'],residual)
            assert residual<=1e-10+1e-10*abs(row['energy'])
        mr=abs(current[0]-initial[0]);er=abs(current[1]-initial[1])
        result['max_mass_residual']=max(result['max_mass_residual'],mr)
        result['max_energy_residual']=max(result['max_energy_residual'],er)
        if closed:
            assert mr<=1e-10*max(1.,initial[0]),(case,'closed mass',mr)
            assert er<=max(2e-5,2e-13*abs(initial[1])),(case,'closed energy',er)
            if previous is not None:
                transferred=math.fsum(r['energy'] for r in exchange)
                gr=abs(ge-previous[2]-transferred);sr=abs(se-previous[3]+transferred)
                result['max_gas_exchange_residual']=max(result['max_gas_exchange_residual'],gr)
                result['max_material_exchange_residual']=max(result['max_material_exchange_residual'],sr)
                assert max(gr,sr)<=max(2e-5,2e-13*abs(initial[1])),(case,'exchange energy closure',gr,sr)
        previous=current
    result['initial_inventory']=initial;result['final_inventory']=previous;result['component_mass_residuals']=component_residuals
    return result


def audit_cht(case,grid=2):
    # Fixed, closed, uniform unit-volume contact fixture. This independent
    # inventory check is only applied to its contact/mixture/pure descendants.
    states=[];gas_wall_errors=[];solid_wall_errors=[];initial_species=None;species_errors={}
    for snapshot in matrix.times(case):
        ge=scalar_sum(snapshot/'rhoE',grid**3);se=1e6*scalar_sum(snapshot/'solid/T',grid**3);mass=scalar_sum(snapshot/'rho',grid**3)
        state=dict(time=float(snapshot.name),gas_energy=ge,solid_energy=se,total_energy=ge+se,mass=mass)
        species={p.name:scalar_sum(p,grid**3) for p in snapshot.glob('rhoY_*')}
        if initial_species is None:initial_species=species
        for key,value in species.items():
            residual=abs(value-initial_species[key]);species_errors[key]=max(species_errors.get(key,0.),residual)
            assert residual<1e-10,(case,key,residual)
        if states:
            heat=(snapshot/'gasConvectiveWallHeatFlux').read_text()
            patch=re.search(r'\bleft\s*\{([^}]+)\}',heat,re.S)[1]
            match=re.search(r'value\s+uniform\s+('+NUMBER+r')\s*;',patch)
            if match:flux=float(match[1])
            else:
                match=re.search(r'value\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;',patch,re.S)
                values=[float(x) for x in re.findall(NUMBER,match[2])];assert len(values)==int(match[1])==grid**2
                flux=math.fsum(values)/grid**2
            transfer=flux*(state['time']-states[-1]['time'])
            gas_wall_errors.append(abs(ge-states[-1]['gas_energy']+transfer))
            solid_wall_errors.append(abs(se-states[-1]['solid_energy']-transfer))
        states.append(state)
    assert len(states)>=2
    er=max(abs(s['total_energy']-states[0]['total_energy']) for s in states)
    mr=max(abs(s['mass']-states[0]['mass']) for s in states)
    assert er<2e-5,(case,'CHT closed energy',er)
    assert mr<1e-10,(case,'CHT closed mass',mr)
    assert max(gas_wall_errors+solid_wall_errors)<2e-5,(case,'CHT interface exchange closure',gas_wall_errors,solid_wall_errors)
    return dict(states=states,max_energy_residual=er,max_mass_residual=mr,max_gas_exchange_residual=max(gas_wall_errors),max_solid_exchange_residual=max(solid_wall_errors),species_mass_residuals=species_errors,scope='between written coupled endpoints')


def compare_fields(left,right):
    a,b=matrix.times(left),matrix.times(right);assert len(a)==len(b)
    maximum={}
    for x,y in zip(a,b):
        assert float(x.name)==float(y.name)
        # Reuse the unchanged existing numerical oracle (relative error < 2e-12).
        errors=matrix.compare_fields(x,y)
        for key,value in errors.items():maximum[key]=max(value,maximum.get(key,0.))
    return dict(maximum_scaled_field_errors=maximum)


def prepare(name,case,meta,grid):
    source=matrix.HERE/'fixtures'/name
    if grid==2:
        matrix.prepare(source,case,meta)
        return
    assert name in ('cht-contact','chmt-mixture'),('uniform refinement requires a uniform fixture',name)
    matrix.copy_fixture(source,case)
    for filename in ('blockMeshDict','blockMeshSolidDict'):
        path=case/'system'/filename
        text,count=re.subn(r'\(2 2 2\)',f'({grid} {grid} {grid})',path.read_text())
        assert count==1,(path,'uniform block dimensions');path.write_text(text)
    matrix.run(['blockMesh','-case',case],case/'mesh.log')
    matrix.run(['blockMesh','-case',case,'-region','solid','-dict',case/'system/blockMeshSolidDict'],case/'mesh-solid.log')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--candidate',type=Path)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--grid',type=int,default=2,help='uniform grid per coordinate, for contact/mixture scale tests')
    parser.add_argument('--cases',nargs='+',default=DEFAULT_CASES)
    parser.add_argument('--end-time',type=float,help='override only total duration for an amortized end-to-end measurement')
    parser.add_argument('--benchmark',action='store_true',help='extend gas-wave/cht-contact/chmt-mixture using existing benchmark time windows')
    parser.add_argument('--repeats',type=int,default=0,help='timed repetitions following one warmup; zero runs one validation pair')
    args=parser.parse_args();assert args.grid>=2;args.work.mkdir(parents=True,exist_ok=False)
    models=json.loads((matrix.HERE/'fixtures.json').read_text())
    report=dict(grid=args.grid,binaries={},cases={},runs=[],timing_scope='native process elapsed; mesh excluded; startup and normal output included')
    for kind,folder in [('baseline',args.baseline),('candidate',args.candidate)]:
        if folder:
            report['binaries'][kind]={name:digest(folder/name) for name in ('CHMT','CHT','gasUGKP') if (folder/name).exists()}
    for name in args.cases:
        meta=models[name];solver=meta['solver'];reference=None;records=[];samples={}
        kinds=['baseline']+(['candidate'] if args.candidate else [])
        for repeat in range(args.repeats+1):
            for kind in (kinds if repeat%2==0 else list(reversed(kinds))):
                case=args.work/f'{name}-{kind}-{repeat}'
                prepare(name,case,meta,args.grid)
                if args.benchmark:
                    durations={'gas-wave':(.001,1e-5),'cht-contact':(.0004,1e-5),'chmt-mixture':(.005,1e-5)}
                    assert name in durations,('benchmark duration unspecified',name)
                    end,dt=durations[name];control=case/'system/controlDict'
                    matrix.entry(control,'endTime',end);matrix.entry(control,'deltaT',dt)
                    if solver!='CHMT':
                        matrix.entry(control,'maxDeltaT',dt)
                        # Keep every coupled endpoint available for budget checks.
                        if solver=='gasUGKP':matrix.entry(control,'writeInterval',end/2)
                if args.end_time is not None:
                    assert args.end_time>0
                    matrix.entry(case/'system/controlDict','endTime',args.end_time)
                binary=(args.baseline if kind=='baseline' else args.candidate)/solver
                elapsed=matrix.run([binary,'-case',case],case/'run.log')
                if reference is None:reference=case
                parity=compare_chmt(reference,case) if solver=='CHMT' else compare_fields(reference,case)
                audit={}
                if solver=='CHMT':audit=audit_chmt(case,not name.startswith('chmt-moving'))
                elif name in ('cht-contact','cht-mixture','cht-pure'):audit=audit_cht(case,args.grid)
                log=(case/'run.log').read_text()
                def scalar_entry(path,key):
                    found=re.search(r'\b'+key+r'\s+('+NUMBER+r')\s*;',path.read_text())
                    return float(found[1]) if found else None
                configuration=dict(end_time=scalar_entry(case/'system/controlDict','endTime'),delta_t=scalar_entry(case/'system/controlDict','deltaT'),grid=args.grid)
                if solver=='CHMT':
                    configuration.update(gas_max_dt=scalar_entry(case/'constant/chmtProperties','gasMaxDt'),coupling_interval=scalar_entry(case/'constant/chmtProperties','couplingInterval'),accepted_windows=audit['snapshots']-1)
                warnings=dict(foam_warning_count=log.count('FOAM Warning'),wall_waveform_mismatch=log.count('single-pass wall waveform/regime mismatch'),solid_iteration_budget=log.count('solid nonlinear iteration budget exhausted'))
                row=dict(case=name,kind=kind,repeat=repeat,warmup=repeat==0,elapsed_seconds=elapsed,parity=parity,audit=audit,configuration=configuration,warnings=warnings)
                if repeat:samples.setdefault(kind,[]).append(elapsed)
                records.append(row);report['runs'].append(row)
                print(json.dumps(dict(case=name,kind=kind,repeat=repeat,elapsed_seconds=elapsed)),flush=True)
                (args.work/'results.json').write_text(json.dumps(report,indent=2)+'\n')
        medians={k:statistics.median(v) for k,v in samples.items()}
        summary=dict(samples=samples,median_seconds=medians)
        if 'candidate' in medians:summary['speedup']=medians['baseline']/medians['candidate']
        report['cases'][name]=summary
        (args.work/'results.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
