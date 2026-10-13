import pathlib,re,json,sys,importlib.util,math,numpy as np
from run_fast import ROOT,D
sys.path.insert(0,str(ROOT/'common'))
from thermal_particle_restart_current import iter_chunks
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
gas=module(ROOT/'applications/gasUGKP/tools/ugkp_particle_restart.py','gas_restart_probe')
def field(path,count):
    text=path.read_text();match=re.search(r'internalField\s+uniform\s+(\([^;]+\)|[^;]+);',text)
    if match:
        val=np.fromstring(match[1].strip('() '),sep=' ');return np.tile(val,(count,1)).reshape((count,-1))
    match=re.search(r'internalField\s+nonuniform\s+List<\w+>\s+(\d+)\s*\((.*?)\)\s*;',text,re.S)
    if not match:raise ValueError(str(path))
    val=np.fromstring(match[2].replace('(',' ').replace(')',' '),sep=' ');return val.reshape((int(match[1]),-1))
def particles(path):
    with path.open('rb') as f:head=f.readline().decode()
    chunks=[c for _,c in iter_chunks(path)] if head.startswith(('UGKP_CHT','UGKP_FSH')) else list(gas.iter_restart_chunks(path))
    return {k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
reports=[]
for case in sorted((ROOT/'work/cases/resident').iterdir()):
    times=[]
    for q in case.iterdir():
        try:t=float(q.name)
        except ValueError:continue
        if t>0 and (q/'gpuResidentStrictParticles.dat').is_file():times.append((t,q))
    if not times:continue
    states=[]
    for t,q in sorted(times):
        pa=particles(q/'gpuResidentStrictParticles.dat');nc=4 if 'periodic' in case.name else 10;vol=.25 if nc==4 else .04*.04*.001
        fields={x:field(q/x,nc) for x in ['rho','rhoE','rhoEs','rhoU','rhoUs','epsilonS','T','p'] if (q/x).is_file()}
        pkin=np.sum(pa['pm']*(.5*(pa['pux']**2+pa['puy']**2+pa['puz']**2)+1.5*pa['pTheta']))
        s={'time':t,'particle_count':len(pa['pm']),'particle_mass':float(np.sum(pa['pm'])),'particle_momentum':[float(np.sum(pa['pm']*pa[x])) for x in ['pux','puy','puz']],'particle_kinetic':float(pkin),'particle_temperature_minmax':[float(np.min(pa['pT'])),float(np.max(pa['pT']))],
           'Eulerian_integrals':{k:(np.sum(a,axis=0)*vol).tolist() for k,a in fields.items() if k in ['rho','rhoE','rhoEs','rhoU','rhoUs','epsilonS']},'gas_eps_weighted_mass':float(np.sum(fields['rho'][:,0]*(1-fields['epsilonS'][:,0]))*vol),'gas_eps_weighted_energy':float(np.sum(fields['rhoE'][:,0]*(1-fields['epsilonS'][:,0]))*vol),
           'particle_field_hashes':{k:__import__('hashlib').sha256(a.tobytes()).hexdigest() for k,a in pa.items()},'physical_dtypes':{k:str(a.dtype) for k,a in pa.items()}}
        if 'pStuck' in pa:
            s['stuck_states']=pa['pStuck'].tolist();s['stuck_faces']=pa['pStuckFaceId'].tolist()
            s['cold_history']={k:pa[k].tolist() for k in pa if 'Cold' in k or 'Contact' in k or k=='pThermalEnthalpyResidual'}
        if 'periodic' in case.name and case.name.startswith('CHT'):s['constant_cp_particle_energy']=float(np.sum(pa['pm']*1000*pa['pT']))
        states.append(s)
    reports.append({'case':case.name,'states':states})
(D/'artifacts/resident-analysis.json').write_text(json.dumps(reports,indent=2)+'\n')
for r in reports:
    s=r['states'][-1];print(json.dumps({'case':r['case'],'final':{k:v for k,v in s.items() if k not in ['particle_field_hashes','physical_dtypes','cold_history']},'cold_profile_ranges':{k:[float(np.min(v)),float(np.max(v))] for k,v in s.get('cold_history',{}).items() if 'Enthalpy' in k or 'Frozen' in k}},ensure_ascii=False))
