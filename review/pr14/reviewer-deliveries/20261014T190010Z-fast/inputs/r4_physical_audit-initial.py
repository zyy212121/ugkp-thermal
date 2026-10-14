from r4_runner import *
import re,decimal,sys
decimal.getcontext().prec=80
Q=decimal.Decimal;Z=Q(0)
def dec(x):return Q.from_float(float.fromhex(x)) if isinstance(x,str) else Q.from_float(x) if isinstance(x,float) else Q(x)
def tolerance(a,b,e=False):return dec(1e-10 if e else 1e-14)+dec(1e-10)*max(abs(a),abs(b))
def chemistry(name):
 log=D/'logs'/('D401-GPU-'+name+'.log');groups=[];headers={};species={};frames=[];fail=[];maxima={};rows=[]
 def add(k,r,a,b,e=False):
  maxima[k]=max(maxima.get(k,Z),abs(r))
  if abs(r)>tolerance(a,b,e):fail.append({'group':len(groups),'field':k,'residual':str(r),'tolerance':str(tolerance(a,b,e))})
 v=json.loads((ROOT/'work/cases'/name/'verification.json').read_text());order=v['species'];ew=v['element_moles_per_kg'];nc=v['gas_cells'];ns=len(order)
 for line in log.open():
  for m in re.finditer(r'R4_CHEM_HALF cell=(\d+) dt=(\S+) V=(\S+) rhoE_before=(\S+) rhoE_after=(\S+?)(?=R4_|\s|$)',line):
   cell=int(m[1]);assert cell not in headers;headers[cell]={'cell':cell,'dt':m[2],'V':m[3],'before':m[4],'after':m[5]}
  if line.startswith('R4_CHEM_SPEC '):
   a=dict(k.split('=',1) for k in line.split()[1:]);key=(int(a['cell']),int(a['species']));assert key not in species;species[key]=a
   if len(species)==nc*ns:
    assert len(headers)==nc;group={'volumes':[],'dt':[],'total_species_mass_before':Z,'total_species_mass_after':Z}
    for c in range(nc):
     h=headers[c];vol=dec(h['V']);group['volumes'].append(h['V']);group['dt'].append(h['dt']);a=dec(h['before'])*vol;b=dec(h['after'])*vol;add('rhoE*V',b-a,a,b,True)
     if float.fromhex(h['before']).hex()!=float.fromhex(h['after']).hex():fail.append({'group':len(groups),'cell':c,'field':'rhoE-bit-preservation'})
     before=[];after=[]
     for k in range(ns):
      s=species[c,k];assert s['V']==h['V'] and s['dt']==h['dt'];before.append(dec(s['before'])*vol);after.append(dec(s['after'])*vol)
     a=sum(before,Z);b=sum(after,Z);add('sum_species_mass',b-a,a,b);group['total_species_mass_before']+=a;group['total_species_mass_after']+=b
     for e in range(4):
      a=sum((before[k]*dec(ew[order[k]][e]) for k in range(ns)),Z);b=sum((after[k]*dec(ew[order[k]][e]) for k in range(ns)),Z);add('element_moles_'+str(e),b-a,a,b)
    groups.append(group);headers={};species={}
  if line.startswith('R4_OBS '):
   a=json.loads(line[7:])
   if a['tag']=='EULER_BEFORE':frames.append({'before':a})
   elif a['tag']=='EULER_AFTER':frames[-1]['after']=a
 assert not headers and not species;assert len(groups)==2*len(frames)==1400
 for i,frame in enumerate(frames):
  for half in [0,1]:
   stage=frame['before' if half==0 else 'after'];g=groups[2*i+half];key=('oldVolume' if half==0 else 'newVolume') if stage['moving'] else 'V'
   assert g['volumes']==stage[key],(name,i,half,'actual stage volume mismatch')
   for c in range(nc):
    dt=dec(g['dt'][c]);assert dt==dec(stage['dt'])/Q(2)
  rows.append({'microstep':i,'dt':frame['before']['dt'],'first_half_volume':groups[2*i]['volumes'],'second_half_volume':groups[2*i+1]['volumes']})
 return {'source_commit':SOURCE,'case':name,'actual_gpu_log':str(log.relative_to(D)),'log_sha256':sha(log),'chemical_launches':len(groups),'cell_half_steps':len(groups)*nc,'species_rows':len(groups)*nc*ns,'actual_matched_stage_volume_and_half_dt':True,'maximum_abs_residuals':{k:str(v) for k,v in maxima.items()},'original_criteria':{'absMass':1e-14,'absEnergy':1e-10,'rel':1e-10},'failures':fail,'actual_stage_volumes':rows,'pass':not fail,'limits':'Ns10 physical species total/element-moles and exact rhoE preservation on both chemical half volumes; chemistry changes individual species, not a claim of unchanged individual species. No production ChemistryAudit/legacy zero used.'}
def gas_budget(name):
 folder=D/'artifacts'/name;states=[json.loads(p.read_text()) for p in sorted(folder.glob('checkpoint-*-physical-decoded.json'),key=lambda p:int(p.name.split('-')[1]))];assert len(states)>=2
 log=D/'logs'/('D401-GPU-'+name+'.log');boundary=[[Z]*7 for _ in states];stagecounts=[0 for _ in states];interface=set(states[0]['surface']['gasFace']);times=[dec(q['time']) for q in states]
 for line in log.open():
  if not line.startswith('R4_OBS '):continue
  r=json.loads(line[7:])
  if r['tag']!='FLUX_BEFORE':continue
  time=dec(r['time']);w=next((i for i in range(1,len(times)) if time<times[i] and time>=times[i-1]),None)
  assert w is not None,(name,time,times);stagecounts[w]+=1;dt=dec(r['ledgerDt'])
  for f in range(r['nf']):
   if r['faceNeighbour'][f]>=0 or f in interface:continue
   vals=[r[k][f] for k in ['gasPhiRho','gasPhiRhoE','gasPhiRhoUx','gasPhiRhoUy','gasPhiRhoUz']]+[r['speciesFlux'][k*r['nf']+f] for k in range(2)]
   for j,x in enumerate(vals):boundary[w][j]+=dec(x)*dt
 rows=[]
 def total(s):return [sum((dec(q['mass']) for q in s['gas']),Z),sum((dec(q['energy']) for q in s['gas']),Z)]+[sum((dec(q['momentum'][j]) for q in s['gas']),Z) for j in range(3)]+[sum((dec(q['species'][j]) for q in s['gas']),Z) for j in range(2)]
 for w in range(1,len(states)):
  a,b=states[w-1:w+1];before,after=total(a),total(b);packets=[q for q in b['ledger'] if q['kind'] in [0,1,3,4]]
  expected=[sum((dec(q['mass']) for q in packets),Z),sum((dec(q['energy']) for q in packets),Z)]+[sum((dec(q['momentum'][j]) for q in packets),Z) for j in range(3)]+[sum((dec(q['species'][j]) for q in packets),Z) for j in range(2)]
  residuals=[after[j]-before[j]+boundary[w][j]-expected[j] for j in range(7)];tols=[tolerance(before[j],after[j],j==1) for j in range(7)]
  rows.append({'window':w,'actual_Euler_stages':stagecounts[w],'before_stock_M_E_Px_Py_Pz_S0_S1':list(map(str,before)),'after_stock':list(map(str,after)),'weighted_final_external_flux':list(map(str,boundary[w])),'consumed_gas_packet_totals':list(map(str,expected)),'residuals':list(map(str,residuals)),'original_tolerances':list(map(str,tols)),'pass':all(abs(r)<=t for r,t in zip(residuals,tols))})
 return {'source_commit':SOURCE,'case':name,'log_sha256':sha(log),'actual_interface_faces':sorted(interface),'rows':rows,'pass':all(q['pass'] for q in rows),'limits':'Complete actual gas physical M/E/vectorP/species plus boundary final flux with actual ledgerDt and macro packet. This does not close global film/support momentum, pore/particle inventory, or three-application gross budgets. Legacy Budget/ChemistryAudit slots ignored.'}
if '--compute' in sys.argv:
 name=sys.argv[-1];q=chemistry(name) if '--chem' in sys.argv else gas_budget(name);p=D/'artifacts'/(name+('-chemical-half-physical.json' if '--chem' in sys.argv else '-gas-physical-budget.json'));p.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:v for k,v in q.items() if k not in ['actual_stage_volumes','rows']}),flush=True);raise SystemExit(0 if q['pass'] else 1)
else:
 for mode,names in [('--chem',['chemistry-full','moving-chemistry']),('--gas',['thermal-full','wet-film','SST-RK','SST-SSPRK2','SST-SSPRK3','moving-GCL','phase-melting','phase-solidification'])]:
  for name in names:
   t=run('ANALYZE-PHYSICAL-'+name,cmd('python3',Path(__file__),'--compute',mode,name),[3,6,8,9,10,13,14,20,21,26,29],expected='Independent actual GPU physical inventory/flux/packet or chemical-half volume; original criteria and full input endpoints retained; no production audit.',inputs=[Path(__file__),D/'logs'/('D401-GPU-'+name+'.log')],kind='ANALYSIS',variant='ANALYSIS_ACTUAL_GPU_D401')
   artifact='artifacts/'+name+('-chemical-half-physical.json' if mode=='--chem' else '-gas-physical-budget.json');assertion(t,t['exit_code']==0,'Actual independent physical scope; full global/gross channels remain separately open.',artifacts=[artifact])
