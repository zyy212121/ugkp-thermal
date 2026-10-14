import json,decimal,collections,math
from r2_runner import *
original_run=run
def run(testid,*args,**kwargs):
 return next((x for x in json.loads((D/'results.json').read_text())['tests'] if x['id']==testid),None) or original_run(testid,*args,**kwargs)
decimal.getcontext().prec=80
def dec(x):return decimal.Decimal.from_float(x) if isinstance(x,float) else decimal.Decimal(x)
zero=dec(0)
def sums(state):
 g=[sum((dec(q['mass']) for q in state['gas']),zero),sum((dec(q['energy']) for q in state['gas']),zero)]
 m=[sum((sum(map(dec,q['condensed']),zero)+sum(map(dec,q['pore']),zero) for q in state['solid']),zero),sum((dec(q['energy']) for q in state['solid']),zero)]
 f=[sum((dec(q['mass']) for q in state.get('film',[])),zero),sum((dec(q['enthalpy'])-dec(q['pressure'])*dec(q['mass'])/dec(1000)+dec(q['kineticEnergy']) for q in state.get('film',[])),zero)]
 return g,m,f
def close(res,before,after,energy=False):return abs(res)<=dec(1e-10 if energy else 1e-14)+dec(1e-10)*max(abs(before),abs(after))
allreports={}
for name in ['wet-film-S209-D204']:
 folder=D/'artifacts/extra'/name;b=ROOT/'work/fixtures/checkpoint-ASan-S207-2';states=[]
 for cp in sorted(p for p in folder.glob('checkpoint-*') if p.is_dir()):
  t=run('HOST-DECODE-S207-'+name+'-'+cp.name,cmd(b,cp),[5,6,8,9,14,19,22,26,28],expected='Actual unchanged currentcheckpointreader hostASan/UBSan; fullactualGPU/CPUcommitted stocks, semanticpackets/speciesledgers.',inputs=[cp/'state.chmt',D/'inputs/checkpoint_actual_reader_S207.cpp'],binary=b,kind='HOST_SANITIZER')
  raw=(D/t['log']).read_text();s=json.loads(raw);states.append(s);(folder/(cp.name+'-S207-decoded.json')).write_text(json.dumps(s,indent=2)+'\n');assertion(t,t['exit_code']==0,'Actual codec hostASan/UBSan no observed diagnostics; expandedphysicalmetadata retained, notfull caseacceptance.')
 if len(states)<2:
  allreports[name]={'state_count':len(states),'verdict':'FAIL','reason':'Actualnative SIGFPE before completedwindow. Initialcodec read is not numerical acceptance.'};continue
 initial=states[0];checks=[];beforeTotal=[sum(x[j] for x in sums(initial)) for j in range(2)]
 for i,s in enumerate(states[1:],1):
  prev=states[i-1];old=sums(prev);now=sums(s);budget=s['budget'];prevB=prev['budget'];initB=initial['budget'];curTotal=[sum(x[j] for x in sums(s)) for j in range(2)]
  mr=curTotal[0]-beforeTotal[0]+dec(budget['boundaryMass'])-dec(initB['boundaryMass']);external=-(dec(budget['boundaryEnergy'])-dec(initB['boundaryEnergy']))+sum((dec(budget[k])-dec(initB[k]) for k in ['bodyWork','supportWork','radiation']),zero);er=curTotal[1]-beforeTotal[1]-external-(dec(budget['filmKineticDefect'])-dec(initB['filmKineticDefect']))
  expectedM=[zero]*4;expectedE=[zero]*4;expectedP=[[zero]*3 for _ in range(4)];expectedY=[[zero]*len(s['gas'][0]['species']) for _ in range(4)]
  for q in s['ledger']:
   kind=q['kind'];pairs={0:[(0,1),(1,-1)],1:[(0,1),(2,-1)],2:[(1,-1),(2,1)],3:[(0,1),(1,-1)],4:[(0,1),(3,-1)],5:[(1,1),(3,-1)]}[kind]
   for p,sign in pairs:
    expectedM[p]+=dec(sign)*dec(q['mass']);expectedE[p]+=dec(sign)*dec(q['energy'])
    for j in range(3):expectedP[p][j]+=dec(sign)*dec(q['momentum'][j])
    for j in range(len(expectedY[p])):expectedY[p][j]+=dec(sign)*dec(q['species'][j])
  rows=[]
  for p in range(4):
   mass=dec(budget['exchangeMass'][p])-dec(prevB['exchangeMass'][p]);energy=dec(budget['exchangeEnergy'][p])-dec(prevB['exchangeEnergy'][p]);rM=mass-expectedM[p];rE=energy-expectedE[p];rP=[dec(budget['exchangeMomentum'][p][j])-dec(prevB['exchangeMomentum'][p][j])-expectedP[p][j] for j in range(3)];rY=[dec(budget['exchangeSpecies'][p][j])-dec(prevB['exchangeSpecies'][p][j])-expectedY[p][j] for j in range(len(expectedY[p]))]
   passed=close(rM,mass,expectedM[p]) and close(rE,energy,expectedE[p],True) and all(close(v,expectedP[p][j]+v,expectedP[p][j]) for j,v in enumerate(rP)) and all(close(v,expectedY[p][j]+v,expectedY[p][j]) for j,v in enumerate(rY));rows.append({'participant':p,'mass_residual':str(rM),'energy_residual_J':str(rE),'momentum_residual':list(map(str,rP)),'species_residual':list(map(str,rY)),'pass':passed})
  sst=[]
  for j,tag in enumerate(['K','Omega']):
   oldsst=sum((dec(q[j]) for q in initial['sst']),zero);newsst=sum((dec(q[j]) for q in s['sst']),zero);a=s['gasSstAudit'];prediction=sum((dec(a[k+tag])-dec(initial['gasSstAudit'][k+tag]) for k in ['transport','source','constraint']),zero);r=newsst-oldsst-prediction;sst.append({'variable':tag,'initial':str(oldsst),'current':str(newsst),'transport':a['transport'+tag],'source':a['source'+tag],'constraint':a['constraint'+tag],'residual':str(r),'pass':close(r,oldsst,newsst,j==0)})
  gasmass=now[0][0]-old[0][0]+dec(budget['boundaryMass'])-dec(prevB['boundaryMass'])-(dec(budget['exchangeMass'][0])-dec(prevB['exchangeMass'][0]));gasE=now[0][1]-old[0][1]+dec(budget['boundaryEnergy'])-dec(prevB['boundaryEnergy'])-(dec(budget['exchangeEnergy'][0])-dec(prevB['exchangeEnergy'][0]))
  gasP=[]
  for j in range(3):
   a=sum((dec(q['momentum'][j]) for q in prev['gas']),zero);v=sum((dec(q['momentum'][j]) for q in s['gas']),zero);r=v-a+dec(budget['boundaryMomentum'][j])-dec(prevB['boundaryMomentum'][j])-(dec(budget['exchangeMomentum'][0][j])-dec(prevB['exchangeMomentum'][0][j]));gasP.append({'residual':str(r),'pass':close(r,a,v)})
  row={'checkpoint':i,'time':s['time'],'global_mass_residual':str(mr),'global_physical_energy_residual_J':str(er),'global_pass':close(mr,beforeTotal[0],curTotal[0]) and close(er,beforeTotal[1],curTotal[1],True),'actual_perrecipient_packets':rows,'all_paired_participants_pass':all(x['pass'] for x in rows),'SST':sst,'gas_mass_residual':str(gasmass),'gas_energy_residual_J':str(gasE),'gas_momentum':gasP,'gas_inventory_pass':close(gasmass,old[0][0],now[0][0]) and close(gasE,old[0][1],now[0][1],True) and all(x['pass'] for x in gasP),'solid_energy_delta_J':str(now[1][1]-old[1][1]),'film_energy_delta_J':str(now[2][1]-old[2][1]),'film_mass_delta':str(now[2][0]-old[2][0]),'actual_budget':budget};checks.append(row)
 report={'source':SOURCE,'checks':checks,'all_checks_pass':all(x['global_pass'] and x['all_paired_participants_pass'] and x['gas_inventory_pass'] and all(y['pass'] for y in x['SST']) for x in checks),'criteria':'OriginalTolerances absMass1e-14 relMass1e-10 absEnergy1e-10 relEnergy1e-10; momentum uses existingmass tolerance as originalpacket checker.80digit independent sums, perparticipant on actual finalized currentmacro packets. Fullfilmphysical H-pM/rho+KE included; rigid-solid fullmomentum notclaimed.','limitations':'Actualfinal-weighted faceflux pointaudit separate; allthree residentapps budgetgap remains; configured2species/elements empty in syntheticmaterialinputs.'};p=folder/'complete-independent-budget-audit.json';p.write_text(json.dumps(report,indent=2)+'\n');allreports[name]=report
  # native completion exit0 remains scoped by independent closure.
 test='EXTRA-'+name;amend(test,verdict='PASS' if report['all_checks_pass'] else 'FAIL',observations='Actual completeGPU/CPU shortcase originalcriteria independentglobal/perrecipient/GASmass-E-P/SST budget PASS='+str(report['all_checks_pass']),artifacts=['artifacts/extra/'+name+'/complete-independent-budget-audit.json'])
 print(name,report['all_checks_pass'],[(x['global_physical_energy_residual_J'],x['gas_energy_residual_J'],[y['residual'] for y in x['SST']],x['all_paired_participants_pass']) for x in checks],flush=True)
(D/'artifacts/extra/complete-budgets-S207.json').write_text(json.dumps(allreports,indent=2)+'\n')
