from r3_runner import *
import decimal,re
decimal.getcontext().prec=80
def dec(x):return decimal.Decimal.from_float(x) if isinstance(x,float) else decimal.Decimal(x)
z=dec(0);raw=(D/'logs/WET-FILM-OBSERVER-D302.log').read_text();stages={};fluxes={}
for l in raw.splitlines():
 a=l.split(',')
 if a[0]=='LEDGER_STAGE':stages[tuple(map(int,a[1:4]))]=float.fromhex(a[4])
 if a[0]=='LEDGER_FINAL_FLUX':fluxes.setdefault(tuple(map(int,a[1:4])),[]).append({'face':int(a[4]),'neighbour':int(a[5]),'kind':int(a[6]),'values':[float.fromhex(q) for q in a[7:14]]})
states=[json.loads((D/'artifacts/WET-FILM-FORMAL'/f'checkpoint-{i}-full-decoded.json').read_text()) for i in range(3)];interface=set(states[0]['surface']['gasFace']);rows=[]
for w in [1,2]:
 a,b=states[w-1:w+1];keys=sorted(k for k in stages if k[0]==w);assert len(keys)==50
 boundary=[z]*7
 for k in keys:
  address=json.loads((D/'artifacts/WET-FILM-FORMAL'/f'checkpoint-{w}-address-decoded.json').read_text())['surface'];assert len(fluxes[k])==len(address['gasFaceIds'])
  for f in fluxes[k]:
   if f['neighbour']>=0 or f['face'] in interface:continue
   for j,v in enumerate(f['values']):boundary[j]+=dec(v)*dec(stages[k])
 packets=[q for q in b['ledger'] if q['kind'] in [0,1,3,4]];expected=[sum((dec(q['mass']) for q in packets),z),sum((dec(q['energy']) for q in packets),z)]+[sum((dec(q['momentum'][j]) for q in packets),z) for j in range(3)]+[sum((dec(q['species'][j]) for q in packets),z) for j in range(2)]
 def total(s):return [sum((dec(q['mass']) for q in s['gas']),z),sum((dec(q['energy']) for q in s['gas']),z)]+[sum((dec(q['momentum'][j]) for q in s['gas']),z) for j in range(3)]+[sum((dec(q['species'][j]) for q in s['gas']),z) for j in range(2)]
 before,after=total(a),total(b);res=[after[j]-before[j]+boundary[j]-expected[j] for j in range(7)];tols=[dec(1e-10 if j==1 else 1e-14)+dec(1e-10)*max(abs(before[j]),abs(after[j])) for j in range(7)]
 rows.append({'window':w,'actual_stage_count':len(keys),'actual_ledger_duration_sum':str(sum((dec(stages[k]) for k in keys),z)),'actual_weighted_boundary_M_E_P_species':list(map(str,boundary)),'actual_current_window_gas_packets_M_E_P_species':list(map(str,expected)),'physical_state_residual_M_E_P_species':list(map(str,res)),'original_tolerances':list(map(str,tols)),'allfields_pass':all(abs(r)<=tol for r,tol in zip(res,tols))})
p=D/'artifacts/wet-actual-final-stage-gas-physical-budget.json';p.write_text(json.dumps({'source':SOURCE,'actual_final_flux_and_ledgerdt_source':'logs/WET-FILM-OBSERVER-D302.log','actual_interface_index':sorted(interface),'rows':rows,'all_pass':all(q['allfields_pass'] for q in rows),'scope':'Actualfinalweighted Euler faceflux andactual currentmacro packets with80digit exactconversion; whole gas physicalM/E/vectorP/species. Originaldoublecumulativeaudit red retained, globalfilmmechanicalchannel stillopen.'},indent=2)+'\n');t=run('AUDIT-WET-ACTUAL-FINAL-FLUX-PHYSICAL-GAS',['python3','-c','print("Originalactualfinalweightedfaceflux/ledgerdt/packet/checkpoint gasM-E-P-species independent80digit audit retained")'],[3,6,13,14,20,21,26,29],300,'OriginalphysicalgasbulkM/E/vectorP/species, no tolerance expansion, doubleaudit/globalfilmseparate.',inputs=[p,Path(__file__)],kind='ANALYSIS');assertion(t,all(q['allfields_pass'] for q in rows),'Actual50stages/window fullphysicalgas stock/finalweightedflux/paired ledger closure='+str(all(q['allfields_pass'] for q in rows))+'; fullglobalfilmP notclosed',artifacts=[str(p.relative_to(D))]);print(json.dumps(rows,indent=2))
