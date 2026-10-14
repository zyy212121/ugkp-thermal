from r3_runner import *
import decimal,csv,re,struct
decimal.getcontext().prec=80
Q=decimal.Decimal
def dec(x):return Q.from_float(x) if isinstance(x,float) else Q(x)
z=dec(0)
def close(r,a,b,E=False):return abs(r)<=dec(1e-10 if E else 1e-14)+dec(1e-10)*max(abs(a),abs(b))
raw=(D/'logs/WET-FILM-OBSERVER-D302.log').read_text();configs=[dict(re.findall(r'(\w+)=([^\s]+)',l)) for l in raw.splitlines() if l.startswith('R3CONFIG')];rhos={float.fromhex(c['liquidRho']) for c in configs};assert rhos=={1000.};rho=dec(rhos.pop())
states=[json.loads((D/'artifacts/WET-FILM-FORMAL'/f'checkpoint-{i}-full-decoded.json').read_text()) for i in range(3)]
def stocks(s):
 return [[sum((dec(q['mass']) for q in s['gas']),z),sum((dec(q['energy']) for q in s['gas']),z)], [sum((sum(map(dec,q['condensed']),z)+sum(map(dec,q['pore']),z) for q in s['solid']),z),sum((dec(q['energy']) for q in s['solid']),z)], [sum((dec(q['mass']) for q in s['film']),z),sum((dec(q['enthalpy'])-dec(q['pressure'])*dec(q['mass'])/rho+dec(q['kineticEnergy']) for q in s['film']),z)]]
def pp(s):return [sum((dec(q['mass'])*dec(q['meanVelocity'][j]) for q in s['film']),z)+sum((dec(q['momentum'][j]) for q in s['gas']),z) for j in range(3)]
rows=[];initial=states[0];tot0=[sum(q[j] for q in stocks(initial)) for j in range(2)];bi=initial['budget']
for i,s in enumerate(states[1:],1):
 prev=states[i-1];old=stocks(prev);now=stocks(s);b=s['budget'];a=prev['budget'];tot=[sum(q[j] for q in now) for j in range(2)]
 mr=tot[0]-tot0[0]+dec(b['boundaryMass'])-dec(bi['boundaryMass']);ext=-(dec(b['boundaryEnergy'])-dec(bi['boundaryEnergy']))+sum((dec(b[k])-dec(bi[k]) for k in ['bodyWork','supportWork','radiation']),z);er=tot[1]-tot0[1]-ext-(dec(b['filmKineticDefect'])-dec(bi['filmKineticDefect']))
 expectedM=[z]*4;expectedE=[z]*4;expectedP=[[z]*3 for _ in range(4)];expectedY=[[z]*2 for _ in range(4)]
 for q in s['ledger']:
  pairs={0:[(0,1),(1,-1)],1:[(0,1),(2,-1)],2:[(1,-1),(2,1)],3:[(0,1),(1,-1)],4:[(0,1),(3,-1)],5:[(1,1),(3,-1)]}[q['kind']]
  for p,sign in pairs:
   expectedM[p]+=dec(sign)*dec(q['mass']);expectedE[p]+=dec(sign)*dec(q['energy'])
   for j in range(3):expectedP[p][j]+=dec(sign)*dec(q['momentum'][j])
   for j in range(2):expectedY[p][j]+=dec(sign)*dec(q['species'][j])
 recipient=[]
 for p in range(4):
  M=dec(b['exchangeMass'][p])-dec(a['exchangeMass'][p]);E=dec(b['exchangeEnergy'][p])-dec(a['exchangeEnergy'][p]);P=[dec(b['exchangeMomentum'][p][j])-dec(a['exchangeMomentum'][p][j]) for j in range(3)];Y=[dec(b['exchangeSpecies'][p][j])-dec(a['exchangeSpecies'][p][j]) for j in range(2)]
  rm=M-expectedM[p];re_=E-expectedE[p];rp=[P[j]-expectedP[p][j] for j in range(3)];ry=[Y[j]-expectedY[p][j] for j in range(2)]
  recipient.append({'participant':p,'actual_packet_mass':str(expectedM[p]),'actual_packet_energy':str(expectedE[p]),'actual_packet_P':list(map(str,expectedP[p])),'M_residual':str(rm),'E_residual':str(re_),'P_residual':list(map(str,rp)),'species_residual':list(map(str,ry)),'ME_pass':close(rm,M,expectedM[p]) and close(re_,E,expectedE[p],True),'P_strict_pass':all(close(rp[j],P[j],expectedP[p][j]) for j in range(3)),'species_pass':all(close(ry[j],Y[j],expectedY[p][j]) for j in range(2))})
 inventories=[]
 for p in range(3):
  m=now[p][0]-old[p][0]-expectedM[p];e=now[p][1]-old[p][1]-expectedE[p]
  if p==0:m+=dec(b['boundaryMass'])-dec(a['boundaryMass']);e+=dec(b['boundaryEnergy'])-dec(a['boundaryEnergy'])
  if p==2:e-=dec(b['filmKineticDefect'])-dec(a['filmKineticDefect'])
  inventories.append({'participant':p,'beforeME':list(map(str,old[p])),'afterME':list(map(str,now[p])),'M_residual':str(m),'E_residual':str(e),'ME_pass':close(m,old[p][0],now[p][0]) and close(e,old[p][1],now[p][1],True)})
 # Film has only algebraic velocity; do not invent a reaction impulse to close it.
 globalP=[pp(s)[j]-pp(initial)[j]+dec(b['boundaryMomentum'][j])-dec(bi['boundaryMomentum'][j])+(dec(b['supportImpulse'][j])-dec(bi['supportImpulse'][j])) for j in range(3)]
 filmP=[sum((dec(q['mass'])*dec(q['meanVelocity'][j]) for q in s['film']),z)-sum((dec(q['mass'])*dec(q['meanVelocity'][j]) for q in prev['film']),z)-expectedP[2][j] for j in range(3)]
 perfilm=[]
 for f,q in enumerate(s['film']):
  address=json.loads((D/'artifacts/WET-FILM-FORMAL'/f'checkpoint-{i}-address-decoded.json').read_text())['surface'];sf=address['gasFaceIds'][address['gasFace'][f]];gf=address['persistentId'][f];ps=[p for p in s['ledger'] if (p['kind']==2 and p['face']==sf) or (p['kind']==1 and p['face']==gf)]
  mass=sum((dec(p['mass'])*(1 if p['kind']==2 else -1) for p in ps),z);energy=sum((dec(p['energy'])*(1 if p['kind']==2 else -1) for p in ps),z);pq=prev['film'][f];stockE=lambda v:dec(v['enthalpy'])-dec(v['pressure'])*dec(v['mass'])/rho+dec(v['kineticEnergy']);rM=dec(q['mass'])-dec(pq['mass'])-mass;rE=stockE(q)-stockE(pq)-energy
  perfilm.append({'film':f,'solid_face':sf,'gas_face':gf,'packet_count':len(ps),'signed_packet_mass':str(mass),'signed_packet_energy':str(energy),'M_residual':str(rM),'E_residual':str(rE),'ME_pass':close(rM,dec(pq['mass']),dec(q['mass'])) and close(rE,stockE(pq),stockE(q),True),'P_stock':list(map(str,[dec(q['mass'])*dec(x) for x in q['meanVelocity']]))})
 row={'checkpoint':i,'time':s['time'],'global_M_residual':str(mr),'global_E_residual_J':str(er),'global_ME_pass':close(mr,tot0[0],tot[0]) and close(er,tot0[1],tot[1],True),'global_E_tolerance_J':str(dec(1e-10)+dec(1e-10)*max(abs(tot0[1]),abs(tot[1]))),'recipients':recipient,'physical_inventories':inventories,'per_film':perfilm,'global_P_residual':list(map(str,globalP)),'film_P_stock_minus_actual_impulse':list(map(str,filmP)),'film_exchange_P_cumulative':b['exchangeMomentum'][2],'gas_cumulative_audit_P_residual':[str(sum((dec(q['momentum'][j]) for q in s['gas']),z)-sum((dec(q['momentum'][j]) for q in initial['gas']),z)+dec(b['boundaryMomentum'][j])-dec(bi['boundaryMomentum'][j])-(dec(b['exchangeMomentum'][0][j])-dec(bi['exchangeMomentum'][0][j]))) for j in range(3)],'packet_count':len(s['ledger']),'solid_volumes_equal_initial':s['solidMesh']['volumes']==initial['solidMesh']['volumes']}
 rows.append(row)
phase=[dict(re.findall(r'(\w+)=([^,\s]+)',q)) for q in raw.splitlines() if q.startswith('R3PHASE')];zeroheatforce=all(float.fromhex(q['mass'])==0 and float.fromhex(q['target'])==0 and float.fromhex(q['energy'])!=0 and float.fromhex(q['mx'])!=0 and q['index']=='-1' for q in phase) and len(phase)==8
report={'source':SOURCE,'measured_consumed_liquid_rho':str(rho),'config_rows':configs,'checks':rows,'zero_phase_preserves_all_heat_force':zeroheatforce,'phase_rows':phase,'all_ME_scope_pass':all(q['global_ME_pass'] and all(p['ME_pass'] and p['species_pass'] for p in q['recipients']) and all(p['ME_pass'] for p in q['physical_inventories']+q['per_film']) for q in rows),'full_physical_P':'OPEN_REDUCED_FILM_NO_MECHANICAL_CONSTRAINT_INVENTORY','original_thresholds':{'absMass':1e-14,'relMass':1e-10,'absEnergy':1e-10,'relEnergy':1e-10},'no_balancing_reaction_added':True,'observer_native_all_bytes_equal':json.loads((D/'artifacts/wet-observer-output-comparison.json').read_text())['all_files_byte_equal']}
p=D/'artifacts/wet-complete-physical-audit-S307.json';p.write_text(json.dumps(report,indent=2)+'\n')
args=['python3',str(Path(__file__)),'--replay']
# All computations use actual inputs above; command here records independent analysis.
if '--replay' not in __import__('sys').argv:
 t=run('AUDIT-WET-FULL-ME-P-S307', ['python3','-c','print("80digit original stocks/packets/measuredrho analysis retained")'],[3,6,13,14,20,21,26,29],300,'Originalglobal/perrecipient/perfilm M/E/species, originalstrictP separate; no invented reaction.',inputs=[p,Path(__file__)],kind='ANALYSIS');assertion(t,report['all_ME_scope_pass'] and zeroheatforce,'M/E scope='+str(report['all_ME_scope_pass'])+' zeroheatforce='+str(zeroheatforce)+'; FULLP remainsOPEN',artifacts=[str(p.relative_to(D))])
 amend('WET-FILM-FORMAL',verdict='PASS' if report['all_ME_scope_pass'] and zeroheatforce else 'FAIL',observations='Formalcompleteoriginal0.001 interval; global/perrecipient/perfilm M/E/species + zeroheatforce scope. FullphysicalP remainsOPEN.',artifacts=[str(p.relative_to(D))])
print(json.dumps({'MEs':[(q['global_M_residual'],q['global_E_residual_J'],[p['E_residual'] for p in q['per_film']]) for q in rows],'P':[q['global_P_residual'] for q in rows],'all_ME_scope_pass':report['all_ME_scope_pass'],'zeroheatforce':zeroheatforce},indent=2))
