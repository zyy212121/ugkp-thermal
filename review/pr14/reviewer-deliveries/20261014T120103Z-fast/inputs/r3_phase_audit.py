from r3_runner import *
import re,decimal
decimal.getcontext().prec=80
def dec(x):return decimal.Decimal.from_float(x) if isinstance(x,float) else decimal.Decimal(x)
z=dec(0)
def close(r,a,b,E=False):return abs(r)<=dec(1e-10 if E else 1e-14)+dec(1e-10)*max(abs(a),abs(b))
binary=ROOT/'work/fixtures/checkpoint-addressing-S307';reports=[]
for name in ['onecell-melting-S308','onecell-solidification-S308']:
 folder=D/'artifacts'/('PHASE-FORMAL-'+name);states=[]
 for cp in sorted(p for p in folder.glob('checkpoint-*') if p.is_dir()):
  t=run('DECODE-PHASE-'+name+'-'+cp.name,cmd(binary,cp),[3,5,6,13,14,20,21,26,29],300,'Actualnativecodec hostASanUBSan for fullnonzero phaseGPUinterval.',inputs=[cp/'state.chmt'],binary=binary,kind='HOST_SANITIZER',variant='SUPPORT_S307');assert t['exit_code']==0;s=json.loads((D/t['log']).read_text());states.append(s);(cp.parent/(cp.name+'-decoded.json')).write_text(json.dumps(s,indent=2)+'\n');assertion(t,True,'Actualnativecodec/no observedASanUBSan diagnostics, physicalaudit separate')
 assert len(states)==2;raw=(D/'logs'/('PHASE-D302-'+name+'.log')).read_text();phases=[dict(re.findall(r'(\w+)=([^,\s]+)',q)) for q in raw.splitlines() if q.startswith('R3PHASE')];configs=[dict(re.findall(r'(\w+)=([^\s]+)',q)) for q in raw.splitlines() if q.startswith('R3CONFIG')];rho=float.fromhex(configs[-1]['liquidRho']);a,b=states
 def stock(s):return [sum((dec(q['mass']) for q in s['gas']),z)+sum((sum(map(dec,q['condensed']),z)+sum(map(dec,q['pore']),z) for q in s['solid']),z)+sum((dec(q['mass']) for q in s['film']),z),sum((dec(q['energy']) for q in s['gas']+s['solid']),z)+sum((dec(q['enthalpy'])-dec(q['pressure'])*dec(q['mass'])/dec(rho)+dec(q['kineticEnergy']) for q in s['film']),z)]
 before=stock(a);after=stock(b);ab=a['budget'];bb=b['budget'];mr=after[0]-before[0]+dec(bb['boundaryMass'])-dec(ab['boundaryMass']);external=-(dec(bb['boundaryEnergy'])-dec(ab['boundaryEnergy']))+sum((dec(bb[k])-dec(ab[k]) for k in ['bodyWork','supportWork','radiation']),z);er=after[1]-before[1]-external-(dec(bb['filmKineticDefect'])-dec(ab['filmKineticDefect']))
 phasepackets=[q for q in b['ledger'] if q['kind']==2];assert len(phasepackets)==1;mass=phasepackets[0]['mass'];last=phases[-1];assert float.fromhex(last['mass'])==mass
 target=float.fromhex(last['target']);porosity=float.fromhex(last['porosity']);materialrho=float.fromhex(last['rho']);expected=0.;expected-=mass/materialrho/(1-porosity);bits=target.hex()==expected.hex();sign=mass>0 if 'melting' in name else mass<0
 balances=[]
 for participant in range(4):
  M=E=z;P=[z]*3
  for q in b['ledger']:
   pairs={0:[(0,1),(1,-1)],1:[(0,1),(2,-1)],2:[(1,-1),(2,1)],3:[(0,1),(1,-1)],4:[(0,1),(3,-1)],5:[(1,1),(3,-1)]}[q['kind']]
   for p,sg in pairs:
    if p!=participant:continue
    M+=dec(sg)*dec(q['mass']);E+=dec(sg)*dec(q['energy'])
    for j in range(3):P[j]+=dec(sg)*dec(q['momentum'][j])
  actualM=dec(bb['exchangeMass'][participant])-dec(ab['exchangeMass'][participant]);actualE=dec(bb['exchangeEnergy'][participant])-dec(ab['exchangeEnergy'][participant]);actualP=[dec(bb['exchangeMomentum'][participant][j])-dec(ab['exchangeMomentum'][participant][j]) for j in range(3)];balances.append({'participant':participant,'packet_M':str(M),'packet_E':str(E),'packet_P':list(map(str,P)),'M_residual':str(actualM-M),'E_residual':str(actualE-E),'P_residual':[str(actualP[j]-P[j]) for j in range(3)],'ME_pass':close(actualM-M,actualM,M) and close(actualE-E,actualE,E,True),'P_audit_pass':all(close(actualP[j]-P[j],actualP[j],P[j]) for j in range(3))})
 globalP=[sum((dec(q['momentum'][j]) for q in b['gas']),z)-sum((dec(q['momentum'][j]) for q in a['gas']),z)+sum((dec(q['mass'])*dec(q['meanVelocity'][j]) for q in b['film']),z)-sum((dec(q['mass'])*dec(q['meanVelocity'][j]) for q in a['film']),z)+dec(bb['boundaryMomentum'][j])-dec(ab['boundaryMomentum'][j])+dec(bb['supportImpulse'][j])-dec(ab['supportImpulse'][j]) for j in range(3)]
 cmp=json.loads((D/'artifacts'/('phase-'+name+'-before-after.json')).read_text());passed=sign and bits and close(mr,before[0],after[0]) and close(er,before[1],after[1],True) and all(q['ME_pass'] for q in balances) and cmp['formal_before_after']['all_files_byte_equal'] and cmp['observer_compare']['all_files_byte_equal']
 report={'name':name,'source':SOURCE,'old_safe_source':PATHS['old_source'],'complete_end':b['time'],'recorded_consumed_config':configs,'actual_phase_candidates':phases,'final_committed_phase_packet':phasepackets[0],'actual_phase_mass':mass,'desired_direction_measured':sign,'target_hex':target.hex(),'original_expression_target_hex':expected.hex(),'nonzero_expression_bit_identical':bits,'solid_volume_initial':a['solidMesh']['volumes'],'solid_volume_final':b['solidMesh']['volumes'],'solid_recession_volume_residual':str(sum(map(dec,b['solidMesh']['volumes']),z)-sum(map(dec,a['solidMesh']['volumes']),z)-dec(target)),'global_M_residual':str(mr),'global_E_residual_J':str(er),'recipients':balances,'global_physical_P_residual':list(map(str,globalP)),'full_P_model_qualification':'reduced film has algebraic velocity and no complete mechanicalreaction channel; scope remainsOPEN, no inventedbalancing term','scope_ME_expression_before_after_pass':passed,'before_after_observer_comparison_ref':'phase-'+name+'-before-after.json','material_corrector_candidates_not_doublecounted':True};p=D/'artifacts'/('phase-'+name+'-complete-audit.json');p.write_text(json.dumps(report,indent=2)+'\n');reports.append(report)
 t=run('AUDIT-PHASE-'+name,['python3','-c','print("Actualfinalsignedphase/stock/originalFP64 expression/fullbeforeafter bitcomparison audit retained")'],[3,6,13,14,20,21,26,29],300,'OriginalME criteria/nonzerotarget expression bits andrawbeforeafter unchanged; fullP retainedseparate.',inputs=[p,Path(__file__)],kind='ANALYSIS');assertion(t,passed,'Nativecompletephase direction='+str(sign)+' originalvolumeexpressionbits='+str(bits)+' globalM='+str(mr)+' E='+str(er)+'; fullP remainsqualifiedopen',artifacts=[str(p.relative_to(D))]);print('PHASE_AUDIT',name,passed,mass,mr,er,globalP,flush=True)
(D/'artifacts/nonzero-phase-two-direction-summary.json').write_text(json.dumps(reports,indent=2)+'\n')
