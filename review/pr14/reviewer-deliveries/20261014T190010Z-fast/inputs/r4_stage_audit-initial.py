from r4_runner import *
import decimal,sys,re,csv
decimal.getcontext().prec=80
Q=decimal.Decimal;Z=Q(0)
def dec(x):return Q.from_float(float.fromhex(x)) if isinstance(x,str) else Q.from_float(x) if isinstance(x,float) else Q(x)
def close(r,a,b,E=False):return abs(r)<=dec(1e-10 if E else 1e-14)+dec(1e-10)*max(abs(a),abs(b))
def values(r,k):return [dec(x) for x in r[k]]
def divergence(r,field,c):
 ids=r['cellFaceId'][r['cellPlaneStart'][c]:r['cellPlaneStart'][c]+r['cellPlaneCount'][c]]
 return sum(((Q(-1) if r['faceOwner'][f]==c else Q(1))*dec(r[field][f]) for f in ids),Z)
def audit_case(log):
 name=log.name[len('D401-GPU-'):-4];frames=[];source=[];before=flux=sst=blend=None;checks=[];failures=[];stage_rows=[];sst_rows=[];blend_rows=[]
 maximum={};counts={'Euler':0,'sst_cells':0,'unconstrained_omega_cells':0,'blend':0}
 def observe(kind,field,c,res,a,b,energy=False):
  key=kind+':'+field;maximum[key]=max(maximum.get(key,Z),abs(res));ok=close(res,a,b,energy)
  if not ok:failures.append({'kind':kind,'field':field,'cell':c,'residual':str(res),'before':str(a),'after':str(b),'tolerance':str(dec(1e-10 if energy else 1e-14)+dec(1e-10)*max(abs(a),abs(b)))})
  return ok
 with log.open() as f:
  for line in f:
   if line.startswith('R4_SST_SOURCE '):source.append(json.loads(line[len('R4_SST_SOURCE '):]));continue
   if not line.startswith('R4_OBS '):continue
   r=json.loads(line[len('R4_OBS '):]);tag=r['tag']
   if tag=='EULER_BEFORE':before=r;source=[];sst=None
   elif tag=='SST_BEFORE':sst=r
   elif tag=='FLUX_BEFORE':
    flux=r
    if sst:
     assert len(source)==r['nc'],(name,len(source),r['nc']);sources={q['cell']:q for q in source}
     for c in range(r['nc']):
      q=sources[c];dt=dec(q['dt']);v0=dec(q['oldVolume']);v1=dec(q['newVolume']);counts['sst_cells']+=1;counts['unconstrained_omega_cells']+=not bool(q['constrainedOmega'])
      for field,phi,src in [('rhoK','fluxK','sourceK'),('rhoOmega','fluxOmega','sourceOmega')]:
       a=dec(sst[field][c])*v0;b=dec(r[field][c])*v1
       contribution=dt*(dec(q[phi])+v0*dec(q[src])) if field=='rhoK' or not q['constrainedOmega'] else Z
       observe('SST_Euler',field,c,b-a-contribution,a,b,field=='rhoK')
      destruction=dec(q['rho'])*dec(q['betaStar'])*dec(q['k'])*dec(q['omega']);compression=dec(q['rho'])*Q(2)/Q(3)*dec(q['divU'])*dec(q['k'])
      sst_rows.append({'stage':counts['Euler'],'cell':c,'omega_constrained':q['constrainedOmega'],'actual_net_K_source':q['sourceK'],'actual_net_omega_source':q['sourceOmega'],'K_destruction_from_actual_rho_k_omega':str(destruction),'K_compression_from_actual_divU':str(compression),'K_production_recovered_from_net_source':str(dec(q['sourceK'])+destruction+compression),'recovered_gross_production_limited_by_source_roundoff':True})
   elif tag=='EULER_AFTER':
    assert flux and before;counts['Euler']+=1;dt=dec(r['dt']);v0=values(flux,'oldVolume' if flux['moving'] else 'V');v1=values(flux,'newVolume' if flux['moving'] else 'V')
    fields=[('rho','gasPhiRho'),('rhoE','gasPhiRhoE'),('rhoUx','gasPhiRhoUx'),('rhoUy','gasPhiRhoUy'),('rhoUz','gasPhiRhoUz')]
    for field,phi in fields:
     for c in range(r['nc']):
      a=dec(flux[field][c])*v0[c];b=dec(r[field][c])*v1[c];expected=dt*divergence(flux,phi,c);observe('gas_Euler',field,c,b-a-expected,a,b,field=='rhoE')
    for k in range(flux.get('ns',0)):
     for c in range(r['nc']):
      ids=flux['cellFaceId'][flux['cellPlaneStart'][c]:flux['cellPlaneStart'][c]+flux['cellPlaneCount'][c]]
      df=sum(((Q(-1) if flux['faceOwner'][f]==c else Q(1))*dec(flux['speciesFlux'][k*flux['nf']+f]) for f in ids),Z)
      a=dec(flux['speciesRho'][k*r['nc']+c])*v0[c];b=dec(r['speciesRho'][k*r['nc']+c])*v1[c];observe('gas_Euler','species'+str(k),c,b-a-dt*df,a,b)
    stage_rows.append({'time':r['time'],'dt':r['dt'],'actual_ledgerDt':r['ledgerDt'],'rk':r['rk'],'nc':r['nc'],'moving':r['moving'],'sst':r['sst']})
   elif tag=='BLEND_BEFORE':blend=r
   elif tag=='BLEND_AFTER':
    assert blend;counts['blend']+=1;w0=dec(r['w0']);w1=dec(r['w1'])
    for field,initial in [('rho','rhoNext'),('rhoE','rhoENext'),('rhoUx','rhoUxNext'),('rhoUy','rhoUyNext'),('rhoUz','rhoUzNext')]+([('rhoK','rhoKInitial'),('rhoOmega','rhoOmegaInitial')] if r['sst'] else []):
     for c in range(r['nc']):
      expected=w0*dec(blend[initial][c])+w1*dec(blend[field][c]);actual=dec(r[field][c]);observe('RK_density_blend',field,c,actual-expected,expected,actual,field in ['rhoE','rhoK'])
    blend_rows.append({'w0':r['w0'],'w1':r['w1'],'rk':r['rk'],'moving':r['moving']})
 return {'identity':'PR14_REVIEW_ONLY','source_commit':SOURCE,'case':name,'actual_log':str(log.relative_to(D)),'log_sha256':sha(log),'original_criteria':{'absMass':1e-14,'relMass':1e-10,'absEnergy':1e-10,'relEnergy':1e-10},'counts':counts,'max_abs_residuals':{k:str(v) for k,v in maximum.items()},'failures':failures,'actual_stage_time_dt_and_ledger_weights':stage_rows,'actual_RK_blends':blend_rows,'actual_SST_source_components':sst_rows,'scope_pass':bool(counts['Euler']) and not failures,'limits':'ActualEuler cellstocks/finalflux/netSSTsource/RKweights only. GrossKproduction reconstructedfromnet canlose cancellation digits; grossOmega components/completewallconstraint inventories andthreeappgross remainopen. Nativeoutputinvariance separately proven; noproductionaudit orreservedzero used.'}

if '--compute' in sys.argv:
 results=[]
 for log in sorted((D/'logs').glob('D401-GPU-*.log')):
  q=audit_case(log);p=D/'artifacts'/(q['case']+'-D401-stage-audit.json');p.write_text(json.dumps(q,indent=2)+'\n');results.append({'case':q['case'],'pass':q['scope_pass'],'counts':q['counts'],'maximums':q['max_abs_residuals'],'failure_count':len(q['failures']),'first_failures':q['failures'][:3],'artifact':str(p.relative_to(D))});print('ACTUAL_STAGE_AUDIT',json.dumps(results[-1]),flush=True)
 (D/'artifacts/D401-stage-audit-summary.json').write_text(json.dumps(results,indent=2)+'\n');raise SystemExit(0 if all(q['pass'] for q in results) else 1)
else:
 t=run('ANALYZE-D401-STAGES',cmd('python3',Path(__file__),'--compute'),[8,9,10,14,20,21,26,29],expected='Independent80digitactualdouble conversion cellEuler inventory/finalflux/SSTnet source andRK actualweights underoriginalcriteria; failures kept not productiongates.',inputs=[Path(__file__),D/'patches/D401-readonly-actual-Euler-RK-SST-chemistry.patch'],kind='ANALYSIS',variant='ANALYSIS_ACTUAL_GPU_D401')
 assertion(t,t['exit_code']==0,'Independent actualGPUstageaudit fullresults artifacts/D401-stage-audit-summary.json; globals/packets/remaininggross separate.',artifacts=['artifacts/D401-stage-audit-summary.json'])
