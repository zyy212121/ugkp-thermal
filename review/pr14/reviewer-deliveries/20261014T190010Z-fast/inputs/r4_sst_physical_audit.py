from r4_runner import *
import decimal,sys,math
decimal.getcontext().prec=80
Q=decimal.Decimal;Z=Q(0)
def d(x):return Q.from_float(float.fromhex(x)) if isinstance(x,str) else Q.from_float(x) if isinstance(x,float) else Q(x)
def faceids(r,c):return r['cellFaceId'][r['cellPlaneStart'][c]:r['cellPlaneStart'][c]+r['cellPlaneCount'][c]]
def walls(r,c):return [f for f in faceids(r,c) if f>=r['ni'] and r['riemannBoundaryKind'][f]==2]
def audit(name):
 log=D/'logs'/('D402-GPU-'+name+'.log');maximum={};fail=[];counts={'Euler':0,'blend':0,'SST_cell_stages':0,'unconstrained_omega':0,'wall_projections':0,'active_floors':0};weighted={};before=sst=flux=blend=None;sources=[];stages=[]
 def check(kind,c,res,a,b,e=False):
  maximum[kind]=max(maximum.get(kind,Z),abs(res));tol=d(1e-10 if e else 1e-14)+d(1e-10)*max(abs(a),abs(b))
  if abs(res)>tol:fail.append({'stage':counts['Euler'],'cell':c,'field':kind,'residual':str(res),'tolerance':str(tol),'expected':str(a),'actual':str(b)})
 def projected(r,c,field,raw):
  rho=max(d(r['rho'][c]),d(r['rhoMin']));floor=rho*d(r['sstKMin' if field=='rhoK' else 'sstOmegaMin']);floored=max(raw,floor);counts['active_floors']+=floored!=raw
  if field=='rhoOmega' and walls(r,c):
   assert r['sstWallTreatment']==0
   # Input/source invariant: every tested kind2 wall T policy is zeroGradient and all owner slots disabled.
   y=max(d(r['sstWallDistance'][c]),d(1e-300));target=Q(6)*(d(r['gasMu'])/rho)/(d(r['coeff_beta1'])*y*y)
   counts['wall_projections']+=1;return rho*max(target,d(r['sstOmegaMin']))
  return floored
 for line in log.open():
  if line.startswith('R4_SST_SOURCE '):sources.append(json.loads(line[14:]));continue
  if not line.startswith('R4_OBS '):continue
  r=json.loads(line[7:]);tag=r['tag']
  if tag=='EULER_BEFORE':before=r;sources=[]
  elif tag=='SST_BEFORE':sst=r
  elif tag=='FLUX_BEFORE':
   flux=r;assert len(sources)==r['nc'];src={s['cell']:s for s in sources}
   for c in range(r['nc']):
    s=src[c];counts['SST_cell_stages']+=1;counts['unconstrained_omega']+=not bool(s['constrainedOmega']);v0=d(s['oldVolume']);v1=d(s['newVolume']);dt=d(s['dt']);rho=d(sst['rho'][c]);k=d(sst['k'][c]);omega=d(sst['omega'][c]);div=d(s['divU'])
    for fld,phi,source in [('rhoK','fluxK','sourceK'),('rhoOmega','fluxOmega','sourceOmega')]:
     a=d(sst[fld][c])*v0;b=d(r[fld][c])*v1;delta=dt*(d(s[phi])+v0*d(s[source])) if fld=='rhoK' or not s['constrainedOmega'] else Z;check('Euler_'+fld,c,b-a-delta,a,b,fld=='rhoK')
    A=[[d(sst['gradU'+i+j][c]) for j in ['xX','xY','xZ']] for i in []]
    A=[[d(sst['gradU'+i+j][c]) for j in ['X','Y','Z']] for i in ['x','y','z']]
    tr=sum((A[i][i] for i in range(3)),Z);s2=Q(2)*sum((((A[i][j]+A[j][i])/Q(2))**2 for i in range(3) for j in range(3)),Z);G=sum(((A[i][j]+A[j][i]-(Q(2)/Q(3)*tr if i==j else Z))*A[i][j] for i in range(3) for j in range(3)),Z)
    coeff=lambda n:d(sst['coeff_'+n]);f1=d(sst['sstF1'][c]);f2=d(sst['sstF2'][c]);gamma=f1*(coeff('gamma1')-coeff('gamma2'))+coeff('gamma2');beta=f1*(coeff('beta1')-coeff('beta2'))+coeff('beta2')
    pk=rho*min(d(sst['nut'][c])*G,coeff('c1')*coeff('betaStar')*k*omega);dk=rho*coeff('betaStar')*k*omega;ck=rho*Q(2)/Q(3)*div*k
    dot=sum((d(sst['gradK'+j][c])*d(sst['gradOmega'+j][c]) for j in ['X','Y','Z']),Z);cd=Q(2)*coeff('alphaOmega2')*dot/max(omega,d(1e-300));limit=coeff('c1')/coeff('a1')*coeff('betaStar')*omega*max(coeff('a1')*omega,coeff('b1')*f2*max(s2,Z).sqrt())
    po=rho*gamma*min(G,limit);do=rho*beta*omega*omega;co=rho*Q(2)/Q(3)*gamma*div*omega;xo=rho*(Q(1)-f1)*cd
    check('gross_K_net_source',c,d(s['sourceK'])-(pk-dk-ck),pk-dk-ck,d(s['sourceK']),True);check('gross_omega_net_source',c,d(s['sourceOmega'])-(po-do-co+xo),po-do-co+xo,d(s['sourceOmega']))
    for key,x in [('K_production',pk),('K_destruction',dk),('K_compression_signed',ck),('omega_production',po),('omega_destruction',do),('omega_compression_signed',co),('omega_cross_signed',xo)]:weighted[key]=weighted.get(key,Z)+d(r['ledgerDt'])*v0*x
  elif tag=='EULER_AFTER':
   counts['Euler']+=1
   for c in range(r['nc']):
    for fld in ['rhoK','rhoOmega']:
     predicted=projected(r,c,fld,d(flux[fld][c]));actual=d(r[fld][c]);check('Euler_recover_'+fld,c,actual-predicted,predicted,actual,fld=='rhoK')
    v0=d(flux['oldVolume' if r['moving'] else 'V'][c]);v1=d(flux['newVolume' if r['moving'] else 'V'][c]);dt=d(r['dt'])
    for fld,phi in [('rho','gasPhiRho'),('rhoE','gasPhiRhoE'),('rhoUx','gasPhiRhoUx'),('rhoUy','gasPhiRhoUy'),('rhoUz','gasPhiRhoUz')]:
     signed=sum(((Q(-1) if flux['faceOwner'][f]==c else Q(1))*d(flux[phi][f]) for f in faceids(flux,c)),Z);a=d(flux[fld][c])*v0;b=d(r[fld][c])*v1;check('gas_Euler_'+fld,c,b-a-dt*signed,a,b,fld=='rhoE')
   stages.append({'time':r['time'],'dt':r['dt'],'ledgerDt':r['ledgerDt'],'rk':r['rk']})
  elif tag=='BLEND_BEFORE':blend=r
  elif tag=='BLEND_AFTER':
   counts['blend']+=1;w0=d(r['w0']);w1=d(r['w1'])
   for c in range(r['nc']):
    for fld,initial in [('rho','rhoNext'),('rhoE','rhoENext'),('rhoUx','rhoUxNext'),('rhoUy','rhoUyNext'),('rhoUz','rhoUzNext'),('rhoK','rhoKInitial'),('rhoOmega','rhoOmegaInitial')]:
     raw=w0*d(blend[initial][c])+w1*d(blend[fld][c]);predicted=projected(r,c,fld,raw) if fld in ['rhoK','rhoOmega'] else raw;actual=d(r[fld][c]);check('RK_'+fld,c,actual-predicted,predicted,actual,fld in ['rhoE','rhoK'])
 return {'source_commit':SOURCE,'case':name,'actual_log':str(log.relative_to(D)),'log_sha256':sha(log),'counts':counts,'maximum_abs_residuals':{k:str(v) for k,v in maximum.items()},'independent_actual_weighted_gross_sources':{k:str(v) for k,v in weighted.items()},'stages':stages,'failures':fail,'pass':not fail and counts['Euler']>0,'original_criteria':{'absMass':1e-14,'absEnergy':1e-10,'rel':1e-10},'limits':'Actual per-stage cell Euler/net/gross source and RK plus independent lowRe omega wall/floor recovery, nine tested T-zeroGradient/owner-disabled cases only. Compression uses observed actual finalized-flux divU. Does not prove unrecorded geometry-interval divU reconstruction, selected numerical wall owners, rollback/graph, global macro SST constraints or three-application gross budgets. D401 naive raw omega blend failures retained.'}
if '--compute' in sys.argv:
 n=sys.argv[-1];q=audit(n);p=D/'artifacts'/(n+'-D402-SST-physical.json');p.write_text(json.dumps(q,indent=2)+'\n');print(json.dumps({k:v for k,v in q.items() if k!='stages'}),flush=True);raise SystemExit(0 if q['pass'] else 1)
else:
 for n in ['SST-RK','SST-SSPRK2','SST-SSPRK3']+[f'SST64-RK{rk}-{geo}' for rk in [1,2,3] for geo in ['static','moving']]:
  t=run('ANALYZE-D402-'+n,cmd('python3',Path(__file__),'--compute',n),[8,9,14,26,29],expected='Actual Euler/RK/gross SST sources and independent lowRe wall omega constraint under unchanged physical criteria.',inputs=[Path(__file__),D/'logs'/('D402-GPU-'+n+'.log')],kind='ANALYSIS',variant='SUPPORT_ANALYSIS_D402');assertion(t,t['exit_code']==0,'Independent actual physical stage scope; wider owner/global channels remain open.',artifacts=['artifacts/'+n+'-D402-SST-physical.json'])
