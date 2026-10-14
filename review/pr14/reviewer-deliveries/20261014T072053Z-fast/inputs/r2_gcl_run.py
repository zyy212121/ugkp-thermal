import shutil,json,collections,decimal
from r2_runner import *
from r2_native import compare_dirs
decimal.getcontext().prec=80
def dec(x):return decimal.Decimal.from_float(x)
for ns,name in [(2,'moving-GCL'),(10,'moving-chemistry')]:
 original=ROOT/'work/cases/extra'/name;case=ROOT/'work/geometry-cases'/name;shutil.copytree(D/'inputs/native/extra'/name,case)
 for rel in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(original/rel,case/rel)
 b=ROOT/'work/fixtures'/('CHMT-geometry-'+str(ns));t=run('ACTUAL-ALL-MICROGEOMETRY-'+name,cmd(b,'-case',case),[8,10,14,26,29],expected='Allactualaccepted microsteps bothrealgeometry stages; fulloutputbytes unchanged toformal original; pairedwithNs10actualchemicalarguments.',binary=b,variant='diagnostic-geometry-D202',kind='DIAGNOSTIC')
 records={};raw=(D/t['log']).read_text()
 for line in raw.splitlines():
  if not line.startswith('GCL2_'):continue
  x=line.split(',');key=tuple(map(int,x[1:4]));q=records.setdefault(key,{'window':key[0],'micro':key[1],'stage':key[2],'cells':{},'faces':{}})
  if x[0]=='GCL2_META':q.update(begin_hex=x[4],end_hex=x[5],interval_hex=x[6])
  elif x[0]=='GCL2_CELL':q['cells'][int(x[4])]=x[5:]
  elif x[0]=='GCL2_FACE':q['faces'][int(x[4])]=x[5]
 mesh=json.loads((D/'artifacts/extra'/name/'checkpoint-0-decoded.json').read_text())['gasMesh'];residuals=[]
 for q in records.values():
  residual=[]
  for c,vals in q['cells'].items():
   sweep=sum((dec(float.fromhex(q['faces'][mesh['cellFaces'][slot]]))*decimal.Decimal(mesh['cellFaceSigns'][slot]) for slot in range(mesh['cellFaceOffsets'][c],mesh['cellFaceOffsets'][c+1])),decimal.Decimal(0));a,bv=map(float.fromhex,vals[:2]);residual.append(float(dec(bv)-dec(a)-sweep))
  q['gcl_cell_residuals']=residual;q['maximum_gcl_abs']=max(map(abs,residual));residuals+=residual
 cmp=compare_dirs(original/'chmtOutput',case/'chmtOutput');cp=ROOT/'work/cases/extra'/name/'chmtOutput';vs=[]
 if ns==10:
  v=json.loads((D/'artifacts/extra'/name/'actual-kernel-arguments-S206.json').read_text())['volume_inputs'];ordered=sorted(records)
  # eachacceptedEuler microstep launches chemical pre/post stages in the sameorder.
  microkeys=sorted({(k[0],k[1]) for k in ordered})
  if len(v)==2*len(microkeys):
   for i,k in enumerate(microkeys):
    old=records[k+(0,)];end=records[k+(1,)];a=v[2*i];b=v[2*i+1]
    va=[old['cells'][c][0] for c in sorted(old['cells'])];vb=[end['cells'][c][1] for c in sorted(end['cells'])]
    vs.append({'window':k[0],'micro':k[1],'first_launch':a['sequence'],'second_launch':b['sequence'],'old_volume_bits_exact':a['volume_hex']==va,'end_volume_bits_exact':b['volume_hex']==vb,'same_half_dt':a['dt_hex']==b['dt_hex'],'volume_changes':va!=vb})
 report={'microstep_count':len(records)//2,'geometry_stage_count':len(records),'all_cell_stage_rows':list(records.values()),'maximum_gcl_abs_m3':max(map(abs,residuals)),'alloutput_comparison':cmp,'chemical_volume_pairs':vs,'chemical_allpairs_exact':bool(vs) and all(x['old_volume_bits_exact'] and x['end_volume_bits_exact'] and x['same_half_dt'] for x in vs)};p=D/'artifacts/extra'/name/'all-micro-GCL-and-chemical-volumes.json';p.write_text(json.dumps(report,indent=2)+'\n');good=cmp['all_files_byte_equal'] and len(records)>0 and max(map(abs,residuals))<=1e-12
 if ns==10:good=good and report['chemical_allpairs_exact']
 assertion(t,good,'Actual'+str(len(records)//2)+' microsteps/'+str(len(records))+' geometry stages; GCLmax='+str(report['maximum_gcl_abs_m3'])+' m3; chemicalpairs='+str(len(vs))+' allold/new bitexact='+str(report['chemical_allpairs_exact'])+' outputsbytesexact='+str(cmp['all_files_byte_equal']),artifacts=[str(p.relative_to(D))]);print(name,t['id'],good,report['maximum_gcl_abs_m3'],len(vs),report['chemical_allpairs_exact'],vs[:2],flush=True)
