import shutil,json,collections
from r2_runner import *
from r2_native import compare_dirs
for name,oldcase in [('SST-SSPRK2',ROOT/'work/cases/extra/SST-SSPRK2'),('SST-SSPRK3',ROOT/'work/cases/extra/SST-SSPRK3'),('moving-GCL',ROOT/'work/cases/extra/moving-GCL'),('dry-ablation',ROOT/'work/cases/extra/dry-ablation')]:
 case=ROOT/'work/ledger-cases'/name;shutil.copytree(D/'inputs/native/extra'/name,case)
 for rel in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(oldcase/rel,case/rel)
 b=ROOT/'work/fixtures/CHMT-actual-ledger2';t=run('ACTUAL-WEIGHTED-LEDGER-'+name,cmd(b,'-case',case),[3,6,8,9,10,13,14,20,21,22,26,29],expected='Actualweighted finalfaceflux/paired packets/gasaccepted inventory under realRK or ALE/ablation; originaloutputs allbitexact.',binary=b,variant='diagnostic-ledger-D203',kind='DIAGNOSTIC')
 raw=(D/t['log']).read_text();stages=[];final=[];paired=[]
 for line in raw.splitlines():
  if line.startswith('LEDGER_STAGE,'):
   x=line.split(',');stages.append({'window':int(x[1]),'micro':int(x[2]),'stage':int(x[3]),'ledger_dt_hex':x[4],'micro_dt_hex':x[5],'raw_dt_hex':x[6],'ratio':float.fromhex(x[4])/float.fromhex(x[5])})
  elif line.startswith('LEDGER_FINAL_FLUX,'):
   x=line.split(',');final.append({'window':int(x[1]),'micro':int(x[2]),'stage':int(x[3]),'face':int(x[4]),'neighbour':int(x[5]),'boundary_kind':int(x[6]),'mass_hex':x[7],'energy_hex':x[8],'momentum_hex':x[9:12],'species_hex':x[12:]})
  elif line.startswith('LEDGER_PAIRED,'):
   x=line.split(',');paired.append({'window':int(x[1]),'micro':int(x[2]),'stage':int(x[3]),'surface_face':int(x[4]),'channel':int(x[5]),'kind':int(x[6]),'limiter_scale_hex':x[7],'weight_hex':x[8],'mass_hex':x[9],'energy_hex':x[10],'momentum_hex':x[11:14]})
 cmp=compare_dirs(oldcase/'chmtOutput',case/'chmtOutput');payload={'stages':stages,'ratio_counts':dict(collections.Counter(str(x['ratio']) for x in stages)),'final_face_fluxes':final,'paired_packets':paired,'alloutput_comparison':cmp,'closure_status':'Independentperparticipant/finalflux/stockaudit pending'};p=D/'artifacts/extra'/name/'actual-weighted-final-fluxes.json';p.write_text(json.dumps(payload,indent=2)+'\n')
 assertion(t,t['exit_code']==0 and cmp['all_files_byte_equal'] and bool(stages),'Actual'+str(len(stages))+' stageweightedfluxsets; alloriginaloutputbyteequal='+str(cmp['all_files_byte_equal'])+' actualratios='+str(payload['ratio_counts'])+'. Independentbalance follows.',artifacts=[str(p.relative_to(D))]);print(name,len(stages),payload['ratio_counts'],cmp['all_files_byte_equal'],flush=True)
