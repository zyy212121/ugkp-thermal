#!/usr/bin/env python3
"""Read-only fixed-source and retained-observer review. Writes only this deliverable."""
from pathlib import Path
from decimal import Decimal, getcontext
import hashlib,json,re,subprocess
getcontext().prec=80
ROOT=Path('/workspace/scratch/98b6339ec81b')
REPO=ROOT/'pr14_fast_fix_20261014'
OUT=Path(__file__).resolve().parent
SOURCE='9d5be6eff69dbced2fc959d422d3ba419a1bc39e'
EVIDENCE='0aa34bc1b0e370ec99e6107faf83ff7f0ac3e668'
DESIGN=OUT/'REVIEWED_DESIGN.md'
LOG=REPO/'review/pr14/reviewer-deliveries/20261014T120103Z-fast/logs/WET-FILM-OBSERVER-D302.log'
DECODE=ROOT/'pr14_fast4_wet_mechanics/checkpoint_1_full_model_state.json'
FILES=['applications/CHMT/gpu/Backend.cu','applications/CHMT/gpu/GasWallMath.H','applications/CHMT/ablation/InterfaceMath.H','applications/CHMT/ablation/CpuSurfaceInterface.C','applications/CHMT/film/FilmMath.H','applications/CHMT/film/CpuFilmDriver.C','applications/CHMT/materials/CpuMaterialDriver.C','applications/CHMT/coupling/IntervalContracts.H','applications/CHMT/coupling/ExchangeLedger.H','applications/CHMT/core/Types.H','common/GpuGasAdvance.cuh','common/GpuGasHostPolicy.cuh']
def git(*args): return subprocess.check_output(['git','-C',str(REPO),*args])
def sha(b): return hashlib.sha256(b).hexdigest()
def provenance(path):
 b=git('show',SOURCE+':'+path)
 return {'path':path,'git_blob':git('rev-parse',SOURCE+':'+path).decode().strip(),'sha256':sha(b),'bytes':len(b),'url':'https://github.com/zyy212121/ugkp-thermal/blob/'+SOURCE+'/'+path}
def D(s): return Decimal.from_float(float.fromhex(s))
lines=LOG.read_text().splitlines(); state=json.loads(DECODE.read_text())['state']
windows=[]; cumulative=Decimal(0)
for window in (1,2):
 stage=[x.split(',') for x in lines if x.startswith(f'LEDGER_STAGE,{window},')]
 weights={tuple(x[1:4]):D(x[4]) for x in stage}; duration=sum(weights.values(),Decimal(0))
 paired=[x.split(',') for x in lines if x.startswith(f'LEDGER_PAIRED,{window},') and x.split(',')[6]=='1']
 phase=[dict(re.findall(r'(\w+)=([^,\s]+)',x)) for x in lines if x.startswith(f'R3PHASE,window={window},')]
 faces=[]
 for face in range(4):
  top=[x for x in paired if int(x[4])==face]; bottom=next(x for x in phase if int(x['film'])==face)
  area=Decimal.from_float(state['surface']['area'][face]); top_j=sum((D(x[11]) for x in top),Decimal(0)); bottom_j=D(bottom['mx'])
  first_p=D(top[0][11])/(area*weights[tuple(top[0][1:4])])
  faces.append({'film':face,'gas_impulse':str(top_j),'bottom_impulse':str(bottom_j),'difference':str(bottom_j-top_j),'pressure_first':str(first_p),'pressure_mean':str(top_j/(area*duration)),'bottom_minus_first_pressure_rule':str(bottom_j-first_p*area*D(bottom['dt']))})
 gap=sum((Decimal(x['difference']) for x in faces),Decimal(0)); cumulative+=gap
 windows.append({'window':window,'gas_stages':len(stage),'top_records':len(paired),'bottom_records':len(phase),'duration':str(duration),'all_top_mass_zero':all(D(x[9])==0 for x in paired),'all_bottom_mass_zero':all(D(x['mass'])==0 for x in phase),'all_scales_one':all(D(x[7])==1 and D(x[8])==1 for x in paired),'faces':faces,'gap':str(gap),'cumulative_gap':str(cumulative)})
record={'scope':'Independent read-only algebra and retained native observer arithmetic; no solver execution or new PASS claim','source_commit':SOURCE,'evidence_commit':EVIDENCE,'sources':[provenance(p) for p in FILES],'observer_log':{'path':str(LOG),'sha256':sha(LOG.read_bytes()),'equals_fixed_evidence':LOG.read_bytes()==git('show',EVIDENCE+':'+str(LOG.relative_to(REPO)))} ,'decoded_state':{'path':str(DECODE),'sha256':sha(DECODE.read_bytes())},'reviewed_design':{'path':str(DESIGN),'sha256':sha(DESIGN.read_bytes()),'bytes':DESIGN.stat().st_size},'windows':windows,'minimal_metadata_bytes_per_face_record':8,'minimal_metadata_bytes_for_4_faces_50_records':1600,'quadrature_weights':{'Euler':['1'],'SSPRK2':['1/2','1/2'],'SSPRK3':['1/6','1/6','2/3']}}
revision=OUT/'REVIEWED_DESIGN_REVISION.md'
record['reviewed_revision']={'path':str(revision),'sha256':sha(revision.read_bytes()),'review_scope':'Sections 5 to 7: one unscaled EOS-pressure-time scalar, original CPU geometry/shear/energy preserved'}
(OUT/'metrics.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'windows':[{'window':x['window'],'gap':x['gap'],'cumulative_gap':x['cumulative_gap']} for x in windows],'source_files':len(FILES),'observer_matches_fixed_commit':record['observer_log']['equals_fixed_evidence'],'reviewed_design_sha256':record['reviewed_design']['sha256']},ensure_ascii=False))
