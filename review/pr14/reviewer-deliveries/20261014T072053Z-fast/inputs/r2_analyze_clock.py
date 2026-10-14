import json,struct,collections
from r2_runner import *
out=D/'artifacts/clock-trace';base=json.loads((out/'baseline-output-comparison.json').read_text());print('BASE DIAGNOSTIC DIFF',[(x['path'],x['expected_sha256'],x['actual_sha256']) for x in base['files'] if not x['byte_equal']])
def read(name):
 rows=[]
 for line in (D/'logs'/('TRACE-NATIVE-'+name+'.log')).read_text().splitlines():
  if line.startswith('F003,'):
   p=line.split(',');rows.append({'window':int(p[1]),'epoch':int(p[2]),'micro':int(p[3]),'event':p[4],'index':int(p[5]),'field':p[6],'hex':p[7],'bits':struct.pack('>d',float.fromhex(p[7])).hex()})
 return rows
b=read('baseline');old=read('tested');formal=read('formal')
assert len(b)==len(old)==len(formal)==21400
formal_equal=old==formal;diff=[];byfield=collections.Counter()
for idx,(x,y) in enumerate(zip(b,formal)):
 assert {k:v for k,v in x.items() if k not in ['hex','bits']}=={k:v for k,v in y.items() if k not in ['hex','bits']}
 if x['bits']!=y['bits']:
  diff.append({'record_index':idx,'baseline':x,'formal':y});byfield[(x['event'],x['field'])]+=1
special=[]
for w in [1,2]:
 for micro in [6,7,49,50]:
  for ev,field in [('begin','micro_time'),('begin','micro_dt'),('stage','packet_dt'),('accepted','end')]:
   xx=[x for x in b if x['window']==w and x['micro']==micro and x['event']==ev and x['field']==field];yy=[x for x in formal if x['window']==w and x['micro']==micro and x['event']==ev and x['field']==field]
   special.append({'window':w,'microstep':micro,'event':ev,'field':field,'baseline_hex':[x['hex'] for x in xx],'formal_hex':[x['hex'] for x in yy]})
summary={'actual_records_each':len(b),'tested_vs_formal_all_trace_bits_equal':formal_equal,'first_difference':diff[0],'difference_counts':[{'event':k[0],'field':k[1],'count':v} for k,v in byfield.items()],'actual_dt_samples':special,'first_wall_data_difference':next((x for x in diff if x['baseline']['event'].startswith('wall')),None),'first_gas_difference':next((x for x in diff if x['baseline']['event']=='gas'),None),'first_packet_difference':next((x for x in diff if x['baseline']['event']=='packet'),None),'scope':'Actual21400 host observer records per version, same nativeGPU thermal100steps. Diagnosed firstchanged acceptedendpoint atwindow1step6, later dt-tail/gas/packet atstep50. Wall actualsamples remain exact. This directly explains the scoped F003 inputs; cannot extrapolate all time-varying boundaries/configs.','baseline_output_difference':base}
(out/'first-divergence-analysis.json').write_text(json.dumps(summary,indent=2)+'\n');(out/'all-bit-differences.json').write_text(json.dumps(diff,indent=2)+'\n');print('FIELDCOUNT',summary['difference_counts']);print('test_formal',formal_equal)
