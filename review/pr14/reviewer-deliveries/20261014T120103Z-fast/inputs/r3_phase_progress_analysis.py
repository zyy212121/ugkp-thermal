from r3_runner import *
import re,collections
path=D/'logs/PHASE-OBSERVER-melting-positive.log';lines=path.read_text(errors='replace').splitlines();rows=[]
for line in lines:
 if line.startswith('R3PHASE,'):
  q={k:v for k,v in (x.split('=',1) for x in line.split(',')[1:])};rows.append(q)
report={'identity':'PR14_REVIEW_ONLY','analysed_utc':utc(),'observer_phase_rows':len(rows),'first_rows':rows[:4],'last_rows':rows[-4:],'max_observed_local_time':max((float.fromhex(q['time']) for q in rows),default=None),'dt_min':min((float.fromhex(q['dt']) for q in rows),default=None),'dt_max':max((float.fromhex(q['dt']) for q in rows),default=None),'nonzero_mass_min':min((float.fromhex(q['mass']) for q in rows),default=None),'nonzero_mass_max':max((float.fromhex(q['mass']) for q in rows),default=None),'configuration_records':list(dict.fromkeys(q for q in lines if q.startswith('R3CONFIG'))),'gdb_attempt':'Readonly cuda-gdb attach exceeded30s withno usable backtrace; observed log plateau follows this attach. Isolate observer pause from uninstrumented formal/old300s noncompletion. Original runtimes preserved.','complete_interval_pass':False}
(D/'artifacts/phase-incomplete-progress.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
(D/'metadata/phase-gdb-unavailable-and-timeout.json').write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','attempts':[{'PID':74958,'command':'gdb --batch -p74958 ...','result':'FileNotFoundError gdb, beforeattach'},{'PID':75767,'command':'/usr/local/cuda/bin/cuda-gdb --batch -p75767 -ex thread apply all bt16 -ex detach','result':'external30sTimeoutExpired; subprocess killed debugger, no usable stack; observerafterattachnotunmodifiedruntime'}],'production_source_unchanged':True},indent=2)+'\n')
