#!/usr/bin/env python3
"""One declared continuation process only; called exclusively by the frozen batch."""
from pathlib import Path
import argparse,datetime,json,os,re,subprocess
ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--label',required=True);ap.add_argument('--binary',type=Path,required=True);ap.add_argument('--env',type=Path,required=True)
a=ap.parse_args();root=a.root.resolve();case=root/('case-'+a.label)
log=root/(a.label+'.log');clock=root/(a.label+'.time')
script=root/('run-'+a.label+'.sh')
import shlex
q=lambda x:shlex.quote(str(x))
script.write_text('#!/usr/bin/env bash\nsource '+q(a.env)+'\nexport OMP_NUM_THREADS=4 UGKP_DIAGNOSTIC_WARNINGS=1 CHT_CUDA_STACK_BYTES=65536\ncd '+q(case)+'\nexec /usr/bin/time -v -o '+q(clock)+' '+q(a.binary)+' -case '+q(case)+'\n')
processes=subprocess.check_output(['ps','-eo','pid,etimes,args'],text=True)
(root/(a.label+'-processes-before.txt')).write_text(processes)
# Exclusive window and compiler quiescence are scheduled/reviewed by the parent,
# not a newly introduced automatic admission gate.
with log.open('w') as output:
 result=subprocess.run(['bash',str(script)],stdout=output,stderr=subprocess.STDOUT)
(root/(a.label+'.exit')).write_text(str(result.returncode)+'\n')
text=log.read_text(errors='replace')
report={'label':a.label,'exit':result.returncode,'log':str(log),'gnu_time':str(clock),
 'warnings':[line for line in text.splitlines() if any(x in line.lower() for x in ('warning','budget exhausted','residual','converged','precision'))],
 'autotune':[line for line in text.splitlines() if any(x in line for x in ('B1cell','B1face','block','autotun'))],
 'statistics_limit':'One serial default-autotuned run; no repeated-run interval. WSL supervisor monotonic time is not the performance metric.'}
(root/(a.label+'-execution.json')).write_text(json.dumps(report,indent=2)+'\n')
raise SystemExit(result.returncode)
