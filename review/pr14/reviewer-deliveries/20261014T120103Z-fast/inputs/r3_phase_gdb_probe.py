from r3_runner import *
import re
done={q['id'] for q in json.loads((D/'results.json').read_text())['tests']}
active=[p for p in (D/'logs').glob('PHASE-*.spec.json') if json.loads(p.read_text())['id'] not in done]
assert len(active)==1,[p.name for p in active]
spec=json.loads(active[0].read_text());log=D/'logs'/(spec['id']+'.log');raw=log.read_text();match=re.search(r'^PID\s*:\s*(\d+)',raw,re.M);assert match,raw
pid=int(match.group(1));record={'identity':'PR14_REVIEW_ONLY','utc':utc(),'active_test':spec['id'],'PID':pid,'binary_sha256':spec['binary_sha256'],'reason':'Validselected-materialphasecontrol showsboundednoacceptedprogress; readonlyhoststacktoidentifyCPUversusCUDAwait. No oldunselectedOOBexecution.','command':['/usr/local/cuda/bin/cuda-gdb','--batch','-p',str(pid),'-ex','thread apply all bt 16','-ex','detach'],'timeout_seconds':30,'timing_scope':'Diagnosticattachpause, notperformance; original300s budget remains.','first_tool_resolution_failure':'gdb notonPATH/root; exactCUDAtoolkitcuda-gdb used. No app attach in that failed tool attempt.'}
(D/'metadata/phase-gdb-probe.json').write_text(json.dumps(record,indent=2)+'\n');(P/'r3_debug_pid.json').write_text(json.dumps(record)+'\n');print(json.dumps(record,indent=2))
