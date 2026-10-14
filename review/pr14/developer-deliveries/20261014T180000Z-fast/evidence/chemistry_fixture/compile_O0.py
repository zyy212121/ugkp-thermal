from pathlib import Path
from datetime import datetime, timezone
import json
import resource
import subprocess
import time

out=Path(__file__).resolve().parent
command=json.loads((out/'command.json').read_text())
command[command.index('-O3')]='-O0'
command[command.index('-o')+1]=str(out/'chemistry_trial-O0')
(out/'O0-command.json').write_text(json.dumps(command,indent=2)+'\n')
started=datetime.now(timezone.utc).isoformat(); start=time.monotonic()
with (out/'O0-native-check-build.log').open('w') as log:
    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
usage=resource.getrusage(resource.RUSAGE_CHILDREN)
record={'command':command,'started_at_utc':started,'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-start,
        'maximum_child_rss_kib':usage.ru_maxrss,
        'child_user_cpu_seconds':usage.ru_utime,'child_system_cpu_seconds':usage.ru_stime,
        'measurement':'Python resource.getrusage(RUSAGE_CHILDREN); max RSS is maximum child-process RSS, not summed simultaneous RSS.'}
(out/'O0-native-check-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
raise SystemExit(result.returncode)
