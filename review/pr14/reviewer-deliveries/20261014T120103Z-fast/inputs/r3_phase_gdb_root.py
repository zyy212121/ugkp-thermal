from pathlib import Path
import json,subprocess,datetime
P=Path(__file__).resolve().parent;record=json.loads((P/'r3_debug_pid.json').read_text());paths=json.loads((P/'r3_paths.json').read_text());D=Path(paths['delivery']);q=subprocess.run(record['command'],capture_output=True,text=True,timeout=30)
output=D/'logs/phase-selected-readonly-gdb.log';output.write_text(q.stdout+q.stderr);record['exit']=q.returncode;record['actual_output']=str(output);record['ended_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();(D/'metadata/phase-gdb-probe.json').write_text(json.dumps(record,indent=2)+'\n');print(q.stdout[-7000:]);print(q.stderr[-1200:])
