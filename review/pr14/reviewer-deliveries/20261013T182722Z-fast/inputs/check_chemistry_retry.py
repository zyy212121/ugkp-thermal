import json,sys,pathlib
p=pathlib.Path(sys.argv[1]);prefix=sys.argv[2] if len(sys.argv)>2 else 'pr';failures=[];checks=[]
def record(name,actual,expected):
    ok=actual==expected;checks.append({'name':name,'pass':ok,'actual':actual,'expected':expected})
    if not ok:failures.append(name)
for label in ['step-limit','success-control','ordinary']:
    base=json.loads((p/f'CHEM-RETRY-base-{label}.log').read_text());new=json.loads((p/f'CHEM-RETRY-{prefix}-{label}.log').read_text())
    record(label+' full state/audit/status publication',new,base)
public=json.loads((p/f'CHEM-RETRY-{prefix}-public-terminal.log').read_text())
record('public path completes interval',(public['status']['code'],public['audit']['integratedTime'],public['canaries_intact']),(0,2e-6,True))
print(json.dumps({'checks':checks,'failures':failures,'all_fields_compared':True,'comparison':'same input/config actual GPU baseline, exact JSON numeric and integer values'},indent=2))
raise SystemExit(1 if failures else 0)
