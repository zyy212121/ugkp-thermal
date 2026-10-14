from r3_runner import *
import re
for rel in ['applications/CHMT/gpu/Backend.cu','applications/CHMT/CHMT.C','applications/CHMT/gpu/Backend.H']:
 p=ROOT/rel
 if not p.exists():continue
 lines=p.read_text().splitlines()
 for i,l in enumerate(lines):
  if any(s in l for s in ['currentCo','Tgas','uploadState','lastAcceptedDt','nextDt']):
   lo=max(0,i-3);hi=min(len(lines),i+8);print('MATCH',rel,i+1,'\n'+'\n'.join(f'{j+1}: {lines[j]}' for j in range(lo,hi)))
print('THERMALCASE_DIRS',[p.name for p in (OLDROOT/'work/cases/chmt').iterdir()])
print('MODELFILES',[str(p.relative_to(ROOT)) for p in (ROOT/'applications/CHMT').rglob('*') if p.name in ['IntervalAudit.H','ModelConfig.H']])
print('OLD_BUDGET_script references',[q['alias'] for q in json.loads((OLD/'metadata/immutable-references.json').read_text())['references'] if 'thermal-full' in q['alias'] and '/inputs/' in '/'+q['alias']][:20])
