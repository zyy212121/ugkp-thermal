from r3_runner import *
import re
b=ROOT/'work/fixtures/chemistry-owner-chronology-S304'
t=run('CHEMISTRY-OWNER-CHRONOLOGY-S304',cmd(b),[10,12,14,26],300,'Chronologicalrealbegin/source/validate/rollback/commit trace forfouroriginalpositivecontrols; no16matrixrepeat.',inputs=[D/'inputs/chemistry-owner-chronology-S304.cu',D/'patches/S304-readonly-owner-chronology.patch'],binary=b,variant='DIAGNOSTIC_S304')
raw=(D/t['log']).read_text();rows=[q for q in raw.splitlines() if q.startswith('R3TRIAL')]
original=(D/'logs/CHEMISTRY-FORMAL-FIXTURE.log').read_text();a=original.split('PASS failed-cell')[0].strip();b='\n'.join(q for q in raw.splitlines() if not q.startswith('R3TRIAL') and not q.startswith('PASS real CUDA')).strip()
comparison={'same_complete_original_owner_semantic_outputs':a==b,'chronological_rows':rows,'source':SOURCE,'original_fixture_result':'CHEMISTRY-FORMAL-FIXTURE','classification':'ReadonlyIO observation; flags/intervals/semanticassertions same; no matrixduplicate.'}
(D/'artifacts/chemistry-owner-chronology.json').write_text(json.dumps(comparison,indent=2)+'\n')
assertion(t,t['exit_code']==0 and a==b and any('event=rollback' in q for q in rows) and any('stepLimits=1' in q for q in rows),'ActualchronologyandALLoriginalsemanticownerstdoutbyteequal afterpureobserverlines='+str(a==b),artifacts=['artifacts/chemistry-owner-chronology.json'])
print('\n'.join(rows),flush=True)
