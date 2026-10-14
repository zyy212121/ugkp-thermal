from r4_runner import *
original=D/'inputs/r4_physical_audit-initial.py'
# The initial interpreter failures and attempted-trial residuals remain in their raw logs/artifacts.
initial=(P/'r4_physical_audit.py').read_text().replace("assert float.fromhex(g['dt'][c]).hex()==(float.fromhex(stage['dt'])*.5).hex()","dt=dec(g['dt'][c]);assert dt==dec(stage['dt'])/Q(2)").replace(' stageindex=0;w=None\n','').replace("  if stageindex%r['rk']==0:\n   time=dec(r['time']);w=next((i for i in range(1,len(times)) if time<times[i] and time>=times[i-1]),None)\n  stageindex+=1","  time=dec(r['time']);w=next((i for i in range(1,len(times)) if time<times[i] and time>=times[i-1]),None)")
original.write_text(initial)
import difflib
patch=D/'patches/S407-physical-observer-analysis-grouping.patch';patch.write_text(''.join(difflib.unified_diff(initial.splitlines(True),(P/'r4_physical_audit.py').read_text().splitlines(True),fromfile='a/test/r4_physical_audit.py',tofile='b/test/r4_physical_audit.py')))
for mode,names in [('--chem',['chemistry-full','moving-chemistry']),('--gas',['SST-SSPRK2','SST-SSPRK3'])]:
 for name in names:
  suffix='-chemical-half-physical.json' if mode=='--chem' else '-gas-physical-budget.json';artifact='artifacts/'+name+suffix
  t=run('ANALYZE-S407-'+name,cmd('python3',P/'r4_physical_audit.py','--compute',mode,name),[8,9,10,14,20,21,26,29],expected='Test-only conversion/actual microstep RK grouping corrected; original criteria unchanged; original logs retained. No new GPU execution needed.',inputs=[patch,P/'r4_physical_audit.py'],kind='ANALYSIS',variant='SUPPORT_ANALYSIS_S407')
  assertion(t,t['exit_code']==0,'Actual recorded physical data checked under unchanged criteria, S407 analysis scope.',artifacts=[artifact])
