from r4_runner import *
for n in ['phase-melting','phase-solidification']:
 p=D/'artifacts'/(n+'-gas-physical-budget.json');shutil.copy2(p,D/'artifacts'/(n+'-S407-discarded-trial-double-count-original.json'))
with (D/'TEST_PLAN.md').open('a') as h:h.write('\nS408分析修正：phase原onecell生成一次discarded outer candidate和一次accepted replay（实际日志one microstep/two material solves），D401初观测聚合错误计入两次压力冲量，原红JSON保留。以实际提交checkpoint gasStages[0].newVolume的原double位值核对最终候选；同time/dt的另一不同geometry候选不属accepted ledger。不改输入/生产重放/阈值。只完成该真实gas库存/通量/packet scope，非globalfilm/support指标。S404默认ModelConfig.physics未解析，不能当实际配置/film KE测量。\n')
for n in ['phase-melting','phase-solidification']:
 t=run('ANALYZE-S408-'+n,cmd('python3',P/'r4_physical_audit.py','--compute','--gas',n),[3,6,13,14,20,21,26,29],expected='Final accepted geometry verified from actual checkpoint; preserve earlier discarded-trial double-count failure and unchanged criteria.',inputs=[P/'r4_physical_audit.py'],kind='ANALYSIS',variant='SUPPORT_ANALYSIS_S408');assertion(t,t['exit_code']==0,'Actual final committed gas history only, independently selected by final volume bits.',artifacts=['artifacts/'+n+'-gas-physical-budget.json'])
