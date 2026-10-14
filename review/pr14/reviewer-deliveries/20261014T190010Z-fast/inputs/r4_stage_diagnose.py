from r4_runner import *
source=P/'r4_stage_audit.py'
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nS407分析工具异常定位：第一次D401分析SIGSEGV/exit139，RSS227340KiB、vmstat oom_kill前后0；不归为GPU生产失败或OOM。保留原partialreports/log/resource，针对moving-chemistry单文件用Pythonfaulthandler原计算诊断，不新GPU重跑或放阈值。wall-constrainedOmega naiveblend旧oracle另记真实约束scope不适配，不能忽略红改整体绿。\n')
t=run('ANALYZE-D401-MOVING-FAULTHANDLER',cmd('python3','-X','faulthandler',source,'--compute','--only','moving-chemistry'),[8,9,10,14,26,29],expected='Diagnosehostanalysisprocesssegfault location onoriginalactualmovingchem log; no numericalcriterion edits.',inputs=[source,D/'logs/D401-GPU-moving-chemistry.log'],kind='ANALYSIS_DIAGNOSTIC',variant='DIAGNOSTIC_S407')
print((D/t['log']).read_text(errors='replace')[-4200:],flush=True)
