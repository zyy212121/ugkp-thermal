from run_fast import ROOT,D,run,cmd,amend
import pathlib,shutil
s=pathlib.Path(__file__).parent/'check_chemistry_retry.py';q=D/'inputs/check_chemistry_retry.py';shutil.copy2(s,q)
t=run('CHECK-CHEM-RETRY-RED',cmd('python3',q,D/'logs','pr'),[10,13,14,26],300,'All native-refinement state/status/audit/budget fields exact versus baseline; false public path continues. This must fail on original to establish regression.',inputs=[q,*list((D/'logs').glob('CHEM-RETRY-*.log'))],kind='ANALYSIS')
amend(t['id'],verdict='FAIL' if t['exit_code'] else 'PASS',observations=(D/t['log']).read_text(),finding_id='GPU-F001')
