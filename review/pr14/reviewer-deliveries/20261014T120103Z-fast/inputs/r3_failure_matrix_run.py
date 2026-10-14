from r3_runner import *
import re
b=ROOT/'work/fixtures/chemistry-trial-S302';assert b.is_file()
t=run('CHEMISTRY-S302-FULL-FAILED-TRANSACTIONS',cmd(b),[10,12,14,26],300,'All16actualfailedTrials: bothcellsfullstatus/semanticstate,allcellfacebudgetarrays+hostledger,rollbackrestoresstateandretainsownerdecisions. FormaloriginalPASSkeptseparate.',inputs=[D/'inputs/chemistry-fixture-S302.cu',D/'patches/S301-complete-failure-transactions.patch',D/'patches/S302-explicit-status-bool.patch'],binary=b,variant='SUPPORT_S301_S302')
raw=(D/t['log']).read_text();kinds=list(map(int,re.findall(r'PASS failed-transaction kind=(\d+)',raw)));statusrows=re.findall(r'^FAIL_STATUS .+$',raw,re.M)
summary={'source_commit':SOURCE,'variant':'SUPPORT_S301_S302','exit':t['exit_code'],'all16actualfailedTrialKinds':kinds,'bothcellstatusrecords':statusrows,'failed_kind10':[q for q in statusrows if q.startswith('FAIL_STATUS kind=10 ')],'original_formal_fixture':'CHEMISTRY-FORMAL-FIXTURE','limitations':'Additionalsupportcompletesoriginalcoveragegaps; submittedoriginalfixtureassertionsnotchangedandstillscoped. No generalallocation/free/sync recoveryclaim.'}
(D/'artifacts/chemistry-S302-full-failure-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
assertion(t,t['exit_code']==0 and kinds==list(range(16)) and len(statusrows)==32,'Actualall16failuretransactions,both6-memberstatuses,fullbytefieldaudit/hostledger/rollback PASS='+str(kinds==list(range(16))),artifacts=['artifacts/chemistry-S302-full-failure-summary.json'])
print(raw[-5500:],flush=True)
