import pathlib,json,shutil,difflib,hashlib
from run_fast import ROOT,D,run,cmd,amend
overlay=ROOT/'work/diagnostic-P002';overlay.mkdir(exist_ok=True);shutil.copytree(ROOT/'common',overlay/'common')
relative='common/operators/advanceGasChemistryKernel.cuh';q=overlay/relative;old=q.read_text()
start=old.index('        status = advanceClosedReactor\n');end=old.index('        gasChemistryDetail::retainDiagnostic',start)
before=old[start:end]
after='''        if (nativeRefinement)
            status = advanceClosedReactorTrial
            (
                input, interval, species.thermo, species.mechanism,
                castChemistryControls<Real>(species.chemistryControls), result, localAudit, &reactorAvailable
            );
        else
            status = advanceClosedReactor
            (
                input, interval, species.thermo, species.mechanism,
                castChemistryControls<Real>(species.chemistryControls), result, localAudit, &reactorAvailable
            );
        if (!reactorAvailable)
            return gasChemistryAdapterDetail::recordFailure
            (
                species, cell, status,
                !nativeRefinement || status.code == ChemistryCode::InvalidModel
            );
'''
new=old[:start]+after+old[end:];q.write_text(new)
diff=''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+relative,tofile='b/'+relative))
(D/'patches/P002-native-chemistry-trial-routing.patch').write_text(diff)
(D/'patches/P002-native-chemistry-trial-routing.json').write_text(json.dumps({'identity':'PR14_REVIEW_ONLY','purpose':'Isolated diagnosis: route preexisting nativeRefinement to preexisting retry Trial API; distinguish StepLimit trial failure from missing backing so native owner can use its original retry policy. Public false remains terminal.','finding':'GPU-F001','source_commit':'802a11e7dfe986488a8bf8305e86424dcc94c9e1','path':relative,'before_sha256':hashlib.sha256(old.encode()).hexdigest(),'after_sha256':hashlib.sha256(new.encode()).hexdigest(),'patch_sha256':hashlib.sha256(diff.encode()).hexdigest(),'scope_limit':'StepLimit plus original-success/public controls. Developer must formally audit actual unavailable thermo/model classifications and native full requested-interval retries before closure.','production_checkout_modified':False},ensure_ascii=False,indent=2)+'\n')
source=D/'inputs/chemistry_retry.cu';binary=ROOT/'work/chemistry-retry-patch'
t=run('BUILD-CHEM-RETRY-P002',cmd('nvcc','-std=c++17','-O2','-lineinfo','--fmad=false','-arch=sm_89','-I',overlay,source,'-o',binary),[10,13,25],300,'Isolated P002 native Trial routing compile; same flags/input as baseline and original.',inputs=[source,q,D/'patches/P002-native-chemistry-trial-routing.patch'],kind='BUILD',variant='diagnostic-P002')
if t['exit_code']:raise SystemExit(0)
for refine,dt,steps,label in [(1,'2e-6',1,'step-limit'),(1,'1e-6',1,'success-control'),(0,'2e-6',1,'public-terminal'),(1,'2e-6',100,'ordinary')]:
    t=run('CHEM-RETRY-patch-'+label,cmd(binary,str(refine),dt,str(steps)),[10,13,14,26],300,'Native=true all fields exactly match original successful retry semantics/baseline; false path preserves actual terminal publication.',inputs=[source,q],binary=binary,variant='diagnostic-P002')
    amend(t['id'],observed=json.loads((D/t['log']).read_text()),verdict='INCONCLUSIVE')
    print((D/t['log']).read_text(),flush=True)
t=run('CHECK-CHEM-RETRY-P002',cmd('python3',D/'inputs/check_chemistry_retry.py',D/'logs','patch'),[10,13,14,26],300,'Same failing assertion unmodified: native baseline exact full-field equality and public terminal completion.',inputs=[D/'inputs/check_chemistry_retry.py',*list((D/'logs').glob('CHEM-RETRY-*.log'))],kind='ANALYSIS',variant='diagnostic-P002')
amend(t['id'],verdict='PASS' if t['exit_code']==0 else 'FAIL',finding_id='GPU-F001',observations=(D/t['log']).read_text())
