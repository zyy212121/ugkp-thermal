import pathlib,shutil,sys,json
from run_fast import ROOT,D,run,cmd,amend,sha
jobs=[('DIFF','tests/fixtures/operator_unification/differential.py'),('PRESSURE','tests/fixtures/pressure/limited_flux_balance.py'),('AUTO','tests/fixtures/gas_auto/gas_auto_pool.py'),('COLLISION','tests/fixtures/shared_operators/collision_behavior.py')]
for group,script in jobs:
    for app in ['gasUGKP','FSH','CHT']:
        out=ROOT/'work/cuda-operators'/(group+'-'+app);topics=[18,25,26,31]+([33] if app!='gasUGKP' and group=='DIFF' else [])
        t=run('CUDA-'+group+'-'+app,cmd('python3',ROOT/script,ROOT,out,app),topics,300,'Actual production CUDA device execution and full fields/reference invariants from frozen input generator. AUTO first compiles only; runtime consumers executed separately. Cold segment may be obsolete: original failure retained and classified by actual assertion.',inputs=[ROOT/script],kind='BUILD' if group=='AUTO' else 'GPU')
        files=[q for q in out.rglob('*') if q.is_file() and q.suffix in ['.cu','.log','.json','.txt']]
        dest=D/'artifacts/cuda-operators'/(group+'-'+app)
        for q in files:target=dest/q.relative_to(out);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(q,target)
        binaries=[q for q in out.rglob('*') if q.is_file() and q.suffix=='' and q.stat().st_mode&0o111]
        amend(t['id'],artifacts=[str(dest.relative_to(D))],generated_binary_hashes={str(q.relative_to(out)):sha(q) for q in binaries})
        if group=='AUTO' and t['exit_code']==0:
            binary=out/'gas_auto_pool'
            for directory,prebuilt,block in [(0,1,32),(1,0,64),(2,0,128),(2,1,128),(0,0,128),(1,1,32)]:
                a=run(f'CUDA-AUTO-RUN-{app}-d{directory}-p{prebuilt}-b{block}',cmd(binary,str(directory),str(prebuilt),str(block)),[12,18,26,31],300,'Actual automatic scheduling prepares directories consumed by actual pool task kernels: seven physical moments/count/status/RNG complete comparison; blocks32/64/128 and transitionsL1→L2/prebuiltL2.',inputs=[out/'gas_auto_pool.cu'],binary=binary)
                amend(a['id'],verdict='PASS' if a['exit_code']==0 else 'FAIL',observations=(D/a['log']).read_text())
        elif t['exit_code']==0:amend(t['id'],verdict='PASS',observations=(D/t['log']).read_text()[-15000:])
        else:amend(t['id'],verdict='FAIL',observations=(D/t['log']).read_text()[-5000:])
