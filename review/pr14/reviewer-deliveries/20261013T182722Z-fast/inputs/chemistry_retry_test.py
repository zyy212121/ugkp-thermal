import pathlib,shutil,subprocess,json,tarfile,io
from run_fast import ROOT,D,run,cmd,amend
support=pathlib.Path(__file__).parent/'chemistry_retry.cu';source=D/'inputs/chemistry_retry.cu';shutil.copy2(support,source)
baseline=ROOT/'work/baseline-common';baseline.mkdir(exist_ok=True)
archive=subprocess.check_output(['git','-C',str(ROOT),'archive','d6110c02745a8ad726208a426c4bb3a28233c1fd','common'])
with tarfile.open(fileobj=io.BytesIO(archive)) as t:t.extractall(baseline)
with (D/'TEST_PLAN.md').open('a') as f:f.write('''\n## 追加 CHEM-RETRY：原成功native内层重试语义\n\nNs2全backed双组分同MW=.028、linearCp=1000、不可逆S0→S1 A=2；rho28、rhoSpecies[28,0]、momentum0、T700、V=.001、maximumSteps1、initial/maxStep1e-6。调用实际生产advanceGasChemistryKernel，dt2e-6、refining=true，预期d611返回StepLimit且species/audit/budget不发布；current应保留该原成功refinement方法，不应静默terminal完成。补dt1e-6成功控制和false公开terminal控制，maxSteps100正常控制；所有rho/momentum/rhoE/species、status/audit/budget/canary输出。CUDA sm89/O2/--fmad=false，同硬件同输入，每次300s；输入与源码SHA执行前spec登记。只恢复原有refinement入口，不新建整步数值拒绝机制。\n''')
for label,inc,flags in [('base',baseline,['-DBASELINE']),('pr',ROOT,[])]:
    binary=ROOT/'work'/('chemistry-retry-'+label)
    c=run('BUILD-CHEM-RETRY-'+label+'-FIXTURE1',cmd('nvcc','-std=c++17','-O2','-lineinfo','--fmad=false','-arch=sm_89','-I',inc,*flags,source,'-o',binary),[10,13,25],300,'Real native CUDA production chemistry adapter compiled with matching baseline/new configuration.',inputs=[source],kind='BUILD')
    if c['exit_code']:continue
    for refine,dt,steps,labelcase in [(1,'2e-6',1,'step-limit'),(1,'1e-6',1,'success-control'),(0,'2e-6',1,'public-terminal'),(1,'2e-6',100,'ordinary')]:
        t=run(f'CHEM-RETRY-{label}-{labelcase}',cmd(binary,str(refine),dt,str(steps)),[10,13,14,26],300,'Record actual all-field publication and diagnostics. Step-limit native=true must retain original retryable failure; normal case must conserve mass/energy and publish audit; false current path computes actual interval.',inputs=[source],binary=binary,variant='baseline' if label=='base' else 'original')
        print((D/t['log']).read_text(),flush=True)
        try:observed=json.loads((D/t['log']).read_text());amend(t['id'],observed=observed,verdict='INCONCLUSIVE')
        except Exception:pass
