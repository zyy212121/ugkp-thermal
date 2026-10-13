import pathlib,sys,shutil,subprocess,json,os
from run_fast import ROOT,D,run,cmd,sha,amend
TOPICS=[3,6,7,8,9,11,12,13,14,15,26,29,32]
checker=pathlib.Path(os.environ.get('CHMT_REVIEW_CHECKER',str(ROOT/'applications/CHMT/verification/coupled_runtime/check_case.py')))
generator=ROOT/'applications/CHMT/verification/coupled_runtime/make_case.py'
def run_case(name):
    case=ROOT/'work/cases/chmt'/name;inputs=[q for q in case.rglob('*') if q.is_file() and not q.name.endswith('.log')]
    binary=ROOT/('work/install/ns10/bin/CHMT' if name.startswith('chemistry-') else 'work/install/ns2/bin/CHMT')
    topics=TOPICS+([10] if name.startswith('chemistry-') else [])
    for region,args in [('gas',[]),('solid',['-region','solid','-dict',str(case/'system/blockMeshSolidDict')])]:
        res=run('MESH-'+name+'-'+region,cmd('blockMesh','-case',case,*args),[8,29],300,'Actual OpenFOAM blockMesh creates 8-cell unit gas/solid region with 4 contact faces.',inputs=inputs,kind='SETUP')
        amend(res['id'],verdict='PASS' if res['exit_code']==0 else 'BLOCKED',observations='Mesh setup only; no numerical/GPU acceptance.')
        if res['exit_code']:return
    res=run('CHMT-'+name,cmd(binary,'-case',case),topics,300,'Two native coupled windows (split: one); full written cell/packet/checkpoint fields judged by pinned independent checker.',inputs=inputs,binary=binary)
    if res['exit_code']:return
    out=case/'chmtOutput'
    dest=D/'artifacts/chmt'/name
    if out.is_dir():shutil.copytree(out,dest)
    check=run('CHECK-CHMT-'+name,cmd('python3',checker,case),topics,300,'Native valid-case UA=2, closed mass/species/element/energy, EOS/caloric, real packet consumer masks and checkpoint clock; original checker tolerances unchanged.',inputs=[checker,*inputs],kind='ANALYSIS')
    amend(check['id'],verdict='PASS' if check['exit_code']==0 else 'FAIL',observations=(D/check['log']).read_text(errors='replace'))
    amend(res['id'],verdict='PASS' if check['exit_code']==0 else 'FAIL',observations='Actual native GPU execution; per-field independent check '+check['id']+'. See all CSV/checkpoint artifacts.',artifacts=[str(dest.relative_to(D))],checker_result_id=check['id'])
def suite(scenarios):
    for scenario in scenarios:
        if not (ROOT/('work/install/ns10/bin/CHMT' if scenario=='chemistry' else 'work/install/ns2/bin/CHMT')).is_file():continue
        for kind in ['full','split']:run_case(scenario+'-'+kind)
        split=ROOT/'work/cases/chmt'/(scenario+'-split')/'chmtOutput'
        checkpoints=list(split.glob('checkpoint-*'))
        if checkpoints:
            source=max(checkpoints,key=lambda q:int(q.name.split('-')[-1]));resumed=ROOT/'work/cases/chmt'/(scenario+'-resumed')
            opts={'single':['--single'],'thermal':[],'diffusion':['--diffusion'],'chemistry':['--chemistry']}[scenario]
            args=['python3',str(generator),str(resumed),*opts,'--restart',str(source)]
            res=run('INPUT-'+scenario+'-resumed',cmd(*args),[5,19,29],300,'Generate continuation input referencing an actual split checkpoint; hash checkpoint before execution.',inputs=[generator,source/'state.chmt'],kind='SETUP')
            if res['exit_code']==0:
                shutil.copytree(resumed,D/'inputs/native/chmt'/resumed.name)
                run_case(resumed.name)
                res=run('RESTART-'+scenario,cmd('python3',checker,resumed,'--reference',ROOT/'work/cases/chmt'/(scenario+'-full'),'--restart-from',source),[5,14,19,29],300,'Bitwise initial checkpoint reserialization; final full gas/material fields abs1e-10 rel5e-12 and synchronized summary exact.',inputs=[checker,source/'state.chmt'],kind='ANALYSIS')
                amend(res['id'],verdict='PASS' if res['exit_code']==0 else 'FAIL',observations=(D/res['log']).read_text(errors='replace'))
        for kind in ['refined','control']:
            if not (ROOT/'work/cases/chmt'/(scenario+'-'+kind)).is_dir():continue
            run_case(scenario+'-'+kind)
            opt='--refinement' if kind=='refined' else '--diffusion-control' if scenario=='diffusion' else '--chemistry-control'
            res=run('COMPARE-'+scenario+'-'+kind,cmd('python3',checker,ROOT/'work/cases/chmt'/(scenario+'-full'),opt,ROOT/'work/cases/chmt'/(scenario+'-'+kind)),TOPICS+([10] if scenario=='chemistry' else []),300,'Original independent refinement/diffusion/chemistry matched-control thresholds, no tolerance changes.',inputs=[checker],kind='ANALYSIS')
            amend(res['id'],verdict='PASS' if res['exit_code']==0 else 'FAIL',observations=(D/res['log']).read_text(errors='replace'))
if __name__=='__main__':suite(sys.argv[1:] or ['thermal','single','diffusion','chemistry'])
