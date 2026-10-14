from r4_runner import *
import re
with (D/'TEST_PLAN.md').open('a') as f:f.write('\n新增SST64 static/moving ×RK1/2/3原因：原8cells全部贴壁，其omega为代数约束，不能证明内部未约束omega实际source；原moving输入未启SST。只在新独立控制保持原.001/100steps、原物性/FP64/阈值，网格2³改4³得到8内部cells与多face blocks；原所有输入/结果不动。static/moving均使用原PrescribedSinusoidal参数，实际Euler/RK源/最终flux/old-newV及omega约束类别分别观察；D401原全部输出bytes须与对应正式新控制一致。新增规模不是缩旧输入或化学/蒸发终点。\n')
for rk in [1,2,3]:
 for moving in [False,True]:
  base=ROOT/'work/cases'/(['SST-RK','SST-SSPRK2','SST-SSPRK3'][rk-1]);name=f'SST64-RK{rk}-'+('moving' if moving else 'static');case=ROOT/'work/cases'/name;assert not case.exists();case.mkdir()
  for n in ['0','constant','system']:shutil.copytree(base/n,case/n)
  for p in [case/'constant/polyMesh',case/'constant/solid/polyMesh']:
   assert p.resolve().is_relative_to(case.resolve()) and p.is_dir();shutil.rmtree(p)
  for p in [case/'system/blockMeshDict',case/'system/blockMeshSolidDict']:
   s=p.read_text();new=re.sub(r'\(2\s+2\s+2\)', '(4 4 4)',s);assert new!=s;p.write_text(new)
  if moving:
   p=case/'constant/chmtProperties';p.write_text(p.read_text()+'\nmeshMotion { policy PrescribedSinusoidal; amplitude (.001 0 0); spatialWaveNumber (1 0 0); spatialOrigin (0 0 0); angularFrequency 100; timeOrigin 0; }\n')
  for reg,args in [('gas',[]),('solid',['-region','solid','-dict',case/'system/blockMeshSolidDict'])]:
   q=run('MESH-'+name+'-'+reg,cmd('blockMesh','-case',case,*args),[8,9,14,26,29],expected='Legalindependent 4³realOF mesh with8interiorcells, wholeoriginal endpoint kept; meshsetupnotnumericalPASS.',inputs=[case/'system/blockMeshDict',case/'system/blockMeshSolidDict'],kind='SETUP');assert q['exit_code']==0;assertion(q,True,'ActualOpenFoam64cells/multiblocksetup only.')
  inp=D/'inputs/cases'/name;shutil.copytree(case,inp)
  for diag in [False,True]:
   target=case
   if diag:
    target=case.parent/('D401-'+name);target.mkdir()
    for n in ['0','constant','system']:shutil.copytree(case/n,target/n)
   b=ROOT/'work/install'/('diag2' if diag else 'ns2')/'bin/CHMT'
   q=run(('D401-' if diag else '')+'GPU-'+name,cmd(b,'-case',target),[8,9,14,26,29],expected='New64cellinterior/static-moving/SST actualsource/RK1-3 originalsmallcontrol full.001; physicalbudget auditseparate.',inputs=[p for p in inp.rglob('*') if p.is_file()],binary=b,variant='DIAGNOSTIC_D401' if diag else 'FORMAL')
   if not diag:archive_output(case,name)
   else:
    rows=[{'path':str(p.relative_to(case/'chmtOutput')),'formal_sha256':sha(p),'observer_sha256':sha(target/'chmtOutput'/p.relative_to(case/'chmtOutput')),'byte_equal':p.read_bytes()==(target/'chmtOutput'/p.relative_to(case/'chmtOutput')).read_bytes()} for p in (case/'chmtOutput').rglob('*') if p.is_file()]
    out=D/'artifacts'/('D401-'+name+'-output-invariance.json');out.write_text(json.dumps({'all_original_files_byte_equal':all(r['byte_equal'] for r in rows),'comparisons':rows},indent=2)+'\n');assertion(q,q['exit_code']==0 and all(r['byte_equal'] for r in rows),'New64cellformal/observer alloriginal outputbytes='+str(all(r['byte_equal'] for r in rows)),artifacts=[str(out.relative_to(D))])
  print('SST64_READY',name,flush=True)
