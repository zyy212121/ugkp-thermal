import pathlib,shutil,re,json,difflib
from run_fast import ROOT,D,run,cmd,amend,sha
original=D/'inputs/native/resident/CHT-coldWall2D-INPUT2';src=D/'inputs/native/resident/CHT-coldWall2D-INPUT3-BASE64';shutil.copytree(original,src);diffs=[]
for relative in ['constant/particleProperties','constant/schedulingProperties']:
 p=src/relative;before=p.read_text()
 after=re.sub(r'^\s*wall(?:Density|SpecificHeat|ThermalConductivity)\s+[^;]+;\n','',before,flags=re.M) if 'particleProperties' in relative else before+'\ngpuPrecision 64;\n'
 p.write_text(after);diffs.append(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/'+relative,tofile='b/'+relative)))
(D/'patches/P008-baseline-configuration-only.patch').write_text(''.join(diffs))
(D/'patches/P008-baseline-configuration-only.json').write_text(json.dumps({'classification':'input admission compatibility only','reason':'Original baseline rejects three obsolete scalar wall dictionary tokens before native cold calculation and defaults GPU physics32. Remove only these tokens that PR has no consumer for; explicitly select existing baseline FP64 route. Run BOTH baseline and PR with same adapted input.','physical_inputs_modified':False,'actual_consumed_defaults_retained':'wall device1800/710/100, wallT473.15, droplet/model unchanged. Does not pretend original dictionary7900/500/15 consumed.','original_failure_preserved':'logs/BASE-COLD2D.log','not_original_success_evidence':True},indent=2)+'\n')
with (D/'TEST_PLAN.md').open('a') as f:f.write('\n追加P008 baseline准入对照：原base相同INPUT2在旧wall字典准入处退出，原PR继续并NaN。删除仅三个PR未消费的旧token并选已有base FP64；同一INPUT3同时运行base/PR，保持所有实际消费物性/初场。仅确定历史失败边界，不作为原成功FP64回归或性能对照。每例300s，原失败不改。\n')
for variant in ['baseline','original']:
 case=ROOT/'work'/('cold2D-INPUT3-'+variant);shutil.copytree(src,case)
 t=run('MESH-COLD2D-INPUT3-'+variant,cmd('blockMesh','-case',case),[33],300,'Same adapted input for old/new numerical refusal boundary.',inputs=list(src.rglob('*')),kind='SETUP',variant=variant)
 if t['exit_code']:continue
 b=ROOT/('work/install/base/bin/CHT' if variant=='baseline' else 'work/install/ns2/bin/CHT')
 t=run('COLD2D-INPUT3-'+variant,cmd(b,'-case',case),[12,18,26,33],300,'Historical FP64 old rejection boundary and current actual NaN publication; no added gate or forbidden array indices.',inputs=list(src.rglob('*')),binary=b,variant=variant)
 amend(t['id'],verdict='BLOCKED' if variant=='baseline' and t['exit_code'] else 'FAIL' if variant=='original' else 'INCONCLUSIVE',observations=(D/t['log']).read_text()[-6000:])
 if variant=='original' and t['exit_code']==0:
  out=D/'artifacts/resident/CHT-coldWall2D-INPUT3';out.mkdir(parents=True,exist_ok=True)
  for q in case.iterdir():
   try:v=float(q.name)
   except ValueError:continue
   if v>0:shutil.copytree(q,out/q.name)
