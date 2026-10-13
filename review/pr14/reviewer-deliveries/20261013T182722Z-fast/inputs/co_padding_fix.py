import pathlib,shutil,json,difflib
from run_fast import ROOT,D,run,cmd,amend,sha
overlay=ROOT/'work/diagnostic-P011'
shutil.copytree(ROOT/'common',overlay/'common',ignore=shutil.ignore_patterns('__pycache__'))
shutil.copytree(ROOT/'applications/CHMT',overlay/'applications/CHMT',ignore=shutil.ignore_patterns('docs','__pycache__','linux64*'))
(overlay/'applications/gasUGKP').symlink_to(ROOT/'applications/gasUGKP',target_is_directory=True)
p=overlay/'applications/CHMT/gpu/Backend.cu';before=p.read_text();needle='struct CoupledCourantValue { double value;bool unavailable; };'
if needle not in before:
 import re
 match=re.search(r'struct CoupledCourantValue\s*\{[^}]+\};',before);assert match
 needle=match.group(0)
assert 'bool unavailable' in needle
replacement=needle.replace('bool unavailable','std::uint64_t unavailable')
after=before.replace(needle,replacement);p.write_text(after)
patch=D/'patches/P011-Co-reduction-result-no-padding.patch';patch.write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='a/applications/CHMT/gpu/Backend.cu',tofile='b/applications/CHMT/gpu/Backend.cu')))
(D/'patches/P011-Co-reduction-result-no-padding.json').write_text(json.dumps({'classification':'isolated metadata result representation correction/diagnostic','finding':'GPU-F005','reason':'Original initcheck reports100 host16byte D2H uninitialized errors, one per Co transform_reduce. double+bool has seven uninitialized padding bytes. Use explicit8byte 0/1 slot; same physical double/max/OR traversal/order and same16byte ABI result size. No Co/Js numeric poisoning established.','before_sha256':sha(ROOT/'applications/CHMT/gpu/Backend.cu'),'after_sha256':sha(p),'patch_sha256':sha(patch),'formal_fix':False,'production_checkout_modified':False,'scope':'Native Ns2 thermal, all CSV/checkpoint bytes equality plus initcheck0; Ns10/other configurations need formal recheck.'},indent=2)+'\n')
with (D/'TEST_PLAN.md').open('a') as f:f.write('\n追加P011：原CHMT initcheck100次16byte result D2H未初始化先保留（red）；调查Co transform_reduce double+bool尾部7字节padding，不先认定Co/Js值失效。隔离改metadata flag为uint64 0/1，same16byte size/maximum/OR/遍历；900s Ns2 native build，300s full-run/全CSV+checkpoint逐字节/新fresh initcheck。无需新初始化扫描、数值门禁或物理运算改写。\n')
binary=ROOT/'work/install/P011/bin/CHMT';binary.parent.mkdir(parents=True,exist_ok=True)
t=run('BUILD-CHMT2-P011',cmd('env',f'FOAM_USER_APPBIN={binary.parent}',f'CHMT_BUILD={ROOT}/work/build/P011-CHMT2','bash',overlay/'applications/CHMT/Allwmake'),[9,12,14,15,24,28],900,'Isolated padding-only producer/transfer representation; matching original Ns2 O2/sm89 flags.',inputs=[patch],kind='BUILD',variant='diagnostic-P011')
if t['exit_code']:raise SystemExit(1)
src=D/'inputs/native/chmt/thermal-full';mesh=ROOT/'work/cases/chmt/thermal-full'
for tool in ['ordinary','initcheck']:
 case=ROOT/'work/Co-padding-P011'/tool;shutil.copytree(src,case)
 for relative in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(mesh/relative,case/relative)
 args=cmd(binary,'-case',case) if tool=='ordinary' else cmd('compute-sanitizer','--tool','initcheck','--error-exitcode','86',binary,'-case',case)
 t=run('CO-P011-'+tool,args,[9,12,14,15,26,28],300,'Same real native input, unchanged field values. Padding diagnostic only. All actual source flags retained.',inputs=list(src.rglob('*')),binary=binary,variant='diagnostic-P011',kind='GPU' if tool=='ordinary' else 'SANITIZER')
 raw=(D/t['log']).read_text();same=True;diffs=[]
 for q in (mesh/'chmtOutput').rglob('*'):
  if not q.is_file():continue
  relative=q.relative_to(mesh/'chmtOutput');other=case/'chmtOutput'/relative;equal=other.exists() and q.read_bytes()==other.read_bytes()
  same &= equal;diffs.append({'path':str(relative),'byte_equal':equal,'original_sha256':sha(q),'diagnostic_sha256':sha(other) if other.exists() else None})
 check=D/'artifacts'/('Co-P011-'+tool+'-full-publication-comparison.json');check.write_text(json.dumps({'all_output_bytes_equal':same,'files':diffs,'initcheck_zero':'ERROR SUMMARY: 0 errors' in raw},indent=2)+'\n')
 verdict='PASS' if t['exit_code']==0 and same and (tool=='ordinary' or 'ERROR SUMMARY: 0 errors' in raw) else 'FAIL'
 amend(t['id'],verdict=verdict,artifact=str(check.relative_to(D)),observations='All native CSV/checkpoint/geometry/publication bytes equal original ='+str(same)+'; initcheck0='+str('ERROR SUMMARY: 0 errors' in raw)+'.')
 print(tool,'same',same,'summary',raw[-1200:])
