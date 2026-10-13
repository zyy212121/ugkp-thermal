from run_fast import ROOT,D,run,cmd,amend,sha
import pathlib,shutil,json
overlay=ROOT/"work/diagnostic-P011"
patch=D/"patches/P011-Co-reduction-result-no-padding.patch"
binary=ROOT/'work/install/P011/bin/CHMT';binary.parent.mkdir(parents=True,exist_ok=True)
t=run('BUILD-CHMT2-P011-v2',cmd('env',f'FOAM_USER_APPBIN={binary.parent}',f'CHMT_BUILD={ROOT}/work/build/P011-CHMT2','bash',overlay/'applications/CHMT/Allwmake'),[9,12,14,15,24,28],900,'Isolated padding-only producer/transfer representation; matching original Ns2 O2/sm89 flags.',inputs=[patch],kind='BUILD',variant='diagnostic-P011')
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
