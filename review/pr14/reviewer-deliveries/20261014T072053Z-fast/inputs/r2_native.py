import sys,re,difflib,shutil,json
from r2_runner import *
CHECKER=ROOT/'applications/CHMT/verification/coupled_runtime/check_case.py'
def reused_inputs(src):return [p for p in src.rglob('*') if p.is_file()]
def compare_dirs(a,b):
 rows=[]
 for rel in sorted({str(p.relative_to(a)) for p in a.rglob('*') if p.is_file()}|{str(p.relative_to(b)) for p in b.rglob('*') if p.is_file()}):
  x=a/rel;y=b/rel;rows.append({'path':rel,'expected_sha256':sha(x) if x.is_file() else None,'actual_sha256':sha(y) if y.is_file() else None,'byte_equal':x.is_file() and y.is_file() and x.read_bytes()==y.read_bytes()})
 return {'all_files_byte_equal':bool(rows) and all(x['byte_equal'] for x in rows),'files':rows}
def chmt(name,tool=None):
 src=OLD/'inputs/native/chmt'/name;case=ROOT/'work/cases/chmt'/(name+('-'+tool if tool else ''))
 shutil.copytree(src,case)
 for rel in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(OLDROOT/'work/cases/chmt'/name/rel,case/rel)
 binary=ROOT/('work/install/ns10/bin/CHMT' if name.startswith('chemistry') else 'work/install/ns2/bin/CHMT')
 args=['compute-sanitizer','--tool',tool,'--target-processes','all','--error-exitcode','86'] if tool else []
 args +=[str(binary),'-case',str(case)]
 t=run('CHMT-'+name+('-'+tool if tool else ''),cmd(*args),[3,5,6,7,8,9,10,12,14,15,26,28,29],expected='Formal native GPU same original input and steps; all original outputs exact comparison to802, unchanged physical checker.',inputs=reused_inputs(src),binary=binary,kind='SANITIZER' if tool else 'GPU')
 out=save_case(case,case.name,'chmt') if (case/'chmtOutput').is_dir() else None
 comparison=compare_dirs(OLD/'artifacts/chmt'/name,case/'chmtOutput') if out else {'all_files_byte_equal':False,'missing_output':True}
 (D/'artifacts'/('compare-'+case.name+'.json')).write_text(json.dumps(comparison,indent=2)+'\n')
 raw=(D/t['log']).read_text();san=tool is None or 'ERROR SUMMARY: 0 errors' in raw
 ck=run('CHECK-'+case.name,cmd('python3',CHECKER,case),[3,5,6,7,9,10,14,26],expected='Original mass/EOS/energy/elements/packet/state rules; current actualschema support only.',inputs=[CHECKER],kind='ANALYSIS') if out else None
 if ck:assertion(ck,ck['exit_code']==0,(D/ck['log']).read_text())
 assertion(t,san and bool(ck) and ck['exit_code']==0 and comparison['all_files_byte_equal'],'Formal GPU, sanitizer '+str(san)+', physical checker and ALL original output bytes equal: '+str(comparison['all_files_byte_equal']),artifacts=[str(out.relative_to(D))] if out else [],comparison='artifacts/compare-'+case.name+'.json')
 return case
def checker_support():
 orig=OLD/'inputs/resident_check_fp64_mean.py';text=orig.read_text();changed=text.replace('from run_fast import ROOT,D','from r2_runner import ROOT,D')
 target=D/'inputs/resident_check_fp64_mean.py';target.write_text(changed)
 (D/'patches/S201-checker-path-adaptation.patch').write_text(''.join(difflib.unified_diff(text.splitlines(True),changed.splitlines(True),fromfile='a/old/resident_check_fp64_mean.py',tofile='b/new/resident_check_fp64_mean.py')))
 (D/'patches/S201-checker-path-adaptation.json').write_text(json.dumps({'classification':'TEST_SUPPORT','old_sha256':sha(orig),'new_sha256':sha(target),'changes':'Import new runner only; same double-mean/caloric/1e-9 threshold/field/restart assertions.'},indent=2)+'\n')
 return target
def resident(name,source_name=None,tool=None,modified=None):
 source_name=source_name or name;src=OLD/'inputs/native/resident'/source_name;case=ROOT/'work/cases/resident'/(name+('-'+tool if tool else ''))
 assert not (D/'logs'/('NATIVE-'+case.name+'.log')).exists()
 shutil.copytree(src,case,dirs_exist_ok=True)
 if modified:modified(case)
 mesh=OLDROOT/'work/cases/resident'/source_name/'constant/polyMesh'
 if not mesh.is_dir():mesh=OLDROOT/'work/cases/resident/CHT-coldWall2D-P010/constant/polyMesh'
 if not (case/'constant/polyMesh').exists():shutil.copytree(mesh,case/'constant/polyMesh')
 if modified:
  inp=D/'inputs/native/resident'/case.name;shutil.copytree(case,inp,ignore=shutil.ignore_patterns('polyMesh'))
 binary=ROOT/'work/install/ns2/bin'/('FSH' if name.startswith('FSH') else 'CHT')
 args=['compute-sanitizer','--tool',tool,'--target-processes','all','--error-exitcode','86'] if tool else []
 t=run('NATIVE-'+case.name,cmd(*args,binary,'-case',case),[5,12,18,19,20,23,26,28,33],expected='Formal cold native run all written profile/carry/temperature abs1e-9 with originalinput physics; actual restart final bytes.',inputs=reused_inputs(D/'inputs/native/resident'/case.name if modified else src),binary=binary,kind='SANITIZER' if tool else 'GPU')
 save_case(case,case.name)
 checker=D/'inputs/resident_check_fsh_double.py' if name.startswith('FSH') else D/'inputs/resident_check_fp64_mean.py'
 ck=run('FIELDS-'+case.name,cmd('python3',checker,'finite',case.name),[23,26,33],expected='Double mean all original fields/times finite; original inverse caloric1e-9K.',inputs=[checker],kind='ANALYSIS')
 raw=(D/t['log']).read_text();good=tool is None or 'ERROR SUMMARY: 0 errors' in raw
 assertion(ck,ck['exit_code']==0,(D/ck['log']).read_text());assertion(t,good and ck['exit_code']==0,'Actual formal cold run; corrected mean-double checker and sanitizer summary independently checked.',checker_result_id=ck['id'])
 return case
def restart(case):
 name=case.name;dest=ROOT/'work/cases/resident'/(name+'-RESUME');src=OLD/'inputs/native/resident/CHT-coldWall2D-INPUT2'
 assert not (D/'logs'/('RESTART-'+name+'.log')).exists()
 shutil.copytree(src,dest,dirs_exist_ok=True)
 shutil.copytree(case/'5e-06',dest/'5e-06',dirs_exist_ok=True);shutil.copytree(case/'constant/polyMesh',dest/'constant/polyMesh',dirs_exist_ok=True)
 p=dest/'system/controlDict';p.write_text(re.sub(r'startTime\s+0;',r'startTime 5e-6;',p.read_text()))
 if name.startswith('FSH'):
  for q in ['system/controlDict','constant/particleProperties']:
   shutil.copy2(case/q,dest/q)
  p=dest/'system/controlDict';p.write_text(re.sub(r'startTime\s+0;',r'startTime 5e-6;',p.read_text()))
 inp=D/'inputs/native/resident'/dest.name;shutil.copytree(dest,inp,ignore=shutil.ignore_patterns('polyMesh'))
 binary=ROOT/'work/install/ns2/bin'/name.split('-')[0]
 t=run('RESTART-'+name,cmd(binary,'-case',dest),[5,18,19,23,26,33],expected='Actual midpoint checkpoint restore; all final decoded/written fields byteequal full.',inputs=reused_inputs(inp),binary=binary)
 save_case(dest,dest.name)
 ck=run('RESTART-CHECK-'+name,cmd('python3',D/'inputs/resident_check_fp64_mean.py','restart',name),[5,18,19,23,26,33],expected='Every final native particle/RNG/profile/carry/history and terminal file bytecomparison.',inputs=[D/'inputs/resident_check_fp64_mean.py'],kind='ANALYSIS')
 assertion(ck,ck['exit_code']==0,(D/ck['log']).read_text());assertion(t,ck['exit_code']==0,'Formal midpoint restart all finalfields bytecomparison.',checker_result_id=ck['id'])
if __name__=='__main__':
 if sys.argv[1]=='chmt':
  chmt('thermal-full');chmt('thermal-full','initcheck')
 elif sys.argv[1]=='ns10':chmt('chemistry-full');chmt('chemistry-control')
 elif sys.argv[1]=='cold':
  checker_support()
  resident('CHT-coldWall1D-INPUT2')
  c=resident('CHT-coldWall2D-INPUT2');restart(c)
  resident('CHT-coldWall1D-WET100','CHT-coldWall1D-P010-WET100')
  resident('CHT-coldWall2D-INPUT2',tool='memcheck');resident('CHT-coldWall2D-INPUT2',tool='synccheck')
