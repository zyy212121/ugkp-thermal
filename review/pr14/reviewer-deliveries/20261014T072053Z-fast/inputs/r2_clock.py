import io,tarfile,shutil,subprocess,json
from r2_runner import *
from r2_native import compare_dirs
TRACE=DEV/'clock-trace';out=D/'artifacts/clock-trace';out.mkdir(parents=True,exist_ok=True)
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nD执行细则：从三固定commit git archive提取applications/common/tools，每版额外Backend observer patch生成并原生构建900s；只记录已有host下载，不增加device copy/算式；同旧thermal-full/同mesh。全部checkpoint/CSV/geometry逐字节对该版未observer输出；baseline/tested使用旧不可变结果，formal使用本轮thermal-full。\n')
for label,source,expected in [('baseline',PATHS['baseline'],OLD/'artifacts/performance/native/baseline-preflight-0'),('tested',PATHS['old_source'],OLD/'artifacts/chmt/thermal-full'),('formal',SOURCE,D/'artifacts/chmt/thermal-full')]:
 overlay=ROOT/'work/clock-source'/(label+'-v1');overlay.mkdir(parents=True)
 payload=subprocess.check_output(['git','-C',str(ROOT),'archive',source,'applications/CHMT','common','tools'])
 with tarfile.open(fileobj=io.BytesIO(payload)) as tar:tar.extractall(overlay,filter='data')
 patch=D/'patches'/('D201-clock-'+label+'.patch')
 t=run('TRACE-PREP-'+label,cmd('python3',TRACE/'make_trace_patch.py',overlay/'applications/CHMT/gpu/Backend.cu',patch),[14,15,27],expected='Emit diagnostic-only exact-source patch; no production source mutation.',inputs=[TRACE/'make_trace_patch.py',TRACE/'clock_trace.H'],kind='SETUP',variant='diagnostic-clock-'+label,source=source)
 assert t['exit_code']==0
 t=run('TRACE-APPLY-'+label,['patch','-p1','--batch','--forward','-i',str(patch)],[14,15,27],expected='Apply observer only in isolated archive.',inputs=[patch],kind='SETUP',cwd=overlay,variant='diagnostic-clock-'+label,source=source)
 assert t['exit_code']==0
 build=ROOT/'work/clock-build'/label;binary=ROOT/'work/install'/('clock-'+label)/'bin/CHMT'
 t=run('TRACE-BUILD-'+label,cmd('env',f'CHMT_BUILD={build}',f'FOAM_USER_APPBIN={binary.parent}','bash',overlay/'applications/CHMT/Allwmake'),[14,15,24,25,27],900,'Real native isolated source+observer compile/link.',inputs=[patch,overlay/'applications/CHMT/gpu/Backend.cu'],kind='BUILD',variant='diagnostic-clock-'+label,source=source)
 if t['exit_code']:continue
 case=ROOT/'work/clock-cases'/label;shutil.copytree(OLD/'inputs/native/chmt/thermal-full',case)
 for rel in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(OLDROOT/'work/cases/chmt/thermal-full'/rel,case/rel)
 t=run('TRACE-NATIVE-'+label,cmd(binary,'-case',case),[14,15,27],expected='Actual100 microsteps clocks/dt/wall/gas/packet hexadecimal trace; diagnostics cannot enter performance timings.',inputs=[patch,*[p for p in (OLD/'inputs/native/chmt/thermal-full').rglob('*') if p.is_file()]],binary=binary,variant='diagnostic-clock-'+label,source=source)
 if not (case/'chmtOutput').exists():continue
 dest=save_case(case,label,'clock-output');comparison=compare_dirs(expected,case/'chmtOutput')
 (out/(label+'-output-comparison.json')).write_text(json.dumps(comparison,indent=2)+'\n')
 raw=(D/t['log']).read_text();records=sum(x.startswith('F003,') for x in raw.splitlines())
 assertion(t,records>0 and comparison['all_files_byte_equal'],'Actualtrace records '+str(records)+'; ALL original output comparison '+str(comparison['all_files_byte_equal']),artifacts=[str(dest.relative_to(D))],trace_records=records)
for label in ['tested','formal']:
 a=D/'logs/TRACE-NATIVE-baseline.log';b=D/'logs'/('TRACE-NATIVE-'+label+'.log')
 if a.exists() and b.exists():
  t=run('TRACE-COMPARE-base-'+label,cmd('python3',TRACE/'compare_trace.py',a,b),[14,15,27],expected='Report actual first bit difference by event; nonempty logs, no tolerance acceptance.',inputs=[a,b,TRACE/'compare_trace.py'],kind='ANALYSIS',variant='diagnostic-clock-'+label)
  report=json.loads((D/t['log']).read_text());(out/('base-'+label+'.json')).write_text(json.dumps(report,indent=2)+'\n');amend(t['id'],verdict='INCONCLUSIVE',observations='Exact crossversion trace observed; interpretation/first divergence investigation follows.',artifacts=['artifacts/clock-trace/base-'+label+'.json'])
