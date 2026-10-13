import pathlib,shutil,json,csv,statistics,time
from run_fast import ROOT,D,run,cmd,amend,sha
base='d6110c02745a8ad726208a426c4bb3a28233c1fd'
template=D/'inputs/native/chmt/thermal-full'
mesh=ROOT/'work/cases/chmt/thermal-full'
dest=D/'artifacts/performance/native';dest.mkdir(parents=True,exist_ok=True)
with (D/'TEST_PLAN.md').open('a') as f:
 f.write('\n追加 native 计时执行细则：同一已生成输入与网格逐字节复制，fresh output；base/PR各一次预检，然后各3次warmup、7次交错重复。外部进程墙钟包含启动、CUDA context、I/O，300s/次。接受实际CSV所有数值列及packet数/accepted_steps一致后才报告速度比；不把启动时间当kernel时间。\n')
rows=[]
def one(variant,phase,i):
 binary=ROOT/('work/install/base/bin/CHMT' if variant=='baseline' else 'work/install/ns2/bin/CHMT')
 name=f'{variant}-{phase}-{i}';case=ROOT/'work/perf-native'/name
 shutil.copytree(template,case)
 for path in ['constant/polyMesh','constant/solid/polyMesh']:
  if (mesh/path).exists():shutil.copytree(mesh/path,case/path)
 t=run('PERF-NATIVE-'+name,cmd(binary,'-case',case),[2,24,27],300,'Matched native 8+8 cell two-window thermal workload; finite actual CSV fields and same work count, warmup3/repeats7.',inputs=list(template.rglob('*')),binary=binary,variant=variant,kind='PERFORMANCE')
 output=case/'chmtOutput';counts={};hashes={}
 if t['exit_code']==0 and output.is_dir():
  for q in output.glob('*.csv'):
   with q.open() as h: data=list(csv.DictReader(h))
   counts[q.name]=len(data);hashes[q.name]=sha(q)
  if phase=='preflight':shutil.copytree(output,dest/name)
 record={'variant':variant,'phase':phase,'index':i,'result_id':t['id'],'elapsed_seconds':t['elapsed_seconds'],'exit_code':t['exit_code'],'source_commit':base if variant=='baseline' else t['source_commit'],'csv_row_counts':counts,'csv_sha256':hashes,'binary_sha256':sha(binary)}
 rows.append(record);(dest/'samples.json').write_text(json.dumps(rows,indent=2)+'\n')
 amend(t['id'],source_commit=record['source_commit'],source_tree=__import__('subprocess').check_output(['git','-C',str(ROOT),'rev-parse',record['source_commit']+'^{tree}'],text=True).strip(),verdict='PASS' if t['exit_code']==0 else 'BLOCKED',observations='Performance sample only, full workload equality separately checked.',artifact=str((dest/'samples.json').relative_to(D)))
 return record
for v in ['baseline','original']:
 if one(v,'preflight',0)['exit_code']:raise SystemExit('preflight blocked, no ratio')
same=True;diffs=[]
a=dest/'baseline-preflight-0';b=dest/'original-preflight-0'
for q in sorted(a.glob('*.csv')):
 counterpart=b/q.name
 if not counterpart.exists():same=False;diffs.append({'file':q.name,'missing':True});continue
 if q.read_bytes()!=counterpart.read_bytes():
  aa=list(csv.reader(q.open()));bb=list(csv.reader(counterpart.open()))
  finite=True;maxabs=0.;maxrel=0.;count=0
  if len(aa)!=len(bb) or aa[0]!=bb[0]:same=False;diffs.append({'file':q.name,'shape_changed':True});continue
  for ar,br in zip(aa[1:],bb[1:]):
   for x,y in zip(ar,br):
    try:xx=float(x);yy=float(y)
    except ValueError:finite &= x==y;continue
    finite &= __import__('math').isfinite(xx) and __import__('math').isfinite(yy)
    maxabs=max(maxabs,abs(xx-yy));maxrel=max(maxrel,abs(xx-yy)/max(abs(xx),abs(yy),1e-300));count+=1
  ok=finite and maxabs==0
  same &= ok;diffs.append({'file':q.name,'numeric_count':count,'finite':finite,'maximum_abs':maxabs,'maximum_rel':maxrel,'exact_numeric_equal':ok})
(dest/'preflight-comparison.json').write_text(json.dumps({'all_csv_numeric_equal':same,'differences':diffs,'scope':'All initial/final/intermediate gas/material/packet/accepted CSV fields; checkpoint binary schema intentionally different.'},indent=2)+'\n')
if not same:raise SystemExit('workload/output difference, no speed ratio')
for i in range(3):
 for v in ['baseline','original']:one(v,'warmup',i)
for i in range(7):
 for v in (['baseline','original'] if i%2==0 else ['original','baseline']):one(v,'measured',i)
summary={}
for v in ['baseline','original']:
 values=[r['elapsed_seconds'] for r in rows if r['phase']=='measured' and r['variant']==v]
 summary[v]={'samples_seconds':values,'median_seconds':statistics.median(values),'minimum_seconds':min(values),'maximum_seconds':max(values),'stdev_seconds':statistics.stdev(values)}
summary['ratio_PR_over_base']=summary['original']['median_seconds']/summary['baseline']['median_seconds']
summary['percent_change']=100*(summary['ratio_PR_over_base']-1)
summary['all_csv_equal']=same;summary['scope']='Process wall time, two native windows incl CUDA context startup and output. No solver-wide or production speedup extrapolation.'
(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
