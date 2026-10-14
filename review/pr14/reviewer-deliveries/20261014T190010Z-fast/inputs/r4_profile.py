from r4_runner import *
import sqlite3,csv,collections,statistics
b=ROOT/'work/install/ns2/bin/CHMT';base=ROOT/'work/cases/thermal-full';case=ROOT/'work/profile-native';case.mkdir()
for n in ['0','constant','system']:shutil.copytree(base/n,case/n)
prefix=ROOT/'work/profile-native-trace'
with (D/'TEST_PLAN.md').open('a') as h:h.write('\n正式470原thermal100步/原flags新Nsight资源观察：新audit删除影响API/D2H计数，旧Co allocator细节可限定复用但不能把旧总API当新实测。独立profiler输入原字节，输出全文件对比；只评实际warm/stable allocation/API/transfer/kernel，诊断IO不计性能。非profiling同输入原终点5完整样本及离散程度另记，不用kernel提速外推求解器。\n')
t=run('NSYS-470-THERMAL100',cmd('nsys','profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx,osrt','--cuda-memory-usage=true','--output='+str(prefix),b,'-case',case),[2,14,15,16,26,27],expected='Original thermal100 native flags/source470, actual warm/stable resource trace; original output bytes invariant.',inputs=[p for n in ['0','constant','system'] for p in (base/n).rglob('*') if p.is_file()],binary=b,kind='DIAGNOSTIC',variant='FORMAL_PROFILING')
if t['exit_code']==0:
 rep=prefix.with_suffix('.nsys-rep');db=prefix.with_suffix('.sqlite');q=run('NSYS-470-EXPORT',cmd('nsys','export','--type=sqlite','--output='+str(db),rep),[16,27],expected='Export actual trace tables, no inference from kernel timing.',inputs=[rep],kind='ANALYSIS')
 if q['exit_code']==0:
  dest=D/'artifacts/nsys';dest.mkdir();c=sqlite3.connect(db);c.row_factory=sqlite3.Row;strings={x['id']:x['value'] for x in c.execute('select * from StringIds')};tables=[x[0] for x in c.execute("select name from sqlite_master where type='table'")];out={}
  for table in tables:
   if table!='StringIds' and not table.startswith('CUPTI_ACTIVITY_KIND_') and table not in ['CUDA_GPU_MEMORY_USAGE_EVENTS','CUDA_MEMORY','NVTX_EVENTS','OSRT_API']:continue
   cols=[x[1] for x in c.execute('pragma table_info('+table+')')];p=dest/(table+'.csv')
   with p.open('w',newline='') as h:w=csv.writer(h,quoting=csv.QUOTE_ALL,escapechar='\\');w.writerow(cols);w.writerows(c.execute('select * from '+table))
   out[table]={'path':str(p.relative_to(D)),'sha256':sha(p)}
  api=[{**dict(x),'name':strings.get(x['nameId'],str(x['nameId']))} for x in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')];ranges=[]
  for r in c.execute('select * from NVTX_EVENTS where end is not null order by start'):
   name=str(strings.get(r['textId'],r['text']))
   if name not in ['thrust::reduce','thrust::transform_reduce']:continue
   counts=collections.Counter(x['name'] for x in api if x['start']>=r['start'] and x['end']<=r['end'] and x['globalTid']==r['globalTid']);ranges.append({'index':len(ranges),'name':name,'start':r['start'],'end':r['end'],'API_counts':dict(counts)})
  memory=[];live=0;peak=0;corr={x['correlationId']:x['name'] for x in api}
  for r in c.execute('select * from CUDA_GPU_MEMORY_USAGE_EVENTS order by start'):
   r=dict(r);name=corr.get(r.get('correlationId'),'UNKNOWN');size=r.get('bytes',0);op='UNKNOWN'
   if 'Malloc' in name:live+=size;op='ALLOCATE'
   elif 'Free' in name:live-=size;op='FREE'
   peak=max(peak,live);memory.append({**r,'correlated_API':name,'operation':op,'live_bytes':live})
  comparison=[{'path':str(p.relative_to(base/'chmtOutput')),'formal_sha256':sha(p),'profile_sha256':sha(case/'chmtOutput'/p.relative_to(base/'chmtOutput')),'byte_equal':p.read_bytes()==(case/'chmtOutput'/p.relative_to(base/'chmtOutput')).read_bytes()} for p in (base/'chmtOutput').rglob('*') if p.is_file()]
  summary={'source_commit':SOURCE,'binary_sha256':sha(b),'tables':out,'runtime_API_counts':dict(collections.Counter(x['name'] for x in api)),'reduction_ranges':ranges,'warm_first_pair':ranges[:2],'stable_remaining_API_counts':dict(sum((collections.Counter(r['API_counts']) for r in ranges[2:]),collections.Counter())),'tracked_peak_cudaMalloc_live_bytes':peak,'tracked_final_live_bytes':live,'all_memory_events_resolved':all(r['operation']!='UNKNOWN' for r in memory),'original_output_bytes_equal':all(r['byte_equal'] for r in comparison),'output_comparison':comparison,'trace_sha256':sha(rep),'sqlite_sha256':sha(db),'limits':'Actual current API/kernel/copy/sync tables; tracked CUDA allocation bytes are not driver peak VRAM. First Co scratch pair vs stable same8cell input only. Profile durations are instrumented; no speed extrapolation.'}
  (dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');(dest/'all-memory-events.json').write_text(json.dumps(memory,indent=2)+'\n');shutil.copy2(rep,dest/rep.name);assertion(t,summary['original_output_bytes_equal'],'Native profile output byte invariance and actual resource tables; timing separate.',artifacts=['artifacts/nsys/summary.json']);assertion(q,True,'Actual raw trace export.',artifacts=['artifacts/nsys/summary.json'])
samples=[]
for i in range(5):
 sample=ROOT/'work'/('timing-thermal-'+str(i));sample.mkdir()
 for n in ['0','constant','system']:shutil.copytree(base/n,sample/n)
 t=run('TIMING-470-THERMAL100-'+str(i),cmd(b,'-case',sample),[16,27],expected='Separate nonprofiling native whole solver sample, same original thermal100; all output bytes unchanged.',binary=b,kind='GPU',variant='FORMAL_TIMING')
 equal=all(p.read_bytes()==(sample/'chmtOutput'/p.relative_to(base/'chmtOutput')).read_bytes() for p in (base/'chmtOutput').rglob('*') if p.is_file());samples.append({'sample':i,'elapsed_receipt_seconds':t['elapsed_seconds'],'resource_summary':t['resource_summary'],'exit_code':t['exit_code'],'output_bytes_equal':equal});assertion(t,t['exit_code']==0 and equal,'Full identical native input/flags/endpoint output comparison; wall sample only.')
(D/'artifacts/nonprofiling-samples.json').write_text(json.dumps({'samples':samples,'median':statistics.median(q['elapsed_receipt_seconds'] for q in samples),'sample_standard_deviation':statistics.stdev(q['elapsed_receipt_seconds'] for q in samples),'min':min(q['elapsed_receipt_seconds'] for q in samples),'max':max(q['elapsed_receipt_seconds'] for q in samples),'limits':'Receipt elapsed includes wrapper/startup; GNUtime resource wall separately retained; five current samples only, no new-old solver speed claim.'},indent=2)+'\n')
