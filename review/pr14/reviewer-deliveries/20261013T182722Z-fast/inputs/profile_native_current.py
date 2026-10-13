import pathlib,shutil,json,sqlite3,csv,subprocess
from run_fast import ROOT,D,run,cmd,amend,sha
src=D/'inputs/native/chmt/thermal-full';mesh=ROOT/'work/cases/chmt/thermal-full';out=D/'artifacts/performance/nsys';out.mkdir(parents=True,exist_ok=True)
with (D/'TEST_PLAN.md').open('a') as f:f.write('\n追加Nsight Systems2025.5.2：原base/PR native CHMT相同两thermal窗口，cuda/nvtx/osrt trace+memory usage，禁CPU采样。300s/例。用于实测kernel/API/复制/同步/分配计数与时间、显存临时分配；profiler开销下的时间不作提速比例，正式3warm/7sample窗口墙钟另存。raw .nsys-rep/sqlite本地保留、哈希/生成命令/完整关键tables CSV自包含交付，避免无意义二进制。检查actual CSV对各自未profiling输出一致。\n')
summary={}
for variant in ['baseline','original']:
 case=ROOT/'work/nsys-native'/variant;
 if not case.exists():shutil.copytree(src,case)
 for relative in ['constant/polyMesh','constant/solid/polyMesh']:
  if not (case/relative).exists():shutil.copytree(mesh/relative,case/relative)
 prefix=ROOT/'work/nsys-native'/('trace-'+variant);binary=ROOT/('work/install/base/bin/CHMT' if variant=='baseline' else 'work/install/ns2/bin/CHMT')
 if variant=='baseline':
  t={'id':'NSYS-NATIVE-baseline'};rep=prefix.with_suffix('.nsys-rep');db=prefix.with_suffix('.sqlite')
 else:
  t=run('NSYS-NATIVE-'+variant,cmd('nsys','profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx,osrt','--cuda-memory-usage=true','--output='+str(prefix),'--force-overwrite=true',binary,'-case',case),[2,16,26,27,29],300,'Native trace of matched successful workload; CUDA API/kernel/copy/sync/alloc actual costs, profiler time not speedup.',inputs=list(src.rglob('*')),binary=binary,variant=variant,kind='PERFORMANCE')
  if t['exit_code']:continue
  rep=prefix.with_suffix('.nsys-rep');db=prefix.with_suffix('.sqlite')
  export=run('NSYS-EXPORT-'+variant,cmd('nsys','export','--type=sqlite','--output='+str(db),rep),[16,27],300,'Export actual trace to inspect independent kernel/API timings and counts.',inputs=[rep],kind='ANALYSIS',variant=variant)
  if export['exit_code']:continue
 conn=sqlite3.connect(db);tables=[r[0] for r in conn.execute("select name from sqlite_master where type='table'")];columns={table:[r[1] for r in conn.execute('pragma table_info('+table+')')] for table in tables}
 selected=['StringIds']+[q for q in tables if q.startswith('CUPTI_ACTIVITY_KIND_') or q in ['CUDA_GPU_MEMORY_USAGE_EVENTS','CUDA_MEMORY','NVTX_EVENTS','OSRT_API']]
 tableinfo={}
 for table in selected:
  q=conn.execute('select * from '+table);cols=[a[0] for a in q.description];rows=q.fetchall();p=out/(variant+'-'+table+'.csv')
  with p.open('w') as h:w=csv.writer(h,quoting=csv.QUOTE_ALL,escapechar=chr(92));w.writerow(cols);w.writerows(rows)
  tableinfo[table]={'columns':cols,'row_count':len(rows),'csv':str(p.relative_to(D)),'sha256':sha(p)}
 strings={int(k):v for k,v in conn.execute('select id,value from StringIds')} if 'StringIds' in tables else {}
 api=[]
 table='CUPTI_ACTIVITY_KIND_RUNTIME'
 if table in tables:
  q=conn.execute('select nameId,count(*),sum(end-start),min(end-start),max(end-start) from '+table+' group by nameId')
  api=[{'name':strings.get(int(k),str(k)),'count':count,'total_ns':total,'min_ns':mn,'max_ns':mx} for k,count,total,mn,mx in q]
 kernels=[];table='CUPTI_ACTIVITY_KIND_KERNEL'
 if table in tables:
  name='demangledName' if 'demangledName' in columns[table] else 'shortName'
  q=conn.execute('select '+name+',count(*),sum(end-start),min(end-start),max(end-start) from '+table+' group by '+name)
  kernels=[{'name':strings.get(int(k),str(k)),'count':count,'total_ns':total,'min_ns':mn,'max_ns':mx} for k,count,total,mn,mx in q]
 transfers=[];table='CUPTI_ACTIVITY_KIND_MEMCPY'
 if table in tables:
  transfers=[{'copy_kind':kind,'count':count,'total_bytes':size,'total_ns':total} for kind,count,size,total in conn.execute('select copyKind,count(*),sum(bytes),sum(end-start) from '+table+' group by copyKind')]
 expected=D/'artifacts/performance/native'/(variant+'-preflight-0');equal=all(q.name in [p.name for p in (case/'chmtOutput').glob('*.csv')] and q.read_bytes()==(case/'chmtOutput'/q.name).read_bytes() for q in expected.glob('*.csv'))
 report={'source_variant':variant,'raw_trace':{'path':str(rep),'sha256':sha(rep),'bytes':rep.stat().st_size},'sqlite':{'path':str(db),'sha256':sha(db),'bytes':db.stat().st_size},'schema':columns,'exported_tables':tableinfo,'cuda_runtime_API':api,'kernels':kernels,'device_transfers':transfers,'native_CSV_equal_to_unprofiled_same_variant':equal,'interpretation':'CUDA API host durations overlap device activity. Do not sum host/kernel/wait into wall time; no measured-speed ratio from instrumented trace. NVTX window ranges may be absent; no inferred internal window CPU time.'}
 (out/(variant+'-summary.json')).write_text(json.dumps(report,indent=2)+'\n');summary[variant]=report
 amend(t['id'],verdict='PASS' if equal and kernels else 'INCONCLUSIVE',observations='Actual Nsight tables exported; native full CSV equals unprofiled same source. Timing includes profiler overhead, counts physical workload.',artifact=str((out/(variant+'-summary.json')).relative_to(D)))
 conn.close()
print('PROFILE COMPLETE',[(k,len(v['kernels']),len(v['cuda_runtime_API'])) for k,v in summary.items()])
