import sqlite3,csv,json,shutil
from r2_runner import *
from r2_native import compare_dirs
src=OLD/'inputs/native/chmt/thermal-full';mesh=OLDROOT/'work/cases/chmt/thermal-full';case=ROOT/'work/profile-formal';shutil.copytree(src,case)
for rel in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(mesh/rel,case/rel)
prefix=ROOT/'work/profile-formal-trace';binary=ROOT/'work/install/ns2/bin/CHMT'
t=run('NSYS-FORMAL',cmd('nsys','profile','--sample=none','--cpuctxsw=none','--trace=cuda,nvtx,osrt','--cuda-memory-usage=true','--output='+str(prefix),binary,'-case',case),[2,9,12,14,15,16,26,27],expected='Same thermal100 microsteps/flags/outputs; measure actual allocations/kernel/D2H/sync/livepeak; profiler not timing speedratio.',inputs=[p for p in src.rglob('*') if p.is_file()],binary=binary,kind='PROFILING')
if t['exit_code']:raise SystemExit(1)
rep=prefix.with_suffix('.nsys-rep');db=prefix.with_suffix('.sqlite')
e=run('NSYS-EXPORT',cmd('nsys','export','--type=sqlite','--output='+str(db),rep),[16,27],expected='Export actual native trace; quote-all CSV reusedP012 serialization.',inputs=[rep],kind='ANALYSIS')
if e['exit_code']:raise SystemExit(1)
dest=D/'artifacts/nsys';dest.mkdir(parents=True)
conn=sqlite3.connect(db);tables=[x[0] for x in conn.execute("select name from sqlite_master where type='table'")];columns={q:[x[1] for x in conn.execute('pragma table_info('+q+')')] for q in tables}
strings={x[0]:x[1] for x in conn.execute('select id,value from StringIds')}
selected=['StringIds']+[q for q in tables if q.startswith('CUPTI_ACTIVITY_KIND_') or q in ['CUDA_GPU_MEMORY_USAGE_EVENTS','CUDA_MEMORY','NVTX_EVENTS','OSRT_API']];tableinfo={}
for table in selected:
 rows=conn.execute('select * from '+table).fetchall();p=dest/(table+'.csv')
 with p.open('w',newline='') as f:w=csv.writer(f,quoting=csv.QUOTE_ALL,escapechar='\\');w.writerow(columns[table]);w.writerows(rows)
 tableinfo[table]={'row_count':len(rows),'sha256':sha(p),'path':str(p.relative_to(D))}
apis=[{'name':strings.get(k,str(k)),'count':n,'total_ns':total,'min_ns':lo,'max_ns':hi} for k,n,total,lo,hi in conn.execute('select nameId,count(*),sum(end-start),min(end-start),max(end-start) from CUPTI_ACTIVITY_KIND_RUNTIME group by nameId')]
ks=columns['CUPTI_ACTIVITY_KIND_KERNEL'];name='demangledName' if 'demangledName' in ks else 'shortName'
kernels=[{'name':strings.get(k,str(k)),'count':n,'total_ns':total,'min_ns':lo,'max_ns':hi} for k,n,total,lo,hi in conn.execute('select '+name+',count(*),sum(end-start),min(end-start),max(end-start) from CUPTI_ACTIVITY_KIND_KERNEL group by '+name)]
transfers=[{'copy_kind':k,'count':n,'bytes':size,'total_ns':ns} for k,n,size,ns in conn.execute('select copyKind,count(*),sum(bytes),sum(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY group by copyKind')]
comparison=compare_dirs(D/'artifacts/chmt/thermal-full',case/'chmtOutput');(dest/'output-comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
report={'source_commit':SOURCE,'binary_sha256':sha(binary),'raw_trace':{'local_path':str(rep),'sha256':sha(rep),'bytes':rep.stat().st_size},'sqlite':{'local_path':str(db),'sha256':sha(db),'bytes':db.stat().st_size},'table_schemas':columns,'exported_tables':tableinfo,'cuda_runtime_API':apis,'kernels':kernels,'device_transfers':transfers,'all_original_output_bytes_equal':comparison['all_files_byte_equal'],'reuse_reference':{'commit':PATHS['old_delivery_commit'],'path':'review/pr14/reviewer-deliveries/20261013T182722Z-fast/artifacts/performance/nsys/original-summary.json','sha256':sha(OLD/'artifacts/performance/nsys/original-summary.json')},'limitation':'Profiler times are instrumented/overlap; no speed ratio. CUDA API memory events cover tracked live/peakbytes, not driver/context peakVRAM. Warm/stable Co-scope and allocation costs analyzed separately.'}
(dest/'formal-summary.json').write_text(json.dumps(report,indent=2)+'\n')
assertion(t,comparison['all_files_byte_equal'] and bool(kernels),'Same100 microstep native publication allbytes equal; actual trace fully exported.',artifact='artifacts/nsys/formal-summary.json')
print([(x['name'],x['count']) for x in apis if any(k in x['name'] for k in ['Malloc','Free','Memcpy','Synchronize','Launch'])]);print('NVTXcols',columns.get('NVTX_EVENTS'))
