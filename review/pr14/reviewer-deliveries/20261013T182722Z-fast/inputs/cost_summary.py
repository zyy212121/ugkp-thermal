import pathlib,json,csv,sqlite3,collections
from run_fast import ROOT,D,sha
out=D/'artifacts/performance/nsys';summary={}
for v in ['baseline','original']:
 p=ROOT/'work/nsys-native'/('trace-'+v+'.sqlite');c=sqlite3.connect(p);j=json.load(open(out/(v+'-summary.json')));tables=[a[0] for a in c.execute("select name from sqlite_master where type='table'")]
 enums={t:c.execute('select * from '+t).fetchall() for t in tables if t.startswith('ENUM') and 'MEMORY' in t}
 (out/(v+'-memory-enums.json')).write_text(json.dumps(enums,indent=2)+'\n')
 events=list(csv.DictReader(open(out/(v+'-CUDA_GPU_MEMORY_USAGE_EVENTS.csv'))))
 categories=collections.Counter((q['memoryOperationType'],q['memKind']) for q in events)
 memory={'enums':enums,'event_counts':{str(k):n for k,n in categories.items()},'events_are_tracked_allocations_not_full_NV_driver_context_memory':True}
 names={int(k):str(val) for k,val in c.execute('select id,value from StringIds')}
 ranges=[]
 if 'NVTX_EVENTS' in tables:
  for text,textId,n,total in c.execute('select text,textId,count(*),sum(end-start) from NVTX_EVENTS group by text,textId'):
   ranges.append({'name':text or names.get(textId,str(textId)),'count':n,'total_ns':total})
 api={a['name']:a for a in j['cuda_runtime_API']}
 summary[v]={'kernel_count':sum(x['count'] for x in j['kernels']),'kernel_total_ms':sum(x['total_ns'] for x in j['kernels'])/1e6,'api':api,'device_transfers':j['device_transfers'],'memory':memory,'nvtx_ranges':ranges,'numeric_publication_equal_to_unprofiled':j['native_CSV_equal_to_unprofiled_same_variant']}
 print(v,'enums',enums,'categories',dict(categories),'ranges',ranges[:6])
 c.close()
(out/'cost-comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
