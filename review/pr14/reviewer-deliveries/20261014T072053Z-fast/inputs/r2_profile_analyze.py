import json,sqlite3,collections,statistics
from r2_runner import *
db=ROOT/'work/profile-formal-trace.sqlite';c=sqlite3.connect(db);c.row_factory=sqlite3.Row
strings={r['id']:r['value'] for r in c.execute('select * from StringIds')};print('memorycols',list(dict(c.execute('select * from CUDA_GPU_MEMORY_USAGE_EVENTS limit 1').fetchone()).keys()))
print('memoryfirst',*[dict(x) for x in c.execute('select * from CUDA_GPU_MEMORY_USAGE_EVENTS order by start limit 3')],sep='\n')
print('NVTX first',*[{**dict(x),'resolved':strings.get(x['textId'],x['text'])} for x in c.execute('select * from NVTX_EVENTS where end is not null order by start limit 6')],sep='\n')
api=[{**dict(x),'name':strings.get(x['nameId'],str(x['nameId']))} for x in c.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME')]
names=collections.Counter(strings.get(x['textId'],x['text']) for x in c.execute('select * from NVTX_EVENTS'));print('NVTXNAMES',names)
report=json.loads((D/'artifacts/nsys/formal-summary.json').read_text());old=json.loads((OLD/'artifacts/performance/nsys/original-summary.json').read_text());diff=[]
oa={x['name']:x for x in old['cuda_runtime_API']};fa={x['name']:x for x in report['cuda_runtime_API']}
for k in sorted(set(oa)|set(fa)):diff.append({'name':k,'old802_count':oa.get(k,{}).get('count',0),'formal_count':fa.get(k,{}).get('count',0),'delta':fa.get(k,{}).get('count',0)-oa.get(k,{}).get('count',0)})
ranges=[]
for row in c.execute('select * from NVTX_EVENTS where end is not null order by start'):
 name=str(strings.get(row['textId'],row['text']))
 if name not in ['thrust::reduce','thrust::transform_reduce']:continue
 events=[x for x in api if x['start']>=row['start'] and x['end']<=row['end'] and x['globalTid']==row['globalTid']]
 counts=collections.Counter(x['name'] for x in events)
 ranges.append({'index':len(ranges),'name':name,'start':row['start'],'end':row['end'],'cuda_API_counts':dict(counts)})
mem=[dict(x) for x in c.execute('select * from CUDA_GPU_MEMORY_USAGE_EVENTS order by start')];api_corr={x['correlationId']:x['name'] for x in api};live=0;peak=0;events=[]
for x in mem:
 name=api_corr.get(x.get('correlationId'),'UNKNOWN');size=x.get('bytes',0)
 if 'Malloc' in name:live+=size;op='ALLOCATE'
 elif 'Free' in name:live-=size;op='FREE'
 else:op='UNKNOWN'
 peak=max(peak,live);events.append({**x,'correlated_API':name,'operation':op,'tracked_live_bytes':live})
result={'API_count_changes':diff,'Co_reduction_ranges':ranges,'Co_warm_first_pair':ranges[:2],'Co_stable_remaining_API_counts':dict(sum((collections.Counter(x['cuda_API_counts']) for x in ranges[2:]),collections.Counter())),'tracked_peak_cudaMalloc_live_bytes':peak,'tracked_final_live_bytes':live,'all_memory_operations_resolved':all(x['operation']!='UNKNOWN' for x in events),'scope':'Actualmemorysize/scope from CUDA_API correlation/NVTX only. Firstpair is initialCo scratch allocation warmup; stable means remaining same8cell thermal scale. Alloriginal kernels remain. Retainedscratch freed atBackenddestruction; no fullGPUpeakclaim.'}
(D/'artifacts/nsys/resource-analysis.json').write_text(json.dumps(result,indent=2)+'\n');(D/'artifacts/nsys/all-memory-events-correlated.json').write_text(json.dumps(events,indent=2)+'\n');print('peak',peak,'end',live,'ranges',len(ranges));print('firstpair',ranges[:2])
