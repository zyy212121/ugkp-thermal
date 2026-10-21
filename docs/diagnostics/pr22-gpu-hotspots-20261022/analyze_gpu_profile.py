from pathlib import Path
import sqlite3,json,statistics
r=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr19-math-merged-20261018');p=r/'work/pr22-native-cuda-20261021/profile-2f4a66f-20steps';results={}
for label in ['old','new']:
 c=sqlite3.connect(p/(label+'.sqlite'));c.row_factory=sqlite3.Row
 ks=[dict(v)for v in c.execute('select k.*,s.value name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.shortName=s.id order by k.start')]
 bvps=[v for v in ks if v['name']=='evaluateGasBoundaryLayerTeamKernel'];assert len(bvps)==40
 lo,hi=bvps[4]['start'],bvps[38]['start'];groups={}
 for v in ks:
  if lo<=v['start']<hi:groups.setdefault(v['name'],[]).append(v)
 rows={n:dict(count=len(vs),total_s=sum(v['end']-v['start']for v in vs)/1e9,per_step_ms=sum(v['end']-v['start']for v in vs)/17e6,median_call_ms=statistics.median(v['end']-v['start']for v in vs)/1e6)for n,vs in groups.items()}
 mem=c.execute('select sum(end-start) from CUPTI_ACTIVITY_KIND_MEMCPY where start>=? and start<?',(lo,hi)).fetchone()[0]/1e9
 api=[dict(v)for v in c.execute('select s.value name,count(*) count,sum(r.end-r.start)/1e9 total_s from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id where r.start>=? and r.start<? group by s.value order by total_s desc',(lo,hi))]
 gpu=sum(v['total_s']for v in rows.values());span=(hi-lo)/1e9
 assert all(a['end']<=b['start']for a,b in zip(ks,ks[1:])), 'No overlap required for additive kernel attribution'
 results[label]=dict(range='step3 start to step20 start; 17 full steps (3..19)',span_s=span,mean_step_s=span/17,kernel_s=gpu,kernel_fraction_of_span=gpu/span,memcpy_s=mem,kernels=rows,api=api,by_step_bvp_ms=[sum(v['end']-v['start']for v in bvps[2*i:2*i+2])/1e6 for i in range(20)])
delta=results['new']['span_s']-results['old']['span_s'];results['delta_s']=delta
results['delta_by_kernel']={n:dict(delta_s=results['new']['kernels'].get(n,{}).get('total_s',0)-results['old']['kernels'].get(n,{}).get('total_s',0),fraction_of_step_span_delta=(results['new']['kernels'].get(n,{}).get('total_s',0)-results['old']['kernels'].get(n,{}).get('total_s',0))/delta)for n in results['new']['kernels']}
(p/'analysis.json').write_text(json.dumps(results,indent=2));print('Verified 40 BVP calls per version, identical interval mapping, no GPU kernel overlap; exported analysis.json')
