import shutil,json,statistics
from r2_runner import *
from r2_native import compare_dirs
binary={'old-production':OLDROOT/'work/install/ns2/bin/CHMT','formal':ROOT/'work/install/ns2/bin/CHMT'}
rows=[];dest=D/'artifacts/timing';dest.mkdir(parents=True)
for phase,n in [('warmup',3),('measured',7)]:
 for i in range(n):
  for variant in (['old-production','formal'] if i%2==0 else ['formal','old-production']):
   case=ROOT/'work/timing'/(variant+'-'+phase+'-'+str(i));src=OLD/'inputs/native/chmt/thermal-full';shutil.copytree(src,case)
   for rel in ['constant/polyMesh','constant/solid/polyMesh']:shutil.copytree(OLDROOT/'work/cases/chmt/thermal-full'/rel,case/rel)
   t=run('TIMING-'+case.name,cmd(binary[variant],'-case',case),[2,9,14,16,26,27],expected='Matched100 microsteps/two windows;3warmups7alternated samples; alloriginalpublication bytes identical before speedratio.',inputs=[p for p in src.rglob('*') if p.is_file()],binary=binary[variant],variant=variant,kind='PERFORMANCE')
   cmp=compare_dirs(OLD/'artifacts/chmt/thermal-full',case/'chmtOutput')
   assertion(t,cmp['all_files_byte_equal'],'Performance only; alloriginal CSV/checkpoint/geometry bytes equal.',full_output_comparison=cmp)
   rows.append({'id':t['id'],'variant':variant,'phase':phase,'index':i,'seconds':t['elapsed_seconds'],'exit_code':t['exit_code'],'all_output_bytes_equal':cmp['all_files_byte_equal'],'binary_sha256':sha(binary[variant])})
   (dest/'samples.json').write_text(json.dumps(rows,indent=2)+'\n')
stats={}
for v in binary:
 a=[x['seconds'] for x in rows if x['variant']==v and x['phase']=='measured'];stats[v]={'samples':a,'median_seconds':statistics.median(a),'min_seconds':min(a),'max_seconds':max(a),'stdev_seconds':statistics.stdev(a)}
stats['ratio_formal_over802']=stats['formal']['median_seconds']/stats['old-production']['median_seconds'];stats['all_output_bytes_equal']=all(x['all_output_bytes_equal'] for x in rows);stats['scope']='Entire process startup/context/IO/two nativewindows100steps; distributions and background desktopGPU load apply. No extrapolation to other solver/config.'
(dest/'summary.json').write_text(json.dumps(stats,indent=2)+'\n');print(json.dumps(stats,indent=2))
