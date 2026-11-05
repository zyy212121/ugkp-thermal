import sqlite3,json,sys
from pathlib import Path
db=sqlite3.connect(sys.argv[1]);out=Path(sys.argv[2])
count,start,end,total=db.execute('select count(*),min(start),max(end),sum(end-start) from CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
inside=db.execute('select count(*),coalesce(sum(bytes),0) from CUPTI_ACTIVITY_KIND_MEMCPY where start < ? and end > ?',(end,start)).fetchone()
copies=list(db.execute('select e.label,count(*),sum(m.bytes) from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on m.copyKind=e.id group by e.label'))
assert count==100 and inside==(0,0),(count,inside)
r={'graph_kernel_count':count,'copies_during_kernel_interval':inside[0],'bytes_during_kernel_interval':inside[1],'kernel_total_ns':total,'all_copies_by_kind':copies}
out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
