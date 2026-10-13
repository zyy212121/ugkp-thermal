import pathlib,shutil,json
from run_fast import ROOT,D
for name in ['gasUGKP-periodic','FSH-periodic-INPUT1','CHT-periodic-INPUT2']:
    src=D/'inputs/native/resident'/name;p=src.with_name(name+'-PHYS1');shutil.copytree(src,p)
    q=p/'0/gpuResidentStrictParticles.dat';rows=q.read_text().splitlines();new=[]
    for i,row in enumerate(rows[1:]):
        values=row.split();values[9]=str(i//32);values[12]=str(i);new.append(' '.join(values))
    q.write_text(rows[0]+'\n'+'\n'.join(new)+'\n');shutil.copytree(p,ROOT/'work/cases/resident'/p.name)
with (D/'TEST_PLAN.md').open('a') as f:f.write('\n## 追加 PHYS1：粒子owner输入修正\n\n准备脚本将schema4/3的cell与orig_id列误写：全部cell0、orig_id重复0–3；与已规划均匀128粒子位置/4-cell库存不一致。保留原候选数据与输出，物理库存判定另用PHYS1：仅cell=i//32、orig_id=i，固定位置/速度/表示质量/seed/所有物性/步长完全不变。它不是生产源修复，旧输出不用于合法均匀闭合案例验收。\n')
print('Corrected particle cell and unique ID input columns; physical values unchanged.')
