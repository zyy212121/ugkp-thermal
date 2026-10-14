import re,shutil,json
from r2_runner import *
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nB多个完整thermal组的证据缺口：原fixture四粒子逐thread初始化不能证明整8lane求解。新增CHT/FSH同原cold2D物性/d=.005/终点1e-5/dt1e-6，在原每cell尺寸(.04,.04,.001)保留的4×4×10真实mesh横向复制成160cells，首层16cells各一个原parcel（原epsilonS与mass），按已有16个独立parcel触发至少两个完整8lane组；其他层粒子库存0，非uniform字段从原首cell逐份复制，不改阈值。原warm相区2373.15沿用，FSH仅对应原有效wall tokens1800/710/100和application。逐输出时刻全部profile doublemean+carry/FSHdouble与pT abs1e-9。新几何是组覆盖输入，不是原fail输入替换。外部300s。\n')
for app in ['CHT','FSH']:
 name=app+'-cold2D-fullgroups-S211';case=ROOT/'work/cases/resident'/name;shutil.copytree(OLD/'inputs/native/resident/CHT-coldWall2D-INPUT2',case)
 q=case/'system/blockMeshDict';s=q.read_text().replace('0.04','0.16').replace('(1 1 10)','(4 4 10)');q.write_text(s)
 for q in (case/'0').iterdir():
  if not q.is_file():continue
  s=q.read_text();m=re.search(r'internalField\s+nonuniform\s+List<(scalar|vector)>\s+10\s*\((.*?)\)\s*;',s,re.S)
  if not m:continue
  kind=m[1];values=re.findall(r'\([^()]*\)',m[2]) if kind=='vector' else m[2].split();assert len(values)==10
  rows=[values[0]]*16+[values[1]]*144
  q.write_text(s[:m.start()]+'internalField nonuniform List<'+kind+'>\n160\n(\n'+'\n'.join(rows)+'\n);'+s[m.end():])
 if app=='FSH':
  q=case/'system/controlDict';q.write_text(q.read_text().replace('application CHT;','application FSH;'))
  q=case/'constant/particleProperties';s=q.read_text();s=re.sub(r'wallDensity\s+[^;]+;','wallDensity 1800;',s);s=re.sub(r'wallSpecificHeat\s+[^;]+;','wallSpecificHeat 710;',s);s=re.sub(r'wallThermalConductivity\s+[^;]+;','wallThermalConductivity 100;',s);q.write_text(s)
 inp=D/'inputs/native/resident'/name;shutil.copytree(case,inp)
 t=run('MESH-'+name,cmd('blockMesh','-case',case),[23,26,33],expected='Physicaloldcellsize transverse repetition160cells with16validactualparcels.',inputs=[q for q in inp.rglob('*') if q.is_file()],kind='SETUP');assertion(t,t['exit_code']==0,'Realnewmesh')
 b=ROOT/'work/install/ns2/bin'/app;t=run('NATIVE-'+name,cmd(b,'-case',case),[5,12,18,19,20,23,26,33],expected='Actualfullnative multiplecomplete thermal8lane groups samephysics andsteps, alltime original1e-9profile consistency.',inputs=[q for q in inp.rglob('*') if q.is_file()],binary=b);save_case(case,name)
 checker=D/'inputs'/('resident_check_fsh_double.py' if app=='FSH' else 'resident_check_fp64_mean.py');ck=run('FIELDS-'+name,cmd('python3',checker,'finite',name),[23,26,33],expected='Allwrittenfields finite; allwritten initializedprofiles doublemean/carry andpT abs1e-9, samecaloric.',inputs=[checker],kind='ANALYSIS');raw=(D/ck['log']).read_text();assertion(ck,ck['exit_code']==0,raw);assertion(t,t['exit_code']==0 and ck['exit_code']==0,'Actualnewphysicalmesh/multiparcelnative; originalmean/caloricchecks '+ck['verdict'])
