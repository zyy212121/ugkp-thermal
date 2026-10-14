from r3_runner import *
import re
with (D/'TEST_PLAN.md').open('a') as h:h.write('\nS308单cell控制support输入坐标错误：模板solid位于[-1,0]，host probe recipe却假定[0,1]，原input gas已[1.001,2.001]造成实际interface gap1.001而非.001，native输入拒绝原log/input/hash完整保留。新独立S308 copy仅将solid points和其blockMeshDict全局平移+1以恢复已声明unitcube控制；温度/厚度/物性/end/数值标准不变、原wet与失败8cell未改。非零GPU验收仍需实际完成/oldlegalcontrol/observer，外部300s，不靠放宽几何阈值。\n')
for name in ['onecell-melting','onecell-solidification']:
 original=ROOT/'work/cases/phase'/name;case=original.parent/(name+'-S308');assert not case.exists();shutil.copytree(original,case,ignore=shutil.ignore_patterns('chmtOutput'))
 p=case/'constant/solid/polyMesh/points';s=p.read_text();m=re.search(r'\n8\s*\((.*?)\)\s*\n',s,re.S);assert m;points=[]
 for line in re.findall(r'\(([^()]*)\)',m[1]):
  v=list(map(float,line.split()));v[0]+=1;points.append('('+' '.join(format(q,'.17g') for q in v)+')')
 assert len(points)==8;p.write_text(s[:m.start(1)]+'\n'+'\n'.join(points)+'\n'+s[m.end(1):]);shutil.copytree(case,D/'inputs/phase'/case.name)
 print('S308READY',case.name,flush=True)
