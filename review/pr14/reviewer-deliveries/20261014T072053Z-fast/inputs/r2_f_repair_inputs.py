import re,shutil,json,difflib
from r2_runner import *
from r2_native import compare_dirs
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nF原始setup红保留：新wet-film/evaporation初始gas/solid共面，没有厚1e-5film的真实空间，正式attachCoupledSurface报告不匹配film-offset vertices，未启动GPU；S209只把gasmesh整体沿真实solidoutward+x平移原厚1e-5，保持cells/volumes、全部物性/终点/阈值原样，记录实际pointsdiff/hash；这是新合法测试输入，不能把旧无film空间输入标通过。\n新CHTcoupled/radiation缺少实际消费0/thermalExchangeState，原版按IO报错退出；S210按当前真实reader及初始state零ledger语义补入initialState=true、exchangeSequence0、两个clock0的初始dictionary，保留原错误，未修改任何生产reader/门禁/数值。重试各300s。\n')
for name in ['wet-film','wet-evaporation']:
 original=ROOT/'work/cases/extra'/name;case=ROOT/'work/cases/extra'/(name+'-S209');shutil.copytree(original,case)
 p=case/'constant/polyMesh/points';old=p.read_text();at=old.index('\n(\n');head=old[:at+3];body=old[at+3:];pat=r'\(\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*\)'
 body,n=re.subn(pat,lambda m:'('+format(float(m[1])+1e-5,'.17g')+' '+m[2]+' '+m[3]+')',body);assert n>0;p.write_text(head+body)
 (D/'patches'/('S209-'+name+'-gaspoints.patch')).write_text(''.join(difflib.unified_diff(old.splitlines(True),p.read_text().splitlines(True),fromfile='a/'+name+'/gasmesh/points',tofile='b/'+name+'-S209/gasmesh/points')))
 inp=D/'inputs/native/extra'/(name+'-S209');shutil.copytree(case,inp,ignore=shutil.ignore_patterns('polyMesh'));(inp/'gasmesh-points.dat').write_text(p.read_text());(inp/'geometry-support.json').write_text(json.dumps({'base_mesh_ref':str(original/'constant/polyMesh'),'base_points_sha256':hashlib.sha256(old.encode()).hexdigest(),'new_points_sha256':sha(p),'translation_m':[1e-5,0,0],'reason':'Actualfinitefilmspace required by existingcoupledshape; sameallvolumes/material/threshold/endtime.'},indent=2)+'\n')
 b=ROOT/'work/install/ns2/bin/CHMT';t=run('EXTRA-'+name+'-S209',cmd(b,'-case',case),[6,8,10,13,14,20,21,22,26,29],expected='Newlegitimate filmoffset geometry; originalnative GPU, allremaininginput parametersunchanged; fullstocks/ledger audit follows.',inputs=[q for q in inp.rglob('*') if q.is_file()],binary=b)
 if (case/'chmtOutput').exists():shutil.copytree(case/'chmtOutput',D/'artifacts/extra'/(name+'-S209'))
for name in ['CHT-actual-coupler','CHT-actual-radiation']:
 original=ROOT/'work/cases/resident'/name;case=ROOT/'work/cases/resident'/(name+'-S210');shutil.copytree(original,case,ignore=shutil.ignore_patterns('core','core.*'))
 p=case/'0/thermalExchangeState';p.write_text('formatVersion 4; initialState true; exchangeSequence "0"; completedTimeIndex 0; completedSimulationTimeS 0; completedRadiationSimulationTimeS 0; previousExchangeSimulationTimeS 0;\n')
 inp=D/'inputs/native/resident'/(name+'-S210');shutil.copytree(case,inp,ignore=shutil.ignore_patterns('polyMesh'))
 b=ROOT/'work/install/ns2/bin/CHT';t=run('NATIVE-'+name+'-S210',cmd(b,'-case',case),[18,19,20,23,26,30,33],expected='Actualfresh initialstate IO backed; originalnative real GPU/solid/DO/mixed singleface path. No source/physical/thresholdchanges.',inputs=[q for q in inp.rglob('*') if q.is_file()],binary=b);save_case(case,name+'-S210')
