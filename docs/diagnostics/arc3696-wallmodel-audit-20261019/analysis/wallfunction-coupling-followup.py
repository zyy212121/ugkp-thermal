from pathlib import Path
import json,csv,subprocess,hashlib,shutil
import numpy as np
O=Path('/mnt/d/tempt/wallfunction-vs-boundarylayer-20261019')
S=Path('/home/lss/OpenFOAM/lss-10/applications/solvers/ugkp-thermal-test/pr19-math-merged-20261018')
P=Path('/mnt/c/Users/25094/Documents/Codex/2026-10-16/1-uniform-work-accepted-source-manifest/work')
text=(P/'wallfunction-cross-model.cpp').read_text().replace('int main(){','int main(int argc,char** argv){').replace('auto&in=model.input;','auto&in=model.input;model.config.nodes=argc>1?std::atoi(argv[1]):24;').replace('WallWorkspace<double,2,24>','WallWorkspace<double,2,128>').replace('l+1<23','l+1<model.config.nodes-1')
source=O/'wallfunction-node-convergence.cpp';source.write_text(text)
cmd=['g++','-std=c++17','-O2','-I'+str(S),'-I'+str(S/'common'),'-I'+str(O),str(source),'-o',str(O/'wallfunction-node-convergence')]
r=subprocess.run(cmd,capture_output=True,text=True);(O/'node-convergence-build.log').write_text(r.stdout+r.stderr);assert r.returncode==0,r.stderr
commands=[];result=[]
for nodes in [24,48,96,128]:
 dest=O/f'node-convergence-{nodes}.csv';args=[str(O/'wallfunction-node-convergence'),str(nodes)]
 with(O/'same-input-wall-models.csv').open()as inp,dest.open('w')as out,(O/f'node-convergence-{nodes}.log').open('w')as err:r=subprocess.run(args,stdin=inp,stdout=out,stderr=err)
 commands.append({'command':args,'exit':r.returncode});assert r.returncode==0
 for row in csv.DictReader(dest.open()):row['nodes']=nodes;result.append(row)
(O/'node-convergence-commands.json').write_text(json.dumps({'build':cmd,'build_exit':0,'runs':commands},indent=2))
for row in result:
 if row['face']=='53702':print(row['dataset'],'nodes',row['nodes'],'ok',row['ok'],'tau',row['BL_shear'],'omega',row['BL_ownerOmega'],'heat',row['BL_heat'],'residual',row['residual'])
profiles=list(csv.DictReader((O/'baseline-internal-profiles.csv').open()));inputs=list(csv.DictReader((O/'same-input-wall-models.csv').open()));averages=[]
for x in [.35,.38,.58,.75,.85]:
 rows=sorted([r for r in profiles if float(r['x_station'])==x],key=lambda r:float(r['y']));face=rows[0]['face'];i=next(r for r in inputs if r['dataset']=='boundaryLayer-old'and r['face']==face)
 y=np.array([float(r['y'])for r in rows]);u=np.array([float(r['Ux'])for r in rows]);t=np.array([float(r['T'])for r in rows]);height=2*float(i['y']);grid=np.unique(np.r_[y[y<height],height]);up=np.interp(grid,y,u);temp=np.interp(grid,y,t);invT=1/temp
 avg=float(np.trapz(up*invT,grid)/np.trapz(invT,grid));averages.append({'x_station':x,'face':face,'owner_y':float(i['y']),'owner_height_for_this_orthogonal_rectangular_mesh':height,'coarse_owner_U':float(i['ownerUx']),'profile_point_U_at_owner_y':float(np.interp(float(i['y']),y,u)),'piecewise_linear_profile_mass_weighted_owner_average_U':avg})
(O/'profile-coarse-consistency.json').write_text(json.dumps(averages,indent=2));print('AVERAGE',averages[2])
for p in [P/'wallfunction-loglayer-probe.cpp',P/'wallfunction-coupling-followup.py']:shutil.copy2(p,O/p.name)
