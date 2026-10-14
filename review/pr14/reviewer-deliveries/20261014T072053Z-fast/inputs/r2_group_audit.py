import importlib.util,json,math,numpy as np,sys
from r2_runner import *
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nS212独立逐parcel校验补充：旧S201/S204 checker通过case名字coldWall分支选择profile断言；新S211名字cold2D只触发有限性检查。保留原调用及其有限性证据，原native记录不得声称该调用已做profile。新增显式无名字判断的每parcel/每时刻double顺序累加profile、doublecarry（FSH现存double无carry字段按0）、原caloric inverse及1e-9阈值；同时核真实particle_count和动态contactAge/状态，不能仅用planned16计数。CHT真实coupler/DO也使用同物理字段校验，并逐写时积分真实壁面4个heatflux与solid库存。所有只是现有真实GPU产物分析，不新增GPU运行。\n')
sp=importlib.util.spec_from_file_location('cold_original_checker',D/'inputs/resident_check_fp64_mean.py');helper=importlib.util.module_from_spec(sp);sp.loader.exec_module(helper)
def field(p,count):
 import re
 s=p.read_text();m=re.search(r'internalField\s+uniform\s+([^;]+);',s)
 if m:return [float(m[1])]*count
 m=re.search(r'internalField\s+nonuniform\s+List<scalar>\s+(\d+)\s*\((.*?)\)\s*;',s,re.S);assert m;return list(map(float,m[2].split()))
reports={}
for name in (sys.argv[1:] or ['CHT-cold2D-fullgroups-S211','FSH-cold2D-fullgroups-S211','CHT-actual-coupler-S210','CHT-actual-radiation-S210']):
 folder=D/'artifacts/resident'/name;states=[];fail=[]
 for q in sorted(folder.iterdir(),key=lambda p:float(p.name)):
  if not(q/'gpuResidentStrictParticles.dat').exists():continue
  p=helper.particles(q/'gpuResidentStrictParticles.dat');n=len(p['pT']);nodes=p['pCold2DNodeSpecificEnthalpy'].reshape(n,-1);carry=p.get('pThermalEnthalpyResidual',np.zeros(n));rows=[]
  for i in range(n):
   mean=sum(float(x) for x in nodes[i])/nodes.shape[1];T=helper.caloric(mean+float(carry[i]));err=abs(float(p['pT'][i])-T);row={'particle':i,'mean_double':mean,'carry_double':float(carry[i]),'pT':float(p['pT'][i]),'inverse_temperature':T,'abs_error_K':err};rows.append(row)
   if not(math.isfinite(err) and err<=1e-9):fail.append({'time':q.name,**row})
  states.append({'time':q.name,'particle_count':n,'profile_dtype':str(nodes.dtype),'rows':rows,'maximum_error_K':max(x['abs_error_K'] for x in rows),'stuck_states':p.get('pStuck',np.zeros(n)).tolist(),'positive_contact_age_count':int(np.count_nonzero(p.get('pContactAge',np.zeros(n))>0)),'positive_cold2D_contact_age_count':int(np.count_nonzero(p.get('pCold2DRingContactAge',np.zeros(n))>0))})
 report={'name':name,'states':states,'failures':fail,'maximum_error_K':max(x['maximum_error_K'] for x in states),'all_counts':[x['particle_count'] for x in states],'criteria':'No case-name shortcut. Everyphysicalparcel at everyoutput; profile mean sequentialfloat64sum, unchangedcaloric/1e-9K. FSHprofile itselfmustfloat64; nofictionalcarryfield. Dynamiccontacts reported, notinferred.','all_pass':not fail and bool(states) and all(x['profile_dtype']=='float64' for x in states) if name.startswith('FSH') else not fail and bool(states)};dest=D/'artifacts/resident'/(name+'-S212-profile-audit.json');dest.write_text(json.dumps(report,indent=2)+'\n');reports[name]=report
 t=run('PROFILE-ALL-PARCELS-'+name,['python3','-c','print("Allactualparcel/timecaloric pointdata captured; externaldriver audit")'],[23,26,33],expected='Everyparcel/time profile+carry agreement abs1e-9 and realgroupcounts.',inputs=[dest,D/'inputs/resident_check_fp64_mean.py'],kind='ANALYSIS');assertion(t,report['all_pass'],'Actualn='+str(report['all_counts'])+' maxerror='+str(report['maximum_error_K'])+' K; activecontacts='+str([x['positive_cold2D_contact_age_count'] for x in states]),artifacts=[str(dest.relative_to(D))])
 if 'S211' in name:amend('NATIVE-'+name,verdict='PASS' if report['all_pass'] else 'FAIL',observations='Realnative 160cells; oldname-filter checker onlyfinite. S212 explicitlychecks everyparcel/profile/time max='+str(report['maximum_error_K'])+'K, counts='+str(report['all_counts'])+', activecontacts='+str([x['positive_cold2D_contact_age_count'] for x in states]),artifacts=[str(dest.relative_to(D))])
 print(name,report['all_pass'],report['maximum_error_K'],report['all_counts'],[x['positive_cold2D_contact_age_count'] for x in states],flush=True)
(D/'artifacts/resident/new-profile-all-parcels-S212.json').write_text(json.dumps(reports,indent=2)+'\n')
