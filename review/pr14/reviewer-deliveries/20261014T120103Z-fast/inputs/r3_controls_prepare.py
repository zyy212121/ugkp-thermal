from r3_runner import *
import re
with (D/'TEST_PLAN.md').open('a') as h:h.write('\n新增等价host-probe配方的一cell真实GPU非零phase控制（不是修改失败8cell输入）：8cell两方向正式/旧版均300s未提交窗口，D302显示localdt约7e-15且真实signedmass，原失败和local诊断不覆盖完整成功。按已存在tests/fixtures/chmt_zero_phase_material/probe.C合法控制独立构造unitcube onecell、gas800/solid700/film800、delta.001、rho1000/cp1000/k2、liquid e0=200000、viscosity1/gas.01，Tm600正方向/900负方向。原wet/evap及已失败8cell输入不改。外部每native/旧safe/readonly300s，不因阻断无限改输入。sameinput/flags bitcompare，全physical字段/收支单列，能否native完成实测。\n')
base=ROOT/'work/cases/extra/wet-film-formal'
for name,tm in [('onecell-melting',600),('onecell-solidification',900)]:
 case=ROOT/'work/cases/phase'/name
 if not case.exists():shutil.copytree(base,case,ignore=shutil.ignore_patterns('chmtOutput'))
 for rel in ['constant/polyMesh','constant/solid/polyMesh']:
  if (case/rel).exists() and not any(q['id']=='MESH-'+name+('-solid' if 'solid' in rel else '-gas') for q in json.loads((D/'results.json').read_text())['tests']):shutil.rmtree(case/rel)
 for rel in ['system/blockMeshDict','system/blockMeshSolidDict']:
  p=case/rel;p.write_text(p.read_text().replace('(2 2 2)','(1 1 1)'))
 p=case/'system/controlDict';p.write_text(re.sub(r'deltaT\s+[^;]+;','deltaT 1e-6;',re.sub(r'endTime\s+[^;]+;','endTime 1e-6;',p.read_text())))
 p=case/'constant/chmtProperties';s=p.read_text().replace('gasConductivity 1; gasViscosity 0;','gasConductivity 2; gasViscosity .01;').replace('conductivity 1;','conductivity 2;').replace('thickness 1e-5;','thickness .001;').replace('pressure 171600;','pressure 229600;').replace('couplingInterval 0.00050000000000000001; gasMaxDt 1.0000000000000001e-05;','couplingInterval 1e-6; gasMaxDt 1e-6;').replace('mean 350;','mean 800;').replace('e0 100000;','e0 200000;').replace('liquidViscosity .01; liquidReferencePressure 171600;','liquidViscosity 1; liquidReferencePressure 0;')
 s+='\nmeltTemperature '+str(tm)+';\nmaterial { phaseCondensed 0; enableMelting true; phaseFilmY (1 0); evaporation Disabled; }\nmeshMotion { policy CoupledRecession; }\n';p.write_text(s)
 for rel,value in [('0/T','800'),('0/rho','1'),('0/solid/solidEnergyDensity','700000000')]:
  p=case/rel;p.write_text(re.sub(r'internalField uniform [^;]+;','internalField uniform '+value+';',p.read_text()))
 p=case/'constant/gasModelProperties';s=p.read_text().replace('coefficients (1040 0 0);','coefficients (1000 0 0);').replace('molarMass 0.028;','molarMass 0.028970601456447055;').replace('molarMass 0.032;','molarMass 0.028970601456447055;');p.write_text(s)
 for region,dictname in [('', 'blockMeshDict'),('solid','blockMeshSolidDict')]:
  args=cmd('blockMesh','-case',case)
  if region:args+=['-region',region,'-dict',str(case/'system'/dictname)]
  existing=next((q for q in json.loads((D/'results.json').read_text())['tests'] if q['id']=='MESH-'+name+('-solid' if region else '-gas')),None)
  t=existing or run('MESH-'+name+('-solid' if region else '-gas'),args,[3,6,13,14,20,21,26,29],300,'Nativelegalonecellmesh unitcube from existinghostprobe; noGPU/physicsacceptance.',inputs=[case/'system'/dictname],kind='SETUP');assertion(t,t['exit_code']==0,'Actualonecellmeshsetup');assert t['exit_code']==0
 p=case/'constant/polyMesh/points';s=p.read_text();m=re.search(r'\n8\s*\((.*?)\)\s*\n',s,re.S);assert m
 points=[]
 for line in re.findall(r'\(([^()]*)\)',m[1]):
  v=list(map(float,line.strip().strip('()').split()));assert len(v)==3;v[0]+=1.001;points.append('('+' '.join(format(q,'.17g') for q in v)+')')
 assert len(points)==8;p.write_text(s[:m.start(1)]+'\n'+'\n'.join(points)+'\n'+s[m.end(1):])
 (case/'gasmesh-points.dat').unlink(missing_ok=True)
 shutil.copytree(case,D/'inputs/phase'/name)
 print('ONECELLREADY',name,flush=True)
# Authoritative ASCII mobile ingress, current actual schema and unchanged parcel mass.
payload=old_read('inputs/native/resident/CHT-coldWall2D-INPUT2/0/gpuResidentStrictParticles.dat').decode();print('ASCII_ORIGINAL',payload)
assert 'SCHEMA3' in payload and '-5' in payload
assert git('diff',PATHS['old_source'],SOURCE,'--','applications/CHT','applications/FSH','common')==''
for app in ['CHT','FSH','FSHCudaBackend']:
 p=OLDROOT/'work/install/ns2/bin'/app;q=ROOT/'work/install/ns2/bin'/app;assert p.is_file();shutil.copy2(p,q);assert sha(p)==sha(q) and q.resolve().is_relative_to(Path(PATHS['repo']).resolve())
receipt={'unchanged_production_dependencies':True,'source_comparison':[PATHS['old_source'],SOURCE],'reused_binary_producer':PATHS['old_source'],'files':{a:sha(ROOT/'work/install/ns2/bin'/a) for a in ['CHT','FSH','FSHCudaBackend']}}
(D/'metadata/resident-binary-reuse.json').write_text(json.dumps(receipt,indent=2)+'\n')
for app in ['CHT','FSH']:
 oldcase=OLDROOT/('work/cases/resident/'+app+'-cold2D-valid16-S214');case=ROOT/('work/cases/resident/'+app+'-active16-ASCII-S306');assert not case.exists();shutil.copytree(oldcase,case,ignore=shutil.ignore_patterns('gpuResidentStrict.log','postProcessing','[1-9]*'))
 # Copy only unchanged physical input trees; prior time directories are not initial data.
 for p in list(case.iterdir()):
  if p.is_dir() and p.name not in ['0','system','constant']:shutil.rmtree(p)
 lines=payload.splitlines();records=[q for q in lines if q.strip() and not q.startswith('#') and len(q.split())==16];assert len(records)==1,records
 row=records[0].split();assert float(row[5])==-5 and float(row[2])==5e-7 and int(row[13])==0
 rows=[]
 for i in range(16):
  r=row.copy();r[0]=format(.02+.04*(i%4),'.17g');r[1]=format(.02+.04*(i//4),'.17g');r[9]=str(i);r[12]=str(i+1);rows.append(' '.join(r))
 new='\n'.join(q.replace(' 1',' 16',1) if 'SCHEMA3' in q else q for q in lines if q not in records)+'\n'+'\n'.join(rows)+'\n';p=case/'0/gpuResidentStrictParticles.dat';p.write_text(new)
 shutil.copytree(case,D/'inputs/resident'/case.name,ignore=shutil.ignore_patterns('polyMesh'))
 (D/'metadata'/('active16-'+app+'-input-provenance.json')).write_text(json.dumps({'original_fixed_commit':PATHS['old_delivery_commit'],'original_alias':'inputs/native/resident/CHT-coldWall2D-INPUT2/0/gpuResidentStrictParticles.dat','original_sha256':hashlib.sha256(payload.encode()).hexdigest(),'new_sha256':sha(p),'original_raw':payload,'new_raw':new,'only_record_changes':['x','y','cell','orig_id','SCHEMA3 count'],'mass_from_unchanged_particleProperties':(case/'constant/particleProperties').read_text(),'geometry_sha256':{str(q.relative_to(case)):sha(q) for q in (case/'constant/polyMesh').iterdir() if q.is_file()}},indent=2)+'\n')
 print('COLDREADY',app,new[:360],flush=True)
