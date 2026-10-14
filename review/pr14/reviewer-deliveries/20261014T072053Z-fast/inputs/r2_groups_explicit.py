import re,json,shutil,sys,numpy as np
from r2_runner import *
sys.path.insert(0,str(ROOT/'common'));from thermal_particle_restart_current import iter_chunks,write_restart
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nS213整组输入校正：S211实际每时刻只有1particle，旧P003提供的初始restart覆盖了新的epsilonS，不能当16组验收。保留该范围失败，S212逐parcel一致性数据仍有效。新独立初始文件通过未变current serializer/P003用途，将原合法单parcel状态复制16份：同mass/d/T/velocity/profile/carry，将各点置于160cellmesh底层16个不同cell中心x/y、同原pz；cell IDs0..15和原seed状态分别记录。没有扩大ε/物性/终点/阈值，显式16个真实输入与场库存相符，原文件/hash保留固定上轮证据。各native300s，要求实际每输出count16及所有profile1e-9；需观察真实contact和完整8lane组，不以输入计数推断运行。\n')
for app in ['CHT','FSH']:
 oldname=app+'-cold2D-fullgroups-S211';name=app+'-cold2D-explicit16-S213';oldcase=ROOT/'work/cases/resident'/oldname;case=ROOT/'work/cases/resident'/name
 if not case.exists():
  shutil.copytree(D/'inputs/native/resident'/oldname,case)
  shutil.copytree(oldcase/'constant/polyMesh',case/'constant/polyMesh')
 p=case/'0/gpuResidentStrictParticles.dat';base=D/'artifacts/resident'/oldname/'1e-06/gpuResidentStrictParticles.dat';marker,c=next(iter_chunks(base));originalsha=sha(p);data={k:np.repeat(v,16,axis=0) for k,v in c.items()};assert len(c['pm'])==1
 for i in range(16):
  data['px'][i]=.02+.04*(i%4);data['py'][i]=.02+.04*(i//4);data['pz'][i]=1e-6
 for key in ['pStuck','pContactAge','pContactDuration','pColdContactAge','pCold2DRingContactAge']:
  if key in data:data[key][:]=0
 if 'pStuckFaceId' in data:data['pStuckFaceId'][:]=-1
 for key in ['pcell','pCell','pCellId']:
  if key in data:data[key]=np.arange(16,dtype=data[key].dtype)
 for key in ['pid','pId','pId64']:
  if key in data:data[key]=np.arange(1,17,dtype=data[key].dtype)
 if app=='FSH':data.pop('pThermalEnthalpyResidual',None)
 assert p.resolve().is_relative_to((ROOT/'work').resolve());p.unlink();write_restart(p,[data],16,16,cht=app=='CHT')
 inp=D/'inputs/native/resident'/name;shutil.copytree(case,inp,ignore=shutil.ignore_patterns('polyMesh'))
 (inp/'explicit-initial-physical-state.json').write_text(json.dumps({'source_single_particle_fixed_commit':PATHS['old_delivery_commit'],'source_single_particle_sha256':originalsha,'new_file_sha256':sha(p),'field_specs':{k:{'dtype':str(v.dtype),'shape':list(v.shape)} for k,v in data.items()},'physical_inputs':{k:data[k].tolist() for k in ['px','py','pz','pm','pT','pux','puy','puz']},'cells':{k:data[k].tolist() for k in data if k.lower() in ['pcell','pcellid']},'observed_count_required':16},indent=2)+'\n')
 b=ROOT/'work/install/ns2/bin'/app;t=run('NATIVE-'+name,cmd(b,'-case',case),[5,12,18,19,20,23,26,33],expected='16explicitvalid actualinitialparticle records; originalfullnative thermal8lane groups, alltime profile original1e-9 checkfollows.',inputs=[q for q in inp.rglob('*') if q.is_file()],binary=b);save_case(case,name)
 amend('NATIVE-'+oldname,verdict='INCONCLUSIVE',observations='Nativeonly1actualparticle (oldprovidedinitialfileoverridesε field), perparceltemperaturePASS but plannedmultiplecompletegroups notcovered; S211 result not16-group acceptance.')
