import shutil,sys,json,numpy as np
from r2_runner import *
sys.path.insert(0,str(ROOT/'common'));from thermal_particle_restart_current import iter_chunks,write_restart
with (D/'TEST_PLAN.md').open('a') as f:f.write('\nS214输入cell/ID实际字段校正：S213旧夹具假定pcell/pCell/pid字段名，真实current schema字段为cell/orig_id，因此实际16个不同位置却都绑定cell0，运行只保留3且无coldcontact，原count/profileFAIL保留为错误输入支持证据，不归为生产F004回归。按真实serialized field_specs改为cell0..15/orig_id1..16；其它所有S213已提交newinput量完全不变。初始profile记录实际取本轮S2111e-6真实GPU产物（不是旧ASCII输入直接解码），已在输入源SHA记录更正。新实际source/mesh/parameters不改、无新增数值条件，300s；要求实际count16和每parcel/time original1e-9及实际contact数。\n')
for app in ['CHT','FSH']:
 oldname=app+'-cold2D-explicit16-S213';name=app+'-cold2D-valid16-S214';case=ROOT/'work/cases/resident'/name;old=ROOT/'work/cases/resident'/oldname
 shutil.copytree(D/'inputs/native/resident'/oldname,case);shutil.copytree(old/'constant/polyMesh',case/'constant/polyMesh')
 p=case/'0/gpuResidentStrictParticles.dat';marker,data=next(iter_chunks(p));oldsha=sha(p);data['cell']=np.arange(16,dtype=data['cell'].dtype);data['orig_id']=np.arange(1,17,dtype=data['orig_id'].dtype)
 assert p.resolve().is_relative_to((ROOT/'work').resolve());p.unlink();write_restart(p,[data],16,16,cht=app=='CHT')
 inp=D/'inputs/native/resident'/name;shutil.copytree(case,inp,ignore=shutil.ignore_patterns('polyMesh'))
 (inp/'actual-input-provenance.json').write_text(json.dumps({'source_record':str(D/'artifacts/resident'/(app+'-cold2D-fullgroups-S211')/'1e-06/gpuResidentStrictParticles.dat'),'source_production_commit':SOURCE,'source_record_sha256':sha(D/'artifacts/resident'/(app+'-cold2D-fullgroups-S211')/'1e-06/gpuResidentStrictParticles.dat'),'corrected_fields':['cell','orig_id'],'before_input_sha256':oldsha,'after_input_sha256':sha(p),'actual_input_cells':data['cell'].tolist(),'actual_orig_ids':data['orig_id'].tolist(),'count':16},indent=2)+'\n')
 b=ROOT/'work/install/ns2/bin'/app;t=run('NATIVE-'+name,cmd(b,'-case',case),[5,12,18,19,20,23,26,33],expected='Actualcorrectcell-index16parcel input/realnative cooperativegroups; unchangedphysics/endpoint/threshold; independentperparcel profiles.',inputs=[q for q in inp.rglob('*') if q.is_file()],binary=b);save_case(case,name)
 amend('NATIVE-'+oldname,verdict='FAIL',observations='Wrongtestinitialcell bindings all0 for16positions; actualcount3/noactivatedcoldprofile, initial16-group scope FAIL; preservepointprofileerr .00155CHT/.001405FSH. Notconfirmedproductioncoldregression.')
