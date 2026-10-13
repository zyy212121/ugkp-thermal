import pathlib,json,shutil,re,hashlib
from run_fast import ROOT,D
for app in ['FSH','CHT']:
    src=D/'inputs/native/resident'/(app+'-periodic');p=src.with_name(src.name+'-INPUT1');shutil.copytree(src,p)
    q=p/'0/gpuResidentStrictParticles.dat';rows=q.read_text().splitlines();q.write_text('UGKP_PARTICLES_SCHEMA3 128\n'+'\n'.join(row+' 0 -1 0' for row in rows[1:])+'\n')
    if app=='CHT':
        q=p/'constant/particleProperties';q.write_text(q.read_text()+'\nparticleRho 1000;\n')
    shutil.copytree(p,ROOT/'work/cases/resident'/p.name)
for mode in ['coldWall1D','coldWall2D']:
    src=D/'inputs/native/resident'/('CHT-'+mode);p=src.with_name(src.name+'-INPUT1');shutil.copytree(src,p)
    original=ROOT/'examples/thermal/singleAluminaDrop/coldWall/0/gpuResidentStrictParticles.dat';rows=original.read_text().splitlines();assert rows[0]=='UGKP_PARTICLES_SCHEMA3 1'
    first=rows[1].split();first[:6]=['0.02','0.02','5e-7','0','0','-5'];(p/'0/gpuResidentStrictParticles.dat').write_text(rows[0]+'\n'+' '.join(first)+'\n')
    for q in p.rglob('*'):
        if q.is_file():assert not q.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1'),str(q)
    shutil.copytree(p,ROOT/'work/cases/resident'/p.name)
(D/'patches/P003-resident-input-adaptation.json').write_text(json.dumps({'classification':'test input adaptation','production_source_modified':False,'FSH':'common gas SCHEMA4 is not a thermal frontend input; use supported SCHEMA3 with the actual 3 legacy stuck columns. Original failed input and native logs retained.','CHT':'supply mandatory particleRho=1000 matching rhoS; supported SCHEMA3 exactly as FSH.','cold':'Fetched one pinned Git LFS input only; original initial header/row preserved and position/velocity changed to select first-step native impact; corrected malformed pointer-based draft retained as unexecuted preparation failure.','LFS_object_sha256':'23f77e2d293ba27bd5013d03df9aac67504ee5a41c6fc13117dfcb0caf964610','input_hashes':{str(q.relative_to(D)):hashlib.sha256(q.read_bytes()).hexdigest() for q in (D/'inputs/native/resident').rglob('*') if q.is_file() and '-INPUT1' in str(q)}},ensure_ascii=False,indent=2)+'\n')
print('Created isolated INPUT1 cases without changing physical parameter values.')
