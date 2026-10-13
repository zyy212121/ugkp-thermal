import pathlib,shutil,re,json,hashlib
from run_fast import ROOT,D
src=D/'inputs/native/resident/CHT-periodic-INPUT1';p=src.with_name('CHT-periodic-INPUT2');shutil.copytree(src,p);text=(p/'0/T').read_text().replace('object      T;','object      Tp;');(p/'0/Tp').write_text(text)
shutil.copytree(p,ROOT/'work/cases/resident'/p.name)
for mode in ['coldWall1D','coldWall2D']:
    p=D/'inputs/native/resident'/('CHT-'+mode+'-INPUT2');original=ROOT/'examples/thermal/singleAluminaDrop/coldWall'
    for sub in ['0','constant','system']:
        (p/sub).mkdir(parents=True,exist_ok=True)
        for q in (original/sub).iterdir():
            if q.is_file():
                assert not q.read_bytes().startswith(b'version https://git-lfs'),str(q)
                shutil.copy2(q,p/sub/q.name)
    q=p/'system/controlDict';x=q.read_text();
    for key,value in [('application','CHT'),('endTime','1e-5'),('deltaT','1e-6'),('writeInterval','1e-6'),('adjustTimeStep','false'),('writePrecision','17')]:x=re.sub(r'\b'+key+r'\s+[^;]+;',key+' '+value+';',x)
    q.write_text(x)
    q=p/'constant/particleProperties';q.write_text(q.read_text().replace('wallInteractionModel coldWall1D;','wallInteractionModel '+mode+';'))
    q=p/'0/gpuResidentStrictParticles.dat';lines=q.read_text().splitlines();row=lines[1].split();row[:6]=['0.02','0.02','5e-7','0','0','-5'];q.write_text(lines[0]+'\n'+' '.join(row)+'\n')
    shutil.copytree(p,ROOT/'work/cases/resident'/p.name)
(D/'patches/P003-resident-input-adaptation.json').write_text(json.dumps({'classification':'test-input-only','production_source_modified':False,'changes':['Thermal frontends need schema3 with original 3 stuck columns, instead of gas-only schema4.','CHT particleRho=1000 exactly matches rhoS; Tp field=300 exactly matches original particle initial T.','cold1D/2D exact pinned LFS small fields downloaded/verified; position/velocity selects first-step impact as preplanned.'],'failed_inputs':'Original and INPUT1 copies and logs retained; no physical threshold/solver behavior modified.','input_hashes':{str(q.relative_to(D)):hashlib.sha256(q.read_bytes()).hexdigest() for q in (D/'inputs/native/resident').rglob('*') if q.is_file() and ('-INPUT1' in str(q) or '-INPUT2' in str(q))}},ensure_ascii=False,indent=2)+'\n')
print('Ready: CHT-periodic-INPUT2, CHT-coldWall1D-INPUT2, CHT-coldWall2D-INPUT2')
