import pathlib,json,shutil,re,hashlib
from run_fast import ROOT,D
base=ROOT/'examples/consistency/dustyBox'
out=D/'inputs/native/resident';out.mkdir(parents=True,exist_ok=True)
for app in ['gasUGKP','CHT','FSH']:
    p=out/(app+'-periodic')
    for sub in ['0','constant','system']:
        (p/sub).mkdir(parents=True,exist_ok=True)
        for src in (base/sub).iterdir():
            if src.is_file() and src.name!='gpuResidentStrictParticles.dat':shutil.copy2(src,p/sub/src.name)
    q=p/'system/blockMeshDict';x=q.read_text().replace('(100 1 1)','(4 1 1)')
    x=x.replace('left\n    {\n        type patch;','left\n    {\n        type cyclic;\n        neighbourPatch right;').replace('right\n    {\n        type patch;','right\n    {\n        type cyclic;\n        neighbourPatch left;');q.write_text(x)
    for q in (p/'0').iterdir():q.write_text(q.read_text().replace('type zeroGradient;','type cyclic;'))
    q=p/'system/controlDict';x=q.read_text();x=re.sub(r'application\s+[^;]+;',f'application {app};',x);x=re.sub(r'endTime\s+[^;]+;','endTime 1e-5;',x);x=re.sub(r'writeInterval\s+[^;]+;','writeInterval 5e-6;',x);x=re.sub(r'writePrecision\s+[^;]+;','writePrecision 17;',x);q.write_text(x)
    q=p/'constant/particleProperties';x=q.read_text();x=re.sub(r'(?:^|\n)(parcelMass|injectionParcelMass|legacyRestartParcelMass)\s+[^;]+;',lambda m:'\n'+m[1]+' 0.00078125;',x)
    if app=='CHT':x+='\nparticleCp 1000;\nparticleTemperatureTransport true;\nparticleTMin 200;\nparticleTMax 4000;\n'
    q.write_text(x)
    q=p/'constant/schedulingProperties';x=q.read_text().replace('21000','256').replace('256;','128;');q.write_text(x)
    rows=['UGKP_PARTICLES_SCHEMA4 128']
    for i in range(128):rows.append(f'{(i+.5)/128:.17g} 0.5 0.5 0 0 0 300 0 0.0001 0 1 {(24681357+i*11400714819323198485)&((1<<64)-1)} {i//32}')
    (p/'0/gpuResidentStrictParticles.dat').write_text('\n'.join(rows)+'\n')
    dst=ROOT/'work/cases/resident'/p.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(p,dst)
    spec={'app':app,'cells':4,'particles':128,'particle_mass':.00078125,'steps':10,'dt':1e-6,'periodic':True,'random_seed':24681357,'expected':'Closed uniform periodic dusty box: all actual gas/particle fields written; mass/momentum/energy inventory and split/restart compared.','tolerances':'mass/momentum abs1e-10+rel1e-10; energy abs1e-7+rel1e-12; same-binary restart full fields abs1e-12+rel1e-11; independent transfer/energy path limits explicitly recorded.'}
    (p/'case-spec.json').write_text(json.dumps(spec,indent=2)+'\n')
wall=ROOT/'examples/thermal/singleAluminaDrop/coldWall'
for mode in ['coldWall1D','coldWall2D']:
    p=out/('CHT-'+mode)
    for sub in ['0','constant','system']:
        (p/sub).mkdir(parents=True,exist_ok=True)
        for src in (wall/sub).iterdir():
            if src.is_file():shutil.copy2(src,p/sub/src.name)
    q=p/'system/controlDict';x=q.read_text();x=re.sub(r'application\s+[^;]+;','application CHT;',x);x=re.sub(r'endTime\s+[^;]+;','endTime 1e-5;',x);x=re.sub(r'deltaT\s+[^;]+;','deltaT 1e-6;',x);x=re.sub(r'writeInterval\s+[^;]+;','writeInterval 1e-6;',x);x=re.sub(r'adjustTimeStep\s+[^;]+;','adjustTimeStep false;',x);x=re.sub(r'writePrecision\s+[^;]+;','writePrecision 17;',x);q.write_text(x)
    q=p/'constant/particleProperties';q.write_text(q.read_text().replace('wallInteractionModel coldWall1D;','wallInteractionModel '+mode+';'))
    q=p/'0/gpuResidentStrictParticles.dat';x=q.read_text();rows=x.splitlines();first=rows[1].split();print(mode+' ORIGINAL '+rows[1])
    # Position and velocity select an actual first-step impact, preserving the
    # model, thermodynamic properties, particle representation and wall input.
    first[:6]=['0.02','0.02','5e-7','0','0','-5'];q.write_text(rows[0]+'\n'+' '.join(first)+'\n')
    dst=ROOT/'work/cases/resident'/p.name;shutil.copytree(p,dst)
    (p/'case-spec.json').write_text(json.dumps({'app':'CHT','wall_model':mode,'cells':10,'particles':1,'dt':1e-6,'steps':10,'expected':'Actual impact establishes cold profile and double residual, wall/gas exchange, final terminal fields and checkpoint; energy evaluated with physical represented particle mass and carry.','precision':'CHT float enthalpy profile, all temperature/physical residual/ledger double.'},indent=2)+'\n')
hashes={str(q.relative_to(D)):hashlib.sha256(q.read_bytes()).hexdigest() for q in out.rglob('*') if q.is_file()}
(D/'logs/prepare-resident-inputs.json').write_text(json.dumps({'input_hashes':hashes},indent=2)+'\n')
with (D/'TEST_PLAN.md').open('a') as f:f.write('\n## 追加 RESIDENT 参数\n\n周期dustyBox：4 cells/128 parcels，单位体积，gas U=1、particle U=0、T=300、solid mass=.1kg、seed24681357，10步dt1e-6；gasUGKP/CHT/FSH实际frontend/backend，共用物理输入与各自支持配置。质量/动量abs1e-10+rel1e-10，能量abs1e-7+rel1e-12；同binary restart全字段abs1e-12+rel1e-11。原生cold1D/2D：原singleAluminaDrop物性与10-cell网格，一颗原表示粒子位置z=5e-7、速度z=-5、10步dt1e-6，首步实际撞壁；profile float的quantization通过double残余与完整ledger核算，而非放宽物理库存验收。输入哈希在prepare-resident-inputs.json及执行前spec。各例300秒。\n')
print('Prepared resident and native cold-wall FAST inputs.')
