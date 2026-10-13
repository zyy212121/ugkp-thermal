import pathlib,shutil,sys,json,re
from run_fast import ROOT,D,run,cmd,amend
def san(id,binary,args,tool,inputs,topics):
    t=run(id,cmd('compute-sanitizer','--tool',tool,'--target-processes','all','--error-exitcode','86',binary,*args),topics,300,'Applicable real CUDA '+tool+'; error summary must be0. No intentional invalid indices. Tool-specific coverage only, numerical nonfinite is separate.',inputs=inputs,binary=binary,kind='SANITIZER')
    raw=(D/t['log']).read_text(errors='replace');zero=('ERROR SUMMARY: 0 errors' in raw or tool=='racecheck' and 'RACECHECK SUMMARY: 0 hazards' in raw)
    amend(t['id'],verdict='PASS' if t['exit_code']==0 and zero else 'FAIL' if t['exit_code'] else 'INCONCLUSIVE',observations=raw[-6000:])
for tool in ['memcheck','initcheck','synccheck','racecheck']:
    san('SAN-CHEM-'+tool,ROOT/'work/chemistry-retry-pr',['1','2e-6','100'],tool,[D/'inputs/chemistry_retry.cu'],[10,12,26,28])
for name in ['gasUGKP-periodic-PHYS1','FSH-periodic-INPUT1-PHYS1','CHT-periodic-INPUT2-PHYS1','CHT-coldWall1D-INPUT2','CHT-coldWall2D-INPUT2']:
    app=name.split('-')[0];original=D/'inputs/native/resident'/name
    if not original.is_dir():continue
    for tool in (['memcheck','initcheck','synccheck'] if 'coldWall2D' in name else ['memcheck']):
        case=ROOT/'work/sanitizer-cases'/(name+'-'+tool);case.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(original,case)
        mesh=run('SAN-MESH-'+name+'-'+tool,cmd('blockMesh','-case',case),[8,28],300,'Independent fresh native mesh for sanitizer, same physical input.',inputs=list(original.rglob('*')),kind='SETUP')
        if mesh['exit_code']:continue
        san('SAN-NATIVE-'+name+'-'+tool,ROOT/'work/install/ns2/bin'/app,['-case',str(case)],tool,[q for q in original.rglob('*') if q.is_file()],[12,18,26,28,31]+([33] if 'coldWall' in name else []))
for app in ['gasUGKP','FSH','CHT']:
    b=ROOT/'work/cuda-operators'/('AUTO-'+app)/'gas_auto_pool'
    if b.is_file():
        for tool in ['memcheck','racecheck','synccheck']:
            san('SAN-AUTO-'+app+'-'+tool,b,['2','0','128'],tool,[b.with_suffix('.cu')],[12,26,28,31])
