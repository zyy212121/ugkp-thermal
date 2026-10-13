import pathlib,shutil,sys
from run_fast import ROOT,D,run,cmd,amend
topics=[4,5,7,8,9,12,17,18,19,26,29,31]
for name in (sys.argv[1:] or ['gasUGKP-periodic','FSH-periodic','CHT-periodic','CHT-coldWall1D','CHT-coldWall2D']):
    case=ROOT/'work/cases/resident'/name;app=name.split('-')[0];binary=ROOT/'work/install/ns2/bin'/app;inputs=[p for p in case.rglob('*') if p.is_file()]
    mesh=run('MESH-'+name,cmd('blockMesh','-case',case),[8,29],300,'Real 4-cell periodic or 10-cell cold-wall mesh; independently generated small input.',inputs=inputs,kind='SETUP')
    amend(mesh['id'],verdict='PASS' if mesh['exit_code']==0 else 'BLOCKED')
    if mesh['exit_code']:continue
    t=run('NATIVE-'+name,cmd(binary,'-case',case),topics+([20,33] if app=='CHT' else []),300,'Actual native short run and full terminal field/restart writes. Inventory acceptance is independent of process exit.',inputs=inputs,binary=binary)
    dest=D/'artifacts/resident'/name;dest.mkdir(parents=True,exist_ok=True)
    for p in case.iterdir():
        try:float(p.name);numeric=True
        except ValueError:numeric=False
        if numeric and p.name!='0':shutil.copytree(p,dest/p.name)
    amend(t['id'],artifacts=[str(dest.relative_to(D))])
    print((D/t['log']).read_text(errors='replace')[-4500:],flush=True)
