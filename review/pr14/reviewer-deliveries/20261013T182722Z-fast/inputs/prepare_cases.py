import pathlib,subprocess,json,hashlib,shutil,re
p=json.loads((pathlib.Path(__file__).parent/'paths.json').read_text());R=pathlib.Path(p['checkout']);D=pathlib.Path(p['delivery']);base=D/'inputs/native/chmt';base.mkdir(parents=True,exist_ok=True)
g=R/'applications/CHMT/verification/coupled_runtime/make_case.py';commands=[]
for scenario,opts,H in [('thermal',[],.0005),('single',['--single'],.0005),('diffusion',['--diffusion'],.0005),('chemistry',['--chemistry'],1e-5)]:
    variants=[('full',[]),('split',['--end-time',str(H)])]
    if scenario!='chemistry': variants.append(('refined',['--refine']))
    if scenario=='diffusion':variants.append(('control',['--diffusion-control']))
    if scenario=='chemistry':variants.append(('control',['--chemistry-control']))
    for name,extra in variants:
        c=base/(scenario+'-'+name)
        if c.exists():continue
        args=['python3',str(g),str(c),*(extra if name=='control' else opts+extra)]
        q=subprocess.run(args,capture_output=True,text=True);commands.append({'argv':args,'exit_code':q.returncode,'stdout':q.stdout,'stderr':q.stderr})
        if q.returncode: print(q.stderr);raise SystemExit(q.returncode)
        dst=R/'work/cases/chmt'/c.name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(c,dst)
files={str(q.relative_to(D)):hashlib.sha256(q.read_bytes()).hexdigest() for q in base.rglob('*') if q.is_file()}
(D/'logs/prepare-chmt-inputs.json').write_text(json.dumps({'commands':commands,'generator_sha256':hashlib.sha256(g.read_bytes()).hexdigest(),'input_hashes':files},indent=2)+'\n')
print('Prepared '+str(len(list(base.iterdir())))+' CHMT native small input sets; '+str(len(files))+' input files.')
for rel in ['examples/consistency/dustyBox/system/controlDict','examples/consistency/dustyBox/system/blockMeshDict','examples/consistency/dustyBox/constant/fluidProperties','examples/consistency/dustyBox/constant/particleProperties','examples/consistency/dustyBox/constant/schedulingProperties','examples/consistency/dustyBox/0/gpuResidentStrictParticles.dat','examples/consistency/dustyBox/constant/solidRegionProperties']:
    q=R/rel;print('\nFILE '+rel+'\n'+q.read_text()[:6000])
