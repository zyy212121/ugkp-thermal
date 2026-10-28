"""Native regression of boundary values actually consumed by the flux kernels.
Build all applications with NVCC_PREPEND_FLAGS=-DUGKP_BOUNDARY_FLUX_TRACE, then:
  python3 stage_flux_regression.py NEW_OUTPUT_DIRECTORY [gas|coupled|steady]
The marker is compile-time only; ordinary binaries have no probe overhead.
"""
from pathlib import Path
import json, os, re, shutil, struct, subprocess, sys
R=Path(__file__).resolve().parents[2];O=Path(sys.argv[1]).resolve();O.mkdir(parents=True,exist_ok=False)
selection=sys.argv[2] if len(sys.argv)>2 else 'all'
def entry(p,key,value):
    s,n=re.subn(r'\b'+key+r'\s+[^;]+;',key+' '+str(value)+';',p.read_text())
    if not n:s+='\n'+key+' '+str(value)+';\n'
    p.write_text(s)
def patch(c,field,name,body):
    p=c/'0'/field;s,n=re.subn(r'\b'+name+r'\s*\{[^}]*\}',name+' { '+body+' }',p.read_text());assert n==1;p.write_text(s)
def particles(c):
    entry(c/'constant/schedulingProperties','gpuResidentPureGasOnly','false')
    entry(c/'constant/schedulingProperties','gpuResidentParticleCapacity','50000')
    for k,v in [('parcelMass','1e-10'),('particleTemperatureTransport','true'),('particleRho','2800'),('particleCp','800'),('particleTMin','1'),('particleTMax','10000')]:entry(c/'constant/particleProperties',k,v)
    p=c/'0/epsilonS';p.write_text(p.read_text().replace('internalField uniform 0;','internalField nonuniform List<scalar> 16 ('+' '.join(['5.357142857142857e-10']+['0']*15)+');'))
    (c/'0/Tp').write_text('FoamFile {version 2.0;format ascii;class volScalarField;object Tp;}\ndimensions [0 0 0 1 0 0 0]; internalField uniform 300; boundaryField { inlet {type fixedValue;value uniform 300;} outlet {type zeroGradient;} wall {type zeroGradient;} top {type symmetryPlane;} sides {type empty;} }\n')
    with (c/'0/gpuResidentStrictParticles.dat').open('wb') as f:
        f.write(b'UGKP_PARTICLES_SCHEMA1_BIN 1 1\n');f.write(struct.pack('<I',1))
        for v in [.25,.005,.005,70,0,0,500,0,2e-4,1e-10]:f.write(struct.pack('<d',v))
        f.write(struct.pack('<iiQQ',0,1,12345,1))
routes=[('gasUGKP','single',False),('gasUGKP','mixture',False),('gasUGKP','mixture',True),('FSH','single',False),('CHT','single',False),('gasUGKPSteady','single',False),('gasUGKPSteady','mixture',True)]
results=[]
for solver,model,parcels in routes:
    group='coupled' if solver in ['FSH','CHT'] else 'steady' if solver=='gasUGKPSteady' else 'gas'
    if selection not in ['all',group,solver]:continue
    for order,scheme in [(1,'Euler'),(2,'SSPRK2'),(3,'SSPRK3')]:
        for graph in [0,1]:
            name=f'{solver}-{model}-particles{int(parcels)}-rk{order}-graph{graph}';c=O/name;c.mkdir()
            for part in ['0','constant','system']:shutil.copytree(R/'validation/shared_boundary/fixtures'/('cht' if solver=='CHT' else 'gas-total')/part,c/part)
            if model=='single':
                for f in ['gasModelProperties','gasTransportProperties','gasWallProperties']:(c/'constant'/f).unlink(missing_ok=True)
            else:(c/'constant/gasWallProperties').unlink(missing_ok=True)
            p=c/'constant/fluidProperties';p.write_text(re.sub(r'turbulence\s*\{[\s\S]*','turbulence { simulationType laminar; }\n',p.read_text()))
            if parcels:particles(c)
            entry(c/'system/controlDict','deltaT',1 if solver=='gasUGKPSteady' else '1e-7')
            entry(c/'system/controlDict','endTime',4 if solver=='gasUGKPSteady' else '4e-7')
            entry(c/'system/controlDict','adjustTimeStep','false')
            entry(c/'system/controlDict','writeControl','timeStep');entry(c/'system/controlDict','writeInterval',1)
            if solver=='gasUGKPSteady':entry(c/'system/controlDict','boundaryTime','1.25e-7')
            p=c/'system/fvSchemes';p.write_text(re.sub(r'(ddtSchemes\s*\{\s*default\s+)\w+',r'\g<1>'+scheme,p.read_text()))
            base=600 if solver=='CHT' else 300;inlet='right' if solver=='CHT' else 'inlet'
            patch(c,'T',inlet,f'type uniformFixedValue; uniformValue table ((0 {base}) (1e-6 {base+100})); value uniform {base};')
            if solver=='CHT':
                p=c/'constant/polyMesh/boundary';p.write_text(re.sub(r'(right\s*\{[^}]*?type\s+)wall',r'\1patch',p.read_text()))
                patch(c,'U','right','type fixedValue; value uniform (-1 0 0);')
            env=dict(os.environ,UGKP_GAS_GRAPH=str(graph),GAS_UGKP_CUDA_STACK_BYTES='65536',CHT_CUDA_STACK_BYTES='65536',OMP_NUM_THREADS='2')
            with (c/'run.log').open('w') as log:status=subprocess.run([R/'bin'/solver,'-case',c],stdout=log,stderr=subprocess.STDOUT,env=env).returncode
            text=(c/'run.log').read_text();assert status==0,(name,status,text[-4000:])
            # Tuning probes precede ToolB1 completion and are not time steps.
            if 'ToolB1:' in text:text=text.split('ToolB1:',1)[1]
            rows=[list(map(float,x.split())) for x in re.findall(r'^BOUNDARY_FLUX_INPUT (.*)$',text,re.M)]
            assert rows,(name,'no native probe output; rebuild with trace macro')
            face=min(x[0] for x in rows);sample=[x[2] for x in rows if x[0]==face]
            if solver=='gasUGKPSteady':
                expected=[base+12.5]*len(sample)
            else:
                offsets={1:[0],2:[0,1],3:[0,1,.5]}[order]
                expected=[base+10*(step+a) for step in range(4) for a in offsets]
            assert len(sample)==len(expected),(name,sample,expected)
            error=max(abs(a-b) for a,b in zip(sample,expected));assert error<1e-9,(name,sample,expected)
            times=[p for p in c.iterdir() if p.is_dir() and re.fullmatch('[0-9.eE+-]+',p.name)]
            last=max(times,key=lambda p:float(p.name))
            written_text=(last/'T').read_text()
            start=re.search(r'\b'+inlet+r'\s*\{',written_text).end();depth=1;end=start
            while depth:
                depth+=(written_text[end]=='{')-(written_text[end]=='}');end+=1
            section=written_text[start:end-1]
            written=float(re.search(r'\bvalue\s+uniform\s+([^;]+);',section)[1])
            expected_written=base+(12.5 if solver=='gasUGKPSteady' else 40)
            assert abs(written-expected_written)<1e-9,(name,written,expected_written)
            results.append(dict(name=name,face=int(face),consumed_T=sample,expected_T=expected,max_error=error,written_T=written))
            print(name,'PASS',sample,flush=True)
(O/'results.json').write_text(json.dumps(results,indent=2)+'\n')
