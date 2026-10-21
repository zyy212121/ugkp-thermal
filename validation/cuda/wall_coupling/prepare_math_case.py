#!/usr/bin/env python3
"""Create the frozen mixed/no-blow 24-node production case from tracked evidence.

Requires numpy. --reference is the verified, extracted mathematical archive.
The template is the repository's original restart archive. Saved initial state
and boundary compositions reproduce the mathematical fixture's manufactured
mixture transformation, rather than silently using the template's pure gas.
"""
import argparse,hashlib,json,re,tarfile,shutil
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
def replace_internal(path, values, vector=False):
    text=path.read_text()
    begin=text.index('internalField');end=text.index('boundaryField',begin)
    if vector:
        rows=['('+' '.join(format(v,'.17g') for v in row)+')' for row in values]
    else:
        rows=[format(v,'.17g') for v in values]
    content='internalField nonuniform List<'+('vector' if vector else 'scalar')+'>\n'+str(len(rows))+'\n(\n'+'\n'.join(rows)+'\n);\n'
    path.write_text(text[:begin]+content+text[end:])

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=20)
    parser.add_argument('--model',choices=['boundaryLayer','wallFunction'],default='boundaryLayer')
    parser.add_argument('--integrator',choices=['Euler','SSPRK2'],default='SSPRK2')
    parser.add_argument('--audit-input',type=Path,help='New raw mixed-state/mesh directory for audit_euler.py')
    args=parser.parse_args()
    out=args.output.resolve()
    if out.exists():raise SystemExit('Output must be new: '+str(out))
    out.mkdir(parents=True)
    archive=ROOT/'docs/diagnostics/bvp-coupling-20261020/restart-inputs.tar.gz'
    with tarfile.open(archive) as tar:
        for member in tar:
            prefix='cases/boundaryLayer/'
            if not member.name.startswith(prefix) or not member.isfile():continue
            rel=Path(member.name[len(prefix):])
            if '..' in rel.parts or rel.is_absolute():raise ValueError(member.name)
            if rel.parts[0] not in ['0','constant','system']:continue
            target=out/rel;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(tar.extractfile(member).read())
    initial=args.reference/'source-sign-audit/math-gap-20261021T0626Z/directional-candidate/TY-directional-200us/initialized-state'
    def raw(name):return np.fromfile(initial/(name+'.bin'),dtype='<f8')
    rho=raw('rho')
    for dest,src in [('rho','rho'),('rhoE','rhoE'),('p','p'),('T','Tgas'),('k','k'),('omega','omega')]:
        replace_internal(out/'0'/dest,raw(src))
    replace_internal(out/'0/U',np.column_stack([raw('U'+d) for d in 'xyz']),True)
    boundary=raw('boundaryMassFraction').reshape(2,-1)
    mesh=(out/'constant/polyMesh/boundary').read_text()
    for j,kind in enumerate(['Air','CO2']):
        path=out/'0'/('Y_'+kind)
        replace_internal(path,raw('speciesRho'+kind)/rho)
        text=path.read_text()
        for patch in ['inlet','top']:
            body=re.search(r'\b'+patch+r'\s*\{(.*?)\}',mesh,re.S)[1]
            count=int(re.search(r'nFaces\s+(\d+)',body)[1]);start=int(re.search(r'startFace\s+(\d+)',body)[1])
            values=boundary[j,start:start+count]
            block=patch+' { type fixedValue; value nonuniform List<scalar>\n'+str(count)+'\n(\n'+'\n'.join(format(v,'.17g') for v in values)+'\n); }'
            text=re.sub(r'\b'+patch+r'\s*\{[^}]*\}',lambda _:block,text)
        path.write_text(text)
    path=out/'constant/gasWallProperties'
    text=path.read_text();text=re.sub(r'speciesFlux\s*\([^)]*\)','speciesFlux (0 0)',text);path.write_text(text)
    path=out/'constant/fluidProperties'
    text=path.read_text();text=re.sub(r'nodes\s+\d+','nodes 24',text);text=re.sub(r'maxIterations\s+\d+','maxIterations 60',text)
    text=re.sub(r'wallTreatment\s+\w+','wallTreatment '+args.model,text);path.write_text(text)
    path=out/'system/fvSchemes'
    text=path.read_text();text=re.sub(r'(ddtSchemes\s*\{\s*default\s+)\w+',lambda m:m[1]+args.integrator,text);path.write_text(text)
    path=out/'system/controlDict';text=path.read_text()
    for name,value in [('startFrom','startTime'),('adjustTimeStep','false'),('maxDeltaT','2e-7'),('writeControl','timeStep'),('startTime','0'),('endTime',format(args.steps*2e-7,'.17g')),('deltaT','2e-7'),('writeInterval',str(args.steps)),('logInterval','1')]:
        text=re.sub(r'\b'+name+r'\s+[^;]+;',name+' '+value+';',text)
    path.write_text(text)
    if args.audit_input:
        raw_out=args.audit_input.resolve()
        if raw_out.exists():raise SystemExit('Audit input must be new: '+str(raw_out))
        raw_out.mkdir(parents=True)
        mesh_input=args.reference/'source-sign-audit/math-gap-20261021T0626Z/directional-candidate/tree/complete-coupling/twod/input-exactR'
        for name in ['V.bin','faceOwner.bin','faceNeighbour.bin']:shutil.copy2(mesh_input/name,raw_out/name)
        for source in initial.glob('*.bin'):shutil.copy2(source,raw_out/source.name)
        (raw_out/'manifest.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in raw_out.glob('*.bin')},indent=2)+'\n')
    files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
    (out/'input-manifest.json').write_text(json.dumps({'model':args.model,'integrator':args.integrator,'steps':args.steps,'dt':2e-7,'nodes':24,'maxIterations':60,'template_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':files},indent=2)+'\n')
    print(out)

if __name__=='__main__':main()
