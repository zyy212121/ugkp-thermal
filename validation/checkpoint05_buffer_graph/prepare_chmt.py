#!/usr/bin/env python3
"""Prepare only: no solver, mesh generator, profiler or fixture is executed."""
import argparse, hashlib, importlib.util, json, re, shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def normalized(path):
    s=re.sub(r'/\*.*?\*/|//[^\n]*','',path.read_text(),flags=re.S)
    return re.sub(r'\s+','',s)
def copy_mesh(case,donor):
    evidence=[]
    for dictionary,mesh in [('system/blockMeshDict','constant/polyMesh'),('system/blockMeshSolidDict','constant/solid/polyMesh')]:
        assert normalized(case/dictionary)==normalized(donor/dictionary),('mesh dictionary mismatch',dictionary,donor)
        origin=donor/mesh;assert origin.is_dir(),origin
        target=case/mesh;assert not target.exists(),target
        shutil.copytree(origin,target)
        evidence.append({'dictionary':dictionary,'donor':str(origin),'dictionaryNormalizedSha256':hashlib.sha256(normalized(case/dictionary).encode()).hexdigest(),
                         'files':{str(x.relative_to(target)):digest(x) for x in target.rglob('*') if x.is_file()}})
    return evidence

def main():
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True)
    a.add_argument('--static-mesh-donor',type=Path,required=True);a.add_argument('--moving-mesh-donor',type=Path,required=True)
    for v in ['baseline32','baseline64','candidate32','candidate64']:a.add_argument('--'+v,type=Path,required=True)
    a.add_argument('--nsys',type=Path,default=Path('/usr/local/bin/nsys'));args=a.parse_args()
    assert not args.output.exists(),'new output directory required; never overwrite a prepared batch'
    runner=load('cp5_molecular_inputs',ROOT/'validation/molecular_transport/native_matrix.py')
    args.output.mkdir(parents=True)
    nodes=[]
    for kind,fixture,windows,end,interval,donor in [
        ('static','chmt-mixture',3,.0015,.0005,args.static_mesh_donor),
        ('moving','chmt-moving-short',8,5.960464477539063e-8,7.450580596923828e-9,args.moving_mesh_donor)]:
      for variant in ['baseline32','candidate32','baseline64','candidate64']:
        bits=int(variant[-2:]);name=kind+'-'+variant;case=args.output/name
        runner.copy_fixture(ROOT/'validation/molecular_transport/fixtures'/fixture,case)
        runner.entry(case/'system/controlDict','endTime',format(end,'.17g'))
        runner.entry(case/'constant/scheme','couplingInterval',format(interval,'.17g'))
        if kind=='static':
            runner.entry(case/'system/controlDict','deltaT','1e-5')
            runner.entry(case/'constant/scheme','maxDeltaT','1e-5')
        (case/'constant/precisionProperties').write_text('FoamFile {version 2.0;format ascii;class dictionary;object precisionProperties;}\nprecision '+str(bits)+';\n')
        meshes=copy_mesh(case,donor.resolve())
        # Reproduce the existing fixture's thin-3D patch preparation only.
        if kind=='moving':
            for patch in [case/'constant/polyMesh/boundary',case/'constant/solid/polyMesh/boundary']:
                patch.write_text(re.sub(r'\btype\s+empty\s*;','type patch;',patch.read_text()))
        binary=getattr(args,variant).resolve();assert binary.is_file(),binary
        output=args.output/(name+'-profile')
        argv=[str(args.nsys.resolve()),'profile','--trace=cuda','--cuda-graph-trace=node','--sample=none','--cpuctxsw=none',
              '--force-overwrite=false','--output='+str(output),str(binary),'-case',str(case)]
        nodes.append({'id':name,'variant':variant,'precision':bits,'fixture':fixture,'case':str(case),'expectedWindows':windows,'endTime':end,'couplingInterval':interval,
          'argv':argv,'envOverrides':{'OMP_NUM_THREADS':'4','CHMT_CUDA_STACK_BYTES':'65536'},
          'environmentRequirement':'Run in the same sourced OpenFOAM10/CUDA environment for all eight nodes; environment captured by parent before freeze.',
          'binarySha256':digest(binary),'inputSha256':{str(x.relative_to(case)):digest(x) for x in case.rglob('*') if x.is_file()},'meshEvidence':meshes,
          'log':str(args.output/(name+'.log')),'profilePrefix':str(output),
          'statsArgv':[str(args.nsys.resolve()),'stats','--report=cuda_api_sum','--format=csv','--output='+str(output)+'-api',str(output)+'.nsys-rep']})
    (args.output/'manifest.json').write_text(json.dumps({'status':'PREPARED_NOT_RUN','baselineCommit':'3e93bcbe8c25e65765dbd1b9e154a73099ed6a26','nodes':nodes},indent=2))
if __name__=='__main__':main()
