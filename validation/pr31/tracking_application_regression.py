"""Compare actual steady particle transport before/after shared tracking geometry.
Uses existing general steady fixtures; does not assert convergence in 200 samples.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
matrix=module('matrix',ROOT/'validation/molecular_transport/native_matrix.py')
steady=module('steady',ROOT/'validation/steady_cpu/of_cases.py')

def particles(path):
    data=path.read_bytes();header,data=data.split(b'\n',1)
    magic,count,chunk=header.split();assert magic==b'UGKP_PARTICLES_SCHEMA1_BIN'
    count=int(count);chunk=int(chunk);fields={};offset=0;read=0
    layout=[(name,'<f8') for name in ('x','y','z','ux','uy','uz','T','theta','d','mass')]+[('cell','<i4'),('status','<i4'),('rng','<u8'),('id','<u8')]
    while read<count:
        n=int(np.frombuffer(data,dtype='<u4',count=1,offset=offset)[0]);offset+=4
        assert 0<n<=min(chunk,count-read)
        for name,dtype in layout:
            value=np.frombuffer(data,dtype=dtype,count=n,offset=offset).copy();offset+=value.nbytes
            fields.setdefault(name,[]).append(value)
        read+=n
    assert offset==len(data) and count>0,'particle transport was not exercised'
    fields={key:np.concatenate(value) for key,value in fields.items()};order=np.argsort(fields['id'])
    return {key:value[order] for key,value in fields.items()}

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--work',type=Path,required=True);a=p.parse_args()
    out=a.work.resolve();out.mkdir(parents=True,exist_ok=False);report={}
    for fixture in ('particles','mixture_particles'):
        cfg=out/(fixture+'.cfg');cfg.write_text((ROOT/f'validation/steady_cpu/cases/{fixture}.cfg').read_text()+'\niterations 200\noutputEvery 50\nmassRate 1\n')
        cases={}
        for label,folder in (('baseline',a.baseline),('candidate',a.candidate)):
            case=out/(fixture+'-'+label);steady.materialize(cfg,case,True)
            matrix.run(['blockMesh','-case',case],case/'mesh.log')
            matrix.run([folder.resolve()/'gasUGKPSteady','-case',case],case/'run.log');cases[label]=case
        left=matrix.times(cases['baseline'])[-1];right=matrix.times(cases['candidate'])[-1]
        errors=matrix.compare_fields(left,right)
        x=particles(left/'gpuResidentStrictParticles.dat');y=particles(right/'gpuResidentStrictParticles.dat');parcel={}
        for key,value in x.items():
            assert value.shape==y[key].shape
            if value.dtype.kind in 'iu':assert np.array_equal(value,y[key]),key
            else:
                error=float(np.max(np.abs(value-y[key])/np.maximum(1,np.maximum(np.abs(value),np.abs(y[key])))))
                assert np.isfinite(error) and error<2e-12,(key,error);parcel[key]=error
        report[fixture]={'sample':left.name,'particles':len(x['id']),'field_errors':errors,'particle_errors':parcel,'scope':'200 physical sampling steps; not steady convergence or a performance benchmark'}
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(fixture,'PASS',flush=True)
if __name__=='__main__':main()
