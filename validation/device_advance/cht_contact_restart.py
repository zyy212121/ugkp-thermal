import argparse
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

import numpy as np

from cht_physical_clock import replace_entry
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tool/restart'))
from thermal_particle_restart_current import field_specs, iter_chunks


def run(binary,fixture,case,dt,steps,renumber=False,restart=None,truncated=False):
    case.mkdir()
    for folder in ('0','system','constant'):
        shutil.copytree(fixture/folder,case/folder)
    if restart:
        shutil.copytree(restart,case/restart.name)
    else:
        f=case/'0/gpuResidentStrictParticles.dat'
        marker,rows=next(iter_chunks(f))
        rows['pStuck'][0]=1
        rows['pStuckFaceId'][0]=12
        rows['pContactDuration'][0]=1
        rows['pTheta'][:]=(.125,.0625)
        header,body=f.read_bytes().split(b'\n',1)
        if renumber:header=re.sub(rb'SCHEMA[0-9]+',b'SCHEMA108',header)
        with f.open('wb') as output:
            output.write(header+b'\n'+struct.pack('<I',2))
            for name,dtype,width in field_specs(marker):
                output.write(np.asarray(rows[name],dtype=dtype).tobytes())
            if truncated:output.truncate(output.tell()-3)
    props=case/'constant/particleProperties'
    replace_entry(props,'gpuResidentStuckWallPatches','(left)')
    with props.open('a') as config:config.write('\ngpuResidentStuckModel { wallInteractionModel reboundContact; }\n')
    replace_entry(case/'constant/radiationProperties','enabled','false')
    solid=case/'constant/solidRegionProperties'
    solid.write_text(solid.read_text().replace('radiationEnabled true','radiationEnabled false').replace('kappa {type constant;value 10;}','kappa {type constant;value 1000;}'))
    scheme=case/'constant/scheme'
    for key in ('couplingInterval','solidContactCouplingInterval'):replace_entry(scheme,key,dt)
    ctl=case/'system/controlDict'
    for key,value in dict(startTime=restart.name if restart else 0,endTime=steps*dt,deltaT=dt,
                          adjustTimeStep='no',writeInterval=1,writeControl='timeStep').items():
        replace_entry(ctl,key,value)
    with (case/'run.log').open('w') as log:
        code=subprocess.run([binary,'-case',case],stdout=log,stderr=subprocess.STDOUT).returncode
    if truncated:
        text=(case/'run.log').read_text()
        assert code and 'code=8' in text and 'truncated restart field' in text,text[-1200:]
        return None,None
    assert code==0,(case/'run.log').read_text()[-1800:]
    output=max(case.glob('*/gpuResidentStrictParticles.dat'),key=lambda f:float(f.parent.name))
    _,rows=next(iter_chunks(output))
    index=np.flatnonzero(rows['orig_id']==42).item()
    mobile=np.flatnonzero(rows['orig_id']==43).item()
    assert rows['pStuck'][index] and not rows['pStuck'][mobile]
    return [float(rows['pTheta'][index]),float(rows['pTheta'][mobile])],output.parent


def main():
    parser=argparse.ArgumentParser()
    for arg in ('candidate','before','fixture','output'):
        parser.add_argument(arg,type=lambda x:Path(x).resolve())
    args=parser.parse_args()
    args.output.mkdir(parents=True)
    dt=2**-24
    old,_=run(args.before,args.fixture,args.output/'old',dt,1)
    first,snapshot=run(args.candidate,args.fixture,args.output/'first',dt,1,True)
    resumed,_=run(args.candidate,args.fixture,args.output/'resumed',dt,2,restart=snapshot)
    run(args.candidate,args.fixture,args.output/'truncated',dt,1,True,truncated=True)
    assert old[0]==0 and first[0]>=.125 and resumed[0]>=first[0],(old,first,resumed)
    assert old[1]==first[1] and first[1]>0,(old,first)
    (args.output/'results.json').write_text(json.dumps(dict(before=old,first=first,resumed=resumed))+'\n')


if __name__=='__main__':
    main()
