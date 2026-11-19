import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from cht_physical_clock import replace_entry


def run(binary,fixture,case):
    case.mkdir()
    for name in ('0','constant','system'):
        shutil.copytree(fixture/name,case/name)
    dt=2**-24
    control=case/'system/controlDict'
    for key,value in dict(endTime=32*dt,deltaT=dt,writeInterval=32,adjustTimeStep='no',logInterval=100).items():
        replace_entry(control,key,value)
    with control.open('a') as out:
        out.write("""
functions
{
    samples
    {
        type probes;
        libs ("libsampling.so");
        executeControl timeStep;
        executeInterval 4;
        writeControl timeStep;
        writeInterval 8;
        fields (p);
        probeLocations ((0.25 0.25 0.25));
    }
}
""")
    log=case/'run.log'
    with log.open('x') as out:
        subprocess.run([binary,'-case',case],stdout=out,stderr=subprocess.STDOUT,check=True,
                       env=dict(os.environ,UGKP_ADVANCE_STATS='1'))
    data=next((case/'postProcessing/samples').rglob('p'))
    records=[tuple(map(float,line.split())) for line in data.read_text().splitlines() if line and not line.startswith('#')]
    match=re.search(r'ADVANCE_BATCHES count=(\d+) steps=(\d+) single=(\d+) max=(\d+)',log.read_text())
    assert match and int(match[2])==32
    return records,list(map(int,match.groups()))


def main():
    parser=argparse.ArgumentParser()
    for key in ('before','candidate','fixture','output'):
        parser.add_argument(key,type=lambda s:Path(s).resolve())
    args=parser.parse_args()
    args.output.mkdir(parents=True)
    previous,a=run(args.before,args.fixture,args.output/'before')
    current,b=run(args.candidate,args.fixture,args.output/'candidate')
    assert current==previous and len(current)>=4 and a[-1]==1 and b[-1]==4,(previous,current,a,b)
    (args.output/'results.json').write_text(json.dumps(dict(samples=current,before=a,candidate=b))+'\n')


if __name__=='__main__':
    main()
