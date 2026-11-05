"""TRANSPORT-P2-001: run the same full production face operator on host/CUDA.

Only the fixture setup/launch differs. No performance measurements.
"""
from pathlib import Path
import argparse
import json
import math
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / 'validation/tests/gas_transport'

def source(native):
    prefix = (HERE / 'gas_state_probe.cpp').read_text().split('void hashBytes', 1)[0]
    if native:
        adapters = ('#define __host__', '#define __global__', '#define __device__',
                    '#define __forceinline__', '#define asm(', 'struct ThreadIndex ')
        prefix = '#include <cuda_runtime.h>\n' + '\n'.join(
            line for line in prefix.splitlines() if not line.startswith(adapters))
    fields = re.findall(r'    ([\w:]+\*) (\w+)\{\};',
                        (HERE / 'legacy_gas_fixture.hpp').read_text())
    allocations = '\n'.join(f's.{name}=allocate<{ty[:-1]}>(8);' for ty, name in fields)
    body = (Path(__file__).with_name('pr_wall_probe.inc')).read_text()
    body = body.replace('ALLOCATE_FIELDS', allocations)
    body = body.replace('NATIVE_MODE', '1' if native else '0')
    return prefix + '\n' + body

def expected(case, kinetic):
    # Independent scalar reference: known LinearCp mixture, stationary wall.
    # Old constant transport intentionally uses owner composition.
    y = .9 if case in (0, 1, 5) else .1
    cp = 1100 + 500*y
    R = 8.31446261815324*((1-y)/.028+y/.04)
    ownerR = 8.31446261815324*(.9/.028+.1/.04)
    rho = ownerR*700/(R*500)
    pr = .7 if case >= 4 else .7+.2*y
    ratio = pr/.9
    P = 9.24*(ratio**.75-1)*(1+.28*math.exp(-.007*ratio))
    A = math.log(9.8)-math.log(.41)-math.log(ratio)+.41*P
    lo, hi = 1., max(2., A+math.log(A)+2)
    for _ in range(100):
        mid = (lo+hi)/2
        if mid-math.log(mid)>A: hi=mid
        else: lo=mid
    yt = (lo+hi)/2/(.41*ratio)
    us = .09**.25*math.sqrt(kinetic)
    yp = us*.5*rho/1.8e-5
    molecular = 1.8e-5*cp/pr
    viscous = yp < yt
    tp = pr*yp if viscous else .9*(math.log(9.8*yp)/.41+P)
    conductivity = max(molecular, cp*rho*us*.5/tp)
    uc = 0 if viscous else us/.41*math.log(9.8*yt)
    correction = 0 if viscous else .5*rho*us*(pr-.9)*uc*uc/tp
    return conductivity*400+correction, conductivity

def verify(rows, allow_known_defect=False):
    failures = []
    for row in rows:
        case, kinetic, q, energy, mass, ja, jb, status, active, ok, fatal, conductivity, mu, cp, pr = row
        ref, refk = expected(int(case), kinetic)
        if not (status == fatal == mass == ja == jb == 0 and active == ok == 1):
            failures.append({'case':case, 'k':kinetic, 'reason':'status/mass/species'})
        for name, value, target in [('wallHeat',q,ref),('fullFaceEnergy',energy,ref),
                                     ('effectiveConductivity',conductivity,refk)]:
            if not math.isclose(value,target,rel_tol=2e-11,abs_tol=2e-11):
                failures.append({'case':case,'k':kinetic,'quantity':name,
                                 'actual':value,'expected':target})
    if allow_known_defect:
        assert failures and {f['case'] for f in failures} == {0}, failures
    else:
        assert not failures, failures
    return failures

def run(output, native=False, include_root=ROOT, allow_known_defect=False):
    output.mkdir(parents=True, exist_ok=False)
    src = output / ('probe.cu' if native else 'probe.cpp')
    src.write_text(source(native))
    exe = output / 'probe'
    cmd = (['/usr/local/cuda/bin/nvcc','-arch=sm_89','--fmad=false'] if native else ['g++'])
    cmd += ['-std=c++17','-O2','-I'+str(include_root/'common'),
            '-I'+str(include_root/'common/gasNumerics'),'-I'+str(HERE),str(src),'-o',str(exe)]
    build = subprocess.run(cmd, capture_output=True, text=True)
    (output/'build.log').write_text('COMMAND: '+ ' '.join(cmd)+'\n'+build.stdout+build.stderr)
    assert build.returncode == 0, build.stderr[-6000:]
    result = subprocess.run([str(exe)],capture_output=True,text=True)
    (output/'rows.txt').write_text(result.stdout)
    (output/'run.stderr').write_text(result.stderr)
    assert result.returncode == 0, result.stderr
    rows = [[float(x) for x in line.split()] for line in result.stdout.splitlines()]
    assert len(rows) == 12
    failures = verify(rows,allow_known_defect)
    report = {'nativeCUDA':native,'states':len(rows),'failures':failures,
              'knownDefectExpected':allow_known_defect,
              'columns':'case k wallHeat fullFaceEnergy mass speciesA speciesB faceStatus active success fatal effectiveConductivity mu faceCp facePr'.split()}
    (output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
    return rows

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--cuda',action='store_true')
    parser.add_argument('--include-root',type=Path,default=ROOT)
    parser.add_argument('--expect-before-defect',action='store_true')
    args=parser.parse_args()
    run(args.output,args.cuda,args.include_root,args.expect_before_defect)
