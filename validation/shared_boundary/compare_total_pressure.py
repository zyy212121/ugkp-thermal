"""Compare the production device evaluator against native OF10 boundary fields.
Usage (source OpenFOAM first): python3 compare_total_pressure.py NEW_OUTPUT_DIR
"""
from pathlib import Path
import json, os, re, shutil, subprocess, sys
R=Path(__file__).resolve().parents[2]
O=Path(sys.argv[1]).resolve();O.mkdir(parents=True,exist_ok=False)
B=O/'build';B.mkdir()
def run(args,log,env=None):
    with (O/log).open('w') as f:
        subprocess.run([str(x) for x in args],stdout=f,stderr=subprocess.STDOUT,check=True,env=env)
run(['/usr/local/cuda/bin/nvcc','-std=c++17','-O2','--fmad=false','-arch=sm_89','-I'+str(R/'common'),R/'validation/shared_boundary/total_pressure.cu','-o',B/'total-pressure'],'device-build.log')
run(['/usr/local/cuda/bin/compute-sanitizer','--tool','memcheck','--leak-check','full','--error-exitcode','9',B/'total-pressure'],'device.log')
# Build the native reference within the output directory, not an external application bin.
shutil.copytree(R/'validation/shared_boundary/native_reference',B/'reference',ignore=shutil.ignore_patterns('linux*'))
run(['wmake',B/'reference'],'reference-build.log',dict(os.environ,FOAM_USER_APPBIN=str(B)))
rows={int(x[0]):list(map(float,x[1:])) for x in re.findall(r'^DEVICE (.*)$',(O/'device.log').read_text(),re.M) for x in [x.split()]}
assert len(rows)==12
results=[]
for i in range(12):
    mode,variant=divmod(i,4);c=O/('case-'+str(i));c.mkdir()
    for part in ['0','constant','system']:shutil.copytree(R/'validation/shared_boundary/fixtures/gas-total'/part,c/part)
    p=c/'constant/polyMesh/points'
    def rotate(m):
        x,y,z=map(float,m.groups());return f'({-.6*x+.8*y:.17g} {-.8*x-.6*y:.17g} {z:.17g})'
    p.write_text(re.sub(r'\(([-+\deE.]+) ([-+\deE.]+) ([-+\deE.]+)\)',rotate,p.read_text()))
    def patch(field,body):
        p=c/'0'/field;s,n=re.subn(r'\binlet\s*\{[^}]*\}','inlet { '+body+' }',p.read_text());assert n==1;p.write_text(s)
    tangent=['','tangentialVelocity constant (-80 60 0);','tangentialVelocity constant (-68 76 0);','tangentialVelocity table ((0 (-40 30 0)) (1 (-120 90 0)));'][variant]
    patch('U','type pressureInletOutletVelocity; '+tangent+' value uniform (-60 -80 0);')
    p=c/'0/U';p.write_text(re.sub(r'internalField\s+uniform\s*\([^)]*\);','internalField uniform (-60 -80 0);',p.read_text()))
    patch('p','type totalPressure; p0 uniform 100000; psi '+('none' if mode==2 else 'thermo:psi')+'; gamma '+('1' if mode==1 else '1.4')+'; value uniform 100000;')
    patch('T','type totalTemperature; T0 uniform 300; gamma 1.4; value uniform 300;')
    run([B/'boundaryReference','-case',c,'-referenceTime','.5'],f'native-{i}.log')
    native=[list(map(float,x.split()[2:])) for x in re.findall(r'^REFERENCE (.*)$',(O/f'native-{i}.log').read_text(),re.M)]
    assert native
    errors=[max(abs(a-b) for a,b in zip(row,rows[i])) for row in native]
    assert max(errors)<1e-8,(i,native,rows[i],errors)
    results.append(dict(index=i,mode=mode,variant=variant,device=rows[i],native=native,max_abs_error=max(errors)))
(O/'results.json').write_text(json.dumps(results,indent=2)+'\n')
print('native_total_pressure_comparisons',len(results),'maximum_error',max(x['max_abs_error'] for x in results))
