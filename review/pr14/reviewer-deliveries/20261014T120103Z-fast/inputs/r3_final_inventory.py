from r3_runner import *
from collections import Counter
oldr=json.loads((OLD/'results.json').read_text());oldc=json.loads((OLD/'coverage.json').read_text());oldf=json.loads((OLD/'findings.json').read_text())
print('GAPS',json.dumps(oldc['remaining_gaps'],ensure_ascii=False,indent=2));print('UNITS',[{k:q.get(k) for k in ['id','verdict','topics','scope','criteria']} for q in oldr['acceptance_units']]);print('FINDS',[{k:q.get(k) for k in ['id','priority','status','title','topics']} for q in oldf['findings']]);print('META',[p.name for p in (D/'metadata').iterdir()])
tests=json.loads((D/'results.json').read_text())['tests'];print('COUNTS',len(tests),Counter(q['verdict'] for q in tests));print('INCONCLUSIVE',[(q['id'],q['kind'],q['exit_code'],q['observations']) for q in tests if q['verdict']=='INCONCLUSIVE']);print('SOURCEschema',(D/'SOURCE.json').read_text()[:1900]);print('ENVschema',(D/'environment.json').read_text()[:1000] if (D/'environment.json').exists() else 'missing')
unsupported=D/'metadata/native-build-info-Ns2.txt'
if unsupported.exists():shutil.copy2(unsupported,D/'metadata/native-build-info-Ns2-unsupported-option-original.txt')
for ns in [2,10]:
 b=ROOT/f'work/install/ns{ns}/bin/CHMT';t=run('NATIVE-BUILD-IDENTITY-Ns'+str(ns),cmd(b,'-build-info'),[1,26],30,'Actualnative executablebuildidentity OpenFoam single-dashoption, noGPU; firstdouble-dashsetupfailure preserved.',inputs=[ROOT/f'work/build/CHMT{ns}/generated/CHMTBuildIdentity.H'],binary=b,kind='IDENTITY');raw=(D/t['log']).read_text();assertion(t,t['exit_code']==0 and SOURCE in raw,'Actualnativeproducer buildinfo/sourcepin contains formal9d='+str(SOURCE in raw));print('BUILDIDENTITY',ns,raw[-450:])
q=subprocess.run(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu','--format=csv'],capture_output=True,text=True,timeout=10);(D/'metadata/final-gpu-resource.txt').write_text(q.stdout+q.stderr)
