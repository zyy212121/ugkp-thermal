from pathlib import Path
import sys,shutil,subprocess
r=Path('/workspace/scratch/98b6339ec81b/pr14_fast_fix_20261014');o=Path(__file__).parent
baseline=o/'chem_baseline';shutil.copytree(r/'common',baseline/'common',dirs_exist_ok=True)
for p in (o/'before_chemistry/common').rglob('*'):
 if p.is_file():q=baseline/'common'/p.relative_to(o/'before_chemistry/common');q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
sys.path.insert(0,str(r/'tests/chemistry'))
from test_gas_rates import SOURCE
pre=SOURCE.split('int main(){')[0]
code=pre+r'''
#include "common/chemistry/LocalStiffIntegrator.H"
#include <cstdio>
int main(){for(int strict=0;strict<2;++strict)for(int scenario=0;scenario<9;++scenario){
Model a;ClosedReactorInput<double,2> b;b.speciesMass[0]=.028;b.totalMass=.028;b.volume=.001;b.internalEnergy=mixtureEnergy(b.speciesMass,700.,a.t);
ChemistryControls<double> c;c.relativeTolerance=1e-7;c.absoluteMassFractionTolerance=1e-13;c.absoluteTemperatureTolerance=1e-6;
double dt=.1;if(scenario==1){a.rx[0].highRate.preExponential=1e8;dt=.001;}if(scenario==2){a.rx[0].highRate.preExponential=20;a.coef[5]=-5e5;dt=.05;}if(scenario==3)c.maximumSteps=1;if(scenario==4)b.totalMass*=2;if(scenario==5)dt=0;if(scenario==6)c.maximumNewtonIterations=1;if(scenario==7)c.minimumStep=.1;if(scenario==8)c.maximumRejectedSteps=0;
ClosedReactorResult<double,2> result{};bool evaluated=false;CHEM_AUDIT_DECL
ChemistryStatus s=strict?advanceClosedReactorTrial(b,dt,a.t,a.m,c,result,CHEM_AUDIT_ARG &evaluated):advanceClosedReactor(b,dt,a.t,a.m,c,result,CHEM_AUDIT_ARG &evaluated);
std::printf("%d %d %d %d %d %d %d %d %a %a %a\n",strict,scenario,int(s.code),int(s.thermoCode),s.reaction,s.attemptedSteps,s.rejectedSteps,int(evaluated),result.speciesMass[0],result.speciesMass[1],result.temperature);
}}
'''
for tag,root in [('before',baseline),('after',r)]:
 src=o/('chem_'+tag+'.cpp');src.write_text(code.replace('CHEM_AUDIT_DECL','ChemistryAudit<double,2> audit;' if tag=='before' else '').replace('CHEM_AUDIT_ARG','audit,' if tag=='before' else ''))
 exe=o/('chem_'+tag);cmd=['g++','-std=c++17','-O0','-Wall','-Wextra','-Werror','-I'+str(root),str(src),'-o',str(exe)]
 build=subprocess.run(cmd,capture_output=True,text=True);(o/('chem_'+tag+'.build.log')).write_text(build.stdout+build.stderr);assert build.returncode==0,build.stderr
 run=subprocess.run([str(exe)],capture_output=True,text=True,timeout=60);(o/('chem_'+tag+'.txt')).write_text(run.stdout+run.stderr);assert run.returncode==0,run.stderr
assert (o/'chem_before.txt').read_bytes()==(o/'chem_after.txt').read_bytes(),'chemistry physical/status mismatch'
print('18 direct/trial chemistry scenarios: bitwise physical outputs + actual attempted/rejected controls + statuses match')
