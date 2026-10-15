#!/usr/bin/env python3
"""Instrument exact production snapshots only; never modify the work repository."""
import argparse, hashlib, json, re, shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('destination');a=p.parse_args()
src=Path(a.source).resolve();dst=Path(a.destination).resolve()
if dst.exists():shutil.rmtree(dst)
shutil.copytree(src/'common',dst/'common')
manifest=[]
def instrument(path,name,entry,wrap=False,occurrence=0):
 f=dst/path;s=f.read_text();matches=list(re.finditer(r'\b'+re.escape(name)+r'\s*\([^;{}]*?\)\s*(?:const\s*)?noexcept\s*\{',s,re.S))
 if len(matches)<=occurrence:raise RuntimeError(f'{path}: missing function {name}; found {len(matches)}')
 m=matches[occurrence];start=m.end()-1;depth=1;i=start+1
 while depth:
  depth += (s[i]=='{')-(s[i]=='}');i+=1
 body=s[start+1:i-1]
 if wrap:body=re.sub(r'\breturn\s+([^;]+);',r'return independentProbe.finish(\1);',body)
 s=s[:start+1]+'\n'+entry+'\n'+body+s[i-1:];f.write_text(s)
 manifest.append({'path':path,'function':name,'entry':entry,'wrap_return':wrap,'original_sha256':hashlib.sha256((src/path).read_bytes()).hexdigest(),'instrumented_sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
new_api='const ReactorContext<R,N,TableReal>& context,Time interval' in (src/'common/chemistry/LocalStiffIntegrator.H').read_text()
step_entry='independent::StepScope<decltype(context.base),decltype(context.thermo),decltype(result)> independentProbe(context.base, interval, context.thermo, result, initialTemperature);' if new_api else 'independent::StepScope<decltype(base),decltype(t),decltype(result)> independentProbe(base, interval, t, result, initialTemperature);'
stage_entry='independent::StageScope<decltype(context),R> independentProbe(context,rhs,(long double)h*coefficient,z,temperature);' if new_api else 'independent::StageScope<decltype(context),R> independentProbe(context,rhs,hgamma,z,temperature);'
instrument('common/chemistry/LocalStiffIntegrator.H','sdirkStep',step_entry,True)
instrument('common/chemistry/LocalStiffIntegrator.H','implicitStage',stage_entry,True)
instrument('common/chemistry/ConservativeReactor.H','prepare','++independent::counters.prepare;')
instrument('common/chemistry/ConservativeReactor.H','state','independent::StateScope<R> independentProbe(z,mass,temperature,mechanism.independentRank);',True)
instrument('common/chemistry/MechanismValidation.H','orthogonalBasis','++independent::counters.basis;')
instrument('common/gasTransport/MixtureThermo.H','invertMixtureEnergy','++independent::counters.inversion;')
instrument('common/chemistry/GasRates.H','evaluateGasRates','independent::RateScope<R> independentProbe(T,c,production,N);',True)
instrument('common/chemistry/GasJacobian.H','evaluateGasRateDerivatives','++independent::counters.derivatives;')
f=dst/'common/chemistry/LocalStiffIntegrator.H';s=f.read_text()
needle='remaining.consume(completed);' if new_api else 'covered+=h;'
replacement=('independent::recordConsumption(completed,accepted);'+needle) if new_api else ('independent::recordConsumption(h,accepted);'+needle)
assert s.count(needle)==1, 'missing exact duration-consumption observation seam'
s=s.replace(needle,replacement);f.write_text(s)
manifest.append({'path':'common/chemistry/LocalStiffIntegrator.H','observation':'log actually selected accepted state and consumed duration immediately before existing accounting operation','original_statement':needle,'replacement':replacement})
(dst/'instrumentation.json').write_text(json.dumps(manifest,indent=2)+'\n')
