import argparse,json,shutil,subprocess
from pathlib import Path
from chmt_particles_native import ROOT,m,prepare,table
from chmt_moving_particles_native import PROBE

def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);reports=[]
 source=out/'physical-state.cpp';source.write_text(PROBE);probe=out/'physical-state'
 subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'applications/CHMT'),'-I'+str(ROOT/'common'),str(source),'-o',str(probe)],check=True)
 for model in ('coldWall1D','coldWall2D'):
  case=out/model;prepare(case,model,True,.002,False)
  p=case/'constant/chmtProperties';p.write_text(p.read_text().replace('temperature 2500','temperature 2000'))
  seconds=m.run([ROOT/'applications/CHMT/.build/native/CHMT','-case',case],case/'run.log')
  data=table(case/'chmtOutput/particles-4.csv');q=data[0]
  assert q['status']=='1' and q['state']=='1' and float(q['contact_area'])>0,q
  assert float(q['temperature'])<2000 and float(q['ux'])==float(q['uy'])==float(q['uz'])==0,q
  restart=out/(model+'-restart');shutil.copytree(case,restart,ignore=shutil.ignore_patterns('chmtOutput'))
  shutil.copytree(case/'chmtOutput/checkpoint-3',restart/'initial-checkpoint')
  p=restart/'constant/chmtProperties';p.write_text(p.read_text()+'\nrestartDirectory "initial-checkpoint";\n')
  m.run([ROOT/'applications/CHMT/.build/native/CHMT','-case',restart],restart/'run.log')
  exact=(case/'chmtOutput/checkpoint-4/state.chmt').read_bytes()==(restart/'chmtOutput/checkpoint-4/state.chmt').read_bytes()
  original=json.loads(subprocess.check_output([probe,case/'chmtOutput/checkpoint-4'],text=True))
  resumed=json.loads(subprocess.check_output([probe,restart/'chmtOutput/checkpoint-4'],text=True))
  assert original==resumed,(original,resumed)
  # Cumulative wall-energy counters restart at zero; subtracting adjacent
  # counters can differ by roundoff. All physical state/profile bytes agree.
  ledger_delta=0.
  for left,right in zip(table(case/'chmtOutput/exchange-4.csv'),table(restart/'chmtOutput/exchange-4.csv')):
   for key in left:
    try:ledger_delta=max(ledger_delta,abs(float(left[key])-float(right[key])))
    except ValueError:assert left[key]==right[key]
  assert ledger_delta<1e-12,ledger_delta
  reports.append(dict(model=model,seconds=seconds,particles=data,restart_checkpoint_byte_identical=exact,physical_state=original,maximum_ledger_difference_J=ledger_delta,
    scope='impact to solidifying deposit, continued heat and restart; ordinary/cold caloric datum switch is not a conservation pass'))
 (out/'results.json').write_text(json.dumps(reports,indent=2)+'\n')
if __name__=='__main__':main()
