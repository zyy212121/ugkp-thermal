"""Native output scheduling parity, including a zero-step final publication."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'validation/molecular_transport'))
import native_matrix as m

def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 meta=json.loads((ROOT/'validation/molecular_transport/fixtures.json').read_text())['chmt-mixture']
 def prepare(name,end,controls=''):
  case=out/name;m.prepare(ROOT/'validation/molecular_transport/fixtures/chmt-mixture',case,meta)
  m.entry(case/'system/controlDict','endTime',end)
  p=case/'constant/chmtProperties';m.entry(p,'couplingInterval',.000001);p.write_text(p.read_text()+'\n'+controls+'\n')
  m.run([ROOT/'bin/CHMT','-case',case],case/'run.log');return case/'chmtOutput'
 baseline=prepare('every-step',.000004)
 sparse=prepare('sparse',.000004,'outputControls {diagnosticInterval 3;fieldInterval 2;checkpointInterval 4;writeInitial false;writeFinal true;}')
 steps=sorted(int(p.name.split('-')[1]) for p in baseline.glob('checkpoint-*'));last=steps[-1];assert last>1
 for prefix,interval,suffix in [('accepted-',3,'.csv'),('gas-',2,'.csv'),('checkpoint-',4,'')]:
  actual={int(p.name[len(prefix):].removesuffix(suffix)) for p in sparse.glob(prefix+'*')}
  expected={i for i in steps if i>0 and i%interval==0}|{last}
  assert actual==expected,(prefix,actual,expected)
 assert (baseline/f'checkpoint-{last}/state.chmt').read_bytes()==(sparse/f'checkpoint-{last}/state.chmt').read_bytes()
 zero=prepare('zero-step',0,'outputControls {diagnosticInterval 0;fieldInterval 0;checkpointInterval 0;writeInitial false;writeFinal true;}')
 assert (zero/'gas-0.csv').is_file() and (zero/'accepted-0.csv').is_file() and (zero/'checkpoint-0/state.chmt').is_file()
 result={'status':'PASS','accepted_steps':last,'final_checkpoint_bitwise_equal':True,'zero_step_final_output':True}
 (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
