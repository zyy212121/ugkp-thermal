"""Frozen reacting-SST wall: bounded donor and zero matching supply."""
import argparse,json,re
from pathlib import Path
from chmt_particles_native import ROOT,m,table

def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);reports=[]
 meta=json.loads((ROOT/'validation/molecular_transport/fixtures.json').read_text())['chmt-moving']
 for absent in (False,True):
  case=out/('absent-matching' if absent else 'uniform')
  m.prepare(ROOT/'validation/molecular_transport/fixtures/chmt-moving',case,meta)
  p=case/'constant/chmtProperties';s=p.read_text().replace('couplingInterval 1.0000000000000002e-06;','couplingInterval .00001;').replace('gasMaxDt 0.00001;','gasMaxDt .000001;').replace('gasNu (0.028000000000000001 0);','gasNu (-.028 .056);').replace('meshMotion { policy CoupledRecession; }','meshMotion { policy Static; }');p.write_text(s)
  for name,value in [('Y_S0',1e-6),('Y_S1',.999999)]:
   p=case/'0'/name;s=re.sub(r'uniform [^;]+;',f'uniform {value};',p.read_text())
   if absent:
    values=[value]*4+([0.] if name=='Y_S0' else [1.])*12
    s=re.sub(r'internalField uniform [^;]+;','internalField nonuniform List<scalar> 16 ('+' '.join(str(v) for v in values)+');',s)
   p.write_text(s)
  p=case/'constant/gasModelProperties';p.write_text(p.read_text().replace('(2e-5 2e-5)','(100 100)'))
  seconds=m.run([ROOT/'applications/CHMT/.build/native/CHMT','-case',case],case/'run.log')
  gas=table(case/'chmtOutput/gas-1.csv');minimum=min(float(g['mass_S0']) for g in gas)
  assert minimum>=0
  material=sum(float(a['mass_C0'])-float(b['mass_C0']) for a,b in zip(table(case/'chmtOutput/material-0.csv'),table(case/'chmtOutput/material-1.csv')))
  extent=sum(float(p['mass']) for p in table(case/'chmtOutput/exchange-1.csv') if p['kind']=='0')
  assert abs(material-extent)<2e-15,(material,extent)
  if absent:assert extent==0
  else:assert extent>0
  reports.append(dict(absent_matching=absent,seconds=seconds,minimum_reactant_kg=minimum,reaction_extent_kg=extent,material_loss_kg=material))
 (out/'results.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports))
if __name__=='__main__':main()
