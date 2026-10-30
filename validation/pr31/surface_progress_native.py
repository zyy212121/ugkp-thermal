"""Resident progress after transport; starvation must not stop the gas step."""
import argparse,json,re
from pathlib import Path
from surface_donor_native import prepare
from chmt_particles_native import ROOT,m,table

def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True)
 a.add_argument('--solver',type=Path,default=ROOT/'applications/CHMT/.build/native/CHMT')
 args=a.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True);reports=[]
 for moving in (False,True):
  case=out/('outflow-moving'+str(int(moving)));prepare(case,2,moving)
  p=case/'constant/chmtProperties';p.write_text(p.read_text()+'\nboundaries {left {kind Interface;} right {kind Outlet;} sides {kind Slip;}}\n')
  # Resolve the tiny physical recession rather than accepting the default
  # geometry snap tolerance; the production tolerance policy is unchanged.
  if moving:p.write_text(p.read_text()+'\ntolerances {absoluteGeometry 1e-16;relativeGeometry 1e-16;}\n')
  p=case/'0/U';p.write_text(p.read_text().replace('internalField uniform (0 0 0);','internalField uniform (100 0 0);'))
  seconds=m.run([args.solver,'-case',case],case/'run.log')
  g0=table(case/'chmtOutput/gas-0.csv')[0];g1=table(case/'chmtOutput/gas-1.csv')[0]
  material=sum(float(a['mass_C0'])-float(b['mass_C0']) for a,b in zip(table(case/'chmtOutput/material-0.csv'),table(case/'chmtOutput/material-1.csv')))
  packets=table(case/'chmtOutput/exchange-1.csv');reaction=sum(float(p['mass']) for p in packets if p['kind']=='0')
  assert float(g1['mass_S0'])>=0 and float(g1['mass_S1'])>=0
  assert abs(material-reaction)<5e-13,(material,reaction)
  assert 0<reaction<float(g0['mass_S0']),reaction
  volume=sum(float(a['volume'])-float(b['volume']) for a,b in zip(table(case/'chmtOutput/material-0.csv'),table(case/'chmtOutput/material-1.csv')))
  if moving:assert abs(volume-material/1000)<1e-14,(volume,material)
  reports.append(dict(moving=moving,seconds=seconds,reactant_remaining=float(g1['mass_S0']),reaction_extent_kg=reaction,material_consumed_kg=material,solid_volume_change=volume))
 (out/'results.json').write_text(json.dumps(reports,indent=2)+'\n');print(json.dumps(reports))
if __name__=='__main__':main()
