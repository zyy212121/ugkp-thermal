"""Native resident SST/ODE profile publication from the full gas application."""
import argparse,csv,json,math,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'validation/molecular_transport'));import native_matrix as m

def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);args=a.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 meta=json.loads((ROOT/'validation/molecular_transport/fixtures.json').read_text());report=[]
 for name in ['gas-wall','gas-ode']:
  case=out/name;m.prepare(ROOT/'validation/molecular_transport/fixtures'/name,case,meta[name])
  p=case/'constant/fluidProperties';p.write_text(p.read_text().replace('boundaryLayer { model','boundaryLayer { writeProfiles true; model'))
  if name=='gas-ode':(case/'constant/gasWallProperties').write_text('FoamFile {version 2.0;format ascii;class dictionary;object gasWallProperties;}\npatches {wall {speciesFlux (0 0.001);}}\n')
  seconds=m.run([ROOT/'bin/gasUGKP','-case',case],case/'run.log')
  assert 'equilibriumStressClosure: localStress' in (case/'run.log').read_text()
  snapshots=m.times(case);rows=0
  for time in snapshots:
   path=time/'uniform/gasBoundaryLayerProfiles.csv';assert path.is_file(),path
   table=list(csv.DictReader(line for line in path.read_text().splitlines() if line and not line.startswith('#')));assert table
   for row in table:
    for value in row.values():assert math.isfinite(float(value)),(path,row)
   rows+=len(table)
   assert (time/'uniform/gasBoundaryLayerSstProfiles.csv').exists()==(name=='gas-wall')
  report.append(dict(case=name,seconds=seconds,writes=len(snapshots),finite_profile_rows=rows,blowing=name=='gas-ode'))
 (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
