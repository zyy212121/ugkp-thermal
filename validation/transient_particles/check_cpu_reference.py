#!/usr/bin/env python3
"""Compare native physical-time fields with the existing serial CPU operator reference."""
from pathlib import Path
import argparse,csv,importlib.util,json,subprocess
sp=importlib.util.spec_from_file_location('native',Path(__file__).with_name('check_native.py'));m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
comp=m.module('compare',m.ROOT/'validation/steady_cpu/compare_application.py')
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for k in ['binary','case','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();o=a.output.resolve();o.mkdir(parents=True,exist_ok=False)
 cfg=o/'input.cfg';cfg.write_text((m.ROOT/'validation/steady_cpu/cases/mixture_particles.cfg').read_text().replace('iterations 20000','iterations 200').replace('outputEvery 1000','outputEvery 20'))
 cmd=[str(a.binary.resolve()),str(cfg),str(o/'cpu.csv')]
 with (o/'run.log').open('w') as log:q=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
 assert q.returncode==0
 rows=[]
 for d in m.dirs(a.case.resolve())[1:]:
  iteration=round(float(d.name)/2e-5);data={name:m.fields.internal(d/name,32) for name,_ in m.fields.MAPPING.values() if (d/name).exists()}
  ya=m.fields.internal(d/'rhoY_A',32);yb=m.fields.internal(d/'rhoY_B',32);parcels=m.parcel(d/'gpuResidentStrictParticles.dat')
  for c in range(32):
   row={'iteration':iteration,'cell':c}
   row.update({column:data[name][c][component] for column,(name,component) in m.fields.MAPPING.items() if name in data})
   row.update(rhoY0=ya[c][0],rhoY1=yb[c][0],rhoSolid=1000*data['epsilonS'][c][0],rhoHp=sum(x[9]*1000*x[6] for x in parcels if x[10]==c)/.0625)
   rows.append(row)
 with (o/'native.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
 gas=comp.comparison(o/'cpu.csv',o/'native.csv',['rho','rhoUx','rhoUy','rhoUz','rhoE','Ux','Uy','Uz','p','T','rhoY0','rhoY1'],1e-8,1e-9,start=20,end=200)
 particles=comp.comparison(o/'cpu.csv.particles.csv',o/'native.csv',list(comp.PARTICLES),1e-8,1e-9,particles=True,start=20,end=200)
 report={'command':cmd,'cpu_binary_sha256':m.sha(a.binary),'gas':gas,'particles':particles,'pass':gas['pass'] and particles['pass'],'scope':'Serial execution of shared production gas/parcel headers, separate from native CUDA/OpenFOAM driver; not an independent discretization. rhoHp reconstructed from actual weighted parcel restart records; no fabricated exchange.'}
 (o/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(report['pass']);assert report['pass']
if __name__=='__main__':main()