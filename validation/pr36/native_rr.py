"""Actual CHT source initialization, RR injection and mass-weighted CDF."""
import argparse,sys,json,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'validation/molecular_transport'),str(ROOT/'tool/restart')]
import native_matrix as m
import thermal_particle_restart_current as wire

def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);a.add_argument('--solver',choices=['CHT','FSH'],default='CHT');args=a.parse_args();out=args.output.resolve()
 meta=json.loads((ROOT/'validation/molecular_transport/fixtures.json').read_text())['cht-contact']
 m.prepare(ROOT/'validation/molecular_transport/fixtures/cht-contact',out,meta)
 for key,value in [('gpuResidentPureGasOnly','false'),('gpuResidentParticleCapacity',8192)]:m.entry(out/'constant/schedulingProperties',key,value)
 for key,value in [('parcelMass',1e-9),('dS',1e-4)]:m.entry(out/'constant/particleProperties',key,value)
 path=out/'constant/particleProperties';path.write_text(path.read_text()+'\ndMin 3e-5; dMax 4e-4; dn 2.5;\nparticleInjectionVelocity (1 0 0);\ngpuResidentStuckModel { sommerfeldThreshold false; }\n')
 path=out/'0/epsilonS';path.write_text(path.read_text().replace('left{type zeroGradient;}','left{type fixedValue;value uniform .001;}'))
 for key,value in [('endTime',2e-6),('deltaT',1e-6),('maxDeltaT',1e-6),('writeInterval',2e-6)]:m.entry(out/'system/controlDict',key,value)
 m.run([ROOT/'bin'/args.solver,'-case',out],out/'run.log')
 final=m.times(out)[-1];chunks=[d for _,d in wire.iter_chunks(final/'gpuResidentStrictParticles.dat')]
 diameter=np.concatenate([d['pd'] for d in chunks]);mass=sum(float(d['pm'].sum()) for d in chunks)
 residual=(final/'gpuResidentStrictSourceResidual.dat').read_text().split();remaining=sum(map(float,residual[3::2]))
 assert abs(mass+remaining-5.6e-6)<1e-15
 assert len(diameter)>5000 and np.all((diameter>=3e-5)&(diameter<=4e-4))
 q=np.sort((diameter/1e-4)**2.5);cdf=-np.expm1(-(q-.3**2.5))/(-np.expm1(-(4**2.5-.3**2.5)))
 ks=max(float(np.max(np.arange(1,len(q)+1)/len(q)-cdf)),float(np.max(cdf-np.arange(len(q))/len(q))))
 assert ks<.035,ks
 assert np.count_nonzero((diameter==3e-5)|(diameter==4e-4))==0
 result=dict(solver=args.solver,count=len(diameter),mass=mass,residual=remaining,ks_distance=ks,minimum=float(diameter.min()),maximum=float(diameter.max()))
 (out/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
