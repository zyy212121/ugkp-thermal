"""A material endpoint correction must also move attached particle state."""
import argparse,json,re,subprocess
from pathlib import Path
from surface_donor_native import prepare,check
from chmt_particles_native import ROOT,m
from chmt_moving_particles_native import PROBE

def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True)
 a.add_argument('--solver',type=Path,default=ROOT/'applications/CHMT/.build/native/CHMT');a.add_argument('--observe-before',action='store_true')
 args=a.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
 case=out/'case';prepare(case,2,True)
 boundary=(case/'constant/polyMesh/boundary').read_text()
 patch=re.search(r'\bleft\s*\{([^}]+)',boundary)[1]
 face=int(re.search(r'startFace\s+(\d+)',patch)[1])
 # Tight geometry accuracy makes the material owner retain its actual
 # reaction-based endpoint instead of snapping to the predictor trajectory.
 d=case/'constant/chmtProperties';d.write_text(d.read_text()+'''\ntolerances {absoluteGeometry 1e-16;relativeGeometry 1e-16;}
enableParticles true;
particle {capacity 8;rhoSolid 2500;dragModel none;heatTransferModel none;contactModel reboundContact;
 contactCoeffs {heatTransfer false;}}
particles {attached {id "42";rng "1234567";cell 0;face 4;state 2;
 position (0 .5 .5);velocity (0 0 0);mass 1e-12;diameter .001;temperature 2500;
 contactAge .0001;contactDuration .01;contactArea 0;
 thermalHistory {temperature 2500;savedVelocity (1 0 0);contactMaximumArea 1e-6;contactPeakFraction .4;specificEnthalpyResidual .125;}
}}
'''.replace('face 4;',f'face {face};'))
 m.run([args.solver,'-case',case],case/'run.log')
 source=out/'probe.cpp';source.write_text(PROBE);exe=out/'probe'
 subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'applications/CHMT'),'-I'+str(ROOT/'common'),str(source),'-o',str(exe)],check=True)
 state=json.loads(subprocess.check_output([exe,case/'chmtOutput/checkpoint-1'],text=True));budget=check(case)
 result=dict(state=state,budget=budget)
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
 if not args.observe_before:
  assert abs(state['particles'][0]['distance'])<3e-16,result
  assert abs(budget['solid_volume_change']-budget['material_consumed']/1000)<3e-15,result
if __name__=='__main__':main()
