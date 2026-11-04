"""Complete CHMT owner transitions and restart, with controlled internal profiles."""
import argparse,json,re,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'validation/pr31'))
from chmt_particles_native import m,table,inventory,prepare

def main():
 a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);out=a.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
 meta=json.loads((ROOT/'validation/molecular_transport/fixtures.json').read_text())['chmt-mixture'];report=[]
 for model in ('coldWall1D','coldWall2D'):
  for name,state,frozen,threshold in [('frozen_rebound',2,True,100),('liquid_rebound',2,False,100),('frozen_deposit',3,True,0),('disabled_frozen_rebound',2,True,'false')]:
   case=out/(model+'-'+name);m.prepare(ROOT/'validation/molecular_transport/fixtures/chmt-mixture',case,meta)
   m.entry(case/'system/controlDict','endTime',2e-6)
   props=case/'constant/chmtProperties';m.entry(props,'couplingInterval',1e-6);m.entry(props,'gasMaxDt',1e-6)
   boundary=(case/'constant/polyMesh/boundary').read_text();left=re.search(r'\bleft\s*\{([^}]+)',boundary)[1];face=int(re.search(r'startFace\s+(\d+)',left)[1])
   temperature=2000 if frozen else 3000
   cp=lambda t:(153.5+.0031*(t-2327.))/.101961276
   h=1273*temperature if frozen else 1273*2317+(.5*(1273+cp(2327))+1.16e6/20)*20+.5*(cp(2337)+cp(temperature))*(temperature-2337)
   area=1e-7 if frozen else 0
   profiles=('coldNodeSpecificEnthalpy ('+' '.join([str(h)]*8)+'); coldRingSolidMass ('+' '.join(['1e-10' if frozen else '0']*8)+'); coldFrozenArea '+str(area)+';') if model=='coldWall1D' else ('cold2DNodeSpecificEnthalpy ('+' '.join([str(h)]*64)+'); cold2DRingContactAge ('+' '.join(['0']*8)+'); cold2DFrozenArea '+str(area)+';')
   props.write_text(props.read_text()+f"""
enableParticles true;
particle {{volumeMode Coupled; capacity 8; rhoSolid 2500; dragModel none; heatTransferModel none; contactModel {model};
 contactCoeffs {{sommerfeldThreshold {threshold}; interfaceResistance 1;}}}}
particles {{attached {{id "42"; rng "1234567"; cell 0; face {face}; state {state};
 position (0 0.25 0.25); velocity (0 0 0); mass 1e-8; diameter .001; temperature {temperature};
 contactAge .000000999999; contactDuration .000001; contactArea 0;
 thermalHistory {{temperature {temperature}; savedVelocity (1 0 0); contactMaximumArea 1e-6; contactPeakFraction .4;
 specificEnthalpyResidual .125; {profiles}}}
}}}}
""")
   m.run([ROOT/'bin/CHMT','-case',case],case/'run.log')
   output=case/'chmtOutput';last=max(int(p.name.split('-')[1]) for p in output.glob('checkpoint-*'))
   first=inventory(output,0);final=inventory(output,last);q=final['parcels'][0]
   assert int(q['status'])==1
   expected_state=1 if state==3 else 0;assert int(q['state'])==expected_state,(model,name,q)
   expected_velocity=0 if frozen else 1
   assert abs(float(q['ux'])-expected_velocity)<1e-10 and abs(float(q['uy']))<1e-10 and abs(float(q['uz']))<1e-10,(model,name,q)
   if state==2:assert int(q['face'])==-1 and float(q['contact_age'])==0 and float(q['contact_duration'])==0
   restart=out/(case.name+'-restart');shutil.copytree(case,restart,ignore=shutil.ignore_patterns('chmtOutput'))
   shutil.copytree(output/'checkpoint-1',restart/'initial-checkpoint')
   rp=restart/'constant/chmtProperties';rp.write_text(rp.read_text()+'\nrestartDirectory "initial-checkpoint";\n')
   m.run([ROOT/'bin/CHMT','-case',restart],restart/'run-restart.log')
   resumed=inventory(restart/'chmtOutput',last)
   for kind in ('gas','solid','particle'):
    assert abs(final[kind]-resumed[kind])<1e-10*max(1.,abs(final[kind]))
   assert q==resumed['parcels'][0]
   report.append(dict(model=model,case=name,final=q,restart_equal=True,total_energy_change_J=sum(final[k]-first[k] for k in ('gas','solid','particle'))))
 # Exercise actual configuration-to-kernel selection on fresh impacts.
 for threshold in ('0','false'):
  case=prepare(out/('impact-'+threshold),'coldWall1D',True,end_time=1e-5)
  props=case/'constant/chmtProperties';props.write_text(props.read_text().replace('sommerfeldThreshold 100;',f'sommerfeldThreshold {threshold};'))
  m.entry(props,'couplingInterval',1e-6);m.entry(props,'gasMaxDt',1e-6)
  m.run([ROOT/'bin/CHMT','-case',case],case/'run.log')
  output=case/'chmtOutput';last=max(int(p.name.split('-')[1]) for p in output.glob('checkpoint-*'))
  q=inventory(output,last)['parcels'][0]
  assert int(q['state']) in ((1,3) if threshold=='0' else (2,)),(threshold,q)
  report.append(dict(case='fresh_impact',threshold=threshold,final=q))
 (out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
