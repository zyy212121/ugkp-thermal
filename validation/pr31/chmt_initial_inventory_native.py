"""CHMT initial output and restart, no time advance and no timing assertions."""
import argparse,json,re,shutil,subprocess
from pathlib import Path
from chmt_particles_native import m,ROOT,table
from chmt_moving_particles_native import PROBE

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    probe=out/'inventory.cpp'
    head=PROBE[:PROBE.index(' std::cout<<std::setprecision(17)')].replace('if(argc!=2)return 2;', 'if(argc<2||argc>3)return 2;')
    probe.write_text(head+r"""
 if(argc==3){
  // Reproduce a legacy initial checkpoint with the omitted derived diagnostic.
  for(auto& particle:state.particles)particle.contactMechanicalEnergy=0;
  if(!chmt::writeCheckpoint(argv[1],model,state,error)){std::cerr<<error;return 5;}
  return 0;
 }
 std::cout<<std::setprecision(17)<<"{\"time\":"<<state.time<<",\"particles\":[";
 for(size_t i=0;i<state.particles.size();++i){const auto& p=state.particles[i];if(i)std::cout<<',';
  std::cout<<"{\"state\":"<<p.state<<",\"mechanical_J\":"<<p.contactMechanicalEnergy
   <<",\"thermal_J\":"<<p.energy<<",\"saved_u\":"<<p.thermal.savedVelocity[0]<<"}";
 }std::cout<<"]}\n";
}
""")
    exe=out/'inventory'
    subprocess.run(['g++','-std=c++17','-O2','-I'+str(ROOT/'applications/CHMT'),'-I'+str(ROOT/'common'),str(probe),'-o',str(exe)],check=True)
    meta=json.loads((ROOT/'validation/molecular_transport/fixtures.json').read_text())['chmt-moving']
    solver=ROOT/'applications/CHMT/.build/native/CHMT';rows=[]
    for state in range(4):
        case=out/f'state-{state}';m.prepare(ROOT/'validation/molecular_transport/fixtures/chmt-moving',case,meta)
        m.entry(case/'system/controlDict','endTime',0)
        boundary=(case/'constant/polyMesh/boundary').read_text()
        wall=re.search(r'\bwall\s*\{([^}]+)',boundary)[1];face=int(re.search(r'startFace\s+(\d+)',wall)[1]) if state else -1
        prop=case/'constant/chmtProperties'
        prop.write_text(prop.read_text()+f"""
 enableParticles true;
 particle {{volumeMode Coupled; capacity 8; rhoSolid 2500; dragModel none; heatTransferModel RanzMarshall; contactModel reboundContact;}}
 particles {{initial {{id "42"; rng "1234567"; cell 0; face {face}; state {state};
 position (0.25 0 0.005); velocity (0 0 0); mass .002; diameter .001; temperature 2800;
 contactAge .0001; contactDuration .001; contactArea 0;
 thermalHistory {{temperature 2800; savedVelocity (100 0 0); contactMaximumArea 1e-6; contactPeakFraction .4; specificEnthalpyResidual .125;}}
 }}}}
""")
        with (case/'run.log').open('w') as log:subprocess.run([str(solver),'-case',str(case)],stdout=log,stderr=subprocess.STDOUT,check=True)
        checkpoint=case/'chmtOutput/checkpoint-0'
        initial=json.loads(subprocess.check_output([str(exe),str(checkpoint)],text=True))
        assert initial['time']==0 and len(initial['particles'])==1,initial
        q=initial['particles'][0];assert q['mechanical_J']==(10. if state>=2 else 0.) and q['saved_u']==100.,q
        restart=out/f'restart-{state}';shutil.copytree(case,restart,ignore=shutil.ignore_patterns('chmtOutput'))
        shutil.copytree(checkpoint,restart/'initial-checkpoint')
        prop=restart/'constant/chmtProperties';prop.write_text(prop.read_text()+'\nrestartDirectory "initial-checkpoint";\n')
        with (restart/'restart.log').open('w') as log:subprocess.run([str(solver),'-case',str(restart)],stdout=log,stderr=subprocess.STDOUT,check=True)
        resumed=json.loads(subprocess.check_output([str(exe),str(restart/'chmtOutput/checkpoint-0')],text=True))
        assert resumed==initial,(initial,resumed)
        csv=list((case/'chmtOutput').glob('particles-*.csv'));assert csv
        for f in csv:
            for row in table(f):
                assert float(row['contact_mechanical_energy'])==q['mechanical_J'],row
        legacy_result=None
        if state>=2:
            legacy=out/f'legacy-{state}'
            shutil.copytree(case,legacy,ignore=shutil.ignore_patterns('chmtOutput'))
            shutil.copytree(checkpoint,legacy/'initial-checkpoint')
            subprocess.run([str(exe),str(legacy/'initial-checkpoint'),'zero-mechanical'],check=True)
            old=json.loads(subprocess.check_output([str(exe),str(legacy/'initial-checkpoint')],text=True))
            assert old['particles'][0]['mechanical_J']==0 and old['particles'][0]['thermal_J']==q['thermal_J']
            prop=legacy/'constant/chmtProperties';prop.write_text(prop.read_text()+'\nrestartDirectory "initial-checkpoint";\n')
            with (legacy/'restart.log').open('w') as log:subprocess.run([str(solver),'-case',str(legacy)],stdout=log,stderr=subprocess.STDOUT,check=True)
            legacy_result=json.loads(subprocess.check_output([str(exe),str(legacy/'chmtOutput/checkpoint-0')],text=True))
            assert legacy_result==initial,(initial,legacy_result)
        rows.append(dict(state=state,initial=initial,restart=resumed,legacy_normalized=legacy_result))
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows))
if __name__=='__main__':main()
