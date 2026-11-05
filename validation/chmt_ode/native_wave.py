"""Native frozen-mixture wave outlet: initial value, evolution, and restart.
Requires OpenFOAM 10 and the locally built CHMT. Evidence goes to --output.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'validation/pr31'))
from chmt_particles_native import m
PROBE = r'''#include "restart/Checkpoint.C"
#include <iostream>
#include <iomanip>
int main(int argc,char** argv){
 if(argc!=4)return 2;
 std::ifstream f(std::string(argv[1])+"/state.chmt",std::ios::binary|std::ios::ate);
 if(!f)return 3;chmt::Bytes bytes(size_t(f.tellg()));f.seekg(0);f.read(reinterpret_cast<char*>(bytes.data()),bytes.size());
 chmt::Decoder decoder(bytes);decoder.offset=72;chmt::ModelConfig model;decoder(model);
 if(decoder.integer(8)!=0x31474643504b4755ULL)return 4;
 ugkp::thermal::visitParticlePhaseConfiguration(decoder,model.particlePhase);
 ugkp::thermal::visitParticleContactConfiguration(decoder,model.particleContact);
 chmt::HostState state;std::string error;
 if(!chmt::readCheckpoint(argv[1],model,state,error)){std::cerr<<error;return 5;}
 std::cout<<std::setprecision(17)<<"[";bool first=true;
 for(int i=std::stoi(argv[2]);i<std::stoi(argv[2])+std::stoi(argv[3]);++i){
  if(!first)std::cout<<",";first=false;std::cout<<state.gasMesh.boundaryPrimitive[i].pressure;
 }
 std::cout<<"]\n";
}
'''
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    probe = out / 'wave.cpp'
    probe.write_text(PROBE)
    exe = out / 'wave'
    subprocess.run(['g++', '-std=c++17', '-O2', '-I'+str(ROOT/'applications/CHMT'),
                    '-I'+str(ROOT/'common'), str(probe), '-o', str(exe)], check=True)
    meta = json.loads((ROOT/'validation/molecular_transport/fixtures.json').read_text())['chmt-moving']
    case = out / 'continuous'
    m.prepare(ROOT/'validation/molecular_transport/fixtures/chmt-moving', case, meta)
    m.entry(case/'system/controlDict', 'endTime', 2e-6)
    config = case/'constant/chmtProperties'
    config.write_text(config.read_text().replace('model reactingSst;', 'model equilibriumOde;'))
    (case/'0/p').write_text('''FoamFile { version 2.0; format ascii; class volScalarField; object p; }
dimensions [1 -1 -2 0 0 0 0]; internalField uniform 101325;
boundaryField {
 inlet {type zeroGradient;} wall {type zeroGradient;}
 top {type zeroGradient;} sides {type zeroGradient;}
 outlet {type waveTransmissive; psi psi; gamma 1.2; fieldInf 101325; lInf 0.1; value uniform 150000;}
}
''')
    solver = ROOT/'applications/CHMT/.build/native/CHMT'
    m.run([solver, '-case', case], case/'run.log')
    patch = re.search(r'\boutlet\s*\{([^}]+)', (case/'constant/polyMesh/boundary').read_text())[1]
    first = re.search(r'startFace\s+(\d+)', patch)[1]
    count = re.search(r'nFaces\s+(\d+)', patch)[1]
    history = [json.loads(subprocess.check_output([exe, case/f'chmtOutput/checkpoint-{i}', first, count], text=True)) for i in range(3)]
    assert history[0] and all(p == 150000 for p in history[0]), history
    assert all(101325 < p < 150000 for p in history[1]), history
    assert all(b < a for a, b in zip(history[1], history[2])), history
    restart = m.resume(case, out/'restart', 'CHMT', solver)
    report = dict(wave_pressure_Pa=history, restart=restart)
    (out/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))
if __name__ == '__main__':
    main()
