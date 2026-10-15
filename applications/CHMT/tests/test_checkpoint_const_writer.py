"""Real const traversal parity and allocation census against a full-state copy."""
from pathlib import Path
import hashlib
import json
import os
import subprocess

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
BASE = 'isolated-full-state-copy'


def _probe(tmp_path, label, source):
    directory = tmp_path / label
    directory.mkdir()
    # Instrument only actual writer calls. No production counters or test hooks.
    source = source.replace('namespace chmt { namespace {',
        'namespace chmt { unsigned long long hashCalls=0,hashBytes=0,modelCopies=0,stateCopies=0; namespace {', 1)
    source = source.replace('std::size_t n){std::uint64_t h=',
        'std::size_t n){++hashCalls;hashBytes+=n;std::uint64_t h=', 1)
    source = source.replace('ModelConfig model=config;HostState copy=state;',
        '++modelCopies;ModelConfig model=config;++stateCopies;HostState copy=state;')
    (directory / 'Checkpoint.C').write_text(source)
    fixture = (APP / 'tests/test_restart.cpp').read_text()
    fixture = fixture.replace('/tmp/chmt-restart-regression.', str(directory / 'restart.'))
    fixture = fixture.replace('writeCheckpoint(', 'measuredWriteCheckpoint(')
    # Expand supplied-state coverage without asking runtime physics to admit it.
    marker = 'unlink(file.c_str());rmdir(directory.c_str());unlink(survivorFile.c_str());'
    assert marker in fixture
    special = r'''
 for(std::uint64_t pattern:{0ULL,0x8000000000000000ULL,1ULL,0x7ff0000000000000ULL,
       0xfff0000000000000ULL,0x7ff8000000001234ULL,0xfff8000000004321ULL}) {
  Real value;std::memcpy(&value,&pattern,8);auto supplied=h;
  supplied.time=value;supplied.solid[0].energy=value;supplied.solidSweepRemainder={value,-0.};
  supplied.gasSstAudit.sourceK=value;supplied.filmAux[0].temperature=value;
  supplied.gas.resize(2);supplied.gasChemistryAudit.resize(2);supplied.sst.resize(2);supplied.particles.resize(2);
  supplied.gasVoidFraction={value,-0.};supplied.normalEnthalpy={value};supplied.normalLiquidFraction={value};supplied.normalAcceptedWallHeat={value};
  for(auto& q:supplied.gas){q.mass=value;q.energy=value;q.momentum={value,-0.,value};for(int k=0;k<Ns;++k)q.species[k]=value;}
  for(auto& q:supplied.sst){q.rhoK=value;q.rhoOmega=value;}
  for(auto& q:supplied.gasChemistryAudit){q.integratedTime=value;q.energyResidual=value;for(int k=0;k<Ns;++k)q.speciesMassChange[k]=value;}
  for(auto& q:supplied.particles){q.mass=value;q.energy=value;q.position={value,-0.,value};q.velocity={value,value,-0.};q.contactRelativeVelocity={value,-0.,value};q.contactMechanicalEnergy=value;}

  auto config=model;config.gasMechanismSource=std::string(512,'x');config.physics.cfl=value;
  require(measuredWriteCheckpoint(directory,config,supplied,error),error);
  const auto encoded=bytes(file);HostState decoded;
  require(readCheckpoint(directory,config,decoded,error),error);
  require(std::memcmp(&decoded.time,&value,8)==0,"raw floating payload roundtrip");
  require(measuredWriteCheckpoint(directory,config,decoded,error)&&bytes(file)==encoded,
          "all supplied invalid-state fields retain exact serialization");
 }
'''
    fixture = fixture.replace(marker, special + '\n ' + marker)
    prelude = r'''
#include "restart/Checkpoint.H"
#include <cstdlib>
#include <new>
#include <fstream>
#include <iterator>
#include <string>
static bool counting=false;
static unsigned long long allocationCalls=0,allocationBytes=0;
void* operator new(std::size_t n){if(counting){++allocationCalls;allocationBytes+=n;}if(void* p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void* operator new[](std::size_t n){return ::operator new(n);}
void operator delete(void* p) noexcept{std::free(p);}
void operator delete[](void* p) noexcept{std::free(p);}
void operator delete(void* p,std::size_t) noexcept{std::free(p);}
void operator delete[](void* p,std::size_t) noexcept{std::free(p);}
namespace chmt {
extern unsigned long long hashCalls,hashBytes,modelCopies,stateCopies;
bool measuredWriteCheckpoint(const std::string& directory,const ModelConfig& config,const HostState& state,std::string& error){
 static unsigned call=0;allocationCalls=allocationBytes=hashCalls=hashBytes=modelCopies=stateCopies=0;
 counting=true;bool result=writeCheckpoint(directory,config,state,error);counting=false;
 const auto allocations=allocationCalls,allocated=allocationBytes,hc=hashCalls,hb=hashBytes;
 const std::string prefix=PROBE_OUTPUT+std::to_string(call++);
 std::ofstream meta(prefix+".counts");meta<<result<<' '<<allocations<<' '<<allocated<<' '<<hc<<' '<<hb<<' '<<modelCopies<<' '<<stateCopies<<'\n';
 if(result){std::ifstream in(directory+"/state.chmt",std::ios::binary);std::ofstream out(prefix+".bin",std::ios::binary);out<<in.rdbuf();}
 return result;
}
}
'''
    (directory / 'probe.cpp').write_text(prelude + fixture)
    executable = directory / 'probe'
    command = ['g++', '-std=c++14', '-O2', '-Wall', '-Wextra', '-Werror', '-pedantic',
               '-Wno-mismatched-new-delete', '-DPROBE_OUTPUT="'+str(directory / 'write-')+'"',
               '-I'+str(APP), '-I'+str(APP/'tests'), '-I'+str(APP/'restart'),
               '-I'+str(REPO/'common'), str(directory/'probe.cpp'),
               str(directory/'Checkpoint.C'), str(APP/'mesh/Geometry.C'),
               '-Wl,--wrap=fsync', '-o', str(executable)]
    env = dict(os.environ, TMPDIR=str(tmp_path))
    subprocess.run(command, check=True, env=env)
    result = subprocess.run([str(executable)], check=True, text=True, capture_output=True, env=env)
    (directory/'run.log').write_text(result.stdout+result.stderr)
    cases = []
    for path in sorted(directory.glob('write-*.counts'), key=lambda p:int(p.stem.split('-')[1])):
        result, calls, size, hashes, hashed, model_copies, state_copies = map(int, path.read_text().split())
        payload = path.with_suffix('.bin')
        cases.append(dict(success=result, allocations=calls, allocation_bytes=size,
                          hash_calls=hashes, hash_bytes=hashed, model_copies=model_copies, state_copies=state_copies,
                          sha256=hashlib.sha256(payload.read_bytes()).hexdigest() if result else None))
    return cases


def test_const_writer_exact_bytes_and_fewer_allocations(tmp_path):
    baseline = (APP/'restart/Checkpoint.C').read_text()
    baseline = baseline.replace('Encoder configData,payload;configData(config);payload(state);',
                                'ModelConfig model=config;HostState copy=state;Encoder configData,payload;configData(model);payload(copy);')
    before = _probe(tmp_path, 'base', baseline)
    after = _probe(tmp_path, 'current', (APP/'restart/Checkpoint.C').read_text())
    receipt = dict(base=BASE, before=before, after=after)
    (tmp_path/'census.json').write_text(json.dumps(receipt, indent=2)+'\n')
    assert len(before)==len(after) and len(before)>20
    assert sum(not item['success'] for item in after)==3  # Real fsync faults.
    for old, new in zip(before, after):
        assert old['success']==new['success']
        assert old['sha256']==new['sha256']
        assert old['hash_calls']==new['hash_calls']==2
        assert old['hash_bytes']==new['hash_bytes']
        assert old['model_copies']==old['state_copies']==1
        assert new['model_copies']==new['state_copies']==0
        assert new['allocations'] < old['allocations'], 'writer still allocates whole-state copies'
        assert new['allocation_bytes'] < old['allocation_bytes']


def test_encoder_const_composites_and_mutable_decoder(tmp_path):
    source = tmp_path/'const.cpp'
    source.write_text('#include "'+str(APP/'restart/Checkpoint.C')+'"\n' + r'''
int main(){
 const chmt::ModelConfig model{};const chmt::HostState state{};
 chmt::Encoder config,payload;config(model);payload(state);
 chmt::HostState decoded;chmt::Decoder reader(payload.data);reader(decoded);
 chmt::Encoder again;again(decoded);
 return again.data==payload.data&&reader.offset==payload.data.size()?0:1;
}
''')
    executable = tmp_path/'const'
    subprocess.run(['g++','-std=c++14','-O2','-Wall','-Wextra','-Werror','-pedantic',
                    '-I'+str(APP),'-I'+str(REPO/'common'),str(source),'-o',str(executable)],
                   check=True,env=dict(os.environ,TMPDIR=str(tmp_path)))
    subprocess.run([str(executable)],check=True)
