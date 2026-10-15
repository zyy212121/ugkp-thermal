import subprocess
from test_shared_device_storage import APP, SOURCE, STUB


def test_actual_temperature_survives_storage_checkpoint_and_first_recovery(tmp_path):
    source = SOURCE[:SOURCE.index(' chmt::SharedGasDeviceStorage storage;')]
    source = source.replace('int main(){', 'int main(int argc,char** argv){assert(argc==2);')
    source = source.replace('molarMass 0.032', 'molarMass 0.028')
    source += r'''
 h.gas[0].mass=0x1.0000003f770b2p-3;
 h.gas[0].momentum={-0x1.9ed40bfbfc652p-18,0,0};
 h.gas[0].energy=0x1.b3621aa0a4edep+15;
 h.gas[0].species[0]=h.gas[0].mass;h.gas[0].species[1]=0;
 m.volumes={0x1p-3};model.physics.enableSst=true;h.sst={{0.0125,1.25}};
 chmt::SharedGasDeviceStorage running;assert(running.configure(model,gas,{},h,error));
 auto& current=running.hostView();current.Tgas[0]=0x1.2bffe86ea2c14p+9;
 assert(running.downloadState(h,error));
 assert(chmt::writeCheckpoint(argv[1],model,h,error));
 chmt::HostState saved;assert(chmt::readCheckpoint(argv[1],model,saved,error));
 chmt::SharedGasDeviceStorage restarted;assert(restarted.configure(model,gas,{},saved,error));
 auto& resumed=restarted.hostView();
 assert(std::memcmp(current.Tgas,resumed.Tgas,sizeof(double))==0);
 const double* live[]={current.rho,current.rhoUx,current.rhoUy,current.rhoUz,current.rhoE,
   current.gasSpecies.rho,current.rhoK,current.rhoOmega};
 const double* again[]={resumed.rho,resumed.rhoUx,resumed.rhoUy,resumed.rhoUz,resumed.rhoE,
   resumed.gasSpecies.rho,resumed.rhoK,resumed.rhoOmega};
 for(int k=0;k<8;++k)assert(std::memcmp(live[k],again[k],(k==5?2:1)*sizeof(double))==0);
 const double density=current.rho[0];double partial[2]={current.gasSpecies.rho[0],current.gasSpecies.rho[1]};
 const double ux=current.rhoUx[0]/density,uy=current.rhoUy[0]/density,uz=current.rhoUz[0]/density;
 const double internal=current.rhoE[0]-.5*density*(ux*ux+uy*uy+uz*uz);
 double continuous=current.Tgas[0],restored=resumed.Tgas[0];
 assert(ugkwp::invertMixtureEnergy(partial,internal,gas.thermoView<2>(),gas.chemistryControls.thermo,continuous)==ugkwp::ThermoStatus::Success);
 assert(ugkwp::invertMixtureEnergy(partial,internal,gas.thermoView<2>(),gas.chemistryControls.thermo,restored)==ugkwp::ThermoStatus::Success);
 assert(std::memcmp(&continuous,&restored,sizeof(double))==0);
 current.Tgas[0]=701.25;
 assert(running.uploadState(saved,error));assert(std::memcmp(current.Tgas,resumed.Tgas,sizeof(double))==0);
 std::printf("temperature continuation=%a restart=%a seed=%a\n",continuous,restored,resumed.Tgas[0]);
}
'''
    source = '#include "restart/Checkpoint.H"\n' + source
    (tmp_path/'cuda_runtime.h').write_text(STUB)
    (tmp_path/'probe.cpp').write_text(source)
    binary = tmp_path/'probe'
    command = ['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic',
               '-I'+str(tmp_path),'-I'+str(APP),'-I'+str(APP.parents[1]/'common'),
               str(tmp_path/'probe.cpp'),str(APP/'restart/Checkpoint.C'),'-o',str(binary)]
    build = subprocess.run(command,capture_output=True,text=True)
    (tmp_path/'build.log').write_text(build.stdout+build.stderr)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(binary),str(tmp_path/'checkpoint')],capture_output=True,text=True)
    (tmp_path/'run.log').write_text(run.stdout+run.stderr)
    assert run.returncode == 0, run.stdout+run.stderr



import struct
import pytest


@pytest.fixture(scope='module')
def numerical_checkpoint_probe(tmp_path_factory):
    root = tmp_path_factory.mktemp('numerical_checkpoint_codec')
    source = root/'codec.cpp'
    source.write_text(r'''
#include "restart/Checkpoint.H"
#include <cassert>
#include <cstring>
#include <cstdio>
int main(int argc,char** argv){
 assert(argc>=3);chmt::ModelConfig model;chmt::HostState state;std::string error;
 if(std::string(argv[1])=="write"){
  state.time=.5;state.acceptedSteps=state.commitSequence=3;state.gas.resize(2);
  state.gas[0].mass=1;state.gas[0].energy=100;state.gas[1].mass=2;state.gas[1].energy=200;
  state.gasTemperature={0x1.2bffe86ea2c14p+9,-0.};
  assert(chmt::writeCheckpoint(argv[2],model,state,error));return 0;
 }
 state.time=123;
 if(!chmt::readCheckpoint(argv[2],model,state,error)){
  assert(state.time==123);std::fprintf(stderr,"unchanged: %s\n",error.c_str());return 2;
 }
 assert(state.time==.5&&state.gas[0].mass==1&&state.gas[1].energy==200);
 if(argc==4)assert(chmt::writeCheckpoint(argv[3],model,state,error));
 std::printf("temperature_count=%zu\n",state.gasTemperature.size());return 0;
}
''')
    command = ['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic',
               '-I'+str(APP),'-I'+str(APP.parents[1]/'common'),str(source),
               str(APP/'restart/Checkpoint.C'),'-o',str(root/'codec')]
    result = subprocess.run(command,capture_output=True,text=True)
    (root/'build.log').write_text(result.stdout+result.stderr)
    assert result.returncode == 0, result.stderr
    return root/'codec'


def _wire(path):
    raw = path.read_bytes()
    config_size, state_size = struct.unpack_from('<QQ',raw,48)
    assert len(raw) == 72+config_size+state_size
    return raw, raw[72+config_size:]


def _digest(raw):
    result = bytearray(raw)
    checksum = 14695981039346656037
    for value in result[72:]:
        checksum = ((checksum^value)*1099511628211)&((1<<64)-1)
    struct.pack_into('<Q',result,64,checksum)
    return result


def test_numeric_extension_and_legacy_state_are_both_readable(numerical_checkpoint_probe,tmp_path):
    binary = numerical_checkpoint_probe
    original = tmp_path/'original'
    subprocess.run([str(binary),'write',str(original)],check=True)
    raw, state = _wire(original/'state.chmt')
    extension_bytes, marker = struct.unpack_from('<Q8s',state,len(state)-16)
    assert marker == b'CHMTNUM1' and extension_bytes == 24
    assert state[-40:-16] == struct.pack('<Qdd',2,float.fromhex('0x1.2bffe86ea2c14p+9'),-0.)
    restored = tmp_path/'restored'
    run = subprocess.run([str(binary),'read',str(original),str(restored)],check=True,capture_output=True,text=True)
    assert 'temperature_count=2' in run.stdout
    assert (restored/'state.chmt').read_bytes() == raw
    old = bytearray(raw[:-40])
    struct.pack_into('<Q',old,8,6)
    struct.pack_into('<Q',old,40,11438850135072464462)
    struct.pack_into('<Q',old,56,len(state)-40)
    legacy = tmp_path/'legacy'
    legacy.mkdir()
    (legacy/'state.chmt').write_bytes(_digest(old))
    upgrade = tmp_path/'upgrade'
    run = subprocess.run([str(binary),'read',str(legacy),str(upgrade)],check=True,capture_output=True,text=True)
    assert 'temperature_count=0' in run.stdout
    assert _wire(upgrade/'state.chmt')[1] == state[:-40]


@pytest.mark.parametrize('missing',[1,8,16,24])
def test_incomplete_numeric_extension_preserves_read_output(numerical_checkpoint_probe,tmp_path,missing):
    binary = numerical_checkpoint_probe
    original = tmp_path/'original'
    subprocess.run([str(binary),'write',str(original)],check=True)
    raw, state = _wire(original/'state.chmt')
    truncated = bytearray(raw[:-missing])
    struct.pack_into('<Q',truncated,56,len(state)-missing)
    (original/'state.chmt').write_bytes(_digest(truncated))
    run = subprocess.run([str(binary),'read',str(original)],capture_output=True,text=True)
    assert run.returncode == 2
    assert 'unchanged:' in run.stderr and 'truncated' in run.stderr
