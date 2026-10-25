from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import gzip
import hashlib
import os
import struct
import subprocess

APP = (SOURCE_ROOT / 'applications/CHMT')
REPO = APP.parents[1]
SCHEMA6_SOURCE = 'c957956874d5b7b32bf570bad47f6bc36ecce72d'
SCHEMA6_CHECKPOINT_BLOB = '9c09e4b59e6ff591935ed5ee9c46e5e6b627f3a1'
FIXTURES = {'dry-3fe0000000000000.chmt.gz': '31d111bfbd3ee224beaa0976a395d9f4fd54de5e395e0bcb5c83877ee791235c', 'dry-8000000000000000.chmt.gz': '83f26f632d1adaeb8bef871618a4b6952e0b5fbbfc272b83f0d11016d0ac745e', 'dry-0000000000000001.chmt.gz': 'cac22c3acd4352ae63ecf3c70d951436db37777f17a396fcccf9a19a916c323a', 'dry-7ff0000000000000.chmt.gz': '0257ac0394fabc3de01f4f9e525e3cafa67f89ff8f2e1cb82b4bca8c3246c142', 'dry-fff0000000000000.chmt.gz': 'c784b47e3a658ca9026e0179c22fec1241324d00b317acb7637f16d392f91aa2', 'dry-7ff8000000001234.chmt.gz': 'e560b28867c3c465bfac48d5266022cb3081596399596103b1571d478ae6e8f5', 'dry-fff8000000004321.chmt.gz': 'a1f7aa76011074ecf7fbe32c784ae72f721103b2c33aee3b31e0ee9aace7e1cc', 'wet-3fe0000000000000.chmt.gz': '3f06d6f7f9fd218342d8eeaf0f976580ccc38ac77cfd3d4e8e444a557a7cd20d', 'wet-8000000000000000.chmt.gz': '95a865fa2d5f572a38460dcb100b00508759323daf3f8ccb223036f09d0f1955', 'wet-0000000000000001.chmt.gz': '587065052c3962a317d86038427288d5e9d9890f3648c803dda9ef8ee8b383bb', 'wet-7ff0000000000000.chmt.gz': '0e6edd7dedd6570e23efaa96b04344c7dc80240776b490483e3b186af0bc57ed', 'wet-fff0000000000000.chmt.gz': 'a8fde1fa09d5063657588ac43984b33a38f33c57f367b574631f3c882b4c146d', 'wet-7ff8000000001234.chmt.gz': '13a124eca0ac844343e84fbc5cd5d4c55f146a0d869efde0dec25045e9f5623a', 'wet-fff8000000004321.chmt.gz': 'd9efa1d47ee848af720e4dbb3419d8a9be0eea63cfab0b070f6f6bc69c6aa18f'}


def wire(path):
    raw = path.read_bytes()
    config_size, state_size = struct.unpack_from('<QQ', raw, 48)
    assert len(raw) == 72+config_size+state_size
    return raw, raw[72:72+config_size], raw[72+config_size:]


def test_schema6_physical_payload_roundtrip(tmp_path):
    fixture = ((VALIDATION_ROOT / 'solvers/CHMT/tests/test_restart.cpp')).read_text().split(' const char* scratch=')[0]
    fixture = fixture.replace('int main(){', 'int main(int argc,char** argv){')
    fixture = fixture.replace('"../devtools/multirate/test_sweep_constraints.cpp"',
                              '"'+str((VALIDATION_ROOT / 'solvers/CHMT/devtools/multirate/test_sweep_constraints.cpp'))+'"')
    fixture += r'''
 require(argc==5,"schema6 reader arguments");
 const auto pattern=std::stoull(argv[4],nullptr,16);Real value;std::memcpy(&value,&pattern,8);
 HostState read;std::string error;require(readCheckpoint(argv[1],model,read,error),error);
 const Real* fields[]={&read.time,&read.solid[0].energy,&read.solidSweepRemainder[0],
   &read.filmAux[0].temperature,&read.gasSstAudit.sourceK};
 for(const Real* field:fields)require(std::memcmp(field,&value,8)==0,"legacy physical field exact bits");
 require(read.solidMesh.points.size()==h.solidMesh.points.size(),"legacy actual geometry count");
 require(read.filmAux.size()==2&&read.solid.size()==2,"legacy physical owner field counts");
 require(read.film.size()==(std::string(argv[3])=="wet"?2u:0u),"legacy wet inventory count");
 if(!read.film.empty())require(read.film[0].mass==.25&&read.film[1].enthalpy==123,"legacy film inventory values");
 require(read.gasTemperature.empty(),"legacy payload has no invented temperature state");
 require(writeCheckpoint(argv[2],model,read,error),error);return 0;
}
'''
    source = tmp_path/'reader.cpp'
    source.write_text(fixture)
    executable = tmp_path/'reader'
    command = ['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic',
               '-Wno-unused-function','-I'+str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'),'-I'+str(REPO/'common'),
               str(source),str(APP/'restart/Checkpoint.C'),str(APP/'mesh/Geometry.C'),
               '-Wl,--wrap=fsync','-o',str(executable)]
    env = dict(os.environ,TMPDIR=str(tmp_path))
    subprocess.run(command,check=True,env=env)
    for name, checksum in FIXTURES.items():
        raw = gzip.decompress(((VALIDATION_ROOT / 'solvers/CHMT/tests/fixtures/checkpoint_schema6')/name).read_bytes())
        assert hashlib.sha256(raw).hexdigest() == checksum
        mode,bits = name.removesuffix('.chmt.gz').split('-')
        old = tmp_path/name.removesuffix('.chmt.gz')
        old.mkdir()
        (old/'state.chmt').write_bytes(raw)
        new = tmp_path/(old.name+'-new')
        subprocess.run([str(executable),str(old),str(new),mode,bits],check=True,env=env)
        old_raw,_,old_state = wire(old/'state.chmt')
        new_raw,_,new_state = wire(new/'state.chmt')
        assert old_state == new_state
        assert old_raw[:8] == new_raw[:8] and old_raw[16:40] == new_raw[16:40]
        assert struct.unpack_from('<Q',old_raw,8)[0] == 6
        assert struct.unpack_from('<Q',new_raw,8)[0] == 9
