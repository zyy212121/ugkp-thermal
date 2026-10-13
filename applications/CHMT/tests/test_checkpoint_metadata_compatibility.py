"""Unmodified pinned and current writers/readers, with exact state wire parity."""
from pathlib import Path
import json
import os
import struct
import subprocess
import tarfile
import io

APP = Path(__file__).resolve().parents[1]
REPO = APP.parents[1]
BASE = '40e1b40efe009b1be0485329e95cc68c7d74c545'


def wire(path):
    raw = path.read_bytes()
    config_size, state_size = struct.unpack_from('<QQ', raw, 48)
    assert len(raw) == 72 + config_size + state_size
    return raw, raw[72:72+config_size], raw[72+config_size:]


def test_old_current_readers_and_exact_state_payload(tmp_path):
    pinned = tmp_path/'pinned'
    pinned.mkdir()
    archive = subprocess.check_output(['git', 'archive', BASE, 'applications/CHMT', 'common'], cwd=REPO)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(pinned, filter='data')
    fixture = (APP/'tests/test_restart.cpp').read_text().split(' const char* scratch=')[0]
    fixture = fixture.replace('int main(){', 'int main(int argc,char** argv){')
    # Use real geometry/field population from the existing checkpoint fixture.
    fixture += r'''
 require(argc==5,"probe arguments");
 #ifdef OLD_METADATA
 model.modelFingerprint="modelxx";model.speciesFingerprint="speciesxx";model.schemaFingerprint="schemaxxxxx";
 model.physics.modelFingerprint=17;model.physics.speciesFingerprint=23;model.physics.schemaFingerprint=29;
#endif
 const bool wet=std::string(argv[3])=="wet";model.physics.enableFilm=wet;
 if(wet){h.film.resize(h.surface.area.size());for(auto& f:h.film){f.mass=0.25;f.enthalpy=123.;f.species[0]=0.25;}}
 const std::uint64_t pattern=std::stoull(argv[4],nullptr,16);Real value;std::memcpy(&value,&pattern,8);
 h.time=value;h.solid[0].energy=value;h.solidSweepRemainder[0]=value;
 h.filmAux[0].temperature=value;h.gasSstAudit.sourceK=value;
 std::string error;
 if(std::string(argv[1])=="write")require(writeCheckpoint(argv[2],model,h,error),error);
 else {HostState read;require(readCheckpoint(argv[2],model,read,error),error);
       require(std::memcmp(&read.time,&value,8)==0,"read exact supplied bits");
       require(writeCheckpoint(std::string(argv[2])+".roundtrip",model,read,error),error);}
 return 0;
 }
'''
    source = tmp_path/'probe.cpp'
    # Prefix references its geometry fixture relatively; use an absolute source
    # include with each compiler's own pinned/current tree.
    executables = {}
    env = dict(os.environ, TMPDIR=str(tmp_path))
    for label, root in [('old', pinned), ('current', REPO)]:
        app = root/'applications/CHMT'
        code = fixture.replace('"../devtools/multirate/test_sweep_constraints.cpp"',
                               '"'+str(app/'devtools/multirate/test_sweep_constraints.cpp')+'"')
        source.write_text(code)
        executable = tmp_path/label
        subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic',
                        '-Wno-unused-function',('-DOLD_METADATA' if label=='old' else '-DCURRENT_METADATA'),'-I'+str(app),'-I'+str(root/'common'),
                        str(source),str(app/'restart/Checkpoint.C'),str(app/'mesh/Geometry.C'),
                        '-Wl,--wrap=fsync','-o',str(executable)],check=True,env=env)
        executables[label] = executable
    receipts = []
    for mode in ['dry', 'wet']:
        for bits in ['3fe0000000000000','8000000000000000','0000000000000001',
                     '7ff0000000000000','fff0000000000000','7ff8000000001234','fff8000000004321']:
            paths = {}
            for label, executable in executables.items():
                path = tmp_path/(label+'-'+mode+'-'+bits)
                subprocess.run([str(executable),'write',str(path),mode,bits],check=True,env=env)
                paths[label] = path
            old, old_config, old_state = wire(paths['old']/'state.chmt')
            new, new_config, new_state = wire(paths['current']/'state.chmt')
            assert old_state == new_state
            # Three numeric words plus three string lengths and supplied text disappear.
            # The unchanged outer header remains 72 bytes.
            assert len(old_config)-len(new_config) == 48 + 7 + 9 + 11
            assert old[:40] == new[:40]  # same magic/version/count representation
            assert old[56:64] == new[56:64]  # same HostState byte count
            for writer, path in paths.items():
                reader = 'current' if writer=='old' else 'old'
                subprocess.run([str(executables[reader]),'read',str(path),mode,bits],check=True,env=env)
                _, _, decoded_state = wire(Path(str(path)+'.roundtrip')/'state.chmt')
                assert decoded_state == old_state
            receipts.append(dict(mode=mode,bits=bits,state_bytes=len(old_state),
                                 old_config_bytes=len(old_config),new_config_bytes=len(new_config),
                                 cross_reads=2))
    (tmp_path/'compatibility.json').write_text(json.dumps(dict(base=BASE,cases=receipts),indent=2)+'\n')
