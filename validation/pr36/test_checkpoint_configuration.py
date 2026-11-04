import sys,subprocess,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'validation/pr31'))
from test_particle_checkpoint_history import build,APP

def test_rr_shape_and_disabled_sommerfeld_are_restart_identity(tmp_path):
 exe=build(tmp_path,r'''
#include <limits>
int main(int argc,char**argv){assert(argc==2);ModelConfig m;m.physics.enableParticles=true;
 m.particlePhase.dn=2.5;m.particleContact.sommerfeldThreshold=std::numeric_limits<double>::infinity();
 HostState s,r;fill(s);std::string error;assert(writeCheckpoint(argv[1],m,s,error));
 assert(readCheckpoint(argv[1],m,r,error));
 auto changed=m;changed.particlePhase.dn=3;assert(!readCheckpoint(argv[1],changed,r,error));
 changed=m;changed.particleContact.sommerfeldThreshold=20;assert(!readCheckpoint(argv[1],changed,r,error));
}
''')
 subprocess.run([exe,tmp_path/'new'],check=True)

def test_schema11_without_rr_shape_decodes_without_shifting_contact_fields(tmp_path):
 # Reproduce the old field sequence explicitly: schema11 had no dn slot.
 source=(APP/'restart/Checkpoint.C').read_text()
 source=source.replace('header.integer(checkpointSchema,8);','header.integer(11,8);')
 old_schema=re.search(r'const char\* const FieldSchema="([^"]+)"',source)[1].replace('phase(dn) contact','phase contact')
 h=14695981039346656037
 for b in old_schema.encode():h=((h^b)*1099511628211)&((1<<64)-1)
 source=source.replace('header.integer(checksum(reinterpret_cast<const unsigned char*>(FieldSchema),std::strlen(FieldSchema)),8);',f'header.integer({h}ULL,8);')
 source=source.replace('visitParticlePhaseConfiguration(configData,config.particlePhase);','visitParticlePhaseConfiguration(configData,config.particlePhase,false);')
 old=tmp_path/'schema11.C';old.write_text(source)
 writer=build(tmp_path,r'''
int main(int argc,char**argv){assert(argc==2);ModelConfig m;m.physics.enableParticles=true;
 m.particleContact.model=4;m.particleContact.sommerfeldThreshold=19.;HostState s;fill(s);std::string error;
 assert(writeCheckpoint(argv[1],m,s,error));}
''',old,'old-writer')
 subprocess.run([writer,tmp_path/'old'],check=True)
 reader=build(tmp_path,r'''
int main(int argc,char**argv){assert(argc==2);ModelConfig m;m.physics.enableParticles=true;
 m.particleContact.model=4;m.particleContact.sommerfeldThreshold=19.;HostState s;std::string error;
 assert(readCheckpoint(argv[1],m,s,error));assert(s.particles.size()==4);
 m.particlePhase.dn=2;assert(!readCheckpoint(argv[1],m,s,error));}
''',name='reader')
 subprocess.run([reader,tmp_path/'old'],check=True)
