import subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def test_no_sommerfeld_threshold_read(tmp_path):
    source=(ROOT/'validation/tests/fixtures/particle_tracking/wall_incident_velocity.cpp').read_text()
    source=source.replace('int main(', 'int incident_main(')
    source=source.replace('GpuReal sommerfeldThreshold = GPU_R(20);', 'CountedThreshold sommerfeldThreshold = GPU_R(20);')
    source=source.replace('struct DeviceState', """struct CountedThreshold {
        GpuReal value; static int reads;
        CountedThreshold(GpuReal v):value(v){}
        operator GpuReal() const {++reads;return value;}
        CountedThreshold& operator=(GpuReal v){value=v;return *this;}
    }; int CountedThreshold::reads=0;
    struct DeviceState""")
    source=source.replace('trackOneParticleLocalFaceWalk(s,', 'trackOneParticleLocalFaceWalk<false>(s,')
    source=source.replace('model == particleWallReboundContact || !deposit', 'true')
    source+='\nint main(){for(int model:{1,3,4}) for(bool deposit:{false,true}) run("constant",model,deposit); require(CountedThreshold::reads==0,"no Sommerfeld threshold reads");return 0;}\n'
    path=tmp_path/'no_sommerfeld.cpp';path.write_text(source)
    for fsh in (0,1):
        exe=tmp_path/f'no_sommerfeld_{fsh}'
        subprocess.run(['g++','-std=c++17','-O2',f'-DINCIDENT_FSH={fsh}',
            '-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/wall'),str(path),'-o',str(exe)],check=True)
        subprocess.run([str(exe)],check=True)
