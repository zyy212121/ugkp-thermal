pass
from pathlib import Path
import shutil,subprocess
ROOT=Path(__file__).resolve().parents[2]
def test_native_checkpoint_precision_matrix(tmp_path):
    fp32=tmp_path/'source32'
    for rel in ['common','applications/CHMT']:
        shutil.copytree(ROOT/rel,fp32/rel,dirs_exist_ok=True,ignore=shutil.ignore_patterns('.build','Make','*.o','*.a','__pycache__'))
        if (ROOT/'precision/fp32'/rel).exists():shutil.copytree(ROOT/'precision/fp32'/rel,fp32/rel,dirs_exist_ok=True)
    build=fp32/'common/PrecisionBuild.H'
    build.write_text(build.read_text().replace('compiledBits = 64','compiledBits = 32'))
    for bits,source in [(64,ROOT),(32,fp32)]:
        subprocess.run(['g++','-std=c++17','-O0','-I'+str(source/'applications/CHMT'),'-I'+str(source/'common'),str(ROOT/'validation/tests/chmt_native_checkpoint.cpp'),'-o',str(tmp_path/f'probe{bits}')],check=True)
    results=[]
    for bits,action,target in [(64,'write',64),(32,'write',32),(64,'read',64),(32,'read',32),(32,'read',64),(64,'reject',32)]:
        p=subprocess.run([str(tmp_path/f'probe{bits}'),action,str(tmp_path/f'case{target}')],capture_output=True,text=True)
        assert p.returncode==0,p.stdout+p.stderr
        results.append(p.stdout+p.stderr)
    assert 'higher than selected precision 32' in results[-2]
    assert 'lower than selected precision 64' in results[-1]
    for bits in [32,64]:
        old=tmp_path/f'old{bits}';old.mkdir()
        data=bytearray((tmp_path/f'case{bits}/state.chmt').read_bytes());data[8:16]=(14).to_bytes(8,'little')
        (old/'state.chmt').write_bytes(data)
        shutil.copy2(tmp_path/f'case{bits}/ugkpPrecision',old/'ugkpPrecision')
        p=subprocess.run([str(tmp_path/f'probe{bits}'),'reject',str(old)],capture_output=True,text=True)
        assert p.returncode==0 and 'unsupported checkpoint schema' in p.stdout,p.stdout+p.stderr
    assert (tmp_path/'case64/state.chmt').stat().st_size-(tmp_path/'case32/state.chmt').stat().st_size==3*32
