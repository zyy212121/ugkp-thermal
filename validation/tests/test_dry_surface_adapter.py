"""Host CHMT dry adapter regression; no CUDA execution claim."""
from pathlib import Path
import subprocess
ROOT = Path(__file__).resolve().parents[2]
def compile_dry_surface_adapter(tmp_path, species):
    exe=tmp_path/('surface_adapter_'+str(species))
    subprocess.run(['g++','-std=c++17','-O1','-Wall','-Wextra','-Werror',
        '-Wno-misleading-indentation','-DUGKWP_GAS_SPECIES='+str(species),'-fsanitize=undefined','-fno-sanitize-recover=undefined',
        '-I'+str(ROOT/'applications/CHMT'),'-I'+str(ROOT/'common'),
        str(ROOT/'validation/tests/fixtures/dry_material/surface_adapter.cpp'),
        str(ROOT/'applications/CHMT/ablation/CpuSurfaceInterface.C'),'-o',str(exe)],check=True)
    subprocess.run([str(exe)],check=True)

def test_dry_surface_adapter(tmp_path):
    compile_dry_surface_adapter(tmp_path,2)

def test_dry_surface_adapter_three_species(tmp_path):
    compile_dry_surface_adapter(tmp_path,3)

def test_legacy_surface_policy_matches_preintegration(tmp_path):
    # Only the policy refactor is compared: all dependencies are the same build.
    baseline=subprocess.check_output(['git','show','3c9788a:applications/CHMT/ablation/InterfaceMath.H'],cwd=ROOT,text=True)
    baseline=baseline.replace('"../../../common/gasWall/WallModel.H"','"'+str(ROOT/'common/gasWall/WallModel.H')+'"')
    overlay=tmp_path/'old/ablation';overlay.mkdir(parents=True)
    (overlay/'InterfaceMath.H').write_text(baseline)
    # Local quoted includes must resolve beside the overridden original header.
    for name in ('InterfaceComposition.H','InterfaceCompositionArithmetic.H'):
        (overlay/name).write_text((ROOT/'applications/CHMT/ablation'/name).read_text())
    output=[]
    for label,extra in [('old',['-I'+str(overlay.parent)]),('new',[])]:
        exe=tmp_path/(label+'.exe')
        subprocess.run(['g++','-std=c++17','-O0','-ffp-contract=off',*extra,
            '-I'+str(ROOT/'applications/CHMT'),'-I'+str(ROOT/'common'),
            str(ROOT/'validation/tests/fixtures/dry_material/legacy_surface_policy.cpp'),'-o',str(exe)],check=True)
        output.append(subprocess.check_output([str(exe)],text=True))
    assert output[0]==output[1]
