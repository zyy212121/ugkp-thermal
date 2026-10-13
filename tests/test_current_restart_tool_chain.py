"""Current owner-format producer/consumer and full-state refinement witnesses."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'common'))
import thermal_particle_restart_current as current


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sample(marker):
    result = {}
    for name, dtype, width in current.field_specs(marker):
        shape = (3,) if width == 1 else (3, width)
        result[name] = np.full(shape, 1 + 2**-40, dtype=dtype)
    result.update(cell=np.array([1, 2, 1], dtype='<i4'), status=np.ones(3, dtype='<i4'),
                  rng=np.array([101, 202, 303], dtype='<u8'),
                  orig_id=np.array([2**53+1, 2**53+2, 2**53+3], dtype='<u8'),
                  pm=np.array([2., 3., 5.]), pStuck=np.zeros(3, dtype='u1'),
                  pStuckFaceId=np.full(3, -1, dtype='<i4'), pDepositionArea=np.zeros(3),
                  pContactPeakFraction=np.zeros(3))
    return result


@pytest.mark.parametrize('marker', [current.FSH_MARKER, current.CHT_MARKER])
def test_current_writer_exact_full_state_and_subfloat_values(tmp_path, marker):
    data = sample(marker)
    path = tmp_path / 'current.dat'
    current.write_restart(path, [data], 3, 2, cht=marker == current.CHT_MARKER)
    chunks = list(current.iter_chunks(path))
    assert len(chunks) == 2
    for name, dtype, width in current.field_specs(marker):
        restored = np.concatenate([chunk[name] for _, chunk in chunks])
        assert restored.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(restored, data[name])
    assert path.read_bytes().splitlines()[0].decode() == f'{marker} 3 2'


@pytest.mark.parametrize('app,marker', [('FSH', current.FSH_MARKER), ('CHT', current.CHT_MARKER)])
def test_current_refinement_preserves_all_owned_fields(tmp_path, app, marker):
    tools = ROOT / 'applications' / app / 'tools'
    sys.path.insert(0, str(tools))
    helper_name = 'fsh_particle_restart' if app == 'FSH' else 'ugkp_particle_restart'
    sys.modules.pop(helper_name, None)
    module = load(f'applications/{app}/tools/' + ('refine_fsh_particle_cells.py' if app == 'FSH' else 'refine_particle_cells.py'), f'current_refine_{app}')
    data = sample(marker)
    source, output = tmp_path / 'source.dat', tmp_path / 'refined.dat'
    current.write_restart(source, [data], 3, 2, cht=marker == current.CHT_MARKER)
    manifest = module.refine_restart(source, output, {1}, 4, None, 3091, chunk_particles=3)
    assert manifest['format'] == marker
    chunks = list(current.iter_chunks(output))
    restored = {name: np.concatenate([chunk[name] for _, chunk in chunks]) for name in data}
    parent = np.repeat(np.arange(3), [4, 1, 4])
    for name in data:
        if name not in ('pm', 'orig_id', 'rng'):
            np.testing.assert_array_equal(restored[name], data[name][parent])
    np.testing.assert_array_equal(restored['pm'], data['pm'][parent] / np.repeat([4, 1, 4], [4, 1, 4]))
    assert len(np.unique(restored['orig_id'])) == 9
    assert len(np.unique(restored['rng'])) == 9
    for name in ('pColdNodeSpecificEnthalpy', 'pCold2DNodeSpecificEnthalpy', 'puxOld') + (('pThermalEnthalpyResidual',) if app == 'CHT' else ()):
        before = np.sum(data['pm'].astype(np.longdouble).reshape((-1,) + (1,)*(data[name].ndim-1)) * data[name], axis=0)
        after = np.sum(restored['pm'].astype(np.longdouble).reshape((-1,) + (1,)*(restored[name].ndim-1)) * restored[name], axis=0)
        np.testing.assert_array_equal(after, before)


READERS = ['examples/thermal/postprocessing/fsh_restart_reader.py'] + [
    f'examples/thermal/{case}/assets/postprocessing/fsh_restart_reader.py'
    for case in ('MSS7_twoPhase_dense', 'MSS7_twoPhase_sparse', 'singleAluminaDrop')]


@pytest.mark.parametrize('reader_path', READERS)
@pytest.mark.parametrize('marker', [current.FSH_MARKER, current.CHT_MARKER])
def test_shipped_reader_current_bytes_standalone(tmp_path, reader_path, marker):
    import subprocess
    data = sample(marker)
    path = tmp_path / 'current.dat'
    current.write_restart(path, [data], 3, 2, cht=marker == current.CHT_MARKER)
    # Copy only the shipped script; it must not depend on repository-relative IO.
    copied = tmp_path / 'standalone_reader.py'
    copied.write_bytes((ROOT / reader_path).read_bytes())
    result = subprocess.run([sys.executable, '-c',
        "import standalone_reader as r; import sys; c=list(r.iter_fsh_chunks(sys.argv[1])); assert c[0]['original_id'][0]==2**53+1; assert c[0]['contact_maximum_area'][0]==1+2**-40; assert sum(len(x['mass']) for x in c)==3", str(path)], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_current_comparator_reads_carry_and_saved_velocity(tmp_path):
    module = load('applications/CHT/devtools/compare_cht_particle_restart.py', 'current_comparator')
    data = sample(current.CHT_MARKER)
    left, right = tmp_path/'left', tmp_path/'right'
    current.write_restart(left, [data], 3, 2, cht=True)
    data['pThermalEnthalpyResidual'][0] += 2**-39
    data['puxOld'][0] += 2**-39
    current.write_restart(right, [data], 3, 2, cht=True)
    result = module.compare(left, right)
    assert result['state_fields_equal']
    assert not result['continuous']['thermal_enthalpy_residual']['exact']
    assert not result['continuous']['ux_old']['exact']


def test_current_single_drop_uses_actual_double_profiles(tmp_path):
    module = load('examples/thermal/singleAluminaDrop/assets/postprocessing/single_alumina_drop_validation.py', 'current_single_drop')
    data = sample(current.FSH_MARKER)
    path = tmp_path/'drop'
    current.write_restart(path, [data], 3, 2)
    states = module.parse_particle_restart(path)
    assert len(states) == 3
    assert states[0].original_id == 2**53+1
    assert states[0].cold_node_specific_enthalpies_j_kg[0] == 1+2**-40
    assert states[0].maximum_contact_area_m2 == 1+2**-40


def test_historical_transient_refinement_uses_existing_zero_default(tmp_path):
    sys.path.insert(0, str(ROOT/'applications/FSH/tools'))
    sys.path.insert(0, str(ROOT/'applications/FSH/tests'))
    from historical_restart_fixture import write_historical_fsh_restart
    from fsh_particle_restart import collect_restart, input_metadata
    from refine_fsh_particle_cells import refine_restart
    data = sample(current.FSH_MARKER)
    data.update(pStuck=np.array([0,2,3],dtype='u1'), pStuckFaceId=np.array([-1,4,5],dtype='<i4'),
                pTheta=np.array([0.,.25,.5]), pContactDuration=np.ones(3),
                pContactMaximumArea=np.ones(3), pContactPeakFraction=np.array([0.,.5,.5]))
    source, output = tmp_path/'old',tmp_path/'new'
    write_historical_fsh_restart(source, [data], 3, 2, schema=6)
    manifest = refine_restart(source, output, {1}, 4, None, 3011)
    restored = collect_restart(output)
    assert manifest['format'] == current.FSH_MARKER
    assert manifest['defaulted_saved_velocity']
    assert manifest['source_field_dtypes']['pColdNodeSpecificEnthalpy'] == '<f4'
    for name in ('puxOld','puyOld','puzOld'):
        np.testing.assert_array_equal(restored[name], np.zeros(9))
    assert set(restored['pStuck']) == {0,2,3}
    assert input_metadata(output)['defaulted_saved_velocity'] is False


def test_native_legacy_default_preserved_without_transient_refusal():
    # Source witness only: full OpenFOAM runtime is not emulated here.
    for app in ('CHT','FSH'):
        text = (ROOT/f'applications/{app}/gpu/GpuResidentStrict.H').read_text()
        assert 'scalarField puxOld(nParticles,0.0), puyOld(nParticles,0.0), puzOld(nParticles,0.0);' in text
        assert 'UploadParticleSavedVelocity(handle_,nParticles,' in text
        assert 'Legacy transient restart lacks saved impact velocity' not in text
    assert 'initializeLegacySavedVelocity' not in (ROOT/'common/GpuThermalParticleRestartCodec.H').read_text()


@pytest.mark.parametrize('app,marker', [('FSH',current.FSH_MARKER),('CHT',current.CHT_MARKER)])
def test_python_writer_actual_native_codec_consumer(tmp_path, app, marker):
    import subprocess
    data = sample(marker)
    path, rewritten = tmp_path/'python.dat',tmp_path/'native.dat'
    current.write_restart(path,[data],3,2,cht=app=='CHT')
    types = {'<f4':'float','<f8':'double','<i4':'std::int32_t','<u8':'std::uint64_t','u1':'std::uint8_t','<u1':'std::uint8_t'}
    declarations, assignments = [],[]
    for name,dtype,width in current.field_specs(marker):
        declarations.append(f'std::vector<{types[dtype]}> {name}({3*width});')
        native = {'cell':'pCellId','status':'pStatus','rng':'pRng','orig_id':'pOrigId'}.get(name,name)
        assignments.append(f'view.{native}={name}.data();copy.{native}={name}.data();')
    code = f'''#include "applications/{app}/gpu/GpuFshParticleRestart.H"
#include <fstream>
#include <vector>
#include <cassert>
int main(){{using namespace Foam::gpuFshRestart;
{''.join(declarations)} MutableView view{{}};ConstView copy{{}};view.count=copy.count=3;
{''.join(assignments)}
std::ifstream input("{path}",std::ios::binary);std::string marker;size_t count;std::uint32_t bound;
input>>marker>>count>>bound;assert(marker==currentFormatName&&count==3);
readCurrentPayload(input,bound,view);assert(input.peek()==std::char_traits<char>::eof());
std::ofstream output("{rewritten}",std::ios::binary);writeCurrent(output,copy);
}}'''
    src,exe=tmp_path/'reader.cpp',tmp_path/'reader'
    src.write_text(code)
    result=subprocess.run(['g++','-std=c++17','-O2','-I',str(ROOT),str(src),'-o',str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    restored=list(current.iter_chunks(rewritten))[0][1]
    for name in data:
        np.testing.assert_array_equal(restored[name],data[name])


@pytest.mark.parametrize('marker', ['UGKP_THERMAL_PARTICLES_SCHEMA7_T32_BIN','UGKP_THERMAL_PARTICLES_SCHEMA7_T64_BIN',current.LEGACY_CHT_MARKER])
def test_historical_input_widths_and_current_output(tmp_path, marker):
    import struct
    # Explicit historical reference producer; never used by application tools.
    data=sample(marker)
    path=tmp_path/'historical'
    with path.open('wb') as stream:
        stream.write(f'{marker} 3 2\n'.encode())
        for start,count in ((0,2),(2,1)):
            stream.write(struct.pack('<I',count))
            for name,dtype,width in current.field_specs(marker):
                data[name][start:start+count].astype(dtype).tofile(stream)
    reader=load(READERS[0],'historical_current_reader')
    chunks=list(reader.iter_fsh_chunks(path))
    assert chunks[0]['contact_duration'].dtype==np.dtype('<f4' if 'T32' in marker else '<f8')
    assert chunks[0]['cold_ring_solid_mass'].dtype==np.dtype('<f4')
    out=tmp_path/'current'
    current.write_restart(out,(d for _,d in current.iter_chunks(path)),3,2,cht=True)
    restored={name:np.concatenate([d[name] for _,d in current.iter_chunks(out)]) for name,_,_ in current.field_specs(current.CHT_MARKER)}
    for name in data:
        np.testing.assert_array_equal(restored[name],data[name])
    assert restored['pColdRingSolidMass'].dtype==np.dtype('<f8')


@pytest.mark.parametrize('marker',[current.FSH_MARKER,current.CHT_MARKER])
def test_cht_cli_routes_current_owner_without_dropped_state(tmp_path,marker):
    import subprocess,json
    source,output=tmp_path/'source',tmp_path/'output'
    current.write_restart(source,[sample(marker)],3,2,cht=marker==current.CHT_MARKER)
    command=[sys.executable,str(ROOT/'applications/CHT/tools/refine_particle_cells.py'),str(source),str(output),'--cell','1','--factor','1']
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    manifest=json.loads(result.stdout)
    assert manifest['format']==marker
    assert manifest['source_format']==marker
    assert not manifest['defaulted_saved_velocity']
    assert manifest['source_field_dtypes']['pColdRingSolidMass']=='<f8'
    before=list(current.iter_chunks(source))
    after=list(current.iter_chunks(output))
    for name in sample(marker):
        np.testing.assert_array_equal(np.concatenate([d[name] for _,d in before]),np.concatenate([d[name] for _,d in after]))


def test_current_reader_uses_actual_chunk_count_not_descriptive_bound(tmp_path):
    data=sample(current.FSH_MARKER);source=tmp_path/'current'
    current.write_restart(source,[data],3,2)
    raw=source.read_bytes();source.write_bytes(raw.replace(b' 3 2\n',b' 3 0\n',1))
    for index,reader_path in enumerate(READERS):
        reader=load(reader_path,f'bound_reader_{index}')
        assert sum(len(c['mass']) for c in reader.iter_fsh_chunks(source))==3


@pytest.mark.parametrize('app,marker',[('CHT',current.CHT_MARKER),('FSH',current.FSH_MARKER)])
def test_empty_current_writer_and_application_reader(tmp_path,app,marker):
    sys.path.insert(0,str(ROOT/f'applications/{app}/tools'))
    filename='ugkp_particle_restart.py' if app=='CHT' else 'fsh_particle_restart.py'
    module=load(f'applications/{app}/tools/{filename}',f'empty_{app}_reader')
    path=tmp_path/'empty'
    current.write_restart(path,[],0,2,cht=app=='CHT')
    restored=module.collect_restart(path)
    assert set(restored)=={name for name,_,_ in current.field_specs(marker)}
    for name,dtype,width in current.field_specs(marker):
        assert restored[name].dtype==np.dtype(dtype)
        assert restored[name].shape==((0,) if width==1 else (0,width))


def test_current_case_reader_keeps_double_incoming_velocity(tmp_path):
    sys.path.insert(0,str(ROOT/'examples/thermal/postprocessing'))
    module=load('examples/thermal/postprocessing/bentsrm_case_reader.py','current_bentsrm_consumer')
    data=sample(current.FSH_MARKER)
    source=tmp_path/'case.dat'
    current.write_restart(source,[data],3,2)
    chunks=list(module.particle_chunks(source))
    assert chunks[0]['cold_ring_solid_mass'][0,0]==1+2**-40
    values,speeds=module.incoming_sommerfeld(source,{1:np.array([[1.,0.,0.]])})
    np.testing.assert_array_equal(speeds,np.full(2,1+2**-40))
    temperature=2327.0
    rho=1000*2.917;surface=.632;mu=5.77e-4*np.exp(9743./temperature)
    speed=1+2**-40;diameter=1+2**-40
    expected=np.sqrt(rho*speed**2*diameter/surface)*(rho*speed*diameter/mu)**.25
    np.testing.assert_allclose(values,np.full(2,expected),rtol=2e-15)


def test_single_drop_parser_and_reader_survive_standalone_copy(tmp_path):
    import subprocess
    assets=ROOT/'examples/thermal/singleAluminaDrop/assets/postprocessing'
    for filename in ('single_alumina_drop_validation.py','fsh_restart_reader.py'):
        (tmp_path/filename).write_bytes((assets/filename).read_bytes())
    current.write_restart(tmp_path/'current',[sample(current.FSH_MARKER)],3,2)
    result=subprocess.run([sys.executable,'-c',"from pathlib import Path; import single_alumina_drop_validation as d; p=d.parse_particle_restart(Path('current')); assert len(p)==3; assert p[0].original_id==2**53+1; assert p[0].cold_node_specific_enthalpies_j_kg[0]==1+2**-40"],cwd=tmp_path,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
