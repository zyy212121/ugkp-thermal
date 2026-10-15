"""Complete supplied consumers; actual host bodies, no native GPU claim."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import importlib.util
import subprocess
import sys
import pytest
from unittest.mock import patch

APP=(SOURCE_ROOT / 'applications/CHMT')
ROOT=APP.parents[1]


def run(tmp_path,code,geometry=False):
    src=tmp_path/'probe.cpp';src.write_text(code);exe=tmp_path/'probe'
    command=['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic','-I'+str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'),'-I'+str((VALIDATION_ROOT / 'solvers/CHMT/tests')),'-I'+str(ROOT/'common'),str(src)]
    if geometry:command.append(str(APP/'mesh/Geometry.C'))
    subprocess.run(command+['-o',str(exe)],check=True)
    result=subprocess.run([str(exe)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return result.stdout


def test_supplied_output_metadata_and_real_stream_error(tmp_path):
    source=((VALIDATION_ROOT / 'solvers/CHMT/tests/test_coupled_output.cpp')).read_text()
    begin=source.index('    for(int bad=0;')
    source=source[:begin]+r'''
    auto supplied=state;supplied.commitSequence=77;supplied.time=std::numeric_limits<Real>::infinity();
    supplied.ledger[0].consumerMask=0;supplied.filmStorage.resize(1);
    for(auto writer:writers){std::ostringstream stream;check(writer(stream,supplied,model,error),error);check(!stream.str().empty(),"supplied CSV produced");}
    supplied.gasWallDiagnosticTime=123;supplied.gasWallDiagnostics.resize(1);
    supplied.gasWallDiagnostics[0].traceTemperature=std::numeric_limits<Real>::quiet_NaN();
    std::ostringstream wall;check(io::writeAcceptedWallLayerFields(wall,supplied,model,error),error);
    check(wall.str().find("nan")!=std::string::npos,"actual nonfinite diagnostic serialized");
    std::ostringstream failed;failed.setstate(std::ios::badbit);
    check(!io::writeAcceptedCoupledSummary(failed,supplied,model,error),"actual stream fault retained");
}
'''
    run(tmp_path,source)


def test_supplied_startup_keeps_actual_geometry_and_aux(tmp_path):
    run(tmp_path,r'''
#define main prior_geometry_main
#include "../devtools/multirate/test_sweep_constraints.cpp"
#undef main
#include "gpu/StateValidation.H"
#include <cassert>
int main(){
 HostState h;h.solidMesh=twoHexahedra(0);h.solid.resize(2);h.surface.area={2};h.surface.normal={{0,0,1}};
 h.surface.edgeOwner={0};h.time=std::numeric_limits<Real>::quiet_NaN();
 ModelConfig m;m.physics.enableGas=false;
 std::string error;const auto points=h.solidMesh.points;
 assert(validateRuntimeState(m,h,error));assert(h.solidMesh.points.size()==points.size());
 assert(h.solidMesh.volumes.size()==2&&h.solidMesh.volumes[0]>0);
 assert(h.filmAux.size()==1&&h.filmAux[0].area==2&&h.filmAux[0].normal.z==1);
 assert(h.surface.oldArea==h.surface.area&&h.surface.sweptEdgeArea.size()==1&&h.surface.sweptEdgeArea[0]==0);
 assert(h.surface.meshVelocity.size()==1);
}
''',True)


def test_supplied_surface_identity_preserves_first_insert(tmp_path):
    run(tmp_path,r'''
#include "materials/SurfacePacketAddressing.H"
#include "configuration/CouplingEntry.H"
#include <cassert>
int main(){using namespace chmt;
 HostState h;h.surface.area={2,3};h.surface.persistentId={7,7};h.surface.gasFace={0,1};h.surface.solidCell={0,1};
 auto endpoint=h;endpoint.gasMesh.topologyHash=99;endpoint.surface.persistentId={8,9};
 SurfacePacketAddressing map;std::string error;assert(map.build(h,endpoint,error));
 ExchangePacket packet;packet.face=7;packet.solidCell=1;std::size_t index=77;
 assert(map.gasSolidIndex(endpoint,packet,index,error)&&index==0);
 packet.face=900;assert(!map.gasSolidIndex(h,packet,index,error));
 assert(validateCouplingEntry("foreign supplied mode",h,error));
}
''')


def test_finalizer_no_solver_preflight_or_control_mutation(tmp_path):
    path=(CASES_ROOT / 'example/tools/production/bentSRM_coldWall/assets/initialization/finalise_thermal_restart.py')
    spec=importlib.util.spec_from_file_location('actual_finalizer',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    case=tmp_path/'case';(case/'1.5').mkdir(parents=True);(case/'system').mkdir();(case/'system/controlDict').write_text('startFrom latestTime; endTime 7;\n')
    (case/'1.5'/'physical.bin').write_bytes(b'actual supplied payload')
    (case/'assets/initialization').mkdir(parents=True)
    module.CASE=case
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args[0],1,stdout='no obsolete certificate refusal')
    with patch.object(sys,'argv',['finalizer','--time','1.5']),patch('subprocess.run',forbidden):
        module.main()
    assert calls==[]
    assert (case/'system/controlDict').read_text()=='startFrom latestTime; endTime 7;\n'
    text=(case/'1.5/thermalExchangeState').read_text()
    assert 'completedSimulationTimeS 1.5;' in text and 'formatVersion 1;' in text
    assert (case/'assets/initialization/formal_1p5_checkpoint/physical.bin').read_bytes()==b'actual supplied payload'


def _body(text, signature):
    start=text.index(signature);brace=text.index('{',start);depth=1;end=brace+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return text[start:end]


@pytest.mark.parametrize('revision,expected_queries,expected_hashes', [
    ('historical',1,3),('working',0,0)])
def test_actual_mirror_staging_removes_proof_reads(tmp_path,revision,expected_queries,expected_hashes):
    path='applications/CHT/gpu/GpuResidentStrict.H'
    source=((ROOT/path).read_text() if revision=='working' else
            ((VALIDATION_ROOT / 'solvers/CHMT/tests/fixtures/thermal_restart_mirror_stage_d611.H')).read_text())
    method=_body(source,'    GpuThermalRestartMirrorMetadata stageThermalRestartMirrors')
    code=r'''
#include <string>
#include <cassert>
using label=int;using word=std::string;struct Time{};
struct fileName:std::string{using std::string::string;std::string name()const{return *this;}};
struct GpuThermalRestartMirrorMetadata {std::string particleFile,epsGPrevFile,sourceResidualFile;int particleCount=0,epsGPrevCount=0,sourceResidualCount=0;std::string particleSha1,epsGPrevSha1,sourceResidualSha1;};
int writes=0,queries=0,hashes=0;bool failWrite=false;
int ugkwpGpuResidentStrictDownloadSourceResidualMass(void*,int* n,int,void*,void*){++queries;*n=2;return 0;}
struct Resident{void* handle_=nullptr;int hostMirrorCells_=3;
 void requireInitialised(const char*)const{}void check(int,const char*)const{}
 int writeParticleRestartMirror(const Time&,bool)const{++writes;if(failWrite)throw 7;return 19;}
 fileName particleRestartFile(const Time&)const{return "particle";}fileName epsGPrevRestartFile(const Time&)const{return "eps";}fileName sourceResidualRestartFile(const Time&)const{return "source";}
 word sha1File(const fileName&)const{++hashes;return "actual-probe-hash";}
METHOD
};
int main(){Resident r;Time t;auto m=r.stageThermalRestartMirrors(t);assert(m.particleCount==19&&writes==1&&queries==EXPECTED_QUERIES&&hashes==EXPECTED_HASHES);
 failWrite=true;bool failed=false;try{r.stageThermalRestartMirrors(t);}catch(int){failed=true;}assert(failed&&writes==2&&queries==EXPECTED_QUERIES&&hashes==EXPECTED_HASHES);}
'''.replace('METHOD',method).replace('EXPECTED_QUERIES',str(expected_queries)).replace('EXPECTED_HASHES',str(expected_hashes))
    run(tmp_path,code)


def test_actual_orchestration_retains_only_requested_stage_geometry(tmp_path):
    source=((VALIDATION_ROOT / 'solvers/CHMT/tests/test_multirate_orchestration.cpp')).read_text().split('int main(){')[0]
    old=_body(source,'bool gasWindowStageGeometry(')
    new='''int geometryCalls=0;bool geometryFault=false;
bool gasWindowStageGeometry(const Backend&,std::array<HostStageGeometry,2>& gas,std::array<HostStageGeometry,2>& solid,std::string& error){++geometryCalls;if(geometryFault){error="actual geometry transfer failure";return false;}gas[0].newVolume={3,4};solid[1].newPoints={{7,8,9}};error.clear();return true;}'''
    assert old in source
    source=source.replace(old,new)+r'''
int main(){using namespace chmt;
 ModelConfig model;CpuMaterialControls material;CpuMaterialDriver cpu(model,nullptr);
 CouplingControls controls;controls.interval=1;controls.minimumInterval=1e-6;controls.gasMaxDt=.01;controls.maxWindowRetries=2;
 HostState initial;GasQ q;q.mass=1;q.species[0]=1;q.energy=3;initial.gas={q};initial.gasMesh.points={{0,0,0}};
 for(bool retain:{false,true}){script=Script{};geometryCalls=0;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;std::string error;
  assert(advanceCoupledWindow(backend,cpu,model,controls,material,.2,retain,accepted,report,error));
  assert(accepted.time==.2&&accepted.gas[0].energy==3);
  assert(geometryCalls==(retain?int(report.acceptedGasMicrosteps):0));assert(report.geometry.size()==(retain?report.acceptedGasMicrosteps:0));
  for(const auto& stage:report.geometry){assert(stage.gas[0].newVolume==std::vector<Real>({3,4}));assert(stage.solid[1].newPoints[0].x==7&&stage.solid[1].newPoints[0].y==8&&stage.solid[1].newPoints[0].z==9);}
  std::cout<<retain<<" "<<report.acceptedGasMicrosteps<<" "<<geometryCalls<<" "<<report.geometry.size()<<"\n";
 }
 script=Script{};geometryFault=true;Backend backend;backend.accepted=initial;HostState accepted=initial;WindowEvolutionReport report;std::string error;
 assert(!advanceCoupledWindow(backend,cpu,model,controls,material,.2,true,accepted,report,error));assert(error.find("actual geometry transfer failure")!=std::string::npos);
}
'''
    src=tmp_path/'trace.cpp';src.write_text(source);exe=tmp_path/'trace'
    subprocess.run(['g++','-std=c++17','-O2','-Wall','-Wextra','-Werror','-pedantic',
                    '-I'+str(APP), '-I'+str(VALIDATION_ROOT / 'solvers/CHMT'),'-I'+str(ROOT/'common'),str(src),str(APP/'io/MultirateEvolution.C'),'-o',str(exe)],check=True)
    result=subprocess.run([str(exe)],check=True,capture_output=True,text=True)
    (tmp_path/'trace-counts.txt').write_text(result.stdout)
    assert result.stdout=='0 20 0 0\n1 20 20 20\n'


@pytest.mark.parametrize('fault',['write','copy'])
def test_finalizer_actual_filesystem_errors_propagate(tmp_path,fault):
    path=(CASES_ROOT / 'example/tools/production/bentSRM_coldWall/assets/initialization/finalise_thermal_restart.py')
    spec=importlib.util.spec_from_file_location('failing_finalizer',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    case=tmp_path/'case';(case/'1.5').mkdir(parents=True);(case/'assets/initialization').mkdir(parents=True)
    module.CASE=case
    target='pathlib.Path.write_text' if fault=='write' else 'shutil.copytree'
    with patch.object(sys,'argv',['finalizer','--time','1.5']),patch(target,side_effect=OSError('actual injected filesystem error')):
        with pytest.raises(OSError,match='actual injected filesystem error'):
            module.main()
