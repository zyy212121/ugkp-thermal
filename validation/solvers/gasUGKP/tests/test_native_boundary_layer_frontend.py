"""Real OpenFOAM dictionary/geometry/CSV adapter checks, with a recording C ABI."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

import csv
import math
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import pytest

ROOT = SOURCE_ROOT
PROBE = r'''
#include "fvCFD.H"
#include "physicoChemicalConstants.H"
#include "gasTransport/GasNumericsIO.H"
#include "gpu/SharedGasModelInput.H"
#include "gpu/BoundaryLayerInput.H"
#include "GpuSchedulingConfiguration.H"
using namespace Foam;
inline bool finiteScalar(scalar x){return std::isfinite(x);}
extern "C" const char* ugkwpGpuResidentStrictLastError(){return "recording backend";}
extern "C" int ugkwpGpuResidentStrictConfigureBoundaryLayerV1(void*,const ugkwpGpu::BoundaryLayerConfigV1* c,
 const int* faces,const int* qo,const int* mo,const int* mc,const double* g,const double* qd,
 const double* qw,const double* mw,const double* js){
 if(c->quadratureCount!=0||qo[0]!=0||qo[1]!=0||qd||qw)return 1;
 double v=g[5],m=g[6];
 Info<<"WALL_UPLOAD model="<<c->model<<" walls="<<c->wallCount<<" donor="<<mc[0]
 <<" volume="<<v<<" moment="<<m<<" matching="<<g[4]<<" owner="<<g[3]
 <<" quadrature="<<qo[c->wallCount]<<" flux="<<js[0]<<","<<js[c->wallCount]<<" nodes="<<c->nodes<<" slots="<<c->workspaceSlots<<nl;
 return !(v>0 && m>0 && g[4]>g[3] && mc[0]!=0 && mw[0]>0 && mo[1]>0 && faces[0]>=0);
}
extern "C" int ugkwpGpuResidentStrictDownloadBoundaryLayerV1(void*,std::uint32_t walls,std::uint32_t species,double* out){
 for(unsigned f=0;f<walls;++f)for(unsigned col=0;col<25+2*species;++col)
 out[f*(25+2*species)+col]=col==15 ? 0.375 : 100*f+col;
 return 0;
}
int main(int argc,char**argv){
 #include "setRootCase.H"
 #include "createTime.H"
 #include "createMesh.H"
 #include "readGpuGasConfiguration.H"
 volScalarField T(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
 boundaryLayerInput.prepare(runTime,mesh,T);
 boundaryLayerInput.configureResident(nullptr);
 runTime.setTime(0.5,1);runTime.writeNow();
 boundaryLayerInput.writeDiagnostics(runTime,nullptr);
 Info<<"WALL_READY family="<<sstWallTreatment<<" turbulence="<<gasTurbulenceModel<<nl;
}
'''

@pytest.fixture(scope="module")
def native_wall_probe(tmp_path_factory):
    if not os.environ.get("WM_PROJECT_DIR"):
        pytest.skip("OpenFOAM environment required")
    path=tmp_path_factory.mktemp("native_wall_frontend")
    (path/"Make").mkdir()
    (path/"probe.C").write_text(PROBE)
    (path/"Make/files").write_text(f"probe.C\nEXE = {path}/probe\n")
    (path/"Make/options").write_text(f"EXE_INC = -DUGKWP_GAS_SPECIES=2 -I{ROOT}/applications/gasUGKP -I{ROOT}/applications/gasUGKP/gpu -I{ROOT}/common -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\nEXE_LIBS = -lfiniteVolume -lmeshTools -lOpenFOAM\n")
    result=subprocess.run(["wmake"],cwd=path,capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    return path/"probe"


def make_case(path,model="finiteRate",sst=False,options="",flux=None):
    spec=importlib.util.spec_from_file_location("wall_frontend_case",(CASES_ROOT / 'test/tools/consistency/gasUGKP/mixtureTransport/make_case.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.create_case(path,cells=8)
    mesh=path/"system/blockMeshDict"
    text=mesh.read_text()
    text=re.sub(r"left \{.*?faces \(\(0 4 7 3\)\); \}","left { type wall; faces ((0 4 7 3)); }",text)
    text=re.sub(r"right \{.*?faces \(\(1 2 6 5\)\); \}","right { type patch; faces ((1 2 6 5)); }",text)
    mesh.write_text(text)
    for field in (path/"0").iterdir():
        value="350" if field.name=="T" else "0"
        text=field.read_text().replace("left { type cyclic; }",f"left {{ type fixedValue; value uniform {value}; }}")
        field.write_text(text.replace("right { type cyclic; }","right { type zeroGradient; }"))
    fluid=path/"constant/fluidProperties"
    controls=f"wallTreatment boundaryLayer; boundaryLayer {{ model {model}; {options} }}"
    turbulence=(f"simulationType RAS; RAS {{ model kOmegaSST; turbulence true; kOmegaSSTCoeffs {{ {controls} }} }}" if sst else f"simulationType laminar; {controls}")
    fluid.write_text(fluid.read_text().replace("mu 0;","mu 1e-5;").replace("simulationType laminar;",turbulence))
    schemes=path/"system/fvSchemes";schemes.write_text(schemes.read_text().replace("fluxScheme HLLC;","fluxScheme rusanovTadmor;"))
    if flux is not None:
        (path/"constant/gasWallProperties").write_text(module.header("gasWallProperties")+flux)
    result=subprocess.run(["blockMesh","-case",str(path)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr


def run_probe(binary,path):
    return subprocess.run([str(binary),"-case",str(path)],capture_output=True,text=True)


@pytest.mark.parametrize("model,sst",[("finiteRate",False),("reactingSst",True),("constantTransport",False),("constantTransport",True)])
def test_native_wall_configuration_geometry_upload_and_csv(native_wall_probe,tmp_path,model,sst):
    make_case(tmp_path,model,sst,options="nodes 16; workspaceSlots 2;",flux="patches { left { speciesFlux (0.01 0.02); } }")
    result=run_probe(native_wall_probe,tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    assert "WALL_READY family=2" in result.stdout
    assert "quadrature=0" in result.stdout
    assert "flux=0.01,0.02 nodes=16 slots=2" in result.stdout
    rows=list(csv.DictReader((tmp_path/"0.5/uniform/gasBoundaryLayer.csv").open()))
    assert len(rows)==1
    assert float(rows[0]["stageTime"])==0.375
    assert float(rows[0]["speciesFlux_A"])==25
    assert float(rows[0]["reactionIntegral_B"])==28
    assert not any("mismatch" in key.lower() for key in rows[0])
    assert not any(name.startswith("budget") for name in rows[0])


def test_native_wall_omitted_flux_defaults_to_zero(native_wall_probe,tmp_path):
    make_case(tmp_path)
    result=run_probe(native_wall_probe,tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    assert "flux=0,0" in result.stdout
    assert "slots=0" in result.stdout


@pytest.mark.parametrize("model,sst,options,flux,error",[
    ("bogus",False,"",None,"model"),
    ("finiteRate",False,"", "patches { missing { speciesFlux (0 0); } }","patch"),
    ("finiteRate",False,"", "patches { left { speciesFlux (0); } }","speciesFlux"),
])
def test_native_wall_rejects_invalid_configuration(native_wall_probe,tmp_path,model,sst,options,flux,error):
    make_case(tmp_path,model,sst,options,flux)
    result=run_probe(native_wall_probe,tmp_path)
    assert result.returncode!=0
    assert error in result.stdout+result.stderr


@pytest.mark.parametrize("mutation,error",[("legacy","mixture"),("particles","gas-only")])
def test_native_wall_requires_shared_gas_only_fixed_temperature(native_wall_probe,tmp_path,mutation,error):
    make_case(tmp_path)
    if mutation=="legacy":
        (tmp_path/"constant/gasModelProperties").unlink()
    elif mutation=="particles":
        p=tmp_path/"constant/schedulingProperties";p.write_text(p.read_text().replace("gpuResidentPureGasOnly true;","gpuResidentPureGasOnly false;"))
    else:
        p=tmp_path/"0/T";p.write_text(p.read_text().replace("type fixedValue; value uniform 350;","type zeroGradient;"))
    result=run_probe(native_wall_probe,tmp_path)
    assert result.returncode!=0
    assert error in result.stdout+result.stderr


def test_native_disabled_ras_uses_laminar_wall_controls(native_wall_probe,tmp_path):
    make_case(tmp_path)
    fluid=tmp_path/"constant/fluidProperties"
    fluid.write_text(fluid.read_text().replace("simulationType laminar;","simulationType RAS; RAS { turbulence false; }"))
    result=run_probe(native_wall_probe,tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    assert "WALL_READY family=2 turbulence=0" in result.stdout


def test_native_wall_accepts_high_resolution_nodes_and_auto_scratch(native_wall_probe,tmp_path):
    make_case(tmp_path,options="nodes 96; workspaceSlots 0;")
    result=run_probe(native_wall_probe,tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    assert "nodes=96 slots=0" in result.stdout


def test_native_sst_has_no_unused_quadrature_at_any_profile_resolution(native_wall_probe,tmp_path):
    counts=[]
    for nodes in (16,32):
        case=tmp_path/str(nodes);case.mkdir()
        make_case(case,"reactingSst",True,options=f"nodes {nodes};")
        result=run_probe(native_wall_probe,case)
        assert result.returncode==0,result.stdout+result.stderr
        counts.append(int(re.search(r"quadrature=(\d+)",result.stdout).group(1)))
    assert counts==[0,0], "production created unused auxiliary owner-source points"


def test_native_sst_legacy_audit_option_does_not_add_output_columns(native_wall_probe,tmp_path):
    make_case(tmp_path,"reactingSst",True,options="budgetAudit true;")
    result=run_probe(native_wall_probe,tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    rows=list(csv.DictReader((tmp_path/"0.5/uniform/gasBoundaryLayer.csv").open()))
    assert not any(name.startswith("budget") for name in rows[0])
    assert len(rows[0])==20
    assert float(rows[0]["stageTime"])==.375


@pytest.mark.parametrize("options,flux",[
    ("nodez 24;",None),
    ("rtol 1e-8; relativeTolerance 1e-7;",None),
    ("nodes 129;",None),
    ("workspaceSlots 4097;",None),
    ("budgetAudit true;",None),
    ("relativeTolerance -1;",None),
    ("", "patches { right { speciesFlux (0 0); } }"),
    ("", "schemaVersion 77; patches { left { speciesFlux (-1 0 123); } }"),
])
def test_native_wall_declarations_do_not_refuse_existing_values(native_wall_probe,tmp_path,options,flux):
    make_case(tmp_path,options=options,flux=flux)
    p=tmp_path/"0/T";p.write_text(p.read_text().replace("type fixedValue; value uniform 350;","type zeroGradient;"))
    result=run_probe(native_wall_probe,tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    assert "WALL_READY family=2" in result.stdout
