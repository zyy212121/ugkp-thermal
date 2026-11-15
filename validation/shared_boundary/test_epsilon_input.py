"""Host-only epsilonS Function1 compatibility and empty-patch input contracts."""
from pathlib import Path
import os, subprocess
import pytest
R=Path(__file__).resolve().parents[2]

@pytest.fixture(scope="module",params=["","precision/fp32/"])
def parser(tmp_path_factory,request):
    out=tmp_path_factory.mktemp("epsilon-input")
    source=out/"input.cpp"
    source.write_text(r"""
#include "gasBoundary/FoamInput.H"
#include <cassert>
#include <limits>
int main(int argc,char** argv) {
    using namespace Foam;
    using namespace ugkp::boundary;
    using namespace ugkp::boundary::foam;
    assert(argc==2);
    assert(epsilonStartupSnapshot(std::numeric_limits<scalar>::quiet_NaN(),0)==0);
    assert(epsilonStartupSnapshot(std::numeric_limits<scalar>::quiet_NaN(),1)==0);
    assert(epsilonStartupSnapshot(.25,0)==0);
    assert(epsilonStartupSnapshot(.25,1)==.25);
    Definition check;check.faces.resize(1);check.faces[0].value[SolidFraction]=.25;
    check.functions.push_back(Function{}); // unrelated gas program survives rollback
    Function bad;bad.constant=std::numeric_limits<decltype(bad.constant)>::infinity();
    check.functions.push_back(bad);check.faces[0].function[SolidFraction]=1;
    assert(!retainFiniteEpsilonProgram(check,1,0,0,1));
    assert(check.functions.size()==1&&check.terms.empty());
    assert(check.faces[0].function[SolidFraction]==-1&&check.faces[0].value[SolidFraction]==.25);
    Function table;table.kind=FunctionKind::Table;table.count=1;
    check.functions.push_back(table);
    Term term;
TERM_NONFINITE
    check.terms.push_back(term);check.faces[0].function[SolidFraction]=1;
    assert(!retainFiniteEpsilonProgram(check,1,0,0,1));
    validate(check);
    IStringStream stream(argv[1]);dictionary d(stream);
    if(!residentEpsilonFunctionSupported(d,"uniformValue")){Info<<"SNAPSHOT"<<nl;return 0;}
    Definition result;result.faces.resize(1);
    try {
        function<scalar>(result,d,"uniformValue",SolidFraction,result.faces,0,1);
        if(!retainFiniteEpsilonProgram(result,0,0,0,1)){Info<<"SNAPSHOT"<<nl;return 0;}
        validate(result);
    } catch(const std::runtime_error& e) {Info<<"PROGRAM_ERROR "<<e.what()<<nl;return 2;}
    assert(result.faces[0].function[SolidFraction]==0);
    Info<<"PROGRAM"<<nl;return 0;
}
""".replace("TERM_NONFINITE",
        "    term.value=std::numeric_limits<decltype(term.value)>::infinity();" if request.param else
        "    term.y=std::numeric_limits<double>::infinity();"))
    foam=Path(os.environ.get("WM_PROJECT_DIR","/opt/openfoam10"))
    lib=Path(os.environ.get("FOAM_LIBBIN",str(foam/"platforms/linux64GccDPInt32Opt/lib")))
    exe=out/"input"
    command=["g++","-std=c++17","-DWM_DP","-DWM_LABEL_SIZE=32","-DNoRepository","-I"+str(R/(request.param+"common"))]
    command+=["-I"+str(foam/"src"/sub/"lnInclude") for sub in ["OpenFOAM","finiteVolume","meshTools","OSspecific/POSIX"]]
    command+=[str(source),"-L"+str(lib),"-Wl,-rpath,"+str(lib),"-lfiniteVolume","-lmeshTools","-lOpenFOAM",
              "-L"+str(lib/"dummy"),"-Wl,-rpath,"+str(lib/"dummy"),"-lPstream","-o",str(exe)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    return exe

@pytest.mark.parametrize("config",[
    "uniformValue constant .1;",
    "uniformValue table ((0 .1)(1 .2));",
    "uniformValue {type table; values ((0 .1)(1 .2)); interpolationScheme linear; outOfBounds repeat;}",
])
def test_supported_program_is_not_frozen(parser,config):
    result=subprocess.run([parser,config],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert "PROGRAM" in result.stdout

@pytest.mark.parametrize("config",[
    "uniformValue sine;",
    "uniformValue {type scale;}",
    "uniformValue {type table; interpolationScheme spline;}",
    "uniformValue {type table; outOfBounds error;}",
])
def test_old_unsupported_program_retains_snapshot_without_rejection(parser,config):
    result=subprocess.run([parser,config],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert "SNAPSHOT" in result.stdout

def test_supported_empty_table_does_not_silently_become_snapshot(parser):
    result=subprocess.run([parser,"uniformValue table ();"],capture_output=True,text=True)
    assert result.returncode==2
    assert "boundary table is empty" in result.stdout

def test_empty_patch_is_skipped_before_epsilon_storage_access():
    for prefix in ["","precision/fp32/"]:
        source=(R/(prefix+"common/gasBoundary/FoamInput.H")).read_text()
        body=source.split('if(mesh.foundObject<volScalarField>("epsilonS"))',1)[1]
        assert body.index("if(isA<emptyPolyPatch>(pp))continue;")<body.index("const auto& field=eps.boundaryField()[patch]")
        assert "retaining the legacy startup snapshot" in body
