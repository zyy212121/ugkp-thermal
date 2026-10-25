"""Run actual OpenFOAM wall input directly, independently of later frontend gates."""
import csv
import os
import subprocess
import pytest
from test_native_boundary_layer_frontend import ROOT, make_case, run_probe

PROBE = r'''
#include "fvCFD.H"
#include "gpu/BoundaryLayerInput.H"
using namespace Foam;
extern "C" const char* ugkwpGpuResidentStrictLastError(){return "recording backend";}
extern "C" int ugkwpGpuResidentStrictConfigureBoundaryLayerV1(void*,const ugkwpGpu::BoundaryLayerConfigV1* c,
 const int* f,const int* qo,const int* mo,const int* mc,const double* g,const double* qd,
 const double* qw,const double* mw,const double* js){
 Info<<"UPLOAD walls="<<c->wallCount<<" quadrature="<<c->quadratureCount
 <<" nodes="<<c->nodes<<" rtol="<<c->relativeTolerance<<" iterations="<<c->maxIterations<<nl;
 if(!c->wallCount)return qo[0]!=0||mo[0]!=0;
 if(c->wallCount!=1||c->quadratureCount||qo[0]||qo[1]||qd||qw||mo[0]||mo[1]!=1||mc[0]!=1||mw[0]!=1)return 1;
 Info<<"GEOMETRY face="<<f[0]<<" owner="<<g[3]<<" matching="<<g[4]<<" volume="<<g[5]
 <<" flux="<<js[0]<<","<<js[1]<<nl;
 return 0;
}
extern "C" int ugkwpGpuResidentStrictDownloadBoundaryLayerV1(void*,std::uint32_t walls,std::uint32_t species,double* out){
 for(unsigned f=0;f<walls;++f)for(unsigned c=0;c<25+2*species;++c)out[f*(25+2*species)+c]=c==16?0:c>=17&&c<25?std::numeric_limits<double>::quiet_NaN():c;
 return 0;
}
int main(int argc,char**argv){
 #include "setRootCase.H"
 #include "createTime.H"
 #include "createMesh.H"
 IOdictionary fluid(IOobject("fluidProperties",runTime.constant(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE));
 const auto& turbulence=fluid.subDict("turbulence");
 ugkwp::GasModelConfiguration gas;gas.mode=ugkwp::GasMode::MixtureFrozen;gas.speciesNames={"A","B"};
 const label model=turbulence.lookupOrDefault<word>("simulationType","laminar")=="RAS"?3:0;
 BoundaryLayerInput input(turbulence,gas,model,false,.9);
 volScalarField temperature(IOobject("T",runTime.timeName(),mesh,IOobject::MUST_READ,IOobject::NO_WRITE),mesh);
 input.prepare(runTime,mesh,temperature);input.configureResident(nullptr);
 pointField changed(mesh.points());for(auto&p:changed)p.x()*=2;mesh.movePoints(changed);
 input.prepare(runTime,mesh,temperature);input.configureResident(nullptr);
 // No writeNow: writeTime is not permission to serialize actual diagnostics.
 runTime.setTime(.375,1);input.writeDiagnostics(runTime,nullptr);
}
'''

@pytest.fixture(scope='module')
def preparation_probe(tmp_path_factory):
    if not os.environ.get('WM_PROJECT_DIR'):
        pytest.skip('OpenFOAM environment required')
    p=tmp_path_factory.mktemp('wall_preparation');(p/'Make').mkdir()
    (p/'probe.C').write_text(PROBE)
    (p/'Make/files').write_text(f'probe.C\nEXE = {p}/probe\n')
    (p/'Make/options').write_text(f'EXE_INC = -DUGKWP_GAS_SPECIES=2 -I{ROOT}/applications/gasUGKP -I{ROOT}/applications/gasUGKP/gpu -I{ROOT}/common -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\nEXE_LIBS = -lfiniteVolume -lmeshTools -lOpenFOAM\n')
    built=subprocess.run(['wmake'],cwd=p,capture_output=True,text=True)
    assert built.returncode==0,built.stdout+built.stderr
    return p/'probe'

@pytest.mark.parametrize('options',[
    'rtol -2; relativeTolerance -1; nodez 12; workspaceSlots 4097; maxIterations 0; budgetAudit true;',
    'nodes 2; stretch .5; geometryTolerance -1; quadratureRefinement 20; maxRayCells 0;',
])
def test_repeat_moving_prepare_replaces_geometry_and_reads_consumed_flux(preparation_probe,tmp_path,options):
    make_case(tmp_path,options=options,flux='schemaVersion 99; patches { left { speciesFlux (-1 .02 999); } right { speciesFlux (); } }')
    p=tmp_path/'0/T';p.write_text(p.read_text().replace('type fixedValue; value uniform 350;','type zeroGradient;'))
    result=run_probe(preparation_probe,tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    import re
    rows=re.findall(r'GEOMETRY face=\d+ owner=([\deE.+-]+) matching=([\deE.+-]+) volume=([\deE.+-]+) flux=(-1,0.02)',result.stdout)
    assert len(rows)==2,result.stdout
    for i in range(3):assert float(rows[1][i])==pytest.approx(2*float(rows[0][i]))
    output=list(csv.DictReader((tmp_path/'0.375/uniform/gasBoundaryLayer.csv').open()))
    assert len(output)==1

@pytest.mark.parametrize('flux,error',[
    ('patches { left { speciesFlux (.1); } }','speciesFlux missing species "B" at index 1'),
    ('patches { absent { speciesFlux (.1 .2); } }','absent'),
])
def test_missing_consumed_species_or_patch_is_explicit(preparation_probe,tmp_path,flux,error):
    make_case(tmp_path,flux=flux)
    result=run_probe(preparation_probe,tmp_path)
    assert result.returncode!=0
    assert error in result.stdout+result.stderr


def test_empty_wall_set_is_real_no_work(preparation_probe,tmp_path):
    make_case(tmp_path)
    p=tmp_path/'constant/polyMesh/boundary';p.write_text(p.read_text().replace('type            wall;','type            patch;'))
    result=run_probe(preparation_probe,tmp_path)
    assert result.returncode==0,result.stdout+result.stderr
    assert result.stdout.count('UPLOAD walls=0')==2
    assert list(csv.DictReader((tmp_path/'0.375/uniform/gasBoundaryLayer.csv').open()))==[]
