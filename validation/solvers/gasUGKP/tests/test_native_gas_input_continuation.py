"""Execute real OpenFOAM input and resident wrapper code with a recording C ABI.

The ABI test double only records transfers or fills deliberately selected output
values. These tests do not claim CUDA execution or validate backend evolution.
"""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

import os
import re
import struct
import subprocess
from pathlib import Path
import pytest
from test_native_gas_model_frontend import make_case as make_mesh_case

ROOT = SOURCE_ROOT
PREFIX = r'''
#include "fvCFD.H"
#include "physicoChemicalConstants.H"
#include "uniformDimensionedFields.H"
#include "gasTransport/GasNumericsIO.H"
#include "gpu/GpuResidentStrict.H"
#include "gpu/GpuDragModel.H"
#include <iomanip>
using namespace Foam;
static int cells=0,faces=0,particleUploads=0,speciesUploads=0;
static double firstMass=0,firstX=0,firstResidual=0,firstEps=0;
inline bool finiteScalar(scalar x){return std::isfinite(x);}
class ProbeResident : public GpuResidentStrictBase {
public:
 void attach(const fvMesh& m){handle_=this;hostMirrorCells_=m.nCells();hostMirrorFaces_=m.nFaces();residentMesh_=&m;cells=m.nCells();faces=m.nFaces();legacyRestartParcelMass_=2;}
 using GpuResidentStrictBase::initialiseBase;
 using GpuResidentStrictBase::readParticleRestartMirrorIfPresent;
 using GpuResidentStrictBase::readEpsGPrevRestartMirrorIfPresent;
 using GpuResidentStrictBase::readSourceResidualRestartMirrorIfPresent;
 using GpuResidentStrictBase::uploadGasBoundaryFields;
 using GpuResidentStrictBase::uploadMesh;
 using GpuResidentStrictBase::downloadGasBoundaryToHostMirror;
 using GpuResidentStrictBase::ensureInternalSize;
 using GpuResidentStrictBase::readBinaryParticleBlock;
};
'''
BODIES = {
    'ugkwpGpuResidentStrictLastError': 'return "injected backend transfer failure";',
    'ugkwpGpuResidentStrictRelease': '',
    'ugkwpGpuResidentStrictCreate': '*handle=reinterpret_cast<void*>(1);cells=nCells;faces=nFaces;std::cout<<"CREATE "<<particleCapacity<<" "<<maxFaceWalkHops<<" "<<collisionalRestitution<<" "<<pressureKickFraction<<" "<<packingFraction<<" "<<packingProjectionIterations<<"\\n";return 0;',
    'ugkwpGpuResidentStrictUploadParticleRestartMirror': '++particleUploads;if(nParticles){firstMass=pm[0];firstX=px[0];}std::cout<<"PARTICLES "<<nParticles<<" "<<firstX<<" "<<firstMass<<"\\n";return 0;',
    'ugkwpGpuResidentStrictUploadEpsGPrev': 'firstEps=epsGPrev[0];std::cout<<"EPS "<<firstEps<<"\\n";return 0;',
    'ugkwpGpuResidentStrictUploadSourceResidualMass': 'if(nSources)firstResidual=residualMass[0];std::cout<<"RESIDUAL "<<nSources<<" "<<firstResidual<<"\\n";return 0;',
    'ugkwpGpuResidentStrictDownloadNut': 'if(std::getenv("FAIL_TRANSFER"))return 7;for(int i=0;i<cells;++i)nut[i]=-10-i;return 0;',
    'ugkwpGpuResidentStrictUploadSpeciesV1': '++speciesUploads;if(speciesUploads>1)for(unsigned s=0;s<identity->speciesCount;++s)for(int c=0;c<cells;++c)if(speciesDensity[s*cells+c]!=-2.-s-c)return 5;std::cout<<"SPECIES_UPLOAD "<<speciesDensity[0]<<" "<<speciesDensity[cells]<<"\\n";return 0;',
    'ugkwpGpuResidentStrictDownloadSpeciesV1': 'for(unsigned s=0;s<identity->speciesCount;++s)for(int c=0;c<cells;++c)speciesDensity[s*cells+c]=-2.0-s-c;return 0;',
    'ugkwpGpuResidentStrictConfigureScheduledInlet': 'std::cout<<"SCHEDULE "<<nFaces<<" "<<inletTemperature<<"\\n";return 0;',
    'ugkwpGpuResidentStrictUploadGasBoundaryFields': 'std::cout<<"BOUNDARY "<<gasBoundaryRho[8]<<" "<<gasBoundaryP[8]<<" "<<gasBoundaryT[8]<<" "<<gasBoundaryRhoFix[8]<<" "<<gasBoundaryUFix[8]<<" "<<gasBoundaryPFix[8]<<" "<<gasBoundaryTFix[8]<<"\\n";return 0;',
    'ugkwpGpuResidentStrictDownloadEpsGPrev': 'for(int i=0;i<cells;++i)epsGPrev[i]=.125+i*.03125;return 0;',
    'ugkwpGpuResidentStrictDownloadSourceResidualMass': '*nSources=0;return 0;',
}
BODIES['ugkwpGpuResidentStrictUploadParticleRestartMirror'] += 'if(std::getenv("VERIFY_PARTICLE")&&nParticles){if(nParticles!=1||px[0]!=.125||py[0]!=2||pz[0]!=3||pux[0]!=4||puy[0]!=5||puz[0]!=6||pT[0]!=300||pTheta[0]!=8||pd[0]!=.001||pm[0]!=(std::getenv("LEGACY_PARTICLE")?2:3)||pCellId[0]!=0||pStatus[0]!=1||pRng[0]!=11||pOrigId[0]!=12)return 8;}return 0;'

BODIES['ugkwpGpuResidentStrictUploadParticleRestartMirror']=BODIES['ugkwpGpuResidentStrictUploadParticleRestartMirror'].replace('return 0;if(std::getenv("VERIFY_PARTICLE")','if(std::getenv("VERIFY_PARTICLE")')
BODIES['ugkwpGpuResidentStrictDownloadSst']='for(int i=0;i<cells;++i){k[i]=-2-i;omega[i]=-3-i;nut[i]=-4-i;}return 0;'
BODIES['ugkwpGpuResidentStrictDownloadParticleRestartMirror']='*nParticles=1;if(!maxParticles)return 0;px[0]=.125;py[0]=2;pz[0]=3;pux[0]=4;puy[0]=5;puz[0]=6;pT[0]=300;pTheta[0]=8;pd[0]=.001;pm[0]=3;pCellId[0]=0;pStatus[0]=1;pRng[0]=11;pOrigId[0]=12;return 0;'
BODIES['ugkwpGpuResidentStrictDownloadGasBoundaryFields']='for(int i=0;i<faces;++i){rho[i]=2;Ux[i]=3;Uy[i]=4;Uz[i]=5;p[i]=6;Tgas[i]=7;}return 0;'
FIELDS=['rho','rhoUx','rhoUy','rhoUz','rhoE','Ux','Uy','Uz','p','Tgas','epsS','rhoUsx','rhoUsy','rhoUsz','rhoEs','rhoDs','rhoHp','Usx','Usy','Usz','theta','Tp','dMeanCell']
BODIES['ugkwpGpuResidentStrictDownloadFields']='for(int i=0;i<cells;++i){'+''.join(f'{name}[i]={100+j}+i;' for j,name in enumerate(FIELDS))+'rhoE[i]=1e35;}return 0;'

MAIN = r'''
int main(int argc,char**argv){
 #include "setRootCase.H"
 #include "createTime.H"
 #include "createMesh.H"
 std::cout<<std::setprecision(17);
 #include "createFields.H"
 ProbeResident resident;resident.attach(mesh);
 std::string action=std::getenv("T07_ACTION")?std::getenv("T07_ACTION"):"fields";
 if(action=="sst_mirror") {runTime.setTime(.00001,1);resident.downloadSstToHostMirror(runTime,k,omega,nut);forAll(k,i)if(k[i]!=-2-i||omega[i]!=-3-i||nut[i]!=-4-i)return 40;std::cout<<"SST_COMPLETE\n";}
 if(action=="particle_write") {runTime.setTime(.00001,1);mkDir(runTime.timePath());if(resident.writeParticleRestartMirror(runTime)!=1)return 41;std::cout<<"PARTICLE_WRITTEN\n";}
 if(action=="full_mirror") {runTime.setTime(.00001,1);resident.downloadToHostMirror(runTime,rho,rhoU,rhoE,U,p,Tgas,epsS,rhoUs,rhoEs,rhoDs,rhoHp,Us,theta,Tp,dMeanCell);
 forAll(rho,i)if(rho[i]!=100+i||rhoU[i]!=vector(101+i,102+i,103+i)||rhoE[i]!=1e35||U[i]!=vector(105+i,106+i,107+i)||p[i]!=108+i||Tgas[i]!=109+i||epsS[i]!=110+i||rhoUs[i]!=vector(111+i,112+i,113+i)||rhoEs[i]!=114+i||rhoDs[i]!=115+i||rhoHp[i]!=116+i||Us[i]!=vector(117+i,118+i,119+i)||theta[i]!=120+i||Tp[i]!=121+i||dMeanCell[i]!=122+i)return 42;std::cout<<"FULL_COMPLETE\n";}
 if(action=="species") {sharedGasModel.uploadInitialSpecies(nullptr);sharedGasModel.downloadSpecies(nullptr,rho);sharedGasModel.uploadInitialSpecies(nullptr);}
 if(action=="mirror") {runTime.setTime(.00001,1);resident.downloadNutToHostMirror(runTime,nut);forAll(nut,i)if(nut[i]!=-10-i)return 21;std::cout<<"MIRROR "<<nut[0]<<" "<<nut[nut.size()-1]<<"\\n";}
 if(action=="restart") {resident.readParticleRestartMirrorIfPresent(runTime);resident.readEpsGPrevRestartMirrorIfPresent(runTime);resident.readSourceResidualRestartMirrorIfPresent(runTime);}
 if(action=="write") {runTime.setTime(.00001,1);mkDir(runTime.timePath());resident.writeEpsGPrevRestartMirror(runTime);resident.writeSourceResidualRestartMirror(runTime);}
 if(action=="schedule") {labelList ids;scalarField times(1,0),values(1,-2);resident.configureScheduledInlet(ids,-300,times,values,times,values);}
 if(action=="boundary") resident.uploadGasBoundaryFields(mesh,rho,U,p,Tgas,Rgas.value(),true,sharedGasModel.active());

 if(action=="boundary_sizes") {
 const label expected=mesh.boundaryMesh()[0].size(),delta=std::atoi(std::getenv("PATCH_DELTA"));
 const std::string selected=std::getenv("PATCH_FIELD");
 auto& rb=rho.boundaryFieldRef()[0];auto& ub=U.boundaryFieldRef()[0];auto& pb=p.boundaryFieldRef()[0];auto& tb=Tgas.boundaryFieldRef()[0];
 if(selected=="rho")rb.setSize(expected+delta);if(selected=="U")ub.setSize(expected+delta);if(selected=="p")pb.setSize(expected+delta);if(selected=="T")tb.setSize(expected+delta);
 forAll(rb,i)rb[i]=-999;forAll(ub,i)ub[i]=vector(-999,-999,-999);forAll(pb,i)pb[i]=-999;forAll(tb,i)tb[i]=-999;
 resident.downloadGasBoundaryToHostMirror(rho,U,p,Tgas);
 if(rb.size()<expected||ub.size()<expected||pb.size()<expected||tb.size()<expected)return 44;
 for(label i=0;i<expected;++i)if(rb[i]!=2||ub[i]!=vector(3,4,5)||pb[i]!=6||tb[i]!=7)return 43;
 if(delta>0 && ((selected=="rho"&&rb[expected]!=-999)||(selected=="U"&&ub[expected]!=vector(-999,-999,-999))||(selected=="p"&&pb[expected]!=-999)||(selected=="T"&&tb[expected]!=-999)))return 45;
 std::cout<<"BOUNDARY_COMPLETE "<<expected<<" "<<selected<<" "<<delta<<"\n";
 }
 if(action=="bad_index") {List<long long> ids(1);ids[0]=4294967296LL;GpuHostReadView<int> converted(ids);std::cout<<"WRAPPED "<<converted.begin()[0]<<"\n";}
 if(action=="conversion") {
 Field<float> native(2);native[0]=.125f;native[1]=-3.5f;
 GpuHostReadView<double> converted(native);if(converted.begin()[0]!=.125||converted.begin()[1]!=-3.5)return 30;
 Field<double> exact(2);exact[0]=1e100;exact[1]=-.75;GpuHostReadView<double> borrowed(exact);if(borrowed.begin()!=exact.begin())return 31;
 List<long long> ids(2);ids[0]=7;ids[1]=-3;GpuHostReadView<int> indices(ids);if(indices.begin()[0]!=7||indices.begin()[1]!=-3)return 32;
 fileName file=runTime.timePath()/"conversion.bin";
 {std::ofstream os(file.c_str(),std::ios::binary);resident.writeBinaryParticleBlock(os,native.begin(),2,"float",file);}
 float decoded[2]={999,999};{std::ifstream is(file.c_str(),std::ios::binary);resident.readBinaryParticleBlock(is,decoded,2,"float",file);}
 if(decoded[0]!=native[0]||decoded[1]!=native[1])return 33;
 std::cout<<"CONVERTED "<<converted.begin()[0]<<" "<<converted.begin()[1]<<"\n";
 }
 if(action=="extra_storage") {rho.primitiveFieldRef().setSize(mesh.nCells()+1);rho[mesh.nCells()]=99;resident.ensureInternalSize(mesh,rho,"rho");std::cout<<"EXTRA "<<rho[mesh.nCells()]<<"\n";}
 if(action=="initialise") {
 resident.initialiseBase(runTime,mesh,rho,rhoU,rhoE,U,p,Tgas,epsS,rhoUs,rhoEs,rhoDs,rhoHp,Us,theta,Tp,dMeanCell,
 injectionParcelMass.value(),legacyRestartParcelMass.value(),10,0,7,gammaG.value(),Rgas.value(),true,rhoS.value(),false,0,0,0,muG.value(),PrG.value(),gasFluxScheme,gasReconstruction,gasLimiter,gasTimeIntegrator,gasRobustFallback,gasTurbulenceModel,lesDeltaCoeff,turbulentPrandtl,waleCw,smagorinskyCs,dS.value(),dS.value(),dS.value(),0,0,300,rhoMinG.value(),TgasMinG.value(),0,0,1,5000,gpuDragModel->modelId(),gpuDragModel->parameters(),g.value(),true,gpuScheduling,&ugkwpProps);
 }
 std::cout<<"FIELDS "<<rho[0]<<" "<<rhoE[0]<<" k="<<k[0]<<" omega="<<omega[0]<<" gamma="<<gammaG.value()<<"\n";
 return 0;
}
'''

def backend_stubs():
    text=(ROOT/'applications/gasUGKP/gpu/GpuBackendApi.H').read_text()
    text+='\n'+(ROOT/'applications/gasUGKP/gpu/GpuResidentStrict.H').read_text()
    found={}
    for match in re.finditer(r'extern "C"\s+(int|void|const char\*)\s+(ugkwp\w+)\s*\(([^;{}]*)\)\s*;',text):
        kind,name,params=match.groups()
        if name in found:continue
        body=BODIES.get(name,'' if kind=='void' else 'return 0;')
        found[name]=f'extern "C" {kind} {name}({params}){{{body}}}'
    return '\n'.join(found.values())

@pytest.fixture(scope='module')
def native_input_probe(tmp_path_factory):
    if not os.environ.get('WM_PROJECT_DIR'):pytest.skip('OpenFOAM environment required')
    root=tmp_path_factory.mktemp('gas_input_continuation');(root/'Make').mkdir()
    (root/'probe.C').write_text(PREFIX+backend_stubs()+MAIN)
    (root/'Make/files').write_text(f'probe.C\nEXE = {root}/probe\n')
    (root/'Make/options').write_text(f'EXE_INC = -DUGKWP_USE_CUDA -DUGKWP_GAS_SPECIES=2 -I{ROOT}/applications/gasUGKP -I{ROOT}/applications/gasUGKP/gpu -I{ROOT}/common -I$(LIB_SRC)/finiteVolume/lnInclude -I$(LIB_SRC)/meshTools/lnInclude\nEXE_LIBS = -lfiniteVolume -lmeshTools -lOpenFOAM\n')
    built=subprocess.run(['wmake'],cwd=root,capture_output=True,text=True)
    assert built.returncode==0,built.stdout+built.stderr
    return root/'probe'

def make_case(path):
    make_mesh_case(path)
    config=path/'system/fvSchemes'
    config.write_text(re.sub(r'fluxScheme [^;]+;', 'fluxScheme Kurganov;', config.read_text()))


def run_probe(binary,path,action='fields',**env):
    return subprocess.run([str(binary),'-case',str(path)],env=dict(os.environ,T07_ACTION=action,**env),capture_output=True,text=True)

def change(path,old,new):path.write_text(path.read_text().replace(old,new))

def test_t07_ordinary_native_fields_match_actual_thermo(native_input_probe,tmp_path):
    import json
    make_case(tmp_path)
    run=run_probe(native_input_probe,tmp_path)
    assert run.returncode==0,run.stdout+run.stderr
    fields=re.search(r'FIELDS (\S+) (\S+)',run.stdout)
    actual=json.loads((tmp_path/'case_parameters.json').read_text())
    assert float(fields[1])==pytest.approx(actual['rho'],rel=1e-14)
    assert float(fields[2])==pytest.approx(actual['rhoE'],rel=1e-14)

@pytest.mark.parametrize('action',['mirror','write','schedule'])
def test_t07_direct_native_calls_do_not_require_write_time_or_nonempty_faces(native_input_probe,tmp_path,action):
    make_case(tmp_path);run=run_probe(native_input_probe,tmp_path,action)
    assert run.returncode==0,run.stdout+run.stderr
    if action=='mirror':assert 'MIRROR -10 -17' in run.stdout
    if action=='schedule':assert 'SCHEDULE 0 -300' in run.stdout
    if action=='write':
        rows=(tmp_path/'1e-05/gpuResidentStrictEpsGPrev.dat').read_text().split()
        assert [float(v) for v in rows]==[8,.125,.15625,.1875,.21875,.25,.28125,.3125,.34375]

def test_t07_supplied_conserved_species_and_signed_complete_download(native_input_probe,tmp_path):
    make_case(tmp_path)
    rho=(tmp_path/'0/rho').read_text();(tmp_path/'0/rhoY_A').write_text(rho.replace('object rho','object rhoY_A').replace('internalField uniform '+re.search(r'internalField uniform ([^;]+)',rho)[1], 'internalField uniform -7'))
    run=run_probe(native_input_probe,tmp_path,'species')
    assert run.returncode==0,run.stdout+run.stderr
    assert 'SPECIES_UPLOAD -7 ' in run.stdout
    assert 'SPECIES_UPLOAD -2 -3' in run.stdout

@pytest.mark.parametrize('modifier',['future_schema','signed_parameters','missing_sst','different_dimensions'])
def test_t07_native_known_inputs_continue(native_input_probe,tmp_path,modifier):
    make_case(tmp_path)
    if modifier=='future_schema':
        for name in ['fluidProperties','particleProperties','schedulingProperties']:
            change(tmp_path/'constant'/name,'schemaVersion 1','schemaVersion 123')
    elif modifier=='signed_parameters':
        change(tmp_path/'constant/particleProperties','rhoS 2800','rhoS -2800')
        change(tmp_path/'constant/fluidProperties','mu 0; Pr 0.72','mu -0.1; Pr -0.72')
    elif modifier=='missing_sst':
        change(tmp_path/'constant/fluidProperties','simulationType laminar;', 'simulationType RAS; RAS { model kOmegaSST; }')
    else:
        (tmp_path/'constant/g').write_text('FoamFile {version 2.0; format ascii; class uniformDimensionedVectorField; object g;} dimensions [0 1 -1 0 0 0 0]; value (1e35 -2 3);')
    run=run_probe(native_input_probe,tmp_path)
    assert run.returncode==0,run.stdout+run.stderr
    if modifier=='missing_sst':assert 'k=9.9999999999999998e-13 omega=9.9999999999999995e-07' in run.stdout

@pytest.mark.parametrize('failure',['missing_thermal','malformed_sst','backend_transfer','short_restart'])
def test_t07_native_actual_input_or_execution_failures_remain(native_input_probe,tmp_path,failure):
    make_case(tmp_path);action='fields';env={}
    if failure=='missing_thermal':
        with (tmp_path/'constant/particleProperties').open('a') as f:f.write('\nparticleTemperatureTransport true;\n')
    elif failure=='malformed_sst':
        change(tmp_path/'constant/fluidProperties','simulationType laminar;', 'simulationType RAS; RAS { model kOmegaSST; }')
        (tmp_path/'0/k').write_text((tmp_path/'0/rho').read_text().replace('object rho','object k').replace('internalField uniform','internalField malformed'))
    elif failure=='backend_transfer':action='mirror';env['FAIL_TRANSFER']='1'
    else:
        action='restart';(tmp_path/'0/gpuResidentStrictParticles.dat').write_bytes(b'UGKP_PARTICLES_SCHEMA1_BIN 1 10\n'+struct.pack('<I',1)+b'\0\0')
    run=run_probe(native_input_probe,tmp_path,action,**env)
    assert run.returncode!=0
    expected={'missing_thermal':'particleRho','malformed_sst':'malformed','backend_transfer':'injected backend transfer failure','short_restart':'Failed reading binary'}[failure]
    assert expected in run.stdout+run.stderr

def test_t07_empty_particle_state_ignores_macro_discrepancy_and_range_proofs(native_input_probe,tmp_path):
    make_case(tmp_path);change(tmp_path/'0/epsilonS','internalField uniform 0;', 'internalField uniform 0.2;')
    (tmp_path/'0/gpuResidentStrictParticles.dat').write_text('UGKP_PARTICLES_SCHEMA4 0\n')
    with (tmp_path/'constant/particleProperties').open('a') as f:f.write('\ngpuResidentCollisionalRestitution -0.5; gpuResidentPressureKickFraction -0.25; gpuResidentPackingFraction 1.1; gpuResidentPackingProjectionIterations 0;\n')
    run=run_probe(native_input_probe,tmp_path,'initialise')
    assert run.returncode==0,run.stdout+run.stderr
    assert 'CREATE 10 0 -0.5 -0.25 1.1000000000000001 0' in run.stdout
    assert 'PARTICLES 0 ' in run.stdout

@pytest.mark.parametrize('mode',['trailing','excess_chunk','signed_mass','extra_cells','future_residual'])
def test_t07_readable_native_restart_data_reaches_real_upload(native_input_probe,tmp_path,mode):
    make_case(tmp_path)
    if mode in ['trailing','excess_chunk','signed_mass']:
        mass=-3 if mode=='signed_mass' else 3
        data=b'UGKP_PARTICLES_SCHEMA1_BIN 1 0\n'+struct.pack('<I',1)+struct.pack('<10d2i2Q',.125,2,3,4,5,6,300,8,.001,mass,0,1,11,12)
        if mode=='trailing':data+=b'future metadata'
        (tmp_path/'0/gpuResidentStrictParticles.dat').write_bytes(data)
    elif mode=='extra_cells':(tmp_path/'0/gpuResidentStrictEpsGPrev.dat').write_text('9\n'+'.75\n'*9)
    else:(tmp_path/'0/gpuResidentStrictSourceResidual.dat').write_text('UGKP_SOURCE_RESIDUAL_SCHEMA99 1\n8 -3\n')
    run=run_probe(native_input_probe,tmp_path,'restart')
    assert run.returncode==0,run.stdout+run.stderr
    if mode in ['trailing','excess_chunk','signed_mass']:assert f'PARTICLES 1 0.125 {mass}' in run.stdout
    elif mode=='extra_cells':assert 'EPS 0.75' in run.stdout
    else:assert 'RESIDUAL 1 -3' in run.stdout


def test_t07_typed_host_views_and_binary_conversion_preserve_real_values(native_input_probe,tmp_path):
    make_case(tmp_path)
    result=run_probe(native_input_probe,tmp_path,'conversion')
    assert result.returncode==0,result.stdout+result.stderr
    assert (tmp_path/'0/conversion.bin').read_bytes()==struct.pack('<2d',.125,-3.5)

def test_t07_excess_internal_storage_is_not_missing_input(native_input_probe,tmp_path):
    make_case(tmp_path)
    result=run_probe(native_input_probe,tmp_path,'extra_storage')
    assert result.returncode==0,result.stdout+result.stderr
    assert 'EXTRA 99' in result.stdout


@pytest.mark.parametrize("geometry", ["patch","wall"])
def test_t07_existing_boundary_precedence_uploads_actual_values(native_input_probe,tmp_path,geometry):
    make_case(tmp_path)
    (tmp_path/'constant/gasModelProperties').unlink()
    path=tmp_path/'constant/polyMesh/boundary'
    path.write_text(re.sub(r'type\s+cyclic;', 'type '+geometry+';',path.read_text()))
    for name in ('rho','p','T','U','epsilonS','Us','theta','rhoE'):
        path=tmp_path/'0'/name
        values={'rho':10,'p':100,'T':5}
        kind=f'fixedValue; value uniform {values[name]};' if name in values else 'zeroGradient;'
        path.write_text(path.read_text().replace('type cyclic;', 'type '+kind))
    result=run_probe(native_input_probe,tmp_path,'boundary')
    assert result.returncode==0,result.stdout+result.stderr
    assert 'BOUNDARY 10 100 5 1 0 1 1' in result.stdout

def test_t07_unrepresentable_actual_c_abi_index_does_not_wrap_to_another_cell(native_input_probe,tmp_path):
    make_case(tmp_path)
    result=run_probe(native_input_probe,tmp_path,'bad_index')
    assert result.returncode!=0
    assert 'index' in result.stdout+result.stderr


@pytest.mark.parametrize("action,marker", [("sst_mirror","SST_COMPLETE"),("full_mirror","FULL_COMPLETE"),("particle_write","PARTICLE_WRITTEN")])
def test_t07_complete_native_publication_outside_write_time(native_input_probe,tmp_path,action,marker):
    make_case(tmp_path)
    result=run_probe(native_input_probe,tmp_path,action)
    assert result.returncode==0,result.stdout+result.stderr
    assert marker in result.stdout
    if action=='particle_write':
        data=(tmp_path/'1e-05/gpuResidentStrictParticles.dat').read_bytes().split(b'\n',1)[1]
        assert data==struct.pack('<I10d2i2Q',1,.125,2,3,4,5,6,300,8,.001,3,0,1,11,12)


@pytest.mark.parametrize('field',['rho','U','p','T'])
@pytest.mark.parametrize('delta',[1,-1])
def test_repair_boundary_publication_fills_all_components_and_preserves_tails(native_input_probe,tmp_path,field,delta):
    make_case(tmp_path)
    result=run_probe(native_input_probe,tmp_path,'boundary_sizes',PATCH_FIELD=field,PATCH_DELTA=str(delta))
    assert result.returncode==0,result.stdout+result.stderr
    assert 'BOUNDARY_COMPLETE' in result.stdout


@pytest.mark.parametrize('mode',['ordinary','legacy','future','empty_future','zero_chunk','short_future','zero_eof','overflow_chunk'])
def test_repair_binary_labels_and_zero_chunks_consume_actual_data(native_input_probe,tmp_path,mode):
    make_case(tmp_path)
    legacy=mode=='legacy';future=mode in ('future','empty_future','short_future')
    version=5 if legacy else 99 if future else 1
    count=0 if mode=='empty_future' else 1
    data=f'UGKP_PARTICLES_SCHEMA{version}_BIN {count} 10\n'.encode()
    if mode in ('zero_chunk','zero_eof'):data+=struct.pack('<I',0)
    if count and mode!='zero_eof':
        data+=struct.pack('<I',2 if mode=='overflow_chunk' else 1)
        values=(.125,2,3,4,5,6,300,8,.001)
        data+=struct.pack('<9d2i2Q',*values,0,1,11,12) if legacy else struct.pack('<10d2i2Q',*values,3,0,1,11,12)
    if mode=='short_future':data=data[:-3]
    (tmp_path/'0/gpuResidentStrictParticles.dat').write_bytes(data)
    env={'VERIFY_PARTICLE':'1'}
    if legacy:env['LEGACY_PARTICLE']='1'
    result=run_probe(native_input_probe,tmp_path,'restart',**env)
    if mode in ('short_future','zero_eof','overflow_chunk'):
        assert result.returncode!=0
        assert ('Invalid binary restart chunk count' if mode=='overflow_chunk' else 'Failed reading binary') in result.stdout+result.stderr
    else:
        assert result.returncode==0,result.stdout+result.stderr
        assert f'PARTICLES {count} ' in result.stdout
