"""CHMT supplied complete tables and selected-mode admission boundaries."""
from validation.tools.paths import SOURCE_ROOT, VALIDATION_ROOT, CASES_ROOT

from pathlib import Path
import subprocess
import json
ROOT=SOURCE_ROOT
APP=ROOT/'applications/CHMT'
BASE='360bf94dd6c5344bb771d5f1f8b02a79bd041d92'

def build_run(tmp_path,body,header=None):
    source=tmp_path/'probe.cpp';source.write_text(body);cmd=['g++','-std=c++17','-O0','-Wall','-Wextra','-Werror','-I'+str(APP),'-I'+str(ROOT/'common'),str(source),'-o',str(tmp_path/'probe')]
    result=subprocess.run(cmd,capture_output=True,text=True);assert result.returncode==0,result.stderr
    result=subprocess.run([str(tmp_path/'probe')],capture_output=True,text=True);assert result.returncode==0,result.stdout+result.stderr
    return result.stdout

GAS=r'''
ugkwp::GasModelConfiguration gas(){return ugkwp::parseGasModelProperties(R"(gasMode mixtureFrozen;species(S0 S1);speciesThermo{S0{model linearCp;molarMass .028;minTemperature 100;maxTemperature 3000;coefficients(1040 0 700);}S1{model linearCp;molarMass .032;minTemperature 100;maxTemperature 3000;coefficients(950 .1 -300);}}diffusion{model constant;coefficients(.001 .002);turbulentSchmidt .8;})");}
'''

def test_complete_foreign_and_single_tables_are_consumed(tmp_path):
    build_run(tmp_path,'#include "configuration/SharedGasModel.H"\n#include <cassert>\n'+GAS+r'''
int main(){auto g=gas();chmt::ModelConfig m;std::string error;g.mode=static_cast<ugkwp::GasMode>(77);g.speciesNames.push_back("extra");g.diffusionCoefficients.push_back(42);g.species[0].minTemperature=-1;assert(chmt::bindSharedGasModel(g,m,error));assert(int(m.physics.gasMode)==77&&m.physics.species[0].Tmin==-1&&m.physics.species[0].e0==700);assert(chmt::bindSingleGasModel(g,"S1",m,error));assert(m.physics.singleGasSpecies==1&&m.physics.species[1].cp1==.1&&m.physics.gasDiffusivity[0]==.001);}
''')

def binding_header(tmp_path,rev=None):
    if rev is None:return '#include "configuration/SharedGasModel.H"\n'
    text=subprocess.check_output(['git','show',rev+':applications/CHMT/configuration/SharedGasModel.H'],cwd=ROOT,text=True)
    text=text.replace('"../../../common/gasTransport/GasModelIO.H"','"'+str(ROOT/'common/gasTransport/GasModelIO.H')+'"')
    path=tmp_path/'SharedGasModel.H';path.write_text(text);return '#include "'+str(path)+'"\n'

def test_old_model_payload_exact_and_removed_copy_cost(tmp_path):
    main=r'''
#include "restart/Checkpoint.C"
#include <cassert>
#include <iostream>
#include <iomanip>
static bool counting=false;static size_t allocationCount=0,allocationBytes=0;
void*operator new(std::size_t n){if(counting){++allocationCount;allocationBytes+=n;}if(void*p=std::malloc(n?n:1))return p;throw std::bad_alloc();}
void operator delete(void*p)noexcept{std::free(p);}void operator delete(void*p,std::size_t)noexcept{std::free(p);}
GAS
int main(){auto g=gas();chmt::ModelConfig m;std::string error;m.modelName=std::string(1024,'a');m.materialSource=std::string(1024,'b');m.mechanismSource=std::string(1024,'c');m.gasPhase=std::string(1024,'d');m.speciesNames[0]=std::string(1024,'e');m.condensedNames[0]=std::string(1024,'f');
 counting=true;assert(chmt::bindSharedGasModel(g,m,error));counting=false;std::cout<<"COST "<<allocationCount<<" "<<allocationBytes<<" MODEL "<<sizeof(m)<<"\n";chmt::Encoder data;data(m);std::cout<<"DATA "<<data.data.size()<<" ";for(auto b:data.data)std::cout<<std::hex<<std::setw(2)<<std::setfill('0')<<int(b);std::cout<<"\n";}
'''.replace('GAS',GAS)
    results={}
    for label,rev in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();results[label]=build_run(d,binding_header(d,rev)+main)
    assert results['old'].splitlines()[1]==results['current'].splitlines()[1]
    old=int(results['old'].split()[1]);new=int(results['current'].split()[1]);assert old>new
    (tmp_path/'binding-parity.json').write_text(json.dumps(results,indent=2))

def test_single_missing_name_retains_output_and_last_match_routing(tmp_path):
    build_run(tmp_path,binding_header(tmp_path)+'#include "restart/Checkpoint.C"\n#include <cassert>\n'+GAS+r'''
int main(){auto g=gas();chmt::ModelConfig m;m.modelName="sentinel";m.physics.gravity.x=91;std::string error;chmt::Encoder before;before(m);assert(!chmt::bindSingleGasModel(g,"absent",m,error));chmt::Encoder after;after(m);assert(before.data==after.data&&!error.empty());g.speciesNames[0]=g.speciesNames[1]="same";assert(chmt::bindSingleGasModel(g,"same",m,error));assert(m.physics.singleGasSpecies==1&&m.physics.species[1].e0==-300);}
''')

def evaluate_probe(rev=None):
    import importlib.util
    spec=importlib.util.spec_from_file_location('helper',(VALIDATION_ROOT / 'tests/test_native_cleanup.py'));helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    p='applications/CHMT/materials/CpuMaterialDriver.C';src=subprocess.check_output(['git','show',rev+':'+p],cwd=ROOT,text=True) if rev else (ROOT/p).read_text()
    body=helper.block(src,'    bool evaluate(const HostState&')
    signature='const HostState&,const PhysicsConfig&,const std::vector<GasPrimitive>&,const std::vector<GasGradient>&,const std::vector<SolidQ>&,const std::vector<FilmQ>&,Real,std::uint64_t,CpuSurfaceResult&,std::string&,const std::vector<Real>*,const std::vector<GasWallClosureContext>*,bool*'
    if rev: signature=signature.replace('std::string&,const std::vector<Real>*','std::string&,const std::vector<Vec3>*,const std::vector<Real>*')
    return r'''
#include "ablation/CpuSurfaceInterface.H"
#include "../common/gasTransport/GasMechanismIO.H"
#include <cassert>
#include <iostream>
namespace chmt {
int prepared=0,sampled=0,contextCalls=0,surfaceCalls=0;bool geometryOk=true,surfaceOk=true;std::uint64_t observedHash=0;const void*observedReactions=nullptr;
struct MockWall{ugkwp::gaswall::WallStatus status;template<class...T>bool prepareGeometry(T&&...){++prepared;return geometryOk;}template<class...T>bool sample(T&&...){++sampled;return true;}
 template<bool=false>bool contexts(const std::vector<GasWallMatchingSample>&,const ugkwp::GasMechanismView<Real,Ns>& view,const ugkwp::gaswall::WallModelConfig<Real>&,std::vector<GasWallClosureContext>&,std::string&,bool*){++contextCalls;observedHash=view.mechanismHash;observedReactions=view.reactions;return true;}
 const ugkwp::gaswall::WallStatus&failureStatus()const{return status;}};
'''.replace('"../common/gasTransport/GasMechanismIO.H"','"'+str(ROOT/'common/gasTransport/GasMechanismIO.H')+'"')+'bool evaluateCpuSurface('+signature+'){++surfaceCalls;return surfaceOk;}\nnamespace detail {bool completeCpuSurface('+signature+'){++surfaceCalls;return surfaceOk;}}\n'+r'''
struct Driver{MockWall storage;MockWall*wall=&storage;ugkwp::GasMechanismConfiguration mechanism;bool terminalWallFailure=false,surfaceNumericalFailure=false;
'''+body+'};}\n'

def test_material_identity_only_consumer_reaches_actual_view_and_keeps_failures(tmp_path):
    main=r'''
int main(){using namespace chmt;Driver d;HostState state;PhysicsConfig p;p.gasMode=ugkwp::GasMode::MixtureChemistry;p.gasMechanismFingerprint=17;d.mechanism.mechanismHash=17;d.mechanism.speciesNames.resize(Ns);d.mechanism.reactions.resize(1);CpuSurfaceResult result;std::string error;assert(d.evaluate(state,p,{},{},{},{},.1,1,result,error));assert(observedHash==17&&observedReactions==d.mechanism.reactions.data());std::cout<<prepared<<" "<<sampled<<" "<<contextCalls<<" "<<surfaceCalls<<"\n";EXTRA}
'''
    results={}
    for label,rev in [('old',BASE),('current',None)]:
        d=tmp_path/label;d.mkdir();results[label]=build_run(d,evaluate_probe(rev)+main.replace('EXTRA',''))
    assert results['old']==results['current'];(tmp_path/'material-parity.json').write_text(json.dumps(results,indent=2))
    d=tmp_path/'admitted';d.mkdir();extra=r'''
 d.mechanism.speciesNames.clear();d.mechanism.mechanismHash=99;assert(d.evaluate(state,p,{},{},{},{},.1,1,result,error));assert(observedHash==99&&observedReactions==d.mechanism.reactions.data());geometryOk=false;int prior=surfaceCalls;assert(!d.evaluate(state,p,{},{},{},{},.1,1,result,error)&&surfaceCalls==prior&&d.terminalWallFailure);geometryOk=true;surfaceOk=false;assert(!d.evaluate(state,p,{},{},{},{},.1,1,result,error));
''';build_run(d,evaluate_probe()+main.replace('EXTRA',extra))


def test_current_material_surface_interface_after_support_removal(tmp_path):
    main=r'''
int main(){using namespace chmt;Driver d;HostState state;PhysicsConfig p;p.gasMode=ugkwp::GasMode::MixtureChemistry;d.mechanism.mechanismHash=17;d.mechanism.speciesNames.resize(Ns);d.mechanism.reactions.resize(1);CpuSurfaceResult result;std::string error;assert(d.evaluate(state,p,{},{},{},{},.1,1,result,error));assert(observedHash==17&&observedReactions==d.mechanism.reactions.data());assert(prepared==1&&sampled==1&&contextCalls==1&&surfaceCalls==1);}
'''
    build_run(tmp_path,evaluate_probe()+main)
