"""Focused real host wrapper methods/configuration fragments; no OpenFOAM/GPU claim."""
from pathlib import Path
import subprocess
import os
import pytest
from test_native_cleanup import block, run_cpp
from test_native_consumer_admission import HEADER_PRELUDE
ROOT=Path(__file__).resolve().parents[1]
BASE='5f1a39ac96e2e7ff9a0640a4db344d9cccd34245'
def source(app,rev=None):
 rev=rev or os.environ.get('RESIDENT_WRAPPER_SOURCE_REVISION')
 p=f'applications/{app}/gpu/GpuResidentStrict.H'
 return subprocess.check_output(['git','show',rev+':'+p],cwd=ROOT,text=True) if rev else (ROOT/p).read_text()
PRE=r'''
#include <cassert>
#include <cmath>
#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include <stdexcept>
using scalar=double;
constexpr double SMALL=1e-15;constexpr int FatalError=1;
#define FatalErrorInFunction std::cerr
int throwingExit(){throw std::runtime_error("wrapper rejection");}
#define exit(x) throwingExit()
'''
@pytest.mark.parametrize('app',['CHT','FSH'])
def test_courant_wrapper_forwards_supplied_scalars_and_native_error(tmp_path,app):
 method=block(source(app),'    scalar computeGasCourant')
 code=PRE+r'''
int calls=0,error=0;double supplied[3];
int ugkwpGpuResidentStrictComputeGasCourant(void*,double dt,double co,double time,double*out){++calls;supplied[0]=dt;supplied[1]=co;supplied[2]=time;*out=dt+co+time;return error;}
struct Resident{void*handle_=this;void requireInitialised(const char*)const{}static bool finiteMirrorScalar(double v){return std::isfinite(v)&&std::abs(v)<1e30;}void check(int e,const char*)const{if(e)throw std::runtime_error("native");}
'''+method+r'''
};int main(){Resident r;assert(r.computeGasCourant(1.,.5,2.)==3.5);assert(r.computeGasCourant(-1.,-.5,-2.)==-3.5&&calls==2);assert(std::isnan(r.computeGasCourant(NAN,.5,2.))&&calls==3);error=1;bool failed=false;try{r.computeGasCourant(1.,.5,2.);}catch(const std::runtime_error&){failed=true;}assert(failed&&calls==4);}
'''
 run_cpp(tmp_path,code)

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_constructor_configuration_fragment_preserves_lookups_without_numerical_admission(tmp_path,app):
 src=source(app);start=src.index('        const scalar collisionalRestitution =');end=src.index('        const bool csrCellLocalPathEnabled',start)
 fragment=src[start:end]
 # Same typed dictionary lookups as production. Inspect native Create separately for original clamps.
 code=PRE+r'''
using label=int;
struct dictionary{template<class T>T lookupOrDefault(const char*name,T fallback)const{std::string n(name);if(n=="gpuResidentPackingFraction")return T(2);if(n=="gpuResidentCollisionalRestitution")return T(-.25);if(n=="gpuResidentPressureKickFraction")return T(3);if(n=="gpuResidentPackingProjectionIterations")return T(0);return fallback;}};
bool finiteMirrorScalar(double v){return std::isfinite(v)&&std::abs(v)<1e30;}
int main(){dictionary d;auto*particleWallProps=&d;bool enableParticleModule=false;
'''+fragment+r'''
assert(collisionalRestitution==-.25&&pressureKickFraction==3&&packingFraction==2&&packingProjectionIterations==0&&!jammingPressureEnabled);}
'''
 run_cpp(tmp_path,code)

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_current_writer_header_is_recognised_by_constructor_probe(tmp_path,app):
 src=source(app);section=block(src,'            if (hasParticleRestart)')
 header=f'applications/{app}/gpu/'+('GpuChtParticleRestart.H' if app=='CHT' else 'GpuFshParticleRestart.H')
 code=PRE+'#include "'+header+'"\n'+r'''
using label=int;using string=std::string;namespace gpuFshRestart=Foam::gpuFshRestart;
int main(){const std::string restartPath="PATH";bool hasParticleRestart=true;int restartParticleCount=-1;
{std::ofstream out(restartPath);out<<gpuFshRestart::currentFormatName<<" 2\n";}
'''.replace('PATH',str(tmp_path/'current.restart'))+section+r'''
assert(restartParticleCount==2);}
'''
 run_cpp(tmp_path,code)

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_terminal_gate_source_inventory(app):
 src=source(app)
 assert 'computeGasCourant requires positive finite' not in src
 assert 'Invalid weighted FSH parcel state at particle' not in src
 assert 'Nonfinite V3 deposition area at particle' not in src
 assert 'gpuResidentCollisionalRestitution must be finite' not in src
 native=(ROOT/f'applications/{app}'/('gpu' if app=='CHT' else 'private_backend')/'GpuResidentStrict.cu').read_text()
 assert 'invalid serial translational cyclic mesh pair' not in native
 assert 'pure-gas advance requires zero particle capacity' not in native
 assert '!std::isfinite(simulationTime) || simulationTime <' not in block(native,'extern "C" int ugkwpGpuResidentStrictAdvanceGasOnly')

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_current_actual_writer_constructor_probe_and_full_restore(tmp_path,app):
 src=source(app)
 methods=block(src,'    label writeParticleRestartMirror')+'\n'+block(src,'    void readParticleRestartMirrorIfPresent')
 probe=block(src,'            if (hasParticleRestart)')
 header=f'applications/{app}/gpu/'+('GpuChtParticleRestart.H' if app=='CHT' else 'GpuFshParticleRestart.H')
 code=PRE+'#include "'+header+'"\n'+r'''
#include <cstring>
#include <type_traits>
#include <filesystem>
using label=int;using string=std::string;using fileName=std::string;
namespace gpuFshRestart=Foam::gpuFshRestart;
template<class T>struct Field:std::vector<T>{using std::vector<T>::vector;T*begin(){return this->data();}const T*cdata()const{return this->data();}Field&operator=(T x){std::fill(this->data(),this->data()+this->size(),x);return *this;}};
using scalarField=Field<double>;using labelList=Field<int>;template<class T>using List=Field<T>;
struct Time{std::string path;std::string timePath()const{return path;}bool writeTime()const{return true;}};
bool isFile(const std::string&p){return std::filesystem::is_regular_file(p);}void mkDir(const std::string&p){std::filesystem::create_directories(p);}
std::vector<std::vector<unsigned char>>written,restored;int uploads=0,savedUploads=0;
int width(int i){return i==18||i==19||i==23?8:i==22?64:1;}
template<class...P>int ugkwpGpuResidentStrictDownloadParticleRestartMirror(void*,int*n,int capacity,P...p){*n=2;if(!capacity)return 0;int index=0;auto fill=[&](auto d){if constexpr(std::is_pointer_v<decltype(d)>){using T=std::remove_pointer_t<decltype(d)>;for(int j=0;j<2*width(index);++j)d[j]=static_cast<T>(index+j+.25);auto*b=reinterpret_cast<const unsigned char*>(d);written.emplace_back(b,b+2*width(index)*sizeof(T));}++index;};(fill(p),...);return 0;}
template<class...P>int ugkwpGpuResidentStrictUploadParticleRestartMirror(void*,int n,P...p){assert(n==2);++uploads;int index=0;auto save=[&](auto d){using T=std::remove_pointer_t<decltype(d)>;auto*b=reinterpret_cast<const unsigned char*>(d);restored.emplace_back(b,b+2*width(index)*sizeof(T));++index;};(save(p),...);return 0;}
int ugkwpGpuResidentStrictDownloadParticleSavedVelocity(void*,int n,double*x,double*y,double*z){for(int i=0;i<n;++i){x[i]=1.25+i;y[i]=2.5+i;z[i]=3.75+i;}return 0;}
int ugkwpGpuResidentStrictUploadParticleSavedVelocity(void*,int n,const double*x,const double*y,const double*z){++savedUploads;for(int i=0;i<n;++i)assert(x[i]==1.25+i&&y[i]==2.5+i&&z[i]==3.75+i);return 0;}
struct Resident{void*handle_=this;int hostMirrorFaces_=4;bool particleWallHeatTransferEnabled_=false;double legacyRestartParcelMass_=1;labelList particleStuckCandidateFaceMask_=labelList(4,1);
void requireInitialised(const char*)const{}void check(int e,const char*)const{if(e)throw std::runtime_error("native");}static bool finiteMirrorScalar(double v){return std::isfinite(v)&&std::abs(v)<1e30;}
static std::string particleRestartFile(const Time&t){return t.path+"/particles";}void writeEpsGPrevRestartMirror(const Time&,bool=false)const{}void writeSourceResidualRestartMirror(const Time&,bool=false)const{}
METHODS
};
int main(){Time runTime{"DIR"};std::filesystem::create_directories(runTime.path);Resident r;assert(r.writeParticleRestartMirror(runTime)==2);
const std::string restartPath=Resident::particleRestartFile(runTime);bool hasParticleRestart=true;int restartParticleCount=-1;
PROBE
assert(restartParticleCount==2);r.readParticleRestartMirrorIfPresent(runTime);assert(uploads==1&&savedUploads==1&&written==restored);
// The same real restore route must expose truncated stored bytes.
std::filesystem::resize_file(restartPath,32);bool failed=false;try{r.readParticleRestartMirrorIfPresent(runTime);}catch(const std::runtime_error&){failed=true;}assert(failed&&uploads==1);
}
'''.replace('METHODS',methods).replace('PROBE',probe).replace('DIR',str(tmp_path/'files'))
 run_cpp(tmp_path,code)

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_additional_numerical_wrapper_inventory(app):
 src=source(app)
 for text in ['gpuResidentParticleCapacity must be non-negative','gpuResidentMaxFaceWalkHops must be positive','GpuGasResidentSolver must not allocate particle storage','Negative GPU resident particle restart count','particleWallDefault must be finite','contains duplicate patch','requires particleWallCoeffs > SMALL','failed paired centre/area validation','violates p=rho*R*T']:
  assert text not in src,text
 assert 'field.primitiveField().size() != mesh.nCells()' not in src

def test_affine_wrapper_uses_actual_supplied_fields_without_identity_certificate(tmp_path):
 method=block(source('CHT'),'    void applyParticleRadiationEnergy')
 code=PRE+r'''
#include "applications/CHT/gpu/GpuParticleRadiationMath.H"
namespace gpuThermal=Foam::gpuThermal;
template<class T>struct Field:std::vector<T>{using std::vector<T>::vector;T*begin(){return this->data();}};
using scalarField=Field<double>;template<class T>using List=Field<T>;
struct fvMesh{int nCells()const{return 2;}};
#define forAll(f,i) for(size_t i=0;i<(f).size();++i)
std::vector<gpuThermal::RadiationAffineTemperatureUpdate>output;int calls=0,error=0;
int ugkwpGpuResidentStrictApplyParticleRadiationAffineTemperature(void*,int n,const gpuThermal::RadiationAffineTemperatureUpdate*p){++calls;output.assign(p,p+n);return error;}
struct Resident{void*handle_=this;const fvMesh*residentMesh_=nullptr;double residentParticleHeatFactorJkgK_=99,residentTpMin_=200,residentTpMax_=4000;
void requireInitialised(const char*)const{}void check(int e,const char*)const{if(e)throw std::runtime_error("native");}
'''+method+r'''
};int main(){Resident r;fvMesh mesh;scalarField energy{2,-3,777},mass{1,2,777},temperature{300,400,777};r.applyParticleRadiationEnergy(mesh,energy,mass,temperature,2);assert(output.size()==2&&calls==1);
for(int i=0;i<2;++i){auto expected=gpuThermal::radiationAffineTemperatureUpdate(energy[i],mass[i],2,temperature[i],200,4000);assert(output[i].scale==expected.scale&&output[i].offset==expected.offset);}
energy.resize(1);bool failed=false;try{r.applyParticleRadiationEnergy(mesh,energy,mass,temperature,2);}catch(const std::runtime_error&){failed=true;}assert(failed&&calls==1);}
'''
 run_cpp(tmp_path,code)

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_actual_internal_extent_checks_and_solid_sum_ignore_surplus(tmp_path,app):
 src=source(app);method=block(src,'    static void ensureInternalSize');start=src.index('            scalar initialSolidVolume = 0;');end=src.index('            reduce(initialSolidVolume',start);summation=src[start:end]
 code=PRE+r'''
using label=int;template<class T>struct Field:std::vector<T>{using std::vector<T>::vector;};using scalarField=Field<double>;
struct fvPatchField{};struct volMesh{};
template<class T,class P,class M>struct GeometricField{Field<T>data;const Field<T>&primitiveField()const{return data;}};
struct fvMesh{scalarField volumes{2,3};int nCells()const{return 2;}const scalarField&V()const{return volumes;}};
using std::max;
#define forAll(f,i) for(size_t i=0;i<(f).size();++i)
struct Resident{
'''+method+r'''
};int main(){fvMesh mesh;GeometricField<double,fvPatchField,volMesh>epsS;epsS.data={.25,.5,999999};Resident::ensureInternalSize(mesh,epsS,"eps");const int nCells=mesh.nCells();
'''+summation+r'''
assert(initialSolidVolume==2.);epsS.data.resize(1);bool failed=false;try{Resident::ensureInternalSize(mesh,epsS,"eps");}catch(const std::runtime_error&){failed=true;}assert(failed);}
'''
 run_cpp(tmp_path,code)

@pytest.mark.parametrize('app',['CHT','FSH'])
def test_actual_wall_configuration_fragment_writes_supplied_values(tmp_path,app):
 from test_native_cleanup import CUDA
 method=block(source(app),'    void uploadMesh');fragment=block(method,'        if (particleWallProps != nullptr)')
 code=CUDA+HEADER_PRELUDE+r'''
constexpr int FatalError=1;
bool finiteMirrorScalar(double x){return std::isfinite(x)&&std::abs(x)<1e30;}
int main(){fvMesh mesh;dictionary d;d.set("particleWallDefault",-2.);d.children["particleWallCoeffs"].set("a",2.5);d.set("gpuResidentStuckWallPatches",wordList{"b","b"});const dictionary*particleWallProps=&d;
scalarField particleBoundaryCoeff(mesh.boundaryMesh().size(),1.);labelList particleBoundaryStuck(mesh.boundaryMesh().size(),0);
'''+fragment+r'''
assert(particleBoundaryCoeff[0]==2.5&&particleBoundaryCoeff[1]==-2.&&particleBoundaryStuck[0]==0&&particleBoundaryStuck[1]==1);}
'''
 run_cpp(tmp_path,code)
