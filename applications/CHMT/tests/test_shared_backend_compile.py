"""C++ syntax/instantiation check after removing launch annotations, not CUDA QA."""
from pathlib import Path
import subprocess
import re
import runpy
APP=Path(__file__).resolve().parents[1]
ROOT=APP.parents[1]

def prepare_backend(tmp_path):
    stub=runpy.run_path(str(APP/'tests/test_shared_device_storage.py'))['STUB']
    stub+=r'''
#include <cmath>
using std::isfinite;
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
#define __CUDACC__
#define asm(...) std::abort()
struct Index{int x=0;}; inline Index blockIdx,blockDim,threadIdx;
using cudaGraph_t=void*;using cudaGraphExec_t=void*;using cudaGraphNode_t=void*;
using cudaGraphNodeType=int; constexpr int cudaGraphNodeTypeKernel=1,cudaStreamCaptureModeThreadLocal=1;
struct cudaKernelNodeParams{void*func=nullptr;void**kernelParams=nullptr;void**extra=nullptr;};
inline int cudaGraphExecDestroy(void*){return 0;}inline int cudaGraphDestroy(void*){return 0;}
inline int cudaStreamCreate(void**){return 0;}inline int cudaStreamBeginCapture(void*,int){return 0;}
inline int cudaStreamEndCapture(void*,void**){return 0;}inline int cudaGraphGetNodes(void*,void**,size_t*){return 0;}
inline int cudaGraphNodeGetType(void*,int*){return 0;}inline int cudaGraphKernelNodeGetParams(void*,cudaKernelNodeParams*){return 0;}
inline int cudaGraphInstantiate(void**,void*,void*,void*,int){return 0;}
inline int cudaGraphExecKernelNodeSetParams(void*,void*,cudaKernelNodeParams*){return 0;}
inline int cudaGraphLaunch(void*,void*){return 0;}
'''
    (tmp_path/'cuda_runtime.h').write_text(stub)
    tree=tmp_path/'tree';(tree/'applications/CHMT/gpu').mkdir(parents=True);(tree/'common').mkdir()
    for p in (ROOT/'common').iterdir():
        if p.name=='GpuGasAdvance.cuh':
            (tree/'common'/p.name).write_text(re.sub(r'<<<[\s\S]*?>>>','',p.read_text()))
        else:(tree/'common'/p.name).symlink_to(p,target_is_directory=p.is_dir())
    source=tree/'applications/CHMT/gpu/Backend.cpp'
    source.write_text(re.sub(r'<<<[\s\S]*?>>>','',(APP/'gpu/Backend.cu').read_text()))
    return source

def compiler_flags(tmp_path):
    return ['g++','-std=c++17','-Werror=return-type','-Wno-unused-variable','-Wno-unused-function',
            '-I'+str(tmp_path),'-I'+str(APP),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics')]

def test_shared_backend_host_syntax(tmp_path):
    source=prepare_backend(tmp_path)
    subprocess.run(compiler_flags(tmp_path)+['-fsyntax-only',str(source)],check=True)

def test_real_interface_hooks_preserve_scaled_exchange(tmp_path):
    source=prepare_backend(tmp_path)
    source.write_text(source.read_text()+r'''
#include "configuration/SharedGasModel.H"
#include <cassert>
int main(){
 using namespace chmt;
 auto gas=ugkwp::parseGasModelProperties(R"(gasMode mixtureFrozen;species(A B);speciesThermo{
 A{model linearCp;molarMass 0.028;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);}
 B{model linearCp;molarMass 0.032;minTemperature 100;maxTemperature 3000;coefficients(1040 0 0);}}
 diffusion{model none;})");
 ModelConfig model;std::string error;assert(bindSharedGasModel(gas,model,error));model.physics.gasConductivity=1;
 HostState h;h.gas.resize(1);h.gas[0].mass=1;h.gas[0].species[0]=1;h.gas[0].energy=1e6;h.solid.resize(1);h.solid[0].condensed[0]=1;
 auto&m=h.gasMesh;m.volumes={1};m.cellCentres={{0,0,0}};m.owner={0,0};m.neighbour={-1,-1};m.periodicPartner={-1,-1};m.cellFaceOffsets={0,2};m.cellFaces={0,1};
 m.faceCentres={{-1,0,0},{1,0,0}};m.areaVectors={{-1,0,0},{1,0,0}};m.boundaryKind={BoundaryKind::Interface,BoundaryKind::Slip};m.boundaryPrimitive.resize(2);
 h.solidMesh.volumes={1};h.solidMesh.owner={0};h.solidMesh.neighbour={-1};
 h.surface.area={1};h.surface.gasFace={0};h.surface.solidFace={0};h.surface.solidCell={0};h.surface.persistentId={7};h.surface.normal={{1,0,0}};h.surface.gasDistance={1};h.surface.solidDistance={1};
 std::unique_ptr<Backend,void(*)(Backend*)> b(createBackend(model,gas,{},GasExecutionOptions{},h,error),destroyBackend);assert(b);
 WallProgram program;program.interval.end=.01;program.surface=h.surface;WallKnot a,z;a.time=0;z.time=.01;
 WallFaceSample sample;sample.temperature=900;sample.primaryKind=ExchangeKind::GasSolid;a.faces={sample};z.faces={sample};program.knots={a,z};
 assert(beginGasWindow(*b,program,error));b->microTime=0;b->microDt=.01;assert(transport::CoupledTrialPolicy::begin(b.get())==0);
 auto&v=b->storage.hostView();v.rho[0]=1;v.Tgas[0]=600;v.p[0]=model.physics.species[0].R*600;v.gasSpecies.soundSpeed[0]=400;
 assert(applyCoupledFaces(b.get(),.01,0)==0);assert(std::abs(v.gasPhiRhoE[0]+300)<1e-10);
 // Simulate only the common donor factor here; no transport is emulated.
 v.gasPhiRhoE[0]*=.5;v.gasPhiRhoUx[0]*=.5;v.gasPhiRhoUy[0]*=.5;v.gasPhiRhoUz[0]*=.5;v.gasPhiRho[0]*=.5;
 for(int s=0;s<Ns;++s)v.gasSpecies.flux[s*b->nFaces]*=.5;
 assert(accountCoupledFaces(b.get(),.01)==0);assert(b->record.packets.size()==2);
 assert(std::abs(b->record.packets[0].energy-1.5)<1e-12);assert(std::abs(b->trialBudget.exchangeEnergy[GasParticipant]-1.5)<1e-12);
 assert(b->record.packets[0].consumerMask==ConsumeGas);
 assert(b->accepted.gas[0].energy==1e6);transport::CoupledTrialPolicy::rollback(b.get());assert(b->trial.gas[0].energy==1e6);
}
''')
    binary=tmp_path/'hooks'
    subprocess.run(compiler_flags(tmp_path)+[str(source),str(APP/'mesh/Geometry.C'),'-o',str(binary)],check=True)
    subprocess.run([str(binary)],check=True)
