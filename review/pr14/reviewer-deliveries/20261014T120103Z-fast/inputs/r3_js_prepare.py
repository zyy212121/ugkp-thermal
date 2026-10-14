from r3_runner import *
import runpy,re
fixture=runpy.run_path(str(ROOT/'applications/CHMT/tests/test_reacting_wall_storage.py'))['storage_source']()
prefix=fixture[:fixture.index(' chmt::SharedGasDeviceStorage storage;')]
assert 'cudaMemcpy' not in prefix and 'cuda_runtime.h' not in prefix
prefix=prefix.replace('int main(){','int main(){\n for(int variant=0;variant<2;++variant){')
prefix=prefix.replace('w.rho=1;w.temperature=700;w.Y[0]=1;w.velocity={5,0,0};','w.rho=variant?0.8:1;w.temperature=variant?900:700;w.Y[0]=1;w.velocity={variant?25.:5.,0,0};')
body=r'''
 p.enableSst=true;p.minDt=1e-12;p.maxDt=.001;p.cfl=.4;h.sst.assign(3,{1,10});m.wallDistance.assign(3,.5);m.boundarySst.resize(m.owner.size());
 for(int f=0;f<int(m.owner.size());++f)m.faceIds.push_back(f);m.oldVolumes=m.volumes;m.oldPoints=m.referencePoints=m.points;
 h.solid.resize(1);h.solid[0].condensed[0]=1;h.solidMesh.volumes={1};h.solidMesh.owner={0};h.solidMesh.neighbour={-1};
 h.surface.solidFace={0};h.surface.solidCell={0};h.surface.persistentId={7};h.surface.gasDistance={.5};h.surface.solidDistance={.5};h.surface.normal={{0,0,1}};
 chmt::GasExecutionOptions options;options.turbulenceModel=3;
 auto owner=[&](double js){std::unique_ptr<chmt::Backend,void(*)(chmt::Backend*)> b(chmt::createBackend(model,canonical,{},options,h,error),chmt::destroyBackend);if(!b)throw std::runtime_error(error);
  chmt::WallProgram program;program.interval.end=.01;program.surface=h.surface;
  chmt::WallKnot a,z;a.time=0;z.time=.01;chmt::WallFaceSample w;w.temperature=600;w.speciesRate[0]=js;w.condensedRate[0]=js;a.faces=z.faces={w};program.knots={a,z};
  if(!chmt::beginGasWindow(*b,program,error))throw std::runtime_error(error);b->microBase=b->trial;b->microDt=variant?.00005:.0001;b->microTime=0;if(chmt::prepareCoupledGeometry(b.get())!=0)throw std::runtime_error(b->error);return b;};
 auto read=[&](const double* ptr,int count){std::vector<double> values(count);auto rc=cudaMemcpy(values.data(),ptr,count*sizeof(double),cudaMemcpyDeviceToHost);if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));if(cudaDeviceSynchronize()!=cudaSuccess)throw std::runtime_error("sync failed");return values;};
 auto b=owner(.013);const auto&v=b->storage.hostView();const auto oldK=read(v.rhoK,3),oldW=read(v.rhoOmega,3);const double referenceDt=variant?.00005:.0001;
 for(double js:{.013,.021,.034}){for(auto&k:b->program.knots)k.faces[0].speciesRate[0]=js;chmt::transport::GasCourantMaxima co;
  if(!chmt::currentGasCourant(*b,referenceDt,co))throw std::runtime_error(chmt::transport::lastError);const auto phi=read(v.gasPhiRho,int(m.owner.size())),area=read(v.magSf,int(m.owner.size()));
  assert(phi[wall]==-area[wall]*js);assert(read(v.rhoK,3)==oldK&&read(v.rhoOmega,3)==oldW);assert(!b->storage.wallMassPublished());
  auto fresh=owner(js);chmt::transport::GasCourantMaxima ref;if(!chmt::currentGasCourant(*fresh,referenceDt,ref))throw std::runtime_error(chmt::transport::lastError);
  const auto&fv=fresh->storage.hostView();assert(std::memcmp(&co.maximum,&ref.maximum,sizeof(double))==0);assert(read(v.gasFluxPositivityScale,3)==read(fv.gasFluxPositivityScale,3));assert(read(v.gasDiffusionNumber,3)==read(fv.gasDiffusionNumber,3));
  std::printf("R3JS fresh variant=%d Js=%a area=%a phi=%a dt=%a Co=%a flux=%a,%a,%a diffusion=%a,%a,%a\n",variant,js,area[wall],phi[wall],referenceDt,co.maximum,read(v.gasFluxPositivityScale,3)[0],read(v.gasFluxPositivityScale,3)[1],read(v.gasFluxPositivityScale,3)[2],read(v.gasDiffusionNumber,3)[0],read(v.gasDiffusionNumber,3)[1],read(v.gasDiffusionNumber,3)[2]);
 }
 if(chmt::prepareCoupledBoundaryLayer(b.get(),referenceDt,0)!=0)throw std::runtime_error(b->error);
 assert(b->storage.wallMassPublished());auto lagged=read(v.gasBoundaryLayer.speciesFlux,1)[0];assert(lagged==-.034);
 for(auto&k:b->program.knots)k.faces[0].speciesRate[0]=.055;chmt::transport::GasCourantMaxima co;if(!chmt::currentGasCourant(*b,referenceDt,co))throw std::runtime_error(chmt::transport::lastError);
 assert(read(v.gasPhiRho,int(m.owner.size()))[wall]==lagged);
 if(chmt::prepareCoupledBoundaryLayer(b.get(),referenceDt,0)!=0)throw std::runtime_error(b->error);assert(read(v.gasBoundaryLayer.speciesFlux,1)[0]==-.055);
 std::printf("R3JS lagged variant=%d published=%a Co=%a nextactualStage=%a dt=%a\n",variant,lagged,co.maximum,read(v.gasBoundaryLayer.speciesFlux,1)[0],referenceDt);
 }
 std::puts("PASS actual CUDA cold-start Js fresh references and lagged stage consumer two changed flow/dt controls");
}
'''
source=D/'inputs/cold-start-Js-actual-native-S310.cu';source.write_text('#include "gpu/Backend.cu"\n#include <cassert>\n#include <cstring>\n'+prefix+body)
with (D/'TEST_PLAN.md').open('a') as h:h.write('\nS310 cold-start Js/history-Co缺口：只取现有host-shim test的3个真实unitcube mesh/物理输入配方，独立CUDA编译直接include完整正式Backend.cu及其原native GPU kernels/SharedGasDeviceStorage，不包含shim/stub或host模拟kernel。hostowner与device pointers分开，每次明确D2H/cudaDeviceSynchronize；真实currentCo依次Js.013→.021→.034，每值与新的nativeowner/device fresh reference bit比较实际Co/前处理数组，assert phi=-area*Js与K/omega不改；真实stage发布.034，Co对.055保留原lagged消费，下一实际stage消费.055。两variant改变rho/T/U(1/700/5→.8/900/25)和dt(1e-4→5e-5)。不人为ready，不改K/omega或生产策略。安全copy错误链已证明范围复用，本newfixture不冒称全资源恢复。compile900/run300，保留失败不调阈值。\n')
binary=ROOT/'work/fixtures/cold-start-Js-S310';t=run('BUILD-COLD-START-JS-S310',cmd('nvcc','-std=c++17','-O2','-arch=sm_89','-DUGKWP_GAS_SPECIES=2','-I'+str(ROOT/'applications/CHMT'),'-I'+str(ROOT),'-I'+str(ROOT/'common'),'-I'+str(ROOT/'common/gasNumerics'),source,ROOT/'applications/CHMT/mesh/Geometry.C','-o',binary),[4,9,12,14,17,26,29],900,'ActualfullnativeGPU Backend source; noCUDAstubs, nohostkernelrewrite.',inputs=[source,ROOT/'applications/CHMT/gpu/Backend.cu'],kind='BUILD',variant='SUPPORT_S310');assert t['exit_code']==0
