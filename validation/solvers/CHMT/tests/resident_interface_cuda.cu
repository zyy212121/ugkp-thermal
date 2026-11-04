#include <cuda_runtime.h>
#include "tests/TestSupport.H"
#include "gasTransport/GasStateView.H"
#include "gasTransport/GasBoundaryLayerModelState.H"
#include "gasNumerics/GpuSstAlgebra.cuh"
#include <chrono>
#include <cstdio>
namespace chmt { struct SharedGasDeviceView:ugkwp::GasStateView<Real,Real,ugkwp::SstCoefficients> {
 ugkwp::GasSpeciesState<Real,Ns> gasSpecies;
 ugkwp::GasBoundaryLayerModelState<Real,Ns> gasBoundaryLayerModel;
}; }
#include "gpu/CoupledSurfaceTransfer.cuh"
#include "gpu/CoupledSurfacePipeline.cuh"
using namespace chmt;
void cudaCheck(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
template<class T>T* alloc(int n){T* p=nullptr;cudaCheck(cudaMallocManaged(&p,n*sizeof(T)));for(int i=0;i<n;++i)new(p+i) T{};return p;}
__global__ void limit(SharedGasDeviceView* v,int count,Real scale){const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=count)return;
 v->gasPhiRho[f]*=scale;v->gasPhiRhoUx[f]*=scale;v->gasPhiRhoUy[f]*=scale;v->gasPhiRhoUz[f]*=scale;v->gasPhiRhoE[f]*=scale;
 for(int s=0;s<Ns;++s)v->gasSpecies.flux[s*count+f]*=scale;}
__global__ void sampleWallProgram(const WallFaceSample* program,int count,GasWallProgramPosition position,WallFaceSample* out){
 const int slot=blockIdx.x*blockDim.x+threadIdx.x;if(slot<count)out[slot]=sampleResidentGasWallFace(program,count,position,slot);
}
void packetNear(const ExchangePacket& a,const ExchangePacket& b){
 using chmt_test::near;near(a.mass,b.mass,"mass");near(a.energy,b.energy,"energy");
 near(a.momentum.x,b.momentum.x,"momentum x");near(a.momentum.y,b.momentum.y,"momentum y");near(a.momentum.z,b.momentum.z,"momentum z");
 near(a.conductive,b.conductive,"conductive");near(a.advective,b.advective,"advective");near(a.pressureWork,b.pressureWork,"sweep pressure work");near(a.viscousWork,b.viscousWork,"viscous work");near(a.liquidKineticAdvection,b.liquidKineticAdvection,"film kinetic");
 for(int s=0;s<Ns;++s){near(a.species[s],b.species[s],"species");near(a.pore[s],b.pore[s],"pore");near(a.poreSweep[s],b.poreSweep[s],"pore sweep");}
 for(int c=0;c<Nc;++c)near(a.condensed[c],b.condensed[c],"condensed");
 chmt_test::check(a.face==b.face&&a.kind==b.kind&&a.consumerMask==b.consumerMask,"identity/consumption");
}
int main(){cudaCheck(cudaDeviceSetLimit(cudaLimitStackSize,65536));constexpr int count=257;
 std::printf("BYTES sample=%zu flux=%zu input=%zu record=%zu endpoint=%zu wall=%zu wallInput=%zu wallOutput=%zu\n",
  sizeof(CoupledSurfaceSample),sizeof(GasQ),sizeof(CoupledSurfaceInput),sizeof(CoupledSurfaceRecord),
  sizeof(CoupledSurfaceDownload),sizeof(CoupledWallDownload),sizeof(ugkwp::gaswall::WallInput<Real,Ns>),sizeof(ugkwp::gaswall::WallOutput<Real,Ns>));const double dt=1e-4;
 auto* v=alloc<SharedGasDeviceView>(1);auto* p=alloc<PhysicsConfig>(1);*p=chmt_test::physics();
 p->gasMode=ugkwp::GasMode::MixtureFrozen;p->gasViscosity=2e-5;p->gasConductivity=.3;
 for(int s=0;s<Ns;++s)p->gasDiffusivity[s]=.01;
 auto bulk=chmt_test::gas(*p);v->nCells=1;v->nFaces=count;v->gasSpecies.mode=p->gasMode;
 auto* zero=alloc<double>(count);auto* owner=alloc<int>(count);v->faceOwner=owner;
 v->rho=alloc<double>(1);*v->rho=bulk.rho;v->p=alloc<double>(1);*v->p=bulk.pressure;v->Tgas=alloc<double>(1);*v->Tgas=bulk.temperature;
 v->Ux=alloc<double>(1);*v->Ux=bulk.velocity.x;v->Uy=alloc<double>(1);*v->Uy=bulk.velocity.y;v->Uz=alloc<double>(1);*v->Uz=bulk.velocity.z;
 v->gasSpecies.rho=alloc<double>(Ns);for(int s=0;s<Ns;++s)v->gasSpecies.rho[s]=bulk.rho*bulk.Y[s];
 v->gasSpecies.soundSpeed=alloc<double>(1);*v->gasSpecies.soundSpeed=bulk.soundSpeed;
 v->gradRhoX=v->gradRhoY=v->gradRhoZ=v->gradTX=v->gradTY=v->gradTZ=zero;
 v->gradPx=v->gradPy=v->gradPz=v->gradUxX=v->gradUxY=v->gradUxZ=zero;
 v->gradUyX=v->gradUyY=v->gradUyZ=v->gradUzX=v->gradUzY=v->gradUzZ=zero;
 v->gasSpecies.gradX=v->gasSpecies.gradY=v->gasSpecies.gradZ=zero;
 v->magSf=alloc<double>(count);v->Sfx=alloc<double>(count);v->Sfy=v->Sfz=zero;
 v->gasPhiRho=alloc<double>(count);v->gasPhiRhoUx=alloc<double>(count);v->gasPhiRhoUy=alloc<double>(count);v->gasPhiRhoUz=alloc<double>(count);v->gasPhiRhoE=alloc<double>(count);v->gasSpecies.flux=alloc<double>(Ns*count);
 auto* faces=alloc<int>(count);auto* in=alloc<CoupledSurfaceInput>(count);auto* wallProgram=alloc<WallFaceSample>(2*count);auto* records=alloc<CoupledSurfaceRecord>(count);
 auto* downloads=alloc<CoupledSurfaceDownload>(count);auto* wallDownloads=alloc<CoupledWallDownload>(count);
 auto* wallInputs=alloc<ugkwp::gaswall::WallInput<Real,Ns>>(count);auto* layers=alloc<ugkwp::gaswall::WallResidentOutput<Real,Ns>>(count);
 auto* exchanges=alloc<ugkwp::GasBoundaryLayerExchange<Real>>(count);auto* slots=alloc<int>(count);
 auto* wallStatuses=alloc<int>(count);auto* modelStatuses=alloc<ugkwp::gaswall::WallStatus>(count);
 v->gasBoundaryLayer.faceSlot=slots;v->gasBoundaryLayer.exchange=exchanges;v->gasBoundaryLayer.status=wallStatuses;
 v->gasBoundaryLayerModel.input=wallInputs;v->gasBoundaryLayerModel.output=layers;v->gasBoundaryLayerModel.status=modelStatuses;
 for(int i=0;i<count;++i){faces[i]=slots[i]=i;v->magSf[i]=1+.001*i;v->Sfx[i]=-v->magSf[i];
  in[i].area=.8*v->magSf[i];in[i].gasDistance=.2;in[i].normal={1,0,0};in[i].sweptVolume=i%2?2e-7:0;
  auto& id=in[i].identity;id.face=500+i;id.step=19;id.geometry=3;id.stage=1;id.gasCell=0;id.solidCell=0;id.filmFace=i;
  auto& w=wallProgram[i];w.temperature=600+i*.1;w.velocity={0,2,3};w.normalVelocity=.1;w.radiationFlux=30;
  w.speciesRate[0]=.0003;w.speciesRate[1]=.0001;w.poreRate[0]=.00004;w.poreRate[1]=-.00001;
  w.poreSweepRate[0]=.00005;w.poreSweepRate[1]=.00002;w.condensedRate[0]=.0004;
 }
 for(int variant=0;variant<4;++variant){
  p->wallModel.family=variant==2?ugkwp::gaswall::WallFamily::BoundaryLayer:ugkwp::gaswall::WallFamily::LowRe;
  p->wallModel.model=ugkwp::gaswall::BoundaryLayerModel::ConstantTransport;
  v->gasBoundaryLayer.enabled=variant==2;
  if(variant==1){p->gasViscosityModel=ugkwp::MolecularViscosityModel::SutherlandWilke;p->gasViscositySpecies=Ns;
   for(int s=0;s<Ns;++s)p->gasSutherland[s]={2e-5+s*1e-6,300.,110.+s,0.028+s*.004};
   for(int a=0;a<Ns;++a)for(int b=0;b<Ns;++b){const auto ma=p->gasSutherland[a].molarMass,mb=p->gasSutherland[b].molarMass;p->gasMassRatioQuarter[a*Ns+b]=std::pow(mb/ma,.25);p->gasWilkeDenominator[a*Ns+b]=std::sqrt(8*(1+ma/mb));}
  }
  if(variant==3){
   p->gasMode=v->gasSpecies.mode=ugkwp::GasMode::SingleLegacy;p->singleGasSpecies=0;
   bulk.Y[0]=1;bulk.Y[1]=0;v->gasSpecies.rho[0]=bulk.rho;v->gasSpecies.rho[1]=0;
   v->gammaGas=1.4;
   for(int i=0;i<count;++i){wallProgram[i].speciesRate[0]=.0004;wallProgram[i].speciesRate[1]=0;
    wallProgram[i].poreRate[0]=.00003;wallProgram[i].poreRate[1]=0;wallProgram[i].poreSweepRate[1]=0;}
  }
  for(int i=0;i<count;++i){
   wallProgram[count+i]=wallProgram[i];wallProgram[count+i].temperature+=10;
   wallProgram[count+i].radiationFlux+=3;
   for(int s=0;s<Ns;++s)wallProgram[count+i].speciesRate[s]*=1.1;
   for(int c=0;c<Nc;++c)wallProgram[count+i].condensedRate[c]*=1.1;
  }
  auto* interpolated=alloc<WallFaceSample>(count);
  const GasWallProgramPosition fractionalPosition{0,1,.1234567890123};
  sampleWallProgram<<<(count+127)/128,128>>>(wallProgram,count,fractionalPosition,interpolated);
  cudaCheck(cudaDeviceSynchronize());
  for(int i=0;i<count;++i){const auto expected=sampleResidentGasWallFace(wallProgram,count,fractionalPosition,i);
   chmt_test::check(interpolated[i].temperature==expected.temperature&&interpolated[i].radiationFlux==expected.radiationFlux,"identical wall interpolation rounding");
   for(int s=0;s<Ns;++s)chmt_test::check(interpolated[i].speciesRate[s]==expected.speciesRate[s],"identical wall species interpolation rounding");
  }
  cudaCheck(cudaFree(interpolated));
  std::vector<ExchangePacket> primary(count),pore(count);std::vector<double> rad(count);
  for(int stage=0;stage<3;++stage){const double weight=stage==2?2./3:1./6,scale=stage==1?.73:1.;
   const GasWallProgramPosition position=stage==1?GasWallProgramPosition{1,1,0}:GasWallProgramPosition{0,1,stage==0?0.:.5};
   for(int i=0;i<count;++i){auto& x=in[i];const auto w=sampleResidentGasWallFace(wallProgram,count,position,i);
    GasWallInput input;input.bulk=bulk;input.temperature=w.temperature;input.gasDistance=x.gasDistance;input.area=x.area;input.gasArea=v->magSf[i];input.dt=dt;input.normal=x.normal;input.velocity=w.velocity;input.normalSpeed=w.normalVelocity;input.sweptVolume=x.sweptVolume;
    for(int s=0;s<Ns;++s){input.speciesRate[s]=w.speciesRate[s];input.poreRate[s]=w.poreRate[s];input.poreSweepRate[s]=w.poreSweepRate[s];}
    for(int c=0;c<Nc;++c)input.condensedRate[c]=w.condensedRate[c];
    if(variant==2){auto& wi=wallInputs[i];wi.pressure=bulk.pressure;wi.matching.temperature=bulk.temperature;wi.matching.velocity[0]=bulk.velocity.x;wi.matching.velocity[1]=bulk.velocity.y;wi.matching.velocity[2]=bulk.velocity.z;
     for(int s=0;s<Ns;++s)wi.matching.massFraction[s]=bulk.Y[s];
     auto& layer=layers[i];layer.traceTemperature=input.temperature;Real mass=0;
     for(int s=0;s<Ns;++s){layer.wallSpeciesFlux[s]=(input.speciesRate[s]+input.poreRate[s])*(input.area/input.gasArea);mass+=layer.wallSpeciesFlux[s];layer.traceMassFraction[s]=bulk.Y[s];}
     GasPrimitive trace;chmt_test::check(makeInterfacePrimitive(layer.traceMassFraction,input.temperature,bulk.pressure,input.velocity,input.normal,input.normalSpeed,mass,*p,trace),"prepared primitive");
     layer.traceVelocity[0]=trace.velocity.x;layer.traceVelocity[1]=trace.velocity.y;layer.traceVelocity[2]=trace.velocity.z;layer.conductiveHeatFlux=-50;layer.traction[0]=.03;layer.traction[1]=-.01;layer.traction[2]=.02;exchanges[i].ready=true;
     input.preparedLayer=&layer;input.preparedMatching=&wi.matching;input.preparedMatchingPressure=&wi.pressure;input.wallContext.matching=wi.matching;input.wallContext.matchingPressure=wi.pressure;
     input.wallContext.mechanicalPressure=bulk.pressure+17;const Vec3 n=-input.normal,traction={layer.traction[0],layer.traction[1],layer.traction[2]};
     const Vec3 force=(-mass*trace.velocity+n*input.wallContext.mechanicalPressure+traction)*input.gasArea;
     v->gasPhiRhoUx[i]=force.x;v->gasPhiRhoUy[i]=force.y;v->gasPhiRhoUz[i]=force.z;
    }
    GasWallResult expected;chmt_test::check(evaluateGasWall(input,x.identity,*p,expected),"CPU wall oracle");
    if(stage==0){primary[i]=scaledIntervalPacket(expected.primary,0);pore[i]=scaledIntervalPacket(expected.pore,0);primary[i].consumerMask=pore[i].consumerMask=ConsumeGas;}
    addIntervalPacket(primary[i],scaledIntervalPacket(expected.primary,weight*scale));addIntervalPacket(pore[i],scaledIntervalPacket(expected.pore,weight*scale));rad[i]+=weight*dt*w.radiationFlux*x.area;
   }
   applyCoupledSurfaceKernel<<<(count+127)/128,128>>>(v,faces,count,in,wallProgram,position,p,dt,stage==0,19,3,records);
   limit<<<(count+127)/128,128>>>(v,count,scale);
   accountCoupledSurfaceKernel<<<(count+127)/128,128>>>(v,faces,count,in,wallProgram,position,p,dt,dt*weight,records);
   cudaCheck(cudaDeviceSynchronize());
  }
  packCoupledSurfaceKernel<<<(count+127)/128,128>>>(records,count,faces,{},downloads,variant==2?wallDownloads:nullptr);
  cudaCheck(cudaDeviceSynchronize());
  for(int i=0;i<count;++i){packetNear(downloads[i].primary,primary[i]);packetNear(downloads[i].pore,pore[i]);chmt_test::check(!records[i].failed,"device closure completion");packetNear(records[i].primary,primary[i]);packetNear(records[i].pore,pore[i]);chmt_test::near(records[i].radiationEnergy,rad[i],"radiation ledger");
   const auto a=packetDelta(records[i].primary),b=packetDelta(records[i].pore);chmt_test::near(a.gas.energy+a.solid.energy+b.gas.energy+b.solid.energy,0,"energy closure");
   Real species=0;for(int s=0;s<Ns;++s)species+=records[i].primary.species[s]+records[i].pore.species[s];chmt_test::near(species,records[i].primary.mass+records[i].pore.mass,"mass/species closure");
  }
  std::printf("PASS resident interface variant=%d faces=%d SSPRK3 mass species momentum energy pore-sweep radiation\n",variant,count);
  if(variant==2){
   exchanges[0].ready=false;wallStatuses[0]=37;modelStatuses[0].node=5;modelStatuses[0].iteration=11;modelStatuses[0].residual=.125;
   applyCoupledSurfaceKernel<<<(count+127)/128,128>>>(v,faces,count,in,wallProgram,GasWallProgramPosition{0,1,.5},p,dt,true,19,3,records);
   packCoupledSurfaceKernel<<<(count+127)/128,128>>>(records,count,faces,{},downloads,wallDownloads);
   cudaCheck(cudaDeviceSynchronize());
   chmt_test::check(downloads[0].failed==2&&wallDownloads[0].transportCode==37
    &&wallDownloads[0].status.node==5&&wallDownloads[0].status.iteration==11&&wallDownloads[0].status.residual==.125,
    "preserve existing wall failure diagnostic at endpoint");
  }
 }
}
