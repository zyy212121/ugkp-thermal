#include <cuda_runtime.h>
#include "gpu/MaterialSurfaceKernels.cuh"
#include "tests/TestSupport.H"
#include "gpu/ScalarWorkLaunch.cuh"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <vector>
using namespace chmt;
void cudaCheck(cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(2);}}
template<class T>T* allocate(int n){T* p=nullptr;cudaCheck(cudaMallocManaged(&p,n*sizeof(T)));for(int i=0;i<n;++i)new(p+i)T{};return p;}
void compare(const SurfacePhysicsResult& a,const SurfacePhysicsResult& b){
    using chmt_test::near;
    near(a.gasState.rho,b.gasState.rho,"trace rho");near(a.gasState.temperature,b.gasState.temperature,"trace T");
    near(a.gasState.pressure,b.gasState.pressure,"trace pressure");near(a.gasState.soundSpeed,b.gasState.soundSpeed,"trace sound speed");
    for(int s=0;s<Ns;++s){near(a.gasSpecies[s],b.gasSpecies[s],"surface species");near(a.poreSpecies[s],b.poreSpecies[s],"pore total");
        near(a.poreDarcy[s],b.poreDarcy[s],"pore physical");near(a.poreSweep[s],b.poreSweep[s],"pore swept");near(a.gasState.Y[s],b.gasState.Y[s],"trace composition");}
    for(int c=0;c<Nc;++c)near(a.condensed[c],b.condensed[c],"surface condensed");
    const double av[]={a.gasEnergy,a.poreEnergy,a.gasAdvective,a.gasConductive,a.gasPressureWork,a.gasViscousWork,a.topTemperature,a.bottomTemperature,a.solidSpeed,a.topSpeed,a.radiationFlux,a.thermalResidual,a.gasWallSpeed,a.solidWallSpeed,a.solidRecessionResidual};
    const double bv[]={b.gasEnergy,b.poreEnergy,b.gasAdvective,b.gasConductive,b.gasPressureWork,b.gasViscousWork,b.topTemperature,b.bottomTemperature,b.solidSpeed,b.topSpeed,b.radiationFlux,b.thermalResidual,b.gasWallSpeed,b.solidWallSpeed,b.solidRecessionResidual};
    for(unsigned i=0;i<sizeof(av)/sizeof(av[0]);++i)if(i!=11)
        near(av[i],bv[i],"surface energy/temperature/velocity",2e-9,2e-10);

    const Vec3 va[]={a.gasState.velocity,a.shear,a.gasShear,a.traction},vb[]={b.gasState.velocity,b.shear,b.gasShear,b.traction};
    for(int i=0;i<4;++i){near(va[i].x,vb[i].x,"vector x",2e-9);near(va[i].y,vb[i].y,"vector y",2e-9);near(va[i].z,vb[i].z,"vector z",2e-9);}
    chmt_test::check(!a.wet&&!b.wet,"dry execution retains zero film storage");
}
// Compare named state fields, never padding; successful statuses deliberately
// retain a NaN residual when no wall iteration was needed.
void identical(Real a,Real b){chmt_test::check(a==b||(std::isnan(a)&&std::isnan(b)),"launch preserves scalar result");}
void identical(Vec3 a,Vec3 b){identical(a.x,b.x);identical(a.y,b.y);identical(a.z,b.z);}
void identical(const SurfacePhysicsResult& a,const SurfacePhysicsResult& b){
    identical(a.gasState.rho,b.gasState.rho);identical(a.gasState.temperature,b.gasState.temperature);
    identical(a.gasState.pressure,b.gasState.pressure);identical(a.gasState.soundSpeed,b.gasState.soundSpeed);
    identical(a.gasState.velocity,b.gasState.velocity);
    for(int s=0;s<Ns;++s){identical(a.gasState.Y[s],b.gasState.Y[s]);identical(a.gasSpecies[s],b.gasSpecies[s]);
        identical(a.poreSpecies[s],b.poreSpecies[s]);identical(a.poreDarcy[s],b.poreDarcy[s]);identical(a.poreSweep[s],b.poreSweep[s]);}
    for(int c=0;c<Nc;++c)identical(a.condensed[c],b.condensed[c]);
    const Real SurfacePhysicsResult::* fields[]={&SurfacePhysicsResult::phaseMass,
        &SurfacePhysicsResult::gasEnergy,&SurfacePhysicsResult::poreEnergy,&SurfacePhysicsResult::phaseEnergy,
        &SurfacePhysicsResult::gasAdvective,&SurfacePhysicsResult::gasConductive,&SurfacePhysicsResult::gasPressureWork,&SurfacePhysicsResult::gasViscousWork,
        &SurfacePhysicsResult::phaseAdvective,&SurfacePhysicsResult::phaseConductive,&SurfacePhysicsResult::phasePressureWork,&SurfacePhysicsResult::phaseViscousWork,
        &SurfacePhysicsResult::topKineticAdvection,&SurfacePhysicsResult::bottomKineticAdvection,
        &SurfacePhysicsResult::topTemperature,&SurfacePhysicsResult::bottomTemperature,&SurfacePhysicsResult::solidSpeed,&SurfacePhysicsResult::topSpeed,
        &SurfacePhysicsResult::radiationFlux,&SurfacePhysicsResult::thermalResidual,&SurfacePhysicsResult::gasWallSpeed,&SurfacePhysicsResult::solidWallSpeed,
        &SurfacePhysicsResult::liquidTopNormal,&SurfacePhysicsResult::liquidBottomNormal,&SurfacePhysicsResult::solidRecessionResidual,&SurfacePhysicsResult::filmVolumeResidual};
    for(auto field:fields)identical(a.*field,b.*field);
    identical(a.shear,b.shear);identical(a.gasShear,b.gasShear);identical(a.traction,b.traction);
    chmt_test::check(a.wet==b.wet,"launch preserves wet state");
}
void identical(const WallFaceSample& a,const WallFaceSample& b){
    identical(a.temperature,b.temperature);identical(a.velocity,b.velocity);
    chmt_test::check(a.primaryKind==b.primaryKind,"launch preserves exchange kind");
    for(int s=0;s<Ns;++s){identical(a.speciesRate[s],b.speciesRate[s]);identical(a.poreRate[s],b.poreRate[s]);identical(a.poreSweepRate[s],b.poreSweepRate[s]);}
    for(int c=0;c<Nc;++c)identical(a.condensedRate[c],b.condensedRate[c]);
    identical(a.radiationFlux,b.radiationFlux);identical(a.solidNormalVelocity,b.solidNormalVelocity);
    identical(a.normalVelocity,b.normalVelocity);identical(a.materialEffusivity,b.materialEffusivity);
}
void identical(const MaterialSurfaceStatus& a,const MaterialSurfaceStatus& b){
    chmt_test::check(a.success==b.success&&a.numericalFailure==b.numericalFailure,"launch preserves completion status");
    chmt_test::check(a.device.code==b.device.code&&a.device.index==b.device.index&&a.device.location==b.device.location,"launch preserves device status");
    identical(a.device.value,b.device.value);
    chmt_test::check(a.wall.code==b.wall.code&&a.wall.node==b.wall.node&&a.wall.iteration==b.wall.iteration,"launch preserves wall status");
    identical(a.wall.residual,b.wall.residual);
}
__global__ void visitScalarWorker(int workers,unsigned* visits){
    const int worker=blockIdx.x*blockDim.x+threadIdx.x;if(worker<workers)atomicAdd(visits+worker,1u);
}
void scalarLaunchCoverage(){
    for(int workers:{0,1,3,31,32,33,97,4097}){
        auto* storage=allocate<unsigned>(workers+2);storage[0]=storage[workers+1]=719;
        ugkp::gpu::ScalarWorkLaunch launch;cudaCheck(ugkp::gpu::scalarWorkLaunch(visitScalarWorker,workers,launch));
        if(workers){chmt_test::check(launch.blocks>0&&launch.threads>0,"nonempty scalar launch");
            visitScalarWorker<<<launch.blocks,launch.threads>>>(workers,storage+1);cudaCheck(cudaGetLastError());cudaCheck(cudaDeviceSynchronize());}
        else chmt_test::check(launch.blocks==0,"empty scalar launch");
        for(int i=0;i<workers;++i)chmt_test::check(storage[i+1]==1,"each scalar worker visited exactly once");
        chmt_test::check(storage[0]==719&&storage[workers+1]==719,"scalar launch guard values");
        std::printf("PASS scalar launch workers=%d blocks=%d threads=%d\n",workers,launch.blocks,launch.threads);cudaCheck(cudaFree(storage));
    }
}
int main(int argc,char** argv){
    int beginVariant=0,endVariant=6;
    if(argc==3&&std::strcmp(argv[1],"--variant")==0){
        char* end=nullptr;const long variant=std::strtol(argv[2],&end,10);
        if(!*argv[2]||*end||variant<0||variant>=endVariant){std::fprintf(stderr,"variant must be an integer in [0,5]\n");return 2;}
        beginVariant=int(variant);endVariant=beginVariant+1;
    }else if(argc!=1){std::fprintf(stderr,"usage: resident-material-surface [--variant 0..5]\n");return 2;}
    cudaCheck(cudaDeviceSetLimit(cudaLimitStackSize,65536));scalarLaunchCoverage();constexpr int count=7,workers=3;
    auto* p=allocate<PhysicsConfig>(1);auto* inputs=allocate<SurfacePhysicsInput>(count);
    auto* cells=allocate<int>(count);auto* solid=allocate<SolidQ>(count);auto* volumes=allocate<double>(count);auto* rates=allocate<SolidQ>(count);
    auto* active=allocate<unsigned char>(count);auto* result=allocate<SurfacePhysicsResult>(count);auto* walls=allocate<WallFaceSample>(count);auto* status=allocate<MaterialSurfaceStatus>(count);
    using Workspace=ugkwp::gaswall::WallWorkspace<double,Ns,24>;auto* scratch=allocate<Workspace>(workers);auto cpuScratch=std::make_unique<Workspace>();
    auto* qDistance=allocate<double>(2);auto* qWeight=allocate<double>(2);qDistance[0]=.125;qDistance[1]=.375;qWeight[0]=qWeight[1]=.5;
    MaterialSurfaceView view;view.count=count;view.inputs=inputs;view.solidCells=cells;view.solid=solid;view.volumes=volumes;view.materialRate=rates;view.active=active;
    view.physicsOut=result;view.wallOut=walls;view.status=status;view.wallWorkspaces=scratch;view.workspaceStride=sizeof(Workspace);view.workspaceCapacity=24;view.workers=workers;
    for(int variant=beginVariant;variant<endVariant;++variant){
        *p={};p->gasMode=ugkwp::GasMode::MixtureFrozen;p->gasConductivity=2;p->gasViscosity=.01;
        for(auto& x:p->species){x.R=287;x.cp0=1000;x.Tmin=100;x.Tmax=3000;}
        for(auto& x:p->condensed){x.rho=1000;x.cp0=1000;x.conductivity=2;x.Tmin=100;x.Tmax=3000;}
        p->material.enableSurfaceReactions=variant>0;p->material.nSurfaceReactions=1;
        auto& reaction=p->material.surfaceReactions[0];reaction.A=.035714285714285712;reaction.condensedNu[0]=-.028;reaction.gasNu[0]=.028;
        p->material.enablePoreOutflow=variant==1;p->permeability=variant==1?1e-12:0;p->poreViscosity=1e-5;
        for(int s=0;s<Ns;++s)p->gasDiffusivity[s]=1e-3;
        if(variant>=2){p->wallModel.family=ugkwp::gaswall::WallFamily::BoundaryLayer;
            p->wallModel.model=variant==2?ugkwp::gaswall::BoundaryLayerModel::ConstantTransport:ugkwp::gaswall::BoundaryLayerModel::ReactingSst;
            if(variant==5)p->wallModel.model=ugkwp::gaswall::BoundaryLayerModel::EquilibriumOde;
            p->wallModel.nodes=8;p->wallModel.maxIterations=60;p->wallModel.relativeTolerance=1e-8;p->wallModel.absoluteTolerance=1e-10;}
        if(variant==4){p->gasViscosityModel=ugkwp::MolecularViscosityModel::SutherlandWilke;p->gasViscositySpecies=Ns;
            for(int s=0;s<Ns;++s)p->gasSutherland[s]={2e-5+s*1e-6,300.,110.+s,.028+s*.004};
            for(int a=0;a<Ns;++a)for(int b=0;b<Ns;++b){const double ma=p->gasSutherland[a].molarMass,mb=p->gasSutherland[b].molarMass;
                p->gasMassRatioQuarter[a*Ns+b]=std::pow(mb/ma,.25);p->gasWilkeDenominator[a*Ns+b]=std::sqrt(8*(1+ma/mb));}}
        std::vector<SurfacePhysicsResult> expected(count);std::vector<WallFaceSample> expectedWall(count);
        for(int c=0;c<count;++c){solid[c]={};rates[c]={};solid[c].condensed[0]=.5+.01*c;solid[c].pore[0]=.0001;volumes[c]=.001;
            solid[c].energy=solid[c].condensed[0]*condensedE(p->condensed[0],600+c)+solid[c].pore[0]*speciesE(p->species[0],600+c);rates[c].condensed[0]=-.01*c;}
        for(int f=0;f<count;++f){inputs[f]={};auto& in=inputs[f];cells[f]=(f*3)%count;active[f]=f!=count-1;
            in.hasSolid=true;in.area=2;in.gasArea=in.solidArea=5;in.gasDistance=in.solidDistance=.5;in.dt=1e-4;in.normal=in.gasNormal=in.solidNormal={1,0,0};
            in.bulk.rho=1;in.bulk.temperature=813.37+f;in.bulk.pressure=287*in.bulk.temperature;in.bulk.velocity={0,2+.2*f,0};in.bulk.Y[0]=.7;in.bulk.Y[1]=.3;
            in.gasInventory=conservativeGas(in.bulk,1,*p);in.aux.pressure=in.bulk.pressure;
            auto& context=in.wallContext;context.matchingPressure=in.bulk.pressure;context.mechanicalPressure=.9*in.bulk.pressure;
            context.matchingDistance=.5;context.ownerDistance=.25;context.matching.temperature=in.bulk.temperature;
            context.matching.velocity[1]=in.bulk.velocity.y;context.matching.massFraction[0]=.7;context.matching.massFraction[1]=.3;
            context.quadrature.distance=qDistance;context.quadrature.volumeWeight=qWeight;context.quadrature.count=2;context.quadrature.volume=1;context.quadrature.firstMoment=.25;
            if(!active[f])continue;
            auto cpu=in;cpu.solid=cpu.solidBase=solid[cells[f]];cpu.solidVolume=volumes[cells[f]];cpu.solidLocalRate=rates[cells[f]];
            cpu.wallContext.workspace=cpuScratch.get();cpu.wallContext.workspaceCapacity=24;ugkwp::gaswall::WallStatus wallStatus;cpu.wallContext.failureStatus=&wallStatus;
            bool numerical=true;DeviceStatus device;
            if(!solveDrySurfaceInterface(cpu,*p,expected[f],&device,&numerical)){
                std::fprintf(stderr,"CPU surface failed variant=%d face=%d wall=%d numerical=%d\n",variant,f,int(wallStatus.code),numerical);return 3;}
            expectedWall[f]=dryWallFaceSample(cpu,expected[f],*p);
            // Device must read resident inventories, never these host placeholders.
            in.solid.energy=-1;in.solidVolume=-1;in.solidLocalRate.condensed[0]=1e30;
        }
        // Poison inactive output too: both launch plans must reset it, while
        // workers 0..2 serially own faces worker, worker+3, worker+6.
        for(int f=0;f<count;++f){result[f].gasEnergy=12345;walls[f].temperature=12345;status[f]={};}
        evaluateMaterialSurfaceKernel<<<1,32>>>(view,p);cudaCheck(cudaGetLastError());cudaCheck(cudaDeviceSynchronize());
        const std::vector<SurfacePhysicsResult> original(result,result+count);
        const std::vector<WallFaceSample> originalWalls(walls,walls+count);
        const std::vector<MaterialSurfaceStatus> originalStatus(status,status+count);
        for(int w=0;w<workers;++w)new(scratch+w)Workspace{};
        for(int f=0;f<count;++f){result[f]={};result[f].gasEnergy=12345;walls[f]={};walls[f].temperature=12345;status[f]={};}
        ugkp::gpu::ScalarWorkLaunch launch;cudaCheck(ugkp::gpu::scalarWorkLaunch(evaluateMaterialSurfaceKernel,workers,launch));
        evaluateMaterialSurfaceKernel<<<launch.blocks,launch.threads>>>(view,p);cudaCheck(cudaGetLastError());cudaCheck(cudaDeviceSynchronize());
        for(int f=0;f<count;++f){
            identical(result[f],original[f]);identical(walls[f],originalWalls[f]);identical(status[f],originalStatus[f]);
            if(!status[f].success)std::fprintf(stderr,"CUDA surface failed variant=%d face=%d wall=%d numerical=%d\n",variant,f,int(status[f].wall.code),status[f].numericalFailure);
            chmt_test::check(status[f].success,"device surface solve");compare(result[f],expected[f]);
            chmt_test::near(walls[f].temperature,expectedWall[f].temperature,"wall T",2e-9);chmt_test::near(walls[f].normalVelocity,expectedWall[f].normalVelocity,"wall speed");
            for(int s=0;s<Ns;++s){chmt_test::near(walls[f].speciesRate[s],expectedWall[f].speciesRate[s],"wall species",2e-9);chmt_test::near(walls[f].poreRate[s],expectedWall[f].poreRate[s],"wall pore",2e-9);}
            if(active[f]){
                const auto energyGoal=[&](const SurfacePhysicsResult& r){
                    const double scale=std::abs(r.gasEnergy)+std::abs(r.poreEnergy)+std::abs(r.phaseEnergy)+std::abs(r.radiationFlux);
                    return p->tolerances.absoluteEnergy/(inputs[f].area*inputs[f].dt)+p->tolerances.relativeEnergy*scale;
                };
                std::printf("ROOT residual variant=%d face=%d CPU=%.17g GPU=%.17g difference=%.17g CPUbudget=%.17g GPUbudget=%.17g\n",variant,f,expected[f].thermalResidual,result[f].thermalResidual,result[f].thermalResidual-expected[f].thermalResidual,energyGoal(expected[f]),energyGoal(result[f]));
                // The existing solver warns and returns on a stalled bracket.
                // Keep that CPU outcome visible instead of imposing a new gate.
                const bool cpuConverged=std::abs(expected[f].thermalResidual)<=energyGoal(expected[f]);
                if(cpuConverged)chmt_test::check(std::abs(result[f].thermalResidual)<=energyGoal(result[f]),"GPU preserves CPU thermal root convergence");
                chmt_test::check(std::abs(result[f].thermalResidual-expected[f].thermalResidual)<=energyGoal(result[f])+energyGoal(expected[f]),"paired thermal residual within original energy budgets");
                double mass=0;for(int s=0;s<Ns;++s)mass+=result[f].gasSpecies[s];for(int c=0;c<Nc;++c)mass-=result[f].condensed[c];
                chmt_test::near(mass,0,"surface reaction mass",1e-12,1e-13);
                chmt_test::near(result[f].gasEnergy,result[f].gasAdvective+result[f].gasConductive+result[f].gasPressureWork+result[f].gasViscousWork,"surface energy channels");}
        }
        std::printf("PASS resident material surface variant=%d faces=%d workers=%d blocks=%d threads=%d\n",variant,count,workers,launch.blocks,launch.threads);
    }
    cudaCheck(cudaFree(qWeight));cudaCheck(cudaFree(qDistance));cudaCheck(cudaFree(scratch));cudaCheck(cudaFree(status));cudaCheck(cudaFree(walls));cudaCheck(cudaFree(result));cudaCheck(cudaFree(active));cudaCheck(cudaFree(rates));cudaCheck(cudaFree(volumes));cudaCheck(cudaFree(solid));cudaCheck(cudaFree(cells));cudaCheck(cudaFree(inputs));cudaCheck(cudaFree(p));
}
