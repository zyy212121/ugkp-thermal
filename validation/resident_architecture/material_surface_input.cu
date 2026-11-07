#include "gpu/ResidentMaterialSurfaceInput.cuh"
#include "gpu/MaterialBackend.H"
#include "tests/TestSupport.H"
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
using namespace chmt;
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool x,const std::string& m){if(!x)throw std::runtime_error(m);}
template<class T>struct Device {
    T* p=nullptr;explicit Device(const std::vector<T>& v){if(v.empty())return;check(cudaMalloc(&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){if(p)cudaFree(p);}
    std::vector<T> read(std::size_t n)const{std::vector<T> v(n);if(n)check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
HostMesh mesh(int faces,bool solid){
    HostMesh m;const int cells=faces?std::min(faces,3):0;
    m.volumes.assign(cells,.001);m.cellCentres.assign(cells,{0,solid?-.25:.25,0});
    for(int f=0;f<faces;++f){m.owner.push_back(f%cells);m.neighbour.push_back(-1);m.areaVectors.push_back({0,(solid?1:-1)*(1+.1*f),0});
        m.faceCentres.push_back({double(f),0,0});m.boundaryKind.push_back(BoundaryKind::Interface);m.periodicPartner.push_back(-1);m.faceIds.push_back(f+100);}
    m.cellFaceOffsets.push_back(0);for(int c=0;c<cells;++c){for(int f=0;f<faces;++f)if(m.owner[f]==c){m.cellFaces.push_back(f);m.cellFaceSigns.push_back(1);}m.cellFaceOffsets.push_back(m.cellFaces.size());}
    return m;
}
double maximum=0,maximumResidualDifference=0,maximumResidualBudgetFraction=0;
void equal(double a,double b){if(a==b||(std::isnan(a)&&std::isnan(b)))return;
    const auto error=std::abs(a-b)/std::max(1.,std::max(std::abs(a),std::abs(b)));maximum=std::max(maximum,error);
    if(!(std::isfinite(error)&&error<2e-9)){std::cerr<<std::setprecision(17)<<"resident="<<a<<" native="<<b<<" scaledError="<<error<<"\n";throw std::runtime_error("native/resident surface field differs");}}
void equal(Vec3 a,Vec3 b){equal(a.x,b.x);equal(a.y,b.y);equal(a.z,b.z);}
void equal(const SurfacePhysicsResult& a,const SurfacePhysicsResult& b){
    equal(a.gasState.rho,b.gasState.rho);equal(a.gasState.temperature,b.gasState.temperature);equal(a.gasState.pressure,b.gasState.pressure);
    equal(a.gasState.soundSpeed,b.gasState.soundSpeed);equal(a.gasState.velocity,b.gasState.velocity);
    for(int s=0;s<Ns;++s){equal(a.gasSpecies[s],b.gasSpecies[s]);equal(a.poreSpecies[s],b.poreSpecies[s]);equal(a.poreDarcy[s],b.poreDarcy[s]);equal(a.poreSweep[s],b.poreSweep[s]);equal(a.gasState.Y[s],b.gasState.Y[s]);}
    for(int c=0;c<Nc;++c)equal(a.condensed[c],b.condensed[c]);
    const double x[]={a.gasEnergy,a.poreEnergy,a.phaseEnergy,a.gasAdvective,a.gasConductive,a.gasPressureWork,a.gasViscousWork,a.topTemperature,a.bottomTemperature,a.solidSpeed,a.topSpeed,a.phaseMass,a.phaseAdvective,a.phaseConductive,a.phasePressureWork,a.phaseViscousWork,a.topKineticAdvection,a.bottomKineticAdvection,a.radiationFlux,a.gasWallSpeed,a.solidWallSpeed,a.liquidTopNormal,a.liquidBottomNormal,a.solidRecessionResidual,a.filmVolumeResidual};
    const double y[]={b.gasEnergy,b.poreEnergy,b.phaseEnergy,b.gasAdvective,b.gasConductive,b.gasPressureWork,b.gasViscousWork,b.topTemperature,b.bottomTemperature,b.solidSpeed,b.topSpeed,b.phaseMass,b.phaseAdvective,b.phaseConductive,b.phasePressureWork,b.phaseViscousWork,b.topKineticAdvection,b.bottomKineticAdvection,b.radiationFlux,b.gasWallSpeed,b.solidWallSpeed,b.liquidTopNormal,b.liquidBottomNormal,b.solidRecessionResidual,b.filmVolumeResidual};
    for(std::size_t i=0;i<sizeof(x)/sizeof(*x);++i)equal(x[i],y[i]);equal(a.shear,b.shear);require(a.wet==b.wet,"wet state differs");equal(a.traction,b.traction);equal(a.gasShear,b.gasShear);
}
int main(){try{
    check(cudaDeviceSetLimit(cudaLimitStackSize,65536));std::size_t comparisons=0;
    for(int nf:{0,1,3,7})for(bool accepted:{false,true}){
        const int requests=2,count=nf*requests,cells=nf?std::min(nf,3):0,workers=std::max(1,std::min(3,count));
        PhysicsConfig physics;physics.gasMode=ugkwp::GasMode::MixtureFrozen;physics.gasConductivity=2;physics.gasViscosity=.01;
        for(auto& s:physics.species){s.R=287;s.cp0=1000;s.Tmin=100;s.Tmax=3000;}
        for(auto& s:physics.condensed){s.rho=1000;s.cp0=1000;s.conductivity=2;s.Tmin=100;s.Tmax=3000;}
        for(auto& d:physics.gasDiffusivity)d=1e-3;
        physics.wallModel.family=ugkwp::gaswall::WallFamily::BoundaryLayer;physics.wallModel.model=ugkwp::gaswall::BoundaryLayerModel::EquilibriumOde;
        physics.wallModel.nodes=8;physics.wallModel.maxIterations=60;physics.wallModel.relativeTolerance=1e-8;physics.wallModel.absoluteTolerance=1e-10;
        HostState host;host.gasMesh=mesh(nf,false);host.solidMesh=mesh(nf,true);host.solid.resize(cells);
        for(int c=0;c<cells;++c){host.solid[c].condensed[0]=.5;host.solid[c].energy=.5*1000*(600+c);host.gasVoidFraction.push_back(.3+.2*c);}
        std::vector<ugkp::gpu::geometry::SurfaceFaceGeometry> metrics(nf);std::vector<ugkwp::gaswall::WallMatchingGeometry> matching(nf);
        for(int f=0;f<nf;++f){host.surface.gasFace.push_back(f);host.surface.solidFace.push_back(f);host.surface.solidCell.push_back(f%cells);
            host.surface.area.push_back(1+.1*f);host.surface.normal.push_back({0,1,0});host.surface.gasDistance.push_back(.25);host.surface.solidDistance.push_back(.25);
            host.surface.baseVelocity.push_back({0,0,0});host.surface.ablating.push_back(f%2);
            metrics[f].area=1+.1*f;metrics[f].normal={0,1,0};metrics[f].gasDistance=metrics[f].solidDistance=.25;
            matching[f].ownerDistance=.25;matching[f].matchingDistance=.5;}
        std::vector<std::vector<GasPrimitive>> bulk(requests,std::vector<GasPrimitive>(nf));
        std::vector<std::vector<GasGradient>> gradients(requests,std::vector<GasGradient>(nf));
        std::vector<std::vector<GasWallClosureContext>> contexts(requests,std::vector<GasWallClosureContext>(nf));
        std::vector<std::vector<WallFaceSample>> rates(requests,std::vector<WallFaceSample>(nf));
        std::vector<CoupledSurfaceDownload> surface(count);std::vector<CoupledWallDownload> walls(count);std::vector<int> ownerCells;std::vector<unsigned char> active;
        for(int r=0;r<requests;++r)for(int f=0;f<nf;++f){auto& b=bulk[r][f];b.rho=1+.01*f;b.temperature=700+10*r;b.Y[0]=.7;b.Y[1]=.3;b.velocity={3,0,0};
            b.pressure=b.rho*287*b.temperature;b.soundSpeed=300;gradients[r][f].velocity[0].y=.01*(f+1);
            auto& context=contexts[r][f];context.matchingPressure=context.mechanicalPressure=b.pressure*1.01;
            context.matching.temperature=b.temperature;context.matching.velocity[0]=3;context.matching.massFraction[0]=.7;context.matching.massFraction[1]=.3;
            context.matchingDistance=.5;context.ownerDistance=.25;
            surface[r*nf+f].bulk=b;surface[r*nf+f].gradient=gradients[r][f];surface[r*nf+f].acceptedWall=rates[r][f];
            walls[r*nf+f].matching.state=context.matching;walls[r*nf+f].matching.pressure=context.matchingPressure;walls[r*nf+f].matching.mechanicalPressure=context.mechanicalPressure;
            ownerCells.push_back(f%cells);active.push_back(!(nf==7&&f==1));}
        // Inactive interface slots retain no physical matching state. They
        // must be skipped before reading it, as in the native producer.
        if(nf==7){host.surface.gasFace[1]=-1;for(int r=0;r<requests;++r){
            contexts[r][1].mechanicalPressure=-1;walls[r*nf+1].matching.mechanicalPressure=-1;}}
        std::vector<MaterialSurfaceRequest> request;
        for(int r=0;r<requests;++r)request.push_back({&bulk[r],&gradients[r],&contexts[r],accepted?&rates[r]:nullptr});
        std::string error;auto* backend=createMaterialBackend(error);require(backend,error);
        ugkp::diagnostics::HostSink sink;sink.fatal=[](void*,const ugkp::diagnostics::Record& record,const ugkp::diagnostics::SourceDescriptor*){
            std::ostringstream message;message<<std::setprecision(17)<<"native fatal code="<<int(record.code)<<" site="<<record.sourceSiteId
                <<" entity="<<record.entityIndex<<" observed="<<record.observed<<" reference="<<record.reference;
            throw std::runtime_error(message.str());};setMaterialDiagnosticsSink(*backend,sink);
        require(beginMaterialWindow(*backend,host,physics,error),error);
        std::vector<CpuSurfaceResult> native;bool numerical=false;
        require(evaluateResidentMaterialSurfaces(*backend,host,request,.001,false,false,native,error,numerical),error);
        destroyMaterialBackend(backend);
        Device<PhysicsConfig> p(std::vector<PhysicsConfig>{physics}),quiet(std::vector<PhysicsConfig>{physics});
        Device<SolidQ> q(host.solid);Device<double> gv(host.gasMesh.volumes),sv(host.solidMesh.volumes),voids(host.gasVoidFraction),area(std::vector<double>(cells,0));
        Device<int> go(host.gasMesh.owner),gf(host.surface.gasFace),sf(host.surface.solidFace),sc(host.surface.solidCell),oc(ownerCells);
        Device<Vec3> ga(host.gasMesh.areaVectors),sa(host.solidMesh.areaVectors),velocity(host.surface.baseVelocity);
        Device<unsigned char> ablate(host.surface.ablating),enabled(active);
        MaterialMeshView gas;gas.cells=cells;gas.faces=nf;gas.owner=go.p;gas.areaVectors=ga.p;gas.volumes=gv.p;
        MaterialMeshView solid;solid.cells=cells;solid.faces=nf;solid.areaVectors=sa.p;solid.volumes=sv.p;
        Device<MaterialMeshView> gm(std::vector<MaterialMeshView>{gas});
        ResidentMaterialState state;state.current=q.p;state.mesh=solid;state.candidateMesh=solid;
        Device<ResidentMaterialState> material(std::vector<ResidentMaterialState>{state});
        Device<ugkp::gpu::geometry::SurfaceFaceGeometry> geometry(metrics);Device<ugkwp::gaswall::WallMatchingGeometry> match(matching);
        Device<CoupledSurfaceDownload> history(surface);Device<CoupledWallDownload> wh(walls);
        Device<int> nr(std::vector<int>{requests});Device<std::size_t> records(std::vector<std::size_t>{0,1});
        Device<SurfacePhysicsInput> inputs{std::vector<SurfacePhysicsInput>(count)};Device<MaterialSurfaceInputError> inputErrors{std::vector<MaterialSurfaceInputError>(count)};
        Device<SurfacePhysicsResult> results{std::vector<SurfacePhysicsResult>(count)};Device<WallFaceSample> wallResults{std::vector<WallFaceSample>(count)};
        Device<MaterialSurfaceStatus> statuses{std::vector<MaterialSurfaceStatus>(count)};
        using Workspace=ugkwp::gaswall::WallWorkspace<double,Ns,24>;Device<Workspace> workspace{std::vector<Workspace>(workers)};
        Device<ugkwp::gaswall::WallOutput<Real,Ns>> teams{std::vector<ugkwp::gaswall::WallOutput<Real,Ns>>(workers)};
        MaterialSurfaceView prototype;prototype.inputs=inputs.p;prototype.solidCells=oc.p;prototype.active=enabled.p;
        prototype.physicsOut=results.p;prototype.wallOut=wallResults.p;prototype.status=statuses.p;prototype.workers=workers;
        prototype.wallWorkspaces=workspace.p;prototype.workspaceStride=sizeof(Workspace);prototype.workspaceCapacity=24;prototype.wallTeamOutput=teams.p;prototype.quietPhysics=quiet.p;
        Device<MaterialSurfaceView> prot(std::vector<MaterialSurfaceView>{prototype}),operation{std::vector<MaterialSurfaceView>(1)};
        ResidentMaterialStep step;step.active=true;step.dt=.001;Device<ResidentMaterialStep> timing(std::vector<ResidentMaterialStep>{step});
        std::vector<int> offsets{0},indices;for(int c=0;c<cells;++c){for(int f=0;f<nf;++f)if(host.surface.gasFace[f]>=0&&host.gasMesh.owner[f]==c)indices.push_back(f);offsets.push_back(indices.size());}
        Device<int> ro(offsets),ri(indices);
        MaterialSurfaceInputBinding binding;binding.faces=nf;binding.requests=nr.p;binding.records=records.p;binding.history={history.p,wh.p,nullptr,std::size_t(requests),std::size_t(nf),nullptr};
        binding.material=material.p;binding.gasMesh=gm.p;binding.surface=geometry.p;binding.gasFaces=gf.p;binding.solidFaces=sf.p;binding.solidCells=sc.p;
        binding.ablating=ablate.p;binding.baseVelocity=velocity.p;binding.baseGasVoidFraction=voids.p;binding.donorArea=area.p;
        binding.matchingGeometry=match.p;binding.matching=true;binding.acceptedFlux=accepted;binding.inputs=inputs.p;binding.errors=inputErrors.p;
        check(cudaDeviceSynchronize());cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        if(cells)ugkwp::gaswall::sumFaceAreaRows<<<1,128,0,stream>>>(cells,ro.p,ri.p,MaterialSurfaceGasArea{gm.p,gf.p},area.p);
        prepareResidentMaterialSurfaceInput<<<1,128,0,stream>>>(binding,timing.p,p.p,prot.p,operation.p);
        evaluateMaterialSurfaceTeamKernel<<<workers,ugkwp::gaswall::detail::wallWarpWidth,0,stream>>>(ugkp::gpu::DeviceValue<MaterialSurfaceView>{operation.p},p.p);
        check(cudaGetLastError());check(cudaStreamEndCapture(stream,&graph));std::size_t nodesCount=0;check(cudaGraphGetNodes(graph,nullptr,&nodesCount));
        std::vector<cudaGraphNode_t> nodes(nodesCount);check(cudaGraphGetNodes(graph,nodes.data(),&nodesCount));
        for(auto node:nodes){cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));require(type==cudaGraphNodeTypeKernel,"surface graph has a host/copy node");}
        check(cudaGraphInstantiateWithFlags(&exec,graph,0));check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));
        const auto actual=results.read(count);const auto actualWall=wallResults.read(count);const auto status=statuses.read(count);const auto inputStatus=inputErrors.read(count);const auto prepared=inputs.read(count);
        const auto areas=area.read(cells);
        for(int r=0;r<requests;++r)for(int f=0;f<nf;++f){const int i=r*nf+f,c=host.gasMesh.owner[f];require(inputStatus[i]==MaterialSurfaceInputError::None,"surface input rejected");require(status[i].success,"resident surface root failed");
            equal(actual[i],native[r].physics[f]);equal(actualWall[i].temperature,native[r].wall[f].temperature);
            for(int k=0;k<Ns;++k){equal(actualWall[i].speciesRate[k],native[r].wall[f].speciesRate[k]);equal(actualWall[i].poreRate[k],native[r].wall[f].poreRate[k]);}
            if(host.surface.gasFace[f]<0){++comparisons;continue;}
            // The native binary retains its production FMA policy, whereas
            // this isolated CUDA probe disables contraction. Cancellation in
            // the thermal residual is judged by the original physical energy
            // budget, not the relative tolerance for temperatures/fluxes.
            const auto budget=[&](const SurfacePhysicsResult& v){
                return physics.tolerances.absoluteEnergy/(host.surface.area[f]*step.dt)
                    +physics.tolerances.relativeEnergy*(std::abs(v.gasEnergy)+std::abs(v.poreEnergy)+std::abs(v.phaseEnergy)+std::abs(v.radiationFlux));};
            const double deviceBudget=budget(actual[i]),nativeBudget=budget(native[r].physics[f]);
            const double residualDifference=std::abs(actual[i].thermalResidual-native[r].physics[f].thermalResidual);
            require(residualDifference<=deviceBudget+nativeBudget,"thermal residual differs beyond native energy budgets");
            if(std::abs(native[r].physics[f].thermalResidual)<=nativeBudget)
                require(std::abs(actual[i].thermalResidual)<=deviceBudget,"native converged root lost convergence");
            maximumResidualDifference=std::max(maximumResidualDifference,residualDifference);
            maximumResidualBudgetFraction=std::max(maximumResidualBudgetFraction,residualDifference/(deviceBudget+nativeBudget));
            std::cout<<std::setprecision(17)<<"ROOT faces="<<nf<<" request="<<r<<" face="<<f<<" accepted="<<accepted
                <<" native="<<native[r].physics[f].thermalResidual<<" resident="<<actual[i].thermalResidual
                <<" nativeBudget="<<nativeBudget<<" residentBudget="<<deviceBudget<<'\n';
            const double available=host.gasMesh.volumes[c]*host.gasVoidFraction[c]*(mag(host.gasMesh.areaVectors[f])/areas[c]);
            GasPrimitive recovered;require(recoverGas(prepared[i].gasInventory,available,physics,recovered),"prepared gas inventory is invalid");
            equal(recovered.temperature,bulk[r][f].temperature);equal(prepared[i].bulk.pressure,contexts[r][f].matchingPressure);
            require(prepared[i].useAcceptedFlux==accepted,"accepted proposal distinction lost");++comparisons;
        }
        check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));
        std::cout<<"PASS native owner vs resident 8ODE input/closure faces="<<nf<<" requests="<<requests<<" accepted="<<accepted<<'\n';
    }
    std::cout<<"PASS "<<comparisons<<" paired native production surface records (including inactive slots); maximum scaled field difference "<<maximum<<"; maximum thermal residual difference "<<maximumResidualDifference<<"; maximum paired energy-budget fraction "<<maximumResidualBudgetFraction<<'\n';
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
