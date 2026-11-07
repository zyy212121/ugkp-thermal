#include "material_slice_probe.H"
#include "applications/CHMT/gpu/ResidentMaterialSlice.cuh"
#include <stdexcept>
#include "common/gpu/StableRowIndex.cuh"
#include "gpu/ResidentMaterialSource.cuh"
namespace materialSliceProbe {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T>struct Device {
    T* p=nullptr;explicit Device(const std::vector<T>& v){if(v.empty())return;
        check(cudaMalloc(&p,v.size()*sizeof(T)));check(cudaMemcpy(p,v.data(),v.size()*sizeof(T),cudaMemcpyHostToDevice));}
    ~Device(){if(p)cudaFree(p);}
};
__global__ void reset(std::size_t faces,std::size_t extra,chmt::MaterialSliceAccumulators used,ugkp::IntervalSliceCursor* cursor){
    if(!blockIdx.x&&!threadIdx.x)*cursor={};
    const auto n=faces>extra?faces:extra;
    for(std::size_t f=blockIdx.x*blockDim.x+threadIdx.x;f<n;f+=blockDim.x*gridDim.x){
        if(f<faces){for(unsigned k=0;k<3;++k)used.packets[3*f+k]={};used.radiation[f]=0;}
        if(f<extra)used.externalRadiation[f]=0;
    }
}
__global__ void observe(chmt::ResidentMaterialSlice slice,chmt::ExchangePacket* packets,int* active,std::size_t capacity,
    const double* radiation,std::size_t radFaces,double* radOutput,ugkp::IntervalSlicePlan* planOutput){
    if(!blockIdx.x&&!threadIdx.x)*planOutput=*slice.plan;
    const auto p=*slice.plan;const auto n=capacity>radFaces?capacity:radFaces;
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=std::size_t(blockDim.x)*gridDim.x){
        if(i<capacity){active[i]=p.error==ugkp::IntervalSliceError::None&&i<slice.slotCount()&&slice.active(i);
            if(active[i])packets[i]=slice.packet(i);}
        if(i<radFaces&&p.error==ugkp::IntervalSliceError::None)radOutput[i]=radiation[i];
    }
}
__global__ void observeCursor(const ugkp::IntervalSliceCursor* cursor,ugkp::IntervalSliceCursor* output){if(!blockIdx.x&&!threadIdx.x)*output=*cursor;}
__global__ void resetSourceBinding(chmt::ResidentMaterialState* state,chmt::ResidentMaterialState initial){if(!blockIdx.x&&!threadIdx.x)*state=initial;}
__global__ void prepareSourceFixture(const chmt::ResidentMaterialState* state){
    const int c=threadIdx.x;if(c>=3)return;auto binding=*state;
    binding.current[c]=binding.initial[c];binding.candidate[c]={};binding.candidate[c].energy=-999;
    binding.gasChange[c]={};binding.candidateGasChange[c]={};
}
__global__ void observeSource(chmt::ResidentMaterialSlice slice,const chmt::ResidentMaterialState* state,
    const chmt::materialSourceDetail::MaterialStepResult* status,chmt::SolidQ* output,int* codes,const int* requested,int* accepted){
    const int c=threadIdx.x;if(c>=3)return;
    const bool valid=slice.plan->error==ugkp::IntervalSliceError::None;
    if(valid){output[c]=state->candidate[c];codes[c]=status[c].phase==2?status[c].code:(status[c].phase?100+status[c].phase:0);}
    if(!c)*accepted=valid&&*requested&&!status[0].phase&&!status[1].phase&&!status[2].phase;
}
__global__ void observeBinding(const chmt::ResidentMaterialState* state,chmt::ResidentMaterialState original,int* parity){
    if(threadIdx.x)return;const auto s=*state;
    const bool even=s.current==original.current&&s.candidate==original.candidate&&s.gasChange==original.gasChange&&s.candidateGasChange==original.candidateGasChange&&s.mesh.volumes==original.mesh.volumes&&s.candidateMesh.volumes==original.candidateMesh.volumes;
    const bool odd=s.current==original.candidate&&s.candidate==original.current&&s.gasChange==original.candidateGasChange&&s.candidateGasChange==original.gasChange&&s.mesh.volumes==original.candidateMesh.volumes&&s.candidateMesh.volumes==original.mesh.volumes;
    *parity=even?0:odd?1:-1;
}
std::vector<Result> evaluate(const Test& t){
    using namespace chmt;const auto f=t.faces,e=t.external.size(),n=t.targets.size();
    const auto capacity=t.intervals.size()*f*3,radCapacity=f>e?f:e;
    Device<CoupledSurfaceDownload> surface(t.surface);Device<ResidentMaterialDriveInterval> intervals(t.intervals);
    Device<double> extra(t.external),usedRad(std::vector<double>(f,0)),nextRad(std::vector<double>(f,0)),
        usedExtra(std::vector<double>(e,0)),nextExtra(std::vector<double>(e,0)),rad(std::vector<double>(radCapacity,-71)),radOutput(std::vector<double>(n*radCapacity,-71));
    Device<ExchangePacket> usedPackets{std::vector<ExchangePacket>(3*f)},nextPackets{std::vector<ExchangePacket>(3*f)},packetOutput{std::vector<ExchangePacket>(n*capacity)};
    Device<ugkp::IntervalSliceCursor> cursor{std::vector<ugkp::IntervalSliceCursor>(1)},cursorOutput{std::vector<ugkp::IntervalSliceCursor>(n)};
    Device<ugkp::IntervalSlicePlan> plan{std::vector<ugkp::IntervalSlicePlan>(1)},planOutput{std::vector<ugkp::IntervalSlicePlan>(n)};
    Device<ugkp::CompensatedTime> targets(t.targets);Device<int> accepted(t.accepted),active{std::vector<int>(n*capacity)};
    std::vector<SolidQ> sourceInitial(3);for(int c=0;c<3;++c){for(int j=0;j<Nc;++j)sourceInitial[c].condensed[j]=1e25*(c+1);
        for(int j=0;j<Ns;++j)sourceInitial[c].pore[j]=1e25*(j+1);sourceInitial[c].energy=1e30;}
    Device<SolidQ> sourceBase(sourceInitial),sourceA(sourceInitial),sourceB{std::vector<SolidQ>(3)},
        sourceChangeA{std::vector<SolidQ>(3)},sourceChangeB{std::vector<SolidQ>(3)},rates{std::vector<SolidQ>(3)};
    Device<Real> volumeA(std::vector<Real>(3,1)),volumeB(std::vector<Real>(3,1));
    ResidentMaterialState original;original.current=sourceA.p;original.candidate=sourceB.p;original.initial=sourceBase.p;
    original.gasChange=sourceChangeA.p;original.candidateGasChange=sourceChangeB.p;
    original.mesh.cells=original.candidateMesh.cells=3;original.mesh.volumes=volumeA.p;original.candidateMesh.volumes=volumeB.p;
    Device<ResidentMaterialState> state(std::vector<ResidentMaterialState>{original});
    ResidentMaterialStep step;step.dt=.125;step.active=step.finalStep=true;
    Device<ResidentMaterialStep> timing(std::vector<ResidentMaterialStep>{step});
    Device<materialSourceDetail::MaterialStepResult> sourceStatus{std::vector<materialSourceDetail::MaterialStepResult>(3)};
    Device<int> sourceAccepted(std::vector<int>(1,0)),parity{std::vector<int>(n)};
    PhysicsConfig config;config.enableReactions=false;Device<PhysicsConfig> physics(std::vector<PhysicsConfig>{config});
    ResidentMaterialDriveHistory h{surface.p,nullptr,intervals.p,t.intervals.size(),f,nullptr};
    MaterialSliceAccumulators used{usedPackets.p,usedRad.p,usedExtra.p},candidate{nextPackets.p,nextRad.p,nextExtra.p};
    ResidentMaterialSlice slice{h,plan.p,used,t.particles};
    Device<int> keys{std::vector<int>(capacity)},sortedKeys{std::vector<int>(capacity)},inputIndices{std::vector<int>(capacity)},
        indices{std::vector<int>(capacity)},counts{std::vector<int>(4)},offsets{std::vector<int>(4)},codes{std::vector<int>(3*n)};
    Device<SolidQ> inventory{std::vector<SolidQ>(3*n)};
    ugkp::gpu::StableRowIndexView rows{keys.p,sortedKeys.p,inputIndices.p,indices.p,counts.p,offsets.p};
    std::size_t scratchBytes=0;check(ugkp::gpu::stableRowIndexWorkspace(int(capacity),3,rows,scratchBytes,nullptr));
    Device<unsigned char> scratch{std::vector<unsigned char>(scratchBytes)};check(cudaDeviceSynchronize());
    cudaStream_t stream;cudaGraph_t graph;cudaGraphExec_t exec;check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
    resetSourceBinding<<<1,1,0,stream>>>(state.p,original);
    reset<<<chmt::material_slice_detail::blocks(radCapacity),128,0,stream>>>(f,e,used,cursor.p);
    for(std::size_t i=0;i<n;++i){
        check(prepareResidentMaterialSlice(slice,cursor.p,targets.p+i,candidate,extra.p,e,t.total,rad.p,plan.p,stream));
        check(ugkp::gpu::buildStableRowIndex(int(capacity),3,ResidentMaterialSliceRow{slice},rows,scratch.p,scratchBytes,stream));
        prepareSourceFixture<<<1,32,0,stream>>>(state.p);
        applyResidentMaterialSources<<<1,32,0,stream>>>(state.p,timing.p,rates.p,slice,offsets.p,indices.p,nullptr,nullptr,nullptr,nullptr,
            physics.p,nullptr,sourceStatus.p,0,.5,100);
        observeSource<<<1,32,0,stream>>>(slice,state.p,sourceStatus.p,inventory.p+3*i,codes.p+3*i,accepted.p+i,sourceAccepted.p);
        observe<<<chmt::material_slice_detail::blocks(capacity>radCapacity?capacity:radCapacity),128,0,stream>>>(slice,
            packetOutput.p?packetOutput.p+i*capacity:nullptr,active.p?active.p+i*capacity:nullptr,capacity,
            rad.p,radCapacity,radOutput.p?radOutput.p+i*radCapacity:nullptr,planOutput.p+i);
        commitResidentMaterialState<<<1,1,0,stream>>>(state.p,sourceAccepted.p);
        observeBinding<<<1,1,0,stream>>>(state.p,original,parity.p+i);
        check(commitResidentMaterialSlice(plan.p,candidate,used,f,e,cursor.p,accepted.p+i,stream));
        observeCursor<<<1,1,0,stream>>>(cursor.p,cursorOutput.p+i);
    }
    check(cudaGetLastError());check(cudaStreamEndCapture(stream,&graph));std::size_t nodesCount=0;check(cudaGraphGetNodes(graph,nullptr,&nodesCount));
    std::vector<cudaGraphNode_t> nodes(nodesCount);check(cudaGraphGetNodes(graph,nodes.data(),&nodesCount));
    for(auto node:nodes){cudaGraphNodeType kind;check(cudaGraphNodeGetType(node,&kind));if(kind!=cudaGraphNodeTypeKernel&&kind!=cudaGraphNodeTypeMemset)throw std::runtime_error("slice graph includes transfer or host node");}
    check(cudaGraphInstantiateWithFlags(&exec,graph,0));for(int i=0;i<3;++i)check(cudaGraphLaunch(exec,stream));check(cudaStreamSynchronize(stream));
    std::vector<ugkp::IntervalSlicePlan> plans(n);std::vector<ugkp::IntervalSliceCursor> cursors(n);
    std::vector<int> enabled(n*capacity);std::vector<ExchangePacket> packets(n*capacity);std::vector<double> radiation(n*radCapacity);
    if(n){check(cudaMemcpy(plans.data(),planOutput.p,n*sizeof(*planOutput.p),cudaMemcpyDeviceToHost));check(cudaMemcpy(cursors.data(),cursorOutput.p,n*sizeof(*cursorOutput.p),cudaMemcpyDeviceToHost));}
    if(!packets.empty()){check(cudaMemcpy(packets.data(),packetOutput.p,packets.size()*sizeof(*packetOutput.p),cudaMemcpyDeviceToHost));check(cudaMemcpy(enabled.data(),active.p,enabled.size()*sizeof(int),cudaMemcpyDeviceToHost));}
    if(!radiation.empty())check(cudaMemcpy(radiation.data(),radOutput.p,radiation.size()*sizeof(double),cudaMemcpyDeviceToHost));
    std::vector<SolidQ> inventories(3*n);std::vector<int> statuses(3*n),parities(n);
    if(n){check(cudaMemcpy(parities.data(),parity.p,n*sizeof(int),cudaMemcpyDeviceToHost));check(cudaMemcpy(inventories.data(),inventory.p,inventories.size()*sizeof(SolidQ),cudaMemcpyDeviceToHost));check(cudaMemcpy(statuses.data(),codes.p,statuses.size()*sizeof(int),cudaMemcpyDeviceToHost));}
    std::vector<Result> result(n);
    for(std::size_t i=0;i<n;++i){auto& out=result[i];out.plan=plans[i];out.cursor=cursors[i];out.bindingParity=parities[i];
        out.inventory.assign(inventories.begin()+3*i,inventories.begin()+3*i+3);out.inventoryStatus.assign(statuses.begin()+3*i,statuses.begin()+3*i+3);
        for(std::size_t j=0;j<capacity;++j)if(enabled[i*capacity+j])out.packets.push_back(packets[i*capacity+j]);
        if(out.plan.error==ugkp::IntervalSliceError::None){const auto count=out.plan.hasRecords()&&f?f:e;
            out.radiation.assign(radiation.begin()+i*radCapacity,radiation.begin()+i*radCapacity+count);}}
    check(cudaGraphExecDestroy(exec));check(cudaGraphDestroy(graph));check(cudaStreamDestroy(stream));return result;
}
}
