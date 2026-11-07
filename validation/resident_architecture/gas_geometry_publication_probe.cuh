#pragma once
#include "common/gasTransport/GasGeometryPolicy.H"
#include "common/operators/publishGasMeshGeometryKernel.cuh"
// Included inside stageProbe, after the standalone device allocation helpers.
#define PROBE_CELL_FIELDS(M) M(Cx) M(Cy) M(Cz) M(V) M(cellLength) M(sstWallDistance)
#define PROBE_FACE_FIELDS(M) M(Sfx) M(Sfy) M(Sfz) M(magSf) M(faceCx) M(faceCy) M(faceCz) M(facePeriodicDx) M(facePeriodicDy) M(facePeriodicDz) M(deltaCoeffs) M(faceWeight)
struct GasPublicationState {
    ugkp::diagnostics::DeviceContext* diagnostics=nullptr;
#define FIELD(name) double* name=nullptr;
    PROBE_CELL_FIELDS(FIELD) PROBE_FACE_FIELDS(FIELD)
#undef FIELD
    const int *faceOwner=nullptr,*faceNeighbour=nullptr,*facePeriodicPair=nullptr;
    ugkwp::GasGeometryState<double> gasGeometry;
    double physicalSentinel=1729.,clockSentinel=-13.;
};
struct GasPublicationProbe {
    int nc,nf;std::vector<int> owner,neighbour,partner;
    Device<int> dOwner,dNeighbour,dPartner;Device<double> data;
    GasPublicationState host{},previous{};Device<GasPublicationState> state;
    std::vector<double> prior;
    static std::vector<int> addressing(const Topology& t,int sign){
        std::vector<int> out(t.faceOffsets.size()-1,-1);
        for(int c=0;c+1<int(t.cellOffsets.size());++c)for(int k=t.cellOffsets[c];k<t.cellOffsets[c+1];++k)
            if(t.cellSigns[k]==sign)out[t.cellFaces[k]]=c;
        return out;
    }
    GasPublicationProbe(const Topology& t,const std::vector<int>& pair,ugkp::diagnostics::DeviceContext* diagnostics):nc(t.cellOffsets.size()-1),nf(t.faceOffsets.size()-1),
        owner(addressing(t,1)),neighbour(addressing(t,-1)),partner(pair),dOwner(owner),dNeighbour(neighbour),dPartner(partner),
        data(std::vector<double>(6*nc+12*nf,-1729.)),state(std::vector<GasPublicationState>(1)),prior(6*nc+12*nf,-1729.){
        std::size_t offset=0;
#define CELL(name) host.name=data.pointer?data.pointer+offset:nullptr;offset+=nc;
        PROBE_CELL_FIELDS(CELL)
#undef CELL
#define FACE(name) host.name=data.pointer?data.pointer+offset:nullptr;offset+=nf;
        PROBE_FACE_FIELDS(FACE)
#undef FACE
        host.diagnostics=diagnostics;host.faceOwner=dOwner.pointer;host.faceNeighbour=dNeighbour.pointer;host.facePeriodicPair=dPartner.pointer;
        check(cudaMemcpy(state.pointer,&host,sizeof(host),cudaMemcpyHostToDevice));previous=host;
    }
    void capture(StageGeometryDeviceView stage,const StageGeometryControls* controls,cudaStream_t stream){
        ugkwp::bindGasStageGeometryKernel<<<1,1,0,stream>>>(state.pointer,stage,controls);
        if(nc)ugkwp::publishGasCellGeometryKernel<<<(nc+127)/128,128,0,stream>>>(state.pointer,stage.endpointGeometry,stage.status);
        if(nf)ugkwp::publishGasFaceGeometryKernel<<<(nf+127)/128,128,0,stream>>>(state.pointer,stage.endpointGeometry,stage.status);
        check(cudaGetLastError());
    }
    static void sameBinding(const ugkwp::GasGeometryState<double>& a,const ugkwp::GasGeometryState<double>& b){
        if(a.enabled!=b.enabled||a.oldVolume!=b.oldVolume||a.newVolume!=b.newVolume||a.faceSweptVolume!=b.faceSweptVolume||a.interval!=b.interval
            ||a.absoluteGeometryTolerance!=b.absoluteGeometryTolerance||a.relativeGeometryTolerance!=b.relativeGeometryTolerance)
            throw std::runtime_error("failed geometry candidate changed binding");
    }
    void checkLatchedFailure(StageGeometryDeviceView stage,const StageGeometryControls* controls,cudaStream_t stream){
        ugkp::diagnostics::DeviceContext stopped{};stopped.fatalClaim=1;
        check(cudaMemcpyAsync(host.diagnostics,&stopped,sizeof(stopped),cudaMemcpyHostToDevice,stream));
        auto sentinel=host;sentinel.gasGeometry={};sentinel.gasGeometry.interval=-23.;
        std::vector<double> untouched(prior.size(),-431.);
        if(!untouched.empty())check(cudaMemcpyAsync(data.pointer,untouched.data(),untouched.size()*sizeof(double),cudaMemcpyHostToDevice,stream));
        check(cudaMemcpyAsync(state.pointer,&sentinel,sizeof(sentinel),cudaMemcpyHostToDevice,stream));
        capture(stage,controls,stream);check(cudaStreamSynchronize(stream));
        if(download(data.pointer,untouched.size())!=untouched)throw std::runtime_error("latched failure published gas metrics");
        sameBinding(download(state.pointer,1)[0].gasGeometry,sentinel.gasGeometry);
    }
    void validate(StageGeometryDeviceView stage,StageGeometryControls controls){
        const auto status=download(stage.status,1)[0];const auto actual=download(data.pointer,prior.size());
        const auto current=download(state.pointer,1)[0];
        if(current.physicalSentinel!=host.physicalSentinel||current.clockSentinel!=host.clockSentinel)throw std::runtime_error("geometry publication changed state/clock");
        if(status.failure!=StageGeometryFailure::None){
            if(actual!=prior)throw std::runtime_error("failed geometry candidate changed metric arrays");
            sameBinding(current.gasGeometry,previous.gasGeometry);return;
        }
        ugkwp::GasGeometryState<double> expectedBinding;expectedBinding.enabled=true;
        expectedBinding.oldVolume=stage.oldVolume;expectedBinding.newVolume=stage.newVolume;expectedBinding.faceSweptVolume=stage.sweptVolume;
        expectedBinding.interval=controls.interval;expectedBinding.absoluteGeometryTolerance=controls.absoluteTolerance;expectedBinding.relativeGeometryTolerance=controls.relativeTolerance;
        sameBinding(current.gasGeometry,expectedBinding);
        const auto cells=download(stage.endpointGeometry.cell,nc);const auto faces=download(stage.endpointGeometry.face,nf);
        const auto distance=download(stage.endpointGeometry.distance,nc);std::vector<double> expected(actual.size());
        auto writeCell=[&](int field,int c,double value){expected[field*nc+c]=value;};
        auto writeFace=[&](int field,int f,double value){expected[6*nc+field*nf+f]=value;};
        for(int c=0;c<nc;++c){auto x=cells[c].centre;writeCell(0,c,x.x);writeCell(1,c,x.y);writeCell(2,c,x.z);
            writeCell(3,c,cells[c].volume);writeCell(4,c,std::cbrt(cells[c].volume));writeCell(5,c,distance[c].distance);}
        // Frozen pre-migration uploadGeometry formula, independently of the
        // new shared metric helper and its publication kernels.
        for(int f=0;f<nf;++f){auto a=faces[f].area,x=faces[f].centre;const double area=magnitude(a);
            const int o=owner[f],n=neighbour[f];Point chord=x-cells[o].centre,shift{};
            if(n>=0)chord=cells[n].centre-cells[o].centre;
            if(partner[f]>=0){const int pair=partner[f];shift=x-faces[pair].centre;chord=cells[owner[pair]].centre+shift-cells[o].centre;}
            const double delta=1/std::max(std::abs(dot(chord,a/area)),.05*magnitude(chord));double weight=.5;
            if(n>=0||partner[f]>=0){const double own=std::abs(dot(x-cells[o].centre,a/area));const Point mapped=cells[o].centre+chord;
                const double other=std::abs(dot(mapped-x,a/area));weight=own+other>0?other/(own+other):.5;}
            const double values[]={a.x,a.y,a.z,area,x.x,x.y,x.z,shift.x,shift.y,shift.z,delta,weight};
            for(int j=0;j<12;++j)writeFace(j,f,values[j]);
        }
        for(std::size_t i=0;i<actual.size();++i){const double a=actual[i],b=expected[i];
            // CPU libm and CUDA cbrt need not round identically. All face/cell
            // fields other than the cube-root length must match exactly.
            if(i>=std::size_t(4*nc)&&i<std::size_t(5*nc)){
                if(!(std::abs(a-b)<=3e-15*std::abs(b)))throw std::runtime_error("resident cell length differs");
            }else if(a!=b)throw std::runtime_error("resident gas geometry metric differs");
        }
        prior=actual;previous=current;
    }
};
#undef PROBE_CELL_FIELDS
#undef PROBE_FACE_FIELDS
