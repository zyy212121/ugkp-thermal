// Included inside ugkp::gpu::geometry after resident geometry primitives.
namespace {
struct StageWarning {unsigned int count=0;double observed=0,reference=0;};
struct StageExecution {
    unsigned long long pending=~0ull;
    int active=0,endpointDifferent=0,sampleDifferent=0,firstWarning=0;
    StageGeometryPhase phase=StageGeometryPhase::Input;
};
__device__ bool sameCoordinateBits(Point a,Point b){
    return __double_as_longlong(a.x)==__double_as_longlong(b.x)
        &&__double_as_longlong(a.y)==__double_as_longlong(b.y)&&__double_as_longlong(a.z)==__double_as_longlong(b.z);
}
__device__ void stageFailure(StageExecution* state,unsigned int category,int index){
    atomicMin(&state->pending,(static_cast<unsigned long long>(category)<<32)|static_cast<unsigned int>(index));
}
__device__ void stageWarning(StageWarning& out,double observed,double reference){
    if(!out.count){out.observed=observed;out.reference=reference;}++out.count;
}
__global__ void initializeStageGeometry(const StageGeometryControls* controls,StageExecution* state,StageGeometryStatus* status){
    if(threadIdx.x||blockIdx.x)return;*state={};*status={};
    const auto c=*controls;
    if(!std::isfinite(c.absoluteTolerance)||c.absoluteTolerance<0||!std::isfinite(c.relativeTolerance)||c.relativeTolerance<0)
        status->failure=StageGeometryFailure::InvalidTolerances;
    else if(!std::isfinite(c.interval)||c.interval<=0)status->failure=StageGeometryFailure::InvalidInterval;
}
__global__ void compareStageEndpoints(int count,const Point* old,const Point* endpoint,StageExecution* state){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p<count&&!sameCoordinateBits(old[p],endpoint[p]))atomicExch(&state->endpointDifferent,1);
}
__global__ void beginStagePhase(StageGeometryPhase phase,int rows,int skip,const StageGeometryStatus* status,StageExecution* state){
    if(threadIdx.x||blockIdx.x)return;
    state->active=status->failure==StageGeometryFailure::None;
    if(skip==1&&!state->endpointDifferent)state->active=0;
    if(skip==2&&!state->endpointDifferent&&!state->sampleDifferent)state->active=0;
    state->pending=~0ull;state->firstWarning=rows;state->phase=phase;
}
__global__ void stageFinitePoints(int count,const Point* points,StageExecution* state){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(state->active&&p<count&&!finite(points[p]))stageFailure(state,0,p);
}
__global__ void stageValidFaces(int count,const FaceResult* faces,StageExecution* state){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(state->active&&f<count&&faces[f].invalid)stageFailure(state,1,f);
}
__global__ void stageValidCells(int count,const CellResult* cells,StageExecution* state,StageWarning* warnings){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=count||!state->active)return;
    const auto r=cells[c];auto& warning=warnings[c];warning={};
    if(!finite(r.closure)||!std::isfinite(r.areaSum)){stageFailure(state,2,c);return;}
    const double closure=magnitude(r.closure),limit=1e-10*r.areaSum;
    if(closure>limit){stageWarning(warning,closure,limit);atomicMin(&state->firstWarning,c);}
    if(r.invalid)stageFailure(state,2,c);
}
__global__ void stageValidDistances(int count,const WallDistanceResult* distances,StageExecution* state){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=count||!state->active)return;
    const auto r=distances[c];if(r.invalidFace>=0||!(r.distance>0)||!std::isfinite(r.distance))stageFailure(state,5,c);
}
__global__ void stageBoundaryResults(const GeometryBoundaryStatus* periodic,const GeometryBoundaryStatus* unique,StageExecution* state){
    if(threadIdx.x||blockIdx.x||!state->active)return;
    if(periodic->failure!=GeometryBoundaryFailure::None)stageFailure(state,3,periodic->index);
    if(unique->failure!=GeometryBoundaryFailure::None)stageFailure(state,4,0);
}
__device__ int stageWarningCutoff(const StageExecution& state,int rows){
    if(state.pending==~0ull)return rows-1;
    const auto category=state.pending>>32;
    if(category<2)return -1;
    return category==2?int(static_cast<unsigned int>(state.pending)):rows-1;
}
__device__ ugkp::diagnostics::Record stageWarningRecord(const StageWarning& warning,int row,bool conservation,bool face){
    using namespace ugkp::diagnostics;
    return makeRecord(conservation?Code::ConservationResidual:Code::AccuracyResidual,conservation?3401:3402,34,0,
        face?EntityKind::Face:EntityKind::Cell,row,-1,warning.observed,warning.reference,warning.observed,
        conservation?Quantity::Conservation:Quantity::Accuracy);
}
__global__ void publishStageFirstWarning(int rows,const StageExecution* state,const StageWarning* warnings,
    bool conservation,bool face,ugkp::diagnostics::DeviceContext* diagnostics){
    if(threadIdx.x||blockIdx.x||!diagnostics||!state->active)return;
    const int first=state->firstWarning;
    if(first<rows&&first<=stageWarningCutoff(*state,rows))
        ugkp::diagnostics::recordWarning(diagnostics,stageWarningRecord(warnings[first],first,conservation,face));
}
__global__ void publishStageWarningCounts(int rows,const StageExecution* state,const StageWarning* warnings,
    bool conservation,bool face,ugkp::diagnostics::DeviceContext* diagnostics){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(!diagnostics||!state->active||c>=rows||c>stageWarningCutoff(*state,rows))return;
    const auto warning=warnings[c];const unsigned int already=c==state->firstWarning?1:0;
    // The first sample has already been published in serial phase order. The
    // shared context retains counts plus that sample, not a per-warning list.
    for(unsigned int k=already;k<warning.count;++k)
        ugkp::diagnostics::recordWarning(diagnostics,stageWarningRecord(warning,c,conservation,face));
}
__global__ void finishStageGeometryPhase(PolyhedralGeometryDeviceView geometry,const GeometryBoundaryStatus* periodic,
    const StageExecution* state,StageGeometryStatus* status){
    if(threadIdx.x||blockIdx.x||!state->active||state->pending==~0ull)return;
    const unsigned int category=state->pending>>32;const int i=static_cast<unsigned int>(state->pending);
    status->index=i;status->phase=state->phase;
    if(category==0)status->failure=StageGeometryFailure::NonfiniteVertex;
    else if(category==1)status->failure=StageGeometryFailure::DegenerateFace;
    else if(category==2){const auto c=geometry.cell[i];
        status->failure=!finite(c.closure)||!std::isfinite(c.areaSum)?StageGeometryFailure::NonfiniteCellClosure:
            c.invalid==1?StageGeometryFailure::NonpositiveVolume:StageGeometryFailure::NonfiniteCentroid;
    }else if(category==3)status->failure=periodic->failure==GeometryBoundaryFailure::PeriodicArea?
        StageGeometryFailure::PeriodicArea:StageGeometryFailure::PeriodicVertices;
    else if(category==4){status->failure=StageGeometryFailure::EmptyDuplicate;status->index=-1;}
    else{const auto d=geometry.distance[i];status->failure=d.invalidFace>=0?StageGeometryFailure::NonfiniteWallTriangle:StageGeometryFailure::InvalidWallDistance;
        if(d.invalidFace>=0)status->index=d.invalidFace;}
}
__global__ void absorbEmptyStageTrajectory(const GeometryBoundaryStatus* input,StageGeometryStatus* out){
    if(threadIdx.x||blockIdx.x||out->failure!=StageGeometryFailure::None||input->failure==GeometryBoundaryFailure::None)return;
    out->phase=StageGeometryPhase::EmptyTrajectory;out->index=-1;
    out->failure=input->failure==GeometryBoundaryFailure::EmptyNormal?StageGeometryFailure::EmptyNormal:StageGeometryFailure::EmptyTangential;
}
__global__ void importStageTrajectory(TrajectoryDeviceView input,StageExecution* state,StageWarning* warnings){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=input.cells||!state->active)return;
    const auto d=input.diagnostics[c];warnings[c]={d.count,d.first.observed,d.first.limit};
    if(d.count)atomicMin(&state->firstWarning,c);
    if(input.failure[c]!=TrajectoryFailure::None)stageFailure(state,2,c);
}
__global__ void finishStageTrajectory(TrajectoryDeviceView input,const StageExecution* state,StageGeometryStatus* status){
    if(threadIdx.x||blockIdx.x||!state->active||state->pending==~0ull)return;
    const int c=static_cast<unsigned int>(state->pending);status->index=c;status->phase=StageGeometryPhase::LinearTrajectory;
    switch(input.failure[c]){
    case TrajectoryFailure::ZeroRadius:status->failure=StageGeometryFailure::ZeroRadius;break;
    case TrajectoryFailure::NonfiniteDeterminant:status->failure=StageGeometryFailure::NonfiniteDeterminant;break;
    case TrajectoryFailure::CollapsedFan:status->failure=StageGeometryFailure::CollapsedFan;break;
    case TrajectoryFailure::NonStarVolume:status->failure=StageGeometryFailure::NonStarVolume;break;
    case TrajectoryFailure::None:break;
    }
}
__global__ void resetStageSample(StageExecution* state){if(!threadIdx.x&&!blockIdx.x)state->sampleDifferent=0;}
__global__ void interpolateStageSample(int count,int sample,int axis,const Point* old,const Point* end,Point* points,StageExecution* state){
    const int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=count)return;
    points[p]=stageSample(old[p],end[p],sample,axis);
    if(!sameCoordinateBits(points[p],old[p]))atomicExch(&state->sampleDifferent,1);
}
__global__ void saveStageSampleVolumes(int count,const CellResult* cells,double* volumes){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<count)volumes[c]=cells[c].volume;
}
__global__ void checkStageInteriorVolumes(int count,const CellResult* old,const CellResult* end,const double* one,const double* two,StageExecution* state){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c<count&&state->active&&stageInteriorVolumeInverts(old[c].volume,one[c],two[c],end[c].volume))stageFailure(state,2,c);
}
__global__ void finishSimpleStagePhase(StageGeometryFailure failure,const StageExecution* state,StageGeometryStatus* status){
    if(threadIdx.x||blockIdx.x||!state->active||state->pending==~0ull)return;
    status->failure=failure;status->phase=state->phase;status->index=static_cast<unsigned int>(state->pending);
}
__global__ void checkStageSweeps(int count,const SweepResult* sweep,StageExecution* state){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f<count&&state->active&&(!finite(sweep[f].area)||!std::isfinite(sweep[f].volume)))stageFailure(state,0,f);
}
struct StageSweepArea {const SweepResult* p;__host__ __device__ Point operator()(int i)const{return p[i].area;}};
struct StageSweepVolume {const SweepResult* p;__host__ __device__ double operator()(int i)const{return p[i].volume;}};
__global__ void checkStageConservation(PolyhedralGeometryDeviceView old,PolyhedralGeometryDeviceView end,
    const StageGeometryControls* controls,StageExecution* state,StageWarning* warnings){
    const int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=old.cells||!state->active)return;
    auto& warning=warnings[c];warning={};
    const auto b=stageCellConservation(c,old.topology,old.cell[c].volume,end.cell[c].volume,
        StageSweepArea{old.sweptFaces},StageSweepVolume{old.sweptFaces},controls->absoluteTolerance,controls->relativeTolerance);
    if(b.invalid){stageFailure(state,2,c);return;}
    if(b.closure>b.closureLimit)stageWarning(warning,b.closure,b.closureLimit);
    if(::fabs(b.residual)>b.residualLimit)stageWarning(warning,b.residual,b.residualLimit);
    if(warning.count)atomicMin(&state->firstWarning,c);
}
__global__ void checkPeriodicStageConservation(int count,const int* partner,const SweepResult* sweep,
    const StageGeometryControls* controls,StageExecution* state,StageWarning* warnings){
    const int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=count||!state->active)return;
    auto& warning=warnings[f];warning={};const int p=partner[f];if(p<0)return;
    const double a=::fabs(sweep[f].volume),b=::fabs(sweep[p].volume);
    const double limit=controls->absoluteTolerance+controls->relativeTolerance*(a<b?b:a),residual=sweep[f].volume+sweep[p].volume;
    if(::fabs(residual)>limit){stageWarning(warning,residual,limit);atomicMin(&state->firstWarning,f);}
}
__global__ void publishStageGeometryFields(PolyhedralGeometryDeviceView old,PolyhedralGeometryDeviceView end,
    const StageGeometryControls* controls,Point* velocity,double* oldVolume,double* newVolume,double* sweptVolume){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<old.cells){oldVolume[i]=old.cell[i].volume;newVolume[i]=end.cell[i].volume;}
    if(i<old.faces){velocity[i]=(end.face[i].centre-old.face[i].centre)/controls->interval;sweptVolume[i]=old.sweptFaces[i].volume;}
}
}
struct ResidentStageGeometry::Implementation {
    int points=0,cells=0,faces=0,axis=-1;bool hasWalls=false;
    std::unique_ptr<ResidentPolyhedralGeometry> old,end,sample;
    std::unique_ptr<ResidentLinearTrajectory> trajectory;
    std::unique_ptr<ResidentGeometryBoundary> boundary;
    Array<Point> sampledPoints,meshVelocity;Array<double> oneThird,twoThird,oldVolume,newVolume,sweptVolume;Array<int> partner;
    Array<StageGeometryStatus> status;Array<StageExecution> execution;Array<StageWarning> warnings;
    void begin(StageGeometryPhase phase,int rows,int skip,cudaStream_t stream){
        beginStagePhase<<<1,1,0,stream>>>(phase,rows,skip,status.pointer,execution.pointer);checked(cudaGetLastError());
    }
    void publishWarnings(int rows,bool conservation,bool face,ugkp::diagnostics::DeviceContext* diagnostics,cudaStream_t stream){
        publishStageFirstWarning<<<1,1,0,stream>>>(rows,execution.pointer,warnings.pointer,conservation,face,diagnostics);checked(cudaGetLastError());
        if(rows){publishStageWarningCounts<<<(rows+127)/128,128,0,stream>>>(rows,execution.pointer,warnings.pointer,conservation,face,diagnostics);checked(cudaGetLastError());}
    }
    void validate(ResidentPolyhedralGeometry& geometry,const Point* coordinates,StageGeometryPhase phase,int skip,
        ugkp::diagnostics::DeviceContext* diagnostics,cudaStream_t stream){
        const auto v=geometry.view();begin(phase,cells,skip,stream);
        if(points){stageFinitePoints<<<(points+127)/128,128,0,stream>>>(points,coordinates,execution.pointer);checked(cudaGetLastError());}
        if(faces){stageValidFaces<<<(faces+127)/128,128,0,stream>>>(faces,v.face,execution.pointer);checked(cudaGetLastError());}
        if(cells){stageValidCells<<<(cells+127)/128,128,0,stream>>>(cells,v.cell,execution.pointer,warnings.pointer);checked(cudaGetLastError());}
        boundary->periodic(coordinates,v.face,stream);boundary->emptyUniqueness(coordinates,stream);
        stageBoundaryResults<<<1,1,0,stream>>>(boundary->periodicStatus(),boundary->emptyUniquenessStatus(),execution.pointer);checked(cudaGetLastError());
        if(hasWalls&&cells){stageValidDistances<<<(cells+127)/128,128,0,stream>>>(cells,v.distance,execution.pointer);checked(cudaGetLastError());}
        publishWarnings(cells,true,false,diagnostics,stream);
        finishStageGeometryPhase<<<1,1,0,stream>>>(v,boundary->periodicStatus(),execution.pointer,status.pointer);checked(cudaGetLastError());
    }
};
ResidentStageGeometry::ResidentStageGeometry(const Topology& topology,int pointCount,const std::vector<int>& walls,
    const std::vector<int>& partner,int axis,const std::vector<int>& mate):data_(new Implementation){
    auto& d=*data_;d.points=pointCount;d.axis=axis;d.hasWalls=!walls.empty();
    d.old.reset(new ResidentPolyhedralGeometry(topology,walls));d.end.reset(new ResidentPolyhedralGeometry(topology,walls));
    d.sample.reset(new ResidentPolyhedralGeometry(topology,walls));d.trajectory.reset(new ResidentLinearTrajectory(topology,pointCount));
    const auto v=d.old->view();d.faces=v.faces;d.cells=v.cells;
    d.boundary.reset(new ResidentGeometryBoundary(topology,v.topology,pointCount,partner,axis,mate));
    d.sampledPoints.reserve(pointCount);d.meshVelocity.reserve(d.faces);d.oneThird.reserve(d.cells);d.twoThird.reserve(d.cells);
    d.oldVolume.reserve(d.cells);d.newVolume.reserve(d.cells);d.sweptVolume.reserve(d.faces);
    d.partner.reserve(partner.size());if(!partner.empty())checked(cudaMemcpy(d.partner.pointer,partner.data(),partner.size()*sizeof(int),cudaMemcpyHostToDevice));
    d.status.reserve(1);d.execution.reserve(1);d.warnings.reserve(std::max(d.cells,d.faces));
}
ResidentStageGeometry::~ResidentStageGeometry()=default;
StageGeometryDeviceView ResidentStageGeometry::view()const{
    const auto& d=*data_;return {d.old->view(),d.end->view(),d.meshVelocity.pointer,d.status.pointer,d.oldVolume.pointer,d.newVolume.pointer,d.sweptVolume.pointer};
}
void ResidentStageGeometry::prepare(const Point* old,const Point* endpoint,const StageGeometryControls* controls,
    ugkp::diagnostics::DeviceContext* diagnostics,void* streamHandle){
    auto& d=*data_;const auto stream=static_cast<cudaStream_t>(streamHandle);
    initializeStageGeometry<<<1,1,0,stream>>>(controls,d.execution.pointer,d.status.pointer);checked(cudaGetLastError());
    if(d.points){compareStageEndpoints<<<(d.points+127)/128,128,0,stream>>>(d.points,old,endpoint,d.execution.pointer);checked(cudaGetLastError());}
    d.old->rebuild(old,stream);d.validate(*d.old,old,StageGeometryPhase::Old,0,diagnostics,stream);
    d.boundary->emptyTrajectory(old,endpoint,stream);
    absorbEmptyStageTrajectory<<<1,1,0,stream>>>(d.boundary->emptyTrajectoryStatus(),d.status.pointer);checked(cudaGetLastError());
    d.end->rebuild(endpoint,stream);d.validate(*d.end,endpoint,StageGeometryPhase::Endpoint,1,diagnostics,stream);
    d.begin(StageGeometryPhase::LinearTrajectory,d.cells,1,stream);d.trajectory->evaluate(old,endpoint,stream);const auto trajectory=d.trajectory->view();
    if(d.cells){importStageTrajectory<<<(d.cells+127)/128,128,0,stream>>>(trajectory,d.execution.pointer,d.warnings.pointer);checked(cudaGetLastError());}
    d.publishWarnings(d.cells,false,false,diagnostics,stream);
    finishStageTrajectory<<<1,1,0,stream>>>(trajectory,d.execution.pointer,d.status.pointer);checked(cudaGetLastError());
    for(int sample=0;sample<3;++sample){
        resetStageSample<<<1,1,0,stream>>>(d.execution.pointer);checked(cudaGetLastError());
        if(d.points){interpolateStageSample<<<(d.points+127)/128,128,0,stream>>>(d.points,sample,d.axis,old,endpoint,d.sampledPoints.pointer,d.execution.pointer);checked(cudaGetLastError());}
        d.sample->rebuild(d.sampledPoints.pointer,stream);
        const auto phase=sample==0?StageGeometryPhase::Midpoint:sample==1?StageGeometryPhase::OneThird:StageGeometryPhase::TwoThird;
        d.validate(*d.sample,d.sampledPoints.pointer,phase,2,diagnostics,stream);
        if(sample&&d.cells){saveStageSampleVolumes<<<(d.cells+127)/128,128,0,stream>>>(d.cells,d.sample->view().cell,sample==1?d.oneThird.pointer:d.twoThird.pointer);checked(cudaGetLastError());}
    }
    const auto initial=d.old->view(),final=d.end->view();d.begin(StageGeometryPhase::Volume,d.cells,0,stream);
    if(d.cells){checkStageInteriorVolumes<<<(d.cells+127)/128,128,0,stream>>>(d.cells,initial.cell,final.cell,d.oneThird.pointer,d.twoThird.pointer,d.execution.pointer);checked(cudaGetLastError());}
    finishSimpleStagePhase<<<1,1,0,stream>>>(StageGeometryFailure::InteriorInversion,d.execution.pointer,d.status.pointer);checked(cudaGetLastError());
    d.begin(StageGeometryPhase::Sweep,d.faces,0,stream);d.old->sweep(old,endpoint,stream);
    if(d.faces){checkStageSweeps<<<(d.faces+127)/128,128,0,stream>>>(d.faces,initial.sweptFaces,d.execution.pointer);checked(cudaGetLastError());}
    finishSimpleStagePhase<<<1,1,0,stream>>>(StageGeometryFailure::NonfiniteSweep,d.execution.pointer,d.status.pointer);checked(cudaGetLastError());
    d.begin(StageGeometryPhase::Conservation,d.cells,0,stream);
    if(d.cells){checkStageConservation<<<(d.cells+127)/128,128,0,stream>>>(initial,final,controls,d.execution.pointer,d.warnings.pointer);checked(cudaGetLastError());}
    d.publishWarnings(d.cells,true,false,diagnostics,stream);
    finishSimpleStagePhase<<<1,1,0,stream>>>(StageGeometryFailure::NonfiniteStageClosure,d.execution.pointer,d.status.pointer);checked(cudaGetLastError());
    d.begin(StageGeometryPhase::PeriodicSweep,d.faces,0,stream);
    if(d.faces){checkPeriodicStageConservation<<<(d.faces+127)/128,128,0,stream>>>(d.faces,d.partner.pointer,initial.sweptFaces,controls,d.execution.pointer,d.warnings.pointer);checked(cudaGetLastError());}
    d.publishWarnings(d.faces,true,true,diagnostics,stream);
    const int rows=std::max(d.cells,d.faces);
    if(rows){publishStageGeometryFields<<<(rows+127)/128,128,0,stream>>>(initial,final,controls,d.meshVelocity.pointer,d.oldVolume.pointer,d.newVolume.pointer,d.sweptVolume.pointer);checked(cudaGetLastError());}
}
