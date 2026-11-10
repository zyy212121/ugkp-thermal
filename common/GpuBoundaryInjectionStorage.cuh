// Included after each backend's DeviceState, memory helpers and syncDeviceState.
void releaseBoundaryInjection(ugkwp::BoundaryInjectionStorage& g) {
    release(g.offsets); release(g.vertices); release(g.scratch);
    release(g.counts); release(g.points); release(g.triangles);
}

// Geometry is deliberately captured from the initial mesh for every application.
__global__ void buildBoundaryInjectionKernel(DeviceState* sp) {
    DeviceState& s=*sp;
    const int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=s.nBoundarySources)return;
    auto& g=s.boundaryInjection;
    const int f=s.sourceFace[j];
    const ugkp::gpu::geometry::Point shift{
        s.sourcePx[j]-s.faceCx[f],s.sourcePy[j]-s.faceCy[f],s.sourcePz[j]-s.faceCz[f]};
    const int begin=g.offsets[j],count=g.offsets[j+1]-begin;
    g.counts[j]=ugkwp::buildBoundaryInjectionTriangles(g.points,
        g.vertices+begin,count,g.scratch+begin,g.triangles+begin,
        {s.faceCx[f],s.faceCy[f],s.faceCz[f]},
        {s.Sfx[f],s.Sfy[f],s.Sfz[f]},shift);
}

extern "C" int ugkwpGpuResidentStrictUploadBoundarySourceGeometry
(void* handle,int nVertices,const int* offsets,const int* vertices,
 int nPoints,const ugkp::gpu::geometry::Point* points) {
    DeviceState* s=asState(handle);
    ugkwp::BoundaryInjectionStorage g;
    int rc=0;
    if(nVertices>0&&s->nBoundarySources>0) {
        rc|=allocate(g.offsets,size_t(s->nBoundarySources+1),"allocate injection polygon offsets");
        rc|=allocate(g.vertices,size_t(nVertices),"allocate injection polygon vertices");
        rc|=allocate(g.scratch,size_t(nVertices),"allocate injection polygon scratch");
        rc|=allocate(g.counts,size_t(s->nBoundarySources),"allocate injection triangle counts");
        rc|=allocate(g.points,size_t(nPoints),"allocate injection polygon points");
        rc|=allocate(g.triangles,size_t(nVertices),"allocate injection triangles");
        if(rc){releaseBoundaryInjection(g);return rc;}
        rc|=copyToDevice(g.offsets,offsets,size_t(s->nBoundarySources+1),"copy injection polygon offsets");
        rc|=copyToDevice(g.vertices,vertices,size_t(nVertices),"copy injection polygon vertices");
        rc|=copyToDevice(g.points,points,size_t(nPoints),"copy injection polygon points");
    }
    if(rc){releaseBoundaryInjection(g);return rc;}
    auto previous=s->boundaryInjection;
    s->boundaryInjection=g;
    rc=syncDeviceState(s,"publish boundary injection storage");
    if(rc){s->boundaryInjection=previous;releaseBoundaryInjection(g);return rc;}
    releaseBoundaryInjection(previous);
    if(!g.offsets)return 0;
    buildBoundaryInjectionKernel<<<(s->nBoundarySources+127)/128,128>>>(s->deviceState);
    auto err=cudaGetLastError();
    if(err==cudaSuccess)err=cudaStreamSynchronize(nullptr);
    if(err!=cudaSuccess){setLastError("build initial injection polygons",err);return 1;}
    return 0;
}
