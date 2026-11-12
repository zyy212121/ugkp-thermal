#pragma once
#include "../Diagnostics.H"
#include <type_traits>
#include <utility>
#include "../gasTransport/GasMeshGeometry.H"
#include "../gpu/ResidentStageGeometry.H"
namespace ugkwp {
template<class State,class=void> struct HasTrackingFaceMetrics : std::false_type {};
template<class State> struct HasTrackingFaceMetrics<State,
    std::void_t<decltype(std::declval<State>().trackingSfx)>> : std::true_type {};

// Addresses/topology are initialized by the gas owner. Only mutable metric
// arrays are written here. Failed candidate geometry is never published.
// These kernels do not touch physical fields, boundary data or resident clocks.
template<class State>
__global__ void publishGasCellGeometryKernel(State* state,
    ugkp::gpu::geometry::PolyhedralGeometryDeviceView geometry,
    const ugkp::gpu::geometry::StageGeometryStatus* status)
{
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    using ugkp::gpu::geometry::StageGeometryFailure;
    if(status&&status->failure!=StageGeometryFailure::None)return;
    auto& s=*state;const int c=blockIdx.x*blockDim.x+threadIdx.x;
    if(c>=geometry.cells)return;
    const auto cell=geometry.cell[c];
    s.Cx[c]=cell.centre.x;s.Cy[c]=cell.centre.y;s.Cz[c]=cell.centre.z;
    s.V[c]=cell.volume;s.cellLength[c]=::cbrt(cell.volume);
    s.sstWallDistance[c]=geometry.distance?geometry.distance[c].distance:1.;
}
template<class State>
__global__ void publishGasFaceGeometryKernel(State* state,
    ugkp::gpu::geometry::PolyhedralGeometryDeviceView geometry,
    const ugkp::gpu::geometry::StageGeometryStatus* status)
{
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    using namespace ugkp::gpu::geometry;
    if(status&&status->failure!=StageGeometryFailure::None)return;
    auto& s=*state;const int f=blockIdx.x*blockDim.x+threadIdx.x;
    if(f>=geometry.faces)return;
    const auto face=geometry.face[f];const int owner=s.faceOwner[f],pair=s.facePeriodicPair[f];
    const bool periodic=pair>=0;const int neighbour=periodic?s.faceOwner[pair]:s.faceNeighbour[f];
    const auto other=neighbour>=0?geometry.cell[neighbour].centre:Point{};
    const auto metric=gasFaceGeometry(face.area,face.centre,geometry.cell[owner].centre,
        neighbour>=0,other,periodic,periodic?geometry.face[pair].centre:Point{});
    s.Sfx[f]=face.area.x;s.Sfy[f]=face.area.y;s.Sfz[f]=face.area.z;
    s.magSf[f]=metric.area;
    if constexpr(HasTrackingFaceMetrics<State>::value){
        s.trackingSfx[f]=face.area.x;s.trackingSfy[f]=face.area.y;s.trackingSfz[f]=face.area.z;
        s.trackingMagSf[f]=magnitude(face.area);
    }
    s.faceCx[f]=face.centre.x;s.faceCy[f]=face.centre.y;s.faceCz[f]=face.centre.z;
    s.facePeriodicDx[f]=metric.periodicShift.x;s.facePeriodicDy[f]=metric.periodicShift.y;s.facePeriodicDz[f]=metric.periodicShift.z;
    s.deltaCoeffs[f]=metric.delta;s.faceWeight[f]=metric.weight;
}
template<class State>
__global__ void bindGasStageGeometryKernel(State* state,
    ugkp::gpu::geometry::StageGeometryDeviceView stage,
    const ugkp::gpu::geometry::StageGeometryControls* controls)
{
    if(!ugkp::diagnostics::blockMayProceed(state->diagnostics))return;
    if(threadIdx.x||blockIdx.x||stage.status->failure!=ugkp::gpu::geometry::StageGeometryFailure::None)return;
    auto& g=state->gasGeometry;g.enabled=true;
    g.oldVolume=stage.oldVolume;g.newVolume=stage.newVolume;g.faceSweptVolume=stage.sweptVolume;
    g.interval=controls->interval;g.absoluteGeometryTolerance=controls->absoluteTolerance;
    g.relativeGeometryTolerance=controls->relativeTolerance;
}
}
