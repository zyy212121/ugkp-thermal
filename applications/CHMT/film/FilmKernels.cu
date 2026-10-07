#include "film/FilmKernels.H"
#include "film/FilmContracts.H"
#include "ablation/InterfaceKernels.H"
#include "gas/KernelSupport.cuh"
#include "gas/AleFlux.H"
#include <sstream>
namespace chmt {
namespace {
__device__ Real surfaceValue(int face, int component, FilmView film, SurfaceView surface) {
    if (component<0) return film.aux[face].pressure;
    if (component==0) return film.q[face].mass/surface.area[face];
    if (component==1) return film.q[face].enthalpy/surface.area[face];
    return film.q[face].species[component-2]/surface.area[face];
}
__device__ Vec3 surfaceGradient(int face, int component, FilmView film, SurfaceView surface,
    const PhysicsConfig& p) {
    Symmetric3 matrix;
    Vec3 rhs{};
    const Real value=surfaceValue(face,component,film,surface);
    Real lo=value, hi=value;
    for (int edge=0; edge<surface.nEdges; ++edge) {
        const int owner=surface.edgeOwner[edge], neighbour=surface.edgeNeighbour[edge];
        if (owner<0 || owner>=film.nFaces || neighbour>=film.nFaces || neighbour<-1) {
            return {undefinedValue(),undefinedValue(),undefinedValue()};
        }
        if (neighbour<0 || (owner!=face && neighbour!=face)) continue;
        const int other=owner==face ? neighbour : owner;
        Vec3 dx=surface.edgeOwnerOffset[edge]-surface.edgeNeighbourOffset[edge];
        if (owner!=face) dx=-dx;
        const Real otherValue=surfaceValue(other,component,film,surface);
        addLeastSquares(matrix,rhs,dx,otherValue-value);
        lo=minValue(lo,otherValue); hi=maxValue(hi,otherValue);
    }
    (void)p;
    (void)lo;
    (void)hi;
    return solveLeastSquares(matrix,rhs);
}
__device__ FilmProfile faceProfile(int face, FilmView film, SurfaceView surface,
    const PhysicsConfig& p) {
    const FilmAux& a=film.aux[face];
    const Vec3 normal=surface.normal[face];
    const Vec3 gravity=p.gravity-normal*dot(p.gravity,normal);
    FilmProfile profile=filmProfile(a.thickness,p.liquidViscosity,a.baseVelocity,
        film.gasShear[face],film.pressureGradient[face]-gravity*p.liquid.rho);
    return withNormalProfile(profile,normal,
        2*dot(a.meanVelocity,normal)-dot(a.topVelocity,normal),dot(a.topVelocity,normal));
}
__global__ void filmRecoverKernel(FilmView film, SurfaceView surface, PhysicsConfig p,
    DeviceStatus* status) {
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if (face>=film.nFaces || gasStopped(status)) return;
    FilmAux a=film.aux[face];
    if (!finite(surface.normal[face]) || !closeEnough(mag(surface.normal[face]),1,1e-12,1e-12)) {
        gasDeviceError(status,ErrorCode::InvalidInput,face,mag(surface.normal[face]),ErrorLocation::Face);
        return;
    }
    a.normal=surface.normal[face];
    a.baseVelocity=surface.baseVelocity ? surface.baseVelocity[face] : Vec3{};
    DeviceStatus local;
    const bool coupled=surface.gasFace && surface.gasFace[face]>=0;
    if (!recoverFilmThermalState(film.q[face],surface.area[face],a.pressure,p,
        surface.normal[face],coupled,a,&local,face)) {
        gasDeviceError(status,static_cast<ErrorCode>(local.code),face,local.value,ErrorLocation::Face);
        return;
    }
    film.aux[face]=a;
}
__global__ void filmProfileKernel(FilmView film, SurfaceView surface, PacketView packets,
    PhysicsConfig p, Real dt, DeviceStatus* status) {
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if (face>=film.nFaces || gasStopped(status)) return;
    const Vec3 normal=surface.normal[face];
    const bool coupled=surface.gasFace && surface.gasFace[face]>=0;
    Vec3 gradient=surfaceGradient(face,-1,film,surface,p);
    if (surface.prescribedPressureGradient) {
        if (coupled && mag(surface.prescribedPressureGradient[face])>0) {
            gasDeviceError(status,ErrorCode::InvalidInput,face,1,ErrorLocation::Face); return;
        }
        gradient+=surface.prescribedPressureGradient[face];
    }
    film.pressureGradient[face]=gradient-normal*dot(gradient,normal);
    if (!coupled) {
        const Vec3 traction=surface.prescribedTopTraction ? surface.prescribedTopTraction[face] : Vec3{};
        film.gasShear[face]=traction-normal*dot(traction,normal);
    }
    FilmAux a=film.aux[face];
    const Vec3 G=film.pressureGradient[face]
        -(p.gravity-normal*dot(p.gravity,normal))*p.liquid.rho;
    DeviceStatus local;
    if (!updateFilmProfileState(a,p,normal,film.gasShear[face],G,coupled,&local,face)) {
        gasDeviceError(status,static_cast<ErrorCode>(local.code),face,local.value,ErrorLocation::Face);
        return;
    }
    (void)packets;
    (void)dt;
    film.aux[face]=a;
}
__device__ bool reconstructFilm(int face, Vec3 offset, FilmView film, SurfaceView surface,
    const PhysicsConfig& p, FilmQ& out) {
    FilmQ base;
    base.mass=surfaceValue(face,0,film,surface);
    base.enthalpy=surfaceValue(face,1,film,surface);
    for (int s=0; s<Ns; ++s) base.species[s]=surfaceValue(face,s+2,film,surface);
    if (p.spatialOrder==1 || base.mass==0) { out=base; return true; }
    FilmQ increment;
    increment.mass=dot(surfaceGradient(face,0,film,surface,p),offset);
    increment.enthalpy=dot(surfaceGradient(face,1,film,surface,p),offset);
    for (int s=0; s<Ns; ++s) {
        increment.species[s]=dot(surfaceGradient(face,s+2,film,surface,p),offset);
    }
    // One shared admissibility scale preserves all reconstructed composition.
    Real scale=1;
    if (p.reconstruction==ReconstructionMode::LimitedLinear) {
        for (int component=0; component<Ns+2; ++component) {
            const Real value=surfaceValue(face,component,film,surface);
            Real lo=value, hi=value;
            for (int edge=0; edge<surface.nEdges; ++edge) {
                const int owner=surface.edgeOwner[edge], neighbour=surface.edgeNeighbour[edge];
                if (neighbour<0 || (owner!=face && neighbour!=face)) continue;
                const int other=owner==face ? neighbour : owner;
                const Real neighbourValue=surfaceValue(other,component,film,surface);
                lo=minValue(lo,neighbourValue); hi=maxValue(hi,neighbourValue);
            }
            const Real change=component==0 ? increment.mass
                : (component==1 ? increment.enthalpy : increment.species[component-2]);
            scale=minValue(scale,barthJespersen(value,lo,hi,change));
        }
    }
    for (int trial=0; trial<48; ++trial) {
        FilmQ candidate;
        candidate.mass=base.mass+scale*increment.mass;
        candidate.enthalpy=base.enthalpy+scale*increment.enthalpy;
        for (int s=0; s<Ns; ++s) candidate.species[s]=base.species[s]+scale*increment.species[s];
        FilmAux aux;
        if (recoverFilm(candidate,1,film.aux[face].pressure,p,aux)) { out=candidate; return true; }
        scale*=0.5;
    }
    return false;
}
__global__ void filmEdgeKernel(FilmView film, SurfaceView surface, PhysicsConfig p,
    Real dt, DeviceStatus* status) {
    const int edge=blockIdx.x*blockDim.x+threadIdx.x;
    if (edge>=surface.nEdges || gasStopped(status)) return;
    const int owner=surface.edgeOwner[edge], neighbour=surface.edgeNeighbour[edge];
    if (owner<0 || owner>=film.nFaces || neighbour>=film.nFaces || neighbour<-1) {
        gasDeviceError(status,ErrorCode::InvalidInput,edge,owner,ErrorLocation::Face); return;
    }
    // An open edge is zero-gradient advective outflow; inflow requires a declared
    // reservoir policy and is rejected, never fabricated from an absent neighbor.
    const Vec3 conormal=surface.edgeConormal[edge];
    const Real length=surface.edgeLength[edge];
    const Real meshRate=surface.sweptEdgeArea ? surface.sweptEdgeArea[edge]/dt : 0;
    const Vec3 velocity=neighbour>=0
        ? (film.aux[owner].meanVelocity+film.aux[neighbour].meanVelocity)*0.5
        : film.aux[owner].meanVelocity;
    const Real lengthRate=length*dot(velocity,conormal)-meshRate;
    if (!(length>0) || (neighbour<0 && lengthRate<0)) {
        gasDeviceError(status,ErrorCode::Unsupported,edge,lengthRate,ErrorLocation::Face); return;
    }
    const int donor=lengthRate>=0 || neighbour<0 ? owner : neighbour;
    const Vec3 offset=donor==owner ? surface.edgeOwnerOffset[edge] : surface.edgeNeighbourOffset[edge];
    FilmQ reconstructed;
    if (!reconstructFilm(donor,offset,film,surface,p,reconstructed)) {
        gasDeviceError(status,ErrorCode::PropertyRange,edge,0,ErrorLocation::Face); return;
    }
    FilmQ flux=filmAdvectiveFlux(reconstructed,lengthRate);
    const Real thickness=reconstructed.mass/p.liquid.rho;
    const Real pressure=film.aux[donor].pressure;
    flux.enthalpy+=pressure*thickness*meshRate;
    if (neighbour>=0 && film.q[owner].mass>0 && film.q[neighbour].mass>0) {
        const Real distance=dot(surface.edgeOwnerOffset[edge]-surface.edgeNeighbourOffset[edge],conormal);
        if (!(distance>0)) {
            gasDeviceError(status,ErrorCode::InvalidInput,edge,distance,ErrorLocation::Face); return;
        }
        const Real meanThickness=0.5*(film.aux[owner].thickness+film.aux[neighbour].thickness);
        flux.enthalpy+=p.liquid.conductivity*length*meanThickness
            *(film.aux[owner].temperature-film.aux[neighbour].temperature)/distance;
    }
    film.edgeFlux[edge]=flux;
}
__global__ void filmRhsKernel(FilmView film, SurfaceView surface, PacketView packets,
    PhysicsConfig p, Real dt, DeviceStatus* status) {
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if (face>=film.nFaces || gasStopped(status)) return;
    FilmQ rhs;
    FilmRateBudget budget;
    Real sideVolumeRate=0;
    for (int edge=0; edge<surface.nEdges; ++edge) {
        int sign=0;
        if (surface.edgeOwner[edge]==face) sign=1;
        else if (surface.edgeNeighbour[edge]==face) sign=-1;
        if (!sign) continue;
        const FilmQ& flux=film.edgeFlux[edge];
        rhs.mass-=sign*flux.mass;
        rhs.enthalpy-=sign*flux.enthalpy;
        for (int s=0; s<Ns; ++s) rhs.species[s]-=sign*flux.species[s];
        budget.edgeEnergyOutflow+=sign*flux.enthalpy;
        const int owner=surface.edgeOwner[edge], neighbour=surface.edgeNeighbour[edge];
        const int donor=flux.mass>=0 || neighbour<0 ? owner : neighbour;
        const FilmProfile profile=faceProfile(donor,film,surface,p);
        const Real meshRate=surface.sweptEdgeArea ? surface.sweptEdgeArea[edge]/dt : 0;
        const Vec3 grid=surface.edgeConormal[edge]*(meshRate/surface.edgeLength[edge]);
        budget.edgeKineticOutflow+=sign*p.liquid.rho*surface.edgeLength[edge]
            *profileKineticFlux(profile,grid,surface.edgeConormal[edge]);
        sideVolumeRate+=sign*profile.delta*meshRate;
    }
    FilmAux a=film.aux[face];
    Real packetMassRate=0;
    bool topOwned=false, bottomOwned=false;
    if (packets.count>=InterfacePacketsPerFace*surface.nFaces) {
        const ExchangePacket& primary=packets.packets[3*face];
        const ExchangePacket& phase=packets.packets[3*face+2];
        topOwned=primary.kind==ExchangeKind::GasFilm && primary.filmFace==face;
        bottomOwned=phase.kind==ExchangeKind::SolidFilm && phase.filmFace==face;
        if (topOwned) {
            packetMassRate-=primary.mass/dt;
            budget.interfaceKineticOutflow+=primary.liquidKineticAdvection/dt;
        }
        if (bottomOwned) {
            packetMassRate+=phase.mass/dt;
            budget.interfaceKineticOutflow+=phase.liquidKineticAdvection/dt;
        }
    }
    const FilmProfile profile=faceProfile(face,film,surface,p);
    if (!surface.gasFace || surface.gasFace[face]<0) {
        a.solidNormalVelocity=dot(a.baseVelocity,surface.normal[face]);
    }
    const bool coupled=surface.gasFace && surface.gasFace[face]>=0;
    if (!coupled) {
        a.normalVelocity=a.solidNormalVelocity
            +((rhs.mass+packetMassRate)/p.liquid.rho-sideVolumeRate)/a.area;
    }
    budget.bodyPower=film.q[face].mass*dot(p.gravity,a.meanVelocity);
    if (surface.prescribedPressureGradient) {
        budget.bodyPower-=film.q[face].mass/p.liquid.rho
            *dot(surface.prescribedPressureGradient[face],a.meanVelocity);
    }
    if (!topOwned) budget.prescribedTopPower=a.area
        *(dot(film.gasShear[face],a.topVelocity)-a.pressure*a.normalVelocity);
    if (!bottomOwned) budget.supportPower=a.area
        *(-dot(profile.bottomShear,a.baseVelocity)+a.pressure*a.solidNormalVelocity);
    if (coupled) {
        budget.radiationPower=surface.radiationPower ? surface.radiationPower[face] : 0;
    } else if (film.q[face].mass>0) {
        budget.radiationPower=a.area*grayRadiation(a.temperature,p);
        rhs.enthalpy+=budget.radiationPower;
    }
    budget.dissipationPower=a.area*profile.dissipation;
    rhs.enthalpy+=budget.bodyPower+budget.supportPower+budget.prescribedTopPower;
    if (!finite(rhs.enthalpy) || !finite(rhs.mass)) {
        gasDeviceError(status,ErrorCode::Nonfinite,face,rhs.enthalpy,ErrorLocation::Face); return;
    }
    if (film.reducedResidual) {
        film.reducedResidual[face]=rhs.enthalpy+budget.edgeEnergyOutflow-budget.bodyPower
            -budget.supportPower-budget.prescribedTopPower-(coupled ? 0 : budget.radiationPower);
    }
    film.aux[face]=a;
    film.rhs[face]=rhs;
    film.rateBudget[face]=budget;
}
bool filmPointers(FilmView film, SurfaceView surface) {
    return film.nFaces==surface.nFaces && film.nEdges==surface.nEdges
        && film.q && film.aux && film.gasShear && film.pressureGradient && surface.area && surface.normal
        && (surface.nEdges==0 || (surface.edgeOwner && surface.edgeNeighbour && surface.edgeLength
            && surface.edgeConormal && surface.edgeOwnerOffset && surface.edgeNeighbourOffset));
}
}
int launchFilmValidate(FilmView film, SurfaceView surface, const PhysicsConfig& p,
    DeviceStatus* status, void* stream) {
    if (film.nFaces==0) return 0;
    if (p.filmThermalMode==FilmThermalMode::ResolvedNormal) return static_cast<int>(cudaErrorNotSupported);
    if (!p.enableFilm || !status || !filmPointers(film,surface) || film.kineticDefect) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    cudaStream_t cudaStream=static_cast<cudaStream_t>(stream);
    filmRecoverKernel<<<(film.nFaces+127)/128,128,0,cudaStream>>>(film,surface,p,status);
    int code=launchError(); if (code) return code;
    filmProfileKernel<<<(film.nFaces+127)/128,128,0,cudaStream>>>
        (film,surface,PacketView{},p,0,status);
    return launchError();
}
int launchFilmRhs(FilmView film, SurfaceView surface, PacketView packets, const PhysicsConfig& p,
    Real dt, DeviceStatus* status, void* stream) {
    if (film.nFaces==0) return 0;
    if (p.filmThermalMode==FilmThermalMode::ResolvedNormal) return static_cast<int>(cudaErrorNotSupported);
    if (!p.enableFilm || !status || !filmPointers(film,surface) || !film.stageBase || !film.rhs || !film.rateBudget
        || (film.nEdges>0 && !film.edgeFlux) || film.kineticDefect || !finite(dt) || dt<=0
        || (packets.count>0 && !packets.packets)) return static_cast<int>(cudaErrorInvalidValue);
    cudaStream_t cudaStream=static_cast<cudaStream_t>(stream);
    filmRecoverKernel<<<(film.nFaces+127)/128,128,0,cudaStream>>>(film,surface,p,status);
    int code=launchError(); if (code) return code;
    filmProfileKernel<<<(film.nFaces+127)/128,128,0,cudaStream>>>(film,surface,packets,p,dt,status);
    code=launchError(); if (code) return code;
    if (film.nEdges>0) {
        filmEdgeKernel<<<(film.nEdges+127)/128,128,0,cudaStream>>>(film,surface,p,dt,status);
        code=launchError(); if (code) return code;
    }
    filmRhsKernel<<<(film.nFaces+127)/128,128,0,cudaStream>>>(film,surface,packets,p,dt,status);
    return launchError();
}
} // namespace chmt

namespace chmt {
namespace {
template<class Input, class Output>
__global__ void contractKernel(const Input* input, Output* output, DeviceStatus* status) {
    Output candidate;
    if (!evaluateFilmContractMath(*input,candidate)) {
        gasDeviceError(status,ErrorCode::InvalidInput,0,0,ErrorLocation::Configuration);
        return;
    }
    *output=candidate;
}
struct ProbeResources {
    void* input=nullptr;
    void* output=nullptr;
    DeviceStatus* status=nullptr;
    cudaStream_t stream=nullptr;
    cudaError_t close() {
        cudaError_t first=cudaSuccess;
        if (stream) {
            const cudaError_t code=cudaStreamSynchronize(stream);
            if (first==cudaSuccess) first=code;
        }
        if (input) {
            const cudaError_t code=cudaFree(input);
            if (first==cudaSuccess) first=code;
            input=nullptr;
        }
        if (output) {
            const cudaError_t code=cudaFree(output);
            if (first==cudaSuccess) first=code;
            output=nullptr;
        }
        if (status) {
            const cudaError_t code=cudaFree(status);
            if (first==cudaSuccess) first=code;
            status=nullptr;
        }
        if (stream) {
            const cudaError_t code=cudaStreamDestroy(stream);
            if (first==cudaSuccess) first=code;
            stream=nullptr;
        }
        return first;
    }
    ~ProbeResources() { close(); }
};
template<class Input, class Output>
bool runContract(const Input& input, Output& output, std::string& error) {
    ProbeResources resources;
    cudaError_t code=cudaStreamCreateWithFlags(&resources.stream,cudaStreamNonBlocking);
    if (code==cudaSuccess) code=cudaMalloc(&resources.input,sizeof(Input));
    if (code==cudaSuccess) code=cudaMalloc(&resources.output,sizeof(Output));
    if (code==cudaSuccess) code=cudaMalloc(reinterpret_cast<void**>(&resources.status),sizeof(DeviceStatus));
    DeviceStatus status;
    Output result;
    if (code==cudaSuccess) code=cudaMemcpyAsync(resources.input,&input,sizeof(Input),
        cudaMemcpyHostToDevice,resources.stream);
    if (code==cudaSuccess) code=cudaMemcpyAsync(resources.status,&status,sizeof(status),
        cudaMemcpyHostToDevice,resources.stream);
    if (code==cudaSuccess) {
        contractKernel<Input,Output><<<1,1,0,resources.stream>>>
            (static_cast<const Input*>(resources.input),static_cast<Output*>(resources.output),resources.status);
        code=cudaPeekAtLastError();
    }
    if (code==cudaSuccess) code=cudaMemcpyAsync(&status,resources.status,sizeof(status),
        cudaMemcpyDeviceToHost,resources.stream);
    if (code==cudaSuccess) code=cudaMemcpyAsync(&result,resources.output,sizeof(result),
        cudaMemcpyDeviceToHost,resources.stream);
    if (code==cudaSuccess) code=cudaStreamSynchronize(resources.stream);
    const cudaError_t cleanup=resources.close();
    if (code==cudaSuccess) code=cleanup;
    if (code!=cudaSuccess) {
        error=std::string("CUDA operator contract: ")+cudaGetErrorString(code); return false;
    }
    if (status.code!=0) {
        error="CUDA operator contract rejected physical input, code "+std::to_string(status.code);
        return false;
    }
    output=result; error.clear(); return true;
}
}
bool evaluateFilmProfileContract(const FilmProfileRequest& input, FilmProfileResult& output,
    std::string& error) { return runContract(input,output,error); }
bool evaluateFilmPressureContract(const FilmPressureRequest& input, FilmPressureResult& output,
    std::string& error) { return runContract(input,output,error); }
bool evaluateFilmPhaseContract(const FilmPhaseRequest& input, FilmPhaseResult& output,
    std::string& error) { return runContract(input,output,error); }
} // namespace chmt
