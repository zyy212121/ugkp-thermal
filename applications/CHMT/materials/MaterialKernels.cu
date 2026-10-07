#include "materials/MaterialKernels.H"
#include "materials/DarcyGradient.H"
#include "materials/MaterialReconstruction.H"
#include "gas/KernelSupport.cuh"
#include "gas/AleFlux.H"
namespace chmt {
namespace {
__global__ void recoverMaterialKernel(SolidView solid, GeometryView geometry, PhysicsConfig p,
    bool endpoint, DeviceStatus* status) {
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if (cell>=solid.nCells || gasStopped(status)) return;
    MaterialPrimitive state;
    DeviceStatus local;
    const Real volume=endpoint ? geometry.newVolume[cell] : geometry.evaluationVolume[cell];
    if (!recoverMaterial(solid.q[cell],volume,p,state,&local,cell)) {
        gasDeviceError(status,local.code ? static_cast<ErrorCode>(local.code) : ErrorCode::InvalidInput,
            cell,local.value,ErrorLocation::Cell);
        return;
    }
    solid.q[cell].porosity=state.porosity;
    solid.temperature[cell]=state.temperature;
    solid.porePressure[cell]=state.pressure;
}
__global__ void materialGradientKernel(SolidView solid, GeometryView geometry, PhysicsConfig p) {
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if (cell>=solid.nCells || gasStopped(solid.status)) return;
    Symmetric3 matrix;
    Vec3 rhsT{}, rhsP{};
    for (int entry=geometry.cellFaceOffsets[cell]; entry<geometry.cellFaceOffsets[cell+1]; ++entry) {
        const int face=geometry.cellFaces[entry];
        if (geometry.boundaryKind[face]==BoundaryKind::Empty
            || geometry.boundaryKind[face]==BoundaryKind::Interface) continue;
        const int other=neighbourCell(cell,face,geometry);
        if (other<0) continue;
        const Vec3 dx=neighbourDisplacement(cell,face,other,geometry);
        Symmetric3 unused;
        addLeastSquares(matrix,rhsT,dx,solid.temperature[other]-solid.temperature[cell]);
        addLeastSquares(unused,rhsP,dx,solid.porePressure[other]-solid.porePressure[cell]);
    }
    solid.gradientTemperature[cell]=solveLeastSquares(matrix,rhsT);
    const Vec3 gradientP=solveLeastSquares(matrix,rhsP);
    solid.gradientPressure[cell]=gradientP;
    Real sensitivity=0;
    for (int entry=geometry.cellFaceOffsets[cell]; entry<geometry.cellFaceOffsets[cell+1]; ++entry) {
        const int face=geometry.cellFaces[entry];
        if (geometry.boundaryKind[face]==BoundaryKind::Empty
            || geometry.boundaryKind[face]==BoundaryKind::Interface) continue;
        const int other=neighbourCell(cell,face,geometry);
        if (other>=0) sensitivity+=darcyGradientSensitivity(matrix,neighbourDisplacement(cell,face,other,geometry));
    }
    solid.pressureGradientSensitivity[cell]=sensitivity;
    Real poreMass=0;
    for (int s=0; s<Ns; ++s) poreMass+=solid.q[cell].pore[s];
    const Real phi=solid.q[cell].porosity;
    const Real rho=phi>0 ? poreMass/(phi*geometry.evaluationVolume[cell]) : 0;
    // poreVelocity is the intrinsic pore velocity, not superficial Darcy speed.
    solid.poreVelocity[cell]=p.permeability>0 && phi>0
        ? (p.gravity*rho-gradientP)*(p.permeability/(p.poreViscosity*phi)) : Vec3{};
}
__device__ Real materialComponent(const SolidQ& q, int index) {
    if (index<Nc) return q.condensed[index];
    index-=Nc;
    if (index<Ns) return q.pore[index];
    index-=Ns;
    if (index<Nr) return q.progress[index];
    return q.energy;
}
__device__ void setMaterialComponent(SolidQ& q, int index, Real value) {
    if (index<Nc) { q.condensed[index]=value; return; }
    index-=Nc;
    if (index<Ns) { q.pore[index]=value; return; }
    index-=Ns;
    if (index<Nr) { q.progress[index]=value; return; }
    q.energy=value;
}
__device__ bool materialFaceDensity(int cell, Vec3 offset, SolidView solid, GeometryView g,
    PhysicsConfig p, SolidQ& out) {
    SolidQ density, increment;
    Real bound=1;
    for (int component=0; component<Nc+Ns+Nr+1; ++component) {
        const Real value=materialComponent(solid.q[cell],component)/g.evaluationVolume[cell];
        setMaterialComponent(density,component,value);
        if (p.spatialOrder==1) continue;
        Symmetric3 matrix;
        Vec3 rhs{};
        Real lo=value, hi=value;
        for (int entry=g.cellFaceOffsets[cell]; entry<g.cellFaceOffsets[cell+1]; ++entry) {
            const int face=g.cellFaces[entry], other=neighbourCell(cell,face,g);
            if (other<0) continue;
            const Real otherValue=materialComponent(solid.q[other],component)/g.evaluationVolume[other];
            addLeastSquares(matrix,rhs,neighbourDisplacement(cell,face,other,g),otherValue-value);
            lo=minValue(lo,otherValue); hi=maxValue(hi,otherValue);
        }
        Real change=dot(solveLeastSquares(matrix,rhs),offset);
        if (p.reconstruction==ReconstructionMode::LimitedLinear) {
            bound=minValue(bound,materialReconstructionLimiter(value,lo,hi,change));
        }
        setMaterialComponent(increment,component,change);
    }
    Real factor=bound;
    for (int iteration=0; iteration<48; ++iteration) {
        SolidQ candidate=addSolid(density,increment,factor);
        candidate.porosity=solid.q[cell].porosity;
        MaterialPrimitive recovered;
        if (recoverMaterial(candidate,1,p,recovered)) {
            candidate.porosity=recovered.porosity; out=candidate; return true;
        }
        factor*=0.5;
    }
    return false;
}
__device__ Vec3 materialFaceOffset(int cell, int face, GeometryView g) {
    const int imageFace=g.owner[face]==cell ? face
        : (g.boundaryKind[face]==BoundaryKind::Periodic ? g.periodicPartner[face] : face);
    return g.faceCentre[imageFace]-g.cellCentre[cell];
}
__global__ void materialFaceKernel(SolidView solid, GeometryView g, PhysicsConfig p,
    Real dt, DeviceStatus* status) {
    const int face=blockIdx.x*blockDim.x+threadIdx.x;
    if (face>=solid.nFaces || gasStopped(status)) return;
    const BoundaryKind kind=g.boundaryKind[face];
    const int left=g.owner[face];
    int right=g.neighbour[face];
    if (kind==BoundaryKind::Periodic) {
        const int pair=g.periodicPartner[face];
        if (face>pair) return;
        right=g.owner[pair];
    }
    // Surface packets replace both physical and mesh-swept material transport.
    if (kind==BoundaryKind::Interface || kind==BoundaryKind::Empty) {
        solid.faceFlux[face]=SolidQ{}; return;
    }
    const Real area=mag(g.areaVector[face]);
    if (!(area>0)) {
        gasDeviceError(status,ErrorCode::InvalidInput,face,area,ErrorLocation::Face); return;
    }
    const Vec3 normal=g.areaVector[face]/area;
    const Real meshRate=g.sweptVolume[face]/dt;
    const int donor=meshRate<=0 || right<0 ? left : right;
    SolidQ donorDensity;
    if (!materialFaceDensity(donor,materialFaceOffset(donor,face,g),solid,g,p,donorDensity)) {
        gasDeviceError(status,ErrorCode::PropertyRange,face,donor,ErrorLocation::Face); return;
    }
    SolidQ flux=materialAleFlux(donorDensity,1,meshRate);
    MaterialPrimitive a, b;
    DeviceStatus local;
    if (!recoverMaterial(solid.q[left],g.evaluationVolume[left],p,a,&local,left)) {
        gasDeviceError(status,ErrorCode::PropertyRange,left,local.value,ErrorLocation::Cell); return;
    }
    if (right>=0) {
        if (!recoverMaterial(solid.q[right],g.evaluationVolume[right],p,b,&local,right)) {
            gasDeviceError(status,ErrorCode::PropertyRange,right,local.value,ErrorLocation::Cell); return;
        }
        const Vec3 dx=neighbourDisplacement(left,face,right,g);
        const Real distance=dot(dx,normal);
        Real normalPressureGradient=0;
        SolidQ physical;
        if (!darcyNormalPressureGradient(dx,normal,b.pressure-a.pressure,
            solid.gradientPressure[left],solid.gradientPressure[right],normalPressureGradient)
            || !darcyConductionFluxWithGradient(solid.q[left],solid.q[right],a,b,
            g.evaluationVolume[left],g.evaluationVolume[right],area,distance,normal,normalPressureGradient,p,physical)) {
            gasDeviceError(status,ErrorCode::InvalidInput,face,distance,ErrorLocation::Face); return;
        }
        if (p.spatialOrder==2 && p.permeability>0) {
            const Real volumeRate=-area*p.permeability/p.poreViscosity
                *(normalPressureGradient-0.5*(a.poreDensity+b.poreDensity)*dot(p.gravity,normal));
            const int poreDonor=volumeRate>=0 ? left : right;
            SolidQ reconstructed;
            MaterialPrimitive recovered;
            if (!materialFaceDensity(poreDonor,materialFaceOffset(poreDonor,face,g),solid,g,p,reconstructed)
                || !recoverMaterial(reconstructed,1,p,recovered)) {
                gasDeviceError(status,ErrorCode::PropertyRange,face,poreDonor,ErrorLocation::Face); return;
            }
            for (int species=0; species<Ns; ++species) {
                physical.energy-=physical.pore[species]
                    *speciesH(p.species[species],volumeRate>=0 ? a.temperature : b.temperature);
                physical.pore[species]=recovered.porosity>0
                    ? volumeRate*reconstructed.pore[species]/recovered.porosity : 0;
                physical.energy+=physical.pore[species]*speciesH(p.species[species],recovered.temperature);
            }
        }
        const Real conductance=seriesConductance(a.conductivity,b.conductivity,distance/2,distance/2);
        const Vec3 gradient=correctedGradient(
            (solid.gradientTemperature[left]+solid.gradientTemperature[right])*0.5,
            dx,b.temperature-a.temperature);
        physical.energy-=area*conductance*(a.temperature-b.temperature);
        physical.energy-=area*conductance*distance*dot(gradient,normal);
        flux=addSolid(flux,physical);
    } else if (g.boundaryPrimitive && g.boundaryPrimitive[face].temperature>0) {
        const Real distance=dot(g.faceCentre[face]-g.cellCentre[left],normal);
        if (!(distance>0)) {
            gasDeviceError(status,ErrorCode::InvalidInput,face,distance,ErrorLocation::Face); return;
        }
        flux.energy+=area*a.conductivity*(a.temperature-g.boundaryPrimitive[face].temperature)/distance;
    }
    solid.faceFlux[face]=flux;
    if (kind==BoundaryKind::Periodic) {
        solid.faceFlux[g.periodicPartner[face]]=addSolid(SolidQ{},flux,-1);
    }
}
__global__ void materialRhsKernel(SolidView solid, GeometryView g, PhysicsConfig p,
    Real dt, DeviceStatus* status) {
    const int cell=blockIdx.x*blockDim.x+threadIdx.x;
    if (cell>=solid.nCells || gasStopped(status)) return;
    Real rates[Nr]{};
    DeviceStatus local;
    if (!boundedReactionRates(solid.q[cell],solid.stageBase[cell],g.evaluationVolume[cell],p,dt,rates,&local,cell)) {
        gasDeviceError(status,local.code ? static_cast<ErrorCode>(local.code) : ErrorCode::InvalidInput,
            cell,local.value,ErrorLocation::Cell);
        return;
    }
    SolidQ rhs;
    for (int r=0; r<p.nReactions; ++r) {
        for (int c=0; c<Nc; ++c) rhs.condensed[c]+=rates[r]*p.reactions[r].condensedNu[c];
        for (int species=0; species<Ns; ++species) rhs.pore[species]+=rates[r]*p.reactions[r].gasNu[species];
    }
    for (int r=0; r<Nr; ++r) {
        rhs.progress[r]=rates[r];
        if (solid.reactionRate) solid.reactionRate[cell*Nr+r]=rates[r];
    }
    for (int entry=g.cellFaceOffsets[cell]; entry<g.cellFaceOffsets[cell+1]; ++entry) {
        rhs=addSolid(rhs,solid.faceFlux[g.cellFaces[entry]],-g.cellFaceSigns[entry]);
    }
    Real poreMass=0;
    for (int s=0; s<Ns; ++s) poreMass+=solid.q[cell].pore[s];
    rhs.energy+=poreMass*dot(p.gravity,solid.poreVelocity[cell]);
    solid.rhs[cell]=rhs;
}
bool validMaterialPointers(SolidView s, GeometryView g, bool endpoint) {
    return s.nCells>=0 && s.nFaces==g.nFaces && s.nCells==g.nCells
        && s.q && s.temperature && s.porePressure
        && (endpoint ? g.newVolume!=nullptr : g.evaluationVolume!=nullptr);
}
}
int launchMaterialValidate(SolidView solid, GeometryView g, const PhysicsConfig& p,
    DeviceStatus* status, void* stream) {
    if (solid.nCells==0) return 0;
    if (!status || !validMaterialPointers(solid,g,true)) return static_cast<int>(cudaErrorInvalidValue);
    recoverMaterialKernel<<<(solid.nCells+127)/128,128,0,static_cast<cudaStream_t>(stream)>>>
        (solid,g,p,true,status);
    return launchError();
}
int launchMaterialRhs(SolidView solid, GeometryView g, PacketView, const PhysicsConfig& p,
    Real dt, DeviceStatus* status, void* stream) {
    if (solid.nCells==0) return 0;
    if (!status || !validMaterialPointers(solid,g,false) || !solid.stageBase || !solid.rhs || !solid.faceFlux
        || !solid.gradientTemperature || !solid.poreVelocity || !solid.gradientPressure
        || !solid.pressureGradientSensitivity || !g.owner || !g.neighbour
        || !g.boundaryKind || !g.periodicPartner || !g.cellFaceOffsets || !g.cellFaces
        || !g.cellFaceSigns || !g.cellCentre || !g.faceCentre || !g.areaVector || !g.sweptVolume
        || !finite(dt) || dt<=0 || (p.enableReactions && !solid.reactionRate)) {
        return static_cast<int>(cudaErrorInvalidValue);
    }
    solid.status=status;
    cudaStream_t cudaStream=static_cast<cudaStream_t>(stream);
    recoverMaterialKernel<<<(solid.nCells+127)/128,128,0,cudaStream>>>(solid,g,p,false,status);
    int code=launchError(); if (code) return code;
    materialGradientKernel<<<(solid.nCells+127)/128,128,0,cudaStream>>>(solid,g,p);
    code=launchError(); if (code) return code;
    if (solid.nFaces>0) {
        materialFaceKernel<<<(solid.nFaces+127)/128,128,0,cudaStream>>>(solid,g,p,dt,status);
        code=launchError(); if (code) return code;
    }
    materialRhsKernel<<<(solid.nCells+127)/128,128,0,cudaStream>>>(solid,g,p,dt,status);
    return launchError();
}
} // namespace chmt
