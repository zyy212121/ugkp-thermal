#ifndef CHMT_MATERIALS_MATERIALTRANSPORTMATH_H
#define CHMT_MATERIALS_MATERIALTRANSPORTMATH_H
#include "materials/DryMaterialAdapter.H"
#include "materials/MaterialFaceTransport.H"
#include "materials/DarcyGradient.H"
#include "materials/MaterialReconstruction.H"
#include "coupling/Coordinator.H"
namespace chmt {

struct MaterialMeshView {
    int cells=0,faces=0;
    const Real* volumes=nullptr;
    const Vec3* cellCentres=nullptr;
    const int* owner=nullptr;
    const int* neighbour=nullptr;
    const Vec3* areaVectors=nullptr;
    const Vec3* faceCentres=nullptr;
    const BoundaryKind* boundaryKind=nullptr;
    const int* periodicPartner=nullptr;
    const int* cellFaceOffsets=nullptr;
    const int* cellFaces=nullptr;
    const int* cellFaceSigns=nullptr;



    const int* diffusionOffsets=nullptr;
    const int* diffusionFaces=nullptr;
    const int* diffusionSides=nullptr;
};
struct MaterialTransportWorkspace {
    MaterialPrimitive* primitive=nullptr;
    Vec3* pressureGradient=nullptr;
    Vec3* poreVelocity=nullptr;
    Real* gradientSensitivity=nullptr;
    Real* stabilityRate=nullptr;
    Real* faceDiffusionLossLeft=nullptr;
    Real* faceDiffusionLossRight=nullptr;
    Real* cellDtLimit=nullptr;
    SolidQ* faceFlux=nullptr;
    SolidQ* rates=nullptr;
    int* cellErrors=nullptr;
    int* faceErrors=nullptr;
    MaterialFaceTransport* splitFlux=nullptr;
};
struct MaterialTransportDeviceResult {
    Real dtLimit=std::numeric_limits<Real>::max();
    int errorCode=0,entity=-1;
    bool numerical=false;
};
enum class MaterialTransportError : int {
    None=0,InvalidControl,InvalidDiffusivity,CellEos,CoincidentCentres,
    PressureGradient,GradientSensitivity,DarcyViscosity,DarcyStencil,
    AleSweep,AleReconstruction,DarcyFace,DarcyReconstruction,InvalidTimestep
};
namespace materialTransport {
CHMT_HD inline int otherCell(const MaterialMeshView& m,int cell,int face){
    if(m.neighbour[face]>=0)return cell==m.owner[face]?m.neighbour[face]:m.owner[face];
    if(m.boundaryKind[face]==BoundaryKind::Periodic)return m.owner[m.periodicPartner[face]];
    return -1;
}
CHMT_HD inline Vec3 displacement(const MaterialMeshView& m,int cell,int face,int other){
    Vec3 dx=m.cellCentres[other]-m.cellCentres[cell];
    if(m.boundaryKind[face]==BoundaryKind::Periodic)dx+=m.faceCentres[face]-m.faceCentres[m.periodicPartner[face]];
    return dx;
}
CHMT_HD inline Real component(const SolidQ& q,int i){if(i<Nc)return q.condensed[i];i-=Nc;if(i<Ns)return q.pore[i];i-=Ns;if(i<Nr)return q.progress[i];return q.energy;}
CHMT_HD inline void setComponent(SolidQ& q,int i,Real x){if(i<Nc){q.condensed[i]=x;return;}i-=Nc;if(i<Ns){q.pore[i]=x;return;}i-=Ns;if(i<Nr){q.progress[i]=x;return;}q.energy=x;}
CHMT_HD inline bool reconstruct(const MaterialMeshView& m,const SolidQ* q,const PhysicsConfig& p,
    int cell,int face,SolidQ& out,MaterialPrimitive* reconstructed=nullptr){
    const int image=m.owner[face]==cell?face:(m.boundaryKind[face]==BoundaryKind::Periodic?m.periodicPartner[face]:face);
    const Vec3 offset=m.faceCentres[image]-m.cellCentres[cell];SolidQ density,increment;Real bound=1;
    for(int k=0;k<Nc+Ns+Nr+1;++k){const Real value=component(q[cell],k)/m.volumes[cell];setComponent(density,k,value);
        if(p.spatialOrder==1)continue;
        Symmetric3 matrix;Vec3 rhs{};Real lo=value,hi=value;
        for(int e=m.cellFaceOffsets[cell];e<m.cellFaceOffsets[cell+1];++e){const int f=m.cellFaces[e],other=otherCell(m,cell,f);if(other<0)continue;
            const Real v=component(q[other],k)/m.volumes[other];addLeastSquares(matrix,rhs,displacement(m,cell,f,other),v-value);lo=minValue(lo,v);hi=maxValue(hi,v);}
        const Real change=dot(solveLeastSquares(matrix,rhs,p.numerics.reconstructionMaximumIterations),offset);if(!finite(change))return false;setComponent(increment,k,change);
        if(p.reconstruction==ReconstructionMode::LimitedLinear)bound=minValue(bound,materialReconstructionLimiter(value,lo,hi,change));
    }
    SolidQ candidate=addSolid(density,increment,bound);candidate.porosity=q[cell].porosity;MaterialPrimitive state;
    if(!recoverMaterial(candidate,1,p,state))return false;
    if(reconstructed)*reconstructed=state;
    candidate.porosity=state.porosity;out=candidate;return true;
}

CHMT_HD inline MaterialTransportError recoverCell(const MaterialMeshView& m,const SolidQ* q,
    const PhysicsConfig& p,int c,const MaterialTransportWorkspace& w) {
    return recoverMaterial<false>(q[c],m.volumes[c],p,w.primitive[c])
        ? MaterialTransportError::None : MaterialTransportError::CellEos;
}
CHMT_HD inline MaterialTransportError pressureCell(const MaterialMeshView& m,
    const PhysicsConfig& p,int c,const MaterialTransportWorkspace& w) {
    Symmetric3 matrix;Vec3 rhs{};w.gradientSensitivity[c]=0;w.poreVelocity[c]={};
        for(int e=m.cellFaceOffsets[c];e<m.cellFaceOffsets[c+1];++e){const int f=m.cellFaces[e],other=otherCell(m,c,f);
            if(m.boundaryKind[f]==BoundaryKind::Empty||m.boundaryKind[f]==BoundaryKind::Interface)continue;
            if(other<0)continue;
            const Vec3 dx=displacement(m,c,f,other);addLeastSquares(matrix,rhs,dx,w.primitive[other].pressure-w.primitive[c].pressure);
            const Real distance=mag(dx);if(!(distance>0)){return MaterialTransportError::CoincidentCentres;}
        }
        const Vec3 grad=solveLeastSquares(matrix,rhs,p.numerics.reconstructionMaximumIterations);if(!finite(grad)){return MaterialTransportError::PressureGradient;}w.pressureGradient[c]=grad;const auto& a=w.primitive[c];
        for(int e=m.cellFaceOffsets[c];e<m.cellFaceOffsets[c+1];++e){const int f=m.cellFaces[e],other=otherCell(m,c,f);
            if(other<0||m.boundaryKind[f]==BoundaryKind::Empty||m.boundaryKind[f]==BoundaryKind::Interface)continue;
            w.gradientSensitivity[c]+=darcyGradientSensitivity(matrix,displacement(m,c,f,other),p.numerics.reconstructionMaximumIterations);}
        if(!finite(w.gradientSensitivity[c])){return MaterialTransportError::GradientSensitivity;}
        if(p.permeability>0&&a.porosity>0){if(!(p.poreViscosity>0)){return MaterialTransportError::DarcyViscosity;}
            w.poreVelocity[c]=(p.gravity*a.poreDensity-grad)*(p.permeability/(p.poreViscosity*a.porosity));}
    return MaterialTransportError::None;
}
CHMT_HD inline MaterialTransportError stabilityCell(const MaterialMeshView& m,const SolidQ* q,
    const PhysicsConfig& p,const Real* swept,Real interval,int c,const MaterialTransportWorkspace& w) {
    Real pressureRate=0,sweepRate=0;
        const Real storage=porePressureStorage(q[c],w.primitive[c].porosity*m.volumes[c],w.primitive[c].temperature,p);
        for(int e=m.cellFaceOffsets[c];e<m.cellFaceOffsets[c+1];++e){const int f=m.cellFaces[e],other=otherCell(m,c,f);
            if(m.boundaryKind[f]==BoundaryKind::Empty||m.boundaryKind[f]==BoundaryKind::Interface)continue;
            if(swept!=nullptr)sweepRate+=absValue(swept[f])/interval/m.volumes[c];
            if(other<0)continue;
            const Real area=mag(m.areaVectors[f]);const Vec3 normal=m.areaVectors[f]*(m.cellFaceSigns[e]/area);
            Real inverseLength=0;
            if(!darcyPressureStencilInverseLength<false>(displacement(m,c,f,other),normal,
                w.gradientSensitivity[c],w.gradientSensitivity[other],inverseLength)){return MaterialTransportError::DarcyStencil;}
            if(storage>0)pressureRate+=darcyPressureRelaxation(storage,p.permeability,p.poreViscosity,
                maxValue(w.primitive[c].poreDensity,w.primitive[other].poreDensity),area,1/inverseLength);
        }
        w.stabilityRate[c]=pressureRate+sweepRate;
    return MaterialTransportError::None;
}
#include "materials/MaterialTransportFace.inl"
CHMT_HD inline void gatherCell(const MaterialMeshView& m,const SolidQ* q,
    const PhysicsConfig& p,Real cfl,int c,const MaterialTransportWorkspace& w) {
    w.rates[c]={};SolidQ outgoing;Real dtLimit=std::numeric_limits<Real>::max();


    for(int e=m.diffusionOffsets[c];e<m.diffusionOffsets[c+1];++e){
        const int f=m.diffusionFaces[e];
        w.stabilityRate[c]+=m.diffusionSides[e]==0?w.faceDiffusionLossLeft[f]:w.faceDiffusionLossRight[f];
    }
    for(int e=m.cellFaceOffsets[c];e<m.cellFaceOffsets[c+1];++e){const auto& flux=w.faceFlux[m.cellFaces[e]];const int sign=m.cellFaceSigns[e];
            w.rates[c]=addSolid(w.rates[c],flux,-sign);
            for(int k=0;k<Nc;++k)outgoing.condensed[k]+=maxValue(0,sign*flux.condensed[k]);
            for(int s=0;s<Ns;++s)outgoing.pore[s]+=maxValue(0,sign*flux.pore[s]);}
        if(w.stabilityRate[c]>0)dtLimit=minValue(dtLimit,cfl/w.stabilityRate[c]);
        Real poreMass=0;for(int s=0;s<Ns;++s)poreMass+=q[c].pore[s];const Real power=poreMass*dot(p.gravity,w.poreVelocity[c]);
        w.rates[c].energy+=power;
        for(int k=0;k<Nc;++k)if(outgoing.condensed[k]>0)dtLimit=minValue(dtLimit,cfl*q[c].condensed[k]/outgoing.condensed[k]);
        for(int s=0;s<Ns;++s)if(outgoing.pore[s]>0)dtLimit=minValue(dtLimit,cfl*q[c].pore[s]/outgoing.pore[s]);
    w.cellDtLimit[c]=dtLimit;
}
inline const char* errorMessage(MaterialTransportError code) {
    switch(code){
    case MaterialTransportError::None:return "";
    case MaterialTransportError::InvalidControl:return "invalid material transport interval/control";
    case MaterialTransportError::InvalidDiffusivity:return "invalid material effective pore diffusivity";
    case MaterialTransportError::CellEos:return "material transport EOS recovery failed in cell ";
    case MaterialTransportError::CoincidentCentres:return "coincident material cell centres";
    case MaterialTransportError::PressureGradient:return "nonfinite material pressure gradient";
    case MaterialTransportError::GradientSensitivity:return "nonfinite material gradient sensitivity";
    case MaterialTransportError::DarcyViscosity:return "invalid Darcy viscosity";
    case MaterialTransportError::DarcyStencil:return "invalid material Darcy stencil geometry";
    case MaterialTransportError::AleSweep:return "nonfinite material ALE sweep";
    case MaterialTransportError::AleReconstruction:return "material ALE reconstruction failed";
    case MaterialTransportError::DarcyFace:return "invalid material Darcy face";
    case MaterialTransportError::DarcyReconstruction:return "material Darcy reconstruction failed";
    case MaterialTransportError::InvalidTimestep:return "invalid material transport timestep";
    }
    return "invalid material transport result";
}
}
}
#endif
