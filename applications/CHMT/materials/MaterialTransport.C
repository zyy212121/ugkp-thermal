#include "materials/MaterialTransport.H"
#include "materials/MaterialTransportMath.H"
#include "materials/MaterialReactionMath.H"
#include "materials/DryMaterialAdapter.H"
#include "materials/DarcyGradient.H"
#include "materials/MaterialReconstruction.H"
#include "materials/StencilMath.H"
#include "coupling/Coordinator.H"
#include "mesh/SweepConstraints.H"
#include <algorithm>
namespace chmt {
bool validateMaterialTransportMesh(const HostMesh& m,std::size_t cells,std::string& error){
    const std::size_t nf=m.owner.size();
    if(m.volumes.size()!=cells||m.cellCentres.size()!=cells||m.neighbour.size()!=nf||m.areaVectors.size()!=nf||m.faceCentres.size()!=nf||m.boundaryKind.size()!=nf||m.periodicPartner.size()!=nf||m.cellFaceOffsets.size()!=cells+1||m.cellFaces.size()!=m.cellFaceSigns.size()||m.cellFaceOffsets.empty()||m.cellFaceOffsets.front()!=0||m.cellFaceOffsets.back()!=static_cast<int>(m.cellFaces.size())){error="material transport mesh/addressing size mismatch";return false;}
    for(std::size_t c=0;c<cells;++c)if(!(m.volumes[c]>0)||!finite(m.volumes[c])||!finite(m.cellCentres[c])||m.cellFaceOffsets[c]>m.cellFaceOffsets[c+1]){error="invalid material cell geometry";return false;}
    for(std::size_t f=0;f<nf;++f){const int o=m.owner[f],n=m.neighbour[f];
        if(o<0||o>=static_cast<int>(cells)||n< -1||n>=static_cast<int>(cells)||!finite(m.areaVectors[f])||!(mag(m.areaVectors[f])>0)||!finite(m.faceCentres[f])){error="invalid material face geometry";return false;}
        if(m.boundaryKind[f]==BoundaryKind::Periodic){const int pair=m.periodicPartner[f];
            if(pair<0||pair>=static_cast<int>(nf)||m.periodicPartner[pair]!=static_cast<int>(f)||m.boundaryKind[pair]!=BoundaryKind::Periodic){error="invalid material cyclic pairing";return false;}}
    }
    for(std::size_t c=0;c<cells;++c)for(int e=m.cellFaceOffsets[c];e<m.cellFaceOffsets[c+1];++e){const int f=m.cellFaces[e];
        if(f<0||f>=static_cast<int>(nf)||(m.cellFaceSigns[e]!=1&&m.cellFaceSigns[e]!=-1)||(m.cellFaceSigns[e]==1?m.owner[f]!=static_cast<int>(c):m.neighbour[f]!=static_cast<int>(c))){error="invalid material cell-face incidence";return false;}}
    return true;
}
bool applyMaterialPacketBatchImpl(const std::vector<SolidQ>& base,const std::vector<ExchangePacket>& packets,std::vector<SolidQ>& output,std::string& error,bool* numericalFailure){
    if(numericalFailure)*numericalFailure=false;
    std::vector<SolidQ> next=base;
    std::vector<std::array<long double,Nc+Ns+1>> sum(base.size()),scale(base.size());std::vector<std::size_t> operations(base.size());
    for(std::size_t cell=0;cell<base.size();++cell){for(int c=0;c<Nc;++c){sum[cell][c]=base[cell].condensed[c];scale[cell][c]=absValue(base[cell].condensed[c]);}
        for(int species=0;species<Ns;++species){sum[cell][Nc+species]=base[cell].pore[species];scale[cell][Nc+species]=absValue(base[cell].pore[species]);}sum[cell][Nc+Ns]=base[cell].energy;}
    // Scatter each actual packet once. No full-domain scan per gas record.
    for(const auto& packet:packets)if(requiredConsumers(packet.kind)&ConsumeSolid){
        if((packet.solidCell<0||static_cast<std::size_t>(packet.solidCell)>=base.size())){if(numericalFailure)*numericalFailure=true;error="material packet owner invalid";return false;}
        const std::size_t cell=packet.solidCell;const auto delta=packetDelta(packet);++operations[cell];sum[cell][Nc+Ns]+=delta.solid.energy;
        for(int c=0;c<Nc;++c){sum[cell][c]+=delta.solid.condensed[c];scale[cell][c]+=absValue(delta.solid.condensed[c]);}
        for(int species=0;species<Ns;++species){sum[cell][Nc+species]+=delta.solid.pore[species];scale[cell][Nc+species]+=absValue(delta.solid.pore[species]);}}
    for(std::size_t cell=0;cell<base.size();++cell){for(int component=0;component<Nc+Ns;++component){Real value=static_cast<Real>(sum[cell][component]);
            if(value<0){const long double bound=(operations[cell]+2)*std::numeric_limits<Real>::epsilon()*scale[cell][component];
                if(-sum[cell][component]>bound){{if(numericalFailure)*numericalFailure=true;error="material packet batch exceeds physical donor inventory";return false;}}
                else {value=0;}
            }
            if(!finite(value)){if(numericalFailure)*numericalFailure=true;error="nonfinite material packet result";return false;}
            if(component<Nc)next[cell].condensed[component]=value;else next[cell].pore[component-Nc]=value;}
        next[cell].energy=static_cast<Real>(sum[cell][Nc+Ns]);if(!finite(next[cell].energy)){if(numericalFailure)*numericalFailure=true;error="nonfinite material packet energy";return false;}}
    output=std::move(next);error.clear();return true;
}
bool applyMaterialPacketBatch(const std::vector<SolidQ>& base,const std::vector<ExchangePacket>& packets,std::vector<SolidQ>& output,std::string& error,bool* numericalFailure){
    return applyMaterialPacketBatchImpl(base,packets,output,error,numericalFailure);
}

bool materialPointsClose(const HostMesh& a,const std::vector<Vec3>& b,const Tolerances& tol){
    if(a.points.size()!=b.size())return false;
    Real length=0;
    for(Real volume:a.volumes){if(!finite(volume)||volume<=0)return false;length=maxValue(length,::cbrt(volume));}
    const Real tolerance=tol.absoluteGeometry+tol.relativeGeometry*length;
    for(std::size_t i=0;i<b.size();++i)if(!finite(b[i])||!finite(a.points[i])||mag(b[i]-a.points[i])>tolerance)return false;
    return true;
}
bool materialSweepsCompatible(const HostMesh& mesh,const SurfaceMesh& surface,const std::vector<Real>& targets,const std::vector<Real>& actual,const Tolerances& tol){
    if(targets.size()!=surface.solidFace.size()||actual.size()!=mesh.owner.size())return false;
    std::vector<Real> limits;if(!materialSweepResidualLimits(mesh,surface.solidFace,limits))return false;
    for(std::size_t f=0;f<targets.size();++f){const int face=surface.solidFace[f];if(face<0)continue;
        if(static_cast<std::size_t>(face)>=mesh.owner.size())return false;
        const int cell=mesh.owner[face];
        if(cell<0||static_cast<std::size_t>(cell)>=mesh.volumes.size())return false;
        const Real tolerance=minValue(limits[f],tol.absoluteGeometry+tol.relativeGeometry*maxValue(absValue(actual[face]),absValue(targets[f]))+64*std::numeric_limits<Real>::epsilon()*mesh.volumes[cell]);
        if(!finite(actual[face])||!finite(targets[f])||absValue(actual[face]-targets[f])>tolerance)return false;}
    return true;
}
bool evaluateMaterialTransportImpl(const HostMesh& m,const std::vector<SolidQ>& q,const PhysicsConfig& p,
    const std::vector<Real>& swept,Real interval,Real cfl,MaterialTransportResult& output,std::string& error,bool* numericalFailure){
    if(numericalFailure)*numericalFailure=true;
    if((!(interval>0)||!finite(interval)||!(finite(cfl)&&cfl>0)||(!swept.empty()&&swept.size()!=m.owner.size()))){error="invalid material transport interval/control";return false;}
    if(q.empty()){output=MaterialTransportResult{};error.clear();return true;}
    if(!validateMaterialTransportMesh(m,q.size(),error))return false;
    MaterialTransportResult result;result.rates.resize(q.size());result.poreVelocity.resize(q.size());result.faceFlux.resize(m.owner.size());
    std::vector<MaterialPrimitive> primitive(q.size());
    std::vector<Vec3> pressureGradient(q.size());
    std::vector<Real> gradientSensitivity(q.size()),stabilityRate(q.size()),cellDtLimit(q.size());
    std::vector<Real> lossLeft(m.owner.size()),lossRight(m.owner.size());
    // Construct the physical-face gather independently from stencil incidence.
    // This is the original global-face diffusion-loss order, including both
    // sides of a self-periodic pair at the same cell.
    std::vector<int> offsets(q.size()+1),diffusionFaces,diffusionSides;
    auto visit=[&](auto append){for(std::size_t f=0;f<m.owner.size();++f){
        const auto kind=m.boundaryKind[f];int right=m.neighbour[f];
        if(kind==BoundaryKind::Interface||kind==BoundaryKind::Empty)continue;
        if(kind==BoundaryKind::Periodic){if(static_cast<int>(f)>m.periodicPartner[f])continue;right=m.owner[m.periodicPartner[f]];}
        if(right>=0){append(m.owner[f],static_cast<int>(f),0);append(right,static_cast<int>(f),1);}
    }};
    visit([&](int cell,int,int){++offsets[cell+1];});
    for(std::size_t c=0;c<q.size();++c)offsets[c+1]+=offsets[c];
    diffusionFaces.resize(offsets.back());diffusionSides.resize(offsets.back());auto cursor=offsets;
    visit([&](int cell,int face,int side){const int index=cursor[cell]++;diffusionFaces[index]=face;diffusionSides[index]=side;});
    const MaterialMeshView mesh{static_cast<int>(q.size()),static_cast<int>(m.owner.size()),m.volumes.data(),m.cellCentres.data(),
        m.owner.data(),m.neighbour.data(),m.areaVectors.data(),m.faceCentres.data(),m.boundaryKind.data(),m.periodicPartner.data(),
        m.cellFaceOffsets.data(),m.cellFaces.data(),m.cellFaceSigns.data(),offsets.data(),diffusionFaces.data(),diffusionSides.data()};
    const MaterialTransportWorkspace w{primitive.data(),pressureGradient.data(),result.poreVelocity.data(),gradientSensitivity.data(),
        stabilityRate.data(),lossLeft.data(),lossRight.data(),cellDtLimit.data(),result.faceFlux.data(),result.rates.data(),nullptr,nullptr};
    const Real* sweep=swept.empty()?nullptr:swept.data();
    auto accepted=[&](MaterialTransportError code,int entity){
        if(code==MaterialTransportError::None)return true;
        error=materialTransport::errorMessage(code);
        if(code==MaterialTransportError::CellEos)error+=std::to_string(entity);
        return false;
    };
    for(int species=0;species<Ns;++species)if(!finite(p.poreEffectiveDiffusivity[species])||p.poreEffectiveDiffusivity[species]<0){
        error="invalid material effective pore diffusivity";return false;
    }
    for(int c=0;c<mesh.cells;++c)if(!accepted(materialTransport::recoverCell(mesh,q.data(),p,c,w),c))return false;
    for(int c=0;c<mesh.cells;++c)if(!accepted(materialTransport::pressureCell(mesh,p,c,w),c))return false;
    for(int c=0;c<mesh.cells;++c)if(!accepted(materialTransport::stabilityCell(mesh,q.data(),p,sweep,interval,c,w),c))return false;
    for(int f=0;f<mesh.faces;++f)if(!accepted(materialTransport::face(mesh,q.data(),p,sweep,interval,f,w),f))return false;
    for(int c=0;c<mesh.cells;++c){materialTransport::gatherCell(mesh,q.data(),p,cfl,c,w);result.dtLimit=minValue(result.dtLimit,cellDtLimit[c]);}
    if((!finite(result.dtLimit)||result.dtLimit<=0)){error="invalid material transport timestep";return false;}
    output=std::move(result);if(numericalFailure)*numericalFailure=false;error.clear();return true;
}
bool evaluateMaterialTransport(const HostMesh& m,const std::vector<SolidQ>& q,const PhysicsConfig& p,
    const std::vector<Real>& swept,Real interval,Real cfl,MaterialTransportResult& output,std::string& error,bool* numericalFailure){
    return evaluateMaterialTransportImpl(m,q,p,swept,interval,cfl,output,error,numericalFailure);
}

bool advanceMaterialReactionsImpl(const SolidQ& base,Real volume,const PhysicsConfig& p,Real interval,
    Real maxStep,Real cfl,int maximum,SolidQ& out,std::uint64_t& accepted,std::string& error,bool* numericalFailure){
    const auto status=advanceMaterialReactionsMath(base,volume,p,interval,maxStep,cfl,maximum,out);
    if(numericalFailure)*numericalFailure=status.numericalFailure;
    error=materialReactionMessage(status.code);
    if(status.code!=MaterialReactionCode::Success)return false;
    accepted=status.acceptedSteps;return true;
}

bool advanceMaterialReactions(const SolidQ& base,Real volume,const PhysicsConfig& p,Real interval,
    Real maxStep,Real cfl,int maximum,SolidQ& out,std::uint64_t& accepted,std::string& error,bool* numericalFailure){
    return advanceMaterialReactionsImpl(base,volume,p,interval,maxStep,cfl,maximum,out,accepted,error,numericalFailure);
}

} // namespace chmt
