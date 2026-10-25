#include "materials/MaterialTransport.H"
#include "materials/DryMaterialAdapter.H"
#include "materials/DarcyGradient.H"
#include "materials/MaterialReconstruction.H"
#include "materials/StencilMath.H"
#include "coupling/Coordinator.H"
#include "mesh/SweepConstraints.H"
#include <algorithm>
namespace chmt { namespace {
int otherCell(const HostMesh& m,int cell,int face){
    if(m.neighbour[face]>=0)return cell==m.owner[face]?m.neighbour[face]:m.owner[face];
    if(m.boundaryKind[face]==BoundaryKind::Periodic)return m.owner[m.periodicPartner[face]];
    return -1;
}
Vec3 displacement(const HostMesh& m,int cell,int face,int other){
    Vec3 dx=m.cellCentres[other]-m.cellCentres[cell];
    if(m.boundaryKind[face]==BoundaryKind::Periodic)dx+=m.faceCentres[face]-m.faceCentres[m.periodicPartner[face]];
    return dx;
}
Real component(const SolidQ& q,int i){if(i<Nc)return q.condensed[i];i-=Nc;if(i<Ns)return q.pore[i];i-=Ns;if(i<Nr)return q.progress[i];return q.energy;}
void setComponent(SolidQ& q,int i,Real x){if(i<Nc){q.condensed[i]=x;return;}i-=Nc;if(i<Ns){q.pore[i]=x;return;}i-=Ns;if(i<Nr){q.progress[i]=x;return;}q.energy=x;}
bool validMesh(const HostMesh& m,std::size_t cells,std::string& error){
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
bool reconstruct(const HostMesh& m,const std::vector<SolidQ>& q,const PhysicsConfig& p,
    int cell,int face,SolidQ& out){
    const int image=m.owner[face]==cell?face:(m.boundaryKind[face]==BoundaryKind::Periodic?m.periodicPartner[face]:face);
    const Vec3 offset=m.faceCentres[image]-m.cellCentres[cell];SolidQ density,increment;Real bound=1;
    for(int k=0;k<Nc+Ns+Nr+1;++k){const Real value=component(q[cell],k)/m.volumes[cell];setComponent(density,k,value);
        if(p.spatialOrder==1)continue;
        Symmetric3 matrix;Vec3 rhs{};Real lo=value,hi=value;
        for(int e=m.cellFaceOffsets[cell];e<m.cellFaceOffsets[cell+1];++e){const int f=m.cellFaces[e],other=otherCell(m,cell,f);if(other<0)continue;
            const Real v=component(q[other],k)/m.volumes[other];addLeastSquares(matrix,rhs,displacement(m,cell,f,other),v-value);lo=minValue(lo,v);hi=maxValue(hi,v);}
        const Real change=dot(solveLeastSquares(matrix,rhs),offset);if(!finite(change))return false;setComponent(increment,k,change);
        if(p.reconstruction==ReconstructionMode::LimitedLinear)bound=minValue(bound,materialReconstructionLimiter(value,lo,hi,change));
    }
    SolidQ candidate=addSolid(density,increment,bound);candidate.porosity=q[cell].porosity;MaterialPrimitive state;
    if(!recoverMaterial(candidate,1,p,state))return false;
    candidate.porosity=state.porosity;out=candidate;return true;
}
} // namespace
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
    if(!validMesh(m,q.size(),error)){
        return false;
    }

    MaterialTransportResult result;result.rates.resize(q.size());result.poreVelocity.resize(q.size());result.faceFlux.resize(m.owner.size());
    std::vector<MaterialPrimitive> primitive(q.size());std::vector<SolidQ> outgoing(q.size());
    std::vector<Vec3> pressureGradient(q.size());std::vector<Real> gradientSensitivity(q.size()),stabilityRate(q.size());
    for(int species=0;species<Ns;++species)if(!finite(p.poreEffectiveDiffusivity[species])||p.poreEffectiveDiffusivity[species]<0){
        error="invalid material effective pore diffusivity";return false;
    }
    for(std::size_t c=0;c<q.size();++c)if(!recoverMaterial<false>(q[c],m.volumes[c],p,primitive[c])){error="material transport EOS recovery failed in cell "+std::to_string(c);return false;}
    for(std::size_t c=0;c<q.size();++c){Symmetric3 matrix;Vec3 rhs{};
        for(int e=m.cellFaceOffsets[c];e<m.cellFaceOffsets[c+1];++e){const int f=m.cellFaces[e],other=otherCell(m,c,f);
            if(m.boundaryKind[f]==BoundaryKind::Empty||m.boundaryKind[f]==BoundaryKind::Interface)continue;
            if(other<0)continue;
            const Vec3 dx=displacement(m,c,f,other);addLeastSquares(matrix,rhs,dx,primitive[other].pressure-primitive[c].pressure);
            const Real distance=mag(dx);if(!(distance>0)){error="coincident material cell centres";return false;}
        }
        const Vec3 grad=solveLeastSquares(matrix,rhs);if(!finite(grad)){error="nonfinite material pressure gradient";return false;}pressureGradient[c]=grad;const auto& a=primitive[c];
        for(int e=m.cellFaceOffsets[c];e<m.cellFaceOffsets[c+1];++e){const int f=m.cellFaces[e],other=otherCell(m,c,f);
            if(other<0||m.boundaryKind[f]==BoundaryKind::Empty||m.boundaryKind[f]==BoundaryKind::Interface)continue;
            gradientSensitivity[c]+=darcyGradientSensitivity(matrix,displacement(m,c,f,other));}
        if(!finite(gradientSensitivity[c])){error="nonfinite material gradient sensitivity";return false;}
        if(p.permeability>0&&a.porosity>0){if(!(p.poreViscosity>0)){error="invalid Darcy viscosity";return false;}
            result.poreVelocity[c]=(p.gravity*a.poreDensity-grad)*(p.permeability/(p.poreViscosity*a.porosity));}
    }
    // Both neighboring least-squares stencils are needed by the corrected
    // pressure stiffness; compute this only after every gradient is available.
    for(std::size_t c=0;c<q.size();++c){Real pressureRate=0,sweepRate=0;
        const Real storage=porePressureStorage(q[c],primitive[c].porosity*m.volumes[c],primitive[c].temperature,p);
        for(int e=m.cellFaceOffsets[c];e<m.cellFaceOffsets[c+1];++e){const int f=m.cellFaces[e],other=otherCell(m,c,f);
            if(m.boundaryKind[f]==BoundaryKind::Empty||m.boundaryKind[f]==BoundaryKind::Interface)continue;
            if(!swept.empty())sweepRate+=absValue(swept[f])/interval/m.volumes[c];
            if(other<0)continue;
            const Real area=mag(m.areaVectors[f]);const Vec3 normal=m.areaVectors[f]*(m.cellFaceSigns[e]/area);
            Real inverseLength=0;
            if(!darcyPressureStencilInverseLength<false>(displacement(m,c,f,other),normal,
                gradientSensitivity[c],gradientSensitivity[other],inverseLength)){error="invalid material Darcy stencil geometry";return false;}
            if(storage>0)pressureRate+=darcyPressureRelaxation(storage,p.permeability,p.poreViscosity,
                maxValue(primitive[c].poreDensity,primitive[other].poreDensity),area,1/inverseLength);
        }
        stabilityRate[c]=pressureRate+sweepRate;
    }
    for(std::size_t f=0;f<m.owner.size();++f){const auto kind=m.boundaryKind[f];const int left=m.owner[f];int right=m.neighbour[f];
        if(kind==BoundaryKind::Interface||kind==BoundaryKind::Empty)continue;
        if(kind==BoundaryKind::Periodic){if(static_cast<int>(f)>m.periodicPartner[f])continue;right=m.owner[m.periodicPartner[f]];}
        const Real area=mag(m.areaVectors[f]),meshRate=swept.empty()?0:swept[f]/interval;const Vec3 normal=m.areaVectors[f]/area;
        if(!finite(meshRate)){error="nonfinite material ALE sweep";return false;}
        SolidQ flux;
        if(meshRate!=0){
            SolidQ density;if(!reconstruct(m,q,p,meshRate<=0||right<0?left:right,f,density)){error="material ALE reconstruction failed";return false;}
            flux=materialAleFlux(density,1,meshRate);
        }
        if(right>=0){const Vec3 dx=displacement(m,left,f,right);const Real distance=dot(dx,normal);MaterialPrimitive a=primitive[left],b=primitive[right];
            SolidQ physical;
            Real normalPressureGradient=0;
            if(!darcyNormalPressureGradient<false>(dx,normal,b.pressure-a.pressure,
                pressureGradient[left],pressureGradient[right],normalPressureGradient)){
                error="invalid material Darcy face";return false;}
            Real velocity=0;
            if(p.permeability>0)velocity=dry::darcyVelocity(p.permeability,p.poreViscosity,
                normalPressureGradient,.5*(a.poreDensity+b.poreDensity),dot(p.gravity,normal));
            auto poreLeft=dryAdapter::poreState(q[left],a),poreRight=dryAdapter::poreState(q[right],b);
            dryAdapter::completePorePair(poreLeft,poreRight);
            // Diffusion uses original cell gradients. Reconstruct only the
            // advective donor; otherwise second-order advection would alter J.
            std::array<double,Ns> diffusivity{};
            for(int species=0;species<Ns;++species)diffusivity[species]=p.poreEffectiveDiffusivity[species];
            struct FaceCalorics {
                const PhysicsConfig& p;
                const dry::PoreFaceState<Ns>& a;
                const dry::PoreFaceState<Ns>& b;
                double enthalpy(std::size_t species,double T)const {
                    // Inactive historical tables need not be configured.
                    return (a.density==0&&b.density==0)||(a.massFraction[species]==0&&b.massFraction[species]==0)
                        ? 0 : speciesH(p.species[species],T);
                }
            } calorics{p,poreLeft,poreRight};
            // Native implicit fvScalarMatrix conduction owns the conductive
            // channel; passing k=0 avoids counting the same heat flux twice.
            auto poreFlux=dry::poreFaceFlux(poreLeft,poreRight,distance,velocity,diffusivity,0,calorics);
            // Bound the explicit operator, not just current net mass outflow:
            // at almost uniform composition, q/outflow misses diffusion stiffness.
            // Total pore inventory is the storage for the mass fractions. Sum
            // face losses before combining with Darcy/ALE stiffness above.
            Real leftMass=0,rightMass=0;
            for(int species=0;species<Ns;++species){leftMass+=q[left].pore[species];rightMass+=q[right].pore[species];}
            if(leftMass>0)stabilityRate[left]+=area*poreFlux.diffusionLossLeft/leftMass;
            if(rightMass>0)stabilityRate[right]+=area*poreFlux.diffusionLossRight/rightMass;
            if(p.spatialOrder==2&&velocity!=0){
                SolidQ reconstructed;MaterialPrimitive recovered;
                if(!reconstruct(m,q,p,velocity>=0?left:right,f,reconstructed)||!recoverMaterial<false>(reconstructed,1,p,recovered)){
                    error="material Darcy reconstruction failed";return false;}
                const auto donor=dryAdapter::poreState(reconstructed,recovered);
                const std::array<double,Ns> noDiffusion{};
                const FaceCalorics donorCalorics{p,donor,donor};
                const auto advective=dry::poreFaceFlux(donor,donor,distance,velocity,noDiffusion,0,donorCalorics);
                for(int species=0;species<Ns;++species){
                    const Real oldEnergy=poreFlux.advection[species]==0?0:poreFlux.advection[species]*calorics.enthalpy(species,velocity>=0?a.temperature:b.temperature);
                    poreFlux.energy+=advective.speciesEnergy[species]-oldEnergy;
                    poreFlux.species[species]=advective.advection[species]+poreFlux.diffusion[species];
                }
            }
            for(int species=0;species<Ns;++species)physical.pore[species]=area*poreFlux.species[species];
            physical.energy=area*poreFlux.energy;
            if(!finite(physical.energy)){error="invalid material Darcy face";return false;}
            flux=addSolid(flux,physical);
        }
        result.faceFlux[f]=flux;if(kind==BoundaryKind::Periodic)result.faceFlux[m.periodicPartner[f]]=addSolid(SolidQ{},flux,-1);
    }
    for(std::size_t c=0;c<q.size();++c){for(int e=m.cellFaceOffsets[c];e<m.cellFaceOffsets[c+1];++e){const auto& flux=result.faceFlux[m.cellFaces[e]];const int sign=m.cellFaceSigns[e];
            result.rates[c]=addSolid(result.rates[c],flux,-sign);
            for(int k=0;k<Nc;++k)outgoing[c].condensed[k]+=maxValue(0,sign*flux.condensed[k]);
            for(int s=0;s<Ns;++s)outgoing[c].pore[s]+=maxValue(0,sign*flux.pore[s]);}
        if(stabilityRate[c]>0)result.dtLimit=minValue(result.dtLimit,cfl/stabilityRate[c]);
        Real poreMass=0;for(int s=0;s<Ns;++s)poreMass+=q[c].pore[s];const Real power=poreMass*dot(p.gravity,result.poreVelocity[c]);
        result.rates[c].energy+=power;
        for(int k=0;k<Nc;++k)if(outgoing[c].condensed[k]>0)result.dtLimit=minValue(result.dtLimit,cfl*q[c].condensed[k]/outgoing[c].condensed[k]);
        for(int s=0;s<Ns;++s)if(outgoing[c].pore[s]>0)result.dtLimit=minValue(result.dtLimit,cfl*q[c].pore[s]/outgoing[c].pore[s]);
    }
    if((!finite(result.dtLimit)||result.dtLimit<=0)){error="invalid material transport timestep";return false;}
    output=std::move(result);if(numericalFailure)*numericalFailure=false;error.clear();return true;
}
bool evaluateMaterialTransport(const HostMesh& m,const std::vector<SolidQ>& q,const PhysicsConfig& p,
    const std::vector<Real>& swept,Real interval,Real cfl,MaterialTransportResult& output,std::string& error,bool* numericalFailure){
    return evaluateMaterialTransportImpl(m,q,p,swept,interval,cfl,output,error,numericalFailure);
}

bool advanceMaterialReactionsImpl(const SolidQ& base,Real volume,const PhysicsConfig& p,Real interval,
    Real maxStep,Real cfl,int maximum,SolidQ& out,std::uint64_t& accepted,std::string& error,bool* numericalFailure){
    if(numericalFailure)*numericalFailure=true;
    if((!(interval>0)||!finite(interval)||maxStep<0||!finite(maxStep)||!(finite(cfl)&&cfl>0)||maximum<=0)){error="invalid reaction local control";return false;}
    SolidQ state=base;MaterialPrimitive material;if(!recoverMaterial<false>(state,volume,p,material,nullptr,-1,numericalFailure)){error="reaction base EOS invalid";return false;}
    if(!p.enableReactions){out=base;accepted=0;error.clear();return true;}
    // Run the existing material controller before reading mechanism entries;
    // its historical count/domain diagnostics must precede adapter allocation.
    ReactionStepControl initialControl;
    if(!reactionStepControl(state,volume,p,cfl,initialControl,numericalFailure,dryAdapter::CoreReactionRates{})){
        error="reaction local EOS/admissibility failure";return false;}
    const auto network=dryAdapter::mechanism(p);
    Real elapsed=0;std::uint64_t count=0;
    while(elapsed<interval){
        if(count==static_cast<std::uint64_t>(maximum))reportPhysicsWarning(p,ugkp::diagnostics::Code::AccuracyResidual,30900,count,maximum,count);
        if(count==std::numeric_limits<std::uint64_t>::max()){error="reaction local substep counter overflow";return false;}
        const Real maximumStep=minValue(interval-elapsed,maxStep>0?maxStep:interval);
        SolidQ trial;Real dt=0;bool active=false;
        if(!dryAdapter::reactionStep(state,volume,p,network,maximumStep,cfl,trial,dt,active,numericalFailure)){
            error="reaction local EOS/admissibility failure";return false;}
        if(!active){out=state;accepted=count;error.clear();return true;}
        if(!finite(dt)||!(dt>0)||!finite(elapsed+dt)||elapsed+dt==elapsed){
            error="reaction local timestep cannot advance time";return false;
        }
        state=trial;elapsed+=dt;++count;
        if(interval-elapsed<=8*std::numeric_limits<Real>::epsilon()*interval)elapsed=interval;
    }
    out=state;accepted=count;error.clear();return true;
}
bool advanceMaterialReactions(const SolidQ& base,Real volume,const PhysicsConfig& p,Real interval,
    Real maxStep,Real cfl,int maximum,SolidQ& out,std::uint64_t& accepted,std::string& error,bool* numericalFailure){
    return advanceMaterialReactionsImpl(base,volume,p,interval,maxStep,cfl,maximum,out,accepted,error,numericalFailure);
}

} // namespace chmt
