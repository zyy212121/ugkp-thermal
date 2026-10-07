#include "materials/MaterialTransport.H"
#include "materials/ReactionStepControl.H"
#include "materials/DarcyGradient.H"
#include "materials/MaterialReconstruction.H"
#include "gas/AleFlux.H"
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
    if(m.volumes.size()!=cells||m.cellCentres.size()!=cells||m.neighbour.size()!=nf||m.areaVectors.size()!=nf
        ||m.faceCentres.size()!=nf||m.boundaryKind.size()!=nf||m.periodicPartner.size()!=nf
        ||m.cellFaceOffsets.size()!=cells+1||m.cellFaces.size()!=m.cellFaceSigns.size()
        ||m.cellFaceOffsets.empty()||m.cellFaceOffsets.front()!=0
        ||m.cellFaceOffsets.back()!=static_cast<int>(m.cellFaces.size())){error="material transport mesh/addressing size mismatch";return false;}
    for(std::size_t c=0;c<cells;++c)if(!(m.volumes[c]>0)||!finite(m.volumes[c])||!finite(m.cellCentres[c])
        ||m.cellFaceOffsets[c]>m.cellFaceOffsets[c+1]){error="invalid material cell geometry";return false;}
    for(std::size_t f=0;f<nf;++f){const int o=m.owner[f],n=m.neighbour[f];
        if(o<0||o>=static_cast<int>(cells)||n< -1||n>=static_cast<int>(cells)||!finite(m.areaVectors[f])
            ||!(mag(m.areaVectors[f])>0)||!finite(m.faceCentres[f])){error="invalid material face geometry";return false;}
        if(m.boundaryKind[f]==BoundaryKind::Periodic){const int pair=m.periodicPartner[f];
            if(pair<0||pair>=static_cast<int>(nf)||m.periodicPartner[pair]!=static_cast<int>(f)
                ||m.boundaryKind[pair]!=BoundaryKind::Periodic){error="invalid material cyclic pairing";return false;}}
    }
    for(std::size_t c=0;c<cells;++c)for(int e=m.cellFaceOffsets[c];e<m.cellFaceOffsets[c+1];++e){const int f=m.cellFaces[e];
        if(f<0||f>=static_cast<int>(nf)||(m.cellFaceSigns[e]!=1&&m.cellFaceSigns[e]!=-1)
            ||(m.cellFaceSigns[e]==1?m.owner[f]!=static_cast<int>(c):m.neighbour[f]!=static_cast<int>(c))){error="invalid material cell-face incidence";return false;}}
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
        const Real change=dot(solveLeastSquares(matrix,rhs),offset);setComponent(increment,k,change);
        if(p.reconstruction==ReconstructionMode::LimitedLinear)bound=minValue(bound,materialReconstructionLimiter(value,lo,hi,change));
    }
    for(int trial=0;trial<48;++trial){SolidQ candidate=addSolid(density,increment,bound);candidate.porosity=q[cell].porosity;MaterialPrimitive state;
        if(recoverMaterial(candidate,1,p,state)){candidate.porosity=state.porosity;out=candidate;return true;}bound*=.5;}
    return false;
}
} // namespace
bool filmThicknessFromMass(Real mass,Real area,Real density,Real& thickness){
    if(!finite(mass)||mass<0||!finite(area)||area<=0||!finite(density)||density<=0)return false;
    const Real value=mass/(density*area);if(!finite(value))return false;thickness=value;return true;
}
bool filmSweepsCompatible(Real oldMass,Real newMass,Real density,Real gasOutwardSweep,Real solidOutwardSweep,Real sideSweep,const Tolerances& tol){
    if(!finite(oldMass)||oldMass<0||!finite(newMass)||newMass<0||!finite(density)||density<=0)return false;
    const Real expected=(newMass-oldMass)/density,actual=-gasOutwardSweep-solidOutwardSweep+sideSweep;
    const Real roundoff=64*std::numeric_limits<Real>::epsilon()*maxValue(oldMass/density,absValue(expected));
    return finite(actual)&&absValue(actual-expected)<=tol.absoluteGeometry+tol.relativeGeometry*maxValue(absValue(actual),absValue(expected))+roundoff;
}
bool applyMaterialPacketBatch(const std::vector<SolidQ>& base,const std::vector<ExchangePacket>& packets,std::vector<SolidQ>& output,Real& numericalMassRoundoff,std::string& error){
    std::vector<SolidQ> next=base;Real correction=0;
    std::vector<std::array<long double,Nc+Ns+1>> sum(base.size()),scale(base.size());std::vector<std::size_t> operations(base.size());
    for(std::size_t cell=0;cell<base.size();++cell){for(int c=0;c<Nc;++c){sum[cell][c]=base[cell].condensed[c];scale[cell][c]=absValue(base[cell].condensed[c]);}
        for(int species=0;species<Ns;++species){sum[cell][Nc+species]=base[cell].pore[species];scale[cell][Nc+species]=absValue(base[cell].pore[species]);}sum[cell][Nc+Ns]=base[cell].energy;}
    // Scatter each actual packet once. No full-domain scan per gas record.
    for(const auto& packet:packets)if(requiredConsumers(packet.kind)&ConsumeSolid){
        if(packet.solidCell<0||static_cast<std::size_t>(packet.solidCell)>=base.size()){error="material packet owner invalid";return false;}
        const std::size_t cell=packet.solidCell;const auto delta=packetDelta(packet);++operations[cell];sum[cell][Nc+Ns]+=delta.solid.energy;
        for(int c=0;c<Nc;++c){sum[cell][c]+=delta.solid.condensed[c];scale[cell][c]+=absValue(delta.solid.condensed[c]);}
        for(int species=0;species<Ns;++species){sum[cell][Nc+species]+=delta.solid.pore[species];scale[cell][Nc+species]+=absValue(delta.solid.pore[species]);}}
    for(std::size_t cell=0;cell<base.size();++cell){for(int component=0;component<Nc+Ns;++component){Real value=static_cast<Real>(sum[cell][component]);
            if(value<0){const long double bound=(operations[cell]+2)*std::numeric_limits<Real>::epsilon()*scale[cell][component];
                if(-sum[cell][component]>bound){error="material packet batch exceeds physical donor inventory";return false;}
                correction-=value;value=0; // Demonstrated arithmetic deficit, explicitly reported.
            }
            if(!finite(value)){error="nonfinite material packet result";return false;}
            if(component<Nc)next[cell].condensed[component]=value;else next[cell].pore[component-Nc]=value;}
        next[cell].energy=static_cast<Real>(sum[cell][Nc+Ns]);if(!finite(next[cell].energy)){error="nonfinite material packet energy";return false;}}
    output=std::move(next);numericalMassRoundoff=correction;error.clear();return true;
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
bool evaluateMaterialTransport(const HostMesh& m,const std::vector<SolidQ>& q,const PhysicsConfig& p,
    const std::vector<Real>& swept,Real interval,Real cfl,MaterialTransportResult& output,std::string& error){
    if(!(interval>0)||!finite(interval)||!(cfl>0&&cfl<=1)||(!swept.empty()&&swept.size()!=m.owner.size())){error="invalid material transport interval/control";return false;}
    if(q.empty()){output=MaterialTransportResult{};error.clear();return true;}
    if(!validMesh(m,q.size(),error))return false;
    MaterialTransportResult result;result.rates.resize(q.size());result.poreVelocity.resize(q.size());result.faceFlux.resize(m.owner.size());
    std::vector<MaterialPrimitive> primitive(q.size());std::vector<SolidQ> outgoing(q.size());
    std::vector<Vec3> pressureGradient(q.size());std::vector<Real> gradientSensitivity(q.size());
    for(std::size_t c=0;c<q.size();++c)if(!recoverMaterial(q[c],m.volumes[c],p,primitive[c])){error="material transport EOS recovery failed in cell "+std::to_string(c);return false;}
    for(std::size_t c=0;c<q.size();++c){Symmetric3 matrix;Vec3 rhs{};
        for(int e=m.cellFaceOffsets[c];e<m.cellFaceOffsets[c+1];++e){const int f=m.cellFaces[e],other=otherCell(m,c,f);
            if(m.boundaryKind[f]==BoundaryKind::Empty||m.boundaryKind[f]==BoundaryKind::Interface)continue;
            if(other<0)continue;
            const Vec3 dx=displacement(m,c,f,other);addLeastSquares(matrix,rhs,dx,primitive[other].pressure-primitive[c].pressure);
            const Real distance=mag(dx);if(!(distance>0)){error="coincident material cell centres";return false;}
        }
        const Vec3 grad=solveLeastSquares(matrix,rhs);pressureGradient[c]=grad;const auto& a=primitive[c];
        for(int e=m.cellFaceOffsets[c];e<m.cellFaceOffsets[c+1];++e){const int f=m.cellFaces[e],other=otherCell(m,c,f);
            if(other<0||m.boundaryKind[f]==BoundaryKind::Empty||m.boundaryKind[f]==BoundaryKind::Interface)continue;
            gradientSensitivity[c]+=darcyGradientSensitivity(matrix,displacement(m,c,f,other));}
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
            if(!darcyPressureStencilInverseLength(displacement(m,c,f,other),normal,
                gradientSensitivity[c],gradientSensitivity[other],inverseLength)){error="invalid material Darcy stencil geometry";return false;}
            if(storage>0)pressureRate+=darcyPressureRelaxation(storage,p.permeability,p.poreViscosity,
                maxValue(primitive[c].poreDensity,primitive[other].poreDensity),area,1/inverseLength);
        }
        if(pressureRate+sweepRate>0)result.dtLimit=minValue(result.dtLimit,cfl/(pressureRate+sweepRate));
    }
    for(std::size_t f=0;f<m.owner.size();++f){const auto kind=m.boundaryKind[f];const int left=m.owner[f];int right=m.neighbour[f];
        if(kind==BoundaryKind::Interface||kind==BoundaryKind::Empty)continue;
        if(kind==BoundaryKind::Periodic){if(static_cast<int>(f)>m.periodicPartner[f])continue;right=m.owner[m.periodicPartner[f]];}
        const Real area=mag(m.areaVectors[f]),meshRate=swept.empty()?0:swept[f]/interval;const Vec3 normal=m.areaVectors[f]/area;
        if(!finite(meshRate)){error="nonfinite material ALE sweep";return false;}
        SolidQ density;if(!reconstruct(m,q,p,meshRate<=0||right<0?left:right,f,density)){error="material ALE reconstruction failed";return false;}
        SolidQ flux=materialAleFlux(density,1,meshRate);
        if(right>=0){const Vec3 dx=displacement(m,left,f,right);const Real distance=dot(dx,normal);MaterialPrimitive a=primitive[left],b=primitive[right];
            a.conductivity=b.conductivity=0;SolidQ physical;
            Real normalPressureGradient=0;
            if(!darcyNormalPressureGradient(dx,normal,b.pressure-a.pressure,
                pressureGradient[left],pressureGradient[right],normalPressureGradient)
                ||!darcyConductionFluxWithGradient(q[left],q[right],a,b,m.volumes[left],m.volumes[right],area,distance,normal,normalPressureGradient,p,physical)){error="invalid material Darcy face";return false;}
            if(p.spatialOrder==2&&p.permeability>0){const Real volumeRate=-area*p.permeability/p.poreViscosity*(normalPressureGradient-.5*(a.poreDensity+b.poreDensity)*dot(p.gravity,normal));
                SolidQ reconstructed;MaterialPrimitive recovered;
                if(!reconstruct(m,q,p,volumeRate>=0?left:right,f,reconstructed)||!recoverMaterial(reconstructed,1,p,recovered)){error="material Darcy reconstruction failed";return false;}
                physical.energy=0;for(int s=0;s<Ns;++s){physical.pore[s]=recovered.porosity>0?volumeRate*reconstructed.pore[s]/recovered.porosity:0;physical.energy+=physical.pore[s]*speciesH(p.species[s],recovered.temperature);}}
            flux=addSolid(flux,physical);
        }
        result.faceFlux[f]=flux;if(kind==BoundaryKind::Periodic)result.faceFlux[m.periodicPartner[f]]=addSolid(SolidQ{},flux,-1);
    }
    for(std::size_t c=0;c<q.size();++c){for(int e=m.cellFaceOffsets[c];e<m.cellFaceOffsets[c+1];++e){const auto& flux=result.faceFlux[m.cellFaces[e]];const int sign=m.cellFaceSigns[e];
            result.rates[c]=addSolid(result.rates[c],flux,-sign);
            for(int k=0;k<Nc;++k)outgoing[c].condensed[k]+=maxValue(0,sign*flux.condensed[k]);
            for(int s=0;s<Ns;++s)outgoing[c].pore[s]+=maxValue(0,sign*flux.pore[s]);}
        Real poreMass=0;for(int s=0;s<Ns;++s)poreMass+=q[c].pore[s];const Real power=poreMass*dot(p.gravity,result.poreVelocity[c]);
        result.rates[c].energy+=power;result.bodyPower+=power;
        for(int k=0;k<Nc;++k)if(outgoing[c].condensed[k]>0)result.dtLimit=minValue(result.dtLimit,cfl*q[c].condensed[k]/outgoing[c].condensed[k]);
        for(int s=0;s<Ns;++s)if(outgoing[c].pore[s]>0)result.dtLimit=minValue(result.dtLimit,cfl*q[c].pore[s]/outgoing[c].pore[s]);
    }
    if(!finite(result.dtLimit)||result.dtLimit<=0||!finite(result.bodyPower)){error="invalid material transport timestep/body work";return false;}
    output=std::move(result);error.clear();return true;
}
bool advanceMaterialReactions(const SolidQ& base,Real volume,const PhysicsConfig& p,Real interval,
    Real maxStep,Real cfl,int maximum,SolidQ& out,std::uint64_t& accepted,std::string& error){
    if(!(interval>0)||!finite(interval)||maxStep<0||!finite(maxStep)||!(cfl>0&&cfl<=1)||maximum<=0){error="invalid reaction local control";return false;}
    SolidQ state=base;MaterialPrimitive material;if(!recoverMaterial(state,volume,p,material)){error="reaction base EOS invalid";return false;}
    if(!p.enableReactions){out=base;accepted=0;error.clear();return true;}
    Real elapsed=0;std::uint64_t count=0;
    while(elapsed<interval){if(count>=static_cast<std::uint64_t>(maximum)){error="reaction local substep limit reached";return false;}
        const Real maximumStep=minValue(interval-elapsed,maxStep>0?maxStep:interval);
        SolidQ trial;Real dt=0;bool active=false;
        if(!reactionStepCandidate(state,volume,p,maximumStep,cfl,trial,dt,active)){
            error="reaction local EOS/accuracy/admissibility rejection";return false;}
        if(!active){out=state;accepted=count;error.clear();return true;}
        if(elapsed+dt==elapsed){error="reaction local timestep cannot advance time";return false;}
        state=trial;elapsed+=dt;++count;
        if(interval-elapsed<=8*std::numeric_limits<Real>::epsilon()*interval)elapsed=interval;
    }
    out=state;accepted=count;error.clear();return true;
}
} // namespace chmt
