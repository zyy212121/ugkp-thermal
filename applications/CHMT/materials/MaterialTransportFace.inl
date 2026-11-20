CHMT_HD inline MaterialTransportError face(const MaterialMeshView& m,const SolidQ* q,
    const PhysicsConfig& p,const Real* swept,Real interval,int f,const MaterialTransportWorkspace& w) {
    w.faceDiffusionLossLeft[f]=0;w.faceDiffusionLossRight[f]=0;
    auto* split=w.splitFlux?&w.splitFlux[f]:nullptr;
    const auto kind=m.boundaryKind[f];const int left=m.owner[f];int right=m.neighbour[f];
        if(kind==BoundaryKind::Interface||kind==BoundaryKind::Empty){w.faceFlux[f]={};return MaterialTransportError::None;}
        if(kind==BoundaryKind::Periodic){if(f>m.periodicPartner[f])return MaterialTransportError::None;right=m.owner[m.periodicPartner[f]];}
        const auto& areaVector=m.areaVectors[f];
        const PhysicalVec3 physicalArea{PhysicalReal(areaVector.x),PhysicalReal(areaVector.y),PhysicalReal(areaVector.z)};
        const PhysicalReal area=mag(physicalArea);
        const PhysicalReal meshRate=swept?PhysicalReal(swept[f])/PhysicalReal(interval):PhysicalReal(0);
        const PhysicalVec3 normal=physicalArea/area;
        if(!finite(meshRate)){return MaterialTransportError::AleSweep;}
        SolidQ flux;
        if(meshRate!=0){
            SolidQ density;
            MaterialPrimitive donor;
            if(!reconstruct(m,q,p,meshRate<=0||right<0?left:right,f,density,split?&donor:nullptr))
                return MaterialTransportError::AleReconstruction;
            if(split){split->sweptTemperature=donor.temperature;density.energy=0;}
            flux=materialAleFlux(density,1,meshRate);
            if(split)for(int s=0;s<Ns;++s)split->sweptPore[s]=flux.pore[s];
        }
        if(right>=0){const PhysicalVec3 dx=displacement(m,left,f,right);const PhysicalReal distance=dot(dx,normal);MaterialPrimitive a=w.primitive[left],b=w.primitive[right];
            SolidQ physical;
            PhysicalReal normalPressureGradient=0;
            if(!darcyNormalPressureGradient<false>(dx,normal,b.pressure-a.pressure,
                w.pressureGradient[left],w.pressureGradient[right],normalPressureGradient)){
                return MaterialTransportError::DarcyFace;}
            PhysicalReal velocity=0;
            if(p.permeability>0)velocity=dry::darcyVelocity(p.permeability,p.poreViscosity,
                normalPressureGradient,PhysicalReal(.5)*(a.poreDensity+b.poreDensity),dot(p.gravity,normal));
            auto poreLeft=dryAdapter::poreState(q[left],a),poreRight=dryAdapter::poreState(q[right],b);
            dryAdapter::completePorePair(poreLeft,poreRight);


            std::array<PhysicalReal,Ns> diffusivity{};
            for(int species=0;species<Ns;++species)diffusivity[species]=p.poreEffectiveDiffusivity[species];
            struct FaceCalorics {
                const PhysicsConfig& p;
                const dry::PoreFaceState<Ns>& a;
                const dry::PoreFaceState<Ns>& b;
                bool computeEnergy;
                CHMT_HD EnergyReal enthalpy(std::size_t species,EnergyReal T)const {

                    return !computeEnergy||(a.density==0&&b.density==0)||(a.massFraction[species]==0&&b.massFraction[species]==0)
                        ? 0 : speciesH(p.species[species],T);
                }
            } calorics{p,poreLeft,poreRight,!split};


            auto poreFlux=dry::poreFaceFlux(poreLeft,poreRight,distance,velocity,diffusivity,0,calorics);




            PhysicalReal leftMass=0,rightMass=0;
            for(int species=0;species<Ns;++species){leftMass+=q[left].pore[species];rightMass+=q[right].pore[species];}
            if(leftMass>0)w.faceDiffusionLossLeft[f]=area*poreFlux.diffusionLossLeft/leftMass;
            if(rightMass>0)w.faceDiffusionLossRight[f]=area*poreFlux.diffusionLossRight/rightMass;
            EnergyReal donorTemperature=velocity>=0?a.temperature:b.temperature;
            if(p.spatialOrder==2&&velocity!=0){
                SolidQ reconstructed;MaterialPrimitive recovered;
                if(!reconstruct(m,q,p,velocity>=0?left:right,f,reconstructed,&recovered))
                    return MaterialTransportError::DarcyReconstruction;
                donorTemperature=recovered.temperature;
                const auto donor=dryAdapter::poreState(reconstructed,recovered);
                const std::array<PhysicalReal,Ns> noDiffusion{};
                const FaceCalorics donorCalorics{p,donor,donor,!split};
                const auto advective=dry::poreFaceFlux(donor,donor,distance,velocity,noDiffusion,0,donorCalorics);
                for(int species=0;species<Ns;++species){
                    const EnergyReal oldEnergy=poreFlux.advection[species]==0?0:EnergyReal(poreFlux.advection[species])*calorics.enthalpy(species,EnergyReal(velocity>=0?a.temperature:b.temperature));
                    poreFlux.energy+=advective.speciesEnergy[species]-oldEnergy;
                    poreFlux.advection[species]=advective.advection[species];
                    poreFlux.species[species]=advective.advection[species]+poreFlux.diffusion[species];
                }
            }
            for(int species=0;species<Ns;++species)physical.pore[species]=area*poreFlux.species[species];
            physical.energy=EnergyReal(area)*poreFlux.energy;
            if(split){
                split->darcyFromLeft=velocity>=0;
                split->darcyTemperature=donorTemperature;
                for(int s=0;s<Ns;++s){
                    split->darcyPore[s]=area*poreFlux.advection[s];
                    split->diffusivePore[s]=area*poreFlux.diffusion[s];
                }
            }
            if(!finite(physical.energy)){return MaterialTransportError::DarcyFace;}
            flux=addSolid(flux,physical);
        }
        w.faceFlux[f]=flux;if(kind==BoundaryKind::Periodic)w.faceFlux[m.periodicPartner[f]]=addSolid(SolidQ{},flux,-1);
    return MaterialTransportError::None;
}
