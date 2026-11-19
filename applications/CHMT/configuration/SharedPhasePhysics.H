#pragma once
#include "configuration/ModelIO.H"
#include "FoamMomentumTransportInput.H"
namespace chmt { namespace io {
inline void readSharedPhasePhysics(ModelConfig& m,const Foam::dictionary& momentum,
    const Foam::dictionary& particles,bool particlesEnabled) {
    auto& p=m.physics;p.enableParticles=particlesEnabled;
    p.enableSst=text(momentum,"simulationType","laminar")=="RAS"
        && ugkp::solverControls::rasModel(momentum.subOrEmptyDict("RAS"),"kOmegaSST")=="kOmegaSST";
            if (momentum.found("RAS")) {
                const auto ras=momentum.subDict("RAS");
                const auto a=ras.subOrEmptyDict("kOmegaSSTCoeffs");
                p.sst.turbulentPrandtl=readPhysical(ras,"turbulentPrandtl",p.sst.turbulentPrandtl);
                p.sst.turbulentSchmidt=readPhysical(ras,"turbulentSchmidt",p.sst.turbulentSchmidt);
                const auto family=text(a,"wallTreatment","lowRe");
                require(family=="boundaryLayer"||family=="lowRe","unsupported CHMT sst.wallTreatment: "+family);
                p.wallModel.family=family=="boundaryLayer"?ugkwp::gaswall::WallFamily::BoundaryLayer:ugkwp::gaswall::WallFamily::LowRe;
                if(a.found("boundaryLayer")){

                    const auto& layer=a.subDict("boundaryLayer");auto& wall=p.wallModel;
                    const auto model=text(layer,"model","reactingSst");

                    require(model=="constantTransport"||model=="finiteRate"||model=="reactingSst"||model=="equilibriumOde","unknown sst.boundaryLayer.model: "+model);
                    wall.model=model=="constantTransport"?ugkwp::gaswall::BoundaryLayerModel::ConstantTransport:
                        model=="equilibriumOde"?ugkwp::gaswall::BoundaryLayerModel::EquilibriumOde:ugkwp::gaswall::BoundaryLayerModel::ReactingSst;
                    const auto stress=text(layer,"equilibriumStressClosure","localStress");
                    require(stress=="wallStress"||stress=="localStress","unknown equilibriumStressClosure: "+stress);
                    wall.localStressCorrection=stress=="localStress";
                    p.wallWorkspaceSlots=layer.lookupOrDefault<Foam::label>("workspaceSlots",p.wallWorkspaceSlots);
                }
                auto& t = p.sst;
                ugkp::solverControls::readSstCoefficients(a,t);
                t.turbulentPrandtl=readPhysical(a,"turbulentPrandtl",t.turbulentPrandtl);
                t.turbulentSchmidt=readPhysical(a,"turbulentSchmidt",t.turbulentSchmidt);
            }
            if (p.enableParticles) {
                const auto& a=particles;
                auto& t=m.particlePhase;
                t.randomSeed=std::stoull(text(a,"gpuResidentRandomSeed","1"));
#define CHMT_PARTICLE_NUMBER(name) t.name=readPhysical(a,#name,t.name)
                t.injectionParcelMass=readPhysical(a,"parcelMass",t.injectionParcelMass);
                t.rhoSolid=readPhysical(a,"rhoS",t.rhoSolid);t.diameter=readPhysical(a,"dS",t.diameter);
                t.dn=Foam::readParticleDiameterShape(a);
                t.minimumDiameter=readPhysical(a,"dMin",t.minimumDiameter);t.maximumDiameter=readPhysical(a,"dMax",t.maximumDiameter);
                t.minimumTemperature=readPhysical(a,"TpMin",t.minimumTemperature);t.maximumTemperature=readPhysical(a,"TpMax",t.maximumTemperature);
                CHMT_PARTICLE_NUMBER(dragResidualRe);t.minimumSolidFraction=readPhysical(a,"epsSMin",t.minimumSolidFraction);
                t.minimumTheta=readPhysical(a,"thetaMin",t.minimumTheta);CHMT_PARTICLE_NUMBER(collisionalRestitution);
                CHMT_PARTICLE_NUMBER(pressureKickFraction);CHMT_PARTICLE_NUMBER(packingFraction);
#undef CHMT_PARTICLE_NUMBER
                t.solveTemperature=flag(a,"solveParticleTemperature",t.solveTemperature);
                t.collisionalPressure=flag(a,"collisionalPressure",t.collisionalPressure);
                t.packingPressure=flag(a,"packingPressure",t.packingPressure);
                const auto drag=text(a,"dragModel","SchillerNaumann");
                require(drag=="none"||drag=="SchillerNaumann"||drag=="GidaspowErgunWenYu","unsupported shared particle dragModel");
                t.dragModel=drag=="none"?0:drag=="SchillerNaumann"?1:2;
                const auto heat=text(a,"particleGasHeatTransferModel","RanzMarshall");
                require(heat=="none"||heat=="RanzMarshall","unsupported shared particle heatTransferModel");
                t.heatTransferModel=heat=="none"?0:1;
                auto& w=m.particleContact;
                const auto contactProperties=a.subOrEmptyDict("gpuResidentStuckModel");
                const auto contact=text(contactProperties,"wallInteractionModel","coldWall1D");
                require(contact=="reboundContact"||contact=="solidifyingDeposition"||contact=="coldWall1D"||contact=="coldWall2D","unsupported shared particle contactModel");
                w.model=contact=="reboundContact"?1:contact=="coldWall2D"?4:3;
                {
                    const auto& c=contactProperties;
#define CHMT_CONTACT_NUMBER(name) w.name=readPhysical(c,#name,w.name)
                    w.sommerfeldThreshold=Foam::readSommerfeldThreshold(c,w.sommerfeldThreshold);CHMT_CONTACT_NUMBER(maximumCoverage);
                    CHMT_CONTACT_NUMBER(depositionHeatTransferEfficiency);CHMT_CONTACT_NUMBER(reflectionHeatTransferEfficiency);
                    CHMT_CONTACT_NUMBER(adhesionEnergyScale);CHMT_CONTACT_NUMBER(contactAngleDegree);
                    CHMT_CONTACT_NUMBER(meltingTemperature);CHMT_CONTACT_NUMBER(mushyRange);CHMT_CONTACT_NUMBER(latentHeat);
                    CHMT_CONTACT_NUMBER(solidDensity);CHMT_CONTACT_NUMBER(solidSpecificHeat);CHMT_CONTACT_NUMBER(solidThermalConductivity);
                    CHMT_CONTACT_NUMBER(pinningThicknessFraction);w.interfaceResistance=readPhysical(c,"interfaceThermalResistance",w.interfaceResistance);
#undef CHMT_CONTACT_NUMBER
                    w.heatTransfer=flag(c,"heatTransfer",false);
                    w.wallTransientResistance=flag(c,"wallTransientResistance",w.wallTransientResistance);
                }
            }

    p.wallModel.enableSst=p.enableSst;
    p.wallModel.turbulentPrandtl=p.sst.turbulentPrandtl;
    p.wallModel.turbulentSchmidt=p.sst.turbulentSchmidt;
}
}}
