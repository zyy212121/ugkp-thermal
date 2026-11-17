#pragma once
#include "configuration/ModelIO.H"
namespace chmt { namespace io {
inline void readSharedPhasePhysics(ModelConfig& m,const Foam::dictionary& momentum,
    const Foam::dictionary& particles,bool particlesEnabled) {
    auto& p=m.physics;p.enableParticles=particlesEnabled;
    p.enableSst=text(momentum,"simulationType","laminar")=="RAS"
        && flag(momentum.subOrEmptyDict("RAS"),"turbulence",true)
        && text(momentum.subOrEmptyDict("RAS"),"model","kOmegaSST")=="kOmegaSST";
            if (momentum.found("RAS")) {
                const auto ras=momentum.subDict("RAS");
                const auto a=ras.subOrEmptyDict("kOmegaSSTCoeffs");
                p.sst.turbulentPrandtl=readNumber(ras,"turbulentPrandtl",p.sst.turbulentPrandtl);
                p.sst.turbulentSchmidt=readNumber(ras,"turbulentSchmidt",p.sst.turbulentSchmidt);
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
#define CHMT_READ_SST(name) t.name=readNumber(a,#name,t.name)
                CHMT_READ_SST(a1);
                CHMT_READ_SST(betaStar);
                CHMT_READ_SST(beta1);
                CHMT_READ_SST(beta2);
                CHMT_READ_SST(gamma1);
                CHMT_READ_SST(gamma2);
                t.sigmaK1=readNumber(a,"alphaK1",t.sigmaK1);
                t.sigmaK2=readNumber(a,"alphaK2",t.sigmaK2);
                t.sigmaOmega1=readNumber(a,"alphaOmega1",t.sigmaOmega1);
                t.sigmaOmega2=readNumber(a,"alphaOmega2",t.sigmaOmega2);
                CHMT_READ_SST(turbulentPrandtl);
                CHMT_READ_SST(turbulentSchmidt);
                t.productionLimit=readNumber(a,"c1",t.productionLimit);
                t.minimumK=readNumber(a,"kMin",t.minimumK);
                t.minimumOmega=readNumber(a,"omegaMin",t.minimumOmega);
#undef CHMT_READ_SST
            }
            if (p.enableParticles) {
                const auto& a=particles;
                auto& t=m.particlePhase;
                t.randomSeed=std::stoull(text(a,"gpuResidentRandomSeed","1"));
#define CHMT_PARTICLE_NUMBER(name) t.name=readNumber(a,#name,t.name)
                t.injectionParcelMass=readNumber(a,"parcelMass",t.injectionParcelMass);
                t.rhoSolid=readNumber(a,"rhoS",t.rhoSolid);t.diameter=readNumber(a,"dS",t.diameter);
                t.dn=Foam::readParticleDiameterShape(a);
                t.minimumDiameter=readNumber(a,"dMin",t.minimumDiameter);t.maximumDiameter=readNumber(a,"dMax",t.maximumDiameter);
                t.minimumTemperature=readNumber(a,"TpMin",t.minimumTemperature);t.maximumTemperature=readNumber(a,"TpMax",t.maximumTemperature);
                CHMT_PARTICLE_NUMBER(dragResidualRe);t.minimumSolidFraction=readNumber(a,"epsSMin",t.minimumSolidFraction);
                t.minimumTheta=readNumber(a,"thetaMin",t.minimumTheta);CHMT_PARTICLE_NUMBER(collisionalRestitution);
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
#define CHMT_CONTACT_NUMBER(name) w.name=readNumber(c,#name,w.name)
                    w.sommerfeldThreshold=Foam::readSommerfeldThreshold(c,w.sommerfeldThreshold);CHMT_CONTACT_NUMBER(maximumCoverage);
                    CHMT_CONTACT_NUMBER(depositionHeatTransferEfficiency);CHMT_CONTACT_NUMBER(reflectionHeatTransferEfficiency);
                    CHMT_CONTACT_NUMBER(adhesionEnergyScale);CHMT_CONTACT_NUMBER(contactAngleDegree);
                    CHMT_CONTACT_NUMBER(meltingTemperature);CHMT_CONTACT_NUMBER(mushyRange);CHMT_CONTACT_NUMBER(latentHeat);
                    CHMT_CONTACT_NUMBER(solidDensity);CHMT_CONTACT_NUMBER(solidSpecificHeat);CHMT_CONTACT_NUMBER(solidThermalConductivity);
                    CHMT_CONTACT_NUMBER(pinningThicknessFraction);w.interfaceResistance=readNumber(c,"interfaceThermalResistance",w.interfaceResistance);
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
