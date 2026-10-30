#ifndef CHMT_CONFIGURATION_MODELIO_H
#define CHMT_CONFIGURATION_MODELIO_H
#include "configuration/InputValues.H"
#include "core/HostState.H"
#include "materials/Thermodynamics.H"
#include "configuration/SharedGasModel.H"
#include "CHMTBuildIdentity.H"
#include <algorithm>
#include <stdexcept>
#include <sstream>
namespace chmt {
    namespace io {
        inline CondensedThermo condensed(const Foam::dictionary& d, int ne) {
            CondensedThermo t;
            t.rho = readNumber(d, "rho");
            t.cp0 = readNumber(d, "cp0");
            t.cp1 = readNumber(d, "cp1");
            t.e0 = readNumber(d, "e0");
            t.conductivity = readNumber(d, "conductivity");
            t.Tmin = readNumber(d, "Tmin");
            t.Tmax = readNumber(d, "Tmax");
            elements(d, t.element, ne);

            return t;
        }
        inline Reaction reaction(const Foam::dictionary& d) {
            Reaction r;
            r.A = readNumber(d, "A");
            r.temperaturePower = readNumber(d, "temperaturePower");
            r.activationEnergy = readNumber(d, "activationEnergy");
            array(d, "condensedNu", r.condensedNu);
            array(d, "gasNu", r.gasNu);
            array(d, "order", r.order);
            return r;
        }
        inline ModelConfig readModel(const Foam::dictionary& d,const ugkwp::GasModelConfiguration& gas) {
            ModelConfig m;
            auto& p = m.physics;
            m.modelName = text(d, "modelName");
            m.materialSource = text(d, "materialSource");
            m.mechanismSource = text(d, "mechanismSource");
            m.sourceFingerprint = CHMT_SOURCE_FINGERPRINT;
            m.baseCommit = CHMT_UPSTREAM_BASE;
            std::string error;
            const Foam::wordList names(d.lookup("condensedNames"));

            for (int c = 0; c<Nc; ++c)m.condensedNames[c] = names[c];
            const Foam::wordList elementNames(d.lookup("elements"));

            p.nElements = elementNames.size();
            for (int e = 0; e<Ne&&e<p.nElements; ++e)m.elementNames[e] = elementNames[e];
            p.enableGas = flag(d, "enableGas", true);
            // Dry CPU material model: legacy film ABI fields remain inactive.
            p.enableReactions = flag(d, "enableReactions");
            p.enableParticles = flag(d, "enableParticles");
            p.enableSst = flag(d, "enableSst");
            p.enableRadiation = flag(d, "enableRadiation");
            const auto recon = text(d, "materialReconstruction", "LimitedLinear");

            p.reconstruction = recon == "SmoothVerification"?ReconstructionMode::SmoothVerification:ReconstructionMode::LimitedLinear;
            p.minDt = readNumber(d, "minDt");
            p.maxDt = readNumber(d, "maxDt");
            p.cfl = readNumber(d, "cfl");
            p.spatialOrder = d.lookupOrDefault<Foam::label>("spatialOrder", 2);
            p.gravity = readVector(d, "gravity");
            p.gasViscosity = readNumber(d, "gasViscosity", 0);
            p.gasConductivity = readNumber(d, "gasConductivity", 0);

            p.permeability = readNumber(d, "permeability", 0);
            p.poreViscosity = readNumber(d, "poreViscosity", 0);
            array(d, "poreEffectiveDiffusivity", p.poreEffectiveDiffusivity, true);
            p.emissivity = readNumber(d, "emissivity", 0);
            p.ambientTemperature = readNumber(d, "ambientTemperature", 0);
            if (d.found("condensedThermo"))for (int c = 0; c<Nc; ++c)p.condensed[c] = condensed(d.subDict("condensedThermo").subDict(m.condensedNames[c]), p.nElements);
            if (d.found("reactions")) {
                const auto& r = d.subDict("reactions");
                const Foam::wordList keys = r.toc();

                p.nReactions = keys.size();
                for (int i = 0; i<p.nReactions; ++i)p.reactions[i] = reaction(r.subDict(keys[i]));
            }
            if (d.found("material")) {
                const auto& a = d.subDict("material");
                auto& t = p.material;
                t.minimumPorosity = readNumber(a, "minimumPorosity", 0);
                t.maximumPorosity = readNumber(a, "maximumPorosity", 1);
                t.phaseCondensed = a.lookupOrDefault<Foam::label>("phaseCondensed", -1);
                t.enableSurfaceReactions = flag(a, "enableSurfaceReactions");
                t.enablePoreOutflow = flag(a, "enablePoreOutflow");
                t.gasContactResistance = readNumber(a, "gasContactResistance", 0);
                t.solidContactResistance = readNumber(a, "solidContactResistance", 0);
                if (a.found("surfaceReactions")) {
                    const auto& r = a.subDict("surfaceReactions");
                    const Foam::wordList keys = r.toc();

                    t.nSurfaceReactions = keys.size();
                    for (int i = 0; i<t.nSurfaceReactions; ++i) {
                        const auto& v = r.subDict(keys[i]);
                        t.surfaceReactions[i] = reaction(v);
                        array(v, "gasOrder", t.surfaceGasOrder[i]);
                    }
                }
            }
            if (d.found("tolerances")) {
                const auto& a = d.subDict("tolerances");
                auto& t = p.tolerances;
#define CHMT_READ_TOL(name) t.name=readNumber(a,#name,t.name)
                CHMT_READ_TOL(absoluteMass);
                CHMT_READ_TOL(relativeMass);
                CHMT_READ_TOL(absoluteEnergy);
                CHMT_READ_TOL(relativeEnergy);
                CHMT_READ_TOL(absoluteTemperature);
                CHMT_READ_TOL(relativeTemperature);
                CHMT_READ_TOL(absoluteGeometry);
                CHMT_READ_TOL(relativeGeometry);
#undef CHMT_READ_TOL
                t.maxCouplingIterations = a.lookupOrDefault<Foam::label>("maxCouplingIterations",
                    t.maxCouplingIterations);
                t.maxRetries = a.lookupOrDefault<Foam::label>("maxRetries", t.maxRetries);
            }
            if (d.found("sst")) {
                const auto& a = d.subDict("sst");
                const auto family=text(a,"wallTreatment","lowRe");
                p.wallModel.family=family=="boundaryLayer"?ugkwp::gaswall::WallFamily::BoundaryLayer:ugkwp::gaswall::WallFamily::LowRe;
                if(a.found("boundaryLayer")){

                    const auto& layer=a.subDict("boundaryLayer");auto& wall=p.wallModel;
                    const auto model=text(layer,"model","reactingSst");

                    wall.model=model=="constantTransport"?ugkwp::gaswall::BoundaryLayerModel::ConstantTransport:ugkwp::gaswall::BoundaryLayerModel::ReactingSst;
                    p.wallWorkspaceSlots=layer.lookupOrDefault<Foam::label>("workspaceSlots",p.wallWorkspaceSlots);
                    wall.nodes=layer.lookupOrDefault<Foam::label>("nodes",wall.nodes);
                    wall.maxIterations=layer.lookupOrDefault<Foam::label>("maxIterations",wall.maxIterations);
                    wall.relativeTolerance=readNumber(layer,"relativeTolerance",wall.relativeTolerance);
                    wall.absoluteTolerance=readNumber(layer,"absoluteTolerance",wall.absoluteTolerance);
                    wall.stretch=readNumber(layer,"stretch",wall.stretch);
                }
                auto& t = p.sst;
#define CHMT_READ_SST(name) t.name=readNumber(a,#name,t.name)
                CHMT_READ_SST(a1);
                CHMT_READ_SST(betaStar);
                CHMT_READ_SST(beta1);
                CHMT_READ_SST(beta2);
                CHMT_READ_SST(gamma1);
                CHMT_READ_SST(gamma2);
                CHMT_READ_SST(sigmaK1);
                CHMT_READ_SST(sigmaK2);
                CHMT_READ_SST(sigmaOmega1);
                CHMT_READ_SST(sigmaOmega2);
                CHMT_READ_SST(turbulentPrandtl);
                CHMT_READ_SST(turbulentSchmidt);
                CHMT_READ_SST(productionLimit);
                CHMT_READ_SST(minimumK);
                CHMT_READ_SST(minimumOmega);
#undef CHMT_READ_SST
            }
            if (d.found("particle")) {
                const auto& a=d.subDict("particle");
                auto& t=m.particlePhase;
                t.capacity=a.lookupOrDefault<Foam::label>("capacity",t.capacity);
                t.maxFaceWalkHops=a.lookupOrDefault<Foam::label>("maxFaceCrossings",t.maxFaceWalkHops);
                t.randomSeed=std::stoull(text(a,"randomSeed","1"));
#define CHMT_PARTICLE_NUMBER(name) t.name=readNumber(a,#name,t.name)
                CHMT_PARTICLE_NUMBER(injectionParcelMass);
                CHMT_PARTICLE_NUMBER(rhoSolid);CHMT_PARTICLE_NUMBER(diameter);
                CHMT_PARTICLE_NUMBER(minimumDiameter);CHMT_PARTICLE_NUMBER(maximumDiameter);
                CHMT_PARTICLE_NUMBER(minimumTemperature);CHMT_PARTICLE_NUMBER(maximumTemperature);
                CHMT_PARTICLE_NUMBER(dragResidualRe);CHMT_PARTICLE_NUMBER(minimumSolidFraction);
                CHMT_PARTICLE_NUMBER(minimumTheta);CHMT_PARTICLE_NUMBER(collisionalRestitution);
                CHMT_PARTICLE_NUMBER(pressureKickFraction);CHMT_PARTICLE_NUMBER(packingFraction);
#undef CHMT_PARTICLE_NUMBER
                t.solveTemperature=flag(a,"solveTemperature",t.solveTemperature);
                t.collisionalPressure=flag(a,"collisionalPressure",t.collisionalPressure);
                t.packingPressure=flag(a,"packingPressure",t.packingPressure);
                t.packingIterations=a.lookupOrDefault<Foam::label>("packingIterations",t.packingIterations);
                t.csrCellLocalPath=flag(a,"csrCellLocalPath",t.csrCellLocalPath);
                t.warpAggregatedBinning=flag(a,"warpAggregatedBinning",t.warpAggregatedBinning);
                t.heavyReductionMode=a.lookupOrDefault<Foam::label>("heavyReductionMode",t.heavyReductionMode);
                t.heavyAutoInterval=a.lookupOrDefault<Foam::label>("heavyAutoInterval",t.heavyAutoInterval);
                t.particleBlockThreads=a.lookupOrDefault<Foam::label>("particleBlockThreads",t.particleBlockThreads);
                t.reductionBlockThreads=a.lookupOrDefault<Foam::label>("reductionBlockThreads",t.reductionBlockThreads);
                const auto drag=text(a,"dragModel","SchillerNaumann");
                require(drag=="none"||drag=="SchillerNaumann"||drag=="GidaspowErgunWenYu","unsupported shared particle dragModel");
                t.dragModel=drag=="none"?0:drag=="SchillerNaumann"?1:2;
                const auto heat=text(a,"heatTransferModel","RanzMarshall");
                require(heat=="none"||heat=="RanzMarshall","unsupported shared particle heatTransferModel");
                t.heatTransferModel=heat=="none"?0:1;
                // The former CHMT particle fields never selected an implemented
                // phase. Do not reinterpret a prescribed coefficient as Ranz--Marshall.
                for(const char* key:{"heatTransferCoefficient","contactHeatTransferCoefficient","contactDuration","emissivity"})
                    require(readNumber(a,key,0)==0,std::string("particle.")+key+" is not a shared thermal-particle model input");
                auto& w=m.particleContact;
                const auto contact=text(a,"contactModel","reboundContact");
                require(contact=="reboundContact"||contact=="solidifyingDeposition"||contact=="coldWall1D"||contact=="coldWall2D","unsupported shared particle contactModel");
                w.model=contact=="reboundContact"?1:contact=="coldWall2D"?4:3;
                if(a.found("contactCoeffs")){
                    const auto& c=a.subDict("contactCoeffs");
#define CHMT_CONTACT_NUMBER(name) w.name=readNumber(c,#name,w.name)
                    CHMT_CONTACT_NUMBER(sommerfeldThreshold);CHMT_CONTACT_NUMBER(maximumCoverage);
                    CHMT_CONTACT_NUMBER(depositionHeatTransferEfficiency);CHMT_CONTACT_NUMBER(reflectionHeatTransferEfficiency);
                    CHMT_CONTACT_NUMBER(adhesionEnergyScale);CHMT_CONTACT_NUMBER(contactAngleDegree);
                    CHMT_CONTACT_NUMBER(meltingTemperature);CHMT_CONTACT_NUMBER(mushyRange);CHMT_CONTACT_NUMBER(latentHeat);
                    CHMT_CONTACT_NUMBER(solidDensity);CHMT_CONTACT_NUMBER(solidSpecificHeat);CHMT_CONTACT_NUMBER(solidThermalConductivity);
                    CHMT_CONTACT_NUMBER(pinningThicknessFraction);CHMT_CONTACT_NUMBER(interfaceResistance);
#undef CHMT_CONTACT_NUMBER
                    w.heatTransfer=flag(c,"heatTransfer",w.heatTransfer);
                    w.wallTransientResistance=flag(c,"wallTransientResistance",w.wallTransientResistance);
                    w.nonlinearIterations=c.lookupOrDefault<Foam::label>("nonlinearIterations",w.nonlinearIterations);
                }
            }
            if (d.found("meshMotion")) {
                const auto& a = d.subDict("meshMotion");
                const auto policy = text(a, "policy");
                auto& t = p.meshMotion;

                t.policy = policy == "Static"?MeshMotionPolicy::Static:policy == "CoupledRecession"?MeshMotionPolicy::CoupledRecession:MeshMotionPolicy::PrescribedSinusoidal;
                t.amplitude = readVector(a, "amplitude");
                t.spatialWaveNumber = readVector(a, "spatialWaveNumber");
                t.spatialOrigin = readVector(a, "spatialOrigin");
                t.angularFrequency = readNumber(a, "angularFrequency", 0);
                t.timeOrigin = readNumber(a, "timeOrigin", 0);
            }
            if(gas.mode==ugkwp::GasMode::SingleLegacy){
                auto metadata=gas;metadata.mode=ugkwp::GasMode::MixtureFrozen;
                require(bindSingleGasModel(metadata,text(d,"singleGasSpecies"),m,error),error);
            } else bindSharedGasModel(gas,m,error);
            p.wallModel.enableSst=p.enableSst;p.wallModel.turbulentPrandtl=p.sst.turbulentPrandtl;
            p.wallModel.turbulentSchmidt=p.sst.turbulentSchmidt;
            return m;
        }
    }
}
#endif
