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
            require(validCondensed(t), "invalid explicit condensed/liquid thermo table");
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
            require(d.lookup<Foam::label>("schemaVersion") == 1, "unsupported chmtProperties schemaVersion");
            ModelConfig m;
            auto& p = m.physics;
            m.modelName = text(d, "modelName");
            m.materialSource = text(d, "materialSource");
            m.mechanismSource = text(d, "mechanismSource");
            m.sourceFingerprint = CHMT_SOURCE_FINGERPRINT;
            m.baseCommit = CHMT_UPSTREAM_BASE;
            m.modelFingerprint = text(d, "modelFingerprint");
            require(m.modelFingerprint.size() == 64
                && m.modelFingerprint.find_first_not_of("0123456789abcdef") == std::string::npos,
                "modelFingerprint must be a lowercase SHA-256 of frozen configuration");
            p.modelFingerprint = std::stoull(m.modelFingerprint.substr(0, 16), nullptr, 16);
            m.speciesFingerprint = std::to_string(p.speciesFingerprint);
            m.schemaFingerprint = std::to_string(checkpointSchema);
            require(!d.found("species")&&!d.found("speciesThermo")&&!d.found("gasDiffusivity"),
                "species, speciesThermo and diffusion belong in constant/gasModelProperties");
            std::string error;
            require(validateSpeciesSet(m, error), error);
            const Foam::wordList names(d.lookup("condensedNames"));
            require(names.size() == Nc, "condensedNames count mismatch");
            for (int c = 0; c<Nc; ++c)m.condensedNames[c] = names[c];
            const Foam::wordList elementNames(d.lookup("elements"));
            require(elementNames.size() <= Ne, "too many elements");
            p.nElements = elementNames.size();
            for (int e = 0; e<p.nElements; ++e)m.elementNames[e] = elementNames[e];
            p.enableGas = flag(d, "enableGas", true);
            p.enableFilm = flag(d, "enableFilm");
            p.enableReactions = flag(d, "enableReactions");
            p.enableParticles = flag(d, "enableParticles");
            p.enableSst = flag(d, "enableSst");
            p.enableRadiation = flag(d, "enableRadiation");
            const auto thermal = text(d, "filmThermalMode", "ThicknessAveraged");
            require(thermal == "ThicknessAveraged" || thermal == "ResolvedNormal", "unsupported filmThermalMode");
            p.filmThermalMode = thermal == "ResolvedNormal"?FilmThermalMode::ResolvedNormal:FilmThermalMode::ThicknessAveraged;
            require(!d.found("reconstruction"), "gas reconstruction belongs in fvSchemes; materialReconstruction controls only material stencils");
            const auto recon = text(d, "materialReconstruction", "LimitedLinear");
            require(recon == "LimitedLinear" || recon == "SmoothVerification", "unsupported reconstruction");
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
            p.emissivity = readNumber(d, "emissivity", 0);
            p.ambientTemperature = readNumber(d, "ambientTemperature", 0);
            if (d.found("condensedThermo"))for (int c = 0; c<Nc; ++c)p.condensed[c] = condensed(d.subDict("condensedThermo").subDict(m.condensedNames[c]), p.nElements);
            if (d.found("liquid"))p.liquid = condensed(d.subDict("liquid"), p.nElements);
            p.liquidViscosity = readNumber(d, "liquidViscosity", 0);
            p.liquidReferencePressure = readNumber(d, "liquidReferencePressure", 0);
            p.meltTemperature = readNumber(d, "meltTemperature", 0);
            if (d.found("reactions")) {
                const auto& r = d.subDict("reactions");
                const Foam::wordList keys = r.toc();
                require(keys.size() <= Nr, "too many reactions");
                p.nReactions = keys.size();
                for (int i = 0; i<p.nReactions; ++i)p.reactions[i] = reaction(r.subDict(keys[i]));
            }
            if (d.found("material")) {
                const auto& a = d.subDict("material");
                auto& t = p.material;
                t.minimumPorosity = readNumber(a, "minimumPorosity", 0);
                t.maximumPorosity = readNumber(a, "maximumPorosity", 1);
                t.phaseCondensed = a.lookupOrDefault<Foam::label>("phaseCondensed", -1);
                t.enableMelting = flag(a, "enableMelting");
                t.enableSurfaceReactions = flag(a, "enableSurfaceReactions");
                t.enablePoreOutflow = flag(a, "enablePoreOutflow");
                array(a, "phaseFilmY", t.phaseFilmY, true);
                t.gasContactResistance = readNumber(a, "gasContactResistance", 0);
                t.solidContactResistance = readNumber(a, "solidContactResistance", 0);
                const auto law = text(a, "evaporation", "Disabled");
                require(law == "Disabled" || law == "HertzKnudsen", "unsupported evaporation law");
                t.evaporation = law == "HertzKnudsen"?EvaporationLaw::HertzKnudsen:EvaporationLaw::Disabled;
                array(a, "evaporationAccommodation", t.evaporationAccommodation, true);
                array(a, "saturationPressureScale", t.saturationPressureScale, true);
                array(a, "saturationTemperature", t.saturationTemperature, true);
                if (a.found("surfaceReactions")) {
                    const auto& r = a.subDict("surfaceReactions");
                    const Foam::wordList keys = r.toc();
                    require(keys.size() <= Nr, "too many surface reactions");
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
                require(text(a,"wallTreatment","lowRe")=="lowRe","CHMT shared SST supports wallTreatment lowRe only");
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
                const auto& a = d.subDict("particle");
                auto& t = p.particle;
                t.condensedIndex = a.lookup<Foam::label>("condensedIndex");
                t.maxFaceCrossings = a.lookupOrDefault<Foam::label>("maxFaceCrossings", 32);
                t.heatTransferCoefficient = readNumber(a, "heatTransferCoefficient");
                t.contactHeatTransferCoefficient = readNumber(a, "contactHeatTransferCoefficient");
                t.contactDuration = readNumber(a, "contactDuration");
                t.emissivity = readNumber(a, "emissivity", 0);
                const auto mode = text(a, "contact");
                require(mode == "ElasticRebound" || mode == "Deposit", "unsupported particle contact");
                t.contact = mode == "Deposit"?ParticleContactMode::Deposit:ParticleContactMode::ElasticRebound;
            }
            if (d.found("meshMotion")) {
                const auto& a = d.subDict("meshMotion");
                const auto policy = text(a, "policy");
                auto& t = p.meshMotion;
                require(policy == "Static" || policy == "CoupledRecession" || policy == "PrescribedSinusoidal",
                    "unsupported meshMotion policy");
                t.policy = policy == "Static"?MeshMotionPolicy::Static:policy == "CoupledRecession"?MeshMotionPolicy::CoupledRecession:MeshMotionPolicy::PrescribedSinusoidal;
                t.amplitude = readVector(a, "amplitude");
                t.spatialWaveNumber = readVector(a, "spatialWaveNumber");
                t.spatialOrigin = readVector(a, "spatialOrigin");
                t.angularFrequency = readNumber(a, "angularFrequency", 0);
                t.timeOrigin = readNumber(a, "timeOrigin", 0);
            }
            require(p.minDt>0 && p.maxDt >= p.minDt && p.cfl>0 && p.cfl <= 1, "invalid common time controls");
            require(p.enableGas, "CHMT requires coupled gas; pure gas belongs to gasUGKP");
            if(gas.mode==ugkwp::GasMode::SingleLegacy){
                auto metadata=gas;metadata.mode=ugkwp::GasMode::MixtureFrozen;
                require(bindSingleGasModel(metadata,text(d,"singleGasSpecies"),m,error),error);
            } else require(bindSharedGasModel(gas,m,error),error);
            return m;
        }
    }
}
#endif
