#include "FoamSommerfeldConfiguration.H"
#include "FoamParticleDiameterDistribution.H"
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
            t.rho = readPhysical(d, "rho");
            t.cp0 = readPhysical(d, "cp0");
            t.cp1 = readPhysical(d, "cp1");
            t.e0 = readPhysical(d, "e0");
            t.conductivity = readPhysical(d, "conductivity");
            t.Tmin = readPhysical(d, "Tmin");
            t.Tmax = readPhysical(d, "Tmax");
            elements(d, t.element, ne);

            return t;
        }
        inline Reaction reaction(const Foam::dictionary& d) {
            Reaction r;
            r.A = readPhysical(d, "A");
            r.temperaturePower = readPhysical(d, "temperaturePower");
            r.activationEnergy = readPhysical(d, "activationEnergy");
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

            require(names.size()==Nc,"condensedNames must match the condensed species capacity");
            for (int c = 0; c<Nc; ++c)m.condensedNames[c] = names[c];
            const Foam::wordList elementNames(d.lookup("elements"));

            require(elementNames.size()<=Ne,"elements exceed configured capacity");
            p.nElements = elementNames.size();
            for (int e = 0; e<Ne&&e<p.nElements; ++e)m.elementNames[e] = elementNames[e];
            p.enableGas = flag(d, "enableGas", true);
            // Dry CPU material model: legacy film ABI fields remain inactive.
            p.enableReactions = flag(d, "enableReactions");
            p.gravity = readPhysicalVector(d, "gravity");

            p.permeability = readPhysical(d, "permeability", 0);
            p.poreViscosity = readPhysical(d, "poreViscosity", 0);
            array(d, "poreEffectiveDiffusivity", p.poreEffectiveDiffusivity, true);
            if (d.found("condensedThermo"))for (int c = 0; c<Nc; ++c)p.condensed[c] = condensed(d.subDict("condensedThermo").subDict(m.condensedNames[c]), p.nElements);
            if (d.found("reactions")) {
                const auto& r = d.subDict("reactions");
                const Foam::wordList keys = r.toc();

                require(keys.size()<=int(sizeof(p.reactions)/sizeof(p.reactions[0])),"reactions exceed configured capacity");
                p.nReactions = keys.size();
                for (int i = 0; i<p.nReactions; ++i)p.reactions[i] = reaction(r.subDict(keys[i]));
            }
            if (d.found("material")) {
                const auto& a = d.subDict("material");
                auto& t = p.material;
                t.minimumPorosity = readPhysical(a, "minimumPorosity", 0);
                t.maximumPorosity = readPhysical(a, "maximumPorosity", 1);
                t.phaseCondensed = a.lookupOrDefault<Foam::label>("phaseCondensed", -1);
                t.enableSurfaceReactions = flag(a, "enableSurfaceReactions");
                t.enablePoreOutflow = flag(a, "enablePoreOutflow");
                t.gasContactResistance = readPhysical(a, "gasContactResistance", 0);
                t.solidContactResistance = readPhysical(a, "solidContactResistance", 0);
                if (a.found("surfaceReactions")) {
                    const auto& r = a.subDict("surfaceReactions");
                    const Foam::wordList keys = r.toc();

                    require(keys.size()<=int(sizeof(t.surfaceReactions)/sizeof(t.surfaceReactions[0])),"surfaceReactions exceed configured capacity");
                    t.nSurfaceReactions = keys.size();
                    for (int i = 0; i<t.nSurfaceReactions; ++i) {
                        const auto& v = r.subDict(keys[i]);
                        t.surfaceReactions[i] = reaction(v);
                        array(v, "gasOrder", t.surfaceGasOrder[i]);
                    }
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
            } else require(bindSharedGasModel(gas,m,error),error);
            p.wallModel.enableSst=p.enableSst;p.wallModel.turbulentPrandtl=p.sst.turbulentPrandtl;
            p.wallModel.turbulentSchmidt=p.sst.turbulentSchmidt;
            return m;
        }
    }
}
#endif
