#include "Checkpoint.H"
#include "coupling/ExchangeLedger.H"
#include "gpu/StateValidation.H"
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>
namespace chmt { namespace {
constexpr std::size_t MaxBytes=512u*1024u*1024u;
using Bytes=std::vector<unsigned char>;
std::uint64_t checksum(const unsigned char* p,std::size_t n){std::uint64_t h=14695981039346656037ULL;for(std::size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}return h;}
struct Encoder;struct Decoder;
template<class Archive> void visit(Archive&, Vec3&);
template<class Archive> void visit(Archive&, GasQ&);
template<class Archive> void visit(Archive&, GasPrimitive&);
template<class Archive> void visit(Archive&, SolidQ&);
template<class Archive> void visit(Archive&, FilmQ&);
template<class Archive> void visit(Archive&, FilmAux&);
template<class Archive> void visit(Archive&, ParticleQ&);
template<class Archive> void visit(Archive&, SstQ&);
template<class Archive> void visit(Archive&, SstPrimitive&);
template<class Archive> void visit(Archive&, SpeciesThermo&);
template<class Archive> void visit(Archive&, CondensedThermo&);
template<class Archive> void visit(Archive&, Reaction&);
template<class Archive> void visit(Archive&, ExchangePacket&);
template<class Archive> void visit(Archive&, FilmPressureVolumeUpdate&);
template<class Archive> void visit(Archive&, Tolerances&);
template<class Archive> void visit(Archive&, Budget&);
template<class Archive> void visit(Archive&, SstConfig&);
template<class Archive> void visit(Archive&, MaterialPolicyConfig&);
template<class Archive> void visit(Archive&, ParticlePolicyConfig&);
template<class Archive> void visit(Archive&, MeshMotionConfig&);
template<class Archive> void visit(Archive&, PhysicsConfig&);
template<class Archive> void visit(Archive&, ModelConfig&);
template<class Archive> void visit(Archive&, HostMesh&);
template<class Archive> void visit(Archive&, SurfaceMesh&);
template<class Archive> void visit(Archive&, HostStageGeometry&);
template<class Archive> void visit(Archive&, HostState&);
struct Encoder {
    Bytes data;
    void integer(std::uint64_t x,int bytes){if(data.size()+bytes>MaxBytes)throw std::runtime_error("checkpoint exceeds size bound");for(int i=0;i<bytes;++i){data.push_back(static_cast<unsigned char>(x&255));x>>=8;}}
    void operator()(Real& x){static_assert(sizeof(Real)==8&&std::numeric_limits<Real>::is_iec559,"checkpoint requires IEEE binary64");std::uint64_t bits;std::memcpy(&bits,&x,8);integer(bits,8);}
    void operator()(int& x){integer(static_cast<std::uint32_t>(x),4);}
    void operator()(unsigned& x){integer(x,4);}
    void operator()(std::uint64_t& x){integer(x,8);}
    void operator()(bool& x){integer(x?1:0,1);}
    void operator()(std::string& x){integer(x.size(),8);if(x.size()>MaxBytes-data.size())throw std::runtime_error("checkpoint string too large");data.insert(data.end(),x.begin(),x.end());}
    template<class T,std::size_t N>void operator()(T(&x)[N]){for(auto& v:x)(*this)(v);}
    template<class T,std::size_t N>void operator()(std::array<T,N>& x){for(auto& v:x)(*this)(v);}
    template<class T>void operator()(std::vector<T>& x){integer(x.size(),8);for(auto& v:x)(*this)(v);}
    template<class T>typename std::enable_if<std::is_enum<T>::value>::type operator()(T& x){int value=static_cast<int>(x);(*this)(value);}
    template<class T>typename std::enable_if<!std::is_enum<T>::value>::type operator()(T& x){visit(*this,x);}
};
struct Decoder {
    const Bytes& data;std::size_t offset=0;
    explicit Decoder(const Bytes& input):data(input){}
    std::uint64_t integer(int bytes){if(std::size_t(bytes)>data.size()-offset)throw std::runtime_error("truncated checkpoint");std::uint64_t x=0;for(int i=0;i<bytes;++i)x|=std::uint64_t(data[offset++])<<(8*i);return x;}
    void operator()(Real& x){const std::uint64_t bits=integer(8);std::memcpy(&x,&bits,8);}
    void operator()(int& x){const auto bits=static_cast<std::uint32_t>(integer(4));std::int32_t value;std::memcpy(&value,&bits,4);x=value;}
    void operator()(unsigned& x){x=static_cast<unsigned>(integer(4));}
    void operator()(std::uint64_t& x){x=integer(8);}
    void operator()(bool& x){auto v=integer(1);if(v>1)throw std::runtime_error("invalid checkpoint boolean");x=v!=0;}
    void operator()(std::string& x){const auto n=integer(8);if(n>data.size()-offset)throw std::runtime_error("invalid checkpoint string count");x.assign(reinterpret_cast<const char*>(data.data()+offset),static_cast<std::size_t>(n));offset+=n;}
    template<class T,std::size_t N>void operator()(T(&x)[N]){for(auto& v:x)(*this)(v);}
    template<class T,std::size_t N>void operator()(std::array<T,N>& x){for(auto& v:x)(*this)(v);}
    template<class T>void operator()(std::vector<T>& x){const auto n=integer(8);if(n>data.size()-offset||n>MaxBytes/sizeof(T))throw std::runtime_error("invalid checkpoint vector count");x.resize(static_cast<std::size_t>(n));for(auto& v:x)(*this)(v);}
    template<class T>typename std::enable_if<std::is_enum<T>::value>::type operator()(T& x){int value=0;(*this)(value);x=static_cast<T>(value);}
    template<class T>typename std::enable_if<!std::is_enum<T>::value>::type operator()(T& x){visit(*this,x);}
};
template<class Archive> void visit(Archive& a,Vec3& x){
    a(x.x);
    a(x.y);
    a(x.z);
}
template<class Archive> void visit(Archive& a,GasQ& x){
    a(x.mass);
    a(x.momentum);
    a(x.energy);
    a(x.species);
}
template<class Archive> void visit(Archive& a,GasPrimitive& x){
    a(x.rho);
    a(x.velocity);
    a(x.temperature);
    a(x.pressure);
    a(x.soundSpeed);
    a(x.Y);
}
template<class Archive> void visit(Archive& a,SolidQ& x){
    a(x.condensed);
    a(x.progress);
    a(x.pore);
    a(x.energy);
    a(x.porosity);
}
template<class Archive> void visit(Archive& a,FilmQ& x){
    a(x.mass);
    a(x.enthalpy);
    a(x.species);
}
template<class Archive> void visit(Archive& a,FilmAux& x){
    a(x.pressure);
    a(x.thickness);
    a(x.temperature);
    a(x.baseVelocity);
    a(x.meanVelocity);
    a(x.topVelocity);
    a(x.normal);
    a(x.area);
    a(x.kineticEnergy);
    a(x.normalVelocity);
    a(x.solidNormalVelocity);
    a(x.solidFront);
    a(x.gasFront);
}
template<class Archive> void visit(Archive& a,ParticleQ& x){
    a(x.contactRelativeVelocity);
    a(x.contactMechanicalEnergy);
    a(x.id);
    a(x.cell);
    a(x.face);
    a(x.state);
    a(x.position);
    a(x.velocity);
    a(x.mass);
    a(x.energy);
    a(x.diameter);
    a(x.contactAge);
    a(x.contactDuration);
    a(x.contactArea);
    a(x.rng);
}
template<class Archive> void visit(Archive& a,SstQ& x){
    a(x.rhoK);
    a(x.rhoOmega);
}
template<class Archive> void visit(Archive& a,SstPrimitive& x){
    a(x.k);
    a(x.omega);
    a(x.eddyViscosity);
    a(x.blend1);
    a(x.blend2);
}
template<class Archive> void visit(Archive& a,SpeciesThermo& x){
    a(x.R);
    a(x.cp0);
    a(x.cp1);
    a(x.e0);
    a(x.Tmin);
    a(x.Tmax);
    a(x.element);
}
template<class Archive> void visit(Archive& a,CondensedThermo& x){
    a(x.rho);
    a(x.cp0);
    a(x.cp1);
    a(x.e0);
    a(x.conductivity);
    a(x.Tmin);
    a(x.Tmax);
    a(x.element);
}
template<class Archive> void visit(Archive& a,Reaction& x){
    a(x.A);
    a(x.temperaturePower);
    a(x.activationEnergy);
    a(x.condensedNu);
    a(x.gasNu);
    a(x.order);
}
template<class Archive> void visit(Archive& a,ExchangePacket& x){
    a(x.step);
    a(x.geometry);
    a(x.face);
    a(x.stage);
    a(x.kind);
    a(x.gasCell);
    a(x.solidCell);
    a(x.filmFace);
    a(x.particleIndex);
    a(x.mass);
    a(x.species);
    a(x.condensed);
    a(x.pore);
    a(x.poreSweep);
    a(x.momentum);
    a(x.energy);
    a(x.conductive);
    a(x.advective);
    a(x.pressureWork);
    a(x.viscousWork);
    a(x.radiation);
    a(x.liquidKineticAdvection);
    a(x.consumerMask);
}
template<class Archive> void visit(Archive& a,FilmPressureVolumeUpdate& x){
    a(x.step);
    a(x.geometry);
    a(x.face);
    a(x.stage);
    a(x.filmFace);
    a(x.oldPV);
    a(x.newPV);
    a(x.consumed);
}
template<class Archive> void visit(Archive& a,Tolerances& x){
    a(x.absoluteMass);
    a(x.relativeMass);
    a(x.absoluteEnergy);
    a(x.relativeEnergy);
    a(x.absoluteTemperature);
    a(x.relativeTemperature);
    a(x.absoluteGeometry);
    a(x.relativeGeometry);
    a(x.maxCouplingIterations);
    a(x.maxRetries);
}
template<class Archive> void visit(Archive& a,Budget& x){
    a(x.boundaryMass);
    a(x.boundarySpecies);
    a(x.boundaryElements);
    a(x.boundaryEnergy);
    a(x.boundaryMomentum);
    a(x.supportImpulse);
    a(x.supportWork);
    a(x.bodyWork);
    a(x.radiation);
    a(x.gclResidual);
    a(x.exchangeMass);
    a(x.exchangeSpecies);
    a(x.exchangeEnergy);
    a(x.exchangeMomentum);
    a(x.filmPressureVolume);
    a(x.filmKineticAdvection);
    a(x.filmKineticStorage);
    a(x.filmReducedResidual);
    a(x.filmKineticDefect);
    a(x.thinLayerGeometryDefect);
    a(x.thinLayerStressDefect);
    a(x.numericalEnergyResidual);
    a(x.turbulenceOmegaConstraint);
    a(x.turbulenceInventory);
    a(x.turbulenceProduction);
    a(x.turbulenceDissipation);
    a(x.turbulenceBoundaryFlux);
    a(x.consumedPackets);
}
template<class Archive> void visit(Archive& a,SstConfig& x){
    a(x.a1);
    a(x.betaStar);
    a(x.beta1);
    a(x.beta2);
    a(x.gamma1);
    a(x.gamma2);
    a(x.sigmaK1);
    a(x.sigmaK2);
    a(x.sigmaOmega1);
    a(x.sigmaOmega2);
    a(x.turbulentPrandtl);
    a(x.turbulentSchmidt);
    a(x.productionLimit);
    a(x.minimumK);
    a(x.minimumOmega);
}
template<class Archive> void visit(Archive& a,MaterialPolicyConfig& x){
    a(x.minimumPorosity);
    a(x.maximumPorosity);
    a(x.phaseCondensed);
    a(x.enableMelting);
    a(x.enableSurfaceReactions);
    a(x.enablePoreOutflow);
    a(x.phaseFilmY);
    a(x.nSurfaceReactions);
    a(x.surfaceReactions);
    a(x.surfaceGasOrder);
    a(x.evaporation);
    a(x.evaporationAccommodation);
    a(x.saturationPressureScale);
    a(x.saturationTemperature);
    a(x.gasContactResistance);
    a(x.solidContactResistance);
}
template<class Archive> void visit(Archive& a,ParticlePolicyConfig& x){
    a(x.condensedIndex);
    a(x.maxFaceCrossings);
    a(x.heatTransferCoefficient);
    a(x.contactHeatTransferCoefficient);
    a(x.contactDuration);
    a(x.emissivity);
    a(x.contact);
}
template<class Archive> void visit(Archive& a,MeshMotionConfig& x){
    a(x.policy);
    a(x.amplitude);
    a(x.spatialWaveNumber);
    a(x.spatialOrigin);
    a(x.angularFrequency);
    a(x.timeOrigin);
}
template<class Archive> void visit(Archive& a,PhysicsConfig& x){
    a(x.enableGas);
    a(x.meshMotion);
    a(x.particle);
    a(x.material);
    a(x.species);
    a(x.condensed);
    a(x.reactions);
    a(x.nReactions);
    a(x.nElements);
    a(x.liquid);
    a(x.liquidViscosity);
    a(x.liquidReferencePressure);
    a(x.meltTemperature);
    a(x.permeability);
    a(x.poreViscosity);
    a(x.gasViscosity);
    a(x.gasConductivity);
    a(x.gasDiffusivity);
    a(x.emissivity);
    a(x.ambientTemperature);
    a(x.gravity);
    a(x.enableFilm);
    a(x.enableReactions);
    a(x.enableParticles);
    a(x.enableSst);
    a(x.enableRadiation);
    a(x.filmThermalMode);
    a(x.reconstruction);
    a(x.sst);
    a(x.tolerances);
    a(x.minDt);
    a(x.maxDt);
    a(x.cfl);
    a(x.spatialOrder);
    a(x.modelFingerprint);
    a(x.speciesFingerprint);
    a(x.schemaFingerprint);
}
template<class Archive> void visit(Archive& a,ModelConfig& x){
    a(x.physics);
    a(x.speciesNames);
    a(x.condensedNames);
    a(x.elementNames);
    a(x.modelName);
    a(x.materialSource);
    a(x.mechanismSource);
    a(x.sourceFingerprint);
    a(x.baseCommit);
    a(x.modelFingerprint);
    a(x.speciesFingerprint);
    a(x.schemaFingerprint);
}
template<class Archive> void visit(Archive& a,HostMesh& x){
    a(x.referencePoints);
    a(x.points);
    a(x.oldPoints);
    a(x.cellCentres);
    a(x.faceCentres);
    a(x.areaVectors);
    a(x.meshVelocity);
    a(x.volumes);
    a(x.oldVolumes);
    a(x.wallDistance);
    a(x.faceOffsets);
    a(x.facePoints);
    a(x.owner);
    a(x.neighbour);
    a(x.periodicPartner);
    a(x.cellFaceOffsets);
    a(x.cellFaces);
    a(x.cellFaceSigns);
    a(x.boundaryKind);
    a(x.boundaryPrimitive);
    a(x.boundarySst);
    a(x.faceIds);
    a(x.geometryVersion);
    a(x.topologyHash);
}
template<class Archive> void visit(Archive& a,SurfaceMesh& x){
    a(x.prescribedTopTraction);
    a(x.prescribedPressureGradient);
    a(x.edgeOwnerOffset);
    a(x.edgeNeighbourOffset);
    a(x.gasDistance);
    a(x.solidDistance);
    a(x.gasFace);
    a(x.solidFace);
    a(x.solidCell);
    a(x.edgeOwner);
    a(x.edgeNeighbour);
    a(x.persistentId);
    a(x.edgeLength);
    a(x.area);
    a(x.oldArea);
    a(x.sweptEdgeArea);
    a(x.edgeConormal);
    a(x.centre);
    a(x.normal);
    a(x.meshVelocity);
    a(x.baseVelocity);
    a(x.gasMapOffsets);
    a(x.gasMapFaces);
    a(x.gasMapWeights);
    a(x.normalOffsets);
    a(x.normalCoordinates);
    a(x.normalLayerMass);
    a(x.normalLayerVolume);
    a(x.oldNormalLayerVolume);
    a(x.normalThickness);
    a(x.normalPressure);
    a(x.normalPressureRate);
    a(x.bottomTemperature);
    a(x.topTemperature);
    a(x.bottomHeatFlux);
    a(x.topHeatFlux);
}
template<class Archive> void visit(Archive& a,HostStageGeometry& x){
    a(x.interval);
    a(x.geometryVersion);
    a(x.topologyHash);
    a(x.oldVolume);
    a(x.newVolume);
    a(x.evaluationVolume);
    a(x.sweptVolume);
    a(x.areaVector);
    a(x.cellCentre);
    a(x.faceCentre);
    a(x.oldPoints);
    a(x.newPoints);
}
template<class Archive> void visit(Archive& a,HostState& x){
    a(x.gasStages);
    a(x.solidStages);
    a(x.nextDt);
    a(x.lastAcceptedDt);
    a(x.gasMesh);
    a(x.solidMesh);
    a(x.surface);
    a(x.gas);
    a(x.solid);
    a(x.film);
    a(x.filmAux);
    a(x.particles);
    a(x.sst);
    a(x.solidSweepRemainder);
    a(x.gasVoidFraction);
    a(x.normalEnthalpy);
    a(x.normalLiquidFraction);
    a(x.normalAcceptedWallHeat);
    a(x.ledger);
    a(x.filmStorage);
    a(x.time);
    a(x.acceptedSteps);
    a(x.rejectedSteps);
    a(x.commitSequence);
    a(x.budget);
}
const char* const FieldSchema="Vec3:x y z;GasQ:mass momentum energy species;GasPrimitive:rho velocity temperature pressure soundSpeed Y;SolidQ:condensed progress pore energy porosity;FilmQ:mass enthalpy species;FilmAux:pressure thickness temperature baseVelocity meanVelocity topVelocity normal area kineticEnergy normalVelocity solidNormalVelocity solidFront gasFront;ParticleQ:contactRelativeVelocity contactMechanicalEnergy id cell face state position velocity mass energy diameter contactAge contactDuration contactArea rng;SstQ:rhoK rhoOmega;SstPrimitive:k omega eddyViscosity blend1 blend2;SpeciesThermo:R cp0 cp1 e0 Tmin Tmax element;CondensedThermo:rho cp0 cp1 e0 conductivity Tmin Tmax element;Reaction:A temperaturePower activationEnergy condensedNu gasNu order;ExchangePacket:step geometry face stage kind gasCell solidCell filmFace particleIndex mass species condensed pore poreSweep momentum energy conductive advective pressureWork viscousWork radiation liquidKineticAdvection consumerMask;FilmPressureVolumeUpdate:step geometry face stage filmFace oldPV newPV consumed;Tolerances:absoluteMass relativeMass absoluteEnergy relativeEnergy absoluteTemperature relativeTemperature absoluteGeometry relativeGeometry maxCouplingIterations maxRetries;Budget:boundaryMass boundarySpecies boundaryElements boundaryEnergy boundaryMomentum supportImpulse supportWork bodyWork radiation gclResidual exchangeMass exchangeSpecies exchangeEnergy exchangeMomentum filmPressureVolume filmKineticAdvection filmKineticStorage filmReducedResidual filmKineticDefect thinLayerGeometryDefect thinLayerStressDefect numericalEnergyResidual turbulenceOmegaConstraint turbulenceInventory turbulenceProduction turbulenceDissipation turbulenceBoundaryFlux consumedPackets;SstConfig:a1 betaStar beta1 beta2 gamma1 gamma2 sigmaK1 sigmaK2 sigmaOmega1 sigmaOmega2 turbulentPrandtl turbulentSchmidt productionLimit minimumK minimumOmega;MaterialPolicyConfig:minimumPorosity maximumPorosity phaseCondensed enableMelting enableSurfaceReactions enablePoreOutflow phaseFilmY nSurfaceReactions surfaceReactions surfaceGasOrder evaporation evaporationAccommodation saturationPressureScale saturationTemperature gasContactResistance solidContactResistance;ParticlePolicyConfig:condensedIndex maxFaceCrossings heatTransferCoefficient contactHeatTransferCoefficient contactDuration emissivity contact;MeshMotionConfig:policy amplitude spatialWaveNumber spatialOrigin angularFrequency timeOrigin;PhysicsConfig:enableGas meshMotion particle material species condensed reactions nReactions nElements liquid liquidViscosity liquidReferencePressure meltTemperature permeability poreViscosity gasViscosity gasConductivity gasDiffusivity emissivity ambientTemperature gravity enableFilm enableReactions enableParticles enableSst enableRadiation filmThermalMode reconstruction sst tolerances minDt maxDt cfl spatialOrder modelFingerprint speciesFingerprint schemaFingerprint;ModelConfig:physics speciesNames condensedNames elementNames modelName materialSource mechanismSource sourceFingerprint baseCommit modelFingerprint speciesFingerprint schemaFingerprint;HostMesh:referencePoints points oldPoints cellCentres faceCentres areaVectors meshVelocity volumes oldVolumes wallDistance faceOffsets facePoints owner neighbour periodicPartner cellFaceOffsets cellFaces cellFaceSigns boundaryKind boundaryPrimitive boundarySst faceIds geometryVersion topologyHash;SurfaceMesh:prescribedTopTraction prescribedPressureGradient edgeOwnerOffset edgeNeighbourOffset gasDistance solidDistance gasFace solidFace solidCell edgeOwner edgeNeighbour persistentId edgeLength area oldArea sweptEdgeArea edgeConormal centre normal meshVelocity baseVelocity gasMapOffsets gasMapFaces gasMapWeights normalOffsets normalCoordinates normalLayerMass normalLayerVolume oldNormalLayerVolume normalThickness normalPressure normalPressureRate bottomTemperature topTemperature bottomHeatFlux topHeatFlux;HostStageGeometry:interval geometryVersion topologyHash oldVolume newVolume evaluationVolume sweptVolume areaVector cellCentre faceCentre oldPoints newPoints;HostState:gasStages solidStages nextDt lastAcceptedDt gasMesh solidMesh surface gas solid film filmAux particles sst solidSweepRemainder gasVoidFraction normalEnthalpy normalLiquidFraction normalAcceptedWallHeat ledger filmStorage time acceptedSteps rejectedSteps commitSequence budget";
bool ready(const HostState& s,const PhysicsConfig& p,std::string& error){
    if(!finite(s.time)||s.time<0||!finite(s.nextDt)||s.nextDt<0||!finite(s.lastAcceptedDt)||s.lastAcceptedDt<0){error="checkpoint has invalid time controls";return false;}
    for(const auto& packet:s.ledger){if(!validatePacketMath(packet,p.tolerances)||packet.consumerMask!=requiredConsumers(packet.kind)){error="checkpoint refuses pending or invalid ledger";return false;}}
    for(const auto& storage:s.filmStorage)if(!storage.consumed||!finite(storage.oldPV)||!finite(storage.newPV)){error="checkpoint refuses pending film storage";return false;}
    const HostMesh* meshesToCheck[]={&s.gasMesh,&s.solidMesh};
    for(const HostMesh* mesh:meshesToCheck)if(!mesh->volumes.empty()){
        const auto cells=mesh->volumes.size(),faces=mesh->owner.size(),points=mesh->points.size();
        if(!faces||!points||mesh->oldVolumes.size()!=cells||mesh->oldPoints.size()!=points||mesh->cellCentres.size()!=cells||mesh->faceCentres.size()!=faces||mesh->areaVectors.size()!=faces||mesh->faceIds.size()!=faces||mesh->boundaryKind.size()!=faces||mesh->boundaryPrimitive.size()!=faces||mesh->periodicPartner.size()!=faces||mesh->faceOffsets.size()!=faces+1||mesh->cellFaceOffsets.size()!=cells+1){error="checkpoint has missing mandatory mesh/history fields";return false;}
        for(std::size_t f=0;f<faces;++f)if(mesh->owner[f]<0||static_cast<std::size_t>(mesh->owner[f])>=cells||mesh->neighbour.size()!=faces||mesh->neighbour[f]<-1||(mesh->neighbour[f]>=0&&static_cast<std::size_t>(mesh->neighbour[f])>=cells)){error="checkpoint cell address exceeds declared inventory";return false;}
        HostMesh rebuilt=*mesh;if(!rebuildGeometry(rebuilt,error))return false;
        if(rebuilt.volumes.size()!=cells||rebuilt.cellCentres.size()!=cells||rebuilt.faceCentres.size()!=faces){error="checkpoint declared mesh count differs from rebuilt topology";return false;}
        if(rebuilt.topologyHash!=mesh->topologyHash){error="checkpoint topology hash mismatch";return false;}
        for(std::size_t c=0;c<cells;++c)if(!closeEnough(rebuilt.volumes[c],mesh->volumes[c],p.tolerances.absoluteGeometry,p.tolerances.relativeGeometry)){error="checkpoint volume differs from actual polyhedron";return false;}
    }
    if(s.gas.size()!=s.gasMesh.volumes.size()){error="checkpoint gas count/geometry mismatch";return false;}
    if(!s.sst.empty()&&s.sst.size()!=s.gas.size()){error="checkpoint SST count mismatch";return false;}
    if(!s.gasVoidFraction.empty()&&s.gasVoidFraction.size()!=s.gas.size()){error="checkpoint void-fraction count mismatch";return false;}
    if(s.normalEnthalpy.size()!=s.normalLiquidFraction.size()){error="checkpoint normal phase count mismatch";return false;}
    if(s.acceptedSteps){
        for(int stage=0;stage<2;++stage){const HostStageGeometry* traces[]={&s.gasStages[stage],&s.solidStages[stage]};const HostMesh* meshes[]={&s.gasMesh,&s.solidMesh};
            for(int m=0;m<2;++m)if(!meshes[m]->volumes.empty()){
                const auto& trace=*traces[m];const auto& mesh=*meshes[m];
                if(!finite(trace.interval)||trace.interval<=0||trace.topologyHash!=mesh.topologyHash||trace.oldVolume.size()!=mesh.volumes.size()||trace.newVolume.size()!=mesh.volumes.size()||trace.evaluationVolume.size()!=mesh.volumes.size()||trace.sweptVolume.size()!=mesh.owner.size()||trace.areaVector.size()!=mesh.owner.size()||trace.oldPoints.size()!=mesh.points.size()||trace.newPoints.size()!=mesh.points.size()||trace.cellCentre.size()!=mesh.volumes.size()||trace.faceCentre.size()!=mesh.owner.size()){error="checkpoint is missing required accepted stage trace";return false;}
            }
        }
    }
    error.clear();return true;
}
std::string systemError(const char* operation){return std::string(operation)+": "+std::strerror(errno);}
bool syncDirectory(const std::string& path,const char* purpose,std::string& error){
    const int fd=::open(path.c_str(),O_RDONLY|O_DIRECTORY);
    if(fd<0){error=systemError(purpose);return false;}
    const int sync=::fsync(fd);const int syncError=errno;const int close=::close(fd);
    if(sync||close){if(sync)errno=syncError;error=systemError(purpose);return false;}
    return true;
}
std::string parentDirectory(std::string path){
    while(path.size()>1&&path.back()=='/')path.pop_back();
    const auto slash=path.find_last_of('/');return slash==std::string::npos?".":(slash==0?"/":path.substr(0,slash));
}
bool writeAll(int fd,const Bytes& bytes){std::size_t done=0;while(done<bytes.size()){const auto n=::write(fd,bytes.data()+done,bytes.size()-done);if(n<0&&errno==EINTR)continue;if(n<=0)return false;done+=static_cast<std::size_t>(n);}return true;}
} // namespace
bool writeCheckpoint(const std::string& directory,const ModelConfig& config,const HostState& state,std::string& error){
    std::string temp;int fd=-1;
    try{
        if(directory.empty()){error="empty checkpoint directory";return false;}
        if(!validateSpeciesSet(config,error)||!ready(state,config.physics,error))return false;
        HostState validated=state;if(!validateRuntimeState(config,validated,error))return false;
        ModelConfig model=config;HostState copy=state;Encoder configData,payload;configData(model);payload(copy);
        Encoder header;const unsigned char magic[8]={'C','H','M','T','C','P','2',0};header.data.insert(header.data.end(),magic,magic+8);
        header.integer(checkpointSchema,8);header.integer(0x01020304,4);header.integer(sizeof(Real),4);
        header.integer(Ns,4);header.integer(Nc,4);header.integer(Nr,4);header.integer(Ne,4);
        header.integer(checksum(reinterpret_cast<const unsigned char*>(FieldSchema),std::strlen(FieldSchema)),8);
        header.integer(configData.data.size(),8);header.integer(payload.data.size(),8);
        Bytes body=configData.data;body.insert(body.end(),payload.data.begin(),payload.data.end());
        header.integer(checksum(body.data(),body.size()),8);if(body.size()>MaxBytes-header.data.size())throw std::runtime_error("checkpoint payload exceeds file-size bound");header.data.insert(header.data.end(),body.begin(),body.end());
        const int directoryResult=::mkdir(directory.c_str(),0700);
        if(directoryResult!=0&&errno!=EEXIST){error=systemError("create checkpoint directory");return false;}
        // Also sync on EEXIST: a previous failed parent fsync may have left
        // this directory present but not durably linked. Retry must re-establish
        // parent durability rather than mistake existence for successful sync.
        if(!syncDirectory(parentDirectory(directory),"sync checkpoint parent directory entry",error))return false;
        struct stat st;if(::stat(directory.c_str(),&st)!=0||!S_ISDIR(st.st_mode)){error="checkpoint destination is not a directory";return false;}
        temp=directory+"/.state.chmt.XXXXXX";std::vector<char> name(temp.begin(),temp.end());name.push_back(0);fd=::mkstemp(name.data());temp=name.data();
        if(fd<0){error=systemError("create checkpoint temporary file");return false;}
        if(!writeAll(fd,header.data)||::fsync(fd)!=0){error=systemError("write/fsync checkpoint");::close(fd);fd=-1;::unlink(temp.c_str());return false;}
        if(::close(fd)!=0){fd=-1;error=systemError("close checkpoint");::unlink(temp.c_str());return false;}fd=-1;
        if(::rename(temp.c_str(),(directory+"/state.chmt").c_str())!=0){error=systemError("atomic checkpoint rename");::unlink(temp.c_str());return false;}temp.clear();
        if(!syncDirectory(directory,"sync checkpoint directory",error))return false;
        error.clear();return true;
    }catch(const std::exception& e){if(fd>=0)::close(fd);if(!temp.empty())::unlink(temp.c_str());error=std::string("checkpoint write: ")+e.what();return false;}
}
bool readCheckpoint(const std::string& directory,const ModelConfig& expected,HostState& output,std::string& error){
    try{
        if(!validateSpeciesSet(expected,error))return false;
        std::ifstream input(directory+"/state.chmt",std::ios::binary|std::ios::ate);if(!input){error="cannot open checkpoint state.chmt";return false;}
        const auto size=input.tellg();if(size<0||static_cast<std::uint64_t>(size)>MaxBytes){error="checkpoint size exceeds bound";return false;}
        Bytes bytes(static_cast<std::size_t>(size));input.seekg(0);if(!bytes.empty())input.read(reinterpret_cast<char*>(bytes.data()),size);if(!input){error="truncated checkpoint read";return false;}
        Decoder decoder(bytes);const unsigned char magic[8]={'C','H','M','T','C','P','2',0};for(auto x:magic)if(decoder.integer(1)!=x)throw std::runtime_error("checkpoint magic mismatch");
        if(decoder.integer(8)!=checkpointSchema||decoder.integer(4)!=0x01020304||decoder.integer(4)!=sizeof(Real)||decoder.integer(4)!=Ns||decoder.integer(4)!=Nc||decoder.integer(4)!=Nr||decoder.integer(4)!=Ne)throw std::runtime_error("checkpoint schema/endian/precision/species-count mismatch");
        if(decoder.integer(8)!=checksum(reinterpret_cast<const unsigned char*>(FieldSchema),std::strlen(FieldSchema)))throw std::runtime_error("checkpoint field schema mismatch");
        const auto configSize=decoder.integer(8),stateSize=decoder.integer(8),digest=decoder.integer(8);
        if(configSize>bytes.size()-decoder.offset||stateSize!=bytes.size()-decoder.offset-configSize)throw std::runtime_error("checkpoint payload length mismatch");
        if(checksum(bytes.data()+decoder.offset,bytes.size()-decoder.offset)!=digest)throw std::runtime_error("checkpoint content checksum mismatch");
        ModelConfig model=expected;Encoder configData;configData(model);
        if(configData.data.size()!=configSize||!std::equal(configData.data.begin(),configData.data.end(),bytes.begin()+decoder.offset))throw std::runtime_error("checkpoint model/species/build/configuration identity mismatch");
        decoder.offset+=configSize;HostState candidate;decoder(candidate);if(decoder.offset!=bytes.size())throw std::runtime_error("checkpoint has unexpected trailing fields");
        if(!ready(candidate,expected.physics,error))return false;
        HostState validated=candidate;if(!validateRuntimeState(expected,validated,error))return false;
        output=std::move(candidate);error.clear();return true;
    }catch(const std::exception& e){error=std::string("checkpoint read: ")+e.what();return false;}
}
} // namespace chmt
