#ifndef GpuThermalExchangeState_H
#define GpuThermalExchangeState_H

#include "dictionary.H"
#include "fileName.H"
#include "label.H"
#include "labelList.H"
#include "scalar.H"
#include "word.H"
#include "volFieldsFwd.H"

#include <cstdint>
#include <stdexcept>
#include <string>

namespace Foam
{

class Time;
class fvMesh;

namespace gpuThermal
{

class ThermalExchangeStateError
:
    public std::runtime_error
{
public:
    explicit ThermalExchangeStateError(const std::string& message);
};

struct ThermalExchangeRestartState
{
    label formatVersion{1};
    bool initialState{true};
    std::uint64_t exchangeSequence{0};
    label completedTimeIndex{0};
    scalar completedSimulationTimeS{0};
    scalar completedRadiationSimulationTimeS{0};
    scalar previousExchangeSimulationTimeS{0};
    word fluidMeshTopologySha1;
    word solidMeshTopologySha1;
    word auxiliarySolidMeshTopologySha1;
    word couplingConfigurationSha1;
    word wallTemperatureSha1UsedForCompletedInterval;
    word newlyUploadedWallTemperatureSha1;
    bool gasWallLedgerConsumed{false};
    bool particleWallLedgerConsumed{false};
    bool particleRadiationApplied{false};
    bool solidStateUpdated{false};
    bool wallTemperatureUploaded{false};
    bool particleMomentsRebuilt{false};
    scalar particleContactEnergyJ{0};
};

struct ThermalExchangeWriteDecision
{
    bool exchangeRequired{false};
    std::uint64_t exchangeSequence{0};
    label timeIndex{0};
    scalar simulationTimeS{0};
    scalar deltaTExchangeS{0};
};

class ThermalExchangeWriteEventStateMachine
{
    ThermalExchangeRestartState committedState_;
    bool pendingExchange_;
    ThermalExchangeWriteDecision pendingDecision_;

public:
    explicit ThermalExchangeWriteEventStateMachine
    (
        const ThermalExchangeRestartState& committedState
    );

    const ThermalExchangeRestartState& committedState() const;

    bool hasPendingExchange() const;

    ThermalExchangeWriteDecision prepareWriteEvent
    (
        bool writeTime,
        label timeIndex,
        scalar simulationTimeS
    );

    void cancelPreparedExchange();

    void commitPreparedExchange
    (
        const ThermalExchangeRestartState& completedState
    );
};

struct ThermalRestartPreflightExpectations
{
    label completedTimeIndex{0};
    scalar completedSimulationTimeS{0};
    word fluidMeshTopologySha1;
    word solidMeshTopologySha1;
    word auxiliarySolidMeshTopologySha1;
    word couplingConfigurationSha1;
    word fluidRegion{"region0"};
    word solidRegion{"solid"};
    word auxiliarySolidRegion;
    label startTimeIndex{0};
    scalar startTimeS{0};
    label fluidCellCount{-1};
    label fluidFaceCount{-1};
    label maximumParticleCount{-1};
    bool allowManifestlessInitialState{false};
    bool rejectNewerUncommittedTime{true};
};

class ThermalRestartPreflight
{
public:
    static const word& stateObjectName();

    static const word& manifestObjectName();

    static void validateStateSemantics
    (
        const ThermalExchangeRestartState& state
    );

    static ThermalExchangeRestartState validateRestartDirectory
    (
        const fileName& timeDirectory,
        const ThermalRestartPreflightExpectations& expected
    );

    static ThermalExchangeRestartState validateBeforeResidentInitialise
    (
        const Time& runTime,
        const fvMesh& fluidMesh,
        const dictionary& couplingDictionary
    );

    static void validateFiniteRestartFields
    (
        const fvMesh& fluidMesh,
        const fvMesh& solidMesh
    );

    static void requirePurgeWriteZero(const dictionary& controlDictionary);

    static void requireLosslessAsciiRestartPrecision
    (
        const dictionary& controlDictionary
    );

    static word canonicalConfigurationSha1(const dictionary& configuration);

    static word canonicalMeshTopologySha1(const fvMesh& mesh);

    static word canonicalWallTemperatureSha1
    (
        const volScalarField& temperature,
        const labelList& patchIds
    );

    static word fileSha1(const fileName& path);

    static dictionary stateDictionary
    (
        const ThermalExchangeRestartState& state
    );

    static ThermalExchangeRestartState stateFromDictionary
    (
        const dictionary& stateDictionary,
        const fileName& sourceName = fileName("thermalExchangeState")
    );

    static void commitManifestAtomically
    (
        const fileName& temporaryManifest,
        const fileName& finalManifest
    );
};

}                        
}                  

#endif
