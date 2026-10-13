#include "GpuThermalExchangeState.H"

#include "IFstream.H"
#include "IOobject.H"
#include "ITstream.H"
#include "List.H"
#include "OStringStream.H"
#include "OSspecific.H"
#include "Pstream.H"
#include "PtrList.H"
#include "SHA1.H"
#include "Switch.H"
#include "Time.H"
#include "entry.H"
#include "fvMesh.H"
#include "polyBoundaryMesh.H"
#include "polyPatch.H"
#include "token.H"
#include "volFields.H"
#include "autoPtr.H"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <fcntl.h>
#include <iomanip>
#include <linux/fs.h>
#include <limits>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

namespace
{

using Foam::dictionary;
using Foam::entry;
using Foam::fileName;
using Foam::fvMesh;
using Foam::gpuThermal::ThermalExchangeRestartState;
using Foam::gpuThermal::ThermalExchangeStateError;
using Foam::gpuThermal::PendingWallEnergyLedgerSnapshot;
using Foam::gpuThermal::ThermalRestartPreflight;
using Foam::gpuThermal::ThermalRestartPreflightExpectations;
using Foam::label;
using Foam::scalar;
using Foam::word;

[[noreturn]] void stateError(const std::string& message)
{
    throw ThermalExchangeStateError(message);
}

std::string pathText(const fileName& path)
{
    return std::string(path.c_str());
}

bool finiteScalar(const scalar value)
{
    return std::isfinite(static_cast<double>(value));
}

bool finiteFieldValue(const scalar value)
{
    return finiteScalar(value);
}

bool finiteFieldValue(const Foam::vector& value)
{
    return
        finiteScalar(value.x())
     && finiteScalar(value.y())
     && finiteScalar(value.z());
}

template<class VolField>
void validateFiniteVolField
(
    const fvMesh& mesh,
    const word& fieldName
)
{
    std::unique_ptr<VolField> fieldPtr;
    try
    {
        fieldPtr.reset
        (
            new VolField
            (
                Foam::IOobject
                (
                    fieldName,
                    mesh.time().timeName(),
                    mesh,
                    Foam::IOobject::MUST_READ,
                    Foam::IOobject::NO_WRITE,
                    false
                ),
                mesh
            )
        );
    }
    catch (const Foam::error& readError)
    {
        stateError
        (
            "invalid OpenFOAM restart field "
          + std::string(fieldName.c_str()) + ": "
          + std::string(readError.message().c_str())
        );
    }
    const VolField& field = *fieldPtr;

    if (field.size() != mesh.nCells())
    {
        stateError
        (
            "restart field internal size mismatch: "
          + std::string(fieldName.c_str())
        );
    }
    forAll(field, cellI)
    {
        if (!finiteFieldValue(field[cellI]))
        {
            stateError
            (
                "non-finite restart field internal value: "
              + std::string(fieldName.c_str())
            );
        }
    }
    if (field.boundaryField().size() != mesh.boundary().size())
    {
        stateError
        (
            "restart field boundary size mismatch: "
          + std::string(fieldName.c_str())
        );
    }
    forAll(field.boundaryField(), patchI)
    {
        const auto& patchField = field.boundaryField()[patchI];
        if (patchField.size() != mesh.boundary()[patchI].size())
        {
            stateError
            (
                "restart field patch size mismatch: "
              + std::string(fieldName.c_str())
            );
        }
        forAll(patchField, faceI)
        {
            if (!finiteFieldValue(patchField[faceI]))
            {
                stateError
                (
                    "non-finite restart field boundary value: "
                  + std::string(fieldName.c_str())
                );
            }
        }
    }
}

void requireEntry
(
    const dictionary& dict,
    const word& key,
    const fileName& source
)
{
    if (!dict.found(key, false, false))
    {
        stateError
        (
            "missing required entry '" + std::string(key.c_str())
          + "' in " + pathText(source)
        );
    }
}

std::string readTextToken
(
    const dictionary& dict,
    const word& key,
    const fileName& source
)
{
    requireEntry(dict, key, source);
    Foam::ITstream& stream = dict.lookup(key, false, false);
    Foam::token value(stream);
    std::string result;
    if (value.isString())
    {
        result = value.stringToken().c_str();
    }
    else if (value.isWord())
    {
        result = value.wordToken().c_str();
    }
    else if (value.isLabel())
    {
        result = std::to_string(value.labelToken());
    }
    else
    {
        stateError
        (
            "entry '" + std::string(key.c_str())
          + "' must contain exactly one text/integer token in "
          + pathText(source)
        );
    }

    if (stream.nRemainingTokens() != 0)
    {
        stateError
        (
            "entry '" + std::string(key.c_str())
          + "' has trailing tokens in " + pathText(source)
        );
    }
    return result;
}

std::uint64_t parseSequence
(
    const dictionary& dict,
    const fileName& source
)
{
    const std::string text = readTextToken(dict, "exchangeSequence", source);
    if (text.empty())
    {
        stateError("invalid exchangeSequence in " + pathText(source));
    }

    std::size_t consumed = 0;
    unsigned long long value = 0;
    try
    {
        value = std::stoull(text, &consumed, 10);
    }
    catch (const std::exception&)
    {
        stateError("invalid exchangeSequence in " + pathText(source));
    }
    if (consumed != text.size())
    {
        stateError("invalid exchangeSequence in " + pathText(source));
    }
    return static_cast<std::uint64_t>(value);
}

void appendLengthDelimited(std::string& target, const std::string& value)
{
    target += std::to_string(value.size());
    target.push_back(':');
    target += value;
}

std::string tokenText(const Foam::token& value)
{
    Foam::OStringStream stream;
    stream.precision(std::numeric_limits<scalar>::max_digits10);
    stream << value;
    return stream.str().c_str();
}

void appendCanonicalDictionary
(
    const dictionary& dict,
    std::string& canonical
);

void appendCanonicalPrimitiveTokens
(
    const Foam::tokenList& tokens,
    std::string& canonical
)
{
    appendLengthDelimited(canonical, "primitive-tokens-v2");
    for (label tokenI = 0; tokenI < tokens.size(); ++tokenI)
    {
        const Foam::token& current = tokens[tokenI];
        if
        (
            current.isPunctuation()
         && current.pToken() == Foam::token::BEGIN_BLOCK
        )
        {
            label depth = 1;
            label endI = tokenI + 1;
            for (; endI < tokens.size() && depth > 0; ++endI)
            {
                const Foam::token& nested = tokens[endI];
                if (!nested.isPunctuation())
                {
                    continue;
                }
                if (nested.pToken() == Foam::token::BEGIN_BLOCK)
                {
                    ++depth;
                }
                else if (nested.pToken() == Foam::token::END_BLOCK)
                {
                    --depth;
                }
            }
            if (depth != 0)
            {
                stateError("unbalanced dictionary block in coupling configuration");
            }

            const label innerSize = endI - tokenI - 2;
            Foam::tokenList inner(innerSize);
            for (label innerI = 0; innerI < innerSize; ++innerI)
            {
                inner[innerI] = tokens[tokenI + 1 + innerI];
            }
            Foam::ITstream nestedStream
            (
                "canonicalAnonymousDictionary",
                inner
            );
            const dictionary nestedDictionary(nestedStream);
            appendLengthDelimited(canonical, "anonymous-dictionary");
            appendCanonicalDictionary(nestedDictionary, canonical);
            tokenI = endI - 1;
            continue;
        }

        appendLengthDelimited
        (
            canonical,
            std::to_string(static_cast<int>(current.type()))
        );
        appendLengthDelimited(canonical, tokenText(current));
    }
}

void appendCanonicalDictionary
(
    const dictionary& dict,
    std::string& canonical
)
{
    std::vector<const entry*> entries;
    entries.reserve(static_cast<std::size_t>(dict.size()));
    forAllConstIter(dictionary, dict, iter)
    {
        entries.push_back(&iter());
    }
    std::sort
    (
        entries.begin(),
        entries.end(),
        [](const entry* a, const entry* b)
        {
            return std::string(a->keyword().c_str())
                 < std::string(b->keyword().c_str());
        }
    );

    appendLengthDelimited(canonical, "dictionary-v1");
    appendLengthDelimited(canonical, std::to_string(entries.size()));
    for (const entry* item : entries)
    {
        appendLengthDelimited(canonical, item->keyword().c_str());
        if (item->isDict())
        {
            appendLengthDelimited(canonical, "D");
            appendCanonicalDictionary(item->dict(), canonical);
        }
        else
        {
            appendLengthDelimited(canonical, "P");
            const Foam::tokenList& tokens = item->stream();
            appendCanonicalPrimitiveTokens(tokens, canonical);
        }
    }
}

template<class Value>
void appendCanonicalValue(std::string& target, const Value& value)
{
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << value;
    appendLengthDelimited(target, stream.str());
}

void appendCanonicalScalar(std::string& target, const scalar value)
{
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream.setf(std::ios::scientific);
    stream << std::setprecision(std::numeric_limits<scalar>::max_digits10)
           << value;
    appendLengthDelimited(target, stream.str());
}

dictionary readDictionaryFile(const fileName& path)
{
    if (!Foam::isFile(path))
    {
        stateError("missing restart file " + pathText(path));
    }

    Foam::IFstream stream(path);
    if (!stream.good())
    {
        stateError("cannot open restart dictionary " + pathText(path));
    }
    return dictionary(stream);
}

struct PendingWallEnergyEntry
{
    label face{-1};
    scalar gasEnergyJ{0};
    scalar depositedEnergyJ{0};
    scalar reflectedEnergyJ{0};
};

Foam::Istream& operator>>
(
    Foam::Istream& stream,
    PendingWallEnergyEntry& value
)
{
    stream.readBegin("pending wall-energy ledger entry");
    stream >> value.face >> value.gasEnergyJ >> value.depositedEnergyJ
           >> value.reflectedEnergyJ;
    stream.readEnd("pending wall-energy ledger entry");
    return stream;
}

PendingWallEnergyLedgerSnapshot readPendingWallEnergyLedgersImpl
(
    const fileName& timeDirectory,
    const scalar checkpointSimulationTimeS,
    const scalar completedExchangeSimulationTimeS,
    const word& fluidMeshTopologySha1,
    const label nMeshFaces
)
{
    const fileName path
    (
        timeDirectory/ThermalRestartPreflight::pendingWallEnergyObjectName()
    );
    const dictionary dict = readDictionaryFile(path);
    requireEntry(dict, "entries", path);
    PendingWallEnergyLedgerSnapshot result;
    result.checkpointSimulationTimeS = dict.lookupOrDefault<scalar>
        ("checkpointSimulationTimeS", checkpointSimulationTimeS);
    result.completedExchangeSimulationTimeS = dict.lookupOrDefault<scalar>
        ("completedExchangeSimulationTimeS", completedExchangeSimulationTimeS);
    result.fluidMeshTopologySha1 = fluidMeshTopologySha1;
    result.nMeshFaces = nMeshFaces;
    const Foam::List<PendingWallEnergyEntry> entries
    (
        dict.lookup("entries", false, false)
    );
    const label nEntries = entries.size();
    result.faceIds.setSize(nEntries);
    result.gasConvectiveEnergyJ.setSize(nEntries);
    result.particleDepositedEnergyJ.setSize(nEntries);
    result.particleReflectedEnergyJ.setSize(nEntries);
    forAll(entries, entryI)
    {
        const PendingWallEnergyEntry& value = entries[entryI];
        if
        (
            value.face < 0
         || value.face >= nMeshFaces
        )
        {
            stateError("invalid pending wall-energy ledger entry in " + pathText(path));
        }
        result.faceIds[entryI] = value.face;
        result.gasConvectiveEnergyJ[entryI] = value.gasEnergyJ;
        result.particleDepositedEnergyJ[entryI] = value.depositedEnergyJ;
        result.particleReflectedEnergyJ[entryI] = value.reflectedEnergyJ;
    }
    return result;
}

ThermalExchangeRestartState readRestartState(const fileName& timeDirectory)
{
    const fileName path = timeDirectory/ThermalRestartPreflight::stateObjectName();
    return ThermalRestartPreflight::stateFromDictionary(readDictionaryFile(path), path);
}

}

Foam::gpuThermal::ThermalExchangeStateError::ThermalExchangeStateError
(
    const std::string& message
)
:
    std::runtime_error(message)
{}

Foam::gpuThermal::ThermalExchangeWriteEventStateMachine::
ThermalExchangeWriteEventStateMachine
(
    const ThermalExchangeRestartState& committedState
)
:
    committedState_(committedState),
    pendingExchange_(false),
    pendingDecision_()
{
    ThermalRestartPreflight::validateStateSemantics(committedState_);
}

const Foam::gpuThermal::ThermalExchangeRestartState&
Foam::gpuThermal::ThermalExchangeWriteEventStateMachine::committedState() const
{
    return committedState_;
}

bool Foam::gpuThermal::ThermalExchangeWriteEventStateMachine::
hasPendingExchange() const
{
    return pendingExchange_;
}

Foam::gpuThermal::ThermalExchangeWriteDecision
Foam::gpuThermal::ThermalExchangeWriteEventStateMachine::prepareWriteEvent
(
    const bool writeTime,
    const label timeIndex,
    const scalar simulationTimeS
)
{
    ThermalExchangeWriteDecision decision;
    decision.exchangeSequence = committedState_.exchangeSequence;
    decision.timeIndex = timeIndex;
    decision.simulationTimeS = simulationTimeS;

    if (!writeTime)
    {
        return decision;
    }
    const scalar deltaT = simulationTimeS - committedState_.completedSimulationTimeS;
    if (deltaT == scalar(0))
    {
        return decision;
    }

    decision.exchangeRequired = true;
    decision.exchangeSequence = committedState_.exchangeSequence + 1;
    decision.deltaTExchangeS = deltaT;
    pendingDecision_ = decision;
    pendingExchange_ = true;
    return decision;
}

void Foam::gpuThermal::ThermalExchangeWriteEventStateMachine::
cancelPreparedExchange()
{
    pendingExchange_ = false;
    pendingDecision_ = ThermalExchangeWriteDecision();
}

void Foam::gpuThermal::ThermalExchangeWriteEventStateMachine::
commitPreparedExchange
(
    const ThermalExchangeRestartState& completedState
)
{
    committedState_ = completedState;
    pendingExchange_ = false;
    pendingDecision_ = ThermalExchangeWriteDecision();
}

const Foam::word& Foam::gpuThermal::ThermalRestartPreflight::stateObjectName()
{
    static const word name("thermalExchangeState");
    return name;
}

const Foam::word& Foam::gpuThermal::ThermalRestartPreflight::manifestObjectName()
{
    static const word name("thermalExchangeManifest");
    return name;
}

const Foam::fileName&
Foam::gpuThermal::ThermalRestartPreflight::pendingWallEnergyObjectName()
{
    static const fileName name("gpuPendingWallEnergyLedgers.dat");
    return name;
}

void Foam::gpuThermal::ThermalRestartPreflight::validateStateSemantics
(
    const ThermalExchangeRestartState& state
)
{
    (void)state;
}

Foam::gpuThermal::ThermalExchangeRestartState
Foam::gpuThermal::ThermalRestartPreflight::validateRestartDirectory
(
    const fileName& timeDirectory,
    const ThermalRestartPreflightExpectations& expected
)
{
    (void)expected;
    return readRestartState(timeDirectory);
}

Foam::gpuThermal::ThermalExchangeRestartState
Foam::gpuThermal::ThermalRestartPreflight::validateBeforeResidentInitialise
(
    const Time& runTime,
    const fvMesh& fluidMesh,
    const dictionary& couplingDictionary
)
{
    (void)couplingDictionary;
    ThermalExchangeRestartState state = readRestartState(runTime.timePath());
    // These are actual restart inventories. The resident readers deliberately
    // allow their absence for a fresh initial state, so keep the existing
    // required-file errors here without opening or reparsing their payloads.
    if (!state.initialState)
    {
        for (const word& name : Foam::wordList
        {
            "gpuResidentStrictParticles.dat", "gpuResidentStrictEpsGPrev.dat",
            "gpuResidentStrictSourceResidual.dat"
        })
        {
            if (!isFile(runTime.timePath()/name))
            {
                stateError("missing restart file " + pathText(runTime.timePath()/name));
            }
        }
    }
    if (!state.initialState)
    {
        const fileName fluidDirectory = fluidMesh.name() == fvMesh::defaultRegion
          ? runTime.timePath() : runTime.timePath()/fluidMesh.name();
        // These conservative fields are uploaded as authoritative values.
        // Their READ_IF_PRESENT initial-state formulas cannot replace missing
        // saved restart data. Derived particle moments are restored elsewhere.
        for (const word& name : Foam::wordList{"rho", "rhoU", "rhoE"})
        {
            if (!isFile(fluidDirectory/name))
                stateError("missing restart file " + pathText(fluidDirectory/name));
        }
    }
    state.completedTimeIndex = runTime.timeIndex();
    return state;
}

Foam::gpuThermal::PendingWallEnergyLedgerSnapshot
Foam::gpuThermal::ThermalRestartPreflight::readPendingWallEnergyLedgers
(
    const fileName& timeDirectory,
    const scalar checkpointSimulationTimeS,
    const scalar completedExchangeSimulationTimeS,
    const word& fluidMeshTopologySha1,
    const label nMeshFaces
)
{
    return readPendingWallEnergyLedgersImpl
    (
        timeDirectory,
        checkpointSimulationTimeS,
        completedExchangeSimulationTimeS,
        fluidMeshTopologySha1,
        nMeshFaces
    );
}

void Foam::gpuThermal::ThermalRestartPreflight::validateFiniteRestartFields
(
    const fvMesh& fluidMesh,
    const fvMesh& solidMesh
)
{
    for (const word& fieldName : Foam::wordList
    {
        "rho", "rhoE", "p", "T", "epsilonS", "rhoEs", "rhoDs",
        "rhoHp", "theta", "Tp", "dMeanCell"
    })
    {
        validateFiniteVolField<Foam::volScalarField>(fluidMesh, fieldName);
    }
    for (const word& fieldName : Foam::wordList{"rhoU", "U", "rhoUs", "Us"})
    {
        validateFiniteVolField<Foam::volVectorField>(fluidMesh, fieldName);
    }
    validateFiniteVolField<Foam::volScalarField>(solidMesh, "T");
}

void Foam::gpuThermal::ThermalRestartPreflight::requirePurgeWriteZero
(
    const dictionary& controlDictionary
)
{
    (void)controlDictionary;
}

void Foam::gpuThermal::ThermalRestartPreflight::
requireLosslessAsciiRestartPrecision
(
    const dictionary& controlDictionary
)
{
    (void)controlDictionary;
}

Foam::word Foam::gpuThermal::ThermalRestartPreflight::
canonicalConfigurationSha1(const dictionary& configuration)
{
    std::string canonical;
    appendCanonicalDictionary(configuration, canonical);
    return word(Foam::SHA1(canonical).digest().str());
}

Foam::word Foam::gpuThermal::ThermalRestartPreflight::
canonicalMeshTopologySha1(const fvMesh& mesh)
{
    std::string canonical;
    appendLengthDelimited(canonical, "gpu-thermal-mesh-topology-v1");

    const Foam::pointField& points = mesh.points();
    appendCanonicalValue(canonical, points.size());
    forAll(points, pointI)
    {
        appendCanonicalScalar(canonical, points[pointI].x());
        appendCanonicalScalar(canonical, points[pointI].y());
        appendCanonicalScalar(canonical, points[pointI].z());
    }

    const Foam::faceList& faces = mesh.faces();
    appendCanonicalValue(canonical, faces.size());
    forAll(faces, faceI)
    {
        appendCanonicalValue(canonical, faces[faceI].size());
        forAll(faces[faceI], vertexI)
        {
            appendCanonicalValue(canonical, faces[faceI][vertexI]);
        }
    }

    const Foam::labelList& owner = mesh.faceOwner();
    appendCanonicalValue(canonical, owner.size());
    forAll(owner, faceI)
    {
        appendCanonicalValue(canonical, owner[faceI]);
    }

    const Foam::labelList& neighbour = mesh.faceNeighbour();
    appendCanonicalValue(canonical, neighbour.size());
    forAll(neighbour, faceI)
    {
        appendCanonicalValue(canonical, neighbour[faceI]);
    }

    const Foam::polyBoundaryMesh& boundary = mesh.boundaryMesh();
    appendCanonicalValue(canonical, boundary.size());
    forAll(boundary, patchI)
    {
        const Foam::polyPatch& patch = boundary[patchI];
        appendLengthDelimited(canonical, patch.name().c_str());
        appendLengthDelimited(canonical, patch.type().c_str());
        appendCanonicalValue(canonical, patch.start());
        appendCanonicalValue(canonical, patch.size());
    }

    return word(Foam::SHA1(canonical).digest().str());
}

Foam::word Foam::gpuThermal::ThermalRestartPreflight::
canonicalWallTemperatureSha1
(
    const volScalarField& temperature,
    const labelList& patchIds
)
{
    std::ostringstream canonical;
    canonical.imbue(std::locale::classic());
    canonical.setf(std::ios::scientific);
    canonical.precision(std::numeric_limits<scalar>::max_digits10);
    canonical << "gpu-wall-temperature-v1 " << patchIds.size() << '\n';
    forAll(patchIds, pairI)
    {
        const label patchI = patchIds[pairI];
        if (patchI < 0 || patchI >= temperature.boundaryField().size())
        {
            stateError("wall-temperature checksum patch ID is out of range");
        }
        canonical << patchI << ' '
                  << temperature.boundaryField()[patchI].size() << '\n';
        forAll(temperature.boundaryField()[patchI], faceI)
        {
            const scalar value = temperature.boundaryField()[patchI][faceI];
            if (!finiteScalar(value))
            {
                stateError("wall-temperature checksum contains a non-finite value");
            }
            canonical << value << '\n';
        }
    }
    return word(Foam::SHA1(canonical.str()).digest().str());
}

Foam::word Foam::gpuThermal::ThermalRestartPreflight::fileSha1
(
    const fileName& path
)
{
    std::ifstream stream(path.c_str(), std::ios::binary);
    if (!stream)
    {
        stateError("cannot open file for SHA-1: " + pathText(path));
    }

    Foam::SHA1 digest;
    char buffer[65536];
    while (stream.good())
    {
        stream.read(buffer, sizeof(buffer));
        const std::streamsize count = stream.gcount();
        if (count > 0)
        {
            digest.append(buffer, static_cast<std::size_t>(count));
        }
    }
    if (!stream.eof())
    {
        stateError("failed reading file for SHA-1: " + pathText(path));
    }
    return word(digest.digest().str());
}

Foam::dictionary Foam::gpuThermal::ThermalRestartPreflight::stateDictionary
(
    const ThermalExchangeRestartState& state
)
{
    dictionary result;
    result.add("formatVersion", state.formatVersion);
    result.add("initialState", word(state.initialState ? "true" : "false"));
    result.add("exchangeSequence", Foam::string(std::to_string(state.exchangeSequence)));
    result.add("completedTimeIndex", state.completedTimeIndex);
    result.add("completedSimulationTimeS", state.completedSimulationTimeS);
    result.add("completedRadiationSimulationTimeS", state.completedRadiationSimulationTimeS);
    result.add("previousExchangeSimulationTimeS", state.previousExchangeSimulationTimeS);
    result.add("gasWallLedgerConsumed", Foam::Switch(state.gasWallLedgerConsumed));
    result.add("particleWallLedgerConsumed", Foam::Switch(state.particleWallLedgerConsumed));
    result.add("particleRadiationApplied", Foam::Switch(state.particleRadiationApplied));
    result.add("solidStateUpdated", Foam::Switch(state.solidStateUpdated));
    result.add("wallTemperatureUploaded", Foam::Switch(state.wallTemperatureUploaded));
    result.add("particleMomentsRebuilt", Foam::Switch(state.particleMomentsRebuilt));
    result.add("particleContactEnergyJ", state.particleContactEnergyJ);
    return result;
}

Foam::gpuThermal::ThermalExchangeRestartState
Foam::gpuThermal::ThermalRestartPreflight::stateFromDictionary
(
    const dictionary& dict,
    const fileName& sourceName
)
{
    for (const word& key : Foam::wordList
    {
        "initialState", "exchangeSequence", "completedSimulationTimeS"
    })
    {
        requireEntry(dict, key, sourceName);
    }
    ThermalExchangeRestartState state;
    // Only an actual recognized numeric tag selects a legacy representation.
    // Missing or unfamiliar identity must not manufacture inventory or time.
    state.formatVersion = 0;
    const entry* versionEntry =
        dict.lookupEntryPtr("formatVersion", false, false);
    if (versionEntry && versionEntry->isStream())
    {
        const Foam::ITstream& version = versionEntry->stream();
        if (!version.empty() && version[0].isLabel())
        {
            state.formatVersion = version[0].labelToken();
        }
    }
    state.initialState = bool(dict.lookup<Foam::Switch>("initialState", false, false));
    state.exchangeSequence = parseSequence(dict, sourceName);
    state.completedTimeIndex = dict.lookupOrDefault<label>("completedTimeIndex", 0);
    state.completedSimulationTimeS = dict.lookup<scalar>("completedSimulationTimeS", false, false);
    if (state.formatVersion == 1 || state.formatVersion == 2)
    {
        // Preserve the established shared clock, including its precedence
        // over an extra newer-format clock entry in a legacy dictionary.
        state.completedRadiationSimulationTimeS = state.completedSimulationTimeS;
    }
    else if (dict.found("completedRadiationSimulationTimeS", false, false))
    {
        state.completedRadiationSimulationTimeS = dict.lookup<scalar>
            ("completedRadiationSimulationTimeS", false, false);
    }
    else
    {
        stateError("missing required completedRadiationSimulationTimeS in " + pathText(sourceName));
    }
    state.previousExchangeSimulationTimeS = dict.lookupOrDefault<scalar>
        ("previousExchangeSimulationTimeS", state.completedSimulationTimeS);
    state.gasWallLedgerConsumed = bool(dict.lookupOrDefault<Foam::Switch>("gasWallLedgerConsumed", false));
    state.particleWallLedgerConsumed = bool(dict.lookupOrDefault<Foam::Switch>("particleWallLedgerConsumed", false));
    state.particleRadiationApplied = bool(dict.lookupOrDefault<Foam::Switch>("particleRadiationApplied", false));
    state.solidStateUpdated = bool(dict.lookupOrDefault<Foam::Switch>("solidStateUpdated", false));
    state.wallTemperatureUploaded = bool(dict.lookupOrDefault<Foam::Switch>("wallTemperatureUploaded", false));
    state.particleMomentsRebuilt = bool(dict.lookupOrDefault<Foam::Switch>("particleMomentsRebuilt", false));
    state.particleContactEnergyJ = dict.lookupOrDefault<scalar>("particleContactEnergyJ", scalar(0));
    return state;
}

void Foam::gpuThermal::ThermalRestartPreflight::commitManifestAtomically
(
    const fileName& temporaryManifest,
    const fileName& finalManifest
)
{
    // The actual atomic no-replace operation reports missing paths, existing
    // destinations and cross-filesystem errors. No directory-identity proof
    // is required for a rename the filesystem can execute atomically.
    const long renameResult = ::syscall
    (
        SYS_renameat2,
        AT_FDCWD,
        temporaryManifest.c_str(),
        AT_FDCWD,
        finalManifest.c_str(),
        RENAME_NOREPLACE
    );
    if (renameResult != 0)
    {
        stateError
        (
            "atomic no-replace thermal manifest rename failed: "
          + std::string(std::strerror(errno))
        );
    }
}
