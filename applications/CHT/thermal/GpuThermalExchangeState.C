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
#include <dirent.h>
#include <fstream>
#include <fcntl.h>
#include <iomanip>
#include <linux/fs.h>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
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

bool restartStateTimeNotAfterDirectory
(
    const scalar saved,
    const scalar directoryTime
)
{
                                                                           
                                                                             
                                                                          
                                                               
    const scalar scale =
        std::max(scalar(1), std::max(std::abs(saved), std::abs(directoryTime)));
    return saved <= directoryTime + scalar(1e-7)*scale;
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

bool validLowerSha1(const word& value)
{
    if (value.size() != 40)
    {
        return false;
    }

    for (const char character : std::string(value.c_str()))
    {
        if
        (
            !(character >= '0' && character <= '9')
         && !(character >= 'a' && character <= 'f')
        )
        {
            return false;
        }
    }
    return true;
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
    if (text.empty() || text.front() == '-')
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

bool sameState
(
    const ThermalExchangeRestartState& a,
    const ThermalExchangeRestartState& b
)
{
    return
        a.formatVersion == b.formatVersion
     && a.initialState == b.initialState
     && a.exchangeSequence == b.exchangeSequence
     && a.completedTimeIndex == b.completedTimeIndex
     && a.completedSimulationTimeS == b.completedSimulationTimeS
     && a.completedRadiationSimulationTimeS
        == b.completedRadiationSimulationTimeS
     && a.previousExchangeSimulationTimeS
        == b.previousExchangeSimulationTimeS
     && a.fluidMeshTopologySha1 == b.fluidMeshTopologySha1
     && a.solidMeshTopologySha1 == b.solidMeshTopologySha1
     && a.auxiliarySolidMeshTopologySha1
        == b.auxiliarySolidMeshTopologySha1
     && a.couplingConfigurationSha1 == b.couplingConfigurationSha1
     && a.wallTemperatureSha1UsedForCompletedInterval
        == b.wallTemperatureSha1UsedForCompletedInterval
     && a.newlyUploadedWallTemperatureSha1
        == b.newlyUploadedWallTemperatureSha1
     && a.gasWallLedgerConsumed == b.gasWallLedgerConsumed
     && a.particleWallLedgerConsumed == b.particleWallLedgerConsumed
     && a.particleRadiationApplied == b.particleRadiationApplied
     && a.solidStateUpdated == b.solidStateUpdated
     && a.wallTemperatureUploaded == b.wallTemperatureUploaded
     && a.particleMomentsRebuilt == b.particleMomentsRebuilt
     && a.particleContactEnergyJ == b.particleContactEnergyJ;
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

struct ManifestArtifact
{
    word role;
    fileName relativePath;
    word sha1;
    label count{-1};
};

Foam::Istream& operator>>(Foam::Istream& stream, ManifestArtifact& artifact)
{
    Foam::string sha1Text;
    stream.readBegin("thermal restart artifact");
    stream >> artifact.role >> artifact.relativePath >> sha1Text
           >> artifact.count;
    stream.readEnd("thermal restart artifact");
    artifact.sha1 = word(sha1Text);
    return stream;
}

struct ParsedManifest
{
    ThermalExchangeRestartState state;
    fileName stateFile;
    word stateSha1;
    word solidFieldName;
    Foam::List<ManifestArtifact> artifacts;
};

ParsedManifest parseManifest
(
    const dictionary& dict,
    const fileName& source
)
{
    ParsedManifest result;
    result.state = ThermalRestartPreflight::stateFromDictionary(dict, source);
    result.stateFile = fileName
    (
        readTextToken(dict, "thermalExchangeStateFile", source)
    );
    result.stateSha1 = word
    (
        readTextToken(dict, "thermalExchangeStateSha1", source)
    );
    result.solidFieldName = word
    (
        readTextToken(dict, "solidFieldName", source)
    );
    requireEntry(dict, "artifacts", source);
    result.artifacts = Foam::List<ManifestArtifact>
    (
        dict.lookup("artifacts", false, false)
    );
    return result;
}

bool validRelativePath(const fileName& relativePath)
{
    const std::string value = pathText(relativePath);
    if (value.empty() || value.front() == '/' || value.front() == '\\')
    {
        return false;
    }
    if (value.find("..") != std::string::npos || value.find('\\') != std::string::npos)
    {
        return false;
    }
    return true;
}

void requireNoTrailingToken(std::istream& stream, const fileName& path)
{
    std::string extra;
    if (stream >> extra)
    {
        stateError("trailing data in restart mirror " + pathText(path));
    }
}

double readFiniteDouble(std::istream& stream, const fileName& path)
{
    double value = 0;
    if (!(stream >> value) || !std::isfinite(value))
    {
        stateError("invalid non-finite scalar in restart mirror " + pathText(path));
    }
    return value;
}

label validateParticleMirror
(
    const fileName& path,
    const label expectedCount,
    const label maximumCount,
    const label fluidCellCount,
    const bool allowLegacyZero = false
)
{
    std::ifstream stream(path.c_str());
    std::string headerLine;
    if (!std::getline(stream, headerLine))
    {
        stateError("missing GPU particle mirror header in " + pathText(path));
    }
    if (allowLegacyZero)
    {
        std::istringstream legacyHeader(headerLine);
        long long legacyCount = -1;
        std::string legacyExtra;
        if
        (
            (legacyHeader >> legacyCount)
         && legacyCount == 0
         && !(legacyHeader >> legacyExtra)
        )
        {
            requireNoTrailingToken(stream, path);
            return 0;
        }
    }
    std::istringstream header(headerLine);
    std::string version;
    long long count = -1;
    long long chunkCapacity = -1;
    if
    (
        !(header >> version >> count)
     || count < 0
     || (expectedCount >= 0 && count != expectedCount)
     || count > std::numeric_limits<label>::max()
     || (maximumCount >= 0 && count > maximumCount)
    )
    {
        stateError("GPU particle mirror header/count mismatch in " + pathText(path));
    }

    const bool legacyText = version == "UGKP_PARTICLES_SCHEMA4";
    const bool weightedBinary = version == "UGKP_PARTICLES_SCHEMA1_BIN";
    if (!legacyText && !weightedBinary)
    {
        stateError("unsupported GPU particle mirror version in " + pathText(path));
    }
    if
    (
        weightedBinary
     && (!(header >> chunkCapacity) || chunkCapacity <= 0)
    )
    {
        stateError("invalid UGKP particle mirror chunk capacity in " + pathText(path));
    }
    std::string headerExtra;
    if (header >> headerExtra)
    {
        stateError("GPU particle mirror header has trailing data in " + pathText(path));
    }

    if (weightedBinary)
    {
        if (sizeof(double) != 8 || sizeof(label) != 4)
        {
            stateError("UGKP binary particle mirror requires DP/Int32 in " + pathText(path));
        }
        long long consumed = 0;
        std::vector<double> scalarBuffer;
        std::vector<label> labelBuffer;
        std::vector<unsigned long long> ullBuffer;
        while (consumed < count)
        {
            std::uint32_t chunkCount = 0;
            stream.read
            (
                reinterpret_cast<char*>(&chunkCount),
                sizeof(chunkCount)
            );
            if
            (
                !stream
             || chunkCount == 0
             || chunkCount > static_cast<std::uint32_t>(chunkCapacity)
             || consumed + chunkCount > count
            )
            {
                stateError("invalid UGKP particle mirror chunk in " + pathText(path));
            }

            const std::size_t n = static_cast<std::size_t>(chunkCount);
            scalarBuffer.resize(n);
            for (int field = 0; field < 10; ++field)
            {
                stream.read
                (
                    reinterpret_cast<char*>(scalarBuffer.data()),
                    static_cast<std::streamsize>(n*sizeof(double))
                );
                if (!stream)
                {
                    stateError("truncated UGKP scalar particle block in " + pathText(path));
                }
                for (const double value : scalarBuffer)
                {
                    if (!std::isfinite(value))
                    {
                        stateError("non-finite UGKP particle scalar in " + pathText(path));
                    }
                }
                if (field == 8 || field == 9)
                {
                    for (const double value : scalarBuffer)
                    {
                        if (!(value > 0.0))
                        {
                            stateError("non-positive UGKP particle diameter/mass in " + pathText(path));
                        }
                    }
                }
            }

            labelBuffer.resize(n);
            stream.read
            (
                reinterpret_cast<char*>(labelBuffer.data()),
                static_cast<std::streamsize>(n*sizeof(label))
            );
            if (!stream)
            {
                stateError("truncated UGKP particle cell block in " + pathText(path));
            }
            for (const label cellId : labelBuffer)
            {
                if (cellId < 0 || (fluidCellCount >= 0 && cellId >= fluidCellCount))
                {
                    stateError("invalid UGKP particle cell in " + pathText(path));
                }
            }
            stream.read
            (
                reinterpret_cast<char*>(labelBuffer.data()),
                static_cast<std::streamsize>(n*sizeof(label))
            );
            if (!stream)
            {
                stateError("truncated UGKP particle status block in " + pathText(path));
            }
            for (const label status : labelBuffer)
            {
                if (status != 1)
                {
                    stateError("invalid UGKP particle status in " + pathText(path));
                }
            }

            ullBuffer.resize(n);
            for (int field = 0; field < 2; ++field)
            {
                stream.read
                (
                    reinterpret_cast<char*>(ullBuffer.data()),
                    static_cast<std::streamsize>
                    (
                        n*sizeof(unsigned long long)
                    )
                );
                if (!stream)
                {
                    stateError("truncated UGKP particle metadata block in " + pathText(path));
                }
            }
            consumed += chunkCount;
        }
        char trailing = 0;
        if (stream.read(&trailing, 1))
        {
            stateError("trailing binary data in UGKP particle mirror " + pathText(path));
        }
        return label(count);
    }

    for (long long particleI = 0; particleI < count; ++particleI)
    {
        std::string rowText;
        if (!std::getline(stream, rowText))
        {
            stateError("missing GPU particle row in " + pathText(path));
        }
        std::istringstream row(rowText);
        double values[9];
        for (double& value : values)
        {
            value = readFiniteDouble(row, path);
        }
        long long cellId = -1;
        long long status = 0;
        unsigned long long rng = 0;
        unsigned long long originalId = 0;
        const bool rowRead = static_cast<bool>
        (
            row >> cellId >> status >> rng >> originalId
        );
        if
        (
            !rowRead
         || cellId < 0
         || cellId > std::numeric_limits<label>::max()
         || (fluidCellCount >= 0 && cellId >= fluidCellCount)
         || status != 1
         || originalId == 0
         || values[8] <= 0
        )
        {
            stateError("invalid GPU particle row in " + pathText(path));
        }
        std::string rowExtra;
        if (row >> rowExtra)
        {
            stateError("GPU particle row has trailing columns in " + pathText(path));
        }
                                                                             
                                                                             
                                             
    }
    requireNoTrailingToken(stream, path);
    return label(count);
}

label validateEpsGPrevMirror
(
    const fileName& path,
    const label expectedCount
)
{
    std::ifstream stream(path.c_str());
    long long count = -1;
    if
    (
        !(stream >> count)
     || count < 0
     || (expectedCount >= 0 && count != expectedCount)
     || count > std::numeric_limits<label>::max()
    )
    {
        stateError("epsGPrev mirror count mismatch in " + pathText(path));
    }
    for (long long cellI = 0; cellI < count; ++cellI)
    {
        const double value = readFiniteDouble(stream, path);
        if (value < 0 || value > 1)
        {
            stateError("epsGPrev value outside [0,1] in " + pathText(path));
        }
    }
    requireNoTrailingToken(stream, path);
    return label(count);
}

label validateSourceResidualMirror
(
    const fileName& path,
    const label expectedCount,
    const label maximumFaceCount
)
{
    std::ifstream stream(path.c_str());
    std::string version;
    long long count = -1;
    if
    (
        !(stream >> version >> count)
     || version != "UGKP_SOURCE_RESIDUAL_SCHEMA1"
     || count < 0
     || (expectedCount >= 0 && count != expectedCount)
     || count > std::numeric_limits<label>::max()
    )
    {
        stateError("source residual mirror header/count mismatch in " + pathText(path));
    }
    std::set<long long> sourceFaces;
    for (long long sourceI = 0; sourceI < count; ++sourceI)
    {
        long long face = -1;
        if
        (
            !(stream >> face)
         || face < 0
         || (maximumFaceCount >= 0 && face >= maximumFaceCount)
        )
        {
            stateError("invalid source face in " + pathText(path));
        }
        const double residual = readFiniteDouble(stream, path);
        if (residual < 0 || !sourceFaces.insert(face).second)
        {
            stateError("invalid/duplicate source residual row in " + pathText(path));
        }
    }
    requireNoTrailingToken(stream, path);
    return label(count);
}

void validateNoNonFiniteFieldToken(const fileName& path)
{
    std::ifstream stream(path.c_str(), std::ios::binary);
    if (!stream)
    {
        stateError("cannot open restart field " + pathText(path));
    }
    const std::string content
    {
        std::istreambuf_iterator<char>(stream),
        std::istreambuf_iterator<char>()
    };
    static const std::regex nonFinite
    (
        R"((^|[^A-Za-z0-9_])[-+]?(nan|inf|infinity)([^A-Za-z0-9_]|$))",
        std::regex_constants::icase
    );
    if (std::regex_search(content, nonFinite))
    {
        stateError("non-finite token in restart field " + pathText(path));
    }
}

struct RequiredArtifact
{
    const char* role;
    std::string path;
    bool counted;
};

std::vector<RequiredArtifact> requiredArtifacts
(
    const ThermalRestartPreflightExpectations& expected
)
{
    const std::string fluidPrefix =
        expected.fluidRegion == fvMesh::defaultRegion
      ? std::string()
      : std::string(expected.fluidRegion.c_str()) + "/";
    std::vector<RequiredArtifact> result
    {
        {"particleRestart", "gpuResidentStrictParticles.dat", true},
        {"epsGPrevRestart", "gpuResidentStrictEpsGPrev.dat", true},
        {"sourceResidualRestart", "gpuResidentStrictSourceResidual.dat", true},
        {"solidTemperature", std::string(expected.solidRegion.c_str()) + "/T", false},
        {"fluidRho", fluidPrefix + "rho", false},
        {"fluidRhoU", fluidPrefix + "rhoU", false},
        {"fluidRhoE", fluidPrefix + "rhoE", false},
        {"fluidU", fluidPrefix + "U", false},
        {"fluidP", fluidPrefix + "p", false},
        {"fluidT", fluidPrefix + "T", false},
        {"fluidEpsilonS", fluidPrefix + "epsilonS", false},
        {"fluidRhoUs", fluidPrefix + "rhoUs", false},
        {"fluidRhoEs", fluidPrefix + "rhoEs", false},
        {"fluidRhoDs", fluidPrefix + "rhoDs", false},
        {"fluidRhoHp", fluidPrefix + "rhoHp", false},
        {"fluidUs", fluidPrefix + "Us", false},
        {"fluidTheta", fluidPrefix + "theta", false},
        {"fluidTp", fluidPrefix + "Tp", false},
        {"fluidDMeanCell", fluidPrefix + "dMeanCell", false}
    };
    if (!expected.auxiliarySolidRegion.empty())
    {
        result.push_back
        (
            {
                "auxiliarySolidTemperature",
                std::string(expected.auxiliarySolidRegion.c_str()) + "/T",
                false
            }
        );
    }
    return result;
}

void validateArtifacts
(
    const fileName& timeDirectory,
    const ParsedManifest& manifest,
    const ThermalRestartPreflightExpectations& expected
)
{
    std::map<std::string, ManifestArtifact> byRole;
    forAll(manifest.artifacts, artifactI)
    {
        const ManifestArtifact& artifact = manifest.artifacts[artifactI];
        const std::string role = artifact.role.c_str();
        if (!byRole.emplace(role, artifact).second)
        {
            stateError("duplicate manifest artifact role '" + role + "'");
        }
    }

    const std::vector<RequiredArtifact> required = requiredArtifacts(expected);
    if (byRole.size() != required.size())
    {
        stateError("thermal restart manifest has missing or unexpected artifacts");
    }

    for (const RequiredArtifact& contract : required)
    {
        const auto found = byRole.find(contract.role);
        if (found == byRole.end())
        {
            stateError
            (
                "thermal restart manifest is missing artifact role '"
              + std::string(contract.role) + "'"
            );
        }
        const ManifestArtifact& artifact = found->second;
        if
        (
            !validRelativePath(artifact.relativePath)
         || pathText(artifact.relativePath) != contract.path
         || !validLowerSha1(artifact.sha1)
         || (contract.counted && artifact.count < 0)
         || (!contract.counted && artifact.count != -1)
        )
        {
            stateError
            (
                "invalid manifest metadata for artifact role '"
              + std::string(contract.role) + "'"
            );
        }

        const fileName path = timeDirectory/artifact.relativePath;
        if (!Foam::isFile(path))
        {
            stateError("missing restart artifact " + pathText(path));
        }
        if (ThermalRestartPreflight::fileSha1(path) != artifact.sha1)
        {
            stateError("restart artifact SHA-1 mismatch for " + pathText(path));
        }

        const std::string role = artifact.role.c_str();
        if (role == "particleRestart")
        {
            validateParticleMirror
            (
                path,
                artifact.count,
                expected.maximumParticleCount,
                expected.fluidCellCount
            );
        }
        else if (role == "epsGPrevRestart")
        {
            validateEpsGPrevMirror(path, artifact.count);
            if
            (
                expected.fluidCellCount >= 0
             && artifact.count != expected.fluidCellCount
            )
            {
                stateError("epsGPrev mirror count does not match fluid mesh cells");
            }
        }
        else if (role == "sourceResidualRestart")
        {
            validateSourceResidualMirror
            (
                path,
                artifact.count,
                expected.fluidFaceCount
            );
        }
        else
        {
            validateNoNonFiniteFieldToken(path);
        }
    }
}

void validateManifestlessInitialFiles
(
    const fileName& timeDirectory,
    const ThermalRestartPreflightExpectations& expected
)
{
    for (const RequiredArtifact& contract : requiredArtifacts(expected))
    {
        const fileName path = timeDirectory/fileName(contract.path);
        if (!contract.counted)
        {
            if (!Foam::isFile(path))
            {
                stateError
                (
                    "manifestless initial state is missing required field "
                  + pathText(path)
                );
            }
            validateNoNonFiniteFieldToken(path);
            continue;
        }

                                                                            
                                                                            
                                                                              
        if (!Foam::isFile(path))
        {
            continue;
        }
        if (std::string(contract.role) == "particleRestart")
        {
            validateParticleMirror
            (
                path,
                -1,
                expected.maximumParticleCount,
                expected.fluidCellCount,
                true
            );
        }
        else if (std::string(contract.role) == "epsGPrevRestart")
        {
            validateEpsGPrevMirror(path, expected.fluidCellCount);
        }
        else if (std::string(contract.role) == "sourceResidualRestart")
        {
            validateSourceResidualMirror(path, -1, expected.fluidFaceCount);
        }
    }
}

bool directoryExists(const fileName& path)
{
    struct stat status;
    return ::stat(path.c_str(), &status) == 0 && S_ISDIR(status.st_mode);
}

bool numericDirectoryTime(const char* name, scalar& value)
{
    if (!name || !*name)
    {
        return false;
    }
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(name, &end);
    if
    (
        errno != 0
     || end == name
     || *end != '\0'
     || !std::isfinite(parsed)
    )
    {
        return false;
    }
    value = scalar(parsed);
    return true;
}

ThermalExchangeRestartState validateDirectoryImpl
(
    const fileName& timeDirectory,
    const ThermalRestartPreflightExpectations& expected,
    const bool scanNewer
);

void rejectNewerUncommittedDirectories
(
    const fileName& timeDirectory,
    const ThermalRestartPreflightExpectations& expected
)
{
    const fileName parent = timeDirectory.path();
    DIR* directory = ::opendir(parent.c_str());
    if (!directory)
    {
        stateError("cannot scan case time directories under " + pathText(parent));
    }

    try
    {
        while (const dirent* item = ::readdir(directory))
        {
            scalar candidateTime = 0;
            if
            (
                !numericDirectoryTime(item->d_name, candidateTime)
             || candidateTime <= expected.completedSimulationTimeS
            )
            {
                continue;
            }

            const fileName candidate = parent/item->d_name;
            if (!directoryExists(candidate))
            {
                continue;
            }
            const fileName manifestPath =
                candidate/ThermalRestartPreflight::manifestObjectName();
            if (!Foam::isFile(manifestPath))
            {
                stateError
                (
                    "newer uncommitted time directory has no thermal manifest: "
                  + pathText(candidate)
                );
            }

            const ParsedManifest parsed = parseManifest
            (
                readDictionaryFile(manifestPath),
                manifestPath
            );
            if (parsed.state.completedSimulationTimeS != candidateTime)
            {
                stateError
                (
                    "newer time directory name does not match its thermal "
                    "manifest simulation time: " + pathText(candidate)
                );
            }
            ThermalRestartPreflightExpectations newerExpected = expected;
            newerExpected.completedTimeIndex = parsed.state.completedTimeIndex;
            newerExpected.completedSimulationTimeS =
                parsed.state.completedSimulationTimeS;
            newerExpected.rejectNewerUncommittedTime = false;
            validateDirectoryImpl(candidate, newerExpected, false);
        }
    }
    catch (...)
    {
        ::closedir(directory);
        throw;
    }
    ::closedir(directory);
}

ThermalExchangeRestartState validateDirectoryImpl
(
    const fileName& timeDirectory,
    const ThermalRestartPreflightExpectations& expected,
    const bool scanNewer
)
{
    if (!directoryExists(timeDirectory))
    {
        stateError("missing restart time directory " + pathText(timeDirectory));
    }
    if
    (
        expected.completedTimeIndex < 0
     || !finiteScalar(expected.completedSimulationTimeS)
     || expected.startTimeIndex < 0
     || !finiteScalar(expected.startTimeS)
     || expected.fluidCellCount < -1
     || expected.fluidFaceCount < -1
     || expected.maximumParticleCount < -1
     || !validLowerSha1(expected.fluidMeshTopologySha1)
     || !validLowerSha1(expected.solidMeshTopologySha1)
     || (!expected.auxiliarySolidRegion.empty()
      && !validLowerSha1(expected.auxiliarySolidMeshTopologySha1))
     || (expected.auxiliarySolidRegion.empty()
      != expected.auxiliarySolidMeshTopologySha1.empty())
     || !validLowerSha1(expected.couplingConfigurationSha1)
     || expected.solidRegion.empty()
    )
    {
        stateError("invalid expected thermal restart preflight contract");
    }

    const fileName manifestPath =
        timeDirectory/ThermalRestartPreflight::manifestObjectName();
    if (!Foam::isFile(manifestPath))
    {
        if
        (
            !expected.allowManifestlessInitialState
         || expected.completedTimeIndex != expected.startTimeIndex
         || expected.completedSimulationTimeS != expected.startTimeS
        )
        {
            stateError
            (
                "missing thermal manifest outside the explicit sequence-zero start state"
            );
        }

        const fileName initialStatePath =
            timeDirectory/ThermalRestartPreflight::stateObjectName();
        const ThermalExchangeRestartState initial =
            ThermalRestartPreflight::stateFromDictionary
            (
                readDictionaryFile(initialStatePath),
                initialStatePath
            );
        ThermalRestartPreflight::validateStateSemantics(initial);
        if
        (
            !initial.initialState
         || initial.exchangeSequence != 0
         || initial.completedTimeIndex != expected.completedTimeIndex
         || initial.completedSimulationTimeS
            != expected.completedSimulationTimeS
         || initial.fluidMeshTopologySha1 != expected.fluidMeshTopologySha1
         || initial.solidMeshTopologySha1 != expected.solidMeshTopologySha1
         || initial.auxiliarySolidMeshTopologySha1
            != expected.auxiliarySolidMeshTopologySha1
         || initial.couplingConfigurationSha1
            != expected.couplingConfigurationSha1
        )
        {
            std::ostringstream detail;
            detail
                << "manifestless state mismatch: "
                << "initial=" << initial.initialState
                << ", sequence=" << initial.exchangeSequence
                << ", completedTimeIndex=" << initial.completedTimeIndex
                << "/" << expected.completedTimeIndex
                << ", completedSimulationTimeS="
                << std::setprecision(17)
                << initial.completedSimulationTimeS
                << "/" << expected.completedSimulationTimeS
                << ", fluidMesh=" << initial.fluidMeshTopologySha1
                << "/" << expected.fluidMeshTopologySha1
                << ", solidMesh=" << initial.solidMeshTopologySha1
                << "/" << expected.solidMeshTopologySha1
                << ", auxiliarySolidMesh="
                << initial.auxiliarySolidMeshTopologySha1
                << "/" << expected.auxiliarySolidMeshTopologySha1
                << ", coupling=" << initial.couplingConfigurationSha1
                << "/" << expected.couplingConfigurationSha1;
            stateError
            (
                detail.str()
            );
        }

        validateManifestlessInitialFiles(timeDirectory, expected);
        if (scanNewer && expected.rejectNewerUncommittedTime)
        {
            rejectNewerUncommittedDirectories(timeDirectory, expected);
        }
        return initial;
    }

    const ParsedManifest manifest = parseManifest
    (
        readDictionaryFile(manifestPath),
        manifestPath
    );
    ThermalRestartPreflight::validateStateSemantics(manifest.state);

    if
    (
        manifest.stateFile != ThermalRestartPreflight::stateObjectName()
     || !validLowerSha1(manifest.stateSha1)
     || manifest.solidFieldName != word("T")
    )
    {
        stateError("invalid fixed thermal state/solid-field manifest metadata");
    }

    const fileName statePath = timeDirectory/manifest.stateFile;
    if (ThermalRestartPreflight::fileSha1(statePath) != manifest.stateSha1)
    {
        stateError("thermalExchangeState SHA-1 mismatch in " + pathText(timeDirectory));
    }
    const ThermalExchangeRestartState state =
        ThermalRestartPreflight::stateFromDictionary
        (
            readDictionaryFile(statePath),
            statePath
        );
    ThermalRestartPreflight::validateStateSemantics(state);
    if (!sameState(state, manifest.state))
    {
        stateError("thermal state and manifest metadata disagree");
    }

    if
    (
        !restartStateTimeNotAfterDirectory
        (
            state.completedSimulationTimeS,
            expected.completedSimulationTimeS
        )
     || state.fluidMeshTopologySha1 != expected.fluidMeshTopologySha1
     || state.solidMeshTopologySha1 != expected.solidMeshTopologySha1
     || state.auxiliarySolidMeshTopologySha1
        != expected.auxiliarySolidMeshTopologySha1
     || state.couplingConfigurationSha1
        != expected.couplingConfigurationSha1
    )
    {
        stateError("stale or topology/configuration-mismatched thermal restart");
    }

    validateArtifacts(timeDirectory, manifest, expected);
    if (scanNewer && expected.rejectNewerUncommittedTime)
    {
        rejectNewerUncommittedDirectories(timeDirectory, expected);
    }
    return state;
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
    if (pendingExchange_)
    {
        stateError("thermal exchange prepare is reentrant while another pair is pending");
    }
    if (!finiteScalar(simulationTimeS) || timeIndex < 0)
    {
        stateError("write event time/index must be finite and non-negative");
    }

    const label committedIndex = committedState_.completedTimeIndex;
    const scalar committedTime = committedState_.completedSimulationTimeS;
    if (timeIndex == committedIndex && simulationTimeS == committedTime)
    {
        return decision;
    }
    if (timeIndex == committedIndex)
    {
        stateError("same completed time index was observed with a different time");
    }
    if (simulationTimeS == committedTime)
    {
        stateError("same completed simulation time was observed with a different index");
    }
    if (timeIndex < committedIndex || simulationTimeS < committedTime)
    {
        stateError("write event moved backwards from the last committed exchange");
    }

    const scalar deltaT = simulationTimeS - committedTime;
    if (!finiteScalar(deltaT) || deltaT <= scalar(0))
    {
        stateError("thermal exchange interval must be finite and strictly positive");
    }
    if (committedState_.exchangeSequence == std::numeric_limits<std::uint64_t>::max())
    {
        stateError("thermal exchange sequence overflow");
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
    if (!pendingExchange_)
    {
        stateError("cannot commit a thermal exchange without a pending prepare");
    }
    ThermalRestartPreflight::validateStateSemantics(completedState);
    if
    (
        completedState.initialState
     || completedState.exchangeSequence != pendingDecision_.exchangeSequence
     || completedState.completedTimeIndex != pendingDecision_.timeIndex
     || completedState.completedSimulationTimeS
        != pendingDecision_.simulationTimeS
     || completedState.previousExchangeSimulationTimeS
        != committedState_.completedSimulationTimeS
     || completedState.fluidMeshTopologySha1
        != committedState_.fluidMeshTopologySha1
     || completedState.solidMeshTopologySha1
        != committedState_.solidMeshTopologySha1
     || completedState.couplingConfigurationSha1
        != committedState_.couplingConfigurationSha1
     ||
        (
            !committedState_.initialState
         && completedState.wallTemperatureSha1UsedForCompletedInterval
            != committedState_.newlyUploadedWallTemperatureSha1
        )
    )
    {
        stateError("committed thermal state does not exactly match the pending exchange");
    }

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

void Foam::gpuThermal::ThermalRestartPreflight::validateStateSemantics
(
    const ThermalExchangeRestartState& state
)
{
    if
    (
        state.formatVersion != 1
     && state.formatVersion != 2
     && state.formatVersion != 3
    )
    {
        stateError("unsupported thermal exchange restart formatVersion");
    }
    if
    (
        state.completedTimeIndex < 0
     || !finiteScalar(state.completedSimulationTimeS)
     || !finiteScalar(state.completedRadiationSimulationTimeS)
     || !finiteScalar(state.previousExchangeSimulationTimeS)
     || !finiteScalar(state.particleContactEnergyJ)
    )
    {
        stateError("thermal exchange restart contains invalid time/index/energy");
    }
    if
    (
        !validLowerSha1(state.fluidMeshTopologySha1)
     || !validLowerSha1(state.solidMeshTopologySha1)
     || (state.formatVersion == 2
      && !validLowerSha1(state.auxiliarySolidMeshTopologySha1))
     || (state.formatVersion >= 3
      && !state.auxiliarySolidMeshTopologySha1.empty()
      && !validLowerSha1(state.auxiliarySolidMeshTopologySha1))
     || !validLowerSha1(state.couplingConfigurationSha1)
     || !validLowerSha1
        (
            state.wallTemperatureSha1UsedForCompletedInterval
        )
     || !validLowerSha1(state.newlyUploadedWallTemperatureSha1)
    )
    {
        stateError("thermal exchange restart contains a non-canonical SHA-1");
    }
    if
    (
        state.formatVersion == 1
     &&
        (
            state.particleContactEnergyJ != scalar(0)
         || state.particleWallLedgerConsumed
         || !state.auxiliarySolidMeshTopologySha1.empty()
        )
    )
    {
        stateError("v1 cannot contain particle-contact or auxiliary-solid state");
    }

    const bool anyComplete =
        state.gasWallLedgerConsumed
     || state.particleWallLedgerConsumed
     || state.particleRadiationApplied
     || state.solidStateUpdated
     || state.wallTemperatureUploaded
     || state.particleMomentsRebuilt;
    const bool allComplete =
        state.gasWallLedgerConsumed
     && (state.formatVersion == 1 || state.particleWallLedgerConsumed)
     && state.particleRadiationApplied
     && state.solidStateUpdated
     && state.wallTemperatureUploaded
     && state.particleMomentsRebuilt;

    if (state.initialState)
    {
        if
        (
            state.exchangeSequence != 0
         || state.previousExchangeSimulationTimeS
            != state.completedSimulationTimeS
         || state.completedRadiationSimulationTimeS
            != state.completedSimulationTimeS
         || anyComplete
        )
        {
            stateError
            (
                "sequence-zero initial state requires equal saved times and all flags false"
            );
        }
    }
    else if
    (
        state.exchangeSequence == 0
     || state.previousExchangeSimulationTimeS
        >= state.completedSimulationTimeS
     || state.completedRadiationSimulationTimeS < scalar(0)
     || state.completedRadiationSimulationTimeS
        > state.completedSimulationTimeS
     || !allComplete
    )
    {
        stateError
        (
            "non-initial committed state requires positive sequence/time interval and all flags true"
        );
    }
}

Foam::gpuThermal::ThermalExchangeRestartState
Foam::gpuThermal::ThermalRestartPreflight::validateRestartDirectory
(
    const fileName& timeDirectory,
    const ThermalRestartPreflightExpectations& expected
)
{
    return validateDirectoryImpl
    (
        timeDirectory,
        expected,
        expected.rejectNewerUncommittedTime
    );
}

Foam::gpuThermal::ThermalExchangeRestartState
Foam::gpuThermal::ThermalRestartPreflight::validateBeforeResidentInitialise
(
    const Time& runTime,
    const fvMesh& fluidMesh,
    const dictionary& couplingDictionary
)
{
    if (Foam::Pstream::parRun())
    {
        stateError("thermal restart preflight requires serial execution");
    }
    const dictionary& control = runTime.controlDict();
    requirePurgeWriteZero(control);
    requireLosslessAsciiRestartPrecision(control);

    const dictionary* coupling = &couplingDictionary;
    if
    (
        couplingDictionary.found("solidThermalCoupling", false, false)
     && couplingDictionary.isDict("solidThermalCoupling")
    )
    {
        coupling = &couplingDictionary.subDict("solidThermalCoupling");
    }
    word solidRegion;
    if (coupling->found("solidRegion", false, false))
    {
        solidRegion = coupling->lookup<word>("solidRegion", false, false);
    }
    else if (coupling->found("primarySolidRegion", false, false))
    {
        solidRegion =
            coupling->lookup<word>("primarySolidRegion", false, false);
    }
    else
    {
        stateError
        (
            "resolved coupling dictionary is missing solidRegion or "
            "primarySolidRegion"
        );
    }

    word auxiliarySolidRegion;
    if (coupling->found("solidSolidInterfaces", false, false))
    {
        const Foam::PtrList<dictionary> interfaces
        (
            coupling->lookup("solidSolidInterfaces", false, false)
        );
        if (interfaces.size() != 1)
        {
            stateError("v2 requires exactly one solid-solid interface");
        }
        const word first
        (
            interfaces[0].lookup<word>("firstRegion", false, false)
        );
        const word second
        (
            interfaces[0].lookup<word>("secondRegion", false, false)
        );
        if (first == solidRegion && second != solidRegion)
        {
            auxiliarySolidRegion = second;
        }
        else if (second == solidRegion && first != solidRegion)
        {
            auxiliarySolidRegion = first;
        }
        else
        {
            stateError("solid-solid interface does not contain primary solid");
        }
    }

    const dictionary* resolvedParticleContract = nullptr;
    if
    (
        couplingDictionary.found("resolvedParticleContract", false, false)
     && couplingDictionary.isDict("resolvedParticleContract")
    )
    {
        resolvedParticleContract =
            &couplingDictionary.subDict("resolvedParticleContract");
    }
    else if
    (
        coupling->found("resolvedParticleContract", false, false)
     && coupling->isDict("resolvedParticleContract")
    )
    {
        resolvedParticleContract = &coupling->subDict("resolvedParticleContract");
    }
    if
    (
        !resolvedParticleContract
     || !resolvedParticleContract->found("particleCapacity", false, false)
    )
    {
        stateError
        (
            "resolved coupling dictionary is missing resolvedParticleContract/particleCapacity"
        );
    }
    const label particleCapacity =
        resolvedParticleContract->lookup<label>
        (
            "particleCapacity",
            false,
            false
        );
    if (particleCapacity < 0)
    {
        stateError("resolved particleCapacity must be non-negative");
    }

    fvMesh solidMesh
    (
        Foam::IOobject
        (
            solidRegion,
            runTime.timeName(),
            runTime,
            Foam::IOobject::MUST_READ,
            Foam::IOobject::NO_WRITE,
            false
        ),
        false
    );

    validateFiniteRestartFields(fluidMesh, solidMesh);
    Foam::autoPtr<fvMesh> auxiliarySolidMesh;
    if (!auxiliarySolidRegion.empty())
    {
        auxiliarySolidMesh.reset
        (
            new fvMesh
            (
                Foam::IOobject
                (
                    auxiliarySolidRegion,
                    runTime.timeName(),
                    runTime,
                    Foam::IOobject::MUST_READ,
                    Foam::IOobject::NO_WRITE,
                    false
                ),
                false
            )
        );
        validateFiniteVolField<Foam::volScalarField>
        (
            auxiliarySolidMesh(), "T"
        );
    }

    ThermalRestartPreflightExpectations expected;
    expected.completedTimeIndex = runTime.timeIndex();
    expected.completedSimulationTimeS = runTime.value();
    expected.fluidMeshTopologySha1 = canonicalMeshTopologySha1(fluidMesh);
    expected.solidMeshTopologySha1 = canonicalMeshTopologySha1(solidMesh);
    if (auxiliarySolidMesh.valid())
    {
        expected.auxiliarySolidMeshTopologySha1 =
            canonicalMeshTopologySha1(auxiliarySolidMesh());
    }
    expected.couplingConfigurationSha1 =
        canonicalConfigurationSha1(couplingDictionary);
    expected.fluidRegion = fluidMesh.name();
    expected.solidRegion = solidRegion;
    expected.auxiliarySolidRegion = auxiliarySolidRegion;
    expected.startTimeIndex = runTime.startTimeIndex();
    expected.startTimeS = control.found("startTime", false, false)
        ? control.lookup<scalar>("startTime", false, false)
        : runTime.startTime().value();
    expected.fluidCellCount = fluidMesh.nCells();
    expected.fluidFaceCount = fluidMesh.nFaces();
    expected.maximumParticleCount = particleCapacity;
    expected.allowManifestlessInitialState =
        control.found("startFrom", false, false)
     && control.lookup<word>("startFrom", false, false) == word("startTime")
     && runTime.timeIndex() == runTime.startTimeIndex()
     && runTime.value() == expected.startTimeS;
    expected.rejectNewerUncommittedTime = true;

    ThermalExchangeRestartState state =
        validateRestartDirectory(runTime.timePath(), expected);

                                                                            
                                                                            
                                                                            
                                                                          
                                                                         
                    
    state.completedTimeIndex = runTime.timeIndex();
    return state;
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
    if
    (
        !controlDictionary.found("purgeWrite", false, false)
     || controlDictionary.lookup<label>("purgeWrite", false, false) != 0
    )
    {
        stateError("thermal coupling requires explicit purgeWrite 0");
    }
}

void Foam::gpuThermal::ThermalRestartPreflight::
requireLosslessAsciiRestartPrecision
(
    const dictionary& controlDictionary
)
{
    if (!controlDictionary.found("writeFormat", false, false))
    {
        stateError("thermal coupling requires explicit writeFormat");
    }
    const word format =
        controlDictionary.lookup<word>("writeFormat", false, false);
    if (format != "ascii")
    {
        return;
    }
    if
    (
        !controlDictionary.found("writePrecision", false, false)
     || controlDictionary.lookup<label>("writePrecision", false, false)
        < static_cast<label>(std::numeric_limits<scalar>::max_digits10)
    )
    {
        stateError
        (
            "ASCII thermal restart requires explicit writePrecision >= "
          + std::to_string(std::numeric_limits<scalar>::max_digits10)
        );
    }
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
    validateStateSemantics(state);
    dictionary result;
    result.add("formatVersion", state.formatVersion);
    result.add("initialState", word(state.initialState ? "true" : "false"));
    result.add("exchangeSequence", Foam::string(std::to_string(state.exchangeSequence)));
    result.add("completedTimeIndex", state.completedTimeIndex);
    result.add("completedSimulationTimeS", state.completedSimulationTimeS);
    if (state.formatVersion >= 3)
    {
        result.add
        (
            "completedRadiationSimulationTimeS",
            state.completedRadiationSimulationTimeS
        );
    }
    result.add
    (
        "previousExchangeSimulationTimeS",
        state.previousExchangeSimulationTimeS
    );
    result.add("fluidMeshTopologySha1", Foam::string(state.fluidMeshTopologySha1));
    result.add("solidMeshTopologySha1", Foam::string(state.solidMeshTopologySha1));
    if (state.formatVersion >= 2)
    {
        result.add
        (
            "auxiliarySolidMeshTopologySha1",
            Foam::string(state.auxiliarySolidMeshTopologySha1)
        );
    }
    result.add
    (
        "couplingConfigurationSha1",
        Foam::string(state.couplingConfigurationSha1)
    );
    result.add
    (
        "wallTemperatureSha1UsedForCompletedInterval",
        Foam::string(state.wallTemperatureSha1UsedForCompletedInterval)
    );
    result.add
    (
        "newlyUploadedWallTemperatureSha1",
        Foam::string(state.newlyUploadedWallTemperatureSha1)
    );
    result.add
    (
        "gasWallLedgerConsumed",
        word(state.gasWallLedgerConsumed ? "true" : "false")
    );
    if (state.formatVersion >= 2)
    {
        result.add
        (
            "particleWallLedgerConsumed",
            word(state.particleWallLedgerConsumed ? "true" : "false")
        );
    }
    result.add
    (
        "particleRadiationApplied",
        word(state.particleRadiationApplied ? "true" : "false")
    );
    result.add
    (
        "solidStateUpdated",
        word(state.solidStateUpdated ? "true" : "false")
    );
    result.add
    (
        "wallTemperatureUploaded",
        word(state.wallTemperatureUploaded ? "true" : "false")
    );
    result.add
    (
        "particleMomentsRebuilt",
        word(state.particleMomentsRebuilt ? "true" : "false")
    );
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
        "formatVersion",
        "initialState",
        "exchangeSequence",
        "completedTimeIndex",
        "completedSimulationTimeS",
        "previousExchangeSimulationTimeS",
        "fluidMeshTopologySha1",
        "solidMeshTopologySha1",
        "couplingConfigurationSha1",
        "wallTemperatureSha1UsedForCompletedInterval",
        "newlyUploadedWallTemperatureSha1",
        "gasWallLedgerConsumed",
        "particleRadiationApplied",
        "solidStateUpdated",
        "wallTemperatureUploaded",
        "particleMomentsRebuilt",
        "particleContactEnergyJ"
    })
    {
        requireEntry(dict, key, sourceName);
    }

    ThermalExchangeRestartState state;
    state.formatVersion = dict.lookup<label>("formatVersion", false, false);
    if (state.formatVersion >= 2)
    {
        requireEntry(dict, "auxiliarySolidMeshTopologySha1", sourceName);
        requireEntry(dict, "particleWallLedgerConsumed", sourceName);
    }
    state.initialState = bool(dict.lookup<Foam::Switch>("initialState", false, false));
    state.exchangeSequence = parseSequence(dict, sourceName);
    state.completedTimeIndex =
        dict.lookup<label>("completedTimeIndex", false, false);
    state.completedSimulationTimeS =
        dict.lookup<scalar>("completedSimulationTimeS", false, false);
    if (state.formatVersion >= 3)
    {
        requireEntry
        (
            dict, "completedRadiationSimulationTimeS", sourceName
        );
        state.completedRadiationSimulationTimeS = dict.lookup<scalar>
        (
            "completedRadiationSimulationTimeS", false, false
        );
    }
    else
    {
        state.completedRadiationSimulationTimeS =
            state.completedSimulationTimeS;
    }
    state.previousExchangeSimulationTimeS =
        dict.lookup<scalar>("previousExchangeSimulationTimeS", false, false);
    state.fluidMeshTopologySha1 = word
    (
        readTextToken(dict, "fluidMeshTopologySha1", sourceName)
    );
    state.solidMeshTopologySha1 = word
    (
        readTextToken(dict, "solidMeshTopologySha1", sourceName)
    );
    if (state.formatVersion >= 2)
    {
        state.auxiliarySolidMeshTopologySha1 = word
        (
            readTextToken
            (
                dict, "auxiliarySolidMeshTopologySha1", sourceName
            )
        );
    }
    state.couplingConfigurationSha1 = word
    (
        readTextToken(dict, "couplingConfigurationSha1", sourceName)
    );
    state.wallTemperatureSha1UsedForCompletedInterval = word
    (
        readTextToken
        (
            dict,
            "wallTemperatureSha1UsedForCompletedInterval",
            sourceName
        )
    );
    state.newlyUploadedWallTemperatureSha1 = word
    (
        readTextToken(dict, "newlyUploadedWallTemperatureSha1", sourceName)
    );
    state.gasWallLedgerConsumed =
        bool(dict.lookup<Foam::Switch>("gasWallLedgerConsumed", false, false));
    if (state.formatVersion >= 2)
    {
        state.particleWallLedgerConsumed = bool
        (
            dict.lookup<Foam::Switch>
            (
                "particleWallLedgerConsumed", false, false
            )
        );
    }
    state.particleRadiationApplied =
        bool(dict.lookup<Foam::Switch>("particleRadiationApplied", false, false));
    state.solidStateUpdated =
        bool(dict.lookup<Foam::Switch>("solidStateUpdated", false, false));
    state.wallTemperatureUploaded =
        bool(dict.lookup<Foam::Switch>("wallTemperatureUploaded", false, false));
    state.particleMomentsRebuilt =
        bool(dict.lookup<Foam::Switch>("particleMomentsRebuilt", false, false));
    state.particleContactEnergyJ =
        dict.lookup<scalar>("particleContactEnergyJ", false, false);
    return state;
}

void Foam::gpuThermal::ThermalRestartPreflight::commitManifestAtomically
(
    const fileName& temporaryManifest,
    const fileName& finalManifest
)
{
    struct stat temporaryStatus;
    struct stat finalStatus;
    errno = 0;
    const bool temporaryIsRegular =
        ::lstat(temporaryManifest.c_str(), &temporaryStatus) == 0
     && S_ISREG(temporaryStatus.st_mode);
    errno = 0;
    const int finalStatusResult = ::lstat(finalManifest.c_str(), &finalStatus);
    const bool finalPathAbsent =
        finalStatusResult != 0 && errno == ENOENT;
    if
    (
        temporaryManifest == finalManifest
     || temporaryManifest.path() != finalManifest.path()
     || !temporaryIsRegular
     || !finalPathAbsent
    )
    {
        stateError
        (
            "atomic manifest commit requires a new temp file in the final directory"
        );
    }
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
