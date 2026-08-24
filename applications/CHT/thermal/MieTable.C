#include "MieTable.H"
#include "SHA1.H"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace Foam
{
namespace gpuThermal
{
namespace
{

struct Node
{
    scalar Qabs, Qsca, Qext;
    scalarField phi;
};

struct Bracket
{
    std::size_t lo;
    scalar w;
};

[[noreturn]] void fail(const fileName& file, const std::string& message)
{
    throw MieTableError("MIE table '" + std::string(file.c_str()) + "': " + message);
}

std::string readFile(const fileName& file)
{
    std::ifstream in(file.c_str(), std::ios::binary);
    if (!in) fail(file, "cannot open file");
    std::ostringstream out;
    out << in.rdbuf();
    if (!in.good() && !in.eof()) fail(file, "read failure");
    std::string s = out.str();
    std::string clean;
    clean.reserve(s.size());
    for (std::size_t i=0; i<s.size(); ++i)
    {
        if (s[i] == '\r')
        {
            if (i+1 < s.size() && s[i+1] == '\n') ++i;
            clean.push_back('\n');
        }
        else clean.push_back(s[i]);
    }
    return clean;
}

std::map<std::string, std::string> readHeader
(
    const fileName& file,
    const std::string& content,
    const std::size_t payloadBegin
)
{
    std::istringstream lines(content.substr(0, payloadBegin));
    std::string line;
    if (!std::getline(lines, line) || line != "MIE_TABLE")
        fail(file, "magic must be MIE_TABLE");

    const std::set<std::string> allowed
    {
        "formatVersion", "units", "interpolation", "generatorSha256",
        "generationArgs", "diameter", "temperature", "mu", "wavelength"
    };
    std::map<std::string, std::string> result;
    while (std::getline(lines, line))
    {
        if (line.empty()) continue;
        const std::size_t p = line.find_first_of(" \t");
        if (p == std::string::npos) fail(file, "malformed header line: " + line);
        const std::string key = line.substr(0, p);
        if (!allowed.count(key)) fail(file, "unknown header field: " + key);
        if (!result.emplace(key, line.substr(p+1)).second)
            fail(file, "duplicate header field: " + key);
    }
    for (const std::string& key : allowed)
        if (!result.count(key)) fail(file, "missing header field: " + key);
    if (result["formatVersion"] != "1") fail(file, "unsupported formatVersion");
    if (result["units"] != "diameter_m temperature_K wavelength_m")
        fail(file, "unexpected units");
    if (result["interpolation"] != "logDiameter_linearTemperature_linearMu")
        fail(file, "unexpected interpolation contract");
    return result;
}

scalar number(const fileName& file, const std::string& label, const std::string& word)
{
    std::istringstream in(word);
    scalar value;
    char extra;
    if (!(in >> value) || (in >> extra) || !std::isfinite(value))
        fail(file, label + " is not a finite scalar: " + word);
    return value;
}

std::size_t indexValue(const fileName& file, const std::string& label, const std::string& word)
{
    std::istringstream in(word);
    std::size_t value;
    char extra;
    if (!(in >> value) || (in >> extra)) fail(file, label + " is not an index");
    return value;
}

void increasing(const fileName& file, const std::string& label, const scalarField& x)
{
    if (x.size() < 2) fail(file, label + " requires at least two values");
    forAll(x, i)
    {
        if (!std::isfinite(x[i])) fail(file, label + " contains non-finite value");
        if (i && !(x[i] > x[i-1])) fail(file, label + " is not strictly increasing");
    }
}

Bracket linearBracket(const scalarField& axis, scalar value)
{
    if (value <= axis[0]) return {0, 0};
    const std::size_t last = axis.size()-1;
    if (value >= axis[last]) return {last-1, 1};
    const scalar* it = std::upper_bound(axis.begin(), axis.end(), value);
    const std::size_t hi = it-axis.begin();
    return {hi-1, (value-axis[hi-1])/(axis[hi]-axis[hi-1])};
}

Bracket logBracket(const scalarField& axis, scalar value)
{
    if (!(value > 0) || !std::isfinite(value)) value = axis[0];
    Bracket b = linearBracket(axis, value);
    const scalar a = std::log(axis[b.lo]);
    const scalar z = std::log(axis[b.lo+1]);
    const scalar v = std::log(std::max(axis[b.lo], std::min(axis[b.lo+1], value)));
    b.w = (v-a)/(z-a);
    return b;
}

scalar blend(const scalar a, const scalar b, const scalar w)
{
    return a + w*(b-a);
}

}             

class MieTableData
{
public:
    fileName file;
    scalarField T, D, mu, band;
    std::vector<Node> nodes;
    std::size_t node(const std::size_t d, const std::size_t t) const
    { return d*static_cast<std::size_t>(T.size()) + t; }
};

MieTable::MieTable(const fileName& tableFile)
{
    const std::string content = readFile(tableFile);
    const std::size_t payloadBegin = content.find("Tvals ");
    if (payloadBegin == std::string::npos || (payloadBegin && content[payloadBegin-1] != '\n'))
        fail(tableFile, "missing Tvals payload");
    readHeader(tableFile, content, payloadBegin);

    const std::string endMarker = "END_MIE_TABLE\n";
    const std::size_t end = content.find(endMarker, payloadBegin);
    if (end == std::string::npos) fail(tableFile, "missing END_MIE_TABLE");
    const std::size_t payloadEnd = end + endMarker.size();
    const std::string checksumPrefix = "payloadSha1 ";
    if (content.compare(payloadEnd, checksumPrefix.size(), checksumPrefix) != 0)
        fail(tableFile, "missing payloadSha1");
    const std::size_t checksumEnd = content.find('\n', payloadEnd);
    if (checksumEnd == std::string::npos || checksumEnd+1 != content.size())
        fail(tableFile, "trailing data after payloadSha1");
    const std::string expected = content.substr
    (
        payloadEnd + checksumPrefix.size(),
        checksumEnd - payloadEnd - checksumPrefix.size()
    );
    const std::string payload = content.substr(payloadBegin, payloadEnd-payloadBegin);
    const std::string actual = Foam::SHA1(payload).digest().str();
    if (actual != expected) fail(tableFile, "payloadSha1 mismatch");

    std::istringstream in(payload);
    auto field = [&](const std::string& expectedName) -> scalarField
    {
        std::string line, name;
        if (!std::getline(in, line)) fail(tableFile, "missing " + expectedName);
        std::istringstream words(line);
        words >> name;
        if (name != expectedName) fail(tableFile, "expected " + expectedName);
        std::vector<scalar> values;
        std::string word;
        while (words >> word) values.push_back(number(tableFile, expectedName, word));
        scalarField result(values.size());
        forAll(result, i) result[i] = values[i];
        return result;
    };

    std::shared_ptr<MieTableData> data(new MieTableData);
    data->file = tableFile;
    data->T = field("Tvals");
    data->D = field("Dvals");
    data->mu = field("Muvals");
    data->band = field("PlanckBandFractions");
    increasing(tableFile, "Tvals", data->T);
    increasing(tableFile, "Dvals", data->D);
    increasing(tableFile, "Muvals", data->mu);
    if (data->band.size() != data->T.size()) fail(tableFile, "PlanckBandFractions size mismatch");
    forAll(data->band, i)
        if (!std::isfinite(data->band[i]) || data->band[i] < 0 || data->band[i] > 1)
            fail(tableFile, "PlanckBandFractions must be in [0,1]");

    std::string line;
    if (!std::getline(in, line) || line != "DATA") fail(tableFile, "missing DATA");
    const std::size_t count = data->D.size()*data->T.size();
    data->nodes.reserve(count);
    for (std::size_t d=0; d<data->D.size(); ++d)
    for (std::size_t t=0; t<data->T.size(); ++t)
    {
        if (!std::getline(in, line)) fail(tableFile, "missing NODE");
        std::istringstream words(line);
        std::string tag, ds, ts, qa, qs, qe, extra;
        if (!(words >> tag >> ds >> ts >> qa >> qs >> qe) || (words >> extra) || tag != "NODE")
            fail(tableFile, "malformed NODE");
        if (indexValue(tableFile, "NODE d", ds) != d || indexValue(tableFile, "NODE t", ts) != t)
            fail(tableFile, "NODE order mismatch");
        Node n;
        n.Qabs = number(tableFile, "Qabs", qa);
        n.Qsca = number(tableFile, "Qsca", qs);
        n.Qext = number(tableFile, "Qext", qe);
        if (n.Qabs < 0 || n.Qsca < 0 || n.Qext < 0) fail(tableFile, "negative efficiency");
        const scalar qScale = std::max(scalar(1), std::max(n.Qext, n.Qabs+n.Qsca));
        if (std::abs(n.Qext-n.Qabs-n.Qsca) > 1e-10*qScale) fail(tableFile, "Qext identity failed");

        if (!std::getline(in, line)) fail(tableFile, "missing PHI");
        std::istringstream phiWords(line);
        phiWords >> tag;
        if (tag != "PHI") fail(tableFile, "malformed PHI");
        n.phi.setSize(data->mu.size());
        forAll(n.phi, i)
        {
            if (!(phiWords >> n.phi[i]) || !std::isfinite(n.phi[i]) || n.phi[i] < 0)
                fail(tableFile, "invalid PHI value");
        }
        if (phiWords >> extra) fail(tableFile, "too many PHI values");
        scalar integral = 0;
        for (label i=1; i<n.phi.size(); ++i)
            integral += 0.5*(n.phi[i-1]+n.phi[i])*(data->mu[i]-data->mu[i-1]);
        if (std::abs(integral-2) > 1e-6) fail(tableFile, "PHI normalization failed");
        data->nodes.push_back(std::move(n));
    }
    if (!std::getline(in, line) || line != "END_MIE_TABLE") fail(tableFile, "missing terminator");
    if (std::getline(in, line)) fail(tableFile, "tokens after terminator");
    data_ = data;
}

MieOpticalSample MieTable::query(const scalar diameter, const scalar temperature) const
{
    const Bracket d = logBracket(data_->D, diameter);
    const Bracket t = linearBracket(data_->T, temperature);
    const Node& a = data_->nodes[data_->node(d.lo,t.lo)];
    const Node& b = data_->nodes[data_->node(d.lo,t.lo+1)];
    const Node& c = data_->nodes[data_->node(d.lo+1,t.lo)];
    const Node& e = data_->nodes[data_->node(d.lo+1,t.lo+1)];
    const auto bi = [&](scalar av, scalar bv, scalar cv, scalar ev)
    { return blend(blend(av,bv,t.w), blend(cv,ev,t.w), d.w); };
    return {bi(a.Qabs,b.Qabs,c.Qabs,e.Qabs), bi(a.Qsca,b.Qsca,c.Qsca,e.Qsca),
        bi(a.Qext,b.Qext,c.Qext,e.Qext), planckBandFraction(temperature)};
}

scalar MieTable::phase(const scalar diameter, const scalar temperature, const scalar mu) const
{
    const Bracket d = logBracket(data_->D, diameter);
    const Bracket t = linearBracket(data_->T, temperature);
    const Bracket m = linearBracket(data_->mu, mu);
    const Node& a = data_->nodes[data_->node(d.lo,t.lo)];
    const Node& b = data_->nodes[data_->node(d.lo,t.lo+1)];
    const Node& c = data_->nodes[data_->node(d.lo+1,t.lo)];
    const Node& e = data_->nodes[data_->node(d.lo+1,t.lo+1)];
    const auto at = [&](const Node& n){ return blend(n.phi[m.lo],n.phi[m.lo+1],m.w); };
    const auto bi = [&](scalar av, scalar bv, scalar cv, scalar ev)
    { return blend(blend(av,bv,t.w), blend(cv,ev,t.w), d.w); };
    const scalar q = bi(a.Qsca,b.Qsca,c.Qsca,e.Qsca);
    if (q <= 0) return 1;
    return bi(a.Qsca*at(a),b.Qsca*at(b),c.Qsca*at(c),e.Qsca*at(e))/q;
}

scalar MieTable::planckBandFraction(const scalar temperature) const
{
    const Bracket t = linearBracket(data_->T, temperature);
    return blend(data_->band[t.lo], data_->band[t.lo+1], t.w);
}

bool MieTable::sharesDataWith(const MieTable& other) const noexcept { return data_.get()==other.data_.get(); }
scalar MieTable::minDiameterM() const { return data_->D[0]; }
scalar MieTable::maxDiameterM() const { return data_->D[data_->D.size()-1]; }
scalar MieTable::minTemperatureK() const { return data_->T[0]; }
scalar MieTable::maxTemperatureK() const { return data_->T[data_->T.size()-1]; }
const scalarField& MieTable::muValues() const { return data_->mu; }

}                        
}                  
