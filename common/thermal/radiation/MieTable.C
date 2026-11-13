#include "MieTable.H"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace Foam
{
namespace gpuThermal
{
using MieReal = ugkp::gpu::RadiationReal;
using MieField = std::vector<MieReal>;
namespace
{

struct Node
{
    MieReal Qabs, Qsca, Qext;
    MieField phi;
};

struct Bracket
{
    std::size_t lo;
    MieReal w;
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

MieReal number(const fileName& file, const std::string& label, const std::string& word)
{
    std::istringstream in(word);
    MieReal value;
    char extra;
    if ((in >> value) && !(in >> extra)) return value;



    char* end = nullptr;
    value = sizeof(MieReal)==sizeof(float) ? MieReal(std::strtof(word.c_str(), &end)) : MieReal(std::strtod(word.c_str(), &end));
    if (end == word.c_str() || *end != '\0')
        fail(file, label + " is not a MieReal: " + word);
    return value;
}

void requireBracketNodes(const fileName& file, const std::string& label, const MieField& x)
{
    if (x.size() < 2) fail(file, label + " requires at least two values");
}

Bracket linearBracket(const MieField& axis, MieReal value)
{
    if (value <= axis[0]) return {0, 0};
    const std::size_t last = axis.size()-1;
    if (value >= axis[last]) return {last-1, 1};


    std::size_t first = 0;
    std::size_t count = axis.size();
    while (count)
    {
        const std::size_t step = count/2;
        const std::size_t middle = first + step;
        if (!(value < axis[middle]))
        {
            first = middle + 1;
            count -= step + 1;
        }
        else
        {
            count = step;
        }
    }

    const std::size_t hi = std::max(std::size_t(1), std::min(first, last));
    return {hi-1, (value-axis[hi-1])/(axis[hi]-axis[hi-1])};
}

Bracket logBracket(const MieField& axis, MieReal value)
{
    if (!(value > 0) || !std::isfinite(value)) value = axis[0];
    Bracket b = linearBracket(axis, value);
    const MieReal a = std::log(axis[b.lo]);
    const MieReal z = std::log(axis[b.lo+1]);
    const MieReal v = std::log(std::max(axis[b.lo], std::min(axis[b.lo+1], value)));
    b.w = (v-a)/(z-a);
    return b;
}

MieReal blend(const MieReal a, const MieReal b, const MieReal w)
{
    return a + w*(b-a);
}

}             

class MieTableData
{
public:
    fileName file;
    MieField T, D, mu, band;
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

    const std::string endMarker = "END_MIE_TABLE\n";
    const std::size_t end = content.find(endMarker, payloadBegin);
    if (end == std::string::npos) fail(tableFile, "missing END_MIE_TABLE");
    const std::size_t payloadEnd = end + endMarker.size();
    const std::string payload = content.substr(payloadBegin, payloadEnd-payloadBegin);

    std::istringstream in(payload);
    auto field = [&](const std::string& expectedName) -> MieField
    {
        std::string line, name;
        if (!std::getline(in, line)) fail(tableFile, "missing " + expectedName);
        std::istringstream words(line);
        words >> name;
        if (name != expectedName) fail(tableFile, "expected " + expectedName);
        std::vector<MieReal> values;
        std::string word;
        while (words >> word) values.push_back(number(tableFile, expectedName, word));
        MieField result(values.size());
        forAll(result, i) result[i] = values[i];
        return result;
    };

    std::shared_ptr<MieTableData> data(new MieTableData);
    data->file = tableFile;
    data->T = field("Tvals");
    data->D = field("Dvals");
    data->mu = field("Muvals");
    data->band = field("PlanckBandFractions");
    requireBracketNodes(tableFile, "Tvals", data->T);
    requireBracketNodes(tableFile, "Dvals", data->D);
    requireBracketNodes(tableFile, "Muvals", data->mu);
    if (data->band.size() < data->T.size()) fail(tableFile, "PlanckBandFractions size mismatch");

    std::string line;
    if (!std::getline(in, line) || line != "DATA") fail(tableFile, "missing DATA");
    const std::size_t diameterCount = static_cast<std::size_t>(data->D.size());
    const std::size_t temperatureCount = static_cast<std::size_t>(data->T.size());
    const std::size_t count = diameterCount*temperatureCount;
    data->nodes.reserve(count);
    for (std::size_t d=0; d<diameterCount; ++d)
    for (std::size_t t=0; t<temperatureCount; ++t)
    {
        if (!std::getline(in, line)) fail(tableFile, "missing NODE");
        std::istringstream words(line);
        std::string tag, ds, ts, qa, qs, qe;
        if (!(words >> tag >> ds >> ts >> qa >> qs >> qe) || tag != "NODE")
            fail(tableFile, "malformed NODE");
        Node n;
        n.Qabs = number(tableFile, "Qabs", qa);
        n.Qsca = number(tableFile, "Qsca", qs);
        n.Qext = number(tableFile, "Qext", qe);

        if (!std::getline(in, line)) fail(tableFile, "missing PHI");
        std::istringstream phiWords(line);
        phiWords >> tag;
        if (tag != "PHI") fail(tableFile, "malformed PHI");
        n.phi.resize(data->mu.size());
        forAll(n.phi, i)
        {
            const std::streampos position = phiWords.tellg();
            if (!(phiWords >> n.phi[i]))
            {
                phiWords.clear();
                phiWords.seekg(position);
                std::string word;
                if (!(phiWords >> word)) fail(tableFile, "invalid PHI value");
                n.phi[i] = number(tableFile, "PHI value", word);
            }
        }
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
    const auto bi = [&](MieReal av, MieReal bv, MieReal cv, MieReal ev)
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
    const auto bi = [&](MieReal av, MieReal bv, MieReal cv, MieReal ev)
    { return blend(blend(av,bv,t.w), blend(cv,ev,t.w), d.w); };
    const MieReal q = bi(a.Qsca,b.Qsca,c.Qsca,e.Qsca);
    if (q == 0) return 1;
    return bi(a.Qsca*at(a),b.Qsca*at(b),c.Qsca*at(c),e.Qsca*at(e))/q;
}

scalar MieTable::planckBandFraction(const scalar temperature) const
{
    const Bracket t = linearBracket(data_->T, temperature);
    return blend(data_->band[t.lo], data_->band[t.lo+1], t.w);
}

ugkp::gpu::MieOpticalTable MieTable::exportGpuTable() const
{
    ugkp::gpu::MieOpticalTable table;
    table.temperature.assign(data_->T.begin(),data_->T.end());
    table.diameter.assign(data_->D.begin(),data_->D.end());
    table.mu.assign(data_->mu.begin(),data_->mu.end());
    table.band.assign(data_->band.begin(),data_->band.end());
    for(const auto& node:data_->nodes){
        table.absorption.push_back(node.Qabs);table.scattering.push_back(node.Qsca);table.extinction.push_back(node.Qext);
        table.phase.insert(table.phase.end(),node.phi.begin(),node.phi.end());
    }
    return table;
}

bool MieTable::sharesDataWith(const MieTable& other) const noexcept { return data_.get()==other.data_.get(); }
scalar MieTable::minDiameterM() const { return data_->D[0]; }
scalar MieTable::maxDiameterM() const { return data_->D[data_->D.size()-1]; }
scalar MieTable::minTemperatureK() const { return data_->T[0]; }
scalar MieTable::maxTemperatureK() const { return data_->T[data_->T.size()-1]; }
scalarField MieTable::muValues() const {
    scalarField result(data_->mu.size());
    forAll(result,i)result[i]=data_->mu[i];
    return result;
}

}                        
}                  
