#pragma once
#include "GasModelIO.H"
#include "MolecularViscosity.H"
namespace ugkwp {
// Transport is independent of calorics, chemistry, solid properties and D.
// All applications read constant/gasTransportProperties with this parser.
struct MolecularTransportConfiguration {
    MolecularViscosityModel model=MolecularViscosityModel::Constant;
    std::vector<std::string> speciesNames;
    std::vector<SutherlandSpecies<double>> species;
    std::vector<double> massRatioQuarter,wilkeDenominator;
    std::uint64_t hash=0; // zero preserves the absent/const identity
    MolecularViscosityView<double> view() const {
        return {model,int(species.size()),species.data(),massRatioQuarter.data(),wilkeDenominator.data()};
    }
};
inline MolecularTransportConfiguration parseMolecularTransport(const std::string& text,
    const std::vector<std::string>& expected={}) {
    using namespace gasModelIoDetail;
    MolecularTransportConfiguration out;if(text.empty())return out;
    Tokens tokens(text);const auto dict=tokens.dictionary();
    const auto tag=word(dict,"viscosityModel");
    if(tag=="const")return out;
    if(tag!="sutherland")throw std::runtime_error("gasTransportProperties: viscosityModel must be const or sutherland");
    out.model=MolecularViscosityModel::SutherlandWilke;
    out.speciesNames=list(dict,"species");
    if(out.speciesNames.empty()||(!expected.empty()&&out.speciesNames!=expected)
        ||(expected.empty()&&out.speciesNames.size()!=1))
        throw std::runtime_error("gasTransportProperties: species must match the transported species order, or contain one legacy gas");
    const auto& entries=subdict(dict,"sutherland");
    out.hash=speciesIdentity(out.speciesNames);hashWord(out.hash,"sutherland-wilke-v1");
    for(const auto& name:out.speciesNames){
        const auto& data=subdict(entries,name);SutherlandSpecies<double> s;
        s.muRef=scalar(data,"muRef");s.TRef=scalar(data,"TRef");s.S=scalar(data,"S");s.molarMass=scalar(data,"molarMass");
        if(!gasFinite(s.muRef)||!(s.muRef>0)||!gasFinite(s.TRef)||!(s.TRef>0)
            ||!gasFinite(s.S)||s.S<0||!gasFinite(s.molarMass)||!(s.molarMass>0))
            throw std::runtime_error("gasTransportProperties: require finite muRef,TRef,molarMass > 0 and S >= 0 for "+name);
        out.species.push_back(s);hashScalar(out.hash,s.muRef);hashScalar(out.hash,s.TRef);
        hashScalar(out.hash,s.S);hashScalar(out.hash,s.molarMass);
    }
    for(const auto& a:out.species)for(const auto& b:out.species){
        out.massRatioQuarter.push_back(std::sqrt(std::sqrt(b.molarMass/a.molarMass)));
        out.wilkeDenominator.push_back(std::sqrt(8*(1+a.molarMass/b.molarMass)));
    }
    return out;
}
template<class Species> inline void validateTransportMolarMasses(const MolecularTransportConfiguration& transport,
    const std::vector<Species>& species) {
    if(transport.model==MolecularViscosityModel::Constant)return;
    if(transport.species.size()!=species.size())
        throw std::runtime_error("molecular transport and thermodynamic species counts differ");
    for(std::size_t i=0;i<species.size();++i)
        if(std::abs(transport.species[i].molarMass-species[i].molarMass)
            >1e-12*species[i].molarMass)
            throw std::runtime_error("gasTransportProperties: molarMass differs from gas calorics for "+transport.speciesNames[i]);
}
inline MolecularTransportConfiguration readMolecularTransport(const std::string& path,
    const std::vector<std::string>& species={}) {
    std::ifstream input(path);
    if(!input){struct stat info;if(::stat(path.c_str(),&info)!=0&&errno==ENOENT)return {};
        throw std::runtime_error("cannot read molecular transport: "+path);}
    const std::string text((std::istreambuf_iterator<char>(input)),{});
    if(input.bad())throw std::runtime_error("failed reading molecular transport: "+path);
    return parseMolecularTransport(text,species);
}
} // namespace ugkwp
