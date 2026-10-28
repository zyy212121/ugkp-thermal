#pragma once
#include "GasModelIO.H"
#include <algorithm>
#include "MolecularViscosity.H"
namespace ugkwp {
// Molecular transport is independent of calorics, chemistry and solid properties.
// All applications read constant/gasTransportProperties with this parser.
struct MolecularTransportConfiguration {
    MolecularViscosityModel model=MolecularViscosityModel::Constant;
    std::vector<std::string> speciesNames;
    std::vector<SutherlandSpecies<double>> species;
    std::vector<double> massRatioQuarter,wilkeDenominator;
    MolecularViscosityView<double> coefficients;
    std::vector<PowerLawDiffusivity<double>> diffusion;
    std::uint64_t hash=0; // zero preserves the absent/const identity
    MolecularViscosityView<double> view() const {
        auto v=coefficients;v.model=model;v.count=int(speciesNames.empty()?species.size():speciesNames.size());
        v.species=species.data();v.massRatioQuarter=massRatioQuarter.data();v.wilkeDenominator=wilkeDenominator.data();
        v.diffusion=diffusion.empty()?nullptr:diffusion.data();v.diffusionCount=int(diffusion.size());return v;
    }
};
inline MolecularTransportConfiguration parseMolecularTransport(const std::string& text,
    const std::vector<std::string>& expected={}) {
    using namespace gasModelIoDetail;
    MolecularTransportConfiguration out;if(text.empty())return out;
    Tokens tokens(text);const auto dict=tokens.dictionary();
    const auto tag=word(dict,"viscosityModel");
    const auto optionalWord=[&](const char* key,const char* fallback){return dict.find(key)==dict.end()?std::string(fallback):word(dict,key);};
    const auto pr=optionalWord("conductivityModel","constantPrandtl");
    const auto dm=optionalWord("diffusivityModel","const");
    if(tag=="const"&&pr=="constantPrandtl"&&dm=="const")return out;
    if(tag!="const"&&tag!="sutherland"&&tag!="massFractionSutherland")throw std::runtime_error("unknown viscosityModel: "+tag);
    if(pr!="constantPrandtl"&&pr!="massFractionPrandtl")throw std::runtime_error("unknown conductivityModel: "+pr);
    if(dm!="const"&&dm!="powerLaw")throw std::runtime_error("unknown diffusivityModel: "+dm);
    if(expected.empty()&&(tag=="massFractionSutherland"||pr=="massFractionPrandtl"||dm=="powerLaw"))
        throw std::runtime_error("composition and species diffusion models require an active transported mixture");
    out.speciesNames=list(dict,"species");
    if(out.speciesNames.empty()||(!expected.empty()&&out.speciesNames!=expected)
        ||(expected.empty()&&out.speciesNames.size()!=1))
        throw std::runtime_error("gasTransportProperties: species must match the transported species order, or contain one legacy gas");
    if(tag=="sutherland"){
    out.model=MolecularViscosityModel::SutherlandWilke;
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
    }
    const auto selected=[&](const auto& data){const auto name=word(data,"species");
        const auto it=std::find(out.speciesNames.begin(),out.speciesNames.end(),name);
        if(it==out.speciesNames.end())throw std::runtime_error("transport composition species not transported: "+name);
        return int(it-out.speciesNames.begin());};
    const auto polynomial=[&](const auto& data,const char* name,double* a){
        const auto values=list(data,name);if(values.empty()||values.size()>3)throw std::runtime_error(std::string(name)+": require 1 to 3 ascending polynomial coefficients");
        for(std::size_t i=0;i<values.size();++i){std::size_t used=0;a[i]=std::stod(values[i],&used);if(used!=values[i].size()||!gasFinite(a[i]))throw std::runtime_error(std::string(name)+": nonfinite or invalid coefficient");}
    };
    const auto positivePolynomial=[&](const double* a,const char* name,bool zeroAllowed){
        double minimum=std::min(a[0],a[0]+a[1]+a[2]);
        if(a[2]>0){double x=-a[1]/(2*a[2]);if(x>0&&x<1)minimum=std::min(minimum,transportPolynomial(a,x));}
        if(!gasFinite(minimum)||(zeroAllowed?minimum<0:minimum<=0))throw std::runtime_error(std::string(name)+": must be positive over mass fractions [0,1]");
    };
    if(tag=="massFractionSutherland"||pr!="constantPrandtl"||dm!="const"){
        if(!out.hash)out.hash=speciesIdentity(out.speciesNames);
        hashWord(out.hash,"local-molecular-transport-v1");hashWord(out.hash,tag);hashWord(out.hash,pr);hashWord(out.hash,dm);
    }
    if(tag=="massFractionSutherland"){
        const auto& d=subdict(dict,"massFractionSutherland");auto& v=out.coefficients;
        out.model=MolecularViscosityModel::MassFractionSutherland;v.compositionSpecies=selected(d);v.compositionTRef=scalar(d,"TRef");
        if(!gasFinite(v.compositionTRef)||v.compositionTRef<=0)throw std::runtime_error("massFractionSutherland: TRef must be positive");
        polynomial(d,"muRefCoefficients",v.muCoefficients);polynomial(d,"SCoefficients",v.sCoefficients);
        positivePolynomial(v.muCoefficients,"muRef",false);positivePolynomial(v.sCoefficients,"S",true);
        hashWord(out.hash,out.speciesNames[v.compositionSpecies]);hashScalar(out.hash,v.compositionTRef);
        for(double a:v.muCoefficients)hashScalar(out.hash,a);
        for(double a:v.sCoefficients)hashScalar(out.hash,a);
    }
    if(pr=="massFractionPrandtl"){
        const auto& d=subdict(dict,"massFractionPrandtl");auto& v=out.coefficients;v.prandtlSpecies=selected(d);
        polynomial(d,"coefficients",v.prandtlCoefficients);positivePolynomial(v.prandtlCoefficients,"Pr",false);
        hashWord(out.hash,out.speciesNames[v.prandtlSpecies]);for(double a:v.prandtlCoefficients)hashScalar(out.hash,a);
    }
    if(dm=="powerLaw")for(const auto& name:out.speciesNames){
        const auto& d=subdict(subdict(dict,"powerLaw"),name);PowerLawDiffusivity<double> a;
        a.DRef=scalar(d,"DRef");a.TRef=scalar(d,"TRef");a.pRef=scalar(d,"pRef");a.temperatureExponent=scalar(d,"temperatureExponent");a.pressureExponent=scalar(d,"pressureExponent");
        const double values[]={a.DRef,a.TRef,a.pRef,a.temperatureExponent,a.pressureExponent};
        for(double x:values){if(!gasFinite(x))throw std::runtime_error("powerLaw: require finite coefficients");hashScalar(out.hash,x);}
        if(a.DRef<0||a.TRef<=0||a.pRef<=0)throw std::runtime_error("powerLaw: DRef >= 0, TRef and pRef > 0 required");
        out.diffusion.push_back(a);
    }
    return out;
}
template<class Species> inline void validateTransportMolarMasses(const MolecularTransportConfiguration& transport,
    const std::vector<Species>& species) {
    if(transport.model!=MolecularViscosityModel::SutherlandWilke)return;
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
