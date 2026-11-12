#ifndef UGKP_FOAM_THERMAL_DISCRETIZATION_H
#define UGKP_FOAM_THERMAL_DISCRETIZATION_H
#include "FoamSolverScheme.H"
#include "fvMesh.H"
#include "IStringStream.H"
namespace ugkp { namespace solverControls {
// Numerical selection is a startup snapshot; framework geometry/flux
// registration remains owned by fvMesh. Native selection syntax is retained.
class ThermalDiscretization {
    Foam::dictionary values_;
public:
    explicit ThermalDiscretization(const Foam::fvMesh& mesh) {
        Foam::IStringStream defaults(
            "laplacianSchemes { default Gauss linear corrected; } "
            "gradSchemes { default Gauss linear; } "
            "interpolationSchemes { default linear; } "
            "ddtSchemes { default Euler; }");
        values_=Foam::dictionary(defaults);
        values_.merge(startupScheme(mesh.time()).section("solidDiscretization",mesh.name()));
        Foam::Info<<"SOLID_DISCRETIZATION_STARTUP region="<<mesh.name()
            <<" source=constant/scheme effective="<<values_<<Foam::nl;
    }
    Foam::ITstream& selection(const Foam::word& category,const Foam::word& name) const {
        const auto& entries=values_.subDict(category);
        auto& stream=entries.lookup(entries.found(name)?name:Foam::word("default"));
        stream.rewind();return stream;
    }
    bool steady() const {
        auto& stream=values_.subDict("ddtSchemes").lookup("default");
        stream.rewind();const Foam::word kind(stream);stream.rewind();
        return kind=="steadyState";
    }
};
inline const ThermalDiscretization& thermalDiscretization(const Foam::fvMesh& mesh) {
    static std::map<std::string,std::unique_ptr<const ThermalDiscretization>> cache;
    const std::string key=std::string(mesh.time().path().c_str())+":"+mesh.name().c_str();
    auto it=cache.find(key);
    if(it==cache.end())it=cache.emplace(key,std::unique_ptr<const ThermalDiscretization>(new ThermalDiscretization(mesh))).first;
    return *it->second;
}
}}
#endif
