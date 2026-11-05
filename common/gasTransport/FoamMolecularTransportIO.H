#pragma once
#include "fvCFD.H"
#include "MolecularTransportIO.H"
namespace Foam {
// Expand includes and dictionary substitutions identically in all applications.
// Keep the case/region registry so OpenFOAM resolves relative includes normally.
inline ugkwp::MolecularTransportConfiguration readFoamMolecularTransport(
    const Time& time,const objectRegistry& registry,
    const std::vector<std::string>& speciesNames)
{
    IOobject object("gasTransportProperties",time.constant(),registry,
        IOobject::READ_IF_PRESENT,IOobject::NO_WRITE,false);
    if(!isFile(object.objectPath(false)))return {};
    IOdictionary dict(object);
    OStringStream expanded;
    expanded.precision(std::numeric_limits<double>::max_digits10);
    static_cast<const dictionary&>(dict).write(expanded,false);
    return ugkwp::parseMolecularTransport(expanded.str(),speciesNames);
}
}
