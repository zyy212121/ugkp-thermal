#ifndef CHMT_CORE_THERMALBOUNDARY_H
#define CHMT_CORE_THERMALBOUNDARY_H
#include "core/HostState.H"
#include <string>
#include <stdexcept>
namespace chmt {
inline ThermalBoundaryKind importedThermalBoundary(const std::string& type,BoundaryKind kind,bool gas) {
    if(kind==BoundaryKind::Periodic) {
        if(type=="cyclic")return ThermalBoundaryKind::ZeroGradient;
    } else if(kind==BoundaryKind::Empty) {
        if(type=="empty")return ThermalBoundaryKind::ZeroGradient;
    } else if(kind==BoundaryKind::Interface) {
        if(type=="zeroGradient")return ThermalBoundaryKind::ZeroGradient;
    } else if(type=="fixedValue" || type=="zeroGradient") {
        if(gas && ((kind==BoundaryKind::Inlet && type!="fixedValue")
            || (kind==BoundaryKind::Outlet && type!="zeroGradient")))
            throw std::runtime_error("gas Inlet requires fixedValue T; Outlet requires zeroGradient T");
        return type=="fixedValue"?ThermalBoundaryKind::FixedValue:ThermalBoundaryKind::ZeroGradient;
    }
    throw std::runtime_error("unsupported thermal patch operator "+type+" for selected boundary kind");
}
inline bool validateThermalBoundaries(const HostMesh& mesh,std::string& error) {
    if(!mesh.thermalBoundary.empty() && mesh.thermalBoundary.size()!=mesh.owner.size()) {
        error="thermal boundary policy size differs from mesh faces";return false;
    }
    for(std::size_t f=0;f<mesh.thermalBoundary.size();++f) {
        const auto policy=mesh.thermalBoundary[f];
        if(policy!=ThermalBoundaryKind::ZeroGradient && policy!=ThermalBoundaryKind::FixedValue) {
            error="invalid thermal boundary policy";return false;
        }
        if(policy==ThermalBoundaryKind::FixedValue) {
            if(mesh.boundaryKind.size()!=mesh.owner.size() || mesh.boundaryPrimitive.size()!=mesh.owner.size()) {
                error="fixed thermal boundary missing kind/primitive arrays";return false;
            }
            const auto kind=mesh.boundaryKind[f];
            if(kind==BoundaryKind::Internal || kind==BoundaryKind::Periodic || kind==BoundaryKind::Empty || kind==BoundaryKind::Interface) {
                error="fixed thermal boundary conflicts with topology/interface policy";return false;
            }
            const Real T=mesh.boundaryPrimitive[f].temperature;
            if(!finite(T) || T<=0){error="fixed thermal boundary requires positive finite kelvin temperature";return false;}
        }
    }
    return true;
}
}
#endif
