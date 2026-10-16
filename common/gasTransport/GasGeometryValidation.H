#pragma once
#include "GasStateView.H"
#include "GasCapabilities.H"

namespace ugkwp
{
// Bridge only unambiguous legacy statuses here. Composition, geometry,
// thermodynamic and step-limit codes mix physical errors with residuals and
// need classification at their producer, not a blanket fatal conversion.
template<class State>
UGKWP_GAS_HD inline bool gasRecordTransportDiagnostic(const State& s,
    int entity, bool face, GasTransportCode code, unsigned site=2600,
    unsigned operation=26, unsigned quantity=ugkp::diagnostics::Quantity::Unspecified)
{
    using namespace ugkp::diagnostics;
    if(!s.diagnostics)return false;
    if(code!=GasTransportCode::NonFiniteState && code!=GasTransportCode::InvalidStorage)return false;
    const bool invalidIndex=entity<0 || entity>=(face?s.nFaces:s.nCells);
    auto record=makeRecord(invalidIndex?Code::InvalidIndex:
        (code==GasTransportCode::NonFiniteState?Code::NonFinite:Code::MissingOperator),
        site,operation,0,face?EntityKind::Face:EntityKind::Cell,entity,-1,
        invalidIndex?entity:int(code),0,0,invalidIndex?Quantity::Index:quantity);
    recordFatal(s.diagnostics,record);
    return true;
}
template<class State>
UGKWP_GAS_HD inline bool gasRecordCellFailure(const State& s,int c,GasTransportCode code)
{
    const bool recorded=gasRecordTransportDiagnostic(s,c,false,code);
    if constexpr (GasStateTraits<State>::speciesCount > 0)
        if(s.gasSpecies.cellStatus && c>=0 && c<s.nCells)
        { if(s.gasSpecies.cellStatus[c]!=int(GasTransportCode::InvalidStorage))s.gasSpecies.cellStatus[c]=int(code); return true; }
    return recorded;
}
template<class State>
UGKWP_GAS_HD inline bool gasRecordFaceFailure(const State& s,int f,GasTransportCode code, unsigned site=2600,
    unsigned operation=26, unsigned quantity=ugkp::diagnostics::Quantity::Unspecified)
{
    const bool recorded=gasRecordTransportDiagnostic(s,f,true,code,site,operation,quantity);
    if constexpr (GasStateTraits<State>::speciesCount > 0)
        if(s.gasSpecies.faceStatus && f>=0 && f<s.nFaces)
        { if(s.gasSpecies.faceStatus[f]!=int(GasTransportCode::InvalidStorage))s.gasSpecies.faceStatus[f]=int(code); return true; }
    return recorded;
}
template<class State>
UGKWP_GAS_HD inline int gasCellFailure(const State& s,int c)
{
    if constexpr (GasStateTraits<State>::speciesCount > 0)
        if(s.gasSpecies.cellStatus)
            return c>=0 && c<s.nCells ? s.gasSpecies.cellStatus[c] : int(GasTransportCode::InvalidStorage);
    return 0;
}
template<class State>
UGKWP_GAS_HD inline int gasFaceFailure(const State& s,int f)
{
    if constexpr (GasStateTraits<State>::speciesCount > 0)
        if(s.gasSpecies.faceStatus)
            return f>=0 && f<s.nFaces ? s.gasSpecies.faceStatus[f] : int(GasTransportCode::InvalidStorage);
    return 0;
}
template<class State,class Time,class Real>
UGKWP_GAS_HD inline bool gasGeometryFaceFrame(const State& s,int f,Time dt,GasFaceFrame<Real>& frame,bool* evaluated=nullptr)
{
    if(evaluated)*evaluated=false;
    if constexpr (GasGeometryCapability<State>::value)
    {
        (void)dt;
        if(f<0 || f>=s.nFaces || !s.gasGeometry.faceSweptVolume || !s.Sfx || !s.Sfy || !s.Sfz)return false;
        const bool valid=makeGasFaceFrame(s.Sfx[f],s.Sfy[f],s.Sfz[f],
            s.gasGeometry.faceSweptVolume[f],s.gasGeometry.interval,frame);
        if(evaluated)*evaluated=true;
        return valid;
    }
    return false;
}
template<class State,class Time,class Real>
UGKWP_GAS_HD inline bool gasGeometryCellVolumes(const State& s,int c,Time dt,Real& oldVolume,Real& newVolume,bool* evaluated=nullptr)
{
    if(evaluated)*evaluated=false;
    if constexpr (GasGeometryCapability<State>::value)
    {
        (void)dt;
        const auto& g=s.gasGeometry;
        if(c<0 || c>=s.nCells || !g.oldVolume || !g.newVolume)return false;
        const Real oldValue=g.oldVolume[c],newValue=g.newVolume[c];
        oldVolume=oldValue;newVolume=newValue;
        if(evaluated)*evaluated=true;
        return gasGeometryDetail::finite(oldValue) && gasGeometryDetail::finite(newValue);
    }
    return false;
}
}
