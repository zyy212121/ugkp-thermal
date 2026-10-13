#pragma once
#include "GasStateView.H"
#include "GasCapabilities.H"

namespace ugkwp
{
// Geometry failures share the transactional status channel, but never access
// partial-density, species-flux, or mixture-thermodynamic storage in legacy mode.
template<class State>
UGKWP_GAS_HD inline bool gasRecordCellFailure(const State& s,int c,GasTransportCode code)
{
    if constexpr (GasStateTraits<State>::speciesCount > 0)
        if(s.gasSpecies.cellStatus && c>=0 && c<s.nCells)
        { if(s.gasSpecies.cellStatus[c]!=int(GasTransportCode::InvalidStorage))s.gasSpecies.cellStatus[c]=int(code); return true; }
    return false;
}
template<class State>
UGKWP_GAS_HD inline bool gasRecordFaceFailure(const State& s,int f,GasTransportCode code)
{
    if constexpr (GasStateTraits<State>::speciesCount > 0)
        if(s.gasSpecies.faceStatus && f>=0 && f<s.nFaces)
        { if(s.gasSpecies.faceStatus[f]!=int(GasTransportCode::InvalidStorage))s.gasSpecies.faceStatus[f]=int(code); return true; }
    return false;
}
template<class State>
UGKWP_GAS_HD inline int gasCellFailure(const State& s,int c)
{
    if constexpr (GasStateTraits<State>::speciesCount > 0)
        if(s.gasSpecies.cellStatus) return s.gasSpecies.cellStatus[c];
    return 0;
}
template<class State>
UGKWP_GAS_HD inline int gasFaceFailure(const State& s,int f)
{
    if constexpr (GasStateTraits<State>::speciesCount > 0)
        if(s.gasSpecies.faceStatus) return s.gasSpecies.faceStatus[f];
    return 0;
}
template<class State,class Time,class Real>
UGKWP_GAS_HD inline bool gasGeometryFaceFrame(const State& s,int f,Time dt,GasFaceFrame<Real>& frame,bool* evaluated=nullptr)
{
    if(evaluated)*evaluated=false;
    if constexpr (GasGeometryCapability<State>::value)
    {
        (void)dt;
        if(!s.gasGeometry.faceSweptVolume || !s.Sfx || !s.Sfy || !s.Sfz)return false;
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
        if(!g.oldVolume || !g.newVolume)return false;
        const Real oldValue=g.oldVolume[c],newValue=g.newVolume[c];
        oldVolume=oldValue;newVolume=newValue;
        if(evaluated)*evaluated=true;
        return gasGeometryDetail::finite(oldValue) && gasGeometryDetail::finite(newValue);
    }
    return false;
}
}
