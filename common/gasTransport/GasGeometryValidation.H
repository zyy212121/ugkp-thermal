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
        { s.gasSpecies.cellStatus[c]=int(code); return true; }
    return false;
}
template<class State>
UGKWP_GAS_HD inline bool gasRecordFaceFailure(const State& s,int f,GasTransportCode code)
{
    if constexpr (GasStateTraits<State>::speciesCount > 0)
        if(s.gasSpecies.faceStatus && f>=0 && f<s.nFaces)
        { s.gasSpecies.faceStatus[f]=int(code); return true; }
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
template<class State,class Time>
UGKWP_GAS_HD inline bool gasGeometryIntervalValid(const State& s,Time dt)
{
    if constexpr (GasGeometryCapability<State>::value)
    {
        using Real=typename std::remove_pointer<decltype(s.rho)>::type;
        const auto& g=s.gasGeometry;
        return gasGeometryDetail::finite(dt) && dt>Time(0)
            && gasGeometryDetail::finite(g.interval) && g.interval>Real(0)
            && gasGeometryDetail::absolute(Time(g.interval)-dt)
                <= Time(64)*std::numeric_limits<Real>::epsilon()*gasGeometryDetail::absolute(dt)
            && g.oldVolume && g.newVolume && (s.nFaces==0 || g.faceSweptVolume)
            && gasGeometryDetail::finite(g.absoluteGeometryTolerance) && g.absoluteGeometryTolerance>=Real(0)
            && gasGeometryDetail::finite(g.relativeGeometryTolerance) && g.relativeGeometryTolerance>=Real(0);
    }
    return false;
}
template<class State,class Time,class Real>
UGKWP_GAS_HD inline bool gasGeometryFaceFrame(const State& s,int f,Time dt,GasFaceFrame<Real>& frame)
{
    if constexpr (GasGeometryCapability<State>::value)
    {
        if(!gasGeometryIntervalValid(s,dt) || f<0 || f>=s.nFaces
            || s.faceOwner[f]<0 || s.faceOwner[f]>=s.nCells
            || !makeGasFaceFrame(s.Sfx[f],s.Sfy[f],s.Sfz[f],s.gasGeometry.faceSweptVolume[f],s.gasGeometry.interval,frame))return false;
        return gasGeometryDetail::equal(s.magSf[f],frame.area);
    }
    return false;
}
template<class State,class Time,class Real>
UGKWP_GAS_HD inline bool gasGeometryCellVolumes(const State& s,int c,Time dt,Real& oldVolume,Real& newVolume)
{
    if constexpr (GasGeometryCapability<State>::value)
    {
        if(!gasGeometryIntervalValid(s,dt) || c<0 || c>=s.nCells)return false;
        const auto& g=s.gasGeometry;
        Real swept=Real(0);
        const int start=s.cellPlaneStart[c],count=s.cellPlaneCount[c];
        if(start<0 || count<0)return false;
        for(int i=0;i<count;++i)
        {
            const int f=s.cellFaceId[start+i];
            if(f<0 || f>=s.nFaces)return false;
            const Real sign=s.faceOwner[f]==c?Real(1):s.faceNeighbour[f]==c?Real(-1):Real(0);
            if(sign==Real(0) || !gasGeometryDetail::finite(g.faceSweptVolume[f]))return false;
            swept+=sign*g.faceSweptVolume[f];
        }
        if(!validateGasEulerGeometry(g.oldVolume[c],g.newVolume[c],swept,
            g.absoluteGeometryTolerance,g.relativeGeometryTolerance))return false;
        oldVolume=g.oldVolume[c];newVolume=g.newVolume[c];return true;
    }
    return false;
}
}
