#pragma once
#include "gasTransport/GasGeometryValidation.H"
#include "gasTransport/MixtureThermo.H"
// Diagnose topology at the producer, before any indexed physical read.
template<class GasState>
__device__ bool gasStencilIndexFailure(const GasState& s,int c,int index,unsigned site)
{
    using namespace ugkp::diagnostics;
    if(s.diagnostics)recordFatal(s.diagnostics,makeRecord(Code::InvalidIndex,site,26,0,
        EntityKind::Cell,c,-1,index,0,0,Quantity::Index));
    ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage);
    return false;
}
template<class GasState>
__device__ bool gasStencilFaceValid(const GasState& s,int c,int f)
{
    if(f<0 || f>=s.nFaces)return gasStencilIndexFailure(s,c,f,30000);
    if(!s.faceOwner || !gasCoupledFaceStorageAvailable(s,f))
    {ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage);return false;}
    const int own=s.faceOwner[f],nei=coupledFaceNeighbour(s,f);
    if(own<0 || own>=s.nCells)return gasStencilIndexFailure(s,c,own,30001);
    if(nei < -1 || nei>=s.nCells || ((f<s.nInternalFaces || isPeriodicFace(s,f)) && nei<0))
        return gasStencilIndexFailure(s,c,nei,30002);
    if(c!=own && c!=nei)return gasStencilIndexFailure(s,c,f,30003);
    return true;
}
template<class GasState,class Scalar>
__device__ bool gasProducerScalarValid(const GasState& s,int c,Scalar x,unsigned site,bool temperature=false)
{
    if(ugkwp::gasFinite(x) && (!temperature || x>=Scalar(0)))return true;
    using namespace ugkp::diagnostics;
    if(s.diagnostics)recordFatal(s.diagnostics,makeRecord(!ugkwp::gasFinite(x)?Code::NonFinite:Code::InvalidPhysicalState,
        site,26,0,EntityKind::Cell,c,-1,double(x),0,0,temperature?Quantity::Temperature:Quantity::Unspecified));
    ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::NonFiniteState);
    return false;
}

template<class GasState>
__device__ bool gasStencilRangeValid(const GasState& s,int c,int start,int count)
{
    if(count<0 || (count>0 && start<0))return gasStencilIndexFailure(s,c,count<0?count:start,30004);
    if(count>0 && !s.cellFaceId)
    {ugkwp::gasRecordCellFailure(s,c,ugkwp::GasTransportCode::InvalidStorage);return false;}
    return true;
}
