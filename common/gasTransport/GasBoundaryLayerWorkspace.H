#pragma once
#include <cstddef>
#include <algorithm>
#include "../gasWall/WallModelTypes.H"
namespace ugkwp {
struct BoundaryLayerWorkspaceSizing
{
    int slots=0;
    std::size_t bytes=0,budget=0;
};
// Initialization-only policy. One warp per SM is a starting configuration,
// not an occupancy or throughput guarantee; explicit slots permit tuning.
inline BoundaryLayerWorkspaceSizing resolveBoundaryLayerWorkspace(
    int faces,int requestedSlots,int multiprocessors,std::size_t freeBytes,std::size_t bytesPerWorkspace)
{
    BoundaryLayerWorkspaceSizing result;
    if(faces<=0 || requestedSlots<0 || requestedSlots>4096 || bytesPerWorkspace==0)return result;
    if(requestedSlots==0)
    {
        if(multiprocessors<=0)return result;
        result.budget=std::min(freeBytes/8,std::size_t(1)<<30);
        const std::size_t desired=std::size_t(multiprocessors)*32;
        result.slots=int(std::min({std::size_t(faces),desired,std::size_t(4096),result.budget/bytesPerWorkspace}));
    }
    else
    {
        result.budget=freeBytes;
        const int slots=std::min(faces,requestedSlots);
        if(std::size_t(slots)>freeBytes/bytesPerWorkspace)return result;
        result.slots=slots;
    }
    result.bytes=std::size_t(result.slots)*bytesPerWorkspace;
    return result;
}
inline int gasBoundaryLayerWorkspaceCapacity(int nodes)
{return nodes<=0 || nodes>128?0:(nodes<=24?24:(nodes<=48?48:128));}
template<class Real,int Ns> inline std::size_t gasBoundaryLayerWorkspaceBytes(int nodes)
{
    switch(gasBoundaryLayerWorkspaceCapacity(nodes))
    {
        case 24:return sizeof(gaswall::WallWorkspace<Real,Ns,24>);
        case 48:return sizeof(gaswall::WallWorkspace<Real,Ns,48>);
        case 128:return sizeof(gaswall::WallWorkspace<Real,Ns,128>);
        default:return 0;
    }
}
}
