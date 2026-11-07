#pragma once
#if defined(__CUDACC__)
#define UGKP_FACE_PARTITION_HD __host__ __device__
#else
#define UGKP_FACE_PARTITION_HD
#endif
namespace ugkwp { namespace gaswall {
template<class R>UGKP_FACE_PARTITION_HD inline R faceAreaShare(R area,R total){return total>R(0)?area/total:R(0);}
// The row index retains original face order. This partition is geometric;
// chemical activity does not remove a face from its owner's exposed area.
template<class R,class Area>UGKP_FACE_PARTITION_HD inline R faceAreaRow(
    const int* indices,int begin,int end,Area area){
    R total=0;for(int i=begin;i<end;++i)total+=area(indices[i]);return total;
}
}}
#undef UGKP_FACE_PARTITION_HD
