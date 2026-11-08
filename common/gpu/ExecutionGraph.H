#ifndef UGKP_GPU_EXECUTION_GRAPH_H
#define UGKP_GPU_EXECUTION_GRAPH_H
#include <vector>
#include <type_traits>
struct CUgraph_st;
struct CUgraphNode_st;
namespace ugkp { namespace gpu {
// Exact captured argument identity. Device contents may change in place;
// addresses, extents and by-value controls must match before graph replay.
struct ExecutionGraphSignature {
    std::vector<unsigned char> bytes;
    template<class T> void add(const T& value){
        static_assert(std::is_trivially_copyable<T>::value,"graph argument must be trivially copyable");
        const auto* first=reinterpret_cast<const unsigned char*>(&value);
        bytes.insert(bytes.end(),first,first+sizeof(T));
    }
    void append(const ExecutionGraphSignature& value){bytes.insert(bytes.end(),value.bytes.begin(),value.bytes.end());}
    bool operator==(const ExecutionGraphSignature& other)const{return bytes==other.bytes;}
};
// CUDA handle declarations without a toolkit dependency for host adapters.
using ExecutionGraph = CUgraph_st*;
using ExecutionGraphNode = CUgraphNode_st*;
}}
#endif
