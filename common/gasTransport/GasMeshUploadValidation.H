#pragma once
#include "../Diagnostics.H"
namespace ugkwp {
// Host upload owns the allocation length. Validate before any device copy or
// kernel can consume cellPlaneStart/count; subtraction avoids integer overflow.
inline bool gasMeshStencilUploadValid(int cells,int planes,const int* starts,
    const int* counts,ugkp::diagnostics::Record& failure)
{
    using namespace ugkp::diagnostics;
    if(cells<0 || planes<0 || (cells && (!starts || !counts))) {
        failure=makeRecord(Code::InvalidIndex,30006,26,0,EntityKind::None,-1,-1,
            cells<0?cells:planes,0,0,Quantity::Index);return false;
    }
    for(int c=0;c<cells;++c) {
        const int start=starts[c],count=counts[c];
        if(count<0 || (count>0 && (start<0 || start>planes || count>planes-start))) {
            failure=makeRecord(Code::InvalidIndex,30005,26,0,EntityKind::Cell,c,
                count<0?1:0,count<0?count:start,planes,count,Quantity::Index);
            return false;
        }
    }
    return true;
}
}
