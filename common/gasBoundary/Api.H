#pragma once
#include "Definition.H"
extern "C" int ugkwpGpuResidentStrictConfigureBoundary
(
    void*,int,const ugkp::boundary::Face*,int,const ugkp::boundary::Function*,
    int,const ugkp::boundary::Term*
);
