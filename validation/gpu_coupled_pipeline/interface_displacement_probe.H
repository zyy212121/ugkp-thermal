#pragma once
#include "gpu/InterfacePointDisplacement.H"
struct InterfaceDisplacementProbe {std::vector<ugkp::gpu::geometry::Point> displacement;int changed=0;};
InterfaceDisplacementProbe probeInterfaceDisplacement(const ugkp::gpu::geometry::Topology&,int,
    const std::vector<int>&,const std::vector<int>&,const std::vector<ugkp::gpu::geometry::Point>&,
    const std::vector<ugkp::gpu::geometry::Point>&,const std::vector<ugkp::gpu::geometry::Point>&,
    const std::vector<ugkp::gpu::geometry::Point>&,const std::vector<double>&,
    const std::vector<double>&,const std::vector<double>&,bool);
