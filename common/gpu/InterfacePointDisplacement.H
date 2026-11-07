#ifndef UGKP_GPU_INTERFACE_POINT_DISPLACEMENT_H
#define UGKP_GPU_INTERFACE_POINT_DISPLACEMENT_H
#include "PolyhedralGeometry.H"
#include <cstddef>
namespace ugkp { namespace gpu { namespace geometry {
// A borrowed scalar member of either a contiguous field or an owner record.
// Stride describes storage, not a physical model or an execution parameter.
struct InterfaceThicknessField {const double* first=nullptr;std::size_t stride=sizeof(double);};
struct InterfacePointDisplacementInput {
    const Point *oldSolid=nullptr,*solidEndpoint=nullptr,*solidArea=nullptr,*oldGas=nullptr;
    InterfaceThicknessField pointThickness,previousFilmThickness,filmThickness;
};
struct InterfacePointDisplacementView {
    const Point* displacement=nullptr;
    const int* changed=nullptr;
};
// Canonical conformal correspondence is initialized by the mesh owner. Surface
// normals retain original face order at shared vertices. The unchanged-offset
// predicate is evaluated on device before a consuming motion solve, so an
// unchanged film cannot acquire artificial motion through rounded X+n*delta.
class ResidentInterfacePointDisplacement {
    struct Implementation;std::unique_ptr<Implementation> data_;
public:
    ResidentInterfacePointDisplacement(const Topology& solid,int solidPoints,int gasPoints,
        const std::vector<int>& solidFaces,const std::vector<int>& solidToGas);
    ~ResidentInterfacePointDisplacement();
    InterfacePointDisplacementView view()const;
    void evaluate(const InterfacePointDisplacementInput*,void* stream=nullptr);
};
}}}
#endif
