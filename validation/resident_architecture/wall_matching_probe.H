#pragma once
#include <array>
#include <vector>
namespace wallMatchingProbe {
struct Mesh {
    std::vector<std::array<double,3>> points,centres;
    std::vector<double> volumes;
    std::vector<int> offsets,indices,owner,neighbour,walls;
    std::vector<unsigned char> physicalWall,constrained,gasCell;
};
struct Result {
    int firstInvalidRow=-1,invalidFace=-1;
    struct Row {int wallFace=-1,owner=-1,donor=-1,failure=-1;
        std::array<double,3> origin,normal,tangent1,tangent2,matchingPoint,firstMomentVector;
        double ownerDistance,matchingDistance,maximumOwnerDistance,matchingDonorDistance,volume,geometryVolume,firstMoment;
    };
    std::vector<Row> row;
};
std::vector<Result> evaluate(const Mesh&,bool);
}
