#pragma once
#include <array>
#include <vector>
namespace wallPolygonProbe {
struct Triangle {int a,b,c;};
struct Result {std::array<double,3> normal{},centre{};double scale=0;int valid=0;std::vector<Triangle> triangles;};
std::vector<std::vector<Result>> evaluate(const std::vector<std::array<double,3>>&,const std::vector<int>&,const std::vector<int>&);
}
