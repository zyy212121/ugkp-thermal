#ifndef UGKP_GPU_ORDERED_UPWIND_PLAN_H
#define UGKP_GPU_ORDERED_UPWIND_PLAN_H
#include <algorithm>
#include <vector>
namespace ugkp { namespace gpu {
// Dependency levels reproduce the existing cell-index Gauss-Seidel sweep.
// Lower-index upstream cells use this sweep; higher indices use its snapshot.
// Face order within each row is retained. No geometrical sweep approximation.
struct OrderedUpwindPlan {
    int cells=0;
    std::vector<int> offsets,neighbour,subtract,face,residualOrder,levelOffsets,levelCells;
    std::vector<double> flux;
    void build(int count,const std::vector<int>& owner,const std::vector<int>& adjacent,
        const std::vector<int>& cellOffsets,const std::vector<int>& cellFaces,const std::vector<double>& orientedFlux){
        cells=count;offsets.assign(1,0);neighbour.clear();subtract.clear();face.clear();flux.clear();residualOrder.clear();
        std::vector<int> level(count,0);
        for(int cell=0;cell<count;++cell){
            const int begin=neighbour.size();
            for(int i=cellOffsets[cell];i<cellOffsets[cell+1];++i){
                const int f=cellFaces[i];if(f>=int(adjacent.size()))continue;
                const double value=orientedFlux[f];int upstream=-1;bool negative=false;
                if(cell==owner[f]&&value<0){upstream=adjacent[f];negative=true;}
                else if(cell==adjacent[f]&&value>0)upstream=owner[f];
                if(upstream<0)continue;
                neighbour.push_back(upstream);subtract.push_back(negative);face.push_back(f);flux.push_back(value);
                if(upstream<cell)level[cell]=std::max(level[cell],level[upstream]+1);
            }
            const int end=neighbour.size();offsets.push_back(end);
            std::vector<int> order;for(int i=begin;i<end;++i)order.push_back(i);
            std::sort(order.begin(),order.end(),[&](int a,int b){return face[a]<face[b];});
            residualOrder.insert(residualOrder.end(),order.begin(),order.end());
        }
        const int levels=count?*std::max_element(level.begin(),level.end())+1:0;
        std::vector<std::vector<int>> groups(levels);
        for(int cell=0;cell<count;++cell)groups[level[cell]].push_back(cell);
        levelOffsets.assign(1,0);levelCells.clear();
        for(const auto& group:groups){levelCells.insert(levelCells.end(),group.begin(),group.end());levelOffsets.push_back(levelCells.size());}
    }
};
}}
#endif
