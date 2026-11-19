#ifndef UGKP_GPU_ORDERED_UPWIND_PLAN_H
#define UGKP_GPU_ORDERED_UPWIND_PLAN_H
#include "RadiationPrecision.H"
#include <algorithm>
#include <vector>
namespace ugkp { namespace gpu {



struct OrderedUpwindPlan {
    int cells=0;
    bool allUpstreamReady=false;
    std::vector<int> offsets,neighbour,subtract,face,residualOrder,levelOffsets,levelCells;
    std::vector<RadiationReal> flux;
    void build(int count,const std::vector<int>& owner,const std::vector<int>& adjacent,
        const std::vector<int>& cellOffsets,const std::vector<int>& cellFaces,const std::vector<RadiationReal>& orientedFlux,
        bool orderByFlow=false){
        cells=count;offsets.assign(1,0);neighbour.clear();subtract.clear();face.clear();flux.clear();residualOrder.clear();
        std::vector<int> level(count,0);
        for(int cell=0;cell<count;++cell){
            const int begin=neighbour.size();
            for(int i=cellOffsets[cell];i<cellOffsets[cell+1];++i){
                const int f=cellFaces[i];if(f>=int(adjacent.size()))continue;
                const RadiationReal value=orientedFlux[f];int upstream=-1;bool negative=false;
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
        allUpstreamReady=false;
        if(orderByFlow){
            std::vector<int> pending(count);
            levelCells.clear();
            levelOffsets.assign(1,0);
            for(int cell=0;cell<count;++cell){
                pending[cell]=offsets[cell+1]-offsets[cell];
                if(!pending[cell])levelCells.push_back(cell);
            }
            for(int begin=0;begin<int(levelCells.size());){
                const int end=levelCells.size();
                levelOffsets.push_back(end);
                for(int row=begin;row<end;++row){
                    const int cell=levelCells[row];
                    for(int i=cellOffsets[cell];i<cellOffsets[cell+1];++i){
                        const int f=cellFaces[i];
                        if(f>=int(adjacent.size()))continue;
                        const bool sends=cell==owner[f]?orientedFlux[f]>0:orientedFlux[f]<0;
                        if(!sends)continue;
                        const int downstream=cell==owner[f]?adjacent[f]:owner[f];
                        if(!--pending[downstream])levelCells.push_back(downstream);
                    }
                }
                begin=end;
            }
            allUpstreamReady=int(levelCells.size())==count;
            if(allUpstreamReady)return;
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
