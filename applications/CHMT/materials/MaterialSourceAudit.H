#pragma once
#include "coupling/IntervalContracts.H"
namespace chmt {



class MaterialSourceAudit {
    const IntervalHistory& history_;
    std::vector<std::size_t> selected_;
    std::size_t next_=0;
public:
    MaterialSourceAudit(const IntervalHistory& history,int budget):history_(history){
        const std::size_t n=history.records().size();
        const auto count=std::min(n?n-1:0,budget>0?std::size_t(budget):std::size_t(0));
        selected_.reserve(count);


        for(std::size_t i=1;i<=count;++i)
            selected_.push_back((n/(count+1))*i+((n%(count+1))*i)/(count+1));
    }
    std::size_t population()const{return history_.records().empty()?0:history_.records().size()-1;}
    void samplesInside(WindowTime begin,WindowTime end,std::vector<std::size_t>& output){
        output.clear();
        while(next_<selected_.size()){
            const auto index=selected_[next_];const auto time=history_.records()[index].localBegin;
            if(time>=end)break;
            ++next_;


            if(time>begin)output.push_back(index);
        }
    }
};
}
