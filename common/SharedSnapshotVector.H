#pragma once
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>
namespace ugkp {
// Large published diagnostic arrays share immutable storage between state
// snapshots. A mutable access detaches; clearing a shared snapshot copies no data.
template<class T> class SharedSnapshotVector {
    std::shared_ptr<std::vector<T>> storage_;
public:
    SharedSnapshotVector()=default;
    SharedSnapshotVector(std::initializer_list<T> x):storage_(std::make_shared<std::vector<T>>(x)){}
    SharedSnapshotVector& operator=(std::vector<T> x){storage_=std::make_shared<std::vector<T>>(std::move(x));return *this;}
    SharedSnapshotVector& operator=(std::initializer_list<T> x){storage_=std::make_shared<std::vector<T>>(x);return *this;}
    const std::vector<T>& values()const {static const std::vector<T> empty;return storage_?*storage_:empty;}
    std::vector<T>& edit(){
        if(!storage_)storage_=std::make_shared<std::vector<T>>();
        else if(!storage_.unique())storage_=std::make_shared<std::vector<T>>(*storage_);
        return *storage_;
    }
    void clear(){storage_.reset();}
    std::size_t size()const{return values().size();}
    bool empty()const{return values().empty();}
    const T* data()const{return values().data();}
    T* data(){return edit().data();}
    const T& operator[](std::size_t i)const{return values()[i];}
    T& operator[](std::size_t i){return edit()[i];}
    auto begin()const{return values().begin();}
    auto end()const{return values().end();}
    auto begin(){return edit().begin();}
    auto end(){return edit().end();}
    void push_back(const T& x){edit().push_back(x);}
    void push_back(T&& x){edit().push_back(std::move(x));}
};
// Preserve the existing vector checkpoint wire format and numeric test visitors.
template<class Archive,class T> void visit(Archive& a,const SharedSnapshotVector<T>& x){a(x.values());}
template<class Archive,class T> void visit(Archive& a,SharedSnapshotVector<T>& x){a(x.edit());}
}
