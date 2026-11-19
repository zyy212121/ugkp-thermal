#pragma once
#include "Types.H"
#include "../../../common/SharedSnapshotVector.H"
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>

namespace chmt {
class AcceptedLedger {
public:
    using ChunkVisitor=std::function<void(const ExchangePacket*,std::size_t)>;
    using Reader=std::function<void(const ChunkVisitor&)>;
private:
    using Rows=ugkp::SharedSnapshotVector<ExchangePacket>;
    struct Archive {
        explicit Archive(Reader r):read(std::move(r)) {}
        Reader read;
        mutable std::mutex countMutex;
        mutable std::optional<std::size_t> count;
        mutable std::once_flag materialize;
        mutable Rows cache;

        void chunks(const ChunkVisitor& visit) const {
            std::size_t n=0;
            read([&](const ExchangePacket* data,std::size_t k) {
                if (k>std::numeric_limits<std::size_t>::max()-n)
                    throw std::length_error("exchange ledger count overflow");
                if (k) visit(data,k);
                n+=k;
            });
            std::lock_guard<std::mutex> locked(countMutex);
            if (count && n!=*count) throw std::runtime_error("frozen exchange ledger extent changed");
            count=n;
        }
        std::size_t size() const {
            {
                std::lock_guard<std::mutex> locked(countMutex);
                if (count) return *count;
            }
            chunks([](const ExchangePacket*,std::size_t){});
            std::lock_guard<std::mutex> locked(countMutex);
            return *count;
        }
        const Rows& rows() const {
            std::call_once(materialize,[&] {
                std::vector<ExchangePacket> all;
                chunks([&](const ExchangePacket* data,std::size_t n){all.insert(all.end(),data,data+n);});
                cache=std::move(all);
            });
            return cache;
        }
    };
    Rows rows_;
    std::shared_ptr<const Archive> archive_;
public:
    AcceptedLedger()=default;
    AcceptedLedger(std::initializer_list<ExchangePacket> rows):rows_(rows){}
    AcceptedLedger& operator=(std::vector<ExchangePacket> rows){rows_=std::move(rows);archive_.reset();return *this;}
    AcceptedLedger& operator=(std::initializer_list<ExchangePacket> rows){rows_=rows;archive_.reset();return *this;}
    void freeze(Reader reader){archive_=std::make_shared<Archive>(std::move(reader));rows_.clear();}
    void clear(){rows_.clear();archive_.reset();}
    std::size_t size() const{return archive_?archive_->size():rows_.size();}
    bool empty() const{return !size();}
    void forEachChunk(const ChunkVisitor& visit) const {
        if (archive_) archive_->chunks(visit);
        else if (!rows_.empty()) visit(rows_.data(),rows_.size());
    }


    const std::vector<ExchangePacket>& values() const{return archive_?archive_->rows().values():rows_.values();}
    std::vector<ExchangePacket>& edit(){if (archive_){rows_=archive_->rows();archive_.reset();}return rows_.edit();}
    const ExchangePacket* data() const{return values().data();}
    ExchangePacket* data(){return edit().data();}
    const ExchangePacket& operator[](std::size_t i) const{return values()[i];}
    ExchangePacket& operator[](std::size_t i){return edit()[i];}
    auto begin() const{return values().begin();}
    auto end() const{return values().end();}
    auto begin(){return edit().begin();}
    auto end(){return edit().end();}
    void push_back(const ExchangePacket& x){edit().push_back(x);}
    void push_back(ExchangePacket&& x){edit().push_back(std::move(x));}
};
template<class Archive> void visit(Archive& a,const AcceptedLedger& ledger){a(ledger.values());}
template<class Archive> void visit(Archive& a,AcceptedLedger& ledger){a(ledger.edit());}
}
