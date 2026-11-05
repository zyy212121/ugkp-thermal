#include "coupling/IntervalAudit.H"
#include <cassert>
#include <iostream>
#include <limits>
int main(){using namespace chmt;
 HostState a;ExchangePacket p;p.energy=1;a.ledger={p};HostState b=a,c=a;
 assert(a.ledger.values().data()==b.ledger.values().data());
 assert(a.ledger.values().data()==c.ledger.values().data());
 b.ledger.clear();assert(b.ledger.empty()&&a.ledger.size()==1);
 c.ledger[0].energy=2;assert(a.ledger.values()[0].energy==1&&c.ledger.values()[0].energy==2);
 assert(a.ledger.values().data()!=c.ledger.values().data());
 ExchangePacket zero;assert(!hasPacketTransfer(zero));
 // Net-zero exchanges with nonzero independent channels must be retained.
 p={};p.species[0]=1;p.species[1]=-1;assert(hasPacketTransfer(p));
 p={};p.poreSweep[0]=1;assert(hasPacketTransfer(p));
 p={};p.conductive=1;p.advective=-1;assert(hasPacketTransfer(p));
 p={};p.condensed[0]=1;assert(hasPacketTransfer(p));
 p={};p.momentum.z=1;assert(hasPacketTransfer(p));
 p={};p.radiation=std::numeric_limits<double>::quiet_NaN();assert(hasPacketTransfer(p));
 IntervalHistory history;std::string error;assert(history.begin({{1,0},0,1},error));
 GasIntervalRecord record;record.begin=0;record.end=1;record.actualDt=1;p={};p.energy=3;record.packets={p};
 assert(history.appendAccepted(record,error));assert(finalizeIntervalLedger(history,a,error));
 assert(a.ledger.size()==1&&a.ledger.values()[0].energy==3);
 assert(c.ledger.values()[0].energy==2);
 HostState released;auto records=history.releaseRecords();
 assert(history.records().empty());assert(finalizeReleasedIntervalLedger(std::move(records),released,error));
 assert(released.ledger.size()==a.ledger.size());
 assert(released.ledger.values()[0].energy==a.ledger.values()[0].energy);
 assert(released.ledger.values()[0].consumerMask==a.ledger.values()[0].consumerMask);
 std::cout<<"LEDGER_STORAGE_PASS snapshot_sharing mutation_isolation exact_optional_zero_filter\n";
}
