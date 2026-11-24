template<class Archive> void visitRetiredExchangeRecords(Archive& archive){
    archive.integer(0,8);
}
inline void visitRetiredExchangeRecords(Decoder& archive){
    const auto count=archive.integer(8);
    if(!count)return;
    Encoder layout;
    layout(ExchangePacket{});
    const auto recordBytes=layout.data.size();
    if(count>(archive.limit-archive.offset)/recordBytes)
        throw std::runtime_error("truncated historical exchange records");
    archive.offset+=static_cast<std::size_t>(count)*recordBytes;
}
