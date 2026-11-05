// Include after resident helper definitions, outside the anonymous namespace.
extern "C" int ugkwpGpuResidentStrictConfigureMolecularTransport(
    void* handle,const ugkwp::MolecularTransportConfiguration* config){
    DeviceState* s=asState(handle);
    if(validateState(s,"molecular transport configuration")||!config)return 1;
    if(s->gasTransportOwner){setLastErrorText("molecular transport is immutable after initialization");return 1;}
    try {
    auto owner=std::make_unique<ugkwp::MolecularTransportDevice>();
    if(!owner){setLastErrorText("allocate molecular transport owner");return 1;}
    auto error=owner->initialize(*config);
    if(error!=cudaSuccess){setLastError("upload molecular transport",error);return 1;}
    const auto view=owner->view();
    error=cudaMemcpy(reinterpret_cast<char*>(s->deviceState)+offsetof(DeviceState,gasViscosity),
        &view,sizeof(view),cudaMemcpyHostToDevice);
    if(error!=cudaSuccess){setLastError("publish molecular transport",error);return 1;}
    s->gasTransportOwner=owner.release();s->gasViscosity=view;
    return 0;
    }catch(const std::exception& e){setLastErrorText(e.what());return 1;}
}
