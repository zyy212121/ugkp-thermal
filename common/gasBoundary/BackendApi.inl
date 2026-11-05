extern "C" int ugkwpGpuResidentStrictConfigureBoundary
(
    void* handle,int count,const ugkp::boundary::Face* faces,
    int functionCount,const ugkp::boundary::Function* functions,
    int termCount,const ugkp::boundary::Term* terms
)
{
    DeviceState* s=asState(handle);
    if(validateState(s,"shared boundary configuration"))return 1;
    if(count!=s->nFaces||count<0||functionCount<0||termCount<0
        ||(count&&!faces)||(functionCount&&!functions)||(termCount&&!terms)) {
        setLastErrorText("invalid shared boundary configuration");return 1;
    }
    ugkp::boundary::Definition input;
    if(count)input.faces.assign(faces,faces+count);
    if(functionCount)input.functions.assign(functions,functions+functionCount);
    if(termCount)input.terms.assign(terms,terms+termCount);
    if(!s->gasBoundaryOwner)s->gasBoundaryOwner=new ugkp::boundary::Resident;
    std::string error;
    if(!s->gasBoundaryOwner->configure(input,s->gasCaptureStream,error)) {
        setLastErrorText(error.c_str());return 1;
    }
    s->gasBoundaryProgram=s->gasBoundaryOwner->view();
    if(!ugkp::boundary::publish(input,*s,[&](auto* dst,const auto& values) {
        return copyToDevice(dst,values.data(),values.size(),"shared gas boundary definition")==0;
    }))return 1;
    return syncDeviceState(s,"publish shared boundary view");
}

extern "C" int ugkwpGpuResidentStrictUploadGasBoundaryFields
(
    void* handle,
    const int* gasBoundaryKind,
    const int* gasBoundaryRhoFix,
    const int* gasBoundaryUFix,
    const int* gasBoundaryPFix,
    const int* gasBoundaryTFix,
    const int* gasBoundaryPWave,
    const double* gasBoundaryPWaveGamma,
    const double* gasBoundaryPWaveFieldInf,
    const double* gasBoundaryPWaveLInf,
    const double* gasBoundaryRho,
    const double* gasBoundaryUx,
    const double* gasBoundaryUy,
    const double* gasBoundaryUz,
    const double* gasBoundaryP,
    const double* gasBoundaryT
)
{
    DeviceState* s=asState(handle);
    if(validateState(s,"gas boundary upload"))return 1;
    if(!gasBoundaryKind||!gasBoundaryRhoFix||!gasBoundaryUFix||!gasBoundaryPFix
        ||!gasBoundaryTFix||!gasBoundaryPWave||!gasBoundaryPWaveGamma
        ||!gasBoundaryPWaveFieldInf||!gasBoundaryPWaveLInf||!gasBoundaryRho
        ||!gasBoundaryUx||!gasBoundaryUy||!gasBoundaryUz||!gasBoundaryP||!gasBoundaryT) {
        setLastErrorText("null shared gas boundary array");return 1;
    }
    std::vector<ugkp::boundary::Face> faces(s->nFaces);
    for(int i=0;i<s->nFaces;++i) {
        auto& f=faces[i];f.geometry=gasBoundaryKind[i];f.rhoMode=gasBoundaryRhoFix[i];
        f.uMode=gasBoundaryUFix[i];f.pMode=gasBoundaryPFix[i];f.tMode=gasBoundaryTFix[i];
        f.wave=gasBoundaryPWave[i];f.waveGamma=gasBoundaryPWaveGamma[i];
        f.waveFieldInf=gasBoundaryPWaveFieldInf[i];f.waveLInf=gasBoundaryPWaveLInf[i];
        f.value[ugkp::boundary::Density]=gasBoundaryRho[i];
        f.value[ugkp::boundary::Ux]=gasBoundaryUx[i];f.value[ugkp::boundary::Uy]=gasBoundaryUy[i];f.value[ugkp::boundary::Uz]=gasBoundaryUz[i];
        f.value[ugkp::boundary::Pressure]=gasBoundaryP[i];f.value[ugkp::boundary::Temperature]=gasBoundaryT[i];
    }
    return ugkwpGpuResidentStrictConfigureBoundary(handle,s->nFaces,faces.data(),0,nullptr,0,nullptr);
}
