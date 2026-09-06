#include <cuda_runtime.h>
#include "GpuPrecisionTypes.H"
#include "wall/GpuFiniteWallContact.H"
#include "../gpu/GpuChtParticleRestart.H"
#include <iostream>
#include <sstream>
#include <vector>
__global__ void checkNumerics(int* result)
{
    int failures=0;
    volatile GpuTime time=1.0;
    const GpuTime dt=1e-8;
    if (!(time+dt>time)) ++failures;
    const auto impact=Foam::gpuThermal::evaluateFiniteWallContactImpact(
        GPU_R(2500),GPU_R(50e-6),GPU_R(1e-8),GPU_R(-0.8191520442889918));
    if (!impact.valid || !(impact.peakTimeFraction>0 && impact.peakTimeFraction<1)) ++failures;
    const auto heat=Foam::gpuThermal::wallInterfaceConductanceTimeIntegral(
        GPU_R(1e-8),GPU_R(1),GpuTime(1),GpuTime(1)+1e-9,
        GPU_R(1e-4),GPU_R(0),GPU_R(10000),true);
    if (!(heat>0) || !isfinite(heat)) ++failures;
    *result=failures;
}
int main()
{
    static_assert(sizeof(GpuTime)==8,"time storage");
    int* device=nullptr; int result=-1;
    if (cudaMalloc(&device,sizeof(int))!=cudaSuccess) return 10;
    checkNumerics<<<1,1>>>(device);
    if (cudaMemcpy(&result,device,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess) return 11;
    cudaFree(device);
    if (result) return 12;
    std::vector<double> px(3, static_cast<double>(1.25)), out_px(3);
    std::vector<double> py(3, static_cast<double>(1.25)), out_py(3);
    std::vector<double> pz(3, static_cast<double>(1.25)), out_pz(3);
    std::vector<double> pux(3, static_cast<double>(1.25)), out_pux(3);
    std::vector<double> puy(3, static_cast<double>(1.25)), out_puy(3);
    std::vector<double> puz(3, static_cast<double>(1.25)), out_puz(3);
    std::vector<double> pT(3, static_cast<double>(1.25)), out_pT(3);
    std::vector<double> pTheta(3, static_cast<double>(1.25)), out_pTheta(3);
    std::vector<double> pd(3, static_cast<double>(1.25)), out_pd(3);
    std::vector<double> pm(3, static_cast<double>(1.25)), out_pm(3);
    std::vector<std::int32_t> pCellId(3, static_cast<std::int32_t>(1)), out_pCellId(3);
    std::vector<std::int32_t> pStatus(3, static_cast<std::int32_t>(1)), out_pStatus(3);
    std::vector<std::uint64_t> pRng(3, static_cast<std::uint64_t>(1)), out_pRng(3);
    std::vector<std::uint64_t> pOrigId(3, static_cast<std::uint64_t>(1)), out_pOrigId(3);
    std::vector<std::uint8_t> pStuck(3, static_cast<std::uint8_t>(1)), out_pStuck(3);
    std::vector<std::int32_t> pStuckFaceId(3, static_cast<std::int32_t>(1)), out_pStuckFaceId(3);
    std::vector<float> pDepositionArea(3, static_cast<float>(1.25)), out_pDepositionArea(3);
    std::vector<double> pContactDuration(3, static_cast<double>(1.0000000000000002)), out_pContactDuration(3);
    std::vector<float> pContactMaximumArea(3, static_cast<float>(1.25)), out_pContactMaximumArea(3);
    std::vector<float> pContactPeakFraction(3, static_cast<float>(1.25)), out_pContactPeakFraction(3);
    std::vector<float> pColdNodeSpecificEnthalpy(24, static_cast<float>(1.25)), out_pColdNodeSpecificEnthalpy(24);
    std::vector<float> pColdRingSolidMass(24, static_cast<float>(1.25)), out_pColdRingSolidMass(24);
    std::vector<float> pColdFrozenArea(3, static_cast<float>(1.25)), out_pColdFrozenArea(3);
    std::vector<double> pColdContactAge(3, static_cast<double>(1.0000000000000002)), out_pColdContactAge(3);
    std::vector<float> pCold2DNodeSpecificEnthalpy(192, static_cast<float>(1.25)), out_pCold2DNodeSpecificEnthalpy(192);
    std::vector<double> pCold2DRingContactAge(24, static_cast<double>(1.0000000000000002)), out_pCold2DRingContactAge(24);
    std::vector<float> pCold2DFrozenArea(3, static_cast<float>(1.25)), out_pCold2DFrozenArea(3);
    Foam::gpuFshRestart::ConstView input{3,
        px.data(), py.data(), pz.data(), pux.data(), puy.data(), puz.data(), pT.data(), pTheta.data(), pd.data(), pm.data(), pCellId.data(), pStatus.data(), pRng.data(), pOrigId.data(), pStuck.data(), pStuckFaceId.data(), pDepositionArea.data(), pContactDuration.data(), pContactMaximumArea.data(), pContactPeakFraction.data(), pColdNodeSpecificEnthalpy.data(), pColdRingSolidMass.data(), pColdFrozenArea.data(), pColdContactAge.data(), pCold2DNodeSpecificEnthalpy.data(), pCold2DRingContactAge.data(), pCold2DFrozenArea.data()};
    Foam::gpuFshRestart::MutableView output{3,
        out_px.data(), out_py.data(), out_pz.data(), out_pux.data(), out_puy.data(), out_puz.data(), out_pT.data(), out_pTheta.data(), out_pd.data(), out_pm.data(), out_pCellId.data(), out_pStatus.data(), out_pRng.data(), out_pOrigId.data(), out_pStuck.data(), out_pStuckFaceId.data(), out_pDepositionArea.data(), out_pContactDuration.data(), out_pContactMaximumArea.data(), out_pContactPeakFraction.data(), out_pColdNodeSpecificEnthalpy.data(), out_pColdRingSolidMass.data(), out_pColdFrozenArea.data(), out_pColdContactAge.data(), out_pCold2DNodeSpecificEnthalpy.data(), out_pCold2DRingContactAge.data(), out_pCold2DFrozenArea.data()};
    for (int version : {5,6})
    {
        std::stringstream io(std::ios::in|std::ios::out|std::ios::binary);
        if (version==5) Foam::gpuFshRestart::writeV5(io,input);
        else Foam::gpuFshRestart::writeV6(io,input);
        std::string magic; size_t n; uint32_t chunk;
        io >> magic >> n >> chunk;
        if (n!=3) return 20;
        if (version==5) Foam::gpuFshRestart::readV5Payload(io,chunk,output);
        else Foam::gpuFshRestart::readV6Payload(io,chunk,output);
        for (size_t i=0; i<px.size(); ++i) if (out_px[i] != (version == 5 ? px[i] : px[i])) return 21;
        for (size_t i=0; i<py.size(); ++i) if (out_py[i] != (version == 5 ? py[i] : py[i])) return 21;
        for (size_t i=0; i<pz.size(); ++i) if (out_pz[i] != (version == 5 ? pz[i] : pz[i])) return 21;
        for (size_t i=0; i<pux.size(); ++i) if (out_pux[i] != (version == 5 ? pux[i] : pux[i])) return 21;
        for (size_t i=0; i<puy.size(); ++i) if (out_puy[i] != (version == 5 ? puy[i] : puy[i])) return 21;
        for (size_t i=0; i<puz.size(); ++i) if (out_puz[i] != (version == 5 ? puz[i] : puz[i])) return 21;
        for (size_t i=0; i<pT.size(); ++i) if (out_pT[i] != (version == 5 ? pT[i] : pT[i])) return 21;
        for (size_t i=0; i<pTheta.size(); ++i) if (out_pTheta[i] != (version == 5 ? pTheta[i] : pTheta[i])) return 21;
        for (size_t i=0; i<pd.size(); ++i) if (out_pd[i] != (version == 5 ? pd[i] : pd[i])) return 21;
        for (size_t i=0; i<pm.size(); ++i) if (out_pm[i] != (version == 5 ? pm[i] : pm[i])) return 21;
        for (size_t i=0; i<pCellId.size(); ++i) if (out_pCellId[i] != (version == 5 ? pCellId[i] : pCellId[i])) return 21;
        for (size_t i=0; i<pStatus.size(); ++i) if (out_pStatus[i] != (version == 5 ? pStatus[i] : pStatus[i])) return 21;
        for (size_t i=0; i<pRng.size(); ++i) if (out_pRng[i] != (version == 5 ? pRng[i] : pRng[i])) return 21;
        for (size_t i=0; i<pOrigId.size(); ++i) if (out_pOrigId[i] != (version == 5 ? pOrigId[i] : pOrigId[i])) return 21;
        for (size_t i=0; i<pStuck.size(); ++i) if (out_pStuck[i] != (version == 5 ? pStuck[i] : pStuck[i])) return 21;
        for (size_t i=0; i<pStuckFaceId.size(); ++i) if (out_pStuckFaceId[i] != (version == 5 ? pStuckFaceId[i] : pStuckFaceId[i])) return 21;
        for (size_t i=0; i<pDepositionArea.size(); ++i) if (out_pDepositionArea[i] != (version == 5 ? pDepositionArea[i] : pDepositionArea[i])) return 21;
        for (size_t i=0; i<pContactDuration.size(); ++i) if (out_pContactDuration[i] != (version == 5 ? static_cast<double>(static_cast<float>(pContactDuration[i])) : pContactDuration[i])) return 21;
        for (size_t i=0; i<pContactMaximumArea.size(); ++i) if (out_pContactMaximumArea[i] != (version == 5 ? pContactMaximumArea[i] : pContactMaximumArea[i])) return 21;
        for (size_t i=0; i<pContactPeakFraction.size(); ++i) if (out_pContactPeakFraction[i] != (version == 5 ? pContactPeakFraction[i] : pContactPeakFraction[i])) return 21;
        for (size_t i=0; i<pColdNodeSpecificEnthalpy.size(); ++i) if (out_pColdNodeSpecificEnthalpy[i] != (version == 5 ? pColdNodeSpecificEnthalpy[i] : pColdNodeSpecificEnthalpy[i])) return 21;
        for (size_t i=0; i<pColdRingSolidMass.size(); ++i) if (out_pColdRingSolidMass[i] != (version == 5 ? pColdRingSolidMass[i] : pColdRingSolidMass[i])) return 21;
        for (size_t i=0; i<pColdFrozenArea.size(); ++i) if (out_pColdFrozenArea[i] != (version == 5 ? pColdFrozenArea[i] : pColdFrozenArea[i])) return 21;
        for (size_t i=0; i<pColdContactAge.size(); ++i) if (out_pColdContactAge[i] != (version == 5 ? static_cast<double>(static_cast<float>(pColdContactAge[i])) : pColdContactAge[i])) return 21;
        for (size_t i=0; i<pCold2DNodeSpecificEnthalpy.size(); ++i) if (out_pCold2DNodeSpecificEnthalpy[i] != (version == 5 ? pCold2DNodeSpecificEnthalpy[i] : pCold2DNodeSpecificEnthalpy[i])) return 21;
        for (size_t i=0; i<pCold2DRingContactAge.size(); ++i) if (out_pCold2DRingContactAge[i] != (version == 5 ? static_cast<double>(static_cast<float>(pCold2DRingContactAge[i])) : pCold2DRingContactAge[i])) return 21;
        for (size_t i=0; i<pCold2DFrozenArea.size(); ++i) if (out_pCold2DFrozenArea[i] != (version == 5 ? pCold2DFrozenArea[i] : pCold2DFrozenArea[i])) return 21;
    }
    std::cout << "PASS physical=" << sizeof(GpuReal)*8
              << " time=64 GPU contact/positive heat/time increment; schema5 conversion/schema6 exact roundtrip\n";
}
