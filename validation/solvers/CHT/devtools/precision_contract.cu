#include <cuda_runtime.h>
#include "GpuPrecisionTypes.H"
#include "wall/GpuFiniteWallContact.H"
#include "../../../../applications/CHT/gpu/GpuChtParticleRestart.H"
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
    std::vector<double> pDepositionArea(3, static_cast<double>(1.25)), out_pDepositionArea(3);
    std::vector<double> pContactDuration(3, static_cast<double>(1.0000000000000002)), out_pContactDuration(3);
    std::vector<double> pContactMaximumArea(3, static_cast<double>(1.25)), out_pContactMaximumArea(3);
    std::vector<double> pContactPeakFraction(3, static_cast<double>(1.25)), out_pContactPeakFraction(3);
    std::vector<float> pColdNodeSpecificEnthalpy(24, static_cast<float>(1.25)), out_pColdNodeSpecificEnthalpy(24);
    std::vector<double> pColdRingSolidMass(24, static_cast<double>(1.25)), out_pColdRingSolidMass(24);
    std::vector<double> pColdFrozenArea(3, static_cast<double>(1.25)), out_pColdFrozenArea(3);
    std::vector<double> pColdContactAge(3, static_cast<double>(1.0000000000000002)), out_pColdContactAge(3);
    std::vector<float> pCold2DNodeSpecificEnthalpy(192, static_cast<float>(1.25)), out_pCold2DNodeSpecificEnthalpy(192);
    std::vector<double> pCold2DRingContactAge(24, static_cast<double>(1.0000000000000002)), out_pCold2DRingContactAge(24);
    std::vector<double> pCold2DFrozenArea(3, static_cast<double>(1.25)), out_pCold2DFrozenArea(3);
    Foam::gpuFshRestart::ConstView input{3,
        px.data(), py.data(), pz.data(), pux.data(), puy.data(), puz.data(), pT.data(), pTheta.data(), pd.data(), pm.data(), pCellId.data(), pStatus.data(), pRng.data(), pOrigId.data(), pStuck.data(), pStuckFaceId.data(), pDepositionArea.data(), pContactDuration.data(), pContactMaximumArea.data(), pContactPeakFraction.data(), pColdNodeSpecificEnthalpy.data(), pColdRingSolidMass.data(), pColdFrozenArea.data(), pColdContactAge.data(), pCold2DNodeSpecificEnthalpy.data(), pCold2DRingContactAge.data(), pCold2DFrozenArea.data()};
    Foam::gpuFshRestart::MutableView output{3,
        out_px.data(), out_py.data(), out_pz.data(), out_pux.data(), out_puy.data(), out_puz.data(), out_pT.data(), out_pTheta.data(), out_pd.data(), out_pm.data(), out_pCellId.data(), out_pStatus.data(), out_pRng.data(), out_pOrigId.data(), out_pStuck.data(), out_pStuckFaceId.data(), out_pDepositionArea.data(), out_pContactDuration.data(), out_pContactMaximumArea.data(), out_pContactPeakFraction.data(), out_pColdNodeSpecificEnthalpy.data(), out_pColdRingSolidMass.data(), out_pColdFrozenArea.data(), out_pColdContactAge.data(), out_pCold2DNodeSpecificEnthalpy.data(), out_pCold2DRingContactAge.data(), out_pCold2DFrozenArea.data()};
    std::vector<double> oldX(3,1.0+0x1p-30),oldY=oldX,oldZ=oldX;
    std::vector<double> outX(3),outY(3),outZ(3),carry(3,0x1p-40),outCarry(3);
    input.puxOld=oldX.data();input.puyOld=oldY.data();input.puzOld=oldZ.data();input.pThermalEnthalpyResidual=carry.data();
    output.puxOld=outX.data();output.puyOld=outY.data();output.puzOld=outZ.data();output.pThermalEnthalpyResidual=outCarry.data();
    std::stringstream io(std::ios::in|std::ios::out|std::ios::binary);
    Foam::gpuFshRestart::writeCurrent(io,input);
    std::string magic;size_t n;uint32_t chunk;io>>magic>>n>>chunk;
    if(n!=3||magic!=Foam::gpuFshRestart::currentFormatName)return 20;
    Foam::gpuFshRestart::readCurrentPayload(io,chunk,output);
    if (px != out_px) return 21;
    if (py != out_py) return 21;
    if (pz != out_pz) return 21;
    if (pux != out_pux) return 21;
    if (puy != out_puy) return 21;
    if (puz != out_puz) return 21;
    if (pT != out_pT) return 21;
    if (pTheta != out_pTheta) return 21;
    if (pd != out_pd) return 21;
    if (pm != out_pm) return 21;
    if (pCellId != out_pCellId) return 21;
    if (pStatus != out_pStatus) return 21;
    if (pRng != out_pRng) return 21;
    if (pOrigId != out_pOrigId) return 21;
    if (pStuck != out_pStuck) return 21;
    if (pStuckFaceId != out_pStuckFaceId) return 21;
    if (pDepositionArea != out_pDepositionArea) return 21;
    if (pContactDuration != out_pContactDuration) return 21;
    if (pContactMaximumArea != out_pContactMaximumArea) return 21;
    if (pContactPeakFraction != out_pContactPeakFraction) return 21;
    if (pColdNodeSpecificEnthalpy != out_pColdNodeSpecificEnthalpy) return 21;
    if (pColdRingSolidMass != out_pColdRingSolidMass) return 21;
    if (pColdFrozenArea != out_pColdFrozenArea) return 21;
    if (pColdContactAge != out_pColdContactAge) return 21;
    if (pCold2DNodeSpecificEnthalpy != out_pCold2DNodeSpecificEnthalpy) return 21;
    if (pCold2DRingContactAge != out_pCold2DRingContactAge) return 21;
    if (pCold2DFrozenArea != out_pCold2DFrozenArea) return 21;
    if(oldX!=outX||oldY!=outY||oldZ!=outZ||carry!=outCarry)return 22;
    std::cout << "PASS physical=" << sizeof(GpuReal)*8
              << " time=64 GPU contact/positive heat/time increment; current mixed-profile exact roundtrip\n";
}
