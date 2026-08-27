#include "../CharacteristicMuscl.cuh"

#include <cmath>
#include <cstdlib>
#include <iostream>

namespace
{

using ugkpcharacteristic::Increment;

void require(const bool condition, const char* message)
{
    if (!condition)
    {
        std::cerr << "FAIL: " << message << '\n';
        std::exit(1);
    }
}

bool close(const double actual, const double expected)
{
    return std::fabs(actual - expected)
        <= 2.0e-13*std::fmax(1.0, std::fmax(std::fabs(actual), std::fabs(expected)));
}

void testPureAcousticPairIsCappedAtTheFace()
{
    Increment centreDifference{1.0, 2.0, 0.0, 0.0, 4.0};
    Increment left{2.0, 4.0, 0.0, 0.0, 8.0};
    Increment right{-2.0, -4.0, 0.0, 0.0, -8.0};
    ugkpcharacteristic::limitFacePair
    (
        left,
        right,
        centreDifference,
        1.0,
        0.0,
        0.0,
        1.0,
        2.0,
        0.5
    );
    require(close(left.rho, 0.5), "left acoustic density increment");
    require(close(left.ux, 1.0), "left acoustic velocity increment");
    require(close(left.p, 2.0), "left acoustic pressure increment");
    require(close(right.rho, -0.5), "right acoustic density increment");
    require(close(right.ux, -1.0), "right acoustic velocity increment");
    require(close(right.p, -2.0), "right acoustic pressure increment");
}

void testContactAndShearRemainIndependent()
{
    Increment centreDifference{1.0, 0.0, 2.0, 0.0, 0.0};
    Increment left{2.0, 0.0, 4.0, 0.0, 0.0};
    Increment right{-2.0, 0.0, -4.0, 0.0, 0.0};
    ugkpcharacteristic::limitFacePair
    (
        left,
        right,
        centreDifference,
        1.0,
        0.0,
        0.0,
        1.0,
        2.0,
        0.5
    );
    require(close(left.rho, 0.5), "left contact increment");
    require(close(left.uy, 1.0), "left shear increment");
    require(close(left.p, 0.0), "contact must not create pressure");
    require(close(right.rho, -0.5), "right contact increment");
    require(close(right.uy, -1.0), "right shear increment");
}

void testWrongDirectionIsRejected()
{
    const Increment centreDifference{1.0, 0.0, 0.0, 0.0, 0.0};
    Increment left{-0.1, 0.0, 0.0, 0.0, 0.0};
    Increment right{0.1, 0.0, 0.0, 0.0, 0.0};
    ugkpcharacteristic::limitFacePair
    (
        left,
        right,
        centreDifference,
        1.0,
        0.0,
        0.0,
        1.0,
        2.0,
        0.5
    );
    require(close(left.rho, 0.0), "opposed left contact slope");
    require(close(right.rho, 0.0), "opposed right contact slope");
}

}             

__global__ void compileDeviceCharacteristic(Increment* increment)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *increment = ugkpcharacteristic::limitOneSide
        (
            *increment,
            *increment,
            1.0,
            0.0,
            0.0,
            1.0,
            2.0,
            0.5
        );
    }
}

int main()
{
    testPureAcousticPairIsCappedAtTheFace();
    testContactAndShearRemainIndependent();
    testWrongDirectionIsRejected();
    std::cout << "PASS: UGKP characteristic MUSCL host/device tests\n";
    return 0;
}
