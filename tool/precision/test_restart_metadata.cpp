#include "PrecisionRestart.H"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

int main(int argc, char** argv)
{
    assert(argc == 2);
    const std::string directory = argv[1];
    std::filesystem::create_directories(directory);
    std::string error;
    ugkp::precision::beginPrecisionWrite(directory);
    assert(ugkp::precision::checkRestartPrecision(directory, error));
    const auto file = ugkp::precision::metadataPath(directory);
    {
        std::ofstream output(file);
        output << "schema 1\ncomputePrecision 32\nrestartPrecision 32\n";
    }
    assert(ugkp::precision::checkRestartPrecision(directory, error)
           == (ugkp::precision::compiledBits <= 32));
    if(ugkp::precision::compiledBits == 64)
        assert(error.find("lower than selected precision 64") != std::string::npos);
    {
        std::ofstream output(file);
        output << "schema 1\ncomputePrecision 64\nrestartPrecision 64\n";
    }
    assert(ugkp::precision::checkRestartPrecision(directory, error));
    {
        std::ofstream output(file);
        output << "schema 99\ncomputePrecision 32\nrestartPrecision 64\n";
    }
    assert(!ugkp::precision::readRestartPrecision(directory).known);
    assert(ugkp::precision::checkRestartPrecision(directory, error));
    ugkp::precision::beginPrecisionWrite(directory);
    assert(!std::filesystem::exists(file));
    ugkp::precision::writeRestartPrecision(directory);
    const auto recorded = ugkp::precision::readRestartPrecision(directory);
    assert(recorded.known);
    assert(recorded.computeBits == ugkp::precision::compiledBits);
    assert(recorded.restartBits == ugkp::precision::compiledBits);
    std::cout << "Precision " << ugkp::precision::compiledBits << " metadata behavior passed.\n";
    return 0;
}
