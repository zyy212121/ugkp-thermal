#include "dictionary.H"
#include "IFstream.H"
#include "OSspecific.H"
#include "error.H"
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <unistd.h>

namespace {
std::filesystem::path caseDirectory(int argc, char** argv)
{
    std::filesystem::path result = std::filesystem::current_path();
    for (int i = 1; i + 1 < argc; ++i)
        if (std::string(argv[i]) == "-case") result = argv[++i];
    return std::filesystem::absolute(result).lexically_normal();
}

int selectedPrecision(const std::filesystem::path& directory)
{
    const auto path = directory / "constant" / "scheme";
    if (!std::filesystem::exists(path)) return 32;
    Foam::FatalError.throwExceptions();
    Foam::FatalIOError.throwExceptions();
    try
    {
        Foam::IFstream input(path.string());
        if (!input.good())
        {
            std::cerr << "Warning: cannot read " << path << "; using default precision 32.\n";
            return 32;
        }
        const Foam::dictionary properties(input);
        const Foam::label bits = properties.lookupOrDefault<Foam::label>("precision", 32);
        if (bits == 32 || bits == 64) return int(bits);
        std::cerr << "Warning: precision " << bits << " in " << path
                  << " is not 32 or 64; using default precision 32.\n";
    }
    catch (const Foam::error& error)
    {
        std::cerr << "Warning: could not parse " << path
                  << "; using default precision 32. " << error.message() << '\n';
    }
    return 32;
}
}

int main(int argc, char** argv)
{
    const auto directory = caseDirectory(argc, argv);
    Foam::setEnv("FOAM_CASE", directory.string(), true);
    Foam::setEnv("FOAM_CASENAME", directory.filename().string(), true);
    const int bits = selectedPrecision(directory);
    const auto solver = std::filesystem::path(argv[0]).filename();
    // The installed entry points are copies of this launcher, not CUDA owners.
    const auto executable = std::filesystem::canonical("/proc/self/exe");
    const auto backend = executable.parent_path() / ".precision"
                       / std::to_string(bits) / solver;
    std::cout << "UGKP startup precision=" << bits
              << " (fixed for this process), backend=" << backend << std::endl;
    ::execv(backend.c_str(), argv);
    std::cerr << "Cannot execute precision " << bits << " backend " << backend
              << ": " << std::strerror(errno) << '\n';
    return 127;
}
