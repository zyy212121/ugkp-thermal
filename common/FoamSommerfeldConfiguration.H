#pragma once
#include "dictionary.H"
#include "token.H"
#include <limits>
namespace Foam {
inline scalar readSommerfeldThreshold(const dictionary& input,scalar fallback=20) {
    if (!input.found("sommerfeldThreshold")) return fallback;
    const ITstream& tokens=input.lookup("sommerfeldThreshold");
    if (tokens.size()==1 && tokens[0].isWord() && tokens[0].wordToken()=="false")
        return std::numeric_limits<scalar>::infinity();
    return readScalar(input.lookup("sommerfeldThreshold"));
}
}
