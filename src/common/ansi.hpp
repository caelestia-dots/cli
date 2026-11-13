#pragma once

#include <qflags.h>
#include <qstring.h>
#include <qstringlist.h>

#include <unistd.h>

#include <cstdlib>
#include <utility>

namespace caelestia::cli::common::ansi {

enum Code : quint16 {
    Reset = 0,
    Bold = 1 << 0,
    Dim = 1 << 1,
    Italic = 1 << 2,
    Underline = 1 << 3,
    Black = 1 << 4,
    Red = 1 << 5,
    Green = 1 << 6,
    Yellow = 1 << 7,
    Blue = 1 << 8,
    Magenta = 1 << 9,
    Cyan = 1 << 10,
    White = 1 << 11,
};
Q_DECLARE_FLAGS(Codes, Code)
Q_DECLARE_OPERATORS_FOR_FLAGS(Codes)

inline QString sequence(Codes codes) {
    using Qt::StringLiterals::operator""_s;

    static constexpr std::pair<Code, int> k_sgr[] = {
        { Bold, 1 },
        { Dim, 2 },
        { Italic, 3 },
        { Underline, 4 },
        { Black, 30 },
        { Red, 31 },
        { Green, 32 },
        { Yellow, 33 },
        { Blue, 34 },
        { Magenta, 35 },
        { Cyan, 36 },
        { White, 37 },
    };

    QStringList parts;
    for (const auto& [code, sgr] : k_sgr)
        if (codes.testFlag(code))
            parts << QString::number(sgr);
    if (parts.isEmpty())
        parts << u"0"_s;
    return u"\033["_s + parts.join(u';') + u'm';
}

inline bool colourSupported() {
    static const bool k_supported = [] {
        if (std::getenv("NO_COLOUR") != nullptr)
            return false;
        if (std::getenv("FORCE_COLOUR") != nullptr)
            return true;
        return isatty(STDOUT_FILENO) == 1;
    }();
    return k_supported;
}

inline QString styled(Codes codes, const QString& text) {
    if (!codes || text.isEmpty() || !colourSupported())
        return text;
    return sequence(codes) + text + sequence(Reset);
}

} // namespace caelestia::cli::common::ansi
