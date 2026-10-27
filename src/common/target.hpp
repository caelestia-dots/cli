#pragma once

#include <qstring.h>

#include "parser/spec.hpp"

namespace caelestia::cli::common {

enum class Target : quint8 {
    Screen,
    Region,
};

inline parser::Choices<Target> targetChoices() {
    using Qt::StringLiterals::operator""_s;
    return { { u"screen"_s, Target::Screen }, { u"region"_s, Target::Region } };
}

} // namespace caelestia::cli::common
