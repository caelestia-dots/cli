#pragma once

#include <qrect.h>
#include <qregularexpression.h>
#include <qstring.h>

#include <optional>

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

inline std::optional<QRect> parseGeometry(const QString& geometry) {
    static const QRegularExpression k_pattern(QStringLiteral("^(-?\\d+),(-?\\d+) (\\d+)x(\\d+)$"));
    const auto match = k_pattern.match(geometry.trimmed());
    if (!match.hasMatch())
        return std::nullopt;

    return QRect{
        match.captured(1).toInt(),
        match.captured(2).toInt(),
        match.captured(3).toInt(),
        match.captured(4).toInt(),
    };
}

} // namespace caelestia::cli::common
