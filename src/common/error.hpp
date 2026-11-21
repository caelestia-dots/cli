#pragma once

#include <qstring.h>
#include <qstringlist.h>

namespace caelestia::cli::common {

int printError(const QString& err, bool json, const QStringList& warnings = {});

} // namespace caelestia::cli::common
