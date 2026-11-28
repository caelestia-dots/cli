#pragma once

#include <qstring.h>
#include <qstringlist.h>

#include <caelestia/ipc/common.hpp>

namespace caelestia::cli::common {

int printError(const QString& err, bool json, const QStringList& warnings = {});
QString messageForError(ipc::Error err);

} // namespace caelestia::cli::common
