#pragma once

#include <qstring.h>

namespace caelestia::cli::record {

quint32 notifyStarted();
void notifyStopped(const QString& path);
void notifyFailed(const QString& summary, const QString& body);

} // namespace caelestia::cli::record
