#pragma once

#include <qstring.h>

#include <caelestia/ipc/common.hpp>

namespace caelestia::cli::screenshot {

QString messageForError(ipc::Error err);

void notify(const QString& cachePath, const QString& savePath, bool copied, bool ok, bool showEdit = true);
void notifyCopied();
void notifyFailed(const QString& summary, const QString& body);

int annotateAndNotify(const QString& cachePath, bool quiet, bool json);

} // namespace caelestia::cli::screenshot
