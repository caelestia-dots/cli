#include "notify.hpp"

#include <qfileinfo.h>

#include <caelestia/ipc/client.hpp>

#include "common/error.hpp"
#include "common/notify.hpp"

namespace caelestia::cli::screenshot {

using Qt::StringLiterals::operator""_s;

QString messageForError(ipc::Error err) {
    switch (err) {
    case ipc::Error::Busy:
        return u"The shell is currently busy"_s;
    case ipc::Error::Cancelled:
        return u"Action cancelled"_s;
    case ipc::Error::NotRunning:
        return u"The shell is not currently running"_s;
    case ipc::Error::Failed:
        return u"An unknown error occurred"_s;
    }
    Q_UNREACHABLE_RETURN({});
}

void notify(const QString& cachePath, const QString& savePath, bool copied, bool ok, bool showEdit) {
    QStringList actions;
    if (showEdit)
        actions << u"Edit"_s;
    if (!ok || savePath.isEmpty())
        actions << u"Save"_s;

    const auto path = QFileInfo(!ok || savePath.isEmpty() ? cachePath : savePath).absoluteFilePath();
    const auto helper = QStringLiteral(CAELESTIA_SCREENSHOT_HELPER_BIN);
    QVariantMap hints;
    if (showEdit)
        hints.insert(u"x-caelestia-action-edit"_s, QStringList{ helper, u"edit"_s, path });
    if (!ok || savePath.isEmpty())
        hints.insert(u"x-caelestia-action-save"_s, QStringList{ helper, u"save"_s, path });

    QString body;
    if (!ok)
        body = u"Failed to save screenshot to %1. The screenshot is available at %2."_s.arg(savePath, path);
    else if (copied && savePath.isEmpty())
        body = u"Copied to clipboard"_s;
    else
        body = u"Saved to %1"_s.arg(path) + (copied ? u" and copied to clipboard"_s : QString());

    common::notify::send({
        .summary = ok ? u"Screenshot taken"_s : u"Failed to save screenshot"_s,
        .body = body,
        .image = cachePath,
        .actions = actions,
        .hints = hints,
    });
}

void notifyFailed(const QString& summary, const QString& body) {
    common::notify::send({
        .summary = summary,
        .body = body,
        .urgency = common::notify::Urgency::Critical,
    });
}

int annotateAndNotify(const QString& cachePath, bool quiet, bool json) {
    const auto result = ipc::annotate(cachePath);
    if (const auto* err = std::get_if<ipc::Error>(&result)) {
        const auto msg = messageForError(*err);
        common::printError(u"Failed to annotate screenshot: %1"_s.arg(msg), json);
        if (!quiet)
            notifyFailed(u"Failed to annotate screenshot"_s, msg);
        return 1;
    }

    const auto& [annotatedPath, copied] = std::get<ipc::SaveResult>(result);
    if (!quiet)
        notify(annotatedPath, copied ? QString() : annotatedPath, copied, true, false);

    return 0;
}

} // namespace caelestia::cli::screenshot
