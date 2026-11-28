#include "notify.hpp"

#include "common/notify.hpp"

namespace caelestia::cli::record {

using Qt::StringLiterals::operator""_s;

quint32 notifyStarted() {
    return common::notify::send({ .summary = u"Recording started"_s, .body = u"Recording..."_s });
}

void notifyStopped(const QString& path) {
    common::notify::send({
        .summary = u"Recording stopped"_s,
        .body = u"Recording saved at %1"_s.arg(path),
        .actions = { u"watch"_s, u"Watch"_s, u"open"_s, u"Open"_s, u"delete"_s, u"Delete"_s },
        .hints = {
            { u"x-caelestia-action-watch"_s, QStringList{ u"xdg-open"_s, path } },
            { u"x-caelestia-action-open"_s, QStringList{ QStringLiteral(CAELESTIA_OPEN_IN_FILE_MANAGER_BIN), path } },
            { u"x-caelestia-action-delete"_s, QStringList{ u"rm"_s, u"-f"_s, u"--"_s, path } },
        },
    });
}

void notifyFailed(const QString& summary, const QString& body) {
    common::notify::send({
        .summary = summary,
        .body = body,
        .urgency = common::notify::Urgency::Critical,
    });
}

} // namespace caelestia::cli::record
