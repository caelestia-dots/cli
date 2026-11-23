#include "session.hpp"

#include <qfile.h>
#include <qjsondocument.h>
#include <qjsonobject.h>
#include <qsavefile.h>

#include <caelestia/config/config.hpp>

#include "common/notify.hpp"

namespace caelestia::cli::record {

using Qt::StringLiterals::operator""_s;

namespace {

QString sessionPath() {
    return config::ConfigPaths::runtimeDir() + u"/record-session.json"_s;
}

} // namespace

Session Session::load() {
    QFile file(sessionPath());
    if (!file.open(QIODevice::ReadOnly))
        return {};

    const auto json = QJsonDocument::fromJson(file.readAll()).object();
    return {
        .notificationId = static_cast<quint32>(json.value(u"notificationId").toInteger()),
        .clipboard = json.value(u"clipboard").toBool(),
        .quiet = json.value(u"quiet").toBool(),
    };
}

void Session::save() const {
    QSaveFile file(sessionPath());
    if (!file.open(QIODevice::WriteOnly))
        return;

    const QJsonObject obj{
        { u"notificationId"_s, static_cast<qint64>(notificationId) },
        { u"clipboard"_s, clipboard },
        { u"quiet"_s, quiet },
    };
    file.write(QJsonDocument(obj).toJson(QJsonDocument::Compact));
    file.commit();
}

void Session::end() const {
    QFile::remove(sessionPath());
    if (notificationId != 0)
        common::notify::close(notificationId);
}

} // namespace caelestia::cli::record
