#include "error.hpp"

#include <qjsonarray.h>
#include <qjsondocument.h>
#include <qjsonobject.h>
#include <qtextstream.h>

namespace caelestia::cli::common {

using Qt::StringLiterals::operator""_s;

int printError(const QString& err, bool json, const QStringList& warnings) {
    if (json) {
        QJsonObject obj{ { u"error"_s, err } };
        if (!warnings.isEmpty())
            obj.insert(u"warnings"_s, QJsonArray::fromStringList(warnings));
        QTextStream(stdout) << QJsonDocument(obj).toJson(QJsonDocument::Indented) << u'\n';
    } else {
        for (const auto& warning : warnings)
            QTextStream(stderr) << warning << u'\n';
        QTextStream(stderr) << err << u'\n';
    }
    return 1;
}

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

} // namespace caelestia::cli::common
