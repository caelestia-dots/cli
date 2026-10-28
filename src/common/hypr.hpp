#pragma once

#include <qbytearray.h>
#include <qjsondocument.h>
#include <qstring.h>
#include <qstringlist.h>

#include <optional>

namespace caelestia::cli::common::hypr {

constexpr int k_timeoutMs = 5000;

QString socketDir();

std::optional<QByteArray> request(const QString& message);
QJsonDocument requestJson(const QString& message);
bool eval(const QString& expression);
bool dispatch(const QString& dispatcher, const QStringList& args = {});

} // namespace caelestia::cli::common::hypr
