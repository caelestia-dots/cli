#include "hypr.hpp"

#include <qlocalsocket.h>

namespace caelestia::cli::common::hypr {

using Qt::StringLiterals::operator""_s;

namespace {

bool connect(QLocalSocket& socket) {
    const auto dir = socketDir();
    if (dir.isEmpty())
        return false;

    socket.connectToServer(dir + u"/.socket.sock"_s);
    return socket.waitForConnected(k_timeoutMs);
}

} // namespace

QString socketDir() {
    const auto runtimeDir = qEnvironmentVariable("XDG_RUNTIME_DIR");
    const auto signature = qEnvironmentVariable("HYPRLAND_INSTANCE_SIGNATURE");
    if (runtimeDir.isEmpty() || signature.isEmpty())
        return {};

    return u"%1/hypr/%2"_s.arg(runtimeDir, signature);
}

std::optional<QByteArray> request(const QString& message) {
    QLocalSocket socket;
    if (!connect(socket))
        return std::nullopt;

    socket.write(message.toUtf8());
    QByteArray response;
    while (socket.waitForReadyRead(k_timeoutMs))
        response += socket.readAll();

    if (socket.state() == QLocalSocket::ConnectedState)
        return std::nullopt;

    return response;
}

QJsonDocument requestJson(const QString& message) {
    const auto response = request(u"j/"_s + message);
    return response ? QJsonDocument::fromJson(*response) : QJsonDocument();
}

bool eval(const QString& expression) {
    return request(u"eval "_s + expression) == "ok";
}

bool dispatch(const QString& dispatcher, const QStringList& args) {
    return request((QStringList{ u"dispatch"_s, dispatcher } + args).join(u' ')) == "ok";
}

} // namespace caelestia::cli::common::hypr
