#include "notify.hpp"

#include <qdbusconnection.h>
#include <qdbusmessage.h>
#include <qdbusreply.h>

namespace caelestia::cli::common::notify {

using Qt::StringLiterals::operator""_s;

namespace {

QDBusMessage method(const QString& name) {
    return QDBusMessage::createMethodCall(u"org.freedesktop.Notifications"_s, u"/org/freedesktop/Notifications"_s,
        u"org.freedesktop.Notifications"_s, name);
}

bool supportsActions() {
    const QDBusReply<QStringList> reply = QDBusConnection::sessionBus().call(method(u"GetCapabilities"_s));
    return reply.isValid() && reply.value().contains(u"x-caelestia-actions");
}

} // namespace

quint32 send(Notification notification) {
    if (!notification.actions.isEmpty() && !supportsActions()) {
        notification.actions.clear();
        notification.hints.removeIf([](const auto& hint) {
            return hint.key().startsWith(u"x-caelestia-action-");
        });
    }
    if (!notification.image.isEmpty())
        notification.hints.insert(u"image-path"_s, notification.image);
    notification.hints.insert(u"urgency"_s, QVariant::fromValue(static_cast<uchar>(notification.urgency)));

    auto msg = method(u"Notify"_s);
    msg << u"Caelestia"_s << 0u << QString() << notification.summary << notification.body << notification.actions
        << notification.hints << -1;
    const QDBusReply<quint32> reply = QDBusConnection::sessionBus().call(msg);
    return reply.isValid() ? reply.value() : 0;
}

void close(quint32 id) {
    auto msg = method(u"CloseNotification"_s);
    msg << id;
    QDBusConnection::sessionBus().call(msg);
}

} // namespace caelestia::cli::common::notify
