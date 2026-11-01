#pragma once

#include <qdbusconnection.h>
#include <qfile.h>
#include <qobject.h>
#include <qprocess.h>
#include <qstring.h>
#include <qstringlist.h>
#include <qvariant.h>

#include <utility>

namespace caelestia::cli::test {

using Qt::StringLiterals::operator""_s;

inline void appendLog(const QString& path, const QString& line) {
    QFile file(path);
    if (file.open(QIODevice::Append))
        file.write((line + u'\n').toUtf8());
}

class FakeNotifications : public QObject {
    Q_OBJECT
    Q_CLASSINFO("D-Bus Interface", "org.freedesktop.Notifications")

public:
    explicit FakeNotifications(QString logPath)
        : m_logPath(std::move(logPath)) {}

    QStringList capabilities = { u"actions"_s, u"x-caelestia-actions"_s };

public slots:

    // NOLINTBEGIN(readability-identifier-naming)
    [[nodiscard]] QStringList GetCapabilities() const { return capabilities; }

    uint Notify(const QString& app, uint replaces, const QString& icon, const QString& summary, const QString& body,
        const QStringList& actions, const QVariantMap& hints, int timeout) {
        Q_UNUSED(replaces)
        Q_UNUSED(icon)
        Q_UNUSED(timeout)
        QStringList hintLines;
        for (auto it = hints.cbegin(); it != hints.cend(); ++it)
            hintLines << u"%1=%2"_s.arg(it.key(), it.value().toStringList().join(u' '));
        appendLog(m_logPath,
            u"Notify %1 %2 %3 [%4] {%5}"_s.arg(app, summary, body, actions.join(u','), hintLines.join(u';')));
        return 42;
    }

    void CloseNotification(uint id) { appendLog(m_logPath, u"CloseNotification %1"_s.arg(id)); }

    // NOLINTEND(readability-identifier-naming)

private:
    QString m_logPath;
};

class FakeFileManager : public QObject {
    Q_OBJECT
    Q_CLASSINFO("D-Bus Interface", "org.freedesktop.FileManager1")

public:
    explicit FakeFileManager(QString logPath)
        : m_logPath(std::move(logPath)) {}

public slots:

    // NOLINTNEXTLINE(readability-identifier-naming)
    void ShowItems(const QStringList& uris, const QString& startupId) {
        Q_UNUSED(startupId)
        appendLog(m_logPath, u"ShowItems %1"_s.arg(uris.join(u' ')));
    }

private:
    QString m_logPath;
};

class FakeBus {
public:
    [[nodiscard]] bool start(const QString& socketPath) {
        const auto address = u"unix:path="_s + socketPath;
        const auto config = socketPath + u".conf"_s;
        QFile file(config);
        if (!file.open(QIODevice::WriteOnly))
            return false;
        file.write(QByteArray(k_config).replace("@ADDRESS@", address.toUtf8()));
        file.close();

        m_daemon.start(u"dbus-daemon"_s, { u"--nofork"_s, u"--print-address"_s, u"--config-file="_s + config });
        if (!m_daemon.waitForReadyRead())
            return false;
        qputenv("DBUS_SESSION_BUS_ADDRESS", address.toUtf8());
        return QDBusConnection::sessionBus().isConnected();
    }

    [[nodiscard]] static bool serve(const QString& service, const QString& path, QObject* object) {
        auto bus = QDBusConnection::sessionBus();
        return bus.registerObject(path, object, QDBusConnection::ExportAllSlots) && bus.registerService(service);
    }

private:
    static constexpr auto k_config = R"(<!DOCTYPE busconfig PUBLIC "-//freedesktop//DTD D-Bus Bus Configuration 1.0//EN"
 "http://www.freedesktop.org/standards/dbus/1.0/busconfig.dtd">
<busconfig>
  <type>session</type>
  <listen>@ADDRESS@</listen>
  <policy context="default">
    <allow send_destination="*" eavesdrop="true"/>
    <allow eavesdrop="true"/>
    <allow own="*"/>
  </policy>
</busconfig>
)";

    QProcess m_daemon;
};

} // namespace caelestia::cli::test
