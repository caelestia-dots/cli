#pragma once

#include <qdir.h>
#include <qtemporarydir.h>

#include <sys/stat.h>

#include <memory>

#include "common/fakebus.hpp"
#include "common.hpp"

namespace caelestia::cli::test {

class DesktopFakes : protected Sandbox {
protected:
    QTemporaryDir m_bin;
    FakeBus m_bus;
    std::unique_ptr<FakeNotifications> m_notifications;

    [[nodiscard]] QString logPath() const { return m_bin.filePath(u"log"_s); }

    [[nodiscard]] QString log() const { return readFile(logPath()); }

    void installTool(const QString& name, const QByteArray& script) {
        const auto path = m_bin.filePath(name);
        writeFile(path, "#!/bin/sh\nPATH=/usr/bin:/bin\n" + script);
        QCOMPARE(chmod(path.toUtf8().constData(), 0755), 0);
    }

    void setupDesktopEnvironment() {
        setupEnvironment();
        QVERIFY(m_bin.isValid());

        const auto runtime = path(u"runtime"_s);
        QVERIFY(QDir().mkpath(runtime));
        QCOMPARE(chmod(runtime.toUtf8().constData(), 0700), 0);
        qputenv("XDG_RUNTIME_DIR", runtime.toUtf8());

        QVERIFY(m_bus.start(runtime + u"/bus"_s));
        m_notifications = std::make_unique<FakeNotifications>(logPath());
        QVERIFY(FakeBus::serve(
            u"org.freedesktop.Notifications"_s, u"/org/freedesktop/Notifications"_s, m_notifications.get()));

        qputenv("PATH", m_bin.path().toUtf8());
        qputenv("FAKE_LOG", logPath().toUtf8());
        installTool(u"wl-copy"_s, R"(printf 'wl-copy %s:' "$*" >> "$FAKE_LOG"; cat >> "$FAKE_LOG"
)");
    }
};

} // namespace caelestia::cli::test
