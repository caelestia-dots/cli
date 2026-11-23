#pragma once

#include <qdir.h>
#include <qjsonobject.h>
#include <qlocalsocket.h>
#include <qrect.h>
#include <qsemaphore.h>
#include <qthread.h>

#include <caelestia/ipc/server.hpp>
#include <caelestia/record/record.hpp>
#include <caelestia/utils/hypr.hpp>
#include <sys/stat.h>

#include <memory>
#include <optional>

#include "common/fakebus.hpp"
#include "common/fakesocket.hpp"
#include "record/record.hpp"
#include "common.hpp"

namespace hypr = caelestia::utils::hypr;

namespace caelestia::cli::test {

constexpr auto k_monitors = R"([
    {"name":"DP-1","x":0,"y":0,"width":2560,"height":1440,"refreshRate":143.9,"focused":true},
    {"name":"HDMI-A-1","x":2560,"y":0,"width":1920,"height":1080,"refreshRate":60.0,"focused":false}
])";

class FakeShell : public QThread {
public:
    explicit FakeShell(std::optional<QRect> region)
        : m_region(region) {
        start();
        m_listening.acquire();
    }

    ~FakeShell() override {
        quit();
        wait();
    }

protected:
    void run() override {
        ipc::Server server;
        connect(&server, &ipc::Server::pickRegionRequested, &server, [this](ipc::PickRegionRequest* request) {
            if (m_region)
                request->respond(*m_region);
            else
                request->reject();
        });
        m_listening.release();
        exec();
    }

private:
    std::optional<QRect> m_region;
    QSemaphore m_listening;
};

class RecordFakes : protected Sandbox {
protected:
    QTemporaryDir m_bin;
    FakeBus m_bus;
    std::unique_ptr<FakeNotifications> m_notifications;
    std::unique_ptr<FakeSocket> m_hypr;

    [[nodiscard]] QString logPath() const { return m_bin.filePath(u"log"_s); }

    [[nodiscard]] QString log() const { return readFile(logPath()); }

    [[nodiscard]] QString recordingsDir() const { return path(u"Videos/Recordings"_s); }

    [[nodiscard]] QStringList recordings() const {
        return QDir(recordingsDir()).entryList({ u"recording_*.mp4"_s }, QDir::Files);
    }

    static Output startCmd(const record::StartArgs& args, bool json = false) {
        return capture([&] {
            return record::start(args, { .command = u"caelestia record start"_s, .json = json });
        });
    }

    static Output toggleCmd(const record::StartArgs& args, bool json = false) {
        return capture([&] {
            return record::toggle(args, { .command = u"caelestia record toggle"_s, .json = json });
        });
    }

    static Output stopCmd(bool json = false, bool quiet = false) {
        return capture([=] {
            return record::stop({ .quiet = quiet }, { .command = u"caelestia record stop"_s, .json = json });
        });
    }

    static Output pauseCmd(bool json = false) {
        return capture([=] {
            return record::pause(record::QuietArgs(), { .command = u"caelestia record pause"_s, .json = json });
        });
    }

    static Output resumeCmd() {
        return capture([] {
            return record::resume(record::QuietArgs(), { .command = u"caelestia record resume"_s });
        });
    }

    [[nodiscard]] static bool waitForState(bool running) {
        for (int i = 0; i < 100; ++i) {
            if (caelestia::record::state().running == running)
                return true;
            QThread::msleep(20);
        }
        return false;
    }

    void installTool(const QString& name, const QByteArray& script) {
        const auto path = m_bin.filePath(name);
        writeFile(path, "#!/bin/sh\nPATH=/usr/bin:/bin\n" + script);
        QCOMPARE(chmod(path.toUtf8().constData(), 0755), 0);
    }

    void setupRecordEnvironment() {
        setupEnvironment();
        QVERIFY(m_bin.isValid());

        const auto runtime = path(u"runtime"_s);
        QVERIFY(QDir().mkpath(runtime));
        QCOMPARE(chmod(runtime.toUtf8().constData(), 0700), 0);
        qputenv("XDG_RUNTIME_DIR", runtime.toUtf8());
        qputenv("HYPRLAND_INSTANCE_SIGNATURE", "test");
        QVERIFY(QDir().mkpath(hypr::socketDir()));
        qputenv("XDG_VIDEOS_DIR", path(u"Videos"_s).toUtf8());

        QVERIFY(m_bus.start(runtime + u"/bus"_s));
        m_notifications = std::make_unique<FakeNotifications>(logPath());
        QVERIFY(FakeBus::serve(
            u"org.freedesktop.Notifications"_s, u"/org/freedesktop/Notifications"_s, m_notifications.get()));

        qputenv("PATH", m_bin.path().toUtf8());
        qputenv("FAKE_LOG", logPath().toUtf8());
        installTool(u"gpu-screen-recorder"_s, R"(echo "gpu-screen-recorder $*" >> "$FAKE_LOG"
[ -n "$RECORDER_EXIT" ] && exit "$RECORDER_EXIT"
trap 'exit 0' INT TERM
trap 'echo pause >> "$FAKE_LOG"' USR2
echo video > "${@: -1}" 2>/dev/null || { eval "out=\$$#"; echo video > "$out"; }
while :; do sleep 0.05; done
)");
        installTool(u"wl-copy"_s, R"(printf 'wl-copy %s:' "$*" >> "$FAKE_LOG"; cat >> "$FAKE_LOG"
)");
    }

    void serveMonitors() {
        m_hypr = std::make_unique<FakeSocket>(hypr::socketDir() + u"/.socket.sock"_s, [](QLocalSocket& conn) {
            conn.waitForReadyRead(hypr::k_timeoutMs);
            conn.readAll();
            conn.write(k_monitors);
        });
    }

    static void stopRecorder() { [[maybe_unused]] const auto stopped = caelestia::record::stop(); }

    void resetFakes() {
        stopRecorder();
        if (m_hypr) {
            m_hypr->wait();
            m_hypr.reset();
        }
        QFile::remove(logPath());
        QFile::remove(hypr::socketDir() + u"/.socket.sock"_s);
        QDir(recordingsDir()).removeRecursively();
        qunsetenv("RECORDER_EXIT");
    }
};

} // namespace caelestia::cli::test
