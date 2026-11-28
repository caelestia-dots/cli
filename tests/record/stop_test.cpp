#include "fakes.hpp"

using namespace caelestia::cli::test;
namespace record = caelestia::cli::record;
namespace recorder = caelestia::record;

namespace {

class TestStop : public QObject, protected RecordFakes {
    Q_OBJECT

private:
    void startRecording(const record::StartArgs& args = {}) {
        serveMonitors();
        QCOMPARE(startCmd(args).exit, 0);
        m_hypr->wait();
        m_hypr.reset();
        QFile::remove(logPath());
    }

    [[nodiscard]] bool hasStoppedNotification() const {
        return log().contains(u"Notify Caelestia Recording stopped"_s);
    }

private slots:

    void initTestCase() { setupRecordEnvironment(); }

    void cleanupTestCase() { stopRecorder(); }

    void init() { resetFakes(); }

    void stopMovesRecordingAndNotifies() {
        startRecording();

        const auto result = stopCmd();
        QCOMPARE(result.exit, 0);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(recordings().size(), 1);
        const auto dest = recordingsDir() + u'/' + recordings().constFirst();
        QCOMPARE(result.out, dest + u'\n');
        QCOMPARE(readFile(dest), u"video\n"_s);
        QVERIFY(!recorder::state().running);
        QVERIFY(!QFile::exists(path(u"runtime/caelestia/record-session.json"_s)));
        QVERIFY(log().contains(u"CloseNotification 42"_s));
        QVERIFY(log().contains(
            u"Notify Caelestia Recording stopped Recording saved at %1 "
            "[watch,Watch,open,Open,delete,Delete] "
            "{urgency=1;x-caelestia-action-delete=rm -f -- %1;"
            "x-caelestia-action-open=%2 %1;"
            "x-caelestia-action-watch=xdg-open %1}"_s.arg(dest, QStringLiteral(CAELESTIA_OPEN_IN_FILE_MANAGER_BIN))));
        QVERIFY(!log().contains(u"wl-copy"_s));
    }

    void stopSendsPlainNotificationWithoutActionSupport() {
        m_notifications->capabilities = { u"actions"_s };
        startRecording();
        QCOMPARE(stopCmd().exit, 0);
        m_notifications->capabilities << u"x-caelestia-actions"_s;
        QVERIFY(log().contains(u"Notify Caelestia Recording stopped Recording saved at %1 [] {urgency=1}"_s.arg(
            recordingsDir() + u'/' + recordings().constFirst())));
    }

    void stopPrintsJson() {
        startRecording();
        const auto result = stopCmd(true);
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.json().value(u"path"_s).toString(), recordingsDir() + u'/' + recordings().constFirst());
        QVERIFY(hasStoppedNotification());
    }

    void stopCopiesPathWhenStartedWithClipboard() {
        startRecording({ .clipboard = true });
        QCOMPARE(stopCmd().exit, 0);
        const auto dest = recordingsDir() + u'/' + recordings().constFirst();
        QVERIFY(log().contains(u"wl-copy --type text/uri-list:file://%1\n"_s.arg(dest)));
        QVERIFY(hasStoppedNotification());
    }

    void quietStopSendsNoNotification() {
        startRecording();
        QCOMPARE(stopCmd(false, true).exit, 0);
        QCOMPARE(recordings().size(), 1);
        QVERIFY(log().contains(u"CloseNotification 42"_s));
        QVERIFY(!log().contains(u"Notify"_s));
    }

    void stopIsQuietWhenStartedQuiet() {
        startRecording({ .quiet = true });
        QCOMPARE(stopCmd().exit, 0);
        QCOMPARE(recordings().size(), 1);
        QVERIFY(log().isEmpty());
    }

    void stopFailsWhenIdle() {
        const auto result = stopCmd();
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.err, u"No recording is running\n"_s);
    }

    void toggleStopsAndHonoursClipboard() {
        startRecording();
        const auto result = toggleCmd({ .clipboard = true });
        QCOMPARE(result.exit, 0);
        QVERIFY(!recorder::state().running);
        QCOMPARE(recordings().size(), 1);
        QVERIFY(log().contains(u"wl-copy --type text/uri-list:file://"_s));
        QVERIFY(hasStoppedNotification());
    }

    void pauseTogglesRecorder() {
        startRecording();
        QCOMPARE(pauseCmd().exit, 0);
        QVERIFY(recorder::state().paused);
        QCOMPARE(resumeCmd().exit, 0);
        QVERIFY(!recorder::state().paused);
    }

    void pauseFailsWhenIdle() {
        const auto result = pauseCmd(true);
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.json().value(u"error"_s).toString(), u"No recording is running"_s);
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestStop)

#include "stop_test.moc"
