#include "fakes.hpp"

using namespace caelestia::cli::test;
namespace record = caelestia::cli::record;
namespace recorder = caelestia::record;
namespace config = caelestia::config;
namespace common = caelestia::cli::common;

namespace {

class TestStart : public QObject, protected RecordFakes {
    Q_OBJECT

private:
    [[nodiscard]] QString recorderLine() const {
        const auto lines = log().split(u'\n');
        for (const auto& line : lines) {
            if (line.startsWith(u"gpu-screen-recorder "_s))
                return line;
        }
        return {};
    }

    [[nodiscard]] QString outputPath() const { return path(u"state/caelestia/record/recording.mp4"_s); }

private slots:

    void initTestCase() { setupRecordEnvironment(); }

    void cleanupTestCase() { stopRecorder(); }

    void init() { resetFakes(); }

    void recordsFocusedMonitor() {
        serveMonitors();
        const auto result = startCmd({});
        QCOMPARE(result.exit, 0);
        QVERIFY(result.out.isEmpty());
        QVERIFY(result.err.isEmpty());
        QCOMPARE(recorderLine(), u"gpu-screen-recorder -w DP-1 -f 144 -o %1"_s.arg(outputPath()));
        QVERIFY(log().contains(u"Notify Caelestia Recording started Recording... [] {urgency=1}"_s));
        QVERIFY(recorder::state().running);
        QVERIFY(QFile::exists(path(u"runtime/caelestia/record-session.json"_s)));
    }

    void recordsAudioAndExtraArgs() {
        serveMonitors();
        writeFile(config::Config::configPath(), R"({"record":{"extraArgs":["-q","ultra"]}})");
        config::Config::instance().reload();

        QCOMPARE(startCmd({ .sound = true }).exit, 0);
        QCOMPARE(
            recorderLine(), u"gpu-screen-recorder -w DP-1 -f 144 -a default_output -q ultra -o %1"_s.arg(outputPath()));

        writeFile(config::Config::configPath(), k_configFixture);
        config::Config::instance().reload();
    }

    void rejectsNonArrayExtraArgs() {
        writeFile(config::Config::configPath(), R"({"record":{"extraArgs":"-q"}})");
        config::Config::instance().reload();

        const auto result = startCmd({});
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.err, u"Config option 'record.extraArgs' should be an array\n"_s);
        QVERIFY(!recorder::state().running);

        writeFile(config::Config::configPath(), k_configFixture);
        config::Config::instance().reload();
    }

    void recordsPickedRegion() {
        serveMonitors();
        const FakeShell shell(QRect(10, 20, 300, 200));
        QCOMPARE(startCmd({ .target = common::Target::Region }).exit, 0);
        QCOMPARE(
            recorderLine(), u"gpu-screen-recorder -w region -region 300x200+10+20 -f 144 -o %1"_s.arg(outputPath()));
    }

    void recordsFixedGeometry() {
        serveMonitors();
        QCOMPARE(startCmd({ .target = common::Target::Region, .geometry = u"2600,10 100x100"_s }).exit, 0);
        QCOMPARE(
            recorderLine(), u"gpu-screen-recorder -w region -region 100x100+2600+10 -f 60 -o %1"_s.arg(outputPath()));
    }

    void rejectsInvalidGeometry() {
        serveMonitors();
        const auto result = startCmd({ .target = common::Target::Region, .geometry = u"nonsense"_s }, true);
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.json().value(u"error"_s).toString(), u"Invalid geometry: nonsense"_s);
        QVERIFY(!recorder::state().running);
        QVERIFY(!log().contains(u"Notify"_s));
    }

    void cancelledPickExitsQuietly() {
        serveMonitors();
        const FakeShell shell(std::nullopt);
        const auto result = startCmd({ .target = common::Target::Region });
        QCOMPARE(result.exit, 0);
        QVERIFY(result.err.isEmpty());
        QVERIFY(log().isEmpty());
    }

    void failsWhenShellNotRunning() {
        serveMonitors();
        const auto result = startCmd({ .target = common::Target::Region });
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.err, u"Failed to select a region: The shell is not currently running\n"_s);
        QVERIFY(log().contains(u"Notify Caelestia Failed to start recording Failed to select a region: "
                               "The shell is not currently running [] {urgency=2}"_s));
        QVERIFY(!log().contains(u"gpu-screen-recorder"_s));
    }

    void refusesWhileRunning() {
        serveMonitors();
        QCOMPARE(startCmd({}).exit, 0);
        m_hypr->wait();
        m_hypr.reset();

        const auto result = startCmd({});
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.err, u"A recording is already running\n"_s);
    }

    void reportsEarlyRecorderExit() {
        serveMonitors();
        qputenv("RECORDER_EXIT", "3");
        const record::StartArgs args{};
        const auto result = startCmd(args, true);
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.json().value(u"error"_s).toString(), u"The recorder exited unexpectedly"_s);
        QVERIFY(log().contains(u"CloseNotification 42"_s));
        QVERIFY(log().contains(
            u"Notify Caelestia Failed to start recording The recorder exited unexpectedly [] {urgency=2}"_s));
        QVERIFY(!QFile::exists(path(u"runtime/caelestia/record-session.json"_s)));
    }

    void quietStartSendsNoNotification() {
        serveMonitors();
        QCOMPARE(startCmd({ .quiet = true }).exit, 0);
        QVERIFY(recorder::state().running);
        QVERIFY(!log().contains(u"Notify"_s));
    }

    void quietFailureSendsNoNotification() {
        serveMonitors();
        qputenv("RECORDER_EXIT", "3");
        const auto result = startCmd({ .quiet = true });
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.err, u"The recorder exited unexpectedly\n"_s);
        QVERIFY(!log().contains(u"Notify"_s));
    }

    void toggleStartsWhenIdle() {
        serveMonitors();
        QCOMPARE(toggleCmd({}).exit, 0);
        QVERIFY(recorder::state().running);
        QCOMPARE(recorderLine(), u"gpu-screen-recorder -w DP-1 -f 144 -o %1"_s.arg(outputPath()));
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestStart)

#include "start_test.moc"
