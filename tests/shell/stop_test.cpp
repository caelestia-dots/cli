#include <qprocess.h>

#include "fakes.hpp"

using namespace caelestia::cli::test;

namespace {

class TestStop : public QObject, protected ShellFakes {
    Q_OBJECT

private slots:

    void initTestCase() { setupShellEnvironment(); }

    void init() { resetFakes(); }

    void stopWithoutInstances() {
        QCOMPARE(run({ u"stop"_s }).exit, 0);
        QCOMPARE(log(), u"qs -c caelestia list -j\nqs -c caelestia list -j\n"_s);
    }

    void stopKillsEveryInstance() {
        writeFile(instancesPath(), k_instances);
        const auto result = run({ u"stop"_s });
        QCOMPARE(result.exit, 0);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(log(), u"qs -c caelestia list -j\n"
                        "qs -c caelestia kill --pid 2000000001\n"
                        "qs -c caelestia kill --pid 2000000002\n"
                        "qs -c caelestia list -j\n"_s);
    }

    void stopForceKillsStuckInstance() {
        QProcess spawn;
        spawn.start(u"/bin/sh"_s, { u"-c"_s, u"/bin/sleep 60 > /dev/null 2>&1 & echo $!"_s });
        QVERIFY(spawn.waitForFinished());
        const auto pid = spawn.readAllStandardOutput().trimmed();
        QVERIFY(pid.toInt() > 0);

        qputenv("QS_STUCK_PID", pid);
        writeFile(instancesPath(), "[{\"pid\":" + pid + "}]");

        const auto result = run({ u"stop"_s });
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.err, u"Shell did not exit gracefully, killing\n"_s);
        QVERIFY(!QFileInfo::exists(u"/proc/"_s + QString::fromUtf8(pid)));
    }

    void stopFailsWhenListFails() {
        qputenv("QS_LIST_FAIL", "1");
        const auto result = run({ u"stop"_s }, true);
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.json().value(u"error"_s).toString(), u"Failed to list shell instances: boom"_s);
    }

    void stopRejectsMalformedList_data() {
        QTest::addColumn<QByteArray>("list");
        QTest::newRow("not json") << QByteArray("garbage");
        QTest::newRow("zero pid") << QByteArray(R"([{"pid":0}])");
        QTest::newRow("negative pid") << QByteArray(R"([{"pid":-1}])");
        QTest::newRow("missing pid") << QByteArray(R"([{}])");
    }

    void stopRejectsMalformedList() {
        QFETCH(const QByteArray, list);
        writeFile(instancesPath(), list);
        const auto result = run({ u"stop"_s });
        QCOMPARE(result.exit, 1);
        QVERIFY(result.err.startsWith(u"Failed to parse shell instance list: "_s));
        QCOMPARE(log(), u"qs -c caelestia list -j\n"_s);
    }

    void stopFailsWhenQsMissing() {
        hideQs();
        const auto result = run({ u"stop"_s }, true);
        QCOMPARE(result.exit, 1);
        QVERIFY(result.json().contains(u"error"_s));
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestStop)

#include "stop_test.moc"
