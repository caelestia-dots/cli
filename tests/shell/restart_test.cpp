#include "fakes.hpp"

using namespace caelestia::cli::test;

namespace {

class TestRestart : public QObject, protected ShellFakes {
    Q_OBJECT

private slots:

    void initTestCase() { setupShellEnvironment(); }

    void init() { resetFakes(); }

    void restartStopsThenStarts() {
        writeFile(instancesPath(), k_instances);
        QCOMPARE(run({ u"restart"_s, u"-d"_s }).exit, 0);
        QVERIFY(log().endsWith(u"qs -c caelestia kill --pid 2000000002\n"
                               "qs -c caelestia list -j\n"
                               "qs -c caelestia -n -d\n"_s));
    }

    void restartDoesNotStartWhenStopFails() {
        qputenv("QS_LIST_FAIL", "1");
        QCOMPARE(run({ u"restart"_s }).exit, 1);
        QVERIFY(!log().contains(u"-n"_s));
    }

    void restartFailsWhenQsMissing() {
        hideQs();
        const auto result = run({ u"restart"_s }, true);
        QCOMPARE(result.exit, 1);
        QVERIFY(result.json().contains(u"error"_s));
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestRestart)

#include "restart_test.moc"
