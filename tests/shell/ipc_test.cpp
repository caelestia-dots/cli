#include "fakes.hpp"

using namespace caelestia::cli::test;

namespace {

class TestIpc : public QObject, protected ShellFakes {
    Q_OBJECT

private slots:

    void initTestCase() { setupShellEnvironment(); }

    void init() { resetFakes(); }

    void ipcShowsCommandsWithoutMessage() {
        const auto result = run({ u"ipc"_s });
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.out, u"ipc show\n"_s);
        QCOMPARE(log(), u"qs -c caelestia ipc show\n"_s);
    }

    void ipcCallsMessage() {
        const auto result = run({ u"ipc"_s, u"mpris"_s, u"getActive"_s, u"trackTitle"_s });
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.out, u"ipc call\n"_s);
        QCOMPARE(log(), u"qs -c caelestia ipc call mpris getActive trackTitle\n"_s);
    }

    void ipcFailsWhenQsMissing() {
        hideQs();
        const auto result = run({ u"ipc"_s }, true);
        QCOMPARE(result.exit, 1);
        QVERIFY(result.json().contains(u"error"_s));
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestIpc)

#include "ipc_test.moc"
