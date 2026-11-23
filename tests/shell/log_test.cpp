#include "fakes.hpp"

using namespace caelestia::cli::test;

namespace {

class TestLog : public QObject, protected ShellFakes {
    Q_OBJECT

private slots:

    void initTestCase() { setupShellEnvironment(); }

    void init() { resetFakes(); }

    void logForwardsOutput() {
        const auto result = run({ u"log"_s, u"--log-rules"_s, u"foo"_s });
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.out, u"line one\nline two"_s);
        QCOMPARE(log(), u"qs -c caelestia log -r foo\n"_s);
    }

    void logFailsWhenQsMissing() {
        hideQs();
        const auto result = run({ u"log"_s }, true);
        QCOMPARE(result.exit, 1);
        QVERIFY(result.json().contains(u"error"_s));
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestLog)

#include "log_test.moc"
