#include "fakes.hpp"

using namespace caelestia::cli::test;

namespace {

class TestStart : public QObject, protected ShellFakes {
    Q_OBJECT

private slots:

    void initTestCase() { setupShellEnvironment(); }

    void init() { resetFakes(); }

    void startForwardsOutput() {
        const auto result = run({ u"start"_s, u"--log-rules"_s, u"foo"_s });
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.out, u"line one\nline two"_s);
        QCOMPARE(log(), u"qs -c caelestia -n --log-rules foo\n"_s);
    }

    void startDaemon() {
        const auto result = run({ u"start"_s, u"-d"_s });
        QCOMPARE(result.exit, 0);
        QVERIFY(result.out.isEmpty());
        QCOMPARE(log(), u"qs -c caelestia -n -d\n"_s);
    }

    void startFailsWhenQsMissing_data() {
        QTest::addColumn<QStringList>("args");
        QTest::newRow("foreground") << QStringList{ u"start"_s };
        QTest::newRow("daemon") << QStringList{ u"start"_s, u"-d"_s };
    }

    void startFailsWhenQsMissing() {
        QFETCH(const QStringList, args);
        hideQs();
        const auto result = run(args, true);
        QCOMPARE(result.exit, 1);
        QVERIFY(result.json().contains(u"error"_s));
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestStart)

#include "start_test.moc"
