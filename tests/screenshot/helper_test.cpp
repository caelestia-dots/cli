#include <qprocess.h>

#include "fakes.hpp"

using namespace caelestia::cli::test;
namespace ipc = caelestia::ipc;

namespace {

class TestHelper : public QObject, protected ScreenshotFakes {
    Q_OBJECT

private:
    struct Result {
        int exit = -1;
        QString err;
    };

    [[nodiscard]] static Result helper(const QStringList& args) {
        QProcess proc;
        proc.start(QStringLiteral(SCREENSHOT_HELPER_BIN), args);
        if (!QTest::qWaitFor(
                [&proc] {
                    return proc.state() == QProcess::NotRunning;
                },
                5000))
            return {};
        return { .exit = proc.exitCode(), .err = QString::fromUtf8(proc.readAllStandardError()) };
    }

private slots:

    void initTestCase() { setupDesktopEnvironment(); }

    void init() { resetFakes(); }

    void rejectsBadUsage_data() {
        QTest::addColumn<QStringList>("args");
        QTest::newRow("no arguments") << QStringList();
        QTest::newRow("missing path") << QStringList{ u"edit"_s };
        QTest::newRow("unknown action") << QStringList{ u"open"_s, shotPath() };
    }

    void rejectsBadUsage() {
        QFETCH(const QStringList, args);

        const auto result = helper(args);
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.err, u"Usage: %1 <edit|save> <path>\n"_s.arg(QStringLiteral(SCREENSHOT_HELPER_BIN)));
        QVERIFY(log().isEmpty());
    }

    void saveAsksWhereToSave() {
        const auto dest = outPath(u"a.png"_s);
        const FakeShell shell({ .pickFile = dest }, logPath());
        QCOMPARE(helper({ u"save"_s, shotPath() }).exit, 0);
        QCOMPARE(readFile(dest), QLatin1StringView(k_shot));
    }

    void editAnnotates() {
        const FakeShell shell({ .annotate = ipc::SaveResult{ .file = shotPath(), .copied = true } }, logPath());
        QCOMPARE(helper({ u"edit"_s, shotPath() }).exit, 0);
        QVERIFY(log().startsWith(u"annotate %1\n"_s.arg(shotPath())));
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestHelper)

#include "helper_test.moc"
