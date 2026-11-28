#include <qscreen.h>

#include "fakes.hpp"

using namespace caelestia::cli::test;
namespace ipc = caelestia::ipc;

namespace {

constexpr auto k_pickedRegion = "screenshot 0,0 0x0 annotate=false\n";

class TestEdit : public QObject, protected ScreenshotFakes {
    Q_OBJECT

private:
    [[nodiscard]] ShellReplies saved() const { return { .screenshot = ipc::SaveResult{ .file = shotPath() } }; }

private slots:

    void initTestCase() { setupDesktopEnvironment(); }

    void init() { resetFakes(); }

    void editsFocusedScreen() {
        const FakeShell shell({ .screenshot = ipc::SaveResult{ .file = shotPath(), .copied = true } }, logPath());
        const auto result = run({ u"screen"_s, u"-e"_s });
        QCOMPARE(result.exit, 0);
        QVERIFY(result.out.isEmpty());
        QVERIFY(result.err.isEmpty());

        const auto screen = QGuiApplication::primaryScreen()->geometry();
        QCOMPARE(log(), u"screenshot %1,%2 %3x%4 annotate=true\n"_s.arg(screen.x())
                                .arg(screen.y())
                                .arg(screen.width())
                                .arg(screen.height()) +
                            notification(u"Screenshot taken"_s, u"Copied to clipboard"_s, shotPath(),
                                { u"edit"_s, u"save"_s }, shotPath()));
    }

    void editsFixedRegion() {
        const FakeShell shell(saved(), logPath());
        QCOMPARE(run({ u"region"_s, u"-e"_s, u"-g"_s, u"10,20 300x200"_s, u"-s"_s, outPath(u"a.png"_s) }).exit, 0);
        QVERIFY(log().startsWith(u"screenshot 10,20 300x200 annotate=true\n"_s));
    }

    void rejectsInvalidGeometry() {
        const FakeShell shell(saved(), logPath());
        const auto result = run({ u"region"_s, u"-e"_s, u"-g"_s, u"nonsense"_s }, true);
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.json().value(u"error"_s).toString(), u"Invalid region geometry: nonsense"_s);
        QVERIFY(log().isEmpty());
    }

    void picksRegion() {
        const FakeShell shell(saved(), logPath());
        QCOMPARE(run({ u"region"_s }).exit, 0);
        QCOMPARE(log(),
            QLatin1StringView(k_pickedRegion) + notification(u"Screenshot taken"_s, u"Saved to %1"_s.arg(shotPath()),
                                                    shotPath(), { u"edit"_s, u"save"_s }, shotPath()));
    }

    void cancelExitsQuietly() {
        const FakeShell shell(ShellReplies(), logPath());
        const auto result = run({ u"region"_s });
        QCOMPARE(result.exit, 0);
        QVERIFY(result.out.isEmpty());
        QVERIFY(result.err.isEmpty());
        QCOMPARE(log(), QLatin1StringView(k_pickedRegion));
    }

    void failsWhenShellNotRunning() {
        const auto result = run({ u"region"_s });
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.err, u"Failed to take screenshot: The shell is not currently running\n"_s);
        QCOMPARE(log(), failure(u"Failed to take screenshot"_s, u"The shell is not currently running"_s));
    }

    void reportsShellError_data() {
        QTest::addColumn<ipc::Error>("error");
        QTest::addColumn<QString>("message");
        QTest::newRow("busy") << ipc::Error::Busy << u"The shell is currently busy"_s;
        QTest::newRow("failed") << ipc::Error::Failed << u"An unknown error occurred"_s;
    }

    void reportsShellError() {
        QFETCH(const ipc::Error, error);
        QFETCH(const QString, message);

        const FakeShell shell({ .screenshot = error }, logPath());
        const auto result = run({ u"region"_s }, true);
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.json().value(u"error"_s).toString(), u"Failed to take screenshot: %1"_s.arg(message));
        QCOMPARE(log(), QLatin1StringView(k_pickedRegion) + failure(u"Failed to take screenshot"_s, message));
    }

    void quietSendsNoNotification() {
        {
            const FakeShell shell(saved(), logPath());
            QCOMPARE(run({ u"region"_s, u"-q"_s }).exit, 0);
        }
        {
            const FakeShell shell({ .screenshot = ipc::Error::Failed }, logPath());
            QCOMPARE(run({ u"region"_s, u"-q"_s }).exit, 1);
        }
        QCOMPARE(log(), QLatin1StringView(k_pickedRegion) + QLatin1StringView(k_pickedRegion));
    }

    void savesToPath() {
        const auto dest = outPath(u"nested/a.png"_s);
        const FakeShell shell(saved(), logPath());
        QCOMPARE(run({ u"region"_s, u"-s"_s, dest }).exit, 0);
        QCOMPARE(readFile(dest), QLatin1StringView(k_shot));
        QCOMPARE(log(),
            QLatin1StringView(k_pickedRegion) +
                notification(u"Screenshot taken"_s, u"Saved to %1"_s.arg(dest), shotPath(), { u"edit"_s }, dest));
    }

    void reportsSaveFailure() {
        const auto dest = shotPath() + u"/a.png"_s;
        const FakeShell shell(saved(), logPath());
        const auto result = run({ u"region"_s, u"-s"_s, dest });
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.err, u"Failed to save screenshot to %1\n"_s.arg(dest));
        QCOMPARE(log(),
            QLatin1StringView(k_pickedRegion) +
                notification(u"Failed to save screenshot"_s,
                    u"Failed to save screenshot to %1. The screenshot is available at %2."_s.arg(dest, shotPath()),
                    shotPath(), { u"edit"_s, u"save"_s }, shotPath()));
    }

    void copiesToClipboard() {
        const FakeShell shell(saved(), logPath());
        QCOMPARE(run({ u"region"_s, u"-c"_s, u"-q"_s }).exit, 0);
        QTRY_COMPARE(log(), QLatin1StringView(k_pickedRegion) + u"wl-copy --type image/png:pixels\n"_s);
    }

    void notifiesSavedAndCopied() {
        const auto dest = outPath(u"a.png"_s);
        const FakeShell shell(saved(), logPath());
        QCOMPARE(run({ u"region"_s, u"-c"_s, u"-s"_s, dest }).exit, 0);
        QVERIFY(log().contains(notification(u"Screenshot taken"_s, u"Saved to %1 and copied to clipboard"_s.arg(dest),
            shotPath(), { u"edit"_s }, dest)));
        QTRY_VERIFY(log().contains(u"pixels"_s));
    }

    void editWithoutPathAsksWhereToSave() {
        const auto dest = outPath(u"picked.png"_s);
        const FakeShell shell({ .screenshot = ipc::SaveResult{ .file = shotPath() }, .pickFile = dest }, logPath());
        QCOMPARE(run({ u"region"_s, u"-e"_s }).exit, 0);
        QCOMPARE(readFile(dest), QLatin1StringView(k_shot));
        QCOMPARE(log(),
            u"screenshot 0,0 0x0 annotate=true\npickFile Save screenshot %1 screenshot.png\n"_s.arg(
                caelestia::config::Config::instance().paths().screenshotsDir()) +
                notification(u"Screenshot taken"_s, u"Saved to %1"_s.arg(dest), shotPath(), { u"edit"_s }, dest));
    }
};

} // namespace

int main(int argc, char* argv[]) {
    return runOffscreen<TestEdit>(argc, argv);
}

#include "edit_test.moc"
