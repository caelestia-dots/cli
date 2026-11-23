#include "fakes.hpp"

using namespace caelestia::cli::test;
namespace ipc = caelestia::ipc;
namespace screenshot = caelestia::cli::screenshot;

namespace {

class TestSave : public QObject, protected ScreenshotFakes {
    Q_OBJECT

private:
    [[nodiscard]] QString annotatedPath() const { return path(u"annotated.png"_s); }

    [[nodiscard]] static QString pickLine() {
        return u"pickFile Save screenshot %1 screenshot.png\n"_s.arg(
            caelestia::config::Config::instance().paths().screenshotsDir());
    }

    [[nodiscard]] Output pickAndSave(bool quiet = false, bool json = false) const {
        return capture([&] {
            return screenshot::pickAndSave(shotPath(), false, quiet, json);
        });
    }

    [[nodiscard]] Output annotate(bool quiet = false) const {
        return capture([&] {
            return screenshot::annotateAndNotify(shotPath(), quiet, false);
        });
    }

private slots:

    void initTestCase() { setupDesktopEnvironment(); }

    void init() {
        resetFakes();
        writeFile(annotatedPath(), "annotated\n");
    }

    void savesToPickedFile() {
        const auto dest = outPath(u"nested/a.png"_s);
        const FakeShell shell({ .pickFile = dest }, logPath());
        const auto result = pickAndSave();
        QCOMPARE(result.exit, 0);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(readFile(dest), QLatin1StringView(k_shot));
        QCOMPARE(log(), pickLine() + notification(u"Screenshot taken"_s, u"Saved to %1"_s.arg(dest), shotPath(),
                                         { u"edit"_s }, dest));
    }

    void overwritesPickedFile() {
        const auto dest = outPath(u"a.png"_s);
        writeFile(dest, "old\n");
        const FakeShell shell({ .pickFile = dest }, logPath());
        QCOMPARE(pickAndSave().exit, 0);
        QCOMPARE(readFile(dest), QLatin1StringView(k_shot));
    }

    void cancelledPickExitsQuietly() {
        const FakeShell shell(ShellReplies(), logPath());
        const auto result = pickAndSave();
        QCOMPARE(result.exit, 0);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(log(), pickLine());
    }

    void reportsPickError() {
        const auto result = pickAndSave();
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.err, u"Failed to save screenshot: The shell is not currently running\n"_s);
        QCOMPARE(log(), failure(u"Failed to save screenshot"_s, u"The shell is not currently running"_s));
    }

    void reportsCopyFailure() {
        const auto dest = shotPath() + u"/a.png"_s;
        const FakeShell shell({ .pickFile = dest }, logPath());
        const auto result = pickAndSave(false, true);
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.json().value(u"error"_s).toString(), u"Failed to save screenshot to %1"_s.arg(dest));
        QCOMPARE(log(), pickLine() + notification(u"Failed to save screenshot"_s,
                                         u"Failed to save screenshot to %1. The screenshot is available at %2."_s.arg(
                                             dest, shotPath()),
                                         shotPath(), { u"edit"_s, u"save"_s }, shotPath()));
    }

    void quietSendsNoNotification() {
        const FakeShell shell({ .pickFile = shotPath() + u"/a.png"_s }, logPath());
        QCOMPARE(pickAndSave(true).exit, 1);
        QCOMPARE(log(), pickLine());
    }

    void annotatedCopyNotifiesWithoutEdit() {
        const FakeShell shell({ .annotate = ipc::SaveResult{ .file = annotatedPath(), .copied = true } }, logPath());
        QCOMPARE(annotate().exit, 0);
        QCOMPARE(
            log(), u"annotate %1\n"_s.arg(shotPath()) + notification(u"Screenshot taken"_s, u"Copied to clipboard"_s,
                                                            annotatedPath(), { u"save"_s }, annotatedPath()));
    }

    void annotatedImageIsSavedToPickedFile() {
        const auto dest = outPath(u"a.png"_s);
        const FakeShell shell({ .annotate = ipc::SaveResult{ .file = annotatedPath() }, .pickFile = dest }, logPath());
        QCOMPARE(annotate().exit, 0);
        QCOMPARE(readFile(dest), u"annotated\n"_s);
        QCOMPARE(log(),
            u"annotate %1\n"_s.arg(shotPath()) + pickLine() +
                notification(u"Screenshot taken"_s, u"Saved to %1"_s.arg(dest), annotatedPath(), { u"edit"_s }, dest));
    }

    void cancelledAnnotateExitsQuietly() {
        const FakeShell shell(ShellReplies(), logPath());
        const auto result = annotate();
        QCOMPARE(result.exit, 0);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(log(), u"annotate %1\n"_s.arg(shotPath()));
    }

    void reportsAnnotateError() {
        const FakeShell shell({ .annotate = ipc::Error::Failed }, logPath());
        const auto result = annotate();
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.err, u"Failed to annotate screenshot: An unknown error occurred\n"_s);
        QCOMPARE(log(), u"annotate %1\n"_s.arg(shotPath()) +
                            failure(u"Failed to annotate screenshot"_s, u"An unknown error occurred"_s));
    }

    void quietAnnotateErrorSendsNoNotification() {
        const FakeShell shell({ .annotate = ipc::Error::Failed }, logPath());
        QCOMPARE(annotate(true).exit, 1);
        QCOMPARE(log(), u"annotate %1\n"_s.arg(shotPath()));
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestSave)

#include "save_test.moc"
