#include <sys/stat.h>

#include "common/fakebus.hpp"
#include "common.hpp"

using namespace caelestia::cli::test;

namespace {

class TestReveal : public QObject, protected Sandbox {
    Q_OBJECT

private:
    QTemporaryDir m_bin;
    FakeBus m_bus;
    std::unique_ptr<FakeFileManager> m_fileManager;

    [[nodiscard]] QString logPath() const { return m_bin.filePath(u"log"_s); }

    [[nodiscard]] QString log() const { return readFile(logPath()); }

    [[nodiscard]] static int openInFileManager(const QString& target) {
        QProcess proc;
        proc.start(QStringLiteral(OPEN_IN_FILE_MANAGER_BIN), { target });
        if (!QTest::qWaitFor(
                [&proc] {
                    return proc.state() == QProcess::NotRunning;
                },
                5000))
            return -1;
        return proc.exitCode();
    }

private slots:

    void initTestCase() {
        setupEnvironment();
        QVERIFY(m_bin.isValid());
        QVERIFY(QDir().mkpath(path(u"runtime"_s)));
        QVERIFY(m_bus.start(path(u"runtime/bus"_s)));

        const auto xdgOpen = m_bin.filePath(u"xdg-open"_s);
        writeFile(xdgOpen, "#!/bin/sh\necho \"xdg-open $*\" >> \"$FAKE_LOG\"\n");
        QCOMPARE(chmod(xdgOpen.toUtf8().constData(), 0755), 0);
        qputenv("PATH", m_bin.path().toUtf8());
        qputenv("FAKE_LOG", logPath().toUtf8());
    }

    void init() { QFile::remove(logPath()); }

    void fallsBackToOpeningDirectory() {
        QCOMPARE(openInFileManager(path(u"Videos/clip.mp4"_s)), 0);
        QCOMPARE(log(), u"xdg-open %1\n"_s.arg(path(u"Videos"_s)));
    }

    void showsItemInFileManager() {
        m_fileManager = std::make_unique<FakeFileManager>(logPath());
        QVERIFY(
            FakeBus::serve(u"org.freedesktop.FileManager1"_s, u"/org/freedesktop/FileManager1"_s, m_fileManager.get()));
        QCOMPARE(openInFileManager(path(u"Videos/clip.mp4"_s)), 0);
        QCOMPARE(log(), u"ShowItems file://%1\n"_s.arg(path(u"Videos/clip.mp4"_s)));
    }

    void rejectsMissingArgument() {
        QProcess proc;
        proc.start(QStringLiteral(OPEN_IN_FILE_MANAGER_BIN), {});
        QVERIFY(proc.waitForFinished());
        QCOMPARE(proc.exitCode(), 1);
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestReveal)

#include "open_in_file_manager_test.moc"
