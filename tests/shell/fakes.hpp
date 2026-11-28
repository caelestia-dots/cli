#pragma once

#include <qtemporarydir.h>

#include <sys/stat.h>

#include "shell/shell.hpp"
#include "common.hpp"

namespace caelestia::cli::test {

constexpr auto k_fakeQs = R"(echo "qs $*" >> "$FAKE_LOG"
shift 2
case "$1" in
    list)
        [ -n "$QS_LIST_FAIL" ] && { echo 'boom' >&2; exit 1; }
        if [ -f "$FAKE_INSTANCES" ] && { [ -z "$QS_STUCK_PID" ] || kill -0 "$QS_STUCK_PID" 2> /dev/null; }; then
            cat "$FAKE_INSTANCES"
        else
            echo 'No running instances for "shell.qml"'
        fi
        ;;
    kill) if [ -z "$QS_STUCK_PID" ]; then rm -f "$FAKE_INSTANCES"; fi ;;
    ipc) echo "ipc $2" ;;
    log) printf 'line one\nline two' ;;
    -n) case "$*" in *-d*) ;; *) printf 'line one\nline two' ;; esac ;;
esac
)";

// Above any real pid_max, so a wrongly signalled pid cannot hit a live process
constexpr auto k_instances = R"([{"pid":2000000001},{"pid":2000000002}])";

class ShellFakes {
protected:
    QTemporaryDir m_dir;

    [[nodiscard]] QString logPath() const { return m_dir.filePath(u"log"_s); }

    [[nodiscard]] QString instancesPath() const { return m_dir.filePath(u"instances"_s); }

    [[nodiscard]] QString log() const { return readFile(logPath()); }

    static Output run(const QStringList& args, bool json = false) { return dispatch(shell::parserNode(), args, json); }

    void setupShellEnvironment() {
        QVERIFY(m_dir.isValid());
        const auto qs = m_dir.filePath(u"bin/qs"_s);
        writeFile(qs, QByteArray("#!/bin/sh\nPATH=/usr/bin:/bin\n") + k_fakeQs);
        QCOMPARE(chmod(qs.toUtf8().constData(), 0755), 0);

        qputenv("FAKE_LOG", logPath().toUtf8());
        qputenv("FAKE_INSTANCES", instancesPath().toUtf8());
    }

    void resetFakes() {
        qputenv("PATH", m_dir.filePath(u"bin"_s).toUtf8());
        qunsetenv("QS_LIST_FAIL");
        qunsetenv("QS_STUCK_PID");
        QFile::remove(logPath());
        QFile::remove(instancesPath());
    }

    void hideQs() { qputenv("PATH", m_dir.filePath(u"empty"_s).toUtf8()); }
};

} // namespace caelestia::cli::test
