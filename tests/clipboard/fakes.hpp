#pragma once

#include <qtemporarydir.h>

#include <sys/stat.h>

#include "common.hpp"

namespace caelestia::cli::test {

class FakeTools {
protected:
    QTemporaryDir m_bin;

    [[nodiscard]] QString logPath() const { return m_bin.filePath(u"log"_s); }

    [[nodiscard]] QString log() const { return readFile(logPath()); }

    void installTool(const QString& name, const QByteArray& script) {
        const auto path = m_bin.filePath(name);
        writeFile(path, "#!/bin/sh\nPATH=/usr/bin:/bin\n" + script);
        QCOMPARE(chmod(path.toUtf8().constData(), 0755), 0);
    }

    void installTools() {
        QVERIFY(m_bin.isValid());
        qputenv("PATH", m_bin.path().toUtf8());
        qputenv("FAKE_LOG", logPath().toUtf8());
        installTool(u"cliphist"_s, R"(echo "cliphist $*" >> "$FAKE_LOG"
case "$1" in
    list) printf '1\thello world\n2\tsecond entry\n' ;;
    decode) printf 'decoded:'; cat ;;
    delete) printf 'deleted:' >> "$FAKE_LOG"; cat >> "$FAKE_LOG" ;;
esac
)");
        installTool(u"fuzzel"_s, R"(echo "fuzzel $*" >> "$FAKE_LOG"
[ -n "$FUZZEL_EXIT" ] && exit "$FUZZEL_EXIT"
[ -n "$FUZZEL_OUT" ] && { printf '%s' "$FUZZEL_OUT"; exit 0; }
head -n 1
)");
        installTool(u"wl-copy"_s, R"(printf 'wl-copy:' >> "$FAKE_LOG"; cat >> "$FAKE_LOG"
)");
    }

    void resetTools() {
        QFile::remove(logPath());
        qunsetenv("FUZZEL_EXIT");
        qunsetenv("FUZZEL_OUT");
    }
};

} // namespace caelestia::cli::test
