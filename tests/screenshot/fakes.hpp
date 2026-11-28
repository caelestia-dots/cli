#pragma once

#include <qdir.h>
#include <qfile.h>
#include <qguiapplication.h>

#include "common/desktopfakes.hpp"
#include "common/fakeshell.hpp"
#include "screenshot/notify.hpp"
#include "screenshot/screenshot.hpp"
#include "common.hpp"

namespace caelestia::cli::test {

constexpr auto k_shot = "pixels\n";

class ScreenshotFakes : protected DesktopFakes {
protected:
    [[nodiscard]] QString shotPath() const { return path(u"shot.png"_s); }

    [[nodiscard]] QString outPath(const QString& name) const { return path(u"out/"_s + name); }

    static Output run(const QStringList& args, bool json = false) {
        return dispatch(screenshot::parserNode(), args, json);
    }

#ifdef CAELESTIA_SCREENSHOT_HELPER_BIN
    [[nodiscard]] static QString notification(const QString& summary, const QString& body, const QString& image,
        const QStringList& actions, const QString& target) {
        QStringList labels;
        QStringList hints{ u"image-path=%1"_s.arg(image), u"urgency=1"_s };
        for (const auto& action : actions) {
            labels << action << action.at(0).toUpper() + action.mid(1);
            hints << u"x-caelestia-action-%1=%2 %1 %3"_s.arg(
                action, QStringLiteral(CAELESTIA_SCREENSHOT_HELPER_BIN), target);
        }
        return u"Notify Caelestia %1 %2 [%3] {%4}\n"_s.arg(summary, body, labels.join(u','), hints.join(u';'));
    }
#endif

    [[nodiscard]] static QString failure(const QString& summary, const QString& body) {
        return u"Notify Caelestia %1 %2 [] {urgency=2}\n"_s.arg(summary, body);
    }

    void resetFakes() {
        QFile::remove(logPath());
        QDir(path(u"out"_s)).removeRecursively();
        writeFile(shotPath(), k_shot);
    }
};

template <typename Test> int runOffscreen(int argc, char* argv[]) {
    qputenv("QT_QPA_PLATFORM", "offscreen");
    const QGuiApplication app(argc, argv);
    Test test;
    return QTest::qExec(&test, argc, argv);
}

} // namespace caelestia::cli::test
