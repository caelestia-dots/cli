#include <qcoreapplication.h>
#include <qdir.h>
#include <qfile.h>
#include <qfileinfo.h>
#include <qtextstream.h>

#include <caelestia/config/config.hpp>
#include <caelestia/ipc/client.hpp>

#include "common/error.hpp"
#include "screenshot/notify.hpp"

using Qt::StringLiterals::operator""_s;
namespace screenshot = caelestia::cli::screenshot;
namespace ipc = caelestia::ipc;
namespace common = caelestia::cli::common;

namespace {

int save(const QString& cachePath) {
    const auto result = ipc::pickFile({
        .title = u"Save screenshot"_s,
        .nameFilters = { u"*.png"_s },
        .filter = ipc::FileFilter::Images | ipc::FileFilter::Files,
        .initialDir = caelestia::config::Config::instance().paths().screenshotsDir(),
        .initialName = u"screenshot.png"_s,
        .mode = ipc::PickFileMode::Save,
    });

    if (const auto* err = std::get_if<ipc::Error>(&result)) {
        if (*err == ipc::Error::Cancelled)
            return 0;

        const auto msg = screenshot::messageForError(*err);
        common::printError(u"Failed to save screenshot: %1"_s.arg(msg), false);
        screenshot::notifyFailed(u"Failed to save screenshot"_s, msg);

        return 1;
    }

    const auto savePath = *std::get_if<QString>(&result);
    const auto ok = QFileInfo(savePath).dir().mkpath(u"."_s) && (!QFile::exists(savePath) || QFile::remove(savePath)) &&
                    QFile::copy(cachePath, savePath);
    if (!ok)
        common::printError(u"Failed to save screenshot to %1"_s.arg(savePath), false);

    screenshot::notify(cachePath, savePath, false, ok);

    return ok ? 0 : 1;
}

} // namespace

int main(int argc, char* argv[]) {
    const QCoreApplication app(argc, argv);
    const auto args = QCoreApplication::arguments();

    if (args.size() != 3 || (args[1] != u"edit" && args[1] != u"save")) {
        QTextStream(stderr) << u"Usage: %1 <edit|save> <path>\n"_s.arg(args.value(0, u"caelestia-screenshot-helper"_s));
        return 1;
    }

    return args[1] == u"edit" ? screenshot::annotateAndNotify(args[2], false, false) : save(args[2]);
}
