#include "screenshot.hpp"

#include <qdir.h>
#include <qeventloop.h>
#include <qfile.h>
#include <qfileinfo.h>
#include <qfuturewatcher.h>
#include <qguiapplication.h>
#include <qjsonarray.h>
#include <qjsonobject.h>
#include <qprocess.h>
#include <qscreen.h>

#include <caelestia/ipc/client.hpp>
#include <caelestia/screenshot/screenshot.hpp>
#include <caelestia/utils/hypr.hpp>

#include <algorithm>
#include <optional>

#include "common/error.hpp"
#include "common/target.hpp"
#include "parser/node.hpp"
#include "screenshot/notify.hpp"

namespace caelestia::cli::screenshot {

using Qt::StringLiterals::operator""_s;
namespace lib = caelestia::screenshot;

namespace {

struct Args {
    common::Target target = common::Target::Screen;
    QString geometry;
    bool edit = false;
    QString savePath;
    bool clipboard = false;
    bool quiet = false;
};

QScreen* focusedScreen() {
    const auto monitors = utils::hypr::requestJson(u"monitors"_s).array();
    const auto it = std::ranges::find_if(monitors, [](const QJsonValue& m) {
        return m.toObject().value(u"focused").toBool();
    });
    if (it == monitors.end())
        return QGuiApplication::primaryScreen();

    const auto name = it->toObject().value(u"name").toString();
    const auto screens = QGuiApplication::screens();
    const auto screen = std::ranges::find_if(screens, [name](const QScreen* s) {
        return s->name() == name;
    });

    return screen == screens.end() ? QGuiApplication::primaryScreen() : *screen;
}

template <typename T> T awaitFuture(QFuture<T> future) {
    if (!future.isFinished()) {
        QEventLoop loop;
        QFutureWatcher<T> watcher;
        QObject::connect(&watcher, &QFutureWatcherBase::finished, &loop, &QEventLoop::quit);
        watcher.setFuture(future);
        loop.exec();
    }
    return future.takeResult();
}

std::optional<QImage> doLocalCapture(const Args& args) {
    if (args.target == common::Target::Screen)
        return awaitFuture(lib::capture(focusedScreen()));

    const auto geom = common::parseGeometry(args.geometry);
    if (!geom || geom->size().isEmpty())
        return {};

    return awaitFuture(lib::capture(*geom));
}

std::pair<int, QString> captureLocal(const Args& args, const parser::Context& context) {
    const auto img = doLocalCapture(args);
    if (!img) {
        common::printError(u"Invalid region geometry: %1"_s.arg(args.geometry), context.json);
        return { 1, {} };
    }

    if (img->isNull()) {
        const auto target = args.target == common::Target::Screen ? u"screen"_s : u"region"_s;
        common::printError(u"Failed to capture %1"_s.arg(target), context.json);

        if (!args.quiet)
            notifyFailed(u"Failed to take screenshot"_s, u"Failed to capture %1"_s.arg(target));

        return { 1, {} };
    }

    const auto cachePath = lib::cache(*img).takeResult();
    auto ok = true;

    if (!args.savePath.isEmpty() && (!QFileInfo(args.savePath).dir().mkpath(u"."_s) || !img->save(args.savePath))) {
        common::printError(u"Failed to save screenshot to %1"_s.arg(args.savePath), context.json);
        ok = false;
    }

    if (!args.quiet)
        notify(cachePath, args.savePath, args.clipboard, ok);

    return { ok ? 0 : 1, cachePath };
}

int saveCopyNotify(const ipc::Result<QString>& scResult, const Args& args, const parser::Context& context) {
    if (const auto* err = std::get_if<ipc::Error>(&scResult)) {
        if (*err == ipc::Error::Cancelled)
            return 0;

        const auto msg = messageForError(*err);
        common::printError(u"Failed to take screenshot: %1"_s.arg(msg), context.json);

        if (!args.quiet)
            notifyFailed(u"Failed to take screenshot"_s, msg);

        return 1;
    }

    const auto cachePath = std::get<QString>(scResult);
    // Empty path means copied to clipboard
    if (cachePath.isEmpty()) {
        if (!args.quiet)
            notifyCopied();
        return 0;
    }

    auto ok = true;

    if (!args.savePath.isEmpty() &&
        (!QFileInfo(args.savePath).dir().mkpath(u"."_s) || !QFile::copy(cachePath, args.savePath))) {
        common::printError(u"Failed to save screenshot to %1"_s.arg(args.savePath), context.json);
        ok = false;
    }

    if (args.clipboard) {
        QProcess proc;
        proc.setProgram(u"wl-copy"_s);
        proc.setArguments({ u"--type"_s, u"image/png"_s });
        proc.setStandardInputFile(cachePath);
        proc.startDetached();
    }

    if (!args.quiet)
        notify(cachePath, args.savePath, args.clipboard, ok);

    return ok ? 0 : 1;
}

int captureScreen(const Args& args, const parser::Context& context) {
    if (!args.edit)
        return captureLocal(args, context).first;

    const auto scResult = ipc::screenshot(false, true);
    return saveCopyNotify(scResult, args, context);
}

int captureAndAnnotate(const Args& args, const parser::Context& context) {
    auto modifiedArgs = args;
    modifiedArgs.savePath.clear();
    modifiedArgs.quiet = true;
    const auto& [code, cachePath] = captureLocal(modifiedArgs, context);

    if (code != 0) {
        if (!args.quiet && !cachePath.isEmpty())
            notify(cachePath, args.savePath, false, false);
        return code;
    }

    return annotateAndNotify(cachePath, args.quiet, context.json);
}

int captureRegion(const Args& args, const parser::Context& context) {
    if (!args.geometry.isEmpty())
        return args.edit ? captureAndAnnotate(args, context) : captureLocal(args, context).first;

    const auto scResult = ipc::screenshot(true, args.edit);
    return saveCopyNotify(scResult, args, context);
}

int run(const Args& args, const parser::Context& context) {
    switch (args.target) {
    case common::Target::Screen:
        return captureScreen(args, context);
    case common::Target::Region:
        return captureRegion(args, context);
    }
    Q_UNREACHABLE_RETURN(1);
}

} // namespace

parser::NodePtr parserNode() {
    return parser::leaf<Args>({
        .name = u"screenshot"_s,
        .help = u"take a screenshot"_s,
        .options = {
            parser::value(&Args::geometry, u'g', u"geometry"_s, u"GEOM"_s, u"capture a fixed region ('X,Y WxH')"_s),
            parser::value(&Args::savePath, u's', u"save"_s, u"PATH"_s, u"save to PATH"_s),
            parser::flag(&Args::edit, u'e', u"edit"_s, u"annotate the captured image"_s),
            parser::flag(&Args::clipboard, u'c', u"clipboard"_s, u"copy to clipboard"_s),
            parser::flag(&Args::quiet, u'q', u"quiet"_s, u"don't send a notification"_s),
        },
        .positionals = {
            parser::positional(&Args::target, u"TARGET"_s, u"what to capture"_s, common::targetChoices()),
        },
        .app = [](int& argc, char** argv) -> std::unique_ptr<QCoreApplication> {
            return std::make_unique<QGuiApplication>(argc, argv);
        },
        .run = run,
    });
}

} // namespace caelestia::cli::screenshot
