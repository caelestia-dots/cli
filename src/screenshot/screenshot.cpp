#include "screenshot.hpp"

#include <qdir.h>
#include <qeventloop.h>
#include <qfile.h>
#include <qfileinfo.h>
#include <qfuturewatcher.h>
#include <qguiapplication.h>
#include <qjsonarray.h>
#include <qjsondocument.h>
#include <qjsonobject.h>
#include <qscreen.h>

#include <caelestia/ipc/client.hpp>
#include <caelestia/screenshot/screenshot.hpp>
#include <caelestia/utils/hypr.hpp>

#include <algorithm>
#include <optional>

#include "common/notify.hpp"
#include "common/target.hpp"
#include "parser/node.hpp"

namespace caelestia::cli::screenshot {

using Qt::StringLiterals::operator""_s;
namespace lib = caelestia::screenshot;

namespace {

struct Args {
    common::Target target = common::Target::Screen;
    QString geometry;
    bool edit = false;
    QString savePath;
    bool quiet = false;
};

int printError(const QString& err, bool json) {
    if (json)
        QTextStream(stdout) << QJsonDocument(QJsonObject{ { u"error"_s, err } }).toJson(QJsonDocument::Indented)
                            << u'\n';
    else
        QTextStream(stderr) << err << u'\n';
    return 1;
}

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

void notify(const QString& cachePath, const QString& savePath, bool ok) {
    QStringList actions;
    actions << u"Edit"_s;
    if (!ok || savePath.isEmpty())
        actions << u"Save"_s;

    const auto path = QFileInfo(savePath.isEmpty() ? cachePath : savePath).absoluteFilePath();
    QVariantMap hints;
    // TODO: replace "helper" with helper bin. edit should open the shell annotation tool, save should open the
    // file dialog
    hints.insert(u"x-caelestia-action-edit"_s, QStringList{ u"helper"_s, u"edit"_s, path });
    if (!ok || savePath.isEmpty())
        hints.insert(u"x-caelestia-action-save"_s, QStringList{ u"helper"_s, u"save"_s, path });

    common::notify::send({
        .summary = ok ? u"Screenshot taken"_s : u"Failed to save screenshot"_s,
        .body = ok ? u"Saved to %1"_s.arg(savePath.isEmpty() ? cachePath : savePath)
                   : u"Failed to save screenshot to %1. The screenshot is available at %2."_s.arg(savePath, cachePath),
        .image = cachePath,
        .actions = actions,
        .hints = hints,
    });
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

int captureLocal(const Args& args, const parser::Context& context) {
    const auto img = doLocalCapture(args);
    if (!img) {
        printError(u"Invalid region geometry: %1"_s.arg(args.geometry), context.json);
        return 1;
    }

    if (img->isNull()) {
        const auto target = args.target == common::Target::Screen ? u"screen"_s : u"region"_s;
        printError(u"Failed to capture %1"_s.arg(target), context.json);

        if (!args.quiet)
            common::notify::send({
                .summary = u"Failed to take screenshot"_s,
                .body = u"Failed to capture %1"_s.arg(target),
                .urgency = common::notify::Urgency::Critical,
            });

        return 1;
    }

    const auto cachePath = lib::cache(*img).takeResult();
    auto ok = true;

    if (!args.savePath.isEmpty())
        if (!QFileInfo(args.savePath).dir().mkpath(u"."_s) || !img->save(args.savePath)) {
            printError(u"Failed to save screenshot to %1"_s.arg(args.savePath), context.json);
            ok = false;
        }

    if (!args.quiet)
        notify(cachePath, args.savePath, ok);

    return ok ? 0 : 1;
}

QString messageForError(ipc::Error err) {
    switch (err) {
    case ipc::Error::Busy:
        return u"Already attempting to take a screenshot"_s;
    case ipc::Error::Cancelled:
        return u"Screenshot cancelled"_s;
    case ipc::Error::NotRunning:
        return u"The shell is not currently running"_s;
    case ipc::Error::Failed:
        return u"An unknown error occurred"_s;
    }
    Q_UNREACHABLE_RETURN({});
}

int saveAndNotify(const ipc::Result<QString>& scResult, const Args& args, const parser::Context& context) {
    if (const auto* err = std::get_if<ipc::Error>(&scResult)) {
        const auto msg = messageForError(*err);
        printError(u"Failed to take screenshot: %1"_s.arg(msg), context.json);

        if (!args.quiet)
            common::notify::send({
                .summary = u"Failed to take screenshot"_s,
                .body = msg,
                .urgency = common::notify::Urgency::Critical,
            });

        return 1;
    }

    const auto cachePath = std::get<QString>(scResult);
    auto ok = true;

    if (!args.savePath.isEmpty())
        if (!QFileInfo(args.savePath).dir().mkpath(u"."_s) || !QFile::copy(cachePath, args.savePath)) {
            printError(u"Failed to save screenshot to %1"_s.arg(args.savePath), context.json);
            ok = false;
        }

    if (!args.quiet)
        notify(cachePath, args.savePath, ok);

    return ok ? 0 : 1;
}

int captureScreen(const Args& args, const parser::Context& context) {
    if (!args.edit)
        return captureLocal(args, context);

    const auto scResult = ipc::screenshot(false, true);
    return saveAndNotify(scResult, args, context);
}

int captureAndAnnotate(const Args& args, const parser::Context& context) {
    // TODO: ipc annotate call
    return 0;
}

int captureRegion(const Args& args, const parser::Context& context) {
    if (!args.geometry.isEmpty())
        return args.edit ? captureAndAnnotate(args, context) : captureLocal(args, context);

    const auto scResult = ipc::screenshot(true, args.edit);
    return saveAndNotify(scResult, args, context);
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
