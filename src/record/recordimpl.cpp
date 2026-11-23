#include <qdatetime.h>
#include <qdeadlinetimer.h>
#include <qdir.h>
#include <qfile.h>
#include <qjsonarray.h>
#include <qjsondocument.h>
#include <qjsonobject.h>
#include <qprocess.h>
#include <qregularexpression.h>
#include <qtextstream.h>
#include <qthread.h>
#include <qurl.h>

#include <caelestia/config/config.hpp>
#include <caelestia/ipc/client.hpp>
#include <caelestia/record/record.hpp>
#include <caelestia/utils/hypr.hpp>

#include <cmath>
#include <optional>
#include <variant>

#include "common/error.hpp"
#include "notify.hpp"
#include "record.hpp"
#include "session.hpp"

namespace caelestia::cli::record {

using Qt::StringLiterals::operator""_s;
namespace recorder = caelestia::record;

namespace {

constexpr int k_startupCheckMs = 500;
constexpr int k_startupPollMs = 50;

QString recordingPath() {
    return config::ConfigPaths::stateDir() + u"/record/recording.mp4"_s;
}

int fail(const QString& summary, const QString& msg, bool quiet, bool json) {
    if (!quiet)
        notifyFailed(summary, msg);
    return common::printError(msg, json);
}

QRect monitorRect(const QJsonObject& monitor) {
    return {
        monitor.value(u"x").toInt(),
        monitor.value(u"y").toInt(),
        monitor.value(u"width").toInt(),
        monitor.value(u"height").toInt(),
    };
}

int refreshRate(const QJsonObject& monitor) {
    return static_cast<int>(std::lround(monitor.value(u"refreshRate").toDouble()));
}

int regionRefreshRate(const QJsonArray& monitors, const QRect& region) {
    int rate = 0;
    for (const auto& value : monitors) {
        const auto monitor = value.toObject();
        if (monitorRect(monitor).intersects(region))
            rate = std::max(rate, refreshRate(monitor));
    }
    return rate;
}

std::optional<QStringList> extraArgs() {
    const auto value = config::Config::instance().value(u"record.extraArgs");
    if (value.isUndefined())
        return QStringList{};
    if (!value.isArray())
        return std::nullopt;

    QStringList args;
    const auto array = value.toArray();
    for (const auto& arg : array)
        args << arg.toString();
    return args;
}

std::variant<QRect, int> getRegion(const StartArgs& args, bool json) {
    if (!args.geometry.isEmpty()) {
        const auto region = common::parseGeometry(args.geometry);
        if (!region || region->size().isEmpty())
            return common::printError(u"Invalid geometry: %1"_s.arg(args.geometry), json);
        return *region;
    }

    const auto res = ipc::pickRegion();
    if (const auto* err = std::get_if<ipc::Error>(&res)) {
        if (*err == ipc::Error::Cancelled)
            return 0;

        return fail(u"Failed to start recording"_s,
            u"Failed to select a region: %1"_s.arg(common::messageForError(*err)), args.quiet, json);
    }

    return std::get<QRect>(res);
}

std::variant<recorder::Options, int> resolveOptions(const StartArgs& args, const parser::Context& context) {
    const auto extra = extraArgs();
    if (!extra)
        return common::printError(u"Config option 'record.extraArgs' should be an array"_s, context.json);

    recorder::Options options{
        .output = recordingPath(),
        .monitor = {},
        .region = {},
        .fps = 0,
        .audio = args.sound,
        .extraArgs = *extra,
    };

    const auto monitors = utils::hypr::requestJson(u"monitors"_s).array();
    if (args.target == common::Target::Region) {
        const auto picked = getRegion(args, context.json);
        if (const auto* exit = std::get_if<int>(&picked))
            return *exit;

        const auto& region = std::get<QRect>(picked);
        options.region = u"%1x%2+%3+%4"_s.arg(region.width()).arg(region.height()).arg(region.x()).arg(region.y());
        options.fps = regionRefreshRate(monitors, region);
        return options;
    }

    for (const auto& value : monitors) {
        const auto monitor = value.toObject();
        if (monitor.value(u"focused").toBool()) {
            options.monitor = monitor.value(u"name").toString();
            options.fps = refreshRate(monitor);
            break;
        }
    }
    return options;
}

void copyToClipboard(const QString& path) {
    QProcess proc;
    proc.setProcessChannelMode(QProcess::ForwardedChannels);
    proc.start(u"wl-copy"_s, { u"--type"_s, u"text/uri-list"_s });
    if (!proc.waitForStarted())
        return;

    proc.write(QUrl::fromLocalFile(path).toString().toUtf8() + '\n');
    proc.closeWriteChannel();
    proc.waitForFinished();
}

bool survivesStartup() {
    const QDeadlineTimer deadline(k_startupCheckMs);
    while (!deadline.hasExpired()) {
        if (!recorder::state().running)
            return false;

        QThread::msleep(k_startupPollMs);
    }

    return recorder::state().running;
}

int stopRecording(bool quiet, bool clipboard, const parser::Context& context) {
    const auto session = Session::load();
    quiet = quiet || session.quiet;

    const auto output = recorder::stop();
    if (!output)
        return fail(u"Failed to stop recording"_s, u"Failed to stop the recorder"_s, quiet, context.json);

    session.end();

    const auto dir = config::Config::instance().paths().recordingsDir();
    QDir().mkpath(dir);
    const auto dest = u"%1/recording_%2.mp4"_s.arg(dir, QDateTime::currentDateTime().toString(u"yyyyMMdd_HH-mm-ss"));
    if (!QFile::rename(*output, dest))
        return fail(u"Failed to stop recording"_s, u"Failed to move recording to %1"_s.arg(dest), quiet, context.json);

    if (clipboard || session.clipboard)
        copyToClipboard(dest);
    if (!quiet)
        notifyStopped(dest);

    if (context.json)
        QTextStream(stdout) << QJsonDocument(QJsonObject{ { u"path"_s, dest } }).toJson(QJsonDocument::Indented)
                            << u'\n';
    else
        QTextStream(stdout) << dest << u'\n';
    return 0;
}

} // namespace

int start(const StartArgs& args, const parser::Context& context) {
    if (recorder::state().running)
        return common::printError(u"A recording is already running"_s, context.json);

    const auto resolved = resolveOptions(args, context);
    if (const auto* exit = std::get_if<int>(&resolved))
        return *exit;

    const auto summary = u"Failed to start recording"_s;
    if (!recorder::start(std::get<recorder::Options>(resolved)))
        return fail(summary, u"Failed to start the recorder"_s, args.quiet, context.json);

    const Session session{
        .notificationId = args.quiet ? 0 : notifyStarted(),
        .clipboard = args.clipboard,
        .quiet = args.quiet,
    };
    session.save();

    if (survivesStartup())
        return 0;

    session.end();
    return fail(summary, u"The recorder exited unexpectedly"_s, args.quiet, context.json);
}

int toggle(const StartArgs& args, const parser::Context& context) {
    if (recorder::state().running)
        return stopRecording(args.quiet, args.clipboard, context);
    return start(args, context);
}

int stop(const QuietArgs& args, const parser::Context& context) {
    if (!recorder::state().running)
        return common::printError(u"No recording is running"_s, context.json);
    return stopRecording(args.quiet, false, context);
}

int pause(const QuietArgs& args, const parser::Context& context) {
    if (!recorder::state().running)
        return common::printError(u"No recording is running"_s, context.json);
    if (!recorder::pause())
        return fail(u"Failed to pause recording"_s, u"Failed to pause the recorder"_s, args.quiet, context.json);
    return 0;
}

int resume(const QuietArgs& args, const parser::Context& context) {
    if (!recorder::state().running)
        return common::printError(u"No recording is running"_s, context.json);
    if (!recorder::resume())
        return fail(u"Failed to resume recording"_s, u"Failed to resume the recorder"_s, args.quiet, context.json);
    return 0;
}

int togglePause(const QuietArgs& args, const parser::Context& context) {
    const auto state = recorder::state();
    if (!state.running)
        return common::printError(u"No recording is running"_s, context.json);
    if (state.paused)
        return resume(args, context);
    return pause(args, context);
}

} // namespace caelestia::cli::record
