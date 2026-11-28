#include <qdeadlinetimer.h>
#include <qjsonarray.h>
#include <qjsondocument.h>
#include <qjsonobject.h>
#include <qprocess.h>
#include <qstandardpaths.h>
#include <qtextstream.h>
#include <qthread.h>

#include <sys/types.h>

#include <csignal> // IWYU pragma: keep (for ::kill)
#include <limits>
#include <optional>
#include <utility>

#include "common/error.hpp"
#include "shell.hpp"

namespace caelestia::cli::shell {

using Qt::StringLiterals::operator""_s;

namespace {

constexpr int k_exitTimeoutMs = 3000;
constexpr int k_killTimeoutMs = 2000;
constexpr int k_pollIntervalMs = 100;

QStringList qsArgs(const QStringList& args) {
    return QStringList{ u"-c"_s, u"caelestia"_s } + args;
}

int run(const QStringList& args, const parser::Context& context) {
    const auto exit = QProcess::execute(u"qs"_s, qsArgs(args));
    if (exit < 0)
        return common::printError(u"Failed to run qs"_s, context.json);
    return exit;
}

std::optional<QList<pid_t>> instancePids(const parser::Context& context) {
    QProcess proc;
    proc.start(u"qs"_s, qsArgs({ u"list"_s, u"-j"_s }), QIODevice::ReadOnly);
    if (!proc.waitForFinished(-1)) {
        common::printError(u"Failed to list shell instances: %1"_s.arg(proc.errorString()), context.json);
        return std::nullopt;
    }

    const auto out = proc.readAllStandardOutput().trimmed();
    if (proc.exitStatus() != QProcess::NormalExit || proc.exitCode() != 0) {
        const auto err = proc.readAllStandardError().trimmed();
        common::printError(
            u"Failed to list shell instances: %1"_s.arg(QString::fromUtf8(err.isEmpty() ? out : err)), context.json);
        return std::nullopt;
    }

    // qs prints a plain text notice instead of JSON when there are no instances
    if (out.startsWith("No running instances"))
        return QList<pid_t>{};

    const auto parseError = [&] {
        common::printError(u"Failed to parse shell instance list: %1"_s.arg(QString::fromUtf8(out)), context.json);
        return std::nullopt;
    };

    QJsonParseError jsonError;
    const auto doc = QJsonDocument::fromJson(out, &jsonError);
    if (jsonError.error != QJsonParseError::NoError || !doc.isArray())
        return parseError();

    QList<pid_t> pids;
    const auto instances = doc.array();
    for (const auto& instance : instances) {
        const auto pid = instance.toObject().value(u"pid").toInteger();
        // 0 and negative pids address process groups when signalled
        if (pid <= 0 || pid > std::numeric_limits<pid_t>::max())
            return parseError();
        pids << static_cast<pid_t>(pid);
    }
    return pids;
}

std::optional<QList<pid_t>> remainingAfter(int timeoutMs, const parser::Context& context) {
    const QDeadlineTimer deadline(timeoutMs);
    while (true) {
        auto pids = instancePids(context);
        if (!pids || pids->isEmpty() || deadline.hasExpired())
            return pids;

        QThread::msleep(k_pollIntervalMs);
    }
}

} // namespace

int start(const StartArgs& args, const parser::Context& context) {
    QStringList qs{ u"-n"_s };
    if (args.logRules)
        qs << u"--log-rules"_s << *args.logRules;

    if (args.daemon)
        qs << u"-d"_s;
    return run(qs, context);
}

int restart(const StartArgs& args, const parser::Context& context) {
    // qs -n silently skips the launch while an old instance is still alive
    if (const auto exit = stop(context); exit != 0)
        return exit;
    return start(args, context);
}

int stop(const parser::Context& context) {
    const auto pids = instancePids(context);
    if (!pids)
        return 1;

    // Without --pid, qs kill only kills one instance
    for (const auto pid : *pids) {
        QProcess kill;
        kill.setProcessChannelMode(QProcess::ForwardedErrorChannel);
        kill.setStandardOutputFile(QProcess::nullDevice());
        kill.start(u"qs"_s, qsArgs({ u"kill"_s, u"--pid"_s, QString::number(pid) }));
        kill.waitForFinished(-1);
    }

    auto remaining = remainingAfter(k_exitTimeoutMs, context);
    if (!remaining)
        return 1;
    if (remaining->isEmpty())
        return 0;

    QTextStream(stderr) << u"Shell did not exit gracefully, killing\n"_s;
    for (const auto pid : std::as_const(*remaining))
        ::kill(pid, SIGKILL);

    remaining = remainingAfter(k_killTimeoutMs, context);
    if (!remaining)
        return 1;
    if (!remaining->isEmpty())
        return common::printError(u"An instance of the shell is still running"_s, context.json);
    return 0;
}

int log(const LogArgs& args, const parser::Context& context) {
    QStringList qs{ u"log"_s };
    if (args.logRules)
        qs << u"-r"_s << *args.logRules;
    return run(qs, context);
}

int ipc(const IpcArgs& args, const parser::Context& context) {
    QStringList qs{ u"ipc"_s };
    if (args.message.isEmpty() || args.message.first() == u"show") {
        qs << u"show"_s;
    } else if (args.message.first() == u"call") {
        qs += args.message;
    } else {
        qs << u"call"_s;
        qs += args.message;
    }
    return run(qs, context);
}

} // namespace caelestia::cli::shell
