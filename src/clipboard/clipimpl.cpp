#include <qjsondocument.h>
#include <qjsonobject.h>
#include <qprocess.h>

#include <optional>

#include "clipboard.hpp"

namespace caelestia::cli::clipboard {

using Qt::StringLiterals::operator""_s;

namespace {

int printError(const QString& err, bool json) {
    if (json)
        QTextStream(stdout) << QJsonDocument(QJsonObject{ { u"error"_s, err } }).toJson(QJsonDocument::Indented)
                            << u'\n';
    else
        QTextStream(stderr) << err << u'\n';
    return 1;
}

int run(QProcess& proc, const QString& program, const QStringList& args, const QByteArray& input) {
    proc.start(program, args);
    if (!proc.waitForStarted())
        return -1;

    proc.write(input);
    proc.closeWriteChannel();
    proc.waitForFinished(-1);
    return proc.exitStatus() == QProcess::NormalExit ? proc.exitCode() : -1;
}

std::optional<QByteArray> cliphist(const QStringList& args, const QByteArray& input, const parser::Context& context) {
    QProcess proc;
    proc.setProcessChannelMode(QProcess::ForwardedErrorChannel);
    if (run(proc, u"cliphist"_s, args, input) != 0) {
        printError(u"Failed to run cliphist: %1"_s.arg(proc.errorString()), context.json);
        return std::nullopt;
    }
    return proc.readAllStandardOutput();
}

int choose(const QStringList& fuzzelArgs, const parser::Context& context, QByteArray& chosen) {
    const auto history = cliphist({ u"list"_s }, {}, context);
    if (!history)
        return 1;

    QProcess fuzzel;
    fuzzel.setProcessChannelMode(QProcess::ForwardedErrorChannel);
    const auto exit = run(fuzzel, u"fuzzel"_s, QStringList{ u"--dmenu"_s } + fuzzelArgs, *history);
    if (exit < 0)
        return printError(u"Failed to run fuzzel: %1"_s.arg(fuzzel.errorString()), context.json);
    if (exit != 0)
        return exit;

    chosen = fuzzel.readAllStandardOutput();
    return chosen.trimmed().isEmpty() ? 1 : 0;
}

} // namespace

int pick(const parser::NoArgs& args, const parser::Context& context) {
    Q_UNUSED(args);

    QByteArray chosen;
    if (const auto exit = choose({ u"--placeholder=Type to search clipboard"_s }, context, chosen); exit != 0)
        return exit;

    const auto decoded = cliphist({ u"decode"_s }, chosen, context);
    if (!decoded)
        return 1;

    QProcess wlCopy;
    wlCopy.setProcessChannelMode(QProcess::ForwardedChannels);
    if (run(wlCopy, u"wl-copy"_s, {}, *decoded) != 0)
        return printError(u"Failed to run wl-copy: %1"_s.arg(wlCopy.errorString()), context.json);
    return 0;
}

int remove(const parser::NoArgs& args, const parser::Context& context) {
    Q_UNUSED(args);

    QByteArray chosen;
    const auto exit = choose({ u"--prompt=del > "_s, u"--placeholder=Delete from clipboard"_s }, context, chosen);
    if (exit != 0)
        return exit;

    return cliphist({ u"delete"_s }, chosen, context) ? 0 : 1;
}

} // namespace caelestia::cli::clipboard
