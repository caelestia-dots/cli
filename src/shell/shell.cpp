#include "shell.hpp"

#include <optional>

#include "parser/node.hpp"

namespace caelestia::cli::shell {

using Qt::StringLiterals::operator""_s;

namespace {

struct StartArgs {
    bool daemon = false;
    std::optional<QString> logRules;
};

struct LogArgs {
    std::optional<QString> logRules;
};

struct IpcArgs {
    QStringList message;
};

parser::Option<StartArgs> logRulesOption() {
    return parser::value(&StartArgs::logRules, {}, u"log-rules"_s, u"RULES"_s, u"log rules to apply"_s);
}

parser::NodePtr startLike(QString name, QString help) {
    return parser::leaf<StartArgs>({
        .name = std::move(name),
        .help = std::move(help),
        .options = {
            parser::flag(&StartArgs::daemon, u'd', u"daemon"_s, u"start the shell detached"_s),
            logRulesOption(),
        },
        .run = [](const StartArgs&, const parser::Context& context) { return parser::notImplemented(context); },
    });
}

} // namespace

parser::NodePtr parserNode() {
    return parser::group({
        .name = u"shell"_s,
        .help = u"manage the shell"_s,
        .children = {
            startLike(u"start"_s, u"start the shell"_s),
            startLike(u"restart"_s, u"kill and restart the shell"_s),
            parser::leaf<parser::NoArgs>({
                .name = u"stop"_s,
                .help = u"kill the shell"_s,
                .run = [](const parser::NoArgs&, const parser::Context& context) { return parser::notImplemented(context); },
            }),
            parser::leaf<LogArgs>({
                .name = u"log"_s,
                .help = u"print the shell log"_s,
                .options = {
                    parser::value(&LogArgs::logRules, {}, u"log-rules"_s, u"RULES"_s, u"log rules to apply"_s),
                },
                .run = [](const LogArgs&, const parser::Context& context) { return parser::notImplemented(context); },
            }),
            parser::leaf<IpcArgs>({
                .name = u"ipc"_s,
                .help = u"send a message to the shell, or list IPC commands"_s,
                .positionals = {
                    parser::rest(&IpcArgs::message, u"MESSAGE"_s, u"the message to send"_s),
                },
                .run = [](const IpcArgs&, const parser::Context& context) { return parser::notImplemented(context); },
            }),
        },
    });
}

} // namespace caelestia::cli::shell
