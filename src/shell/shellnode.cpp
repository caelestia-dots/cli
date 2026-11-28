#include <optional>

#include "shell.hpp"

namespace caelestia::cli::shell {

using Qt::StringLiterals::operator""_s;

namespace {

template <typename T> parser::Option<T> logRulesOption(std::optional<QString> T::* member) {
    return parser::value(member, {}, u"log-rules"_s, u"RULES"_s, u"log rules to apply"_s);
}

parser::NodePtr startLike(QString name, QString help, int (*run)(const StartArgs&, const parser::Context&)) {
    return parser::leaf<StartArgs>({
        .name = std::move(name),
        .help = std::move(help),
        .options = {
            parser::flag(&StartArgs::daemon, u'd', u"daemon"_s, u"start the shell detached"_s),
            logRulesOption(&StartArgs::logRules),
        },
        .run = run,
    });
}

} // namespace

parser::NodePtr parserNode() {
    return parser::group({
        .name = u"shell"_s,
        .help = u"manage the shell"_s,
        .children = {
            startLike(u"start"_s, u"start the shell"_s, start),
            startLike(u"restart"_s, u"kill and restart the shell"_s, restart),
            parser::leaf<parser::NoArgs>({
                .name = u"stop"_s,
                .help = u"kill the shell"_s,
                .run = [](const parser::NoArgs&, const parser::Context& context) { return stop(context); },
            }),
            parser::leaf<LogArgs>({
                .name = u"log"_s,
                .help = u"print the shell log"_s,
                .options = {
                    logRulesOption(&LogArgs::logRules),
                },
                .run = log,
            }),
            parser::leaf<IpcArgs>({
                .name = u"ipc"_s,
                .help = u"send a message to the shell, or list IPC commands"_s,
                .positionals = {
                    parser::rest(&IpcArgs::message, u"MESSAGE"_s, u"the message to send"_s),
                },
                .run = ipc,
            }),
        },
    });
}

} // namespace caelestia::cli::shell
