#pragma once

#include "parser/node.hpp"

namespace caelestia::cli::shell {

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

parser::NodePtr parserNode();

int start(const StartArgs& args, const parser::Context& context);
int restart(const StartArgs& args, const parser::Context& context);
int stop(const parser::Context& context);
int log(const LogArgs& args, const parser::Context& context);
int ipc(const IpcArgs& args, const parser::Context& context);

} // namespace caelestia::cli::shell
