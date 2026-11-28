#pragma once

#include "common/target.hpp"
#include "parser/node.hpp"

namespace caelestia::cli::record {

struct StartArgs {
    common::Target target = common::Target::Screen;
    QString geometry;
    bool sound = false;
    bool clipboard = false;
    bool quiet = false;
};

struct QuietArgs {
    bool quiet = false;
};

parser::NodePtr parserNode();

int start(const StartArgs& args, const parser::Context& context);
int toggle(const StartArgs& args, const parser::Context& context);
int stop(const QuietArgs& args, const parser::Context& context);
int pause(const QuietArgs& args, const parser::Context& context);
int resume(const QuietArgs& args, const parser::Context& context);
int togglePause(const QuietArgs& args, const parser::Context& context);

} // namespace caelestia::cli::record
