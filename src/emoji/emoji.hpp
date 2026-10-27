#pragma once

#include "parser/node.hpp"

namespace caelestia::cli::emoji {

parser::NodePtr parserNode();

int list(const parser::NoArgs& args, const parser::Context& context);
int pick(const parser::NoArgs& args, const parser::Context& context);
#ifdef CAELESTIA_DEV_COMMANDS
int fetch(const parser::NoArgs& args, const parser::Context& context);
#endif

} // namespace caelestia::cli::emoji
