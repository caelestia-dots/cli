#pragma once

#include "parser/node.hpp"

namespace caelestia::cli::clipboard {

parser::NodePtr parserNode();

int pick(const parser::NoArgs& args, const parser::Context& context);
int remove(const parser::NoArgs& args, const parser::Context& context);

} // namespace caelestia::cli::clipboard
