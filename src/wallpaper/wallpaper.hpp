#pragma once

#include <optional>

#include "parser/node.hpp"

namespace caelestia::cli::wallpaper {

struct SetArgs {
    QString path;
    bool noSmart = false;
};

struct RandomArgs {
    std::optional<QString> dir;
    bool noSmart = false;
};

parser::NodePtr parserNode();

int get(const parser::NoArgs& args, const parser::Context& context);
int set(const SetArgs& args, const parser::Context& context);
int random(const RandomArgs& args, const parser::Context& context);

} // namespace caelestia::cli::wallpaper
