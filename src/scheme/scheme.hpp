#pragma once

#include <caelestia/scheme/scheme.hpp>

#include "parser/node.hpp"

namespace caelestia::cli::scheme {

struct ListArgs {
    bool names = false;
    bool flavours = false;
    bool modes = false;
};

struct GetArgs {
    bool name = false;
    bool flavour = false;
    bool mode = false;
};

struct SetArgs {
    std::optional<QString> name;
    std::optional<QString> flavour;
    std::optional<caelestia::scheme::Mode> mode;
};

parser::NodePtr parserNode();

int list(const ListArgs& args, const parser::Context& context);
int get(const GetArgs& args, const parser::Context& context);
int set(const SetArgs& args, const parser::Context& context);
int random(const parser::NoArgs& args, const parser::Context& context);

} // namespace caelestia::cli::scheme
