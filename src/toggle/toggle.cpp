#include "toggle.hpp"

#include "parser/node.hpp"

namespace caelestia::cli::toggle {

using Qt::StringLiterals::operator""_s;

namespace {

struct Args {
    QString workspace;
};

} // namespace

parser::NodePtr parserNode() {
    return parser::leaf<Args>({
        .name = u"toggle"_s,
        .help = u"toggle a special workspace"_s,
        .positionals = {
            parser::positional(&Args::workspace, u"WORKSPACE"_s, u"the workspace to toggle"_s),
        },
        .run = [](const Args&, const parser::Context& context) { return parser::notImplemented(context); },
    });
}

} // namespace caelestia::cli::toggle
