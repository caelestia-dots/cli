#include "toggle.hpp"

#include <qtextstream.h>

#include "common/hypr.hpp"
#include "parser/node.hpp"

namespace caelestia::cli::toggle {

using Qt::StringLiterals::operator""_s;

namespace {

struct Args {
    QString workspace;
};

int run(const Args& args, const parser::Context& context) {
    const auto success = common::hypr::eval(u"require(\"utils.functions\").toggle(\"%1\")()"_s.arg(args.workspace));
    if (!success) {
        const auto err = u"Failed to toggle workspace %1"_s.arg(args.workspace);
        if (context.json)
            QTextStream(stdout) << u"{ \"error\": \""_s << err << u"\" }\n"_s;
        else
            QTextStream(stderr) << err << u'\n';

        return 1;
    }

    return 0;
}

} // namespace

parser::NodePtr parserNode() {
    return parser::leaf<Args>({
        .name = u"toggle"_s,
        .help = u"toggle a special workspace"_s,
        .positionals = {
            parser::positional(&Args::workspace, u"WORKSPACE"_s, u"the workspace to toggle"_s),
        },
        .run = run,
    });
}

} // namespace caelestia::cli::toggle
