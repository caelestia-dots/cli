#include "screenshot.hpp"

#include "common/target.hpp"
#include "parser/node.hpp"

namespace caelestia::cli::screenshot {

using Qt::StringLiterals::operator""_s;

namespace {

struct Args {
    common::Target target = common::Target::Screen;
    QString geometry;
    bool freeze = false;
};

} // namespace

parser::NodePtr parserNode() {
    return parser::leaf<Args>({
        .name = u"screenshot"_s,
        .help = u"take a screenshot"_s,
        .options = {
            parser::value(&Args::geometry, u'g', u"geometry"_s, u"GEOM"_s, u"capture a fixed region ('x,y wxh')"_s),
            parser::flag(&Args::freeze, {}, u"freeze"_s, u"freeze the screen while selecting a region"_s),
        },
        .positionals = {
            parser::positional(&Args::target, u"TARGET"_s, u"what to capture"_s, common::targetChoices()),
        },
        .run = [](const Args&, const parser::Context& context) { return parser::notImplemented(context); },
    });
}

} // namespace caelestia::cli::screenshot
