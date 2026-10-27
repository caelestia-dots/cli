#include "emoji.hpp"

#include "parser/node.hpp"

namespace caelestia::cli::emoji {

using Qt::StringLiterals::operator""_s;

namespace {

parser::NodePtr simple(QString name, QString help) {
    return parser::leaf<parser::NoArgs>({
        .name = std::move(name),
        .help = std::move(help),
        .run =
            [](const parser::NoArgs&, const parser::Context& context) {
                return parser::notImplemented(context);
            },
    });
}

} // namespace

parser::NodePtr parserNode() {
    return parser::group({
        .name = u"emoji"_s,
        .help = u"emoji/glyph utilities"_s,
        .children = {
            simple(u"pick"_s, u"open the emoji/glyph picker"_s),
#ifdef CAELESTIA_DEV_COMMANDS
            simple(u"fetch"_s, u"fetch emoji/glyph data from remote"_s),
#endif
        },
    });
}

} // namespace caelestia::cli::emoji
