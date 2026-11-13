#include "emoji.hpp"

namespace caelestia::cli::emoji {

using Qt::StringLiterals::operator""_s;

parser::NodePtr parserNode() {
    return parser::group({
        .name = u"emoji"_s,
        .help = u"emoji/glyph utilities"_s,
        .children = {
            parser::leaf<parser::NoArgs>({
                .name = u"list"_s,
                .help = u"print the emoji/glyph data"_s,
                .run = list,
            }),
            parser::leaf<parser::NoArgs>({
                .name = u"pick"_s,
                .help = u"open the emoji/glyph picker"_s,
                .run = pick,
            }),
#ifdef CAELESTIA_DEV_COMMANDS
            parser::leaf<parser::NoArgs>({
                .name = u"fetch"_s,
                .help = u"fetch emoji/glyph data from remote"_s,
                .run = fetch,
            }),
#endif
        },
    });
}

} // namespace caelestia::cli::emoji
