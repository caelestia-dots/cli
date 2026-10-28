#include "clipboard.hpp"

namespace caelestia::cli::clipboard {

using Qt::StringLiterals::operator""_s;

parser::NodePtr parserNode() {
    return parser::group({
        .name = u"clipboard"_s,
        .help = u"clipboard history"_s,
        .children = {
            parser::leaf<parser::NoArgs>({
                .name = u"pick"_s,
                .help = u"open the clipboard history picker"_s,
                .run = pick,
            }),
            parser::leaf<parser::NoArgs>({
                .name = u"delete"_s,
                .help = u"delete entries from the clipboard history"_s,
                .run = remove,
            }),
        },
    });
}

} // namespace caelestia::cli::clipboard
