#include "clipboard.hpp"

#include "parser/node.hpp"

namespace caelestia::cli::clipboard {

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
        .name = u"clipboard"_s,
        .help = u"clipboard history"_s,
        .children = {
            simple(u"pick"_s, u"open the clipboard history picker"_s),
            simple(u"delete"_s, u"delete entries from the clipboard history"_s),
        },
    });
}

} // namespace caelestia::cli::clipboard
