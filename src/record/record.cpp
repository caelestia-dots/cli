#include "record.hpp"

#include "common/target.hpp"
#include "parser/node.hpp"

namespace caelestia::cli::record {

using Qt::StringLiterals::operator""_s;

namespace {

struct StartArgs {
    common::Target target = common::Target::Screen;
    QString geometry;
    bool sound = false;
    bool clipboard = false;
};

parser::NodePtr startLike(QString name, QString help) {
    return parser::leaf<StartArgs>({
        .name = std::move(name),
        .help = std::move(help),
        .options = {
            parser::value(&StartArgs::geometry, u'g', u"geometry"_s, u"GEOM"_s, u"record a fixed region ('x,y wxh')"_s),
            parser::flag(&StartArgs::sound, u's', u"sound"_s, u"record audio"_s),
            parser::flag(&StartArgs::clipboard, u'c', u"clipboard"_s, u"copy the recording path to the clipboard"_s),
        },
        .positionals = {
            parser::positional(&StartArgs::target, u"TARGET"_s, u"what to record"_s, common::targetChoices()),
        },
        .run = [](const StartArgs&, const parser::Context& context) { return parser::notImplemented(context); },
    });
}

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
        .name = u"record"_s,
        .help = u"manage screen recording"_s,
        .children = {
            startLike(u"start"_s, u"start a recording"_s),
            startLike(u"toggle"_s, u"start a recording, or stop the current one"_s),
            simple(u"stop"_s, u"stop the recording"_s),
            simple(u"pause"_s, u"pause/resume the recording"_s),
        },
    });
}

} // namespace caelestia::cli::record
