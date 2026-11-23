#include "record.hpp"

namespace caelestia::cli::record {

using Qt::StringLiterals::operator""_s;

namespace {

parser::NodePtr startLike(
    QString name, QString help, std::function<int(const StartArgs&, const parser::Context&)> run) {
    return parser::leaf<StartArgs>({
        .name = std::move(name),
        .help = std::move(help),
        .options = {
            parser::value(&StartArgs::geometry, u'g', u"geometry"_s, u"GEOM"_s, u"record a fixed region ('x,y wxh')"_s),
            parser::flag(&StartArgs::sound, u's', u"sound"_s, u"record audio"_s),
            parser::flag(&StartArgs::clipboard, u'c', u"clipboard"_s, u"copy the recording path to the clipboard"_s),
            parser::flag(&StartArgs::quiet, u'q', u"quiet"_s, u"don't send a notification"_s),
        },
        .positionals = {
            parser::positional(&StartArgs::target, u"TARGET"_s, u"what to record"_s, common::targetChoices()),
        },
        .run = std::move(run),
    });
}

parser::NodePtr simple(QString name, QString help, std::function<int(const QuietArgs&, const parser::Context&)> run) {
    return parser::leaf<QuietArgs>({
        .name = std::move(name),
        .help = std::move(help),
        .options = {
            parser::flag(&QuietArgs::quiet, u'q', u"quiet"_s, u"don't send a notification"_s),
        },
        .run = std::move(run),
    });
}

} // namespace

parser::NodePtr parserNode() {
    return parser::group({
        .name = u"record"_s,
        .help = u"manage screen recording"_s,
        .children = {
            startLike(u"start"_s, u"start a recording"_s, start),
            startLike(u"toggle"_s, u"start a recording, or stop the current one"_s, toggle),
            simple(u"stop"_s, u"stop the recording"_s, stop),
            simple(u"pause"_s, u"pause the recording"_s, pause),
            simple(u"resume"_s, u"resume the recording"_s, resume),
            simple(u"toggle-pause"_s, u"pause or resume the recording"_s, togglePause),
        },
    });
}

} // namespace caelestia::cli::record
