#include "wallpaper.hpp"

#include <optional>

#include "parser/node.hpp"

namespace caelestia::cli::wallpaper {

using Qt::StringLiterals::operator""_s;

namespace {

struct SetArgs {
    QString path;
    bool noSmart = false;
};

struct RandomArgs {
    std::optional<QString> dir;
    bool noFilter = false;
    double minSize = 0.8;
    bool noSmart = false;
};

struct SchemeArgs {
    std::optional<QString> path;
    bool noSmart = false;
};

template <typename T> parser::Option<T> noSmartOption() {
    return parser::flag(&T::noSmart, {}, u"no-smart"_s, u"do not change the scheme mode based on wallpaper colour"_s);
}

} // namespace

parser::NodePtr parserNode() {
    return parser::group({
        .name = u"wallpaper"_s,
        .help = u"manage the wallpaper"_s,
        .children = {
            parser::leaf<parser::NoArgs>({
                .name = u"get"_s,
                .help = u"print the current wallpaper path"_s,
                .run = [](const parser::NoArgs&, const parser::Context& context) { return parser::notImplemented(context); },
            }),
            parser::leaf<SetArgs>({
                .name = u"set"_s,
                .help = u"switch to a wallpaper"_s,
                .options = { noSmartOption<SetArgs>() },
                .positionals = {
                    parser::positional(&SetArgs::path, u"PATH"_s, u"the wallpaper to switch to"_s),
                },
                .run = [](const SetArgs&, const parser::Context& context) { return parser::notImplemented(context); },
            }),
            parser::leaf<RandomArgs>({
                .name = u"random"_s,
                .help = u"switch to a random wallpaper"_s,
                .options = {
                    parser::flag(&RandomArgs::noFilter, {}, u"no-filter"_s, u"do not filter by size"_s),
                    parser::value(&RandomArgs::minSize, {}, u"min-size"_s, u"FRACTION"_s,
                        u"minimum size relative to the largest monitor (default 0.8)"_s),
                    noSmartOption<RandomArgs>(),
                },
                .positionals = {
                    parser::positional(&RandomArgs::dir, u"DIR"_s, u"the directory to pick from"_s),
                },
                .run = [](const RandomArgs&, const parser::Context& context) { return parser::notImplemented(context); },
            }),
            parser::leaf<SchemeArgs>({
                .name = u"scheme"_s,
                .help = u"print the scheme for a wallpaper"_s,
                .options = { noSmartOption<SchemeArgs>() },
                .positionals = {
                    parser::positional(&SchemeArgs::path, u"PATH"_s, u"the wallpaper (default: current)"_s),
                },
                .run = [](const SchemeArgs&, const parser::Context& context) { return parser::notImplemented(context); },
            }),
        },
    });
}

} // namespace caelestia::cli::wallpaper
