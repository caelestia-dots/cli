#include "wallpaper.hpp"

namespace caelestia::cli::wallpaper {

using Qt::StringLiterals::operator""_s;

namespace {

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
                .run = get,
            }),
            parser::leaf<SetArgs>({
                .name = u"set"_s,
                .help = u"switch to a wallpaper"_s,
                .options = { noSmartOption<SetArgs>() },
                .positionals = {
                    parser::positional(&SetArgs::path, u"PATH"_s, u"the wallpaper to switch to"_s),
                },
                .run = set,
            }),
            parser::leaf<RandomArgs>({
                .name = u"random"_s,
                .help = u"switch to a random wallpaper"_s,
                .options = { noSmartOption<RandomArgs>() },
                .positionals = {
                    parser::positional(&RandomArgs::dir, u"DIR"_s, u"the directory to pick from"_s),
                },
                .run = random,
            }),
        },
    });
}

} // namespace caelestia::cli::wallpaper
