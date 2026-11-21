#include "scheme.hpp"

namespace caelestia::cli::scheme {

using Qt::StringLiterals::operator""_s;
using caelestia::scheme::Mode;

parser::NodePtr parserNode() {
    return parser::group({
        .name = u"scheme"_s,
        .help = u"manage the colour scheme"_s,
        .children = {
            parser::leaf<ListArgs>({
                .name = u"list"_s,
                .help = u"list available schemes"_s,
                .options = {
                    parser::flag(&ListArgs::names, u'n', u"names"_s, u"list scheme names"_s),
                    parser::flag(&ListArgs::flavours, u'f', u"flavours"_s, u"list scheme flavours"_s),
                    parser::flag(&ListArgs::modes, u'm', u"modes"_s, u"list scheme modes"_s),
                },
                .run = list,
            }),
            parser::leaf<GetArgs>({
                .name = u"get"_s,
                .help = u"get scheme properties"_s,
                .options = {
                    parser::flag(&GetArgs::name, u'n', u"name"_s, u"print the current scheme name"_s),
                    parser::flag(&GetArgs::flavour, u'f', u"flavour"_s, u"print the current scheme flavour"_s),
                    parser::flag(&GetArgs::mode, u'm', u"mode"_s, u"print the current scheme mode"_s),
                },
                .run = get,
            }),
            parser::leaf<SetArgs>({
                .name = u"set"_s,
                .help = u"set the current scheme"_s,
                .options = {
                    parser::value(&SetArgs::name, u'n', u"name"_s, u"NAME"_s, u"the name of the scheme to switch to"_s),
                    parser::value(&SetArgs::flavour, u'f', u"flavour"_s, u"FLAVOUR"_s, u"the flavour to switch to"_s),
                    parser::choice(&SetArgs::mode, u'm', u"mode"_s, u"MODE"_s, u"the mode to switch to"_s,
                        { { u"dark"_s, Mode::Dark }, { u"light"_s, Mode::Light } }),
                },
                .requireOption = true,
                .run = set,
            }),
            parser::leaf<parser::NoArgs>({
                .name = u"random"_s,
                .help = u"switch to a random scheme"_s,
                .run = random,
            }),
        },
    });
}

} // namespace caelestia::cli::scheme
