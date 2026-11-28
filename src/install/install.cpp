#include "install.hpp"

#include <optional>

#include "parser/node.hpp"

namespace caelestia::cli::install {

using Qt::StringLiterals::operator""_s;

namespace {

struct Args {
    std::optional<QString> aurHelper;
    std::optional<QString> enableComponents;
    std::optional<QString> disableComponents;
    bool noConfirm = false;
};

} // namespace

parser::NodePtr parserNode() {
    return parser::leaf<Args>({
        .name = u"install"_s,
        .help = u"install the Caelestia dotfiles"_s,
        .options = {
            parser::value(&Args::aurHelper, {}, u"aur-helper"_s, u"HELPER"_s, u"the AUR helper to use"_s),
            parser::value(&Args::enableComponents, {}, u"enable-components"_s, u"LIST"_s,
                u"comma-separated list of components to enable"_s),
            parser::value(&Args::disableComponents, {}, u"disable-components"_s, u"LIST"_s,
                u"comma-separated list of components to disable"_s),
            parser::flag(&Args::noConfirm, {}, u"noconfirm"_s, u"use defaults for all prompts"_s),
        },
        .run = [](const Args&, const parser::Context& context) { return parser::notImplemented(context); },
    });
}

} // namespace caelestia::cli::install
