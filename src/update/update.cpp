#include "update.hpp"

#include <optional>

#include "parser/node.hpp"

namespace caelestia::cli::update {

using Qt::StringLiterals::operator""_s;

namespace {

struct Args {
    std::optional<QString> aurHelper;
    bool noConfirm = false;
};

} // namespace

parser::NodePtr parserNode() {
    return parser::leaf<Args>({
        .name = u"update"_s,
        .help = u"update the Caelestia dotfiles"_s,
        .options = {
            parser::value(&Args::aurHelper, {}, u"aur-helper"_s, u"HELPER"_s, u"the AUR helper to use"_s),
            parser::flag(&Args::noConfirm, {}, u"noconfirm"_s, u"use defaults for all prompts"_s),
        },
        .run = [](const Args&, const parser::Context& context) { return parser::notImplemented(context); },
    });
}

} // namespace caelestia::cli::update
