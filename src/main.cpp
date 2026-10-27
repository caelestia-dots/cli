#include <qcoreapplication.h>
#include <qprocess.h>
#include <qstring.h>
#include <qstringlist.h>

#include <caelestia/utils/components.hpp>

#include "clipboard/clipboard.hpp"
#include "emoji/emoji.hpp"
#include "install/install.hpp"
#include "parser/node.hpp"
#include "record/record.hpp"
#include "scheme/scheme.hpp"
#include "screenshot/screenshot.hpp"
#include "shell/shell.hpp"
#include "toggle/toggle.hpp"
#include "update/update.hpp"
#include "wallpaper/wallpaper.hpp"

using namespace caelestia::cli;
using Qt::StringLiterals::operator""_s;

namespace {

constexpr QStringView k_libComp = u"caelestia-lib";
constexpr QStringView k_shellComp = u"caelestia-shell";
constexpr QStringView k_cliComp = u"caelestia-cli";

QString version() {
    const auto components = caelestia::utils::installedComponents();
    QString verStr;

    // TODO: dots info
    // TODO: improve format

    for (const auto name : { k_libComp, k_shellComp, k_cliComp }) {
        if (components.contains(name)) {
            const auto comp = components.value(name);
            verStr += u"%1 %2 (revision %3, distributed by %4)\n"_s.arg(
                name, comp.version, comp.gitRevision, comp.distributor.isEmpty() ? u"unknown"_s : comp.distributor);
        } else {
            verStr += u"%1 (not installed)\n"_s.arg(name);
        }
    }

    QProcess qsProc;
    qsProc.start(u"qs"_s, { u"--version"_s });
    qsProc.setStandardErrorFile(QProcess::nullDevice());
    if (qsProc.waitForFinished(5000)) {
        verStr += QString::fromUtf8(qsProc.readAllStandardOutput());
    } else {
        if (qsProc.error() == QProcess::FailedToStart)
            verStr += u"Quickshell (not installed)\n"_s;
        else
            verStr += u"Quickshell (error: %1)\n"_s.arg(qsProc.errorString());
    }

    // TODO: local shell info

    return verStr;
}

} // namespace

int main(int argc, char** argv) {
    const QCoreApplication app(argc, argv);

    const auto root = parser::group({
        .name = u"caelestia"_s,
        .help = u"A convenience CLI for Caelestia"_s,
        .children = {
            shell::parserNode(),
            toggle::parserNode(),
            scheme::parserNode(),
            screenshot::parserNode(),
            record::parserNode(),
            clipboard::parserNode(),
            emoji::parserNode(),
            wallpaper::parserNode(),
            install::parserNode(),
            update::parserNode(),
        },
        .version = version,
    });

    return root->dispatch(QCoreApplication::arguments().mid(1), parser::Context{ .command = root->name() });
}
