#include <qdirlisting.h>
#include <qfileinfo.h>
#include <qrandom.h>
#include <qstandardpaths.h>

#include <caelestia/media/analysis.hpp>
#include <caelestia/wallpaper/wallpaper.hpp>

#include "common/error.hpp"
#include "wallpaper.hpp"

namespace caelestia::cli::wallpaper {

using Qt::StringLiterals::operator""_s;
namespace lib = caelestia::wallpaper;

namespace {

QString escape(QString s) {
    return s.replace(u'\\', u"\\\\"_s).replace(u'\"', u"\\\""_s);
}

QString defaultDir() {
    return QStandardPaths::writableLocation(QStandardPaths::PicturesLocation) + u"/Wallpapers"_s;
}

} // namespace

int get(const parser::NoArgs& args, const parser::Context& context) {
    Q_UNUSED(args);

    bool ok;
    const auto wall = lib::currentWallpaper(&ok);
    if (ok) {
        if (context.json)
            QTextStream(stdout) << u"{ \"path\": \""_s << escape(wall) << u"\" }\n"_s;
        else
            QTextStream(stdout) << wall << u'\n';
    } else {
        return common::printError(u"Wallpaper has not been set or is invalid"_s, context.json);
    }
    return 0;
}

int set(const SetArgs& args, const parser::Context& context) {
    try {
        lib::set(args.path, !args.noSmart).waitForFinished();
    } catch (const std::exception& e) {
        return common::printError(QString::fromUtf8(e.what()), context.json);
    }

    const auto path = QFileInfo(args.path).absoluteFilePath();
    if (context.json)
        QTextStream(stdout) << u"{ \"path\": \""_s << escape(path) << u"\" }\n"_s;
    else
        QTextStream(stdout) << u"Wallpaper set to "_s << path << u'\n';
    return 0;
}

int random(const RandomArgs& args, const parser::Context& context) {
    QString chosen;
    qsizetype seen = 0;
    auto* const gen = QRandomGenerator::global();

    using F = QDirListing::IteratorFlag;
    for (const auto& entry : QDirListing(args.dir.value_or(defaultDir()), F::FilesOnly | F::Recursive)) {
        if (!media::isSupported(entry.absoluteFilePath()))
            continue;

        if (gen->bounded(++seen) == 0)
            chosen = entry.absoluteFilePath();
    }

    return set({ .path = chosen, .noSmart = args.noSmart }, context);
}

} // namespace caelestia::cli::wallpaper
