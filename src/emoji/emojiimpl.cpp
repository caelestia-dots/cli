#include <qfile.h>
#include <qjsondocument.h>
#include <qjsonobject.h>
#include <qprocess.h>

#ifdef CAELESTIA_DEV_COMMANDS
#include <qeventloop.h>
#include <qjsonarray.h>
#include <qnetworkaccessmanager.h>
#include <qnetworkreply.h>
#include <qsavefile.h>

#include <array>
#include <stdexcept>
#endif

#include "emoji.hpp"

namespace caelestia::cli::emoji {

using Qt::StringLiterals::operator""_s;

namespace {

constexpr QStringView k_dataResource = u":/emoji/emojis.txt";

QByteArray data() {
    QFile file(k_dataResource.toString());
    if (!file.open(QIODevice::ReadOnly))
        return {};
    return file.readAll();
}

int printError(const QString& err, bool json) {
    if (json)
        QTextStream(stdout) << QJsonDocument(QJsonObject{ { u"error"_s, err } }).toJson(QJsonDocument::Indented)
                            << u'\n';
    else
        QTextStream(stderr) << err << u'\n';
    return 1;
}

int run(QProcess& proc, const QString& program, const QStringList& args, const QByteArray& input) {
    proc.start(program, args);
    if (!proc.waitForStarted())
        return -1;

    proc.write(input);
    proc.closeWriteChannel();
    proc.waitForFinished(-1);
    return proc.exitStatus() == QProcess::NormalExit ? proc.exitCode() : -1;
}

} // namespace

int list(const parser::NoArgs& args, const parser::Context& context) {
    Q_UNUSED(args);
    Q_UNUSED(context);

    QTextStream(stdout) << data();
    return 0;
}

int pick(const parser::NoArgs& args, const parser::Context& context) {
    Q_UNUSED(args);

    QProcess fuzzel;
    fuzzel.setProcessChannelMode(QProcess::ForwardedErrorChannel);
    const auto exit = run(fuzzel, u"fuzzel"_s, { u"--dmenu"_s, u"--placeholder=Type to search emojis"_s }, data());
    if (exit < 0)
        return printError(u"Failed to run fuzzel: %1"_s.arg(fuzzel.errorString()), context.json);
    if (exit != 0)
        return exit;

    const auto chosen = QString::fromUtf8(fuzzel.readAllStandardOutput()).simplified().section(u' ', 0, 0);
    if (chosen.isEmpty())
        return 1;

    QProcess wlCopy;
    wlCopy.setProcessChannelMode(QProcess::ForwardedChannels);
    if (run(wlCopy, u"wl-copy"_s, {}, chosen.toUtf8()) != 0)
        return printError(u"Failed to run wl-copy: %1"_s.arg(wlCopy.errorString()), context.json);
    return 0;
}

#ifdef CAELESTIA_DEV_COMMANDS

namespace {

constexpr QStringView k_emojiUrl =
    u"https://raw.githubusercontent.com/milesj/emojibase/refs/heads/master/packages/data/en/compact.raw.json";
constexpr QStringView k_glyphUrl =
    u"https://raw.githubusercontent.com/ryanoasis/nerd-fonts/refs/heads/master/glyphnames.json";

constexpr std::array<QStringView, 23> k_extraLines = {
    u"¿? question upside down reversed spanish",
    u"← left arrow",
    u"↑ up arrow",
    u"→ right arrow",
    u"↓ down arrow",
    u"←↑→↓ all directions up down left right arrows",
    u"⇇ leftwards paired arrows",
    u"⇉ rightwards paired arrows",
    u"⇈ upwards paired arrows",
    u"⇊ downwards paired arrows",
    u"⬱ three leftwards arrows",
    u"⇶ three rightwards arrows",
    u"• dot circle separator",
    u"「」 japanese quote square bracket",
    u"¯\\_(ツ)_/¯ shrug idk i dont know",
    u"(ง🔥ﾛ🔥)ง person with fire eyes eyes on fire",
    u"↵ enter key return",
    u"° degrees",
    u"™ tm trademark",
    u"® registered trademark",
    u"© copyright",
    u"— em dash",
    u"󰖳 windows super key",
};

QJsonDocument download(QStringView url) {
    QNetworkAccessManager manager;
    QNetworkReply* const reply = manager.get(QNetworkRequest(QUrl(url.toString())));

    QEventLoop loop;
    QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    loop.exec();

    if (reply->error() != QNetworkReply::NoError)
        throw std::runtime_error(u"Failed to fetch %1: %2"_s.arg(url, reply->errorString()).toStdString());

    const auto doc = QJsonDocument::fromJson(reply->readAll());
    reply->deleteLater();
    return doc;
}

QStringList emojiLines(const QJsonArray& emojis) {
    QStringList lines;
    for (const auto& value : emojis) {
        const auto emoji = value.toObject();
        QStringList line{ emoji.value(u"unicode"_s).toString() };

        const auto emoticon = emoji.value(u"emoticon"_s);
        if (emoticon.isString()) {
            line << emoticon.toString();
        } else {
            const auto emoticons = emoticon.toArray();
            for (const auto& e : emoticons)
                line << e.toString();
        }

        line << emoji.value(u"label"_s).toString();

        const auto tags = emoji.value(u"tags"_s).toArray();
        for (const auto& tag : tags)
            line << tag.toString();

        lines << line.join(u' ');
    }
    return lines;
}

QStringList glyphLines(const QJsonObject& glyphs) {
    QStringList order;
    QHash<QString, QStringList> buckets;
    for (auto it = glyphs.constBegin(); it != glyphs.constEnd(); ++it) {
        if (it.key() == u"METADATA")
            continue;

        const auto glyph = it->toObject().value(u"char"_s).toString();
        if (!buckets.contains(glyph))
            order << glyph;
        buckets[glyph] << u"nf-"_s + it.key();
    }

    QStringList lines;
    for (const auto& glyph : std::as_const(order))
        lines << glyph + u"  "_s + buckets.value(glyph).join(u' ');
    return lines;
}

} // namespace

int fetch(const parser::NoArgs& args, const parser::Context& context) {
    Q_UNUSED(args);

    try {
        QStringList lines;
        for (const auto line : k_extraLines)
            lines << line.toString();
        lines << emojiLines(download(k_emojiUrl).array()) << glyphLines(download(k_glyphUrl).object());

        QSaveFile file(QStringLiteral(CAELESTIA_EMOJI_DATA_PATH));
        if (!file.open(QIODevice::WriteOnly) || file.write(lines.join(u'\n').toUtf8()) < 0 || !file.commit())
            throw std::runtime_error(
                u"Failed to write %1: %2"_s.arg(file.fileName(), file.errorString()).toStdString());
    } catch (const std::exception& e) {
        return printError(QString::fromUtf8(e.what()), context.json);
    }

    return 0;
}

#endif

} // namespace caelestia::cli::emoji
