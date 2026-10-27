
#include <qjsonarray.h>
#include <qjsondocument.h>
#include <qjsonobject.h>
#include <qrandom.h>

#include <caelestia/scheme/smart.hpp>
#include <caelestia/theme/theme.hpp>
#include <caelestia/wallpaper/wallpaper.hpp>

#include "common/ansi.hpp"
#include "scheme.hpp"

namespace caelestia::cli::scheme {

using Qt::StringLiterals::operator""_s;
namespace lib = caelestia::scheme;
namespace ansi = common::ansi;

namespace {

constexpr QStringView k_dynamic = u"dynamic";

}

int list(const ListArgs& args, const parser::Context& context) {
    auto schemes = lib::Scheme::all().takeResult();
    schemes << lib::Scheme::forColour(Qt::black);

    const auto argCount = args.names + args.flavours + args.modes;
    if (context.json) {
        QJsonObject json;
        if (args.names && argCount == 1) {
            QJsonArray names;
            for (const auto& scheme : schemes)
                names << scheme->name();
            json.insert(u"names"_s, names);
            QTextStream(stdout) << QJsonDocument(json).toJson(QJsonDocument::Indented) << u'\n';
            return 0;
        }

        if (args.flavours && argCount == 1) {
            const auto state = lib::loadStateRef();
            if (!state) {
                // TODO: print error
                return 1;
            }

            const auto scheme = lib::Scheme::forName(state->name).takeResult();

            QJsonArray flavours;
            for (const auto& flavour : scheme->flavours())
                flavours << flavour.name();
            json.insert(u"flavours"_s, flavours);
            QTextStream(stdout) << QJsonDocument(json).toJson(QJsonDocument::Indented) << u'\n';
            return 0;
        }

        if (args.modes && argCount == 1) {
            const auto state = lib::loadStateRef();
            if (!state) {
                // TODO: print error
                return 1;
            }

            const auto scheme = lib::Scheme::forName(state->name).takeResult();
            const auto flavour = scheme->flavour(state->flavour);

            QJsonArray modes;
            for (const auto& mode : flavour->modes())
                modes << lib::modeName(mode);
            json.insert(u"modes"_s, modes);
            QTextStream(stdout) << QJsonDocument(json).toJson(QJsonDocument::Indented) << u'\n';
            return 0;
        }

        QJsonArray schemeArr;
        for (const auto& scheme : schemes) {
            // TODO
        }
        json.insert(u"schemes"_s, schemeArr);
        QTextStream(stdout) << QJsonDocument(json).toJson(QJsonDocument::Indented) << u'\n';

        return 0;
    }

    for (const auto& scheme : schemes)
        QTextStream(stdout) << scheme->name() << u'\n';

    return 0;
}

int get(const GetArgs& args, const parser::Context& context) {
    const auto state = lib::loadStateRef();
    if (!state) {
        const auto err = u"Scheme has not been set or is invalid"_s;
        if (context.json)
            QTextStream(stdout) << u"{ \"error\": \""_s << err << u"\" }\n"_s;
        else
            QTextStream(stderr) << err << u'\n';
        return 1;
    }

    if (args.name + args.flavour + args.mode == 1) {
        const auto print = [json = context.json](const QString& label, const QString& val) {
            if (json)
                QTextStream(stdout) << u"{ \""_s << label << u"\": \""_s << val << u"\" }\n"_s;
            else
                QTextStream(stdout) << val << u'\n';
        };

        if (args.name)
            print(u"name"_s, state->name);
        if (args.flavour)
            print(u"flavour"_s, state->flavour);
        if (args.mode)
            print(u"mode"_s, lib::modeName(state->mode));

        return 0;
    }

    const auto print = [json = context.json](const QString& label, const QString& val) {
        if (json)
            QTextStream(stdout) << u"    \""_s + label.toLower() + u"\": \""_s << val << u"\"\n"_s;
        else
            QTextStream(stdout) << ansi::styled(ansi::Bold | ansi::Cyan, label + u':') << u' ' << val << u'\n';
    };

    if (context.json)
        QTextStream(stdout) << u"{\n"_s;

    const auto printAll = args.name + args.flavour + args.mode == 0;
    if (printAll || args.name)
        print(u"Name"_s, state->name);
    if (printAll || args.flavour)
        print(u"Flavour"_s, state->flavour);
    if (printAll || args.mode)
        print(u"Mode"_s, lib::modeName(state->mode));

    if (context.json)
        QTextStream(stdout) << u"}\n"_s;

    return 0;
}

int set(const SetArgs& args, const parser::Context& context) {
    if (args.name.has_value() + args.flavour.has_value() + args.mode.has_value() == 0) {
        const auto err = u"Missing scheme name, flavour or mode"_s;
        if (context.json)
            QTextStream(stdout) << u"{ \"error\": \""_s << err << u"\" }\n"_s;
        else
            QTextStream(stderr) << err << u'\n';
        return 1;
    }

    QStringList warnings;
    try {
        const auto invalidName =
            args.name && *args.name != k_dynamic &&
            !std::ranges::any_of(lib::Scheme::all().takeResult(), [name = *args.name](const auto& s) {
                return s->name() == name;
            });
        if (invalidName)
            warnings << u"Scheme \"%1\" does not exist"_s.arg(*args.name);

        auto name = args.name;
        if (!name || invalidName) {
            const auto state = lib::loadStateRef();
            name = state ? state->name : lib::defaultScheme().takeResult().name;
        }
        const auto scheme = *name == k_dynamic ? lib::Scheme::forMedia(wallpaper::currentWallpaper()).takeResult()
                                               : lib::Scheme::forName(*name).takeResult();

        const auto invalidFlavour = args.flavour && !std::ranges::any_of(scheme->flavours(), [args](const auto& f) {
            return f.name() == *args.flavour;
        });
        if (invalidFlavour)
            warnings << u"\"%1\" is not a flavour of scheme %2"_s.arg(*args.flavour, scheme->name());
        std::optional<lib::Flavour> flavour;
        if (args.flavour && !invalidFlavour) {
            flavour = scheme->flavour(*args.flavour);
        } else {
            const auto state = lib::loadStateRef();
            if (state && std::ranges::any_of(scheme->flavours(), [flavour = state->flavour](const auto& f) {
                    return f.name() == flavour;
                }))
                flavour = scheme->flavour(state->flavour);
            else if (args.flavour && *args.flavour == k_dynamic)
                flavour = scheme->flavour(
                    lib::smartChoice(media::analyseForPath(wallpaper::currentWallpaper()).takeResult()).flavour);
            else
                flavour = scheme->defaultFlavour();
        }

        const auto invalidMode = args.mode && !flavour->modes().contains(*args.mode);
        if (invalidMode)
            warnings << u"\"%1\" is not a mode of scheme %2 (%3)"_s.arg(
                lib::modeName(*args.mode), scheme->name(), flavour->name());
        const auto mode = args.mode && !invalidMode ? *args.mode : flavour->defaultMode();

        theme::Theme(flavour->data(mode).takeResult()).apply().waitForFinished();

        if (context.json) {
            QJsonObject json;
            json.insert(u"name"_s, scheme->name());
            json.insert(u"flavour"_s, flavour->name());
            json.insert(u"mode"_s, lib::modeName(mode));
            json.insert(u"warnings"_s, QJsonArray::fromStringList(warnings));
            QTextStream(stdout) << QJsonDocument(json).toJson(QJsonDocument::Indented) << u'\n';
        } else {
            for (const auto& warning : std::as_const(warnings))
                QTextStream(stderr) << warning << u'\n';
            QTextStream(stdout) << u"Scheme set to %1 (%2, %3)\n"_s.arg(
                scheme->name(), flavour->name(), lib::modeName(mode));
        }
    } catch (const std::exception& e) {
        if (context.json) {
            QJsonObject json;
            json.insert(u"error"_s, QString::fromUtf8(e.what()));
            json.insert(u"warnings"_s, QJsonArray::fromStringList(warnings));
            QTextStream(stdout) << QJsonDocument(json).toJson(QJsonDocument::Indented) << u'\n';
        } else {
            for (const auto& warning : std::as_const(warnings))
                QTextStream(stderr) << warning << u'\n';
            QTextStream(stderr) << e.what() << u'\n';
        }
        return 1;
    }
    return 0;
}

int random(const parser::NoArgs& args, const parser::Context& context) {
    return 0;
}

} // namespace caelestia::cli::scheme
