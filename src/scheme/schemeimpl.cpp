
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

void printJson(const QJsonObject& json) {
    QTextStream(stdout) << QJsonDocument(json).toJson(QJsonDocument::Indented) << u'\n';
}

int printError(const QString& err, bool json) {
    if (json)
        QTextStream(stdout) << u"{ \"error\": \""_s << err << u"\" }\n"_s;
    else
        QTextStream(stderr) << err << u'\n';
    return 1;
}

void printLine(const QString& key, const QStringList& values, int indent = 0,
    ansi::Codes keyCodes = ansi::Bold | ansi::Cyan, ansi::Codes valueCodes = {}) {
    QTextStream(stdout) << QString(indent, u' ') << ansi::styled(keyCodes, key + u':') << u' '
                        << ansi::styled(valueCodes, values.join(u", "_s)) << u'\n';
}

QJsonObject entry(const QString& name, const QString& key, const QStringList& values) {
    return { { u"name"_s, name }, { key, QJsonArray::fromStringList(values) } };
}

QStringList flavourNames(const lib::Scheme& scheme) {
    QStringList names;
    const auto flavours = scheme.flavours();
    for (const auto& flavour : flavours)
        names << flavour.name();
    return names;
}

QStringList modeNames(const QList<lib::Mode>& modes) {
    QStringList names;
    for (const auto mode : modes)
        names << lib::modeName(mode);
    return names;
}

QList<lib::Mode> schemeModes(const lib::Scheme& scheme) {
    QList<lib::Mode> modes;
    const auto flavours = scheme.flavours();
    for (const auto& flavour : flavours) {
        const auto flavourModes = flavour.modes();
        for (const auto mode : flavourModes)
            if (!modes.contains(mode))
                modes << mode;
    }
    return modes;
}

} // namespace

int list(const ListArgs& args, const parser::Context& context) {
    const auto argCount = args.names + args.flavours + args.modes;
    const auto all = argCount == 0 || argCount == 3;

    QList<std::shared_ptr<lib::Scheme>> schemes;
    std::shared_ptr<lib::Scheme> current;
    std::optional<lib::Flavour> currentFlavour;
    try {
        if (all || args.names) {
            schemes = lib::Scheme::all().takeResult();
            schemes << lib::Scheme::forColour(Qt::black);
        } else {
            const auto state = lib::loadStateRef();
            if (!state)
                return printError(u"Scheme has not been set or is invalid"_s, context.json);
            current = state->name == k_dynamic ? lib::Scheme::forColour(Qt::black)
                                               : lib::Scheme::forName(state->name).takeResult();
            if (!current)
                return printError(u"Scheme \"%1\" does not exist"_s.arg(state->name), context.json);
            currentFlavour = current->flavour(state->flavour);
        }
    } catch (const std::exception& e) {
        return printError(QString::fromUtf8(e.what()), context.json);
    }

    if (argCount == 1) {
        QString key;
        QStringList values;
        if (args.names) {
            key = u"names"_s;
            for (const auto& scheme : std::as_const(schemes))
                values << scheme->name();
        } else if (args.flavours) {
            key = u"flavours"_s;
            values = flavourNames(*current);
        } else {
            if (!currentFlavour)
                return printError(u"Scheme flavour has not been set or is invalid"_s, context.json);
            key = u"modes"_s;
            values = modeNames(currentFlavour->modes());
        }

        if (context.json) {
            printJson({ { key, QJsonArray::fromStringList(values) } });
        } else {
            for (const auto& value : std::as_const(values))
                QTextStream(stdout) << value << u'\n';
        }
        return 0;
    }

    if (all) {
        QJsonArray schemeArr;
        for (const auto& scheme : std::as_const(schemes)) {
            const auto flavours = scheme->flavours();
            if (context.json) {
                QJsonArray flavourArr;
                for (const auto& flavour : flavours)
                    flavourArr << entry(flavour.name(), u"modes"_s, modeNames(flavour.modes()));
                schemeArr << QJsonObject{ { u"name"_s, scheme->name() }, { u"flavours"_s, flavourArr } };
            } else {
                QTextStream(stdout) << ansi::styled(ansi::Bold | ansi::Cyan, scheme->name()) << u'\n';
                for (const auto& flavour : flavours)
                    printLine(flavour.name(), modeNames(flavour.modes()), 2, {}, ansi::Dim);
            }
        }

        if (context.json)
            printJson({ { u"schemes"_s, schemeArr } });

        return 0;
    }

    QString key;
    QString childKey;
    QList<std::pair<QString, QStringList>> entries;
    if (args.names && args.flavours) {
        key = u"schemes"_s;
        childKey = u"flavours"_s;
        for (const auto& scheme : std::as_const(schemes))
            entries << std::pair{ scheme->name(), flavourNames(*scheme) };
    } else if (args.names && args.modes) {
        key = u"schemes"_s;
        childKey = u"modes"_s;
        for (const auto& scheme : std::as_const(schemes))
            entries << std::pair{ scheme->name(), modeNames(schemeModes(*scheme)) };
    } else {
        key = u"flavours"_s;
        childKey = u"modes"_s;
        const auto flavours = current->flavours();
        for (const auto& flavour : flavours)
            entries << std::pair{ flavour.name(), modeNames(flavour.modes()) };
    }

    if (context.json) {
        QJsonArray arr;
        for (const auto& [name, values] : std::as_const(entries))
            arr << entry(name, childKey, values);
        printJson({ { key, arr } });
    } else {
        for (const auto& [name, values] : std::as_const(entries))
            printLine(name, values);
    }

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

    QJsonObject json;
    const auto print = [&json, jsonOut = context.json](const QString& label, const QString& val) {
        if (jsonOut)
            json.insert(label.toLower(), val);
        else
            QTextStream(stdout) << ansi::styled(ansi::Bold | ansi::Cyan, label + u':') << u' ' << val << u'\n';
    };

    const auto printAll = args.name + args.flavour + args.mode == 0;
    if (printAll || args.name)
        print(u"Name"_s, state->name);
    if (printAll || args.flavour)
        print(u"Flavour"_s, state->flavour);
    if (printAll || args.mode)
        print(u"Mode"_s, lib::modeName(state->mode));

    if (context.json)
        printJson(json);

    return 0;
}

int set(const SetArgs& args, const parser::Context& context) {
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
    Q_UNUSED(args);

    SetArgs setArgs;
    try {
        auto* const gen = QRandomGenerator::global();
        const auto schemes = lib::Scheme::all().takeResult();
        if (schemes.isEmpty())
            return printError(u"No schemes available"_s, context.json);

        const auto& scheme = schemes[gen->bounded(schemes.size())];
        const auto flavours = scheme->flavours();
        const auto& flavour = flavours[gen->bounded(flavours.size())];
        const auto modes = flavour.modes();
        setArgs = { .name = scheme->name(), .flavour = flavour.name(), .mode = modes[gen->bounded(modes.size())] };
    } catch (const std::exception& e) {
        return printError(QString::fromUtf8(e.what()), context.json);
    }

    return set(setArgs, context);
}

} // namespace caelestia::cli::scheme
