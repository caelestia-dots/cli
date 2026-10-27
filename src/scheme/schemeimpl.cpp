
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
constexpr QStringView k_noState = u"Scheme has not been set or is invalid";

using Entries = QList<std::pair<QString, QStringList>>;

void printJson(const QJsonObject& json) {
    QTextStream(stdout) << QJsonDocument(json).toJson(QJsonDocument::Indented) << u'\n';
}

void printWarnings(const QStringList& warnings) {
    for (const auto& warning : warnings)
        QTextStream(stderr) << warning << u'\n';
}

int printError(const QString& err, bool json, const QStringList& warnings = {}) {
    if (json) {
        QJsonObject obj{ { u"error"_s, err } };
        if (!warnings.isEmpty())
            obj.insert(u"warnings"_s, QJsonArray::fromStringList(warnings));
        printJson(obj);
    } else {
        printWarnings(warnings);
        QTextStream(stderr) << err << u'\n';
    }
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

Entries flavourEntries(const lib::Scheme& scheme) {
    Entries entries;
    const auto flavours = scheme.flavours();
    for (const auto& flavour : flavours)
        entries << std::pair{ flavour.name(), modeNames(flavour.modes()) };
    return entries;
}

std::shared_ptr<lib::Scheme> schemeForName(const QString& name) {
    return name == k_dynamic ? lib::Scheme::forColour(Qt::black) : lib::Scheme::forName(name).takeResult();
}

void printValues(const QString& key, const QStringList& values, bool json) {
    if (json) {
        printJson({ { key, QJsonArray::fromStringList(values) } });
        return;
    }

    for (const auto& value : values)
        QTextStream(stdout) << value << u'\n';
}

void printEntries(const QString& key, const QString& childKey, const Entries& entries, bool json) {
    if (json) {
        QJsonArray arr;
        for (const auto& [name, values] : entries)
            arr << entry(name, childKey, values);
        printJson({ { key, arr } });
        return;
    }

    for (const auto& [name, values] : entries)
        printLine(name, values);
}

void printAll(const QList<std::shared_ptr<lib::Scheme>>& schemes, bool json) {
    QJsonArray schemeArr;
    for (const auto& scheme : schemes) {
        const auto entries = flavourEntries(*scheme);
        if (json) {
            QJsonArray flavourArr;
            for (const auto& [name, modes] : entries)
                flavourArr << entry(name, u"modes"_s, modes);
            schemeArr << QJsonObject{ { u"name"_s, scheme->name() }, { u"flavours"_s, flavourArr } };
        } else {
            QTextStream(stdout) << ansi::styled(ansi::Bold | ansi::Cyan, scheme->name()) << u'\n';
            for (const auto& [name, modes] : entries)
                printLine(name, modes, 2, {}, ansi::Dim);
        }
    }

    if (json)
        printJson({ { u"schemes"_s, schemeArr } });
}

int listCurrent(const ListArgs& args, const parser::Context& context) {
    const auto state = lib::loadStateRef();
    if (!state)
        return printError(k_noState.toString(), context.json);

    const auto current = schemeForName(state->name);
    if (!current)
        return printError(u"Scheme \"%1\" does not exist"_s.arg(state->name), context.json);

    if (args.flavours && args.modes) {
        printEntries(u"flavours"_s, u"modes"_s, flavourEntries(*current), context.json);
    } else if (args.flavours) {
        printValues(u"flavours"_s, flavourNames(*current), context.json);
    } else {
        const auto flavour = current->flavour(state->flavour);
        if (!flavour)
            return printError(u"Scheme flavour has not been set or is invalid"_s, context.json);

        printValues(u"modes"_s, modeNames(flavour->modes()), context.json);
    }

    return 0;
}

int listNamed(const ListArgs& args, const parser::Context& context) {
    auto schemes = lib::Scheme::all().takeResult();
    schemes << lib::Scheme::forColour(Qt::black);

    if (args.flavours == args.modes) {
        if (!args.flavours) {
            QStringList names;
            for (const auto& scheme : std::as_const(schemes))
                names << scheme->name();
            printValues(u"names"_s, names, context.json);
        } else {
            printAll(schemes, context.json);
        }
        return 0;
    }

    Entries entries;
    for (const auto& scheme : std::as_const(schemes))
        entries << std::pair{ scheme->name(), args.flavours ? flavourNames(*scheme) : modeNames(schemeModes(*scheme)) };
    printEntries(u"schemes"_s, args.flavours ? u"flavours"_s : u"modes"_s, entries, context.json);
    return 0;
}

std::shared_ptr<lib::Scheme> resolveScheme(const std::optional<QString>& requested, QStringList& warnings) {
    auto name = requested;
    if (name && *name != k_dynamic && !std::ranges::any_of(lib::Scheme::all().takeResult(), [&name](const auto& s) {
            return s->name() == *name;
        })) {
        warnings << u"Scheme \"%1\" does not exist"_s.arg(*name);
        name.reset();
    }

    if (!name) {
        const auto state = lib::loadStateRef();
        name = state ? state->name : lib::defaultScheme().takeResult().name;
    }

    return *name == k_dynamic ? lib::Scheme::forMedia(wallpaper::currentWallpaper()).takeResult()
                              : lib::Scheme::forName(*name).takeResult();
}

lib::Flavour resolveFlavour(const lib::Scheme& scheme, const std::optional<QString>& requested, QStringList& warnings) {
    if (requested) {
        if (auto flavour = scheme.flavour(*requested))
            return *flavour;
        warnings << u"\"%1\" is not a flavour of scheme %2"_s.arg(*requested, scheme.name());
    }

    if (const auto state = lib::loadStateRef())
        if (auto flavour = scheme.flavour(state->flavour))
            return *flavour;

    if (scheme.name() == k_dynamic)
        if (auto flavour = scheme.flavour(
                lib::smartChoice(media::analyseForPath(wallpaper::currentWallpaper()).takeResult()).flavour))
            return *flavour;

    return scheme.defaultFlavour();
}

lib::Mode resolveMode(const QString& schemeName, const lib::Flavour& flavour, const std::optional<lib::Mode>& requested,
    QStringList& warnings) {
    if (requested && flavour.modes().contains(*requested))
        return *requested;

    if (requested)
        warnings << u"\"%1\" is not a mode of scheme %2 (%3)"_s.arg(
            lib::modeName(*requested), schemeName, flavour.name());

    return flavour.defaultMode();
}

} // namespace

int list(const ListArgs& args, const parser::Context& context) {
    const auto argCount = args.names + args.flavours + args.modes;
    const auto listAll = argCount == 0 || argCount == 3;
    try {
        return listAll || args.names
                   ? listNamed(listAll ? ListArgs{ .names = true, .flavours = true, .modes = true } : args, context)
                   : listCurrent(args, context);
    } catch (const std::exception& e) {
        return printError(QString::fromUtf8(e.what()), context.json);
    }
}

int get(const GetArgs& args, const parser::Context& context) {
    const auto state = lib::loadStateRef();
    if (!state)
        return printError(k_noState.toString(), context.json);

    const auto single = args.name + args.flavour + args.mode == 1;
    const auto printAll = args.name + args.flavour + args.mode == 0;

    QJsonObject json;
    const auto print = [&](const QString& label, const QString& val) {
        if (context.json)
            json.insert(label.toLower(), val);
        else if (single)
            QTextStream(stdout) << val << u'\n';
        else
            QTextStream(stdout) << ansi::styled(ansi::Bold | ansi::Cyan, label + u':') << u' ' << val << u'\n';
    };

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
        const auto scheme = resolveScheme(args.name, warnings);
        const auto flavour = resolveFlavour(*scheme, args.flavour, warnings);
        const auto mode = resolveMode(scheme->name(), flavour, args.mode, warnings);

        theme::Theme(flavour.data(mode).takeResult()).apply().waitForFinished();

        if (context.json) {
            printJson({
                { u"name"_s, scheme->name() },
                { u"flavour"_s, flavour.name() },
                { u"mode"_s, lib::modeName(mode) },
                { u"warnings"_s, QJsonArray::fromStringList(warnings) },
            });
        } else {
            printWarnings(warnings);
            QTextStream(stdout) << u"Scheme set to %1 (%2, %3)\n"_s.arg(
                scheme->name(), flavour.name(), lib::modeName(mode));
        }
    } catch (const std::exception& e) {
        return printError(QString::fromUtf8(e.what()), context.json, warnings);
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
