#include <qstringlist.h>
#include <qtextstream.h>

#include <algorithm>
#include <cstdio>

#include "node.hpp"

namespace caelestia::cli::parser {

using Qt::StringLiterals::operator""_s;

namespace {

constexpr qsizetype k_helpColumn = 24;
constexpr auto k_indent = u"  ";

enum class Global : quint8 {
    Help,
    Version,
    Json,
};

struct GlobalSpec {
    Global id;
    OptionSpec spec;
};

const QList<GlobalSpec>& globalSpecs() {
    static const QList<GlobalSpec> k_specs = {
        {
            .id = Global::Help,
            .spec = { .shortName = u'h', .longName = u"help"_s, .help = u"show this help message and exit"_s },
        },
        {
            .id = Global::Version,
            .spec = { .shortName = u'v', .longName = u"version"_s, .help = u"print the current version"_s },
        },
        {
            .id = Global::Json,
            .spec = { .longName = u"json"_s, .help = u"print machine-readable output where supported"_s },
        },
    };
    return k_specs;
}

struct Found {
    std::optional<Global> global;
    qsizetype option = -1;
    const OptionSpec* spec = nullptr;
};

template <typename Pred> const GlobalSpec* findGlobal(bool allowVersion, Pred matches) {
    for (const auto& global : globalSpecs()) {
        if (global.id == Global::Version && !allowVersion)
            continue;
        if (matches(global.spec))
            return &global;
    }
    return nullptr;
}

template <typename Pred> std::optional<Found> find(const QList<OptionSpec>& options, bool allowVersion, Pred matches) {
    for (qsizetype i = 0; i < options.size(); ++i) {
        if (matches(options[i]))
            return Found{ .option = i, .spec = &options[i] };
    }

    if (const auto* global = findGlobal(allowVersion, matches))
        return Found{ .global = global->id, .spec = &global->spec };

    return std::nullopt;
}

QChar normaliseShort(QChar c, bool allowVersion) {
    return c == u'V' && allowVersion ? u'v' : c;
}

std::optional<Found> findLong(const QList<OptionSpec>& options, bool allowVersion, const QString& name) {
    return find(options, allowVersion, [&](const OptionSpec& spec) {
        return spec.longName == name;
    });
}

std::optional<Found> findShort(const QList<OptionSpec>& options, bool allowVersion, QChar c) {
    const auto wanted = normaliseShort(c, allowVersion);
    return find(options, allowVersion, [&](const OptionSpec& spec) {
        return spec.shortName == wanted;
    });
}

std::optional<QList<Global>> globalFlags(const QString& arg, bool allowVersion, QString& unknown) {
    QList<Global> flags;
    if (arg.startsWith(u"--"_s)) {
        const auto name = arg.mid(2);
        const auto* found = findGlobal(allowVersion, [&](const OptionSpec& spec) {
            return spec.longName == name;
        });
        if (found == nullptr) {
            unknown = arg;
            return std::nullopt;
        }

        flags << found->id;
        return flags;
    }

    for (qsizetype j = 1; j < arg.size(); ++j) {
        const auto wanted = normaliseShort(arg[j], allowVersion);
        const auto* found = findGlobal(allowVersion, [&](const OptionSpec& spec) {
            return spec.shortName == wanted;
        });
        if (found == nullptr) {
            unknown = u"-"_s + arg[j];
            return std::nullopt;
        }

        flags << found->id;
    }

    return flags;
}

QString optionLabel(const OptionSpec& spec) {
    QString label;
    if (!spec.shortName.isNull())
        label += u'-' + QString(spec.shortName) + u", "_s;

    label += u"--"_s + spec.longName;

    if (!spec.isFlag())
        label += u' ' + spec.metavar;

    return label;
}

QString optionUsage(const OptionSpec& spec) {
    auto usage = spec.shortName.isNull() ? u"--"_s + spec.longName : u'-' + QString(spec.shortName);
    if (!spec.isFlag())
        usage += u' ' + spec.metavar;
    return u'[' + usage + u']';
}

QString positionalName(const PositionalSpec& spec) {
    if (!spec.choices.isEmpty())
        return u'{' + spec.choices.join(u',') + u'}';
    return spec.name;
}

QString positionalUsage(const PositionalSpec& spec) {
    auto name = positionalName(spec);
    if (spec.required)
        return name;
    if (spec.variadic)
        return u'[' + name + u" ...]"_s;
    return u'[' + name + u']';
}

QString rows(const QList<std::pair<QString, QString>>& entries) {
    qsizetype width = 0;
    for (const auto& [label, help] : entries)
        width = std::max(width, label.size());
    width = std::min(width, k_helpColumn);

    QString text;
    for (const auto& [label, help] : entries) {
        text += k_indent + label;
        if (label.size() > width)
            text += u'\n' + QString(width + 4, u' ');
        else
            text += QString(width - label.size() + 2, u' ');
        text += help + u'\n';
    }
    return text;
}

QString leafUsage(const QString& command, const LeafShape& shape) {
    QStringList parts{ u"usage:"_s, command, u"[-h]"_s };
    for (const auto& option : shape.options)
        parts << optionUsage(option);
    for (const auto& positional : shape.positionals)
        parts << positionalUsage(positional);
    return parts.join(u' ') + u'\n';
}

QString leafHelp(const QString& command, const LeafShape& shape) {
    auto text = leafUsage(command, shape) + u'\n' + shape.help + u'\n';

    if (!shape.positionals.isEmpty()) {
        QList<std::pair<QString, QString>> entries;
        for (const auto& positional : shape.positionals)
            entries.emplaceBack(positionalName(positional), positional.help);
        text += u"\npositional arguments:\n"_s + rows(entries);
    }

    const auto& helpSpec = globalSpecs().first().spec;
    QList<std::pair<QString, QString>> entries = {
        { optionLabel(helpSpec), helpSpec.help },
    };
    for (const auto& option : shape.options)
        entries.emplaceBack(optionLabel(option), option.help);
    text += u"\noptions:\n"_s + rows(entries);

    return text;
}

int fail(const QString& usage, const Context& context, const QString& message) {
    QTextStream(stderr) << usage << context.command << u": error: "_s << message << u'\n';
    return 2;
}

int print(FILE* stream, const QString& text) {
    QTextStream(stream) << text;
    return stream == stdout ? 0 : 2;
}

struct PositionalLimit {
    qsizetype required = 0;
    qsizetype max = 0;
    bool variadic = false;
};

PositionalLimit positionalLimit(const QList<PositionalSpec>& positionals) {
    PositionalLimit limit;
    for (const auto& spec : positionals) {
        if (spec.required)
            ++limit.required;
        if (spec.variadic)
            limit.variadic = true;
        else
            ++limit.max;
    }
    return limit;
}

class LeafParser {
public:
    LeafParser(const LeafShape& shape, Context& context, const AssignIndexed& assignOption,
        const AssignIndexed& assignPositional)
        : m_shape(shape)
        , m_context(context)
        , m_assignOption(assignOption)
        , m_assignPositional(assignPositional)
        , m_usage(leafUsage(context.command, shape)) {}

    std::optional<int> parse(const QStringList& args) {
        QStringList positionals;
        auto onlyPositionals = false;

        for (qsizetype i = 0; i < args.size(); ++i) {
            const QString& arg = args[i];
            if (onlyPositionals || arg.size() < 2 || !arg.startsWith(u'-')) {
                positionals << arg;
            } else if (arg == u"--"_s) {
                onlyPositionals = true;
            } else if (arg.startsWith(u"--"_s)) {
                if (const auto exit = parseLong(arg, args, i))
                    return exit;
            } else if (const auto exit = parseShortCluster(arg, args, i)) {
                return exit;
            }
        }

        return assignPositionals(positionals);
    }

private:
    std::optional<int> parseLong(const QString& arg, const QStringList& args, qsizetype& i) {
        const auto eq = arg.indexOf(u'=');
        const auto name = arg.mid(2, eq == -1 ? -1 : eq - 2);
        const auto found = findLong(m_shape.options, false, name);
        if (!found)
            return fail(m_usage, m_context, u"unrecognized arguments: "_s + arg);

        std::optional<QString> value;
        if (eq != -1)
            value = arg.mid(eq + 1);

        if (found->spec->isFlag() && value)
            return fail(m_usage, m_context, u"argument --%1: ignored explicit argument '%2'"_s.arg(name, *value));

        if (!found->spec->isFlag() && !value) {
            if (++i >= args.size())
                return fail(m_usage, m_context, u"argument --%1: expected one argument"_s.arg(name));

            value = args[i];
        }

        return apply(*found, value.value_or(QString()));
    }

    std::optional<int> parseShortCluster(const QString& arg, const QStringList& args, qsizetype& i) {
        for (qsizetype j = 1; j < arg.size(); ++j) {
            const auto c = arg[j];
            const auto found = findShort(m_shape.options, false, c);
            if (!found)
                return fail(m_usage, m_context, u"unrecognized arguments: -"_s + c);

            if (found->spec->isFlag()) {
                if (const auto exit = apply(*found, {}))
                    return exit;
                continue;
            }

            auto value = arg.mid(j + 1);
            if (value.isEmpty()) {
                if (++i >= args.size())
                    return fail(m_usage, m_context, u"argument -%1: expected one argument"_s.arg(c));

                value = args[i];
            }

            return apply(*found, value);
        }

        return std::nullopt;
    }

    std::optional<int> apply(const Found& found, const QString& value) {
        if (found.global) {
            switch (*found.global) {
            case Global::Help:
                return print(stdout, leafHelp(m_context.command, m_shape));
            case Global::Json:
                m_context.json = true;
                break;
            case Global::Version:
                break;
            }
            return std::nullopt;
        }

        const auto& spec = *found.spec;
        if (!spec.choices.isEmpty() && !spec.choices.contains(value))
            return fail(m_usage, m_context,
                u"argument --%1: invalid choice: '%2' (choose from %3)"_s.arg(
                    spec.longName, value, spec.choices.join(u", ")));

        if (!m_assignOption(found.option, value))
            return fail(m_usage, m_context, u"argument --%1: invalid value: '%1'"_s.arg(spec.longName, value));

        return std::nullopt;
    }

    std::optional<int> assignPositionals(const QStringList& values) {
        const auto limit = positionalLimit(m_shape.positionals);
        if (values.size() < limit.required) {
            QStringList missing;
            for (const auto& spec : m_shape.positionals)
                if (spec.required)
                    missing << spec.name;

            return fail(m_usage, m_context, u"the following arguments are required: "_s + missing.join(u", "));
        }

        if (!limit.variadic && values.size() > limit.max)
            return fail(m_usage, m_context, u"unrecognized arguments: "_s + values.mid(limit.max).join(u' '));

        qsizetype index = 0;
        for (const auto& value : values) {
            const PositionalSpec& spec = m_shape.positionals[index];
            if (!spec.choices.isEmpty() && !spec.choices.contains(value))
                return fail(m_usage, m_context,
                    u"argument %1: invalid choice: '%2' (choose from %3)"_s.arg(
                        spec.name, value, spec.choices.join(u", ")));

            if (!m_assignPositional(index, value))
                return fail(m_usage, m_context, u"argument %1: invalid value: '%2'"_s.arg(spec.name, value));

            if (!spec.variadic)
                ++index;
        }

        return std::nullopt;
    }

    const LeafShape& m_shape;
    Context& m_context;
    const AssignIndexed& m_assignOption;
    const AssignIndexed& m_assignPositional;
    QString m_usage;
};

} // namespace

std::optional<int> parseLeaf(const LeafShape& shape, const QStringList& args, Context& context,
    const AssignIndexed& assignOption, const AssignIndexed& assignPositional) {
    return LeafParser(shape, context, assignOption, assignPositional).parse(args);
}

int usageError(const Context& context, const QString& message) {
    QTextStream(stderr) << context.command << u": error: "_s << message << u'\n';
    return 2;
}

int notImplemented(const Context& context) {
    QTextStream(stderr) << context.command << u": not implemented\n"_s;
    return 1;
}

const Node* Group::child(const QString& name) const {
    for (const auto& node : m_spec.children)
        if (node->name() == name)
            return node.get();
    return nullptr;
}

QString Group::usage(const QString& command) const {
    QStringList parts{ u"usage:"_s, command, u"[-h]"_s };

    if (m_spec.version) {
        for (const auto& global : globalSpecs())
            if (global.id != Global::Help)
                parts << optionUsage(global.spec);
    }

    parts << u"COMMAND ..."_s;
    return parts.join(u' ') + u'\n';
}

QString Group::helpText(const QString& command) const {
    QList<std::pair<QString, QString>> commands;
    for (const auto& node : m_spec.children)
        commands.emplaceBack(node->name(), node->help());

    QList<std::pair<QString, QString>> options;
    for (const auto& global : globalSpecs()) {
        if (global.id == Global::Help || m_spec.version)
            options.emplaceBack(optionLabel(global.spec), global.spec.help);
    }

    return u"%1\n%2\n\ncommands:\n%3\noptions:\n%4"_s.arg(usage(command), m_spec.help, rows(commands), rows(options));
}

int Group::dispatch(const QStringList& args, Context context) const {
    const auto root = static_cast<bool>(m_spec.version);
    const auto usageText = usage(context.command);

    for (qsizetype i = 0; i < args.size(); ++i) {
        const auto& arg = args[i];
        if (arg.size() < 2 || !arg.startsWith(u'-')) {
            const auto* next = child(arg);
            if (next == nullptr) {
                QStringList names;
                for (const auto& node : m_spec.children)
                    names << node->name();

                return fail(usageText, context,
                    u"argument COMMAND: invalid choice: '%1' (choose from %2)"_s.arg(arg, names.join(u", "_s)));
            }

            context.command += u' ' + next->name();
            return next->dispatch(args.mid(i + 1), std::move(context));
        }

        QString unknown;
        const auto flags = globalFlags(arg, root, unknown);
        if (!flags)
            return fail(usageText, context, u"unrecognized arguments: "_s + unknown);

        for (const Global flag : *flags) {
            switch (flag) {
            case Global::Help:
                return print(stdout, helpText(context.command));
            case Global::Version:
                return print(stdout, m_spec.version() + u'\n');
            case Global::Json:
                context.json = true;
                break;
            }
        }
    }

    return print(stderr, helpText(context.command));
}

NodePtr group(GroupSpec spec) {
    return std::make_shared<Group>(std::move(spec));
}

} // namespace caelestia::cli::parser
