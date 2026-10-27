#pragma once

#include <qchar.h>
#include <qlist.h>
#include <qstring.h>
#include <qstringlist.h>

#include <functional>
#include <optional>
#include <utility>

namespace caelestia::cli::parser {

struct Context {
    QString command;
    bool json = false;
};

struct NoArgs {};

struct OptionSpec {
    QChar shortName;
    QString longName;
    QString help;
    QString metavar;
    QStringList choices;

    [[nodiscard]] bool isFlag() const { return metavar.isEmpty(); }
};

struct PositionalSpec {
    QString name;
    QString help;
    QStringList choices;
    bool required = false;
    bool variadic = false;
};

template <typename T> using Assign = std::function<bool(T&, const QString&)>;

template <typename T> struct Option {
    OptionSpec spec;
    Assign<T> assign;
};

template <typename T> struct Positional {
    PositionalSpec spec;
    Assign<T> assign;
};

template <typename E> using Choices = QList<std::pair<QString, E>>;

template <typename E> QStringList choiceNames(const Choices<E>& choices) {
    QStringList names;
    for (const auto& [name, value] : choices)
        names << name;
    return names;
}

template <typename T, typename M, typename E> Assign<T> assignChoice(M T::* member, Choices<E> choices) {
    return [member, choices = std::move(choices)](T& args, const QString& value) {
        for (const auto& [name, choice] : choices) {
            if (name == value) {
                args.*member = choice;
                return true;
            }
        }
        return false;
    };
}

template <typename T> Option<T> flag(bool T::* member, QChar shortName, const QString& longName, const QString& help) {
    return {
        {
            .shortName = shortName,
            .longName = longName,
            .help = help,
        },
        [member](T& args, const QString&) {
            args.*member = true;
            return true;
        },
    };
}

template <typename T>
Option<T> value(
    QString T::* member, QChar shortName, const QString& longName, const QString& metavar, const QString& help) {
    return {
        {
            .shortName = shortName,
            .longName = longName,
            .help = help,
            .metavar = metavar,
        },
        [member](T& args, const QString& value) {
            args.*member = value;
            return true;
        },
    };
}

template <typename T>
Option<T> value(std::optional<QString> T::* member, QChar shortName, const QString& longName, const QString& metavar,
    const QString& help) {
    return {
        {
            .shortName = shortName,
            .longName = longName,
            .help = help,
            .metavar = metavar,
        },
        [member](T& args, const QString& value) {
            args.*member = value;
            return true;
        },
    };
}

template <typename T>
Option<T> value(
    double T::* member, QChar shortName, const QString& longName, const QString& metavar, const QString& help) {
    return {
        {
            .shortName = shortName,
            .longName = longName,
            .help = help,
            .metavar = metavar,
        },
        [member](T& args, const QString& value) {
            bool ok = false;
            args.*member = value.toDouble(&ok);
            return ok;
        },
    };
}

template <typename T, typename E>
Option<T> choice(std::optional<E> T::* member, QChar shortName, const QString& longName, const QString& metavar,
    const QString& help, Choices<E> choices) {
    return {
        {
            .shortName = shortName,
            .longName = longName,
            .help = help,
            .metavar = metavar,
            .choices = choiceNames(choices),
        },
        assignChoice<T, std::optional<E>, E>(member, std::move(choices)),
    };
}

template <typename T> Positional<T> positional(QString T::* member, const QString& name, const QString& help) {
    return {
        {
            .name = name,
            .help = help,
            .required = true,
        },
        [member](T& args, const QString& value) {
            args.*member = value;
            return true;
        },
    };
}

template <typename T>
Positional<T> positional(std::optional<QString> T::* member, const QString& name, const QString& help) {
    return {
        {
            .name = name,
            .help = help,
        },
        [member](T& args, const QString& value) {
            args.*member = value;
            return true;
        },
    };
}

template <typename T, typename E>
Positional<T> positional(E T::* member, const QString& name, const QString& help, Choices<E> choices) {
    return {
        {
            .name = name,
            .help = help,
            .choices = choiceNames(choices),
        },
        assignChoice<T, E, E>(member, std::move(choices)),
    };
}

template <typename T> Positional<T> rest(QStringList T::* member, const QString& name, const QString& help) {
    return {
        {
            .name = name,
            .help = help,
            .variadic = true,
        },
        [member](T& args, const QString& value) {
            (args.*member) << value;
            return true;
        },
    };
}

} // namespace caelestia::cli::parser
