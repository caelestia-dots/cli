#pragma once

#include <qlist.h>
#include <qstring.h>
#include <qstringlist.h>
#include <qtclasshelpermacros.h>

#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "spec.hpp"

namespace caelestia::cli::parser {

class Node {
public:
    Q_DISABLE_COPY_MOVE(Node)

    Node() = default;
    virtual ~Node() = default;

    [[nodiscard]] virtual const QString& name() const = 0;
    [[nodiscard]] virtual const QString& help() const = 0;
    [[nodiscard]] virtual int dispatch(const QStringList& args, Context context) const = 0;
};

using NodePtr = std::shared_ptr<const Node>;

struct LeafShape {
    QString help;
    QList<OptionSpec> options;
    QList<PositionalSpec> positionals;
};

using AssignIndexed = std::function<bool(qsizetype, const QString&)>;

[[nodiscard]] std::optional<int> parseLeaf(const LeafShape& shape, const QStringList& args, Context& context,
    const AssignIndexed& assignOption, const AssignIndexed& assignPositional);

int usageError(const Context& context, const QString& message);
int notImplemented(const Context& context);

template <typename T> struct LeafSpec {
    QString name;
    QString help;
    QList<Option<T>> options;
    QList<Positional<T>> positionals;
    std::function<int(const T&, const Context&)> run;
};

template <typename T> class Leaf final : public Node {
public:
    explicit Leaf(LeafSpec<T> spec)
        : m_spec(std::move(spec)) {
        m_shape.help = m_spec.help;
        for (const auto& option : m_spec.options)
            m_shape.options << option.spec;
        for (const auto& positional : m_spec.positionals)
            m_shape.positionals << positional.spec;
    }

    [[nodiscard]] const QString& name() const override { return m_spec.name; }

    [[nodiscard]] const QString& help() const override { return m_spec.help; }

    [[nodiscard]] int dispatch(const QStringList& args, Context context) const override {
        T parsed{};
        const auto exit = parseLeaf(
            m_shape, args, context,
            [&](qsizetype index, const QString& value) {
                return m_spec.options[index].assign(parsed, value);
            },
            [&](qsizetype index, const QString& value) {
                return m_spec.positionals[index].assign(parsed, value);
            });
        if (exit)
            return *exit;
        return m_spec.run(parsed, context);
    }

private:
    LeafSpec<T> m_spec;
    LeafShape m_shape;
};

template <typename T> NodePtr leaf(LeafSpec<T> spec) {
    return std::make_shared<Leaf<T>>(std::move(spec));
}

struct GroupSpec {
    QString name;
    QString help;
    std::vector<NodePtr> children;
    std::function<QString()> version;
};

class Group final : public Node {
public:
    explicit Group(GroupSpec spec)
        : m_spec(std::move(spec)) {}

    [[nodiscard]] const QString& name() const override { return m_spec.name; }

    [[nodiscard]] const QString& help() const override { return m_spec.help; }

    [[nodiscard]] int dispatch(const QStringList& args, Context context) const override;

private:
    [[nodiscard]] const Node* child(const QString& name) const;
    [[nodiscard]] QString usage(const QString& command) const;
    [[nodiscard]] QString helpText(const QString& command) const;

    GroupSpec m_spec;
};

NodePtr group(GroupSpec spec);

} // namespace caelestia::cli::parser
