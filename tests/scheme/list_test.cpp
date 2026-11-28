#include <qjsonarray.h>

#include <caelestia/scheme/scheme.hpp>

#include "scheme/scheme.hpp"
#include "common.hpp"

using namespace caelestia::cli::test;
namespace scheme = caelestia::cli::scheme;
namespace lib = caelestia::scheme;

namespace {

class TestList : public QObject, protected Sandbox {
    Q_OBJECT

private:
    static Output list(const scheme::ListArgs& args, bool json = false) {
        return capture([&] {
            return scheme::list(args, { .command = u"caelestia scheme list"_s, .json = json });
        });
    }

    static QJsonObject entry(const QJsonArray& arr, const QString& name) {
        for (const auto& value : arr)
            if (value.toObject().value(u"name"_s).toString() == name)
                return value.toObject();
        return {};
    }

private slots:

    void initTestCase() { setupEnvironment(); }

    void init() { QFile::remove(lib::statePath()); }

    void listsAll() {
        const auto result = list({});
        QCOMPARE(result.exit, 0);
        QVERIFY(result.err.isEmpty());
        QVERIFY(result.out.contains(u"caelestia\n  default: light, dark\n"_s));
        QVERIFY(result.out.contains(u"catppuccin\n  frappe: dark\n  latte: light\n"_s));
        QVERIFY(result.out.endsWith(u"  cmf: light, dark\n"_s));
    }

    void listsAllJson() {
        const auto result = list(scheme::ListArgs(), true);
        QCOMPARE(result.exit, 0);
        const auto schemes = result.json().value(u"schemes"_s).toArray();
        QCOMPARE(schemes.last().toObject().value(u"name"_s).toString(), u"dynamic"_s);

        const auto flavours = entry(schemes, u"catppuccin"_s).value(u"flavours"_s).toArray();
        QCOMPARE(flavours.size(), 4);
        QCOMPARE(entry(flavours, u"latte"_s).value(u"modes"_s).toArray(), QJsonArray{ u"light"_s });
    }

    void allFlagsMatchNoFlags() { QCOMPARE(list({ true, true, true }, true).out, list(scheme::ListArgs(), true).out); }

    void listsNames() {
        const auto result = list({ .names = true });
        QCOMPARE(result.exit, 0);
        const auto names = result.out.split(u'\n', Qt::SkipEmptyParts);
        QVERIFY(names.contains(u"caelestia"_s));
        QVERIFY(names.contains(u"catppuccin"_s));
        QCOMPARE(names.last(), u"dynamic"_s);
        QVERIFY(std::ranges::is_sorted(names.first(names.size() - 1)));
    }

    void listsNamesJson() {
        const auto result = list({ .names = true }, true);
        QCOMPARE(result.exit, 0);
        const auto names = result.json().value(u"names"_s).toArray();
        QVERIFY(names.contains(u"gruvbox"_s));
        QCOMPARE(names.last().toString(), u"dynamic"_s);
    }

    void listsFlavoursOfCurrent() {
        seedScheme(u"catppuccin"_s, u"mocha"_s, lib::Mode::Dark);
        const auto result = list({ .flavours = true });
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.out, u"frappe\nlatte\nmacchiato\nmocha\n"_s);
    }

    void listsFlavoursOfCurrentJson() {
        seedScheme(u"catppuccin"_s, u"mocha"_s, lib::Mode::Dark);
        const auto result = list({ .flavours = true }, true);
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.json().value(u"flavours"_s).toArray(),
            (QJsonArray{ u"frappe"_s, u"latte"_s, u"macchiato"_s, u"mocha"_s }));
    }

    void listsFlavoursOfDynamic() {
        seedScheme(u"dynamic"_s, u"vibrant"_s, lib::Mode::Dark);
        const auto result = list({ .flavours = true });
        QCOMPARE(result.exit, 0);
        QVERIFY(result.out.contains(u"tonalspot\n"_s));
        QVERIFY(result.out.contains(u"vibrant\n"_s));
    }

    void listsModesOfCurrent() {
        seedScheme(u"caelestia"_s, u"default"_s, lib::Mode::Dark);
        QCOMPARE(list({ .modes = true }).out, u"light\ndark\n"_s);
        seedScheme(u"catppuccin"_s, u"latte"_s, lib::Mode::Light);
        QCOMPARE(list({ .modes = true }).out, u"light\n"_s);
    }

    void listsModesOfCurrentJson() {
        seedScheme(u"catppuccin"_s, u"mocha"_s, lib::Mode::Dark);
        const auto result = list({ .modes = true }, true);
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.json().value(u"modes"_s).toArray(), QJsonArray{ u"dark"_s });
    }

    void listsNamesAndFlavours() {
        const auto result = list({ .names = true, .flavours = true });
        QCOMPARE(result.exit, 0);
        QVERIFY(result.out.contains(u"catppuccin: frappe, latte, macchiato, mocha\n"_s));
        QVERIFY(result.out.contains(u"caelestia: default\n"_s));
    }

    void listsNamesAndFlavoursJson() {
        const auto result = list({ .names = true, .flavours = true }, true);
        QCOMPARE(result.exit, 0);
        const auto schemes = result.json().value(u"schemes"_s).toArray();
        QCOMPARE(entry(schemes, u"gruvbox"_s).value(u"flavours"_s).toArray(),
            (QJsonArray{ u"hard"_s, u"medium"_s, u"soft"_s }));
    }

    void listsNamesAndModes() {
        const auto result = list({ .names = true, .modes = true });
        QCOMPARE(result.exit, 0);
        QVERIFY(result.out.contains(u"caelestia: light, dark\n"_s));
        QVERIFY(result.out.contains(u"catppuccin: dark, light\n"_s));
        QVERIFY(result.out.contains(u"dracula: dark\n"_s));
    }

    void listsNamesAndModesJson() {
        const auto result = list({ .names = true, .modes = true }, true);
        QCOMPARE(result.exit, 0);
        const auto schemes = result.json().value(u"schemes"_s).toArray();
        QCOMPARE(entry(schemes, u"dracula"_s).value(u"modes"_s).toArray(), QJsonArray{ u"dark"_s });
    }

    void listsFlavoursAndModes() {
        seedScheme(u"catppuccin"_s, u"mocha"_s, lib::Mode::Dark);
        const auto result = list({ .flavours = true, .modes = true });
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.out, u"frappe: dark\nlatte: light\nmacchiato: dark\nmocha: dark\n"_s);
    }

    void listsFlavoursAndModesJson() {
        seedScheme(u"catppuccin"_s, u"mocha"_s, lib::Mode::Dark);
        const auto result = list({ .flavours = true, .modes = true }, true);
        QCOMPARE(result.exit, 0);
        const auto flavours = result.json().value(u"flavours"_s).toArray();
        QCOMPARE(flavours.size(), 4);
        QCOMPARE(entry(flavours, u"latte"_s).value(u"modes"_s).toArray(), QJsonArray{ u"light"_s });
    }

    void failsWhenUnset() {
        const auto result = list({ .flavours = true });
        QCOMPARE(result.exit, 1);
        QVERIFY(result.out.isEmpty());
        QCOMPARE(result.err, u"Scheme has not been set or is invalid\n"_s);
    }

    void failsWhenUnsetJson() {
        const auto result = list({ .modes = true }, true);
        QCOMPARE(result.exit, 1);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(result.json().value(u"error"_s).toString(), u"Scheme has not been set or is invalid"_s);
    }

    void failsWhenSchemeIsMissing() {
        seedScheme(u"nope"_s, u"default"_s, lib::Mode::Dark);
        const auto result = list({ .flavours = true });
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.err, u"Scheme \"nope\" does not exist\n"_s);
    }

    void failsWhenFlavourIsMissing() {
        seedScheme(u"caelestia"_s, u"nope"_s, lib::Mode::Dark);
        const auto result = list({ .modes = true });
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.err, u"Scheme flavour has not been set or is invalid\n"_s);
    }

    void dispatchesFromParser() {
        seedScheme(u"catppuccin"_s, u"mocha"_s, lib::Mode::Dark);
        const auto result = dispatch(scheme::parserNode(), { u"list"_s, u"-f"_s });
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.out, u"frappe\nlatte\nmacchiato\nmocha\n"_s);
        QCOMPARE(dispatch(scheme::parserNode(), { u"list"_s, u"extra"_s }).exit, 2);
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestList)

#include "list_test.moc"
