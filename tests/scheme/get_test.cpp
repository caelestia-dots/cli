#include <caelestia/scheme/scheme.hpp>

#include "scheme/scheme.hpp"
#include "common.hpp"

using namespace caelestia::cli::test;
namespace scheme = caelestia::cli::scheme;
namespace lib = caelestia::scheme;

namespace {

class TestGet : public QObject, protected Sandbox {
    Q_OBJECT

private:
    static Output get(const scheme::GetArgs& args, bool json = false) {
        return capture([&] {
            return scheme::get(args, { .command = u"caelestia scheme get"_s, .json = json });
        });
    }

private slots:

    void initTestCase() { setupEnvironment(); }

    void init() { QFile::remove(lib::statePath()); }

    void failsWhenUnset() {
        const auto result = get({});
        QCOMPARE(result.exit, 1);
        QVERIFY(result.out.isEmpty());
        QCOMPARE(result.err, u"Scheme has not been set or is invalid\n"_s);
    }

    void failsWhenUnsetJson() {
        const auto result = get({}, true);
        QCOMPARE(result.exit, 1);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(result.json().value(u"error"_s).toString(), u"Scheme has not been set or is invalid"_s);
    }

    void failsWhenStateIsInvalid() {
        writeFile(lib::statePath(), R"({"name":"caelestia"})");
        QCOMPARE(get({}).exit, 1);
    }

    void printsAll() {
        seedScheme(u"caelestia"_s, u"default"_s, lib::Mode::Dark);
        const auto result = get({});
        QCOMPARE(result.exit, 0);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(result.out, u"Name: caelestia\nFlavour: default\nMode: dark\n"_s);
    }

    void printsAllJson() {
        seedScheme(u"caelestia"_s, u"default"_s, lib::Mode::Light);
        const auto result = get({}, true);
        QCOMPARE(result.exit, 0);
        const auto json = result.json();
        QCOMPARE(json.value(u"name"_s).toString(), u"caelestia"_s);
        QCOMPARE(json.value(u"flavour"_s).toString(), u"default"_s);
        QCOMPARE(json.value(u"mode"_s).toString(), u"light"_s);
    }

    void printsSingleProperty() {
        seedScheme(u"catppuccin"_s, u"mocha"_s, lib::Mode::Dark);
        QCOMPARE(get({ .name = true }).out, u"catppuccin\n"_s);
        QCOMPARE(get({ .flavour = true }).out, u"mocha\n"_s);
        QCOMPARE(get({ .mode = true }).out, u"dark\n"_s);
    }

    void printsSinglePropertyJson() {
        seedScheme(u"catppuccin"_s, u"mocha"_s, lib::Mode::Dark);
        const auto result = get({ .flavour = true }, true);
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.json(), (QJsonObject{ { u"flavour"_s, u"mocha"_s } }));
    }

    void printsSelectedProperties() {
        seedScheme(u"catppuccin"_s, u"mocha"_s, lib::Mode::Dark);
        const auto result = get({ .name = true, .mode = true });
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.out, u"Name: catppuccin\nMode: dark\n"_s);
    }

    void printsSelectedPropertiesJson() {
        seedScheme(u"catppuccin"_s, u"mocha"_s, lib::Mode::Dark);
        const auto result = get({ .name = true, .mode = true }, true);
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.json(), (QJsonObject{ { u"name"_s, u"catppuccin"_s }, { u"mode"_s, u"dark"_s } }));
    }

    void dispatchesFromParser() {
        seedScheme(u"caelestia"_s, u"default"_s, lib::Mode::Dark);
        const auto result = dispatch(scheme::parserNode(), { u"get"_s, u"-n"_s });
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.out, u"caelestia\n"_s);
        QCOMPARE(dispatch(scheme::parserNode(), { u"get"_s, u"extra"_s }).exit, 2);
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestGet)

#include "get_test.moc"
