#include <qjsonarray.h>

#include <caelestia/scheme/scheme.hpp>
#include <caelestia/wallpaper/wallpaper.hpp>

#include "scheme/scheme.hpp"
#include "common.hpp"

using namespace caelestia::cli::test;
namespace scheme = caelestia::cli::scheme;
namespace lib = caelestia::scheme;
using lib::Mode;

namespace {

class TestSet : public QObject, protected Sandbox {
    Q_OBJECT

private:
    QString m_image;

    static Output set(const scheme::SetArgs& args, bool json = false) {
        return capture([&] {
            return scheme::set(args, { .command = u"caelestia scheme set"_s, .json = json });
        });
    }

    [[nodiscard]] static lib::SchemeRef current() {
        const auto state = lib::loadStateRef();
        if (!state)
            throw std::runtime_error("no scheme state");
        return *state;
    }

    static void verifyCurrent(const QString& name, const QString& flavour, Mode mode) {
        const auto state = current();
        QCOMPARE(state.name, name);
        QCOMPARE(state.flavour, flavour);
        QCOMPARE(state.mode, mode);
    }

private slots:

    void initTestCase() {
        setupEnvironment();
        m_image = saveImage(path(u"blue.png"_s));
        QVERIFY(!m_image.isEmpty());
    }

    void init() {
        QFile::remove(lib::statePath());
        QFile::remove(caelestia::wallpaper::statePath());
    }

    void setsScheme() {
        const auto result = set({ .name = u"catppuccin"_s, .flavour = u"mocha"_s, .mode = Mode::Dark });
        QCOMPARE(result.exit, 0);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(result.out, u"Scheme set to catppuccin (mocha, dark)\n"_s);
        verifyCurrent(u"catppuccin"_s, u"mocha"_s, Mode::Dark);
    }

    void setsSchemeJson() {
        const auto result = set({ .name = u"catppuccin"_s, .flavour = u"latte"_s }, true);
        QCOMPARE(result.exit, 0);
        const auto json = result.json();
        QCOMPARE(json.value(u"name"_s).toString(), u"catppuccin"_s);
        QCOMPARE(json.value(u"flavour"_s).toString(), u"latte"_s);
        QCOMPARE(json.value(u"mode"_s).toString(), u"light"_s);
        QVERIFY(json.value(u"warnings"_s).toArray().isEmpty());
    }

    void defaultsFlavourAndMode() {
        QCOMPARE(set({ .name = u"gruvbox"_s }).out, u"Scheme set to gruvbox (hard, dark)\n"_s);
        verifyCurrent(u"gruvbox"_s, u"hard"_s, Mode::Dark);
    }

    void keepsCurrentFlavourWhenAvailable() {
        seedScheme(u"gruvbox"_s, u"soft"_s, Mode::Dark);
        QCOMPARE(set({ .name = u"everforest"_s }).out, u"Scheme set to everforest (soft, dark)\n"_s);
    }

    void keepsCurrentSchemeWhenOnlyModeGiven() {
        seedScheme(u"caelestia"_s, u"default"_s, Mode::Dark);
        QCOMPARE(set({ .mode = Mode::Light }).exit, 0);
        verifyCurrent(u"caelestia"_s, u"default"_s, Mode::Light);
    }

    void fallsBackToDefaultSchemeWhenUnset() {
        QCOMPARE(set({ .mode = Mode::Light }).out, u"Scheme set to caelestia (default, light)\n"_s);
    }

    void warnsOnUnknownScheme() {
        seedScheme(u"catppuccin"_s, u"mocha"_s, Mode::Dark);
        const auto result = set({ .name = u"nope"_s });
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.err, u"Scheme \"nope\" does not exist\n"_s);
        QCOMPARE(result.out, u"Scheme set to catppuccin (mocha, dark)\n"_s);
    }

    void warnsOnUnknownFlavour() {
        const auto result = set({ .name = u"catppuccin"_s, .flavour = u"nope"_s });
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.err, u"\"nope\" is not a flavour of scheme catppuccin\n"_s);
        verifyCurrent(u"catppuccin"_s, u"frappe"_s, Mode::Dark);
    }

    void warnsOnUnknownMode() {
        const auto result = set({ .name = u"catppuccin"_s, .flavour = u"mocha"_s, .mode = Mode::Light });
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.err, u"\"light\" is not a mode of scheme catppuccin (mocha)\n"_s);
        verifyCurrent(u"catppuccin"_s, u"mocha"_s, Mode::Dark);
    }

    void warnsJson() {
        const auto result = set({ .name = u"nope"_s, .flavour = u"nope"_s }, true);
        QCOMPARE(result.exit, 0);
        QVERIFY(result.err.isEmpty());
        const auto json = result.json();
        QCOMPARE(json.value(u"name"_s).toString(), u"caelestia"_s);
        QCOMPARE(json.value(u"warnings"_s).toArray(),
            (QJsonArray{ u"Scheme \"nope\" does not exist"_s, u"\"nope\" is not a flavour of scheme caelestia"_s }));
    }

    void setsDynamicScheme() {
        writeFile(caelestia::wallpaper::statePath(), m_image.toUtf8());
        const auto result = set({ .name = u"dynamic"_s, .flavour = u"vibrant"_s, .mode = Mode::Light });
        QCOMPARE(result.exit, 0);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(result.out, u"Scheme set to dynamic (vibrant, light)\n"_s);
        verifyCurrent(u"dynamic"_s, u"vibrant"_s, Mode::Light);
    }

    void picksDynamicFlavourFromWallpaper() {
        writeFile(caelestia::wallpaper::statePath(), m_image.toUtf8());
        QCOMPARE(set({ .name = u"dynamic"_s }).out, u"Scheme set to dynamic (tonalspot, dark)\n"_s);
        verifyCurrent(u"dynamic"_s, u"tonalspot"_s, Mode::Dark);
    }

    void dynamicFallsBackToDefaultWallpaper() {
        const auto result = set({ .name = u"dynamic"_s });
        QCOMPARE(result.exit, 0);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(lib::loadStateRef()->name, u"dynamic"_s);
    }

    void dispatchesFromParser() {
        const auto result = dispatch(scheme::parserNode(), { u"set"_s, u"-n"_s, u"catppuccin"_s, u"-f"_s, u"latte"_s });
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.out, u"Scheme set to catppuccin (latte, light)\n"_s);
        QCOMPARE(dispatch(scheme::parserNode(), { u"set"_s }).exit, 2);
        QCOMPARE(dispatch(scheme::parserNode(), { u"set"_s, u"-m"_s, u"dim"_s }).exit, 2);
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestSet)

#include "set_test.moc"
