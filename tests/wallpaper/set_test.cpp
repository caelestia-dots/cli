#include <caelestia/scheme/scheme.hpp>
#include <caelestia/wallpaper/wallpaper.hpp>

#include "wallpaper/wallpaper.hpp"
#include "common.hpp"

using namespace caelestia::cli::test;
namespace wallpaper = caelestia::cli::wallpaper;
using caelestia::wallpaper::statePath;
namespace scheme = caelestia::scheme;

namespace {

class TestSet : public QObject, protected Sandbox {
    Q_OBJECT

private:
    QString m_image;

    static Output set(const wallpaper::SetArgs& args, bool json = false) {
        return capture([&] {
            return wallpaper::set(args, { .command = u"caelestia wallpaper set"_s, .json = json });
        });
    }

    [[nodiscard]] static scheme::SchemeData loadedScheme() {
        const auto state = scheme::loadState();
        if (!state)
            throw std::runtime_error("no scheme state");
        return *state;
    }

    static void seedDynamicScheme() {
        const auto seed = scheme::defaultScheme().result();
        QVERIFY(scheme::saveState(
            { .name = u"dynamic"_s, .flavour = u"vibrant"_s, .mode = scheme::Mode::Light, .colours = seed.colours }));
    }

private slots:

    void initTestCase() {
        setupEnvironment();
        m_image = saveImage(path(u"blue.png"_s));
        QVERIFY(!m_image.isEmpty());
    }

    void init() {
        QFile::remove(statePath());
        QFile::remove(scheme::statePath());
    }

    void setsWallpaper() {
        const auto result = set({ .path = m_image });
        QCOMPARE(result.exit, 0);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(result.out, u"Wallpaper set to "_s + m_image + u'\n');
        QCOMPARE(readFile(statePath()), m_image);
    }

    void setsWallpaperJson() {
        const auto result = set({ .path = m_image }, true);
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.json().value(u"path"_s).toString(), m_image);
    }

    void resolvesRelativePath() {
        const auto previous = QDir::currentPath();
        QVERIFY(QDir::setCurrent(m_root.path()));
        const auto result = set({ .path = u"blue.png"_s });
        QVERIFY(QDir::setCurrent(previous));

        QCOMPARE(result.exit, 0);
        QCOMPARE(result.out, u"Wallpaper set to "_s + m_image + u'\n');
        QCOMPARE(readFile(statePath()), m_image);
    }

    void rejectsUnsupportedFile() {
        const auto text = path(u"notes.txt"_s);
        writeFile(text, "hello");

        const auto result = set({ .path = text });
        QCOMPARE(result.exit, 1);
        QVERIFY(result.out.isEmpty());
        QVERIFY(result.err.contains(u"not a supported image or video"_s));
        QVERIFY(!QFileInfo::exists(statePath()));
    }

    void rejectsUnsupportedFileJson() {
        const auto result = set({ .path = path(u"missing.png"_s) }, true);
        QCOMPARE(result.exit, 1);
        QVERIFY(result.err.isEmpty());
        QVERIFY(result.json().value(u"error"_s).toString().contains(u"not a supported image or video"_s));
    }

    void smartPicksDynamicVariant() {
        seedDynamicScheme();
        QCOMPARE(set({ .path = m_image }).exit, 0);
        const auto data = loadedScheme();
        QCOMPARE(data.flavour, u"tonalspot"_s);
        QCOMPARE(data.mode, scheme::Mode::Dark);
    }

    void noSmartKeepsDynamicVariant() {
        seedDynamicScheme();
        QCOMPARE(set({ .path = m_image, .noSmart = true }).exit, 0);
        const auto data = loadedScheme();
        QCOMPARE(data.flavour, u"vibrant"_s);
        QCOMPARE(data.mode, scheme::Mode::Light);
    }

    void dispatchesFromParser() {
        seedDynamicScheme();
        const auto result = dispatch(wallpaper::parserNode(), { u"set"_s, u"--no-smart"_s, m_image });
        QCOMPARE(result.exit, 0);
        QCOMPARE(readFile(statePath()), m_image);
        QCOMPARE(loadedScheme().flavour, u"vibrant"_s);
        QCOMPARE(dispatch(wallpaper::parserNode(), { u"set"_s }).exit, 2);
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestSet)

#include "set_test.moc"
