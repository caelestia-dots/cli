#include <qset.h>

#include <caelestia/wallpaper/wallpaper.hpp>

#include "wallpaper/wallpaper.hpp"
#include "common.hpp"

using namespace caelestia::cli::test;
namespace wallpaper = caelestia::cli::wallpaper;
using caelestia::wallpaper::statePath;

namespace {

class TestRandom : public QObject, protected Sandbox {
    Q_OBJECT

private:
    QString m_dir;
    QSet<QString> m_images;

    static Output random(const wallpaper::RandomArgs& args, bool json = false) {
        return capture([&] {
            return wallpaper::random(args, { .command = u"caelestia wallpaper random"_s, .json = json });
        });
    }

    [[nodiscard]] static QString current() { return readFile(statePath()); }

private slots:

    void initTestCase() {
        setupEnvironment();
        m_dir = path(u"walls"_s);
        m_images = {
            saveImage(m_dir + u"/a.png"_s),
            saveImage(m_dir + u"/b.png"_s),
            saveImage(m_dir + u"/nested/deep/c.png"_s, 0xFFEEEEEE),
        };
        QVERIFY(!m_images.contains(QString()));
        writeFile(m_dir + u"/notes.txt"_s, "hello");
    }

    void init() { QFile::remove(statePath()); }

    void picksSupportedFileRecursively() {
        QSet<QString> seen;
        for (int i = 0; i < 40 && seen != m_images; ++i) {
            const auto result = random({ .dir = m_dir });
            QCOMPARE(result.exit, 0);
            QVERIFY(m_images.contains(current()));
            QCOMPARE(result.out, u"Wallpaper set to "_s + current() + u'\n');
            seen.insert(current());
        }
        QCOMPARE(seen, m_images);
    }

    void printsJson() {
        const auto result = random({ .dir = m_dir }, true);
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.json().value(u"path"_s).toString(), current());
    }

    void failsWithoutCandidates() {
        const auto empty = path(u"empty"_s);
        QVERIFY(QDir().mkpath(empty));
        writeFile(empty + u"/notes.txt"_s, "hello");

        const auto result = random({ .dir = empty });
        QCOMPARE(result.exit, 1);
        QVERIFY(result.out.isEmpty());
        QVERIFY(!result.err.isEmpty());
        QVERIFY(!QFileInfo::exists(statePath()));
    }

    void failsWithMissingDir() { QCOMPARE(random({ .dir = path(u"missing"_s) }).exit, 1); }

    void defaultsToPicturesWallpapers() {
        const auto only = saveImage(path(u"Pictures/Wallpapers/only.png"_s));
        QVERIFY(!only.isEmpty());
        QCOMPARE(random({}).exit, 0);
        QCOMPARE(current(), only);
    }

    void dispatchesFromParser() {
        const auto result = dispatch(wallpaper::parserNode(), { u"random"_s, u"--no-smart"_s, m_dir });
        QCOMPARE(result.exit, 0);
        QVERIFY(m_images.contains(current()));
        QCOMPARE(dispatch(wallpaper::parserNode(), { u"random"_s, m_dir, u"extra"_s }).exit, 2);
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestRandom)

#include "random_test.moc"
