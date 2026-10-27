#include <caelestia/wallpaper/wallpaper.hpp>

#include "wallpaper/wallpaper.hpp"
#include "common.hpp"

using namespace caelestia::cli::test;
namespace wallpaper = caelestia::cli::wallpaper;
using caelestia::wallpaper::statePath;

namespace {

class TestGet : public QObject, protected Sandbox {
    Q_OBJECT

private:
    QString m_image;

    static Output get(bool json = false) {
        return capture([=] {
            return wallpaper::get({}, { .command = u"caelestia wallpaper get"_s, .json = json });
        });
    }

private slots:

    void initTestCase() {
        setupEnvironment();
        m_image = saveImage(path(u"blue.png"_s));
        QVERIFY(!m_image.isEmpty());
    }

    void init() { QFile::remove(statePath()); }

    void failsWhenUnset() {
        const auto result = get();
        QCOMPARE(result.exit, 1);
        QVERIFY(result.out.isEmpty());
        QCOMPARE(result.err, u"Wallpaper has not been set or is invalid\n"_s);
    }

    void failsWhenUnsetJson() {
        const auto result = get(true);
        QCOMPARE(result.exit, 1);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(result.json().value(u"error"_s).toString(), u"Wallpaper has not been set or is invalid"_s);
    }

    void failsWhenStateIsNotAnImage() {
        writeFile(statePath(), path(u"gone.png"_s).toUtf8());
        QCOMPARE(get().exit, 1);
    }

    void printsCurrent() {
        writeFile(statePath(), m_image.toUtf8());
        const auto result = get();
        QCOMPARE(result.exit, 0);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(result.out, m_image + u'\n');
    }

    void printsCurrentJson() {
        writeFile(statePath(), m_image.toUtf8());
        const auto result = get(true);
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.json().value(u"path"_s).toString(), m_image);
    }

    void escapesJson() {
        const auto quoted = saveImage(path(u"say \"hi\"\\there.png"_s));
        QVERIFY(!quoted.isEmpty());
        writeFile(statePath(), quoted.toUtf8());
        const auto result = get(true);
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.json().value(u"path"_s).toString(), quoted);
    }

    void dispatchesFromParser() {
        writeFile(statePath(), m_image.toUtf8());
        const auto result = dispatch(wallpaper::parserNode(), { u"get"_s });
        QCOMPARE(result.exit, 0);
        QCOMPARE(result.out, m_image + u'\n');
        QCOMPARE(dispatch(wallpaper::parserNode(), { u"get"_s, u"extra"_s }).exit, 2);
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestGet)

#include "get_test.moc"
