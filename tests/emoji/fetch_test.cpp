#include <qurl.h>

#include "emoji/emoji.hpp"
#include "common.hpp"

using namespace caelestia::cli::test;
namespace emoji = caelestia::cli::emoji;

namespace {

constexpr auto k_emojiFixture =
    "[{\"unicode\":\"😀\",\"label\":\"grinning face\",\"tags\":[\"face\",\"grin\"],\"emoticon\":\":D\"},"
    "{\"unicode\":\"😁\",\"label\":\"beaming face\",\"tags\":[\"eye\",\"grin\"],\"emoticon\":[\":-D\",\"XD\"]},"
    "{\"unicode\":\"🫠\",\"label\":\"melting face\"}]";

constexpr auto k_glyphFixture = "{\"METADATA\":{\"website\":\"https://example.com\"},"
                                "\"cod-account\":{\"char\":\"\uEB99\",\"code\":\"eb99\"},"
                                "\"md-account\":{\"char\":\"\U000F0004\",\"code\":\"f0004\"},"
                                "\"oct-person\":{\"char\":\"\uF415\",\"code\":\"f415\"},"
                                "\"seti-account\":{\"char\":\"\uF415\",\"code\":\"f415\"}}";

class TestFetch : public QObject, protected Sandbox {
    Q_OBJECT

private:
    emoji::FetchSources m_sources;

    Output fetch(bool json = false) {
        return capture([this, json] {
            return emoji::fetchFrom(m_sources, json);
        });
    }

    QString fixture(const QString& name, const char* content) {
        const auto file = path(name);
        writeFile(file, content);
        return QUrl::fromLocalFile(file).toString();
    }

private slots:

    void initTestCase() {
        setupEnvironment();
        m_sources = {
            .emojiUrl = fixture(u"emojis.json"_s, k_emojiFixture),
            .glyphUrl = fixture(u"glyphs.json"_s, k_glyphFixture),
            .outPath = path(u"out/emojis.txt"_s),
        };
        QVERIFY(QDir().mkpath(path(u"out"_s)));
    }

    void init() const { QFile::remove(m_sources.outPath); }

    void defaultsPointAtRemoteAndSourceTree() {
        const auto defaults = emoji::defaultFetchSources();
        QVERIFY(defaults.emojiUrl.startsWith(u"https://"_s));
        QVERIFY(defaults.glyphUrl.startsWith(u"https://"_s));
        QVERIFY(defaults.outPath.endsWith(u"/src/emoji/data/emojis.txt"_s));
    }

    void writesCombinedData() {
        const auto result = fetch();
        QCOMPARE(result.exit, 0);
        QVERIFY(result.out.isEmpty());
        QVERIFY(result.err.isEmpty());

        const auto lines = readFile(m_sources.outPath).split(u'\n');
        QCOMPARE(lines.first(), u"¿? question upside down reversed spanish"_s);
        QCOMPARE(lines.at(22), u"󰖳 windows super key"_s);
        QCOMPARE(lines.at(23), u"😀 :D grinning face face grin"_s);
        QCOMPARE(lines.at(24), u"😁 :-D XD beaming face eye grin"_s);
        QCOMPARE(lines.at(25), u"🫠 melting face"_s);
        QCOMPARE(lines.mid(26), (QStringList{ u"\uEB99  nf-cod-account"_s, u"\U000F0004  nf-md-account"_s,
                                    u"\uF415  nf-oct-person nf-seti-account"_s }));
    }

    void matchesBundledFormat() {
        QCOMPARE(fetch().exit, 0);
        const auto bundled = readFile(u":/emoji/emojis.txt"_s).split(u'\n');
        const auto written = readFile(m_sources.outPath).split(u'\n');
        QCOMPARE(written.mid(0, 23), bundled.mid(0, 23));
        QVERIFY(!written.last().endsWith(u'\n'));
    }

    void failsWhenSourceIsMissing() {
        m_sources.emojiUrl = QUrl::fromLocalFile(path(u"missing.json"_s)).toString();
        const auto result = fetch();
        QCOMPARE(result.exit, 1);
        QVERIFY(result.err.startsWith(u"Failed to fetch file://"_s));
        QVERIFY(!QFile::exists(m_sources.outPath));
        m_sources.emojiUrl = fixture(u"emojis.json"_s, k_emojiFixture);
    }

    void failsWhenSourceIsMissingJson() {
        m_sources.glyphUrl = QUrl::fromLocalFile(path(u"missing.json"_s)).toString();
        const auto result = fetch(true);
        QCOMPARE(result.exit, 1);
        QVERIFY(result.err.isEmpty());
        QVERIFY(result.json().value(u"error"_s).toString().startsWith(u"Failed to fetch file://"_s));
        m_sources.glyphUrl = fixture(u"glyphs.json"_s, k_glyphFixture);
    }

    void failsWhenOutputIsUnwritable() {
        const auto outPath = m_sources.outPath;
        m_sources.outPath = path(u"nodir/emojis.txt"_s);
        const auto result = fetch();
        QCOMPARE(result.exit, 1);
        QVERIFY(result.err.startsWith(u"Failed to write "_s));
        m_sources.outPath = outPath;
    }

    void dispatchesFromParser() { QCOMPARE(dispatch(emoji::parserNode(), { u"fetch"_s, u"extra"_s }).exit, 2); }
};

} // namespace

QTEST_GUILESS_MAIN(TestFetch)

#include "fetch_test.moc"
