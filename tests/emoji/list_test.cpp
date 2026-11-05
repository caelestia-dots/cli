#include "emoji/emoji.hpp"
#include "common.hpp"

using namespace caelestia::cli::test;
namespace emoji = caelestia::cli::emoji;

namespace {

class TestList : public QObject {
    Q_OBJECT

private slots:

    void printsBundledData() {
        const auto result = capture([] {
            return emoji::list({}, { .command = u"caelestia emoji list"_s });
        });
        QCOMPARE(result.exit, 0);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(result.out, readFile(u":/emoji/emojis.txt"_s));
        QVERIFY(result.out.startsWith(u"¿? question upside down reversed spanish\n"_s));
        QVERIFY(result.out.contains(u"\n😀 grinning face "_s));
        QVERIFY(result.out.contains(u"nf-"_s));
    }

    void dispatchesFromParser() {
        const auto result = dispatch(emoji::parserNode(), { u"list"_s });
        QCOMPARE(result.exit, 0);
        QVERIFY(!result.out.isEmpty());
        QCOMPARE(dispatch(emoji::parserNode(), { u"list"_s, u"extra"_s }).exit, 2);
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestList)

#include "list_test.moc"
