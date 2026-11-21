#include "clipboard/clipboard.hpp"
#include "fakes.hpp"

using namespace caelestia::cli::test;
namespace clipboard = caelestia::cli::clipboard;

namespace {

class TestPick : public QObject, protected FakeTools {
    Q_OBJECT

private:
    static Output pick(bool json = false) {
        return capture([=] {
            return clipboard::pick({}, { .command = u"caelestia clipboard pick"_s, .json = json });
        });
    }

private slots:

    void initTestCase() { installTools(); }

    void init() { resetTools(); }

    void copiesDecodedSelection() {
        const auto result = pick();
        QCOMPARE(result.exit, 0);
        QVERIFY(result.out.isEmpty());
        QVERIFY(result.err.isEmpty());
        QCOMPARE(log(), u"cliphist list\n"
                        "fuzzel --dmenu --placeholder=Type to search clipboard\n"
                        "cliphist decode\n"
                        "wl-copy:decoded:1\thello world\n"_s);
    }

    void passesThroughFuzzelExit() {
        qputenv("FUZZEL_EXIT", "2");
        const auto result = pick();
        QCOMPARE(result.exit, 2);
        QVERIFY(result.err.isEmpty());
        QVERIFY(!log().contains(u"cliphist decode"_s));
    }

    void failsOnEmptySelection() {
        qputenv("FUZZEL_OUT", "\n");
        QCOMPARE(pick().exit, 1);
        QVERIFY(!log().contains(u"cliphist decode"_s));
    }

    void failsWhenCliphistMissing() {
        QVERIFY(QFile::remove(m_bin.filePath(u"cliphist"_s)));
        const auto result = pick();
        QCOMPARE(result.exit, 1);
        QVERIFY(result.err.startsWith(u"Failed to run cliphist: "_s));

        const auto json = pick(true);
        QCOMPARE(json.exit, 1);
        QVERIFY(json.err.isEmpty());
        QVERIFY(json.json().value(u"error"_s).toString().startsWith(u"Failed to run cliphist: "_s));
        installTools();
    }

    void failsWhenFuzzelMissing() {
        QVERIFY(QFile::remove(m_bin.filePath(u"fuzzel"_s)));
        const auto result = pick();
        QCOMPARE(result.exit, 1);
        QVERIFY(result.err.startsWith(u"Failed to run fuzzel: "_s));
        installTools();
    }

    void dispatchesFromParser() {
        QCOMPARE(dispatch(clipboard::parserNode(), { u"pick"_s }).exit, 0);
        QVERIFY(log().contains(u"wl-copy:"_s));
        QCOMPARE(dispatch(clipboard::parserNode(), { u"pick"_s, u"extra"_s }).exit, 2);
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestPick)

#include "pick_test.moc"
