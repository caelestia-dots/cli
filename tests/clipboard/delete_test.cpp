#include "clipboard/clipboard.hpp"
#include "fakes.hpp"

using namespace caelestia::cli::test;
namespace clipboard = caelestia::cli::clipboard;

namespace {

class TestDelete : public QObject, protected FakeTools {
    Q_OBJECT

private:
    static Output remove(bool json = false) {
        return capture([=] {
            return clipboard::remove({}, { .command = u"caelestia clipboard delete"_s, .json = json });
        });
    }

private slots:

    void initTestCase() { installTools(); }

    void init() { resetTools(); }

    void deletesSelection() {
        const auto result = remove();
        QCOMPARE(result.exit, 0);
        QVERIFY(result.out.isEmpty());
        QVERIFY(result.err.isEmpty());
        QCOMPARE(log(), u"cliphist list\n"
                        "fuzzel --dmenu --prompt=del >  --placeholder=Delete from clipboard\n"
                        "cliphist delete\n"
                        "deleted:1\thello world\n"_s);
    }

    void passesThroughFuzzelExit() {
        qputenv("FUZZEL_EXIT", "2");
        QCOMPARE(remove().exit, 2);
        QVERIFY(!log().contains(u"cliphist delete"_s));
    }

    void failsOnEmptySelection() {
        qputenv("FUZZEL_OUT", "\n");
        QCOMPARE(remove().exit, 1);
        QVERIFY(!log().contains(u"cliphist delete"_s));
    }

    void failsWhenCliphistMissing() {
        QVERIFY(QFile::remove(m_bin.filePath(u"cliphist"_s)));
        const auto result = remove(true);
        QCOMPARE(result.exit, 1);
        QVERIFY(result.json().value(u"error"_s).toString().startsWith(u"Failed to run cliphist: "_s));
        installTools();
    }

    void dispatchesFromParser() {
        QCOMPARE(dispatch(clipboard::parserNode(), { u"delete"_s }).exit, 0);
        QVERIFY(log().contains(u"deleted:"_s));
        QCOMPARE(dispatch(clipboard::parserNode(), { u"delete"_s, u"--bogus"_s }).exit, 2);
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestDelete)

#include "delete_test.moc"
