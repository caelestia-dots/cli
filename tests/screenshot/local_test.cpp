#include "fakes.hpp"

using namespace caelestia::cli::test;

namespace {

class TestLocal : public QObject, protected ScreenshotFakes {
    Q_OBJECT

private slots:

    void initTestCase() { setupDesktopEnvironment(); }

    void init() { resetFakes(); }

    void rejectsInvalidGeometry_data() {
        QTest::addColumn<QString>("geometry");
        QTest::newRow("malformed") << u"nonsense"_s;
        QTest::newRow("empty") << u"10,10 0x0"_s;
    }

    void rejectsInvalidGeometry() {
        QFETCH(const QString, geometry);

        const auto result = run({ u"region"_s, u"-g"_s, geometry });
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.err, u"Invalid region geometry: %1\n"_s.arg(geometry));
        QVERIFY(log().isEmpty());
    }

    void reportsCaptureFailure_data() {
        QTest::addColumn<QStringList>("args");
        QTest::addColumn<QString>("message");
        QTest::newRow("screen") << QStringList{ u"screen"_s } << u"Failed to capture screen"_s;
        QTest::newRow("region") << QStringList{ u"region"_s, u"-g"_s, u"0,0 10x10"_s } << u"Failed to capture region"_s;
    }

    void reportsCaptureFailure() {
        QFETCH(const QStringList, args);
        QFETCH(const QString, message);

        const auto result = run(args);
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.err, message + u'\n');
        QCOMPARE(log(), failure(u"Failed to take screenshot"_s, message));
    }

    void quietCaptureFailureSendsNoNotification() {
        const auto result = run({ u"screen"_s, u"-q"_s }, true);
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.json().value(u"error"_s).toString(), u"Failed to capture screen"_s);
        QVERIFY(log().isEmpty());
    }
};

} // namespace

int main(int argc, char* argv[]) {
    return runOffscreen<TestLocal>(argc, argv);
}

#include "local_test.moc"
