#include <qbytearray.h>
#include <qdir.h>
#include <qlocalsocket.h>
#include <qtemporarydir.h>
#include <qtest.h>

#include <optional>

#include "common.hpp"
#include "common/fakesocket.hpp"
#include "common/hypr.hpp"
#include "toggle/toggle.hpp"

using namespace caelestia::cli::test;
namespace hypr = caelestia::cli::common::hypr;
namespace toggle = caelestia::cli::toggle;

namespace {

constexpr auto k_expectedEval = R"(eval require("utils.functions").toggle("foo")())";

class TestToggle : public QObject {
    Q_OBJECT

private:
    static Output run(bool json = false) { return dispatch(toggle::parserNode(), { u"foo"_s }, json); }

    static QString socketPath() { return hypr::socketDir() + u"/.socket.sock"_s; }

private slots:

    void init() {
        m_runtime.emplace();
        QVERIFY(m_runtime->isValid());
        qputenv("XDG_RUNTIME_DIR", m_runtime->path().toUtf8());
        qputenv("HYPRLAND_INSTANCE_SIGNATURE", "test");
        QVERIFY(QDir().mkpath(hypr::socketDir()));
    }

    void sendsEvalForWorkspace() {
        QByteArray received;
        FakeSocket server(socketPath(), [&](QLocalSocket& conn) {
            conn.waitForReadyRead(hypr::k_timeoutMs);
            received = conn.readAll();
            conn.write("ok");
        });

        const auto result = run();
        server.wait();
        QCOMPARE(result.exit, 0);
        QVERIFY(result.out.isEmpty());
        QVERIFY(result.err.isEmpty());
        QCOMPARE(received, k_expectedEval);
    }

    void failsOnErrorReply() {
        FakeSocket server(socketPath(), [](QLocalSocket& conn) {
            conn.waitForReadyRead(hypr::k_timeoutMs);
            conn.readAll();
            conn.write("error");
        });

        const auto result = run();
        server.wait();
        QCOMPARE(result.exit, 1);
        QVERIFY(result.out.isEmpty());
        QCOMPARE(result.err, u"Failed to toggle workspace foo\n"_s);
    }

    void failsOnErrorReplyJson() {
        FakeSocket server(socketPath(), [](QLocalSocket& conn) {
            conn.waitForReadyRead(hypr::k_timeoutMs);
            conn.readAll();
            conn.write("error");
        });

        const auto result = run(true);
        server.wait();
        QCOMPARE(result.exit, 1);
        QVERIFY(result.err.isEmpty());
        QCOMPARE(result.json().value(u"error"_s).toString(), u"Failed to toggle workspace foo"_s);
    }

    void failsWithoutInstance() {
        qunsetenv("HYPRLAND_INSTANCE_SIGNATURE");
        const auto result = run();
        QCOMPARE(result.exit, 1);
        QCOMPARE(result.err, u"Failed to toggle workspace foo\n"_s);
    }

    void rejectsBadArguments() {
        QCOMPARE(dispatch(toggle::parserNode(), {}).exit, 2);
        QCOMPARE(dispatch(toggle::parserNode(), { u"foo"_s, u"bar"_s }).exit, 2);
    }

private:
    std::optional<QTemporaryDir> m_runtime;
};

} // namespace

QTEST_GUILESS_MAIN(TestToggle)

#include "toggle_test.moc"
