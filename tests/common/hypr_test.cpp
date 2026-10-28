#include <qbytearray.h>
#include <qdir.h>
#include <qjsonobject.h>
#include <qlocalsocket.h>
#include <qtemporarydir.h>
#include <qtest.h>

#include "common/fakesocket.hpp"
#include "common/hypr.hpp"

using Qt::StringLiterals::operator""_s;
namespace hypr = caelestia::cli::common::hypr;
using caelestia::cli::test::FakeSocket;

namespace {

class TestHypr : public QObject {
    Q_OBJECT

private slots:

    void init() {
        m_runtime.emplace();
        QVERIFY(m_runtime->isValid());
        qputenv("XDG_RUNTIME_DIR", m_runtime->path().toUtf8());
        qputenv("HYPRLAND_INSTANCE_SIGNATURE", "test");
        QVERIFY(QDir().mkpath(hypr::socketDir()));
    }

    void dispatchSendsRequest() {
        QByteArray received;
        FakeSocket server(hypr::socketDir() + u"/.socket.sock"_s, [&](QLocalSocket& conn) {
            conn.waitForReadyRead(hypr::k_timeoutMs);
            received = conn.readAll();
            conn.write("ok");
        });

        QVERIFY(hypr::dispatch(u"togglespecialworkspace"_s, { u"foo"_s }));
        server.wait();
        QCOMPARE(received, "dispatch togglespecialworkspace foo");
    }

    void requestJsonPrefixesAndParses() {
        QByteArray received;
        FakeSocket server(hypr::socketDir() + u"/.socket.sock"_s, [&](QLocalSocket& conn) {
            conn.waitForReadyRead(hypr::k_timeoutMs);
            received = conn.readAll();
            conn.write(R"({"configProvider":"lua"})");
        });

        const auto doc = hypr::requestJson(u"status"_s);
        server.wait();
        QCOMPARE(received, "j/status");
        QCOMPARE(doc.object().value(u"configProvider"_s).toString(), u"lua"_s);
    }

    void requestFailsWithoutInstance() {
        qunsetenv("HYPRLAND_INSTANCE_SIGNATURE");
        QVERIFY(!hypr::request(u"version"_s));
        QVERIFY(hypr::requestJson(u"version"_s).isNull());
    }

    void requestFailsWithoutSocket() { QVERIFY(!hypr::request(u"version"_s)); }

private:
    std::optional<QTemporaryDir> m_runtime;
};

} // namespace

QTEST_GUILESS_MAIN(TestHypr)

#include "hypr_test.moc"
