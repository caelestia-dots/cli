#include <qtemporarydir.h>
#include <qtest.h>

#include <memory>

#include "common/fakebus.hpp"
#include "common/notify.hpp"

using Qt::StringLiterals::operator""_s;
namespace notify = caelestia::cli::common::notify;
using caelestia::cli::test::FakeBus;
using caelestia::cli::test::FakeNotifications;

namespace {

class TestNotify : public QObject {
    Q_OBJECT

private:
    [[nodiscard]] QString logPath() const { return m_dir.path() + u"/log"_s; }

    [[nodiscard]] QString log() const {
        QFile file(logPath());
        return file.open(QIODevice::ReadOnly) ? QString::fromUtf8(file.readAll()) : QString();
    }

private slots:

    void initTestCase() {
        QVERIFY(m_dir.isValid());
        QVERIFY(m_bus.start(m_dir.path() + u"/bus"_s));
        m_notifications = std::make_unique<FakeNotifications>(logPath());
        QVERIFY(FakeBus::serve(
            u"org.freedesktop.Notifications"_s, u"/org/freedesktop/Notifications"_s, m_notifications.get()));
    }

    void init() {
        QFile::remove(logPath());
        m_notifications->capabilities = { u"actions"_s, u"x-caelestia-actions"_s };
    }

    void sendPassesImageAsPathHint() {
        QCOMPARE(notify::send({ .summary = u"Shot"_s, .body = u"Saved"_s, .image = u"/tmp/shot.png"_s }), 42u);
        QVERIFY(log().contains(u"Notify Caelestia Shot Saved [] {image-path=/tmp/shot.png}"_s));
    }

    void sendStripsOnlyActionHintsWithoutActionSupport() {
        m_notifications->capabilities = { u"actions"_s };
        notify::send({
            .summary = u"Shot"_s,
            .body = u"Saved"_s,
            .image = u"/tmp/shot.png"_s,
            .actions = { u"open"_s, u"Open"_s },
            .hints = {
                { u"x-caelestia-action-open"_s, QStringList{ u"xdg-open"_s, u"/tmp/shot.png"_s } },
                { u"urgency"_s, QStringList{ u"2"_s } },
            },
        });
        QVERIFY(log().contains(u"Notify Caelestia Shot Saved [] {image-path=/tmp/shot.png;urgency=2}"_s));
    }

private:
    QTemporaryDir m_dir;
    FakeBus m_bus;
    std::unique_ptr<FakeNotifications> m_notifications;
};

} // namespace

QTEST_GUILESS_MAIN(TestNotify)
#include "notify_test.moc"
