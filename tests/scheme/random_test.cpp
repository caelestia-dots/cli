#include <qset.h>

#include <caelestia/scheme/scheme.hpp>

#include "scheme/scheme.hpp"
#include "common.hpp"

using namespace caelestia::cli::test;
namespace scheme = caelestia::cli::scheme;
namespace lib = caelestia::scheme;

namespace {

class TestRandom : public QObject, protected Sandbox {
    Q_OBJECT

private:
    static Output random(bool json = false) {
        return capture([=] {
            return scheme::random({}, { .command = u"caelestia scheme random"_s, .json = json });
        });
    }

    static void verifyCurrentExists() {
        const auto state = lib::loadStateRef();
        QVERIFY(state);
        const auto scheme = lib::Scheme::forName(state->name).result();
        QVERIFY(scheme);
        const auto flavour = scheme->flavour(state->flavour);
        QVERIFY(flavour);
        QVERIFY(flavour->modes().contains(state->mode));
    }

private slots:

    void initTestCase() { setupEnvironment(); }

    void init() { QFile::remove(lib::statePath()); }

    void picksExistingCombination() {
        QSet<QString> seen;
        for (int i = 0; i < 20; ++i) {
            const auto result = random();
            QCOMPARE(result.exit, 0);
            QVERIFY(result.err.isEmpty());
            verifyCurrentExists();
            const auto state = *lib::loadStateRef();
            QCOMPARE(
                result.out, u"Scheme set to %1 (%2, %3)\n"_s.arg(state.name, state.flavour, lib::modeName(state.mode)));
            seen.insert(state.name);
        }
        QVERIFY(seen.size() > 1);
    }

    void printsJson() {
        const auto result = random(true);
        QCOMPARE(result.exit, 0);
        verifyCurrentExists();
        const auto state = *lib::loadStateRef();
        const auto json = result.json();
        QCOMPARE(json.value(u"name"_s).toString(), state.name);
        QCOMPARE(json.value(u"flavour"_s).toString(), state.flavour);
        QCOMPARE(json.value(u"mode"_s).toString(), lib::modeName(state.mode));
    }

    void dispatchesFromParser() {
        QCOMPARE(dispatch(scheme::parserNode(), { u"random"_s }).exit, 0);
        verifyCurrentExists();
        QCOMPARE(dispatch(scheme::parserNode(), { u"random"_s, u"extra"_s }).exit, 2);
    }
};

} // namespace

QTEST_GUILESS_MAIN(TestRandom)

#include "random_test.moc"
