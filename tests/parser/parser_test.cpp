#include <qstringlist.h>
#include <qtest.h>

#include <optional>

#include "parser/node.hpp"

using namespace caelestia::cli;
using Qt::StringLiterals::operator""_s;

namespace {

enum class Mode : quint8 {
    Light,
    Dark,
};

enum class Target : quint8 {
    Screen,
    Region,
};

struct CaptureArgs {
    Target target = Target::Screen;
    QString geometry;
    bool freeze = false;
    bool sound = false;
};

struct SetArgs {
    std::optional<QString> name;
    std::optional<Mode> mode;
    double minSize = 0.8;
};

struct ToggleArgs {
    QString workspace;
};

struct IpcArgs {
    QStringList message;
};

struct Captured {
    CaptureArgs capture;
    SetArgs set;
    ToggleArgs toggle;
    IpcArgs ipc;
    parser::Context context;
    int runs = 0;
};

parser::NodePtr tree(Captured& out) {
    const auto remember = [&out](const auto& args, const parser::Context& context, auto member) {
        out.*member = args;
        out.context = context;
        ++out.runs;
        return 0;
    };

    return parser::group({
        .name = u"caelestia"_s,
        .help = u"root"_s,
        .children = {
            parser::leaf<CaptureArgs>({
                .name = u"screenshot"_s,
                .help = u"capture"_s,
                .options = {
                    parser::value(&CaptureArgs::geometry, u'g', u"geometry"_s, u"GEOM"_s, u"region"_s),
                    parser::flag(&CaptureArgs::freeze, {}, u"freeze"_s, u"freeze"_s),
                    parser::flag(&CaptureArgs::sound, u's', u"sound"_s, u"sound"_s),
                },
                .positionals = {
                    parser::positional(&CaptureArgs::target, u"TARGET"_s, u"target"_s,
                        { { u"screen"_s, Target::Screen }, { u"region"_s, Target::Region } }),
                },
                .run = [=](const CaptureArgs& a, const parser::Context& c) { return remember(a, c, &Captured::capture); },
            }),
            parser::group({
                .name = u"scheme"_s,
                .help = u"scheme"_s,
                .children = {
                    parser::leaf<SetArgs>({
                        .name = u"set"_s,
                        .help = u"set"_s,
                        .options = {
                            parser::value(&SetArgs::name, u'n', u"name"_s, u"NAME"_s, u"name"_s),
                            parser::choice(&SetArgs::mode, u'm', u"mode"_s, u"MODE"_s, u"mode"_s,
                                { { u"dark"_s, Mode::Dark }, { u"light"_s, Mode::Light } }),
                            parser::value(&SetArgs::minSize, {}, u"min-size"_s, u"FRACTION"_s, u"size"_s),
                        },
                        .requireOption = true,
                        .run = [=](const SetArgs& a, const parser::Context& c) { return remember(a, c, &Captured::set); },
                    }),
                    parser::leaf<parser::NoArgs>({
                        .name = u"random"_s,
                        .help = u"random"_s,
                        .run = [](const parser::NoArgs&, const parser::Context&) { return 7; },
                    }),
                },
            }),
            parser::leaf<ToggleArgs>({
                .name = u"toggle"_s,
                .help = u"toggle"_s,
                .positionals = { parser::positional(&ToggleArgs::workspace, u"WORKSPACE"_s, u"ws"_s) },
                .run = [=](const ToggleArgs& a, const parser::Context& c) { return remember(a, c, &Captured::toggle); },
            }),
            parser::leaf<IpcArgs>({
                .name = u"ipc"_s,
                .help = u"ipc"_s,
                .positionals = { parser::rest(&IpcArgs::message, u"MESSAGE"_s, u"msg"_s) },
                .run = [=](const IpcArgs& a, const parser::Context& c) { return remember(a, c, &Captured::ipc); },
            }),
        },
        .version = [] { return u"caelestia test"_s; },
    });
}

class TestParser : public QObject {
    Q_OBJECT

private:
    Captured m_out;
    parser::NodePtr m_root;

    int run(const QStringList& args) { return m_root->dispatch(args, parser::Context{ .command = u"caelestia"_s }); }

private slots:

    void init() {
        m_out = {};
        m_root = tree(m_out);
    }

    void defaults() {
        QCOMPARE(run({ u"screenshot"_s }), 0);
        QCOMPARE(m_out.runs, 1);
        QCOMPARE(m_out.capture.target, Target::Screen);
        QVERIFY(m_out.capture.geometry.isEmpty());
        QVERIFY(!m_out.capture.freeze);
        QCOMPARE(m_out.context.command, u"caelestia screenshot"_s);
        QVERIFY(!m_out.context.json);
    }

    void flagsValuesAndChoicePositional() {
        QCOMPARE(run({ u"screenshot"_s, u"-g"_s, u"1,2 3x4"_s, u"--freeze"_s, u"region"_s }), 0);
        QCOMPARE(m_out.capture.target, Target::Region);
        QCOMPARE(m_out.capture.geometry, u"1,2 3x4"_s);
        QVERIFY(m_out.capture.freeze);
    }

    void longEquals() {
        QCOMPARE(run({ u"screenshot"_s, u"--geometry=1,2 3x4"_s }), 0);
        QCOMPARE(m_out.capture.geometry, u"1,2 3x4"_s);
    }

    void shortClusterAndAttachedValue() {
        QCOMPARE(run({ u"screenshot"_s, u"-sg1,2 3x4"_s }), 0);
        QVERIFY(m_out.capture.sound);
        QCOMPARE(m_out.capture.geometry, u"1,2 3x4"_s);
    }

    void positionalChoiceRejected() {
        QCOMPARE(run({ u"screenshot"_s, u"window"_s }), 2);
        QCOMPARE(m_out.runs, 0);
    }

    void missingOptionValue() {
        QCOMPARE(run({ u"screenshot"_s, u"-g"_s }), 2);
        QCOMPARE(run({ u"screenshot"_s, u"--geometry"_s }), 2);
    }

    void flagWithExplicitValue() { QCOMPARE(run({ u"screenshot"_s, u"--freeze=yes"_s }), 2); }

    void unknownOption() {
        QCOMPARE(run({ u"screenshot"_s, u"--bogus"_s }), 2);
        QCOMPARE(run({ u"screenshot"_s, u"-x"_s }), 2);
        QCOMPARE(run({ u"--bogus"_s, u"screenshot"_s }), 2);
    }

    void tooManyPositionals() { QCOMPARE(run({ u"screenshot"_s, u"screen"_s, u"extra"_s }), 2); }

    void requiredPositional() {
        QCOMPARE(run({ u"toggle"_s }), 2);
        QCOMPARE(run({ u"toggle"_s, u"special"_s }), 0);
        QCOMPARE(m_out.toggle.workspace, u"special"_s);
    }

    void variadicWithTerminator() {
        QCOMPARE(run({ u"ipc"_s, u"--"_s, u"-x"_s, u"y"_s }), 0);
        QCOMPARE(m_out.ipc.message, (QStringList{ u"-x"_s, u"y"_s }));
        QCOMPARE(run({ u"ipc"_s }), 0);
        QVERIFY(m_out.ipc.message.isEmpty());
    }

    void optionalAndEnumOptions() {
        QCOMPARE(run({ u"scheme"_s, u"set"_s, u"-m"_s, u"dark"_s }), 0);
        QVERIFY(!m_out.set.name);
        QCOMPARE(m_out.set.mode, Mode::Dark);
        QCOMPARE(m_out.set.minSize, 0.8);
        QCOMPARE(run({ u"scheme"_s, u"set"_s, u"-m"_s, u"blue"_s }), 2);
    }

    void doubleValues() {
        QCOMPARE(run({ u"scheme"_s, u"set"_s, u"--min-size"_s, u"0.5"_s }), 0);
        QCOMPARE(m_out.set.minSize, 0.5);
        QCOMPARE(run({ u"scheme"_s, u"set"_s, u"--min-size"_s, u"big"_s }), 2);
    }

    void requiredOption() {
        QCOMPARE(run({ u"scheme"_s, u"set"_s }), 2);
        QCOMPARE(run({ u"scheme"_s, u"set"_s, u"--json"_s }), 2);
        QCOMPARE(m_out.runs, 0);
        QCOMPARE(run({ u"scheme"_s, u"set"_s, u"-n"_s, u"x"_s }), 0);
        QCOMPARE(m_out.runs, 1);
        QCOMPARE(run({ u"scheme"_s, u"set"_s, u"-h"_s }), 0);
        QCOMPARE(m_out.runs, 1);
    }

    void groupRequiresSubcommand() {
        QCOMPARE(run({}), 2);
        QCOMPARE(run({ u"scheme"_s }), 2);
        QCOMPARE(run({ u"scheme"_s, u"bogus"_s }), 2);
        QCOMPARE(m_out.runs, 0);
    }

    void helpAtEveryLevel() {
        QCOMPARE(run({ u"-h"_s }), 0);
        QCOMPARE(run({ u"scheme"_s, u"--help"_s }), 0);
        QCOMPARE(run({ u"scheme"_s, u"set"_s, u"-h"_s }), 0);
        QCOMPARE(m_out.runs, 0);
    }

    void version() {
        QCOMPARE(run({ u"-v"_s }), 0);
        QCOMPARE(run({ u"-V"_s }), 0);
        QCOMPARE(run({ u"--version"_s }), 0);
        QCOMPARE(run({ u"scheme"_s, u"-v"_s }), 2);
        QCOMPARE(run({ u"screenshot"_s, u"-v"_s }), 2);
    }

    void globalsAnywhere() {
        QCOMPARE(run({ u"--json"_s, u"scheme"_s, u"set"_s, u"-n"_s, u"x"_s }), 0);
        QVERIFY(m_out.context.json);
        QCOMPARE(run({ u"scheme"_s, u"--json"_s, u"set"_s, u"-n"_s, u"x"_s }), 0);
        QVERIFY(m_out.context.json);
        QCOMPARE(run({ u"scheme"_s, u"set"_s, u"-n"_s, u"x"_s, u"--json"_s }), 0);
        QVERIFY(m_out.context.json);
    }

    void runExitCodePropagates() { QCOMPARE(run({ u"scheme"_s, u"random"_s }), 7); }
};

} // namespace

QTEST_APPLESS_MAIN(TestParser)

#include "parser_test.moc"
