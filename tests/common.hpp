#pragma once

#include <qbytearray.h>
#include <qdir.h>
#include <qfile.h>
#include <qfileinfo.h>
#include <qimage.h>
#include <qjsondocument.h>
#include <qjsonobject.h>
#include <qlogging.h>
#include <qstring.h>
#include <qstringlist.h>
#include <qtemporarydir.h>
#include <qtemporaryfile.h>
#include <qtest.h>

#include <caelestia/config/config.hpp>
#include <caelestia/scheme/scheme.hpp>
#include <unistd.h>

#include <cstdio>
#include <utility>

#include "parser/node.hpp"

namespace caelestia::cli::test {

using Qt::StringLiterals::operator""_s;

constexpr auto k_configFixture = R"({"theme":{"targets":{
    "terminal":false,"hypr":false,"discord":false,"spicetify":false,"pandora":false,"fuzzel":false,
    "btop":false,"nvtop":false,"htop":false,"gtk":false,"qt":false,"warp":false,"chromium":false,
    "zed":false,"cava":false
}}})";

inline void writeFile(const QString& path, const QByteArray& data) {
    QVERIFY(QDir().mkpath(QFileInfo(path).absolutePath()));
    QFile file(path);
    QVERIFY(file.open(QIODevice::WriteOnly));
    QCOMPARE(file.write(data), data.size());
}

inline QString readFile(const QString& path) {
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly))
        return {};
    return QString::fromUtf8(file.readAll());
}

inline QString saveImage(const QString& path, QRgb colour = 0xFF2962FF) {
    QDir().mkpath(QFileInfo(path).absolutePath());
    QImage image(8, 8, QImage::Format_RGB32);
    image.fill(colour);
    return image.save(path) ? path : QString();
}

inline void seedScheme(const QString& name, const QString& flavour, caelestia::scheme::Mode mode) {
    const auto seed = caelestia::scheme::defaultScheme().result();
    QVERIFY(caelestia::scheme::saveState({ .name = name, .flavour = flavour, .mode = mode, .colours = seed.colours }));
}

struct Output {
    int exit = -1;
    QString out;
    QString err;

    [[nodiscard]] QJsonObject json() const { return QJsonDocument::fromJson(out.toUtf8()).object(); }
};

class Capture {
public:
    Capture() {
        fflush(stdout);
        fflush(stderr);
        m_savedOut = dup(STDOUT_FILENO);
        m_savedErr = dup(STDERR_FILENO);
        if (!m_out.open() || !m_err.open())
            return;
        static_cast<void>(dup2(m_out.handle(), STDOUT_FILENO));
        static_cast<void>(dup2(m_err.handle(), STDERR_FILENO));
        m_previousHandler = qInstallMessageHandler([](QtMsgType, const QMessageLogContext&, const QString&) {
        });
    }

    ~Capture() { restore(); }

    Q_DISABLE_COPY_MOVE(Capture)

    Output finish(int exit) {
        restore();
        return { .exit = exit, .out = readFile(m_out.fileName()), .err = readFile(m_err.fileName()) };
    }

private:
    void restore() {
        if (m_savedOut < 0)
            return;
        qInstallMessageHandler(m_previousHandler);
        fflush(stdout);
        fflush(stderr);
        static_cast<void>(dup2(m_savedOut, STDOUT_FILENO));
        static_cast<void>(dup2(m_savedErr, STDERR_FILENO));
        close(m_savedOut);
        close(m_savedErr);
        m_savedOut = -1;
    }

    QTemporaryFile m_out;
    QTemporaryFile m_err;
    int m_savedOut = -1;
    int m_savedErr = -1;
    QtMessageHandler m_previousHandler = nullptr;
};

template <typename F> Output capture(F&& run) {
    Capture capture;
    return capture.finish(std::forward<F>(run)());
}

inline Output dispatch(const parser::NodePtr& root, const QStringList& args, bool json = false) {
    return capture([&] {
        return root->dispatch(args, { .command = u"caelestia "_s + root->name(), .json = json });
    });
}

class Sandbox {
protected:
    QTemporaryDir m_root;

    [[nodiscard]] QString path(const QString& name) const { return m_root.filePath(name); }

    void setupEnvironment() {
        QVERIFY(m_root.isValid());
        qputenv("HOME", m_root.path().toUtf8());
        qputenv("XDG_CACHE_HOME", path(u"cache"_s).toUtf8());
        qputenv("XDG_CONFIG_HOME", path(u"config"_s).toUtf8());
        qputenv("XDG_DATA_HOME", path(u"data"_s).toUtf8());
        qputenv("XDG_STATE_HOME", path(u"state"_s).toUtf8());
        qunsetenv("HYPRLAND_INSTANCE_SIGNATURE");

        writeFile(config::Config::configPath(), k_configFixture);
        config::Config::instance().reload();
    }
};

} // namespace caelestia::cli::test
