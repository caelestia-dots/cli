#include <qcoreapplication.h>
#include <qdbusconnection.h>
#include <qdbusmessage.h>
#include <qfileinfo.h>
#include <qprocess.h>
#include <qurl.h>

#include <cstdio>
#include <cstdlib>

using Qt::StringLiterals::operator""_s;

namespace {

bool showInFileManager(const QString& path) {
    auto msg = QDBusMessage::createMethodCall(u"org.freedesktop.FileManager1"_s, u"/org/freedesktop/FileManager1"_s,
        u"org.freedesktop.FileManager1"_s, u"ShowItems"_s);
    msg << QStringList{ QUrl::fromLocalFile(path).toString() } << QString();
    return QDBusConnection::sessionBus().call(msg).type() == QDBusMessage::ReplyMessage;
}

} // namespace

int main(int argc, char* argv[]) {
    const QCoreApplication app(argc, argv);
    const auto args = QCoreApplication::arguments();
    if (args.size() != 2) {
        std::fprintf(stderr, "Usage: %s <path>\n", argc > 0 ? argv[0] : "caelestia-open-in-file-manager");
        return EXIT_FAILURE;
    }

    const QFileInfo file(args.at(1));
    if (showInFileManager(file.absoluteFilePath()))
        return EXIT_SUCCESS;
    return QProcess::execute(u"xdg-open"_s, { file.absolutePath() }) == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
