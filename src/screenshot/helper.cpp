#include <qcoreapplication.h>
#include <qtextstream.h>

#include "screenshot/notify.hpp"

using Qt::StringLiterals::operator""_s;
namespace screenshot = caelestia::cli::screenshot;

int main(int argc, char* argv[]) {
    const QCoreApplication app(argc, argv);
    const auto args = QCoreApplication::arguments();

    if (args.size() != 3 || (args[1] != u"edit" && args[1] != u"save")) {
        QTextStream(stderr) << u"Usage: %1 <edit|save> <path>\n"_s.arg(args.value(0, u"caelestia-screenshot-helper"_s));
        return 1;
    }

    return args[1] == u"edit" ? screenshot::annotateAndNotify(args[2], false, false)
                              : screenshot::pickAndSave(args[2], false, false, false);
}
