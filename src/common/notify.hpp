#pragma once

#include <qstring.h>
#include <qstringlist.h>
#include <qvariant.h>

namespace caelestia::cli::common::notify {

enum class Urgency : uchar {
    Low,
    Normal,
    Critical,
};

struct Notification {
    QString summary;
    QString body;
    QString image;
    QStringList actions;
    QVariantMap hints;
    Urgency urgency = Urgency::Normal;
};

quint32 send(Notification notification);
void close(quint32 id);

} // namespace caelestia::cli::common::notify
