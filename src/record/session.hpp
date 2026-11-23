#pragma once

#include <qtypes.h>

namespace caelestia::cli::record {

struct Session {
    quint32 notificationId = 0;
    bool clipboard = false;
    bool quiet = false;

    static Session load();
    void save() const;
    void end() const;
};

} // namespace caelestia::cli::record
