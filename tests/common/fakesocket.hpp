#pragma once

#include <qlocalserver.h>
#include <qlocalsocket.h>
#include <qsemaphore.h>
#include <qstring.h>
#include <qthread.h>

#include <functional>
#include <utility>

#include "common/hypr.hpp"

namespace caelestia::cli::test {

class FakeSocket : public QThread {
public:
    FakeSocket(QString path, std::function<void(QLocalSocket&)> handler)
        : m_path(std::move(path))
        , m_handler(std::move(handler)) {
        start();
        m_listening.acquire();
    }

protected:
    void run() override {
        QLocalServer server;
        const auto listening = server.listen(m_path);
        m_listening.release();
        if (!listening || !server.waitForNewConnection(common::hypr::k_timeoutMs))
            return;
        auto* conn = server.nextPendingConnection();
        m_handler(*conn);
        conn->disconnectFromServer();
        if (conn->state() != QLocalSocket::UnconnectedState)
            conn->waitForDisconnected(common::hypr::k_timeoutMs);
    }

private:
    QString m_path;
    std::function<void(QLocalSocket&)> m_handler;
    QSemaphore m_listening;
};

} // namespace caelestia::cli::test
