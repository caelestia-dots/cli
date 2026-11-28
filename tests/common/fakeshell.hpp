#pragma once

#include <qrect.h>
#include <qsemaphore.h>
#include <qstring.h>
#include <qthread.h>

#include <caelestia/ipc/client.hpp>
#include <caelestia/ipc/server.hpp>

#include <utility>
#include <variant>

#include "common/fakebus.hpp"

namespace caelestia::cli::test {

struct ShellReplies {
    ipc::Result<QRect> region = ipc::Error::Cancelled;
    ipc::Result<ipc::SaveResult> screenshot = ipc::Error::Cancelled;
    ipc::Result<ipc::SaveResult> annotate = ipc::Error::Cancelled;
    ipc::Result<QString> pickFile = ipc::Error::Cancelled;
};

class FakeShell : public QThread {
public:
    explicit FakeShell(ShellReplies replies, QString logPath = {})
        : m_replies(std::move(replies))
        , m_logPath(std::move(logPath)) {
        start();
        m_listening.acquire();
    }

    ~FakeShell() override {
        quit();
        wait();
    }

    Q_DISABLE_COPY_MOVE(FakeShell)

protected:
    void run() override {
        ipc::Server server;
        connect(&server, &ipc::Server::pickRegionRequested, &server, [this](ipc::PickRegionRequest* request) {
            if (const auto* region = std::get_if<QRect>(&m_replies.region))
                request->respond(*region);
            else
                request->reject(std::get<ipc::Error>(m_replies.region));
        });
        connect(&server, &ipc::Server::screenshotRequested, &server, [this](ipc::ScreenshotRequest* request) {
            const auto region = request->region();
            appendLog(m_logPath, u"screenshot %1,%2 %3x%4 annotate=%5"_s.arg(region.x())
                                     .arg(region.y())
                                     .arg(region.width())
                                     .arg(region.height())
                                     .arg(request->annotate() ? u"true" : u"false"));
            replySaved(request, m_replies.screenshot);
        });
        connect(&server, &ipc::Server::annotateRequested, &server, [this](ipc::AnnotateRequest* request) {
            appendLog(m_logPath, u"annotate %1"_s.arg(request->file()));
            replySaved(request, m_replies.annotate);
        });
        connect(&server, &ipc::Server::pickFileRequested, &server, [this](ipc::PickFileRequest* request) {
            appendLog(
                m_logPath, u"pickFile %1 %2 %3"_s.arg(request->title(), request->initialDir(), request->initialName()));
            if (const auto* file = std::get_if<QString>(&m_replies.pickFile))
                request->respond(*file);
            else
                request->reject(std::get<ipc::Error>(m_replies.pickFile));
        });
        m_listening.release();
        exec();
    }

private:
    template <typename Request> static void replySaved(Request* request, const ipc::Result<ipc::SaveResult>& reply) {
        if (const auto* error = std::get_if<ipc::Error>(&reply)) {
            request->reject(*error);
            return;
        }

        const auto& [file, copied] = std::get<ipc::SaveResult>(reply);
        if (copied)
            request->respondCopied(file);
        else
            request->respond(file);
    }

    ShellReplies m_replies;
    QString m_logPath;
    QSemaphore m_listening;
};

} // namespace caelestia::cli::test
