/*************************************************************************
 * Copyright (C) [2022] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#pragma once
#include <unistd.h>

#include <memory>
#include <mutex>//NOLINT
#include <string>

#include "socket_comm.h"

namespace mluoptest {

class SocketCreator {
 public:
  static std::shared_ptr<SocketCreator> getInstance() {
    if (nullptr == instance_) {
      std::lock_guard<std::mutex> lock(mutex_);
      if (nullptr == instance_) {
        instance_ = std::shared_ptr<SocketCreator>(new SocketCreator());
      }
    }
    return instance_;
  }

  int getClientSocket() { return client_socket_; }
  SocketCreator(const SocketCreator &) = delete;
  SocketCreator &operator=(const SocketCreator &) = delete;
  ~SocketCreator() {}

  // TODO(niewenchang): use function pointer, but i do not know how to do
  void sendEndMsg() {
    JsonMsg json_msg;
    json_msg.setEnd();
    json_msg.setBasicInfo();
    sendMsg(json_msg);
  }

  void sendStartMsg() {
    JsonMsg json_msg;
    json_msg.setStart();
    json_msg.setBasicInfo();
    sendMsg(json_msg);
  }

  void sendBeforeReadFileMsg(std::string fileName) {
    JsonMsg json_msg;
    json_msg.setBasicInfo();
    json_msg.setBeforeReadFile(fileName);
    sendMsg(json_msg);
  }

  void sendAfterReadFileMsg(std::string fileName, size_t fileSize,
                            double onceIoSpeed) {
    JsonMsg json_msg;
    json_msg.setBasicInfo();
    json_msg.setAfterReadFile(fileName, fileSize, onceIoSpeed);
    sendMsg(json_msg);
  }

 private:
  SocketCreator() { getConnectEnv(); }

  void getConnectEnv();
  int32_t connectSocket();
  int sendMsg(JsonMsg json_msg);

  bool enable_monitor_ = false;
  int client_socket_ = -1;
  static std::shared_ptr<SocketCreator> instance_;
  static std::mutex mutex_;
};
}  // namespace mluoptest
