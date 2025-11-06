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
#include "socket_creator.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <iostream>

#include "variable.h"

namespace mluoptest {

int32_t SocketCreator::connectSocket() {
  client_socket_ = socket(AF_INET, SOCK_STREAM, 0);
  if (client_socket_ == -1) {
    std::cerr
        << "Error creating socket, this error does not affect test results."
        << std::endl;
    return -1;
  }

  struct sockaddr_in serverAddr;
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(CNNL_GTEST_PORT);
  // TODO(niewenchang): may use configurable ip, because gtest can run on
  // different ip.
  serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
  // connect to server
  if (connect(client_socket_, (struct sockaddr*)&serverAddr,
              sizeof(serverAddr)) == -1) {
    std::cerr
        << "connect to server failed, this error does not affect test results."
        << std::endl;
    return -1;
  }

  return 1;
}

// TODO(niewenchang): 目前每次发送数据都会重复connect，实际上是可以复用的
// 之后考虑数据发送这个过程本身放到队列里，后台线程处理发送
int SocketCreator::sendMsg(JsonMsg json_msg) {
  if (false == enable_monitor_) {  // not set env
    return -1;
  }

  if (-1 == connectSocket()) {  // connect failed     close(client_socket_);
    return -1;                  // program will abort if not return
  }

  std::string send_msg(json_msg.getJsonStr());
  char buffer[CNNL_GTEST_JSON_SIZE] = {0};
  strcpy(buffer, send_msg.c_str());//NOLINT
  // std::cout << "buffer = " << buffer << std::endl;  // for debug, not delete
  // TODO(niewenchang): may use non-blocking method, like create a thread
  if (send(client_socket_, &buffer, CNNL_GTEST_JSON_SIZE, 0) == -1) {
    std::cerr << "Error sending data, this error does not affect test results."
              << std::endl;
    close(client_socket_);
    return -1;
  }

  close(client_socket_);
  return 1;
}
void SocketCreator::getConnectEnv() {
  enable_monitor_ = getEnv("CNNL_GTEST_ENABLE_MONITOR", false);
}

std::shared_ptr<mluoptest::SocketCreator> mluoptest::SocketCreator::instance_ =
    nullptr;
std::mutex mluoptest::SocketCreator::mutex_;

}  // namespace mluoptest
