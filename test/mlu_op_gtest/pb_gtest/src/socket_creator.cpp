/*************************************************************************
 * Copyright (C) [2019-2024] by Cambricon, Inc.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *************************************************************************/
#include <iostream>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>

#include "socket_creator.h"
#include "variable.h"

namespace mluoptest {

int32_t SocketCreator::connectSocket() {
  client_socket_ = socket(AF_INET, SOCK_STREAM, 0);
  if (client_socket_ == -1)
{     std::cerr << "Error creating socket, this error does not affect test results." << std::endl;     return -1;   }

  struct sockaddr_in serverAddr;
  serverAddr.sin_family = AF_INET;
  serverAddr.sin_port = htons(CNNL_GTEST_PORT);
  // TODO(niewenchang): may use configurable ip, because gtest can run on different ip.
  serverAddr.sin_addr.s_addr = inet_addr("127.0.0.1");
  // connect to server
  if (connect(client_socket_, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) == -1)
{     std::cerr << "connect to server failed, this error does not affect test results." << std::endl;     return -1;   }

  return 1;
}

// TODO(niewenchang): 目前每次发送数据都会重复connect，实际上是可以复用的
// 之后考虑数据发送这个过程本身放到队列里，后台线程处理发送
int SocketCreator::sendMsg(JsonMsg json_msg) {
  if (false == enable_monitor_)
{  // not set env     
  return -1;   }

  if (-1 == connectSocket())
{  // connect failed     close(client_socket_);     
  return -1;  // program will abort if not return  
   }

  std::string send_msg(json_msg.getJsonStr());
  char buffer[CNNL_GTEST_JSON_SIZE] = {0};
  strcpy(buffer, send_msg.c_str());
  // std::cout << "buffer = " << buffer << std::endl;  // for debug, not delete
  // TODO(niewenchang): may use non-blocking method, like create a thread
  if (send(client_socket_, &buffer, CNNL_GTEST_JSON_SIZE, 0) == -1)
{     std::cerr << "Error sending data, this error does not affect test results." << std::endl;     close(client_socket_);     return -1;   }

  close(client_socket_);
  return 1;
}
void SocketCreator::getConnectEnv()
{   enable_monitor_ = getEnv("CNNL_GTEST_ENABLE_MONITOR", false); }

std::shared_ptr<mluoptest::SocketCreator> mluoptest::SocketCreator::instance_ = nullptr;
std::mutex mluoptest::SocketCreator::mutex_;

}// namespace mluoptest
