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

#include <unistd.h>      // C system headers
#include <sys/socket.h>  // C system headers
#include <arpa/inet.h>   // C system headers
#include <netinet/in.h>  // C system headers

#include <string>        // C++ system headers
#include <iostream>
#include <utility>
#include <chrono>  // NOLINT(build/c++11)

#include <nlohmann/json.hpp>  // third-party headers

#include "basic_tools.h"      // project headers

using json = nlohmann::json;

#define CNNL_GTEST_PORT 16315      // date of establishment of cambricon
#define CNNL_GTEST_JSON_SIZE 4096  // now 512 is enough

/*
json like this
{
 "label": xxx,  // eg: after_read, before_read
 "status": 1/0,   // start or end
 "pid": xxx,
 "tid": xxx,
 "time": xxx,
 "file_info":

{      "file_name": xxx,      "file_size": xxx,      "once_io_speed": xxx    }

}
*/

class JsonMsg {
 public:
  JsonMsg() = default;
  // used for server
  explicit  JsonMsg(json inputJson) : json_(inputJson) {}
  explicit JsonMsg(std::string jsonStr) : dump_str_(jsonStr) {
    std::string str = std::move(jsonStr);
    json_ = json::parse(str);
  }

  // reset the json
  void clear() { json_.clear(); }

  // used for client
  void setBasicInfo() {
    pid_ = static_cast<int64_t>(getpid());
    json_["pid"] = pid_;
    tid_ = static_cast<int64_t>(pthread_self());
    json_["tid"] = tid_;
    time_ = static_cast<int64_t>(mluoptest::getCurrentTimeT());
    json_["time"] = time_;
  }

  void setAfterReadFile(std::string fileName, size_t fileSize,
                        double onceIoSpeed) {
    json_["label"] = "after_read_file";
    json_["file_info"] =

        {{"file_name", fileName}

         ,
         {"file_size", fileSize},
         {"once_io_speed", onceIoSpeed}};
  }

  void setBeforeReadFile(std::string fileName) {
    json_["label"] = "before_read_file";
    json_["file_info"]["file_name"] = fileName;
  }

  void setStart() {
    json_["label"] = "process_status";
    json_["status"] = 1;
  }

  void setEnd() {
    json_["label"] = "process_status";
    json_["status"] = 0;
  }

  json get() { return json_; }
  std::string getJsonStr() {
    json2Str();
    return dump_str_;
  }

  size_t getSizeOfDumpStr() { return dump_str_.size(); }

 private:
  json json_;
  // for data transmission
  std::string dump_str_ = "";
  void json2Str() { dump_str_ = json_.dump(); }
  // contents of json
  int64_t pid_ = -1;
  int64_t tid_ = -1;
  int64_t time_ = -1;
  // int64_t high_time_ = -1;  // may use high precision time later
  std::string case_path_ = "";  // path of pb
  std::string file_path_ = "";  // path of data
  size_t file_size_ = -1;
};
