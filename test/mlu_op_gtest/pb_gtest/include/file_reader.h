/*************************************************************************
 * Copyright (C) [2025] by Cambricon, Inc.
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
#include <memory>
#include <string>

namespace mluoptest {

class FileReader {
 public:
  virtual size_t read(void *data, size_t length,
                      const std::string &filepath) = 0;
  virtual ~FileReader() = default;
};

class FileReaderFactory {
 public:
  virtual std::shared_ptr<FileReader> create() = 0;
};

class BinFactory : public FileReaderFactory {
 public:
  virtual std::shared_ptr<FileReader> create() override;  // NOLINT
};

class ZstdFactory : public FileReaderFactory {
 public:
  virtual std::shared_ptr<FileReader> create() override;  // NOLINT
};

class FileReaderCreator {
 public:
  explicit FileReaderCreator(const std::string file_path)
      : file_path_(file_path) {
    setRealFilePath();
  }

  std::shared_ptr<FileReader> getFileReader();
  std::string getRealFilePath();

 private:
  std::string file_path_;
  void setRealFilePath();
};

}  // namespace mluoptest
