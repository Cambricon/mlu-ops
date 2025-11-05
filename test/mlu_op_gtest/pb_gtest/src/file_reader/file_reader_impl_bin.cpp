#include "file_reader.h"
#include "thread_pool.h"
#include "tools.h"

#include <vector>
#include <fstream>
#include <iostream>
#include <future>
#include <tuple>

namespace mluoptest {

// Class for reading binary files.
// Improve the speed of reading through multi-threading.
// Each thread reads a fixed size and loops through the file according to the
// number of opened threads until the complete file is read.
// When 32 threads are enabled, the speed is incresased by about 4 times compared
// with single thread.
// Users can modify the number of threads through "CNNL_GTEST_FILE_READ_THREAD_NUM"
// environment variable
class BinFileReader : public FileReader {
  public:
    size_t read(void *data, size_t length, const std::string &filepath) final;
    size_t readPart(size_t offset, size_t read_size);
    virtual ~BinFileReader() {}
    void init(void *data, size_t length, const std::string &filepath);
  private:
    char *data_;
    size_t length_;
    std::string file_name_;
    const size_t k_once_size_ = 1 << 26;
    std::queue<std::future<size_t>> res_;
};

std::shared_ptr<FileReader> BinFactory::create() {
  return std::make_shared<BinFileReader>();
}

void BinFileReader::init(void *data, size_t length, const std::string &filepath) {
  data_ = (char *)data;
  length_ = length;
  file_name_ = filepath;
}

size_t BinFileReader::readPart(size_t offset, size_t read_size) {
  std::ifstream fin(file_name_, std::ios::in|std::ios::binary);
  if (!fin.good()) {
    throw std::runtime_error("Failed to open file: " + file_name_);
  }
  fin.seekg(offset, std::ios::beg);
  char *temp = data_ + offset;
  fin.read(temp, read_size);
  return read_size;
}

size_t BinFileReader::read(void *data, size_t length, const std::string &filepath) {
  init(data, length, filepath);
  size_t thread_num = getEnvInt("CNNL_GTEST_FILE_READ_THREAD_NUM", 32);
  ThreadPool pool(thread_num);

  size_t repeat = length / k_once_size_;
  size_t rem = length_ % k_once_size_;
  size_t offset = 0;

  for (size_t i = 0; i < repeat; ++i) {
    res_.push(pool.enqueue(std::bind(&BinFileReader::readPart, this, offset, k_once_size_)));
    offset += k_once_size_;
  }
  if (rem) {
    res_.push(pool.enqueue(std::bind(&BinFileReader::readPart, this, offset, rem)));
  }
  size_t total_size = 0;
  while (!res_.empty()) {
    total_size += res_.front().get();
    res_.pop();
  }
  return total_size;
}

}  // namespace mluoptest