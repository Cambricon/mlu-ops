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

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <zstd.h>


#include <fstream>
#include <iostream>
#include <tuple>
#include <vector>
#include <chrono>//NOLINT
#include "ResourcePool.h"
#include "basic_tools.h"
#include "file_reader.h"
#include "pzstd/Buffer.h"
#include "pzstd/SkippableFrame.h"
#include "runtime.h"
#include "thread_pool.h"
#include "tools.h"

#define _LARGE_FILES 1
#if 0
static std::vector<char> readBinaryFile(const std::string& filename) {
  // Open the file in binary mode
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }

  // Read the file contents into a vector<char>
  std::vector<char> buffer(std::istreambuf_iterator<char>(file), {});

  return buffer;
}
#endif

static std::tuple<void *, off_t> readBinaryFileMmap(
    const std::string &filename) {
  int fd = open(filename.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd == -1) {
    perror("Error opening input file");
    throw std::runtime_error("Failed to open file: " + filename);
  }

  // Get the size of the input file
  off_t file_size = lseek(fd, 0, SEEK_END);
  lseek(fd, 0, SEEK_SET);

  // Map the input file into memory
  void *input_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if (input_data == MAP_FAILED) {
    perror("Error mapping input file into memory");
    close(fd);
    throw std::runtime_error("Failed to mmap file");
  }
  close(fd);
  // XXX use struct maybe more readable then tuple
  return std::make_tuple(input_data, file_size);
}

namespace mluoptest {

class ZstdStrategy {
 public:
  virtual size_t read(void *data, size_t length,
                      const std::string &filepath) = 0;
  ~ZstdStrategy() = default;
};

// Class for reading files compressed by zstd or pzstd.
class ZstdFileReader : public FileReader {
 public:
  virtual size_t read(void *data, size_t length, const std::string &filepath) final;//NOLINT
  virtual ~ZstdFileReader() {}

 private:
  std::shared_ptr<ZstdStrategy> selectZstdStrategy(void *data, size_t length,
                                                   const std::string &filepath);
};

class SingleThreadZstdStrategy : public ZstdStrategy {
 public:
  virtual size_t read(void *data, size_t length,const std::string &filepath) final;//NOLINT
};

// Class for reading files which compressed by pzstd.
// Each thread reads and decompresses a fixed size of data until the entire file
// is traversed Users can modify the number of threads through
// "CNNL_GTEST_FILE_READ_THREAD_NUM" environment variable
class ParallelZstdStrategy : public ZstdStrategy {
 public:
  explicit ParallelZstdStrategy() = default;//NOLINT
  // use same pointer to data with class executor
  ParallelZstdStrategy(const ParallelZstdStrategy &) = delete;
  ParallelZstdStrategy &operator=(const ParallelZstdStrategy &) = delete;

  virtual size_t read(void *data, size_t length,const std::string &filepath) final;//NOLINT

 private:
  std::atomic<size_t> actual_tensor_size_{0};
  std::queue<std::future<void>> res_;
  std::string file_name_;
  size_t thread_num_ = 1;
  size_t length_ = 1;
  std::unique_ptr<ResourcePool<ZSTD_DStream>> d_stream_pool_;
  char *data_;
  const size_t kstep = 1 << 23;  // kstep is const value of compress

  void init(void *data, size_t length, const std::string &file_name);
  void initDStream();
  void asyncReadAndDecompress(
      size_t input_file_offset,  // for input.zst
      size_t frame_size,         // .zst once read
      char *output,              // *data ptr of cnnl_gtest malloc before
      FILE *fd);
};

std::shared_ptr<FileReader> ZstdFactory::create() {
  return std::make_shared<ZstdFileReader>();
}

static size_t readHeader(size_t *offset, FILE *fd) {
  fseek(fd, (long long)(*offset), SEEK_SET);//NOLINT
  Buffer header_buffer(SkippableFrame::kSize);  // const 12
  auto bytesRead =
      std::fread(header_buffer.data(), 1, header_buffer.size(), fd);
  if (bytesRead != SkippableFrame::kSize) {
    throw std::runtime_error("Failed to read .zst file header.");
  }
  auto frame_size = SkippableFrame::tryRead(header_buffer.range());
  // std::cout << "frameSize = " << frame_size << std::endl;  // for debug
  *offset += (frame_size + SkippableFrame::kSize);
  return frame_size;
}

size_t SingleThreadZstdStrategy::read(void *data, size_t length,
                                      const std::string &filepath) {
  // auto compressed_data = readBinaryFile(filepath);
  // size_t result = ZSTD_decompress(data, length, compressed_data.data(),
  // compressed_data.size());
  auto compressed_data = readBinaryFileMmap(filepath);
  size_t result = ZSTD_decompress(data, length, std::get<0>(compressed_data),
                                  std::get<1>(compressed_data));
  munmap(std::get<0>(compressed_data), std::get<1>(compressed_data));
  if (ZSTD_isError(result)) {
    throw std::runtime_error("Zstd decompression failed: " +
                             std::string(ZSTD_getErrorName(result)));
  }
  if (result != length) {
    std::cerr << "zstd decompress got " << result << std::endl;
    throw std::runtime_error("Zstd decompression unexpected size " +
                             std::to_string(result) +
                             " != " + std::to_string(length));
  }

  return result;
}

static size_t advance(ZSTD_inBuffer &in_buffer) {
  size_t pos = in_buffer.pos;
  in_buffer.src = static_cast<const unsigned char *>(in_buffer.src) + pos;
  in_buffer.size -= pos;
  in_buffer.pos = 0;
  return pos;
}

static size_t split(ZSTD_outBuffer &out_buffer) {
  size_t pos = out_buffer.pos;
  out_buffer.dst = static_cast<unsigned char *>(out_buffer.dst) + pos;
  out_buffer.size -= pos;
  out_buffer.pos = 0;
  return pos;
}

size_t ParallelZstdStrategy::read(void *data, size_t length,
                                  const std::string &filepath) {
  init(data, length, filepath);
  FILE *fd = std::fopen(file_name_.c_str(), "rb");
  size_t offset = 0;  // used for determine whether the file ends
  size_t file_size = getFileSize(filepath);
  ThreadPool pool(thread_num_);
  char *current_data = (char *)data;
  while (true) {
    // data_offset is after header
    size_t data_offset = offset + SkippableFrame::kSize;
    size_t frame_size = readHeader(&offset, fd);
    res_.push(pool.enqueue(
        std::bind(&ParallelZstdStrategy::asyncReadAndDecompress, this,
                  data_offset, frame_size, current_data, fd)));
    // output_offset(*data) for each thread
    current_data += kstep;  // kstep is const value of compress, now is 1 << 23

    if (offset == file_size) {
      break;
    }
  }
  // sync all threads
  while (!res_.empty()) {
    res_.front().get();
    res_.pop();
  }
  if (fd) {
    fclose(fd);
    fd = nullptr;
  }
  d_stream_pool_.reset();  // I do not know why, it is unique_ptr, delete it
                           // will memory leak -.-
  return actual_tensor_size_;
}

void ParallelZstdStrategy::init(void *data, size_t length,
                                const std::string &file_name) {
  thread_num_ = getEnvInt("CNNL_GTEST_FILE_READ_THREAD_NUM", 32);
  file_name_ = file_name;
  length_ = length;
  data_ = (char *)data;
  initDStream();
}

void ParallelZstdStrategy::initDStream() {
  d_stream_pool_.reset(new ResourcePool<ZSTD_DStream>{
      [this]() -> ZSTD_DStream * {
        auto zds = ZSTD_createDStream();
        if (zds) {
          auto err = ZSTD_initDStream(zds);
          if (ZSTD_isError(err)) {
            ZSTD_freeDStream(zds);
            return nullptr;
          }
        }
        return zds;
      },
      [](ZSTD_DStream *zds) { ZSTD_freeDStream(zds); }});
}

void ParallelZstdStrategy::asyncReadAndDecompress(
    size_t input_file_offset,  // for input.zst
    size_t frame_size,         // .zst once read
    char *output,              // *data ptr of cnnl_gtest malloc before
    FILE *fd) {
  // thread_num now malloc is fast enough cannot use runtime.allocate, it is not
  // thread-safe
  (void)fd;
  std::shared_ptr<char> input_data(new char[frame_size],
                                   [](char *p) { delete[] p; });
  FILE *file_part = std::fopen(file_name_.c_str(), "rb");
  if (!file_part) {
    throw std::runtime_error("Failed to open file: " + file_name_);
  }
  fseek(file_part, input_file_offset, SEEK_SET);  // read current frame
  auto bytesRead = std::fread(input_data.get(), 1, frame_size, file_part);
  fclose(file_part);
  if (bytesRead != frame_size) {
    throw std::runtime_error("Failed to open file: " + file_name_);
  }

  // 统计解压后的文件大小
  actual_tensor_size_ += ZSTD_getFrameContentSize(input_data.get(), frame_size);

  // for stream decompress, once decompress chunk_size
  const size_t chunk_size = ZSTD_DStreamInSize();  // const int = 131072
  const size_t out_size = ZSTD_DStreamOutSize();   // const int = 131072
  auto ctx = d_stream_pool_->get();

  char *current_input = input_data.get();
  size_t rem_size = frame_size;
  char *current_output = output;  // addr for output for each thread
  while (rem_size > 0) {
    size_t current_chunk_size = std::min(chunk_size, rem_size);
    rem_size -= current_chunk_size;
    ZSTD_inBuffer in_buffer{current_input, current_chunk_size, 0};
    size_t rem_chunk_size =
        current_chunk_size;  // split input to several chunks
    while (rem_chunk_size > 0) {
      if (length_ <= static_cast<size_t>(current_output - data_)) {
        throw std::runtime_error(
            "when decompressing, tensor size in \".zst\" is larger than tensor "
            "size in pb,"
            " please check whether the data file is valid!");
      }
      size_t out_chunk_size =
          std::min((length_ - (current_output - data_)), out_size);
      ZSTD_outBuffer out_buffer{current_output, out_chunk_size, 0};
      auto zstd_ret = ZSTD_decompressStream(ctx.get(), &out_buffer, &in_buffer);
      if (ZSTD_isError(zstd_ret)) {
        throw std::runtime_error("Zstd decompression failed, reason is: " +
                                 std::string(ZSTD_getErrorName(zstd_ret)));
      }
      size_t output_pos = split(out_buffer);
      size_t input_pos = advance(in_buffer);
      rem_chunk_size -= input_pos;
      current_output += output_pos;
    }
    current_input += current_chunk_size;
  }
}

size_t ZstdFileReader::read(void *data, size_t length,
                            const std::string &filepath) {
  auto zstd_strategy = selectZstdStrategy(data, length, filepath);
  return zstd_strategy->read(data, length, filepath);
}

std::shared_ptr<ZstdStrategy> ZstdFileReader::selectZstdStrategy(
    void *data, size_t length, const std::string &filepath) {
  (void)data;
  (void)length;
  if (getFileSize(filepath) < SkippableFrame::kSize) {  // little file
    return std::make_shared<SingleThreadZstdStrategy>();
  }
  FILE *fd = std::fopen(filepath.c_str(), "rb");
  size_t offset = 0;
  if (0 == readHeader(&offset, fd)) {  // not have pzstd file header
    fclose(fd);
    return std::make_shared<SingleThreadZstdStrategy>();
  }
  fclose(fd);
  return std::make_shared<ParallelZstdStrategy>();
}

}  // namespace mluoptest

