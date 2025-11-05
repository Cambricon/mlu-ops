#pragma once
#include <memory>
#include <string>

namespace mluoptest {

class FileReader {
 public:
  virtual size_t read(void *data, size_t length,
                      const std::string &filepath) = 0;
  virtual ~FileReader() = default;
}

;

class FileReaderFactory {
 public:
  virtual std::shared_ptr<FileReader> create() = 0;
}

;

class BinFactory : public FileReaderFactory {
 public:
  virtual std::shared_ptr<FileReader> create() override;
}

;

class ZstdFactory : public FileReaderFactory {
 public:
  virtual std::shared_ptr<FileReader> create() override;
}

;

class FileReaderCreator {
 public:
  FileReaderCreator(const std::string file_path) : file_path_(file_path) {
    setRealFilePath();
  }

  std::shared_ptr<FileReader> getFileReader();
  std::string getRealFilePath();

 private:
  std::string file_path_;
  void setRealFilePath();
};

}  // namespace mluoptest