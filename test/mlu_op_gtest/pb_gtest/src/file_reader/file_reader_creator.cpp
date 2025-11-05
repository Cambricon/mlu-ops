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
#include "file_reader.h"
#include "tools.h"
namespace mluoptest {

class SearchFile {
 public:
  SearchFile(const std::string &file_path) : file_path_(file_path) {}
  std::string searchFilePath();

 private:
  std::string file_path_;
};

// Try to search the file accurately, if it cannot be found,
// change the suffix to find it. If cannot find it again,
// return empty string, then an error will be reported.
std::string SearchFile::searchFilePath() {
  auto file_path = file_path_;
  auto ext = getFileExtension(file_path);
  // 1.file exists
  if (fileExists(file_path)) {
    return file_path;
  }
  // set the env false, will not try to change the suffix of data file
  bool fuzzy_search = getEnv("CNNL_GTEST_FUZZY_SEARCH_DATA_FILE", true);
  if (!fuzzy_search) {
    return std::string();
  }
  const int ext_length = 4;  // ".zst"
  // 2.file exists after remove ".zst"
  if ("zst" == ext) {
    if (file_path.size() <= ext_length) {
      return std::string();
    }
    auto bin_file = file_path.substr(0, file_path.size() - ext_length);
    if (fileExists(bin_file)) {
      file_path_ = bin_file;
      LOG(WARNING) << file_path << " does not exist, use " << bin_file
                   << " instead";
      return bin_file;
    }
  }
  // 3.file exists after add ".zst"
  auto zst_file = file_path + ".zst";
  if (fileExists(zst_file)) {
    file_path_ = zst_file;
    LOG(WARNING) << file_path << " does not exist, use " << zst_file
                 << " instead";
    return zst_file;
  }
  return std::string();
}

std::string FileReaderCreator::getRealFilePath() { return file_path_; }

void FileReaderCreator::setRealFilePath() {
  auto search_file = std::make_shared<SearchFile>(file_path_);
  file_path_ = search_file->searchFilePath();
}

std::shared_ptr<FileReader> FileReaderCreator::getFileReader() {
  // check whether file is exist.
  bool find_data_file_failed = !file_path_.empty();
  GTEST_CHECK(find_data_file_failed,
              "Please check the data file path of tensor in pb/prototxt");
  std::shared_ptr<FileReaderFactory> reader_factory = nullptr;

  if (getFileExtension(file_path_) == "zst") {
    reader_factory = std::make_shared<ZstdFactory>();
    // some binary data file not end with ".bin", such as:
    // /SOFT_TRAIN/release_test/split/split_data_split_float16_1648201505743.pb
    // so cannot use extention ".bin" to create object
  } else {
    reader_factory = std::make_shared<BinFactory>();
  }
  return reader_factory->create();
}

}  // namespace mluoptest
