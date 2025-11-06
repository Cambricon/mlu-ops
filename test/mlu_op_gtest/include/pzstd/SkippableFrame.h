/*************************************************************************
 * Copyright (C) [2024] by Cambricon, Inc.
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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include "pzstd/Range.h"

namespace mluoptest {
/**
 * We put a skippable frame before each frame.
 * It contains a skippable frame magic number, the size of the skippable frame,
 * and the size of the next frame.
 * Each skippable frame is exactly 12 bytes in little endian format.
 * The first 8 bytes are for compatibility with the ZSTD format.
 * If we have N threads, the output will look like
 *
 * [0x184D2A50|4|size1] [frame1 of size size1]
 * [0x184D2A50|4|size2] [frame2 of size size2]
 * ...
 * [0x184D2A50|4|sizeN] [frameN of size sizeN]
 *
 * Each sizeX is 4 bytes.
 *
 * These skippable frames should allow us to skip through the compressed file
 * and only load at most N pages.
 */
class SkippableFrame {
 public:
  static constexpr std::size_t kSize = 12;

 private:
  std::uint32_t frameSize_;
  // std::array<std::uint8_t, kSize> data_;
  std::array<char, kSize> data_;
  static constexpr std::uint32_t kSkippableFrameMagicNumber = 0x184D2A50;
  // Could be improved if the size fits in less bytes
  static constexpr std::uint32_t kFrameContentsSize = kSize - 8;

 public:
  // Write the skippable frame to data_ in LE format.
  explicit SkippableFrame(std::uint32_t size);

  // Read the skippable frame from bytes in LE format.
  static std::size_t tryRead(ByteRange bytes);
  // static std::size_t tryRead(MutableByteRange bytes);

  ByteRange data() const { return {data_.data(), data_.size()}; }

  // Size of the next frame.
  std::size_t frameSize() const { return frameSize_; }
};
}  // namespace mluoptest
