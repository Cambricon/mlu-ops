// Copyright 2025 Cambricon Inc. All Rights Reserved.
/*
 * Originally from Meta Platforms, Inc. and affiliates.
 * Licensed under both the BSD-style license and GPLv2.
 */

#include "pzstd/SkippableFrame.h"

#include <cstdio>
#include "pzstd/Range.h"

namespace mluoptest {

namespace {
// safer C++-style cast
inline uint32_t readU32(const void *memPtr) {
  return *static_cast<const uint32_t *>(memPtr);
}
}  // namespace

/* static */ std::size_t SkippableFrame::tryRead(ByteRange bytes) {
  if (bytes.size() < SkippableFrame::kSize ||
      readU32(bytes.begin()) != kSkippableFrameMagicNumber ||
      readU32(bytes.begin() + 4) != kFrameContentsSize) {
    return 0;
  }

  return readU32(bytes.begin() + 8);
}

}  // namespace mluoptest
