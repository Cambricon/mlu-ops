/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under both the BSD-style license (found in the
 * LICENSE file in the root directory of this source tree) and the GPLv2 (found
 * in the COPYING file in the root directory of this source tree).
 */
#include "pzstd/SkippableFrame.h"

#include <cstdio>

#include "pzstd/Range.h"

using namespace mluoptest;

static inline uint32_t readU32(const void *memPtr) {
  return *(const uint32_t *)memPtr;
}

/* static */ std::size_t SkippableFrame::tryRead(ByteRange bytes) {
  if (bytes.size() < SkippableFrame::kSize ||
      readU32(bytes.begin()) != kSkippableFrameMagicNumber ||
      readU32(bytes.begin() + 4) != kFrameContentsSize) {
    return 0;
  }

  return readU32(bytes.begin() + 8);
}
