/projs/platform/liuyubo/pr11.6/mlu-ops/test/mlu_op_gtest/CMakeLists.txt#!/bin/bash

# 确保我们在正确的分支
git fetch origin

# 获取与 master 分支不同的文件
CHANGED_FILES=$(git diff --name-only origin/master | grep -E '\.(cpp|cc|c|h|hpp)$')

# 如果没有文件改动，则退出
if [ -z "$CHANGED_FILES" ]; then
  echo "No C++ files changed relative to master."
  exit 0
fi

# 对每个文件运行 cpplint
for file in $CHANGED_FILES; do
  echo "Running cpplint on $file"
  cpplint "$file"
done
