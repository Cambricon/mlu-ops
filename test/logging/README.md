# 日志引擎功能回归（CPU-only，无需 MLU 卡）

三个独立的断言式回归程序，覆盖日志引擎关键路径（fmt 格式化、spdlog sink、
env 读取、文件 IO、退出期生命周期）。修改 log_core 或升级 third_party 依赖后
应跑一遍。全部 exit=0 即通过。

| 程序 | 验证内容 |
|---|---|
| `smoke_log` | 综合冒烟：屏幕打印带 head；callback 收到 body（无 head）+ 原始 severity；默认不落盘；setLogFile 后启用落盘 |
| `verify_logfile` | 落盘语义 14 断言：新文件/同名/默认文件 trunc；改名切换；disable 释放 fd（/proc/self/fd，仅 Linux）；re-enable；空串返回 BAD_PARAM |
| `verify_exit_fmt` | 退出期安全：全局对象在 ~LogEngine 之后析构时打 LOG_*_FMT 被 alive flag 拦截，程序 exit=0 且无输出 |

## 构建（在 open_mlu_ops 根目录下，需 neuware 环境）

```bash
cd open_mlu_ops
NEUWARE=${NEUWARE_HOME:-/usr/local/neuware}
for t in smoke_log verify_logfile verify_exit_fmt; do
  g++ -std=c++17 -O2 -DFMT_HEADER_ONLY \
    -I . \
    -I third_party/fmt/include \
    -I third_party/spdlog/include \
    -I $NEUWARE/include \
    test/logging/$t.cpp \
    core/log_core/log_engine.cpp \
    core/logging_api.cpp \
    core/tool.cpp \
    -L $NEUWARE/lib64 -lcnrt -Wl,-rpath,$NEUWARE/lib64 -lpthread \
    -o /tmp/$t
done
```

## 运行

```bash
cd /tmp   # verify_logfile 会在 cwd 生成 mlu_op_auto_log，建议临时目录
/tmp/smoke_log        # 期望结尾 ALL PASS，exit=0
/tmp/verify_logfile   # 期望 14 条 PASS + ALL PASS，exit=0
/tmp/verify_exit_fmt  # 期望无任何输出，exit=0
```

## 备注

- `verify_logfile` 的 fd 检查依赖 Linux `/proc/self/fd`；本目录程序均为开发期
  回归工具，不接入仓库 CMake 主构建。
- 无 neuware 的纯 CPU 环境：可自备 cnrt 头文件桩（`cnrtGetDevice` 返回 -1）
  替换 `$NEUWARE/include` 与 `-lcnrt` 即可编译运行（断言不涉及卡号）。
