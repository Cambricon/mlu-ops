# Contribution

非常欢迎您贡献文档和代码，我们鼓励开发者以各种方式参与文档和代码的反馈与贡献。包括但不限于：

- 修改拼写错误和代码错误
- 添加新算子
- 添加文档或将文档翻译成其它语言

在参与贡献前，请先阅读遵守以下准则。

## 如何添加新算子

1. 调研算子功能，撰写算子设计文档，参考[BANGC-OPS算子设计文档模板](docs/bangc-docs/BANGC-OPS-Operator-Design-Doc-Template.md)或[BANGPy-OPS算子设计文档模板](docs/bangpy-docs/BANGPy-OPS-Operator-Design-Doc-Template.md)，主要包括：
    - 算子需求分析
    - 算子接口设计
    - 算子实现方案设计

2. 算子代码开发，参考[BANGC-OPS算子开发流程](docs/bangc-docs/BANGC-OPS-Operator-Development-Process.md)或[BANGPy-OPS算子开发流程](docs/bangpy-docs/BANGPy-OPS-Operator-Development-Process.md)、[PULL REQUEST流程](./docs/pr.md)，主要包括：
    - 算子设计文档提交 `PR`（Pull Requset），其中 BANGC 算子设计文档目录为`docs/bangc-docs/design_docs` ，BANGPy 算子设计文档目录为`docs/bangpy-docs/design_docs`
    - GTest 代码开发
    - 算子伪代码开发
    - 算子主体代码开发

3. 完成测试并撰写测试报告，参考[MLU-OPS性能验收标准](docs/MLU-OPS-Performance-Acceptance-Standard.md)、[MLU-OPS测试报告模板](docs/MLU-OPS-Test-Report-Template.md)、[MLU-OPS-Accuracy-Acceptance-Standard](docs/MLU-OPS-Accuracy-Acceptance-Standard.md)，主要包括：
    - 测例规模
    - 测例数据类型
    - 性能测试 
    - 稳定性测试
    - 内存泄露测试
    
4. 算子代码、算子测试报告一起提交 `PR`

## 代码风格

### Python 和 BANGPy 

- 遵循 [PEP8](https://www.python.org/dev/peps/pep-0008/)

- 采用 [Pylint](https://pypi.org/project/pylint/) 检查代码格式

- 安装 Pylint

    ```shell
    pip install pylint  # install
    ```

- 手动检查代码格式

    ```bash
    python3 -m pylint ./bangpy-ops --rcfile=./bangpy-ops/utils/pylintrc
    ```

### C++ 和 BANGC

- 遵循 [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

- 采用 [Cpplint](https://pypi.org/project/cpplint/) 检查代码格式（ pre commit 自动触发格式检查）

    ```shell
    pip install cpplint  # install
    ```

## 其它

1.  `PR` 合入需要至少两个点赞

2. 开发周期建议：算子文档设计１周，算子代码开发和测试报告１～２周，根据修改意见完善设计方案和代码１～２周