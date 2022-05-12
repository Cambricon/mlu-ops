# Contribution

非常欢迎您贡献文档和代码，我们鼓励开发者以各种方式参与文档和代码的反馈与贡献。包括但不限于

- 修改拼写错误和代码错误
- 添加文档或将文档翻译成其它语言
- 添加新算子

在参与贡献前，请先阅读遵守以下准则。



## 如何添加新算子

1. 算子文档设计，参考[BANGC-OPS算子设计文档模板](docs/bangc-docs/BANGC-OPS算子设计文档模板.md)或[BANGPy-OPS算子设计文档模板](docs/bangpy-docs/BANGPy-OPS算子设计文档模板.md)。
2. 算子代码开发，参考[BANGC-OPS算子开发流程](docs/bangc-docs/BANGC-OPS算子开发流程.md)或[BANGPy-OPS算子开发流程](docs/bangpy-docs/BANGPy-OPS算子开发流程.md)。
3. 算子测试报告，参考[MLU-OPS性能验收标准](docs/MLU-OPS性能验收标准.md)和[MLU-OPS测试报告模板](docs/MLU-OPS测试报告模板.md)和[MLU-OPS精度验收标准](docs/MLU-OPS精度验收标准.md)。
4. 提交`PR`（Pull Requset），参考[PULL REQUEST流程](./pr.md)，其中算子文档独立提交`PR`，算子代码和算子测试报告一起提交`PR`。



## 代码风格

### Python 和 BANGPy 

- 遵循[PEP8](https://www.python.org/dev/peps/pep-0008/)

- 采用[Pylint](https://pypi.org/project/pylint/)检查代码格式

- 安装pylint

```shell
pip install pylint
```

- 手动检查代码格式

```
python3 -m pylint ./bangpy-ops  --rcfile=./bangpy-ops/utils/pylintrc
```

### C++ 和 BANGC

- 遵循[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html)

- 采用[Cpplint](https://pypi.org/project/cpplint/)检查代码格式（pre commit自动触发格式检查）

 ```shell
 #安装cpplint
 pip install cpplint
 ```



##　其它

1. `PR`合入需要至少两个点赞。

2. 开发周期建议：算子文档设计１周，算子代码开发和测试报告１～２周，根据修改意见完善设计方案和代码１～２周。

   
