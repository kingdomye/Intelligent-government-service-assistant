# 基于 MindSpore 的多智能体协同辅助政务服务助手

[![DOI](https://zenodo.org/badge/1023408577.svg)](https://doi.org/10.5281/zenodo.19511336)
[![CI](https://github.com/yruichen/Intelligent-government-service-assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/yruichen/Intelligent-government-service-assistant/actions/workflows/ci.yml)

本项目面向不熟悉数字政务、存在视力或阅读障碍等情况的用户，提供业务分类、材料问答、表单辅助填写、流程生成、人脸识别和语音交互能力。

## 当前状态

仓库中的 Python 服务已经整理为可安装的 `qgai` 包。模型与训练环境体积较大，默认不随源码分发；相关能力首次调用时才加载。前端源码和 Java 服务源码当前不在本仓库中，因此旧文档中的 `Code/front` 启动步骤不再有效。

## 快速开始

要求 Python 3.10 或更高版本。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
qgai
```

默认监听本机地址：

- HTTP：`127.0.0.1:10925`
- 业务 WebSocket：`127.0.0.1:4440`
- 工具 WebSocket：`127.0.0.1:3304`

若需要模型能力，按需安装额外依赖：

```bash
pip install -e '.[qa]'
pip install -e '.[speech]'
pip install -e '.[face]'
pip install -e '.[local-llm]'
```

MindSpore、bitsandbytes 等依赖受操作系统、Python 版本和硬件平台限制，请按部署机器选择兼容版本。环境变量及模型路径见 [.env.example](.env.example)。

## 配置

配置统一通过环境变量传入，不在源码中保存密钥或机器路径。常用变量包括：

| 变量 | 用途 | 默认值 |
| --- | --- | --- |
| `HF_API_TOKEN` | Hugging Face 远程业务分类 | 无 |
| `QGAI_BASE_MODEL_PATH` | 本地基础大模型路径 | 无 |
| `QGAI_LORA_MODEL_PATH` | 本地 LoRA 权重路径 | 无 |
| `QGAI_QA_MODEL_PATH` | 问答模型权重路径 | 包内约定路径 |
| `QGAI_WHISPER_MODEL` | Whisper 模型名或路径 | `medium` |
| `QGAI_AES_KEY` | 可选传输加密密钥 | 无（明文 JSON） |

服务默认只绑定回环地址。只有在明确配置了防火墙、认证、TLS 和反向代理后，才建议绑定 `0.0.0.0`。

## 开发

本地验证不需要下载模型：

```bash
python -m compileall -q Code/qgai
python scripts/check_repository_hygiene.py
```

依赖、包入口和可选能力统一维护在 `pyproject.toml`。工程边界与后续拆分方向见 [docs/architecture.md](docs/architecture.md)。

## 仓库卫生与安全

以下内容不应提交到 Git：

- `.env`、API 密钥和加密密钥；
- 模型权重、虚拟环境和训练中间产物；
- JAR、ZIP、音视频样例等构建或演示制品；
- IDE 配置、缓存和运行时生成文件；
- 未脱敏训练语料、真实表单、个人照片、任务书和答辩材料。

二进制发布物应放在 GitHub Releases、对象存储或模型仓库中，并在发布说明中提供校验值。此前暴露过的 ModelArts 密钥必须立即吊销；详见 [SECURITY.md](SECURITY.md)。

## 许可证

本项目采用 [MIT License](LICENSE)。
