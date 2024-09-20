# Savant4RedT

> 🎇 **Composition for Information Security Red Team** 🎇

![pic](docs/img/image_2.jpg)

## Introduction

![pic](docs/img/image_3.jpg)

为了解决信息内容安全检测的精度挑战和成本问题，我们的小队分别将 `LLM For Security` 和 `Security For LLM` 形成框架雏形，
提出了基于 `IPEX-LLM` 框架的功能化大模型信息安全红队测试专家组 `Savant4RedT`。我们的小队希望通过一系列数据增强、结构优化、
大模型微调、量化部署技术，获得能够有效识别 **存在信息内容安全问题** 的自然语段，并针对信息识别，结合我们小队整理的 `SOP` 情报
库，实现安全响应；我们的小队也希望以专家组的形式，缓解大模型在 `CPU` 上进行推理的工作压力，通过解绑和负载 **专注于单一目标** 
的传统 `NLP` 模型，实现结构优化和性能提升。

## Quick Start

### 下载模型权重

请将模型权重下载到 `models` 文件夹下 (创建文件夹即可)，即 `models/Savant4RedT-1_8B-Content`。

模型权重链接为 [Link](https://www.modelscope.cn/models/SaaRaaS/Savant4RedT-1_8B-Content) 。

### 安装依赖

所用 python 版本为 3.11。

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install ipex-llm==2.1.0b20240805
pip install transformers==4.37.0
pip install accelerate==0.33.0
pip install streamlit==1.38.0
pip install einops==0.8.0
pip install sentencepiece==0.2.0
pip install py-cpuinfo==9.0.0
```

### 网页体验

对于网页体验 `Demo`，可以尝试执行 `python` 文件 `start.py`

```bash
python start.py
```

对于其他细节内容，请访问 `quick_start.md` 文件，其链接为 [Link](docs/quick_start.md)

## Acknowledgement

![pic](docs/img/image_1.png)

---

### 🏆 荣誉墙

+ **🎖️ [2024-08-11] 进入 `Intel LLM Hackathon` 挑战赛 `50` 强 (50/1618) 🎖️**

+ **🥈 [2024-09-20] 荣获 `Intel LLM Hackathon` 挑战赛亚军 (二等奖) 🥈**

### 🥤 Supporting

+ **✨ 感谢 [Claire](https://space.bilibili.com/14888344?spm_id_from=333.1007.0.0) 同学提供 Logo 等美术支持 ✨**

+ **✨ 感谢 `Intel` 官方提供框架、算力等技术支持 ✨**

### Members

<a href="https://github.com/SaaRaaS-1300/Savant4RedT/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=SaaRaaS-1300/Savant4RedT" />
</a>


---

![pic](docs/img/image_4.png)
