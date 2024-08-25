# 基于InternLM的中华食谱大模型🍲

<div align="center">
  <img src="logo.png" width="1024"/>
  <br /><br />
</div>
Logo由通义AI生成。

## 🚩目录

- [🚩目录](#目录)
- [🍳项目简介](#项目简介)
- [🛠️技术架构](#技术架构)
- [📺Demo](#Demo)
- [💻使用指南](#使用指南)
  - [1. 数据准备](#1-数据准备)
  - [2.  指令数据微调](#2-指令数据微调)
  - [3. 可视化](#3-可视化)
  - [4. LMDeploy部署](#4-LMDeploy部署)
- [😄更新说明](#更新说明)
- [😰遇到的问题](#遇到的问题)
- [😘鸣谢](#鸣谢)


## 🍳项目简介

本项目基于网上搜集到的食谱数据，构建一个食谱大模型。通过和用户的交互，期望它能达到以下效果：

- 根据图片生成食谱
- 饮食推荐
- 食品识别
- 利用RAG技术从网上检索最新的数据

## 🛠️技术架构

目前项目主要基于上海人工智能实验室的开源模型InternVL2-2B，4B，8B模型，使用的数据来自美食天下，下厨房等食谱网站，通过爬取它们的数据集构建用于模型训练的指令微调数据集。使用XTuner对模型进行Lora微调，后续计划结合langchain实现RAG检索增强的效果，可视化界面使用streamlit实现。项目框架图如下：

![](/framework.png)

## 📺Demo
目前是GIF动图，比较模糊，后续更新更清晰的视频。
<img src="doc/demo.gif" alt="Description" width="100%">

## 💻使用指南

### 1. 数据准备

- 从[美食天下](https://www.meishichina.com/)爬取食谱网站：```python data_url_data.py```
- 从爬取的食谱网站下载数据，包括图片，食材，步骤：```python download_data.py```
- 因为有些图片无法使用，对数据进行一次过滤：```python filter_data.py```
- 生成指令微调数据集，根据图片回答名称、食材、食谱：```python construct_instruct_data.py```
### 2. 指令数据微调
（1）首先下载对应的模型：
```python
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型
os.system('huggingface-cli download --resume-download OpenGVLab/InternVL2-2B --local-dir /root/model/InternVL2-2B')

```
（2）然后复制对应的配置文件：
```shell
xtuner copy-cfg internvl_v2_internlm2_2b_lora_finetune.py
```

（3）然后修改配置文件信息，主要修改要微调的模型以及数据集位置：
```python
path = '/root/model/InternVL2-2B'

# Data
data_root = '/root/data/MeiShiTianXia/'
data_path = data_root + 'instruction_dataset.json'
image_folder = data_root
```
（4）然后执行微调命令：
```shell
NPROC_PER_NODE=1 xtuner train 配置文件 --deepspeed deepspeed_zero1
```

（5）最后合并Lora权重：
```
python3 /root/Project/InternLM/code/XTuner/xtuner/configs/internvl/v1_5/convert_to_official.py ./internvl_v2_internlm2_4b_lora_finetune_copy.py /root/Project/ReceipeLLM/work_dirs/internvl_v2_internlm2_4b_lora_finetune_copy/iter_8500.pth /root/Project/ReceipeLLM/InternVL2-4B-Receipe/
```

使用xtuner convert merge会报错。

### 3. 可视化
使用命令```streamlit run web_demo.py```，界面如下：

<img src="doc/demo.png" alt="Description" width="100%">

### 4. LMDeploy部署
（1）启动API服务器：
```shell
lmdeploy serve api_server \
    /root/Project/ReceipeLLM/InternVL2-2B-Receipe \
    --model-format hf \
    --quant-policy 0 \
    --server-name 0.0.0.0 \
    --server-port 23333 \
    --tp 1
```
（2）然后以Gradio网页形式连接API服务器：
```shell
lmdeploy serve gradio http://localhost:23333 \
    --server-name 0.0.0.0 \
    --server-port 6006
```

也可以直接使用如下命令部署：
```shell
lmdeploy serve gradio /root/Project/ReceipeLLM/InternVL2-2B-Receipe --cache-max-entry-count 0.1
```
需要注意的是部署的4B模型没有上传图片按钮，2B模型有。

## 😄更新说明

- 8.16：使用2887条食品图像-文本数据进行微调
- 8.23：使用6794条食品图像-文本数据进行微调InternVL2-4B

如果你想加一些颜文字，可以看这个网址：[地址](https://www.emojiall.com/zh-hans/emoji/%F0%9F%91%A8%F0%9F%8F%BF%E2%80%8D%F0%9F%8D%B3)

## 😰遇到的问题
1、使用InternVL2-4B微调的模型进行推理时报错RuntimeError: shape '[-1, 0]' is invalid for input of size 77，但是InternVL2-2B不会，这确实是一个存在的问题：[Issue](https://www.modelscope.cn/models/OpenGVLab/InternVL2-4B/feedback/issueDetail/13820)

解决办法：跟换transformers版本：transformers 4.37.2，[参考链接](https://github.com/OpenGVLab/InternVL/issues/405)

## 😘鸣谢

如果你也想做大模型，可以来这里了解：[书生浦语项目](https://github.com/InternLM/Tutorial)
