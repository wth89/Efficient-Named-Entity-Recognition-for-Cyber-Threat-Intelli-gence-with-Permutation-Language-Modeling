# Efficient Named Entity Recognition for Cyber Threat Intelligence

基于 XLNet-CRF 架构的网络威胁情报命名实体识别系统，支持三个专业领域的实体识别任务。

## 🔍 项目简介

本项目实现了一个高效的命名实体识别（NER）系统，专门用于网络威胁情报（Cyber Threat Intelligence）分析。  
系统采用 **XLNet + CRF** 架构，适配以下三个专业领域：

- **CTI-Reports **

- 识别哈希值、IP 地址、恶意软件、URL 等网络威胁实体  

  > 📄 `xlnet_crf_cti.py`

- **DNRTI  **
  识别黑客组织、攻击手段、工具、时间、地点等威胁情报元素  

  > 📄`xlnet_crf_dnrti.py`

- **MalwareTextDB**  
  识别恶意软件相关的实体、行为和修饰语  

  > 📄 `xlnet_crf_malware.py`

---

## 🧠 模型架构

- **XLNet-CRF 混合架构**：结合 XLNet 的双向建模能力与 CRF 的标签依赖优势  

- **统一训练框架**：三种任务使用相同架构，标签体系不同  

- **支持长序列处理**：最大输入长度为 256 tokens  

  > 📄 `xlnet_crf_cti.py:17`

---

## ⚙️ 训练配置

- **Batch Size**：32  

  > 📄 `xlnet_crf_cti.py:18`

- **学习率设置**：

  - XLNet 部分：`5e-5`
  - CRF 层：`8e-5`

- **训练轮数**：

  - CTI 和 Malware：80轮
  - DNRTI：90轮  

  > 📄 `xlnet_crf_cti.py:20`

---



## 📁 项目结构

```text
├── xlnet_crf_cti.py           # CTI 训练脚本
├── xlnet_crf_dnrti.py         # DNRTI 训练脚本
├── xlnet_crf_malware.py       # Malware 训练脚本
├── predict.py                 # 预测与推理脚本
├── print_metric_cti.py        # CTI 评估图表生成
├── print_metric_dnrti.py      # DNRTI 评估图表生成
├── print_metric_malware.py    # Malware 评估图表生成
├── datasets/                  # 数据目录
│   ├── CTI-reports/
│   ├── DNRTI/
│   └── MalwareTextDB/
└── outputs/
    ├── 训练日志文件
    ├── 模型检查点
    └── 性能图表

```
---

## 📦 安装依赖

```bash
pip install torch transformers numpy pandas matplotlib seaborn
```

## 🚀 使用方法

### 1️⃣ 模型训练

训练 CTI 模型：

```
python xlnet_crf_cti.py
```

训练 DNRTI 模型：

```
python xlnet_crf_dnrti.py
```

训练 Malware 模型：

```
python xlnet_crf_malware.py
```

### 2️⃣ 模型预测

```
python predict.py
```

> 📄 预测逻辑见：`predict.py:79-150`

### 3️⃣ 性能评估与可视化

```
python print_metric_cti.py       # CTI 性能图表  
python print_metric_dnrti.py     # DNRTI 性能图表  
python print_metric_malware.py   # Malware 性能图表
```

## 📚 数据格式

系统使用 **BIO 标注格式**。每个数据集包含以下文件（TSV 格式）：

```
train.txt   # 训练集
valid.txt   # 验证集
test.txt    # 测试集
```

> 📄 数据读取代码参考：`xlnet_crf_cti.py:29-43`

------

## 📤 输出文件

训练完成后将生成以下文件：

- `.pt`：模型检查点
- `.txt`：训练日志
- `.png`：性能图表

> 📄 输出逻辑见：`xlnet_crf_cti.py:22`

------

## 📈 评估指标

系统提供以下 NER 常用评估指标：

- **Accuracy**（准确率）
- **Precision**（精确率）
- **Recall**（召回率）
- **F1-Score**（F1 分数）

> 📄 评估代码：`xlnet_crf_cti.py:642-668`
