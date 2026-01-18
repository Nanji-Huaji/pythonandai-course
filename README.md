# Python 与人工智能课程大作业 - IMDB 影评情感分析

本项目是“Python与人工智能”课程的大作业，旨在实现一个基于 **Bi-LSTM (双向长短期记忆网络)** 的 IMDB 影评情感分析系统。项目包含完整的数据处理、模型训练、结果可视化流程，并部署了一个交互式的 Web 应用。

## 📁 项目结构

```
.
├── src/                    # 源代码目录
│   ├── app.py              # Streamlit Web 应用入口
│   ├── train.py            # 模型训练脚本
│   ├── model.py            # Pytorch 模型定义 (Bi-LSTM)
│   ├── data_utils.py       # 数据处理和词表工具
│   └── visualize_results.py # 训练结果可视化
├── re/                     # 课程报告 LaTeX 源码 (BUPT 模板)
├── beam/                   # 演示幻灯片 LaTeX 源码 (Beamer)
├── requirements.txt        # Python 依赖清单
├── lstm-model.pt           # 训练好的模型权重 (运行训练脚本后生成)
├── vocab.pt                # 词表文件 (运行训练脚本后生成)
└── README.md               # 项目说明文档
```

## 🛠️ 环境依赖

虽然代码可以在 CPU 上运行，但建议使用 NVIDIA GPU 进行训练以加速收敛。

请确保已安装 Python 3.8+，然后运行以下命令安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖库包括：
*   **PyTorch**: 深度学习框架
*   **Streamlit**: Web 应用开发
*   **Datasets (HuggingFace)**: 数据集加载
*   **Scikit-learn**: 评估指标

## 🚀 快速开始

### 1. 训练模型

首先，需要下载 IMDB 数据集并训练模型。运行以下命令启动训练过程：

```bash
python src/train.py
```

该脚本会自动：
1. 下载 IMDB 数据集。
2. 构建词表并保存为 `vocab.pt`。
3. 训练 Bi-LSTM 模型。
4. 将训练好的模型保存为 `lstm-model.pt`。
5. 在测试集上输出准确率。

### 2. 运行 Web 应用

模型训练完成后，启动 Streamlit 应用进行交互式体验：

```bash
streamlit run src/app.py
```

浏览器将自动打开页面（通常是 `http://localhost:8501`）。你可以在界面中输入任意英文影评，模型将判断其情感倾向（正面/负面）并给出置信度。

### 3. 可视化 (可选)

如果需要生成训练过程的损失曲线或其它图表：

```bash
python src/visualize_results.py
```

## 🧠 模型架构

本项目使用的是 **Bi-Directional LSTM** (双向 LSTM) 模型：
*   **Embedding 层**: 将单词索引转换为密集向量。
*   **LSTM 层**: 双向结构，捕捉上下文信息。
*   **Linear 层**: 将 LSTM 输出映射到情感分类分数。
*   **Dropout**: 防止过拟合。

## 📝 课程报告与演示

*   `re/` 目录下包含了项目报告的 LaTeX 源码。
*   `beam/` 目录下包含了答辩 PPT 的 Beamer 源码。

## 👥 作者

*   田天一 (Tian Tianyi)
