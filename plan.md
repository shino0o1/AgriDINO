# 项目架构

AgriDINO_repo/
├── data/
│   ├── __init__.py
│   ├── dataset.py        # 数据集加载逻辑（读取 Agricap 的图文和 Metadata）
│   └── transforms.py     # 图像预处理（Stage1的Multi-crop、Stage2的常规Resize）
├── models/
│   ├── __init__.py
│   ├── vision.py         # 视觉塔：加载 DINOv3 主干 + MAP (Multihead Attention Pooling)
│   ├── text.py           # 文本塔：加载 RoBERTa/SciBERT + Extended Positional Encoding
│   ├── 
│   └── agridino.py       # 顶层拼装：把 vision 和 text 连起来，定义双分支投影头
├── utils/
│   ├── __init__.py
│   ├── losses.py         # 核心难点：Taxonomy-aware Soft-target Contrastive Loss
│   └── metrics.py        # 评测指标：Recall@K (图文检索) 和 Accuracy (Zero-shot)
├── scripts/
│   ├── train_stage1.py   # DINOv3 自监督微调脚本（Teacher-Student EMA）
│   ├── train_stage2.py   # 图文对齐训练脚本（冻结视觉，训练文本+MAP+投影头）
│   └── evaluate.py       # 模型评测脚本
├── config.yaml           # 把超参数(学习率、温度参数、路径)全放这里
└── requirements.txt      # 依赖包列表

# Roadmap

## 阶段 1：基础数据管道搭建 (Data Pipeline)
*   **任务**：实现 Agricap 及其子数据集的 DataLoader。
*   **关键点**：
    *   **Stage 1 数据**：只需要图像。需要实现 DINO 的 `Multi-crop Augmentation`（全局截取 [0.4, 1.0] 和局部截取[0.05, 0.4]）。
    *   **Stage 2 数据**：需要产出 `(Image, Short Text, Long Text, Crop_ID, Disease_ID)` 的五元组。

Agricap数据示例：
  {
    "crop_class": "cabbage",
    "pest_disease_class": "Alternaria Leaf Spot",
    "short_description": "Cabbage leaf exhibiting numerous brown, concentric lesions and extensive necrotic areas.",
    "long_description": "The cabbage leaf ... impacted portions.",
    "f_path": "/share/home/u11059/apps/wzh/dinov3/final_data/images/cabbage_disease/cabbage disease _1 (2).jpg",
    "Chinese_description": "",
    "pest_disease_class_1": "",
    "pest_disease_class_2": ""
  },
  {
    "crop_class": "cabbage",
    "pest_disease_class": "Alternaria Leaf Spot",
    "short_description": "...",
    "long_description": "The cabbage leaf exhibits widespread symptoms of disease...",
    "f_path": "/share/home/u11059/apps/wzh/dinov3/final_data/images/cabbage_disease/cabbage disease _1 (3).jpg",
    "Chinese_description": "",
    "pest_disease_class_1": "",
    "pest_disease_class_2": ""
  },

## 阶段 2：Stage 1 - DINOv3 农业领域自监督微调
*   **任务**：加载预训练的 DINOv3（ViT-L/16），实现 Teacher-Student 架构和 EMA（指数移动平均）更新。
*   **捷径**：不要自己从头写 DINO 损失！可以借鉴 DINOv3 开源库（DINOv3开源库已下载到本项目目录下的 dinov3 文件夹中）。
*   **工程细节**：只用 Agricap 的 240k 图像，跑 30 个 epoch。由于是自监督，Teacher 网络的权重是由 Student 网络平滑更新（EMA）来的。

## 阶段 3：Stage 2 - 核心双分支网络搭建 (模型核心)
*   **任务**：冻结 DINOv3，搭建文本编码器、MAP 池化层和投影头。
*   **模块拆解**：
    1.  **全局视觉特征**：直接提取 DINO CLS token 并做 L2 归一化。
    2.  **细粒度视觉特征 (MAP)**：写一个 `Multihead Attention Pooling` 模块：“用 PyTorch 写一个 MAP 层，输入是 DINO 视觉 Patch tokens [B, N, D]。它需要一个可学习的 Query token，对 Patch tokens 做 Cross-Attention，聚合得到一个维度为 [B, D] 的向量，最后做 L2 归一化。”
    3.  **双分支文本编码器**：使用 Hugging Face `AutoModel` 加载文本模型，分别过两个 Linear Projection 头，得到全局和细粒度文本特征。

## 阶段 4：Taxonomy-aware Soft-target Contrastive Loss（难度最高）
*   **任务**：实现论文中引入层级结构先验的对比损失（公式 9, 10, 11, 12）。
*   **做法**：将论文中的这几个公式直接截图喂给 Claude 3.5 Sonnet 或 GPT-4o，让它翻译成 PyTorch 代码。
*   **参数对照**：论文中设定了 $\epsilon=0.1$（主正样本权重 0.9），$\alpha=0.5$（跨作物同病害），$\beta=0.25$（同作物不同病害）。

## 阶段 5：验证与评估管道
*   **任务**：实现 Zero-shot（全局分支做图文点乘）和跨模态检索（Recall@1, 5, 10）。


# 注意事项

平台兼容性: 在接下来的代码生成中，请注意：该项目主要在 Windows 系统下进行 Debug 和开发，但最终会部署到 Linux Ubuntu 集群进行多卡训练。请保证所有代码平台无关（Platform-agnostic）：严格使用 pathlib 处理路径；处理好 Windows 下 DataLoader 的 num_workers 兼容性；并为后续 Linux 多卡预留好 nccl 后端的 DDP 训练架构。

