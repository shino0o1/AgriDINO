dinov3/
├── .github/                      # GitHub 配置目录（CI、Issue/PR 模板等）。用于自动化检查、规范贡献流程，对训练/推理代码无直接影响。
├── notebooks/                    # 示例 Notebook（快速上手、可视化/推理演示）。适合在交互环境里验证模型效果与接口用法。
├── dinov3/                       # 核心 Python 包：训练、自监督方法、模型定义、评测与部署入口都在这里
│   ├── __init__.py               # 包入口与版本/导出符号的集中管理（便于 `import dinov3` 后暴露关键 API）。
│   │
│   ├── data/                     # 数据管线：数据集封装、采样策略、增强与 batch 组装
│   │   ├── __init__.py           # data 子模块导出与注册入口（通常用于聚合常用构件）。
│   │   ├── adapters.py           # 数据“适配层”：把不同来源/结构的数据样本转换成训练/评测所需的统一字段格式（如 image、target、meta 等）。
│   │   ├── augmentations.py      # 数据增强策略集合：面向自监督训练的强增强（多裁剪、颜色扰动、模糊等）与相关组合逻辑。
│   │   ├── collate.py            # batch 拼装（collate_fn）：处理变长字段/多视角输入、mask 信息等，确保 DataLoader 输出张量结构满足训练前向。
│   │   ├── loaders.py            # DataLoader 构建逻辑：根据配置创建 dataset、sampler、collate_fn、worker 参数等，产出可直接训练的迭代器。
│   │   ├── masking.py            # mask 相关工具：为 iBOT/patch-level 任务生成 patch mask、控制 mask 比例与形状等。
│   │   ├── meta_loaders.py       # “元”加载器：用于组合/切换多个数据源或多任务数据流（例如不同数据集混合、分阶段加载等）。
│   │   ├── samplers.py           # 采样器：分布式采样、重复采样、类别均衡或特殊 batch 组织策略，保证多卡训练中各 rank 数据划分一致。
│   │   ├── transforms.py         # 图像预处理与基础变换：resize/crop/normalize 等；与 augmentations.py 相比更偏“基础处理/可复用组件”。
│   │   └── datasets/             # 具体数据集实现（训练或下游评测会复用）
│   │       ├── __init__.py       # 数据集注册/导出。
│   │       ├── ade20k.py         # ADE20K 语义分割相关数据集封装（路径组织、标注读取、样本字段定义等）。
│   │       ├── coco_captions.py  # COCO Captions 文本-图像数据集封装（用于文本相关评测/训练管线）。
│   │       ├── decoders.py       # 解码器：把磁盘格式（JPEG/PNG、可能的序列化样本）解码成 PIL/ndarray/tensor 的统一接口。
│   │       ├── extended.py       # 扩展/包装型数据集：对基础数据集做额外字段、过滤、映射、拼接等增强能力。
│   │       ├── image_net.py      # ImageNet-1k 数据集实现（分类评测或线性探测等常用）。
│   │       ├── image_net_22k.py  # ImageNet-22k / 大规模类别数据集实现（常用于大规模预训练或扩展评测）。
│   │       └── nyu.py            # NYU Depth（或相关深度数据集）封装：深度评测/训练所需的数据读取与样本结构。
│   │
│   ├── models/                   # Backbone 模型定义：ViT/ConvNeXt 等
│   │   ├── __init__.py           # 模型工厂/注册：集中提供创建模型的入口（按名称/配置构建）。
│   │   ├── convnext.py           # ConvNeXt 结构实现与相关变体配置（用于蒸馏/对比实验或作为视觉主干）。
│   │   └── vision_transformer.py # Vision Transformer 实现：patch embed、位置编码、block 堆叠、输出特征组织等（DINO 系列核心主干）。
│   │
│   ├── layers/                   # 模型“积木层”：Attention/Block/Norm/Head/位置编码等可复用组件
│   │   ├── __init__.py           # layers 子模块聚合导出。
│   │   ├── attention.py          # 注意力实现：多头注意力、可能包含高效实现/数值稳定性处理与 mask 支持等。
│   │   ├── block.py              # Transformer Block：Attention + FFN + Norm + 残差/DropPath 等组装逻辑。
│   │   ├── dino_head.py          # DINO Head：投影头/分类 token 的映射头（自监督损失常用的 MLP head）。
│   │   ├── ffn_layers.py         # FFN/MLP 层：线性层堆叠、激活、dropout 等（与 block.py 配合使用）。
│   │   ├── fp8_linear.py         # FP8 线性层/相关支持：面向加速与显存优化的低精度算子封装（依赖硬件/后端支持）。
│   │   ├── layer_scale.py        # LayerScale：对残差分支做可学习缩放，改善深层网络训练稳定性。
│   │   ├── patch_embed.py        # Patch Embedding：把图像切 patch 并映射到 token 维度（ViT 的输入层）。
│   │   ├── rms_norm.py           # RMSNorm：替代 LayerNorm 的归一化实现（常见于大模型训练）。
│   │   ├── rope_position_encoding.py # RoPE 旋转位置编码：为注意力注入相对位置信息（更适合长序列/高分辨率 token）。
│   │   └── sparse_linear.py      # 稀疏线性层：用于稀疏计算/结构化稀疏的线性映射封装（优化速度/参数效率）。
│   │
│   ├── loss/                     # 自监督/辅助损失实现
│   │   ├── __init__.py           # loss 聚合导出。
│   │   ├── dino_clstoken_loss.py # DINO 类 token 损失：student/teacher 输出对齐、温度、中心化等逻辑（典型 DINO 框架核心）。
│   │   ├── gram_loss.py          # Gram 相关损失：通常用于特征统计/相关性匹配（比如风格/全局结构约束的变体）。
│   │   ├── ibot_patch_loss.py    # iBOT patch-level 损失：对 masked patch token 的预测与对齐（masking.py 生成 mask）。
│   │   └── koleo_loss.py         # KoLeo 类正则/均匀性损失：常用于鼓励表示分布更均匀、提升特征空间性质。
│   │
│   ├── train/                    # 训练主流程与训练期组件（优化器组、lr 调度、元架构等）
│   │   ├── __init__.py           # 训练相关入口聚合。
│   │   ├── cosine_lr_scheduler.py# Cosine 学习率调度器：warmup + cosine decay 等常见策略实现。
│   │   ├── multidist_meta_arch.py# 多重蒸馏/多教师（multi-distillation）元架构：组织多个 teacher/student 的前向与损失聚合。
│   │   ├── param_groups.py       # 参数分组：为不同模块设置不同 lr/weight decay（如 bias、norm、head、backbone 分组）。
│   │   ├── ssl_meta_arch.py      # 自监督训练元架构：封装 student/teacher、EMA 更新、multi-crop 前向、损失组合与日志指标等“训练核心逻辑”。
│   │   └── train.py              # 训练入口脚本：读取配置、初始化分布式与日志、构建数据与模型、启动训练循环与 checkpoint。
│   │
│   ├── eval/                     # 下游评测与基准：线性探测、KNN、检测/分割/深度/文本等任务
│   │   ├── __init__.py           # eval 模块入口。
│   │   ├── accumulators.py       # 指标累积器：跨 step/rank 累积预测与统计量（便于统一计算指标与做同步）。
│   │   ├── data.py               # 评测数据构建：为 eval 任务创建 dataset/dataloader/预处理，屏蔽不同任务数据差异。
│   │   ├── helpers.py            # 评测辅助函数：模型封装、特征提取、分布式 gather、结果格式化等通用能力。
│   │   ├── knn.py                # KNN 评测：提取特征后做 KNN 分类评估（快速衡量表征质量）。
│   │   ├── linear.py             # 线性探测：冻结 backbone，训练线性分类头（ImageNet 等标准评测方式）。
│   │   ├── log_regression.py     # Logistic Regression 评测：与 linear probe 类似，但优化/正则化形式不同。
│   │   ├── results.py            # 结果记录与汇总：把评测结果落盘/打印/表格化，便于对比不同实验。
│   │   ├── setup.py              # 评测环境/参数初始化：解析参数、分布式设置、随机种子、日志目录等。
│   │   ├── utils.py              # eval 通用工具：如多进程同步、张量处理、评测阶段公用的小工具函数。
│   │   ├── metrics/              # 通用指标实现
│   │   │   ├── __init__.py       # 指标导出。
│   │   │   ├── classification.py # 分类指标：top-1/top-5、混淆矩阵相关、分布式聚合等。
│   │   │   └── imagenet_c.py     # ImageNet-C 鲁棒性评测指标/读取逻辑（不同 corruption 的聚合统计等）。
│   │   ├── depth/                # 深度估计评测/训练（下游任务）
│   │   │   ├── __init__.py
│   │   │   ├── checkpoint_utils.py   # 深度任务 checkpoint 读写/兼容工具（可能处理 backbone 对接）。
│   │   │   ├── config.py             # 深度任务配置定义/默认值。
│   │   │   ├── configs/              # 深度任务的具体实验配置（不同 backbone、分辨率、训练超参）。
│   │   │   ├── data.py               # 深度任务数据管线（读取 depth label、mask、resize 策略等）。
│   │   │   ├── datasets/             # 深度数据集实现（NYU、KITTI 等可能在这里扩展）。
│   │   │   ├── eval.py               # 深度任务评测入口（推理 + 指标计算）。
│   │   │   ├── loss.py               # 深度任务损失（如 L1/Scale-invariant 等）。
│   │   │   ├── metrics.py            # 深度指标（AbsRel、RMSE、δ1/δ2/δ3 等常见指标）。
│   │   │   ├── models/               # 深度头/解码器结构（在 backbone 特征上接任务头）。
│   │   │   ├── run.py                # 深度任务运行脚本（train/eval 的统一入口或调度）。
│   │   │   ├── schedulers.py         # 深度任务学习率/训练日程调度。
│   │   │   ├── train.py              # 深度任务训练循环。
│   │   │   ├── transforms.py         # 深度任务专用变换（对齐 depth label、裁剪策略等）。
│   │   │   ├── utils.py              # 深度任务工具函数（可视化、后处理、分布式同步等）。
│   │   │   └── visualization_utils.py# 深度预测可视化（深度图着色、对比 GT/Pred 等）。
│   │   ├── detection/            # 目标检测评测/适配（下游任务）
│   │   │   ├── __init__.py
│   │   │   ├── config.py         # 检测任务配置（模型、数据、训练/评测超参）。
│   │   │   ├── models/           # 检测模型/头部（如基于特征金字塔、DETR 风格等对接实现）。
│   │   │   └── util/             # 检测任务工具（box ops、matcher、distributed utils 等常见辅助）。
│   │   ├── segmentation/         # 语义分割评测/训练（下游任务）
│   │   │   ├── __init__.py
│   │   │   ├── config.py         # 分割任务配置（数据集、backbone、decoder、训练日程等）。
│   │   │   ├── configs/          # 分割任务实验配置集合（不同数据集/模型规模的 yaml/py 配置）。
│   │   │   ├── eval.py           # 分割评测（mIoU 等）。
│   │   │   ├── inference.py      # 分割推理脚本（可能支持单图/批量、可视化输出等）。
│   │   │   ├── loss.py           # 分割损失（CE、Dice、aux loss 等）。
│   │   │   ├── metrics.py        # 分割指标实现（IoU、pixel acc、类别统计等）。
│   │   │   ├── models/           # 分割头/decoder（如 linear、UPerNet 类结构的对接等）。
│   │   │   ├── run.py            # 分割任务运行入口（train/eval 调度）。
│   │   │   ├── schedulers.py     # 分割训练调度策略。
│   │   │   ├── train.py          # 分割训练循环。
│   │   │   └── transforms.py     # 分割数据变换（同时处理 image 与 segmentation mask 的几何/颜色变换）。
│   │   └── text/                 # 文本相关评测/训练（DINOtxt 等方向）
│   │       ├── __init__.py
│   │       ├── ac_comp_parallelize.py # 文本/多模态训练的并行化与编译/激活检查点相关加速封装。
│   │       ├── build_dinotxt.py       # 构建 DINOtxt 组件（组装 vision tower + text tower + 对齐头等）。
│   │       ├── clip_loss.py           # CLIP 风格对比损失（图文对齐）。
│   │       ├── configs/               # 文本/多模态实验配置（数据、tokenizer、模型宽深等）。
│   │       ├── dinotxt_model.py       # DINOtxt 模型定义：多模态整体前向与输出结构。
│   │       ├── gram_loss.py           # 文本/多模态场景下的 Gram 类损失变体（用于额外对齐/约束）。
│   │       ├── text_tower.py          # 文本塔封装：把 tokenizer 输出映射到 text embedding。
│   │       ├── text_transformer.py    # 文本 Transformer 主干实现（attention/block 等文本侧组件）。
│   │       ├── tokenizer.py           # tokenizer 逻辑：文本预处理、编码、padding/截断等。
│   │       ├── train_dinotxt.py       # DINOtxt 训练入口：数据、分布式、优化、评测与保存。
│   │       └── vision_tower.py        # 视觉塔封装：复用 dinov3 backbone 作为图像编码器并输出对齐特征。
│   │
│   ├── distributed/               # 分布式训练抽象（对 torch.distributed 的薄封装）
│   │   ├── __init__.py            # 分布式工具导出。
│   │   ├── torch_distributed_primitives.py # 常用原语：broadcast/all_reduce/all_gather 等的统一接口与便捷封装。
│   │   └── torch_distributed_wrapper.py    # 更高层封装：初始化进程组、rank/world_size 管理、异常处理与同步策略等。
│   │
│   ├── fsdp/                      # FSDP/编译/并行训练相关加速
│   │   ├── __init__.py
│   │   └── ac_compile_parallelize.py # activation checkpoint + compile + 并行化策略的组合实现，用于大模型训练降显存/提吞吐。
│   │
│   ├── checkpointer/              # checkpoint 管理：保存/加载/恢复训练
│   │   ├── __init__.py
│   │   └── checkpointer.py        # 统一 checkpoint 读写：模型权重、优化器、调度器、训练状态（epoch/iter）等；可能支持多格式与兼容逻辑。
│   │
│   ├── logging/                   # 日志与实验记录
│   │   ├── __init__.py            # 日志系统入口：logger 初始化、格式、不同后端（stdout/文件/第三方平台）适配等。
│   │   └── helpers.py             # 日志辅助：指标聚合、分布式 rank 过滤打印、计时器、吞吐统计等。
│   │
│   ├── configs/                   # 配置系统：默认配置与训练配置集合
│   │   ├── __init__.py
│   │   ├── config.py              # 配置解析/校验/合并：把 yaml/命令行参数映射为训练可用的结构体/字典，并做默认值填充。
│   │   ├── ssl_default_config.yaml# 自监督训练默认配置：数据增强、模型规模、优化器、teacher-student 参数等的基线。
│   │   └── train/                 # 具体训练方案配置（可直接用于启动实验）
│   │       ├── __init__.py
│   │       ├── dinov3_vit7b16_pretrain.yaml         # ViT-7B/16 的预训练配置（大规模自监督训练主配置）。
│   │       ├── dinov3_vit7b16_gram_anchor.yaml      # 加入 gram/anchor 相关约束或额外损失的变体配置。
│   │       ├── dinov3_vit7b16_high_res_adapt.yaml   # 高分辨率适配/继续训练配置（更细粒度 token 或更大输入）。
│   │       ├── dinov3_vitl16_lvd1689m_distilled.yaml# 蒸馏配置（基于特定大规模数据集/教师模型设置的 distillation 方案）。
│   │       ├── vitl_im1k_lin834.yaml                # ImageNet-1k 线性评测/探测相关配置（线性头训练超参）。
│   │       ├── multi_distillation_test.yaml         # 多重蒸馏测试配置（小规模 sanity check）。
│   │       ├── distillation_convnext/               # ConvNeXt 蒸馏相关的一组配置（教师/学生结构与训练策略）。
│   │       └── multidist_tests/                     # multi-distillation 的更多测试配置集合（覆盖不同组合/超参）。
│   │
│   ├── run/                       # 运行与任务提交（更偏“工程化入口”）
│   │   ├── __init__.py
│   │   ├── init.py                # 运行初始化：环境变量、分布式/随机种子、日志目录等的统一初始化入口。
│   │   └── submit.py              # 作业提交脚本：面向集群/调度系统的提交封装（生成命令、资源规格、多机多卡参数等）。
│   │
│   ├── hub/                       # “模型动物园/对外 API”：提供 torch.hub/下游任务封装的友好加载接口
│   │   ├── __init__.py
│   │   ├── backbones.py           # backbone 构建与加载：按名称返回预训练模型、处理权重下载/加载、输出特征接口规范化。
│   │   ├── classifiers.py         # 分类器封装：在 backbone 上接分类头或提供线性探测形式的快速调用。
│   │   ├── depthers.py            # 深度估计封装：backbone + depth head 的一键构建/加载与推理接口。
│   │   ├── detectors.py           # 检测模型封装：提供下游检测任务的加载与推理入口（隐藏内部实现细节）。
│   │   ├── segmentors.py          # 分割模型封装：提供分割任务的一键构建/权重加载/推理。
│   │   ├── dinotxt.py             # 图文/多模态（DINOtxt）相关的 hub 接口：一键加载与特征提取/对齐。
│   │   └── utils.py               # hub 小工具：如权重路径解析、下载缓存、输入规范化等轻量辅助。
│   │
│   ├── env/                       # 环境/运行时设置
│   │   └── __init__.py            # 环境检测与兼容：例如 CUDA/torch 版本特性开关、性能相关默认设置等（用于减少平台差异）。
│   │
│   ├── thirdparty/                # 第三方代码内置/镜像（用于减少外部依赖或保持复现实验一致性）
│   │   ├── __init__.py
│   │   └── CLIP/                  # CLIP 相关第三方实现（用于图文对齐或文本侧评测/训练依赖）
│   │       ├── __init__.py
│   │       └── clip/              # CLIP 核心实现目录（具体文件未在当前结果中完整展开；通常包含模型定义、tokenizer、权重加载等）。
│   │
├── hubconf.py                     # torch.hub 入口：定义 `torch.hub.load(...)` 可调用的模型/任务构建函数，面向用户友好加载。
├── DATASETS.md                    # 数据集准备说明：告诉你训练/评测需要的数据格式、下载与目录组织方式。
├── MODEL_CARD.md                  # 模型卡：模型用途、训练数据/限制、评测结果与使用注意事项。
├── README.md                      # 总览文档：项目介绍、安装、快速使用、训练/评测入口说明（架构理解首选）。
├── pyproject.toml                 # Python 项目元数据与工具配置（打包、格式化、lint 等）。
├── setup.py                       # 安装脚本（editable install、打包发布等）。
├── conda.yaml                     # Conda 环境定义：依赖与版本锁定，便于复现。
├── requirements.txt               # 运行依赖（训练/推理所需的 Python 包）。
├── requirements-dev.txt           # 开发依赖（测试、lint、格式化、文档生成等）。
├── LICENSE.md                     # 许可证声明。
├── CODE_OF_CONDUCT.md             # 行为准则。
├── CONTRIBUTING.md                # 贡献指南（提交 PR、代码风格、测试要求等）。
└── .docstr.yaml / .gitignore ...  # 文档/忽略规则等工程辅助文件。