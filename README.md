# GF 假彩色输入的 SegFormer 基线与结构消融

用于四波段 TIFF 的二分类语义分割，包含配置控制的结构消融、训练、按轮次断点续训、独立评估、单图/文件夹预测、曲线和预测图片导出。支持原版基线、完整改进解码器、独立 FreqFusion，以及改进解码器与 FreqFusion 的组合实验；旧 A/B 实验保留用于复现。

## 基线设置

- 默认模型：SegFormer-B0，Hugging Face Transformers 实现，保留原始 MLP 解码器。
- 编码器：`nvidia/mit-b0` 的 ImageNet-1K 预训练权重；解码器随机初始化；训练时全部参数更新。
- 假定 TIFF 源波段顺序为 `[B, G, R, NIR]`，选择零起始索引 `[3, 2, 1]`，实际输入为 `[NIR, R, G]`。TIFF 本身不改写。
- 首层预训练滤波器：NIR 使用 RGB 滤波器均值；R、G 使用对应的 RGB 滤波器。
- 只用训练集计算各输入波段均值和标准差，验证、评估、预测复用相同统计。
- 增强：水平/垂直翻转、90° 旋转。损失：不加权交叉熵。
- AdamW：编码器学习率 `6e-5`、解码器 `6e-4`、weight decay `0.01`；余弦学习率；默认 100 轮、batch size 8、seed 42。
- 验证集的**前景 IoU**用于选择最佳模型。测试集不参与训练或选模。

这些是初始基线设置，不代表已通过实验确认最优。模型改进的消融应保持输入、预训练、划分和训练设置一致。

## FreqFusion：原版基线加新模块

使用 `configs/segformer_b0_freqfusion.yaml`，独立训练 **Baseline + FreqFusion**：

- `freqfusion: true`：启用 FreqFusion。原版基线配置为 `false`；旧 YAML 未写此字段时也默认关闭。
- `decoder_fusion: mlp`、`decoder_upsampling: bilinear`：关闭此前改造的渐进融合和可学习上采样。
- `module_a: false`、`module_b: false`：关闭旧 A/B。本轮不支持 FreqFusion 与旧 A/B 组合。
- 编码器、NIR/R/G 输入、ImageNet 初始化、损失、学习率、100 轮和 seed 42 等沿用基线设置；新增模块随机初始化，训练时全部参数更新。

**网络中发生的变化：**保留四层特征的原线性投影。FreqFusion 按 F4 → F3 → F2 → F1 逐级对齐、处理特征；聚合特征的通道数依次为 D、2D、3D、4D，其中 D=256。最后仍使用原版解码器的四层特征拼接顺序和原融合层、BN、ReLU、dropout、分类层，在 F1 尺度分类后双线性恢复 logits 到输入尺寸。这里没有加入此前 `progressive` 解码器的三个 `3×3 Conv → BN → ReLU` 融合块。

FreqFusion 根据两路特征生成低通／高通核，并通过局部相似度引导的重采样调整上采样特征；目标是检验内容相关的跨尺度融合是否有效，不能预设提高指标。该实现是依据 [FreqFusion 论文](https://arxiv.org/abs/2408.12879)（TPAMI 2024，DOI：10.1109/TPAMI.2024.3449959）与[作者源码](https://github.com/Linwei-Chen/FreqFusion)机制进行的独立实现和本工程适配，不是逐字复制作者代码，也不声称数值复现原论文结果。

| 字段 | 本轮配置值 | 含义 |
| --- | --- | --- |
| `freqfusion` | `true` | 模块总开关，默认 `false` |
| `freqfusion_compressed_channels` | `64` | 核预测和特征相似度计算所用压缩通道数 |
| `freqfusion_lowpass_kernel` | `5` | 自适应低通核边长 |
| `freqfusion_highpass_kernel` | `3` | 自适应高通核边长 |
| `freqfusion_resample_groups` | `4` | 局部相似度引导的重采样分组数 |
| `freqfusion_checkpoint` | `true` | 训练时重计算部分中间量以节省显存，可能增加耗时 |
| `freqfusion_chunk_size` | `32` | 内容相关重组按通道分块计算的块大小，用于控制峰值显存 |

模块采用纯 PyTorch 运算，不新增 MMCV、自定义 CUDA 扩展或其他依赖。它仍会增加计算和显存开销。组合实验使用下节的独立配置 `configs/segformer_b0_decoder_freqfusion.yaml`；本节的 FreqFusion 单开配置保持 `mlp + bilinear`。

适配细节：动态重组采用低分辨率邻域展开和通道分块；边缘使用 `replicate` 填充；奇数尺寸的相邻特征通过补齐、计算和裁剪处理；保留 Hugging Face 的四路通道拼接顺序。这些实现选择应在复现实验时记录。

### 训练、续训与测试

在 AutoDL 项目根目录开始新的独立训练，输出目录必须尚未使用：

```bash
cd /root/segformer
python train.py --config configs/segformer_b0_freqfusion.yaml --output-dir outputs/gf16m_freqfusion_s42
```

启动日志和 `architecture.json` 应显示 `freqfusion: true`、`decoder_fusion: mlp`、`decoder_upsampling: bilinear`，且 A/B 均为 `false`。`model_summary.json` 单独记录 FreqFusion 参数数量；本轮 `decoder_redesign` 为 0，表示没有实例化此前的解码器改造部分。

首次下载预训练编码器超时，可在同一个终端执行 `source /etc/network_turbo` 后重试。若编码器已经完整缓存，可强制本次新训练离线加载：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python train.py --config configs/segformer_b0_freqfusion.yaml --output-dir outputs/gf16m_freqfusion_s42
```

离线变量不会下载缺失文件。已有本地 Hugging Face 编码器目录时，可在训练命令追加 `--pretrained-model /root/models/mit-b0`。新实验仍从 ImageNet 编码器初始化，不能使用旧基线或改进解码器的 `--resume` 来启动。

中断后恢复 FreqFusion 实验自身的完整状态：

```bash
python train.py --resume outputs/gf16m_freqfusion_s42/checkpoints/last.pt
```

结构与模块配置随检查点保存，恢复、评估和预测会自动重建。将 YAML 中的 `freqfusion` 改为 `false`，只对下一次独立新训练生效；不能用它改变已有检查点的结构。在本轮其他开关保持不变时，关闭 FreqFusion 就是原版基线。

验证集用于选择方案；确定本轮结果后，用验证集选出的最佳检查点评估测试集：

```bash
python evaluate.py --data-root dataset --checkpoint outputs/gf16m_freqfusion_s42/checkpoints/best.pt --split test
```

指标默认保存在 `outputs/gf16m_freqfusion_s42/evaluation/test-时间戳/metrics.json`，默认只导出 8 张预览，但指标覆盖全部测试图像。导出测试集每张图的预测掩膜、叠加图和有真值的对比图：

```bash
python predict.py --checkpoint outputs/gf16m_freqfusion_s42/checkpoints/best.pt --input dataset/images/test --mask-dir dataset/masks/test
```

预测默认保存在 `outputs/gf16m_freqfusion_s42/predictions/时间戳/` 的 `masks/`、`overlays/`、`comparisons/`。原始实验目录保留不动；比较时使用相同数据划分、初始化来源和训练设置。

## 完整改进解码器 + FreqFusion：组合实验

使用 `configs/segformer_b0_decoder_freqfusion.yaml`，同时开启 `decoder_fusion: progressive`、`decoder_upsampling: learned` 和 `freqfusion: true`。旧 `module_a`、`module_b` 均关闭。四组主实验对应关系如下：

| 实验 | 配置文件 | `decoder_fusion` | `decoder_upsampling` | `freqfusion` |
| --- | --- | --- | --- | --- |
| 原版基线 | `configs/segformer_b0.yaml` | `mlp` | `bilinear` | `false` |
| 基线 + 完整改进解码器 | `configs/segformer_b0_decoder_full.yaml` | `progressive` | `learned` | `false` |
| 基线 + FreqFusion | `configs/segformer_b0_freqfusion.yaml` | `mlp` | `bilinear` | `true` |
| 基线 + 完整改进解码器 + FreqFusion | `configs/segformer_b0_decoder_freqfusion.yaml` | `progressive` | `learned` | `true` |

**组合路径：**四层编码特征先各自投影到 256 通道，从 F4 开始依次与 F3、F2、F1 交互。每一级先由 FreqFusion 处理浅层特征并对齐深层特征，再将两路相加，经过 `3×3 Conv → BN → ReLU`；融合特征始终为 256 通道。最后使用两级可学习上采样恢复至输入尺寸，经 dropout 和分类层输出预测。FreqFusion 作用在逐级融合内部，不是串接在最终预测图之后。

组合版本与 FreqFusion 单开版本使用相同机制和超参数，但单开 MLP 路径会逐级拼接通道，组合路径采用相加融合，因此模块输入通道和参数量不同。实际参数量以各实验的 `model_summary.json` 为准，不能直接将两个单开实验的参数增量相加。

保持与前三组一致的数据划分、NIR/R/G、ImageNet 编码器初始化、交叉熵、学习率、batch size 8、100 轮和 seed 42，独立开始新训练：

```bash
cd /root/segformer
python train.py --config configs/segformer_b0_decoder_freqfusion.yaml --output-dir outputs/gf16m_decoder_freqfusion_s42
```

若完整编码器已经缓存，可在同一训练命令前加 `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`。本地编码器目录也可通过 `--pretrained-model /root/models/mit-b0` 指定。新实验的解码器和 FreqFusion 随机初始化，不使用前三组的检查点续训；输出目录必须尚未使用。

启动后核对 `architecture.json`：`decoder_fusion` 为 `progressive`、`decoder_upsampling` 为 `learned`、`freqfusion` 为 `true`，A/B 均为 `false`。`model_summary.json` 中 `decoder_redesign` 与 `freqfusion` 参数量均应大于 0。配置和结构随检查点保存，评估与预测会自动恢复。

仅在组合实验自身中断时恢复其状态：

```bash
python train.py --resume outputs/gf16m_decoder_freqfusion_s42/checkpoints/last.pt
```

验证集用于选择最佳检查点。训练完成后评估测试集：

```bash
python evaluate.py --data-root dataset --checkpoint outputs/gf16m_decoder_freqfusion_s42/checkpoints/best.pt --split test
```

测试指标保存在 `outputs/gf16m_decoder_freqfusion_s42/evaluation/test-时间戳/metrics.json`。默认评估预览为 8 张，指标覆盖全部测试图像。导出测试集全部预测图片：

```bash
python predict.py --checkpoint outputs/gf16m_decoder_freqfusion_s42/checkpoints/best.pt --input dataset/images/test --mask-dir dataset/masks/test
```

预测文件默认保存在 `outputs/gf16m_decoder_freqfusion_s42/predictions/时间戳/`。已有前三组在设置一致时可沿用，本轮补充组合实验即可；是否互补及是否稳定提升，需要根据实验结果判断。

## 新解码器：保留独立消融

以下配置保留此前只改解码器的路线：编码器及 F1、F2、F3、F4 四层输出保持原结构，不增加原图分支，三个配置的 `module_a`、`module_b` 均为 `false`，默认关闭 FreqFusion。是否存在稳定收益仍需通过相同设置下的独立实验和多个 seed 判断。

### 融合与上采样

四层特征均先经过各自的线性投影，统一到 256 通道。融合可选：

- `decoder_fusion: mlp`：精确复用原版四层并行 MLP 融合，包括对齐到 F1、拼接、原融合卷积、BN 和 ReLU。
- `decoder_fusion: progressive`：从投影后的 F4 开始，依次双线性对齐到 F3、F2、F1 的实际空间尺寸；每次与对应投影特征相加，再经过 `3×3 Conv → BN → ReLU`。三次融合后停在 F1，标称为输入的 1/4 尺度。

融合后的特征采用以下一种输出方式：

- `decoder_upsampling: bilinear`：在 F1 尺度经过 dropout 和 `1×1` 分类层输出 logits，再由现有训练、评估和预测流程双线性插值到输入尺寸。
- `decoder_upsampling: learned`：先用 `1×1` 卷积降到 64 通道，双线性上采样到原输入的半尺寸，再用 `3×3 Conv → BN → ReLU` 细化；随后用 `1×1` 卷积降到 32 通道，双线性上采样到精确原输入尺寸，再用 `3×3 Conv → BN → ReLU` 细化；最后经过 dropout 和 `1×1` 分类层输出 logits。各级目标尺寸由实际输入尺寸确定，不依赖连续两次固定倍率来推算最终尺寸。

这里的可学习上采样是在插值后的特征上学习细化，内部仍使用双线性插值；它不是超分辨率，也不保证补回编码或降采样中丢失的信息。

### 配置与解码器内部四组消融

新增的完整配置字段如下，YAML 未写这些字段时自动采用默认值：

| 字段 | 默认值 | 可选值与含义 |
| --- | --- | --- |
| `decoder_fusion` | `mlp` | `mlp`：原版并行融合；`progressive`：F4 → F3 → F2 → F1 渐进融合 |
| `decoder_upsampling` | `bilinear` | `bilinear`：F1 分类后插值 logits；`learned`：逐级上采样并细化特征，在原尺寸分类 |
| `upsample_channels` | `[64, 32]` | 两级可学习上采样的通道数，仅 `learned` 使用 |

当 A/B 和 FreqFusion 均关闭且采用默认 `mlp + bilinear` 时，使用原始 Hugging Face SegFormer 模型；旧 baseline YAML 不包含新字段时行为保持一致。各新配置完整保留 baseline 的预训练来源、NIR/R/G 波段、数据路径、seed 42、100 轮、batch size 8、优化器和损失等设置。

| 版本 | 完整配置文件 | 融合 | 输出方式 | 用途 |
| --- | --- | --- | --- | --- |
| 原版基线 | `configs/segformer_b0.yaml` | `mlp` | `bilinear` | 当前路线的对照 |
| 仅渐进融合 | `configs/segformer_b0_decoder_progressive.yaml` | `progressive` | `bilinear` | 检查逐级融合的作用 |
| 仅可学习上采样 | `configs/segformer_b0_decoder_upsample.yaml` | `mlp` | `learned` | 检查原尺寸分类前的特征细化 |
| 完整解码器 | `configs/segformer_b0_decoder_full.yaml` | `progressive` | `learned` | 首轮方向验证 |

全部解码器参数（包括投影、融合、上采样和分类层）使用 `decoder_lr: 0.0006` 参数组；编码器使用 `encoder_lr: 0.00006`。ImageNet 预训练只加载编码器，解码器随机初始化，训练时全部参数更新。各版本必须独立开始新训练，不能用旧 `best.pt` 或前一版本的 `--resume` 启动新结构。

首轮可先训练完整解码器，与已有、设置一致的原版基线比较；不要求立即跑满四组。若需要区分融合与上采样各自的作用，再独立运行两组单项消融。一次 seed 的结果仅用于判断下一步方向，不能据此确认稳定增益。

## 旧 A/B 结构与配置开关（保留复现）

本节是之前的局部补充与门控路线，保留其配置和诊断流程用于复现。旧配置采用默认 `mlp + bilinear` 且关闭 FreqFusion；其中 `configs/segformer_b0.yaml` 的 A/B 均关闭，使用原版 SegFormer。旧 A/B 处理发生在编码器输出与原版 MLP 解码器之间，编码器结构和原版解码器计算保留：

- **A：多感受野局部补充。** 默认分别处理 F1、F2。每个局部分支采用深度卷积、GELU 和逐点卷积，生成新增信息；A-only 将分支结果取平均后加回原特征。
- **B：深浅联合门控。** 由原浅层特征、A 的局部候选和对齐后的 F3/F4 上下文生成每个分支的空间 sigmoid 权重，再将加权补充加回原特征。门控只作用于新增信息，不直接过滤原浅层路径。各分支权重独立，不使用必须总和为 1 的 softmax，因此可同时减小。
- **浅层门控对照。** `gate_mode: shallow` 取消门控生成中的 F3/F4 上下文；最终原版解码器仍使用 F3、F4，避免把移除深层特征误当成取消引导。

B 用于选择 A 的候选信息，因此不提供 B 开启而 A 关闭的版本。局部增强和门控是否改善识别、是否减少误检，均由实验检验，不预设结果。

所有结构设置在 YAML 中配置：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `module_a` | `false` | 开启旧局部残差分支；关闭时跳过该模块，解码器仍由新字段决定 |
| `module_b` | `false` | 开启分支门控，要求 A 已开启 |
| `local_kernels` | `[3, 5, 7]` | 局部分支卷积核；奇数 3–31，可重复 |
| `refinement_stages` | `[1, 2]` | 处理 F1/F2，编号从 1 开始；可选 `[1]` 或 `[2]` |
| `gate_mode` | `deep_shallow` | 深浅联合门控；`shallow` 为浅层门控对照，仅 B 开启时使用 |
| `gate_reduction` | `4` | 门控中间通道约为当前浅层通道的 1/4，最少 1 通道 |
| `gate_init_probability` | `0.3333333333333333` | 各 sigmoid 门控的初始值，必须严格介于 0 和 1 之间 |
| `residual_init_std` | `0.001` | 局部分支末尾逐点卷积权重的初始化标准差 |

默认三个分支的门控初值为 1/3，与 A-only 的平均权重一致；局部分支以小幅残差开始。配置值不会按分支数量静默改写。修改分支数量且启用 B 时，需明确选择门控初值：两条及以上分支可用 `1/分支数`；单分支不能设为 1，可设合法值如 0.5，并记录初始增强幅度的差异。多种配置之间不保证全部随机初始化参数逐位相同。

提供以下完整配置，训练设置均为 NIR/R/G、ImageNet 编码器预训练、不加权交叉熵和 100 轮：

| 配置文件 | A | B | 用途 |
| --- | --- | --- | --- |
| `segformer_b0.yaml` | 关闭 | 关闭 | 原版基线 |
| `segformer_b0_a.yaml` | 开启 | 关闭 | 多感受野、固定平均融合 |
| `segformer_b0_ab.yaml` | 开启 | 深浅联合 | 旧 A/B 完整结构 |
| `segformer_b0_a_shallow_gate.yaml` | 开启 | 浅层门控 | 检验深层引导的作用 |
| `segformer_b0_single_scale.yaml` | `[3, 3, 3]` | 关闭 | A 的单感受野对照，保留三个独立分支 |
| `segformer_b0_ab_single_scale.yaml` | `[3, 3, 3]` | 深浅联合 | 完整结构中的单感受野对照 |

单感受野配置保留三个独立分支及 1/3 的门控初值，其深度卷积参数量仍与 `[3, 5, 7]` 不完全一致，需要记录实际模型规模。`[3]` 可作为更少参数的工程选择，但不能用来排除参数量影响；若同时启用 B，其门控初值必须小于 1，例如 0.5。

结构配置随检查点保存；续训、评估和预测按检查点重建结构，不需要另传 A/B 配置。更改结构应开启新的独立训练，不能将基线的训练后权重当作改进版本的续训起点来与独立训练基线比较。

## 目录结构

```text
configs/
  segformer_b0.yaml
  segformer_b0_freqfusion.yaml
  segformer_b0_decoder_freqfusion.yaml
  segformer_b0_decoder_progressive.yaml
  segformer_b0_decoder_upsample.yaml
  segformer_b0_decoder_full.yaml
  segformer_b0_a.yaml
  segformer_b0_ab.yaml
  segformer_b0_a_shallow_gate.yaml
  segformer_b0_single_scale.yaml
  segformer_b0_ab_single_scale.yaml
segformer_baseline/
  config.py          配置读取和校验
  dataset.py         TIFF 读取、增强、归一化、数据指纹
  model.py           模型构建和预训练适配
  refinement.py      旧 A 局部补充、B 分支门控和原版解码器包装
  decoder.py         渐进融合、可学习上采样和解码器包装
  freqfusion.py      自适应滤波、局部相似度重采样和内容相关重组
  engine.py          训练与评估循环
  metrics.py         二分类指标
  checkpoint.py      检查点和随机状态恢复
  visualization.py   训练曲线、预测图片
  utils.py           日志、环境记录、文件导出
train.py
evaluate.py
predict.py
diagnose.py          已训练 A/B 模型的残差、门控与临时旁路诊断
requirements.txt
dataset/             不参与 Git 管理
outputs/             不参与 Git 管理
```

数据集必须按文件名配对：

```text
dataset/
  images/{train,val,test}/*.tif
  masks/{train,val,test}/*.tif
```

当前数据是 `256×256×4` HWC 图像、`256×256` 掩膜，训练/验证/测试分别为 700/87/89 对。支持 `.tif`、`.tiff`；图像需有限数值；掩膜接受 0、1、255（1/255 都视为目标）。其他标签值会报错。训练/评估批次内的空间尺寸需一致。

当前切片来自同一张原始大图，缺少原图位置记录；现有划分尚未证实空间独立，跨划分重叠和空间相关性也未完全排除。首轮保留现有 dataset，不删除其中的全零背景切片，各版本使用同一划分。结果应限定为当前单场景切片划分下的比较，不能直接解释为跨日期或跨海域泛化；后续若获得独立场景，可增加外部评估。

GF 的 B-G-R-NIR 存储顺序来自数据描述，元数据没有波段名称；需要保证实际导出没有重新排列波段。

## AutoDL 环境

推荐使用已选定的 PyTorch 2.1.2 / Python 3.10 / CUDA 11.8 基础镜像。`requirements.txt` 不安装或升级 PyTorch，补装项目库即可：

```bash
cd /root/segformer
python -m pip install -r requirements.txt
```

检查 Python 和 GPU：

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

默认要求 CUDA，GPU 不可用会明确报错；需要 CPU 时显式传 `--device cpu`。从新 Conda 环境开始时，必须在该环境单独安装 GPU 版 PyTorch。

GitHub / Hugging Face 连接超时时，可在**同一个服务器终端**启用 AutoDL 官方学术加速，再重试：

```bash
source /etc/network_turbo
```

官方说明：<https://www.autodl.com/docs/network_turbo/>。安装普通软件包时若代理影响访问，可以按官方说明关闭：

```bash
unset http_proxy
unset https_proxy
```

首次新训练会下载编码器。可用 `--pretrained-model /root/models/mit-b0` 指定本地、由 Hugging Face `save_pretrained` 保存的分类编码器目录。续训、评估和预测直接从检查点重建模型，不需要重新下载预训练权重。

按当前保存镜像的方案，代码、dataset、outputs 放在 `/root/segformer/`，使用系统盘中的 Python 环境。标准实例的 `/root/autodl-tmp` 属于数据盘，不包含在系统盘镜像里。运行结束后关机，再保存镜像；镜像保留保存时的状态。

## 训练

所有命令在项目根目录执行；相对路径以终端当前目录为准。

前台训练（终端有动态进度条）：

```bash
python train.py --config configs/segformer_b0.yaml --output-dir outputs/segformer-b0-seed42-run1
```

命令行可以覆盖 YAML 中的常用参数，例如 `--epochs 100 --batch-size 8 --seed 42`。配置文件不写 `output_dir` 时，自动创建带时间戳的实验目录。新训练拒绝覆盖已有实验；重复实验请换新目录。

后台运行（SSH 断开后继续）：

```bash
mkdir -p outputs/segformer-b0-seed42-run1
nohup python -u train.py \
  --config configs/segformer_b0.yaml \
  --output-dir outputs/segformer-b0-seed42-run1 \
  > outputs/segformer-b0-seed42-run1/console.log 2>&1 &
```

查看后台日志：

```bash
tail -f outputs/segformer-b0-seed42-run1/console.log
```

`Ctrl+C` 退出 `tail` 不会停止训练。每 20 个 batch 输出一次摘要，避免动态进度条刷满后台日志；可用 `--log-every` 修改频率。不要同时在同一个实验目录启动多个训练进程。

从头随机初始化的独立实验用 `--no-pretrained`，输出到新目录。

### 首轮 GF 16 m 解码器实验（保留复现）

先独立训练完整解码器，使用新实验目录：

```bash
python train.py --config configs/segformer_b0_decoder_full.yaml --output-dir outputs/gf16m_decoder_full_s42
```

若服务器首次下载 Hugging Face 权重超时，在同一个终端执行 `source /etc/network_turbo` 后重试；已有本地编码器目录时，在上述命令追加 `--pretrained-model /root/models/mit-b0`。已有的完整训练检查点在续训、评估和预测时会自动离线重建保存的结构，不另传 `--config`，也不需要下载编码器。

训练后先评估验证集：

```bash
python evaluate.py \
  --data-root dataset \
  --checkpoint outputs/gf16m_decoder_full_s42/checkpoints/best.pt \
  --split val
```

对比已有原版基线时，核对其数据划分、波段、预训练、损失、训练设置和 seed 是否一致。优先比较前景 IoU、F1、Precision、Recall、固定样本预测及模型规模和耗时；这轮先验证方向，尚无增益或创新结论。只有需要定位两部分作用时，再按四组表补跑独立实验，例如：

```bash
python train.py --config configs/segformer_b0.yaml --output-dir outputs/gf16m_decoder_baseline_s42
python train.py --config configs/segformer_b0_decoder_progressive.yaml --output-dir outputs/gf16m_decoder_progressive_s42
python train.py --config configs/segformer_b0_decoder_upsample.yaml --output-dir outputs/gf16m_decoder_upsample_s42
```

以上输出路径都应是新目录；已存在时换新名称，避免覆盖旧结果。所有版本独立加载 ImageNet 编码器并随机初始化解码器，不能将旧基线 `best.pt` 当成新结构的续训起点。需要恢复完整解码器自身的中断训练时，直接使用它的 `last.pt`：

```bash
python train.py --resume outputs/gf16m_decoder_full_s42/checkpoints/last.pt
```

预测同样直接指定检查点，结构和归一化统计由检查点恢复：

```bash
python predict.py \
  --checkpoint outputs/gf16m_decoder_full_s42/checkpoints/best.pt \
  --input dataset/images/val \
  --mask-dir dataset/masks/val
```

有初步收益后，再对主要版本运行相同的多个 seed 并报告均值与标准差；按验证集选择结构后，才进行最终测试集评估。

### 旧 GF 16 m A/B 结构实验（保留复现）

如需复现旧路线，按顺序独立训练 **Baseline → A 固定融合 → A+B 深浅联合门控**。不要使用 `--resume` 从前一版本启动后一版本。以下实验目录仅为示例，已存在时请换新名称：

```bash
python train.py --config configs/segformer_b0.yaml --output-dir outputs/gf16m-b0-baseline-seed42
python train.py --config configs/segformer_b0_a.yaml --output-dir outputs/gf16m-b0-a-seed42
python train.py --config configs/segformer_b0_ab.yaml --output-dir outputs/gf16m-b0-ab-seed42
```

先比较三个版本的验证集最佳前景 IoU、F1、Precision、Recall 和固定样本预测图，检查新增分支是否有收益、门控是否改变误检与漏检。数据划分、输入、预训练来源、损失、训练轮次与 batch size 保持一致。按验证集结果选择结构，不使用测试集反复调参。

有初步收益后，再运行浅层门控和单感受野对照：

```bash
python train.py --config configs/segformer_b0_a_shallow_gate.yaml --output-dir outputs/gf16m-b0-shallow-gate-seed42
python train.py --config configs/segformer_b0_single_scale.yaml --output-dir outputs/gf16m-b0-single-scale-seed42
python train.py --config configs/segformer_b0_ab_single_scale.yaml --output-dir outputs/gf16m-b0-ab-single-scale-seed42
```

最终对主要版本至少运行三个相同的 seed 集合，例如 42、43、44，分别使用新输出目录并报告均值与标准差。每组仍独立加载 ImageNet 编码器，解码器和新增模块随机初始化。模型规模和运行开销应一并比较；一次 seed 的提升不足以确认稳定收益。

确定结构后，使用各版本验证集选出的 `best.pt` 做测试评估和相同测试样本的预测，命令与下文一致，只需替换检查点路径。旧基线检查点继续用于原版模型的评估/预测，旧完整训练状态继续按原结构恢复。

### 训练产物

```text
outputs/实验名称/
  config.yaml                  实际训练配置
  run_config.json              同一配置的 JSON 版本
  preprocessing.json           输入波段和训练集归一化统计
  architecture.json            FreqFusion、解码器融合/上采样及旧 A/B 结构设置
  model_summary.json           模型参数量及模型结构记录
  environment.json             原始环境、GPU、Git 提交号和工作区状态
  sessions.json                每次启动/恢复的环境和参数
  requirements-lock.txt        初次运行的 pip freeze
  requirements-lock-resume-*.txt  恢复时的 pip freeze
  dataset_manifest.json        train/val 文件名、大小和 SHA256
  train.log                    持久训练日志
  console.log                  使用上述 nohup 命令时生成
  metrics.csv                  每轮损失、验证指标、学习率和耗时
  metrics.jsonl                同一历史记录的 JSONL 版本
  summary.json                 最近完成轮次、最佳轮次、完成状态
  checkpoints/
    last.pt                    最近完成的一轮，完整续训状态
    best.pt                    验证集前景 IoU 最优的一轮
  curves/
    loss.png
    metrics.png
    learning_rate.png
  visualizations/epoch_0001/comparisons/*.png
```

每轮保存排序固定的前 8 张验证图对比，可用 `--visualization-samples 0` 关闭。图像包含原图、真值、预测和误差；没有人为挑选展示样本。曲线每轮更新。学习率字段记录该轮实际使用的学习率，`next_*` 记录下一轮学习率。

## 断点续训

```bash
python train.py --resume outputs/segformer-b0-seed42-run1/checkpoints/last.pt
```

- 从最近**完成**轮次的下一轮恢复。第 31 轮中途断电，最近检查点是第 30 轮，则第 31 轮重跑。
- 恢复模型、AdamW、学习率调度器、最佳指标、历史记录、Python/NumPy/PyTorch/CUDA 随机状态和 DataLoader 生成器状态。
- 用临时文件写入再替换检查点；`last.pt` 先保存，恢复时可修复中断造成的未更新 `best.pt`。
- 恢复复用检查点的归一化统计，并验证 train/val 内容指纹；数据变化会拒绝恢复。
- 保存的是完整原计划，例如共 100 轮，中断后依然完成到 100 轮。续训禁止更改总轮次、学习率、batch size、seed、输入波段等设置，避免悄悄改变实验。新设置请开始新实验。
- 可更改 `--data-root`、`--output-dir`、`--device`、`--num-workers` 和日志/预览数量。更改 workers、硬件或软件可能改变数值结果；随机状态恢复不保证跨环境逐位一致。
- 迁移服务器时复制**完整实验目录**，保留 `last.pt` 和 `best.pt`。指纹与绝对数据路径无关，目录移动后可以恢复。
- 第一轮尚未完成时还没有检查点。当前不支持在一轮中途精确恢复 batch 位置。
- 旧简版的三/四通道 `best.pt` 可评估/预测；缺少通道字段时按已知旧版的输入顺序恢复。它没有优化器状态，不能完整续训。

恢复到新目录的例子：

```bash
python train.py \
  --resume outputs/原实验/checkpoints/last.pt \
  --data-root /root/segformer/dataset \
  --output-dir outputs/恢复实验
```

## 独立评估

用验证集选出的 `best.pt` 对测试集评估：

```bash
python evaluate.py \
  --data-root dataset \
  --checkpoint outputs/segformer-b0-seed42-run1/checkpoints/best.pt \
  --split test
```

默认输出到 `outputs/实验名称/evaluation/test-时间戳/`。支持 `--split val`、`--output-dir`、`--batch-size`、`--num-workers`、`--device` 和 `--visualization-samples`。

结果包含 `metrics.json`、`per_image_metrics.csv`、混淆矩阵 CSV/PNG、固定样本可视化、数据清单和环境记录。`metrics.json` 记录检查点 SHA256、轮次、输入波段和指标。

### 指标定义

混淆矩阵按**行是真值、列是预测**排列：`[[TN, FP], [FN, TP]]`。

- `foreground_iou`：前景交并比；用于选模。
- `background_iou`：背景交并比。
- `miou`：前景与背景 IoU 的均值。
- `dice` / `foreground_f1`：前景 Dice 和 F1，二分类分割时数值相同。
- `precision`、`recall`：前景精确率、召回率。
- `pixel_accuracy`：所有像素的准确率。

总体指标由整个数据集**累计像素混淆矩阵**计算，不是逐图指标均值。分母为 0 的指标记为 JSON `null` / CSV 空值；mIoU 只平均已定义的类别 IoU。真值和预测都无前景时，该图前景 IoU/Dice 为未定义，不计成满分；误检时对应前景 IoU 为 0。逐图结果用于诊断，报告时应说明统计方式。

## 模块作用诊断

当 A、A+B 的结果接近原版时，用已经训练完成的 `best.pt` 检查模块实际改变了多少特征、已训练模型是否依赖这些改动。默认使用验证集，无需重新训练或下载预训练权重：

```bash
python diagnose.py \
  --checkpoint outputs/gf16m_a_s42/checkpoints/best.pt \
  --data-root dataset \
  --split val

python diagnose.py \
  --checkpoint outputs/gf16m_ab_s42/checkpoints/best.pt \
  --data-root dataset \
  --split val
```

检查点目录应与实际训练结果一致。支持 `--output-dir`、`--batch-size`、`--num-workers`、`--device` 和 `--log-every`，默认输出到 `outputs/实验名称/diagnostics/val-时间戳/`。诊断针对带 A 的模型；原版检查点仍使用 `evaluate.py` 评估。

在同一份已训练权重上比较以下推理模式：

| 模式 | 临时操作 | 检查什么 |
| --- | --- | --- |
| `normal` | 使用检查点保存的完整结构 | 当前模型的正常表现 |
| `no_residual` | 所有配置选中的浅层均直接传入原特征 F，新增补充为 0 | 模型对新增补充的依赖 |
| `uniform_gates` | 仅 A+B 提供，浅层输出改为 F + 三支补充的平均；一般 K 支时等价于各支权重固定为 1/K | 学习到的门控是否影响输出 |

正常模式记录 F1/F2 原特征、各局部分支、未加权平均补充及最终新增量的 RMS，以及新增量/原特征的 RMS 比值。每图统计汇总为均值、标准差和分位数；原特征 RMS 为 0 时比值记为未定义并单独计数。A+B 还记录每支门控的均值、空间变化、分位数、相对初始化概率的变化，以及门控最后一层权重是否离开全零初值。全体门控的分位数使用有误差界说明的直方图近似，均值、标准差和初值偏差直接从实际门控计算。

各模式比较全验证集累计像素指标、交叉熵、相对正常模式的 IoU/F1 百分点差，以及前景概率平均绝对变化和预测像素变化。另报告真值或任一预测前景并集区域内的变化比例，避免大量背景掩盖前景改动。

判断时需要结合内部统计与输出对照：

- 新增量和临时旁路后的输出变化都很小，支持当前模型对补充依赖较弱的判断；不能仅凭一个任意百分比断言模块失效。
- 门控空间变化很小，仍可能学到了有用的全局分支权重；应同时看它与初值的差异，以及固定均匀权重后的预测变化。
- 独立 sigmoid 门控的权重和不固定。改为均匀权重同时改变分支分配和补充总幅度，单靠这一对照不能证明空间选择有效或深层语义引导有效。
- 临时旁路或固定门控使指标变差，表示该已训练模型依赖模块；这不自动说明它优于独立训练的原版。
- 门控或补充有明显变化，但正常模式没有更好的验证表现，提示当前变化未带来净收益；不能仅凭这一次诊断定位所有训练原因。

这些是**训练后的作用与敏感性检查**，不替代之前各版本独立训练的正式消融。临时关闭补充的模型仍保留该检查点训练后的编码器和解码器，不能当作重新训练的原版；固定门控也不能当作独立训练的 A。进一步调整方案使用验证集结果。

诊断结果文件：

```text
diagnostic_report.json             模式对照、残差/门控汇总、训练配置和权重来源
interventions.csv                  各模式指标和相对正常模式的变化
stage_statistics.csv               各浅层的逐图统计汇总
per_image_stage_statistics.csv     每张图的残差和门控统计
per_image_interventions.csv        每张图在各模式下的指标及输出变化
dataset_manifest.json              本次验证/测试数据的文件指纹
environment.json                  本次诊断的环境记录
diagnose.log                       诊断日志
```

将 A、A+B 各自的 `diagnostic_report.json` 保存到本地结果目录，就可以结合现有训练与预测记录判断下一步调整方向。报告还保留检查点的训练配置、训练数据指纹和归一化参数，便于核对实验控制变量。

## 预测和图片导出

单张图像无需标签：

```bash
python predict.py \
  --checkpoint outputs/segformer-b0-seed42-run1/checkpoints/best.pt \
  --input dataset/images/test/000001.tif
```

上面的文件名只是示例，请换成真实文件名。整个文件夹预测（标签可选）：

```bash
python predict.py \
  --checkpoint outputs/segformer-b0-seed42-run1/checkpoints/best.pt \
  --input dataset/images/test \
  --mask-dir dataset/masks/test \
  --save-probabilities
```

默认输出到 `outputs/实验名称/predictions/时间戳/`：

```text
masks/*.tif.png                单通道二值预测，背景 0、目标 255
overlays/*.tif.png             原图与黄色预测区域叠加
comparisons/*.tif.png          无标签时原图/预测/叠加，有标签时原图/真值/预测/误差
probabilities/*.tif            可选 float32 前景 softmax 概率，范围 [0,1]
per_image_metrics.csv          提供 mask-dir 时生成
prediction_manifest.json      模型和输入指纹、设置、可选总体指标
environment.json
predict.log
```

保留原文件名后再加 `.png`，例如 `000001.tif.png`，避免不同 TIFF 扩展名的同名文件互相覆盖。误差图：绿色 TP、红色 FP、蓝色 FN、黑色 TN。假彩色显示固定 NIR-R-G，各波段做 2%/98% 百分位拉伸；这只用于显示，模型仍使用训练集统计归一化。

预测与评估统一采用 logits 的 argmax，不额外调整阈值。按单张图推理并恢复到原像素尺寸；主要面向当前切片，不包含大幅遥感影像的滑窗拼接。导出的 PNG 和概率 TIFF **不复制 GeoTIFF 坐标元数据**。

## Git 管理与服务器更新

`dataset/`、`outputs/`、权重、虚拟环境等由 `.gitignore` 排除。代码提交使用当前 Git 用户身份。

已用 Git 克隆的服务器更新：

```bash
cd /root/segformer
source /etc/network_turbo
git pull --ff-only
python -m pip install -r requirements.txt
```

如果服务器只上传了代码且没有 `.git`，重新上传源码、`configs/`、`segformer_baseline/`、`requirements.txt` 和 README，保留已有 dataset/outputs。更新正在训练的代码应在该进程结束或停止后进行，避免下一次启动 worker 时导入不同版本。

实验结果可从系统盘保存到私有镜像；Git 忽略不影响镜像内容。论文记录应包括代码提交、依赖版本、数据指纹、输入波段、预训练来源、训练设置和随机种子。一次运行只能给出一个 seed 的结果；后续需要按论文实验设计汇总多个 seed。
