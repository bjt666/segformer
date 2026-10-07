# GF 假彩色输入的 SegFormer 基线

用于四波段 TIFF 的二分类语义分割，包含训练、按轮次断点续训、独立评估、单图/文件夹预测、曲线和预测图片导出。

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

## 目录结构

```text
configs/segformer_b0.yaml
segformer_baseline/
  config.py          配置读取和校验
  dataset.py         TIFF 读取、增强、归一化、数据指纹
  model.py           模型构建和预训练适配
  engine.py          训练与评估循环
  metrics.py         二分类指标
  checkpoint.py      检查点和随机状态恢复
  visualization.py   训练曲线、预测图片
  utils.py           日志、环境记录、文件导出
train.py
evaluate.py
predict.py
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

### 训练产物

```text
outputs/实验名称/
  config.yaml                  实际训练配置
  run_config.json              同一配置的 JSON 版本
  preprocessing.json           输入波段和训练集归一化统计
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
- 旧简版的三通道 `best.pt` 可评估/预测；它没有优化器状态，不能完整续训。

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
