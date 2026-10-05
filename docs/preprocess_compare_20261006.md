# PIL 与 Torchvision 预处理对照：2026-10-06

结论：本轮保留 PIL 默认实现。Torchvision 的 CPU 预处理在这些样例上较快，
但输入张量和最终相对深度不完全一致，两张图片的有效深度掩膜也有差异，不能视为无损替换。

## 协议与条件

- 独立入口 `app.preprocess_compare`，协议 `offline_preprocessor_comparison_v1`，不修改生产配置或默认后端。
- 开始时间：`2026-10-06T02:47:45.928764+08:00`（UTC+8）。使用已有四张公开本地图片，顺序见下表；文件哈希与 manifest 核对。
- 每图每种处理器预热 5 次，正式 30 次；每图 30 对、共 120 对正式 CPU 耗时。
- 每对按正式索引交替先后：偶数索引 PIL→Torchvision，奇数索引反向；正式索引从 0 重新开始。
- 每次从 BGR 原图重新转换 RGB 并调用处理器；无预处理缓存。两种处理器均使用 CPU，Torch 线程数 14。
- 计时包含 BGR 转换、处理器、输出尺寸/类型/有限值检查及 NumPy 视图构造与调用返回开销；
  不含读取图片、模型加载、GPU 推理、原图网格恢复或完整感知流水线，不能与旧阶段报告直接对比。
- 数值检查在计时之外执行：每图比较一次归一化输入张量，然后 PIL、Torchvision 各做一次深度估计。
  两者复用同一份已校验的 Depth Anything V2 Small 模型和现有后端 predict、插值、无效值处理代码。
- 模型实际设备：cuda / NVIDIA GeForce RTX 4060 Laptop GPU；深度配置 image_size=518。
- 包版本：python 3.12.9, numpy 2.2.6, torch 2.7.0+cu118, ultralytics 8.4.37, transformers 5.6.2, opencv-python 4.12.0.88, pillow 11.2.1, torchvision 0.22.0+cu118。
- 来源实现为本机 Transformers 5.6.2 的 DPTImageProcessorPil 与 DPTImageProcessor；没有复制第三方实现代码。
- 未控制电源、温度或后台负载；单次进程实验，不代表独立多轮稳态结论。
- 执行时代码修订 `60fddebc5da4aaa4955c0e03b31f3c574c993ca1`，`code_dirty=true`；原始报告保留真实状态和源码哈希。

## CPU 预处理耗时

单位为 ms，每个实现每图 30 个正式样本。

| 图片 | PIL mean | PIL p95 | Torchvision mean | Torchvision p95 | 归一化输入尺寸（NCHW） |
| --- | ---: | ---: | ---: | ---: | --- |
| street_people_cars.jpg | 30.83 | 32.52 | 11.13 | 11.69 | 1×3×518×784 |
| sidewalk_traffic.jpg | 44.95 | 45.59 | 20.44 | 21.28 | 1×3×770×518 |
| indoor_chairs.jpg | 39.29 | 40.44 | 19.71 | 20.31 | 1×3×770×518 |
| bicycle.jpg | 36.84 | 37.78 | 18.43 | 19.20 | 1×3×728×518 |

## 实现间数值差异

每张图片两种实现的张量尺寸和 dtype=float32 相同，但输入张量及最终相对深度的 exact_equal 均为 false。
以下均是两个实现之间的绝对差，不是相对真值的误差。相对深度无量纲，不作尺度对齐，也不换算为米。

| 图片 | 输入平均绝对差 | 输入最大绝对差 | 相对深度平均绝对差 | 相对深度最大绝对差 | 有效掩膜不一致像素数 |
| --- | ---: | ---: | ---: | ---: | ---: |
| street_people_cars.jpg | 0.00008013 | 0.01750714 | 0.00022702 | 0.06127068 | 21 |
| sidewalk_traffic.jpg | 0.00004455 | 0.01750720 | 0.00019672 | 0.01853538 | 12 |
| indoor_chairs.jpg | 0.00003644 | 0.01750731 | 0.00024160 | 0.02456117 | 0 |
| bicycle.jpg | 0.00006319 | 0.01750731 | 0.00027093 | 0.03168559 | 0 |

比较规则：同形时分别计算有限值掩膜；差值只在共同有效位置计算，报告共同有效计数与掩膜不一致计数。
深度结果已经过原估计器的有限且非负检查，负值已置 NaN；没有忽略掩膜变化来宣称等价。
`exact_equal` 要求 dtype、掩膜和共同有效位置值完全相等；没有共同有效值或尺寸不同时返回明确状态与 null 差值。
缺少检测、距离与风险真值，不能判断这些数值差异对准确率或实际使用是否可接受。
每图仅有一对不计时模型输出，未测同后端重复推理波动，不能把全部最终深度差异严格归因于处理器。
但预处理输入本身已有确定差异，足以否定本次“输入逐值完全等价”的替换假设。

## 使用与验证

```powershell
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
$env:YOLO_CONFIG_DIR=(Get-Location).Path
New-Item -ItemType Directory -Force outputs/test_tmp | Out-Null
$env:TEMP=(Join-Path (Get-Location).Path 'outputs/test_tmp')
$env:TMP=$env:TEMP
D:\python\python.exe -m app.preprocess_compare --image data/raw/test_images/street_people_cars.jpg --image data/raw/test_images/sidewalk_traffic.jpg --image data/raw/test_images/indoor_chairs.jpg --image data/raw/test_images/bicycle.jpg --warmup 5 --iterations 30 --output outputs/preprocess_compare_20261006_run01.json
```

复跑使用新输出名称。`--config` 默认使用现有 `configs/system.ini`；支持 1–100 个不同图片路径，
预热 0–100 次、正式 1–1000 对。程序只在显式运行时读取本地模型和图片，任一失败不保存部分结果。
原始报告位于忽略目录，不保存图片、张量或深度数组；无自动上传或清理。
报告 SHA256：`611556358d0fa0bf3d21f4aa1c51f1291d32149b7e96f2c59fa1ba08c7bec31c`。

- 新增 4 项合成测试；完整 `D:\python\python.exe -m unittest discover -s tests -q` 共 159 项通过。
- 同环境 Ruff lint（含 ANN）、format（50 文件）、`app.preprocess_compare --help` 与差异检查通过。
- 测试包含配对顺序、预热排除、不缓存输出、非法时钟、无共同有效值、形状/掩膜差异及输出防覆盖。
- 已有 Torchvision 可用，没有安装依赖、修改默认深度配置或模型权重；实时风险与语音路径未改。
- 未开启摄像头、麦克风或声音；未測端到端收益、内存、摄像头 FPS、准确率或米制距离误差。
- 对照工具是研究实验入口；候选实现尚未作为系统可选后端接入。本次代码与文档本地提交，不自动推送。
