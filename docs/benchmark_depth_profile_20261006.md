# 深度子阶段同步诊断：2026-10-06

使用显式 `--profile-depth` 模式，检查此前深度阶段长尾内部可能值得关注的环节。
这是新协议的一次描述性诊断，不是性能优化对照，也不能据此解释上一轮长尾的根因。

## 运行条件与复现

- 开始时间：`2026-10-06T02:37:55.408035+08:00`（UTC+8）。
- 四张现有公开本地图片，按下表顺序执行；每图预热 5 次、正式 30 次，总正式样本 120。
- YOLO11n + Depth Anything V2 Small 相对深度；沿用 `configs/system.ini`，检测 640、置信度 0.35，深度尺寸 518。
- 两个模型均为 CUDA / NVIDIA GeForce RTX 4060 Laptop GPU，PyTorch 线程数 14。
- 依赖版本：python 3.12.9, numpy 2.2.6, torch 2.7.0+cu118, ultralytics 8.4.37, transformers 5.6.2, opencv-python 4.12.0.88。
- 模型只加载一次，逐图预热；模型加载、读取/解码、哈希和报告保存不计入正式耗时。
- 未修改或控制电源模式、温度和后台负载；采样期间有少量文档读写，未运行单元测试。
- 当前协议对每次深度调用前、to_device/inference/resize/to_cpu 后执行设备同步。
  前置同步计入外层 depth_ms，但排除在内部阶段 total 之外；精确边界见 [基准说明](benchmark.md)。
- 代码修订：`29c9b0993dd6526e876829a8d054b67c46d1045c`，`code_dirty=true`，保留执行时真实源码哈希，不回填提交号。

```powershell
$env:YOLO_CONFIG_DIR=(Get-Location).Path
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
New-Item -ItemType Directory -Force outputs/test_tmp | Out-Null
$env:TEMP=(Join-Path (Get-Location).Path 'outputs/test_tmp')
$env:TMP=$env:TEMP
D:\python\python.exe -m app.benchmark --image data/raw/test_images/street_people_cars.jpg --image data/raw/test_images/sidewalk_traffic.jpg --image data/raw/test_images/indoor_chairs.jpg --image data/raw/test_images/bicycle.jpg --depth --profile-depth --warmup 5 --iterations 30 --output outputs/benchmark_depth_profile_20261006_run01.json
```

复跑必须更换输出名称。没有开启摄像头、麦克风、声音或外部 LLM，没有新增依赖。

## 实测结果

以下为主机墙钟毫秒，包含阶段同步和部分诊断开销，不是纯 GPU 算子时间。

| 图片 | 预处理 mean | 到设备 mean | 前向 mean | 尺寸恢复 mean | 回传 mean | 输出校验 mean | 内部总计 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| street_people_cars.jpg | 47.98 | 1.45 | 54.60 | 0.27 | 1.38 | 4.34 | 110.03 |
| sidewalk_traffic.jpg | 49.88 | 1.15 | 56.44 | 0.35 | 2.06 | 7.45 | 117.33 |
| indoor_chairs.jpg | 45.98 | 1.06 | 46.84 | 0.29 | 1.90 | 6.71 | 102.80 |
| bicycle.jpg | 42.21 | 1.00 | 44.28 | 0.28 | 1.55 | 6.02 | 95.33 |

| 合并 120 样本的深度阶段 | mean | median | p95 | max |
| --- | ---: | ---: | ---: | ---: |
| input_validation | 0.006 | 0.005 | 0.008 | 0.061 |
| preprocessing | 46.512 | 46.093 | 52.310 | 68.114 |
| to_device | 1.165 | 1.064 | 1.613 | 1.904 |
| inference | 50.539 | 46.689 | 64.711 | 88.182 |
| resize | 0.298 | 0.273 | 0.335 | 1.925 |
| to_cpu | 1.722 | 1.316 | 2.795 | 3.692 |
| output_validation | 6.130 | 6.236 | 7.984 | 9.505 |
| total | 106.372 | 102.866 | 134.388 | 152.828 |

外层深度 mean **106.66 ms**，内部七阶段 total mean **106.37 ms**；外层完整流水线 mean **124.39 ms**。
所有 120 个样本的内部 total 与七阶段和逐项核对一致，且不大于其外层 depth_ms。
阶段 p95/max 对应的样本未必相同，不可相加解释总耗时的 p95/max。

## 能支持的判断与限制

- 本轮预处理（46.51 ms）和前向（50.54 ms）占据主要的深度内部平均时间，传输和插值均较小。
  预处理采用现有 PIL 路径；没有试验替换处理器、模型、输入尺度、精度或融合算法。
- 本轮外层深度最大值约 153.67 ms，没有重现旧样本约 273.71 ms 的深度长尾；这不证明问题已解决。
- 前向计时包含 Python 调用、设备执行、同步等待及系统调度影响，不能将其全部当作 GPU 计算。
- 同步会改变异步重叠与调度，加上跨时段负载差异，不能拿本轮 124.39 ms 与此前 149.92 ms 宣称提升。
- 若进一步尝试降低预处理成本，应使用相同输入做数值/尺寸/有效掩膜对照，并用未加同步基准独立验证；
  本次未做此项优化。有限图片无真值，不能报告准确率、米制误差、摄像头 FPS 或行走安全性。
- 未测内存、告警/音频延迟、GPU 温度或内核级耗时；真实 MPS 未验证。

## 输出与验证

- 本地报告：`outputs/benchmark_depth_profile_20261006_run01.json`，SHA256：`26a3c2f236606a4fa316464709a55b99c4d3f7dade779279694aa54da8257bd8`。
- 包含逐次深度子阶段、逐图/整体统计、配置、图片/权重/源码哈希；不保存原始图像、深度数组或目标事实。
- 原有报告不覆盖；新增 schema 3 明确标识同步诊断，原跨轮汇总会拒绝混用。
- 使用 `D:\python\python.exe -m unittest discover -s tests -q`：155 项通过，含完整函数注解检查。
- 同环境 `-m ruff check .`、`-m ruff format --check .`（48 文件）、`-m app.benchmark --help` 通过。
- 初次定向测试因 Windows 沙箱系统临时目录权限失败；将 TEMP/TMP 指向项目忽略目录后重跑通过。
- 6 项新增测试覆盖阶段边界、同步分派、时钟/顺序/同步/推理失败清理、预热剔除、诊断协议隔离。
  使用真实后端方法与合成 CPU 张量验证诊断前后深度和掩膜一致；不是模型精度验证。
- 新增/修改函数有完整类型注解和中文 docstring；本地提交，不自动推送。
