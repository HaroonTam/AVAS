# 离线延迟基准

`app.benchmark` 使用本地图片重复运行已加载的 YOLO11、可选 Depth Anything V2、
空间融合和离线场景有效性检查。单图保留原有 schema 1 / `offline_repeated_image_v1`；
重复指定 `--image` 时使用 schema 2 / `offline_multi_image_v1`。
它不打开摄像头、麦克风或语音输出，不调用 Agent 或云端服务。

在项目根目录、已有本地权重和完整依赖的环境中运行：

```powershell
$env:YOLO_CONFIG_DIR=(Get-Location).Path
$env:HF_HUB_OFFLINE='1'
$env:TRANSFORMERS_OFFLINE='1'
D:\python\python.exe -m app.benchmark --image data/raw/test_images/street_people_cars.jpg --depth --warmup 3 --iterations 30 --output outputs/benchmark_street_run01.json
```

每次选择新的输出名称；任何已有路径都会拒绝，保存时再次采用独占创建。
默认预热 3 次、正式采样 10 次；允许预热 0–100 次、采样 1–1000 次。
省略 `--depth` 只测检测与无深度融合，此时 `depth_ms=null`，没有深度统计项。
`--config` 默认使用 `configs/system.ini`，设备选择遵循现有自动选择和回退规则。
任一模型或处理阶段失败则中止，不把降级结果混入完整流程统计。
次数有界不代表原生推理调用有硬超时；底层算子挂起仍需终止进程。

## 计时边界

所有值均为 `perf_counter` 测得的主机墙钟毫秒。模型只构造一次，预热执行完整流程，
但不进入正式样本。正式样本保留逐次数据和 mean、median、p95、min、max；
p95 使用 NumPy 线性插值，小样本分位数只能用于检查报告流程。

| 字段 | 包含范围 |
| --- | --- |
| detection | `detect` 调用，包括预处理、推理、结果回传和结构化框解析 |
| depth | `estimate` 调用，包括预处理、推理、CPU 回传、尺寸恢复和校验 |
| fusion | 检测帧构造、区域深度统计及方向等融合处理 |
| offline_validation | 离线场景构造与风险引擎的离线拒绝检查 |
| total | 从检测调用前到上述检查完成的顺序处理时间 |

现有模型适配器在返回前读取 CPU 结果，因此这些值覆盖结果可用前的等待，
不是 CUDA 内核事件耗时。模型加载、图片读取/解码、身份哈希、报告统计/保存不计入 total。
离线拒绝检查不能作为实时风险评估或告警延迟；报告强制 `current_scene=false`，
检查结果必须为 `offline_not_current` 且无告警事件。
`serial_processing_rate_hz=1000/mean_total_ms` 仅是重复图片顺序处理速率，不能称为摄像头 FPS。
后台负载、温度及电源状态可能影响结果；固定种子不能保证跨设备数值或延迟一致。

## 报告与复现

JSON 保存输入文件名、文件及解码像素身份、尺寸、采样时间、配置、实际设备、
权重 SHA256、包版本、平台/处理器、CUDA 设备名称、PyTorch 线程数、随机种子、
Git 修订及应用源码哈希。`code_dirty` 只检查 app、configs、pyproject.toml、uv.lock；
它不是整个工作区状态。提交前试跑的报告应保留其 dirty 标记和源码哈希，不回填提交号。
正式对照应在固定修订、相同设备与输入条件下另存多次结果。

报告不保存原图、深度数组、目标类别/框/距离或音频，仅保留每次目标数量与性能元数据。
输出只在显式执行时保存，默认不上传，无自动过期；用户可自行删除对应 JSON。
配置中含本地模型路径，分享报告前可检查路径隐私。`outputs/` 已由 Git 忽略。

现有公开图片的来源及许可见 [测试图片说明](../data/test_images.md)，
只作为功能与延迟输入，没有检测框、距离或风险真值。
报告明确列出未测项目：准确率、米制距离误差、摄像头 FPS、告警与音频延迟、内存。
相对深度仍不能解释为米制距离。本入口不替代独立的数据集精度评估和真实硬件验收。

## 多图片重复采样协议

例如以下固定顺序的四图片实验（离线环境变量同上）：

```powershell
D:\python\python.exe -m app.benchmark --image data/raw/test_images/street_people_cars.jpg --image data/raw/test_images/sidewalk_traffic.jpg --image data/raw/test_images/indoor_chairs.jpg --image data/raw/test_images/bicycle.jpg --depth --warmup 5 --iterations 30 --output outputs/benchmark_multi_20261005_run01.json
```

- 输入按参数顺序执行，不扫描目录、不随机打乱。接受 1–100 个不同的解析后路径；
  同一路径重复指定会拒绝，不同路径的相同内容仍各算一项，文件/像素哈希可用于辨别。
- 所有图片先读取、解码和计算身份，再构造一次模型。图片保留在内存中，
  大图或大量输入需要相应主机内存；数量上限不等于内存用量上限。
- 对每张图片，先执行 `warmup` 次完整流程，再连续采样 `iterations` 次，然后切换下一张。
  换图后重新预热，预热值不进入任何统计。报告时间戳是各组预热开始前的 UTC 时间。
- `images` 按执行顺序保存 `image_index`、名称、哈希、尺寸、时间、逐次 `samples` 和
  `summary_ms`。样本在数组中的位置是该图正式迭代顺序，不含预热。
- 顶层 `iterations` 和 `warmup` 均是每图次数；`sample_count` 是图片数乘正式次数。
  顶层 `summary_ms` 将所有正式样本合并，样本等权；各图片采样数相等，所以图片等权。
  总体 p95/median 从合并样本计算，**不是**各图 p95/median 的平均。
  `serial_processing_rate_hz` 为 1000 除以合并样本平均 total，**不是**各图速率的平均。
- 按图分块采样会把温度、时序和后台负载变化与图片顺序混杂；结果只描述本次协议。
  单次运行、四个无真值样例和重复相关样本不能代表真实场景总体或稳定性能置信区间。
- 任一图片缺失、解码失败或推理失败，中止且不保存部分成功报告；旧结果不覆盖。

具体实测条件与汇总见 [2026-10-05 多图实验](benchmark_multi_20261005.md)。
