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

## 多轮报告汇总

`app.benchmark_summary` 只读取已有多图 JSON，不运行模型或打开硬件。
在固定干净代码修订下，分别启动新的 `app.benchmark` 进程执行相同命令，
每轮使用不同输出名称；沿用相同图片顺序、配置、设备、预热和正式次数。
例如连续三轮（同一 PowerShell 窗口，离线环境变量同上）：

```powershell
foreach ($round in 1..3) {
  $report = 'outputs/benchmark_repeated_20261005_round{0:D2}.json' -f $round
  D:\python\python.exe -m app.benchmark --image data/raw/test_images/street_people_cars.jpg --image data/raw/test_images/sidewalk_traffic.jpg --image data/raw/test_images/indoor_chairs.jpg --image data/raw/test_images/bicycle.jpg --depth --warmup 5 --iterations 30 --output $report
  if ($LASTEXITCODE -ne 0) { throw 'Benchmark failed; inspect this round before continuing.' }
}
D:\python\python.exe -m app.benchmark_summary --report outputs/benchmark_repeated_20261005_round01.json --report outputs/benchmark_repeated_20261005_round02.json --report outputs/benchmark_repeated_20261005_round03.json --output outputs/benchmark_repeated_20261005_summary.json
```

上面的文件名对应已完成实验；复跑必须改为新名称。汇总接受 2–20 个
schema 2 / `offline_multi_image_v1` 报告，要求已知 Git 修订、`code_dirty=false`，
修订及源码哈希、权重、完整配置、已记录运行环境、采样设置和有序图片身份一致。
单图旧协议、dirty 报告、缺失必要身份、样本数量不符、非法耗时或深度模式不一致会拒绝。
文件内容哈希相同或测量开始时间相同的报告会拒绝，避免复制/重复路径计数；
这不是报告来源认证，修改时间戳与内容的副本仍不能由离线汇总证明其独立性。

输出协议是 `offline_multi_run_summary_v1`，统计从原始 `samples` 重新计算，
忽略输入的预计算 `summary_ms`。顶层与每张图片均保存：

| 字段 | 含义 |
| --- | --- |
| `pooled_summary_ms` | 全部样本直接合并的 mean/median/p95/min/max |
| `per_run_summary_ms` | 每轮单独统计，数组顺序与 `sources` 一致 |
| `run_mean_summary_ms` | 每轮平均耗时组成的新样本集；另含分母 n−1 的 `sample_stddev_ms` |
| `sample_count` | 实际纳入该范围的正式样本数 |

每轮、每图采样数必须相同，因此合并均值对轮与图片等权；合并分位数不是轮分位数的平均。
轮均值标准差描述运行间均值波动，不是所有帧的标准差、标准误或置信区间；
少量轮数的 p95 仅为描述值，不应解释为稳定的总体尾部估计。
汇总保留输入报告名称/哈希/开始时间、共同条件及汇总工具源码哈希，原始报告只读且不覆盖。
输出失败不生成部分统计报告；持久化与输入一样由用户显式执行，无自动上传或过期清理。

元数据相等不证明硬件、电源、温度、驱动或后台负载受控；当前报告未覆盖全部运行环境。
独立启动进程也不等于统计独立。不要把这一工具当作精度评估或自动性能回归判定。
本次结果见 [三轮离线实验](benchmark_repeated_20261005.md)。

## 最慢样本定位

汇总新增可选读取的 `slowest_samples` 字段，原有 schema 1 / 协议与统计字段定义不变。
默认输出全局总耗时最高的 10 个正式样本（不足 10 个时全部输出），并列按原始索引升序。
`run_index` 对应 `sources`，`image_index` 对应输入图片顺序，`sample_index` 对应该图正式样本数组；
三者均从 0 开始。记录包括原始 `timings_ms`、同轮同图阶段中位数
`same_run_image_median_ms` 和可为负的 `delta_from_median_ms`。
深度禁用时这三个对象的 depth 均为 null。参照包含该组所有正式样本，也包含被选样本。

这是描述性定位，不是异常检验或因果归因；各阶段中位数及其增量不具有可加性。
输出没有样本级采集时间，不可据此对齐 GPU 温度或系统事件。
不同图的常态耗时不同，全局前十不保证覆盖所有图片；完整的逐图分布仍见原统计。
实际发现及复现命令见 [尾部延迟诊断](benchmark_tail_20261005.md)。

## 显式深度阶段诊断

在原命令上增加 `--profile-depth`（必须同时指定 `--depth`），启用有设备同步的阶段墙钟诊断。
默认命令不调用诊断计时器或新增设备同步，保留原单图/多图协议与 JSON 字段。
诊断使用 schema 3 / `offline_depth_stage_profile_v1`，单图也采用 `images` 数组；
原 `app.benchmark_summary` 会拒绝此协议，防止与未加同步的原始基准混合汇总。

每个正式样本新增 `depth_stages_ms`，每图与顶层新增 `depth_stage_summary_ms`，
统计口径仍是 mean/median/p95/min/max，正式次数相同、图片等权；所有预热样本均排除。
无论 CPU、CUDA 或 MPS 均使用主机 `perf_counter` 墙钟，**不是** CUDA event 或内核计时。

| 深度子阶段 | 边界及同步 |
| --- | --- |
| input_validation | 前置设备同步完成后开始，检查输入帧 ID、类型与尺寸 |
| preprocessing | BGR→RGB、连续内存与处理器归一化/缩放/张量构造（当前 PIL 后端在 CPU） |
| to_device | `.to(device)` 完成后同步设备 |
| inference | 模型前向、输出形状检查，随后同步设备 |
| resize | 双三次插值恢复到原图尺寸，随后同步设备 |
| to_cpu | float 转换、CPU 回传和 NumPy 视图构造，随后同步设备 |
| output_validation | 深度尺寸/类型、复制、有效掩膜、只读标记和结果对象构造 |
| total | 前置同步完成后的七阶段总计；包括同步等待及部分诊断记录开销 |

每次 estimate 前先排空选定设备；CUDA 用 `torch.cuda.synchronize()`，
MPS 用 `torch.mps.synchronize()`，CPU 同步为空操作。随后在四个设备阶段末同步。
外层 `depth_ms`/`total_ms` 包含前置同步、诊断封装与返回开销，
内部 `depth_stages_ms.total` 不含前置同步和最终记录/返回开销，因此两者不要求相等。
计时边界之间的 Python 记录、上下文切换及同步调用开销会分摊到相邻阶段；不是纯算子时间。

同步可能消除异步重叠或改变调度，主机墙钟还受 CPU/GPU 竞争影响。
这些诊断值不能直接与未加同步的旧基准作性能提升/退化比较，也不能定位底层算子根因。
诊断器只保留最近一次完整结果，开始新调用先清空；阶段顺序、时钟、同步或推理错误均中止报告。
新模式仅接入离线入口，不改变实时告警控制流、深度数值算法、精度设置或模型权重。
MPS 分派仅通过模拟测试，未做真实 MPS 实验；CPU 合成张量验证数值一致性，CUDA 实测见
[2026-10-06 深度阶段诊断](benchmark_depth_profile_20261006.md)。
