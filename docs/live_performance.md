# E1 实时性与资源测量

## 当前状态（2026-10-17）

测量入口与合成回归检查已准备；**真机数据未测量，E1 尚未验收**。
用户已确认 F4：允许增加实时入口的计时与测量记录，不改告警派发逻辑。
本次没有打开相机、麦克风或扬声器。硬件运行必须由用户在含权重与数据的主工作区执行。

代码核对更正：`b6fb2ad` 的 `PerceptionPipeline.process()` 已记录 `detection`、`depth`、
`fusion`，并通过 `latencies_ms` 写入 JSONL；并非只有 `risk` 与 `capture_to_dispatch`。
本次复用这些字段，不重复插桩，不改其语义。

## 执行方式

先将本地 E1 实现提交应用到主工作区并重新校验，然后在自己的 PowerShell 终端运行：

```powershell
Set-Location -LiteralPath 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1 -Python 'D:\python\python.exe'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/measure_live.ps1 -Python 'D:\python\python.exe' -Frames 600
```

第二条脚本显式打开相机与本地语音输出，不启用麦克风。使用现有 `configs/system.ini`，
不修改默认配置或风险规则；实际设备与模型加载情况以 `stderr.log` 为准。
每次在被忽略的 `outputs/live_perf_<时间>_<UUID>/` 下创建独立目录，保留：

- `frames.jsonl`：原有结构化场景和时延，附实际图像尺寸、累计结果数和单调时间。
- `resources.json`：配置与权重 SHA256、包版本、按配置解析的设备和 PyTorch 显存分配峰值。
- `memory_samples.json`：PowerShell 对主 Python 进程的内存采样序列。
- `run.json`：代码修订、相关路径未提交状态、解释器、请求帧数、退出码和内存统计。
- `summary.json`：正常完整帧分段 mean/p50/p95/max、计数与有效帧处理 FPS。
- `stderr.log`：模型设备信息、告警和故障日志。

所有结果仅保存在 `outputs/`，不提交；不保存原始图像或音频。脚本不下载模型、不新增依赖。
用户贴回脚本输出、两个退出码；发生异常时还需贴回错误信息。不要删除失败运行或覆盖已有结果。
用户可以用以下独立入口重新汇总（输出必须使用新的文件名）：

```powershell
D:\python\python.exe -m app.live_performance --input outputs/<本次目录>/frames.jsonl --output outputs/<本次目录>/summary_review.json
```

## 计时与统计边界

| 字段 | 边界与单位 |
| --- | --- |
| `detection` | 检测器调用前后主机单调墙钟，毫秒，含其预处理、推理与结果转换 |
| `depth` | 深度估计调用前后墙钟，毫秒；失败调用可有耗时，但不混入正常样本池 |
| `fusion` | 构造检测帧至融合完成，毫秒；含必要时的无深度重试，不含之后风险对象构造 |
| `risk` | 主循环的 `assess_scene` 调用，毫秒；不含流水线入口的时效预检查 |
| `capture_to_dispatch` | 相机原生读取开始的保守时间近似至 `dispatch_warnings` 返回，毫秒；包含读取、传输、排队和推理，不是曝光起点或可听起声 |
| `valid_processing_fps` | 首个结果派发后至最后结果派发的有效完整结果数 / 该段墙钟秒数 |

计时不添加 CUDA 同步或 CUDA events；测量是主机调用墙钟，不能解释为纯 GPU kernel 时间。
沿用现有调用中的 CPU 输出转换。不会把离线 `total` 当作实时端到端延迟。
性能记录在告警提交后附加；JSON 序列化、写盘与外部内存采样仍可能影响后续帧，开销未测量。

首个结果一律作为预热单列，不根据耗时事后挑选。正常统计池只接纳之后的
`current_scene=true` 且 `depth_status=available` 行；陈旧帧、深度故障与终止故障分别计数。
陈旧与深度故障花费的时间保留在 FPS 分母中。终止故障后的等待和语音排空不在
“最后结果派发”窗口内，因此报告同时显示终止故障计数；不得用中途失败的区间当作完整成功运行。
第一帧预热策略不保证已经达到热稳定；其余长尾保留，不做额外剔除。
缺失、乱序或损坏的正常结果行明确报错，不静默删除。空延迟池为 `null`。

有效帧处理 FPS 是摄像头驱动应用流水线的吞吐率。有界队列会丢弃旧帧，因此它**不是摄像头
原生采集 FPS**；原生采集 FPS 仍标记未测量，本次不改变采集进程来增加计数。

## 资源边界

- CUDA：模型加载后、相机启动前调用 `torch.cuda.reset_peak_memory_stats()`，退出收尾读取
  `torch.cuda.max_memory_allocated()`，单位字节。包括重置时仍存活的模型张量及之后的分配峰值；
  不包含此前模型加载的瞬时峰值。它不是缓存保留峰值、驱动占用或整卡显存占用。
- 非 CUDA 或采样失败为 `null`，分别注明 `not_cuda` / `unavailable`，不得写成 0 字节。
  资源探针失败仅记录未知，不关闭感知与告警。报告写盘失败会返回非零退出码。
- 内存：PowerShell 以请求的 100 ms 间隔轮询主 Python 进程 `WorkingSet64` 和
  `PeakWorkingSet64`；实际间隔见采样时间。记录采样工作集最大值与观察到的进程生命周期峰值。
  范围从启动至退出，包括模型初始化；不包含相机或语音子进程，也不代表全系统峰值。
  进程最后退出前的瞬间可能未采到，必须保留这一限制。无样本时为 `null`。

## 结果与验收待办

分段 mean/p50/p95/max、实际分辨率、实际帧数、有效帧处理 FPS、CUDA 分配峰值、
主进程工作集峰值：**未测量**。RTX 4060 Laptop 与内置 HP 摄像头是交接提供的环境，
本次尚未运行确认。原生摄像头 FPS、完整进程树内存峰值、测量开销：**未测量**。

收到真机输出后核对代码修订、设备、权重身份、600 个结果是否齐全、预热/故障分类、
退出码与所有计时边界，再登记实测表与 checkpoint；存在缺项时如实保留，不提前宣布 E1 完成。
本次不复测任务 B 的提交到起声延迟，不改变其声程与端点启动未校准的限制。
本报告不代表行走安全性、模型识别准确率或米制距离误差。
