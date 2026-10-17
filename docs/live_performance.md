# E1 实时性与资源测量

## 当前状态（2026-10-17）

测量入口与合成回归检查已准备；用户两次真机采集均因首帧等待超时失败，
**正常运行性能未测量，E1 尚未验收**。
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

## 首次真机尝试：失败（2026-10-17 02:48:37）

依据用户贴回的完整终端输出；原始目录保留在主工作区
`outputs/live_perf_20261017_024837_fffdfc82de384a209cc4059353e2bb7f/`，未复制或提交。
运行修订 `6273a571f4509295094bf83fd53a8ea6507c41a1`，相关路径无未提交改动；
主工作区 236 项测试、Ruff 检查和格式检查通过。

- 请求 600 帧，实际处理结果 **0**，终止故障 **1**；首帧、实际分辨率、FPS 与所有分段统计均为
  `null`，即**未测量**。日志是 `Current scene invalidated: camera read timeout`，
  资源报告 `live_failed=true`。发出了降级提示日志，用户是否听到未确认。
- YOLO11 和 Depth Anything V2 日志均为 CUDA；GPU 为 RTX 4060 Laptop，torch 2.7.0+cu118。
  相机读取使用既有 3 秒有界等待；此窗口包含启动后首帧到达前的等待。
  当前没有足够证据判定是子进程启动、驱动、设备占用还是其他原因；没有改生产超时配置。
- **仅限这次失败运行的资源观测**：441 次内存采样，采样工作集最大值 809431040 字节，
  观察到的主进程生命周期工作集峰值 827809792 字节；CUDA 分配峰值 99148288 字节。
  未处理任何有效帧，不得将这些值作为正常推理资源基线。
- 检测权重 SHA256：`0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`；
  深度权重 SHA256：`3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`；
  配置 SHA256：`1993eb14a2157a7bf6f1c49792bb03edd7cbd854b40056a544a19467dc965ad2`。
- `live_exit_code=null` 是采集包装脚本的记录缺陷，不能回填为推测的数字。
  `summary_exit_code=0` 仅表示失败记录被成功解析，不代表真机采集成功。

### 本次修复与复验范围

Windows PowerShell 5.1 无设备复现：Python 子进程休眠 0.5 秒后退出 7；
启用 stdout/stderr 重定向时，旧流程读到空退出码，在存活时缓存 `Process.Handle` 后读到 7。
采集脚本现已保留句柄，并显式检查非空零退出码、请求结果数齐全、有预热后有效结果、无终止故障。
只要不满足这些条件便显示 `measurement_complete=False` 并报错。

新增回归检查执行生产 PowerShell 脚本，只将启动目标替换为不访问设备的短进程，
验证退出 7 的保留及退出 0 但空采集的拒绝。合成输出附 `SYNTHETIC_TEST_ONLY.txt` 标记。
该检查还暴露副本 LF 换行的无 BOM 中文脚本被 Windows PowerShell 5.1 错读，
脚本改为 UTF-8 BOM 后通过。它不能证明相机超时已经解决。

下一步由用户在主工作区应用修复、关闭其他正在使用相机的软件后，按相同配置重试一次。
若仍首帧超时，另做用户终端的启动诊断；不凭当前日志放宽超时、不把失败资源样本混入正式池。

## 第二次真机尝试：超时复现，退出码修复有效（2026-10-17 03:52:39）

用户贴回主工作区已快进至 `cfabbaae8c263dc915509bf79c11d5b0d05f4aa7`，237 项测试、
Ruff 全绿；相关路径无未提交改动。原始目录保留：
`outputs/live_perf_20261017_035239_069c5b7f256e41f7883459015d36ddbe/`。

- **通过的仅为退出码记录与失败判据**：`live_exit_code=2`、`summary_exit_code=0`、
  `measurement_complete=False`。退出码不再为空，失败没有被误报为完整采集。
- **采集失败**：请求 600 帧，处理结果仍为 0，终止故障 1；首帧、实际尺寸、FPS 与各阶段
  延迟仍未测量。日志继续为 `camera read timeout`。本次还出现
  `Camera cleanup failed: Cannot close a process while it is still running`。
- **失败运行资源观测**：708 个内存样本，采样工作集峰值 813821952 字节，观察到的主进程
  生命周期工作集峰值 826966016 字节；CUDA 分配峰值 99148288 字节。设备、依赖版本、
  配置与权重 SHA256 均与第一次相同。没有有效帧，这些值不进入正常运行性能基线。
- **根因范围**：代码中 `CameraService.close()` 在 terminate 后只 join 一次便无条件 close，
  与此次“进程仍存活”的错误相符；该清理边界现已修复并通过合成测试。首次读取超时的
  根因仍未知，不把清理修复宣称为相机启动修复。

### 下一步：独立启动诊断，暂停重复 600 帧采集

清理现在在有界等待、terminate、再次等待后检查进程；必要时 kill 并再等 1 秒。
仍存活则明确报告并保留句柄供重试，不对运行中的进程调用 close；队列仅清理一次。
每次等待均有界，没有改变读取的 3 秒配置、感知或告警派发逻辑。
Windows 上 kill 与 terminate 不代表两种不同的底层强杀能力；仍可能失败，真实退出行为待复验。

新增 `app.camera_probe`：独立子进程记录进入模块、OpenCV 导入、默认后端打开相机、
设置分辨率、首次 read 与 release 的阶段和单调时刻。父进程默认 15 秒超时终止诊断，
保留已输出阶段；只保存元数据，不保存图像、不使用模型或语音。这个 15 秒仅属于
独立诊断，未写入生产配置。独立进程启动路径不同，成功不证明完整生产路径已恢复。

由用户在主工作区合并本次修复、校验通过后执行一次：

```powershell
$probeOutput = 'outputs/camera_probe_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N') + '.json'
D:\python\python.exe -m app.camera_probe --index 0 --width 640 --height 480 --timeout 15 --output $probeOutput
Write-Output "probe_exit_code=$LASTEXITCODE report=$probeOutput"
```

会打开内置摄像头读取一帧并释放，不打开麦克风或扬声器。已有报告拒绝覆盖。
用户贴回完整 JSON、退出码及异常日志。根据最后到达的阶段再决定后续检查，
不预先断定设备占用或驱动故障，不自行切换生产后端或放宽新鲜度。
