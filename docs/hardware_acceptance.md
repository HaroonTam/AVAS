# 真实硬件验收协议（Phase 1.2）

本文件是实时链路在**真实相机、麦克风、扬声器**上首次验收的可复现协议。
它不修改任何生产配置；执行者按顺序运行并如实记录，缺少设备或步骤失败时**中止并记录失败**，
不得估算、补全或推断结果。

截至本协议编写时，本项目**从未在真实设备上完成过验收**。
在本协议记录完成之前，不得声称行走安全、识别准确率、摄像头 FPS 或可听告警延迟。
首次执行记录见 [实时阶段完成记录](development_checkpoint.md) 的
「真实硬件验收（Phase 1.2，2026-10-07，修订 84cad7f）」一节：A/B/C/D 已全部执行，
其中有判据未通过（语音可听性与报告共 3 项）且仍有未测量项，上述禁止声明范围不变。

## 0. 前置检查（缺一即中止）

| 检查 | 方法 | 通过条件 |
| --- | --- | --- |
| 解释器与依赖 | `D:\python\python.exe -m app.main --help` | 打印用法，无 ImportError |
| 合成基线 | `powershell -NoProfile -File scripts/check.ps1` | 当前 203 项测试通过、Ruff 全绿 |
| 中文语音 | 见 `[speech]`；系统需已安装并启用至少一个 zh-* 音色 | 能听到合成中文 |
| 摄像头 | 确认 `[camera] index` 与实际设备一致 | `--max-frames 5` 能读到帧 |
| 安装朝向 | `[fusion] camera_orientation` | 已知前向填 `forward`，否则必须填 `unknown` |
| 交互终端 | Windows Terminal / PowerShell 真实终端 | 需要 `--console` 时 stdin 必须为 tty |

`[camera] width/height` 只是请求值，实际尺寸以帧为准。
`camera_orientation=forward` 是人为主导约定，不是自动识别结果；不确定时填 `unknown`，
否则方向与风险结论建立在未经验证的假设上。

结果日志一律写入被 Git 忽略的 `outputs/`；不要提交原始图像、音频或含个人信息的日志。

## A. 静音链路（先确认感知与时效，不播放声音）

```powershell
D:\python\python.exe -m app.live --max-frames 300 --no-speech > outputs\live_A_<日期>.jsonl
```

每帧输出一行 JSON。记录以下字段，不要只记“看起来正常”：

| 记录项 | 来源字段 | 说明 |
| --- | --- | --- |
| 帧数与退出码 | 行数、进程退出码 | 300 帧；0 正常，2 表示降级（深度／语音／场景故障） |
| 当前场景比例 | `current_scene` | 应为 true；出现 false 说明帧过期或场景失效 |
| 深度状态分布 | `depth_status` | `available` / `unavailable` / `rejected` 各多少帧 |
| 检测／深度／融合／风险耗时 | `latencies_ms` 各键 | 报告 mean、p50、p95，不要只给单帧 |
| 采集到派发延迟 | `latencies_ms.capture_to_dispatch` | **不是**告警可听延迟 |
| 观测目标数 | `observations` 长度 | 与画面是否一致由人工核对 |

通过判据：300 帧内无异常退出；深度状态明确（可用或明确 unavailable）；无可疑的耗时增长趋势。

## B. 有声告警（真实扬声器）

```powershell
D:\python\python.exe -m app.live --max-frames 300 > outputs\live_B_<日期>.jsonl
```

人工听感记录（必须如实描述，写不出就写“未验证”）：

- 中文发音是否可懂、语速是否合适；
- 降级提示是否按新节奏约每 30 秒 2 条，而不是持续刷屏；
- 高风险告警（需要合成或真实近距离目标）是否立即发声、能否打断普通回答；
- 日志 `speech_status`、`speech_accepted` 与实际听到的是否一致。

限制：`capture_to_dispatch` 与 `speech_accepted` 都只表示**提交**。
若要报告“提交到可听起声”的延迟，必须另行测量（例如录音后对齐日志时间戳），
并在报告中写明测量方法与误差来源；未测量就写“未测量”。

## C. 交互与语音输入（需要真实终端）

```powershell
D:\python\python.exe -m app.live --console --voice-input --speak-replies
```

按顺序输入并记录每条的 `status` 与回答是否**只引用当前帧事实**：

```text
帮助
描述周围
前方有什么
当前风险
寻找 椅子
左侧的椅子在哪里
listen            （说“寻找椅子”）
cancel            （在一次听取中输入）
quit
```

通过判据：回答与当前帧一致、米制距离仍为不可用、听取消与退出生效、退出后摄像头与麦克风被释放。

## D. 故障路径（每项单独记录）

| 故障 | 操作 | 期望行为 |
| --- | --- | --- |
| 相机失效 | 运行中遮挡或拔掉摄像头 | 场景立即失效；不回答旧事实；输出降级提示；退出码 2 |
| 声音不可用 | 关闭音频输出或让语音后端失败 | 文字与日志保留；明确 `spoken alerts unavailable`；感知与告警逻辑不停止 |
| 深度模型缺失 | 临时把 `[depth] model_dir` 指向不存在路径 | 保留检测与方向；`distance_status=metric_unavailable`；退出码 2 |

故障注入只改本地配置副本或临时环境，验收结束后恢复原配置，并在报告中说明改动。

## 结果记录模板

```text
日期／时间：
Git 修订：<git rev-parse --short HEAD>（若工作区有改动需注明）
解释器与设备：<python 路径>，摄像头 <型号/索引>，扬声器 <设备>，麦克风 <设备>
配置差异：<相对 configs/system.ini 的改动，例如 camera_orientation=unknown>
日志：outputs/live_A_<日期>.jsonl 等（均已忽略）

A 静音链路：帧数 __，current_scene true __%，depth_status 分布 __，
  detection mean/p50/p95 __/__/__ ms，depth __/__/__ ms，fusion __/__/__ ms，
  risk __/__/__ ms，capture_to_dispatch __/__/__ ms，退出码 __
B 有声告警：听感描述 __；降级提示节奏 __；高风险抢占 __；提交到可听起声延迟 __（未测则写未测）
C 交互：逐条 status __；是否只引用当前帧事实 __；语音识别结果 __；取消/退出 __
D 故障路径：三项结果 __

失败与未验证项：
限制声明：本结果不代表行走安全性、模型精度或米制距离误差。
```

## 回填与提交

1. 在 `docs/development_checkpoint.md` 新增一节，如实写明通过项、失败项与未测量项；
2. 若发现代码或配置缺陷，按 `AGENTS.md` §26 做最小修改并补测试，不要只改文档；
3. 提交前执行 `powershell -NoProfile -File scripts/check.ps1` 并按 §15 检查暂存内容；
4. **不提交** `outputs/` 日志、原始图像、音频或任何 `.env` 变体。
