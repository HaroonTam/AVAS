# 下一阶段任务要求（交付给外部执行者）

本文档面向接手本仓库的外部 AI 执行者（例如 ChatGPT）。它自包含地说明：当前状态、硬性约束、
必须先向用户确认的决策门，以及四项候选工作的目标、步骤、产出物与验收判据。

仓库：`D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2`
执行前请先读 `AGENTS.md`（强制规则）与 `docs/development_checkpoint.md`（最新进展在文件顶部）。

## 0. 状态更新（2026-10-08，晚于本文档提交）

- 任务 A：**通过**（2026-10-07）。
- 任务 B：**测量完成**——2026-10-08 用 v9 协议一次集齐 5／5／3 配额，合规池 26 条，
  提交→起声 p50 951.9 ms／p95 984.1／max 1009.5；声程与端点启动延迟未校准。
  方法、样本、误差来源与局限见 [可听告警延迟](audible_latency.md)。
- 任务 C：**完成**——D3 选定「公开数据集子集」，已按 COCO val2017 全量 5000 张评估：
  AP 0.379／AP50 0.534／AP75 0.412，工作点 P 0.790／R 0.451；见
  [检测精度评估](detection_accuracy.md)。
- 任务 D：**未开始**。D4 尚未选定方案，且标定相机未确定；目前只做了流程验证
  （公开真值集、另一域，不作标定依据），见 [米制标定流程验证](metric_calibration_validation.md)。
- 第四节以下的验收判据仍然适用；下面 §1 保留本文档提交时的状态，不再逐条回填，
  测试基线由当时的 208 项增至 **229 项**。

## 1. 当前状态（截至本文档提交）

- 已完成：YOLO11 检测、Depth Anything V2 相对深度、同帧融合与方向、确定性风险引擎、
  实时摄像头入口、本地中文 TTS 调度、受限文字交互与固定命令 STT、无硬件合成演示、
  离线性能基准、真实硬件验收（Phase 1.2）及其修复。
- 真实硬件验收已执行并复验；首次验收暴露的 3 项语音缺陷（降级提示只播出 1 条、
  150 字符帮助被 15 秒固定超时截断并永久置位故障、相机故障结束提示等待过短）已修复并复验。
- 测试基线：`powershell -NoProfile -File scripts/check.ps1` 通过（208 项测试、Ruff lint/format 全绿）。
  修订号以 `git log -1 --oneline` 为准，不要把本文档中的哈希写死。
- 解释器：`D:\python\python.exe`（Python 3.12，torch 2.7.0+cu118，CUDA，RTX 4060 Laptop）。
- 已确认可用的硬件：内置摄像头 `HP True Vision FHD Camera`（`[camera] index = 0`）、
  3 个已启用 `zh-*` 音色（首选 Microsoft Huihui Desktop）、扬声器与麦克风可用。

## 2. 硬性约束（违反即返工）

1. **不得伪造任何测量结果**。没测就写"未测量"，并说明原因；不得用合成数据冒充真机数据，
   不得把 `capture_to_dispatch`、`speech_accepted` 当作可听延迟或成功率。
2. **风险决策与告警不依赖 LLM**。不得把 Agent、网络或 LLM 放进告警通路。
3. **Agent 不得编造**目标、距离、方向、置信度或风险状态；缺失信息必须保持明确未知。
4. 相对深度**不是米**。没有标定与有效区间时，米制字段必须为 `null`。
5. `.env` 与任何环境变体**永不提交**；`outputs/`、模型权重、数据集、原始图像与音频都不提交。
6. 每个新/改动的 Python 函数必须有完整类型注解和**中文 docstring**；标识符保持原文。
7. 每个完整改动完成后：运行 `scripts/check.ps1` → 按 `AGENTS.md` §15 检查暂存内容 →
   **本地提交、不推送**；并在 `docs/development_checkpoint.md` 顶部新增一节如实记录
   （含通过项、失败项、未测量项）。
8. 只做最小一致改动；不要重命名公共 API、不要替换核心模型、不要引入新依赖，
   除非用户明确同意并在记录中说明影响。

## 3. 决策门（必须先问用户，不要自行假设）

| 编号 | 问题 | 为什么必须由用户定 |
| --- | --- | --- |
| D1 | 是否执行任务 A（临时开启 `independent_hazard_labels` 验证真机危险告警抢占） | 会临时改变风险规则配置，需用户同意并知道是实验阈值 |
| D2 | 可听延迟用 **B-1 同机采集**（准，需装开发用采集依赖）还是 **B-2 外部录音**（快，误差 ≥ ±200 ms） | 影响是否新增开发依赖与测量精度 |
| D3 | 检测精度用哪种数据：人工标注现有 4 张图（只能演示流程）还是引入公开数据集子集（需来源/许可/划分） | 影响工作量、许可与结论适用范围 |
| D4 | 米制方案：自标定（相对深度→映射 + 真值点）还是改用 `Depth Anything V2 Metric` 权重 | 后者改变模型身份，需评估精度/延迟/体积/许可与论文可比性 |

**在这四个问题得到回答之前，不要开始对应任务**。可以先把不需要决策的准备工作做完
（例如阅读代码、列出步骤、准备脚本骨架，但不要提交半成品）。

## 4. 任务 A：高风险告警的真机可听抢占验证

**背景**：默认配置下没有米制标定，真机不会产生 `hazard` 事件，因此"危险告警抢占普通回答"
从未在真机上验证。`independent_hazard_labels` 是已有配置项（有单元测试覆盖），
在足够检测分数与 front 方向下产生 medium 危险事件，可用于这一验证。

- 前置：D1 已获同意。
- 步骤：
  1. 复制 `configs/system.ini` 为临时副本 `configs/system.hw_hazard.ini`，
     仅添加 `independent_hazard_labels = person,chair`；**不要改动默认 `configs/system.ini`**。
  2. 运行 A（可机读）：`D:\python\python.exe -m app.live --config configs/system.hw_hazard.ini --max-frames 300 > outputs\live_hazard_<日期>.jsonl`
     逐帧统计 `safety.events` 中 `kind="hazard"` 的数量、等级、`speech_accepted` 合计。
  3. 运行 B（听感）：
     `D:\python\python.exe -m app.live --config configs/system.hw_hazard.ini --console --speak-replies`
  4. 听感序列（细节）：画面内先**不要**有人 → 输入 `描述周围`（普通回答约 15 秒窗口）
     → 在其播放期间把人移到画面正中（需检测分数 ≥ 0.5 且方向为 front）
     → 记录危险提示是否立即播出并打断普通回答、之后是否约每 3 秒重播一次 → `quit`。
  5. 验收后删除临时副本，并确认 `git diff -- configs/` 为空。
- 机读统计脚本（保存为 `outputs/summarize_hazard.py`，忽略目录，不提交；编码自适应）：
  ```python
  import json, pathlib, sys

  raw = pathlib.Path(sys.argv[1]).read_bytes()
  for encoding in ("utf-8-sig", "utf-16", "gbk"):
      try:
          text = raw.decode(encoding)
          break
      except UnicodeDecodeError:
          continue
  rows = [json.loads(line) for line in text.splitlines() if line.strip()]
  hazards = [
      event
      for row in rows
      for event in (row.get("safety") or {}).get("events", [])
      if event["kind"] == "hazard"
  ]
  print(
      "frames", len(rows),
      "hazard", len(hazards),
      "accepted", sum(row.get("speech_accepted", 0) for row in rows),
  )
  print(sorted({(event["level"], event["message"]) for event in hazards}))
  ```
  期望：`hazard` 数量 > 0、等级 `medium`、`accepted ≥ 1`、消息为
  `前方存在配置规则关注的目标，请注意。`
- 产出物：`outputs/` 下 JSONL（忽略，不提交）+ checkpoint 新增一节，含事件计数、听感描述与限制。
- 验收判据：日志出现 `hazard` 事件且 `speech_accepted ≥ 1`；用户确认危险提示打断或先于普通回答；
  降级提示未刷屏（每 30 秒 2 条量级）。
- 禁止：把 `independent_hazard_labels` 写回默认配置；把"合成阈值"说成"已验证阈值"；
  没听到却说"已验证可听抢占"。

## 5. 任务 B：可听告警延迟（提交 → 起声）

**背景**：项目已实测本机中文朗读时长（150 字符 39.47 s、8 字符 3.17 s，拟合约 1.13 s + 0.256 s/字符），
但"提交时刻到实际起声"从未测量。**未测量前不得给出任何延迟数字。**

- 前置：用户已确认 D2。
- **方法 B-1（同机采集，推荐）**：采集音频与提交时间戳来自同一台机器的时钟，无需外部对齐。
  需要一次性安装**开发用**采集依赖（`sounddevice` 或 `soundcard`）；
  **不得**写入 `pyproject.toml` 的生产依赖，需在报告中记录版本。
  1. 安装并记录：`D:\python\python.exe -m pip install sounddevice`；`pip show sounddevice`。
  2. 写一次性探针 `outputs/audible_latency_probe.py`（忽略目录，不提交；音频只留在内存，
     不落盘、不提交）：
     - 打开 48 kHz 单声道输入流（麦克风；有立体声混音／loopback 时优先），
       流启动后立即记录 `t0 = time.monotonic()` 与 `stream.latency`（API 缓冲，计入误差）；
     - 使用**生产**的 `SpeechWorker(WindowsSpeechBackend(rate=0))`，包装 `submit()`：
       返回 True 时记录 `(time.monotonic(), text)`；
     - 依次提交真实提示文本：8 字符与 16 字符各 5 条、150 字符 3 条；
       每条 `wait_idle` 后再提交下一条，避免排队（排队时间不属于起声延迟）；
     - 结束后按 20 ms 窗口 RMS 找起声样本 `onset_sample`，则
       `onset_abs ≈ t0 + stream.latency + onset_sample / 48000`，
       `latency_i = onset_abs - submit_time_i`（麦克风到扬声器声程约 3 ms/m，计入误差）。
  3. 运行：`D:\python\python.exe outputs\audible_latency_probe.py`；安静环境，音量与验收时一致。
  4. 统计：按文本长度分组与合并给出样本数、mean、p50、p95、max；剔除误检时说明规则与剔除数量。
- **方法 B-2（外部录音，备用，误差大）**：无法安装采集依赖时使用。
  1. 手机持续录音并靠近扬声器；同时运行 `app.live` 并保存逐帧 JSONL。
  2. 报告中必须写明时钟对齐方法、人工起声判读方式、误差下界（≥ ±200 ms）与样本数；
     不得把结果写成精确延迟，必要时只给区间。
- 产出物：`docs/audible_latency.md`（方法、设备与版本、样本、统计、误差来源、局限）+
  checkpoint 一节；并明确写出这是"提交→起声"，不是场景采集→起声，也不是识别延迟。
- 验收判据：报告区分提交与起声，写明采集缓冲、起点检测阈值、声程与调度抖动等误差来源；
  不可靠时如实写"未测量"。
- 禁止：用 `speech_accepted`、`worker_running`、播放时长估算或合成计时冒充可听延迟；
  把麦克风采集的原始音频提交到仓库。

## 6. 任务 C：检测精度评估

- 前置：D3 已选定数据方案。
- 指标：Precision、Recall、mAP@50、mAP@50:95、推理延迟；类别必须使用权重实际的 `names`。
- 步骤：
  1. 标注或准备数据，记录来源、许可、版本、划分与标注规则。
  2. 写独立评估入口（不要改动 `app/main.py` 与 `app/live.py` 的生产路径），
     使用 `Yolo11Detector` 与 `configs/system.ini` 的权重；保存配置、权重 SHA256、
     输入清单、代码修订与结果，禁止覆盖既有结果。
  3. 报告漏检/误检示例与类别局限性（COCO 类别不含"门""台阶"，不得声称支持）。
- 产出物：评估脚本 + 报告文档 + `outputs/` 原始结果（忽略，不提交）+ checkpoint 一节。
- 验收判据：报告中样本量明确；若只有 4 张图，必须写明"仅验证流程，不构成泛化结论"；
  不提交数据集与权重。
- 禁止：把合成测试说成精度验证；把 COCO 类别覆盖说成"支持门/台阶识别"。

## 7. 任务 D：深度与米制距离标定

- 前置：D4 已选定方案。
- 方案 A（自标定）：
  1. 采集真值：卷尺实测若干目标距离（多场景、多距离、记录相机分辨率与安装条件），
     同时保存相对深度统计（复用现有融合输出）。
  2. 新增 `app/fusion/metric_calibration.py`：从相对深度统计与标定参数产出
     `MetricEvidence(distance_m, support_id, valid_min_m, valid_max_m)`；
     **有效区间之外必须返回 `None`**，不得外推。
  3. `support_id` 指向标定记录（文件/版本）；风险引擎的米制分支已实现并有测试，可直接复用。
  4. 报告 MAE、RMSE、AbsRel 与米制误差，明确有效范围与失效条件。
- 方案 B（Metric 权重）：必须先在报告中给出对精度、延迟、体积、许可与论文可比性的影响，
  经用户同意后再替换；`docs/depth_model.md` 的固定版本与 SHA256 记录需同步更新。
- 验收判据：新函数有类型注解与中文 docstring；新增测试覆盖有效区间、区间外、无效输入、
  缺失标定；米制字段只有在证据可用时才非空；风险阈值需实验依据，默认值不得沿用合成阈值。
- 禁止：用相对深度直接当米；用 LLM 或任意 JSON 构造 `MetricEvidence`；
  未经同意改动默认模型或权重身份。

## 8. 建议执行顺序

1. 任务 A（低成本、验证已有配置分支）→ 2. 任务 B（方法 1 优先）→ 3. 任务 D（论文核心）→
4. 任务 C（需数据准备，可与 D 并行）。

## 9. 通用交付格式

- 每次改动：`scripts/check.ps1` 全绿 → §15 检查（`git ls-files`、`git diff --cached --name-status`、
  `git diff --cached --check`）→ 本地提交（不推送）→ checkpoint 顶部新增一节。
- 提交信息风格与仓库一致：`fix: ...` / `feat: ...` / `docs: ...`，正文说明改了什么、
  验证了什么、仍未验证什么。
- 报告失败与限制时使用与既有记录相同的口径：**通过 / 失败 / 未测量** 分开列，
  并声明"不代表行走安全性、模型识别准确率或米制距离误差"。
- 需要硬件（相机、麦克风、扬声器）的步骤必须由用户在自己的终端执行；
  AI 只能给命令、读取结果并记录。受限沙箱内 `tempfile.mkdtemp` 子目录写入会被拒绝，
  测试已改用项目内临时目录，遇到其它受限情况按实际报错如实记录。

## 10. 交接提示词（可直接复制给执行者）

```text
你是本仓库的接手执行者，负责执行真实硬件验收之后的两项任务。

仓库：D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2
（若你没有文件访问权限，请先让我粘贴 AGENTS.md 与 docs/next_tasks.md 全文，再按同样流程执行。）

第一步，按顺序读这四份文件：
1. AGENTS.md —— 强制规则与红线
2. docs/next_tasks.md —— 本次任务要求（§3 决策门、§4 任务 A、§5 任务 B、§10 本提示词）
3. docs/development_checkpoint.md —— 顶部三节（最新进展；最近一次是 C 会话收尾复验）
4. docs/hardware_acceptance.md —— 硬件验收协议（命令与记录模板）

当前基线：
- 修订以 git log -1 --oneline 为准（交接时为 76758a6）；工作区应只有未跟踪的 .idea/
- 校验入口：powershell -NoProfile -File scripts/check.ps1（交接时为 208 项测试、Ruff lint/format 全绿；
  当前为 **229 项**——以实跑结果为准，不要按固定条数判失败）
- 解释器：D:\python\python.exe（Python 3.12，torch 2.7.0+cu118，CUDA，RTX 4060 Laptop）
- 硬件：内置摄像头 HP True Vision FHD Camera（[camera] index = 0）、3 个已启用 zh-* 音色
  （首选 Microsoft Huihui Desktop）、扬声器与麦克风可用
- 已完成：Phase 1.2 真机验收、3 项语音缺陷修复与真机复验、回答与固定反馈排队期限修复、
  C 会话收尾复验（退出码 0、cancel 路径通过、无 powershell 子进程泄漏）

本次范围：只做任务 A（决策门 D1）与任务 B（决策门 D2）。
任务 C/D 在我明确回答 D3/D4 之前不要开始，也不要预先写实现。
开工前必须先向我确认：
- D1：是否同意临时开启 independent_hazard_labels（只改临时配置副本，验收后删除并确认默认配置未变）
- D2：选 B-1 同机采集（需安装开发用 sounddevice，不得写入生产依赖）还是 B-2 外部录音（误差 ≥ ±200 ms）

硬性要求：
- 没测就写“未测量”；不得估算、不得用合成数据冒充真机数据；capture_to_dispatch、
  speech_accepted、worker_running 与播放时长估算都不是可听告警延迟
- 危险告警通路不得引入 LLM、网络或 Agent；Agent 不得编造目标/距离/方向/风险；相对深度不是米
- 绝不提交 .env 及任何环境变体；outputs/、模型权重、数据集、原始图像与音频都不提交；
  麦克风采集的原始音频只在内存中使用，不落盘、不提交
- 每个新增或修改的函数都要有完整类型注解与中文 docstring
- 每个完整改动：跑 scripts/check.ps1 → 按 AGENTS.md §15 检查暂存内容 → 本地提交（不推送）
  → 在 docs/development_checkpoint.md 顶部新增一节，如实区分“通过 / 失败 / 未测量”
- 需要相机、麦克风、扬声器的步骤由我在自己的终端执行；你只给命令、读我贴回的输出并记录，
  不要在我未确认时尝试打开设备

交付物与判据：
- 任务 A：outputs/live_hazard_A_<日期>.jsonl（忽略，不提交）+ checkpoint 一节。
  判据：出现 hazard 事件、等级 medium、speech_accepted ≥ 1，且我听感确认危险提示打断普通回答
- 任务 B：docs/audible_latency.md（方法、设备与版本、样本数、mean/p50/p95/max、误差来源、局限）
  + checkpoint 一节。判据：明确测的是“提交→起声”，并写明采集缓冲、起声检测阈值、声程与调度抖动

请先只回答三件事，不要直接开工：
① 你建议的执行顺序；② 现在需要我回答的决策问题；③ 第一步要我运行的确切命令，以及要贴回的内容。
```

若执行者没有文件访问权限：先粘贴 `AGENTS.md` 的核心边界章节与本文档全文，
并要求它把结论仍写入本文档说明的仓库文件（由用户代为提交）。
