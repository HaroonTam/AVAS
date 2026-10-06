# 下一阶段任务要求（交付给外部执行者）

本文档面向接手本仓库的外部 AI 执行者（例如 ChatGPT）。它自包含地说明：当前状态、硬性约束、
必须先向用户确认的决策门，以及四项候选工作的目标、步骤、产出物与验收判据。

仓库：`D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2`
执行前请先读 `AGENTS.md`（强制规则）与 `docs/development_checkpoint.md`（最新进展在文件顶部）。

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
| D2 | 可听延迟用哪种方法：录音对齐（快、粗）还是 WASAPI loopback（准、需采集脚本，可能新增可选依赖） | 影响是否新增依赖与测量精度 |
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
  3. 运行 B（听感）：`D:\python\python.exe -m app.live --config configs/system.hw_hazard.ini --console --speak-replies`。
     先输入 `描述周围` 让普通回答开始播放，然后在普通回答播放期间把人物移入画面，
     观察危险提示是否**立即**播出并打断普通回答。
  4. 验收后删除临时副本，并确认 `git status` 中 `configs/system.ini` 无改动。
- 产出物：`outputs/` 下 JSONL（忽略，不提交）+ checkpoint 新增一节，含事件计数、听感描述与限制。
- 验收判据：日志出现 `hazard` 事件且 `speech_accepted ≥ 1`；用户确认危险提示打断或先于普通回答；
  降级提示未刷屏（每 30 秒 2 条量级）。
- 禁止：把 `independent_hazard_labels` 写回默认配置；把"合成阈值"说成"已验证阈值"。

## 5. 任务 B：可听告警延迟（提交 → 起声）

**背景**：项目已实测本机中文朗读时长（150 字符 39.47 s、8 字符 3.17 s，拟合约 1.13 s + 0.256 s/字符），
但"提交时刻到实际起声"从未测量。**未测量前不得给出任何延迟数字。**

- 前置：D2 已选定方法。
- 方法 1（录音对齐，推荐先做）：
  1. 非 console 模式运行并保存逐帧日志：`... -m app.live --max-frames 300 > outputs\live_latency_<日期>.jsonl`；
     日志中每帧的提交时刻可由运行时长与 `latencies_ms` 推算，但**必须**另行记录一个可对齐的同步事件
     （例如开始时同时敲击桌面/让程序先播一条固定文本），用录音同时捕捉该同步事件与告警起声。
  2. 用录音软件（手机或系统）录制整段；对齐同一同步事件后，测量每条告警"日志提交时刻 → 录音起声时刻"。
  3. 报告至少给出样本数、mean、p50、p95、最大、异常值与误差来源。
- 方法 2（WASAPI loopback）：需写采集脚本，可能新增**可选**依赖；必须经用户同意，
  放进可选依赖组并把脚本保留在被忽略目录或 `scripts/`，不得影响生产依赖。
- 产出物：`docs/audible_latency.md`（方法、条件、样本、统计、误差来源、局限）+ checkpoint 一节。
- 验收判据：报告明确写出测量的是"提交→起声"，与 `capture_to_dispatch` 区分；
  写明录音设备与扬声器时钟不同步、人工对齐精度等误差来源；不可靠时如实写"未测量"。
- 禁止：用 `speech_accepted`、`worker_running` 或播放时长估算冒充可听延迟。

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
