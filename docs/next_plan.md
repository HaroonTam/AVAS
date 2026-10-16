# 下一阶段计划与交接（2026-10-17）

本文档承接 [下一阶段任务要求](next_tasks.md)：其中任务 A（真机危险告警抢占）、任务 B（可听告警延迟）、
任务 C（检测精度）已完成，本文档只描述**剩余工作**（E1–E5）与可直接粘贴给 Codex 的交接提示词。
规则以 `AGENTS.md` 为准，本文档不重复全部红线，只补充本阶段的具体口径。

## 1. 当前基线（2026-10-17，交接时）

- 修订：`main` = `506219f`；工作区与该修订一致，工作区干净（主工作区另有未跟踪 `.idea/`）。
- 校验：`powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1` →
  `Ran 229 tests … OK`、`ruff check` All checks passed、`ruff format --check` 57 files already formatted、
  退出码 0。**本机 Windows PowerShell 5.1 禁止未签名脚本，必须带 `-ExecutionPolicy Bypass`。**
- 解释器：`D:\python\python.exe`（Python 3.12.9、torch 2.7.0+cu118、CUDA、RTX 4060 Laptop）。
- 硬件：内置摄像头 `HP True Vision FHD Camera`（`[camera] index = 0`）、3 个已启用 `zh-*` 音色
  （首选 Microsoft Huihui Desktop）、扬声器与麦克风可用。
- 若在 Codex 工作区（worktree）中执行：仓库元数据位于主工作区
  `D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2\.git`，
  git 写入该目录需要相应文件权限；工作区副本不含模型权重与 `data/raw/`，
  因此真机与模型相关步骤必须在主工作区执行。
- 已完成且有档：Phase 1.2 真机验收；任务 A（结论见 `docs/development_checkpoint.md`）；
  任务 B 提交→起声 p50 951.9 ms／合规池 26 条（`docs/audible_latency.md`，声程与端点启动延迟未校准）；
  任务 C COCO val2017 全量 AP 0.379／AP50 0.534、工作点 P 0.790／R 0.451（`docs/detection_accuracy.md`）；
  米制标定流程验证（`docs/metric_calibration_validation.md`，**另一域，不是本机参数**）。
- 最近两次基线修复，**不要回退**：
  1. `pyproject.toml` 设 `[tool.ruff] src = []` 与 `[tool.ruff.lint.isort] known-first-party = ["app", "tests"]`：
     ultralytics 运行时会在仓库根生成 `Ultralytics/` 设置目录，若沿用目录启发式判 first-party，
     同一提交在不同工作区的 lint 结果会相反。**新增 first-party 包时必须补进该列表。**
  2. `tests/test_spoken_reply.py` 的 `CLOCK_ROUNDING_TOLERANCE_S = 1e-6`：单调时钟落在
     `now + 50.0` 跨二进制阶的窗口时，精确 `>=` 比较会因舍入差 `2**-44` 秒而偶发失败。
- 验证 lint 时**不要**用 `ruff --config <文件>`：它会替换已发现的配置并静默关掉 `I` 规则，
  造成"全绿"的假象（本仓库 2026-10-17 已踩过一次）。

## 2. 尚未完成（本文档的全部范围）

| 编号 | 工作 | 状态 | 前置 |
| --- | --- | --- | --- |
| E1 | 实时性与资源测量（分段延迟、摄像头 FPS、显存/内存峰值） | 未测量 | 无（可立即做） |
| E2 | 任务 D 米制标定（`MetricEvidence`） | 未开始 | 决策 F1 + 相机与安装固定（F2） |
| E3 | 本项目场景精度（自采标注小样本） | 未做 | 相机采集 |
| E4 | 场景内危险准确率（漏报／误报） | 未做 | E2（需米制与阈值） |
| E5 | LLM 交互 Agent | 未实现 | 决策 F3 |

## 3. 决策门（必须先问用户，不要自行假设）

| 编号 | 问题 | 为什么必须由用户定 |
| --- | --- | --- |
| F1 | 米制方案：本机自标定，还是改用 Depth Anything V2 **Metric** 权重 | 后者改变模型身份，需先评估精度／延迟／体积／许可与论文可比性 |
| F2 | 标定用哪台相机、安装是否固定、分辨率与内参是否已知 | `support_id` 必须指向固定安装条件下的本机测量记录 |
| F3 | LLM 后端：云端 API 还是本地模型；是否允许联网与新增依赖；是否只允许结构化场景数据出网 | 涉及隐私边界与依赖策略 |
| F4 | 是否接受为 E1 修改 `app/live.py` 的计时字段（不改告警派发逻辑） | 触及实时入口的改动范围 |

F1／F2 未回答前不得开始 E2；F3 未回答前不得开始 E5。

## 4. E1 实时性与资源测量（建议先做）

现状缺口（已核实）：`app/live.py` 的逐帧 JSONL 只写 `risk` 与 `capture_to_dispatch` 两个计时字段；
检测、深度、融合的耗时只在离线基准 `app/benchmark.py` 里有（键名 `detection`／`depth`／`fusion`／`total`）；
摄像头 FPS、显存与内存峰值从未测量，而 `AGENTS.md` §13 要求这几项独立测量。

步骤：

1. 在实时循环内分别为检测、深度、融合计时，键名与离线基准保持一致
   （`detection`／`depth`／`fusion`，外加既有 `risk`），保留 `capture_to_dispatch` 的原语义；
   计时不得改变帧处理顺序、告警派发或队列行为。
2. 记录每帧时间戳与累计帧数，使 FPS 可按「有效帧数 ÷ 墙钟时间」计算，并记录分辨率与设备。
3. 资源峰值：显存用 `torch.cuda.max_memory_allocated()`（`torch.cuda.reset_peak_memory_stats()` 后采样）；
   进程内存用操作系统侧采样（PowerShell `Get-Process` 的 `WorkingSet64`/`PeakWorkingSet64`）。
   **不新增 Python 依赖**：`psutil` 虽在本机存在，但未在 `pyproject.toml` 声明。
4. 真机命令（由用户在自己的终端执行，约 1–2 分钟；输出到被忽略的 `outputs/`）：

   ```powershell
   D:\python\python.exe -m app.live --config configs/system.ini --max-frames 600 > outputs\live_perf_<日期>.jsonl
   ```

5. 统计：分段延迟 mean／p50／p95／max、FPS（首帧预热与故障行单列）、显存与内存峰值。

产出：`docs/live_performance.md`（命令、设备、分辨率、帧数、统计、计时边界、局限）+
`outputs/` 原始 JSONL（忽略，不提交）+ checkpoint 一节。

验收：写清每个数字的计时边界；**不得**把 `capture_to_dispatch` 或任何排队时间说成可听告警延迟；
不得据此声称行走安全或模型精度；未测项如实写「未测量」。

禁止：为计时引入阻塞或改变告警派发；把离线基准的 `total` 当作端到端或告警延迟；
提交 `outputs/`、原始帧或音频。

## 5. E2 任务 D 米制标定（论文核心，前置 F1/F2）

1. 固定相机与安装，记录分辨率、安装高度／俯仰与光照条件。
2. 采真值：同一安装下用卷尺实测多个目标距离（建议覆盖 0.5–5 m，每档多场景多点），
   同时保存生产融合输出的相对深度统计（用 `configs/system.ini` 的 `[fusion]`，不改统计方法）。
3. 新增 `app/fusion/metric_calibration.py`：由相对深度统计与标定参数产出
   `MetricEvidence(distance_m, support_id, valid_min_m, valid_max_m)`；
   **同时拟合** affine-in-Z 与 affine-in-1/Z 两种形式，用留出集 AbsRel 择优；
   **有效区间外必须返回 `None`**，不得外推。
4. 报告 MAE／RMSE／AbsRel、米制误差、有效区间与失效条件；`support_id` 指向标定记录文件与版本。
5. 新增测试：区间内、区间外、无效输入、缺失标定、非正距离拒绝；米制字段只有在证据可用时才非空。
6. 标定完成后才有资格谈风险阈值：默认 `high_distance_m = 1.0`／`medium_distance_m = 2.0`
   目前是**未验证的研究默认值**，不得称为已验证阈值；`independent_hazard_labels` 保持默认关闭。

产出：标定模块 + `docs/metric_calibration.md`（含真值采集记录与误差）+ `outputs/` 标定数据（忽略）+
checkpoint 一节。

验收：新函数全部有类型注解与中文 docstring；`MetricEvidence` 只由标定路径产生；
报告严格区分「流程验证（iBims-1，另一域）」与「本机标定」。

禁止：把 iBims-1 的 `(a, b)` 当本机参数；用相对深度直接当米；用 LLM 或任意 JSON 构造
`MetricEvidence`；未经同意改动默认模型或权重身份。

## 6. E3 本项目场景精度

1. 用本项目相机采集室内场景小样本（建议 50–100 张），覆盖人／椅子／自行车／车辆等风险类别与不同距离。
2. 记录来源、许可、标注规则与划分；复用 `app/detection_accuracy.py` 的评估口径
   （独立入口，不 import `app.main`／`app.live`），保存配置、权重 SHA256、输入清单、代码修订与结果，
   禁止覆盖既有结果。
3. 报告 P／R／mAP@50／mAP@50:95、推理延迟、漏检／误检示例与类别局限
   （COCO 类别不含「门」「台阶」，不得声称支持）。

验收：**必须写明样本量与小样本性质**（"仅验证流程与类别局限，不构成泛化结论"）；
不得把 COCO 结果或本小样本结果外推为"室内识别可用"；不提交数据集与权重。

## 7. E4 场景内危险准确率

前置 E2。先定义危险真值（示例："front 方向 1.5 m 内出现 person／chair 即视为应告警"），
再统计漏报与误报、逐阈值边界的表现，并报告数据来源、场景数与判定规则。
**未定义真值前不得给出任何准确率数字。**

## 8. E5 LLM 交互 Agent（前置 F3）

- LLM 只做语言组织；输入是结构化场景 JSON，不上传图像或原始音频；联网行为显式可配置。
- 告警通路不得依赖 LLM：超时、报错或返回误导内容时，现有确定性提示与告警必须照常工作。
- Agent 不得编造目标、距离、方向、置信度或风险；缺失字段保持未知。
- 测试覆盖：LLM 不可用时的降级路径、不得压制告警、不得用未提供的字段作答。

验收：`scripts/check.ps1` 全绿；新增测试使用合成数据，不依赖网络与真实模型；
报告明确未验证项（真实对话质量、网络延迟、模型版本漂移）。

## 9. 建议顺序与交付格式

顺序：**E1（零决策）** → 同时回答 F1／F2 → **E2** → **E4** → E3（可与 E2 并行）→ E5（回答 F3 后）。

每个完整改动：跑 `scripts/check.ps1` → 按 `AGENTS.md` §15 检查暂存内容
（`git ls-files`、`git diff --cached --name-status`、`git diff --cached --check`）→ **本地提交、不推送** →
在 `docs/development_checkpoint.md` 顶部新增一节，如实区分「通过／失败／未测量」。
报告失败与限制时沿用既有口径，并声明"不代表行走安全性、模型识别准确率或米制距离误差"。

## 10. 交接到 Codex 的提示词（可直接复制）

```text
你是本仓库的接手执行者，负责真实硬件验收与任务 A/B/C 之后剩余的 E1–E5 工作。

仓库：D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2

第一步，按顺序读这四份文件：
1. AGENTS.md —— 强制规则与红线
2. docs/next_plan.md —— 本阶段计划（§1 基线、§3 决策门、§4–§8 五条轨道、§10 本提示词）
3. docs/development_checkpoint.md —— 顶部两节（最新进展）
4. docs/next_tasks.md —— 任务 A/B 的原始要求（已完成，仅作口径参考）

当前基线（2026-10-17）：
- 修订 main = 506219f；校验入口 powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1
  （本机 PowerShell 5.1 禁止未签名脚本，必须带 -ExecutionPolicy Bypass），应 229 项测试通过、Ruff 全绿
- 解释器 D:\python\python.exe（Python 3.12.9、torch 2.7.0+cu118、CUDA、RTX 4060 Laptop）
- 硬件：内置摄像头 HP True Vision FHD Camera（[camera] index = 0）、3 个已启用 zh-* 音色、
  扬声器与麦克风可用
- 已完成：任务 A 真机危险告警抢占；任务 B 可听告警延迟（合规池 26 条，p50 951.9 ms，声程未校准）；
  任务 C COCO val2017 检测精度（AP 0.379／AP50 0.534，工作点 P 0.790／R 0.451）；
  米制标定流程验证（公开真值集，另一域，不是本机参数）

不要回退这两处基线修复，也不要用错误的验证方式：
1. pyproject.toml 的 [tool.ruff] src = [] 与 known-first-party = ["app", "tests"]：
   ultralytics 运行时会在仓库根生成 Ultralytics/ 目录，否则同一提交在不同工作区 lint 结果会相反；
   新增 first-party 包必须补进该列表
2. tests/test_spoken_reply.py 的 CLOCK_ROUNDING_TOLERANCE_S = 1e-6：单调时钟跨二进制阶时，
   精确 >= 比较会差 2^-44 秒而偶发失败
3. 验证 lint 时不要用 ruff --config <文件>：它会替换已发现配置并静默关掉 I 规则，得到假的"全绿"

开工前请只回答三件事，不要直接动手：
① 你建议先做哪条轨道（E1–E5）与理由；② 需要我回答哪些决策门（F1 米制方案／F2 相机与安装／
F3 LLM 后端／F4 是否允许改实时入口计时）；③ 第一步要我运行的确切命令，以及要贴回的内容。

硬性要求：
- 没测就写"未测量"，不得估算；不得用合成数据冒充真机数据；capture_to_dispatch、speech_accepted、
  worker_running 与播放时长估算都不是可听告警延迟
- 危险告警通路不得引入 LLM、网络或 Agent；Agent 不得编造目标/距离/方向/风险；相对深度不是米
- 绝不提交 .env 及任何环境变体；outputs/、模型权重、数据集、原始图像与音频都不提交
- 每个新增或修改的 Python 函数都要有完整类型注解与中文 docstring
- 需要相机、麦克风、扬声器的步骤由我在自己的终端执行：你只给命令、读我贴回的输出并记录，
  不要在我未确认时尝试打开设备
- 每个完整改动：跑校验 → 按 AGENTS.md §15 检查暂存内容 → 本地提交（不推送）→ 在
  docs/development_checkpoint.md 顶部新增一节，如实区分"通过／失败／未测量"

交付物与判据见 docs/next_plan.md §4–§8 各自的「产出」与「验收」。
```
