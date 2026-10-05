# 实时阶段完成记录

2026-10-01 额度恢复后完成代码审核、故障修复、测试和文档。当前阶段实现完成，硬件效果仍未验证。

- 实时入口：`app.live`；运行方式见 `docs/live_pipeline.md`。
- 采集隔离进程、容量 1 队列、故障标志、读取超时与资源清理。
- Windows 中文 TTS 后台调度，高优先级抢占、失效取消、过期队列清理、超时与有界关闭。
- 过期结果清空当前对象并标为无效；深度失败保留检测且返回降级状态。
- 68 项合成测试通过；未启用真实相机或声音。
- 真实设备、中文语音和性能需用户显式运行检查；不宣称行走安全性已验证。
- 本阶段改动未提交、未推送，保留用户 .idea/ 和所有既有实验文件。

不再需要额度恢复续跑，关闭 heartbeat `avas`。后续工作应由新的用户任务确定。

## 场景工具阶段（2026-10-01，新聊天接续）

- 保留上述实时阶段全部未提交改动，没有新建工作树、提交或推送。
- 新增 `app/scene_store.py`：单帧内存存储、双时钟有效期、失效清理及并发读取边界。
- 新增 `app/agent/tools.py`：五个只读工具，明确处理帧内 ID、多目标、未知距离与风险。
- `app.live.main` 可注入共享存储；先提交告警后发布，故障与退出清空。
- 使用说明见 `docs/agent_tools.md`；工具 API 不是完整 LLM Agent 或 STT 接入。
- 82 项合成 unittest、Ruff lint/format、两个 CLI 帮助与离线 uv 锁检查通过。
- 本次命令：`D:\python\python.exe -m unittest discover -s tests -q`、
  `D:\python\python.exe -m ruff check .`、`D:\python\python.exe -m ruff format --check .`、
  `D:\python\python.exe -m app.main --help`、`D:\python\python.exe -m app.live --help`、
  `uv --no-cache lock --check --offline`、`git diff --check`。
  默认 uv 缓存初始化失败，使用 `--no-cache` 后离线检查通过，没有安装或下载依赖。
- 没有运行模型或硬件实验，没有开启相机、麦克风、语音或新增云端传输。
- 后续需要确定 Agent/STT 接入方案，并完成最终回答的事实校验与播放前时效校验。
- 保留原环境 `D:\python\python.exe` 与现有模型，未安装新依赖；不使用额外额度购买或重置券。

## 本地文字交互阶段（2026-10-01）

- 新增 `app/agent/vision_agent.py`：有限中文／英文命令解析、结构化请求、
  查询草稿与输出前再次查询；不一致时拒绝旧回答。
- `VisionAssistant.respond(text)` 是本地 Python 入口，尚无实时控制台输入线程，
  尚未接入普通回答 TTS、LLM 或 STT。使用方式见 `docs/local_assistant.md`。
- 新增 11 项测试：命令解析、未知事实、多目标消歧、支持米制证据、草稿过期、
  换帧、事实变化、伪造草稿、工具故障及阻塞交互期间的独立告警提交。
- 使用 `D:\python\python.exe` 执行 `-m unittest discover -s tests -q`（93 项通过）、
  `-m ruff check .`（通过）、`-m ruff format --check .`（34 文件通过）。
- 全函数注解检查包含在测试中；新增函数均包含中文 docstring。
- 未修改依赖和模型，未启用硬件或网络；未提交、推送或覆盖既有实验结果。
- 后续工作：可先接入独立文字控制台，再为普通回答增加播放前校验；
  LLM 与 STT 仍需确定后端及隐私配置。真实硬件验证仍由用户显式运行。

## 实时文字控制台阶段（2026-10-05）

- 新增 `app/agent/console.py`，通过 `app.live --console` 显式启用 Windows
  非阻塞字符轮询与独立查询线程；回答写入 stderr，隐藏逐帧 JSON。
- `quit`／`exit`／`退出` 请求主循环退出；输入结束或故障不停止独立告警通路。
- 限制每行 256 字符，超长行整体拒绝；没有新增请求队列或后台云服务。
- 输入源要求 Windows 真实交互终端，重定向 stdin 会在启动设备前拒绝。
- 普通回答仍只显示文字，尚未接入播放前校验及普通回答 TTS、LLM、STT。
- 使用 `D:\python\python.exe` 执行 `-m unittest discover -s tests -q`
  （102 项通过）、`-m ruff check .`、`-m ruff format --check .`
  （36 文件）、`-m app.live --help`，以及 `git diff --check` 均通过。
- 9 项新增合成测试验证输入编辑、超长行、查询退出、EOF、故障隔离、
  有界关闭、迟到回答丢弃，以及实时入口接线；不代表真实终端或硬件已验证。
- 未打开摄像头、麦克风或声音，未安装依赖，未提交或推送。
- 下一步可接入普通回答的播放前复核与低优先级语音，继续保持告警独立和抢占。

## 普通回答语音阶段（2026-10-05）

- 已新增 `SpokenReply` 及 `--speak-replies`，显式开启后复用现有语音队列，
  普通回答固定优先级 0；必须配合 `--console`，禁止配合 `--no-speech`。
- 查询前获取 `SceneLease`，回答按原流程复核后入队；语音后端启动前执行
  常数时间的撤销和双时钟检查，不调用 Agent、网络或风险计算。
- 任何新发布（包括同帧）、场景失效和过期使旧凭据不可用；过期后不能回拨复活。
- 超长、缺乏场景依据、后端不可用或队列拒绝均保留文字和明确状态。
- 已启动短句沿用原有完成／抢占／取消／超时策略，不随每次换帧截断；
  检查到硬件起声之间的延迟未测量，高帧率下排队回答可能经常被丢弃。
- 11 项新增合成测试覆盖普通回答提交及抢占、排队换帧／过期、故障取消、
  时钟回拨、查询期间换帧、超长、后端／队列失败、关闭、参数与实时接线。
- `D:\python\python.exe -m unittest discover -s tests -q`：113 项通过。
  同环境 `-m ruff check .`、`-m ruff format --check .`（38 文件）、
  `-m app.live --help` 和 `git diff --check` 通过，暂存区为空。
- 没有开启摄像头或播放声音，未添加依赖、提交或推送；未测量真实音频延迟与可用性。
- LLM 与 STT 尚未接入；下一阶段可补无硬件演示与端到端验收脚本，或选择后端后接入语音输入。

## Git 提交约定与合成演示（2026-10-05）

- 用户要求后续每个完整改动完成检查后提交 Git，不自动推送；已同步中英文 AGENTS。
- 既有实时与交互阶段已统一提交为 `91d23f2`，保留个人 `.idea/` 未跟踪。
- 新增 `app.demo` 和 `docs/synthetic_demo.md`，运行 6 个合成场景，
  覆盖实际融合、场景、风险、Agent 和模拟语音调度，支持 `--json` 及失败退出码。
- 演示输入、时间和米制证据均为合成夹具，报告含 synthetic 标识；
  不运行真实 YOLO11／Depth Anything V2，不访问摄像头、声音或网络。
- 新增 4 项验收测试，117 项 unittest 通过；Ruff lint/format 与演示入口验证通过。
- 未改依赖、权重或实验结果。此阶段不测量真实模型精度、FPS 或声音延迟。
- 已完成阶段与此演示分别创建本地提交；不推送，提交前检查禁入路径与暂存差异。

## 本地受限语音输入（2026-10-05）

- 新增 `app/speech/stt.py`：Windows System.Speech 固定命令语法，单次隐藏进程，
  总等待上限、取消回收、分数门槛和严格 JSON 命令白名单。
- `--console --voice-input` 只构造后端；每次输入 `listen`／`听取` 后才开麦克风。
  可配合 `--speak-replies`，但不做持续监听、自由听写或自动安装识别器。
- STT 在控制台工作线程及子进程运行，识别等待不阻塞感知和告警；
  单次识别失败显示明确状态并保留键盘查询。配置新增 `[stt]`。
- 新增 14 项合成测试；`D:\python\python.exe -m unittest discover -s tests -q`
  共 131 项通过。`-m ruff check .`、`-m ruff format --check .`（42 文件）、
  `-m app.live --help`、PowerShell 脚本静态语法解析与 `git diff --check` 通过。
- 只读枚举确认 zh-CN/zh-TW 注册项；沙箱内 SAPI 初始化失败，沙箱外中文引擎初始化
  和固定语法加载成功。验证脚本删除了音频设备绑定与 Recognize 调用。
- 未录音、开启摄像头或播放声音，未测试识别准确率、回声、设备兼容性与起播延迟。
- 未增加依赖或持久音频数据；此完整改动按用户要求单独本地提交，不推送。
- 下一步应做用户显式启动的硬件验收，或在确定后端后接入受约束的 LLM 意图解析。
