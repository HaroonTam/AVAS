# 视障人士视觉辅助系统

基于 YOLO11 与 Depth Anything V2 的本科毕业设计研究原型。

## 当前进度

第一阶段：已建立 YOLO11 本地单张图片检测入口、配置校验、检测框解析、可选结果绘图与合成单元测试。
已接入 Depth Anything V2 Small 的本地相对深度推理、原始深度存档与预览。
已实现同帧目标与相对深度融合、左／前／右图像方向估计及可配置统计方法。
已实现配置化确定性风险引擎、场景时效校验与有界告警事件去重接口。
已接入实时摄像头入口和 Windows 本地中文语音调度，支持高优先级抢占与故障降级。
已实现带时效校验的单帧场景存储及五个只读 Agent 场景工具。
已新增本地受限文字交互和输出前事实复核，可处理明确的中文查询命令。
米制标定、LLM 交互 Agent、STT 尚未实现；真实摄像头与可听语音效果尚未完成硬件验证。
当前仍为研究原型，不能视为经过验证的行走辅助。离线图片不作为当前场景。

## 环境准备（PowerShell，Python 3.10+）

已有项目环境时，可先运行无需硬件的合成验收演示：

```powershell
D:\python\python.exe -m app.demo
```

该入口不打开摄像头、不加载模型或播放声音；`--json` 输出带合成标识的机器可读报告。
使用固定测试输入验证融合、风险、查询和模拟语音通路，不代表真实性能或安全性。
详见 [无硬件演示说明](docs/synthetic_demo.md)。

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e ".[vision]"
```

依赖唯一来源为 `pyproject.toml`。CUDA 环境请按 PyTorch 官方安装选择器安装与驱动兼容的版本；
不假定默认 pip 安装具有 GPU 支持。配置支持 `auto`、`cpu`、`cuda`、`mps`。
设备不可用时记录 CPU 回退；算子执行失败会明确退出，不隐瞒故障。

## 准备模型并运行

从 [Ultralytics YOLO11 官方说明](https://docs.ultralytics.com/models/yolo11/)
的 Detection 模型表下载可信的 `yolo11n.pt`，放到 `models/yolo11n.pt`。
也可在 `configs/system.ini` 指定其他 YOLO11 检测权重；相对路径以配置文件所在目录为基准。
默认配置按程序文件所在位置查找项目的 `configs/system.ini`，不依赖启动工作目录。
显式传入 `--config` 或 `--image` 的相对路径仍以当前工作目录为基准；
在 IDE 中从其他目录启动时，图片建议使用绝对路径。
程序不自动下载权重。模型仅初始化一次，检测器可重复处理帧。

```powershell
.\.venv\Scripts\python -m app.main --help
.\.venv\Scripts\python -m app.main --image "D:\images\example.jpg"
```

输出 JSON 含原图尺寸、实际模型类别表、检测结果及单次调用耗时。
`bbox` 是原图像素坐标 `[x1,y1,x2,y2]`，左上包含、右下不包含，向外取整并裁剪至图像边界。
置信度为模型检测分数，不是距离可信度或安全概率。
类别来自实际权重的 `names`；没有出现在表中的门、台阶等类别不属于已实现能力。
空检测列表不说明道路安全。距离始终为 `metric_unavailable`，风险为 `unknown`。
耗时包含首轮预热、预处理、推理、后处理和结果转换，不等于稳态 FPS 或完整系统延迟。

推理使用本地解码的 BGR 图像；不上传，默认不保存原始图片或绘制结果。
结构化结果输出至终端，不建立持久化存储。仅使用可信来源的 PyTorch checkpoint。
Ultralytics 软件及模型的使用须遵循其 [AGPL-3.0 / 企业许可说明](https://www.ultralytics.com/license)。
未在本项目中分发第三方权重或数据集。

## 查看检测框

显式传入 `--output-image` 可保存带类别和置信度的检测图：

```powershell
python -m app.main --image data/raw/test_images/street_people_cars.jpg --output-image outputs/street_result.png
```

PyCharm 使用模块 `app.main`，工作目录为 `$PROJECT_DIR$`，参数填写：

```text
--image data/raw/test_images/street_people_cars.jpg --output-image outputs/street_result.png
```

输出支持 PNG/JPG/JPEG 和中文路径，自动创建父目录；相对路径以工作目录为基准。
已有文件不会被覆盖，重复实验请更换输出文件名。图片含源图内容，保存后保留至手动删除。
`outputs/` 已被 Git 忽略。检测框颜色仅用于显示，不代表风险等级；
标签使用模型原始英文类别及检测分数，不显示未经验证的距离或安全结论。
绘图不修改送入感知的原始数组，也不计入 `detection_latency_ms`。

## Depth Anything V2 相对深度

当前环境已经准备好模型。新环境安装依赖：

```powershell
python -m pip install -e ".[vision,depth]"
```

使用相同图片运行检测和深度估计（输出文件名需尚不存在）：

```powershell
python -m app.main --image data/raw/test_images/street_people_cars.jpg --depth --output-depth outputs/street_relative.npz --output-depth-image outputs/street_relative.png
```

PyCharm 中将以上 `--image` 开始的部分填入参数，工作目录保持 `$PROJECT_DIR$`。
也可同时添加 `--output-image outputs/street_boxes.png` 保存目标框。
`--depth` 默认不启用；只有传入输出参数才保存深度文件。

深度图中亮表示相对近，暗表示相对远，紫色为无效像素。
显示使用每张图片独立的最小最大值归一化，不可跨图片按亮度比较距离。
`.npz` 保存未归一化的 float32 相对深度、有效掩膜、离线输入标识及模型版本。
深度值**不是米**；`distance_status` 仍为 `metric_unavailable`，`risk_level` 仍为 `unknown`。
深度失败时保留检测结果并返回退出码 2；模型加载与推理均不联网。
模型来源、固定版本、重新下载方式及插值差异见 [深度模型说明](docs/depth_model.md)。

## 目标与深度融合、方向估计

以上命令现在自动输出 `fusion.objects`：每个目标包含帧内 ID、检测分数、原图框、方向、
相对逆深度统计、采样质量和明确的距离／风险未知状态。未启用或无法获得深度时仍输出目标与方向。
不需要新增命令行参数；实验方法及镜像等设置位于 `configs/system.ini` 的 `[fusion]`。

默认取框中心宽高各 50% 区域，过滤无效值并裁去分位数极端值后计算中位数。
提供 `center_pixel`、`box_mean`、`box_median`、`center_region_median` 对照；
当前检测器无分割掩膜，因此未实现掩膜中位数。
方向以框中心在画面中的水平位置分成左／前／右，边界归前方；配置示例假定前向未镜像输入。
镜像输入设 `mirrored=true`；摄像头安装未知或非前向时设 `camera_orientation=unknown`，方向返回 null。
这些是图像相对方向，不能解释为经过标定的世界方向、可通行方向或安全判断。

详见 [融合算法、字段和实验限制](docs/fusion.md)。离线结果没有采集时间，
`capture_timestamp_ms=null`、`current_scene=false`，不能直接用于当前场景回答。

## 确定性风险引擎

命令行 JSON 新增 `safety`，包含逐目标评估、规则配置和独立计时。
离线图片始终标记 `offline_not_current`，不会触发实时告警；相对深度不能用于米制阈值。
当前距离来源未经米制支持，因此不会把真实图片中的相对深度数值当作米数。

`configs/system.ini` 的 `[risk]` 配置距离、检测分数、时效和冷却参数。
默认 1 m／2 m **仅为合成实验阈值，未经实际行走验证**。
核心可处理可信米制适配器提供的证据；本阶段只用合成输入验证该分支，尚未实现米制适配器。
未知方向、缺失或无效距离、低分数、未覆盖类别和空检测都不能解释为安全。
告警事件与 Agent 无依赖；实时入口已接入 Windows 本地 TTS 和优先级抢占，实际硬件效果待验证。
详见 [风险规则与验证范围](docs/risk_engine.md)。

## 实时摄像头与本地语音

在项目根目录运行（启动后将打开摄像头并尝试播放中文提示）：

```powershell
D:\python\python.exe -m app.live --max-frames 100
```

当前完整环境为 `D:\python\python.exe`；使用其他解释器需先准备相同依赖。
PyCharm 设置模块 `app.live`、工作目录 `$PROJECT_DIR$`，参数 `--max-frames 100`。
省略 `--max-frames` 持续运行，Ctrl+C 退出；`--no-speech` 用于静音诊断。
确认 `[camera]` 设备索引及 `[fusion]` 安装朝向后运行。未知安装应设为 unknown。
详细故障状态、语音要求和未验证限制见 [实时运行说明](docs/live_pipeline.md)。

## Agent 场景工具

`SceneTools` 提供 `get_scene`、`find_object`、`get_object_distance`、
`get_current_risks`、`describe_surroundings`，每次查询重新检查时效。
多目标匹配要求明确选择，距离查询必须同时提供帧 ID 与目标 ID；无有效米制证据时返回 null。
实时告警先独立提交，之后才发布查询场景；相机故障和退出清空当前场景。
目前是本地工具 API，未接入外部 LLM 或 STT。
接口与后续接入边界见 [Agent 工具说明](docs/agent_tools.md)。
`VisionAssistant.respond(text)` 提供确定性文字接口，支持查询环境、风险和目标。
回答草稿输出前重新核对事实与时效；已提供 Windows 文字控制台和可选普通回答 TTS。
显式运行 `D:\python\python.exe -m app.live --console --no-speech` 可在摄像头运行时输入请求。
该命令会开启真实摄像头；输入 `quit` 退出，控制台模式隐藏逐帧 JSON。
需要普通回答朗读时显式运行 `D:\python\python.exe -m app.live --console --speak-replies`。
该模式开启声音，使用低优先级队列；播放前拒绝已过期或已换帧的回答，告警仍可抢占。
支持语法、嵌入方式与限制见 [本地交互说明](docs/local_assistant.md)。

## 验证

已下载的公开测试图片及来源、许可、运行示例见 [测试图片说明](data/test_images.md)。
图片保存在被 Git 忽略的 `data/raw/test_images/`，不随代码提交。

```powershell
python -m unittest discover -s tests -v
python -m app.main --help
```

检测核心单元测试只需要 NumPy，无需模型、摄像头、网络或 GPU。
可视化测试还需要 OpenCV，缺少时会显式跳过。
覆盖边界裁剪、无效输出、置信度阈值、设备选择、配置、失败路径以及全函数注解检查。
开发环境已有 Ruff 时，可执行 `python -m ruff check .` 和 `python -m ruff format --check .`。
合成测试不验证模型精度或现实安全性；尚未测量真实精度、FPS、显存或内存占用。

## 后续里程碑

1. 准备权重与本地测试图片，验证 YOLO11 类别覆盖与实测延迟。
2. 已接入 Depth Anything V2；继续在更多场景检查相对深度表现并测量稳态延迟。
3. 已实现同帧空间融合、方向与实时场景时效检查；米制标定仍待实现。
4. 已实现确定性风险引擎、实时入口与本地语音调度；下一步进行显式硬件验证与延迟测量。
5. 已实现受时效约束的场景工具；后续接入 Agent 交互与 STT，复用本地优先语音通路。

架构边界与实验要求见 [docs/architecture.md](docs/architecture.md) 和中英文 AGENTS 规范。
