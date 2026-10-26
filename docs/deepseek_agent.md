# E5：DeepSeek 云端语言编排

2026-10-26，用户确认选择 DeepSeek V4.1 Flash，并允许调用 DeepSeek 官方 API，
仅发送结构化场景和解析后的请求意图；禁止原图、原始音频出网，允许必要依赖。
本实现使用 Python 标准库，无新增依赖、安装或模型下载。

## 接入范围与隐私

- 官方 HTTPS 地址固定为 `https://api.deepseek.com/chat/completions`，模型参数为
  `deepseek-flash`。不读取自定义端点或代理环境变量，不跟随 HTTP 重定向。
- 只有实时入口显式指定 `--console --agent-config configs/deepseek.ini`，且配置中
  `enabled` 与 `allow_network` 同时为真，才启用云端编排。默认实时命令仍使用本地交互。
- 密钥仅读取进程环境变量 `DEEPSEEK_API_KEY`；不自动读取 `.env`。
  密钥不放入命令行、配置、日志、测试夹具或提交。配置中不需要填写密钥。
- 请求含固定系统说明，以及白名单 JSON：已解析的意图名称、目标类别、方向筛选或
  帧内身份；当前工具结果的状态、帧 ID、采集时间、风险等级和至多三个目标。
  每个目标只含 ID、类别、置信度、图像方向、有效米制距离或 null、距离状态、风险等级。
- 原始用户问句、STT 转录原文、图像、音频、深度图、模型张量、标定记录路径、
  `metric_support_id` 和风险消息正文不发送。没有多轮历史、文件上传或联网搜索。
  本地工具继续检查米制证据，云端不能提供或创建 `MetricEvidence`。
- 应用不持久保存请求或原始响应；日志仅记录固定后端状态。云服务自身的数据保留政策
  需按实际使用条款确认，此实现不能控制服务商侧保留。

## 事实约束与控制流

第一版保留既有有限命令解析，如“寻找 椅子”“前方有什么”“距离 帧ID 目标ID”。
它没有实现开放式闲聊或任意自然语言意图解析。帮助、当前风险、未找到、歧义、
空检测和失效场景继续本地处理，不发送给模型。

模型只可选择已有目标子句的两种本地表达模板：`identity_first`（先目标身份）或
`direction_first`（先图像方向）。例如，**以下是合成示例，不是当前观测**：

```json
{"frame_id":"synthetic:1","styles":[{"object_id":1,"focus":"direction_first"}]}
```

本地校验要求帧身份匹配、每个目标恰好出现一次、字段严格匹配且枚举合法。
自由文本、新目标、新距离、额外字段、重复键、重复 ID、缺失目标、截断、拒绝或空响应
均退回本地模板。即便 JSON 有效，也不能授权新事实。最终措辞的全部场景事实来自本地；
目标显示顺序、风险前缀、未知项说明、候选消歧和道路安全限制不交由模型修改。
这使模型的语言自由度较小，是本版本有意限定的范围。

流程为：本地解析 → 当前场景查询与复核 → 独立子进程 HTTPS → 有限表达计划校验 →
再次复核同一场景 → 本地模板 → 普通回答语音适配。场景变化或过期时拒绝旧计划，
不改成用旧事实回答。原场景凭据仍在语音播放前检查；任何新发布也会撤销旧凭据。

HTTP 请求在独立进程中运行，默认总预算 2 秒，包含启动、DNS、连接及响应；
无自动重试。控制台查询线程不占用感知循环或语音工作线程。用户可在等待时输入
`取消`／`cancel` 或 `退出`／`quit`；超时和取消会终止并有界回收请求进程，迟到结果丢弃。
请求体上限 16 KiB，HTTP 响应上限默认 64 KiB，模型输出上限默认 512 token。

紧急风险判断与 `dispatch_warnings` 继续独立执行，语音优先级不受模型输出影响。
密钥缺失、超时、HTTP 错误或非法计划恢复本地回答；如果等待期间场景已失效，
只能输出重新查询的固定提示。日志中 `accepted` 只表示模型计划校验通过，
不代表回答仍新鲜或声音已实际播出。

## 配置与用户执行命令

`configs/deepseek.ini` 是单独的、显式启用配置；生产 `configs/system.ini` 未改变。
`timeout_s` 可在 0.1–10 秒内配置，但这不会延长场景期限。
`allow_network=false` 或 `enabled=false` 会使用本地交互。
没有 `--agent-config` 时不读取此配置、不启动网络子进程。

先在用户自己的 PowerShell 终端进入主工作区，以隐藏输入方式设置本次会话密钥；
不要粘贴密钥到聊天或写进脚本：

```powershell
Set-Location -LiteralPath 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'
$deepseekSecret = Read-Host 'DeepSeek API Key（隐藏输入）' -AsSecureString
$env:DEEPSEEK_API_KEY = [System.Net.NetworkCredential]::new('', $deepseekSecret).Password
Remove-Variable deepseekSecret
```

第一次可仅做合成连通检查（会产生一次 API 请求及可能的费用，不打开任何设备）：

```powershell
D:\python\python.exe -m app.agent.deepseek_check --agent-config configs/deepseek.ini --send-synthetic
$LASTEXITCODE
```

请贴回 JSON 和退出码，勿贴密钥。`synthetic_only=true` 表明仅使用合成数据；
`backend_status=accepted`／退出 0 表示一次编排成功。`answer_status` 与 `fresh_answer`
另行报告回答是否还有效：即使 API 成功，也可能 `answer_expired`／`false`。
这不验证真实场景精度、真实对话质量或语音效果。

后端状态含义：

| 状态 | 含义与处理 |
| --- | --- |
| `disabled` | 配置开关未同时开启；检查独立 Agent 配置 |
| `missing_key` | 当前进程未获得非空 `DEEPSEEK_API_KEY` |
| `timeout` | 总预算耗尽；检查连接情况，保留原场景新鲜度 |
| `cancelled` | 用户取消或程序关闭，结果丢弃 |
| `unavailable_or_invalid` | 网络、鉴权、额度、响应大小或格式校验失败；检查账户与连接，不能贴回秘密或原始响应 |
| `accepted` | 有限计划通过校验，仍需回答时效及语音凭据检查 |

用户准备进行真实设备交互时再执行：

```powershell
D:\python\python.exe -m app.live --config configs/system.ini --console --speak-replies --agent-config configs/deepseek.ini
```

该命令打开摄像头并启用扬声器。需要语音输入时再显式加 `--voice-input`，
输入“听取”后开启单次本地识别；仅解析后的命令意图会发送给云端。
既有摄像头启动等待问题应沿用 E1 已确认的配置，不在 E5 中修改等待或告警参数。
会话使用完毕可清除密钥：

```powershell
Remove-Item Env:\DEEPSEEK_API_KEY -ErrorAction SilentlyContinue
```

## 验证与限制

合成测试覆盖白名单出网投影、两开关和密钥缺失、事实及风险保留、未知方向和距离、
多目标消歧、空检测、非法输出、重复 JSON 键、超时、取消、场景改变与过期、
固定 HTTPS 目的地、HTTP 错误和重定向拒绝、响应大小、子进程回收、实时配置接线，
以及云端停滞或返回误导内容时本地告警仍派发并抢占模拟普通语音。
没有测试需要真实网络、密钥、模型、GPU、摄像头或音频设备。

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1 -Python 'D:\python\python.exe'
```

未验证：真实 DeepSeek API 对话效果、网络响应时间、费用、速率限制下的表现、
模型版本漂移，以及接入后的真实语音与可听抢占效果。`deepseek-flash` 是服务别名，
不是固定权重快照；官方文档查询日为 2026-10-26。

生产新鲜度默认 1000 ms，而且每次场景发布都会撤销上一版凭据。
网络请求很可能跨越帧更新；本版本会拒绝回答，实时问答成功率尚未测量。
不能以调大排队期限或场景新鲜度来掩盖该限制，后续需根据真实测试另行设计。
本接入没有测量感知准确率、FPS、内存、米制距离误差或可听告警延迟；
`capture_to_dispatch`、后端 `accepted` 和语音入队状态均不是可听告警延迟。
合成测试不代表真实环境安全性。

API 参数依据：[官方接入](https://api-docs.deepseek.com/quick_start/agent_integrations/)、
[JSON 输出](https://api-docs.deepseek.com/guides/json_mode/)、
[思考模式开关](https://api-docs.deepseek.com/guides/thinking_mode/)。
使用 `json_object` 并进行本地严格校验，没有假设服务支持严格 JSON Schema。
