# 无硬件合成演示

在项目根目录运行：

```powershell
D:\python\python.exe -m app.demo
```

不打开摄像头，不加载 YOLO11 或 Depth Anything V2 权重，不播放声音，不调用网络。
只使用基础 Python 依赖和 NumPy；没有新增安装步骤。
输入是固定的合成 Detection、相对深度数组和单独声明的米制测试夹具。
真实执行的模块是融合、场景存储、风险规则、Agent 查询和语音队列调度。
语音后端只把待播文字保存在本次进程内存中，不启动 PowerShell。
此外执行一段固定控制台脚本，使用合成识别器替代录音设备；真实运行控制台、
识别结果白名单解析、Agent 与普通语音适配器。无需交互式终端或人工输入。

| 用例 | 验收内容 |
| --- | --- |
| `relative_depth_is_not_meters` | 合成相对逆深度 7 保持非米制，融合得到图像前方，风险未知 |
| `ordinary_reply_simulated` | 普通回答经过场景凭据检查进入记录后端，不实际发声 |
| `multiple_matches` | 两个椅子返回歧义状态，需要明确选择 |
| `independent_warning` | 人工注入 0.5 米测试证据，风险引擎先提交高风险告警，再查询 Agent |
| `expired_scene` | 推进合成时钟至有效期之后，不再返回当前事实 |
| `invalid_scene` | 显式失效后拒绝历史对象查询 |

交互流程使用前方、左侧各一把椅子的独立合成发布，所有米制距离未知：
取消首次听取 → 低分数识别被拒绝 → “前方有什么” → “寻找椅子” →
“左侧的椅子在哪里” → 键盘退出。首次识别器在取消后故意返回成功，以检查迟到结果被丢弃。

| 交互检查 | 验收内容 |
| --- | --- |
| `cancel_then_retry` | 取消后的汽车查询不执行，仍可再次听取 |
| `low_confidence_feedback` | 低分数结果不查询目标，固定失败反馈进入模拟语音 |
| `direction_summary_and_retry` | 前方摘要及左侧重查分别只展开对应目标，距离保持不可用 |
| `candidate_directions` | 多目标回答显示两个候选的真实合成方向，并提示重新按方向寻找 |
| `simulated_speech_and_shutdown` | 两条固定反馈和三条事实回答均被记录，退出与资源关闭无故障 |

脚本只替换设备边界，不安装或启动 Windows STT 后端。演示使用识别协议
`CommandRecognizer` 注入合成实现，生产环境仍使用原有 `WindowsCommandRecognizer`。

米制夹具的 `support_id=synthetic-demo-only`，不是校准结果，
0.5 米绝非从相对深度 7 推导。场景内部的 `live` 仅用于测试时效协议；
数据被限定在演示自己的局部存储中，不会发布到真实运行的场景存储。
所有报告均标记为 synthetic，不能作为当前环境观测或模型实验结果使用。

## 机器可读结果

```powershell
D:\python\python.exe -m app.demo --json
```

stdout 仅输出一个 JSON 文档，包含 schema_version、synthetic、配置、
各用例结果、模拟语音文字与总体 passed。提示及风险日志写入 stderr。
当前 `schema_version=2` 保留原有六个 `cases`，新增 `interaction`，包含 synthetic、checks、
transcript、simulated_speech 和 passed。顶层 simulated_speech 包含全部七条模拟语音；
interaction 内只包含本段五条。任意交互检查失败也使总体 passed 为 false。
全部通过返回 0，用例失败返回 1；意外异常同样返回非零并保留异常诊断。
程序不自动保存文件；如自行重定向输出，请使用新的文件名，避免覆盖已有实验记录。

合成墙上时间与单调时间同步推进，报告不混入实际运行耗时。
真实语音队列的有界完成等待最多 2 秒，仅用于防止演示卡住，不能解释为实测告警延迟。
演示的每次交互语音提交后等待记录后端完成，以稳定输出顺序；整个控制台脚本最多等待
15 秒，再调用有界关闭。该等待只在合成演示中使用，不添加到生产感知循环。
重复正常执行应产生相同的语义报告。

## 验证范围

新增测试禁止摄像头、模型初始化、系统语音与子进程入口，检查 JSON 输出、
重复结果一致性，以及模拟语音后端失败时的退出码。
另验证识别脚本故障即使不影响原有场景用例，也必须使总体验收失败。
此演示不验证检测准确率、真实深度质量、音量、可听延迟、FPS 或行走安全。
真实硬件运行仍使用 `app.live`，由用户明确启动。
