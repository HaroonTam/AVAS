# 当前场景存储与 Agent 工具

本阶段实现本地、确定性的工具层，尚未接入 LLM；受限 STT 已作为独立输入层接入。
另有有限命令解析及确定性回答层，见 [本地文字交互](local_assistant.md)。
不新增依赖，不联网，不上传或保存相机图像、麦克风音频。

## 数据流与保留

`app.scene_store.SceneStore` 只在进程内保留最新一帧 `RiskScene` 的不可变副本。
配置复用 `[risk] freshness_ms`，以原始采集 Unix 毫秒计算时效，发布不会重置采集时间。
发布时记录剩余有效时间的单调时钟期限；查询同时检查墙上时间与单调期限。
有效期边界本身允许读取，超出边界、未来时间、缺失时间、离线和显式失效场景均不返回事实。
读取发现失效后清除存储引用，墙上时钟回拨不会使该存储中的旧帧重新可读。
较旧的有效发布不能覆盖存储中的较新帧；非法发布清空旧事实并抛出 `ValueError`。

`app.live.main(..., scene_store=store)` 支持嵌入式共享存储，必须使用与实时配置相同的
`RiskConfig`。未传入时入口自行创建存储。实时循环先提交独立告警，再发布查询场景；
发布校验失败会记录错误并继续下一帧，最终退出码为 2。
相机故障和退出会使存储失效。工具查询不持锁执行风险计算，不让发布端等待查询评估。
没有新增持久化存储；清理是删除内存引用，不保证已经交给调用方的结果副本被远程撤销。

## 工具接口

`app.agent.tools.SceneTools(store)` 提供以下只读接口：

| 方法 | 结果与约束 |
| --- | --- |
| `get_scene()` | 当前帧对象、采集时间、风险等级与事件；失效时对象和事件为空 |
| `find_object(label, direction=None)` | 按类别及可选图像方向 `left/front/right` 精确匹配；未知方向不满足定向查询，多个结果仍为 `ambiguous` |
| `get_object_distance(frame_id, object_id)` | 同时核对帧与 ID；帧改变返回 `frame_mismatch`，不自动选择新目标 |
| `get_current_risks()` | 保留引擎全部对象风险、未知原因、危险和降级事件 |
| `describe_surroundings(limit=3)` | 最多展开 1–10 个对象，优先危险、已知近距离和前方；同类危险提示合并 |

结果为不可变 `ToolResult`，可用 `json.dumps(asdict(result), allow_nan=False)` 序列化。
`risk_level` 和 `events` 始终描述整帧，即使 `objects` 因查找或摘要被筛选。
非法参数抛出 `ValueError`；无匹配不是目标不存在，空检测不是安全。
查找类别不执行指令、不模糊猜测、不把中文名称擅自映射为模型不支持的类别。

只有 `MetricEvidence.is_usable()` 确认可用时才输出 `distance_m`，同时返回支持记录 ID。
缺失米制支持为 `metric_unavailable`，无效证据为 `metric_invalid`，两者米数均为 null。
本工具层不能验证支持记录真实性：发布端必须是可信米制适配器，禁止让 LLM 构造证据。
当前真实感知没有米制适配器，所有真实距离仍不可用。
方向为上游图像相对方向，缺失时为 null；不推断可通行方向。

## 嵌入方式与后续边界

```python
from pathlib import Path

from app.agent.tools import SceneTools
from app.safety.config import load_risk_config
from app.scene_store import SceneStore

store = SceneStore(load_risk_config(Path("configs/system.ini")))
tools = SceneTools(store)
result = tools.get_scene()  # 尚未发布实时观测，返回 scene_unavailable
```

嵌入方显式运行实时入口并传入同一个 store 后，独立交互线程才能查询实时事实。
此示例本身不会开启设备；实时入口可用 `--console` 显式启用 Windows 文字提问，
尚未提供工具服务器。
不要将 `publish`、`invalidate` 或规则配置注册成 LLM 工具。
未来接入 LLM 时必须在最终回答或播放前重新核对当前帧和时效；
不能把此前工具结果无限期缓存，也不能直接播放未经事实校验的生成文本。
当前本地回答层已实现输出前重新查询和一致性校验；普通回答可显式接入低优先级队列，
语音后端启动前检查对应场景版本凭据，详见本地文字交互说明。
目前不存在任意 LLM 输出的事实校验器，不能宣称已实现完整安全 Agent。

## 验证范围

合成测试覆盖五个接口、时效边界、双时钟、失效清理、序列化、无效米制证据、
多目标选择、帧内身份、危险与未知保留、参数验证及指令形状文本。
实时测试验证先告警后发布、退出清理、发布故障不停止后续告警；
并发测试验证查询评估期间失效不被阻塞且旧结果被拒绝。
未测量真实精度、FPS、内存或语音延迟；没有启用真实硬件。
