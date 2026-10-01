# 离线目标与深度融合

## 输入与坐标约定

`DetectionFrame` 将检测绑定到明确的 `frame_id`、原图宽高；`RelativeDepth` 必须具有相同身份，
深度和布尔掩膜形状必须为 `(height, width)`。错误身份、尺寸、数据类型均拒绝融合，空检测也验证。
框使用半开区间 `[x1,y1,x2,y2]`；检测解析器负责裁剪，融合拒绝超出原图的框，不悄悄修复坐标错位。
当前 YOLO11 已还原原图框，深度适配器已插值回原图网格；融合不再缩放或翻转深度数组。
两个分支必须使用同一解码图像，不允许调用方把裁剪、填充或镜像后的另一图像冒充同一原图。
同身份、同尺寸校验不能检测调用方错误标记的坐标变换。

CLI 用解码图像字节的 SHA256 作为离线输入身份。它不是摄像头序号或跨帧跟踪标识。
对象 ID 从 1 开始，只在该帧有效。`capture_timestamp_ms=null` 表示没有真实采集时间，
`current_scene=false` 表示离线结果不能作为当前环境事实；本阶段不提供实时场景缓存或 Agent 工具。
未来摄像头接入时必须另外实现采集时间、过期拒绝与场景失效。

## 统计方法与质量

| method | 采样区域和统计 |
| --- | --- |
| `center_region_median` | 默认；中心宽高各 `inner_fraction=0.5`，内边界向内取整，取中位数 |
| `box_median` | 整个有效框内中位数 |
| `box_mean` | 整个有效框内均值 |
| `center_pixel` | `((x1+x2)//2, (y1+y2)//2)`；偶数宽高取偏右下像素，仅作实验基线 |

没有分割模型输入，故不支持分割掩膜中位数；未知方法直接拒绝。
所有方法使用原始浮点相对逆深度，绝不使用 PNG 预览或其归一化值。
先按有效掩膜过滤，并再次排除 NaN、Inf 和负值；零沿用深度适配器的有效语义。
在剩余像素上计算 `trim_fraction` 与 `1-trim_fraction` 分位数，只保留闭区间内的值。
默认 `trim_fraction=0.05`；阈值重复时可能保留更多像素。设为 0 可比较未裁尾基线。
全部方法共用相同过滤策略；这是一种可配置统计启发式，尚未证实其精度优于其他方法。

`sample_count` 为区域像素数；`valid_count` 为裁尾前有效像素数，`valid_fraction` 为其占比；
`retained_count` 为裁尾后数量。默认裁尾后至少 9 像素，裁尾前有效比例至少 0.5（等于边界可用）。
中心像素基线明确使用 1 像素最低数量，其余质量条件不变。
可用时输出 `relative_depth` 和裁尾后四分位距 `relative_depth_iqr`，否则二者为 null。
数值越大表示相对越近，无量纲，不能跨图像直接比较，也不代表物体表面的最近点。
内部框仍可能包含背景、遮挡或细物体外部区域；有效像素多、离散度小不证明距离准确。

状态包括 `available`、`depth_unavailable`、`empty_region`、`insufficient_valid_pixels`。
微小框内缩后可能为空，此时不擅自退回单像素。保留检测置信度，与深度统计质量明确分开。
尚未实现标定，始终输出 `calibration_status=not_calibrated`、`distance_m=null`、
`distance_status=metric_unavailable`、`risk_level=unknown`。

## 方向与安装

配置 `[fusion]` 的边界满足 `0 < left_boundary < right_boundary < 1`，默认 1/3、2/3。
用框中心横坐标除以原图宽度；小于左边界为 left，大于右边界为 right，其余为 front。
边界本身归 front。`mirrored=true` 先反转水平位置，深度区域仍按原始输入网格取值。
配置样例 `camera_orientation=forward` 表示人为约定的前向、水平正置视角，
不是自动识别出的相机安装事实。`forward` 仅适用于该约定；非前向、旋转或未知安装应设为 `unknown`。
未知时 `direction=null`、`direction_status=camera_orientation_unknown`。
可用方向标记 `image_relative`；front 只是画面中部，不能推导安全通行方向或世界方位。
旧 INI 缺少融合段时采用保守默认值，安装朝向 unknown。

## 故障与验证

融合本身拒绝错帧或错尺寸深度。CLI 捕获此失败并丢弃该深度，保留有效检测和方向，
`fusion.status=depth_rejected`、`depth.status=rejected`，返回退出码 2。
正常融合流程的 `fusion.status=available` 只表示流程完成，单个目标仍可能深度不可用。
深度关闭或模型故障时，对象深度为 `depth_unavailable`。空目标列表不表示道路安全。

运行 `python -m unittest discover -s tests -v`。合成测试覆盖各统计方法、异常像素、裁尾、
有效比例、微小框、图像边界、错帧错尺寸、方向边界及镜像、未知安装、旧配置与 CLI 降级。
`fusion.latency_ms` 包括构建检测帧、同帧校验、方向和目标统计，不含模型、JSON 序列化或保存。
真实图片功能验证仅检查链路能否运行；未测量精度、稳态 FPS、内存或实际行走安全性。

### 2026-10-01 本地功能验证

使用已有 `street_people_cars.jpg` 和本地 YOLO11n、Depth Anything V2 Small，CUDA 执行一次离线调用。
输出保存在 `outputs/fusion_20261001_120245/`：`result.json`、`environment.json` 和运行日志，
目录被 Git 忽略，未覆盖已有实验。环境记录包含配置、源码和两份权重 SHA256、输入图片 SHA256、
Python／依赖版本及设备；深度固定版本记录在结果中，测试图片来源见 `data/test_images.md`。

| 检查 | 结果 |
| --- | --- |
| 检测／融合目标数 | 13／13 |
| 相对深度统计可用 | 13 |
| 左／前／右 | 2／8／3 |
| 米制距离／风险 | 全部 null／unknown |
| 单次检测调用 | 532.41 ms |
| 单次深度调用 | 122.65 ms |
| 单次融合 | 1.86 ms |

这是模型已加载后的首次调用计时，不含模型加载；不与前次记录作性能优劣比较。
未做人工深度真值、方向标注或稳态重复测量，因此不报告精度、FPS 或内存改善。
本次完整环境解释器为 `D:\python\python.exe`；系统 Anaconda 与仓库 `.venv` 未安装完整可选依赖。
