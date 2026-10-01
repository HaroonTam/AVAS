# 本地功能测试图片

这组图片用于 YOLO11 单张图片入口试跑和后续深度图可视检查。
图片来自 Pexels，适用 [Pexels License](https://www.pexels.com/license/)，
允许免费下载和使用，具体限制以许可页面为准；不得将未修改图片作为独立商品出售，
不得暗示图中人物或品牌为项目背书。

图片目录：`data/raw/test_images/`（已被 Git 忽略）。
下载日期：2026-09-30。请求的是 CDN 宽度上限 1280 像素的压缩 JPEG 版本，
自行车图片实际宽度为 1276 像素，其余为 1280 像素。
保持宽高比，不裁剪；不是原始全分辨率文件。
实际尺寸、文件大小、SHA256 与下载地址见 `data/test_images_manifest.json`。

| 文件名 | 场景用途（不是模型结果或人工标注） | 作者 | 来源 |
| --- | --- | --- | --- |
| `street_people_cars.jpg` | 城市街道中的行人与车辆 | Hugo Sykes | [Pexels 27666752](https://www.pexels.com/photo/a-city-street-with-people-walking-and-cars-driving-27666752/) |
| `bicycle.jpg` | 自行车与街边背景 | Joe Chen | [Pexels 327698](https://www.pexels.com/photo/bicycle-parked-on-the-street-327698/) |
| `indoor_chairs.jpg` | 室内桌椅与遮挡 | Max Vakhtbovych | [Pexels 6434622](https://www.pexels.com/photo/a-table-and-a-chair-in-a-room-6434622/) |
| `sidewalk_traffic.jpg` | 人行道、行人与道路车辆 | Darya Egorova | [Pexels 9999588](https://www.pexels.com/photo/people-walking-on-the-sidewalk-near-the-buildings-and-moving-cars-on-the-road-9999588/) |

在项目根目录运行（需要先按 README 准备权重）：

```powershell
python -m app.main --image data/raw/test_images/street_people_cars.jpg
python -m app.main --image data/raw/test_images/bicycle.jpg
python -m app.main --image data/raw/test_images/indoor_chairs.jpg
python -m app.main --image data/raw/test_images/sidewalk_traffic.jpg
```

这些图片没有目标框标注、米制距离真值或风险真值，不用于报告 mAP、测距误差或安全性。
图片间不存在连续帧或同一摄像机的保证，不用于运动估计。
不纳入训练或正式评估划分，仅作为人工检查样例。
