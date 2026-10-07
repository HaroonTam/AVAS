# 检测精度评估：COCO val2017（任务 C）

## 当前结论（2026-10-08，尚未运行）

评估入口与指标实现已完成并通过仓库校验（`app/accuracy_metrics.py`、
`app/detection_accuracy.py`、18 个合成用例，见提交 `1ca57a2`），但**尚未在 COCO
val2017 上运行**：数据由用户在终端下载到 `data/raw/coco/`（目录已被忽略）。
因此本文的 mAP 一类数字全部记为**未测量**，不得用任何其它来源的数字代填。

## 评估对象与方法（预登记，运行前固定）

- 权重：`configs/system.ini` 的 `[detection].weights`（当前 `yolo11n.pt`），输入尺寸与
  生产一致（`image_size = 640`），设备按配置解析；**COCO 预训练权重、不微调**。
- 数据：COCO val2017（5000 张）+ `instances_val2017.json`。val2017 是 train2017 之外的
  标准留出集；权重的类别名必须与标注类别名一一对应，缺名即报错退出。
- 两个阈值分开报告：导出用 `--eval-confidence 0.001`（为画 PR 曲线）；另报生产工作点
  `confidence = 0.35` 的精确率／召回率／F1 与 TP/FP/FN。两者口径不混用。
- 指标：AP@[.5:.95]、AP50、AP75、APs／APm／APl（按真值框面积与 32²、96² 划分）、
  AR@1／AR@10／AR@100；COCO 101 点插值；每图最多 100 个预测。
  无真值的类别记为 `null`（未测量）而不是 0，并列出其 `category_id`。
- 实现：`app/accuracy_metrics.py`（纯指标与匹配，可用合成数据单测）+
  `app/detection_accuracy.py`（独立入口，不 import `app.main`／`app.live`）。
  面积区间以外的真值按 COCO 的**忽略**语义处理：与之重叠的预测既不记 TP 也不记 FP，
  也不进入召回分母；crowd 真值同理。
- 延迟：单次前向 `detect()` 的墙钟毫秒，图片解码与哈希不计入；这是单次遍历统计，
  与 `app.benchmark` 的重复采样协议不同，报告中分列。
- 复现：结果 JSON 记录权重与标注 SHA256、配置名、实际设备与 CUDA 名称、包版本、
  Git 修订与 `app/` 源码哈希；输出路径**独占**，拒绝覆盖任何既有结果。
- 数据与许可：COCO 标注为 CC BY 4.0，图片各自保留 Flickr 许可；数据只落
  `data/raw/`（忽略），不提交数据集或权重。

## 局限（必须随结果一起报告）

- 只评估 COCO 预训练权重，未在本项目数据上微调；结论只适用于 COCO val2017 分布与
  640 输入，**不能**外推到室内视障场景或本机摄像头。
- COCO 类别不含 `door`、`stairs`，不得声称支持门与台阶识别。
- 未测量米制距离误差、告警延迟与摄像头帧率；单模型单次运行，未做跨设备对照。
- 4 张 Pexels 样例图没有目标框标注，仍只作功能样例，不用于 mAP
  （见 [测试图片说明](../data/test_images.md)）。

## 运行方式（数据就绪后）

```powershell
$env:YOLO_CONFIG_DIR=(Get-Location).Path
$env:HF_HUB_OFFLINE='1'
D:\python\python.exe -B -m app.detection_accuracy --annotations data/raw/coco/annotations/instances_val2017.json --images-dir data/raw/coco/val2017 --output outputs/detection_accuracy_val2017_20261008_run01.json
```

## 待填结果

数据到位并跑完后补：AP／AP50／AP75、APs／APm／APl、AR@1/10/100、工作点 P/R/F1 与
TP/FP/FN、检测延迟 mean／p50／p95、逐类 AP 摘要、漏检与误检的最差图像清单及其原因分析，
以及实际使用的权重与标注 SHA256。
