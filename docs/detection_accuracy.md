# 检测精度评估：COCO val2017（任务 C）

## 当前结论（2026-10-08，val2017 全量 5000 张已测）

在 COCO val2017 全量 5000 张图上评估 `models/yolo11n.pt`（COCO 预训练权重、未微调）：

| 指标 | 值 |
| --- | ---: |
| AP@[.5:.95] | 0.379 |
| AP50 | 0.534 |
| AP75 | 0.412 |
| AP small | 0.061 |
| AP medium | 0.277 |
| AP large | 0.508 |
| AR@1 | 0.308 |
| AR@10 | 0.504 |
| AR@100 | 0.553 |

生产工作点（`confidence = 0.35`、IoU 0.5）：精确率 **0.790**、召回率 **0.451**、F1 **0.574**
（TP 16,380／FP 4,342／FN 19,955）。单张 `detect()` 耗时 mean **12.36 ms**、p50 11.04、
p95 19.61 ms（最大值 624 ms 属个别离群，原因未查明，不计入 p95 之外的解释）。

**这些数字只描述 COCO 预训练权重在 COCO val2017 上的表现**，不是本系统在室内视障场景的
精度，也不能据此声称支持「门」「台阶」——COCO 类别里没有这两类。

## 评估对象与方法（运行前预登记，实际执行与之一致）

- 权重：`configs/system.ini` 的 `[detection].weights`（`models/yolo11n.pt`，
  sha256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`），
  输入尺寸与生产一致（640），设备 `auto` → RTX 4060 Laptop（CUDA）。
- 数据：COCO val2017 全量 5000 张 + `instances_val2017.json`
  （sha256 `e8c7f7908f1d7278341fae127d0da654f102f11bd7b21d8aeefa635b8c810b6f`，
  80 类、36,781 个标注实例、其中 446 个 crowd）。权重类别名与标注类别名**一一对应（0 缺失）**。
- 两个阈值分开报告：导出用 `--eval-confidence 0.001`（画 PR 曲线，共 652,959 条预测行），
  另报生产工作点 0.35 的口径。两者不混用。
- 指标：AP@[.5:.95]、AP50、AP75、APs／APm／APl、AR@1／AR@10／AR@100；COCO 101 点插值；
  `maxDets = 100` 按 COCO 语义在**每张图每个类别**内生效；面积区间以外的真值与 crowd 真值
  都按**忽略**处理（与之重叠的预测既不记 TP 也不记 FP，也不进召回分母）。
- 无真值的类别记为 `null` 并列出其 `category_id`（本次 80 类全部有真值，故无此类）。
- 复现：结果 JSON 记录权重与标注 SHA256、配置名、实际设备与 CUDA 名称、包版本、
  Python／平台、Git 修订与 `app/` 源码哈希；输出路径独占，不覆盖既有结果。

## 数据来源与完整性核对

- 官方两包经 **HTTP** 下载（`https://images.cocodataset.org` 的 TLS 校验在本机被 DNS
  代理接管，证书名不匹配，HTTPS 无法建立）：`val2017.zip` 815,585,330 字节（≈778 MiB）、
  `annotations_trainval2017.zip` 252,907,541 字节，均与公开记录的归档大小一致，ZIP 魔数正常。
- 解压后与标注逐张交叉核对：**5000/5000 图像齐全、图像尺寸与标注元数据 0 处不符**。
- 数据只落 `data/raw/coco/`（Git 忽略），不提交数据集或权重；COCO 标注为 CC BY 4.0，
  图片各自保留 Flickr 许可。

## 指标口径修正（如实记录）

首轮运行（`outputs/detection_accuracy_val2017_20261008_run01.json`）得到 AP 0.247／
AP50 0.318，明显异常。原因是实现错误：把 COCO 的 `maxDets = 100` 当成「每个类别全局取
前 100 个预测」，而 COCO 的语义是「每张图每个类别取前 100」；前者会把多实例类别（如
person 有 10,777 个真值）的召回整段截掉。修正为按图按类截断后，同一份数据上得到
AP 0.379／AP50 0.534（本轮 `run02`）。**run01 保留但不使用**；两种口径的差别有专门用例
覆盖（`tests/test_accuracy_metrics.py::test_detection_cap_applies_per_image_not_per_class`）。

另外，导出阈值 0.001 时模型会输出零宽或零高的退化预测行（实测 40 张图、54 行，分数
0.0010–0.0057），而生产解析器对这类行是硬错误。评估入口因此注入一个受控后端，显式
丢弃并计数这些行（报告字段 `protocol.degenerate_rows_filtered = 54`），其余流程与判据不变。

## 逐类与示例（选取）

| 类别 | AP | AP50 | 真值数 |
| --- | ---: | ---: | ---: |
| person | 0.509 | 0.740 | 10,777 |
| bus | 0.624 | 0.748 | 283 |
| refrigerator | 0.564 | 0.715 | 126 |
| motorcycle | 0.421 | 0.653 | 367 |
| car | 0.359 | 0.569 | 1,918 |
| bicycle | 0.271 | 0.470 | 314 |
| chair | 0.250 | 0.404 | 1,771 |
| book | 0.096 | — | 1,129 |
| handbag | 0.087 | — | 540 |
| hair drier | 0.000 | — | 11 |

AP 最高的类别：bear 0.736、giraffe 0.689、airplane 0.680、train 0.670、cat 0.665。
最差三类（handbag／book／spoon／hair drier）都是小目标或密集遮挡类别，与 APs 仅 0.061 一致。
漏检最集中的图像是拥挤场景（例如 `000000435081.jpg`：62 个真值、仅 21 条预测、
13 命中 49 漏检），属于小目标与遮挡问题，不是类别映射错误。
**对本站而言值得注意**：`chair`（AP 0.250）与 `bicycle`（0.271）偏低，而这两类正是项目
`obstacle_labels` 里的风险目标；该结论只适用于 COCO 分布，但说明不能把「检测到椅子」当作
可靠前提，风险引擎对未知/缺失信息保持 UNKNOWN 的设计仍然必要。

## 局限（必须随结果一起报告）

- 只评估 COCO 预训练权重，未在本项目数据上微调；结论只适用于 COCO val2017 分布与 640
  输入，**不能**外推到室内视障场景或本机摄像头。
- COCO 类别不含 `door`、`stairs`，不得声称支持门与台阶识别。
- 未与第三方公开数值逐项对齐：可引用的在线来源在本机不可访问（DNS 被代理接管），
  因此不引用任何记忆中的数值；如需对照，应在能访问文档的环境重跑或补引来源。
- 小目标能力弱（APs 0.061），长尾/遮挡类别接近 0；最差图像的漏检原因未逐张人工复核。
- 延迟为单次遍历统计（含首次离群 624 ms），与 `app.benchmark` 的重复采样协议不同；
  本入口不测量米制距离误差、告警延迟与摄像头帧率（可听告警延迟另见
  [可听告警延迟](audible_latency.md)，已测量但声程未校准）；单模型单次运行，未做跨设备对照。
- 4 张 Pexels 样例图没有目标框标注，仍只作功能样例，不用于 mAP
  （见 [测试图片说明](../data/test_images.md)）。

## 运行方式（可复跑）

```powershell
$env:YOLO_CONFIG_DIR=(Get-Location).Path
$env:HF_HUB_OFFLINE='1'
D:\python\python.exe -B -m app.detection_accuracy --annotations data/raw/coco/annotations/instances_val2017.json --images-dir data/raw/coco/val2017 --output outputs/detection_accuracy_val2017_<日期>_run<序号>.json
```

说明：`--config` 默认 `configs/system.ini`；`--eval-confidence` 默认 0.001 且不得高于工作点
阈值；`--max-images` 只用于确定性子集（按文件名排序截断）；输出路径已存在时直接拒绝。
