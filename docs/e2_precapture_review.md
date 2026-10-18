# E2 预采集审查与相对深度诊断

日期：2026-10-18。21 条开发预采集，另保留 1 条试拍，尚未进行模型推理或米制拟合。

## 数据核对

| 距离档 | 条数 | 实测距离（m） |
| --- | ---: | --- |
| 0.5 | 4 | 0.495、0.500、0.505、0.519 |
| 1 | 4 | 0.900、1.000、1.005、1.010 |
| 1.5 | 4 | 1.495、1.500、1.510、1.520 |
| 2 | 4 | 1.990、2.000、2.000、2.010 |
| 3 | 5 | 2.990、3.000、3.000、3.010、3.030 |

0.900 m 已由用户确认是实际读数，不修正为 0.990 m。额外一条 3 m 样本保留。
试拍 `e2_e021c702fb1843acb93c6ffea762885a` 仅从本轮清单排除，原记录不删除。
21 条均成功解码为 640×480，哈希一致、图像哈希不重复、ROI 合法、配置快照匹配。
这些检查不证明人工读数或姿态精度；卷尺精度 0.1 cm 为用户报告，总不确定度仍未知。

全部属于 scene01、omen_h0265_v1，单房间、同一板材，不能随机拆帧充当独立测试集。
图像曝光/色彩有明显变化，人物位置也变化；原始环境文字一致不意味着受控光照完全一致。
本批仅用于诊断流程和提出正式采集方案，不支持泛化结论。

## 提手孔与 ROI 更正

用户确认黑色矩形为贯穿板面的提手孔。原始中心区域上沿在部分样本中与孔洞相交，
所以早前“黑色区域完全在统计范围外”的判断不适用于全部样本。孔洞像素不能赋予板面真值。

不改原始图像、真值或框。新增显式策略 `board_lower_half_v1`：
给定原框 `(x1,y1,x2,y2)`，诊断框为 `(x1,ceil((y1+y2)/2),x2,y2)`，
然后调用生产 `sampling_region` / `summarize_depth`，沿用 center_region_median 与全部过滤参数。
本策略是在看模型结果前因已确认孔洞定义，不按误差择优，也不宣称自动适用于其他目标。
21 张缩略图逐图检查后，绿色统计区域均位于孔洞下方可见板面；这不验证板面姿态/距离真值。
诊断框不同于生产 YOLO 检测框，结果只能标为人工 ROI 测深诊断。

本地审查证据（忽略，不提交）：

- `outputs/e2_review_20261018_v1/`：原框审查，保留用于追溯，不用作本次推理清单。
- `outputs/e2_review_20261018_v2/report.json`：冻结 21 条记录路径与哈希、原框、诊断框、统计框。
- `outputs/e2_review_20261018_v2/roi_review.jpg`：蓝色为下半板诊断框，绿色为实际统计区域。

复现纯审查（目录必须尚不存在）：

```powershell
$e2Audit = 'outputs/e2_review_' + [guid]::NewGuid().ToString('N')
D:\python\python.exe -m app.e2_offline --output $e2Audit --exclude-sample e2_e021c702fb1843acb93c6ffea762885a --board-lower-half
```

## 用户下一步：相对深度离线诊断

在含已有权重的主工作区运行，不打开相机/音频、不下载权重。仅加载一次现有固定 Small 相对模型，
直接读取本地原图；原图不裁剪送模，仅在原图网格的输出上应用上述诊断框。
清单中的记录被更改、图像哈希不符、配置不一致、重复图像或混合安装将拒绝执行。

```powershell
Set-Location 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'
$e2Relative = 'outputs/e2_relative_' + [guid]::NewGuid().ToString('N')
D:\python\python.exe -m app.e2_offline --manifest outputs/e2_review_20261018_v2/report.json --infer-relative --output $e2Relative
$LASTEXITCODE
```

贴回报告路径与退出码即可，不需上传图像。逐样本错误保留在报告中，并导致 CLI 返回 2；
模型初始化失败也返回 2，已有空输出目录不代表成功。已有目录拒绝覆盖，重试使用新目录。
输出为无量纲 relative_inverse_depth、有效像素数与 IQR；不计算米制误差、不拟合、
不产生 MetricEvidence。不可将该数值当米或据此启用距离风险阈值。

## Metric 候选审查进展

已找到官方 [Metric Indoor Small Transformers 模型卡](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf)：
面向室内、Hypersim 微调、24.8M 参数，兼容 Transformers；可与当前 Small 路线形成候选对照。
参考 [官方 Metric 代码目录](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)。
目前未下载权重；具体不可变 revision、权重哈希、文件体积、许可链与接口需继续审查。
本机准确率、延迟与内存影响未测量，不因参数规模相同就声称性能相同。
使用室内微调米制权重会改变训练域与模型身份，报告须将其与当前相对模型区分。
后续完成候选审查和显式下载步骤后，再按同图同诊断框比较；生产路线仍由用户审阅结果后决定。

## 验证

完整 `scripts/check.ps1 -Python D:\python\python.exe`：257 tests OK、Ruff 检查通过、66 文件格式通过。
离线入口 help 通过；真实本地图像纯审查通过。测试使用合成数据验证模型隔离、哈希冻结、
防覆盖、配置一致性、下半 ROI 及生产统计复用。真实模型诊断未运行，不能声称精度通过。
