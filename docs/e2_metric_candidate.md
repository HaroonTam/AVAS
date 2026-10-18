# E2 Metric Indoor Small 候选与用户运行步骤

审查日期：2026-10-18。候选只用于离线研究对照，不替换默认模型，不签发 MetricEvidence。
助手已读取官方元数据与小型配置，未下载权重、未运行模型或硬件。

## 固定身份与许可来源

- 仓库：[depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf)。
- 固定 revision：`8078d68a9c75a972131914f6afd0c1723be0da7f`；来源为官方模型 API，下载不跟随 main。
- safetensors：99,173,660 字节（约 99.17 MB），SHA256
  `e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3`，来自该仓库 LFS 元数据。
- config.json、preprocessor_config.json、README.md 已从固定版本读取并计算 SHA256，
  与权重哈希一起固定在 app/e2_metric.py；每次推理前校验全部四个文件。
- 官方 [Metric 说明](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth)
  将 Indoor 路线标为 Hypersim 微调，Small 为 24.8M 参数；输出定义为米。
- 固定配置为 `depth_estimation_type=metric`、`max_depth=20`。20 m 是模型输出设置，
  **不是本机验证范围**；本机有效支持区间仍未建立。
- [上游许可声明](https://github.com/DepthAnything/Depth-Anything-V2#license) 将 Small 归为 Apache-2.0。
  HF 转换仓库当前模型卡元数据未单列 license 字段，也没有独立 LICENSE 文件；
  本项目记录为“依上游 Small 许可声明溯源”，不伪称转换仓库单独声明了许可。
  本次不分发权重；若未来再分发，需补齐适用许可/NOTICE 和转换来源核验。

## 对本项目的影响

与当前相对模型均为 Small，但训练目标与权重身份不同，论文须作为不同候选报告。
其权重字节数与当前相对 Small 一致；这不证明延迟、内存或准确率相同。
本机 Metric 的准确率、延迟、FPS、峰值内存均未测量。暂不据此选择生产路线。

本地已安装 transformers 5.6.2、torch 2.7.0+cu118、numpy 2.2.6、Pillow 11.2.1、
safetensors 0.7.0；不新增依赖。使用 CUDA 可用性检查及 CPU/MPS 回退选择，
报告实际选用设备；不同后端的实际算子支持仍由运行验证。

沿用当前相对基线的 PIL 预处理、配置输入 518、BGR→RGB、全图推理、bicubic 原图网格对齐。
不使用可视化归一化深度图，不对输出逐图缩放，不用真值校正候选预测。
区域选择与过滤统计核复用生产实现。相对路径原有非负规则保持不变；
Metric 过滤非有限、非正和超过模型设置 20 m 的预测，不钳位为有效距离。
该过滤是模型数值边界，不代表能识别所有域外目标。

## 用户执行步骤

第一步仅下载固定候选（需要联网，约 99.18 MB 文件内容，不上传照片）：

```powershell
Set-Location 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'
D:\python\python.exe -m app.e2_metric --download
$LASTEXITCODE
```

默认保存到忽略目录 `models/depth-anything-v2-metric-indoor-small-hf/`。
逐文件验证通过才将 .part 改名；已有且哈希正确的文件跳过，已有错误文件不覆盖。
若失败，保留 .part 并贴回错误；不要将失败或部分下载视为完成，也不要盲目覆盖目录。

只有下载退出码 0 后，再执行本地推理：

```powershell
$e2Metric = 'outputs/e2_metric_' + [guid]::NewGuid().ToString('N')
D:\python\python.exe -m app.e2_metric --infer --manifest outputs/e2_review_20261018_v2/report.json --output $e2Metric
$LASTEXITCODE
```

不打开相机或音频；推理强制 local_files_only、trust_remote_code=False，使用 safetensors。
对同一冻结清单、同一原图、同一 board_lower_half_v1 诊断框运行，21 条依次处理，模型只加载一次。
报告保存预测米数、有效像素、IQR、真值、输入哈希、模型身份、设备与代码身份；
每条 `metric_evidence=null`，预测尚未本机验证，不得接入距离告警。
逐样本错误保留并导致退出码 2；旧输出目录拒绝覆盖。无效统计也保留，后续同时报覆盖率与误差。

把报告路径和退出码贴回即可，不必上传照片。下一步在同一批开发数据上计算未校正 Metric
的 MAE/RMSE/AbsRel、有效覆盖率和近距离过估，再设计多布景正式采集。
本批不能作为相对标定的独立测试集，也不能直接比较“同批拟合误差”和 Metric 泛化精度。
1 m/2 m 风险阈值仍未验证，默认生产模型、配置及告警通路不变。

## 验证范围

新增测试模拟下载损坏/哈希验证、防覆盖、单位隔离、生产统计一致性和冻结同图推理。
测试不下载真实权重、不调用模型或硬件。真实接口兼容性与预测表现待用户执行验证。

## 用户运行与未校正基线结果（2026-10-18）

用户在 0a4080c 上执行推理退出 0，CUDA、torch 2.7.0+cu118、输入设置 518。
报告 `outputs/e2_metric_4d926deb66de4dfbbd12e1d96eb9633b/report.json` 保留、不提交。
成功执行说明初始化时固定文件哈希检查通过；未单独取得下载终端日志。
助手只读复核 manifest SHA256、21 条样本身份/图像哈希/记录哈希及统计 ROI 与 v2 清单一致。
未由助手重新推理。全部 21 条 available、无逐样本错误，valid_fraction=1.0，
有效输出覆盖率 21/21；这不是准确率。MetricEvidence 全为空。

误差以用户卷尺读数 Z 为真值、未校正预测 p 为估计，使用全部 21 条：
MAE=mean(abs(p-Z))；RMSE=sqrt(mean((p-Z)^2))；AbsRel=mean(abs(p-Z)/Z)。
未尺度对齐、未拟合偏置、未剔除难例；像素过滤仍沿用既定协议。

| 实测距离档（m） | n | 平均预测（m） | MAE（m） | RMSE（m） | AbsRel |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.495–0.519 | 4 | 1.094438 | 0.589688 | 0.590937 | 1.167889 |
| 0.900–1.010 | 4 | 1.681461 | 0.702711 | 0.711084 | 0.720203 |
| 1.495–1.520 | 4 | 2.308850 | 0.802600 | 0.804306 | 0.532727 |
| 1.990–2.010 | 4 | 2.807118 | 0.807118 | 0.809083 | 0.403536 |
| 2.990–3.030 | 5 | 3.920083 | 0.914083 | 0.920720 | 0.304097 |
| 全部（按样本等权） | 21 | 2.436566 | 0.770423 | 0.782673 | 0.610377 |

21/21 都高估距离；偏差 mean(p-Z)=+0.770423 m，最大绝对误差 1.035200 m
（真值 2.990 m，预测 4.025200 m）。近距离高估对距离告警尤其不利；不接入生产风险分支。
单场景板面、未知总真值不确定度、低安装位置、曝光变化与人工派生 ROI 限制仍适用，
不能将偏差直接归因为某一因素，也不能将其解释为全部真实场景的误差。

这只建立 B0 未校正基线，不代表相对自标定已经胜出。下一步先核对真值参考点与板面姿态，
再设计按独立布景分组的拟合/验证/测试采集；相对路线同时拟合两种形式，
若加入 Metric 校正候选，须在独立测试启封前固定形式。不得用本批拟合误差冒充留出精度。
未测量：延迟、FPS、峰值内存、独立测试精度、本机有效支持区间与风险阈值表现。
