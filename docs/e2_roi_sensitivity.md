# E2 固定图像的ROI敏感性对照

2026-10-26，续接02。工具已准备，真实模型推理待用户执行。
本轮针对相对深度r的布景漂移，使用现有相对模型对已有图像推理；没有新采集步骤。
这是一项看过开发数据之后定义的探索性诊断，不是独立验证或候选选型。

## 固定选择与区域定义

协议为 `configs/e2_roi_sensitivity.json`，从此前比较清单中按布景及预注册名义距离
选择全部重复，不按误差或亮度挑选。实际距离继续保留原始读数，不修改为名义距离。

| 布景 | 原划分 | 0.5米档 | 3米档 |
|---|---|---|---|
| scene10 | supplemental_fit | p03、p04 | p09、p10 |
| scene11 | new_validation | p01、p02 | p09、p10 |
| scene12 | supplemental_fit | p03、p04 | p09、p10 |

清单已生成：`outputs/e2_roi_prepared_20261026_v1/manifest.json`，共12条。
准备阶段核验完整84条源数据及模型文件哈希，未构造模型或调用相机。

两种区域同时从一张完整图像的同一深度图读取：

1. `original_statistics_v1`：原可见框下半部再取中心区域，精确复用既有生产统计核。
2. `center_half_of_statistics_v1`：以第一种统计框为基础，宽高各保留中心50%，
   左上向上取整、右下向下取整。面积约为第一种的四分之一，受像素取整影响。
   第二种框直接按框内中位数取值，不再隐式收缩。

无效像素过滤、5%分位数修剪、最小有效像素和比例要求沿用原配置。
较小框像素不足时返回不可用，不调低最小像素数。当前12条较小框最小面积为55像素。
该规则只缩小已有可见统计区域，无法恢复画外箱面，也不保证所有图像采到同一物理部位。

## 执行与输出

`app.e2_roi_sensitivity` 的prepare模式只生成清单。infer模式先重建清单并核对所有身份、
协议、配置、区域、模型文件哈希，再单次加载本地Depth Anything V2 Small相对模型。
每图仅执行一次全帧预测，保持518模型输入及原图网格对齐，不分别裁剪两区域送入模型。

每条保留：

- 原区域与中心缩小区域的相对值、IQR、有效像素数与状态；
- `center_minus_original_relative`：同次推理两区域相对值之差；
- `rerun_minus_historical_relative`：此次原区域减历史原区域相对值，以检查重跑差异；
- 原始角色、实际读数、名义距离、图像/记录身份和坐标；
- `.npz`浮点深度、有效掩膜、原始帧身份、模型身份、单位，以及文件SHA256。

相对值与差值都没有米制单位。报告不拟合新系数，不根据scene11误差选择区域。
执行环境记录实际设备、NumPy/Torch/Transformers版本、代码提交、执行源码SHA256及快照。
与历史值的差异可能还受设备、库版本及数值执行影响，不能自动解释为ROI效应。

全部12条保留；有异常或任一对照不可用时进程退出2，报告仍保存可检查的结果。
`available_pair_count`与`unavailable_pair_count`明确有效对数及分母。
模型初始化或输入校验失败时尚未生成结果报告，请保留终端报错。
原图只读，不复制进结果目录；浮点图保存在本地忽略目录，旧输出禁止覆盖。

## 现在执行的完整命令

```powershell
Set-Location 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'

$e2RoiOutput = 'outputs/e2_roi_relative_' + [guid]::NewGuid().ToString('N')

D:\python\python.exe -m app.e2_roi_sensitivity --infer --manifest outputs/e2_roi_prepared_20261026_v1/manifest.json --output $e2RoiOutput

if ($LASTEXITCODE -ne 0) { throw 'ROI对照失败或存在不可用记录，请保留终端输出及结果目录。' }
Write-Host "结果报告：$e2RoiOutput/report.json"
```

完成后提供终端结果路径，继续检查两区域差值及各布景差异是否持续存在。
不需要摆放纸箱、选择ROI或打开摄像头。

如需在另一次实验中重新准备清单，可使用独占新目录：

```powershell
Set-Location 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'
$e2RoiPrepared = 'outputs/e2_roi_prepared_' + [guid]::NewGuid().ToString('N')
D:\python\python.exe -m app.e2_roi_sensitivity --prepare --output $e2RoiPrepared
if ($LASTEXITCODE -ne 0) { throw 'ROI对照清单准备失败，请保留终端输出。' }
Write-Host "清单路径：$e2RoiPrepared/manifest.json"
```

已冻结的清单如果与当前输入不匹配，入口会拒绝运行；先核对变化原因，不直接修改清单绕过。

## 判断范围

优先比较同一次深度图的两ROI差值，再核对原ROI重跑与历史的差异。
只有在重跑差异可解释的前提下，才讨论与历史诊断的数值对照。
若缩小ROI后组间相对值差异仍远大于区域变化，支持继续调查全图/布景因素；
若区域变化较大，则先检查具体采样位置和物体纹理。两者都不是曝光或反光的因果证明。

scene11仍保留开发验证角色。研究验收目标不变，scene07/08继续锁定，生产米制支持未获批准。
未启动真实模型，暂无本轮实测ROI差值、精度、FPS、延迟或内存结论。
