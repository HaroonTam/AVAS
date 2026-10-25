# E2 边界补采（研究目标已确认）

## 2026-10-25：scene12 替代未完成的 scene09

scene09 原位置不足 3 m，仅一条 p01，保留原记录但不计为完整拟合布景。
scene10 已完成 12 条，scene11 已完成 10 条。用户确认另一个位置有实际不同背景/布置且
可覆盖 3.25 m，新登记 scene12（fit），使用独立计划
`configs/e2_boundary_fit_replacement.json`、collection_id `e2_boundary_fit_replacement_v1`。
旧计划与哈希不改。后续补充拟合集采用 scene10 + scene12 共 24 条；scene09 单条作为
未完成组留存，不悄悄合入正式比较。多计划合并入口需显式记录这个取舍，不能要求补回旧位置。

scene12 继续使用同一 A02 纸箱和相机安装，距离档 0.4/0.5/0.6/2.75/3/3.25 m，各两次。
先执行首条并审查图像：

```powershell
Set-Location 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/collect_e2_boundary.ps1 -Group scene12 -StartPlacement 1 -EndPlacement 1
```

首条核对后，再执行剩余 11 条：

```powershell
Set-Location 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/collect_e2_boundary.ps1 -Group scene12 -StartPlacement 2 -EndPlacement 12
```

下方为原始计划与操作记录，不要求重采已完成的 scene10/11。

2026-10-24。用户已确认研究目标，固定于 `configs/e2_research_acceptance.json`。
仅目标得到确认，最终候选系数、范围策略和独立测试授权尚未冻结；scene07/08 继续锁定。

## 补采设计

当前 A1 范围限制后的验证覆盖仅 15/20，近端和远端明显不足。补采用于研究边界拟合，
不能保证补采后一定通过，也不改变已确认的误差/覆盖率目标。

| 数据 | 新计划 | 布景 | 距离档（米，每档两次独立摆放） | 数量 |
| --- | --- | --- | --- | --- |
| 补充 fit | configs/e2_boundary_fit.json | scene09、scene10 | 0.4、0.5、0.6、2.75、3、3.25 | 各 12，共 24 |
| 新 validation | configs/e2_boundary_validation.json | scene11 | 0.5、1、1.5、2、3 | 10 |

0.4 与 3.25 m 是边界外的开发支撑点，不能据此直接扩张宣称的工作范围。
完整原始 fit/validation 保留，绝不把原验证失败样本重新命名为拟合数据。
最终研究目标仍考察 0.5–3 m，新增输出范围和拒绝规则须经新 validation 比较后再冻结。
补充 fit 与新 validation 使用独立 collection_id，旧 collection plan 完全不改，避免破坏哈希。

每组使用真实不同背景/布置，记录灯光、反光和与旧组的区别，不能仅换编号。
继续使用同一 A02 黄色纸箱（正面 0.243×0.271 m），安装、屏幕角度、相机高度保持已确认条件。
先检查每个新布景 p01；图像中有效统计区不能包含手、背景或孔洞。反光按实际情况记录。
0.4 m 可能使目标底部截断；若无法选择足够的平整可见正面，停止并报告，不移动相机凑框。
若现场没有 3.25 m 空间，先说明，不填名义距离冒充实测值。

## 完整操作命令

新脚本默认只采一条；指定起止编号可以续采。首次运行询问布景说明，后续同组可沿用已保存说明；
若条件实际变化，显式传 `-Conditions '实际说明'`，不要沿用不真实的记录。
脚本检查空值、非正数、NaN/Inf；图像框选后 Enter，回终端 yes 保存。
任意采集失败即停止，已保存编号由 Python 入口拒绝重复；按未保存编号续采。

先执行 scene09 / p01（约 0.4 m）：

```powershell
Set-Location 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/collect_e2_boundary.ps1 -Group scene09 -StartPlacement 1 -EndPlacement 1
```

检查 p01 后采 scene09 剩余 11 条：

```powershell
Set-Location 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/collect_e2_boundary.ps1 -Group scene09 -StartPlacement 2 -EndPlacement 12
```

布置新的 scene10，先采 p01，再经图像检查后执行余下编号：

```powershell
Set-Location 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/collect_e2_boundary.ps1 -Group scene10 -StartPlacement 1 -EndPlacement 1
```

```powershell
Set-Location 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/collect_e2_boundary.ps1 -Group scene10 -StartPlacement 2 -EndPlacement 12
```

再布置新的验证 scene11，同样先检查 p01（约 0.5 m）：

```powershell
Set-Location 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/collect_e2_boundary.ps1 -Group scene11 -StartPlacement 1 -EndPlacement 1
```

```powershell
Set-Location 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/collect_e2_boundary.ps1 -Group scene11 -StartPlacement 2 -EndPlacement 10
```

可不启用相机查看计划：

```powershell
Set-Location 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/collect_e2_boundary.ps1 -Group scene09 -StartPlacement 1 -EndPlacement 12 -Preview
```

## 后续分析边界

采集后分别按 collection_id 审查、冻结并推理；不混入旧审查清单，不复写报告。
现有 app.e2_compare 按单计划完整集合验证，尚不能自动合并原始与补充的多个计划。
合并比较入口需在本轮数据完成后扩展显式输入验证：原 fit 30 + 补充 fit 24 可参与拟合；
原 validation 20 仅作已使用的开发对照，新 validation 10 单独报告并按确认目标评价。
不靠把新旧 validation 合并平均来掩盖新场景失败；任何新的公式仍须披露开发次序。
本阶段仅准备采集，不提前生成新的支持范围、解锁最终测试或改变生产告警。
