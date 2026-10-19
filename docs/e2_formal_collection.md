# E2 多布景与不同目标的分组采集

2026-10-20。工具已准备，真实采集待用户执行；无相机或模型由助手启动。
用户已确认既有真值从镜头地面投影量至板前表面，板竖直正对镜头；这是用户确认，非精密几何校验。
现有 scene01 的 21 条保持为开发预采集，试拍另留，不迁入拟合/验证/测试。

## 分组和目标

预注册文件：configs/e2_collection.json；开始保存后哈希固定，不要修改此文件重分组。

| 布景 | 划分 | 每组采集 |
| --- | --- | --- |
| scene02、scene03、scene04 | fit | 0.5/1/1.5/2/3 m，每档两次独立摆放，10 条 |
| scene05、scene06 | validation | 同上，每组 10 条 |
| scene07、scene08 | test | 预留；当前工具锁定，不采集、不读取用于调参 |

先完成 scene02 第一条并审查目标/框，再采该组余下 9 条。不同布景需实际改变背景、周围布置或
光照条件，不能只改文件名；相机安装保持不变。工具只能检查记录一致性，不能证明布景物理独立。
测试组须等候选公式、选择规则、覆盖率和误差验收要求冻结后另行开放，不因已预注册就提前采。

可使用无孔哑光板、硬纸板或纸箱平整正面；记录被测面的宽高（米），不是纸箱深度。
每块物理目标有独立 target-id；材质或尺寸变化时换编号。同一目标可跨布景出现，但这不能验证
对未知目标的泛化；正式协议应说明至少一个独立测试布景是否使用未参与拟合的目标。
每个目标覆盖多个距离档，避免把目标身份与距离完全绑定。人体、椅子等多表面目标留给后续生产验证。

本轮延续 board_lower_half_v1：人工框选可见平面目标，取该框下半部后再用生产中心区域统计。
即便新目标无孔，也不单独换统计策略。下半部不能包含孔洞、折痕、手或背景；有这些情况时先换目标
或调整姿态，不事后根据模型误差选择 ROI。保存原图及原框，派生框在审查时生成。

## 当前终端一次性填写

请先实际布置 scene02：相机仍按既定位置，改变原 scene01 的背景或周围布置。
若这两天移动了相机或屏幕、无法确认恢复原安装，请先报告，不复用安装编号强行混池。

```powershell
Set-Location 'D:\A Visual Assistance System for Visually Impaired People Based on YOLO11 and Depth Anything V2'
$e2Group = 'scene02'
$e2Target = Read-Host '目标编号（例如 board02；同一物体保持一致）'
$e2Material = Read-Host '材质、颜色、表面特征（例如哑光硬纸板）'
$e2Width = Read-Host '被测正面的实际宽度（米）'
$e2Height = Read-Host '被测正面的实际高度（米）'
$e2Instrument = Read-Host '工具名称、最小刻度、已知标称精度'
$e2Conditions = Read-Host '实际背景布置、光照及与 scene01 的区别'
```

## 每次独立摆放后执行

第一条先摆在约 0.5 m，仍按镜头地面投影量到目标正面地面投影，填实际读数。
以下每次只采一帧，不是自动循环；不使用未测量的名义档位替代真值。

```powershell
$e2Placement = Read-Host '本布景独立摆放编号（首次 p01，之后 p02、p03…）'
$e2Distance = Read-Host '本次实际距离（米）'
D:\python\python.exe -m app.e2_capture --save-local --collection-plan configs/e2_collection.json --group-id $e2Group --installation-id omen_h0265_v1 --placement-id $e2Placement --target-id $e2Target --target-material $e2Material --target-width-m $e2Width --target-height-m $e2Height --distance-m $e2Distance --instrument $e2Instrument --conditions $e2Conditions
$LASTEXITCODE
```

框选平面目标、Enter、回终端 yes 保存。未保存的失败/取消可重试同一摆放编号；
已保存编号不能再次使用。把目标移开重新摆放、重新测量后才使用下一个编号。
第一条贴回记录路径与退出码，核对后再继续。更换目标或布景时更新对应变量，不能沿用错误描述。
工具不把 0.1 cm 自动当作总测量不确定度；未知仍保留 null。

## 无硬件审查及离线接口

正式记录使用 e2_board_collection_v1，保存计划哈希、split、目标描述与 placement-id。
旧 e2_board_precapture_v1 和原始数据保留；两种记录不会被默认审查混合。
拟合/验证均按集合一次审查，输出冻结清单供两模型共同读取：

```powershell
$e2FitAudit = 'outputs/e2_fit_audit_' + [guid]::NewGuid().ToString('N')
D:\python\python.exe -m app.e2_offline --split fit --collection-id e2_board_multiscene_v1 --board-lower-half --output $e2FitAudit
```

这个命令不要求已达到 30 条，必须另核对布景数量、每档独立摆放数与图像质量；
少量样本的审查通过不代表采集完成。验证集合将 --split 改为 validation。
随后已有相对与 Metric 离线入口可读取对应清单，但当前先采第一组，不提前使用验证集选择公式。
正式参数拟合、候选比较汇总与测试解锁仍待实现；本轮没有自动拟合或发布支持记录。
所有输出仍在被忽略目录，保留期限未指定，不自动删除。
