# Depth Anything V2 Small 接入说明

## 固定来源与许可

- 官方实现：[DepthAnything/Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2)。
- 使用的官方转换版：[depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)。
- 固定 revision：`5426e4f0f36572d16453bbda7a8389317b1bef99`。
- 许可：该 Small 模型的模型卡声明 Apache-2.0；此结论不推广到其他尺寸的模型。
- 权重大小：99,173,660 字节。
- 权重 SHA256：`3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`。
- Transformers：`5.6.2`，依赖定义在 `pyproject.toml`，解析结果在 `uv.lock`。

本地目录为 `models/depth-anything-v2-small-hf/`，包含 `config.json`、
`preprocessor_config.json`、`model.safetensors` 及原始模型卡 `README.md`。
这些下载文件被 Git 忽略，不随项目分发。程序验证权重 SHA256，使用
`local_files_only=True`、`trust_remote_code=False` 和 safetensors 加载。

## 新机器准备

在项目根目录执行以下 PowerShell 命令；只下载模型，不上传图片：

```powershell
$depthModelDir = 'models/depth-anything-v2-small-hf'
$depthRevision = '5426e4f0f36572d16453bbda7a8389317b1bef99'
$depthBaseUrl = 'https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/resolve/'
New-Item -ItemType Directory -Force -Path $depthModelDir | Out-Null
foreach ($depthFilename in @('config.json', 'preprocessor_config.json', 'README.md', 'model.safetensors')) {
    $depthTarget = Join-Path $depthModelDir $depthFilename
    if (Test-Path -LiteralPath $depthTarget) { throw "Already exists: $depthTarget" }
    Invoke-WebRequest -Uri ($depthBaseUrl + $depthRevision + '/' + $depthFilename) -OutFile $depthTarget -TimeoutSec 180
}
Get-FileHash (Join-Path $depthModelDir 'model.safetensors') -Algorithm SHA256
```

对照上述 SHA256；中断下载产生的残缺文件不可作为模型使用。
配置中的 `model_dir` 相对配置文件定位，`image_size=518`，可配置设备。
目前仅支持这份固定 Small 相对深度权重，不能替换成米制或其他尺寸权重。

## 语义和对齐

输入沿用 OpenCV BGR 原图，在模型边界转换成 RGB。
处理器采用 PIL 后端，保持长宽比，按模型配置归一化并调整到 14 的倍数，
不裁剪、不填充；模型输出经双三次插值回到原图 H×W，`align_corners=False`。
官方说明 Transformers 与原仓库的缩放方式可能产生差异，因此该实现作为单独实验基线，
不声称与官方 OpenCV 实现数值相同。

结果为无量纲相对逆深度，值越大表示相对越近；不能倒数后直接当作米。
有限且非负的像素视为数值有效，非有限或负值统一为 NaN 并置无效掩膜。
双三次插值可能在边缘产生负值，本实现将其明确标记为无效而非掩盖。
“数值有效”不等于具有已知精度或可信度。常量图可输出，但不提供远近区分能力。

PNG 只用于检查，每图归一化；无效位置为紫色，常量有效图为中灰。
原始 `.npz` 的 `relative_depth` 未做显示归一化；用 `allow_pickle=False` 读取。
离线 `frame_id` 是解码 BGR 像素的 SHA256，仅标识这份输入，不代表时间戳或跟踪身份。
尚未实现目标深度区域统计、标定、风险阈值或实时场景时效判断。

## 验证范围

合成测试覆盖无效值、错位尺寸、缺少权重、校验失败、同帧标识、常量图、
中文存档路径、文件覆盖保护，以及深度失败时保留检测结果。
真实本地 CUDA 推理已在街道图片上运行，输出与原图一致的 853×1280 深度图。
首轮深度调用约 296.8 ms，有效像素比例约 99.7343%，这只是一次功能试跑，
不是稳定 FPS、精度评估或安全验证；没有测量内存或显存峰值。
