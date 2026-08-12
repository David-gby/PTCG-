# PTCG 外框＋内框模型交付包

版本：2026-07-17-feedback-v2

这是一个可离线运行的 Python 推理交付包，用于把原始卡牌照片依次处理为：

```text
原始照片
  → 实体外框分割
  → 原图物理四边精修与几何门控
  → 630×880 透视矫正卡图
  → 内框分割粗定位
  → 稳定化与四边神经网络精修
  → 外框四角、内框四边及置信度 JSON
```

## 最快运行方法

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt

python run_pipeline.py `
  --image examples\input\card.jpg `
  --output demo_output `
  --device 0
```

没有 CUDA 时使用：

```powershell
python run_pipeline.py --image examples\input\card.jpg --output demo_output --device cpu
```

省略 `--device` 时，程序会在 CUDA 可用时使用 GPU，否则自动使用 CPU。

## 输出文件

`demo_output` 中会生成：

- `result.json`：统一的外框、透视矫正和内框结果；
- `outer_frame_overlay.jpg`：原图上的最终物理外框；
- `rectified_card.jpg`：630×880 矫正卡图；
- `inner_frame_overlay.jpg`：红色为内框粗定位，青色为最终精修结果。

失败时仍会生成 `result.json`，其中包含 `stage`、`error_code` 和 `message`。外框或几何门控失败时不会伪造内框结果。

## 作为库集成

如果同学的 EXE 由 Python/PyInstaller 构建，可以直接导入：

```python
import cv2
from ptcg_inference import CardFramePipeline

pipeline = CardFramePipeline(device="0")  # 批量时只创建一次
image_bgr = cv2.imread("card.jpg")
result = pipeline.infer_image(image_bgr)

if result["success"]:
    outer_corners = result["outer_frame"]["points"]
    inner_box = result["inner_frame"]["final_box"]
else:
    print(result["stage"], result["error_code"], result["message"])
```

批量推理时必须复用同一个 `CardFramePipeline`，否则每张图都会重新加载四个权重。

## 目录结构

```text
PTCG_Model_Handoff_20260717/
├── run_pipeline.py              一键命令行入口
├── ptcg_inference.py            统一 Python API
├── export_feedback.py           人工修正反馈导出入口
├── prepare_feedback_training.py 反馈校验和训练格式转换
├── feedback_schema.json         反馈 JSON Schema
├── requirements.txt             依赖范围
├── model_manifest.json          模型角色、大小和 SHA-256
├── SHA256SUMS.txt               整包文件校验清单
├── models/                      四个模型权重
├── card_quality_processor/      外框检测、精修、门控和透视矫正源码
├── inner_frame/                 内框稳定化、精修和门控源码
├── configs/                     外框参数
├── docs/                        输入输出、集成、数据许可和失败案例说明
├── examples/                    测试图片与预期输出
├── tests/                       端到端烟雾测试
└── training/                    训练脚本参考，不包含训练图片
```

## 必须一起分发的文件

不能只复制某一个 `.pt`。外框结果依赖 `outer_seg.pt` 后的物理四边精修和几何门控；内框结果依赖 YOLO 粗模型、稳定化代码、边缘精修权重及 `gate.json`。缺少其中任何一项，结果都不等于本交付包的验证结果。

## 人工反馈闭环

标注软件人工调整外框或内框后，可以导出标准反馈包：

```powershell
python export_feedback.py `
  --image examples\input\card.jpg `
  --prediction demo_output\result.json `
  --output feedback_batch_001 `
  --outer "235.3,401.8;1242.0,447.8;1201.8,1795.5;221.4,1802.5" `
  --inner "25.5,25.5,605.2,855.2" `
  --issue-tags "outer_inset,strong_glare" `
  --annotator "reviewer01" `
  --status corrected `
  --approve
```

先校验反馈：

```powershell
python prepare_feedback_training.py validate --feedback feedback_batch_001
```

再转换为四套训练格式：

```powershell
python prepare_feedback_training.py convert `
  --feedback feedback_batch_001 `
  --output feedback_training_001 `
  --split train
```

默认只有人工修正且明确批准的样本才进入训练。仅有模型预测、未审核、被拒绝或“无明确内框”的记录不会自动成为训练真值。

## 进一步说明

- [输入输出协议](docs/INPUT_OUTPUT_SPEC.md)
- [EXE 集成说明](docs/INTEGRATION_GUIDE.md)
- [已知失败案例](docs/KNOWN_FAILURES.md)
- [训练数据格式](docs/TRAINING_DATA_FORMAT.md)
- [数据来源与许可待确认项](docs/DATA_AND_LICENSE.md)
- [测试环境与性能说明](docs/TESTED_ENVIRONMENT.md)
- [人工反馈与训练回流流程](docs/FEEDBACK_WORKFLOW.md)

本包中的内框 v4 仍标记为候选模型：现有 47 张验证图参与门控选择，116 张回归图不是全新盲测集。正式上线前应增加完全独立盲测。
