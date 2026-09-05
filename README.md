# CardScope

CardScope 是用于卡牌居中检测（centering）的 Web 平台，面向 TCG 卡牌（如 PTCG）的量产质检场景。上传卡牌实拍照片后，平台自动完成外框检测、透视矫正、内框检测与居中偏差计算，并给出合格判定。检测结果按企业隔离，测试人员可手动修正几何并提交反馈，管理员审核后用于模型再训练。

当前版本：0.9.2

## 功能

- 外框检测与矫正：先由分割模型粗定位实体轮廓，再对四边做高分辨率条带精修，经门控后输出角点与透视矫正结果。
- 内框检测：对矫正图检测内框四边，计算卡面居中度与偏差百分比。
- 居中判定：偏差百分比超过设定阈值（默认 5%）判定不合格，并输出可视化结果。
- 参考图配准：上传实拍图加标准图，平台自动归一化两图并做配准，给出偏移量与置信度。
- 手动修正与反馈：测试人员在结果页可调整外框角点与内框参考线，提交修正几何进入反馈审核队列。
- 模型再训练：管理员审核通过的标注导出到训练池，供自动训练流程使用；模型上线仍是管理员操作。
- 访问控制：企业只能访问自己的检测结果，管理员可查看全部记录；分享链接使用带令牌的保护下载地址。

## 目录结构

| 目录 / 文件 | 说明 |
| --- | --- |
| `platform_server.py` | 唯一入口。启动 HTTP 服务、前端、批处理 worker、预标注引擎与自动训练 worker。 |
| `platform_training_worker.py` | 自动训练 worker 的入口。 |
| `platform_config.json` | 平台级配置（判定阈值、反馈策略、安全选项）。 |
| `studio_config.json` | 应用级配置（工作目录、上传限制、矫正输出尺寸）。 |
| `ml_backend/` | 模型与推理包。 |
| `platform_app/` | 平台业务服务：`service.py`（核心服务）、`http_server.py`（HTTP 层）、`database.py`（存储）、`auto_training.py`（自动训练）。 |
| `studio/` | 数据与标注相关模块：`store.py`、`http_api.py`、`exports.py`、`ml_prelabel_engine.py`、`security.py` 等。 |
| `web/` | 静态前端：企业检测页 `enterprise.html`、管理端 `admin.html` 与登录页。 |
| `schemas/` | JSON Schema（标注、标签、工作室配置）。 |
| `deployment/server/` | 部署脚本：Ubuntu 安装、systemd 服务模板、Caddy 反代示例。 |
| `docs/superpowers/specs/` | 平台设计文档。 |

## 环境要求

- Python 3.10+
- 推理默认用 CPU 即可跑通；如需 GPU 加速，安装带 CUDA 的 torch。

## 安装

```bash
# 基础依赖（推理、Web 服务）
pip install -r requirements.txt

# 含训练依赖（torch 等），训练模型时需要
pip install -r requirements-ml.txt
```

## 启动

```bash
python platform_server.py
```

启动后默认监听 `127.0.0.1:8765`，自动打开浏览器。终端会打印企业检测链接、内部管理链接和访问凭据文件路径。

常用参数：

- `--host 0.0.0.0`：监听局域网或交给反代，对外发布前请配置 HTTPS。
- `--port <端口>`：改端口。
- `--workspace <目录>`：指定工作目录（默认 `platform_workspace/`）。
- `--no-browser`：不自动打开浏览器。

首次启动会在工作目录下生成访问凭据，凭据文件路径打印在终端里。

## 工作流程

1. 测试人员登录企业检测页，上传一张卡牌实拍照片。
2. 服务检测外框、矫正透视、检测内框、计算居中偏差，保存检测记录并展示结果。
3. 偏差超阈值则判不合格；结果页可导出结果 JSON（企业只能拿自己的）。
4. 测试人员可手动修正外框角点与内框参考线，提交进入反馈队列。
5. 管理员在管理端审核反馈，可通过的几何导出到训练池。
6. 管理员触发自动训练，模型上线仍由管理员操作。

## 模型与推理管线

模型清单与版本见 `ml_backend/model_manifest.json`。主要模型：

| 模型 | 作用 |
| --- | --- |
| `outer_seg.pt` | 外框实体轮廓分割粗定位（YOLOv8s segmentation）。 |
| `outer_line_refiner_v1.pt` | 外框四边高分辨率条带精修，抗阴影与套壳干扰。 |
| `outer_pose.pt` | 外框四角 Pose 回退模型。 |
| `inner_frame_yolo_v3_base_candidate.pt` | 内框分割粗定位。 |
| `inner_frame_edge_refiner_v4_candidate.pt` | 内框四边稳定精修，负责右/下边并作为左/上边回退。 |
| `inner_frame_edge_refiner_top_left.pt` | 左/上边商标干扰专项专家，仅在置信度优势满足门槛时启用。 |

推理走 `ml_backend/ptcg_inference.py` 的 `CardFramePipeline`。

## 命令行工具

单图推理：

```bash
python ml_backend/run_pipeline.py --image <图片路径> --output <输出目录>
```

质量处理工具（`ml_backend/card_quality_processor`）：

```bash
python -m ml_backend.card_quality_processor detect-outer --image <图> --output <输出目录>
python -m ml_backend.card_quality_processor batch-outer --input <目录> --output <输出目录>
python -m ml_backend.card_quality_processor detect-outer-pose --image <图> --output <输出目录>
python -m ml_backend.card_quality_processor batch-outer-pose --input <目录> --output <输出目录>
```

## 训练

训练脚本在 `ml_backend/training/` 下：

- 外框：`train_outer_seg.py`、`train_line_refiner.py`、`train_outer_pose.py`
- 内框：`train_segmentation.py`、`train_refiner.py`、`evaluate_refiner.py`

训练数据目录约定见 `ml_backend/training/data/`。

## 部署

生产目标为 Ubuntu 22.04 / 24.04。systemd 运行 `platform_server.py`（仅监听本机），Caddy 终止 HTTPS：

```bash
# 参考：Ubuntu 安装脚本
deployment/server/install_ubuntu.sh

# systemd 服务模板与 env 示例
deployment/server/cardscope.service.template
deployment/server/cardscope.env.example

# Caddy 反代示例
deployment/server/Caddyfile.example
```

持久状态放在 `/var/lib/cardscope/platform_workspace`，发布代码放在 `/opt/cardscope/releases`。版本更新保留图片、SQLite 数据、反馈、结果 JSON 与训练状态。

## 配置

- `platform_config.json`：`centering.pass_deviation_percent` 为合格偏差阈值；`security` 段控制分享令牌与 HTTPS 要求。
- `studio_config.json`：上传字节数、像素、尺寸上限，以及矫正输出尺寸（默认 630×880）。
- `ml_backend/configs/default_config.yaml`：外框检测与矫正的参数（OpenCV、Pose、分割回退、边缘精修等）。

## 支持图片格式

JPEG、PNG、WEBP、BMP、TIFF、GIF；安装 `pillow-heif` 后支持 HEIC / HEIF。

## 说明

项目文档与设计讨论主要在 `docs/` 与 `deployment/server/` 下，新增工作台或部署事项请参阅对应文档。
