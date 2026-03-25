# human3r-teleop-runtime

`human3r-teleop-runtime` 是一个面向在线推理与遥操作集成的 Human3R 运行时封装。

它将上游 Human3R 中与在线推理服务相关的部分单独整理出来，提供更清晰的输入、推理、导出与网络服务接口，便于作为独立仓库维护，或作为更大系统中的 submodule 使用。

## 项目定位

本仓库负责：

- 单帧图像预处理
- Human3R 在线递归推理封装
- 世界坐标系人体结果导出
- TCP socket 服务端

本仓库不负责：

- 模型训练
- 评测脚本
- 数据集预处理
- 权重分发
- 大型资源文件管理

## 适用方式

推荐将本仓库作为运行时层使用：

- 上游 Human3R 负责模型与研究实现
- 本仓库负责工程化的在线推理封装
- 上层系统通过本仓库接入服务，而不是直接依赖上游仓库中的零散脚本

## 目录结构

```text
src/human3r_teleop_runtime/
  upstream.py
  preprocess.py
  runtime.py
  export.py
  geometry.py
  ops.py
  smpl.py
  server.py
  socket_server.py
```

各模块职责如下：

- `upstream.py`
  上游 Human3R 路径、导入与模型加载。

- `preprocess.py`
  输入图像预处理。

- `runtime.py`
  在线递归推理逻辑。

- `export.py`
  世界坐标系人体结果导出。

- `geometry.py` / `ops.py`
  仓库内部使用的几何与辅助运算。

- `smpl.py`
  人体导出所需的 SMPL 支持层。

- `server.py` / `socket_server.py`
  推理服务与命令行入口。

## 对上游 Human3R 的依赖

本仓库目前仍依赖上游 Human3R 仓库。

运行时需要能够访问一个上游 Human3R 目录，可通过以下方式指定：

- 设置环境变量 `HUMAN3R_ROOT`
- 启动时传入 `--upstream-root`

上游仓库至少应包含：

- `add_ckpt_path.py`
- `src/dust3r/...`
- `src/croco/...`
- `src/models/...`

## 直接依赖

本仓库声明的直接 Python 依赖如下：

- `torch`
- `opencv-python`
- `numpy`
- `roma`
- `einops`

建议在已经可以正常运行上游 Human3R 推理的 Python 环境中安装本仓库：

```bash
pip install -e .
```

## 权重与资源

模型权重和大体积资源文件不应提交到本仓库。

建议：

- 权重单独存放
- 启动服务时通过参数传入权重路径

## 服务端使用方法

最常见的用法是直接启动 socket 服务端：

```bash
export HUMAN3R_ROOT=/path/to/Human3R

python -m human3r_teleop_runtime.socket_server \
  --model-path /path/to/checkpoints/human3r_model.pth \
  --upstream-root /path/to/Human3R \
  --host 127.0.0.1 \
  --port 19999 \
  --device cuda \
  --size 256
```

常用参数说明：

- `--model-path`
  Human3R 权重路径。

- `--upstream-root`
  上游 Human3R 仓库根目录。若已设置 `HUMAN3R_ROOT`，可省略。

- `--host`
  服务监听地址。

- `--port`
  服务监听端口。

- `--device`
  推理设备，通常为 `cuda` 或 `cpu`。

- `--size`
  输入图像缩放尺寸。

- `--use-ttt3r`
  是否启用相关时序选项。

- `--tf32`
  在支持的 GPU 上启用 TF32。

- `--warmup`
  启动后执行一次预热。

- `--reset-on-new-client`
  每次有新客户端连接时重置流式状态。

## 客户端说明

当前本仓库只包含服务端与运行时封装，不包含正式整理后的客户端实现。

推荐由上层项目或独立脚本承担客户端职责。客户端需要完成：

- 图像采集
- JPEG 编码
- TCP 发送
- JSON 接收与解析

## 通信协议

当前采用简单的 TCP socket 协议：

1. 客户端发送 4 字节大端长度
2. 客户端发送一帧 JPEG 编码图像
3. 服务端返回一行 JSON

## 返回结果

服务端返回结果通常包含以下字段：

- `frame_id`
- `server_latency_sec`
- `persons`
- `named_joints_world`
- `root_world`
- `head_world`

其中 `persons` 是核心字段。

每个 `person` 通常包含：

- `id`
- `root_world`
- `head_world`
- `left_wrist_world`
- `right_wrist_world`
- `left_ankle_world`
- `right_ankle_world`
- `named_joints_world`
- `joints_world`

如果上层系统希望使用稳定字段访问，建议优先使用：

- `named_joints_world`

如果需要完整关节数组，可使用：

- `joints_world`

## 仓库边界

本仓库的目标是提供一个更清晰的运行时封装层，而不是完全脱离 Human3R 的独立推理框架。
