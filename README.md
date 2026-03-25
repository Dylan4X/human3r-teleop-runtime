# human3r-teleop-runtime

`human3r-teleop-runtime` 是一个面向遥操作与在线服务场景的 Human3R 运行时封装。

它将在线推理、输入预处理、结果导出与网络服务从上游 Human3R 仓库中单独整理出来，提供一个更清晰、更适合集成的运行时接口。

## 适用场景

本仓库适合以下用途：

- 作为独立的运行时仓库维护
- 作为更大遥操作系统中的 submodule 使用
- 作为在线人体重建服务的轻量封装层

本仓库不负责训练、评测、数据集处理和权重管理。

## 功能概览

当前版本提供以下能力：

- 单帧图像预处理
- Human3R 在线递归推理封装
- 世界坐标系人体关节导出
- TCP socket 推理服务
- 对上游 Human3R 的统一适配入口

## 目录结构

```text
src/human3r_teleop_runtime/
  upstream.py
  preprocess.py
  runtime.py
  export.py
  geometry.py
  ops.py
  server.py
  socket_server.py
```

各模块职责如下：

- `upstream.py`  
  负责上游 Human3R 的路径配置、导入和模型加载。

- `preprocess.py`  
  负责将输入图像转换为在线推理所需格式。

- `runtime.py`  
  封装 Human3R 的在线递归推理逻辑。

- `export.py`  
  将模型输出整理为稳定的世界坐标系人体结果。

- `geometry.py` / `ops.py`  
  提供本仓库内部使用的几何与辅助运算函数，减少对上游工具模块的直接依赖。

- `server.py` / `socket_server.py`  
  提供网络推理服务及命令行启动入口。

## 与上游 Human3R 的关系

本仓库目前仍然依赖上游 Human3R 仓库，定位是“运行时封装层”，而不是完全独立的新实现。

运行时需要能够访问一个上游 Human3R 仓库。可通过以下方式指定：

- 设置环境变量 `HUMAN3R_ROOT`
- 启动时传入 `--upstream-root`

上游仓库至少应包含以下内容：

- `add_ckpt_path.py`
- `src/dust3r/...`
- `src/croco/...`
- `src/models/...`

## 依赖说明

### 本仓库直接依赖

- `torch`
- `opencv-python`
- `numpy`
- `roma`
- `einops`

这些依赖已声明在 `pyproject.toml` 中。

### 当前仍保留的核心上游依赖

当前版本仍依赖以下上游能力：

- Human3R 模型本体
- `SMPL_Layer`
- 上游 `src/models/...` 中的 SMPL / SMPL-X 资源
- Human3R 模型内部的在线推理相关接口

这意味着：

- 本仓库已经减少了部分工具层依赖
- 但暂时仍不能完全脱离上游 Human3R 独立运行

## 权重与资源文件

模型权重和大体积资源文件不应提交到本仓库。

建议做法：

- 权重单独存放
- 启动服务时通过参数传入权重路径

## 安装方式

建议在已经能够正常运行上游 Human3R 推理的 Python 环境中安装本仓库：

```bash
pip install -e .
```

如果上游 Human3R 的运行环境尚未准备完整，本仓库即使安装成功，也可能在运行时因缺少上游依赖而报错。

## 使用方法

示例：

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

常用参数如下：

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

## 通信协议

当前采用简单的 TCP socket 协议：

1. 客户端发送 4 字节大端长度
2. 客户端发送一帧 JPEG 编码图像
3. 服务端返回一行 JSON 结果

返回结果中通常包含以下字段：

- `frame_id`
- `server_latency_sec`
- `persons`
- `named_joints_world`
- `root_world`
- `head_world`

## 当前状态

当前版本已经完成第一轮依赖收缩，部分小型工具函数已内聚至本仓库。

同时，当前版本已经完成基础 smoke test，能够在上游 Human3R 权重与示例视频上正常输出世界坐标系人体结果。

## 后续方向

后续工作将继续围绕以下目标推进：

- 进一步减少对上游工具与资源层的直接依赖
- 收口模型内部接口耦合
- 提供更稳定的集成接口

## 说明

本仓库当前重点是“可集成的运行时封装”，而不是“完全独立的通用 SDK”。
