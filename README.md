# human3r-teleop-runtime

`human3r-teleop-runtime` 是一个面向在线推理与遥操作集成的 Human3R 运行时封装。

它的目标很明确：把上游 Human3R 中与“在线接收图像、执行流式推理、输出世界坐标人体结果、通过网络对外服务”相关的部分单独整理出来，形成一个更清晰、更适合作为集成组件维护的仓库。

这个仓库适合作为：

- 一个单独维护的运行时仓库
- 更大遥操作/机器人系统中的 submodule
- 在线人体重建服务的推理层

它**不是**训练仓库，也**不是**评测仓库。

## 这个仓库是做什么的

如果第一次看到这个仓库，可以直接把它理解为：

“一个基于上游 Human3R 的在线推理服务封装，用来把输入图像变成世界坐标系的人体结果，并通过 socket 对外提供服务。”

它主要负责四件事：

1. 把单帧图像整理成 Human3R 在线推理需要的输入格式
2. 维护 Human3R 的递归推理状态
3. 将模型输出整理成更稳定的世界坐标结果
4. 提供一个 TCP socket 服务端，方便外部客户端接入

## 它不负责什么

这个仓库不负责以下内容：

- 模型训练
- 论文评测
- 数据集预处理
- 权重分发
- 大型资源文件管理

这些内容仍然属于上游 Human3R 或外部环境管理范围。

## 典型使用方式

推荐的使用方式是：

1. 保留一个可正常运行的上游 Human3R 仓库
2. 用本仓库作为更干净的运行时封装层
3. 上层系统只对接本仓库暴露的服务或接口
4. 不再直接依赖上游仓库里零散的实验脚本

换句话说，这个仓库的定位是：

- 上游 Human3R：模型与研究实现
- 本仓库：面向工程集成的 runtime 层

## 当前包含的能力

当前版本已经包含：

- 单帧图像预处理
- 在线递归推理封装
- 世界坐标系人体关节导出
- TCP socket 服务端
- 一部分几何与辅助运算的本地实现
- 一部分 SMPL 导出支持的本地实现

## 当前仍然依赖什么

虽然这个仓库已经做了一轮依赖收缩，但它**仍然依赖上游 Human3R**。

目前仍保留的关键依赖主要有：

- Human3R 模型本体
- 上游 `src/models/...` 中的 SMPL / SMPL-X 资源
- Human3R 模型内部的在线推理相关接口

因此，本仓库当前的定位仍然是：

“更干净的运行时封装”，而不是“完全独立的新实现”。

## 上游仓库要求

运行本仓库时，需要能够访问一个上游 Human3R 仓库。

可通过以下两种方式指定：

- 设置环境变量 `HUMAN3R_ROOT`
- 启动时使用 `--upstream-root`

上游仓库至少应包含：

- `add_ckpt_path.py`
- `src/dust3r/...`
- `src/croco/...`
- `src/models/...`

## 安装方式

建议直接在已经可以正常运行上游 Human3R 推理的 Python 环境中安装本仓库：

```bash
pip install -e .
```

如果上游 Human3R 环境本身还没有准备好，那么即使本仓库能安装成功，运行时仍可能因为上游依赖缺失而报错。

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
  负责上游 Human3R 路径、导入与模型加载。

- `preprocess.py`
  负责将单帧图像转换为在线推理输入。

- `runtime.py`
  封装 Human3R 在线递归推理逻辑。

- `export.py`
  负责将模型输出整理为世界坐标人体结果。

- `geometry.py` / `ops.py`
  提供仓库内部使用的几何与辅助函数。

- `smpl.py`
  提供本地化后的 SMPL 导出支持层。

- `server.py`
  提供服务端实现。

- `socket_server.py`
  提供命令行启动入口。

## 服务端怎么用

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

## 客户端在哪里

当前这个仓库**只包含服务端与运行时封装**，**不包含正式整理后的客户端实现**。

也就是说：

- 服务端在本仓库中
- 客户端目前仍由上层项目或现有脚本负责

如果你的系统需要客户端，当前建议做法是：

1. 在上层项目中实现客户端
2. 客户端负责采集图像、编码 JPEG、发送给本服务端
3. 客户端接收服务端返回的 JSON 结果并做后处理

后续如果需要，也可以把客户端协议实现单独整理进本仓库，或者放到上层主项目中维护。

## 客户端如何与服务端通信

当前协议很简单：

1. 客户端先发送 4 字节大端长度
2. 客户端再发送一帧 JPEG 编码图像
3. 服务端返回一行 JSON

因此客户端只需要完成：

- 图像采集
- JPEG 编码
- TCP 发送
- JSON 接收与解析

## 返回结果里有什么

服务端返回结果通常包含以下字段：

- `frame_id`
- `server_latency_sec`
- `persons`
- `named_joints_world`
- `root_world`
- `head_world`

其中 `persons` 是最核心的字段。

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

## 当前验证状态

当前版本已经完成基础 smoke test，验证内容包括：

- 使用上游 Human3R 权重
- 使用上游示例视频
- 在空闲 GPU 上执行前几帧在线推理
- 正常返回世界坐标人体结果

因此，当前仓库已经具备作为运行时封装层使用的基本条件。

## 当前边界

当前阶段更适合把这个仓库当作：

- Human3R 的工程封装层
- 上层系统的运行时依赖

而不是当作：

- 完全脱离 Human3R 的独立推理框架
- 长期稳定不变的通用 SDK

## 后续方向

后续可以继续沿以下方向演进：

- 进一步收口对上游模型内部接口的耦合
- 视需要补充客户端实现
- 提供更稳定的 Python API
- 完善与上层遥操作框架的集成说明
