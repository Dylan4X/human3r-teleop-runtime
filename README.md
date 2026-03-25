# human3r-teleop-runtime

这是一个面向遥操作/远程控制场景的 Human3R 运行时封装仓库。

它的目标不是替代上游 Human3R，也不是彻底与上游解耦，而是把在线推理、输入预处理、结果导出和网络服务这几部分单独整理出来，形成一个更干净、更适合集成的运行时层。

这个仓库后续适合作为：

- 独立 GitHub 仓库
- 更大遥操作框架中的 submodule

## 这个仓库包含什么

- 单帧输入预处理
- Human3R 的在线递归推理状态封装
- 世界坐标系人体关节导出
- 简单的 TCP socket 服务端
- 对上游 Human3R 的统一适配入口

## 这个仓库不包含什么

- 训练代码
- 评测脚本
- 数据集预处理
- 模型权重
- 大体积资源文件

## 目录结构

```text
src/human3r_teleop_runtime/
  upstream.py
  preprocess.py
  runtime.py
  export.py
  server.py
  socket_server.py
```

各模块职责如下：

- `upstream.py`
  统一处理对上游 Human3R 的路径和导入依赖。
- `preprocess.py`
  负责把单帧图像整理成 Human3R 在线推理所需输入。
- `runtime.py`
  封装在线递归推理主逻辑。
- `export.py`
  将模型输出整理为更稳定的世界坐标系关节结果。
- `server.py`
  提供 socket 推理服务封装。
- `socket_server.py`
  提供命令行启动入口。

## 与上游 Human3R 的关系

这个仓库目前仍然依赖上游 Human3R 的内部实现，因此它更准确的定位是：

“面向集成的运行时封装层”，而不是“完全独立的新实现”。

运行时需要能访问一个上游 Human3R 仓库，并通过以下两种方式之一指定：

- 设置环境变量 `HUMAN3R_ROOT`
- 启动时传入 `--upstream-root`

上游仓库至少需要包含这些内容：

- `add_ckpt_path.py`
- `src/dust3r/...`
- `src/croco/...`
- `src/models/...`

当前对上游的依赖被尽量集中在 `human3r_teleop_runtime.upstream` 中，目的是让上层项目尽量不要直接依赖零散的 Human3R 脚本。

## 本仓库自己的 Python 依赖

当前直接依赖如下：

- `torch`
- `opencv-python`
- `numpy`
- `roma`
- `einops`

这些依赖已经写在 `pyproject.toml` 中。

## 仍然来自上游 Human3R 的依赖

目前这个仓库在运行时仍然会使用上游 Human3R 中的若干模块，例如：

- `dust3r.model`
- `dust3r.utils.camera`
- `dust3r.utils.geometry`
- `dust3r.utils.image`
- `dust3r.utils.smpl_layer`
- `dust3r.post_process`
- `dust3r.smpl_model`

因此如果上游 Human3R 的内部接口发生变化，这个仓库也可能需要同步调整。

## 关于模型权重

模型权重不应提交到本仓库中。

建议做法是：

- 权重单独存放
- 启动服务时通过参数传入权重路径

这样仓库本身会保持更干净，也更适合公开或作为子模块维护。

## 基本用法

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

## 推荐使用流程

建议按下面的顺序使用：

1. 准备上游 Human3R 仓库
2. 准备可用的 Human3R 权重
3. 设置 `HUMAN3R_ROOT` 或传入 `--upstream-root`
4. 启动 `socket_server`
5. 让上层客户端按协议发送 JPEG 帧并接收 JSON 结果

## 环境准备

这个仓库本身只声明了较小的一组直接依赖，但实际运行仍然依赖上游 Human3R 的环境。

因此更稳妥的做法通常是：

- 直接使用上游 Human3R 已经能正常推理的 Python 环境
- 在这个环境里安装本仓库

例如：

```bash
pip install -e .
```

如果上游 Human3R 依赖尚未安装完整，即使本仓库自身可以成功安装，运行时也仍然可能报错。

## 启动服务

最常见的启动方式：

```bash
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
  上游 Human3R 仓库根目录。如果已经设置 `HUMAN3R_ROOT`，这个参数可以不传。
- `--host`
  服务监听地址。
- `--port`
  服务监听端口。
- `--device`
  推理设备，通常为 `cuda` 或 `cpu`。
- `--size`
  输入缩放尺寸，目前默认设计为在线推理场景使用。
- `--use-ttt3r`
  是否启用相关时序选项。
- `--tf32`
  在支持的 GPU 上启用 TF32。
- `--warmup`
  启动后先进行一次预热。
- `--reset-on-new-client`
  每次有新客户端连接时重置流式状态。

## 与客户端的对接方式

推荐的客户端工作流如下：

1. 读取摄像头或视频流图像
2. 将单帧编码为 JPEG
3. 先发送 4 字节大端长度
4. 再发送 JPEG 二进制内容
5. 读取服务端返回的一行 JSON
6. 从 JSON 中取出人体世界坐标结果

## 返回结果说明

当前返回结构以 `persons` 为核心，每个人通常包含：

- `id`
- `root_world`
- `head_world`
- `left_wrist_world`
- `right_wrist_world`
- `left_ankle_world`
- `right_ankle_world`
- `named_joints_world`
- `joints_world`

其中：

- `named_joints_world` 更适合上层做稳定字段访问
- `joints_world` 更适合调试或保留完整关节序列

## 作为 submodule 使用的建议

如果这个仓库后续作为更大遥操作框架的 submodule 使用，推荐做法是：

1. 上层主项目负责管理整体环境
2. 上层主项目通过配置提供：
   - 上游 Human3R 路径
   - 权重路径
   - host / port
   - 设备选择
3. 上层只依赖本仓库暴露的运行时接口，不直接引用零散的上游实验脚本

这样做的好处是：

- 上游升级时影响面更小
- 推理服务接口更稳定
- 代码职责更清楚

## 当前限制

目前仍然存在这些限制：

- 还没有彻底摆脱上游 `dust3r` 内部接口
- 上游 Human3R 改动后，这里可能需要同步适配
- 当前重点是在线推理和服务封装，不是通用 SDK

## Socket 协议

当前通信协议尽量保持简单：

- 客户端先发送 4 字节的大端长度
- 然后发送一帧 JPEG 编码图像
- 服务端返回一行 JSON 结果

返回结果中通常包含这些字段：

- `frame_id`
- `server_latency_sec`
- `persons`
- `named_joints_world`
- `root_world`
- `head_world`

## 当前状态

这还是第一版抽离结果，目标主要是：

- 减少与训练/评测/实验脚本的混杂
- 给上层遥操作框架提供更清晰的运行时入口
- 把对上游 Human3R 的依赖集中到更少的模块中

目前它已经比直接引用原始仓库中的零散脚本更适合集成，但还不是最终定型版本。

## 已完成的第一轮减依赖

当前已经把一批原本来自上游 Human3R / dust3r 的小型工具函数收进本仓库，减少了对上游工具模块的直接引用。

目前已本地化的函数包括：

- 相机位姿解码
- 焦距估计
- 坐标变换
- `unpad_uv`
- `log_optimal_transport`
- `nms`
- `apply_threshold`
- 基础相机内参构造

因此，这个仓库现在对上游的依赖已经更多集中在：

- Human3R 模型本体
- `SMPL_Layer`
- 模型内部递归推理接口

## 当前仍然保留的核心上游依赖

下面这些依赖暂时还没有移除：

- `dust3r.model.ARCroco3DStereo`
- `dust3r.utils.smpl_layer.SMPL_Layer`
- 上游 `src/models/...` 中的 SMPL / SMPL-X 资源
- Human3R 模型内部若干推理接口和属性

其中最关键的是，当前在线推理仍然会直接调用模型内部接口，例如：

- 编码图像
- 初始化递归状态
- 执行递归 rollout
- 执行 downstream head

这部分耦合暂时保留，是因为它直接关系到你当前重写后的在线推理链路。

## 为什么还不能完全脱离上游

原因主要有两类：

1. 模型内部接口耦合较深  
   当前在线推理不是只调用一个公开的 `forward()`，而是直接使用了 Human3R 模型的多段内部流程。

2. SMPL / 几何资源仍依赖上游仓库  
   人体导出部分仍然需要上游已有的模型资源与部分实现。

所以当前策略不是“彻底脱离”，而是：

- 先把能稳定搬进来的小工具搬进来
- 再逐步把大依赖收口到更少的 adapter 中

## 实际验证情况

当前版本已经做过一次轻量验证：

- 使用上游 Human3R 权重
- 使用上游示例视频 `GoodMornin1.mp4`
- 在空闲 GPU 上进行了前几帧 smoke test
- 成功输出世界坐标人体结果
- 返回结果中包含 `persons` 和 `named_joints_world`

这说明当前第一轮减依赖后，基础推理链路仍然可用。

## 后续建议

接下来建议继续沿下面的顺序推进：

1. 继续减少 `SMPL_Layer` 周边依赖
2. 进一步收口模型内部接口调用
3. 视需要补一个更稳定的 Python API 层
4. 如果后续要长期作为 submodule 使用，再补更明确的版本约束与接口约定
