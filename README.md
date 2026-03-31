# Trace Processor

一个用 Rust 编写的高性能 GPU Kernel 提取和分析工具，用于从 Paddle Profiler / sglang / vllm / FastDeploy 的 JSON trace 文件中提取 GPU 操作记录并进行统计分析。

## 使用方法

### 编译

```bash
cd trace_processor
cargo build --release
```

### 命令一览

```bash
# 查看帮助
./target/release/trace_processor

# 按时间范围提取 GPU 操作
./target/release/trace_processor extract <输入JSON> <输出CSV> <开始时间,结束时间>

# 统计 ProfileStep 内 GPU 操作的平均耗时
./target/release/trace_processor stats <输入JSON> <输出CSV> [起始kernel名称] [decode最大耗时ms]

# 分析 decode step 延迟（支持 sglang / vllm / fastdeploy）
./target/release/trace_processor decode-steps <framework> <输入JSON> [--output-csv <路径>] [--min-ms <值>] [--max-ms <值>]
```

## 命令详解

### 1. `extract` - 时间范围提取

从 JSON 文件中提取指定时间范围内的 GPU 操作记录。

```bash
./target/release/trace_processor extract ../naive_spec_2.json output.csv 2684054.000,2687705.250
```

**输出 CSV 格式：**

| 列名 | 说明 |
|------|------|
| `kernel_name` | GPU kernel/操作的完整名称 |
| `start_time_us` | 开始时间（微秒） |
| `end_time_us` | 结束时间（微秒） |
| `duration_us` | 执行耗时（微秒） |

### 2. `stats` - ProfileStep 统计分析

分析所有 ProfileStep 内的 GPU 操作，计算跨 step 的平均值。

**特性：**
- 自动过滤 prefill 阶段（耗时 > 30ms）
- 只统计 decode 阶段（耗时 10~20ms）
- 计算每个操作的平均开始时间、结束时间、持续时间和空泡时间
- 支持自定义起始 kernel 名称进行裁剪

**参数说明：**
- `输入JSON`：Paddle Profiler 生成的 trace JSON 文件
- `输出CSV`：统计结果输出文件
- `起始kernel名称`（可选）：指定每个 ProfileStep 中开始统计的第一个 kernel 名称（包含匹配）
  - 默认值：`recover_decode_task`
  - 传入 `none` 可禁用裁剪，统计完整的 ProfileStep
- `decode最大耗时ms`（可选）：decode 阶段最大耗时阈值（毫秒）
  - 默认值：`30`
  - 超过此阈值的 ProfileStep 被视为 prefill 阶段并过滤掉

```bash
# 使用默认起始 kernel (recover_decode_task) 和默认阈值 (30ms)
./target/release/trace_processor stats ../naive_spec_2.json profile_stats.csv

# 指定自定义起始 kernel
./target/release/trace_processor stats ../naive_spec_2.json profile_stats.csv my_custom_kernel

# 禁用裁剪，统计完整 ProfileStep
./target/release/trace_processor stats ../naive_spec_2.json profile_stats.csv none

# 禁用裁剪，并设置 decode 最大耗时为 50ms
./target/release/trace_processor stats ../naive_spec_2.json profile_stats.csv none 50

# 使用默认起始 kernel，设置 decode 最大耗时为 25ms
./target/release/trace_processor stats ../naive_spec_2.json profile_stats.csv recover_decode_task 25
```

**输出 CSV 格式：**

| 列名 | 说明 |
|------|------|
| `operation_name` | GPU 操作名称（按时间顺序排列） |
| `avg_start_time_us` | 平均开始时间（相对于 ProfileStep 开始，μs） |
| `avg_end_time_us` | 平均结束时间（相对于 ProfileStep 开始，μs） |
| `avg_duration_us` | 平均持续时间（μs） |
| `bubble_time_us` | 空泡时间（前一个操作结束到当前操作开始的间隔，μs） |

**空泡时间计算逻辑：**
- 第一个操作：从 ProfileStep 开始到第一个操作开始的时间
- 后续操作：当前操作开始时间 - 前一个操作结束时间

### 3. `decode-steps` - Decode 延迟分析

分析 sglang / vllm / FastDeploy 推理框架的 decode step 延迟，计算统计指标（mean、std、min、max、median、P90/P95/P99）。

**支持的框架：**

| 框架 | 解析策略 |
|------|---------|
| `sglang` | 按 `get_next_batch_to_run` 事件的时间戳间隔计算 |
| `vllm` | 按 `step_with_batch_queue` 事件的时间戳间隔计算 |
| `fastdeploy` | 按 `ProfileStep#N[...ms]` 事件的 `dur` 字段直接获取 |

**参数说明：**
- `framework`：推理框架名称（`sglang` / `vllm` / `fastdeploy`）
- `输入JSON`：trace JSON 文件路径
- `--output-csv <路径>`（可选）：输出延迟数据到 CSV 文件
- `--min-ms <值>`（可选）：最小延迟过滤阈值，默认 `10.0` ms
- `--max-ms <值>`（可选）：最大延迟过滤阈值，默认 `30.0` ms

```bash
# 分析 sglang trace
./target/release/trace_processor decode-steps sglang sglang_trace.json

# 分析 vllm trace，自定义过滤范围
./target/release/trace_processor decode-steps vllm vllm_trace.json --min-ms 12 --max-ms 50

# 分析 FastDeploy trace，输出 CSV
./target/release/trace_processor decode-steps fastdeploy fd_trace.json --output-csv latencies.csv
```

**输出示例：**

```
============================================================
sglang Decode Step Statistics
============================================================
Count:    79
Mean:     17.800 ms
Std Dev:  1.967 ms
Min:      10.639 ms
Max:      22.581 ms
Median:   17.837 ms
P90:      19.574 ms
P95:      20.426 ms
P99:      21.646 ms
```

**解析逻辑说明：**
- sglang / vllm：收集目标事件的 `ts` 时间戳，排序后仅保留前 50% 时间范围内的数据（确保 decode 阶段已充分加载），然后计算相邻时间戳的间隔作为 step 延迟
- FastDeploy：直接匹配 `ProfileStep#\d+[...ms]` 格式的事件，使用 `dur` 字段作为 step 延迟

## 依赖项

- `serde` v1.0 - 序列化框架
- `serde_json` v1.0 - JSON 解析
- `csv` v1.3 - CSV 生成
- `regex` v1 - 正则表达式（用于 FastDeploy 事件名匹配）

## 性能

处理 531 MB 的 JSON 文件（1,588,124 个事件）：
- 处理时间：约 5-10 秒（取决于硬件）
- 内存占用：约 1-2 GB（加载完整 JSON）

## 技术细节

### GPU 操作过滤逻辑

工具会筛选满足以下条件的事件：
1. `cat` 字段 = `"Kernel"`、`"Memcpy"` 或 `"Memset"`
2. `ph` 字段 = `"X"`（完整事件，包含持续时间）
3. 存在 `args.start_time` 和 `args.end_time` 字段

### ProfileStep 过滤逻辑（stats 命令）

- decode 阶段：耗时 ≤ 阈值（保留）
- prefill 阶段：耗时 > 阈值（过滤）
- 阈值可通过 `decode_max_duration_ms` 参数配置，默认值为 30ms

### JSON 结构要求

输入 JSON 文件应包含 `traceEvents` 数组。不同命令对事件格式的要求如下：

**`extract` / `stats` 命令**（Paddle Profiler 格式）：

```json
{
  "traceEvents": [
    {
      "name": "kernel_name[xx us]",
      "cat": "Kernel",
      "ph": "X",
      "args": {
        "start_time": "xxxx.xxx us",
        "end_time": "yyyy.yyy us"
      }
    }
  ]
}
```

**`decode-steps` 命令**（sglang / vllm / FastDeploy 格式）：

```json
{
  "traceEvents": [
    {
      "name": "python/sglang/.../get_next_batch_to_run",
      "ph": "X",
      "ts": 7240857455077.244,
      "dur": 91.79
    }
  ]
}
```

## 项目结构

```
src/
├── main.rs           # 命令行入口，参数解析
├── common.rs         # 共享数据结构（TraceEvent）和工具函数（JSON 加载、时间解析）
├── extractor.rs      # 时间范围提取功能
├── profile_stats.rs  # ProfileStep 统计分析功能
└── decode_steps.rs   # Decode step 延迟分析（sglang / vllm / FastDeploy）
```

## 许可证

MIT License