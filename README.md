# Kernel Extractor

一个用 Rust 编写的高性能 GPU Kernel 提取和分析工具，用于从 PyTorch Profiler 的 JSON trace 文件中提取 GPU 操作记录并进行统计分析。

## 功能特性

- ✅ 高性能处理大型 JSON 文件（500+ MB）
- ✅ 精确的时间范围过滤
- ✅ ProfileStep 级别的 GPU 操作统计分析
- ✅ 自动过滤 prefill 阶段，只保留 decode 阶段数据
- ✅ 计算空泡时间（GPU 操作间的空闲间隔）
- ✅ 自动按时间排序
- ✅ 标准 CSV 格式输出
- ✅ 进度显示

## 使用方法

### 编译

```bash
cd kernel_extractor
cargo build --release
```

### 命令一览

```bash
# 查看帮助
./target/release/kernel_extractor

# 按时间范围提取 GPU 操作
./target/release/kernel_extractor extract <输入JSON> <输出CSV> <开始时间,结束时间>

# 统计 ProfileStep 内 GPU 操作的平均耗时
./target/release/kernel_extractor stats <输入JSON> <输出CSV> [起始kernel名称]
```

## 命令详解

### 1. `extract` - 时间范围提取

从 JSON 文件中提取指定时间范围内的 GPU 操作记录。

```bash
./target/release/kernel_extractor extract ../naive_spec_2.json output.csv 2684054.000,2687705.250
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
- `输入JSON`：PyTorch Profiler 生成的 trace JSON 文件
- `输出CSV`：统计结果输出文件
- `起始kernel名称`（可选）：指定每个 ProfileStep 中开始统计的第一个 kernel 名称（包含匹配）
  - 默认值：`recover_decode_task`
  - 传入 `none` 可禁用裁剪，统计完整的 ProfileStep

```bash
# 使用默认起始 kernel (recover_decode_task)
./target/release/kernel_extractor stats ../naive_spec_2.json profile_stats.csv

# 指定自定义起始 kernel
./target/release/kernel_extractor stats ../naive_spec_2.json profile_stats.csv my_custom_kernel

# 禁用裁剪，统计完整 ProfileStep
./target/release/kernel_extractor stats ../naive_spec_2.json profile_stats.csv none
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

## 依赖项

- `serde` v1.0 - 序列化框架
- `serde_json` v1.0 - JSON 解析
- `csv` v1.3 - CSV 生成

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

- decode 阶段：耗时 ≤ 30ms（保留）
- prefill 阶段：耗时 > 30ms（过滤）

### JSON 结构要求

输入 JSON 文件应符合 PyTorch Profiler 的标准格式：

```json
{
  "traceEvents": [
    {
      "name": "ProfileStep#1234[15.xxx ms]",
      "cat": "ProfileStep",
      "ph": "X",
      "args": {
        "start_time": "0.000 us",
        "end_time": "15xxx.xxx us"
      }
    },
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

## 项目结构

```
src/
├── main.rs           # 命令行入口，参数解析
├── extractor.rs      # 时间范围提取功能
└── profile_stats.rs  # ProfileStep 统计分析功能
```

## 许可证

MIT License