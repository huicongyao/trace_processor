use serde::{Deserialize, Serialize};
use serde_json::Value;

use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// ProfileStep 事件
#[derive(Debug, Clone)]
pub struct ProfileStep {
    pub name: String,
    pub start_time: f64,
    pub end_time: f64,
}

/// GPU 操作记录
#[derive(Debug, Clone)]
pub struct GpuOperation {
    pub name: String,
    pub start_time: f64,
    pub end_time: f64,
    pub duration: f64,
}

/// 输出的统计记录
#[derive(Debug, Serialize)]
pub struct ProfileStatsRecord {
    pub operation_name: String,
    pub avg_start_time_us: f64,
    pub avg_end_time_us: f64,
    pub avg_duration_us: f64,
    /// 空泡时间：前一个操作结束到当前操作开始的时间间隔
    pub bubble_time_us: f64,
}

/// 追踪事件结构（用于解析）
#[derive(Debug, Deserialize)]
struct TraceEvent {
    name: String,
    #[serde(default)]
    cat: Option<String>,
    #[serde(default)]
    ph: Option<String>,
    #[serde(default)]
    args: Option<TraceArgs>,
}

#[derive(Debug, Deserialize)]
struct TraceArgs {
    #[serde(default)]
    start_time: Option<String>,
    #[serde(default)]
    end_time: Option<String>,
}

/// 解析时间字符串，如 "6609483.000 us"
fn parse_time_from_string(time_str: &str) -> Option<f64> {
    time_str
        .trim()
        .split_whitespace()
        .next()
        .and_then(|s| s.parse::<f64>().ok())
}

/// 标准化操作名称：去掉方括号中的动态时间信息
/// 例如 "MEMCPY_DtoH[2.464 us]" -> "MEMCPY_DtoH"
/// 例如 "kernel_name[123.456 us]" -> "kernel_name"
fn normalize_op_name(name: &str) -> &str {
    // 找到最后一个 '[' 的位置，检查是否是时间后缀
    if let Some(bracket_pos) = name.rfind('[') {
        let suffix = &name[bracket_pos..];
        // 检查是否匹配 "[数字 us]" 或 "[数字 ms]" 格式
        if suffix.ends_with(" us]") || suffix.ends_with(" ms]") {
            return &name[..bracket_pos];
        }
    }
    name
}

/// 从 JSON 文件中统计 ProfileStep 内 GPU 操作的平均耗时
/// 
/// # Arguments
/// * `input_file` - 输入的 JSON trace 文件路径
/// * `output_file` - 输出的 CSV 统计文件路径  
/// * `trim_start_kernel` - 可选，指定每个 ProfileStep 中开始统计的第一个 kernel 名称（包含匹配）
pub fn analyze_profile_stats(input_file: &str, output_file: &str, trim_start_kernel: Option<&str>) -> Result<(), Box<dyn Error>> {
    println!("Processing JSON file: {}", input_file);

    // 打开并解析 JSON 文件
    let file = File::open(input_file)?;
    let reader = BufReader::new(file);
    
    println!("Parsing JSON (this may take a while for large files)...");
    let json: Value = serde_json::from_reader(reader)?;
    
    // 获取 traceEvents 数组
    let trace_events = json["traceEvents"]
        .as_array()
        .ok_or("traceEvents not found or not an array")?;

    println!("Total events in file: {}", trace_events.len());

    // 第一遍：收集所有 ProfileStep 和 GPU 操作
    let mut profile_steps: Vec<ProfileStep> = Vec::new();
    let mut gpu_operations: Vec<GpuOperation> = Vec::new();

    for event_value in trace_events {
        let event: TraceEvent = match serde_json::from_value(event_value.clone()) {
            Ok(e) => e,
            Err(_) => continue,
        };

        if let (Some(cat), Some(ph), Some(args)) = (&event.cat, &event.ph, &event.args) {
            if let (Some(start_str), Some(end_str)) = (&args.start_time, &args.end_time) {
                if let (Some(start), Some(end)) = (
                    parse_time_from_string(start_str),
                    parse_time_from_string(end_str),
                ) {
                    if cat == "ProfileStep" && ph == "X" {
                        profile_steps.push(ProfileStep {
                            name: event.name.clone(),
                            start_time: start,
                            end_time: end,
                        });
                    } else if (cat == "Kernel" || cat == "Memcpy" || cat == "Memset") && ph == "X" {
                        // 标准化名称：去掉动态时间后缀，因为执行时间可以通过 start/end 计算
                        gpu_operations.push(GpuOperation {
                            name: normalize_op_name(&event.name).to_string(),
                            start_time: start,
                            end_time: end,
                            duration: end - start,
                        });
                    }
                }
            }
        }
    }

    println!("Found {} ProfileSteps", profile_steps.len());
    println!("Found {} GPU operations", gpu_operations.len());

    if profile_steps.is_empty() {
        return Err("No ProfileStep events found".into());
    }

    // 按开始时间排序
    profile_steps.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());
    gpu_operations.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());

    // 过滤掉 prefill 阶段（耗时 > 30ms 的 ProfileStep）
    // decode 通常耗时 10～20ms，prefill 耗时 40～50ms
    const DECODE_MAX_DURATION_US: f64 = 30000.0; // 30ms
    let total_before_filter = profile_steps.len();
    profile_steps.retain(|step| {
        let duration = step.end_time - step.start_time;
        duration <= DECODE_MAX_DURATION_US
    });
    
    let filtered_count = total_before_filter - profile_steps.len();
    println!(
        "Filtered out {} prefill steps (duration > {}ms), {} decode steps remaining",
        filtered_count,
        DECODE_MAX_DURATION_US / 1000.0,
        profile_steps.len()
    );

    if profile_steps.is_empty() {
        return Err("No decode ProfileStep events found after filtering".into());
    }

    // 为每个 ProfileStep 收集其时间范围内的 GPU 操作
    // 并转换为相对于 ProfileStep 开始的相对时间
    let mut step_operations: Vec<Vec<GpuOperation>> = Vec::new();

    for step in &profile_steps {
        let mut ops_in_step: Vec<GpuOperation> = Vec::new();
        
        for op in &gpu_operations {
            // GPU 操作在 ProfileStep 时间范围内
            if op.start_time >= step.start_time && op.end_time <= step.end_time {
                // 转换为相对时间（相对于 ProfileStep 开始）
                let relative_start = op.start_time - step.start_time;
                let relative_end = op.end_time - step.start_time;
                
                ops_in_step.push(GpuOperation {
                    name: op.name.clone(),
                    start_time: relative_start,
                    end_time: relative_end,
                    duration: op.duration,
                });
            }
        }
        
        // 按相对开始时间排序
        ops_in_step.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());
        
        // 如果指定了起始 kernel，从该 kernel 开始，裁去在这之前的操作
        if let Some(trim_kernel) = trim_start_kernel {
            if let Some(start_idx) = ops_in_step.iter().position(|op| op.name.contains(trim_kernel)) {
                // 获取新的起始时间点
                let new_base_time = ops_in_step[start_idx].start_time;
                
                // 裁剪并重新计算相对时间
                ops_in_step = ops_in_step[start_idx..]
                    .iter()
                    .map(|op| GpuOperation {
                        name: op.name.clone(),
                        start_time: op.start_time - new_base_time,
                        end_time: op.end_time - new_base_time,
                        duration: op.duration,
                    })
                    .collect();
                
                println!("ProfileStep '{}': {} GPU operations (trimmed from '{}' at index {})", 
                         step.name, ops_in_step.len(), trim_kernel, start_idx);
            } else {
                println!("ProfileStep '{}': {} GPU operations (trim kernel '{}' not found)", 
                         step.name, ops_in_step.len(), trim_kernel);
            }
        } else {
            println!("ProfileStep '{}': {} GPU operations (no trimming)", 
                     step.name, ops_in_step.len());
        }
        
        step_operations.push(ops_in_step);
    }

    // 计算每个位置的平均值
    // 使用操作名称序列作为对齐依据
    let stats = calculate_average_stats(&step_operations)?;

    println!("\nCalculated statistics for {} unique operations", stats.len());

    // 写入 CSV
    write_stats_to_csv(&stats, output_file)?;

    // 打印预览
    print_stats_preview(&stats, 10);

    Ok(())
}

/// 每个参考位置的累计统计数据
struct PositionStats {
    total_start: f64,
    total_end: f64,
    total_duration: f64,
    total_bubble: f64,
    count: usize,
}

impl Default for PositionStats {
    fn default() -> Self {
        Self {
            total_start: 0.0,
            total_end: 0.0,
            total_duration: 0.0,
            total_bubble: 0.0,
            count: 0,
        }
    }
}

/// 计算跨 ProfileStep 的平均统计
fn calculate_average_stats(step_operations: &[Vec<GpuOperation>]) -> Result<Vec<ProfileStatsRecord>, Box<dyn Error>> {
    if step_operations.is_empty() {
        return Err("No ProfileStep data available".into());
    }

    // 选择 GPU 操作数出现次数最多的非空 step 作为参考序列
    // 出现次数最多的操作数代表"典型操作序列"，更具代表性
    use std::collections::HashMap;
    let mut length_counts: HashMap<usize, usize> = HashMap::new();
    for ops in step_operations.iter().filter(|ops| !ops.is_empty()) {
        *length_counts.entry(ops.len()).or_insert(0) += 1;
    }

    let most_common_length = length_counts
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(len, _)| *len)
        .ok_or("All ProfileSteps are empty")?;

    let reference_step = step_operations
        .iter()
        .find(|ops| ops.len() == most_common_length)
        .ok_or("All ProfileSteps are empty")?;

    let num_operations = reference_step.len();
    let num_steps = step_operations.len();

    println!("Reference step has {} operations (most common, appeared {} times), total {} steps", 
             num_operations, length_counts[&most_common_length], num_steps);

    // 预分配每个参考位置的统计数据
    let mut position_stats: Vec<PositionStats> = (0..num_operations)
        .map(|_| PositionStats::default())
        .collect();

    // 遍历所有 step，只处理与参考序列长度相同且操作名称一致的 step
    let mut skipped_count = 0;
    let mut name_mismatch_count = 0;
    for (_step_idx, step_ops) in step_operations.iter().enumerate() {
        // 跳过长度不匹配的 step
        if step_ops.len() != num_operations {
            skipped_count += 1;
            continue;
        }

        // 检查操作名称是否与参考序列一致（名称已在收集阶段标准化）
        let names_match = step_ops.iter()
            .zip(reference_step.iter())
            .all(|(cur_op, ref_op)| cur_op.name == ref_op.name);
        
        if !names_match {
            name_mismatch_count += 1;
            continue;
        }

        let mut prev_end_time = 0.0; // 上一个操作的结束时间

        for (idx, cur_op) in step_ops.iter().enumerate() {
            let stats = &mut position_stats[idx];
            stats.total_start += cur_op.start_time;
            stats.total_end += cur_op.end_time;
            stats.total_duration += cur_op.duration;

            // 空泡时间 = 当前开始 - 上一个操作的结束
            let bubble = cur_op.start_time - prev_end_time;
            stats.total_bubble += bubble.max(0.0); // 确保不为负数

            prev_end_time = cur_op.end_time;
            stats.count += 1;
        }
    }

    if skipped_count > 0 {
        println!(
            "Skipped {} steps due to operation count mismatch (expected {})",
            skipped_count, num_operations
        );
    }

    if name_mismatch_count > 0 {
        println!(
            "Skipped {} steps due to operation name mismatch",
            name_mismatch_count
        );
    }

    // 生成最终统计结果
    let mut stats: Vec<ProfileStatsRecord> = Vec::new();

    for (idx, ref_op) in reference_step.iter().enumerate() {
        let pos_stats = &position_stats[idx];

        if pos_stats.count > 0 {
            let count = pos_stats.count as f64;
            stats.push(ProfileStatsRecord {
                operation_name: ref_op.name.clone(),
                avg_start_time_us: pos_stats.total_start / count,
                avg_end_time_us: pos_stats.total_end / count,
                avg_duration_us: pos_stats.total_duration / count,
                bubble_time_us: pos_stats.total_bubble / count,
            });
        }
    }

    Ok(stats)
}

/// 将统计结果写入 CSV
fn write_stats_to_csv(stats: &[ProfileStatsRecord], output_file: &str) -> Result<(), Box<dyn Error>> {
    println!("Writing statistics to CSV file: {}", output_file);
    let csv_file = File::create(output_file)?;
    let mut wtr = csv::Writer::from_writer(BufWriter::new(csv_file));

    for record in stats {
        wtr.serialize(record)?;
    }

    wtr.flush()?;
    println!("Successfully wrote {} records to {}", stats.len(), output_file);

    Ok(())
}

/// 打印预览
fn print_stats_preview(stats: &[ProfileStatsRecord], count: usize) {
    if !stats.is_empty() {
        println!("\n--- Preview (first {} records) ---", count.min(stats.len()));
        println!("{:<50} {:>12} {:>12} {:>12} {:>12}", "Operation", "Start(us)", "End(us)", "Dur(us)", "Bubble(us)");
        println!("{}", "-".repeat(98));
        
        for record in stats.iter().take(count) {
            let name = if record.operation_name.len() > 47 {
                format!("{}...", &record.operation_name[..47])
            } else {
                record.operation_name.clone()
            };
            println!(
                "{:<50} {:>12.3} {:>12.3} {:>12.3} {:>12.3}",
                name,
                record.avg_start_time_us,
                record.avg_end_time_us,
                record.avg_duration_us,
                record.bubble_time_us
            );
        }
    }
}