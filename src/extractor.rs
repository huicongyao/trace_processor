use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::{BufReader, BufWriter};

/// 追踪事件结构
#[derive(Debug, Deserialize)]
pub struct TraceEvent {
    pub name: String,
    #[serde(default)]
    pub cat: Option<String>,
    #[serde(default)]
    pub ph: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    pub ts: Option<i64>,
    #[serde(default)]
    #[allow(dead_code)]
    pub dur: Option<f64>,
    #[serde(default)]
    pub args: Option<TraceArgs>,
}

/// 事件参数结构
#[derive(Debug, Deserialize)]
pub struct TraceArgs {
    #[serde(default)]
    pub start_time: Option<String>,
    #[serde(default)]
    pub end_time: Option<String>,
}

/// 输出的 Kernel 记录
#[derive(Debug, Serialize)]
pub struct KernelRecord {
    pub kernel_name: String,
    pub start_time_us: f64,
    pub end_time_us: f64,
    pub duration_us: f64,
}

/// 提取配置
pub struct ExtractConfig {
    pub input_file: String,
    pub output_file: String,
    pub start_time: f64,
    pub end_time: f64,
}

/// 解析时间字符串，如 "6609483.000 us"
pub fn parse_time_from_string(time_str: &str) -> Option<f64> {
    time_str
        .trim()
        .split_whitespace()
        .next()
        .and_then(|s| s.parse::<f64>().ok())
}

/// 从 JSON 文件中提取 Kernel 事件
pub fn extract_kernels(config: &ExtractConfig) -> Result<Vec<KernelRecord>, Box<dyn Error>> {
    println!("Processing JSON file: {}", config.input_file);
    println!("Time range: {} us to {} us", config.start_time, config.end_time);

    // 打开并解析 JSON 文件
    let file = File::open(&config.input_file)?;
    let reader = BufReader::new(file);
    
    println!("Parsing JSON (this may take a while for large files)...");
    let json: Value = serde_json::from_reader(reader)?;
    
    // 获取 traceEvents 数组
    let trace_events = json["traceEvents"]
        .as_array()
        .ok_or("traceEvents not found or not an array")?;

    println!("Total events in file: {}", trace_events.len());

    // 收集符合条件的 kernel 记录
    let mut kernel_records: Vec<KernelRecord> = Vec::new();
    let mut processed = 0;
    let total = trace_events.len();

    for event_value in trace_events {
        processed += 1;
        if processed % 100000 == 0 {
            println!("Processed {}/{} events...", processed, total);
        }

        let event: TraceEvent = match serde_json::from_value(event_value.clone()) {
            Ok(e) => e,
            Err(_) => continue,
        };

        // 筛选条件：
        // 1. 类别是 "Kernel"、"Memcpy" 或 "Memset"
        // 2. 阶段是 "X" (完整事件)
        // 3. 有 args 字段，包含 start_time 和 end_time
        if let (Some(cat), Some(ph), Some(args)) = (&event.cat, &event.ph, &event.args) {
            let is_gpu_operation = cat == "Kernel" 
                || cat == "Memcpy" 
                || cat == "Memset";
            
            if is_gpu_operation && ph == "X" {
                if let (Some(start_str), Some(end_str)) = (&args.start_time, &args.end_time) {
                    if let (Some(start), Some(end)) = (
                        parse_time_from_string(start_str),
                        parse_time_from_string(end_str),
                    ) {
                        // 检查时间范围
                        if start >= config.start_time && end <= config.end_time {
                            let duration = end - start;
                            kernel_records.push(KernelRecord {
                                kernel_name: event.name.clone(),
                                start_time_us: start,
                                end_time_us: end,
                                duration_us: duration,
                            });
                        }
                    }
                }
            }
        }
    }

    // 按开始时间排序
    kernel_records.sort_by(|a, b| a.start_time_us.partial_cmp(&b.start_time_us).unwrap());

    println!("Found {} kernel events in the specified time range", kernel_records.len());

    Ok(kernel_records)
}

/// 将 Kernel 记录写入 CSV 文件
pub fn write_to_csv(records: &[KernelRecord], output_file: &str) -> Result<(), Box<dyn Error>> {
    println!("Writing to CSV file: {}", output_file);
    let csv_file = File::create(output_file)?;
    let mut wtr = csv::Writer::from_writer(BufWriter::new(csv_file));

    for record in records {
        wtr.serialize(record)?;
    }

    wtr.flush()?;
    println!("Successfully wrote {} records to {}", records.len(), output_file);

    Ok(())
}

/// 打印预览信息
pub fn print_preview(records: &[KernelRecord], count: usize) {
    if !records.is_empty() {
        println!("\n--- Preview (first {} records) ---", count.min(records.len()));
        for (i, record) in records.iter().take(count).enumerate() {
            println!(
                "{}. {} | {:.3} -> {:.3} us | {:.3} us",
                i + 1,
                record.kernel_name,
                record.start_time_us,
                record.end_time_us,
                record.duration_us
            );
        }
    }
}