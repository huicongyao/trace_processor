use regex::Regex;
use serde::Deserialize;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::BufWriter;

use crate::common::{load_trace_json, TraceEvent};

/// Supported inference frameworks.
#[derive(Debug, Clone, Copy)]
pub enum Framework {
    Sglang,
    Vllm,
    Fastdeploy,
}

impl fmt::Display for Framework {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Framework::Sglang => write!(f, "sglang"),
            Framework::Vllm => write!(f, "vllm"),
            Framework::Fastdeploy => write!(f, "fastdeploy"),
        }
    }
}

impl Framework {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "sglang" => Some(Framework::Sglang),
            "vllm" => Some(Framework::Vllm),
            "fastdeploy" | "fd" => Some(Framework::Fastdeploy),
            _ => None,
        }
    }
}

/// Configuration for decode step analysis.
pub struct DecodeStepsConfig {
    pub framework: Framework,
    pub input_file: String,
    pub output_csv: Option<String>,
    pub min_ms: f64,
    pub max_ms: f64,
}

/// Statistics computed from decode step latencies.
pub struct DecodeStats {
    pub count: usize,
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Extract decode step latencies from sglang trace.
/// Uses time intervals between consecutive 'get_next_batch_to_run' event starts.
/// Only uses the first 50% of the time range to ensure steps are fully loaded.
fn parse_sglang_steps(events: &[serde_json::Value]) -> Vec<f64> {
    const TARGET: &str = "python/sglang/srt/managers/scheduler.py(2071): get_next_batch_to_run";
    extract_interval_latencies(events, TARGET)
}

/// Extract decode step latencies from vllm trace.
/// Uses time intervals between consecutive 'step_with_batch_queue' event starts.
/// Only uses the first 50% of the time range to ensure steps are fully loaded.
fn parse_vllm_steps(events: &[serde_json::Value]) -> Vec<f64> {
    const TARGET: &str = "vllm/v1/engine/core.py(421): step_with_batch_queue";
    extract_interval_latencies(events, TARGET)
}

/// Shared logic for sglang/vllm: collect timestamps of a named event, use first 50%
/// of the time range, then compute consecutive intervals in ms.
fn extract_interval_latencies(events: &[serde_json::Value], target_name: &str) -> Vec<f64> {
    let mut timestamps: Vec<f64> = Vec::new();

    for event_value in events {
        let event: TraceEvent = match TraceEvent::deserialize(event_value) {
            Ok(e) => e,
            Err(_) => continue,
        };

        if event.name == target_name && event.ph.as_deref() == Some("X") {
            if let Some(ts) = event.ts {
                timestamps.push(ts);
            }
        }
    }

    if timestamps.len() < 2 {
        return Vec::new();
    }

    timestamps.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    // Only keep the first 50% of the time range.
    let time_range = timestamps[timestamps.len() - 1] - timestamps[0];
    let cutoff = timestamps[0] + time_range / 2.0;
    timestamps.retain(|&ts| ts <= cutoff);

    // Compute intervals between consecutive timestamps (μs → ms).
    timestamps
        .windows(2)
        .map(|w| (w[1] - w[0]) / 1000.0)
        .collect()
}

/// Extract decode step latencies from fastdeploy trace.
/// Uses the `dur` field from `ProfileStep#N[...ms]` events directly.
fn parse_fastdeploy_steps(events: &[serde_json::Value]) -> Vec<f64> {
    let pattern = Regex::new(r"^ProfileStep#\d+\[[\d.]+\s*ms\]").unwrap();
    let mut durations: Vec<f64> = Vec::new();

    for event_value in events {
        let event: TraceEvent = match TraceEvent::deserialize(event_value) {
            Ok(e) => e,
            Err(_) => continue,
        };

        if pattern.is_match(&event.name) && event.ph.as_deref() == Some("X") {
            if let Some(dur) = event.dur {
                durations.push(dur / 1000.0); // μs → ms
            }
        }
    }

    durations
}

/// Filter latencies to keep only valid decode steps within [min_ms, max_ms].
fn filter_decode_steps(latencies: &[f64], min_ms: f64, max_ms: f64) -> Vec<f64> {
    latencies
        .iter()
        .copied()
        .filter(|&lat| lat >= min_ms && lat <= max_ms)
        .collect()
}

/// Compute the p-th percentile using linear interpolation (matches numpy default).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    assert!(!sorted.is_empty());
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = p / 100.0 * (sorted.len() - 1) as f64;
    let lower = rank.floor() as usize;
    let upper = rank.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let frac = rank - lower as f64;
        sorted[lower] * (1.0 - frac) + sorted[upper] * frac
    }
}

/// Compute statistics from a list of latencies.
fn compute_statistics(latencies: &[f64]) -> Option<DecodeStats> {
    if latencies.is_empty() {
        return None;
    }

    let mut sorted = latencies.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let n = sorted.len() as f64;
    let mean = sorted.iter().sum::<f64>() / n;
    let variance = sorted.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;

    Some(DecodeStats {
        count: sorted.len(),
        mean,
        std_dev: variance.sqrt(),
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        median: percentile(&sorted, 50.0),
        p90: percentile(&sorted, 90.0),
        p95: percentile(&sorted, 95.0),
        p99: percentile(&sorted, 99.0),
    })
}

/// Print statistics in the same format as the Python script.
fn print_statistics(stats: &DecodeStats, name: &str) {
    println!("\n{}", "=".repeat(60));
    println!("{} Decode Step Statistics", name);
    println!("{}", "=".repeat(60));
    println!("Count:    {}", stats.count);
    println!("Mean:     {:.3} ms", stats.mean);
    println!("Std Dev:  {:.3} ms", stats.std_dev);
    println!("Min:      {:.3} ms", stats.min);
    println!("Max:      {:.3} ms", stats.max);
    println!("Median:   {:.3} ms", stats.median);
    println!("P90:      {:.3} ms", stats.p90);
    println!("P95:      {:.3} ms", stats.p95);
    println!("P99:      {:.3} ms", stats.p99);
}

/// Write latencies to a single-column CSV file.
fn write_latencies_csv(latencies: &[f64], output_file: &str) -> Result<(), Box<dyn Error>> {
    println!("Writing latencies to CSV: {}", output_file);
    let file = File::create(output_file)?;
    let mut wtr = csv::Writer::from_writer(BufWriter::new(file));

    wtr.write_record(["latency_ms"])?;
    for &lat in latencies {
        wtr.write_record([format!("{:.6}", lat)])?;
    }
    wtr.flush()?;

    println!("Successfully wrote {} records to {}", latencies.len(), output_file);
    Ok(())
}

/// Main entry point: analyze decode step latencies for a single framework.
pub fn analyze_decode_steps(config: &DecodeStepsConfig) -> Result<(), Box<dyn Error>> {
    let json = load_trace_json(&config.input_file)?;
    let trace_events = json["traceEvents"]
        .as_array()
        .ok_or("traceEvents not found or not an array")?;

    println!("\nExtracting {} decode step latencies...", config.framework);

    let raw_latencies = match config.framework {
        Framework::Sglang => parse_sglang_steps(&trace_events),
        Framework::Vllm => parse_vllm_steps(&trace_events),
        Framework::Fastdeploy => parse_fastdeploy_steps(&trace_events),
    };

    println!(
        "\nRaw count (before filtering): {} steps",
        raw_latencies.len()
    );

    let filtered = filter_decode_steps(&raw_latencies, config.min_ms, config.max_ms);

    println!(
        "Filtered count (keeping {:.1}-{:.1}ms): {} steps (removed {})",
        config.min_ms,
        config.max_ms,
        filtered.len(),
        raw_latencies.len() - filtered.len()
    );

    match compute_statistics(&filtered) {
        Some(stats) => {
            print_statistics(&stats, &config.framework.to_string());
        }
        None => {
            println!("\n{}: No data available", config.framework);
        }
    }

    if let Some(ref csv_path) = config.output_csv {
        write_latencies_csv(&filtered, csv_path)?;
    }

    Ok(())
}
