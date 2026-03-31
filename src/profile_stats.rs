use serde::{Deserialize, Serialize};

use std::error::Error;
use std::fs::File;
use std::io::BufWriter;

use crate::common::{load_trace_json, parse_time_from_string, TraceEvent};

/// ProfileStep event.
#[derive(Debug, Clone)]
pub struct ProfileStep {
    pub name: String,
    pub start_time: f64,
    pub end_time: f64,
}

/// GPU operation record.
#[derive(Debug, Clone)]
pub struct GpuOperation {
    pub name: String,
    pub start_time: f64,
    pub end_time: f64,
    pub duration: f64,
}

/// Output statistics record.
#[derive(Debug, Serialize)]
pub struct ProfileStatsRecord {
    pub operation_name: String,
    pub avg_start_time_us: f64,
    pub avg_end_time_us: f64,
    pub avg_duration_us: f64,
    /// Bubble time: gap between the end of the previous operation and the start of the current one.
    pub bubble_time_us: f64,
}

/// Normalize operation name by stripping the trailing dynamic duration suffix.
/// e.g. "MEMCPY_DtoH[2.464 us]" -> "MEMCPY_DtoH"
/// e.g. "kernel_name[123.456 us]" -> "kernel_name"
fn normalize_op_name(name: &str) -> &str {
    // Find the last '[' and check if it is a duration suffix.
    if let Some(bracket_pos) = name.rfind('[') {
        let suffix = &name[bracket_pos..];
        // Match "[<number> us]" or "[<number> ms]" format.
        if suffix.ends_with(" us]") || suffix.ends_with(" ms]") {
            return &name[..bracket_pos];
        }
    }
    name
}

/// Compute average GPU operation statistics within ProfileSteps from a JSON trace file.
///
/// # Arguments
/// * `input_file` - Path to the input JSON trace file.
/// * `output_file` - Path to the output CSV statistics file.
/// * `trim_start_kernel` - Optional kernel name to start counting from within each ProfileStep.
/// * `decode_max_duration_ms` - Maximum duration threshold (ms); steps exceeding this are treated as prefill and filtered out.
pub fn analyze_profile_stats(
    input_file: &str,
    output_file: &str,
    trim_start_kernel: Option<&str>,
    decode_max_duration_ms: f64,
) -> Result<(), Box<dyn Error>> {
    println!("Processing JSON file: {}", input_file);

    let json = load_trace_json(input_file)?;
    let trace_events = json["traceEvents"]
        .as_array()
        .ok_or("traceEvents not found or not an array")?;

    // First pass: collect all ProfileSteps and GPU operations.
    let mut profile_steps: Vec<ProfileStep> = Vec::new();
    let mut gpu_operations: Vec<GpuOperation> = Vec::new();

    for event_value in trace_events {
        let event: TraceEvent = match TraceEvent::deserialize(event_value) {
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
                        // Normalize name: strip dynamic duration suffix since timing is derived from start/end.
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

    // Sort by start time.
    profile_steps.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());
    gpu_operations.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());

    // Filter out prefill steps (duration exceeds threshold).
    // Decode typically takes 10-20ms; prefill takes 40-50ms.
    let decode_max_duration_us = decode_max_duration_ms * 1000.0; // convert to microseconds
    let total_before_filter = profile_steps.len();
    profile_steps.retain(|step| {
        let duration = step.end_time - step.start_time;
        duration <= decode_max_duration_us
    });

    let filtered_count = total_before_filter - profile_steps.len();
    println!(
        "Filtered out {} prefill steps (duration > {}ms), {} decode steps remaining",
        filtered_count,
        decode_max_duration_ms,
        profile_steps.len()
    );

    if profile_steps.is_empty() {
        return Err("No decode ProfileStep events found after filtering".into());
    }

    // For each ProfileStep, collect GPU operations within its time range
    // and convert to relative timestamps.
    let mut step_operations: Vec<Vec<GpuOperation>> = Vec::new();

    for step in &profile_steps {
        let mut ops_in_step: Vec<GpuOperation> = Vec::new();

        for op in &gpu_operations {
            // GPU operation falls within the ProfileStep time range.
            if op.start_time >= step.start_time && op.end_time <= step.end_time {
                // Convert to relative time (relative to ProfileStep start).
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

        // If a trim kernel is specified, discard operations before it.
        if let Some(trim_kernel) = trim_start_kernel {
            if let Some(start_idx) = ops_in_step
                .iter()
                .position(|op| op.name.contains(trim_kernel))
            {
                // Get the new base timestamp.
                let new_base_time = ops_in_step[start_idx].start_time;

                // Trim and recompute relative timestamps.
                ops_in_step = ops_in_step[start_idx..]
                    .iter()
                    .map(|op| GpuOperation {
                        name: op.name.clone(),
                        start_time: op.start_time - new_base_time,
                        end_time: op.end_time - new_base_time,
                        duration: op.duration,
                    })
                    .collect();

                println!(
                    "ProfileStep '{}': {} GPU operations (trimmed from '{}' at index {})",
                    step.name,
                    ops_in_step.len(),
                    trim_kernel,
                    start_idx
                );
            } else {
                println!(
                    "ProfileStep '{}': {} GPU operations (trim kernel '{}' not found)",
                    step.name,
                    ops_in_step.len(),
                    trim_kernel
                );
            }
        } else {
            println!(
                "ProfileStep '{}': {} GPU operations (no trimming)",
                step.name,
                ops_in_step.len()
            );
        }

        step_operations.push(ops_in_step);
    }

    // Compute per-position averages.
    // Use the operation name sequence as the alignment reference.
    let stats = calculate_average_stats(&step_operations)?;

    println!(
        "\nCalculated statistics for {} unique operations",
        stats.len()
    );

    // Write to CSV.
    write_stats_to_csv(&stats, output_file)?;

    // Print preview.
    print_stats_preview(&stats, 10);

    Ok(())
}

/// Accumulated statistics for each reference position.
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

/// Compute average statistics across ProfileSteps.
fn calculate_average_stats(
    step_operations: &[Vec<GpuOperation>],
) -> Result<Vec<ProfileStatsRecord>, Box<dyn Error>> {
    if step_operations.is_empty() {
        return Err("No ProfileStep data available".into());
    }

    // Select the most common operation count among non-empty steps as the reference sequence.
    // The most frequent count represents the "typical operation sequence".
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

    println!(
        "Reference step has {} operations (most common, appeared {} times), total {} steps",
        num_operations, length_counts[&most_common_length], num_steps
    );

    // Pre-allocate statistics for each reference position.
    let mut position_stats: Vec<PositionStats> = (0..num_operations)
        .map(|_| PositionStats::default())
        .collect();

    // Iterate over all steps; only process those matching the reference length and names.
    let mut skipped_count = 0;
    let mut name_mismatch_count = 0;
    for (_step_idx, step_ops) in step_operations.iter().enumerate() {
        // Skip steps with mismatched operation count.
        if step_ops.len() != num_operations {
            skipped_count += 1;
            continue;
        }

        // Verify operation names match the reference sequence (names already normalized).
        let names_match = step_ops
            .iter()
            .zip(reference_step.iter())
            .all(|(cur_op, ref_op)| cur_op.name == ref_op.name);

        if !names_match {
            name_mismatch_count += 1;
            continue;
        }

        let mut prev_end_time = 0.0; // end time of previous operation

        for (idx, cur_op) in step_ops.iter().enumerate() {
            let stats = &mut position_stats[idx];
            stats.total_start += cur_op.start_time;
            stats.total_end += cur_op.end_time;
            stats.total_duration += cur_op.duration;

            // Bubble time = current start - previous end.
            let bubble = cur_op.start_time - prev_end_time;
            stats.total_bubble += bubble.max(0.0); // clamp to non-negative

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

    // Generate final statistics.
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

/// Write statistics to a CSV file.
fn write_stats_to_csv(
    stats: &[ProfileStatsRecord],
    output_file: &str,
) -> Result<(), Box<dyn Error>> {
    println!("Writing statistics to CSV file: {}", output_file);
    let csv_file = File::create(output_file)?;
    let mut wtr = csv::Writer::from_writer(BufWriter::new(csv_file));

    for record in stats {
        wtr.serialize(record)?;
    }

    wtr.flush()?;
    println!(
        "Successfully wrote {} records to {}",
        stats.len(),
        output_file
    );

    Ok(())
}

/// Print a preview of statistics.
fn print_stats_preview(stats: &[ProfileStatsRecord], count: usize) {
    if !stats.is_empty() {
        println!(
            "\n--- Preview (first {} records) ---",
            count.min(stats.len())
        );
        println!(
            "{:<50} {:>12} {:>12} {:>12} {:>12}",
            "Operation", "Start(us)", "End(us)", "Dur(us)", "Bubble(us)"
        );
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
