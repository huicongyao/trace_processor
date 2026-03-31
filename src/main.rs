mod common;
mod decode_steps;
mod extractor;
mod profile_stats;

use decode_steps::{DecodeStepsConfig, Framework};
use extractor::{extract_kernels, print_preview, write_to_csv, ExtractConfig};
use std::error::Error;

fn print_usage(program: &str) {
    eprintln!("GPU Kernel Extractor - Extract and analyze GPU operations from trace files\n");
    eprintln!("Usage:");
    eprintln!(
        "  {} extract <input_json> <output_csv> <start_time_us,end_time_us>",
        program
    );
    eprintln!("      Extract GPU operations within a specific time range\n");
    eprintln!(
        "  {} stats <input_json> <output_csv> [trim_start_kernel] [decode_max_duration_ms]",
        program
    );
    eprintln!("      Analyze ProfileStep GPU operations and calculate averages");
    eprintln!("      trim_start_kernel: Optional kernel name to start counting from (default: recover_decode_task)");
    eprintln!("                         Use 'none' to disable trimming");
    eprintln!("      decode_max_duration_ms: Maximum duration threshold in ms for decode steps (default: 30)");
    eprintln!(
        "                              ProfileSteps exceeding this are filtered as prefill\n"
    );
    eprintln!(
        "  {} decode-steps <framework> <input_json> [--output-csv <path>] [--min-ms <val>] [--max-ms <val>]",
        program
    );
    eprintln!("      Analyze decode step latency from sglang/vllm/fastdeploy traces");
    eprintln!("      framework: sglang | vllm | fastdeploy");
    eprintln!("      --output-csv: Optional CSV output path for latencies");
    eprintln!("      --min-ms:     Minimum latency filter in ms (default: 10.0)");
    eprintln!("      --max-ms:     Maximum latency filter in ms (default: 30.0)\n");
    eprintln!("Examples:");
    eprintln!(
        "  {} extract naive_spec_2.json output.csv 2684054.000,2687705.250",
        program
    );
    eprintln!("  {} stats naive_spec_2.json profile_stats.csv", program);
    eprintln!(
        "  {} stats naive_spec_2.json profile_stats.csv none 50",
        program
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage(&args[0]);
        std::process::exit(1);
    }

    let command = &args[1];

    match command.as_str() {
        "extract" => {
            if args.len() != 5 {
                eprintln!("Error: 'extract' requires 3 arguments");
                eprintln!(
                    "Usage: {} extract <input_json> <output_csv> <start_time_us,end_time_us>",
                    args[0]
                );
                std::process::exit(1);
            }

            let time_range: Vec<f64> = args[4]
                .split(',')
                .filter_map(|s| s.parse::<f64>().ok())
                .collect();

            if time_range.len() != 2 {
                eprintln!("Invalid time range format. Expected: start,end");
                std::process::exit(1);
            }

            let config = ExtractConfig {
                input_file: args[2].clone(),
                output_file: args[3].clone(),
                start_time: time_range[0],
                end_time: time_range[1],
            };

            println!("Output CSV: {}", config.output_file);
            let kernel_records = extract_kernels(&config)?;
            write_to_csv(&kernel_records, &config.output_file)?;
            print_preview(&kernel_records, 5);
        }

        "stats" => {
            if args.len() < 4 || args.len() > 6 {
                eprintln!("Error: 'stats' requires 2-4 arguments");
                eprintln!("Usage: {} stats <input_json> <output_csv> [trim_start_kernel] [decode_max_duration_ms]", args[0]);
                std::process::exit(1);
            }

            let input_file = &args[2];
            let output_file = &args[3];

            // Parse optional trim_start_kernel argument.
            // Defaults to "recover_decode_task"; pass "none" to disable trimming.
            let trim_start_kernel: Option<&str> = if args.len() >= 5 {
                let kernel = args[4].as_str();
                if kernel.eq_ignore_ascii_case("none") {
                    None
                } else {
                    Some(kernel)
                }
            } else {
                Some("recover_decode_task") // default
            };

            // Parse optional decode_max_duration_ms argument (default: 30ms).
            let decode_max_duration_ms: f64 = if args.len() == 6 {
                args[5].parse().unwrap_or_else(|_| {
                    eprintln!(
                        "Warning: Invalid decode_max_duration_ms '{}', using default 30ms",
                        args[5]
                    );
                    30.0
                })
            } else {
                30.0 // default
            };

            profile_stats::analyze_profile_stats(
                input_file,
                output_file,
                trim_start_kernel,
                decode_max_duration_ms,
            )?;
        }

        "decode-steps" => {
            if args.len() < 4 {
                eprintln!("Error: 'decode-steps' requires at least 2 arguments");
                eprintln!(
                    "Usage: {} decode-steps <framework> <input_json> [--output-csv <path>] [--min-ms <val>] [--max-ms <val>]",
                    args[0]
                );
                std::process::exit(1);
            }

            let framework = Framework::from_str(&args[2]).unwrap_or_else(|| {
                eprintln!(
                    "Unknown framework '{}'. Supported: sglang, vllm, fastdeploy",
                    args[2]
                );
                std::process::exit(1);
            });
            let input_file = args[3].clone();

            // Parse optional flags.
            let mut output_csv: Option<String> = None;
            let mut min_ms: f64 = 10.0;
            let mut max_ms: f64 = 30.0;
            let mut i = 4;
            while i < args.len() {
                match args[i].as_str() {
                    "--output-csv" => {
                        i += 1;
                        output_csv = Some(args.get(i).cloned().unwrap_or_else(|| {
                            eprintln!("Error: --output-csv requires a value");
                            std::process::exit(1);
                        }));
                    }
                    "--min-ms" => {
                        i += 1;
                        min_ms = args
                            .get(i)
                            .and_then(|s| s.parse().ok())
                            .unwrap_or_else(|| {
                                eprintln!("Error: --min-ms requires a numeric value");
                                std::process::exit(1);
                            });
                    }
                    "--max-ms" => {
                        i += 1;
                        max_ms = args
                            .get(i)
                            .and_then(|s| s.parse().ok())
                            .unwrap_or_else(|| {
                                eprintln!("Error: --max-ms requires a numeric value");
                                std::process::exit(1);
                            });
                    }
                    other => {
                        eprintln!("Unknown option: {}", other);
                        std::process::exit(1);
                    }
                }
                i += 1;
            }

            let config = DecodeStepsConfig {
                framework,
                input_file,
                output_csv,
                min_ms,
                max_ms,
            };

            decode_steps::analyze_decode_steps(&config)?;
        }

        _ => {
            eprintln!("Unknown command: {}", command);
            print_usage(&args[0]);
            std::process::exit(1);
        }
    }

    Ok(())
}
