use serde::Deserialize;
use serde_json::Value;
use std::error::Error;
use std::fs::File;
use std::io::BufReader;

/// Trace event structure.
#[derive(Debug, Deserialize)]
pub struct TraceEvent {
    pub name: String,
    #[serde(default)]
    pub cat: Option<String>,
    #[serde(default)]
    pub ph: Option<String>,
    #[serde(default)]
    pub ts: Option<f64>,
    #[serde(default)]
    pub dur: Option<f64>,
    #[serde(default)]
    pub args: Option<TraceArgs>,
}

/// Event arguments.
#[derive(Debug, Deserialize)]
pub struct TraceArgs {
    #[serde(default)]
    pub start_time: Option<String>,
    #[serde(default)]
    pub end_time: Option<String>,
}

/// Parse a time string, e.g. "6609483.000 us".
pub fn parse_time_from_string(time_str: &str) -> Option<f64> {
    time_str
        .trim()
        .split_whitespace()
        .next()
        .and_then(|s| s.parse::<f64>().ok())
}

/// Load and parse a JSON trace file, returning the root Value.
/// Caller should access `json["traceEvents"].as_array()` to get events.
pub fn load_trace_json(input_file: &str) -> Result<Value, Box<dyn Error>> {
    println!("Processing JSON file: {}", input_file);
    let file = File::open(input_file)?;
    let reader = BufReader::new(file);

    println!("Parsing JSON (this may take a while for large files)...");
    let json: Value = serde_json::from_reader(reader)?;

    let count = json["traceEvents"]
        .as_array()
        .ok_or("traceEvents not found or not an array")?
        .len();

    println!("Total events in file: {}", count);
    Ok(json)
}
