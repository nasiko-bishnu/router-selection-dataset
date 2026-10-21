//! Nasiko's current router (llm-router on main) picking a tier for every question of the measured 600.
//!
//! Copied into a nasiko checkout as llm-router/examples/nasiko_router_eval.rs by test_nasiko.py.
//! For each question it calls main's own `routing::classify` (regex request type, then a Thompson-sampled
//! tier over the (tier, request type) cells), with a seeded RNG.
//!   MODE=cold   : no learned cells, only the cold-start priors (a fresh deployment)
//!   MODE=online : after each pick the chosen cell learns with `update_cell` from the measured label
//!                 (1 when the picked tier is at or above it, else 0): best-case, instant feedback
//! Reads EVAL_SET (jsonl: query, label LOW/MID/HIGH) and writes OUT (jsonl: tier, request_type, latency_us).
use std::io::{BufRead, Write};
use std::time::Instant;

use nasiko_llm_router::routing::classifier::{Cell, CellMap, Tier, classify, tier_prior, update_cell};
use rand::SeedableRng;
use rand::rngs::StdRng;

fn name(t: Tier) -> &'static str {
    match t {
        Tier::Tier3 => "low",
        Tier::Tier2 => "mid",
        Tier::Tier1 => "high",
    }
}

fn rank(t: &str) -> u8 {
    match t {
        "low" => 0,
        "mid" => 1,
        _ => 2,
    }
}

fn main() {
    let path = std::env::var("EVAL_SET").expect("EVAL_SET");
    let out_path = std::env::var("OUT").expect("OUT");
    let seed: u64 = std::env::var("SEED").unwrap_or_else(|_| "0".into()).parse().expect("SEED");
    let online = std::env::var("MODE").unwrap_or_else(|_| "cold".into()) == "online";
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cells = CellMap::new();
    let mut out = std::io::BufWriter::new(std::fs::File::create(&out_path).expect("create OUT"));
    for line in std::io::BufReader::new(std::fs::File::open(&path).expect("open EVAL_SET")).lines() {
        let line = line.expect("read line");
        if line.trim().is_empty() {
            continue;
        }
        let row: serde_json::Value = serde_json::from_str(&line).expect("valid json line");
        let query = row["query"].as_str().expect("query");
        let started = Instant::now();
        let (tier, rt) = classify(query, "eval", &cells, &mut rng);
        let latency_us = started.elapsed().as_micros() as u64;
        if online {
            let label = row["label"].as_str().expect("label").to_lowercase();
            let obs = if rank(name(tier)) >= rank(&label) { 1.0 } else { 0.0 };
            let cell = cells.get(&(tier, rt)).copied().unwrap_or(Cell { quality_mean: tier_prior(tier, rt), samples: 0 });
            cells.insert((tier, rt), update_cell(cell, obs));
        }
        let rec = serde_json::json!({"tier": name(tier), "request_type": rt.as_str(), "latency_us": latency_us});
        writeln!(out, "{rec}").expect("write OUT");
    }
    out.flush().expect("flush OUT");
}
