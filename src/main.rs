mod args;
mod implementations;
mod matrix;
mod multiplier;
mod parse;
mod sources;

use std::fs;
use std::path::Path;

use clap::Parser;

use args::Args;
use matrix::Matrix;
use multiplier::implementation;
use multiplier::{MultiplierInfo, MultiplierStat};

type Error = dyn std::error::Error;
type Result<T> = std::result::Result<T, Box<Error>>;

fn main() {
    let cli = Args::parse();

    let (m1, m2) = match parse::parse_file(Path::new(&cli.input)) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("unable to parse given input file: {}", e);
            return;
        }
    };

    let mut multiplier = match implementation(cli.mode) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("unable to create multiplier: {}", e);
            return;
        }
    };

    match multiplier.info() {
        Ok(MultiplierInfo::OnDeviceMultiplier) => println!("multiplication does not use OpenCl"),
        Ok(MultiplierInfo::OpenClMultiplier {
            device_name,
            platform_name,
        }) => {
            println!("Platform: {}", platform_name);
            println!("Device: {}", device_name);
        }
        Err(e) => {
            eprintln!("unable to get multiplier info, {}", e);
            return;
        }
    }

    let res = match multiplier.multiply(&m1, &m2) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("unable to multiply matrices: {}", e);
            return;
        }
    };

    // unwrap is safe because multiply succeeded
    let MultiplierStat {
        total_time,
        gpu_time,
    } = multiplier.stat().unwrap();

    println!("Total time: {}", total_time);
    println!("Kernel time: {}", gpu_time);

    if let Err(e) = fs::write(cli.output, res.to_string()) {
        eprintln!("unable to write results, {}", e);
    }
}
