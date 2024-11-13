use clap::{Parser, Subcommand, ValueEnum};

#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
#[clap(rename_all = "lowercase")]
pub enum DeviceType {
    /// Dedicated gpu
    DGpu = 0,
    /// Integrated gpu
    IGpu = 1,
    /// Any gpu
    Gpu = 2,
    /// Any cpu
    Cpu = 3,
    /// Any device found
    #[default]
    All = 4,
}

/// 4 implementations are provided as of time of writing
#[derive(Debug, Subcommand)]
pub enum Mode {
    /// Basic implementation is just 3 loops on the host
    Basic,
    /// Easy implementation is a naive implementation that uses the gpu
    Easy {
        device_type: Option<DeviceType>,
        index: Option<usize>,
    },
    /// Medium implementation is a less naive implementation that uses local memory on the gpu
    Medium {
        device_type: Option<DeviceType>,
        index: Option<usize>,
    },
    /// Hard is an okay implementation that optimized thread throughput
    Hard {
        device_type: Option<DeviceType>,
        index: Option<usize>,
    },
}

#[derive(Debug, Parser)]
#[command(about = "Matrix multiplication on the GPU", long_about = None)]
pub struct Args {
    /// Input file with the matrices that are to be multiplied
    pub input: String,
    /// Output file where the result of the multiplication will be
    pub output: String,
    /// Choose where to multiply the matrices
    #[command(subcommand)]
    pub mode: Mode,
}
