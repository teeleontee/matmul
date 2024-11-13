use super::args::Mode;
use super::Matrix;
use super::Result;

use super::implementations::{BasicMultiplier, EasyMultiplier, HardMultiplier, MediumMultiplier};

/// Anyone who implements this trait will have the ability to multiply matrices
pub trait Multiplier {
    /// Info on the devices that are performing multiplication
    fn info(&self) -> Result<MultiplierInfo>;
    /// Multiply two matrices
    fn multiply(&mut self, m1: &Matrix, m2: &Matrix) -> Result<Matrix>;
    /// Gives statistics on the last run of multiplier.
    ///
    /// Is `None` if the [Multiply] hasn't yet been used
    fn stat(&self) -> Option<MultiplierStat>;
}

/// Matrix multiplication can happen on device or on the gpu
pub enum MultiplierInfo {
    OnDeviceMultiplier,
    OpenClMultiplier {
        /// If matrix multiplication is done using OpenCl then this is the name of the device
        device_name: String,
        /// If matrix multiplication is done using OpenCl then this is the name of the platform
        platform_name: String,
    },
}

/// Run statistics for multiplication
pub struct MultiplierStat {
    /// Total time of execution
    pub total_time: u64,
    /// Kernel time, is zero if multiplication happens on device
    pub gpu_time: u64,
}

/// Provided a mode return a multipliplier trait object
pub fn implementation(mode: Mode) -> Result<Box<dyn Multiplier>> {
    match mode {
        Mode::Basic => Ok(Box::new(BasicMultiplier::default())),
        Mode::Easy { device_type, index } => {
            let device_type = device_type.unwrap_or_default();
            let index = index.unwrap_or_default();
            Ok(Box::new(EasyMultiplier::new(device_type, index)?))
        }
        Mode::Medium { device_type, index } => {
            let device_type = device_type.unwrap_or_default();
            let index = index.unwrap_or_default();
            Ok(Box::new(MediumMultiplier::new(device_type, index)?))
        }
        Mode::Hard { device_type, index } => {
            let device_type = device_type.unwrap_or_default();
            let index = index.unwrap_or_default();
            Ok(Box::new(HardMultiplier::new(device_type, index)?))
        }
    }
}
