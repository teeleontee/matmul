use std::io;

use opencl3::device::Device;
use opencl3::device::{CL_DEVICE_TYPE_ALL, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU};
use opencl3::types::cl_device_type;
use opencl3::{
    device::get_all_devices,
    error_codes::ClError,
    event::{
        get_event_profiling_info, Event, CL_PROFILING_COMMAND_END, CL_PROFILING_COMMAND_START,
    },
};

use super::args::DeviceType;
use super::Result;

mod basic;
mod easy;
mod hard;
mod medium;
#[rustfmt::skip]
#[cfg(test)]
mod tests;

pub use basic::BasicMultiplier;
pub use easy::EasyMultiplier;
pub use hard::HardMultiplier;
pub use medium::MediumMultiplier;

#[derive(Clone, Copy)]
struct TimeStat {
    total_time: u64,
    kernel_time: u64,
}

impl From<DeviceType> for cl_device_type {
    fn from(value: DeviceType) -> Self {
        match value {
            DeviceType::All => CL_DEVICE_TYPE_ALL,
            DeviceType::Cpu => CL_DEVICE_TYPE_CPU,
            DeviceType::Gpu | DeviceType::IGpu | DeviceType::DGpu => CL_DEVICE_TYPE_GPU,
        }
    }
}

fn get_profiling_info(event: Event) -> Result<u64> {
    let start: u64 = get_event_profiling_info(event.get(), CL_PROFILING_COMMAND_START)
        .map_err(ClError)?
        .into();
    let end: u64 = get_event_profiling_info(event.get(), CL_PROFILING_COMMAND_END)
        .map_err(ClError)?
        .into();

    let res = (end - start) / (1e6 as u64);

    Ok(res)
}

fn get_device(device_type: DeviceType, device_index: usize) -> Result<Device> {
    let devices = get_all_devices(device_type.into())?
        .iter()
        .map(|id| Device::from(*id))
        .collect::<Vec<_>>();

    if device_type == DeviceType::All {
        let device = devices.get(device_index).ok_or_else(|| {
            let msg = format!("invalid `device_index`: {device_index}, out of bounds");
            io::Error::new(io::ErrorKind::Other, msg)
        })?;

        return Ok(*device);
    }

    let mut filter_dev_type = vec![];

    for device in devices.iter() {
        if device.dev_type()? == device_type.into() {
            filter_dev_type.push(*device);
        }
    }

    match device_type {
        DeviceType::DGpu => {
            let mut filter_unified_memory = vec![];

            for &device in devices.iter() {
                let unified_mem = device.host_unified_memory()?;
                if unified_mem {
                    filter_unified_memory.push(device);
                }
            }

            let res = filter_unified_memory.get(device_index).ok_or_else(|| {
                let msg = format!("invalid `device_index`: {device_index}, out of bounds");
                io::Error::new(io::ErrorKind::Other, msg)
            })?;

            Ok(*res)
        }
        DeviceType::IGpu => {
            let mut filter_unified_memory = vec![];

            for &device in devices.iter() {
                let unified_mem = device.host_unified_memory()?;
                if !unified_mem {
                    filter_unified_memory.push(device);
                }
            }

            let res = filter_unified_memory.get(device_index).ok_or_else(|| {
                let msg = format!("invalid `device_index`: {device_index}, out of bounds");
                io::Error::new(io::ErrorKind::Other, msg)
            })?;

            Ok(*res)
        }
        _ => {
            let res = filter_dev_type.get(device_index).ok_or_else(|| {
                let msg = format!("invalid `device_index`: {device_index}, out of bounds");
                io::Error::new(io::ErrorKind::Other, msg)
            })?;

            Ok(*res)
        }
    }
}
