use opencl3::command_queue::CommandQueue;
use opencl3::command_queue::CL_QUEUE_PROFILING_ENABLE;
use opencl3::context::Context;
use opencl3::device::Device;
use opencl3::error_codes::ClError;
use opencl3::kernel::Kernel;
use opencl3::memory::{create_buffer, Buffer};
use opencl3::memory::{CL_MEM_READ_ONLY, CL_MEM_WRITE_ONLY};
use opencl3::platform::Platform;
use opencl3::program::Program;
use opencl3::program::CL_STD_3_0;
use opencl3::types::cl_float;
use opencl3::types::{CL_FALSE, CL_TRUE};

use crate::args::DeviceType;
use crate::multiplier::{Multiplier, MultiplierInfo, MultiplierStat};
use crate::sources;
use crate::Matrix;
use crate::Result;

use super::TimeStat;

#[derive(Clone)]
pub struct EasyMultiplier {
    device: Device,
    stat: Option<TimeStat>,
}

impl EasyMultiplier {
    pub fn new(device_type: DeviceType, index: usize) -> Result<Self> {
        let device = super::get_device(device_type, index)?;

        Ok(Self { device, stat: None })
    }

    fn platform(&self) -> Result<Platform> {
        let platform = self.device.platform()?;
        Ok(Platform::new(platform))
    }
}

impl Multiplier for EasyMultiplier {
    fn multiply(&mut self, m1: &Matrix, m2: &Matrix) -> Result<Matrix> {
        let context = Context::from_device(&self.device)?;
        let queue =
            CommandQueue::create_default_with_properties(&context, CL_QUEUE_PROFILING_ENABLE, 0)?;

        let m1_size = std::mem::size_of::<cl_float>() * m1.rows * m1.cols;
        let mut m1_buf: Buffer<cl_float> = unsafe {
            let mem = create_buffer(
                context.get(),
                CL_MEM_READ_ONLY,
                m1_size,
                std::ptr::null_mut(),
            )
            .map_err(ClError)?;
            Buffer::new(mem)
        };

        let m2_size = std::mem::size_of::<cl_float>() * m2.rows * m2.cols;
        let mut m2_buf: Buffer<cl_float> = unsafe {
            let mem = create_buffer(
                context.get(),
                CL_MEM_READ_ONLY,
                m2_size,
                std::ptr::null_mut(),
            )
            .map_err(ClError)?;
            Buffer::new(mem)
        };

        let mut res = Matrix::create_empty(m1.rows, m2.cols);

        let m3_size = std::mem::size_of::<cl_float>() * m1.rows * m2.cols;
        let mut m3_buf: Buffer<cl_float> = unsafe {
            let mem = create_buffer(
                context.get(),
                CL_MEM_WRITE_ONLY,
                m3_size,
                std::ptr::null_mut(),
            )
            .map_err(ClError)?;
            Buffer::new(mem)
        };

        let write_event1 =
            unsafe { queue.enqueue_write_buffer(&mut m1_buf, CL_FALSE, 0, m1.data.as_ref(), &[])? };

        let write_event2 =
            unsafe { queue.enqueue_write_buffer(&mut m2_buf, CL_FALSE, 0, m2.data.as_ref(), &[])? };

        let write_event3 = unsafe {
            queue.enqueue_write_buffer(&mut m3_buf, CL_FALSE, 0, res.data.as_ref(), &[])?
        };

        let program =
            Program::create_and_build_from_source(&context, sources::EASY_SOURCE, CL_STD_3_0)?;
        let kernel = Kernel::create(&program, sources::KERNEL_NAME)?;

        unsafe {
            kernel.set_arg(0, &m1_buf)?;
            kernel.set_arg(1, &m2_buf)?;
            kernel.set_arg(2, &m3_buf)?;
            kernel.set_arg(3, &m2.cols)?;
            kernel.set_arg(4, &m1.rows)?;
            kernel.set_arg(5, &m1.cols)?;
        }

        let kernel_event = unsafe {
            let global_work_sizes = [m2.cols, m1.rows];
            queue.enqueue_nd_range_kernel(
                kernel.get(),
                2,
                std::ptr::null_mut(),
                global_work_sizes.as_ptr(),
                std::ptr::null_mut(),
                &[],
            )?
        };
        let read_event =
            unsafe { queue.enqueue_read_buffer(&m3_buf, CL_TRUE, 0, &mut res.data, &[])? };

        let read_time = super::get_profiling_info(read_event)?;
        let kernel_time = super::get_profiling_info(kernel_event)?;
        let write_time1 = super::get_profiling_info(write_event1)?;
        let write_time2 = super::get_profiling_info(write_event2)?;
        let write_time3 = super::get_profiling_info(write_event3)?;

        let total_time = write_time1 + write_time2 + write_time3 + kernel_time + read_time;

        self.stat = Some(TimeStat {
            total_time,
            kernel_time,
        });

        Ok(res)
    }

    fn info(&self) -> Result<MultiplierInfo> {
        let device_name = self.device.name()?;
        let platform_name = self.platform()?.name()?;

        let res = MultiplierInfo::OpenClMultiplier {
            device_name,
            platform_name,
        };

        Ok(res)
    }

    fn stat(&self) -> Option<MultiplierStat> {
        self.stat.map(|stat| MultiplierStat {
            total_time: stat.total_time,
            gpu_time: stat.kernel_time,
        })
    }
}

mod tests {
    use crate::implementations::EasyMultiplier;

    #[allow(dead_code)]
    struct Fixture {
        multiplier: EasyMultiplier,
    }

    impl Default for Fixture {
        fn default() -> Self {
            let multiplier = EasyMultiplier::new(crate::args::DeviceType::All, 0)
                .expect("unable to create multiplier");
            Fixture { multiplier }
        }
    }

    #[test]
    fn basic_multiplication_test() {
        let fixture = Fixture::default();
        let _multiplier = fixture.multiplier;
    }
}
