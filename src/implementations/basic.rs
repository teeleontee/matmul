use std::io;
use std::time;

use crate::multiplier::{Multiplier, MultiplierInfo, MultiplierStat};
use crate::Matrix;
use crate::Result;

#[derive(Default)]
pub struct BasicMultiplier {
    last_time: Option<u64>,
}

impl Multiplier for BasicMultiplier {
    fn multiply(&mut self, m1: &Matrix, m2: &Matrix) -> Result<Matrix> {
        let instant = time::Instant::now();
        if m1.cols != m2.rows {
            return Err(io::Error::from(io::ErrorKind::InvalidData).into());
        }

        let mut res = Matrix::create_empty(m1.rows, m2.cols);

        for i in 0..m1.rows {
            for j in 0..m2.cols {
                let mut sum = 0f32;
                for k in 0..m1.cols {
                    sum += m1.get(i, k) * m2.get(k, j);
                }
                res.set(i, j, sum);
            }
        }

        let elapsed = instant.elapsed().as_millis() as u64;

        self.last_time = Some(elapsed);

        Ok(res)
    }

    fn info(&self) -> Result<MultiplierInfo> {
        Ok(MultiplierInfo::OnDeviceMultiplier)
    }

    fn stat(&self) -> Option<MultiplierStat> {
        self.last_time.map(|total_time| MultiplierStat {
            total_time,
            gpu_time: 0,
        })
    }
}
