use std::fs;
use std::io::{self, BufRead};
use std::path::Path;

use super::Matrix;

use super::Result;

/// Parses a file for two matrices
pub fn parse_file(path: &Path) -> Result<(Matrix, Matrix)> {
    let file = fs::File::open(path)?;
    let mut reader = io::BufReader::new(file);
    let mut buf = String::new();

    let _ = reader.read_line(&mut buf)?;
    let dims = buf
        .split(' ')
        .map(|str| str.trim().parse::<usize>())
        .collect::<std::result::Result<Vec<_>, _>>()?;

    if dims.len() != 3 {
        return Err(io::Error::from(io::ErrorKind::InvalidData).into());
    }

    let n = *dims
        .first()
        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))?;
    let m = *dims
        .get(1)
        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))?;
    let k = *dims
        .get(2)
        .ok_or_else(|| io::Error::from(io::ErrorKind::InvalidData))?;

    let mut data1 = Vec::with_capacity(n * m);
    let mut data2 = Vec::with_capacity(m * k);

    buf.clear();

    for _ in 0..n {
        buf.clear();
        let _ = reader.read_line(&mut buf);
        let mut nums = buf
            .split(' ')
            .map(|str| str.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()?;

        if nums.len() != m {
            return Err(io::Error::from(io::ErrorKind::InvalidData).into());
        }

        data1.append(&mut nums)
    }

    for _ in 0..m {
        buf.clear();
        let _ = reader.read_line(&mut buf);
        let mut nums = buf
            .split(' ')
            .map(|str| str.trim().parse::<f32>())
            .collect::<std::result::Result<Vec<_>, _>>()?;

        if nums.len() != k {
            return Err(io::Error::from(io::ErrorKind::InvalidData).into());
        }

        data2.append(&mut nums);
    }

    // assert no lines are left in the input file
    buf.clear();
    let sz = reader.read_line(&mut buf)?;

    if sz != 0 {
        return Err(io::Error::from(io::ErrorKind::InvalidData).into());
    }

    let m1 = Matrix::create(n, m, &data1)?;
    let m2 = Matrix::create(m, k, &data2)?;

    Ok((m1, m2))
}
