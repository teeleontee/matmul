use std::fmt::{Debug, Display};

use super::Result;

pub struct Matrix {
    /// Count of rows of the Matrix
    pub rows: usize,
    /// Count of columns of the Matrix
    pub cols: usize,
    /// Actual matrix data in the form of a vec
    pub data: Vec<f32>,
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }

        self.data.iter().zip(other.data.iter()).all(|(a, b)| {
            let res = a - b;
            res.abs() < 0.01
        })
    }
}

impl Matrix {
    /// Create a Matrix given the rows and cols and data of a matrix
    ///
    /// May fail if provided bad arguments, as in rows * cols != data.len()
    pub fn create(rows: usize, cols: usize, data: &[f32]) -> Result<Self> {
        if rows * cols != data.len() {
            let err_msg = format!(
                "InvalidData, {} * {} != {} (data size)",
                rows,
                cols,
                data.len()
            );
            return Err(err_msg.into());
        }

        let res = Self {
            rows,
            cols,
            data: data.to_vec(),
        };

        Ok(res)
    }

    /// Creates an zeroed out Matrix of given size
    pub fn create_empty(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0f32; rows * cols],
        }
    }

    /// Return an iterator to the matrix data
    ///
    /// Is used in tests
    #[allow(unused)]
    pub fn iter(&self) -> std::slice::Iter<'_, f32> {
        self.data.iter()
    }

    /// Return a mutable iterator to the matrix data
    #[allow(unused)]
    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, f32> {
        self.data.iter_mut()
    }

    /// Get the element at row `row` and col `col` of the matrix
    ///
    /// Can panic if given bad arguments (index out of bounds)
    #[inline]
    pub fn get(&self, row: usize, col: usize) -> f32 {
        self.data[row * self.cols + col]
    }

    /// Set the value of row `row` and col `col` to a provided value
    ///
    /// Can panic if given bad arguments (index out of bounds)
    #[inline]
    pub fn set(&mut self, row: usize, cols: usize, new: f32) {
        self.data[row * self.cols + cols] = new;
    }

    /// Creates a Matrix from self that is padded out with zeroes so that the new dimensions are
    /// divisible by `tile`
    ///
    /// This is an optimization for various implementations
    pub fn create_zero_padded(&self, tile: usize) -> Matrix {
        if self.rows % tile == 0 && self.cols == 0 {
            return Matrix {
                rows: self.rows,
                cols: self.cols,
                data: self.data.clone(),
            };
        }

        let mod_rows = self.rows % tile;
        let mod_cols = self.cols % tile;
        let new_rows = self.rows - mod_rows + tile;
        let new_cols = self.cols - mod_cols + tile;

        let mut res = Matrix::create_empty(new_rows, new_cols);

        for i in 0..self.rows {
            for j in 0..self.cols {
                res.set(i, j, self.get(i, j));
            }
        }

        res
    }

    /// Returns a new trimmed matrix from self provided a new row and column size
    pub fn create_trimmed(&self, rows: usize, cols: usize) -> Matrix {
        let mut res = Matrix::create_empty(rows, cols);

        for i in 0..rows {
            for j in 0..cols {
                let value = self.get(i, j);
                res.set(i, j, value);
            }
        }

        res
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(&self, f)
    }
}

impl Debug for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for i in 0..self.rows {
            for j in 0..self.cols {
                let value = self.get(i, j);
                write!(f, "{} ", value)?;
            }
            writeln!(f)?;
        }

        Ok(())
    }
}
