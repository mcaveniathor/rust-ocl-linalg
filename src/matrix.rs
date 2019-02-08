use crate::util::Scalar;
use num::abs;
use std::fmt;
use std::fmt::Write;
use std::ops;

/// A generic matrix type with m rows and n columns. Data is stored in a flat(1-D) Vec
pub struct Matrix<T: Scalar> {
    pub m: usize,
    pub n: usize,
    pub data: Vec<T>,
}

impl<T: Scalar> Matrix<T> {
    /// Returns a new zero matrix with the provided dimensions.
    pub fn zero(m_in: usize, n_in: usize) -> Matrix<T> {
        Matrix {
            m: m_in,
            n: n_in,
            // All numeric types can be cloned from bool, so we convert false to zero
            data: vec![T::from(0); m_in * n_in],
        }
    }

    /// Returns a new identity matrix with the provided dimensions.
    pub fn identity(m_in: usize) -> Matrix<T> {
        let mut d = Vec::with_capacity(m_in * m_in);
        for r in 0..m_in {
            for c in 0..m_in {
                if r == c {
                    d.push(T::from(1));
                } else {
                    d.push(T::from(0));
                }
            }
        }
        Matrix {
            m: m_in,
            n: m_in,
            data: d,
        }
    }

    /// Returns a new diagonal matrix with the given values along the main diagonal
    pub fn diagonal(m_in: usize, diag: Vec<T>) -> Matrix<T> {
        assert_eq!(m_in, diag.len());
        let mut d: Vec<T> = Vec::with_capacity(m_in * m_in);
        let mut i = diag.iter();
        for r in 0..m_in {
            for c in 0..m_in {
                if r == c {
                    match i.next() {
                        Some(t) => d.push(*t),
                        None => panic!("Error: Expected {} entries, found {}", m_in, r - 1),
                    }
                } else {
                    d.push(T::from(0));
                }
            }
        }
        Matrix {
            m: m_in,
            n: m_in,
            data: d,
        }
    }

    /// Returns a new matrix with the given dimensions given a flattened(1 dimensional) input vector.
    pub fn new_from_flat_vec(m_in: usize, n_in: usize, data_in: Vec<T>) -> Matrix<T> {
        assert!(data_in.len() == m_in * n_in);
        Matrix {
            m: m_in,
            n: n_in,
            data: data_in,
        }
    }

    /// Returns a new matrix with the given dimensions from a 2d input vector
    pub fn new_from_2d_vec(m_in: usize, n_in: usize, data_in: Vec<Vec<T>>) -> Matrix<T> {
        assert!(data_in.len() == m_in && data_in[0].len() == n_in);
        let mut d: Vec<T> = Vec::with_capacity(m_in * n_in);
        for row in data_in {
            for col in row {
                d.push(col);
            }
        }
        Matrix {
            m: m_in,
            n: n_in,
            data: d,
        }
    }

    /// Returns the dimensions of the matrix as a tuple.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    /// Returns the specified row of the matrix as a 1xN matrix (row vector).
    pub fn row(&self, r: usize) -> Matrix<T> {
        assert!(r < self.m);
        Matrix {
            m: 1,
            n: self.n,
            data: (self.data[(r * self.n)..((r + 1) * self.n)]).to_vec(),
        }
    }

    /// Returns the specified column of the matrix as an M*1 matrix (column vector)
    pub fn col(&self, c: usize) -> Matrix<T> {
        assert!(c < self.n);
        let mut d: Vec<T> = Vec::with_capacity(self.m);
        for r in 0..self.m {
            d.push(self.data[r * self.n + c]);
        }
        Matrix {
            m: self.m,
            n: 1,
            data: d,
        }
    }

    /// Returns the value of the r-th row and i-th column of the matrix
    pub fn value_at(&self, r: usize, c: usize) -> T {
        assert!(r < self.m && c < self.n);
        self.data[r * self.n + c]
    }

    /// Set the value of a given entry in the matrix
    pub fn set_value_at(&mut self, r:usize, c: usize, val: T) {
        self.data[r*self.n + c] = val;
    }

    /// Returns the submatrix bounded by r_min, c_min, r_max, and c_max
    pub fn submatrix(&self, r_min: usize, c_min: usize, r_max: usize, c_max: usize) -> Matrix<T> {
        assert!(r_max < self.m && c_max < self.n && r_min <= r_max && c_min <= c_max);
        let mut d: Vec<T> = Vec::with_capacity((r_max - r_min) * (c_max - c_min));
        for r in r_min..r_max + 1 {
            for c in c_min..c_max + 1 {
                d.push(self.data[r * self.n + c]);
            }
        }
        Matrix {
            m: r_max - r_min + 1,
            n: c_max - c_min + 1,
            data: d,
        }
    }

    /// Recursively compute the determinant of the matrix.
    pub fn determinant(&self) -> T {
        assert!(self.is_square());
        if self.m == 2 {
            // the previous assertion guarantees that n is also 2, so no need to check
            let x = (self.value_at(0, 0) * self.value_at(1, 1))
                - (self.value_at(0, 1) * self.value_at(1, 0));
            x
        } else {
            let mut det: T = T::from(0);
            for i in 0..self.m {
                // iterate across columns in the top row
                let mut d: Vec<T> = Vec::with_capacity((self.m - 1) * (self.m - 1));
                for j in 1..self.m {
                    // remaining rows
                    for k in 0..self.m {
                        // columns in remaining rows
                        if k == i {
                            continue;
                        } else {
                            d.push(self.data[j * self.m + k]);
                        }
                    }
                }
                let mut mult = self.data[i];
                if i % 2 != 0 {
                    // odd rows only
                    mult = mult * T::from(-1);
                }
                det =
                    det + Matrix::new_from_flat_vec(self.m - 1, self.n - 1, d).determinant() * mult;
            }
            det
        }
    }

    /// Return whether or not the matrix is square (i.e. the number of rows is equal to the number of columns).
    pub fn is_square(&self) -> bool {
        self.m == self.n
    }

    /// Return whether or not the matrix is invertible.
    pub fn is_invertible(&self) -> bool {
        self.determinant() == T::from(0)
    }

    /// Swap the specified two rows in the matrix
    pub fn swap_rows(&mut self, r1: usize, r2: usize) {
        assert!(r1 < self.m && r2 < self.m);
        for i in 0 .. self.n {
            let tmp = self.value_at(r1, i);
            self.set_value_at(r1, i, self.value_at(r2,i));
            self.set_value_at(r2, i, tmp);
        }
    }

    /// converts the matrix to row-echelon form using Gaussian elimination
    pub fn row_echelon(&mut self) {
        for r in 0..self.m {
            for mut c in 0..self.n {
                //  pick the pivot with the highest absolute value to minimize rounding errors with floating point types
                let mut max_pivot: T = T::from(0);
                let mut max_row = 0;
                for r2 in r..self.m {
                    if abs(self.value_at(r2, c)) > max_pivot {
                        max_pivot = self.value_at(r2, c);
                        max_row = r2;
                    }
                }
                if max_pivot == T::from(0) {
                    c += 1; // no pivot found in this column
                }
                else {
                    self.swap_rows(r, max_row);
                    for i in r+1 ..self.m {
                        let f = self.value_at(i, c) / self.value_at(r, c); // this might introduce some issues with integer arithmetic. will have to sort that out
                        self.set_value_at(i, c, T::from(0));
                        for j in c + 1 .. self.n {
                            self.set_value_at(i, j, self.value_at(i, j) - self.value_at(r, j) * f);
                        }
                    }
                }
            }
        }
    }
}

impl<T: Scalar> fmt::Debug for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut output = String::new();
        writeln!(output, "Rows: {}, Columns: {}", self.m, self.n)?;
        for r in 0..self.m {
            write!(output, "|\t")?;
            for c in 0..self.n {
                write!(output, "{}\t", self.data[r * self.n + c])?;
            }
            writeln!(output, "|")?;
        }
        write!(f, "{}", output)
    }
}

impl<T: Scalar> fmt::Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut output = String::new();
        for r in 0..self.m {
            write!(output, "|\t")?;
            for c in 0..self.n {
                write!(output, "{}\t", self.data[r * self.n + c])?;
            }
            writeln!(output, "|")?;
        }
        write!(f, "{}", output)
    }
}

impl<T: Scalar> PartialEq for Matrix<T> {
    fn eq(&self, other: &Matrix<T>) -> bool {
        self.m == other.m && self.n == other.n && self.data == other.data
    }
}

/// define the use of the * operator for multiplication by a scalar
/// TODO: OpenCL
impl<T: Scalar> ops::Mul<T> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, rhs: T) -> Matrix<T> {
        //TODO: opencl kernel for this
        Matrix {
            m: self.m,
            n: self.n,
            data: self.data.iter().map(|x| *x * rhs).collect(),
        }
    }
}

/// Multiply by a scalar and assign the value with the *= operator
/// TODO: OpenCL
impl<T: Scalar> ops::MulAssign<T> for Matrix<T> {
    fn mul_assign(&mut self, rhs: T) {
        self.data = self.data.iter().map(|x| *x * rhs).collect();
    }
}

/// provide the ability to clone a matrix object
impl<T: Scalar> Clone for Matrix<T> {
    fn clone(&self) -> Matrix<T> {
        Matrix {
            m: self.m,
            n: self.n,
            data: self.data.clone(),
        }
    }
}

/// Add two matrices together with the + operator
/// TODO: OpenCL
impl<T: Scalar> ops::Add<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, rhs: Matrix<T>) -> Matrix<T> {
        assert_eq!(self.dimensions(), rhs.dimensions());
        let mut d: Vec<T> = Vec::with_capacity(self.m * self.n);
        for r in 0..self.m {
            for c in 0..self.n {
                d.push(self.data[r * self.n + c] + rhs.data[r * self.n + c]);
            }
        }
        Matrix {
            m: self.m,
            n: self.n,
            data: d,
        }
    }
}

/// Subtract two matrices using the - operator
/// TODO: OpenCL
impl<T: Scalar> ops::Sub<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, rhs: Matrix<T>) -> Matrix<T> {
        assert_eq!(self.dimensions(), rhs.dimensions());
        let mut d: Vec<T> = Vec::with_capacity(self.m * self.n);
        for r in 0..self.m {
            for c in 0..self.n {
                d.push(self.data[r * self.n + c] - rhs.data[r * self.n + c]);
            }
        }
        Matrix {
            m: self.m,
            n: self.n,
            data: d,
        }
    }
}
