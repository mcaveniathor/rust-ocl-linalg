use crate::util::Scalar;
use crate::vector::Vector;
use num::abs;
use std::fmt::{Debug, Display, Formatter, Result, Write};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

/// A generic matrix type with m rows and n columns. Data is stored in a flat(1-D) std::Vec
#[derive(PartialEq)]
pub struct Matrix<T: Scalar> {
    m: usize,
    n: usize,
    data: Vec<T>,
}

impl<T: Scalar> Matrix<T> {
    /// Returns a new zero matrix with the provided dimensions.
    pub fn zero(m: usize, n: usize) -> Matrix<T> {
        let data = vec![T::from(0); m * n];
        Matrix { m, n, data }
    }

    /// Returns a new matrix with the specified row and column removed. use a value of r_in > m to only
    /// remove a column or c_in > n to only remove a row
    pub fn minus_row_col(&self, r_in: usize, c_in: usize) -> Matrix<T> {
        let (mut m, mut n) = (self.m, self.n);
        let mut data = Vec::with_capacity(self.num_elements() - m - n + 1);
        let mut r = 0;
        while r < m {
            println!("{}", r);
            if r == r_in {
                r += 1;
            }
            for c in 0..n {
                if c == c_in {
                    continue;
                } else {
                    data.push(self.value_at(r, c));
                }
            }
            r += 1;
        }
        m = m - 1;
        n = n - 1;
        Matrix { m, n, data }
    }

    /// Returns a new identity matrix with the provided dimensions.
    pub fn identity(m: usize) -> Matrix<T> {
        let n = m;
        let mut data = Vec::with_capacity(m * n);
        for r in 0..m {
            for c in 0..n {
                if r == c {
                    data.push(T::from(1));
                } else {
                    data.push(T::from(0));
                }
            }
        }
        Matrix { m, n, data }
    }

    /// Returns a new diagonal square matrix with the given values along the main diagonal
    pub fn diagonal(m: usize, diag: Vec<T>) -> Matrix<T> {
        assert_eq!(m, diag.len());
        let n = m;
        let mut data: Vec<T> = Vec::with_capacity(m * n);
        let mut i = diag.iter();
        for r in 0..m {
            for c in 0..n {
                if r == c {
                    match i.next() {
                        Some(t) => data.push(*t),
                        None => panic!("Error: Expected {} entries, found {}", m, r - 1),
                    }
                } else {
                    data.push(T::from(0));
                }
            }
        }
        Matrix { m, n, data }
    }

    /// Returns a new matrix with the given dimensions given a flattened(1 dimensional) input vector.
    pub fn new_from_flat_vec(m: usize, n: usize, data: Vec<T>) -> Matrix<T> {
        assert!(data.len() == m * n);
        Matrix { m, n, data }
    }

    /// Returns a new matrix with the given dimensions from a 2d input vector
    pub fn new_from_2d_vec(m: usize, n: usize, data_in: Vec<Vec<T>>) -> Matrix<T> {
        assert!(data_in.len() == m && data_in[0].len() == n);
        let mut data: Vec<T> = Vec::with_capacity(m * n);
        for row in data_in {
            for col in row {
                data.push(col);
            }
        }
        Matrix { m, n, data }
    }

    /// Returns the dimensions of the matrix as a tuple.
    pub fn dimensions(&self) -> (usize, usize) {
        (self.m, self.n)
    }

    /// Returns the number of elements in the matrix.
    pub fn num_elements(&self) -> usize {
        self.m * self.n
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
    pub fn set_value_at(&mut self, r: usize, c: usize, val: T) {
        self.data[r * self.n + c] = val;
    }

    /// Returns the submatrix bounded by r_min, c_min, r_max, and c_max
    pub fn submatrix(&self, r_min: usize, c_min: usize, r_max: usize, c_max: usize) -> Matrix<T> {
        assert!(r_max < self.m && c_max < self.n && r_min <= r_max && c_min <= c_max);
        let mut data: Vec<T> = Vec::with_capacity((r_max - r_min) * (c_max - c_min));
        for r in r_min..r_max + 1 {
            for c in c_min..c_max + 1 {
                data.push(self.value_at(r, c));
            }
        }
        let (m, n) = (r_max - r_min + 1, c_max - c_min + 1);
        Matrix { m, n, data }
    }

    /// Recursively compute the determinant of the matrix.
    pub fn determinant(&self) -> T {
        assert!(self.is_square());
        let (m, n) = self.dimensions();
        if m == 2 {
            // the previous assertion guarantees that n is also 2, so no need to check
            let x = (self.value_at(0, 0) * self.value_at(1, 1))
                - (self.value_at(0, 1) * self.value_at(1, 0));
            x
        } else {
            let mut det: T = T::from(0);
            for i in 0..n {
                // iterate across columns in the top row
                let mut d: Vec<T> = Vec::with_capacity((m - 1) * (m - 1));
                for j in 1..m {
                    // remaining rows
                    for k in 0..n {
                        // columns in remaining rows
                        if k == i {
                            continue;
                        } else {
                            d.push(self.value_at(j, k));
                        }
                    }
                }
                let mut mult = self.value_at(0, i);
                if i % 2 != 0 {
                    // odd rows only
                    mult = mult * T::from(-1);
                }
                det = det + Matrix::new_from_flat_vec(m - 1, n - 1, d).determinant() * mult;
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
        let (m, n) = self.dimensions();
        assert!(r1 < m && r2 < m);
        for i in 0..n {
            let tmp = self.value_at(r1, i);
            self.set_value_at(r1, i, self.value_at(r2, i));
            self.set_value_at(r2, i, tmp);
        }
    }

    /// converts the matrix to row-echelon form using Gaussian elimination
    pub fn row_echelon(&mut self) {
        let mut c = 0;
        let (m, n) = self.dimensions();
        for r in 0..m {
            while c < n {
                //  pick the pivot with the highest absolute value to minimize rounding errors with floating point types
                let mut max_pivot: T = T::from(0);
                let mut pivot_exists = false;
                let mut max_row = 0;
                for r2 in r..m {
                    if abs(self.value_at(r2, c)) > max_pivot {
                        max_pivot = self.value_at(r2, c);
                        pivot_exists = true;
                        max_row = r2;
                    }
                }
                if !pivot_exists {
                    c += 1; // no pivot found in this column
                } else {
                    self.swap_rows(r, max_row);
                    for i in r + 1..m {
                        let f = self.value_at(i, c) / self.value_at(r, c); // this might introduce some issues with integer arithmetic. will have to sort that out
                        self.set_value_at(i, c, T::from(0));
                        for j in c + 1..n {
                            self.set_value_at(i, j, self.value_at(i, j) - self.value_at(r, j) * f);
                        }
                    }
                }
                c += 1;
            }
        }
    }

    /// creates a Matrix<T> using a Vector<T>
    pub fn from_vector(m: usize, n: usize, v_in: Vector<T>) -> Matrix<T> {
        let data = v_in.get_data();
        assert!(m == 1 && n == data.len() || n == 1 && m == data.len());
        Matrix { m, n, data }
    }
}

/// define the use of the * operator for multiplication by a scalar
/// TODO: OpenCL
impl<T: Scalar> Mul<T> for Matrix<T> {
    type Output = Matrix<T>;
    fn mul(self, rhs: T) -> Matrix<T> {
        let (m, n) = self.dimensions();
        let data = self.data.iter().map(|x| *x * rhs).collect();
        Matrix { m, n, data }
    }
}

/// Multiply by a scalar and assign the value with the *= operator
/// TODO: OpenCL
impl<T: Scalar> MulAssign<T> for Matrix<T> {
    fn mul_assign(&mut self, rhs: T) {
        let (m, n) = self.dimensions();
        for r in 0..m {
            for c in 0..n {
                self.set_value_at(r, c, self.value_at(r, c) * rhs);
            }
        }
    }
}

/// Add two matrices together with the + operator
/// TODO: OpenCL
impl<T: Scalar> Add<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn add(self, rhs: Matrix<T>) -> Matrix<T> {
        let (m, n) = self.dimensions();
        assert_eq!((m, n), rhs.dimensions());
        let mut data: Vec<T> = Vec::with_capacity(m * n);
        for r in 0..m {
            for c in 0..n {
                data.push(self.value_at(r, c) + rhs.value_at(r, c));
            }
        }
        Matrix { m, n, data }
    }
}

/// Add another identically-sized matrix to the caller in place
/// using the += operator
/// TODO: OpenCL
impl<T: Scalar> AddAssign<Matrix<T>> for Matrix<T> {
    fn add_assign(&mut self, rhs: Matrix<T>) {
        let (m, n) = self.dimensions();
        assert_eq!((m, n), rhs.dimensions());
        for r in 0..m {
            for c in 0..n {
                self.set_value_at(r, c, self.value_at(r, c) + rhs.value_at(r, c));
            }
        }
    }
}

/// Subtract two matrices using the - operator
/// TODO: OpenCL
impl<T: Scalar> Sub<Matrix<T>> for Matrix<T> {
    type Output = Matrix<T>;
    fn sub(self, rhs: Matrix<T>) -> Matrix<T> {
        let (m, n) = self.dimensions();
        assert_eq!((m, n), rhs.dimensions());
        let mut data: Vec<T> = Vec::with_capacity(m * n);
        for r in 0..m {
            for c in 0..n {
                data.push(self.value_at(r, c) - rhs.value_at(r, c))
            }
        }
        Matrix { m, n, data }
    }
}

///Subtract another identically-sized matrix from the caller in place
/// using the -= operator
impl<T: Scalar> SubAssign<Matrix<T>> for Matrix<T> {
    fn sub_assign(&mut self, rhs: Matrix<T>) {
        let (m, n) = self.dimensions();
        assert_eq!((m, n), rhs.dimensions());
        for r in 0..m {
            for c in 0..n {
                self.set_value_at(r, c, self.value_at(r, c) - rhs.value_at(r, c));
            }
        }
    }
}

/// Debug print the dimensions of the caller, its dimensions,
///  and its contents and their indices along with labelled rows and columns
impl<T: Scalar> Debug for Matrix<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let mut output = String::new();
        let (m, n) = self.dimensions();
        writeln!(output, "Rows: {}, Columns: {}", m, n)?;
        write!(output, "\t\t")?;
        for i in 0..n {
            write!(output, "{}\t", i)?;
        }
        writeln!(output, "\n")?;
        for r in 0..m {
            write!(output, "{}\t|\t", r)?;
            for c in 0..n {
                write!(output, "{}\t", self.value_at(r, c))?;
            }
            writeln!(output, "|")?;
        }
        write!(f, "{}", output)
    }
}

/// Print the elements of the caller.
impl<T: Scalar> Display for Matrix<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let (m, n) = self.dimensions();
        let mut output = String::new();
        for r in 0..m {
            write!(output, "|\t")?;
            for c in 0..n {
                write!(output, "{}\t", self.value_at(r, c))?;
            }
            writeln!(output, "|")?;
        }
        write!(f, "{}", output)
    }
}

/// provide the ability to clone a matrix object
impl<T: Scalar> Clone for Matrix<T> {
    fn clone(&self) -> Matrix<T> {
        let (m, n) = self.dimensions();
        let data = self.data.clone();
        Matrix { m, n, data }
    }
}

#[cfg(test)]
mod tests {
    use super::Matrix;
    /*

    MATRIX_3_2      ROW_VECTOR_1_2      COL_VECTOR_3_1
    0   1           0   1               0
    2   3                               2
    4   5                               4


    Note that testing is being done with integers; an approximate equality test would be needed for floating point values.

    */

    #[test]
    fn test_matrix_zero() {
        let z = mat!((3, 2), [0, 0, 0, 0, 0, 0]);
        assert_eq!(z, Matrix::zero(3, 2));
    }

    #[test]
    fn test_matrix_identity() {
        let m = mat!((3, 3), [1, 0, 0, 0, 1, 0, 0, 0, 1]);
        assert_eq!(Matrix::identity(3), m);
    }

    #[test]
    fn test_matrix_from_2d_vec() {
        assert_eq!(
            mat!((3, 2), [0, 1, 2, 3, 4, 5]),
            Matrix::new_from_2d_vec(3, 2, vec![vec![0, 1], vec![2, 3], vec![4, 5]])
        );
    }

    #[test]
    fn test_matrix_from_flat_vec() {
        assert_eq!(
            mat!((3, 2), [0, 1, 2, 3, 4, 5]),
            Matrix::new_from_flat_vec(3, 2, vec![0, 1, 2, 3, 4, 5])
        );
    }

    #[test]
    fn test_matrix_row() {
        assert_eq!(row_vec!([0, 1]), mat!((3, 2), [0, 1, 2, 3, 4, 5]).row(0));
    }

    #[test]
    fn test_matrix_col() {
        assert_eq!(col_vec!([0, 2, 4]), mat!((3, 2), [0, 1, 2, 3, 4, 5]).col(0));
    }

    #[test]
    fn test_matrix_value_at() {
        let (m, n) = mat!((3, 2), [0, 1, 2, 3, 4, 5]).dimensions();
        for r in 0..m {
            for c in 0..n {
                assert_eq!(
                    (r * n + c) as i32,
                    mat!((3, 2), [0, 1, 2, 3, 4, 5]).value_at(r, c)
                );
            }
        }
    }

    #[test]
    fn test_matrix_scalar_multiply() {
        let m = mat!((3, 2), [0, 2, 4, 6, 8, 10]);
        assert_eq!(mat!((3, 2), [0, 1, 2, 3, 4, 5]) * 2, m);
    }

    #[test]
    fn test_matrix_scalar_multiply_assign() {
        let mut m = mat!((3, 2), [0, 1, 2, 3, 4, 5]);
        m *= 2;
        assert_eq!(mat!((3, 2), [0, 2, 4, 6, 8, 10]), m);
    }

    #[test]
    fn test_matrix_submatrix() {
        let m = mat!((2, 2), [2, 3, 4, 5]);
        assert_eq!(mat!((3, 2), [0, 1, 2, 3, 4, 5]).submatrix(1, 0, 2, 1), m);
    }

    #[test]
    fn test_matrix_determinant() {
        let m = mat!((3, 3), [0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(0, m.determinant());
    }

    #[test]
    fn test_matrix_add() {
        let m = mat!((3, 2), [1, 1, 1, 1, 1, 1]);
        let m2 = mat!((3, 2), [1, 2, 3, 4, 5, 6]);
        assert_eq!(m2, mat!((3, 2), [0, 1, 2, 3, 4, 5]) + m);
    }

    #[test]
    fn test_matrix_subtract() {
        let m = mat!((3, 2), [1, 1, 1, 1, 1, 1]);
        let m2 = mat!((3, 2), [-1, 0, 1, 2, 3, 4]);
        assert_eq!(m2, mat!((3, 2), [0, 1, 2, 3, 4, 5]) - m);
    }

    #[test]
    fn test_matrix_diagonal() {
        let m = mat!((3, 3), [1, 0, 0, 0, 2, 0, 0, 0, 3]);
        assert_eq!(m, Matrix::diagonal(3, vec![1, 2, 3]));
    }

    #[test]
    fn test_matrix_set_value_at() {
        let mut m: Matrix<i32> = Matrix::zero(3, 3);
        m.set_value_at(2, 2, 1);
        assert_eq!(m.value_at(2, 2), 1);
    }

    #[test]
    fn test_matrix_swap_rows() {
        let mut m: Matrix<i32> = Matrix::identity(2);
        let m2 = mat!((2, 2), [0, 1, 1, 0]);
        m.swap_rows(0, 1);
        assert_eq!(m2, m);
    }

    // TODO
    #[test]
    fn test_matrix_row_echelon() {
        let mut m = mat!((2, 2), [1.0, 2.0, 3.0, 4.0]);
        m.row_echelon();
    }
}
