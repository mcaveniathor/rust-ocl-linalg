use crate::matrix::Matrix;
use crate::util::Scalar;
use std::fmt::{Debug, Display, Formatter, Result, Write};
use std::ops::{Add, AddAssign, Mul, MulAssign, Sub, SubAssign};

#[derive(PartialEq)]
pub struct Vector<T: Scalar> {
    data: Vec<T>,
}

impl<T: Scalar> Vector<T> {
    pub fn as_row_vector(&self) -> Matrix<T> {
        let (m, n) = (1, self.len());
        Matrix::new_from_flat_vec(m, n, self.get_data())
    }

    pub fn as_column_vector(&self) -> Matrix<T> {
        let (m, n) = (self.len(), 1);
        Matrix::new_from_flat_vec(m, n, self.get_data())
    }
    /// creates a Vector<T> object from a slice of T's
    pub fn new(data: Vec<T>) -> Vector<T> {
        Vector { data }
    }

    /// Calculates and returns the dot product (scalar product) of the vector with another.
    pub fn dot(&self, other: &Vector<T>) -> T {
        assert_eq!(self.len(), other.len());
        let mut sum = T::from(0);
        for i in 0..self.len() {
            sum = sum + self.value_at(i) * other.value_at(i);
        }
        sum
    }

    pub fn cross(&self, other: &Vector<T>) -> Vector<T> {
        assert!(self.len() == 3 && other.len() == 3);
        let mut data: Vec<T> = Vec::with_capacity(3);
        let mut tmp: Vec<T> = Vec::with_capacity(9);
        for _ in 0..3 {
            tmp.push(T::from(0));
        }
        for d in self.get_data() {
            tmp.push(d);
        }
        for d in other.get_data() {
            tmp.push(d);
        }
        let m = Matrix::new_from_flat_vec(3, 3, tmp);
        for i in 0..3 {
            let mut det = m.minus_row_col(0, i).determinant();
            if i == 1 {
                det = det * T::from(-1);
            }
            data.push(det);
        }
        Vector { data }
    }

    /// Returns the value of the i-th delement of the vector.
    pub fn value_at(&self, i: usize) -> T {
        if i < self.len() {
            self.data[i]
        } else {
            T::from(0)
        }
    }

    /// Set the value of the i-th element of the vector. If i is greater than the number of elements in the vector,
    /// reserve additional space and fill the elements in between with zeroes
    pub fn set_value_at(&mut self, i: usize, t_in: T) {
        if i > self.len() - 1 {
            if self.data.capacity() < i {
                self.reserve(i - self.data.capacity());
            }
            for _ in self.len()..i + 1 {
                self.data.push(T::from(0));
            }
        }
        self.data[i] = t_in;
    }

    /// Returns the length of the vector.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Pushes i elements with value 0 onto the end of the vector.
    pub fn push_zeros(&mut self, i: usize) {
        if self.data.capacity() + i < i {
            self.data.reserve(i);
        }
        for _ in 0..i {
            self.data.push(T::from(0));
        }
    }

    /// Allocates space for i additional elements and fills them with zeroes
    pub fn reserve(&mut self, i: usize) {
        assert!(i > 0);
        self.data.reserve(i);
        self.push_zeros(i);
    }

    pub fn get_data(&self) -> Vec<T> {
        self.data.clone()
    }
}

/// Defines the use of the * operator to multiply a copy of the vector by a scalar quantity
impl<T: Scalar> Mul<T> for Vector<T> {
    type Output = Vector<T>;
    fn mul(self, rhs: T) -> Vector<T> {
        let data = self.data.iter().map(|x| *x * rhs).collect();
        Vector { data }
    }
}

/// Multiply the vector by a scalar in place with the *= operator
impl<T: Scalar> MulAssign<T> for Vector<T> {
    fn mul_assign(&mut self, rhs: T) {
        for i in 0..self.len() {
            self.set_value_at(i, self.value_at(i) * rhs);
        }
    }
}

/// Add two vectors together elementwise and return the result using the + operator.
impl<T: Scalar> Add<Vector<T>> for Vector<T> {
    type Output = Vector<T>;
    fn add(self, rhs: Vector<T>) -> Vector<T> {
        let mut longer_len = self.len();
        if rhs.len() > longer_len {
            longer_len = rhs.len();
        }
        let mut data: Vec<T> = Vec::with_capacity(longer_len);
        for i in 0..longer_len {
            data.push(self.value_at(i) + rhs.value_at(i));
        }
        Vector { data }
    }
}

/// Add another vector to the caller in place with the += operator.
impl<T: Scalar> AddAssign<Vector<T>> for Vector<T> {
    fn add_assign(&mut self, rhs: Vector<T>) {
        let mut longer_len = self.len();
        if rhs.len() > longer_len {
            longer_len = rhs.len();
            if self.data.capacity() < longer_len {
                self.reserve(longer_len - self.data.capacity());
            }
        }
        for i in 0..longer_len {
            self.set_value_at(i, self.value_at(i) + rhs.value_at(i));
        }
    }
}

/// Add two vectors together elementwise and return the result using the - operator.
impl<T: Scalar> Sub<Vector<T>> for Vector<T> {
    type Output = Vector<T>;
    fn sub(self, rhs: Vector<T>) -> Vector<T> {
        let mut longer_len = self.len();
        if rhs.len() > longer_len {
            longer_len = rhs.len();
        }
        let mut data: Vec<T> = Vec::with_capacity(longer_len);
        for i in 0..longer_len {
            data.push(self.value_at(i) - rhs.value_at(i));
        }
        Vector { data }
    }
}

/// Subtract another vector from the caller in place with the -= operator.
impl<T: Scalar> SubAssign<Vector<T>> for Vector<T> {
    fn sub_assign(&mut self, rhs: Vector<T>) {
        let mut longer_len = self.len();
        if rhs.len() > longer_len {
            longer_len = rhs.len();
            if self.data.capacity() < longer_len {
                self.reserve(longer_len - self.data.capacity());
            }
        }
        for i in 0..longer_len {
            self.set_value_at(i, self.value_at(i) - rhs.value_at(i));
        }
    }
}

/// Prints the vector's length and all of its elements and their indices
impl<T: Scalar> Debug for Vector<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let mut output = String::new();
        writeln!(output, "Length: {}", self.len())?;
        write!(output, "< ")?;
        for i in 0..self.len() {
            write!(output, "{}: {}", i, self.data[i])?;
            if i != self.len() - 1 {
                write!(output, ", ")?;
            } else {
                write!(output, " >")?;
            }
        }
        write!(f, "{}", output)
    }
}

/// Prints just the vector itself
impl<T: Scalar> Display for Vector<T> {
    fn fmt(&self, f: &mut Formatter) -> Result {
        let mut output = String::new();
        write!(output, "< ")?;
        for i in 0..self.len() {
            write!(output, "{}", self.data[i])?;
            if i != self.len() - 1 {
                write!(output, ", ")?;
            } else {
                write!(output, " >")?;
            }
        }
        write!(f, "{}", output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_new() {
        let x = Vector::new(vec![1, 2, 3]);
        assert_eq!(v![1, 2, 3], x);
    }

    #[test]
    fn test_vector_mul() {
        assert_eq!(v![0, 1, 2].mul(2), v![0, 2, 4]);
    }

    #[test]
    fn test_vector_value_at() {
        assert_eq!(v![0, 1, 2].value_at(2), 2);
    }

    #[test]
    fn test_vector_set_value_at() {
        let mut ve = v![0, 1, 2];
        ve.set_value_at(1, 5);
        assert_eq!(ve, v![0, 5, 2]);
        ve.set_value_at(5, 3);
        assert_eq!(ve, v![0, 5, 2, 0, 0, 3]);
    }

    #[test]
    fn test_vector_len() {
        assert_eq!(v![0, 1, 2].len(), 3);
    }

    #[test]
    fn test_vector_reserve() {
        let mut ve = v![0, 1, 2];
        ve.reserve(3);
        assert_eq!(ve.value_at(4), 0);
    }

    #[test]
    fn test_vector_mul_op() {
        assert_eq!(v![0, 1, 2] * 3, v![0, 3, 6]);
    }

    #[test]
    fn test_vector_mul_assign_op() {
        let mut ve = v![0, 1, 2];
        ve *= 2;
        assert_eq!(ve, v![0, 2, 4]);
    }

    #[test]
    fn test_vector_add_op() {
        assert_eq!(v![0, 1, 2] + v![1, 1, 1], v![1, 2, 3]);
    }

    #[test]
    fn test_vector_add_assign_op() {
        let mut v1 = v![0, 1, 2];
        let v2 = v![1, 2, 3, 4];
        v1 += v2;
        assert_eq!(v1, v![1, 3, 5, 4]);
    }

    #[test]
    fn test_vector_sub() {
        assert_eq!((v![0, 1, 2] - v![1, 1, 1]), v![-1, 0, 1]);
    }

    #[test]
    fn test_vector_sub_assign_op() {
        let mut ve = v![0, 1, 2];
        let v2 = v![5, 5];
        ve -= v2;
        assert_eq!(ve, v![-5, -4, 2]);
    }

    #[test]
    fn test_vector_dot() {
        let m = v![1, 2, 3];
        let n = v![4, 5, 6];
        assert_eq!(32, m.dot(&n));
    }

}
