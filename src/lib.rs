#[allow(unused_imports)]
#[macro_use]
extern crate lazy_static;
extern crate num;
pub mod gpu;
pub mod matrix;
pub mod util;

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;
    lazy_static! {
        static ref MATRIX_3_2: Matrix<i32> = Matrix {
            m: 3,
            n: 2,
            data: vec![0, 1, 2, 3, 4, 5]
        };
        static ref ROW_VECTOR_1_2: Matrix<i32> = Matrix::new_from_flat_vec(1, 2, vec![0, 1]);
        static ref COL_VECTOR_3_1: Matrix<i32> = Matrix::new_from_flat_vec(3, 1, vec![0, 2, 4]);
    }

    /*

    MATRIX_3_2      ROW_VECTOR_1_2      COL_VECTOR_3_1
    0   1           0   1               0
    2   3                               2
    4   5                               4


    Note that testing is being done with integers; an approximate equality test would be needed for floating point values.

    */
    #[test]
    fn test_matrix_zero() {
        let z = Matrix::new_from_flat_vec(3, 2, vec![0, 0, 0, 0, 0, 0]);
        assert_eq!(z, Matrix::zero(3, 2));
    }

    #[test]
    fn test_matrix_identity() {
        let m = Matrix::new_from_flat_vec(3, 3, vec![1, 0, 0, 0, 1, 0, 0, 0, 1]);
        assert_eq!(Matrix::identity(3), m);
    }

    #[test]
    fn test_matrix_from_2d_vec() {
        assert_eq!(
            *MATRIX_3_2,
            Matrix::new_from_2d_vec(3, 2, vec![vec![0, 1], vec![2, 3], vec![4, 5]])
        );
    }

    #[test]
    fn test_matrix_from_flat_vec() {
        assert_eq!(
            *MATRIX_3_2,
            Matrix::new_from_flat_vec(3, 2, vec![0, 1, 2, 3, 4, 5])
        );
    }

    #[test]
    fn test_matrix_row() {
        assert_eq!(*ROW_VECTOR_1_2, (*MATRIX_3_2).row(0));
    }

    #[test]
    fn test_matrix_col() {
        assert_eq!(*COL_VECTOR_3_1, (*MATRIX_3_2).col(0));
    }

    #[test]
    fn test_matrix_value_at() {
        for r in 0..(*MATRIX_3_2).m {
            for c in 0..(*MATRIX_3_2).n {
                assert_eq!(
                    (r * (*MATRIX_3_2).n + c) as i32,
                    (*MATRIX_3_2).value_at(r, c)
                );
            }
        }
    }

    #[test]
    fn test_matrix_scalar_multiply() {
        let m = Matrix::new_from_flat_vec(3, 2, vec![0, 2, 4, 6, 8, 10]);
        assert_eq!(MATRIX_3_2.clone() * 2, m);
    }

    #[test]
    fn test_matrix_scalar_multiply_assign() {
        let mut m = MATRIX_3_2.clone();
        m *= 2;
        assert_eq!(Matrix::new_from_flat_vec(3, 2, vec![0, 2, 4, 6, 8, 10]), m);
    }

    #[test]
    fn test_matrix_submatrix() {
        let m = Matrix::new_from_flat_vec(2, 2, vec![2, 3, 4, 5]);
        assert_eq!((*MATRIX_3_2).submatrix(1, 0, 2, 1), m);
    }

    #[test]
    fn test_matrix_determinant() {
        let m = Matrix::new_from_flat_vec(3, 3, vec![0, 1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(0, m.determinant());
    }

    #[test]
    fn test_matrix_add() {
        let m = Matrix::new_from_flat_vec(3, 2, vec![1, 1, 1, 1, 1, 1]);
        let m2 = Matrix::new_from_flat_vec(3, 2, vec![1, 2, 3, 4, 5, 6]);
        assert_eq!(m2, MATRIX_3_2.clone() + m);
    }

    #[test]
    fn test_matrix_subtract() {
        let m = Matrix::new_from_flat_vec(3, 2, vec![1, 1, 1, 1, 1, 1]);
        let m2 = Matrix::new_from_flat_vec(3, 2, vec![-1, 0, 1, 2, 3, 4]);
        assert_eq!(m2, MATRIX_3_2.clone() - m);
    }

    #[test]
    fn test_matrix_diagonal() {
        let m = Matrix::new_from_flat_vec(3, 3, vec![1, 0, 0, 0, 2, 0, 0, 0, 3]);
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
        let m2: Matrix<i32> = Matrix::new_from_flat_vec(2, 2, vec![0, 1, 1, 0]);
        m.swap_rows(0, 1);
        assert_eq!(m2, m);
    }

    #[test]
    fn test_matrix_row_echelon() {
        let mut m: Matrix<f32> = Matrix::new_from_flat_vec(3, 3, vec![2.0, 4.0, 1.0, 6.0, 27.0, 0.0, 9.0, 12.0, 13.0]);
    }

}
